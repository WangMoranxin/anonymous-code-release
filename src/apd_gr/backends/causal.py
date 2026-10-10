from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from .base import TargetBackend, select_context


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _tie_peft_input_output_embeddings(model):
    """Preserve tied embeddings after PEFT wraps saved modules."""
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("Model does not expose both input and output embeddings")
    if hasattr(input_embeddings, "original_module") and hasattr(output_embeddings, "original_module"):
        output_embeddings.original_module.weight = input_embeddings.original_module.weight
    input_saved = getattr(input_embeddings, "modules_to_save", None)
    output_saved = getattr(output_embeddings, "modules_to_save", None)
    if input_saved is not None and output_saved is not None:
        for adapter_name in sorted(set(input_saved.keys()) & set(output_saved.keys())):
            output_saved[adapter_name].weight = input_saved[adapter_name].weight
    else:
        output_embeddings.weight = input_embeddings.weight
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = True


def _last_valid_indices(attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask.long() * positions.view(1, -1)).max(dim=1).values


class CausalBackend(TargetBackend):
    """Standard Hugging Face causal-LM execution for Llama, Qwen, and PEFT."""

    def __init__(self, args):
        adapter_path = os.path.join(os.path.abspath(args.target_ckpt), "adapter_config.json")
        self.peft_config = _read_json(adapter_path) if os.path.isfile(adapter_path) else None
        self.args = args
        self.target_ckpt = os.path.abspath(args.target_ckpt)
        source = args.base_model or (
            self.peft_config["base_model_name_or_path"] if self.peft_config else self.target_ckpt
        )
        self.config = AutoConfig.from_pretrained(source, local_files_only=True)

    def load_tokenizer(self, padding_side="right"):
        source = self.args.tokenizer_path or self.target_ckpt
        if self.peft_config and not os.path.exists(os.path.join(source, "tokenizer_config.json")):
            source = self.args.base_model or self.peft_config["base_model_name_or_path"]
        tokenizer_kwargs = dict(
            model_max_length=self.args.model_max_length,
            padding_side=padding_side,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **tokenizer_kwargs)
        # Some full-model checkpoints save the generic ``PreTrainedTokenizer``
        # class even though only tokenizer.json is usable. Transformers 4.45
        # then returns False for the slow-tokenizer request instead of raising.
        # Keep slow tokenizers for existing targets, but recover with the fast
        # tokenizer used to train checkpoints such as baseline/parallel_draft-beam.
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **tokenizer_kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def load_model(self, tokenizer, device, dtype):
        source = self.args.base_model or (
            self.peft_config["base_model_name_or_path"] if self.peft_config else self.target_ckpt
        )
        model = AutoModelForCausalLM.from_pretrained(
            source, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=dtype
        )
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        if self.peft_config:
            from peft import PeftModel

            # PEFT otherwise infers CUDA and asks safetensors to load this large adapter
            # directly on-device; load alongside the CPU base model, then freeze() moves both.
            model = PeftModel.from_pretrained(
                model, self.target_ckpt, is_trainable=False, torch_device="cpu"
            )
            _tie_peft_input_output_embeddings(model)
        return self.freeze(model, device, dtype)

    def _run(self, model, input_ids, attention_mask):
        # Avoid materializing full-vocabulary logits for every prompt position.
        causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        backbone = getattr(causal_lm, "model", None)
        if backbone is None:
            raise TypeError(f"Causal target has no decoder backbone: {type(causal_lm).__name__}")
        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def extract_cache(self, model, batch, layer_ids, position_token_ids, teacher_cache):
        prompt = self._run(model, batch["input_ids"], batch["attention_mask"])
        context = select_context(prompt.hidden_states, layer_ids)
        last = _last_valid_indices(batch["attention_mask"])
        root = prompt.hidden_states[-1][torch.arange(last.shape[0], device=last.device), last]
        teacher = None
        if teacher_cache == "full_legal":
            full = self._run(model, batch["full_input_ids"], batch["full_attention_mask"])
            gather = batch["teacher_positions"].unsqueeze(-1).expand(
                -1, -1, full.hidden_states[-1].shape[-1]
            )
            hidden = full.hidden_states[-1].gather(1, gather)
            head = model.get_output_embeddings()
            if hasattr(head, "modules_to_save"):
                head = head.modules_to_save[head.active_adapter]
            max_legal = max(map(len, position_token_ids))
            teacher = torch.zeros(
                (hidden.shape[0], self.args.item_token_num, max_legal),
                dtype=torch.float32,
                device=hidden.device,
            )
            for pos, legal in enumerate(position_token_ids):
                ids = torch.tensor(legal, device=head.weight.device)
                weight = head.weight.index_select(0, ids)
                teacher[:, pos, : len(legal)] = F.linear(
                    hidden[:, pos].to(weight.dtype), weight
                ).float()
        return context, root, teacher

    @torch.no_grad()
    def prompt_features(self, model, input_ids, attention_mask, layer_ids):
        out = self._run(model, input_ids, attention_mask)
        last = _last_valid_indices(attention_mask)
        root = out.hidden_states[-1][torch.arange(last.shape[0], device=last.device), last]
        return select_context(out.hidden_states, layer_ids), root
