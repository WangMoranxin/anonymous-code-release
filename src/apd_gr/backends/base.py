from __future__ import annotations

import os

import torch
from transformers import AutoConfig, AutoTokenizer


def select_context(hidden_states, layer_ids):
    selected = []
    for layer_id in layer_ids:
        index = int(layer_id) + 1
        if not 0 < index < len(hidden_states):
            raise ValueError(
                f"Layer {layer_id} is unavailable; model exposes {len(hidden_states) - 1} layers"
            )
        selected.append(hidden_states[index])
    return torch.cat(selected, dim=-1)


class TargetBackend:
    """Model-execution contract; dataset and SID semantics belong to integrations."""

    kind = ""
    is_encoder_decoder = False
    uses_integer_codes = False

    def __init__(self, args):
        self.args = args
        self.target_ckpt = os.path.abspath(args.target_ckpt)
        self.config = AutoConfig.from_pretrained(self.target_ckpt, local_files_only=True)

    @property
    def hidden_size(self):
        return int(getattr(self.config, "d_model", getattr(self.config, "hidden_size", 0)))

    @property
    def num_layers(self):
        return int(
            getattr(self.config, "num_layers", getattr(self.config, "num_hidden_layers", 0))
        )

    @property
    def required_item_token_num(self):
        return None

    def load_tokenizer(self, padding_side="right"):
        tokenizer = AutoTokenizer.from_pretrained(
            self.target_ckpt,
            model_max_length=self.args.model_max_length,
            padding_side=padding_side,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, tokenizer, device, dtype):
        """Load and freeze the target model. Implemented by concrete backends."""
        raise NotImplementedError

    def extract_cache(self, model, batch, layer_ids, position_token_ids, teacher_cache):
        """Return context, root state, and optional legal-token teacher logits."""
        raise NotImplementedError

    def prompt_features(self, model, input_ids, attention_mask, layer_ids):
        """Return context and the target root state for decoding."""
        raise NotImplementedError

    def freeze(self, model, device, dtype):
        model = model.to(device=device, dtype=dtype).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
