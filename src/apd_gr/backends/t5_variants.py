from __future__ import annotations

import torch
from transformers import T5Config, T5ForConditionalGeneration

from .base import select_context
from .t5 import T5Backend
from apd_gr.integrations.loader import load_diger_module, load_integration_module


class CoFiRecBackend(T5Backend):
    """T5 backend with CoFiRec group and within-group position embeddings."""

    def load_model(self, tokenizer, device, dtype):
        module = load_integration_module(
            self.args.repo_root, "modeling_cofirec.py", "cofirec"
        )
        model = module.CoFiRec.from_pretrained(
            self.target_ckpt,
            config=T5Config.from_pretrained(self.target_ckpt, local_files_only=True),
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        return self.freeze(model, device, dtype)

    def encoder_embeddings(self, model, input_ids, attention_mask):
        embeds = model.shared(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        if input_ids.shape[1] > model.group_embedding.num_embeddings * int(model.group_size):
            raise ValueError("Input exceeds CoFiRec positional capacity")
        group_ids = positions // int(model.group_size)
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        mask = mask * (input_ids != model.config.eos_token_id).unsqueeze(-1)
        pos = model.group_embedding(group_ids) + model.pos_embedding(
            positions % int(model.group_size)
        )
        return embeds + pos * mask


class DigerBackend(T5Backend):
    """DIGER's position-specific embedding and wrapped T5 execution."""

    def load_model(self, tokenizer, device, dtype):
        data_module = load_diger_module(self.args.repo_root, "data.py", "diger_data")
        _, num_items, _, _, _ = data_module.load_split_data(self.diger_config)
        model_module = load_diger_module(self.args.repo_root, "model.py", "diger_model")
        model = model_module.Model(
            config=self.diger_config,
            model=T5ForConditionalGeneration(config=self.config),
            n_items=num_items,
            code_length=self.diger_config["code_length"],
            code_number=self.diger_config["code_num"],
        )
        try:
            state = torch.load(self.target_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.target_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.device = device
        return self.freeze(model, device, dtype)

    def encoder_embeddings(self, model, input_ids, attention_mask):
        safe_ids = input_ids.clone()
        safe_ids[safe_ids < 0] = 0
        hidden_size = int(model.hidden_size)
        dtype = model.token_embeddings[0].weight.dtype
        embeds = torch.zeros(
            (*safe_ids.shape, hidden_size), device=safe_ids.device, dtype=dtype
        )
        for pos in range(int(model.code_length)):
            embeds[:, pos :: int(model.code_length)] = model.token_embeddings[pos](
                safe_ids[:, pos :: int(model.code_length)]
            )
        pad_embed = model.model.shared.weight[0].to(
            device=embeds.device, dtype=dtype
        ).view(1, 1, hidden_size)
        return torch.where(attention_mask.unsqueeze(-1), embeds, pad_embed)

    def _encode(self, model, input_ids, attention_mask, layer_ids):
        embeds = self.encoder_embeddings(model, input_ids, attention_mask)
        out = model.model.encoder(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out, select_context(out.hidden_states, layer_ids)

    def _root(self, model, encoder_hidden, attention_mask):
        start = torch.zeros(
            (encoder_hidden.shape[0], 1), dtype=torch.long, device=encoder_hidden.device
        )
        out = model.model.decoder(
            inputs_embeds=model.model.shared(start),
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state[:, 0]

    @torch.no_grad()
    def extract_cache(self, model, batch, layer_ids, position_token_ids, teacher_cache):
        encoded, context = self._encode(
            model, batch["input_ids"], batch["attention_mask"], layer_ids
        )
        root = self._root(model, encoded.last_hidden_state, batch["attention_mask"])
        teacher = None
        if teacher_cache == "full_legal":
            out = model(
                encoder_outputs=(encoded.last_hidden_state,),
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            teacher = out.logits.float()
        return context, root, teacher


class ReSIDBackend(T5Backend):
    """Standard T5 execution loaded from a raw ReSID state dict."""

    def load_model(self, tokenizer, device, dtype):
        model = T5ForConditionalGeneration(self.config)
        try:
            state = torch.load(self.target_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.target_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        return self.freeze(model, device, dtype)
