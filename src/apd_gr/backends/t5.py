from __future__ import annotations

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .base import TargetBackend, select_context


class T5Backend(TargetBackend):
    """Standard T5 encoder/decoder execution shared by T5-based integrations."""

    is_encoder_decoder = True

    def load_tokenizer(self, padding_side="right"):
        return T5Tokenizer.from_pretrained(
            self.target_ckpt,
            model_max_length=self.args.model_max_length,
            padding_side=padding_side,
            local_files_only=True,
        )

    def load_model(self, tokenizer, device, dtype):
        model = T5ForConditionalGeneration.from_pretrained(
            self.target_ckpt,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        return self.freeze(model, device, dtype)

    def encoder_embeddings(self, model, input_ids, attention_mask):
        return model.shared(input_ids)

    def _encode(self, model, input_ids, attention_mask, layer_ids):
        out = model.encoder(
            inputs_embeds=self.encoder_embeddings(model, input_ids, attention_mask),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out, select_context(out.hidden_states, layer_ids)

    def _root(self, model, encoder_hidden, attention_mask):
        start = torch.full(
            (encoder_hidden.shape[0], 1),
            int(model.config.decoder_start_token_id),
            dtype=torch.long,
            device=encoder_hidden.device,
        )
        out = model.decoder(
            input_ids=start,
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
                use_cache=False,
                return_dict=True,
            )
            max_legal = max(map(len, position_token_ids))
            teacher = torch.zeros(
                (batch["input_ids"].shape[0], self.args.item_token_num, max_legal),
                dtype=torch.float32,
                device=batch["input_ids"].device,
            )
            for pos, legal in enumerate(position_token_ids):
                ids = torch.tensor(legal, device=teacher.device)
                teacher[:, pos, : len(legal)] = out.logits[:, pos].index_select(-1, ids).float()
        return context, root, teacher

    @torch.no_grad()
    def prompt_features(self, model, input_ids, attention_mask, layer_ids):
        encoded, context = self._encode(model, input_ids, attention_mask, layer_ids)
        return context, self._root(model, encoded.last_hidden_state, attention_mask)
