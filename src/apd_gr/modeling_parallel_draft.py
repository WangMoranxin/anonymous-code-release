from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


@dataclass
class ParallelDraftOutput(ModelOutput):
    logits: Optional[List[torch.Tensor]] = None
    target_ids: Optional[torch.Tensor] = None
    target_local_ids: Optional[torch.Tensor] = None
    eval_mask: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos[:, :, -q.shape[-2] :, :]) + (
        _rotate_half(q) * sin[:, :, -q.shape[-2] :, :]
    )
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _get_position_embeddings(rotary_emb, hidden_states, position_ids):
    # Match online and preserve model-specific RoPE scaling (e.g. Llama 3).
    return rotary_emb(hidden_states, position_ids)


class ParallelDraftAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class ParallelDraftDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = ParallelDraftAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings,
        past_key_values: Optional[object] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class ParallelDraftHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, position_token_ids):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.position_token_ids = [list(map(int, ids)) for ids in position_token_ids]
        self.item_token_num = len(self.position_token_ids)
        self.token_emb_list = nn.ModuleList(
            [nn.Embedding(len(ids), self.hidden_size) for ids in self.position_token_ids]
        )
        self.transition = nn.ModuleList(
            [nn.RNNCell(self.hidden_size, self.hidden_size) for _ in self.position_token_ids]
        )
        self.logit_head_list = nn.ModuleList(
            [nn.Linear(self.hidden_size, len(ids), bias=False) for ids in self.position_token_ids]
        )
        for pos, token_ids in enumerate(self.position_token_ids):
            ids = torch.tensor(token_ids, dtype=torch.long)
            mapping = torch.full((self.vocab_size,), -1, dtype=torch.long)
            mapping[ids] = torch.arange(len(token_ids), dtype=torch.long)
            self.register_buffer(f"position_ids_{pos}", ids, persistent=True)
            self.register_buffer(f"global_to_local_{pos}", mapping, persistent=False)

    def local_ids(self, pos: int, global_token_ids: torch.Tensor) -> torch.Tensor:
        mapping = getattr(self, f"global_to_local_{int(pos)}")
        local = mapping.to(global_token_ids.device)[global_token_ids.long()]
        if bool((local < 0).any().item()):
            bad = global_token_ids[local < 0][:8].detach().cpu().tolist()
            raise ValueError(f"Token ids {bad} are illegal for item position {pos}")
        return local

    def global_ids(self, pos: int) -> torch.Tensor:
        return getattr(self, f"position_ids_{int(pos)}")

    def logits_for_position(
        self, *, state: torch.Tensor, hidden_states: torch.Tensor, pos: int
    ) -> torch.Tensor:
        return self.logit_head_list[int(pos)](state + hidden_states)

    def update_state(
        self, *, state: torch.Tensor, pos: int, token_ids: torch.Tensor
    ) -> torch.Tensor:
        local_ids = self.local_ids(pos, token_ids)
        token_emb = self.token_emb_list[int(pos)](local_ids).to(state.dtype)
        return self.transition[int(pos)](token_emb, state)

    def forward(
        self,
        *,
        draft_hidden: torch.Tensor,
        root_hidden: torch.Tensor,
        target_ids: torch.Tensor,
    ):
        item_token_num = draft_hidden.shape[1]
        if item_token_num <= 0:
            return [], target_ids.new_empty(target_ids.shape)
        state = root_hidden
        logits = []
        target_local_ids = []
        for pos in range(item_token_num):
            logits.append(
                self.logits_for_position(
                    state=state,
                    hidden_states=draft_hidden[:, pos, :],
                    pos=pos,
                )
            )
            target_local_ids.append(self.local_ids(pos, target_ids[:, pos]))
            if pos + 1 < item_token_num:
                state = self.update_state(
                    state=state,
                    pos=pos,
                    token_ids=target_ids[:, pos],
                )
        return logits, torch.stack(target_local_ids, dim=1)


class ParallelDraftModel(LlamaPreTrainedModel):
    _no_split_modules = ["ParallelDraftDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.hidden_size = int(config.hidden_size)
        self.target_hidden_size = int(getattr(config, "target_hidden_size", self.hidden_size))
        self.raw_context_dim = len(self.target_layer_ids) * self.target_hidden_size
        self.item_token_num = int(config.item_token_num)
        self.proposal_len = self.item_token_num
        self.mask_token_id = int(config.mask_token_id)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                ParallelDraftDecoderLayer(config, idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * self.target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.draft_head = ParallelDraftHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            position_token_ids=config.position_token_ids,
        )
        self.post_init()

    def _build_attention_mask(
        self,
        *,
        context_len: int,
        position_ids: torch.Tensor,
        context_limit_positions: torch.Tensor,
        context_attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, q_len = position_ids.shape
        device = position_ids.device
        ctx_positions = torch.arange(context_len, device=device).view(1, 1, context_len)
        context_limit_positions = context_limit_positions.to(device=device, dtype=torch.long)
        if context_limit_positions.ndim == 0:
            context_limit_positions = context_limit_positions.view(1).expand(bsz)
        if context_limit_positions.shape[0] != bsz:
            raise ValueError(
                "context_limit_positions must be scalar or have batch size "
                f"{bsz}, got {tuple(context_limit_positions.shape)}"
            )
        allow_context = ctx_positions <= context_limit_positions.view(bsz, 1, 1)
        if context_attention_mask is not None:
            allow_context = allow_context & context_attention_mask.to(
                device=device, dtype=torch.bool
            ).view(bsz, 1, context_len)
        draft_positions = torch.arange(q_len, device=device).view(1, 1, q_len)
        q_idx = torch.arange(q_len, device=device).view(1, q_len, 1)
        allow_draft = draft_positions <= q_idx
        allow = torch.cat(
            [
                allow_context.expand(bsz, q_len, context_len),
                allow_draft.expand(bsz, q_len, q_len),
            ],
            dim=-1,
        )
        mask = torch.zeros((bsz, 1, q_len, context_len + q_len), device=device)
        return mask.masked_fill(~allow.unsqueeze(1), float("-inf"))

    def _context_is_projected(self, value) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.detach().flatten()[0].cpu().item())
        return bool(value)

    def project_target_context(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden_states))

    def _prepare_target_context(
        self,
        target_hidden_states: torch.Tensor,
        context_is_projected=False,
    ) -> torch.Tensor:
        if self._context_is_projected(context_is_projected):
            if target_hidden_states.shape[-1] != self.hidden_size:
                raise ValueError(
                    "Projected context should already have hidden_size dimension "
                    f"{self.hidden_size}, got {target_hidden_states.shape[-1]}"
                )
            return target_hidden_states
        if target_hidden_states.shape[-1] != self.raw_context_dim:
            raise ValueError(
                "Raw target context has unexpected last dimension "
                f"{target_hidden_states.shape[-1]}; expected {self.raw_context_dim}. "
                "Pass context_is_projected=True for projected-context cache."
            )
        return self.project_target_context(target_hidden_states)

    def _forward_backbone(
        self,
        *,
        target_hidden_states: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_limit_positions: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        context_is_projected=False,
        past_key_values: Optional[object] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self._prepare_target_context(
            target_hidden_states,
            context_is_projected=context_is_projected,
        )
        if attention_mask is None:
            if context_limit_positions is None:
                context_limit_positions = position_ids[:, 0] - 1
            attention_mask = self._build_attention_mask(
                context_len=target_hidden_states.shape[1],
                position_ids=position_ids,
                context_limit_positions=context_limit_positions,
                context_attention_mask=context_attention_mask,
            ).to(dtype=hidden_states.dtype)
        full_position_ids = torch.cat(
            [
                torch.arange(target_hidden_states.shape[1], device=position_ids.device)
                .unsqueeze(0)
                .expand(position_ids.shape[0], -1),
                position_ids,
            ],
            dim=1,
        )
        position_embeddings = _get_position_embeddings(
            self.rotary_emb,
            torch.cat([target_hidden_states, hidden_states], dim=1),
            full_position_ids,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        return self.norm(hidden_states)

    def _last_valid_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        masked_positions = attention_mask.long() * positions.view(1, -1)
        return masked_positions.max(dim=1).values

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        target_ids: torch.Tensor = None,
        root_hidden: Optional[torch.Tensor] = None,
        context_is_projected=False,
        **kwargs,
    ) -> ParallelDraftOutput:
        del kwargs
        bsz = input_ids.shape[0]
        device = input_ids.device
        if self.proposal_len <= 0:
            raise ValueError("item_token_num must be positive for parallel_draft")
        if target_ids.shape[1] != self.item_token_num:
            raise ValueError(
                f"Expected target_ids with {self.item_token_num} positions, "
                f"got {target_ids.shape[1]}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                target_hidden_states.shape[:2],
                dtype=torch.long,
                device=device,
            )
        last_pos = self._last_valid_indices(attention_mask)
        if root_hidden is None:
            if target_last_hidden_states is None:
                raise ValueError("root_hidden or target_last_hidden_states must be provided")
            batch_idx = torch.arange(bsz, device=device)
            root_hidden = target_last_hidden_states[batch_idx, last_pos, :]
        offsets = torch.arange(1, self.proposal_len + 1, device=device).view(1, -1)
        draft_position_ids = last_pos.unsqueeze(1) + offsets
        eval_mask = torch.ones(
            (bsz, self.item_token_num),
            dtype=torch.bool,
            device=device,
        )

        noise_ids = torch.full(
            (bsz, self.proposal_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        draft_hidden = self._forward_backbone(
            target_hidden_states=target_hidden_states,
            noise_embedding=self.embed_tokens(noise_ids),
            position_ids=draft_position_ids,
            context_limit_positions=last_pos,
            context_attention_mask=attention_mask,
            context_is_projected=context_is_projected,
        )
        logits, target_local_ids = self.draft_head(
            draft_hidden=draft_hidden,
            root_hidden=root_hidden,
            target_ids=target_ids,
        )
        return ParallelDraftOutput(
            logits=logits,
            target_ids=target_ids,
            target_local_ids=target_local_ids,
            eval_mask=eval_mask,
            hidden_states=draft_hidden,
        )

    @torch.inference_mode()
    def draft_hidden_for_item_tokens(
        self,
        *,
        target_hidden_states: torch.Tensor,
        prompt_last_position: torch.Tensor,
        context_attention_mask: Optional[torch.Tensor] = None,
        proposal_len: Optional[int] = None,
        context_is_projected=False,
    ):
        proposal_len = int(proposal_len or self.proposal_len)
        device = target_hidden_states.device
        noise_ids = torch.full(
            (target_hidden_states.shape[0], proposal_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        prompt_last_position = prompt_last_position.to(device=device, dtype=torch.long)
        if prompt_last_position.ndim == 0:
            prompt_last_position = prompt_last_position.view(1).expand(
                target_hidden_states.shape[0]
            )
        position_ids = prompt_last_position.unsqueeze(1) + torch.arange(
            1, proposal_len + 1, device=device
        ).view(1, -1)
        return self._forward_backbone(
            target_hidden_states=target_hidden_states,
            noise_embedding=self.embed_tokens(noise_ids),
            position_ids=position_ids,
            context_limit_positions=prompt_last_position,
            context_attention_mask=context_attention_mask,
            context_is_projected=context_is_projected,
        )
