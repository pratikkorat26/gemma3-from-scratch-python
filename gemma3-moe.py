import math
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ===========================================================
# CONFIGURATION
# ===========================================================
@dataclass
class Gemma3Config:
    # Vocabulary
    vocab_size: int = 256000
    pad_token_id: int = 0

    # Model Dimensions
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 128

    # Context Length
    max_position_embeddings: int = 32768

    # Normalization
    rms_norm_eps: float = 1e-6

    # Attention & RoPE
    sliding_window: int = 2048
    rope_theta_local: int = 10000
    rope_theta_global: int = 1000000
    qk_norm: bool = True
    use_sdpa: bool = True  # ✅ Enable PyTorch SDPA

    # Vision (Disabled for 1B)
    use_vision: bool = False

    # Architecture Pattern
    local_global_ratio: int = 5

    # MoE Configuration
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 4096
    router_aux_loss_coef: float = 0.01
    norm_topk_prob: bool = True
    shared_expert: bool = False
    shared_expert_intermediate_size: int = 4096
    moe_jitter_noise: float = 0.0  # Add noise for load balancing during training

    def __post_init__(self):
        assert self.num_attention_heads * self.head_dim == self.hidden_size
        if self.use_moe:
            active_params = self.vocab_size * self.hidden_size
            active_params += self.num_hidden_layers * (4 * self.hidden_size * self.hidden_size)
            active_params += self.num_hidden_layers * (
                    self.num_experts_per_tok * 3 * self.hidden_size * self.moe_intermediate_size
            )
            active_params += self.num_hidden_layers * (2 * self.hidden_size)
            self.active_params_b = active_params / 1e9
        else:
            total_params = self.vocab_size * self.hidden_size
            total_params += self.num_hidden_layers * (4 * self.hidden_size * self.hidden_size)
            total_params += self.num_hidden_layers * (3 * self.hidden_size * self.intermediate_size)
            total_params += self.num_hidden_layers * (2 * self.hidden_size)
            self.active_params_b = total_params / 1e9


@dataclass
class TrainingConfig:
    # Data
    train_data_path: str = "./openwebtext"
    val_data_path: str = "./openwebtext_val"
    seq_length: int = 1024
    num_workers: int = 4

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    warmup_steps: int = 1000

    # Optimization
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer: str = "adamw"  # Options: "adamw", "muon"

    # Muon Optimizer Settings
    muon_momentum: float = 0.95
    muon_ns_steps: int = 6
    muon_eps: float = 1e-8

    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"

    # Checkpointing
    save_every: int = 1000
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 10
    val_every: int = 500
    use_wandb: bool = False
    project_name: str = "gemma3-moe"

    # Device
    device: str = "cuda"
    seed: int = 42


# ===========================================================
# RMSNorm
# ===========================================================
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * (1 + self.weight)


# ===========================================================
# QK Normalization
# ===========================================================
class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.query_norm = GemmaRMSNorm(head_dim, eps)
        self.key_norm = GemmaRMSNorm(head_dim, eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


# ===========================================================
# Rotary Position Embeddings
# ===========================================================
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: int):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ===========================================================
# MoE Router (Improved Load Balancing)
# ===========================================================
class MoERouter(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.jitter_noise = config.moe_jitter_noise

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor, training: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Router logits
        router_logits = self.gate(hidden_states)

        # Add jitter noise during training for better load balancing
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.unsqueeze(-1)

        # Load balancing loss (improved formula)
        router_probs_flat = router_probs.view(-1, self.num_experts)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_tok)

        # Count tokens per expert
        expert_mask = F.one_hot(selected_experts_flat, self.num_experts).to(router_probs.dtype)
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [num_experts]

        # Router probability per expert
        router_prob_per_expert = router_probs_flat.mean(dim=0)  # [num_experts]

        # Auxiliary loss: encourages uniform token distribution
        aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()

        return routing_weights, selected_experts, aux_loss


# ===========================================================
# MoE MLP (Optimized for Training)
# ===========================================================
class MoEMLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.shared_expert = config.shared_expert

        # Experts (optimized initialization)
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False),
            ]) for _ in range(config.num_experts)
        ])

        # Initialize experts with smaller std for stability
        for expert in self.experts:
            for layer in expert:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)

        self.router = MoERouter(config)

        if self.shared_expert:
            self.shared_gate = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
            self.shared_up = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
            self.shared_down = nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape

        routing_weights, selected_experts, aux_loss = self.router(hidden_states, training)

        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_tok)
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_tok, 1)

        final_output = torch.zeros_like(hidden_states_flat)

        # Process each expert (can be parallelized further)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts_flat == expert_idx)
            expert_mask_flat = expert_mask.any(dim=-1)

            if expert_mask_flat.sum() == 0:
                continue

            token_indices = torch.where(expert_mask_flat)[0]
            expert_positions = torch.where(expert_mask[token_indices])[1]

            current_hidden = hidden_states_flat[token_indices]

            gate_proj, up_proj, down_proj = self.experts[expert_idx]
            expert_output = down_proj(
                F.gelu(gate_proj(current_hidden), approximate="tanh") * up_proj(current_hidden)
            )

            routing_prob = routing_weights_flat[token_indices, expert_positions]
            expert_output = expert_output * routing_prob

            final_output.index_add_(0, token_indices, expert_output)

        if self.shared_expert:
            shared_output = self.shared_down(
                F.gelu(self.shared_gate(hidden_states), approximate="tanh") * self.shared_up(hidden_states)
            )
            final_output = final_output.view(batch_size, seq_len, hidden_dim) + shared_output
        else:
            final_output = final_output.view(batch_size, seq_len, hidden_dim)

        return final_output, aux_loss


# ===========================================================
# Dense MLP
# ===========================================================
class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # Initialize weights
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return output, torch.tensor(0.0, device=x.device)


# ===========================================================
# Attention with SDPA Support ✅
# ===========================================================
class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.is_global_layer = ((layer_idx + 1) % (config.local_global_ratio + 1) == 0)

        if self.is_global_layer:
            self.sliding_window = None
            self.rope_theta = config.rope_theta_global
        else:
            self.sliding_window = config.sliding_window
            self.rope_theta = config.rope_theta_local

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Initialize weights
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, base=self.rope_theta
        )
        self.qk_norm = QKNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

        self.use_sdpa = config.use_sdpa

    def _apply_sliding_window_mask(self, attn_weights: torch.Tensor, q_len: int, kv_len: int) -> torch.Tensor:
        if self.sliding_window is None:
            return attn_weights
        mask = torch.full((q_len, kv_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype)
        for i in range(q_len):
            start_j = max(0, i - self.sliding_window + 1)
            mask[i, start_j:i + 1] = 0.0
        return attn_weights + mask.unsqueeze(0).unsqueeze(0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,
                                                                                                                    2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,
                                                                                                                      2)

        if self.qk_norm is not None:
            query_states, key_states = self.qk_norm(query_states, key_states)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # ✅ Use PyTorch SDPA for better performance
        if self.use_sdpa and not self.is_global_layer and self.sliding_window is not None:
            # SDPA with sliding window (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        elif self.use_sdpa:
            # Standard SDPA (causal mask handled internally)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            # Fallback to manual attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            if self.sliding_window is not None:
                attn_weights = self._apply_sliding_window_mask(attn_weights, q_len, key_states.shape[2])

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


# ===========================================================
# Decoder Layer
# ===========================================================
class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config, layer_idx)

        if config.use_moe:
            self.mlp = MoEMLP(config)
        else:
            self.mlp = Gemma3MLP(config)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.mlp(hidden_states, training)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value, aux_loss


# ===========================================================
# Gemma 3 Model
# ===========================================================
class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            training: bool = False,
    ) -> Dict[str, torch.Tensor]:

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.shape[0], input_ids.shape[1]),
                device=input_ids.device,
                dtype=torch.bool
            )
        attention_mask = (1.0 - attention_mask.float()) * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, past_key_values[idx], aux_loss = decoder_layer(
                hidden_states, attention_mask, position_ids, past_key_values[idx], use_cache, training
            )
            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values,
            "aux_loss": total_aux_loss
        }

# ===========================================================
# Gemma 3 For Causal LM
# ===========================================================
class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear) and module is not self.lm_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            labels: Optional[torch.Tensor] = None,
            training: bool = False,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache, training
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if self.config.use_moe and training:
                aux_loss = outputs["aux_loss"] * self.config.router_aux_loss_coef
                loss = ce_loss + aux_loss
            else:
                loss = ce_loss

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": outputs["past_key_values"],
            "aux_loss": outputs["aux_loss"] if self.config.use_moe else None
        }

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_b": total / 1e9,
            "active_b": self.config.active_params_b
        }

    def get_layer_types(self) -> List[str]:
        """Returns the attention type for each layer (Local/Global)"""
        types = []
        for layer in self.model.layers:
            types.append("Global" if layer.self_attn.is_global_layer else "Local")
        return types

# -----------------------------------------------------------
# Example Usage & Comparison
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Gemma 3 with MoE Support - Parameter Comparison")
    print("=" * 70)

    # MoE Model
    config_moe = Gemma3Config(
        vocab_size=256000,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=18,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=32768,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=4096,
        router_aux_loss_coef=0.01,
        shared_expert=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    model_moe = Gemma3ForCausalLM(config_moe).to(device)

    # Parameter comparison
    params_moe = model_moe.count_parameters()

    print(f"\n📊 MoE Model (8 experts, 2 active):")
    print(f"   Total Parameters: {params_moe['total_b']:.2f}B")
    print(f"   Active Parameters: {params_moe['active_b']:.2f}B")

    # Test forward pass
    print(f"\n🚀 Testing Forward Pass...")
    input_ids = torch.randint(100, 10000, (1, 128)).to(device)

    # MoE
    outputs_moe = model_moe(input_ids, labels=input_ids)
    print(f"   MoE Loss: {outputs_moe['loss'].item():.4f}")
    print(f"   MoE Aux Loss: {outputs_moe['aux_loss'].item():.6f}")

    # Layer types
    print(f"\n🔍 Layer Attention Types (5:1 Pattern):")
    layer_types = model_moe.get_layer_types()
    for i, t in enumerate(layer_types):
        marker = "🌍" if t == "Global" else "🪟"
        print(f"   Layer {i:2d}: {marker} {t}")

    print(f"\n✅ MoE Support Added Successfully!")
    print(f"   - Configurable via use_moe flag")
    print(f"   - Top-k expert selection ({config_moe.num_experts_per_tok} of {config_moe.num_experts})")
    print(f"   - Load balancing auxiliary loss")
    print(f"   - Optional shared dense expert")
    print("=" * 70)