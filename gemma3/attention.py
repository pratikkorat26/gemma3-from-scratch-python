import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from gemma3.rope import apply_rope_single
from gemma3.paged_kv import PagedKVCache


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = torch.mean(x_f * x_f, dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale)
        if self.shift is not None:
            out = out + self.shift
        return out.to(orig_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        rope=None,
        sliding_window: Optional[int] = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: Optional[float] = None,
        dtype=None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        scale_base = float(query_pre_attn_scalar) if query_pre_attn_scalar is not None else float(head_dim)
        self.scale = scale_base ** -0.5
        self.sliding_window = sliding_window
        self.rope = rope

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def _append_to_paged_cache(
        self,
        *,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
    ) -> None:
        batch_size, num_kv_groups, q_len, head_dim = k_new.shape
        block_size = paged_kv_cache.block_size
        device = k_new.device

        start_pos = kv_lens.unsqueeze(1)
        token_offsets = torch.arange(q_len, device=device, dtype=torch.long).unsqueeze(0)
        positions = start_pos + token_offsets
        block_slots = positions // block_size
        block_offsets = positions % block_size
        block_ids = block_tables.gather(1, block_slots)

        if (block_ids < 0).any():
            raise ValueError("paged KV block table is missing an assigned block")

        batch_idx_all = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, q_len)
        token_idx_all = torch.arange(q_len, device=device).unsqueeze(0).expand(batch_size, q_len)
        valid_batch = batch_idx_all.reshape(-1)
        valid_token = token_idx_all.reshape(-1)
        valid_offsets = block_offsets.reshape(-1)
        valid_block_ids = block_ids.reshape(-1)

        k_vals = k_new[valid_batch, :, valid_token, :]
        v_vals = v_new[valid_batch, :, valid_token, :]
        k_vals = k_vals.permute(1, 0, 2).reshape(num_kv_groups, batch_size * q_len, head_dim)
        v_vals = v_vals.permute(1, 0, 2).reshape(num_kv_groups, batch_size * q_len, head_dim)

        num_blocks = paged_kv_cache.k_blocks.shape[0]
        bi = valid_block_ids.unsqueeze(0).expand(num_kv_groups, -1)
        of = valid_offsets.unsqueeze(0).expand(num_kv_groups, -1)

        k_flat = paged_kv_cache.k_blocks.view(num_kv_groups, num_blocks * block_size, head_dim)
        v_flat = paged_kv_cache.v_blocks.view(num_kv_groups, num_blocks * block_size, head_dim)
        idx = (bi * block_size + of).unsqueeze(-1).expand(-1, -1, head_dim)
        k_flat.scatter_(1, idx, k_vals)
        v_flat.scatter_(1, idx, v_vals)

    def _gather_batch_kv(
        self,
        *,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        paged_kv_cache: PagedKVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather dense KV for a batch, right-padded to ``max_seq_len``.

        Returns:
            k, v with shape ``[B, G, max_seq_len, D]``.
        """
        batch_size = block_tables.shape[0]
        block_size = paged_kv_cache.block_size
        device = block_tables.device
        dtype = paged_kv_cache.k_blocks.dtype

        k = torch.zeros(
            (batch_size, self.num_kv_groups, max_seq_len, self.head_dim),
            device=device,
            dtype=dtype,
        )
        v = torch.zeros_like(k)

        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx].item())
            if seq_len <= 0:
                continue
            needed_blocks = (seq_len + block_size - 1) // block_size
            if needed_blocks > block_tables.shape[1]:
                raise ValueError("paged KV block table does not cover the active sequence")
            block_ids = block_tables[batch_idx, :needed_blocks]
            if (block_ids < 0).any() or block_ids.numel() != needed_blocks:
                raise ValueError("paged KV block table does not cover the active sequence")
            k_chunks = paged_kv_cache.k_blocks[block_ids]
            v_chunks = paged_kv_cache.v_blocks[block_ids]
            k_seq = k_chunks.permute(1, 0, 2, 3).reshape(
                self.num_kv_groups, needed_blocks * block_size, self.head_dim
            )[:, :seq_len, :]
            v_seq = v_chunks.permute(1, 0, 2, 3).reshape(
                self.num_kv_groups, needed_blocks * block_size, self.head_dim
            )[:, :seq_len, :]
            k[batch_idx, :, :seq_len, :] = k_seq
            v[batch_idx, :, :seq_len, :] = v_seq
        return k, v

    def _build_attn_mask(
        self,
        *,
        q_len: int,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return additive mask ``[B, 1, q_len, max_seq_len]`` (0 keep, -inf block)."""
        batch_size = seq_lens.shape[0]
        q_pos = kv_lens.unsqueeze(1) + torch.arange(q_len, device=device, dtype=torch.long).unsqueeze(0)
        k_pos = torch.arange(max_seq_len, device=device, dtype=torch.long).view(1, 1, max_seq_len)
        q_pos = q_pos.unsqueeze(-1)  # [B, q_len, 1]

        blocked = k_pos > q_pos
        if self.sliding_window is not None:
            blocked = blocked | (k_pos < q_pos - self.sliding_window + 1)
        # Pad keys beyond each sequence's true length.
        blocked = blocked | (k_pos >= seq_lens.view(batch_size, 1, 1))

        mask = torch.zeros((batch_size, q_len, max_seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask.unsqueeze(1)  # [B, 1, q_len, S]

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, kv_lens: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        batch_size, _, q_len, _ = q.shape
        device = q.device
        if bool((kv_lens == kv_lens[0]).all().item()):
            cos, sin = self.rope.get_cos_sin(
                seq_len=q_len,
                offset=int(kv_lens[0].item()),
                device=device,
                dtype=dtype,
            )
            return apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)

        q_rows = []
        k_rows = []
        for batch_idx in range(batch_size):
            cos, sin = self.rope.get_cos_sin(
                seq_len=q_len,
                offset=int(kv_lens[batch_idx].item()),
                device=device,
                dtype=dtype,
            )
            q_rows.append(apply_rope_single(q[batch_idx : batch_idx + 1], cos, sin))
            k_rows.append(apply_rope_single(k[batch_idx : batch_idx + 1], cos, sin))
        return torch.cat(q_rows, dim=0), torch.cat(k_rows, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        batch_size, q_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self._apply_rope(q, k, kv_lens, x.dtype)

        self._append_to_paged_cache(
            k_new=k,
            v_new=v,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache,
        )

        seq_lens = kv_lens + q_len
        max_seq_len = int(seq_lens.max().item())
        k_cache, v_cache = self._gather_batch_kv(
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            paged_kv_cache=paged_kv_cache,
        )

        # [B, H, T, D] via GQA expand without Python batch loops.
        q_b = q.reshape(batch_size, self.num_kv_groups, self.group_size, q_len, self.head_dim)
        q_b = q_b.permute(0, 2, 1, 3, 4).reshape(
            batch_size * self.group_size, self.num_kv_groups, q_len, self.head_dim
        )
        k_b = k_cache.unsqueeze(1).expand(batch_size, self.group_size, self.num_kv_groups, max_seq_len, self.head_dim)
        k_b = k_b.reshape(batch_size * self.group_size, self.num_kv_groups, max_seq_len, self.head_dim)
        v_b = v_cache.unsqueeze(1).expand(batch_size, self.group_size, self.num_kv_groups, max_seq_len, self.head_dim)
        v_b = v_b.reshape(batch_size * self.group_size, self.num_kv_groups, max_seq_len, self.head_dim)

        attn_mask = self._build_attn_mask(
            q_len=q_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            kv_lens=kv_lens,
            dtype=q.dtype,
            device=device,
        )
        # Expand mask across GQA groups: [B, 1, T, S] -> [B*g, 1, T, S]
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.group_size, 1, q_len, max_seq_len)
        attn_mask = attn_mask.reshape(batch_size * self.group_size, 1, q_len, max_seq_len)

        out = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        out = out.reshape(batch_size, self.group_size, self.num_kv_groups, q_len, self.head_dim)
        out = out.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_heads, q_len, self.head_dim)
        out = out.transpose(1, 2).reshape(batch_size, q_len, self.num_heads * self.head_dim)
        return self.out_proj(out)
