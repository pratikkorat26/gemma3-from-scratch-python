"""
Experimental paged-KV attention path (vLLM-style concepts) in a separate file.

This module is intentionally standalone and not wired into the main model yet.
It demonstrates:
1) Block-table based KV storage
2) Appending new KV into paged cache
3) Gathering per-request contiguous KV view for attention compute

Notes:
- This is a readable reference implementation, not a fused-kernel implementation.
- True vLLM performance needs custom kernels / paged attention kernels.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma3.attention import RMSNorm
from gemma3.rope import apply_rope_single


@dataclass
class PagedKVCache:
    """
    Global KV block storage.

    Shapes:
      k_blocks: [num_blocks, G, block_size, Hd]
      v_blocks: [num_blocks, G, block_size, Hd]
    """

    k_blocks: torch.Tensor
    v_blocks: torch.Tensor
    block_size: int

    @classmethod
    def empty(
        cls,
        *,
        num_blocks: int,
        num_kv_groups: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        k = torch.zeros((num_blocks, num_kv_groups, block_size, head_dim), device=device, dtype=dtype)
        v = torch.zeros((num_blocks, num_kv_groups, block_size, head_dim), device=device, dtype=dtype)
        return cls(k_blocks=k, v_blocks=v, block_size=block_size)


class PagedGroupedQueryAttention(nn.Module):
    """
    Readable paged-KV GQA attention.

    Expected runtime inputs:
      block_tables: [B, max_blocks_per_req], int64, -1 for unused slots
      kv_lens:      [B], current token count before this forward

    This module appends current-step K/V into cache and uses gathered KV for SDPA.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        *,
        rope=None,
        sliding_window: Optional[int] = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        self.rope = rope
        self.sliding_window = sliding_window

        base = float(query_pre_attn_scalar) if query_pre_attn_scalar is not None else float(head_dim)
        self.scale = base ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def _attn_mask(self, q_len: int, kv_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)  # [T,1]
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)                   # [1,S]
        mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
        mask = mask.masked_fill(k_pos > q_pos, float("-inf"))
        if self.sliding_window is not None:
            min_k = q_pos - self.sliding_window + 1
            mask = mask.masked_fill(k_pos < min_k, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,S]

    def _append_to_cache(
        self,
        *,
        k_new: torch.Tensor,          # [B,G,T,Hd]
        v_new: torch.Tensor,          # [B,G,T,Hd]
        block_tables: torch.Tensor,   # [B,max_blocks], int64
        kv_lens: torch.Tensor,        # [B]
        cache: PagedKVCache,
    ) -> None:
        B, G, T, Hd = k_new.shape
        bs = cache.block_size
        for b in range(B):
            start = int(kv_lens[b].item())
            for t in range(T):
                pos = start + t
                block_slot = pos // bs
                offset = pos % bs
                block_id = int(block_tables[b, block_slot].item())
                if block_id < 0:
                    raise ValueError("block_table has -1 for active position; allocator must pre-assign blocks")
                cache.k_blocks[block_id, :, offset, :] = k_new[b, :, t, :]
                cache.v_blocks[block_id, :, offset, :] = v_new[b, :, t, :]

    def _gather_sequence_kv(
        self,
        *,
        block_table: torch.Tensor,  # [max_blocks]
        seq_len: int,
        cache: PagedKVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns contiguous KV for one request:
          k_seq, v_seq: [G, seq_len, Hd]
        """
        bs = cache.block_size
        needed_blocks = (seq_len + bs - 1) // bs
        blocks = block_table[:needed_blocks]
        valid = blocks[blocks >= 0]
        if valid.numel() != needed_blocks:
            raise ValueError("block_table missing assigned block for active sequence range")

        k_chunks = cache.k_blocks[valid]  # [n_blk,G,bs,Hd]
        v_chunks = cache.v_blocks[valid]  # [n_blk,G,bs,Hd]

        k_seq = k_chunks.permute(1, 0, 2, 3).reshape(self.num_kv_groups, needed_blocks * bs, self.head_dim)
        v_seq = v_chunks.permute(1, 0, 2, 3).reshape(self.num_kv_groups, needed_blocks * bs, self.head_dim)
        return k_seq[:, :seq_len, :], v_seq[:, :seq_len, :]

    def forward(
        self,
        x: torch.Tensor,                 # [B,T,D]
        *,
        block_tables: torch.Tensor,      # [B,max_blocks], int64
        kv_lens: torch.Tensor,           # [B], int64, length before this forward
        cache: PagedKVCache,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B,H,T,Hd]
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # [B,G,T,Hd]
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # [B,G,T,Hd]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            # Per-request offset. For readability we apply one request at a time.
            q_out = []
            k_out = []
            for b in range(B):
                off = int(kv_lens[b].item())
                cos, sin = self.rope.get_cos_sin(seq_len=T, offset=off, device=device, dtype=x.dtype)
                q_out.append(apply_rope_single(q[b : b + 1], cos, sin))
                k_out.append(apply_rope_single(k[b : b + 1], cos, sin))
            q = torch.cat(q_out, dim=0)
            k = torch.cat(k_out, dim=0)

        self._append_to_cache(k_new=k, v_new=v, block_tables=block_tables, kv_lens=kv_lens, cache=cache)

        # Readable per-request attention compute. This can be vectorized later.
        outputs = []
        for b in range(B):
            seq_len = int(kv_lens[b].item()) + T
            k_seq, v_seq = self._gather_sequence_kv(
                block_table=block_tables[b],
                seq_len=seq_len,
                cache=cache,
            )  # [G,S,Hd], [G,S,Hd]

            q_b = q[b : b + 1]  # [1,H,T,Hd]
            k_b = k_seq.unsqueeze(0)  # [1,G,S,Hd]
            v_b = v_seq.unsqueeze(0)  # [1,G,S,Hd]

            # GQA reshape for SDPA without repeat_interleave.
            q_b = q_b.reshape(1, self.num_kv_groups, self.group_size, T, self.head_dim)
            q_b = q_b.reshape(self.group_size, self.num_kv_groups, T, self.head_dim)  # [g,G,T,Hd]

            k_b = k_b.unsqueeze(1).expand(1, self.group_size, -1, -1, -1)
            v_b = v_b.unsqueeze(1).expand(1, self.group_size, -1, -1, -1)
            k_b = k_b.reshape(self.group_size, self.num_kv_groups, seq_len, self.head_dim)
            v_b = v_b.reshape(self.group_size, self.num_kv_groups, seq_len, self.head_dim)

            mask = self._attn_mask(T, seq_len, device=device, dtype=q_b.dtype)
            out_b = F.scaled_dot_product_attention(
                q_b, k_b, v_b, attn_mask=mask, dropout_p=0.0, scale=self.scale
            )  # [g,G,T,Hd]

            out_b = out_b.reshape(1, self.group_size, self.num_kv_groups, T, self.head_dim)
            out_b = out_b.reshape(1, self.num_heads, T, self.head_dim)
            out_b = out_b.transpose(1, 2).reshape(1, T, self.num_heads * self.head_dim)
            outputs.append(out_b)

        out = torch.cat(outputs, dim=0)  # [B,T,H*Hd]
        return self.out_proj(out)
