from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

LayerCache = List[Tuple[torch.Tensor, torch.Tensor]]


def _common_prefix_len(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _clone_kv(kv: LayerCache) -> LayerCache:
    return [(k.detach().clone(), v.detach().clone()) for (k, v) in kv]


def _kv_bytes(kv: LayerCache) -> int:
    total = 0
    for k, v in kv:
        total += int(k.numel()) * int(k.element_size())
        total += int(v.numel()) * int(v.element_size())
    return total


def _slice_kv_prefix(kv: LayerCache, prefix_tokens: int) -> LayerCache:
    return [
        (
            k[:, :, :prefix_tokens, :].detach().clone(),
            v[:, :, :prefix_tokens, :].detach().clone(),
        )
        for (k, v) in kv
    ]


@dataclass
class PrefixCacheHit:
    matched_tokens: int
    past_kv: LayerCache


@dataclass
class PrefixCacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    bytes_used: int = 0


class _RadixNode:
    def __init__(self) -> None:
        self.children: Dict[int, Tuple[Tuple[int, ...], _RadixNode]] = {}
        self.key: Optional[Tuple[int, ...]] = None


@dataclass
class _Entry:
    key: Tuple[int, ...]
    full_past_kv: LayerCache
    num_tokens: int
    num_bytes: int


class PrefixCache:
    """
    In-memory radix-prefix cache for prompt KV reuse.

    v1 stores KV snapshots for full prompts and reuses the longest cached key
    that is a prefix of the incoming prompt. Returned KV is sliced to requested
    prefix length and cloned before handing to the scheduler.
    """

    def __init__(self, *, min_tokens: int, max_bytes: int):
        self.min_tokens = max(1, int(min_tokens))
        self.max_bytes = max(0, int(max_bytes))
        self._root = _RadixNode()
        self._entries: Dict[Tuple[int, ...], _Entry] = {}
        self._lru: "OrderedDict[Tuple[int, ...], None]" = OrderedDict()
        self._stats = PrefixCacheStats()

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            inserts=self._stats.inserts,
            evictions=self._stats.evictions,
            bytes_used=self._stats.bytes_used,
        )

    def _touch(self, key: Tuple[int, ...]) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)

    def _set_value(self, key: Tuple[int, ...]) -> None:
        node = self._root
        remaining = key
        while True:
            if not remaining:
                node.key = key
                return

            first = remaining[0]
            edge = node.children.get(first)
            if edge is None:
                child = _RadixNode()
                child.key = key
                node.children[first] = (remaining, child)
                return

            label, child = edge
            shared = _common_prefix_len(label, remaining)
            if shared == len(label):
                node = child
                remaining = remaining[shared:]
                continue

            split_node = _RadixNode()
            old_suffix = label[shared:]
            split_node.children[old_suffix[0]] = (old_suffix, child)

            new_label = label[:shared]
            node.children[first] = (new_label, split_node)

            new_suffix = remaining[shared:]
            if not new_suffix:
                split_node.key = key
            else:
                new_child = _RadixNode()
                new_child.key = key
                split_node.children[new_suffix[0]] = (new_suffix, new_child)
            return

    def _clear_value(self, key: Tuple[int, ...]) -> None:
        node = self._root
        remaining = key
        while True:
            if not remaining:
                if node.key == key:
                    node.key = None
                return

            edge = node.children.get(remaining[0])
            if edge is None:
                return
            label, child = edge
            if len(remaining) < len(label) or remaining[: len(label)] != label:
                return
            node = child
            remaining = remaining[len(label) :]

    def _evict_until_fit(self) -> None:
        while self._stats.bytes_used > self.max_bytes and self._lru:
            key, _ = self._lru.popitem(last=False)
            entry = self._entries.pop(key, None)
            if entry is None:
                continue
            self._stats.bytes_used -= entry.num_bytes
            self._stats.evictions += 1
            self._clear_value(key)

    def lookup(self, prompt_ids: List[int], *, max_prefix_tokens: int) -> Optional[PrefixCacheHit]:
        if len(prompt_ids) < self.min_tokens:
            self._stats.misses += 1
            return None

        query = tuple(prompt_ids)
        node = self._root
        pos = 0
        best_key: Optional[Tuple[int, ...]] = None
        best_len = 0

        while True:
            if node.key is not None:
                best_key = node.key
                best_len = pos

            if pos >= len(query):
                break

            edge = node.children.get(query[pos])
            if edge is None:
                break

            label, child = edge
            remaining = query[pos:]
            shared = _common_prefix_len(label, remaining)
            if shared < len(label):
                break

            pos += shared
            node = child

        if best_key is None:
            self._stats.misses += 1
            return None

        entry = self._entries.get(best_key)
        if entry is None:
            self._stats.misses += 1
            return None

        matched = min(best_len, max_prefix_tokens, entry.num_tokens)
        if matched < self.min_tokens:
            self._stats.misses += 1
            return None

        self._touch(best_key)
        self._stats.hits += 1
        return PrefixCacheHit(
            matched_tokens=matched,
            past_kv=_slice_kv_prefix(entry.full_past_kv, matched),
        )

    def insert(self, prompt_ids: List[int], past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]) -> None:
        if len(prompt_ids) < self.min_tokens:
            return
        if past_kv is None:
            return
        if any(layer is None for layer in past_kv):
            return

        key = tuple(prompt_ids)
        full_kv = _clone_kv([(k, v) for (k, v) in past_kv if k is not None and v is not None])
        if not full_kv:
            return

        num_bytes = _kv_bytes(full_kv)
        if num_bytes > self.max_bytes:
            return

        previous = self._entries.get(key)
        if previous is not None:
            self._stats.bytes_used -= previous.num_bytes

        self._entries[key] = _Entry(
            key=key,
            full_past_kv=full_kv,
            num_tokens=len(prompt_ids),
            num_bytes=num_bytes,
        )
        self._set_value(key)
        self._lru[key] = None
        self._touch(key)
        self._stats.bytes_used += num_bytes
        self._stats.inserts += 1
        self._evict_until_fit()
