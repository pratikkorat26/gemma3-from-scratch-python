from collections import deque
from typing import Deque, Optional

from inference.types import RequestState


class KVBlockManager:
    def __init__(
        self,
        *,
        max_kv_cache_tokens: int,
        block_size: int,
        num_blocks: Optional[int] = None,
    ):
        if max_kv_cache_tokens <= 0:
            raise ValueError("max_kv_cache_tokens must be > 0")
        if block_size <= 0:
            raise ValueError("kv_block_size must be > 0")

        derived_blocks = max_kv_cache_tokens // block_size
        if num_blocks is None:
            num_blocks = derived_blocks
        if num_blocks <= 0:
            raise ValueError("num_kv_blocks must be > 0")

        self.max_kv_cache_tokens = int(max_kv_cache_tokens)
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self._free_block_ids: Deque[int] = deque(range(self.num_blocks))
        self._allocated_blocks = 0

    @property
    def reserved_tokens(self) -> int:
        return self._allocated_blocks * self.block_size

    @property
    def allocated_blocks(self) -> int:
        return self._allocated_blocks

    def _required_blocks(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return (int(token_count) + self.block_size - 1) // self.block_size

    def can_allocate_for(self, request: RequestState, total_tokens: int) -> bool:
        required_blocks = self._required_blocks(total_tokens)
        additional_blocks = required_blocks - len(request.block_table)
        return additional_blocks <= len(self._free_block_ids)

    def ensure_capacity(self, request: RequestState, total_tokens: int) -> bool:
        required_blocks = self._required_blocks(total_tokens)
        additional_blocks = required_blocks - len(request.block_table)
        if additional_blocks <= 0:
            return True
        if additional_blocks > len(self._free_block_ids):
            return False

        for _ in range(additional_blocks):
            request.block_table.append(self._free_block_ids.popleft())
        self._allocated_blocks += additional_blocks
        return True

    def release(self, request: RequestState) -> None:
        if not request.block_table:
            return
        for block_id in request.block_table:
            self._free_block_ids.append(block_id)
        self._allocated_blocks = max(0, self._allocated_blocks - len(request.block_table))
        request.block_table.clear()
