import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    """
    KV cache 的最小管理单元。

    物理块：GPU 显存中的实际存储
    逻辑块：序列视角的连续块序列
    块表：逻辑块到物理块的映射
    """
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1          # 块内 token 的哈希值，-1 表示未计算或不完整块
        self.ref_count = 0      # 引用计数：有多少个序列在使用这个块
        self.token_ids = []     # 块内存储的 token ID（用于哈希冲突检测）


    def update(self, h: int, token_ids: list[int]):
        """更新块的哈希和 token 信息（块变完整时调用）"""
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        """重置块为空闲状态"""
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    """
    全局块分配器，管理整个 KV cache 池。

    核心数据结构：
        - blocks: 所有 Block 对象的数组
        - hash_to_block_id: 哈希值到块 ID 的前缀缓存映射
        - free_block_ids: 空闲块队列（deque，O(1) 分配）
        - used_block_ids: 已使用块集合（O(1) 检查）
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size: int = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # ===== 哈希计算 =====

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """
        计算块的上下文敏感哈希。

        【前缀哈希链】prefix_hash_value 是前一个块的哈希值，
        将其混入当前哈希确保上下文敏感性：
        即使两个序列有相同的 token 块，如果它们的前缀不同，
        也会产生不同的哈希值，不会错误共享。
        """
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            # 将前缀哈希混入（8 字节 little-endian）
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        # 混入 token_ids（int32 数组的原始字节）
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # ===== 块分配/释放底层操作 =====

    def _allocate_block(self, block_id: int) -> Block:
        """从空闲池中分配一个块"""
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """将块归还空闲池（仅在 ref_count == 0 时调用）"""
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # ===== Prefill 阶段的序列级操作 =====

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来容纳整个序列"""
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        """
        为序列分配所有 KV cache 块（Prefill 时调用）。

        【前缀缓存流程】
        对每个逻辑块（逐块处理）：
        1. 对完整块计算哈希（不完整块哈希为 -1）
        2. 在 hash_to_block_id 中查找
        3. 命中 + 无冲突 -> 增加 ref_count，更新 num_cached_tokens
        4. 未命中或冲突 -> 从 free 队列分配新块

        哈希冲突处理：
        - hash_to_block_id 找到相同哈希的块
        - 但 token_ids 不相同 -> 判定为冲突
        - 重新分配新块
        """
        h = -1  # 前缀哈希，逐块串联
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # 只对完整块计算哈希，不完整块的哈希总是 -1
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 缓存命中检查：哈希存在且 token_ids 精确匹配
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # 【前缀缓存命中！】
                seq.num_cached_tokens += self.block_size
                if block_id not in self.used_block_ids:
                    # 块有哈希记录但不在 used（可能是之前的 prefix 命中了但还没分配）
                    block = self._allocate_block(block_id)
                else:
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1  # 增加引用计数
            else:
                # 【缓存未命中】从空闲池分配新块
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        """
        释放序列的所有块（序列完成或被抢占时调用）。

        递减每个块的 ref_count，归零后归还空闲池。
        """
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = []
        seq.num_cached_tokens = 0

    # ===== Decode 阶段的增量操作 =====

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以追加一个 token。

        只有当前位置是块边界（num_tokens % block_size == 0）
        且需要新块时，才检查是否有空闲块。
        块内追加（同一块继续填充）总是可以的。
        """
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        """
        追加 token 后的块管理（token 已加入 seq，但 GPU 上的块还没更新）。

        三种情况：
        1. num_tokens % block_size == 0: 最后一个块刚刚变完整
           -> 计算哈希，建立前缀缓存
        2. num_tokens % block_size == 1: 需要新块
           -> 分配新块，加入 block_table
        3. 其他情况: 继续填充当前块，无需操作
        """
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        if seq.num_tokens % self.block_size == 0:
            # 最后一个块刚好填满 -> 计算哈希，加入缓存
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        elif seq.num_tokens % self.block_size == 1:
            # 新块开始 -> 分配新物理块
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        else:
            # 继续填充当前块，块哈希保持 -1（不完整）
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
