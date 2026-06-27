"""
=============================================================================
序列 (Sequence) — LLM 推理的基本调度单元
=============================================================================

【什么是 Sequence？】
    一个 Sequence 代表一个完整的对话/生成过程，包含：
    - 所有 token ID（prompt + 已生成的 completion）
    - 状态机（WAITING → RUNNING → FINISHED）
    - 分页 KV cache 的块表（block_table）
    - 采样参数（temperature, max_tokens 等）

【状态机 (SequenceStatus)】
    WAITING  → 刚创建，等待调度器分配 GPU 资源
    RUNNING  → 正在被模型处理（prefill 或 decode）
    FINISHED → 生成完成（遇到 EOS 或达到 max_tokens）

【关键属性】

    block_table: list[int]
        该序列使用的物理 KV cache 块 ID 列表。
        块表是 PagedAttention 的核心数据结构：
        - block_table[i] 表示第 i 个逻辑块的物理块 ID
        - 不同序列可以共享某些块（前缀缓存）

    num_cached_tokens: int
        通过前缀缓存命中的 token 数。
        在 Prefill 时，这部分 token 的 KV 已在 cache 中，
        无需重新计算，直接从 cache 读取即可。

【序列化优化 (__getstate__/__setstate__)】
    在多卡场景下，Sequence 对象需要通过共享内存传递。
    我们做了优化：
    - Prefill 阶段（num_completion_tokens == 0）: 传输全部 token_ids
    - Decode 阶段: 只传输最后一个 token
    因为 Decode 时其他 worker 只需要知道当前要生成什么，
    不需要完整的 token 历史。
"""

from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    LLM 推理的基本调度单元。
    
    每个 Sequence 跟踪一个独立的生成过程，包含所有必要的状态信息。
    """
    counter = count()  # 全局自增 ID 计数器

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        
        # 【copy()】关键！必须复制一份，否则外部修改变量会影响内部状态
        self.token_ids = copy(token_ids)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        
        # Token 计数
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)  # 初始时全部是 prompt
        
        # 前缀缓存命中的 token 数（初始为 0，由 BlockManager.allocate 更新）
        self.num_cached_tokens = 0
        
        # 【块表】物理 KV cache 块 ID 列表
        self.block_table = []
        
        # 采样参数（从 SamplingParams 提取到 Sequence 上，方便访问）
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 completion token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # ===== 块相关计算 =====
    
    @property
    def num_cached_blocks(self):
        """前缀缓存命中的块数（向上取整）"""
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        """序列总共占用的块数（向上取整）"""
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        """最后一个（不完整）块包含的 token 数"""
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

    def block(self, i):
        """获取第 i 个块的 token ID 列表"""
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    # ===== Token 操作 =====

    def append_token(self, token_id):
        """追加一个生成的 token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    # ===== 序列化（多卡共享内存通信） =====
    
    def __getstate__(self):
        """
        序列化优化：
        - Prefill 阶段: 传输全部 token_ids（worker 需要完整的 prompt 信息）
        - Decode 阶段: 只传输最后一个 token（大幅减少通信量）
        """
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """反序列化"""
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            self.token_ids = last_token_or_ids  # 完整 token 列表
        else:
            self.token_ids = [last_token_or_ids]  # 只有最后一个 token
        self.last_token = self.token_ids[-1] if self.token_ids else None