import trition
import triton.language as tl
from utils import get_context
import torch
import torch.nn as nn

# 算子1
@triton.jit
def store_kvcache_kernel(
    key_ptr,       # [输入] 新计算的 K 值 (num_tokens, num_kv_heads, head_dim)
    value_ptr,     # [输入] 新计算的 V 值
    k_cache_ptr,   # [输出] KV cache 中的 K 区域 (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,   # [输出] KV cache 中的 V 区域
    slot_mapping_ptr,  # [输入] 每个 token 对应的 cache slot 索引
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,  # 每块容纳的 token 数
):
    """
    将新计算的 K、V 值存入分页 KV cache。
    
    每个线程处理一个 (token, head) 对：
    - token_idx = program_id(0):  确定处理哪个 token
    - head_idx  = program_id(1):  确定处理哪个 head
    
    Cache 寻址:
        slot_idx = slot_mapping[token_idx]
        block_idx = slot_idx // block_size     # 哪个物理块
        block_offset = slot_idx % block_size    # 块内偏移
    
    【为什么需要这个 kernel？】
        vLLM 的 KV cache 不是按序列连续存储的，而是分散在不同的物理块中。
        每次计算新的 KV 后，需要根据 slot_mapping 将它们写入正确的物理位置。
        slot_mapping 由 ModelRunner 在数据准备阶段计算。
    """
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx == -1:
        return
    
    #将slot索引转换为(block_id, block_offset)
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    head_idx = tl.program_id(1)

    head_offsets = tl.arange(0,head_dim)
    # 输入的地址计算（二维：token -> head）
    input_offset = (token_idx * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    # 缓存的地址计算（三维：block -> position -> head）
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                   block_offset * num_kv_heads * head_dim +
                   head_idx * head_dim +
                   head_offsets) 
    
    # 加载并存储
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)

def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Python 包装函数：验证形状并启动 Triton kernel。
    
    Args:
        key/value: (num_tokens, num_kv_heads, head_dim) — 本次计算的新 KV
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim) — 分页缓存池
        slot_mapping: (num_tokens,) — 每个 token 的写入位置
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel(
        key_ptr=key,
        value_ptr=value,
        k_cache_ptr=k_cache,
        v_cache_ptr=v_cache,
        slot_mapping_ptr=slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
