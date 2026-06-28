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

#kernel2: Flash Attention

@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Q 方向的分块大小
    BLOCK_N: tl.constexpr,  # KV 方向的分块大小
):    
    start_m = tl.program_id(0)   # Q 块索引
    off_h = tl.program_id(1)     # head 索引
    seq_idx = tl.program_id(2)   # 序列索引

    kv_head_idx = off_h // (num_heads // num_kv_heads)

    #读取该序列的起止位置
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    if start_m * BLOCK_M >= seq_len:
        return
    
    # Q 块内的 token 偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)  # head_dim 维度的偏移

    # 加载 Q 块 (BLOCK_M, head_dim)
    # Q 存储格式: (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    mask_m = offs_m < seq_len  # 有效 token 的 mask
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # 【Online Softmax 状态初始化】
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10  # running max
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)  # 累加器

    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    #主循环，逐块处理kv
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # 加载 K 块 (head_dim, BLOCK_N) — 注意 K 的存储格式
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # QK^T — 计算 attention scores (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale  # 除以 sqrt(head_dim)

        # 【因果 mask】只允许 attend 到当前位置及之前
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # 【Online Softmax 更新】
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)  # 旧累加器的缩放因子
        p = tl.exp(qk - m_i_new[:, None])  # 当前块的 softmax（未归一化）
        
        # 缩放旧的累加器
        acc = acc * alpha[:, None]
        
        # 加载 V 块 (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # 累加加权值: acc += softmax(block_n) @ V_block_n
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # 更新 normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # 【最终归一化】acc / l_i = softmax(QK^T/sqrt(d)) @ V
    acc = acc / l_i[:, None]
    
    # 存储输出 O
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Prefill 阶段的 FlashAttention 封装。
    
    【为什么 prefill 用 FlashAttention 而 decode 用 PagedAttention？】
    - Prefill: 同时处理一个序列的很多 token（可能几百到几千），
      Q 也是多个 token，需要用 FlashAttention 高效计算密集 attention
    - Decode: 每个序列只有 1 个新 token(Q)，但需要 attend 到很长的历史(KV)，
      且 KV 分散在不同物理块中，需要 PagedAttention 从块表读取
    
    【Block Size 选择】
    BLOCK_M 和 BLOCK_N 要根据 head_dim 动态选择：
        - head_dim 越小，每个元素占用显存越少，可以用更大的 block
        - 限制因素是 SRAM 大小（通常 48KB-256KB）
        - 太大的 block 会导致 """
        q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.empty_like(q)
    
    # 【自适应 block size】根据 head_dim 选择合适的 tile 大小
    # SRAM 预算: BLOCK_M * head_dim * 4(Q) + BLOCK_N * head_dim * 4(K) + BLOCK_N * head_dim * 4(V) + BLOCK_M * BLOCK_N * 4(scores)
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    num_seqs = cu_seqlens.shape[0] - 1
    
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # 【3D Grid】按 (Q block, head, sequence) 启动
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output