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

@trition.jit
def paged_attention_decode_kernel(
    output_ptr,      # [输出] Attention 结果
    query_ptr,       # [输入] Q 值 (batch_size, num_heads, head_dim)
    k_cache_ptr,     # [输入] K 缓存池 (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,     # [输入] V 缓存池
    block_tables_ptr,    # [输入] 块表 (batch_size, max_num_blocks)
    context_lens_ptr,    # [输入] 各序列的上下文长度
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,  # 每次处理的 KV token 块大小
):
    """
    【PagedAttention Decode Kernel】
    
    Decode 阶段的特殊场景：
    - 每个序列只有 1 个 Q token，需要 attend 到大量历史 KV token
    - KV 分散在不同物理块中（分页存储），不能直接连续读取
    
    本 kernel 通过 block_tables（块表）将逻辑位置映射到物理块，
    逐块从 KV cache 中加载并计算 attention。
    
    【Grid 设计】
        - program_id(0): batch_idx  — 序列索引
        - program_id(1): head_idx   — Q head 索引
        每个线程处理一个序列的一个 head
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # GQA 映射
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # 加载单个 Q 向量 (head_dim,)
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Online Softmax 状态
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N

    if token_start < context_len:
        offs_n = token_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < context_len

        for i in range(BLOCK_N):
            token_idx = token_start + i
            if token_idx < context_len:
                block_num = token_idx // block_size
                block_offset = token_idx % block_size

                if block_num < max_num_blocks:
                    block_table_offset = batch_idx * max_num_blocks + block_num
                    physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                    if physical_block_idx != -1:
                        k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                block_offset * num_kv_heads * head_dim +
                                kv_head_idx * head_dim + offs_d)
                        k_vec = tl.load(k_cache_ptr + k_offset)

                        score = tl.sum(q * k_vec) * scale

                        mask_i = tl.arange(0, BLOCK_N) == i
                        qk = tl.where(mask_i, score, qk)
        
        qk = tl.where(mask_n, qk, -1e10)
        
        m_ij = tl.max(qk)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)

        acc = acc * alpha
        l_i = l_i * alpha

        #开始累积v
        for i in range(BLOCK_N):
            token_idx = token_start + i
            if token_idx < context_len:
                block_num = token_idx // block_size
                block_offset = token_idx % block_size

                if block_num < max_num_blocks:
                    block_table_offset = batch_idx * max_num_blocks + block_num
                    physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                    if physical_block_idx != -1:
                        v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                block_offset * num_kv_heads * head_dim +
                                kv_head_idx * head_dim + offs_d)
                        v_vec = tl.load(v_cache_ptr + v_offset)

                        mask_i = tl.arange(0, BLOCK_N) == i
                        weight_i = tl.sum(tl.where(mask_i, p, 0.0))

                        acc = acc + weight_i * v_vec
                        l_i = l_i + weight_i    
        m_i = m_i_new

            # 归一化输出
    output = acc / l_i
    
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Decode 阶段的 PagedAttention 封装。
    
    与 FlashAttention prefill 对比：
    - Q shape: (batch_size, num_heads, head_dim) — 每个序列只有 1 个 token
    - KV 从分页缓存池读取：不连续存储，通过 block_tables 映射
    - block_tables[i][j] = 序列 i 的第 j 个逻辑块对应的物理块 ID
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    query = query.contiguous()
    output = torch.empty_like(query)
    
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    # Grid: (batch_size, num_heads) — 每个 (序列, head) 启动一个线程
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output, query, k_cache, v_cache,
        block_tables, context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output

class Attention(nn.Module):
    """
    整合了 KV Cache 存储、FlashAttention (prefill) 和 PagedAttention (decode)
    的完整 Attention 层。
    
    【Forward 逻辑】
        1. 将当前计算的 K, V 存入 KV cache（如果 cache 已分配）
        2. 如果是 Prefill: 调用 flash_attention_prefill (varlen)
        3. 如果是 Decode:  调用 paged_attention_decode (从 cache 读取)
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        # 【GQA】num_kv_heads 可以小于 num_heads（默认等于）
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        # KV cache 初始化为空张量，由 ModelRunner.allocate_kv_cache() 分配
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Prefill 时 (total_tokens, num_heads, head_dim)
               Decode 时  (batch_size, num_heads, head_dim)
            k, v: 与 q 对应
        
        Returns:
            (total_tokens_or_batch_size, num_heads * head_dim) — 合并 head 维度
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ===== Step 1: 存储当前 KV 到缓存 =====
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # 处理 4D batched → 3D varlen 的转换
            if k.dim() == 4:
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # 计算 scale = 1 / sqrt(head_dim)
        scale = self.scale / (self.head_dim ** 0.5)

        # ===== Step 2: 选择 Attention 算法 =====
        if context.is_prefill:
            # 【Prefill】使用 FlashAttention（varlen 模式）
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # 【Decode】使用 PagedAttention（从 KV cache 读取历史）
            o = paged_attention_decode(
                q, k_cache, v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)