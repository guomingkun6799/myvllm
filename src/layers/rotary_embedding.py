from turtle import forward

import torch.nn as nn
import torch

def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对 Query 或 Key 应用旋转位置编码。
    
    数学公式（对每一对维度 2i, 2i+1）：
        x_2i'   = x_2i * cos(θ) - x_{2i+1} * sin(θ)
        x_2i+1' = x_2i * sin(θ) + x_{2i+1} * cos(θ)
    
    支持两种输入模式：
    - 3D varlen: (total_tokens, num_heads, head_dim) — Prefill 阶段
    - 4D batched: (B, seq_len, num_heads, head_dim)  — 备用模式
    """
    if x.dim() == 3:
        total_tokens, num_heads, head_dim = x.shape
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x1, x2 = x.chunk(2, dim=-1)

        # 对每一对维度执行 2D 旋转
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # 【Batched 模式】备用，支持 (B, seq_len, num_heads, head_dim)
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # 广播维度: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        x1, x2 = x.chunk(2, dim=-1)

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)

class RotaryEmbedding(nn.Module):
    """
    管理 RoPE 的 cos/sin 缓存，并在 forward 时应用到 Q 和 K。
    
    参数说明：
        base: 频率基数，决定了旋转的"速度"。常见值：10000 (GPT-NeoX), 500000 (Llama)
              base 越大，频率越低，长距离位置变化越慢 = 更擅长长上下文
        rotary_embedding: 应用 RoPE 的维度数（通常等于 head_dim）
        max_position: 预计算缓存的最大位置数
    """
    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # 【Llama 3.2 专用参数】NTK-aware 频率缩放
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        self.rotary_embedding = rotary_embedding
        self.max_position = max_position

        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # 【Llama 3.2 NTK-aware 频率缩放】
            # 核心思想：不同波长的维度需要不同的频率缩放因子
            #   - 短波长（高频）= 短距离依赖 → 不缩放或小缩放
            #   - 长波长（低频）= 长距离依赖 → 大缩放
            import math
            inv_freq = self.inv_freq
            wave_len = 2 * math.pi / inv_freq  # 每个频率对应的波长
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # 无平滑过渡：硬阈值
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                # 有平滑过渡：根据波长线性插值缩放因子
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        # 预计算 cos/sin的缓存
        positions = torch.arange(self.max_position).float()
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # 【register_buffer】注册为 buffer（非参数张量），随模型一同加载/保存/移动设备
        self.register_buffer("cos_sin_cache", cos_sin_cache)
    
    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # 查表获取 cos 和 sin
        cos, sin = cos_sin.chunk(2, dim=-1)       # 拆分 cos/sin 两半
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )