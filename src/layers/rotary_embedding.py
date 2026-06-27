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