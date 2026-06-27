import torch.nn as nn
import torch
import torch.distributions as dist

class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

# 最简单的Linear层
class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 直接拷贝，不切分
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)    

# 【列并行】沿输出维度切分：W_full -> [W_0, W_1, ..., W_{tp-1}]
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        # output_size // tp_size: 每张 GPU 只存 1/tp_size 的输出维度
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        full_data_output_size = loaded_weights.size(0)
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0)
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# 【合并列并行】将多个矩阵合并为一个，减少 kernel launch
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int],  # e.g. [intermediate_size, intermediate_size] 对应 gate 和 up
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        # 总输出维度 = 各矩阵输出维度之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # 计算在已切分矩阵中的偏移位置
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整权重中切分出对应 TP rank 的分片
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    """
    QKV 的切分策略与其他列并行不同：
    不是按输出维度均分，而是按 head 粒度分配完整 head。

    例如 32 Q heads + 8 KV heads, 4 GPUs:
        GPU 0: Q heads 0-7, K heads 0-1, V heads 0-1
        GPU 1: Q heads 8-15, K heads 2-3, V heads 2-3
        ...
    
    这样每个 GPU 持有完整的 head（head_dim 不切分），
    Attention 计算可以完全在本 GPU 内完成，无需通信。
    """
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        # 每个 GPU 分配的 head 数（总 head 数 / TP 数）
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        """
        按 head 粒度加载 Q、K、V 权重。
        
        QKV 在合并矩阵中的布局（以 per-GPU 视角）：
            [Q_heads_for_this_gpu | K_heads_for_this_gpu | V_heads_for_this_gpu]
        
        每个 head 占 head_size 个输出维度，
        tp_rank 决定加载哪些 head。
        """
        param_data = param.data
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        
        # 计算各分量在 per-GPU 矩阵中的位置
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            # K 紧跟在 Q 之后
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            # V 紧跟在 K 之后
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整权重中按 head 粒度切分
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)

        
class RowParallelLinear(LinearBase):
    """
    行并行：每张 GPU 只处理输入的一部分维度。
    
    前向需要 all_reduce(SUM) 来聚合各 GPU 的部分结果。
    
    使用场景：Attention 的 O projection 和 FFN 的 down projection
    这两个都是 ColumnParallel 的配对层，输出必须 Replicated。
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        # input_size // tp_size: 每张 GPU 只处理 1/tp_size 的输入维度
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        full_data_input_size = loaded_weights.size(1)  # 注意沿 dim=1 切分
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        # narrow(dim=1): 沿输入维度切分
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        # 【关键通信】all_reduce(SUM)：各 GPU 的部分结果求和得到完整输出
        # 只有 tp_size > 1 时才需要通信，单卡场景跳过以节省开销
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result