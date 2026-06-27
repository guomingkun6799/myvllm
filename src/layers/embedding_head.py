from turtle import forward

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from utils import get_context

class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding。
    
    切分策略：按 vocab_size 维度切分（不是 hidden_size 维度）。
    
    Forward 的 mask 机制：
        - 只有属于本 GPU 负责范围的 token 才会执行 embedding lookup
        - 其他 token 输出 0，通过 all_reduce 汇总后得到正确结果
    """
    def __init__(self,num_embeddings: int, embedding_dim:int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        self.num_embeddings = num_embeddings
        #填充到tp_size的整数倍
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        #每个GPU只分配num_embeddings_per_partition行
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))
        self.weight.weight_loader = self.weight.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # 算一下要哪个部分
        offset = self.tp_rank * self.num_embeddings_per_partition  # 0 * 2500 = 0
        shard_size = self.num_embeddings_per_partition             # 2500
        
        actual_start = min(offset, self.num_embeddings)    # 实际从哪里开始
        actual_end = min(offset + shard_size, self.num_embeddings)  # 实际到哪里结束
        actual_size = max(0, actual_end - actual_start)    # 实际有几张

        if actual_size > 0:
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        if actual_size < shard_size:
            param_data[actual_size:].zero_()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # 标记哪些token属于本gpu
        # 条件：token_id 在 [tp_rank * partition_size, (tp_rank+1) * partition_size)
        # 且 token_id < num_embeddings（不超过原始词表大小）
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)

        # 【Step 2: 偏移索引】将全局 token ID 映射到本地索引
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)

        # 【Step 3: Embedding Lookup】只在本地词表查询
        output = F.embedding(x, self.weight)

        # 【Step 4: 通信汇总】
        if dist.get_world_size() > 1:
            # 再次 mask：确保不属于本 GPU 的 token 输出为 0
            # （因为索引 0 的 embedding 可能非零）
            output = mask.unsqueeze(1) * output
            # 【all_reduce(SUM)】将各 GPU 的 embedding 汇总
            # 由于只有正确的 GPU 输出非零值，sum 结果即完整 embedding
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        context = get_context(x)

        if context.is_prefill:
            last_token = context.cu_seqlens_q[1:] - 1
            x = x[last_token].contiguous()
        
        logits = torch.nn.functional.linear(x, self.weight, bias=False)

        if self.tp_size > 1:
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, gather_list=all_logits, dst=0)
            if self.tp_rank == 0:
                # 拼接: (batch, padded_vocab_size)
                logits = torch.cat(all_logits, dim=-1)
                # 裁剪到原始 vocab_size
                logits = logits[..., :self.num_embeddings]

        return logits