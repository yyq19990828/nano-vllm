import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):  # 词表维度切分的并行 Embedding：每个 rank 只持有词表的一段

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()                          # 当前进程在 TP 组中的序号
        self.tp_size = dist.get_world_size()                    # TP 组大小
        assert num_embeddings % self.tp_size == 0               # 要求词表能被均匀切分
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size  # 本 rank 负责的词数
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # 本分片词表起点
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 本分片词表终点（开区间）
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # 仅存本分片
        self.weight.weight_loader = self.weight_loader          # 挂载自定义权重加载函数，供 loader 识别

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)                         # 本 rank 应接收的行数
        start_idx = self.tp_rank * shard_size                   # 在完整权重中对应的行偏移
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 沿词表维切片
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # 标记落在本分片的 token
            x = mask * (x - self.vocab_start_idx)               # 本分片内转成局部 id；非本片置 0（占位）
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                           # 非本片的查询结果清零
            dist.all_reduce(y)                                  # 各 rank 求和合成完整嵌入
        return y


class ParallelLMHead(VocabParallelEmbedding):                   # LM 输出头：复用词表切分，将隐藏态投影回词表

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias                                         # 本实现不支持 bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1         # 每条序列最后一个 token 的位置
            x = x[last_indices].contiguous()                    # prefill 阶段只对末 token 计算 logits
        logits = F.linear(x, self.weight)                       # 本 rank 得到词表分片对应的 logits
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)                  # 仅 rank0 汇聚所有分片
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # 沿词表维拼回完整 logits
        return logits
