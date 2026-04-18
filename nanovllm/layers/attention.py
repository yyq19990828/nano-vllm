import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,             # [N, num_kv_heads, head_dim] 连续存放的新 K
    key_stride,          # key 第 0 维 stride (= num_kv_heads*head_dim)
    value_ptr,           # [N, num_kv_heads, head_dim] 连续存放的新 V
    value_stride,        # value 第 0 维 stride
    k_cache_ptr,         # [num_blocks*block_size, num_kv_heads, head_dim] 全局 KV cache
    v_cache_ptr,
    slot_mapping_ptr,    # [N] 每个 token 在全局 cache 中的槽位索引(-1 表示跳过)
    D: tl.constexpr,     # num_kv_heads * head_dim,作为向量宽度一次性搬运
):
    idx = tl.program_id(0)                                 # 每个 program 处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                                  # 预留槽 / 被掩码的 token 不写
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)             # 按 slot 定位到 cache 中的行
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # key/value: [N, num_kv_heads, head_dim];k_cache/v_cache: [total_slots, num_kv_heads, head_dim];slot_mapping: [N]
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1               # 末维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim   # head 间紧凑
    assert k_cache.stride(1) == D and v_cache.stride(1) == D           # cache 每 slot 为 D 个元素
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,         # 本 rank 上的 Q head 数 (TP 切分后)
        head_dim,          # 每个 head 的维度
        scale,             # softmax 缩放,一般 1 / sqrt(head_dim)
        num_kv_heads,      # 本 rank 上的 KV head 数 (GQA 下 < num_heads)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])                 # 由 ModelRunner 在分配 KV cache 后回填
    # q: [num_tokens, num_heads, head_dim];k,v: [num_tokens, num_kv_heads, head_dim]
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()                                        # 线程局部,携带本次 batch 的元数据
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # 将新 token 的 K/V 写入全局 cache
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache 命中:直接用 cache 作 K/V 源,block_table 指引物理块
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,                        # 变长 prefill:用 cu_seqlens 打包多条 seq
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                        softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode: 每条 seq 仅 1 个新 query,q 需要补回长度维
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,  # 从分块 cache 中按 block_table 读取历史 K/V
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
