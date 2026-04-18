import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,                      # 模型隐藏层维度 d_model
        num_heads: int,                        # Query 头总数（所有 TP rank 聚合后）
        num_kv_heads: int,                     # KV 头总数 (GQA: num_kv_heads <= num_heads)
        max_position: int = 4096 * 32,         # RoPE 支持的最大位置索引
        head_dim: int | None = None,           # 单个注意力头维度；默认 hidden_size/num_heads
        rms_norm_eps: float = 1e-06,           # Q/K RMSNorm 的 epsilon
        qkv_bias: bool = False,                # QKV 投影是否带 bias（Qwen2 为 True, Qwen3 为 False）
        rope_theta: float = 10000,             # RoPE 基频 base
        rope_scaling: tuple | None = None,     # RoPE 缩放策略（如 YaRN / NTK）
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()                              # 张量并行大小
        self.total_num_heads = num_heads                             # 分片前 Q 头总数
        assert self.total_num_heads % tp_size == 0                   # 必须能整除 TP 大小
        self.num_heads = self.total_num_heads // tp_size             # 本 rank 持有的 Q 头数
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size       # 本 rank 持有的 KV 头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim                 # 本 rank Q 张量通道数
        self.kv_size = self.num_kv_heads * self.head_dim             # 本 rank K/V 张量通道数
        self.scaling = self.head_dim ** -0.5                         # 注意力缩放因子 1/sqrt(d_k)
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(           # 列并行：把 Q/K/V 三个投影合并为一个 GEMM
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(             # 行并行：输出投影，内部 all-reduce 跨 TP 聚合
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(                  # 构造 RoPE 位置编码模块
            self.head_dim,
            rotary_dim=self.head_dim,                # 参与旋转的维度（此处全部参与）
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(                       # FlashAttention + KV cache 包装层
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,                     # [num_tokens] 每个 token 的绝对位置
        hidden_states: torch.Tensor,                 # [num_tokens, hidden_size] 融合 batch 后的扁平张量
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)                                      # 一次 GEMM 得到 Q|K|V 拼接
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 按通道切三份
        q = q.view(-1, self.num_heads, self.head_dim)                           # [N, Hq, D]
        k = k.view(-1, self.num_kv_heads, self.head_dim)                        # [N, Hkv, D]
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,                            # 输入/输出维度 d_model
        intermediate_size: int,                      # FFN 中间维度（SwiGLU 两分支各占一份）
        hidden_act: str,                             # 激活函数名；Qwen3 固定使用 silu
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(   # 把 gate_proj 与 up_proj 合并为一次列并行 GEMM
            hidden_size,
            [intermediate_size] * 2,                      # 两个输出分支尺寸：[gate, up]
            bias=False,
        )
        self.down_proj = RowParallelLinear(               # 行并行下投影；内部 all-reduce
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()                        # SwiGLU: silu(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,                       # Q 头总数
            num_kv_heads=config.num_key_value_heads,                    # KV 头总数（GQA）
            max_position=config.max_position_embeddings,                # 训练支持的最大序列长度
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),           # Qwen2 默认 True, Qwen3 配置里为 False
            head_dim=getattr(config, 'head_dim', None),                 # Qwen3 显式给出；Qwen2 由 hidden/num_heads 推导
            rope_theta=getattr(config, "rope_theta", 1000000),          # Qwen3 通常用 1e6 以支持长上下文
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,                # 残差流；首层为 None, 后续层传入上一层残差
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states   # 首层：仅 norm
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)        # 融合 add+RMSNorm
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)   # attn 后再次融合
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual                                                     # residual 延迟到下一层相加


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)   # 词表按 TP 切分
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)                    # 最终层归一化

    def forward(
        self,
        input_ids: torch.Tensor,                 # [num_tokens] 扁平 token id
        positions: torch.Tensor,                 # [num_tokens] 每 token 的位置
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)                                # 末尾 add+norm 合并
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {                    # HF 权重名 → (合并模块名, 分片标识), loader 据此合并 QKV/gate+up
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)   # 输出头按词表维度 TP 切分
        if config.tie_word_embeddings:                                         # 小模型常见：共享 embedding 与 lm_head
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
