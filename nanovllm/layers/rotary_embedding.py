from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,       # [..., head_dim]      待旋转的 Q 或 K
    cos: torch.Tensor,     # [..., head_dim / 2]  cos(mθ_i),不重复到 head_dim
    sin: torch.Tensor,     # [..., head_dim / 2]  sin(mθ_i)
) -> torch.Tensor:
    # "half-rotated" 排布:第 i 维与第 i+d/2 维配对共享 θ_i;与原论文相邻配对 (q_{2i},q_{2i+1}) 数学等价,但必须与 HF/Qwen 权重约定一致。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)     # 各 [..., head_dim / 2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype) # [..., head_dim]


class RotaryEmbedding(nn.Module):
    # 预计算所有位置的 cos/sin,推理时按 position 直接索引。
    # head_size=每头维度;rotary_dim=参与旋转的维度(此处强制==head_size);
    # max_position_embeddings=最大位置;base=θ 底数(常用 1e4,长上下文 1e6)。
    # 频率: θ_i = 1 / base^(2i / rotary_dim),i = 0 .. rotary_dim/2 - 1

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  # [rotary_dim/2]  每子空间 θ_i
        t = torch.arange(max_position_embeddings, dtype=torch.float)                               # [max_pos]       位置索引 m
        freqs = torch.einsum("i,j -> ij", t, inv_freq)                                             # [max_pos, rotary_dim/2]  外积 mθ_i
        cos = freqs.cos()
        sin = freqs.sin()
        # cache: [max_pos, 1, rotary_dim];末维前半 cos / 后半 sin(省一半存储,不重复到完整 head_dim);中间 1 维用于对 num_heads 广播。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,   # [num_tokens]                          每个 token 的绝对位置
        query: torch.Tensor,       # [num_tokens, num_heads,    head_dim]
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]                           # [num_tokens, 1, rotary_dim];维度 1 广播到 head 数
        cos, sin = cos_sin.chunk(2, dim=-1)                               # 各 [num_tokens, 1, rotary_dim/2]
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_rope_cache: RotaryEmbedding | None = None


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    global _rope_cache
    if _rope_cache is not None:
        return _rope_cache
    assert rope_scaling is None or rope_scaling.get("rope_type") in (None, "default"), \
        f"Unsupported rope_scaling type: {rope_scaling}"
    _rope_cache = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _rope_cache
