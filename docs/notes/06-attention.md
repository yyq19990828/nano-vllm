# Attention 与 KV Cache

## 一句话总结
`Attention` 层本身是 GPU 计算的薄壳:先用 Triton kernel 把本步新生成的 K/V 按 `slot_mapping` 散落写入分块 KV Cache,再根据 `is_prefill` 分流走 `flash_attn_varlen_func`(prefill,支持变长批 + prefix 命中)或 `flash_attn_with_kvcache`(decode,单 token 从 cache 读历史)。所有跨层共享的元数据(cu_seqlens / slot_mapping / block_tables / context_lens)通过线程局部的 `Context` 单例传入,避免把参数层层透传。

## 核心流程

```
Attention.forward(q, k, v):

1. Triton kernel: store_kvcache()
   - 按 slot_mapping 将 k, v 写入物理 KV Cache
   - slot == -1 的 token 跳过(比如 prefix cache 已命中的部分,不需要重复写)

2. 计算注意力
   - Prefill: flash_attn_varlen_func
     - 变长序列拼接, cu_seqlens 标记边界 (前缀和)
     - 支持 block_table (前缀缓存场景:此时 K/V 不来自当前 batch 而直接从 cache 取)
   - Decode: flash_attn_with_kvcache
     - 每条 seq 只有 1 个 query token, q.unsqueeze(1) 补回长度维
     - 从分块 KV Cache 按 block_table 读取历史 K/V, cache_seqlens 标记每条 seq 的有效长度
```

## 代码锚点
- store_kvcache_kernel: [attention.py:10-30](../../nanovllm/layers/attention.py#L10-L30)
- store_kvcache (host 侧 dispatch + assert): [attention.py:33-40](../../nanovllm/layers/attention.py#L33-L40)
- prefill 分支 (`flash_attn_varlen_func`): [attention.py:64-70](../../nanovllm/layers/attention.py#L64-L70)
- decode 分支 (`flash_attn_with_kvcache`): [attention.py:71-74](../../nanovllm/layers/attention.py#L71-L74)
- Context 读取 (`get_context`): [attention.py:60](../../nanovllm/layers/attention.py#L60)
- Context 定义: [utils/context.py](../../nanovllm/utils/context.py)
- Context 的写入方: [model_runner.py 中 prepare_prefill / prepare_decode](../../nanovllm/engine/model_runner.py)

## 关键设计决策

### 为什么 prefill 和 decode 用不同的 flash-attn API

- **Prefill**:batch 内每条 seq 的 prompt 长度不同,全部 query 被 flatten 拼成 `[sum(seqlens), num_heads, head_dim]`,用 `cu_seqlens_q/k`(前缀和)告诉 kernel 每条 seq 在哪里开始结束。单次调用即可完成整批变长计算,避免 padding 浪费。
- **Decode**:每步只产生 1 个新 token,问题变成"把 1 个 Q 对上整段历史 KV"。`flash_attn_with_kvcache` 针对这个场景做了专门优化(支持 paged KV,query 长度为 1),不需要 cu_seqlens,只需 `cache_seqlens` + `block_table` 指路。
- 两条路径都用 `causal=True`,prefill 是下三角 mask,decode 等价于 query 去看完整历史。

### 为什么用 Triton kernel 写 KV Cache 而不是 `index_put_`

- **散落写(scatter)模式**:`slot_mapping[i] = slot` 表示第 `i` 个新 token 要写入 cache 的第 `slot` 行。`index_put_` 能做,但会引入 PyTorch dispatcher 开销和额外的 kernel 启动。
- **融合读 + 写**:Triton kernel 里 `num_kv_heads * head_dim` 一次 `tl.load` / `tl.store`,一个 program 一个 token,寄存器流水直达。
- **分支处理 `slot == -1`**:kernel 内直接 `return` 跳过,省掉一次 gather+scatter;PyTorch 侧要么 mask 要么分两次写。
- **与 CUDA Graph 兼容**:kernel 启动形状固定 `(N,)`,适合 capture。

### slot_mapping:逻辑位置 → 物理位置

- KV Cache 被切成 `block_size=256` 的物理块,每条 seq 维护一张 `block_table`,记录自己占用了哪些物理块(逻辑到物理块的映射)。
- 对 batch 中第 `i` 个新 token:设它属于某条 seq,在该 seq 内的绝对位置是 `p`,那么
  ```
  block_idx_in_seq = p // block_size
  offset_in_block  = p %  block_size
  physical_block   = block_table[seq][block_idx_in_seq]
  slot_mapping[i]  = physical_block * block_size + offset_in_block
  ```
- `slot_mapping` 即把这套计算**预先在 host 侧做完**并打平成一维,kernel 只需 `slot * D` 定位即可,完全不用再查 block_table 做间接寻址。
- `-1` 的槽位用在 prefix cache 命中的 prompt 前缀:这些 token 的 KV 早已写入 cache,当前步不需要重写。

### Context 作为线程局部单例

- Attention 需要的 prefill/decode 判别、cu_seqlens、block_tables 等是**整个 batch 共享**的元数据,与当前层无关。
- 走函数参数链要穿透 Model → DecoderLayer → Attention 三层,噪音大。改成 `set_context(...)` / `get_context()` 模块级单例后,`ModelRunner` 在每步开头填好,各层 forward 按需 `get_context()` 取,解耦干净。
- 代价是**隐式状态**:CUDA Graph capture 时要特别小心,确保 capture 与 replay 时 context 内容的张量指针不变(model_runner 中通过固定 buffer + `copy_(...)` 处理)。

## 我的理解 / 类比

把 KV Cache 想成一个"停车场",每个 `block_size` 大小的车位组成一排(块),`block_table` 是每辆车(每条 seq)拿到的车位表。`slot_mapping` 就是调度员递给 Triton kernel 的一张便签:"第 i 个新到的乘客停到第 slot 号槽"。`-1` 就是"这个乘客的车早就在里面了,别再挪"。

Prefill 像是一次性把一整车乘客送进场 —— 车上有很多座位,长度各异,用 `cu_seqlens` 当号牌条纹;Decode 则是每步只送一个新乘客,但要让他和历史乘客打个照面(attention),所以 API 专门针对 query_len=1 优化。

## Q&A

### Q: `store_kvcache_kernel` 这套 Triton DSL 怎么读?

**Triton 的心智模型**:写的不是"一个线程做啥",而是"一个 **program** 处理一整块数据",块内元素的并行由编译器向量化。对照 CUDA:

| CUDA | Triton |
|---|---|
| `__global__` kernel | `@triton.jit` 函数 |
| thread block | **program**(`tl.program_id(0)` 取编号) |
| grid | 启动时的 `[(N,)]` |
| `threadIdx + blockDim` 循环 | `tl.arange(0, D)` 生成向量偏移,一次性 load/store |

**参数类型**:
- tensor → 自动退化为 GPU 指针
- Python 标量 → 运行时参数
- `tl.constexpr` → **编译期常量**,不同取值触发重编译(因为 `tl.arange(0, D)` 要在编译期知道向量宽度才能分配寄存器)

**逐行**(以 `store_kvcache_kernel` 为例):
```python
idx  = tl.program_id(0)                        # 本 program 的编号 ∈ [0, N)
slot = tl.load(slot_mapping_ptr + idx)         # 标量 load:目标槽位
if slot == -1: return                          # 哨兵:跳过不需要写的 token
key_offsets = idx * key_stride + tl.arange(0, D)   # 向量偏移 [idx*s, idx*s+1, ..., idx*s+D-1]
key = tl.load(key_ptr + key_offsets)           # 向量 load:一次把 D 个元素读进寄存器
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)     # 向量 store
```

**关键 trick**:`tl.load(ptr + 向量偏移)` 一次搬运整段连续内存,编译器自动生成合并访存(coalesced access)。每个 program 只有 1 次向量 load + 1 次向量 store,没有循环、没有同步、没有 shared memory,是搬运类算子的最小实现。

**对比等价 CUDA**:
```cuda
int idx = blockIdx.x;
int slot = slot_mapping[idx];
if (slot == -1) return;
for (int i = threadIdx.x; i < D; i += blockDim.x)    // 这层循环 + 合并访存 + 向量化
    k_cache[slot*D + i] = key[idx*key_stride + i];   // 都要自己手写
```
Triton 把"线程内循环 + 向量化 + 合并访存"交给编译器,你只表达"对长度 D 的向量做一次 load/store"。

### Q: `k_cache.numel() and v_cache.numel()` 这个守卫在防什么?

`self.k_cache` 在 `__init__` 里是 `torch.tensor([])`(空 tensor,`numel()==0`),只有 `ModelRunner` 分配好真正的 KV cache 显存后才会回填进来。所以这个判断 = "cache 是否已经分配"。在 warmup / dummy run / profile 阶段 cache 还没分配,直接跳过写入即可。

### Q: Qwen3 对 Q、K 做了 RMSNorm（QK Norm），为什么 V 不需要？

参考代码：[qwen3.py:68-69](../../nanovllm/models/qwen3.py#L68-L69)、[qwen3.py:82-83](../../nanovllm/models/qwen3.py#L82-L83)

**Q、K 需要 RMSNorm 的原因：** Q·K 点积的量级 = Q 的模 × K 的模 × cos(θ)。训练过程中如果某些 head 的 Q 或 K 模长失控增长，点积就会爆炸 → softmax 饱和 → 梯度消失 → 该 head "坏死"。RMSNorm 把 Q、K 各自的模长归一化到稳定范围，让点积只反映**方向相似度**，不受模长干扰。这在深层网络、大 head_dim、高学习率场景下尤其关键（论文：*Scaling ViTs to 22B*）。

**V 不需要的原因：**
1. **V 不参与点积**。V 只是被 softmax 权重线性加权求和，不存在"两个向量相乘导致量级爆炸"的问题。
2. **V 的输出已有 RMSNorm 兜底**。Attention 输出 = softmax(QK/√d)·V，随后经过 `post_attention_layernorm`（RMSNorm）再进 FFN。即使 V 的模长偏大，下游的 RMSNorm 会校正。
3. **对 V 做 Norm 反而有害**。归一化会压缩 V 的表达容量——V 本该自由编码"该传递什么信息"，强行归一化等于丢掉幅度信息，实验上掉点。

一句话：**QK Norm 治的是点积爆炸，V 不做点积、下游已有 Norm，归一化它既无必要又损表达力。**

## 遗留问题
- [x] `flash_attn_with_kvcache` 对 `num_kv_heads < num_heads`(GQA)是如何复制/广播 K/V 的?是否需要我们在上游做 repeat?
  - **不需要 repeat**。flash-attn 两个 API 都原生支持 `num_heads_q != num_heads_k`(要求整除),kernel 内部按 group 隐式广播:Q head `h` 自动对上 KV head `h // (num_heads_q // num_heads_k)`,**在寄存器里复用**,不做物化展开。
  - 若在上游 `repeat_interleave` 到 `num_heads` 份,KV cache 就膨胀 `num_heads/num_kv_heads` 倍,GQA 省显存的意义就没了;即使只在算时 repeat 也多一次显存往返,而 flash-attn 把这步融合在片上。
  - 代码上 `Attention` 保留 `num_heads` 和 `num_kv_heads` 两个独立字段([attention.py:46-57](../../nanovllm/layers/attention.py#L46-L57)),forward 原样喂 `k,v: [N, num_kv_heads, head_dim]` 给 flash-attn,全程不 repeat。TP 切分时两者同时按 rank 切,需满足 `num_kv_heads % tp_size == 0` 才能对齐。
- [ ] 当同一 batch 中有 seq 命中 prefix cache、有 seq 未命中,prefill 分支里 `k, v = k_cache, v_cache` 是否会把未命中 seq 的新 K/V 也强制走 cache 路径?(猜测:未命中 seq 的 slot 已在上一步 `store_kvcache` 写入 cache,所以读 cache 仍然正确,block_table 统一指路。)
- [ ] Triton kernel 里 `D` 作 `tl.constexpr` 意味着每换一个模型(head_dim 变)就要重新编译,编译缓存由谁管理?
