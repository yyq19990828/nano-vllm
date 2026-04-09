# 模型执行与 CUDA Graph

## 一句话总结

ModelRunner 是 Engine 与 GPU 之间的翻译层：将 Scheduler 输出的 Sequence 列表转换为 GPU tensor，
通过 Eager 或 CUDA Graph 执行模型前向，最后用 Gumbel-max 采样得到下一个 token。

## 整体架构

```
LLMEngine.step()
    │
    ▼
ModelRunner.run(seqs, is_prefill)
    │
    ├─ 1. 数据准备 ─────────────────────────────────────────────┐
    │   ├─ prepare_prefill(seqs)  → input_ids, positions        │
    │   │   + set_context(cu_seqlens_q/k, slot_mapping, ...)    │ Sequence → GPU tensor
    │   └─ prepare_decode(seqs)   → input_ids, positions        │
    │       + set_context(slot_mapping, context_lens, ...)      │
    │                                                           │
    ├─ 2. 模型前向 ─────────────────────────────────────────────┤
    │   └─ run_model(input_ids, positions, is_prefill)          │
    │       ├─ Eager:  model(input_ids, positions) 直接执行      │
    │       └─ Graph:  拷贝输入到 buffer → graph.replay()       │
    │                                                           │
    ├─ 3. 采样 (仅 rank0) ──────────────────────────────────────┤
    │   └─ sampler(logits, temperatures) → token_ids            │
    │                                                           │
    └─ 4. reset_context() 清理 ─────────────────────────────────┘
```

## 核心流程详解

### 1. 数据准备: Sequence → GPU Tensor

#### Prefill 阶段 — prepare_prefill() (model_runner.py:139)

多个序列的 token 拼接为一个扁平 tensor，用 `cu_seqlens` 标记边界（flash-attention 的 varlen 接口）：

```
序列 A: [t0, t1, t2, t3]  (4 tokens, 其中 0 个缓存)
序列 B: [t0, t1, t2, t3, t4, t5]  (6 tokens, 其中 4 个前缀缓存命中)

input_ids  = [t0, t1, t2, t3, t4, t5]   ← A 全部 + B 未缓存的 2 个
positions  = [0, 1, 2, 3, 4, 5]

cu_seqlens_q = [0, 4, 6]     ← Q 长度: A=4, B=2 (只算未缓存的)
cu_seqlens_k = [0, 4, 10]    ← K 长度: A=4, B=6 (attend 到全部上下文, 含缓存)
```

**Q 和 K 长度不同的原因**：前缀缓存命中时，KV cache 中已有历史 K/V，无需重新计算，
但 attention 仍需 attend 到这些历史位置，所以 `seqlen_k = 全部长度`，`seqlen_q = 未缓存长度`。

**slot_mapping**：标记每个新 token 的 KV 应写入 GPU KV cache 的哪个物理 slot。
只映射从 `num_cached_blocks` 到 `num_blocks` 的块（跳过已缓存的）：

```
slot = block_table[i] * block_size + offset_in_block
```

**何时需要 block_tables**：只有 `cu_seqlens_k > cu_seqlens_q`（有前缀缓存）时才需要，
因为 attention 需要通过 block_table 索引读取缓存的 KV。无缓存时全部 KV 直接在当前 forward 中产生。

#### Decode 阶段 — prepare_decode() (model_runner.py:178)

每个序列只输入 1 个 token（刚生成的 `last_token`），但需 attend 到全部历史 KV：

```
序列 A: 长度 100 → input_ids=[last_token], positions=[99], context_lens=[100]
序列 B: 长度 200 → input_ids=[last_token], positions=[199], context_lens=[200]

slot_mapping = [block_table[-1] * block_size + last_block_num_tokens - 1]
               ↑ 新 token 写入 KV cache 的精确物理位置
```

Decode 总是需要 `block_tables`（不像 prefill 可以跳过），因为所有历史 KV 都在 cache 中。

### 2. Context 传递机制 (context.py)

ModelRunner 通过全局 `Context` 对象将 attention 参数传递给 Attention 层，
避免侵入模型前向签名（`model(input_ids, positions)` 保持简洁）：

```python
@dataclass
class Context:
    is_prefill: bool          # 区分 prefill/decode, 决定用哪种 flash-attention 接口
    cu_seqlens_q/k: Tensor    # prefill: flash_attn_varlen_func 的序列边界
    max_seqlen_q/k: int       # prefill: 最长序列长度
    slot_mapping: Tensor      # 新 KV 写入位置
    context_lens: Tensor      # decode: 每个序列的上下文长度
    block_tables: Tensor      # KV cache 物理块索引表
```

生命周期：`set_context()` → 模型前向 → `reset_context()`，每次 `run()` 调用一个完整周期。

### 3. 模型前向: Eager vs CUDA Graph (model_runner.py:205)

```
                     run_model(input_ids, positions, is_prefill)
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
            is_prefill?      enforce_eager?       bs > 512?
                 │                  │                  │
            YES──┤──────── YES ────┤──────── YES ────┤
                 ▼                                    ▼
           Eager 模式                           Eager 模式
       model(input_ids, positions)          (超出预录制范围)
                 │
            全部 NO
                 ▼
          CUDA Graph 模式
       1. 选择 >= bs 的最小档位
       2. 拷贝输入到固定 buffer
       3. graph.replay()
       4. 从 outputs buffer 取结果
```

**Eager 模式**：标准 PyTorch 前向，每次动态构建计算图。Prefill 必须用 eager，
因为每次输入形状不同（序列数和长度都变化），无法预录制。

**CUDA Graph 模式**：将完整的 kernel 调用序列录制为一个图，replay 时跳过 Python 调度。
decode 阶段每次只处理 1 token/seq，计算量小，Python 调度开销占比高，所以 Graph 收益大。

### 4. CUDA Graph 录制 (model_runner.py:235)

初始化时一次性录制多个 batch size 的 graph：

```
graph_bs = [1, 2, 4, 8, 16, 32, ..., max_bs]
                                         │
        从大到小录制 ◄───────────────────┘

对每个 bs:
  1. set_context(decode 模式, buffer[:bs] 切片)
  2. warmup: outputs[:bs] = model(input_ids[:bs], ...)     ← 触发 kernel 编译
  3. capture: with torch.cuda.graph(graph, pool):
                 outputs[:bs] = model(input_ids[:bs], ...)  ← 录制 kernel 序列
  4. 保存 graph, 共享 pool
```

**关键设计**：
- **所有 graph 共享固定 buffer**：`input_ids`, `positions`, `slot_mapping` 等是预分配的最大尺寸 tensor，
  不同 bs 通过 `[:bs]` 切片使用同一块内存。replay 时只需修改 buffer 内容，地址不变。
- **共享 graph_pool**：所有 graph 的内部临时 tensor 从同一个 memory pool 分配，减少显存碎片。
- **从大到小录制**：最大 bs 先占用 pool，后续小 bs 复用已有空间。

### 5. 采样: Gumbel-Max (sampler.py)

```python
@torch.compile  # ← Triton 编译, 融合为单个 kernel
def forward(logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(1))   # 温度缩放
    probs = torch.softmax(logits, dim=-1)
    # Gumbel-max trick: argmax(log(probs) + gumbel_noise) = argmax(probs / exp_noise)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

**Gumbel-max trick** 等价于按概率分布采样，但可以用 argmax 实现，对 GPU 友好（无需 `torch.multinomial` 的排序）。
`temperature=0` 时 softmax 输出接近 one-hot，效果等价于 greedy decoding。

**仅 rank0 采样**：多 GPU 时，所有 rank 执行相同的模型前向（NCCL allreduce 保证一致），
但采样只需做一次，所以只有 rank0 调用 sampler，worker 返回 None。

### 6. 多 GPU 通信: SharedMemory + Event

```
                    Rank 0 (主进程)                     Rank 1+ (Worker)
                         │                                   │
   LLMEngine.step()      │                                   │
        │                 │                                   │
   ModelRunner.call()     │                                   │
        │                 │                                   │
   write_shm() ──────────┼─── SharedMemory (1MB) ──────────► read_shm()
   event.set() ──────────┼─── Event ──────────────────────► event.wait()
        │                 │                                   │
   method(*args) ◄────── NCCL allreduce 同步 ────────► method(*args)
        │                 │                                   │
   return token_ids       │                              return None
```

**SharedMemory 协议**：`[4字节长度][pickle(方法名, *参数)]`

**为什么不直接用 NCCL 广播？** SharedMemory 用于传递 Python 对象（方法名、Sequence 列表等），
这些无法直接通过 NCCL 发送（NCCL 只支持 GPU tensor）。SharedMemory + Event 是轻量级的进程间 Python 对象传递方案。

**Worker 生命周期**：`__init__` 末尾进入 `loop()` 阻塞，等待 rank0 分发指令（run/exit 等）。
收到 `exit` 时跳出循环，清理资源。

## 代码锚点

| 功能 | 位置 |
|------|------|
| ModelRunner 类定义 | model_runner.py:15 |
| `__init__` 初始化 6 阶段 | model_runner.py:17-55 |
| warmup_model | model_runner.py:104-112 |
| allocate_kv_cache | model_runner.py:114-132 |
| prepare_prefill | model_runner.py:139-176 |
| prepare_decode | model_runner.py:178-194 |
| run_model (Eager/Graph 分支) | model_runner.py:205-222 |
| run (统一入口) | model_runner.py:224-231 |
| capture_cudagraph (Graph 录制) | model_runner.py:234-268 |
| Graph replay | model_runner.py:211-221 |
| Context 定义 | context.py:6-14 |
| set_context / reset_context | context.py:21-27 |
| Sampler (Gumbel-max) | sampler.py:5-15 |
| SharedMemory 通信 | model_runner.py:68-89 |

## 关键设计决策

### 为什么 Prefill 不用 CUDA Graph？

CUDA Graph 要求**每次 replay 的 tensor 形状完全一致**（地址固定）。
Prefill 阶段每次输入的序列数和长度都不同（`cu_seqlens` 变长），无法预录制。

理论上可以为每种输入形状都录制一个 graph，但组合爆炸不现实。
vLLM 也是同样的做法：prefill = eager, decode = graph。

### CUDA Graph 的 batch size 选择策略

```python
graph_bs = [1, 2, 4, 8] + list(range(16, max_bs+1, 16))
```

- **小 bs (1-8)**：用 2 的幂次，覆盖低并发场景，粒度细避免过多 padding 浪费
- **大 bs (16-512)**：每隔 16 一个档位，与 GPU warp size (32) 对齐，平衡录制数量和 padding 开销
- 实际 bs=5 时选择 graph_bs=8，多出的 3 个位置用无效值填充（slot_mapping=-1, context_lens=0）

### Padding 到固定 batch size 的性能影响

额外的 padding 序列会参与 GEMM 计算（浪费算力），但：
- Decode 阶段每序列只有 1 个 token，GEMM 本身很小，padding 几个影响极小
- CUDA Graph replay 省掉的 Python 调度开销（~1ms/step）远大于 padding 带来的多余计算
- 16 的间隔意味着最多浪费 15/16 ≈ 6% 的计算量（大 bs 时）

### 为什么采样用 Gumbel-max 而不是 multinomial？

`torch.multinomial` 内部需要计算 CDF 和二分查找，无法被 `torch.compile` 有效优化。
Gumbel-max trick 只用 `exponential_()` + `div_()` + `argmax()`，全是逐元素操作 + 规约，
`@torch.compile` 可以将整个 forward 融合为单个 Triton kernel。

## Q&A

### Q1: SharedMemory 里面存了什么？为什么分配 1MB？

**存的内容**：pickle 序列化的 `[方法名, *参数]`，协议格式为 `[4字节长度][pickle数据]`。
最常见的调用是 `call("run", seqs, True)`，即传递 Sequence 对象列表。

**Sequence 的序列化优化**（sequence.py:74-76）：
```python
def __getstate__(self):
    return (num_tokens, num_prompt_tokens, num_cached_tokens, block_table,
            token_ids if num_completion_tokens == 0 else last_token)
```
- Prefill 时传完整 `token_ids`（需要让 worker 知道输入内容）
- Decode 时只传 `last_token`（1 个 int），体积从 KB 级降到字节级

**1MB 的估算**：
- Decode 最坏：512 序列 × ~40 字节 ≈ 20KB
- Prefill 最坏：少量序列 × 4096 tokens × 8 字节 ≈ 32KB/seq
- 1MB 绑绑有余，且只占 `/dev/shm`（tmpfs 内存），不占 GPU 显存
- 溢出时 `shm.buf[4:n+4] = data` 会越界报错，无显式检查

### Q2: num_kv_heads 为什么要除以 world_size？

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size  # model_runner.py:127
```

这是 **GQA (Grouped-Query Attention) + Tensor Parallelism** 的配合。
每个 GPU 只负责一部分 KV head，所以分配 KV cache 时按本 rank 实际持有的 head 数计算：

```
Qwen2-7B: num_kv_heads=4, TP=2
  → rank0 持有 head 0,1  → 分配 2 个 head 的 KV cache
  → rank1 持有 head 2,3  → 分配 2 个 head 的 KV cache
```

如果不除，每个 rank 会分配全量 KV cache，浪费一半显存。

### Q3: pin_memory + non_blocking 的作用？只设 pin_memory 不设 non_blocking 呢？

```python
torch.tensor(..., pin_memory=True).cuda(non_blocking=True)  # model_runner.py 中大量使用
```

**pin_memory**：将 tensor 分配在锁页内存（物理地址固定，不会被 OS 换出），
GPU DMA 引擎可以直接访问，省掉一次到临时 buffer 的拷贝。

**non_blocking**：`.cuda()` 提交传输请求后 CPU 立即返回，不等 GPU 搬完。

```
无 pin_memory:
  普通内存 → [拷贝到临时 pinned buffer] → DMA 传到 GPU    (两次拷贝, CPU 阻塞)

有 pin_memory, 无 non_blocking:
  pinned 内存 → DMA 传到 GPU                              (一次拷贝, CPU 仍阻塞)

有 pin_memory + non_blocking:
  pinned 内存 → DMA 传到 GPU                              (一次拷贝, CPU 不阻塞)
```

**只设 pin_memory 不设 non_blocking**：传输速度更快（省一次 memcpy），但 CPU 仍逐个等待每个 `.cuda()` 完成。加上 `non_blocking` 后，`prepare_prefill/decode` 中连续创建的多个 tensor（input_ids、positions、slot_mapping 等）可以流水线式传输，总等待时间从串行之和变为近似并行。

**安全性**：异步传输期间如果 CPU 修改了 pinned buffer 会数据竞争，但这里每个 tensor 创建后不再修改，所以安全。

### Q4: SharedMemory 和 Event 是如何搭配工作的？

两者分工明确：**SharedMemory 传内容，Event 传信号**。

```
SharedMemory  → 数据载体: 存 pickle 序列化的 [方法名, *参数]
Event         → 同步通知: 告诉 worker "数据准备好了"
```

为什么需要 Event？SharedMemory 本身没有"新数据到达"的通知机制，worker 不知道何时去读，
busy-loop 会浪费 CPU。Event 提供阻塞等待能力。

**完整流程**：

```
Rank 0                                    Worker
──────                                    ──────
                                          event.wait()  ← 阻塞睡眠
write_shm:
  shm[0:4] = 长度
  shm[4:n+4] = pickle数据
  for event: event.set() ──────────────►  被唤醒
                                          read_shm:
                                            读 shm
                                            event.clear()  ← 重置等下次
method(*args) ◄──── NCCL 同步 ────►       method(*args)
```

**关键设计**：
- **每个 worker 一个 Event**（rank0 持有 `list[Event]`）：避免多 worker 共享一个 Event 时
  谁先 `clear()` 会让其他 worker 错过信号
- **手动 set/clear**：Python `Event` 不支持 auto-reset，必须读完后手动 `clear()`
- **隐式同步**：rank0 不知道 worker 是否读完，依赖 NCCL allreduce 强制所有 rank 在下次操作前同步

**类比**：留言板（shm）+ 门铃（event）。rank0 写完留言后按门铃，worker 听到铃声跑来看板子，
看完回屋等下一次铃声。

## 遗留问题

- [ ] CUDA Graph 录制时 `from_pool` 共享 memory pool 的具体行为？显存是复用还是叠加？
- [ ] `pin_memory=True` + `.cuda(non_blocking=True)` 的异步传输, 有没有可能与 kernel 执行产生竞争？
- [ ] 多 GPU 时 SharedMemory 大小固定 1MB, Sequence 列表很大时会溢出吗？
- [ ] temperature=0 时 Gumbel-max 的数值稳定性？softmax 输出接近 one-hot 是否足够准确？
