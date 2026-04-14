# 张量并行

## 一句话总结
nano-vllm 用 "rank0 单进程调度 + 多 worker 进程镜像执行" 的架构把 Qwen3 的权重按 Megatron-LM 风格切到多张 GPU 上: QKV/Gate/Up 按输出维度(ColumnParallel)切, O_proj/Down 按输入维度(RowParallel)切, 只在每个 TransformerBlock 的 Attention 和 MLP 出口各做一次 AllReduce, 通信量仅为 2 × num_layers × hidden × tokens。

## 核心流程

```
进程拓扑:
  LLMEngine (主进程 = rank0)
    ├── 拥有 Scheduler / BlockManager / Tokenizer
    ├── 持有 model_runner (rank0, 持 list[Event])
    └── spawn 出 rank 1..N-1 worker 进程 (各持 1 个 Event)
          └── 每个 worker 直接在 __init__ 里进入 self.loop() 永不返回

一次 step() 的指令广播 (rank0 视角):
  1. Scheduler 选出 seqs → model_runner.call("run", seqs, is_prefill)
  2. rank0 的 call():
       a) world_size>1 时, write_shm("run", seqs, is_prefill)
          - pickle.dumps([method_name, *args]) → [4B长度 | payload] 写入 SharedMemory
          - 遍历 list[Event], 对每个 worker event.set()
       b) 自己也调用 self.run(seqs, is_prefill)
  3. 各 worker 在 read_shm() 阻塞: event.wait() → 读出 4B 长度 → 反序列化 → event.clear()
  4. 所有 rank 进入同一个 self.run(...), NCCL 集合通信天然同步

权重切分 (Qwen3 TransformerBlock 内):
  Attention:
    qkv_proj   = QKVParallelLinear     → Column 切 (每 rank 持 num_heads/TP 个 Q、num_kv/TP 个 K/V)
    o_proj     = RowParallelLinear     → Row 切    (forward 内 AllReduce 聚合)
  MLP:
    gate_up_proj = MergedColumnParallelLinear → Column 切 (gate 和 up 拼在同一个 weight 里)
    down_proj    = RowParallelLinear          → Row 切    (forward 内 AllReduce)

  输入/输出层:
    Embedding (VocabParallelEmbedding) → 按 vocab 维切, forward 内 AllReduce 聚合
    LMHead   (ParallelLMHead)         → 按 vocab 维切, forward 内 **dist.gather 到 rank0** (非 AllGather)
    RMSNorm / RoPE                     → 各 rank 完整复制 (参数小, 不值得切)
    KV cache                           → 按 num_kv_heads // TP 切 (见 allocate_kv_cache)

一个 Block 的前向只产生 2 次 AllReduce: 一次在 Attention 出口, 一次在 MLP 出口。
整个模型末尾还有 1 次 gather(非 AllReduce), 把 vocab 维的 shard 单向收到 rank0。
```

## 代码锚点
- 多进程 spawn + Event 数组: [llm_engine.py:21-30](../../nanovllm/engine/llm_engine.py#L21-L30)
- ModelRunner 持有 rank / world_size / event: [model_runner.py:19-27](../../nanovllm/engine/model_runner.py#L19-L27)
- NCCL 初始化: [model_runner.py:29](../../nanovllm/engine/model_runner.py#L29)
- Worker 阶段: rank0 建 SharedMemory / 其它 rank 连接并进入 loop: [model_runner.py:54-61](../../nanovllm/engine/model_runner.py#L54-L61)
- 事件循环: [model_runner.py:75-90](../../nanovllm/engine/model_runner.py#L75-L90)
- write_shm/read_shm 二进制协议: [model_runner.py:83-100](../../nanovllm/engine/model_runner.py#L83-L100)
- call() 统一广播入口: [model_runner.py:102-107](../../nanovllm/engine/model_runner.py#L102-L107)
- KV head 按 TP 切分: [model_runner.py:128](../../nanovllm/engine/model_runner.py#L128)
- 采样只在 rank0 上做: [model_runner.py:252-254](../../nanovllm/engine/model_runner.py#L252-L254)
- ColumnParallelLinear (按输出维切): [linear.py:54-73](../../nanovllm/layers/linear.py#L54-L73)
- MergedColumnParallelLinear (gate+up 拼一起): [linear.py:76-93](../../nanovllm/layers/linear.py#L76-L93)
- QKVParallelLinear (按 q/k/v 分 shard 加载): [linear.py:96-128](../../nanovllm/layers/linear.py#L96-L128)
- RowParallelLinear + forward 内 AllReduce: [linear.py:131-153](../../nanovllm/layers/linear.py#L131-L153)
- packed_modules_mapping (HF 原始权重名 → 切分后模块名): [qwen3.py:186-192](../../nanovllm/models/qwen3.py#L186-L192)
- VocabParallelEmbedding (按 vocab 切 + AllReduce): [embed_head.py:9-42](../../nanovllm/layers/embed_head.py#L9-L42)
- ParallelLMHead (按 vocab 切 + gather 到 rank0): [embed_head.py:45-66](../../nanovllm/layers/embed_head.py#L45-L66)

## 关键设计决策

### 为什么用 SharedMemory + Event 而不是 torch.distributed 广播指令?

NCCL `broadcast` 只能传 GPU tensor, 把 Python 对象(Sequence 列表、方法名、bool 标志)塞进 tensor 需要人工序列化+填充+对齐, 开销和复杂度都远高于 `pickle.dumps` 写 1MB 共享内存。而且调度指令属于 "控制面" 数据, 量很小、频率高, 走 CPU 侧的 SharedMemory 不占 NCCL 带宽, 也不会和张量并行的集合通信互相阻塞。Event 提供 CPU 侧的 "就绪信号", 和 shm 配合恰好构成了一条 "主发 → 从收" 的轻量 RPC 通道。

**分工清晰**: NCCL 只负责张量层面的 AllReduce(数据面), SharedMemory+Event 负责 Python 对象级的方法调用(控制面)。

### 为什么 QKV 用 Column 切, O_proj 用 Row 切?

这是 Megatron-LM 的经典推导, 目的是让两次切分 **自然对齐**, 从而把整个 Attention 块的通信压缩到 "1 次 AllReduce":

```
x (hidden)  → QKV(ColumnParallel)     → 每 rank 得到 [num_heads/TP, d] 的 Q/K/V  (无通信)
            → FlashAttention 本地算    → 每 rank 得到 [num_heads/TP, d] 的 out   (无通信)
            → reshape 为 [hidden/TP] 行向量 → 正好是 O_proj 的 Row 切的 "输入分片"
            → O_proj(RowParallel) forward 内 AllReduce → 得到完整 hidden
```

如果反过来 QKV 用 Row、O_proj 用 Column, 那么 QKV 之前就要 AllReduce 一次, attention 本地算完后还要 AllGather 头, O_proj 之后再做一次聚合 —— 通信次数翻 3 倍。

MLP 侧同理: `gate_up_proj` 按 Column 切 → SiLU × 按元素逐位做, 天然对齐 → `down_proj` 按 Row 切 → AllReduce。

### 为什么 gate 和 up 合并成一个 MergedColumnParallelLinear?

Qwen3 的 `gate_proj(x) * silu(up_proj(x))` 原本是两个独立 Linear, 合并后:
1. 只发起一个 GEMM kernel, 利用 cuBLAS 在大矩阵上的更高吞吐;
2. 共享同一份输入 x 的加载, 减少一次内存读;
3. 加载权重时通过 `loaded_shard_id` (0 或 1) 分别写入同一个 param 的前一半/后一半, 对外仍然能吃 HF 原始的 `gate_proj.weight` / `up_proj.weight`。

`packed_modules_mapping` 正是为这种 "训练时分开、推理时合并" 的映射服务的: 把原始权重名翻译成 `(合并后模块名, shard_id)`, 由 `load_model` 在加载阶段完成拆-合。

### 为什么 KV cache 也要按 TP 切?

`num_kv_heads // self.world_size` —— 每个 rank 只需要存自己那一份 KV head 对应的 cache。既然 QKV 的 K/V 输出已经在 ColumnParallel 切分下只有 `num_kv_heads/TP` 个头, 对应的 cache 也只需要这么多, 显存占用直接 ÷TP。如果不切, 所有 rank 冗余存全量 KV, TP 就失去了扩显存的最大价值。

### 为什么采样只在 rank0 做? (lm_head 用 gather 不用 AllGather)

`logits = model(x)` 前半段 (attention + MLP) 的每一个 Block 出口都会做 RowParallel AllReduce, 所以进入 `lm_head` 之前每个 rank 都持有完整的 hidden_states。但 `ParallelLMHead` **本身是按 vocab 维切的 Column 变体**, 每个 rank 只算 `[bs, vocab_size/TP]` 的分片, 然后:

```python
# embed_head.py:62-65
if self.tp_size > 1:
    all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
    dist.gather(logits, all_logits, 0)                                      # 单向 gather 到 rank0
    logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None       # worker 返回 None
```

关键是用了 `dist.gather` 而不是 `dist.all_gather`: **只把 vocab shard 收到 rank0**, worker 返回 `None`。这是一个有意识的通信优化 —— 既然只有 rank0 要采样, 广播给 worker 就是纯浪费:

- vocab_size 对大模型可能是 152064, `[max_bs=512, vocab]` 的 fp16 tensor ≈ 150MB, AllGather 会让每个 rank 都占这 150MB, 而 gather 只有 rank0 占
- 通信量: AllGather 需要每 rank 发出 `vocab/TP × bs` 并接收其它 `(TP-1)` 份; gather 只需要发到 rank0, 节省大约一半带宽

所以 `run()` 里的 `if self.rank == 0: self.sampler(...)` **不是一个额外加的守卫**, 而是被迫的 —— worker 侧 `logits` 本来就是 `None`, 强行采样会直接崩。采样天生只发生在 rank0, 随机数也只需要 rank0 一份, 完全不存在跨 rank 对齐 RNG 的问题。

**附带的 prefill 优化**: `ParallelLMHead.forward` 开头有一段

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

prefill 时每个序列只需要最后一个 token 的 logits(用来预测下一个), 所以在做 gemm 和 gather 之前先把 `[total_tokens, hidden]` 筛成 `[num_seqs, hidden]`, 计算量和通信量都按序列平均长度成比例降低。这也是 prefill 能塞下长 prompt 而 lm_head 不会爆显存的关键。

### 启动顺序里的一个小细节: worker 的 `__init__` 永远不 return

`ctx.Process(target=ModelRunner, ...)` → worker 进入 `ModelRunner.__init__` → 走到 `self.loop()` 就永远在 while True 里等指令, `__init__` 根本不会返回, 所以子进程里的 ModelRunner 对象只存在于 loop 栈帧上。这种 "构造函数即主循环" 的写法很紧凑, 但也意味着 worker 侧不能在外部拿到 ModelRunner 实例 —— 它完全由 rank0 通过 shm 远程驱动。

## 我的理解 / 类比

可以把整套机制想成 **一个指挥家 + 若干乐手**:

- **指挥家 (rank0)**: 手里拿着完整乐谱 (Scheduler 状态、Sequence 列表), 决定下一拍演奏什么 (run/warmup/exit)。他通过 "敲节拍棒 (Event.set)" 和 "公共曲谱板 (SharedMemory)" 告诉乐手们现在演奏哪一小节。
- **乐手 (rank1..N)**: 每人只拿着自己那一声部的乐谱 (切分后的权重), 不知道曲目的整体结构, 只负责 "听到节拍→看曲谱板→演奏自己那一段→和大家合声 (AllReduce)"。
- **合声点 (AllReduce)**: 每个 Transformer Block 的 Attention 和 MLP 末尾就是两个合声点, 所有乐手必须在这里声音合一, 之后又各自演奏下一段。
- **只指挥不演奏的部分 (采样、tokenize、调度)**: 只发生在指挥家那里, 乐手完全不参与。

Megatron-LM 风格切分的精髓就一句话: **让两次矩阵乘的切分方向相反, 中间的逐元素操作 (SiLU, attention, 重排) 天然对齐, 从而把通信压缩到每个 block 末尾只做一次 AllReduce。** 方向对不齐, 通信量就翻倍。

## 遗留问题
- [x] ~~nano-vllm 的 lm_head 到底是 replicated 还是 column-parallel?~~ → **按 vocab 维 Column 切, 聚合用 dist.gather 只送 rank0**, 见 [embed_head.py:45-66](../../nanovllm/layers/embed_head.py#L45-L66)
- [ ] CUDA Graph 是各 rank 独立录制的, 每个 rank 的 `capture_cudagraph` 里都调 `self.model(...)` —— 录制过程本身是否也会触发跨 rank 的 NCCL 通信? 如果是, graph.replay() 时这些通信是否也被录进了图里?
- [ ] SharedMemory 只有 1MB, 如果一次 `run(seqs=...)` 的 pickle 超过 1MB (极端大 batch + 长 block_table) 会静默溢出还是报错?
- [ ] `dist.init_process_group("nccl", "tcp://localhost:2333", ...)` 写死了端口, 同机多实例启动会端口冲突, 为什么不用 file:// 或 env://?
- [ ] 为什么不像 vLLM 那样把 Scheduler 也放到独立进程? 把 rank0 同时承担 "调度 + 模型执行" 是不是会让 prefill/decode 的 Python 循环阻塞 NCCL 通信窗口?
