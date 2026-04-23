# 整体架构速查

## 一句话总结

nano-vllm 是一个 ~1200 行的轻量 vLLM 实现：用 Scheduler 做 continuous batching 把多条请求合并调度，用 BlockManager 以 256-token 块为单位管理 KV Cache（含 xxhash 前缀缓存），用 ModelRunner 在 GPU 上执行前向推理（支持 CUDA Graph 加速 decode + 多卡张量并行），最终通过 LLMEngine 把 tokenize → schedule → forward → sample → detokenize 串成完整的生成循环。

## 核心数据流

```
LLM.generate(prompts)
  │
  ├─ 1. tokenize: 文本 → token_ids
  ├─ 2. add_request: token_ids → Sequence 对象, 加入 scheduler.waiting 队列
  │
  └─ 3. while not finished:
       │
       ├─ scheduler.schedule()
       │   ├─ 优先尝试 prefill: 从 waiting 取序列, 分配 KV cache 块 (含前缀缓存匹配)
       │   └─ 无 waiting 则 decode: 从 running 取序列, 按需追加新块, 显存不足时 preempt
       │
       ├─ model_runner.run(seqs, is_prefill)
       │   ├─ prepare: Sequence → GPU tensor (input_ids, positions, slot_mapping, block_tables...)
       │   ├─ forward: 模型前向 → logits
       │   │   ├─ prefill: eager 模式 (变长输入)
       │   │   └─ decode: CUDA Graph replay (固定形状, 省去 Python 调度开销)
       │   └─ sample: logits + temperature → next token_id
       │
       └─ scheduler.postprocess(seqs, token_ids)
            ├─ 将新 token 追加到 Sequence
            └─ 检查终止条件 (EOS / max_tokens) → 标记 FINISHED, 释放 KV cache 块

  └─ 4. detokenize: token_ids → 文本输出
```

## 模块职责与依赖关系

```
┌──────────────────────────────────────────────────────┐
│                   LLM (llm.py)                       │
│              纯透传, 继承 LLMEngine                    │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│               LLMEngine (llm_engine.py)              │
│  · tokenize/detokenize (AutoTokenizer)               │
│  · 主循环: add_request → step → generate             │
│  · 启动 TP worker 进程                                │
├──────────────┬───────────────────┬───────────────────┤
│              │                   │                   │
│   Scheduler  │    ModelRunner    │    Sequence        │
│ (scheduler)  │  (model_runner)   │   (sequence)      │
│              │                   │                   │
│ · 双阶段调度  │  · 模型加载       │  · 数据载体        │
│   prefill优先│  · KV cache 分配  │  · token_ids      │
│ · preempt    │  · 输入准备       │  · block_table    │
│              │  · CUDA Graph     │  · 状态机          │
│  ┌───────┐   │  · TP 通信        │  WAITING→RUNNING  │
│  │Block  │   │                   │  →FINISHED        │
│  │Manager│   │                   │                   │
│  └───────┘   │                   │                   │
└──────────────┴───────────────────┴───────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Qwen3Model      Layers         Utils
  (models/)      (layers/)       (utils/)
  · Attention    · attention.py   · context.py
  · MLP          · linear.py      · loader.py
  · RMSNorm      · layernorm.py
                 · rotary.py
                 · sampler.py
```

## 关键概念速查

| 概念 | 含义 | 代码位置 |
|------|------|----------|
| Prefill | 首次处理完整 prompt, 一次性填充所有 token 的 KV Cache | scheduler.py `schedule()` 前半段 |
| Decode | 自回归逐 token 生成, 每步只算 1 个新 token 但 attend 全部历史 | scheduler.py `schedule()` 后半段 |
| Continuous Batching | prefill 和 decode 的序列混合调度, 不等整批完成就插入新请求 | scheduler.py: 每次 `step()` 重新调度 |
| Block | 256 token 为单位的 KV Cache 物理块, GPU 上预分配连续 tensor | block_manager.py `Block` 类 |
| Prefix Caching | 相同前缀的序列共享 KV 块, 用 xxhash 链式哈希做匹配 | block_manager.py `allocate()` |
| Preemption | 显存不足时将 running 序列踢回 waiting, 释放其 KV cache 块 | scheduler.py `preempt()` |
| CUDA Graph | 预录制 decode 计算图, replay 跳过 Python 开销, 覆盖多种 batch size | model_runner.py `capture_cudagraph()` |
| Context | engine 和 layers 间传递 attention 参数 (cu_seqlens, slot_mapping 等) 的全局对象 | utils/context.py |
| Slot Mapping | 新 token 写入 KV cache 的物理位置索引 (block_id × block_size + offset) | model_runner.py `prepare_*()` |
| Tensor Parallel | 多 GPU 并行: rank0 通过 SharedMemory+Event 广播指令, NCCL 做 all-reduce | model_runner.py, linear.py |

## 关键数据结构

### Sequence (序列)
```
Sequence
├── seq_id: int                 # 全局唯一 ID
├── token_ids: list[int]        # prompt + 已生成的 token
├── status: SequenceStatus      # WAITING → RUNNING → FINISHED
├── block_table: list[int]      # 占用的 KV cache 块 ID 列表
├── num_cached_tokens: int      # 前缀缓存命中的 token 数
├── num_prompt_tokens: int      # prompt 长度 (不变)
├── num_tokens: int             # 当前总长度 (持续增长)
└── temperature / max_tokens    # 采样参数
```

### Block (KV Cache 块)
```
Block
├── block_id: int               # GPU KV cache 数组中的索引
├── ref_count: int              # 引用计数, 支持多序列共享
├── hash: int                   # 链式哈希 (-1 = 未填满/不可缓存)
└── token_ids: list[int]        # 块内容, 用于缓存命中时防碰撞校验
```

### Context (注意力参数)
```
Context
├── is_prefill: bool
├── cu_seqlens_q/k: Tensor      # prefill: flash-attn 的累积序列长度
├── max_seqlen_q/k: int         # prefill: 最大序列长度
├── slot_mapping: Tensor        # 新 token → KV cache 物理位置
├── context_lens: Tensor        # decode: 每个序列的上下文长度
└── block_tables: Tensor        # block_id 查找表 [num_seqs, max_blocks]
```

## 两种执行模式对比

| | CUDA Graph (默认) | Eager (`enforce_eager=True`) |
|---|---|---|
| 适用阶段 | decode (固定形状) | prefill (变长) + 调试 |
| 预录制 batch size | [1,2,4,8,16,32,...,512] | 不需要 |
| 运行时 | replay 预录制的 kernel 序列 | 标准 PyTorch forward |
| 优势 | 省去 Python/CUDA 调度开销 | 灵活, 支持动态形状 |
| 劣势 | 额外显存 (固定 buffer) | decode 时调度开销占比高 |

## 张量并行通信机制

```
rank0 (主进程)                        rank1..N (worker 子进程)
─────────────────                     ──────────────────────
LLMEngine.step()
  → model_runner.call("run", ...)
    → write_shm(method, args)         loop():
      ┌─ pickle 序列化 → SharedMemory    event.wait() ← 阻塞
      └─ event.set() ──────────────→    event 触发, read_shm()
                                         pickle 反序列化 → call("run", ...)
    → self.run(seqs, is_prefill)        → self.run(seqs, is_prefill)
      ├─ model forward (NCCL all-reduce 自动同步各 rank)
      └─ rank0 做 sampling              └─ 不做 sampling, 返回 None
```

## 初始化顺序 (不可变)

```
1. LLMEngine.__init__
   ├─ Config: 加载 HuggingFace 模型配置
   ├─ 启动 TP worker 进程 (rank 1..N)
   ├─ ModelRunner.__init__ (rank 0)
   │   ├─ dist.init_process_group("nccl")
   │   ├─ 构建模型 + 加载权重
   │   ├─ warmup_model(): 最大输入前向, 测峰值显存
   │   ├─ allocate_kv_cache(): 剩余显存 → KV cache 块数 → 分配
   │   └─ capture_cudagraph(): 从大到小录制各 batch size
   ├─ Tokenizer: 加载分词器, 获取 eos_token_id
   └─ Scheduler: 初始化 BlockManager (用 ModelRunner 算出的块数)
```

## 我的理解 / 类比

<!--
把 nano-vllm 想象成一个餐厅:
- Sequence = 一桌客人的点单 (prompt) + 已上的菜 (生成的 token)
- Scheduler = 前台经理: 决定先接待哪桌 (prefill), 哪桌继续上菜 (decode), 太挤了就让后来的等一等 (preempt)
- BlockManager = 仓库管理员: 256 份食材一箱, 相同的菜可以共用同一箱原料 (prefix caching)
- ModelRunner = 后厨: 拿到食材和订单后做菜 (forward), CUDA Graph 就是预设的"快速出餐流程"
- Context = 后厨和前台之间的传菜窗口: 前台把订单信息放上去, 后厨直接看
- SharedMemory + Event = 总厨 (rank0) 和各灶台 (rank1..N) 之间的对讲机
-->

## Q&A: 多进程基础概念详解

> 注意: 这里说的是多**进程** (multiprocessing)，不是多线程 (threading)。
> Python 的 GIL 使多线程无法真正并行执行 CPU/GPU 计算，所以 GPU 并行推理必须用多进程。

### Q1: `ctx = mp.get_context("spawn")` 是什么？为什么需要它？

`mp.get_context()` 选择**进程启动方式**。Python multiprocessing 有三种启动方式：

| 方式 | 原理 | 特点 |
|------|------|------|
| `fork` | 复制父进程的整个内存空间 | 快，但 CUDA 初始化后 fork 会导致 GPU 状态损坏 |
| `spawn` | 启动全新的 Python 解释器，从零执行 | 安全，但稍慢（需要重新 import 模块） |
| `forkserver` | 预启动一个服务进程，后续从它 fork | 折中方案 |

**nano-vllm 选 `spawn` 的原因：** CUDA 运行时在 fork 后的子进程中行为未定义（会 hang 或 crash）。`spawn` 让每个子进程从干净状态启动，各自独立初始化 CUDA，安全可靠。

`ctx` 是一个 context 对象，后续通过 `ctx.Process()`、`ctx.Event()` 创建的进程和同步原语都遵循 `spawn` 方式，保持一致性。

```python
# 这两种写法等价:
ctx = mp.get_context("spawn")
process = ctx.Process(...)     # 用 spawn 上下文创建

mp.set_start_method("spawn")
process = mp.Process(...)      # 全局设置，影响所有后续调用
```

用 `get_context()` 而不是 `set_start_method()` 的好处是**不污染全局状态**——如果其他库也用了 multiprocessing，不会互相冲突。

---

### Q2: `ctx.Event()` 是什么？怎么实现跨进程通知？

`Event` 是最简单的跨进程同步原语，本质是一个**共享的布尔标志位**：

```python
event = ctx.Event()    # 创建时 flag = False

# 进程 A (通知方):
event.set()            # flag → True，所有 wait() 的进程被唤醒

# 进程 B (等待方):
event.wait()           # 阻塞，直到 flag 变为 True
event.clear()          # flag → False，重置，准备下一次等待
```

**底层实现：** `spawn` 模式下，Event 内部用的是操作系统的**命名信号量 (named semaphore)**，通过文件系统或内核对象实现跨进程共享。创建 Event 的进程把它通过 `args` 传给子进程时，pickle 序列化的是信号量的**名字/句柄**，而不是值本身——所以两个进程操作的是同一个内核对象。

**在 nano-vllm 中的使用：**

```
rank0                              worker (rank 1)
─────                              ───────────────
event = ctx.Event()  ──(传给子进程)──→  self.event = event
                                       │
  ... 准备好指令 ...                     event.wait()  ← 阻塞中
  event.set()  ─────────────────────→  唤醒！读取指令
                                       event.clear()  ← 重置
  ... 准备下一条指令 ...                  event.wait()  ← 再次阻塞
  event.set()  ─────────────────────→  唤醒！
```

**为什么每个 worker 一个 Event，而不是共用一个？**
如果共用，`clear()` 会出现竞态：worker A 刚被唤醒还没读完指令，worker B 就把 flag 清了。每个 worker 独立的 Event 让各自按自己的节奏 wait/clear，互不干扰。

---

### Q3: `ctx.Process(target=ModelRunner, args=(...))` 发生了什么？

```python
process = ctx.Process(target=ModelRunner, args=(config, i, event))
process.start()
```

这两行做了以下事情：

**第一步：`Process()` — 只是创建对象，还没有启动进程**
- 记录 `target=ModelRunner` (要调用什么)
- 记录 `args=(config, i, event)` (调用参数)
- 此时子进程**不存在**

**第二步：`start()` — 真正创建子进程**

```
父进程 (rank 0)                          操作系统
─────────────                            ──────
process.start()
  → pickle.dumps(ModelRunner)  ────→    创建新的 Python 解释器进程
  → pickle.dumps((config, i, event))    │
                                        ▼
                                    子进程启动
                                    → import 所有必要模块
                                    → pickle.loads(...) 反序列化参数
                                    → 调用 ModelRunner(config, i, event)
                                      即执行 ModelRunner.__init__()
                                        → init_process_group
                                        → 加载模型到 GPU i
                                        → warmup / allocate / capture
                                        → self.loop()  ← 进入事件循环，阻塞等待
```

**关键点：`target=ModelRunner` 传的是类，不是函数**

- 如果 `target` 是函数 `f`，子进程执行 `f(*args)`
- 如果 `target` 是类 `C`，子进程执行 `C(*args)`，即调用 `__init__`
- ModelRunner 的 `__init__` 末尾对 worker (rank > 0) 调用了 `self.loop()`，这是一个永不返回的死循环——所以子进程启动后就一直活着，等待指令

**参数通过 pickle 序列化传递：**

因为 `spawn` 是启动全新解释器，父子进程不共享内存，所以参数必须序列化成字节流传过去。这就是为什么用 `torch.multiprocessing` 而不是标准 `multiprocessing`——前者注册了 CUDA tensor 的自定义序列化方法。

---

### Q4: `process.start()` vs `process.join()` 的区别？

```python
process.start()    # 启动子进程，父进程继续往下执行（非阻塞）
process.join()     # 父进程阻塞，等子进程结束后才继续（阻塞）
```

在 nano-vllm 中：
- `start()` 在 `__init__` 中调用——启动 worker 后立刻继续创建下一个
- `join()` 在 `exit()` 中调用——关闭时等所有 worker 退出，防止僵尸进程

```python
# llm_engine.py __init__:
for i in range(1, config.tensor_parallel_size):
    process = ctx.Process(...)
    process.start()     # 启动后立刻继续循环，不等子进程初始化完成
    # 子进程的初始化和父进程的后续代码是并行执行的！

# 那怎么确保所有进程都初始化完了？
# → dist.barrier() 在 ModelRunner.__init__ 中同步：
#   rank0 和所有 worker 都到达 barrier 后才继续
```

---

### Q5: 完整的进程生命周期

```
时间轴 →

主进程 (rank 0):
  start worker1 ──→ start worker2 ──→ ModelRunner(rank=0) ──→ ... 推理循环 ... ──→ exit()
                                       ├ init_process_group                        ├ call("exit")
                                       ├ load model                                │  → write_shm("exit")
                                       ├ warmup                                    │  → events.set()
                                       ├ allocate_kv_cache                         ├ del model_runner
                                       ├ capture_cudagraph                         └ join() 等待子进程结束
                                       ├ create SharedMemory
                                       └ barrier() ← 等 worker 就绪

worker1 (rank 1):
                    ModelRunner(rank=1)
                      ├ init_process_group
                      ├ load model (GPU 1)
                      ├ warmup
                      ├ allocate_kv_cache
                      ├ capture_cudagraph
                      ├ barrier() ← 与 rank0 同步
                      ├ attach SharedMemory
                      └ loop():
                          event.wait() → run() → event.wait() → run() → ... → "exit" → break → 进程结束

worker2 (rank 2):
                                      (同 worker1，用 GPU 2)
```

---

### Q6: Event vs SharedMemory vs NCCL — 三种通信方式各管什么？

nano-vllm 的多进程通信用了三种不同机制，各有分工：

| 机制 | 传递什么 | 方向 | 用途 |
|------|---------|------|------|
| **Event** | 1 bit 信号 (flag) | rank0 → worker | "有新指令了，醒来干活" |
| **SharedMemory** | pickle 序列化的方法名+参数 | rank0 → worker | 告诉 worker 该调用什么方法、传什么参数 |
| **NCCL** | GPU tensor (如 all-reduce 的梯度/激活值) | 所有 rank ↔ 所有 rank | 模型前向中各 GPU 之间同步中间计算结果 |

**为什么不只用一种？**
- Event 只能传 1 bit，不能传数据，但唤醒速度最快
- SharedMemory 能传数据但需要手动序列化，且只适合小数据（1MB）
- NCCL 专为 GPU tensor 设计，带宽高（NVLink 可达 600GB/s），但只能传 tensor

它们的配合就像：
- Event = 门铃（叮咚！有活干了）
- SharedMemory = 门口的信箱（放着具体的工作指令）
- NCCL = 厨房之间的传送带（传递大量食材/半成品）

## 遗留问题
- [ ] flash-attention 的 varlen API 具体怎么用 cu_seqlens_q/k 区分不同序列？
- [ ] CUDA Graph replay 时为什么用固定 buffer 而不能直接传新 tensor？
- [ ] prefix caching 的链式哈希: 如果只有中间部分不同、前后都相同，能部分命中吗？
- [ ] preempt 后序列被踢回 waiting，重新 schedule 时会重新做 prefill 吗？KV cache 全部丢失？
