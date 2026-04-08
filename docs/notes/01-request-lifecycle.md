# 请求生命周期

## 一句话总结
用户传入 prompts 字符串列表, 经过 tokenize → 创建 Sequence → 循环 (调度→GPU执行→后处理) → decode 回字符串, 整个过程由 LLMEngine.generate() 编排。


## 核心流程

```
用户调用:
  llm.generate(["introduce yourself"], SamplingParams(temperature=0.6, max_tokens=256))

═══════════════════════════════════════════════════════════════

1. 参数标准化                          llm_engine.py:L67-L68
   sampling_params 不是 list → 复制成与 prompts 等长的 list

2. add_request() × N                   llm_engine.py:L69-L70
   对每个 prompt:
   ├── tokenize: str → token_ids       llm_engine.py:L43-L44
   │   "introduce yourself" → [123, 456, ...]
   ├── 创建 Sequence 对象               llm_engine.py:L45
   │   Sequence(token_ids, sampling_params)
   │   - 分配 seq_id (全局递增)
   │   - status = WAITING
   │   - block_table = [] (尚未分配 KV 块)
   └── 加入调度器等待队列                llm_engine.py:L46
       scheduler.add(seq) → self.waiting.append(seq)

3. 主循环 while not is_finished()      llm_engine.py:L73-L88
   │
   ├── step()                          llm_engine.py:L48-L54
   │   ├── scheduler.schedule()        → 选出本轮要处理的 seqs + 是否 prefill
   │   ├── model_runner.call("run")    → GPU 前向计算, 返回 next token_ids
   │   └── scheduler.postprocess()     → 追加 token, 判断终止
   │
   └── 收集已完成序列的 token_ids 到 outputs dict

4. 排序 + decode                        llm_engine.py:L89-L90
   outputs 按 seq_id 排序 (保证顺序与输入一致)
   tokenizer.decode(token_ids) → 最终文本
```


## 各阶段详解

### 2a. Sequence 对象创建 (sequence.py:L14-L29)

```python
Sequence(token_ids=[123, 456, ...], sampling_params)
```

创建后的关键字段:
| 字段 | 值 | 含义 |
|------|----|------|
| seq_id | 全局递增 | 用于最终排序输出 |
| status | WAITING | 初始状态, 等待调度 |
| token_ids | prompt 的 token 列表 | 会随 decode 不断 append |
| num_prompt_tokens | len(token_ids) | 固定不变, 区分 prompt/completion |
| num_tokens | len(token_ids) | 随 decode 递增 |
| block_table | [] | 调度时由 BlockManager 填充 |
| temperature | 来自 SamplingParams | 采样温度 |
| max_tokens | 来自 SamplingParams | 最大生成长度 |

### 3a. schedule() 调度 (scheduler.py:L24-L58)

```
schedule() 被调用:
├── waiting 非空 → Prefill 阶段 (is_prefill=True)
│   贪心取 waiting 队首序列:
│   - 检查 num_batched_tokens + len(seq) ≤ 16384
│   - 检查 block_manager.can_allocate(seq)
│   - 通过 → allocate KV 块, status=RUNNING, 移入 running 队列
│   - 不通过 → break, 剩余的继续等
│
└── waiting 为空 → Decode 阶段 (is_prefill=False)
    遍历 running 中所有序列:
    - can_append? → may_append (可能追加新块)
    - 内存不足 → preempt 末尾序列 (status=WAITING, 释放块, 塞回 waiting 队首)
```

### 3b. model_runner.call("run") (GPU 执行)

```
seqs + is_prefill → ModelRunner:
├── prefill: 拼接所有序列全部 token, 一次前向 → 每个序列得到 1 个 next token
└── decode:  每个序列只取最后 1 个 token, 一次前向 → 每个序列得到 1 个 next token
```

### 3c. postprocess() 后处理 (scheduler.py:L65-L71)

```python
for seq, token_id in zip(seqs, token_ids):
    seq.append_token(token_id)           # token_ids 列表追加, num_tokens++
    if token_id == eos or 达到 max_tokens:
        seq.status = FINISHED            # 标记完成
        block_manager.deallocate(seq)    # 释放 KV 块
        running.remove(seq)              # 移出 running 队列
```


## 完整状态流转

```
                add_request()          schedule()
  [创建] ──────────→ WAITING ──────────→ RUNNING
                        ↑                  │
                        │   preempt()      │  每次 step():
                        └──────────────────│  append_token()
                                           │
                                           │  token==eos 或 达到 max_tokens
                                           ↓
                                       FINISHED
                                           │
                                           ↓
                                    deallocate + 移出 running
                                    收集到 outputs dict
```


## 关键设计决策

### 为什么 generate() 不直接调 model.forward(), 要绕一圈 scheduler?
批处理。多条 prompt 可以合并在一个 batch 里同时跑 GPU, 比逐条推理快很多。
scheduler 负责决定哪些序列能凑成一批, 同时管理有限的 GPU 显存 (KV Cache 块)。

### 为什么 prefill 和 decode 要分开?
输入形状不同:
- prefill: 每个序列有完整 prompt (几百~几千 token), 变长
- decode: 每个序列只需 1 个 token, 等长
分开处理可以用不同的优化路径 (prefill 用 varlen flash-attn, decode 用 CUDA Graph)。

### 为什么 outputs 按 seq_id 排序?
多条序列完成顺序不确定 (短的先完成), 但用户期望输出顺序与输入一致。
seq_id 是按 add_request 顺序递增的, 排序后恢复原始顺序。

### 为什么 Sequence.__getstate__ 做了特殊处理?
张量并行时, Rank 0 要通过 SharedMemory 把序列信息传给其他 rank。
序列化优化: decode 阶段只传 last_token 而非完整 token_ids, 减少传输量。
(其他 rank 只需要 block_table 和最新 token 就够了, 不需要完整历史)


## 代码锚点汇总

| 环节 | 位置 |
|------|------|
| generate() 入口 | engine/llm_engine.py:L59 |
| 参数标准化 | engine/llm_engine.py:L67-L68 |
| add_request (tokenize + 创建 Sequence) | engine/llm_engine.py:L42-L46 |
| step() 主循环 | engine/llm_engine.py:L48-L54 |
| schedule() | engine/scheduler.py:L24-L58 |
| postprocess() | engine/scheduler.py:L65-L71 |
| decode 输出 | engine/llm_engine.py:L89-L90 |
| Sequence 定义 | engine/sequence.py:L14-L29 |
| SamplingParams 定义 | sampling_params.py:L4-L8 |


## 遗留问题
- [ ] prefill 阶段多个序列的 token 是怎么拼接成一个 batch 的? (→ 04-model-execution.md)
- [ ] block_manager.allocate() 具体做了什么? (→ 03-memory-management.md)
- [ ] num_tokens 的 throughput 计算: prefill 时 num_tokens>0, decode 时用负数表示 batch size, 这个设计意图?
