# 调度策略

## 一句话总结

Scheduler 在每个推理步骤被调用一次，采用 **prefill 优先** 策略：优先将 WAITING 队列中的新序列送入 prefill，只有当 WAITING 为空时才对 RUNNING 序列执行 decode。内存不足时通过抢占（preempt）末尾序列释放 KV cache。

## 核心流程

```
schedule() 每个 step 调用一次:
├── WAITING 队列非空 → Prefill 阶段
│   - 从 WAITING 头部贪心取序列
│   - 受两个约束: max_num_seqs（最大并发数）、max_num_batched_tokens（最大 token 总数）
│   - 为每个序列分配 KV cache 块（allocate）
│   - 状态变更: WAITING → RUNNING
│   - 任一约束不满足则停止，返回已调度序列
│
└── WAITING 队列为空 → Decode 阶段
    - 从 RUNNING 队列逐个取出序列
    - 检查是否能追加 1 个 token 的 KV cache（can_append）
    - 内存不足时: 从 RUNNING 尾部抢占序列（RUNNING → WAITING），释放其 block
    - 抢占到连自己都不够时: 抢占自身，放弃本轮
    - 成功分配的序列放回 RUNNING 头部，返回已调度序列
```

### 状态机

```
                 schedule(prefill)           postprocess(EOS/max_tokens)
  add() → WAITING ──────────────→ RUNNING ──────────────────────────→ FINISHED
              ↑                      │
              └──────────────────────┘
                   preempt(内存不足)
```

### 调用链

```
LLMEngine.generate()
  └── while not is_finished():
        └── step()
              ├── scheduler.schedule()       ← 决定本轮参与计算的序列
              ├── model_runner.run()          ← GPU 前向推理，生成 token
              └── scheduler.postprocess()    ← 追加 token，标记完成的序列
```

## 代码锚点
- schedule() 主逻辑: scheduler.py:L24-L58
- prefill 分支: scheduler.py:L25-L41
- decode 分支: scheduler.py:L43-L58
- preemption 逻辑: scheduler.py:L60-L63
- postprocess: scheduler.py:L65-L71

## 关键设计决策

### 为什么 prefill 优先于 decode?

Prefill 是一次性处理整个 prompt，之后序列就可以进入 decode 逐 token 生成。优先 prefill 可以：
1. **降低排队延迟** — 新请求尽快开始处理，而不是等所有已有序列 decode 完
2. **实现 continuous batching** — 新序列随时可以加入正在运行的 batch，不需要等一批全部完成

### preemption 为什么选择末尾序列?

`self.running.pop()` 抢占的是队列尾部（最后加入的序列）。这是 **LIFO 抢占**：
- 最早进入的序列已经生成了更多 token，抢占它浪费更多已完成的计算
- 最新进入的序列损失最小，重新 prefill 的代价也最低
- 类似操作系统中的 LRU 思想：保护"投资"最多的序列

### 和 Orca / vLLM 的 iteration-level scheduling 的关系

这就是 **iteration-level scheduling**（Orca 论文提出）的简化实现：
- 每个推理步骤（iteration）独立做调度决策，而非固定 batch 跑到底
- 序列完成后立即释放资源，新序列可以立即加入
- vLLM 在此基础上加了 PagedAttention（block 级别的 KV cache 管理），nano-vllm 同样实现了 block_manager

## 关键细节

### while...else 语法（decode 阶段）

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break              # break → else 不执行，序列被抢占
else:
    # 没有 break → can_append 为 True，分配成功
    num_seqs += 1
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
```

Python 的 `while...else`：循环正常结束（条件变 False）执行 else；被 break 中断则跳过 else。

### extendleft + reversed 保序

```python
self.running.extendleft(reversed(scheduled_seqs))
```

`extendleft` 会逐个插入头部（自带反转效果），所以先 `reversed` 再 `extendleft` = 按原序放回队列头部。

### postprocess 的终止条件

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
```

两种情况结束生成：遇到 EOS token（且未设置忽略），或达到最大生成长度。

## 我的理解 / 类比

可以类比餐厅叫号系统：
- **WAITING** = 等位区（新来的客人排队）
- **RUNNING** = 用餐中（正在占用餐桌/KV cache）
- **prefill** = 领客人入座、上菜（一次性处理 prompt）
- **decode** = 客人一口一口吃（逐 token 生成）
- **preempt** = 餐厅满了，让最后来的客人先回等位区，把桌子让给更早的客人
- **postprocess** = 客人吃完结账离开，释放餐桌

## Q&A

### Q: while 还能和 else 同级搭配？

是的，Python 的 `while...else`（以及 `for...else`）：循环正常结束（条件变 False）执行 else；被 `break` 中断则跳过 else。可以把 else 理解为 "no break"。decode 阶段利用这个语法区分"成功分配"和"被迫抢占自身"两种情况。

### Q: schedule() 需要被循环调用？

是的。`LLMEngine.step()` 每次调用一次 `schedule()`，而 `generate()` 在 `while not is_finished()` 循环中反复调用 `step()`。每个推理步骤都要重新调度，因为序列状态在持续变化（有完成的、有新加入的、内存在波动）。

### Q: 这里算简单的 PD 分离的实现吗？

不算。这里是 **PD 分阶段（phase-level separation）**，不是真正的 **PD 分离（disaggregation）**。

- **nano-vllm 做的**：同一 GPU 上，每个 step 要么 prefill 要么 decode，不混合。实现简单，对 CUDA graph 友好。
- **真正的 PD 分离**（Splitwise / DistServe / Mooncake）：prefill 和 decode 跑在不同 GPU 集群上，并行执行，需要跨节点传输 KV cache。

| | nano-vllm（分阶段） | PD 分离（disaggregation） |
|---|---|---|
| 硬件 | 同一组 GPU | 不同 GPU 集群 |
| 并行度 | 互斥，同一时刻只做一种 | 同时进行 |
| KV cache | 本地显存 | 需要跨节点传输 |
| 复杂度 | 低 | 高（传输、路由、负载均衡） |

vLLM 后来的 chunked prefill 甚至打破了分阶段限制，允许 prefill 和 decode 混在同一个 batch 里。

## 遗留问题
- [ ] block_manager 的 can_append vs can_allocate 区别是什么？allocate 是整块分配，append 是追加单 token？
- [ ] prefix caching（num_cached_tokens）如何与调度交互？
- [ ] 如果所有序列都被 preempt 了会怎样？下一轮 schedule 会重新 prefill 它们？
