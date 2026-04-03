# Nano-vLLM 学习路线图

本文档为项目初学者设计，帮助你从零理解这个 ~1200 行的 LLM 推理引擎。

## 前置知识

在阅读代码前，建议对以下概念有基本了解：

| 领域 | 需要了解的内容 |
|------|---------------|
| Transformer | 自注意力机制、KV Cache 的作用 |
| PyTorch | `nn.Module`、`torch.compile`、`torch.inference_mode` |
| GPU 编程 | CUDA 基本概念 (kernel, stream, graph) |
| 分布式 | AllReduce 操作的含义 (张量并行部分需要) |

不需要提前掌握：Triton、Flash-Attention API、vLLM 源码。这些在阅读过程中会自然理解。

---

## 阶段一：跑通示例，建立直觉

**目标：** 看到推理结果，理解用户 API

**阅读顺序：**

```
1. example.py          ← 看用法：LLM + SamplingParams → generate
2. nanovllm/__init__.py  ← 公共 API 只暴露了 LLM 和 SamplingParams
3. nanovllm/llm.py       ← LLM 是 LLMEngine 的薄封装 (~20行)
```

**动手：**
```bash
pip install -e .
# 修改 example.py 中的模型路径, 然后:
python example.py
```

**收获：** 理解用户调用链 `LLM.generate() → LLMEngine.generate()`

---

## 阶段二：理解推理主循环

**目标：** 搞清楚 "一个 prompt 怎么变成输出 text"

**阅读顺序：**

```
4. config.py                ← 配置项的含义, 特别是 max_num_seqs, max_num_batched_tokens
5. sampling_params.py       ← 采样参数 (很简单, 3个字段)
6. engine/sequence.py       ← 核心数据结构: token_ids, block_table, status
7. engine/llm_engine.py     ← 重点! generate() 的主循环
```

**关键理解：** `llm_engine.py:48` 的 `step()` 方法是整个引擎的心跳：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()    # 决定这一步处理哪些序列
    token_ids = self.model_runner.call("run", ...)   # GPU 计算得到 next token
    self.scheduler.postprocess(seqs, token_ids)      # 追加 token, 检查终止条件
```

**收获：** 理解 schedule → run → postprocess 三步循环

---

## 阶段三：调度与内存管理

**目标：** 理解 "哪些序列上 GPU" 和 "KV Cache 怎么分配"

**阅读顺序：**

```
8. engine/scheduler.py      ← 两阶段调度: prefill 优先, decode 兜底
9. engine/block_manager.py  ← KV Cache 块分配 + 前缀缓存
```

**Scheduler 核心逻辑 (scheduler.py:24-58)：**

```
schedule() 被调用时:
├── 有 WAITING 序列? → Prefill 阶段
│   贪心地把 WAITING 序列移入 RUNNING, 分配 KV 块
│   返回 is_prefill=True
│
└── 无 WAITING 序列 → Decode 阶段
    为所有 RUNNING 序列生成下一个 token
    如果内存不够 → 抢占(preempt)末尾序列回到 WAITING
    返回 is_prefill=False
```

**BlockManager 关键概念：**
- KV Cache 按 256 token 为一块 (block) 管理
- `allocate()`: 为新序列分配块, 通过 xxhash 检测前缀缓存命中
- `deallocate()`: 序列完成时释放块
- `may_append()`: decode 时可能需要追加新块 (当前块写满时)

**收获：** 理解批处理调度策略和显存管理机制

---

## 阶段四：GPU 执行与模型计算

**目标：** 理解数据怎么从 CPU 送到 GPU，模型怎么跑

**阅读顺序：**

```
10. engine/model_runner.py    ← GPU 侧执行的全部逻辑
11. models/qwen3.py           ← Transformer 模型结构
```

**ModelRunner 的关键方法：**

| 方法 | 作用 |
|------|------|
| `prepare_prefill()` | 构建 prefill 输入张量: input_ids, positions, cu_seqlens, slot_mapping |
| `prepare_decode()` | 构建 decode 输入张量: 每个序列只取最后一个 token |
| `run_model()` | 选择 Eager 或 CUDA Graph 执行模型 |
| `capture_cudagraph()` | 初始化时为每个 batch size 录制 CUDA Graph |
| `allocate_kv_cache()` | 根据剩余显存计算并分配 KV Cache |

**Qwen3 模型结构 (qwen3.py)：**

```
Embedding → [DecoderLayer × N] → RMSNorm → LMHead
                  │
           RMSNorm + Attention (QKV → RoPE → FlashAttn → O_proj)
                  │
           RMSNorm + MLP (Gate+Up → SiLU*Mul → Down)
```

**收获：** 理解 prefill vs decode 的输入差异, CUDA Graph 的录制与回放

---

## 阶段五：底层算子

**目标：** 理解各优化算子的实现

**阅读顺序 (可按兴趣选读)：**

```
12. layers/attention.py         ← Flash-Attention 调用 + Triton KV Cache 存储 kernel
13. layers/linear.py            ← 张量并行的线性层 (Column/Row/QKV 切分)
14. layers/embed_head.py        ← 词表并行 Embedding 和 LM Head
15. layers/layernorm.py         ← RMSNorm + 残差融合 (torch.compile)
16. layers/rotary_embedding.py  ← RoPE 位置编码
17. layers/activation.py        ← SiLU × Mul (SwiGLU 激活)
18. layers/sampler.py           ← Gumbel-max 采样 (torch.compile)
```

**attention.py 最值得细读：**
- `store_kvcache_kernel`: Triton kernel, 按 slot_mapping 将 K/V 写入 cache
- Prefill 时用 `flash_attn_varlen_func` (变长序列拼接)
- Decode 时用 `flash_attn_with_kvcache` (从 cache 读 KV)

**收获：** 理解性能优化的具体实现

---

## 阶段六：工具层与多 GPU

**目标：** 理解辅助机制

```
19. utils/context.py   ← 全局 Context: engine 设置, layers 读取 (避免传参污染)
20. utils/loader.py    ← safetensors 权重加载 + packed_modules_mapping 分片
```

**多 GPU 通信 (model_runner.py:41-88)：**
- Rank 0 通过 SharedMemory 写入序列化指令
- Rank 1+ 通过 Event 等待, 反序列化后执行相同方法
- NCCL AllReduce 用于 RowParallelLinear 的结果聚合

**收获：** 理解完整系统如何协同工作

---

## 学习路线图总览

```
阶段一 (30 min)              阶段二 (1-2 h)            阶段三 (1-2 h)
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ 跑通示例      │    →     │ 推理主循环    │    →     │ 调度与内存    │
│              │          │              │          │              │
│ example.py   │          │ llm_engine   │          │ scheduler    │
│ llm.py       │          │ sequence     │          │ block_manager│
│ __init__.py  │          │ config       │          │              │
└──────────────┘          └──────────────┘          └──────────────┘
                                                           │
                                                           ▼
阶段六 (1 h)                阶段五 (2-3 h)            阶段四 (2-3 h)
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ 工具与多GPU   │    ←     │ 底层算子      │    ←     │ GPU执行与模型 │
│              │          │              │          │              │
│ context.py   │          │ attention    │          │ model_runner │
│ loader.py    │          │ linear       │          │ qwen3        │
│ 多GPU通信     │          │ layernorm... │          │              │
└──────────────┘          └──────────────┘          └──────────────┘
```

## 调试技巧

| 场景 | 方法 |
|------|------|
| 想跳过 CUDA Graph 直接调试 | `LLM(model, enforce_eager=True)` |
| 想看调度行为 | 在 `scheduler.py:schedule()` 打印 `len(self.waiting)`, `len(self.running)` |
| 想看 KV Cache 使用 | 打印 `len(block_manager.free_block_ids)` |
| 想看每步生成了什么 | 在 `llm_engine.py:step()` 后打印 `token_ids` |
| 显存不足 | 减小 `max_num_seqs` 或 `max_num_batched_tokens` |

## 推荐对照阅读

如果你同时在学习 vLLM，以下是 nano-vllm 与 vLLM 的对应关系：

| Nano-vLLM | vLLM 对应 |
|-----------|-----------|
| `Scheduler` | `vllm.core.scheduler.Scheduler` |
| `BlockManager` | `vllm.core.block_manager.BlockSpaceManager` |
| `ModelRunner` | `vllm.worker.model_runner.ModelRunner` |
| `Sequence` | `vllm.sequence.Sequence` + `SequenceGroup` |
| `Context` | `vllm.attention.backends.*.Metadata` |
| `LLMEngine` | `vllm.engine.llm_engine.LLMEngine` |
