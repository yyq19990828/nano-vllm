# Nano-vLLM 仓库架构

## 目录结构总览

```
nano-vllm/
├── nanovllm/                        # 核心包
│   ├── __init__.py                  # 公共 API: LLM, SamplingParams
│   ├── llm.py                       # 用户入口 (LLMEngine 薄封装)
│   ├── config.py                    # 全局配置 (dataclass)
│   ├── sampling_params.py           # 采样参数 (temperature, max_tokens)
│   │
│   ├── engine/                      # 推理引擎核心
│   │   ├── llm_engine.py            # 引擎主循环: tokenize → schedule → run → decode
│   │   ├── scheduler.py             # 两阶段调度器 (prefill / decode)
│   │   ├── block_manager.py         # KV Cache 块管理 + 前缀缓存
│   │   ├── model_runner.py          # GPU 执行: 模型加载、CUDA Graph、张量并行
│   │   └── sequence.py              # 序列数据结构 (token_ids, block_table, status)
│   │
│   ├── models/                      # 模型实现
│   │   └── qwen3.py                 # Qwen2/Qwen3 (当前唯一模型)
│   │
│   ├── layers/                      # 可复用算子层
│   │   ├── attention.py             # Flash-Attention + Triton KV Cache 存储
│   │   ├── linear.py                # 张量并行 Linear (Column/Row/QKV/Merged)
│   │   ├── embed_head.py            # 词表并行 Embedding + LM Head
│   │   ├── layernorm.py             # RMSNorm (支持残差融合)
│   │   ├── activation.py            # SiluAndMul (SwiGLU)
│   │   ├── rotary_embedding.py      # RoPE 位置编码
│   │   └── sampler.py               # Gumbel-max 采样 (torch.compile)
│   │
│   └── utils/
│       ├── context.py               # 全局 Context: 在 engine 和 layers 间传递注意力参数
│       └── loader.py                # safetensors 权重加载 + 张量并行分片
│
├── example.py                       # 基本用法示例
├── bench.py                         # 吞吐量基准测试
└── pyproject.toml                   # 包配置与依赖
```

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户调用层                                  │
│                                                                     │
│   llm = LLM(model_path)                                            │
│   outputs = llm.generate(prompts, sampling_params)                  │
│                          │                                          │
│                    llm.py (薄封装)                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLMEngine (llm_engine.py)                      │
│                                                                     │
│   职责: 推理主循环编排                                                │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  generate() 主循环:                                       │      │
│   │                                                          │      │
│   │  1. tokenize (AutoTokenizer)                             │      │
│   │  2. 创建 Sequence, 加入 Scheduler                         │      │
│   │  3. while not finished:                                  │      │
│   │       seqs, is_prefill = scheduler.schedule()            │      │
│   │       token_ids = model_runner.run(seqs, is_prefill)     │      │
│   │       scheduler.postprocess(seqs, token_ids)             │      │
│   │  4. decode token_ids → text                              │      │
│   └──────────────────────────────────────────────────────────┘      │
│                    │                           │                     │
│         ┌─────────▼─────────┐       ┌─────────▼──────────┐         │
│         │    Scheduler      │       │   ModelRunner       │         │
│         │  (scheduler.py)   │       │ (model_runner.py)   │         │
│         └─────────┬─────────┘       └─────────┬──────────┘         │
│                   │                           │                     │
│         ┌─────────▼─────────┐                 │                     │
│         │  BlockManager     │                 │                     │
│         │(block_manager.py) │                 │                     │
│         └───────────────────┘                 │                     │
└───────────────────────────────────────────────┼─────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU 计算层                                    │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              Qwen3ForCausalLM (models/qwen3.py)         │       │
│   │                                                         │       │
│   │   Embedding ──→ N × DecoderLayer ──→ RMSNorm ──→ LMHead│       │
│   │                      │                                  │       │
│   │              ┌───────┴────────┐                         │       │
│   │              │  DecoderLayer  │                         │       │
│   │              │                │                         │       │
│   │              │  RMSNorm       │                         │       │
│   │              │     ↓          │                         │       │
│   │              │  Attention     │ ← Flash-Attention       │       │
│   │              │  (QKV + RoPE   │   + KV Cache            │       │
│   │              │   + Output)    │                         │       │
│   │              │     ↓          │                         │       │
│   │              │  RMSNorm       │                         │       │
│   │              │     ↓          │                         │       │
│   │              │  MLP           │ ← SwiGLU                │       │
│   │              │  (Gate+Up      │   (SiluAndMul)          │       │
│   │              │   + Down)      │                         │       │
│   │              └────────────────┘                         │       │
│   └─────────────────────────────────────────────────────────┘       │
│                           │                                         │
│                    ┌──────▼──────┐                                   │
│                    │   Sampler   │ ← Gumbel-max (torch.compile)     │
│                    └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 推理流程详解

### 1. 两阶段调度 (Scheduler)

```
                    schedule()
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
     有 WAITING 序列?            无 WAITING 序列
            │                       │
            ▼                       ▼
    ┌───────────────┐      ┌────────────────┐
    │  Prefill 阶段  │      │  Decode 阶段    │
    │               │      │                │
    │ - 贪心调度     │      │ - 逐token生成   │
    │   WAITING序列  │      │ - 内存不足时    │
    │ - 分配KV块     │      │   抢占(preempt) │
    │ - WAITING→     │      │   末尾序列      │
    │   RUNNING     │      │                │
    └───────────────┘      └────────────────┘
            │                       │
            ▼                       ▼
     is_prefill=True         is_prefill=False
```

**约束条件:**
- 每批最多 `max_num_seqs` (512) 个序列
- 每批最多 `max_num_batched_tokens` (16384) 个 token

### 2. KV Cache 块管理 (BlockManager)

```
    ┌──────────────────────────────────────────────┐
    │          物理 KV Cache (GPU 显存)              │
    │                                              │
    │  Block 0   Block 1   Block 2   Block 3  ...  │
    │  [256tok]  [256tok]  [256tok]  [256tok]      │
    └──────┬───────┬───────┬───────┬───────────────┘
           │       │       │       │
           ▼       ▼       ▼       ▼
    ┌─────────────────────────────────────┐
    │     Sequence A: block_table=[0,2]   │  ← 逻辑块 → 物理块映射
    │     Sequence B: block_table=[1,3]   │
    └─────────────────────────────────────┘

    前缀缓存 (Prefix Caching):
    ┌──────────────────────────┐
    │  xxhash(token_ids)       │
    │       │                  │
    │       ▼                  │
    │  hash → block_id 映射    │ ← 相同前缀的序列共享物理块
    │  命中: ref_count++       │
    │  未命中: 分配新块         │
    └──────────────────────────┘
```

### 3. CUDA Graph 执行模式

```
    ┌─────────────────────────────────────────┐
    │             ModelRunner.run()            │
    │                    │                    │
    │        ┌───────────┴──────────┐         │
    │        ▼                      ▼         │
    │   is_prefill?             is_decode?    │
    │        │                      │         │
    │        ▼                      ▼         │
    │   Eager 执行            enforce_eager?  │
    │   (变长输入,             │         │    │
    │    不可图化)         ┌───┴───┐          │
    │                     Yes     No          │
    │                      │       │          │
    │                      ▼       ▼          │
    │                  Eager  CUDA Graph      │
    │                  执行    replay()       │
    │                         │               │
    │              ┌──────────▼──────────┐    │
    │              │ 预录制的图           │    │
    │              │ batch_size:          │    │
    │              │ [1,2,4,8,16..512]   │    │
    │              │ 向上取整到最近尺寸    │    │
    │              └─────────────────────┘    │
    └─────────────────────────────────────────┘
```

### 4. 张量并行 (多GPU)

```
    ┌────────────────────────────────────────────────────┐
    │                  LLMEngine (Rank 0)                 │
    │                       │                            │
    │            ┌──────────┴──────────┐                  │
    │            ▼                     ▼                  │
    │     ModelRunner              ModelRunner            │
    │      (Rank 0)                (Rank 1..N)           │
    │         │                       │                  │
    │         │   SharedMemory+Event  │                  │
    │         │ ◄────────────────────►│  ← 序列化传参     │
    │         │                       │                  │
    │         │      NCCL AllReduce   │                  │
    │         │ ◄────────────────────►│  ← 梯度同步      │
    │         │                       │                  │
    │    ┌────┴────┐            ┌─────┴────┐             │
    │    │ Column  │            │ Column   │             │
    │    │ Shard   │            │ Shard    │             │
    │    │ (Q,K,V, │            │ (Q,K,V,  │             │
    │    │  Gate,  │            │  Gate,   │             │
    │    │  Up)    │            │  Up)     │             │
    │    └────┬────┘            └─────┬────┘             │
    │         │      AllReduce        │                  │
    │    ┌────┴────┐ ◄───────►  ┌─────┴────┐            │
    │    │  Row    │            │  Row     │             │
    │    │  Shard  │            │  Shard   │             │
    │    │ (O_proj,│            │ (O_proj, │             │
    │    │  Down)  │            │  Down)   │             │
    │    └─────────┘            └──────────┘             │
    └────────────────────────────────────────────────────┘

    Rank 0: 编排调度 + 采样
    Rank 1+: 仅 GPU 计算, 通过 SharedMemory 接收指令
```

### 5. Attention 层数据流

```
    ┌─────────────────────────────────────────────┐
    │            Attention.forward(q, k, v)        │
    │                       │                     │
    │         Triton Kernel: store_kvcache()       │
    │         将 k, v 写入物理 KV Cache             │
    │                       │                     │
    │           ┌───────────┴──────────┐          │
    │           ▼                      ▼          │
    │     is_prefill?             is_decode?      │
    │           │                      │          │
    │           ▼                      ▼          │
    │  flash_attn_varlen_func   flash_attn_with_  │
    │  (变长序列, 支持           kvcache           │
    │   前缀缓存block_table)   (单token, 从       │
    │                           cache读取KV)      │
    └─────────────────────────────────────────────┘
```

### 6. Context 传参机制

```
    ModelRunner                              Attention Layer
    ┌──────────────┐                        ┌──────────────┐
    │ prepare_     │   set_context()        │              │
    │ prefill()    │ ──────────────────►    │ get_context() │
    │ prepare_     │   全局 Context:        │              │
    │ decode()     │   - is_prefill         │ 读取:         │
    │              │   - cu_seqlens_q/k     │ - slot_mapping│
    │              │   - slot_mapping       │ - block_tables│
    │              │   - block_tables       │ - cu_seqlens  │
    │              │   - context_lens       │              │
    └──────────────┘                        └──────────────┘

    作用: 避免在 model.forward() 签名中传递大量注意力参数
         engine 层设置 context, layers 层读取 context
```

## 核心优化技术汇总

| 优化 | 实现位置 | 原理 |
|------|---------|------|
| Prefix Caching | `block_manager.py` | xxhash 哈希相同 token 前缀, 共享物理 KV 块 |
| CUDA Graph | `model_runner.py` | 预录制 decode 计算图, `replay()` 减少 kernel launch 开销 |
| Flash-Attention v2 | `attention.py` | O(N) 内存的注意力, 区分 prefill/decode 两种 API |
| Triton KV Store | `attention.py` | 自定义 Triton kernel 按 slot 写入 KV Cache |
| 张量并行 | `linear.py`, `embed_head.py` | Column/Row 并行切分, NCCL AllReduce 聚合 |
| torch.compile | `sampler.py`, `layernorm.py`, `rotary_embedding.py`, `activation.py` | 编译关键算子 (采样、RMSNorm、RoPE、SiLU) |
| 残差融合 | `layernorm.py` | `add_rms_forward` 将残差加法与 RMSNorm 融合 |
| 序列化优化 | `sequence.py` | 自定义 `__getstate__` 减少多GPU pickle 传输量 |
