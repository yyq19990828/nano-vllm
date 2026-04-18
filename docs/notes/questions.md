# 未解决的问题

学习过程中遇到的疑问, 定期回顾并尝试回答。

## 格式
```
### [日期] 问题描述
- 来源: 哪个文件/哪段代码引发的
- 当前猜测: (如果有的话)
- 解答: (弄懂后填写)
- 状态: 未解决 / 已解决
```

---

<!-- 在下方记录问题 -->

### [2026-04-10] Sequence.__getstate__ / __setstate__ 的作用是什么?
- 来源: `nanovllm/engine/sequence.py:74-83`
- 解答:
  - 这两个方法是 Python pickle 序列化/反序列化钩子, 用于**多 GPU 张量并行时的通信优化**
  - 调用链路: rank0 在 `model_runner.py` 的 `write_shm()` 中通过 `pickle.dumps()` 将 Sequence 列表写入共享内存, worker 端 `read_shm()` 中 `pickle.loads()` 还原
  - **核心优化**: 不序列化整个 Sequence 对象, 只传 worker 需要的字段, 且按阶段区分:
    - Prefill 阶段 (`num_completion_tokens == 0`): 传完整 `token_ids`, 因为 worker 需要全部 token 做前向计算
    - Decode 阶段: 只传 `last_token`(一个 int), 因为 decode 每次只需最新 token, 避免传输整个 token 列表
  - **效果**: decode 阶段每个 Sequence 只传几个 int, 大幅减少共享内存的序列化/反序列化开销
- 状态: 已解决

### [2026-04-10] prepare_prefill 具体在做什么?
- 来源: `nanovllm/engine/model_runner.py:153-194`
- 解答:
  - 核心任务: 把多个 Sequence 转换为 flash-attention 需要的 GPU tensor 格式
  - 产出的 tensor 及用途:
    - `input_ids`: 只包含未缓存的 token(跳过 `seq[:num_cached_tokens]`), 送入模型做前向
    - `positions`: 对应的绝对位置(从 `num_cached_tokens` 开始), 用于 RoPE
    - `cu_seqlens_q`: Q 的累积序列长度, 只含未缓存 token 的长度
    - `cu_seqlens_k`: K 的累积序列长度, 含全部 token(包括缓存的), 因为 attention 要看到完整上下文
    - `slot_mapping`: 每个新 token 写入 KV cache 的物理 slot 地址, 跳过已缓存的块
    - `block_tables`: 仅当 K > Q(有前缀缓存命中)时才构建, 用于从 KV cache 读取已缓存的 KV
  - **Q ≠ K 是关键**: 有前缀缓存时, Q 长度(需计算的) < K 长度(需 attend 到的全部上下文)
  - 所有 tensor 通过 `set_context()` 以 thread-local 方式传给 Attention 层, 避免侵入模型前向签名
- 状态: 已解决

### [2026-04-10] Transformer block 中只有 KV cache 可以跨步骤复用吗?
- 来源: `nanovllm/engine/model_runner.py` (run_model / prepare_decode)
- 解答:
  - **是的, 只有 KV cache 是跨步骤复用的中间结果**
  - 原因: attention 是 Transformer 中**唯一涉及 token 间交互的操作**
    - `Attention = softmax(Q · K^T / √d) · V`, 当前 token 的 Q 要和所有历史 token 的 K、V 交互
    - 如果不缓存, decode 每步都要重算前面所有 token 的 K/V, 复杂度从 O(n) 变 O(n²)
  - 其他组件都是逐 token 独立计算, 无状态, 不需要缓存:
    - FFN/MLP: 逐 token 独立, token 间无交互
    - RMSNorm: 逐 token 归一化
    - RoPE: 根据位置算旋转矩阵
    - 残差连接: 只是加法
  - Q 不缓存: attention 计算 Q×K^T, 历史 token 的 Q 不会再被用到
- 状态: 已解决

### [2026-04-10] Attention 计算中各变量的维度?
- 来源: attention 机制基础
- 解答:
  - 单 head, decode 阶段 (d=head_dim, n=历史序列长度):
    - `Q`: (1, d) — 当前 token
    - `K^T`: (d, n) — 历史 K 转置
    - `Q·K^T`: (1, n) — 每个历史 token 一个注意力分数
    - `softmax`: (1, n) — 归一化为概率
    - `·V`: (1, n)×(n, d) = (1, d) — 加权求和得输出
  - 多 head (以 Qwen3 为例: 32 query heads, 8 kv heads, head_dim=128):
    - `Q`: (1, 32, 128), `K`: (n, 8, 128), `V`: (n, 8, 128)
    - GQA: 每 4 个 Q head 共享 1 组 KV, 减少 KV cache 大小
  - Prefill 阶段: Q 变为 (s, d), 注意力矩阵为 (s, s), 计算量远大于 decode
  - 这也是 prefill 为 compute-bound、decode 为 memory-bound 的原因
- 状态: 已解决

### [2026-04-13] @torch.inference_mode() 的作用及与 no_grad 的区别?
- 来源: `nanovllm/engine/model_runner.py` 中 `run_model` / `capture_cudagraph` 的装饰器
- 解答:
  - 作用: 告诉 PyTorch 这段代码不需要梯度, 也不记录任何用于反向传播的信息
  - 与 `torch.no_grad()` 的区别:
    - `no_grad`: 不计算梯度, 但创建的 tensor 仍然是支持 autograd 的普通 tensor(只是 `requires_grad=False`)
    - `inference_mode`: 更彻底, 创建的 tensor 是 `InferenceTensor`, 完全脱离 autograd 系统, 不能参与任何梯度计算
  - `inference_mode` 比 `no_grad` **更快**的原因:
    - 不记录计算图(和 no_grad 一样)
    - 不创建 autograd 的版本计数器(version counter)
    - 不跟踪 tensor 的创建历史
  - 单次操作开销小, 但推理时每个 forward 涉及大量 tensor 操作, 累积差异可观
  - 为什么 nano-vllm 用它: 推理引擎永远不需要梯度, 所有模型前向入口都用 `inference_mode` 是最优选择
- 状态: 已解决

### [2026-04-13] Context 类及 `_CONTEXT` 全局变量是干什么的?
- 来源: `nanovllm/utils/context.py`
- 解答:
  - **定位**: 一个 dataclass + 模块级全局对象, 作为 `ModelRunner` 和 `Attention` 层之间的 attention 元信息中转站
  - **解决的核心矛盾**: flash-attention paged/varlen 版本需要 `cu_seqlens_q/k`、`slot_mapping`、`block_tables`、`context_lens` 等一堆 metadata, 但又不想把这些参数沿着 `Qwen3Model → DecoderLayer → Attention` 一路显式传递, 破坏模型 forward 的 HF 风格干净签名
  - **三个 API 分工**:
    - `set_context(...)`: ModelRunner 在 prepare_prefill/decode 中调用, 整体替换 `_CONTEXT` 对象(非原地修改, 避免脏字段)
    - `get_context()`: Attention 层 forward 中调用, 根据 `is_prefill` 分支选择 `flash_attn_varlen_func` 或 `flash_attn_with_kvcache`
    - `reset_context()`: `ModelRunner.run()` 结束时调用, 清空引用防止 tensor 泄漏, 尤其对 CUDA Graph capture 很重要
  - **Prefill vs Decode 用的字段不同**: prefill 用 `cu_seqlens_q/k`、`max_seqlen_q/k`; decode 用 `context_lens`、`block_tables`; `slot_mapping` 两者都用
- 状态: 已解决

### [2026-04-13] 为什么要在模块顶层写 `_CONTEXT = Context()`?
- 来源: `nanovllm/utils/context.py:16`
- 解答:
  - **空对象模式**: 提供一个字段全为默认值的实例, 让 `get_context()` 永远返回有效对象, 调用方不用判空
  - **让 `global _CONTEXT` 有合法绑定目标**: 首次 `get_context()`(在任何 `set_context` 之前)不会 `NameError`, 静态检查工具也能正确识别
  - **和 `reset_context()` 语义呼应**: 两行完全一致, 顶层初始化就是模块的"初始 reset 状态", 维持不变式——`_CONTEXT` 始终是一个有效的 `Context` 实例
  - **多进程天然隔离**: TP 多进程下每个 worker 独立 import, 各自执行一次顶层代码, 拿到独立副本, 互不干扰
- 状态: 已解决

### [2026-04-13] 只要 import 了 context.py 里的对象, `_CONTEXT` 全局变量就会被创建吗?
- 来源: `nanovllm/utils/context.py` + 模块导入机制
- 解答:
  - **是的, 而且不管 import 的是哪个名字**. `import` 的粒度是**模块**, 不是模块里的某个名字
  - `from X import Y` 的两步: (1) 加载整个模块 X(顶层所有代码跑一遍) (2) 从 X 里取出 Y 绑定到当前命名空间. 所以不论 import 的是 `Context`、`set_context`、`get_context` 还是别的, 顶层的 `_CONTEXT = Context()` 都会被执行
  - **只执行一次**: Python 有 `sys.modules` 模块缓存, 后续任何地方再 import 同一模块直接返回缓存对象, 顶层代码不重复执行. 这正是"全局中转站"能成立的基础——所有 importer 共享同一个 `_CONTEXT`
  - **多进程例外**: TP 子进程是全新的 Python 解释器, `sys.modules` 从零开始, 每个 worker 第一次 import 时会重新执行一次顶层代码, 得到**自己进程空间里独立的一份**
- 状态: 已解决

### [2026-04-13] Python 不像 C++ 有头文件守卫, 多文件多次 import 全局变量会冲突吗?
- 来源: `nanovllm/utils/context.py` + 多文件共享 `_CONTEXT` 的场景
- 解答:
  - **不会冲突**. Python 的 `import` 是对象级缓存, 不是 C/C++ 那种文本粘贴
  - **机制**: `import X` 实际执行
    ```python
    if "X" in sys.modules: return sys.modules["X"]   # 命中直接返回
    else: 新建 module → 放入 sys.modules → 执行顶层代码 → 返回
    ```
    关键点是**先放缓存再跑代码**, 所以一个模块的顶层代码全进程只会执行一次, `_CONTEXT = Context()` 只跑一次
  - **和 C/C++ 的对比**:
    - C/C++ `#include` 是预处理器文本粘贴, 每个翻译单元独立, 必须靠 `#pragma once` / `ifndef` 守卫防止重复定义
    - Python 靠 `sys.modules` 天然去重, 无需守卫
  - **多个文件 import 的结果**: model_runner.py、attention.py、其他文件不论用什么方式 import context, 拿到的都是**同一个 module 对象**, 访问的是**同一份 `_CONTEXT`**. 这正是"ModelRunner 写、Attention 读"能成立的基础
  - **多进程例外(不是冲突)**: TP 子进程是独立 Python 解释器, 各有自己的 `sys.modules`, 每个 worker 独立一份 `_CONTEXT`——这是进程隔离, 不是多次 import 冲突
  - **真正要当心的坑——名字快照 vs 对象引用**: 参见下一条
- 状态: 已解决

### [2026-04-13] `from X import 变量` 的名字快照陷阱, 以及 `get_context()` 访问器模式的深层原因
- 来源: `nanovllm/utils/context.py` 的 `set_context` / `get_context` 设计
- 解答:
  - **本质**: `from X import Y` 等价于 `import X; Y = X.Y`, 是一次**普通赋值**——在当前文件命名空间建一个**新名字** `Y`, 绑定到 import 那一刻 X 里 `Y` 当前指向的对象
  - **两个命名空间里的 `Y` 是两个独立的绑定**, 最初指向同一个对象. 之后 X 模块内部 `global Y; Y = 新对象` 只改**X 的绑定**, 当前文件里那个快照**不会**跟着变——就像拍了张照片, 被拍的人后续走动照片不会更新
  - **对象可变性的区别**:
    - 如果导入的是可变对象, 并**只原地改内部状态**(如 `d["k"] = v`), 两边都能看到(因为两边名字还指向同一个对象)
    - 但只要那边做了**重新赋值**(`global d; d = 新 dict`), 快照就断链
  - **`_CONTEXT` 为什么偏偏踩这个坑**: `set_context` / `reset_context` 的实现都是 `_CONTEXT = Context(...)`——**整体 new 一个新对象再重新绑定**, 不是原地修改. 这恰好命中"快照断链"场景
  - **错误示范**: `from nanovllm.utils.context import _CONTEXT`——attention.py 里的 `_CONTEXT` 永远停在模块加载时的初始空 Context, `set_context` 怎么调都同步不过来, flash-attn 会读到 `is_prefill=False` / `cu_seqlens_q=None` 直接崩
  - **`get_context()` 访问器为什么能解决**:
    - 消费方 `from context import get_context` 快照的是**函数对象本身**, 而 `get_context` 这个名字在 context 模块里从未被重新绑定, 所以快照永远有效
    - 函数体里的 `return _CONTEXT` 在**函数执行时**才做模块作用域查找, 去 context 模块的**当前**命名空间里取 `_CONTEXT` 的最新绑定
    - 于是消费方握住的是一个"稳定入口", 真正的数据每次调用动态取, 完美绕开快照问题
  - **为什么不改成原地修改 `Context` 字段**: (1) 8 个字段写 8 行太啰嗦, 整体替换一行搞定 (2) 未用字段自动回到默认值, 防止脏数据泄漏 (3) `reset_context` 也能一行清空 (4) 语义上每次 set 是"开启新 batch"的原子信号. 一旦选了整体替换策略, 就必然配套用访问器模式
  - **延伸**: 函数、类也有同样的快照问题. monkey-patch 测试翻车常出于此——要么 `import X` 后用 `X.f()` 调用, 要么提供访问器
- 状态: 已解决

### [2026-04-14] `dist.gather` 和 `dist.all_gather` 的区别? 为什么 nano-vllm 的 lm_head 选 gather?
- 来源: `nanovllm/layers/embed_head.py:62-65` 的 `ParallelLMHead.forward`
- 解答:
  - **结果分布差异**:
    - `gather(tensor, gather_list, dst)`: **只有 dst rank** 拿到完整列表, 其它 rank 返回 None
    - `all_gather(tensor_list, tensor)`: **所有 rank** 都拿到完整列表, 结果相同
  - **API 差异**: gather 时非 dst rank 传 `gather_list=None`; all_gather 所有 rank 都要预分配 buffer
  - **通信成本** (单 shard 大小 S, world_size N, NCCL Ring):
    - gather: 总带宽 (N-1)·S, 只 dst 占满
    - all_gather: 总带宽 N·(N-1)·S, 每 rank 都占满 → **比 gather 贵 N 倍**
    - all_reduce: ≈ 2N·(N-1)·S (= reduce_scatter + all_gather)
  - **nano-vllm 为什么选 gather**: lm_head 输出 `[bs, vocab_size]`, Qwen3 vocab≈152064, fp16+bs=512 时完整 logits ≈600MB/rank. 采样只在 rank0 做, 广播给 worker 纯属浪费 —— gather 让 worker 只需发出自己的 shard(~150MB), 不用占 600MB 接收 buffer
  - **对称家族**:
    - Scatter ←→ Gather (1发多 / 多发1)
    - Broadcast ←→ Reduce (广播 / 规约到单点)
    - AllGather ←→ ReduceScatter
    - AllReduce = AllGather + local reduce = ReduceScatter + AllGather
  - **选型口诀**: 带 "All" 前缀 = 结果广播给所有 rank; 不带 "All" = 结果只留在 src/dst 一个 rank. 下一步所有 rank 都要用 → All 版本; 只有一个 rank 消费(采样/日志/checkpoint) → 非 All 版本
  - **和 nano-vllm 里 RowParallelLinear 的对比**: o_proj/down_proj 用 AllReduce 是因为下一个 Block 的每个 rank 都要继续跑 Attention/MLP, 不能缺数据; lm_head 用 gather 是因为它是模型的**最后一层**, 只有 rank0 要把 logits 喂给 sampler
- 状态: 已解决

### [2026-04-14] `weight_loader()` 方法是什么? 为什么要挂到 `nn.Parameter` 上?
- 来源: `nanovllm/layers/linear.py:25-29` 的 `self.weight.weight_loader = self.weight_loader`, 以及 `nanovllm/utils/loader.py` 的 `load_model`
- 解答:
  - **定位**: 每个 Parameter 的"自定义加载钩子" —— 把"如何从 HF 原始权重切出本 rank 那一份"的逻辑以函数形式绑在 `nn.Parameter` 对象上, 让通用的 `load_model()` 可以对所有层统一调用
  - **绑定**: `Module.__init__` 里 `self.weight.weight_loader = self.weight_loader`, 利用 Python 动态属性赋值把方法塞进 Parameter(nn.Parameter 本身没这个属性)
  - **调用**: `load_model()` 遍历 safetensors, 对每个权重名找到对应 param, `getattr(param, "weight_loader")(param, loaded_weight, shard_id?)`
  - **四种实现的分工**:
    - `ReplicatedLinear`: 直接 `param.data.copy_(loaded_weight)`, 不切
    - `ColumnParallelLinear`: `loaded_weight.narrow(0, start_idx, shard_size)` 按 dim=0 切本 rank 那份
    - `RowParallelLinear`: 同上但按 dim=1 切
    - `MergedColumnParallelLinear` / `QKVParallelLinear`: 多一个 `loaded_shard_id` 参数, 把 gate/up 或 q/k/v 分别写入合并后 param 的不同段, 每段再按 tp 切
  - **三个核心好处**:
    1. **loader 与 Module 解耦**: loader 只负责"找到 param", 切分逻辑由 Module 自己决定, 符合开闭原则; 新增 Parallel 类不用改 loader
    2. **避免全量 host→device 传输**: `narrow` 是 view 操作不复制数据, 只有最后 `param_data.copy_()` 才真正传到 GPU —— TP=8 时每 rank 只传 1/8, 节省 PCIe 带宽和显存峰值
    3. **支持 packed modules 一对多映射**: `packed_modules_mapping` 把 HF 原始的 `q_proj/k_proj/v_proj` 映射到合并后的 `qkv_proj` + shard_id("q"/"k"/"v"), `weight_loader` 按 shard_id 把三个独立权重写入同一个合并 param 的不同偏移, 完美支持"训练时分开、推理时合并"
  - **设计模式本质**: "参数自带装配说明书" —— Parameter 不光是一块数据, 还带着"如何从原始文件装配出自己"的知识
- 状态: 已解决

### [2026-04-14] `ColumnParallelLinear` / `RowParallelLinear` 里的 Column/Row 到底切什么?
- 来源: `nanovllm/layers/linear.py:54-73` / `131-153`, Megatron-LM 论文
- 解答:
  - **命名来自数学视角 Y=X·W (W shape=[in,out])**, 但 PyTorch 存储是 W=[out,in], 所以代码里的 `tp_dim` 和数学上的方向相反
  - **Column Parallel = 按输出维(列)切 W**:
    - 数学: `W = [W₀ | W₁]`, 每 rank 持 `[in, out/tp]` 的一"纵条"
    - 代码: `tp_dim=0` (PyTorch 存储 `[out, in]` 里切 out 就是 dim=0), `divide(output_size, tp_size)`
    - **输入**: 完整 X(各 rank 共享) **输出**: 分片(各 rank 只有部分 out 维)
    - **通信**: **无** —— 各 rank 算不同的输出通道, 互不相关
    - **典型用途**: qkv_proj, gate_up_proj (Block 入口)
  - **Row Parallel = 按输入维(行)切 W**:
    - 数学: `W = [W₀; W₁]`, 每 rank 持 `[in/tp, out]` 的一"横条"
    - 代码: `tp_dim=1`, `divide(input_size, tp_size)`
    - **输入**: 分片 X(由上一层 Column 的输出天然分发) **输出**: 形状完整的"部分和"(每 rank 都有 [B, out], 但数值只是局部矩阵乘的一部分)
    - **通信**: **AllReduce** —— 把各 rank 的部分和加起来得到真正的 Y
    - **bias 只在 rank0 加**: 否则 AllReduce 会把 bias 累加 tp_size 次; rank0 加一次, AllReduce 后结果 = `(Y₀+Y₁+...) + b` 正确
    - **典型用途**: o_proj, down_proj (Block 出口)
  - **为什么 Column+Row 配对是最优**:
    ```
    x (完整) → QKV(Column, 无通信) → Q/K/V 各持 num_heads/TP 个头
             → FlashAttention(本地) → out 各持 hidden/TP 的一部分(天然对齐下一层)
             → O_proj(Row, AllReduce) → 完整 hidden
    ```
    一个完整 Attention 块只需 **1 次 AllReduce**. 反过来 Row→Column 需要 3 次通信(reduce_scatter + all_reduce + all_gather), 通信量翻 3 倍
  - **速记表**:

    | | Column Parallel | Row Parallel |
    |---|---|---|
    | 切谁 | W 的 out 维 | W 的 in 维 |
    | 代码 tp_dim | 0 | 1 |
    | 输入 | 完整 | 上层分片 |
    | 输出 | 分片 | 部分和(需 reduce) |
    | 通信 | 无 | AllReduce |
    | 用途 | 入口(qkv/gate_up) | 出口(o/down) |

  - **TP 对上层透明的关键**: 一对 Column+Row 组合后, 入口广播 x / 出口 AllReduce 得到完整 y, 对外看起来就像没有 TP 的普通 Linear 对, 调用者(Attention/MLP)根本不需要感知权重被切了
- 状态: 已解决

### [2026-04-14] torch.distributed 的 gather/scatter 和 ONNX 算子同名是一回事吗?
- 来源: 对照 `embed_head.py:64` 的 `dist.gather` 和 ONNX Gather/Scatter 算子
- 解答:
  - **完全不是一回事, 只是重名**. 处在不同抽象层次:
    - `torch.distributed.gather/scatter`: **集合通信原语**, 跨进程/GPU 搬运数据, 底层走 NCCL/Gloo/MPI, 需要 `init_process_group`
    - `ONNX Gather/Scatter`: **张量索引算子**, 单个 tensor 内部按 indices 取/放元素, 本质是 numpy fancy indexing / CUDA gather-load 和 scatter-store, 单 GPU 就能跑
  - **ONNX Gather 例**: `data[indices]`, 如 `F.embedding(token_ids, weight)` 就是典型的 ONNX Gather(按 token id 查 embedding 表)
  - **ONNX Scatter 例**: `data[indices] = updates`, 如 flash-attention 用 `slot_mapping` 写 KV cache 本质就是一次 Scatter
  - **dist.gather 例**: N 个 rank 各持 shard, 调用后只有 dst rank 拿到完整列表 —— **数据在进程间搬运**
  - **词源**: MPI 标准(1994)定义了 `MPI_Gather/Scatter`; numpy/APL 传统里 gather-load/scatter-store 指 CPU 按索引访存. 两个名字各有合理词源, 到深度学习时代才在同一代码库共存
  - **记忆窍门**: **问"这个操作需要多个 GPU 才能执行吗?"** —— 需要 → 集合通信; 不需要 → 张量算子
  - **nano-vllm 里同时出现的例子**: `VocabParallelEmbedding.forward` 先做 ONNX 风格的 Gather (`F.embedding`, 按 token id 查表), 再做集合通信风格的 AllReduce (跨 rank 聚合), 正好是两个抽象层次叠加
- 状态: 已解决

### [2026-04-14] "按 TP 切分" 到底是按什么切?
- 来源: `nanovllm/engine/model_runner.py:128` 的 `num_kv_heads // self.world_size`, 以及文档 05-tensor-parallel.md
- 解答:
  - **TP = Tensor Parallel (张量并行)**, "按 TP 切分" = 按 `tensor_parallel_size` 把一个张量均匀分成 N 份, 每个 rank 只持有并计算自己那一份
  - **切什么**:
    - 权重: Column 切按输出维(dim=0), Row 切按输入维(dim=1), 见 `linear.py:63/140` 的 `divide(size, tp_size)`
    - KV cache: 按 `num_kv_heads` 切, 每 rank 只存 `num_kv_heads // TP` 个 head 的 K/V, 显存占用 ÷TP
    - Embedding/LMHead/Norm: 不切, 各 rank 复制(参数小, 不值得切)
  - **目的不是提速, 是扩显存**: 7B 模型权重 14GB, 单卡 24GB 放不下大 KV cache; TP=2 后每卡只需 7GB 权重, 剩下 17GB 全给 KV cache, 能跑的 batch/上下文翻倍. 速度提升会被 AllReduce 通信吃掉一部分
  - **和 DP/PP 的区别**: TP 切一层内的权重(每层 forward 内 AllReduce); PP 按层切(层间传 activation); DP 不切权重只切 batch(推理不用)
  - **nano-vllm 只实现 TP**: 推理场景最能解决"单卡装不下", 且 NVLink 上的 AllReduce 开销可接受
- 状态: 已解决

### [2026-04-16] `torch.narrow` 到底做什么? 为什么 weight_loader 里到处用它?
- 来源: `nanovllm/layers/linear.py:69/91/126/146` 的 `loaded_weight.narrow(...)` / `param_data.narrow(...)`
- 解答:
  - **定位**: 在指定维度上取一段连续切片, **返回视图 (view)** 而非副本 —— 和原 tensor 共享存储, 零拷贝
  - **签名**: `tensor.narrow(dim, start, length)` → 在 `dim` 维上从 `start` 截取长度为 `length` 的子张量, 形状除 `dim` 外其它维都不变
  - **等价写法**: `narrow(0, s, L)` ≡ `tensor[s:s+L]`, `narrow(1, s, L)` ≡ `tensor[:, s:s+L]`. 差别只在于 narrow 的 dim 是参数, 可以**动态**指定(代码里 `self.tp_dim` 是 0 或 1, 用 narrow 一行写完; 用切片语法要 `if tp_dim==0: t[s:s+L] else: t[:, s:s+L]`)
  - **view 语义的关键**:
    - 返回值和原 tensor 共享存储, **写入会传染**. 所以 nano-vllm 里 `param_data = param.data.narrow(...)` 后 `param_data.copy_(loaded_weight)` 能把数据写回原 param 的对应段 —— 这是"把 shard 写入合并权重的特定段"能成立的基础
    - 必须是连续段, 不能跳步. `narrow(0, 2, 5)` 取 index 2..6; 如果要不连续索引得用 `index_select` (会拷贝)
  - **nano-vllm 里的两种用法**:
    1. **切"远端"的 loaded_weight**: `ColumnParallelLinear.weight_loader` 里 `loaded_weight.narrow(tp_dim, start_idx, shard_size)` —— 从 HF 的完整权重里**只取本 rank 那一段**再 copy 到 GPU, 避免把全量数据搬到显存 (TP=8 时每 rank 只传 1/8)
    2. **切"本地"的 param_data**: `MergedColumnParallelLinear.weight_loader` 里 `param_data = param_data.narrow(tp_dim, shard_offset, shard_size)` —— 先把合并 param 的**目标段**取成 view, 再 copy 到这个 view 就等于写入合并 param 的指定偏移. 这是 gate/up 或 q/k/v 分别加载到同一 param 不同段的实现手段
  - **为什么不用 `chunk` / `split`**:
    - `chunk(n, dim)` 一次切成 n 份, 要取第 k 份得 `chunk(...)[k]`, 内部同样是 view; nano-vllm 里**两种都用了** —— `loaded_weight.chunk(tp_size, tp_dim)[tp_rank]` 用来按 tp 均匀切, `narrow` 用来按任意偏移切(非均等的 shard_offset/shard_size)
    - narrow 的优势是**任意 `(start, length)`**, 适合 merged param 里每段偏移不一的情况; chunk 只能等分
  - **和 `torch.slice` / `__getitem__` 的关系**: 底层都是 as_strided, 语义等价. narrow 更"显式", 多参数形式更适合在循环/动态 dim 场景里写
  - **坑**: (1) narrow 返回 view, 如果对返回值做 in-place 修改会改到原 tensor, 反之亦然 —— 这正是 weight_loader 故意利用的特性, 但调试时要留意 (2) view 不连续时(比如 narrow 之后想做 reshape), 可能需要 `.contiguous()` 显式拷贝
- 状态: 已解决

