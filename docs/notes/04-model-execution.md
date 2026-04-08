# 模型执行与 CUDA Graph

## 一句话总结
<!-- ModelRunner 如何在 GPU 上执行模型推理 -->


## 核心流程

```
ModelRunner.run(seqs, is_prefill):
1. 准备输入张量
   - prefill: prepare_prefill() → 拼接所有序列 token, 构建 cu_seqlens
   - decode:  prepare_decode()  → 每个序列只取最后一个 token

2. 执行模型
   - prefill → 始终 Eager 执行 (变长, 无法 graph 化)
   - decode  → enforce_eager? Eager : CUDA Graph replay
     - batch_size 向上取整到预录制尺寸 [1,2,4,8,16,...,512]

3. 采样
   - Sampler (Gumbel-max) → next token_ids
```

## 代码锚点
- prepare_prefill: model_runner.py:L___
- prepare_decode: model_runner.py:L___
- run_model: model_runner.py:L___
- CUDA Graph 录制: model_runner.py:L___
- CUDA Graph replay: model_runner.py:L___
- KV Cache 分配: model_runner.py:L___

## 关键设计决策
<!--
- 为什么 prefill 不用 CUDA Graph?
- CUDA Graph 的 batch size 选择策略 (为什么是这些尺寸)?
- padding 到固定 batch size 的性能影响?
-->


## 我的理解 / 类比


## 遗留问题
- [ ] 
