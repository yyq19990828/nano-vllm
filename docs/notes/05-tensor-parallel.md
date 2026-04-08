# 张量并行

## 一句话总结
<!-- 多 GPU 推理如何通过张量并行实现 -->


## 核心流程

```
多 GPU 协作:

Rank 0 (主进程):
  - 运行 LLMEngine 全部逻辑 (调度、采样)
  - 通过 SharedMemory 写入指令 (序列化)
  - 通过 Event 通知其他 rank

Rank 1+ (工作进程):
  - Event.wait() 等待指令
  - 反序列化, 执行相同的模型前向计算
  - NCCL AllReduce 聚合结果

权重切分策略:
  - ColumnParallel: QKV, Gate, Up → 按输出维度切分
  - RowParallel: O_proj, Down → 按输入维度切分, AllReduce 聚合
```

## 代码锚点
- 多进程启动: model_runner.py:L___
- SharedMemory 通信: model_runner.py:L___
- ColumnParallelLinear: linear.py:L___
- RowParallelLinear: linear.py:L___
- packed_modules_mapping: qwen3.py:L___

## 关键设计决策
<!--
- 为什么用 SharedMemory + Event 而不是 torch.distributed?
- 为什么 QKV 用 Column 切, O_proj 用 Row 切?
- Megatron-LM 风格的张量并行和这里的实现有什么异同?
-->


## 我的理解 / 类比


## 遗留问题
- [ ] 
