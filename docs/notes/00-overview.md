# 整体架构速查

## 一句话总结
<!-- 用自己的话概括整个系统 -->


## 核心数据流
```
LLM.generate(prompts)
  → tokenize → 创建 Sequence
  → while not finished:
      scheduler.schedule()    → 决定处理哪些序列 (prefill or decode)
      model_runner.run()      → GPU 前向计算得到 next token
      scheduler.postprocess() → 追加 token, 检查终止
  → decode → text
```

## 关键概念速查

| 概念 | 含义 | 代码位置 |
|------|------|----------|
| Prefill | 首次处理完整 prompt, 填充 KV Cache | scheduler.py |
| Decode | 自回归逐 token 生成 | scheduler.py |
| Block | 256 token 为单位的 KV Cache 物理块 | block_manager.py |
| Prefix Caching | 相同前缀的序列共享 KV 块 (xxhash) | block_manager.py |
| CUDA Graph | 预录制 decode 计算图, replay 减少开销 | model_runner.py |
| Context | engine 和 layers 间传递注意力参数的全局对象 | utils/context.py |

## 我的理解 / 类比
<!-- 用自己的语言重述, 可以类比其他你熟悉的系统 -->


## 遗留问题
- [ ] 
