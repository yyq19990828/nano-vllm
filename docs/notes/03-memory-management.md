# KV Cache 与内存管理

## 一句话总结
<!-- BlockManager 如何管理 GPU 显存中的 KV Cache -->


## 核心流程

```
Block 分配:
1. allocate(seq)
   - 对每个 block 大小的 token 段计算 xxhash
   - hash 命中 → 复用已有物理块 (ref_count++)
   - hash 未命中 → 从 free_block_ids 分配新块
   - 记录 seq.block_table = [物理块ID列表]

2. may_append(seq)  (decode 阶段)
   - 当前块写满 → 分配新块追加到 block_table

3. deallocate(seq)  (序列完成)
   - ref_count-- , 归零时回收到 free_block_ids
```

## 代码锚点
- allocate: block_manager.py:L___
- may_append: block_manager.py:L___
- deallocate: block_manager.py:L___
- prefix caching (xxhash): block_manager.py:L___
- block_size 定义: block_manager.py:L___ 或 config.py:L___

## 关键设计决策
<!--
- 为什么 block_size = 256? (对比 vLLM 的 16)
- xxhash 做前缀缓存的优缺点?
- 为什么用 ref_count 而不是直接 copy-on-write?
-->


## 我的理解 / 类比


## 遗留问题
- [ ] 
