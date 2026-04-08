# Attention 与 KV Cache

## 一句话总结
<!-- Flash-Attention 如何与 KV Cache 配合工作 -->


## 核心流程

```
Attention.forward(q, k, v):

1. Triton kernel: store_kvcache()
   - 按 slot_mapping 将 k, v 写入物理 KV Cache

2. 计算注意力
   - Prefill: flash_attn_varlen_func
     - 变长序列拼接, cu_seqlens 标记边界
     - 支持 block_table (前缀缓存场景)
   - Decode: flash_attn_with_kvcache
     - 只有 1 个 query token
     - 从 KV Cache 读取历史 K/V
```

## 代码锚点
- store_kvcache_kernel: attention.py:L___
- prefill attention: attention.py:L___
- decode attention: attention.py:L___
- Context 读取: attention.py:L___ (get_context)

## 关键设计决策
<!--
- 为什么 prefill 和 decode 用不同的 flash-attn API?
- Triton kernel 写 KV Cache 相比直接 index_put_ 有什么优势?
- slot_mapping 是如何从逻辑位置映射到物理位置的?
-->


## 我的理解 / 类比


## 遗留问题
- [ ] 
