# KV Cache 与内存管理

## 一句话总结

BlockManager 将 GPU 显存划分为固定大小的物理块（256 tokens/块），通过引用计数实现块的共享与回收，
并利用 xxhash 链式哈希实现前缀缓存——相同前缀的序列可直接复用已有的 KV cache，跳过重复计算。

## 整体架构

```
                    ModelRunner.allocate_kv_cache()
                    计算可用块数, 分配 GPU tensor
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  GPU 显存: kv_cache tensor                              │
│  shape: [2, num_layers, num_blocks, block_size,         │
│          num_kv_heads, head_dim]                         │
│  2 = K cache + V cache                                  │
└─────────────────────────────────────────────────────────┘
                              ▲
                              │ block_table[i] 索引
                              │
┌─────────────────────────────────────────────────────────┐
│  BlockManager (CPU 端逻辑管理)                            │
│                                                         │
│  blocks[]          ── Block 对象数组, 按 block_id 索引     │
│  free_block_ids    ── 空闲块队列 (deque)                  │
│  used_block_ids    ── 正在使用的块集合 (set)               │
│  hash_to_block_id  ── 链式哈希 → block_id 映射表          │
└─────────────────────────────────────────────────────────┘
```

**关键点**：BlockManager 本身不持有 GPU tensor，它只管理逻辑映射（block_id → 物理位置）。
真正的 KV cache tensor 由 `ModelRunner.allocate_kv_cache()` 一次性分配，
再通过 `module.k_cache = self.kv_cache[0, layer_id]` 绑定到每层 Attention。

## 核心数据结构

### Block (block_manager.py:8-27)

```python
class Block:
    block_id: int      # GPU KV cache 数组中的索引
    ref_count: int     # 引用计数: 有多少序列共享此块, 归零可回收
    hash: int          # 链式哈希, -1 表示块未填满(不可缓存)
    token_ids: list    # 块中的 token 内容, 用于缓存命中时二次校验(防哈希碰撞)
```

### Sequence 中的内存相关字段 (sequence.py)

```python
class Sequence:
    block_table: list[int]     # 该序列占用的物理块 ID 列表
    num_cached_tokens: int     # 前缀缓存命中的 token 数(这些不需要重新计算)
    block_size: int = 256      # 类变量, 每块容纳的 token 数

    # 派生属性:
    num_cached_blocks → num_cached_tokens // block_size
    num_blocks        → ceil(num_tokens / block_size)
    last_block_num_tokens → num_tokens - (num_blocks-1) * block_size
```

## 核心流程

### 1. KV Cache 预分配 (model_runner.py:100-118)

启动时 ModelRunner 计算可用块数并分配 GPU tensor：

```
可用显存 = 总显存 × gpu_memory_utilization - 已用显存 - 峰值显存 + 当前显存
单块字节 = 2(KV) × num_layers × block_size × num_kv_heads × head_dim × dtype_size
num_blocks = 可用显存 // 单块字节
```

这个 `num_blocks` 写入 `config.num_kvcache_blocks`，Scheduler 创建 BlockManager 时使用。

### 2. 块分配 — allocate(seq) (block_manager.py:68-103)

Prefill 阶段，Scheduler 调用 `block_manager.allocate(seq)` 为新序列分配块：

```
对序列的每个 block_size 大小的 token 段:
  │
  ├─ 块已满? → 计算链式哈希 h = xxh64(前一块哈希 + 当前块tokens)
  │              │
  │              ├─ hash_to_block_id 命中 且 token_ids 匹配?
  │              │    ├─ YES, 块在 used_block_ids → ref_count++ (共享)
  │              │    ├─ YES, 块在 free_block_ids → 重新 allocate (幸运复活)
  │              │    └─ NO  → cache_miss = True, 后续块全部跳过缓存查找
  │              │
  │              └─ 命中时: seq.num_cached_tokens += block_size
  │
  └─ 块未满(最后一个块) → hash = -1, 不可缓存
  │
  └─ cache_miss? → 从 free_block_ids[0] 分配新块
  │
  └─ 满块注册: block.update(h, token_ids); hash_to_block_id[h] = block_id
  └─ seq.block_table.append(block_id)
```

**链式哈希的关键性质**：一旦某个块未命中，后续所有块都不可能命中（因为哈希依赖前一块），
所以 `cache_miss = True` 后直接跳过缓存查找，是正确且高效的剪枝。

### 3. 追加 token — may_append(seq) (block_manager.py:121-144)

Decode 阶段每生成一个 token，Scheduler 调用 `may_append(seq)` 维护 block_table：

```
                    len(seq) % block_size
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        == 1            == 0           其他
   溢出到新块         刚好填满        块还有空位
        │               │               │
   分配新空块       计算哈希并注册      无需操作
   追加到          到 hash_to_block_id   (hash=-1)
   block_table     供后续序列复用
```

`can_append()` 的巧妙之处：只有 `len(seq) % block_size == 1` 时才需要 1 个新块，其他时候需要 0 个。

### 4. 块释放 — deallocate(seq) (block_manager.py:105-114)

序列完成或被抢占时调用：

```
倒序遍历 seq.block_table:
  block.ref_count--
  if ref_count == 0:
    _deallocate_block → 从 used_block_ids 移到 free_block_ids 尾部
```

**倒序释放的原因**：靠前的块更可能是公共前缀（被多个序列共享，ref_count > 1），
倒序遍历优先释放靠后的独占块，保留高复用价值的前缀块。

**释放到队尾**：`free_block_ids.append(block_id)` 将块追加到队尾，
而分配时从 `free_block_ids[0]`（队头）取块。这种 FIFO 策略使释放的块尽量晚被复用，
给前缀缓存更多存活机会——即使块被释放，其内容仍在 GPU 上，后续序列仍可能命中。

### 5. 抢占 — preempt (scheduler.py:60-63)

当 decode 阶段内存不足（`can_append()` 返回 False）时：

```
running 队列中最后加入的序列被踢出:
  1. deallocate(seq)  → 释放其所有块
  2. seq.status = WAITING
  3. waiting.appendleft(seq)  → 放回等待队首, 优先重新调度
```

这是 **最简单的抢占策略**（踢最新的），被抢占的序列重新进入 prefill 阶段时需要重新分配块，
但前缀缓存可能帮它恢复部分 KV cache。

## 代码锚点

| 功能 | 位置 |
|------|------|
| Block 类定义 | block_manager.py:8-27 |
| BlockManager 初始化 | block_manager.py:31-36 |
| 链式哈希计算 | block_manager.py:39-47 |
| allocate (prefill 块分配) | block_manager.py:68-103 |
| deallocate (块释放) | block_manager.py:105-114 |
| can_append / may_append | block_manager.py:116-144 |
| KV cache tensor 分配 | model_runner.py:100-118 |
| KV cache 绑定到 Attention 层 | model_runner.py:113-118 |
| block_size 定义 | config.py:17 (`kvcache_block_size = 256`) |
| Sequence.block_table 定义 | sequence.py:26 |
| 抢占逻辑 | scheduler.py:46-51, 60-63 |

## 关键设计决策

### 为什么 block_size = 256？（对比 vLLM 的 16）

nano-vllm 用 256 而 vLLM 默认 16，两者权衡不同：
- **大块 = 更少的块管理开销**：序列的 block_table 更短，哈希计算次数更少
- **大块 = 前缀缓存粒度更粗**：两个序列需要共享至少 256 个连续相同 token 才能命中，
  短前缀（如 system prompt < 256 tokens）无法被缓存
- **大块 = 内部碎片更大**：最后一个块平均浪费 128 tokens 的空间
- `config.py:22` 限制 `kvcache_block_size % 256 == 0`，配合 flash-attention 的对齐要求

### xxhash 链式哈希做前缀缓存

**优点**：
- O(1) 查找，极快（xxhash 本身是非加密哈希，速度远超 SHA）
- 链式设计自动编码了"从序列开头到当前位置"的完整前缀信息
- 内存开销极小：每块只存一个 int64 哈希 + token_ids 用于碰撞校验

**缺点/风险**：
- 哈希碰撞：虽然概率极低（64-bit），但代码做了 `token_ids` 二次校验兜底
- 块必须完全填满才能参与缓存（`hash = -1` 表示未填满），丢失了最后一个不满块的缓存机会

### 为什么用 ref_count 而不是 copy-on-write？

ref_count 是够用的最简方案：
- nano-vllm 的块内容一旦写入就不会修改（KV cache 只追加不覆盖）
- 因此不存在"写时复制"的场景——不需要 COW
- ref_count 只需一个整数，无锁（单线程 CPU 端调度），无额外内存拷贝

## 端到端示例

假设 block_size = 256，两个序列共享 512 tokens 的 system prompt：

```
Seq A: [system_prompt(512) + user_A(100)]  共 612 tokens → 3 blocks
Seq B: [system_prompt(512) + user_B(200)]  共 712 tokens → 3 blocks

=== Seq A allocate ===
block 0: tokens[0:256]   → hash_0 = xxh64(tokens[0:256])     → 新分配 block#5
block 1: tokens[256:512] → hash_1 = xxh64(hash_0 + tokens)   → 新分配 block#6
block 2: tokens[512:612] → 未满, hash=-1                      → 新分配 block#7
seq_A.block_table = [5, 6, 7]
seq_A.num_cached_tokens = 0 (首次, 无缓存可命中)

=== Seq B allocate ===
block 0: tokens[0:256]   → hash_0 命中 block#5, token_ids 匹配 → ref_count=2, 共享!
block 1: tokens[256:512] → hash_1 命中 block#6, token_ids 匹配 → ref_count=2, 共享!
block 2: tokens[512:712] → 未满 + 前两块已命中, 但内容不同 → cache_miss, 新分配 block#8
seq_B.block_table = [5, 6, 8]
seq_B.num_cached_tokens = 512 (前两块命中, 跳过 512 tokens 的 prefill 计算!)

=== GPU 显存实际使用 ===
block#5: 被 A 和 B 共享 (ref_count=2), 只存一份
block#6: 被 A 和 B 共享 (ref_count=2), 只存一份
block#7: A 独占
block#8: B 独占
总计: 4 blocks (而非 6 blocks), 节省 33% 显存

=== Seq A 完成, deallocate ===
倒序: block#7(ref=1→0, 释放) → block#6(ref=2→1, 保留) → block#5(ref=2→1, 保留)
Seq B 仍可正常使用 block#5 和 block#6
```

## 遗留问题

- [ ] 大 block_size (256) 对短 system prompt 场景的前缀缓存命中率影响有多大？
- [ ] 抢占策略只踢最新序列，是否存在饥饿问题？（同一序列反复被抢占）
- [ ] `_allocate_block` 用 `free_block_ids.remove(block_id)` 是 O(n) 操作，大量块时是否成为瓶颈？
- [ ] 哈希碰撞的实际概率和 token_ids 校验的性能开销是否值得 benchmark？
