# 采样策略

## 一句话总结
用 **Exponential / Gumbel-max trick** 在一行 GPU 算子里完成 "按 softmax 概率采样", 避免 `torch.multinomial` 的 CPU 同步与额外内核, 全程 `@torch.compile` 融合。

## 核心流程

```
Sampler(logits[B, V], temperatures[B]):
  1. temperature scaling:
       logits = logits.float() / temperatures[:, None]
  2. softmax → probs
  3. Exponential trick (等价于 Gumbel-max):
       E ~ Exp(1)                         # torch.empty_like(probs).exponential_(1)
       token_id = argmax(probs / E)       # 一步出结果
  4. temperature 越小, logits 被放大, softmax 越尖锐, 越接近 greedy
     (代码里通过 assert temperature > 1e-10 禁止真正的 0)
```

**为什么 `argmax(p / E)` 等于按 p 采样?**
令 `E_i ~ Exp(1)` 独立同分布, 则 `-log E_i ~ Gumbel(0, 1)`。所以
`argmax_i (p_i / E_i) = argmax_i (log p_i − log E_i) = argmax_i (log p_i + g_i)`,
正是经典的 Gumbel-max trick — 输出严格服从 Categorical(p)。

## 代码锚点
- Sampler 实现: [nanovllm/layers/sampler.py:5-15](../../nanovllm/layers/sampler.py#L5-L15)
- `@torch.compile` 装饰: [nanovllm/layers/sampler.py:10](../../nanovllm/layers/sampler.py#L10)
- 调用入口 (rank0 only): [nanovllm/engine/model_runner.py:252-254](../../nanovllm/engine/model_runner.py#L252-L254)
- 温度收集: [nanovllm/engine/model_runner.py:217-222](../../nanovllm/engine/model_runner.py#L217-L222)
- `temperature` 字段来源: [nanovllm/sampling_params.py:6](../../nanovllm/sampling_params.py#L6), 在 [nanovllm/engine/sequence.py:27](../../nanovllm/engine/sequence.py#L27) 拷贝到 Sequence

## 关键设计决策

### 1. 为什么用 Gumbel/Exponential trick 而不是 `torch.multinomial`?
- `multinomial` 内部要做累积分布 + 二分查找, 还会触发隐式 CPU 同步, 在小 batch decode 时延迟感人。
- Gumbel-max 写成 `probs / E` + `argmax`, 全是 elementwise + reduce, 可以被 `torch.compile` 融合成单个 (或极少数) Triton kernel, 无需同步。
- 对每个序列都可以使用**不同的温度** (因为 batch 内不同请求的 `sampling_params` 可能不同), 实现上只是 `logits / temperatures.unsqueeze(1)` 一行。

### 2. `@torch.compile` 的作用
- 第一次调用触发编译, 把 div / softmax / exponential_ / div / argmax 这一串融合, 减少 kernel launch。
- 由于输入 shape 只有 `[B, V]`, 在 batch_size 变化时可能触发重编译; 但 V (词表) 固定, B 的取值集合也是 CUDA Graph 那批 (`[1,2,...,512]`), 重编译次数有限。

### 3. 为什么没有实现 top-k / top-p?
- 项目定位是 "**~1200 行的 minimal vLLM**", 优先展示批调度 / KV cache / TP / CUDA Graph 这些**系统**特性, 把采样砍到只剩温度。
- top-k/top-p 需要额外的 sort 或 scan, 会破坏当前单 kernel 的简洁性; 真要加, 一般在 `softmax` 之前对 `logits` 做 in-place mask 即可。

### 4. greedy 的处理
- 严格 `temperature == 0` 在公式里会除零, 因此 `SamplingParams.__post_init__` 里 `assert temperature > 1e-10`。
- 用户想要 "贪心" 时传一个极小温度 (例如 `1e-6`): logits 被放大到 softmax 几乎是 one-hot, Exponential 噪声乘进去也压不住最大值, 行为等价于 argmax。

### 5. 只在 rank0 采样
- 多卡张量并行时, 各 rank 的 `logits` 在 `LMHead` 内部已经 `all_gather` 成完整词表, 但只有 rank0 跑采样并广播 `token_id`, 避免重复采样导致不同 rank 抽到不同 token。
- 对应代码: `model_runner.py:252-254` 里 `if self.rank == 0` 的判断。

## 我的理解 / 类比
- Gumbel-max trick ≈ "给每个候选独立加一份噪声, 噪声服从 Gumbel 分布, 然后选最大的那个"。直觉是: 概率高的 logit 即使被噪声打压也大概率赢, 概率低的偶尔靠噪声翻盘 — 长期频率正好等于 softmax 概率。
- Exponential 形式 (`p / E`) 是同一件事的另一种写法, 数值上更容易在 GPU 上一行算完。
- 类比 "抽签": 给每个 token 发一根长度 ~ `Exp(1)/p_i` 的签, 谁的签**最短**谁中 (即 `p_i / E_i` 最大)。概率越大的 token 平均签越短, 中签率正好是 `p_i`。

## 遗留问题
- [ ] `@torch.compile` 在 batch_size 变化频繁时, 重编译开销 vs. 节省的 launch 开销, 实际收益多大? 是否有 benchmark?
- [ ] 如果未来要支持 top-p, 在保持 `torch.compile` 友好的前提下, 最优实现是 `sort + cumsum + mask` 还是 `radix select`?
- [ ] 多卡场景下让所有 rank 各自采样 + 用同一个 seed 是否比 "rank0 采样 + 广播" 更省一次通信?
