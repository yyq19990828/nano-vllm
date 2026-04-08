# 采样策略

## 一句话总结
<!-- Gumbel-max 采样的原理和实现 -->


## 核心流程

```
Sampler(logits, sampling_params):
  1. temperature scaling: logits / temperature
  2. Gumbel-max trick:
     - gumbel_noise = -log(-log(uniform_random))
     - token_id = argmax(logits + gumbel_noise)
  3. 当 temperature = 0 时退化为 greedy (argmax)
```

## 代码锚点
- Sampler 实现: sampler.py:L___
- torch.compile 装饰: sampler.py:L___

## 关键设计决策
<!--
- 为什么用 Gumbel-max 而不是标准的 softmax + multinomial?
- torch.compile 对采样性能的影响有多大?
- 为什么没有实现 top-k / top-p?
-->


## 我的理解 / 类比


## 遗留问题
- [ ] 
