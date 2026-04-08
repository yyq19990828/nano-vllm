"""
nano-vllm 学习调试入口
用法: 直接运行或在 VS Code 中 F5 调试

调试建议的断点位置 (按学习阶段):
  阶段1 - 请求生命周期:
    - engine/llm_engine.py  → generate() 入口, step() 主循环
  阶段2 - 调度:
    - engine/scheduler.py   → schedule(), postprocess()
  阶段3 - 内存管理:
    - engine/block_manager.py → allocate(), may_append(), deallocate()
  阶段4 - GPU 执行:
    - engine/model_runner.py  → prepare_prefill(), prepare_decode(), run_model()
  阶段5 - 注意力:
    - layers/attention.py     → forward()
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# 配置区 - 根据你的环境修改
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), ".huggingface", "Qwen3-0.6B")
ENFORCE_EAGER = True    # True=跳过 CUDA Graph, 方便断点调试
TENSOR_PARALLEL = 1     # 单卡调试用 1


def test_single_prompt():
    """最简单的单条推理, 适合跟踪完整生命周期"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, enforce_eager=ENFORCE_EAGER, tensor_parallel_size=TENSOR_PARALLEL)

    prompt = "What is 1+1?"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 断点打在这里, 然后 Step Into 进入 generate()
    outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=32))

    print(f"Prompt: {prompt!r}")
    print(f"Output: {outputs[0]['text']!r}")


def test_batch_scheduling():
    """多条不同长度的 prompt, 观察调度器行为"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, enforce_eager=ENFORCE_EAGER, tensor_parallel_size=TENSOR_PARALLEL)

    prompts_raw = [
        "Hi",                               # 短 prompt
        "Explain quicksort step by step",    # 中等 prompt
        "Write a poem about the ocean " * 5, # 长 prompt
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_raw
    ]

    # 观察: scheduler 如何分批处理不同长度的序列
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=64))

    for raw, output in zip(prompts_raw, outputs):
        print(f"\nPrompt: {raw[:50]}...")
        print(f"Output: {output['text'][:100]}...")


def test_prefix_caching():
    """相同前缀的 prompt, 观察 prefix caching 是否命中"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, enforce_eager=ENFORCE_EAGER, tensor_parallel_size=TENSOR_PARALLEL)

    shared_prefix = "You are a helpful assistant. Answer concisely.\n\n"
    prompts_raw = [
        shared_prefix + "What is Python?",
        shared_prefix + "What is Rust?",
        shared_prefix + "What is Go?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_raw
    ]

    # 在 block_manager.py 的 allocate() 打断点
    # 观察第 2、3 条是否复用了第 1 条的 block
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))

    for raw, output in zip(prompts_raw, outputs):
        print(f"\nPrompt: ...{raw[len(shared_prefix):]}")
        print(f"Output: {output['text'][:100]}")


def test_greedy_vs_sample():
    """对比 greedy 和 sampling 的输出差异"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, enforce_eager=ENFORCE_EAGER, tensor_parallel_size=TENSOR_PARALLEL)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me a joke"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 在 sampler.py 打断点, 对比两种模式下的采样逻辑
    greedy_out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64))
    sample_out = llm.generate([prompt], SamplingParams(temperature=0.8, max_tokens=64))

    print(f"Greedy:  {greedy_out[0]['text'][:100]}")
    print(f"Sampled: {sample_out[0]['text'][:100]}")


# ============================================================
# 入口 - 选择要调试的测试
# ============================================================
if __name__ == "__main__":
    # 取消注释你想调试的测试:
    test_single_prompt()
    # test_batch_scheduling()
    # test_prefix_caching()
    # test_greedy_vs_sample()
