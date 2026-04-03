# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines of Python) providing high-performance LLM inference with batch optimization, prefix caching, tensor parallelism, and CUDA graphs. Currently supports Qwen2/Qwen3 models. API mirrors vLLM's interface.

## Commands

```bash
# Install locally (editable)
pip install -e .

# Run basic generation example
python example.py

# Run throughput benchmark (256 sequences)
python bench.py
```

No test suite exists. Validate changes via `example.py` and `bench.py`.

## Architecture

**Request flow:** `LLM.generate()` → `LLMEngine` (tokenize, loop: schedule → run model → postprocess) → decoded outputs

**Key components:**

- **`nanovllm/llm.py`** - Thin user-facing wrapper around LLMEngine
- **`nanovllm/engine/llm_engine.py`** - Main orchestration: tokenize, schedule, run, decode loop
- **`nanovllm/engine/scheduler.py`** - Two-phase scheduling (prefill then decode), preemption when memory-constrained. Sequence states: WAITING → RUNNING → FINISHED
- **`nanovllm/engine/block_manager.py`** - Block-based KV cache allocation (256 tokens/block) with xxhash prefix caching
- **`nanovllm/engine/model_runner.py`** - GPU execution: model loading, KV cache allocation, CUDA graph capture/replay, tensor parallelism via multiprocessing + SharedMemory
- **`nanovllm/engine/sequence.py`** - Sequence dataclass (token_ids, block_table, status)
- **`nanovllm/models/qwen3.py`** - Qwen3ForCausalLM (also handles Qwen2). Only model implementation currently; new models go here
- **`nanovllm/layers/`** - Reusable layers: flash-attention with KV cache (`attention.py`), tensor-parallel linear layers (`linear.py`), RMSNorm with residual fusion (`layernorm.py`), RoPE (`rotary_embedding.py`), compiled Gumbel-max sampler (`sampler.py`)
- **`nanovllm/utils/context.py`** - Thread-local context passing attention parameters between engine and layers
- **`nanovllm/utils/loader.py`** - Safetensors weight loading with tensor-parallel sharding

**Two execution modes:**
- **CUDA Graph** (default): pre-recorded graphs for batch sizes [1,2,4,8,16..512 step 16]
- **Eager** (`enforce_eager=True`): standard PyTorch, useful for debugging

**Tensor parallelism:** Rank 0 orchestrates via SharedMemory + Events; all ranks do NCCL collective ops. Linear layers split column-wise (QKV, gate/up) or row-wise (output projections).

## Dependencies

Python 3.10-3.12, PyTorch >= 2.4, triton >= 3.0, transformers >= 4.51, flash-attn, xxhash.

## Adding a New Model

1. Create `nanovllm/models/<name>.py` with a `ForCausalLM` class following the Qwen3 pattern
2. Define `packed_modules_mapping` for weight sharding
3. Update the import in `model_runner.py` to select the model class based on config
