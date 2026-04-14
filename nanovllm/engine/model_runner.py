import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """GPU 执行器: 连接 Engine/Scheduler、BlockManager/Sequence、Model/Attention 三大模块的核心桥梁
    职责: 模型加载、KV cache 分配、输入数据准备(Sequence → GPU tensor)、CUDA Graph 管理、多 GPU 通信"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size #* 进程数, 需要根据 GPU 数量调整
        self.rank = rank #* 进程编号, 从 0 到 world_size-1
        self.event = event  #* rank0 持有 list[Event] 用于通知各 worker; worker 持有单个 Event 用于等待

        #--- 阶段1: 初始化分布式通信 & GPU 设备 ---#
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        #--- 阶段2: 临时切换默认 dtype/device, 确保后续创建的 tensor 都在 GPU 上且精度正确 ---#
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        #--- 阶段3: 模型构建与加载 ---#
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        #--- 阶段4: warmup → allocate_kv_cache → capture_cudagraph ---#
        #* 顺序不可变: warmup 测出峰值显存 → 据此计算 KV cache 可用块数 → 最后录制 CUDA Graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        #--- 阶段5: 恢复默认 dtype/device, 避免后续非模型代码意外在 GPU 上创建 tensor ---#
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        #--- 阶段6: 多 GPU 时, rank0 创建共享内存用于广播指令, worker 进入事件循环等待 ---#
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  #* 1MB 共享内存
                dist.barrier()  #* 等待所有 worker 准备好
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm") #* 其余 worker 连接到同一块共享内存
                self.loop()  #* worker 阻塞在此, 等待 rank0 通过 shm 分发指令

    def exit(self):
        """清理资源: 关闭共享内存、释放 CUDA Graph、销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  #* 只有创建者(rank0)负责删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker 事件循环: 阻塞等待 rank0 通过 shm 分发方法名和参数, 然后执行"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Worker 从共享内存读取指令, 协议: [4字节长度][pickle序列化的(方法名, *参数)]"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()   #* 阻塞直到 rank0 set() 通知
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) #* 用pickle 反序列化得到方法名和参数
        self.event.clear()  #* 重置事件, 等待下一次通知
        return method_name, args

    def write_shm(self, method_name, *args):
        """Rank0 将指令写入共享内存并通知所有 worker"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args]) #* 用 pickle 序列化方法名和参数, 转为字节流
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:  #* 逐个通知每个 worker
            event.set()

    def call(self, method_name, *args):
        """统一调用入口: rank0 先广播指令给 worker, 然后所有 rank 执行同一方法(保持 NCCL 同步)"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """用最大可能输入跑一次前向, 目的是触发 PyTorch 的峰值显存分配
        这样 allocate_kv_cache 才能准确测量"模型运行时最多用多少显存", 把剩余空间留给 KV cache"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  #* 重置峰值统计, 确保只测本次 warmup 的峰值
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]  #* 构造最大输入: 最多序列数 × 最大长度
        self.run(seqs, True)  #* prefill 模式前向, 触发所有中间 tensor 的分配
        torch.cuda.empty_cache()  #* 释放临时 tensor, 但峰值记录已保留

    def allocate_kv_cache(self): #* 以一张卡为单位(当前rank能看见的GPU)
        """根据 warmup 测得的峰值显存, 计算剩余空间能容纳多少 KV cache 块, 并一次性分配"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]      #* warmup 时的显存峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  #* 当前占用(模型权重等常驻)
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  #* TP 切分后每个 rank 的 KV head 数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        #* 单块字节 = 2(K+V) × 层数 × block_size × kv_heads × head_dim × 元素字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        #* 可用显存 = 总显存 × 利用率 - 非PyTorch占用(used-current) - 运行时峰值(peak)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        #* 统一分配所有层的 KV cache 为一个连续 tensor, 避免碎片
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        #* 将 KV cache 的每层切片绑定到对应 Attention 层的 k_cache/v_cache 属性
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """将各序列的 block_table 对齐为等长(短的用 -1 填充), 转为 GPU tensor
        shape: [num_seqs, max_num_blocks], 供 flash-attention 按 block_table 索引 KV cache"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prefill 阶段数据准备: 将 Sequence 列表转换为 flash-attention 所需的 GPU tensor (fa 把所有输入当成一个大 batch 一起处理, 以充分利用并行度)
        关键: 有前缀缓存时, Q 长度(需要计算的) < K 长度(需要 attend 到的全部上下文)"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]   #* Q 的累积序列长度(只包含未缓存的 token)
        cu_seqlens_k = [0]   #* K 的累积序列长度(包含全部 token, 含已缓存的)
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []    #* 每个新 token 要写入 KV cache 的物理位置
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])  #* 只取未缓存的 token 作为输入
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))  #* 对应的位置编码
            seqlen_q = seqlen - seq.num_cached_tokens  #* Q 长度 = 总长 - 已缓存
            seqlen_k = seqlen  #* K 长度 = 总长(需要 attend 到全部上下文, 包括缓存部分)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 时无真实 block_table
                continue
            #* slot_mapping: 从 num_cached_blocks 开始, 只映射需要写入的块(跳过已缓存的)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  #* 物理块起始 slot
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  #* 满块
                else:
                    end = start + seq.last_block_num_tokens  #* 最后一个块可能不满
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    #* K > Q 说明有前缀缓存命中, 需要 block_tables 读取缓存的 KV
            block_tables = self.prepare_block_tables(seqs)
        #* pin_memory + non_blocking: CPU→GPU 异步传输, 与计算重叠
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        #* 通过 thread-local context 传递给 Attention 层, 避免侵入模型前向签名
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Decode 阶段数据准备: 每个序列只有 1 个新 token(last_token), 但需要 attend 到全部历史"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)       #* 只输入最新生成的 token
            positions.append(len(seq) - 1)         #* 位置 = 序列长度 - 1
            context_lens.append(len(seq))           #* KV cache 中的上下文长度(用于 attention mask)
            #* 新 token 写入 KV cache 的 slot = 最后一个块的起始位置 + 块内偏移
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)  #* decode 总是需要 block_tables 来读取 KV cache
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集各序列的采样温度, 转为 GPU tensor 供 Sampler 使用"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行模型前向: prefill 用 eager 模式(变长), decode 用 CUDA Graph(固定形状, 更快)"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            #* Prefill / eager / 超大 batch: 直接跑 PyTorch eager forward
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            #* Decode + CUDA Graph: 将输入拷贝到预录制的固定 buffer, 然后 replay
            bs = input_ids.size(0)
            context = get_context()
            #* 选择 >= bs 的最小预录制 batch size (如 bs=5 → 选 graph_bs=8)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            #* 将实际输入写入 graph 录制时使用的固定 buffer(地址不变, graph replay 自动读取)
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)   #* -1 = 无效 slot, 填充超出 bs 的部分
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()  #* 回放预录制的 CUDA kernel 序列, 省去 Python 调度开销
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Engine 调用的统一入口: 数据准备 → 模型前向 → 采样, 返回生成的 token_ids
        只有 rank0 执行采样(因为只有 rank0 需要结果), worker 返回 None"""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()  #* 清理 thread-local context, 避免泄漏到下一次调用
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为 decode 阶段预录制 CUDA Graph, 覆盖多种 batch size: [1,2,4,8,16,32,...,max_bs]
        CUDA Graph 将一次前向的所有 kernel 录制为一个图, replay 时跳过 Python 调度,
        对 decode(每次只算 1 token/seq, 计算量小, 调度开销占比高)提速显著"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        #* 预分配固定 buffer, 所有 graph 共享同一块内存(通过切片 [:bs] 适配不同 batch size)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  #* 离散的 batch size 档位
        self.graphs = {}
        self.graph_pool = None  #* 所有 graph 共享同一个 CUDA memory pool, 减少显存碎片

        #* 从大到小录制: 大 bs 先占用 pool, 小 bs 复用已有空间
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    #* warmup: 让 CUDA 完成 kernel 编译等一次性工作
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    #* capture: 录制 kernel 调用序列
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  #* 第一个 graph 创建 pool, 后续复用
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        #* 保存 buffer 引用, run_model() 中通过修改这些 buffer 的内容来传递实际输入
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
