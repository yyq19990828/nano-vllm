from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """表示 GPU 上一段固定大小的 KV cache 存储空间（默认256 tokens）"""

    def __init__(self, block_id):
        self.block_id = block_id   #* 在 GPU KV cache 数组中的索引
        self.ref_count = 0         #* 引用计数: 有多少个序列正在共享这个块, 归零时可回收
        self.hash = -1             #* 块内容的链式哈希, -1 表示块未填满(不可缓存)
        self.token_ids = []        #* 块中实际的 token 内容, 用于缓存命中时二次校验(防哈希碰撞)

    def update(self, hash: int, token_ids: list[int]):
        """块填满后, 注册其哈希和内容, 使其可被前缀缓存匹配"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """块被重新分配时重置状态, ref_count 初始化为 1"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  #* 所有块对象, 按 block_id 索引
        self.hash_to_block_id: dict[int, int] = dict()  #* 前缀缓存核心: 链式哈希 → block_id 的映射表
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #* 空闲块队列
        self.used_block_ids: set[int] = set()  #* 正在被使用的块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算块的链式哈希: hash = xxh64(前一块哈希 + 当前块token_ids)
        链式设计使得哈希代表的是"从序列开头到当前块"的完整前缀,
        这样两个不同序列只要前缀内容一致, 哈希就相同, 即可共享 KV cache"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  #* 将前一块的哈希作为前缀混入
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """将指定块从空闲队列移到已使用集合, 并重置其状态"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()  #* ref_count 置为 1, 清空哈希和 token_ids
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """将引用计数归零的块归还到空闲队列"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)  #* 追加到队尾, 尽量延迟复用, 给前缀缓存更多存活机会

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来容纳整个序列(最坏情况: 全部缓存未命中)"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为新序列分配 KV cache 块, 同时尝试前缀缓存匹配
        逐块遍历序列的 token, 对每个块:
        1. 计算链式哈希(只有填满的块才计算)
        2. 查缓存表: 命中则共享, 未命中则分配新块
        一旦某个块未命中, 后续所有块都不可能命中(因为链式哈希依赖前一块)"""
        assert not seq.block_table
        h = -1              #* 前一块的哈希, 初始为 -1
        cache_miss = False  #* 一旦为 True, 后续块全部跳过缓存查找
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  #* 取出第 i 个块对应的 token_ids
            #* 只有填满 block_size 的块才计算哈希, 最后一个不满的块 hash=-1(不可缓存)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            #* 哈希未找到, 或找到但 token_ids 不匹配(哈希碰撞) → 标记为缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                #* 缓存未命中: 从空闲队列头部取一个新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                #* 缓存命中: 这个块的 KV cache 已经在 GPU 上, 不需要重新计算
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    #* 块正在被其他序列使用 → 直接增加引用计数(共享)
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    #* 块已被释放到空闲队列但内容还在(幸运复活) → 重新 allocate
                    block = self._allocate_block(block_id)
            if h != -1:
                #* 满块: 注册哈希和内容到缓存表, 供后续序列复用
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块, 倒序遍历以优先释放靠后的块
        (靠前的块更可能是公共前缀, 被其他序列共享, ref_count > 1 不会真正释放)"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:  #* 没有其他序列引用了, 真正归还到空闲队列
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查 decode 阶段能否追加新 token
        只有当序列长度 % block_size == 1 时(刚溢出到新块), 才需要 1 个空闲块"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 阶段每生成一个新 token 后调用, 维护 block_table 和前缀缓存
        根据序列长度对 block_size 取余, 分三种情况处理:"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            #* 余数==1: 新 token 溢出到新块, 上一个块刚好填满(hash!=-1)
            #* 需要分配一个新的空块追加到 block_table
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            #* 余数==0: 最后一个块刚好被填满, 计算哈希并注册到缓存表
            #* 这样未来新序列如果有相同前缀, 就能命中这个块
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            #* 其他情况: 块还没填满, 无需任何操作
            assert last_block.hash == -1
