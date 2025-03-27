import torch
import torch.distributed as dist
import os

def init_distributed_env(backend="nccl"):
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if world_size > 1:
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        # 初始化进程组
        dist.init_process_group(backend=backend)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def tensor_to_bytes(tensor):
    """将张量序列化为字节"""
    buffer = torch.ByteStorage.from_buffer(tensor.contiguous().cpu().numpy().tobytes())
    return torch.ByteTensor(buffer)

def bytes_to_tensor(bytes_tensor, dtype=torch.float32, device="cuda"):
    """将字节反序列化为张量"""
    tensor = torch.FloatTensor(torch.frombuffer(bytes_tensor.cpu().numpy(), dtype=torch.float32))
    return tensor.to(device=device, dtype=dtype)