import torch
import torch.distributed as dist
from node_state_manager import NodeStateManager
from dist_utils import tensor_to_bytes, bytes_to_tensor
import pickle
import io
import threading
import logging

class DistributedNodeStateManager(NodeStateManager):
    def __init__(self):
        super().__init__()
        self.distributed_mode = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.node_assignments = {}  # 节点ID到GPU的映射
        self.node_specs = {} # 节点专业化，指示哪些节点类型在哪些GPU上运行
        
    def initialize(self, is_distributed, rank, world_size, local_rank, node_specs=None):
        """初始化分布式模式"""
        self.distributed_mode = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        
        if node_specs:
            self.node_specs = node_specs
            
        logging.info(f"DistributedNodeStateManager initialized: rank={rank}, world_size={world_size}")
            
    def register_persistent_node(self, node_id, node_data):
        """注册持久化节点，支持通配符节点类型匹配"""
        with self.lock:
            # 根据节点类型决定该节点应该运行在哪个GPU上
            if self.distributed_mode and "class_type" in node_data:
                class_type = node_data["class_type"]
                assigned_gpu = None
                
                # 1. 检查是否有精确匹配
                for gpu_id, specs in self.node_specs.items():
                    if class_type in specs:
                        self.node_assignments[node_id] = int(gpu_id)
                        assigned_gpu = int(gpu_id)
                        break
                
                # 2. 如果没有精确匹配，检查通配符
                if assigned_gpu is None:
                    for gpu_id, specs in self.node_specs.items():
                        if "*" in specs or "ALL" in specs:
                            self.node_assignments[node_id] = int(gpu_id)
                            assigned_gpu = int(gpu_id)
                            break
                
                #     # 2. 如果没有精确匹配，检查通配符并应用负载均衡
                # if assigned_gpu is None:
                #     wildcard_gpus = []
                #     for gpu_id, specs in self.node_specs.items():
                #         if "*" in specs or "ALL" in specs:
                #             wildcard_gpus.append(int(gpu_id))
                    
                #     if wildcard_gpus:
                #         # 简单轮询均衡 - 可以根据节点ID或类型hash来分配
                #         assigned_gpu = wildcard_gpus[hash(node_id) % len(wildcard_gpus)]
                #         self.node_assignments[node_id] = assigned_gpu
            
            # 默认分配给当前GPU
            if node_id not in self.node_assignments:
                self.node_assignments[node_id] = self.rank
                
            # 只有负责该节点的GPU需要存储完整数据
            if self.node_assignments.get(node_id, 0) == self.rank:
                super().register_persistent_node(node_id, node_data)
                
            # 广播节点分配信息
            if self.distributed_mode:
                self._broadcast_node_assignment(node_id)
    
    def get_node_output(self, node_id, output_index=0):
        """获取节点输出，如果在远程则获取远程节点输出"""
        with self.lock:
            # 如果节点在当前进程，直接获取
            if not self.distributed_mode or self.node_assignments.get(node_id, 0) == self.rank:
                return super().get_node_output(node_id, output_index)
            
            # 否则从远程获取
            return self._fetch_remote_output(node_id, output_index)
    
    def set_node_output(self, node_id, outputs):
        """设置节点输出，并在分布式模式下广播可用性"""
        with self.lock:
            super().set_node_output(node_id, outputs)
            
            if self.distributed_mode:
                self._broadcast_output_available(node_id)
    
    def _broadcast_node_assignment(self, node_id):
        """广播节点分配信息"""
        if not self.distributed_mode:
            return
            
        data = pickle.dumps({
            "node_id": node_id,
            "assigned_rank": self.node_assignments[node_id]
        })
        size_tensor = torch.tensor([len(data)], dtype=torch.long, device="cuda")
        dist.broadcast(size_tensor, src=0)
        
        data_tensor = torch.ByteTensor(list(data)).to("cuda")
        data_tensor_padded = torch.zeros(size_tensor.item(), dtype=torch.uint8, device="cuda")
        data_tensor_padded[:len(data)] = data_tensor
        dist.broadcast(data_tensor_padded, src=0)
    
    def _broadcast_output_available(self, node_id):
        """广播节点输出可用性"""
        if not self.distributed_mode:
            return
            
        data = pickle.dumps({
            "node_id": node_id,
            "has_output": True,
            "source_rank": self.rank
        })
        size_tensor = torch.tensor([len(data)], dtype=torch.long, device="cuda")
        dist.broadcast(size_tensor, src=self.rank)
        
        data_tensor = torch.ByteTensor(list(data)).to("cuda")
        data_tensor_padded = torch.zeros(size_tensor.item(), dtype=torch.uint8, device="cuda")
        data_tensor_padded[:len(data)] = data_tensor
        dist.broadcast(data_tensor_padded, src=self.rank)
    
    def _fetch_remote_output(self, node_id, output_index):
        """从远程获取节点输出"""
        if not self.distributed_mode:
            return None
            
        # 向负责该节点的GPU请求数据
        target_rank = self.node_assignments.get(node_id, 0)
        
        # 发送请求
        req_data = pickle.dumps({
            "node_id": node_id, 
            "output_index": output_index,
            "requester_rank": self.rank
        })
        req_size = torch.tensor([len(req_data)], dtype=torch.long, device="cuda")
        dist.send(req_size, dst=target_rank)
        
        req_tensor = torch.ByteTensor(list(req_data)).to("cuda")
        req_tensor_padded = torch.zeros(req_size.item(), dtype=torch.uint8, device="cuda")
        req_tensor_padded[:len(req_data)] = req_tensor
        dist.send(req_tensor_padded, dst=target_rank)
        
        # 接收响应大小
        resp_size = torch.zeros(1, dtype=torch.long, device="cuda")
        dist.recv(resp_size, src=target_rank)
        
        # 接收响应数据
        resp_data = torch.zeros(resp_size.item(), dtype=torch.uint8, device="cuda")
        dist.recv(resp_data, src=target_rank)
        
        # 解析响应
        resp_bytes = resp_data.cpu().numpy().tobytes()
        output_data = pickle.loads(resp_bytes)
        
        return output_data["output"]
        
    def handle_remote_requests(self):
        """处理来自其他GPU的请求"""
        if not self.distributed_mode:
            return
            
        threading.Thread(target=self._request_handler_loop, daemon=True).start()
        
    def _request_handler_loop(self):
        """请求处理循环"""
        while True:
            try:
                # 接收请求大小
                req_size = torch.zeros(1, dtype=torch.long, device="cuda")
                dist.recv(req_size)
                
                # 接收请求数据
                req_data = torch.zeros(req_size.item(), dtype=torch.uint8, device="cuda")
                dist.recv(req_data)
                
                # 解析请求
                req_bytes = req_data.cpu().numpy().tobytes()
                request = pickle.loads(req_bytes)
                
                # 处理请求
                node_id = request["node_id"]
                output_index = request["output_index"]
                requester_rank = request["requester_rank"]
                
                # 获取输出
                output = super().get_node_output(node_id, output_index)
                
                # 序列化响应
                resp_data = pickle.dumps({"output": output})
                resp_size = torch.tensor([len(resp_data)], dtype=torch.long, device="cuda")
                
                # 发送响应大小
                dist.send(resp_size, dst=requester_rank)
                
                # 发送响应数据
                resp_tensor = torch.ByteTensor(list(resp_data)).to("cuda")
                resp_tensor_padded = torch.zeros(resp_size.item(), dtype=torch.uint8, device="cuda")
                resp_tensor_padded[:len(resp_data)] = resp_tensor
                dist.send(resp_tensor_padded, dst=requester_rank)
                
            except Exception as e:
                logging.error(f"Error handling remote request: {e}") 