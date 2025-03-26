import threading

class NodeStateManager:
    def __init__(self):
        self.persistent_nodes = {}  # 存储持久化节点的状态
        self.node_outputs = {}      # 存储节点的输出
        self.node_inputs = {}       # 存储自定义节点输入
        self.lock = threading.RLock()
        
    def register_persistent_node(self, node_id, node_data):
        """注册一个需要持久化的节点"""
        with self.lock:
            self.persistent_nodes[node_id] = {
                "data": node_data,
                "instance": None,
                "last_executed": None
            }
            
    def get_node_output(self, node_id, output_index=0):
        """获取节点的输出"""
        with self.lock:
            if node_id not in self.node_outputs:
                return None
                
            outputs = self.node_outputs[node_id]
            
            # 处理不同类型的输出
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                # 输出是元组或列表 - 标准ComfyUI节点格式
                if 0 <= output_index < len(outputs):
                    return outputs[output_index]
                else:
                    return None
            else:
                # 输出是单一值 - 某些节点只返回一个对象
                if output_index == 0:
                    return outputs
                else:
                    return None
        
    def set_node_output(self, node_id, outputs):
        """设置节点的输出"""
        with self.lock:
            # 直接存储原始输出，不进行任何转换
            self.node_outputs[node_id] = outputs
            
    def set_custom_input(self, node_id, input_name, input_value):
        """设置节点的自定义输入"""
        with self.lock:
            if node_id not in self.node_inputs:
                self.node_inputs[node_id] = {}
            self.node_inputs[node_id][input_name] = input_value
            
    def get_custom_inputs(self, node_id):
        """获取节点的所有自定义输入"""
        with self.lock:
            return self.node_inputs.get(node_id, {})

    def export_state(self):
        """导出当前状态管理器的内容，用于调试和检查"""
        with self.lock:
            return {
                "persistent_nodes": self.persistent_nodes,
                "node_outputs": {k: str(type(v)) for k, v in self.node_outputs.items()},  # 只返回类型，避免大量数据
                "node_inputs": self.node_inputs
            } 