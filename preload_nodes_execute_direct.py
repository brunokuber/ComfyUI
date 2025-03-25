import sys
import json
import nodes
from node_state_manager import NodeStateManager
from execution import execute_single_node

# 创建状态管理器
state_manager = NodeStateManager()

# 加载节点配置
with open("preload_nodes_sample_api.json", "r") as f:
    nodes_config = json.load(f)

# 按依赖顺序注册和执行节点
execution_order = ["4", "5", "6", "7", "3", "8", "9"]

# 注册所有节点
for node_id, node_data in nodes_config.items():
    state_manager.register_persistent_node(node_id, node_data)
    print(f"Registered node {node_id}: {node_data['class_type']}")

# 按顺序直接执行节点
for node_id in execution_order:
    node_data = nodes_config[node_id]
    print(f"\nExecuting node {node_id}: {node_data['class_type']}")
    
    # 直接调用执行函数，避免序列化问题
    outputs, error = execute_single_node(None, node_id, node_data, None, state_manager)
    
    if error:
        print(f"Error executing node {node_id}: {error}")
    else:
        # 安全打印输出信息
        output_info = []
        for i, output in enumerate(outputs):
            output_type = type(output).__name__
            if hasattr(output, "shape"):
                output_info.append(f"Output {i}: {output_type} with shape {output.shape}")
            else:
                output_info.append(f"Output {i}: {output_type}")
        
        print(f"Node {node_id} executed successfully")
        print("Outputs:", ", ".join(output_info))