from execution import execute_single_node
from node_state_manager import NodeStateManager

# 创建状态管理器
state_manager = NodeStateManager()

# 注册节点
node_id = "4"
node_data = {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
}
state_manager.register_persistent_node(node_id, node_data)

# 执行节点
outputs, error = execute_single_node(None, node_id, node_data, None, state_manager)

if error:
    print(f"Error: {error}")
else:
    print(f"Node executed successfully. Outputs: {outputs}")