import sys
import json
import nodes
import traceback
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

# 节点执行追踪工具函数
def check_node_outputs(node_id):
    print(f"Checking outputs for node {node_id}:")
    raw_output = state_manager.node_outputs.get(node_id)
    if raw_output is None:
        print("  No outputs found")
        return
    
    # 处理不同类型的输出
    if isinstance(raw_output, (tuple, list)):
        # 多输出节点
        for i, output in enumerate(raw_output):
            print(f"  Output {i}: {type(output).__name__}")
            print_object_info(output)
    else:
        # 单输出节点
        print(f"  Single output: {type(raw_output).__name__}")
        print_object_info(raw_output)

def print_object_info(obj):
    """打印对象的基本信息"""
    if hasattr(obj, "shape"):
        print(f"    Shape: {obj.shape}")
    if hasattr(obj, "__dict__"):
        attrs = list(obj.__dict__.keys())[:5] if len(obj.__dict__) > 5 else list(obj.__dict__.keys())
        print(f"    Attributes: {attrs}" + ("..." if len(obj.__dict__) > 5 else ""))
    if hasattr(obj, "keys") and callable(getattr(obj, "keys")):
        keys = list(obj.keys())[:5] if len(obj.keys()) > 5 else list(obj.keys())
        print(f"    Keys: {keys}" + ("..." if len(obj.keys()) > 5 else ""))

# 在现有脚本中添加额外的调试函数
def print_all_stored_outputs():
    """打印状态管理器中所有存储的节点输出"""
    print("\n=== All Stored Node Outputs ===")
    for node_id, outputs in state_manager.node_outputs.items():
        print(f"Node {node_id}:")
        if isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                print(f"  Output {i}: {type(output).__name__}")
        else:
            print(f"  Single output: {type(outputs).__name__}")
    print("===============================\n")

# 按顺序直接执行节点
for node_id in execution_order:
    node_data = nodes_config[node_id]
    print(f"\nExecuting node {node_id}: {node_data['class_type']}")
    
    # 检查输入引用
    for input_name, input_value in node_data["inputs"].items():
        if isinstance(input_value, list) and len(input_value) == 2:
            ref_node_id, output_idx = input_value
            ref_output = state_manager.get_node_output(ref_node_id, output_idx)
            print(f"  Input {input_name} references node {ref_node_id}[{output_idx}]: "
                  f"{type(ref_output).__name__ if ref_output else 'Not found'}")
    
    # 执行节点
    try:
        outputs, error = execute_single_node(None, node_id, node_data, None, state_manager)
        
        if error:
            print(f"Error executing node {node_id}: {error}")
        else:
            print(f"Node {node_id} executed successfully")
            check_node_outputs(node_id)
    except Exception as e:
        print(f"Exception during execution: {e}")
        traceback.print_exc()

# 打印所有存储的输出状态
print_all_stored_outputs()