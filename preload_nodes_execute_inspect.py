import sys
import json
import nodes
import traceback
import inspect
import copy
from node_state_manager import NodeStateManager
from execution import execute_single_node

# 创建状态管理器
state_manager = NodeStateManager()

# 加载节点配置
with open("preload_nodes_sample_api.json", "r") as f:
    nodes_config = json.load(f)

# 检查CLIPTextEncode节点的实现
clip_encode_class = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]
print("\n=== CLIPTextEncode Node Inspection ===")
print(f"Input types: {clip_encode_class().INPUT_TYPES()}")
print(f"Function signature: {inspect.signature(getattr(clip_encode_class(), 'encode'))}")
print("======================================\n")

# 按依赖顺序注册和执行节点
execution_order = ["4", "5", "6", "7", "3", "8", "9"]

# 注册所有节点
for node_id, node_data in nodes_config.items():
    state_manager.register_persistent_node(node_id, node_data)
    node_instance = nodes.NODE_CLASS_MAPPINGS[node_data["class_type"]]()
    state_manager.persistent_nodes[node_id]["instance"] = node_instance
    print(f"Registered node {node_id}: {node_data['class_type']}")

# 手动测试CLIP处理
print("\n=== Manual Test of CLIP Node ===")
# 执行CheckpointLoader节点
checkpoint_outputs, _ = execute_single_node(None, "4", nodes_config["4"], None, state_manager)
if checkpoint_outputs:
    model, clip, vae = checkpoint_outputs
    
    # 手动执行CLIPTextEncode
    clip_node = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    try:
        # 直接调用节点方法，避免中间处理
        clip_text = "beautiful scenery nature"
        print(f"Direct encoding with CLIP object type: {type(clip).__name__}")
        print(f"CLIP attributes: {dir(clip)[:5]}...")
        
        # 检查clip是否有tokenize方法
        if hasattr(clip, 'tokenize'):
            print("CLIP has tokenize method!")
        else:
            print("CLIP missing tokenize method!")
            
        result = clip_node.encode(clip_text, clip)
        print(f"Direct encoding successful! Result type: {type(result).__name__}")
    except Exception as e:
        print(f"Direct encoding failed: {e}")
        traceback.print_exc()
print("==============================\n")

# 按顺序执行节点
for node_id in execution_order:
    node_data = nodes_config[node_id]
    print(f"\nExecuting node {node_id}: {node_data['class_type']}")
    
    # 执行节点
    try:
        outputs, error = execute_single_node(None, node_id, node_data, None, state_manager)
        
        if error:
            print(f"Error executing node {node_id}: {error}")
        else:
            print(f"Node {node_id} executed successfully")
            
            # 记录输出
            if isinstance(outputs, (list, tuple)):
                for i, output in enumerate(outputs):
                    print(f"  Output {i}: {type(output).__name__}")
            else:
                print(f"  Output: {type(outputs).__name__}")
    except Exception as e:
        print(f"Exception during execution: {e}")
        traceback.print_exc()

# 打印节点输出状态
print("\n=== Final Node Outputs ===")
for node_id in state_manager.node_outputs:
    outputs = state_manager.node_outputs[node_id]
    print(f"Node {node_id}:")
    if isinstance(outputs, (list, tuple)):
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {type(output).__name__} at {id(output)}")
    else:
        print(f"  Output: {type(outputs).__name__} at {id(outputs)}") 