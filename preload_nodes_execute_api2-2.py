import requests
import json

server_url = "http://127.0.0.1:8188"

# 加载节点配置
with open("preload_nodes_sample_api.json", "r") as f:
    nodes_config = json.load(f)

# 按依赖顺序注册和执行节点，移除模型加载节点(4)
execution_order = ["5", "6", "7", "3", "8", "9"]  # 移除了节点4（模型加载）

# 先注册所有节点，除了模型加载节点
for node_id in nodes_config:
    if node_id != "4":  # 跳过模型加载节点
        node_data = nodes_config[node_id]
        # 使用正确的API格式 - 注意与服务器端定义匹配
        register_data = {
            "node_id": node_id,
            "node_data": node_data
        }
        register_response = requests.post(
            f"{server_url}/api/node/register",
            json=register_data
        )
        print(f"Node {node_id} registration: {register_response.status_code}")
        if register_response.status_code != 200:
            print(register_response.text)

# 按顺序执行节点
for node_id in execution_order:
    node_data = nodes_config[node_id]
    response = requests.post(
        f"{server_url}/api/node/execute/{node_id}",
        json={"node_data": node_data}  # 包装在node_data字段中
    )
    print(f"Node {node_id} executed: {response.status_code}")
    try:
        print(response.json())
    except:
        print("Could not parse JSON response")
    
    # 如果是最后一个节点（SaveImage），获取输出
    if node_id == "9":
        try:
            result = response.json()
            print(f"Final result: {result}")
        except:
            print("Could not parse JSON response")