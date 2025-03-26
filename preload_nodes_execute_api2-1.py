import requests
import json

server_url = "http://127.0.0.1:8188"

# 加载节点配置
with open("preload_nodes_sample_api.json", "r") as f:
    nodes_config = json.load(f)

# 只处理模型加载节点(4)
node_id = "4"
node_data = nodes_config[node_id]

# 注册节点
register_data = {
    "node_id": node_id,
    "node_data": node_data
}
register_response = requests.post(
    f"{server_url}/api/node/register",
    json=register_data
)
print(f"节点 {node_id} 注册结果: {register_response.status_code}")
if register_response.status_code != 200:
    print(register_response.text)

# 执行节点
response = requests.post(
    f"{server_url}/api/node/execute/{node_id}",
    json={"node_data": node_data}
)
print(f"节点 {node_id} 执行结果: {response.status_code}")

# 解析并打印结果
try:
    result = response.json()
    print(f"执行结果: {result}")
    
    # 获取输出数据并打印类型信息
    if "outputs" in result:
        output_data = result["outputs"]
        print(f"节点4输出数据: {output_data}")
        
        # 打印每个输出槽位的数据类型
        for output_index, output_value in output_data.items():
            if isinstance(output_value, dict) and "type" in output_value:
                print(f"输出槽位 {output_index} 的数据类型: {output_value['type']}")
            else:
                print(f"输出槽位 {output_index} 的Python类型: {type(output_value)}")
    else:
        print("结果中没有找到'output'字段")
except:
    print("无法解析JSON响应")
    print("原始响应:", response.text)