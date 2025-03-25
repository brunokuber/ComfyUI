import requests
import json

# 服务器地址
server_url = "http://127.0.0.1:8188"

# 要执行的节点ID
node_id = "4"  # 例如，CheckpointLoaderSimple节点

# 节点数据
node_data = {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
}

# 执行节点
response = requests.post(
    f"{server_url}/api/node/execute/{node_id}", 
    json=node_data
)

# 打印结果
print(response.json())
