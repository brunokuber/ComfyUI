import requests
import json

server_url = "http://127.0.0.1:8188"

# 先执行CheckpointLoaderSimple节点
checkpoint_node_id = "4"
checkpoint_node_data = {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
}

response = requests.post(
    f"{server_url}/api/node/execute/{checkpoint_node_id}", 
    json=checkpoint_node_data
)
print(f"Checkpoint node executed: {response.status_code}")

# 然后执行CLIPTextEncode节点，使用前一个节点的输出
clip_node_id = "6"
clip_node_data = {
    "class_type": "CLIPTextEncode",
    "inputs": {
        "text": "beautiful scenery nature",
        "clip": [checkpoint_node_id, 1]  # 引用checkpoint节点的第二个输出
    }
}

response = requests.post(
    f"{server_url}/api/node/execute/{clip_node_id}", 
    json=clip_node_data
)
print(f"CLIP node executed: {response.status_code}")
print(response.json())