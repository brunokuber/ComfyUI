import requests
import json

server_url = "http://127.0.0.1:8188"

def check_node_state():
    response = requests.get(f"{server_url}/api/node/state")
    if response.status_code == 200:
        state = response.json()
        print(json.dumps(state, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    check_node_state() 