import base64
import json

def solve(payload: str):
    # Decode Base64
    decoded = base64.b64decode(payload).decode("utf-8")

    # Parse JSON
    data = json.loads(decoded)

    # Extract input
    input_data = data["adaptInput"]

    # Priority mapping
    priority_map = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }

    # Transform V1 -> V2
    return {
        "id": input_data["user"]["id"],
        "name": input_data["user"]["fullName"],
        "action": input_data["action"].lower(),
        "priority": priority_map[input_data["metadata"]["priority"]],
    }