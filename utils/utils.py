from langchain_core.messages import HumanMessage
from datetime import datetime
import json
import os
from constant.constant import CONVERSATION_DATA
def save_messages_to_json(messages, folder=CONVERSATION_DATA, filename=None):
    # Convert messages to serializable dicts
    # data = [{"role": "user" if isinstance(msg, HumanMessage) else "bot", "content": msg.content} for msg in messages]

    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Use timestamp as filename if not provided
    data = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "bot"
        data.append({"role": role, "content": msg.content})

    os.makedirs(folder, exist_ok=True)
    if filename is None:
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)