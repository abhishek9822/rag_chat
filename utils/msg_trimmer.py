from langchain_core.messages import trim_messages
from models.llm import llm

def get_message_trimmer(max_tokens=512):
    return trim_messages(
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )