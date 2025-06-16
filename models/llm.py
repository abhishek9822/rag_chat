import os
from euriai import EuriaiLangChainLLM
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EURI_API_KEY")

llm = EuriaiLangChainLLM(
    api_key=API_KEY,
    model="gpt-4.1-nano",
    temperature=0.7,
    max_tokens=300
)