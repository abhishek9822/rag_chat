from ingestion.loader import DataLoader
from ingestion.preprocessor import Preprocessor
from ingestion.embedder import Embedder
from ingestion.indexer import VectorStore
from graph.memory_graph import build_rag_graph
from prompts.prompts import prompt,prompt_rag_fusion,rewrite_prompt,perspectives_prompt
from models.llm import llm
#retriever
from retrieval.chat import Chat
from utils.msg_trimmer import get_message_trimmer
from retrieval.filters import filter_messages
from retrieval.retriever import Retriever
from constant.constant import RAW_DATA
from langchain_core.messages import HumanMessage

class RagPipeline:
    def __init__(self):
        self.db=None
        self.retriever =None
        self.graph = None

    def data_ingestion(self):
        loader = DataLoader(raw_data_path=RAW_DATA)
        docs = loader.load_text_data()
        # print(f"docs: {docs}")

        preprocessor = Preprocessor(docs=docs)
        chunks = preprocessor.text_splitter()
        

        embedder = Embedder(chunks)
        _ = embedder.generate_embeddings()

        vector_store = VectorStore(chunks=chunks,embedding_model=embedder.embedding_model)
        self.db = vector_store.store_vector()

        return self.db
    
    def setup_retriever(self):
        self.retriever = Retriever(
            db=self.db,
            llm=llm,
            prompt=prompt,
            rewrite_prompt=rewrite_prompt,
            perspectives_prompt=perspectives_prompt,
            prompt_rag_fusion=prompt_rag_fusion)
        
    def chat_with_memory(self):
        if not self.db or not self.retriever:
            self.data_ingestion()
            self.setup_retriever()

        self.graph = build_rag_graph(self.retriever)
        chat = Chat(graph=self.graph)
        chat.chatbot()

    def run_pipeline(self):
        # retriever =self.data_ingestion()
        # generator = self.text_generation()
        self.chat_with_memory()
        # return generator