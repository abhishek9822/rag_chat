import os
from euriai.langchain_embed import EuriaiEmbeddings
from logger import logging
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("EURI_API_KEY")
class Embedder:
    def __init__(self, chunks):
        self.chunks = chunks

        if not API_KEY:
            logging.critical("Missing API key for EuriaiEmbeddings. Set 'EURI_API_KEY' environment variable.")
            raise EnvironmentError("API key not found for EuriaiEmbeddings.")

        try:
            self.embedding_model = EuriaiEmbeddings(api_key=API_KEY)
        except Exception as e:
            logging.error(f"Failed to initialize embedding model: {e}")
            raise

    def generate_embeddings(self):
        logging.info("Starting embedding generation...")

        try:
            contents = [chunk.page_content for chunk in self.chunks]
            embeddings = self.embedding_model.embed_documents(contents)

            logging.info(f"Embeddings generated successfully. Total vectors: {len(embeddings)}")
            return embeddings

        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            return []

