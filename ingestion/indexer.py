from langchain_postgres.vectorstores import PGVector
import uuid
from langchain.indexes import SQLRecordManager,index
from langchain_core.documents import Document
from constant.constant import COLLECTION_NAME,NAMESPACE
import  os
from logger import logging
from dotenv import load_dotenv
load_dotenv()

connection = os.getenv("CONNECTION")

class VectorStore:
    def __init__(self, chunks, embedding_model):
        self.chunks = chunks
        self.embedding_model = embedding_model

    def store_vector(self):
        logging.info("Storing vectors into the database...")

        try:
            db = PGVector.from_documents(
                documents=self.chunks,
                embedding=self.embedding_model,
                connection=connection
            )
            logging.info(f"Stored {len(self.chunks)} chunks to pgvector successfully.")

            vectorstore = PGVector(
                embeddings=self.embedding_model,
                collection_name=COLLECTION_NAME,
                connection=connection,
                use_jsonb=True,
            )

            record_manager = SQLRecordManager(
                namespace=NAMESPACE,
                db_url=connection,
            )

            record_manager.create_schema()
            logging.info("SQLRecordManager schema created successfully.")

            indexing_result = index(
                self.chunks,
                record_manager,
                vectorstore,
                cleanup="incremental",
                source_id_key="source",
            )

            logging.info(f"Indexing completed. Result: {indexing_result}")
            return db

        except Exception as e:
            logging.error(f"Failed to store and index vectors: {e}")
            return None


        # results = vectorstore.similarity_search("ancient greek society", k=1)
        # for doc in results:
        #     print(doc.page_content)

