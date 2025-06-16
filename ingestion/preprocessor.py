from langchain_text_splitters import RecursiveCharacterTextSplitter
from constant.constant import CHUNK_SIZE,CHUNK_OVERLAP
from logger import logging


class Preprocessor:
    def __init__(self, docs):
        self.docs = docs

    def text_splitter(self):
        logging.info("Starting text splitting process...")

        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splitted_docs = splitter.split_documents(self.docs)

            logging.info(f"Data split into {len(splitted_docs)} chunks.")
            return splitted_docs

        except Exception as e:
            logging.error(f"Error during text splitting: {e}")
            return []
