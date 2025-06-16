from langchain_community.document_loaders import TextLoader,PyPDFLoader
from pathlib import Path
from logger import logging

class DataLoader:
    def __init__(self, raw_data_path: str):
        self.data = raw_data_path

    def load_text_data(self):
        logging.info("Starting to load raw data...")

        try:
            loader = TextLoader(self.data, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = Path(self.data).name

            logging.info(f"Data loaded from {self.data} successfully. Total documents: {len(docs)}")
            return docs

        except FileNotFoundError:
            logging.error(f"File not found: {self.data}")
        except UnicodeDecodeError:
            logging.error(f"Encoding issue while reading: {self.data}")
        except Exception as e:
            logging.error(f"Unexpected error while loading raw data: {e}")

        return []

    def load_multiple_files(self, folder_path: str):
        all_docs = []
        folder = Path(folder_path)

        if not folder.exists():
            logging.error(f"Folder not found: {folder_path}")
            return []

        logging.info(f"Scanning folder: {folder_path} for .txt files")

        try:
            for file in folder.iterdir():
                if file.suffix == ".txt":
                    try:
                        loader = TextLoader(str(file), encoding="utf-8")
                        docs = loader.load()

                        for doc in docs:
                            doc.metadata["source"] = file.name

                        all_docs.extend(docs)
                        logging.info(f"Loaded {len(docs)} documents from file: {file.name}")

                    except Exception as e:
                        logging.warning(f"Failed to load file {file.name}: {e}")

            logging.info(f"Total documents loaded from folder {folder_path}: {len(all_docs)}")
            return all_docs

        except Exception as e:
            logging.error(f"Unexpected error while loading multiple files: {e}")
            return []
        
    def load_pdf_file(self,file_path):
        try:
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = Path(file_path).name
            logging.info(f"Data loaded from {file_path}. Total documents: {len(docs)}")
            return docs
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except UnicodeDecodeError:
            logging.error(f"Encoding issue while reading: {file_path}")
        except Exception as e:
            logging.error(f"Unexpected error while loading raw data: {e}")
        return []


    def run_text_loader(self):
        try:
            docs = self.load_text_data()
            if docs:
                logging.info("Text loader ran successfully.")
            else:
                logging.warning("Text loader returned no documents.")
        except Exception as e:
            logging.error(f"Error in run_text_loader: {e}")