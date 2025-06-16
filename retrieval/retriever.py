import os
from euriai import EuriaiLangChainLLM
from langchain_core.runnables import  chain,RunnableLambda
from langchain_core.messages import AIMessage
from logger import logging
from retrieval.reranker import reciprocal_rag_fusion
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("EURI_API_KEY")


class Retriever:
    def __init__(self, db, llm, prompt, rewrite_prompt, perspectives_prompt, prompt_rag_fusion):
        self.db = db
        self.llm = llm
        self.prompt = prompt
        self.rewrite_prompt = rewrite_prompt
        self.perspectives_prompt = perspectives_prompt
        self.prompt_rag_fusion = prompt_rag_fusion
        self.retriever = self.db.as_retriever(search_kwargs={"k": 2})
        self.llm_chain = prompt | self.llm

    def get_documents(self, query):
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            return []

    def format_context(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def run_basic_qa(self, query, chat_history=""):
        logging.info("[Basic QA] Running...")
        try:
            docs = self.get_documents(query)
            context = self.format_context(docs)
            result = self.llm_chain.invoke({
                "context": context,
                "question": query,
                "chat_history": chat_history
            })
            print(result)

            logging.info("[Basic QA] Result:\n%s", result)
            return AIMessage(content=result)

        except Exception as e:
            logging.error(f"[Basic QA] Failed: {e}")
            return AIMessage(content="Sorry, an error occurred during basic QA.")

    def parse_rewritten_output(self, message):
        return message.strip('"').strip("**")

    def run_rewritten_query_qa(self, query,chat_history=""):
        logging.info("[Rewritten QA] Running...")

        try:
            rewriter = self.rewrite_prompt | self.llm | RunnableLambda(self.parse_rewritten_output)

            @chain
            def qa_rrr(input_query):
                new_query = rewriter.invoke({
                    "chat_history":chat_history,
                    "question": input_query,
                    "context": ""

                })
                logging.info("[Rewritten QA] New query: %s", new_query)

                docs = self.get_documents(new_query)
                context = self.format_context(docs)
                result = self.llm_chain.invoke({
                    "context": context,
                    "question": input_query,
                    "chat_history": chat_history
                })

                return AIMessage(content=result)

            result = qa_rrr.invoke(query)
            logging.info("[Rewritten QA] Result:\n%s", result)
            return result

        except Exception as e:
            logging.error(f"[Rewritten QA] Failed: {e}")
            return "Sorry, an error occurred during rewritten QA."

    def parse_query_output(self, message):
        return message.split('\n')

    def get_unique_union(self, document_lists):
        try:
            deduped_docs = {
                doc.page_content: doc for sublist in document_lists for doc in sublist
            }
            return list(deduped_docs.values())
        except Exception as e:
            logging.error(f"[Multi-query QA] Deduplication failed: {e}")
            return []

    def multi_query_qa(self, query,chat_history=""):
        logging.info("[Multi-query QA] Running...")

        try:
            query_gen = self.perspectives_prompt | self.llm | self.parse_query_output
            retrieval_chain = query_gen | self.retriever.batch | self.get_unique_union
            docs = retrieval_chain.invoke(query)
            @chain
            def multi_query_chain(docs=docs):
                
                context = self.format_context(docs)
                prompt_input = self.prompt.format({"context": context, "question": input,"chat_history":chat_history})
                answer = self.llm.invoke(prompt_input)
                return answer

            result = multi_query_chain.invoke(docs)
            logging.info("[Multi-query QA] Result:\n%s", result)
            return result

        except Exception as e:
            logging.error(f"[Multi-query QA] Failed: {e}")
            return "Sorry, an error occurred during multi-query QA."

    def rag_fusion(self, query,chat_history =""):
        logging.info("[RAG Fusion] Running...")

        try:
            query_gen = self.prompt_rag_fusion | self.llm | self.parse_query_output
            retrieval_chain = query_gen | self.retriever.batch | reciprocal_rag_fusion

            docs = retrieval_chain.invoke(query)
            logging.info("[RAG Fusion] Retrieved context:\n%s", docs[0].page_content if docs else "No documents.")

            @chain
            def rag_fusion_chain(input):
                docs = retrieval_chain.invoke(input)
                context = self.format_context(docs)

                prompt_input = self.prompt.invoke({"context": context, "question": input,"chat_history":chat_history})
                answer = self.llm.invoke(prompt_input)
                return answer

            result = rag_fusion_chain.invoke(query)
            logging.info("[RAG Fusion] Result:\n%s", result)
            return result

        except Exception as e:
            logging.error(f"[RAG Fusion] Failed: {e}")
            return "Sorry, an error occurred during RAG Fusion."

    def generate_answer(self, query):
        logging.info("[Pipeline Start] Query: %s", query)

        self.run_basic_qa(query)
        self.run_rewritten_query_qa(query)
        self.multi_query_qa(query)
        self.rag_fusion(query)

        logging.info("[Pipeline Complete]")


