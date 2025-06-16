from langchain.prompts import ChatPromptTemplate


# prompt	=	ChatPromptTemplate.from_messages([
# 				("system",	"""You	are	a	helpful	assistant.	Answ 								of	your	ability."""),
# 				("placeholder",	"{messages}"), ])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the question based only on the context below."),
#     ("human", "Context:\n{context}\n\nQuestion: {question}")
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers based on the provided context and remembers previous turns."),
    ("human", "Chat history:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}")
])

# Rewrite Prompt for Search Query Enhancement
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide a better search query for a web search engine to answer the given question. End the queries with '**'."),
    ("human", "Chat history:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}")
])

# RAG Fusion Prompt – Generate Multiple Search Queries (With History & Context)
prompt_rag_fusion = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates multiple search queries based on a single input question."),
    ("human", "Chat history:\n{chat_history}\n\nContext:\n{context}\n\nGenerate 4 diverse search queries related to:\n{question}\nOutput:")
])

# Perspective Prompt – For Vector Search Reformulation
perspectives_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database."),
    ("human", "Chat history:\n{chat_history}\n\nContext:\n{context}\n\nOriginal question:\n{question}\n\nGenerate 5 alternative phrasings of the question, separated by newlines.")
])

# rewrite_prompt = ChatPromptTemplate.from_template([
#     ("system","Provide a better search query for web search engine to answer the given question,end the queries with '**'."),
#     ("human", "Chat history:\n{chat_history}\n\nContext:\n{context}\n\nQuestion:{questionn}")])
#     # """Provide a better search query for web search engine to answer the given question,end the queries with '**'.\n\nQuestion: {x} \nAnswer: """


# prompt_rag_fusion = ChatPromptTemplate.from_template(
#     """You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate multiple search queries related to: {question} \n Output (4 queries):""")

# perspectives_prompt = ChatPromptTemplate.from_template(
#     """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
#     By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
#     Provide these alternative questions separated by newlines.
#     Original question: {question}"""
# )

