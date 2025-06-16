from  typing import Annotated,TypedDict
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver


from typing import Annotated, TypedDict , List
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.prebuilt import add_messages
# from langgraph.graph.schema import END

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    strategy: str  # e.g., "basic", "rewrite", "multi", "fusion"

def decide_strategy(state: ChatState) -> dict:
    query = state["messages"][-1].content.lower()
    
    # Rule-based decision (can be replaced with LLM)
    if "rewrite" in query:
        strategy = "rewritten_qa"
    elif "multi" in query:
        strategy = "multi_query_qa"
    elif "fusion" in query or "combine" in query:
        strategy = "rag_fusion"
    else:
        strategy = "basic_qa"
    return {"strategy":strategy}

def basic_qa_node(state: ChatState, retriever) -> dict:
    query = state["messages"][-1].content
    history = build_chat_history(state["messages"])
    result = retriever.run_basic_qa(query, chat_history=history)
    print("i am from memory graph",result)
    return {"messages": [result]}

def rewritten_qa_node(state: ChatState, retriever) -> dict:
    query = state["messages"][-1].content
    result = retriever.run_rewritten_query_qa(query)
    return {"messages": [result]}

def multi_query_node(state: ChatState, retriever) -> dict:
    query = state["messages"][-1].content
    result = retriever.multi_query_qa(query)
    return {"messages": [result]}

def rag_fusion_node(state: ChatState, retriever) -> dict:
    query = state["messages"][-1].content
    result = retriever.rag_fusion(query)
    return {"messages": [result]}

def build_rag_graph(retriever):
    
    builder = StateGraph(ChatState)

    builder.add_node("router", decide_strategy)

    # Add all QA nodes
    builder.add_node("basic_qa", lambda state: basic_qa_node(state, retriever))
    builder.add_node("rewritten_qa", lambda state: rewritten_qa_node(state, retriever))
    builder.add_node("multi_query_qa", lambda state: multi_query_node(state, retriever))
    builder.add_node("rag_fusion", lambda state: rag_fusion_node(state, retriever))

    # Decision node
    builder.add_conditional_edges(
        "router",
        lambda state: state["strategy"],
        {
            "basic_qa": "basic_qa",
            "rewritten_qa": "rewritten_qa",
            "multi_query_qa": "multi_query_qa",
            "rag_fusion": "rag_fusion"
        }
    )

    # Connect entry and QA nodes to END
    builder.set_entry_point("router")
    for node in ["basic_qa", "rewritten_qa", "multi_query_qa", "rag_fusion"]:
        builder.add_edge(node, END)

    # Use persistent memory
    # memory = SqliteSaver("./chat_memory.db")
    # memory = SqliteSaver("sqlite:///chat_memory.db")
    memory = MemorySaver()

    return builder.compile(checkpointer=memory)

# class ChatState(TypedDict):
#     messages:Annotated[list[BaseMessage],add_messages]

def build_chat_history(messages : List[BaseMessage], max_turns=5):
    """Formats the last N turns of chat history into a string."""
    history = []
    for msg in messages[-2*max_turns:]:  # human+AI per turn
        role = "User" if isinstance(msg, HumanMessage) else "Bot"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)

# def build_rag_graph(retriever):
#     def qa_node(state: ChatState)->dict:
#         query = state["messages"][-1].content
#         chat_history = build_chat_history(state["messages"])

#         result = retriever.run_basic_qa(query,chat_history=chat_history)

#         return {"messages":[result]}
    
#     builder = StateGraph(ChatState)
#     builder.add_node("basic_qa",qa_node)
#     builder.set_entry_point("basic_qa")
#     builder.add_edge("basic_qa",END)

#     memory = MemorySaver()
#     return builder.compile(checkpointer=memory)

