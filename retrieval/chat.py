from langchain_core.messages import HumanMessage,AIMessage
from utils.msg_trimmer import get_message_trimmer
from retrieval.filters import filter_messages
from utils.utils import save_messages_to_json
class Chat:
    def __init__(self,graph):
        self.graph =  graph
        self.bot_message =None

    def chatbot(self):
        state = {"messages":[]}
        thread_id = {"configurable": {"thread_id": "chat-thread"}}

        trimmer = get_message_trimmer()

        print("chatbot with memory started. Type 'exit' to quite.\n")
        print("chatbot with memory started. Type 'exit' to quite.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in  ["exit","quit"]:
                break
            state["messages"].append(HumanMessage(content=user_input))
            filtered = filter_messages(state["messages"],include_types=["human","ai"])
            state["messages"] = trimmer.invoke(filtered)

            for event in self.graph.stream(state,thread_id):
                # bot_response = event['messages'][-1].content
                # print(f"bot:{bot_response.content}")
                state.update(event)
                print("event",event)
                if "messages" in event:
                    print("event:", event)

                # loop through each node in the event
                for node_output in event.values():
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if isinstance(msg, AIMessage):
                                print("Bot:", msg.content)
                                state["messages"].append(msg)

                if self.bot_message:
                    state["messages"].append(self.bot_message)

        save_messages_to_json(state["messages"])