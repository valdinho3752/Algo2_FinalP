from langchain.chains import ConversationalRetrievalChain
from Vector_store import Vector_Store
from langchain.llms import OpenAI

class Conv_agent:
    def __init__(self) :
        self.vectorstore = Vector_Store().vectordb

        self.chat = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), self.vectorstore.as_retriever())

        self.chat_history = []
        
    def conversation(self, user_message):
        response = self.chat({"question": user_message, "chat_history": self.chat_history})
        self.chat_history.append((user_message, response["answer"]))

        return response["answer"]


# conv = Conv_agent()

# print(conv.KG_insert('como instalo windows'))