from langchain.vectorstores import Chroma
from Splitting import Splitter
from langchain.embeddings import OpenAIEmbeddings
from OpenIA_connection import GPT_connection

class Vector_Store:
    def __init__(self):

        self.db_directory = 'docs/chroma'

        self.api_key = GPT_connection().apiKey

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)

        self.docs = Splitter().splitted_data

        self.documents = []

        self.vectordb = Chroma.from_documents(documents=self.docs, embedding=self.embeddings, persist_directory=self.db_directory)


# print(docs)
# print(type(docs))
# print(type(docs[-1]))
# print(type(docs[-1][0]))

# print(vectordb._collection.count())