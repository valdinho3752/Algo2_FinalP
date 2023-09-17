from langchain.text_splitter  import RecursiveCharacterTextSplitter, CharacterTextSplitter
from Load_data import LoadData

class Splitter:

    def __init__(self):
        self.data = LoadData().docs
        self.chunk_size = 100
        self.chunk_overlap = 50
        self.r_splitter = RecursiveCharacterTextSplitter( chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap, separators=["\n", "\n\n","\n\n" * 3, " ", ""])
        self.c_splitter = CharacterTextSplitter( chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap, separator = ' ')
        self.splitted_data = self.split(self.data)
    
    def split(self, data):
        chunks = self.c_splitter.split_documents(data)
        return chunks
    

# split = Splitter()
# print(split.splitted_data)
# print(type(split.splitted_data))
# print(type(split.splitted_data[0]))
# print(type(split.splitted_data[0][0]))