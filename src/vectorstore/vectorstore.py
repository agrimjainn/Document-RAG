# from typing import List
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document

# class VectorStore:
#     """Manages vector store application"""

#     def __init__(self):
#         self.embedding = OpenAIEmbeddings()
#         self.vectorstore = None
#         self.retriever = None

#     def create_vectorstore(self, documents: List[Document]):
#         """
#         Create vector store from documents
        
#         Args:
#             documents: List of documents to embed
#         """
#         self.vectorstore = FAISS.from_documents(documents, self.embedding)
#         self.retriever = self.vectorstore.as_retriever()

#     def get_retriever(self, query: str, k: int=4) ->List[Document]:
#         """
#         Retrieve relevant documents to retrieve

#         Args:
#             query: Search Query
#             k: Number of documents to retrieve

#         Returns:
#             List of relevant documents
#         """

#         if self.retriever is None:
#             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
#         return self.retriever.invoke(query)
    


from typing import List
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store application"""

    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        if not documents:
            raise ValueError("No documents provided to vector store.")
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever