from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

class VectoreStore:
    def __init__(self):
        self.embedding = HuggingFaceBgeEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self, query: str, k: int=4) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)        