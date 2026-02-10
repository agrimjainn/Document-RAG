""" Docstring for src.document_ingestion.document_processor

    Module for loading and splitting documents
"""

from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader, TextLoader, PyPDFDirectoryLoader)
from pathlib import Path


class DocumentProcessor:

    def __init__(self, chunk_size: int=500, chunk_overlap: int=50):
        """
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url:str) -> List[Document]:
        """Loads documents from urls"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_from_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Loads all the PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self, file_path):
        """Load documents from TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    
    def load_from_pdf(self, file_path:Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file"""
        loader = PyPDFLoader(str(file_path)) #"data" if this doesnt workout
        return loader.load()
    
    def load_documents(self, sources:List[str]) -> List[Document]:
        """
        Load documents from given sources.

        Args:
            sources (List[str]): URLs or file paths to load documents from.

        Returns:
            List[Document]: Loaded documents from all valid sources.
        """

        docs: List[Document] = []

        for src in sources:
            if src.startswith("http://" or src.startswith("https://")):
                docs.extend(self.load_from_url(src))

            path = Path("data")
            if path.is_dir():
                docs.extend(self.load_from_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(f"Unsupported source type: {src}."
                                 "Use URL, .txt file, or PDF Directory"
                                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            Documents: List of documents to split

        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    # To be called only for URLs
    def process_urls(self, urls:List[str]) -> List[Document]:

        """
        Complete pipeline to load and split documents

        Args:
            urls: List of URLs to process
        
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)