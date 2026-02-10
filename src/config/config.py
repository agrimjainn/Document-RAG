# """Configuration module for Agentic RAG system"""

# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# # Load environment variables
# load_dotenv()

# class Config:
#     """Configuration class for RAG system"""
    
#     # API Keys
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
#     # Model Configuration
#     LLM_MODEL = "openai:gpt-4o"
    
#     # Document Processing
#     CHUNK_SIZE = 500
#     CHUNK_OVERLAP = 50
    
#     # Default URLs
#     DEFAULT_URLS = [
#         "https://lilianweng.github.io/posts/2023-06-23-agent/",
#         "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
#     ]
    
#     @classmethod
#     def get_llm(cls):
#         """Initialize and return the LLM model"""
#         os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
#         return init_chat_model(cls.LLM_MODEL)

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class Config:
    """Configuration class for RAG system"""

    GQ_TOKEN = os.getenv("GROQ_API_KEY")


    # Groq model
    LLM_MODEL = "llama-3.1-8b-instant"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("API KEY MISSING")
        
        return ChatGroq(
            api_key=api_key,
            model=cls.LLM_MODEL,
            temperature=0.2
        )
