# """Langgraph nodes for RAG workflow + reAct agent inside generator_context"""

# from typing import List, Optional
# from src.state.rag_state import RAGState

# from langchain_core.documents import Document
# from langchain_core.tools import Tool
# from langchain_core.messages import HumanMessage

# from langgraph.prebuilt import create_react_agent  # ✅ correct import

# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


# class RAGNodes:
#     """Contains the node functions for RAG workflow"""

#     def __init__(self, retriever, llm):
#         self.retriever = retriever
#         self.llm = llm
#         self._agent = None  # Lazy init agent

#     def retrieve_docs(self, state: RAGState) -> RAGState:
#         """Classic retriever node"""
#         docs = self.retriever.invoke(state.question)
#         return RAGState(
#             question=state.question,
#             retrieved_docs=docs
#         )

#     def _build_tools(self) -> List[Tool]:
#         """Build retriever + wikipedia tools"""

#         def retriever_tool_fn(query: str) -> str:
#             docs: List[Document] = self.retriever.invoke(query)

#             if not docs:
#                 return "No documents found."

#             merged = []
#             for i, d in enumerate(docs[:8], start=1):
#                 meta = d.metadata if hasattr(d, "metadata") else {}
#                 title = meta.get("title") or meta.get("source") or f"doc_{i}"
#                 merged.append(f"[{i}] {title}\n{d.page_content}")
#             return "\n\n".join(merged)

#         retriever_tool = Tool(
#             name="retriever",
#             description="Fetch passages from indexed corpus.",  # ✅ fixed typo
#             func=retriever_tool_fn
#         )

#         wiki = WikipediaQueryRun(
#             api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
#         )

#         wikipedia_tool = Tool(
#             name="wikipedia",
#             description="Search Wikipedia for general knowledge.",
#             func=wiki.run
#         )

#         return [retriever_tool, wikipedia_tool]

#     def _build_agent(self):
#         """reAct agent with tools"""
#         tools = self._build_tools()
#         system_prompt = (
#             "You are a helpful RAG agent. "
#             "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
#             "Return only the final useful answer."
#         )
#         self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)  # ✅

#     def generate_answer(self, state: RAGState) -> RAGState:
#         """Generate answer using reAct agent with retriever + wikipedia"""

#         if self._agent is None:
#             self._build_agent()

#         result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

#         messages = result.get("messages", [])
#         answer: Optional[str] = None

#         if messages:
#             answer_msg = messages[-1]
#             answer = getattr(answer_msg, "content", None)

#         return RAGState(
#             question=state.question,
#             retrieved_docs=state.retrieved_docs,
#             answer=answer or "Could not generate answer."
#         )


"""Langgraph nodes for RAG workflow + reAct agent inside generator_context"""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent  # ✅ correct import

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


from langchain_core.tools import tool  # ✅ use decorator instead of Tool class
from typing import List
from langchain_core.documents import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def _build_tools(self):
        _retriever = self.retriever  # ✅ different name for the captured instance

        @tool
        def document_search(query: str) -> str:
            """Fetch passages from indexed corpus for the given query."""
            docs: List[Document] = _retriever.invoke(query)  # ✅ calls the actual retriever
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        @tool
        def wikipedia(query: str) -> str:
            """Search Wikipedia for general knowledge about the given query."""
            wiki = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
            )
            return wiki.run(query)

        return [document_search, wikipedia]

    def _build_agent(self):
        from langgraph.prebuilt import create_react_agent
        tools = self._build_tools()
        system_prompt = (
    "You are a helpful RAG agent. "
    "Prefer 'document_search' for user-provided docs; use 'wikipedia' for general knowledge. "
    "Return only the final useful answer."
)
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", [])
        answer = getattr(messages[-1], "content", None) if messages else None

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )