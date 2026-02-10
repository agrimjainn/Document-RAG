# from langgraph.graph import StateGraph, END
# from src.state.rag_state import RAGState
# from src.nodes.reactnode import RAGNodes # To do: change maybe

# class GraphBuilder:
#     "Builds and manages the langgraph workflow"

#     def __init__(self, retriever, llm):
#         """
#         Initialize graph builder
        
#         Args:
#             retriever: Documents retriever instance
#             llm: Large Language Model instance
#         """

#         self.nodes = RAGNodes(retriever=retriever, llm=llm)  # ✅ instantiate with args
#         self.graph = None

#     def build(self):
#         """
#         Build the RAG workflow graph

#         Returns:
#             Complied graph instance
#         """
#         # Create State Graph
#         builder = StateGraph(RAGState)

#         # Add Nodes
#         builder.add_node("retriever", self.nodes.retrieve_docs)
#         builder.add_node("responder", self.nodes.generate_answer)

#         # Set entry point
#         builder.set_entry_point("retriever")

#         # Add edges
#         builder.add_edge("retriever","responder")
#         builder.add_edge("responder", END)

#         # Compile graph
#         self.graph = builder.compile()
#         return self.graph
    
#     def run(self, question: str) -> dict:
#         """
#         Run the RAG workflow

#         Args:
#             question: User Question

#         Returns:
#             Final state with answer
#         """

#         if self.graph is None:
#             self.build()
#         inital_state = RAGState(question=question)
#         return self.graph.invoke(inital_state)


from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnode import RAGNodes

class GraphBuilder:
    "Builds and manages the langgraph workflow"

    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever=retriever, llm=llm)  # ✅ instance, not class
        self.graph = None

    def build(self):
        builder = StateGraph(RAGState)

        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)