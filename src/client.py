from pathlib import Path
from typing import TypedDict, Annotated
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from data import DocumentStore, PDFDocumentStore

class ChatClient:

    class State(TypedDict):
        messages: Annotated[list[HumanMessage | AIMessage], add_messages]
        context: list[Document]

    def __init__(self, model, store):
        self._model = model
        self._store = store
        self._init_llm()
    
    def _init_llm(self):
        self._llm =  ChatOllama(model=self._model) # check if we want other parameters
        self._prompt = ChatPromptTemplate([
            ('system', "You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer just say that you don't know. Use three sentances maximum and keep your answers concise."),
            ('human', "Question: {question}\nContext: {context}\nAnswer:")
        ], input_variables=["question", "context"])

        retrieve = ('retrieve', lambda state, client=self: client.retrieve(state))
        generate = ('generate', lambda state, client=self: client.generate(state))
        graph_builder = StateGraph(self.State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self._graph = graph_builder.compile()

    def retrieve(self, state : State) -> State:
        question = state["messages"][-1] if state["messages"] else AIMessage(content="Welcome, how can I help you today?")
        match question:
            case AIMessage():
                return state
            case HumanMessage():
                retrieved_docs = self._store.query(question.content, None)
                return self.State(messages=state["messages"], context=retrieved_docs)
            case _:
                return state # Not sure what message we received, do nothing

    def generate(self, state : State) -> State:
        doc_content = '\n\n'.join(doc.page_content for doc in state['context'])
        question = state["messages"][-1] if state["messages"] else AIMessage(content="Welcome, how can I help you today?")
        messages = self._prompt.invoke({'question' : question.content, 'context' : doc_content})
        response = self._llm.invoke(messages)

        return self.State(
                messages = state["messages"] + [AIMessage(content=response.content)],
                context=state["context"]
        )

    def ask(self, question : str) -> State:
        return self._graph.invoke(self.State(messages=[HumanMessage(content=question)]))



if __name__ == "__main__":
    BASE_PATH = Path("../data")
    DEFAULT_COLLECTION = Path("publications")

    load_dotenv()
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
    PERSIST_DIRECTORY=os.getenv("PERSIST_DIRECTORY")
    CHAT_MODEL=os.getenv("LLM_MODEL")

    store = PDFDocumentStore(EMBEDDING_MODEL, PERSIST_DIRECTORY)
    print("Initialized store")
    #pdfs = PDFDocumentStore.find_pdfs(BASE_PATH, DEFAULT_COLLECTION)
    #store.load(pdfs)
    client = ChatClient(CHAT_MODEL, store)
    print("Initialized chat client.")
    question = input("Ask a question about the paper:\n>> ") #"What is crystal island?"
    result = client.ask(question)
    print(f"There are {len(result['context'])} documents in the context.")
    print(f"Answer: {result['messages'][-1].content}")

