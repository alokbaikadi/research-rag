from dotenv import load_dotenv
from pathlib import Path
import os

from data import PDFDocumentStore
from client import ChatClient

def main():
    
    load_dotenv()
    BASE_PATH = Path(os.getenv("DATA_DIR"))
    DEFAULT_COLLECTION = Path("publications")

    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
    PERSIST_DIRECTORY=os.getenv("PERSIST_DIRECTORY")
    CHAT_MODEL=os.getenv("LLM_MODEL")

    store = PDFDocumentStore(EMBEDDING_MODEL, PERSIST_DIRECTORY)
    print("Initialized store")
    pdfs = PDFDocumentStore.find_pdfs(BASE_PATH, DEFAULT_COLLECTION)
    store.load(pdfs)
    client = ChatClient(CHAT_MODEL, store)
    print("Initialized chat client.")
    thread_id = client.get_thread()

    print("\n\nHello. I am a chatbot designed to answer questions about a research paper.")
    while True:
        try:
            question = input(">> ") #"What is crystal island?"
            result = client.ask(question, thread_id)
            print(f"\n# {result}")
        except EOFError: # Catch ctrl+d / EOF
            print("\n# Goodbye!")
            break


if __name__ == "__main__":
    main()

