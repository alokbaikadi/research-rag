from typing import Optional
from pathlib import Path
from uuid import uuid4
import os

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentStore:
    _model : str
    _persist_path : str
    _store_db : Chroma

    def __init__(self, model : str, persist_path: str):
        self._model = model
        self._persist_path = persist_path
        self._collection = "default collection"

        self._init_vectorstore()


    def _init_vectorstore(self):
        self._embeddings = OllamaEmbeddings(model=self._model)
        self._store_db = Chroma(
            collection_name ="Documents", # TODO: fix this
            embedding_function=self._embeddings,
            persist_directory=self._persist_path
        )

    def _store(self, chunks) -> None:
        print("Creating embeddings")
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        self._store_db.add_documents(documents=chunks, ids=uuids)

    def load(self, paths: list[Path]):
        pass

    def query(self, question : str, collection: str) -> Document:
        return self._store_db.similarity_search(question)



class PDFDocumentStore(DocumentStore):
    _docs : list[Document] = []

    def load(self, paths : list[Path]) -> list[Document]:
        print("Loading Documents")
        self._docs = []
        for path in tqdm(paths):
            loader = PyPDFLoader(path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self._docs.extend(splitter.split_documents(docs))
        self._store(self._docs)

    @staticmethod
    def find_pdfs(base : Path, collection : Optional[Path] = None) -> list[Path]:
        docs = []

        if collection is not None:
            path = base / collection
        else:
            path = base
        for filename in os.listdir(path):
            _,ext = os.path.splitext(filename)
            if ext == ".pdf":
                print(f"Found: {filename}")
                docs.append(path / filename)
        return docs



if __name__ == "__main__":
    BASE_PATH = Path("../data")
    DEFAULT_COLLECTION = Path("publications")

    load_dotenv()
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
    PERSIST_DIRECTORY=os.getenv("PERSIST_DIRECTORY")

    store = PDFDocumentStore(EMBEDDING_MODEL, PERSIST_DIRECTORY)
    pdfs = PDFDocumentStore.find_pdfs(BASE_PATH, DEFAULT_COLLECTION)
    store.load(pdfs)
    QUERY = "What is crystal island?"
    print(f"query: {QUERY}")
    result = store.query(QUERY, None)
    print(f"query result: {len(result)} chunks found")
    print(f"first result: {result[0]}")
