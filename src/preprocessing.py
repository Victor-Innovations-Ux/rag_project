##
## FREE PROJECT, 2025
## RAG_SYSTEM DEMO
## File description:
## Chunk docs
##

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error while loading the PDF '{file_path}': {e}")
        return []

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    try:
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"Error while splitting the documents: {e}")
        return []
