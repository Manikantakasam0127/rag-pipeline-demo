# basic document ingestion
# TODO: add proper error handling

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    # tried 1000 chunk size first but results were worse
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
