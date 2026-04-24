# FAISS vector search for RAG
# switched from basic similarity to hybrid search
# TODO: add re-ranking later

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_vectorstore(chunks):
    # using small model - cheaper and fast enough
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def retrieve_docs(vectorstore, query, k=5):
    # k=5 works better than k=3 for our docs
    results = vectorstore.similarity_search(query, k=k)
    return results
