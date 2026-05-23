from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def run_ingestion():
    # 1. Φόρτωση του νομικού κειμένου
    loader = TextLoader("data/norm_hackatext.txt", encoding='utf-8')
    documents = loader.load()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)

    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("data/faiss_index")
    print("✅ FAISS index created and saved in data/faiss_index")

if __name__ == "__main__":
    run_ingestion()