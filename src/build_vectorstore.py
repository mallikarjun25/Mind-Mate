# build_vectorstore.py (polished)
"""
Builds FAISS vectorstore from mental health PDFs.
"""
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data")
DB_FAISS_PATH = os.path.join(ROOT_DIR, "vectorstore", "db_faiss")

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main():
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    main()
