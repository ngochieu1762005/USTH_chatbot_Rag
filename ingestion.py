import os
# from dotenv import load_dotenv
from time import sleep

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load env if needed
# load_dotenv()

# Local embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load PDFs
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

# Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Persist directory (DB will be stored locally)
PERSIST_DIR = "chroma_db"

# ---- Chroma Ingestion (Batch Safe Version) ----
def embed_in_batches(docs, batch_size=10, delay=0.2):
    # initialize / load existing chroma db
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model
    )

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            vector_store.add_documents(batch)
            vector_store.persist()
            print(f" Batch {i//batch_size + 1} stored successfully")
        except Exception as e:
            print(f" Error embedding batch {i//batch_size + 1}: {e}")

        sleep(delay)

embed_in_batches(docs)
print("Data ingestion complete!")
