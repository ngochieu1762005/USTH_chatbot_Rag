RAG Team Intro AI – Local PDF RAG Chatbot

This project builds a Retrieval-Augmented Generation (RAG) chatbot that runs locally and answers questions based on your PDF documents.

The system:

Loads PDFs from the data/ directory

Splits content into chunks

Creates embeddings using HuggingFace

Stores everything in ChromaDB (persistent with tenant + database)

Provides a Streamlit chatbot interface

⚙️ Technologies

Python

LangChain

ChromaDB (>= 1.3.x)

HuggingFace Embeddings

Streamlit

PDF Loader