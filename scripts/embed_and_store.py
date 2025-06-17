from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os

# Paths
chunks_file = "data/chunks.json"
faiss_index_folder = "embeddings/faiss_index"

# Step 1: Load chunks
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Step 2: Set up embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Step 3: Create FAISS index from chunks
print("🔍 Creating embeddings...")
faiss_index = FAISS.from_texts(texts=chunks, embedding=embedding_model)

# Step 4: Save FAISS index
faiss_index.save_local(faiss_index_folder)
print(f"✅ Embeddings saved to {faiss_index_folder}")
