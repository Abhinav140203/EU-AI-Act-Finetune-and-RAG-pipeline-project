from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Input and output paths
input_file = "data/combined_text.txt"
output_file = "data/chunks.json"

# Step 1: Load text
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Step 2: Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # length of each chunk (characters)
    chunk_overlap=100,      # slight overlap between chunks for better context
    length_function=len
)

chunks = splitter.split_text(full_text)

# Step 3: Save chunks
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"✅ Chunking complete: {len(chunks)} chunks saved to {output_file}")
