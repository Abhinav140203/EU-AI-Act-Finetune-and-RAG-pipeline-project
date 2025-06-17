import os
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Load environment variables ===
load_dotenv()

# === Step 1: Load FAISS retriever ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("embeddings/faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", k=5)

# === Step 2: Load Fine-Tuned TinyLLaMA ===
model_path = "tinyllama-euai-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# === Step 3: Prompt Template ===
prompt_template = """You are an assistant helping users understand the EU AI Act.
Use the following context to answer the question clearly and accurately.

Context:
{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# === Step 4: CLI Interface ===
while True:
    query = input("\nAsk a question about the EU AI Act (or type 'exit'): ")
    if query.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Inject context into the prompt
    full_prompt = prompt.format(context=context, question=query)

    # Generate answer
    response = generator(full_prompt)[0]["generated_text"]

    # Print result
    print("\n🔍 Answer:")
    print(response.replace(full_prompt, "").strip())  # Remove prompt from start
