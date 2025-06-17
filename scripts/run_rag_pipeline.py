import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load FAISS index
index_folder = "embeddings/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(index_folder, embedding_model, allow_dangerous_deserialization=True)

# Step 2: Load Groq's LLaMA model
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",  # You can also try mixtral or gemma if you prefer
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 3: Define prompt template
prompt_template = """You are an assistant helping users understand the EU AI Act.
Use the following context to answer the question clearly and accurately.

Context:
{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Step 4: Setup RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", k=5),
    return_source_documents=True
)

# Step 5: CLI loop
while True:
    query = input("\nAsk a question about the EU AI Act (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain(query)
    print("\n🔍 Answer:")
    print(result["result"])
