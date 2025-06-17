import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

# Load vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("embeddings/faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load Groq LLaMA model
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template (same as before)
prompt_template = """You are an assistant helping users understand the EU AI Act.
Use the following context to answer the question clearly and accurately.

Context:
{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", k=5),
    return_source_documents=True
)

# EU AI Act questions to generate answers for
questions = [
    "What are the risk categories defined in the EU AI Act?",
    "What AI practices are prohibited?",
    "What obligations do high-risk AI system providers have?",
    "What is considered a foundation model under the EU AI Act?",
    "Are there rules for general-purpose AI systems?",
    "What are the penalties for non-compliance?",
    "What role do national authorities play?",
    "How does the Act define AI systems?",
    "Are biometric identification systems regulated?",
    "What is the conformity assessment procedure?",
    "How are users of AI systems affected?",
    "What transparency obligations exist?",
    "Are there sandbox environments supported?",
    "How does the Act relate to GDPR?",
    "Is emotion recognition banned?",
    "Are there specific rules for education or employment sectors?",
    "How does the Act address public safety?",
    "What are the key dates in the regulation?",
    "What enforcement mechanisms are included?",
    "Does the Act allow exemptions for research?",
]

# Output list
output_data = []

# Generate Q&A pairs
for question in questions:
    print(f"🔍 Generating answer for: {question}")
    try:
        response = qa_chain({"query": question})
        output_data.append({
            "instruction": question,
            "input": "",
            "output": response["result"].strip()
        })
    except Exception as e:
        print(f"❌ Failed for: {question}\n{e}")


# Save to file
with open("data/qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✅ Q&A dataset saved to data/qa_dataset.json")
