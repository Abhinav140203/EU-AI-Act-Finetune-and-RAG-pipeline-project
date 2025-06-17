import os
import torch
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
index_path = "embeddings/faiss_index"
db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

# Prompt template
prompt_template = """You are an assistant helping users understand the EU AI Act.
Use the following context to answer the question clearly and accurately.

Context:
{context}

Question: {question}
Helpful Answer:"""

# Groq model setup
llm_groq = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Fine-tuned TinyLLaMA setup
@st.cache_resource
def get_tinyllama_pipeline():
    model_path = "tinyllama-euai-finetuned"
    use_cuda = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        return_full_text=False  # <--- ADD THIS
    )
    return pipe

# Streamlit UI
st.set_page_config(page_title="EU AI Act Q&A", page_icon="📜", layout="centered")
st.title("📜 EU AI Act Q&A Assistant")
st.markdown("Ask any question related to the EU AI Act. Select the model below:")

model_choice = st.radio("Choose a model", ["Groq (LLaMA3)", "Fine-Tuned TinyLLaMA"])
show_sources = st.checkbox("Show source chunks", value=True)

query = st.text_input("🔎 Enter your question", placeholder="What is EU AI Act")
submit = st.button("Get Answer")

if submit and query:
    retriever = db.as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    if model_choice == "Groq (LLaMA3)":
        with st.spinner("Asking Groq LLaMA3..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_groq,
                retriever=retriever,
                return_source_documents=show_sources
)
        response = qa_chain.invoke({"query": query})
        st.markdown(f"**🤖 Answer:** {response['result']}")


    else:
        with st.spinner("Generating answer from TinyLLaMA... please wait (may take 30–90s on CPU)"):
            try:
                pipe = get_tinyllama_pipeline()
                prompt_input = f"""You are a helpful assistant who answers questions about the EU AI Act.

                Context:
                {context}

                Question: {query}

                Answer:"""

                result = pipe(prompt_input)[0]["generated_text"]

# Remove the prompt from the generated output
                answer = result.replace(prompt_input, "").strip()

                st.markdown(f"**🦙 TinyLLaMA Answer:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")

    if show_sources:
        st.markdown("---")
        st.subheader("📚 Source Chunks")
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:500]}...")  # preview first 500 chars
