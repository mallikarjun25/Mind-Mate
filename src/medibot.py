import os
import torch
import streamlit as st
from textblob import TextBlob
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="Mind Mate", page_icon="🧠", layout="wide")

# === CONFIGURATION ===
DB_FAISS_PATH = r"C:\\Users\\maith1\\Downloads\\mind_mate\\Mind-Mate\\src\\vectorstore\\db_faiss"
MODEL_CACHE_DIR = r"C:\\Users\\maith1\\.cache\\huggingface\\hub"

# App title and description
st.title("🧠 Mind Mate")
st.markdown("Conversations Made Easy!")

# Model download details
WEIGHTS = {
    "Mistral": (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q4_0.gguf"
    ),
    "LLaMA": (
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "llama-2-7b-chat.Q4_0.gguf"
    ),
    "Phi-3-mini": (
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf"
    ),
}

# Prompt template for RetrievalQA
PROMPT = PromptTemplate(
    template="""
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say \"I don't know\".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
    input_variables=["context", "question"]
)

# Load FAISS index
@st.cache_resource
def load_vectorstore():
    with st.spinner("Loading vector database..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

# Load GPT-2 for perplexity
@st.cache_resource
def load_perplexity_model():
    with st.spinner("Loading perplexity model..."):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        return tokenizer, model

# Compute perplexity
def compute_perplexity(text: str, tokenizer, model) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

# Compute hallucination rate
def hallucination_rate(answer: str, context: str) -> float:
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    hallucinated = [s for s in sentences if s.lower() not in context.lower()]
    return round(len(hallucinated) / max(len(sentences), 1), 2)

# Download models if not already present
@st.cache_resource
def download_model(name, repo_id, filename):
    progress_text = f"Downloading {name}..."
    progress_bar = st.progress(0)
    
    with st.spinner(progress_text):
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            library_name="llama-cpp"
        )
        progress_bar.progress(100)
    
    return local_path

# Load LLM using llama-cpp-python
@st.cache_resource
def load_llm(model_path, name):
    with st.spinner(f"Loading {name} model..."):
        return LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            max_tokens=512,
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            n_threads=12,
            verbose=False
        )

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_models = []
for model_name in WEIGHTS.keys():
    if st.sidebar.checkbox(model_name, value=True):
        selected_models.append(model_name)

# Query input
query = st.text_input("Enter your question:", "")

# Run button
if st.button("Ask") and query and selected_models:
    # Load resources
    db = load_vectorstore()
    tokenizer, perp_model = load_perplexity_model()
    
    # Set up columns for results
    cols = st.columns(len(selected_models))
    
    # Process each selected model
    for i, name in enumerate(selected_models):
        with cols[i]:
            st.subheader(name)
            
            # Download model if needed
            repo_id, filename = WEIGHTS[name]
            try:
                model_path = download_model(name, repo_id, filename)
                
                # Load model and build chain
                llm = load_llm(model_path, name)
                
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 1}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Process query
                with st.spinner(f"Getting response from {name}..."):
                    out = chain.invoke({"query": query})
                    answer = out.get("result", "")
                    context = " ".join(doc.page_content for doc in out.get("source_documents", []))
                
                # Compute metrics
                sentiment = TextBlob(answer).sentiment
                hall_rate = hallucination_rate(answer, context)
                perplex = compute_perplexity(answer, tokenizer, perp_model)
                
                # Display results
                st.markdown("### Answer")
                st.write(answer)
                
                # Display metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Polarity", f"{sentiment.polarity:.3f}")
                    st.metric("Subjectivity", f"{sentiment.subjectivity:.3f}")
                with metrics_col2:
                    st.metric("Hallucination Rate", f"{hall_rate:.2f}")
                    st.metric("Perplexity", f"{perplex:.2f}")
                
                # Option to view source context
                with st.expander("View Source Context"):
                    st.write(context)
                    
            except Exception as e:
                st.error(f"Error processing {name}: {str(e)}")
else:
    if not query:
        st.info("Please enter a question above to get started.")
    elif not selected_models:
        st.warning("Please select at least one model from the sidebar.")