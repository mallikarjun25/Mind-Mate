import os
from huggingface_hub import hf_hub_download
from textblob import TextBlob
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# === CONFIGURATION ===
DB_FAISS_PATH = r"C:\\Users\\maith1\\Downloads\\mind_mate\\Mind-Mate\\src\\vectorstore\\db_faiss"
MODEL_CACHE_DIR = r"C:\\Users\\maith1\\.cache\\huggingface\\hub"

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

# Store downloaded model paths
DOWNLOADED_MODELS = {}

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
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Load GPT-2 for perplexity
def load_perplexity_model():
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
def download_models():
    for name, (repo_id, filename) in WEIGHTS.items():
        print(f"Downloading {name} → {filename} …")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            library_name="llama-cpp"
        )
        DOWNLOADED_MODELS[name] = local_path
        print(f"  ✔ saved to {local_path}\n")

# Load LLM using llama-cpp-python
def load_llm(name: str):
    model_path = DOWNLOADED_MODELS.get(name)
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model file not found for {name}")
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

# Main interactive loop
def main():
    # Download models if needed
    download_models()

    # Load resources
    db = load_vectorstore()
    tokenizer, perp_model = load_perplexity_model()
    query = input("Enter your query: ")

    for name in WEIGHTS:
        try:
            llm = load_llm(name)
        except Exception as e:
            print(f"Skipping {name}: error loading model: {e}")
            continue

        # Build RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Invoke model
        out = chain.invoke({"query": query})
        answer = out.get("result", "")
        context = " ".join(doc.page_content for doc in out.get("source_documents", []))

        # Compute metrics
        sentiment = TextBlob(answer).sentiment
        hall_rate = hallucination_rate(answer, context)
        perplex = compute_perplexity(answer, tokenizer, perp_model)

        # Display results
        print(f"=== {name} ===")
        print(f"Answer: {answer}\n")
        print(f"Polarity: {sentiment.polarity:.3f} | Subjectivity: {sentiment.subjectivity:.3f}")
        print(f"Hallucination Rate: {hall_rate:.2f} | Perplexity: {perplex:.2f}\n")

if __name__ == "__main__":
    main()