import os
import csv
from datetime import datetime
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
OUTPUT_CSV = f"model_outputs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

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

DOWNLOADED_MODELS = {}

PROMPT = PromptTemplate(
    template="""
You are a helpful, cautious mental health assistant.
Use only the information provided below to answer the user's question.
If you don't know the answer based on the provided information, say "I don't know." Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
    input_variables=["context", "question"]
)

QUESTIONS = [
    "How does group behavior in stressful situations contribute to psychological disorders such as anxiety and depression?",
    "What psychological mechanisms explain aggressive behavior in large crowds, according to emergent-norm theory?",
    "How does Judith Herman define the stages of recovery from complex psychological trauma?",
    "In what ways does political and social denial impact the treatment and acknowledgment of psychological trauma survivors?",
    "According to Herman, how does prolonged trauma disrupt an individual's sense of self and interpersonal relationships?",
    "What are the neurobiological effects of trauma on brain structures like the amygdala, hippocampus, and prefrontal cortex?",
    "How does trauma affect the body’s physiological regulation systems, according to van der Kolk’s research?",
    "Why is it often difficult for trauma survivors to verbalize their experiences through traditional talk therapy?",
    "What body-centered therapies are recommended for trauma healing beyond cognitive behavioral therapy?",
    "How does van der Kolk describe the relationship between traumatic memory and physical health symptoms?"
]

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def load_perplexity_model():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return tokenizer, model

def download_models():
    downloaded = {}
    for name, (repo_id, filename) in WEIGHTS.items():
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            library_name="llama-cpp"
        )
        downloaded[name] = local_path
    return downloaded

def load_llm(model_path: str, model_name: str):
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model file not found for {model_name} at {model_path}")

    params = {
        "model_path": model_path,
        "n_ctx": 4096,
        "max_tokens": 512,
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.95,
        "n_threads": 12,
        "verbose": False
    }
    if "Phi-3" in model_name:
        params["rope_scaling"] = "dynamic"

    return LlamaCpp(**params)

def compute_perplexity(text: str, tokenizer, model) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def hallucination_rate(answer: str, context: str) -> float:
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    hallucinated = [s for s in sentences if s.lower() not in context.lower()]
    return round(len(hallucinated) / max(len(sentences), 1), 2)

def ask_batch(model_name, model_path, questions):
    llm = load_llm(model_path, model_name)
    db = load_vectorstore()
    tokenizer, perp_model = load_perplexity_model()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    batch_results = []

    for question in questions:
        actual_query = question
        if "Phi-3" in model_name:
            actual_query = "<s> " + question

        out = chain.invoke({"query": actual_query})
        answer = out.get("result", "")
        context = " ".join(doc.page_content for doc in out.get("source_documents", []))

        sentiment = TextBlob(answer).sentiment
        hall_rate = hallucination_rate(answer, context)
        perplex = compute_perplexity(answer, tokenizer, perp_model)

        batch_results.append({
            "Question": question,
            "Model": model_name,
            "Answer": answer,
            "Perplexity": perplex,
            "Hallucination Rate": hall_rate,
            "Polarity": sentiment.polarity,
            "Subjectivity": sentiment.subjectivity
        })

    return batch_results

def main():
    downloaded_models = download_models()
    results = []

    for model_name, model_path in downloaded_models.items():
        model_results = ask_batch(model_name, model_path, QUESTIONS)
        results.extend(model_results)

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
