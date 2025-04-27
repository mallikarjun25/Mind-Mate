import os
import warnings
import chainlit as cl
from huggingface_hub import hf_hub_download
from textblob import TextBlob
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import logging

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific Huggingface and Chainlit minor warnings
warnings.filterwarnings("ignore", message="Translation file for en-IN not found.*")
warnings.filterwarnings("ignore", message="Translated markdown file for en-IN not found.*")

# Set logging to show only important errors
logging.getLogger("chainlit").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


DB_FAISS_PATH = r"C:\\Users\\maith1\\Downloads\\mind_mate\\Mind-Mate\\src\\vectorstore\\db_faiss"
MODEL_CACHE_DIR = r"C:\\Users\\maith1\\.cache\\huggingface\\hub"

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
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
    input_variables=["context", "question"]
)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def download_models():
    for name, (repo_id, filename) in WEIGHTS.items():
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            library_name="llama-cpp"
        )
        DOWNLOADED_MODELS[name] = local_path

def load_llm(name: str):
    model_path = DOWNLOADED_MODELS.get(name)
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model file not found for {name}")

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

    # Special fix: if loading Phi-3-mini, enable rope_scaling=dynamic
    if "Phi-3" in name:
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

@cl.on_chat_start
async def start():
    await cl.Message(content="⚡ Loading MindMate fast mode...").send()
    try:
        download_models()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        perp_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

        # Preload ALL LLMs and Chains
        models = {}
        chains = {}
        for name in WEIGHTS:
            llm = load_llm(name)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            models[name] = llm
            chains[name] = chain

        cl.user_session.set("chains", chains)
        cl.user_session.set("tokenizer", tokenizer)
        cl.user_session.set("perp_model", perp_model)

        await cl.Message(content="✅ MindMate ready! Ask your question now.").send()
    except Exception as e:
        await cl.Message(content=f"Initialization error: {e}").send()

@cl.on_message
async def main(message):
    chains = cl.user_session.get("chains")
    tokenizer = cl.user_session.get("tokenizer")
    perp_model = cl.user_session.get("perp_model")
    user_query = message.content

    if not (chains and tokenizer and perp_model):
        await cl.Message(content="Error: Session not initialized.").send()
        return

    for name, chain in chains.items():
        try:
            actual_query = user_query
            if "Phi-3" in name:
                actual_query = "<s> " + user_query  # Add BOS token for Phi-3

            out = chain.invoke({"query": actual_query})

            answer = out.get("result", "")
            context = " ".join(doc.page_content for doc in out.get("source_documents", []))

            sentiment = TextBlob(answer).sentiment
            hall_rate = hallucination_rate(answer, context)
            perplex = compute_perplexity(answer, tokenizer, perp_model)

            response = f"""**{name}**
**Answer:** {answer}

Polarity: {sentiment.polarity:.3f} | Subjectivity: {sentiment.subjectivity:.3f}
Hallucination Rate: {hall_rate:.2f} | Perplexity: {perplex:.2f}
"""
            await cl.Message(content=response).send()

        except Exception as e:
            await cl.Message(content=f"Skipping {name}: {e}").send()
