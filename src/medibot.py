import os
import streamlit as st
import pandas as pd
from textblob import TextBlob
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# === CONFIG ===
DB_FAISS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./vectorstore/db_faiss"))
LLM_REPOS = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "LLaMA3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Falcon": "tiiuae/falcon-7b-instruct"
}

# === Load tokenizer/model for perplexity ===
@st.cache_resource
def load_perplexity_model():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# === Prompt ===
def set_custom_prompt():
    return PromptTemplate(
        template="""
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know" — do not make up an answer.
Only use information from the context. Respond in a clear and direct way.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
        input_variables=["context", "question"]
    )

# === FAISS Vector DB ===
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# === Load LLM ===
def load_llm(repo_id):
    HF_TOKEN = os.environ.get("HF_TOKEN")
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

# === Hallucination Detection ===
def hallucination_rate(answer, context):
    answer_snippets = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    hallucinated = [s for s in answer_snippets if s.lower() not in context.lower()]
    return round(len(hallucinated) / max(len(answer_snippets), 1), 2)

# === Query Handler ===
def query_model(question, repo_id, retriever, tokenizer, gpt2_model):
    llm = load_llm(repo_id)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )
    result = qa_chain.invoke({'query': question})
    answer = result["result"]
    context = " ".join([doc.page_content for doc in result["source_documents"]])
    sentiment = TextBlob(answer).sentiment
    halluc_rate = hallucination_rate(answer, context)
    perplexity = compute_perplexity(answer, tokenizer, gpt2_model)

    return {
        "response": answer,
        "polarity": round(sentiment.polarity, 3),
        "subjectivity": round(sentiment.subjectivity, 3),
        "hallucination_rate": halluc_rate,
        "perplexity": round(perplexity, 2)
    }

# === UI ===
st.set_page_config(page_title="MindMate Multi-LLM QA", layout="wide")
st.title("🧠 MindMate - Multi-LLM Mental Health QA")
st.markdown("Ask a mental health question to compare answers from **Mistral**, **LLaMA3**, and **Falcon**.\n\nEach response is analyzed for sentiment, hallucination rate, and perplexity.")

user_query = st.text_input("🔍 Enter your question:")

if user_query:
    st.info("Running queries... please wait ⏳")
    retriever = get_vectorstore().as_retriever(search_kwargs={'k': 3})
    tokenizer, gpt2_model = load_perplexity_model()

    rows = []
    for model_name, repo in LLM_REPOS.items():
        try:
            result = query_model(user_query, repo, retriever, tokenizer, gpt2_model)
            rows.append({
                "Model": model_name,
                "Response": result["response"],
                "Polarity": result["polarity"],
                "Subjectivity": result["subjectivity"],
                "Hallucination Rate": result["hallucination_rate"],
                "Perplexity": result["perplexity"]
            })
        except Exception as e:
            rows.append({
                "Model": model_name,
                "Response": f"Error: {str(e)}",
                "Polarity": 0,
                "Subjectivity": 0,
                "Hallucination Rate": 1.0,
                "Perplexity": 100.0
            })

    df = pd.DataFrame(rows)

    # === Summary Table ===
    st.subheader("📊 Model Comparison")
    styled_df = df.style.format({
        "Polarity": "{:.2f}",
        "Subjectivity": "{:.2f}",
        "Hallucination Rate": "{:.2f}",
        "Perplexity": "{:.2f}"
    }).applymap(lambda v: "color: red" if isinstance(v, float) and v > 0.5 else "", subset=["Hallucination Rate"])
    st.dataframe(styled_df, use_container_width=True)

    # === Expanded View ===
    for row in rows:
        st.markdown(f"### 🤖 {row['Model']}")
        st.markdown(f"**Response:** {row['Response']}")
        st.markdown(f"*Polarity:* `{row['Polarity']}` | *Subjectivity:* `{row['Subjectivity']}` | *Hallucination Rate:* `{row['Hallucination Rate']}` | *Perplexity:* `{row['Perplexity']}`")
        st.markdown("---")
