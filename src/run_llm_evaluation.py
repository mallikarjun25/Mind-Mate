import os
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from textblob import TextBlob

# Paths
EVAL_QUESTIONS_PATH = "../evaluation/evaluation_questions.csv"
DB_FAISS_PATH = "./vectorstore/db_faiss"
OUTPUT_CSV_PATH = "../evaluation/llm_comparison_with_sentiment.csv"

# Load evaluation questions
questions_df = pd.read_csv(EVAL_QUESTIONS_PATH)

# Load GPT-2 tokenizer/model for perplexity
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Function to compute perplexity
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# Models to evaluate
LLM_REPOS = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "LLaMA3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Falcon": "tiiuae/falcon-7b-instruct"
}

# Custom prompt
prompt = PromptTemplate(
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

# Load FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load HuggingFace LLM
def load_llm(repo_id):
    hf_token = os.environ.get("HF_TOKEN")
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

# Hallucination checker
def hallucinated(answer, context):
    answer_snippets = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    hallucinated_parts = [s for s in answer_snippets if s.lower() not in context.lower()]
    return len(hallucinated_parts) / max(len(answer_snippets), 1)

# Evaluation loop
results = []
for model_name, repo_id in LLM_REPOS.items():
    print(f"\n🔍 Running model: {model_name}")
    llm = load_llm(repo_id)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    for _, row in questions_df.iterrows():
        question = row["question"]
        try:
            response = qa_chain.invoke({'query': question})
            answer = response["result"]
            context_text = " ".join([doc.page_content for doc in response["source_documents"]])
            sentiment = TextBlob(answer).sentiment
            halluc_rate = hallucinated(answer, context_text)
            perplexity = compute_perplexity(answer)

            results.append({
                "model": model_name,
                "question": question,
                "response": answer,
                "response_length": len(answer),
                "sentiment_polarity": sentiment.polarity,
                "sentiment_subjectivity": sentiment.subjectivity,
                "hallucination_rate": round(halluc_rate, 2),
                "perplexity": round(perplexity, 2)
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "question": question,
                "response": f"Error: {str(e)}",
                "response_length": 0,
                "sentiment_polarity": 0,
                "sentiment_subjectivity": 0,
                "hallucination_rate": 1.0,
                "perplexity": 100.0
            })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\n✅ Evaluation complete! Results saved to {OUTPUT_CSV_PATH}")