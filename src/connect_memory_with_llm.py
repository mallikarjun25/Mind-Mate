import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment token
HF_TOKEN = os.environ.get("HF_TOKEN")

# Define available models
LLM_REPOS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "falcon": "tiiuae/falcon-7b-instruct"
}

def load_llm(model_key):
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPOS[model_key],
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def query_model(user_query, model_key):
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(model_key),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    response = qa_chain.invoke({'query': user_query})
    return response

if __name__ == "__main__":
    model_choice = input("Choose model (mistral/llama3/falcon): ").strip().lower()
    user_query = input("Enter your query: ")
    output = query_model(user_query, model_choice)
    print("RESULT:", output["result"])
    print("SOURCE DOCUMENTS:", output["source_documents"])
