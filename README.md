# MindMate 🧠💬

## Overview
MindMate is a **Retrieval-Augmented Generation (RAG)** based chatbot for **mental health support**, designed to deliver reliable, context-aware answers while minimizing hallucination and bias.

This project evaluates three free/open-source LLMs in a RAG setup:
- **Mistral-7B**
- **Llama-2-7B-Chat**
- **Phi-3-mini**

It provides:
- On-demand chatbot access (Chainlit interface).
- Batch evaluation across 10 domain-specific mental health questions.
- Comparative metrics like **perplexity**, **hallucination rate**, and **sentiment consistency**.

---

## Project Structure
```bash
Mind-Mate/
├── data/ # PDFs used as knowledge base
├── src/
    ├── build_vectorstore.py # Create FAISS vectorstore from PDFs
    ├── llm_evaluation.py # Batch evaluation of LLMs on questions
    ├── medibot.py # Chainlit chatbot app
├── requirements.txt # Project dependencies
├── README.md # Project overview and usage
```

---

## Setup Instructions
1. **Clone the repository**:
    ```bash
    git clone https://github.com/mallikarjun25/Mind-Mate
    cd Mind-Mate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare data**:
    - Mental health PDFs inside the `data/` folder.

4. **Build vectorstore**:
    ```bash
    python src/build_vectorstore.py
    ```

5. **Run Evaluation**:
    ```bash
    python src/llm_evaluation.py
    ```
    Results saved to `model_outputs_<timestamp>.csv`.

6. **Launch Chainlit App**:
    ```bash
    chainlit run src/medibot.py
    ```

---

## Models Used
| Model         | Source                                          | Quantization |
|---------------|--------------------------------------------------|--------------|
| Mistral-7B    | TheBloke/Mistral-7B-Instruct-v0.2-GGUF           | Q4_0         |
| Llama-2-7B    | TheBloke/Llama-2-7B-Chat-GGUF                    | Q4_0         |
| Phi-3-mini    | microsoft/Phi-3-mini-4k-instruct-gguf            | Q4           |

*All models downloaded from Hugging Face using `huggingface_hub`.*

---

## Evaluation Metrics
- **Perplexity**: Measures language fluency.
- **Hallucination Rate**: Percentage of sentences not grounded in retrieved context.
- **Sentiment Analysis**: Measures emotional polarity and subjectivity.

---

## References
- [SouLLMate: Enhancing Diverse Mental Health Support with Adaptive LLMs](https://arxiv.org/pdf/2410.16322)
- [LLM-Therapist: A RAG-Based Behavioral Therapist Assistant](https://ieeexplore.ieee.org/abstract/document/10901139)
