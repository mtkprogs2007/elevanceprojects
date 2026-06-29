# Integrated Enterprise AI Engineering Suite

## Problem Statement
Deploying individual conversational nodes and analytical frameworks creates architectural fragmentation, high operational token overhead, and context isolation. This unified engineering repository addresses these inefficiencies by consolidating six distinct machine learning and large language model pipelines under a centralized dashboard matrix interface (`bigapp.py`).

## Core Portfolio Architecture & Feature Breakdown

| Module Directory | Feature Scope / Task Experiment | Primary Technical Stack |
| :--- | :--- | :--- |
| **gemini_chatbot/** | Task 1: RAG Knowledge base parsing system context strings. | FAISS Vector Store, LangChain, Gemini API |
| **multi-modal/** | Task 2: Visual reasoning engine analyzing image payloads with context tokens. | Gemini Multi-Modal Vision API, Streamlit |
| **med_qa_chatbot/** | Task 3: Clinical QA system featuring environment credential routing. | Dotenv, Google GenerativeAI SDK |
| **arxiv_expert_chatbot/** | Task 4: arXiv Research Discovery engine parsing Kaggle metadata sets. | DataFrames, Snapshot Parsing, Streamlit |
| **multilingual_chatbot/** | Task 5: Cross-lingual localization and real-time localized synthesis layers. | Regional Translation Maps, Streamlit UI |
| **sentiment_chatbot/** | Task 6: Sentiment-aware prioritization tracker routing inbound log alerts. | TextBlob/Classifier Analytics, Port Mapping |

## Methodology & Reproducibility
* **Preprocessing & Context Truncation:** Raw workspace text data is chunked using semantic tokenizers to prevent database context vector overflow.
* **API Validation:** Fixed potential 401 request faults by replacing implicit Application Default Credentials (ADC) configurations with explicit local environment loaders.
* **Session Port Isolation:** Child execution paths run on unique, isolated localhost ports (`8502`–`8507`) to eliminate thread sharing or address assignment conflicts.

## Getting Started & Execution
To initialize the parent workspace panel, launch the master script from the repository root:
```bash
python -m streamlit run bigapp.py