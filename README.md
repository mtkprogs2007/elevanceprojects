# Internship Project Directory: AI Chatbot Frameworks

Welcome to the primary repository containing my core deliverables for the AI Chatbot Development internship. This repository features six standalone AI engine architectures built to process domain-specific knowledge bases, multimodal visual datasets, and conversational workflows across various local and cloud environments.

---
### Download the Core Datasets
Due to storage constraints and file size limits on GitHub and personal cloud drives, the massive dataset files are omitted from the remote repository tracking structures. You can retrieve them directly from their open-source platforms:

* **arXiv Metadata Snapshot** (For `arxiv_expert_chatbot`): 
  Download the official 1 GB data stream file (`arxiv-metadata-oai-snapshot.json`) directly from [Kaggle: arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). Place the raw JSON file directly inside your local `arxiv_expert_chatbot/` directory before running the scripts.
* **Local Knowledge Bases** (For `gemini_chatbot` & `multi-modal`):
  The target standard reference `dataset.csv` tracking tables are already safely committed directly inside their respective task directories.
## 📁 Repository Structure Overview

Each sub-module contains its own entry scripts and environmental configuration definitions:

* **`arxiv_expert_chatbot/`**: Advanced semantic processing terminal querying and indexing academic research metadata seamlessly.
* **`gemini_chatbot/`**: Core LLM contextual interface matching retrieval-augmented generation concepts with custom local index storage arrays.
* **`med_qa_chatbot/`**: Precision clinical QA assistant utilizing medical domain data inputs to serve contextual medical responses.
* **`multi-modal/`**: Computer vision processing system extracting structural metrics and semantic details directly from image inputs.
* **`multilingual_chatbot/`**: Real-time localized contextual engine providing dynamic cross-lingual translations without context leakage.
* **`sentiment_chatbot/`**: Customer emotion processing framework assessing context-driven behavioral trends over long conversation threads.

---

## 🛠️ General Installation & Setup Instructions

To provision your environment to verify these execution scripts, follow these instructions:

### 1. Set Up Your API Keys
Create a local `.env` file within the folder of the specific task you want to execute using the template below:
```env
GOOGLE_API_KEY="your_validated_api_key_here"

```

### 2. Dependency Management

Navigate inside any target sub-directory and install the explicit dependency packages via pip:

```bash
cd target_chatbot_folder
pip install -r requirements.txt

```

### 3. Execution Interface

Execute the primary user interface using either standard Python runtimes or Streamlit deployment parameters depending on the specific task design:

```bash
# For standard scripts
python main.py

# For web application dashboards
streamlit run app.py

```

---

*Developed as part of the Advanced GenAI and Agentic workflows training framework.*

```

---

