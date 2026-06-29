import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="ElevanceSkills AI Suite Hub", layout="wide")

st.sidebar.title("🚀 AI Engineering Portal")
st.sidebar.markdown("---")

task = st.sidebar.radio(
    "Select Feature Module:",
    [
        "Task 1: RAG Knowledge Engine",
        "Task 2: Multi-Modal Dashboard",
        "Task 3: Clinical QA Module",
        "Task 4: arXiv Intelligence Engine",
        "Task 5: Multilingual Chatbot",
        "Task 6: Sentiment Support Tracker"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Select a feature block above to load the corresponding training expansion model.")

# Folder mapping configuration
module_map = {
    "Task 1: RAG Knowledge Engine": "gemini_chatbot",
    "Task 2: Multi-Modal Dashboard": "multi-modal",
    "Task 3: Clinical QA Module": "med_qa_chatbot",
    "Task 4: arXiv Intelligence Engine": "arxiv_expert_chatbot",
    "Task 5: Multilingual Chatbot": "multilingual_chatbot",
    "Task 6: Sentiment Support Tracker": "sentiment_chatbot"
}

target_folder = module_map[task]
st.title(f"🛠️ Displaying {task}")
st.write(f"Active Root Workspace: `/{target_folder}`")

# Run child streamlit instances cleanly using a sub-process bridge
if st.button(f"Launch {target_folder} Module"):
    script_path = os.path.join(target_folder, "app.py")
    if not os.path.exists(script_path):
        script_path = os.path.join(target_folder, "main.py")
        
    if os.path.exists(script_path):
        st.success(f"Spinning up instance on a separate workspace port...")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
    else:
        st.error(f"Could not locate an app.py or main.py entrypoint inside {target_folder}/")