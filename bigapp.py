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

# Explicit file names based on your screenshots
file_map = {
    "gemini_chatbot": "main.py",
    "multi-modal": "main.py",
    "med_qa_chatbot": "app.py",
    "arxiv_expert_chatbot": "app.py",
    "multilingual_chatbot": "app.py",
    "sentiment_chatbot": "app.py"
}

# Unique target ports to prevent address conflicts
port_map = {
    "gemini_chatbot": "8502",
    "multi-modal": "8503",
    "med_qa_chatbot": "8504",
    "arxiv_expert_chatbot": "8505",
    "multilingual_chatbot": "8506",
    "sentiment_chatbot": "8507"
}

if st.button(f"Launch {target_folder} Module"):
    # Target absolute base directory paths cleanly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir_path = os.path.join(base_dir, target_folder)
    script_name = file_map[target_folder]
    
    st.success(f"Spinning up {target_folder} on port {port_map[target_folder]}...")
    
    # Launch sub-process cleanly handling separated directory states
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", script_name, "--server.port", port_map[target_folder]],
        cwd=target_dir_path
    )