import streamlit as st
import google.generativeai as genai
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

# Initialize application environment and API routing keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ Environment Configuration Error: GOOGLE_API_KEY variable is missing.")

DATASET_FILE = "arxiv-metadata-oai-snapshot.json"

def stream_arxiv_kaggle_dataset(search_term="", max_results=8):
    """
    Streams the 3GB Kaggle dataset line-by-line to minimize memory footprint.
    Applies real-time domain filtering and structural keyword validation.
    """
    if not os.path.exists(DATASET_FILE):
        st.sidebar.warning("⚠️ Kaggle JSON file not detected locally. Displaying baseline backup.")
        return load_static_backup()

    records = []
    search_lower = search_term.lower().strip() if search_term else ""

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
                categories = paper.get("categories", "")
                
                # Filter specifically for target Computer Science (cs) categories
                if any(cs_cat in categories for cs_cat in ["cs.AI", "cs.LG", "cs.CV", "cs.CL"]):
                    paper_id = paper.get("id", "")
                    
                    # Chronological optimization: Skip historical data (2007-2017) to prioritize deep learning literature
                    if search_lower and any(paper_id.startswith(f"{str(y).zfill(2)}") for y in range(7, 18)):
                        continue
                        
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()
                    
                    if search_lower:
                        if search_lower in title or search_lower in abstract:
                            records.append(parse_paper_entry(paper))
                    else:
                        records.append(parse_paper_entry(paper))
                        
                if len(records) >= max_results:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    return pd.DataFrame(records)

def parse_paper_entry(paper):
    """Maps raw dataset values into structured metadata components."""
    cat_mapping = {
        "cs.AI": "Artificial Intelligence",
        "cs.LG": "Machine Learning",
        "cs.CV": "Computer Vision",
        "cs.CL": "Computation & Language"
    }
    raw_cats = paper.get("categories", "").split()
    concepts = [cat_mapping[c] for c in raw_cats if c in cat_mapping]
    if not concepts:
        concepts = ["Computer Science Research"]

    return {
        "id": paper.get("id", "N/A"),
        "authors": paper.get("authors", "Unknown Authors"),
        "title": paper.get("title", "Untitled Paper").strip(),
        "categories": paper.get("categories", "cs.AI"),
        "concepts": concepts,
        "abstract": paper.get("abstract", "No abstract available.").strip()
    }

def load_static_backup():
    """Fallback structural framework configuration used if the main file is missing."""
    data = {
        "id": ["1801.00076"],
        "authors": "Tong Guo et al.",
        "title": "Bidirectional Attention for SQL Generation",
        "categories": "cs.CL, cs.LG",
        "concepts": [["Machine Learning", "Computation & Language"]],
        "abstract": "Generating structural query language (SQL) queries from natural language is a long-standing open problem."
    }
    return pd.DataFrame(data)

def build_concept_network(filtered_df):
    """Constructs a relational intersection network map using NetworkX."""
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        paper_title = row['title']
        if len(paper_title) > 30:
            paper_title = paper_title[:27] + "..."
        concepts = row['concepts']
        
        for concept in concepts:
            G.add_edge(paper_title, concept)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                G.add_edge(concepts[i], concepts[j])
    return G

# --- UI LAYOUT & STYLE CONFIGURATION ---
st.set_page_config(page_title="arXiv Kaggle Intelligence Engine", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .research-card {
        background-color: #0f172a; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
    }
    /* Fixed viewport constraint to manage chat-scrolling behavior */
    .scroll-chat {
        height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 arXiv Knowledge Discovery Terminal")
st.markdown("##### *Production Environment: Local Kaggle Dataset Ingestion Pipeline*")
st.markdown("---")

col_data, col_chat = st.columns([1, 1])

with col_data:
    st.subheader("🔍 Local Kaggle Subcategory Filters")
    search_term = st.text_input("Query file records by keyword:", placeholder="Type a concept to parse the dataset...")
    
    with st.spinner("Processing local dataset file streams..."):
        filtered_df = stream_arxiv_kaggle_dataset(search_term=search_term, max_results=8)

    st.metric("Filtered Records Tracked", len(filtered_df))
    
    for idx, row in filtered_df.iterrows():
        st.markdown(f"""
        <div class='research-card'>
            <span style='color: #94a3b8; font-size: 11px;'>ID: {row['id']} | {row['authors'][:50]}</span>
            <h5 style='margin-top: 3px; margin-bottom: 6px; color: #f8fafc;'>{row['title']}</h5>
            <p style='font-size: 12px; color: #cbd5e1;'>{row['abstract'][:140]}...</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("🕸️ Algorithmic Concept Map")
    
    if not filtered_df.empty:
        concept_graph = build_concept_network(filtered_df)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        pos = nx.spring_layout(concept_graph, k=0.7, seed=42)
        node_colors = ['#4f46e5' if node in filtered_df['title'].apply(lambda x: x[:27]+"..." if len(x)>30 else x).values else '#06b6d4' for node in concept_graph.nodes()]
        
        nx.draw_networkx_nodes(concept_graph, pos, node_color=node_colors, node_size=180, ax=ax)
        nx.draw_networkx_edges(concept_graph, pos, edge_color='#334155', alpha=0.5, ax=ax)
        nx.draw_networkx_labels(concept_graph, pos, font_size=8, font_color='#f8fafc', ax=ax)
        
        plt.axis('off')
        st.pyplot(fig)

with col_chat:
    st.subheader("💬 Academic Dialogue Engine")
    
    if "arxiv_history" not in st.session_state:
        st.session_state.arxiv_history = []
        
    # Injecting the CSS-managed scroll viewport wrapper
    st.markdown('<div class="scroll-chat">', unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.arxiv_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)
            
    user_dialogue = st.chat_input("Prompt follow-up conceptual validations here...")
    
    if user_dialogue:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_dialogue)
        st.session_state.arxiv_history.append({"role": "user", "content": user_dialogue})
        
        # Structure the current workspace contents to pass as context constraints to the LLM
        context_block = "\n\n".join([
            f"Paper Title: {r['title']}\nAbstract: {r['abstract']}" for _, r in filtered_df.iterrows()
        ])
        
        system_architecture_prompt = f"""
        You are a distinguished research assistant specializing in Computer Science papers from the arXiv Kaggle dataset collection.
        Your goal is to explain advanced algorithmic methods, design choices, and academic research trends thoroughly and clearly.
        
        Use the following locally matched dataset papers to ground your analysis:
        {context_block}
        
        Structure your responses with clear bullet points, conceptual breakdowns, or step-by-step logic. Connect findings back to standard literature whenever appropriate.
        """
        
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Compiling academic response proofs..."):
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=system_architecture_prompt
                    )
                    chat_session = model.start_chat(history=[])
                    response = chat_session.send_message(user_dialogue)
                    st.markdown(response.text)
                
        st.session_state.arxiv_history.append({"role": "assistant", "content": response.text})
        st.rerun()