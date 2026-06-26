import os
import streamlit as st
from dotenv import load_dotenv

# Initialize environment configurations instantly
load_dotenv()

import google.generativeai as genai
from PIL import Image
from langchain_helper import parse_csv_to_records, match_local_context

# Configure the flagship API key securely
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Multi-Modal Reasoning Assistant",
    page_icon="👁️",
    layout="wide"
)

# ==========================================
# 🎨 HIGH-END DESIGN ENGINE (CUSTOM CSS)
# ==========================================
st.markdown("""
    <style>
        /* Main application background gradient */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1e38 50%, #111827 100%);
            color: #f8fafc;
        }
        
        /* Modernized Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95) !important;
            border-right: 2px solid #3b82f6;
        }
        
        /* Glowing Top Banner Headers */
        .main-title {
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0px 0px 20px rgba(139, 92, 246, 0.3);
        }
        
        /* Glassmorphic Metric Info Boxes */
        .info-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 5px solid #8b5cf6;
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        
        /* Interactive Input Bay Wrapper */
        .input-bay {
            background: rgba(30, 41, 59, 0.7);
            border: 1px dashed #3b82f6;
            padding: 1.5rem;
            border-radius: 16px;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }

        /* Customize Streamlit Buttons to have vibrant neon gradients */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.3s ease-in-out !important;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
            background: linear-gradient(90deg, #5c54ed 0%, #8b4eff 100%) !important;
        }
        
        /* Secondary Danger Button Overrides */
        div[data-testid="stSidebar"] div.stButton > button:first-child {
            background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%) !important;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Persistent storage configurations inside the RAM layer
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataset_records" not in st.session_state:
    st.session_state.dataset_records = []

# ==========================================
# ⚙️ SIDEBAR CONTROL NODE
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color: #3b82f6; font-weight:700;'>⚙️ Core Engine</h2>", unsafe_allow_html=True)
    st.caption("Advanced Multi-Modal Processing Terminal")
    
    if st.button("🧹 Wipe Conversation Cache", use_container_width=True):
        st.session_state.chat_history = []
        st.sidebar.success("Thread memory cache wiped!")
        st.rerun()
        
    st.divider()
    st.markdown("<h3 style='color: #8b5cf6;'>🧬 In-Memory Sync</h3>", unsafe_allow_html=True)
    st.caption("Mount administrative datasets natively into volatile RAM structures.")
    
    uploaded_csv = st.file_uploader("Select data file (.csv)", type=["csv"], key="panel_csv")
    
    if st.button("🚀 Execute Ingestion Loop", use_container_width=True):
        if uploaded_csv is not None:
            with st.spinner("Compiling documentation matrices..."):
                try:
                    records = parse_csv_to_records(uploaded_csv)
                    st.session_state.dataset_records = records
                    st.sidebar.success(f"💚 Active Matrix Nodes: {len(records)}")
                except Exception as e:
                    st.sidebar.error(f"Ingestion Aborted: {str(e)}")
        else:
            st.sidebar.warning("Attach a valid dataset context asset first.")

# ==========================================
# 🤖 MAIN WORKSPACE
# ==========================================
st.markdown("<h1 class='main-title'>🤖 Multi-Modal AI Reasoning Assistant</h1>", unsafe_allow_html=True)

# Dynamic Dashboard Status Counters
col1, col2 = st.columns(2)
with col1:
    status_color = "#10b981" if st.session_state.dataset_records else "#ef4444"
    status_text = "SYNCED" if st.session_state.dataset_records else "EMPTY"
    st.markdown(f"""
        <div class='info-card' style='border-left: 5px solid {status_color};'>
            <span style='color: #94a3b8; font-size: 0.9rem;'>KNOWLEDGE MATRIX STATUS</span><br>
            <strong style='font-size: 1.4rem; color: {status_color};'>{status_text} ({len(st.session_state.dataset_records)} Rows Locked)</strong>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class='info-card' style='border-left: 5px solid #3b82f6;'>
            <span style='color: #94a3b8; font-size: 0.9rem;'>ACTIVE CONVERSATIONAL TURNS</span><br>
            <strong style='font-size: 1.4rem; color: #3b82f6;'>{len(st.session_state.chat_history)} Thread Messages Locked</strong>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# Stably render the multi-turn historic conversational blocks
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Media Asset workspace module inside a highlighted block container
st.markdown("<div class='input-bay'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #ec4899; margin-top:0;'>🖼️ Visual Matrix Input Bay</h3>", unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "Drop analytical assets or systemic diagrams here for cross-modal verification passes:", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    preview_col, _ = st.columns([1, 3])
    with preview_col:
        st.image(uploaded_image, caption="Active Media Source Target", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Standardized Chat Input interface
user_query = st.chat_input("Input processing requests or run matrix scans...")

# ==========================================
# ADVANCED VALIDATION EXECUTION
# ==========================================
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
        
    formatted_history = ""
    for turn in st.session_state.chat_history:
        formatted_history += f"{turn['role'].capitalize()}: {turn['content']}\n"

    cleaned_context = match_local_context(user_query, st.session_state.dataset_records)

    # MANDATORY ASSIGNMENT CHECK: MISSING ASSET INTERCEPTION
    image_indicators = ["image", "picture", "diagram", "screenshot", "graph", "chart", "photo", "this look like"]
    is_referring_to_image = any(indicator in user_query.lower() for indicator in image_indicators)
    
    if is_referring_to_image and not uploaded_image:
        assistant_fallback = "⚠️ **Validation Warning:** I noticed your query references a visual asset or diagram, but the 'Visual Matrix Input Bay' is currently empty. Please upload the corresponding image file above so I can execute precise cross-modal analysis for you."
        
        with st.chat_message("assistant"):
            st.write(assistant_fallback)
            
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_fallback})
        st.stop()
        
    else:
        system_instruction = f"""You are an advanced, context-aware multi-modal AI reasoning assistant.

[BACKGROUND RELEVANT FAQS]
{cleaned_context if cleaned_context else "No direct database rows overlaps found."}

[CONVERSATIONAL THREAD MEMORY LOGS]
{formatted_history if formatted_history else "Initial entry turn."}"""

        # Build the payload contents cleanly
        content_payload = []
        full_text_prompt = f"{system_instruction}\n\nUser Prompt: {user_query}"
        content_payload.append(full_text_prompt)

        if uploaded_image:
            try:
                pil_image = Image.open(uploaded_image)
                content_payload.append(pil_image)
            except Exception as img_err:
                st.error(f"Image read anomaly: {str(img_err)}")

        with st.spinner("Processing cross-modal reasoning metrics via Gemini 2.5 Flash..."):
            try:
                # 🌟 FULLY UNIFIED GENERATION MODEL INTERFACE TO GEMINI 2.5 FLASH
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                response = model.generate_content(content_payload)
                
                with st.chat_message("assistant"):
                    st.markdown("#### **System Output:**")
                    st.write(response.text)
                    
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                
            except Exception as pipeline_error:
                with st.chat_message("assistant"):
                    st.error(f"Multi-Modal Pipeline Failure: {str(pipeline_error)}")