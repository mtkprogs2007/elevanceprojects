import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
# Import your RAG function from your helper file
#  To this:
from langchain_helper import get_qa_chain, add_csv_file_to_db, query_relevant_context

# Configure the Google Generative AI SDK using your environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# ==========================================
# 1. PAGE CONFIGURATION & QoL STYLING
# ==========================================
st.set_page_config(
    page_title="Nullclass Support Platform",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Conversational Memory in Session State for Multi-Turn Tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 2. SIDEBAR PANEL (ADMIN & CONTROLS)
# ==========================================
with st.sidebar:
    st.title("⚙️ System Control Node")
    st.caption("Active Phase 2 Evaluation Node — Dynamic RAG Core")
    
    # QoL Feature: Instant memory reset button for clear presentation evaluations
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Conversation cache cleared!")
        st.rerun()
        
    st.divider()
    
    # Live Knowledge Sync (Admin Upload Layer)
    st.markdown("### 🧬 Live Knowledge Sync (Admin)")
    st.caption("Inject documentation or live corrections directly into the vector store.")
    
    src_type = st.selectbox("Select Knowledge Source Type", ["Bulk FAQ Matrix (.csv)"])
    
    uploaded_file = st.file_uploader(
        "Select bulk data table (Requires 'prompt' and 'response' headings)",
        type=["csv"],
        key="admin_csv_uploader"
    )
    
    if st.button("Execute Bulk Indexing", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing embeddings and hot-merging vector matrices..."):
                try:
                    # Pass the file buffer and name to your langchain_helper module
                    add_csv_file_to_db(uploaded_file, uploaded_file.name)
                    st.success(f"Successfully appended batch records from '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Ingestion Pipeline Error: {str(e)}")
        else:
            st.warning("Please upload a valid CSV file first.")

# ==========================================
# 3. MAIN CHAT INTERFACE LAYOUT
# ==========================================
st.title("💬 Chat Sandbox")
st.markdown("Submit queries below. The bot evaluates both historical training records and live uploaded updates simultaneously.")
st.caption("Ask about structural course timelines, payment facilities, or installations:")

# Display Active Conversation History Stream
st.divider()
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Multi-Modal Workspace: Optional Image Input Uploader for the Next Task
with st.expander("🖼️ Multi-Modal Input Bay (Optional Input Matrix)", expanded=False):
    uploaded_image = st.file_uploader(
        "Upload an image asset to analyze alongside your text question", 
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Visual Content Payload", width=400)

# Main Active Text Chat Input
user_query = st.chat_input("Enter customer phrase or testing query...")

# ==========================================
# 4. EXECUTION FLOW & PREDICTION PIPELINE
# ==========================================
if user_query:
    # 1. Display the user query instantly in the UI stream
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # 2. Retrieve matched semantic facts from the vector store using your QA Chain backend
   
    matched_facts = []
    try:
        # Use your dedicated context retriever function which uses .invoke() internally
        matched_facts = query_relevant_context(user_query)
    except Exception as e:
        st.warning(f"Vector Retrieval Notification: {str(e)}")

    # 3. Process and cleanse document text snippets to remove literal '\n' escape characters
    
    cleaned_context = ""
    if matched_facts:
        if isinstance(matched_facts, str):
            cleaned_context = matched_facts.replace("\\n", "\n")
        else:
            fact_contents = [doc.page_content for doc in matched_facts if hasattr(doc, 'page_content')]
            cleaned_context = "\n\n".join(fact_contents).replace("\\n", "\n")

    # 4. Construct a completely scannable Prompt Template with Guardrails
    guardrail_prompt = f"""
    You are a helpful customer support assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say "I'm sorry, I don't have that information in my knowledge base yet. Let me check with our team." Do not make things up.

    Context:
    {cleaned_context if cleaned_context else "No baseline data matches found for this query."}

    Question: {user_query}
    Answer:
    """

    # 5. Execute API payload delivery using a stable production identifier
    with st.spinner("Generating production system output..."):
        try:
            content_payload = []
            
            # Multi-Modal Check: If an image is uploaded, append the PIL object to the payload
            if uploaded_image:
                pil_image = Image.open(uploaded_image)
                content_payload.append(pil_image)
                
            # Append the constructed prompt block
            content_payload.append(guardrail_prompt)
            
            # Query the production Gemini model engine directly
            model = genai.GenerativeModel("gemini-2.5-flash")
            prediction = model.generate_content(content_payload)
            
            # 6. Render clean, scannable Markdown output to the user
            with st.chat_message("assistant"):
                st.markdown("#### **System Output:**")
                st.write(prediction.text)
                
            # Track response in history state array
            st.session_state.chat_history.append({"role": "assistant", "content": prediction.text})
            
        except Exception as error_exception:
            with st.chat_message("assistant"):
                st.error(f"Prediction Pipeline Error: {str(error_exception)}")

    # 7. Render Collapsible Evaluation Diagnostics Panel
    with st.expander("🔍 System Diagnostic — Active Context References", expanded=False):
        if matched_facts:
            st.text(matched_facts)
        else:
            st.text("No active semantic overlapping data matches found.")