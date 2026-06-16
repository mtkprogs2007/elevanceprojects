import streamlit as st
import google.generativeai as genai
import pandas as pd
import spacy
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini Engine 
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ System Configuration Error: Unable to establish secure connection to medical knowledge base network.")

# Load SpaCy NLP Core Pipelines (Kept identical for matching logic)
@st.cache_resource
def load_clinical_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_clinical_nlp()

# Structured MedQuAD Dataset (Kept identical)
@st.cache_data
def load_extended_medquad():
    data = {
        "question": [
            "What are the symptoms of Diabetes?",
            "How is Hypertension treated?",
            "What causes Flu?",
            "What are treatments for Migraine headaches?",
            "How can you prevent Type 2 Diabetes?",
            "What diagnostic tests are used for Hypertension?"
        ],
        "answer": [
            "Common symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), unexplained weight loss, increased hunger, and chronic fatigue.",
            "Hypertension management relies on a multi-stage approach: lifestyle modifications (low-sodium DASH diet, regular aerobic exercise) and medical management using ACE inhibitors, ARBs, or beta-blockers.",
            "The flu is caused by influenza viruses that infect the respiratory tract. It spreads mainly by droplets made when people with flu cough, sneeze or talk.",
            "Migraine therapeutic pathways include acute abortive treatments (triptans, NSAIDs like ibuprofen) and preventive medications (beta-blockers, topiramate) along with trigger avoidance.",
            "Type 2 Diabetes prevention involves maintaining a healthy weight through a balanced, low-glycemic diet, engaging in at least 150 minutes of moderate exercise per week, and monitoring blood sugar levels.",
            "Hypertension is routinely diagnosed via recurrent sphygmomanometer blood pressure readings. Secondary diagnostic tracking includes ambulatory blood pressure monitoring, ECG tracking, and basic metabolic panels."
        ],
        "focus": ["Diabetes", "Hypertension", "Flu", "Migraine", "Diabetes", "Hypertension"],
        "qtype": ["Symptoms", "Treatment", "Causes", "Treatment", "Prevention", "Diagnosis"]
    }
    return pd.DataFrame(data)

df_med = load_extended_medquad()

# --- CLINICAL ANALYTICAL LOGIC CORE (Kept identical) ---

def extract_clinical_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "CHEMICAL", "ORG", "GPE"]]
    if not entities:
        entities = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
    return list(set(entities))

def evaluate_retrieval_matrix(user_query, dataset):
    tokens = extract_clinical_entities(user_query)
    user_query_lower = user_query.lower()
    
    scored_rows = []
    for idx, row in dataset.iterrows():
        score = 0
        for t in tokens:
            if t.lower() in row['question'].lower() or t.lower() in row['focus'].lower():
                score += 3
        
        if "symptom" in user_query_lower and row['qtype'] == "Symptoms": score += 2
        if "treat" in user_query_lower or "cure" in user_query_lower and row['qtype'] == "Treatment": score += 2
        if "cause" in user_query_lower or "why" in user_query_lower and row['qtype'] == "Causes": score += 2
        if "prevent" in user_query_lower and row['qtype'] == "Prevention": score += 2
        if "test" in user_query_lower or "diagnose" in user_query_lower and row['qtype'] == "Diagnosis": score += 2
        
        if score > 0:
            scored_rows.append((score, row))
            
    scored_rows.sort(key=lambda x: x[0], reverse=True)
    return scored_rows[0][1] if scored_rows else None

# --- UI / UX MEDICAL HOSPITAL INTERFACE ---
st.set_page_config(page_title="CarePulse Omnia - Patient Desk", page_icon="🏥", layout="wide")

# Hospital Professional Custom Styling
st.markdown("""
    <style>
    .hospital-card {
        background-color: #111827; 
        padding: 20px; 
        border-radius: 12px; 
        border-top: 4px solid #00b4d8; 
        margin-bottom: 15px;
    }
    .clinic-badge {
        background-color: #03045e;
        color: #00b4d8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Hospital Header
st.title("🏥 CarePulse Omnia")
st.markdown("##### *St. Jude Multispecialty Hospital Group — Interactive Patient Care & Information Desk*")
st.markdown("---")

# Layout Split
col_main, col_sidebar_info = st.columns([2, 1])

with col_main:
    st.markdown("### 🔍 Medical Inquiry Desk")
    st.caption("Type your questions regarding common medical conditions, preventive care guidelines, or diagnostic tracking options below.")
    
    user_query = st.text_input(
        "Search Patient Education Database:", 
        placeholder="e.g., What diagnostic tests are used for Hypertension?",
        label_visibility="collapsed"
    )
    
    if user_query:
        with st.spinner("Accessing verified clinical education protocols..."):
            matched_profile = evaluate_retrieval_matrix(user_query, df_med)
            
            if matched_profile is not None:
                context_str = f"Focus Area: {matched_profile['focus']}\nCategory: {matched_profile['qtype']}\nOfficial MedQuAD Documentation: {matched_profile['answer']}"
            else:
                context_str = "Rely on verified medical knowledge guidelines to formulate a response."
            
            prompt = f"""
            You are an automated patient information assistant at St. Jude Multispecialty Hospital. 
            Answer the user's question accurately using a warm, clear, and easy-to-understand tone for regular patients.
            Avoid technical AI or computer engineering jargon. Use the provided trusted medical context where helpful:
            
            {context_str}
            
            Question: {user_query}
            
            Format your response cleanly using normal headings, lists, and a polite bedside manner.
            At the bottom, add a prominent box or note reminding them that this information does not substitute for consulting an actual physician at St. Jude Hospital.
            """
            
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 👩‍⚕️ Hospital Clinical Guidance Board")
            st.markdown(response.text,unsafe_allow_html=True)

with col_sidebar_info:
    st.markdown("### 📊 Live Diagnostic Insights")
    
    if user_query:
        extracted_terms = extract_clinical_entities(user_query)
        matched_profile = evaluate_retrieval_matrix(user_query, df_med)
        
        # Clinical focus card
        st.markdown("<div class='hospital-card'>", unsafe_allow_html=True)
        st.markdown("##### 🎯 Identified Medical Targets")
        if extracted_terms:
            for term in extracted_terms:
                st.markdown(f"• **Condition/Topic:** `{term}`")
        else:
            st.caption("No specific condition keyword isolated from request.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Hospital Routing Department classification card
        st.markdown("<div class='hospital-card'>", unsafe_allow_html=True)
        st.markdown("##### 🏢 Relevant Clinical Specialty")
        if matched_profile is not None:
            # Map focus categories to real medical departments
            dept_map = {
                "Diabetes": "Endocrinology & Diabetology",
                "Hypertension": "Cardiology & Vascular Health",
                "Flu": "General Medicine & Infectious Diseases",
                "Migraine": "Neurology & Pain Management"
            }
            dept = dept_map.get(matched_profile['focus'], "General Outpatient Clinic")
            
            st.markdown(f"Primary Focus: **{matched_profile['focus']}**")
            st.markdown(f"Inquiry Type: <span class='clinic-badge'>{matched_profile['qtype']}</span>", unsafe_allow_html=True)
            st.markdown(f"Recommended Department: **{dept}**")
        else:
            st.warning("Query requires general routing. Please speak with the front desk triage nurse.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.info("Input a patient inquiry to instantly route to relevant clinical categories and specialties.")

# Sidebar Navigation Elements
with st.sidebar:
    st.markdown("## 🏢🏥") 
    st.header("St. Jude Hospital Systems")
    st.markdown("**System Portal Version:** CarePulse 4.2")
    st.markdown("**Information Source:** Institutional Clinical Protocols")
    st.markdown("---")
    st.subheader("🏥 Active Hospital Indexes")
    st.caption("This system contains pre-approved education data for the following categories:")
    st.dataframe(
        df_med[['focus', 'qtype']].rename(columns={'focus': 'Medical Condition', 'qtype': 'Guideline Category'}), 
        use_container_width=True, 
        hide_index=True
    )