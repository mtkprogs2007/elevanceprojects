import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# Ensure helper operations have immediate access to variables
load_dotenv()

# Configure the native SDK with your key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

class NativeGoogleEmbeddings:
    """
    Custom lightweight wrapper utilizing the direct, native Google SDK 
    integrated directly with the modern generation model framework.
    """
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 🌟 UNIFIED TO MODERN TEXT EMBEDDING SPECIFICATIONS
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

def parse_csv_to_records(file_buffer):
    """
    Parses incoming CSV data tables and structures them into simple 
    Python list containers, completely bypassing third-party vector frameworks.
    """
    file_buffer.seek(0)
    try:
        df = pd.read_csv(file_buffer, encoding='utf-8')
    except UnicodeDecodeError:
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer, encoding='latin1')
        
    if 'prompt' not in df.columns or 'response' not in df.columns:
        raise ValueError("Invalid matrix structure. Required columns: 'prompt' and 'response'.")
        
    records = []
    for _, row in df.iterrows():
        clean_prompt = str(row['prompt']).replace("\\n", "\n").strip()
        clean_response = str(row['response']).replace("\\n", "\n").strip()
        records.append({"q": clean_prompt, "a": clean_response})
        
    return records

def match_local_context(user_query, records, max_matches=3):
    """
    Scans internal memory maps for matching keywords to retrieve context dynamically.
    """
    if not records:
        return ""
        
    query_words = set(user_query.lower().split())
    scored_records = []
    
    for item in records:
        combined_text = (item["q"] + " " + item["a"]).lower()
        score = sum(1 for word in query_words if word in combined_text)
        if score > 0:
            scored_records.append((score, item))
            
    scored_records.sort(key=lambda x: x[0], reverse=True)
    
    matched_texts = [f"Q: {item['q']}\nA: {item['a']}" for _, item in scored_records[:max_matches]]
    return "\n\n".join(matched_texts)