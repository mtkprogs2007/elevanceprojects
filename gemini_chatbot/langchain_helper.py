import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
load_dotenv()

# Global Configuration Parameters for Reproducibility
VECTOR_DB_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PRIMARY_LLM_MODEL = "gemini-1.5-flash"

# Initialize a modern, stable HuggingFace embedding model
instructor_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize modern Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model=PRIMARY_LLM_MODEL, 
    google_api_key=os.environ.get("GOOGLE_API_KEY"), 
    temperature=0.1
)

def create_vector_db():
    """
    Data Preprocessing & Vectorization Pipeline.
    Loads raw CSV data chunks, extracts features via local transformer embeddings,
    and builds a local FAISS vector index.
    """
    loader = CSVLoader(file_path="dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(VECTOR_DB_PATH)

def initialize_base_db():
    """Wrapper function interface for main.py execution block."""
    create_vector_db()

def query_relevant_context(user_query, k=3):
    # Load your local FAISS database
    vectordb = FAISS.load_local(
        VECTOR_DB_PATH, 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Set up a retriever to fetch exactly 'k' documents
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # Modern LangChain replacement for v0.3+ compatibility
    return retriever.invoke(user_query)
def get_qa_chain(experiment_mode="advanced"):
    """
    Model Experimentation Framework.
    Allows toggling between standard search and a strict high-threshold retrieval model
    to meet optimization design specifications.
    """
    vectordb = FAISS.load_local(
        VECTOR_DB_PATH, 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Model Comparison Setup: Adjust parameters based on the experimental mode
    if experiment_mode == "baseline":
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    else:
        # Advanced experimental configuration with a strict structural relevance threshold
        retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.65, "k": 3})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Modern LCEL execution graph linkage
    base_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    # Encapsulate output inside the expected data registry schema for main.py
    final_chain = {
        "query": RunnablePassthrough()
    } | RunnablePassthrough().assign(result=base_chain)
    
    return final_chain


def add_csv_file_to_db(file_obj, file_name):
    """
    Advanced Feature: Bulk Indexing.
    Processes an uploaded CSV file, forces strict Document formatting compatibility,
    and merges it into the existing local FAISS vector store.
    """
    # 1. Load the uploaded CSV with the correct encoding fallback
    df = pd.read_csv(file_obj, encoding="latin1")
    
    # 2. Explicitly map rows to proper LangChain Document objects matching main.py's format
    new_docs = []
    for _, row in df.iterrows():
        # Ensure 'prompt' and 'response' text exist safely
        prompt_text = str(row.get('prompt', ''))
        response_text = str(row.get('response', ''))
        
        # Format the text exactly how CSVLoader does it so your QA chain works perfectly too
        combined_content = f"prompt: {prompt_text}\nresponse: {response_text}"
        
        doc = Document(
            page_content=combined_content,
            metadata={"source": file_name, "row": _}
        )
        new_docs.append(doc)
    
    # 3. Load your existing vector database
    vectordb = FAISS.load_local(
        VECTOR_DB_PATH, 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 4. Generate embeddings for the clean new Document objects
    new_vectordb = FAISS.from_documents(new_docs, instructor_embeddings)
    
    # 5. Atomic merge and update local storage files
    vectordb.merge_from(new_vectordb)
    vectordb.save_local(VECTOR_DB_PATH)