import streamlit as st
import json
import httpx
import os
from dotenv import load_dotenv
import base64
from PIL import Image, ImageDraw
import io
import re
import asyncio
import faiss

# NEW IMPORTS FOR ROBUST FILE DELETION
import gc
import time 

# Import new libraries for vector DB and embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# NEW: Import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import functions from the knowledge graph file
from knowledge_graph import get_info_from_symptoms

# Load environment variables
load_dotenv()

# Fetch API keys from .env file
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- Persistent Storage Configuration ---
DATA_DIR = "./data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
os.makedirs(DATA_DIR, exist_ok=True)


# --- New Functions for Vector DB ---

# Function to handle file and extract text
def handle_file_upload(uploaded_file):
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif uploaded_file.name.endswith(".txt"):
        # The TextLoader can directly read from the saved file
        loader = TextLoader(file_path)
        documents = loader.load()
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    else:
        st.warning("Only PDF, DOCX, and TXT files are supported.")
        os.remove(file_path)  # Clean up the unsupported file
        return None

    # If documents are in string format, convert them to a LangChain-compatible list
    if isinstance(documents, str):
        from langchain.docstore.document import Document
        documents = [Document(page_content=documents)]

    return documents

# Function to split text into chunks and create a vector store
def get_faiss_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Use HuggingFace embeddings which are free and run locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create or update a Faiss vector store
    if os.path.exists(FAISS_INDEX_PATH):
        # If an index exists, load it and add the new documents
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(docs)
    else:
        # Otherwise, create a new one
        vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save the updated vector store to disk
    vector_store.save_local(FAISS_INDEX_PATH)
    
    return vector_store


# --- Triage Safety System with Deterministic Overrides ---
def triage_safety_check(prompt):
    """
    Checks for critical symptoms and returns an emergency override response.
    Returns a dictionary with 'triage_needed' as True if a red-flag is found.
    """
    emergency_symptoms = {
        "chest pain": "ER",
        "severe shortness of breath": "ER",
        "loss of consciousness": "ER",
        "sudden vision loss": "ER",
        "severe bleeding": "ER"
    }
    
    for symptom, action in emergency_symptoms.items():
        if symptom in prompt.lower():
            return {
                "triage_needed": True,
                "action": action,
                "message": f"Based on your mention of **{symptom}**, this could be a medical emergency. Please go to the **nearest Emergency Room** immediately or call your local emergency services."
            }
    return {"triage_needed": False}


# --- The rest of the code is the same as before, with a few changes ---

def set_custom_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700&display=swap');
        :root {
            --primary-color: #4A90E2;
            --accent-color: #50C9C3;
            --background-color: #F0F4F8;
            --text-color: #2C3E50;
            --chat-user-bg: var(--accent-color);
            --chat-user-text: white;
            --chat-assistant-bg: #E3E9F0;
            --chat-assistant-text: var(--text-color);
            --card-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --primary-color: #72B1FF;
                --accent-color: #8AD2CD;
                --background-color: #1E2733;
                --text-color: #EAECEF;
                --chat-user-bg: #405973;
                --chat-user-text: #EAECEF;
                --chat-assistant-bg: #2B3A4C;
                --chat-assistant-text: #EAECEF;
                --card-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
        }
        body {
            font-family: 'Outfit', sans-serif;
            color: var(--text-color);
        }
        .main .block-container {
            background-color: var(--background-color); 
            padding: 3rem; 
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        }
        h1, h2, h3, h4, h5, h6 {color: var(--primary-color) !important; font-family: 'Outfit', sans-serif;}
        .stButton>button {
            background-color: var(--primary-color); 
            color: white; 
            border-radius: 25px; 
            padding: 0.75rem 2.5rem; 
            font-weight: bold; 
            border: none; 
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: var(--accent-color); 
            transform: translateY(-3px); 
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        div[data-testid="stSidebar"] {
            background-color: var(--primary-color); 
            padding: 2rem; 
            color: white;
        }
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
            color: white !important;
        }
        .chat-message-container {
            border-radius: 20px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            font-size: 1rem;
            max-width: 85%;
        }
        .chat-user-message {
            background-color: var(--chat-user-bg);
            color: var(--chat-user-text);
            margin-left: auto;
            border-bottom-right-radius: 5px;
            border-top-right-radius: 20px;
            border-bottom-left-radius: 20px;
            border-top-left-radius: 20px;
            text-align: right;
        }
        .chat-assistant-message {
            background-color: var(--chat-assistant-bg);
            color: var(--chat-assistant-text);
            margin-right: auto;
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
            border-top-left-radius: 20px;
            border-bottom-left-radius: 5px;
            text-align: left;
        }
        .diagnosis-card {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: var(--card-shadow);
            border-top: 8px solid var(--primary-color);
            position: relative;
        }
        .disclaimer {
            background-color: rgba(74, 144, 226, 0.1); 
            border-left: 4px solid var(--primary-color); 
            padding: 1rem; 
            border-radius: 8px;
            margin-top: 2rem;
        }
        .dot-spinner {
            --uib-size: 1.5rem;
            --uib-speed: .9s;
            --uib-color: var(--accent-color);
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: var(--uib-size);
            height: calc(var(--uib-size) * .17);
        }
        .dot-spinner__dot {
            position: relative;
            width: calc(var(--uib-size) * .14);
            height: calc(var(--uib-size) * .14);
            border-radius: 50%;
            background-color: var(--uib-color);
            transform: scale(0);
            animation: pulse var(--uib-speed) ease-in-out infinite;
        }
        .dot-spinner__dot:nth-child(1) {
            animation-delay: calc(var(--uib-speed) * -.3);
        }
        .dot-spinner__dot:nth-child(2) {
            animation-delay: calc(var(--uib-speed) * -.15);
        }
        .dot-spinner__dot:nth-child(3) {
            animation-delay: 0s;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(0); opacity: 0; }
            50% { transform: scale(1); opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

# Function to get a base64 encoded image for the logo
def get_medical_logo():
    logo_color = "#4A90E2"
    
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 40, 120, 160], fill=logo_color)
    draw.rectangle([40, 80, 160, 120], fill=logo_color)
    draw.ellipse([30, 30, 170, 170], outline=logo_color, width=8)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Page configuration
st.set_page_config(
    page_title="Medisense AI Assistant", 
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom theme
set_custom_theme()

# Display logo and header
logo = get_medical_logo()
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <img src="data:image/png;base64,{logo}" style="height: 70px; margin-right: 1rem;">
    <div>
        <h1 style="margin: 0; font-size: 2.5rem;">Medisense AI Assistant</h1> <p style="margin: 0; font-size: 1.1rem; color: #666;">Your personal AI for health insights & web search</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat history and state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add the initial greeting only once
    st.session_state.messages.append({"role": "assistant", "content": "Hello! 👋 I'm your **Medisense AI Assistant**. I can help analyze your symptoms or search for health-related information on the web. Please describe your symptoms in detail to get started."}) 

if "symptom_count" not in st.session_state:
    st.session_state.symptom_count = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    # Check if a FAISS index exists on disk and load it
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            st.session_state.vector_store = None


# --- AI Interaction Logic ---
async def call_api(conversation_history, mode, selected_api, rag_context=""):
    try:
        if selected_api == "OpenRouter":
            if not OPENROUTER_API_KEY:
                return {"error": {"message": "OpenRouter API Key not found in .env file.", "code": 401}}
            
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://yourwebsite.com",
                "X-Title": "SymptomCheckerApp"
            }
            payload = {
                "model": "deepseek/deepseek-chat-v3.1:free", 
                "messages": conversation_history,
                "temperature": 0
            }
            
            if mode == "symptoms":
                # Updated prompt to be more strict about JSON format
                system_prompt_content = """You are a helpful medical assistant. Gather information about the patient's symptoms by asking follow-up questions one by one. Do not provide a diagnosis until you have asked at least 3-4 relevant follow-up questions about their symptoms, severity, duration, and other relevant factors.

Your entire response must be a single, valid JSON object and nothing else. Do not include any text, explanations, or code fences like ```json.

For follow-up questions (when you need more information):
{
  "is_final": false,
  "follow_up_question": "<a single, clear question about symptoms, duration, severity, location, etc.>",
  "diagnosis_possible": "<brief summary of what you're considering>",
  "is_greeting": false
}

If the user greets you (e.g., "Hi", "Hello"), respond with a friendly greeting:
{
  "is_final": false,
  "is_greeting": true,
  "follow_up_question": "<a friendly greeting and ask them to describe their symptoms>",
  "diagnosis_possible": ""
}

Only provide a diagnosis when you have gathered sufficient information about their symptoms."""
            
            elif mode == "rag":
                 # RAG mode does not require JSON
                 system_prompt_content = f"""You are a helpful AI assistant. You have been provided with some context about the user's uploaded documents. Your task is to answer the user's question based ONLY on the provided context. Do not use any outside knowledge. If the answer cannot be found in the context, politely state that you cannot find the information in the provided documents.
                 Context: {rag_context}"""
                 # Context is now passed as an argument
                 
            else: # mode == "diagnosis"
                # Diagnosis mode requires JSON
                system_prompt_content = """You are a helpful medical assistant. Based on the full conversation history, provide a comprehensive final diagnosis. Your entire response must be a single, valid JSON object and nothing else. Do not include any text, explanations, or code fences like ```json.

{
  "is_final": true,
  "is_greeting": false,
  "diagnosis": "<most likely diagnosis based on symptoms discussed>",
  "confidence": "<high/medium/low based on symptom clarity>",
  "advice": "<detailed recommended next steps, precautions, when to see a doctor, etc.>"
}"""
            
            payload["messages"].insert(0, {"role": "system", "content": system_prompt_content})

        elif selected_api == "Perplexity":
            if not PERPLEXITY_API_KEY:
                return {"error": {"message": "Perplexity API Key not found in .env file.", "code": 401}}

            api_url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            system_prompt_content = """You are a helpful web search assistant. Your task is to provide a concise and accurate summary of information based on your web search. Answer the user's question directly and cite your sources when possible."""
            
            payload = {
                "model": "pplx-70b-online", 
                "messages": [{"role": "system", "content": system_prompt_content}] + conversation_history,
                "temperature": 0
            }

        else:
            return {"error": {"message": "Invalid API selection.", "code": 400}}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=45
            )

        if response.status_code != 200:
            return {"error": {"message": f"API Error: {response.text}", "code": response.status_code}}

        response_text = response.json()["choices"][0]["message"]["content"]
        
        if selected_api == "OpenRouter":
            # --- FIX: RAG mode should be treated as plain text ---
            if mode == "rag":
                # Returns a dict with 'content' for the chat loop to display
                return {"success": True, "content": response_text}
            
            # --- Only 'symptoms' and 'diagnosis' modes require JSON parsing ---
            try:
                # First, try to parse the raw response.
                response_data = json.loads(response_text)
                return response_data
            except json.JSONDecodeError:
                # New and improved regex to handle various LLM outputs
                # It finds the first opening brace '{' and the last closing brace '}'
                start_index = response_text.find('{')
                end_index = response_text.rfind('}')
                
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_string = response_text[start_index:end_index+1]
                    try:
                        response_data = json.loads(json_string)
                        return response_data
                    except json.JSONDecodeError:
                        return {"error": {"message": "Failed to parse the extracted JSON object.", "raw_response": response_text}}
                else:
                    # This is the original JSON error fallback. 
                    return {"error": {"message": "Failed to locate a valid JSON object in the response.", "raw_response": response_text}}
        else:
            # Perplexity (Web Search) is always plain text
            return {"success": True, "content": response_text}

    except Exception as e:
        return {"error": {"message": f"Something went wrong: {str(e)}", "code": 500}}

def is_web_search_query(prompt):
    search_keywords = ["what is", "latest", "current status", "recent data", "web search", "search for", "find information about", "look up", "tell me about", "information on", "news about", "updates on", "recent news"]
    if any(keyword in prompt.lower() for keyword in search_keywords):
        return True
    return False

def is_user_greeting(prompt):
    greetings = ["hi", "hello", "namaste", "hey", "hii", "helo", "hiii"]
    return prompt.strip().lower() in greetings

def should_provide_diagnosis():
    """Determine if we should provide a final diagnosis based on conversation history"""
    symptom_responses = 0
    for message in st.session_state.messages:
        if message["role"] == "user" and not is_user_greeting(message["content"]) and not is_web_search_query(message["content"]):
            symptom_responses += 1
    
    # Provide diagnosis after 3+ meaningful symptom-related exchanges
    return symptom_responses >= 3

def display_diagnosis_card(result):
    """Display a well-formatted diagnosis card with proper data validation"""
    # Validate that we have actual diagnosis data
    if not result.get('diagnosis') or result.get('diagnosis') in ['N/A', '', None]:
        st.error("Diagnosis information is not available. Please provide more details about your symptoms.")
        return
    
    confidence = result.get('confidence', 'medium').lower()
    confidence_color = "#27AE60" if confidence == "high" else "#F39C12" if confidence == "medium" else "#E74C3C"
    
    st.markdown(f"""
    <style>
        .diagnosis-card-colored {{
            border-top: 8px solid {confidence_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""<div class="diagnosis-card diagnosis-card-colored">""", unsafe_allow_html=True)
    st.subheader("🩺 Diagnosis Results")
    
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown(f"""<h5 style="font-size: 1rem; color: var(--primary-color);">Possible Diagnosis</h5>
        <p style="font-size: 1.1rem; font-weight: bold;">{result.get('diagnosis', 'Unable to determine')}</p>""", unsafe_allow_html=True)
    
    with d_col2:
        st.markdown(f"""<h5 style="font-size: 1rem; color: var(--primary-color);">Confidence Level</h5>
        <p style="font-size: 1.1rem; font-weight: bold; color: {confidence.capitalize()};">{confidence.capitalize()}</p>""", unsafe_allow_html=True)
    
    # New: Add lab recommendations from the knowledge graph
    user_symptoms = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    conditions, lab_codes = get_info_from_symptoms(" ".join(user_symptoms).split())
    if lab_codes:
        st.markdown(f"""<h5 style="font-size: 1rem; color: var(--primary-color);">Lab Recommendations</h5>
        <p style="font-size: 1.1rem; font-weight: bold;">{", ".join(lab_codes)}</p>""", unsafe_allow_html=True)
    
    st.markdown(f"""<h5 style="font-size: 1rem; color: var(--primary-color);">Recommendations</h5>
    <p style="font-size: 1.1rem;">{result.get('advice', 'Please consult with a healthcare professional for proper medical advice.')}</p>""", unsafe_allow_html=True)
    
    st.markdown("""</div>""", unsafe_allow_html=True)

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            if message["content"].get("is_final") and message["content"].get("diagnosis") and message["content"].get("diagnosis") != "N/A":
                display_diagnosis_card(message["content"])
            elif message["content"].get("follow_up_question") and not message["content"].get("is_final"):
                st.markdown(message["content"].get("follow_up_question"))
            else:
                if message["content"].get("follow_up_question"):
                    st.markdown(message["content"].get("follow_up_question"))
        else:
            st.markdown(message["content"])

# Callback function to reset the conversation (FILE LOCKING FIX APPLIED HERE)
def reset_conversation():
    st.info("Attempting to reset conversation and clear Knowledge Base...")
    
    # 1. Clear conversation state
    st.session_state.messages = []
    st.session_state.symptom_count = 0
    
    # 2. CRITICAL FIX: Explicitly clear the vector store reference
    # Setting the vector_store to None *before* deleting the files helps release the file lock.
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.session_state.vector_store = None
    
    # --- ENHANCED CLEANUP ---
    # Trigger garbage collection and wait briefly to try and force file handle release
    import gc
    import time
    gc.collect() 
    time.sleep(0.5) 
    # ------------------------
    
    # 3. Delete the persistent FAISS files
    faiss_deleted = False
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            # Delete the main index file
            os.remove(FAISS_INDEX_PATH)
            # FAISS also creates a .pkl file for metadata; delete it if it exists
            pkl_path = FAISS_INDEX_PATH.replace(".faiss", ".pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            st.success("Knowledge Base files successfully deleted.")
            faiss_deleted = True
        except PermissionError:
            # The warning remains for the user to manually intervene
            st.error("⚠️ **Permission Denied:** Could not delete FAISS files. The files may be locked by another process (e.g., your IDE or a previous session). **Please stop Streamlit (Ctrl+C in terminal) and restart it to clear the lock.**")
            return # Stop the rest of the reset process if file deletion failed
        except Exception as e:
            st.error(f"An unexpected error occurred during FAISS file deletion: {e}")
            return
    
    # 4. Clear all other files from the data directory
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                st.warning(f"Could not delete file: {file_name}. It may be locked.")
            except Exception as e:
                st.warning(f"Error deleting file {file_name}: {e}")
                
    st.success("Conversation has been reset and all uploaded documents are cleared.")
    st.experimental_rerun() # Use rerun to clear the UI fully


# Callback function to handle file upload
def handle_upload_callback():
    # The file uploader widget returns None if nothing is selected
    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file:
        with st.spinner("Processing documents..."):
            documents = handle_file_upload(uploaded_file)
            if documents:
                st.session_state.vector_store = get_faiss_vector_store(documents)
                st.success("Documents successfully processed and uploaded to Vector DB!")
                st.info("Now you can ask questions related to the documents.")
                # The file uploader widget state is handled automatically.
                # No need to manually set `st.session_state.uploaded_file = None`

# Chat input and processing
if prompt := st.chat_input("Describe your symptoms or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            
            # --- New: Step 1 - Perform the deterministic safety check first ---
            triage_result = triage_safety_check(prompt)
            if triage_result["triage_needed"]:
                st.markdown(triage_result["message"])
                st.session_state.messages.append({"role": "assistant", "content": triage_result["message"]})
            else:
                # --- Step 2 - If no red flags, proceed with existing LLM logic ---
                if is_user_greeting(prompt):
                    st.markdown("Hello there! How can I help you today? Please tell me about your symptoms or ask a health-related question.")
                    st.session_state.messages.append({"role": "assistant", "content": "Hello there! How can I help you today? Please tell me about your symptoms or ask a health-related question."})
                else:
                    chat_history_for_api = []
                    for m in st.session_state.messages[:-1]:
                        content_to_send = m["content"]
                        if isinstance(content_to_send, dict):
                            if content_to_send.get("follow_up_question"):
                                content_to_send = content_to_send["follow_up_question"]
                            # RAG response has a 'content' key. We must check for follow_up_question first.
                            elif m["role"] == "assistant" and content_to_send.get("content"):
                                content_to_send = content_to_send["content"]
                            else:
                                content_to_send = str(content_to_send)
                        chat_history_for_api.append({"role": m["role"], "content": content_to_send})
                    
                    chat_history_for_api.append({"role": "user", "content": prompt})
                    
                    result = None
                    if is_web_search_query(prompt):
                        selected_api = "Perplexity"
                        mode = "search"
                        chat_history_for_api = [{"role": "user", "content": prompt}]
                        result = asyncio.run(call_api(chat_history_for_api, mode, selected_api))
                    elif st.session_state.vector_store:
                        selected_api = "OpenRouter"
                        mode = "rag"
                        
                        retriever = st.session_state.vector_store.as_retriever()
                        docs = retriever.invoke(prompt)
                        
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Pass context to the call_api function
                        result = asyncio.run(call_api(chat_history_for_api, mode, selected_api, rag_context=context))
                    else:
                        selected_api = "OpenRouter"
                        if should_provide_diagnosis():
                            mode = "diagnosis"
                            result = asyncio.run(call_api(chat_history_for_api, mode, selected_api))
                        else:
                            mode = "symptoms"
                            result = asyncio.run(call_api(chat_history_for_api, mode, selected_api))
                    
                    if result: # Only process if an API call was made
                        if "error" in result:
                            st.error(f"Error: {result['error'].get('message', 'Unknown error')}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error'].get('message', 'Unknown error')}"})
                        else:
                            if selected_api == "OpenRouter":
                                # RAG/Document Search Result (non-JSON, returns 'content' key)
                                if result.get('content'):
                                    st.markdown(result.get("content", "I couldn't find information in your documents about that. Please try rephrasing your question."))
                                    st.session_state.messages.append({"role": "assistant", "content": result.get("content", "I couldn't find information in your documents about that.")})
                                # Diagnosis Result (JSON, returns 'is_final' key)
                                elif result.get("is_final") and result.get("diagnosis") and result.get("diagnosis") != "N/A":
                                    display_diagnosis_card(result)
                                    st.session_state.messages.append({"role": "assistant", "content": result})
                                    # BUTTON IS SHOWN HERE FOR FINAL DIAGNOSIS
                                    st.button("🔄 Start New Conversation", on_click=reset_conversation) 
                                # Symptom Follow-up Result (JSON, returns 'follow_up_question' key)
                                elif result.get("follow_up_question"):
                                    st.markdown(result["follow_up_question"])
                                    st.session_state.messages.append({"role": "assistant", "content": result})
                                # Fallback (generic symptom message if JSON is unexpected)
                                else:
                                    st.markdown("I received an unexpected response. I need more information to help you better. Can you describe your symptoms in more detail?")
                                    st.session_state.messages.append({"role": "assistant", "content": "I received an unexpected response. I need more information to help you better. Can you describe your symptoms in more detail?"})
                            else: # Perplexity (Web Search) - Correctly handles non-JSON
                                st.markdown(result.get("content", "I couldn't find information about that. Please try rephrasing your question."))
                                st.session_state.messages.append({"role": "assistant", "content": result.get("content", "I couldn't find information about that.")})
                                # BUTTON IS CORRECTLY ABSENT HERE


# --- Sidebar Content ---
with st.sidebar:
    st.title("App Information")
    st.markdown("""
    <div style="color: white; font-size: 0.9rem;">
    This **Medisense AI Assistant** is designed to help you with your health-related queries and provide web-based information. It uses advanced language models to analyze your symptoms and offer possible insights.
    </div>
    """, unsafe_allow_html=True) 
    
    st.markdown("---")

    # Here is the new document upload section
    st.title("Document Upload")
    st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        type=["pdf", "txt", "docx"],
        key="uploaded_file",
        on_change=handle_upload_callback
    )

    st.markdown("---")
    
    st.title("Health Tips")
    st.markdown("""
    <div style="color: white; font-size: 0.9rem;">
    💤 Get 7-9 hours of sleep daily<br>
    💧 Drink plenty of water throughout the day<br>
    🏃‍♂️ Incorporate regular physical activity<br>
    🥗 Eat a balanced diet rich in fruits and vegetables<br>
    🧘‍♀️ Practice mindfulness to reduce stress<br>
    🩺 Regular health check-ups are important<br>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.title("Emergency Contacts")
    st.markdown("""
    <div style="color: white; font-size: 0.9rem;">
    🚨 <strong>Emergency:</strong> 108 (India)<br>
    🏥 <strong>Medical Emergency:</strong> 102<br>
    ☎️ <strong>Ambulance:</strong> 108<br>
    <br>
    <strong>⚠️ If you're experiencing a medical emergency, please call emergency services immediately!</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.title("Useful Resources")
    st.markdown("""
    <div style="color: white; font-size: 0.9rem;">
    • <a href="https://www.who.int" target="_blank" style="color: #8AD2CD;">World Health Organization (WHO)</a><br>
    • <a href="https://www.cdc.gov" target="_blank" style="color: #8AD2CD;">Centers for Disease Control (CDC)</a><br>
    • <a href="https://www.nih.gov" target="_blank" style="color: #8AD2CD;">National Institutes of Health (NIH)</a><br>
    • <a href="https://www.mohfw.gov.in/" target="_blank" style="color: #8AD2CD;">Ministry of Health & Family Welfare (India)</a><br>
    </div>
    """, unsafe_allow_html=True)