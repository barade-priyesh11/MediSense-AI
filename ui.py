import streamlit as st
import json
import httpx
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# OpenRouter API key - using the correct format from .env file
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Custom theme and styling
def set_custom_theme():
    # Define custom colors - changed to blue color scheme
    primary_color = "#1565C0"  # Deep blue
    secondary_color = "#42A5F5"  # Light blue
    background_color = "#F5F9FF"  # Very light blue
    text_color = "#333333"  # Dark gray
    success_color = "#4CAF50"  # Green
    warning_color = "#FF9800"  # Orange
    error_color = "#F44336"  # Red
    
    # Apply custom CSS
    st.markdown(f"""
    <style>
        .main .block-container {{background-color: {background_color}; padding: 1rem; border-radius: 10px;}}
        h1, h2, h3 {{color: {primary_color} !important; font-family: 'Helvetica Neue', sans-serif;}}
        .stButton>button {{background-color: {primary_color}; color: white; border-radius: 20px; padding: 0.5rem 2rem; font-weight: bold; border: none; transition: all 0.3s;}}
        .stButton>button:hover {{background-color: {secondary_color}; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1);}}
        .stTextInput>div>div>input {{border-radius: 20px; border: 2px solid {secondary_color}; padding: 0.75rem;}}
        .stTextInput>div>div>input:focus {{border: 2px solid {primary_color}; box-shadow: 0 0 5px rgba(21, 101, 192, 0.3);}}
        div[data-testid="stSidebar"] {{background-color: {primary_color}; padding: 1rem; color: white;}}
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {{color: white !important;}}
        div[data-testid="stSidebar"] .stMarkdown {{color: white;}}
        # .diWe are seeking an experienced Machine Learning / AI Engineer with a strong background in healthcare/medical AI to build a next-generation symptom checker and lab recommendation engine. The system will guide users through an adaptive Q&A, identify potential underlying systems/conditions, and recommend evidence-based lab panels with triage safety checks.agnosis-card {{background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 0.5rem 0;}}
        .disclaimer {{background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid {warning_color}; padding: 0.75rem; border-radius: 4px;}}
        /* Remove extra spacing */
        .element-container {{margin-bottom: 0.5rem !important;}}
        .stMarkdown {{margin-bottom: 0 !important;}}
        p {{margin-bottom: 0.5rem !important;}}
    </style>
    """, unsafe_allow_html=True)

# Function to get a base64 encoded image for the logo
def get_medical_logo():
    # Create a simple medical logo using PIL
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a medical cross - updated to blue color
    draw.rectangle([80, 40, 120, 160], fill=(21, 101, 192, 255))  # Vertical rectangle
    draw.rectangle([40, 80, 160, 120], fill=(21, 101, 192, 255))  # Horizontal rectangle
    
    # Draw a circle around the cross
    draw.ellipse([30, 30, 170, 170], outline=(21, 101, 192, 255), width=8)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Page configuration
st.set_page_config(
    page_title="AI Symptom Checker", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
set_custom_theme()

# Display logo and header
logo = get_medical_logo()
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <img src="data:image/png;base64,{logo}" style="height: 60px; margin-right: 0.75rem;">
    <div>
        <h1 style="margin: 0; font-size: 2rem;">AI Symptom Checker</h1>
        <p style="margin: 0; font-size: 1rem; color: #666;">AI assistance for your health</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""<p style="font-size: 1rem; margin-bottom: 1rem;">Describe your symptoms and get instant AI-based suggestions.</p>""", unsafe_allow_html=True)

# Function to check symptoms using OpenRouter API
async def check_symptom(symptom):
    try:
        payload = {
            "model": "deepseek/deepseek-chat-v3.1:free",
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"""
Patient has the following symptom: {symptom}.
Please respond ONLY in the following JSON format:
{{
  "diagnosis": "<possible diagnosis>",
  "confidence": "<high/medium/low>",
  "advice": "<recommended next steps or precautions>",
  "follow_up_questions": ["<question1>", "<question2>", "<question3>"]
}}
The follow_up_questions should be 3 important questions to ask the patient to better understand their condition.
Do not include any additional text.
"""}
            ],
            "temperature": 0
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourwebsite.com",
            "X-Title": "SymptomCheckerApp"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # Increased timeout
            )

        if response.status_code != 200:
            return {"error": {"message": response.text, "code": response.status_code}}

        diagnosis_text = response.json()["choices"][0]["message"]["content"]
        
        # Parse the JSON response
        try:
            diagnosis_data = json.loads(diagnosis_text)
            return diagnosis_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return {"error": {"message": "Failed to parse AI response", "raw_response": diagnosis_text}}

    except Exception as e:
        return {"error": {"message": f"Something went wrong: {str(e)}", "code": 500}}

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    # Input section with styled container
    st.subheader("Enter Your Symptoms")
    
    # Input field with placeholder
    symptom = st.text_input("", placeholder="Example: fever, headache, cough...")
    
    # Styled button
    check_button = st.button("Check Symptoms", use_container_width=True)
    st.markdown("""</div>""", unsafe_allow_html=True)

with col2:
    # Health tips in the sidebar
    st.subheader("Health Tips")
    st.markdown("""
    * Get 8 hours of sleep daily
    * Drink at least 2 liters of water per day
    * Exercise regularly
    * Maintain a balanced diet
    * Avoid stress
    """)
    st.markdown("""</div>""", unsafe_allow_html=True)

if check_button:
    if not symptom:
        st.warning("Please enter a symptom first.")
    else:
        with st.spinner("Fetching diagnosis from AI..."):
            # Use asyncio to run the async function
            import asyncio
            result = asyncio.run(check_symptom(symptom))
            
            if "error" in result:
                st.error(f"Error: {result['error'].get('message', 'Unknown error')}")
                if "raw_response" in result.get("error", {}):
                    st.text(f"Raw response: {result['error']['raw_response']}")
            else:
                st.success("Diagnosis received!")
                
                # Display diagnosis in a card format
                st.markdown("""<div class="diagnosis-card">""", unsafe_allow_html=True)
                st.subheader("Diagnosis Details")
                
                # Create three columns for diagnosis details
                d_col1, d_col2, d_col3 = st.columns(3)
                
                with d_col1:
                    st.markdown(f"""<h3 style="font-size: 1.2rem; color: #1565C0;">Possible Diagnosis</h3>
                    <p style="font-size: 1.1rem; font-weight: bold;">{result.get('diagnosis', 'N/A')}</p>""", unsafe_allow_html=True)
                
                with d_col2:
                    confidence = result.get('confidence', 'N/A')
                    confidence_color = "#4CAF50" if confidence == "high" else "#FF9800" if confidence == "medium" else "#F44336"
                    st.markdown(f"""<h3 style="font-size: 1.2rem; color: #1565C0;">Confidence Level</h3>
                    <p style="font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{confidence}</p>""", unsafe_allow_html=True)
                
                with d_col3:
                    st.markdown(f"""<h3 style="font-size: 1.2rem; color: #1565C0;">Advice</h3>
                    <p style="font-size: 1.1rem;">{result.get('advice', 'N/A')}</p>""", unsafe_allow_html=True)
                
                # Add follow-up questions section
                if "follow_up_questions" in result and result["follow_up_questions"]:
                    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                    st.markdown("""<h3 style="font-size: 1.2rem; color: #1565C0;">Follow-up Questions</h3>""", unsafe_allow_html=True)
                    
                    for i, question in enumerate(result["follow_up_questions"]):
                        st.markdown(f"""<p style="font-size: 1rem;">🔹 {question}</p>""", unsafe_allow_html=True)
                
                st.markdown("""</div>""", unsafe_allow_html=True)
                
                # Add a disclaimer with better styling
                st.markdown("""<div class="disclaimer">
                <p style="font-weight: bold; margin-bottom: 0.5rem;">⚠️ Disclaimer:</p>
                <p>This is an AI-generated suggestion and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
                </div>""", unsafe_allow_html=True)

# Add information about the app in the sidebar
st.sidebar.markdown(f"""<img src="data:image/png;base64,{logo}" style="height: 60px; margin-bottom: 1rem;">""", unsafe_allow_html=True)
st.sidebar.title("About the App")
st.sidebar.markdown("""
<div style="color: white;">
This app uses AI to provide preliminary symptom analysis.
It is for educational purposes only and not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

# Add contact information
st.sidebar.title("Contact Us")
st.sidebar.markdown("""
<div style="color: white;">
Email: support@healthcareai.com<br>
Phone: +91 1234567890
</div>
""", unsafe_allow_html=True)