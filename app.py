import os
import streamlit as st
import nltk
import ssl
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup as bs
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import cohere

# Handle SSL and NLTK data download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)

# Get API key from secrets or environment variables
try:
    COHERE_API_KEY = st.secrets["cohere"]["api_key"]
except (KeyError, FileNotFoundError):
    COHERE_API_KEY = os.environ.get('COHERE_API_KEY', '')

if not COHERE_API_KEY:
    st.error("❌ Cohere API key not found. Please configure it in your deployment platform's secrets.")
    st.stop()

# Rest of your existing code continues here...
CHUNK_SIZE = 1000
NUMBER_OF_MATCHES = 3
ps = PorterStemmer()

documents = []
original_docs = []
vectors = None
vectorizer = TfidfVectorizer()

# ... (rest of your existing functions and code)

def get_resp(query, context):
    client = cohere.Client(COHERE_API_KEY)
    
    prompt_context = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context)])
    
    prompt = f"""You are an AI assistant. Use the provided context to answer the user's query accurately and concisely.

Available Context:
{prompt_context}

User Query: {query}

Please provide a helpful answer based on the context above. If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."

Answer:"""
    
    try:
        available_models = ["command", "command-light", "command-r", "command-r-08-2024"]
        
        for model in available_models:
            try:
                response = client.chat(
                    model=model,
                    message=prompt,
                    max_tokens=300,
                    temperature=0.3,
                )
                return response.text.strip()
            except:
                continue
                
        return "I apologize, but I'm currently unable to process your request. Please try again later."
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# ... (rest of your Streamlit UI code)
