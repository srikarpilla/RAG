import os
import streamlit as st
import nltk
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup as bs
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import cohere

nltk.download("punkt")

# Load Cohere API key from Streamlit secrets
COHERE_API_KEY = st.secrets["cohere"]["api_key"]

CHUNK_SIZE = 1000
NUMBER_OF_MATCHES = 3
ps = PorterStemmer()

documents = []
original_docs = []
vectors = None
vectorizer = TfidfVectorizer()

def process_text(txt, chunk_size=CHUNK_SIZE):
    sentences = nltk.sent_tokenize(txt)
    original_text = []
    processed_text = []
    chunk = ""
    for x in sentences:
        if len(chunk + x) < chunk_size:
            chunk += " " + x
        else:
            original_text.append(chunk.strip())
            processed_text.append(' '.join(ps.stem(word) for word in chunk.split()))
            chunk = x
    if chunk:
        original_text.append(chunk.strip())
        processed_text.append(' '.join(ps.stem(word) for word in chunk.split()))
    return original_text, processed_text

def read_pdf(filepath):
    with open(filepath, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return process_text(text)

def read_html(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        soup = bs(file, "html.parser")
        text = soup.get_text(separator=" ")
    return process_text(text)

def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    return process_text(text)

def add_document(texts):
    global vectors, documents
    documents.extend(texts)
    vectors = vectorizer.fit_transform(documents)
    return vectors

def process_and_add_document(filepath, filetype):
    if filetype == "pdf":
        original_data, processed_text = read_pdf(filepath)
    elif filetype == "html":
        original_data, processed_text = read_html(filepath)
    elif filetype == "txt":
        original_data, processed_text = read_text(filepath)
    else:
        raise ValueError("Unsupported file format")

    original_docs.extend(original_data)
    return add_document(processed_text)

def find_best_matches(query, top_n=NUMBER_OF_MATCHES):
    processed_query = [' '.join(ps.stem(word) for word in query.split())]
    query_vector = vectorizer.transform(processed_query)
    similarity = (query_vector * vectors.T).toarray()[0]
    best_match_indices = similarity.argsort()[::-1][:top_n]
    return [original_docs[i] for i in best_match_indices]

def get_resp(query, context):
    client = cohere.Client(COHERE_API_KEY)
    prompt_context = "\n".join(context)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant. Use the provided context to answer the user's query accurately and concisely."
        },
        {
            "role": "system",
            "content": prompt_context
        },
        {
            "role": "user",
            "content": query
        }
    ]
    response = client.chat(
        model="command-r-plus",
        messages=messages,
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def reset_database():
    global documents, original_docs, vectors
    documents.clear()
    original_docs.clear()
    vectors = None

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) Chatbot")

st.sidebar.header("Upload Document (PDF, HTML, TXT)")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document:",
    type=["pdf", "html", "txt"]
)

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        reset_database()
        process_and_add_document(uploaded_file.name, ext)
        st.sidebar.success(f"File '{uploaded_file.name}' indexed successfully.")
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")

if vectors is not None:
    user_input = st.text_input("Ask a question about the uploaded document:")
    if st.button("Ask"):
        if user_input.strip():
            context = find_best_matches(user_input)
            answer = get_resp(user_input, context)
            st.markdown("**Answer:**")
            st.write(answer)
            st.markdown("**Relevant Context:**")
            for i, c in enumerate(context, 1):
                st.markdown(f"{i}. {c}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload and index a document to start chatting.")
