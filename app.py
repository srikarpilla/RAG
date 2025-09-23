import os
import streamlit as st
import nltk
import ssl
from pypdf import PdfReader
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

# Download all required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        st.warning("punkt_tab not available, using standard punkt tokenizer")

# Get API key from secrets or environment variables
try:
    COHERE_API_KEY = st.secrets["cohere"]["api_key"]
except (KeyError, FileNotFoundError):
    COHERE_API_KEY = os.environ.get('COHERE_API_KEY', '')

if not COHERE_API_KEY:
    st.error("❌ Cohere API key not found. Please configure it in your deployment platform's secrets.")
    st.stop()

# Initialize global variables in session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'original_docs' not in st.session_state:
    st.session_state.original_docs = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer()

CHUNK_SIZE = 1000
NUMBER_OF_MATCHES = 3
ps = PorterStemmer()

def process_text(txt, chunk_size=CHUNK_SIZE):
    # Use try-except for tokenizer fallback
    try:
        sentences = nltk.sent_tokenize(txt)
    except LookupError:
        # Fallback if tokenizer fails
        st.warning("NLTK tokenizer not available, using simple sentence splitting")
        sentences = txt.split('. ')
    
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
    try:
        with open(filepath, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return process_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return [], []

def read_html(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            soup = bs(file, "html.parser")
            text = soup.get_text(separator=" ")
        return process_text(text)
    except Exception as e:
        st.error(f"Error reading HTML: {e}")
        return [], []

def read_text(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        return process_text(text)
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return [], []

def add_document(texts):
    st.session_state.documents.extend(texts)
    if st.session_state.documents:  # Only fit if we have documents
        st.session_state.vectors = st.session_state.vectorizer.fit_transform(st.session_state.documents)
    return st.session_state.vectors

def process_and_add_document(filepath, filetype):
    try:
        if filetype == "pdf":
            original_data, processed_text = read_pdf(filepath)
        elif filetype == "html":
            original_data, processed_text = read_html(filepath)
        elif filetype == "txt":
            original_data, processed_text = read_text(filepath)
        else:
            st.error(f"Unsupported file format: {filetype}")
            return False

        if not original_data:
            st.error("No content could be extracted from the file.")
            return False

        st.session_state.original_docs.extend(original_data)
        add_document(processed_text)
        return True
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False

def find_best_matches(query, top_n=NUMBER_OF_MATCHES):
    if st.session_state.vectors is None or st.session_state.vectors.shape[0] == 0:
        return ["No documents available for search."]
    
    try:
        processed_query = [' '.join(ps.stem(word) for word in query.split())]
        query_vector = st.session_state.vectorizer.transform(processed_query)
        similarity = (query_vector * st.session_state.vectors.T).toarray()[0]
        best_match_indices = similarity.argsort()[::-1][:top_n]
        return [st.session_state.original_docs[i] for i in best_match_indices]
    except Exception as e:
        st.error(f"Error finding matches: {e}")
        return ["Error processing your query."]

def get_resp(query, context):
    try:
        client = cohere.Client(COHERE_API_KEY)
        
        prompt_context = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context)])
        
        prompt = f"""You are an AI assistant. Use the provided context to answer the user's query accurately and concisely.

Available Context:
{prompt_context}

User Query: {query}

Please provide a helpful answer based on the context above. If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."

Answer:"""
        
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

def reset_database():
    st.session_state.documents.clear()
    st.session_state.original_docs.clear()
    st.session_state.vectors = None
    st.session_state.vectorizer = TfidfVectorizer()  # Reset vectorizer too

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) Chatbot")

st.sidebar.header("Upload Document (PDF, HTML, TXT)")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document:",
    type=["pdf", "html", "txt"],
    key="file_uploader"
)

# Add a button to reset the database
if st.sidebar.button("Reset Database"):
    reset_database()
    st.sidebar.success("Database reset successfully!")
    st.rerun()

# Process uploaded file
if uploaded_file is not None:
    # Check if this is a new file upload
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Processing document..."):
            # Save the file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            ext = uploaded_file.name.split('.')[-1].lower()
            success = process_and_add_document(uploaded_file.name, ext)
            
            if success:
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.document_processed = True
                st.sidebar.success(f"✅ File '{uploaded_file.name}' indexed successfully!")
                st.sidebar.info(f"📄 Indexed {len(st.session_state.original_docs)} chunks from the document.")
                # Remove temporary file
                if os.path.exists(uploaded_file.name):
                    os.remove(uploaded_file.name)
                st.rerun()
            else:
                st.sidebar.error(f"❌ Error processing file: {uploaded_file.name}")
                # Remove temporary file
                if os.path.exists(uploaded_file.name):
                    os.remove(uploaded_file.name)

# Check if we have documents indexed and show the chat interface
if (st.session_state.vectors is not None and 
    st.session_state.vectors.shape[0] > 0 and 
    len(st.session_state.original_docs) > 0):
    
    st.success(f"✅ Document ready! You can now ask questions about the uploaded content.")
    st.subheader("Ask a Question")
    
    user_input = st.text_input(
        "Enter your question about the uploaded document:", 
        key="question_input",
        placeholder="Type your question here..."
    )
    
    col1, col2 = st.columns([1, 6])
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    
    if ask_button and user_input.strip():
        with st.spinner("🔍 Searching for relevant context and generating answer..."):
            # Find relevant context
            context = find_best_matches(user_input)
            
            # Generate answer
            answer = get_resp(user_input, context)
            
            # Display results
            st.markdown("### 🤖 Answer:")
            st.success(answer)
            
            # Show relevant context in an expandable section
            with st.expander("📚 View Relevant Context Used", expanded=False):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**Context Chunk {i}:**")
                    if len(c) > 300:
                        st.text(c[:300] + "...")
                        with st.expander(f"Show full context chunk {i}"):
                            st.text(c)
                    else:
                        st.text(c)
                    st.markdown("---")
    
    # Show document statistics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Info")
    st.sidebar.info(f"**Chunks processed:** {len(st.session_state.original_docs)}")
    
else:
    # Show initial instructions
    st.info("👆 Upload a document (PDF, HTML, or TXT) to start chatting with the RAG system.")
    st.markdown("""
    ### How to use:
    1. **Upload a document** using the sidebar uploader
    2. **Wait for processing** - you'll see a success message when done
    3. **Ask questions** about the document content
    4. **View answers** with the relevant context used
    
    **Supported formats:** PDF, HTML, TXT files
    **Max file size:** 200MB (Streamlit limit)
    """)

# Add debug information in sidebar (optional)
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("Debug Information:")
    st.sidebar.write(f"Documents in memory: {len(st.session_state.documents)}")
    st.sidebar.write(f"Original docs: {len(st.session_state.original_docs)}")
    st.sidebar.write(f"Vectors shape: {st.session_state.vectors.shape if st.session_state.vectors is not None else 'None'}")
