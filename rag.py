import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import os

# --- Setup (fill your own keys in Secrets or environment variables) ---
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

index_name = "srikar"  # Your Pinecone index name

# Initialize embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings()

# Initialize Pinecone vector store
@st.cache_resource(show_spinner=False)
def load_pinecone_index():
    import pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")
    return Pinecone.from_existing_index(index_name, embedding_model, pinecone_index=pinecone.Index(index_name))

# Initialize language model
@st.cache_resource(show_spinner=False)
def load_llm():
    return HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

# Define prompt template
prompt_template = """
You are a MLOPs Engineer, the user is going to ask some questions about Machine Learning Operations.
Use the following context to answer the question.
IF YOU DON'T KNOW THE ANSWER, JUST SAY DON'T KNOW
KEEP THE ANSWER BRIEF

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load models
embedding_model = load_embedding_model()
vectorstore = load_pinecone_index()
llm = load_llm()

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("📖 RAG Q&A for MLOps")

user_query = st.text_input("Enter your question about Machine Learning Operations:")

if user_query:
    with st.spinner("Searching for answers..."):
        answer = qa_chain.run(user_query)
    st.markdown("### Answer:")
    st.write(answer)
