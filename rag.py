# Install necessary packages if running fresh (uncomment if needed)
# !pip install langchain pinecone-client huggingface_hub sentence_transformers

import os
import time
import warnings
warnings.filterwarnings("ignore")

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

import pinecone
PINECONE_API_KEY = "7G0SW3WIlozJq4xEBSGLP8473NBo9eWsX48IO4N5"  # Coherent key (assuming it's the Pinecone API key)
SRKAR_KEY = "pcsk_4CUNx5_UcD5JvBnujd1UdkGJAFg6LXqim31JRmDquw1jryL392Wg7n7KkJGKFj86mmHW3H"  # Srikar Key (used in code as userdata.get)
HF_TOKEN = "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"  # Hugging Face API Token


# Set your API keys either from environment variables or assign here
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # or your actual key string
HF_API_TOKEN = os.getenv("HF_API_TOKEN")          # or your actual key string

# Validate API keys exist
assert PINECONE_API_KEY is not None, "Pinecone API key is required"
assert HF_API_TOKEN is not None, "HuggingFace API token is required"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")  # environment might vary

# Load the document to create knowledge base
loader = TextLoader("Machine Learning Operations.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Initialize embeddings
embedding = HuggingFaceEmbeddings()

# Set Pinecone index name
index_name = "srikar"

# Create Pinecone index if doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric="cosine")
    # Wait for the index to be ready
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Create Pinecone vectorstore from documents
index = pinecone.Index(index_name)
docsearch = Pinecone.from_documents(docs, embedding, index_name=index_name, pinecone_index=index)

# Initialize HuggingFaceHub LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id, 
    huggingfacehub_api_token=HF_API_TOKEN
)

# Define prompt template for MLOps domain
template = """
You are a MLOps Engineer, the user is going to ask some questions about Machine Learning Operations.
Use the following context to answer the question.
IF YOU DON'T KNOW THE ANSWER, JUST SAY DON'T KNOW
KEEP THE ANSWER BRIEF

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create a retrieval question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=False,
    combine_prompt=prompt
)

# To query the system:
query = "What is MLOps?"
answer = qa_chain.run(query)
print(answer)
