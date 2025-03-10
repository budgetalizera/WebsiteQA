import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import os
import sys
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext

# Load environment variables
load_dotenv()
api_key_g = os.getenv("GROQ_API_KEY")
api_key_p = os.getenv("PINECONE_API_KEY")

# Ensure API keys are set
if not api_key_g:
    raise ValueError("Missing GROQ API Key. Set GROQ_API_KEY in .env")
if not api_key_p:
    raise ValueError("Missing PINECONE API Key. Set PINECONE_API_KEY in .env")

# Initialize LLM (Groq Llama3)
llm = Groq(api_key=api_key_g, model="llama3-8b-8192")

# Setup Embedding Model (Single Instance)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to validate URL
def is_valid_url(url):
    regex = re.compile(
        r'^(https?://)?'  # http:// or https:// (optional)
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # Domain name
        r'(:\d+)?'  # Optional port
        r'(/.*)?$',  # Optional path
        re.IGNORECASE
    )
    return re.match(regex, url)

# Function to crawl website
def crawl_website(start_url, max_pages=50):
    visited = set()
    sitemap = []
    queue = [start_url]
    
    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue
        except requests.RequestException:
            continue
        
        visited.add(url)
        sitemap.append(url)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            if urlparse(absolute_url).netloc == urlparse(start_url).netloc and absolute_url not in visited:
                queue.append(absolute_url)
    
    return sitemap

# Function for cleaning text
def clean_text(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    return ' '.join(soup.stripped_strings)

# Function for scraping data from sitemap URLs
def scrape_sitemap_urls(sitemap_urls):
    scraped_data = []
    
    
    for url in sitemap_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue
        except requests.RequestException:
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = clean_text(soup)
        scraped_data.append(f"URL: {url}\n{page_text}\n\n")
    
    return scraped_data

# Function to save scraped data
def save_scraped_data(scraped_data, filename="scraped_data.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for data in scraped_data:
            f.write(data + "\n")
    return filename

# Initialize Pinecone
def initialize_pinecone(index_name):
    pc = Pinecone(api_key=api_key_p)
    try:
        pc.describe_index(index_name)
        return pc, index_name, True
    except Exception:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        return pc, index_name, False

# Load Documents
def load_documents(file_path):
    reader = SimpleDirectoryReader(input_files=[file_path])
    return reader.load_data()

# Function to chunk the document using LlamaIndex
def chunk_documents(docs, chunk_size=512, chunk_overlap=50):
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.get_nodes_from_documents(docs)

# Upload Chunks to Pinecone
def generate_and_upload_embeddings(chunks, index):
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text
        embedding = embed_model.get_text_embedding(chunk_text)
        metadata = {"chunk_id": i, "text": chunk_text}
        index.upsert(vectors=[(f"chunk-{i}", embedding, metadata)])

# Retrieve Related Sections from Pinecone
def retrieve_related_sections(user_query, index):
    query_vector = embed_model.get_text_embedding(user_query)
    response = index.query(vector=query_vector, top_k=5, include_metadata=True)
    return [match["metadata"]["text"] for match in response.get("matches", [])]

# Generate a Response from Context
def generate_response(user_query, relevant_sections):
    context = "\n\n".join(relevant_sections)
    prompt = f"""
        You are an expert in Extracting data from websites.
        Use the following context to answer the user's query.
        
        Context:
        {context}
        
        User Query:
        {user_query}
        
        Provide a clear and structured response.
    """
    return llm.complete(prompt=prompt).text

# Query the RAG system
def query_rag_pipeline(user_query, index):
    relevant_sections = retrieve_related_sections(user_query, index)
    return generate_response(user_query, relevant_sections)





######################################### Streamlit UI######################################################
st.title("Web Scraping & RAG System")
st.sidebar.header("Settings")

website_url = st.sidebar.text_input("Enter Website URL")
user_query = st.sidebar.text_input("Enter Query")

if st.sidebar.button("Run Pipeline"):
    if not website_url or not user_query:
        st.sidebar.error("Please enter both a website URL and a query.")
    else:
        with st.status("🚀 Initializing the RAG Pipeline...", expanded=True) as status:
            parsed_url = urlparse(website_url)
            domain_parts = parsed_url.netloc.split(".")
            website_name = domain_parts[1] if len(domain_parts) > 1 else domain_parts[0]

            st.write(f"🌐 Extracted website name: `{website_name}`")

            pc, index_name, already_exists = initialize_pinecone(website_name)

            if already_exists:
                st.write(f"🔍 Index `{index_name}` already exists. Skipping data collection.")
                response = query_rag_pipeline(user_query, pc.Index(name=index_name))
            else:
                st.write("🕷️ Crawling website for data...")
                sitemap = crawl_website(website_url)

                st.write("📄 Scraping data from sitemap URLs...")
                scraped_data = scrape_sitemap_urls(sitemap)

                st.write("💾 Saving scraped data...")
                filename = save_scraped_data(scraped_data)

                st.write("📚 Loading documents...")
                docs = load_documents(filename)

                st.write("🔍 Splitting documents into chunks...")
                chunks = chunk_documents(docs)

                st.write("🧠 Generating and uploading embeddings to Pinecone...")
                generate_and_upload_embeddings(chunks, pc.Index(name=index_name))

                st.write("✅ Data processing completed!")

                st.write("🤖 Running the RAG query...")
                response = query_rag_pipeline(user_query, pc.Index(name=index_name))

            status.update(label="✅ RAG Pipeline Completed!", state="complete")

        st.subheader("Generated Response:")
        st.write(response)