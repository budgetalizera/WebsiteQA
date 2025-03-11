import streamlit as st
import tldextract
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import os
import sys
import asyncio
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
from googletrans import Translator, LANGUAGES

##################################################################################################################

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


##################################################################################################################

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

# Function for detecting language
def detect_and_translate(text, target_language="en"):
    translator = Translator()
    
    # Handle async coroutine execution properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    detected_lang_code = loop.run_until_complete(translator.detect(text)).lang
    detected_lang = LANGUAGES.get(detected_lang_code, "Unknown")
    # print("🔍 Detected language:", detected_lang)
    translated_text = loop.run_until_complete(translator.translate(text, dest=target_language)).text
    # print("🔤 Translated text:", translated_text)
    
    return detected_lang, translated_text

def translate_query(text, target_language="en"):
    translator = Translator()
    
    # Handle async coroutine execution properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    translated_text = loop.run_until_complete(translator.translate(text, dest=target_language)).text
    return translated_text

# Function to crawl website
# def crawl_website(start_url, max_pages=50):
#     visited = set()
#     sitemap = []
#     queue = [start_url]
    
#     while queue and len(visited) < max_pages:
#         url = queue.pop(0)
#         if url in visited:
#             continue
        
#         try:
#             response = requests.get(url, timeout=5)
#             if response.status_code != 200:
#                 continue
#         except requests.RequestException:
#             continue
        
#         visited.add(url)
#         sitemap.append(url)
        
#         soup = BeautifulSoup(response.text, 'html.parser')
#         for link in soup.find_all('a', href=True):
#             absolute_url = urljoin(url, link['href'])
#             if urlparse(absolute_url).netloc == urlparse(start_url).netloc and absolute_url not in visited:
#                 queue.append(absolute_url)
    
#     return sitemap

def crawl_website(start_url, max_pages=50):
    visited = set()
    sitemap = []
    queue = [start_url]

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                continue
            
            # Ensure the content is HTML
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue  

            response.encoding = response.apparent_encoding  # Fix encoding
            soup = BeautifulSoup(response.text, 'html.parser')

        except requests.RequestException:
            continue
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
        
        visited.add(url)
        sitemap.append(url)

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
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)  # Ensure the folder exists
    
    file_path = os.path.join(data_folder, filename)  # Save in "data" folder
    
    with open(file_path, "w", encoding="utf-8") as f:  # Overwrite file if it exists
        for data in scraped_data:
            f.write(data + "\n")
    
    return file_path  # Return the full file path

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
    # print("💀💀💀",file_path)
    reader = SimpleDirectoryReader(input_dir="data")
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
def generate_response(user_query, relevant_sections, user_language="english"):
    context = "\n\n".join(relevant_sections)
    
    instruction_message = (
        f"The requested information is currently unavailable on the website. "
        f"Please visit the official website for further details. (Response in {user_language} language with proper meaningful way.)"
    )

    prompt = f"""
        You are an expert in extracting data from websites.
        Use the following context to answer the user's query in the requested language.
        
        Context:
        {context}
        
        User Query:
        {user_query}
        
        Provide a clear and structured response in the {user_language} language.
        
        Important Instructions:
        - Only answer the question if it is relevant to the provided context.
        - If the query is unrelated to the given context, respond with:
          "{instruction_message}"
        - Do not generate assumptions or hallucinate answers beyond the provided information.
    """
    
    return llm.complete(prompt=prompt).text

# Query the RAG system
def query_rag_pipeline(user_query, index, user_language="en"):
    relevant_sections = retrieve_related_sections(user_query, index)
    # print("🤞🤞🤞🤞🤞", user_language)
    return generate_response(user_query, relevant_sections, user_language)





######################################### Streamlit UI######################################################

# Streamlit UI
st.title("Web Scraping & RAG System")
st.sidebar.header("Settings")

website_url = st.sidebar.text_input("Enter Website URL")

# Language selection dropdown (Default: Detect Language)
language_options = ["Detect Language", "English", "French", "Spanish", "German", "Chinese", "Hindi"]
selected_language = st.sidebar.selectbox("Select Language", language_options, index=0)
# print("❤️💀❤️ Selected Language:", selected_language)

user_query = st.sidebar.text_input("Enter Query")

if st.sidebar.button("Run Pipeline"):
    if not website_url or not user_query:
        st.sidebar.error("Please enter both a website URL and a query.")
    else:
        with st.status("🚀 Initializing the RAG Pipeline...", expanded=True) as status:
            start_time = time.time()
            
            # Detect language if default option is selected
            if selected_language == "Detect Language":
                detected_lang, llm_query = detect_and_translate(user_query, "en")
                st.write(f"🔍 Detected language: `{detected_lang}`")
            else:
                detected_lang = selected_language
                st.write(f"🔍 Selected language: `{detected_lang}`")
                llm_query = translate_query(user_query)  # Use the query as-is if language is manually selected

            st.write(f"🔤 Final query in for LLM: `{llm_query}`")
            
            
            website_name = tldextract.extract(website_url).domain
            st.write(f"🌐 Extracted website name: `{website_name}`")

            pc, index_name, already_exists = initialize_pinecone(website_name)

            if already_exists:
                st.write(f"🔍 Index `{index_name}` already exists. Skipping data collection.")
                response = query_rag_pipeline(llm_query, pc.Index(name=index_name), detected_lang)
            else:
                crawl_start = time.time()
                st.write("🕷️ Crawling website for data...")
                sitemap = crawl_website(website_url)
                st.write(f"✅ Crawling completed in {time.time() - crawl_start:.2f} seconds.")

                scrape_start = time.time()
                st.write("📄 Scraping data from sitemap URLs...")
                scraped_data = scrape_sitemap_urls(sitemap)
                st.write(f"✅ Scraping completed in {time.time() - scrape_start:.2f} seconds.")

                save_start = time.time()
                st.write("💾 Saving scraped data...")
                filename = save_scraped_data(scraped_data)
                st.write(f"✅ Data saved in {time.time() - save_start:.2f} seconds.")

                st.write("📚 Loading documents...")
                docs = load_documents(filename)

                st.write("🔍 Splitting documents into chunks...")
                chunks = chunk_documents(docs)

                embed_start = time.time()
                st.write("🧠 Generating and uploading embeddings to Pinecone...")
                generate_and_upload_embeddings(chunks, pc.Index(name=index_name))
                st.write(f"✅ Embeddings uploaded in {time.time() - embed_start:.2f} seconds.")

                st.write("✅ Data processing completed!")
                query_start = time.time()
                st.write("🤖 Running the RAG query...")
                response = query_rag_pipeline(llm_query, pc.Index(name=index_name), detected_lang)
                st.write(f"✅ Query processed in {time.time() - query_start:.2f} seconds.")

            total_time = time.time() - start_time
            status.update(label=f"✅ RAG Pipeline Completed in {total_time:.2f} seconds!", state="complete")

        st.subheader("Generated Response:")
        st.write(response)

