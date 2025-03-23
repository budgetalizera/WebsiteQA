import streamlit as st
import tldextract
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import os
import asyncio
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from googletrans import Translator, LANGUAGES
import torchaudio
import numpy as np
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    SeamlessM4TFeatureExtractor,
    SeamlessM4TTokenizer,
)
import tempfile
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from markdown import markdown
from langdetect import detect
from kokoro import KPipeline
from IPython.display import Audio   
import os
import spacy


#! Constant________________________________________________________________________________________________________________________________________________________

DetectorFactory.seed = 0  


SPEAKER_MAP = {
    "a": {"Female": "af_heart", "Male": "am_michael"},  # American English
    "f": {"Female": "ff_siwis", "Male": "ff_siwis"},  # French
    "e": {"Female": "ef_dora", "Male": "em_alex"},  # Spanish
    "j": {"Female": "jf_alpha", "Male": "jm_kumo"},  # Japanese
    "z": {"Female": "zf_xiaobei", "Male": "zm_yunxi"},  # Mandarin Chinese
    "h": {"Female": "hf_alpha", "Male": "hm_omega"},  # Hindi
    "i": {"Female": "if_sara", "Male": "im_nicola"},  # Italian
    "p": {"Female": "pf_dora", "Male": "pm_alex"},  # Brazilian Portuguese
}
# Your existing language map
TTS_LANG_CODE_MAP = {
    "en": "a",  # American English
    "fr": "f",  # French
    "es": "e",  # Spanish
    "ja": "j",  # Japanese
    "zh": "z",  # Mandarin Chinese
    "hi": "h",  # Hindi
    "it": "i",  # Italian
    "pt": "p",  # Brazilian Portuguese
}


# Language map for easier human-friendly language name mapping
LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "zh": "Mandarin Chinese",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Brazilian Portuguese",
}


load_dotenv()

    # Loading API keys
api_key_g = os.getenv("GROQ_API_KEY")
api_key_p = os.getenv("PINECONE_API_KEY")
hf_token =  os.getenv("HF_TOKEN")

if not api_key_g:
        raise ValueError("Missing GROQ API Key. Set GROQ_API_KEY in .env")
if not api_key_p:
        raise ValueError("Missing PINECONE API Key. Set PINECONE_API_KEY in .env")
if not hf_token:
        raise ValueError("Missing hf_token Key. Set hf_token in .env")


# os.system("pip install en-core-web-sm")
# print("en-core-web-sm installing....")
# spacy.load("en_core_web_sm")
# print("en-core-web-sm loading....")


#! Model Loading / Intialization -------------------------------------------------------------------------------------------------


# Caching function to initialize models (cached resource)
@st.cache_resource
def initialize_app():
    """Initialize and cache models, API keys"""
    st.write("🔄 **Initializing the app...**")
    
    # Load LLaMA models and embedding model
    llm = Groq(api_key=api_key_g, model="llama3-8b-8192", temperature = 0.17 )
    st.write("✅ **LLaMA model loaded successfully.**")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    st.write("✅ **HuggingFace embedding model loaded successfully.**")

    # Load SeamlessM4T model
    st.write("🧩 **Loading SeamlessM4T model...**")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "ai4bharat/indic-seamless"
    s_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME , token=hf_token)
    s_processor = SeamlessM4TFeatureExtractor.from_pretrained(MODEL_NAME, token=hf_token)
    s_tokenizer = SeamlessM4TTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    st.write("✅ **SeamlessM4T model loaded successfully.**")


    st.write("🎉 **App initialization complete.**")
    return llm, embed_model, s_model, s_processor, s_tokenizer, device



#! SCRAPPING AND RESPONSE GENERATION THROUGH RAG PIPELINE----------------------------------------------------------------------------

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

    translated_text = loop.run_until_complete(translator.translate(text, dest=target_language)).text
    return detected_lang, translated_text

def translate_query(text, target_language="en"):
    translator = Translator()
    
    # Handle async coroutine execution properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    translated_text = loop.run_until_complete(translator.translate(text, dest=target_language)).text
    
    return translated_text

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
    response = index.query(vector=query_vector, top_k=10, include_metadata=True)
    return [match["metadata"]["text"] for match in response.get("matches", [])]

# Generate a Response from Context
def generate_response(user_query, relevant_sections, user_language="english"):
    context = "\n\n".join(relevant_sections)
    
    instruction_message = (
        f"The requested information is currently unavailable on the website. "
        f"Please visit the official website for further details."
    )

    prompt = f"""
        You are an expert in extracting data from websites.
        Use the following context to answer the user's query.
        
        Context:
        {context}
        
        User Query:
        {user_query}
        
        Provide a clear and structured response in markdown format. 
        
        Important Instructions:
        - Response in proper meaningful way. Don't give response in any other language.
        - Only answer the question if it is relevant to the provided context.
        - If the query is unrelated to the given context, respond with:
          "{instruction_message}"
        - Do not generate assumptions or hallucinate answers beyond the provided information.
    """
 
    response = llm.complete(prompt=prompt).text
    
    #! #### Language Translation for response
    translator = Translator()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    translated_response = loop.run_until_complete(translator.translate(response, dest=user_language)).text


    #! ##### Structured opuput
    # structured_response = markdown(translated_response)
    #### Cleaned Response
    cleaned_response = re.sub(r"\*", "", translated_response)
    return cleaned_response


# Query the RAG system
def query_rag_pipeline(user_query, index, user_language="en"):
    relevant_sections = retrieve_related_sections(user_query, index)
    return generate_response(user_query, relevant_sections, user_language)

#! Audio Functionality (STT)--------------------------------------------------------------------------------------------------

def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name
    
    audio, orig_freq = torchaudio.load(temp_audio_path)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
    os.remove(temp_audio_path)
    return audio

def transcribe_audio(audio, model, processor, tokenizer, tgt_lang):
    audio_inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    text_out = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    transcript = tokenizer.decode(text_out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print("🖊❤", transcript)
    return transcript



#! Audio Functionality (TTS)----------------------------------------------------------------------------------------------------

async def detect_and_fallback(text, target_lang_code="en"):
    try:
        detected_lang_code = detect(text)  # Assuming you have a language detection function
    except Exception as e:
        detected_lang_code = "unknown"
        print(f"Language detection failed: {e}")
    
    supported_languages = set(LANGUAGE_MAP.keys())
    is_fallback = False
    
    if detected_lang_code not in supported_languages or detected_lang_code == "unknown":
        try:
            translated_text = await asyncio.to_thread(
                GoogleTranslator(source="auto", target=target_lang_code).translate, text
            )
            target_language = LANGUAGE_MAP.get(target_lang_code, "English")
            tts_lang_code = TTS_LANG_CODE_MAP.get(target_lang_code, "unknown")
            is_fallback = True 
            return target_lang_code, target_language, translated_text, tts_lang_code, is_fallback
        except Exception as e:
            
            return target_lang_code, "Translation Failed", "", "unknown", True
    else:
        detected_language = LANGUAGE_MAP.get(detected_lang_code, "Unknown Language")
        tts_lang_code = TTS_LANG_CODE_MAP.get(detected_lang_code, "unknown")
        return detected_lang_code, detected_language, text, tts_lang_code, is_fallback


def get_speakers(language):
    return SPEAKER_MAP.get(language, {}).items()

def generate_audio(text, lang_code, voice, gender):
  

    start_time = time.time()  # Start the timer
    text = re.sub(r'[#=*-]+', '', text)
    try:
        pipeline = KPipeline(lang_code=lang_code)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return None, None
    
    try:
        generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
        all_audio = []

        for i, (gs, ps, audio) in enumerate(generator):
            print("❤ Segment of text", i)
            all_audio.append(audio)

        final_audio = torch.cat(all_audio, dim=0)
        output_path = f"{voice}_{gender}.wav"
        print(f"✔ Audio generated successfully")
        sf.write(output_path, final_audio.numpy(), 24000)
        print(f"✔ Audio saved successfully")
        
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        st.write(f"✅ Speech Generation completed in {elapsed_time:.2f} seconds.")
        return output_path, final_audio
    
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None, None



#! NECESSASRY FUNCTION FOR UI ----------------------------------------------------------------------------------------------------



def handle_text_query():
    start_time = time.time()
    st.write(f"✍ Text Query Mode")

    # Detect or use selected language
    if selected_language == "Detect Language":
        detected_lang, llm_query = detect_and_translate(user_query, "en")
        st.write(f"🔍 Detected language: `{detected_lang}`")
    else:
        detected_lang = selected_language
        st.write(f"🔍 Selected language: `{detected_lang}`")
        llm_query = translate_query(user_query)

    st.write(f"🔤 Final query for LLM: `{llm_query}`")

    # Extract website name and initialize Pinecone
    website_name = tldextract.extract(website_url).domain
    st.write(f"🌐 Extracted website name: `{website_name}`")
    pc, index_name, already_exists = initialize_pinecone(website_name)

    if already_exists:
        st.write(f"🔍 Index `{index_name}` already exists. Skipping data collection.")
        response = query_rag_pipeline(llm_query, pc.Index(name=index_name), detected_lang)
    else:
        # Crawl, scrape, and process data if index doesn't exist
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
    st.write(f"✅ Response Generated in {total_time:.2f} seconds!")
    return response


def handle_audio_query():
    start_time = time.time()

    st.write(f"🔊 Audio Query Mode")

    target_languages = {
        "English": "eng", "Hindi": "hin", "Gujarati": "guj", "Tamil": "tam", "Telugu": "tel",
        "Marathi": "mar", "Assamese": "asm", "Bengali": "ben", "Urdu": "urd", "Kannada": "kan",
        "Sindhi": "snd", "Nepali": "nep"
    }

    if selected_language == "Detect Language":
        st.write(f"❌ Please select an appropriate language for transcription.")
        response = None
    else:
        detected_lang = selected_language
        st.write(f"🔍 Selected language: `{detected_lang}`")

        if detected_lang in target_languages:
            audio = process_audio(audio_query)
            transcript = transcribe_audio(audio, s_model, s_processor, s_tokenizer, target_languages[detected_lang])

            st.write(f"🖊 Transcribed Text: `{transcript}`")

            llm_query = translate_query(transcript)
            st.write(f"🔤 Final query for LLM: `{llm_query}`")

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
            st.write(f"✅ Response Generated in {total_time:.2f} seconds!")

        else:
            st.write(f"❌ Selected language `{detected_lang}` is not supported for transcription.")
            response = None

    return response










#! STREAMLIT UI -------------------------------------------------------------------------------------------------------------------


st.title("WebQ: Multilingual Q&A Assistant for Any Website")
st.subheader("A Multilingual Chatbot for Web Scraping and Language Processing")

with st.expander("🔍 Logs / Status", expanded=False):
    with st.spinner("🚀 Loading models and initializing application..."):
        llm, embed_model, s_model, s_processor, s_tokenizer, device = initialize_app()


st.sidebar.header("Settings")

website_url = st.sidebar.text_input("Enter Website URL")
website_name = tldextract.extract(website_url).domain


# Language selection dropdown (Default: Detect Language)
language_options = [
    "Detect Language", "Assamese", "Bengali", "Chinese", "English", 
    "French", "German", "Gujarati", "Hindi", "Kannada", 
    "Marathi", "Nepali", "Spanish", "Tamil", "Telugu"
]
selected_language = st.sidebar.selectbox("Select Language", language_options, index=0)

# Initialize session state variables
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'audio_query' not in st.session_state:
    st.session_state.audio_query = None

# Layout with two columns for Text and Audio inputs
col1, col2 = st.sidebar.columns(2)

# Text input in the first column
with col1:
    user_query = st.text_area("Enter Query", value=st.session_state.user_query , label_visibility="hidden")

# Audio input in the second column (simulating an audio recording)
with col2:
    audio_query= st.audio_input("Record a voice Query")

# When text is entered, clear the audio input
if user_query:
    st.session_state.user_query = user_query
    st.session_state.audio_query = None  # Clear audio when text is entered
 

# When audio input is triggered, clear the text input
if audio_query:
    st.session_state.user_query = ""
    st.session_state.audio_query = audio_query


if "mic_disabled" not in st.session_state:
    st.session_state.mic_disabled = True  


if st.sidebar.button(f"Ask anything about {website_name}"):
    if not website_url or (not user_query and not audio_query):
        st.sidebar.error("Please enter both a website URL and a query (either text or audio).")
    else:
        with st.expander("🔍 Logs / Status", expanded=False):
            if st.session_state.user_query:
                response = handle_text_query()
            elif st.session_state.audio_query:
                response = handle_audio_query()
    

        if response:
                st.subheader("Response:")
                st.markdown(response, unsafe_allow_html=True)
                detected_lang_code, language, result, tts_lang_code, is_fallback = asyncio.run(detect_and_fallback(response))

               
                speakers = get_speakers(tts_lang_code)
                 
                audio_data = None

                with st.expander("🔍 Logs / Status", expanded=False):
                    if is_fallback:
                        st.warning( "⚠️ Your language is not supported for speech generation. We have generated speech in **English** instead.")
                    for gender, speaker in speakers:

                         st.write(f"Detected Language for speech Generation: **{language}**")
                         st.write(f"Generated speech by **{gender}** speaker.")
                         output_path, audio_data = generate_audio(result, tts_lang_code, speaker, gender)
                         
                         if audio_data is not None:
                           st.subheader(f"🎧 Generated Speech in {speaker} ({gender}):")
                           st.audio(output_path)
               
                    if audio_data is None:
                        st.error("Something went wrong for generating audio. Please try again.")   
                    

                    
 

 

