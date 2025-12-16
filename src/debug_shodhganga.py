import streamlit as st
import asyncio
import yaml
import json
import pandas as pd
import time
import os
import aiohttp
import PyPDF2
from io import BytesIO
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv 

# --- Load Environment Variables ---
load_dotenv()

# Import your existing modules
from paper_retriever import PaperRetrieverManager
from vector_store import VectorStore
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor
from shodhganga_retriever import ShodhgangaRetriever
from folder_manager import FolderManager

# --- Page Config ---
st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🧬",
    layout="wide"
)

# --- Helper Functions ---

def load_config(config_path='config/config.yaml'):
    if not Path(config_path).exists():
        st.error(f"Config file not found at {config_path}.")
        st.stop()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# --- Gemini Configuration ---
def configure_gemini():
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        try:
            genai.configure(api_key=env_key)
            return True
        except: pass

    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except: return False
    return False

# --- NEW: Function to Fetch URL Content & Summarize ---
async def fetch_and_analyze_link(url, title, context="Thesis Chapter"):
    """
    Simulates Gemini 'visiting' a link by downloading the PDF/Page 
    in Python and sending the text to the API.
    """
    text_content = ""
    
    # 1. Download Content
    try:
        # Handle Shodhganga specific port issues
        clean_url = url.replace(":8443", "") if ":8443" in url else url
        
        async with aiohttp.ClientSession() as session:
            async with session.get(clean_url, ssl=False, timeout=60) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    data = await response.read()
                    
                    # If PDF
                    if 'pdf' in content_type or url.endswith('.pdf') or 'bitstream' in url:
                        try:
                            reader = PyPDF2.PdfReader(BytesIO(data))
                            # Extract text from first 10 pages (to keep it fast and within token limits)
                            limit_pages = min(len(reader.pages), 10)
                            text_content = "\n".join([reader.pages[i].extract_text() for i in range(limit_pages)])
                        except Exception as e:
                            return f"Error reading PDF format: {e}"
                    else:
                        # Assume HTML/Text
                        text_content = data.decode('utf-8', errors='ignore')[:10000]
                else:
                    return f"Failed to reach link (HTTP {response.status})"
    except Exception as e:
        return f"Connection error: {str(e)}"

    if not text_content:
        return "Empty content found at link."

    # 2. Send to Gemini
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are analyzing a research document from the following source: "{title}".
        
        I have downloaded the content from the link. Please analyze this text:
        
        TEXT CONTENT (Excerpt):
        {text_content[:15000]}
        
        Please provide:
        1. **Core Subject**: What is this specific document/chapter about?
        2. **Key Arguments/Data**: What are the main points?
        3. **Relevance**: How does this relate to the thesis title?
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

# --- Session State ---
if 'components_loaded' not in st.session_state:
    with st.spinner("Initializing components..."):
        config = load_config()
        st.session_state.folder_mgr = FolderManager(config['storage']['papers_directory'])
        st.session_state.paper_manager = PaperRetrieverManager(config['apis'], st.session_state.folder_mgr)
        st.session_state.vector_store = VectorStore(config['vector_db'])
        st.session_state.pdf_processor = PDFProcessor(config)
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store, config['rag'])
        if 'shodhganga' in config['apis']:
            st.session_state.shodhganga = ShodhgangaRetriever(config['apis']['shodhganga'])
        else:
            st.session_state.shodhganga = None
        st.session_state.config = config
        st.session_state.components_loaded = True

# --- UI ---
st.sidebar.title("🧬 Deep Researcher")
page = st.sidebar.radio("Navigate", ["Search & Download", "Shodhganga Thesis", "Chat with Papers", "Library & Manage"])
gemini_enabled = configure_gemini()
if gemini_enabled:
    st.sidebar.success("AI Active")

# --- PAGE: Search & Download ---
if page == "Search & Download":
    st.header("🌐 Search Global Papers")
    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Query")
    with col2: num = st.number_input("Max", 1, 20, 3)
    sources = st.multiselect("Sources", ["arxiv", "semantic_scholar"], default=["arxiv"])
    
    if st.button("Fetch"):
        status = st.status("Processing...", expanded=True)
        async def run_search():
            res = []
            for src in sources:
                fldr = st.session_state.folder_mgr.create_search_folder(src, query)
                papers = await st.session_state.paper_manager.search_papers(query, [src], num)
                for p in papers:
                    with open(fldr/f"{p.id}.json", 'w') as f:
                        from dataclasses import asdict
                        json.dump(asdict(p), f, indent=2, default=str)
                    
                    txt = await st.session_state.pdf_processor.download_and_extract_pdf(p) if p.pdf_url else None
                    await st.session_state.vector_store.add_paper(p, txt)
                    res.append(p)
            return res
        
        results = run_async(run_search())
        status.update(label="Done", state="complete")
        if results:
            st.dataframe(pd.DataFrame([{"Title": p.title, "Source": p.source} for p in results]), width=2000)

# --- PAGE: Shodhganga Thesis (UPDATED) ---
elif page == "Shodhganga Thesis":
    st.header("🇮🇳 Shodhganga Thesis Search")
    
    col1, col2 = st.columns([3, 1])
    with col1: s_query = st.text_input("Topic")
    with col2: s_uni = st.text_input("University")
    s_max = st.number_input("Max Results", 1, 10, 3)
    
    if st.button("Search Theses", type="primary"):
        status = st.status("Searching Shodhganga...", expanded=True)
        
        async def run_shodh():
            fldr = st.session_state.folder_mgr.create_search_folder('shodhganga', s_query)
            theses = await st.session_state.shodhganga.search_theses(s_query, s_uni, s_max)
            processed = []
            
            if not theses: return []
            
            for t in theses:
                status.write(f"Processing: {t.title[:50]}...")
                tf = st.session_state.folder_mgr.create_thesis_folder(fldr, t.id, t.title)
                await st.session_state.shodhganga.save_thesis_info(t, tf)
                
                # Basic Indexing
                await st.session_state.vector_store.add_paper(t, f"Title: {t.title}\nAbstract: {t.abstract}")
                
                processed.append({
                    "Title": t.title, 
                    "Researcher": ", ".join(t.authors),
                    "University": t.metadata.get('university', 'Unknown'),
                    "Chapters": len(t.chapter_links) if t.chapter_links else 0,
                    "Abstract": t.abstract,
                    "ChapterLinks": t.chapter_links, # Save links for Gemini
                    "ID": t.id
                })
            return processed

        results = run_async(run_shodh())
        status.update(label="Done", state="complete")
        
        if results:
            st.session_state.shodh_results = results

    if hasattr(st.session_state, 'shodh_results') and st.session_state.shodh_results:
        st.subheader("📚 Results")
        df = pd.DataFrame(st.session_state.shodh_results)
        st.dataframe(df[["Title", "Researcher", "University", "Chapters"]], width=2000)
        
        st.markdown("### 📝 AI Deep Analysis")
        for item in st.session_state.shodh_results:
            with st.expander(f"📄 {item['Title']}"):
                st.info(f"**Abstract:** {item['Abstract']}")
                
                if gemini_enabled:
                    st.markdown("#### 🤖 Analyze Specific Content")
                    
                    # Option 1: Summarize Abstract
                    if st.button("✨ Summarize Abstract", key=f"abs_{item['ID']}"):
                         model = genai.GenerativeModel('gemini-1.5-flash')
                         resp = model.generate_content(f"Summarize this thesis abstract: {item['Abstract']}")
                         st.success(resp.text)

                    # Option 2: Analyze specific Chapters (Gemini "goes to link")
                    if item['ChapterLinks']:
                        st.markdown("or select a chapter to analyze:")
                        
                        # Create a dropdown for chapters
                        chapter_map = {f"{c['name']}": c['url'] for c in item['ChapterLinks']}
                        selected_chap = st.selectbox("Select Chapter", list(chapter_map.keys()), key=f"sel_{item['ID']}")
                        
                        if st.button(f"🚀 Gemini: Go to Link & Analyze", key=f"link_{item['ID']}"):
                            target_url = chapter_map[selected_chap]
                            
                            with st.status("🤖 Gemini Agent Working...", expanded=True) as agent_status:
                                agent_status.write("1️⃣ Connecting to Shodhganga...")
                                agent_status.write(f"🔗 Target: {target_url}")
                                
                                # Run the Fetch -> Extract -> Summarize pipeline
                                summary = run_async(fetch_and_analyze_link(target_url, item['Title']))
                                
                                agent_status.write("2️⃣ Content Downloaded & Read")
                                agent_status.write("3️⃣ Generating Insight...")
                            
                            st.markdown(f"#### 🧠 Analysis of: {selected_chap}")
                            st.markdown(summary)
                    else:
                        st.caption("No direct chapter links found to analyze.")
                else:
                    st.warning("Enable Gemini to use AI features.")

# --- PAGE: Chat ---
elif page == "Chat with Papers":
    st.header("💬 Chat")
    try: papers = st.session_state.vector_store.get_all_papers_metadata()
    except: papers = []
    
    opts = ["All"] + [p['title'] for p in papers]
    sel = st.selectbox("Context", opts)
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: 
        with st.chat_message(m["role"]): st.markdown(m["content"])
        
    if p := st.chat_input("Ask..."):
        st.session_state.messages.append({"role":"user", "content":p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                filter = {"paper_id": next((x['id'] for x in papers if x['title'] == sel), None)} if sel != "All" else None
                res = run_async(st.session_state.rag_engine.query(p, context_filter=filter))
                st.markdown(res.answer)
                st.session_state.messages.append({"role":"assistant", "content":res.answer})

# --- PAGE: Library ---
elif page == "Library & Manage":
    st.header("📚 Library")
    try: papers = st.session_state.vector_store.get_all_papers_metadata()
    except: papers = []
    
    if not papers: st.info("Empty")
    else:
        for p in papers:
            with st.expander(f"📄 {p['title']}"):
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"**ID:** {p['id']}")
                    st.write(f"**Source:** {p['source']}")
                with col2:
                    if st.button("Delete", key=f"del_{p['id']}"):
                        run_async(st.session_state.vector_store.delete_paper(p['id']))
                        st.session_state.folder_mgr.delete_paper_files(p['id'])
                        st.rerun()