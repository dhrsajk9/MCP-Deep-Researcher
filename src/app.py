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

# --- Import Modules ---
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
    """Setup Gemini and return the selected model"""
    st.sidebar.markdown("---")
    
    # Check session state for key
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_key)

    if not api_key:
        return False, None

    st.session_state.api_key = api_key

    try:
        genai.configure(api_key=api_key)
        st.sidebar.markdown("### 🤖 Model Settings")
        model_options = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]
        selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
        return True, selected_model
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")
        return False, None

# --- AI Functions ---
def get_gemini_summary(paper_id, paper_title, model_name, fallback_text=None):
    """Fetch data from DB and Elaborate with Gemini"""
    db_text = get_paper_content_from_db(paper_id)
    source_text = db_text if db_text else fallback_text
    
    if not source_text: return "No content found."
        
    try:
        model = genai.GenerativeModel(model_name)
        safe_text = source_text[:15000] 
        prompt = f"""
        You are an expert researcher. Paper Title: "{paper_title}".
        Provide a **Detailed Elaboration** of the text below.
        Structure: 1. Executive Summary, 2. Methodology, 3. Key Findings, 4. Core Concepts.
        SOURCE TEXT: {safe_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error: {str(e)}"

def get_paper_content_from_db(paper_id):
    try:
        collection = st.session_state.vector_store.collection
        result = collection.get(where={"paper_id": paper_id}, include=["documents"])
        if result and result['documents']: return "\n\n".join(result['documents'])
    except: pass
    return None

async def fetch_and_analyze_link(url, title, model_name):
    text_content = ""
    try:
        clean_url = url.replace(":8443", "") if ":8443" in url else url
        async with aiohttp.ClientSession() as session:
            async with session.get(clean_url, ssl=False, timeout=60) as response:
                if response.status == 200:
                    data = await response.read()
                    if 'pdf' in url.lower() or 'bitstream' in url:
                        try:
                            reader = PyPDF2.PdfReader(BytesIO(data))
                            text_content = "\n".join([p.extract_text() for p in reader.pages[:10]])
                        except: return "Error reading PDF."
                    else:
                        text_content = data.decode('utf-8', errors='ignore')[:10000]
    except Exception as e: return f"Connection error: {e}"

    if not text_content: return "Empty content."

    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"Analyze text from '{title}':\n\n{text_content[:15000]}\n\nProvide: 1. Main Topic 2. Key Arguments 3. Conclusion"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI Error: {e}"

# --- Session State ---
if 'components_loaded' not in st.session_state:
    with st.spinner("Initializing..."):
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

gemini_active, active_model = configure_gemini()
if gemini_active:
    st.sidebar.success(f"Active: {active_model}")

# --- PAGE: Search & Download ---
if page == "Search & Download":
    st.header("🌐 Search Global Papers")
    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Query")
    with col2: num = st.number_input("Count", 1, 20, 3)
    sources = st.multiselect("Sources", ["arxiv", "semantic_scholar"], default=["arxiv"])
    
    if st.button("Fetch Papers", type="primary"):
        status = st.status("Processing...", expanded=True)
        async def run_search():
            res = []
            for src in sources:
                fldr = st.session_state.folder_mgr.create_search_folder(src, query)
                papers = await st.session_state.paper_manager.search_papers(query, [src], num)
                if not papers: continue
                for p in papers:
                    with open(fldr/f"{p.id}.json", 'w', encoding='utf-8') as f:
                        from dataclasses import asdict
                        json.dump(asdict(p), f, indent=2, default=str)
                    
                    pdf_path = None
                    dl_status = "Skipped"
                    if p.pdf_url:
                        try:
                            # Pass folder to save PDF correctly
                            result = await st.session_state.pdf_processor.download_and_extract_pdf(p, save_dir=fldr)
                            if result and os.path.exists(result):
                                pdf_path = result
                                dl_status = "Success"
                        except Exception as e: dl_status = f"Failed: {e}"
                    
                    # Index
                    inferred_path = fldr / f"{p.id}.pdf"
                    actual_path = str(inferred_path) if inferred_path.exists() else None
                    await st.session_state.vector_store.add_paper(p, result if isinstance(result, str) else None)
                    
                    res.append({
                        "Title": p.title, "Authors": ", ".join(p.authors[:3]), 
                        "Source": src, "PDF Status": dl_status, 
                        "Abstract": p.abstract, "ID": p.id, "FilePath": actual_path 
                    })
            return res
        
        results = run_async(run_search())
        status.update(label="Done", state="complete")
        if results: st.session_state.search_results = results

    if "search_results" in st.session_state:
        results = st.session_state.search_results
        st.dataframe(pd.DataFrame(results)[["Title", "Authors", "PDF Status"]], width=2000)
        for item in results:
            with st.expander(f"📄 {item['Title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Authors:** {item['Authors']}")
                    st.info(item['Abstract'])
                with col2:
                    if gemini_active:
                        if st.button("✨ Elaborate", key=f"sum_{item['ID']}"):
                            with st.spinner("AI Thinking..."):
                                s = get_gemini_summary(item['ID'], item['Title'], active_model, item['Abstract'])
                                st.success(s)
                    if item.get("FilePath") and os.path.exists(item["FilePath"]):
                        with open(item["FilePath"], "rb") as f:
                            st.download_button("📥 Open PDF", f, file_name=f"{item['ID']}.pdf", mime="application/pdf")

# --- PAGE: Shodhganga ---
elif page == "Shodhganga Thesis":
    st.header("🇮🇳 Shodhganga Search")
    col1, col2 = st.columns([3, 1])
    with col1: q = st.text_input("Topic")
    with col2: u = st.text_input("University")
    
    if st.button("Search", type="primary"):
        status = st.status("Searching...", expanded=True)
        async def run_shodh():
            fldr = st.session_state.folder_mgr.create_search_folder('shodhganga', q)
            try: theses = await st.session_state.shodhganga.search_theses(q, u, 3)
            except: return []
            if not theses: return []
            
            processed = []
            async with aiohttp.ClientSession() as session:
                for t in theses:
                    status.write(f"Processing: {t.title[:40]}...")
                    tf = st.session_state.folder_mgr.create_thesis_folder(fldr, t.id, t.title)
                    await st.session_state.shodhganga.save_thesis_info(t, tf)
                    
                    full_text = ""
                    dl_status = "Metadata"
                    if t.chapter_links:
                        dl_text = []
                        for ch in t.chapter_links:
                            try:
                                url = ch['url'].replace(":8443","") if ":8443" in ch['url'] else ch['url']
                                async with session.get(url, ssl=False, timeout=30) as r:
                                    if r.status==200:
                                        d = await r.read()
                                        with open(tf/f"chap_{ch.get('chapter_number','x')}.pdf", 'wb') as f: f.write(d)
                                        import PyPDF2
                                        from io import BytesIO
                                        reader = PyPDF2.PdfReader(BytesIO(d))
                                        txt = "\n".join([p.extract_text() for p in reader.pages[:10]])
                                        dl_text.append(txt)
                            except: pass
                        if dl_text:
                            full_text = "\n\n".join(dl_text)
                            dl_status = f"✅ {len(dl_text)} Chaps"
                    
                    await st.session_state.vector_store.add_paper(t, full_text or t.abstract)
                    processed.append({
                        "Title": t.title, "ID": t.id, "Abstract": t.abstract,
                        "Links": t.chapter_links, "Status": dl_status,
                        "Researcher": ", ".join(t.authors), "University": t.metadata.get('university')
                    })
            return processed
        res = run_async(run_shodh())
        status.update(label="Done", state="complete")
        if res: st.session_state.shodh_res = res
        
    if "shodh_res" in st.session_state:
        df = pd.DataFrame(st.session_state.shodh_res)
        st.dataframe(df[["Title", "Researcher", "University", "Status"]], width=2000)
        for t in st.session_state.shodh_res:
            with st.expander(t['Title']):
                st.write(t['Abstract'])
                if gemini_active:
                    if st.button("✨ Elaborate Abstract", key=f"elab_{t['ID']}"):
                        s = get_gemini_summary(t['ID'], t['Title'], active_model, t['Abstract'])
                        st.write(s)
                    if t['Links']:
                        url = t['Links'][0]['url']
                        if st.button("🚀 Analyze Ch 1", key=f"ch_{t['ID']}"):
                            with st.spinner("Analyzing..."):
                                s = run_async(fetch_and_analyze_link(url, t['Title'], active_model))
                                st.write(s)

# --- PAGE: Chat (IMPROVED WITH GEMINI RAG) ---
elif page == "Chat with Papers":
    st.header("💬 Chat with Knowledge Base")
    
    # 1. Fetch available papers
    try: papers = st.session_state.vector_store.get_all_papers_metadata()
    except: papers = []
    
    # 2. Context Selector
    col1, col2 = st.columns([3, 1])
    with col1:
        sel = st.selectbox("Select Context", ["All Papers"] + [p['title'] for p in papers])
    with col2:
        st.metric("Indexed Papers", len(papers))
    
    # 3. Chat History
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: 
        with st.chat_message(m["role"]): st.markdown(m["content"])
        
    # 4. Chat Input
    if prompt := st.chat_input("Ask a question based on the papers..."):
        # Display User Message
        st.session_state.messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Generate Response
        with st.chat_message("assistant"):
            if not gemini_active:
                st.error("Please enter Gemini API Key in the sidebar.")
            else:
                with st.spinner("🔍 Retrieving & Thinking..."):
                    # Step A: Retrieve Context
                    fid = next((x['id'] for x in papers if x['title'] == sel), None) if sel != "All Papers" else None
                    filter_dict = {"paper_id": fid} if fid else None
                    
                    # Direct query to vector store for flexibility
                    # We retrieve 5 chunks of context
                    try:
                        embedding = st.session_state.vector_store.embedding_model.encode([prompt]).tolist()
                        results = st.session_state.vector_store.collection.query(
                            query_embeddings=embedding,
                            n_results=5,
                            where=filter_dict
                        )
                        
                        context_pieces = []
                        sources = []
                        if results['documents']:
                            for i, doc in enumerate(results['documents'][0]):
                                meta = results['metadatas'][0][i]
                                context_pieces.append(f"Source ({meta.get('title','')}): {doc}")
                                sources.append(meta.get('title', 'Unknown'))
                        
                        full_context = "\n\n".join(context_pieces)
                        
                    except Exception as e:
                        full_context = ""
                        st.error(f"Retrieval Error: {e}")

                    # Step B: Generate Answer with Gemini
                    if full_context:
                        model = genai.GenerativeModel(active_model)
                        system_prompt = f"""
                        You are a helpful research assistant. Answer the user's question based ONLY on the provided context below.
                        If the answer is not in the context, politely say you couldn't find it in the selected papers.
                        Cite the paper titles when possible.

                        CONTEXT:
                        {full_context}

                        USER QUESTION:
                        {prompt}
                        """
                        try:
                            response = model.generate_content(system_prompt)
                            answer = response.text
                        except Exception as e:
                            answer = f"Gemini Error: {e}"
                    else:
                        answer = "I couldn't find any relevant information in the database matching your query."

                    # Step C: Display
                    st.markdown(answer)
                    
                    # Show Sources
                    if sources:
                        with st.expander("📚 Sources / Retrieved Context"):
                            for s in set(sources):
                                st.markdown(f"- *{s}*")
                            
                    st.session_state.messages.append({"role":"assistant", "content":answer})

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