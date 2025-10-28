import click
import asyncio
import json
import yaml
from pathlib import Path
from paper_retriever import PaperRetrieverManager
from vector_store import VectorStore
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor
from shodhganga_retriever import ShodhgangaRetriever
from folder_manager import FolderManager

@click.group()
def cli():
    """Enhanced Research Paper MCP Server CLI with organized folder structure"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--source', '-s', multiple=True, default=['arxiv'], help='Sources to search (arxiv, semantic_scholar, shodhganga)')
@click.option('--max-results', '-n', default=10, help='Maximum results')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def search(query, source, max_results, config):
    """Search for research papers - creates organized folders per search"""
    asyncio.run(_search_papers(query, list(source), max_results, config))

@cli.command()
@click.option('--query', '-q', required=True, help='Search query for theses')
@click.option('--university', '-u', help='Filter by university name')
@click.option('--max-results', '-n', default=5, help='Maximum results')
@click.option('--download-full', '-d', is_flag=True, default=True, help='Download all chapters')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def search_shodhganga(query, university, max_results, download_full, config):
    """Search Shodhganga - each thesis gets its own folder with separate chapters"""
    asyncio.run(_search_shodhganga_theses(query, university, max_results, download_full, config))

@cli.command()
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def ask(question, config):
    """Ask questions about all stored papers (searches entire main folder)"""
    asyncio.run(_ask_papers(question, config))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def stats(config):
    """Show database statistics"""
    asyncio.run(_show_stats(config))

async def _search_papers(query, sources, max_results, config_path):
    """Search for papers with organized folder structure"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize folder manager
    folder_mgr = FolderManager(config['storage']['papers_directory'])
    
    # Process each source separately
    for source in sources:
        click.echo(f"\n{'='*80}")
        click.echo(f"Searching {source.upper()} for: '{query}'")
        click.echo(f"{'='*80}")
        
        # Create search folder for this query and source
        search_folder = folder_mgr.create_search_folder(source, query)
        click.echo(f"📁 Created search folder: {search_folder}")
        
        # Initialize components
        manager = PaperRetrieverManager(config['apis'], folder_mgr)
        vector_store = VectorStore(config['vector_db'])
        pdf_processor = PDFProcessor(config)
        
        # Search papers
        papers = await manager.search_papers(query, [source], max_results)
        
        if not papers:
            click.echo(f"❌ No papers found from {source}")
            continue
        
        click.echo(f"\n✅ Found {len(papers)} papers from {source}:")
        
        for i, paper in enumerate(papers, 1):
            click.echo(f"\n{i}. 📄 {paper.title}")
            click.echo(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            click.echo(f"   URL: {paper.url}")
            
            # Show relevance if available
            if paper.metadata:
                relevance = paper.metadata.get('relevance_score', 0)
                strategy = paper.metadata.get('strategy', 'Unknown')
                click.echo(f"   Relevance: {relevance:.1f} (via {strategy})")
            
            # Save to search folder
            paper_file = search_folder / f"{paper.id}.json"
            with open(paper_file, 'w', encoding='utf-8') as f:
                from dataclasses import asdict
                json.dump(asdict(paper), f, indent=2, ensure_ascii=False)
            
            # Download PDF
            full_text = None
            if paper.pdf_url:
                click.echo("   📥 Downloading PDF...")
                try:
                    # Save PDF to search folder
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(paper.pdf_url, timeout=60) as response:
                            if response.status == 200:
                                pdf_content = await response.read()
                                pdf_file = search_folder / f"{paper.id}.pdf"
                                with open(pdf_file, 'wb') as f:
                                    f.write(pdf_content)
                                click.echo(f"   ✅ PDF saved: {pdf_file.name}")
                                
                                # Extract text
                                import PyPDF2
                                from io import BytesIO
                                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                                text_parts = []
                                for page in pdf_reader.pages:
                                    text_parts.append(page.extract_text())
                                full_text = "\n".join(text_parts)
                                
                                # Save extracted text
                                text_file = search_folder / f"{paper.id}.txt"
                                with open(text_file, 'w', encoding='utf-8') as f:
                                    f.write(full_text)
                                click.echo(f"   📝 Text extracted: {len(full_text)} chars")
                            
                except Exception as e:
                    click.echo(f"   ⚠️ PDF error: {e}")
            
            # Add to vector store
            await vector_store.add_paper(paper, full_text)
            click.echo("   ✅ Added to vector database")
        
        click.echo(f"\n📂 All {source} papers saved to: {search_folder}")

async def _search_shodhganga_theses(query, university, max_results, download_full, config_path):
    """Search Shodhganga with web scraping and save detailed notes"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'shodhganga' not in config['apis']:
        click.echo("❌ Shodhganga is not configured in config.yaml")
        return
    
    # Initialize folder manager
    folder_mgr = FolderManager(config['storage']['papers_directory'])
    
    # Create search folder for this query
    search_folder = folder_mgr.create_search_folder('shodhganga', query)
    
    click.echo(f"\n{'='*80}")
    click.echo(f"🔍 Searching Shodhganga for: '{query}'")
    if university:
        click.echo(f"   University filter: {university}")
    click.echo(f"📁 Search folder: {search_folder}")
    click.echo(f"{'='*80}\n")
    
    shodhganga = ShodhgangaRetriever(config['apis']['shodhganga'])
    vector_store = VectorStore(config['vector_db'])
    
    # Search (web scraping)
    click.echo("🌐 Scraping Shodhganga website (respectfully)...")
    theses = await shodhganga.search_theses(query, university, max_results)
    
    if not theses:
        click.echo("❌ No theses found")
        return
    
    click.echo(f"✅ Found {len(theses)} theses\n")
    
    # Create a summary file for the search
    summary_lines = [
        f"# Shodhganga Search Results",
        f"",
        f"**Query**: {query}",
        f"**University Filter**: {university or 'None'}",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Results Found**: {len(theses)}",
        f"",
        f"---",
        f""
    ]
    
    for i, thesis in enumerate(theses, 1):
        click.echo(f"{i}. 📖 {thesis.title}")
        click.echo(f"   👥 Authors: {', '.join(thesis.authors)}")
        
        if thesis.metadata:
            click.echo(f"   🏛️  University: {thesis.metadata.get('university', 'Unknown')}")
            click.echo(f"   📅 Year: {thesis.metadata.get('year', 'Unknown')}")
            click.echo(f"   🎓 Degree: {thesis.metadata.get('degree', 'Unknown')}")
            click.echo(f"   📚 Guide: {thesis.metadata.get('guide', 'Unknown')}")
            click.echo(f"   📑 Chapters found: {thesis.metadata.get('chapters_count', 0)}")
        
        click.echo(f"   🔗 URL: {thesis.url}")
        
        # Create thesis folder
        thesis_folder = folder_mgr.create_thesis_folder(search_folder, thesis.id, thesis.title)
        click.echo(f"   📁 Folder: {thesis_folder.name}")
        
        # Save comprehensive thesis information
        click.echo("   💾 Saving thesis information...")
        note_file = await shodhganga.save_thesis_info(thesis, thesis_folder)
        click.echo(f"   ✅ Saved detailed note: THESIS_INFO.md")
        click.echo(f"   ✅ Saved metadata: metadata.json")
        
        if thesis.chapter_links:
            click.echo(f"   ✅ Saved {len(thesis.chapter_links)} chapter links: chapter_links.json")
            click.echo(f"   📄 Chapters available:")
            for j, chapter in enumerate(thesis.chapter_links[:5], 1):  # Show first 5
                click.echo(f"      {j}. {chapter['name']}")
            if len(thesis.chapter_links) > 5:
                click.echo(f"      ... and {len(thesis.chapter_links) - 5} more")
        
        # Add to summary
        summary_lines.extend([
            f"## {i}. {thesis.title}",
            f"",
            f"- **Authors**: {', '.join(thesis.authors)}",
            f"- **University**: {thesis.metadata.get('university', 'Unknown') if thesis.metadata else 'Unknown'}",
            f"- **Year**: {thesis.published_date}",
            f"- **Chapters**: {len(thesis.chapter_links) if thesis.chapter_links else 0}",
            f"- **Folder**: `{thesis_folder.name}`",
            f"- **URL**: {thesis.url}",
            f"",
        ])
        
        # Add to vector store (with abstract and metadata)
        try:
            # Create searchable text from metadata
            searchable_text = f"""
Title: {thesis.title}

Authors: {', '.join(thesis.authors)}

University: {thesis.metadata.get('university', 'Unknown') if thesis.metadata else 'Unknown'}
Department: {thesis.metadata.get('department', 'Unknown') if thesis.metadata else 'Unknown'}
Year: {thesis.published_date}
Degree: {thesis.metadata.get('degree', 'Unknown') if thesis.metadata else 'Unknown'}

Abstract:
{thesis.abstract}

Subjects: {', '.join(thesis.metadata.get('subject', [])) if thesis.metadata and thesis.metadata.get('subject') else 'Not specified'}
"""
            
            await vector_store.add_paper(thesis, searchable_text)
            click.echo("   ✅ Added to vector database (searchable)")
        except Exception as e:
            click.echo(f"   ⚠️  Failed to add to vector DB: {e}")
        
        click.echo()
    
    # Save search summary
    summary_file = search_folder / "SEARCH_SUMMARY.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    click.echo(f"📂 All theses saved to: {search_folder}")
    click.echo(f"📋 Search summary: SEARCH_SUMMARY.md")
    click.echo(f"\n💡 Tip: Check THESIS_INFO.md in each thesis folder for complete details and chapter links!")

# Add datetime import at top
from datetime import datetime

async def _ask_papers(question, config_path):
    """Ask papers - searches entire main folder"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_store = VectorStore(config['vector_db'])
    rag_engine = RAGEngine(vector_store, config['rag'])
    
    click.echo(f"\n🔍 Searching all papers for: '{question}'")
    click.echo(f"📂 Searching in: {config['storage']['papers_directory']}\n")
    
    result = await rag_engine.query(question)
    
    click.echo(f"❓ Question: {result.query}")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}\n")
    click.echo(f"💡 Answer:\n{result.answer}\n")
    
    if result.sources:
        click.echo(f"📚 Sources ({len(result.sources)}):")
        for i, source in enumerate(result.sources[:5], 1):
            authors = source['authors'][:2] if source['authors'] else ['Unknown']
            authors_str = ', '.join(authors)
            click.echo(f"   {i}. {source['title'][:60]}...")
            click.echo(f"      Authors: {authors_str}, Similarity: {source['similarity']:.3f}")

async def _show_stats(config_path):
    """Show statistics"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    folder_mgr = FolderManager(config['storage']['papers_directory'])
    vector_store = VectorStore(config['vector_db'])
    stats = vector_store.get_collection_stats()
    
    click.echo("📊 Database Statistics:\n")
    click.echo("Vector Store:")
    click.echo(f"   Total documents: {stats['total_documents']}")
    click.echo(f"   Collection: {stats['collection_name']}")
    click.echo(f"   Embedding model: {stats['embedding_model']}\n")
    
    # Count papers by source
    click.echo("Papers by Source:")
    for source in ['arxiv', 'shodhganga', 'semantic_scholar']:
        search_folders = folder_mgr.get_search_folders(source)
        total_papers = 0
        for folder in search_folders:
            if source == 'shodhganga':
                # Count thesis folders
                thesis_folders = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith('thesis_')]
                total_papers += len(thesis_folders)
            else:
                # Count JSON files
                json_files = list(folder.glob("*.json"))
                total_papers += len(json_files)
        
        click.echo(f"   {source}: {total_papers} papers in {len(search_folders)} searches")
    
    # Recent searches
    click.echo("\nRecent Searches:")
    all_searches = folder_mgr.get_search_folders()
    all_searches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for folder in all_searches[:5]:
        source = folder.parent.name
        click.echo(f"   📁 {folder.name} ({source})")

if __name__ == '__main__':
    cli()

@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--source', '-s', multiple=True, default=['arxiv'], help='Sources to search (arxiv, semantic_scholar, shodhganga)')
@click.option('--max-results', '-n', default=10, help='Maximum results')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def search(query, source, max_results, config):
    """Search for research papers from multiple sources"""
    asyncio.run(_search_papers(query, list(source), max_results, config))

@cli.command()
@click.option('--query', '-q', required=True, help='Search query for theses')
@click.option('--university', '-u', help='Filter by university name')
@click.option('--max-results', '-n', default=5, help='Maximum results')
@click.option('--download-full', '-d', is_flag=True, default=True, help='Download all chapters')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def search_shodhganga(query, university, max_results, download_full, config):
    """Search for Indian university theses from Shodhganga"""
    asyncio.run(_search_shodhganga_theses(query, university, max_results, download_full, config))

@cli.command()
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def ask(question, config):
    """Ask questions about stored papers"""
    asyncio.run(_ask_papers(question, config))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def stats(config):
    """Show database statistics"""
    asyncio.run(_show_stats(config))

async def _search_papers(query, sources, max_results, config_path):
    """Search for papers implementation"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    manager = PaperRetrieverManager(config['apis'])
    vector_store = VectorStore(config['vector_db'])
    pdf_processor = PDFProcessor(config)
    
    papers = await manager.search_papers(query, sources, max_results)
    
    click.echo(f"Found {len(papers)} papers:")
    
    for paper in papers:
        click.echo(f"\n📄 {paper.title}")
        click.echo(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        click.echo(f"   Source: {paper.source}")
        click.echo(f"   URL: {paper.url}")
        
        # Show relevance and strategy if available
        if paper.metadata:
            relevance = paper.metadata.get('relevance_score', 0)
            strategy = paper.metadata.get('strategy', 'Unknown')
            click.echo(f"   Relevance: {relevance:.1f} (via {strategy})")
        
        # Download PDF and extract text if available
        full_text = None
        if paper.pdf_url:
            click.echo("   📥 Downloading PDF...")
            try:
                full_text = await pdf_processor.download_and_extract_pdf(paper)
                if full_text:
                    click.echo(f"   ✅ PDF processed ({len(full_text)} chars)")
                else:
                    click.echo("   ⚠️ PDF processing failed, using abstract only")
            except Exception as e:
                click.echo(f"   ⚠️ PDF error: {e}")
        
        # Add to vector store
        await vector_store.add_paper(paper, full_text)
        click.echo("   ✅ Added to vector database")

async def _search_shodhganga_theses(query, university, max_results, download_full, config_path):
    """Search Shodhganga theses implementation"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'shodhganga' not in config['apis']:
        click.echo("❌ Shodhganga is not configured in config.yaml")
        return
    
    click.echo(f"🔍 Searching Shodhganga for: {query}")
    if university:
        click.echo(f"   University filter: {university}")
    
    shodhganga = ShodhgangaRetriever(config['apis']['shodhganga'])
    vector_store = VectorStore(config['vector_db'])
    
    theses = await shodhganga.search_theses(query, university, max_results)
    
    if not theses:
        click.echo("❌ No theses found")
        return
    
    click.echo(f"\n📚 Found {len(theses)} theses:")
    
    for i, thesis in enumerate(theses, 1):
        click.echo(f"\n{i}. 📖 {thesis.title}")
        click.echo(f"   👥 Authors: {', '.join(thesis.authors)}")
        
        if thesis.metadata:
            click.echo(f"   🏛️ University: {thesis.metadata.get('university', 'Unknown')}")
            click.echo(f"   📅 Year: {thesis.metadata.get('year', 'Unknown')}")
            click.echo(f"   🎓 Degree: {thesis.metadata.get('degree', 'Unknown')}")
            click.echo(f"   📑 Chapters found: {thesis.metadata.get('chapters_count', 0)}")
        
        click.echo(f"   🔗 URL: {thesis.url}")
        
        # Download full thesis if requested
        if download_full and thesis.pdf_urls:
            click.echo(f"   📥 Downloading {len(thesis.pdf_urls)} chapters...")
            try:
                full_text = await shodhganga.download_full_thesis(thesis)
                if full_text:
                    click.echo(f"   ✅ Downloaded and combined {len(thesis.pdf_urls)} chapters ({len(full_text)} chars)")
                    
                    # Add to vector store
                    await vector_store.add_paper(thesis, full_text)
                    click.echo("   ✅ Added to vector database")
                else:
                    click.echo("   ⚠️ Failed to download thesis")
            except Exception as e:
                click.echo(f"   ❌ Error: {e}")
        else:
            # Add with abstract only
            await vector_store.add_paper(thesis, None)
            click.echo("   ✅ Added metadata to vector database")

async def _ask_papers(question, config_path):
    """Ask papers implementation"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_store = VectorStore(config['vector_db'])
    rag_engine = RAGEngine(vector_store, config['rag'])
    
    result = await rag_engine.query(question)
    
    click.echo(f"\n❓ Question: {result.query}")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}")
    click.echo(f"\n💡 Answer:\n{result.answer}")
    
    if result.sources:
        click.echo(f"\n📚 Sources ({len(result.sources)}):")
        for i, source in enumerate(result.sources[:5], 1):
            authors = source['authors'][:2] if source['authors'] else ['Unknown']
            authors_str = ', '.join(authors)
            click.echo(f"   {i}. {source['title'][:60]}...")
            click.echo(f"      Authors: {authors_str}, Similarity: {source['similarity']:.3f}")

async def _show_stats(config_path):
    """Show statistics implementation"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_store = VectorStore(config['vector_db'])
    stats = vector_store.get_collection_stats()
    
    click.echo("📊 Vector Database Statistics:")
    click.echo(f"   Total documents: {stats['total_documents']}")
    click.echo(f"   Collection name: {stats['collection_name']}")
    click.echo(f"   Embedding model: {stats['embedding_model']}")
    
    # Show stored papers by source
    papers_dir = Path(config['storage']['papers_directory'])
    if papers_dir.exists():
        paper_files = list(papers_dir.glob("**/*.json"))
        
        sources = {}
        for paper_file in paper_files:
            try:
                with open(paper_file, 'r') as f:
                    paper_data = json.load(f)
                    source = paper_data.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            except:
                continue
        
        click.echo(f"   Stored papers: {len(paper_files)}")
        if sources:
            click.echo("   By source:")
            for source, count in sources.items():
                click.echo(f"      {source}: {count}")

if __name__ == '__main__':
    cli()