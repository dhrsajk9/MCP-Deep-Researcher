import click
import asyncio
import json
import yaml
from pathlib import Path
from paper_retriever import PaperRetrieverManager
from vector_store import VectorStore
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor

@click.group()
def cli():
    """Research Paper MCP Server CLI"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--source', '-s', multiple=True, default=['arxiv'], help='Sources to search')
@click.option('--max-results', '-n', default=10, help='Maximum results')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def search(query, source, max_results, config):
    """Search for research papers"""
    asyncio.run(_search_papers(query, list(source), max_results, config))

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
        
        # Download PDF and extract text if available
        full_text = None
        if paper.pdf_url:
            click.echo("   📥 Downloading PDF...")
            full_text = await pdf_processor.download_and_extract_pdf(paper)
        
        # Add to vector store
        await vector_store.add_paper(paper, full_text)
        click.echo("   ✅ Added to vector database")

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
        for i, source in enumerate(result.sources[:3], 1):
            click.echo(f"   {i}. {source['title']} (similarity: {source['similarity']:.2f})")

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

if __name__ == '__main__':
    cli()