import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent,
    CallToolRequest, ListResourcesRequest, ListToolsRequest
)

# Local imports
from paper_retriever import PaperRetrieverManager, ResearchPaper
from vector_store import VectorStore
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor
from shodhganga_retriever import ShodhgangaRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchMCPServer:
    """MCP Server for research paper retrieval and RAG"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.server = Server("research-paper-server")
        
        # Initialize components
        logger.info("Initializing MCP server components...")
        self.paper_manager = PaperRetrieverManager(self.config['apis'])
        self.vector_store = VectorStore(self.config['vector_db'])
        self.rag_engine = RAGEngine(self.vector_store, self.config['rag'])
        self.pdf_processor = PDFProcessor(self.config)
        
        # Initialize Shodhganga retriever if configured
        if 'shodhganga' in self.config['apis']:
            self.shodhganga = ShodhgangaRetriever(self.config['apis']['shodhganga'])
            logger.info("Shodhganga retriever initialized")
        else:
            self.shodhganga = None
        
        # Storage
        self.papers_dir = Path(self.config['storage']['papers_directory'])
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info("MCP server initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources (stored papers)"""
            resources = []
            
            # Add vector store statistics as a resource
            stats = self.vector_store.get_collection_stats()
            resources.append(Resource(
                uri="vector-store://stats",
                name="Vector Store Statistics",
                description=f"Statistics about the vector database ({stats['total_documents']} papers)",
                mimeType="application/json"
            ))
            
            # Add stored papers as resources
            for paper_file in self.papers_dir.glob("*.json"):
                resources.append(Resource(
                    uri=f"paper://{paper_file.stem}",
                    name=f"Paper: {paper_file.stem}",
                    description="Stored research paper metadata",
                    mimeType="application/json"
                ))
            
            return resources
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            tools = [
                Tool(
                    name="search_papers",
                    description="Search for research papers from arXiv, Semantic Scholar, and Shodhganga",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for papers"
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["arxiv", "semantic_scholar", "shodhganga"]},
                                "description": "Sources to search",
                                "default": ["arxiv"]
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 10,
                                "description": "Maximum number of results"
                            },
                            "store_locally": {
                                "type": "boolean",
                                "default": True,
                                "description": "Store papers in local vector database"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_shodhganga",
                    description="Search for Indian university theses from Shodhganga with full chapter download",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for theses"
                            },
                            "university": {
                                "type": "string",
                                "description": "Filter by university name (optional)"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 5,
                                "description": "Maximum number of results"
                            },
                            "download_full_thesis": {
                                "type": "boolean",
                                "default": True,
                                "description": "Download all chapters and combine into single document"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="ask_papers",
                    description="Ask questions about stored research papers using RAG",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the research papers"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 5,
                                "description": "Maximum number of relevant chunks to retrieve"
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="get_paper_summary",
                    description="Get a summary of a specific paper by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "ID of the paper to summarize"
                            }
                        },
                        "required": ["paper_id"]
                    }
                ),
                Tool(
                    name="list_stored_papers",
                    description="List all papers stored in the vector database",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls"""
            
            logger.info(f"Tool called: {name} with arguments: {arguments}")
            
            try:
                if name == "search_papers":
                    return await self._handle_search_papers(arguments)
                
                elif name == "search_shodhganga":
                    return await self._handle_search_shodhganga(arguments)
                
                elif name == "ask_papers":
                    return await self._handle_ask_papers(arguments)
                
                elif name == "get_paper_summary":
                    return await self._handle_get_paper_summary(arguments)
                
                elif name == "list_stored_papers":
                    return await self._handle_list_stored_papers(arguments)
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
    
    async def _handle_search_papers(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle paper search requests"""
        query = args["query"]
        sources = args.get("sources", ["arxiv"])
        max_results = args.get("max_results", 10)
        store_locally = args.get("store_locally", True)
        
        try:
            # Search for papers
            papers = await self.paper_manager.search_papers(
                query=query,
                sources=sources,
                max_results=max_results
            )
            
            results = []
            stored_count = 0
            
            for paper in papers:
                # Store locally if requested
                if store_locally:
                    try:
                        # Save paper metadata
                        paper_file = self.papers_dir / f"{paper.id}.json"
                        with open(paper_file, 'w') as f:
                            from dataclasses import asdict
                            json.dump(asdict(paper), f, indent=2)
                        
                        # Download PDF and extract text if available
                        full_text = None
                        if paper.pdf_url:
                            full_text = await self.pdf_processor.download_and_extract_pdf(paper)
                        
                        # Add to vector store
                        await self.vector_store.add_paper(paper, full_text)
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to store paper {paper.id}: {e}")
                
                # Prepare result
                paper_info = {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
                    "published_date": paper.published_date,
                    "source": paper.source,
                    "url": paper.url,
                    "categories": paper.categories,
                    "relevance_score": paper.metadata.get('relevance_score', 0) if paper.metadata else 0
                }
                results.append(paper_info)
            
            response = {
                "query": query,
                "sources_searched": sources,
                "total_found": len(papers),
                "stored_locally": stored_count if store_locally else 0,
                "papers": results
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return [TextContent(
                type="text", 
                text=f"Error searching papers: {str(e)}"
            )]
    
    async def _handle_search_shodhganga(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle Shodhganga thesis search"""
        if not self.shodhganga:
            return [TextContent(
                type="text",
                text="Shodhganga retriever is not configured. Please add shodhganga configuration to config.yaml"
            )]
        
        query = args["query"]
        university = args.get("university")
        max_results = args.get("max_results", 5)
        download_full = args.get("download_full_thesis", True)
        
        try:
            # Search Shodhganga
            theses = await self.shodhganga.search_theses(
                query=query,
                university=university,
                max_results=max_results
            )
            
            results = []
            
            for thesis in theses:
                thesis_info = {
                    "id": thesis.id,
                    "title": thesis.title,
                    "authors": thesis.authors,
                    "abstract": thesis.abstract[:300] + "..." if len(thesis.abstract) > 300 else thesis.abstract,
                    "university": thesis.metadata.get('university', 'Unknown') if thesis.metadata else 'Unknown',
                    "year": thesis.metadata.get('year', 'Unknown') if thesis.metadata else 'Unknown',
                    "url": thesis.url
                }
                
                # Download full thesis if requested
                if download_full:
                    try:
                        full_text = await self.shodhganga.download_full_thesis(thesis)
                        if full_text:
                            thesis_info["chapters_downloaded"] = thesis.metadata.get('chapters_count', 0) if thesis.metadata else 0
                            thesis_info["total_size"] = len(full_text)
                            
                            # Save and add to vector store
                            thesis_file = self.papers_dir / f"{thesis.id}.json"
                            with open(thesis_file, 'w') as f:
                                from dataclasses import asdict
                                json.dump(asdict(thesis), f, indent=2)
                            
                            await self.vector_store.add_paper(thesis, full_text)
                            thesis_info["stored"] = True
                        else:
                            thesis_info["stored"] = False
                            thesis_info["error"] = "Failed to download thesis"
                    except Exception as e:
                        logger.error(f"Error downloading thesis {thesis.id}: {e}")
                        thesis_info["stored"] = False
                        thesis_info["error"] = str(e)
                
                results.append(thesis_info)
            
            response = {
                "query": query,
                "university_filter": university,
                "total_found": len(theses),
                "theses": results
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error searching Shodhganga: {e}")
            return [TextContent(
                type="text",
                text=f"Error searching Shodhganga: {str(e)}"
            )]
    
    async def _handle_ask_papers(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle RAG queries about papers"""
        question = args["question"]
        max_results = args.get("max_results", 5)
        
        try:
            result = await self.rag_engine.query(
                question=question,
                max_results=max_results
            )
            
            response = {
                "question": result.query,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
                "context_length": len(result.context_used)
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return [TextContent(
                type="text",
                text=f"Error processing question: {str(e)}"
            )]
    
    async def _handle_get_paper_summary(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle paper summary requests"""
        paper_id = args["paper_id"]
        
        try:
            summary = await self.rag_engine.get_paper_summary(paper_id)
            
            if summary:
                return [TextContent(
                    type="text",
                    text=json.dumps(summary, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Paper {paper_id} not found in vector database"
                )]
                
        except Exception as e:
            logger.error(f"Error getting paper summary: {e}")
            return [TextContent(
                type="text",
                text=f"Error getting summary: {str(e)}"
            )]
    
    async def _handle_list_stored_papers(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle listing stored papers"""
        try:
            # Get papers from local storage
            papers = []
            for paper_file in self.papers_dir.glob("*.json"):
                with open(paper_file, 'r') as f:
                    paper_data = json.load(f)
                    papers.append({
                        "id": paper_data["id"],
                        "title": paper_data["title"],
                        "authors": paper_data["authors"][:3] if len(paper_data["authors"]) > 3 else paper_data["authors"],
                        "source": paper_data["source"],
                        "published_date": paper_data.get("published_date", "Unknown")
                    })
            
            # Get vector store stats
            stats = self.vector_store.get_collection_stats()
            
            response = {
                "vector_store_stats": stats,
                "stored_papers": len(papers),
                "papers": papers
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error listing papers: {e}")
            return [TextContent(
                type="text",
                text=f"Error listing papers: {str(e)}"
            )]

async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Research Paper MCP Server...")
    
    try:
        server_instance = ResearchMCPServer()
        
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server running on stdio")
            await server_instance.server.run(
                read_stream,
                write_stream,
                server_instance.server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())