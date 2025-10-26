import asyncio
import logging
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from vector_store import VectorStore
from server import RAGEngine

logging.basicConfig(level=logging.INFO)

# Create FastMCP server
mcp = FastMCP("research-papers")

# Config (could be YAML-driven)
config = {
    "max_context_length": 4000,
    "max_retrieved_chunks": 5,
    "similarity_threshold": 0.15,
    "settings": {
        "persist_directory": "./data/vectors",
        "collection_name": "research_papers",
        "embedding_model": "all-mpnet-base-v2",
        "chunk_size": 1000,
        "chunk_overlap": 150
    }
}

# Initialize components
vector_store = VectorStore({"settings": config["settings"]})
rag_engine = RAGEngine(vector_store, config)

# ----------------------------
# Define MCP tools
# ----------------------------
@mcp.tool()
async def query_research_papers(question: str) -> CallToolResult:
    """Ask a question to the research paper database"""
    result = await rag_engine.query(question)
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence
                }, indent=2)
            )
        ]
    )

@mcp.tool()
def get_paper_summary(paper_id: str) -> CallToolResult:
    """Get a summary of a research paper by ID"""
    summary = rag_engine.get_paper_summary(paper_id)

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text="Paper not found" if not summary else json.dumps(summary, indent=2)
            )
        ]
    )


@mcp.tool()
async def ping() -> CallToolResult:
    """Health check tool"""
    return CallToolResult(
        content=[TextContent(type="text", text="pong 🏓 MCP server is alive!")]
    )

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    print("MCP server starting with tools: query_research_papers, get_paper_summary, ping")
    asyncio.run(mcp.run())
