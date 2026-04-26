import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from rag_engine import RAGEngine
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("research-papers")

_vector_store: Optional[VectorStore] = None
_rag_engine: Optional[RAGEngine] = None
_init_error: Optional[str] = None
_project_root = Path(__file__).resolve().parent.parent


def _load_config() -> Dict[str, Any]:
    """Load config/config.yaml with safe fallback defaults."""
    config_path = _project_root / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Normalize path-like settings relative to the project root.
        vector_settings = data.get("vector_db", {}).get("settings", {})
        persist_dir = vector_settings.get("persist_directory")
        if persist_dir and not Path(persist_dir).is_absolute():
            vector_settings["persist_directory"] = str(_project_root / persist_dir)
        return data

    return {
        "vector_db": {
            "settings": {
                "persist_directory": str(_project_root / "data" / "vectors"),
                "collection_name": "research_papers",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 150,
            }
        },
        "rag": {
            "max_context_length": 8000,
            "max_retrieved_chunks": 12,
            "similarity_threshold": 0.05,
        },
    }


def _ensure_engine() -> Optional[str]:
    """
    Lazy-initialize heavy components.
    This keeps MCP startup fast so clients can connect without timing out.
    """
    global _vector_store, _rag_engine, _init_error

    if _rag_engine is not None:
        return None

    if _init_error is not None:
        return _init_error

    try:
        cfg = _load_config()
        vector_cfg = cfg.get("vector_db", {})
        rag_cfg = cfg.get("rag", {})

        _vector_store = VectorStore(vector_cfg)
        _rag_engine = RAGEngine(_vector_store, rag_cfg)
        logger.info("RAG engine initialized successfully")
        return None
    except Exception as e:
        _init_error = (
            "Server initialized, but RAG backend failed to load. "
            f"Details: {e}"
        )
        logger.exception("Failed to initialize RAG backend")
        return _init_error

# ----------------------------
# Define MCP tools
# ----------------------------
@mcp.tool()
async def query_research_papers(question: str) -> CallToolResult:
    """Ask a question to the research paper database"""
    init_error = _ensure_engine()
    if init_error:
        return CallToolResult(
            content=[TextContent(type="text", text=init_error)]
        )

    result = await _rag_engine.query(question)
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
async def get_paper_summary(paper_id: str) -> CallToolResult:
    """Get a summary of a research paper by ID"""
    init_error = _ensure_engine()
    if init_error:
        return CallToolResult(
            content=[TextContent(type="text", text=init_error)]
        )

    summary = _rag_engine.get_paper_summary(paper_id)

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
        content=[TextContent(type="text", text="pong MCP server is alive")]
    )


@mcp.tool()
async def server_status() -> CallToolResult:
    """Show whether RAG backend is ready."""
    if _rag_engine is not None:
        status = {"server": "ready", "rag_backend": "initialized"}
    elif _init_error:
        status = {"server": "ready", "rag_backend": "error", "details": _init_error}
    else:
        status = {"server": "ready", "rag_backend": "lazy_not_initialized"}

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(status, indent=2))]
    )

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    logger.info(
        "MCP server starting with tools: "
        "query_research_papers, get_paper_summary, ping, server_status"
    )
    asyncio.run(mcp.run())
