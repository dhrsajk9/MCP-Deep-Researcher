# Deep Researcher (MCP + CLI + Streamlit)

Deep Researcher lets you search research papers from:
- arXiv
- Semantic Scholar
- Shodhganga (Indian theses)

It can download papers locally, extract text, index content in a vector DB (Chroma), and answer questions with RAG.

This repo supports:
- an MCP server (for Claude Desktop and other MCP clients),
- a CLI,
- and a Streamlit app.

## Features
- Multi-source paper search.
- Local storage in organized folders under `data/papers`.
- PDF + metadata + extracted text persistence.
- Vector indexing in `data/vectors`.
- RAG Q&A over indexed content.

## Repository Layout
- `src/server.py`: Primary MCP server (search + storage + RAG tools).
- `src/mcp_server.py`: Lightweight MCP server (query-focused tools).
- `src/cli.py`: CLI interface.
- `src/app.py`: Streamlit UI.
- `config/config.yaml`: Main configuration.

## Prerequisites
- Python 3.10+ recommended.
- Git.

## Quick Start

### 1. Clone
```bash
git clone <https://github.com/dhrsajk9/MCP-Deep-Researcher>
cd MCP-Deep-Researcher
```

### 2. Create virtual environment
Windows (PowerShell):
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Create data directories
```bash
mkdir -p data/papers data/vectors
```
On Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force data\papers, data\vectors
```

## Running as MCP Server

Use `src/server.py` for full functionality (search + local storage + RAG).

### Claude Desktop Configuration
Update `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-papers": {
      "command": "C:\\path\\to\\python.exe",
      "args": [
        "-u",
        "C:\\absolute\\path\\to\\MCP-Deep-Researcher\\src\\server.py"
      ],
      "cwd": "C:\\absolute\\path\\to\\MCP-Deep-Researcher",
      "env": {
        "PYTHONPATH": "C:\\absolute\\path\\to\\MCP-Deep-Researcher\\src",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

Important:
- Use absolute paths for `command`, `args`, and `cwd`.
- Set `cwd` to project root. Without this, files can be written to the MCP client's app directory instead of this repo.
- Restart Claude Desktop fully after changing config.

### Generic MCP Client Configuration
For any MCP client that supports stdio transport:
- command: Python executable
- args: `["-u", "<absolute path>/src/server.py"]`
- cwd: project root (`<absolute path>/MCP-Deep-Researcher`)
- env: `PYTHONPATH=<absolute path>/src`

If your client supports tool discovery, you should see tools such as:
- `search_papers`
- `search_shodhganga`
- `ask_papers`
- `get_paper_summary`
- `list_stored_papers`

## Local Storage Behavior

Downloaded/search results are saved under:
- `data/papers/<source>/search_<query>_<timestamp>/`

Per paper, expected files are:
- `<paper_id>.json` (metadata)
- `<paper_id>.pdf` (downloaded PDF when available)
- `<paper_id>.txt` (extracted text, or metadata fallback when extraction is empty)

## CLI Usage

Search arXiv papers:
```bash
python src/cli.py search -q "transformers attention" -s arxiv -n 5
```

Search Shodhganga theses:
```bash
python src/cli.py search_shodhganga -q "computer vision" -u "Indian Institute of Technology" -n 3
```

Ask a question over indexed papers:
```bash
python src/cli.py ask -q "What is self-attention and why is it useful?"
```

## Streamlit Usage
```bash
streamlit run src/app.py
```
Then open:
- [http://localhost:8501](http://localhost:8501)

## Optional: Gemini API Key
Needed for some Streamlit AI features.

Create `.env` in project root:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

You can also enter the key from the Streamlit sidebar.

## Quick Verify (2-5 minutes)

Use this checklist right after setup to confirm everything works.

### 1. Verify Python and dependencies
```bash
python -c "import mcp, chromadb, sentence_transformers, PyPDF2; print('ok')"
```
Expected output:
```text
ok
```

### 2. Verify config paths resolve in project
```bash
python -c "from src.server import ResearchMCPServer; s=ResearchMCPServer(); print(s.papers_dir)"
```
Expected output ends with your repo path, for example:
```text
.../MCP-Deep-Researcher/data/papers
```

### 3. Verify MCP client sees tools
After starting from your MCP client, confirm tool list includes:
- `search_papers`
- `ask_papers`
- `list_stored_papers`

If these are missing, check MCP config points to `src/server.py` and restart the client.

### 4. Verify local paper storage
Run one search from MCP client:
- Query: `transformers`
- Sources: `["arxiv"]`
- Max results: `1`

Then check that a folder was created under:
- `data/papers/arxiv/search_<query>_<timestamp>/`

And inside it you should see:
- `<paper_id>.json`
- `<paper_id>.pdf` (if URL is available)
- `<paper_id>.txt`

## Troubleshooting

### MCP connected but papers do not appear in `data/papers`
- Ensure MCP config uses `src/server.py` (not only `src/mcp_server.py`).
- Ensure `cwd` is set to project root.
- Ensure absolute paths are used in config.

### MCP error: "Unexpected token ... is not valid JSON"
- This usually means non-protocol stdout output polluted the MCP stream.
- Use `-u` and keep runtime logs on stderr (current repo code already does this for core paths).
- Restart the client after updating.

### Python ENOENT / cannot spawn interpreter
- Verify configured Python path exists.
- If using `.venv`, ensure environment has been created first.
