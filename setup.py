from setuptools import setup, find_packages

setup(
    name="research-mcp-server",
    version="1.0.0",
    description="MCP Server for research paper retrieval and RAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mcp==1.14.1",
        "arxiv==2.2.0",
        "chromadb==1.1.0",
        "sentence-transformers==5.1.1",
        "torch==2.8.0",
        "PyPDF2==3.0.1",
        "click==8.3.0",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
        "requests==2.32.5"
    ],
    entry_points={
        "console_scripts": [
            "research-mcp=cli:cli",
        ],
    },
    python_requires=">=3.8",
)