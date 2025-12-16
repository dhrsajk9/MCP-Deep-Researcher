# 🧬 Deep Researcher

Deep Researcher is a powerful tool designed to streamline academic research. It allows users to search for papers from **ArXiv**, **Semantic Scholar**, and **Shodhganga** (Indian Theses), download them locally, index them into a Vector Database, and interact with them using **Google Gemini AI** via a RAG (Retrieval-Augmented Generation) pipeline.

---

## 🚀 Features

* **Multi-Source Search:** Fetch papers from ArXiv, Semantic Scholar, and Shodhganga.
* **Local Storage:** Papers (PDFs and Metadata) are stored locally in an organized folder structure.
* **Vector Search:** Uses ChromaDB to index paper content for semantic search.
* **AI Chat (RAG):** Chat with your library or specific papers using Gemini 1.5 Flash/Pro.
* **Dual Interface:** Includes both a **Command Line Interface (CLI)** and a modern **Streamlit Web UI**.
* **PDF Analysis:** Automatic text extraction and summarizing capabilities.

---

## 🛠️ Installation & Setup

Follow these steps to set up the project on your local machine.

### Step 1: Create and Activate Virtual Environment

It is recommended to run this project in an isolated virtual environment.

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\activate


Mac / Linux

python3 -m venv .venv
source .venv/bin/activate

Step 2: Install Dependencies

Install the project and required dependencies:

pip install -e .
pip install -r requirements.txt

Step 3: Create Data Directories

These folders store downloaded papers and vector embeddings.

Windows (PowerShell)
mkdir .\data\papers -Force
mkdir .\data\vectors -Force

Mac / Linux
mkdir -p ./data/papers
mkdir -p ./data/vectors

⚙️ Configuration (Gemini API Key)

To enable AI features, you need a Google Gemini API key.

1️⃣ Get API Key

Visit: https://aistudio.google.com/

2️⃣ CLI Configuration

Create a .env file in the project root:

GEMINI_API_KEY=your_actual_api_key_here

3️⃣ Streamlit Configuration

Enter the API key directly in the sidebar of the web app.

💻 Usage: Command Line Interface (CLI)

Use the CLI for fast searches and batch processing.

1️⃣ Search & Download Papers (ArXiv)
py .\src\cli.py search -q "transformers attention" -s arxiv -n 5

2️⃣ Search Shodhganga Theses
py .\src\cli.py search_shodhganga -q "computer vision" -u "Indian Institute of Technology" -n 3

3️⃣ Ask Questions (RAG Chat)
py .\src\cli.py ask -q "What is self-attention and why is it useful?"

🖥️ Usage: Streamlit Web Interface

For a visual dashboard with PDF viewing and interactive chat.

Run the App
streamlit run .\src\app.py


The app will open at:

http://localhost:8501
