"""
Folder structure manager for organized paper storage

Structure:
data/
├── papers/
│   ├── arxiv/
│   │   ├── search_neural_networks_20250126_143022/
│   │   │   ├── paper1.json
│   │   │   ├── paper1.pdf
│   │   │   ├── paper2.json
│   │   │   └── paper2.pdf
│   │   └── search_deep_learning_20250126_150000/
│   │       └── ...
│   ├── shodhganga/
│   │   ├── search_machine_learning_20250126_143022/
│   │   │   ├── thesis_12345/
│   │   │   │   ├── metadata.json
│   │   │   │   ├── chapter_01.pdf
│   │   │   │   ├── chapter_02.pdf
│   │   │   │   └── ...
│   │   │   └── thesis_67890/
│   │   │       └── ...
│   │   └── search_deep_learning_20250126_150000/
│   │       └── ...
│   └── semantic_scholar/
│       └── ...
└── vectors/
    └── [vector database files]
"""

from pathlib import Path
from datetime import datetime
import re
from typing import Optional
import os

class FolderManager:
    """Manages organized folder structure for papers"""
    
    def __init__(self, base_dir: str = "./data/papers"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create source directories
        self.arxiv_dir = self.base_dir / "arxiv"
        self.shodhganga_dir = self.base_dir / "shodhganga"
        self.semantic_scholar_dir = self.base_dir / "semantic_scholar"
        
        for dir_path in [self.arxiv_dir, self.shodhganga_dir, self.semantic_scholar_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def create_search_folder(self, source: str, query: str) -> Path:
        """Create a timestamped search folder for a query"""
        
        # Get source directory
        if source == "arxiv":
            source_dir = self.arxiv_dir
        elif source == "shodhganga":
            source_dir = self.shodhganga_dir
        elif source == "semantic_scholar":
            source_dir = self.semantic_scholar_dir
        else:
            source_dir = self.base_dir / source
            source_dir.mkdir(exist_ok=True)
        
        # Create safe folder name from query
        safe_query = self._sanitize_folder_name(query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"search_{safe_query}_{timestamp}"
        
        # Create folder
        search_folder = source_dir / folder_name
        search_folder.mkdir(parents=True, exist_ok=True)
        
        return search_folder
    
    def create_thesis_folder(self, search_folder: Path, thesis_id: str, thesis_title: str) -> Path:
        """Create a folder for a specific thesis within a search folder"""
        
        # Create safe folder name
        safe_title = self._sanitize_folder_name(thesis_title)[:50]  # Limit length
        folder_name = f"thesis_{thesis_id}_{safe_title}"
        
        thesis_folder = search_folder / folder_name
        thesis_folder.mkdir(parents=True, exist_ok=True)
        
        return thesis_folder
    
    def _sanitize_folder_name(self, name: str) -> str:
        """Convert query/title to safe folder name"""
        # Remove special characters, keep alphanumeric and spaces
        safe_name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        safe_name = re.sub(r'[\s]+', '_', safe_name)
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        # Trim
        safe_name = safe_name.strip('_')
        # Limit length
        return safe_name[:50]
    
    def get_all_paper_files(self) -> list:
        """Recursively get all paper JSON files for RAG"""
        paper_files = []
        
        # Search all subdirectories
        for json_file in self.base_dir.rglob("*.json"):
            # Skip if it's not a metadata or paper file
            if json_file.name in ['metadata.json', 'config.json']:
                paper_files.append(json_file)
            elif not json_file.name.startswith('.'):
                paper_files.append(json_file)
        
        return paper_files
    
    def get_all_pdf_files(self) -> list:
        """Recursively get all PDF files"""
        return list(self.base_dir.rglob("*.pdf"))
    
    def get_search_folders(self, source: Optional[str] = None) -> list:
        """Get all search folders, optionally filtered by source"""
        
        if source:
            if source == "arxiv":
                base = self.arxiv_dir
            elif source == "shodhganga":
                base = self.shodhganga_dir
            elif source == "semantic_scholar":
                base = self.semantic_scholar_dir
            else:
                return []
            
            return [d for d in base.iterdir() if d.is_dir() and d.name.startswith('search_')]
        else:
            # Get from all sources
            all_folders = []
            for source_dir in [self.arxiv_dir, self.shodhganga_dir, self.semantic_scholar_dir]:
                all_folders.extend([d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith('search_')])
            return all_folders
    def delete_paper_files(self, paper_id: str):
        """Find and delete physical files associated with a paper ID"""
        deleted = False
        # 1. Search for simple paper files (JSON/PDF)
        for path in self.base_dir.rglob(f"{paper_id}.*"):
            try:
                os.remove(path)
                deleted = True
                print(f"Deleted file: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        
        # 2. Search for thesis folders (Shodhganga)
        # These are directories named thesis_{paper_id}_...
        for path in self.base_dir.rglob(f"thesis_{paper_id}_*"):
            if path.is_dir():
                try:
                    import shutil
                    shutil.rmtree(path)
                    deleted = True
                    print(f"Deleted folder: {path}")
                except Exception as e:
                    print(f"Error deleting folder {path}: {e}")
                    
        return deleted