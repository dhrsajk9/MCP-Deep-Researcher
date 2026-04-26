import PyPDF2
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, Dict
import logging

class PDFProcessor:
    """Process and extract text from PDF files"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.papers_dir = Path(config['storage']['papers_directory'])
        self.max_file_size = self._parse_size(config['storage']['max_file_size'])
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '50MB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    # UPDATED: Added save_dir parameter
    async def download_and_extract_pdf(self, paper: 'ResearchPaper', save_dir: Path = None) -> Optional[str]:
        """Download PDF and extract text content"""
        
        if not paper.pdf_url:
            return None
        
        try:
            # Use the provided save_dir, otherwise default to base papers_dir
            target_dir = save_dir if save_dir else self.papers_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_path = target_dir / f"{paper.id}.pdf"
            
            # Check if already exists
            if pdf_path.exists():
                full_text = self._extract_text_from_pdf(pdf_path)
                full_text = self._normalize_extracted_text(full_text)
                if not full_text:
                    full_text = self._fallback_text_from_metadata(paper)
                self._write_text_file(target_dir, paper.id, full_text)
                return full_text
            
            # Download
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url, ssl=False, timeout=60) as response:
                    if response.status == 200:
                        content_length = response.headers.get('Content-Length')
                        if content_length and int(content_length) > self.max_file_size:
                            logging.warning(f"PDF too large: {paper.id}")
                            return None
                        
                        content = await response.read()
                        
                        with open(pdf_path, 'wb') as f:
                            f.write(content)
                        
                        full_text = self._extract_text_from_pdf(pdf_path)
                        full_text = self._normalize_extracted_text(full_text)
                        if not full_text:
                            full_text = self._fallback_text_from_metadata(paper)
                        self._write_text_file(target_dir, paper.id, full_text)
                        return full_text
                    
        except Exception as e:
            logging.error(f"Error downloading PDF {paper.id}: {e}")
            return None
            
        return None
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF"""
        try:
            # Ensure path is a Path object
            pdf_path = Path(pdf_path)
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                # Limit pages to prevent freezing on massive files
                max_pages = min(len(pdf_reader.pages), 20)
                for i in range(max_pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    text_content.append(page_text or "")
                return "\n".join(text_content)
        except Exception as e:
            logging.error(f"Extraction error {pdf_path}: {e}")
            return ""

    def _normalize_extracted_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        normalized = text.replace("\x00", "").strip()
        return normalized

    def _fallback_text_from_metadata(self, paper: 'ResearchPaper') -> str:
        title = getattr(paper, "title", "") or ""
        abstract = getattr(paper, "abstract", "") or ""
        authors = getattr(paper, "authors", []) or []
        authors_text = ", ".join(authors)

        fallback = (
            f"Title: {title}\n"
            f"Authors: {authors_text}\n"
            f"Abstract: {abstract}\n\n"
            "[Note] PDF text extraction returned empty content. "
            "Stored metadata/abstract fallback."
        )
        return fallback.strip()

    def _write_text_file(self, target_dir: Path, paper_id: str, text: str) -> None:
        text_path = target_dir / f"{paper_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text or "")
