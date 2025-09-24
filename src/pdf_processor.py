import PyPDF2
import requests
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
    
    async def download_and_extract_pdf(self, paper: 'ResearchPaper') -> Optional[str]:
        """Download PDF and extract text content"""
        
        if not paper.pdf_url:
            return None
        
        try:
            # Download PDF
            pdf_path = await self._download_pdf(paper)
            if not pdf_path:
                return None
            
            # Extract text
            text_content = self._extract_text_from_pdf(pdf_path)
            
            # Update paper metadata
            paper.local_path = str(pdf_path)
            
            return text_content
            
        except Exception as e:
            logging.error(f"Error processing PDF for paper {paper.id}: {e}")
            return None
    
    async def _download_pdf(self, paper: 'ResearchPaper') -> Optional[Path]:
        """Download PDF file"""
        
        pdf_filename = f"{paper.id}.pdf"
        pdf_path = self.papers_dir / pdf_filename
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url) as response:
                    if response.status == 200:
                        content_length = response.headers.get('Content-Length')
                        
                        if content_length and int(content_length) > self.max_file_size:
                            logging.warning(f"PDF too large for paper {paper.id}")
                            return None
                        
                        content = await response.read()
                        
                        if len(content) > self.max_file_size:
                            logging.warning(f"PDF too large for paper {paper.id}")
                            return None
                        
                        with open(pdf_path, 'wb') as f:
                            f.write(content)
                        
                        return pdf_path
                    
        except Exception as e:
            logging.error(f"Error downloading PDF for paper {paper.id}: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF"""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                
                return "\n".join(text_content)
                
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            return ""