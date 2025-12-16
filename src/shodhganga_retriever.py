"""
Working Shodhganga scraper based on actual site structure
Tested with: https://shodhganga.inflibnet.ac.in/simple-search?query=machine+learning
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from urllib.parse import urljoin, quote
from datetime import datetime

@dataclass
class Thesis:
    """Data structure for Shodhganga theses"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    url: str
    # Added categories field to fix AttributeError in vector_store.py
    categories: List[str] = None
    chapter_links: List[Dict] = None
    source: str = "shodhganga"
    local_path: Optional[str] = None
    metadata: Dict = None

class ShodhgangaRetriever:
    """Working Shodhganga scraper based on actual website structure"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get('base_url', 'https://shodhganga.inflibnet.ac.in')
        self.rate_limit = config.get('rate_limit', 3)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def search_theses(
        self,
        query: str,
        university: Optional[str] = None,
        max_results: int = 5
    ) -> List[Thesis]:
        """Search for theses on Shodhganga"""
        self.logger.info(f"Searching Shodhganga for: {query}")
        theses = []
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                search_url = f"{self.base_url}/simple-search"
                params = {'query': query, 'go': ''}
                
                html = await self._fetch_page(session, search_url, params)
                if not html:
                    return []
                
                soup = BeautifulSoup(html, 'html.parser')
                result_table = soup.find('table', class_='table')
                if not result_table:
                    result_table = soup.find('table')
                if not result_table:
                    return []
                
                rows = result_table.find_all('tr')[1:]
                
                for row in rows[:max_results]:
                    try:
                        thesis = await self._parse_row(row, session, university)
                        if thesis:
                            theses.append(thesis)
                            await asyncio.sleep(self.rate_limit)
                    except Exception as e:
                        self.logger.error(f"Error parsing row: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error searching Shodhganga: {e}")
        
        return theses
    
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params, timeout=self.timeout, ssl=False) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:
                        await asyncio.sleep(self.rate_limit * (attempt + 1) * 2)
                    else:
                        await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(self.rate_limit)
        return None
    
    async def _parse_row(self, row, session: aiohttp.ClientSession, university_filter: Optional[str]) -> Optional[Thesis]:
        cells = row.find_all('td')
        if len(cells) < 4:
            return None
        
        upload_date = cells[0].get_text(strip=True)
        title_link = cells[1].find('a')
        if not title_link:
            return None
        title = title_link.get_text(strip=True)
        thesis_url = title_link.get('href', '')
        if thesis_url and not thesis_url.startswith('http'):
            thesis_url = urljoin(self.base_url, thesis_url)
            
        researcher = cells[2].get_text(strip=True)
        authors = [researcher] if researcher else ['Unknown']
        guide = cells[3].get_text(strip=True)
        
        match = re.search(r'/handle/(\d+/\d+)', thesis_url)
        thesis_id = match.group(1).replace('/', '_') if match else 'unknown'
        
        detail_html = await self._fetch_page(session, thesis_url)
        if not detail_html:
            return Thesis(
                id=thesis_id, title=title, authors=authors, abstract="No details", 
                published_date=upload_date, url=thesis_url, categories=[], 
                chapter_links=[], source="shodhganga", metadata={}
            )
        
        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        metadata = self._extract_metadata(detail_soup)
        
        if university_filter:
            if university_filter.lower() not in metadata.get('university', '').lower():
                return None
        
        chapter_links = self._extract_chapter_links(detail_soup, thesis_url)
        
        return Thesis(
            id=thesis_id,
            title=title,
            authors=authors,
            abstract=metadata.get('abstract', 'No abstract available'),
            published_date=metadata.get('year', upload_date),
            url=thesis_url,
            categories=metadata.get('subject', []),
            chapter_links=chapter_links,
            source="shodhganga",
            metadata={
                'university': metadata.get('university', 'Unknown'),
                'department': metadata.get('department', 'Unknown'),
                'guide': guide,
                'degree': metadata.get('degree', 'Unknown'),
                'subject': metadata.get('subject', []),
                'upload_date': upload_date
            }
        )
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        metadata = {}
        tables = soup.find_all('table')
        for table in tables:
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    if 'university' in label: metadata['university'] = value
                    elif 'department' in label: metadata['department'] = value
                    elif 'abstract' in label: metadata['abstract'] = value
                    elif 'subject' in label: metadata['subject'] = [s.strip() for s in value.split(',')]
                    elif 'year' in label or 'date' in label:
                         m = re.search(r'\b(19|20)\d{2}\b', value)
                         if m: metadata['year'] = m.group()
        
        if 'abstract' not in metadata:
            div = soup.find('div', class_=re.compile(r'abstract', re.I))
            if div: metadata['abstract'] = div.get_text(strip=True)
            
        return metadata
    
    def _extract_chapter_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        links = []
        for a in soup.find_all('a', href=re.compile(r'\.pdf$|/bitstream/', re.I)):
            href = a.get('href', '')
            if not href: continue
            if not href.startswith('http'): href = urljoin(base_url, href)
            name = a.get_text(strip=True) or 'Document'
            if href not in [x['url'] for x in links]:
                links.append({'name': name, 'url': href, 'type': 'pdf'})
        return links
    
    async def save_thesis_info(self, thesis: Thesis, thesis_folder: Path):
        # Using list construction to avoid f-string SyntaxError
        lines = [
            "# Shodhganga Thesis Information", "",
            f"- **Title**: {thesis.title}",
            f"- **Author**: {', '.join(thesis.authors)}",
            f"- **University**: {thesis.metadata.get('university', 'Unknown')}",
            "", "## Abstract", str(thesis.abstract), "",
            "## Metadata", "```json",
            json.dumps(asdict(thesis), indent=2, ensure_ascii=False),
            "```"
        ]
        
        with open(thesis_folder / "THESIS_INFO.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
        with open(thesis_folder / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(thesis), f, indent=2, ensure_ascii=False)