"""
Alternative Shodhganga scraper using browse/discover endpoints
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
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
    chapter_links: List[Dict] = None
    source: str = "shodhganga"
    local_path: Optional[str] = None
    metadata: Dict = None

class ShodhgangaRetrieverV2:
    """Alternative Shodhganga scraper with multiple strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get('base_url', 'https://shodhganga.inflibnet.ac.in')
        self.rate_limit = config.get('rate_limit', 3)
        self.timeout = config.get('timeout', 30)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def search_theses(
        self,
        query: str,
        university: Optional[str] = None,
        max_results: int = 5
    ) -> List[Thesis]:
        """Search with multiple fallback strategies"""
        
        self.logger.info(f"Searching Shodhganga for: {query}")
        
        # Try multiple search strategies
        strategies = [
            self._search_simple_search,
            self._search_discover,
            self._browse_by_subject,
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                self.logger.info(f"Trying strategy {i}/{len(strategies)}: {strategy.__name__}")
                theses = await strategy(query, university, max_results)
                
                if theses:
                    self.logger.info(f"Success with strategy {i}: found {len(theses)} theses")
                    return theses
                    
            except Exception as e:
                self.logger.error(f"Strategy {i} failed: {e}")
                continue
        
        self.logger.warning("All search strategies failed")
        return []
    
    async def _search_simple_search(self, query: str, university: Optional[str], max_results: int) -> List[Thesis]:
        """Strategy 1: Use simple-search endpoint"""
        
        search_url = f"{self.base_url}/simple-search"
        params = {
            'query': query,
            'start': 0,
            'rpp': min(max_results * 2, 20)
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            html = await self._fetch_page(session, search_url, params)
            if not html:
                return []
            
            soup = BeautifulSoup(html, 'html.parser')
            return await self._parse_search_results(soup, session, university, max_results)
    
    async def _search_discover(self, query: str, university: Optional[str], max_results: int) -> List[Thesis]:
        """Strategy 2: Use discover endpoint"""
        
        search_url = f"{self.base_url}/discover"
        params = {
            'query': query,
            'scope': '/',
            'rpp': min(max_results * 2, 20)
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            html = await self._fetch_page(session, search_url, params)
            if not html:
                return []
            
            soup = BeautifulSoup(html, 'html.parser')
            return await self._parse_search_results(soup, session, university, max_results)
    
    async def _browse_by_subject(self, query: str, university: Optional[str], max_results: int) -> List[Thesis]:
        """Strategy 3: Browse by subject (fallback)"""
        
        # This would browse categories related to the query
        # For now, return empty - can be implemented if search fails
        return []
    
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Optional[str]:
        """Fetch a page with error handling"""
        
        try:
            async with session.get(url, params=params, timeout=self.timeout, ssl=False) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
        
        return None
    
    async def _parse_search_results(self, soup: BeautifulSoup, session: aiohttp.ClientSession, university_filter: Optional[str], max_results: int) -> List[Thesis]:
        """Parse search results with flexible selectors"""
        
        theses = []
        
        # Try multiple selectors for result items
        result_containers = (
            soup.find_all('div', class_='artifact-description') or
            soup.find_all('tr', class_='ds-table-row') or
            soup.find_all('div', class_=re.compile(r'result|item', re.I)) or
            soup.find_all('article')
        )
        
        if not result_containers:
            # Try finding any links with /handle/ in them
            handle_links = soup.find_all('a', href=re.compile(r'/handle/\d+/\d+'))
            if handle_links:
                self.logger.info(f"Found {len(handle_links)} handle links")
                # Create pseudo-containers from links
                result_containers = [link.parent for link in handle_links if link.parent]
        
        self.logger.info(f"Found {len(result_containers)} potential result containers")
        
        for container in result_containers[:max_results * 2]:
            try:
                thesis = await self._parse_single_result(container, session, university_filter)
                if thesis:
                    theses.append(thesis)
                    
                    if len(theses) >= max_results:
                        break
                    
                    await asyncio.sleep(self.rate_limit)
                    
            except Exception as e:
                self.logger.error(f"Error parsing result: {e}")
                continue
        
        return theses
    
    async def _parse_single_result(self, container, session: aiohttp.ClientSession, university_filter: Optional[str]) -> Optional[Thesis]:
        """Parse a single search result"""
        
        # Find title and URL
        title_link = (
            container.find('a', class_='artifact-title') or
            container.find('a', href=re.compile(r'/handle/')) or
            container.find('a')
        )
        
        if not title_link:
            return None
        
        title = title_link.get_text(strip=True)
        thesis_url = title_link.get('href', '')
        
        if not thesis_url.startswith('http'):
            thesis_url = urljoin(self.base_url, thesis_url)
        
        # Extract ID
        thesis_id_match = re.search(r'/handle/(\d+/\d+)', thesis_url)
        thesis_id = thesis_id_match.group(1).replace('/', '_') if thesis_id_match else 'unknown'
        
        self.logger.info(f"Parsing: {title[:50]}...")
        
        # Fetch detailed page
        detail_html = await self._fetch_page(session, thesis_url)
        if not detail_html:
            return None
        
        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        
        # Extract metadata
        metadata = self._extract_metadata_flexible(detail_soup)
        
        # Apply university filter
        if university_filter:
            thesis_university = metadata.get('university', '').lower()
            if university_filter.lower() not in thesis_university:
                return None
        
        # Extract chapter links
        chapter_links = self._extract_chapter_links_flexible(detail_soup, thesis_url)
        
        thesis = Thesis(
            id=thesis_id,
            title=title,
            authors=metadata.get('authors', ['Unknown']),
            abstract=metadata.get('abstract', 'No abstract available'),
            published_date=metadata.get('year', 'Unknown'),
            url=thesis_url,
            chapter_links=chapter_links,
            source="shodhganga",
            metadata={
                'university': metadata.get('university', 'Unknown'),
                'department': metadata.get('department', 'Unknown'),
                'guide': metadata.get('guide', 'Unknown'),
                'degree': metadata.get('degree', 'Unknown'),
                'year': metadata.get('year', 'Unknown'),
                'subject': metadata.get('subject', []),
                'chapters_count': len(chapter_links) if chapter_links else 0,
            }
        )
        
        return thesis
    
    def _extract_metadata_flexible(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata with flexible selectors"""
        
        metadata = {}
        
        # Try multiple methods to extract metadata
        
        # Method 1: Look for metadata tables
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    
                    self._parse_metadata_field(label, value, metadata)
        
        # Method 2: Look for definition lists
        for dl in soup.find_all('dl'):
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            for dt, dd in zip(dts, dds):
                label = dt.get_text(strip=True).lower()
                value = dd.get_text(strip=True)
                self._parse_metadata_field(label, value, metadata)
        
        # Method 3: Look for meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if 'author' in name:
                if 'authors' not in metadata:
                    metadata['authors'] = []
                metadata['authors'].append(content)
            elif 'date' in name:
                year_match = re.search(r'\b(19|20)\d{2}\b', content)
                if year_match:
                    metadata['year'] = year_match.group()
        
        # Method 4: Look for abstract
        if 'abstract' not in metadata:
            abstract_div = soup.find(['div', 'section'], class_=re.compile(r'abstract', re.I))
            if abstract_div:
                metadata['abstract'] = abstract_div.get_text(strip=True)
        
        return metadata
    
    def _parse_metadata_field(self, label: str, value: str, metadata: Dict):
        """Parse a single metadata field"""
        
        if any(word in label for word in ['author', 'researcher', 'scholar']):
            metadata['authors'] = [value]
        elif any(word in label for word in ['guide', 'supervisor']):
            metadata['guide'] = value
        elif 'university' in label or 'institution' in label:
            metadata['university'] = value
        elif 'department' in label or 'faculty' in label:
            metadata['department'] = value
        elif 'year' in label or 'date' in label:
            year_match = re.search(r'\b(19|20)\d{2}\b', value)
            if year_match:
                metadata['year'] = year_match.group()
        elif 'degree' in label:
            metadata['degree'] = value
        elif 'subject' in label or 'keyword' in label:
            metadata['subject'] = [s.strip() for s in value.split(',')]
        elif 'abstract' in label:
            metadata['abstract'] = value
    
    def _extract_chapter_links_flexible(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract chapter links with flexible selectors"""
        
        chapter_links = []
        
        # Look for PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$|bitstream|download', re.I))
        
        for link in pdf_links:
            href = link.get('href', '')
            name = link.get_text(strip=True) or link.get('title', '') or f"Document {len(chapter_links) + 1}"
            
            if href:
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                
                chapter_info = {
                    'name': name,
                    'url': href,
                    'type': 'pdf'
                }
                
                # Try to extract chapter number
                chapter_match = re.search(r'chapter[_\s-]*(\d+)|ch[_\s-]*(\d+)|(\d+)[_\s-]*chapter', name, re.I)
                if chapter_match:
                    chapter_num = next(g for g in chapter_match.groups() if g)
                    chapter_info['chapter_number'] = int(chapter_num)
                
                if href not in [c['url'] for c in chapter_links]:
                    chapter_links.append(chapter_info)
        
        # Sort by chapter number
        chapter_links.sort(key=lambda x: x.get('chapter_number', 999))
        
        return chapter_links
    
    async def save_thesis_info(self, thesis: Thesis, thesis_folder: Path):
        """Save thesis information"""
        
        # (Same as before - keep the existing implementation)
        pass