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
        
        # Use a modern browser user agent
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
                # Build search URL - matches the actual site format
                search_url = f"{self.base_url}/simple-search"
                params = {
                    'query': query,
                    'go': ''  # Empty 'go' parameter as seen in actual URL
                }
                
                self.logger.info(f"Fetching: {search_url}?query={query}")
                
                # Fetch search results
                html = await self._fetch_page(session, search_url, params)
                if not html:
                    self.logger.error("Failed to fetch search results")
                    return []
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find the results table - based on your screenshot
                # The table has headers: Upload Date, Title, Researcher, Guide(s)
                result_table = soup.find('table', class_='table')
                
                if not result_table:
                    # Try finding any table
                    result_table = soup.find('table')
                
                if not result_table:
                    self.logger.error("Could not find results table")
                    self.logger.info(f"Page title: {soup.find('title').text if soup.find('title') else 'None'}")
                    return []
                
                # Find all result rows (skip header row)
                rows = result_table.find_all('tr')[1:]  # Skip first row (header)
                
                self.logger.info(f"Found {len(rows)} result rows")
                
                for row in rows[:max_results]:
                    try:
                        thesis = await self._parse_row(row, session, university)
                        if thesis:
                            theses.append(thesis)
                            self.logger.info(f"Successfully parsed: {thesis.title[:60]}...")
                            
                            # Respectful rate limiting
                            await asyncio.sleep(self.rate_limit)
                    except Exception as e:
                        self.logger.error(f"Error parsing row: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error searching Shodhganga: {e}", exc_info=True)
        
        self.logger.info(f"Found {len(theses)} theses")
        return theses
    
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Optional[str]:
        """Fetch a page with retries and error handling"""
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params, timeout=self.timeout, ssl=False) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:
                        wait_time = self.rate_limit * (attempt + 1) * 2
                        self.logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(self.rate_limit * 2)
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
                await asyncio.sleep(self.rate_limit)
        
        return None
    
    async def _parse_row(self, row, session: aiohttp.ClientSession, university_filter: Optional[str]) -> Optional[Thesis]:
        """Parse a single result row from the table"""
        
        cells = row.find_all('td')
        if len(cells) < 4:
            return None
        
        # Based on the table structure:
        # Column 0: Upload Date
        # Column 1: Title (with link)
        # Column 2: Researcher (author)
        # Column 3: Guide(s)
        
        upload_date_cell = cells[0]
        title_cell = cells[1]
        researcher_cell = cells[2]
        guide_cell = cells[3]
        
        # Extract title and URL
        title_link = title_cell.find('a')
        if not title_link:
            return None
        
        title = title_link.get_text(strip=True)
        thesis_url = title_link.get('href', '')
        
        # Make absolute URL
        if thesis_url and not thesis_url.startswith('http'):
            thesis_url = urljoin(self.base_url, thesis_url)
        
        # Extract researcher (author)
        researcher = researcher_cell.get_text(strip=True)
        authors = [researcher] if researcher else ['Unknown']
        
        # Extract guide
        guide = guide_cell.get_text(strip=True)
        
        # Extract upload date
        upload_date = upload_date_cell.get_text(strip=True)
        
        # Extract thesis ID from URL
        # Typical format: /handle/10603/123456
        thesis_id_match = re.search(r'/handle/(\d+/\d+)', thesis_url)
        thesis_id = thesis_id_match.group(1).replace('/', '_') if thesis_id_match else 'unknown'
        
        self.logger.info(f"Fetching details for: {title[:50]}...")
        
        # Fetch detailed page
        detail_html = await self._fetch_page(session, thesis_url)
        if not detail_html:
            self.logger.warning(f"Could not fetch details for {title}")
            # Still create thesis with basic info
            return Thesis(
                id=thesis_id,
                title=title,
                authors=authors,
                abstract="Details not available - check the URL",
                published_date=upload_date,
                url=thesis_url,
                chapter_links=[],
                source="shodhganga",
                metadata={
                    'guide': guide,
                    'upload_date': upload_date,
                    'university': 'Unknown',
                    'department': 'Unknown',
                    'degree': 'Unknown',
                    'year': upload_date.split('-')[2] if '-' in upload_date else 'Unknown',
                    'chapters_count': 0
                }
            )
        
        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        
        # Extract metadata from detail page
        metadata = self._extract_metadata(detail_soup)
        
        # Apply university filter if specified
        if university_filter:
            thesis_university = metadata.get('university', '').lower()
            if university_filter.lower() not in thesis_university:
                self.logger.info(f"Skipping - university filter not matched: {thesis_university}")
                return None
        
        # Extract chapter links
        chapter_links = self._extract_chapter_links(detail_soup, thesis_url)
        
        # Build thesis object
        thesis = Thesis(
            id=thesis_id,
            title=title,
            authors=authors,
            abstract=metadata.get('abstract', 'No abstract available'),
            published_date=metadata.get('year', upload_date.split('-')[2] if '-' in upload_date else 'Unknown'),
            url=thesis_url,
            chapter_links=chapter_links,
            source="shodhganga",
            metadata={
                'university': metadata.get('university', 'Unknown'),
                'department': metadata.get('department', 'Unknown'),
                'guide': guide,
                'degree': metadata.get('degree', 'Unknown'),
                'year': metadata.get('year', upload_date.split('-')[2] if '-' in upload_date else 'Unknown'),
                'subject': metadata.get('subject', []),
                'upload_date': upload_date,
                'chapters_count': len(chapter_links) if chapter_links else 0,
                'raw_metadata': metadata
            }
        )
        
        return thesis
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from thesis detail page"""
        
        metadata = {}
        
        try:
            # Look for metadata tables (DSpace typical structure)
            meta_tables = soup.find_all('table', class_='ds-includeSet-table')
            
            if not meta_tables:
                # Try any table
                meta_tables = soup.find_all('table')
            
            for table in meta_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        # Parse different metadata fields
                        if 'author' in label or 'researcher' in label or 'scholar' in label:
                            if 'authors' not in metadata:
                                metadata['authors'] = []
                            metadata['authors'].append(value)
                        elif 'guide' in label or 'supervisor' in label:
                            metadata['guide'] = value
                        elif 'university' in label or 'institution' in label:
                            metadata['university'] = value
                        elif 'department' in label or 'faculty' in label or 'school' in label:
                            metadata['department'] = value
                        elif 'year' in label or 'date' in label:
                            year_match = re.search(r'\b(19|20)\d{2}\b', value)
                            if year_match:
                                metadata['year'] = year_match.group()
                        elif 'degree' in label or 'qualification' in label:
                            metadata['degree'] = value
                        elif 'subject' in label or 'keyword' in label:
                            metadata['subject'] = [s.strip() for s in value.split(',') if s.strip()]
                        elif 'abstract' in label or 'summary' in label:
                            metadata['abstract'] = value
            
            # Try to find abstract in a dedicated div
            if 'abstract' not in metadata:
                abstract_div = soup.find('div', class_=re.compile(r'abstract', re.I))
                if abstract_div:
                    metadata['abstract'] = abstract_div.get_text(strip=True)
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_chapter_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract chapter PDF links"""
        
        chapter_links = []
        
        try:
            # Find all PDF links
            # DSpace typically uses bitstream URLs
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$|/bitstream/|/retrieve/', re.I))
            
            for link in pdf_links:
                href = link.get('href', '')
                # Get link text or title
                name = link.get_text(strip=True) or link.get('title', '') or 'Document'
                
                # Skip if empty
                if not href or not name:
                    continue
                
                # Make absolute URL
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                
                chapter_info = {
                    'name': name,
                    'url': href,
                    'type': 'pdf'
                }
                
                # Try to extract chapter number from name
                chapter_match = re.search(r'chapter[_\s-]*(\d+)|ch[_\s-]*(\d+)|(\d+)[_\s-]*chapter', name, re.IGNORECASE)
                if chapter_match:
                    chapter_num = next(g for g in chapter_match.groups() if g)
                    chapter_info['chapter_number'] = int(chapter_num)
                
                # Avoid duplicates
                if href not in [c['url'] for c in chapter_links]:
                    chapter_links.append(chapter_info)
            
            # Sort by chapter number if available
            chapter_links.sort(key=lambda x: x.get('chapter_number', 999))
            
            self.logger.info(f"Found {len(chapter_links)} chapter links")
        
        except Exception as e:
            self.logger.error(f"Error extracting chapter links: {e}")
        
        return chapter_links
    
    async def save_thesis_info(self, thesis: Thesis, thesis_folder: Path):
        """Save all thesis information to a comprehensive note file"""
        
        # Create detailed note
        note_content = f"""# Shodhganga Thesis Information

## Basic Information
- **Title**: {thesis.title}
- **Author(s)**: {', '.join(thesis.authors)}
- **University**: {thesis.metadata.get('university', 'Unknown') if thesis.metadata else 'Unknown'}
- **Department**: {thesis.metadata.get('department', 'Unknown') if thesis.metadata else 'Unknown'}
- **Year**: {thesis.published_date}
- **Degree**: {thesis.metadata.get('degree', 'Unknown') if thesis.metadata else 'Unknown'}
- **Guide/Supervisor**: {thesis.metadata.get('guide', 'Unknown') if thesis.metadata else 'Unknown'}
- **Upload Date**: {thesis.metadata.get('upload_date', 'Unknown') if thesis.metadata else 'Unknown'}

## Thesis URL
{thesis.url}

## Abstract
{thesis.abstract}

## Subjects/Keywords
{', '.join(thesis.metadata.get('subject', [])) if thesis.metadata and thesis.metadata.get('subject') else 'Not specified'}

## Chapter Links ({len(thesis.chapter_links) if thesis.chapter_links else 0} chapters)

"""
        
        if thesis.chapter_links:
            for i, chapter in enumerate(thesis.chapter_links, 1):
                chapter_num = chapter.get('chapter_number', i)
                note_content += f"""
### Chapter {chapter_num}: {chapter['name']}
- **URL**: {chapter['url']}
- **Type**: {chapter.get('type', 'pdf').upper()}

"""
        else:
            note_content += "*No chapter links found - you may need to visit the URL above to access files*\n"
        
        note_content += f"""
## Download Instructions

To download chapters manually:
1. Visit the thesis URL above
2. Click on each chapter link
3. Save the PDF files to this folder

Or use a download manager like wget:
```bash
# Example for first chapter (update URL)
wget "{thesis.chapter_links[0]['url'] if thesis.chapter_links else thesis.url}" -O chapter_01.pdf
```

## Metadata (JSON)

```json
{json.dumps(asdict(thesis), indent=2, ensure_ascii=False)}
```

---
*Retrieved from Shodhganga on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        # Save note as markdown
        note_file = thesis_folder / "THESIS_INFO.md"
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        # Also save as JSON for programmatic access
        json_file = thesis_folder / "metadata.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(thesis), f, indent=2, ensure_ascii=False)
        
        # Save chapter links separately for easy access
        if thesis.chapter_links:
            links_file = thesis_folder / "chapter_links.json"
            with open(links_file, 'w', encoding='utf-8') as f:
                json.dump(thesis.chapter_links, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved thesis information to {note_file}")
        
        return note_file