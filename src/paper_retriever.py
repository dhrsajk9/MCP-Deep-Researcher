import asyncio
import aiohttp
import arxiv
import feedparser
import json
import os
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class ResearchPaper:
    """Data structure for research papers"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    url: str
    pdf_url: Optional[str] = None
    categories: List[str] = None
    source: str = "arxiv"
    local_path: Optional[str] = None
    metadata: Dict = None

class ArxivRetriever:
    """Retrieve papers from arXiv API"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = arxiv.Client()
        self.rate_limit = config.get('rate_limit', 3)
    
    async def search_papers(
        self, 
        query: str, 
        max_results: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[ResearchPaper]:
        """Search for papers on arXiv"""
        
        # Build search query
        if categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({category_query})"
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            paper = ResearchPaper(
                id=result.entry_id.split('/')[-1],
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                published_date=result.published.isoformat(),
                url=result.entry_id,
                pdf_url=result.pdf_url,
                categories=[cat for cat in result.categories],
                source="arxiv",
                metadata={
                    "comment": getattr(result, 'comment', ''),
                    "journal_ref": getattr(result, 'journal_ref', ''),
                    "doi": getattr(result, 'doi', '')
                }
            )
            papers.append(paper)
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit)
        
        return papers

class SemanticScholarRetriever:
    """Retrieve papers from Semantic Scholar API"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['base_url']
        self.api_key = config.get('api_key')
        self.session = None
    
    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_papers(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[ResearchPaper]:
        """Search for papers on Semantic Scholar"""
        
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'paperId,title,authors,abstract,publicationDate,url,openAccessPdf'
        }
        
        async with self.session.get(f"{self.base_url}/search", params=params) as response:
            data = await response.json()
        
        papers = []
        for item in data.get('data', []):
            paper = ResearchPaper(
                id=item['paperId'],
                title=item.get('title', ''),
                authors=[author['name'] for author in item.get('authors', [])],
                abstract=item.get('abstract', ''),
                published_date=item.get('publicationDate', ''),
                url=item.get('url', ''),
                pdf_url=item.get('openAccessPdf', {}).get('url') if item.get('openAccessPdf') else None,
                source="semantic_scholar",
                metadata=item
            )
            papers.append(paper)
        
        return papers

class PaperRetrieverManager:
    """Manager for multiple paper retrieval sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.retrievers = {}
        
        # Initialize retrievers based on config
        if 'arxiv' in config:
            self.retrievers['arxiv'] = ArxivRetriever(config['arxiv'])
        
        if 'semantic_scholar' in config:
            self.retrievers['semantic_scholar'] = SemanticScholarRetriever(config['semantic_scholar'])
    
    async def search_papers(
        self, 
        query: str, 
        sources: List[str] = None,
        max_results: int = 10
    ) -> List[ResearchPaper]:
        """Search papers across multiple sources"""
        
        if sources is None:
            sources = list(self.retrievers.keys())
        
        all_papers = []
        
        for source in sources:
            if source in self.retrievers:
                try:
                    if source == 'semantic_scholar':
                        async with self.retrievers[source] as retriever:
                            papers = await retriever.search_papers(query, max_results)
                    else:
                        papers = await self.retrievers[source].search_papers(query, max_results)
                    
                    all_papers.extend(papers)
                except Exception as e:
                    logging.error(f"Error retrieving from {source}: {e}")
        
        # Remove duplicates based on title similarity
        return self._deduplicate_papers(all_papers)
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        # Simple deduplication - in practice, you might want more sophisticated matching
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_lower = paper.title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers