import asyncio
import aiohttp
import arxiv
import feedparser
import json
import os
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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
    """Enhanced arXiv retriever focusing on foundational papers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = arxiv.Client()
        self.rate_limit = config.get('rate_limit', 1)
        
        # Foundational search strategies
        self.foundational_terms = {
            'neural networks': [
                'perceptron', 'multilayer perceptron', 'backpropagation', 
                'feedforward neural network', 'artificial neural network',
                'Hopfield network', 'Kohonen', 'self-organizing map'
            ],
            'deep learning': [
                'deep neural network', 'convolutional neural network', 'CNN',
                'recurrent neural network', 'RNN', 'LSTM', 'autoencoder',
                'deep belief network', 'restricted boltzmann machine'
            ],
            'machine learning': [
                'support vector machine', 'decision tree', 'random forest',
                'naive bayes', 'k-means', 'principal component analysis',
                'linear regression', 'logistic regression'
            ],
            'transformers': [
                'attention mechanism', 'self-attention', 'BERT', 'GPT',
                'sequence-to-sequence', 'encoder-decoder', 'multihead attention'
            ]
        }
        
        # Known influential papers/authors
        self.influential_authors = {
            'neural networks': ['Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio'],
            'deep learning': ['Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio', 'Ian Goodfellow'],
            'transformers': ['Ashish Vaswani', 'Jakob Uszkoreit', 'Noam Shazeer'],
        }
    
    def _get_search_strategies(self, query: str) -> List[Dict]:
        """Get multiple search strategies for comprehensive coverage"""
        query_lower = query.lower()
        strategies = []
        
        # Strategy 1: Direct query with relevance sorting
        strategies.append({
            'query': f'ti:"{query}" OR abs:"{query}"',
            'description': 'Direct query',
            'sort_by': arxiv.SortCriterion.Relevance,
            'max_results': 15
        })
        
        # Strategy 2: Foundational terms
        for topic, terms in self.foundational_terms.items():
            if topic in query_lower:
                foundational_query = ' OR '.join([f'ti:"{term}"' for term in terms[:3]])
                strategies.append({
                    'query': foundational_query,
                    'description': f'Foundational {topic} terms',
                    'sort_by': arxiv.SortCriterion.Relevance,
                    'max_results': 10
                })
                break
        
        # Strategy 3: Category + query combination
        category_mappings = {
            'neural networks': ['cs.NE', 'cs.LG', 'cs.AI'],
            'machine learning': ['cs.LG', 'stat.ML'],
            'deep learning': ['cs.LG', 'cs.CV', 'cs.NE'],
            'transformers': ['cs.CL', 'cs.LG'],
        }
        
        for topic, cats in category_mappings.items():
            if topic in query_lower:
                cat_query = ' OR '.join([f'cat:{cat}' for cat in cats])
                combined_query = f'({query}) AND ({cat_query})'
                strategies.append({
                    'query': combined_query,
                    'description': f'Category-based {topic}',
                    'sort_by': arxiv.SortCriterion.Relevance,
                    'max_results': 10
                })
                break
        
        return strategies
    
    async def search_papers(
        self, 
        query: str, 
        max_results: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[ResearchPaper]:
        """Enhanced search with multiple strategies"""
        
        all_papers = []
        strategies = self._get_search_strategies(query)
        
        print(f"Using {len(strategies)} search strategies for: {query}")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"Strategy {i}: {strategy['description']}")
            
            try:
                search = arxiv.Search(
                    query=strategy['query'],
                    max_results=strategy['max_results'],
                    sort_by=strategy['sort_by'],
                    sort_order=arxiv.SortOrder.Descending
                )
                
                strategy_papers = []
                query_terms = query.lower().split()
                
                for result in self.client.results(search):
                    relevance_score = self._calculate_relevance(result, query, query_terms)
                    
                    if relevance_score > 0:
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
                                "doi": getattr(result, 'doi', ''),
                                "relevance_score": relevance_score,
                                "strategy": strategy['description']
                            }
                        )
                        strategy_papers.append(paper)
                
                strategy_papers.sort(key=lambda p: p.metadata['relevance_score'], reverse=True)
                all_papers.extend(strategy_papers[:5])
                
                await asyncio.sleep(self.rate_limit)
                
            except Exception as e:
                logging.error(f"Strategy {i} failed: {e}")
                continue
        
        # Deduplicate and rank
        unique_papers = self._deduplicate_and_rank(all_papers, query)
        return unique_papers[:max_results]
    
    def _calculate_relevance(self, result, original_query: str, query_terms: List[str]) -> float:
        """Calculate enhanced relevance score"""
        score = 0.0
        title_lower = result.title.lower()
        abstract_lower = result.summary.lower()
        
        # Title matches (highest weight)
        for term in query_terms:
            if term in title_lower:
                score += 3.0
        
        # Abstract matches
        for term in query_terms:
            if term in abstract_lower:
                score += 1.0
        
        # Author influence bonus
        author_names = [author.name.lower() for author in result.authors]
        for topic, influential_authors in self.influential_authors.items():
            if topic in original_query.lower():
                for influential_author in influential_authors:
                    for author_name in author_names:
                        if influential_author.lower() in author_name:
                            score += 2.0
                            break
        
        return score
    
    def _deduplicate_and_rank(self, papers: List[ResearchPaper], query: str) -> List[ResearchPaper]:
        """Deduplicate and rank papers by relevance"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        unique_papers.sort(key=lambda p: p.metadata.get('relevance_score', 0), reverse=True)
        return unique_papers

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
    
    async def search_papers(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
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
    
    def __init__(self, config: Dict, folder_manager=None):
        self.config = config
        self.retrievers = {}
        self.folder_manager = folder_manager
        
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
        
        return self._deduplicate_papers(all_papers)
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_lower = paper.title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers