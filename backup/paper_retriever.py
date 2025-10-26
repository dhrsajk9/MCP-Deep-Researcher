# Enhanced paper_retriever.py with foundational paper search

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
            'computer vision': [
                'image classification', 'object detection', 'feature extraction',
                'edge detection', 'optical character recognition', 'face recognition'
            ],
            'natural language processing': [
                'text classification', 'sentiment analysis', 'machine translation',
                'named entity recognition', 'part-of-speech tagging', 'parsing'
            ],
            'transformers': [
                'attention mechanism', 'self-attention', 'BERT', 'GPT',
                'sequence-to-sequence', 'encoder-decoder', 'multihead attention'
            ]
        }
        
        # Known influential papers/authors
        self.influential_authors = {
            'neural networks': ['Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio', 'David Rumelhart'],
            'deep learning': ['Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio', 'Ian Goodfellow'],
            'transformers': ['Ashish Vaswani', 'Jakob Uszkoreit', 'Noam Shazeer'],
            'computer vision': ['Yann LeCun', 'Alex Krizhevsky', 'Karen Simonyan'],
            'nlp': ['Tomas Mikolov', 'Christopher Manning', 'Dan Jurafsky']
        }
    
    def _get_search_strategies(self, query: str) -> List[Dict]:
        """Get multiple search strategies for comprehensive coverage"""
        query_lower = query.lower()
        strategies = []
        
        # Strategy 1: Direct query with broad date range
        strategies.append({
            'query': f'ti:"{query}" OR abs:"{query}"',
            'description': 'Direct query',
            'sort_by': arxiv.SortCriterion.Relevance,
            'max_results': 20
        })
        
        # Strategy 2: Foundational terms
        for topic, terms in self.foundational_terms.items():
            if topic in query_lower:
                foundational_query = ' OR '.join([f'ti:"{term}"' for term in terms[:5]])
                strategies.append({
                    'query': foundational_query,
                    'description': f'Foundational {topic} terms',
                    'sort_by': arxiv.SortCriterion.Relevance,
                    'max_results': 15
                })
                break
        
        # Strategy 3: Category-based search with relevance sorting
        category_mappings = {
            'neural networks': ['cs.NE', 'cs.LG', 'cs.AI'],
            'machine learning': ['cs.LG', 'stat.ML'],
            'deep learning': ['cs.LG', 'cs.CV', 'cs.NE'],
            'computer vision': ['cs.CV'],
            'natural language processing': ['cs.CL'],
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
                    'max_results': 15
                })
                break
        
        # Strategy 4: Author-based search for influential papers
        for topic, authors in self.influential_authors.items():
            if topic in query_lower:
                author_query = ' OR '.join([f'au:"{author}"' for author in authors[:3]])
                combined_query = f'({query}) AND ({author_query})'
                strategies.append({
                    'query': combined_query,
                    'description': f'Influential authors in {topic}',
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
            print(f"  Query: {strategy['query']}")
            
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
                    # Enhanced relevance scoring
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
                
                # Sort by relevance and take top results
                strategy_papers.sort(key=lambda p: p.metadata['relevance_score'], reverse=True)
                all_papers.extend(strategy_papers[:5])  # Top 5 from each strategy
                
                print(f"  Found {len(strategy_papers)} relevant papers")
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit)
                
            except Exception as e:
                logging.error(f"Strategy {i} failed: {e}")
                continue
        
        # Deduplicate and rank final results
        unique_papers = self._deduplicate_and_rank(all_papers, query)
        
        print(f"Total unique papers found: {len(unique_papers)}")
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
        
        # Journal/venue bonus
        journal_ref = getattr(result, 'journal_ref', '').lower()
        high_impact_venues = ['nature', 'science', 'pnas', 'nips', 'icml', 'iclr', 'neurips']
        for venue in high_impact_venues:
            if venue in journal_ref:
                score += 1.5
                break
        
        # Category relevance
        categories = [cat.lower() for cat in result.categories]
        relevant_categories = ['cs.lg', 'cs.ai', 'cs.ne', 'cs.cv', 'cs.cl']
        for cat in categories:
            if any(rel_cat in cat for rel_cat in relevant_categories):
                score += 0.5
        
        return score
    
    def _deduplicate_and_rank(self, papers: List[ResearchPaper], query: str) -> List[ResearchPaper]:
        """Deduplicate and rank papers by relevance"""
        # Remove exact duplicates
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        # Sort by combined relevance score and diversity
        unique_papers.sort(key=lambda p: (
            p.metadata.get('relevance_score', 0),
            -len(p.title),  # Prefer more specific titles
            p.published_date
        ), reverse=True)
        
        return unique_papers

# Enhanced vector store with better similarity calculation
class EnhancedVectorStore:
    """Enhanced vector store with better similarity handling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.settings = config['settings']
        
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.settings['persist_directory']
        )
        
        # Use a better embedding model for research papers
        embedding_model_name = self.settings.get('embedding_model', 'all-mpnet-base-v2')
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.settings['collection_name'],
            metadata={"description": "Research papers collection"}
        )
        
        logging.info(f"Initialized vector store with {self.collection.count()} documents")
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks with better sentence boundaries"""
        chunk_size = chunk_size or self.settings.get('chunk_size', 800)
        overlap = overlap or self.settings.get('chunk_overlap', 100)
        
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences first
        import re
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for i in range(0, len(words), chunk_size//10):
                        chunk_words = words[i:i + chunk_size//10]
                        chunks.append(' '.join(chunk_words))
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    async def add_paper(self, paper: 'ResearchPaper', full_text: Optional[str] = None):
        """Add a research paper to the vector store with enhanced processing"""
        
        # Prepare text content for embedding
        text_content = []
        
        # Enhanced title and abstract formatting
        title_abstract = f"Research Paper: {paper.title}\n\nAuthors: {', '.join(paper.authors[:3])}\n\nAbstract: {paper.abstract}\n\nCategories: {', '.join(paper.categories) if paper.categories else 'N/A'}"
        text_content.append(title_abstract)
        
        # Add full text if available
        if full_text and len(full_text.strip()) > 100:
            chunks = self._chunk_text(full_text)
            # Enhance chunks with context
            for i, chunk in enumerate(chunks):
                enhanced_chunk = f"Paper: {paper.title}\n\nContent Section {i+1}:\n{chunk}"
                text_content.append(enhanced_chunk)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(text_content)} chunks...")
        embeddings = self.embedding_model.encode(text_content, show_progress_bar=True).tolist()
        
        # Prepare metadata
        metadata = []
        for i, text in enumerate(text_content):
            chunk_metadata = {
                'paper_id': paper.id,
                'title': paper.title,
                'authors': json.dumps(paper.authors),
                'source': paper.source,
                'published_date': paper.published_date,
                'url': paper.url,
                'chunk_index': i,
                'chunk_type': 'title_abstract' if i == 0 else 'content',
                'categories': json.dumps(paper.categories) if paper.categories else json.dumps([]),
                'relevance_score': paper.metadata.get('relevance_score', 0) if paper.metadata else 0
            }
            metadata.append(chunk_metadata)
        
        # Generate unique IDs for each chunk
        ids = [f"{paper.id}_{i}" for i in range(len(text_content))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=text_content,
            metadatas=metadata,
            ids=ids
        )
        
        logging.info(f"Added paper {paper.id} with {len(text_content)} chunks to vector store")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """Enhanced search with better query processing"""
        
        # Use lower default threshold
        similarity_threshold = similarity_threshold or 0.15
        
        print(f"Enhanced vector search - Query: '{query}', Threshold: {similarity_threshold}")
        
        try:
            # Generate multiple query variations
            query_variations = [
                query,
                f"What are {query}?",
                f"Applications of {query}",
                f"Introduction to {query}",
                f"{query} research paper"
            ]
            
            all_results = []
            
            for q_var in query_variations:
                query_embedding = self.embedding_model.encode([q_var]).tolist()
                
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=max_results * 2,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        distance = results['distances'][0][i]
                        # Use cosine similarity (better for semantic search)
                        similarity = max(0, 1 - distance)
                        
                        if similarity >= similarity_threshold:
                            result = {
                                'document': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'similarity': similarity,
                                'distance': distance,
                                'query_variant': q_var
                            }
                            all_results.append(result)
            
            # Remove duplicates and sort by similarity
            seen_docs = set()
            unique_results = []
            for result in all_results:
                doc_key = result['document'][:100]  # Use first 100 chars as key
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            final_results = unique_results[:max_results]
            
            print(f"Found {len(final_results)} unique results above threshold {similarity_threshold}")
            for i, result in enumerate(final_results):
                print(f"  {i+1}. Similarity: {result['similarity']:.3f} - {result['metadata'].get('title', 'Unknown')[:50]}...")
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error in enhanced vector search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.settings['collection_name'],
            'embedding_model': self.settings['embedding_model']
        }