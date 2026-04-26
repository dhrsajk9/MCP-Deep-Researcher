import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import json
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for research papers using ChromaDB"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.settings = config['settings']
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.settings['persist_directory']
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.settings['embedding_model']
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.settings['collection_name'],
            metadata={"description": "Research papers collection"}
        )
        
        logging.info(f"Initialized vector store with {self.collection.count()} documents")
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks with better sentence boundaries"""
        chunk_size = chunk_size or self.settings.get('chunk_size', 1000)
        overlap = overlap or self.settings.get('chunk_overlap', 150)
        
        if len(text) <= chunk_size:
            return [text]
        
        # Better sentence splitting
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
        
        # Enhanced title and abstract formatting for better matching
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
        logger.info("Generating embeddings for %s chunks...", len(text_content))
        embeddings = self.embedding_model.encode(text_content).tolist()
        
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
        """Enhanced search with much lower thresholds and multiple query variations"""
        
        # Use much lower default threshold for more results
        similarity_threshold = similarity_threshold or 0.05  # Very low threshold
        
        logger.info(
            "Enhanced vector search - Query: '%s', Threshold: %s",
            query,
            similarity_threshold,
        )
        
        try:
            # Generate multiple query variations for better matching
            query_variations = [
                query,
                f"What are {query}?",
                f"Applications of {query}",
                f"{query} research",
                f"Introduction to {query}",
                f"{query} machine learning",
                f"{query} artificial intelligence"
            ]
            
            all_results = []
            
            for q_var in query_variations:
                query_embedding = self.embedding_model.encode([q_var]).tolist()
                
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=max_results * 4,  # Get many more results
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        distance = results['distances'][0][i]
                        similarity = max(0, 1 - distance)
                        
                        # Very lenient threshold
                        if similarity >= similarity_threshold:
                            result = {
                                'document': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'similarity': similarity,
                                'distance': distance,
                                'query_variant': q_var
                            }
                            all_results.append(result)
            
            # Remove duplicates based on document content
            seen_docs = set()
            unique_results = []
            for result in all_results:
                doc_key = result['document'][:100]  # Use first 100 chars as key
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append(result)
            
            # Sort by similarity and take top results
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            final_results = unique_results[:max_results]
            
            logger.info(
                "Found %s unique results above threshold %s",
                len(final_results),
                similarity_threshold,
            )
            for i, result in enumerate(final_results):
                title = result['metadata'].get('title', 'Unknown')[:40]
                logger.info("  %s. Sim: %.3f - %s...", i + 1, result['similarity'], title)
            
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
    
    async def delete_paper(self, paper_id: str):
        """Delete all vectors for a specific paper ID"""
        try:
            self.collection.delete(where={"paper_id": paper_id})
            logging.info(f"Deleted vectors for paper {paper_id}")
            return True
        except Exception as e:
            logging.error(f"Error deleting vectors: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.settings['collection_name'],
            'embedding_model': self.settings['embedding_model']
        }
    
    def get_all_papers_metadata(self) -> List[Dict]:
        """Get comprehensive metadata for all unique papers in DB"""
        try:
            # Fetch all metadata
            data = self.collection.get(include=["metadatas"])
            seen = set()
            papers = []
            
            for m in data['metadatas']:
                pid = m.get('paper_id')
                # Only process if we haven't seen this paper ID yet
                if pid and pid not in seen:
                    seen.add(pid)
                    
                    # Parse Authors (stored as JSON string)
                    try:
                        authors = json.loads(m.get('authors', '[]'))
                        if isinstance(authors, list):
                            authors = ", ".join(authors)
                    except:
                        authors = m.get('authors', 'Unknown')

                    # Build detailed object
                    papers.append({
                        'id': pid,
                        'title': m.get('title', 'Unknown'),
                        'authors': authors,
                        'source': m.get('source', 'Unknown').capitalize(),
                        'date': m.get('published_date', 'Unknown'),
                        'url': m.get('url', '#'),
                        'chunk_count': 1 # We could count chunks if needed, but simple is fast
                    })
            return papers
        except Exception as e:
            logging.error(f"Error fetching metadata: {e}")
            return []
