import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import json
import logging
from dataclasses import asdict

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
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.settings['chunk_size']
        overlap = overlap or self.settings['chunk_overlap']
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_exclamation = chunk.rfind('!')
                last_question = chunk.rfind('?')
                
                sentence_end = max(last_period, last_exclamation, last_question)
                if sentence_end > chunk_size * 0.5:  # Only if we're not cutting too much
                    chunk = chunk[:sentence_end + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    async def add_paper(self, paper: 'ResearchPaper', full_text: Optional[str] = None):
        """Add a research paper to the vector store"""
        
        # Prepare text content for embedding
        text_content = []
        
        # Add title and abstract
        title_abstract = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        text_content.append(title_abstract)
        
        # Add full text if available
        if full_text:
            chunks = self._chunk_text(full_text)
            text_content.extend(chunks)
        
        # Generate embeddings
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
                'categories': json.dumps(paper.categories) if paper.categories else json.dumps([])
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
        """Search for relevant paper chunks with improved logic"""
        
        similarity_threshold = similarity_threshold or self.config.get('similarity_threshold', 0.7)
        
        print(f"Vector store search - Query: '{query}', Max results: {max_results}, Threshold: {similarity_threshold}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"Raw results count: {len(results['documents'][0]) if results['documents'] else 0}")
            
            if not results['documents'] or not results['documents'][0]:
                print("No documents found in vector store")
                return []
            
            # Process results
            search_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                print(f"Document {i}: similarity={similarity:.3f}, distance={distance:.3f}")
                
                if similarity >= similarity_threshold:
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': similarity,
                        'distance': distance
                    }
                    search_results.append(result)
                    print(f"  -> Included (above threshold)")
                else:
                    print(f"  -> Excluded (below threshold {similarity_threshold})")
            
            print(f"Final results after filtering: {len(search_results)}")
            return search_results
            
        except Exception as e:
            logging.error(f"Error in vector store search: {e}")
            print(f"Vector store search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.settings['collection_name'],
            'embedding_model': self.settings['embedding_model']
        }
    
    async def delete_paper(self, paper_id: str):
        """Delete all chunks of a paper from the vector store"""
        # Find all chunks for this paper
        results = self.collection.get(
            where={"paper_id": paper_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logging.info(f"Deleted paper {paper_id} from vector store")