import asyncio
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class RAGResult:
    """Result from RAG query"""
    answer: str
    sources: List[Dict]
    confidence: float
    query: str
    context_used: str

class RAGEngine:
    """RAG engine for querying research papers"""
    
    def __init__(self, vector_store: 'VectorStore', config: Dict):
        self.vector_store = vector_store
        self.config = config
        self.max_context_length = config.get('max_context_length', 4000)
        self.max_retrieved_chunks = config.get('max_retrieved_chunks', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
    
    async def query(
        self, 
        question: str, 
        context_filter: Optional[Dict] = None,
        max_results: int = None
    ) -> RAGResult:
        """Query the research paper database"""
        
        max_results = max_results or self.max_retrieved_chunks
        
        # 1. Retrieve relevant chunks
        search_results = await self.vector_store.search(
            query=question,
            max_results=max_results,
            similarity_threshold=self.similarity_threshold
        )
        
        if not search_results:
            return RAGResult(
                answer="I couldn't find any relevant research papers to answer your question.",
                sources=[],
                confidence=0.0,
                query=question,
                context_used=""
            )
        
        # 2. Prepare context
        context_parts = []
        sources = []
        current_length = 0
        
        for result in search_results:
            # Add source information
            metadata = result['metadata']
            source_info = {
                'title': metadata.get('title', ''),
                'authors': metadata.get('authors', ''),
                'paper_id': metadata.get('paper_id', ''),
                'similarity': result['similarity'],
                'chunk_type': metadata.get('chunk_type', ''),
                'url': metadata.get('url', '')
            }
            sources.append(source_info)
            
            # Add to context if we have space
            chunk_text = result['document']
            if current_length + len(chunk_text) < self.max_context_length:
                context_parts.append(f"Source: {metadata.get('title', 'Unknown')}\n{chunk_text}")
                current_length += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 3. Generate answer (using the context)
        answer = await self._generate_answer(question, context, sources)
        
        # 4. Calculate confidence
        avg_similarity = sum(r['similarity'] for r in search_results) / len(search_results)
        confidence = min(avg_similarity, 1.0)
        
        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=question,
            context_used=context
        )
    
    async def _generate_answer(
        self, 
        question: str, 
        context: str, 
        sources: List[Dict]
    ) -> str:
        """Generate an answer based on the retrieved context"""
        
        # This is a simplified version - in practice, you might want to:
        # 1. Use a language model (OpenAI, Anthropic, local model)
        # 2. Apply more sophisticated prompt engineering
        # 3. Implement answer quality scoring
        
        if not context:
            return "I couldn't find relevant information to answer your question."
        
        # For now, return a formatted response with the relevant information
        source_titles = [s['title'] for s in sources if s['title']]
        unique_titles = list(dict.fromkeys(source_titles))  # Remove duplicates while preserving order
        
        answer_parts = [
            f"Based on the research papers I found, here's what I can tell you about your question: '{question}'",
            "",
            "Key findings from the literature:",
            context[:1000] + "..." if len(context) > 1000 else context,
            "",
            f"This information comes from {len(unique_titles)} research paper(s):",
        ]
        
        for i, title in enumerate(unique_titles[:3], 1):  # Show max 3 titles
            answer_parts.append(f"{i}. {title}")
        
        if len(unique_titles) > 3:
            answer_parts.append(f"... and {len(unique_titles) - 3} more papers")
        
        return "\n".join(answer_parts)
    
    async def get_paper_summary(self, paper_id: str) -> Optional[Dict]:
        """Get a summary of a specific paper"""
        
        # Search for all chunks of this paper
        results = await self.vector_store.collection.get(
            where={"paper_id": paper_id},
            include=['documents', 'metadatas']
        )
        
        if not results['documents']:
            return None
        
        # Find the title/abstract chunk
        title_abstract_chunk = None
        content_chunks = []
        
        for i, metadata in enumerate(results['metadatas']):
            if metadata.get('chunk_type') == 'title_abstract':
                title_abstract_chunk = results['documents'][i]
            else:
                content_chunks.append(results['documents'][i])
        
        summary = {
            'paper_id': paper_id,
            'title_abstract': title_abstract_chunk,
            'content_chunks': len(content_chunks),
            'metadata': results['metadatas'][0] if results['metadatas'] else {}
        }
        
        return summary