import asyncio
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Result from RAG query"""
    answer: str
    sources: List[Dict]
    confidence: float
    query: str
    context_used: str

class RAGEngine:
    """Enhanced RAG engine for querying research papers"""
    
    def __init__(self, vector_store: 'VectorStore', config: Dict):
        self.vector_store = vector_store
        self.config = config
        self.max_context_length = config.get('max_context_length', 8000)
        self.max_retrieved_chunks = config.get('max_retrieved_chunks', 12)
        self.similarity_threshold = config.get('similarity_threshold', 0.05)  # Very low threshold
    
    async def query(
        self, 
        question: str, 
        context_filter: Optional[Dict] = None,
        max_results: int = None
    ) -> RAGResult:
        """Query the research paper database with enhanced processing"""
        
        max_results = max_results or self.max_retrieved_chunks
        
        logger.info("Enhanced RAG search for: %s", question)
        logger.info("Using similarity threshold: %s", self.similarity_threshold)
        
        # 1. Retrieve relevant chunks with multiple attempts
        search_results = await self.vector_store.search(
            query=question,
            max_results=max_results * 2,  # Get more results initially
            similarity_threshold=self.similarity_threshold
        )
        
        logger.info("Found %s relevant chunks", len(search_results))
        
        # Try with even lower threshold if no results
        if not search_results:
            logger.info("Retrying with ultra-low threshold...")
            search_results = await self.vector_store.search(
                query=question,
                max_results=max_results * 2,
                similarity_threshold=0.01  # Ultra-low threshold
            )
            logger.info("Retry found %s chunks", len(search_results))
        
        if not search_results:
            return RAGResult(
                answer="I couldn't find any relevant research papers to answer your question. This could mean:\n1. No papers in the database discuss this topic\n2. Try searching for papers on this topic first\n3. The similarity threshold might be too strict",
                sources=[],
                confidence=0.0,
                query=question,
                context_used=""
            )
        
        # 2. Prepare context and sources with enhanced processing
        context_parts = []
        sources = []
        current_length = 0
        
        # Sort results by similarity
        search_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        for result in search_results[:max_results]:
            # Add source information
            metadata = result['metadata']
            source_info = {
                'title': metadata.get('title', 'Unknown Title'),
                'authors': json.loads(metadata.get('authors', '[]')) if metadata.get('authors') else [],
                'paper_id': metadata.get('paper_id', ''),
                'similarity': round(result['similarity'], 3),
                'chunk_type': metadata.get('chunk_type', ''),
                'url': metadata.get('url', ''),
                'categories': json.loads(metadata.get('categories', '[]')) if metadata.get('categories') else [],
                'relevance_score': metadata.get('relevance_score', 0)
            }
            sources.append(source_info)
            
            # Add to context if we have space
            chunk_text = result['document']
            chunk_header = f"\n=== From: {source_info['title']} (Similarity: {source_info['similarity']}) ===\n"
            full_chunk = chunk_header + chunk_text
            
            if current_length + len(full_chunk) < self.max_context_length:
                context_parts.append(full_chunk)
                current_length += len(full_chunk)
        
        context = "\n".join(context_parts)
        
        # 3. Generate enhanced answer
        answer = await self._generate_enhanced_answer(question, context, sources, search_results)
        
        # 4. Calculate confidence based on similarity and number of sources
        if search_results:
            avg_similarity = sum(r['similarity'] for r in search_results[:max_results]) / min(len(search_results), max_results)
            # Boost confidence based on number of sources found
            source_bonus = min(len(sources) / 5.0, 0.3)  # Up to 30% bonus for having multiple sources
            confidence = min(avg_similarity + source_bonus, 1.0)
        else:
            confidence = 0.0
        
        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=question,
            context_used=context[:500] + "..." if len(context) > 500 else context
        )
    
    async def _generate_enhanced_answer(
        self, 
        question: str, 
        context: str, 
        sources: List[Dict],
        search_results: List[Dict]
    ) -> str:
        """Generate a comprehensive answer based on the retrieved context"""
        
        if not context or not sources:
            return f"I couldn't find relevant information to answer '{question}'. Try adding more papers to the database on this topic."
        
        # Extract key information
        unique_papers = {}
        for source in sources:
            paper_id = source['paper_id']
            if paper_id not in unique_papers:
                unique_papers[paper_id] = {
                    'title': source['title'],
                    'authors': source['authors'][:2] if source['authors'] else ['Unknown'],
                    'similarity': source['similarity'],
                    'categories': source['categories'][:2] if source['categories'] else [],
                    'relevance_score': source.get('relevance_score', 0)
                }
        
        # Generate structured answer
        answer_parts = [
            f"Based on {len(search_results)} relevant sections from {len(unique_papers)} research paper(s), here's what I found about: '{question}'",
            ""
        ]
        
        # Add high-confidence findings
        high_confidence_results = [r for r in search_results if r['similarity'] > 0.3]
        if high_confidence_results:
            answer_parts.append("🔍 **High-Confidence Findings:**")
            for i, result in enumerate(high_confidence_results[:2], 1):
                content = result['document'][:300] + "..." if len(result['document']) > 300 else result['document']
                title = result['metadata'].get('title', 'Unknown')
                answer_parts.append(f"{i}. From '{title[:50]}...': {content}")
            answer_parts.append("")
        
        # Add medium-confidence findings if no high-confidence ones
        elif search_results:
            answer_parts.append("📋 **Key Information Found:**")
            for i, result in enumerate(search_results[:3], 1):
                content = result['document'][:250] + "..." if len(result['document']) > 250 else result['document']
                title = result['metadata'].get('title', 'Unknown')
                similarity = result['similarity']
                answer_parts.append(f"{i}. From '{title[:50]}...' (sim: {similarity:.2f}): {content}")
            answer_parts.append("")
        
        # Add contextual summary
        if context:
            # Extract the most relevant sentences that answer the question
            sentences = []
            for part in context.split('.'):
                part = part.strip()
                if len(part) > 20:  # Skip very short fragments
                    sentences.append(part)
            
            # Find sentences that contain question keywords
            question_words = set(question.lower().split())
            relevant_sentences = []
            
            for sentence in sentences[:10]:  # Check first 10 sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
            
            # Sort by relevance and take top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                answer_parts.append("💡 **Direct Answers from Research:**")
                for sentence, overlap in relevant_sentences[:2]:
                    if len(sentence) > 30:  # Skip very short sentences
                        clean_sentence = sentence.replace('\n', ' ').strip()
                        answer_parts.append(f"• {clean_sentence}.")
                answer_parts.append("")
        
        # Add paper references
        answer_parts.append(f"📚 **Source Papers ({len(unique_papers)}):**")
        for i, (paper_id, info) in enumerate(unique_papers.items(), 1):
            authors_str = ", ".join(info['authors']) if info['authors'] else "Unknown authors"
            categories_str = ", ".join(info['categories']) if info['categories'] else ""
            category_info = f" [{categories_str}]" if categories_str else ""
            relevance_info = f" (relevance: {info['relevance_score']:.1f})" if info['relevance_score'] > 0 else ""
            
            answer_parts.append(f"{i}. {info['title']}")
            answer_parts.append(f"   Authors: {authors_str}{category_info}{relevance_info}")
            answer_parts.append(f"   Similarity: {info['similarity']:.3f}")
        
        # Add guidance based on confidence
        avg_similarity = sum(s['similarity'] for s in sources) / len(sources) if sources else 0
        
        if avg_similarity < 0.15:
            answer_parts.append("\n⚠️ **Note:** The similarity scores are quite low, which suggests the papers may not directly address your specific question. Consider:")
            answer_parts.append("• Rephrasing your question with different terms")
            answer_parts.append("• Searching for more papers specifically on this topic")
            answer_parts.append("• Asking a more general question about the broader topic")
        elif avg_similarity < 0.25:
            answer_parts.append("\n💭 **Note:** Moderate relevance found. The answer is based on somewhat related content from the research papers.")
        else:
            answer_parts.append("\n✅ **Note:** High relevance found. This answer is based on directly relevant content from the research papers.")
        
        return "\n".join(answer_parts)
    
    def get_paper_summary(self, paper_id: str) -> Optional[Dict]:
            try:
                # Get all chunks for this paper from vector store
                collection = self.vector_store.collection
    
                results = collection.get(
                    where={"paper_id": paper_id},
                    include=['documents', 'metadatas']
                )
    
                if not results['documents']:
                    return None
    
                # Find the title/abstract chunk
                title_abstract_chunk = None
                content_chunks = []
                metadata = None
    
                for i, doc_metadata in enumerate(results['metadatas']):
                    if doc_metadata.get('chunk_type') == 'title_abstract':
                        title_abstract_chunk = results['documents'][i]
                        metadata = doc_metadata
                    else:
                        content_chunks.append(results['documents'][i])
    
                summary = {
                    'paper_id': paper_id,
                    'title_abstract': title_abstract_chunk,
                    'content_chunks': len(content_chunks),
                    'metadata': (
                        metadata if metadata 
                        else (results['metadatas'][0] if results['metadatas'] else {})
                    )
                }
    
                return summary
    
            except Exception as e:
                logging.error(f"Error getting paper summary: {e}")
                return None
