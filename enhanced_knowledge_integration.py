#!/usr/bin/env python3
"""
Enhanced Knowledge Integration System
Integrates comprehensive support data with video transcripts for complete support bot knowledge
"""

import os
import json
import time
import re
from typing import List, Dict, Optional
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedKnowledgeIntegrator:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str):
        """Initialize the enhanced knowledge integrator"""
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        
        # Initialize OpenAI with new API
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone with new API
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(pinecone_index)
            print("✅ Pinecone initialized successfully")
        except Exception as e:
            print(f"⚠️ Pinecone initialization failed: {e}")
            self.index = None
    
    def create_semantic_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Create semantic chunks with overlapping context for better retrieval
        
        Args:
            text: The text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text, start_pos, end_pos, and context
        """
        chunks = []
        
        # Split by sentences first to maintain semantic coherence
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        start_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                # Create chunk with context
                chunk_text = current_chunk.strip()
                
                # Add overlapping context from previous chunk
                context_before = ""
                context_after = ""
                
                if chunks:
                    # Get last 100 chars from previous chunk as context
                    prev_chunk = chunks[-1]['text']
                    context_before = prev_chunk[-100:] if len(prev_chunk) > 100 else prev_chunk
                
                # Get next sentence as context if available
                if i < len(sentences) - 1:
                    next_sentence = sentences[i + 1].strip()
                    context_after = next_sentence[:100] if next_sentence else ""
                
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(chunk_text),
                    'context_before': context_before,
                    'context_after': context_after,
                    'sentence_count': len([s for s in chunk_text.split('.') if s.strip()]),
                    'word_count': len(chunk_text.split())
                })
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
                current_chunk = overlap_text + " " + sentence
                start_pos = start_pos + len(chunk_text) - len(overlap_text)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append({
                'text': chunk_text,
                'start_pos': start_pos,
                'end_pos': start_pos + len(chunk_text),
                'context_before': chunks[-1]['text'][-100:] if chunks else "",
                'context_after': "",
                'sentence_count': len([s for s in chunk_text.split('.') if s.strip()]),
                'word_count': len(chunk_text.split())
            })
        
        return chunks

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return None
    
    def store_embedding(self, embedding: List[float], metadata: Dict, company_name: str) -> bool:
        """Store embedding in Pinecone"""
        if not self.index:
            print("⚠️ Pinecone not available, skipping storage")
            return False
        
        try:
            # Ensure metadata values are strings for Pinecone
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    clean_metadata[key] = json.dumps(value)
                else:
                    clean_metadata[key] = str(value)
            
            # Add company name to metadata
            clean_metadata['company_name'] = company_name
            
            # Generate unique ID
            vector_id = f"{company_name}_{int(time.time() * 1000)}"
            
            self.index.upsert(
                vectors=[{
                    'id': vector_id,
                    'values': embedding,
                    'metadata': clean_metadata
                }]
            )
            return True
        except Exception as e:
            print(f"❌ Error storing embedding: {e}")
            return False
    
    def store_semantic_chunks(self, text: str, company_name: str, source_info: Dict) -> List[Dict]:
        """
        Store text as semantic chunks with enhanced metadata
        
        Args:
            text: The text to store
            company_name: Company name for namespace
            source_info: Additional source information
            
        Returns:
            List of stored chunk information
        """
        if not self.index:
            print("⚠️ Pinecone not available, skipping storage")
            return []
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(text)
        stored_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding for the chunk text
                embedding = self.create_embedding(chunk['text'])
                if not embedding:
                    continue
                
                # Enhanced metadata for better retrieval
                metadata = {
                    'text': chunk['text'],
                    'context_before': chunk['context_before'],
                    'context_after': chunk['context_after'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos'],
                    'sentence_count': chunk['sentence_count'],
                    'word_count': chunk['word_count'],
                    'chunk_type': 'semantic',
                    'company_name': company_name,
                    **source_info  # Include source information
                }
                
                # Store in Pinecone
                success = self.store_embedding(embedding, metadata, company_name)
                if success:
                    stored_chunks.append({
                        'chunk_index': i,
                        'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text'],
                        'word_count': chunk['word_count'],
                        'sentence_count': chunk['sentence_count']
                    })
                
            except Exception as e:
                print(f"❌ Error storing chunk {i}: {e}")
                continue
        
        return stored_chunks
    
    def search_knowledge(self, query_embedding: List[float], company_name: str, top_k: int = 5) -> List:
        """Search knowledge base for relevant content"""
        if not self.index:
            print("⚠️ Pinecone not available, falling back to text search")
            return []
        
        try:
            results = self.index.query(
                vector=query_embedding,
                filter={"company_name": company_name},
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            print(f"❌ Error searching knowledge: {e}")
            return []
    
    def search_with_context(self, query: str, company_name: str, top_k: int = 5) -> List[Dict]:
        """
        Search with enhanced context retrieval
        
        Args:
            query: Search query
            company_name: Company name
            top_k: Number of results to return
            
        Returns:
            List of results with enhanced context
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        if not query_embedding:
            return []
        
        # Search for relevant chunks
        matches = self.search_knowledge(query_embedding, company_name, top_k)
        
        # Enhance results with context
        enhanced_results = []
        for match in matches:
            metadata = match.metadata
            
            # Reconstruct full context
            full_context = ""
            if metadata.get('context_before'):
                full_context += metadata['context_before'] + " "
            full_context += metadata.get('text', '')
            if metadata.get('context_after'):
                full_context += " " + metadata['context_after']
            
            enhanced_results.append({
                'id': match.id,
                'score': match.score,
                'text': metadata.get('text', ''),
                'full_context': full_context.strip(),
                'chunk_index': metadata.get('chunk_index', 0),
                'total_chunks': metadata.get('total_chunks', 1),
                'word_count': metadata.get('word_count', 0),
                'sentence_count': metadata.get('sentence_count', 0),
                'metadata': metadata  # Include the full metadata
            })
        
        return enhanced_results
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful support assistant. Provide clear, accurate, and complete answers based on the information provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I'm sorry, I couldn't generate an answer at this time."
    
    def cleanup(self):
        """Cleanup resources"""
        pass

    def hybrid_search(self, query: str, company_name: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword-based retrieval
        
        Args:
            query: Search query
            company_name: Company name
            top_k: Number of results to return
            
        Returns:
            List of enhanced results
        """
        try:
            # Strategy 1: Semantic search
            semantic_results = self.search_with_context(query, company_name, top_k)
            
            # Strategy 2: Keyword-based search (if we have keyword matching)
            keyword_results = self._keyword_search(query, company_name, top_k)
            
            # Strategy 3: Exact phrase matching
            phrase_results = self._phrase_search(query, company_name, top_k)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, phrase_results, top_k
            )
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            # Fallback to semantic search
            return self.search_with_context(query, company_name, top_k)
    
    def _keyword_search(self, query: str, company_name: str, top_k: int) -> List[Dict]:
        """Keyword-based search using query expansion"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Create embeddings for each keyword
            keyword_embeddings = []
            for keyword in keywords:
                embedding = self.create_embedding(keyword)
                if embedding:
                    keyword_embeddings.append(embedding)
            
            if not keyword_embeddings:
                return []
            
            # Search for each keyword
            all_results = []
            for embedding in keyword_embeddings:
                results = self.search_knowledge(embedding, company_name, top_k=3)
                # Convert Pinecone match objects to dictionaries
                for match in results:
                    if hasattr(match, 'metadata'):
                        all_results.append({
                            'id': getattr(match, 'id', ''),
                            'score': getattr(match, 'score', 0),
                            'metadata': match.metadata
                        })
            
            # Deduplicate and rank
            return self._deduplicate_results(all_results)
            
        except Exception as e:
            print(f"❌ Error in keyword search: {e}")
            return []
    
    def _phrase_search(self, query: str, company_name: str, top_k: int) -> List[Dict]:
        """Exact phrase matching search"""
        try:
            # For exact phrase matching, we'd need to implement text-based search
            # This is a placeholder for future implementation
            return []
            
        except Exception as e:
            print(f"❌ Error in phrase search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction - in production, use NLP libraries
        import re
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _combine_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict], 
                               phrase_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine and rank search results from different strategies"""
        try:
            # Create a scoring system
            all_results = {}
            
            # Add semantic results with high weight
            for result in semantic_results:
                if not isinstance(result, dict):
                    continue
                result_id = result.get('id', '')
                if result_id not in all_results:
                    all_results[result_id] = result.copy()
                    all_results[result_id]['combined_score'] = result.get('score', 0) * 0.6  # 60% weight
                else:
                    all_results[result_id]['combined_score'] += result.get('score', 0) * 0.6
            
            # Add keyword results with medium weight
            for result in keyword_results:
                if not isinstance(result, dict):
                    continue
                result_id = result.get('id', '')
                if result_id not in all_results:
                    all_results[result_id] = result.copy()
                    all_results[result_id]['combined_score'] = result.get('score', 0) * 0.3  # 30% weight
                else:
                    all_results[result_id]['combined_score'] += result.get('score', 0) * 0.3
            
            # Add phrase results with high weight for exact matches
            for result in phrase_results:
                if not isinstance(result, dict):
                    continue
                result_id = result.get('id', '')
                if result_id not in all_results:
                    all_results[result_id] = result.copy()
                    all_results[result_id]['combined_score'] = result.get('score', 0) * 0.8  # 80% weight for exact matches
                else:
                    all_results[result_id]['combined_score'] += result.get('score', 0) * 0.8
            
            # Sort by combined score
            sorted_results = sorted(
                all_results.values(), 
                key=lambda x: x.get('combined_score', 0), 
                reverse=True
            )
            
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"❌ Error combining search results: {e}")
            return semantic_results[:top_k]  # Fallback to semantic results
    
    def _deduplicate_results(self, results: List) -> List[Dict]:
        """Remove duplicate results based on ID"""
        try:
            seen_ids = set()
            unique_results = []
            
            for result in results:
                # Handle both dictionary and object results
                if isinstance(result, dict):
                    result_id = result.get('id', '')
                else:
                    result_id = getattr(result, 'id', '') if hasattr(result, 'id') else ''
                
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            print(f"❌ Error deduplicating results: {e}")
            return results
