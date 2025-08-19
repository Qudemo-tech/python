#!/usr/bin/env python3
"""
Enhanced Knowledge Integration System
Handles Pinecone vector database operations for knowledge storage and retrieval
"""

import os
import logging
from typing import List, Dict, Optional, Any
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class EnhancedKnowledgeIntegrator:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str = "qudemo-knowledge"):
        """Initialize the Enhanced Knowledge Integrator"""
        try:
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.pinecone_environment = "us-east-1"  # Default environment
            self.index_name = pinecone_index or "qudemo-knowledge"
            
            # Create or connect to index
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            
            logger.info("✅ Enhanced Knowledge Integrator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Knowledge Integrator: {e}")
            raise
    
    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_environment
                    )
                )
                logger.info(f"✅ Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"✅ Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"❌ Error ensuring index exists: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            raise
    
    def store_semantic_chunks(self, chunks: List[Dict], company_name: str) -> Dict:
        """Store semantic chunks in Pinecone"""
        try:
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = self.generate_embedding(chunk.get('text', ''))
                
                # Create metadata
                metadata = {
                    'company_name': company_name,
                    'chunk_index': i,
                    'text': chunk.get('text', ''),
                    'full_context': chunk.get('full_context', ''),
                    'source': chunk.get('source', 'web_scraping'),
                    'title': chunk.get('title', 'Unknown'),
                    'url': chunk.get('url', ''),
                    'processed_at': chunk.get('processed_at', ''),
                    'chunk_id': f"{company_name}_{i}"
                }
                
                # Add to vectors list
                vectors_to_upsert.append({
                    'id': f"{company_name}_{i}_{hash(chunk.get('text', ''))}",
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"✅ Stored {len(vectors_to_upsert)} semantic chunks for {company_name}")
            
            return {
                'success': True,
                'chunks_stored': len(vectors_to_upsert),
                'company_name': company_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing semantic chunks: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_with_context(self, query: str, company_name: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant chunks with context"""
        try:
            # Generate embedding for query
            if query:
                query_embedding = self.generate_embedding(query)
            else:
                # For empty queries, use a neutral embedding
                query_embedding = [0.0] * 1536
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={'company_name': company_name}
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'full_context': match.metadata.get('full_context', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'metadata': match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching with context: {e}")
            return []
    
    def delete_company_knowledge(self, company_name: str) -> Dict:
        """Delete all knowledge for a company"""
        try:
            # Delete all vectors for the company
            self.index.delete(filter={'company_name': company_name})
            
            logger.info(f"✅ Deleted all knowledge for company: {company_name}")
            
            return {
                'success': True,
                'company_name': company_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error deleting company knowledge: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_company_stats(self, company_name: str) -> Dict:
        """Get statistics for a company's knowledge"""
        try:
            # Query to get all chunks for the company
            results = self.search_with_context("", company_name, top_k=10000)
            
            return {
                'success': True,
                'company_name': company_name,
                'total_chunks': len(results),
                'last_updated': max([r.get('metadata', {}).get('processed_at', '') for r in results], default='Unknown')
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting company stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }
