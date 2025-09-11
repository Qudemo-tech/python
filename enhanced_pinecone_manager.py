#!/usr/bin/env python3
"""
Enhanced Pinecone Manager for Standard Plan
Multi-index architecture with optimized performance
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any
from pinecone import Pinecone
from openai import OpenAI
import asyncio
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedPineconeManager:
    """Enhanced Pinecone Manager with multi-index architecture for Standard Plan"""
    
    def __init__(self):
        """Initialize enhanced Pinecone manager with Standard Plan features"""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Multi-index architecture for Standard Plan
        self.indexes = {
            'knowledge': 'qudemo-knowledge-index',
            'video': 'qudemo-video-index',
            'analytics': 'qudemo-analytics-index',
            'legacy': 'qudemo-index'  # Add legacy index for backward compatibility
        }
        
        # Index status tracking
        self.index_status = {}
        
        # Embedding cache for performance optimization
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000  # Maximum number of cached embeddings
        
        # Index configurations for Standard Plan
        self.index_configs = {
            'knowledge': {
                'dimension': 3072,  # text-embedding-3-large
                'metric': 'cosine',
                'spec': {
                    'serverless': {
                        'cloud': 'aws',
                        'region': 'us-east-1'
                    }
                }
            },
            'video': {
                'dimension': 3072,  # text-embedding-3-large
                'metric': 'cosine',
                'spec': {
                    'serverless': {
                        'cloud': 'aws',
                        'region': 'us-east-1'
                    }
                }
            },
            'analytics': {
                'dimension': 1536,  # text-embedding-3-small for analytics
                'metric': 'cosine',
                'spec': {
                    'serverless': {
                        'cloud': 'aws',
                        'region': 'us-east-1'
                    }
                }
            },
            'legacy': {
                'dimension': 1536,  # OpenAI embedding dimension
                'metric': 'cosine',
                'spec': {
                    'serverless': {
                        'cloud': 'aws',
                        'region': 'us-east-1'
                    }
                }
            }
        }
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Rate limiting for OpenAI API
        self.last_embedding_time = 0
        self.min_embedding_interval = 0.1  # 100ms between embedding requests
        
        # Performance monitoring
        self.performance_metrics = {
            'query_times': [],
            'storage_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize indexes
        self._initialize_indexes()
        
        logger.info("✅ Enhanced Pinecone Manager initialized with Standard Plan features")
    
    def _initialize_indexes(self):
        """Initialize all indexes for Standard Plan with enhanced error handling"""
        try:
            for index_name, config in self.index_configs.items():
                index_full_name = self.indexes[index_name]
                
                # Check if index exists
                try:
                    index = self.pc.Index(index_full_name)
                    self.index_status[index_name] = 'exists'
                    logger.info(f"✅ Index {index_full_name} already exists")
                except Exception as e:
                    # Create new index
                    logger.info(f"🔧 Creating new index: {index_full_name}")
                    try:
                        self.pc.create_index(
                            name=index_full_name,
                            dimension=config['dimension'],
                            metric=config['metric'],
                            spec=config['spec']
                        )
                        self.index_status[index_name] = 'created'
                        logger.info(f"✅ Created index: {index_full_name}")
                        
                        # Wait for index to be ready
                        import time
                        time.sleep(5)
                        
                    except Exception as create_error:
                        logger.error(f"❌ Failed to create index {index_full_name}: {create_error}")
                        self.index_status[index_name] = 'failed'
                        
                        # For video index, try fallback to legacy index
                        if index_name == 'video':
                            logger.warning(f"⚠️ Video index creation failed, will use legacy index as fallback")
                            self.index_status[index_name] = 'fallback'
                    
        except Exception as e:
            logger.error(f"❌ Error initializing indexes: {e}")
            raise
    
    def get_index(self, index_type: str):
        """Get Pinecone index by type"""
        if index_type not in self.indexes:
            raise ValueError(f"Invalid index type: {index_type}. Valid types: {list(self.indexes.keys())}")
        
        return self.pc.Index(self.indexes[index_type])
    
    def route_content_to_index(self, content_type: str, metadata: Dict) -> str:
        """Route content to optimal index based on type and metadata"""
        
        # Video content routing
        if content_type in ['video_transcript', 'youtube_transcript', 'loom_transcript', 'timestamp']:
            return 'video'
        
        # Knowledge content routing
        elif content_type in ['help_center', 'faq', 'documentation', 'web_scraping', 'article']:
            return 'knowledge'
        
        # Analytics content routing
        elif content_type in ['search_query', 'user_feedback', 'performance_metric']:
            return 'analytics'
        
        # Default to knowledge for unknown types
        else:
            return 'knowledge'
    
    async def generate_embedding(self, text: str, model: str = 'text-embedding-3-large') -> List[float]:
        """Generate embedding with caching and robust retry logic"""
        # Check cache first
        cache_key = self._generate_cache_key(text, model)
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            logger.debug(f"✅ Cache hit for embedding: {cache_key[:20]}...")
            return self.embedding_cache[cache_key]
        
        self.cache_misses += 1
        
        max_retries = 3
        base_delay = 2
        
        # Fallback models in case primary model fails
        fallback_models = ['text-embedding-3-small', 'text-embedding-ada-002']
        models_to_try = [model] + fallback_models
        
        # Rate limiting to prevent API overload
        current_time = time.time()
        time_since_last = current_time - self.last_embedding_time
        if time_since_last < self.min_embedding_interval:
            sleep_time = self.min_embedding_interval - time_since_last
            logger.debug(f"⏳ Rate limiting: waiting {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        for model_to_try in models_to_try:
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    # Use direct synchronous call instead of asyncio.to_thread to avoid connection issues
                    response = self.openai_client.embeddings.create(
                        model=model_to_try,
                        input=text,
                        timeout=30  # 30 second timeout
                    )
                    
                    embedding = response.data[0].embedding
                    embedding_time = time.time() - start_time
                    
                    # Update rate limiting timestamp
                    self.last_embedding_time = time.time()
                    
                    # Track performance
                    self.performance_metrics['query_times'].append(embedding_time)
                    
                    # Cache the embedding
                    self._cache_embedding(cache_key, embedding)
                    
                    if model_to_try != model:
                        logger.info(f"✅ Embedding generated with fallback model {model_to_try}")
                    else:
                        logger.debug(f"✅ Embedding generated successfully with {model_to_try}")
                    return embedding
                    
                except Exception as e:
                    logger.warning(f"⚠️ Embedding attempt {attempt + 1} with {model_to_try} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + (hash(str(attempt)) % 2)
                        logger.info(f"⏳ Retrying {model_to_try} in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(f"⚠️ Model {model_to_try} failed after {max_retries} attempts, trying next model...")
                        break  # Try next model
        
        # If all models failed
        logger.error(f"❌ All embedding models failed: {models_to_try}")
        raise Exception(f"Failed to generate embedding with all models: {models_to_try}")
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate stable cache key for embedding"""
        try:
            import hashlib
            
            # Create stable hash using blake2b
            text_hash = hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()
            return f"{model}:{text_hash}"
            
        except Exception as e:
            logger.error(f"❌ Error generating cache key: {e}")
            # Fallback to simple hash
            return f"{model}:{hash(text)}"
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Cache embedding with LRU eviction"""
        try:
            # Check cache size and evict if necessary
            if len(self.embedding_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
                logger.debug(f"🗑️ Evicted embedding from cache: {oldest_key[:20]}...")
            
            # Add to cache
            self.embedding_cache[cache_key] = embedding
            logger.debug(f"💾 Cached embedding: {cache_key[:20]}...")
            
        except Exception as e:
            logger.error(f"❌ Error caching embedding: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.embedding_cache),
            'max_cache_size': self.max_cache_size
        }
    
    async def store_semantic_chunks(self, chunks: List[Dict], company_name: str, qudemo_id: str, 
                            content_type: str = 'web_scraping') -> Dict:
        """Store semantic chunks in optimal index with Standard Plan features"""
        try:
            start_time = time.time()
            logger.info(f"🔧 Storing {len(chunks)} chunks for {company_name} qudemo {qudemo_id}")
            
            # Route to optimal index
            target_index_type = self.route_content_to_index(content_type, {})
            target_index = self.get_index(target_index_type)
            
            # Create namespace for qudemo isolation
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Prepare vectors for batch upsert
            vectors_to_upsert = []
            
            successful_chunks = 0
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}: {chunk.get('title', 'Untitled')[:50]}...")
                    
                    # Generate embedding with retry logic
                    embedding = await self.generate_embedding(chunk['text'])
                    
                    # Prepare metadata with Standard Plan optimizations
                    metadata = {
                        'text': chunk['text'],
                        'content': chunk['text'],  # Also store as 'content' for compatibility
                        'source': chunk.get('source', 'unknown'),
                        'source_type': content_type,
                        'title': chunk.get('title', ''),
                        'url': chunk.get('url', ''),
                        'processed_at': chunk.get('processed_at', datetime.now().isoformat()),
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'chunk_type': 'semantic',
                        'chunk_index': chunk.get('chunk_index', i),
                        'total_chunks': len(chunks),
                        'quality_score': chunk.get('quality_score', 85),
                        'difficulty_level': chunk.get('difficulty_level', 'intermediate'),
                        'content_category': chunk.get('content_category', 'general'),
                        'has_steps': chunk.get('has_steps', False),
                        'is_complete': chunk.get('is_complete', True),
                        'word_count': chunk.get('word_count', 0),
                        'content_has_text': True  # Data contract flag
                    }
                    
                    # Add video-specific metadata
                    if content_type in ['video_transcript', 'youtube_transcript', 'loom_transcript']:
                        metadata.update({
                            'start_timestamp': chunk.get('start_timestamp', 0),
                            'end_timestamp': chunk.get('end_timestamp', 0),
                            'video_url': chunk.get('video_url', ''),
                            'video_type': chunk.get('video_type', 'unknown')
                        })
                    
                    # Create vector record
                    vector_record = {
                        'id': f"{namespace}-{content_type}-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'values': embedding,
                        'metadata': metadata
                    }
                    
                    vectors_to_upsert.append(vector_record)
                    successful_chunks += 1
                    logger.info(f"✅ Chunk {i+1} processed successfully")
                    
                except Exception as e:
                    failed_chunks += 1
                    logger.error(f"❌ Error processing chunk {i+1}: {e}")
                    continue
            
            logger.info(f"📊 Chunk processing summary: {successful_chunks} successful, {failed_chunks} failed")
            
            if vectors_to_upsert:
                # Batch upsert to Pinecone
                target_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                
                storage_time = time.time() - start_time
                self.performance_metrics['storage_times'].append(storage_time)
                
                logger.info(f"✅ Stored {len(vectors_to_upsert)} chunks in {target_index_type} index, namespace: {namespace}")
                
                return {
                    'success': True,
                    'chunks_stored': len(vectors_to_upsert),
                    'index_type': target_index_type,
                    'namespace': namespace,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'storage_time': storage_time
                }
            else:
                logger.warning("⚠️ No chunks to store")
                return {
                    'success': False,
                    'error': 'No chunks to store',
                    'chunks_stored': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error storing semantic chunks: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_stored': 0
            }
    
    async def search_with_context(self, query: str, company_name: str, qudemo_id: str, 
                           content_types: List[str] = None, top_k: int = 5) -> Dict:
        """Search across multiple indexes with context-aware routing"""
        try:
            start_time = time.time()
            
            # Default to searching all content types
            if content_types is None:
                content_types = ['knowledge', 'video']
            
            # Map content types to indexes and ensure legacy index is included for video content
            mapped_content_types = []
            for content_type in content_types:
                if content_type == 'video_transcript':
                    mapped_content_types.extend(['video', 'legacy'])
                elif content_type == 'knowledge':
                    mapped_content_types.append('knowledge')
                elif content_type == 'analytics':
                    mapped_content_types.append('analytics')
                else:
                    # For unknown content types, search in all indexes
                    mapped_content_types.extend(['knowledge', 'video', 'analytics', 'legacy'])
            
            # Remove duplicates and use mapped types
            content_types = list(set(mapped_content_types))
            
            all_results = []
            
            for index_type in content_types:
                try:
                    index = self.get_index(index_type)
                    namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
                    
                    # Generate query embedding
                    query_embedding = await self.generate_embedding(query)
                    
                    # Search in specific index
                    # Use different filters based on index type
                    if index_type == 'legacy':
                        # Legacy index uses 'company' instead of 'company_name'
                        filter_query = {
                            'company': {'$eq': company_name}
                        }
                    else:
                        # New indexes use 'company_name' and 'qudemo_id'
                        filter_query = {
                            'company_name': {'$eq': company_name},
                            'qudemo_id': {'$eq': qudemo_id}
                        }
                    
                    results = index.query(
                        vector=query_embedding,
                        namespace=namespace,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter_query
                    )
                    
                    # Add index type to results
                    for match in results.matches:
                        match.metadata['index_type'] = index_type
                        all_results.append(match)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Search failed for index {index_type}: {e}")
                    continue
            
            # Sort all results by score
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Take top_k results
            final_results = all_results[:top_k]
            
            search_time = time.time() - start_time
            self.performance_metrics['query_times'].append(search_time)
            
            return {
                'success': True,
                'results': final_results,
                'total_found': len(all_results),
                'search_time': search_time,
                'searched_indexes': content_types
            }
            
        except Exception as e:
            logger.error(f"❌ Error in search with context: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_knowledge_summary(self, company_name: str, qudemo_id: str) -> Dict:
        """Get comprehensive knowledge summary across all indexes"""
        try:
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            summary = {
                'knowledge_chunks': 0,
                'video_chunks': 0,
                'analytics_chunks': 0,
                'legacy_chunks': 0,
                'total_chunks': 0,
                'content_distribution': {},
                'quality_metrics': {}
            }
            
            # Check each index
            for index_type, index_name in self.indexes.items():
                try:
                    index = self.pc.Index(index_name)
                    
                    # Get namespace stats
                    stats = index.describe_index_stats()
                    total_vectors = stats.get('total_vector_count', 0)
                    
                    if total_vectors > 0:
                        # Query namespace to get count
                        dummy_embedding = [0.0] * self.index_configs[index_type]['dimension']
                        results = index.query(
                            vector=dummy_embedding,
                            top_k=1,
                            include_metadata=True,
                            namespace=namespace
                        )
                        
                        chunk_count = len(results.matches) if results.matches else 0
                        summary[f'{index_type}_chunks'] = chunk_count
                        summary['total_chunks'] += chunk_count
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error getting stats for {index_type}: {e}")
                    continue
            
            return {
                'success': True,
                'data': summary
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting knowledge summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for optimization"""
        try:
            avg_query_time = sum(self.performance_metrics['query_times']) / len(self.performance_metrics['query_times']) if self.performance_metrics['query_times'] else 0
            avg_storage_time = sum(self.performance_metrics['storage_times']) / len(self.performance_metrics['storage_times']) if self.performance_metrics['storage_times'] else 0
            
            return {
                'success': True,
                'metrics': {
                    'avg_query_time': avg_query_time,
                    'avg_storage_time': avg_storage_time,
                    'total_queries': len(self.performance_metrics['query_times']),
                    'total_storage_ops': len(self.performance_metrics['storage_times']),
                    'cache_hit_rate': self.performance_metrics['cache_hits'] / (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_cache(self):
        """Clean up performance metrics cache"""
        self.performance_metrics = {
            'query_times': [],
            'storage_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info("🧹 Performance metrics cache cleaned")

# Global instance
_enhanced_pinecone_manager = None

def initialize_enhanced_pinecone_manager() -> bool:
    """Initialize the enhanced Pinecone manager"""
    global _enhanced_pinecone_manager
    try:
        _enhanced_pinecone_manager = EnhancedPineconeManager()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Pinecone Manager: {e}")
        return False

def get_enhanced_pinecone_manager() -> EnhancedPineconeManager:
    """Get the global enhanced Pinecone manager instance"""
    if _enhanced_pinecone_manager is None:
        raise RuntimeError("Enhanced Pinecone Manager not initialized. Call initialize_enhanced_pinecone_manager() first.")
    return _enhanced_pinecone_manager
