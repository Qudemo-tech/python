#!/usr/bin/env python3
"""
Enhanced Knowledge Integration with Standard Plan Multi-Index Architecture
Optimized for better Q&A performance and content routing
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import the enhanced Pinecone manager
from enhanced_pinecone_manager import get_enhanced_pinecone_manager

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedKnowledgeIntegrator:
    """Enhanced Knowledge Integrator with Standard Plan optimizations"""
    
    def __init__(self):
        """Initialize enhanced knowledge integrator"""
        try:
            self.pinecone_manager = get_enhanced_pinecone_manager()
            logger.info("✅ Enhanced Knowledge Integrator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Knowledge Integrator: {e}")
            raise
    
    async def store_semantic_chunks(self, chunks: List[Dict], company_name: str, qudemo_id: str) -> Dict:
        """Store semantic chunks with intelligent content routing"""
        try:
            logger.info(f"🔧 Storing {len(chunks)} chunks for {company_name} qudemo {qudemo_id}")
            
            # Process chunks with enhanced metadata
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Enhance chunk metadata
                    enhanced_chunk = self._enhance_chunk_metadata(chunk, i, len(chunks))
                    enhanced_chunks.append(enhanced_chunk)
                    
                except Exception as e:
                    logger.error(f"❌ Error enhancing chunk {i}: {e}")
                    continue
            
            if not enhanced_chunks:
                return {
                    'success': False,
                    'error': 'No chunks to process',
                    'chunks_stored': 0
                }
            
            # Store chunks using enhanced Pinecone manager
            store_result = await self.pinecone_manager.store_semantic_chunks(
                chunks=enhanced_chunks,
                company_name=company_name,
                qudemo_id=qudemo_id,
                content_type='web_scraping'
            )
            
            if store_result['success']:
                logger.info(f"✅ Successfully stored {store_result['chunks_stored']} chunks")
                return store_result
            else:
                logger.error(f"❌ Failed to store chunks: {store_result.get('error', 'Unknown error')}")
                return store_result
                
        except Exception as e:
            logger.error(f"❌ Error in store_semantic_chunks: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_stored': 0
            }
    
    def _enhance_chunk_metadata(self, chunk: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """Enhance chunk metadata for better search and Q&A"""
        try:
            text = chunk.get('text', '')
            
            # Analyze content for better categorization
            content_analysis = self._analyze_content(text)
            
            # Enhanced metadata
            enhanced_chunk = {
                'text': text,
                'source': chunk.get('source_url', chunk.get('source', 'unknown')),
                'source_type': chunk.get('source_type', 'web_scraping'),
                'title': chunk.get('title', ''),
                'url': chunk.get('url', ''),
                'processed_at': datetime.now().isoformat(),
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'quality_score': chunk.get('quality_score', 85),
                'difficulty_level': content_analysis['difficulty_level'],
                'content_category': content_analysis['content_category'],
                'has_steps': content_analysis['has_steps'],
                'is_complete': content_analysis['is_complete'],
                'word_count': len(text.split()),
                'keywords': content_analysis['keywords'],
                'summary': content_analysis['summary'],
                'content_type': content_analysis['content_type'],
                'target_audience': content_analysis['target_audience'],
                'prerequisites': content_analysis['prerequisites'],
                'estimated_time': content_analysis['estimated_time']
            }
            
            return enhanced_chunk
            
        except Exception as e:
            logger.error(f"❌ Error enhancing chunk metadata: {e}")
            # Return basic chunk if enhancement fails
            return chunk
    
    def _analyze_content(self, text: str) -> Dict:
        """Analyze content for intelligent categorization"""
        try:
            text_lower = text.lower()
            
            # Difficulty level detection
            difficulty_level = 'intermediate'
            if any(word in text_lower for word in ['beginner', 'basic', 'start', 'first time']):
                difficulty_level = 'beginner'
            elif any(word in text_lower for word in ['advanced', 'expert', 'professional', 'enterprise']):
                difficulty_level = 'advanced'
            
            # Content category detection
            content_category = 'general'
            if any(word in text_lower for word in ['setup', 'install', 'configuration', 'setup guide']):
                content_category = 'setup'
            elif any(word in text_lower for word in ['tutorial', 'how to', 'step by step', 'guide']):
                content_category = 'tutorial'
            elif any(word in text_lower for word in ['troubleshoot', 'error', 'fix', 'problem', 'issue']):
                content_category = 'troubleshooting'
            elif any(word in text_lower for word in ['api', 'integration', 'webhook', 'endpoint']):
                content_category = 'integration'
            elif any(word in text_lower for word in ['faq', 'question', 'answer', 'common']):
                content_category = 'faq'
            
            # Step detection
            has_steps = any(word in text_lower for word in ['step', '1.', '2.', '3.', 'first', 'second', 'third'])
            
            # Completeness detection
            is_complete = len(text.split()) > 50  # Basic threshold
            
            # Keyword extraction
            keywords = self._extract_keywords(text)
            
            # Summary generation
            summary = text[:200] + "..." if len(text) > 200 else text
            
            # Content type detection
            content_type = 'help_center'
            if any(word in text_lower for word in ['video', 'tutorial', 'screencast']):
                content_type = 'video_transcript'
            elif any(word in text_lower for word in ['api', 'documentation', 'reference']):
                content_type = 'documentation'
            
            # Target audience detection
            target_audience = 'user'
            if any(word in text_lower for word in ['developer', 'engineer', 'technical']):
                target_audience = 'developer'
            elif any(word in text_lower for word in ['admin', 'administrator', 'manager']):
                target_audience = 'admin'
            
            # Prerequisites detection
            prerequisites = []
            if any(word in text_lower for word in ['prerequisite', 'requirement', 'before you begin']):
                prerequisites = ['Basic knowledge required']
            
            # Estimated time detection
            estimated_time = '5-10 minutes'
            if any(word in text_lower for word in ['quick', 'fast', 'simple']):
                estimated_time = '2-5 minutes'
            elif any(word in text_lower for word in ['comprehensive', 'detailed', 'complete']):
                estimated_time = '15-30 minutes'
            
            return {
                'difficulty_level': difficulty_level,
                'content_category': content_category,
                'has_steps': has_steps,
                'is_complete': is_complete,
                'keywords': keywords,
                'summary': summary,
                'content_type': content_type,
                'target_audience': target_audience,
                'prerequisites': prerequisites,
                'estimated_time': estimated_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing content: {e}")
            return {
                'difficulty_level': 'intermediate',
                'content_category': 'general',
                'has_steps': False,
                'is_complete': True,
                'keywords': [],
                'summary': text[:100] if text else '',
                'content_type': 'help_center',
                'target_audience': 'user',
                'prerequisites': [],
                'estimated_time': '5-10 minutes'
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            # Simple keyword extraction (can be enhanced with NLP)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            words = text.lower().split()
            keywords = [word for word in words if word not in common_words and len(word) > 3]
            
            # Return top 10 keywords
            return list(set(keywords))[:10]
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    async def search_knowledge(self, query: str, company_name: str, qudemo_id: str, 
                              content_types: List[str] = None, top_k: int = 5) -> Dict:
        """Search knowledge with context-aware routing"""
        try:
            logger.info(f"🔍 Searching knowledge for: {query}")
            
            # Use enhanced Pinecone manager for search
            search_result = await self.pinecone_manager.search_with_context(
                query=query,
                company_name=company_name,
                qudemo_id=qudemo_id,
                content_types=content_types,
                top_k=top_k
            )
            
            if search_result['success']:
                # Enhance results with additional context
                enhanced_results = self._enhance_search_results(search_result['results'])
                search_result['enhanced_results'] = enhanced_results
                
                logger.info(f"✅ Found {len(enhanced_results)} relevant results")
                return search_result
            else:
                logger.error(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")
                return search_result
                
        except Exception as e:
            logger.error(f"❌ Error in search_knowledge: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def _enhance_search_results(self, results: List) -> List[Dict]:
        """Enhance search results with additional context"""
        try:
            enhanced_results = []
            
            for result in results:
                try:
                    metadata = result.metadata
                    
                    # Create enhanced result
                    enhanced_result = {
                        'id': result.id,
                        'score': result.score,
                        'text': metadata.get('text', ''),
                        'source': metadata.get('source', ''),
                        'source_type': metadata.get('source_type', ''),
                        'title': metadata.get('title', ''),
                        'url': metadata.get('url', ''),
                        'content_category': metadata.get('content_category', 'general'),
                        'difficulty_level': metadata.get('difficulty_level', 'intermediate'),
                        'has_steps': metadata.get('has_steps', False),
                        'keywords': metadata.get('keywords', []),
                        'summary': metadata.get('summary', ''),
                        'target_audience': metadata.get('target_audience', 'user'),
                        'prerequisites': metadata.get('prerequisites', []),
                        'estimated_time': metadata.get('estimated_time', '5-10 minutes'),
                        'index_type': metadata.get('index_type', 'knowledge'),
                        'quality_score': metadata.get('quality_score', 85),
                        'word_count': metadata.get('word_count', 0)
                    }
                    
                    # Add video-specific enhancements
                    if metadata.get('source_type') in ['video_transcript', 'youtube_transcript', 'loom_transcript']:
                        enhanced_result.update({
                            'start_timestamp': metadata.get('start_timestamp', 0),
                            'end_timestamp': metadata.get('end_timestamp', 0),
                            'video_url': metadata.get('video_url', ''),
                            'video_type': metadata.get('video_type', 'unknown')
                        })
                    
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    logger.error(f"❌ Error enhancing result: {e}")
                    continue
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Error enhancing search results: {e}")
            return []
    
    async def get_knowledge_summary(self, company_name: str, qudemo_id: str) -> Dict:
        """Get comprehensive knowledge summary across all indexes"""
        try:
            logger.info(f"📊 Getting knowledge summary for {company_name} qudemo {qudemo_id}")
            
            # Use enhanced Pinecone manager for summary
            summary_result = self.pinecone_manager.get_knowledge_summary(
                company_name=company_name,
                qudemo_id=qudemo_id
            )
            
            if summary_result['success']:
                logger.info(f"✅ Knowledge summary retrieved successfully")
                return summary_result
            else:
                logger.error(f"❌ Failed to get knowledge summary: {summary_result.get('error', 'Unknown error')}")
                return summary_result
                
        except Exception as e:
            logger.error(f"❌ Error in get_knowledge_summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for optimization"""
        try:
            # Use enhanced Pinecone manager for performance metrics
            metrics_result = self.pinecone_manager.get_performance_metrics()
            
            if metrics_result['success']:
                logger.info("✅ Performance metrics retrieved successfully")
                return metrics_result
            else:
                logger.error(f"❌ Failed to get performance metrics: {metrics_result.get('error', 'Unknown error')}")
                return metrics_result
                
        except Exception as e:
            logger.error(f"❌ Error in get_performance_metrics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Global instance
_enhanced_knowledge_integration = None

def initialize_enhanced_knowledge_integration() -> bool:
    """Initialize the enhanced knowledge integration"""
    global _enhanced_knowledge_integration
    try:
        _enhanced_knowledge_integration = EnhancedKnowledgeIntegrator()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Knowledge Integration: {e}")
        return False

def get_enhanced_knowledge_integration() -> EnhancedKnowledgeIntegrator:
    """Get the global enhanced knowledge integration instance"""
    if _enhanced_knowledge_integration is None:
        raise RuntimeError("Enhanced Knowledge Integration not initialized. Call initialize_enhanced_knowledge_integration() first.")
    return _enhanced_knowledge_integration
