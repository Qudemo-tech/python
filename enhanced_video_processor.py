#!/usr/bin/env python3
"""
Enhanced Video Processor with Standard Plan Multi-Index Integration
Optimized for better transcript quality and storage
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import enhanced components
from enhanced_knowledge_integration import get_enhanced_knowledge_integration

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    """Enhanced Video Processor with Standard Plan optimizations"""
    
    def __init__(self):
        """Initialize enhanced video processor"""
        self.knowledge_integration = None
        logger.info("✅ Enhanced Video Processor initialized")
    
    def _get_knowledge_integration(self):
        """Get knowledge integration (lazy initialization)"""
        if self.knowledge_integration is None:
            try:
                self.knowledge_integration = get_enhanced_knowledge_integration()
                logger.info("✅ Enhanced Knowledge Integration loaded")
            except Exception as e:
                logger.warning(f"⚠️ Could not load Enhanced Knowledge Integration: {e}")
                # Don't raise error, just log warning
        return self.knowledge_integration
    
    async def process_youtube_video(self, video_url: str, company_name: str, qudemo_id: str) -> Dict:
        """Process YouTube video with enhanced transcript extraction using Gemini"""
        try:
            logger.info(f"🎥 Processing YouTube video for {company_name} qudemo {qudemo_id}")
            
            # Use the new Gemini Transcription Processor
            from gemini_transcription import GeminiTranscriptionProcessor
            
            # Initialize the processor
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')
            
            if not all([gemini_api_key, pinecone_api_key, openai_api_key]):
                raise Exception("Missing required API keys: GEMINI_API_KEY, PINECONE_API_KEY, OPENAI_API_KEY")
            
            processor = GeminiTranscriptionProcessor(
                gemini_api_key=gemini_api_key,
                pinecone_api_key=pinecone_api_key,
                openai_api_key=openai_api_key
            )
            
            # Process the video with qudemo_id for proper namespace
            result = await processor.process_video_with_qudemo(video_url, company_name, qudemo_id)
            
            if result and result.get('success'):
                logger.info(f"✅ Successfully processed YouTube video using Gemini")
                return {
                    'success': True,
                    'chunks_stored': result.get('chunks_stored', result.get('chunks_created', 0)),
                    'video_type': 'youtube',
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'video_title': result.get('title', ''),
                    'video_duration': result.get('duration', ''),
                    'total_segments': result.get('chunks_stored', result.get('chunks_created', 0)),
                    'storage_details': result
                }
            else:
                logger.error(f"❌ Failed to process YouTube video: {result.get('error', 'Unknown error') if result else 'No result'}")
                return {
                    'success': False,
                    'error': f"Processing failed: {result.get('error', 'Unknown error') if result else 'No result'}",
                    'chunks_stored': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing YouTube video: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_stored': 0
            }
    
    async def process_loom_video(self, video_url: str, company_name: str, qudemo_id: str) -> Dict:
        """Process Loom video with Whisper transcription using loom_processor"""
        try:
            logger.info(f"🎥 Processing Loom video for {company_name} qudemo {qudemo_id}")
            
            # Use the Loom Video Processor with Whisper
            from loom_processor import LoomVideoProcessor
            
            # Initialize the processor
            openai_api_key = os.getenv('OPENAI_API_KEY')
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            
            if not all([openai_api_key, pinecone_api_key]):
                raise Exception("Missing required API keys: OPENAI_API_KEY, PINECONE_API_KEY")
            
            processor = LoomVideoProcessor(
                openai_api_key=openai_api_key,
                pinecone_api_key=pinecone_api_key
            )
            
            # Process the video with qudemo_id for proper namespace
            result = processor.process_video(video_url, company_name, qudemo_id)
            
            if result and result.get('success'):
                logger.info(f"✅ Successfully processed Loom video using Whisper")
                return {
                    'success': True,
                    'chunks_stored': result.get('chunks_created', 0),
                    'video_type': 'loom',
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'video_title': result.get('title', ''),
                    'video_duration': result.get('duration', ''),
                    'total_segments': result.get('chunks_created', 0),
                    'storage_details': result
                }
            else:
                logger.error(f"❌ Failed to process Loom video: {result.get('error', 'Unknown error') if result else 'No result'}")
                return {
                    'success': False,
                    'error': f"Processing failed: {result.get('error', 'Unknown error') if result else 'No result'}",
                    'chunks_stored': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing Loom video: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_stored': 0
            }
    
    def _create_enhanced_video_chunks(self, transcript_data: Dict, video_url: str, 
                                    company_name: str, qudemo_id: str, video_type: str) -> List[Dict]:
        """Create enhanced video chunks with better metadata"""
        try:
            transcript = transcript_data.get('transcript', [])
            video_title = transcript_data.get('title', 'Video Transcript')
            video_duration = transcript_data.get('duration', 0)
            
            chunks_data = []
            
            for i, segment in enumerate(transcript):
                try:
                    # Extract text and timestamps
                    text = segment.get('text', '')
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    
                    # Skip empty segments
                    if not text.strip():
                        continue
                    
                    # Analyze segment content
                    content_analysis = self._analyze_video_segment(text, start_time, end_time)
                    
                    # Create enhanced chunk
                    chunk_data = {
                        'text': text,
                        'source_url': video_url,
                        'source_type': f'{video_type}_transcript',
                        'title': video_title,
                        'url': video_url,
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'chunk_index': i,
                        'chunk_size': len(text),
                        'quality_score': content_analysis['quality_score'],
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'video_url': video_url,
                        'video_type': video_type,
                        'processed_at': datetime.now().isoformat(),
                        'segment_duration': end_time - start_time if end_time > start_time else 0,
                        'word_count': len(text.split()),
                        'content_category': content_analysis['content_category'],
                        'difficulty_level': content_analysis['difficulty_level'],
                        'has_steps': content_analysis['has_steps'],
                        'is_complete': content_analysis['is_complete'],
                        'keywords': content_analysis['keywords'],
                        'summary': content_analysis['summary']
                    }
                    
                    chunks_data.append(chunk_data)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing video segment {i}: {e}")
                    continue
            
            logger.info(f"✅ Created {len(chunks_data)} enhanced video chunks")
            return chunks_data
            
        except Exception as e:
            logger.error(f"❌ Error creating enhanced video chunks: {e}")
            return []
    
    def _analyze_video_segment(self, text: str, start_time: float, end_time: float) -> Dict:
        """Analyze video segment content for better categorization"""
        try:
            text_lower = text.lower()
            
            # Quality score based on content length and structure
            quality_score = 85  # Base score
            
            # Adjust based on content length
            if len(text) > 100:
                quality_score += 10
            elif len(text) < 20:
                quality_score -= 15
            
            # Content category detection
            content_category = 'general'
            if any(word in text_lower for word in ['step', 'first', 'second', 'third', 'next', 'then']):
                content_category = 'tutorial'
                quality_score += 5
            elif any(word in text_lower for word in ['error', 'problem', 'issue', 'fix', 'troubleshoot']):
                content_category = 'troubleshooting'
                quality_score += 5
            elif any(word in text_lower for word in ['setup', 'install', 'configure', 'initialize']):
                content_category = 'setup'
                quality_score += 5
            
            # Difficulty level detection
            difficulty_level = 'intermediate'
            if any(word in text_lower for word in ['beginner', 'basic', 'start', 'first time', 'simple']):
                difficulty_level = 'beginner'
            elif any(word in text_lower for word in ['advanced', 'expert', 'professional', 'complex']):
                difficulty_level = 'advanced'
            
            # Step detection
            has_steps = any(word in text_lower for word in ['step', '1.', '2.', '3.', 'first', 'second', 'third'])
            
            # Completeness detection
            is_complete = len(text.split()) > 30 and text.strip().endswith(('.', '!', '?'))
            
            # Keyword extraction
            keywords = self._extract_video_keywords(text)
            
            # Summary generation
            summary = text[:150] + "..." if len(text) > 150 else text
            
            # Adjust quality score based on completeness
            if is_complete:
                quality_score += 5
            if has_steps:
                quality_score += 5
            
            # Ensure quality score is within bounds
            quality_score = max(50, min(100, quality_score))
            
            return {
                'quality_score': quality_score,
                'content_category': content_category,
                'difficulty_level': difficulty_level,
                'has_steps': has_steps,
                'is_complete': is_complete,
                'keywords': keywords,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing video segment: {e}")
            return {
                'quality_score': 75,
                'content_category': 'general',
                'difficulty_level': 'intermediate',
                'has_steps': False,
                'is_complete': True,
                'keywords': [],
                'summary': text[:100] if text else ''
            }
    
    def _extract_video_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from video segment"""
        try:
            # Simple keyword extraction (can be enhanced with NLP)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'so', 'very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'than', 'too', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
            
            words = text.lower().split()
            keywords = [word for word in words if word not in common_words and len(word) > 3]
            
            # Return top 8 keywords
            return list(set(keywords))[:8]
            
        except Exception as e:
            logger.error(f"❌ Error extracting video keywords: {e}")
            return []
    
    async def process_video_batch(self, video_urls: List[str], company_name: str, qudemo_id: str, 
                                video_types: List[str] = None) -> Dict:
        """Process multiple videos in batch"""
        try:
            logger.info(f"🎬 Processing batch of {len(video_urls)} videos for {company_name} qudemo {qudemo_id}")
            
            if not video_types:
                # Auto-detect video types
                video_types = []
                for url in video_urls:
                    if 'youtube.com' in url or 'youtu.be' in url:
                        video_types.append('youtube')
                    elif 'loom.com' in url:
                        video_types.append('loom')
                    else:
                        video_types.append('unknown')
            
            if len(video_types) != len(video_urls):
                raise ValueError("Number of video types must match number of video URLs")
            
            batch_results = {
                'success': True,
                'total_videos': len(video_urls),
                'processed_videos': 0,
                'failed_videos': 0,
                'total_chunks_stored': 0,
                'results': [],
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'total_duration': None
            }
            
            start_time = datetime.now()
            
            # Process videos sequentially
            for i, (video_url, video_type) in enumerate(zip(video_urls, video_types)):
                try:
                    logger.info(f"🎬 Processing video {i + 1}/{len(video_urls)}: {video_type} - {video_url}")
                    
                    if video_type == 'youtube':
                        result = await self.process_youtube_video(video_url, company_name, qudemo_id)
                    elif video_type == 'loom':
                        result = await self.process_loom_video(video_url, company_name, qudemo_id)
                    else:
                        result = {
                            'success': False,
                            'error': f'Unsupported video type: {video_type}',
                            'chunks_stored': 0
                        }
                    
                    if result['success']:
                        batch_results['processed_videos'] += 1
                        batch_results['total_chunks_stored'] += result['chunks_stored']
                        batch_results['results'].append({
                            'video_url': video_url,
                            'video_type': video_type,
                            'status': 'success',
                            'chunks_stored': result['chunks_stored'],
                            'result': result
                        })
                        logger.info(f"✅ Video {i + 1}/{len(video_urls)} processed successfully")
                    else:
                        batch_results['failed_videos'] += 1
                        batch_results['results'].append({
                            'video_url': video_url,
                            'video_type': video_type,
                            'status': 'failed',
                            'error': result.get('error', 'Unknown error'),
                            'chunks_stored': 0
                        })
                        logger.error(f"❌ Video {i + 1}/{len(video_urls)} failed: {result.get('error', 'Unknown error')}")
                    
                    # Add small delay between videos for memory cleanup
                    if i < len(video_urls) - 1:
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    batch_results['failed_videos'] += 1
                    batch_results['results'].append({
                        'video_url': video_url,
                        'video_type': video_type,
                        'status': 'error',
                        'error': str(e),
                        'chunks_stored': 0
                    })
                    logger.error(f"❌ Video {i + 1}/{len(video_urls)} error: {e}")
            
            # Calculate batch completion metrics
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            batch_results['end_time'] = end_time.isoformat()
            batch_results['total_duration'] = total_duration
            batch_results['success'] = batch_results['failed_videos'] == 0
            
            # Log batch completion
            logger.info(f"🎬 Batch processing completed for {company_name} qudemo {qudemo_id}:")
            logger.info(f"   ✅ Processed: {batch_results['processed_videos']}/{batch_results['total_videos']}")
            logger.info(f"   ❌ Failed: {batch_results['failed_videos']}/{batch_results['total_videos']}")
            logger.info(f"   📊 Total chunks stored: {batch_results['total_chunks_stored']}")
            logger.info(f"   ⏱️ Total duration: {total_duration/60:.1f} minutes")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch video processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_videos': len(video_urls),
                'processed_videos': 0,
                'failed_videos': len(video_urls),
                'total_chunks_stored': 0,
                'results': []
            }

# Global instance
_enhanced_video_processor = None

def initialize_enhanced_video_processor() -> bool:
    """Initialize the enhanced video processor"""
    global _enhanced_video_processor
    try:
        _enhanced_video_processor = EnhancedVideoProcessor()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Video Processor: {e}")
        return False

def get_enhanced_video_processor() -> EnhancedVideoProcessor:
    """Get the global enhanced video processor instance"""
    if _enhanced_video_processor is None:
        raise RuntimeError("Enhanced Video Processor not initialized. Call initialize_enhanced_video_processor() first.")
    return _enhanced_video_processor
