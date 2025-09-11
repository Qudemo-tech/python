#!/usr/bin/env python3
"""
Hybrid Chunking Processor
Combines deterministic signals (STT + Video Intelligence) with LLM refinement
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import re

# Import Google processor
from google_video_intelligence_processor import get_google_video_processor

# Import existing components
from enhanced_pinecone_manager import get_enhanced_pinecone_manager
from enhanced_knowledge_integration import get_enhanced_knowledge_integration

# Import subtopic chunking
from subtopic_chunking_processor import get_subtopic_processor

# Import utility functions
from hybrid_utils import intervals_overlap, calculate_overlap_percentage, extract_keywords_from_text, calculate_quality_score

# Configure logging
logger = logging.getLogger(__name__)

class HybridChunkingProcessor:
    """Hybrid chunking processor using deterministic signals + LLM refinement"""
    
    def __init__(self):
        """Initialize hybrid chunking processor"""
        try:
            self.google_processor = get_google_video_processor()
            self.pinecone_manager = get_enhanced_pinecone_manager()
            self.knowledge_integrator = get_enhanced_knowledge_integration()
            self.subtopic_processor = get_subtopic_processor()
            
            # Configuration
            self.config = {
                'min_chunk_size': 100,
                'max_chunk_size': 1000,
                'chunk_overlap': 200,
                'min_segment_duration': 10,
                'max_segment_duration': 120,
                'confidence_threshold': 0.5,
                'enable_subtopic_chunking': True  # Enable subtopic chunking
            }
            
            logger.info("✅ Hybrid Chunking Processor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hybrid Chunking Processor: {e}")
            raise
    
    async def process_video_hybrid(
        self, 
        video_url: str, 
        company_name: str, 
        qudemo_id: str,
        use_youtube_captions: bool = True
    ) -> Dict:
        """
        Process video using hybrid approach: deterministic signals + LLM refinement
        
        Args:
            video_url: Video URL to process
            company_name: Company name for isolation
            qudemo_id: QuDemo ID for isolation
            use_youtube_captions: Whether to try YouTube captions first
            
        Returns:
            Processing results with enhanced chunks and metadata
        """
        try:
            logger.info(f"🎬 Processing video with hybrid approach: {video_url}")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # Step 1: Get deterministic signals from Google services
            if not self.google_processor:
                return {
                    'success': False,
                    'error': 'Google Video Intelligence processor not initialized',
                    'video_url': video_url
                }
            
            deterministic_result = await self.google_processor.process_video_with_deterministic_signals(
                video_url, company_name, qudemo_id, use_youtube_captions
            )
            
            if not deterministic_result['success']:
                logger.warning("⚠️ Deterministic processing failed, falling back to existing method")
                return await self._fallback_to_existing_method(video_url, company_name, qudemo_id)
            
            # Step 2: Process segments for subtopics (if enabled)
            if self.config.get('enable_subtopic_chunking', True):
                logger.info("🔍 Processing segments for subtopic detection")
                segments = deterministic_result.get('segments', [])
                transcription_text = deterministic_result.get('transcription', '')
                
                if segments and transcription_text:
                    processed_segments = await self.subtopic_processor.process_segments_for_subtopics(
                        segments, transcription_text
                    )
                    deterministic_result['segments'] = processed_segments
                    logger.info(f"✅ Processed {len(segments)} segments into {len(processed_segments)} total segments")
            
            # Step 3: Enhance chunks with additional metadata
            enhanced_chunks = await self._enhance_chunks_with_metadata(
                deterministic_result['chunks'], 
                deterministic_result.get('labels', []),
                deterministic_result.get('shots', []),
                video_url, company_name, qudemo_id
            )
            
            # Step 4: Store in Pinecone with enhanced metadata
            storage_result = await self._store_enhanced_chunks(
                enhanced_chunks, company_name, qudemo_id
            )
            
            if not storage_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to store chunks: {storage_result.get('error')}",
                    'video_url': video_url
                }
            
            # Step 4: Calculate final metrics
            final_metrics = self._calculate_final_metrics(
                enhanced_chunks, deterministic_result['validation_metrics']
            )
            
            logger.info(f"✅ Hybrid processing completed: {len(enhanced_chunks)} chunks stored")
            
            return {
                'success': True,
                'chunks_created': len(enhanced_chunks),
                'chunks_stored': storage_result.get('chunks_stored', 0),
                'validation_metrics': final_metrics,
                'time_source': deterministic_result.get('time_source', 'hybrid'),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'processing_method': 'hybrid_deterministic_llm'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid video processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
    
    async def _enhance_chunks_with_metadata(
        self, 
        chunks: List[Dict], 
        labels: List[Dict],
        shots: List[Dict],
        video_url: str,
        company_name: str, 
        qudemo_id: str
    ) -> List[Dict]:
        """Enhance chunks with additional metadata for better Q&A"""
        try:
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Find relevant labels for this chunk
                chunk_labels = self._find_relevant_labels(chunk, labels)
                
                # Find relevant shots for this chunk
                chunk_shots = self._find_relevant_shots(chunk, shots)
                
                # Calculate content quality metrics
                quality_metrics = self._calculate_content_quality(chunk['text'])
                
                # Create enhanced chunk
                enhanced_chunk = {
                    **chunk,  # Keep original chunk data
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'labels': chunk_labels,
                    'shots': chunk_shots,
                    'quality_score': quality_metrics['quality_score'],
                    'difficulty_level': quality_metrics['difficulty_level'],
                    'content_category': quality_metrics['content_category'],
                    'keywords': quality_metrics['keywords'],
                    'has_steps': quality_metrics['has_steps'],
                    'is_complete': quality_metrics['is_complete'],
                    'word_count': len(chunk['text'].split()),
                    'char_count': len(chunk['text']),
                    'processed_at': datetime.now().isoformat(),
                    'processing_method': 'hybrid_deterministic_llm'
                }
                
                enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"🔧 Enhanced {len(enhanced_chunks)} chunks with metadata")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"❌ Error enhancing chunks: {e}")
            return chunks
    
    def _find_relevant_labels(self, chunk: Dict, labels: List[Dict]) -> List[Dict]:
        """Find labels relevant to this chunk's time range using interval overlap"""
        try:
            chunk_start = chunk['start_time']
            chunk_end = chunk['end_time']
            
            relevant_labels = []
            for label in labels:
                # Use interval overlap logic
                if self._intervals_overlap(
                    (chunk_start, chunk_end),
                    (label['start_time'], label['end_time'])
                ):
                    # Calculate overlap percentage
                    overlap_start = max(chunk_start, label['start_time'])
                    overlap_end = min(chunk_end, label['end_time'])
                    overlap_duration = overlap_end - overlap_start
                    chunk_duration = chunk_end - chunk_start
                    overlap_percentage = overlap_duration / chunk_duration if chunk_duration > 0 else 0
                    
                    if overlap_percentage > 0.1:  # At least 10% overlap
                        relevant_labels.append({
                            'label': label['label'],
                            'category': label['category'],
                            'confidence': label['confidence'],
                            'overlap_percentage': overlap_percentage
                        })
            
            return relevant_labels
            
        except Exception as e:
            logger.error(f"❌ Error finding relevant labels: {e}")
            return []
    
    def _intervals_overlap(self, a: tuple, b: tuple) -> bool:
        """
        Check if two intervals overlap using utility function
        
        Args:
            a: (start_time, end_time) for first interval
            b: (start_time, end_time) for second interval
            
        Returns:
            True if intervals overlap, False otherwise
        """
        return intervals_overlap(a, b)
    
    def _find_relevant_shots(self, chunk: Dict, shots: List[Dict]) -> List[Dict]:
        """Find shots relevant to this chunk's time range using interval overlap"""
        try:
            chunk_start = chunk['start_time']
            chunk_end = chunk['end_time']
            
            relevant_shots = []
            for shot in shots:
                # Use interval overlap logic
                if self._intervals_overlap(
                    (chunk_start, chunk_end),
                    (shot['start_time'], shot['end_time'])
                ):
                    relevant_shots.append({
                        'start_time': shot['start_time'],
                        'end_time': shot['end_time'],
                        'confidence': shot['confidence']
                    })
            
            return relevant_shots
            
        except Exception as e:
            logger.error(f"❌ Error finding relevant shots: {e}")
            return []
    
    def _reconstruct_text_from_words(self, words: List[Dict]) -> str:
        """
        Reconstruct text from STT words with proper punctuation and casing
        
        Args:
            words: List of word dictionaries with 'word' and optional 'punctuation'
            
        Returns:
            Properly formatted text string
        """
        if not words:
            return ""
        
        # Start with first word
        text_parts = [words[0]['word']]
        
        for i in range(1, len(words)):
            current_word = words[i]['word']
            prev_word = words[i-1]['word']
            
            # Add space between words
            text_parts.append(' ')
            
            # Handle punctuation from STT if available
            if 'punctuation' in words[i-1] and words[i-1]['punctuation']:
                # Remove the space we just added and add punctuation
                text_parts[-1] = words[i-1]['punctuation']
                text_parts.append(' ')
            
            # Add current word
            text_parts.append(current_word)
        
        # Join and apply basic text reconstruction heuristics
        text = ''.join(text_parts)
        return self._apply_text_reconstruction_heuristics(text)
    
    def _apply_text_reconstruction_heuristics(self, text: str) -> str:
        """
        Apply heuristics to improve text reconstruction quality
        
        Args:
            text: Raw text from word concatenation
            
        Returns:
            Improved text with better punctuation and casing
        """
        if not text:
            return text
        
        # Basic sentence capitalization
        sentences = text.split('. ')
        if len(sentences) > 1:
            sentences = [sentences[0].capitalize()] + [s.capitalize() for s in sentences[1:]]
            text = '. '.join(sentences)
        else:
            text = text.capitalize()
        
        # Fix common abbreviations
        text = text.replace(' i ', ' I ')
        text = text.replace(' i.', ' I.')
        text = text.replace(' i,', ' I,')
        
        # Ensure proper spacing around punctuation
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        return text
    
    def _calculate_content_quality(self, text: str) -> Dict:
        """Calculate content quality metrics"""
        try:
            text_lower = text.lower()
            words = text.split()
            word_count = len(words)
            
            # Use utility function for base quality score
            base_quality = calculate_quality_score(text)
            quality_score = int(base_quality * 100)  # Convert to 0-100 scale
            
            # Check for complete sentences
            sentence_endings = text.count('.') + text.count('!') + text.count('?')
            
            # Content category detection
            content_category = 'general'
            if any(word in text_lower for word in ['tutorial', 'how to', 'step', 'guide']):
                content_category = 'tutorial'
            elif any(word in text_lower for word in ['demo', 'example', 'show']):
                content_category = 'demonstration'
            elif any(word in text_lower for word in ['explain', 'what is', 'definition']):
                content_category = 'explanation'
            
            # Difficulty level
            difficulty_level = 'beginner'
            if any(word in text_lower for word in ['advanced', 'complex', 'sophisticated']):
                difficulty_level = 'advanced'
            elif any(word in text_lower for word in ['intermediate', 'moderate']):
                difficulty_level = 'intermediate'
            
            # Check for steps
            has_steps = any(word in text_lower for word in ['step', 'first', 'second', 'next', 'then'])
            
            # Check if complete
            is_complete = sentence_endings > 0 and word_count > 20
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            
            return {
                'quality_score': min(100, max(0, quality_score)),
                'difficulty_level': difficulty_level,
                'content_category': content_category,
                'keywords': keywords,
                'has_steps': has_steps,
                'is_complete': is_complete
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating content quality: {e}")
            return {
                'quality_score': 50,
                'difficulty_level': 'beginner',
                'content_category': 'general',
                'keywords': [],
                'has_steps': False,
                'is_complete': False
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using utility function"""
        try:
            return extract_keywords_from_text(text, max_keywords=5)
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    async def _store_enhanced_chunks(
        self, 
        chunks: List[Dict], 
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Store enhanced chunks in Pinecone"""
        try:
            if not self.pinecone_manager:
                return {
                    'success': False,
                    'error': 'Pinecone manager not initialized'
                }
            
            # Store chunks using enhanced Pinecone manager
            result = await self.pinecone_manager.store_semantic_chunks(
                chunks=chunks,
                company_name=company_name,
                qudemo_id=qudemo_id,
                content_type='video_hybrid'
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error storing enhanced chunks: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_final_metrics(
        self, 
        chunks: List[Dict], 
        validation_metrics: Dict
    ) -> Dict:
        """Calculate final processing metrics"""
        try:
            if not chunks:
                return validation_metrics
            
            # Calculate chunk-level metrics
            total_chunks = len(chunks)
            chunks_with_labels = sum(1 for c in chunks if c.get('labels'))
            chunks_with_high_quality = sum(1 for c in chunks if c.get('quality_score', 0) > 70)
            
            # Calculate average metrics
            avg_quality_score = sum(c.get('quality_score', 0) for c in chunks) / total_chunks
            avg_word_count = sum(c.get('word_count', 0) for c in chunks) / total_chunks
            
            # Combine with validation metrics
            final_metrics = {
                **validation_metrics,
                'total_chunks': total_chunks,
                'chunks_with_labels': chunks_with_labels,
                'chunks_with_high_quality': chunks_with_high_quality,
                'label_coverage': chunks_with_labels / total_chunks if total_chunks > 0 else 0,
                'quality_coverage': chunks_with_high_quality / total_chunks if total_chunks > 0 else 0,
                'avg_quality_score': avg_quality_score,
                'avg_word_count': avg_word_count
            }
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating final metrics: {e}")
            return validation_metrics
    
    async def _fallback_to_existing_method(
        self, 
        video_url: str, 
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Fallback to existing video processing method"""
        try:
            logger.info("🔄 Falling back to existing video processing method")
            
            # Import existing video processor
            from enhanced_video_processor import get_enhanced_video_processor
            
            video_processor = get_enhanced_video_processor()
            if not video_processor:
                return {
                    'success': False,
                    'error': 'No video processor available',
                    'video_url': video_url
                }
            
            # Use existing method
            result = await video_processor.process_video_with_qudemo(
                video_url, company_name, qudemo_id
            )
            
            # Add fallback indicator
            if result.get('success'):
                result['processing_method'] = 'fallback_existing'
                result['time_source'] = 'fallback'
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in fallback method: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url
            }

# Global instance
_hybrid_processor = None

def initialize_hybrid_chunking_processor() -> bool:
    """Initialize global hybrid chunking processor"""
    global _hybrid_processor
    try:
        _hybrid_processor = HybridChunkingProcessor()
        logger.info("✅ Hybrid Chunking Processor initialized globally")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Hybrid Chunking Processor: {e}")
        return False

def get_hybrid_chunking_processor() -> Optional[HybridChunkingProcessor]:
    """Get global hybrid chunking processor instance"""
    return _hybrid_processor
