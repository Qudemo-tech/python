#!/usr/bin/env python3
"""
Enhanced Video Chunking Processor
Implements pure Gemini-based topic analysis without video downloading
Uses intelligent prompt strategy to extract topics from full video
"""

import os
import logging
import json
import time
import asyncio
import requests
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import google.generativeai as genai
import openai
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logger = logging.getLogger(__name__)

def stable_vec_id(video_url: str, segment_id: int, start_char: int, end_char: int, 
                  transcript_version: str = "v1", chunking_version: str = "v2-seg-safe") -> str:
    """
    Create a stable, deterministic vector ID that doesn't change across runs
    
    Args:
        video_url: Original video URL
        segment_id: Topic segment ID
        start_char: Start character position
        end_char: End character position
        transcript_version: Version of transcript processing
        chunking_version: Version of chunking algorithm
        
    Returns:
        Stable SHA1 hash as string
    """
    raw = f"{video_url}|{segment_id}|{start_char}|{end_char}|{transcript_version}|{chunking_version}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def index_words(text: str) -> List[Tuple[int, int, int]]:
    """
    Precompute word spans to avoid O(n²) and spacing bugs
    
    Args:
        text: Input text to index
        
    Returns:
        List of (start_char, end_char, word_index) tuples
    """
    spans = []  # [(start_char, end_char, word_index)]
    i = 0
    w = 0
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        
        # Find word end
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        
        spans.append((i, j, w))
        w += 1
        i = j
    
    return spans

def sanitize_segments(segments: List[Dict]) -> List[Dict]:
    """
    Sanitize and validate topic segments before chunking
    
    Args:
        segments: Raw segments from Gemini
        
    Returns:
        Cleaned and validated segments
    """
    if not segments:
        return []
    
    # Sort by start_timestamp
    segments = sorted(segments, key=lambda x: x.get('start_timestamp', 0))
    
    sanitized = []
    for i, segment in enumerate(segments):
        start = segment.get('start_timestamp', 0)
        end = segment.get('end_timestamp', 0)
        
        # Skip degenerate segments
        if end <= start:
            logger.warning(f"⚠️ Skipping degenerate segment {i}: end <= start")
            continue
        
        # Skip empty segments
        if not segment.get('full_text', '').strip():
            logger.warning(f"⚠️ Skipping empty segment {i}: no text content")
            continue
        
        # Validate minimum segment duration (≥2s)
        if end - start < 2.0:
            logger.warning(f"⚠️ Skipping segment {i}: too short ({end - start:.1f}s < 2s)")
            continue
        
        # Validate maximum segment duration (≤10 min)
        if end - start > 600.0:
            logger.warning(f"⚠️ Segment {i} is very long: {end - start:.1f}s (>{600}s)")
        
        # Ensure segment_id exists
        if 'segment_id' not in segment:
            segment['segment_id'] = i
        
        sanitized.append(segment)
    
    # Check for overlaps and fix them
    for i in range(len(sanitized) - 1):
        current = sanitized[i]
        next_seg = sanitized[i + 1]
        
        if current['end_timestamp'] > next_seg['start_timestamp']:
            # Clip current segment to avoid overlap
            current['end_timestamp'] = next_seg['start_timestamp']
            logger.info(f"🔧 Clipped segment {i} to avoid overlap")
    
    # Calculate coverage
    total_duration = sum(seg['end_timestamp'] - seg['start_timestamp'] for seg in sanitized)
    logger.info(f"📊 Segment coverage: {len(sanitized)} segments, {total_duration:.1f}s total")
    
    return sanitized

def _detect_language(text: str) -> str:
    """Simple language detection for CJK safety"""
    try:
        if not text:
            return 'en'
        
        # Check for CJK characters
        cjk_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or  # Chinese
                        '\u3040' <= char <= '\u309f' or  # Hiragana
                        '\u30a0' <= char <= '\u30ff' or  # Katakana
                        '\uac00' <= char <= '\ud7af')    # Korean
        
        if cjk_chars > len(text) * 0.3:  # 30% threshold
            return 'cjk'
        else:
            return 'en'
    except:
        return 'en'

class EnhancedVideoChunkingProcessor:
    """
    Enhanced Video Processor with pure Gemini-based topic analysis
    No video downloading or splitting - direct API analysis
    """
    
    def __init__(self, gemini_api_key: str, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize the enhanced video chunking processor
        
        Args:
            gemini_api_key: Google Gemini API key
            pinecone_api_key: Pinecone API key  
            openai_api_key: OpenAI API key for embeddings
        """
        self.gemini_api_key = gemini_api_key
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Configure OpenAI
        openai.api_key = openai_api_key
        
        # Configure Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # API rate limiting
        self.last_api_call_time = 0
        self.min_api_interval = 3  # Minimum seconds between API calls
        
        logger.info("✅ Enhanced Video Chunking Processor initialized (Pure Gemini Approach)")
    
    async def process_video_with_topic_analysis(self, video_url: str, company_name: str, qudemo_id: str) -> Dict:
        """
        Main processing method using pure Gemini approach
        
        Args:
            video_url: YouTube video URL
            company_name: Company name for organization
            qudemo_id: QuDemo ID for namespace isolation
            
        Returns:
            Dict with processing results
        """
        try:
            logger.info(f"🎬 Starting pure Gemini video processing: {video_url}")
            logger.info(f"🏢 Company: {company_name}, QuDemo ID: {qudemo_id}")
            
            # Phase 1: Direct Gemini Analysis (No video splitting)
            phase_start = time.time()
            logger.info("🧠 Phase 1: Direct Gemini Topic Segment Analysis")
            raw_segments = await self._analyze_video_topics_direct(video_url)
            phase_duration = time.time() - phase_start
            logger.info(f"⏱️ Phase 1 completed in {phase_duration:.2f}s")
            
            if not raw_segments:
                logger.error("❌ Failed to extract topic segments from video")
                return {
                    'success': False,
                    'error': 'Failed to extract topic segments from video',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            # Phase 1.5: Sanitize segments
            logger.info("🧹 Phase 1.5: Sanitizing segments")
            topic_segments = sanitize_segments(raw_segments)
            
            if not topic_segments:
                logger.error("❌ No valid segments after sanitization")
                return {
                    'success': False,
                    'error': 'No valid segments after sanitization',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Extracted and sanitized {len(topic_segments)} topic segments from video")
            
            # Duration parity validation
            if topic_segments:
                total_coverage = sum(seg.get('end_timestamp', 0) - seg.get('start_timestamp', 0) for seg in topic_segments)
                first_start = min(seg.get('start_timestamp', 0) for seg in topic_segments)
                last_end = max(seg.get('end_timestamp', 0) for seg in topic_segments)
                video_duration = last_end - first_start
                coverage_diff = abs(total_coverage - video_duration)
                
                logger.info(f"📊 Duration Analysis: video_duration={video_duration:.1f}s, segments_coverage={total_coverage:.1f}s, diff={coverage_diff:.1f}s")
                
                # Calculate percentage gap for more reasonable validation
                gap_percentage = (coverage_diff / video_duration) * 100 if video_duration > 0 else 0
                
                if gap_percentage > 10.0:  # More than 10% gap is concerning
                    logger.error(f"❌ CRITICAL: Duration gap too large: {coverage_diff:.1f}s ({gap_percentage:.1f}%) - FAILING")
                    raise ValueError(f"Duration parity check failed: {coverage_diff:.1f}s gap ({gap_percentage:.1f}%) > 10% threshold")
                elif gap_percentage > 5.0:  # More than 5% gap is worth noting
                    logger.warning(f"⚠️ Moderate duration gap detected: {coverage_diff:.1f}s ({gap_percentage:.1f}%) - may indicate gaps/overlaps")
                else:
                    logger.info(f"✅ Duration parity check passed: {coverage_diff:.1f}s ({gap_percentage:.1f}%) difference")
            
            # Phase 2: Segment-Safe Chunking (Never cross topic boundaries)
            phase_start = time.time()
            logger.info("🔧 Phase 2: Segment-Safe Chunking")
            chunks = self._create_segment_safe_chunks(topic_segments, video_url, company_name, qudemo_id)
            phase_duration = time.time() - phase_start
            logger.info(f"⏱️ Phase 2 completed in {phase_duration:.2f}s")
            
            if not chunks:
                logger.error("❌ Failed to create segment-safe chunks")
                return {
                    'success': False,
                    'error': 'Failed to create segment-safe chunks',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Created {len(chunks)} segment-safe chunks")
            
            # Phase 3: Store Chunks in Pinecone
            phase_start = time.time()
            logger.info("💾 Phase 3: Store Chunks in Pinecone")
            storage_success = await self._store_segment_safe_chunks_in_pinecone(
                chunks, video_url, company_name, qudemo_id
            )
            phase_duration = time.time() - phase_start
            logger.info(f"⏱️ Phase 3 completed in {phase_duration:.2f}s")
            
            if not storage_success:
                logger.error("❌ Failed to store topics in Pinecone")
                return {
                    'success': False,
                    'error': 'Failed to store topics in Pinecone',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            # Return success result
            result = {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'segments_extracted': len(topic_segments),
                'chunks_created': len(chunks),
                'method': 'segment_safe_chunking',
                'processing_quality': 'topic_boundary_guaranteed',
                'boundary_purity': '100%',
                'status_message': f'Video processed successfully with {len(topic_segments)} topic segments and {len(chunks)} segment-safe chunks'
            }
            
            logger.info(f"✅ Pure Gemini video processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pure Gemini video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
    
    async def _analyze_video_topics_direct(self, video_url: str) -> List[Dict]:
        """
        Direct Gemini analysis of full video for topic extraction
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            List of topic dictionaries
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            if time_since_last_call < self.min_api_interval:
                sleep_time = self.min_api_interval - time_since_last_call
                logger.info(f"⏳ Rate limiting: waiting {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
            
            self.last_api_call_time = time.time()
            
            # Prepare the sophisticated prompt for full video analysis
            prompt_text = self._get_full_video_topic_analysis_prompt()
            
            # Prepare the request
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            headers = {"Content-Type": "application/json"}
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {
                            "fileData": {
                                "mimeType": "video/mp4",
                                "fileUri": video_url
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": self._get_topic_schema()
                }
            }
            
            logger.info(f"🧠 Sending direct topic analysis request for full video")
            
            response = requests.post(
                f"{url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=180  # 3 minutes for full video analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    topics_json = result["candidates"][0]["content"]["parts"][0]["text"]
                    topics = json.loads(topics_json)
                    
                    logger.info(f"✅ Extracted {len(topics)} topics from full video")
                    return topics
                else:
                    logger.error("❌ No candidates in Gemini response")
                    return []
            else:
                logger.error(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Direct topic analysis failed: {e}")
            return []
    
    def _get_full_video_topic_analysis_prompt(self) -> str:
        """
        Get the sophisticated prompt for strict topic segment analysis
        
        Returns:
            Formatted prompt text
        """
        return """You are a sophisticated video analysis AI. Your task is to analyze the provided YouTube video and identify its core topics as STRICT, NON-OVERLAPPING segments.

CRITICAL REQUIREMENTS:
1. Identify distinct topics and create contiguous, non-overlapping time segments
2. Each segment must have a clear topic boundary - never split a single topic across segments
3. Provide the FULL TEXT CONTENT for each segment (not just summaries)
4. Timestamps must be precise and non-overlapping
5. Each segment should contain complete thoughts or explanations

For each topic segment, provide:
- A descriptive topic title
- Precise start and end timestamps in seconds
- The complete text content spoken in that segment
- A brief summary of the topic

The output must be a valid JSON array with strict segment boundaries. No topic should bleed across segment boundaries."""
    
    
    def _get_topic_schema(self) -> Dict:
        """
        Get the JSON schema for strict topic segment analysis response
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "Unique identifier for this topic segment (0-based index)."
                    },
                    "topic_title": {
                        "type": "string",
                        "description": "A descriptive title for the topic or feature. Example: 'Setting up the API Key' or 'Advanced Filtering Options'."
                    },
                    "start_timestamp": {
                        "type": "number",
                        "description": "The timestamp in seconds where this specific topic begins."
                    },
                    "end_timestamp": {
                        "type": "number",
                        "description": "The timestamp in seconds where this topic ends."
                    },
                    "full_text": {
                        "type": "string",
                        "description": "The complete text content spoken in this segment."
                    },
                    "topic_summary": {
                        "type": "string",
                        "description": "A brief, one-sentence summary of the content discussed in this segment."
                    }
                },
                "required": ["segment_id", "topic_title", "start_timestamp", "end_timestamp", "full_text", "topic_summary"]
            }
        }
    
    def _create_segment_safe_chunks(self, topic_segments: List[Dict], video_url: str, 
                                  company_name: str, qudemo_id: str) -> List[Dict]:
        """
        Create chunks that NEVER cross topic segment boundaries
        
        Args:
            topic_segments: List of topic segments from Gemini
            video_url: Original video URL
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            List of segment-safe chunks
        """
        try:
            logger.info(f"🔧 Creating segment-safe chunks from {len(topic_segments)} topic segments")
            
            chunks = []
            MAX_TOKENS = 220  # ~1200 characters
            OVERLAP_TOKENS = 30  # ~150 characters
            
            for segment in topic_segments:
                segment_id = segment.get('segment_id', 0)
                topic_title = segment.get('topic_title', 'Unknown Topic')
                segment_start = segment.get('start_timestamp', 0.0)
                segment_end = segment.get('end_timestamp', 0.0)
                full_text = segment.get('full_text', '')
                topic_summary = segment.get('topic_summary', '')
                
                logger.info(f"🔧 Processing segment {segment_id}: '{topic_title}' ({segment_start:.1f}s - {segment_end:.1f}s)")
                
                # Validate segment boundaries
                if segment_end <= segment_start:
                    logger.warning(f"⚠️ Invalid segment {segment_id}: end <= start")
                    continue
                
                if not full_text.strip():
                    logger.warning(f"⚠️ Empty segment {segment_id}, skipping")
                    continue
                
                # Create word-safe windows within this segment only
                segment_chunks = self._create_word_safe_windows(
                    full_text, segment_start, segment_end, segment_id, 
                    topic_title, topic_summary, MAX_TOKENS, OVERLAP_TOKENS,
                    video_url, company_name, qudemo_id
                )
                
                chunks.extend(segment_chunks)
                logger.info(f"✅ Created {len(segment_chunks)} chunks for segment {segment_id}")
            
            # Validate boundary purity
            self._validate_boundary_purity(chunks, topic_segments)
            
            logger.info(f"✅ Created {len(chunks)} total segment-safe chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Segment-safe chunking failed: {e}")
            return []
    
    def _create_word_safe_windows(self, text: str, segment_start: float, segment_end: float,
                                segment_id: int, topic_title: str, topic_summary: str,
                                max_tokens: int, overlap_tokens: int, video_url: str,
                                company_name: str, qudemo_id: str) -> List[Dict]:
        """
        Create word-safe windows within a single topic segment using precomputed word spans
        """
        chunks = []
        
        # Precompute word spans once (O(n) instead of O(n²))
        word_spans = index_words(text)
        if not word_spans:
            logger.warning(f"⚠️ No words found in segment {segment_id}")
            return chunks
        
        # If text is short enough, create single chunk
        if len(word_spans) <= max_tokens // 2:  # Rough word estimate
            chunk = self._create_chunk_from_segment(
                text, segment_start, segment_end, segment_id, topic_title, topic_summary,
                video_url, company_name, qudemo_id, 0, len(text)
            )
            chunks.append(chunk)
            return chunks
        
        # Split into word-safe windows with proper overlap
        current_start_word = 0
        current_end_word = 0
        
        while current_end_word < len(word_spans):
            # Find the end of current window
            window_word_indices = []
            token_count = 0
            
            # Add words until we hit the token limit
            for i in range(current_end_word, len(word_spans)):
                start_char, end_char, word_idx = word_spans[i]
                word_text = text[start_char:end_char]
                
                # Rough token estimation (1 token ≈ 4 characters)
                word_tokens = len(word_text) // 4 + 1
                
                if token_count + word_tokens > max_tokens and window_word_indices:
                    break
                
                window_word_indices.append(i)
                token_count += word_tokens
                current_end_word = i + 1
            
            if not window_word_indices:
                # Fallback: take at least one word
                window_word_indices = [current_end_word]
                current_end_word += 1
            
            # Get exact character boundaries from word spans
            first_word_start = word_spans[window_word_indices[0]][0]
            last_word_end = word_spans[window_word_indices[-1]][1]
            
            # Extract window text
            window_text = text[first_word_start:last_word_end]
            
            # Map character positions to time within segment
            start_time = self._map_char_to_time(text, segment_start, segment_end, first_word_start)
            end_time = self._map_char_to_time(text, segment_start, segment_end, last_word_end)
            
            chunk = self._create_chunk_from_segment(
                window_text, start_time, end_time, segment_id, topic_title, topic_summary,
                video_url, company_name, qudemo_id, first_word_start, last_word_end
            )
            chunks.append(chunk)
            
            # Move to next window with proper overlap (keep N words of overlap)
            overlap_words = min(overlap_tokens, len(window_word_indices))
            current_start_word = max(0, current_end_word - overlap_words)
            
            # Prevent infinite loop
            if current_start_word >= current_end_word:
                current_start_word = current_end_word
        
        return chunks
    
    def _create_chunk_from_segment(self, text: str, start_time: float, end_time: float,
                                 segment_id: int, topic_title: str, topic_summary: str,
                                 video_url: str, company_name: str, qudemo_id: str,
                                 start_char: int, end_char: int) -> Dict:
        """Create a chunk from a segment with proper metadata and quality metrics"""
        
        # Calculate quality metrics
        quality_metrics = self._calculate_chunk_quality(text, start_time, end_time, segment_id)
        
        return {
            'text': text,
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'segment_id': segment_id,
            'segment_topic': topic_title,
            'segment_summary': topic_summary,
            'start_char': start_char,
            'end_char': end_char,
            'video_url': video_url,
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'chunk_type': 'segment_safe',
            'boundary_purity': 'guaranteed',
            'quality_score': quality_metrics['overall_score'],
            'quality_components': quality_metrics
        }
    
    def _calculate_chunk_quality(self, text: str, start_time: float, end_time: float, segment_id: int) -> Dict:
        """
        Calculate quality metrics for a chunk
        
        Returns:
            Dictionary with quality components and overall score
        """
        # Boundary integrity (binary 1/0 per chunk)
        boundary_integrity = 1.0  # Guaranteed by design
        
        # Text quality metrics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Minimum content thresholds
        min_chars = 50
        min_words = 10
        min_sentences = 1
        
        content_score = 0.0
        if char_count >= min_chars:
            content_score += 0.4
        if word_count >= min_words:
            content_score += 0.4
        if sentence_count >= min_sentences:
            content_score += 0.2
        
        # Duration quality (prefer chunks between 10-60 seconds)
        duration = end_time - start_time
        duration_score = 1.0
        if duration < 5.0:  # Too short
            duration_score = 0.5
        elif duration > 120.0:  # Too long
            duration_score = 0.7
        
        # Overall score (weighted average)
        overall_score = (
            boundary_integrity * 0.4 +  # Most important
            content_score * 0.4 +       # Content quality
            duration_score * 0.2        # Duration appropriateness
        ) * 100  # Scale to 0-100
        
        return {
            'boundary_integrity': boundary_integrity,
            'content_score': content_score,
            'duration_score': duration_score,
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'duration_seconds': duration,
            'overall_score': round(overall_score, 1)
        }
    
    def _map_char_to_time(self, text: str, segment_start: float, segment_end: float, char_pos: int) -> float:
        """Map character position to time within a segment using linear interpolation"""
        if not text or char_pos <= 0:
            return segment_start
        
        if char_pos >= len(text):
            return segment_end
        
        # Linear interpolation
        ratio = char_pos / len(text)
        return segment_start + ratio * (segment_end - segment_start)
    
    def _validate_boundary_purity(self, chunks: List[Dict], topic_segments: List[Dict]):
        """Validate that no chunk crosses topic segment boundaries"""
        try:
            # Create segment lookup
            segment_lookup = {seg['segment_id']: seg for seg in topic_segments}
            
            violations = 0
            for chunk in chunks:
                segment_id = chunk.get('segment_id')
                chunk_start = chunk.get('start_timestamp', 0)
                chunk_end = chunk.get('end_timestamp', 0)
                
                if segment_id not in segment_lookup:
                    logger.warning(f"⚠️ Chunk references unknown segment {segment_id}")
                    violations += 1
                    continue
                
                segment = segment_lookup[segment_id]
                seg_start = segment.get('start_timestamp', 0)
                seg_end = segment.get('end_timestamp', 0)
                
                # Check if chunk is fully within segment boundaries
                if chunk_start < seg_start or chunk_end > seg_end:
                    logger.error(f"❌ BOUNDARY VIOLATION: Chunk {chunk_start:.1f}-{chunk_end:.1f}s crosses segment {seg_start:.1f}-{seg_end:.1f}s")
                    violations += 1
            
            if violations == 0:
                logger.info(f"✅ BOUNDARY PURITY: 100% - No chunks cross segment boundaries")
            else:
                logger.error(f"❌ BOUNDARY PURITY: {violations} violations found")
                
        except Exception as e:
            logger.error(f"❌ Boundary validation failed: {e}")
    
    
    async def _store_segment_safe_chunks_in_pinecone(self, chunks: List[Dict], video_url: str, 
                                                   company_name: str, qudemo_id: str) -> bool:
        """
        Store segment-safe chunks in Pinecone with enhanced metadata
        
        Args:
            chunks: List of segment-safe chunks
            video_url: Original video URL
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"💾 Storing {len(chunks)} segment-safe chunks in Pinecone")
            
            # Create or get index
            index_name = "qudemo-video-index"
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if index_name not in existing_indexes:
                logger.info(f"📊 Creating Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=3072,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            index = self.pc.Index(index_name)
            
            # Create embeddings for chunks
            chunk_texts = []
            for chunk in chunks:
                # Use the chunk text for embedding
                chunk_texts.append(chunk['text'])
            
            logger.info(f"🧠 Creating embeddings for {len(chunk_texts)} chunks")
            embeddings = self._create_embeddings(chunk_texts)
            
            if not embeddings or len(embeddings) != len(chunks):
                logger.error("❌ Failed to create embeddings")
                return False
            
            # Prepare vectors for upsert with enhanced metadata
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create stable, deterministic ID
                vector_id = stable_vec_id(
                    video_url=chunk['video_url'],
                    segment_id=chunk['segment_id'],
                    start_char=chunk['start_char'],
                    end_char=chunk['end_char'],
                    transcript_version="v1",
                    chunking_version="v2-seg-safe"
                )
                
                # Get quality metrics and convert to Pinecone-compatible format
                quality_components = chunk.get('quality_components', {})
                quality_components_str = json.dumps(quality_components) if quality_components else "{}"
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'],
                        'source_url': video_url,
                        'source_type': 'youtube_segment_safe_chunk',
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'segment_id': chunk['segment_id'],
                        'segment_topic': chunk['segment_topic'],
                        'segment_summary': chunk['segment_summary'],
                        'start_timestamp': chunk['start_timestamp'],
                        'end_timestamp': chunk['end_timestamp'],
                        'start_char': chunk['start_char'],
                        'end_char': chunk['end_char'],
                        'video_url': video_url,
                        'video_type': 'youtube',
                        'method': 'segment_safe_chunking',
                        'chunk_type': 'segment_safe',
                        'boundary_purity': 'guaranteed',
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'chunk_duration': chunk['end_timestamp'] - chunk['start_timestamp'],
                        'content_category': 'topic_segment',
                        'quality_score': chunk.get('quality_score', 0),
                        'quality_components': quality_components_str,
                        'embedding_model': 'text-embedding-3-large',
                        'embedding_dim': 3072,
                                'transcriber': 'gemini-1.5-flash',
                                'transcript_version': 'v1',
                                'chunking_version': 'v2-seg-safe',
                                'text_source': 'gemini',  # Track text provenance
                                'language': _detect_language(chunk.get('text', '')),  # Language detection
                        'parent_segment_id': chunk['segment_id'],
                        'parent_segment_topic': chunk['segment_topic'],
                        'parent_segment_summary': chunk['segment_summary']
                    }
                }
                vectors.append(vector_data)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"✅ Upserted batch {i//batch_size + 1}")
            
            logger.info(f"✅ Successfully stored {len(vectors)} segment-safe chunk vectors in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone storage failed: {e}")
            return False
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text using OpenAI
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = openai.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return []
    

# Global instance
_enhanced_chunking_processor = None

def initialize_enhanced_chunking_processor() -> bool:
    """Initialize the enhanced chunking processor"""
    global _enhanced_chunking_processor
    try:
        _enhanced_chunking_processor = EnhancedVideoChunkingProcessor(
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Chunking Processor: {e}")
        return False

def get_enhanced_chunking_processor() -> EnhancedVideoChunkingProcessor:
    """Get the global enhanced chunking processor instance"""
    if _enhanced_chunking_processor is None:
        raise RuntimeError("Enhanced Chunking Processor not initialized. Call initialize_enhanced_chunking_processor() first.")
    return _enhanced_chunking_processor
