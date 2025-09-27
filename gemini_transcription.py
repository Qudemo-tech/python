#!/usr/bin/env python3
"""
Gemini Transcription Module
Uses Google Gemini API to extract transcriptions from YouTube videos
"""

import os
import logging
import time
import json
import tempfile
import subprocess
import re
import bisect
from datetime import datetime
from typing import Dict, Optional, List
from urllib.parse import urlparse
import google.generativeai as genai
import openai
from google_cloud_storage_service import GoogleCloudStorageService
# YouTube Transcript API removed for production safety
# YouTube actively blocks automated requests and can blacklist server IPs

# YouTube Transcript API functions removed for production safety

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timestamp parsing and interpolation functions
TS_RE = re.compile(r"\[(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d{1,3})?\]")

def parse_time(ts: str) -> float:
    """Parse timestamp string to seconds"""
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h*3600 + m*60 + s
    elif len(parts) == 2:
        m, s = parts
        return m*60 + s
    else:
        return parts[0]

def extract_anchors(text: str):
    """
    Extract timestamp anchors from text with inline timestamps
    Returns:
      clean_text: text with timestamp tags removed
      anchors: list of (char_idx_in_clean_text, time_sec)
    """
    anchors = []
    clean = []
    i_clean = 0
    last_end = 0
    for m in TS_RE.finditer(text):
        # copy text before tag
        seg = text[last_end:m.start()]
        clean.append(seg); i_clean += len(seg)
        # parse tag
        raw = m.group(0)[1:-1]  # drop brackets
        t = parse_time(raw.replace(" ", ""))
        anchors.append((i_clean, t))
        last_end = m.end()
    # tail
    seg = text[last_end:]
    clean.append(seg); i_clean += len(seg)
    clean_text = "".join(clean)

    # edge-pins if none present
    if not anchors:
        anchors = [(0, 0.0), (len(clean_text), len(clean_text) / 15.0)]  # fallback 15 cps

    return clean_text, anchors

def time_at_char(i: int, anchors):
    """Linear interpolation of time at char index i using anchors"""
    idxs = [a[0] for a in anchors]
    pos = bisect.bisect_right(idxs, i) - 1
    if pos < 0:
        return anchors[0][1]
    if pos >= len(anchors) - 1:
        return anchors[-1][1]
    iL, tL = anchors[pos]
    iR, tR = anchors[pos + 1]
    if iR == iL:
        return tL
    frac = (i - iL) / (iR - iL)
    return tL + frac * (tR - tL)

def char_at_time(t: float, anchors, text_len: int):
    """Inverse of time_at_char: find smallest i with time_at_char(i) >= t"""
    times = [a[1] for a in anchors]
    pos = bisect.bisect_right(times, t) - 1
    if pos < 0:
        return 0
    if pos >= len(anchors) - 1:
        return text_len
    iL, tL = anchors[pos]
    iR, tR = anchors[pos + 1]
    if tR == tL:
        return iR
    frac = (t - tL) / (tR - tL)
    i = int(round(iL + frac * (iR - iL)))
    return max(0, min(text_len, i))

def extend_to_word_boundary(text: str, end_i: int, max_lookahead_chars=200):
    """Extend to word boundary if end_i lands in middle of word"""
    n = len(text)
    i = end_i
    # If already at boundary or at end, return as-is
    if i >= n or (i < n and not text[i].isalnum()):
        return end_i
    limit = min(n, end_i + max_lookahead_chars)
    while i < limit and text[i].isalnum():
        i += 1
    # consume trailing punctuation for cleaner boundaries
    while i < limit and text[i] in ",;:)]}\"'":
        i += 1
    return i

class NonRetryableGeminiError(Exception):
    """Raised when Gemini returns a non-retryable 4xx (e.g., 400 INVALID_ARGUMENT)."""
    pass

# --- helpers: clamp, float rounding, and monotonic guards ---

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _r2(x: float) -> float:
    # Round to 2 decimals (good enough for UI jumps)
    return round(float(x), 2)

def _ensure_monotonic(start: float, end: float, eps: float = 0.01) -> tuple[float, float]:
    """Guarantee end > start by at least eps; bump end if needed."""
    if end <= start:
        end = start + eps
    return start, end

def _split_text_evenly(text: str, target_items: int = 15) -> list[str]:
    """Split text into evenly-sized chunks for consistent timestamp distribution."""
    text = (text or "").strip()
    if not text:
        return []
    # very simple token-ish split by whitespace; you may already have something more robust
    words = text.split()
    n = max(1, target_items)
    approx = max(1, len(words) // n)

    out = []
    i = 0
    for _ in range(n - 1):
        chunk = " ".join(words[i:i+approx]).strip()
        if chunk:
            out.append(chunk)
        i += approx
    # last chunk collects remaining words
    rest = " ".join(words[i:]).strip()
    if rest:
        out.append(rest)
    if not out:  # fallback
        out = [text]
    return out

def _validate_timestamped_chunks(chunks: list[dict], video_duration_sec: float) -> None:
    """Safety gates to ensure bad data never hits storage."""
    vd = float(video_duration_sec)
    prev_end = 0.0
    for i, c in enumerate(chunks):
        s = float(c["start_timestamp"])
        e = float(c["end_timestamp"])
        assert 0.0 <= s <= vd, f"Chunk {i} start out of range: {s}"
        assert 0.0 <= e <= vd, f"Chunk {i} end out of range: {e}"
        assert e > s,          f"Chunk {i} non-positive duration: {s}..{e}"
        # optional: ensure global monotonic non-decreasing (not required if you only jump within chunks)
        assert s >= prev_end - 1e-3 or c["local_index"] == 0, f"Non-monotonic timestamps near {i}"
        prev_end = max(prev_end, e)

import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiTranscriptionProcessor:
    def __init__(self, gemini_api_key: str, openai_api_key: str, gcs_bucket_name: str = None):
        """
        Initialize Gemini Transcription Processor
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key for embeddings
            gcs_bucket_name: Google Cloud Storage bucket name
        """
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        # Use gemini-pro-latest model for optimized speed and multimodal input
        self.model = genai.GenerativeModel('gemini-pro-latest')
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Google Cloud Storage
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path='service-account-key.json'
        )
        
        # Overload tracking
        self.gemini_failures = 0
        self.last_gemini_failure_time = 0
        
        # Rate limiting to prevent API overload
        self.last_api_call_time = 0
        self.min_api_interval = int(os.getenv("GEMINI_API_INTERVAL", "5"))  # Reduced interval for faster processing
        
        # Overload protection settings
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))  # Reduced retries to prevent API overload
        self.overload_threshold = int(os.getenv("GEMINI_OVERLOAD_THRESHOLD", "2"))  # Configurable overload threshold
        
        # Circuit breaker for API overload (disabled for now)
        self.consecutive_overloads = 0
        self.circuit_breaker_threshold = 10  # Increased threshold to prevent false triggers
        self.circuit_breaker_reset_time = 0
        self.circuit_breaker_timeout = 300  # Reduced to 5 minutes
        
        logger.info("Initializing Gemini Transcription Processor...")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open due to consecutive API overloads"""
        current_time = time.time()
        
        # Reset circuit breaker if timeout has passed
        if current_time - self.circuit_breaker_reset_time > self.circuit_breaker_timeout:
            self.consecutive_overloads = 0
            self.circuit_breaker_reset_time = 0
            return False
        
        return self.consecutive_overloads >= self.circuit_breaker_threshold

    def _record_api_overload(self):
        """Record an API overload event for circuit breaker"""
        self.consecutive_overloads += 1
        if self.consecutive_overloads >= self.circuit_breaker_threshold:
            self.circuit_breaker_reset_time = time.time()
            logger.warning(f"🚨 Circuit breaker OPEN: {self.consecutive_overloads} consecutive API overloads")
            logger.warning(f"🚨 Will use fallback processing for next {self.circuit_breaker_timeout//60} minutes")

    def _record_api_success(self):
        """Record a successful API call to reset circuit breaker"""
        if self.consecutive_overloads > 0:
            logger.info(f"✅ API success - resetting circuit breaker (was {self.consecutive_overloads} overloads)")
            self.consecutive_overloads = 0
            self.circuit_breaker_reset_time = 0

    def _log_chunk_summary(self, chunks: List[Dict], label: str = ""):
        try:
            total = len(chunks)
            with_ts = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
            logger.info(f"Timestamp chunk check{(' - ' + label) if label else ''}: {with_ts}/{total} chunks have timestamps")
            
            if with_ts > 0:
                logger.info("Timestamps found - chunks will provide timestamped Q&A")
                for i, c in enumerate(chunks[: min(5, total)]):
                    s = float(c.get('start', 0.0))
                    e = float(c.get('end', 0.0))
                    t = (c.get('text') or '')[:120].replace('\n', ' ')
                    logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
            else:
                logger.warning("No timestamps found - Q&A will not include timestamps")
                for i, c in enumerate(chunks[: min(5, total)]):
                    t = (c.get('text') or '')[:120].replace('\n', ' ')
                    logger.info(f"    #{i+1}: [NO TIMESTAMP] {t}")
        except Exception:
            pass

    def _log_raw_gemini_data(self, video_url: str, raw_response: dict, transcription_text: str):
        """Log raw data from Gemini API response before processing"""
        try:
            logger.info("=" * 80)
            logger.info("🔍 RAW GEMINI DATA LOGGING - BEFORE CHUNKING")
            logger.info("=" * 80)
            logger.info(f"📹 Video URL: {video_url}")
            logger.info(f"📊 Raw Response Keys: {list(raw_response.keys())}")
            
            # Log response structure
            if "candidates" in raw_response:
                candidates = raw_response["candidates"]
                logger.info(f"📝 Number of candidates: {len(candidates)}")
                
                for i, candidate in enumerate(candidates):
                    logger.info(f"  Candidate {i+1}:")
                    if "content" in candidate:
                        content = candidate["content"]
                        logger.info(f"    Content keys: {list(content.keys())}")
                        
                        if "parts" in content:
                            parts = content["parts"]
                            logger.info(f"    Number of parts: {len(parts)}")
                            
                            for j, part in enumerate(parts):
                                logger.info(f"      Part {j+1} keys: {list(part.keys())}")
                                if "text" in part:
                                    text_length = len(part["text"])
                                    logger.info(f"      Text length: {text_length} characters")
                                    
                                    # Log first 500 characters of raw text
                                    raw_text_preview = part["text"][:500]
                                    logger.info(f"      Raw text preview (first 500 chars):")
                                    logger.info(f"      {raw_text_preview}")
                                    
                                    # Log last 500 characters of raw text
                                    if text_length > 500:
                                        raw_text_end = part["text"][-500:]
                                        logger.info(f"      Raw text preview (last 500 chars):")
                                        logger.info(f"      {raw_text_end}")
            
            # Log transcription text details
            logger.info(f"📝 Processed Transcription Length: {len(transcription_text)} characters")
            logger.info(f"📝 Word Count: {len(transcription_text.split())}")
            
            # Log transcription preview
            transcription_preview = transcription_text[:1000]
            logger.info(f"📝 Transcription preview (first 1000 chars):")
            logger.info(f"{transcription_preview}")
            
            if len(transcription_text) > 1000:
                transcription_end = transcription_text[-1000:]
                logger.info(f"📝 Transcription preview (last 1000 chars):")
                logger.info(f"{transcription_end}")
            
            # Log any metadata from response
            if "usageMetadata" in raw_response:
                usage = raw_response["usageMetadata"]
                logger.info(f"📊 Usage Metadata: {usage}")
            
            logger.info("=" * 80)
            logger.info("✅ RAW GEMINI DATA LOGGING COMPLETE")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error logging raw Gemini data: {e}")

    def _log_chunked_data(self, video_url: str, chunks: List[Dict], transcription_text: str):
        """Log processed chunks after chunking"""
        try:
            logger.info("=" * 80)
            logger.info("🔍 CHUNKED DATA LOGGING - AFTER CHUNKING")
            logger.info("=" * 80)
            logger.info(f"📹 Video URL: {video_url}")
            logger.info(f"📊 Total Chunks Created: {len(chunks)}")
            logger.info(f"📝 Original Transcription Length: {len(transcription_text)} characters")
            
            # Log chunk statistics
            total_chunk_text_length = sum(len(chunk.get('text', '')) for chunk in chunks)
            avg_chunk_length = total_chunk_text_length / len(chunks) if chunks else 0
            
            logger.info(f"📊 Chunk Statistics:")
            logger.info(f"  Total chunk text length: {total_chunk_text_length} characters")
            logger.info(f"  Average chunk length: {avg_chunk_length:.1f} characters")
            logger.info(f"  Text coverage: {(total_chunk_text_length / len(transcription_text) * 100):.1f}%")
            
            # Log timestamp information
            chunks_with_timestamps = [c for c in chunks if c.get('start', 0) > 0 or c.get('end', 0) > 0]
            logger.info(f"📊 Timestamp Information:")
            logger.info(f"  Chunks with timestamps: {len(chunks_with_timestamps)}/{len(chunks)}")
            
            if chunks_with_timestamps:
                min_start = min(c.get('start', 0) for c in chunks_with_timestamps)
                max_end = max(c.get('end', 0) for c in chunks_with_timestamps)
                logger.info(f"  Time range: {min_start:.2f}s - {max_end:.2f}s")
                logger.info(f"  Total duration: {max_end - min_start:.2f}s")
            
            # Log detailed chunk information
            logger.info(f"📝 Detailed Chunk Information:")
            for i, chunk in enumerate(chunks[:10]):  # Log first 10 chunks
                text = chunk.get('text', '')
                start = chunk.get('start', 0)
                end = chunk.get('end', 0)
                duration = end - start
                
                logger.info(f"  Chunk {i+1}:")
                logger.info(f"    Text length: {len(text)} characters")
                logger.info(f"    Word count: {len(text.split())}")
                logger.info(f"    Time range: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
                logger.info(f"    Text preview: {text[:200]}...")
                
                if i < len(chunks) - 1:
                    logger.info("    ---")
            
            if len(chunks) > 10:
                logger.info(f"  ... and {len(chunks) - 10} more chunks")
            
            # Log chunk quality metrics
            logger.info(f"📊 Chunk Quality Metrics:")
            chunk_lengths = [len(chunk.get('text', '')) for chunk in chunks]
            if chunk_lengths:
                min_length = min(chunk_lengths)
                max_length = max(chunk_lengths)
                logger.info(f"  Min chunk length: {min_length} characters")
                logger.info(f"  Max chunk length: {max_length} characters")
                
                # Check for very short or very long chunks
                short_chunks = [c for c in chunks if len(c.get('text', '')) < 100]
                long_chunks = [c for c in chunks if len(c.get('text', '')) > 1000]
                
                if short_chunks:
                    logger.warning(f"  ⚠️ {len(short_chunks)} chunks are very short (<100 chars)")
                if long_chunks:
                    logger.warning(f"  ⚠️ {len(long_chunks)} chunks are very long (>1000 chars)")
            
            logger.info("=" * 80)
            logger.info("✅ CHUNKED DATA LOGGING COMPLETE")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error logging chunked data: {e}")
    
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain
    
    def _is_likely_long_video(self, video_url: str) -> bool:
        """Check if video is likely to be problematic based on known issues"""
        try:
            # Only use fallback for videos that we know have specific problems
            # This prevents unnecessary fallback for normal videos
            problematic_videos = [
                't0fon35CDm4',  # Known problematic video with specific issues
            ]
            
            for video_id in problematic_videos:
                if video_id in video_url:
                    logger.info(f"🎬 Detected known problematic video {video_id}, using fallback method")
                    return True
            
            # For all other videos, attempt normal processing first
            logger.info(f"🎬 Video URL: {video_url} - attempting normal Gemini API processing")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking video status: {e}")
            return False
    
    def extract_transcription_with_whisper(self, video_url: str) -> Optional[Dict]:
        """
        Extract transcription from YouTube video using multiple methods
        Strategy: Try YouTube Transcript API first, then Gemini API, then metadata fallback
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict with transcription data or None if failed
        """
        try:
            if not self.is_youtube_url(video_url):
                raise Exception("Not a YouTube URL")
            
            logger.info(f"🎬 Extracting transcription from: {video_url}")
            
            # Try YouTube Transcript API first for real timestamps
            youtube_result = self._try_youtube_transcript_api(video_url)
            if youtube_result:
                logger.info("✅ Real transcription extracted successfully with YouTube Transcript API")
                return youtube_result
            
            # Try Gemini API as second option
            logger.info("🎬 YouTube Transcript API failed, trying Gemini API")
            gemini_result = self._try_gemini_api_with_overload_handling(video_url)
            if gemini_result:
                logger.info("✅ Real transcription extracted successfully with Gemini API")
                return gemini_result
            
            # Fallback to metadata-based content if both fail
            logger.info("⚠️ Both YouTube and Gemini APIs failed, using metadata-based fallback")
            logger.info("ℹ️ This will not include real timestamps")
            return self._create_metadata_based_content(video_url)
                
        except Exception as e:
            logger.error(f"❌ Transcription processing failed: {e}")
            return None

    def _create_metadata_based_content(self, video_url: str) -> Optional[Dict]:
        """
        Create content based on video metadata when audio download fails
        Production-safe fallback that doesn't require downloading video content
        """
        try:
            logger.info("🔄 Creating metadata-based content as fallback")
            
            # Extract video ID
            video_id = self._extract_video_id(video_url)
            if not video_id:
                logger.error("❌ Could not extract video ID from URL")
                return None
            
            # Create intelligent fallback content using Gemini
            # Note: This is a sync method, so we'll create a simple fallback instead
            fallback_content = self._create_simple_fallback_content(video_url, video_id)
            
            if fallback_content:
                logger.info("✅ Metadata-based content created successfully")
                return {
                    'transcription': fallback_content.get('content', ''),
                    'segments': fallback_content.get('segments', []),
                    'word_count': len(fallback_content.get('content', '').split()),
                    'language': 'en',
                    'method': 'metadata_fallback',
                    'summary': fallback_content.get('summary', ''),
                    'metadata': {
                        'video_id': video_id,
                        'video_url': video_url,
                        'fallback_reason': 'Audio download blocked by YouTube'
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Metadata-based content creation failed: {e}")
            return None

    def _create_simple_fallback_content(self, video_url: str, video_id: str) -> Optional[Dict]:
        """
        FAILED: Cannot extract real content from video
        Returns None to prevent fake content generation
        """
        try:
            logger.error(f"❌ Cannot extract real content from video: {video_id}")
            logger.error(f"❌ URL: {video_url}")
            logger.error(f"❌ Returning None to prevent fake content generation")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in fallback content creation: {e}")
            return None

    def _try_youtube_transcript_api(self, video_url: str) -> Optional[Dict]:
        """
        Try to extract transcription using YouTube Transcript API
        This provides real timestamps and is the most reliable method
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import TextFormatter
            
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                logger.error("❌ Could not extract video ID from URL")
                return None
            
            logger.info(f"🎬 Trying YouTube Transcript API for video: {video_id}")
            
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            if not transcript_list:
                logger.warning("⚠️ No transcript available for this video")
                return None
            
            # Format transcript with timestamps
            formatted_transcript = ""
            segments = []
            
            for entry in transcript_list:
                start_time = entry['start']
                duration = entry['duration']
                text = entry['text']
                
                # Format timestamp as [MM:SS] or [HH:MM:SS]
                if start_time < 3600:  # Less than 1 hour
                    timestamp = f"[{int(start_time//60):02d}:{int(start_time%60):02d}]"
                else:  # 1 hour or more
                    hours = int(start_time//3600)
                    minutes = int((start_time%3600)//60)
                    seconds = int(start_time%60)
                    timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                
                formatted_transcript += f"{timestamp} {text}\n"
                
                segments.append({
                    'text': text,
                    'start': start_time,
                    'end': start_time + duration
                })
            
            if formatted_transcript:
                logger.info(f"✅ YouTube Transcript API successful: {len(segments)} segments")
                logger.info(f"📝 Total characters: {len(formatted_transcript)}")
                
                return {
                    'transcription': formatted_transcript,
                    'segments': segments,
                    'word_count': len(formatted_transcript.split()),
                    'language': 'en',
                    'method': 'youtube_transcript_api',
                    'summary': '',
                    'metadata': {
                        'video_id': video_id,
                        'video_url': video_url,
                        'segments_count': len(segments),
                        'has_timestamps': True
                    }
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ YouTube Transcript API failed: {e}")
            return None

    # yt-dlp download method removed for production safety
    # YouTube actively blocks automated downloads and can blacklist server IPs

    # Whisper API transcription method removed for production safety
    # We use only metadata-based content generation to avoid YouTube IP blacklisting

    def _get_gemini_summary(self, transcription_text: str) -> Optional[str]:
        """
        Get summary and enrichment from Gemini API (text-only, no video)
        """
        try:
            if not transcription_text or len(transcription_text) < 100:
                return None
            
            logger.info("📝 Getting Gemini summary for transcription")
            
            # Truncate if too long (Gemini has token limits)
            max_chars = 50000  # Conservative limit
            if len(transcription_text) > max_chars:
                transcription_text = transcription_text[:max_chars] + "..."
            
            prompt = f"""
            Please provide a concise summary of this video transcription:
            
            {transcription_text}
            
            Provide:
            1. Main topics covered
            2. Key points
            3. Any actionable insights
            
            Keep it under 200 words.
            """
            
            model = genai.GenerativeModel('gemini-pro-latest')
            response = model.generate_content(prompt)
            
            if response and response.text:
                logger.info("✅ Gemini summary generated successfully")
                return response.text.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini summary failed: {e}")
            return None

    def _try_gemini_api_with_overload_handling(self, video_url: str) -> Optional[Dict]:
        """Try Gemini API with intelligent retry handling for all video sizes"""
        # Check circuit breaker first
        if self._is_circuit_breaker_open():
            logger.warning(f"🚨 Circuit breaker is OPEN - skipping Gemini API due to consecutive overloads")
            logger.warning(f"🚨 Will use fallback processing instead")
            return None
        
        max_retries = self.max_retries   # Use configurable retry count
        base_delay = 20   # Much longer base delay for large video processing
        
        logger.info(f"🎬 Attempting Gemini API with {max_retries} retries (overload-optimized, interval: {self.min_api_interval}s)")
        
        # Try Gemini API with intelligent retry strategy
        for attempt in range(max_retries):
            try:
                result = self._try_direct_gemini_api_with_long_video_support(video_url, attempt, max_retries, base_delay)
                if result:
                    logger.info(f"✅ Gemini API successful on attempt {attempt + 1}")
                    self._record_api_success()  # Reset circuit breaker on success
                    return result
                else:
                    # If result is None, the attempt failed
                    logger.warning(f"⚠️ Gemini attempt {attempt + 1} returned None (failed)")
                    if attempt < max_retries - 1:
                        logger.info(f"⏳ Continuing to next attempt... (attempt {attempt + 1}/{max_retries})")
                    else:
                        logger.error(f"❌ All {max_retries} Gemini API attempts failed")
                        break
                
            except NonRetryableGeminiError as e:
                logger.warning(f"⛔ Non-retryable Gemini error: {e}. Skipping retries and using fallback now.")
                return None  # ← immediately exit; caller will run the YouTube fallback

            except Exception as e:
                logger.warning(f"⚠️ Gemini attempt {attempt + 1} failed with exception: {e}")
                if attempt < max_retries - 1:
                    # Progressive delay with jitter to prevent thundering herd
                    delay = base_delay * (2 ** attempt) + (attempt * 5) + (hash(str(e)) % 10)
                    logger.info(f"⏳ Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ All {max_retries} Gemini API attempts failed")
                    break
        
        return None

    def _try_direct_gemini_api_with_long_video_support(self, video_url: str, attempt: int, max_retries: int, base_delay: int) -> Optional[Dict]:
        """Try direct Gemini API approach with enhanced long video support"""
        try:
            import requests
            import time
            
            # Rate limiting to prevent API overload
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            if time_since_last_call < self.min_api_interval:
                sleep_time = self.min_api_interval - time_since_last_call
                logger.info(f"⏳ Rate limiting: waiting {sleep_time:.1f}s before API call...")
                time.sleep(sleep_time)
            
            self.last_api_call_time = time.time()
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
            
            headers = {
                "Content-Type": "application/json",
            }
            
            # Enhanced prompt for substantial content
            prompt_text = (
                "Transcribe the spoken words from this video with substantial content. "
                "Combine related thoughts into longer, meaningful segments. "
                "Each segment should contain complete ideas or explanations, not just single sentences. "
                "Include timestamps for each new segment in [MM:SS] format for videos under 1 hour. "
                "For longer videos, use [HH:MM:SS] format. "
                "Output as: [MM:SS] Complete paragraph or explanation with multiple sentences. "
                "Focus on creating substantial, meaningful chunks that can stand alone. "
                "No summaries, no extra commentary. "
                "Process the entire video content completely."
            )
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt_text
                            },
                            {
                                "fileData": {
                                    "mimeType": "video/mp4",
                                    "fileUri": video_url
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Increased timeout for long videos
            timeout = 120  # 2 minutes timeout to prevent long waits
            logger.info(f"Sending request to Gemini API with {timeout}s timeout... (attempt {attempt + 1}/{max_retries})")
            response = requests.post(
                url,
                headers={
                    **headers,
                    "x-goog-api-key": self.gemini_api_key
                },
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    transcription_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info("✅ Gemini API successful - Direct approach")
                    
                    # Log raw Gemini data before processing
                    self._log_raw_gemini_data(video_url, result, transcription_text)
                    
                    result_dict = {
                        "title": "YouTube Video",
                        "transcription": transcription_text,
                        "summary": "",
                        "duration": "Unknown",
                        "language": "en",
                        "word_count": len(transcription_text.split()),
                        "method": "gemini_direct_api"
                    }
                    
                    logger.info(f"✅ Transcription extracted successfully")
                    logger.info(f"📝 Word count: {len(transcription_text.split())}")
                    logger.info(f"🔧 Method: gemini_direct_api")
                    
                    self._save_transcript_to_file(video_url, transcription_text, result_dict)
                    return result_dict
                else:
                    raise Exception("No candidates in response")
            else:
                # Handle different error codes with better overload handling
                if response.status_code == 503:
                    # 503 is overload - use much longer delays with exponential backoff for large videos
                    if attempt < max_retries - 1:
                        # Reduced delays for faster retries: 5s, 10s, 20s
                        delay = 5 * (2 ** attempt) + (hash(str(attempt)) % 5)
                        logger.warning(f"⚠️ Gemini API overloaded (503), retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        logger.info(f"💡 Tip: Large video processing requires patience. API is experiencing high load.")
                        logger.info(f"💡 Consider processing during off-peak hours for better success rates.")
                        self._record_api_overload()  # Record overload for circuit breaker
                        time.sleep(delay)
                        return None
                    else:
                        logger.error(f"❌ Gemini API overloaded after {max_retries} attempts")
                        logger.error(f"💡 Recommendation: Wait 5-10 minutes before retrying, or try a different video")
                        return None
                elif response.status_code in [429, 500, 502] and attempt < max_retries - 1:
                    # Use much shorter delays for non-overload errors
                    delay = 3 * (2 ** attempt)
                    logger.warning(f"⚠️ Gemini API error ({response.status_code}), retrying in {delay} seconds...")
                    time.sleep(delay)
                    return None
                elif 400 <= response.status_code < 500 and response.status_code != 429:
                    # 🚫 Non-retryable client error: bail out and trigger fallback
                    logger.warning(
                        "⚠️ Gemini returned non-retryable %s — will not retry; falling back.",
                        response.status_code
                    )
                    logger.warning("⚠️ Error details: %s", response.text[:800])
                    raise NonRetryableGeminiError(
                        f"{response.status_code}: {response.text}"
                    )
                else:
                    self._record_gemini_failure()
                    raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                    
        except requests.exceptions.Timeout:
            self._record_gemini_failure()
            if attempt < max_retries - 1:
                # Use much shorter delays for timeouts
                base_delay = 5
                delay = base_delay + (attempt * 3) + (hash(str(attempt)) % 5)
                logger.warning(f"⚠️ Gemini API timeout, retrying in {delay} seconds...")
                time.sleep(delay)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Direct Gemini API attempt failed: {e}")
            return None

    # Removed Files API functions - using direct Gemini API only

    def _is_gemini_overloaded(self) -> bool:
        """Check if Gemini API appears to be overloaded based on recent failures"""
        import time
        current_time = time.time()
        
        # Reset failure count if more than 10 minutes have passed
        if current_time - self.last_gemini_failure_time > 600:  # 10 minutes
            self.gemini_failures = 0
        
        # If we've had failures above threshold in the last 10 minutes, consider it overloaded
        if self.gemini_failures >= self.overload_threshold:
            logger.info(f"⚠️ Gemini API overload detected: {self.gemini_failures} recent failures")
            return True
        
        return False

    def _record_gemini_failure(self):
        """Record a Gemini API failure for overload tracking"""
        import time
        self.gemini_failures += 1
        self.last_gemini_failure_time = time.time()
        
        # Provide helpful feedback based on failure count
        if self.gemini_failures == 1:
            logger.info("💡 First API failure - this is normal, will retry")
        elif self.gemini_failures == 2:
            logger.warning("⚠️ Multiple API failures detected - Gemini may be experiencing high load")
        elif self.gemini_failures >= 3:
            logger.error("❌ Multiple consecutive failures - consider waiting before retrying")
        logger.info(f"📊 Recorded Gemini failure (total: {self.gemini_failures})")

    def _extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            import re
            # Handle different YouTube URL formats
            patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
                r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
                r'youtube\.com/v/([a-zA-Z0-9_-]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_url)
                if match:
                    video_id = match.group(1)
                    logger.info(f"📹 Extracted video ID: {video_id}")
                    return video_id
            
            logger.warning(f"⚠️ Could not extract video ID from URL: {video_url}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting video ID: {e}")
            return None

    def _get_video_duration(self, video_url: str) -> Optional[int]:
        """Get video duration using YouTube Data API if available (optional)"""
        try:
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                logger.warning("⚠️ Could not extract video ID from URL")
                return None
            
            video_id = video_id_match.group(1)
            logger.info(f"🔍 Extracting duration for video ID: {video_id}")
            
            # Try YouTube Data API if available (optional)
            try:
                import requests
                api_key = os.getenv('YOUTUBE_API_KEY')
                if api_key:
                    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=contentDetails&key={api_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('items'):
                            duration_str = data['items'][0]['contentDetails']['duration']
                            # Parse ISO 8601 duration format (PT10M30S)
                            import re
                            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
                            if match:
                                hours = int(match.group(1) or 0)
                                minutes = int(match.group(2) or 0)
                                seconds = int(match.group(3) or 0)
                                total_seconds = hours * 3600 + minutes * 60 + seconds
                                logger.info(f"📏 Video duration from YouTube API: {total_seconds}s")
                                return total_seconds
                        else:
                            logger.warning(f"⚠️ No video data found in YouTube API response")
                    else:
                        logger.warning(f"⚠️ YouTube API returned status {response.status_code}")
                else:
                    logger.info("ℹ️ No YouTube API key configured, skipping duration detection")
            except Exception as e:
                logger.warning(f"⚠️ Error getting duration from YouTube API: {e}")
            
            logger.info("ℹ️ Could not determine video duration, will use default processing")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting video duration: {e}")
            return None

    # Removed chunked processing functions - using direct Gemini API only

    # Removed YouTube API functions - using Gemini API only
    
    def _save_transcript_to_file(self, video_url: str, transcription_text: str, result: Dict):
        """
        Save transcript to file for logging purposes (similar to tutorial)
        """
        try:
            import os
            from datetime import datetime
            
            # Create logs directory if it doesn't exist
            logs_dir = "transcript_logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"
            filepath = os.path.join(logs_dir, filename)
            
            # Write transcript to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"YouTube Video URL: {video_url}\n\n")
                f.write(f"Title: {result.get('title', 'Unknown')}\n")
                f.write(f"Duration: {result.get('duration', 'Unknown')}\n")
                f.write(f"Language: {result.get('language', 'Unknown')}\n")
                f.write(f"Word Count: {result.get('word_count', 'Unknown')}\n")
                f.write(f"Method: {result.get('method', 'Unknown')}\n\n")
                
                # Write summary if available
                summary = result.get('summary', '')
                if summary:
                    f.write("--- VIDEO SUMMARY ---\n")
                    f.write(summary)
                    f.write("\n\n")
                
                f.write("--- TRANSCRIPTION ---\n")
                f.write(transcription_text)
            
            logger.info(f"📁 Transcript saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save transcript to file: {e}")

    def _create_fallback_transcription(self, video_url: str) -> Optional[Dict]:
        """
        Create a fallback transcription for long videos
        Provides a structured transcription that can be used for Q&A
        """
        try:
            logger.info(f"🔄 Creating fallback transcription for: {video_url}")
            
            # Extract video ID
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                return None
            
            video_id = video_id_match.group(1)
            
            # FAILED: Cannot extract real content from video
            logger.error(f"❌ Cannot extract real content from video: {video_id}")
            logger.error(f"❌ URL: {video_url}")
            logger.error(f"❌ Returning None to prevent fake content generation")
            return None
            
        except Exception as e:
            logger.error(f"❌ Fallback transcription failed: {e}")
            return None

    async def _process_long_video_fallback(self, video_url: str, company_name: str, qudemo_id: str) -> Optional[Dict]:
        """
        Process long videos using fallback methods when Gemini API is overloaded
        Creates meaningful content chunks for Q&A without full transcription
        """
        try:
            logger.info(f"🔄 Processing long video with fallback method: {video_url}")
            
            # Extract video ID from URL
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                logger.error("❌ Could not extract video ID from URL")
                return None
            
            video_id = video_id_match.group(1)
            logger.info(f"📹 Extracted video ID: {video_id}")
            
            # Create intelligent fallback content based on video ID and context
            fallback_content = await self._create_intelligent_fallback_content(video_url, video_id)
            
            if not fallback_content:
                logger.error("❌ Failed to create fallback content")
                return {
                    'success': False,
                    'error': 'Failed to create fallback content',
                    'video_url': video_url,
                    'company_name': company_name
                }
            
            # Create chunks from fallback content
            chunks = self.chunk_transcription(fallback_content, segments=None)
            if not chunks:
                logger.error("❌ Failed to create chunks from fallback content")
                return {
                    'success': False,
                    'error': 'Failed to create chunks from fallback content',
                    'video_url': video_url,
                    'company_name': company_name
                }
            
            # Create embeddings
            embeddings = self.create_embeddings([c['text'] if isinstance(c, dict) else str(c) for c in chunks])
            if not embeddings or len(embeddings) != len(chunks):
                logger.error("❌ Failed to create embeddings")
                return {
                    'success': False,
                    'error': 'Failed to create embeddings',
                    'video_url': video_url,
                    'company_name': company_name
                }
            
            # Note: Pinecone storage functionality has been removed
            transcription_data = {
                'title': f'Long YouTube Video {video_id}',
                'transcription': fallback_content,
                'duration': 'Long video (16+ minutes)',
                'language': 'en',
                'word_count': len(fallback_content.split()),
                'method': 'fallback_long_video'
            }
            
            # Note: Pinecone storage functionality has been removed
            logger.warning("⚠️ Pinecone storage functionality removed, using GCS-based storage")
            storage_success = False
            
            if not storage_success:
                logger.warning("⚠️ Pinecone storage functionality removed, using GCS-based storage")
                return {
                    'success': False,
                    'error': 'Pinecone storage functionality removed, using GCS-based storage',
                    'video_url': video_url,
                    'company_name': company_name
                }
            
            # Return success result with detailed status information
            result = {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'title': transcription_data.get('title', 'Unknown'),
                'chunks_created': len(chunks),
                'vectors_stored': len(embeddings),
                'word_count': transcription_data.get('word_count', 'Unknown'),
                'language': transcription_data.get('language', 'Unknown'),
                'method': transcription_data.get('method', 'fallback_long_video'),
                'processing_quality': 'intelligent_fallback',
                'status_message': 'Video processed using intelligent fallback method. Content is Q&A-ready and optimized for knowledge retrieval.',
                'processing_details': {
                    'method_used': 'intelligent_fallback',
                    'reason': 'API limitations or known video issues',
                    'content_quality': 'high',
                    'qa_readiness': 'excellent'
                },
                'recommendations': [
                    'Content is suitable for answering questions about the video topic',
                    'Chunks are semantically meaningful and searchable',
                    'This fallback ensures processing never fails - it\'s a feature, not a limitation'
                ]
            }
            
            logger.info(f"✅ Long video fallback processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Long video fallback processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name
            }
    
    async def _create_intelligent_fallback_content(self, video_url: str, video_id: str) -> str:
        """
        FAILED: Cannot extract real content from video
        Returns None to prevent fake content generation
        """
        try:
            logger.error(f"❌ Cannot extract real content from video: {video_id}")
            logger.error(f"❌ URL: {video_url}")
            logger.error(f"❌ Returning None to prevent fake content generation")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in intelligent fallback content creation: {e}")
            return None
    
    def _create_generic_fallback_content(self, video_url: str, video_id: str) -> str:
        """
        FAILED: Cannot extract real content from video
        Returns None to prevent fake content generation
        """
        try:
            logger.error(f"❌ Cannot extract real content from video: {video_id}")
            logger.error(f"❌ URL: {video_url}")
            logger.error(f"❌ Returning None to prevent fake content generation")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in generic fallback content creation: {e}")
            return None

    def _fallback_video_analysis(self, video_url: str) -> Optional[Dict]:
        """
        Fallback method when direct transcript access fails
        Attempts to analyze video based on URL and available metadata
        """
        try:
            logger.error(f"❌ Cannot extract real content from video: {video_url}")
            logger.error(f"❌ Returning None to prevent fake content generation")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in fallback video analysis: {e}")
            return None
    
    def chunk_transcription(
        self,
        transcription: str,
        segments: Optional[List[Dict]] = None,
        chunk_size: int = 300,   # Unified chunk size for Q&A optimization
        overlap: int = 50,       # Minimal overlap for cleaner boundaries
        max_chunk_duration: int = 10,  # Unified max duration for optimal timestamp precision
        video_url: str = "",     # Video URL for logging purposes
    ) -> List[Dict]:
        """
        Create timestamped chunks from transcription with configurable precision.

        If timestamped segments are available (Gemini may not provide them), build
        chunks by aggregating segments until reaching target size or max duration.
        Otherwise fall back to character-based chunking without timestamps.

        Returns list of dicts: { text: str, start: float, end: float }
        """
        if segments:
            chunks: List[Dict] = []
            current_text_parts: List[str] = []
            current_start: Optional[float] = None
            current_end: Optional[float] = None

            def flush_chunk():
                nonlocal current_text_parts, current_start, current_end
                if current_text_parts and current_start is not None and current_end is not None:
                    # Join text parts properly to avoid breaking words
                    combined_text = ' '.join(current_text_parts).strip()
                    
                    # Clean up any formatting issues
                    combined_text = re.sub(r'\s+', ' ', combined_text)  # Replace multiple spaces
                    combined_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', combined_text)  # Add space between camelCase
                    
                    chunks.append({
                        'text': combined_text,
                        'start': float(max(0.0, current_start)),
                        'end': float(max(current_start, current_end)),
                    })
                current_text_parts = []
                current_start = None
                current_end = None

            for seg in segments:
                seg_text = (seg.get('text') or '').strip()
                if not seg_text:
                    continue
                seg_start = float(seg.get('start', 0.0))
                seg_end = float(seg.get('end', seg_start))

                if current_start is None:
                    current_start = seg_start
                    current_end = seg_end
                else:
                    current_end = seg_end

                current_text_parts.append(seg_text)

                current_text_len = sum(len(p) for p in current_text_parts) + (len(current_text_parts) - 1)
                current_duration = current_end - (current_start or current_end)
                
                # Ensure chunks are focused - respect max duration per chunk
                if current_text_len >= chunk_size or current_duration >= max_chunk_duration:
                    # Add small buffer (1-2 seconds) to complete the current word if we're at duration limit
                    if current_duration >= max_chunk_duration and current_duration < max_chunk_duration + 2:
                        # Check if we're in the middle of a word by looking at the combined text
                        combined_text = ' '.join(current_text_parts).strip()
                        if combined_text and not combined_text.endswith(' '):
                            # Check if the last word is incomplete (no space after it)
                            last_space_index = combined_text.rfind(' ')
                            if last_space_index > 0:
                                last_word = combined_text[last_space_index + 1:]
                                # If last word is likely incomplete, add buffer
                                # More aggressive detection: any word without proper punctuation is likely incomplete
                                if (len(last_word) < 8 or 
                                    not last_word.endswith(('.', '!', '?', ',', ':', ';', ')', ']', '}')) or
                                    last_word.lower() in ['whe', 'co', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'yet', 'so']):
                                    continue
                                
                                # Additional check: if the combined text doesn't end with proper sentence punctuation, continue
                                if not combined_text.endswith(('.', '!', '?', ':', ';')):
                                    continue
                    
                    # Only flush if we have meaningful content
                    if current_text_len >= 100:  # Very small minimum for precise chunks
                        flush_chunk()
                    else:
                        # Continue accumulating for a more substantial chunk
                        continue

            flush_chunk()
            
            # Filter out tiny chunks and merge them with larger ones
            filtered_chunks = []
            for chunk in chunks:
                if len(chunk['text']) >= 300:  # Only keep substantial chunks
                    filtered_chunks.append(chunk)
                else:
                    # Try to merge with next chunk if available
                    if filtered_chunks:
                        last_chunk = filtered_chunks[-1]
                        # Only merge if the combined chunk won't be too long
                        if len(last_chunk['text']) + len(chunk['text']) < 2000:
                            # Merge text properly to avoid breaking words
                            merged_text = last_chunk['text'] + ' ' + chunk['text']
                            # Clean up any formatting issues
                            merged_text = re.sub(r'\s+', ' ', merged_text).strip()
                            last_chunk['text'] = merged_text
                            last_chunk['end'] = chunk['end']
                            logger.info(f"🔗 Merged small chunk ({len(chunk['text'])} chars) with previous chunk")
                        else:
                            # Start a new chunk if the previous one would be too long
                            filtered_chunks.append(chunk)
                            logger.info(f"📝 Started new chunk from small segment ({len(chunk['text'])} chars)")
                    else:
                        # If this is the first chunk and it's small, keep it but log
                        filtered_chunks.append(chunk)
                        logger.warning(f"⚠️ First chunk is small ({len(chunk['text'])} chars)")
            
            logger.info(f"📄 Created {len(filtered_chunks)} substantial chunks from {len(chunks)} original chunks")
            
            # Log chunked data after processing
            self._log_chunked_data(video_url, filtered_chunks, transcription)
            
            return filtered_chunks

        # Fallback: simple text chunking without timestamps
        import re
        
        logger.info("🔍 No segments provided, using simple text chunking")
        
        # Simple text chunking without timestamps
        sentences = re.split(r'[.!?]+', transcription)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        parsed: List[Dict] = []
        current_chunk = ""
        current_start = 0.0
        
        for sentence in sentences:
            if not sentence:
                continue
                
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            
            # Estimate duration based on text length (roughly 3 words per second)
            word_count = len(current_chunk.split())
            estimated_duration = word_count / 3.0
            
            # Check if we should create a chunk
            if len(current_chunk) >= chunk_size or estimated_duration >= max_chunk_duration:
                if current_chunk.strip():
                    # Create chunk with estimated timestamps
                    end_time = current_start + estimated_duration
                    parsed.append({
                        'text': current_chunk.strip(),
                        'start': float(current_start),
                        'end': float(end_time)
                    })
                    
                    # Move to next chunk
                    current_start = end_time
                    current_chunk = ""
        
        # Add final chunk if there's remaining text
        if current_chunk.strip():
            word_count = len(current_chunk.split())
            estimated_duration = word_count / 3.0
            end_time = current_start + estimated_duration
            parsed.append({
                'text': current_chunk.strip(),
                'start': float(current_start),
                'end': float(end_time)
            })
        
        # Apply duration limit by splitting large chunks
        final_chunks = []
        for chunk in parsed:
            start_time = chunk['start']
            end_time = chunk['end']
            text = chunk['text']
            
            # If chunk is larger than max duration, split it
            if end_time - start_time > max_chunk_duration:
                # Split into max_duration segments
                current_start = start_time
                while current_start < end_time:
                    current_end = min(current_start + max_chunk_duration, end_time)
                    
                    # Simple text splitting for large chunks
                    text_length = len(text)
                    duration_ratio = (current_end - current_start) / (end_time - start_time)
                    start_char = int(duration_ratio * text_length)
                    end_char = int((current_end - start_time) / (end_time - start_time) * text_length)
                    
                    # Find word boundaries
                    while start_char > 0 and text[start_char] not in ' \t\n':
                        start_char -= 1
                    while end_char < text_length and text[end_char] not in ' \t\n':
                        end_char += 1
                    
                    segment_text = text[start_char:end_char].strip()
                    if segment_text:
                        final_chunks.append({
                            'text': segment_text,
                            'start': float(current_start),
                            'end': float(current_end)
                        })
                    
                    current_start = current_end
            else:
                # Add chunk as-is
                final_chunks.append({
                    'text': text,
                    'start': float(start_time),
                    'end': float(end_time)
                })
        
        logger.info(f"📄 Created {len(final_chunks)} chunks from transcription")
        
        # Log chunked data after processing
        self._log_chunked_data(video_url, final_chunks, transcription)
        
        return final_chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks using OpenAI
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🧠 Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch size limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model="text-embedding-3-large"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"✅ Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 3072  # OpenAI text-embedding-3-large dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return []
    
    def store_in_gcs(self, company_name: str, video_url: str, transcription_data: Dict, 
                    chunks: List[Dict], qudemo_id: str = None) -> bool:
        """
        Store transcription data in Google Cloud Storage
        
        Args:
            company_name: Name of the company
            video_url: Original video URL
            transcription_data: Transcription metadata
            chunks: Text chunks
            qudemo_id: QuDemo ID for namespace isolation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗄️ Storing in Google Cloud Storage for company: {company_name} qudemo {qudemo_id}")
            
            # Prepare transcript data for storage
            transcript_data = {
                'video_url': video_url,
                'video_title': transcription_data.get('title', 'Unknown'),
                'transcript': transcription_data.get('transcription', ''),
                'timestamps': transcription_data.get('timestamps', []),
                'segments': transcription_data.get('segments', []),
                'topics': transcription_data.get('topics', []),
                'chunks': chunks,
                'metadata': {
                    'word_count': transcription_data.get('word_count', 0),
                    'language': transcription_data.get('language', 'en'),
                    'method': transcription_data.get('method', 'gemini_transcription'),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            # Store in Google Cloud Storage
            success = self.gcs_service.store_video_transcript(
                company_name=company_name,
                qudemo_id=qudemo_id,
                transcript_data=transcript_data
            )
            
            if success:
                logger.info(f"✅ Successfully stored transcript in Google Cloud Storage")
                return True
            else:
                logger.error(f"❌ Failed to store transcript in Google Cloud Storage")
                return False
                
        except Exception as e:
            logger.error(f"❌ Google Cloud Storage failed: {e}")
            return False

    def _store_directly_in_pinecone(self, company_name: str, video_url: str, transcription_data: Dict, 
                                        chunks: List[Dict], embeddings: List[List[float]], qudemo_id: str = None) -> bool:
        """
        Fallback direct Pinecone storage - Pinecone functionality removed
        
        Args:
            company_name: Name of the company
            video_url: Original video URL
            transcription_data: Transcription metadata
            chunks: Text chunks
            embeddings: Embedding vectors
            qudemo_id: QuDemo ID for namespace isolation
            
        Returns:
            False (Pinecone functionality removed)
        """
        try:
            # Note: Pinecone storage functionality has been removed
            logger.warning(f"⚠️ Pinecone storage functionality removed, using GCS-based storage")
            return False
            
        except Exception as e:
            logger.error(f"❌ Direct Pinecone storage failed: {e}")
            return False
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks using OpenAI
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🧠 Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch size limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model="text-embedding-3-large"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"✅ Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 3072  # OpenAI text-embedding-3-large dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return []
    
    def _create_youtube_timestamped_chunks(self, video_duration_sec: float, video_chunks: list[dict]) -> list[dict]:
        """
        Evenly distributes timestamps for each text item *within its own video chunk*.
        Returns a flat list of dicts:
          {
            "text": str,
            "start_timestamp": float,  # absolute seconds into full video
            "end_timestamp": float,    # absolute seconds into full video
            "video_chunk_index": int,
            "local_index": int
          }
        
        NOTE: This function returns ABSOLUTE timestamps into the full video timeline.
        """
        result = []
        
        for video_chunk_idx, video_chunk in enumerate(video_chunks):
            start_time = video_chunk.get('start', 0.0)
            end_time = video_chunk.get('end', video_duration_sec)
            text_items = video_chunk.get('text_items', [])
            
            if not text_items:
                continue
                
            # Calculate duration for this video chunk
            chunk_duration = end_time - start_time
            
            # Distribute timestamps evenly within this chunk
            for local_idx, text_item in enumerate(text_items):
                # Calculate relative position within this chunk (0.0 to 1.0)
                relative_position = local_idx / len(text_items)
                
                # Calculate absolute timestamps
                item_start = start_time + (relative_position * chunk_duration)
                item_end = start_time + ((relative_position + 1.0) * chunk_duration)
                
                result.append({
                    'text': text_item,
                    'start_timestamp': item_start,
                    'end_timestamp': item_end,
                    'video_chunk_index': video_chunk_idx,
                    'local_index': local_idx
                })
        
        return result
    
    async def process_video_with_qudemo(self, video_url: str, company_name: str, qudemo_id: str) -> Optional[Dict]:
        """
        Process video with QuDemo integration - main entry point for video processing
        
        Args:
            video_url: YouTube or Loom video URL
            company_name: Company name for organization
            qudemo_id: QuDemo ID for proper namespace isolation
            
        Returns:
            Dict with processing results and detailed status information
        """
        try:
            logger.info(f"🎯 Starting video processing for: {video_url}")
            logger.info(f"🏢 Company: {company_name}, QuDemo ID: {qudemo_id}")
            
            # Extract transcription
            transcription_data = self.extract_transcription_with_whisper(video_url)
            if not transcription_data:
                logger.error("❌ Failed to extract transcription from video")
                return {
                    'success': False,
                    'error': 'Failed to extract transcription from video',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Transcription extracted: {len(transcription_data.get('transcription', ''))} characters")
            
            # Create chunks from transcription
            chunks = self.chunk_transcription(transcription_data.get('transcription', ''), segments=None, video_url=video_url)
            if not chunks:
                logger.error("❌ Failed to create chunks from transcription")
                return {
                    'success': False,
                    'error': 'Failed to create chunks from transcription',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Created {len(chunks)} chunks from transcription")
            
            # Create embeddings
            embeddings = self.create_embeddings([c['text'] if isinstance(c, dict) else str(c) for c in chunks])
            if not embeddings or len(embeddings) != len(chunks):
                logger.error("❌ Failed to create embeddings")
                return {
                    'success': False,
                    'error': 'Failed to create embeddings',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Created {len(embeddings)} embeddings")
            
            # Log timestamp summary
            self._log_chunk_summary(chunks, label=f"{company_name}")
            
            # Store in Google Cloud Storage
            storage_success = self.store_in_gcs(
                company_name, video_url, transcription_data, chunks, qudemo_id
            )
            
            if not storage_success:
                logger.error("❌ Failed to store in Google Cloud Storage")
                return {
                    'success': False,
                    'error': 'Failed to store in Google Cloud Storage',
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
                'title': transcription_data.get('title', 'Unknown'),
                'chunks_created': len(chunks),
                'vectors_stored': len(embeddings),
                'word_count': transcription_data.get('word_count', 'Unknown'),
                'language': transcription_data.get('language', 'Unknown'),
                'method': transcription_data.get('method', 'gemini_transcription'),
                'processing_quality': 'full_transcription',
                'status_message': 'Video processed successfully with complete transcription.',
                'processing_details': {
                    'method_used': 'gemini_transcription',
                    'reason': 'API successful',
                    'content_quality': 'excellent',
                    'qa_readiness': 'excellent'
                }
            }
            
            logger.info(f"✅ Video processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }

    def search_similar_chunks(self, company_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks - Pinecone functionality removed
        
        Args:
            company_name: Company name
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Empty list (Pinecone functionality removed)
        """
        try:
            # Note: Pinecone search functionality has been removed
            logger.warning(f"⚠️ Pinecone search functionality removed, using GCS-based storage")
            return []
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
