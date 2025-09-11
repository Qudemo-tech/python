#!/usr/bin/env python3
"""
Hybrid System Utility Functions
Common utilities for the hybrid video processing and Q&A system
"""

import re
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import logging

logger = logging.getLogger(__name__)

def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """
    Check if two intervals overlap
    
    Args:
        a: (start_time, end_time) for first interval
        b: (start_time, end_time) for second interval
        
    Returns:
        True if intervals overlap, False otherwise
    """
    a_start, a_end = a
    b_start, b_end = b
    return a_start < b_end and b_start < a_end

def youtube_start_url(video_url: str, start_s: float) -> str:
    """
    Build YouTube deep-link URL with start timestamp
    
    Args:
        video_url: Original YouTube URL
        start_s: Start time in seconds
        
    Returns:
        YouTube URL with start timestamp parameter
    """
    try:
        u = urlparse(video_url)
        qs = parse_qs(u.query)
        qs['t'] = [f"{int(start_s)}s"]
        return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs, doseq=True), u.fragment))
    except Exception as e:
        logger.error(f"❌ Error building YouTube start URL: {e}")
        return video_url

def extract_youtube_id(video_url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL
    
    Args:
        video_url: YouTube URL
        
    Returns:
        YouTube video ID or None if not found
    """
    try:
        # Handle various YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error extracting YouTube ID: {e}")
        return None

def format_timestamp(seconds: float) -> str:
    """
    Format timestamp in MM:SS or HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
            
    except Exception as e:
        logger.error(f"❌ Error formatting timestamp: {e}")
        return "00:00"

def normalize_text_for_search(text: str) -> str:
    """
    Normalize text for better search matching
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    try:
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation for exact matching
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
        
    except Exception as e:
        logger.error(f"❌ Error normalizing text: {e}")
        return text.lower()

def calculate_overlap_percentage(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    """
    Calculate overlap percentage between two intervals
    
    Args:
        interval1: (start, end) for first interval
        interval2: (start, end) for second interval
        
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    try:
        start1, end1 = interval1
        start2, end2 = interval2
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        total_duration = max(end1, end2) - min(start1, start2)
        
        if total_duration == 0:
            return 0.0
        
        return overlap_duration / total_duration
        
    except Exception as e:
        logger.error(f"❌ Error calculating overlap percentage: {e}")
        return 0.0

def validate_timestamp_range(start_time: float, end_time: float, max_duration: float = 3600) -> bool:
    """
    Validate timestamp range
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        max_duration: Maximum allowed duration in seconds
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check basic validity
        if start_time < 0 or end_time < 0:
            return False
        
        if start_time >= end_time:
            return False
        
        # Check duration
        duration = end_time - start_time
        if duration > max_duration:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating timestamp range: {e}")
        return False

def merge_overlapping_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge overlapping intervals
    
    Args:
        intervals: List of (start, end) intervals
        
    Returns:
        List of merged intervals
    """
    try:
        if not intervals:
            return []
        
        # Sort by start time
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            
            # If current interval overlaps with last, merge them
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
        
    except Exception as e:
        logger.error(f"❌ Error merging intervals: {e}")
        return intervals

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple heuristics
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    try:
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'a', 'an', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'there', 'here'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(max_keywords)]
        
        return top_keywords
        
    except Exception as e:
        logger.error(f"❌ Error extracting keywords: {e}")
        return []

def calculate_quality_score(text: str) -> float:
    """
    Calculate content quality score based on various factors
    
    Args:
        text: Input text
        
    Returns:
        Quality score (0.0 to 1.0)
    """
    try:
        if not text or len(text.strip()) == 0:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor
        word_count = len(text.split())
        if word_count > 50:
            score += 0.2
        elif word_count > 20:
            score += 0.1
        elif word_count < 10:
            score -= 0.2
        
        # Sentence structure
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings > 0:
            score += 0.1
        
        # Completeness indicators
        if any(word in text.lower() for word in ['complete', 'finished', 'done', 'conclusion']):
            score += 0.1
        
        # Technical content indicators
        if any(word in text.lower() for word in ['step', 'process', 'method', 'technique', 'procedure']):
            score += 0.1
        
        # Clarity indicators
        if any(word in text.lower() for word in ['explain', 'describe', 'demonstrate', 'show', 'example']):
            score += 0.1
        
        return min(1.0, max(0.0, score))
        
    except Exception as e:
        logger.error(f"❌ Error calculating quality score: {e}")
        return 0.5

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove extra spaces and dots
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = re.sub(r'\.+', '.', sanitized)
        
        # Limit length
        if len(sanitized) > 100:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:95] + ('.' + ext if ext else '')
        
        return sanitized
        
    except Exception as e:
        logger.error(f"❌ Error sanitizing filename: {e}")
        return "sanitized_file"

def validate_video_url(url: str) -> Dict:
    """
    Validate video URL and determine platform
    
    Args:
        url: Video URL
        
    Returns:
        Validation result with platform info
    """
    try:
        result = {
            'valid': False,
            'platform': None,
            'video_id': None,
            'error': None
        }
        
        if not url or not isinstance(url, str):
            result['error'] = 'Invalid URL format'
            return result
        
        # Check YouTube
        youtube_id = extract_youtube_id(url)
        if youtube_id:
            result['valid'] = True
            result['platform'] = 'youtube'
            result['video_id'] = youtube_id
            return result
        
        # Check Loom
        if 'loom.com' in url:
            result['valid'] = True
            result['platform'] = 'loom'
            return result
        
        # Check other platforms
        if any(domain in url for domain in ['vimeo.com', 'dailymotion.com', 'twitch.tv']):
            result['valid'] = True
            result['platform'] = 'other'
            return result
        
        result['error'] = 'Unsupported platform'
        return result
        
    except Exception as e:
        logger.error(f"❌ Error validating video URL: {e}")
        return {
            'valid': False,
            'platform': None,
            'video_id': None,
            'error': str(e)
        }
