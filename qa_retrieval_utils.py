#!/usr/bin/env python3
"""
Q&A Retrieval Utilities
Helper functions for Q&A processing and retrieval
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class QARetrievalUtils:
    """Utility class for Q&A retrieval operations"""
    
    def __init__(self):
        """Initialize Q&A retrieval utilities"""
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for better matching"""
        if not text:
            return []
        
        # Simple keyword extraction - remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words (simple approach)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap"""
        if not text1 or not text2:
            return 0.0
        
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_best_matches(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find best matching candidates for a query"""
        if not query or not candidates:
            return []
        
        # Calculate similarity scores
        scored_candidates = []
        for candidate in candidates:
            text = candidate.get('text', '')
            if text:
                similarity = self.calculate_similarity(query, text)
                scored_candidates.append({
                    **candidate,
                    'similarity_score': similarity
                })
        
        # Sort by similarity score (descending)
        scored_candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Return top k results
        return scored_candidates[:top_k]
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS or HH:MM:SS format"""
        if seconds < 0:
            return "0:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def extract_time_range(self, text: str, start_time: float, end_time: float) -> Dict:
        """Extract time range information from text"""
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'formatted_start': self.format_timestamp(start_time),
            'formatted_end': self.format_timestamp(end_time),
            'formatted_duration': self.format_timestamp(end_time - start_time)
        }
    
    def create_qa_response(self, answer: str, sources: List[Dict], confidence: float = 0.0) -> Dict:
        """Create a standardized Q&A response"""
        return {
            'answer': answer,
            'sources': sources,
            'confidence_score': confidence,
            'total_sources': len(sources),
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
    
    def validate_transcript_data(self, transcript_data: Dict) -> bool:
        """Validate transcript data structure"""
        required_fields = ['segments', 'language', 'duration']
        
        for field in required_fields:
            if field not in transcript_data:
                self.logger.warning(f"Missing required field '{field}' in transcript data")
                return False
        
        # Check segments structure
        segments = transcript_data.get('segments', [])
        if not isinstance(segments, list):
            self.logger.warning("Segments must be a list")
            return False
        
        # Check each segment has required fields
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                self.logger.warning(f"Segment {i} must be a dictionary")
                return False
            
            required_segment_fields = ['text', 'start', 'end']
            for field in required_segment_fields:
                if field not in segment:
                    self.logger.warning(f"Segment {i} missing required field '{field}'")
                    return False
        
        return True
    
    def merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping transcript segments"""
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        merged = []
        current_segment = sorted_segments[0].copy()
        
        for segment in sorted_segments[1:]:
            # Check if segments overlap
            if segment.get('start', 0) <= current_segment.get('end', 0):
                # Merge segments
                current_segment['end'] = max(current_segment.get('end', 0), segment.get('end', 0))
                current_segment['text'] += ' ' + segment.get('text', '')
            else:
                # No overlap, add current segment and start new one
                merged.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment
        merged.append(current_segment)
        
        return merged
    
    def extract_context_around_match(self, segments: List[Dict], match_segment: Dict, context_seconds: float = 30.0) -> List[Dict]:
        """Extract context segments around a match"""
        if not segments or not match_segment:
            return []
        
        match_start = match_segment.get('start', 0)
        match_end = match_segment.get('end', 0)
        
        context_start = max(0, match_start - context_seconds)
        context_end = match_end + context_seconds
        
        context_segments = []
        for segment in segments:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            # Check if segment overlaps with context window
            if (segment_start < context_end and segment_end > context_start):
                context_segments.append(segment)
        
        return context_segments
