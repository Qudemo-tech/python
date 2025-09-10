#!/usr/bin/env python3
"""
Unified Chunking Configuration for Video Processing
Standardized parameters for optimal Q&A performance
"""

# Unified chunking parameters for all processors
UNIFIED_CHUNKING_CONFIG = {
    # Core chunking parameters
    'CHUNK_SIZE': 300,           # characters (optimal for Q&A)
    'MAX_DURATION': 10,          # seconds (optimal for timestamp precision)
    'MIN_DURATION': 3,           # seconds (avoid too short chunks)
    'OVERLAP': 50,               # characters (maintain context)
    'MIN_CONTENT_LENGTH': 30,    # minimum characters for valid chunk
    
    # Timestamp patterns (flexible)
    'TIMESTAMP_PATTERNS': [
        r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]',  # [MM:SS] or [HH:MM:SS]
        r'\[(\d+)\]',                            # [SS] seconds only
        r'(\d{1,2}):(\d{2})(?::(\d{2}))?',      # MM:SS or HH:MM:SS (no brackets)
        r'(\d+)s',                               # 120s format
        r'(\d+)m(\d+)s',                        # 2m30s format
    ],
    
    # Duration estimation
    'WORDS_PER_SECOND': 2.5,     # average speaking rate
    'CHARS_PER_SECOND': 12.5,    # average characters per second
    
    # Quality thresholds
    'MIN_QUALITY_SCORE': 0.3,    # minimum content quality
    'MAX_VIDEO_DURATION': 3600,  # 1 hour max (reasonable limit)
    
    # Q&A optimized metadata fields
    'ESSENTIAL_METADATA': [
        'text', 'start_timestamp', 'end_timestamp', 'video_url', 
        'chunk_type', 'content_quality', 'word_count', 'is_complete_sentence'
    ],
    
    # Optional metadata (can be added if needed)
    'OPTIONAL_METADATA': [
        'topic_keywords', 'speaker_info', 'content_category'
    ]
}

# Semantic boundary detection patterns
SEMANTIC_BOUNDARY_PATTERNS = {
    'topic_transitions': [
        r'\b(now|next|moving on|let\'s talk about|another thing|also|furthermore)\b',
        r'\b(first|second|third|finally|in conclusion|to summarize)\b',
        r'\b(so|anyway|alright|okay|well)\b',
    ],
    'sentence_endings': [
        r'[.!?]\s+[A-Z]',  # sentence followed by capital letter
        r'[.!?]\s*\n',     # sentence at end of line
    ],
    'question_patterns': [
        r'\b(what|how|why|when|where|which|who)\b.*\?',
        r'\b(can you|do you|have you|are you)\b.*\?',
    ]
}

# Content quality scoring weights
CONTENT_QUALITY_WEIGHTS = {
    'word_count': 0.3,           # longer content generally better
    'sentence_completeness': 0.2, # complete sentences preferred
    'keyword_density': 0.2,      # relevant keywords
    'readability': 0.15,         # clear, readable text
    'topic_relevance': 0.15,     # relevance to main topic
}

def get_chunking_config():
    """Get unified chunking configuration"""
    return UNIFIED_CHUNKING_CONFIG

def get_semantic_patterns():
    """Get semantic boundary detection patterns"""
    return SEMANTIC_BOUNDARY_PATTERNS

def get_quality_weights():
    """Get content quality scoring weights"""
    return CONTENT_QUALITY_WEIGHTS
