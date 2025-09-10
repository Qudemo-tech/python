#!/usr/bin/env python3
"""
Unified Chunking Utilities for Video Processing
Standardized chunking logic for optimal Q&A performance
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from unified_chunking_config import (
    get_chunking_config, 
    get_semantic_patterns, 
    get_quality_weights
)

logger = logging.getLogger(__name__)

class UnifiedChunkingProcessor:
    """Unified chunking processor for all video types"""
    
    def __init__(self):
        self.config = get_chunking_config()
        self.semantic_patterns = get_semantic_patterns()
        self.quality_weights = get_quality_weights()
    
    def extract_timestamps(self, transcription: str) -> List[Tuple[re.Match, float, int, int, int]]:
        """
        Extract timestamps from transcription using flexible patterns
        
        Returns:
            List of (match, start_time_seconds, hours, minutes, seconds)
        """
        valid_timestamps = []
        
        for pattern in self.config['TIMESTAMP_PATTERNS']:
            matches = list(re.finditer(pattern, transcription, re.IGNORECASE))
            logger.info(f"🔍 Pattern '{pattern}' found {len(matches)} matches")
            
            for match in matches:
                try:
                    if '[' in pattern:  # Bracket format
                        groups = match.groups()
                        if len(groups) == 3:  # [HH:MM:SS] or [MM:SS]
                            hh, mm, ss = groups
                            hours = int(hh) if hh else 0
                            minutes = int(mm)
                            seconds = int(ss) if ss else 0
                        elif len(groups) == 2:  # [MM:SS]
                            mm, ss = groups
                            hours = 0
                            minutes = int(mm)
                            seconds = int(ss)
                        elif len(groups) == 1:  # [SS]
                            hours = 0
                            minutes = 0
                            seconds = int(groups[0])
                    else:  # No bracket format
                        groups = match.groups()
                        if len(groups) == 3:  # HH:MM:SS or MM:SS
                            hh, mm, ss = groups
                            hours = int(hh) if hh else 0
                            minutes = int(mm)
                            seconds = int(ss) if ss else 0
                        elif len(groups) == 2:  # MM:SS
                            mm, ss = groups
                            hours = 0
                            minutes = int(mm)
                            seconds = int(ss)
                        elif len(groups) == 1:  # SS or 120s
                            hours = 0
                            minutes = 0
                            seconds = int(groups[0])
                    
                    # Convert to total seconds
                    start_time = hours * 3600 + minutes * 60 + seconds
                    
                    # Validate timestamp (more flexible than before)
                    if start_time <= self.config['MAX_VIDEO_DURATION']:
                        valid_timestamps.append((match, start_time, hours, minutes, seconds))
                        logger.debug(f"✅ Valid timestamp: {hours:02d}:{minutes:02d}:{seconds:02d} ({start_time}s)")
                    else:
                        logger.warning(f"⚠️ Skipping timestamp {hours:02d}:{minutes:02d}:{seconds:02d} - too large ({start_time}s)")
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Invalid timestamp format: {match.group(0)} - {e}")
                    continue
        
        # Sort by start time
        valid_timestamps.sort(key=lambda x: x[1])
        logger.info(f"🔧 Extracted {len(valid_timestamps)} valid timestamps")
        return valid_timestamps
    
    def create_timestamped_chunks(self, transcription: str, video_url: str, 
                                company_name: str, qudemo_id: str) -> List[Dict]:
        """
        Create timestamped chunks from transcription
        """
        chunks = []
        valid_timestamps = self.extract_timestamps(transcription)
        
        if valid_timestamps:
            logger.info(f"🔧 Creating timestamped chunks from {len(valid_timestamps)} timestamps")
            
            for i, (match, start_time, hours, minutes, seconds) in enumerate(valid_timestamps):
                # Get text for this timestamp
                text_start = match.end()
                text_end = valid_timestamps[i + 1][0].start() if i + 1 < len(valid_timestamps) else len(transcription)
                chunk_text = transcription[text_start:text_end].strip()
                
                # Clean and validate chunk text
                chunk_text = self._clean_chunk_text(chunk_text)
                
                if len(chunk_text) >= self.config['MIN_CONTENT_LENGTH']:
                    # Calculate end time
                    if i + 1 < len(valid_timestamps):
                        end_time = valid_timestamps[i + 1][1]
                    else:
                        # Estimate end time based on text length
                        estimated_duration = len(chunk_text) / self.config['CHARS_PER_SECOND']
                        end_time = start_time + min(estimated_duration, self.config['MAX_DURATION'])
                    
                    # Create chunk with essential metadata
                    chunk = self._create_chunk_metadata(
                        chunk_text, start_time, end_time, video_url, 
                        company_name, qudemo_id, i, len(valid_timestamps)
                    )
                    chunks.append(chunk)
                    logger.debug(f"🔧 Created timestamped chunk {i+1}: {start_time}s → {end_time}s")
                else:
                    logger.debug(f"🔧 Skipping chunk {i+1}: insufficient content ({len(chunk_text)} chars)")
        else:
            logger.info("🔧 No timestamps found, using fallback chunking")
            chunks = self._create_fallback_chunks(transcription, video_url, company_name, qudemo_id)
        
        return chunks
    
    def _create_fallback_chunks(self, transcription: str, video_url: str, 
                              company_name: str, qudemo_id: str) -> List[Dict]:
        """
        Create fallback chunks when no timestamps are available
        """
        chunks = []
        
        # Estimate video duration
        estimated_duration = max(60, len(transcription.split()) / self.config['WORDS_PER_SECOND'])
        chunk_duration = self.config['MAX_DURATION']
        num_chunks = max(4, min(60, int(estimated_duration / chunk_duration)))
        
        logger.info(f"🔧 Fallback: Estimated duration: {estimated_duration:.1f}s, creating {num_chunks} chunks")
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, estimated_duration)
            
            # Extract text for this time segment
            text_start = int((start_time / estimated_duration) * len(transcription))
            text_end = int((end_time / estimated_duration) * len(transcription))
            chunk_text = transcription[text_start:text_end].strip()
            
            # Clean and validate
            chunk_text = self._clean_chunk_text(chunk_text)
            
            if len(chunk_text) >= self.config['MIN_CONTENT_LENGTH']:
                chunk = self._create_chunk_metadata(
                    chunk_text, start_time, end_time, video_url,
                    company_name, qudemo_id, i, num_chunks
                )
                chunks.append(chunk)
                logger.debug(f"🔧 Created fallback chunk {i+1}: {start_time}s → {end_time}s")
        
        return chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean chunk text for better quality
        """
        # Remove timestamp patterns
        for pattern in self.config['TIMESTAMP_PATTERNS']:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = text.strip()
        
        return text
    
    def _create_chunk_metadata(self, text: str, start_time: float, end_time: float,
                             video_url: str, company_name: str, qudemo_id: str,
                             chunk_index: int, total_chunks: int) -> Dict:
        """
        Create standardized chunk metadata optimized for Q&A
        """
        # Calculate content quality score
        quality_score = self._calculate_content_quality(text)
        
        # Check if chunk contains complete sentences
        is_complete_sentence = self._is_complete_sentence(text)
        
        # Extract topic keywords
        topic_keywords = self._extract_topic_keywords(text)
        
        return {
            # Essential metadata for Q&A
            'text': text,
            'start_timestamp': float(start_time),
            'end_timestamp': float(end_time),
            'video_url': video_url,
            'chunk_type': 'video',
            'content_quality': quality_score,
            'word_count': len(text.split()),
            'is_complete_sentence': is_complete_sentence,
            
            # Additional useful metadata
            'topic_keywords': topic_keywords,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_duration': end_time - start_time,
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'processed_at': datetime.now().isoformat(),
            
            # Q&A optimization flags
            'has_question': '?' in text,
            'has_instruction': any(word in text.lower() for word in ['how to', 'step', 'first', 'then', 'next']),
            'has_explanation': any(word in text.lower() for word in ['because', 'since', 'therefore', 'so']),
        }
    
    def _calculate_content_quality(self, text: str) -> float:
        """
        Calculate content quality score for Q&A optimization
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        scores = {}
        
        # Word count score (normalized)
        word_count = len(text.split())
        scores['word_count'] = min(1.0, word_count / 50)  # Optimal around 50 words
        
        # Sentence completeness score
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        scores['sentence_completeness'] = min(1.0, complete_sentences / 3)  # Optimal around 3 sentences
        
        # Readability score (simple heuristic)
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        scores['readability'] = max(0, 1.0 - (avg_word_length - 5) / 10)  # Optimal around 5 chars/word
        
        # Topic relevance (keyword density)
        relevant_keywords = ['product', 'feature', 'benefit', 'solution', 'problem', 'customer', 'business']
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in text.lower())
        scores['topic_relevance'] = min(1.0, keyword_count / 3)  # Optimal around 3 relevant keywords
        
        # Calculate weighted score
        total_score = sum(
            scores.get(metric, 0) * weight 
            for metric, weight in self.quality_weights.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _is_complete_sentence(self, text: str) -> bool:
        """
        Check if chunk contains complete sentences
        """
        # Simple heuristic: ends with punctuation and has subject-verb structure
        ends_with_punctuation = text.strip().endswith(('.', '!', '?'))
        has_verb_indicators = any(word in text.lower() for word in ['is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'will', 'can', 'should'])
        
        return ends_with_punctuation and has_verb_indicators
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """
        Extract topic keywords from text
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words and get most frequent
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'a', 'an'}
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        
        # Return top 5 most frequent keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(5)]
    
    def detect_semantic_boundaries(self, chunks: List[Dict]) -> List[Dict]:
        """
        Detect semantic boundaries and adjust chunk boundaries accordingly
        """
        improved_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            
            # Check for topic transitions
            has_topic_transition = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in self.semantic_patterns['topic_transitions']
            )
            
            # Check for incomplete sentences at boundaries
            has_incomplete_sentence = not self._is_complete_sentence(text)
            
            # If chunk has issues, try to improve it
            if has_topic_transition or has_incomplete_sentence:
                improved_chunk = self._improve_chunk_boundaries(chunk, chunks, i)
                improved_chunks.append(improved_chunk)
            else:
                improved_chunks.append(chunk)
        
        return improved_chunks
    
    def _improve_chunk_boundaries(self, chunk: Dict, all_chunks: List[Dict], index: int) -> Dict:
        """
        Improve chunk boundaries for better semantic coherence
        """
        # For now, return the chunk as-is
        # This can be enhanced with more sophisticated boundary detection
        return chunk
