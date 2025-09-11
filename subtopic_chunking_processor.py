#!/usr/bin/env python3
"""
Subtopic Chunking Processor
Breaks down large topics into smaller, more specific subtopic chunks
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import re
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SubtopicChunkingProcessor:
    """Processor for breaking down large topics into subtopic-level chunks"""
    
    def __init__(self):
        """Initialize subtopic chunking processor"""
        try:
            # Configure OpenAI
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            # Configuration
            self.config = {
                'min_subtopic_duration': 30,  # Minimum 30 seconds for a subtopic
                'max_subtopic_duration': 300,  # Maximum 5 minutes for a subtopic
                'min_subtopic_text_length': 100,  # Minimum 100 characters
                'max_subtopic_text_length': 2000,  # Maximum 2000 characters
                'subtopic_overlap': 50,  # 50 character overlap between subtopics
                'confidence_threshold': 0.7  # Minimum confidence for subtopic detection
            }
            
            logger.info("✅ Subtopic Chunking Processor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Subtopic Chunking Processor: {e}")
            raise
    
    async def detect_subtopics(self, segment: Dict, transcription_text: str) -> List[Dict]:
        """
        Detect subtopics within a large segment using LLM analysis
        
        Args:
            segment: The main segment containing the large topic
            transcription_text: Full transcription text for context
            
        Returns:
            List of subtopic segments with timestamps and metadata
        """
        try:
            logger.info(f"🔍 Detecting subtopics in segment: {segment.get('title', 'Unknown')}")
            
            # Extract the text for this segment
            segment_text = segment.get('text', '')
            if not segment_text or len(segment_text) < self.config['min_subtopic_text_length']:
                logger.info("⚠️ Segment too short for subtopic detection")
                return [segment]  # Return original segment if too short
            
            # Use LLM to detect subtopics
            subtopics = await self._analyze_subtopics_with_llm(segment, transcription_text)
            
            if not subtopics:
                logger.info("⚠️ No subtopics detected, returning original segment")
                return [segment]
            
            # Convert subtopics to segment format
            subtopic_segments = []
            for i, subtopic in enumerate(subtopics):
                subtopic_segment = {
                    'text': subtopic['text'],
                    'start_time': subtopic['start_time'],
                    'end_time': subtopic['end_time'],
                    'title': subtopic['title'],
                    'summary': subtopic['summary'],
                    'keywords': subtopic['keywords'],
                    'subtopic_of': segment.get('title', 'Unknown'),
                    'parent_segment_id': segment.get('id', f"segment_{i}"),
                    'subtopic_id': f"subtopic_{i}",
                    'confidence': subtopic['confidence'],
                    'refined': True
                }
                subtopic_segments.append(subtopic_segment)
            
            logger.info(f"✅ Detected {len(subtopic_segments)} subtopics")
            return subtopic_segments
            
        except Exception as e:
            logger.error(f"❌ Error detecting subtopics: {e}")
            return [segment]  # Return original segment on error
    
    async def _analyze_subtopics_with_llm(self, segment: Dict, transcription_text: str) -> List[Dict]:
        """
        Use LLM to analyze and detect subtopics within a segment
        
        Args:
            segment: The main segment
            transcription_text: Full transcription for context
            
        Returns:
            List of detected subtopics with metadata
        """
        try:
            segment_text = segment.get('text', '')
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', start_time + 60)
            
            # Create a focused prompt for subtopic detection
            prompt = f"""
            Analyze this video segment and identify distinct subtopics within it. Each subtopic should be a complete, self-contained explanation of a specific aspect.

            SEGMENT TEXT:
            {segment_text}

            SEGMENT TIMESTAMP: {start_time:.1f}s - {end_time:.1f}s

            Please identify 2-5 distinct subtopics within this segment. For each subtopic, provide:

            1. A clear, descriptive title (3-8 words)
            2. The specific text content for that subtopic
            3. Estimated start and end times within the segment
            4. A brief summary (1-2 sentences)
            5. Key keywords/topics covered
            6. Confidence level (0.0-1.0)

            Focus on:
            - Step-by-step processes
            - Different features or components
            - Specific examples or demonstrations
            - Different aspects of the same topic
            - Before/after scenarios
            - Different tools or methods

            Return as JSON array with this structure:
            [
                {{
                    "title": "Subtopic Title",
                    "text": "Specific text content for this subtopic...",
                    "start_time": 123.4,
                    "end_time": 156.7,
                    "summary": "Brief summary of this subtopic",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "confidence": 0.85
                }}
            ]

            IMPORTANT: Only return valid JSON. No additional text or explanations.
            """
            
            # Call OpenAI API (v1.0+ compatible)
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing video content and identifying distinct subtopics within larger segments. You provide accurate, structured analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response text
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            try:
                subtopics = json.loads(response_text)
                
                # Validate and clean subtopics
                validated_subtopics = []
                for subtopic in subtopics:
                    if self._validate_subtopic(subtopic, start_time, end_time):
                        validated_subtopics.append(subtopic)
                
                logger.info(f"✅ LLM detected {len(validated_subtopics)} valid subtopics")
                return validated_subtopics
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response: {response_text[:500]}...")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error in LLM subtopic analysis: {e}")
            return []
    
    def _validate_subtopic(self, subtopic: Dict, segment_start: float, segment_end: float) -> bool:
        """
        Validate a detected subtopic
        
        Args:
            subtopic: The subtopic to validate
            segment_start: Start time of parent segment
            segment_end: End time of parent segment
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['title', 'text', 'start_time', 'end_time', 'summary', 'keywords', 'confidence']
            for field in required_fields:
                if field not in subtopic:
                    logger.warning(f"⚠️ Subtopic missing required field: {field}")
                    return False
            
            # Validate text length
            text = subtopic.get('text', '')
            if len(text) < self.config['min_subtopic_text_length']:
                logger.warning(f"⚠️ Subtopic text too short: {len(text)} chars")
                return False
            
            if len(text) > self.config['max_subtopic_text_length']:
                logger.warning(f"⚠️ Subtopic text too long: {len(text)} chars")
                return False
            
            # Validate timestamps
            start_time = float(subtopic.get('start_time', 0))
            end_time = float(subtopic.get('end_time', 0))
            
            if start_time < segment_start or end_time > segment_end:
                logger.warning(f"⚠️ Subtopic timestamps outside segment range: {start_time}-{end_time}")
                return False
            
            if end_time <= start_time:
                logger.warning(f"⚠️ Invalid subtopic timestamps: {start_time}-{end_time}")
                return False
            
            # Validate duration
            duration = end_time - start_time
            if duration < self.config['min_subtopic_duration']:
                logger.warning(f"⚠️ Subtopic too short: {duration}s")
                return False
            
            if duration > self.config['max_subtopic_duration']:
                logger.warning(f"⚠️ Subtopic too long: {duration}s")
                return False
            
            # Validate confidence
            confidence = float(subtopic.get('confidence', 0))
            if confidence < self.config['confidence_threshold']:
                logger.warning(f"⚠️ Subtopic confidence too low: {confidence}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating subtopic: {e}")
            return False
    
    def should_break_into_subtopics(self, segment: Dict) -> bool:
        """
        Determine if a segment should be broken into subtopics
        
        Args:
            segment: The segment to analyze
            
        Returns:
            True if should be broken into subtopics, False otherwise
        """
        try:
            text = segment.get('text', '')
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', start_time + 60)
            duration = end_time - start_time
            
            # Check duration threshold (more than 3 minutes)
            if duration > 180:  # 3 minutes
                logger.info(f"📏 Segment duration {duration:.1f}s exceeds threshold, considering subtopics")
                return True
            
            # Check text length threshold (more than 1000 characters)
            if len(text) > 1000:
                logger.info(f"📝 Segment text length {len(text)} exceeds threshold, considering subtopics")
                return True
            
            # Check for step indicators
            step_indicators = [
                'step', 'first', 'second', 'third', 'next', 'then', 'now', 'after',
                'before', 'finally', 'lastly', 'additionally', 'furthermore',
                '1.', '2.', '3.', '4.', '5.'
            ]
            
            text_lower = text.lower()
            step_count = sum(1 for indicator in step_indicators if indicator in text_lower)
            
            if step_count >= 3:
                logger.info(f"🔢 Segment has {step_count} step indicators, considering subtopics")
                return True
            
            # Check for topic transition words
            transition_words = [
                'now let\'s', 'next we\'ll', 'another', 'also', 'besides',
                'in addition', 'furthermore', 'moreover', 'similarly',
                'on the other hand', 'however', 'meanwhile'
            ]
            
            transition_count = sum(1 for word in transition_words if word in text_lower)
            
            if transition_count >= 2:
                logger.info(f"🔄 Segment has {transition_count} transition words, considering subtopics")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error analyzing segment for subtopic breaking: {e}")
            return False
    
    async def process_segments_for_subtopics(self, segments: List[Dict], transcription_text: str) -> List[Dict]:
        """
        Process all segments and break large ones into subtopics
        
        Args:
            segments: List of main segments
            transcription_text: Full transcription for context
            
        Returns:
            List of processed segments (original + subtopics)
        """
        try:
            logger.info(f"🔍 Processing {len(segments)} segments for subtopic detection")
            
            processed_segments = []
            
            for i, segment in enumerate(segments):
                logger.info(f"📊 Processing segment {i+1}/{len(segments)}: {segment.get('title', 'Unknown')}")
                
                # Check if segment should be broken into subtopics
                if self.should_break_into_subtopics(segment):
                    logger.info(f"✂️ Breaking segment into subtopics: {segment.get('title', 'Unknown')}")
                    
                    # Detect subtopics
                    subtopic_segments = await self.detect_subtopics(segment, transcription_text)
                    
                    if len(subtopic_segments) > 1:
                        logger.info(f"✅ Created {len(subtopic_segments)} subtopics")
                        processed_segments.extend(subtopic_segments)
                    else:
                        logger.info("⚠️ No subtopics created, keeping original segment")
                        processed_segments.append(segment)
                else:
                    logger.info("✅ Segment is appropriate size, keeping as-is")
                    processed_segments.append(segment)
            
            logger.info(f"🎉 Processed {len(segments)} segments into {len(processed_segments)} total segments")
            return processed_segments
            
        except Exception as e:
            logger.error(f"❌ Error processing segments for subtopics: {e}")
            return segments  # Return original segments on error

# Global instance
_subtopic_processor = None

def get_subtopic_processor() -> SubtopicChunkingProcessor:
    """Get global subtopic processor instance"""
    global _subtopic_processor
    if _subtopic_processor is None:
        _subtopic_processor = SubtopicChunkingProcessor()
    return _subtopic_processor
