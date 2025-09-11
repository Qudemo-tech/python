#!/usr/bin/env python3
"""
Enhanced Hybrid Q&A System
Combines Pinecone semantic search with GCS direct transcript search
Handles multiple videos per QuDemo with correct video URL and timestamp associations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from enhanced_qa_semantic import EnhancedSemanticQA
from gcs_qa_service import GCSQAService

logger = logging.getLogger(__name__)

class EnhancedHybridQA:
    """Hybrid Q&A system that combines semantic search with direct transcript search"""
    
    def __init__(self):
        """Initialize hybrid Q&A system"""
        self.semantic_qa = EnhancedSemanticQA()
        self.gcs_qa = GCSQAService()
        
        # Quality thresholds for fallback logic
        self.SEMANTIC_QUALITY_THRESHOLD = 0.3
        self.GCS_QUALITY_THRESHOLD = 0.5
        
        logger.info("✅ Enhanced Hybrid Q&A system initialized")
    
    async def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        Answer a question using hybrid approach
        
        Args:
            question: The question to answer
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            Dict with answer, timestamp, and metadata
        """
        try:
            logger.info(f"🔀 Hybrid QA processing: {question}")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # Try semantic search first (better for complex questions)
            semantic_result = self.semantic_qa.ask_question(question, company_name, qudemo_id)
            
            # Try GCS direct search (better for specific video content)
            gcs_result = await self.gcs_qa.ask_question(question, company_name, qudemo_id)
            
            # Evaluate both results and choose the best one
            best_result = self._select_best_result(semantic_result, gcs_result, question)
            
            # Enhance the result with additional metadata
            enhanced_result = self._enhance_result(best_result, question, company_name, qudemo_id)
            
            logger.info(f"✅ Hybrid QA completed successfully")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Hybrid QA processing failed: {e}")
            return self._create_fallback_response(question)
    
    def _select_best_result(self, semantic_result: Dict, gcs_result: Dict, question: str) -> Dict:
        """Select the best result from semantic and GCS search"""
        try:
            # Check if either result failed
            if not semantic_result.get('success', False) and not gcs_result.get('success', False):
                logger.warning("⚠️ Both semantic and GCS search failed")
                return self._create_fallback_response(question)
            
            if not semantic_result.get('success', False):
                logger.info("📊 Using GCS result (semantic failed)")
                return gcs_result
            
            if not gcs_result.get('success', False):
                logger.info("📊 Using semantic result (GCS failed)")
                return semantic_result
            
            # Both succeeded - compare quality
            semantic_quality = self._calculate_result_quality(semantic_result, question)
            gcs_quality = self._calculate_result_quality(gcs_result, question)
            
            logger.info(f"📊 Quality comparison: Semantic={semantic_quality:.3f}, GCS={gcs_quality:.3f}")
            
            # Choose the higher quality result
            if semantic_quality >= gcs_quality:
                logger.info("📊 Using semantic result (higher quality)")
                return semantic_result
            else:
                logger.info("📊 Using GCS result (higher quality)")
                return gcs_result
                
        except Exception as e:
            logger.error(f"❌ Error selecting best result: {e}")
            return semantic_result if semantic_result.get('success', False) else gcs_result
    
    def _calculate_result_quality(self, result: Dict, question: str) -> float:
        """Calculate quality score for a result"""
        try:
            quality_score = 0.0
            
            # Base success score
            if result.get('success', False):
                quality_score += 0.3
            
            # Confidence score
            confidence = result.get('confidence', 0) or result.get('confidence_score', 0)
            quality_score += confidence * 0.3
            
            # Answer length score (prefer substantial answers)
            answer = result.get('answer', '')
            if 100 <= len(answer) <= 1000:
                quality_score += 0.2
            elif len(answer) > 1000:
                quality_score += 0.1
            
            # Video URL presence (for video questions)
            if any(keyword in question.lower() for keyword in ['video', 'show', 'watch', 'play']):
                if result.get('video_url'):
                    quality_score += 0.2
            
            # Timestamp presence (for video questions)
            if result.get('start', 0) > 0 and result.get('end', 0) > 0:
                quality_score += 0.1
            
            # Sources presence
            if result.get('sources') and len(result['sources']) > 0:
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating result quality: {e}")
            return 0.0
    
    def _enhance_result(self, result: Dict, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Enhance the result with additional metadata and formatting"""
        try:
            # Ensure video URL is correct for multiple videos
            if result.get('video_url'):
                result['video_url'] = self._validate_video_url(result['video_url'], company_name, qudemo_id)
            
            # Add hybrid metadata
            result['hybrid_qa'] = True
            result['processing_method'] = 'hybrid_semantic_gcs'
            
            # Enhance answer formatting if needed
            if result.get('answer') and len(result['answer']) < 50:
                result['answer'] = self._enhance_short_answer(result['answer'], question)
            
            # Ensure proper timestamp formatting
            if result.get('start', 0) > 0 and result.get('end', 0) > 0:
                result['formatted_timestamp'] = self._format_timestamp(result['start'], result['end'])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error enhancing result: {e}")
            return result
    
    def _validate_video_url(self, video_url: str, company_name: str, qudemo_id: str) -> str:
        """Validate and correct video URL for multiple videos"""
        try:
            if not video_url or not video_url.strip():
                return ''
            
            # Basic validation - ensure it's a valid URL
            if not video_url.startswith(('http://', 'https://')):
                logger.warning(f"⚠️ Invalid video URL format: {video_url}")
                return ''
            
            # For now, return the URL as-is
            # In the future, we could add more sophisticated validation
            return video_url.strip()
            
        except Exception as e:
            logger.error(f"❌ Error validating video URL: {e}")
            return video_url
    
    def _enhance_short_answer(self, answer: str, question: str) -> str:
        """Enhance short answers with more context"""
        try:
            if len(answer) < 50:
                # Add context based on question type
                if 'how to' in question.lower():
                    return f"Here's how to {question.replace('how to ', '').replace('?', '')}:\n\n{answer}"
                elif 'what is' in question.lower():
                    return f"Here's what {question.replace('what is ', '').replace('?', '')} is:\n\n{answer}"
                else:
                    return f"Here's the answer to your question:\n\n{answer}"
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error enhancing short answer: {e}")
            return answer
    
    def _format_timestamp(self, start: float, end: float) -> str:
        """Format timestamp for display"""
        try:
            if start == 0 and end == 0:
                return ""
            
            start_str = self._seconds_to_timestamp(start)
            end_str = self._seconds_to_timestamp(end)
            
            return f"{start_str} - {end_str}"
            
        except Exception as e:
            logger.error(f"❌ Error formatting timestamp: {e}")
            return ""
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS or HH:MM:SS format"""
        try:
            if seconds < 60:
                return f"{int(seconds):02d}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}:{secs:02d}"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours}:{minutes:02d}:{secs:02d}"
                
        except Exception as e:
            logger.error(f"❌ Error converting seconds to timestamp: {e}")
            return "00:00"
    
    def _create_fallback_response(self, question: str) -> Dict:
        """Create a fallback response when all methods fail"""
        return {
            'success': False,
            'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
            'start': 0,
            'end': 0,
            'video_url': None,
            'sources': [],
            'total_sources': 0,
            'search_score': 0,
            'content_types_found': [],
            'difficulty_level': 'unknown',
            'estimated_time': 'unknown',
            'confidence_score': 0.0,
            'fallback_reason': 'all_search_methods_failed',
            'hybrid_qa': True,
            'processing_method': 'hybrid_fallback'
        }


# Global instance for singleton pattern
_enhanced_hybrid_qa_instance = None

def initialize_enhanced_hybrid_qa():
    """Initialize the enhanced hybrid QA system"""
    global _enhanced_hybrid_qa_instance
    try:
        _enhanced_hybrid_qa_instance = EnhancedHybridQA()
        logger.info("✅ Enhanced Hybrid QA system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Hybrid QA: {e}")
        return False

def get_enhanced_hybrid_qa():
    """Get the enhanced hybrid QA system instance"""
    global _enhanced_hybrid_qa_instance
    if _enhanced_hybrid_qa_instance is None:
        logger.warning("⚠️ Enhanced Hybrid QA not initialized, initializing now...")
        initialize_enhanced_hybrid_qa()
    return _enhanced_hybrid_qa_instance
