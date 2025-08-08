"""
Health Checks Module
Handles system status, memory monitoring, and health check endpoints
"""

import os
import sys
import logging
import psutil
from datetime import datetime
from typing import Dict

# Import from other modules
from video_processing import get_processors_status, get_video_mappings

# Configure logging
logger = logging.getLogger(__name__)

def log_memory_usage() -> float:
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"💾 Memory usage: {memory_mb:.1f} MB")
        return memory_mb
    except Exception as e:
        logger.error(f"❌ Failed to get memory usage: {e}")
        return 0.0

def get_system_status() -> Dict:
    """Get comprehensive system status"""
    try:
        # Check processors
        processors_status = get_processors_status()
        
        # Check API keys
        gemini_key = "✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Missing"
        pinecone_key = "✅ Set" if os.getenv("PINECONE_API_KEY") else "❌ Missing"
        openai_key = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
        supabase_url = "✅ Set" if os.getenv("SUPABASE_URL") else "❌ Missing"
        supabase_key = "✅ Set" if os.getenv("SUPABASE_ANON_KEY") else "❌ Missing"
        
        # Get memory usage
        memory_mb = log_memory_usage()
        
        # Get video mappings count
        video_mappings = get_video_mappings()
        video_count = len(video_mappings)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "processors": processors_status,
            "api_keys": {
                "gemini": gemini_key,
                "pinecone": pinecone_key,
                "openai": openai_key,
                "supabase_url": supabase_url,
                "supabase_key": supabase_key
            },
            "memory_mb": memory_mb,
            "video_mappings": video_count,
            "python_version": sys.version,
            "environment": "production" if os.getenv("RENDER") else "development"
        }
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_memory_status() -> Dict:
    """Get detailed memory status with thresholds"""
    try:
        memory_mb = log_memory_usage()
        
        # Get memory thresholds
        memory_threshold = 1900
        cleanup_threshold = 1400
        
        status = {
            "memory_mb": memory_mb,
            "memory_threshold_mb": memory_threshold,
            "cleanup_threshold_mb": cleanup_threshold,
            "status": "ok" if memory_mb < cleanup_threshold else "warning" if memory_mb < memory_threshold else "critical",
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Memory status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_health_check() -> Dict:
    """Basic health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    }

def get_debug_videos() -> Dict:
    """Debug endpoint to show video mappings"""
    try:
        video_mappings = get_video_mappings()
        return {
            'success': True,
            'video_mappings': video_mappings,
            'count': len(video_mappings)
        }
    except Exception as e:
        logger.error(f"❌ Debug videos failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_debug_qa_test(company_name: str, question: str = "What is this video about?") -> Dict:
    """Debug endpoint to test Q&A functionality"""
    try:
        from qa_system import answer_question
        
        logger.info(f"🧪 Testing Q&A for {company_name}: {question}")
        
        result = answer_question(company_name, question)
        
        return {
            'success': True,
            'company_name': company_name,
            'question': question,
            'answer': result['answer'],
            'sources': result['sources']
        }
        
    except Exception as e:
        logger.error(f"❌ Debug Q&A test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_debug_chunks(company_name: str, question: str = "spotlight feature") -> Dict:
    """Debug endpoint to inspect raw Pinecone chunks"""
    try:
        from qa_system import query_pinecone
        import openai
        
        logger.info(f"🔍 Debugging chunks for {company_name}: {question}")
        
        # Create embedding for the question
        q_embedding = openai.embeddings.create(
            input=[question],
            model="text-embedding-3-small",
            timeout=15
        ).data[0].embedding
        
        # Query Pinecone
        matches = query_pinecone(company_name, q_embedding, top_k=6)
        top_chunks = [m["metadata"] for m in matches]
        
        # Return raw chunk data for inspection
        return {
            'success': True,
            'company_name': company_name,
            'question': question,
            'chunks_count': len(top_chunks),
            'chunks': top_chunks,
            'raw_matches': [{"id": m["id"], "score": m["score"], "metadata_keys": list(m["metadata"].keys())} for m in matches]
        }
        
    except Exception as e:
        logger.error(f"❌ Debug chunks failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }
