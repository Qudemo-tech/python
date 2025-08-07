from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict
import json
import os
import io
import re
import uuid
from dotenv import load_dotenv
import logging
import time
import random
import requests
from collections import defaultdict
from fastapi import HTTPException
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import urllib.request
import requests
import tempfile
import psutil
import sys

# Import our processors
from gemini_transcription import GeminiTranscriptionProcessor
from loom_processor import LoomVideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"💾 Memory usage: {memory_mb:.1f}MB")
    return memory_mb

load_dotenv()

# Global video URL mapping
VIDEO_URL_MAPPING = {}

# Initialize processors
gemini_processor = None
loom_processor = None

def initialize_processors():
    """Initialize Gemini and Loom processors"""
    global gemini_processor, loom_processor
    
    try:
        # Get API keys
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([gemini_api_key, pinecone_api_key, openai_api_key]):
            logger.error("❌ Missing required API keys for processors")
            return False
        
        # Initialize Gemini processor
        gemini_processor = GeminiTranscriptionProcessor(gemini_api_key, pinecone_api_key, openai_api_key)
        logger.info("✅ Gemini processor initialized")
        
        # Initialize Loom processor
        loom_processor = LoomVideoProcessor(openai_api_key, pinecone_api_key)
        logger.info("✅ Loom processor initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize processors: {e}")
        return False

def fetch_video_urls_from_supabase():
    """Fetch video URL mappings from Supabase"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not found")
            return {}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        response = supabase.table('videos').select('video_name, video_url').execute()
        
        mappings = {}
        for row in response.data:
            mappings[row['video_name']] = row['video_url']
        
        logger.info(f"📝 Fetched {len(mappings)} video mappings from Supabase")
        return mappings
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch video mappings: {e}")
        return {}

def initialize_existing_mappings():
    """Initialize existing video URL mappings"""
    global VIDEO_URL_MAPPING
    VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
    logger.info(f"📝 Initialized {len(VIDEO_URL_MAPPING)} video mappings")

def answer_question(company_name, question):
    """Answer questions using stored vectors"""
    try:
        if not gemini_processor and not loom_processor:
            raise Exception("No processors available")
        
        # Use either processor for search (they both have the same interface)
        processor = gemini_processor or loom_processor
        
        # Search for similar chunks
        search_results = processor.search_similar_chunks(company_name, question, top_k=6)
        
        if not search_results:
            return {
                'answer': "I don't have enough information to answer that question. Please make sure videos have been processed for this company.",
                'sources': []
            }
        
        # Format context from search results
        context_parts = []
        sources = []
        
        for result in search_results:
            if result['score'] > 0.3:  # Only use relevant results
                context_parts.append(result['text'])
                sources.append({
                    'text': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                    'video_url': result['video_url'],
                    'title': result['title'],
                    'score': result['score']
                })
        
        if not context_parts:
            return {
                'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ensure videos have been processed.",
                'sources': []
            }
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        # Create answer using OpenAI
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            prompt = f"""Based on the following context from video transcriptions, answer the question. 
            If the context doesn't contain enough information to answer the question, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on video transcriptions. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            # Fallback: return context-based answer
            return {
                'answer': f"Based on the available information: {context[:300]}...",
                'sources': sources
            }
        
    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        return {
            'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
            'sources': []
        }

def process_video(video_url, company_name, bucket_name, source=None, meeting_link=None):
    """Main function to process a video for a company"""
    try:
        logger.info(f"🎯 Processing video: {video_url}")
        
        # Check if it's a YouTube URL
        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            logger.info("📺 YouTube video detected, using Gemini transcription")
            
            if gemini_processor:
                result = gemini_processor.process_video(video_url, company_name)
                if result and result.get('success'):
                    logger.info(f"✅ YouTube video processed successfully with Gemini")
                    return {
                        'success': True,
                        'method': 'gemini_transcription',
                        'title': result.get('title', 'Unknown'),
                        'chunks_created': result.get('chunks_created', 0),
                        'vectors_stored': result.get('vectors_stored', 0),
                        'word_count': result.get('word_count', 'Unknown')
                    }
                else:
                    logger.error("❌ Gemini transcription failed")
                    return {'error': 'Gemini transcription failed'}
            else:
                logger.error("❌ Gemini processor not available")
                return {'error': 'Gemini processor not available'}
        
        # For Loom videos, use Loom processor
        elif 'loom.com' in video_url:
            logger.info("🎬 Loom video detected, using Loom processor")
            
            if loom_processor:
                result = loom_processor.process_video(video_url, company_name)
                if result and result.get('success'):
                    logger.info(f"✅ Loom video processed successfully")
                    return {
                        'success': True,
                        'method': 'loom_transcription',
                        'title': result.get('title', 'Unknown'),
                        'chunks_created': result.get('chunks_created', 0),
                        'vectors_stored': result.get('vectors_stored', 0),
                        'word_count': result.get('word_count', 'Unknown')
                    }
                else:
                    logger.error("❌ Loom transcription failed")
                    return {'error': 'Loom transcription failed'}
            else:
                logger.error("❌ Loom processor not available")
                return {'error': 'Loom processor not available'}
        
        else:
            logger.warning(f"⚠️ Unknown video platform: {video_url}")
            return {'error': 'Unsupported video platform'}
            
    except Exception as e:
        logger.error(f"❌ Video processing error: {e}")
        return {'error': f'Processing failed: {str(e)}'}

# FastAPI app setup
app = FastAPI(title="QuDemo Video Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting for video processing
last_request_time = defaultdict(float)
MIN_REQUEST_INTERVAL = 5  # Minimum 5 seconds between requests per company

# Pydantic models
class ProcessVideoRequest(BaseModel, extra='allow'):
    video_url: str
    company_name: str
    bucket_name: Optional[str] = None
    source: Optional[str] = None
    meeting_link: Optional[str] = None
    is_loom: bool = True

class AskQuestionRequest(BaseModel):
    question: str
    company_name: str

class AskQuestionCompanyRequest(BaseModel):
    question: str

class GenerateSummaryRequest(BaseModel):
    questions_and_answers: List[Dict[str, str]]
    buyer_name: Optional[str] = None
    company_name: Optional[str] = None

# API endpoints
@app.post("/process-video/{company_name}")
async def process_video_endpoint(company_name: str, request: Request):
    """Process a video for a specific company"""
    try:
        # Rate limiting check
        current_time = time.time()
        time_since_last_request = current_time - last_request_time[company_name]
        
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request
            logger.warning(f"⚠️ Rate limiting: {company_name} made request too quickly. Waiting {wait_time:.1f} seconds...")
            raise HTTPException(
                status_code=429, 
                detail=f"Too many requests. Please wait {wait_time:.1f} seconds before trying again."
            )
        
        last_request_time[company_name] = current_time
        
        # Parse request body
        body = await request.json()
        
        # Validate required fields
        if 'video_url' not in body:
            raise HTTPException(status_code=400, detail="video_url is required")
        
        video_url = body['video_url']
        bucket_name = body.get('bucket_name', company_name.lower().replace(' ', '-'))
        source = body.get('source')
        meeting_link = body.get('meeting_link')
        
        logger.info(f"🎬 Processing video for {company_name}: {video_url}")
        
        # Process the video
        result = process_video(video_url, company_name, bucket_name, source, meeting_link)
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"Video processed successfully using {result.get('method', 'unknown')} method",
                'data': result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown processing error'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video processing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question_endpoint(request: AskQuestionRequest):
    """Ask a question about processed videos"""
    try:
        logger.info(f"❓ Question for {request.company_name}: {request.question}")
        
        result = answer_question(request.company_name, request.question)
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['sources']
        }
        
    except Exception as e:
        logger.error(f"❌ Question endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/{company_name}")
async def ask_question_company_endpoint(company_name: str, request: AskQuestionCompanyRequest):
    """Ask a question for a specific company"""
    try:
        logger.info(f"❓ Question for {company_name}: {request.question}")
        
        result = answer_question(company_name, request.question)
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['sources']
        }
        
    except Exception as e:
        logger.error(f"❌ Company question endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary")
async def generate_summary_endpoint(request: GenerateSummaryRequest):
    """Generate a summary from questions and answers"""
    try:
        logger.info(f"📝 Generating summary for {request.company_name}")
        
        # Format Q&A for summary
        qa_text = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in request.questions_and_answers
        ])
        
        # Generate summary using OpenAI
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            prompt = f"""Based on the following questions and answers from a video demo session, create a concise summary.
            
            Questions and Answers:
            {qa_text}
            
            Company: {request.company_name or 'Unknown'}
            Buyer: {request.buyer_name or 'Unknown'}
            
            Summary:"""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of demo sessions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'summary': summary,
                'company_name': request.company_name,
                'buyer_name': request.buyer_name,
                'qa_count': len(request.questions_and_answers)
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI summary generation failed: {e}")
            return {
                'success': False,
                'error': f"Summary generation failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"❌ Summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/qa-test/{company_name}")
async def debug_qa_test(company_name: str, question: str = "What is this video about?"):
    """Debug endpoint to test Q&A functionality"""
    try:
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

@app.get("/debug/videos")
async def debug_videos():
    """Debug endpoint to show video mappings"""
    try:
        return {
            'success': True,
            'video_mappings': VIDEO_URL_MAPPING,
            'count': len(VIDEO_URL_MAPPING)
        }
    except Exception as e:
        logger.error(f"❌ Debug videos failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/status")
async def status_check():
    """Get system status"""
    try:
        # Check processors
        gemini_status = "✅ Available" if gemini_processor else "❌ Not available"
        loom_status = "✅ Available" if loom_processor else "❌ Not available"
        
        # Check API keys
        gemini_key = "✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Missing"
        pinecone_key = "✅ Set" if os.getenv("PINECONE_API_KEY") else "❌ Missing"
        openai_key = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
        
        # Get memory usage
        memory_mb = log_memory_usage()
        
        # Get video mappings count
        video_count = len(VIDEO_URL_MAPPING)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "processors": {
                "gemini": gemini_status,
                "loom": loom_status
            },
            "api_keys": {
                "gemini": gemini_key,
                "pinecone": pinecone_key,
                "openai": openai_key
            },
            "memory_mb": memory_mb,
            "video_mappings": video_count,
            "python_version": sys.version,
            "environment": "production" if os.getenv("RENDER") else "development"
        }
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory-status")
async def memory_status():
    """Get current memory usage status"""
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def add_video_url_mapping(local_filename, original_url):
    """Add video URL mapping"""
    global VIDEO_URL_MAPPING
    filename = os.path.basename(local_filename)
    VIDEO_URL_MAPPING[filename] = original_url
    logger.info(f"📝 Added video mapping: {filename} -> {original_url}")
    
    # Also upsert to Supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # For Loom videos, store the original share URL as the video_url for playback
            video_url_for_playback = original_url
            
            # Upsert by video_name
            supabase.table('videos').upsert({
                'video_name': filename,
                'video_url': video_url_for_playback
            }, on_conflict=['video_name']).execute()
            logger.info(f"📝 Upserted video mapping to Supabase: {filename} -> {video_url_for_playback}")
    except Exception as e:
        logger.error(f"❌ Failed to upsert video mapping to Supabase: {e}")

def get_original_video_url(local_filename):
    """Get original video URL from local filename"""
    global VIDEO_URL_MAPPING
    if '[' in local_filename:
        filename = local_filename.split('[')[0].strip()
    else:
        filename = local_filename

    original_url = VIDEO_URL_MAPPING.get(filename)
    if not original_url:
        # Try to refresh mapping from Supabase
        logger.info(f"🔄 Refreshing video mappings from Supabase for: {filename}")
        VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
        original_url = VIDEO_URL_MAPPING.get(filename)
    if original_url:
        logger.info(f"🔗 Found video mapping: {filename} -> {original_url}")
    else:
        logger.warning(f"⚠️ No video mapping found for: {filename}")
    return original_url

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize processors and mappings on startup"""
    logger.info("🚀 Starting QuDemo Video Processing API...")
    
    # Initialize processors
    if initialize_processors():
        logger.info("✅ All processors initialized successfully")
    else:
        logger.error("❌ Failed to initialize some processors")
    
    # Initialize video mappings
    initialize_existing_mappings()
    
    logger.info("🎉 API startup complete!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port) 