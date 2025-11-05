#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Pinecone Standard Plan Multi-Index Architecture
Optimized for Q&A, video processing, and web scraping
"""

import os
import logging
import json
import asyncio
from typing import List, Optional, Dict
from datetime import datetime
from openai import OpenAI
import requests

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tempfile
import shutil

# Enhanced components
# from final_gemini_scraper import FinalGeminiScraper  # Removed - requires Playwright vds
from gcs_qa_service import GCSQAService
from google_cloud_storage_service import GoogleCloudStorageService
from simple_gemini_transcriber import SimpleGeminiTranscriber
from document_processor import DocumentProcessor
from company_api import router as company_router
from company_bucket_service import initialize_company_bucket_service, get_company_bucket_service
from website_scraper import WebsiteScraper
from heygen_service import HeyGenService
from avatar_video_processor import AvatarVideoProcessor
# from enhanced_scraper_with_failure_handling import initialize_enhanced_scraper, get_enhanced_scraper

# New universal scraper system
# from universal_help_scraper import UniversalScraperIntegration

# Video processing imports
from loom_processor import LoomVideoProcessorGCS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
loom_processor_gcs = None
gcs_qa_service = None
simple_transcriber = None
document_processor = None
company_bucket_service = None
website_scraper = None
heygen_service = None
avatar_video_processor = None
openai_client = None
supabase = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    global loom_processor_gcs, gcs_qa_service, simple_transcriber, document_processor, company_bucket_service, website_scraper, heygen_service, avatar_video_processor, openai_client, supabase
    
    try:
        logger.info("🚀 Starting Enhanced QuDemo Python Backend (GCS-based)...")
        
        
        # Initialize GCS Q&A Service (NEW - Google Cloud Storage based)
        try:
            # Check for GCS credentials in multiple locations
            service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            google_app_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            # Check local file
            local_service_account = 'service-account-key.json'
            # Check Render secret files directory
            render_secret_path = '/etc/secrets/service-account-key.json'
            
            credentials_found = False
            credentials_path = None
            
            if service_account_json:
                logger.info(f"🔍 GCS credentials found in GOOGLE_SERVICE_ACCOUNT_JSON environment variable")
                credentials_found = True
            elif google_app_creds and os.path.exists(google_app_creds):
                logger.info(f"🔍 GCS credentials found at GOOGLE_APPLICATION_CREDENTIALS: {google_app_creds}")
                credentials_found = True
                credentials_path = google_app_creds
            elif os.path.exists(render_secret_path):
                logger.info(f"🔍 GCS credentials found in Render secret files: {render_secret_path}")
                credentials_found = True
                credentials_path = render_secret_path
                # Set environment variable for other services
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = render_secret_path
            elif os.path.exists(local_service_account):
                logger.info(f"🔍 GCS credentials found in local file: {local_service_account}")
                credentials_found = True
                credentials_path = local_service_account
                # Set environment variable for other services
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_service_account
            
            if credentials_found:
                logger.info(f"🔍 GCS credentials found - initializing service")
                gcs_qa_service = GCSQAService()
                logger.info("✅ GCS Q&A Service initialized (Google Cloud Storage based)")
            else:
                logger.warning("⚠️ GCS credentials not found, GCS Q&A Service disabled")
                logger.warning("⚠️ Checked locations:")
                logger.warning(f"  - GOOGLE_SERVICE_ACCOUNT_JSON env var: {'✓' if service_account_json else '✗'}")
                logger.warning(f"  - GOOGLE_APPLICATION_CREDENTIALS: {google_app_creds or 'Not set'}")
                logger.warning(f"  - Render secret file: {render_secret_path} ({'✓' if os.path.exists(render_secret_path) else '✗'})")
                logger.warning(f"  - Local file: {local_service_account} ({'✓' if os.path.exists(local_service_account) else '✗'})")
                gcs_qa_service = None
        except Exception as e:
            logger.error(f"❌ GCS Q&A Service initialization error: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            logger.error("❌ GCS Q&A Service will be disabled - some features may not work")
            gcs_qa_service = None
        
        # Initialize Simple Gemini Transcriber (NEW - Simple GCS-based transcription)
        try:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                simple_transcriber = SimpleGeminiTranscriber(
                    api_key=gemini_api_key,
                    gcs_bucket_name='qudemo-video-transcripts'
                )
                logger.info("✅ Simple Gemini Transcriber initialized (GCS-based)")
            else:
                logger.warning("⚠️ GEMINI_API_KEY not found, Simple Transcriber not initialized")
                simple_transcriber = None
        except Exception as e:
            logger.error(f"❌ Simple Gemini Transcriber initialization error: {e}")
            simple_transcriber = None
        
        # Initialize Document Processor
        try:
            document_processor = DocumentProcessor()
            logger.info("✅ Document Processor initialized")
            
            # Initialize Website Scraper
            website_scraper = WebsiteScraper()
            logger.info("✅ Website Scraper initialized")
        except Exception as e:
            logger.error(f"❌ Document Processor initialization error: {e}")
            document_processor = None
        
        # Initialize HeyGen Service for AI Avatar Videos
        try:
            heygen_service = HeyGenService()
            logger.info("✅ HeyGen Service initialized (AI Avatar Videos)")
            
            # Initialize Avatar Video Processor
            if heygen_service and gcs_qa_service:
                # Get Supabase client for database updates
                try:
                    from supabase import create_client
                    supabase_url = os.getenv('SUPABASE_URL')
                    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
                    supabase_client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None
                except:
                    supabase_client = None
                
                avatar_video_processor = AvatarVideoProcessor(
                    heygen_service=heygen_service,
                    gcs_service=gcs_qa_service.gcs_service,
                    supabase_client=supabase_client
                )
                logger.info("✅ Avatar Video Processor initialized")
            else:
                logger.warning("⚠️ Avatar Video Processor not initialized (missing dependencies)")
        except Exception as e:
            logger.error(f"❌ HeyGen Service initialization error: {e}")
            heygen_service = None
            avatar_video_processor = None
        
        # Set GOOGLE_APPLICATION_CREDENTIALS if service account file exists
        if os.path.exists('service-account-key.json'):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account-key.json'
            logger.info("✅ Set GOOGLE_APPLICATION_CREDENTIALS to service-account-key.json")
        
        # Initialize Company Bucket Service (NEW - Immediate bucket creation)
        try:
            if initialize_company_bucket_service():
                company_bucket_service = get_company_bucket_service()
                logger.info("✅ Company Bucket Service initialized (Immediate GCS bucket creation)")
            else:
                logger.error("❌ Company Bucket Service initialization failed")
                company_bucket_service = None
        except Exception as e:
            logger.error(f"❌ Company Bucket Service initialization error: {e}")
            company_bucket_service = None
        
        # Initialize OpenAI Client
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                openai_client = OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI Client initialized")
            else:
                logger.warning("⚠️ OpenAI API key not found - OpenAI features will be limited")
                openai_client = None
        except Exception as e:
            logger.error(f"❌ OpenAI Client initialization error: {e}")
            openai_client = None
        
        # Initialize Supabase Client
        try:
            from supabase import create_client
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase Client initialized")
            else:
                logger.warning("⚠️ Supabase credentials not found - database features will be limited")
                supabase = None
        except Exception as e:
            logger.error(f"❌ Supabase Client initialization error: {e}")
            supabase = None
        
        # Initialize GCS-based Loom Video Processor (NEW - Replaces Pinecone-based processor)
        try:
            if openai_api_key:
                loom_processor_gcs = LoomVideoProcessorGCS(
                    openai_api_key=openai_api_key
                )
                logger.info("✅ GCS-based Loom Video Processor initialized")
            else:
                logger.warning("⚠️ OPENAI_API_KEY not found, Loom Processor not initialized")
                loom_processor_gcs = None
        except Exception as e:
            logger.error(f"❌ GCS-based Loom Video Processor initialization error: {e}")
            loom_processor_gcs = None
        
        logger.info("🎉 All GCS-based components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down GCS-based QuDemo Python Backend...")
    try:
        # Graceful shutdown - close any active connections
        import asyncio
        import signal
        
        # Give some time for active requests to complete
        await asyncio.sleep(1)
        
        logger.info("✅ Shutdown completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="GCS-based QuDemo Python Backend",
    description="Optimized backend with Google Cloud Storage architecture",
    version="3.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(company_router, prefix="/api/company", tags=["Company Management"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://qu-demo.vercel.app",
        "https://qudemo.com",
        "https://qudemo-frontend.vercel.app",
        "https://qudemo.vercel.app",
        "https://testqudemo.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuDemoContentRequest(BaseModel):
    video_urls: Optional[List[str]] = []
    website_urls: Optional[List[str]] = []

class UrlRequest(BaseModel):
    url: str

class BatchUrlRequest(BaseModel):
    urls: List[str]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GCS-based QuDemo Python Backend",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "Google Cloud Storage Architecture",
            "Enhanced Q&A with Context-Aware Answers",
            "Intelligent Content Routing",
            "Advanced Video Processing",
            "Smart Web Scraping"
        ]
    }

@app.post("/")
async def root_post():
    """Handle POST requests to root - return helpful error"""
    return {
        "error": "Invalid endpoint",
        "message": "POST requests are not supported at the root path",
        "hint": "Use specific endpoints like /ask/{company_name}/{qudemo_id}",
        "available_endpoints": {
            "Q&A": "POST /ask/{company_name}/{qudemo_id}",
            "Video Processing": "POST /process-video",
            "Document Processing": "POST /process-document",
            "Health Check": "GET /health",
            "Status": "GET /status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        components_status = {
            "loom_processor_gcs": loom_processor_gcs is not None,
            "gcs_qa_service": gcs_qa_service is not None,
            "simple_transcriber": simple_transcriber is not None,
            "company_bucket_service": company_bucket_service is not None
        }
        
        all_healthy = all(components_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/heygen-voices")
async def get_heygen_voices():
    """
    Get available HeyGen voices for avatar video generation
    
    NOTE: Voice IDs should be replaced with actual HeyGen voice IDs from your HeyGen account.
    To get real HeyGen voice IDs:
    1. Go to HeyGen Studio > Voice Library
    2. Select a voice and inspect the API call to get the voice_id
    3. Replace the IDs below with actual HeyGen voice IDs
    
    Currently using placeholder IDs for preview purposes.
    """
    try:
        voices = [
            # Custom Voice (Default - Recommended)
            {
                "id": "01d674cfd32b4728a3fddd21b7e7d543",
                "name": "Custom Professional (Recommended)",
                "description": "Our custom trained voice - warm, professional, and engaging",
                "language": "English (US)",
                "gender": "Male",
                "sample_text": "Hello! Welcome to our platform. I'm here to answer your questions and help you succeed.",
                "rate": 0.95,
                "pitch": 0.9,
                "is_default": True,
                "is_custom": True
            },
            
            # Female Voices
            {
                "id": "1bd001e7e50f421d891986aad5158bc8",
                "name": "Sara - Professional Female",
                "description": "Clear, professional female voice ideal for business presentations",
                "language": "English (US)",
                "gender": "Female",
                "sample_text": "Hello! I can help you understand our product better and guide you through the features.",
                "rate": 1.0,
                "pitch": 1.2,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "af94e4b95b6d41e79a6b543fc7b16501",
                "name": "Emma - Friendly Female",
                "description": "Warm and approachable female voice perfect for customer engagement",
                "language": "English (US)",
                "gender": "Female",
                "sample_text": "Welcome! I'm excited to show you what we can do and help you get started!",
                "rate": 1.0,
                "pitch": 1.15,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "c1d9d7e9f8a74c8e9e1b2f3c4d5e6f7a",
                "name": "Lisa - Energetic Female",
                "description": "Upbeat and dynamic female voice for engaging content",
                "language": "English (US)",
                "gender": "Female",
                "sample_text": "Hi there! Let's dive into this exciting opportunity together!",
                "rate": 1.05,
                "pitch": 1.25,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "d4e8f3c2b1a94d5e6f7a8b9c0d1e2f3a",
                "name": "Rachel - Corporate Female",
                "description": "Sophisticated and authoritative female voice for executive content",
                "language": "English (US)",
                "gender": "Female",
                "sample_text": "Good afternoon. Let me present our strategic solution and key benefits.",
                "rate": 0.95,
                "pitch": 1.1,
                "is_default": False,
                "is_custom": False
            },
            
            # Male Voices
            {
                "id": "2d5b0e6cf36f4355b6f8c3c0f6c5e935",
                "name": "Mike - Friendly Male",
                "description": "Conversational and personable male voice",
                "language": "English (US)",
                "gender": "Male",
                "sample_text": "Hi there! Let me walk you through this and show you how everything works.",
                "rate": 1.0,
                "pitch": 0.95,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8",
                "name": "David - Corporate Male",
                "description": "Professional and authoritative male voice for business content",
                "language": "English (US)",
                "gender": "Male",
                "sample_text": "Good day. Let me explain our solution and demonstrate its key capabilities.",
                "rate": 0.9,
                "pitch": 0.85,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
                "name": "James - Deep Male",
                "description": "Rich, deep male voice with commanding presence",
                "language": "English (US)",
                "gender": "Male",
                "sample_text": "Welcome. Allow me to guide you through our comprehensive platform.",
                "rate": 0.85,
                "pitch": 0.75,
                "is_default": False,
                "is_custom": False
            },
            {
                "id": "f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
                "name": "Ryan - Energetic Male",
                "description": "Upbeat and enthusiastic male voice",
                "language": "English (US)",
                "gender": "Male",
                "sample_text": "Hey! I'm really excited to show you all the amazing features we have!",
                "rate": 1.05,
                "pitch": 1.0,
                "is_default": False,
                "is_custom": False
            }
        ]
        
        return {
            "success": True,
            "voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching HeyGen voices: {e}")
        return {
            "success": False,
            "error": str(e),
            "voices": []
        }

@app.get("/memory-status")
async def memory_status():
    """Memory status endpoint for health checks"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "success": True,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percentage": memory.percent
            },
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        return {
            "success": True,
            "memory": {
                "total": "N/A",
                "available": "N/A", 
                "used": "N/A",
                "percentage": "N/A"
            },
            "status": "psutil_not_available",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Memory status check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/ask/{company_name}/{qudemo_id}")
async def ask_question(company_name: str, qudemo_id: str, request: QuestionRequest):
    """Ask a question and get context-aware answer using GCS Q&A service (primary) with fallbacks"""
    try:
        # Try GCS Q&A service first (primary for Google Cloud Storage)
        # Always try to use GCS service directly
        logger.info(f"❓ Processing question for {company_name} qudemo {qudemo_id} using GCS Q&A (forced)")
        
        # Create GCS service instance directly
        gcs_service = GCSQAService()
        answer_result = await gcs_service.ask_question(
            question=request.question,
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if answer_result['success']:
            # Fix branding: Replace "Q-Demo" with "Qudemo" and fix "Chatwoot" spelling
            answer_text = answer_result['answer']
            answer_text = answer_text.replace("Q-Demo", "Qudemo")
            answer_text = answer_text.replace("Q-demo", "Qudemo")
            answer_text = answer_text.replace("q-demo", "Qudemo")
            
            # Fix Chatwoot spelling variations
            answer_text = answer_text.replace("Chatwood", "Chatwoot")
            answer_text = answer_text.replace("chatwood", "Chatwoot")
            answer_text = answer_text.replace("Chat wood", "Chatwoot")
            answer_text = answer_text.replace("chat wood", "Chatwoot")
            answer_text = answer_text.replace("Chat Wood", "Chatwoot")
            answer_text = answer_text.replace("Chatwot", "Chatwoot")
            answer_text = answer_text.replace("chatwot", "Chatwoot")
            
            return {
                'success': True,
                'answer': answer_text,
                'sources': answer_result.get('sources', []),
                'total_sources': len(answer_result.get('sources', [])) if answer_result.get('sources') else answer_result.get('total_sources', 0),
                'search_score': answer_result.get('search_score', 0),
                'confidence_score': answer_result.get('confidence', answer_result.get('confidence_score', 0)),
                'content_types_found': answer_result.get('content_types_found', []),
                'difficulty_level': answer_result.get('difficulty_level', 'intermediate'),
                'estimated_time': answer_result.get('estimated_time', '2-3 minutes'),
                'start': answer_result.get('timestamp', 0) if gcs_qa_service else (answer_result.get('timestamp', {}).get('start_time', 0) if answer_result.get('timestamp') else (answer_result.get('sources', [{}])[0].get('start_timestamp', 0) if answer_result.get('sources') else 0)),
                'end': answer_result.get('end', 0) if gcs_qa_service else (answer_result.get('timestamp', {}).get('end_time', 0) if answer_result.get('timestamp') else (answer_result.get('sources', [{}])[0].get('end_timestamp', 0) if answer_result.get('sources') else 0)),
                'video_url': answer_result.get('video_url', '') if gcs_qa_service else (answer_result.get('sources', [{}])[0].get('video_url', '') if answer_result.get('sources') else ''),
                'video_title': answer_result.get('video_title', '') if gcs_qa_service else (answer_result.get('sources', [{}])[0].get('video_title', '') if answer_result.get('sources') else ''),
                'timestamp': answer_result.get('timestamp', 0) if gcs_qa_service else (answer_result.get('timestamp', {}).get('start_time', 0) if answer_result.get('timestamp') else (answer_result.get('sources', [{}])[0].get('start_timestamp', 0) if answer_result.get('sources') else 0)),
                'formatted_timestamp': answer_result.get('formatted_timestamp', '') if gcs_qa_service else (answer_result.get('timestamp', {}).get('formatted_start', '') if answer_result.get('timestamp') else ''),
                'answer_source': 'gcs_transcript_search',
                # Avatar video fields for AI-generated video answers
                'has_avatar_video': answer_result.get('has_avatar_video', False),
                'avatar_video_url': answer_result.get('avatar_video_url', ''),
                'faq_id': answer_result.get('faq_id', '')
            }
        else:
            # Check if this is a "no relevant content" case vs actual error
            error_message = answer_result.get('error', 'Unknown error')
            is_no_content = 'no relevant content found' in error_message.lower() or 'no relevant information' in error_message.lower()
            
            if is_no_content:
                # Fix branding in error messages too
                no_content_msg = 'No relevant information found'
                no_content_msg = no_content_msg.replace("Q-Demo", "Qudemo")
                no_content_msg = no_content_msg.replace("Q-demo", "Qudemo")
                no_content_msg = no_content_msg.replace("q-demo", "Qudemo")
                
                # Fix Chatwoot spelling variations in error messages
                no_content_msg = no_content_msg.replace("Chatwood", "Chatwoot")
                no_content_msg = no_content_msg.replace("chatwood", "Chatwoot")
                no_content_msg = no_content_msg.replace("Chat wood", "Chatwoot")
                no_content_msg = no_content_msg.replace("chat wood", "Chatwoot")
                
                return {
                    'success': False,
                    'error': no_content_msg,
                    'answer': no_content_msg,
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [],
                    'total_sources': 0,
                    'search_score': 0,
                    'confidence_score': 0,
                    'content_types_found': [],
                    'difficulty_level': 'unknown',
                    'estimated_time': 'unknown',
                    'formatted_timestamp': '',
                    'answer_source': 'gcs_transcript_search'
                }
            else:
                # This is an actual error, show the error message
                # Fix branding in error messages
                fixed_error = error_message.replace("Q-Demo", "Qudemo").replace("Q-demo", "Qudemo").replace("q-demo", "Qudemo")
                fixed_answer = answer_result.get('answer', '').replace("Q-Demo", "Qudemo").replace("Q-demo", "Qudemo").replace("q-demo", "Qudemo")
                
                # Fix Chatwoot spelling variations in error messages
                fixed_error = fixed_error.replace("Chatwood", "Chatwoot").replace("chatwood", "Chatwoot").replace("Chat wood", "Chatwoot").replace("chat wood", "Chatwoot")
                fixed_answer = fixed_answer.replace("Chatwood", "Chatwoot").replace("chatwood", "Chatwoot").replace("Chat wood", "Chatwoot").replace("chat wood", "Chatwoot")
                
                return {
                    'success': False,
                    'error': fixed_error,
                    'answer': fixed_answer,
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [],
                    'total_sources': 0,
                    'search_score': 0,
                    'confidence_score': 0,
                    'content_types_found': [],
                    'difficulty_level': 'unknown',
                    'estimated_time': 'unknown',
                    'formatted_timestamp': '',
                    'answer_source': 'gcs_transcript_search'
                }
            
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/generate-suggested-questions/{company_name}/{qudemo_id}")
async def generate_suggested_questions(company_name: str, qudemo_id: str):
    """Generate fresh suggested questions for a QuDemo without storing them"""
    try:
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not available")
        
        logger.info(f"🤖 GENERATING FRESH suggested questions for {company_name}/{qudemo_id}")
        logger.info(f"🔍 Company: {company_name}, QuDemo: {qudemo_id}")
        
        # Generate fresh suggested questions (no storage)
        suggested_questions = gcs_qa_service.generate_and_store_suggested_questions(company_name, qudemo_id)
        
        logger.info(f"🎯 Generated {len(suggested_questions)} fresh questions")
        logger.info(f"📝 Questions: {suggested_questions}")
        
        return {
            'success': True,
            'suggested_questions': suggested_questions,
            'total_questions': len(suggested_questions),
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'generated_at': datetime.now().isoformat(),
            'fresh_generation': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating suggested questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/suggested-questions/{company_name}/{qudemo_id}")
async def delete_suggested_question(company_name: str, qudemo_id: str, request: Request):
    """Delete a suggested question by question text"""
    try:
        body = await request.json()
        question_text = body.get('question', '').strip()
        
        if not question_text:
            return {"success": False, "error": "Question text is required"}
        
        logger.info(f"🗑️ Deleting suggested question: '{question_text}' from {company_name}/{qudemo_id}")
        
        # Get current FAQs from GCS
        if not gcs_qa_service:
            return {"success": False, "error": "GCS QA service not available"}
        
        faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
        
        if not faqs_data:
            return {"success": False, "error": "FAQs not found"}
        
        faqs = faqs_data.get('faqs', [])
        
        # Find and remove the FAQ with matching question
        faq_found = False
        updated_faqs = []
        deleted_faq = None
        
        for faq in faqs:
            if faq.get('question', '').strip() == question_text:
                faq_found = True
                deleted_faq = faq
                logger.info(f"📝 Found FAQ to delete: {faq.get('id')} - {faq.get('question')}")
            else:
                updated_faqs.append(faq)
        
        if not faq_found:
            return {"success": False, "error": f"Question not found: '{question_text}'"}
        
        # Save updated FAQs back to GCS
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
        blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
        
        faq_data = {
            "version": "1.0",
            "qudemo_id": qudemo_id,
            "company_name": company_name,
            "updated_at": datetime.now().isoformat(),
            "faqs": updated_faqs
        }
        
        blob.upload_from_string(json.dumps(faq_data, indent=2), content_type='application/json')
        
        logger.info(f"✅ Deleted suggested question. Remaining FAQs: {len(updated_faqs)}")
        
        # Try to delete associated avatar video from database
        video_deleted = False
        if deleted_faq:
            try:
                faq_id = deleted_faq.get('id')
                delete_response = supabase.table('avatar_videos').delete().eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                
                if delete_response.data:
                    video_deleted = True
                    logger.info(f"✅ Deleted avatar video from database for FAQ {faq_id}")
                    
            except Exception as video_error:
                logger.warning(f"⚠️ Error deleting avatar video: {video_error}")
        
        return {
            "success": True,
            "message": "Suggested question deleted successfully",
            "deleted_question": question_text,
            "deleted_faq_id": deleted_faq.get('id') if deleted_faq else None,
            "remaining_questions": len([f for f in updated_faqs if not f.get('is_fallback') and not f.get('is_intro')]),
            "video_deleted": video_deleted
        }
        
    except Exception as e:
        logger.error(f"❌ Error deleting suggested question: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/suggested-questions/{company_name}/{qudemo_id}")
async def get_suggested_questions(company_name: str, qudemo_id: str):
    """Get stored suggested questions for a QuDemo (from FAQs, excluding fallbacks)"""
    try:
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not available")
        
        logger.info(f"📖 FETCHING FAQ questions for {company_name}/{qudemo_id}")
        
        # First, try to get FAQs (which include video FAQs + suggested question FAQs)
        try:
            faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
            
            if faqs_data and 'faqs' in faqs_data:
                # Extract questions from FAQs, excluding fallback FAQs
                faq_questions = [
                    faq['question'] 
                    for faq in faqs_data['faqs'] 
                    if not faq.get('is_fallback', False) 
                    and not faq.get('is_intro', False)
                    and faq.get('question') not in ['NO_ANSWER_FOUND', 'SALES_INQUIRY', 'INTRO_VIDEO']
                ]
                
                if faq_questions:
                    logger.info(f"✅ Retrieved {len(faq_questions)} FAQ questions (excluding {len(faqs_data['faqs']) - len(faq_questions)} fallback FAQs)")
                    
                    # Fix branding
                    fixed_questions = []
                    for question in faq_questions:
                        fixed_question = question.replace("Q-Demo", "Qudemo")
                        fixed_question = fixed_question.replace("Q-demo", "Qudemo")
                        fixed_question = fixed_question.replace("q-demo", "Qudemo")
                        fixed_question = fixed_question.replace("Chatwood", "Chatwoot")
                        fixed_question = fixed_question.replace("chatwood", "Chatwoot")
                        fixed_question = fixed_question.replace("Chat wood", "Chatwoot")
                        fixed_question = fixed_question.replace("chat wood", "Chatwoot")
                        fixed_question = fixed_question.replace("Chat Wood", "Chatwoot")
                        fixed_question = fixed_question.replace("Chatwot", "Chatwoot")
                        fixed_question = fixed_question.replace("chatwot", "Chatwoot")
                        fixed_questions.append(fixed_question)
                    
                    logger.info(f"✅ Returning {len(fixed_questions)} FAQ questions")
                    return {
                        "success": True,
                        "questions": fixed_questions,
                        "suggested_questions": fixed_questions  # For backward compatibility
                    }
        except Exception as faq_error:
            logger.error(f"❌ Error retrieving FAQs: {faq_error}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            logger.warning(f"⚠️ Falling back to old suggested questions format")
        
        # Fallback: Get old suggested questions if FAQs not available
        logger.info(f"📖 Falling back to old suggested questions format")
        stored_questions = gcs_qa_service.gcs_service.get_suggested_questions(company_name, qudemo_id)
        
        if stored_questions:
            logger.info(f"✅ Retrieved {len(stored_questions)} stored suggested questions (PRE-SHUFFLED if multiple videos)")
            
            # Fix branding: Replace "Q-Demo" with "Qudemo" and fix "Chatwoot" spelling in all questions
            # This is especially important for the welcome Qudemo (ID: 48b29bfb-b290-4669-9f25-ee411cdb1d9d)
            fixed_questions = []
            for question in stored_questions:
                # Replace all variations of Q-Demo with Qudemo
                fixed_question = question.replace("Q-Demo", "Qudemo")
                fixed_question = fixed_question.replace("Q-demo", "Qudemo")
                fixed_question = fixed_question.replace("q-demo", "Qudemo")
                
                # Fix Chatwoot spelling variations
                fixed_question = fixed_question.replace("Chatwood", "Chatwoot")
                fixed_question = fixed_question.replace("chatwood", "Chatwoot")
                fixed_question = fixed_question.replace("Chat wood", "Chatwoot")
                fixed_question = fixed_question.replace("chat wood", "Chatwoot")
                fixed_question = fixed_question.replace("Chat Wood", "Chatwoot")
                fixed_question = fixed_question.replace("Chatwot", "Chatwoot")
                fixed_question = fixed_question.replace("chatwot", "Chatwoot")
                
                fixed_questions.append(fixed_question)
            
            logger.info(f"✅ Returning {len(fixed_questions)} questions for {company_name}/{qudemo_id}")
            
            return {
                "success": True,
                "suggested_questions": fixed_questions
            }
        else:
            logger.info(f"📝 No stored suggested questions found for {company_name}/{qudemo_id}")
            return {
                "success": True,
                "suggested_questions": ["What is this about?"]
            }
        
    except Exception as e:
        logger.error(f"❌ Error fetching suggested questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/sources/{company_name}")
async def get_knowledge_sources_company(company_name: str):
    """Get knowledge sources for a company using GCS (optimized to avoid excessive logging)"""
    try:
        logger.info(f"📚 Getting knowledge sources for company {company_name}")
        
        # Use GCS service to get company data
        gcs_service = gcs_qa_service.gcs_service
        if gcs_service:
            # Get list of qudemos for the company (without individual processing)
            qudemos = gcs_service.list_qudemos(company_name)
            
            # Return a simplified response without processing each QuDemo individually
            # This prevents excessive logging and improves performance
            knowledge_sources = []
            
            # Only process a limited number of QuDemos to avoid excessive logging
            max_qudemos_to_process = 5  # Limit to prevent excessive API calls
            qudemos_to_process = qudemos[:max_qudemos_to_process]
            
            logger.info(f"📊 Found {len(qudemos)} QuDemos for company {company_name}, processing {len(qudemos_to_process)} for knowledge sources")
            
            for qudemo_id in qudemos_to_process:
                try:
                    # Get basic transcript data without detailed logging
                    transcript_data = gcs_service.get_video_transcript(
                        company_name=company_name,
                        qudemo_id=qudemo_id
                    )
                    
                    if transcript_data:
                        video_url = transcript_data.get('video_url', '')
                        video_title = transcript_data.get('video_title', 'Unknown')
                        chunks = transcript_data.get('chunks', [])
                        
                        knowledge_sources.append({
                            'type': 'video',
                            'url': video_url,
                            'title': video_title,
                            'chunks_count': len(chunks),
                            'qudemo_id': qudemo_id,
                            'company_name': company_name
                        })
                except Exception as e:
                    # Reduce logging verbosity for company-level endpoint
                    logger.debug(f"⚠️ Could not get transcript for {qudemo_id}: {e}")
                    continue
            
            # Add summary for remaining QuDemos without processing them
            remaining_qudemos = len(qudemos) - len(qudemos_to_process)
            if remaining_qudemos > 0:
                knowledge_sources.append({
                    'type': 'summary',
                    'url': '',
                    'title': f'... and {remaining_qudemos} more QuDemos',
                    'chunks_count': 0,
                    'qudemo_id': 'summary',
                    'company_name': company_name,
                    'is_summary': True
                })
            
            logger.info(f"✅ Retrieved {len(knowledge_sources)} knowledge sources for company {company_name}")
            
            return {
                "success": True,
                "data": {
                    "sources": knowledge_sources,
                    "total_sources": len(knowledge_sources),
                    "total_qudemos": len(qudemos),
                    "processed_qudemos": len(qudemos_to_process),
                    "remaining_qudemos": remaining_qudemos
                }
            }
        else:
            return {
                "success": False,
                "error": "GCS service not initialized",
                "data": {"sources": [], "total_sources": 0}
            }
            
    except Exception as e:
        logger.error(f"❌ Error getting knowledge sources for company: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {"sources": [], "total_sources": 0}
        }

@app.get("/knowledge/sources/{company_name}/{qudemo_id}")
async def get_knowledge_sources_qudemo(company_name: str, qudemo_id: str):
    """Get knowledge sources for a specific qudemo using GCS"""
    try:
        logger.info(f"📚 Getting knowledge sources for {company_name} qudemo {qudemo_id}")
        
        # Use GCS Q&A service to get transcript data
        if gcs_qa_service:
            transcript_data = gcs_qa_service.gcs_service.get_video_transcript(
                company_name=company_name,
                qudemo_id=qudemo_id
            )
            
            if transcript_data:
                # Extract relevant information from transcript data
                chunks = transcript_data.get('chunks', [])
                segments = transcript_data.get('segments', [])
                video_url = transcript_data.get('video_url', '')
                video_title = transcript_data.get('video_title', 'Unknown')
                
                # Create knowledge sources summary
                knowledge_sources = {
                    'video_sources': [
                        {
                            'type': 'video',
                            'url': video_url,
                            'title': video_title,
                            'chunks_count': len(chunks),
                            'segments_count': len(segments),
                            'transcript_length': len(transcript_data.get('transcript', '')),
                            'processed_at': transcript_data.get('metadata', {}).get('processed_at', 'Unknown')
                        }
                    ],
                    'total_chunks': len(chunks),
                    'total_segments': len(segments),
                    'total_sources': 1,
                    'storage_type': 'gcs',
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
                
                logger.info(f"✅ Retrieved GCS knowledge sources: {len(chunks)} chunks, {len(segments)} segments")
                
                return {
                    'success': True,
                    'data': knowledge_sources,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            else:
                logger.warning(f"⚠️ No transcript data found for {company_name}/{qudemo_id}")
                return {
                    'success': True,
                    'data': {
                        'video_sources': [],
                        'total_chunks': 0,
                        'total_segments': 0,
                        'total_sources': 0,
                        'storage_type': 'gcs',
                        'company_name': company_name,
                        'qudemo_id': qudemo_id
                    },
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
        else:
            raise HTTPException(status_code=500, detail="GCS Q&A service not initialized")
            
    except Exception as e:
        logger.error(f"❌ Error getting knowledge sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/website-count/{company_name}/{qudemo_id}")
async def get_website_count(company_name: str, qudemo_id: str):
    """Get website count for a QuDemo"""
    try:
        logger.info(f"🌐 Getting website count for {company_name}/{qudemo_id}")
        
        # Get website count from GCS
        if gcs_qa_service:
            count = gcs_qa_service.gcs_service.get_website_count(company_name, qudemo_id)
            return {
                "success": True,
                "data": {
                    "count": count
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "count": 0
                }
            }
        
    except Exception as e:
        logger.error(f"❌ Error getting website count: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/upload-loom-media/{company_name}/{qudemo_id}")
async def upload_loom_media(
    company_name: str, 
    qudemo_id: str,
    video_url: str = Form(...),
    media_file: UploadFile = File(...)
):
    """Upload media file for Loom video processing with real transcription"""
    try:
        logger.info(f"📁 Uploading media file for Loom video: {video_url}")
        logger.info(f"🏢 Company: {company_name}, QuDemo ID: {qudemo_id}")
        logger.info(f"📄 File: {media_file.filename}, Size: {media_file.size} bytes")
        
        # Validate file type
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mp3', '.wav', '.m4a'}
        file_extension = os.path.splitext(media_file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(media_file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"💾 Saved uploaded file to: {temp_file_path}")
        
        # Process with media file using GCS-based Loom processor
        if loom_processor_gcs:
            result = loom_processor_gcs.process_video(
                video_url, company_name, qudemo_id
            )
        else:
            raise HTTPException(status_code=500, detail="GCS-based Loom processor not available")
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
            logger.info(f"🗑️ Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up temporary file: {e}")
        
        if result and result.get('success'):
            logger.info(f"✅ Successfully processed Loom video with media file")
            return {
                "success": True,
                "message": "Loom video processed successfully with real transcription",
                "chunks_stored": result.get('chunks_stored', 0),
                "video_type": result.get('video_type', 'loom_video'),
                "method": result.get('method', 'whisper_transcription'),
                "duration": result.get('duration', 0),
                "total_segments": result.get('total_segments', 0)
            }
        else:
            logger.error(f"❌ Loom video processing failed: {result}")
            raise HTTPException(
                status_code=500, 
                detail=f"Loom video processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"❌ Error uploading Loom media: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-video-enhanced/{company_name}/{qudemo_id}")
async def process_video_enhanced(company_name: str, qudemo_id: str, request: QuDemoContentRequest):
    """Process video using enhanced chunking and topic analysis strategy"""
    try:
        if not enhanced_chunking_processor:
            raise HTTPException(status_code=500, detail="Enhanced Chunking Processor not initialized")
        
        if not request.video_urls or len(request.video_urls) == 0:
            raise HTTPException(status_code=400, detail="No video URLs provided")
        
        results = []
        
        for video_url in request.video_urls:
            try:
                logger.info(f"🎥 Processing video with enhanced chunking: {video_url}")
                
                result = await enhanced_chunking_processor.process_video_with_topic_analysis(
                    video_url, company_name, qudemo_id
                )
                
                results.append({
                    'video_url': video_url,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing video {video_url}: {e}")
                results.append({
                    'video_url': video_url,
                    'result': {
                        'success': False,
                        'error': str(e)
                    }
                })
        
        return {
            "success": True,
            "message": "Enhanced video processing completed",
            "results": results,
            "total_videos": len(request.video_urls),
            "method": "enhanced_chunking_topic_analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced video processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-presenter-photo")
async def upload_presenter_photo(
    presenterPhoto: UploadFile = File(...),
    qudemoId: str = Form(...),
    companyName: str = Form(...)
):
    """Upload presenter photo to GCS and return public URL"""
    try:
        logger.info(f"📸 Uploading presenter photo for QuDemo: {qudemoId}, Company: {companyName}")
        
        # Read file content
        file_content = await presenterPhoto.read()
        logger.info(f"📊 Read {len(file_content)} bytes from file: {presenterPhoto.filename}")
        
        # Validate file size (max 5MB)
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
        
        # Upload to GCS
        if gcs_qa_service:
            file_path = f"{companyName}/{qudemoId}/presenter_photo.jpg"
            
            # Get company bucket
            bucket = gcs_qa_service.gcs_service.client.bucket(f"qudemo-{companyName.lower().replace(' ', '-')}")
            if not bucket.exists():
                bucket = gcs_qa_service.gcs_service.client.create_bucket(bucket.name)
                logger.info(f"✅ Created bucket for company: {bucket.name}")
            
            # Upload file
            blob = bucket.blob(file_path)
            blob.upload_from_string(file_content, content_type=presenterPhoto.content_type or 'image/jpeg')
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"✅ Presenter photo uploaded successfully: {public_url}")
            
            return {
                "success": True,
                "presenter_photo_url": public_url,
                "qudemo_id": qudemoId,
                "company_name": companyName
            }
        else:
            raise HTTPException(status_code=500, detail="GCS service not initialized")
            
    except Exception as e:
        logger.error(f"❌ Error uploading presenter photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    qudemo_id: str = Form(...),
    document_id: str = Form(...),
    mime_type: str = Form(...)
):
    """Process a document and extract text content"""
    try:
        if not document_processor:
            logger.error("❌ Document processor not available")
            return {
                "success": False,
                "error": "Document processor not available",
                "document_id": document_id
            }
        
        logger.info(f"📄 Processing document: {document_id} for {company_name}/{qudemo_id}")
        
        # Read file content
        file_content = await file.read()
        logger.info(f"📊 Read {len(file_content)} bytes from file: {file.filename}")
        
        # Process the document
        success = document_processor.process_document_from_content(
            company_name=company_name,
            qudemo_id=qudemo_id,
            document_id=document_id,
            file_content=file_content,
            mime_type=mime_type,
            filename=file.filename
        )
        
        if success:
            logger.info(f"✅ Document processed successfully: {document_id}")
            
            # Generate suggested questions after successful document processing
            try:
                logger.info(f"🤖 Generating suggested questions after document processing for {company_name}/{qudemo_id}")
                suggested_questions = gcs_qa_service.generate_and_store_suggested_questions(company_name, qudemo_id)
                logger.info(f"✅ Generated {len(suggested_questions)} suggested questions after document processing")
            except Exception as e:
                logger.error(f"❌ Error generating suggested questions after document processing: {e}")
            
            # Check if presenter photo exists and generate FAQs for avatar videos
            try:
                await generate_faq_for_avatar_videos(company_name, qudemo_id)
            except Exception as faq_error:
                logger.error(f"❌ Error generating FAQs for avatar videos: {faq_error}")
            
            # Notify Node.js backend of successful processing
            try:
                import requests
                node_api_url = os.getenv('NODE_API_BASE_URL', 'http://localhost:5000')
                notification_url = f"{node_api_url}/api/documents/{document_id}/processing-complete"
                
                notification_data = {
                    "success": True,
                    "message": "Document processed successfully"
                }
                
                response = requests.post(notification_url, json=notification_data, timeout=10)
                if response.ok:
                    logger.info(f"✅ Successfully notified Node.js backend of document completion: {document_id}")
                else:
                    logger.warning(f"⚠️ Failed to notify Node.js backend: {response.status_code}")
            except Exception as notify_error:
                logger.warning(f"⚠️ Error notifying Node.js backend: {notify_error}")
            
            return {
                "success": True,
                "message": "Document processed successfully",
                "document_id": document_id,
                "company_name": company_name,
                "qudemo_id": qudemo_id
            }
        else:
            logger.error(f"❌ Failed to process document: {document_id}")
            
            # Notify Node.js backend of failed processing
            try:
                import requests
                node_api_url = os.getenv('NODE_API_BASE_URL', 'http://localhost:5000')
                notification_url = f"{node_api_url}/api/documents/{document_id}/processing-complete"
                
                notification_data = {
                    "success": False,
                    "error": "Failed to process document"
                }
                
                response = requests.post(notification_url, json=notification_data, timeout=10)
                if response.ok:
                    logger.info(f"✅ Successfully notified Node.js backend of document failure: {document_id}")
                else:
                    logger.warning(f"⚠️ Failed to notify Node.js backend: {response.status_code}")
            except Exception as notify_error:
                logger.warning(f"⚠️ Error notifying Node.js backend: {notify_error}")
            
            return {
                "success": False,
                "error": "Failed to process document",
                "document_id": document_id
            }
            
    except Exception as e:
        logger.error(f"❌ Document processing error: {str(e)}")
        return {
            "success": False,
            "error": f"Document processing failed: {str(e)}",
            "document_id": document_id
        }

@app.get("/search-documents/{company_name}/{qudemo_id}")
async def search_documents(company_name: str, qudemo_id: str, query: str):
    """Search for content in documents"""
    try:
        if not document_processor:
            logger.error("❌ Document processor not available")
            return {"results": [], "error": "Document processor not available"}
        
        logger.info(f"🔍 Searching documents for: {company_name}/{qudemo_id} - Query: {query}")
        
        results = document_processor.search_document_content(
            company_name=company_name,
            qudemo_id=qudemo_id,
            query=query
        )
        
        logger.info(f"📊 Document search completed: {len(results)} results found")
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"❌ Document search error: {str(e)}")
        return {"results": [], "error": str(e)}

@app.post("/process-website/{company_name}/{qudemo_id}")
async def process_website(company_name: str, qudemo_id: str, website_url: str = Form(...)):
    """Process website scraping with dynamic depth and adaptive timeout"""
    try:
        if not website_scraper:
            raise HTTPException(status_code=500, detail="Website scraper not initialized")
        
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS service not initialized")
        
        logger.info(f"🌐 Starting website processing for: {website_url}")
        
        # Scrape the website
        website_data = await website_scraper.scrape_website(website_url)
        
        # Store the scraped content
        if website_data and website_data.get('scraped_pages'):
            storage_success = gcs_qa_service.store_website_content(company_name, qudemo_id, website_data)
            
            if storage_success:
                logger.info(f"✅ Website processing completed: {website_data['total_pages']} pages scraped")
                return {
                    "success": True,
                    "website_id": website_data['website_id'],
                    "total_pages": website_data['total_pages'],
                    "scraping_status": website_data['scraping_status'],
                    "base_url": website_data['base_url'],
                    "errors": website_data.get('errors', [])
                }
            else:
                logger.error("❌ Failed to store website content")
                return {
                    "success": False,
                    "error": "Failed to store website content",
                    "website_data": website_data
                }
        else:
            logger.warning(f"⚠️ No content scraped from website: {website_url}")
            return {
                "success": False,
                "error": "No content could be scraped from the website",
                "website_data": website_data
            }
            
    except Exception as e:
        logger.error(f"❌ Website processing error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/process-qudemo-content/{company_name}/{qudemo_id}")
async def process_qudemo_content(company_name: str, qudemo_id: str, request: QuDemoContentRequest):
    """Process qudemo content with optimized processing order: Videos first, then website"""
    try:
        # Check if we have GCS services available
        if not gcs_qa_service and not simple_transcriber:
            error_msg = "GCS services not initialized. Please check: 1) GEMINI_API_KEY environment variable, 2) service-account-key.json file, 3) Google Cloud credentials"
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info(f"🔄 Processing qudemo content for {company_name} qudemo {qudemo_id}")
        
        # 🎯 OPTIMIZATION STRATEGY: Videos First, Then Website
        logger.info("🚀 Using optimized processing order: Videos (fast) → Website (may take longer)")
        
        total_chunks = 0
        processing_order = []
        processing_errors = []
        successful_content = {
            "videos": [],
            "websites": []
        }
        
        # Step 1: Process videos first (faster processing)
        if request.video_urls and len(request.video_urls) > 0:
            logger.info(f"🎥 Step 1: Processing {len(request.video_urls)} videos (fast processing first)")
            logger.info("⏱️ Video processing typically takes 1-3 minutes per video")
            processing_order.append("videos")
            
            # Process videos sequentially to avoid conflicts
            youtube_videos = []
            loom_videos = []
            
            # Separate videos by type
            for video_url in request.video_urls:
                if 'youtube.com' in video_url or 'youtu.be' in video_url:
                    youtube_videos.append(video_url)
                elif 'loom.com' in video_url:
                    loom_videos.append(video_url)
            
            logger.info(f"📊 Video processing plan: {len(youtube_videos)} YouTube videos, {len(loom_videos)} Loom videos")
            
            # Process YouTube videos first (to avoid yt-dlp conflicts)
            for i, video_url in enumerate(youtube_videos):
                try:
                    logger.info(f"🔍 Processing YouTube video {i+1}/{len(youtube_videos)}: {video_url}")
                    logger.info(f"🎬 Detected video type: youtube")
                    
                    # Process video using Simple Gemini Transcriber (PRIMARY - GCS-based)
                    if simple_transcriber:
                        logger.info(f"🎥 Processing YouTube video with SIMPLE GCS TRANSCRIBER: {video_url}")
                        
                        # Create QuDemo folder if it doesn't exist
                        if company_bucket_service:
                            folder_result = company_bucket_service.create_qudemo_folder(company_name, qudemo_id)
                            if folder_result.get('success'):
                                logger.info(f"📁 QuDemo folder ensured: {company_name}/{qudemo_id}")
                        
                        result = await simple_transcriber.process_video_with_gcs_storage(
                            video_url, company_name, qudemo_id
                        )
                        
                        # Convert result format to match expected structure
                        if result and result.get('success'):
                            result = {
                                'success': True,
                                'chunks_stored': result.get('chunks_created', 0),
                                'video_type': 'youtube',
                                'company_name': company_name,
                                'qudemo_id': qudemo_id,
                                'method': 'simple_gemini_transcriber',
                                'segments_processed': result.get('segments_created', 0),
                                'topics_extracted': 0  # Simple transcriber doesn't extract topics
                            }
                        else:
                            result = {
                                'success': False,
                                'error': result.get('error', 'Unknown error') if result else 'No result returned',
                                'chunks_stored': 0
                            }
                    elif enhanced_video_processor:
                        # Fallback to enhanced video processor
                        logger.info(f"🎥 Using enhanced video processor for YouTube video: {video_url}")
                        result = await enhanced_video_processor.process_youtube_video(
                            video_url, company_name, qudemo_id
                        )
                    else:
                        # Final fallback to direct processor
                        logger.info(f"🎥 Using direct processor for YouTube video: {video_url}")
                        from video_processing import process_video
                        result = await process_video(video_url, company_name, qudemo_id)
                        
                        # Convert result format to match enhanced processor
                        if result and result.get('success'):
                            result = {
                                'success': True,
                                'chunks_stored': result.get('result', {}).get('chunks_created', 0),
                                'video_type': 'youtube',
                                'company_name': company_name,
                                'qudemo_id': qudemo_id
                            }
                        else:
                            result = {
                                'success': False,
                                'error': result.get('error', 'Unknown error') if result else 'No result returned',
                                'chunks_stored': 0
                            }
                    
                    logger.info(f"📊 YouTube video processing result: {result}")
                    
                    if result and result.get('success'):
                        chunks_stored = result.get('chunks_stored', 0)
                        total_chunks += chunks_stored
                        successful_content["videos"].append({
                            "url": video_url,
                            "type": "youtube",
                            "chunks_stored": chunks_stored
                        })
                        logger.info(f"✅ YouTube video processed: {chunks_stored} chunks stored")
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No result'
                        processing_errors.append({
                            "type": "video",
                            "url": video_url,
                            "error": error_msg
                        })
                        logger.error(f"❌ YouTube video processing failed: {error_msg}")
                        
                    # Add delay between YouTube videos to prevent conflicts
                    if i < len(youtube_videos) - 1:
                        logger.info("⏳ Waiting 5s before processing next YouTube video...")
                        import time
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing YouTube video {video_url}: {e}")
                    continue
            
            # Process Loom videos after YouTube videos
            for i, video_url in enumerate(loom_videos):
                try:
                    logger.info(f"🔍 Processing Loom video {i+1}/{len(loom_videos)}: {video_url}")
                    logger.info(f"🎬 Detected video type: loom")
                    
                    # Process video using GCS-based Loom processor
                    if loom_processor_gcs:
                        logger.info(f"🎥 Processing Loom video with GCS: {video_url}")
                        result = loom_processor_gcs.process_video(
                            video_url, company_name, qudemo_id
                        )
                        
                        # Convert result format to match expected structure
                        if result and result.get('success'):
                            result = {
                                'success': True,
                                'chunks_stored': result.get('segments_count', 0),  # Use segments_count as chunks
                                'video_type': 'loom',
                                'company_name': company_name,
                                'qudemo_id': qudemo_id,
                                'method': 'loom_processor_gcs'
                            }
                        else:
                            result = {
                                'success': False,
                                'error': result.get('error', 'Unknown error') if result else 'No result returned',
                                'chunks_stored': 0
                            }
                    else:
                        logger.error("❌ GCS-based Loom processor not available")
                        result = {
                            'success': False,
                            'error': 'GCS-based Loom processor not initialized',
                            'chunks_stored': 0
                        }
                    
                    logger.info(f"📊 Loom video processing result: {result}")
                    
                    if result and result.get('success'):
                        chunks_stored = result.get('chunks_stored', 0)
                        total_chunks += chunks_stored
                        successful_content["videos"].append({
                            "url": video_url,
                            "type": "loom",
                            "chunks_stored": chunks_stored
                        })
                        logger.info(f"✅ Loom video processed: {chunks_stored} chunks stored")
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No result'
                        processing_errors.append({
                            "type": "video",
                            "url": video_url,
                            "error": error_msg
                        })
                        logger.error(f"❌ Loom video processing failed: {error_msg}")
                        
                    # Add delay between Loom videos to prevent conflicts
                    if i < len(loom_videos) - 1:
                        logger.info("⏳ Waiting 5s before processing next Loom video...")
                        import time
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing Loom video {video_url}: {e}")
                    continue
        
        # Step 2: Process website (may take longer)
        if request.video_urls and len(request.video_urls) > 0:
            logger.info("✅ Step 1 (Videos) completed successfully!")
            
        if request.website_urls and len(request.website_urls) > 0:
            logger.info(f"🌐 Step 2: Processing {len(request.website_urls)} websites")
            processing_order.append("websites")
            
            websites_processed = 0
            websites_failed = 0
            
            for i, website_url in enumerate(request.website_urls):
                try:
                    logger.info(f"🌐 Processing website {i+1}/{len(request.website_urls)}: {website_url}")
                    
                    # Use the new website scraper
                    if website_scraper:
                        website_data = await website_scraper.scrape_website(website_url)
                        
                        if website_data and website_data.get('scraped_pages'):
                            # Store website content using GCS service
                            storage_success = gcs_qa_service.store_website_content(company_name, qudemo_id, website_data)
                            
                            if storage_success:
                                total_chunks += website_data.get('total_pages', 0)
                                successful_content["websites"].append({
                                    "url": website_url,
                                    "pages_scraped": website_data.get('total_pages', 0),
                                    "status": website_data.get('scraping_status', 'completed')
                                })
                                websites_processed += 1
                                logger.info(f"✅ Website {i+1} processed successfully: {website_data.get('total_pages', 0)} pages")
                            else:
                                logger.error(f"❌ Failed to store website {i+1} content")
                                processing_errors.append({
                                    "type": "website",
                                    "url": website_url,
                                    "error": "Failed to store website content",
                                    "error_type": "storage_error"
                                })
                                websites_failed += 1
                        else:
                            logger.warning(f"⚠️ No content scraped from website {i+1}: {website_url}")
                            
                            # Check the actual scraping errors to determine the failure type
                            scraping_errors = website_data.get('errors', []) if website_data else []
                            error_details = website_data.get('analysis', {}) if website_data else {}
                            
                            # Determine error type based on actual scraping results, not URL patterns
                            if scraping_errors:
                                # Check if any error indicates bot detection
                                bot_detection_errors = [error for error in scraping_errors if 
                                    'bot detection' in error.lower() or 
                                    'captcha' in error.lower() or
                                    'cloudflare' in error.lower() or
                                    'challenge' in error.lower()]
                                
                                if bot_detection_errors:
                                    error_msg = "Site has bot protection that prevents scraping - try uploading documents instead"
                                    error_type = "crm_bot_detection"
                                    logger.info(f"🌐 Bot detection confirmed from scraping errors: {website_url}")
                                else:
                                    error_msg = f"Scraping failed: {', '.join(scraping_errors[:2])}"  # Show first 2 errors
                                    error_type = "scraping_error"
                                    logger.info(f"🌐 Scraping failed for other reasons: {website_url}")
                            else:
                                # No specific errors, generic failure message
                                error_msg = "No content could be scraped - site may have bot protection or other restrictions"
                                error_type = "scraping_error"
                                logger.info(f"🌐 Generic scraping failure: {website_url}")
                            
                            processing_errors.append({
                                "type": "website",
                                "url": website_url,
                                "error": error_msg,
                                "error_type": error_type
                            })
                            websites_failed += 1
                    else:
                        logger.error("❌ Website scraper not initialized")
                        processing_errors.append({
                            "type": "website",
                            "url": website_url,
                            "error": "Website scraper not available",
                            "error_type": "service_error"
                        })
                        websites_failed += 1
                        
                except Exception as e:
                    logger.error(f"❌ Website {i+1} processing error: {e}")
                    processing_errors.append({
                        "type": "website",
                        "url": website_url,
                        "error": str(e),
                        "error_type": "processing_error"
                    })
                    websites_failed += 1
            
            # Final website processing summary
            if websites_processed > 0:
                logger.info(f"✅ Step 2 (Websites) completed: {websites_processed}/{len(request.website_urls)} websites processed successfully")
            if websites_failed > 0:
                logger.info(f"⚠️ Step 2 (Websites) partial failure: {websites_failed}/{len(request.website_urls)} websites failed")
        
        if total_chunks > 0:
            logger.info(f"🎉 Processing completed! Total chunks stored: {total_chunks}")
        else:
            logger.info("❌ No content could be processed - all sources failed")
        
        # Notify Node.js backend that processing is complete
        try:
            logger.info("🔄 Notifying Node.js backend of processing completion...")
            
            # Get Node.js backend URL from environment
            node_backend_url = os.getenv('NODE_BACKEND_URL', 'http://localhost:5000')
            
            # Set a timeout for the entire notification process
            import signal
            import threading
            
            # Use threading timeout instead of signal (Windows compatible)
            notification_timeout = 10  # 10 seconds timeout
            
            # Prepare notification data - only include successful content
            notification_data = {
                'qudemo_id': qudemo_id,
                'company_name': company_name,
                'success': True,  # ⚠️ CRITICAL: Node.js backend checks this field!
                'processing_complete': True,
                'total_chunks_stored': total_chunks,
                'videos': [video['url'] for video in successful_content['videos']],
                'websites': [website['url'] for website in successful_content['websites']],
                'videos_processed': len(successful_content['videos']),
                'website_processed': len(successful_content['websites']),
                'documents_processed': 0,  # Add this for completeness
                'processing_order': processing_order,
                # Add error information for the frontend
                'processing_errors': processing_errors,
                'has_errors': len(processing_errors) > 0,
                'has_anti_bot_protection': any(error.get('protection_detected', False) for error in processing_errors)
            }
            
            # Send notification to Node.js backend
            import requests
            
            # Try multiple possible endpoints
            endpoints_to_try = [
                f"{node_backend_url}/api/qudemos/{qudemo_id}/processing-complete",
                f"{node_backend_url}/api/qudemos/{qudemo_id}/complete",
                f"{node_backend_url}/api/processing-complete/{qudemo_id}",
                f"{node_backend_url}/api/complete/{qudemo_id}"
            ]
            
            notification_success = False
            
            def try_notification():
                nonlocal notification_success
                for endpoint in endpoints_to_try:
                    try:
                        notification_response = requests.post(
                            endpoint,
                            json=notification_data,
                            timeout=5  # Very short timeout to prevent hanging
                        )
                        
                        if notification_response.status_code == 200:
                            logger.info(f"✅ Successfully notified Node.js backend at: {endpoint}")
                            notification_success = True
                            return
                        else:
                            logger.debug(f"🔍 Endpoint {endpoint} returned {notification_response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        logger.debug(f"🔍 Endpoint {endpoint} failed: {e}")
                        continue
            
            # Use threading timeout for Windows compatibility
            notification_thread = threading.Thread(target=try_notification)
            notification_thread.daemon = True
            notification_thread.start()
            notification_thread.join(timeout=notification_timeout)
            
            if notification_thread.is_alive():
                logger.info("ℹ️ Notification timeout - continuing without notification")
            
            if not notification_success:
                logger.info("ℹ️ Node.js backend notification skipped - endpoint not available or backend not running")
                logger.info("ℹ️ This is normal if Node.js backend is not running or endpoint doesn't exist")
                
        except Exception as e:
            logger.info(f"ℹ️ Node.js backend notification skipped: {e}")
            # Don't fail the entire request if notification fails
        
        # Check if any content was successfully processed
        if total_chunks == 0:
            logger.error("❌ No content could be processed - QuDemo should not be created")
            raise HTTPException(
                status_code=400, 
                detail={
                    'success': False,
                    'message': "No content could be processed. All sources failed due to restrictions or errors.",
                    'total_chunks_stored': 0,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'processing_order': processing_order,
                    'optimization_note': "All content sources failed to process",
                    # Enhanced status information
                    'successful_content': successful_content,
                    'processing_errors': processing_errors,
                    'has_errors': True,
                    'has_anti_bot_protection': any(error.get('protection_detected', False) for error in processing_errors),
                    # Add the structure that Node.js backend expects
                    'videos': [],
                    'websites': [],
                    'videos_processed': 0,
                    'website_processed': 0
                }
            )
        
        # Generate suggested questions after successful processing
        try:
            if total_chunks > 0:  # Only generate if we have processed content
                logger.info(f"🤖 Generating suggested questions for {company_name}/{qudemo_id}")
                suggested_questions = gcs_qa_service.generate_and_store_suggested_questions(company_name, qudemo_id)
                logger.info(f"✅ Generated {len(suggested_questions)} suggested questions")
            else:
                logger.info("⏭️ Skipping suggested questions generation - no content processed")
                suggested_questions = []
        except Exception as e:
            logger.error(f"❌ Error generating suggested questions: {e}")
            suggested_questions = []
        
        # Generate user-friendly status message
        status_message = _generate_processing_status_message(successful_content, processing_errors, total_chunks)
        
        # Debug: Log processing errors
        logger.info(f"🔍 Processing errors to return: {processing_errors}")
        logger.info(f"🔍 Has errors: {len(processing_errors) > 0}")
        logger.info(f"🔍 Has anti-bot protection: {any(error.get('protection_detected', False) for error in processing_errors)}")
        
        return {
            'success': True,
            'message': status_message,
            'total_chunks_stored': total_chunks,
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'processing_order': processing_order,
            'optimization_note': "Videos processed first for faster results, website processed second",
            # Enhanced status information
            'successful_content': successful_content,
            'processing_errors': processing_errors,
            'has_errors': len(processing_errors) > 0,
            'has_anti_bot_protection': any(error.get('protection_detected', False) for error in processing_errors),
            # Add the structure that Node.js backend expects
            'videos': [video['url'] for video in successful_content['videos']],
            'websites': [website['url'] for website in successful_content['websites']],
            'videos_processed': len(successful_content['videos']),
            'website_processed': len(successful_content['websites']),
            'suggested_questions_generated': len(suggested_questions)
        }
            
    except Exception as e:
        logger.error(f"❌ Error processing qudemo content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gcs/status")
async def get_gcs_status():
    """Get Google Cloud Storage connection and bucket status"""
    try:
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not initialized")
        
        # Get GCS service status
        gcs_service = gcs_qa_service.gcs_service
        companies = gcs_service.list_companies()
        
        return {
            'success': True,
            'status': {
                'gcs_connected': True,
                'companies_count': len(companies),
                'companies': companies,
                'bucket_type': 'company-specific buckets'
            },
            'timestamp': datetime.now().isoformat()
        }
            
    except Exception as e:
        logger.error(f"❌ Error getting GCS status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-faqs/{company_name}/{qudemo_id}")
async def manually_generate_faqs(company_name: str, qudemo_id: str):
    """Manually generate FAQs for an existing QuDemo (useful for QuDemos created before FAQ feature)"""
    try:
        logger.info(f"🔨 Manually generating FAQs for: {company_name}/{qudemo_id}")
        
        # Generate FAQs (function fetches presenter photo internally)
        result = await generate_faq_for_avatar_videos(
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if result:
            return {
                "success": True,
                "message": f"Generated {len(result.get('faqs', []))} FAQs successfully",
                "qudemo_id": qudemo_id,
                "company_name": company_name,
                "faq_count": len(result.get('faqs', [])),
                "faqs": result.get('faqs', [])
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate FAQs"
            }
            
    except Exception as e:
        logger.error(f"❌ Error manually generating FAQs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup-qudemo/{company_name}/{qudemo_id}")
async def cleanup_qudemo_data(company_name: str, qudemo_id: str):
    """Clean up all GCS data for a specific QuDemo (replaces Pinecone cleanup)"""
    try:
        logger.info(f"🧹 Starting GCS cleanup for QuDemo: {qudemo_id} in company: {company_name}")
        
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not initialized")
        
        # Use GCS Q&A service to delete QuDemo data
        deletion_success = gcs_qa_service.delete_qudemo(company_name, qudemo_id)
        
        if deletion_success:
            logger.info(f"✅ GCS cleanup successful for {company_name}/{qudemo_id}")
            return {
                "success": True,
                "message": f"GCS data cleanup completed for {company_name}/{qudemo_id}",
                "deleted_files": "All QuDemo files deleted from GCS",
                "company_name": company_name,
                "qudemo_id": qudemo_id
            }
        else:
            logger.error(f"❌ GCS cleanup failed for {company_name}/{qudemo_id}")
            return {
                "success": False,
                "error": "Failed to delete QuDemo data from GCS",
                "company_name": company_name,
                "qudemo_id": qudemo_id
            }
        
    except Exception as e:
        logger.error(f"❌ Error in GCS QuDemo cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup-all-qudemo-data/{company_name}/{qudemo_id}")
async def cleanup_all_qudemo_data(company_name: str, qudemo_id: str):
    """
    Clean up ALL data for a specific QuDemo from GCS
    This is a comprehensive cleanup that handles GCS storage system
    """
    try:
        logger.info(f"🧹 Starting comprehensive GCS cleanup for QuDemo: {qudemo_id} in company: {company_name}")
        
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not initialized")
        
        # Clean up GCS data
        deletion_success = gcs_qa_service.delete_qudemo(company_name, qudemo_id)
        
        if deletion_success:
            logger.info(f"✅ Comprehensive GCS cleanup successful for {company_name}/{qudemo_id}")
            return {
                "success": True,
                "message": f"Comprehensive GCS cleanup completed for {company_name}/{qudemo_id}",
                "data": {
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'gcs_cleanup': {'success': True, 'message': 'GCS data deleted successfully'},
                    'overall_success': True
                }
            }
        else:
            logger.error(f"❌ Comprehensive GCS cleanup failed for {company_name}/{qudemo_id}")
            return {
                "success": False,
                "error": "Failed to delete QuDemo data from GCS",
                "data": {
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'gcs_cleanup': {'success': False, 'message': 'GCS deletion returned false'},
                    'overall_success': False
                }
            }
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive GCS QuDemo cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup all QuDemo data: {str(e)}")

@app.delete("/delete-company-data/{company_name}")
async def delete_company_data(company_name: str):
    """Delete ALL GCS data for a company (all QuDemos, all buckets)"""
    try:
        logger.info(f"🗑️ Starting complete GCS company deletion for: {company_name}")
        
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not initialized")
        
        # Use GCS service to delete company bucket
        gcs_service = gcs_qa_service.gcs_service
        deletion_success = gcs_service.delete_company_bucket(company_name)
        
        if deletion_success:
            logger.info(f"✅ GCS company deletion successful for {company_name}")
            return {
                "success": True,
                "message": f"GCS company data deletion completed for {company_name}",
                "data": {
                    'company_name': company_name,
                    'deleted_bucket': f"qudemo-{company_name.lower().replace(' ', '-')}",
                    'deleted_files': "All company files and QuDemos deleted from GCS"
                }
            }
        else:
            logger.error(f"❌ GCS company deletion failed for {company_name}")
            return {
                "success": False,
                "error": "Failed to delete company data from GCS",
                "data": {
                    'company_name': company_name,
                    'deleted_bucket': None,
                    'deleted_files': 0
                }
            }
        
    except Exception as e:
        logger.error(f"❌ Error in GCS company data deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Universal Scraper Endpoints (disabled - module deleted)

# Request models moved to top of file

# ============================================================================
# HELPER FUNCTIONS FOR AVATAR VIDEO FEATURE
# ============================================================================

async def generate_faq_for_avatar_videos(company_name: str, qudemo_id: str):
    """Generate FAQ pairs from documents for avatar video generation"""
    try:
        logger.info(f"🤖 Checking if FAQ generation needed for {company_name}/{qudemo_id}")
        
        # Check if presenter photo exists (query Supabase directly)
        presenter_photo_url = None
        presenter_name = None
        
        try:
            # Get QuDemo details from Supabase
            from supabase import create_client
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            if not supabase_url or not supabase_key:
                logger.error(f"❌ Supabase credentials not found in .env")
                return
            
            supabase = create_client(supabase_url, supabase_key)
            response = supabase.table('qudemos_new').select('presenter_photo_url, presenter_name').eq('id', qudemo_id).single().execute()
            
            if response.data:
                presenter_photo_url = response.data.get('presenter_photo_url')
                presenter_name = response.data.get('presenter_name') or 'Presenter'
                logger.info(f"📸 Presenter photo: {presenter_photo_url}")
                logger.info(f"👤 Presenter name: {presenter_name}")
            
            if not presenter_photo_url:
                logger.info(f"ℹ️ No presenter photo found - skipping FAQ generation")
                return
            
            logger.info(f"✅ Presenter photo found: {presenter_photo_url}")
            logger.info(f"🎬 Generating FAQs for avatar videos...")
            
        except Exception as e:
            logger.error(f"❌ Error checking presenter photo: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return
        
        # Get all content (documents + video transcripts)
        all_faqs = []
        
        # 1. Generate FAQs from VIDEO TRANSCRIPTS
        logger.info(f"📹 Generating FAQs from video transcripts...")
        video_faqs = []
        
        if gcs_qa_service:
            try:
                transcript_data = gcs_qa_service.gcs_service.get_video_transcript(company_name, qudemo_id)
                
                if transcript_data and 'videos' in transcript_data and len(transcript_data['videos']) > 0:
                    # Get chunks/segments from the first video
                    video_data = transcript_data['videos'][0]
                    chunks = video_data.get('chunks', [])
                    segments = video_data.get('segments', [])
                    
                    # Use chunks (YouTube/Gemini) or segments (Loom/Whisper)
                    if chunks:
                        # Combine all transcript chunks (YouTube/Gemini format)
                        transcript_text = "\n\n".join([
                            f"[{chunk.get('start_time', '')}-{chunk.get('end_time', '')}] {chunk.get('text', '')}"
                            for chunk in chunks
                        ])
                        logger.info(f"📹 Using chunks from YouTube/Gemini transcript ({len(chunks)} chunks)")
                    elif segments:
                        # Combine all transcript segments (Loom/Whisper format)
                        transcript_text = "\n\n".join([
                            f"[{seg.get('formatted_start', '')}-{seg.get('formatted_end', '')}] {seg.get('text', '')}"
                            for seg in segments
                        ])
                        logger.info(f"📹 Using segments from Loom/Whisper transcript ({len(segments)} segments)")
                    else:
                        transcript_text = ""
                    
                    if transcript_text:
                        transcript_text = transcript_text[:10000]  # Limit to 10k chars to avoid token limits
                        
                        logger.info(f"🎥 Video transcript length: {len(transcript_text)} characters")
                        
                        # STEP 1: Identify all distinct topics/concepts (for comprehensive coverage)
                        logger.info(f"🔍 Step 1: Identifying all topics in video...")
                        
                        topics_prompt = f"""Identify high-level topics from this transcript for customer FAQs.

Prioritize: Product uniqueness, target audience, value proposition, benefits, time/cost savings.
Then: Key features, pricing, technical specs.
Max 10 topics.

Video transcript:
{transcript_text}

Return JSON: [{{"topic": "description", "importance": "high/medium"}}]"""

                        try:
                            topics_response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": topics_prompt}],
                                temperature=0.3,
                                max_tokens=500
                            )
                            
                            topics_json = topics_response.choices[0].message.content.strip()
                            if topics_json.startswith("```json"):
                                topics_json = topics_json.replace("```json", "").replace("```", "").strip()
                            
                            topics = json.loads(topics_json)
                            logger.info(f"✅ Identified {len(topics)} topics")
                            for i, topic in enumerate(topics, 1):
                                logger.info(f"   Topic {i}: {topic.get('topic', 'Unknown')} ({topic.get('importance', 'medium')} importance)")
                            
                        except Exception as topic_error:
                            logger.error(f"❌ Error identifying topics: {topic_error}")
                            topics = []
                        
                        # STEP 2: Generate comprehensive FAQs covering all topics
                        logger.info(f"🤖 Step 2: Generating FAQs to cover all {len(topics)} topics...")
                        
                        topics_list = "\n".join([f"- {t.get('topic', '')}" for t in topics])
                        
                        prompt = f"""Generate FAQs from this transcript covering these topics:
{topics_list}

Questions: Short (5-10 words), simple, no compound questions. Start with "What is/How does/Why/Who".
Answers: Clear, concise (100-150 words max 1000 chars), focus on benefits. Use bullet points for key features.

Create 1 unique FAQ per topic. Synthesize, don't copy transcript.

Video transcript:
{transcript_text}

Return JSON only:
[{{"question": "...", "answer": "...", "category": "video", "source": "video_transcript"}}]"""

                        try:
                            if not openai_client:
                                logger.error("❌ OpenAI client not initialized")
                                video_faqs = []
                            else:
                                response = openai_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.7,
                                    max_tokens=2500
                                )
                                
                                faq_json = response.choices[0].message.content.strip()
                                
                                if faq_json.startswith("```json"):
                                    faq_json = faq_json.replace("```json", "").replace("```", "").strip()
                                
                                # Fix: Use strict=False to handle control characters in JSON
                                try:
                                    video_faqs = json.loads(faq_json, strict=False)
                                except json.JSONDecodeError as json_error:
                                    # If strict=False doesn't work, try to clean the JSON string
                                    logger.warning(f"⚠️ JSON parsing failed, attempting to clean: {json_error}")
                                    # Replace common control characters
                                    faq_json_cleaned = faq_json.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                    video_faqs = json.loads(faq_json_cleaned, strict=False)
                                
                                logger.info(f"✅ Generated {len(video_faqs)} FAQ pairs from video transcript")
                                
                                # Log the generated FAQs
                                logger.info(f"📋 Generated Video FAQs:")
                                for i, faq in enumerate(video_faqs, 1):
                                    q = faq.get('question', 'N/A')
                                    a = faq.get('answer', 'N/A')
                                    logger.info(f"   FAQ {i}:")
                                    logger.info(f"   Q: {q}")
                                    logger.info(f"   A: {a[:100]}... ({len(a)} chars)")
                            
                        except Exception as gpt_error:
                            logger.error(f"❌ Error calling GPT-4 for video FAQs: {gpt_error}")
                            video_faqs = []
                    else:
                        logger.info(f"ℹ️ No transcript text available (no chunks or segments found)")
                else:
                    logger.info(f"ℹ️ No video transcript found")
                    
            except Exception as e:
                logger.error(f"❌ Error processing video transcript: {e}")
                video_faqs = []
        
        all_faqs.extend(video_faqs)
        
        # 2. Generate FAQs from DOCUMENTS
        logger.info(f"📄 Generating FAQs from documents...")
        document_faqs = []
        
        if document_processor:
            all_documents = document_processor.search_document_content(
                company_name=company_name,
                qudemo_id=qudemo_id,
                query=""  # Empty query returns all content
            )
            
            if all_documents:
                # Combine all document content
                combined_content = "\n\n".join([doc.get('content', '') for doc in all_documents])
                combined_content = combined_content[:10000]  # Limit to 10k chars to avoid token limits
                
                logger.info(f"📄 Combined document content length: {len(combined_content)} characters")
                
                # STEP 1: Identify all distinct topics/concepts in documents
                logger.info(f"🔍 Step 1: Identifying all topics in documents...")
                
                doc_topics_prompt = f"""Identify high-level topics from this document for customer FAQs.

Prioritize: Product uniqueness, target audience, value proposition, benefits, time/cost savings.
Then: Key features, pricing, use cases.
Max 10 topics.

Document content:
{combined_content}

Return JSON: [{{"topic": "description", "importance": "high/medium"}}]"""

                try:
                    doc_topics_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": doc_topics_prompt}],
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    doc_topics_json = doc_topics_response.choices[0].message.content.strip()
                    if doc_topics_json.startswith("```json"):
                        doc_topics_json = doc_topics_json.replace("```json", "").replace("```", "").strip()
                    
                    doc_topics = json.loads(doc_topics_json)
                    logger.info(f"✅ Identified {len(doc_topics)} topics in documents")
                    for i, topic in enumerate(doc_topics, 1):
                        logger.info(f"   Topic {i}: {topic.get('topic', 'Unknown')} ({topic.get('importance', 'medium')} importance)")
                    
                except Exception as doc_topic_error:
                    logger.error(f"❌ Error identifying document topics: {doc_topic_error}")
                    doc_topics = []
                
                # STEP 2: Generate comprehensive FAQs covering all document topics
                logger.info(f"🤖 Step 2: Generating FAQs to cover all {len(doc_topics)} document topics...")
                
                doc_topics_list = "\n".join([f"- {t.get('topic', '')}" for t in doc_topics])
                
                prompt = f"""Generate FAQs from this document covering these topics:
{doc_topics_list}

Questions: Short (5-10 words), simple, no compound questions. Start with "What is/How does/Why/Who".
Answers: Clear, concise (100-150 words max 1000 chars), focus on benefits. Use bullet points for key features.

Create 1 unique FAQ per topic. Synthesize, don't copy document text.

Document content:
{combined_content}

Return JSON only:
[{{"question": "...", "answer": "...", "category": "features", "source": "document"}}]"""

                try:
                    if not openai_client:
                        logger.error("❌ OpenAI client not initialized")
                        document_faqs = []
                    else:
                        response = openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=2500
                        )
                        
                        faq_json = response.choices[0].message.content.strip()
                        
                        if faq_json.startswith("```json"):
                            faq_json = faq_json.replace("```json", "").replace("```", "").strip()
                        
                        # Fix: Use strict=False to handle control characters in JSON
                        try:
                            document_faqs = json.loads(faq_json, strict=False)
                        except json.JSONDecodeError as json_error:
                            # If strict=False doesn't work, try to clean the JSON string
                            logger.warning(f"⚠️ JSON parsing failed, attempting to clean: {json_error}")
                            # Replace common control characters
                            faq_json_cleaned = faq_json.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                            document_faqs = json.loads(faq_json_cleaned, strict=False)
                        
                        logger.info(f"✅ Generated {len(document_faqs)} FAQ pairs from documents")
                        
                        # Log the generated FAQs
                        logger.info(f"📋 Generated Document FAQs:")
                        for i, faq in enumerate(document_faqs, 1):
                            q = faq.get('question', 'N/A')
                            a = faq.get('answer', 'N/A')
                            logger.info(f"   FAQ {i}:")
                            logger.info(f"   Q: {q}")
                            logger.info(f"   A: {a[:100]}... ({len(a)} chars)")
                    
                except Exception as gpt_error:
                    logger.error(f"❌ Error calling GPT-4 for document FAQs: {gpt_error}")
                    document_faqs = []
            else:
                logger.info(f"ℹ️ No documents found")
        else:
            logger.info(f"ℹ️ Document processor not available")
        
        all_faqs.extend(document_faqs)
        
        # 3. Generate FAQs from SUGGESTED QUESTIONS (with cached answers)
        logger.info(f"🎯 Generating FAQs from suggested questions...")
        suggested_faqs = []
        
        if gcs_qa_service:
            try:
                # Get suggested questions from GCS
                suggested_data = gcs_qa_service.gcs_service.get_suggested_questions_with_metadata(company_name, qudemo_id)
                
                if suggested_data and 'video_questions' in suggested_data:
                    logger.info(f"📥 Found suggested questions in GCS")
                    
                    # Extract all questions with answers
                    for video_data in suggested_data['video_questions']:
                        questions_with_answers = video_data.get('questions_with_answers', [])
                        
                        for qa in questions_with_answers:
                            question = qa.get('question', '')
                            answer = qa.get('answer', '')
                            
                            if question and answer:
                                # Limit answer to 1000 characters
                                if len(answer) > 1000:
                                    # Try to end at a sentence
                                    truncated = answer[:1000]
                                    last_period = truncated.rfind('.')
                                    if last_period > 900:  # Only use if we're close to 1000
                                        answer = truncated[:last_period + 1]
                                    else:
                                        answer = truncated + "..."
                                
                                suggested_faqs.append({
                                    "question": question,
                                    "answer": answer,
                                    "category": "suggested_question",
                                    "source": "video_suggested",
                                    "video_url": qa.get('video_url', ''),
                                    "timestamp": qa.get('timestamp', 0)
                                })
                    
                    logger.info(f"✅ Generated {len(suggested_faqs)} FAQs from suggested questions")
                    
                    # Log a few examples
                    logger.info(f"📋 Sample Suggested Question FAQs:")
                    for i, faq in enumerate(suggested_faqs[:3], 1):
                        q = faq.get('question', 'N/A')
                        a = faq.get('answer', 'N/A')
                        logger.info(f"   FAQ {i}:")
                        logger.info(f"   Q: {q}")
                        logger.info(f"   A: {a[:100]}... ({len(a)} chars)")
                else:
                    logger.info(f"ℹ️ No suggested questions found in GCS")
                    
            except Exception as e:
                logger.error(f"❌ Error processing suggested questions: {e}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                suggested_faqs = []
        else:
            logger.info(f"ℹ️ GCS QA service not available")
        
        all_faqs.extend(suggested_faqs)
        
        logger.info(f"📊 Total FAQs generated BEFORE DEDUPLICATION: {len(all_faqs)} (Videos: {len(video_faqs)}, Documents: {len(document_faqs)}, Suggested: {len(suggested_faqs)})")
        
        # DEDUPLICATION: Remove duplicate or very similar questions
        unique_faqs = []
        seen_questions = set()
        
        for faq in all_faqs:
            question = faq.get('question', '').strip().lower()
            # Normalize question for comparison
            normalized_question = question.replace('?', '').replace('.', '').strip()
            
            # Check if we've seen a very similar question
            is_duplicate = False
            for seen_q in seen_questions:
                # Check for exact match or very high similarity
                if normalized_question == seen_q or normalized_question in seen_q or seen_q in normalized_question:
                    is_duplicate = True
                    logger.warning(f"⚠️ Duplicate question detected: '{question[:50]}...'")
                    break
            
            if not is_duplicate:
                unique_faqs.append(faq)
                seen_questions.add(normalized_question)
                logger.info(f"✅ Added unique question: '{question[:50]}...'")
            else:
                logger.warning(f"❌ Skipped duplicate: '{question[:50]}...'")
        
        all_faqs = unique_faqs
        logger.info(f"📊 Total FAQs AFTER DEDUPLICATION: {len(all_faqs)} unique FAQs")
        
        # ⚠️ LIMIT: Cap at 2 content FAQs (+ 3 special = 5 total) - TEMPORARY FOR TESTING
        MAX_CONTENT_FAQS = 2  # Reduced from 7 to 2 for testing (saves HeyGen credits)
        if len(all_faqs) > MAX_CONTENT_FAQS:
            logger.warning(f"⚠️ Limiting FAQs from {len(all_faqs)} to {MAX_CONTENT_FAQS}")
            all_faqs = all_faqs[:MAX_CONTENT_FAQS]
        
        logger.info(f"📊 Total FAQs AFTER LIMIT: {len(all_faqs)} content FAQs (will add 1 intro + 2 fallback FAQs = {len(all_faqs) + 3} total = max 5 videos) [TESTING MODE]")
        
        # Store FAQs in GCS (regardless of document availability)
        if gcs_qa_service:
            bucket_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            bucket = gcs_qa_service.gcs_service.client.bucket(bucket_name)
            
            # Add default fallback FAQs for common scenarios
            default_faqs = [
                {
                    "id": "faq_intro",
                    "question": "INTRO_VIDEO",
                    "answer": f"Welcome! I'm here to guide you through this interactive demo. I'll be answering your questions and showing you everything you need to know. Feel free to ask me anything about our product, features, or how we can help solve your challenges. Let's get started!",
                    "category": "intro",
                    "estimated_duration": 15.0,
                    "is_intro": True
                },
                {
                    "id": "faq_fallback_no_answer",
                    "question": "NO_ANSWER_FOUND",
                    "answer": "I apologize, but I don't have specific information about that in our knowledge base. However, I'd be happy to connect you with our team who can help answer your questions in detail. Please use the 'Book a Meeting' option below to schedule a call with our experts.",
                    "category": "fallback",
                    "estimated_duration": 15.0,
                    "is_fallback": True
                },
                {
                    "id": "faq_fallback_sales",
                    "question": "SALES_INQUIRY",
                    "answer": "I'd be delighted to connect you with our sales team! They're experts at understanding your specific needs and can provide personalized guidance. Please click on the 'Book a Meeting' button below to schedule a convenient time to chat with one of our team members. We look forward to speaking with you!",
                    "category": "fallback",
                    "estimated_duration": 18.0,
                    "is_fallback": True
                }
            ]
            
            faq_data = {
                "version": "1.0",
                "qudemo_id": qudemo_id,
                "company_name": company_name,
                "presenter_name": presenter_name,
                "presenter_photo_url": presenter_photo_url,
                "generated_at": datetime.now().isoformat(),
                "faqs": [
                    {
                        "id": f"faq_{str(i+1).zfill(3)}",
                        "question": faq["question"],
                        "answer": faq["answer"],
                        "category": faq.get("category", "general"),
                        "source": faq.get("source", "unknown"),
                        "estimated_duration": len(faq["answer"].split()) * 0.4  # ~0.4 seconds per word
                    }
                    for i, faq in enumerate(all_faqs)
                ] + default_faqs  # Add default fallback FAQs
            }
            
            # Store in GCS
            faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
            blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
            blob.upload_from_string(json.dumps(faq_data, indent=2), content_type='application/json')
            logger.info(f"✅ Stored {len(faq_data['faqs'])} FAQs in GCS: {blob.name}")
            logger.info(f"   - Content FAQs (after limit): {len(all_faqs)}")
            logger.info(f"     • Video FAQs: {len(video_faqs)}")
            logger.info(f"     • Document FAQs: {len(document_faqs)}")
            logger.info(f"   - Special FAQs: {len(default_faqs)} (1 intro + 2 fallback)")
            logger.info(f"   💰 FAQ Limit: {MAX_CONTENT_FAQS} content + {len(default_faqs)} special = {len(faq_data['faqs'])} TOTAL VIDEOS")
            logger.info(f"   💰 HeyGen Credits: {len(faq_data['faqs'])} videos will be generated!")
            
            # ⚠️ OPTIONAL: Save FAQs to local JSON file for review (useful for debugging)
            try:
                local_faq_file = f"faq_test_{company_name.replace(' ', '_')}_{qudemo_id[:8]}.json"
                with open(local_faq_file, 'w', encoding='utf-8') as f:
                    json.dump(faq_data, f, indent=2, ensure_ascii=False)
                logger.info(f"📄 Saved FAQs to local file for review: {local_faq_file}")
            except Exception as save_error:
                logger.error(f"❌ Error saving local FAQ file: {save_error}")
            
            # Generate avatar videos using HeyGen (background task)
            if avatar_video_processor and presenter_photo_url:
                logger.info(f"🎬 Starting avatar video generation for {len(faq_data['faqs'])} FAQs...")
                logger.info(f"⏱️ Estimated time: 3-5 minutes per video ({len(faq_data['faqs'])} videos total)")
                logger.info(f"🔄 Videos will be generated in the background")
                
                # Get voice_id from qudemo (if available)
                voice_id_to_use = None
                try:
                    qudemo_voice = supabase.table('qudemos_new').select('voice_id').eq('id', qudemo_id).execute()
                    if qudemo_voice.data and len(qudemo_voice.data) > 0:
                        voice_id_to_use = qudemo_voice.data[0].get('voice_id')
                        logger.info(f"🎤 Using voice ID from qudemo: {voice_id_to_use}")
                except Exception as voice_error:
                    logger.warning(f"⚠️ Could not fetch voice_id, using default: {voice_error}")
                
                # Run video generation in background (don't block the response)
                asyncio.create_task(
                    avatar_video_processor.process_faq_videos(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        presenter_photo_url=presenter_photo_url,
                        faqs=faq_data["faqs"],
                        voice_id=voice_id_to_use
                    )
                )
                
                logger.info(f"✅ Background video generation task started")
                logger.info(f"💰 HeyGen Credits: {len(faq_data['faqs'])} videos will be generated")
            else:
                if not avatar_video_processor:
                    logger.warning("⚠️ Avatar video processor not available - skipping video generation")
                if not presenter_photo_url:
                    logger.warning("⚠️ No presenter photo - skipping video generation")
            
            return faq_data
            
    except Exception as e:
        logger.error(f"❌ Error generating FAQs for avatar videos: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return None

def _generate_processing_status_message(successful_content: Dict, processing_errors: List[Dict], total_chunks: int) -> str:
    """Generate user-friendly processing status message"""
    messages = []
    
    # Add successful content summary
    if successful_content['videos']:
        video_count = len(successful_content['videos'])
        total_video_chunks = sum(video['chunks_stored'] for video in successful_content['videos'])
        messages.append(f"✅ {video_count} video(s) processed successfully ({total_video_chunks} chunks)")
    
    if successful_content['websites']:
        website_count = len(successful_content['websites'])
        total_website_chunks = sum(website['pages_scraped'] for website in successful_content['websites'])
        messages.append(f"✅ {website_count} website(s) scraped successfully ({total_website_chunks} chunks)")
    
    # Add error information
    if processing_errors:
        messages.append("\n⚠️ Processing Issues:")
        
        for error in processing_errors:
            if error['type'] == 'website':
                if error.get('protection_detected', False):
                    messages.append(f"🛡️ Website '{error['url']}' has anti-bot protection - scraping blocked")
                    messages.append(f"   Reason: {error['error']}")
                else:
                    messages.append(f"❌ Website '{error['url']}' scraping failed")
                    messages.append(f"   Reason: {error['error']}")
            elif error['type'] == 'video':
                messages.append(f"❌ Video '{error['url']}' processing failed")
                messages.append(f"   Reason: {error['error']}")
    
    # Add overall summary
    if total_chunks > 0:
        messages.append(f"\n📊 Total: {total_chunks} content chunks available for Q&A")
        
        if processing_errors:
            messages.append("💡 You can ask questions about the successfully processed content")
        else:
            messages.append("🎉 All content processed successfully!")
    else:
        messages.append("\n❌ No content could be processed")
        messages.append("🛡️ All content sources failed due to restrictions or errors")
        messages.append("💡 Please try different content sources or contact support")
        messages.append("⚠️ QuDemo will not be created without successful content")
    
    return "\n".join(messages)

# Debug endpoint to examine stored chunks in GCS
@app.get("/debug-chunks/{company_name}/{qudemo_id}")
async def debug_chunks(company_name: str, qudemo_id: str):
    """Debug endpoint to examine what chunks are stored for a qudemo in GCS"""
    try:
        if not gcs_qa_service:
            return {"success": False, "error": "GCS Q&A service not initialized"}
        
        # Get transcript data from GCS
        transcript_data = gcs_qa_service.gcs_service.get_video_transcript(
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if transcript_data:
            chunks = transcript_data.get('chunks', [])
            segments = transcript_data.get('segments', [])
            
            chunks_info = []
            for i, chunk in enumerate(chunks):
                chunks_info.append({
                    'chunk_index': i,
                    'text_preview': chunk.get('text', '')[:200] + '...' if chunk.get('text') else '',
                    'start_timestamp': chunk.get('start_timestamp', 0),
                    'end_timestamp': chunk.get('end_timestamp', 0),
                    'source': chunk.get('source', 'unknown'),
                    'title': chunk.get('title', 'Unknown')
                })
            
            return {
                "success": True,
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "total_chunks": len(chunks_info),
                "total_segments": len(segments),
                "chunks": chunks_info,
                "storage_type": "gcs"
            }
        else:
            return {
                "success": False,
                "error": "No transcript data found for this QuDemo",
                "company_name": company_name,
                "qudemo_id": qudemo_id
            }
        
    except Exception as e:
        logger.error(f"❌ Debug chunks error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/debug-gcs-structure")
async def debug_gcs_structure():
    """Debug endpoint to examine Google Cloud Storage structure"""
    try:
        if not gcs_qa_service:
            return {"success": False, "error": "GCS Q&A service not initialized"}

        structure = gcs_qa_service.gcs_service.get_storage_structure()
        companies = gcs_qa_service.gcs_service.list_companies()

        return {
            "success": True,
            "bucket_type": "company-specific buckets",
            "bucket_prefix": "qudemo-",
            "companies": companies,
            "structure": structure
        }

    except Exception as e:
        logger.error(f"❌ Debug GCS structure error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/delete-company-bucket")
async def delete_company_bucket(request: dict):
    """Delete entire company bucket and all its contents"""
    try:
        company_name = request.get("company_name")
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required")
        
        logger.info(f"🗑️ Deleting company bucket for: {company_name}")
        
        # Check if GCS service is available
        if not gcs_qa_service:
            error_msg = "GCS service not initialized"
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Delete the company bucket
        success = gcs_qa_service.gcs_service.delete_company_bucket(company_name)
        
        if success:
            logger.info(f"✅ Successfully deleted bucket for company: {company_name}")
            return {
                "success": True,
                "message": f"Successfully deleted bucket for company: {company_name}",
                "company_name": company_name
            }
        else:
            logger.error(f"❌ Failed to delete bucket for company: {company_name}")
            return {
                "success": False,
                "error": f"Failed to delete bucket for company: {company_name}",
                "company_name": company_name
            }
            
    except Exception as e:
        logger.error(f"❌ Delete company bucket error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/make-bucket-public/{company_name}")
async def make_bucket_public(company_name: str):
    """Make an existing GCS bucket publicly readable (one-time fix for existing buckets)"""
    try:
        logger.info(f"🌍 Making bucket public for company: {company_name}")
        
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        
        # Set IAM policy to make bucket publicly readable
        policy = bucket.get_iam_policy(requested_policy_version=3)
        policy.bindings.append({
            "role": "roles/storage.objectViewer",
            "members": {"allUsers"}
        })
        bucket.set_iam_policy(policy)
        
        logger.info(f"✅ Made bucket publicly readable: {bucket.name}")
        
        return {
            "success": True,
            "message": f"Bucket {bucket.name} is now publicly readable",
            "bucket_name": bucket.name
        }
        
    except Exception as e:
        logger.error(f"❌ Error making bucket public: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/make-avatar-videos-public/{company_name}/{qudemo_id}")
async def make_avatar_videos_public(company_name: str, qudemo_id: str):
    """Make all avatar videos for a QuDemo publicly accessible (fix for existing videos)"""
    try:
        logger.info(f"🌍 Making avatar videos public for {company_name}/{qudemo_id}")
        
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        
        # List all avatar videos for this QuDemo
        video_prefix = f"{company_name}/{qudemo_id}/avatar_videos/"
        blobs = list(bucket.list_blobs(prefix=video_prefix))
        
        if not blobs:
            logger.warning(f"⚠️ No avatar videos found at {video_prefix}")
            return {
                "success": False,
                "message": "No avatar videos found",
                "count": 0
            }
        
        logger.info(f"📹 Found {len(blobs)} avatar videos to make public")
        
        # Make each video public
        success_count = 0
        errors = []
        
        for blob in blobs:
            try:
                blob.make_public()
                logger.info(f"✅ Made public: {blob.name}")
                success_count += 1
            except Exception as e:
                error_msg = f"Failed to make {blob.name} public: {str(e)}"
                logger.error(f"❌ {error_msg}")
                errors.append(error_msg)
        
        logger.info(f"✅ Successfully made {success_count}/{len(blobs)} videos public")
        
        return {
            "success": True,
            "message": f"Made {success_count}/{len(blobs)} videos public",
            "total_videos": len(blobs),
            "success_count": success_count,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error making avatar videos public: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# FAQ MANAGEMENT ENDPOINTS
# ============================================

@app.get("/faqs/{company_name}/{qudemo_id}")
async def get_faqs(company_name: str, qudemo_id: str):
    """Get all FAQs for a QuDemo (for FAQ editor)"""
    try:
        logger.info(f"📋 Fetching FAQs for {company_name}/{qudemo_id}")
        
        if not gcs_qa_service:
            return {"success": False, "error": "GCS QA service not available"}
        
        # Get FAQs from GCS
        faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
        
        if not faqs_data:
            return {
                "success": False,
                "message": "No FAQs found",
                "faqs": []
            }
        
        # Extract the faqs array
        faqs = faqs_data.get('faqs', [])
        
        if not faqs:
            return {
                "success": False,
                "message": "No FAQs found",
                "faqs": []
            }
        
        # Get avatar video URLs from database
        for faq in faqs:
            faq_id = faq.get('id')
            if faq_id:
                try:
                    # Query avatar_videos table
                    response = supabase.table('avatar_videos').select('video_url, status').eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                    if response.data and len(response.data) > 0:
                        faq['video_url'] = response.data[0].get('video_url')
                        faq['video_status'] = response.data[0].get('status')
                except Exception as e:
                    logger.error(f"❌ Error fetching video for {faq_id}: {e}")
                    faq['video_url'] = None
                    faq['video_status'] = None
        
        return {
            "success": True,
            "faqs": faqs,
            "count": len(faqs)
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching FAQs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.patch("/faqs/{company_name}/{qudemo_id}/{faq_id}/question")
async def update_faq_question(company_name: str, qudemo_id: str, faq_id: str, request: Request):
    """Update FAQ question only (no video regeneration)"""
    try:
        body = await request.json()
        new_question = body.get('question', '').strip()
        
        if not new_question:
            return {"success": False, "error": "Question cannot be empty"}
        
        logger.info(f"✏️ Updating question for FAQ {faq_id}: {new_question}")
        
        # Get current FAQs from GCS
        if not gcs_qa_service:
            return {"success": False, "error": "GCS QA service not available"}
        
        faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
        
        if not faqs_data:
            return {"success": False, "error": "FAQs not found"}
        
        faqs = faqs_data.get('faqs', [])
        
        # Update the question
        updated = False
        for faq in faqs:
            if faq.get('id') == faq_id:
                faq['question'] = new_question
                updated = True
                break
        
        if not updated:
            return {"success": False, "error": f"FAQ {faq_id} not found"}
        
        # Save back to GCS
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
        blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
        
        faq_data = {
            "version": "1.0",
            "qudemo_id": qudemo_id,
            "company_name": company_name,
            "updated_at": datetime.now().isoformat(),
            "faqs": faqs
        }
        
        blob.upload_from_string(json.dumps(faq_data, indent=2), content_type='application/json')
        
        logger.info(f"✅ Updated question for FAQ {faq_id}")
        
        return {
            "success": True,
            "message": "Question updated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating FAQ question: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.patch("/faqs/{company_name}/{qudemo_id}/{faq_id}/answer")
async def update_faq_answer(company_name: str, qudemo_id: str, faq_id: str, request: Request):
    """Update FAQ answer and regenerate AI video"""
    try:
        body = await request.json()
        new_answer = body.get('answer', '').strip()
        
        if not new_answer:
            return {"success": False, "error": "Answer cannot be empty"}
        
        if len(new_answer) > 1000:
            return {"success": False, "error": "Answer too long (max 1000 characters)"}
        
        logger.info(f"✏️ Updating answer for FAQ {faq_id}: {new_answer[:100]}...")
        
        # Get current FAQs from GCS
        if not gcs_qa_service:
            return {"success": False, "error": "GCS QA service not available"}
        
        faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
        
        if not faqs_data:
            return {"success": False, "error": "FAQs not found"}
        
        faqs = faqs_data.get('faqs', [])
        
        # Update the answer
        updated_faq = None
        old_answer = None
        for faq in faqs:
            if faq.get('id') == faq_id:
                old_answer = faq.get('answer', '')
                faq['answer'] = new_answer
                updated_faq = faq
                logger.info(f"📝 FAQ {faq_id} answer updated:")
                logger.info(f"   Old: {old_answer[:100]}...")
                logger.info(f"   New: {new_answer[:100]}...")
                break
        
        if not updated_faq:
            return {"success": False, "error": f"FAQ {faq_id} not found"}
        
        # Save back to GCS
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
        blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
        
        faq_data = {
            "version": "1.0",
            "qudemo_id": qudemo_id,
            "company_name": company_name,
            "updated_at": datetime.now().isoformat(),
            "faqs": faqs
        }
        
        blob.upload_from_string(json.dumps(faq_data, indent=2), content_type='application/json')
        
        logger.info(f"✅ Updated answer for FAQ {faq_id} in GCS: {company_name}/{qudemo_id}/{faq_filename}")
        logger.info(f"📄 FAQ JSON saved with {len(faqs)} FAQs")
        
        # Get presenter photo URL from database
        try:
            response = supabase.table('qudemos_new').select('presenter_photo_url, presenter_name').eq('id', qudemo_id).execute()
            if response.data and len(response.data) > 0:
                presenter_photo_url = response.data[0].get('presenter_photo_url')
                presenter_name = response.data[0].get('presenter_name', 'Presenter')
            else:
                logger.warning(f"⚠️ No presenter photo found for QuDemo {qudemo_id}")
                return {
                    "success": True,
                    "message": "Answer updated, but no presenter photo found for video generation",
                    "video_regenerated": False
                }
        except Exception as e:
            logger.error(f"❌ Error fetching presenter photo: {e}")
            return {
                "success": True,
                "message": "Answer updated, but error fetching presenter photo",
                "video_regenerated": False
            }
        
        # Regenerate video in background
        if avatar_video_processor and presenter_photo_url:
            logger.info(f"🎬 Starting video regeneration for FAQ {faq_id}")
            
            asyncio.create_task(
                avatar_video_processor.process_single_faq_video_update(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    presenter_photo_url=presenter_photo_url,
                    faq=updated_faq
                )
            )
            
            return {
                "success": True,
                "message": "Answer updated and video regeneration started",
                "video_regenerated": True
            }
        else:
            return {
                "success": True,
                "message": "Answer updated, but video processor not available",
                "video_regenerated": False
            }
        
    except Exception as e:
        logger.error(f"❌ Error updating FAQ answer: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/faqs/{company_name}/{qudemo_id}/{faq_id}")
async def delete_faq(company_name: str, qudemo_id: str, faq_id: str):
    """Delete a specific FAQ and optionally its associated avatar video"""
    try:
        logger.info(f"🗑️ Deleting FAQ {faq_id} from {company_name}/{qudemo_id}")
        
        # Get current FAQs from GCS
        if not gcs_qa_service:
            return {"success": False, "error": "GCS QA service not available"}
        
        faqs_data = gcs_qa_service.get_faqs(company_name, qudemo_id)
        
        if not faqs_data:
            return {"success": False, "error": "FAQs not found"}
        
        faqs = faqs_data.get('faqs', [])
        
        # Find and remove the FAQ
        faq_found = False
        updated_faqs = []
        deleted_faq = None
        
        for faq in faqs:
            if faq.get('id') == faq_id:
                faq_found = True
                deleted_faq = faq
                logger.info(f"📝 Found FAQ to delete: {faq.get('question', 'Unknown')}")
            else:
                updated_faqs.append(faq)
        
        if not faq_found:
            return {"success": False, "error": f"FAQ {faq_id} not found"}
        
        # Save updated FAQs back to GCS
        gcs_service = GoogleCloudStorageService()
        bucket = gcs_service._get_company_bucket(company_name)
        faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
        blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
        
        faq_data = {
            "version": "1.0",
            "qudemo_id": qudemo_id,
            "company_name": company_name,
            "updated_at": datetime.now().isoformat(),
            "faqs": updated_faqs
        }
        
        blob.upload_from_string(json.dumps(faq_data, indent=2), content_type='application/json')
        
        logger.info(f"✅ Deleted FAQ {faq_id} from GCS. Remaining FAQs: {len(updated_faqs)}")
        
        # Try to delete associated avatar video from database
        video_deleted = False
        try:
            # Delete from avatar_videos table
            delete_response = supabase.table('avatar_videos').delete().eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
            
            if delete_response.data:
                video_deleted = True
                logger.info(f"✅ Deleted avatar video from database for FAQ {faq_id}")
            else:
                logger.info(f"ℹ️ No avatar video found in database for FAQ {faq_id}")
                
        except Exception as video_error:
            logger.warning(f"⚠️ Error deleting avatar video: {video_error}")
        
        return {
            "success": True,
            "message": f"FAQ deleted successfully",
            "deleted_faq": {
                "id": faq_id,
                "question": deleted_faq.get('question', ''),
                "answer": deleted_faq.get('answer', '')[:100] + "..." if len(deleted_faq.get('answer', '')) > 100 else deleted_faq.get('answer', '')
            },
            "remaining_faqs": len(updated_faqs),
            "video_deleted": video_deleted
        }
        
    except Exception as e:
        logger.error(f"❌ Error deleting FAQ: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
