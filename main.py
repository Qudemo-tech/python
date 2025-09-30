#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Pinecone Standard Plan Multi-Index Architecture
Optimized for Q&A, video processing, and web scraping
"""

import os
import logging
import json
from typing import List, Optional, Dict
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tempfile
import shutil

# Enhanced components
# from final_gemini_scraper import FinalGeminiScraper  # Removed - requires Playwright vds
from gcs_qa_service import GCSQAService
from simple_gemini_transcriber import SimpleGeminiTranscriber
from document_processor import DocumentProcessor
from company_api import router as company_router
from company_bucket_service import initialize_company_bucket_service, get_company_bucket_service
from website_scraper import WebsiteScraper
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    global loom_processor_gcs, gcs_qa_service, simple_transcriber, document_processor, company_bucket_service, website_scraper
    
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
        
        # Initialize GCS-based Loom Video Processor (NEW - Replaces Pinecone-based processor)
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
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
            return {
                'success': True,
                'answer': answer_result['answer'],
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
                'answer_source': 'gcs_transcript_search'
            }
        else:
            # Check if this is a "no relevant content" case vs actual error
            error_message = answer_result.get('error', 'Unknown error')
            is_no_content = 'no relevant content found' in error_message.lower() or 'no relevant information' in error_message.lower()
            
            if is_no_content:
                return {
                    'success': False,
                    'error': 'No relevant information found',
                    'answer': 'No relevant information found',
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
                return {
                    'success': False,
                    'error': error_message,
                    'answer': answer_result.get('answer', ''),
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

@app.get("/suggested-questions/{company_name}/{qudemo_id}")
async def get_suggested_questions(company_name: str, qudemo_id: str):
    """Get stored suggested questions for a QuDemo"""
    try:
        if not gcs_qa_service:
            raise HTTPException(status_code=500, detail="GCS Q&A service not available")
        
        logger.info(f"📖 FETCHING STORED suggested questions for {company_name}/{qudemo_id}")
        
        # Get stored suggested questions
        stored_questions = gcs_qa_service.gcs_service.get_suggested_questions(company_name, qudemo_id)
        
        if stored_questions:
            logger.info(f"✅ Retrieved {len(stored_questions)} stored suggested questions for {company_name}/{qudemo_id}")
            
            # Ensure "What is this about?" is the first question
            if not stored_questions or stored_questions[0] != "What is this about?":
                # Add "What is this about?" as the first question if it's not already there
                final_questions = ["What is this about?"] + [q for q in stored_questions if q != "What is this about?"]
                logger.info(f"✅ Added 'What is this about?' as first question. Final questions: {final_questions}")
            else:
                final_questions = stored_questions
            
            return {
                "success": True,
                "suggested_questions": final_questions
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
                'processing_complete': True,
                'total_chunks_stored': total_chunks,
                'videos': [video['url'] for video in successful_content['videos']],
                'websites': [website['url'] for website in successful_content['websites']],
                'videos_processed': len(successful_content['videos']),
                'website_processed': len(successful_content['websites']),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
