"""
QuDemo Video Processing API - Main Application
Clean, refactored version with modular architecture
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import modules
from video_processing import (
    initialize_processors, 
    process_video, 
    initialize_existing_mappings,
    ProcessVideoRequest
)
from qa_system import (
    initialize_qa_system,
    answer_question,
    generate_summary,
    AskQuestionRequest,
    AskQuestionCompanyRequest,
    GenerateSummaryRequest
)
from health_checks import (
    get_system_status,
    get_memory_status,
    get_health_check
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("🚀 Starting QuDemo Video Processing API...")
    
    # Initialize video processing
    if initialize_processors():
        logger.info("✅ All processors initialized successfully")
    else:
        logger.error("❌ Failed to initialize some processors")
    
    # Initialize Q&A system
    if initialize_qa_system():
        logger.info("✅ Q&A system initialized successfully")
    else:
        logger.error("❌ Failed to initialize Q&A system")
    
    # Initialize video mappings
    initialize_existing_mappings()
    
    logger.info("🎉 API startup complete!")
    
    yield
    
    # Shutdown (if needed)
    logger.info("🛑 Shutting down QuDemo Video Processing API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="QuDemo Video Processing API",
    description="API for processing videos and answering questions",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Video Processing Endpoints
@app.post("/process-video/{company_name}")
async def process_video_endpoint(company_name: str, request: Request):
    """Process a video for a specific company"""
    try:
        # Parse request body
        body = await request.json()
        video_url = body.get("video_url")
        bucket_name = body.get("bucket_name")
        source = body.get("source")
        meeting_link = body.get("meeting_link")
        
        if not video_url:
            raise HTTPException(status_code=400, detail="video_url is required")
        
        logger.info(f"🎬 Processing video for {company_name}: {video_url}")
        
        # Process the video
        result = process_video(
            video_url=video_url,
            company_name=company_name,
            bucket_name=bucket_name,
            source=source,
            meeting_link=meeting_link
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"❌ Video processing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Q&A Endpoints
@app.post("/ask-question")
async def ask_question_endpoint(request: AskQuestionRequest):
    """Ask a question with company name in request body"""
    try:
        logger.info(f"❓ Question for {request.company_name}: {request.question}")
        
        result = answer_question(request.company_name, request.question)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'video_url': result.get('video_url'),
            'start': result.get('start'),
            'end': result.get('end')
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
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'video_url': result.get('video_url'),
            'start': result.get('start'),
            'end': result.get('end')
        }
        
    except Exception as e:
        logger.error(f"❌ Company question endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary")
async def generate_summary_endpoint(request: GenerateSummaryRequest):
    """Generate a summary from questions and answers"""
    try:
        logger.info(f"📝 Generating summary for {request.company_name}")
        
        result = generate_summary(
            questions_and_answers=request.questions_and_answers,
            buyer_name=request.buyer_name,
            company_name=request.company_name
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        logger.error(f"❌ Summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoints
@app.get("/status")
async def status_check():
    """Get system status"""
    return get_system_status()

@app.get("/memory-status")
async def memory_status():
    """Get current memory usage status"""
    return get_memory_status()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return get_health_check()



# Startup and shutdown events are now handled by the lifespan event handler above

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
