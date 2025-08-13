"""
QuDemo Video Processing API - Main Application
Clean, refactored version with modular architecture
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from datetime import datetime

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

# Import new knowledge processing modules
from web_scraper import WebScraper
from document_processor import DocumentProcessor
from knowledge_integration import KnowledgeIntegrator
from enhanced_qa import EnhancedQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances for knowledge processing
web_scraper = None
document_processor = None
knowledge_integrator = None
enhanced_qa = None

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global web_scraper, document_processor, knowledge_integrator, enhanced_qa
    
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
    
    # Initialize knowledge processing modules
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your_openai_api_key_here" and openai_api_key != "your-openai-api-key-here":
            web_scraper = WebScraper(openai_api_key)
            document_processor = DocumentProcessor()
            knowledge_integrator = KnowledgeIntegrator(openai_api_key)
            enhanced_qa = EnhancedQA(openai_api_key)
            logger.info("✅ Knowledge processing modules initialized successfully")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not found or invalid - knowledge processing disabled")
            logger.info("💡 To enable knowledge processing, set OPENAI_API_KEY environment variable")
    except Exception as e:
        logger.error(f"❌ Failed to initialize knowledge processing modules: {e}")
    
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
        
        # Process the video with extended timeout for Loom videos
        is_loom_video = 'loom.com' in video_url
        timeout_seconds = 810.0 if is_loom_video else 270.0  # 13.5 minutes for Loom, 4.5 minutes for others
        
        logger.info(f"⏱️ Using timeout: {timeout_seconds/60:.1f} minutes for {'Loom' if is_loom_video else 'other'} video")
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    process_video,
                    video_url=video_url,
                    company_name=company_name,
                    bucket_name=bucket_name,
                    source=source,
                    meeting_link=meeting_link
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            timeout_minutes = int(timeout_seconds / 60)
            logger.error(f"❌ Video processing timeout for {company_name}: {video_url} after {timeout_minutes} minutes")
            raise HTTPException(
                status_code=408, 
                detail=f"Video processing timed out after {timeout_minutes} minutes. Loom videos take longer to process. Please try again."
            )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
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

# Knowledge Processing Endpoints
@app.post("/process-website/{company_name}")
async def process_website_endpoint(company_name: str, request: Request):
    """Process website knowledge for a company"""
    global enhanced_qa
    
    try:
        if not enhanced_qa:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge processing not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        # Parse request body
        body = await request.json()
        website_url = body.get('website_url')
        knowledge_source_id = body.get('knowledge_source_id')  # New parameter
        
        if not website_url:
            raise HTTPException(status_code=400, detail="Website URL is required")
        
        logger.info(f"🌐 Processing website for {company_name}: {website_url}")
        
        # Process website knowledge with timeout (4.5 minutes)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    enhanced_qa.process_website_knowledge,
                    company_name,
                    website_url,
                    knowledge_source_id
                ),
                timeout=270.0  # 4.5 minutes
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Website processing timeout for {company_name}: {website_url}")
            raise HTTPException(status_code=408, detail="Website processing timed out. Please try again.")
        
        if result['success']:
            return {
                "success": True,
                "message": "Website knowledge processed successfully",
                "data": result['data']
            }
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Website processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-document/{company_name}")
async def process_document_endpoint(company_name: str, request: Request):
    """Process document knowledge for a specific company"""
    global document_processor, knowledge_integrator
    
    try:
        if not document_processor or not knowledge_integrator:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge processing not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        # Parse multipart form data
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(status_code=400, detail="file is required")
        
        # Read file content
        file_content = await file.read()
        filename = file.filename
        
        logger.info(f"📄 Processing document for {company_name}: {filename}")
        
        # Process document
        document_data = document_processor.process_uploaded_file(file_content, filename)
        
        if not document_data:
            raise HTTPException(status_code=400, detail="Failed to extract text from document")
        
        # Process document knowledge
        success = knowledge_integrator.process_document_knowledge(
            company_name=company_name,
            document_data=document_data,
            doc_processor=document_processor
        )
        
        if success:
            return {
                "success": True,
                "message": f"Document knowledge processed successfully for {company_name}",
                "filename": filename,
                "word_count": document_data.get("word_count", 0)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process document knowledge")
            
    except Exception as e:
        logger.error(f"❌ Document processing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/source/{source_id}/content")
async def get_knowledge_source_content_endpoint(source_id: str, company_name: str = Query(None)):
    """Get knowledge source content from Pinecone for preview"""
    global enhanced_qa
    
    try:
        if not enhanced_qa:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge retrieval not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        logger.info(f"📄 Getting knowledge source content: {source_id} (company: {company_name})")
        
        # Get content from Pinecone using the source ID and company name
        content_data = enhanced_qa.get_source_content(source_id, company_name)
        
        if content_data:
            return {
                "success": True,
                "data": content_data
            }
        else:
            raise HTTPException(status_code=404, detail="Knowledge source content not found")
            
    except Exception as e:
        logger.error(f"❌ Knowledge source content endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-enhanced/{company_name}")
async def ask_enhanced_question_endpoint(company_name: str, request: Request):
    """Ask a question using enhanced Q&A with all knowledge sources"""
    global enhanced_qa
    
    try:
        if not enhanced_qa:
            raise HTTPException(status_code=503, detail="Enhanced Q&A not available")
        
        # Parse request body
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        logger.info(f"🤖 Enhanced Q&A for {company_name}: {question}")
        
        # Get enhanced answer
        result = enhanced_qa.answer_question_enhanced(company_name, question)
        
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "source_type": result.get("source_type", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "video_timestamp": result.get("video_timestamp")
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced Q&A endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-summary/{company_name}")
async def get_knowledge_summary_endpoint(company_name: str):
    """Get knowledge summary for a specific company"""
    global knowledge_integrator
    
    try:
        if not knowledge_integrator:
            raise HTTPException(status_code=503, detail="Knowledge processing not available")
        
        summary = knowledge_integrator.get_knowledge_summary(company_name)
        return summary
        
    except Exception as e:
        logger.error(f"❌ Knowledge summary endpoint error: {e}")
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
    try:
        status = get_health_check()
        
        # Add Whisper model status
        try:
            from video_processing import loom_processor
            if loom_processor:
                whisper_loaded = loom_processor.is_whisper_loaded()
                status['whisper_model_loaded'] = whisper_loaded
                status['whisper_model_status'] = 'loaded' if whisper_loaded else 'not_loaded'
            else:
                status['whisper_model_loaded'] = False
                status['whisper_model_status'] = 'processor_not_initialized'
        except Exception as e:
            status['whisper_model_loaded'] = False
            status['whisper_model_status'] = f'error: {str(e)}'
        
        return status
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory-debug")
async def memory_debug():
    """Detailed memory debugging endpoint"""
    try:
        import psutil
        import gc
        
        # Get process info
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after GC
        memory_after_gc = process.memory_info().rss / 1024 / 1024
        
        return {
            "process_memory_mb": memory_mb,
            "memory_after_gc_mb": memory_after_gc,
            "memory_freed_mb": memory_mb - memory_after_gc,
            "system_total_mb": system_memory.total / 1024 / 1024,
            "system_available_mb": system_memory.available / 1024 / 1024,
            "system_percent": system_memory.percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/force-cleanup")
async def force_cleanup():
    """Force memory cleanup"""
    try:
        import gc
        import psutil
        
        # Get memory before cleanup
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024
        
        return {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_before - memory_after,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

# Startup and shutdown events are now handled by the lifespan event handler above

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
