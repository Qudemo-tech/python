#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Pinecone Standard Plan Multi-Index Architecture
Optimized for Q&A, video processing, and web scraping
"""

import os
import logging
from typing import List, Optional
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Enhanced components
from enhanced_pinecone_manager import initialize_enhanced_pinecone_manager, get_enhanced_pinecone_manager
from enhanced_knowledge_integration import initialize_enhanced_knowledge_integration, get_enhanced_knowledge_integration
from enhanced_qa_simple import initialize_simple_enhanced_qa, get_simple_enhanced_qa
from final_gemini_scraper import FinalGeminiScraper

# Video processing imports
from enhanced_video_processor import initialize_enhanced_video_processor, get_enhanced_video_processor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
enhanced_pinecone_manager = None
enhanced_knowledge_integration = None
enhanced_qa_system = None
enhanced_video_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    global enhanced_pinecone_manager, enhanced_knowledge_integration, enhanced_qa_system, enhanced_video_processor
    
    try:
        logger.info("🚀 Starting Enhanced QuDemo Python Backend...")
        
        # Initialize Enhanced Pinecone Manager
        if initialize_enhanced_pinecone_manager():
            enhanced_pinecone_manager = get_enhanced_pinecone_manager()
            logger.info("✅ Enhanced Pinecone Manager initialized")
        else:
            logger.error("❌ Failed to initialize Enhanced Pinecone Manager")
            return
        
        # Initialize Enhanced Knowledge Integration
        if initialize_enhanced_knowledge_integration():
            enhanced_knowledge_integration = get_enhanced_knowledge_integration()
            logger.info("✅ Enhanced Knowledge Integration initialized")
        else:
            logger.error("❌ Failed to initialize Enhanced Knowledge Integration")
            return
        
        # Initialize Enhanced Q&A System
        if initialize_simple_enhanced_qa():
            enhanced_qa_system = get_simple_enhanced_qa()
            logger.info("✅ Enhanced Q&A System initialized")
        else:
            logger.error("❌ Failed to initialize Enhanced Q&A System")
            return
        
        # Initialize Enhanced Video Processor
        try:
            if initialize_enhanced_video_processor():
                enhanced_video_processor = get_enhanced_video_processor()
                logger.info("✅ Enhanced Video Processor initialized")
            else:
                logger.warning("⚠️ Enhanced Video Processor initialization failed, will use direct processor")
                enhanced_video_processor = None
        except Exception as e:
            logger.warning(f"⚠️ Enhanced Video Processor initialization failed: {e}, will use direct processor")
            enhanced_video_processor = None
        
        # Initialize existing video processing system as fallback
        try:
            from video_processing import initialize_processors
            if initialize_processors():
                logger.info("✅ Existing video processing system initialized as fallback")
            else:
                logger.warning("⚠️ Failed to initialize existing video processing system")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize existing video processing system: {e}")
        
        logger.info("🎉 All enhanced components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down Enhanced QuDemo Python Backend...")
    try:
        if enhanced_pinecone_manager:
            enhanced_pinecone_manager.cleanup_cache()
            logger.info("🧹 Enhanced Pinecone Manager cache cleaned")
        logger.info("✅ Shutdown completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Enhanced QuDemo Python Backend",
    description="Optimized backend with Pinecone Standard Plan multi-index architecture",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuDemoContentRequest(BaseModel):
    video_urls: Optional[List[str]] = []
    website_url: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced QuDemo Python Backend",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Pinecone Standard Plan Multi-Index Architecture",
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
            "pinecone_manager": enhanced_pinecone_manager is not None,
            "knowledge_integration": enhanced_knowledge_integration is not None,
            "qa_system": enhanced_qa_system is not None,
            "video_processor": enhanced_video_processor is not None
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

@app.post("/ask/{company_name}/{qudemo_id}")
async def ask_question(company_name: str, qudemo_id: str, request: QuestionRequest):
    """Ask a question and get context-aware answer using enhanced Q&A system"""
    try:
        if not enhanced_qa_system:
            raise HTTPException(status_code=500, detail="Enhanced Q&A System not initialized")
        
        logger.info(f"❓ Processing question for {company_name} qudemo {qudemo_id}")
        
        # Use enhanced Q&A system to get answer
        answer_result = enhanced_qa_system.ask_question(
            question=request.question,
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if answer_result['success']:
            return {
                'success': True,
                'answer': answer_result['answer'],
                'sources': answer_result['sources'],
                'total_sources': answer_result['total_sources'],
                'search_score': answer_result['search_score'],
                'content_types_found': answer_result['content_types_found'],
                'difficulty_level': answer_result['difficulty_level'],
                'estimated_time': answer_result['estimated_time'],
                'start': answer_result.get('start', 0),
                'end': answer_result.get('end', 0),
                'video_url': answer_result.get('video_url'),
                'answer_source': answer_result.get('source', 'combined')
            }
        else:
            return {
                'success': False,
                'error': answer_result.get('error', 'Unknown error'),
                'answer': answer_result.get('answer', ''),
                'sources': []
            }
            
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/sources/{company_name}/{qudemo_id}")
async def get_knowledge_sources_qudemo(company_name: str, qudemo_id: str):
    """Get knowledge sources for a specific qudemo"""
    try:
        if not enhanced_knowledge_integration:
            raise HTTPException(status_code=500, detail="Enhanced Knowledge Integration not initialized")
        
        logger.info(f"📚 Getting knowledge sources for {company_name} qudemo {qudemo_id}")
        
        # Get knowledge summary
        summary_result = await enhanced_knowledge_integration.get_knowledge_summary(
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if summary_result['success']:
            return {
                'success': True,
                'data': summary_result['data'],
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to get knowledge summary: {summary_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Error getting knowledge sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-qudemo-content/{company_name}/{qudemo_id}")
async def process_qudemo_content(company_name: str, qudemo_id: str, request: QuDemoContentRequest):
    """Process qudemo content with optimized processing order: Videos first, then website"""
    try:
        if not enhanced_knowledge_integration:
            raise HTTPException(status_code=500, detail="Enhanced Knowledge Integration not initialized")
        
        logger.info(f"🔄 Processing qudemo content for {company_name} qudemo {qudemo_id}")
        
        # 🎯 OPTIMIZATION STRATEGY: Videos First, Then Website
        logger.info("🚀 Using optimized processing order: Videos (fast) → Website (may take longer)")
        
        total_chunks = 0
        processing_order = []
        
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
                    
                    # Process video using enhanced video processor or fallback
                    if enhanced_video_processor:
                        logger.info(f"🎥 Processing YouTube video: {video_url}")
                        result = await enhanced_video_processor.process_youtube_video(
                            video_url, company_name, qudemo_id
                        )
                    else:
                        # Fallback to direct processor
                        logger.info(f"🎥 Using direct processor for YouTube video: {video_url}")
                        from video_processing import process_video
                        result = process_video(video_url, company_name, qudemo_id)
                        
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
                        logger.info(f"✅ YouTube video processed: {chunks_stored} chunks stored")
                    else:
                        logger.error(f"❌ YouTube video processing failed: {result.get('error', 'Unknown error') if result else 'No result'}")
                        
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
                    
                    # Process video using enhanced video processor or fallback
                    if enhanced_video_processor:
                        logger.info(f"🎥 Processing Loom video: {video_url}")
                        result = await enhanced_video_processor.process_loom_video(
                            video_url, company_name, qudemo_id
                        )
                    else:
                        # Fallback to direct processor
                        logger.info(f"🎥 Using direct processor for Loom video: {video_url}")
                        from video_processing import process_video
                        result = process_video(video_url, company_name, qudemo_id)
                        
                        # Convert result format to match enhanced processor
                        if result and result.get('success'):
                            result = {
                                'success': True,
                                'chunks_stored': result.get('result', {}).get('chunks_created', 0),
                                'video_type': 'loom',
                                'company_name': company_name,
                                'qudemo_id': qudemo_id
                            }
                        else:
                            result = {
                                'success': False,
                                'error': result.get('error', 'Unknown error') if result else 'No result returned',
                                'chunks_stored': 0
                            }
                    
                    logger.info(f"📊 Loom video processing result: {result}")
                    
                    if result and result.get('success'):
                        chunks_stored = result.get('chunks_stored', 0)
                        total_chunks += chunks_stored
                        logger.info(f"✅ Loom video processed: {chunks_stored} chunks stored")
                    else:
                        logger.error(f"❌ Loom video processing failed: {result.get('error', 'Unknown error') if result else 'No result'}")
                        
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
            
        if request.website_url:
            logger.info(f"🌐 Step 2: Processing website (this may take longer): {request.website_url}")
            logger.info("⏱️ Website scraping typically takes 3-10 minutes depending on content size")
            processing_order.append("website")
            
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
            
            scraper = FinalGeminiScraper(gemini_api_key=gemini_api_key)
            website_results = await scraper.scrape_website_comprehensive(request.website_url)
            
            if website_results and len(website_results) > 0:
                # Store website results
                store_result = await enhanced_knowledge_integration.store_semantic_chunks(
                    chunks=website_results,
                    company_name=company_name,
                    qudemo_id=qudemo_id
                )
                
                if store_result['success']:
                    total_chunks += store_result['chunks_stored']
                    logger.info(f"✅ Website processed: {store_result['chunks_stored']} chunks stored")
                else:
                    logger.error(f"❌ Website storage failed: {store_result.get('error', 'Unknown error')}")
        
        # Final completion message
        if request.website_url:
            logger.info("✅ Step 2 (Website) completed successfully!")
        
        logger.info(f"🎉 All processing completed! Total chunks stored: {total_chunks}")
        
        return {
            'success': True,
            'message': f"Successfully processed qudemo content. Total chunks stored: {total_chunks}",
            'total_chunks_stored': total_chunks,
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'processing_order': processing_order,
            'optimization_note': "Videos processed first for faster results, website processed second"
        }
            
    except Exception as e:
        logger.error(f"❌ Error processing qudemo content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pinecone/status")
async def get_pinecone_status():
    """Get Pinecone connection and index status"""
    try:
        if not enhanced_pinecone_manager:
            raise HTTPException(status_code=500, detail="Enhanced Pinecone Manager not initialized")
        
        status = enhanced_pinecone_manager.get_status()
        return {
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
            
    except Exception as e:
        logger.error(f"❌ Error getting Pinecone status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
