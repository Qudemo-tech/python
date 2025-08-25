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

# Suppress all Chrome/Selenium warnings and errors
import os
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_PRINT_FIRST_LINE'] = 'False'

import logging
logging.getLogger('selenium').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('WDM').setLevel(logging.ERROR)
logging.getLogger('faiss').setLevel(logging.ERROR)

# Import modules
from video_processing import (
    initialize_processors, 
    process_video, 
    initialize_existing_mappings,
    ProcessVideoRequest
)
from health_checks import (
    get_system_status,
    get_memory_status,
    get_health_check
)

# Import clean knowledge processing modules
from enhanced_qa import EnhancedQASystem, initialize_enhanced_qa
# Note: EnhancedKnowledgeIntegrator is no longer available - knowledge integration is handled by Node.js backend

# Configure logging with larger buffer and rotation
import logging
import logging.handlers
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            'app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)

logger = logging.getLogger(__name__)

# Global instances for clean knowledge processing
enhanced_qa_system = None
# Note: enhanced_knowledge_integrator is no longer used - knowledge integration is handled by Node.js backend

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global enhanced_qa_system
    
    # Startup
    logger.info("🚀 Starting QuDemo Video Processing API...")
    
    # Initialize video processing
    if initialize_processors():
        logger.info("✅ All processors initialized successfully")
    else:
        logger.error("❌ Failed to initialize some processors")
    
    # Initialize Q&A system
    success = await initialize_enhanced_qa()
    if success:
        logger.info("✅ Enhanced Q&A system initialized successfully")
        # Import the global instance after successful initialization
        from enhanced_qa import enhanced_qa_system
        global enhanced_qa_system
    else:
        logger.error("❌ Failed to initialize Enhanced Q&A system")
        logger.error("🔧 This will prevent website processing from working")
        logger.error("🔧 Please check your API keys in the .env file")
        enhanced_qa_system = None
    
    # Note: Knowledge integrator initialization removed - now handled by Node.js backend
    logger.info("ℹ️ Knowledge integration is now handled by the Node.js backend")
    
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
@app.post("/process-video/{company_name}/{qudemo_id}")
async def process_video_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Process a video for a specific qudemo within a company"""
    try:
        # Parse request body
        body = await request.json()
        video_url = body.get("video_url")
        bucket_name = body.get("bucket_name")
        source = body.get("source")
        meeting_link = body.get("meeting_link")
        
        if not video_url:
            raise HTTPException(status_code=400, detail="video_url is required")
        
        logger.info(f"🎬 Processing video for {company_name} qudemo {qudemo_id}: {video_url}")
        
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
                    qudemo_id=qudemo_id,
                    bucket_name=bucket_name,
                    source=source,
                    meeting_link=meeting_link
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            timeout_minutes = int(timeout_seconds / 60)
            logger.error(f"❌ Video processing timeout for {company_name} qudemo {qudemo_id}: {video_url} after {timeout_minutes} minutes")
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

@app.post("/process-videos-batch/{company_name}/{qudemo_id}")
async def process_videos_batch_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Process multiple videos for a specific qudemo sequentially"""
    try:
        # Parse request body
        body = await request.json()
        video_urls = body.get("video_urls", [])
        source = body.get("source", "batch")
        meeting_link = body.get("meeting_link")
        
        if not video_urls or not isinstance(video_urls, list):
            raise HTTPException(status_code=400, detail="video_urls must be a non-empty list")
        
        if len(video_urls) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")
        
        logger.info(f"🎬 Starting batch processing for {company_name} qudemo {qudemo_id}: {len(video_urls)} videos")
        
        # Initialize batch results
        batch_results = {
            "success": True,
            "company_name": company_name,
            "qudemo_id": qudemo_id,
            "total_videos": len(video_urls),
            "processed_videos": 0,
            "failed_videos": 0,
            "results": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration": None
        }
        
        start_time = datetime.now()
        
        # Process videos sequentially
        for i, video_url in enumerate(video_urls, 1):
            try:
                logger.info(f"🎬 Processing video {i}/{len(video_urls)}: {video_url}")
                
                # Check if it's a Loom video for timeout
                is_loom_video = 'loom.com' in video_url
                timeout_seconds = 810.0 if is_loom_video else 270.0  # 13.5 minutes for Loom, 4.5 minutes for others
                
                logger.info(f"⏱️ Using timeout: {timeout_seconds/60:.1f} minutes for {'Loom' if is_loom_video else 'other'} video")
                
                # Process the video
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        process_video,
                        video_url=video_url,
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        bucket_name=None,
                        source=source,
                        meeting_link=meeting_link
                    ),
                    timeout=timeout_seconds
                )
                
                if result["success"]:
                    batch_results["processed_videos"] += 1
                    batch_results["results"].append({
                        "video_url": video_url,
                        "status": "success",
                        "result": result["result"],
                        "processing_order": i
                    })
                    logger.info(f"✅ Video {i}/{len(video_urls)} processed successfully")
                else:
                    batch_results["failed_videos"] += 1
                    batch_results["results"].append({
                        "video_url": video_url,
                        "status": "failed",
                        "error": result["error"],
                        "processing_order": i
                    })
                    logger.error(f"❌ Video {i}/{len(video_urls)} failed: {result['error']}")
                
            except asyncio.TimeoutError:
                timeout_minutes = int(timeout_seconds / 60)
                batch_results["failed_videos"] += 1
                batch_results["results"].append({
                    "video_url": video_url,
                    "status": "timeout",
                    "error": f"Processing timed out after {timeout_minutes} minutes",
                    "processing_order": i
                })
                logger.error(f"⏰ Video {i}/{len(video_urls)} timed out after {timeout_minutes} minutes")
                
            except Exception as e:
                batch_results["failed_videos"] += 1
                batch_results["results"].append({
                    "video_url": video_url,
                    "status": "error",
                    "error": str(e),
                    "processing_order": i
                })
                logger.error(f"❌ Video {i}/{len(video_urls)} error: {e}")
            
            # Add small delay between videos for memory cleanup
            if i < len(video_urls):
                await asyncio.sleep(2)
        
        # Calculate batch completion metrics
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        batch_results["end_time"] = end_time.isoformat()
        batch_results["total_duration"] = total_duration
        batch_results["success"] = batch_results["failed_videos"] == 0
        
        # Log batch completion
        logger.info(f"🎬 Batch processing completed for {company_name} qudemo {qudemo_id}:")
        logger.info(f"   ✅ Processed: {batch_results['processed_videos']}/{batch_results['total_videos']}")
        logger.info(f"   ❌ Failed: {batch_results['failed_videos']}/{batch_results['total_videos']}")
        logger.info(f"   ⏱️ Total duration: {total_duration/60:.1f} minutes")
        
        return batch_results
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Batch processing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Q&A Endpoints
@app.post("/ask-question/{qudemo_id}")
async def ask_question_endpoint(qudemo_id: str, request: Request):
    """Ask a question for a specific qudemo"""
    try:
        body = await request.json()
        company_name = body.get("company_name")
        question = body.get("question")
        
        if not company_name or not question:
            raise HTTPException(status_code=400, detail="company_name and question are required")
        
        logger.info(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
        
        result = enhanced_qa_system.ask_question(question, company_name, qudemo_id)
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Question endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/{company_name}/{qudemo_id}")
async def ask_question_company_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Ask a question for a specific qudemo in a company"""
    try:
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        logger.info(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
        
        # Call the ask_question method (not async)
        result = enhanced_qa_system.ask_question(question, company_name, qudemo_id)
        
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to process question'))
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'video_url': result.get('video_url'),
            'start': result.get('start'),
            'end': result.get('end'),
            'video_title': result.get('video_title')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Company question endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary/{qudemo_id}")
async def generate_summary_endpoint(qudemo_id: str, request: Request):
    """Generate a summary from questions and answers for a specific qudemo"""
    try:
        body = await request.json()
        questions_and_answers = body.get("questions_and_answers", [])
        buyer_name = body.get("buyer_name")
        company_name = body.get("company_name")
        
        if not questions_and_answers or not company_name:
            raise HTTPException(status_code=400, detail="questions_and_answers and company_name are required")
        
        logger.info(f"📝 Generating summary for {company_name} qudemo {qudemo_id}")
        
        result = enhanced_qa_system.get_knowledge_summary(company_name, qudemo_id)
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Processing Endpoints
@app.post("/process-website/{company_name}/{qudemo_id}")
async def process_website_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Process website knowledge for a specific qudemo - comprehensive website crawling"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            logger.error("❌ Enhanced QA system not available")
            raise HTTPException(
                status_code=503, 
                detail="Knowledge processing not available. Please check API keys configuration and restart the server."
            )
        
        # Parse request body
        body = await request.json()
        website_url = body.get('website_url')
        
        if not website_url:
            raise HTTPException(status_code=400, detail="Website URL is required")
        
        logger.info(f"🌐 Processing website for {company_name} qudemo {qudemo_id}: {website_url}")
        logger.info(f"📊 COMPREHENSIVE MODE: Up to 50 collections × 100 articles per collection")
        logger.info(f"🌐 DOMAIN RESTRICTION: Will only crawl within the same domain")
        logger.info(f"⏱️ EXTENDED TIMEOUT: 300 minutes (5 hours) for very large website crawling")
        
        # Process website knowledge with extended timeout for comprehensive scraping
        try:
            result = await asyncio.wait_for(
                enhanced_qa_system.process_website_knowledge(website_url, company_name, qudemo_id),
                timeout=18000.0  # 300 minutes (5 hours) timeout for very large websites
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Website processing timeout for {company_name} qudemo {qudemo_id}: {website_url}")
            raise HTTPException(status_code=408, detail="Website processing timed out after 5 hours. This is normal for very large websites. Consider processing in smaller batches or using background job processing.")
        
        if result.get('success'):
            data = result.get('data', {})
            summary = data.get('summary', {})
            
            return {
                "success": True,
                "message": "Website knowledge processed successfully",
                "data": {
                    "chunks": data.get('chunks', []),
                    "qa_pairs": data.get('qa_pairs', []),
                    "summary": {
                        "total_items": summary.get('total_items', 0),
                        "enhanced": summary.get('enhanced', 0),
                        "faqs": summary.get('faqs', 0),
                        "beginner": summary.get('beginner', 0),
                        "intermediate": summary.get('intermediate', 0),
                        "advanced": summary.get('advanced', 0)
                    }
                },
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "website_url": website_url
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Website processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/sources/{company_name}/{qudemo_id}")
async def get_knowledge_sources_endpoint(company_name: str, qudemo_id: str):
    """Get all knowledge sources for a specific qudemo"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge retrieval not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        logger.info(f"📄 Getting knowledge sources for company: {company_name} qudemo: {qudemo_id}")
        
        # Get knowledge summary which includes all sources
        summary_data = enhanced_qa_system.get_knowledge_summary(company_name, qudemo_id)
        
        if summary_data and summary_data.get("success"):
            return summary_data
        else:
            raise HTTPException(status_code=404, detail="Knowledge sources not found")
            
    except Exception as e:
        logger.error(f"❌ Knowledge sources endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/source/{source_id}/content/{qudemo_id}")
async def get_knowledge_source_content_endpoint(source_id: str, qudemo_id: str, company_name: str = Query(None)):
    """Get knowledge source content from Pinecone for preview for a specific qudemo"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge retrieval not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        logger.info(f"📄 Getting knowledge source content: {source_id} (company: {company_name}, qudemo: {qudemo_id})")
        
        # Get content from Pinecone using the source ID, company name, and qudemo ID
        content_data = enhanced_qa_system.get_source_content(source_id, company_name, qudemo_id)
        
        if content_data and content_data.get("success"):
            return content_data
        else:
            raise HTTPException(status_code=404, detail="Knowledge source content not found")
            
    except Exception as e:
        logger.error(f"❌ Knowledge source content endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-enhanced/{company_name}/{qudemo_id}")
async def ask_enhanced_question_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Ask a question using enhanced Q&A with all knowledge sources for a specific qudemo"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(status_code=503, detail="Enhanced Q&A not available")
        
        # Parse request body
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        logger.info(f"🤖 Enhanced Q&A for {company_name} qudemo {qudemo_id}: {question}")
        
        # Get enhanced answer
        result = enhanced_qa_system.ask_question(question, company_name, qudemo_id)
        
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "source_type": result.get("source_type", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "video_timestamp": result.get("video_timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced Q&A endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Qudemo Content Processing Endpoint
@app.post("/process-qudemo-content/{company_name}/{qudemo_id}")
async def process_qudemo_content_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Process all content for a qudemo automatically - videos first, then website if provided"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            logger.error("❌ Enhanced QA system not available")
            raise HTTPException(
                status_code=503, 
                detail="Content processing not available. Please check API keys configuration and restart the server."
            )
        
        # Parse request body
        body = await request.json()
        video_urls = body.get('video_urls', [])
        website_url = body.get('website_url')
        
        if not video_urls and not website_url:
            raise HTTPException(status_code=400, detail="At least one content source (video_urls or website_url) is required")
        
        logger.info(f"🚀 Processing qudemo content for {company_name} qudemo {qudemo_id}")
        logger.info(f"📹 Videos: {len(video_urls) if video_urls else 0}")
        logger.info(f"🌐 Website: {website_url if website_url else 'None'}")
        
        # Process all content automatically
        result = await enhanced_qa_system.process_qudemo_content(
            company_name=company_name,
            qudemo_id=qudemo_id,
            video_urls=video_urls,
            website_url=website_url
        )
        
        if result.get('success'):
            return {
                "success": True,
                "message": result.get('message', 'Content processed successfully'),
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "videos_processed": result.get('videos_processed', 0),
                "website_processed": result.get('website_processed', False),
                "total_chunks": result.get('total_chunks', 0),
                "processing_order": result.get('processing_order', []),
                "errors": result.get('errors', [])
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Content processing failed'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Qudemo content processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Management Endpoints
@app.delete("/delete-qudemo-data/{company_name}/{qudemo_id}")
async def delete_qudemo_data_endpoint(company_name: str, qudemo_id: str):
    """Delete all data for a specific qudemo from Pinecone"""
    try:
        logger.info(f"🗑️ Deleting all data for company: {company_name} qudemo: {qudemo_id}")
        
        # Initialize Q&A system if not already done
        import os
        
        # Initialize if not already done
        success = await initialize_enhanced_qa()
        
        # Get the initialized enhanced_knowledge_integrator after initialization
        from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        
        # Get index name
        index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Get the index from enhanced_knowledge_integrator
        if enhanced_knowledge_integrator and enhanced_knowledge_integrator.index:
            index = enhanced_knowledge_integrator.index
        else:
            # Fallback: create new integrator
            integrator = EnhancedKnowledgeIntegrator(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                pinecone_index=index_name
            )
            index = integrator.index
        
        # Create namespace from company name and qudemo ID
        namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
        logger.info(f"🗑️ Deleting namespace: {namespace} from index: {index_name}")
        
        # Get index stats to check if namespace exists
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                logger.warning(f"⚠️ No vectors found in index {index_name}")
                return {
                    "success": True,
                    "message": "No data found to delete",
                    "deleted_count": 0
                }
            
            # Delete all vectors in the namespace
            # Pinecone doesn't have a direct delete by namespace method, so we need to:
            # 1. Query all vectors in the namespace
            # 2. Delete them by ID
            
            # Query all vectors in the namespace
            dummy_embedding = [0.0] * 1536  # OpenAI embedding dimension
            
            try:
                # Get all vectors in the namespace
                query_response = index.query(
                    vector=dummy_embedding,
                    top_k=10000,  # Get all vectors (max limit)
                    include_metadata=True,
                    namespace=namespace
                )
                
                if not query_response.matches:
                    logger.info(f"✅ No vectors found in namespace '{namespace}'")
                    return {
                        "success": True,
                        "message": "No data found in namespace",
                        "deleted_count": 0,
                        "namespace": namespace
                    }
                
                # Extract vector IDs to delete
                vector_ids_to_delete = [match.id for match in query_response.matches]
                logger.info(f"🗑️ Found {len(vector_ids_to_delete)} vectors to delete in namespace '{namespace}'")
                
                # Delete vectors by ID
                if vector_ids_to_delete:
                    # Delete in batches of 1000 (Pinecone limit)
                    batch_size = 1000
                    total_deleted = 0
                    
                    for i in range(0, len(vector_ids_to_delete), batch_size):
                        batch = vector_ids_to_delete[i:i + batch_size]
                        logger.info(f"🗑️ Deleting batch {i//batch_size + 1}: {len(batch)} vectors")
                        
                        try:
                            index.delete(ids=batch, namespace=namespace)
                            total_deleted += len(batch)
                            logger.info(f"✅ Deleted batch {i//batch_size + 1}: {len(batch)} vectors")
                        except Exception as batch_error:
                            logger.error(f"❌ Failed to delete batch {i//batch_size + 1}: {batch_error}")
                            # Continue with other batches
                    
                    logger.info(f"🎉 Successfully deleted {total_deleted} vectors from namespace '{namespace}'")
                    
                    return {
                        "success": True,
                        "message": f"Successfully deleted {total_deleted} vectors",
                        "deleted_count": total_deleted,
                        "namespace": namespace,
                        "index": index_name
                    }
                else:
                    return {
                        "success": True,
                        "message": "No vectors found to delete",
                        "deleted_count": 0,
                        "namespace": namespace
                    }
                    
            except Exception as query_error:
                logger.error(f"❌ Error querying namespace '{namespace}': {query_error}")
                return {
                    "success": False,
                    "error": f"Failed to query namespace: {str(query_error)}",
                    "namespace": namespace
                }
                
        except Exception as stats_error:
            logger.error(f"❌ Error getting index stats: {stats_error}")
            return {
                "success": False,
                "error": f"Failed to get index stats: {str(stats_error)}"
            }
            
    except Exception as e:
        logger.error(f"❌ Delete qudemo data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-video-data/{company_name}/{qudemo_id}")
async def delete_video_data_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Delete video data from Pinecone when processing fails for a specific qudemo"""
    try:
        logger.info(f"🗑️ Deleting video data from Pinecone for company: {company_name} qudemo: {qudemo_id}")
        
        body = await request.json()
        video_id = body.get("video_id")
        video_url = body.get("video_url")
        
        if not video_id and not video_url:
            raise HTTPException(status_code=400, detail="video_id or video_url is required")
        
        # Initialize Q&A system if not already done
        import os
        
        # Initialize if not already done
        success = await initialize_enhanced_qa()
        
        # Get the initialized enhanced_knowledge_integrator after initialization
        from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        
        # Get index name
        index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Get the index from enhanced_knowledge_integrator
        if enhanced_knowledge_integrator and enhanced_knowledge_integrator.index:
            index = enhanced_knowledge_integrator.index
        else:
            # Fallback: create new integrator
            integrator = EnhancedKnowledgeIntegrator(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                pinecone_index=index_name
            )
            index = integrator.index
        
        # Create namespace from company name and qudemo ID
        namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
        
        # Delete vectors by metadata filter
        try:
            # Delete by video_id if provided
            if video_id:
                index.delete(filter={"video_id": video_id}, namespace=namespace)
                logger.info(f"✅ Deleted vectors with video_id: {video_id}")
            
            # Delete by video_url if provided
            if video_url:
                index.delete(filter={"video_url": video_url}, namespace=namespace)
                logger.info(f"✅ Deleted vectors with video_url: {video_url}")
            
            # Also delete by company_name, qudemo_id and source type
            index.delete(filter={
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "source": "video"
            }, namespace=namespace)
            logger.info(f"✅ Deleted video vectors for company: {company_name} qudemo: {qudemo_id}")
            
            return {
                "success": True,
                "message": "Video data deleted from Pinecone successfully",
                "deleted_video_id": video_id,
                "deleted_video_url": video_url
            }
            
        except Exception as delete_error:
            logger.error(f"❌ Error deleting from Pinecone: {delete_error}")
            return {
                "success": False,
                "error": f"Failed to delete from Pinecone: {str(delete_error)}"
            }
            
    except Exception as e:
        logger.error(f"❌ Delete video data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-website-data/{company_name}/{qudemo_id}")
async def delete_website_data_endpoint(company_name: str, qudemo_id: str, request: Request):
    """Delete website data from Pinecone when processing fails for a specific qudemo"""
    try:
        logger.info(f"🗑️ Deleting website data from Pinecone for company: {company_name} qudemo: {qudemo_id}")
        
        body = await request.json()
        website_url = body.get("website_url")
        knowledge_source_id = body.get("knowledge_source_id")
        
        if not website_url and not knowledge_source_id:
            raise HTTPException(status_code=400, detail="website_url or knowledge_source_id is required")
        
        # Initialize Q&A system if not already done
        import os
        
        # Initialize if not already done
        success = await initialize_enhanced_qa()
        
        # Get the initialized enhanced_knowledge_integrator after initialization
        from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        
        # Get index name
        index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Get the index from enhanced_knowledge_integrator
        if enhanced_knowledge_integrator and enhanced_knowledge_integrator.index:
            index = enhanced_knowledge_integrator.index
        else:
            # Fallback: create new integrator
            integrator = EnhancedKnowledgeIntegrator(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                pinecone_index=index_name
            )
            index = integrator.index
        
        # Create namespace from company name and qudemo ID
        namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
        
        # Delete vectors by metadata filter
        try:
            # Delete by website_url if provided
            if website_url:
                index.delete(filter={"url": website_url}, namespace=namespace)
                logger.info(f"✅ Deleted vectors with website_url: {website_url}")
            
            # Delete by knowledge_source_id if provided
            if knowledge_source_id:
                index.delete(filter={"knowledge_source_id": knowledge_source_id}, namespace=namespace)
                logger.info(f"✅ Deleted vectors with knowledge_source_id: {knowledge_source_id}")
            
            # Also delete by company_name, qudemo_id and source type
            index.delete(filter={
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "source": "web_scraping"
            }, namespace=namespace)
            logger.info(f"✅ Deleted website vectors for company: {company_name} qudemo: {qudemo_id}")
            
            return {
                "success": True,
                "message": "Website data deleted from Pinecone successfully",
                "deleted_website_url": website_url,
                "deleted_knowledge_source_id": knowledge_source_id
            }
            
        except Exception as delete_error:
            logger.error(f"❌ Error deleting from Pinecone: {delete_error}")
            return {
                "success": False,
                "error": f"Failed to delete from Pinecone: {str(delete_error)}"
            }
            
    except Exception as e:
        logger.error(f"❌ Delete website data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-document-data/{company_name}")
async def delete_document_data_endpoint(company_name: str, request: Request):
    """Delete document data from Pinecone when processing fails"""
    try:
        logger.info(f"🗑️ Deleting document data from Pinecone for company: {company_name}")
        
        body = await request.json()
        file_name = body.get("file_name")
        knowledge_source_id = body.get("knowledge_source_id")
        
        if not file_name and not knowledge_source_id:
            raise HTTPException(status_code=400, detail="file_name or knowledge_source_id is required")
        
        # Initialize Q&A system if not already done
        import os
        
        # Initialize if not already done
        success = await initialize_enhanced_qa()
        
        # Get the initialized enhanced_knowledge_integrator after initialization
        # from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        
        # Get index name
        index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Get the index from enhanced_knowledge_integrator
        # if enhanced_knowledge_integrator and enhanced_knowledge_integrator.index:
        #     index = enhanced_knowledge_integrator.index
        # else:
        #     # Fallback: create new integrator
        #     integrator = EnhancedKnowledgeIntegrator(
        #         openai_api_key=os.getenv("OPENAI_API_KEY"),
        #         pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        #         pinecone_index=index_name
        #     )
        #     index = integrator.index
        
        # Delete vectors by metadata filter
        try:
            # Delete by file_name if provided
            if file_name:
                index.delete(filter={"file_name": file_name})
                logger.info(f"✅ Deleted vectors with file_name: {file_name}")
            
            # Delete by knowledge_source_id if provided
            if knowledge_source_id:
                index.delete(filter={"knowledge_source_id": knowledge_source_id})
                logger.info(f"✅ Deleted vectors with knowledge_source_id: {knowledge_source_id}")
            
            # Also delete by company_name and source type
            index.delete(filter={
                "company_name": company_name,
                "source": "document"
            })
            logger.info(f"✅ Deleted document vectors for company: {company_name}")
            
            return {
                "success": True,
                "message": "Document data deleted from Pinecone successfully",
                "deleted_file_name": file_name,
                "deleted_knowledge_source_id": knowledge_source_id
            }
            
        except Exception as delete_error:
            logger.error(f"❌ Error deleting from Pinecone: {delete_error}")
            return {
                "success": False,
                "error": f"Failed to delete from Pinecone: {str(delete_error)}"
            }
            
    except Exception as e:
        logger.error(f"❌ Delete document data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoints
@app.get("/status")
async def status_check():
    """Get system status"""
    return get_system_status()

@app.get("/memory-status")
async def memory_status():
    """Get current memory usage status"""
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

# Test endpoint to get qudemo data without authentication (for debugging)
@app.get("/test/qudemo/{company_name}/{qudemo_id}")
async def test_get_qudemo_data(company_name: str, qudemo_id: str):
    """Test endpoint to get qudemo data without authentication"""
    try:
        # Get knowledge sources
        knowledge_result = await enhanced_qa_system.get_knowledge_summary(company_name, qudemo_id)
        
        # Get video chunks if any
        video_chunks = []
        try:
            # This would normally come from Node.js backend, but for testing we'll create a mock
            video_chunks = [
                {
                    "id": f"video_{qudemo_id}_1",
                    "title": "Sample Video",
                    "chunks_count": 0,
                    "status": "processed"
                }
            ]
        except Exception as e:
            print(f"Warning: Could not get video data: {e}")
        
        return {
            "success": True,
            "data": {
                "qudemo_id": qudemo_id,
                "company_name": company_name,
                "knowledge_sources": knowledge_result.get("data", {}).get("sources", []),
                "videos": video_chunks,
                "total_knowledge": len(knowledge_result.get("data", {}).get("sources", [])),
                "total_videos": len(video_chunks)
            }
        }
    except Exception as e:
        print(f"Error in test endpoint: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
