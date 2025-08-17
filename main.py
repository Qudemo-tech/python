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
from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator

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
enhanced_knowledge_integrator = None

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global enhanced_qa_system, enhanced_knowledge_integrator
    
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
    
    # Initialize knowledge integrator (separate from QA system)
    global enhanced_knowledge_integrator
    
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key and openai_api_key != "your_openai_api_key_here" and openai_api_key != "your-openai-api-key-here":
            # Initialize knowledge integrator for embeddings and storage
            enhanced_knowledge_integrator = EnhancedKnowledgeIntegrator(
                openai_api_key=openai_api_key,
                pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                pinecone_index=os.getenv('PINECONE_INDEX')
            )
            logger.info("✅ Knowledge integrator initialized successfully")
        else:
            logger.warning("⚠️ OpenAI API key not found or invalid - knowledge storage disabled")
            logger.info("💡 To enable knowledge storage, set OPENAI_API_KEY environment variable")
            enhanced_knowledge_integrator = None
    except Exception as e:
        logger.error(f"❌ Failed to initialize knowledge integrator: {e}")
        enhanced_knowledge_integrator = None
    
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

@app.post("/process-videos-batch/{company_name}")
async def process_videos_batch_endpoint(company_name: str, request: Request):
    """Process multiple videos for a specific company sequentially"""
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
        
        logger.info(f"🎬 Starting batch processing for {company_name}: {len(video_urls)} videos")
        
        # Initialize batch results
        batch_results = {
            "success": True,
            "company_name": company_name,
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
        logger.info(f"🎬 Batch processing completed for {company_name}:")
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
@app.post("/ask-question")
async def ask_question_endpoint(request: Request):
    """Ask a question with company name in request body"""
    try:
        body = await request.json()
        company_name = body.get("company_name")
        question = body.get("question")
        
        if not company_name or not question:
            raise HTTPException(status_code=400, detail="company_name and question are required")
        
        logger.info(f"❓ Question for {company_name}: {question}")
        
        result = enhanced_qa_system.ask_question(question, company_name)
        
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

@app.post("/ask/{company_name}")
async def ask_question_company_endpoint(company_name: str, request: Request):
    """Ask a question for a specific company"""
    try:
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        logger.info(f"❓ Question for {company_name}: {question}")
        
        # Call the async ask_question method
        result = await enhanced_qa_system.ask_question(question, company_name)
        
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

@app.post("/generate-summary")
async def generate_summary_endpoint(request: Request):
    """Generate a summary from questions and answers"""
    try:
        body = await request.json()
        questions_and_answers = body.get("questions_and_answers", [])
        buyer_name = body.get("buyer_name")
        company_name = body.get("company_name")
        
        if not questions_and_answers or not company_name:
            raise HTTPException(status_code=400, detail="questions_and_answers and company_name are required")
        
        logger.info(f"📝 Generating summary for {company_name}")
        
        result = enhanced_qa_system.get_knowledge_summary(company_name)
        
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
@app.post("/process-website/{company_name}")
async def process_website_endpoint(company_name: str, request: Request):
    """Process website knowledge for a company with 3-page limit"""
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
        
        logger.info(f"🌐 Processing website for {company_name}: {website_url}")
        logger.info(f"📊 Using 3-page limit: 3 collections × 3 articles per collection")
        
        # Process website knowledge with timeout (10 minutes for comprehensive scraping)
        try:
            result = await asyncio.wait_for(
                enhanced_qa_system.process_website_knowledge(website_url, company_name),
                timeout=600.0  # 10 minutes timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Website processing timeout for {company_name}: {website_url}")
            raise HTTPException(status_code=408, detail="Website processing timed out. Please try again.")
        
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

# Removed document processing endpoint - using clean system instead

# Removed old enhanced website processing endpoint - using clean scraper instead

@app.get("/knowledge/sources/{company_name}")
async def get_knowledge_sources_endpoint(company_name: str):
    """Get all knowledge sources for a company"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge retrieval not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        logger.info(f"📄 Getting knowledge sources for company: {company_name}")
        
        # Get knowledge summary which includes all sources
        summary_data = enhanced_qa_system.get_knowledge_summary(company_name)
        
        if summary_data and summary_data.get("success"):
            return summary_data
        else:
            raise HTTPException(status_code=404, detail="Knowledge sources not found")
            
    except Exception as e:
        logger.error(f"❌ Knowledge sources endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/source/{source_id}/content")
async def get_knowledge_source_content_endpoint(source_id: str, company_name: str = Query(None)):
    """Get knowledge source content from Pinecone for preview"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge retrieval not available. Please set OPENAI_API_KEY environment variable to enable this feature."
            )
        
        logger.info(f"📄 Getting knowledge source content: {source_id} (company: {company_name})")
        
        # Get content from Pinecone using the source ID and company name
        content_data = enhanced_qa_system.get_source_content(source_id, company_name)
        
        if content_data and content_data.get("success"):
            return content_data
        else:
            raise HTTPException(status_code=404, detail="Knowledge source content not found")
            
    except Exception as e:
        logger.error(f"❌ Knowledge source content endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-enhanced/{company_name}")
async def ask_enhanced_question_endpoint(company_name: str, request: Request):
    """Ask a question using enhanced Q&A with all knowledge sources"""
    global enhanced_qa_system
    
    try:
        if not enhanced_qa_system:
            raise HTTPException(status_code=503, detail="Enhanced Q&A not available")
        
        # Parse request body
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        logger.info(f"🤖 Enhanced Q&A for {company_name}: {question}")
        
        # Get enhanced answer
        result = enhanced_qa_system.ask_question(question, company_name)
        
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

# Removed old knowledge summary and scraping stats endpoints - using clean system instead

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

@app.delete("/delete-company-data/{company_name}")
async def delete_company_data_endpoint(company_name: str):
    """Delete all company data from Pinecone"""
    try:
        logger.info(f"🗑️ Deleting all data for company: {company_name}")
        
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
        
        # Create namespace from company name
        namespace = company_name.lower().replace(' ', '-')
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
        logger.error(f"❌ Delete company data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-knowledge-source/{company_name}")
async def delete_knowledge_source_endpoint(company_name: str, request: Request):
    """Delete specific knowledge source from Pinecone"""
    try:
        logger.info(f"🗑️ Deleting knowledge source for company: {company_name}")
        
        body = await request.json()
        source_id = body.get("source_id")
        source_type = body.get("source_type")
        source_url = body.get("source_url")
        title = body.get("title")
        
        if not source_id:
            raise HTTPException(status_code=400, detail="source_id is required")
        
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
        
        # Create namespace from company name
        namespace = company_name.lower().replace(' ', '-')
        logger.info(f"🗑️ Deleting knowledge source from namespace: {namespace}")
        
        # Query for vectors that match the source criteria
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
            
            # Filter vectors that match the source criteria
            vectors_to_delete = []
            for match in query_response.matches:
                metadata = match.metadata or {}
                
                # Check if this vector belongs to the source we want to delete
                # Match by source_id, source_url, or title
                if (metadata.get("source_id") == source_id or
                    metadata.get("source_url") == source_url or
                    metadata.get("title") == title or
                    metadata.get("source_type") == source_type):
                    vectors_to_delete.append(match.id)
            
            if not vectors_to_delete:
                logger.info(f"✅ No vectors found matching source criteria")
                return {
                    "success": True,
                    "message": "No matching vectors found",
                    "deleted_count": 0,
                    "namespace": namespace,
                    "source_id": source_id
                }
            
            logger.info(f"🗑️ Found {len(vectors_to_delete)} vectors to delete for source: {source_id}")
            
            # Delete vectors by ID
            if vectors_to_delete:
                # Delete in batches of 1000 (Pinecone limit)
                batch_size = 1000
                total_deleted = 0
                
                for i in range(0, len(vectors_to_delete), batch_size):
                    batch = vectors_to_delete[i:i + batch_size]
                    logger.info(f"🗑️ Deleting batch {i//batch_size + 1}: {len(batch)} vectors")
                    
                    try:
                        index.delete(ids=batch, namespace=namespace)
                        total_deleted += len(batch)
                        logger.info(f"✅ Deleted batch {i//batch_size + 1}: {len(batch)} vectors")
                    except Exception as batch_error:
                        logger.error(f"❌ Failed to delete batch {i//batch_size + 1}: {batch_error}")
                        # Continue with other batches
                
                logger.info(f"🎉 Successfully deleted {total_deleted} vectors for source: {source_id}")
                
                return {
                    "success": True,
                    "message": f"Successfully deleted {total_deleted} vectors",
                    "deleted_count": total_deleted,
                    "namespace": namespace,
                    "source_id": source_id,
                    "index": index_name
                }
            else:
                return {
                    "success": True,
                    "message": "No vectors found to delete",
                    "deleted_count": 0,
                    "namespace": namespace,
                    "source_id": source_id
                }
                
        except Exception as query_error:
            logger.error(f"❌ Error querying namespace '{namespace}': {query_error}")
            return {
                "success": False,
                "error": f"Failed to query namespace: {str(query_error)}",
                "namespace": namespace
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete knowledge source error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive Support Bot Endpoints
@app.post("/scrape-company-support/{company_name}")
async def scrape_company_support_endpoint(company_name: str, request: Request):
    """Scrape comprehensive support data from company websites"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        body = await request.json()
        website_urls = body.get("website_urls", [])
        max_pages_per_url = body.get("max_pages_per_url", 10)
        
        if not website_urls:
            raise HTTPException(status_code=400, detail="website_urls is required")
        
        logger.info(f"🕷️ Starting comprehensive support scraping for {company_name}")
        
        # Scrape support data
        support_data = enhanced_knowledge_integrator.scrape_company_support_data(
            company_name, website_urls, max_pages_per_url
        )
        
        # Get summary
        summary = enhanced_knowledge_integrator.get_knowledge_summary(company_name)
        
        return {
            "success": True,
            "company_name": company_name,
            "support_data_count": len(support_data),
            "summary": summary,
            "message": f"Successfully scraped {len(support_data)} support data items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Support scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrate-video-transcripts/{company_name}")
async def integrate_video_transcripts_endpoint(company_name: str, request: Request):
    """Integrate video transcripts with support data"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        body = await request.json()
        video_transcripts = body.get("video_transcripts", [])
        
        if not video_transcripts:
            raise HTTPException(status_code=400, detail="video_transcripts is required")
        
        logger.info(f"🎬 Integrating video transcripts for {company_name}")
        
        # Integrate video transcripts
        integrated_data = enhanced_knowledge_integrator.integrate_video_transcripts(
            company_name, video_transcripts
        )
        
        return {
            "success": True,
            "company_name": company_name,
            "integrated_data_count": len(integrated_data),
            "message": f"Successfully integrated {len(integrated_data)} data items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance-support-data/{company_name}")
async def enhance_support_data_endpoint(company_name: str):
    """Enhance support data with AI-generated content"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        logger.info(f"🤖 Enhancing support data for {company_name}")
        
        # Enhance support data
        enhanced_data = enhanced_knowledge_integrator.enhance_support_data_with_ai(company_name)
        
        return {
            "success": True,
            "company_name": company_name,
            "enhanced_data_count": len(enhanced_data),
            "message": f"Successfully enhanced {len(enhanced_data)} support data items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Support data enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-raw-content/{company_name}")
async def process_raw_content_endpoint(company_name: str, request: Request):
    """Process raw HTML content and extract complete content chunks without Q&A generation"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        body = await request.json()
        html_content = body.get("html_content", "")
        url = body.get("url", "")
        
        if not html_content:
            raise HTTPException(status_code=400, detail="html_content is required")
        
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        
        logger.info(f"🔍 Processing raw content for {company_name} from: {url}")
        
        # Use enhanced knowledge integrator to process raw content
        content_chunks = enhanced_knowledge_integrator.process_raw_content(
            company_name, html_content, url
        )
        
        # Get summary
        summary = enhanced_knowledge_integrator.get_knowledge_summary(company_name)
        
        return {
            "success": True,
            "company_name": company_name,
            "content_chunks_count": len(content_chunks),
            "summary": summary,
            "message": f"Successfully processed {len(content_chunks)} content chunks"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Process raw content error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-vector-embeddings/{company_name}")
async def create_vector_embeddings_endpoint(company_name: str):
    """Create vector embeddings for company knowledge"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        logger.info(f"🔢 Creating vector embeddings for {company_name}")
        
        # Create vector embeddings
        embeddings = enhanced_knowledge_integrator.create_vector_embeddings(company_name)
        
        return {
            "success": True,
            "company_name": company_name,
            "embeddings_count": len(embeddings),
            "message": f"Successfully created {len(embeddings)} vector embeddings"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Vector embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-support-question")
async def ask_support_question_endpoint(request: Request):
    """Ask a support question using comprehensive knowledge base"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        body = await request.json()
        company_name = body.get("company_name")
        question = body.get("question")
        
        if not company_name or not question:
            raise HTTPException(status_code=400, detail="company_name and question are required")
        
        logger.info(f"🤖 Support question for {company_name}: {question}")
        
        # Generate support response
        response = enhanced_knowledge_integrator.generate_support_response(company_name, question)
        
        return {
            "success": True,
            "company_name": company_name,
            "question": question,
            "response": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Support question error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/support-knowledge-summary/{company_name}")
async def get_support_knowledge_summary_endpoint(company_name: str):
    """Get comprehensive summary of company support knowledge"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        summary = enhanced_knowledge_integrator.get_knowledge_summary(company_name)
        
        return {
            "success": True,
            "company_name": company_name,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Knowledge summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-company-knowledge/{company_name}")
async def export_company_knowledge_endpoint(company_name: str):
    """Export company knowledge to JSON file"""
    try:
        if not enhanced_knowledge_integrator:
            raise HTTPException(status_code=503, detail="Support bot system not initialized")
        
        filename = f"company_knowledge_{company_name}_{int(time.time())}.json"
        enhanced_knowledge_integrator.export_company_knowledge(company_name, filename)
        
        return {
            "success": True,
            "company_name": company_name,
            "filename": filename,
            "message": f"Successfully exported company knowledge to {filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Knowledge export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events are now handled by the lifespan event handler above

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.delete("/delete-video-data/{company_name}")
async def delete_video_data_endpoint(company_name: str, request: Request):
    """Delete video data from Pinecone when processing fails"""
    try:
        logger.info(f"🗑️ Deleting video data from Pinecone for company: {company_name}")
        
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
        
        # Delete vectors by metadata filter
        try:
            # Delete by video_id if provided
            if video_id:
                index.delete(filter={"video_id": video_id})
                logger.info(f"✅ Deleted vectors with video_id: {video_id}")
            
            # Delete by video_url if provided
            if video_url:
                index.delete(filter={"video_url": video_url})
                logger.info(f"✅ Deleted vectors with video_url: {video_url}")
            
            # Also delete by company_name and source type
            index.delete(filter={
                "company_name": company_name,
                "source": "video"
            })
            logger.info(f"✅ Deleted video vectors for company: {company_name}")
            
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

@app.delete("/delete-website-data/{company_name}")
async def delete_website_data_endpoint(company_name: str, request: Request):
    """Delete website data from Pinecone when processing fails"""
    try:
        logger.info(f"🗑️ Deleting website data from Pinecone for company: {company_name}")
        
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
        
        # Delete vectors by metadata filter
        try:
            # Delete by website_url if provided
            if website_url:
                index.delete(filter={"url": website_url})
                logger.info(f"✅ Deleted vectors with website_url: {website_url}")
            
            # Delete by knowledge_source_id if provided
            if knowledge_source_id:
                index.delete(filter={"knowledge_source_id": knowledge_source_id})
                logger.info(f"✅ Deleted vectors with knowledge_source_id: {knowledge_source_id}")
            
            # Also delete by company_name and source type
            index.delete(filter={
                "company_name": company_name,
                "source": "web_scraping"
            })
            logger.info(f"✅ Deleted website vectors for company: {company_name}")
            
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
