#!/usr/bin/env python3
"""
Test version of main.py for deployment testing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="QuDemo Video Processing API - Test")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "QuDemo Python Backend - Test Version", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "QuDemo Python Backend is running",
        "version": "test-1.0.0"
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint for basic functionality"""
    try:
        # Test basic imports
        import numpy as np
        import requests
        import yt_dlp
        
        return {
            "status": "success",
            "message": "All basic imports working",
            "numpy_version": np.__version__,
            "yt_dlp_version": yt_dlp.version.__version__
        }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Import error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

@app.post("/test-video")
async def test_video_processing():
    """Test video processing endpoint (simplified)"""
    try:
        # Test if we can import video processing modules
        import yt_dlp
        
        return {
            "status": "success",
            "message": "Video processing modules available",
            "yt_dlp_available": True
        }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Video processing import error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting test server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 