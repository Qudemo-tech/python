#!/usr/bin/env python3
"""
Very simple test application for deployment
"""
from fastapi import FastAPI
import os

# Create FastAPI app
app = FastAPI(title="QuDemo Simple Test")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "QuDemo Python Backend - Simple Test", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "QuDemo Python Backend is running",
        "version": "simple-1.0.0"
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "status": "success",
        "message": "Simple test endpoint working"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5001))
    print(f"Starting simple test server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 