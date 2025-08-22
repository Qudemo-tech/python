#!/usr/bin/env python3
"""
Enhanced Knowledge Integration System
Handles Pinecone vector database operations for knowledge storage and retrieval
"""

import requests
import json
import os
from datetime import datetime

# Configuration
NODE_API_URL = os.getenv('NODE_API_URL', 'http://localhost:3001')
PYTHON_API_URL = os.getenv('PYTHON_API_URL', 'http://localhost:5000')

def process_qudemo_knowledge_source(qudemo_id, source_id, source_url, source_type, title):
    """
    Process a knowledge source for a specific qudemo
    """
    try:
        print(f"🔍 Processing knowledge source {source_id} for qudemo {qudemo_id}")
        
        # Call Python backend for processing
        python_payload = {
            "source_url": source_url,
            "source_type": source_type,
            "title": title,
            "qudemo_id": str(qudemo_id),
            "source_id": str(source_id)
        }
        
        response = requests.post(
            f"{PYTHON_API_URL}/process_knowledge_source",
            json=python_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Update source status in Node.js database
            update_payload = {
                "status": "processed",
                "processed_at": datetime.now().isoformat(),
                "metadata": result.get("metadata", {})
            }
            
            update_response = requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if update_response.status_code == 200:
                print(f"✅ Successfully processed knowledge source {source_id}")
                return True
            else:
                print(f"❌ Failed to update source status: {update_response.text}")
                return False
        else:
            print(f"❌ Python processing failed: {response.text}")
            
            # Update source status to failed
            update_payload = {
                "status": "failed",
                "processed_at": datetime.now().isoformat(),
                "metadata": {"error": response.text}
            }
            
            requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            return False
            
    except Exception as e:
        print(f"❌ Error processing knowledge source: {str(e)}")
        
        # Update source status to failed
        try:
            update_payload = {
                "status": "failed",
                "processed_at": datetime.now().isoformat(),
                "metadata": {"error": str(e)}
            }
            
            requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
        except:
            pass
            
        return False

def get_pending_knowledge_sources():
    """
    Get all pending knowledge sources from Node.js backend
    """
    try:
        response = requests.get(
            f"{NODE_API_URL}/api/knowledge/pending",
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"❌ Failed to get pending sources: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting pending sources: {str(e)}")
        return []

def process_all_pending_sources():
    """
    Process all pending knowledge sources
    """
    pending_sources = get_pending_knowledge_sources()
    
    print(f"🔍 Found {len(pending_sources)} pending knowledge sources")
    
    for source in pending_sources:
        process_qudemo_knowledge_source(
            source["qudemo_id"],
            source["id"],
            source["source_url"],
            source["source_type"],
            source["title"]
        )

if __name__ == "__main__":
    process_all_pending_sources()
