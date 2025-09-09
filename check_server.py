#!/usr/bin/env python3
"""
Check if server is using the right code
"""

import requests
import json

def check_server():
    # Test health endpoint first
    try:
        health_url = "http://localhost:5001/health"
        response = requests.get(health_url)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test a simple question
    url = "http://localhost:5001/ask/mycom/70da2e6e-6466-4f35-a985-ac4ccf992f23"
    data = {
        "question": "test question"
    }
    
    try:
        print("\n🧪 Testing simple question")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer source: {result.get('answer_source', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            print(f"Error: {result.get('error', 'none')}")
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    check_server()
