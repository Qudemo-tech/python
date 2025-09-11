#!/usr/bin/env python3
"""
Manual cleanup script for orphaned QuDemo data
This script will force delete specific QuDemo data from Pinecone
"""

import requests
import json
import sys

# Configuration
PYTHON_API_URL = "http://localhost:5001"
COMPANY_NAME = "nmvbhjb"

# QuDemo IDs that should be deleted (from the logs)
ORPHANED_QUDEMO_IDS = [
    "3146d258-67d7-4c3e-ae5f-2af1d995eb99",
    "e987fd8d-c0d7-4a58-b3e6-ddccefa53311", 
    "e060d659-71ea-4883-b64d-d4b7d7df185e",
    "7ce09e59-30ad-4594-8929-77f406a03a75",
    "eb7864b2-2f57-4a63-8e60-8178803588ac"
]

def cleanup_qudemo(company_name: str, qudemo_id: str) -> dict:
    """Clean up a specific QuDemo"""
    try:
        url = f"{PYTHON_API_URL}/force-cleanup-qudemo/{company_name}/{qudemo_id}"
        print(f"🧹 Cleaning up QuDemo: {qudemo_id}")
        
        response = requests.delete(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message', 'Cleanup completed')}")
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return {"success": False, "error": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main cleanup function"""
    print("🚀 Starting orphaned QuDemo cleanup...")
    print(f"📊 Company: {COMPANY_NAME}")
    print(f"📊 Orphaned QuDemos: {len(ORPHANED_QUDEMO_IDS)}")
    print("-" * 50)
    
    results = []
    total_vectors_deleted = 0
    
    for qudemo_id in ORPHANED_QUDEMO_IDS:
        result = cleanup_qudemo(COMPANY_NAME, qudemo_id)
        results.append({
            'qudemo_id': qudemo_id,
            'result': result
        })
        
        if result.get('success'):
            data = result.get('data', {})
            vectors_deleted = data.get('total_vectors_deleted', 0)
            total_vectors_deleted += vectors_deleted
            print(f"   📊 Vectors deleted: {vectors_deleted}")
        
        print("-" * 30)
    
    # Summary
    print("\n🎉 CLEANUP SUMMARY")
    print(f"📊 Total QuDemos processed: {len(ORPHANED_QUDEMO_IDS)}")
    print(f"📊 Total vectors deleted: {total_vectors_deleted}")
    
    successful = sum(1 for r in results if r['result'].get('success'))
    print(f"✅ Successful cleanups: {successful}")
    print(f"❌ Failed cleanups: {len(ORPHANED_QUDEMO_IDS) - successful}")
    
    # Save results
    with open('cleanup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: cleanup_results.json")

if __name__ == "__main__":
    main()
