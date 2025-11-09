"""Verify GCS file with no caching"""
import json
from google.cloud import storage
import time
import hashlib

# Initialize FRESH client
storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json'

print("=" * 80)
print("🔍 FRESH VERIFICATION (NO CACHE)")
print("=" * 80)
print(f"Blob path: {blob_path}")
print(f"Bucket: {bucket.name}")
print()

# Get blob and reload metadata
blob = bucket.blob(blob_path)
blob.reload()

print(f"Blob exists: {blob.exists()}")
print(f"Content type: {blob.content_type}")
print(f"Size: {blob.size} bytes")
print(f"Updated: {blob.updated}")
print(f"MD5: {blob.md5_hash}")
print()

# Download with explicit no-cache
print("📥 Downloading (no cache)...")
content = blob.download_as_bytes().decode('utf-8')

print(f"Downloaded {len(content)} bytes")
print()

# Check brand names
kudomo_count = content.count('Kudomo')
qudemo_count = content.count('Qudemo')

print("=" * 80)
if kudomo_count == 0:
    print("✅ SUCCESS! NO 'Kudomo' found!")
    print(f"✅ 'Qudemo' count: {qudemo_count}")
    
    data = json.loads(content)
    print("\n📋 Sample questions:")
    for i, faq in enumerate(data['faqs'][:3], 1):
        print(f"   {i}. {faq['question']}")
else:
    print(f"❌ STILL HAS 'Kudomo': {kudomo_count}")
    print(f"   'Qudemo' count: {qudemo_count}")
    
    print("\n🔍 Sample content:")
    lines = content.split('\n')
    for i, line in enumerate(lines[:30], 1):
        if 'question' in line.lower() or 'answer' in line.lower():
            print(f"   Line {i}: {line[:80]}")

print("=" * 80)

