"""Fresh verification with cache busting"""
import json
from google.cloud import storage
import time

# Create FRESH client
print("Creating FRESH GCS client...")
client = storage.Client.from_service_account_json('service-account-key.json')

bucket = client.bucket('qudemo-sample-qudemo')
blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')

# Force reload metadata
blob.reload()

print(f"Blob MD5: {blob.md5_hash}")
print(f"Blob updated: {blob.updated}")
print()

# Download as bytes to bypass any text caching
content_bytes = blob.download_as_bytes()
content_text = content_bytes.decode('utf-8')

faq_data = json.loads(content_text)

print("📋 First 3 FAQs video_status:")
for i in range(min(3, len(faq_data['faqs']))):
    faq = faq_data['faqs'][i]
    print(f"\n{i+1}. {faq['id']}")
    print(f"   has_avatar_video: {faq.get('has_avatar_video')}")
    print(f"   video_status: {faq.get('video_status', '❌ MISSING')}")
    print(f"   video_url exists: {'YES' if faq.get('video_url') else 'NO'}")

# Check if video_status is in the raw JSON text
if '"video_status"' in content_text:
    print("\n✅ 'video_status' field IS in the JSON file!")
    count = content_text.count('"video_status"')
    print(f"   Found {count} occurrences")
else:
    print("\n❌ 'video_status' field NOT in the JSON file!")

