"""Download and verify GCS FAQ file"""
import json
from google.cloud import storage

storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')

# Download and check
print("📥 Downloading FAQ file from GCS...")
faq_text = blob.download_as_text()

# Check for brand name
if 'Kudomo' in faq_text:
    print("❌ STILL HAS 'Kudomo' in file!")
    print(f"   Count: {faq_text.count('Kudomo')}")
elif 'Qudemo' in faq_text:
    print("✅ File has 'Qudemo' - brand name is correct!")
    print(f"   Count: {faq_text.count('Qudemo')}")

# Show a sample
faq_data = json.loads(faq_text)
print(f"\n📋 Sample FAQ (first one):")
faq = faq_data['faqs'][0]
print(f"   ID: {faq.get('id')}")
print(f"   Question: {faq['question']}")
print(f"   has_avatar_video: {faq.get('has_avatar_video')}")
print(f"   video_status: {faq.get('video_status', 'MISSING')}")
print(f"   video_url exists: {'YES' if faq.get('video_url') else 'NO'}")

