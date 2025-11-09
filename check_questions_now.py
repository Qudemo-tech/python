"""Check what questions are actually in GCS RIGHT NOW"""
import json
from google.cloud import storage

client = storage.Client.from_service_account_json('service-account-key.json')
bucket = client.bucket('qudemo-sample-qudemo')
blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')

# Force reload
blob.reload()

# Download as bytes
content = blob.download_as_bytes().decode('utf-8')
data = json.loads(content)

print("=" * 80)
print("📋 QUESTIONS IN GCS RIGHT NOW:")
print("=" * 80)
print(f"Blob updated: {blob.updated}")
print(f"Blob MD5: {blob.md5_hash}")
print()

for i, faq in enumerate(data['faqs'], 1):
    print(f"{i}. {faq['question']}")
    print(f"   ID: {faq['id']}")
    print(f"   has_avatar_video: {faq.get('has_avatar_video')}")
    print(f"   video_status: {faq.get('video_status', 'MISSING')}")
    print()

# Check brand name
kudomo_count = content.count('Kudomo')
qudemo_count = content.count('Qudemo')

print("=" * 80)
print(f"Brand name check:")
print(f"  'Kudomo' count: {kudomo_count}")
print(f"  'Qudemo' count: {qudemo_count}")
print("=" * 80)

if kudomo_count > 0:
    print("\n❌ PROBLEM: File still has 'Kudomo'!")
else:
    print("\n✅ File has correct 'Qudemo' brand name")

