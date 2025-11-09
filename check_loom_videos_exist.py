"""Check if Loom video files exist in GCS"""
from google.cloud import storage

client = storage.Client.from_service_account_json('service-account-key.json')
bucket = client.bucket('qudemo-sample-qudemo')

# Check if the 7 Loom videos exist
base_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/avatar_videos'

print("=" * 80)
print("🔍 CHECKING IF LOOM VIDEO FILES EXIST IN GCS")
print("=" * 80)
print(f"Bucket: {bucket.name}")
print(f"Path: {base_path}")
print("=" * 80)
print()

for i in range(1, 8):
    faq_id = f"faq_kudomo_{i:03d}"
    video_path = f"{base_path}/{faq_id}.mp4"
    
    blob = bucket.blob(video_path)
    
    if blob.exists():
        blob.reload()
        print(f"✅ {i}. {faq_id}.mp4")
        print(f"   Size: {blob.size / 1024 / 1024:.2f} MB")
        print(f"   URL: {blob.public_url[:80]}...")
        print(f"   Content-Type: {blob.content_type}")
    else:
        print(f"❌ {i}. {faq_id}.mp4 - NOT FOUND")
    print()

print("=" * 80)

