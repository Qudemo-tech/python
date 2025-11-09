"""Check if the 2 new Loom videos exist in GCS"""
from google.cloud import storage

client = storage.Client.from_service_account_json('service-account-key.json')
bucket = client.bucket('qudemo-sample-qudemo')

base_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/avatar_videos'

new_videos = [
    'faq_loom_create_qudemo.mp4',
    'faq_loom_view_interactions.mp4'
]

print("=" * 80)
print("🔍 CHECKING NEW LOOM VIDEOS IN GCS")
print("=" * 80)

for video_name in new_videos:
    video_path = f"{base_path}/{video_name}"
    blob = bucket.blob(video_path)
    
    if blob.exists():
        blob.reload()
        print(f"\n✅ {video_name}")
        print(f"   Size: {blob.size / 1024 / 1024:.2f} MB")
        print(f"   URL: {blob.public_url}")
        print(f"   Content-Type: {blob.content_type}")
    else:
        print(f"\n❌ {video_name} - NOT FOUND")

print("\n" + "=" * 80)

