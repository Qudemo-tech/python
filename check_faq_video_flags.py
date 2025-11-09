"""Check if FAQs have has_avatar_video flag"""
import json
from google.cloud import storage

storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')

faq_data = json.loads(blob.download_as_text())

print("=" * 80)
print("🔍 CHECKING FAQ VIDEO FLAGS")
print("=" * 80)
print(f"Total FAQs: {len(faq_data['faqs'])}")
print()

for i, faq in enumerate(faq_data['faqs'], 1):
    faq_id = faq.get('id', 'NO_ID')
    has_video = faq.get('has_avatar_video', False)
    video_url = faq.get('video_url', 'NO_URL')
    
    status = "✅" if has_video else "❌"
    
    print(f"{i}. {status} FAQ ID: {faq_id}")
    print(f"   Question: {faq['question'][:60]}...")
    print(f"   has_avatar_video: {has_video}")
    print(f"   video_url exists: {'YES' if video_url != 'NO_URL' else 'NO'}")
    if video_url != 'NO_URL':
        print(f"   video_url: {video_url[:80]}...")
    print()

print("=" * 80)

# Check if ALL have video flag
all_have_video = all(faq.get('has_avatar_video', False) for faq in faq_data['faqs'])
if all_have_video:
    print("✅ ALL FAQs have has_avatar_video=true")
else:
    missing = [faq['id'] for faq in faq_data['faqs'] if not faq.get('has_avatar_video', False)]
    print(f"❌ Some FAQs missing has_avatar_video flag: {missing}")

