"""Force fix Qudemo brand name - FINAL VERSION"""
import json
from google.cloud import storage

# Create fresh client
client = storage.Client.from_service_account_json('service-account-key.json')
bucket = client.bucket('qudemo-sample-qudemo')
blob_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json'

print("=" * 80)
print("🔧 FORCE FIXING BRAND NAME: Kudomo → Qudemo")
print("=" * 80)

# Download fresh
blob = bucket.blob(blob_path)
blob.reload()
content = blob.download_as_bytes().decode('utf-8')

print(f"\nBEFORE:")
print(f"  'Kudomo' count: {content.count('Kudomo')}")
print(f"  'Qudemo' count: {content.count('Qudemo')}")

# Parse JSON
data = json.loads(content)

# Fix ALL text fields
for faq in data['faqs']:
    # Fix question
    if 'question' in faq:
        faq['question'] = faq['question'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')
    
    # Fix answer
    if 'answer' in faq:
        faq['answer'] = faq['answer'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')

# Convert back to JSON
fixed_json = json.dumps(data, indent=2, ensure_ascii=False)

print(f"\nAFTER:")
print(f"  'Kudomo' count: {fixed_json.count('Kudomo')}")
print(f"  'Qudemo' count: {fixed_json.count('Qudemo')}")

# Delete old blob
print(f"\n🗑️  Deleting old file...")
blob.delete()

# Wait
import time
time.sleep(1)

# Upload new blob
print(f"📤 Uploading new file...")
new_blob = bucket.blob(blob_path)
new_blob.upload_from_string(fixed_json, content_type='application/json')
new_blob.cache_control = 'no-cache'
new_blob.patch()

# Wait for propagation
time.sleep(2)

# Verify
print(f"\n✅ Verifying...")
verify_blob = bucket.blob(blob_path)
verify_blob.reload()
verify_content = verify_blob.download_as_bytes().decode('utf-8')

kudomo_count = verify_content.count('Kudomo')
qudemo_count = verify_content.count('Qudemo')

print(f"\nVERIFICATION:")
print(f"  'Kudomo' count: {kudomo_count}")
print(f"  'Qudemo' count: {qudemo_count}")

if kudomo_count == 0:
    print("\n" + "=" * 80)
    print("✅✅✅ SUCCESS! All 'Kudomo' replaced with 'Qudemo'!")
    print("=" * 80)
    
    # Show sample questions
    verify_data = json.loads(verify_content)
    print("\n📋 Sample questions:")
    for i in range(min(3, len(verify_data['faqs']))):
        print(f"   {i+1}. {verify_data['faqs'][i]['question']}")
    
    print("\n⚠️ IMPORTANT: RESTART THE PYTHON BACKEND to load the fixed file!")
    print("=" * 80)
else:
    print(f"\n❌ Still has {kudomo_count} 'Kudomo' references!")

