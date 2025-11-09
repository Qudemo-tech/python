"""
Force replace the FAQ file with corrected brand name
"""

import json
from google.cloud import storage
import time

# Initialize GCS
storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json'

print("=" * 80)
print("🔧 FORCE REPLACING FAQ FILE WITH CORRECT BRAND NAME")
print("=" * 80)

# Step 1: Download and fix
print("\n📥 Downloading current file...")
blob = bucket.blob(blob_path)
faq_data = json.loads(blob.download_as_text())

print(f"   Total FAQs: {len(faq_data['faqs'])}")

# Fix brand names
print("\n🔧 Fixing brand names...")
for faq in faq_data['faqs']:
    faq['question'] = faq['question'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')
    faq['answer'] = faq['answer'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')

# Convert to JSON
fixed_json = json.dumps(faq_data, indent=2, ensure_ascii=False)
print(f"   Fixed JSON - 'Qudemo' count: {fixed_json.count('Qudemo')}")
print(f"   Fixed JSON - 'Kudomo' count: {fixed_json.count('Kudomo')}")

# Step 2: DELETE old file
print("\n🗑️  Deleting old file...")
try:
    blob.delete()
    print("   ✅ Old file deleted")
except:
    print("   ⚠️  File might not exist or already deleted")

# Wait a moment
time.sleep(1)

# Step 3: Upload NEW file
print("\n📤 Uploading new file...")
new_blob = bucket.blob(blob_path)
new_blob.upload_from_string(fixed_json, content_type='application/json')
new_blob.cache_control = 'no-cache, no-store, must-revalidate'
new_blob.patch()
print("   ✅ New file uploaded with no-cache headers")

# Wait for propagation
time.sleep(2)

# Step 4: Verify
print("\n✅ Verifying new file...")
verify_blob = bucket.blob(blob_path)
verify_blob.reload()  # Force reload metadata
verify_text = verify_blob.download_as_text()

print(f"   'Kudomo' count: {verify_text.count('Kudomo')}")
print(f"   'Qudemo' count: {verify_text.count('Qudemo')}")

if verify_text.count('Kudomo') == 0:
    verify_data = json.loads(verify_text)
    print("\n" + "=" * 80)
    print("✅✅✅ SUCCESS! File replaced with correct brand name!")
    print("=" * 80)
    print("\n📋 Verified questions:")
    for i, faq in enumerate(verify_data['faqs'][:5], 1):
        print(f"   {i}. {faq['question']}")
    print("=" * 80)
else:
    print("\n❌ ERROR: File still contains 'Kudomo'!")
    print("   This might be a GCS caching issue. Try again in a few seconds.")

