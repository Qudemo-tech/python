"""
Avatar Video Processor - Background task for generating avatar videos with HeyGen
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class AvatarVideoProcessor:
    """Process avatar video generation in background"""
    
    def __init__(self, heygen_service, gcs_service, supabase_client=None):
        """
        Initialize processor
        
        Args:
            heygen_service: HeyGen service instance
            gcs_service: Google Cloud Storage service
            supabase_client: Optional Supabase client for database updates
        """
        self.heygen = heygen_service
        self.gcs = gcs_service
        self.supabase = supabase_client
        
        logger.info("✅ Avatar Video Processor initialized")
    
    async def process_faq_videos(self, company_name: str, qudemo_id: str, 
                                presenter_photo_url: str, faqs: List[Dict], 
                                voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all FAQ videos concurrently
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            presenter_photo_url: GCS URL of presenter photo
            faqs: List of FAQ dictionaries with id, question, answer
            voice_id: Optional HeyGen voice ID (uses default if not provided)
            
        Returns:
            dict: Processing summary with success/failure counts
        """
        try:
            logger.info(f"🎬 Starting avatar video generation for {len(faqs)} FAQs")
            logger.info(f"📍 QuDemo: {company_name}/{qudemo_id}")
            
            # Update QuDemo status to 'processing' before starting
            if self.supabase:
                try:
                    self.supabase.table('qudemos_new').update({
                        'avatar_generation_status': 'processing',
                        'avatar_videos_total': len(faqs),
                        'avatar_videos_completed': 0,
                        'avatar_generation_started_at': datetime.utcnow().isoformat()
                    }).eq('id', qudemo_id).execute()
                    logger.info(f"✅ Updated QuDemo status to 'processing' with {len(faqs)} total videos")
                except Exception as e:
                    logger.error(f"❌ Failed to update QuDemo status: {e}")
            
            # Step 1: Upload presenter photo to HeyGen
            logger.info(f"📤 Step 1: Uploading presenter photo to HeyGen...")
            logger.info(f"📷 GCS URL: {presenter_photo_url}")
            image_key = self.heygen.upload_presenter_photo(presenter_photo_url)
            
            if not image_key:
                logger.error("❌ Failed to upload presenter photo - aborting video generation")
                
                # Update status to failed
                if self.supabase:
                    try:
                        self.supabase.table('qudemos_new').update({
                            'avatar_generation_status': 'failed'
                        }).eq('id', qudemo_id).execute()
                    except Exception as e:
                        logger.error(f"❌ Failed to update QuDemo status to failed: {e}")
                
                return {
                    "success": False,
                    "error": "Failed to upload presenter photo to HeyGen",
                    "videos_generated": 0,
                    "videos_failed": len(faqs)
                }
            
            logger.info(f"✅ Presenter photo uploaded, image_key: {image_key}")
            logger.info(f"🎤 Voice ID for videos: {voice_id if voice_id else 'Default (will use HeyGen service default)'}")
            
            # Wait for HeyGen to process the uploaded image
            logger.info(f"⏳ Waiting 15 seconds for HeyGen to process the image...")
            await asyncio.sleep(15)
            
            # Step 2: Create placeholder records in avatar_videos table for progress tracking
            logger.info(f"📝 Step 2: Creating placeholder records for progress tracking...")
            if self.supabase:
                try:
                    for faq in faqs:
                        faq_id = faq.get('id', 'unknown')
                        question = faq.get('question', '')
                        answer = faq.get('answer', '')
                        
                        # Check if record already exists
                        existing = self.supabase.table('avatar_videos').select('id').eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                        
                        if not existing.data or len(existing.data) == 0:
                            # Create placeholder record with 'processing' status
                            self.supabase.table('avatar_videos').insert({
                                "qudemo_id": qudemo_id,
                                "faq_id": faq_id,
                                "question": question,
                                "answer": answer,
                                "status": "processing",
                                "created_at": datetime.utcnow().isoformat()
                            }).execute()
                            logger.info(f"✅ Created placeholder record for {faq_id}")
                        else:
                            logger.info(f"ℹ️ Record already exists for {faq_id}, skipping")
                except Exception as e:
                    logger.error(f"❌ Failed to create placeholder records: {e}")
            
            # Step 3: Generate videos concurrently
            logger.info(f"🎥 Step 3: Generating {len(faqs)} videos concurrently...")
            
            tasks = []
            for faq in faqs:
                task = self.process_single_faq_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    image_key=image_key,
                    faq=faq,
                    voice_id=voice_id
                )
                tasks.append(task)
            
            # Run all video generation tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            success_count = sum(1 for r in results if r and isinstance(r, dict) and r.get('success'))
            failed_count = len(faqs) - success_count
            
            logger.info(f"✅ Video generation complete: {success_count} succeeded, {failed_count} failed")
            
            # Step 4: Update QuDemo status based on results
            if self.supabase:
                try:
                    if success_count == 0:
                        # All videos failed
                        self.supabase.table('qudemos_new').update({
                            'avatar_generation_status': 'failed',
                            'avatar_videos_completed': 0
                        }).eq('id', qudemo_id).execute()
                        logger.info(f"❌ All videos failed for QuDemo {qudemo_id}")
                    elif success_count == len(faqs):
                        # All videos succeeded
                        self.supabase.table('qudemos_new').update({
                            'avatar_generation_status': 'completed',
                            'has_avatar_videos': True,
                            'avatar_videos_completed': success_count,
                            'avatar_videos_total': len(faqs),
                            'avatar_generation_completed_at': datetime.utcnow().isoformat()
                        }).eq('id', qudemo_id).execute()
                        logger.info(f"✅ All videos completed for QuDemo {qudemo_id}")
                    else:
                        # Some videos succeeded, some failed
                        self.supabase.table('qudemos_new').update({
                            'avatar_generation_status': 'completed',  # Mark as completed even with some failures
                            'has_avatar_videos': True,
                            'avatar_videos_completed': success_count,
                            'avatar_videos_total': len(faqs)
                        }).eq('id', qudemo_id).execute()
                        logger.info(f"⚠️ Partial completion: {success_count}/{len(faqs)} videos for QuDemo {qudemo_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to update QuDemo flag: {e}")
            
            return {
                "success": True,
                "videos_generated": success_count,
                "videos_failed": failed_count,
                "total_faqs": len(faqs),
                "results": [r for r in results if not isinstance(r, Exception)]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in avatar video processing: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            
            # Update status to failed on exception
            if self.supabase:
                try:
                    self.supabase.table('qudemos_new').update({
                        'avatar_generation_status': 'failed'
                    }).eq('id', qudemo_id).execute()
                except Exception as update_error:
                    logger.error(f"❌ Failed to update QuDemo status to failed: {update_error}")
            
            return {
                "success": False,
                "error": str(e),
                "videos_generated": 0,
                "videos_failed": len(faqs)
            }
    
    async def process_single_faq_video(self, company_name: str, qudemo_id: str, 
                                      image_key: str, faq: Dict, 
                                      voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single FAQ video
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            image_key: HeyGen image_key (with /original suffix removed)
            faq: FAQ dictionary with id, question, answer
            voice_id: Optional HeyGen voice ID (uses default if not provided)
            
        Returns:
            dict: Result with success status and video info
        """
        faq_id = faq.get('id', 'unknown')
        question = faq.get('question', '')
        answer = faq.get('answer', '')
        
        try:
            logger.info(f"🎬 Processing FAQ: {faq_id}")
            
            # Step 1: Generate video with HeyGen
            video_title = f"{company_name} - {faq_id}"
            video_id = self.heygen.generate_video(
                image_key=image_key,
                script=answer,  # HeyGen service will truncate to 1000 chars
                video_title=video_title,
                voice_id=voice_id  # Pass voice_id to HeyGen
            )
            
            if not video_id:
                logger.error(f"❌ Failed to start video generation for {faq_id}")
                return {"success": False, "faq_id": faq_id, "error": "Failed to start generation"}
            
            logger.info(f"⏳ Waiting for video {faq_id} to complete...")
            
            # Step 2: Wait for video to complete (async, non-blocking for other videos)
            video_url = await self.heygen.wait_for_video_completion(video_id, max_wait_seconds=600)
            
            if not video_url:
                logger.error(f"❌ Video generation failed or timed out for {faq_id}")
                return {"success": False, "faq_id": faq_id, "error": "Generation failed or timed out"}
            
            # Step 3: Download video from HeyGen
            logger.info(f"⬇️ Downloading video for {faq_id}...")
            video_data = self.heygen.download_video(video_url)
            
            if not video_data:
                logger.error(f"❌ Failed to download video for {faq_id}")
                return {"success": False, "faq_id": faq_id, "error": "Failed to download video"}
            
            # Step 4: Upload video to GCS
            logger.info(f"☁️ Uploading video to GCS for {faq_id}...")
            
            # Get company bucket
            bucket = self.gcs._get_company_bucket(company_name)
            
            # Create blob path: company_name/qudemo_id/avatar_videos/faq_id.mp4
            video_path = f"{company_name}/{qudemo_id}/avatar_videos/{faq_id}.mp4"
            blob = bucket.blob(video_path)
            
            # Upload video to GCS
            blob.upload_from_string(video_data, content_type='video/mp4')
            
            # Make the video publicly accessible
            try:
                blob.make_public()
                logger.info(f"✅ Video made public: {video_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not make video public (might already be public via bucket policy): {e}")
            
            # Get public URL
            gcs_video_url = f"https://storage.googleapis.com/{bucket.name}/{video_path}"
            
            logger.info(f"✅ Video uploaded to GCS: {gcs_video_url}")
            
            # Step 5: Update database record to completed status
            if self.supabase:
                try:
                    video_record = {
                        "video_url": gcs_video_url,
                        "heygen_video_id": video_id,
                        "status": "completed",
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Update existing record (should always exist from Step 2)
                    result = self.supabase.table('avatar_videos').update(video_record).eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                    
                    if result.data and len(result.data) > 0:
                        logger.info(f"✅ Updated avatar_videos record for {faq_id} to completed")
                        
                        # Update QuDemo progress count
                        completed_count = self.supabase.table('avatar_videos').select('id').eq('qudemo_id', qudemo_id).eq('status', 'completed').execute()
                        if completed_count.data:
                            self.supabase.table('qudemos_new').update({
                                'avatar_videos_completed': len(completed_count.data)
                            }).eq('id', qudemo_id).execute()
                            logger.info(f"📊 Progress: {len(completed_count.data)} videos completed")
                    else:
                        # Fallback: insert if update failed (record might not exist)
                        logger.warning(f"⚠️ Record not found for {faq_id}, inserting new record")
                        self.supabase.table('avatar_videos').insert({
                            "qudemo_id": qudemo_id,
                            "faq_id": faq_id,
                            "question": question,
                            "answer": answer,
                            "video_url": gcs_video_url,
                            "heygen_video_id": video_id,
                            "status": "completed",
                            "created_at": datetime.utcnow().isoformat()
                        }).execute()
                        logger.info(f"✅ Inserted avatar_videos record for {faq_id}")
                        
                except Exception as db_error:
                    logger.error(f"❌ Failed to update database: {db_error}")
                    # Don't fail the whole process if DB update fails
            
            # Step 6: Update FAQ file with video URL for instant retrieval
            logger.info(f"📝 Step 6: Updating FAQ file with video URL for {faq_id}...")
            try:
                self._update_faq_with_video_url(company_name, qudemo_id, faq_id, gcs_video_url)
                logger.info(f"✅ FAQ file updated with video URL")
            except Exception as update_error:
                logger.error(f"⚠️ Failed to update FAQ file: {update_error}")
                # Don't fail the whole process if FAQ update fails
            
            logger.info(f"🎉 Successfully processed video for {faq_id}")
            
            return {
                "success": True,
                "faq_id": faq_id,
                "video_url": gcs_video_url,
                "heygen_video_id": video_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing video for {faq_id}: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "faq_id": faq_id,
                "error": str(e)
            }
    
    def _update_faq_with_video_url(self, company_name: str, qudemo_id: str, faq_id: str, video_url: str):
        """
        Update FAQ file in GCS to include video URL for instant retrieval
        This enables the new simplified Q&A architecture
        """
        try:
            import json
            
            # Get company bucket
            bucket = self.gcs._get_company_bucket(company_name)
            
            # Load existing FAQ file
            faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
            blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
            
            if not blob.exists():
                logger.warning(f"⚠️ FAQ file not found: {faq_filename}")
                return
            
            # Download and parse
            faqs_content = blob.download_as_text()
            faqs_data = json.loads(faqs_content)
            
            # Find and update the FAQ
            updated = False
            for faq in faqs_data.get('faqs', []):
                if faq.get('id') == faq_id:
                    faq['video_url'] = video_url
                    faq['video_status'] = 'completed'
                    updated = True
                    logger.info(f"✅ Updated FAQ {faq_id} with video URL")
                    break
            
            if not updated:
                logger.warning(f"⚠️ FAQ {faq_id} not found in FAQ file")
                return
            
            # Save back to GCS
            blob.upload_from_string(
                json.dumps(faqs_data, indent=2),
                content_type='application/json'
            )
            logger.info(f"✅ FAQ file saved to GCS")
            
        except Exception as e:
            logger.error(f"❌ Error updating FAQ file: {e}")
            raise
    
    async def process_single_faq_video_update(self, company_name: str, qudemo_id: str,
                                              presenter_photo_url: str, faq: Dict) -> Dict[str, Any]:
        """
        Regenerate a single FAQ video (for editing)
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            presenter_photo_url: GCS URL of presenter photo
            faq: FAQ dictionary with id, question, answer
            
        Returns:
            dict: Processing result
        """
        try:
            faq_id = faq.get('id')
            logger.info(f"🔄 Regenerating video for FAQ {faq_id}")
            logger.info(f"📍 QuDemo: {company_name}/{qudemo_id}")
            
            # Step 1: Upload presenter photo to HeyGen (or reuse existing image_key if available)
            logger.info(f"📤 Uploading presenter photo to HeyGen...")
            logger.info(f"📷 GCS URL: {presenter_photo_url}")
            
            image_key = self.heygen.upload_presenter_photo(presenter_photo_url)
            
            if not image_key:
                logger.error(f"❌ Failed to upload presenter photo for {faq_id}")
                return {"success": False, "faq_id": faq_id, "error": "Failed to upload presenter photo"}
            
            logger.info(f"✅ Presenter photo uploaded, image_key: {image_key}")
            
            # Wait a bit for HeyGen to process the image
            logger.info(f"⏳ Waiting 15 seconds for HeyGen to process the image...")
            await asyncio.sleep(15)
            
            # Step 2: Generate the video
            logger.info(f"🎬 Generating new video for FAQ {faq_id}")
            
            # Get voice_id from qudemo (if available)
            voice_id_to_use = None
            try:
                qudemo_data = self.supabase.table('qudemos_new').select('voice_id').eq('id', qudemo_id).execute()
                if qudemo_data.data and len(qudemo_data.data) > 0:
                    voice_id_to_use = qudemo_data.data[0].get('voice_id')
                    logger.info(f"🎤 Using voice ID from qudemo: {voice_id_to_use}")
            except Exception as voice_error:
                logger.warning(f"⚠️ Could not fetch voice_id, using default: {voice_error}")
            
            # Process the single FAQ video
            result = await self.process_single_faq_video(
                company_name=company_name,
                qudemo_id=qudemo_id,
                image_key=image_key,
                faq=faq,
                voice_id=voice_id_to_use
            )
            
            if result.get('success'):
                logger.info(f"✅ Successfully regenerated video for FAQ {faq_id}")
                
                # Update the existing avatar_videos record (don't insert new)
                if self.supabase:
                    try:
                        update_data = {
                            'answer': faq.get('answer'),
                            'video_url': result.get('video_url'),
                            'heygen_video_id': result.get('heygen_video_id'),
                            'status': 'completed',
                            'updated_at': datetime.now().isoformat()
                        }
                        
                        response = self.supabase.table('avatar_videos')\
                            .update(update_data)\
                            .eq('qudemo_id', qudemo_id)\
                            .eq('faq_id', faq_id)\
                            .execute()
                        
                        logger.info(f"✅ Updated avatar_videos record for {faq_id}")
                    except Exception as db_error:
                        logger.error(f"❌ Error updating database for {faq_id}: {db_error}")
                
                return {"success": True, "faq_id": faq_id, "video_url": result.get('video_url')}
            else:
                logger.error(f"❌ Failed to regenerate video for {faq_id}")
                return {"success": False, "faq_id": faq_id, "error": result.get('error')}
            
        except Exception as e:
            logger.error(f"❌ Error regenerating video for {faq_id}: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "faq_id": faq_id,
                "error": str(e)
            }

