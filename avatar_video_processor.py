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
                                presenter_photo_url: str, faqs: List[Dict]) -> Dict[str, Any]:
        """
        Process all FAQ videos concurrently
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            presenter_photo_url: GCS URL of presenter photo
            faqs: List of FAQ dictionaries with id, question, answer
            
        Returns:
            dict: Processing summary with success/failure counts
        """
        try:
            logger.info(f"🎬 Starting avatar video generation for {len(faqs)} FAQs")
            logger.info(f"📍 QuDemo: {company_name}/{qudemo_id}")
            
            # Step 1: Upload presenter photo to HeyGen
            logger.info(f"📤 Step 1: Uploading presenter photo to HeyGen...")
            logger.info(f"📷 GCS URL: {presenter_photo_url}")
            image_key = self.heygen.upload_presenter_photo(presenter_photo_url)
            
            if not image_key:
                logger.error("❌ Failed to upload presenter photo - aborting video generation")
                return {
                    "success": False,
                    "error": "Failed to upload presenter photo to HeyGen",
                    "videos_generated": 0,
                    "videos_failed": len(faqs)
                }
            
            logger.info(f"✅ Presenter photo uploaded, image_key: {image_key}")
            
            # Wait for HeyGen to process the uploaded image
            logger.info(f"⏳ Waiting 15 seconds for HeyGen to process the image...")
            await asyncio.sleep(15)
            
            # Step 2: Generate videos concurrently
            logger.info(f"🎥 Step 2: Generating {len(faqs)} videos concurrently...")
            
            tasks = []
            for faq in faqs:
                task = self.process_single_faq_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    image_key=image_key,
                    faq=faq
                )
                tasks.append(task)
            
            # Run all video generation tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            success_count = sum(1 for r in results if r and isinstance(r, dict) and r.get('success'))
            failed_count = len(faqs) - success_count
            
            logger.info(f"✅ Video generation complete: {success_count} succeeded, {failed_count} failed")
            
            # Step 3: Update QuDemo has_avatar_videos flag if any videos succeeded
            if success_count > 0 and self.supabase:
                try:
                    self.supabase.table('qudemos_new').update({
                        'has_avatar_videos': True
                    }).eq('id', qudemo_id).execute()
                    logger.info(f"✅ Updated QuDemo {qudemo_id} with has_avatar_videos=true")
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
            return {
                "success": False,
                "error": str(e),
                "videos_generated": 0,
                "videos_failed": len(faqs)
            }
    
    async def process_single_faq_video(self, company_name: str, qudemo_id: str, 
                                      image_key: str, faq: Dict) -> Dict[str, Any]:
        """
        Process a single FAQ video
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            image_key: HeyGen image_key (with /original suffix removed)
            faq: FAQ dictionary with id, question, answer
            
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
                video_title=video_title
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
            
            # Step 5: Store in database (avatar_videos table)
            if self.supabase:
                try:
                    video_record = {
                        "qudemo_id": qudemo_id,
                        "faq_id": faq_id,
                        "question": question,
                        "answer": answer,
                        "video_url": gcs_video_url,
                        "heygen_video_id": video_id,
                        "status": "completed",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    # Check if record exists
                    existing = self.supabase.table('avatar_videos').select('id').eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                    
                    if existing.data and len(existing.data) > 0:
                        # Update existing
                        self.supabase.table('avatar_videos').update(video_record).eq('id', existing.data[0]['id']).execute()
                        logger.info(f"✅ Updated avatar_videos record for {faq_id}")
                    else:
                        # Insert new
                        self.supabase.table('avatar_videos').insert(video_record).execute()
                        logger.info(f"✅ Inserted avatar_videos record for {faq_id}")
                        
                except Exception as db_error:
                    logger.error(f"❌ Failed to store in database: {db_error}")
                    # Don't fail the whole process if DB update fails
            
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

