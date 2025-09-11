#!/usr/bin/env python3
"""
Google Video Intelligence + Speech-to-Text Processor
Hybrid approach using deterministic services for timestamps and shot boundaries
"""

import os
import logging
import asyncio
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import re

# Configure logging first
logger = logging.getLogger(__name__)

# Google Cloud imports with fallback
try:
    # Try different import methods
    try:
        from google.cloud import videointelligence_v1
        from google.cloud import speech_v1
        from google.cloud import storage
        from google.oauth2 import service_account
        GOOGLE_CLOUD_AVAILABLE = True
        logger.info("✅ Google Cloud services imported successfully")
    except ImportError:
        # Try alternative import method
        import google.cloud.videointelligence_v1 as videointelligence_v1
        import google.cloud.speech_v1 as speech_v1
        import google.cloud.storage as storage
        from google.oauth2 import service_account
        GOOGLE_CLOUD_AVAILABLE = True
        logger.info("✅ Google Cloud services imported with alternative method")
except ImportError as e:
    logger.warning(f"⚠️ Google Cloud services not available: {e}")
    logger.warning("⚠️ Hybrid video processing will use fallback methods")
    GOOGLE_CLOUD_AVAILABLE = False
    # Create dummy classes for fallback
    class videointelligence_v1:
        class VideoIntelligenceServiceClient:
            pass
    class speech_v1:
        class SpeechClient:
            pass
    class storage:
        class Client:
            pass
    class service_account:
        class Credentials:
            pass

class GoogleVideoIntelligenceProcessor:
    """Enhanced video processor using Google's Video Intelligence + Speech-to-Text"""
    
    def __init__(self, service_account_path: str = None):
        """Initialize Google Cloud services"""
        try:
            if not GOOGLE_CLOUD_AVAILABLE:
                logger.warning("⚠️ Google Cloud services not available - using fallback mode")
                self.credentials = None
                self.video_client = None
                self.speech_client = None
                self.storage_client = None
                self.bucket_name = None
                self.temp_dir = tempfile.mkdtemp(prefix='qudemo_processing_')
                return
            
            # Initialize credentials
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                self.credentials = credentials
            else:
                # Use default credentials (for production)
                self.credentials = None
            
            # Initialize clients
            self.video_client = videointelligence_v1.VideoIntelligenceServiceClient(credentials=self.credentials)
            self.speech_client = speech_v1.SpeechClient(credentials=self.credentials)
            self.storage_client = storage.Client(credentials=self.credentials)
            
            # Configuration
            self.bucket_name = os.getenv('GCS_BUCKET_NAME', 'qudemo-video-processing')
            self.temp_dir = tempfile.mkdtemp(prefix='qudemo_processing_')
            
            logger.info("✅ Google Video Intelligence Processor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Video Intelligence Processor: {e}")
            logger.warning("⚠️ Falling back to basic mode")
            self.credentials = None
            self.video_client = None
            self.speech_client = None
            self.storage_client = None
            self.bucket_name = None
            self.temp_dir = tempfile.mkdtemp(prefix='qudemo_processing_')
    
    async def process_video_with_deterministic_signals(
        self, 
        video_url: str, 
        company_name: str, 
        qudemo_id: str,
        use_youtube_captions: bool = True
    ) -> Dict:
        """Process video with deterministic signals or fallback to basic processing"""
        if not GOOGLE_CLOUD_AVAILABLE or not self.video_client:
            logger.warning("⚠️ Google Cloud services not available - using fallback processing")
            return await self._fallback_processing(video_url, company_name, qudemo_id)
        """
        Process video using Google's deterministic services for timestamps and shot boundaries
        
        Args:
            video_url: Video URL to process
            company_name: Company name for isolation
            qudemo_id: QuDemo ID for isolation
            use_youtube_captions: Whether to try YouTube captions first
            
        Returns:
            Processing results with deterministic timestamps and shot boundaries
        """
        try:
            logger.info(f"🎬 Processing video with deterministic signals: {video_url}")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # Step 1: Try YouTube captions first (most accurate)
            if use_youtube_captions and self._is_youtube_url(video_url):
                captions_result = await self._extract_youtube_captions(video_url)
                if captions_result['success']:
                    logger.info("✅ Using YouTube captions for timestamps")
                    return await self._process_with_captions(
                        captions_result['data'], video_url, company_name, qudemo_id
                    )
            
            # Step 2: Download video and process with Google services
            logger.info("📥 Downloading video for Google services processing")
            video_path = await self._download_video(video_url)
            
            if not video_path:
                return {
                    'success': False,
                    'error': 'Failed to download video',
                    'video_url': video_url
                }
            
            # Step 3: Upload to GCS for processing
            gcs_uri = await self._upload_to_gcs(video_path, company_name, qudemo_id)
            
            # Step 4: Run Google services in parallel
            logger.info("🔄 Running Google services in parallel...")
            
            # Run Speech-to-Text and Video Intelligence concurrently
            speech_task = self._run_speech_to_text(gcs_uri)
            video_intel_task = self._run_video_intelligence(gcs_uri)
            
            speech_result, video_intel_result = await asyncio.gather(
                speech_task, video_intel_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(speech_result, Exception):
                logger.error(f"❌ Speech-to-Text failed: {speech_result}")
                speech_result = None
            
            if isinstance(video_intel_result, Exception):
                logger.error(f"❌ Video Intelligence failed: {video_intel_result}")
                video_intel_result = None
            
            # Step 5: Combine results and create segments
            combined_result = await self._combine_deterministic_signals(
                speech_result, video_intel_result, video_url, company_name, qudemo_id
            )
            
            # Cleanup
            await self._cleanup_temp_files(video_path, gcs_uri)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"❌ Error in process_video_with_deterministic_signals: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
    
    async def _extract_youtube_captions(self, video_url: str) -> Dict:
        """Extract YouTube captions with exact timestamps"""
        try:
            # Extract video ID
            video_id = self._extract_youtube_id(video_url)
            if not video_id:
                return {'success': False, 'error': 'Invalid YouTube URL'}
            
            # Try to get captions using YouTube API or yt-dlp
            # Note: This is a placeholder - you'll need to implement actual caption extraction
            # For now, return failure to fall back to Google services
            logger.info(f"📺 YouTube captions extraction not implemented for {video_id}")
            return {'success': False, 'error': 'YouTube captions not available'}
            
        except Exception as e:
            logger.error(f"❌ Error extracting YouTube captions: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _download_video(self, video_url: str) -> Optional[str]:
        """Download video to temporary file"""
        try:
            # This is a placeholder - you'll need to implement actual video download
            # For now, return None to indicate download not available
            logger.warning("⚠️ Video download not implemented - using placeholder")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error downloading video: {e}")
            return None
    
    async def _upload_to_gcs(self, video_path: str, company_name: str, qudemo_id: str) -> str:
        """Upload video to Google Cloud Storage with improved bucket handling"""
        try:
            from google.api_core.exceptions import NotFound
            
            # Get or create bucket with proper error handling
            bucket = self._get_or_create_bucket(self.bucket_name)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{company_name}_{qudemo_id}_{timestamp}.mp4"
            blob_name = f"video-processing/{filename}"
            
            # Upload file
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(video_path)
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"✅ Uploaded video to GCS: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"❌ Error uploading to GCS: {e}")
            raise
    
    def _get_or_create_bucket(self, bucket_name: str, location: str = "US"):
        """Get or create GCS bucket with proper error handling"""
        try:
            from google.api_core.exceptions import NotFound
            
            try:
                # Try to get existing bucket
                bucket = self.storage_client.get_bucket(bucket_name)
                logger.info(f"✅ Using existing bucket: {bucket_name}")
                return bucket
            except NotFound:
                # Create new bucket
                bucket = self.storage_client.bucket(bucket_name)
                bucket.location = location
                bucket.storage_class = "STANDARD"  # Cost-effective storage class
                bucket = self.storage_client.create_bucket(bucket)
                logger.info(f"✅ Created new bucket: {bucket_name} in {location}")
                return bucket
                
        except Exception as e:
            logger.error(f"❌ Error getting/creating bucket {bucket_name}: {e}")
            raise
    
    async def _run_speech_to_text(self, gcs_uri: str) -> Dict:
        """Run Google Speech-to-Text for word-level timestamps with optimized config for long videos"""
        try:
            logger.info("🎤 Running Speech-to-Text...")
            
            # Configure Speech-to-Text request optimized for long videos
            config = speech_v1.RecognitionConfig(
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,  # Get word-level timestamps
                use_enhanced=True,  # Use enhanced model
                model="video",  # Optimized for video content
                audio_channel_count=2,  # Handle stereo audio
                enable_separate_recognition_per_channel=False,  # Combine channels
                # Don't force sample_rate_hertz - let service detect
                # encoding=speech_v1.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            )
            
            audio = speech_v1.RecognitionAudio(uri=gcs_uri)
            
            # Run long-running operation
            operation = self.speech_client.long_running_recognize(
                config=config, audio=audio
            )
            
            # Wait for completion
            response = operation.result(timeout=300)  # 5 minute timeout
            
            # Process results
            words = []
            transcript_parts = []
            
            for result in response.results:
                alternative = result.alternatives[0]
                transcript_parts.append(alternative.transcript)
                
                # Extract word-level timestamps
                for word_info in alternative.words:
                    words.append({
                        'word': word_info.word,
                        'start_time': word_info.start_time.total_seconds(),
                        'end_time': word_info.end_time.total_seconds(),
                        'confidence': word_info.confidence
                    })
            
            full_transcript = ' '.join(transcript_parts)
            
            logger.info(f"✅ Speech-to-Text completed: {len(words)} words, {len(full_transcript)} chars")
            
            return {
                'success': True,
                'transcript': full_transcript,
                'words': words,
                'confidence': sum(w['confidence'] for w in words) / len(words) if words else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Speech-to-Text failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _run_video_intelligence(self, gcs_uri: str) -> Dict:
        """Run Google Video Intelligence for shot boundaries and labels"""
        try:
            logger.info("🎥 Running Video Intelligence...")
            
            # Configure Video Intelligence request
            features = [
                videointelligence_v1.Feature.SHOT_CHANGE_DETECTION,
                videointelligence_v1.Feature.LABEL_DETECTION,
                videointelligence_v1.Feature.OBJECT_TRACKING,
            ]
            
            # Run annotation
            operation = self.video_client.annotate_video(
                request={
                    "input_uri": gcs_uri,
                    "features": features,
                    "video_context": videointelligence_v1.VideoContext(
                        label_detection_config=videointelligence_v1.LabelDetectionConfig(
                            label_detection_mode=videointelligence_v1.LabelDetectionMode.SHOT_AND_FRAME_MODE,
                            stationary_camera=True,
                        )
                    ),
                }
            )
            
            # Wait for completion
            result = operation.result(timeout=300)  # 5 minute timeout
            
            # Process results
            shots = []
            labels = []
            
            # Extract shot boundaries
            for annotation_result in result.annotation_results:
                for shot in annotation_result.shot_annotations:
                    shots.append({
                        'start_time': shot.start_time_offset.total_seconds(),
                        'end_time': shot.end_time_offset.total_seconds(),
                        'confidence': shot.confidence
                    })
                
                # Extract labels
                for label in annotation_result.segment_label_annotations:
                    for segment in label.segments:
                        labels.append({
                            'label': label.entity.description,
                            'category': label.category_entities[0].description if label.category_entities else 'general',
                            'start_time': segment.segment.start_time_offset.total_seconds(),
                            'end_time': segment.segment.end_time_offset.total_seconds(),
                            'confidence': segment.confidence
                        })
            
            logger.info(f"✅ Video Intelligence completed: {len(shots)} shots, {len(labels)} labels")
            
            return {
                'success': True,
                'shots': shots,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"❌ Video Intelligence failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _combine_deterministic_signals(
        self, 
        speech_result: Dict, 
        video_intel_result: Dict, 
        video_url: str,
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Combine Speech-to-Text and Video Intelligence results"""
        try:
            logger.info("🔧 Combining deterministic signals...")
            
            # Extract data
            words = speech_result.get('words', []) if speech_result else []
            shots = video_intel_result.get('shots', []) if video_intel_result else []
            labels = video_intel_result.get('labels', []) if video_intel_result else []
            transcript = speech_result.get('transcript', '') if speech_result else ''
            
            # Create segment candidates using shot boundaries
            segment_candidates = self._propose_segments_from_shots(shots, words, labels)
            
            # Use LLM to refine segments (titles, summaries, validation)
            refined_segments = await self._refine_segments_with_llm(
                segment_candidates, transcript, video_url, company_name, qudemo_id
            )
            
            # Create chunks using word anchors
            chunks = self._create_chunks_with_word_anchors(
                refined_segments, words, video_url, company_name, qudemo_id
            )
            
            # Add validation metrics
            validation_metrics = self._calculate_validation_metrics(chunks, words, shots)
            
            logger.info(f"✅ Combined signals: {len(chunks)} chunks, {len(refined_segments)} segments")
            
            return {
                'success': True,
                'chunks': chunks,
                'segments': refined_segments,
                'words': words,
                'shots': shots,
                'labels': labels,
                'transcript': transcript,
                'validation_metrics': validation_metrics,
                'time_source': 'stt' if words else 'fallback',
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error combining deterministic signals: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url
            }
    
    def _propose_segments_from_shots(
        self, 
        shots: List[Dict], 
        words: List[Dict], 
        labels: List[Dict]
    ) -> List[Dict]:
        """Propose segment candidates using shot boundaries and other signals"""
        try:
            segments = []
            
            if not shots:
                # Fallback: create segments based on time intervals
                if words:
                    total_duration = max(w['end_time'] for w in words)
                    segment_duration = 30  # 30 second segments
                    num_segments = max(1, int(total_duration / segment_duration))
                    
                    for i in range(num_segments):
                        start_time = i * segment_duration
                        end_time = min((i + 1) * segment_duration, total_duration)
                        segments.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'confidence': 0.5,
                            'source': 'time_interval'
                        })
                else:
                    # Ultimate fallback
                    segments.append({
                        'start_time': 0,
                        'end_time': 60,
                        'confidence': 0.3,
                        'source': 'fallback'
                    })
            else:
                # Use shot boundaries as segment candidates
                for i, shot in enumerate(shots):
                    segments.append({
                        'start_time': shot['start_time'],
                        'end_time': shot['end_time'],
                        'confidence': shot['confidence'],
                        'source': 'shot_boundary',
                        'shot_id': i
                    })
            
            # Add label-based segment hints
            for label in labels:
                if label['confidence'] > 0.7:  # High confidence labels
                    segments.append({
                        'start_time': label['start_time'],
                        'end_time': label['end_time'],
                        'confidence': label['confidence'],
                        'source': 'label_detection',
                        'label': label['label'],
                        'category': label['category']
                    })
            
            # Sort by start time and merge overlapping segments
            segments.sort(key=lambda x: x['start_time'])
            merged_segments = self._merge_overlapping_segments(segments)
            
            logger.info(f"🔧 Proposed {len(merged_segments)} segment candidates")
            return merged_segments
            
        except Exception as e:
            logger.error(f"❌ Error proposing segments: {e}")
            return []
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping segments to avoid duplication"""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current['start_time'] <= last['end_time']:
                # Merge segments
                last['end_time'] = max(last['end_time'], current['end_time'])
                last['confidence'] = max(last['confidence'], current['confidence'])
                if 'labels' not in last:
                    last['labels'] = []
                if 'label' in current:
                    last['labels'].append(current['label'])
            else:
                # No overlap, add as new segment
                merged.append(current)
        
        return merged
    
    async def _refine_segments_with_llm(
        self, 
        segment_candidates: List[Dict], 
        transcript: str,
        video_url: str,
        company_name: str, 
        qudemo_id: str
    ) -> List[Dict]:
        """Use LLM to refine segments (titles, summaries, validation)"""
        try:
            # This is where you'd call your existing LLM (Gemini/GPT) to:
            # 1. Title each segment
            # 2. Validate/merge/split segments semantically
            # 3. Produce summaries/keywords
            # 4. NEVER let LLM invent times - only use existing timestamps
            
            logger.info(f"🤖 Refining {len(segment_candidates)} segments with LLM...")
            
            # For now, return segments with basic titles
            refined_segments = []
            for i, segment in enumerate(segment_candidates):
                refined_segment = segment.copy()
                refined_segment.update({
                    'title': f"Segment {i+1}",
                    'summary': f"Content from {segment['start_time']:.1f}s to {segment['end_time']:.1f}s",
                    'keywords': [],
                    'refined': True
                })
                refined_segments.append(refined_segment)
            
            return refined_segments
            
        except Exception as e:
            logger.error(f"❌ Error refining segments with LLM: {e}")
            return segment_candidates
    
    def _create_chunks_with_word_anchors(
        self, 
        segments: List[Dict], 
        words: List[Dict],
        video_url: str,
        company_name: str, 
        qudemo_id: str
    ) -> List[Dict]:
        """Create chunks using word anchors instead of linear interpolation"""
        try:
            chunks = []
            
            for segment in segments:
                # Find words within this segment
                segment_words = [
                    w for w in words 
                    if segment['start_time'] <= w['start_time'] < segment['end_time']
                ]
                
                if not segment_words:
                    continue
                
                # Create text from words
                segment_text = ' '.join(w['word'] for w in segment_words)
                
                # Create chunk with word-level precision
                chunk = {
                    'text': segment_text,
                    'start_time': segment_words[0]['start_time'],
                    'end_time': segment_words[-1]['end_time'],
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'time_source': 'stt_word_anchors',
                    'word_count': len(segment_words),
                    'segment_id': segment.get('shot_id', 0),
                    'labels': segment.get('labels', []),
                    'confidence': segment.get('confidence', 0.5),
                    'processed_at': datetime.now().isoformat()
                }
                
                chunks.append(chunk)
            
            logger.info(f"🔧 Created {len(chunks)} chunks with word anchors")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating chunks with word anchors: {e}")
            return []
    
    def _calculate_validation_metrics(
        self, 
        chunks: List[Dict], 
        words: List[Dict], 
        shots: List[Dict]
    ) -> Dict:
        """Calculate validation metrics for sanity checks"""
        try:
            if not chunks or not words:
                return {
                    'duration_parity': 0,
                    'anchor_coverage': 0,
                    'label_utility': 0
                }
            
            # Duration parity: Σ chunk durations ≈ total duration
            total_chunk_duration = sum(c['end_time'] - c['start_time'] for c in chunks)
            total_video_duration = max(w['end_time'] for w in words)
            duration_parity = 1 - abs(total_chunk_duration - total_video_duration) / total_video_duration
            
            # Anchor coverage: % of chunk boundaries that land on word anchors
            anchor_coverage = 0
            if chunks:
                anchored_boundaries = 0
                total_boundaries = len(chunks) * 2  # start and end
                
                for chunk in chunks:
                    # Check if start time matches a word anchor
                    if any(abs(w['start_time'] - chunk['start_time']) < 0.1 for w in words):
                        anchored_boundaries += 1
                    # Check if end time matches a word anchor
                    if any(abs(w['end_time'] - chunk['end_time']) < 0.1 for w in words):
                        anchored_boundaries += 1
                
                anchor_coverage = anchored_boundaries / total_boundaries if total_boundaries > 0 else 0
            
            # Label utility: % of chunks with labels
            chunks_with_labels = sum(1 for c in chunks if c.get('labels'))
            label_utility = chunks_with_labels / len(chunks) if chunks else 0
            
            return {
                'duration_parity': duration_parity,
                'anchor_coverage': anchor_coverage,
                'label_utility': label_utility,
                'total_chunks': len(chunks),
                'total_words': len(words),
                'total_shots': len(shots)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating validation metrics: {e}")
            return {}
    
    async def _cleanup_temp_files(self, video_path: str, gcs_uri: str):
        """Clean up temporary files and GCS objects"""
        try:
            # Clean up local file
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"🧹 Cleaned up local file: {video_path}")
            
            # Clean up GCS object
            if gcs_uri:
                # Extract bucket and blob name from URI
                gcs_path = gcs_uri.replace('gs://', '')
                bucket_name, blob_name = gcs_path.split('/', 1)
                
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.delete()
                logger.info(f"🧹 Cleaned up GCS object: {gcs_uri}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        return 'youtube.com' in url or 'youtu.be' in url
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def _fallback_processing(self, video_url: str, company_name: str, qudemo_id: str) -> Dict:
        """Fallback processing when Google Cloud services are not available"""
        try:
            logger.info(f"🔄 Using fallback processing for: {video_url}")
            
            # Return a basic result that indicates fallback was used
            return {
                'success': True,
                'chunks': [],
                'labels': [],
                'shots': [],
                'validation_metrics': {
                    'duration_parity': 0.0,
                    'anchor_coverage': 0.0,
                    'label_utility': 0.0
                },
                'time_source': 'fallback',
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'processing_method': 'fallback_basic',
                'message': 'Google Cloud services not available - using fallback processing'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in fallback processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }

# Global instance
_google_processor = None

def initialize_google_video_processor(service_account_path: str = None) -> bool:
    """Initialize global Google Video Intelligence processor"""
    global _google_processor
    try:
        _google_processor = GoogleVideoIntelligenceProcessor(service_account_path)
        logger.info("✅ Google Video Intelligence Processor initialized globally")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Video Intelligence Processor: {e}")
        return False

def get_google_video_processor() -> Optional[GoogleVideoIntelligenceProcessor]:
    """Get global Google Video Intelligence processor instance"""
    return _google_processor

