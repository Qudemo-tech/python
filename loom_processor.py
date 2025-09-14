#!/usr/bin/env python3
"""
Loom Video Processor with GCS Storage (YouTube Format)
Processes Loom videos and stores them in GCS with the same format as YouTube Gemini videos
"""

import os
import logging
import time
import tempfile
import subprocess
from typing import Optional, Dict, List, Any
from datetime import datetime
from google_cloud_storage_service import GoogleCloudStorageService
from simple_transcription import SimpleTranscriptionService

logger = logging.getLogger(__name__)

class LoomVideoProcessorGCS:
    """Loom video processor that stores in GCS with YouTube Gemini format"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the Loom video processor"""
        self.openai_api_key = openai_api_key
        self.transcriber = SimpleTranscriptionService(openai_api_key)
        self.gcs_service = GoogleCloudStorageService()
        logger.info("✅ Loom Video Processor GCS initialized")
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def create_gemini_format_transcript(self, segments: List[Dict]) -> str:
        """Create Gemini-style raw transcript from segments"""
        raw_transcript = ""
        for segment in segments:
            start_time = self.format_timestamp(segment.get('start', 0))
            end_time = self.format_timestamp(segment.get('end', 0))
            text = segment.get('text', '').strip()
            raw_transcript += f"[{start_time}-{end_time}] {text}\n\n"
        return raw_transcript.strip()
    
    def create_timestamps_array(self, segments: List[Dict]) -> List[Dict]:
        """Create timestamps array in YouTube format"""
        timestamps = []
        for segment in segments:
            timestamp = {
                "start_timestamp": int(segment.get('start', 0)),
                "end_timestamp": int(segment.get('end', 0)),
                "text": segment.get('text', '').strip(),
                "formatted_start": self.format_timestamp(segment.get('start', 0)),
                "formatted_end": self.format_timestamp(segment.get('end', 0))
            }
            timestamps.append(timestamp)
        return timestamps
    
    def create_segments_array(self, segments: List[Dict]) -> List[Dict]:
        """Create segments array in YouTube format"""
        formatted_segments = []
        for segment in segments:
            formatted_segment = {
                "start_timestamp": int(segment.get('start', 0)),
                "end_timestamp": int(segment.get('end', 0)),
                "text": segment.get('text', '').strip(),
                "formatted_start": self.format_timestamp(segment.get('start', 0)),
                "formatted_end": self.format_timestamp(segment.get('end', 0))
            }
            formatted_segments.append(formatted_segment)
        return formatted_segments
    
    def download_loom_video(self, video_url: str) -> Optional[str]:
        """Download Loom video using yt-dlp with quality fallback"""
        try:
            # Create temporary file path
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"loom_video_{int(time.time())}.mp4")
            
            # Try different quality options
            quality_options = [
                'best[height<=720]',  # 720p
                'best[height<=480]',  # 480p
                'best'                # Any quality
            ]
            
            for i, quality in enumerate(quality_options):
                try:
                    logger.info(f"Attempting download with {quality} quality (attempt {i+1}/{len(quality_options)})")
                    
                    cmd = [
                        'python', '-m', 'yt_dlp',
                        '--no-warnings',
                        '--retries', '2',
                        '--fragment-retries', '2',
                        '--format', quality,
                        '--output', temp_video_path,
                        video_url
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1000:
                        file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
                        logger.info(f"✅ Successfully downloaded {quality} quality video: {file_size_mb:.1f} MB")
                        return temp_video_path
                    else:
                        logger.warning(f"⚠️ {quality} quality download failed: {result.stderr}")
                        # Clean up failed file
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        
                except Exception as e:
                    logger.warning(f"⚠️ {quality} quality download error: {e}")
                    continue
            
            logger.error("❌ All download attempts failed")
            return None
                
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return None
    
    def extract_video_title(self, video_url: str) -> str:
        """Extract video title from Loom URL"""
        try:
            # Extract video ID from Loom URL
            if 'loom.com/share/' in video_url:
                video_id = video_url.split('loom.com/share/')[1].split('?')[0]
                return f"Loom Video - {video_id}"
            return "Loom Video"
        except:
            return "Loom Video"
    
    def process_video(self, video_url: str, company_name: str, qudemo_id: str = None) -> Optional[Dict]:
        """Process Loom video and store in GCS with YouTube format"""
        try:
            logger.info(f"🎬 Processing Loom video: {video_url}")
            logger.info(f"🏢 Company: {company_name}, Qudemo ID: {qudemo_id}")
            
            # Step 1: Download video
            logger.info("🎥 Downloading Loom video...")
            video_path = self.download_loom_video(video_url)
            if not video_path:
                raise Exception("Failed to download Loom video")
            
            # Step 2: Transcribe video
            logger.info("🎤 Transcribing video with Whisper API...")
            transcription_data = self.transcriber.transcribe_video(video_path)
            if not transcription_data:
                raise Exception("Failed to transcribe video")
            
            # Clean up video file
            try:
                os.unlink(video_path)
                logger.info("🧹 Cleaned up temporary video file")
            except:
                pass
            
            # Step 3: Extract data
            segments = transcription_data.get('segments', [])
            if not segments:
                raise Exception("No segments found in transcription")
            
            # Step 4: Create YouTube Gemini format
            raw_transcript = self.create_gemini_format_transcript(segments)
            timestamps = self.create_timestamps_array(segments)
            formatted_segments = self.create_segments_array(segments)
            video_title = self.extract_video_title(video_url)
            
            # Step 5: Create transcript data in YouTube format
            transcript_data = {
                "video_url": video_url,
                "video_title": video_title,
                "transcript": raw_transcript,
                "timestamps": timestamps,
                "segments": formatted_segments,
                "topics": [],
                "chunks": [],  # No chunks needed for Q&A
                "processed_at": datetime.now().isoformat()
            }
            
            # Step 6: Store in GCS
            logger.info("💾 Storing transcript in GCS...")
            storage_success = self.gcs_service.store_video_transcript(
                company_name, 
                qudemo_id, 
                transcript_data
            )
            
            if not storage_success:
                raise Exception("Failed to store transcript in GCS")
            
            # Step 7: Return success result
            result = {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'title': video_title,
                'transcript_length': len(raw_transcript),
                'segments_count': len(segments),
                'method': 'loom_whisper_gcs',
                'storage': 'gcs'
            }
            
            logger.info(f"✅ Loom video processing completed successfully")
            logger.info(f"📊 Transcript: {len(raw_transcript)} chars, {len(segments)} segments")
            logger.info(f"💾 Stored in GCS: {company_name}/{qudemo_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }

def main():
    """Test the Loom processor"""
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    processor = LoomVideoProcessorGCS(openai_key)
    
    # Test with Loom video
    loom_url = "https://www.loom.com/share/cdca2f74fedc4a119df8219a8c86cd8c?sid=f3dae5d3-90c0-4ffb-9f85-c57ee7cd0a02"
    
    print("🎬 Testing Loom Video Processor GCS")
    print(f"📹 Video: {loom_url}")
    
    result = processor.process_video(loom_url, 'test_company', 'test_qudemo')
    
    if result and result.get('success'):
        print("✅ Processing successful!")
        print(f"📊 Transcript: {result.get('transcript_length', 0)} chars")
        print(f"📊 Segments: {result.get('segments_count', 0)}")
        print(f"💾 Storage: {result.get('storage', 'Unknown')}")
    else:
        print("❌ Processing failed")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
