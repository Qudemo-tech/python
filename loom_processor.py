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
    
    def compress_video_for_whisper(self, input_path: str) -> Optional[str]:
        """Compress video to under 25MB for Whisper API"""
        try:
            # Check if file is already small enough
            file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            if file_size_mb <= 25:
                logger.info(f"✅ Video already under 25MB: {file_size_mb:.1f} MB")
                return input_path
            
            # Create compressed file path
            temp_dir = tempfile.gettempdir()
            compressed_path = os.path.join(temp_dir, f"loom_compressed_{int(time.time())}.mp4")
            
            logger.info(f"🗜️ Compressing video from {file_size_mb:.1f} MB to under 25MB...")
            
            # Use ffmpeg to compress video
            # Target: 20MB max (leave some buffer)
            # More aggressive compression for large files
            if file_size_mb > 100:
                target_bitrate = "200k"  # Very aggressive for large files
                audio_bitrate = "32k"
            elif file_size_mb > 50:
                target_bitrate = "300k"  # Aggressive for medium files
                audio_bitrate = "48k"
            else:
                target_bitrate = "500k"  # Conservative for smaller files
                audio_bitrate = "64k"
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-b:v', target_bitrate,
                '-c:a', 'aac',
                '-b:a', audio_bitrate,
                '-movflags', '+faststart',
                '-preset', 'fast',  # Faster encoding
                '-crf', '28',  # Constant rate factor for quality
                '-y',  # Overwrite output file
                compressed_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(compressed_path):
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                logger.info(f"✅ Video compressed successfully: {compressed_size_mb:.1f} MB")
                
                # If still too large, try more aggressive compression
                if compressed_size_mb > 25:
                    logger.warning(f"⚠️ Compressed video still too large ({compressed_size_mb:.1f} MB), trying ultra-aggressive compression...")
                    
                    # Ultra-aggressive compression
                    ultra_compressed_path = os.path.join(temp_dir, f"loom_ultra_compressed_{int(time.time())}.mp4")
                    ultra_cmd = [
                        'ffmpeg',
                        '-i', compressed_path,
                        '-c:v', 'libx264',
                        '-b:v', '100k',  # Ultra-low bitrate
                        '-c:a', 'aac',
                        '-b:a', '16k',   # Ultra-low audio bitrate
                        '-movflags', '+faststart',
                        '-preset', 'ultrafast',
                        '-crf', '35',    # Very high compression
                        '-y',
                        ultra_compressed_path
                    ]
                    
                    ultra_result = subprocess.run(ultra_cmd, capture_output=True, text=True, timeout=300)
                    
                    if ultra_result.returncode == 0 and os.path.exists(ultra_compressed_path):
                        ultra_size_mb = os.path.getsize(ultra_compressed_path) / (1024 * 1024)
                        logger.info(f"✅ Ultra-compressed video: {ultra_size_mb:.1f} MB")
                        
                        # Clean up intermediate file
                        try:
                            os.unlink(compressed_path)
                        except:
                            pass
                        
                        compressed_path = ultra_compressed_path
                        compressed_size_mb = ultra_size_mb
                    else:
                        logger.warning(f"⚠️ Ultra-compression failed, using regular compression: {compressed_size_mb:.1f} MB")
                
                # Clean up original file
                try:
                    os.unlink(input_path)
                    logger.info("🧹 Cleaned up original video file")
                except:
                    pass
                
                return compressed_path
            else:
                logger.error(f"❌ Video compression failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video compression error: {e}")
            return None

    def get_available_formats(self, video_url: str) -> List[str]:
        """Get available formats for a Loom video"""
        try:
            cmd = [
                'python', '-m', 'yt_dlp',
                '--list-formats',
                '--no-warnings',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                formats = []
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'mp4' in line.lower() or 'webm' in line.lower():
                        # Extract format ID (usually first column)
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            formats.append(parts[0])
                logger.info(f"📋 Available formats: {formats}")
                return formats
            else:
                logger.warning(f"⚠️ Could not list formats: {result.stderr}")
                return []
                
        except Exception as e:
            logger.warning(f"⚠️ Error listing formats: {e}")
            return []

    def download_loom_video(self, video_url: str) -> Optional[str]:
        """Download Loom video using yt-dlp with comprehensive fallback strategies"""
        try:
            # Create temporary file path
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"loom_video_{int(time.time())}.mp4")
            
            # Strategy 1: Try to get available formats first
            logger.info("🔍 Checking available formats...")
            available_formats = self.get_available_formats(video_url)
            
            # Strategy 2: Try different download approaches
            download_strategies = [
                # Strategy 1: Use available formats if found
                {
                    'name': 'Available Formats',
                    'formats': available_formats[:3] if available_formats else [],  # Try top 3 formats
                    'extra_args': ['--no-warnings', '--retries', '3', '--fragment-retries', '3']
                },
                # Strategy 2: Standard quality fallback
                {
                    'name': 'Standard Quality',
                    'formats': ['best[height<=720]', 'best[height<=480]', 'best'],
                    'extra_args': ['--no-warnings', '--retries', '2', '--fragment-retries', '2']
                },
                # Strategy 3: Any available format
                {
                    'name': 'Any Format',
                    'formats': ['best'],
                    'extra_args': ['--no-warnings', '--retries', '1', '--fragment-retries', '1', '--format', 'best']
                },
                # Strategy 4: Force specific formats
                {
                    'name': 'Force Formats',
                    'formats': ['worst', 'best[ext=mp4]', 'best[ext=webm]'],
                    'extra_args': ['--no-warnings', '--retries', '1']
                },
                # Strategy 5: Use cookies and headers (for private videos)
                {
                    'name': 'With Headers',
                    'formats': ['best'],
                    'extra_args': [
                        '--no-warnings', 
                        '--retries', '2',
                        '--add-header', 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    ]
                }
            ]
            
            for strategy_idx, strategy in enumerate(download_strategies):
                logger.info(f"🎯 Strategy {strategy_idx + 1}: {strategy['name']}")
                
                formats_to_try = strategy['formats']
                if not formats_to_try:
                    logger.info("⏭️ No formats to try in this strategy, skipping...")
                    continue
                
                for format_idx, format_id in enumerate(formats_to_try):
                    try:
                        logger.info(f"📥 Attempting download with format '{format_id}' (strategy {strategy_idx + 1}, attempt {format_idx + 1})")
                        
                        # Build command
                        cmd = ['python', '-m', 'yt_dlp']
                        cmd.extend(strategy['extra_args'])
                        
                        # Add format if not already in extra_args
                        if '--format' not in ' '.join(strategy['extra_args']):
                            cmd.extend(['--format', format_id])
                        
                        cmd.extend(['--output', temp_video_path, video_url])
                        
                        # Run download with timeout
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                        
                        # Check if download was successful
                        if result.returncode == 0 and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1000:
                            file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
                            logger.info(f"✅ Successfully downloaded with format '{format_id}': {file_size_mb:.1f} MB")
                            
                            # Compress video if too large for Whisper
                            compressed_path = self.compress_video_for_whisper(temp_video_path)
                            return compressed_path
                        else:
                            logger.warning(f"⚠️ Format '{format_id}' download failed: {result.stderr}")
                            # Clean up failed file
                            if os.path.exists(temp_video_path):
                                try:
                                    os.remove(temp_video_path)
                                except:
                                    pass
                            
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚠️ Format '{format_id}' download timed out")
                        if os.path.exists(temp_video_path):
                            try:
                                os.remove(temp_video_path)
                            except:
                                pass
                        continue
                    except Exception as e:
                        logger.warning(f"⚠️ Format '{format_id}' download error: {e}")
                        if os.path.exists(temp_video_path):
                            try:
                                os.remove(temp_video_path)
                            except:
                                pass
                        continue
                
                # If we've tried all formats in this strategy and none worked, continue to next strategy
                logger.info(f"❌ Strategy {strategy_idx + 1} ({strategy['name']}) failed, trying next strategy...")
            
            # Strategy 6: Last resort - try with different yt-dlp options
            logger.info("🆘 Last resort: Trying with minimal options...")
            try:
                minimal_cmd = [
                    'python', '-m', 'yt_dlp',
                    '--format', 'worst',
                    '--output', temp_video_path,
                    '--no-warnings',
                    video_url
                ]
                
                result = subprocess.run(minimal_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1000:
                    file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
                    logger.info(f"✅ Last resort download successful: {file_size_mb:.1f} MB")
                    
                    compressed_path = self.compress_video_for_whisper(temp_video_path)
                    return compressed_path
                else:
                    logger.error(f"❌ Last resort download failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"❌ Last resort download error: {e}")
            
            logger.error("❌ All download strategies failed")
            return None
                
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return None
    
    def normalize_loom_url(self, video_url: str) -> str:
        """Normalize Loom URL to handle different formats"""
        try:
            # Remove any extra parameters that might cause issues
            if 'loom.com/share/' in video_url:
                # Extract the base URL with video ID
                base_url = video_url.split('?')[0]  # Remove query parameters
                
                # Ensure it has the proper format
                if not base_url.endswith('/'):
                    base_url += '/'
                
                logger.info(f"🔄 Normalized Loom URL: {base_url}")
                return base_url
            
            return video_url
        except Exception as e:
            logger.warning(f"⚠️ Error normalizing URL: {e}")
            return video_url

    def validate_loom_url(self, video_url: str) -> bool:
        """Validate if the URL is a proper Loom URL"""
        try:
            # Check if it's a Loom URL
            if 'loom.com/share/' not in video_url:
                return False
            
            # Extract video ID
            video_id = video_url.split('loom.com/share/')[1].split('?')[0].split('/')[0]
            
            # Check if video ID looks valid (alphanumeric, reasonable length)
            if len(video_id) < 10 or len(video_id) > 50:
                return False
            
            # Check if it contains only valid characters
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', video_id):
                return False
            
            logger.info(f"✅ Valid Loom URL detected: {video_id}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error validating Loom URL: {e}")
            return False

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
    
    def _log_raw_loom_data(self, video_url: str, raw_transcript: str, segments: List[Dict]):
        """Log raw data from Loom video processing like Gemini does"""
        try:
            logger.info("=" * 80)
            logger.info("🔍 RAW LOOM DATA LOGGING - BEFORE CHUNKING")
            logger.info("=" * 80)
            logger.info(f"📹 Video URL: {video_url}")
            logger.info(f"📊 Raw Transcript Length: {len(raw_transcript)} characters")
            logger.info(f"📊 Word Count: {len(raw_transcript.split())}")
            logger.info(f"📊 Segments Count: {len(segments)}")
            
            # Log raw transcript preview
            transcript_preview = raw_transcript[:1000]
            logger.info(f"📝 Raw transcript preview (first 1000 chars):")
            logger.info(f"{transcript_preview}")
            
            if len(raw_transcript) > 1000:
                transcript_end = raw_transcript[-1000:]
                logger.info(f"📝 Raw transcript preview (last 1000 chars):")
                logger.info(f"{transcript_end}")
            
            # Log segment details with timestamps
            logger.info(f"📊 Segment Details:")
            for i, segment in enumerate(segments[:10]):  # Log first 10 segments
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '')
                duration = end_time - start_time
                
                logger.info(f"  Segment {i+1}:")
                logger.info(f"    Time range: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
                logger.info(f"    Text length: {len(text)} characters")
                logger.info(f"    Word count: {len(text.split())}")
                logger.info(f"    Text preview: {text[:200]}...")
                
                if i < len(segments) - 1 and i < 9:
                    logger.info("    ---")
            
            if len(segments) > 10:
                logger.info(f"  ... and {len(segments) - 10} more segments")
            
            # Log timestamp statistics
            if segments:
                start_times = [s.get('start', 0) for s in segments]
                end_times = [s.get('end', 0) for s in segments]
                min_start = min(start_times)
                max_end = max(end_times)
                total_duration = max_end - min_start
                
                logger.info(f"📊 Timestamp Statistics:")
                logger.info(f"  Time range: {min_start:.2f}s - {max_end:.2f}s")
                logger.info(f"  Total duration: {total_duration:.2f}s")
                logger.info(f"  Average segment duration: {total_duration/len(segments):.2f}s")
                
                # Check for gaps or overlaps
                gaps = []
                overlaps = []
                for i in range(len(segments) - 1):
                    current_end = segments[i].get('end', 0)
                    next_start = segments[i + 1].get('start', 0)
                    if next_start > current_end:
                        gaps.append(next_start - current_end)
                    elif next_start < current_end:
                        overlaps.append(current_end - next_start)
                
                if gaps:
                    logger.info(f"  Gaps detected: {len(gaps)} gaps, avg: {sum(gaps)/len(gaps):.2f}s")
                if overlaps:
                    logger.info(f"  Overlaps detected: {len(overlaps)} overlaps, avg: {sum(overlaps)/len(overlaps):.2f}s")
            
            logger.info("=" * 80)
            logger.info("✅ RAW LOOM DATA LOGGING COMPLETE")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error logging raw Loom data: {e}")
    
    def process_video(self, video_url: str, company_name: str, qudemo_id: str = None) -> Optional[Dict]:
        """Process Loom video and store in GCS with YouTube format"""
        try:
            logger.info(f"🎬 Processing Loom video: {video_url}")
            logger.info(f"🏢 Company: {company_name}, Qudemo ID: {qudemo_id}")
            
            # Step 0: Validate and normalize URL
            logger.info("🔍 Validating Loom URL...")
            if not self.validate_loom_url(video_url):
                raise Exception(f"Invalid Loom URL format: {video_url}")
            
            normalized_url = self.normalize_loom_url(video_url)
            logger.info(f"✅ Using normalized URL: {normalized_url}")
            
            # Step 1: Download video
            logger.info("🎥 Downloading Loom video...")
            video_path = self.download_loom_video(normalized_url)
            if not video_path:
                raise Exception("Failed to download Loom video")
            
            # Step 2: Transcribe video
            logger.info("🎤 Transcribing video with Whisper API...")
            transcription_data = self.transcriber.transcribe_video(video_path)
            if not transcription_data:
                raise Exception("Failed to transcribe video")
            
            # Clean up video file (could be original or compressed)
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
                'video_url': video_url,  # Use original URL for consistency
                'normalized_url': normalized_url,  # Include normalized URL for debugging
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'title': video_title,
                'transcript_length': len(raw_transcript),
                'segments_count': len(segments),
                'method': 'loom_whisper_gcs_enhanced',
                'storage': 'gcs'
            }
            
            logger.info(f"✅ Loom video processing completed successfully")
            logger.info(f"📊 Transcript: {len(raw_transcript)} chars, {len(segments)} segments")
            logger.info(f"💾 Stored in GCS: {company_name}/{qudemo_id}")
            
            # Log timestamp and raw transcript data like Gemini does
            self._log_raw_loom_data(video_url, raw_transcript, segments)
            
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
    """Test the Loom processor with multiple URLs"""
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    processor = LoomVideoProcessorGCS(openai_key)
    
    # Test with multiple Loom videos including the problematic one
    test_urls = [
        "https://www.loom.com/share/cdca2f74fedc4a119df8219a8c86cd8c?sid=f3dae5d3-90c0-4ffb-9f85-c57ee7cd0a02",  # Working URL
        "https://www.loom.com/share/473fad25ebd24b5ea8091503253dfecf",  # Problematic URL
        "https://www.loom.com/share/473fad25ebd24b5ea8091503253dfecf?sid=test123",  # With session ID
    ]
    
    print("🎬 Testing Enhanced Loom Video Processor GCS")
    
    for i, loom_url in enumerate(test_urls, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {loom_url}")
        print(f"{'='*60}")
        
        # Test URL validation first
        is_valid = processor.validate_loom_url(loom_url)
        print(f"🔍 URL Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if is_valid:
            normalized = processor.normalize_loom_url(loom_url)
            print(f"🔄 Normalized URL: {normalized}")
        
        # Test processing
        result = processor.process_video(loom_url, 'test_company', f'test_qudemo_{i}')
        
        if result and result.get('success'):
            print("✅ Processing successful!")
            print(f"📊 Transcript: {result.get('transcript_length', 0)} chars")
            print(f"📊 Segments: {result.get('segments_count', 0)}")
            print(f"💾 Storage: {result.get('storage', 'Unknown')}")
            print(f"🔧 Method: {result.get('method', 'Unknown')}")
        else:
            print("❌ Processing failed")
            if result:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main()