#!/usr/bin/env python3
"""
Large Video Processor - Handles videos that exceed memory limits
"""
import os
import tempfile
import logging
import whisper
import yt_dlp
import requests
from typing import List, Dict, Tuple
import psutil
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LargeVideoProcessor:
    def __init__(self, max_memory_mb: int = 400, chunk_duration: int = 300):
        """
        Initialize large video processor
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
        """
        self.max_memory_mb = max_memory_mb
        self.chunk_duration = chunk_duration
        self.temp_dir = tempfile.mkdtemp()
        
    def log_memory(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 {stage} - Memory: {memory_mb:.1f}MB")
        return memory_mb
    
    def download_large_video(self, video_url: str, output_filename: str) -> str:
        """Download large video with memory monitoring"""
        self.log_memory("Before download")
        
        if os.path.exists(video_url):
            logger.info(f"Using local file: {video_url}")
            return video_url
            
        if video_url.startswith('http') and ('youtube.com' in video_url or 'youtu.be' in video_url):
            return self._download_youtube_large(video_url, output_filename)
        elif video_url.startswith('http'):
            return self._download_http_large(video_url, output_filename)
        else:
            raise Exception("Invalid video input")
    
    def _download_youtube_large(self, video_url: str, output_filename: str) -> str:
        """Download large YouTube video with quality selection"""
        ydl_opts = {
            'format': 'worst[height<=720]/worst[height<=480]/worst',  # Progressive quality fallback
            'outtmpl': output_filename,
            'quiet': True,
            'max_filesize': '100M',  # Allow larger files for chunked processing
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            self.log_memory("After YouTube download")
            return output_filename
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            raise
    
    def _download_http_large(self, video_url: str, output_filename: str) -> str:
        """Download large HTTP video with streaming"""
        r = requests.get(video_url, stream=True, timeout=60)
        r.raise_for_status()
        
        # Check file size
        content_length = r.headers.get('content-length')
        if content_length:
            file_size_mb = int(content_length) / (1024 * 1024)
            if file_size_mb > 200:  # Allow up to 200MB for chunked processing
                logger.warning(f"Large file detected: {file_size_mb:.1f}MB")
        
        with open(output_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    # Check memory during download
                    if self.log_memory("Downloading") > self.max_memory_mb:
                        raise Exception("Memory limit exceeded during download")
        
        return output_filename
    
    def split_video_into_chunks(self, video_path: str) -> List[str]:
        """Split large video into smaller chunks"""
        import subprocess
        
        self.log_memory("Before video splitting")
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            ffmpeg_available = True
            logger.info("✅ ffmpeg is available for video chunking")
        except (subprocess.CalledProcessError, FileNotFoundError):
            ffmpeg_available = False
            logger.warning("⚠️ ffmpeg not available, using fallback processing")
        
        if not ffmpeg_available:
            # Fallback: process the entire video without chunking
            logger.info("🔄 Using fallback: processing entire video without chunking")
            return [video_path]
        
        # Get video duration
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        
        try:
            duration = float(subprocess.check_output(cmd).decode().strip())
            logger.info(f"Video duration: {duration:.1f} seconds")
        except:
            duration = 600  # Default to 10 minutes if can't detect
        
        # Calculate number of chunks
        num_chunks = max(1, int(duration / self.chunk_duration))
        chunk_files = []
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            end_time = min((i + 1) * self.chunk_duration, duration)
            
            chunk_filename = os.path.join(self.temp_dir, f"chunk_{i:03d}.mp4")
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-y',  # Overwrite output
                chunk_filename
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                chunk_files.append(chunk_filename)
                logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_filename}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {i}: {e}")
                continue
        
        self.log_memory("After video splitting")
        return chunk_files
    
    def transcribe_chunks(self, chunk_files: List[str], company_name: str) -> List[Dict]:
        """Transcribe video chunks with memory management"""
        all_chunks = []
        model = whisper.load_model("tiny")  # Use tiny model for memory efficiency
        
        for i, chunk_file in enumerate(chunk_files):
            try:
                self.log_memory(f"Before transcribing chunk {i+1}")
                
                # Transcribe chunk
                result = model.transcribe(chunk_file, task="translate")
                
                # Process segments
                for seg in result["segments"]:
                    chunk_data = {
                        "source": f"{os.path.basename(chunk_file)} [{seg['start']:.1f}s - {seg['end']:.1f}s]",
                        "text": seg["text"].strip(),
                        "start": seg["start"],
                        "end": seg["end"],
                        "chunk_index": i
                    }
                    all_chunks.append(chunk_data)
                
                # Clean up chunk file immediately
                os.remove(chunk_file)
                self.log_memory(f"After transcribing chunk {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i}: {e}")
                continue
        
        return all_chunks
    
    def process_large_video(self, video_url: str, company_name: str) -> Dict:
        """Process large video in chunks"""
        try:
            # Download video
            video_filename = f"large_video_{os.getpid()}.mp4"
            video_path = self.download_large_video(video_url, video_filename)
            
            # Split into chunks
            chunk_files = self.split_video_into_chunks(video_path)
            
            # Transcribe chunks
            chunks = self.transcribe_chunks(chunk_files, company_name)
            
            # Clean up original video
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Clean up temp directory
            try:
                os.rmdir(self.temp_dir)
            except:
                pass  # Directory not empty
            
            return {
                "success": True,
                "chunks_count": len(chunks),
                "chunks": chunks,
                "message": f"Processed large video in {len(chunk_files)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Large video processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Usage example
if __name__ == "__main__":
    processor = LargeVideoProcessor(max_memory_mb=400, chunk_duration=300)
    result = processor.process_large_video(
        "https://youtu.be/example",
        "test_company"
    )
    print(result) 