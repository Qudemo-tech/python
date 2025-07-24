#!/usr/bin/env python3
"""
Simple Video Processor - Fallback for when ffmpeg is not available
"""
import os
import tempfile
import logging
import whisper
import yt_dlp
import requests
from typing import List, Dict
import psutil
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SimpleVideoProcessor:
    def __init__(self, max_memory_mb: int = 400):
        """
        Initialize simple video processor (no chunking)
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        
    def log_memory(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 {stage} - Memory: {memory_mb:.1f}MB")
        return memory_mb
    
    def download_video(self, video_url: str, output_filename: str) -> str:
        """Download video with memory monitoring"""
        self.log_memory("Before download")
        
        if os.path.exists(video_url):
            logger.info(f"Using local file: {video_url}")
            return video_url
            
        if video_url.startswith('http') and ('youtube.com' in video_url or 'youtu.be' in video_url):
            return self._download_youtube(video_url, output_filename)
        elif video_url.startswith('http'):
            return self._download_http(video_url, output_filename)
        else:
            raise Exception("Invalid video input")
    
    def _download_youtube(self, video_url: str, output_filename: str) -> str:
        """Download YouTube video with lowest quality"""
        ydl_opts = {
            'format': 'worst[height<=360]',  # Very low quality to save memory
            'outtmpl': output_filename,
            'quiet': True,
            'max_filesize': '25M',  # Very small file size limit
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
    
    def _download_http(self, video_url: str, output_filename: str) -> str:
        """Download HTTP video with strict size limits"""
        r = requests.get(video_url, stream=True, timeout=30)
        r.raise_for_status()
        
        # Check file size
        content_length = r.headers.get('content-length')
        if content_length:
            file_size_mb = int(content_length) / (1024 * 1024)
            if file_size_mb > 25:  # Very strict limit
                raise Exception(f"File too large: {file_size_mb:.1f}MB (max 25MB)")
        
        with open(output_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    # Check memory during download
                    if self.log_memory("Downloading") > self.max_memory_mb:
                        raise Exception("Memory limit exceeded during download")
        
        return output_filename
    
    def transcribe_video(self, video_path: str, company_name: str) -> List[Dict]:
        """Transcribe video using tiny Whisper model"""
        self.log_memory("Before transcription")
        
        # Use tiny model for memory efficiency
        model = whisper.load_model("tiny")
        result = model.transcribe(video_path, task="translate")
        
        # Process segments
        chunks = []
        for seg in result["segments"]:
            chunk_data = {
                "source": f"{os.path.basename(video_path)} [{seg['start']:.1f}s - {seg['end']:.1f}s]",
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
                "chunk_index": 0  # Single chunk
            }
            chunks.append(chunk_data)
        
        self.log_memory("After transcription")
        return chunks
    
    def process_video(self, video_url: str, company_name: str) -> Dict:
        """Process video without chunking"""
        try:
            # Download video
            video_filename = f"simple_video_{os.getpid()}.mp4"
            video_path = self.download_video(video_url, video_filename)
            
            # Transcribe video
            chunks = self.transcribe_video(video_path, company_name)
            
            # Clean up video file
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"🗑️ Cleaned up video file: {video_path}")
            
            return {
                "success": True,
                "chunks_count": len(chunks),
                "chunks": chunks,
                "message": "Video processed without chunking (simple mode)"
            }
            
        except Exception as e:
            logger.error(f"Simple video processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Usage example
if __name__ == "__main__":
    processor = SimpleVideoProcessor(max_memory_mb=400)
    result = processor.process_video(
        "https://youtu.be/example",
        "test_company"
    )
    print(result) 