#!/usr/bin/env python3
"""
Loom Video Processor
Handles Loom video processing with transcription and vector storage
"""

import os
import logging
import time
import json
from typing import Dict, Optional, List
import requests
import tempfile
import whisper
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import openai
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoomVideoProcessor:
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        """
        Initialize Loom Video Processor
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            pinecone_api_key: Pinecone API key
        """
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-core")
        
        # Initialize Whisper model (lazy loading)
        self._whisper_model = None
        
        logger.info("🔧 Initializing Loom Video Processor...")
    
    def get_whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            logger.info("🎤 Loading Whisper model...")
            self._whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
        return self._whisper_model
    
    def is_loom_url(self, url: str) -> bool:
        """Check if URL is a Loom URL"""
        domain = urlparse(url).netloc.lower()
        return 'loom.com' in domain
    
    def extract_loom_video_info(self, loom_url: str) -> Optional[Dict]:
        """
        Extract video information from Loom URL
        
        Args:
            loom_url: Loom video URL
            
        Returns:
            Dict with video info or None if failed
        """
        try:
            logger.info(f"🎬 Extracting Loom video info from: {loom_url}")
            
            # Parse Loom URL to get video ID
            parsed_url = urlparse(loom_url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            if len(path_parts) >= 2:
                video_id = path_parts[-1]  # Last part is usually the video ID
                
                            # Try different Loom API endpoints
            api_urls = [
                f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-video",
                f"https://www.loom.com/api/sessions/{video_id}",
                f"https://www.loom.com/api/v2/sessions/{video_id}"
            ]
            
            video_data = None
            for api_url in api_urls:
                try:
                    logger.info(f"🔗 Trying Loom API: {api_url}")
                    response = requests.get(api_url, timeout=30)
                    if response.status_code == 200:
                        video_data = response.json()
                        logger.info(f"✅ Success with API: {api_url}")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Failed with API {api_url}: {e}")
                    continue
            
            if not video_data:
                # Fallback: try to extract info from the share page
                logger.info("🔄 Trying fallback method - extracting from share page")
                try:
                    share_response = requests.get(loom_url, timeout=30)
                    if share_response.status_code == 200:
                        # Look for video data in the page
                        import re
                        content = share_response.text
                        
                        # Try multiple patterns to find video URL
                        video_url_patterns = [
                            r'"videoUrl":"([^"]+)"',
                            r'"url":"([^"]*\.mp4[^"]*)"',
                            r'"video_url":"([^"]+)"',
                            r'"src":"([^"]*\.mp4[^"]*)"',
                            r'https://[^"]*\.mp4[^"]*'
                        ]
                        
                        video_url = None
                        for pattern in video_url_patterns:
                            match = re.search(pattern, content)
                            if match:
                                video_url = match.group(1) if match.groups() else match.group(0)
                                if video_url.startswith('http'):
                                    break
                        
                        # Try to find title
                        title_patterns = [
                            r'"title":"([^"]+)"',
                            r'"name":"([^"]+)"',
                            r'<title>([^<]+)</title>'
                        ]
                        
                        title = None
                        for pattern in title_patterns:
                            match = re.search(pattern, content)
                            if match:
                                title = match.group(1)
                                break
                        
                        if video_url:
                            video_data = {
                                'url': video_url,
                                'title': title if title else 'Unknown',
                                'duration': 0
                            }
                            logger.info(f"✅ Extracted video data from share page: {video_url}")
                        else:
                            logger.warning("⚠️ No video URL found in share page")
                except Exception as e:
                    logger.error(f"❌ Fallback method failed: {e}")
            
            if not video_data:
                # Create a minimal video data structure for processing
                logger.warning("⚠️ Could not extract video data, using minimal structure")
                video_data = {
                    'url': loom_url,
                    'title': f'Loom Video - {video_id}',
                    'duration': 0
                }
            
            # Extract relevant info
            video_info = {
                'video_id': video_id,
                'title': video_data.get('title', 'Unknown'),
                'duration': video_data.get('duration', 0),
                'video_url': video_data.get('url'),
                'thumbnail_url': video_data.get('thumbnailUrl')
            }
            
            logger.info(f"✅ Loom video info extracted: {video_info['title']}")
            return video_info
                
        except Exception as e:
            logger.error(f"❌ Failed to extract Loom video info: {e}")
            return None
    
    def download_loom_video(self, video_url: str, output_path: str) -> bool:
        """
        Download Loom video to local file
        
        Args:
            video_url: Direct video URL
            output_path: Local path to save video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📥 Downloading Loom video to: {output_path}")
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(video_url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()

            # Check if we got a video file. Loom share pages return text/html; treat that as failure
            content_type = response.headers.get('content-type', '').lower()
            is_video_like = any(t in content_type for t in ['video', 'mp4', 'webm', 'ogg', 'application/octet-stream'])
            if not is_video_like:
                logger.warning(f"⚠️ Response doesn't appear to be a video file (content-type: {content_type}). Will use yt-dlp fallback.")
                return False

            # Download the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify the file is valid
            import os
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # Less than 1KB
                logger.error(f"❌ Downloaded file is too small: {file_size} bytes")
                return False

            logger.info(f"✅ Loom video downloaded successfully: {file_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download Loom video: {e}")
            return False
    
    def validate_and_enhance_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Validate and enhance timestamps to ensure precision (YouTube-style)
        
        Args:
            segments: List of segments with timestamps
            
        Returns:
            Enhanced segments with validated timestamps
        """
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            start = float(segment.get('start', 0.0))
            end = float(segment.get('end', start))
            
            # Validate timestamps
            if start < 0:
                start = 0.0
            if end <= start:
                # Estimate duration based on text length (YouTube-style)
                word_count = len(text.split())
                estimated_duration = max(word_count * 0.4, 0.5)  # ~0.4s per word, min 0.5s
                end = start + estimated_duration
            
            # Ensure reasonable duration
            if end - start > 30:  # Cap at 30 seconds per segment
                end = start + 30
            
            enhanced_segments.append({
                'text': text,
                'start': start,
                'end': end
            })
        
        logger.info(f"✅ Enhanced {len(enhanced_segments)} segments with validated timestamps")
        return enhanced_segments

    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """
        Transcribe video using Whisper with enhanced timestamp precision
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with transcription data or None if failed
        """
        try:
            logger.info(f"🎤 Transcribing video: {video_path}")
            
            # Load Whisper model
            model = self.get_whisper_model()
            
            # Transcribe video with detailed segments for better timestamp precision
            result = model.transcribe(
                video_path,
                word_timestamps=True,  # Enable word-level timestamps for precision
                verbose=False
            )
            
            # Process segments to ensure precise timestamps (similar to YouTube API)
            enhanced_segments = []
            for segment in result.get('segments', []):
                # Ensure we have precise start and end times
                start = float(segment.get('start', 0.0))
                end = float(segment.get('end', start))
                
                # If end time is missing or same as start, estimate it
                if end <= start:
                    # Estimate duration based on text length (similar to YouTube API)
                    text = segment.get('text', '').strip()
                    estimated_duration = max(len(text.split()) * 0.5, 1.0)  # ~0.5s per word, min 1s
                    end = start + estimated_duration
                
                enhanced_segments.append({
                    'text': segment.get('text', '').strip(),
                    'start': start,
                    'end': end
                })
            
            # Validate and enhance timestamps (YouTube-style precision)
            enhanced_segments = self.validate_and_enhance_timestamps(enhanced_segments)
            
            transcription_data = {
                'transcription': result['text'],
                'segments': enhanced_segments,  # Use enhanced segments
                'language': result.get('language', 'en'),
                'word_count': len(result['text'].split())
            }
            
            logger.info(f"✅ Transcription completed: {transcription_data['word_count']} words")
            logger.info(f"🌐 Language: {transcription_data.get('language', 'Unknown')}")
            logger.info(f"📊 Enhanced segments created: {len(enhanced_segments)}")
            
            # Log the full transcription content
            transcription_text = transcription_data.get('transcription', '')
            if transcription_text:
                logger.info("📄 FULL TRANSCRIPTION CONTENT:")
                logger.info("=" * 80)
                logger.info(transcription_text[:2000] + ("..." if len(transcription_text) > 2000 else ""))
                logger.info("=" * 80)
                if len(transcription_text) > 2000:
                    logger.info(f"📄 (Showing first 2000 characters of {len(transcription_text)} total)")
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def chunk_transcription(
        self,
        transcription: str,
        segments: Optional[List[Dict]] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        max_chunk_duration: int = 60,
    ) -> List[Dict]:
        """
        Create timestamped chunks from transcription (YouTube-style precision).

        If Whisper segments (with start/end) are provided, build chunks by aggregating
        consecutive segments until reaching the target size or max duration, and record
        start/end timestamps. Otherwise, fall back to character-based chunking without
        timestamps.

        Returns list of dicts: { text: str, start: float, end: float }
        """
        if segments:
            chunks: List[Dict] = []
            current_text_parts: List[str] = []
            current_start: Optional[float] = None
            current_end: Optional[float] = None

            def flush_chunk():
                nonlocal current_text_parts, current_start, current_end
                if current_text_parts and current_start is not None and current_end is not None:
                    # Ensure we have valid timestamps (YouTube-style precision)
                    chunk_start = float(max(0.0, current_start))
                    chunk_end = float(max(current_start, current_end))
                    
                    # Log chunk details for debugging
                    logger.info(f"📄 Creating chunk: start={chunk_start:.2f}s, end={chunk_end:.2f}s, duration={chunk_end-chunk_start:.2f}s")
                    
                    chunks.append({
                        'text': ' '.join(current_text_parts).strip(),
                        'start': chunk_start,
                        'end': chunk_end,
                    })
                current_text_parts = []
                current_start = None
                current_end = None

            for seg in segments:
                seg_text = (seg.get('text') or '').strip()
                if not seg_text:
                    continue
                seg_start = float(seg.get('start', 0.0))
                seg_end = float(seg.get('end', seg_start))

                if current_start is None:
                    current_start = seg_start
                    current_end = seg_end
                else:
                    current_end = seg_end

                current_text_parts.append(seg_text)

                # Heuristics to flush chunk: length or duration cap (YouTube-style)
                current_text_len = sum(len(p) for p in current_text_parts) + (len(current_text_parts) - 1)
                current_duration = current_end - (current_start or current_end)
                
                # More aggressive chunking for better timestamp precision (like YouTube)
                if current_text_len >= chunk_size or current_duration >= max_chunk_duration:
                    flush_chunk()

            # Flush remaining
            flush_chunk()

            logger.info(f"📄 Created {len(chunks)} timestamped chunks from segments")
            
            # Log detailed chunk information (YouTube-style debugging)
            for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                logger.info(f"    Chunk {i+1}: [{chunk['start']:.2f}s → {chunk['end']:.2f}s] {chunk['text'][:100]}...")
            
            return chunks

        # Fallback: character-based chunks without timestamps
        fallback_chunks: List[Dict] = []
        pos = 0
        while pos < len(transcription):
            end = pos + chunk_size
            if end < len(transcription):
                for i in range(end, max(pos + chunk_size - 100, pos), -1):
                    if transcription[i] in '.!?':
                        end = i + 1
                        break
            text_chunk = transcription[pos:end].strip()
            if text_chunk:
                fallback_chunks.append({'text': text_chunk, 'start': 0.0, 'end': 0.0})
            pos = end - overlap
            if pos >= len(transcription):
                break
        logger.info(f"📄 Created {len(fallback_chunks)} chunks from transcription (no timestamps)")
        return fallback_chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks using OpenAI
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🧠 Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch size limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"✅ Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 1536  # OpenAI embedding dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return []
    
    def store_in_pinecone(self, company_name: str, video_url: str, video_info: Dict, 
                         transcription_data: Dict, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Store transcription chunks and embeddings in Pinecone
        
        Args:
            company_name: Name of the company
            video_url: Original video URL
            video_info: Video metadata
            transcription_data: Transcription data
            chunks: Text chunks
            embeddings: Embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗄️ Storing in Pinecone for company: {company_name}")
            
            # Create or get single shared index
            index_name = self.default_index_name
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                try:
                    logger.info(f"📊 Creating new Pinecone index: {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    # Wait for index to be ready
                    time.sleep(10)
                except Exception as ce:
                    msg = str(ce)
                    if 'max serverless indexes' in msg.lower() or 'forbidden' in msg.lower():
                        if existing_indexes:
                            fallback = existing_indexes[0]
                            logger.warning(f"⚠️ Index quota reached; falling back to existing index: {fallback}")
                            index_name = fallback
                        else:
                            logger.error("❌ No existing Pinecone indexes available to fallback to.")
                            raise
                    else:
                        raise
            
            # Get index and namespace per company
            index = self.pc.Index(index_name)
            namespace = company_name.lower().replace(' ', '-')
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{company_name}_{video_url}_{i}"
                
                # Extract and validate timestamps
                chunk_start = float(chunk.get('start', 0.0)) if isinstance(chunk, dict) else 0.0
                chunk_end = float(chunk.get('end', 0.0)) if isinstance(chunk, dict) else 0.0
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company': company_name,
                        'video_url': video_url,
                        'chunk_index': i,
                        'text': chunk['text'] if isinstance(chunk, dict) else str(chunk),
                        'start': chunk_start,
                        'end': chunk_end,
                        'title': video_info.get('title', 'Unknown'),
                        'duration': video_info.get('duration', 'Unknown'),
                        'language': transcription_data.get('language', 'Unknown'),
                        'word_count': transcription_data.get('word_count', 'Unknown'),
                        'source_type': 'loom_video'
                    }
                }
                
                # Debug timestamp storage
                if chunk_start > 0.0 or chunk_end > 0.0:
                    logger.info(f"💾 Storing chunk {i+1}: start={chunk_start:.2f}s, end={chunk_end:.2f}s")
                
                vectors.append(vector_data)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"✅ Upserted batch {i//batch_size + 1}")
            
            logger.info(f"✅ Successfully stored {len(vectors)} vectors in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone storage failed: {e}")
            return False
    
    def process_video(self, video_url: str, company_name: str) -> Optional[Dict]:
        """
        Complete Loom video processing pipeline
        
        Args:
            video_url: Loom video URL
            company_name: Company name for organization
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            logger.info(f"🎯 Processing Loom video: {video_url}")
            
            # Step 1: Extract video info
            video_info = self.extract_loom_video_info(video_url)
            if not video_info:
                raise Exception("Failed to extract video info")
            
            # Step 2: Download video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            download_success = self.download_loom_video(video_info['video_url'], temp_video_path)
            if not download_success:
                # Try yt-dlp as fallback
                logger.info("🔄 Trying yt-dlp fallback for video download")
                try:
                    import subprocess
                    import sys

                    # Ensure parent dir exists and remove any empty pre-created file
                    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
                    try:
                        if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) == 0:
                            os.remove(temp_video_path)
                            logger.info("🧹 Removed empty temp file before yt-dlp download")
                    except Exception:
                        pass

                    # Use yt-dlp to download the best available format and merge to mp4 if needed
                    cmd = [
                        sys.executable, '-m', 'yt_dlp',
                        '--no-warnings',
                        '--retries', '3', '--fragment-retries', '3',
                        '--restrict-filenames',
                        '--merge-output-format', 'mp4',
                        '--force-overwrites',
                        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        '--referer', 'https://www.loom.com/',
                        '--add-header', 'Origin: https://www.loom.com',
                        '--add-header', 'Sec-Fetch-Mode: navigate',
                        '--output', temp_video_path,
                        video_url
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                    # yt-dlp may append extension; normalize to expected path
                    if result.returncode == 0 and not os.path.exists(temp_video_path):
                        candidate_mp4 = temp_video_path if temp_video_path.endswith('.mp4') else f"{temp_video_path}.mp4"
                        if os.path.exists(candidate_mp4):
                            try:
                                os.replace(candidate_mp4, temp_video_path)
                            except Exception:
                                pass

                    if result.returncode == 0 and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1024:
                        logger.info("✅ Video downloaded successfully with yt-dlp")
                        download_success = True
                    else:
                        logger.error(f"❌ yt-dlp failed (code {result.returncode}): {result.stderr or result.stdout}")
                        raise Exception("Failed to download video with yt-dlp")

                except Exception as e:
                    logger.error(f"❌ yt-dlp fallback failed: {e}")
                    raise Exception("Failed to download video")
            
            if not download_success:
                raise Exception("Failed to download video")
            
            try:
                # Step 3: Transcribe video
                transcription_data = self.transcribe_video(temp_video_path)
                if not transcription_data:
                    raise Exception("Failed to transcribe video")
                
                # Step 4: Chunk the transcription
                transcription = transcription_data.get('transcription', '')
                if not transcription:
                    raise Exception("Empty transcription")
                segments = transcription_data.get('segments', [])
                
                # Use smaller chunks for Loom videos for better timestamp precision
                chunk_size = 800  # Reduced from 1000
                max_chunk_duration = 45  # Reduced from 60 seconds
                
                chunks = self.chunk_transcription(transcription, segments=segments, 
                                                chunk_size=chunk_size, 
                                                max_chunk_duration=max_chunk_duration)
                if not chunks:
                    raise Exception("Failed to create chunks")
                
                # Step 5: Create embeddings
                embeddings = self.create_embeddings([c['text'] if isinstance(c, dict) else str(c) for c in chunks])
                if not embeddings or len(embeddings) != len(chunks):
                    raise Exception("Failed to create embeddings")
                
                # Step 6: Store in Pinecone
                storage_success = self.store_in_pinecone(
                    company_name, video_url, video_info, transcription_data, chunks, embeddings
                )

                # Log a brief timestamp summary for Loom chunks
                try:
                    total = len(chunks)
                    with_ts = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
                    logger.info(f"🧪 Timestamp chunk check (Loom - {company_name}): {with_ts}/{total} chunks have timestamps")
                    for i, c in enumerate(chunks[: min(5, total)]):
                        s = float(c.get('start', 0.0))
                        e = float(c.get('end', 0.0))
                        t = (c.get('text') or '')[:120].replace('\n', ' ')
                        logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
                except Exception:
                    pass
                
                if not storage_success:
                    raise Exception("Failed to store in Pinecone")
                
                # Return success result
                result = {
                    'success': True,
                    'video_url': video_url,
                    'company_name': company_name,
                    'title': video_info.get('title', 'Unknown'),
                    'chunks_created': len(chunks),
                    'vectors_stored': len(embeddings),
                    'word_count': transcription_data.get('word_count', 'Unknown'),
                    'language': transcription_data.get('language', 'Unknown'),
                    'method': 'loom_transcription'
                }
                
                logger.info(f"✅ Loom video processing completed successfully")
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                    logger.info("🧹 Cleaned up temporary video file")
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            return None
    
    def search_similar_chunks(self, company_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks in Pinecone
        
        Args:
            company_name: Company name
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            if not query_embedding:
                raise Exception("Failed to create query embedding")
            
            # Get index
            index_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            index = self.pc.Index(index_name)
            
            # Search
            results = index.query(
                vector=query_embedding[0],
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'video_url': match.metadata.get('video_url', ''),
                    'title': match.metadata.get('title', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
