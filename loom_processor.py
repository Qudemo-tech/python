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
    
    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """
        Transcribe video using Whisper
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with transcription data or None if failed
        """
        try:
            logger.info(f"🎤 Transcribing video: {video_path}")
            
            # Load Whisper model
            model = self.get_whisper_model()
            
            # Transcribe video
            result = model.transcribe(video_path)
            
            transcription_data = {
                'transcription': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'en'),
                'word_count': len(result['text'].split())
            }
            
            logger.info(f"✅ Transcription completed: {transcription_data['word_count']} words")
            return transcription_data
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def chunk_transcription(self, transcription: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split transcription into overlapping chunks for better vector search
        
        Args:
            transcription: Full transcription text
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(transcription):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(transcription):
                # Look for sentence endings near the end
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if transcription[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = transcription[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(transcription):
                break
        
        logger.info(f"📄 Created {len(chunks)} chunks from transcription")
        return chunks
    
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
                         transcription_data: Dict, chunks: List[str], embeddings: List[List[float]]) -> bool:
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
            
            # Create or get index
            index_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
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
            
            # Get index
            index = self.pc.Index(index_name)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{company_name}_{video_url}_{i}"
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company': company_name,
                        'video_url': video_url,
                        'chunk_index': i,
                        'text': chunk,
                        'title': video_info.get('title', 'Unknown'),
                        'duration': video_info.get('duration', 'Unknown'),
                        'language': transcription_data.get('language', 'Unknown'),
                        'word_count': transcription_data.get('word_count', 'Unknown'),
                        'source_type': 'loom_video'
                    }
                }
                vectors.append(vector_data)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
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
                
                chunks = self.chunk_transcription(transcription)
                if not chunks:
                    raise Exception("Failed to create chunks")
                
                # Step 5: Create embeddings
                embeddings = self.create_embeddings(chunks)
                if not embeddings or len(embeddings) != len(chunks):
                    raise Exception("Failed to create embeddings")
                
                # Step 6: Store in Pinecone
                storage_success = self.store_in_pinecone(
                    company_name, video_url, video_info, transcription_data, chunks, embeddings
                )
                
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
