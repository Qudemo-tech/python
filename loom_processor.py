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
                
                # Construct API URL for Loom video
                api_url = f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-video"
                
                logger.info(f"🔗 Fetching video from Loom API: {api_url}")
                
                # Fetch video info
                response = requests.get(api_url, timeout=30)
                response.raise_for_status()
                
                video_data = response.json()
                
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
                
            else:
                logger.error("❌ Invalid Loom URL format")
                return None
                
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
            
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Loom video downloaded successfully")
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
