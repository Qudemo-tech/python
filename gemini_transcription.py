#!/usr/bin/env python3
"""
Gemini Transcription Module
Uses Google Gemini API to extract transcriptions from YouTube videos
"""

import os
import logging
import time
import json
from typing import Dict, Optional, List
from urllib.parse import urlparse
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiTranscriptionProcessor:
    def __init__(self, gemini_api_key: str, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize Gemini Transcription Processor
        
        Args:
            gemini_api_key: Google Gemini API key
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key for embeddings
        """
        self.gemini_api_key = gemini_api_key
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        logger.info("🔧 Initializing Gemini Transcription Processor...")
    
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain
    
    def extract_transcription_with_gemini(self, video_url: str) -> Optional[Dict]:
        """
        Extract transcription from YouTube video using Gemini API
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict with transcription data or None if failed
        """
        try:
            if not self.is_youtube_url(video_url):
                raise Exception("Not a YouTube URL")
            
            logger.info(f"🎬 Extracting transcription from: {video_url}")
            
            # Create prompt for Gemini
            prompt = f"""
            Please extract the full transcription from this YouTube video: {video_url}
            
            Return the transcription in the following JSON format:
            {{
                "title": "Video title",
                "transcription": "Full transcription text",
                "duration": "Video duration in seconds",
                "language": "Detected language",
                "word_count": "Number of words in transcription"
            }}
            
            If you cannot access the video or extract transcription, return:
            {{
                "error": "Error message",
                "accessible": false
            }}
            """
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if response.text:
                try:
                    # Try to parse as JSON
                    result = json.loads(response.text)
                    
                    if "error" in result:
                        logger.error(f"❌ Gemini extraction failed: {result['error']}")
                        return None
                    
                    logger.info(f"✅ Transcription extracted successfully")
                    logger.info(f"📹 Title: {result.get('title', 'Unknown')}")
                    logger.info(f"📝 Word count: {result.get('word_count', 'Unknown')}")
                    
                    return result
                    
                except json.JSONDecodeError:
                    # If not JSON, treat as plain transcription
                    logger.warning("⚠️ Response not in JSON format, treating as plain text")
                    return {
                        "title": "Unknown",
                        "transcription": response.text,
                        "duration": "Unknown",
                        "language": "Unknown",
                        "word_count": len(response.text.split())
                    }
            else:
                logger.error("❌ Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini transcription failed: {e}")
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
    
    def store_in_pinecone(self, company_name: str, video_url: str, transcription_data: Dict, 
                         chunks: List[str], embeddings: List[List[float]]) -> bool:
        """
        Store transcription chunks and embeddings in Pinecone
        
        Args:
            company_name: Name of the company
            video_url: Original video URL
            transcription_data: Transcription metadata
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
                        'title': transcription_data.get('title', 'Unknown'),
                        'duration': transcription_data.get('duration', 'Unknown'),
                        'language': transcription_data.get('language', 'Unknown'),
                        'word_count': transcription_data.get('word_count', 'Unknown'),
                        'source_type': 'youtube_gemini'
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
        Complete video processing pipeline
        
        Args:
            video_url: YouTube video URL
            company_name: Company name for organization
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            logger.info(f"🎯 Processing video: {video_url}")
            
            # Step 1: Extract transcription with Gemini
            transcription_data = self.extract_transcription_with_gemini(video_url)
            if not transcription_data:
                raise Exception("Failed to extract transcription")
            
            # Step 2: Chunk the transcription
            transcription = transcription_data.get('transcription', '')
            if not transcription:
                raise Exception("Empty transcription")
            
            chunks = self.chunk_transcription(transcription)
            if not chunks:
                raise Exception("Failed to create chunks")
            
            # Step 3: Create embeddings
            embeddings = self.create_embeddings(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Failed to create embeddings")
            
            # Step 4: Store in Pinecone
            storage_success = self.store_in_pinecone(
                company_name, video_url, transcription_data, chunks, embeddings
            )
            
            if not storage_success:
                raise Exception("Failed to store in Pinecone")
            
            # Return success result
            result = {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'title': transcription_data.get('title', 'Unknown'),
                'chunks_created': len(chunks),
                'vectors_stored': len(embeddings),
                'word_count': transcription_data.get('word_count', 'Unknown'),
                'language': transcription_data.get('language', 'Unknown'),
                'method': 'gemini_transcription'
            }
            
            logger.info(f"✅ Video processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
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

# Example usage
if __name__ == "__main__":
    # Load environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([gemini_api_key, pinecone_api_key, openai_api_key]):
        print("❌ Missing required API keys")
        exit(1)
    
    # Initialize processor
    processor = GeminiTranscriptionProcessor(gemini_api_key, pinecone_api_key, openai_api_key)
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    test_company = "TestCompany"
    
    print("🧪 Testing Gemini Transcription Processor...")
    
    # Test processing (commented out to avoid actual processing)
    # result = processor.process_video(test_url, test_company)
    # if result:
    #     print(f"✅ Processing result: {result}")
    # else:
    #     print("❌ Processing failed")
