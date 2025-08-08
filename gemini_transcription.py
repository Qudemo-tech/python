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
        # Use gemini-1.5-flash model for optimized speed and multimodal input
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
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
            
            # Use Gemini's native YouTube URL processing with correct API method
            try:
                # Use the raw API call with the exact structure from the working Insomnia example
                import requests
                
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                
                headers = {
                    "Content-Type": "application/json",
                }
                
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": "Transcribe only the spoken words from this YouTube video. Do not include any summaries, descriptions, or additional text. Just provide the raw, continuous transcription."
                                },
                                {
                                    "fileData": {
                                        "mimeType": "video/mp4",
                                        "fileUri": video_url
                                    }
                                }
                            ]
                        }
                    ]
                }
                
                # Call Gemini API with the exact structure from Insomnia
                logger.info("Sending request to Gemini API...")
                response = requests.post(
                    f"{url}?key={self.gemini_api_key}",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        transcription_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("✅ Request successful - Using Gemini raw API video analysis")
                        
                        # Create result structure
                        result_dict = {
                            "title": "YouTube Video",
                            "transcription": transcription_text,
                            "summary": "",
                            "duration": "Unknown",
                            "language": "en",
                            "word_count": len(transcription_text.split()),
                            "method": "gemini_raw_api_analysis"
                        }
                        
                        # Log the transcription
                        logger.info(f"✅ Transcription extracted successfully")
                        logger.info(f"📝 Word count: {len(transcription_text.split())}")
                        logger.info(f"🔧 Method: gemini_raw_api_analysis")
                        
                        # Log the full transcription content
                        logger.info("📄 FULL TRANSCRIPTION CONTENT:")
                        logger.info("=" * 80)
                        logger.info(transcription_text[:2000] + ("..." if len(transcription_text) > 2000 else ""))
                        logger.info("=" * 80)
                        if len(transcription_text) > 2000:
                            logger.info(f"📄 (Showing first 2000 characters of {len(transcription_text)} total)")
                        
                        # Save transcript to file for logging purposes
                        self._save_transcript_to_file(video_url, transcription_text, result_dict)
                        
                        return result_dict
                    else:
                        raise Exception("No candidates in response")
                else:
                    raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
            except Exception as e:
                # Fallback to text-only approach if video processing fails
                logger.warning(f"⚠️ Using fallback text-only approach: {e}")
                prompt = f"""
                Please extract the full transcription from this YouTube video: {video_url}
                
                Transcribe only the spoken words from this YouTube video. Do not include any summaries, descriptions, or additional text. Just provide the raw, continuous transcription.
                """
                
                # Call Gemini API
                response = self.model.generate_content(prompt)
            
            # Response handling is now done directly in the try block above
            else:
                logger.error("❌ Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini transcription failed: {e}")
            return None
    
    def _save_transcript_to_file(self, video_url: str, transcription_text: str, result: Dict):
        """
        Save transcript to file for logging purposes (similar to tutorial)
        """
        try:
            import os
            from datetime import datetime
            
            # Create logs directory if it doesn't exist
            logs_dir = "transcript_logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"
            filepath = os.path.join(logs_dir, filename)
            
            # Write transcript to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"YouTube Video URL: {video_url}\n\n")
                f.write(f"Title: {result.get('title', 'Unknown')}\n")
                f.write(f"Duration: {result.get('duration', 'Unknown')}\n")
                f.write(f"Language: {result.get('language', 'Unknown')}\n")
                f.write(f"Word Count: {result.get('word_count', 'Unknown')}\n")
                f.write(f"Method: {result.get('method', 'Unknown')}\n\n")
                
                # Write summary if available
                summary = result.get('summary', '')
                if summary:
                    f.write("--- VIDEO SUMMARY ---\n")
                    f.write(summary)
                    f.write("\n\n")
                
                f.write("--- TRANSCRIPTION ---\n")
                f.write(transcription_text)
            
            logger.info(f"📁 Transcript saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save transcript to file: {e}")

    def _fallback_video_analysis(self, video_url: str) -> Optional[Dict]:
        """
        Fallback method when direct transcript access fails
        Attempts to analyze video based on URL and available metadata
        """
        try:
            logger.info(f"🔄 Attempting fallback analysis for: {video_url}")
            
            # Extract video ID from URL
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                logger.error("❌ Could not extract video ID from URL")
                return None
            
            video_id = video_id_match.group(1)
            logger.info(f"📹 Extracted video ID: {video_id}")
            
            # Create a basic analysis prompt
            prompt = f"""
            Analyze this YouTube video based on its ID: {video_id}
            
            Please provide a summary of what this video might be about based on:
            1. The video ID pattern
            2. Common YouTube video content patterns
            3. Any available metadata
            
            Return in JSON format:
            {{
                "title": "Estimated video title",
                "transcription": "Summary of likely content based on video ID and patterns",
                "duration": "Unknown",
                "language": "en",
                "word_count": "Number of words in summary",
                "method": "fallback_analysis"
            }}
            """
            
            # Call Gemini for fallback analysis
            response = self.model.generate_content(prompt)
            
            if response.text:
                try:
                    result = json.loads(response.text)
                    logger.info(f"✅ Fallback analysis completed")
                    logger.info(f"📹 Estimated title: {result.get('title', 'Unknown')}")
                    logger.info(f"📝 Word count: {result.get('word_count', 'Unknown')}")
                    return result
                except json.JSONDecodeError:
                    logger.warning("⚠️ Fallback response not in JSON format")
                    return {
                        "title": f"YouTube Video ({video_id})",
                        "transcription": f"Video analysis for {video_id}. Content could not be directly accessed due to YouTube restrictions.",
                        "duration": "Unknown",
                        "language": "en",
                        "word_count": len(response.text.split()),
                        "method": "fallback_analysis"
                    }
            else:
                logger.error("❌ Empty fallback response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Fallback analysis failed: {e}")
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
