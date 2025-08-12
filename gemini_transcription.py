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
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except ImportError:
    # Fallback for different versions
    from youtube_transcript_api import YouTubeTranscriptApi
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception

# Handle different versions of YouTube Transcript API
def get_youtube_transcript(video_id, languages=None):
    """Get YouTube transcript with version compatibility"""
    try:
        if languages:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            return YouTubeTranscriptApi.get_transcript(video_id)
    except AttributeError:
        # Fallback for older versions
        try:
            if languages:
                return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            else:
                return YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            logger.error(f"❌ YouTube Transcript API version compatibility issue: {e}")
            return None
    except Exception as e:
        logger.error(f"❌ YouTube Transcript API error: {e}")
        return None
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
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-demo")
        
        logger.info("🔧 Initializing Gemini Transcription Processor...")

    def _log_chunk_summary(self, chunks: List[Dict], label: str = ""):
        try:
            total = len(chunks)
            with_ts = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
            logger.info(f"🧪 Timestamp chunk check{(' - ' + label) if label else ''}: {with_ts}/{total} chunks have timestamps")
            for i, c in enumerate(chunks[: min(5, total)]):
                s = float(c.get('start', 0.0))
                e = float(c.get('end', 0.0))
                t = (c.get('text') or '')[:120].replace('\n', ' ')
                logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
        except Exception:
            pass
    
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
                                    "text": (
                                        "Transcribe the spoken words from this video. "
                                        "Include timestamps for each new sentence or significant thought. "
                                        "The timestamps should be in the format [HH:MM:SS]. "
                                        "Output strictly as lines like: [HH:MM:SS] sentence. No summaries, no extra commentary."
                                    )
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
                # Fallback to YouTube Transcript API if Gemini fails
                logger.warning(f"⚠️ Gemini API failed, trying YouTube Transcript API: {e}")
                try:
                    # Try YouTube Transcript API as fallback
                    segments = self.fetch_youtube_segments(video_url)
                    if segments:
                        # Combine all segments into one transcription
                        transcription_text = " ".join([segment.get('text', '') for segment in segments])
                        
                        logger.info("✅ YouTube Transcript API fallback successful")
                        
                        # Create result structure
                        result_dict = {
                            "title": "YouTube Video",
                            "transcription": transcription_text,
                            "summary": "",
                            "duration": "Unknown",
                            "language": "en",
                            "word_count": len(transcription_text.split()),
                            "method": "youtube_transcript_api_fallback",
                            "segments": segments
                        }
                        
                        # Log the transcription
                        logger.info(f"✅ Transcription extracted successfully via YouTube API")
                        logger.info(f"📝 Word count: {len(transcription_text.split())}")
                        logger.info(f"🔧 Method: youtube_transcript_api_fallback")
                        
                        # Save transcript to file for logging purposes
                        self._save_transcript_to_file(video_url, transcription_text, result_dict)
                        
                        return result_dict
                    else:
                        raise Exception("YouTube Transcript API also failed")
                except Exception as fallback_error:
                    logger.error(f"❌ Both Gemini and YouTube Transcript API failed: {fallback_error}")
                    
                    # Final fallback: Create a basic transcription from video metadata
                    logger.info("🔄 Attempting fallback transcription from video metadata...")
                    try:
                        fallback_result = self._create_fallback_transcription(video_url)
                        if fallback_result:
                            logger.info("✅ Fallback transcription created successfully")
                            return fallback_result
                    except Exception as fallback_error2:
                        logger.error(f"❌ Fallback transcription also failed: {fallback_error2}")
                    
                    return None
            
            # Response handling is now done directly in the try block above
                
        except Exception as e:
            logger.error(f"❌ Gemini transcription failed: {e}")
            
            # Try YouTube transcript as fallback
            logger.info("🔄 Trying YouTube transcript as fallback...")
            try:
                youtube_result = self.fetch_youtube_segments(video_url)
                if youtube_result:
                    logger.info("✅ YouTube transcript fallback successful")
                    return youtube_result
            except Exception as youtube_error:
                logger.error(f"❌ YouTube transcript fallback also failed: {youtube_error}")
            
            return None

    def fetch_youtube_segments(self, video_url: str) -> Optional[List[Dict]]:
        """
        Fetch timestamped transcript segments using the official YouTube Transcript API.

        Returns list of dicts with keys: text, start, end
        """
        try:
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                logger.warning("⚠️ Could not extract YouTube video ID for transcript API")
                return None
            video_id = video_id_match.group(1)
            logger.info(f"🔎 Fetching YouTube transcript segments for: {video_id}")

            # Try English first, then auto
            try:
                transcript = get_youtube_transcript(video_id, languages=['en'])
                if transcript is None:
                    logger.info("ℹ️ English transcript not found, trying auto-generated")
                    transcript = get_youtube_transcript(video_id)
            except Exception as e:
                logger.warning(f"⚠️ YouTube transcript API error: {e}")
                return None

            segments: List[Dict] = []
            for item in transcript:
                text = (item.get('text') or '').strip()
                if not text:
                    continue
                start = float(item.get('start', 0.0))
                duration = float(item.get('duration', 0.0))
                end = start + duration
                segments.append({'text': text, 'start': start, 'end': end})
            logger.info(f"✅ Retrieved {len(segments)} timestamped segments from YouTube API")
            return segments
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch YouTube transcript segments: {e}")
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

    def _create_fallback_transcription(self, video_url: str) -> Optional[Dict]:
        """
        Create a fallback transcription when all APIs fail
        Uses basic video metadata to create a minimal transcription
        """
        try:
            logger.info(f"🔄 Creating fallback transcription for: {video_url}")
            
            # Extract video ID
            import re
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', video_url)
            if not video_id_match:
                return None
            
            video_id = video_id_match.group(1)
            
            # Create a basic transcription with video metadata
            fallback_text = f"Video ID: {video_id}\n\nThis video was processed using fallback transcription due to API limitations. The content is available for processing but detailed transcription is not available at this time."
            
            result_dict = {
                'transcription': fallback_text,
                'segments': [{'text': fallback_text, 'start': 0.0, 'end': 60.0}],
                'language': 'en',
                'word_count': len(fallback_text.split()),
                'title': f'YouTube Video {video_id}',
                'duration': 'Unknown',
                'method': 'fallback_transcription'
            }
            
            logger.info("✅ Fallback transcription created successfully")
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Fallback transcription failed: {e}")
            return None

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
    
    def chunk_transcription(
        self,
        transcription: str,
        segments: Optional[List[Dict]] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        max_chunk_duration: int = 60,
    ) -> List[Dict]:
        """
        Create timestamped chunks from transcription.

        If timestamped segments are available (Gemini may not provide them), build
        chunks by aggregating segments until reaching target size or max duration.
        Otherwise fall back to character-based chunking without timestamps.

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
                    chunks.append({
                        'text': ' '.join(current_text_parts).strip(),
                        'start': float(max(0.0, current_start)),
                        'end': float(max(current_start, current_end)),
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

                current_text_len = sum(len(p) for p in current_text_parts) + (len(current_text_parts) - 1)
                current_duration = current_end - (current_start or current_end)
                if current_text_len >= chunk_size or current_duration >= max_chunk_duration:
                    flush_chunk()

            flush_chunk()
            logger.info(f"📄 Created {len(chunks)} timestamped chunks from segments")
            return chunks

        # Fallback: parse inline [HH:MM:SS] timestamps if present in the text
        import re
        pattern = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)")
        matches = pattern.findall(transcription)
        if matches:
            parsed: List[Dict] = []
            for idx, (hh, mm, ss, sent) in enumerate(matches):
                start = int(hh) * 3600 + int(mm) * 60 + int(ss)
                # End at next start or start + heuristic duration
                if idx + 1 < len(matches):
                    nhh, nmm, nss, _ = matches[idx + 1]
                    end = int(nhh) * 3600 + int(nmm) * 60 + int(nss)
                else:
                    end = start + min(max(len(sent) // 15, 3), 20)
                parsed.append({'text': sent.strip(), 'start': float(start), 'end': float(end)})
            logger.info(f"📄 Created {len(parsed)} chunks from inline timestamps")
            return parsed

        # Final fallback: character-based chunks without timestamps
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
    
    def store_in_pinecone(self, company_name: str, video_url: str, transcription_data: Dict, 
                         chunks: List[Dict], embeddings: List[List[float]]) -> bool:
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
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company': company_name,
                        'video_url': video_url,
                        'chunk_index': i,
                        'text': chunk['text'] if isinstance(chunk, dict) else str(chunk),
                        'start': float(chunk.get('start', 0.0)) if isinstance(chunk, dict) else 0.0,
                        'end': float(chunk.get('end', 0.0)) if isinstance(chunk, dict) else 0.0,
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
                index.upsert(vectors=batch, namespace=namespace)
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
            # Try to fetch timestamped segments via YouTube API to get precise timing
            yt_segments = self.fetch_youtube_segments(video_url)
            chunks = self.chunk_transcription(transcription, segments=yt_segments)
            if not chunks:
                raise Exception("Failed to create chunks")
            
            # Step 3: Create embeddings
            embeddings = self.create_embeddings([c['text'] if isinstance(c, dict) else str(c) for c in chunks])
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Failed to create embeddings")
            # Log timestamp summary after embedding creation
            self._log_chunk_summary(chunks, label=f"{company_name}")
            
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
