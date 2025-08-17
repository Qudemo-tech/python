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
        # Create instance and use fetch method
        api = YouTubeTranscriptApi()
        if languages:
            return api.fetch(video_id, languages=languages)
        else:
            return api.fetch(video_id)
    except Exception as e:
        logger.warning(f"⚠️ YouTube Transcript API failed for {video_id}: {e}")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        logger.info("🔧 Initializing Gemini Transcription Processor...")

    def _log_chunk_summary(self, chunks: List[Dict], label: str = ""):
        try:
            total = len(chunks)
            with_ts = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
            logger.info(f"🧪 Timestamp chunk check{(' - ' + label) if label else ''}: {with_ts}/{total} chunks have timestamps")
            
            if with_ts > 0:
                logger.info("✅ Timestamps found - chunks will provide timestamped Q&A")
                for i, c in enumerate(chunks[: min(5, total)]):
                    s = float(c.get('start', 0.0))
                    e = float(c.get('end', 0.0))
                    t = (c.get('text') or '')[:120].replace('\n', ' ')
                    logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
            else:
                logger.warning("⚠️ No timestamps found - Q&A will not include timestamps")
                for i, c in enumerate(chunks[: min(5, total)]):
                    t = (c.get('text') or '')[:120].replace('\n', ' ')
                    logger.info(f"    #{i+1}: [NO TIMESTAMP] {t}")
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
            
            # Try Gemini API with retry logic
            gemini_result = self._try_gemini_api(video_url)
            if gemini_result:
                return gemini_result
            
            # Fallback to YouTube Transcript API
            youtube_result = self._try_youtube_api(video_url)
            if youtube_result:
                return youtube_result
            
            # Final fallback: Create basic transcription
            fallback_result = self._create_fallback_transcription(video_url)
            if fallback_result:
                return fallback_result
            
            return None
                
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None

    def _try_gemini_api(self, video_url: str) -> Optional[Dict]:
        """Try Gemini API with enhanced retry logic for overload handling"""
        max_retries = 5  # Increased from 3 to 5
        base_delay = 3   # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
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
                
                logger.info(f"Sending request to Gemini API... (attempt {attempt + 1}/{max_retries})")
                response = requests.post(
                    f"{url}?key={self.gemini_api_key}",
                    headers=headers,
                    json=data,
                    timeout=90  # Increased timeout from 60 to 90 seconds
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        transcription_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("✅ Gemini API successful - Using raw API video analysis")
                        
                        result_dict = {
                            "title": "YouTube Video",
                            "transcription": transcription_text,
                            "summary": "",
                            "duration": "Unknown",
                            "language": "en",
                            "word_count": len(transcription_text.split()),
                            "method": "gemini_raw_api_analysis"
                        }
                        
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
                        
                        self._save_transcript_to_file(video_url, transcription_text, result_dict)
                        return result_dict
                    else:
                        raise Exception("No candidates in response")
                else:
                    # Enhanced retry logic with exponential backoff
                    if response.status_code == 503 and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 3s, 6s, 12s, 24s
                        logger.warning(f"⚠️ Gemini API overloaded (503), retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    elif response.status_code in [429, 500, 502] and attempt < max_retries - 1:
                        # Handle other rate limiting and server errors
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Gemini API error ({response.status_code}), retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Gemini API timeout, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.warning(f"⚠️ Gemini API failed after {max_retries} attempts due to timeouts")
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Gemini API attempt {attempt + 1} failed: {e}, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.warning(f"⚠️ Gemini API failed after {max_retries} attempts: {e}")
        
        return None

    def _try_youtube_api(self, video_url: str) -> Optional[Dict]:
        """Try YouTube Transcript API as fallback"""
        try:
            segments = self.fetch_youtube_segments(video_url)
            if segments:
                transcription_text = " ".join([segment.get('text', '') for segment in segments])
                
                logger.info("✅ YouTube Transcript API fallback successful")
                
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
                
                logger.info(f"✅ Transcription extracted successfully via YouTube API")
                logger.info(f"📝 Word count: {len(transcription_text.split())}")
                logger.info(f"🔧 Method: youtube_transcript_api_fallback")
                
                self._save_transcript_to_file(video_url, transcription_text, result_dict)
                return result_dict
            else:
                raise Exception("YouTube Transcript API returned no segments")
        except Exception as e:
            logger.error(f"❌ YouTube Transcript API failed: {e}")
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
            transcript = get_youtube_transcript(video_id, languages=['en'])
            if transcript is None:
                transcript = get_youtube_transcript(video_id)
            
            if transcript is None:
                logger.warning(f"⚠️ No transcript available for video {video_id}")
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
            # Silently fail - this is just a fallback
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

        # Fallback: parse inline timestamps if present in the text
        import re
        
        # Log the first few lines to debug timestamp format
        first_lines = transcription[:500].split('\n')[:5]
        logger.info(f"🔍 Debug: First 5 lines of transcription:")
        for i, line in enumerate(first_lines):
            logger.info(f"  Line {i+1}: {line}")
        
        # Try different timestamp formats: [HH:MM:SS], [MM:SS], [SS]
        patterns = [
            r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)",  # [HH:MM:SS]
            r"\[(\d{2}):(\d{2})\]\s*(.+)",           # [MM:SS]
            r"\[(\d+)\]\s*(.+)"                       # [SS]
        ]
        
        for pattern_str in patterns:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(transcription)
            logger.info(f"🔍 Pattern '{pattern_str}' found {len(matches)} matches")
            if matches:
                parsed: List[Dict] = []
                for idx, match in enumerate(matches):
                    if len(match) == 4:  # [HH:MM:SS] format
                        hh, mm, ss, sent = match
                        start = int(hh) * 3600 + int(mm) * 60 + int(ss)
                    elif len(match) == 3:  # [MM:SS] format
                        mm, ss, sent = match
                        start = int(mm) * 60 + int(ss)
                    elif len(match) == 2:  # [SS] format
                        ss, sent = match
                        start = int(ss)
                    else:
                        continue
                    
                    # End at next start or start + heuristic duration
                    if idx + 1 < len(matches):
                        next_match = matches[idx + 1]
                        if len(next_match) == 4:  # [HH:MM:SS] format
                            nhh, nmm, nss, _ = next_match
                            end = int(nhh) * 3600 + int(nmm) * 60 + int(nss)
                        elif len(next_match) == 3:  # [MM:SS] format
                            nmm, nss, _ = next_match
                            end = int(nmm) * 60 + int(nss)
                        elif len(next_match) == 2:  # [SS] format
                            nss, _ = next_match
                            end = int(nss)
                        else:
                            end = start + min(max(len(sent) // 15, 3), 20)
                    else:
                        end = start + min(max(len(sent) // 15, 3), 20)
                    
                    parsed.append({'text': sent.strip(), 'start': float(start), 'end': float(end)})
                
                logger.info(f"📄 Created {len(parsed)} chunks from inline timestamps (format: {pattern_str})")
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
                            logger.info(f"ℹ️ Using existing index: {fallback}")
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
                        'source_type': 'video'
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
                'method': transcription_data.get('method', 'gemini_transcription')  # Use the actual method from transcription_data
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


