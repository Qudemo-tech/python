from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict
import openai
import faiss
import numpy as np
import json
import os
import io
import re
import uuid
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
import whisper
import yt_dlp
from datetime import datetime
from supabase import create_client, Client
from pinecone import Pinecone
import urllib.request
import requests
from google.cloud import storage
import tempfile
import psutil
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"💾 Memory usage: {memory_mb:.1f}MB")
    return memory_mb

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = None  # Explicitly disable proxy usage

# --- Resource Loading and Answering Logic ---
RESOURCE_CACHE = {}

# Global video URL mapping
VIDEO_URL_MAPPING = {}

# Pinecone initialization
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "qudemo-index")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Helper: upsert vectors to Pinecone
def upsert_chunks_to_pinecone(company_name, chunks, embeddings):
    vectors = []
    logger.info(f"🔍 Storing {len(chunks)} chunks for company: {company_name}")
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{company_name}-{uuid.uuid4().hex[:8]}-{i}"
        meta = {
            "company_name": company_name,
            "text": chunk["text"],
            "context": chunk.get("context", ""),
            "source": chunk.get("source", ""),
            "original_video_url": chunk.get("original_video_url", ""),
            "type": chunk.get("type", "video"),
            "start": chunk.get("start"),
            "end": chunk.get("end")
        }
        vectors.append((vector_id, emb, meta))
        
        # Log first few chunks for debugging
        if i < 3:
            logger.info(f"📝 Chunk {i+1}: company={meta['company_name']}, source={meta['source']}, text_length={len(meta['text'])}")
    
    try:
        index.upsert(vectors)
        logger.info(f"✅ Successfully stored {len(vectors)} chunks in Pinecone for company: {company_name}")
    except Exception as e:
        logger.error(f"❌ Failed to store chunks in Pinecone: {e}")
        raise

# Helper: query Pinecone for similar chunks
def query_pinecone(company_name, embedding, top_k=6):
    logger.info(f"🔍 Querying Pinecone for company: {company_name}")
    try:
        result = index.query(vector=embedding, top_k=top_k, include_metadata=True, filter={"company_name": company_name})
        matches = result["matches"]
        logger.info(f"✅ Found {len(matches)} matches for company: {company_name}")
        
        # Log first few matches for debugging
        for i, match in enumerate(matches[:3]):
            meta = match.get("metadata", {})
            logger.info(f"📝 Match {i+1}: company={meta.get('company_name')}, source={meta.get('source')}, score={match.get('score', 0):.3f}")
        
        return matches
    except Exception as e:
        logger.error(f"❌ Pinecone query failed for company {company_name}: {e}")
        raise

def fetch_video_urls_from_supabase():
    """Fetch video URLs from Supabase videos table"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase credentials not found in environment variables")
            return {}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Query videos table to get video_url mappings using video_name as key
        response = supabase.table('videos').select('video_url, video_name').execute()
        
        video_mappings = {}
        for video in response.data:
            if video.get('video_url') and video.get('video_name'):
                video_name = video.get('video_name', '').strip()
                if video_name:
                    video_mappings[video_name] = video['video_url']
                    logger.info(f"📝 Mapped {video_name} -> {video['video_url']}")
        
        logger.info(f"📝 Fetched {len(video_mappings)} video mappings from Supabase")
        return video_mappings
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch video URLs from Supabase: {e}")
        return {}

def initialize_existing_mappings():
    """Initialize mappings for existing videos from Supabase"""
    global VIDEO_URL_MAPPING
    supabase_mappings = fetch_video_urls_from_supabase()
    VIDEO_URL_MAPPING.update(supabase_mappings)
    logger.info(f"📝 Initialized {len(VIDEO_URL_MAPPING)} video mappings from database")

# Initialize existing mappings when the module loads
initialize_existing_mappings()

def load_resources_for_company(company_name):
    # Use company name as bucket name (sanitized) - same logic as process_video
    bucket_name = company_name.lower().replace(' ', '_').replace('-', '_')
    transcript_json_path = "transcripts/transcript_chunks.json"
    faiss_gcs_path = "faiss_indexes/faiss_index.bin"
    faiss_local_path = f"faiss_index_{company_name}.bin"
    
    try:
        chunks = load_transcript_chunks(bucket_name, transcript_json_path)
        faiss_index = load_faiss_index(faiss_local_path, bucket_name, faiss_gcs_path)
        
        RESOURCE_CACHE[company_name] = {
            "chunks": chunks,
            "faiss_index": faiss_index,
            "bucket_name": bucket_name
        }
        return RESOURCE_CACHE[company_name]
    except Exception as e:
        logger.error(f"Failed to load resources for company {company_name}: {e}")
        raise

def get_resources(company_name):
    if company_name not in RESOURCE_CACHE:
        return load_resources_for_company(company_name)
    return RESOURCE_CACHE[company_name]

def answer_question(company_name, question):
    try:
        logger.info(f"QUESTION for {company_name}: {question}")
        # Create embedding for the question
        try:
            q_embedding = openai.embeddings.create(
                input=[question],
                model="text-embedding-3-small",
                timeout=15
            ).data[0].embedding
            logger.info("✅ Created embedding for the question.")
        except Exception as e:
            logger.error(f"❌ Failed to create question embedding: {e}")
            return {"error": "Failed to create question embedding."}
        # Query Pinecone
        try:
            matches = query_pinecone(company_name, q_embedding, top_k=6)
            top_chunks = [m["metadata"] for m in matches]
            logger.info(f"🔎 Retrieved top {len(top_chunks)} chunks from Pinecone.")
        except Exception as e:
            logger.error(f"❌ Pinecone query failed: {e}")
            return {"error": f"Pinecone query failed: {e}"}
        
        # Rerank chunks using GPT-3.5
        try:
            rerank_prompt = f"Question: {question}\n\nHere are the chunks:\n"
            for i, chunk in enumerate(top_chunks):
                snippet = chunk["text"][:500].strip().replace("\n", " ")
                rerank_prompt += f"{i+1}. [{chunk.get('type', 'video')}] {chunk.get('context','')}\n{snippet}\n\n"
            rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."
            rerank_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": rerank_prompt}],
                timeout=20
            )
            best_index = int(re.findall(r"\d+", rerank_response.choices[0].message.content)[0]) - 1
            best_chunk = top_chunks[best_index]
            logger.info(f"🏅 GPT-3.5-turbo reranked chunk #{best_index+1} as the most relevant.")
        except Exception as e:
            best_chunk = top_chunks[0]
            logger.warning(f"⚠️ Reranking failed, falling back to top Pinecone chunk: {e}")
        
        # Generate answer using GPT-4
        try:
            context = "\n\n".join([
                f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks[:3]
            ])
            system_prompt = (
                f"You are a product expert bot with full knowledge of {company_name} derived from video transcripts. "
                "Use clear, confident, and concise answers—no more than 700 characters. "
                "Use bullet points or short paragraphs if needed. Do not include inline citations like [source](...)."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            completion = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=20
            )
            raw_answer = completion.choices[0].message.content
            logger.info("✅ Generated answer with GPT-4.")
        except Exception as e:
            logger.error(f"❌ Failed to generate GPT-4 answer: {e}")
            return {"error": "Failed to generate answer."}
        
        def strip_sources(text):
            return re.sub(r'\[source\]\([^)]+\)', '', text).strip()
        
        def format_answer(text):
            text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
            text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
            return re.sub(r'\n+', '\n', text).strip()
        
        raw_answer = strip_sources(raw_answer)
        clean_answer = format_answer(raw_answer)
        sources = [chunk["source"] for chunk in top_chunks]

        # If the answer is 'I do not have that information' or similar, do not return video info
        no_info_phrases = [
            "I do not have that information",
            "I don't have that information",
            "I do not know",
            "I don't know",
            "no information available",
            "I couldn't find any information",
            "Sorry, I couldn't find",
            "Sorry, I do not have"
        ]
        if any(phrase.lower() in clean_answer.lower() for phrase in no_info_phrases):
            return {
                "answer": clean_answer,
                "sources": sources
            }

        # Get original video URL using metadata
        video_url = best_chunk.get("original_video_url")
        if not video_url:
            # Try to resolve from the chunk's source filename
            source_filename = best_chunk.get("source", "").split(" [")[0].strip()
            video_url = get_original_video_url(source_filename)
            if not video_url:
                logger.warning(f"⚠️ No video mapping found for: {source_filename}")
                video_url = best_chunk.get("source")
        logger.info(f"📤 Returning final answer. Video URL: {video_url}")
        
        # Get more accurate start/end from top 3 chunks
        relevant_chunks = top_chunks[:3]
        start = min(chunk.get("start", 0) for chunk in relevant_chunks)
        end = max(chunk.get("end", 0) for chunk in relevant_chunks)
        # Add a 2-second buffer before and after
        start = max(0, start - 2)
        end = end + 2
        
        return {
            "answer": clean_answer,
            "sources": sources,
            "video_url": video_url,
            "start": start,
            "end": end
        }
    except Exception as e:
        logger.error(f"Error in answer_question for {company_name}: {e}")
        return {"error": f"Failed to answer question: {str(e)}"}

def fetch_cookies_from_supabase(bucket_name, file_name, destination_path):
    """Download cookies file from Supabase Storage."""
    import os
    from supabase import create_client
    logger.info(f"Attempting to download cookies from Supabase Storage: bucket='{bucket_name}', file='{file_name}'")
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        # Use service role key for backend admin access to storage
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase URL or Service Role Key not found.")
            return

        supabase = create_client(supabase_url, supabase_key)
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with open(destination_path, "wb+") as f:
            res = supabase.storage.from_(bucket_name).download(file_name)
            f.write(res)
        
        logger.info(f"✅ Downloaded cookies from Supabase Storage to {destination_path}")

    except Exception as e:
        import traceback
        logger.error(f"❌ Failed to download cookies from Supabase Storage: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())


# Video processing functions
def download_video(video_url, output_filename):
    """Download video from URL using yt-dlp, using cookies if available, or handle local file paths."""
    # If it's a local file path, just return it
    if os.path.exists(video_url):
        logger.info(f"Using local file: {video_url}")
        return video_url
    
    # Check if it's a Loom URL
    if 'loom.com' in video_url:
        logger.warning(f"⚠️ Loom URL detected in fallback download: {video_url}")
        logger.warning(f"⚠️ This should be handled by LoomVideoProcessor, not fallback download")
        raise Exception("Loom videos should be processed by LoomVideoProcessor. yt-dlp download failed.")
    
    # If it's a YouTube link, use yt-dlp
    if video_url.startswith('http') and ('youtube.com' in video_url or 'youtu.be' in video_url):
        cookies_bucket = "cookies"
        cookies_file = "www.youtube.com_cookies.txt"
        cookies_path = os.path.join(tempfile.gettempdir(), "cookies.txt")
        fetch_cookies_from_supabase(cookies_bucket, cookies_file, cookies_path)
        
        ydl_opts = {
            'format': 'worst[height<=480]',  # Use lowest quality to save memory
            'outtmpl': output_filename,
            'quiet': True,
            'max_filesize': '50M',  # Limit file size to 50MB
            'cookiefile': cookies_path,
        }
        if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
            ydl_opts['cookiefile'] = cookies_path
            logger.info(f"Using cookies file for yt-dlp: {cookies_path}")
        else:
            logger.info("No cookies file found, downloading without authentication.")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            error_msg = str(e)
            # Check for common restriction errors
            if (
                'Sign in to confirm you\'re not a bot' in error_msg or
                'This video is age-restricted' in error_msg or
                'does not look like a Netscape format cookies file' in error_msg or
                'This video is private' in error_msg or
                'HTTP Error 403' in error_msg or
                'This video is unavailable' in error_msg
            ):
                logger.error(f"❌ Video download failed due to YouTube restrictions: {error_msg}")
                raise Exception("This YouTube video cannot be processed because it is restricted. Please provide a public, unrestricted video or upload your own file.")
            else:
                logger.error(f"❌ Video download failed: {error_msg}")
                raise
        return output_filename
    
    # If it's another HTTP(S) URL, download it directly
    if video_url.startswith('http'):
        logger.info(f"Downloading video from direct URL: {video_url}")
        # Use streaming to avoid loading entire file into memory
        r = requests.get(video_url, stream=True, timeout=30)
        r.raise_for_status()
        
        # Check file size before downloading
        content_length = r.headers.get('content-length')
        if content_length:
            file_size_mb = int(content_length) / (1024 * 1024)
            if file_size_mb > 50:  # Limit to 50MB
                raise Exception(f"File too large: {file_size_mb:.1f}MB (max 50MB)")
        
        with open(output_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
        
        # Validate the downloaded file
        try:
            from loom_video_processor import LoomVideoProcessor
            processor = LoomVideoProcessor()
            if not processor.validate_video_file_enhanced(output_filename):
                os.remove(output_filename)
                raise Exception("Downloaded file appears to be invalid (possibly HTML error page)")
        except Exception as e:
            logger.error(f"❌ Downloaded file validation failed: {e}")
            if os.path.exists(output_filename):
                os.remove(output_filename)
            raise
        
        return output_filename
    
    raise Exception("Invalid video input: must be a file path or a valid URL")

def transcribe_video(video_path, company_name, original_video_url=None):
    """Transcribe video using Whisper and return chunks"""
    # Use tiny model to save memory
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path, task="translate",verbose=True)
    
    # Generate context
    try:
        context_input = "\n".join([seg["text"] for seg in result["segments"][:10]])
        context_prompt = f"Summarize the main topic or context of this transcript:\n\n{context_input}"
        context_resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize transcripts."},
                {"role": "user", "content": context_prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )
        context = context_resp.choices[0].message.content.strip()
        logger.info(f"📘 Context: {context}")
    except Exception as e:
        logger.error(f"❌ Context generation failed: {e}")
        context = "No context available."
    
    # Format time for chunks
    def format_time(t):
        h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace('.', ',')
    
    # Create transcript chunks
    chunks = []
    for seg in result["segments"]:
        chunk = {
            "source": f"{os.path.basename(video_path)} [{format_time(seg['start'])} - {format_time(seg['end'])}]",
            "original_video_url": original_video_url,  # Store the original video URL
            "text": seg["text"].strip(),
            "context": context,
            "type": "video",
            "start": seg["start"],
            "end": seg["end"]
        }
        chunks.append(chunk)
    
    return chunks, context

def build_faiss_index(chunks, bucket_name, company_name):
    """Build FAISS index from transcript chunks"""
    texts = [chunk["text"].strip()[:3000] for chunk in chunks if "text" in chunk and chunk["text"].strip()]
    
    if not texts:
        raise ValueError("❌ No valid text chunks found to build FAISS index.")
    
    logger.info(f"📦 Total chunks to embed: {len(texts)}")
    
    # Create embeddings
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            embeddings.extend([e.embedding for e in response.data])
            logger.info(f"✅ Created embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"❌ Embedding failed at batch {i}-{i+batch_size}: {e}")
            # Try with a smaller batch size as fallback
            try:
                for text in batch:
                    response = openai.embeddings.create(
                        input=[text],
                        model="text-embedding-3-small"
                    )
                    embeddings.extend([e.embedding for e in response.data])
                logger.info(f"✅ Created embeddings for batch {i//batch_size + 1} (individual)")
            except Exception as e2:
                logger.error(f"❌ Individual embedding also failed: {e2}")
                raise
    
    # Build and save FAISS index
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    
    # Save locally and upload to GCS
    local_path = f"faiss_index_{company_name}.bin"
    gcs_path = "faiss_indexes/faiss_index.bin"
    
    faiss.write_index(index, local_path)
    upload_to_gcs(local_path, bucket_name, gcs_path)
    
    logger.info(f"✅ Built FAISS index with {len(vectors)} vectors for {company_name}")
    return index

def process_video(video_url, company_name, bucket_name, source=None, meeting_link=None):
    """Main function to process a video for a company"""
    try:
        # Initial memory check and cleanup
        initial_memory = log_memory_usage()
        logger.info(f"📥 Starting video processing for company: {company_name}")
        logger.info(f"📥 Raw request body: {video_url}")
        
        # Get configuration settings
        try:
            from render_deployment_config import get_render_optimized_settings
            config = get_render_optimized_settings()
            memory_fail_threshold = config['memory_fail_threshold']
            memory_cleanup_threshold = config['memory_cleanup_threshold']
        except ImportError:
            # Fallback settings for 2GB RAM
            memory_fail_threshold = 1800
            memory_cleanup_threshold = 1200
        
        # If memory is already high, perform cleanup
        if initial_memory > memory_cleanup_threshold:
            logger.warning(f"⚠️ High initial memory usage: {initial_memory:.1f}MB")
            logger.info("🧹 Performing memory cleanup before processing...")
            
            # Force garbage collection
            import gc
            for i in range(3):
                gc.collect()
            
            # Check memory after cleanup
            post_cleanup_memory = log_memory_usage()
            logger.info(f"🧹 Memory after cleanup: {post_cleanup_memory:.1f}MB (was {initial_memory:.1f}MB)")
            
            # If still too high, fail early
            if post_cleanup_memory > memory_fail_threshold:
                raise Exception(f"Memory usage too high after cleanup: {post_cleanup_memory:.1f}MB. Please try again later or use a smaller video.")
        
        # Check if this is a Loom video first
        try:
            from loom_video_processor import is_loom_url, LoomVideoProcessor
            if is_loom_url(video_url):
                logger.info(f"🎬 Processing Loom video: {video_url}")
                
                # Use configuration-based memory limit
                memory_limit = config.get('max_memory_mb', 1500)  # Default to 1.5GB for 2GB plan
                processor = LoomVideoProcessor(max_memory_mb=memory_limit)
                
                # Check if Loom API key is available
                loom_api_key = os.getenv('LOOM_API_KEY')
                if not loom_api_key:
                    logger.warning("⚠️ LOOM_API_KEY not found - using yt-dlp fallback method")
                
                try:
                    result = processor.process_loom_video(video_url, company_name)
                    
                    if result["success"]:
                        # Process the chunks for embeddings
                        chunks = result["chunks"]
                        texts = [chunk["text"] for chunk in chunks]
                        embeddings = []
                        
                        # Use configuration-based batch size
                        batch_size = config.get('embedding_batch_size', 5)
                        
                        logger.info(f"📦 Processing {len(texts)} chunks in batches of {batch_size}")
                        
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i+batch_size]
                            try:
                                # Memory check before each batch
                                current_memory = log_memory_usage()
                                if current_memory > memory_cleanup_threshold:
                                    logger.warning(f"⚠️ High memory usage before batch {i//batch_size + 1}: {current_memory:.1f}MB")
                                
                                response = openai.embeddings.create(
                                    input=batch,
                                    model="text-embedding-3-small"
                                )
                                embeddings.extend([e.embedding for e in response.data])
                                logger.info(f"✅ Processed embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                                
                                # Light cleanup after each batch
                                import gc
                                gc.collect()
                                
                            except Exception as e:
                                logger.error(f"❌ Embedding failed at batch {i}-{i+batch_size}: {e}")
                                raise
                        
                        # Upsert to Pinecone
                        upsert_chunks_to_pinecone(company_name, chunks, embeddings)
                        
                        # Final cleanup
                        import gc
                        gc.collect()
                        
                        return {
                            "success": True,
                            "data": {
                                "video_filename": f"loom_video_{uuid.uuid4().hex[:8]}",
                                "chunks_count": len(chunks),
                                "message": "Loom video processed successfully"
                            }
                        }
                    else:
                        logger.error(f"❌ Loom video processing failed: {result.get('error', 'Unknown error')}")
                        raise Exception(f"Loom video processing failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Loom video processing failed: {e}")
                    # Don't fall back to regular download for Loom videos
                    raise Exception(f"Loom video processing failed: {str(e)}. Please check if the video URL is accessible and try again.")
        except ImportError:
            logger.warning("⚠️ Loom video processor not available - using standard methods")
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            logger.info("🔄 Falling back to standard video processing methods")
        
        # Check if this is a large video that needs chunked processing
        is_large_video = False
        
        # For YouTube videos, check if it's likely to be large
        if video_url.startswith('http') and ('youtube.com' in video_url or 'youtu.be' in video_url):
            # Use yt-dlp to get video info without downloading
            try:
                ydl_opts = {'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    duration = info.get('duration', 0)
                    # If video is longer than 10 minutes, use chunked processing
                    if duration > 600:  # 10 minutes
                        is_large_video = True
                        logger.info(f"📹 Large video detected: {duration//60} minutes, using chunked processing")
            except:
                pass  # Continue with normal processing if info extraction fails
        
        # For direct URLs, check file size
        elif video_url.startswith('http'):
            try:
                r = requests.head(video_url, timeout=10)
                content_length = r.headers.get('content-length')
                if content_length:
                    file_size_mb = int(content_length) / (1024 * 1024)
                    if file_size_mb > 50:  # If larger than 50MB
                        is_large_video = True
                        logger.info(f"📹 Large video detected: {file_size_mb:.1f}MB, using chunked processing")
            except:
                pass  # Continue with normal processing if size check fails
        
        # Use chunked processing for large videos
        if is_large_video:
            try:
                from large_video_processor import LargeVideoProcessor
                processor = LargeVideoProcessor(max_memory_mb=400, chunk_duration=300)
                result = processor.process_large_video(video_url, company_name)
            except ImportError as e:
                logger.warning(f"Large video processor not available: {e}")
                logger.info("🔄 Trying simple video processor")
                try:
                    from simple_video_processor import SimpleVideoProcessor
                    processor = SimpleVideoProcessor(max_memory_mb=400)
                    result = processor.process_video(video_url, company_name)
                except ImportError:
                    logger.info("🔄 Falling back to normal processing for large video")
                    is_large_video = False
            except Exception as e:
                logger.error(f"Large video processing failed: {e}")
                logger.info("🔄 Trying simple video processor")
                try:
                    from simple_video_processor import SimpleVideoProcessor
                    processor = SimpleVideoProcessor(max_memory_mb=400)
                    result = processor.process_video(video_url, company_name)
                except ImportError:
                    logger.info("🔄 Falling back to normal processing")
                    is_large_video = False
            
            if result["success"]:
                # Process the chunks for embeddings
                chunks = result["chunks"]
                texts = [chunk["text"] for chunk in chunks]
                embeddings = []
                
                # Use Render-optimized batch size
                try:
                    from render_deployment_config import get_render_optimized_settings
                    render_settings = get_render_optimized_settings()
                    batch_size = render_settings['embedding_batch_size']
                except (ImportError, AttributeError):
                    batch_size = 2  # Conservative default for Render
                
                logger.info(f"📦 Processing {len(texts)} chunks in batches of {batch_size}")
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    try:
                        # Memory check before each batch
                        current_memory = log_memory_usage()
                        if current_memory > 450:  # 450MB threshold for 512MB plan
                            logger.warning(f"⚠️ High memory usage before batch {i//batch_size + 1}: {current_memory:.1f}MB")
                        
                        response = openai.embeddings.create(
                            input=batch,
                            model="text-embedding-3-small"
                        )
                        embeddings.extend([e.embedding for e in response.data])
                        logger.info(f"✅ Processed embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        
                        # Force cleanup after each batch
                        import gc
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"❌ Embedding failed at batch {i}-{i+batch_size}: {e}")
                        raise
                
                # Upsert to Pinecone
                upsert_chunks_to_pinecone(company_name, chunks, embeddings)
                
                return {
                    "success": True,
                    "data": {
                        "video_filename": f"large_video_chunked_{uuid.uuid4().hex[:8]}",
                        "chunks_count": len(chunks),
                        "bucket_name": bucket_name,
                        "context": "Large video processed in chunks",
                        "processing_method": "chunked"
                    }
                }
            else:
                return result
        
        # Normal processing for smaller videos
        video_filename = f"downloaded_video_{uuid.uuid4().hex[:8]}.mp4"
        logger.info(f"📥 Downloading video: {video_url}")
        download_video(video_url, video_filename)
        add_video_url_mapping(video_filename, video_url)
        logger.info(f"📥 Transcribing video: {video_filename}")
        chunks, context = transcribe_video(video_filename, company_name, video_url)
        
        # Clean up video file immediately after transcription to save memory
        if os.path.exists(video_filename):
            os.remove(video_filename)
            logger.info(f"🗑️ Cleaned up video file: {video_filename}")
        
        log_memory_usage()
        
        # Create embeddings for all chunks with smaller batch size
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        batch_size = 5  # Reduced batch size to save memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                embeddings.extend([e.embedding for e in response.data])
                logger.info(f"✅ Processed embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"❌ Embedding failed at batch {i}-{i+batch_size}: {e}")
                for text in batch:
                    try:
                        response = openai.embeddings.create(
                            input=[text],
                            model="text-embedding-3-small"
                        )
                        embeddings.extend([e.embedding for e in response.data])
                    except Exception as e2:
                        logger.error(f"❌ Individual embedding also failed: {e2}")
                        raise
        
        # Upsert to Pinecone
        upsert_chunks_to_pinecone(company_name, chunks, embeddings)
        
        return {
            "success": True,
            "data": {
                "video_filename": video_filename,
                "chunks_count": len(chunks),
                "bucket_name": bucket_name,
                "context": context
            }
        }
    except Exception as e:
        logger.error(f"❌ Video processing failed: {e}")
        # Clean up video file on error too
        if 'video_filename' in locals() and os.path.exists(video_filename):
            os.remove(video_filename)
        return {
            "success": False,
            "error": str(e)
        }

# FastAPI app setup
app = FastAPI(title="QuDemo Video Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ProcessVideoRequest(BaseModel, extra='allow'):
    video_url: str
    company_name: str
    bucket_name: Optional[str] = None  # Make bucket_name optional
    source: Optional[str] = None
    meeting_link: Optional[str] = None
    is_youtube: bool = True

class AskQuestionRequest(BaseModel):
    question: str
    company_name: str

class AskQuestionCompanyRequest(BaseModel):
    question: str

class GenerateSummaryRequest(BaseModel):
    questions_and_answers: List[Dict[str, str]]  # Each dict: {"question": ..., "answer": ...}
    buyer_name: Optional[str] = None
    company_name: Optional[str] = None

# API endpoints
@app.post("/process-video/{company_name}")
async def process_video_endpoint(company_name: str, request: Request):
    """Process a video for a specific company"""
    try:
        # Log the raw request for debugging
        body = await request.body()
        logger.info(f"📥 Received request for company: {company_name}")
        logger.info(f"📥 Raw request body: {body}")
        logger.info(f"📥 Content-Type: {request.headers.get('content-type')}")
        
        # Try to parse the JSON body
        try:
            json_body = await request.json()
            logger.info(f"📥 Parsed JSON: {json_body}")
        except Exception as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # Try to validate with Pydantic model
        try:
            validated_request = ProcessVideoRequest(**json_body)
            logger.info(f"📥 Validated request: {validated_request.model_dump()}")
        except ValidationError as e:
            logger.error(f"❌ Validation error: {e}")
            logger.error(f"❌ Validation details: {e.errors()}")
            raise HTTPException(status_code=422, detail=f"Validation error: {e.errors()}")
        
        # Use provided bucket_name or derive from company_name
        bucket_name = validated_request.bucket_name
        if not bucket_name:
            bucket_name = company_name.lower().replace(' ', '_').replace('-', '_')
            logger.info(f"📥 Using derived bucket name: {bucket_name}")
        
        result = process_video(
            video_url=validated_request.video_url,
            company_name=company_name,
            bucket_name=bucket_name,
            source=validated_request.source,
            meeting_link=validated_request.meeting_link
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video for {company_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question_endpoint(request: AskQuestionRequest):
    """Ask a question about a company's video content"""
    try:
        result = answer_question(request.company_name, request.question)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error answering question for {request.company_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/{company_name}")
async def ask_question_company_endpoint(company_name: str, request: AskQuestionCompanyRequest):
    """Ask a question about a specific company's video content"""
    try:
        logger.info(f"📝 Question for {company_name}: {request.question}")
        result = answer_question(company_name, request.question)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error answering question for {company_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary")
async def generate_summary_endpoint(request: GenerateSummaryRequest):
    """Generate a concise summary for a buyer's Q&A using OpenAI"""
    try:
        # Compose the prompt for ChatGPT
        qa_pairs = request.questions_and_answers
        buyer = request.buyer_name or "the buyer"
        company = request.company_name or "the company"
        qa_text = "\n".join([
            f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}" for qa in qa_pairs
        ])
        prompt = (
            f"You are an expert sales assistant. Below are questions asked by {buyer} during a product demo for {company}, and the answers provided by the bot. "
            "Summarize in under 100 words what the buyer is interested in and what they want, so a salesperson can quickly understand their needs. "
            "Be concise, specific, and focus on actionable insights for the sales team.\n\n"
            f"{qa_text}\n\nSummary:"
        )
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/debug/qa-test/{company_name}")
async def debug_qa_test(company_name: str, question: str = "What is this video about?"):
    """Debug endpoint to test Q&A functionality"""
    try:
        logger.info(f"🔍 Debug Q&A test for company: {company_name}")
        logger.info(f"🔍 Question: {question}")
        
        result = answer_question(company_name, question)
        
        logger.info(f"🔍 Q&A result: {result}")
        
        return {
            "company_name": company_name,
            "question": question,
            "result": result,
            "has_video_url": bool(result.get("video_url")),
            "video_url": result.get("video_url"),
            "has_timestamp": bool(result.get("start")),
            "timestamp": result.get("start")
        }
        
    except Exception as e:
        logger.error(f"❌ Debug Q&A test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/videos")
async def debug_videos():
    """Debug endpoint to check video mappings in Supabase"""
    try:
        logger.info("🔍 Debug: Checking video mappings in Supabase")
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not found"}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Query videos table
        response = supabase.table('videos').select('*').execute()
        
        videos = response.data
        logger.info(f"🔍 Debug: Found {len(videos)} videos in Supabase")
        
        # Return debug information
        debug_info = {
            "total_videos": len(videos),
            "videos": []
        }
        
        for video in videos[:10]:  # Show first 10 videos
            debug_info["videos"].append({
                "video_name": video.get("video_name"),
                "video_url": video.get("video_url"),
                "original_url": video.get("original_url"),
                "video_type": video.get("video_type"),
                "created_at": video.get("created_at")
            })
        
        return debug_info
        
    except Exception as e:
        logger.error(f"❌ Debug videos endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/pinecone/{company_name}")
async def debug_pinecone_company(company_name: str):
    """Debug endpoint to check what's stored in Pinecone for a company"""
    try:
        logger.info(f"🔍 Debug: Checking Pinecone for company: {company_name}")
        
        # Create a dummy embedding to query with
        dummy_embedding = [0.0] * 1536  # OpenAI embedding dimension
        
        # Query Pinecone with a dummy embedding to see what's stored
        result = index.query(
            vector=dummy_embedding, 
            top_k=10, 
            include_metadata=True, 
            filter={"company_name": company_name}
        )
        
        matches = result["matches"]
        logger.info(f"🔍 Debug: Found {len(matches)} chunks for company: {company_name}")
        
        # Return debug information
        debug_info = {
            "company_name": company_name,
            "total_chunks": len(matches),
            "chunks": []
        }
        
        for i, match in enumerate(matches[:5]):  # Show first 5 chunks
            meta = match.get("metadata", {})
            debug_info["chunks"].append({
                "index": i + 1,
                "company_name": meta.get("company_name"),
                "source": meta.get("source"),
                "text_preview": meta.get("text", "")[:100] + "..." if len(meta.get("text", "")) > 100 else meta.get("text", ""),
                "original_video_url": meta.get("original_video_url"),
                "score": match.get("score", 0)
            })
        
        return debug_info
        
    except Exception as e:
        logger.error(f"❌ Debug endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status_check():
    """Status check endpoint with basic info about stored data"""
    try:
        # Check Pinecone index
        try:
            # Get index stats
            index_stats = index.describe_index_stats()
            total_vectors = index_stats.get('total_vector_count', 0)
            logger.info(f"📊 Pinecone index stats: {total_vectors} total vectors")
        except Exception as e:
            logger.error(f"❌ Failed to get Pinecone stats: {e}")
            total_vectors = "Error"
        
        # Check video mappings
        try:
            video_count = len(VIDEO_URL_MAPPING)
            logger.info(f"📊 Video mappings: {video_count} videos")
        except Exception as e:
            logger.error(f"❌ Failed to get video mappings: {e}")
            video_count = "Error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pinecone_vectors": total_vectors,
            "video_mappings": video_count,
            "python_version": sys.version,
            "environment": "production" if os.getenv("RENDER") else "development"
        }
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def add_video_url_mapping(local_filename, original_url):
    global VIDEO_URL_MAPPING
    filename = os.path.basename(local_filename)
    VIDEO_URL_MAPPING[filename] = original_url
    logger.info(f"📝 Added video mapping: {filename} -> {original_url}")
    
    # Also upsert to Supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # For Loom videos, store the original share URL as the video_url for playback
            # Loom share URLs can be embedded and played directly
            video_url_for_playback = original_url
            
            # Upsert by video_name - only include columns that exist in schema
            supabase.table('videos').upsert({
                'video_name': filename,
                'video_url': video_url_for_playback
            }, on_conflict=['video_name']).execute()
            logger.info(f"📝 Upserted video mapping to Supabase: {filename} -> {video_url_for_playback}")
    except Exception as e:
        logger.error(f"❌ Failed to upsert video mapping to Supabase: {e}")

def get_original_video_url(local_filename):
    global VIDEO_URL_MAPPING
    if '[' in local_filename:
        filename = local_filename.split('[')[0].strip()
    else:
        filename = local_filename

    original_url = VIDEO_URL_MAPPING.get(filename)
    if not original_url:
        # Try to refresh mapping from Supabase
        logger.info(f"🔄 Refreshing video mappings from Supabase for: {filename}")
        VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
        original_url = VIDEO_URL_MAPPING.get(filename)
    if original_url:
        logger.info(f"🔗 Found video mapping: {filename} -> {original_url}")
    else:
        logger.warning(f"⚠️ No video mapping found for: {filename}")
    return original_url

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 