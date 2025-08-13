"""
Q&A System Module
Handles question answering, Pinecone queries, and response generation
"""

import os
import re
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel

# Import OpenAI and Pinecone
import openai
from pinecone import Pinecone

# Import video processing functions
from video_processing import get_original_video_url

# Configure logging
logger = logging.getLogger(__name__)

# Global clients
pc = None

# Default Pinecone index name
default_index_name = os.getenv("PINECONE_INDEX", "qudemo-demo")

class AskQuestionRequest(BaseModel):
    question: str
    company_name: str

class AskQuestionCompanyRequest(BaseModel):
    question: str

class GenerateSummaryRequest(BaseModel):
    questions_and_answers: List[Dict[str, str]]
    buyer_name: Optional[str] = None
    company_name: Optional[str] = None

def initialize_qa_system():
    """Initialize Q&A system with OpenAI and Pinecone clients"""
    global pc
    
    try:
        # Initialize Pinecone
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            logger.error("❌ PINECONE_API_KEY not found")
            return False
        
        pc = Pinecone(api_key=pinecone_key)
        logger.info("✅ Pinecone client initialized")
        
        # Initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.error("❌ OPENAI_API_KEY not found")
            return False
        
        openai.api_key = openai_key
        logger.info("✅ OpenAI client initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Q&A system: {e}")
        return False

def query_pinecone(company_name: str, embedding: List[float], top_k: int = 6):
    """Query Pinecone for relevant chunks"""
    try:
        # Use a single shared index with per-company namespace
        index_name = os.getenv("PINECONE_INDEX", "qudemo-demo")
        logger.info(f"🔍 Querying Pinecone index: {index_name}")

        existing_indexes = [index.name for index in pc.list_indexes()]
        logger.info(f"🔍 Available Pinecone indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            logger.warning(f"⚠️ Index {index_name} not found. Available indexes: {existing_indexes}")
            
            # Try to find the best fallback index - prioritize existing indexes
            if existing_indexes:
                # Use the first available index
                fallback_index = existing_indexes[0]
                logger.warning(f"⚠️ Falling back to existing index: {fallback_index}")
                index_name = fallback_index
            else:
                logger.error(f"❌ No available Pinecone indexes found")
                return []

        index = pc.Index(index_name)
        namespace = company_name.lower().replace(' ', '-')
        
        logger.info(f"🔍 Querying namespace: {namespace} in index: {index_name}")

        # First, let's check what's in this namespace
        try:
            stats = index.describe_index_stats()
            logger.info(f"🔍 Index stats: {stats}")
            
            # Check if namespace exists and has data
            if 'namespaces' in stats and namespace in stats['namespaces']:
                namespace_stats = stats['namespaces'][namespace]
                logger.info(f"🔍 Namespace '{namespace}' stats: {namespace_stats}")
            else:
                logger.warning(f"⚠️ Namespace '{namespace}' not found in index stats")
                logger.info(f"🔍 Available namespaces: {list(stats.get('namespaces', {}).keys())}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get index stats: {e}")

        # First try with namespace
        try:
            matches = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            ).matches
            logger.info(f"✅ Found {len(matches)} matches with namespace '{namespace}'")
        except Exception as e:
            logger.warning(f"⚠️ Query with namespace '{namespace}' failed: {e}")
            # Try without namespace to see what's available
            try:
                matches = index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True
                ).matches
                logger.warning(f"⚠️ Found {len(matches)} matches without namespace (all data)")
                
                # Filter by company in metadata
                company_matches = [m for m in matches if m.metadata.get('company', '').lower() == company_name.lower()]
                logger.info(f"🔍 Filtered to {len(company_matches)} matches for company '{company_name}'")
                matches = company_matches
            except Exception as e2:
                logger.error(f"❌ Query without namespace also failed: {e2}")
                matches = []
        logger.info(f"✅ Found {len(matches)} matches for company: {company_name}")
        
        # Debug: Log the first match metadata to see what we're getting
        if matches:
            first_match = matches[0].metadata
            logger.info(f"🔍 First match metadata keys: {list(first_match.keys())}")
            logger.info(f"🔍 First match has timestamps: start={first_match.get('start', 'NOT_FOUND')}, end={first_match.get('end', 'NOT_FOUND')}")
            logger.info(f"🔍 First match source: {first_match.get('source', 'NOT_FOUND')}")

        return matches
        
    except Exception as e:
        logger.error(f"❌ Pinecone query failed: {e}")
        raise e

def answer_question(company_name: str, question: str) -> Dict:
    """Answer a question using the Q&A system"""
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
            
            # Check if we have any chunks
            if not top_chunks:
                logger.warning(f"⚠️ No chunks found for company: {company_name}")
                return {"answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ensure videos have been processed.", "sources": []}
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
            response_text = rerank_response.choices[0].message.content
            numbers = re.findall(r"\d+", response_text)
            if numbers:
                best_index = int(numbers[0]) - 1
                if 0 <= best_index < len(top_chunks):
                    best_chunk = top_chunks[best_index]
                    logger.info(f"🏅 GPT-3.5-turbo reranked chunk #{best_index+1} as the most relevant.")
                else:
                    best_chunk = top_chunks[0]
                    logger.warning(f"⚠️ Invalid chunk index {best_index+1}, falling back to top chunk")
            else:
                best_chunk = top_chunks[0]
                logger.warning(f"⚠️ No number found in rerank response: {response_text}")
        except Exception as e:
            best_chunk = top_chunks[0]
            logger.warning(f"⚠️ Reranking failed, falling back to top Pinecone chunk: {e}")
        
        # Generate answer using GPT-4
        try:
            # Clean the text data before sending to GPT (use more chunks for richer context)
            cleaned_chunks = []
            for chunk in top_chunks[:6]:
                text = chunk['text']
                # Remove JSON formatting and extract clean transcription
                if '```json' in text:
                    # Try to extract the transcription from JSON
                    import json
                    try:
                        # Find the JSON part
                        json_start = text.find('{')
                        json_end = text.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            json_text = text[json_start:json_end]
                            json_data = json.loads(json_text)
                            clean_text = json_data.get('transcription', text)
                        else:
                            clean_text = text
                    except:
                        clean_text = text
                else:
                    clean_text = text
                
                # Clean up the text
                clean_text = clean_text.replace('\n', ' ').replace('  ', ' ').strip()
                cleaned_chunks.append(clean_text[:500])
            
            context = "\n\n".join([
                f"Video Content: {text}" for text in cleaned_chunks
            ])
            
            system_prompt = (
                f"You are a product demo assistant for {company_name}. "
                "Answer ONLY using the provided video content. If the content does not contain relevant info, say: "
                "'The provided content doesn't contain information about [topic].' Do NOT use external knowledge. "
                "Return ONLY the following markdown sections in this exact order and nothing else: \n"
                "- TL;DR: <one concise line>\n"
                "- Steps:\n  1. <step>\n  2. <step>\n  3. <step>\n"
                "- Notes:\n  - <note>\n  - <note>\n"
                "Keep it crisp (roughly 120-180 words total)."
            )
            user_prompt = (
                "Video Content:\n" + context + "\n\n" +
                "Question: " + question + "\n\n" +
                "Respond using the exact structure above."
            )
            completion = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=30,
                max_tokens=350,
                temperature=0.2
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
        sources = []
        for chunk in top_chunks:
            title = chunk.get("title")
            video_url = chunk.get("video_url")
            if title and title != "Unknown":
                sources.append(title)
            elif video_url:
                # Extract video ID from URL for a cleaner source name
                if "youtu.be" in video_url or "youtube.com" in video_url:
                    video_id = video_url.split("?")[0].split("/")[-1]
                    sources.append(f"YouTube Video ({video_id})")
                else:
                    sources.append(f"Video ({video_url.split('/')[-1]})")
            else:
                sources.append("Unknown")

        # If the answer indicates no information was found, do not return video info
        no_info_phrases = [
            "I do not have that information",
            "I don't have that information",
            "I do not know",
            "I don't know",
            "no information available",
            "I couldn't find any information",
            "Sorry, I couldn't find",
            "Sorry, I do not have",
            "doesn't contain information",
            "does not contain information",
            "not contain information"
        ]
        if any(phrase.lower() in clean_answer.lower() for phrase in no_info_phrases):
            return {
                "answer": clean_answer,
                "sources": sources
            }

        # Get original video URL using metadata
        video_url = best_chunk.get("video_url") or best_chunk.get("original_video_url")
        if not video_url:
            # Try to resolve from the chunk's source filename
            source_filename = best_chunk.get("source", "").split(" [")[0].strip()
            video_url = get_original_video_url(source_filename)
            if not video_url:
                logger.warning(f"⚠️ No video mapping found for: {source_filename}")
                video_url = best_chunk.get("source")
        logger.info(f"📤 Returning final answer. Video URL: {video_url}")
        
        # Enhanced timestamp handling for better accuracy (YouTube-style precision)
        start = float(best_chunk.get('start', 0.0)) if isinstance(best_chunk, dict) else 0.0
        end = float(best_chunk.get('end', 0.0)) if isinstance(best_chunk, dict) else 0.0
        
        # Debug timestamp information
        logger.info(f"🔍 Best chunk timestamps - start: {start}, end: {end}")
        logger.info(f"🔍 Best chunk source type: {best_chunk.get('source_type', 'unknown')}")
        logger.info(f"🔍 Best chunk source: {best_chunk.get('source', 'unknown')}")
        logger.info(f"🔍 Best chunk metadata keys: {list(best_chunk.keys()) if isinstance(best_chunk, dict) else 'NOT_DICT'}")
        
        # Check if we have valid timestamps
        if start == 0.0 and end == 0.0:
            logger.warning(f"⚠️ Best chunk has no timestamps, looking for alternatives...")
            # Fallback: aggregate across top few chunks if best chunk lacks timestamps
            relevant_chunks = [c for c in top_chunks if isinstance(c, dict)]
            ts_candidates = []
            
            for i, chunk in enumerate(relevant_chunks[:3]):  # Check top 3 chunks
                chunk_start = float(chunk.get('start', 0.0))
                chunk_end = float(chunk.get('end', 0.0))
                chunk_type = chunk.get('source_type', 'unknown')
                chunk_source = chunk.get('source', 'unknown')
                logger.info(f"🔍 Chunk {i+1}: start={chunk_start}, end={chunk_end}, type={chunk_type}, source={chunk_source}")
                
                if chunk_start > 0.0 or chunk_end > 0.0:
                    ts_candidates.append((chunk_start, chunk_end))
            
            if ts_candidates:
                start = min(s for s, _ in ts_candidates)
                end = max(e for _, e in ts_candidates)
                logger.info(f"✅ Found timestamps from alternatives: start={start}, end={end}")
            else:
                logger.warning(f"⚠️ No valid timestamps found in any chunks")
                # Try to extract from source filename or other metadata
                source_info = best_chunk.get('source', '')
                if source_info and '[' in source_info and ']' in source_info:
                    try:
                        # Try to extract timestamp from source string like "video.mp4 [120-180]"
                        timestamp_match = re.search(r'\[(\d+)-(\d+)\]', source_info)
                        if timestamp_match:
                            start = float(timestamp_match.group(1))
                            end = float(timestamp_match.group(2))
                            logger.info(f"✅ Extracted timestamps from source string: start={start}, end={end}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract timestamps from source: {e}")
        
        # YouTube-style timestamp precision for all video types
        if end > start:
            # Use minimal buffer for precise timestamps (like YouTube)
            buffer = 0.3  # Very small buffer for precision
            start = max(0.0, start - buffer)
            end = end + buffer
            logger.info(f"🎯 Final timestamps with minimal buffer: start={start}, end={end}")
        else:
            logger.warning(f"⚠️ Invalid timestamp range: start={start}, end={end}")
        
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

def generate_summary(questions_and_answers: List[Dict[str, str]], 
                    buyer_name: Optional[str] = None, 
                    company_name: Optional[str] = None) -> Dict:
    """Generate a summary from questions and answers"""
    try:
        logger.info(f"📝 Generating summary for {company_name}")
        
        # Format Q&A for summary
        qa_text = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in questions_and_answers
        ])
        
        # Generate summary using OpenAI
        prompt = f"""Based on the following questions and answers from a video demo session, create a concise summary.
        
        Questions and Answers:
        {qa_text}
        
        Company: {company_name or 'Unknown'}
        Buyer: {buyer_name or 'Unknown'}
        
        Summary:"""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries of demo sessions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        
        return {
            'success': True,
            'summary': summary,
            'company_name': company_name,
            'buyer_name': buyer_name,
            'qa_count': len(questions_and_answers)
        }
        
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        return {
            'success': False,
            'error': f"Summary generation failed: {str(e)}"
        }

def answer_question_with_enhanced_fallback(company_name: str, question: str, use_enhanced: bool = True) -> Dict:
    """Answer question with optional enhanced Q&A fallback"""
    try:
        # Try enhanced Q&A first if requested and available
        if use_enhanced:
            try:
                from enhanced_qa import EnhancedQA
                import os
                
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    enhanced_qa = EnhancedQA(openai_api_key)
                    result = enhanced_qa.answer_question_enhanced(company_name, question)
                    
                    # If enhanced Q&A found something, return it
                    if result.get("source_type") != "none" and result.get("answer"):
                        logger.info(f"✅ Enhanced Q&A provided answer for {company_name}")
                        return {
                            "answer": result["answer"],
                            "sources": result.get("sources", []),
                            "video_url": result.get("video_timestamp", {}).get("video_url"),
                            "start": result.get("video_timestamp", {}).get("start"),
                            "end": result.get("video_timestamp", {}).get("end"),
                            "source_type": result.get("source_type"),
                            "confidence": result.get("confidence")
                        }
                    else:
                        logger.info(f"⚠️ Enhanced Q&A found no relevant content, falling back to video-only")
                        
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Q&A failed, falling back to video-only: {e}")
        
        # Fall back to original video-only Q&A
        logger.info(f"🎬 Using video-only Q&A for {company_name}")
        return answer_question(company_name, question)
        
    except Exception as e:
        logger.error(f"❌ Enhanced fallback Q&A failed: {e}")
        return {"error": f"Failed to answer question: {str(e)}"}
