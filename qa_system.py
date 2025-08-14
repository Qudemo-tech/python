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
                # Safely get text with fallback
                text = chunk.get("text", "")
                if not text:
                    text = chunk.get("content", "")  # Try alternative field name
                snippet = text[:500].strip().replace("\n", " ")
                
                # Safely get chunk type and context
                chunk_type = chunk.get('type', chunk.get('source_type', 'content'))
                chunk_context = chunk.get('context', chunk.get('source_info', ''))
                rerank_prompt += f"{i+1}. [{chunk_type}] {chunk_context}\n{snippet}\n\n"
            
            rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."
            
            # Log the rerank prompt for debugging
            logger.info(f"🔍 Rerank prompt length: {len(rerank_prompt)} characters")
            
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
            logger.error(f"❌ Reranking error details: {str(e)}")
            # Log chunk structure for debugging
            if top_chunks:
                logger.info(f"🔍 First chunk keys: {list(top_chunks[0].keys())}")
                logger.info(f"🔍 First chunk text preview: {str(top_chunks[0].get('text', 'NO_TEXT'))[:100]}")
        
        # Analyze chunks to separate video and scraped data
        video_chunks = []
        scraped_chunks = []
        faq_chunks = []
        
        logger.info(f"🔍 Analyzing {len(top_chunks)} chunks...")
        for i, chunk in enumerate(top_chunks):
            chunk_type = chunk.get('source_type', 'unknown')
            logger.info(f"🔍 Chunk {i+1}: type={chunk_type}, keys={list(chunk.keys())}")
            
            # Check if it's FAQ data (has both question and answer)
            if chunk.get('answer') and chunk.get('question'):
                faq_chunks.append(chunk)
                logger.info(f"📋 FAQ chunk found: {chunk.get('question', '')[:50]}...")
            
            # Check if it's video data
            elif chunk_type == 'video' or chunk.get('video_url') or chunk.get('start') is not None:
                video_chunks.append(chunk)
                logger.info(f"🎬 Video chunk found: {chunk.get('title', '')[:50]}...")
            
            # Check if it's scraped data
            elif chunk_type == 'website' or chunk.get('source_type') == 'website':
                scraped_chunks.append(chunk)
                logger.info(f"🌐 Scraped chunk found: {chunk.get('text', '')[:50]}...")
                
                # Also check if scraped data is highly relevant to the question
                scraped_text = chunk.get('text', '').strip()
                if scraped_text and len(scraped_text) > 50:  # Only consider substantial text
                    # Check if this scraped text is highly relevant to the user's question
                    import difflib
                    
                    # Check for key terms that indicate this is a direct answer
                    key_terms = ['puzzle', 'accounting', 'platform', 'quickbooks', 'xero', 'startup', 'technology']
                    question_lower = question.lower()
                    scraped_lower = scraped_text.lower()
                    
                    # Count how many key terms are present in both question and scraped text
                    matching_terms = sum(1 for term in key_terms if term in question_lower and term in scraped_lower)
                    
                    # If we have good term matching and substantial text, treat as FAQ
                    if matching_terms >= 2 and len(scraped_text) > 100:
                        logger.info(f"🎯 Found highly relevant scraped data with {matching_terms} matching terms")
                        # Create a FAQ-like structure for this scraped data
                        faq_chunks.append({
                            'question': question,
                            'answer': scraped_text,
                            'source_type': 'website',
                            'source': chunk.get('source', 'scraped_data')
                        })
        
        logger.info(f"📊 Found: {len(video_chunks)} video chunks, {len(scraped_chunks)} scraped chunks, {len(faq_chunks)} FAQ chunks")
        
        # Priority 1: Check for exact FAQ matches first
        use_faq_answer = False
        raw_answer = None
        
        for chunk in faq_chunks:
            faq_question = chunk.get('question', '').strip()
            faq_answer = chunk.get('answer', '').strip()
            
            # Check if the user's question matches the FAQ question (fuzzy match)
            import difflib
            similarity = difflib.SequenceMatcher(None, question.lower(), faq_question.lower()).ratio()
            
            logger.info(f"🔍 Checking FAQ: '{faq_question}' vs '{question}' (similarity: {similarity:.2f})")
            
            if similarity > 0.6:  # Lowered threshold for better FAQ matching
                logger.info(f"✅ Found FAQ match (similarity: {similarity:.2f})")
                logger.info(f"🔍 FAQ Question: {faq_question}")
                logger.info(f"🔍 FAQ Answer: {faq_answer[:100]}...")
                raw_answer = faq_answer
                use_faq_answer = True
                break
        
        # Only generate new answer if we don't have a matching FAQ answer
        if not use_faq_answer:
            # Determine the best approach based on available data
            has_video = len(video_chunks) > 0
            has_scraped = len(scraped_chunks) > 0
            
            logger.info(f"🎯 Strategy: has_video={has_video}, has_scraped={has_scraped}")
            
            if has_video and has_scraped:
                # Both video and scraped data available - combine them
                logger.info("🔄 Combining video and scraped data...")
                combined_chunks = video_chunks + scraped_chunks
                content_type = "Combined Video & Website Content"
                should_return_video_url = True
            elif has_video and not has_scraped:
                # Only video data available
                logger.info("🎬 Using video data only...")
                combined_chunks = video_chunks
                content_type = "Video Content"
                should_return_video_url = True
            elif has_scraped and not has_video:
                # Only scraped data available
                logger.info("🌐 Using scraped data only...")
                combined_chunks = scraped_chunks
                content_type = "Website Content"
                should_return_video_url = False
            else:
                # Fallback to all chunks
                logger.info("⚠️ Using all available chunks...")
                combined_chunks = top_chunks
                content_type = "Mixed Content"
                should_return_video_url = False
            
            # Generate answer using GPT-4
            try:
                # Clean the text data before sending to GPT
                cleaned_chunks = []
                for chunk in combined_chunks[:6]:
                    # Safely get text with multiple fallbacks
                    text = chunk.get('text', '')
                    if not text:
                        text = chunk.get('content', '')  # Try alternative field name
                    if not text:
                        text = str(chunk)  # Last resort
                    
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
                    if clean_text:  # Only add non-empty chunks
                        cleaned_chunks.append(clean_text[:500])
                
                                # Ensure we have at least some content
                if not cleaned_chunks:
                    logger.error("❌ No valid text content found in chunks")
                    return {"error": "No valid content found to generate answer."}
                
                # Content type is already determined above
                
                context = "\n\n".join([
                    f"{content_type}: {text}" for text in cleaned_chunks
                ])
                
                system_prompt = (
                    f"You are a product demo assistant for {company_name}. "
                    f"Answer ONLY using the provided {content_type.lower()}. If the content does not contain relevant info, say: "
                    "'The provided content doesn't contain information about [topic].' Do NOT use external knowledge. "
                    "Return ONLY the following markdown sections in this exact order and nothing else: \n"
                    "- TL;DR: <one concise line>\n"
                    "- Steps:\n  1. <step>\n  2. <step>\n  3. <step>\n"
                    "- Notes:\n  - <note>\n  - <note>\n"
                    "Keep it crisp (roughly 120-180 words total)."
                )
                user_prompt = (
                    f"{content_type}:\n" + context + "\n\n" +
                    "Question: " + question + "\n\n" +
                    "Respond using the exact structure above."
                )
                # Log the prompts for debugging
                logger.info(f"🔍 System prompt length: {len(system_prompt)} characters")
                logger.info(f"🔍 User prompt length: {len(user_prompt)} characters")
                logger.info(f"🔍 Context length: {len(context)} characters")
                
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
                logger.error(f"❌ GPT-4 error details: {str(e)}")
                # Try with a simpler prompt as fallback
                try:
                    logger.info("🔄 Trying fallback with simpler prompt...")
                    fallback_prompt = f"Based on this content: {context[:1000]}...\n\nQuestion: {question}\n\nAnswer:"
                    fallback_completion = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": fallback_prompt}],
                        timeout=20,
                        max_tokens=200,
                        temperature=0.3
                    )
                    raw_answer = fallback_completion.choices[0].message.content
                    logger.info("✅ Generated fallback answer with GPT-3.5.")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
                    return {"error": "Failed to generate answer after multiple attempts."}

        
        def strip_sources(text):
            return re.sub(r'\[source\]\([^)]+\)', '', text).strip()
        
        def format_answer(text):
            text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
            text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
            return re.sub(r'\n+', '\n', text).strip()
        
        # If we used a FAQ answer, don't apply video-specific formatting
        if use_faq_answer:
            clean_answer = raw_answer  # Use FAQ answer as-is
        else:
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
        video_url = None
        if should_return_video_url and video_chunks:
            # Get video URL from the first video chunk
            video_chunk = video_chunks[0]
            video_url = video_chunk.get("video_url") or video_chunk.get("original_video_url")
            if not video_url:
                # Try to resolve from the chunk's source filename
                source_filename = video_chunk.get("source", "").split(" [")[0].strip()
                video_url = get_original_video_url(source_filename)
                if not video_url:
                    logger.warning(f"⚠️ No video mapping found for: {source_filename}")
                    video_url = video_chunk.get("source")
        logger.info(f"📤 Returning final answer. Video URL: {video_url}")
        
        # Enhanced timestamp handling for better accuracy (YouTube-style precision)
        start = 0.0
        end = 0.0
        
        if video_chunks:
            # Get timestamps from the first video chunk
            video_chunk = video_chunks[0]
            start = float(video_chunk.get('start', 0.0)) if isinstance(video_chunk, dict) else 0.0
            end = float(video_chunk.get('end', 0.0)) if isinstance(video_chunk, dict) else 0.0
            
            # Debug timestamp information
            logger.info(f"🔍 Video chunk timestamps - start: {start}, end: {end}")
            logger.info(f"🔍 Video chunk source type: {video_chunk.get('source_type', 'unknown')}")
            logger.info(f"🔍 Video chunk source: {video_chunk.get('source', 'unknown')}")
            logger.info(f"🔍 Video chunk metadata keys: {list(video_chunk.keys()) if isinstance(video_chunk, dict) else 'NOT_DICT'}")
            
            # Check if we have valid timestamps
            if start == 0.0 and end == 0.0:
                logger.warning(f"⚠️ Video chunk has no timestamps, looking for alternatives...")
                # Fallback: aggregate across video chunks if first chunk lacks timestamps
                ts_candidates = []
                
                for i, chunk in enumerate(video_chunks[:3]):  # Check top 3 video chunks
                    chunk_start = float(chunk.get('start', 0.0))
                    chunk_end = float(chunk.get('end', 0.0))
                    chunk_type = chunk.get('source_type', 'unknown')
                    chunk_source = chunk.get('source', 'unknown')
                    logger.info(f"🔍 Video chunk {i+1}: start={chunk_start}, end={chunk_end}, type={chunk_type}, source={chunk_source}")
                    
                    if chunk_start > 0.0 or chunk_end > 0.0:
                        ts_candidates.append((chunk_start, chunk_end))
                
                if ts_candidates:
                    start = min(s for s, _ in ts_candidates)
                    end = max(e for _, e in ts_candidates)
                    logger.info(f"✅ Found timestamps from video alternatives: start={start}, end={end}")
                else:
                    logger.warning(f"⚠️ No valid timestamps found in any video chunks")
                    # Try to extract from source filename or other metadata
                    source_info = video_chunk.get('source', '')
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
            else:
                logger.info(f"✅ Using timestamps from video chunk: start={start}, end={end}")
        else:
            logger.info("🎬 No video chunks found, skipping timestamp logic")
        
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
