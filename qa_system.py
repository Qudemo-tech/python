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
        index_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
        logger.info(f"🔍 Querying Pinecone index: {index_name}")

        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.warning(f"⚠️ Index {index_name} not found. Available indexes: {existing_indexes}")
            return []

        index = pc.Index(index_name)

        matches = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        ).matches
        logger.info(f"✅ Found {len(matches)} matches for company: {company_name}")

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
            # Clean the text data before sending to GPT
            cleaned_chunks = []
            for chunk in top_chunks[:3]:
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
                f"You are a product expert bot with full knowledge of {company_name} derived from video transcripts. "
                "IMPORTANT: You can ONLY answer based on the provided video content. "
                "If the video content does not contain information relevant to the question, you MUST say 'The provided content doesn't contain information about [specific topic].' "
                "DO NOT make up information or use external knowledge. "
                "Give direct, concise answers in 1-2 sentences maximum based ONLY on the video content provided. "
                "No verbose explanations, no timestamp references, no bullet points. "
                "Just straight, factual answers from the video content."
            )
            user_prompt = f"Video Content:\n{context}\n\nQuestion: {question}"
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
