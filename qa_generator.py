#!/usr/bin/env python3
"""
Q&A Generator Module
Pre-generates questions and answers from scraped content for efficient retrieval
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, openai_api_key: str):
        """Initialize Q&A generator"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        
    def calculate_questions_per_page(self, content: str) -> int:
        """Calculate how many questions to generate based on content length"""
        word_count = len(content.split())
        
        if word_count < 500:
            return 3
        elif word_count < 1500:
            return 6
        elif word_count < 3000:
            return 8
        else:
            return 12
    
    def calculate_questions_per_video(self, transcript: str, duration_seconds: float) -> int:
        """Calculate questions for video content based on duration"""
        # 1 question per 30 seconds, minimum 3, maximum 15
        questions = max(3, min(15, int(duration_seconds / 30)))
        return questions
    
    def generate_qa_pairs(self, content: str, source_info: str, source_type: str, 
                         content_title: str = None, num_questions: int = None) -> List[Dict]:
        """Generate Q&A pairs from content"""
        try:
            # Calculate number of questions to generate
            if num_questions is not None:
                # Use provided number of questions
                pass
            elif source_type == 'video':
                # For videos, we'll use a default duration or extract from transcript
                num_questions = self.calculate_questions_per_video(content, 300)  # Default 5 min
            else:
                num_questions = self.calculate_questions_per_page(content)
            
            logger.info(f"🔍 Generating {num_questions} Q&A pairs for {source_type} content")
            
            # Prepare content for Q&A generation
            content_preview = content[:2000]  # Limit content for prompt
            
            # Create system prompt for Q&A generation
            system_prompt = f"""You are an expert at creating questions and answers from content. 
Generate {num_questions} high-quality Q&A pairs from the provided content.

Content Source: {source_info}
Content Type: {source_type}
Content Title: {content_title or 'N/A'}

Guidelines:
1. Create diverse question types: "What is...", "How to...", "Why...", "Can I...", "What are..."
2. Questions should be specific and answerable from the content
3. Answers should be concise but complete (50-150 words)
4. Focus on practical, actionable information
5. Include both basic and advanced questions
6. For video content, focus on key points and demonstrations
7. For website content, focus on features, procedures, and explanations

Return ONLY a JSON array with this exact structure:
[
  {{
    "question": "What is the main purpose of this feature?",
    "answer": "The main purpose is to...",
    "question_type": "what_is",
    "difficulty": "basic"
  }}
]

Question types: what_is, how_to, why, can_i, what_are, feature, procedure
Difficulty levels: basic, intermediate, advanced"""

            user_prompt = f"Content to analyze:\n\n{content_preview}\n\nGenerate {num_questions} Q&A pairs:"
            
            # Generate Q&A pairs using GPT-4
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                import json
                qa_pairs = json.loads(response_text)
                
                # Validate and enhance Q&A pairs
                enhanced_pairs = []
                for i, qa in enumerate(qa_pairs):
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        enhanced_qa = {
                            'id': f"qa_{i+1}",
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'question_type': qa.get('question_type', 'general'),
                            'difficulty': qa.get('difficulty', 'basic'),
                            'source_info': source_info,
                            'source_type': source_type,
                            'content_title': content_title,
                            'word_count': len(qa['answer'].split())
                        }
                        enhanced_pairs.append(enhanced_qa)
                
                logger.info(f"✅ Generated {len(enhanced_pairs)} Q&A pairs")
                return enhanced_pairs
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse Q&A JSON: {e}")
                logger.error(f"Response: {response_text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to generate Q&A pairs: {e}")
            return []
    
    def generate_qa_from_chunks(self, chunks: List[Dict], source_info: str, 
                                source_type: str, content_title: str = None) -> List[Dict]:
        """Generate Q&A pairs from multiple chunks based on content length (only for website/document content)"""
        try:
            all_qa_pairs = []
            total_chars = 0
            
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                
                if not chunk_text:
                    continue
                
                # Only generate Q&A for website and document content
                chunk_source_type = chunk_metadata.get('source_type', source_type)
                if chunk_source_type not in ['website', 'document']:
                    logger.debug(f"⏭️ Skipping Q&A generation for {chunk_source_type} chunk")
                    continue
                
                # Calculate number of Q&A pairs based on content length
                char_count = len(chunk_text)
                qa_count = self._calculate_qa_count(char_count)
                total_chars += char_count
                
                logger.info(f"📄 Chunk {i+1}: {char_count:,} characters → {qa_count} Q&A pairs")
                
                # Generate Q&A for this chunk
                chunk_qa_pairs = self.generate_qa_pairs(
                    content=chunk_text,
                    source_info=source_info,
                    source_type=chunk_source_type,
                    content_title=f"{content_title} - Chunk {i+1}" if content_title else f"Chunk {i+1}",
                    num_questions=qa_count
                )
                
                # Add chunk metadata to Q&A pairs
                for qa in chunk_qa_pairs:
                    qa['chunk_index'] = i
                    qa['chunk_metadata'] = chunk_metadata
                    qa['char_count'] = char_count
                    qa['qa_count_generated'] = qa_count
                
                all_qa_pairs.extend(chunk_qa_pairs)
            
            logger.info(f"✅ Generated {len(all_qa_pairs)} Q&A pairs from {len(chunks)} chunks ({total_chars:,} total characters)")
            return all_qa_pairs
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Q&A from chunks: {e}")
            return []
    
    def _calculate_qa_count(self, char_count: int) -> int:
        """Calculate number of Q&A pairs based on content length"""
        # Dynamic Q&A generation based on content size
        if char_count < 500:
            return 1  # Very short content: 1 Q&A pair
        elif char_count < 1000:
            return 2  # Short content: 2 Q&A pairs
        elif char_count < 2000:
            return 3  # Medium content: 3 Q&A pairs
        elif char_count < 4000:
            return 4  # Long content: 4 Q&A pairs
        elif char_count < 8000:
            return 5  # Very long content: 5 Q&A pairs
        else:
            return 6  # Extremely long content: 6 Q&A pairs (max)
    
    def search_qa_pairs(self, question: str, qa_pairs: List[Dict], top_k: int = 3) -> List[Dict]:
        """Search Q&A pairs for relevant answers"""
        try:
            if not qa_pairs:
                return []
            
            # Create question embedding
            question_embedding = self.openai_client.embeddings.create(
                input=[question],
                model="text-embedding-3-small"
            ).data[0].embedding
            
            # Create embeddings for all questions
            questions = [qa['question'] for qa in qa_pairs]
            question_embeddings = self.openai_client.embeddings.create(
                input=questions,
                model="text-embedding-3-small"
            ).data
            
            # Calculate similarities
            similarities = []
            for i, embedding in enumerate(question_embeddings):
                # Simple cosine similarity (you could use numpy for better performance)
                similarity = self.cosine_similarity(question_embedding, embedding.embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and return top matches
            similarities.sort(reverse=True)
            top_matches = []
            
            for similarity, index in similarities[:top_k]:
                qa_pair = qa_pairs[index].copy()
                qa_pair['similarity_score'] = similarity
                top_matches.append(qa_pair)
            
            logger.info(f"🔍 Found {len(top_matches)} relevant Q&A pairs for question")
            return top_matches
            
        except Exception as e:
            logger.error(f"❌ Failed to search Q&A pairs: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0
