"""
Advanced Q&A Retrieval Utilities
Implements anchor → expand → rerank system to fix intro keyword spam
"""

import re
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict

class QARetrievalUtils:
    def __init__(self):
        # Procedural verbs (how-to vibes)
        self.procedural_verbs = [
            'create', 'click', 'open', 'navigate', 'select', 'configure', 'install', 
            'set', 'connect', 'map', 'upload', 'save', 'enable', 'disable', 'run', 
            'deploy', 'train', 'index', 'test', 'verify', 'build', 'make', 'add',
            'remove', 'update', 'edit', 'modify', 'generate', 'process', 'execute',
            'implement', 'setup', 'initialize', 'start', 'stop', 'restart', 'launch'
        ]
        
        # Step/list cues
        self.step_cues = [
            '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
            'step', 'next', 'then', 'finally', 'first', 'second', 'third',
            'let me show you', 'we\'ll do', 'now we', 'after that', 'following',
            'subsequently', 'meanwhile', 'additionally', 'furthermore'
        ]
        
        # Domain entities (tool nouns)
        self.domain_entities = [
            'pinecone', 'fastapi', 'gcs', 'gemini', 'openai', 'supabase',
            'hubspot', 'grain', 'tango', 'crm', 'api', 'webhook', 'dashboard',
            'schema', 'index', 'vector', 'embedding', 'transcript', 'chunk'
        ]
        
        # Red flag terms (intro fluff)
        self.fluff_terms = [
            'welcome', 'agenda', 'in this video', 'subscribe', 'like the video',
            'introduction', 'context setting', 'overview', 'prerequisites',
            'before we start', 'let\'s begin', 'today we\'re going to'
        ]
        
        # General intent phrases
        self.general_intent = [
            'overview', 'intro', 'what is', 'getting started', 'prereq', 'prerequisite',
            'high level', 'conceptual', 'basics', 'fundamentals', 'introduction'
        ]
        
        # Specific intent indicators
        self.specific_intent = [
            'disqualified', 'qualified', 'a/b test', 'retry policy', 'webhook',
            'configure', 'connect', 'export', 'deploy', 'fine-tune', 'dashboard',
            'schema', 'api key', 'pinecone index', 'gcs bucket', 'gemini api'
        ]

    def classify_question_intent(self, question: str) -> Dict[str, bool]:
        """Classify question as general or specific intent"""
        question_lower = question.lower()
        
        is_general = any(term in question_lower for term in self.general_intent)
        is_specific = any(term in question_lower for term in self.specific_intent)
        
        # Mutual exclusion: if both true, specific wins
        if is_general and is_specific:
            is_general = False
            
        return {
            'is_general': is_general,
            'is_specific': is_specific
        }

    def extract_procedural_density(self, text: str) -> float:
        """Calculate procedural density (how-to vibes)"""
        text_lower = text.lower()
        hits = sum(1 for verb in self.procedural_verbs if verb in text_lower)
        
        # Calculate hits per minute (assuming ~150 words per minute)
        word_count = len(text.split())
        minutes = max(word_count / 150, 0.1)  # Avoid division by zero
        
        hits_per_minute = hits / minutes
        return math.log(1 + hits_per_minute) * 2.5

    def extract_steps_score(self, text: str) -> float:
        """Calculate steps/list cues score"""
        text_lower = text.lower()
        hits = sum(1 for cue in self.step_cues if cue in text_lower)
        
        if hits >= 2:
            return 2.0
        elif hits == 1:
            return 0.5
        else:
            return 0.0

    def extract_cooccurrence(self, text: str, question: str) -> float:
        """Check for entity co-occurrence with question entities"""
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Extract entities from question
        question_entities = [entity for entity in self.domain_entities if entity in question_lower]
        
        if not question_entities:
            return 0.0
            
        # Check if text contains both question entities and domain entities
        text_entities = [entity for entity in self.domain_entities if entity in text_lower]
        
        if question_entities and text_entities:
            return 2.0
        else:
            return 0.0

    def extract_novelty_score(self, text: str, intro_text: str) -> float:
        """Calculate novelty vs intro (simple word overlap)"""
        if not intro_text:
            return 0.0
            
        text_words = set(text.lower().split())
        intro_words = set(intro_text.lower().split())
        
        if not intro_words:
            return 0.0
            
        overlap = len(text_words.intersection(intro_words))
        similarity = overlap / len(intro_words)
        
        if similarity < 0.6:
            return 2.0
        else:
            return -1.0

    def extract_fluff_penalty(self, text: str) -> float:
        """Calculate penalty for intro fluff"""
        text_lower = text.lower()
        hits = sum(1 for fluff in self.fluff_terms if fluff in text_lower)
        
        if hits >= 1:
            return -2.0
        else:
            return 0.0

    def calculate_position_prior(self, question: str, chunk_start_seconds: int) -> float:
        """Calculate position prior based on question intent"""
        intent = self.classify_question_intent(question)
        
        if intent['is_specific']:
            # Penalty for early chunks in specific questions
            if chunk_start_seconds < 120:  # First 2 minutes
                return -min(6.0, 6.0)  # Maximum penalty
            else:
                return 0.0
        elif intent['is_general']:
            # Bonus for early chunks in general questions
            if chunk_start_seconds < 120:
                return 2.0
            else:
                return 0.0
        else:
            return 0.0

    def calculate_section_score(self, section: Dict, question: str, intro_text: str = "") -> float:
        """Calculate comprehensive section score"""
        # Extract features
        procedural_density = self.extract_procedural_density(section['text'])
        steps_score = self.extract_steps_score(section['text'])
        cooccurrence = self.extract_cooccurrence(section['text'], question)
        novelty = self.extract_novelty_score(section['text'], intro_text)
        fluff_penalty = self.extract_fluff_penalty(section['text'])
        
        # Duration coverage (section length in seconds)
        span_seconds = section.get('span_seconds', 90)
        span_score = min(span_seconds / 90, 1.5)
        
        # Position prior
        position_prior = self.calculate_position_prior(question, section['start_seconds'])
        
        # Combine scores (α=0.5 for BM25, β=1.0 for embedding similarity)
        bm25_mean = section.get('bm25_mean', 0.0)
        embed_mean = section.get('embed_mean', 0.0)
        
        final_score = (
            0.5 * bm25_mean +
            1.0 * embed_mean +
            procedural_density +
            steps_score +
            cooccurrence +
            novelty +
            span_score +
            fluff_penalty +
            position_prior
        )
        
        return final_score

    def merge_chunks_to_sections(self, chunks: List[Dict], radius: int = 3) -> List[Dict]:
        """Merge chunks into sections with context windows"""
        if not chunks:
            return []
            
        sections = []
        used_chunks = set()
        
        for i, chunk in enumerate(chunks):
            if i in used_chunks:
                continue
                
            # Create section starting from this chunk
            section_chunks = [chunk]
            used_chunks.add(i)
            
            # Expand context window (±radius chunks)
            start_idx = max(0, i - radius)
            end_idx = min(len(chunks), i + radius + 1)
            
            for j in range(start_idx, end_idx):
                if j != i and j not in used_chunks:
                    # Check if chunks are contiguous (within 60 seconds)
                    current_end = chunk.get('timestamp', 0) + 30  # Assuming 30s chunks
                    next_start = chunks[j].get('timestamp', 0)
                    
                    if abs(next_start - current_end) <= 60:
                        section_chunks.append(chunks[j])
                        used_chunks.add(j)
            
            # Sort chunks by timestamp
            section_chunks.sort(key=lambda x: x.get('timestamp', 0))
            
            # Create section
            if section_chunks:
                section = {
                    'chunks': section_chunks,
                    'text': ' '.join([c.get('text', '') for c in section_chunks]),
                    'start_seconds': section_chunks[0].get('timestamp', 0),
                    'end_seconds': section_chunks[-1].get('timestamp', 0) + 30,
                    'span_seconds': (section_chunks[-1].get('timestamp', 0) + 30) - section_chunks[0].get('timestamp', 0),
                    'chunk_count': len(section_chunks)
                }
                sections.append(section)
        
        return sections

    def find_earliest_deep_chunk(self, section: Dict, question: str) -> Dict:
        """Find the earliest chunk in section that meets depth criteria"""
        chunks = section.get('chunks', [])
        if not chunks:
            return chunks[0] if chunks else {}
            
        # Calculate depth threshold for each chunk
        depth_threshold = 2.5
        
        for chunk in chunks:
            procedural_density = self.extract_procedural_density(chunk.get('text', ''))
            steps_score = self.extract_steps_score(chunk.get('text', ''))
            cooccurrence = self.extract_cooccurrence(chunk.get('text', ''), question)
            
            depth_score = procedural_density + steps_score + cooccurrence
            
            if depth_score >= depth_threshold:
                return chunk
                
        # If no chunk meets depth threshold, return earliest with max combined score
        best_chunk = max(chunks, key=lambda c: (
            self.extract_procedural_density(c.get('text', '')) +
            self.extract_steps_score(c.get('text', '')) +
            self.extract_cooccurrence(c.get('text', ''), question)
        ))
        
        return best_chunk

    def cap_early_hits(self, chunks: List[Dict], first_seconds: int = 120, max_hits: int = 2) -> List[Dict]:
        """Cap the number of chunks from early video sections"""
        early_chunks = []
        late_chunks = []
        
        for chunk in chunks:
            if chunk.get('timestamp', 0) < first_seconds:
                early_chunks.append(chunk)
            else:
                late_chunks.append(chunk)
        
        # Keep only top max_hits from early chunks
        early_chunks = early_chunks[:max_hits]
        
        # Combine and return
        return early_chunks + late_chunks
