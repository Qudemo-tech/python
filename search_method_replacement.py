    def search_transcript_for_question(self, company_name: str, qudemo_id: str, 
                                     question: str) -> Optional[Dict[str, Any]]:
        """Search transcript using anchor → expand → rerank system"""
        try:
            # Get transcript data
            transcript_data = self.get_video_transcript(company_name, qudemo_id)
            if not transcript_data:
                return None
            
            # Get chunks from transcript data
            chunks = transcript_data.get('chunks', [])
            if not chunks:
                return None
            
            logger.info(f"🔍 Searching through {len(chunks)} chunks for question: {question}")
            
            # Get intro text for novelty calculation
            intro_text = ""
            for chunk in chunks[:3]:  # First 3 chunks likely contain intro
                if 'building two browser agents' in chunk.get('text', '').lower():
                    intro_text = chunk.get('text', '')
                    break
            
            # Step A: Anchor (high-recall, quick) - Get top 60 chunks by keyword score
            anchors = self._get_anchor_chunks(chunks, question, k=60)
            
            # Step B: Expand (local context sweep) - Merge chunks into sections
            sections = self.qa_utils.merge_chunks_to_sections(anchors, radius=3)
            
            # Step C: Rerank sections by answerability
            scored_sections = []
            for section in sections:
                # Calculate BM25 mean for section
                bm25_mean = self._calculate_bm25_mean(section, question)
                section['bm25_mean'] = bm25_mean
                section['embed_mean'] = 0.0  # Placeholder for embedding similarity
                
                # Calculate comprehensive section score
                section_score = self.qa_utils.calculate_section_score(section, question, intro_text)
                scored_sections.append((section, section_score))
            
            # Sort sections by score
            scored_sections.sort(key=lambda x: x[1], reverse=True)
            
            # Log detailed selection process
            self._log_section_selection(question, scored_sections[:5])
            
            if scored_sections:
                # Get the best section
                best_section, best_score = scored_sections[0]
                
                # Find the earliest deep chunk within the best section
                chosen_chunk = self.qa_utils.find_earliest_deep_chunk(best_section, question)
                
                # Get timestamp from chosen chunk
                timestamp_seconds = chosen_chunk.get('timestamp', 0)
                formatted_timestamp = chosen_chunk.get('formatted_timestamp', '00:00-00:00')
                
                # Combine chunks from the best section for comprehensive answer
                section_chunks = best_section.get('chunks', [])
                combined_answer = " ".join([chunk.get('text', '') for chunk in section_chunks])
                
                # Clean up the answer
                combined_answer = combined_answer.strip()
                if len(combined_answer) > 3000:
                    combined_answer = combined_answer[:3000] + "..."
                
                logger.info(f"✅ Best section found at {formatted_timestamp} with score {best_score:.2f}")
                logger.info(f"📝 Selected {len(section_chunks)} chunks for answer (total length: {len(combined_answer)} chars)")
                
                # Format the answer professionally
                formatted_answer = self._format_professional_answer(combined_answer, question, section_chunks)
                
                return {
                    'answer': formatted_answer,
                    'timestamp': timestamp_seconds,
                    'formatted_timestamp': formatted_timestamp,
                    'confidence': best_score,
                    'sources': section_chunks[:3]  # Top 3 chunks from section
                }
            
            logger.warning(f"⚠️ No relevant sections found for question: {question}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to search transcript: {e}")
            return None
