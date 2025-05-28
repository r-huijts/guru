#!/usr/bin/env python3
"""
Fine-Tuning Dataset Generator for Speech Transcripts
===================================================

This script processes raw speech transcripts and generates datasets suitable for 
fine-tuning LLMs in various formats (Alpaca, ChatML, Q&A, etc.)

Author: AI Assistant
Usage: python generate_finetuning_dataset.py
"""

import json
import re
import os
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import random

class TranscriptProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transcripts = []
        self.speaker_name = "Teacher"  # Default name, can be customized
        
    def load_transcripts(self) -> List[str]:
        """Load all transcript files from the data directory."""
        transcript_files = list(self.data_dir.glob("*.txt"))
        transcripts = []
        
        for file_path in transcript_files:
            print(f"📖 Loading {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    transcripts.append(content)
                    
        print(f"✅ Loaded {len(transcripts)} transcripts")
        return transcripts
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning that handles both paragraph and single-line formats."""
        # Remove common transcription artifacts
        text = re.sub(r'\\[.*?\\]', '', text)  # Remove [inaudible], [applause], etc.
        text = re.sub(r'\\(.*?\\)', '', text)  # Remove (background noise), etc.
        text = re.sub(r'>>.*?<<', '', text)  # Remove >>speaker<< tags
        
        # Clean up repetitive thank yous and similar patterns
        text = re.sub(r'(\\s*Thank you\\.?\\s*){3,}.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(\\s*you\\s+){5,}', ' you ', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and normalize
        text = re.sub(r'\\s+', ' ', text)
        text = text.strip()
        
        # Detect if this needs smart segmentation
        # Check for lack of natural paragraph breaks in long text
        line_count = len(text.split('\\n'))
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        # If it's a long text with very few line breaks, apply smart segmentation
        if (line_count <= 5 and len(text) > 1000 and sentence_count > 15):
            print(f"   📝 Applying enhanced smart segmentation for better text structure...")
            text = self.segment_single_line_text(text)
        
        return text
    
    def segment_single_line_text(self, text: str) -> str:
        """Convert single-line text into properly segmented paragraphs with better granularity."""
        # Split on strong sentence endings followed by capital letters
        sentences = re.split(r'(?<=[.!?])\\s+(?=[A-Z])', text)
        
        # Group sentences into logical paragraphs (smaller groups for better training)
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            
            # Create smaller segments (2-3 sentences) for better training examples
            if (len(current_paragraph) >= 2 and 
                (self.is_topic_shift(sentence) or 
                 len(current_paragraph) >= 3 or
                 len(' '.join(current_paragraph).split()) >= 60)):  # ~60 words max
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def is_topic_shift(self, sentence: str) -> bool:
        """Detect potential topic shifts in the text."""
        topic_shift_indicators = [
            'now', 'so', 'but', 'however', 'therefore', 'because',
            'let me', 'i want to', 'the important thing', 'you must understand',
            'this is', 'what happens', 'the problem is', 'you see'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower[:50] for indicator in topic_shift_indicators)

    def segment_text(self, text: str) -> List[str]:
        """Enhanced segmentation that works better with both formats."""
        segments = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph.split()) < 15:
                continue
            
            # If paragraph is very long, split it further
            if len(paragraph.split()) > 150:
                # Split on natural breaks within long paragraphs
                sub_segments = self.split_long_paragraph(paragraph)
                segments.extend(sub_segments)
            else:
                segments.append(paragraph)
        
        # Filter segments by quality
        quality_segments = []
        for segment in segments:
            if self.is_quality_segment(segment):
                quality_segments.append(segment)
        
        return quality_segments
    
    def split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split very long paragraphs into smaller, coherent segments."""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            
            # Create segment when we have 2-4 sentences or hit a natural break
            if (len(current_segment) >= 2 and 
                (len(' '.join(current_segment).split()) >= 40 or 
                 self.is_natural_break(sentence))):
                segments.append(' '.join(current_segment))
                current_segment = []
        
        # Add remaining sentences
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def is_natural_break(self, sentence: str) -> bool:
        """Identify natural breaking points in the text."""
        break_indicators = [
            'yes or no?', 'isn\'t it?', 'alright?', 'okay?', 
            'you see?', 'understand?', 'right?', 'yes?'
        ]
        
        sentence_lower = sentence.lower().strip()
        return any(sentence_lower.endswith(indicator) for indicator in break_indicators)
    
    def is_quality_segment(self, segment: str) -> bool:
        """Enhanced quality check for segments with less aggressive filtering."""
        words = segment.split()
        
        # Basic length check (more permissive)
        if len(words) < 15 or len(words) > 150:  # Reduced minimum from 20 to 15
            return False
        
        # Check for meaningful content
        meaningful_words = ['consciousness', 'meditation', 'awareness', 'mind', 'body',
                          'life', 'existence', 'being', 'reality', 'truth', 'wisdom',
                          'spiritual', 'divine', 'soul', 'enlightenment', 'peace',
                          'anxiety', 'problem', 'solution', 'understand', 'important',
                          'experience', 'process', 'human', 'nature', 'energy',
                          'question', 'answer', 'know', 'see', 'feel', 'think',
                          'create', 'become', 'transform', 'change', 'grow']
        
        segment_lower = segment.lower()
        meaningful_count = sum(1 for word in meaningful_words if word in segment_lower)
        
        # More lenient meaningful content requirement
        if meaningful_count < 1:  # Reduced from 2 to 1
            return False
        
        # Avoid repetitive segments (like "Thank you" repeated)
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.25:  # More lenient (was 0.3)
            return False
        
        # Check for complete thoughts (should end with proper punctuation)
        if not segment.strip()[-1] in '.!?':
            return False
        
        return True
    
    def segment_by_topics(self, text: str) -> List[str]:
        """Segment text into coherent topic-based chunks."""
        # Split on strong topic transitions
        segments = []
        
        # Look for natural break points
        break_patterns = [
            r'\n\n+',  # Paragraph breaks
            r'(?<=\.)\s+(?=[A-Z][a-z]+,)',  # Before direct address
            r'(?<=\.)\s+(?=So[,\s])',  # Before "So," transitions
            r'(?<=\.)\s+(?=Now[,\s])',  # Before "Now," transitions
            r'(?<=\.)\s+(?=See[,\s])',  # Before "See," explanations
            r'(?<=\.)\s+(?=But[,\s])',  # Before "But," contrasts
            r'(?<=\.)\s+(?=Let me)',  # Before "Let me" explanations
            r'(?<=\.)\s+(?=I want you to)',  # Before direct instructions
        ]
        
        current_text = text
        for pattern in break_patterns:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                segments.extend([part.strip() for part in parts if part.strip()])
                break
        else:
            # If no natural breaks found, split by sentences in chunks
            sentences = re.split(r'(?<=\.)\s+', text)
            chunk_size = 5  # 5 sentences per chunk
            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i+chunk_size])
                if chunk.strip():
                    segments.append(chunk.strip())
        
        # Filter out very short segments
        segments = [seg for seg in segments if len(seg.split()) > 10]
        return segments
    
    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Enhanced Q&A extraction that works with both paragraph and single-line formats."""
        qa_pairs = []
        
        # Method 1: Direct question-answer patterns
        # Pattern for explicit Q&A (Question? Answer.)
        qa_pattern = r'([^.!?]*\?)[\s]*([^.!?]*[.!])'
        matches = re.findall(qa_pattern, text, re.IGNORECASE)
        
        for question, answer in matches:
            q = question.strip()
            a = answer.strip()
            if len(q.split()) > 3 and len(a.split()) > 8:
                qa_pairs.append((q + "?", a))
        
        # Method 2: Rhetorical questions with explanations
        # Look for questions followed by explanatory content
        rhetorical_pattern = r'([^.!?]*\?)[\s]*([^?]*?)(?=\?|$|\.[\s]*[A-Z])'
        rhetorical_matches = re.findall(rhetorical_pattern, text, re.DOTALL)
        
        for question, explanation in rhetorical_matches:
            q = question.strip()
            exp = explanation.strip()
            if (len(q.split()) > 3 and len(exp.split()) > 10 and 
                len(exp.split()) < 100):  # Not too long
                # Clean up the explanation
                exp = re.sub(r'\s+', ' ', exp).strip()
                if exp and not exp.endswith('.'):
                    exp += '.'
                qa_pairs.append((q + "?", exp))
        
        # Method 3: Teaching patterns (common in spiritual discourse)
        teaching_patterns = [
            r'(What (?:is|does|happens|should)[^?]*\?)[\s]*([^.!?]*[.!])',
            r'(How (?:do|can|would)[^?]*\?)[\s]*([^.!?]*[.!])',
            r'(Why (?:do|does|is)[^?]*\?)[\s]*([^.!?]*[.!])',
            r'(When (?:you|we|one)[^?]*\?)[\s]*([^.!?]*[.!])',
            r'(If (?:you|we|one)[^?]*\?)[\s]*([^.!?]*[.!])'
        ]
        
        for pattern in teaching_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for question, answer in matches:
                q = question.strip()
                a = answer.strip()
                if len(q.split()) > 4 and len(a.split()) > 8:
                    qa_pairs.append((q, a))
        
        # Method 4: Statement-explanation patterns
        # Look for definitive statements followed by explanations
        statement_pattern = r'([^.!?]*(?:is|are|means|happens)[^.!?]*[.!])[\s]*([^.!?]*[.!])'
        statement_matches = re.findall(statement_pattern, text)
        
        for statement, explanation in statement_matches:
            stmt = statement.strip()
            exp = explanation.strip()
            if (len(stmt.split()) > 5 and len(exp.split()) > 10 and
                len(stmt.split()) < 25 and len(exp.split()) < 80):
                # Convert statement to question
                question = self.statement_to_question(stmt)
                if question:
                    qa_pairs.append((question, exp))
        
        # Remove duplicates and filter quality
        unique_pairs = []
        seen_questions = set()
        
        for q, a in qa_pairs:
            q_clean = re.sub(r'\s+', ' ', q.lower().strip())
            if (q_clean not in seen_questions and 
                self.is_quality_qa_pair(q, a)):
                unique_pairs.append((q, a))
                seen_questions.add(q_clean)
        
        return unique_pairs
    
    def statement_to_question(self, statement: str) -> str:
        """Convert a statement into a question for Q&A pairs."""
        statement = statement.strip()
        
        # Simple patterns to convert statements to questions
        if ' is ' in statement.lower():
            # "Anxiety is..." -> "What is anxiety?"
            parts = statement.lower().split(' is ', 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                if len(subject.split()) <= 3:
                    return f"What is {subject}?"
        
        if ' means ' in statement.lower():
            # "Yoga means..." -> "What does yoga mean?"
            parts = statement.lower().split(' means ', 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                if len(subject.split()) <= 3:
                    return f"What does {subject} mean?"
        
        if ' happens ' in statement.lower():
            # "This happens when..." -> "What happens when...?"
            if statement.lower().startswith('this happens'):
                return statement.replace('This happens', 'What happens', 1) + "?"
        
        return None
    
    def is_quality_qa_pair(self, question: str, answer: str) -> bool:
        """Check if a Q&A pair meets quality standards."""
        q_words = question.split()
        a_words = answer.split()
        
        # Length checks
        if len(q_words) < 4 or len(q_words) > 30:
            return False
        if len(a_words) < 8 or len(a_words) > 150:
            return False
        
        # Question should end with question mark
        if not question.strip().endswith('?'):
            return False
        
        # Answer should end with proper punctuation
        if not answer.strip()[-1] in '.!':
            return False
        
        # Avoid very repetitive content
        q_unique = len(set(q_words)) / len(q_words)
        a_unique = len(set(a_words)) / len(a_words)
        
        if q_unique < 0.6 or a_unique < 0.4:
            return False
        
        # Check for meaningful content
        meaningful_words = ['what', 'how', 'why', 'when', 'where', 'who',
                          'consciousness', 'meditation', 'mind', 'life', 'spiritual',
                          'anxiety', 'problem', 'solution', 'understand', 'important']
        
        q_lower = question.lower()
        has_meaningful = any(word in q_lower for word in meaningful_words)
        
        return has_meaningful
    
    def generate_instruction_response_pairs(self, segments: List[str]) -> List[Dict]:
        """Generate instruction-response pairs optimized for LLAMA instruction-following."""
        pairs = []
        
        # Enhanced instruction templates that match the speaker's teaching style
        instruction_templates = [
            "What is your understanding of {}?",
            "How would you explain {} to someone seeking wisdom?",
            "Can you share your perspective on {}?",
            "What does {} truly mean in the context of human experience?",
            "How should one approach the concept of {}?",
            "What wisdom can you offer about {}?",
            "Please elaborate on the nature of {}",
            "How does {} relate to consciousness and human life?",
            "What insights do you have about {}?",
            "Can you guide me in understanding {}?",
        ]
        
        # Question-style instructions that prompt teaching responses
        question_templates = [
            "Someone asks: What is {}? How would you respond?",
            "A seeker wants to understand {}. What would you tell them?",
            "How would you teach someone about {}?",
            "If someone is confused about {}, how would you clarify it?",
            "What would you say to someone struggling to understand {}?",
        ]
        
        # Wisdom-seeking instructions
        wisdom_templates = [
            "Share your wisdom about {}",
            "Offer guidance on {}",
            "Provide insight into {}",
            "Illuminate the truth about {}",
            "Help me understand the deeper meaning of {}",
        ]
        
        all_templates = instruction_templates + question_templates + wisdom_templates
        
        for segment in segments:
            # Extract key concepts from the segment
            words = segment.split()
            if len(words) < 20:
                continue
                
            # Expanded key terms based on the actual transcript content
            key_terms = []
            spiritual_terms = [
                'consciousness', 'meditation', 'awareness', 'mind', 'body', 'life', 'existence', 
                'being', 'reality', 'truth', 'wisdom', 'spiritual', 'divine', 'soul', 
                'enlightenment', 'peace', 'masculine', 'feminine', 'biology', 'identity',
                'survival', 'expansion', 'boundaries', 'freedom', 'liberation', 'intelligence',
                'creation', 'memory', 'thought', 'brain', 'yoga', 'meditation', 'prayer',
                'listening', 'waiting', 'willingness', 'volunteer', 'vision', 'desire',
                'joy', 'suffering', 'fear', 'love', 'compassion', 'understanding',
                'human', 'nature', 'universe', 'cosmos', 'energy', 'breath', 'death',
                'birth', 'transformation', 'growth', 'experience', 'perception'
            ]
            
            # Also look for key phrases that indicate teaching moments
            teaching_phrases = [
                'see', 'understand', 'realize', 'know', 'experience', 'become',
                'must', 'should', 'need to', 'have to', 'important', 'essential',
                'the question is', 'the problem is', 'what happens is'
            ]
            
            for term in spiritual_terms:
                if term in segment.lower():
                    key_terms.append(term)
            
            # Check for teaching phrases to create contextual instructions
            has_teaching_phrase = any(phrase in segment.lower() for phrase in teaching_phrases)
            
            if key_terms:
                # Create instruction based on key terms
                main_term = random.choice(key_terms)
                instruction = random.choice(all_templates).format(main_term)
                
                pairs.append({
                    "instruction": instruction,
                    "input": "",
                    "output": segment
                })
            elif has_teaching_phrase:
                # For segments with teaching language but no specific key terms
                teaching_instructions = [
                    "What wisdom would you share with someone seeking understanding?",
                    "How would you guide someone on their spiritual journey?",
                    "What important truth should people realize?",
                    "Share your perspective on human nature and consciousness.",
                    "What essential understanding should one develop?",
                ]
                pairs.append({
                    "instruction": random.choice(teaching_instructions),
                    "input": "",
                    "output": segment
                })
            else:
                # Generic instruction for other segments
                pairs.append({
                    "instruction": "Share your wisdom and perspective on this topic.",
                    "input": "",
                    "output": segment
                })
        
        return pairs
    
    def generate_conversation_format(self, segments: List[str]) -> List[Dict]:
        """Generate conversation-style training data."""
        conversations = []
        
        user_prompts = [
            "Can you share some wisdom about life?",
            "I'm seeking guidance. What would you tell me?",
            "How can I better understand myself?",
            "What is the nature of consciousness?",
            "How should I approach spiritual growth?",
            "What does it mean to be truly alive?",
            "Can you help me understand the mind?",
            "What is the path to inner peace?",
            "How can I transcend my limitations?",
            "What is the essence of being human?",
        ]
        
        for i, segment in enumerate(segments):
            if len(segment.split()) < 15:
                continue
                
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": random.choice(user_prompts)
                    },
                    {
                        "role": "assistant", 
                        "content": segment
                    }
                ]
            }
            conversations.append(conversation)
        
        return conversations
    
    def generate_qa_format(self, qa_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Generate Q&A format training data."""
        qa_data = []
        
        for question, answer in qa_pairs:
            qa_data.append({
                "question": question,
                "answer": answer,
                "source": "speech_transcript"
            })
        
        return qa_data
    
    def generate_teaching_examples(self, segments: List[str]) -> List[Dict]:
        """Generate direct teaching examples that capture the speaker's wisdom-sharing style."""
        teaching_examples = []
        
        # Instructions that prompt the model to teach in the speaker's style
        teaching_prompts = [
            "Share a profound insight about human nature.",
            "Explain what it means to live consciously.",
            "Offer guidance for someone seeking spiritual growth.",
            "What is the most important thing people should understand about themselves?",
            "How can someone transcend their limitations?",
            "What wisdom would you share about the nature of existence?",
            "Explain the relationship between mind, body, and consciousness.",
            "How should one approach the journey of self-discovery?",
            "What does it mean to be truly free?",
            "Share your understanding of what makes life meaningful.",
            "How can someone move beyond fear and suffering?",
            "What is the essence of spiritual wisdom?",
            "Explain the importance of consciousness in human life.",
            "How does one develop true understanding?",
            "What would you tell someone who feels lost in life?",
        ]
        
        # Select the most wisdom-rich segments for direct teaching
        wisdom_segments = []
        wisdom_keywords = [
            'consciousness', 'understand', 'realize', 'see', 'wisdom', 'truth',
            'important', 'must', 'essential', 'fundamental', 'nature', 'life',
            'experience', 'become', 'transcend', 'beyond', 'freedom', 'liberation'
        ]
        
        for segment in segments:
            if len(segment.split()) > 30:  # Longer segments for teaching
                keyword_count = sum(1 for keyword in wisdom_keywords if keyword in segment.lower())
                if keyword_count >= 2:  # Segments with multiple wisdom keywords
                    wisdom_segments.append(segment)
        
        # Create teaching examples
        for i, segment in enumerate(wisdom_segments[:15]):  # Limit to top 15 wisdom segments
            teaching_examples.append({
                "instruction": teaching_prompts[i % len(teaching_prompts)],
                "input": "",
                "output": segment
            })
        
        return teaching_examples
    
    def save_dataset(self, data: List[Dict], filename: str, format_type: str):
        """Save dataset in specified format."""
        output_dir = Path("datasets")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        if format_type == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} examples to {filepath}")
    
    def generate_all_formats(self):
        """Generate datasets in all supported formats."""
        print("🚀 Starting dataset generation...")
        
        # Load and process transcripts
        raw_transcripts = self.load_transcripts()
        
        all_segments = []
        all_qa_pairs = []
        
        for i, transcript in enumerate(raw_transcripts):
            print(f"📝 Processing transcript {i+1}/{len(raw_transcripts)}...")
            
            # Clean the text
            cleaned = self.clean_text(transcript)
            
            # Segment into topics
            segments = self.segment_text(cleaned)
            all_segments.extend(segments)
            
            # Extract Q&A pairs
            qa_pairs = self.extract_qa_pairs(cleaned)
            all_qa_pairs.extend(qa_pairs)
        
        print(f"✨ Generated {len(all_segments)} segments and {len(all_qa_pairs)} Q&A pairs")
        
        # Generate different dataset formats
        print("\n🎯 Generating training datasets...")
        
        # 1. Primary: Alpaca-style instruction-response (optimized for LLAMA)
        alpaca_data = self.generate_instruction_response_pairs(all_segments)
        self.save_dataset(alpaca_data, "alpaca_format.jsonl", "jsonl")
        
        # 2. Q&A format converted to Alpaca style for LLAMA
        qa_data = self.generate_qa_format(all_qa_pairs)
        qa_alpaca_data = []
        for qa in qa_data:
            qa_alpaca_data.append({
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"]
            })
        self.save_dataset(qa_alpaca_data, "qa_alpaca_format.jsonl", "jsonl")
        
        # 3. Combined LLAMA-optimized format
        llama_combined = alpaca_data + qa_alpaca_data
        
        # Add some direct teaching examples
        teaching_examples = self.generate_teaching_examples(all_segments)
        llama_combined.extend(teaching_examples)
        
        # Shuffle for better training
        random.shuffle(llama_combined)
        self.save_dataset(llama_combined, "llama_optimized.jsonl", "jsonl")
        
        # 4. Keep conversation format for reference
        conversation_data = self.generate_conversation_format(all_segments)
        self.save_dataset(conversation_data, "conversation_format.jsonl", "jsonl")
        
        # 5. Pure Q&A format for reference
        self.save_dataset(qa_data, "qa_format.json", "json")
        
        # Generate summary
        self.generate_summary(len(all_segments), len(all_qa_pairs), len(llama_combined))
    
    def generate_summary(self, segments: int, qa_pairs: int, total_examples: int):
        """Generate a summary of the dataset creation process."""
        summary = {
            "dataset_info": {
                "total_segments": segments,
                "qa_pairs_extracted": qa_pairs,
                "total_training_examples": total_examples,
                "target_model": "LLAMA (optimized)",
                "formats_generated": [
                    "llama_optimized.jsonl (RECOMMENDED)",
                    "alpaca_format.jsonl",
                    "qa_alpaca_format.jsonl",
                    "conversation_format.jsonl", 
                    "qa_format.json"
                ]
            },
            "llama_optimization": {
                "primary_format": "llama_optimized.jsonl",
                "instruction_style": "Question-answering and wisdom-sharing focused",
                "response_style": "Preserves speaker's authentic teaching voice",
                "key_features": [
                    "Enhanced instruction templates for spiritual/philosophical content",
                    "Optimized Q&A extraction from natural speech patterns",
                    "Direct teaching examples for wisdom-sharing",
                    "Expanded keyword recognition for better context matching"
                ]
            },
            "usage_recommendations": {
                "llama_optimized": "🎯 PRIMARY: Best for LLAMA instruction-following with authentic voice",
                "alpaca_format": "Standard Alpaca format for general instruction-following",
                "qa_alpaca_format": "Q&A pairs in Alpaca format for focused question-answering",
                "conversation_format": "Reference: Chat-style format (not optimal for LLAMA)",
                "qa_format": "Reference: Pure Q&A pairs in JSON format"
            },
            "runpod_llama_tips": [
                "🚀 Use llama_optimized.jsonl for best results",
                "📊 Learning rate: 1e-5 to 3e-5 (start conservative)",
                "🔄 Batch size: 4-8 depending on GPU memory",
                "📈 Epochs: 3-5 (monitor for overfitting)",
                "⚡ Method: LoRA recommended (rank 16-64)",
                "🎯 Focus: This data is optimized for wisdom/teaching responses",
                "⚠️ Monitor: Watch for overfitting due to specialized domain",
                "🔍 Validation: Keep some examples aside for testing"
            ]
        }
        
        with open("datasets/dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("🎉 LLAMA-OPTIMIZED DATASET GENERATION COMPLETE!")
        print("="*70)
        print(f"📊 Total segments processed: {segments}")
        print(f"❓ Q&A pairs extracted: {qa_pairs}")
        print(f"🎯 Total training examples: {total_examples}")
        print(f"🦙 Target model: LLAMA (instruction-following optimized)")
        print("\n📁 Generated files:")
        for fmt in summary["dataset_info"]["formats_generated"]:
            if "RECOMMENDED" in fmt:
                print(f"   🌟 datasets/{fmt}")
            else:
                print(f"   • datasets/{fmt}")
        print("   • datasets/dataset_summary.json")
        print("\n🚀 RECOMMENDED: Use llama_optimized.jsonl for best results!")
        print("💡 Check dataset_summary.json for detailed RunPod tips!")

def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning datasets from speech transcripts")
    parser.add_argument("--data-dir", default="data", help="Directory containing transcript files")
    parser.add_argument("--speaker-name", default="Teacher", help="Name to use for the speaker")
    
    args = parser.parse_args()
    
    processor = TranscriptProcessor(args.data_dir)
    processor.speaker_name = args.speaker_name
    processor.generate_all_formats()

if __name__ == "__main__":
    main() 