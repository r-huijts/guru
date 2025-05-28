#!/usr/bin/env python3
"""
🧘‍♂️ Spiritual AI Evaluation Suite
==================================

Comprehensive evaluation of fine-tuned spiritual wisdom AI using multiple metrics:
- Response coherence and relevance
- Spiritual concept understanding
- Teaching quality assessment
- Comparative analysis with baseline
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpiritualAIEvaluator:
    """Comprehensive evaluation suite for spiritual wisdom AI."""
    
    def __init__(self, model_path: str = "models/spiritual-wisdom-llama"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
        # Evaluation test sets
        self.spiritual_concepts = [
            "consciousness", "meditation", "enlightenment", "mindfulness",
            "inner peace", "self-realization", "awareness", "presence",
            "compassion", "wisdom", "suffering", "attachment"
        ]
        
        self.test_questions = {
            "basic_concepts": [
                "What is consciousness?",
                "What is meditation?",
                "What is mindfulness?",
                "What is enlightenment?"
            ],
            "practical_guidance": [
                "How can I find inner peace?",
                "How do I overcome anxiety?",
                "How can I be more present?",
                "How do I deal with difficult emotions?"
            ],
            "philosophical_depth": [
                "What is the nature of reality?",
                "What is the meaning of life?",
                "What is the relationship between mind and consciousness?",
                "How does suffering lead to wisdom?"
            ],
            "teaching_scenarios": [
                "A student asks: I feel lost in life, what should I do?",
                "Someone says: I can't stop my racing thoughts during meditation",
                "A person asks: How do I know if I'm making spiritual progress?",
                "Someone struggles: I feel disconnected from others, how can I connect?"
            ]
        }
    
    def load_model(self):
        """Load the fine-tuned model for evaluation."""
        logger.info(f"🚀 Loading model from {self.model_path}")
        
        try:
            # Try Unsloth first
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            logger.info("✅ Unsloth model loaded for inference")
            
        except Exception as e:
            logger.warning(f"⚠️ Unsloth loading failed: {e}")
            
            # Fallback to standard transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("✅ Standard transformers model loaded")
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate a response to a given prompt."""
        # Format prompt
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
            
            # Clean artifacts
            response = response.replace("user", "").replace("assistant", "").strip()
            response = re.sub(r'\n\s*\n', '\n\n', response)
            response = re.sub(r'^\s+', '', response, flags=re.MULTILINE)
            response = response.strip()
        else:
            response = full_response
        
        return response
    
    def evaluate_spiritual_concepts(self) -> Dict:
        """Test understanding of core spiritual concepts."""
        logger.info("🧘‍♂️ Evaluating spiritual concept understanding...")
        
        results = {}
        for concept in self.spiritual_concepts:
            prompt = f"Explain the spiritual concept of {concept}"
            response = self.generate_response(prompt)
            
            # Simple metrics
            concept_mentioned = concept.lower() in response.lower()
            response_length = len(response.split())
            
            results[concept] = {
                "response": response,
                "concept_mentioned": concept_mentioned,
                "response_length": response_length,
                "coherent": response_length > 10 and not response.startswith("I don't")
            }
        
        return results
    
    def evaluate_question_categories(self) -> Dict:
        """Evaluate responses across different question categories."""
        logger.info("📊 Evaluating question categories...")
        
        results = {}
        for category, questions in self.test_questions.items():
            logger.info(f"   Testing {category}...")
            category_results = []
            
            for question in questions:
                response = self.generate_response(question)
                
                # Quality metrics
                metrics = {
                    "question": question,
                    "response": response,
                    "length": len(response.split()),
                    "has_spiritual_terms": any(term in response.lower() for term in 
                                             ["consciousness", "awareness", "mind", "peace", "meditation", "wisdom"]),
                    "is_coherent": len(response.split()) > 15 and not response.startswith("I don't"),
                    "is_helpful": "?" not in response or len(response.split()) > 20
                }
                
                category_results.append(metrics)
            
            results[category] = category_results
        
        return results
    
    def calculate_quality_scores(self, concept_results: Dict, category_results: Dict) -> Dict:
        """Calculate overall quality scores."""
        logger.info("📈 Calculating quality scores...")
        
        # Concept understanding score
        concept_scores = []
        for concept, result in concept_results.items():
            score = 0
            if result["concept_mentioned"]: score += 0.3
            if result["coherent"]: score += 0.4
            if result["response_length"] > 20: score += 0.3
            concept_scores.append(score)
        
        concept_avg = sum(concept_scores) / len(concept_scores) if concept_scores else 0
        
        # Category performance scores
        category_scores = {}
        for category, results in category_results.items():
            scores = []
            for result in results:
                score = 0
                if result["has_spiritual_terms"]: score += 0.25
                if result["is_coherent"]: score += 0.35
                if result["is_helpful"]: score += 0.25
                if result["length"] > 25: score += 0.15
                scores.append(score)
            
            category_scores[category] = sum(scores) / len(scores) if scores else 0
        
        overall_score = (concept_avg + sum(category_scores.values()) / len(category_scores)) / 2
        
        return {
            "concept_understanding": concept_avg,
            "category_scores": category_scores,
            "overall_quality": overall_score,
            "grade": self.get_grade(overall_score)
        }
    
    def get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9: return "A+ (Enlightened Master)"
        elif score >= 0.8: return "A (Wise Teacher)"
        elif score >= 0.7: return "B+ (Good Guide)"
        elif score >= 0.6: return "B (Helpful Advisor)"
        elif score >= 0.5: return "C+ (Learning Student)"
        elif score >= 0.4: return "C (Beginner)"
        else: return "D (Needs More Training)"
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run the complete evaluation suite."""
        logger.info("🌟 Starting Comprehensive Spiritual AI Evaluation")
        logger.info("=" * 60)
        
        # Load model
        self.load_model()
        
        # Run evaluations
        concept_results = self.evaluate_spiritual_concepts()
        category_results = self.evaluate_question_categories()
        quality_scores = self.calculate_quality_scores(concept_results, category_results)
        
        # Compile results
        evaluation_report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "concept_understanding": concept_results,
            "category_performance": category_results,
            "quality_scores": quality_scores,
            "summary": {
                "total_concepts_tested": len(self.spiritual_concepts),
                "total_questions_tested": sum(len(q) for q in self.test_questions.values()),
                "overall_grade": quality_scores["grade"],
                "overall_score": f"{quality_scores['overall_quality']:.2f}"
            }
        }
        
        # Save report
        report_path = Path("spiritual_ai_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        # Print summary
        self.print_evaluation_summary(quality_scores)
        
        logger.info(f"📄 Full evaluation report saved to: {report_path}")
        
        return evaluation_report
    
    def print_evaluation_summary(self, scores: Dict):
        """Print a formatted evaluation summary."""
        logger.info("\n🎯 SPIRITUAL AI EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"🧘‍♂️ Concept Understanding: {scores['concept_understanding']:.2f}/1.0")
        
        logger.info("\n📊 Category Performance:")
        for category, score in scores['category_scores'].items():
            logger.info(f"   {category.replace('_', ' ').title()}: {score:.2f}/1.0")
        
        logger.info(f"\n🏆 Overall Quality Score: {scores['overall_quality']:.2f}/1.0")
        logger.info(f"🎓 Grade: {scores['grade']}")
        
        # Recommendations
        if scores['overall_quality'] >= 0.8:
            logger.info("✅ Excellent spiritual AI! Ready for production use.")
        elif scores['overall_quality'] >= 0.6:
            logger.info("👍 Good spiritual AI! Minor improvements possible.")
        else:
            logger.info("🔧 Needs improvement. Consider more training or data.")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Spiritual Wisdom AI")
    parser.add_argument("--model", default="models/spiritual-wisdom-llama", 
                       help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    evaluator = SpiritualAIEvaluator(args.model)
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main() 