#!/usr/bin/env python3
"""
🚀 Unsloth + Llama 3.2 Fine-tuning Script
==========================================

Fine-tunes Llama 3.2 using Unsloth for spiritual wisdom instruction following.
Based on the comprehensive implementation plan and optimized for 2x faster training.

Author: AI Assistant
Dataset: 521 spiritual wisdom entries in instruction-input-output format
Target: unsloth/Llama-3.2-3B-Instruct (6GB VRAM)
"""

import json
import torch
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpiritualWisdomTrainer:
    """
    Spiritual Wisdom Fine-tuning Pipeline using Unsloth + Llama 3.2
    
    Features:
    - 2x faster training with Unsloth optimizations
    - Memory efficient 4-bit quantization
    - QLoRA for parameter-efficient fine-tuning
    - Comprehensive monitoring and checkpointing
    """
    
    def __init__(self, 
                 model_name: str = "unsloth/Llama-3.2-3B-Instruct",
                 dataset_path: str = "datasets/llama_optimized.jsonl",
                 output_dir: str = "models/spiritual-wisdom-llama",
                 max_seq_length: int = 2048):
        
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        logger.info(f"🧘‍♂️ Initializing Spiritual Wisdom Trainer")
        logger.info(f"📱 Model: {self.model_name}")
        logger.info(f"📊 Dataset: {self.dataset_path}")
        logger.info(f"💾 Output: {self.output_dir}")
    
    def check_requirements(self):
        """Verify system requirements and dependencies."""
        logger.info("🔍 Checking system requirements...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available. Training will be slow on CPU.")
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 6:
                logger.warning("⚠️  Less than 6GB VRAM. Consider using Llama-3.2-1B-Instruct")
        
        # Check dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        logger.info("✅ System requirements check passed")
    
    def load_model_and_tokenizer(self):
        """Load Unsloth-optimized model and tokenizer."""
        logger.info(f"🚀 Loading Unsloth-optimized model: {self.model_name}")
        
        try:
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect optimal dtype
                load_in_4bit=True,  # 4-bit quantization for memory efficiency
            )
            
            logger.info("✅ Model and tokenizer loaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"📊 Total parameters: {total_params:,}")
            logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
            
        except ImportError:
            logger.error("❌ Unsloth not installed. Run: pip install unsloth[colab-new]")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_lora(self):
        """Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("🔧 Setting up LoRA configuration...")
        
        try:
            from peft import LoraConfig, get_peft_model
            
            # LoRA configuration based on documentation
            lora_config = LoraConfig(
                r=16,  # LoRA rank (conservative start)
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                    "gate_proj", "up_proj", "down_proj"      # MLP layers
                ],
                lora_alpha=16,      # LoRA scaling (matches rank for 1.0 scaling)
                lora_dropout=0.05,  # Prevent overfitting
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters after LoRA
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"🎯 LoRA trainable parameters: {trainable_params:,}")
            logger.info(f"📈 Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            
        except ImportError:
            logger.error("❌ PEFT not installed. Run: pip install peft")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to setup LoRA: {e}")
            raise
    
    def load_dataset(self):
        """Load and prepare the spiritual wisdom dataset."""
        logger.info(f"📚 Loading dataset from {self.dataset_path}")
        
        try:
            # Load JSONL dataset
            data = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            logger.info(f"📊 Loaded {len(data)} training examples")
            
            # Format prompts for Llama 3.2 chat template
            formatted_data = []
            for item in data:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output = item.get('output', '')
                
                # Combine instruction and input if both exist
                if input_text:
                    user_message = f"{instruction}\n\n{input_text}"
                else:
                    user_message = instruction
                
                # Format using Llama 3.2 chat template
                formatted_text = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{output}<|eot_id|>"
                )
                
                formatted_data.append({"text": formatted_text})
            
            # Convert to Hugging Face dataset
            from datasets import Dataset
            self.dataset = Dataset.from_list(formatted_data)
            
            logger.info("✅ Dataset formatted and ready for training")
            
            # Show sample
            logger.info("📝 Sample formatted example:")
            logger.info(f"{formatted_data[0]['text'][:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def setup_training_arguments(self):
        """Configure training arguments optimized for Unsloth."""
        logger.info("⚙️ Setting up training arguments...")
        
        from transformers import TrainingArguments
        
        # Calculate steps based on dataset size
        dataset_size = len(self.dataset)
        batch_size = 2
        gradient_accumulation_steps = 4
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Target ~3 epochs
        max_steps = (dataset_size * 3) // effective_batch_size
        
        self.training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=25,
            save_steps=100,
            save_total_limit=3,
            
            # Training parameters
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            warmup_steps=50,
            
            # Optimization
            learning_rate=2e-4,
            weight_decay=0.01,
            optim="adamw_8bit",  # Unsloth-optimized optimizer
            lr_scheduler_type="linear",
            
            # Memory and precision
            fp16=True,  # Memory optimization
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            
            # Evaluation and saving
            evaluation_strategy="no",  # No validation set for now
            save_strategy="steps",
            load_best_model_at_end=False,
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Reporting
            report_to=None,  # Disable wandb/tensorboard for now
            run_name=f"spiritual-wisdom-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   📊 Dataset size: {dataset_size}")
        logger.info(f"   🔢 Effective batch size: {effective_batch_size}")
        logger.info(f"   📈 Max steps: {max_steps}")
        logger.info(f"   🎓 Learning rate: {self.training_args.learning_rate}")
        logger.info(f"   💾 Save every: {self.training_args.save_steps} steps")
    
    def create_trainer(self):
        """Create the SFT (Supervised Fine-Tuning) trainer."""
        logger.info("🏋️‍♂️ Creating SFT trainer...")
        
        try:
            from trl import SFTTrainer
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                args=self.training_args,
                packing=False,  # Don't pack sequences for better quality
            )
            
            logger.info("✅ SFT trainer created successfully")
            
        except ImportError:
            logger.error("❌ TRL not installed. Run: pip install trl")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to create trainer: {e}")
            raise
    
    def train(self):
        """Execute the fine-tuning process."""
        logger.info("🚀 Starting fine-tuning process...")
        logger.info("🧘‍♂️ Teaching AI the ancient wisdom of spiritual masters...")
        
        try:
            # Start training
            start_time = datetime.now()
            
            self.trainer.train()
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"⏱️  Training duration: {training_duration}")
            
            # Save final model
            logger.info("💾 Saving final model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info(f"🎉 Model saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def test_model(self, test_prompts: Optional[List[str]] = None):
        """Test the fine-tuned model with sample spiritual questions."""
        logger.info("🧪 Testing fine-tuned model...")
        
        if test_prompts is None:
            test_prompts = [
                "What is the nature of consciousness?",
                "How can I find inner peace?",
                "What is the purpose of meditation?",
                "How do I overcome anxiety and fear?",
                "What is the meaning of life?"
            ]
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("🤖 Generating responses to test questions:")
        
        for prompt in test_prompts:
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
                max_length=self.max_seq_length
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = response.replace("<|eot_id|>", "").strip()
            
            logger.info(f"\n❓ Question: {prompt}")
            logger.info(f"🧘‍♂️ Response: {response[:200]}...")
            logger.info("-" * 50)
    
    def run_full_pipeline(self):
        """Execute the complete fine-tuning pipeline."""
        logger.info("🌟 Starting Spiritual Wisdom Fine-tuning Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check requirements
            self.check_requirements()
            
            # Step 2: Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Step 3: Setup LoRA
            self.setup_lora()
            
            # Step 4: Load and prepare dataset
            self.load_dataset()
            
            # Step 5: Setup training arguments
            self.setup_training_arguments()
            
            # Step 6: Create trainer
            self.create_trainer()
            
            # Step 7: Train the model
            self.train()
            
            # Step 8: Test the model
            self.test_model()
            
            logger.info("🎉 Pipeline completed successfully!")
            logger.info("🧘‍♂️ Your AI spiritual teacher is ready to share wisdom!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def main():
    """Main entry point for the fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 for spiritual wisdom using Unsloth"
    )
    
    parser.add_argument(
        "--model", 
        default="unsloth/Llama-3.2-3B-Instruct",
        help="Model name (default: unsloth/Llama-3.2-3B-Instruct)"
    )
    
    parser.add_argument(
        "--dataset", 
        default="datasets/llama_optimized.jsonl",
        help="Path to dataset file (default: datasets/llama_optimized.jsonl)"
    )
    
    parser.add_argument(
        "--output", 
        default="models/spiritual-wisdom-llama",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only run model testing (requires existing model)"
    )
    
    args = parser.parse_args()
    
    # Create trainer instance
    trainer = SpiritualWisdomTrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_length=args.max_length
    )
    
    if args.test_only:
        # Load existing model and test
        trainer.load_model_and_tokenizer()
        trainer.test_model()
    else:
        # Run full pipeline
        trainer.run_full_pipeline()


if __name__ == "__main__":
    main() 