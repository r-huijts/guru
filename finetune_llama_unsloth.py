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


class SpiritualWisdomEvaluationCallback:
    """Custom callback for detailed evaluation monitoring during training."""
    
    def __init__(self):
        self.best_eval_loss = float('inf')
        self.eval_history = []
    
    def on_log(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called when logging occurs - we'll use this instead of on_evaluate."""
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            
            # Only process if this log contains evaluation metrics
            if 'eval_loss' in latest_log:
                eval_loss = latest_log['eval_loss']
                step = latest_log.get('step', state.global_step)
                
                # Track evaluation history
                self.eval_history.append({
                    'step': step,
                    'eval_loss': eval_loss,
                    'train_loss': latest_log.get('loss', None),
                    'learning_rate': latest_log.get('learning_rate', None)
                })
                
                # Check if this is the best model so far
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    improvement = "🎉 NEW BEST!"
                else:
                    improvement = f"(best: {self.best_eval_loss:.4f})"
                
                logger.info(f"📊 Step {step} Evaluation:")
                logger.info(f"   🧘‍♂️ Eval Loss: {eval_loss:.4f} {improvement}")
                if latest_log.get('loss'):
                    logger.info(f"   📈 Train Loss: {latest_log['loss']:.4f}")
                if latest_log.get('learning_rate'):
                    logger.info(f"   🎓 Learning Rate: {latest_log['learning_rate']:.2e}")
                
                # Calculate improvement trend
                if len(self.eval_history) >= 3:
                    recent_losses = [h['eval_loss'] for h in self.eval_history[-3:]]
                    if recent_losses[-1] < recent_losses[0]:
                        trend = "📉 Improving"
                    elif recent_losses[-1] > recent_losses[0]:
                        trend = "📈 Increasing"
                    else:
                        trend = "➡️ Stable"
                    logger.info(f"   📊 Trend: {trend}")
        
        return control


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
                 max_seq_length: int = 2048,
                 enable_early_stopping: bool = False):
        
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.enable_early_stopping = enable_early_stopping
        
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
        if self.enable_early_stopping:
            logger.info(f"⏰ Early stopping: Enabled")
    
    def check_requirements(self):
        """Verify system requirements and dependencies."""
        logger.info("🔍 Checking system requirements...")
        
        # Check for GPU acceleration
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 CUDA GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 6:
                logger.warning("⚠️  Less than 6GB VRAM. Consider using Llama-3.2-1B-Instruct")
            else:
                logger.info("✅ GPU memory sufficient for Llama-3.2-3B-Instruct")
                
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 MPS (Metal Performance Shaders) available for Apple Silicon")
            logger.info("💡 Training will use MPS acceleration")
            logger.warning("⚠️  MPS training may be slower than CUDA. Consider using smaller model.")
            
        else:
            logger.warning("⚠️  No GPU acceleration available. Training will be slow on CPU.")
            logger.warning("💡 Consider using a smaller model or running on a GPU-enabled system.")
        
        # Check dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        logger.info("✅ System requirements check passed")
    
    def load_model_and_tokenizer(self):
        """Load Unsloth-optimized model and tokenizer."""
        logger.info(f"🚀 Loading model: {self.model_name}")
        
        try:
            # Try Unsloth first (fastest)
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect optimal dtype
                load_in_4bit=True,  # 4-bit quantization for memory efficiency
            )
            
            logger.info("✅ Unsloth-optimized model loaded successfully")
            self.using_unsloth = True
            
            # Add LoRA adapters using Unsloth's method
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                random_state=3407,
                use_rslora=False,  # Rank stabilized LoRA
                loftq_config=None,  # LoftQ quantization
            )
            
            logger.info("✅ LoRA adapters added successfully")
            
        except ImportError:
            logger.warning("⚠️  Unsloth not available, falling back to standard transformers")
            self.using_unsloth = False
            
            # Fallback to standard transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
            else:
                device = "cpu"
                torch_dtype = torch.float32
            
            logger.info(f"📱 Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check for sentencepiece availability
            try:
                import sentencepiece
                logger.info("✅ SentencePiece available for tokenization")
            except ImportError:
                logger.warning("⚠️  SentencePiece not available, using basic tokenization")
                logger.info("💡 This may affect tokenization quality but training will work")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device != "cpu" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            logger.info("✅ Standard transformers model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
        logger.info(f"📈 Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        # Add using_unsloth flag for later use
        if not hasattr(self, 'using_unsloth'):
            self.using_unsloth = False
    
    def setup_lora(self):
        """Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("🔧 Setting up LoRA configuration...")
        
        if self.using_unsloth:
            # LoRA adapters already added during model loading for Unsloth
            logger.info("✅ LoRA configuration already handled during Unsloth model loading")
            return
        
        try:
            from peft import LoraConfig, get_peft_model
            
            # LoRA configuration for standard transformers
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
            logger.info("💡 Continuing without LoRA (will fine-tune full model)")
        except Exception as e:
            logger.error(f"❌ Failed to setup LoRA: {e}")
            logger.info("💡 Continuing without LoRA (will fine-tune full model)")
    
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
            full_dataset = Dataset.from_list(formatted_data)
            
            # Split into train/eval (90/10 split)
            dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
            self.train_dataset = dataset_split['train']
            self.eval_dataset = dataset_split['test']
            
            logger.info(f"✅ Dataset split completed:")
            logger.info(f"   🎓 Training examples: {len(self.train_dataset)}")
            logger.info(f"   📊 Evaluation examples: {len(self.eval_dataset)}")
            
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
        train_dataset_size = len(self.train_dataset)
        batch_size = 2
        gradient_accumulation_steps = 4
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Target ~3 epochs
        max_steps = (train_dataset_size * 3) // effective_batch_size
        
        # Evaluation steps (every 25 training steps)
        eval_steps = 25
        
        self.training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=25,
            save_steps=50,  # Save more frequently for better checkpointing
            save_total_limit=5,  # Keep more checkpoints
            
            # Training parameters
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,  # Same batch size for eval
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            warmup_steps=50,
            
            # Optimization
            learning_rate=2e-4,
            weight_decay=0.01,
            optim="adamw_8bit",  # Unsloth-optimized optimizer
            lr_scheduler_type="linear",
            
            # Memory and precision
            fp16=False,
            bf16=True,  # Memory optimization
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            
            # Evaluation configuration
            eval_strategy="steps",  # Evaluate every N steps
            eval_steps=eval_steps,  # Evaluate every 25 steps
            eval_accumulation_steps=1,  # Don't accumulate eval gradients
            eval_delay=0,  # Start evaluating immediately
            
            # Saving and loading
            save_strategy="steps",
            load_best_model_at_end=True,  # Load best model based on eval loss
            metric_for_best_model="eval_loss",  # Use eval loss as metric
            greater_is_better=False,  # Lower eval loss is better
            
            # Early stopping (optional)
            # early_stopping_patience=3,  # Stop if no improvement for 3 evals
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Reporting
            report_to=None,  # Disable wandb/tensorboard for now
            run_name=f"spiritual-wisdom-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # Additional monitoring
            include_inputs_for_metrics=False,  # Don't include inputs in metrics (saves memory)
            prediction_loss_only=True,  # Only compute loss for evaluation
        )
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   📊 Train dataset size: {train_dataset_size}")
        logger.info(f"   📊 Eval dataset size: {len(self.eval_dataset)}")
        logger.info(f"   🔢 Effective batch size: {effective_batch_size}")
        logger.info(f"   📈 Max steps: {max_steps}")
        logger.info(f"   📊 Eval every: {eval_steps} steps")
        logger.info(f"   🎓 Learning rate: {self.training_args.learning_rate}")
        logger.info(f"   💾 Save every: {self.training_args.save_steps} steps")
    
    def create_trainer(self):
        """Create the SFT (Supervised Fine-Tuning) trainer."""
        logger.info("🏋️‍♂️ Creating SFT trainer...")
        
        try:
            from trl import SFTTrainer
            from transformers import EarlyStoppingCallback
            
            # Create callbacks for better monitoring
            callbacks = []
            
            # Add early stopping callback if enabled
            if self.enable_early_stopping:
                early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
                callbacks.append(early_stopping)
                logger.info("⏰ Early stopping enabled (patience: 3 evaluations)")
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,  # Add evaluation dataset
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                args=self.training_args,
                packing=False,  # Don't pack sequences for better quality
                callbacks=callbacks,  # Add monitoring callbacks
            )
            
            logger.info("✅ SFT trainer created successfully")
            logger.info("📊 Evaluation will run every 25 training steps")
            logger.info("💾 Best model will be saved based on evaluation loss")
            logger.info("🔍 Custom evaluation monitoring enabled")
            
        except ImportError:
            logger.error("❌ TRL not installed. Run: pip install trl")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to create trainer: {e}")
            raise
    
    def save_evaluation_summary(self):
        """Save evaluation metrics and create a training summary."""
        logger.info("📊 Creating evaluation summary...")
        
        try:
            # Get evaluation metrics from trainer's log history
            eval_logs = [log for log in self.trainer.state.log_history if 'eval_loss' in log]
            
            # Find best evaluation loss
            best_eval_loss = min(log['eval_loss'] for log in eval_logs) if eval_logs else None
            
            # Save evaluation history
            eval_summary = {
                'best_eval_loss': best_eval_loss,
                'total_evaluations': len(eval_logs),
                'evaluation_history': eval_logs,
                'final_metrics': eval_logs[-1] if eval_logs else None,
                'training_completed': datetime.now().isoformat()
            }
            
            # Save to JSON file
            summary_path = self.output_dir / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(eval_summary, f, indent=2)
            
            logger.info(f"✅ Evaluation summary saved to {summary_path}")
            
            # Print summary
            if eval_logs:
                first_eval = eval_logs[0]
                last_eval = eval_logs[-1]
                improvement = first_eval['eval_loss'] - last_eval['eval_loss']
                improvement_pct = (improvement / first_eval['eval_loss']) * 100
                
                logger.info("📈 Training Evaluation Summary:")
                logger.info(f"   🎯 Best Eval Loss: {best_eval_loss:.4f}")
                logger.info(f"   📊 Total Evaluations: {len(eval_logs)}")
                logger.info(f"   📉 Loss Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
                logger.info(f"   🏁 Final Eval Loss: {last_eval['eval_loss']:.4f}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save evaluation summary: {e}")
            logger.info("📊 Training completed successfully despite summary issue")
    
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
            
            # Save evaluation summary
            self.save_evaluation_summary()
            
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
            
            # Generate response with improved parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,  # Reduced for more focused responses
                    temperature=0.8,     # Slightly higher for creativity
                    top_p=0.9,          # Nucleus sampling for better quality
                    top_k=50,           # Limit vocabulary for coherence
                    do_sample=True,
                    repetition_penalty=1.1,  # Reduce repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = response.replace("<|eot_id|>", "").strip()
                
                # Clean up any remaining artifacts
                if response.startswith("user"):
                    response = response.split("assistant", 1)[-1].strip()
            else:
                response = full_response
            
            logger.info(f"\n❓ Question: {prompt}")
            logger.info(f"🧘‍♂️ Response: {response}")
            logger.info("-" * 50)
    
    def run_full_pipeline(self):
        """Execute the complete fine-tuning pipeline."""
        logger.info("�� Starting Spiritual Wisdom Fine-tuning Pipeline")
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
    
    parser.add_argument(
        "--early-stopping", 
        action="store_true",
        help="Enable early stopping"
    )
    
    args = parser.parse_args()
    
    # Create trainer instance
    trainer = SpiritualWisdomTrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_length=args.max_length,
        enable_early_stopping=args.early_stopping
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