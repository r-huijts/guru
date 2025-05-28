#!/usr/bin/env python3
"""
🤗 Download and Merge Spiritual AI Model
========================================

Download LoRA adapters from Hugging Face, merge with base model,
and prepare for GGUF conversion to Ollama.
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_and_merge_model(
    hf_model_id: str = "RuudFontys/spiritual-wisdom-llama-3b",
    base_model_id: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    output_dir: str = "models/spiritual-wisdom-llama-merged",
    use_auth_token: bool = True
):
    """
    Download LoRA adapters from HF and merge with base model.
    
    Args:
        hf_model_id: Your Hugging Face model repository
        base_model_id: Base model used for training
        output_dir: Where to save the merged model
        use_auth_token: Whether to use HF authentication
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧘‍♂️ Starting Spiritual AI Download and Merge Process")
    logger.info("=" * 60)
    
    # Step 1: Load base model
    logger.info(f"📥 Loading base model: {base_model_id}")
    try:
        # Load base model in full precision for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Base model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        return False
    
    # Step 2: Load tokenizer
    logger.info(f"📝 Loading tokenizer from: {base_model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Step 3: Load LoRA adapters from Hugging Face
    logger.info(f"🔄 Loading LoRA adapters from: {hf_model_id}")
    try:
        # Load the PEFT model (LoRA adapters)
        model = PeftModel.from_pretrained(
            base_model,
            hf_model_id,
            use_auth_token=use_auth_token
        )
        logger.info("✅ LoRA adapters loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA adapters: {e}")
        logger.info("💡 Make sure you're logged in: huggingface-cli login")
        return False
    
    # Step 4: Merge LoRA adapters with base model
    logger.info("🔗 Merging LoRA adapters with base model...")
    try:
        merged_model = model.merge_and_unload()
        logger.info("✅ Model merged successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to merge model: {e}")
        return False
    
    # Step 5: Save merged model
    logger.info(f"💾 Saving merged model to: {output_path}")
    try:
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info("✅ Merged model saved successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False
    
    # Step 6: Verify the saved model
    logger.info("🔍 Verifying saved model...")
    saved_files = list(output_path.glob("*"))
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    
    for req_file in required_files:
        if any(req_file in str(f) for f in saved_files):
            logger.info(f"✅ Found: {req_file}")
        else:
            logger.warning(f"⚠️  Missing: {req_file}")
    
    # Check for model weights
    weight_files = [f for f in saved_files if f.suffix in ['.bin', '.safetensors']]
    if weight_files:
        logger.info(f"✅ Found {len(weight_files)} weight file(s)")
        for wf in weight_files:
            size_mb = wf.stat().st_size / (1024 * 1024)
            logger.info(f"   📁 {wf.name}: {size_mb:.1f} MB")
    else:
        logger.warning("⚠️  No weight files found")
    
    logger.info("\n🎉 Download and merge completed successfully!")
    logger.info(f"📁 Merged model location: {output_path.absolute()}")
    logger.info("🚀 Next step: Convert to GGUF with convert_to_gguf.py")
    
    return True


def test_merged_model(model_path: str = "models/spiritual-wisdom-llama-merged"):
    """Test the merged model with a simple generation."""
    logger.info("🧪 Testing merged model...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test prompt
        prompt = "What is the nature of consciousness?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("✅ Model test successful!")
        logger.info(f"🧘‍♂️ Sample response: {response[len(prompt):].strip()[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and merge spiritual AI model")
    parser.add_argument("--hf-model", default="RuudFontys/spiritual-wisdom-llama-3b",
                       help="Hugging Face model ID with LoRA adapters")
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                       help="Base model ID")
    parser.add_argument("--output", default="models/spiritual-wisdom-llama-merged",
                       help="Output directory for merged model")
    parser.add_argument("--test", action="store_true",
                       help="Test the merged model after creation")
    parser.add_argument("--no-auth", action="store_true",
                       help="Don't use HF authentication token")
    
    args = parser.parse_args()
    
    # Download and merge
    success = download_and_merge_model(
        hf_model_id=args.hf_model,
        base_model_id=args.base_model,
        output_dir=args.output,
        use_auth_token=not args.no_auth
    )
    
    if not success:
        logger.error("❌ Download and merge failed")
        return False
    
    # Test if requested
    if args.test:
        test_success = test_merged_model(args.output)
        if not test_success:
            logger.warning("⚠️  Model test failed, but merge may still be valid")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Convert to GGUF: python convert_to_gguf.py --model-path models/spiritual-wisdom-llama-merged")
    logger.info("2. Test in Ollama: ollama run spiritual-ai")
    
    return True


if __name__ == "__main__":
    main() 