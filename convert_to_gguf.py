#!/usr/bin/env python3
"""
🔄 Convert Spiritual AI to GGUF for Ollama
=========================================

Convert your Hugging Face spiritual AI model to GGUF format
so it can be used with Ollama locally.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required tools are installed."""
    try:
        import torch
        import transformers
        logger.info("✅ PyTorch and Transformers available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    
    # Check for llama.cpp convert script
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        logger.warning("⚠️  llama.cpp not found. Will attempt to clone it.")
        return False
    
    return True


def clone_llama_cpp():
    """Clone llama.cpp repository for conversion tools."""
    logger.info("📥 Cloning llama.cpp repository...")
    try:
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)
        logger.info("✅ llama.cpp cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to clone llama.cpp: {e}")
        return False


def convert_to_gguf(model_path: str, output_path: str = None):
    """Convert Hugging Face model to GGUF format."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"❌ Model path does not exist: {model_path}")
        return False
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.name}.gguf"
    
    output_path = Path(output_path)
    
    # Ensure llama.cpp is available
    if not check_dependencies():
        if not clone_llama_cpp():
            return False
    
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    
    logger.info(f"🔄 Converting {model_path} to GGUF format...")
    logger.info(f"📁 Output will be saved to: {output_path}")
    
    try:
        # Run the conversion
        cmd = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", "f16"  # Use f16 for good balance of size/quality
        ]
        
        logger.info(f"🚀 Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info(f"✅ Conversion completed! GGUF file: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Conversion failed: {e}")
        return False


def create_ollama_modelfile(gguf_path: str, model_name: str = "spiritual-ai"):
    """Create Ollama Modelfile for the converted model."""
    gguf_path = Path(gguf_path)
    
    if not gguf_path.exists():
        logger.error(f"❌ GGUF file not found: {gguf_path}")
        return False
    
    modelfile_content = f'''# Spiritual AI Ollama Model
FROM "{gguf_path.absolute()}"

# System prompt for spiritual guidance
SYSTEM """You are a wise spiritual teacher and guide. You offer compassionate, insightful guidance on spiritual matters, meditation, mindfulness, and personal growth. Your responses are thoughtful, non-dogmatic, and respectful of all spiritual traditions. You help people find inner peace, understanding, and wisdom."""

# Parameters for better spiritual responses
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|eot_id|>"

# Template for Llama 3.2 format
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
'''
    
    modelfile_path = Path("Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"✅ Modelfile created: {modelfile_path}")
    
    # Create the Ollama model
    logger.info(f"🦙 Creating Ollama model: {model_name}")
    try:
        subprocess.run([
            "ollama", "create", model_name, "-f", str(modelfile_path)
        ], check=True)
        
        logger.info(f"🎉 Ollama model '{model_name}' created successfully!")
        logger.info(f"🚀 Test it with: ollama run {model_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create Ollama model: {e}")
        return False


def main():
    """Main conversion workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Spiritual AI to GGUF for Ollama")
    parser.add_argument("--model-path", default="models/spiritual-wisdom-llama",
                       help="Path to the Hugging Face model")
    parser.add_argument("--output", help="Output GGUF file path")
    parser.add_argument("--name", default="spiritual-ai", 
                       help="Name for the Ollama model")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="Skip GGUF conversion (if already done)")
    
    args = parser.parse_args()
    
    logger.info("🧘‍♂️ Starting Spiritual AI → Ollama conversion")
    logger.info("=" * 50)
    
    gguf_path = args.output
    
    if not args.skip_conversion:
        # Step 1: Convert to GGUF
        if not convert_to_gguf(args.model_path, args.output):
            logger.error("❌ Conversion failed")
            return False
        
        if gguf_path is None:
            gguf_path = Path(args.model_path).parent / f"{Path(args.model_path).name}.gguf"
    else:
        if gguf_path is None:
            logger.error("❌ Must specify --output when using --skip-conversion")
            return False
    
    # Step 2: Create Ollama model
    if not create_ollama_modelfile(gguf_path, args.name):
        logger.error("❌ Ollama model creation failed")
        return False
    
    logger.info("🎉 Conversion complete! Your spiritual AI is ready in Ollama!")
    logger.info(f"🚀 Try it: ollama run {args.name}")


if __name__ == "__main__":
    main() 