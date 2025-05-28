#!/usr/bin/env python3
"""
🦙 Simple Ollama Setup for Spiritual AI
=======================================

Simplified approach: Use llama.cpp to convert directly from HF to GGUF
without dealing with complex LoRA merging issues.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_llama_cpp():
    """Check if llama.cpp is available."""
    try:
        result = subprocess.run(['python', '-c', 'import llama_cpp'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ llama-cpp-python is available")
            return True
    except:
        pass
    
    logger.info("📦 Installing llama-cpp-python...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'llama-cpp-python'], 
                      check=True)
        logger.info("✅ llama-cpp-python installed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Failed to install llama-cpp-python")
        return False


def download_conversion_script():
    """Download the official llama.cpp conversion script."""
    script_url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    script_path = Path("convert_hf_to_gguf.py")
    
    if script_path.exists():
        logger.info("✅ Conversion script already exists")
        return str(script_path)
    
    logger.info("📥 Downloading conversion script...")
    try:
        import urllib.request
        urllib.request.urlretrieve(script_url, script_path)
        logger.info("✅ Conversion script downloaded")
        return str(script_path)
    except Exception as e:
        logger.error(f"❌ Failed to download conversion script: {e}")
        return None


def create_modelfile(model_name: str = "spiritual-ai"):
    """Create Ollama Modelfile for the spiritual AI."""
    modelfile_content = f'''FROM ./{model_name}.gguf

TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM """You are a wise spiritual teacher and guide. You provide thoughtful, compassionate responses about spiritual matters, meditation, consciousness, and personal growth. Your responses are grounded in wisdom traditions while being accessible to modern seekers."""
'''
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    logger.info("✅ Modelfile created")
    return "Modelfile"


def main():
    """Main setup process."""
    logger.info("🧘‍♂️ Setting up Spiritual AI for Ollama")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_llama_cpp():
        return False
    
    # Download conversion script
    script_path = download_conversion_script()
    if not script_path:
        return False
    
    # Create Modelfile
    create_modelfile()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Convert your HF model to GGUF:")
    logger.info("   python convert_hf_to_gguf.py RuudFontys/spiritual-wisdom-llama-3b --outfile spiritual-ai.gguf")
    logger.info("\n2. Create Ollama model:")
    logger.info("   ollama create spiritual-ai -f Modelfile")
    logger.info("\n3. Test your model:")
    logger.info("   ollama run spiritual-ai 'What is consciousness?'")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 