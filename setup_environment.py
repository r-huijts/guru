#!/usr/bin/env python3
"""
🛠️ Environment Setup Script for Spiritual Wisdom Fine-tuning
============================================================

This script helps set up the environment for Unsloth + Llama 3.2 fine-tuning.
It checks system requirements, installs dependencies, and verifies the setup.

Usage: python setup_environment.py
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import importlib.util

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n{step}. {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible."""
    print_step("1", "Checking Python Version")
    
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required for this project")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_system_info():
    """Display system information."""
    print_step("2", "System Information")
    
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"🧠 CPU: {platform.processor()}")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print(f"🔢 GPU Count: {gpu_count}")
            
            if gpu_memory < 6:
                print("⚠️  Warning: Less than 6GB VRAM. Consider using Llama-3.2-1B-Instruct")
            else:
                print("✅ GPU memory sufficient for Llama-3.2-3B-Instruct")
        else:
            print("⚠️  CUDA not available. Training will be slow on CPU.")
    except ImportError:
        print("📦 PyTorch not installed yet")

def install_dependencies():
    """Install required dependencies."""
    print_step("3", "Installing Dependencies")
    
    # Check if requirements.txt exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing packages from requirements.txt...")
    
    try:
        # Install PyTorch with CUDA support first
        print("🔥 Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        
        # Install other requirements
        print("📚 Installing other dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed correctly."""
    print_step("4", "Verifying Installation")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("unsloth", "Unsloth")
    ]
    
    all_good = True
    
    for package, name in packages_to_check:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"✅ {name} installed")
            else:
                print(f"❌ {name} not found")
                all_good = False
        except ImportError:
            print(f"❌ {name} not found")
            all_good = False
    
    # Special check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA support available")
        else:
            print("⚠️  CUDA not available")
    except ImportError:
        print("❌ PyTorch not properly installed")
        all_good = False
    
    return all_good

def check_dataset():
    """Check if the dataset is available."""
    print_step("5", "Checking Dataset")
    
    dataset_path = Path("datasets/llama_optimized.jsonl")
    
    if dataset_path.exists():
        # Count lines in dataset
        with open(dataset_path, 'r') as f:
            line_count = sum(1 for line in f if line.strip())
        
        print(f"✅ Dataset found: {dataset_path}")
        print(f"📊 Dataset size: {line_count} examples")
        
        if line_count < 100:
            print("⚠️  Small dataset size. Consider adding more examples.")
        
        return True
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Run generate_finetuning_dataset.py to create the dataset")
        return False

def create_directories():
    """Create necessary directories."""
    print_step("6", "Creating Directories")
    
    directories = [
        "models",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ All directories created")

def display_next_steps():
    """Display next steps for the user."""
    print_step("7", "Next Steps")
    
    print("🎯 Your environment is ready! Here's what you can do next:")
    print()
    print("1. 🏃‍♂️ Run the fine-tuning script:")
    print("   python finetune_llama_unsloth.py")
    print()
    print("2. 🔧 Customize training parameters:")
    print("   python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct")
    print()
    print("3. 🧪 Test an existing model:")
    print("   python finetune_llama_unsloth.py --test-only")
    print()
    print("4. 📊 Monitor training:")
    print("   tail -f training.log")
    print()
    print("5. 🎮 Check GPU usage during training:")
    print("   watch -n 1 nvidia-smi")

def main():
    """Main setup function."""
    print_header("Spiritual Wisdom Fine-tuning Environment Setup")
    
    print("🧘‍♂️ Welcome to the Spiritual Wisdom AI Fine-tuning Setup!")
    print("This script will prepare your environment for training an AI spiritual teacher.")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Display system info
    check_system_info()
    
    # Step 3: Install dependencies
    print("\n🤔 Do you want to install/update dependencies? (y/n): ", end="")
    if input().lower().startswith('y'):
        if not install_dependencies():
            print("❌ Setup failed during dependency installation")
            sys.exit(1)
    
    # Step 4: Verify installation
    if not verify_installation():
        print("❌ Some packages are missing. Please check the installation.")
        sys.exit(1)
    
    # Step 5: Check dataset
    dataset_ok = check_dataset()
    
    # Step 6: Create directories
    create_directories()
    
    # Step 7: Display next steps
    display_next_steps()
    
    print_header("Setup Complete! 🎉")
    
    if dataset_ok:
        print("🌟 Everything is ready! You can now start fine-tuning your spiritual AI teacher.")
    else:
        print("⚠️  Setup complete, but dataset is missing. Generate it first before training.")
    
    print("🧘‍♂️ May your AI journey bring wisdom and enlightenment! ✨")

if __name__ == "__main__":
    main() 