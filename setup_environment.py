#!/usr/bin/env python3
"""
🛠️ Environment Setup Script for Spiritual Wisdom Fine-tuning
============================================================

This script helps set up the environment for Unsloth + Llama 3.2 fine-tuning.
It creates a virtual environment, installs dependencies, and verifies the setup.

Usage: python3 setup_environment.py
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import importlib.util
import venv

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

def create_virtual_environment():
    """Create and setup virtual environment."""
    print_step("2", "Creating Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print(f"📁 Virtual environment already exists at: {venv_path}")
        print("🤔 Do you want to recreate it? (y/n): ", end="")
        if input().lower().startswith('y'):
            print("🗑️  Removing existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    try:
        print(f"🏗️  Creating virtual environment at: {venv_path}")
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        
        # Display activation instructions
        if platform.system() == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"\n💡 To activate the virtual environment manually, run:")
        print(f"   {activate_cmd}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get the Python executable path for the virtual environment."""
    venv_path = Path("venv")
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    return str(python_exe)

def check_system_info():
    """Display system information."""
    print_step("3", "System Information")
    
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"🧠 CPU: {platform.processor()}")
    
    # Check for GPU acceleration (if PyTorch is available)
    try:
        import torch
        
        if platform.system() == "Darwin":
            # macOS - check for MPS (Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 MPS (Metal Performance Shaders) available for Apple Silicon")
                print("💡 Training will use MPS acceleration (good performance)")
            else:
                print("⚠️  MPS not available. Training will use CPU (slower)")
        else:
            # Linux/Windows - check for CUDA
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
        if platform.system() == "Darwin":
            print("📦 PyTorch not installed yet (will support MPS on Apple Silicon)")
        else:
            print("📦 PyTorch not installed yet (will be installed with CUDA support)")

def install_dependencies():
    """Install required dependencies in virtual environment."""
    print_step("4", "Installing Dependencies in Virtual Environment")
    
    # Check if requirements.txt exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Get virtual environment Python
    venv_python = get_venv_python()
    
    if not Path(venv_python).exists():
        print("❌ Virtual environment Python not found. Please create venv first.")
        return False
    
    print("📦 Installing packages in virtual environment...")
    
    try:
        # Upgrade pip first
        print("⬆️  Upgrading pip...")
        subprocess.run([
            venv_python, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Step 1: Install PyTorch based on platform
        print("🔥 Installing PyTorch...")
        if platform.system() == "Darwin":
            # macOS - use default PyTorch (supports MPS for Apple Silicon)
            print("🍎 Detected macOS - installing PyTorch with MPS support...")
            subprocess.run([
                venv_python, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ], check=True)
        else:
            # Linux/Windows - try CUDA version
            print("🐧 Detected Linux/Windows - installing PyTorch with CUDA support...")
            subprocess.run([
                venv_python, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
        
        # Step 2: Install core dependencies (excluding problematic ones)
        print("📚 Installing core dependencies...")
        core_packages = [
            "transformers>=4.36.0",
            "datasets>=2.14.0", 
            "accelerate>=0.24.0",
            "tokenizers>=0.15.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0"
        ]
        
        for package in core_packages:
            print(f"   Installing {package.split('>=')[0]}...")
            subprocess.run([
                venv_python, "-m", "pip", "install", package
            ], check=True)
        
        # Step 2.5: Install sentencepiece (optional, can fail on some systems)
        print("🔤 Installing sentencepiece (optional)...")
        try:
            subprocess.run([
                venv_python, "-m", "pip", "install", "sentencepiece>=0.1.99"
            ], check=True, timeout=180)  # 3 minute timeout
            print("✅ sentencepiece installed successfully")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  sentencepiece installation failed (optional package)")
            print("   This is common on macOS without build tools")
            print("   The training will work without it")
        
        # Step 3: Install PEFT and TRL (require PyTorch)
        print("🎯 Installing PEFT and TRL...")
        subprocess.run([
            venv_python, "-m", "pip", "install", "peft>=0.7.0", "trl>=0.7.0"
        ], check=True)
        
        # Step 4: Install optional packages
        print("🎨 Installing optional packages...")
        optional_packages = [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0", 
            "jupyter>=1.0.0",
            "scikit-learn>=1.3.0"
        ]
        
        for package in optional_packages:
            try:
                print(f"   Installing {package.split('>=')[0]}...")
                subprocess.run([
                    venv_python, "-m", "pip", "install", package
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"   ⚠️  Skipped {package.split('>=')[0]} (optional)")
        
        # Step 5: Install Unsloth (most problematic, install last)
        print("🚀 Installing Unsloth (this may take a while)...")
        try:
            # Try the standard installation first
            subprocess.run([
                venv_python, "-m", "pip", "install", "unsloth[colab-new]"
            ], check=True, timeout=300)  # 5 minute timeout
            print("✅ Unsloth installed successfully")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  Standard Unsloth installation failed, trying alternative...")
            try:
                # Try without the colab-new extra
                subprocess.run([
                    venv_python, "-m", "pip", "install", "unsloth"
                ], check=True, timeout=300)
                print("✅ Unsloth (basic) installed successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print("❌ Unsloth installation failed")
                print("💡 You can continue without Unsloth, but training will be slower")
                print("   Try installing manually later: pip install unsloth")
        
        # Step 6: Try to install bitsandbytes (platform dependent)
        print("🔧 Installing bitsandbytes...")
        try:
            subprocess.run([
                venv_python, "-m", "pip", "install", "bitsandbytes>=0.41.0"
            ], check=True)
            print("✅ bitsandbytes installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  bitsandbytes installation failed (may not be available on this platform)")
            print("   Training will work without it, but may use more memory")
        
        print("✅ Dependencies installed successfully in virtual environment")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Try running the setup script again")
        print("   - Check your internet connection")
        print("   - On Apple Silicon, some packages may need Xcode command line tools")
        print("   - You can install missing packages manually after activation")
        return False

def verify_installation():
    """Verify that key packages are installed correctly in virtual environment."""
    print_step("5", "Verifying Installation in Virtual Environment")
    
    venv_python = get_venv_python()
    
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
            result = subprocess.run([
                venv_python, "-c", f"import {package}; print('✅ {name} installed')"
            ], capture_output=True, text=True, check=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError:
            print(f"❌ {name} not found")
            all_good = False
    
    # Special check for GPU acceleration in virtual environment
    try:
        if platform.system() == "Darwin":
            # Check for MPS on macOS
            result = subprocess.run([
                venv_python, "-c", 
                "import torch; print('✅ MPS acceleration available' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else '⚠️  MPS not available - using CPU')"
            ], capture_output=True, text=True, check=True)
            print(result.stdout.strip())
        else:
            # Check for CUDA on Linux/Windows
            result = subprocess.run([
                venv_python, "-c", 
                "import torch; print('✅ CUDA support available' if torch.cuda.is_available() else '⚠️  CUDA not available')"
            ], capture_output=True, text=True, check=True)
            print(result.stdout.strip())
    except subprocess.CalledProcessError:
        print("❌ PyTorch not properly installed")
        all_good = False
    
    return all_good

def check_dataset():
    """Check if the dataset is available."""
    print_step("6", "Checking Dataset")
    
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
    print_step("7", "Creating Directories")
    
    directories = [
        "models",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ All directories created")

def create_activation_scripts():
    """Create convenient activation scripts."""
    print_step("8", "Creating Activation Scripts")
    
    # Create activation script for different platforms
    if platform.system() == "Windows":
        # Windows batch file
        activate_script = Path("activate_venv.bat")
        with open(activate_script, 'w') as f:
            f.write("@echo off\n")
            f.write("echo 🧘‍♂️ Activating Spiritual Wisdom AI Environment...\n")
            f.write("call venv\\Scripts\\activate\n")
            f.write("echo ✅ Virtual environment activated!\n")
            f.write("echo 🚀 Ready to train your AI spiritual teacher!\n")
            f.write("echo.\n")
            f.write("echo Quick commands:\n")
            f.write("echo   python finetune_llama_unsloth.py    - Start training\n")
            f.write("echo   python test_dataset.py             - Test dataset\n")
            f.write("echo   deactivate                         - Exit environment\n")
        
        print(f"📝 Created Windows activation script: {activate_script}")
        print(f"   Run: {activate_script}")
        
    else:
        # Unix/Linux/macOS shell script
        activate_script = Path("activate_venv.sh")
        with open(activate_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("echo '🧘‍♂️ Activating Spiritual Wisdom AI Environment...'\n")
            f.write("source venv/bin/activate\n")
            f.write("echo '✅ Virtual environment activated!'\n")
            f.write("echo '🚀 Ready to train your AI spiritual teacher!'\n")
            f.write("echo ''\n")
            f.write("echo 'Quick commands:'\n")
            f.write("echo '  python finetune_llama_unsloth.py    - Start training'\n")
            f.write("echo '  python test_dataset.py             - Test dataset'\n")
            f.write("echo '  deactivate                         - Exit environment'\n")
            f.write("echo ''\n")
            f.write("# Keep shell open in activated environment\n")
            f.write("exec $SHELL\n")
        
        # Make executable
        activate_script.chmod(0o755)
        
        print(f"📝 Created Unix activation script: {activate_script}")
        print(f"   Run: ./{activate_script}")

def display_next_steps():
    """Display next steps for the user."""
    print_step("9", "Next Steps")
    
    print("🎯 Your virtual environment is ready! Here's what you can do next:")
    print()
    
    if platform.system() == "Windows":
        print("1. 🏃‍♂️ Activate the environment and start training:")
        print("   activate_venv.bat")
        print("   python finetune_llama_unsloth.py")
        print()
        print("2. 🔧 Or activate manually:")
        print("   venv\\Scripts\\activate")
    else:
        print("1. 🏃‍♂️ Activate the environment and start training:")
        print("   ./activate_venv.sh")
        print("   python finetune_llama_unsloth.py")
        print()
        print("2. 🔧 Or activate manually:")
        print("   source venv/bin/activate")
    
    print()
    print("3. 🧪 Test dataset (in activated environment):")
    print("   python test_dataset.py")
    print()
    print("4. 📊 Monitor training (in activated environment):")
    print("   tail -f training.log")
    print()
    print("5. 🎮 Check GPU usage:")
    print("   watch -n 1 nvidia-smi")
    print()
    print("6. 🚪 Exit virtual environment:")
    print("   deactivate")

def main():
    """Main setup function."""
    print_header("Spiritual Wisdom Fine-tuning Environment Setup")
    
    print("🧘‍♂️ Welcome to the Spiritual Wisdom AI Fine-tuning Setup!")
    print("This script will create a virtual environment and prepare everything for training.")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Step 3: Display system info
    check_system_info()
    
    # Step 4: Install dependencies
    print("\n🤔 Do you want to install/update dependencies in the virtual environment? (y/n): ", end="")
    if input().lower().startswith('y'):
        if not install_dependencies():
            print("❌ Setup failed during dependency installation")
            sys.exit(1)
    
    # Step 5: Verify installation
    if not verify_installation():
        print("❌ Some packages are missing. Please check the installation.")
        sys.exit(1)
    
    # Step 6: Check dataset
    dataset_ok = check_dataset()
    
    # Step 7: Create directories
    create_directories()
    
    # Step 8: Create activation scripts
    create_activation_scripts()
    
    # Step 9: Display next steps
    display_next_steps()
    
    print_header("Setup Complete! 🎉")
    
    if dataset_ok:
        print("🌟 Everything is ready! Your virtual environment contains all dependencies.")
        print("🧘‍♂️ Activate the environment and start training your spiritual AI teacher!")
    else:
        print("⚠️  Setup complete, but dataset is missing. Generate it first before training.")
    
    print("\n💡 Remember: Always activate the virtual environment before running scripts!")
    print("🧘‍♂️ May your AI journey bring wisdom and enlightenment! ✨")

if __name__ == "__main__":
    main() 