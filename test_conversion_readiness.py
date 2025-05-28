#!/usr/bin/env python3
"""
🔍 Test Conversion Readiness
============================

Check if your local environment is ready for GGUF conversion.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_dependencies():
    """Check if required Python packages are available."""
    required_packages = ['torch', 'transformers', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing.append(package)
            logger.error(f"❌ {package} is missing")
    
    return len(missing) == 0, missing


def check_git():
    """Check if git is available for cloning llama.cpp."""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Git is available: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Git is not working properly")
            return False
    except FileNotFoundError:
        logger.error("❌ Git is not installed")
        return False


def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Ollama is available: {result.stdout.strip()}")
            
            # Check if Ollama service is running
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Ollama service is running")
                    return True
                else:
                    logger.warning("⚠️  Ollama is installed but service may not be running")
                    return True
            except:
                logger.warning("⚠️  Could not check Ollama service status")
                return True
        else:
            logger.error("❌ Ollama is not working properly")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama is not installed")
        return False


def check_disk_space():
    """Check available disk space for conversion."""
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (1024**3)
        
        if free_gb >= 10:
            logger.info(f"✅ Sufficient disk space: {free_gb}GB available")
            return True
        else:
            logger.warning(f"⚠️  Low disk space: {free_gb}GB available (recommend 10GB+)")
            return False
    except Exception as e:
        logger.error(f"❌ Could not check disk space: {e}")
        return False


def simulate_model_check():
    """Simulate checking for model files."""
    model_path = Path("models/spiritual-wisdom-llama")
    
    if model_path.exists() and any(model_path.iterdir()):
        logger.info("✅ Model files found")
        return True
    else:
        logger.info("📥 Model files not found locally (expected - they're on RunPod)")
        logger.info("   You'll need to download them first:")
        logger.info("   1. On RunPod: tar -czf spiritual-model.tar.gz models/spiritual-wisdom-llama/")
        logger.info("   2. Download the tar.gz file")
        logger.info("   3. Extract: tar -xzf spiritual-model.tar.gz")
        return False


def main():
    """Run all readiness checks."""
    logger.info("🔍 Checking GGUF Conversion Readiness")
    logger.info("=" * 50)
    
    checks = [
        ("Python Dependencies", check_python_dependencies),
        ("Git", check_git),
        ("Ollama", check_ollama),
        ("Disk Space", check_disk_space),
        ("Model Files", simulate_model_check)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n🔍 Checking {name}...")
        try:
            if name == "Python Dependencies":
                success, missing = check_func()
                results.append((name, success))
                if not success:
                    logger.info(f"   Install missing: pip install {' '.join(missing)}")
            else:
                success = check_func()
                results.append((name, success))
        except Exception as e:
            logger.error(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 READINESS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✅ READY" if success else "❌ NEEDS ATTENTION"
        logger.info(f"{name:20} {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    if passed >= 4:  # Model files are expected to be missing
        logger.info("🎉 Your environment is ready for GGUF conversion!")
        logger.info("📥 Next step: Download your model files from RunPod")
    else:
        logger.info("⚠️  Please address the issues above before proceeding")
    
    return passed >= 4


if __name__ == "__main__":
    main() 