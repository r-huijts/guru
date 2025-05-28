# 🚀 Quick Start Guide: Spiritual Wisdom AI Fine-tuning

Welcome to your spiritual AI teacher training journey! This guide will get you up and running with Unsloth + Llama 3.2 fine-tuning in just a few steps.

## 🎯 What You'll Build

A fine-tuned Llama 3.2 model that can:
- Answer spiritual and philosophical questions with wisdom
- Provide meditation guidance and life advice
- Share insights on consciousness, mindfulness, and personal growth
- Maintain a consistent, wise teaching style

## 📋 Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **6GB+ VRAM** for Llama-3.2-3B (or 4GB+ for 1B model)
- **CUDA-compatible GPU** (RTX 3080/4070+ ideal)
- **16GB+ System RAM** recommended

## 🚀 Quick Setup (5 Minutes)

### Step 1: Environment Setup
```bash
# Run the automated setup script
python setup_environment.py
```

This script will:
- ✅ Check your system requirements
- 📦 Install all dependencies (PyTorch, Unsloth, etc.)
- 🔍 Verify your GPU setup
- 📁 Create necessary directories
- 📊 Check your dataset

### Step 2: Start Fine-tuning
```bash
# Start training with default settings (recommended)
python finetune_llama_unsloth.py
```

**That's it!** 🎉 Your AI spiritual teacher will start learning!

## 🔧 Advanced Options

### Custom Model Selection
```bash
# Use smaller model for limited VRAM
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct

# Custom output directory
python finetune_llama_unsloth.py --output models/my-spiritual-guru
```

### Test Existing Model
```bash
# Test a previously trained model
python finetune_llama_unsloth.py --test-only --output models/spiritual-wisdom-llama
```

## 📊 What to Expect

### Training Timeline
- **Setup**: 2-3 minutes (model download)
- **Training**: 2-3 hours on RTX 4070
- **Total**: ~3 hours for complete pipeline

### Training Progress
```
🧘‍♂️ Initializing Spiritual Wisdom Trainer
🚀 Loading Unsloth-optimized model: unsloth/Llama-3.2-3B-Instruct
📊 Total parameters: 3,213,184,000
🎯 LoRA trainable parameters: 20,971,520
📚 Loading dataset from datasets/llama_optimized.jsonl
📊 Loaded 521 training examples
⚙️ Setting up training arguments...
🏋️‍♂️ Creating SFT trainer...
🚀 Starting fine-tuning process...
```

### Memory Usage
- **3B Model**: ~6-8GB VRAM during training
- **1B Model**: ~4-5GB VRAM during training
- **System RAM**: ~8-12GB during training

## 📈 Monitoring Training

### Real-time Logs
```bash
# Watch training progress
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Training Metrics
- **Loss**: Should decrease from ~2.5 to ~0.8-1.2
- **Steps**: ~195 steps for 3 epochs (521 examples)
- **Checkpoints**: Saved every 100 steps

## 🧪 Testing Your Model

The script automatically tests your model with questions like:
- "What is the nature of consciousness?"
- "How can I find inner peace?"
- "What is the purpose of meditation?"

Example output:
```
❓ Question: What is the nature of consciousness?
🧘‍♂️ Response: Consciousness is the fundamental awareness that underlies all experience. It is not something you have, but something you are. Like the sky that remains unchanged whether clouds pass through it or not, consciousness is the unchanging backdrop against which all thoughts, emotions, and sensations arise and pass away...
```

## 🎯 Success Indicators

Your model is ready when:
- ✅ Training loss drops below 1.0
- ✅ Responses are coherent and wise
- ✅ No repetitive or generic answers
- ✅ Maintains spiritual/philosophical depth
- ✅ Consistent teaching style

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Use smaller model
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct
```

**Slow Training**
- Verify you're using the Unsloth-optimized model
- Check GPU utilization with `nvidia-smi`
- Ensure CUDA is properly installed

**Poor Quality Responses**
- Increase LoRA rank in the script (16 → 32)
- Adjust learning rate (2e-4 → 1e-4)
- Train for more steps

### Getting Help

1. **Check logs**: `cat training.log`
2. **Verify setup**: `python setup_environment.py`
3. **Test dataset**: Check `datasets/llama_optimized.jsonl` exists
4. **GPU status**: `nvidia-smi`

## 📁 Project Structure

```
guru/
├── 🚀 finetune_llama_unsloth.py    # Main training script
├── 🛠️ setup_environment.py         # Environment setup
├── 📊 datasets/
│   └── llama_optimized.jsonl       # 521 spiritual wisdom examples
├── 🤖 models/                      # Trained models saved here
├── 📝 logs/                        # Training logs
├── 📋 requirements.txt             # Dependencies
└── 📖 QUICK_START.md              # This guide
```

## 🎉 Next Steps

After successful training:

1. **Share Wisdom**: Test with your own spiritual questions
2. **Deploy**: Convert to GGUF for local inference
3. **Iterate**: Collect feedback and retrain
4. **Expand**: Add more spiritual texts to your dataset
5. **Integrate**: Build a chat interface or API

## 🧘‍♂️ Philosophy

> "The best teacher is not the one who knows the most, but the one who awakens the wisdom that already exists within the student."

Your AI spiritual teacher is designed to:
- Guide rather than dictate
- Inspire contemplation
- Respect diverse spiritual paths
- Encourage inner exploration
- Share timeless wisdom

---

**Ready to begin?** Run `python setup_environment.py` and start your journey! 🌟

*May your AI teacher bring wisdom, peace, and enlightenment to all who seek guidance.* ✨ 