# 🧘‍♂️ Spiritual Wisdom AI Fine-tuning Project

Transform ancient wisdom into modern AI guidance using **Unsloth + Llama 3.2** for lightning-fast fine-tuning in a clean virtual environment!

## 🌟 What This Project Does

This project fine-tunes **Llama 3.2** to become a wise spiritual teacher that can:

- 🧠 Answer deep philosophical and spiritual questions
- 🧘‍♀️ Provide meditation guidance and mindfulness advice  
- 💡 Share insights on consciousness, life purpose, and personal growth
- 🌱 Offer wisdom from various spiritual traditions
- 🤝 Maintain a compassionate, non-judgmental teaching style

## 🚀 Quick Start (5 Minutes)

### 1. Setup Virtual Environment
```bash
# Create isolated environment and install everything
python3 setup_environment.py
```

### 2. Activate & Train
```bash
# On macOS/Linux
./activate_venv.sh

# On Windows  
activate_venv.bat

# Start training (in activated environment)
python finetune_llama_unsloth.py
```

**That's it!** 🎉 Your AI will start learning from 520+ spiritual wisdom examples in a clean, isolated environment.

## 📊 Project Highlights

- **🚀 2x Faster Training**: Powered by Unsloth optimizations
- **💾 Memory Efficient**: 4-bit quantization + QLoRA
- **📚 Rich Dataset**: 520 carefully curated spiritual wisdom entries
- **🎯 Production Ready**: Complete pipeline from training to inference
- **🔧 Highly Configurable**: Easy to customize and extend
- **🏠 Virtual Environment**: Clean, isolated dependencies

## 🎯 Technical Specifications

| Component | Details |
|-----------|---------|
| **Model** | `unsloth/Llama-3.2-3B-Instruct` (optimized) |
| **Dataset** | 520 spiritual wisdom Q&A pairs |
| **Training** | QLoRA with rank-16, 3 epochs |
| **Memory** | 6-8GB VRAM (3B model) or 4-5GB (1B model) |
| **Time** | ~2-3 hours on RTX 4070 |
| **Environment** | Isolated virtual environment |
| **Output** | Fine-tuned model ready for spiritual guidance |

## 📁 Project Structure

```
guru/
├── 🚀 finetune_llama_unsloth.py    # Main training script
├── 🛠️ setup_environment.py         # Automated environment setup
├── 🧪 test_dataset.py              # Dataset verification
├── 📊 datasets/
│   ├── llama_optimized.jsonl       # 520 spiritual wisdom examples
│   ├── alpaca_format.jsonl         # Alternative format
│   └── conversation_format.jsonl   # Chat format
├── 🤖 models/                      # Trained models (created during training)
├── 📝 logs/                        # Training logs
├── 📋 requirements.txt             # All dependencies
├── 🏠 venv/                        # Virtual environment (auto-created)
├── 🔧 activate_venv.sh/.bat        # Easy activation scripts
├── 📖 QUICK_START.md              # Detailed setup guide
└── 🗺️ unsloth_llama_finetuning_plan.md  # Complete implementation plan
```

## 🔧 Advanced Usage

### Custom Model Selection (in activated environment)
```bash
# Use smaller model for limited VRAM
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct

# Custom output directory
python finetune_llama_unsloth.py --output models/my-spiritual-guru
```

### Testing and Evaluation (in activated environment)
```bash
# Test dataset format
python test_dataset.py

# Test existing model
python finetune_llama_unsloth.py --test-only
```

### Monitoring Training (in activated environment)
```bash
# Watch training progress
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Virtual Environment Management
```bash
# Check installed packages
pip list

# Update dependencies
pip install --upgrade transformers

# Reset environment completely
rm -rf venv/
python3 setup_environment.py
```

## 📈 Expected Results

After training, your AI spiritual teacher will respond like this:

**Question**: "What is the nature of consciousness?"

**AI Response**: "Consciousness is the fundamental awareness that underlies all experience. It is not something you have, but something you are. Like the sky that remains unchanged whether clouds pass through it or not, consciousness is the unchanging backdrop against which all thoughts, emotions, and sensations arise and pass away. To understand consciousness, you must first recognize that you are not your thoughts, but the awareness that observes them..."

## 🛠️ System Requirements

### Minimum Requirements
- **Python 3.8+** (Python 3.10+ recommended)
- **4GB VRAM** (for 1B model)
- **8GB System RAM**
- **CUDA-compatible GPU**

### Recommended Setup
- **6GB+ VRAM** (RTX 3080/4070+ for 3B model)
- **16GB+ System RAM**
- **Fast SSD** for dataset loading
- **CUDA 12.1+**

## 🏠 Virtual Environment Benefits

Using a virtual environment provides:
- **🔒 Isolation**: No conflicts with other Python projects
- **🧹 Clean State**: Fresh environment every time
- **📦 Reproducibility**: Exact dependency versions
- **🚀 Easy Reset**: Delete `venv/` folder to start over
- **🤝 Sharing**: Portable setup for team collaboration

## 🎓 Learning Resources

- **📖 [Quick Start Guide](QUICK_START.md)** - Step-by-step setup
- **🗺️ [Implementation Plan](unsloth_llama_finetuning_plan.md)** - Technical details
- **🔗 [Unsloth Documentation](https://github.com/unslothai/unsloth)** - Framework docs
- **🤗 [Model Repository](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)** - Pre-optimized model

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🔍 Improve Dataset**: Add more spiritual wisdom examples
2. **⚡ Optimize Training**: Experiment with hyperparameters
3. **🧪 Add Tests**: Create evaluation benchmarks
4. **📚 Documentation**: Improve guides and examples
5. **🐛 Bug Reports**: Report issues and suggest fixes

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Unsloth Team** for the incredible optimization framework
- **Meta AI** for the Llama 3.2 foundation model
- **Hugging Face** for the transformers ecosystem
- **Spiritual Teachers** whose wisdom forms our dataset

## 🧘‍♂️ Philosophy

> "The best teacher is not the one who knows the most, but the one who awakens the wisdom that already exists within the student."

This AI is designed to:
- **Guide** rather than dictate
- **Inspire** contemplation and self-discovery
- **Respect** diverse spiritual paths and beliefs
- **Encourage** inner exploration and growth
- **Share** timeless wisdom with modern accessibility

---

**Ready to create your AI spiritual teacher?** 

Start with: `python3 setup_environment.py` 🌟

*May this AI bring wisdom, peace, and enlightenment to all who seek guidance.* ✨ 