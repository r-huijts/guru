# 🚀 Quick Start Guide: Spiritual Wisdom AI Fine-tuning

Welcome to your spiritual AI teacher training journey! This guide will get you up and running with Llama 3.2 fine-tuning in just a few steps using a clean virtual environment.

## 🎯 What You'll Build

A fine-tuned Llama 3.2 model that can:
- Answer spiritual and philosophical questions with wisdom
- Provide meditation guidance and life advice
- Share insights on consciousness, mindfulness, and personal growth
- Maintain a consistent, wise teaching style

## 📋 Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **6GB+ VRAM** for Llama-3.2-3B (or 4GB+ for 1B model)
- **Apple Silicon Mac** with MPS support (or CUDA GPU for other systems)
- **16GB+ System RAM** recommended

## 🚀 Quick Setup (5 Minutes)

### Step 1: Create Virtual Environment & Install Dependencies
```bash
# Run the automated setup (creates venv and installs everything)
python3 setup_environment.py
```

### Step 2: Activate Environment
```bash
# Easy activation with helpful info
./activate_venv.sh

# Or manually:
source venv/bin/activate
```

### Step 3: Verify Setup
```bash
# Test the dataset and environment
python test_dataset.py
```

### Step 4: Start Fine-tuning
```bash
# Begin training (will use MPS on Apple Silicon)
python finetune_llama_unsloth.py

# Or with custom options:
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct --max-length 1024
```

## 🎛️ Training Options

```bash
# Show all available options
python finetune_llama_unsloth.py --help

# Common configurations:
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct  # Smaller model (4GB VRAM)
python finetune_llama_unsloth.py --max-length 1024                      # Shorter sequences
python finetune_llama_unsloth.py --output ./my_spiritual_model           # Custom output path
```

## 📊 What to Expect

- **Dataset**: 520 high-quality spiritual wisdom examples
- **Training Time**: 30-60 minutes on Apple Silicon
- **Memory Usage**: ~6GB VRAM for 3B model, ~4GB for 1B model
- **Output**: Fine-tuned model ready for spiritual guidance

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# If setup fails, try manual creation:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Memory Issues
```bash
# Use smaller model:
python finetune_llama_unsloth.py --model unsloth/Llama-3.2-1B-Instruct

# Reduce sequence length:
python finetune_llama_unsloth.py --max-length 1024
```

### Package Issues
```bash
# Reinstall problematic packages:
pip install --force-reinstall transformers datasets
```

## 🧘‍♂️ Next Steps

1. **Test Your Model**: The script will automatically test the model after training
2. **Experiment**: Try different prompts and see how your AI teacher responds
3. **Iterate**: Adjust training parameters and retrain if needed
4. **Deploy**: Use your fine-tuned model in applications or chatbots

## 📁 Project Structure

```
guru/
├── venv/                          # Virtual environment
├── datasets/llama_optimized.jsonl # Training dataset (520 entries)
├── models/                        # Output directory
├── logs/                         # Training logs
├── finetune_llama_unsloth.py     # Main training script
├── test_dataset.py               # Dataset verification
├── setup_environment.py          # Environment setup
├── activate_venv.sh              # Easy activation
└── requirements.txt              # Dependencies
```

## 💡 Tips for Success

- **Start Small**: Use the 1B model first to test everything works
- **Monitor Memory**: Watch Activity Monitor during training
- **Be Patient**: First run downloads the model (~6GB)
- **Experiment**: Try different spiritual questions after training

Happy training! 🧘‍♂️✨

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

## 💡 Virtual Environment Tips

### Daily Workflow
```bash
# Start your session
./activate_venv.sh                    # Activate environment
python finetune_llama_unsloth.py     # Train your model
python test_dataset.py               # Test dataset
deactivate                           # Exit when done
```

### Managing the Environment
```bash
# Check what's installed
pip list

# Update a package
pip install --upgrade transformers

# Completely reset environment
rm -rf venv/
python3 setup_environment.py
```

---

**Ready to begin?** Run `python3 setup_environment.py` and start your journey! 🌟

*May your AI teacher bring wisdom, peace, and enlightenment to all who seek guidance.* ✨ 

## 📋 Prerequisites

- Python 3.8+
- 6GB+ GPU memory (recommended)
- Basic familiarity with command line

## 🛠️ Setup (One-time)

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 2. Run Setup Script
```bash
python setup_environment.py
```

This will:
- Install all required dependencies
- Download and prepare the spiritual wisdom dataset
- Set up the project structure

## 🎯 Training Your Spiritual AI

### Basic Training
```bash
# Activate virtual environment first
source venv/bin/activate

# Start training with evaluation monitoring
python finetune_llama_unsloth.py
```

### Advanced Training Options
```bash
# Custom model and dataset
python finetune_llama_unsloth.py \
    --model "unsloth/Llama-3.2-1B-Instruct" \
    --dataset "datasets/custom_wisdom.jsonl" \
    --output "models/my-spiritual-ai"

# Enable early stopping to prevent overfitting
python finetune_llama_unsloth.py --early-stopping

# Custom sequence length for longer conversations
python finetune_llama_unsloth.py --max-length 4096
```

## 📊 Evaluation and Monitoring

The training now includes comprehensive evaluation monitoring:

### Real-time Evaluation
- **Evaluation every 25 steps**: Monitor model performance during training
- **Best model tracking**: Automatically saves the best performing checkpoint
- **Loss trend analysis**: See if your model is improving, stable, or overfitting
- **Custom metrics**: Detailed logging of training progress

### What You'll See During Training
```
📊 Step 25 Evaluation:
   🧘‍♂️ Eval Loss: 2.1234 🎉 NEW BEST!
   📈 Train Loss: 2.3456
   🎓 Learning Rate: 1.50e-04
   📊 Trend: 📉 Improving

📊 Step 50 Evaluation:
   🧘‍♂️ Eval Loss: 2.0987 🎉 NEW BEST!
   📈 Train Loss: 2.2134
   🎓 Learning Rate: 1.40e-04
   📊 Trend: 📉 Improving
```

### Post-Training Analysis
After training completes, you'll get:
- **Evaluation summary**: Complete training metrics saved to `evaluation_summary.json`
- **Best model**: Automatically loaded and saved
- **Improvement analysis**: How much your model improved during training

## 🧪 Testing Your Model

### Quick Test
```bash
# Test the trained model
python finetune_llama_unsloth.py --test-only
```

### Custom Test Questions
Edit the `test_prompts` in the script to ask your own spiritual questions!

## 📁 Output Structure

After training, you'll have:
```
models/spiritual-wisdom-llama/
├── pytorch_model.bin          # Your fine-tuned model
├── tokenizer.json            # Tokenizer files
├── config.json               # Model configuration
├── evaluation_summary.json   # Training metrics and analysis
└── logs/                     # Detailed training logs
```

## 🎛️ Training Configuration

The script automatically optimizes for:
- **Memory efficiency**: 4-bit quantization + LoRA
- **Speed**: Unsloth 2x faster training
- **Quality**: Proper evaluation and checkpointing
- **Monitoring**: Real-time loss tracking and trend analysis

### Key Settings
- **Batch size**: 2 per device (8 effective with gradient accumulation)
- **Learning rate**: 2e-4 (optimized for spiritual content)
- **Evaluation**: Every 25 steps with automatic best model selection
- **Checkpoints**: Saved every 50 steps (keep 5 best)
- **Early stopping**: Optional (use `--early-stopping` flag)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Use smaller model: `--model "unsloth/Llama-3.2-1B-Instruct"`
   - Reduce sequence length: `--max-length 1024`

2. **Slow training on CPU**
   - Consider using Google Colab or RunPod for GPU access
   - The script will warn you about CPU training

3. **Poor model responses**
   - Check evaluation metrics in `evaluation_summary.json`
   - Consider training for more epochs or adjusting learning rate
   - Enable early stopping to prevent overfitting

4. **Import errors**
   - Rerun: `python setup_environment.py`
   - Activate virtual environment: `source venv/bin/activate`

## 🎉 Success Indicators

Your training is going well if you see:
- ✅ Decreasing evaluation loss over time
- ✅ "📉 Improving" trend indicators
- ✅ Regular "🎉 NEW BEST!" messages
- ✅ Final improvement percentage > 10%

## 🔄 Next Steps

1. **Test thoroughly**: Try various spiritual questions
2. **Iterate**: Adjust hyperparameters based on evaluation metrics
3. **Expand dataset**: Add more spiritual wisdom examples
4. **Deploy**: Use your fine-tuned model in applications

## 💡 Pro Tips

- **Monitor evaluation loss**: If it stops improving, consider early stopping
- **Check trends**: Consistent "📈 Increasing" trends may indicate overfitting
- **Save checkpoints**: The best model is automatically saved for you
- **Use evaluation data**: The 10% held-out data gives unbiased performance estimates

Happy training! 🧘‍♂️✨ 