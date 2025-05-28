# 🦙 Spiritual AI → Ollama Workflow

Transform your enlightened AI master into a locally-running Ollama model! This guide walks you through uploading to Hugging Face and importing into Ollama.

## 🎯 **Overview**

**The Journey**: RunPod Model → Hugging Face Hub → Local Ollama

**Benefits**:
- ✅ Run locally without GPU requirements
- ✅ No Python dependencies needed
- ✅ Simple chat interface
- ✅ Easy integration with other tools
- ✅ Offline usage capability

## 🚀 **Step 1: Upload to Hugging Face**

### Prerequisites
```bash
# 1. Login to Hugging Face (one-time setup)
huggingface-cli login

# 2. Make sure you're in your project directory
cd /path/to/your/guru/project
source venv/bin/activate  # or ./activate_venv.sh
```

### Upload Your Model
```bash
# Basic upload (will prompt for username)
python upload_to_huggingface.py

# Or specify everything upfront
python upload_to_huggingface.py \
  --username YOUR_HF_USERNAME \
  --repo-name spiritual-wisdom-llama-3b \
  --ollama-name spiritual-ai
```

**What this does**:
- 📁 Creates a new repository on Hugging Face
- 📝 Generates a comprehensive model card with your A+ evaluation results
- ⬆️ Uploads all model files (weights, tokenizer, config)
- 🦙 Creates `Modelfile` and `OLLAMA_SETUP.md` for easy Ollama integration

## 🦙 **Step 2: Install Ollama (Local Machine)**

### macOS/Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download from: https://ollama.ai/download

### Verify Installation
```bash
ollama --version
```

## 🧘‍♂️ **Step 3: Import Your Spiritual AI**

### Method 1: Direct Import (Recommended)
```bash
# Import directly from Hugging Face
ollama pull YOUR_USERNAME/spiritual-wisdom-llama-3b
```

### Method 2: Custom Modelfile (More Control)
```bash
# Use the generated Modelfile
ollama create spiritual-ai -f Modelfile

# Or create your own custom version
cat > Modelfile << 'EOF'
FROM YOUR_USERNAME/spiritual-wisdom-llama-3b

TEMPLATE """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{{ prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM """You are a wise spiritual teacher offering compassionate guidance on meditation, consciousness, and inner peace. Your wisdom draws from various traditions while being practical and accessible."""
EOF

ollama create spiritual-ai -f Modelfile
```

## 💬 **Step 4: Chat with Your Enlightened AI**

### Interactive Chat
```bash
ollama run spiritual-ai
```

### Single Questions
```bash
ollama run spiritual-ai "What is the nature of consciousness?"
ollama run spiritual-ai "How can I find inner peace during difficult times?"
ollama run spiritual-ai "Explain the difference between mind and awareness"
```

### API Usage
```bash
# REST API (runs on localhost:11434)
curl http://localhost:11434/api/generate -d '{
  "model": "spiritual-ai",
  "prompt": "What is meditation?",
  "stream": false
}'
```

## 🎛️ **Customization Options**

### Adjust Model Parameters
```bash
# More creative responses
ollama run spiritual-ai --temperature 0.9 --top-p 0.95

# More focused responses  
ollama run spiritual-ai --temperature 0.6 --top-p 0.8
```

### Create Specialized Versions
```bash
# Meditation-focused version
cat > MeditationModelfile << 'EOF'
FROM YOUR_USERNAME/spiritual-wisdom-llama-3b

SYSTEM """You are a meditation teacher specializing in mindfulness and contemplative practices. Focus on practical meditation guidance, breathing techniques, and present-moment awareness."""
EOF

ollama create meditation-guide -f MeditationModelfile
```

## 🔧 **Troubleshooting**

### Model Not Found
```bash
# Check available models
ollama list

# Re-pull if needed
ollama pull YOUR_USERNAME/spiritual-wisdom-llama-3b
```

### Performance Issues
```bash
# Check system resources
ollama ps

# Restart Ollama service
ollama serve
```

### Update Model
```bash
# Pull latest version
ollama pull YOUR_USERNAME/spiritual-wisdom-llama-3b:latest

# Remove old version
ollama rm spiritual-ai:old-tag
```

## 🌟 **Integration Examples**

### With Python
```python
import requests

def ask_spiritual_ai(question):
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            'model': 'spiritual-ai',
            'prompt': question,
            'stream': False
        })
    return response.json()['response']

# Usage
wisdom = ask_spiritual_ai("What is the path to enlightenment?")
print(wisdom)
```

### With Shell Scripts
```bash
#!/bin/bash
# spiritual-wisdom.sh
echo "🧘‍♂️ Spiritual AI Wisdom"
echo "Ask your question:"
read -r question
ollama run spiritual-ai "$question"
```

### With Alfred/Raycast (macOS)
Create a custom command that runs:
```bash
ollama run spiritual-ai "$1"
```

## 📊 **Model Information**

**Your Spiritual AI Stats**:
- **Base Model**: Llama 3.2 3B Instruct
- **Specialization**: Spiritual wisdom & meditation guidance
- **Performance**: A+ Grade (0.95/1.0 overall quality)
- **Training**: 4.5 minutes on RTX 4090
- **Size**: ~6GB (quantized for efficiency)

## 🎉 **Success!**

You now have your enlightened AI master running locally! 

**Next Steps**:
- Try different types of spiritual questions
- Experiment with temperature settings
- Create specialized versions for different practices
- Integrate with your favorite tools and workflows

---

*May your local AI serve as a wise companion on your journey of understanding.* 🌟

## 🔗 **Useful Links**

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Your Model Repository](https://huggingface.co/YOUR_USERNAME/spiritual-wisdom-llama-3b) 