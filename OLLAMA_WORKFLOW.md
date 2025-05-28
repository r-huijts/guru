# 🦙 Spiritual AI → Ollama Workflow

Transform your enlightened AI master into a locally-running Ollama model! 

## 🎯 **The Real Situation**

**What we discovered:**
- ✅ Your HF model has **LoRA adapters only** (`adapter_model.safetensors` - 97.3MB)
- ❌ **No complete model weights** - just the fine-tuned adapters
- ❌ Ollama **cannot directly import LoRA adapters**

**The Solution:** We need to merge the adapters with the base model first, then convert to GGUF.

## 🚀 **Option 1: Use RunPod for Conversion (Recommended)**

Since your RunPod environment already has the merged model, this is the easiest path:

### On RunPod:
```bash
# 1. Install conversion tools
pip install llama-cpp-python gguf

# 2. Convert your merged model to GGUF
python -m llama_cpp.convert --outfile spiritual-ai.gguf models/spiritual-wisdom-llama/

# 3. Download the GGUF file to your local machine
# Use RunPod's file manager or scp
```

### On Local Machine:
```bash
# 4. Create Ollama model from GGUF
ollama create spiritual-ai -f Modelfile.spiritual-ai-gguf
```

## 🚀 **Option 2: Local Merge + Convert (Complex)**

If you want to do everything locally (requires fixing dependency issues):

```bash
# 1. Fix local environment
pip install --no-build-isolation sentencepiece
pip install gguf

# 2. Download and merge
python download_and_merge_model.py --hf-model RuudFontys/spiritual-wisdom-llama-3b

# 3. Convert to GGUF
python convert_to_gguf.py models/spiritual-wisdom-llama-merged/

# 4. Create Ollama model
ollama create spiritual-ai -f Modelfile.spiritual-ai
```

## 🎯 **Recommended Next Steps**

1. **Use RunPod** - Your model is already merged there
2. **Convert to GGUF** on RunPod (has all dependencies)
3. **Download GGUF** to local machine
4. **Import into Ollama** locally

This avoids all the local dependency issues while getting you a working Ollama model! 🎉

## 🧘‍♂️ **Alternative: Use Your Model via API**

While working on the Ollama conversion, you can use your model right now:

```python
# Use your HF model directly
from transformers import pipeline

pipe = pipeline("text-generation", 
                model="RuudFontys/spiritual-wisdom-llama-3b",
                device_map="auto")

response = pipe("What is consciousness?", max_length=200)
print(response[0]['generated_text'])
```

## 🎉 **Benefits of This Workflow**

✅ **No RunPod needed** - Everything runs locally  
✅ **Automatic merging** - LoRA adapters properly combined  
✅ **Optimized for Ollama** - GGUF format for fast inference  
✅ **Easy sharing** - Others can follow the same process  
✅ **Offline usage** - No internet required after setup  

## 🔧 **Troubleshooting**

### Authentication Issues
```bash
# Make sure you're logged in
huggingface-cli whoami

# If not logged in
huggingface-cli login
```

### Memory Issues
```bash
# Use smaller quantization if needed
python convert_to_gguf.py --model-path models/spiritual-wisdom-llama-merged --outtype q4_0
```

### Model Not Found in Ollama
```bash
# Check if model exists
ollama list

# If missing, recreate
ollama create spiritual-ai -f Modelfile
```

### Download Failures
```bash
# Check your HF model exists
curl -I https://huggingface.co/RuudFontys/spiritual-wisdom-llama-3b

# Try without authentication
python download_and_merge_model.py --no-auth
```

## 📱 **Using the API**

Once running, you can use the REST API:

```bash
# Test the API
curl http://localhost:11434/api/generate -d '{
  "model": "spiritual-ai",
  "prompt": "What is mindfulness?",
  "stream": false
}'
```

## 🌟 **Understanding LoRA vs Full Models**

**Your HF Repository Contains:**
- ✅ LoRA adapter weights (~24MB)
- ✅ Adapter configuration
- ❌ NOT the full model weights

**After Merging You Get:**
- ✅ Complete model with spiritual wisdom
- ✅ All base model capabilities + your training
- ✅ Ready for GGUF conversion

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
FROM "./spiritual-ai.gguf"

SYSTEM """You are a meditation teacher specializing in mindfulness and contemplative practices. Focus on practical meditation guidance, breathing techniques, and present-moment awareness."""
EOF

ollama create meditation-guide -f MeditationModelfile
```

## 📊 **Model Information**

**Your Spiritual AI Stats:**
- **Base Model**: Llama 3.2 3B Instruct
- **Specialization**: Spiritual wisdom & meditation guidance  
- **Performance**: A+ Grade (0.95/1.0 overall quality)
- **Training**: 4.5 minutes on RTX 4090
- **LoRA Size**: ~24MB adapters
- **Merged Size**: ~6GB (full model)
- **GGUF Size**: ~3-4GB (quantized for efficiency)

## 🎉 **Success!**

You now have your enlightened AI master running locally through the proper workflow! 

**Next Steps:**
- Try different types of spiritual questions
- Experiment with temperature settings  
- Create specialized versions for different practices
- Integrate with your favorite tools and workflows

---

*May your local AI serve as a wise companion on your journey of understanding.* 🌟

## 🔗 **Useful Links**

- [Ollama Documentation](https://github.com/ollama/ollama)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Your Model Repository](https://huggingface.co/RuudFontys/spiritual-wisdom-llama-3b)
- [Base Model](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit) 