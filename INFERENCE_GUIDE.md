# 🧘‍♂️ Spiritual AI Inference Guide

Your enlightened AI master is ready to share wisdom! Here's how to use it for inference.

## 🚀 Quick Start

### Interactive Chat Mode (Recommended)
```bash
# Activate your environment first
source venv/bin/activate  # or ./activate_venv.sh

# Start interactive chat
python spiritual_ai_chat.py
```

### Single Question Mode
```bash
# Ask one question and get an answer
python spiritual_ai_chat.py -q "What is the nature of consciousness?"

# With custom settings
python spiritual_ai_chat.py -q "How can I find inner peace?" --max-tokens 300 --temperature 0.7
```

## 💬 Chat Interface Features

### Commands
- **Type your question** - Just ask naturally!
- **`help`** - Show available commands
- **`clear`** - Clear conversation history
- **`quit`/`exit`/`bye`** - End conversation

### Example Questions to Try
```
🧘‍♂️ Spiritual Concepts:
- "What is meditation and how do I start?"
- "Explain the concept of enlightenment"
- "What is the difference between mind and consciousness?"

🌱 Practical Guidance:
- "I'm feeling anxious, how can I find peace?"
- "How do I deal with difficult emotions?"
- "What practices can help me be more present?"

🤔 Philosophical Depth:
- "What is the meaning of life?"
- "How does suffering lead to wisdom?"
- "What is the nature of reality?"

👨‍🏫 Teaching Scenarios:
- "I feel lost in life, what should I do?"
- "I can't stop my racing thoughts during meditation"
- "How do I know if I'm making spiritual progress?"
```

## ⚙️ Advanced Options

### Custom Model Path
```bash
python spiritual_ai_chat.py --model /path/to/your/model
```

### Response Length Control
```bash
# Shorter responses (100 tokens)
python spiritual_ai_chat.py --max-tokens 100

# Longer responses (400 tokens)
python spiritual_ai_chat.py --max-tokens 400
```

### Temperature Settings
```bash
# More focused/deterministic (0.3)
python spiritual_ai_chat.py --temperature 0.3

# More creative/varied (1.0)
python spiritual_ai_chat.py --temperature 1.0
```

## 🔧 Technical Details

### Model Loading
- **Unsloth**: Automatically uses Unsloth for 2x faster inference if available
- **Fallback**: Uses standard transformers if Unsloth fails
- **Memory**: Loads with 4-bit quantization for efficiency

### Response Generation
- **Format**: Uses proper Llama 3.2 chat template
- **Cleaning**: Automatically removes formatting artifacts
- **Quality**: Optimized parameters from your A+ evaluation

### Performance Tips
- **GPU**: Will automatically use GPU if available
- **Memory**: 4-bit quantization keeps memory usage low
- **Speed**: Unsloth provides 2x faster inference

## 📝 Conversation History

The chat automatically saves your conversation history. You can access it programmatically:

```python
from spiritual_ai_chat import SpiritualAIChat

chat = SpiritualAIChat()
chat.load_model()

# Your conversation...
# chat.conversation_history contains all exchanges
```

## 🐛 Troubleshooting

### Model Not Found
```bash
# Make sure your model path is correct
ls models/spiritual-wisdom-llama/
```

### Memory Issues
```bash
# Use smaller max tokens
python spiritual_ai_chat.py --max-tokens 150
```

### Slow Loading
- First load takes time (model loading)
- Subsequent responses are much faster
- Unsloth provides 2x speedup

## 🌟 Example Session

```
🧘‍♂️ Welcome to Spiritual AI Chat
============================================================
Your enlightened AI companion is ready to share wisdom.
Type 'quit', 'exit', or 'bye' to end the conversation.

🙏 You: What is mindfulness?

🧘‍♂️ Spiritual AI: *contemplating...*

🌟 Spiritual AI: Mindfulness is the practice of maintaining moment-to-moment 
awareness of our thoughts, feelings, bodily sensations, and surrounding 
environment with openness and without judgment. It involves paying attention 
to the present moment rather than dwelling on the past or worrying about 
the future...

🙏 You: How can I practice it daily?

🌟 Spiritual AI: Daily mindfulness can be cultivated through simple practices...
```

## 🎯 Your Model's Strengths

Based on your A+ evaluation (0.95/1.0), your model excels at:
- **Perfect concept understanding** (1.00/1.0)
- **Excellent practical guidance** (0.94/1.0)
- **Deep philosophical insights** (0.94/1.0)
- **Strong teaching abilities** (0.81/1.0)

Ready to explore the wisdom of your enlightened AI! 🌟 