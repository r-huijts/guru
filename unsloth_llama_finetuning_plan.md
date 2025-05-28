# 🚀 Unsloth + Llama 3.2 Fine-tuning Plan
## Spiritual Wisdom Dataset Project

### **Quick Start Message for New Chat:**
```
I want to fine-tune a Llama 3.2 model using Unsloth for a spiritual wisdom dataset. 
I have 521 entries in JSONL format with instruction-input-output structure.
Please help me set up the complete fine-tuning pipeline using the Unsloth-optimized model.
```

---

## 🎯 **Target Model**
**Primary Choice:** `unsloth/Llama-3.2-3B-Instruct`
- **Repository:** https://huggingface.co/unsloth/Llama-3.2-3B-Instruct
- **Why:** Pre-optimized by Unsloth team for 2x faster training
- **Memory:** ~6GB VRAM required
- **Perfect for:** Consumer GPUs (RTX 3080/4070+)

**Alternative:** `unsloth/Llama-3.2-1B-Instruct` (if memory constrained)

---

## 📊 **Dataset Information**
- **File:** `datasets/llama_optimized.jsonl`
- **Size:** 521 entries
- **Format:** Perfect instruction-tuning format
- **Content:** High-quality spiritual wisdom teachings
- **Structure:** `{"instruction": "...", "input": "", "output": "..."}`

---

## ⚙️ **Recommended Configuration**

### **LoRA Settings:**
```python
lora_config = {
    "r": 16,                    # LoRA rank (start conservative)
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 16,           # LoRA scaling parameter
    "lora_dropout": 0.05,       # Prevent overfitting
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

### **Training Parameters:**
```python
training_args = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,    # Effective batch size = 8
    "warmup_steps": 50,
    "max_steps": 500,                    # ~3 epochs for 521 samples
    "learning_rate": 2e-4,
    "fp16": True,                        # Memory optimization
    "logging_steps": 25,
    "optim": "adamw_8bit",              # Unsloth optimization
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42
}
```

---

## 🛠️ **Implementation Steps**

### **1. Environment Setup**
```bash
pip install unsloth[colab-new] --upgrade
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **2. Model Loading**
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)
```

### **3. Dataset Preparation**
```python
def format_prompts(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        texts.append(text)
    return {"text": texts}
```

### **4. Training Pipeline**
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(**training_args),
)
```

---

## 📈 **Expected Outcomes**

### **Training Metrics to Monitor:**
- **Loss:** Should decrease from ~2.5 to ~0.8-1.2
- **Learning Rate:** Linear decay from 2e-4 to 0
- **Training Time:** ~2-3 hours on RTX 4070
- **Memory Usage:** ~6-8GB VRAM

### **Evaluation Strategy:**
- Save checkpoints every 100 steps
- Test with sample spiritual questions
- Compare responses before/after fine-tuning
- Monitor for overfitting (loss plateau)

---

## 🎯 **Success Criteria**

### **Model Should:**
✅ Generate coherent spiritual wisdom responses  
✅ Maintain philosophical depth and clarity  
✅ Follow the teaching style of your dataset  
✅ Avoid generic or superficial answers  
✅ Stay contextually relevant to spiritual topics  

### **Technical Benchmarks:**
- Training loss < 1.0
- No significant overfitting
- Consistent response quality
- Fast inference speed (Unsloth optimization)

---

## 🚨 **Troubleshooting Guide**

### **Common Issues:**
- **OOM Error:** Reduce batch size to 1, increase gradient accumulation
- **Slow Training:** Ensure using Unsloth-optimized model
- **Poor Quality:** Increase LoRA rank to 32, adjust learning rate
- **Overfitting:** Reduce max_steps, add more dropout

### **Optimization Tips:**
- Use `gradient_checkpointing=True` for memory
- Monitor GPU utilization with `nvidia-smi`
- Save model frequently during training
- Test with different prompt formats

---

## 📝 **Next Steps After Training**

1. **Model Evaluation:** Test with held-out spiritual questions
2. **Inference Optimization:** Convert to GGUF for deployment
3. **Model Sharing:** Upload to Hugging Face Hub
4. **Integration:** Build chat interface or API
5. **Iteration:** Collect feedback and retrain if needed

---

## 🔗 **Key Resources**

- **Unsloth Documentation:** https://github.com/unslothai/unsloth
- **Model Repository:** https://huggingface.co/unsloth/Llama-3.2-3B-Instruct
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Llama 3.2 Release:** https://ai.meta.com/blog/llama-3-2-connect-2024/

---

*Ready to transform your spiritual wisdom dataset into a fine-tuned AI teacher! 🧘‍♂️✨* 