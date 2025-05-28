# 🦙 Guru LLAMA Fine-Tuning Dataset Generator

A powerful Python tool for converting speech transcripts into high-quality training datasets optimized for LLAMA instruction-following models. Perfect for creating AI models that can answer questions and share wisdom in a specific speaker's authentic voice.

## 🎯 Features

- **Smart Text Processing**: Handles both paragraph-formatted and single-line transcript files
- **Multiple Output Formats**: Alpaca, ChatML, Q&A, and optimized LLAMA formats
- **Quality Filtering**: Intelligent filtering to ensure high-quality training examples
- **Q&A Extraction**: Automatically extracts question-answer pairs from natural speech
- **RunPod Optimized**: Includes specific recommendations for RunPod fine-tuning

## 📊 Dataset Statistics

From 11 speech transcript files (~134KB total), the generator produces:
- **520 total training examples**
- **414 high-quality text segments**
- **95 extracted Q&A pairs**
- **Multiple format options** for different model architectures

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/r-huijts/guru-llama-finetuning.git
cd guru-llama-finetuning
pip install -r requirements.txt
```

### Usage

1. **Add your transcript files** to the `data/` directory (`.txt` format)
2. **Run the generator**:
   ```bash
   python generate_finetuning_dataset.py
   ```
3. **Find your datasets** in the `datasets/` directory

### Recommended File for LLAMA Fine-tuning
Use `datasets/llama_optimized.jsonl` - it's specifically optimized for LLAMA instruction-following with:
- Enhanced instruction templates
- Authentic voice preservation  
- Optimal segment granularity
- Mixed training example types

## 📁 Project Structure

```
guru-llama-finetuning/
├── data/                          # Input transcript files (.txt)
├── datasets/                      # Generated training datasets
│   ├── llama_optimized.jsonl     # 🌟 RECOMMENDED for LLAMA
│   ├── alpaca_format.jsonl       # Standard Alpaca format
│   ├── qa_alpaca_format.jsonl    # Q&A in Alpaca format
│   ├── conversation_format.jsonl # ChatML style
│   ├── qa_format.json           # Pure Q&A pairs
│   └── dataset_summary.json     # Detailed statistics & tips
├── generate_finetuning_dataset.py # Main script
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🎯 Output Formats

### LLAMA Optimized (Recommended)
```json
{
  "instruction": "What is your understanding of consciousness?",
  "input": "",
  "output": "Consciousness is the very basis of all physical existence..."
}
```

### Q&A Format
```json
{
  "question": "How does one develop true understanding?",
  "answer": "True understanding comes when you stop identifying with...",
  "source": "speech_transcript"
}
```

## 🔧 RunPod Fine-Tuning Tips

The generator includes specific recommendations for RunPod LLAMA fine-tuning:

- **Learning Rate**: 1e-5 to 3e-5 (start conservative)
- **Batch Size**: 4-8 depending on GPU memory
- **Epochs**: 3-5 (monitor for overfitting)
- **Method**: LoRA recommended (rank 16-64)
- **GPU**: RTX 4090 24GB works perfectly
- **Template**: Use Axolotl Jupyter Lab template

## 📈 Quality Features

### Smart Text Processing
- **Single-line detection**: Automatically handles both formatted and unformatted transcripts
- **Repetition cleanup**: Removes transcription artifacts and repetitive content
- **Topic segmentation**: Creates coherent, focused training examples

### Enhanced Q&A Extraction
- **Natural speech patterns**: Extracts Q&A from conversational flow
- **Teaching moments**: Identifies rhetorical questions and explanations
- **Quality filtering**: Ensures meaningful, complete question-answer pairs

### Instruction Optimization
- **Spiritual/philosophical focus**: Templates designed for wisdom-sharing content
- **Authentic voice**: Preserves the speaker's unique teaching style
- **Diverse prompts**: Multiple instruction styles for robust training

## 🛠️ Customization

### Adding New Transcript Files
Simply add `.txt` files to the `data/` directory. The script handles:
- Single-line format (entire transcript in one line)
- Paragraph format (properly formatted with line breaks)
- Mixed formats in the same dataset

### Adjusting Quality Filters
Modify the `is_quality_segment()` method in `generate_finetuning_dataset.py`:
- **Minimum words**: Currently 15 (adjustable)
- **Maximum words**: Currently 150 (adjustable)
- **Meaningful content**: Keyword-based filtering
- **Uniqueness**: Repetition detection

### Custom Instruction Templates
Add your own instruction templates in `generate_instruction_response_pairs()`:
```python
custom_templates = [
    "Your custom instruction template for {}",
    "Another template asking about {}",
]
```

## 📊 Dataset Quality Metrics

The generator provides detailed statistics in `dataset_summary.json`:
- Total segments processed
- Q&A pairs extracted
- Format-specific example counts
- Quality filtering results
- RunPod-specific recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Optimized for spiritual/philosophical content fine-tuning
- Designed for RunPod cloud GPU training
- Built with LLAMA instruction-following in mind

---

**Ready to create your own wisdom-sharing AI model?** 🚀

Start by adding your transcript files to the `data/` directory and running the generator! 