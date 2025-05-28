#!/usr/bin/env python3
"""
🧪 Dataset Verification Script
=============================

Quick test to verify the spiritual wisdom dataset format and show sample entries.
This helps ensure everything is ready for fine-tuning.

Usage: python test_dataset.py
"""

import json
from pathlib import Path

def test_dataset():
    """Test and display information about the dataset."""
    dataset_path = Path("datasets/llama_optimized.jsonl")
    
    print("🧪 Testing Spiritual Wisdom Dataset")
    print("=" * 50)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Run generate_finetuning_dataset.py to create the dataset")
        return False
    
    # Load and analyze dataset
    data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON error on line {line_num}: {e}")
                        return False
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False
    
    print(f"✅ Dataset loaded successfully")
    print(f"📊 Total entries: {len(data)}")
    
    # Analyze dataset structure
    required_fields = ['instruction', 'input', 'output']
    field_counts = {field: 0 for field in required_fields}
    
    for entry in data:
        for field in required_fields:
            if field in entry and entry[field]:
                field_counts[field] += 1
    
    print(f"\n📋 Field Analysis:")
    for field, count in field_counts.items():
        percentage = (count / len(data)) * 100
        print(f"   {field}: {count}/{len(data)} ({percentage:.1f}%)")
    
    # Show sample entries
    print(f"\n📝 Sample Entries:")
    print("-" * 50)
    
    for i, entry in enumerate(data[:3]):
        print(f"\n🔸 Entry {i+1}:")
        print(f"   Instruction: {entry.get('instruction', 'N/A')[:100]}...")
        print(f"   Input: {entry.get('input', 'N/A')[:50]}...")
        print(f"   Output: {entry.get('output', 'N/A')[:100]}...")
    
    # Analyze content lengths
    instruction_lengths = [len(entry.get('instruction', '')) for entry in data]
    output_lengths = [len(entry.get('output', '')) for entry in data]
    
    print(f"\n📏 Content Length Analysis:")
    print(f"   Instructions - Avg: {sum(instruction_lengths)/len(instruction_lengths):.0f} chars")
    print(f"   Instructions - Range: {min(instruction_lengths)}-{max(instruction_lengths)} chars")
    print(f"   Outputs - Avg: {sum(output_lengths)/len(output_lengths):.0f} chars")
    print(f"   Outputs - Range: {min(output_lengths)}-{max(output_lengths)} chars")
    
    # Format sample for Llama 3.2
    print(f"\n🤖 Llama 3.2 Format Sample:")
    print("-" * 50)
    
    sample = data[0]
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    # Combine instruction and input if both exist
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction
    
    # Format using Llama 3.2 chat template
    formatted_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )
    
    print(formatted_text[:300] + "..." if len(formatted_text) > 300 else formatted_text)
    
    print(f"\n✅ Dataset verification complete!")
    print(f"🎯 Ready for fine-tuning with {len(data)} spiritual wisdom examples")
    
    return True

if __name__ == "__main__":
    test_dataset() 