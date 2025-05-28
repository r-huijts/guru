#!/usr/bin/env python3
"""
🧘‍♂️ Test Spiritual AI in Ollama
================================

Quick test script to verify your spiritual AI is working correctly in Ollama.
"""

import requests
import json
import time
import sys

def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running with {len(models)} models")
            return True, models
        else:
            print("❌ Ollama is not responding correctly")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Make sure it's running:")
        print("   Try: ollama serve")
        return False, []

def find_spiritual_model(models):
    """Find the spiritual AI model in the available models."""
    spiritual_models = []
    for model in models:
        name = model.get('name', '')
        if any(keyword in name.lower() for keyword in ['spiritual', 'wisdom', 'llama']):
            spiritual_models.append(name)
    
    if spiritual_models:
        print(f"🧘‍♂️ Found spiritual models: {spiritual_models}")
        return spiritual_models[0]  # Return the first match
    else:
        print("❌ No spiritual AI model found in Ollama")
        print("Available models:")
        for model in models:
            print(f"   - {model.get('name', 'Unknown')}")
        return None

def test_spiritual_ai(model_name):
    """Test the spiritual AI with sample questions."""
    
    test_questions = [
        "What is consciousness?",
        "How can I find inner peace?",
        "What is the nature of meditation?",
        "How do I deal with anxiety and stress?"
    ]
    
    print(f"\n🧘‍♂️ Testing Spiritual AI: {model_name}")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 30)
        
        try:
            # Send request to Ollama
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    'model': model_name,
                    'prompt': question,
                    'stream': False,
                    'options': {
                        'temperature': 0.8,
                        'top_p': 0.9,
                        'max_tokens': 200
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response received')
                
                # Clean up the response
                answer = answer.strip()
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                
                print(f"🌟 Answer: {answer}")
                
                # Simple quality check
                if len(answer) > 50 and any(word in answer.lower() for word in 
                    ['consciousness', 'peace', 'meditation', 'breath', 'mind', 'awareness', 'present']):
                    print("✅ Response looks spiritual and relevant!")
                else:
                    print("⚠️ Response might need improvement")
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out - model might be loading")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay between requests
        time.sleep(1)

def main():
    """Main test function."""
    print("🧘‍♂️ Spiritual AI Ollama Test")
    print("=" * 40)
    
    # Test Ollama connection
    is_connected, models = test_ollama_connection()
    if not is_connected:
        sys.exit(1)
    
    # Find spiritual model
    model_name = find_spiritual_model(models)
    if not model_name:
        print("\n💡 To import your spiritual AI:")
        print("   ollama pull YOUR_USERNAME/spiritual-wisdom-llama-3b")
        sys.exit(1)
    
    # Test the model
    test_spiritual_ai(model_name)
    
    print("\n🎉 Testing complete!")
    print(f"Your spiritual AI ({model_name}) is ready for enlightened conversations! 🌟")

if __name__ == "__main__":
    main() 