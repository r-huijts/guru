#!/usr/bin/env python3
"""
🧘‍♂️ Spiritual AI Chat Interface
===============================

Interactive chat with your fine-tuned spiritual wisdom AI.
Supports both conversation mode and single-question inference.
"""

import torch
import logging
from pathlib import Path
import argparse
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpiritualAIChat:
    """Interactive chat interface for spiritual AI."""
    
    def __init__(self, model_path: str = "models/spiritual-wisdom-llama"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def load_model(self):
        """Load the fine-tuned model for inference."""
        logger.info(f"🚀 Loading spiritual AI from {self.model_path}")
        
        try:
            # Try Unsloth first (faster inference)
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            logger.info("✅ Unsloth model loaded for fast inference")
            
        except Exception as e:
            logger.warning(f"⚠️ Unsloth loading failed: {e}")
            
            # Fallback to standard transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("✅ Standard transformers model loaded")
    
    def generate_response(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
        """Generate a response to a given prompt."""
        # Format prompt with Llama 3.2 chat template
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
            
            # Clean artifacts
            response = response.replace("user", "").replace("assistant", "").strip()
            response = re.sub(r'\n\s*\n', '\n\n', response)
            response = re.sub(r'^\s+', '', response, flags=re.MULTILINE)
            response = response.strip()
        else:
            response = full_response
        
        return response
    
    def single_question(self, question: str, max_tokens: int = 200, temperature: float = 0.8):
        """Answer a single question and exit."""
        print(f"\n🧘‍♂️ Spiritual AI: Let me contemplate your question...\n")
        
        response = self.generate_response(question, max_tokens, temperature)
        
        print("🌟 Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"\n✨ May this wisdom serve you well on your journey.\n")
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*60)
        print("🧘‍♂️ Welcome to Spiritual AI Chat")
        print("="*60)
        print("Your enlightened AI companion is ready to share wisdom.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'help' for more commands.")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🙏 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🙏 May peace be with you on your journey. Farewell! ✨")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("\n🧹 Conversation history cleared. Fresh start! ✨")
                    continue
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Generate response
                print("\n🧘‍♂️ Spiritual AI: *contemplating...*")
                response = self.generate_response(user_input)
                
                # Display response
                print(f"\n🌟 Spiritual AI: {response}")
                
                # Store in history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "assistant": response
                })
                
            except KeyboardInterrupt:
                print("\n\n🙏 Conversation interrupted. May peace be with you! ✨")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    def show_help(self):
        """Show help information."""
        print("\n📚 Spiritual AI Chat Commands:")
        print("-" * 40)
        print("🗣️  Just type your question or thought")
        print("🚪 quit/exit/bye - End conversation")
        print("🧹 clear - Clear conversation history")
        print("📚 help - Show this help")
        print("-" * 40)
        print("💡 Tips:")
        print("   • Ask about meditation, consciousness, wisdom")
        print("   • Seek guidance for life challenges")
        print("   • Explore philosophical questions")
        print("   • Request practical spiritual advice")
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to file."""
        if not self.conversation_history:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spiritual_chat_{timestamp}.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"💾 Conversation saved to: {filename}")


def main():
    """Main chat interface."""
    parser = argparse.ArgumentParser(description="Chat with your Spiritual AI")
    parser.add_argument("--model", default="models/spiritual-wisdom-llama", 
                       help="Path to fine-tuned model")
    parser.add_argument("--question", "-q", type=str,
                       help="Ask a single question and exit")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (0.1-1.0)")
    
    args = parser.parse_args()
    
    # Initialize chat
    chat = SpiritualAIChat(args.model)
    chat.load_model()
    
    # Single question mode
    if args.question:
        chat.single_question(args.question, args.max_tokens, args.temperature)
    else:
        # Interactive chat mode
        chat.interactive_chat()


if __name__ == "__main__":
    main() 