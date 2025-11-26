import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

print("Downloading GGUF model from HuggingFace...")

# Download model
model_path = hf_hub_download(
    repo_id="kkkkkkatherine/llama-3.2-1b-finetome-1000steps-gguf",
    filename="model.gguf",
    local_dir="./model"
)

print(f"Model downloaded to: {model_path}")
print("Loading GGUF model with optimized settings...")

# Load with optimized settings
llm = Llama(
    model_path=model_path,
    n_ctx=1024,              # Reduced from 2048 (faster)
    n_threads=6,             # Increased from 4 (use more CPU)
    n_batch=512,             # Added: larger batch for faster processing
    n_gpu_layers=0,
    verbose=False,
    use_mlock=True,          # Keep model in RAM
    use_mmap=True,           # Use memory mapping
)

print("Model loaded successfully!")

def chat(message, history):
    """Handle chat interactions"""
    # Build conversation (keep it short)
    conversation = ""
    
    # Only use last 3 turns of history to keep context short
    recent_history = history[-3:] if len(history) > 3 else history
    
    for human, assistant in recent_history:
        conversation += f"User: {human}\n"
        conversation += f"Assistant: {assistant}\n"
    
    conversation += f"User: {message}\n"
    conversation += "Assistant:"
    
    # Generate with optimized settings
    response = llm(
        conversation,
        max_tokens=128,          # Reduced from 256 (faster)
        temperature=0.7,
        top_p=0.9,
        top_k=40,               # Added: limit sampling
        repeat_penalty=1.1,
        stop=["User:", "\n\n"],
        echo=False,
    )
    
    return response['choices'][0]['text'].strip()

# Create interface WITHOUT example caching
demo = gr.ChatInterface(
    fn=chat,
    title="kkkkkkatherine/llama-3.2-1b-finetome-1000steps-gguf",
    description=(
        "Best model from 8 experiments (1000 steps, 23% loss improvement) | "
        "Optimized with GGUF Q4_K_M quantization | "
        "ID2223 Lab 2"
    ),
    examples=[
        "What is machine learning?",
        "Explain AI briefly",
        "What is LoRA?",
    ],
    cache_examples=False,  # IMPORTANT: Disable caching
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()