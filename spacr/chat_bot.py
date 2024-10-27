import os
import torch
from multiprocessing import Queue
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_queue):
    """Load the Falcon-7B model and send the status via a queue."""
    try:
        print("Loading Falcon-7B model...")
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

        # Set `pad_token` to `eos_token` if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
        model.config.pad_token_id = tokenizer.pad_token_id

        # Send the loaded model and tokenizer back via the queue
        model_queue.put((model, tokenizer, "Model loaded successfully!"))
    except Exception as e:
        model_queue.put((None, None, f"Error loading model: {e}"))

def language_chat(tokenizer, model, user_input):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

