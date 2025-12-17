import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Setup some parameters
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER_NAME = "eigenben/llama-3-8b-bnb-4bit-alpaca-finetuned"

# Setup the Model:
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True
)

model = FastLanguageModel.for_inference(model)

# Run Inference:
# Format prompt manually for Llama-3 (base model doesn't have chat template)
prompt = "Describe anything special about a sequence. Your input is: 1, 1, 2, 3, 5, 8,"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

model.generate(**inputs, streamer=text_streamer, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)

