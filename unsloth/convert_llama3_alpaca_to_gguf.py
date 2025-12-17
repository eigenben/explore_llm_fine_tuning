import torch
from unsloth import FastLanguageModel

# Setup some parameters
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER_NAME = "eigenben/llama-3-8b-bnb-4bit-alpaca-finetuned"
GGUF_ADAPTER_NAME = "eigenben/llama-3-8b-bnb-4bit-alpaca-finetuned-gguf"

# Setup the Model:
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True
)

model = FastLanguageModel.for_inference(model)

model.save_pretrained_gguf(GGUF_ADAPTER_NAME, tokenizer, quantization_method="q4_k_m")
model.push_to_hub_gguf(GGUF_ADAPTER_NAME, tokenizer, quantization_method="q4_k_m")

