import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Setup Device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load Model and Tokenizer:
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
).to(device)

adapter_name = "eigenben/mistral-7b-open-assistant-lora"
model.load_adapter(adapter_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
print(pipe("### Human: Hello!### Assistant:", max_new_tokens=100))
