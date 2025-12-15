import torch
from transformers import pipeline

# Setup Device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup Pipeline:
pipe = pipeline("text-generation", model="eigenben/smollm-business-news-generator", device=device)

prompts = [
    "Q1",
    "Wall",
    "Google"
]

print("Prompts: ", prompts)

for prompt in prompts:
    print("Prediction: ", pipe(prompt, do_sample=True, temperature=0.1, max_new_tokens=30)[0]['generated_text'])
