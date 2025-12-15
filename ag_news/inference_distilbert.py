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
pipe = pipeline("text-classification", model="eigenben/distilbert-ag-news", device=device)

prompts = [
    "The soccer match between Spain and Portugal ended in a terrible result for Portugal"
]

print("Prompts: ", prompts)
print("Predictions: ", pipe(prompts))
