import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from random import randint

model_id = "eigenben/code-llama-3-1-8b-text-to-sql"

# Load Model with PEFT adapter
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

eval_dataset = load_dataset("b-mc2/sql-create-context", split="train")
eval_dataset = eval_dataset.shuffle().select(range(2000))
rand_idx = randint(0, len(eval_dataset) - 1)

# Format the dataset into chat messages
sample = eval_dataset[rand_idx]
messages = [
    {"role": "system", "content": "You are a helpful assistant that generates SQL queries based on the given context and question."},
    {"role": "user", "content": f"Context: {sample['context']}\n\nQuestion: {sample['question']}"}
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=pipe.tokenizer.eos_token_id,
    pad_token_id=pipe.tokenizer.pad_token_id
)

print(f"Context:\n{sample['context']}")
print(f"\nQuestion:\n{sample['question']}")
print(f"\nOriginal Answer:\n{sample['answer']}")
print(f"\nGenerated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

def evaluate(sample):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates SQL queries based on the given context and question."},
        {"role": "user", "content": f"Context: {sample['context']}\n\nQuestion: {sample['question']}"}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    # Note: This uses exact string matching which is very strict.
    # Semantically correct SQL queries with different formatting/capitalization will be counted as incorrect.
    # Consider using SQL parsing or execution-based evaluation for more accurate results.
    if predicted_answer == sample["answer"]:
        return 1
    else:
        return 0

success_rate = []
number_of_eval_samples = 50

for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
    success_rate.append(evaluate(s))

accuracy = sum(success_rate) / len(success_rate)

print(f"Accuracy: {accuracy*100:.2f}%")
