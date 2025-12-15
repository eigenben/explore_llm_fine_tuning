import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Setup Device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup Dataset:
raw_datasets = load_dataset("fancyzhx/ag_news")
filtered_datasets = raw_datasets.filter(lambda example: example['label'] == 2)
filtered_datasets = filtered_datasets.remove_columns('label')

# Setup Tokenizer & Model:
model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True)

tokenized_datasets = filtered_datasets.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train:
training_args = TrainingArguments(
    "smollm-business-news-generator",
    push_to_hub=True,
    per_device_train_batch_size=16,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
trainer.push_to_hub()
