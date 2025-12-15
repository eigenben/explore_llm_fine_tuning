import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Setup Device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup Dataset:
raw_datasets = load_dataset("fancyzhx/ag_news")
raw_train_dataset = raw_datasets['train']
num_labels = len(set(raw_train_dataset['label']))
num_samples = 10000

# Setup Model & Tokenizer:
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(device)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt')

tokenized_datasets = raw_datasets.map(tokenize, batched=True)
shuffled_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_split = shuffled_dataset.select(range(num_samples))

# Establish Metrics:
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1 = f1_score.compute(predictions=preds, references=labels, average='weighted')
    return {"accuracy": acc['accuracy'], "f1": f1['f1']}

# Train:
batch_size = 64
training_args = TrainingArguments(
    "distilbert-ag-news",
    push_to_hub=True,
    num_train_epochs=2,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=small_split,
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

trainer.train()
