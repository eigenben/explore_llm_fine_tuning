import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Setup Device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Prepare Dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
train_dataset = dataset['train'].select(range(2000))
test_dataset = dataset['test']

# Setup Model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_id = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto").to(device)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    "mistral-7b-open-assistant-lora",
    push_to_hub=True,
    per_device_train_batch_size=32,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    gradient_checkpointing=True,
    max_length=512,
    dataset_text_field="text",
    packing=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()
