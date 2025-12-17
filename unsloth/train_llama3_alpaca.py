import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, apply_chat_template
from trl import SFTConfig, SFTTrainer

# Setup some parameters
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER_NAME = "llama-3-8b-bnb-4bit-alpaca-finetuned"

# Setup the Model:
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True
)

# Prepare our Data:
model = FastLanguageModel.get_peft_model(model)
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

# Convert from columns ['instruction', 'input', 'output', 'text'] => ['conversation']
dataset = standardize_sharegpt(to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,
))

# Apply a Chat Template:
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

# Applies template and sets as 'text' column so column_names = ['conversation', 'text']
dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

# Configure and Initialize the Trainer:
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        num_train_epochs = 1,
        max_steps = 100,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use TrackIO/WandB etc
    ),
)

# Show memory usage before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the Model:
trainer_stats = trainer.train()

# Show memory usage after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the Model:
model.save_pretrained(ADAPTER_NAME)
tokenizer.save_pretrained(ADAPTER_NAME)
model.push_to_hub(ADAPTER_NAME)
tokenizer.push_to_hub(ADAPTER_NAME)

