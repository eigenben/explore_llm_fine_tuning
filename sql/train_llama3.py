import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""

def create_conversation(sample):
    return {
      "messages": [
        {"role": "system", "content": system_message.format(schema=sample["context"])},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]}
      ]
    }

# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
dataset = dataset.train_test_split(test_size=2500/12500)  # split dataset into 10,000 training samples and 2,500 test samples

# Setup Model:
model_id = "meta-llama/Meta-Llama-3.1-8B" # or `mistralai/Mistral-7B-v0.1`
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# Set up chat template and special tokens
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}{{ eos_token }}"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False  # Required for gradient checkpointing

peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="output/code-llama-3-1-8b-text-to-sql", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="wandb",                      # report metrics to wandb
    # SFT-specific parameters
    max_length=2048,                        # max sequence length for model and packing of the dataset
    packing=True,                           # pack multiple short examples in the same input sequence
    dataset_kwargs={
        "add_special_tokens": False,        # We template with special tokens
        "append_concat_token": False,       # No need to add additional separator token
    }
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],         # Use the train split from DatasetDict
    eval_dataset=dataset["test"],           # Add evaluation dataset
    peft_config=peft_config,
    processing_class=tokenizer,             # Use processing_class instead of deprecated tokenizer param
)

trainer.train()
trainer.push_to_hub()
trainer.save_model()
