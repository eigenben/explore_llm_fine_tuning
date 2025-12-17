from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model.config.use_cache = False  # Disable KV cache for training
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    report_to="wandb",
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=8,  # Accumulate gradients to maintain effective batch size
    gradient_checkpointing=True,  # Trade compute for memory
    bf16=True,  # Use mixed precision (bf16 if supported, else fp16)
    max_length=512,  # Limit sequence length
    max_prompt_length=256,  # Limit prompt length
    optim="adamw_8bit",  # Use 8-bit optimizer
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
