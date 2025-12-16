from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
    args=SFTConfig(
        "Qwen2.5-VL-3B-Instruct-llava-mix",
        max_length=None,
        push_to_hub=True,
        report_to="wandb",
    ),
)

trainer.train()
trainer.push_to_hub()
