from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
    args=SFTConfig(
        "qwen3-0.6b-capybara",
        push_to_hub=True,
        report_to="wandb",
    )
)

trainer.train()
trainer.push_to_hub()
