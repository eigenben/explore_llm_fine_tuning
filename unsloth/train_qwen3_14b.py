import torch
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

model_name = "unsloth/Qwen3-14B"
adapter_name = "qwen3-14b-lora-openmath-finetome"
max_steps = 40
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    device_map = "auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("Model Prepared: ", model)

# Prepare Dataset
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

print("Reasoning Dataset: ", reasoning_dataset)
print("Non-Reasoning Dataset: ", non_reasoning_dataset)

def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }

reasoning_conversations = tokenizer.apply_chat_template(
    list(reasoning_dataset.map(generate_conversation, batched = True)["conversations"]),
    tokenize = False,
)

print("")
print("Example Reasoning Conversation: ", reasoning_conversations[0])
print("")

dataset = standardize_sharegpt(non_reasoning_dataset)

non_reasoning_conversations = tokenizer.apply_chat_template(
    list(dataset["conversations"]),
    tokenize = False,
)

print("")
print("Example Non-Reasoning Conversation: ", non_reasoning_conversations[0])
print("")

print("Reasoning Conversations Length: ", len(reasoning_conversations))
print("Non-Reasoning Conversations Length: ", len(non_reasoning_conversations))

chat_percentage = 0.25

non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 2407,
)

print("After Sampling, Reasoning Conversations Length: ", len(reasoning_conversations))
print("After Sampling, Non-Reasoning Conversations Length: ", len(non_reasoning_subset))

data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

print("Combined Dataset Length", len(combined_dataset))


# Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        max_steps = max_steps,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Training Complete.")
print("")
print("")


# Inference Example
messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
]
print("Running Inference on: ", messages)
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
]
print("Running Inference on: ", messages)
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # Disable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024, # Increase for longer outputs!
    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# Save the Model
model.save_pretrained(adapter_name)
tokenizer.save_pretrained(adapter_name)
model.push_to_hub(adapter_name, tokenizer)

print("Saved model.")
