import torch
import numpy as np
from typing import Callable
from unsloth import FastLanguageModel, execute_with_time_limit, check_python_modules, create_locked_down_function
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer
from datasets import Dataset
from game_board import GameBoard

MODEL_NAME = "unsloth/gpt-oss-20b"
ADAPTER_NAME = "gpt-oss-20b-grpo-2048-game"
MAX_SEQ_LENGTH = 768
LORA_RANK = 4

# Setup Model:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    offload_embedding = True, # Reduces VRAM by 1GB
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = LORA_RANK * 2,  # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# Setup RL Strategy:
def _execute_strategy(strategy : Callable, game : GameBoard):
    assert callable(strategy)

    steps = 0
    while game.state() == "ongoing":
        action = strategy(list(game.board()))
        steps += 1
        if type(action) is not str:
            return steps, "failed"
        game.do_action(action)
    return steps, game.state()

@execute_with_time_limit(5)
def execute_strategy(strategy : Callable, game : GameBoard):
    return _execute_strategy(strategy, game)

def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first : second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"): return fx
    return None

def function_works(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_strategy = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores

def no_cheating(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0) # Penalize heavily!
        else:
            scores.append(-1.0) # Failed creating function
    return scores

global PRINTER
PRINTER = 0
def strategy_succeeds(completions, **kwargs):
    global PRINTER
    scores = []
    # Generate a random game board with seed
    seed = np.random.randint(10000)
    for completion in completions:
        printed = False
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if PRINTER % 5 == 0:
            printed = True
            print(function)
        PRINTER += 1
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        try:
            game = GameBoard(size = 6, seed = seed, target = 2048, probability_fours = 0.10)
            steps, game_state = execute_strategy(new_strategy, game)
            print(f"Steps = {steps} State = {game_state}")
            if printed is False:
                print(function)
            print(game.board().pretty())
            if game_state == "success":
                scores.append(20.0) # Success - massively reward!
            else:
                scores.append(2.0) # Failed but function works!
        except TimeoutError as e:
            print("Timeout")
            scores.append(-1.0) # Failed with timeout
        except Exception as e:
            print(f"Exception = {str(e)}")
            scores.append(-3.0) # Failed
    return scores

# Prepare Data & Prompts:
prompt = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "W", "A", "S", "D" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "W" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

dataset = Dataset.from_list([{"prompt" : [{"role": "user", "content": prompt.strip()}], "answer" : 0, "reasoning_effort": "low"}]*1000)
maximum_length = len(tokenizer.apply_chat_template([{"role": "user", "content": prompt.strip()}], add_generation_prompt = True))

print("*" * 20)
print("Using Dataset: ", dataset[0])
print("*" * 20)
print("")

max_prompt_length = maximum_length + 1 # + 1 just in case!
max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 800,
    save_steps = 100,
    report_to = "wandb", # Can use Weights & Biases, TrackIO
    output_dir = "outputs",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,
        no_cheating,
        strategy_succeeds,
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

print("*" * 20)
print("Starting Training:")
print("*" * 20)
print("")

trainer.train()

print("")
print("*" * 20)
print("Training Complete!")
print("*" * 20)
print("")

# Evaluate Sample:
print("*" * 20)
print("Evaluating Example:")
print("*" * 20)

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)

# Save Model:
model.save_pretrained(ADAPTER_NAME)
tokenizer.save_pretrained(ADAPTER_NAME)
model.push_to_hub(ADAPTER_NAME)
tokenizer.push_to_hub(ADAPTER_NAME)
