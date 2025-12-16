# LLM Fine-Tuning Exploration

A collection of practical examples demonstrating various LLM fine-tuning techniques, from simple classification to complex instruction-following tasks.

## Overview

This repository explores different fine-tuning approaches across multiple models and datasets, showcasing techniques like LoRA, QLoRA, and full fine-tuning with varying model sizes (135M to 8B parameters).

## Examples

| Directory | Model | Size | Technique | Dataset | Task |
|-----------|-------|------|-----------|---------|------|
| `ag_news/` | DistilBERT | Base | Full fine-tuning | AG News | Text classification |
| `ag_news/` | SmolLM | 135M | Full + LoRA | AG News (Business) | Text generation |
| `open_assistant/` | Mistral | 7B | QLoRA (4-bit) | OpenAssistant Guanaco | Instruction following |
| `sql/` | Llama 3.1 | 8B | QLoRA (4-bit) | SQL-Create-Context | Text-to-SQL |
| `capybara/` | Qwen3 | 0.6B | Minimal SFT | Capybara | General purpose |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

Each directory contains training and inference scripts:

```bash
# Example: Train SmolLM with LoRA
uv run python ag_news/train_smollm_135m_lora.py

# Example: Run inference
uv run python ag_news/inference_smollm_135m.py
```

## Key Features

- **Parameter-Efficient Fine-Tuning**: LoRA and QLoRA examples for resource-constrained environments
- **4-bit Quantization**: BitsAndBytes integration for large models (Mistral, Llama)
- **Gradient Checkpointing**: Memory optimization for training large models
- **Experiment Tracking**: Weights & Biases integration across all examples
- **HuggingFace Integration**: Models pushed to Hub after training

## Requirements

- Python >= 3.12
- PyTorch 2.6.0+
- Transformers, PEFT, TRL, Accelerate, BitsAndBytes
- CUDA GPU recommended (CPU and Apple Silicon MPS supported)

## Models Trained

All fine-tuned models are available on HuggingFace Hub under the `eigenben` account.
