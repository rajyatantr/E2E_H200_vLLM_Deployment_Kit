#!/usr/bin/env python3
"""
finetune_qwen_vl.py — Fine-tune Qwen2.5-VL on health claim form data

Uses LoRA (Low-Rank Adaptation) to fine-tune Qwen2.5-VL-7B-Instruct
on annotated training data from the ADE pipeline.

Why 7B for fine-tuning (not 72B):
  - 7B fits in H200 with full training overhead (~60GB with LoRA)
  - Fine-tuned 7B often beats un-tuned 72B on specific domains
  - Much faster training: ~5min per epoch vs hours for 72B
  - Can always serve the LoRA adapter on top of the 7B base

After fine-tuning:
  1. Merge LoRA weights: python3 finetune_qwen_vl.py --merge-only
  2. Serve with vLLM:    bash deploy/start_vllm.sh --model ./training/merged_model --quant null --dtype bfloat16
  3. Run extraction:     python3 qwen_vl_extract.py --config config.yaml <pdf>

Requirements:
  pip install transformers peft accelerate bitsandbytes pillow

Usage:
  python3 training/finetune_qwen_vl.py                           # Train with defaults
  python3 training/finetune_qwen_vl.py --config config.yaml      # Use config settings
  python3 training/finetune_qwen_vl.py --epochs 5 --lr 2e-4      # Override settings
  python3 training/finetune_qwen_vl.py --merge-only               # Just merge LoRA → full model
  python3 training/finetune_qwen_vl.py --eval                     # Evaluate on test split
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_config(config_path=None):
    """Load fine-tuning config from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("finetune", {})
    except Exception:
        return {}


def prepare_dataset(data_path, test_split=0.1):
    """Load fine-tuning data and split into train/test."""
    with open(data_path) as f:
        examples = json.load(f)

    if not examples:
        print("ERROR: No training examples found.")
        sys.exit(1)

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(examples)

    split_idx = max(1, int(len(examples) * (1 - test_split)))
    train = examples[:split_idx]
    test = examples[split_idx:]

    print(f"  Dataset: {len(examples)} total → {len(train)} train, {len(test)} test")
    return train, test


def setup_model_and_tokenizer(base_model, lora_config, use_qlora=False):
    """Load base model with LoRA configuration."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    print(f"  Loading base model: {base_model}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, **model_kwargs
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def format_example_for_training(example, processor):
    """Convert a training example to model input format."""
    from PIL import Image
    import torch

    messages = example["messages"]
    user_msg = messages[0]
    assistant_msg = messages[1]

    # Extract image path from user message
    image = None
    text_parts = []
    for part in user_msg["content"]:
        if part.get("type") == "text":
            text_parts.append(part["text"])
        elif part.get("type") == "image":
            img_path = part["image"].replace("file://", "")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")

    # Build the full conversation text
    user_text = "\n".join(text_parts)

    # Use the processor to create inputs
    conversation = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_msg["content"]},
    ]

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )

    if image is not None:
        inputs = processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True, truncation=True,
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt", padding=True, truncation=True,
        )

    return inputs


class HealthClaimDataset:
    """Simple dataset for health claim form training."""

    def __init__(self, examples, processor, max_length=4096):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return format_example_for_training(self.examples[idx], self.processor)


def train(model, processor, train_data, test_data, training_config, output_dir):
    """Run LoRA fine-tuning."""
    from transformers import TrainingArguments, Trainer
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Save training data alongside checkpoints for reproducibility
    with open(os.path.join(output_dir, "train_data.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", True),
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch" if test_data else "no",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",  # Disable wandb etc
    )

    train_dataset = HealthClaimDataset(train_data, processor)
    eval_dataset = HealthClaimDataset(test_data, processor) if test_data else None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"\n  Starting training...")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Grad accum: {args.gradient_accumulation_steps}")
    print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    trainer.train()

    # Save LoRA adapter
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"\n  LoRA adapter saved to: {adapter_path}")

    return adapter_path


def merge_lora(base_model, adapter_path, merged_dir):
    """Merge LoRA weights into the base model for vLLM serving."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    import torch

    print(f"  Loading base model: {base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to: {merged_dir}")
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)

    # Also save processor
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    print(f"  Merged model ready!")
    print(f"\n  To serve with vLLM:")
    print(f"    bash deploy/start_vllm.sh --model {merged_dir} --quant null --dtype bfloat16")

    return merged_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for health claims")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data", default=None,
                        help="Training data JSON (from export_finetune.py)")
    parser.add_argument("--base-model", default=None,
                        help="Base model (default: Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--method", choices=["lora", "qlora"], default=None,
                        help="Fine-tuning method")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--output-dir", default=None,
                        help="Checkpoint output directory")
    parser.add_argument("--merged-dir", default=None,
                        help="Merged model output directory")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge LoRA → full model (skip training)")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter (for --merge-only)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    cfg = load_config(args.config)

    # Resolve settings (CLI > config > defaults)
    base_model = args.base_model or cfg.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    method = args.method or cfg.get("method", "lora")
    output_dir = args.output_dir or cfg.get("output_dir", str(project_dir / "training" / "checkpoints"))
    merged_dir = args.merged_dir or cfg.get("merged_dir", str(project_dir / "training" / "merged_model"))

    lora_config = cfg.get("lora", {
        "r": 64, "alpha": 128, "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    })

    training_config = cfg.get("training", {
        "epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4, "warmup_ratio": 0.1, "bf16": True,
    })

    # CLI overrides
    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.lr:
        training_config["learning_rate"] = args.lr

    print("=" * 60)
    print("  Qwen2.5-VL Fine-Tuning for Health Claim Forms")
    print("=" * 60)

    # Merge-only mode
    if args.merge_only:
        adapter = args.adapter_path or os.path.join(output_dir, "lora_adapter")
        if not os.path.exists(adapter):
            print(f"ERROR: Adapter not found at {adapter}")
            print(f"Train first or pass --adapter-path")
            sys.exit(1)
        merge_lora(base_model, adapter, merged_dir)
        return

    # Load training data
    data_path = args.data
    if not data_path:
        # Try default paths
        for candidate in [
            project_dir / "training" / "finetune_data_qwen.json",
            project_dir / "training" / "finetune_data_sharegpt.json",
        ]:
            if candidate.exists():
                data_path = str(candidate)
                break

    if not data_path or not os.path.exists(data_path):
        print("ERROR: No training data found.")
        print("Run these steps first:")
        print("  1. python3 training/collect_training_data.py --pdf-dir /path/to/pdfs")
        print("  2. python3 training/annotate.py")
        print("  3. python3 training/export_finetune.py")
        sys.exit(1)

    print(f"  Base model: {base_model}")
    print(f"  Method: {method}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print()

    # Load data
    train_data, test_data = prepare_dataset(data_path)

    # Setup model
    use_qlora = method == "qlora"
    model, processor = setup_model_and_tokenizer(base_model, lora_config, use_qlora)

    # Train
    adapter_path = train(model, processor, train_data, test_data,
                         training_config, output_dir)

    # Merge
    print("\n" + "=" * 60)
    print("  Merging LoRA weights into base model...")
    print("=" * 60)
    merge_lora(base_model, adapter_path, merged_dir)

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print("=" * 60)
    print(f"  LoRA adapter: {adapter_path}")
    print(f"  Merged model: {merged_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Serve: bash deploy/start_vllm.sh --model {merged_dir} --quant null --dtype bfloat16")
    print(f"    2. Test:  python3 qwen_vl_extract.py <pdf> -o result.json")
    print(f"    3. Eval:  python3 tests/check_accuracy.py --ocr-output result.json --ground-truth tests/ground_truth_health_claim.json")


if __name__ == "__main__":
    main()
