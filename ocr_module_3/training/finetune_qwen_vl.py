#!/usr/bin/env python3
"""
finetune_qwen_vl.py — Fine-tune Qwen2.5-VL on health claim form data

Supports both 7B and 72B models using LoRA / QLoRA:

  ┌─────────────┬────────────┬─────────────┬──────────────────────┐
  │ Model       │ Method     │ H200 VRAM   │ Use Case             │
  ├─────────────┼────────────┼─────────────┼──────────────────────┤
  │ 7B          │ LoRA       │ ~30 GB      │ Fast experiments     │
  │ 7B          │ QLoRA      │ ~15 GB      │ Multi-GPU / smaller  │
  │ 72B         │ QLoRA      │ ~100 GB     │ Best accuracy (H200) │
  │ 72B         │ LoRA       │ ~280 GB     │ Needs multi-node     │
  └─────────────┴────────────┴─────────────┴──────────────────────┘

72B QLoRA on H200 (141GB):
  - Base model in 4-bit NF4 (bitsandbytes): ~38GB
  - LoRA adapters (r=64, bf16):              ~3GB
  - Optimizer states (AdamW):                ~6GB
  - Activations (gradient checkpointing):    ~40-60GB
  - Total: ~90-110GB → FITS on H200

After fine-tuning:
  1. Merge LoRA:    python3 finetune_qwen_vl.py --merge-only
  2. Quantize AWQ:  python3 finetune_qwen_vl.py --quantize-awq  (optional, for fast inference)
  3. Serve:         bash deploy/start_vllm.sh --model ./training/merged_model

Requirements:
  pip install transformers peft accelerate bitsandbytes pillow

Usage:
  # Fine-tune 72B with QLoRA (recommended for H200):
  python3 training/finetune_qwen_vl.py --base-model Qwen/Qwen2.5-VL-72B-Instruct --method qlora

  # Fine-tune 7B with LoRA (faster, good for experiments):
  python3 training/finetune_qwen_vl.py --base-model Qwen/Qwen2.5-VL-7B-Instruct --method lora

  # Merge LoRA adapter into full model:
  python3 training/finetune_qwen_vl.py --merge-only

  # Quantize merged model to AWQ for fast vLLM inference:
  python3 training/finetune_qwen_vl.py --quantize-awq
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

    import random
    random.seed(42)
    random.shuffle(examples)

    split_idx = max(1, int(len(examples) * (1 - test_split)))
    train = examples[:split_idx]
    test = examples[split_idx:]

    print(f"  Dataset: {len(examples)} total → {len(train)} train, {len(test)} test")
    return train, test


def estimate_vram(base_model, method, lora_r=64):
    """Estimate VRAM usage for a given configuration."""
    is_72b = "72" in base_model.lower()
    param_b = 72 if is_72b else 7

    if method == "qlora":
        model_gb = param_b * 0.5  # 4-bit ≈ 0.5 GB per billion params
        adapter_gb = param_b * 0.04 * (lora_r / 64)  # LoRA in bf16
        optim_gb = adapter_gb * 2  # AdamW states
        act_gb = param_b * 0.7  # With gradient checkpointing
    else:  # lora (bf16 base)
        model_gb = param_b * 2  # bf16 ≈ 2 GB per billion params
        adapter_gb = param_b * 0.04 * (lora_r / 64)
        optim_gb = adapter_gb * 2
        act_gb = param_b * 0.7

    total = model_gb + adapter_gb + optim_gb + act_gb
    return {
        "model_gb": round(model_gb, 1),
        "adapter_gb": round(adapter_gb, 1),
        "optimizer_gb": round(optim_gb, 1),
        "activations_gb": round(act_gb, 1),
        "total_gb": round(total, 1),
    }


def setup_model_and_tokenizer(base_model, lora_config, method="lora",
                                gradient_checkpointing=True):
    """Load base model with LoRA/QLoRA configuration."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    use_qlora = method == "qlora"
    print(f"  Loading base model: {base_model}")
    print(f"  Method: {'QLoRA (4-bit NF4 + LoRA)' if use_qlora else 'LoRA (bf16 base)'}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, **model_kwargs
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

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

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"  Parameters: {total/1e9:.1f}B total, {trainable/1e6:.1f}M trainable "
          f"({trainable/total*100:.2f}%)")

    # Print actual VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    return model, processor


def format_example_for_training(example, processor):
    """Convert a training example to model input format."""
    from PIL import Image

    messages = example["messages"]
    user_msg = messages[0]
    assistant_msg = messages[1]

    image = None
    text_parts = []
    for part in user_msg["content"]:
        if part.get("type") == "text":
            text_parts.append(part["text"])
        elif part.get("type") == "image":
            img_path = part["image"].replace("file://", "")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")

    user_text = "\n".join(text_parts)

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
    """Dataset for health claim form training."""

    def __init__(self, examples, processor, max_length=4096):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return format_example_for_training(self.examples[idx], self.processor)


def train(model, processor, train_data, test_data, training_config, output_dir):
    """Run LoRA/QLoRA fine-tuning."""
    from transformers import TrainingArguments, Trainer

    os.makedirs(output_dir, exist_ok=True)

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
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch" if test_data else "no",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
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
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print()

    trainer.train()

    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"\n  LoRA adapter saved to: {adapter_path}")

    return adapter_path


def merge_lora(base_model, adapter_path, merged_dir):
    """Merge LoRA weights back into the full-precision base model.

    Note: For 72B, this requires ~145GB RAM (loads full bf16 model).
    Use a high-RAM machine or add --merge-offload for CPU offloading.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    import torch

    print(f"  Loading base model (bf16): {base_model}")
    print(f"  NOTE: 72B merge requires ~145GB RAM. Using CPU offload if needed.")

    # For 72B, try to load with CPU offload to avoid OOM
    is_72b = "72" in base_model.lower()
    if is_72b:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",  # Load on CPU for merge
            low_cpu_mem_usage=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to: {merged_dir}")
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, max_shard_size="5GB")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    print(f"  Merged model ready!")
    return merged_dir


def quantize_awq(merged_dir, awq_output_dir):
    """Quantize merged model to AWQ format for fast vLLM inference.

    Requires: pip install autoawq
    """
    print(f"\n  Quantizing to AWQ format...")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {awq_output_dir}")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoProcessor
    except ImportError:
        print("ERROR: autoawq not installed. Run: pip install autoawq")
        sys.exit(1)

    model = AutoAWQForCausalLM.from_pretrained(merged_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"  Quantizing (this takes 30-60 minutes for 72B)...")
    model.quantize(processor.tokenizer, quant_config=quant_config)

    os.makedirs(awq_output_dir, exist_ok=True)
    model.save_quantized(awq_output_dir)
    processor.save_pretrained(awq_output_dir)

    print(f"  AWQ model saved to: {awq_output_dir}")
    print(f"\n  To serve with vLLM:")
    print(f"    bash deploy/start_vllm.sh --model {awq_output_dir} --quant awq_marlin --dtype float16")

    return awq_output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for health claims")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data", default=None,
                        help="Training data JSON (from export_finetune.py)")
    parser.add_argument("--base-model", default=None,
                        help="Base model (default from config.yaml)")
    parser.add_argument("--method", choices=["lora", "qlora"], default=None,
                        help="Fine-tuning method (qlora recommended for 72B)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="LoRA rank (default: 64)")
    parser.add_argument("--output-dir", default=None,
                        help="Checkpoint output directory")
    parser.add_argument("--merged-dir", default=None,
                        help="Merged model output directory")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge LoRA → full model (skip training)")
    parser.add_argument("--quantize-awq", action="store_true",
                        help="Quantize merged model to AWQ for fast vLLM serving")
    parser.add_argument("--awq-output", default=None,
                        help="AWQ model output directory")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter (for --merge-only)")
    parser.add_argument("--estimate-vram", action="store_true",
                        help="Just print VRAM estimate and exit")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    cfg = load_config(args.config)

    # Resolve settings (CLI > config > defaults)
    base_model = args.base_model or cfg.get("base_model", "Qwen/Qwen2.5-VL-72B-Instruct")
    output_dir = args.output_dir or cfg.get("output_dir", str(project_dir / "training" / "checkpoints"))
    merged_dir = args.merged_dir or cfg.get("merged_dir", str(project_dir / "training" / "merged_model"))
    awq_dir = args.awq_output or str(project_dir / "training" / "awq_model")

    # Auto-detect method: qlora for 72B, lora for 7B
    is_72b = "72" in base_model.lower()
    default_method = "qlora" if is_72b else "lora"
    method = args.method or cfg.get("method", default_method)

    lora_config = cfg.get("lora", {
        "r": 64, "alpha": 128, "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    })
    if args.lora_r:
        lora_config["r"] = args.lora_r

    training_config = cfg.get("training", {
        "epochs": 3, "batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4, "warmup_ratio": 0.1, "bf16": True,
        "gradient_checkpointing": True,
    })
    # 72B needs smaller batch and more grad accumulation
    if is_72b and "batch_size" not in cfg.get("training", {}):
        training_config["batch_size"] = 1
        training_config["gradient_accumulation_steps"] = 16

    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.lr:
        training_config["learning_rate"] = args.lr

    print("=" * 60)
    print("  Qwen2.5-VL Fine-Tuning for Health Claim Forms")
    print("=" * 60)
    print(f"  Base model: {base_model}")
    print(f"  Method: {method}{' (auto-detected for 72B)' if method == default_method and is_72b else ''}")
    print(f"  LoRA rank: {lora_config.get('r', 64)}")
    print()

    # VRAM estimate
    vram = estimate_vram(base_model, method, lora_config.get("r", 64))
    print(f"  Estimated VRAM:")
    print(f"    Model:       {vram['model_gb']:.1f} GB {'(4-bit NF4)' if method == 'qlora' else '(bf16)'}")
    print(f"    LoRA:        {vram['adapter_gb']:.1f} GB (bf16)")
    print(f"    Optimizer:   {vram['optimizer_gb']:.1f} GB")
    print(f"    Activations: {vram['activations_gb']:.1f} GB (gradient checkpointing)")
    print(f"    Total:       ~{vram['total_gb']:.0f} GB")

    if args.estimate_vram:
        return

    if vram["total_gb"] > 140:
        print(f"\n  WARNING: Estimated {vram['total_gb']:.0f}GB exceeds H200's 141GB!")
        print(f"  Consider: --method qlora, smaller --lora-r, or --base-model 7B")
    print()

    # AWQ quantization mode
    if args.quantize_awq:
        if not os.path.exists(merged_dir):
            print(f"ERROR: Merged model not found at {merged_dir}")
            print(f"Run --merge-only first.")
            sys.exit(1)
        quantize_awq(merged_dir, awq_dir)
        return

    # Merge-only mode
    if args.merge_only:
        adapter = args.adapter_path or os.path.join(output_dir, "lora_adapter")
        if not os.path.exists(adapter):
            print(f"ERROR: Adapter not found at {adapter}")
            print(f"Train first or pass --adapter-path")
            sys.exit(1)
        merge_lora(base_model, adapter, merged_dir)
        print(f"\n  Next: re-quantize to AWQ for fast serving:")
        print(f"    python3 training/finetune_qwen_vl.py --quantize-awq")
        return

    # Load training data
    data_path = args.data
    if not data_path:
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

    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print()

    # Load data
    train_data, test_data = prepare_dataset(data_path)

    # Setup model
    model, processor = setup_model_and_tokenizer(
        base_model, lora_config, method=method,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
    )

    # Train
    adapter_path = train(model, processor, train_data, test_data,
                         training_config, output_dir)

    # Free training memory before merge
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Merge
    print("\n" + "=" * 60)
    print("  Merging LoRA weights into base model...")
    print("=" * 60)
    merge_lora(base_model, adapter_path, merged_dir)

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print("=" * 60)
    print(f"  LoRA adapter:  {adapter_path}")
    print(f"  Merged model:  {merged_dir}")
    print()
    if is_72b:
        print(f"  For fast inference, quantize to AWQ:")
        print(f"    python3 training/finetune_qwen_vl.py --quantize-awq")
        print(f"    bash deploy/start_vllm.sh --model {awq_dir} --quant awq_marlin --dtype float16")
    else:
        print(f"  Serve with vLLM:")
        print(f"    bash deploy/start_vllm.sh --model {merged_dir} --quant null --dtype bfloat16")
    print()
    print(f"  Test:")
    print(f"    python3 qwen_vl_extract.py <pdf> -o result.json")
    print(f"    python3 tests/check_accuracy.py --ocr-output result.json --ground-truth tests/ground_truth_health_claim.json")


if __name__ == "__main__":
    main()
