#!/usr/bin/env python3
"""
export_finetune.py — Export annotated training samples to Qwen2.5-VL fine-tuning format

Takes reviewed/approved training samples and converts them into the
conversation format required by Qwen2.5-VL fine-tuning:

  [
    {
      "messages": [
        {"role": "user", "content": [
          {"type": "text", "text": "Extract these fields..."},
          {"type": "image", "image": "file:///path/to/page.png"}
        ]},
        {"role": "assistant", "content": '{"field": "value", ...}'}
      ]
    },
    ...
  ]

Each page becomes a separate training example with its ground-truth fields.

Usage:
  python3 training/export_finetune.py
  python3 training/export_finetune.py --min-status reviewed --output training/finetune_data.json
  python3 training/export_finetune.py --format sharegpt  # ShareGPT format for LLaMA-Factory
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_vl_extract import PAGE_FIELD_MAP


def build_ground_truth_for_page(page_num, extracted_data, corrections):
    """Build the corrected ground truth for a specific page's fields."""
    fields = PAGE_FIELD_MAP.get(page_num, {}).get("fields", {})
    result = {}

    for field_key in fields:
        # Use correction if available, otherwise extracted value
        if field_key in corrections:
            result[field_key] = corrections[field_key]
        elif field_key in extracted_data:
            result[field_key] = extracted_data[field_key]

    return result


def build_page_prompt(page_num):
    """Build the extraction prompt for a page (mirrors qwen_vl_extract.py)."""
    info = PAGE_FIELD_MAP[page_num]
    hint = info["section_hint"]
    fields = info["fields"]

    field_lines = "\n".join(f'  - "{k}": {v}' for k, v in fields.items())
    field_keys = list(fields.keys())

    return (
        f"You are an expert at reading scanned insurance claim forms.\n"
        f"This is page {page_num} of a health insurance claim form.\n"
        f"Page description: {hint}\n\n"
        f"Extract ONLY these fields from the image:\n{field_lines}\n\n"
        f"Rules:\n"
        f"- Return ONLY a JSON object with these exact keys: {json.dumps(field_keys)}\n"
        f"- For amounts/financials: return digits only, no commas or currency symbols\n"
        f"- For dates: return as DD/MM/YY or DD/MM/YYYY exactly as written\n"
        f"- For PAN: return exactly 10 uppercase alphanumeric characters\n"
        f"- For IFSC: return exactly as written (letters + digits)\n"
        f"- For mobile: return exactly 10 digits\n"
        f"- Read handwritten values very carefully — zoom in mentally\n"
        f"- Use null if a field is truly not present on this page\n\n"
        f"JSON:"
    )


def export_qwen_format(samples, output_path, image_base_dir):
    """Export to Qwen2.5-VL conversation format."""
    training_examples = []

    for sample in samples:
        extracted = sample.get("extracted_data", {})
        corrections = sample.get("corrections", {})
        page_images = sample.get("page_images", {})

        for page_num in sorted(PAGE_FIELD_MAP.keys()):
            pn_str = str(page_num)
            if pn_str not in page_images and page_num not in page_images:
                continue

            # Get image path
            img_file = page_images.get(pn_str) or page_images.get(page_num)
            if not img_file:
                continue
            img_path = os.path.join(image_base_dir, img_file)
            if not os.path.exists(img_path):
                continue

            # Build ground truth (corrected) for this page
            gt = build_ground_truth_for_page(page_num, extracted, corrections)
            if not gt:
                continue

            # Build conversation
            prompt = build_page_prompt(page_num)
            answer = json.dumps(gt, ensure_ascii=False)

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            }
            training_examples.append(example)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)

    return len(training_examples)


def export_sharegpt_format(samples, output_path, image_base_dir):
    """Export to ShareGPT format (used by LLaMA-Factory)."""
    training_examples = []

    for sample in samples:
        extracted = sample.get("extracted_data", {})
        corrections = sample.get("corrections", {})
        page_images = sample.get("page_images", {})

        for page_num in sorted(PAGE_FIELD_MAP.keys()):
            pn_str = str(page_num)
            img_file = page_images.get(pn_str) or page_images.get(page_num)
            if not img_file:
                continue
            img_path = os.path.join(image_base_dir, img_file)
            if not os.path.exists(img_path):
                continue

            gt = build_ground_truth_for_page(page_num, extracted, corrections)
            if not gt:
                continue

            prompt = build_page_prompt(page_num)
            answer = json.dumps(gt, ensure_ascii=False)

            example = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": answer},
                ],
                "images": [os.path.abspath(img_path)],
            }
            training_examples.append(example)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)

    return len(training_examples)


def main():
    parser = argparse.ArgumentParser(
        description="Export training data to fine-tuning format"
    )
    parser.add_argument("--data-dir", default=None,
                        help="Training data directory")
    parser.add_argument("--output", default=None,
                        help="Output file path")
    parser.add_argument("--min-status", default="reviewed",
                        choices=["pending", "reviewed", "approved"],
                        help="Minimum annotation status to include")
    parser.add_argument("--format", default="qwen",
                        choices=["qwen", "sharegpt"],
                        help="Output format")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = args.data_dir or str(project_dir / "training" / "data")

    if args.output:
        output_path = args.output
    else:
        output_path = str(project_dir / "training" / f"finetune_data_{args.format}.json")

    # Load samples with sufficient annotation status
    status_order = {"pending": 0, "reviewed": 1, "approved": 2}
    min_level = status_order.get(args.min_status, 1)

    samples = []
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(data_dir, f)) as fp:
                sample = json.load(fp)
            sample_level = status_order.get(sample.get("status", "pending"), 0)
            if sample_level >= min_level:
                samples.append(sample)

    if not samples:
        print(f"No samples with status >= '{args.min_status}' found in {data_dir}")
        print(f"Run 'python3 training/annotate.py' to review samples first.")
        sys.exit(1)

    print(f"Exporting {len(samples)} samples to {args.format} format...")

    if args.format == "qwen":
        count = export_qwen_format(samples, output_path, data_dir)
    else:
        count = export_sharegpt_format(samples, output_path, data_dir)

    print(f"Exported {count} training examples to: {output_path}")
    print(f"  (from {len(samples)} annotated samples, ~{count//len(samples)} pages each)")


if __name__ == "__main__":
    main()
