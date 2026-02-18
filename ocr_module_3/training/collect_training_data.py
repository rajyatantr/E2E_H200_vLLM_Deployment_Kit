#!/usr/bin/env python3
"""
collect_training_data.py — Batch inference + training data collection

Runs the extraction pipeline on a directory of PDFs, saving each result
as a training sample (page images + extracted fields + metadata).

Training samples are stored in training/data/ as:
  {doc_hash}_{timestamp}.json   — extraction result + metadata
  {doc_hash}_page{N}.png        — rendered page images (if save_page_images=true)

These samples can then be:
  1. Reviewed/corrected with annotate.py
  2. Exported to fine-tuning format with export_finetune.py
  3. Used to fine-tune with finetune_qwen_vl.py

Usage:
  python3 training/collect_training_data.py --pdf-dir /path/to/pdfs
  python3 training/collect_training_data.py --pdf /single/file.pdf
  python3 training/collect_training_data.py --pdf-dir /pdfs --ground-truth-dir /gt
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_vl_extract import run_pipeline, pass1_render_pdf, load_config


def hash_file(filepath, algo="sha256"):
    """Quick hash of file contents for deduplication."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_training_sample(pdf_path, result, page_images, output_dir,
                         save_images=True, ground_truth=None):
    """Save one extraction result as a training sample."""
    os.makedirs(output_dir, exist_ok=True)

    doc_hash = hash_file(pdf_path)
    timestamp = int(time.time())
    base_name = f"{doc_hash}_{timestamp}"

    # Save page images as PNG files
    image_files = {}
    if save_images:
        for img in page_images:
            pn = img["page_num"]
            img_path = os.path.join(output_dir, f"{base_name}_page{pn}.png")
            png_bytes = base64.b64decode(img["base64_png"])
            with open(img_path, "wb") as f:
                f.write(png_bytes)
            image_files[pn] = f"{base_name}_page{pn}.png"

    # Build training sample
    sample = {
        "id": base_name,
        "source_pdf": os.path.basename(pdf_path),
        "source_hash": doc_hash,
        "timestamp": timestamp,
        "status": "pending",  # pending → reviewed → approved
        "extracted_data": result.get("structured_data", {}),
        "ground_truth": ground_truth,  # None until annotated
        "corrections": {},  # field → corrected_value (filled during annotation)
        "page_images": image_files,
        "pages_processed": result.get("pages_processed", 0),
        "model": result.get("model", ""),
        "pipeline": result.get("pipeline", ""),
        "timing": result.get("timing", {}),
        "validation_failures": result.get("validation_failures", {}),
    }

    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    return json_path


def load_ground_truth(gt_dir, pdf_name):
    """Try to find a matching ground truth file for a PDF."""
    if not gt_dir:
        return None

    # Try exact name match (minus extension)
    stem = Path(pdf_name).stem
    candidates = [
        os.path.join(gt_dir, f"ground_truth_{stem}.json"),
        os.path.join(gt_dir, f"{stem}.json"),
        os.path.join(gt_dir, f"{stem}_gt.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data by running extraction on PDFs"
    )
    parser.add_argument("--pdf", help="Single PDF file to process")
    parser.add_argument("--pdf-dir", help="Directory of PDF files to process")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: training/data)")
    parser.add_argument("--ground-truth-dir", default=None,
                        help="Directory with ground truth JSON files")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-images", action="store_true",
                        help="Don't save page images (saves disk space)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip PDFs that already have training samples")
    args = parser.parse_args()

    if not args.pdf and not args.pdf_dir:
        parser.error("Provide --pdf or --pdf-dir")

    # Determine output directory
    project_dir = Path(__file__).parent.parent
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(project_dir / "training" / "data")

    # Collect PDF paths
    pdf_paths = []
    if args.pdf:
        pdf_paths.append(args.pdf)
    if args.pdf_dir:
        for f in sorted(os.listdir(args.pdf_dir)):
            if f.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(args.pdf_dir, f))

    if not pdf_paths:
        print("No PDF files found.")
        sys.exit(1)

    # Check for existing samples if --skip-existing
    existing_hashes = set()
    if args.skip_existing and os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".json"):
                existing_hashes.add(f.split("_")[0])

    print("=" * 60)
    print("  Training Data Collection")
    print("=" * 60)
    print(f"  PDFs to process: {len(pdf_paths)}")
    print(f"  Output dir: {output_dir}")
    print(f"  Save images: {not args.no_images}")
    print(f"  Ground truth dir: {args.ground_truth_dir or 'None'}")
    print()

    # Load config
    cfg = load_config(args.config)

    results = {"processed": 0, "skipped": 0, "errors": 0}

    for i, pdf_path in enumerate(pdf_paths, 1):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n[{i}/{len(pdf_paths)}] {pdf_name}")

        # Skip if already processed
        if args.skip_existing:
            doc_hash = hash_file(pdf_path)
            if doc_hash in existing_hashes:
                print(f"  Skipping (already collected)")
                results["skipped"] += 1
                continue

        try:
            # Run extraction
            result = run_pipeline(pdf_path, config=cfg)

            # Re-render pages to save images (run_pipeline doesn't return them)
            page_images = []
            if not args.no_images:
                page_images, _ = pass1_render_pdf(
                    pdf_path,
                    target_long_edge=cfg.get("extraction", {}).get("target_long_edge", 1792)
                )

            # Load ground truth if available
            gt = load_ground_truth(args.ground_truth_dir, pdf_name)

            # Save training sample
            json_path = save_training_sample(
                pdf_path, result, page_images, output_dir,
                save_images=not args.no_images,
                ground_truth=gt,
            )
            print(f"  Saved: {json_path}")
            results["processed"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            results["errors"] += 1

    print("\n" + "=" * 60)
    print(f"  Done: {results['processed']} processed, "
          f"{results['skipped']} skipped, {results['errors']} errors")
    print(f"  Training data: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
