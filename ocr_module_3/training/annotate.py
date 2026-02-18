#!/usr/bin/env python3
"""
annotate.py — CLI tool for reviewing and correcting training samples

Walks through training samples in training/data/, displays extracted values,
and lets you correct any mistakes. Corrected samples are marked as "reviewed"
and can be exported for fine-tuning.

Usage:
  python3 training/annotate.py                    # Review all pending
  python3 training/annotate.py --status pending   # Only pending samples
  python3 training/annotate.py --sample-id abc123 # Review specific sample
  python3 training/annotate.py --stats            # Show annotation stats
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_samples(data_dir, status_filter=None, sample_id=None):
    """Load training samples from data directory."""
    samples = []
    if not os.path.exists(data_dir):
        return samples

    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(data_dir, f)
        with open(path) as fp:
            sample = json.load(fp)
        sample["_path"] = path

        if sample_id and sample.get("id") != sample_id:
            continue
        if status_filter and sample.get("status") != status_filter:
            continue
        samples.append(sample)

    return samples


def print_sample(sample, show_gt=True):
    """Display a training sample for review."""
    print(f"\n{'='*60}")
    print(f"  Sample: {sample['id']}")
    print(f"  PDF: {sample.get('source_pdf', '?')}")
    print(f"  Status: {sample.get('status', '?')}")
    print(f"  Model: {sample.get('model', '?')}")
    print(f"  Pages: {sample.get('pages_processed', '?')}")
    print(f"{'='*60}")

    extracted = sample.get("extracted_data", {})
    ground_truth = sample.get("ground_truth", {})
    corrections = sample.get("corrections", {})

    # Flatten ground truth for comparison
    gt_flat = {}
    if ground_truth and "fields" in ground_truth:
        _flatten_dict(ground_truth["fields"], gt_flat)

    print(f"\n{'FIELD':<40} {'EXTRACTED':<30} {'GROUND TRUTH':<25} {'STATUS'}")
    print("-" * 120)

    for field in sorted(extracted.keys()):
        if field.startswith("_"):
            continue
        ext_val = str(extracted[field]) if extracted[field] is not None else "(null)"
        gt_val = gt_flat.get(field, "")
        corrected = corrections.get(field)

        if corrected:
            status = f"CORRECTED → {corrected}"
        elif gt_val and gt_val.lower() == ext_val.lower():
            status = "MATCH"
        elif gt_val:
            status = "MISMATCH"
        else:
            status = ""

        # Truncate long values
        ext_display = ext_val[:28] if len(ext_val) > 28 else ext_val
        gt_display = gt_val[:23] if len(str(gt_val)) > 23 else str(gt_val)
        print(f"  {field:<38} {ext_display:<30} {gt_display:<25} {status}")


def _flatten_dict(d, result, prefix=""):
    """Flatten nested dict, keeping leaf values keyed by their field name."""
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten_dict(v, result, f"{prefix}{k}.")
        else:
            # Use the leaf key name (e.g., "hospital_name" not the full path)
            result[k] = str(v) if v is not None else ""


def annotate_sample(sample):
    """Interactive annotation of a single sample."""
    print_sample(sample)

    extracted = sample.get("extracted_data", {})
    corrections = sample.get("corrections", {})

    print(f"\nCommands:")
    print(f"  <field> = <value>  — Correct a field (e.g., pan = ASFPP0101Q)")
    print(f"  approve            — Mark as approved (no more corrections needed)")
    print(f"  skip               — Skip to next sample")
    print(f"  done               — Save and exit")
    print()

    while True:
        try:
            inp = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "done"

        if not inp:
            continue

        if inp.lower() == "approve":
            sample["status"] = "approved"
            _save_sample(sample)
            print(f"  Marked as approved.")
            return "next"

        if inp.lower() == "skip":
            if corrections:
                sample["status"] = "reviewed"
                _save_sample(sample)
            return "next"

        if inp.lower() == "done":
            if corrections:
                sample["status"] = "reviewed"
                _save_sample(sample)
            return "done"

        if inp.lower() == "show":
            print_sample(sample)
            continue

        # Parse correction: field = value
        if "=" in inp:
            parts = inp.split("=", 1)
            field = parts[0].strip()
            value = parts[1].strip()

            if field in extracted:
                corrections[field] = value
                sample["corrections"] = corrections
                print(f"  Corrected: {field} = {value}")
            else:
                # Fuzzy match
                matches = [f for f in extracted if field.lower() in f.lower()]
                if len(matches) == 1:
                    corrections[matches[0]] = value
                    sample["corrections"] = corrections
                    print(f"  Corrected: {matches[0]} = {value}")
                elif matches:
                    print(f"  Multiple matches: {', '.join(matches)}")
                else:
                    print(f"  Field not found: {field}")
                    print(f"  Available: {', '.join(sorted(extracted.keys()))}")
        else:
            print(f"  Unknown command. Use 'field = value', 'approve', 'skip', or 'done'")


def _save_sample(sample):
    """Save sample back to its JSON file."""
    path = sample.pop("_path", None)
    if path:
        with open(path, "w") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        sample["_path"] = path


def print_stats(data_dir):
    """Print annotation statistics."""
    samples = load_samples(data_dir)
    total = len(samples)
    by_status = {}
    for s in samples:
        status = s.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    print(f"\n  Training Data Statistics")
    print(f"  Directory: {data_dir}")
    print(f"  Total samples: {total}")
    for status, count in sorted(by_status.items()):
        pct = count / total * 100 if total else 0
        print(f"    {status}: {count} ({pct:.0f}%)")

    # Count corrections
    total_corrections = sum(
        len(s.get("corrections", {})) for s in samples
    )
    print(f"  Total field corrections: {total_corrections}")


def main():
    parser = argparse.ArgumentParser(description="Annotate training samples")
    parser.add_argument("--data-dir", default=None,
                        help="Training data directory")
    parser.add_argument("--status", default="pending",
                        help="Filter by status (pending/reviewed/approved/all)")
    parser.add_argument("--sample-id", default=None,
                        help="Review a specific sample by ID")
    parser.add_argument("--stats", action="store_true",
                        help="Show annotation statistics")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = args.data_dir or str(project_dir / "training" / "data")

    if args.stats:
        print_stats(data_dir)
        return

    status_filter = None if args.status == "all" else args.status
    samples = load_samples(data_dir, status_filter=status_filter,
                           sample_id=args.sample_id)

    if not samples:
        print(f"No {'matching ' if status_filter else ''}samples found in {data_dir}")
        return

    print(f"Found {len(samples)} samples to review")

    for i, sample in enumerate(samples, 1):
        print(f"\n  [{i}/{len(samples)}]")
        action = annotate_sample(sample)
        if action == "done":
            break

    print(f"\nDone. Run with --stats to see annotation progress.")


if __name__ == "__main__":
    main()
