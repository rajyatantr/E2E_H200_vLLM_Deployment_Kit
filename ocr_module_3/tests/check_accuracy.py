"""
OCR Accuracy Checker for Health Claim Forms

Compares OCR output text against ground truth fields, tables, and financial values.
Reports character-level, word-level, field-level, table-level, and financial accuracy.

Usage:
    # From saved OCR output file:
    python check_accuracy.py --ocr-output ocr_output.txt

    # Direct from API:
    python check_accuracy.py --api-url http://localhost:8093 --pdf "Health Claim form.pdf"

    # With custom ground truth and report saving:
    python check_accuracy.py --ocr-output ocr_output.txt --ground-truth custom_gt.json --save-report report.json
"""

import json
import argparse
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text


def character_similarity(a: str, b: str) -> float:
    """Character-level similarity (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def fuzzy_match(needle: str, haystack: str, threshold: float = 0.75) -> Tuple[bool, float, str]:
    """
    Check if a ground truth value appears in OCR output text.
    Uses sliding window with fuzzy matching.

    Returns: (found, best_score, best_match_context)
    """
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)

    if not needle_norm:
        return True, 1.0, ""

    # Exact substring match
    if needle_norm in haystack_norm:
        return True, 1.0, needle

    # Sliding window fuzzy match
    needle_len = len(needle_norm)
    best_score = 0.0
    best_context = ""

    for window_mult in [1.0, 0.9, 1.1, 0.8, 1.2, 1.5]:
        window_size = max(1, int(needle_len * window_mult))
        for i in range(len(haystack_norm) - window_size + 1):
            window = haystack_norm[i:i + window_size]
            score = SequenceMatcher(None, needle_norm, window).ratio()
            if score > best_score:
                best_score = score
                start = max(0, i - 10)
                end = min(len(haystack), i + window_size + 10)
                best_context = haystack[start:end].strip()

    found = best_score >= threshold
    return found, best_score, best_context


def flatten_ground_truth(gt: dict, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten nested ground truth dict into (field_path, value) pairs."""
    results = []
    for key, value in gt.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(flatten_ground_truth(value, path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    results.append((f"{path}[{i}]", item))
                elif isinstance(item, dict):
                    results.extend(flatten_ground_truth(item, f"{path}[{i}]"))
        elif isinstance(value, str) and value.strip():
            results.append((path, value))
        elif isinstance(value, (int, float)):
            results.append((path, str(value)))
    return results


def print_header(title: str, char: str = "=", width: int = 100):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# =============================================================================
# FIELD ACCURACY CHECK
# =============================================================================

def check_field_accuracy(ocr_text: str, fields: dict, threshold: float) -> Dict:
    """Check individual field values against OCR output."""
    all_fields = flatten_ground_truth(fields)
    results = []
    found_count = 0
    total_sim = 0.0

    print_header("FIELD-LEVEL ACCURACY")
    print(f"\n{'FIELD':<60} {'EXPECTED':<22} {'SCORE':>6} {'STATUS':>7}")
    print("-" * 100)

    for field_path, expected in all_fields:
        found, score, context = fuzzy_match(expected, ocr_text, threshold)
        if found:
            found_count += 1
        total_sim += score

        status = "PASS" if found else "MISS"
        field_disp = field_path[-59:] if len(field_path) > 59 else field_path
        val_disp = expected[:21] if len(expected) > 21 else expected
        print(f"  {field_disp:<58} {val_disp:<22} {score:>5.0%} {status:>7}")

        results.append({
            "field": field_path,
            "expected": expected,
            "score": round(score, 4),
            "found": found,
            "context": context
        })

    total = len(all_fields)
    accuracy = (found_count / total * 100) if total > 0 else 0
    avg_sim = (total_sim / total * 100) if total > 0 else 0

    print(f"\n  Result: {found_count}/{total} fields found ({accuracy:.1f}%)")
    print(f"  Average character similarity: {avg_sim:.1f}%")

    return {
        "total": total,
        "found": found_count,
        "accuracy_pct": round(accuracy, 2),
        "avg_similarity_pct": round(avg_sim, 2),
        "details": results
    }


# =============================================================================
# TABLE ACCURACY CHECK
# =============================================================================

def check_table_accuracy(ocr_text: str, tables: dict, threshold: float) -> Dict:
    """Check table content accuracy — validates cell values and row structure."""
    print_header("TABLE-LEVEL ACCURACY")

    table_results = {}
    total_cells = 0
    found_cells = 0
    total_rows_detected = 0
    total_rows = 0

    for table_name, table_data in tables.items():
        desc = table_data.get("description", table_name)
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])

        print(f"\n  Table: {desc}")
        print(f"  Columns: {columns}")
        print(f"  Expected rows: {len(rows)}")
        print()

        row_results = []
        table_cells_found = 0
        table_cells_total = 0
        rows_detected = 0

        for r_idx, row in enumerate(rows):
            row_found_any = False
            cell_results = {}

            for col in columns:
                cell_value = row.get(col, "")
                if not cell_value or not cell_value.strip():
                    continue  # Skip empty cells

                table_cells_total += 1
                total_cells += 1

                found, score, context = fuzzy_match(cell_value, ocr_text, threshold)

                if found:
                    table_cells_found += 1
                    found_cells += 1
                    row_found_any = True

                cell_results[col] = {
                    "expected": cell_value,
                    "score": round(score, 4),
                    "found": found
                }

                status = "PASS" if found else "MISS"
                val_disp = cell_value[:30] if len(cell_value) > 30 else cell_value
                print(f"    Row {r_idx+1:>2} | {col:<15} | {val_disp:<32} | {score:>5.0%} {status}")

            if row_found_any:
                rows_detected += 1

            row_results.append({
                "row_index": r_idx,
                "cells": cell_results,
                "row_detected": row_found_any
            })

        total_rows += len(rows)
        total_rows_detected += rows_detected

        t_accuracy = (table_cells_found / table_cells_total * 100) if table_cells_total > 0 else 0
        r_accuracy = (rows_detected / len(rows) * 100) if rows else 0
        print(f"\n  Cell accuracy: {table_cells_found}/{table_cells_total} ({t_accuracy:.1f}%)")
        print(f"  Row detection: {rows_detected}/{len(rows)} ({r_accuracy:.1f}%)")

        table_results[table_name] = {
            "total_cells": table_cells_total,
            "cells_found": table_cells_found,
            "cell_accuracy_pct": round(t_accuracy, 2),
            "total_rows": len(rows),
            "rows_detected": rows_detected,
            "row_accuracy_pct": round(r_accuracy, 2),
            "rows": row_results
        }

    overall_cell_acc = (found_cells / total_cells * 100) if total_cells > 0 else 0
    overall_row_acc = (total_rows_detected / total_rows * 100) if total_rows > 0 else 0

    print_header("TABLE SUMMARY", "-", 60)
    print(f"  Total cell accuracy:  {found_cells}/{total_cells} ({overall_cell_acc:.1f}%)")
    print(f"  Total row detection:  {total_rows_detected}/{total_rows} ({overall_row_acc:.1f}%)")

    return {
        "total_cells": total_cells,
        "cells_found": found_cells,
        "cell_accuracy_pct": round(overall_cell_acc, 2),
        "total_rows": total_rows,
        "rows_detected": total_rows_detected,
        "row_accuracy_pct": round(overall_row_acc, 2),
        "tables": table_results
    }


# =============================================================================
# FINANCIAL VALUES CHECK (strict — must be exact)
# =============================================================================

def check_financial_accuracy(ocr_text: str, financial_values: list) -> Dict:
    """Check financial/monetary values with STRICT matching (exact numbers)."""
    print_header("FINANCIAL VALUES ACCURACY (STRICT)")

    results = []
    found_count = 0
    total = len(financial_values)

    print(f"\n{'FIELD':<40} {'EXPECTED':>12} {'PAGE':>5} {'STATUS':>8}")
    print("-" * 70)

    for fv in financial_values:
        field = fv["field"]
        value = str(fv["value"])
        page = fv.get("page", "?")

        # Strict: check if the exact number appears in OCR output
        # Allow for comma-separated formats (e.g., "1,01,150" or "101150" or "101,150")
        value_clean = value.replace(",", "").replace(" ", "")

        # Build regex patterns for the number
        patterns = [
            re.escape(value_clean),  # exact: 101150
        ]

        # Add comma variants
        if len(value_clean) >= 4:
            # Indian format: 1,01,150
            indian = ""
            digits = value_clean
            if len(digits) > 3:
                indian = digits[-3:]
                digits = digits[:-3]
                while digits:
                    indian = digits[-2:] + "," + indian if len(digits) >= 2 else digits + "," + indian
                    digits = digits[:-2]
            patterns.append(re.escape(indian))

            # Western format: 101,150
            western = ""
            d = value_clean
            while len(d) > 3:
                western = "," + d[-3:] + western
                d = d[:-3]
            western = d + western
            patterns.append(re.escape(western))

        ocr_clean = ocr_text.replace(",", "").replace(" ", "")
        found = value_clean in ocr_clean

        # Also check with commas in original text
        if not found:
            for pat in patterns:
                if re.search(pat, ocr_text):
                    found = True
                    break

        if found:
            found_count += 1

        status = "EXACT" if found else "MISS"
        print(f"  {field:<38} {value:>12}  p{page:<3}  {status:>8}")

        results.append({
            "field": field,
            "expected": value,
            "page": page,
            "found": found
        })

    accuracy = (found_count / total * 100) if total > 0 else 0

    print(f"\n  Financial accuracy: {found_count}/{total} ({accuracy:.1f}%)")

    if accuracy < 100:
        print("\n  WARNING: Financial values mismatch detected!")
        print("  This is CRITICAL for insurance claims processing.")
        missed = [r for r in results if not r["found"]]
        for r in missed:
            print(f"    MISSING: {r['field']} = {r['expected']} (page {r['page']})")

    return {
        "total": total,
        "found": found_count,
        "accuracy_pct": round(accuracy, 2),
        "all_exact": found_count == total,
        "details": results
    }


# =============================================================================
# HANDWRITTEN VALUES CHECK
# =============================================================================

def check_handwritten_accuracy(ocr_text: str, hw_values: list, threshold: float) -> Dict:
    """Check key handwritten values (the hardest for OCR)."""
    print_header("HANDWRITTEN VALUES ACCURACY")

    results = []
    found_count = 0
    total_score = 0.0

    print(f"\n{'VALUE':<45} {'SCORE':>6} {'STATUS':>8}")
    print("-" * 65)

    for value in hw_values:
        found, score, context = fuzzy_match(value, ocr_text, threshold)
        total_score += score
        if found:
            found_count += 1

        status = "PASS" if found else "MISS"
        val_disp = value[:44] if len(value) > 44 else value
        print(f"  {val_disp:<43} {score:>5.0%} {status:>8}")

        results.append({
            "value": value,
            "score": round(score, 4),
            "found": found,
            "context": context
        })

    total = len(hw_values)
    accuracy = (found_count / total * 100) if total > 0 else 0
    avg_score = (total_score / total * 100) if total > 0 else 0

    print(f"\n  Handwritten accuracy: {found_count}/{total} ({accuracy:.1f}%)")
    print(f"  Average similarity:   {avg_score:.1f}%")

    return {
        "total": total,
        "found": found_count,
        "accuracy_pct": round(accuracy, 2),
        "avg_similarity_pct": round(avg_score, 2),
        "details": results
    }


# =============================================================================
# MAIN ACCURACY CHECK
# =============================================================================

def check_accuracy(ocr_text: str, ground_truth_path: str, threshold: float = 0.75) -> Dict:
    """Run all accuracy checks and produce a comprehensive report."""

    with open(ground_truth_path, 'r') as f:
        gt = json.load(f)

    print_header(f"OCR ACCURACY REPORT: {gt.get('document_name', 'Unknown')}", "=", 100)
    print(f"  Total pages: {gt.get('total_pages', '?')}")
    print(f"  Match threshold: {threshold:.0%}")
    print(f"  OCR output length: {len(ocr_text):,} characters")

    # 1. Field accuracy
    field_results = check_field_accuracy(
        ocr_text, gt.get("fields", {}), threshold
    )

    # 2. Table accuracy
    table_results = {}
    if "tables" in gt and gt["tables"]:
        table_results = check_table_accuracy(
            ocr_text, gt["tables"], threshold
        )

    # 3. Financial accuracy (strict)
    financial_results = {}
    if "critical_financial_values" in gt and gt["critical_financial_values"]:
        financial_results = check_financial_accuracy(
            ocr_text, gt["critical_financial_values"]
        )

    # 4. Handwritten values
    hw_results = {}
    if "key_handwritten_values" in gt and gt["key_handwritten_values"]:
        hw_results = check_handwritten_accuracy(
            ocr_text, gt["key_handwritten_values"], threshold
        )

    # ==========================================================================
    # OVERALL SUMMARY
    # ==========================================================================
    print_header("OVERALL SUMMARY", "=", 100)

    scores = []

    f_acc = field_results.get("accuracy_pct", 0)
    scores.append(f_acc)
    print(f"  Field accuracy:           {field_results.get('found', 0)}/{field_results.get('total', 0)} ({f_acc:.1f}%)")

    if table_results:
        t_acc = table_results.get("cell_accuracy_pct", 0)
        scores.append(t_acc)
        print(f"  Table cell accuracy:      {table_results.get('cells_found', 0)}/{table_results.get('total_cells', 0)} ({t_acc:.1f}%)")
        print(f"  Table row detection:      {table_results.get('rows_detected', 0)}/{table_results.get('total_rows', 0)} ({table_results.get('row_accuracy_pct', 0):.1f}%)")

    if financial_results:
        fin_acc = financial_results.get("accuracy_pct", 0)
        scores.append(fin_acc)
        all_exact = financial_results.get("all_exact", False)
        marker = "ALL EXACT" if all_exact else "HAS ERRORS"
        print(f"  Financial accuracy:       {financial_results.get('found', 0)}/{financial_results.get('total', 0)} ({fin_acc:.1f}%) [{marker}]")

    if hw_results:
        hw_acc = hw_results.get("accuracy_pct", 0)
        scores.append(hw_acc)
        print(f"  Handwritten accuracy:     {hw_results.get('found', 0)}/{hw_results.get('total', 0)} ({hw_acc:.1f}%)")

    # Weighted overall score
    # Financial accuracy weighted 2x because errors there are critical
    if financial_results:
        weighted = (f_acc + (table_results.get("cell_accuracy_pct", 0) if table_results else 0) + fin_acc * 2 + (hw_acc if hw_results else 0))
        weight_count = 1 + (1 if table_results else 0) + 2 + (1 if hw_results else 0)
    else:
        weighted = sum(scores)
        weight_count = len(scores)

    overall = weighted / weight_count if weight_count > 0 else 0

    if overall >= 95:
        grade = "EXCELLENT"
    elif overall >= 85:
        grade = "GOOD"
    elif overall >= 70:
        grade = "ACCEPTABLE"
    elif overall >= 50:
        grade = "NEEDS IMPROVEMENT"
    else:
        grade = "POOR"

    print(f"\n  Weighted Overall Score:    {overall:.1f}%")
    print(f"  Grade:                    {grade}")

    # Missed items summary
    missed_fields = [r for r in field_results.get("details", []) if not r["found"]]
    missed_financial = [r for r in financial_results.get("details", []) if not r["found"]] if financial_results else []
    missed_hw = [r for r in hw_results.get("details", []) if not r["found"]] if hw_results else []

    if missed_fields or missed_financial or missed_hw:
        print_header("MISSED ITEMS DETAIL", "-", 100)

        if missed_financial:
            print("\n  CRITICAL - Missed Financial Values:")
            for r in missed_financial:
                print(f"    {r['field']}: expected '{r['expected']}' (page {r.get('page', '?')})")

        if missed_fields:
            print(f"\n  Missed Fields ({len(missed_fields)}):")
            for r in missed_fields[:15]:  # Show top 15
                print(f"    {r['field']}: expected '{r['expected'][:40]}' (score: {r['score']:.0%})")
            if len(missed_fields) > 15:
                print(f"    ... and {len(missed_fields) - 15} more")

        if missed_hw:
            print(f"\n  Missed Handwritten Values ({len(missed_hw)}):")
            for r in missed_hw:
                print(f"    '{r['value']}' (score: {r['score']:.0%})")

    print("\n" + "=" * 100)

    return {
        "document_name": gt.get("document_name"),
        "overall_score_pct": round(overall, 2),
        "grade": grade,
        "field_accuracy": field_results,
        "table_accuracy": table_results,
        "financial_accuracy": financial_results,
        "handwritten_accuracy": hw_results,
    }


# =============================================================================
# API INTEGRATION
# =============================================================================

def fetch_ocr_output_from_api(api_url: str, pdf_path: str) -> str:
    """Send PDF to OCR API and get the output text."""
    import requests

    url = f"{api_url.rstrip('/')}/api/process/single"
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        data = {'conversion_type': 'yaml'}
        print(f"Sending {pdf_path} to {url}...")
        response = requests.post(url, files=files, data=data, timeout=600)

    if response.status_code != 200:
        print(f"ERROR: API returned status {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    if result.get("status") != "success":
        print(f"ERROR: Processing failed: {result.get('error_message')}")
        sys.exit(1)

    return result["content"]


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OCR Accuracy Checker - Fields, Tables, Financial Values, Handwriting"
    )
    parser.add_argument("--ocr-output", type=str, help="Path to OCR output text file")
    parser.add_argument("--api-url", type=str, help="OCR API URL (e.g., http://localhost:8093)")
    parser.add_argument("--pdf", type=str, help="Path to PDF file (used with --api-url)")
    parser.add_argument("--ground-truth", type=str,
                        default=str(Path(__file__).parent / "ground_truth_health_claim.json"),
                        help="Path to ground truth JSON")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Min similarity to count as match (0.0-1.0)")
    parser.add_argument("--save-report", type=str, help="Save JSON report to path")

    args = parser.parse_args()

    # Get OCR text
    if args.ocr_output:
        with open(args.ocr_output, 'r') as f:
            ocr_text = f.read()
    elif args.api_url and args.pdf:
        ocr_text = fetch_ocr_output_from_api(args.api_url, args.pdf)
    else:
        print("ERROR: Provide either --ocr-output <file> or --api-url <url> --pdf <file>")
        print("\nExamples:")
        print('  python check_accuracy.py --ocr-output ocr_output.txt')
        print('  python check_accuracy.py --api-url http://localhost:8093 --pdf "Health Claim form.pdf"')
        sys.exit(1)

    # Run accuracy check
    results = check_accuracy(ocr_text, args.ground_truth, args.threshold)

    # Save report
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON report saved to: {args.save_report}")

    return results


if __name__ == "__main__":
    main()
