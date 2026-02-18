#!/usr/bin/env python3
"""
qwen_vl_extract.py — ADE Pipeline v2: Single Qwen2.5-VL model

Replaces the hybrid OlmOCR + Qwen-text dual-model setup with a single
Qwen2.5-VL-7B-Instruct that reads PDF page images directly.

5-Pass Architecture:
  Pass 1: Python   → Render PDF pages to base64 PNG images (PyMuPDF)
  Pass 2: Qwen-VL  → Page-specific extraction (image + text prompt per page)
  Pass 3: Python   → Deterministic validation (format, financial reconciliation)
  Pass 4: Qwen-VL  → Targeted re-extraction of failed fields (re-read image)
  Pass 5: Python   → Normalization and final output

H200 deployment (single vLLM instance):
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 10 \
    --trust-remote-code --limit-mm-per-prompt image=1 \
    --mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 1003520}'

Usage:
  python qwen_vl_extract.py "Health Claim form.pdf" -o /tmp/qwen_vl_result.json
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QWEN_VL_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TARGET_LONG_EDGE = 1792  # px — better for handwriting than default 1288

# ---------------------------------------------------------------------------
# PAGE_FIELD_MAP — maps each page to fields + section hints
# Derived from ground truth structure of health insurance claim forms
# ---------------------------------------------------------------------------

PAGE_FIELD_MAP = {
    1: {
        "section_hint": "Claim Acknowledgment page with document checklist table",
        "fields": {
            "insured_name": "Full name of the insured person (handwritten at top)",
            "patient_name": "Full name of the patient (handwritten, may be same as insured)",
            "name_of_corporate": "Corporate/employer name (printed or handwritten)",
            "mobile_no": "10-digit mobile number (handwritten)",
            "email_id": "Email address of insured (handwritten)",
            "date_of_claim_submission": "Date claim was submitted (DD/MM/YYYY, handwritten)",
            "claim_submitted_by": "Name of person who submitted the claim",
            "type_of_claim": "Type of claim (e.g. Main Hospitalization)",
            "claim_submitted_at": "Location/office where claim was submitted (e.g. PHS)",
        },
    },
    2: {
        "section_hint": "Claim Form Part A — has sections: A (Primary Insured), C (Insured Person), D (Hospitalization Details), E (Claim Amounts), G (Bank Details). Contains bills table at bottom.",
        "fields": {
            "policy_no": "Policy number (long numeric, Section A)",
            "gender": "Gender of insured person (Male/Female, Section C)",
            "age_years": "Age in years (numeric, Section C)",
            "hospital_name": "Name of hospital (Section D, handwritten)",
            "room_category": "Hospital room category (e.g. Single occupancy, Section D)",
            "hospitalization_due_to": "Cause: Illness, Injury, or Maternity (Section D)",
            "date_of_admission": "Date of admission (DD/MM/YY, Section D, handwritten)",
            "date_of_discharge": "Date of discharge (DD/MM/YY, Section D, handwritten)",
            "medico_legal": "Medico-legal case Yes/No (Section D)",
            "hospitalization_expenses": "Hospitalization expenses amount in rupees (digits only, Section E)",
            "post_hospitalization_expenses": "Post-hospitalization expenses amount in rupees (digits only, Section E)",
            "pan": "PAN card number (10 chars like AAAAA9999A, Section G)",
            "bank_name": "Bank name and branch (Section G, handwritten)",
            "ifsc_code": "Bank IFSC code (Section G, handwritten)",
            "relationship_to_primary_insured": "Relationship (Self, Spouse, etc., Section C)",
            "occupation": "Occupation of insured person (Section C)",
        },
    },
    3: {
        "section_hint": "Declaration page — signed by insured, has handwritten date and place at bottom",
        "fields": {
            "declaration_date": "Date on the declaration (DD/MM/YY, handwritten at bottom)",
            "declaration_place": "Place written on the declaration (handwritten, e.g. Bangalore)",
        },
    },
    4: {
        "section_hint": "Claim Form Part B (hospital section) — has sections: A (Hospital Details), B (Patient Details), C (Diagnosis). Hospital address at bottom.",
        "fields": {
            "treating_doctor": "Name of treating doctor (Section A, handwritten)",
            "qualification": "Doctor qualification like MBBS, MD (Section A)",
            "registration_no_with_state_code": "Doctor registration number with state code (Section A)",
            "type_of_hospital": "Network or Non-Network (Section A)",
            "ip_registration_number": "In-patient registration number (Section B)",
            "type_of_admission": "Emergency, Planned, Day Care, or Maternity (Section B)",
            "status_at_discharge": "Status at discharge (e.g. Discharge to home, Section B)",
            "total_claimed_amount": "Total amount claimed in rupees (digits only, Section B)",
            "primary_diagnosis": "Primary diagnosis (Section C, handwritten)",
            "additional_diagnosis": "Additional/secondary diagnosis (Section C)",
            "hospital_address": "Full hospital address including area, city, pin code",
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_post(url, payload, timeout=300):
    """POST JSON to the vLLM OpenAI-compatible API."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_json_response(text):
    """Extract JSON object from LLM response (handles markdown fences)."""
    clean = text.strip()
    # Strip markdown code fences
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Pass 1: PDF → Base64 PNG images
# ---------------------------------------------------------------------------

def pass1_render_pdf(pdf_path):
    """Render each page of *pdf_path* as a high-res base64-encoded PNG."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF")
        sys.exit(1)

    print("[Pass 1] PDF → Base64 PNG rendering...")
    t0 = time.time()
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Scale so longest edge ≈ TARGET_LONG_EDGE
        rect = page.rect
        long_edge = max(rect.width, rect.height)
        zoom = TARGET_LONG_EDGE / long_edge if long_edge > 0 else 1.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        pages.append({
            "page_num": page_num + 1,  # 1-indexed
            "base64_png": b64,
            "width": pix.width,
            "height": pix.height,
        })

    doc.close()
    elapsed = time.time() - t0
    print(f"  Rendered {len(pages)} pages in {elapsed:.1f}s "
          f"(target {TARGET_LONG_EDGE}px)")
    return pages, elapsed


# ---------------------------------------------------------------------------
# Pass 2: Vision extraction per page
# ---------------------------------------------------------------------------

def _build_page_prompt(page_num, field_map):
    """Build the extraction prompt for a single page."""
    info = field_map[page_num]
    hint = info["section_hint"]
    fields = info["fields"]

    field_lines = "\n".join(
        f'  - "{k}": {v}' for k, v in fields.items()
    )
    field_keys = list(fields.keys())

    prompt = (
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
    return prompt


def pass2_extract(page_images, field_map):
    """Send each page image + targeted prompt to Qwen-VL for extraction."""
    print("[Pass 2] Qwen-VL → Page-specific extraction...")
    t0 = time.time()

    all_extracted = {}
    total_tokens = 0

    for img in page_images:
        pn = img["page_num"]
        if pn not in field_map:
            continue

        prompt = _build_page_prompt(pn, field_map)
        b64_url = f"data:image/png;base64,{img['base64_png']}"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": b64_url},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        print(f"  Page {pn}: extracting {len(field_map[pn]['fields'])} fields ...", end=" ", flush=True)
        try:
            resp = api_post(f"{QWEN_VL_URL}/v1/chat/completions", payload)
            content = resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            total_tokens += tokens

            parsed = _parse_json_response(content)
            if parsed:
                all_extracted.update(parsed)
                print(f"OK ({tokens} tokens, {len(parsed)} fields)")
            else:
                print(f"WARN: could not parse JSON")
                # Store raw response for debugging
                all_extracted[f"_raw_page_{pn}"] = content
        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"  Pass 2 done in {elapsed:.1f}s ({total_tokens} total tokens)")
    return all_extracted, elapsed


# ---------------------------------------------------------------------------
# Pass 3: Deterministic Python validation
# ---------------------------------------------------------------------------

VALIDATION_RULES = {
    # Date fields → DD/MM/YY or DD/MM/YYYY
    "date_of_admission": "date",
    "date_of_discharge": "date",
    "date_of_claim_submission": "date",
    "declaration_date": "date",
    # Financial fields → must be numeric (digits only)
    "hospitalization_expenses": "financial",
    "post_hospitalization_expenses": "financial",
    "total_claimed_amount": "financial",
    # PAN → AAAAA9999A (5 letters, 4 digits, 1 letter)
    "pan": "pan",
    # IFSC → 4 letters + 0 + 6 alphanumeric (11 chars)
    "ifsc_code": "ifsc",
    # Mobile → 10 digits
    "mobile_no": "mobile",
}

# Required fields (must be present and non-null)
REQUIRED_FIELDS = [
    "insured_name", "patient_name", "policy_no", "hospital_name",
    "date_of_admission", "date_of_discharge", "hospitalization_expenses",
    "total_claimed_amount", "treating_doctor", "primary_diagnosis",
]


def _validate_date(value):
    """Check DD/MM/YY or DD/MM/YYYY format."""
    if not value:
        return False, "empty"
    return bool(re.match(r"^\d{1,2}/\d{1,2}/(\d{2}|\d{4})$", str(value).strip())), "expected DD/MM/YY(YY)"


def _validate_financial(value):
    """Financial value must be numeric (digits only, after stripping commas)."""
    if not value:
        return False, "empty"
    cleaned = re.sub(r"[₹,\s\.\-]", "", str(value))
    return bool(re.match(r"^\d+$", cleaned)), f"non-numeric: '{value}'"


def _validate_pan(value):
    """PAN format: AAAAA9999A (5 letters, 4 digits, 1 letter)."""
    if not value:
        return False, "empty"
    return bool(re.match(r"^[A-Z]{5}\d{4}[A-Z]$", str(value).strip().upper())), f"bad PAN: '{value}'"


def _validate_ifsc(value):
    """IFSC: 4 letters + 7 alphanumeric (11 chars total)."""
    if not value:
        return False, "empty"
    v = str(value).strip().upper()
    return bool(re.match(r"^[A-Z]{4}[A-Z0-9]{7}$", v)), f"bad IFSC: '{value}'"


def _validate_mobile(value):
    """Mobile: 10 digits."""
    if not value:
        return False, "empty"
    digits = re.sub(r"[\s\-\+]", "", str(value))
    # Strip country code prefix
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    return bool(re.match(r"^\d{10}$", digits)), f"not 10 digits: '{value}'"


VALIDATORS = {
    "date": _validate_date,
    "financial": _validate_financial,
    "pan": _validate_pan,
    "ifsc": _validate_ifsc,
    "mobile": _validate_mobile,
}


def pass3_validate(extracted):
    """Deterministic validation: format checks + financial reconciliation."""
    print("[Pass 3] Python → Deterministic validation...")
    t0 = time.time()
    failures = {}  # field → reason

    # 1. Required field presence
    for field in REQUIRED_FIELDS:
        val = extracted.get(field)
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "null":
            failures[field] = "required field missing"

    # 2. Format validation
    for field, rule_type in VALIDATION_RULES.items():
        val = extracted.get(field)
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "null":
            if field in REQUIRED_FIELDS:
                failures.setdefault(field, "required field missing")
            continue
        validator = VALIDATORS[rule_type]
        ok, reason = validator(val)
        if not ok:
            failures[field] = reason

    # 3. Financial reconciliation: parts should not exceed total
    hosp = extracted.get("hospitalization_expenses")
    post_hosp = extracted.get("post_hospitalization_expenses")
    total = extracted.get("total_claimed_amount")

    if hosp and post_hosp and total:
        try:
            h = int(re.sub(r"[^\d]", "", str(hosp)))
            p = int(re.sub(r"[^\d]", "", str(post_hosp)))
            t = int(re.sub(r"[^\d]", "", str(total)))
            if h + p > t * 1.1:  # 10% tolerance for rounding
                failures["_financial_reconciliation"] = (
                    f"hosp({h}) + post_hosp({p}) = {h+p} > total({t})"
                )
        except (ValueError, TypeError):
            pass

    elapsed = time.time() - t0
    status = "PASS" if not failures else f"FAIL ({len(failures)} issues)"
    print(f"  Validation: {status}")
    for field, reason in failures.items():
        print(f"    {field}: {reason}")
    print(f"  Pass 3 done in {elapsed:.1f}s")
    return failures, elapsed


# ---------------------------------------------------------------------------
# Pass 4: Targeted re-extraction for failed fields
# ---------------------------------------------------------------------------

# Map fields back to which page they come from
FIELD_TO_PAGE = {}
for _pn, _info in PAGE_FIELD_MAP.items():
    for _fld in _info["fields"]:
        FIELD_TO_PAGE[_fld] = _pn


def pass4_reextract(failures, extracted, page_images, field_map):
    """Re-extract only failed fields with focused prompts + slightly higher temp."""
    # Filter to extractable failures (skip _financial_reconciliation, etc.)
    reextract_fields = {
        f: reason for f, reason in failures.items()
        if not f.startswith("_") and f in FIELD_TO_PAGE
    }

    if not reextract_fields:
        print("[Pass 4] No fields to re-extract — skipping.")
        return extracted, 0.0

    print(f"[Pass 4] Qwen-VL → Re-extracting {len(reextract_fields)} failed fields...")
    t0 = time.time()

    # Group failed fields by page
    page_fields = {}
    for field, reason in reextract_fields.items():
        pn = FIELD_TO_PAGE[field]
        page_fields.setdefault(pn, []).append((field, reason))

    total_tokens = 0
    updated = dict(extracted)

    for pn, fields_reasons in page_fields.items():
        img = next((i for i in page_images if i["page_num"] == pn), None)
        if not img:
            continue

        field_instructions = []
        field_keys = []
        for field, reason in fields_reasons:
            prev_val = extracted.get(field)
            desc = field_map[pn]["fields"].get(field, "")
            hint = f'  - "{field}": {desc}'
            if prev_val and str(prev_val).strip().lower() != "null":
                hint += f' (previous attempt returned "{prev_val}" which failed: {reason})'
            field_instructions.append(hint)
            field_keys.append(field)

        prompt = (
            f"Look at this scanned page very carefully.\n"
            f"I need you to re-read these specific fields — the previous extraction had errors.\n"
            f"Page {pn}: {field_map[pn]['section_hint']}\n\n"
            f"Fields to re-extract (read the image very carefully):\n"
            + "\n".join(field_instructions)
            + f"\n\nCommon handwriting mistakes to watch for:\n"
            f"- 0 vs O, 1 vs I vs l, 5 vs S, 6 vs G, 8 vs B\n"
            f"- Dates: slash vs dash, 2-digit vs 4-digit year\n"
            f"- Amounts: extra/missing digits, commas misread as digits\n\n"
            f"Return ONLY JSON with keys: {json.dumps(field_keys)}\n"
            f"JSON:"
        )

        b64_url = f"data:image/png;base64,{img['base64_png']}"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64_url}},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }

        print(f"  Page {pn}: re-extracting {[f for f, _ in fields_reasons]} ...", end=" ", flush=True)
        try:
            resp = api_post(f"{QWEN_VL_URL}/v1/chat/completions", payload)
            content = resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            total_tokens += tokens

            parsed = _parse_json_response(content)
            if parsed:
                for k, v in parsed.items():
                    if v is not None and str(v).strip().lower() != "null":
                        updated[k] = v
                print(f"OK ({tokens} tokens)")
            else:
                print("WARN: could not parse")
        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"  Pass 4 done in {elapsed:.1f}s ({total_tokens} tokens)")
    return updated, elapsed


# ---------------------------------------------------------------------------
# Pass 5: Normalization and final output
# ---------------------------------------------------------------------------

NULL_LIKE = {"null", "none", "n/a", "na", "[]", "[ ]", "n.a.", "-", "--", ""}


def pass5_normalize(extracted):
    """Normalize extracted data into clean final output."""
    print("[Pass 5] Python → Normalization...")
    t0 = time.time()
    result = {}

    for key, value in extracted.items():
        if key.startswith("_"):
            continue  # skip internal keys
        if value is None:
            result[key] = None
            continue

        v = str(value).strip()

        # Remove null-like values
        if v.lower() in NULL_LIKE:
            result[key] = None
            continue

        # Financial fields → pure digits
        if key in ("hospitalization_expenses", "post_hospitalization_expenses",
                    "total_claimed_amount"):
            digits = re.sub(r"[^\d]", "", v)
            result[key] = digits if digits else None
            continue

        # Date normalization → keep DD/MM/YY or DD/MM/YYYY as-is
        if key in ("date_of_admission", "date_of_discharge",
                    "date_of_claim_submission", "declaration_date"):
            # Strip any whitespace around slashes
            v = re.sub(r"\s*/\s*", "/", v)
            # Convert DD-MM-YY(YY) to DD/MM/YY(YY)
            v = re.sub(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$", r"\1/\2/\3", v)
            result[key] = v
            continue

        # PAN / IFSC → uppercase
        if key in ("pan", "ifsc_code"):
            result[key] = v.upper()
            continue

        # Mobile → digits only
        if key == "mobile_no":
            digits = re.sub(r"[^\d]", "", v)
            if digits.startswith("91") and len(digits) == 12:
                digits = digits[2:]
            result[key] = digits
            continue

        # Everything else — light cleanup
        result[key] = v

    elapsed = time.time() - t0
    non_null = sum(1 for v in result.values() if v is not None)
    print(f"  Normalized {len(result)} fields ({non_null} non-null)")
    print(f"  Pass 5 done in {elapsed:.1f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path):
    """Execute the full 5-pass pipeline and return results dict."""
    print("=" * 60)
    print("  ADE PIPELINE v2 — Single Qwen2.5-VL")
    print("  (replaces OlmOCR + Qwen dual-model setup)")
    print("=" * 60)
    print(f"  PDF: {pdf_path}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  vLLM endpoint: {QWEN_VL_URL}")
    print()

    total_start = time.time()

    # Pass 1: Render PDF → images
    page_images, t1 = pass1_render_pdf(pdf_path)
    if not page_images:
        print("ERROR: No pages rendered from PDF")
        sys.exit(1)

    # Pass 2: Vision extraction per page
    extracted, t2 = pass2_extract(page_images, PAGE_FIELD_MAP)

    # Pass 3: Deterministic validation
    failures, t3 = pass3_validate(extracted)

    # Pass 4: Targeted re-extraction of failed fields
    extracted, t4 = pass4_reextract(failures, extracted, page_images, PAGE_FIELD_MAP)

    # Pass 5: Normalization
    final_data, t5 = pass5_normalize(extracted)

    total_time = time.time() - total_start

    # Build output compatible with check_accuracy.py (values appear in text)
    output = {
        "document": pdf_path.split("/")[-1] if "/" in pdf_path else pdf_path,
        "pipeline": "qwen_vl_single_model_v2",
        "model": MODEL_NAME,
        "structured_data": final_data,
        "timing": {
            "pass1_render_seconds": round(t1, 2),
            "pass2_extract_seconds": round(t2, 2),
            "pass3_validate_seconds": round(t3, 2),
            "pass4_reextract_seconds": round(t4, 2),
            "pass5_normalize_seconds": round(t5, 2),
            "total_seconds": round(total_time, 2),
        },
        "pages_processed": len(page_images),
        "validation_failures": failures if failures else {},
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="ADE Pipeline v2 — Single Qwen2.5-VL extraction"
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file path")
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM endpoint URL (default: http://localhost:8000)")
    args = parser.parse_args()

    if args.vllm_url:
        global QWEN_VL_URL
        QWEN_VL_URL = args.vllm_url.rstrip("/")

    result = run_pipeline(args.pdf_path)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(output_json)

    timing = result["timing"]
    print(
        f"\n  Total time: {timing['total_seconds']:.1f}s "
        f"(render: {timing['pass1_render_seconds']:.1f}s + "
        f"extract: {timing['pass2_extract_seconds']:.1f}s + "
        f"validate: {timing['pass3_validate_seconds']:.1f}s + "
        f"re-extract: {timing['pass4_reextract_seconds']:.1f}s + "
        f"normalize: {timing['pass5_normalize_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    main()
