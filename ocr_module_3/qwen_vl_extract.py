#!/usr/bin/env python3
"""
qwen_vl_extract.py — ADE Pipeline v2: Single Qwen2.5-VL model

Replaces the hybrid OlmOCR + Qwen-text dual-model setup with a single
Qwen2.5-VL-72B-Instruct-AWQ that reads PDF page images directly.

6-Pass Architecture:
  Pass 1: Python   → Render PDF pages to base64 PNG images (PyMuPDF)
  Pass 2: Qwen-VL  → Parallel page-specific extraction (image+text per page)
  Pass 3: Python   → Deterministic validation (format, financial reconciliation)
  Pass 4: Qwen-VL  → Agentic re-extraction (strategy per field type, voting)
  Pass 5: Python   → Cross-page consistency checks
  Pass 6: Python   → Normalization and final output

H200 deployment (single vLLM instance, awq_marlin for speed):
  vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ --port 8000 --dtype float16 \
    --quantization awq_marlin --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --max-num-seqs 10 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QWEN_VL_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
TARGET_LONG_EDGE = 1792  # px — balanced for handwriting + model processing

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


def _extract_single_page(img, field_map):
    """Extract fields from a single page image (used by thread pool)."""
    pn = img["page_num"]
    prompt = _build_page_prompt(pn, field_map)
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
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    resp = api_post(f"{QWEN_VL_URL}/v1/chat/completions", payload)
    content = resp["choices"][0]["message"]["content"]
    tokens = resp.get("usage", {}).get("completion_tokens", 0)
    parsed = _parse_json_response(content)
    return pn, parsed, tokens, content


def pass2_extract(page_images, field_map):
    """Send all page images in parallel to Qwen-VL for extraction."""
    print("[Pass 2] Qwen-VL → Parallel page extraction...")
    t0 = time.time()

    all_extracted = {}
    total_tokens = 0

    # Filter to pages that have field mappings
    pages_to_extract = [img for img in page_images if img["page_num"] in field_map]
    num_pages = len(pages_to_extract)
    print(f"  Sending {num_pages} pages in parallel...", flush=True)

    with ThreadPoolExecutor(max_workers=num_pages) as executor:
        futures = {
            executor.submit(_extract_single_page, img, field_map): img["page_num"]
            for img in pages_to_extract
        }

        for future in as_completed(futures):
            pn = futures[future]
            try:
                pn, parsed, tokens, raw_content = future.result()
                total_tokens += tokens
                if parsed:
                    all_extracted.update(parsed)
                    print(f"  Page {pn}: OK ({tokens} tokens, {len(parsed)} fields)")
                else:
                    print(f"  Page {pn}: WARN — could not parse JSON")
                    all_extracted[f"_raw_page_{pn}"] = raw_content
            except Exception as e:
                print(f"  Page {pn}: ERROR — {e}")

    elapsed = time.time() - t0
    print(f"  Pass 2 done in {elapsed:.1f}s ({total_tokens} total tokens, {num_pages} pages parallel)")
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
# Pass 4: Agentic re-extraction (strategy per field type + majority voting)
# ---------------------------------------------------------------------------

# Map fields back to which page they come from
FIELD_TO_PAGE = {}
for _pn, _info in PAGE_FIELD_MAP.items():
    for _fld in _info["fields"]:
        FIELD_TO_PAGE[_fld] = _pn

# Field type classification for strategy selection
FINANCIAL_FIELDS = {"hospitalization_expenses", "post_hospitalization_expenses",
                    "total_claimed_amount"}
CODE_FIELDS = {"pan", "ifsc_code", "mobile_no", "policy_no",
               "ip_registration_number", "registration_no_with_state_code"}
MEDICAL_FIELDS = {"primary_diagnosis", "additional_diagnosis"}
NAME_FIELDS = {"treating_doctor", "insured_name", "patient_name"}

# Fields that appear on multiple pages (page → field name on that page)
CROSS_PAGE_FIELDS = {
    "hospital_name": [2, 4],
    "date_of_admission": [2, 4],
    "date_of_discharge": [2, 4],
    "gender": [2, 4],
    "age_years": [2, 4],
}

# Number of voting attempts for critical fields
VOTE_ATTEMPTS = 3


def _build_strategy_prompt(field, prev_val, page_num, section_hint, is_verification=False):
    """Build a field-type-specific prompt for agentic re-extraction.

    When is_verification=True, uses neutral prompts (no "this may be wrong" bias).
    """

    if field in FINANCIAL_FIELDS:
        return (
            f"Look at page {page_num} of this health insurance claim form.\n"
            f"Page: {section_hint}\n\n"
            f"Read the EXACT monetary amount for: {field}\n\n"
            f"Instructions for reading the amount:\n"
            f"1. Find the field labeled for this amount on the form\n"
            f"2. Read EACH DIGIT one at a time, left to right\n"
            f"3. Watch for: 1 vs 7, 0 vs 6, 5 vs 3, 4 vs 9\n"
            f"4. Count the total number of digits carefully\n"
            f"5. Return ONLY digits, no commas or symbols\n\n"
            f'Return JSON: {{"{field}": "<digits only>"}}\n'
            f"JSON:"
        )

    if field in CODE_FIELDS:
        fmt_hint = ""
        if field == "pan":
            fmt_hint = "PAN format: 5 letters + 4 digits + 1 letter (e.g. ABCDE1234F)"
        elif field == "ifsc_code":
            fmt_hint = "IFSC format: 4 letters + 0 + 6 alphanumeric (11 chars, e.g. HDFC0001234)"
        elif field == "mobile_no":
            fmt_hint = "Indian mobile: exactly 10 digits starting with 6-9"
        elif field == "ip_registration_number":
            fmt_hint = "IP/OP registration: typically starts with IP or OP followed by digits"
        elif field == "registration_no_with_state_code":
            fmt_hint = "Medical registration: state code + digits (e.g. KMC12345)"

        return (
            f"Look at page {page_num} of this health insurance claim form.\n"
            f"Page: {section_hint}\n\n"
            f"Read this code/number carefully: {field}\n"
            f"{fmt_hint}\n\n"
            f"Instructions:\n"
            f"1. Find this field on the form\n"
            f"2. Read EACH CHARACTER one at a time: spell it out like "
            f"\"first char is H, second is D, third is F, fourth is C...\"\n"
            f"3. Then combine them into the final value\n"
            f"4. Common confusions: 0↔O, 1↔I↔l, 5↔S, 8↔B, P↔R, F↔E\n\n"
            f'Return JSON: {{"{field}": "<your reading>"}}\n'
            f"JSON:"
        )

    if field in MEDICAL_FIELDS:
        return (
            f"Look at page {page_num} of this health insurance claim form.\n"
            f"Page: {section_hint}\n\n"
            f"Read the medical diagnosis field: {field}\n\n"
            f"This is a MEDICAL DIAGNOSIS written by a doctor. Common diagnoses include:\n"
            f"- Acute Cholecystitis, Chronic Cholecystitis\n"
            f"- Acute Appendicitis, Acute Pancreatitis\n"
            f"- Acute Tonsillitis, Acute Follicular Tonsillitis\n"
            f"- Dengue Fever, Typhoid, Malaria\n"
            f"- Hernia (Inguinal/Umbilical), Kidney Stones\n"
            f"- Fracture, Ligament Tear, Disc Prolapse\n"
            f"- Pneumonia, Bronchitis, Asthma\n\n"
            f"Read the handwriting carefully and match to the closest real diagnosis.\n\n"
            f'Return JSON: {{"{field}": "<diagnosis>"}}\n'
            f"JSON:"
        )

    if field in NAME_FIELDS:
        return (
            f"Look at page {page_num} of this health insurance claim form.\n"
            f"Page: {section_hint}\n\n"
            f"Read this person's name: {field}\n\n"
            f"Instructions:\n"
            f"1. This is a handwritten name — read each letter carefully\n"
            f"2. Indian names: look for common patterns (Dr., Kumar, Singh, etc.)\n"
            f"3. Return the name exactly as written, with proper capitalization\n\n"
            f'Return JSON: {{"{field}": "<name>"}}\n'
            f"JSON:"
        )

    # Generic fallback
    return (
        f"Look at page {page_num} of this health insurance claim form.\n"
        f"Page: {section_hint}\n\n"
        f"Read this field very carefully: {field}\n\n"
        f'Return JSON: {{"{field}": "<your reading>"}}\n'
        f"JSON:"
    )


def _vote_single_attempt(field, img, prompt, temperature):
    """One voting attempt for a single field."""
    b64_url = f"data:image/png;base64,{img['base64_png']}"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": b64_url}},
        ]}],
        "max_tokens": 256,
        "temperature": temperature,
    }
    resp = api_post(f"{QWEN_VL_URL}/v1/chat/completions", payload)
    content = resp["choices"][0]["message"]["content"]
    tokens = resp.get("usage", {}).get("completion_tokens", 0)
    parsed = _parse_json_response(content)
    value = parsed.get(field) if parsed else None
    return value, tokens


def _majority_vote(values):
    """Return the most common non-null value from a list."""
    clean = [str(v).strip() for v in values
             if v is not None and str(v).strip().lower() != "null"]
    if not clean:
        return None
    from collections import Counter
    counts = Counter(clean)
    return counts.most_common(1)[0][0]


def pass4_agentic(failures, extracted, page_images, field_map):
    """Agentic re-extraction: picks strategy per field, votes on critical fields.

    Only re-extracts fields that FAILED Pass 3 validation.
    For voting fields, includes Pass 2's original value as one vote.
    """
    reextract_fields = {
        f: reason for f, reason in failures.items()
        if not f.startswith("_") and f in FIELD_TO_PAGE
    }

    if not reextract_fields:
        print("[Pass 4] No fields to re-extract — skipping.")
        return extracted, 0.0

    print(f"[Pass 4] Agentic re-extraction → {len(reextract_fields)} fields "
          f"({sum(1 for f in reextract_fields if f in FINANCIAL_FIELDS)} financial, "
          f"{sum(1 for f in reextract_fields if f in CODE_FIELDS)} codes, "
          f"{sum(1 for f in reextract_fields if f in MEDICAL_FIELDS)} medical)...")
    t0 = time.time()

    total_tokens = 0
    updated = dict(extracted)

    # Build all tasks: (field, img, prompt, temperature)
    tasks = []
    for field, reason in reextract_fields.items():
        pn = FIELD_TO_PAGE[field]
        img = next((i for i in page_images if i["page_num"] == pn), None)
        if not img:
            continue

        prev_val = extracted.get(field, "")
        hint = field_map[pn]["section_hint"]
        prompt = _build_strategy_prompt(field, prev_val, pn, hint)

        # Financial + code fields get voting (3 attempts at different temps)
        if field in FINANCIAL_FIELDS or field in CODE_FIELDS:
            for temp in [0.1, 0.2, 0.3]:
                tasks.append((field, img, prompt, temp))
        else:
            # Single attempt at temp 0.15
            tasks.append((field, img, prompt, 0.15))

    # Execute all tasks in parallel
    field_votes = {}  # field → list of values
    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
        future_map = {}
        for field, img, prompt, temp in tasks:
            fut = executor.submit(_vote_single_attempt, field, img, prompt, temp)
            future_map[fut] = field

        for future in as_completed(future_map):
            field = future_map[future]
            try:
                value, tokens = future.result()
                total_tokens += tokens
                field_votes.setdefault(field, []).append(value)
            except Exception as e:
                print(f"    {field}: attempt error — {e}")

    # Resolve votes — include Pass 2's original value as a vote for multi-attempt fields
    for field, votes in field_votes.items():
        orig = extracted.get(field)
        if len(votes) > 1 and orig is not None and str(orig).strip():
            votes.append(str(orig).strip())  # Pass 2's value gets a vote too

        winner = _majority_vote(votes)
        if winner is not None and str(winner).strip().lower() != "null":
            strategy = "vote" if len(votes) > 1 else "single"
            unique = set(str(v) for v in votes if v is not None)
            if len(votes) > 1:
                print(f"  {field}: {strategy} [{len(votes)} attempts, "
                      f"{len(unique)} unique] → {winner}")
            else:
                print(f"  {field}: {strategy} → {winner}")
            updated[field] = winner

    elapsed = time.time() - t0
    print(f"  Pass 4 done in {elapsed:.1f}s ({total_tokens} tokens, {len(tasks)} API calls)")
    return updated, elapsed


# ---------------------------------------------------------------------------
# Pass 5: Cross-page consistency
# ---------------------------------------------------------------------------

def pass5_cross_page(extracted, page_images, field_map):
    """Extract cross-page fields from alternate pages and apply corrections.

    Page 4 (Part B, hospital-filled) is preferred for: dates, gender, age
    because hospital staff entries are typically more legible/accurate.
    """
    print("[Pass 5] Qwen-VL → Cross-page consistency...")
    t0 = time.time()
    corrections = 0
    total_tokens = 0

    # Fields where page 4 (hospital section) is more reliable
    PREFER_PAGE_4 = {"date_of_admission", "date_of_discharge", "gender", "age_years"}

    tasks = []
    for field, pages in CROSS_PAGE_FIELDS.items():
        current_val = extracted.get(field)
        if current_val is None:
            continue

        current_page = FIELD_TO_PAGE.get(field)
        other_pages = [p for p in pages if p != current_page]
        if not other_pages:
            continue

        alt_page = other_pages[0]
        img = next((i for i in page_images if i["page_num"] == alt_page), None)
        if not img:
            continue

        alt_hint = field_map.get(alt_page, {}).get("section_hint", "")
        prompt = (
            f"Look at page {alt_page} of this health insurance claim form.\n"
            f"Page: {alt_hint}\n\n"
            f"Read the value for: {field}\n"
            f"Return ONLY a JSON object: {{\"{field}\": \"<value>\"}}\n"
            f"JSON:"
        )
        tasks.append((field, img, prompt, 0.1, current_val, current_page, alt_page))

    if not tasks:
        elapsed = time.time() - t0
        print(f"  No cross-page fields to check.")
        print(f"  Pass 5 done in {elapsed:.1f}s ({corrections} corrections)")
        return extracted, elapsed

    updated = dict(extracted)
    with ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
        future_map = {}
        for field, img, prompt, temp, cur_val, cur_pg, alt_pg in tasks:
            fut = executor.submit(_vote_single_attempt, field, img, prompt, temp)
            future_map[fut] = (field, cur_val, cur_pg, alt_pg)

        for future in as_completed(future_map):
            field, cur_val, cur_pg, alt_pg = future_map[future]
            try:
                alt_val, tokens = future.result()
                total_tokens += tokens
                if alt_val is None:
                    continue

                cur_str = str(cur_val).strip()
                alt_str = str(alt_val).strip()

                if cur_str.lower() == alt_str.lower():
                    print(f"  {field}: pages {cur_pg}&{alt_pg} agree → {cur_val}")
                else:
                    # Mismatch — decide which to keep
                    if field in PREFER_PAGE_4 and alt_pg == 4:
                        updated[field] = alt_str
                        corrections += 1
                        print(f"  {field}: MISMATCH p{cur_pg}=\"{cur_val}\" vs p{alt_pg}=\"{alt_val}\" → using p{alt_pg} (hospital)")
                    elif field in PREFER_PAGE_4 and cur_pg == 4:
                        # Already have p4's value, keep it
                        print(f"  {field}: MISMATCH p{cur_pg}=\"{cur_val}\" vs p{alt_pg}=\"{alt_val}\" → keeping p{cur_pg} (hospital)")
                    else:
                        # For other fields, keep the primary extraction
                        print(f"  {field}: MISMATCH p{cur_pg}=\"{cur_val}\" vs p{alt_pg}=\"{alt_val}\" → keeping p{cur_pg}")
            except Exception as e:
                print(f"  {field}: cross-page error — {e}")

    elapsed = time.time() - t0
    print(f"  Pass 5 done in {elapsed:.1f}s ({corrections} corrections, {total_tokens} tokens)")
    return updated, elapsed


# ---------------------------------------------------------------------------
# Pass 6: Normalization and final output
# ---------------------------------------------------------------------------

NULL_LIKE = {"null", "none", "n/a", "na", "[]", "[ ]", "n.a.", "-", "--", ""}


def pass6_normalize(extracted):
    """Normalize extracted data into clean final output."""
    print("[Pass 6] Python → Normalization...")
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
    print(f"  Pass 6 done in {elapsed:.1f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path):
    """Execute the full 6-pass pipeline and return results dict."""
    print("=" * 60)
    print("  ADE PIPELINE v2 — Single Qwen2.5-VL (Agentic)")
    print("  6-pass: render → extract → validate → agentic → xpage → norm")
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

    # Pass 2: Vision extraction per page (parallel)
    extracted, t2 = pass2_extract(page_images, PAGE_FIELD_MAP)

    # Pass 3: Deterministic validation
    failures, t3 = pass3_validate(extracted)

    # Pass 4: Agentic re-extraction (strategy per field type + voting)
    extracted, t4 = pass4_agentic(failures, extracted, page_images, PAGE_FIELD_MAP)

    # Pass 5: Cross-page consistency
    extracted, t5 = pass5_cross_page(extracted, page_images, PAGE_FIELD_MAP)

    # Pass 6: Normalization
    final_data, t6 = pass6_normalize(extracted)

    total_time = time.time() - total_start

    # Build output compatible with check_accuracy.py (values appear in text)
    output = {
        "document": pdf_path.split("/")[-1] if "/" in pdf_path else pdf_path,
        "pipeline": "qwen_vl_single_model_v2_agentic",
        "model": MODEL_NAME,
        "structured_data": final_data,
        "timing": {
            "pass1_render_seconds": round(t1, 2),
            "pass2_extract_seconds": round(t2, 2),
            "pass3_validate_seconds": round(t3, 2),
            "pass4_agentic_seconds": round(t4, 2),
            "pass5_crosspage_seconds": round(t5, 2),
            "pass6_normalize_seconds": round(t6, 2),
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
        f"agentic: {timing['pass4_agentic_seconds']:.1f}s + "
        f"xpage: {timing['pass5_crosspage_seconds']:.1f}s + "
        f"normalize: {timing['pass6_normalize_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    main()
