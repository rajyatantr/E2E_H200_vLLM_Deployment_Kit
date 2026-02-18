#!/usr/bin/env python3
"""
hybrid_extract.py - Hybrid ADE pipeline
  Stage 1: OlmOCR (vision) → raw OCR text from scanned pages
  Stage 2: Qwen 2.5 7B (text) → structure into JSON schema
  Stage 3: Qwen 2.5 7B (text) → cross-page validation & discrepancy detection
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


OLMOCR_URL = "http://localhost:8093"
QWEN_URL = "http://localhost:8000"


def api_post(url, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def multipart_upload(url, filepath):
    """Upload a file via multipart/form-data."""
    import mimetypes
    boundary = "----HybridADEBoundary"
    filename = filepath.split("/")[-1]
    mime = mimetypes.guess_type(filepath)[0] or "application/pdf"

    with open(filepath, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def stage1_ocr(pdf_path):
    """Stage 1: Send PDF to OlmOCR for raw text extraction."""
    print("[Stage 1] OlmOCR → Raw text extraction...")
    t0 = time.time()
    result = multipart_upload(f"{OLMOCR_URL}/api/process/single", pdf_path)
    elapsed = time.time() - t0
    print(f"  OCR completed in {elapsed:.1f}s, {len(result.get('pages', []))} pages")
    return result, elapsed


def _parse_json_response(text):
    """Try multiple strategies to extract JSON from LLM response."""
    import re
    clean = text.strip()
    # Strip markdown code fences
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0].strip()
    # Try direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object in the text
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _truncate_page(content, max_chars=3000):
    """Truncate page content, keeping the most important parts."""
    if len(content) <= max_chars:
        return content
    # Keep the beginning (has key fields) and trim tables/repetitive content
    # Remove large table HTML blocks to save tokens
    import re
    # Compress table markup
    content = re.sub(r'<table>.*?</table>', '[TABLE CONTENT]', content, flags=re.DOTALL)
    # Remove checkbox placeholders
    content = re.sub(r'\u2610 \[blank\] ', '', content)
    content = re.sub(r'\[ \] ', '', content)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated]"
    return content


def stage2_structure(raw_ocr_pages, schema):
    """Stage 2: Send raw OCR text to Qwen for structured extraction per page pair."""
    print("[Stage 2] Qwen → Structured JSON extraction...")
    t0 = time.time()

    # Process pages 1+2 and pages 3+4 separately, then merge
    results = []
    page_pairs = []
    data_pages = [p for p in raw_ocr_pages if p["page_num"] != 3]  # Skip page 3 (guidance only)

    for page in data_pages:
        page_pairs.append(f"=== PAGE {page['page_num']} ===\n{_truncate_page(page['content'])}")

    schema_keys = list(schema.keys())
    schema_brief = ", ".join(schema_keys)

    all_text = "\n\n".join(page_pairs)

    prompt = f"""Extract these fields from the health insurance claim form OCR text below. Return ONLY a JSON object.

Fields to extract: {schema_brief}

Rules: Use null if not found. Dates as DD/MM/YYYY. Amounts as numbers only. Read handwritten values carefully.

OCR TEXT:
{all_text}

JSON:"""

    result = api_post(f"{QWEN_URL}/v1/chat/completions", {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.1,
    })

    content = result["choices"][0]["message"]["content"]

    # Try to parse JSON from the response
    structured = _parse_json_response(content)
    if structured is None:
        print(f"  WARNING: Could not parse JSON from Qwen response")
        structured = {"raw_response": content}

    elapsed = time.time() - t0
    tokens = result.get("usage", {})
    print(f"  Structuring completed in {elapsed:.1f}s ({tokens.get('completion_tokens', '?')} tokens)")
    return structured, elapsed


def stage3_validate(structured_data, raw_ocr_pages):
    """Stage 3: Send structured data back to Qwen for cross-page validation."""
    print("[Stage 3] Qwen → Cross-page validation...")
    t0 = time.time()

    # Truncate pages and skip page 3 (guidance only)
    data_pages = [p for p in raw_ocr_pages if p["page_num"] != 3]
    all_pages_text = ""
    for page in data_pages:
        all_pages_text += f"\n=== PAGE {page['page_num']} ===\n{_truncate_page(page['content'], max_chars=1500)}"

    # Compact JSON representation of structured data
    struct_compact = json.dumps(structured_data, separators=(',', ':'))

    prompt = f"""Compare extracted data against OCR text. Find discrepancies, missing fields, incorrect values, financial errors. Return JSON only.

EXTRACTED:
{struct_compact}

OCR:
{all_pages_text}

Return JSON: {{"discrepancies":[{{"field":"...","issue":"...","severity":"critical|warning"}}],"corrections":[{{"field":"...","extracted":"...","correct":"..."}}],"financial_check":{{"total_matches":true/false,"notes":"..."}},"overall_status":"pass|fail","confidence":0.0-1.0}}

JSON:"""

    result = api_post(f"{QWEN_URL}/v1/chat/completions", {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.1,
    })

    content = result["choices"][0]["message"]["content"]

    validation = _parse_json_response(content)
    if validation is None:
        print(f"  WARNING: Could not parse validation JSON")
        validation = {"raw_response": content}

    elapsed = time.time() - t0
    tokens = result.get("usage", {})
    print(f"  Validation completed in {elapsed:.1f}s ({tokens.get('completion_tokens', '?')} tokens)")
    return validation, elapsed


def main():
    parser = argparse.ArgumentParser(description="Hybrid ADE Pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    args = parser.parse_args()

    schema = {
        "insured_name": "Full name of the insured person",
        "patient_name": "Full name of the patient",
        "policy_no": "Policy number",
        "company_tpa_id": "Company TPA ID number",
        "corporate_name": "Name of corporate/employer",
        "employee_no": "Employee number",
        "mobile_no": "Mobile phone number",
        "email_id": "Email address",
        "hospital_name": "Name of hospital where admitted",
        "hospital_type": "Network or Non-Network",
        "hospital_address": "Full hospital address",
        "treating_doctor": "Name of treating doctor",
        "doctor_qualification": "Doctor qualification (MBBS, MD, etc.)",
        "doctor_registration_no": "Doctor registration number with state code",
        "gender_part_a": "Gender as marked in Part A (insured's section)",
        "gender_part_b": "Gender as marked in Part B (hospital's section)",
        "age_years": "Age in years",
        "date_of_birth": "Date of birth (DD/MM/YYYY)",
        "date_of_admission": "Date of admission (DD/MM/YYYY)",
        "time_of_admission": "Time of admission",
        "date_of_discharge": "Date of discharge (DD/MM/YYYY)",
        "type_of_admission": "Emergency, Planned, Day Care, or Maternity",
        "hospitalization_due_to": "Injury, Illness, or Maternity",
        "primary_diagnosis": "Primary diagnosis",
        "additional_diagnosis": "Additional diagnosis if any",
        "pre_hospitalization_expenses": "Pre-hospitalization expenses (number only)",
        "hospitalization_expenses": "Hospitalization expenses (number only)",
        "post_hospitalization_expenses": "Post-hospitalization expenses (number only)",
        "total_claimed_amount": "Total amount claimed (number only)",
        "pan": "PAN card number",
        "bank_name": "Bank name and branch",
        "account_number": "Bank account number",
        "ifsc_code": "Bank IFSC code",
        "ip_registration_number": "In-patient registration number",
        "date_of_claim_submission": "Date claim was submitted",
        "declaration_date": "Date on declaration",
        "declaration_place": "Place on declaration",
    }

    print("=" * 60)
    print("  HYBRID ADE PIPELINE")
    print("  OlmOCR (vision) + Qwen 2.5 7B (text)")
    print("=" * 60)
    print(f"  PDF: {args.pdf_path}")
    print()

    total_start = time.time()

    # Stage 1: OCR
    ocr_result, ocr_time = stage1_ocr(args.pdf_path)
    pages = ocr_result.get("pages", [])
    if not pages:
        print("ERROR: No pages extracted from PDF")
        sys.exit(1)

    # Stage 2: Structure
    structured, struct_time = stage2_structure(pages, schema)

    # Stage 3: Validate
    validation, val_time = stage3_validate(structured, pages)

    total_time = time.time() - total_start

    # Combine results
    final = {
        "document": args.pdf_path.split("/")[-1],
        "pipeline": "hybrid_olmocr_qwen",
        "structured_data": structured,
        "validation": validation,
        "timing": {
            "ocr_seconds": round(ocr_time, 2),
            "structuring_seconds": round(struct_time, 2),
            "validation_seconds": round(val_time, 2),
            "total_seconds": round(total_time, 2),
        },
        "pages_processed": len(pages),
    }

    # Output
    output_json = json.dumps(final, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(output_json)

    print(f"\n  Total time: {total_time:.1f}s (OCR: {ocr_time:.1f}s + Structure: {struct_time:.1f}s + Validate: {val_time:.1f}s)")


if __name__ == "__main__":
    main()
