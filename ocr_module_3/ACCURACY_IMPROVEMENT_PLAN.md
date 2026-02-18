# Hybrid ADE Pipeline v2: Accuracy Improvement Plan

## Current State (v1)
- **Accuracy**: 30% field-level on health insurance claim form
- **Pipeline**: 3-stage (OlmOCR OCR → Qwen structuring → Qwen validation)
- **Timing**: ~174s total (OCR: 166s, Structure: 3s, Validate: 5s)
- **Architecture**: Dual vLLM on H200 — OlmOCR port 8093 (0.55 GPU) + Qwen port 8000 (0.40 GPU)

## Root Causes of Low Accuracy

### 1. Destructive Truncation
`_truncate_page()` strips all `<table>` HTML which contains financial data (hospitalization_expenses, post_hospitalization_expenses, pharmacy_bill_amount). Stage 3 validation truncates to 1500 chars/page, losing critical data.

### 2. All-Pages-At-Once Prompting
Sending 3 pages of OCR text with 34 fields in a single prompt exceeds Qwen's effective context. No indication of which fields appear on which page.

### 3. No Retry/Re-extraction
Unlike `vits_extractor.py` which has a 4-pass pipeline with targeted re-extraction, `hybrid_extract.py` makes exactly one structuring call. If OCR misreads a value, there's no second attempt.

### 4. Page 3 Skipped
Page 3 contains declaration_date and declaration_place (handwritten), but is filtered out entirely.

### 5. No Deterministic Validation
No Python-level checks for format validity (dates, PAN, IFSC, mobile), financial reconciliation (amounts should sum to total), or cross-page consistency.

## Proposed v2 Architecture: 5-Pass Pipeline

```
Pass 1: OlmOCR  → Raw OCR per page (higher image resolution: 1792 vs 1288)
Pass 2: Qwen    → Page-specific extraction (one call per page, ~6 fields each)
Pass 3: Python  → Deterministic validation (format, financial reconciliation)
Pass 4: Qwen    → Targeted re-extraction of missing/failed fields only
Pass 5: Python  → Normalization and cleanup
```

### Pass 1: Higher Resolution OCR
- Set `TARGET_IMAGE_DIM=1792` (from 1288) on H200 instance
- Better handwriting recognition at higher pixel density
- Within OlmOCR's max_pixels limit (1280 * 28 * 28 = 1,003,520)

### Pass 2: Page-Specific Extraction
Add `PAGE_FIELD_MAP` mapping each page to its specific fields with section hints:

| Page | Fields | Section Hints |
|------|--------|---------------|
| 1 | insured_name, patient_name, mobile_no, email_id, date_of_claim_submission | Claim acknowledgment, document checklist |
| 2 | policy_no, gender, age, hospital, dates, financial amounts, PAN, bank, IFSC | Part A: Sections A/C/D/E/G |
| 3 | declaration_date, declaration_place | Declaration page, handwritten near signature |
| 4 | doctor, qualification, registration_no, diagnosis, total_claimed_amount, IP reg | Part B: Hospital section |

Each Qwen call: ~4000 chars OCR + ~6 fields + hints = ~1600 tokens (well within 8192 limit).

### Pass 3: Python Validation
- Required field presence check
- Date format: DD/MM/YY or DD/MM/YYYY
- Financial fields: must be numeric
- PAN: AAAAA9999A format
- IFSC: 4 letters + 7 alphanumeric
- Mobile: 10 digits
- **Financial reconciliation**: hospitalization + post_hospitalization ≤ total_claimed

### Pass 4: Targeted Re-extraction
- Only for fields that failed Pass 3 validation
- Prompt includes previous wrong value + common OCR error patterns
- Higher temperature (0.2) for diverse interpretation
- Page-specific, using fuller content (5000 chars)

### Pass 5: Normalization
- Financial values → pure digits
- Date normalization
- PAN/IFSC → uppercase
- Remove null-like values ("[]", "[ ]", "N/A")
- Merge page-variant duplicates

## Fix: `_truncate_page()`
- **Stop stripping tables** — financial data lives in table markup
- Keep head + tail (financial summaries at page bottom)
- Increase default from 3000 to 4000 chars
- Only remove checkbox placeholders and excess whitespace

## Expected Results

| Metric | v1 (Current) | v2 (Target) |
|--------|-------------|-------------|
| Field accuracy | 30% | 70-80% |
| Financial accuracy | ~25% | 75%+ |
| Handwritten accuracy | ~30% | 65-75% |
| Total time | 174s | 200-250s |
| Qwen calls | 2 | 5-6 |

## Files to Modify
- `ocr_module_3/hybrid_extract.py` — rewrite pipeline logic (~70% of file)
- H200 instance: env var `TARGET_IMAGE_DIM=1792`, restart OlmOCR

## Verification Steps
1. Upload updated `hybrid_extract.py` to H200
2. Restart OlmOCR with higher resolution
3. Run: `python hybrid_extract.py "Health Claim form.pdf" -o /tmp/hybrid_v2_result.json`
4. Evaluate: `python tests/check_accuracy.py /tmp/hybrid_v2_result.json`
5. Target: 70%+ weighted accuracy

## v1 Benchmark Results (for comparison)

```json
{
  "correct_fields": ["insured_name", "patient_name", "hospital_name", "gender_part_a",
                     "gender_part_b", "age_years", "type_of_admission",
                     "total_claimed_amount", "date_of_claim_submission"],
  "wrong_fields": ["policy_no", "mobile_no", "email_id", "hospital_type", "treating_doctor",
                   "doctor_registration_no", "date_of_admission", "date_of_discharge",
                   "hospitalization_due_to", "primary_diagnosis", "additional_diagnosis",
                   "hospitalization_expenses", "post_hospitalization_expenses", "pan",
                   "bank_name", "ifsc_code", "ip_registration_number",
                   "declaration_date", "declaration_place"],
  "score": "9/30 (30%)"
}
```
