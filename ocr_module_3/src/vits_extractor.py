import base64
import json
import re
import uuid
import yaml
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from vllm import LLM, SamplingParams
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
import PyPDF2
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging
import time

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.settings import settings

logger = logging.getLogger(__name__)


class OlmOCRProcessor:
    """OCR processor using OlmOCR-2-7B with vLLM batch inference and ADE capabilities."""

    def __init__(
        self,
        model_name: str = None,
        max_workers: int = None
    ):
        self.model_name = model_name or settings.model_name
        self.max_workers = max_workers or settings.max_workers

        # Determine model path (local or HuggingFace)
        model_path = settings.get_model_path(self.model_name)
        self._model_path = str(model_path) if model_path.exists() else self.model_name
        model_to_load = self._model_path

        logger.info(f"Loading model via vLLM: {model_to_load}")

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_to_load,
            max_model_len=settings.vllm_max_model_len,
            max_num_seqs=settings.vllm_max_num_seqs,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            tensor_parallel_size=settings.vllm_tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )

        # Default sampling params (used for batch OCR first pass)
        self.sampling_params = SamplingParams(
            temperature=settings.initial_temperature,
            max_tokens=2048,
        )

        # Cache the prompt text (it never changes)
        self._prompt_text = build_no_anchoring_v4_yaml_prompt()

        logger.info("OlmOCR processor initialized with vLLM (v2 + ADE)")

    # ──────────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────────

    def get_pdf_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get total number of pages in PDF."""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)

    def _parse_yaml_output(self, raw_text: str) -> Dict:
        """Parse YAML output from model into structured dict.

        Returns:
            {"parsed": dict|None, "raw": str, "parse_success": bool}
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        text = re.sub(r'^```(?:yaml|yml)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        try:
            parsed = yaml.safe_load(text)
            if parsed is None:
                return {"parsed": None, "raw": raw_text, "parse_success": False}
            return {"parsed": parsed, "raw": raw_text, "parse_success": True}
        except yaml.YAMLError:
            return {"parsed": None, "raw": raw_text, "parse_success": False}

    def _infer_with_retry(self, conversation: list) -> Dict:
        """Infer a single page with retry logic on YAML parse failure.

        Uses temperature escalation from initial_temperature to max_temperature.

        Returns:
            {"text": str, "parsed": dict|None, "retries": int, "parse_success": bool}
        """
        temp = settings.initial_temperature
        best_text = ""
        best_parsed = None

        for attempt in range(settings.max_page_retries):
            params = SamplingParams(temperature=temp, max_tokens=2048)

            try:
                outputs = self.llm.chat(
                    messages=[conversation],
                    sampling_params=params,
                )
                text = outputs[0].outputs[0].text
            except Exception as e:
                logger.warning(f"Inference attempt {attempt + 1} failed: {e}")
                temp = min(temp + settings.temperature_step, settings.max_temperature)
                continue

            result = self._parse_yaml_output(text)

            if not best_text:
                best_text = text
                best_parsed = result["parsed"]

            if result["parse_success"]:
                return {
                    "text": text,
                    "parsed": result["parsed"],
                    "retries": attempt,
                    "parse_success": True,
                }

            # Keep best result (longest text)
            if len(text) > len(best_text):
                best_text = text
                best_parsed = result["parsed"]

            temp = min(temp + settings.temperature_step, settings.max_temperature)
            logger.info(f"  Retry {attempt + 1}: YAML parse failed, escalating temperature to {temp:.2f}")

        return {
            "text": best_text,
            "parsed": best_parsed,
            "retries": settings.max_page_retries,
            "parse_success": best_parsed is not None,
        }

    # ──────────────────────────────────────────────
    # PDF / Image Rendering
    # ──────────────────────────────────────────────

    def _render_page(
        self,
        pdf_path: str,
        page_num: int,
        target_dim: int = None
    ) -> Dict:
        """Render a single PDF page and build chat message.

        Designed to run in ThreadPoolExecutor for parallel rendering.
        """
        if target_dim is None:
            target_dim = settings.target_image_dim

        image_base64 = render_pdf_to_base64png(
            str(pdf_path),
            page_num,
            target_longest_image_dim=target_dim
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]

        return {"page_num": page_num, "conversation": conversation}

    # ──────────────────────────────────────────────
    # Core Processing Methods
    # ──────────────────────────────────────────────

    def process_pdf_page(
        self,
        pdf_path: Union[str, Path],
        page_num: int,
        target_dim: int = None
    ) -> str:
        """Process a single page of PDF."""
        try:
            rendered = self._render_page(str(pdf_path), page_num, target_dim)
            result = self._infer_with_retry(rendered["conversation"])
            return result["text"]
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            raise

    def process_image(
        self,
        image_path: Union[str, Path],
        target_dim: int = 1288
    ) -> Dict:
        """Process a single image file with retry logic.

        Returns:
            Dict with text, parsed, retries, parse_success
        """
        try:
            image = Image.open(image_path).convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            return self._infer_with_retry(conversation)

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def process_document(
        self,
        file_path: Union[str, Path],
        conversion_type: str = "yaml",
        request_id: str = None
    ) -> Dict:
        """Process a document (PDF or image) with vLLM batch inference, retry, and per-page detail.

        Returns dictionary with extracted content, per-page results, and timing metrics.
        """
        file_path = Path(file_path)
        request_id = request_id or str(uuid.uuid4())
        start_total = time.time()

        try:
            logger.info(f"[{request_id[:8]}] Processing document: {file_path.name}")
            file_ext = file_path.suffix.lower()

            if file_ext == '.pdf':
                total_pages = self.get_pdf_page_count(file_path)
                logger.info(f"[{request_id[:8]}] PDF has {total_pages} pages")

                # Phase 1: Parallel page rendering (CPU-bound I/O)
                start_render = time.time()
                rendered_pages = []
                with ThreadPoolExecutor(max_workers=min(total_pages, 8)) as executor:
                    futures = {
                        executor.submit(
                            self._render_page, str(file_path), page_num
                        ): page_num
                        for page_num in range(1, total_pages + 1)
                    }
                    for future in as_completed(futures):
                        rendered_pages.append(future.result())

                rendered_pages.sort(key=lambda x: x["page_num"])
                render_time_ms = int((time.time() - start_render) * 1000)
                logger.info(f"[{request_id[:8]}] Rendered {total_pages} pages in {render_time_ms}ms")

                # Phase 2: Batch inference + selective retry
                start_infer = time.time()
                conversations = [rp["conversation"] for rp in rendered_pages]

                outputs = self.llm.chat(
                    messages=conversations,
                    sampling_params=self.sampling_params,
                    use_tqdm=True,
                )

                # Parse each output and identify failures
                page_results = []
                failed_indices = []
                retries_total = 0

                for i, rp in enumerate(rendered_pages):
                    page_text = outputs[i].outputs[0].text
                    parsed_result = self._parse_yaml_output(page_text)

                    if parsed_result["parse_success"]:
                        page_results.append({
                            "page_num": rp["page_num"],
                            "content": page_text,
                            "parsed": parsed_result["parsed"],
                            "status": "success",
                            "retries": 0,
                            "parse_success": True,
                        })
                    else:
                        failed_indices.append(i)
                        page_results.append({
                            "page_num": rp["page_num"],
                            "content": page_text,
                            "parsed": None,
                            "status": "failed",
                            "retries": 0,
                            "parse_success": False,
                        })

                # Retry failed pages individually with temperature escalation
                if failed_indices:
                    logger.info(
                        f"[{request_id[:8]}] {len(failed_indices)} pages failed YAML parse, retrying..."
                    )
                    for idx in failed_indices:
                        retry_result = self._infer_with_retry(
                            rendered_pages[idx]["conversation"]
                        )
                        retries_total += retry_result["retries"]
                        page_results[idx] = {
                            "page_num": rendered_pages[idx]["page_num"],
                            "content": retry_result["text"],
                            "parsed": retry_result["parsed"],
                            "status": "success" if retry_result["parse_success"] else "failed",
                            "retries": retry_result["retries"],
                            "parse_success": retry_result["parse_success"],
                        }

                inference_time_ms = int((time.time() - start_infer) * 1000)
                logger.info(f"[{request_id[:8]}] Inference completed in {inference_time_ms}ms")

                # Phase 3: Assemble results
                all_text_parts = []
                for pr in page_results:
                    all_text_parts.append(f"--- PAGE {pr['page_num']} ---\n{pr['content']}\n")

                content = "\n".join(all_text_parts)

            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                start_render = time.time()
                render_time_ms = 0

                start_infer = time.time()
                img_result = self.process_image(file_path)
                inference_time_ms = int((time.time() - start_infer) * 1000)

                content = img_result["text"]
                total_pages = 1
                retries_total = img_result["retries"]
                page_results = [{
                    "page_num": 1,
                    "content": img_result["text"],
                    "parsed": img_result["parsed"],
                    "status": "success" if img_result["parse_success"] else "failed",
                    "retries": img_result["retries"],
                    "parse_success": img_result["parse_success"],
                }]

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            pages_succeeded = sum(1 for p in page_results if p["status"] == "success")
            pages_failed = total_pages - pages_succeeded
            total_time_ms = int((time.time() - start_total) * 1000)

            if pages_failed == 0:
                status = "success"
            elif pages_succeeded > 0:
                status = "partial"
            else:
                status = "failed"

            return {
                "document_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "conversion_type": conversion_type,
                "content": content,
                "pages": page_results,
                "status": status,
                "error_message": None,
                "metadata": {
                    "num_pages": total_pages,
                    "pages_succeeded": pages_succeeded,
                    "pages_failed": pages_failed,
                    "retries_total": retries_total,
                    "render_time_ms": render_time_ms,
                    "inference_time_ms": inference_time_ms,
                    "total_time_ms": total_time_ms,
                    "file_type": file_ext,
                    "request_id": request_id,
                }
            }

        except Exception as e:
            logger.error(f"[{request_id[:8]}] Error processing {file_path}: {str(e)}")
            total_time_ms = int((time.time() - start_total) * 1000)
            return {
                "document_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "conversion_type": conversion_type,
                "content": None,
                "pages": [],
                "status": "failed",
                "error_message": str(e),
                "metadata": {
                    "num_pages": 0,
                    "pages_succeeded": 0,
                    "pages_failed": 0,
                    "retries_total": 0,
                    "render_time_ms": 0,
                    "inference_time_ms": 0,
                    "total_time_ms": total_time_ms,
                    "file_type": file_path.suffix.lower(),
                    "request_id": request_id,
                }
            }

    # ──────────────────────────────────────────────
    # ADE: Agentic Document Extraction
    # ──────────────────────────────────────────────

    def extract_with_schema(
        self,
        file_path: Union[str, Path],
        schema: Dict,
        request_id: str = None
    ) -> Dict:
        """Agentic Document Extraction: schema-driven, multi-pass, self-validating.

        4-pass pipeline:
          Pass 1 - OCR extraction (batch vLLM inference)
          Pass 2 - Schema-guided structuring (single vLLM call)
          Pass 3 - Validation (Python)
          Pass 4 - Targeted re-extraction for missing fields (batch vLLM)

        Args:
            file_path: Path to document
            schema: Extraction schema dict with required_fields, optional_fields, etc.
            request_id: Optional tracking ID

        Returns:
            Dict with structured_json, validation results, and full metadata
        """
        file_path = Path(file_path)
        request_id = request_id or str(uuid.uuid4())
        start_total = time.time()
        schema_name = schema.get("document_type", "unknown")

        logger.info(f"[{request_id[:8]}] ADE starting for {file_path.name} with schema '{schema_name}'")

        # ── Pass 1: OCR Extraction ──
        start_ocr = time.time()
        ocr_result = self.process_document(file_path, request_id=request_id)
        ocr_time_ms = int((time.time() - start_ocr) * 1000)

        if ocr_result["status"] == "failed" and not ocr_result.get("content"):
            return {
                "document_name": file_path.name,
                "request_id": request_id,
                "schema_used": schema_name,
                "structured_json": None,
                "missing_fields": schema.get("required_fields", []),
                "validation_errors": ["OCR extraction failed completely"],
                "validation_passed": False,
                "raw_ocr_content": None,
                "pages": ocr_result.get("pages", []),
                "status": "failed",
                "error_message": ocr_result.get("error_message"),
                "metadata": {
                    "num_pages": 0,
                    "ocr_time_ms": ocr_time_ms,
                    "structuring_time_ms": 0,
                    "validation_time_ms": 0,
                    "reextraction_time_ms": 0,
                    "total_time_ms": int((time.time() - start_total) * 1000),
                    "retries_total": 0,
                    "fields_extracted": 0,
                    "fields_required": len(schema.get("required_fields", [])),
                    "fields_missing": len(schema.get("required_fields", [])),
                }
            }

        raw_ocr_content = ocr_result["content"]

        # ── Pass 2: Schema-Guided Structuring ──
        start_struct = time.time()
        structured_json = self._schema_structuring_pass(raw_ocr_content, schema)
        structuring_time_ms = int((time.time() - start_struct) * 1000)
        logger.info(f"[{request_id[:8]}] Structuring pass completed in {structuring_time_ms}ms")

        # ── Pass 3: Validation ──
        start_val = time.time()
        missing_fields, validation_errors = self._validate_against_schema(
            structured_json, schema
        )
        validation_time_ms = int((time.time() - start_val) * 1000)
        logger.info(
            f"[{request_id[:8]}] Validation: {len(missing_fields)} missing, "
            f"{len(validation_errors)} errors"
        )

        # ── Pass 4: Targeted Re-extraction (if needed) ──
        reextraction_time_ms = 0
        if missing_fields:
            start_reext = time.time()
            structured_json = self._targeted_reextraction(
                raw_ocr_content, structured_json, missing_fields, schema
            )
            reextraction_time_ms = int((time.time() - start_reext) * 1000)

            # Re-validate after re-extraction
            missing_fields, validation_errors = self._validate_against_schema(
                structured_json, schema
            )
            logger.info(
                f"[{request_id[:8]}] After re-extraction: {len(missing_fields)} still missing"
            )

        total_time_ms = int((time.time() - start_total) * 1000)
        validation_passed = len(missing_fields) == 0 and len(validation_errors) == 0

        required_count = len(schema.get("required_fields", []))
        extracted_count = required_count - len(missing_fields)

        if validation_passed:
            status = "success"
        elif extracted_count > 0:
            status = "partial"
        else:
            status = "failed"

        logger.info(
            f"[{request_id[:8]}] ADE complete: {status} | "
            f"{extracted_count}/{required_count} fields | {total_time_ms}ms total"
        )

        return {
            "document_name": file_path.name,
            "request_id": request_id,
            "schema_used": schema_name,
            "structured_json": structured_json,
            "missing_fields": missing_fields,
            "validation_errors": validation_errors,
            "validation_passed": validation_passed,
            "raw_ocr_content": raw_ocr_content,
            "pages": ocr_result.get("pages", []),
            "status": status,
            "error_message": None,
            "metadata": {
                "num_pages": ocr_result.get("metadata", {}).get("num_pages", 0),
                "ocr_time_ms": ocr_time_ms,
                "structuring_time_ms": structuring_time_ms,
                "validation_time_ms": validation_time_ms,
                "reextraction_time_ms": reextraction_time_ms,
                "total_time_ms": total_time_ms,
                "retries_total": ocr_result.get("metadata", {}).get("retries_total", 0),
                "fields_extracted": extracted_count,
                "fields_required": required_count,
                "fields_missing": len(missing_fields),
            }
        }

    def _schema_structuring_pass(self, ocr_text: str, schema: Dict) -> Dict:
        """Pass 2: Use vLLM to extract structured JSON from OCR text guided by schema."""
        required_fields = schema.get("required_fields", [])
        optional_fields = schema.get("optional_fields", [])
        field_formats = schema.get("field_formats", {})
        doc_type = schema.get("document_type", "document")

        # Build field descriptions
        req_desc = []
        for f in required_fields:
            fmt = field_formats.get(f, "text")
            req_desc.append(f'  "{f}": {fmt}')

        opt_desc = []
        for f in optional_fields:
            fmt = field_formats.get(f, "text")
            opt_desc.append(f'  "{f}": {fmt}')

        prompt = f"""You are a precise data extraction agent. Given the OCR text below from a {doc_type}, extract ONLY the following fields and return valid JSON.

Required fields (must extract):
{chr(10).join(req_desc)}

Optional fields (extract if present):
{chr(10).join(opt_desc)}

Rules:
1. Only extract information explicitly stated in the text
2. Do not infer or guess missing information
3. Return null for fields that are not found in the text
4. Preserve exact formatting for dates, codes, and IDs
5. For financial amounts, return digits only (no commas or currency symbols)
6. Return ONLY a valid JSON object, no additional text or explanation

OCR Text:
{ocr_text[:6000]}

Return ONLY valid JSON:"""

        conversation = [{"role": "user", "content": prompt}]

        params = SamplingParams(temperature=0, max_tokens=2048)

        try:
            outputs = self.llm.chat(
                messages=[conversation],
                sampling_params=params,
            )
            response_text = outputs[0].outputs[0].text.strip()

            # Strip markdown code fences
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?```\s*$', '', response_text)
            response_text = response_text.strip()

            structured = json.loads(response_text)
            return structured

        except json.JSONDecodeError:
            logger.warning("Structuring pass returned invalid JSON, attempting YAML fallback")
            try:
                structured = yaml.safe_load(response_text)
                if isinstance(structured, dict):
                    return structured
            except Exception:
                pass

            # Last resort: return empty dict with nulls
            result = {f: None for f in required_fields + optional_fields}
            return result

        except Exception as e:
            logger.error(f"Structuring pass failed: {e}")
            return {f: None for f in required_fields + optional_fields}

    def _validate_against_schema(
        self, structured_json: Dict, schema: Dict
    ) -> tuple:
        """Pass 3: Validate extracted data against schema.

        Returns:
            (missing_fields: list, validation_errors: list)
        """
        if not structured_json:
            return schema.get("required_fields", []), ["No structured data to validate"]

        missing_fields = []
        validation_errors = []
        field_formats = schema.get("field_formats", {})
        financial_fields = schema.get("financial_fields", [])

        # Check required fields
        for field in schema.get("required_fields", []):
            value = structured_json.get(field)
            if value is None or value == "" or value == "null":
                missing_fields.append(field)
                continue

            # Validate financial fields are numeric
            if field in financial_fields:
                clean_val = str(value).replace(",", "").replace(" ", "").strip()
                if not clean_val.replace(".", "").isdigit():
                    validation_errors.append(
                        f"Financial field '{field}' has non-numeric value: '{value}'"
                    )

        # Validate field formats for present fields
        for field, fmt in field_formats.items():
            value = structured_json.get(field)
            if value is None or value == "":
                continue

            str_val = str(value).strip()

            if "numeric" in fmt.lower():
                clean = str_val.replace(",", "").replace(" ", "")
                if not clean.replace(".", "").isdigit():
                    validation_errors.append(
                        f"Field '{field}' expected {fmt}, got '{str_val}'"
                    )

            elif "10-digit" in fmt.lower():
                digits = re.sub(r'\D', '', str_val)
                if len(digits) != 10:
                    validation_errors.append(
                        f"Field '{field}' expected 10-digit number, got '{str_val}'"
                    )

            elif "10 chars" in fmt.lower():
                if len(str_val) != 10:
                    validation_errors.append(
                        f"Field '{field}' expected 10 chars, got {len(str_val)} chars"
                    )

        return missing_fields, validation_errors

    def _targeted_reextraction(
        self,
        ocr_text: str,
        structured_json: Dict,
        missing_fields: List[str],
        schema: Dict
    ) -> Dict:
        """Pass 4: Re-extract specific missing fields with focused prompts."""
        field_formats = schema.get("field_formats", {})

        logger.info(f"Targeted re-extraction for {len(missing_fields)} fields: {missing_fields}")

        # Build a single prompt asking for all missing fields
        field_list = []
        for field in missing_fields:
            fmt = field_formats.get(field, "text")
            field_list.append(f'  - "{field}" (format: {fmt})')

        prompt = f"""From the document text below, find the values for these specific fields.
Return ONLY a valid JSON object with the field names as keys.

Fields to find:
{chr(10).join(field_list)}

Rules:
1. Only extract values explicitly present in the text
2. Return null if a field cannot be found
3. For financial amounts, return digits only
4. Return ONLY valid JSON

Document text:
{ocr_text[:6000]}

JSON:"""

        conversation = [{"role": "user", "content": prompt}]
        params = SamplingParams(temperature=0.2, max_tokens=1024)

        try:
            outputs = self.llm.chat(
                messages=[conversation],
                sampling_params=params,
            )
            response_text = outputs[0].outputs[0].text.strip()

            # Strip markdown code fences
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?```\s*$', '', response_text)

            reextracted = json.loads(response_text.strip())

            # Merge non-null values into structured_json
            for field, value in reextracted.items():
                if value is not None and value != "" and value != "null":
                    structured_json[field] = value
                    logger.info(f"  Re-extracted '{field}': {value}")

        except Exception as e:
            logger.warning(f"Targeted re-extraction failed: {e}")

        return structured_json

    # ──────────────────────────────────────────────
    # Batch & Stats
    # ──────────────────────────────────────────────

    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        conversion_type: str = "yaml"
    ) -> List[Dict]:
        """Process multiple documents."""
        logger.info(f"Processing batch of {len(file_paths)} documents")

        results = []
        for file_path in file_paths:
            result = self.process_document(file_path, conversion_type)
            results.append(result)

        successful = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Batch complete: {successful}/{len(results)} successful")

        return results

    def get_stats(self, results: List[Dict]) -> Dict:
        """Get processing statistics."""
        total = len(results)
        successful = sum(1 for r in results if r["status"] == "success")
        failed = total - successful

        return {
            "total_documents": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0
        }

    # ──────────────────────────────────────────────
    # Model Status & Cleanup
    # ──────────────────────────────────────────────

    def get_model_status(self) -> Dict:
        """Return model info and GPU stats."""
        status = {
            "model_name": self.model_name,
            "model_path": self._model_path,
            "loaded": hasattr(self, 'llm') and self.llm is not None,
            "vllm_config": {
                "max_model_len": settings.vllm_max_model_len,
                "max_num_seqs": settings.vllm_max_num_seqs,
                "gpu_memory_utilization": settings.vllm_gpu_memory_utilization,
                "tensor_parallel_size": settings.vllm_tensor_parallel_size,
            },
        }

        try:
            import torch
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
                )
                status["gpu_memory_used_gb"] = round(
                    torch.cuda.memory_allocated(0) / (1024**3), 2
                )
        except ImportError:
            pass

        return status

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            del self.llm
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Cleanup complete")


if __name__ == "__main__":
    processor = OlmOCRProcessor()

    test_file = "/home/arcaai/Ananth_WSL_Workspace/OCR/Test/data/Tampered.png"
    result = processor.process_document(test_file)

    print(f"\nDocument: {result['document_name']}")
    print(f"Status: {result['status']}")
    print(f"\nExtracted Content:\n{result['content'][:500]}...")

    processor.cleanup()
