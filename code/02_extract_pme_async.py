

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from project_config import load_config

CONFIG = load_config(REPO_ROOT)
LLM_CONFIG = CONFIG.get("llm", {})

API_KEY = os.getenv(str(LLM_CONFIG.get("api_key_env", "ARK_API_KEY")))
ARK_URL = str(LLM_CONFIG.get("api_base_url", ""))
MODEL_ID = str(LLM_CONFIG.get("model_id", ""))
BASE_DIR = REPO_ROOT / "work"


DEFAULT_INPUT = BASE_DIR / "sample" / "a61b_llm_input.jsonl"


DEFAULT_PROMPT = BASE_DIR / "sample" / "pme_prompt_v1.txt"


OUTPUT_DIR = BASE_DIR / "llm_outputs"


LOG_DIR = BASE_DIR / "logs"


RAW_OUTPUT = OUTPUT_DIR / "pme_outputs_raw.jsonl"


PARSED_OUTPUT = OUTPUT_DIR / "pme_outputs_parsed.jsonl"


FAILED_OUTPUT = OUTPUT_DIR / "pme_failed.jsonl"


STATUS_FILE = LOG_DIR / "pme_run_status.json"


DEFAULT_CONCURRENCY = int(LLM_CONFIG.get("default_concurrency", 30))
MAX_OUTPUT_TOKENS = int(LLM_CONFIG.get("max_output_tokens", 1200))
REQUEST_TIMEOUT = int(LLM_CONFIG.get("request_timeout_seconds", 90))
MAX_RETRIES = int(LLM_CONFIG.get("max_retries", 5))






FALLBACK_SYSTEM_PROMPT = """
You are extracting structured technical knowledge elements from US patent text.

Input fields:
- title: original patent title.
- abstract: original patent abstract.
- claim_1: the first claim, used especially to identify the solution mechanism.

Extract exactly three elements:
- P: technical problem / technical need / task / improvement target. Existing limitation, defect, risk, unmet need, task to be solved, concrete functional objective, or performance aspect to be improved.
- M: solution mechanism. Core component, structure, material, composition, process step, control logic, device architecture, or implementation method.
- E: technical effect. Explicit or directly implied function, performance improvement, efficiency improvement, safety improvement, stability improvement, risk reduction, reliability improvement, cost reduction, simplification, or new technical capability.

Return only valid JSON. Do not output markdown. Do not add explanation.

Required JSON keys:
{
  "P": {
    "present": true/false,
    "normalized_text": string or null,
    "evidence_span": string or null,
    "source_field": "title" or "abstract" or "claim_1" or null,
    "confidence": number between 0 and 1,
    "absence_reason": string or null
  },
  "M": {
    "present": true/false,
    "normalized_text": string or null,
    "evidence_span": string or null,
    "source_field": "title" or "abstract" or "claim_1" or null,
    "confidence": number between 0 and 1,
    "absence_reason": string or null
  },
  "E": {
    "present": true/false,
    "normalized_text": string or null,
    "evidence_span": string or null,
    "source_field": "title" or "abstract" or "claim_1" or null,
    "confidence": number between 0 and 1,
    "absence_reason": string or null
  }
}

Extraction preference:
- Extract P, M, and E when they are supported by the provided title, abstract, or claim_1. Avoid unnecessary nulls, but do not force every patent to have P and E.
- For P and E, present=true should require more than merely naming the invention's object. There must be a supported technical purpose, need, function, improvement target, capability, or effect.
- Avoid returning null. Use present=false only when the element is genuinely unsupported by any continuous text span in title, abstract, or claim_1.
- Do not leave an element as null merely because the wording is indirect, compressed, legalistic, implicit, or spread across the patent text.
- Prefer concise but informative normalized_text over null.
- Prefer a short, continuous evidence_span that best supports the extracted element.
- M should almost always be extracted if claim_1 describes a concrete technical structure, component, material, composition, process, device, system, or method.
- P and E are often less explicit than M in patent text. Still extract them when the text indicates a technical purpose, intended use, function, improvement direction, target performance, capability, or task.


A61B medical-device-specific guidance:
- In A61B medical-device patents, P may be a diagnostic, surgical, therapeutic, monitoring, imaging, sensing, navigation, sampling, safety, patient-comfort, accuracy, invasiveness, reliability, workflow, calibration, or device-control need.
- Do not mark P as false merely because the patent frames the need as a clinical task or device function rather than using words like problem, drawback, limitation, or need. If a continuous span supports a concrete medical-device task or improvement target, extract it as P.
- In A61B patents, M may be an imaging system, sensor arrangement, catheter, endoscope, surgical instrument, implantable or wearable device, electrode, optical/acoustic/ultrasound component, control algorithm, signal-processing module, data-acquisition method, anatomical navigation method, device structure, or medical workflow mechanism.
- M should focus on the core technical mechanism that makes the diagnostic, surgical, monitoring, imaging, or therapeutic device work. Do not copy a long claim; choose the central clause or mechanism.
- In A61B patents, E may be improved detection, measurement, localization, visualization, diagnosis, treatment guidance, minimally invasive operation, reduced patient risk, reduced discomfort, improved accuracy, faster workflow, improved stability, safer operation, or better monitoring capability.
- Do not mark E as false merely because the text states an enabled clinical or technical capability rather than using words like advantage, effect, or improvement. If a continuous span supports what the medical device enables, improves, reduces, detects, monitors, controls, guides, prevents, or makes possible, extract it as E.
- Still require every present=true P, M, and E to have a continuous verbatim evidence_span from exactly one of title, abstract, or claim_1. Do not invent missing P or E when no span supports them.
Special rules for P:
- P should capture the technical problem, unmet need, task to be solved, or improvement target addressed by the invention.
- P is broader than an explicitly stated defect, but narrower than a general application field or invention object.
- P is present=true when the provided text explicitly supports at least one of the following:
  1. an existing limitation, defect, drawback, disadvantage, risk, failure mode, instability, inefficiency, or uncertainty;
  2. an unmet technical need or technical requirement;
  3. a task to be solved or technical challenge to be addressed;
  4. a performance aspect to be improved, such as capacity, cycle life, safety, stability, conductivity, efficiency, reliability, durability, manufacturability, energy density, power output, control accuracy, or cost;
  5. a functional objective that the invention is designed to achieve, but only when the text states a concrete technical function rather than merely naming an object.
- Do not require explicit words such as "problem", "defect", "drawback", "limitation", or "need".
- If the abstract states a concrete improvement or effect, P may be expressed as the corresponding improvement target. For example, if the text states improved cycle characteristics, P may be the need to improve cycle characteristics.
- If the text states a concrete function or capability, P may be expressed as the technical task of achieving that function or capability.
- Do not mark P as present=true merely because the text names the invention object, technical field, component, material, device, system, or application area.
- Do not treat "a battery", "an electrode", "an electrolyte", "a wind turbine", "a system", "a method", or similar invention objects as P.
- Do not use a title alone as P evidence unless the title contains a concrete function, improvement target, need, risk, or task.
- Prefer abstract as the source_field for P. Use claim_1 only if claim_1 contains a concrete task, functional objective, improvement target, need, risk, or limitation.
- P is allowed to be missing when the text only describes what the invention is or what it comprises, without stating a need, task, function, improvement target, limitation, or risk.
- For P evidence_span, select the shortest continuous verbatim span that directly supports the need, task, function, improvement target, limitation, risk, or problem.

Special rules for E:
- E does not need to be phrased as "advantage" or "effect" in the text.
- If the text states what the invention enables, achieves, improves, reduces, prevents, stabilizes, controls, increases, decreases, simplifies, or makes possible, extract that as E.
- If the effect is expressed as a function of the claimed mechanism, extract the function as E.
- If only a use or capability is stated, E may be the directly supported capability, not an unsupported performance claim.
- Title alone should be used for E only when it explicitly states a technical effect, improvement, performance outcome, risk reduction, function achieved, or new capability.
- A descriptive property in the title, such as "lightweight", "composite", "solid", "secondary", or "non-aqueous", is not sufficient for E unless the text clearly presents it as an achieved effect or advantage.

Special rules for M:
- M should summarize the core solution mechanism, not copy the entire claim.
- The normalized_text for M should usually be 30-80 words. Avoid outputs longer than 120 words unless absolutely necessary.
- For M evidence_span, select the shortest continuous verbatim span that contains the core component, material, structure, composition, process step, device architecture, or method.
- Prefer an evidence_span of 20-60 words from claim_1. If claim_1 is too long or legally complex, choose one representative clause rather than the whole claim.
- Do not use an entire claim as evidence_span unless the claim is very short.
- Do not paraphrase, simplify, normalize symbols, or change punctuation inside evidence_span.

Evidence rules:
- For each present=true element, evidence_span must be a continuous verbatim substring from exactly one source field: title, abstract, or claim_1.
- source_field must be one of: title, abstract, claim_1.
- Do not create an evidence_span by stitching together non-contiguous fragments.
- Do not modify the wording, spelling, punctuation, symbols, units, or order of evidence_span.
- Preserve ASCII patent notation exactly, including d.sub.002, .ANG., wt.%, mol%, g/cc, %, numerals, chemical formulas, and claim references.
- Do not invent information that is not supported by the patent text.
- Do not use general technical knowledge beyond the provided text.
- In normalized_text, preserve patent ASCII notation exactly when it appears in the source text. Do not convert .degree., .ltoreq., .gtoreq., .ANG., d.sub.002, m.sup.2, Li.sub.x, wt.%, mol%, or g/cc into Unicode symbols.
- If the source uses patent notation such as .degree. C. or .ltoreq., keep that notation in normalized_text.

Output discipline:
- Return exactly one JSON object.
- Do not wrap the JSON in code fences.
- Do not include any text before or after the JSON.
- If an element is present=true, normalized_text, evidence_span, source_field, confidence, and absence_reason must be filled as follows:
  - normalized_text: concise standardized expression of the element.
  - evidence_span: verbatim continuous substring from one source field.
  - source_field: title, abstract, or claim_1.
  - confidence: number between 0 and 1.
  - absence_reason: null.
- If an element is present=false, use:
  - normalized_text: null
  - evidence_span: null
  - source_field: null
  - confidence: number between 0 and 1
  - absence_reason: brief reason.
""".strip()






def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_space(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def load_done_request_ids(parsed_path: Path) -> set[str]:
    done = set()
    if not parsed_path.exists():
        return done

    with parsed_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                rid = obj.get("request_id")
                if rid:
                    done.add(rid)
            except Exception:
                continue
    return done


def extract_output_text(response_json: Dict[str, Any]) -> str:

    choices = response_json.get("choices")
    if isinstance(choices, list):
        parts = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue

            message = choice.get("message")
            if not isinstance(message, dict):
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())

        if parts:
            return "\n".join(parts).strip()

    return ""


def strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_model_json(text: str) -> Dict[str, Any]:
    clean = strip_json_fence(text)
    return json.loads(clean)


def validate_element(name: str, elem: Any) -> Tuple[bool, str]:

    if not isinstance(elem, dict):
        return False, f"{name} is not an object"

    if "present" not in elem:
        if elem.get("evidence_span") or elem.get("normalized_text"):
            elem["present"] = True
        else:
            elem["present"] = False

    if not isinstance(elem["present"], bool):
        if str(elem["present"]).lower() == "true":
            elem["present"] = True
        elif str(elem["present"]).lower() == "false":
            elem["present"] = False
        else:
            return False, f"{name}.present is not boolean"

    if "normalized_text" not in elem:
        elem["normalized_text"] = None

    if "evidence_span" not in elem:
        elem["evidence_span"] = None

    if "source_field" not in elem:
        elem["source_field"] = None

    if "confidence" not in elem or elem["confidence"] is None:
        elem["confidence"] = 0.8

    if "absence_reason" not in elem:
        elem["absence_reason"] = None

    if elem["source_field"] not in ("title", "abstract", "claim_1", None):
        return False, f"{name}.source_field invalid: {elem['source_field']}"

    if elem["present"] is False:
        elem["evidence_span"] = None
        elem["source_field"] = None

    return True, ""

def normalize_for_evidence_check(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)

    return text.strip()


def evidence_span_valid(
    elem: Dict[str, Any],
    input_sources: Dict[str, str],
) -> Tuple[Optional[bool], Optional[str]]:
    if not elem.get("present"):
        return None, None

    span = elem.get("evidence_span")
    source_field = elem.get("source_field")

    if not span or not source_field:
        return False, "present=true but evidence_span/source_field is empty"

    source_text = input_sources.get(source_field) or ""

    if span in source_text:
        return True, None

    span_norm = normalize_for_evidence_check(span)
    source_norm = normalize_for_evidence_check(source_text)

    if span_norm and span_norm in source_norm:
        return True, "matched_after_normalization"


    for field_name, field_text in input_sources.items():
        field_norm = normalize_for_evidence_check(field_text)
        if span_norm and span_norm in field_norm:
            return True, f"matched_in_different_source_field:{field_name}"

    return False, f"evidence_span not found in {source_field}"

def build_user_text(record: Dict[str, Any]) -> str:
    src = record["input_sources"]

    return f"""
PATENT_ID: {record["patent_id"]}
FIELDS: {", ".join(record.get("fields", []))}
GRANT_YEAR: {record.get("grant_year")}

TITLE:
{src.get("title", "")}

ABSTRACT:
{src.get("abstract", "")}

CLAIM_1:
{src.get("claim_1", "")}

Return JSON with exactly keys P, M, E.
""".strip()


def raw_response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()






async def call_ark_api(
    client: httpx.AsyncClient,
    system_prompt: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:

    user_text = build_user_text(record)

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": system_prompt + "\n\nYou must output valid JSON only.",
            },
            {
                "role": "user",
                "content": user_text + "\n\nReturn a single valid JSON object only.",
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = await client.post(ARK_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()





async def process_one(
    record: Dict[str, Any],
    client: httpx.AsyncClient,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
) -> Tuple[str, bool]:
    request_id = record["request_id"]
    patent_id = record["patent_id"]

    async with semaphore:
        last_error = None
        start_time = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api_start = time.time()
                resp_json = await call_ark_api(client, system_prompt, record)
                latency = time.time() - api_start

                output_text = extract_output_text(resp_json)
                if not output_text:
                    raise ValueError("empty model output text")

                raw_obj = {
                    "request_id": request_id,
                    "patent_id": patent_id,
                    "model": MODEL_ID,
                    "created_at": now_iso(),
                    "attempt": attempt,
                    "latency_seconds": round(latency, 3),
                    "usage": resp_json.get("usage"),
                    "response_id": resp_json.get("id"),
                    "status": resp_json.get("status"),
                    "raw_response": output_text,
                    "raw_response_sha256": raw_response_hash(output_text),
                }


                async with write_lock:
                    append_jsonl(RAW_OUTPUT, raw_obj)


                model_obj = parse_model_json(output_text)

                for elem_name in ["P", "M", "E"]:
                    ok, msg = validate_element(elem_name, model_obj.get(elem_name))
                    if not ok:
                        raise ValueError(msg)

                parsed = {
                    "request_id": request_id,
                    "patent_id": patent_id,
                    "fields": record.get("fields"),
                    "sample_roles": record.get("sample_roles"),
                    "grant_year": record.get("grant_year"),
                    "grant_date": record.get("grant_date"),
                    "model": MODEL_ID,
                    "created_at": raw_obj["created_at"],
                    "attempt": attempt,
                }

                all_span_valid = True

                for elem_name in ["P", "M", "E"]:
                    elem = model_obj[elem_name]
                    span_ok, span_note = evidence_span_valid(elem, record["input_sources"])

                    parsed[f"{elem_name}_present"] = elem.get("present")
                    parsed[f"{elem_name}_text"] = elem.get("normalized_text")
                    parsed[f"{elem_name}_evidence_span"] = elem.get("evidence_span")
                    parsed[f"{elem_name}_source_field"] = elem.get("source_field")
                    parsed[f"{elem_name}_confidence"] = elem.get("confidence")
                    parsed[f"{elem_name}_absence_reason"] = elem.get("absence_reason")
                    parsed[f"{elem_name}_span_valid"] = span_ok
                    parsed[f"{elem_name}_span_note"] = span_note

                    if span_ok is False:
                        all_span_valid = False

                parsed["all_present_spans_valid"] = all_span_valid




                async with write_lock:

                    append_jsonl(PARSED_OUTPUT, parsed)

                return request_id, True

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                text = e.response.text[:1000]
                last_error = f"HTTP {status}: {text}"

                if status == 429 or status >= 500:
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
                    continue

                break

            except Exception as e:
                last_error = repr(e)
                await asyncio.sleep(min(30, 2 ** attempt + random.random()))

        failed_obj = {
            "request_id": request_id,
            "patent_id": patent_id,
            "model": MODEL_ID,
            "created_at": now_iso(),
            "error": last_error,
            "max_retries": MAX_RETRIES,
            "elapsed_seconds": round(time.time() - start_time, 3),
        }

        async with write_lock:
            append_jsonl(FAILED_OUTPUT, failed_obj)

        return request_id, False






async def run(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not API_KEY:
        raise RuntimeError("Set the environment variable named by llm.api_key_env before running this script.")
    if not ARK_URL or not MODEL_ID:
        raise RuntimeError("Set llm.api_base_url and llm.model_id in config.yaml before running this script.")

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")



    system_prompt = FALLBACK_SYSTEM_PROMPT
    print("Using the built-in FALLBACK_SYSTEM_PROMPT; the external prompt file is not read.")

    done_ids = load_done_request_ids(PARSED_OUTPUT)

    all_records = list(read_jsonl(input_path))

    if args.limit:
        all_records = all_records[: args.limit]

    pending = [r for r in all_records if r["request_id"] not in done_ids]

    total = len(all_records)
    skipped = len(all_records) - len(pending)

    print(f"Input records: {total}")
    print(f"Previously completed and skipped: {skipped}")
    print(f"Pending records: {len(pending)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Model: {MODEL_ID}")
    print(f"Start time: {now_iso()}")

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    limits = httpx.Limits(
        max_connections=args.concurrency + 20,
        max_keepalive_connections=args.concurrency + 20,
    )

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    success_count = 0
    fail_count = 0
    start = time.time()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                process_one(
                    record=record,
                    client=client,
                    system_prompt=system_prompt,
                    semaphore=semaphore,
                    write_lock=write_lock,
                )
            )
            for record in pending
        ]

        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            _, ok = await task

            if ok:
                success_count += 1
            else:
                fail_count += 1

            elapsed = time.time() - start
            speed = idx / elapsed * 60 if elapsed > 0 else 0
            remaining = len(pending) - idx
            eta_min = remaining / speed if speed > 0 else None

            if idx % args.progress_every == 0 or idx == len(pending):
                status = {
                    "updated_at": now_iso(),
                    "model": MODEL_ID,
                    "input_total": total,
                    "skipped_done": skipped,
                    "pending_total": len(pending),
                    "processed_this_run": idx,
                    "success_this_run": success_count,
                    "failed_this_run": fail_count,
                    "speed_records_per_min": round(speed, 2),
                    "eta_minutes": round(eta_min, 2) if eta_min is not None else None,
                    "raw_output": str(RAW_OUTPUT),
                    "parsed_output": str(PARSED_OUTPUT),
                    "failed_output": str(FAILED_OUTPUT),
                }

                STATUS_FILE.write_text(
                    json.dumps(status, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                eta_text = f"{eta_min:.1f} minutes" if eta_min is not None else "unknown"

                print(
                    f"[{now_iso()}] {idx}/{len(pending)} "
                    f"succeeded={success_count} failed={fail_count} "
                    f"rate={speed:.1f} records/minute ETA={eta_text}"
                )

    print("Run complete.")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"raw: {RAW_OUTPUT}")
    print(f"parsed: {PARSED_OUTPUT}")
    print(f"failed: {FAILED_OUTPUT}")
    print(f"status: {STATUS_FILE}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the LLM input JSONL file",
    )

    parser.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT),
        help="Path to the system-prompt file",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N records for testing, e.g. --limit 200",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Asynchronous concurrency, e.g. --concurrency 80",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress after this many processed records",
    )

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
