"""Run a resumable P/M/E extraction batch through a compatible chat API."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

from project_config import setting, work_dir
from core import ROLES, iter_inputs, iter_jsonl, parse_response, span_check, user_text

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/pme.txt"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def manifest_for(args, prompt: str) -> dict:
    fingerprint = hashlib.sha256()
    with args.input.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(block)
    return {
        "model": args.model,
        "endpoint": args.endpoint,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_sha256": fingerprint.hexdigest(),
        "temperature": args.temperature,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "thinking_disabled": args.disable_thinking,
    }


async def run(args, prompt: str, manifest: dict):
    import httpx

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set the {args.api_key_env} environment variable before --send")
    if not args.endpoint or not args.model:
        raise SystemExit("Set extraction.endpoint and extraction.model, or pass --endpoint and --model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "batch_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise SystemExit("Batch settings/input differ from this output directory; use a new --output-dir")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    parsed_path = args.output_dir / "parsed.jsonl"
    completed = {record["request_id"] for record in iter_jsonl(parsed_path)} if parsed_path.exists() else set()
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(2, args.concurrency * 2))
    lock = asyncio.Lock()
    counts = {"completed": 0, "failed": 0, "skipped_completed": 0, "requests_sent": 0}

    async def append(path, record):
        async with lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def worker(client):
        while True:
            record = await queue.get()
            if record is None:
                queue.task_done()
                return
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": prompt + "\n\nYou must output valid JSON only."},
                    {"role": "user", "content": user_text(record) + "\n\nReturn a single valid JSON object only."},
                ],
                "temperature": args.temperature,
                "top_p": 1,
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            if args.disable_thinking:
                payload["thinking"] = {"type": "disabled"}
            last_error = "UNKNOWN"
            succeeded = False
            for attempt in range(1, args.max_attempts + 1):
                try:
                    counts["requests_sent"] += 1
                    response = await client.post(
                        args.endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    )
                    if response.status_code >= 400:
                        last_error = f"HTTP_{response.status_code}"
                        if response.status_code not in (408, 409, 429) and response.status_code < 500:
                            break
                        response.raise_for_status()
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                    elements = parse_response(content)
                    span_status = {name: span_check(elements[name], record["input_sources"])[2] for name in ROLES}
                    output = {
                        "request_id": record["request_id"],
                        "patent_id": record["patent_id"],
                        "grant_year": record.get("grant_year"),
                        "input_sources_sha256": record["input_sources_sha256"],
                        "system_prompt_sha256": manifest["system_prompt_sha256"],
                        "model": args.model,
                        "created_at": utc_now(),
                        "attempt": attempt,
                        "usage": body.get("usage"),
                        "elements": elements,
                        "span_status": span_status,
                    }
                    # Invalid roles remain recorded and become missing in prepare_roles.py.
                    # They do not trigger a full-patent redraw or replace valid roles.
                    await append(parsed_path, output)
                    counts["completed"] += 1
                    succeeded = True
                    break
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
                    if not isinstance(error, httpx.HTTPStatusError):
                        last_error = type(error).__name__
                    if attempt < args.max_attempts:
                        await asyncio.sleep(min(30, 2 ** (attempt - 1)) + random.random())
            if not succeeded:
                await append(args.output_dir / "errors.jsonl", {
                    "request_id": record["request_id"], "patent_id": record["patent_id"],
                    "error": last_error, "attempts": attempt, "created_at": utc_now(),
                })
            queue.task_done()

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        workers = [asyncio.create_task(worker(client)) for _ in range(args.concurrency)]
        selected = []
        for record in iter_inputs(args.input):
            if record["request_id"] in completed:
                counts["skipped_completed"] += 1
                continue
            selected.append({"request_id": record["request_id"], "patent_id": record["patent_id"]})
            await queue.put(record)
            if args.limit and len(selected) >= args.limit:
                break
        for _ in workers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*workers)
    success = {record["request_id"] for record in iter_jsonl(parsed_path)} if parsed_path.exists() else set()
    failures = [record for record in selected if record["request_id"] not in success]
    counts["failed"] = len(failures)
    (args.output_dir / "failed.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in failures), encoding="utf-8"
    )
    (args.output_dir / "status.json").write_text(json.dumps({**counts, "finished_at": utc_now()}, indent=2), encoding="utf-8")
    print(json.dumps(counts))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=work_dir("extraction") / "input.jsonl")
    parser.add_argument("--output-dir", type=Path, default=work_dir("extraction"))
    parser.add_argument("--model", default=setting("extraction", "model", ""))
    parser.add_argument("--endpoint", default=setting("extraction", "endpoint", ""))
    parser.add_argument("--api-key-env", default=setting("extraction", "api_key_env", "TFSC_LLM_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=setting("extraction", "concurrency", 4))
    parser.add_argument("--temperature", type=float, default=setting("extraction", "temperature", 0.0))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=0, help="0 processes all remaining input records")
    parser.add_argument("--disable-thinking", action="store_true", help="Send provider-specific thinking=disabled option")
    parser.add_argument("--send", action="store_true", help="Make API requests; omitting this flag only validates the batch")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error("Concurrency/attempts must be positive and limit nonnegative")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    manifest = manifest_for(args, prompt)
    count = sum(1 for _ in iter_inputs(args.input))
    if not args.send:
        print(json.dumps({"input_records": count, "network_requests": 0, "batch": manifest}, indent=2))
        return
    asyncio.run(run(args, prompt, manifest))


if __name__ == "__main__":
    main()
