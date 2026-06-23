#!/opt/miniconda3/bin/python3
"""Generate 10,000 Q/A correction pairs via Ollama (gemma4:latest).

Rotates evenly across:
  - 14 domains
  - 3 difficulty levels
  - 5 question types

Uses ThreadPoolExecutor (4 workers) with batched prompts (5 entries/call).

Output: ~/Desktop/Testguard/DatasetsClaude/dataset2_qa/dataset2.jsonl
Errors:  ~/Desktop/Testguard/DatasetsClaude/dataset2_qa/errors_log.txt
Final:   dataset2.parquet (in same dir)
"""

import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import pyarrow as pa
import pyarrow.parquet as pq

random.seed(42)

# ── config ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:latest"

OUT_DIR = os.path.expanduser("~/Desktop/Testguard/DatasetsClaude/dataset2_qa")
JSONL_PATH = os.path.join(OUT_DIR, "dataset2.jsonl")
ERROR_LOG = os.path.join(OUT_DIR, "errors_log.txt")
PARQUET_PATH = os.path.join(OUT_DIR, "dataset2.parquet")

TOTAL = 10_000
BATCH_SIZE = 5  # entries per Ollama call
NUM_WORKERS = 4

DOMAINS = [
    "Physics", "Biology", "Mathematics", "Chemistry",
    "Literature", "History", "Philosophy", "Geography",
    "Economics", "Computer Science", "Law", "Psychology",
    "Sociology", "General Culture",
]

DIFFICULTIES = ["beginner", "intermediate", "advanced"]
QUESTION_TYPES = ["explanation", "definition", "comparison", "application", "analysis"]

# 14 * 3 * 5 = 210 combinations → ~47-48 each for 10,000 total
_COMBOS = [(d, f, t) for d in DOMAINS for f in DIFFICULTIES for t in QUESTION_TYPES]

# Thread-safe JSONL writer
_jsonl_lock = threading.Lock()


def build_batch_prompt(batch_entries: list[dict]) -> str:
    """Build a prompt that asks for a JSON array of 5 Q/A objects."""
    lines = []
    for item in batch_entries:
        lines.append(
            f"- id_prefix=\"{item['domain'][:4].lower()}_{item['difficulty']}_{item['question_type']}\", "
            f"domain={item['domain']}, difficulty={item['difficulty']}, "
            f"question_type={item['question_type']}"
        )
    specs = "\n".join(lines)

    return f"""You are an expert educator creating high-quality question-answer correction pairs for training an AI tutor.

Generate exactly {BATCH_SIZE} JSON objects as a JSON array. Each object must have these exact keys:
- id: a unique string using the provided id_prefix (just append _XXXXX)
- domain: the domain string given
- difficulty: the difficulty string given
- question_type: the question_type string given
- question: a clear, well-formulated question at the specified difficulty and type for the specified domain
- reference_answer: the correct answer (exactly 2-4 sentences, concise)
- corrected_answer: short constructive tutor feedback (2-4 sentences) for a student who was NEARLY correct but made ONE conceptual or factual mistake

Use these specifications for each entry (exactly {BATCH_SIZE} entries in the array):
{specs}

Rules:
1. Return ONLY the raw JSON array — no markdown fences, no commentary, no extra text.
2. reference_answer must be 2-4 sentences max — concise and precise.
3. corrected_answer must sound like constructive feedback, not a full rewrite.
4. Make the student error realistic and common for the given difficulty and domain.
5. Questions must test genuine understanding, not trivia.
6. Vary the sub-topics within each domain — do not repeat the same sub-topic across calls.
7. Output must be valid JSON parseable by json.loads().

JSON array:"""


def call_ollama(prompt: str, retries: int = 5) -> str | None:
    """Call Ollama /api/chat with the given prompt. Returns raw text or None."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": 0.92,
            "num_predict": 4096,
        },
        "stream": False,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()

            # Strip markdown fences if the model ignored instructions
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0] if "```" in text else text
                text = text.strip()
            return text
        except requests.exceptions.RequestException as e:
            wait = random.uniform(2, 4) * (attempt + 1)
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                return None


def parse_and_validate(obj: dict) -> dict | None:
    """Validate a single entry dict — all required fields present and non-empty."""
    required = ["id", "domain", "difficulty", "question_type", "question",
                 "reference_answer", "corrected_answer"]
    if not all(k in obj and obj[k] for k in required):
        return None
    for k in required:
        if not isinstance(obj[k], str) or not obj[k].strip():
            return None
    return obj


def parse_batch_response(raw: str) -> list[dict]:
    """Parse a JSON array response. Returns list of valid entries. Skips & logs on error."""
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    valid = []
    for item in items:
        parsed = parse_and_validate(item)
        if parsed is not None:
            valid.append(parsed)
    return valid


def log_error(entry: dict):
    os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def append_jsonl(obj: dict):
    """Thread-safe append to JSONL."""
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
    with _jsonl_lock:
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_batch(batch_entries: list[dict], batch_idx: int) -> tuple[int, int]:
    """Process a batch of 5 entries: call Ollama, parse array, write valid ones.
    Returns (good_count, error_count)."""
    prompt = build_batch_prompt(batch_entries)
    good = 0
    errors = 0

    for attempt in range(7):
        raw = call_ollama(prompt)
        if raw is None:
            continue

        items = parse_batch_response(raw)
        if items:
            for obj in items:
                # Assign real IDs
                domain = obj["domain"]
                difficulty = obj["difficulty"]
                qtype = obj["question_type"]
                obj["id"] = f"{domain[:4].lower()}_{difficulty}_{qtype}_{batch_idx:05d}"
                append_jsonl(obj)
                good += 1
            return good, errors
        # Parse failed — retry

    # All retries exhausted — log each slot as an error
    for entry in batch_entries:
        log_error({
            "seq": batch_idx, "domain": entry["domain"],
            "difficulty": entry["difficulty"],
            "question_type": entry["question_type"],
            "error": "All 7 retries exhausted for batch"
        })
        errors += 1
    return good, errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {TOTAL} entries via Ollama ({MODEL})")
    print(f"  Domains: {len(DOMAINS)}")
    print(f"  Difficulties: {len(DIFFICULTIES)}")
    print(f"  Question types: {len(QUESTION_TYPES)}")
    print(f"  Combinations: {len(_COMBOS)}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Batch size: {BATCH_SIZE} entries/call")
    print(f"  Output: {JSONL_PATH}")
    print()

    # Build schedule: repeat the combo list to cover TOTAL
    schedule = (_COMBOS * ((TOTAL // len(_COMBOS)) + 1))[:TOTAL]

    # Split schedule into batches of BATCH_SIZE
    batches = []
    for i in range(0, len(schedule), BATCH_SIZE):
        chunk = schedule[i : i + BATCH_SIZE]
        batch_entries = [
            {"domain": d, "difficulty": f, "question_type": t}
            for d, f, t in chunk
        ]
        batches.append((batch_entries, (i // BATCH_SIZE) + 1))

    total_batches = len(batches)
    good = 0
    errors = 0
    start = time.time()
    completed_batches = 0

    # Track progress counter across batches
    progress_lock = threading.Lock()
    last_progress = [0]

    def on_batch_done(batch_good: int, batch_errors: int):
        nonlocal good, errors
        with progress_lock:
            good += batch_good
            errors += batch_errors
            completed = completed_batches * BATCH_SIZE + batch_good + batch_errors
            # Actually we track differently since we don't know exact from futures
            # We'll track via a shared counter

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all batches
        future_map = {}
        for batch_entries, batch_idx in batches:
            future = executor.submit(process_batch, batch_entries, batch_idx)
            future_map[future] = batch_idx

        # Collect results, print progress every 100 entries
        processed = 0
        for future in as_completed(future_map):
            batch_good, batch_errors = future.result()
            good += batch_good
            errors += batch_errors
            processed += BATCH_SIZE

            if processed % 100 == 0 or processed >= TOTAL:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (TOTAL - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed}/{TOTAL}] good={good} errors={errors} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s"
                )

    total_time = time.time() - start
    print()
    print(f"Done in {total_time:.1f}s")
    print(f"  Valid: {good}")
    print(f"  Errors: {errors}")

    # ── Convert to Parquet ──
    print(f"\nConverting to Parquet...")
    entries = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    table = pa.Table.from_pylist(entries)
    pq.write_table(table, PARQUET_PATH)
    print(f"  Written: {PARQUET_PATH}")
    print(f"  Rows: {len(entries)}")

    # Summary
    summary = {
        "status": "ok" if errors == 0 else "partial",
        "total": TOTAL,
        "valid": good,
        "errors": errors,
        "parquet_path": PARQUET_PATH,
        "jsonl_path": JSONL_PATH,
        "error_log": ERROR_LOG,
        "duration_seconds": round(total_time, 1),
    }
    print(f"\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
