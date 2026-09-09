#!/usr/bin/env python3
"""Validate literature extraction records against the skill's bundled schema.

Each paper lives in papers/{paper_id}/ with record.json,
paper.md, and the original PDF. Per record:

  1. JSON Schema validation (draft 2020-12).
  2. Folder name equals the record paper_id.
  3. Limitation/quote parity, one quote per limitation.
  4. Warnings only. Each evidence quote should appear in paper.md.

Usage:
  uv run python3 validate.py

Exits non-zero if any record fails. Warnings do not affect the exit code.
"""

import json
import re
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = Path(__file__).resolve().parent / "extraction-schema.json"
LIT_DIR = REPO_ROOT / "papers"


def normalize(text):
    """Make text comparable across PDF conversion artifacts."""
    text = text.replace("\xad", "").replace("\u200b", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"[*_#`$]", "", text)
    text = re.sub(r"\\+", "", text)
    for curly, plain in (
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2013", "-"),
        ("\u2014", "-"),
    ):
        text = text.replace(curly, plain)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.:;!?\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = text.replace("-", "")
    return text.strip()


if not LIT_DIR.is_dir():
    print(f"no literature directory at {LIT_DIR}")
    sys.exit(0)
paths = sorted(LIT_DIR.glob("*/record.json"))
if not paths:
    print(f"no records found under {LIT_DIR}")
    sys.exit(0)

schema = json.loads(SCHEMA_PATH.read_text())
validator = jsonschema.Draft202012Validator(schema)

failed = 0
warned = 0
total = 0
for path in paths:
    total += 1
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL {path.parent.name}: record.json is not valid JSON ({e})")
        failed += 1
        continue

    errors = [
        f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(record)
    ]

    paper_id = record.get("paper_id")
    if paper_id and paper_id != path.parent.name:
        errors.append(
            f"folder: paper_id {paper_id} does not match folder name {path.parent.name}"
        )

    limitations = record.get("content", {}).get("author_stated_limitations", [])
    quotes = record.get("evidence", {}).get("author_stated_limitation_quotes", [])
    if len(limitations) != len(quotes):
        errors.append(
            f"parity: {len(limitations)} limitations but {len(quotes)} quotes "
            f"(they must match one to one)"
        )

    if errors:
        failed += 1
        print(f"FAIL {path.parent.name}")
        for err in errors:
            print(f"     {err}")
    else:
        print(f"PASS {path.parent.name}")

    paper = path.parent / "paper.md"
    if not paper.is_file():
        continue
    text = normalize(paper.read_text())
    evidence = record.get("evidence", {})
    checks = [
        ("problem_quote", evidence.get("problem_quote")),
        ("proposed_solution_quote", evidence.get("proposed_solution_quote")),
        ("claim_quote", evidence.get("claim_quote")),
    ]
    checks += [
        ("author_stated_limitation_quotes", q)
        for q in evidence.get("author_stated_limitation_quotes", [])
    ]
    for label, quote in checks:
        if quote and normalize(quote) not in text:
            warned += 1
            print(f"     warning: {label} not found in paper.md")

print(f"\n{total - failed}/{total} records valid, {warned} warnings")
sys.exit(1 if failed else 0)
