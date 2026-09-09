"""Convert hardcoded author-date citations in the chapter draft.qmd
to Quarto citekeys that resolve against references.json.

Run after scripts/build-references.py whenever references.json changes.
Idempotent: already-converted @keys are left untouched.

Citations whose papers have no record in papers/ are left as
literal text and reported, so they stay visible until their Zotero entries
exist.

Usage: python scripts/convert-citations.py
"""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QMD = ROOT / "chapters" / "chapter-2" / "draft.qmd"
REFS = ROOT / "references.json"

# Author + year combinations where several records share a surname and year,
# or where the draft's year label differs from the record year.
LABELED = {
    ("alam", "2021", "a"): "alam2021crisisbench",
    ("alam", "2021", "b"): "alam2021humaid",
    ("liu", "2019", "a"): "liu2019endtoend",
    ("liu", "2019", "b"): "liu2019multitask",
    ("liu", "2021", "a"): "liu2021conflictaverse",
    ("liu", "2021", "b"): "liu2021impartial",
    ("chen", "2024", ""): "chen2024multitask",
    ("miranda", "2023", ""): "miranda2023developing",
    # Both Sheth records carry year 2025 in references.json (survey year fix
    # is deferred), so narrative citations need explicit disambiguation.
    ("sheth", "2025", ""): "sheth2025comilingua",
    ("sheth", "2026", ""): "sheth2025beyond",  # record year is 2025, see README
}

# Papers cited in the draft that have no record yet: never converted.
SKIP_SURNAMES = {"loshchilov", "warner", "ali", "nag"}


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def build_index() -> dict:
    index = {}
    for entry in json.loads(REFS.read_text()):
        authors = entry.get("author") or []
        if not authors or "issued" not in entry:
            continue
        surname = fold(authors[0].get("family", ""))
        year = str(entry["issued"]["date-parts"][0][0])
        index.setdefault((surname, year), []).append(entry["id"])
    return index


INDEX = build_index()


def resolve(surname: str, year: str, label: str) -> str | None:
    folded = fold(surname)
    if not folded or folded in SKIP_SURNAMES:
        return None
    if (folded, year, label) in LABELED:
        return LABELED[(folded, year, label)]
    exact = INDEX.get((folded, year), [])
    if len(exact) == 1:
        return exact[0]
    # Draft may use a short form of a compound surname (Kayi / Sarioglu Kayi).
    partial = [ids for (sn, yr), ids in INDEX.items()
               if yr == year and len(folded) >= 4 and (sn.startswith(folded) or folded in sn)]
    if len(partial) == 1:
        return partial[0][0]
    return None


def convert_narrative(text: str, report: dict) -> str:
    pattern = re.compile(
        r"([A-ZÀ-Ž][A-Za-zà-ž'’\-]+)"
        r"(?:\s+et al\.|\s+and\s+[A-ZÀ-Ž][A-Za-zà-ž'’\-]+|\s*&\s*[A-ZÀ-Ž][A-Za-zà-ž'’\-]+)?"
        r"\s*\(((?:19|20)\d{2})([ab]?)\)"
    )

    def replace(match: re.Match) -> str:
        key = resolve(match.group(1), match.group(2), match.group(3))
        if key is None:
            return match.group(0)
        report[key] = report.get(key, 0) + 1
        return f"@{key}"

    return pattern.sub(replace, text)


def convert_parenthetical(text: str, report: dict) -> str:
    year_pattern = re.compile(r"\b((?:19|20)\d{2})([ab]?)\b")
    surname_pattern = re.compile(r"[A-ZÀ-Ž][A-Za-zà-ž'’\-]+")

    def replace(match: re.Match) -> str:
        group = match.group(1)
        if not year_pattern.search(group):
            return match.group(0)
        keys = []
        for item in group.split(";"):
            item = item.strip()
            surname_match = surname_pattern.search(item)
            surname = surname_match.group(0) if surname_match else ""
            years = year_pattern.findall(item)
            if not years:
                return match.group(0)
            for year, label in years:
                key = resolve(surname, year, label)
                if key is None:
                    return match.group(0)
                keys.append(key)
        for key in keys:
            report[key] = report.get(key, 0) + 1
        return "[" + "; ".join(f"@{k}" for k in keys) + "]"

    return re.sub(r"\(([^()]+)\)", replace, text)


def main() -> None:
    text = QMD.read_text()
    report: dict = {}
    text = convert_narrative(text, report)
    text = convert_parenthetical(text, report)
    QMD.write_text(text)

    total = sum(report.values())
    print(f"converted {total} citation instances, {len(report)} distinct keys\n")
    for key in sorted(report):
        print(f"  @{key}")
    leftovers = []
    for pattern in re.finditer(
        r"[A-ZÀ-Ž][A-Za-zà-ž'’\-]+(?:\s+et al\.)?[^.()]{0,30}\((?:19|20)\d{2}[ab]?\)", text
    ):
        leftovers.append(pattern.group(0))
    if leftovers:
        print("\nLEFT AS LITERAL (no record, needs Zotero entry):")
        for item in sorted(set(leftovers)):
            print(f"  {item}")


if __name__ == "__main__":
    main()
