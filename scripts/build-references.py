"""Build references.bib (BibTeX) from papers/*/record.json.

Quarto's Typst engine reads the bibliography natively, and its supported
formats are BibTeX and Hayagriva YAML, so the render pipeline uses .bib.
Better BibTeX also exports BibTeX, which keeps the Zotero loop simple.

A CSL JSON copy (references.json) is written as well as a key index
and a Zotero-importable mirror of the same entries.

Citekeys follow the Better BibTeX style `auth.lower + year + shorttitle (1)`
so that keys generated here usually match a Zotero re-export with the same
key formula. Collisions get a letter suffix.

Usage: python scripts/build-references.py
"""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIT_DIR = ROOT / "papers"
OUT_BIB = ROOT / "references.bib"
OUT_JSON = ROOT / "references.json"

STOPWORDS = {
    "a", "an", "the", "on", "of", "for", "to", "in", "and", "do", "does",
    "towards", "toward", "is", "are", "as", "at", "by", "with", "from",
}


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def title_word(title: str) -> str:
    for word in title.split():
        cleaned = re.sub(r"[^A-Za-z0-9]", "", word)
        if cleaned and cleaned.lower() not in STOPWORDS:
            return cleaned.lower()
    return "title"


def make_key(surname: str, year: int, title: str, used: set) -> str:
    surname = re.sub(r"[^A-Za-z0-9]", "", ascii_fold(surname)).lower()
    base = f"{surname}{year}{title_word(title)}"
    key, suffix = base, ""
    while key in used:
        key = base + suffix
        suffix = chr(ord("a") + (ord(suffix) + 1 - ord("a")) % 26) if suffix else "a"
    used.add(key)
    return key


def csl_type(venue_type: str) -> str:
    venue_type = (venue_type or "").lower()
    if "webpage" in venue_type:
        return "webpage"
    if "legislation" in venue_type:
        return "bill"
    if "preprint" in venue_type:
        return "preprint"
    if "journal" in venue_type:
        return "article-journal"
    if "conference" in venue_type or "workshop" in venue_type:
        return "paper-conference"
    return "article"


def clean_venue(venue: str) -> str:
    return re.sub(r"\s*\((?:conference|journal|workshop|preprint)\)\s*$", "", venue or "").strip()


def to_csl(record: dict, key: str) -> dict:
    bib = record.get("bibliographic", {})
    year = bib.get("year")
    entry = {
        "id": key,
        "type": csl_type(bib.get("venue_type", "")),
        "title": bib.get("title", "").strip(),
        "author": [
            {"family": a.get("last_name", ""), "given": a.get("first_name", "")}
            for a in bib.get("authors", [])
            if a.get("last_name")
        ],
        "issued": {"date-parts": [[int(year)]]} if year else None,
    }
    venue = clean_venue(bib.get("venue", ""))
    if venue:
        entry["container-title"] = venue
    ident = bib.get("doi_or_url", "")
    if ident.startswith("10."):
        entry["DOI"] = ident
    elif ident.startswith("http"):
        entry["URL"] = ident
    return {k: v for k, v in entry.items() if v not in (None, "", [])}


def bibtex_type(venue_type: str) -> str:
    venue_type = (venue_type or "").lower()
    if "journal" in venue_type:
        return "article"
    if "conference" in venue_type or "workshop" in venue_type:
        return "inproceedings"
    return "misc"


# Words that citeproc's APA sentence-casing must not lowercase.
PROPER_NOUNS = {
    "algerian", "arabic", "philippine", "philippines", "filipino", "tagalog",
    "taglish", "english", "hindi", "spanish", "sinhala", "odia", "cebuano",
    "twitter", "southeast", "asian", "austronesian", "typhoon", "haiyan",
    "yolanda", "ulysses", "vamco", "nash", "manifold", "mixup", "fleiss",
    "cohen", "krippendorff", "bayesian", "bayes",
}


def bibtex_escape(value: str) -> str:
    return re.sub(r"([&%#_])", r"\\\1", value)


def protect_proper_nouns(title: str) -> str:
    def brace(text: str) -> str:
        return "{" + text + "}"
    return re.sub(
        r"[A-Za-z]+(?:-[A-Za-z]+)*",
        lambda m: brace(m.group(0))
        if any(part.lower() in PROPER_NOUNS for part in m.group(0).split("-"))
        else m.group(0),
        title,
    )


def to_bibtex(record: dict, key: str) -> str:
    bib = record.get("bibliographic", {})
    author_list = [a for a in bib.get("authors", []) if a.get("last_name")]
    if len(author_list) == 1 and not author_list[0].get("first_name"):
        # Group author (agency, government): brace the whole name so BibTeX
        # reads one literal author instead of splitting on commas and "and".
        authors = "{" + author_list[0]["last_name"] + "}"
    else:
        authors = " and ".join(
            f"{a.get('last_name', '')}, {a.get('first_name', '')}".strip(", ")
            for a in author_list
        )
    venue = clean_venue(bib.get("venue", ""))
    entry_type = bibtex_type(bib.get("venue_type", ""))
    if entry_type == "article":
        venue_field = "journal"
    elif entry_type == "inproceedings":
        venue_field = "booktitle"
    else:
        venue_field = "howpublished"
    fields = [
        ("author", authors),
        ("title", protect_proper_nouns(bibtex_escape(bib.get("title", "")))),
        (venue_field, venue),
        ("year", str(bib.get("year") or "n.d.")),
    ]
    ident = bib.get("doi_or_url", "")
    if ident.startswith("10."):
        fields.append(("doi", ident))
    elif ident.startswith("http"):
        fields.append(("url", ident))
    if entry_type == "misc" and "arxiv" in (
        (venue or "") + (bib.get("doi_or_url") or "")
    ).lower():
        fields.append(("note", "Preprint"))
    body = ",\n".join(f"  {name} = {{{bibtex_escape(value)}}}" for name, value in fields if value)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def main() -> None:
    entries, keys, used = [], [], set()
    for path in sorted(LIT_DIR.glob("*/record.json")):
        record = json.loads(path.read_text())
        bib = record.get("bibliographic", {})
        authors = bib.get("authors") or []
        surname = authors[0].get("last_name", "unknown") if authors else "unknown"
        year = int(bib.get("year", 0) or 0)
        # Non-paper records (webpages, legislation) carry an explicit citekey so
        # long group authors do not explode the generated key.
        key = record.get("citekey")
        if key:
            used.add(key)
        else:
            key = make_key(surname, year, bib.get("title", ""), used)
        entries.append(to_csl(record, key))
        keys.append(to_bibtex(record, key))

    OUT_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    OUT_BIB.write_text("\n\n".join(keys) + "\n")
    print(f"wrote {len(keys)} entries to {OUT_BIB.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    for entry in entries:
        authors = entry.get("author", [])
        surname = authors[0]["family"] if authors else "?"
        year = entry["issued"]["date-parts"][0][0] if "issued" in entry else "?"
        print(f"  {entry['id']:32s} {surname}, {year}")


if __name__ == "__main__":
    main()
