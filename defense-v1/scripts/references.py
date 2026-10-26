# /// script
# requires-python = ">=3.12"
# ///
"""Build the reference list for the last slides from the works cited in the presentation.

Only the works cited by author and year on a slide or in script.md are listed. Works that are
only named, such as HumAID, LibMTL, Cohen's kappa, Dunnett's test, and the optimizers, are not.

Pandoc formats the entries with the paper's own bibliography and APA style file, so the
slide matches the manuscript. Run with `uv run scripts/references.py` from the defense-v1 folder.
"""

import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUTPUT = Path(__file__).parent.parent / "data" / "references.json"

# The works cited in the presentation, by their keys in bibliography.bib. Update this list when a
# citation is added to or removed from the slides or the script
PRESENTATION_KEYS = [
    "ali2026humaidner",
    "bundnis2025worldriskreport",
    "caoile2025urdaneta",
    "ermino2022urgency",
    "gama2026uniform",
    "kurin2022unitary",
    "pagasa2026cyclone",
    "seeberger2025generalizing",
    "somido2023actionable",
    "yute2021satellite",
    "zhu2025prefinetuning",
]


def cited_keys() -> list[str]:
    return sorted(PRESENTATION_KEYS)


def formatted_entries(keys: list[str]) -> list[str]:
    document = "---\nnocite: |\n  " + ", ".join(f"@{key}" for key in keys) + "\n---\n"
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as file:
        file.write(document)
    output = subprocess.run(
        ["pandoc", file.name, "--citeproc", "--to", "html",
         "--bibliography", str(ROOT / "bibliography.bib"), "--csl", str(ROOT / "apa.csl")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    Path(file.name).unlink()

    # Each entry is a csl-entry div. Its links are turned into plain text, since the slide is
    # read from a screen and not clicked
    entries = re.findall(r'<div id="ref-[^"]+" class="csl-entry"[^>]*>\s*(.*?)\s*</div>', output, re.S)
    entries = [re.sub(r"<a [^>]*>(.*?)</a>", r"\1", entry, flags=re.S) for entry in entries]
    return [re.sub(r"\s+", " ", entry).strip() for entry in entries]


def main() -> None:
    keys = cited_keys()
    entries = formatted_entries(keys)
    if len(entries) != len(keys):
        raise SystemExit(f"Pandoc formatted {len(entries)} entries for {len(keys)} cited works")
    OUTPUT.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(entries)} references to {OUTPUT}")
    # A plain text preview of the first entry, to check the formatting by eye
    print(html.unescape(re.sub(r"<[^>]+>", "", entries[0])))


if __name__ == "__main__":
    main()
