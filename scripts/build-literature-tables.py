"""Build the literature tables in research/ from their CSV sources.

Each *-literature-table.md renders the matching *-literature.csv with one
row per paper. Cell text is normalized to the thesis writing rules in
AGENTS.md: semicolons become commas, en and em dashes become hyphens.

Usage: python scripts/build-literature-tables.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "literature"
LIT_DIR = ROOT / "papers"


def cell(text: str) -> str:
    """Escape raw text for one markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ")


def plain(text: str) -> str:
    """Normalize punctuation to the thesis writing rules."""
    return (text or "").replace("\u2013", "-").replace("\u2014", "-").replace(";", ",")


def field(name: str):
    return lambda row: plain(cell(row[name]))


def paper_link(row: dict) -> str:
    paper_id = row["paper_id"]
    return f"[{paper_id}](file://{LIT_DIR / paper_id})"


def venue_cell(row: dict) -> str:
    return f"{plain(cell(row['venue']))} ({plain(cell(row['venue_type']))})"


def abstract_cell(row: dict) -> str:
    return (
        "<details><summary>Expand Abstract</summary><br>"
        f"{plain(cell(row['abstract']))}</details>"
    )


TABLES = [
    {
        "csv": TABLES_DIR / "primary-literature.csv",
        "out": TABLES_DIR / "primary-literature-table.md",
        "title": "# Primary Literature Table (Manuscript Citations)",
        "columns": [
            ("Citation", field("citation")),
            ("Title", field("title")),
            ("Venue (Type)", venue_cell),
            ("Thematic Pillar", field("pillar")),
            ("Method / Model", field("algorithm_or_model")),
            ("Specific Role in Thesis", field("role_in_thesis")),
            ("Abstract", abstract_cell),
        ],
    },
    {
        "csv": TABLES_DIR / "secondary-literature.csv",
        "out": TABLES_DIR / "secondary-literature-table.md",
        "title": "# Secondary Literature Table (Background Evidence Repository)",
        "columns": [
            ("Citation", field("citation")),
            ("Title", field("title")),
            ("Venue (Type)", venue_cell),
            ("Themes", field("themes")),
            ("Method / Model", field("algorithm_or_model")),
            ("Core Problem", field("problem")),
            ("Proposed Solution", field("proposed_solution")),
            ("Primary Claim", field("claim")),
            ("Abstract", abstract_cell),
        ],
    },
]


def render(spec: dict, rows: list[dict]) -> str:
    header = "| No. | " + " | ".join(name for name, _ in spec["columns"]) + " |"
    separator = "|" + "---|" * (len(spec["columns"]) + 1)
    lines = [spec["title"], "", header, separator]
    for i, row in enumerate(rows, 1):
        cells = [str(i)] + [fn(row) for _, fn in spec["columns"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    for spec in TABLES:
        with spec["csv"].open(newline="") as f:
            rows = list(csv.DictReader(f))
        spec["out"].write_text(render(spec, rows))
        print(f"wrote {spec['out'].relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
