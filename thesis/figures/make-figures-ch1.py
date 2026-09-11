#!/usr/bin/env python3
"""Generate the vector PDF figures of chapter 1 with pymupdf.

Outputs, one PDF per figure, into this folder:

  ipo-diagram-1.pdf  overall study conceptual framework (IPO)
  ipo-diagram-2.pdf  software artifact conceptual framework (IPO)

Rerun after editing: mise run figures
"""

import math
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
BLACK = (0, 0, 0)
FILL = (0.955, 0.955, 0.955)
HEAD = "hebo"
BODY = "helv"


def width(text, size, font=BODY):
    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


def center_lines(page, rect, text, size, font=BODY, lead=1.25):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if width(trial, size, font) <= rect.width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    y = rect.y0 + size
    for ln in lines:
        x = rect.x0 + (rect.width - width(ln, size, font)) / 2
        page.insert_text(pymupdf.Point(x, y), ln, fontname=font, fontsize=size)
        y += size * lead
    return len(lines)


def box(page, rect, title, lines=None):
    page.draw_rect(rect, color=BLACK, fill=FILL, width=0.9)
    n_title = center_lines(page, pymupdf.Rect(rect.x0 + 3, rect.y0 + 4,
                                              rect.x1 - 3, rect.y1 - 2),
                           title, size=8.3, font=HEAD)
    if lines:
        body = pymupdf.Rect(rect.x0 + 3, rect.y0 + 4 + n_title * 8.3 * 1.25 + 1,
                            rect.x1 - 3, rect.y1 - 2)
        for item in lines:
            n = center_lines(page, body, item, size=7.6)
            body = pymupdf.Rect(body.x0, body.y0 + n * 7.6 * 1.25, body.x1, body.y1)


def arrow(page, p1, p2, width=0.9):
    page.draw_line(p1, p2, color=BLACK, width=width)
    ang = math.atan2(p2.y - p1.y, p2.x - p1.x)
    for da in (math.radians(152), math.radians(-152)):
        page.draw_line(p2, pymupdf.Point(p2.x + 5 * math.cos(ang + da),
                                         p2.y + 5 * math.sin(ang + da)),
                       color=BLACK, width=width)


def overall():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=180)
    stages = [
        ("INPUT", [
            "I1. Taglish disaster tweet", "corpus, four typhoon events",
            "I2. Four multilingual", "Transformer encoders",
            "I3. MTO configurations with", "STL and static baselines",
            "I4. Local FP16 workstation", "testbed",
        ]),
        ("PROCESS", [
            "P1. Normalization,", "tokenization, BIO", "alignment, LOEO splits",
            "P2. Multi-task fine-tuning", "with gradient diagnostics",
            "P3. Bootstrap intervals and", "Friedman-Nemenyi tests",
            "P4. FastAPI embedding with", "gazetteer resolution",
        ]),
        ("OUTPUT", [
            "O1. Hard-sharing multi-task", "model, O(1) inference",
            "O2. Benchmark tables across", "12 paired runs",
            "O3. Bootstrap and", "non-parametric results",
            "O4. Prototype disaster", "triage application",
        ]),
    ]
    margin, gap, w = 8, 20, 122
    for i, (title, lines) in enumerate(stages):
        x0 = margin + i * (w + gap)
        box(page, pymupdf.Rect(x0, 14, x0 + w, 170), title, lines)
        if i:
            arrow(page, pymupdf.Point(x0 - gap, 92), pymupdf.Point(x0, 92))
    doc.save(HERE / "ipo-diagram-1.pdf")


def artifact():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=212)
    replay = pymupdf.Rect(8, 22, 96, 62)
    manual = pymupdf.Rect(8, 92, 96, 132)
    service = pymupdf.Rect(118, 46, 216, 128)
    gazetteer = pymupdf.Rect(238, 46, 336, 128)
    database = pymupdf.Rect(238, 156, 336, 200)
    dashboard = pymupdf.Rect(124, 156, 222, 200)
    centers = pymupdf.Rect(8, 156, 108, 200)
    box(page, replay, "Replayed", ["event streams"])
    box(page, manual, "Manual", ["ingestion"])
    box(page, service, "Asynchronous FastAPI service",
        ["single-pass NER,", "intent, and urgency"])
    box(page, gazetteer, "Gazetteer resolution",
        ["Philippine toponyms,", "OOV fallback"])
    box(page, database, "Relational incident database",
        ["SQLite or PostgreSQL"])
    box(page, dashboard, "Interactive triage dashboard",
        ["New to Resolved", "lifecycle states"])
    box(page, centers, "Municipal command centers")
    arrow(page, pymupdf.Point(replay.x1, 42), pymupdf.Point(service.x0, 72))
    arrow(page, pymupdf.Point(manual.x1, 112), pymupdf.Point(service.x0, 102))
    arrow(page, pymupdf.Point(service.x1, 87), pymupdf.Point(gazetteer.x0, 87))
    arrow(page, pymupdf.Point((gazetteer.x0 + gazetteer.x1) / 2, gazetteer.y1),
          pymupdf.Point((database.x0 + database.x1) / 2, database.y0))
    arrow(page, pymupdf.Point(database.x0, 178), pymupdf.Point(dashboard.x1, 178))
    arrow(page, pymupdf.Point(dashboard.x0, 178), pymupdf.Point(centers.x1, 178))
    doc.save(HERE / "ipo-diagram-2.pdf")


if __name__ == "__main__":
    overall()
    artifact()
