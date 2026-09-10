#!/usr/bin/env python3
"""Generate the vector PDF figures of chapter 3 with pymupdf.

Outputs, one PDF per figure, into this folder:

  fig-3-1-pipeline.pdf    end-to-end research pipeline
  fig-3-2-model.pdf       model layer flow under hard parameter sharing
  fig-3-3-dataflow.pdf    prototype service data flow
  fig-3-4-usecases.pdf    UML use case diagram of the prototype

Rerun after editing: uv run python3 chapters/chapter-3/figures/make-figures.py
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


def pipeline():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=128)
    specs = [
        ("Baseline Analysis",
         ["Corpus assembly", "STL baselines", "EW fine-tuning and", "encoder selection"]),
        ("Design and Artifact Creation",
         ["Hard-sharing trainer", "MTO configurations", "Gradient hooks", "Prototype service"]),
        ("Empirical Benchmarking",
         ["120 runs", "MLflow records", "Gradient diagnostics", "Efficiency timings"]),
        ("Comparative Validation",
         ["Bootstrap intervals", "Friedman-Nemenyi", "Profile analysis", "Transfer correlation"]),
    ]
    margin, gap, w, h = 8, 16, 93, 96
    y0 = 22
    put_title = pymupdf.Rect(8, 4, 424, 18)
    center_lines(page, put_title, "DSRM phases of the study", size=8, font=HEAD)
    rects = []
    for i, (title, lines) in enumerate(specs):
        r = pymupdf.Rect(margin + i * (w + gap), y0, margin + i * (w + gap) + w, y0 + h)
        box(page, r, title, lines)
        rects.append(r)
        if i:
            arrow(page, pymupdf.Point(rects[i - 1].x1, y0 + h / 2),
                  pymupdf.Point(r.x0, y0 + h / 2))
    doc.save(HERE / "fig-3-1-pipeline.pdf")


def model():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=176)
    box(page, pymupdf.Rect(8, 62, 76, 114), "Tweet text")
    box(page, pymupdf.Rect(94, 62, 156, 114), "Tokenizer", ["subword pieces"])
    enc = pymupdf.Rect(174, 30, 252, 146)
    box(page, enc, "Shared multilingual encoder")
    center_lines(page, pymupdf.Rect(174, 150, 252, 168), "shared parameters", size=7.2)
    heads = [
        (pymupdf.Rect(288, 16, 424, 56), "Token head", ["BIO labels for", "LOC and INFRA"]),
        (pymupdf.Rect(288, 60, 424, 100), "Intent head", ["merged triage classes"]),
        (pymupdf.Rect(288, 104, 424, 144), "Urgency head", ["four-level scale,", "binary conversion"]),
    ]
    for r, title, lines in heads:
        box(page, r, title, lines)
        arrow(page, pymupdf.Point(enc.x1, (enc.y0 + enc.y1) / 2),
              pymupdf.Point(r.x0, (r.y0 + r.y1) / 2))
    arrow(page, pymupdf.Point(76, 88), pymupdf.Point(94, 88))
    arrow(page, pymupdf.Point(156, 88), pymupdf.Point(174, 88))
    doc.save(HERE / "fig-3-2-model.pdf")


def dataflow():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=150)
    box(page, pymupdf.Rect(8, 22, 84, 58), "Replayed", ["event streams"])
    box(page, pymupdf.Rect(8, 82, 84, 118), "Manual entry")
    svc = pymupdf.Rect(98, 38, 186, 112)
    box(page, svc, "FastAPI service",
        ["single-pass inference:", "entities, intent,", "urgency"])
    gaz = pymupdf.Rect(200, 44, 272, 106)
    box(page, gaz, "PSGC gazetteer", ["unmapped-span", "fallback"])
    db = pymupdf.Rect(286, 44, 344, 106)
    box(page, db, "PostgreSQL", ["incident store"])
    dsh = pymupdf.Rect(358, 30, 424, 120)
    box(page, dsh, "Dashboard",
        ["New,", "Acknowledged,", "Responding,", "Resolved"])
    for r in (pymupdf.Rect(8, 22, 84, 58), pymupdf.Rect(8, 82, 84, 118)):
        arrow(page, pymupdf.Point(r.x1, (r.y0 + r.y1) / 2),
              pymupdf.Point(svc.x0, (svc.y0 + svc.y1) / 2))
    arrow(page, pymupdf.Point(svc.x1, 75), pymupdf.Point(gaz.x0, 75))
    arrow(page, pymupdf.Point(gaz.x1, 75), pymupdf.Point(db.x0, 75))
    arrow(page, pymupdf.Point(db.x1, 75), pymupdf.Point(dsh.x0, 75))
    doc.save(HERE / "fig-3-3-dataflow.pdf")


def usecases():
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=240)
    boundary = pymupdf.Rect(118, 8, 424, 232)
    page.draw_rect(boundary, color=BLACK, width=0.9)
    center_lines(page, pymupdf.Rect(118, 12, 424, 26),
                 "Prototype triage service", size=8.5, font=HEAD)
    cx, cy = 62, 108
    page.draw_circle(pymupdf.Point(cx, cy - 34), 9, color=BLACK, width=0.9)
    page.draw_line(pymupdf.Point(cx, cy - 25), pymupdf.Point(cx, cy + 4), color=BLACK, width=0.9)
    page.draw_line(pymupdf.Point(cx - 13, cy - 14), pymupdf.Point(cx + 13, cy - 14), color=BLACK, width=0.9)
    page.draw_line(pymupdf.Point(cx, cy + 4), pymupdf.Point(cx - 11, cy + 22), color=BLACK, width=0.9)
    page.draw_line(pymupdf.Point(cx, cy + 4), pymupdf.Point(cx + 11, cy + 22), color=BLACK, width=0.9)
    center_lines(page, pymupdf.Rect(10, cy + 26, 114, cy + 56),
                 "Municipal command operator", size=7.6)
    cases = [
        ("Ingest replayed messages", 44),
        ("Monitor incident dashboard", 76),
        ("Acknowledge incident", 108),
        ("Mark incident responding", 140),
        ("Mark incident resolved", 172),
        ("View mapped locations", 204),
    ]
    for label, y in cases:
        r = pymupdf.Rect(140, y - 12, 402, y + 12)
        page.draw_oval(r, color=BLACK, width=0.9)
        center_lines(page, pymupdf.Rect(r.x0 + 8, y - 6, r.x1 - 8, y + 9),
                     label, size=7.8)
        arrow(page, pymupdf.Point(112, cy), pymupdf.Point(140, y), width=0.7)
    doc.save(HERE / "fig-3-4-usecases.pdf")


if __name__ == "__main__":
    pipeline()
    model()
    dataflow()
    usecases()
    print("figures written to", HERE)
