#!/usr/bin/env python3
"""Fetch papers into papers/ as paper.md (plus the original PDF).

Two tiers, per URL:

  1. arXiv papers. Convert https://arxiv.org/html/{id} through the local
     crw API (127.0.0.1:3000). This text is derived from the LaTeX source
     and is preferred. The crw banner above the paper title is trimmed.
  2. Fallback for everything else. Download the PDF, keep it as
     {paper_id}.pdf, and convert it with pymupdf4llm.

A folder is complete when it holds both paper.md and the PDF. Missing
PDFs are backfilled best-effort on later runs, so an arXiv paper fetched
as HTML still ends up with its archival PDF. Papers that cannot be
fetched automatically are reported as MANUAL. Drop the PDF into the
created folder by hand and run the script again. It will find the PDF
and convert it.

Usage:
  uv run python3 scripts/fetch-paper.py URL [URL ...]
"""

import contextlib
import http.client
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIT_DIR = REPO_ROOT / "papers"
CRW_API = "http://127.0.0.1:3000/v1/scrape"
UA = {"User-Agent": "undergraduate-research literature fetcher"}
MIN_PAPER_CHARS = 5000


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)
LIT_DIR.mkdir(parents=True, exist_ok=True)
for url in sys.argv[1:]:
    m = re.match(r"^https?://arxiv\.org/(?:abs|pdf)/([^/?#]+)", url)
    if m:
        paper_id = f"arxiv-{m.group(1)}"
    else:
        m = re.match(r"^https?://aclanthology\.org/([^/?#]+?)(?:\.pdf)?/?$", url)
        if m:
            paper_id = f"acl-{m.group(1)}"
        else:
            m = re.match(r"^https?://(?:www\.)?jmlr\.org/papers/(.+?)\.html$", url)
            if m:
                paper_id = f"jmlr-{m.group(1)}"
            else:
                m = re.match(
                    r"^https?://proceedings\.mlr\.press/([^/?#]+)/([^/?#]+?)\.html$",
                    url,
                )
                if m:
                    paper_id = f"pmlr-{m.group(2)}"
                else:
                    m = re.search(
                        r"^https?://openreview\.net/(?:forum|pdf)\?(?:.*&)?id=([^&#]+)",
                        url,
                    )
                    if m:
                        paper_id = f"openreview-{m.group(1)}"
                    else:
                        m = re.match(r"^https?://(?:dx\.)?doi\.org/(.+?)/?$", url)
                        paper_id = f"doi-{m.group(1)}" if m else None
    if paper_id:
        paper_id = re.sub(r"[/\\:]", "-", paper_id)
    if not paper_id:
        print(f"SKIP    unrecognized URL {url}")
        continue

    folder = LIT_DIR / paper_id
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / f"{paper_id}.pdf"
    md_path = folder / "paper.md"
    missing_md = not md_path.exists()
    missing_pdf = not pdf_path.exists()
    if not missing_md and not missing_pdf:
        print(f"OK      {paper_id} folder already complete")
        continue

    # Tier 1. arXiv HTML, derived from the LaTeX source.
    if missing_md and paper_id.startswith("arxiv-"):
        payload = json.dumps(
            {
                "url": f"https://arxiv.org/html/{paper_id[len('arxiv-') :]}",
                "formats": ["markdown"],
            }
        ).encode()
        req = urllib.request.Request(
            CRW_API, data=payload, headers={"Content-Type": "application/json"}
        )
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.load(r).get("data") or {}
            md = body.get("markdown") or ""
            if (body.get("metadata") or {}).get("statusCode") == 200 and len(
                md
            ) >= MIN_PAPER_CHARS:
                i = md.find("\n# ")
                if i != -1:
                    md = md[i + 1 :]  # trim the arXiv banner above the title heading
                md_path.write_text(md)
                missing_md = False
                print(f"OK      {paper_id} converted from arXiv HTML")
        if missing_md:
            print(f"NOTE    {paper_id} no arXiv HTML, falling back to PDF")

    # PDF download, also runs as a backfill for HTML-fetched folders.
    if missing_pdf:
        try:
            candidate = None
            m = re.match(r"^https?://arxiv\.org/abs/([^/?#]+)", url)
            if m:
                candidate = f"https://arxiv.org/pdf/{m.group(1)}"
            m = re.match(r"^https?://aclanthology\.org/([^/?#]+?)(?:\.pdf)?/?$", url)
            if m:
                candidate = f"https://aclanthology.org/{m.group(1)}.pdf"
            m = re.match(r"^(https?://proceedings\.mlr\.press/[^?#]+?)\.html$", url)
            if m:
                candidate = f"{m.group(1)}.pdf"
            m = re.search(r"^https?://openreview\.net/forum\?(?:.*&)?id=([^&#]+)", url)
            if m:
                candidate = f"https://openreview.net/pdf?id={m.group(1)}"
            m = re.match(r"^https?://(?:www\.)?jmlr\.org/papers/v(\d+)/(.+?)\.html$", url)
            if m:
                base = (
                    f"https://www.jmlr.org/papers/volume{m.group(1)}/{m.group(2)}"
                )
                candidate = f"{base}/{m.group(2)}.pdf"

            data = None
            if candidate:
                data = fetch(candidate)
            else:
                data = fetch(url)
                if not data.startswith(b"%PDF"):
                    html = data.decode("utf-8", "ignore")
                    link = re.search(
                        r'<meta[^>]+name="citation_pdf_url"[^>]+content="([^"]+)"', html
                    )
                    link = link or re.search(r'href="([^"]+\.pdf)"', html)
                    data = fetch(link.group(1)) if link else None
            if not data or not data.startswith(b"%PDF"):
                print(
                    f"MANUAL  {paper_id} no automatic PDF found, drop the PDF into {folder}"
                )
                continue
            pdf_path.write_bytes(data)
            missing_pdf = False
            print(f"OK      {paper_id} downloaded {len(data) // 1024} KB")
        except (OSError, http.client.HTTPException) as e:
            print(
                f"MANUAL  {paper_id} download failed ({e}), drop the PDF into {folder}"
            )
            continue

    if missing_md:
        try:
            import pymupdf4llm
        except ImportError:
            print(f"SKIP    {paper_id} pymupdf4llm is missing, rerun with uv run")
            continue
        md_path.write_text(pymupdf4llm.to_markdown(str(pdf_path)))
        print(f"OK      {paper_id} converted to paper.md")
