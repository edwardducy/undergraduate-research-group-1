# MinerU Extraction Pipeline Specification & Operations Guide

This document defines the technical architecture, operational workflows, parameter references, and resource management protocols for the PDF extraction pipeline in the thesis research repository.

---

## 1. Overview & Pipeline Purpose

The extraction pipeline converts heterogeneous, unstructured documents—including peer-reviewed research papers, Philippine disaster situational reports (DSWD DROMIC), audit bulletins (COA), and statutory acts—into machine-readable, structured representations for citation indexing, retrieval-augmented synthesis, and thesis writing.

Extraction is powered by **MinerU** using its `hybrid-engine` backend, which pairs layout analysis and optical structure detection with Vision-Language Models (VLMs) to extract:
* Semantic document hierarchy (headings, paragraphs, footnotes)
* Inline and block mathematical formulas rendered in LaTeX syntax
* Structured Markdown tables
* Bounding-box annotations for visual layout verification
* Cropped image assets of figures, charts, and diagrams

---

## 2. Environment & Prerequisites

* **Python Runtime:** Python `< 3.14`
* **Package & Task Runner:** `uv`
* **Local Model Cache:** All pipelines enforce `MINERU_MODEL_SOURCE=local` to consume pre-downloaded weights, ensuring deterministic, air-gapped execution without runtime network calls.

---

## 3. Directory Layout & Artifact Schema

The pipeline operates between two top-level directories:

```
thesis/
├── corpus-raw/                                   # Input: Raw authoritative source documents
│   └── <stem>.pdf
└── corpus-extracted/                         # Output: Multimodal document cache
    └── <stem>/
        └── hybrid_txt/
            ├── <stem>.md                         # Full extracted Markdown document
            ├── <stem>_content_list.json          # Block-level layout metadata & bounding boxes
            ├── <stem>_content_list_v2.json       # Extended layout metadata (schema v2)
            ├── <stem>_layout.pdf                 # Visual bounding box verification overlay
            ├── <stem>_origin.pdf                 # Snapshot copy of input document
            ├── <stem>_middle.json                # Intermediate layout analysis records
            ├── <stem>_model.json                 # Raw model activation coordinates
            └── images/                           # High-resolution figure & chart crops
                └── <sha256>.jpg
```

Downstream systems (such as `corpus-extracted/index.json`) locate the primary output document via:
`corpus-extracted/<stem>/hybrid_txt/<stem>.md`.

---

## 4. Operational Workflows

### 4.1 Batch Processing Mode (Persistent API Daemon — Recommended)

When processing multiple documents ($\ge 5$ papers), running a dedicated local FastAPI server avoids reloading model weights into memory on each invocation. This reduces total processing time by approximately 60%.

#### Step 1: Start the API Daemon (Terminal 1)
```bash
MINERU_MODEL_SOURCE=local \
MINERU_API_MAX_CONCURRENT_REQUESTS=1 \
MINERU_API_TASK_RETENTION_SECONDS=300 \
MINERU_API_OUTPUT_ROOT=/tmp/mineru-output \
  uv run --no-sync python -m mineru.cli.fast_api --host 127.0.0.1 --port 51139
```

#### Step 2: Dispatch Ingestion Tasks (Terminal 2)
```bash
MINERU_MODEL_SOURCE=local uv run --no-sync mineru \
  -p corpus-raw/<stem>.pdf \
  -o corpus-extracted \
  -b hybrid-engine \
  -m txt \
  --effort high \
  --api-url http://127.0.0.1:51139
```

> [!IMPORTANT]
> **Daemon Lifecycle Management:** The FastAPI server process accumulates ~60 MB of resident RAM per page processed. When executing long batches, restart the daemon every **100–150 pages** to prevent memory exhaustion.

---

### 4.2 Standalone Single-Document Ingestion

For ad-hoc additions of individual papers, execute the CLI directly:

```bash
MINERU_MODEL_SOURCE=local uv run --no-sync mineru \
  -p corpus-raw/<stem>.pdf \
  -o corpus-extracted \
  -b hybrid-engine \
  -m txt \
  --effort high
```

> [!WARNING]
> **Single-Document Invocations Only:** Always supply a single file to `-p`. Passing entire directories directly to the standalone CLI can cause thread locks or worker stalls. Use a shell loop over individual files instead.

---

## 5. Configuration & Parameter Reference

### 5.1 CLI Arguments

| Flag | Value | Description / Rationale |
|---|---|---|
| `-p` | `<path>` | Path to the source document (PDF). |
| `-o` | `corpus-extracted` | Target output directory where `<stem>/hybrid_txt/` will be generated. |
| `-b` | `hybrid-engine` | Combines layout analysis with VLM-based visual element extraction. |
| `-m` | `txt` | Skips redundant OCR on digital PDFs with a text layer. Use `ocr` for scanned reports. |
| `--effort` | `high` | Enables detailed chart and table structural analysis. Set to `medium` to disable. |
| `--api-url` | `http://127.0.0.1:51139` | Directs task execution to the persistent FastAPI server daemon. |
| `-s` / `-e` | `<int>` *(Optional)* | Start (`-s`) and end (`-e`) page numbers for partial document processing. |

### 5.2 Environment Variables

| Variable | Recommended Value | Functional Purpose |
|---|---|---|
| `MINERU_MODEL_SOURCE` | `local` | Enforces use of cached model weights; blocks external downloads. |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | `1` | Enforces strictly serial task execution to keep VRAM/RAM within safe bounds. |
| `MINERU_API_TASK_RETENTION_SECONDS` | `300` | Prunes completed task artifacts after 5 minutes (prevents disk bloat). |
| `MINERU_API_OUTPUT_ROOT` | `/tmp/mineru-output` | Redirects temporary extraction staging out of the project repository. |

---

## 6. Performance Characteristics & Resource Modeling

### 6.1 Latency Model

Execution latency is characterized by cold initialization overhead followed by linear per-page processing:

$$\text{Cold CLI Run:}\quad T_{\text{cold}} \approx 62\,\text{s} + 1.9\,\text{s} \times N_{\text{pages}}$$

$$\text{Warm Daemon Run:}\quad T_{\text{warm}} \approx 1.9\,\text{s} \times N_{\text{pages}}$$

* **Single Document Benchmark (20 pages):** ~100 s cold vs. ~38 s warm.
* **Corpus Benchmark (36 papers, 617 pages):** ~56 minutes cold (~37 minutes spent reloading models) vs. ~20 minutes via the warm API daemon.

### 6.2 Concurrency & Compute Constraints

* **Concurrency Ceiling:** Concurrency must be locked to `1` (`MINERU_API_MAX_CONCURRENT_REQUESTS=1`). Parallel requests trigger immediate out-of-memory (OOM) termination on current system memory configurations.
* **Memory Growth:** The daemon process does not automatically free all memory allocated during page OCR and VLM passes. Monitor memory via `htop` or `ps`, and cycle the daemon process periodically.

---

## 7. Data Integrity, Caching, & Known Caveats

### 7.1 Idempotent Ingestion Script

To prevent redundant compute, automated pipelines and scripts must check whether the target Markdown artifact already exists before dispatching an extraction job.

The following turnkey loop scans [`corpus-raw/`](corpus-raw/), verifies the cache, and sequentially processes any missing documents through the active API daemon:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Ingest all unextracted PDFs in corpus-raw/ via the local FastAPI daemon
for pdf in corpus-raw/*.pdf; do
  [ -e "$pdf" ] || continue
  stem=$(basename "$pdf" .pdf)
  target_md="corpus-extracted/${stem}/hybrid_txt/${stem}.md"

  if [ -f "$target_md" ]; then
    echo "[SKIP] ${stem}: already extracted at ${target_md}"
  else
    echo "[RUN]  Processing ${stem}..."
    MINERU_MODEL_SOURCE=local uv run --no-sync mineru \
      -p "$pdf" \
      -o corpus-extracted \
      -b hybrid-engine \
      -m txt \
      --effort high \
      --api-url http://127.0.0.1:51139
  fi
done
```

### 7.2 Chart & Visual Data Reliability

* **Approximated Table Values:** Charts analyzed under `--effort high` are rendered into Markdown `<details>` blocks containing tables. These numerical values are Vision-Language Model inferences and frequently introduce estimation symbols (e.g., `~84`) or inconsistent intervals (`0~100`).
* **Ground-Truth Policy:** Treat markdown `<details>` chart tables as qualitative indicators only. For empirical data, statistical baselines, or quantitative metrics in the thesis, consult the original chart figure in `corpus-extracted/<stem>/hybrid_txt/images/` or the source PDF.
