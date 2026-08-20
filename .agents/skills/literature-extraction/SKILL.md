---
name: literature-extraction
description: >-
  Extracts structured research data, methodology, findings, boundaries, and
  research gaps from academic literature markdown files in
  research/literature/ based on research/literature-extraction-schema.json.
  Use this skill when the user requests extracting paper details, analyzing
  literature, or identifying research gaps.
---

# Literature Extraction Skill

## Overview
- **Purpose**: This skill guides the primary agent and subagents in extracting structured academic data and research gaps from literature markdown files.
- **Authoritative Schema**: All extractions must conform strictly to `research/literature-extraction-schema.json`.
- **Target Location**: All candidate papers reside as subdirectories within `research/literature/`.

## Workflow Steps

### 1. Identify Target Papers
- **Directory Scan**: Scan `research/literature/` for target paper subdirectories.
- **Exclusion Filter**: Exclude binary PDF files, diagram image files, and the missing sources tracker at `research/literature-not-found.md`.
- **Target Files**: Locate the primary markdown document within each paper subdirectory (for example, `mtl-survey.md`).

### 2. Read the Extraction Schema
- **Schema Reference**: Read `research/literature-extraction-schema.json` to verify required properties and field descriptions.
- **Required Properties**: Ensure the extraction includes all seventeen root-level properties:
  - **title**: Formal paper title.
  - **authors**: List of author names.
  - **year**: Publication year as an integer or null.
  - **venue**: Publication outlet or conference name, or null.
  - **publication_type**: Publication genre (such as empirical study, systematic survey, benchmark dataset, system paper, or theoretical analysis).
  - **problem_statement**: Investigated problem, objective, or phenomenon.
  - **domain**: Application area, language setting, or operational environment.
  - **theoretical_framework**: Underlying theories or assumptions, or null.
  - **methodology**: Primary approach, algorithm, or analytical design.
  - **datasets**: Evaluated corpora, datasets, or artifacts, or null if non-empirical.
  - **evaluation**: Benchmark metrics, scores, or qualitative procedures, or null if non-empirical.
  - **key_findings**: Array of primary empirical or theoretical findings.
  - **scope_and_delimitations**: Intentional study boundaries chosen by the authors.
  - **limitations**: Object containing **verbatim_excerpt** and **summary** of acknowledged weaknesses.
  - **generalizability_gap**: Untested populations, languages, or scales, or null.
  - **methodological_gap**: Measurement, baseline, or tool shortcomings, or null.
  - **future_work**: Array of future research directions recommended by the authors, or null.

### 3. Subagent Invocation Strategy
- **Subagent Delegation**: The primary agent invokes a generic subagent for each target paper using `invoke_subagent` (Antigravity, native) or `subagent` (Pi with `pi-subagents` extension via `pi install npm:pi-subagents`).
- **Subagent Role**: Set the subagent role to `Literature Data Extractor` (provide this as the role description in the subagent prompt, not as a custom agent name).
- **Subagent Prompt Instructions**:
  - Provide the path to the paper markdown file.
  - Provide the path to `research/literature-extraction-schema.json`.
  - Direct the subagent to read the entire markdown document thoroughly.
  - Direct the subagent to populate all fields according to the schema rules.
  - Instruct the subagent to use exact verbatim excerpts for the limitation quote.
  - Instruct the subagent to assign explicit null values to inapplicable properties.

### 4. Output Storage and Verification
- **Output File**: Save the resulting JSON object as `summary.json` within the respective paper subdirectory (for example, `research/literature/<paper_directory>/summary.json`).
- **JSON Validation**: Validate that `summary.json` contains valid, parseable JSON and matches all schema constraints.
- **Summary Report**: The primary agent reports the completed extractions and lists any skipped or invalid entries.
