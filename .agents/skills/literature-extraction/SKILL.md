---
name: literature-extraction
description: Extract structured records from research papers for the thesis literature review. Papers live as Markdown under research/literature/{paper_id}/. One subagent per paper follows the bundled schema in extraction-schema.json. Validated records land beside the paper as record.json. Use whenever the user asks to extract papers or build records. Use it when the user gives a paper URL, arXiv ID, or DOI for extraction. Use it for the reading list in research/sources.md. Use it to validate existing records. Also trigger on loose phrasings like "run the list" or "do the next batch of papers".
---

# Literature extraction

One paper in, one validated JSON record out.

## Files

- Schema. `.agents/skills/literature-extraction/extraction-schema.json` is the single source of truth. Its field descriptions are the extraction instructions. Each subagent must read the file itself.
- Source list. `research/sources.md`, one paper URL per line.
- Papers. `research/literature/{paper_id}/` holds three files. `paper.md` is the converted full text. `record.json` is the extraction record. `{paper_id}.pdf` is the original PDF.

## Workflow

1. Read the schema file first.
2. Collect the target URLs. Use the whole list, a subset, or whatever the user names.
3. Skip a URL when `research/literature/{paper_id}/record.json` exists and passes validation. Redo it only when the user asks.
4. Ensure the paper text. Run `uv run python3 scripts/fetch-paper.py {url...}` for the papers that still lack paper.md. Papers reported as MANUAL need a PDF dropped by hand into their folder. Tell the user about them and continue with the rest.
5. Spawn one general-purpose subagent per ready paper. Ready means paper.md exists. Run them in parallel batches of about five. Use the prompt template below.
6. Write each returned record to `research/literature/{paper_id}/record.json` yourself. Then run `uv run python3 .agents/skills/literature-extraction/validate.py` with no arguments to validate everything.
7. When validation fails, retry that paper once. Append the validation errors to the subagent prompt after step 2 of the template. When the second try also fails, keep the file and note the failure.
8. Close with one line per paper stating extracted, skipped, manual, or failed.

## Subagent prompt template

```
You are doing a literature extraction task.

1. Read the JSON schema file at
   .agents/skills/literature-extraction/extraction-schema.json.
   Its field descriptions ARE your instructions.
2. Read the paper at research/literature/{paper_id}/paper.md.
3. Produce one JSON record. It must follow the schema and validate against it.
   Use {url} as the source URL for paper_id and doi_or_url.
4. Your final message must contain ONLY the JSON record. No markdown fences,
   no commentary. Do not write any files. Never fetch the web.
```

## Notes

- Extraction reads the local paper.md and needs no network.
- One extraction costs roughly 100k tokens and a few minutes. Warn the user before a full list run.
- Records are canonical data for later semantic search. Never hand-edit or delete a record. Re-extract instead.
- Quote misses in paper.md print as warnings only. They never fail a record.
- The repository holds full paper texts. Keep it private.

## Example

`example-record.json` is a real record that passed this pipeline. Compare new output against it for field shape, quote length, and paraphrase style.
