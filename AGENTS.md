# AGENTS.md

## Project Context

- This repository contains Quarto sources for thesis PDF output.
- The audience includes non-native English readers requiring clear prose.
- The thesis spans three chapters plus literature records and references.

## Commands

- Run `quarto render` for full output.
- Run `quarto render chapters/chapter-1/revised-draft.qmd` for chapter 1 checks.
- Run `quarto render chapters/chapter-2/revised-draft.qmd` for chapter 2 checks.
- Run `quarto render chapters/chapter-3/draft.qmd` for chapter 3 checks.
- Run `python scripts/build-references.py` before final renders when references change.
- Run `mise run <task>` to run defined project tasks (`build:refs`, `build:tables`, `build:figures`, `render`).

## File Map

- Thesis chapters live in `chapters/chapter-1`, `chapters/chapter-2`, and `chapters/chapter-3`.
- Style mechanics live in `docs/apa7-mechanics.md`.
- Full manual text lives in `docs/apa7-publication-manual.md`.
- Literature tables live in `literature/` with CSV sources.
- References live in `references.bib` with JSON mirror.

## Writing Rules

- Keep each sentence at 30 words or fewer with varied length.
- Avoid first-person pronouns and second-person pronouns in prose.
- Phrase own actions impersonally in active voice.
- Keep numbered thesis headings in order without skipped levels.
- Follow `docs/apa7-mechanics.md` for remaining APA mechanics.
- Consult `docs/apa7-publication-manual.md` only when the draft file lacks detail.

## Validation Checklist

- Confirm sentence length, pronoun ban, and heading order.
- Confirm every citation has a matching reference entry.
- Confirm tables and figures include callouts by number.
- Confirm Quarto output builds without warnings.
