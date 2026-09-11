# AGENTS.md

## Project Context

- The audience includes non-native English readers requiring clear prose.

## Commands

Run all commands from the repository root. `mise` tasks handle paths.

- Run `mise run build` for a full rebuild: references, tables, vocabulary, render.
- Run `mise run render` for the combined thesis and all chapter PDFs.
- Run `mise run chapter 2` to render one chapter (accepts 1, 2, or 3).
- Run `mise run lint` to lint the three chapter drafts with Vale.
- Run `mise run vocab` after adding new technical terms to the prose.
- Run `cd thesis && quarto render` for direct Quarto control.

## Writing Rules

- Keep each sentence at 30 words or fewer with varied length.
- Avoid first-person pronouns and second-person pronouns in prose.
- Phrase own actions impersonally in active voice.
- Follow `thesis/docs/apa7-mechanics.md` for remaining APA mechanics.
- "Refer to `thesis/docs/apa7-publication-manual.md` for detailed guidance on APA 7th edition standards.
