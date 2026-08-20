# Raw Markdown Guidelines

- To ensure readers understand files easily in raw text form, structure all Markdown documents clearly.
- To preserve raw text readability, limit formatting syntax strictly to section headings, bulleted lists, and bold text.

# Writing Guidelines

- To support non-native English readers, pair demonstrative pronouns with explicit nouns instead of using standalone pronouns such as "this" or "it" without referents.
- To keep text easy to translate, write syntactically clear and direct sentences.
- To express logical relationships clearly, use active voice and explicit conjunctions.
- To prevent ambiguity, use literal language and avoid idioms, metaphors, and contractions.

# Vale Guidelines

- To check user-written Markdown files across the repository, run pnpm exec vale .
- To turn off Vale on individual reference files, use in-file HTML comments (<!-- vale off --> and <!-- vale on -->) instead of editing .vale.ini.

# Marker Guidelines

- To organize extracted papers, keep one subfolder per paper in research/literature/ for the extracted markdown and metadata file.
- To extract digital papers efficiently, run Marker in fast mode with OCR turned off.
- To prevent accidental commits, keep source PDF files local, and never commit them.
- To track missing sources, record unavailable papers in research/literature-not-found.md instead of creating a note.

# Commit Guidelines

- To structure Git commits properly, follow the formatting rules and types defined in docs/commit-conventions.md.
- To maintain a uniform repository history, inspect the last 5 to 10 commits with git log and match their style.

