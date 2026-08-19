# Raw Markdown Guidelines

- To ensure readers understand files easily in raw text form, structure all Markdown documents clearly.
- To preserve raw text readability, limit formatting syntax strictly to section headings, bulleted lists, and bold text.

# Writing Guidelines

- To support an audience that speaks English as a second language, write clear and accessible text.
- To keep text easy to translate, write short and direct sentences.
- To convey meaning clearly, use active voice and explicit conjunctions.
- To prevent ambiguity, avoid obscure terms, idioms, and contractions.

# Vale Guidelines

- To check user-written Markdown files across the repository, run pnpm exec vale .
- To turn off Vale on individual reference files, use in-file HTML comments (<!-- vale off --> and <!-- vale on -->) instead of editing .vale.ini.

# Marker Guidelines

- To organize extracted papers, keep one subfolder per paper in research/literature/ for the extracted markdown and metadata file.
- To prevent accidental commits, keep source PDF files local and never commit them.
- To track missing sources, record unavailable papers in research/literature/not-found.md instead of creating a note.
