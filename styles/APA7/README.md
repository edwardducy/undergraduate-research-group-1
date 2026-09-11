# APA7 Vale style

Encodes the mechanically checkable rules from `thesis/docs/apa7-publication-manual.md`
(condensed: `thesis/docs/apa7-mechanics.md`). Every rule message cites the APA section
it enforces. Institutional (CEU) requirements live separately in `styles/CEU/`.

Run with `mise run lint` or `vale <file>`.

## Automated rules (styles/APA7/)

| Rule | Source |
| --- | --- |
| ParagraphLength | 4.6 — split paragraphs over ~250 words |
| Feel | 4.4 |
| Wordy | 4.5 — wordy-phrase replacements |
| Noncombative | 4.7 — "completely overlooked" → "did not address" |
| Colloquial | 4.8 — colloquialisms, vague quantities |
| Contractions | 4.8 |
| Hedging | 4.14 — "it would appear" |
| Anthropomorphic | 4.11 — theories/models that "conclude" |
| SelfReference | 4.16 — "the present authors" |
| SlashPronouns | 4.18 — "(s)he", "s/he", "he/she" |
| WhoForPeople | 4.19 — "participants that were" → "who" |
| Since, While | 4.22 |
| BetweenRange, BothAsWellAs | 4.24 |
| SerialComma | 4.24 / 6.3 — heuristic, suggestion only |
| BiasAgeDisability | 5.3–5.4 |
| MalesFemales | 5.3 |
| BiasGender | 5.5, 5.8 |
| BiasRace | 5.7 |
| BiasSES | 5.9 |
| AgencyFailed | 5.6 — "failed to complete" → "did not complete" |
| DoubleSpace | 6.1 |
| TechSpellings | 6.11 — "e-mail" → "email" |
| DataPlural | 6.11 |
| LyHyphen | 6.12 |
| SeriesNoun | 6.19 — "table 1" → "Table 1" |
| ApostrophePlural | 6.26 / 6.39 — "the 1960's" → "the 1960s" |
| EtAl | 6.29 — "et al" without the period |
| LatinAbbrev | 6.29 — e.g., i.e., etc., viz., vs., cf. |
| HyphenRange, SpacedDash, DoubleHyphen | 6.6 |
| UnitPlurals, UnitSpace | 6.26 / 6.27 |
| StatNoLeadingZero, StatAddLeadingZero, PValueZero, PValueDecimals | 6.36 |
| TypeErrors | 6.37 — "Type 2 error" → "Type II error" |
| PercentSymbol, PercentRange | 6.44 |
| OperatorSpacing | 6.45 |
| EqAbbrev | 6.47 — "Eq. 3" → "Equation 3" |
| Pointer | 7.5 — no "table above/on page/here" |
| LetteredDisplay | 7.19 — "Table 1A" |
| Ibid | 8.16 |
| PageRange | 8.25 — "p. 34-36" → "pp. 34–36" |
| QuotePunctuation | 6.7 — periods/commas outside closing quotes |
| NestedParens, AdjacentParens | 6.8 |
| AndOr | 6.10 |
| Intensifiers | 4.2 — "Importantly," |
| QuoteEllipsis | 8.31 |
| DOILabel | 9.34 |
| UrlShortener | 9.36 |
| Vale.Spelling (built-in) | 6.11 Merriam-Webster + project vocab |

## Documented candidates (audited, not implemented)

The full manual audit (67 implementable candidates) stopped at these because
they are false-positive-prone or structurally invisible to Vale. Revisit if a
manual pass finds them worthwhile:

- Restrictive "which" (4.21) — needs comma/clause parsing; noisier than useful.
- Sentence-initial numerals/lowercase abbreviations and "Based on" (6.33,
  6.26, 4.23) — Vale `^`/`$` anchors bind to the file, not the sentence scope.
- Sentence-ending URLs (6.2) — same `$` anchoring limitation.
- 40-word quotation threshold (8.26/8.27) — word counting across a quoted span
  is unreliable in RE2; the char-length proxy needs a leading quote at a word
  boundary, which Vale's wrapping rarely permits.
- Same-source overcitation (8.1) — occurrence counting cannot distinguish
  repeated citations of one source from several distinct sources.
- Heading title case (6.17), disease/theory capitalization (6.16), condition
  names (6.20) — interact badly with numbered headings, Quarto syntax, and
  title-case contexts.
- Trailing-symbol patterns like "18–20%" (6.44) and quote-final checks
  (6.7/8.31 variants) — Vale boundary-wrapping rejects keys ending in
  non-word characters; PercentRange covers the "18%–20%" direction only.
- Possessives of names ending in "s" (6.11), all-caps "'s" plurals vs
  possessives (6.26), italic statistical symbols in markdown (6.44) —
  ambiguous with legitimate forms.
- Acronyms used fewer than three times (6.24) — occurrence counting flags
  table-only and standard acronyms (ANOVA, IQ).
- Ampersand in narrative citations (8.17) — false-positives on table rows,
  where "&" is correct.
- Everything in the manual's judgment territory — ethics (1.x), bias nuances
  (5.1–5.2, 5.6, 5.10), tense control (4.12), passive voice (4.13),
  subject-verb agreement (4.15), misplaced modifiers (4.23), colon/comma
  clause rules (6.3, 6.5, 6.13), numeral-context rules (6.32–6.40),
  abbreviation definition at first use (6.25), heading structure (2.26–2.27),
  citation/reference correspondence (8–10), and all table/figure rendering
  (7.x) — stays with the AGENTS.md checklist and human review.

## Vocabulary

`styles/config/vocabularies/Thesis/accept.txt` whitelists thesis terms for the
spell-checker. **Do not add common English words**, even capitalized ones:
Vale masks vocabulary matches out of the text before other rules run, so an
accepted "The" or "50" silently blinds every phrase rule around them. The
collector in `scripts/build-vocab.py` therefore only harvests tokens with
digits, internal capitals, acronyms, or non-ASCII characters; run it after
chapters introduce new technical terms, and add proper nouns (Kalahi,
Cityscapes, Batayan, typhoon names…) to its MANUAL list. Entries are stored as
case-insensitive patterns (`[Tt]aglish`) so Vale.Terms never enforces a single
capitalization.
