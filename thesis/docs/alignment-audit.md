# Chapter Alignment Audit — `draft.md` vs. Chapters 2–3

**Date:** 2026-09-11 (verified 2026-09-11 — see Verification log)
**Compared:** `archive/chapter-1/draft.md` (legacy Chapter 1) against `chapters/chapter-2/draft.qmd` and `chapters/chapter-3/draft.qmd`
**Method:** Fresh section-by-section outline of `draft.md`, cross-check of every design commitment against the current chapter drafts. Factual disputes were settled against the paper records under `papers/` and confirmed against the published sources (web).
**Status:** Advisory. No manuscript files were modified.

---

## Bottom line

- The **skeleton aligns**: corpus name, size targets, typhoons, LOEO folds, seeds, the 7 MTO algorithms vs. STL/EW/LS, training constants, gradient diagnostics, two-tier statistics, DSRM/IPO structure — Chapters 2–3 explicitly lean on Chapter 1 for all of these.
- The **content diverges** on five fundamental points. Almost all are design constants that were revised *after* `draft.md` was archived; Chapter 1 (both versions) is the stale side, except for two Chapter 2 text bugs.
- One design block — **the stratified code-switching apparatus** — exists only in Chapters 2–3 and is promised by no Chapter 1 problem or objective. It is the main reason the three chapters do not close.
- Two attribution claims fail source verification: the "four-level" urgency scale (the source is binary) and the "10" HumAID categories (the paper defines 11).

---

## Fundamental mismatches

### 1. Urgency task definition — both existing versions are wrong

| | Claim | Source |
|---|---|---|
| `draft.md` | 3-tier urgency, macro F1 over 3 classes | `draft.md:41,204` |
| Chapter 3 | "four-level scale of @sarioglukayi2020detecting", converted to binary | `chapter-3/draft.qmd:25` |
| **Verified** | The Sarioglu Kayi paper defines urgency as "**a binary classification task** where the labels correspond to urgency status" | `papers/acl-2020.coling-main.414/paper.md`; confirmed on [ACL Anthology](https://aclanthology.org/2020.coling-main.414/) |

**Ruling.** Urgency = **binary, urgent / not urgent**, cited directly to Sarioglu Kayi (2020). Chapter 3 already scores it this way; only its "four-level scale" attribution is wrong. Strike "3-tier" from Chapter 1 (unsourced) and "four-level scale" from Chapter 3 (unattributable). The same "3-tier urgency" cell also sits in the current chapter-1 qmd (`chapters/chapter-1/draft.qmd:41`). Fundamental because urgency is one of the three heads in the thesis title.

### 2. Intent taxonomy — the archived draft's number is wrong

| | Claim | Source |
|---|---|---|
| `draft.md` | HumAID has "10 humanitarian intent categories" | `draft.md:37` |
| Chapter 3 | intent merged from "11 HumAID categories"; merged count never stated | `chapter-3/draft.qmd:136,25` |
| **Verified** | The HumAID paper defines **11** annotation categories | `papers/arxiv-2104.03090/paper.md`; confirmed on [arXiv:2104.03090](https://arxiv.org/abs/2104.03090) and the [ICWSM 2021 proceedings](https://ojs.aaai.org/index.php/ICWSM/article/view/18116/17919) |

The 11: Caution and advice · Sympathy and support · Requests or urgent needs · Displaced people and evacuations · Injured or dead people · Missing or found people · Infrastructure and utility damage · Rescue, volunteering, or donation effort · Other relevant information · Not humanitarian · Don't know or can't judge. (The "10 classes" figure circulates in follow-up benchmark work that drops or merges a label; the original specification is 11.)

**Ruling.** Fix 10 → 11 everywhere. Keep the 4-class merge as a design choice, but Chapter 3 must state it explicitly — the merged class count currently appears in **no** chapter, leaving "intent macro F1" with an undefined class space.

### 3. The stratified code-switching apparatus is orphaned

Word-level language tags, three language profiles, the code-mixing index, subword fertility, and hypotheses H13/H14 exist **only in Chapters 2–3**. No specific problem or objective in either version of Chapter 1 promises them — yet Chapter 3 claims its hypothesis pairs "map onto the specific problems of Chapter 1" (`chapter-3/draft.qmd:152`) and assigns H13/H14's topic to "Specific Problem 4" (`chapter-3/draft.qmd:228`), whose actual content in Chapter 1 is bootstrap/Friedman validation and the Δm–conflict correlation.

**Ruling.** See removals 1–2 below. This is the single most consequential decision: it changes Chapter 3's run plan and Chapter 2's synthesis.

### 4. Corpus validation protocol

| | Protocol |
|---|---|
| `draft.md:208` | "senior expert adjudication with an agreement threshold of span-level F1 ≥ 0.70" |
| Chapter 3 (`:136-138`) | three judgments, 2-of-3 approval, per-token majority, Fleiss' kappa with revision trigger |

**Ruling.** Adopt Chapter 3's protocol. The 2-of-3 rule is verifiably HumAID's own published procedure (66% agreement threshold, confirmed in the paper text). The 0.70 expert threshold appears in no source and is circular — span-level F1 needs a gold standard, and the definition never says whose labels serve as one.

### 5. Efficiency measurement platform and statistic

| | Platform | Statistic |
|---|---|---|
| `draft.md:139,172,212` | GPU testbed, CUDA event synchronization | sample mean ± 95% CI |
| Chapter 3 (`:95,144,238-244`) | CPU hardware (the LGU deployment condition) | median |

**Ruling.** Adopt Chapter 3's side — it matches the thesis's own deployment argument (commodity workstations, CPU-only mode, CPU stress test). Chapter 1 moves, not the reverse.

---

## Remove from the research

1. **H13/H14 as hypotheses** (`chapter-3/draft.qmd:160`). A descriptive robustness check, not a research question any chapter asks — and the weakest tests in the design: Friedman–Nemenyi over 10 configurations inside each profile draws on ~1,500–2,500-tweet held-out folds split three ways. Removing them drops the hypothesis count from 14 to 10 and makes Chapter 3's "maps onto Chapter 1" claim true.
2. **The word-level language-tag layer as a manual, full-corpus layer** (`chapter-3/draft.qmd:25,32,134`). The only per-word annotation in the design, the most subjective call in the codebook (mixed vs. ambiguous), tripling annotation labor to feed what Chapter 3 itself calls "a descriptive companion" analysis. Its citation also fails: TweetTaglish is *document-level* language-mixing labels (`chapter-2/draft.qmd:93`; `draft.md:39`), not a word-level tag convention. **Keep** the code-mixing index and subword fertility as descriptive statistics computed automatically on saved predictions — the Taglish story survives at near-zero cost.
3. **Training runtime as a benchmarked metric** (`draft.md:59,139,254`). Chapter 3 never operationalizes it; wall-clock is hardware-confounded at K=3 scale. State overhead analytically instead: surgery methods need K per-task backward passes, weighting methods a single weighted pass, FAMO is designed to avoid per-task backpropagation (`chapter-2/draft.qmd:69`; confirmed against the [NeurIPS 2023 paper](https://neurips.cc/virtual/2023/poster/69900), which targets the O(T) cost of per-task gradients).
4. **M_VRAM (peak GPU memory) from the serving battery** (`draft.md:139,224`). Meaningless under Chapter 3's CPU measurement condition — Chapter 3 is internally inconsistent here, listing pynvml/GPU-memory readings while measuring on CPU (`chapter-3/draft.qmd:95,144`). Host RAM stays. GPU memory may survive as a one-line training-footprint descriptor.
5. **CUDA-event / mean ± 95% CI wording, and the "gradient metrics versus STL" comparison in SP1** (`draft.md:172,55`). Single-task models have no cross-task gradient geometry; Chapter 3 already handles this correctly (H01 is F1-only).
6. **"Senior expert adjudication, span-level F1 ≥ 0.70"** (`draft.md:208`) — replaced by Chapter 3's source-faithful protocol (mismatch 4).
7. **Unverified encoder parameter counts** (~279M / 178M / 313M / 125M, `draft.md:194`) — no other chapter uses them; the deployment argument rests on measured latency and RAM. Remove or verify against model cards.
8. **Do not revive, if `draft.md` is ever reused:** the NDRRMC-based "twenty typhoons / five catastrophic" claim (per-event sitreps cannot support an annual average; the current qmd's PAGASA re-attribution is correct — [PAGASA states](https://www.pagasa.dost.gov.ph/climate/tropical-cyclone-information) "the average of 20 TCs … with about 8 or 9 of them crossing the Philippines") and the manual 27-entry reference list with the Olteanu hashtag citation (no bib record exists; the claim now correctly cites Ermino).

---

## Keep — realign instead

- **Weighted F1** (`chapter-2/draft.qmd:25,130`). Keep as a secondary descriptive metric: free (same predictions), and the imbalance rationale is sound for a skewed intent set. Hypotheses stay on macro F1.
- **Batayan.** Do not drop it from Chapter 1's Table 1 — it is the strongest evidence for the "Philippine benchmarks are task-isolated" gap claim. Instead, add one review paragraph to Chapter 2, which covers Kalahi but never Batayan (verified: zero Batayan mentions in `chapters/chapter-2/draft.qmd`).
- **PostgreSQL.** Settled by Chapter 3 (`:41,329`); just delete the "or SQLite" alternative from Chapter 1 (`draft.md:129,228`).
- **Chapter 2 text bugs.** "Nine candidate backbones" → four (`chapter-2/draft.qmd:83,87`); "10 optimization configurations against … baselines" → "ten configurations, comprising the seven MTO methods and the three baselines" (`chapter-2/draft.qmd:120,134`).
- **Editorial.** `B-INF/I-INF` vs. `INFRA` tag naming; "Multi-Task" vs. "multitask" hyphenation; the Table 1 / Table 1 numbering collision (adopt Chapter 3's chapter-prefixed numbering thesis-wide).
- **Spearman variables.** Chapter 1 names cosines *and* R_max/min (`draft.md:246`); Chapter 3 uses the mean pairwise cosine only (`chapter-3/draft.qmd:184`). Align on Chapter 3's narrower version.

---

## Action checklist

| # | Item | Action | Where |
|---|---|---|---|
| 1 | Urgency = binary, per Sarioglu Kayi | Fix attribution; delete "3-tier" and "four-level" | Ch1 + Ch3 |
| 2 | HumAID = 11 categories | Fix "10"; restate merged count = 4 in Ch3 | Ch1 + Ch3 |
| 3 | Stratified apparatus | Remove manual language-tag layer + H13/H14; keep CMI/fertility as descriptive auto-computed | Ch2 + Ch3 |
| 4 | Corpus validation | Adopt 2-of-3 + Fleiss kappa; delete expert / 0.70 wording | Ch1 |
| 5 | Efficiency | Adopt CPU + median; delete M_VRAM, CUDA events, mean ± CI | Ch1 (+ fix Ch3 pynvml inconsistency) |
| 6 | Training runtime | Remove as metric; one analytic sentence instead | Ch1 |
| 7 | Gradient metrics vs. STL | Remove from Specific Problem 1 | Ch1 |
| 8 | Parameter counts | Remove or verify against model cards | Ch1 |
| 9 | Weighted F1 | Keep, as secondary descriptive metric | Ch3 (add); Ch2 already promises it |
| 10 | Batayan | Add review paragraph | Ch2 |
| 11 | "Nine backbones", "10 vs. 3" phrasing | Fix text | Ch2 |
| 12 | SQLite, tag naming, hyphenation, numbering | Editorial alignment | All |

---

## Evidence index

- **Urgency source (binary):** `papers/acl-2020.coling-main.414/paper.md` — "binary classification task … urgency status"; [ACL Anthology](https://aclanthology.org/2020.coling-main.414/)
- **HumAID label set (11):** `papers/arxiv-2104.03090/paper.md`, annotation scheme section; [arXiv:2104.03090](https://arxiv.org/abs/2104.03090); [ICWSM 2021](https://ojs.aaai.org/index.php/ICWSM/article/view/18116/17919)
- **HumAID 2-of-3 protocol:** same paper — 66% label-agreement threshold, three judgments per tweet
- **PAGASA cyclone climatology:** [Tropical Cyclone Information](https://www.pagasa.dost.gov.ph/climate/tropical-cyclone-information) — average of 20 TCs per year, about 8 or 9 crossing the Philippines
- **FAMO design:** [NeurIPS 2023 poster](https://neurips.cc/virtual/2023/poster/69900) — targets the O(T) cost of per-task gradients
- **Chapter 2 key lines:** `:25` weighted-F1 adoption · `:81` CMI forward reference · `:83,87` "nine candidate backbones" · `:93` TweetTaglish document-level · `:120,134` "10 configurations against … baselines"
- **Chapter 3 key lines:** `:25,32,35` corpus layers · `:41,54,329` PostgreSQL · `:95` median + pynvml · `:136-138` annotation protocol, "11 HumAID categories" · `:144` CPU measurement · `:152,160` hypothesis mapping, H13/H14 · `:198` binary urgency scoring · `:220-234` CMI + fertility
- **`draft.md` key lines:** `:37,41,204` urgency/intent counts · `:55,172,212` GPU/CUDA/mean ± CI · `:59,139,254` training runtime · `:129,228` SQLite-or-PostgreSQL · `:208` expert adjudication / 0.70 · `:246` Spearman variables

---

## Verification log (2026-09-11)

All claims in this audit were re-verified after the chapter drafts gained an H1 heading + blank line each (+2 line shift; content otherwise unchanged since the analysis pass).

- **In-repo pointers:** 37 content checks re-run against the current files — all claims found verbatim; line numbers above refreshed to the shifted positions. All `draft.md` pointers matched exactly.
- **Long quotes confirmed intact:** "serve as two of the study's nine candidate backbones" (Ch2), "converts to a binary urgent or not urgent value" (Ch3), "The pairs map onto the specific problems of Chapter 1" (Ch3).
- **Source-paper claims re-verified in local records and confirmed on the web:**
  - Sarioglu Kayi (2020) urgency task is binary — local paper text + [ACL Anthology](https://aclanthology.org/2020.coling-main.414/).
  - HumAID defines 11 annotation categories — 11 definitions counted programmatically in the local paper copy; count confirmed via [arXiv](https://arxiv.org/abs/2104.03090) / [ICWSM](https://ojs.aaai.org/index.php/ICWSM/article/view/18116/17919). The "10 classes" figure comes from follow-up variants that drop or merge a label, so `draft.md`'s "10" traces to no source.
  - HumAID approval rule is 2-of-3 at 66% agreement — confirmed in the paper text.
  - PAGASA: "average of 20 TCs … about 8 or 9 of them crossing the Philippines" — confirmed verbatim on [PAGASA's page](https://www.pagasa.dost.gov.ph/climate/tropical-cyclone-information), validating the current chapter-1 qmd's re-attribution.
  - FAMO targets the O(T) cost of per-task gradients — confirmed via [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/69900) / [OpenReview](https://openreview.net/forum?id=zMeemcUeXL).
- **Nuance noted:** the only `papers/` match for "Olteanu" is a different paper's record citing Olteanu et al. (2014) CrisisLexT6 — not a record for Olteanu et al. (2015). `references.bib` contains no Olteanu and no NDRRMC entries, supporting removal item 8.
