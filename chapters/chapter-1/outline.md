# Chapter 1 Outline

Proofreading plan for Chapter 1: Introduction. The chapter is content-complete: all seven items in `chapters/chapter-1/feedback.md` are applied, so this outline guides a proofreading pass, not a rebuild. Sources: `chapters/chapter-1/draft.md`, `feedback.md`, the CEU rubric in `chapters/chapter-1/guidelines.md` (superseded on narrative structure by `feedback.md`), the 55 primary and 48 secondary records in `literature/*.csv`, the paper texts under `papers/{paper_id}/paper.md`, and `AGENTS.md`.

## Conventions

- Proofreading target is `chapters/chapter-1/draft.md`. The proofread text then converts to `draft.qmd` in the apaquarto-pdf pipeline, matching Chapters 2 and 3. Headings take institutional numbers 1.1 through 1.8 at conversion, matching the `## 2.1 ...` style in Chapter 2; the opening before 1.1 stays unnumbered.
- AGENTS.md governs all mechanics: 30-word sentence cap, impersonal active voice, no contractions, numeral and citation rules.
- Citation base is the 55 papers in `literature/primary-literature.csv`. Chapter 2 is primary-only; Chapter 1 additionally keeps three secondary records already present in `references.bib` (`ruder2017overview`, `hevner2004design`, `chen2024multitask`) plus institutional and legal references. Four draft entries have no record anywhere and need entries added (see Reference accounting).
- Verification rule, same as Chapter 2: every statistic must trace verbatim to `papers/{paper_id}/paper.md` or the paper's CSV record. Unverifiable numbers are dropped or replaced with the nearest vetted value.
- Surname initials per APA 8.20: B. Liu (2021 CAGrad; 2023 FAMO) and L. Liu (2021 IMTL) already carry initials. The two Chen first authors do not: write Z. Chen et al. (2018, GradNorm) and S. Chen et al. (2024, overview) at every citation.
- At qmd conversion the manual reference list is replaced by bib-driven rendering. Entries without bib keys vanish silently, so the four missing records below must exist before conversion.

## Feedback status

All seven feedback items are applied in `draft.md`:

1. Chapter titled "Introduction". Applied.
2. Title page on its own page. Leftover, mechanical: handled by manuscript-level Quarto front matter.
3. References on a standalone page. Leftover, mechanical: page-break handling at qmd conversion.
4. Background merged into one cohesive narrative without Macro/Meso/Micro breaks. Applied; this overrides the funnel model still described in `chapters/chapter-1/guidelines.md`.
5. SOP restructured to a general problem paragraph plus four interrogative questions. Applied.
6. Two IPO diagrams with narrative below each. Applied in text, but `figures/ipo-diagram-1.png` and `figures/ipo-diagram-2.png` do not exist; the `figures/` directory is absent.
7. Objectives enumerated and paired one-to-one with the problems. Applied.

## Section plan

### Opening, unnumbered

Three paragraphs before 1.1: the disaster-triage motivation and the three concurrent tasks (token-level NER, sequence-level intent, sequence-level urgency); the STL O(K) versus hard-parameter-sharing O(1) contrast and the two gradient failure modes on Taglish; the study's response, a DSRM-grounded benchmark of MTO algorithms against EW, LS, and STL baselines under 4-fold LOEO with gradient diagnostics and a web prototype. No citations; all sourcing is deferred to 1.1.

### 1.1 Background of the Study

One cohesive narrative (feedback item 4) in nine moves:

1. Triage information types: location entities, humanitarian intent, urgency (Alam et al., 2021; Wang et al., 2021).
2. Hard parameter sharing: single multilingual encoder feeding lightweight task heads, O(1) latency, inductive bias (Z. Chen et al., 2024; Ruder, 2017).
3. Multi-objective optimization and the two failure modes, with inner-product math (Sener & Koltun, 2018; B. Liu et al., 2021; Navon et al., 2022; Yu et al., 2020; Z. Chen et al., 2018; Weld et al., 2022).
4. MTO taxonomy: dynamic loss weighting (Kendall et al., 2018; Z. Chen et al., 2018; B. Liu et al., 2023) versus gradient surgery (Yu et al., 2020; B. Liu et al., 2021; L. Liu et al., 2021; Navon et al., 2022).
5. Empirical skepticism under adaptive optimizers and the intra-task half-batch baseline (Xin et al., 2022; Elich et al., 2024).
6. Philippine disaster context, four named typhoons, civic hashtags (NDRRMC sitreps; Olteanu et al., 2015).
7. Taglish linguistic complexity and tokenizer fragmentation (Herrera et al., 2022; Montalan et al., 2025; Petrov et al., 2023; Adouane & Bernardy, 2020).
8. Local deployment constraints, cloud-LLM risks, Data Privacy Act of 2012.
9. Table 1 corpus gap, the token-versus-sequence gradient asymmetry argument, and the corpus construction response.

Table 1 compares five corpora (HumAID, Batayan, TweetTaglish, TLUNIFIED-NER, proposed Multi-Task Corpus) on six structural columns and is called out in text.

### 1.2 Statement of the Problem

General problem paragraph summarizing the full chain: local-hardware triage, O(K) to O(1), gradient failure modes, conflicting MTO evidence under AdamW, mini-batch noise confound, and the missing Taglish corpus. Four interrogative specific problems keyed to the DSRM phases: baseline analysis (Phase 1), design and artifact creation (Phase 2), empirical benchmarking (Phase 3), comparative validation (Phase 4).

### 1.3 Objectives of the Study

General objective, then four declarative specific objectives mirroring 1.2 one-to-one (feedback item 7). Metrics named identically to 1.2: F1 variants, transfer, efficiency, runtime, statistics.

### 1.4 Theoretical Framework

Three foundations. First, DSRM (Peffers et al., 2007; Hevner et al., 2004) with the six-activity lifecycle mapped onto the four study phases. Second, complexity and inductive-bias analysis (Ruder, 2017) with two displayed equations for shared-forward-pass time and parameter memory. Third, multi-objective optimization and gradient dynamics (Sener & Koltun, 2018; Z. Chen et al., 2018; Kendall et al., 2018; Elich et al., 2024; B. Liu et al., 2021; Xin et al., 2022; Yu et al., 2020) with the vector-loss formulation, the first-order Taylor expansion, and the two bold-labeled failure modes. Closes with the intra-task baseline (Elich et al., 2024), the two-tier statistics (Demšar, 2006), and the handoff to the IPO framework.

### 1.5 Conceptual Framework

IPO model across the four DSRM phases. Figure 1 (overall study): inputs I1 to I4 (corpus, four candidate encoders, optimization configurations, workstation testbed), processes P1 to P4 (preprocessing and LOEO splitting, fine-tuning with gradient diagnostics, statistical validation, FastAPI deployment), outputs O1 to O4 (model, benchmark tables, inferential results, prototype). Figure 2 (software artifact): asynchronous FastAPI service, relational incident database, gazetteer resolution with fallback, and the four-state incident lifecycle. Narrative explanation follows each figure, per feedback item 6.

### 1.6 Scope and Delimitations

Scope: corpus volume and four typhoon events, 4-fold LOEO with three seeds, four encoder-only backbones, seven MTO methods against STL, EW, and LS, full metric battery, statistical protocol, software deliverable. Delimitations: Taglish only, Twitter/X historical events only, text only, encoder-only hard parameter sharing, three triage tasks only, research artifact rather than operational integration.

### 1.7 Significance of the Study

Three beneficiary groups in order: disaster response agencies (primary), machine learning and crisis informatics researchers (secondary), academic institutions and future system developers (tertiary).

### 1.8 Definition of Terms

34 alphabetized operational definitions, mathematical where applicable: execution latency, gradient disparity ratio, dynamic weighting, directional conflict, intra-task baseline, macro F1, corpus, throughput, negative transfer, non-parametric testing, Pareto optimality, memory metrics, relative transfer, sequence labeling versus classification, STL baseline, span F1, Spearman rho, LS, Taglish, runtime, EW. Terms match the metrics used in 1.2 and 1.3 and the treatments in Chapter 3.

## Reference accounting

The draft carries 27 entries.

- 20 primary, all with bib keys: adouane2020when, alam2021humaid, chen2018gradnorm, demsar2006statistical, elich2024examining, herrera2022tweettaglish, kendall2018multitask, liu2021conflictaverse, liu2021impartial, liu2023famo, miranda2023developing, montalan2025batayan, navon2022multitask, peffers2007design, petrov2023language, sener2018multitask, wang2021transformerbased, weld2022survey, xin2022current, yu2020gradient.
- 3 secondary, already in `references.bib`: chen2024multitask, hevner2004design, ruder2017overview.
- 4 in neither CSV nor `references.bib`; each needs a record before qmd conversion or the citations must be replaced: Olteanu et al. (2015), NDRRMC (2020), NDRRMC (2022), Republic Act No. 10173.
- Year letters: alam2021humaid collides thesis-wide with alam2021crisisbench, and miranda2023developing with miranda2023calamancy. Chapter 1 cites only one of each pair, so verify the rendered 2021a/2021b and 2023a/2023b suffixes once the manuscript bibliography assembles.

## Defect register

### Statistics

1. Typhoon-frequency misattribution. "An average of twenty typhoons annually, with approximately five causing catastrophic damage" is cited to the NDRRMC situational reports, which are per-event documents that state no annual average. Traced sources: [PAGASA](https://www.pagasa.dost.gov.ph/climate/tropical-cyclone-information) gives an average of 20 tropical cyclones per year in the Philippine Area of Responsibility; the [CFE-DM hazard review](https://www.cfe-dmha.org/LinkClick.aspx?fileticket=8RPXU4W0zzQ%253D&portalid=0) gives an average of five destructive ones. Fix: add a PAGASA or CFE-DM record and re-attribute; keep the sitreps for event-level facts only.
2. Tokenizer-inflation claim. "(inflating sequence length by 2x to 7x)" attributed to Petrov et al. (2023) is unvetted. The paper record supports Tagalog fertility of 1.74 to 3.28 subword pieces per word across six tokenizers and length ratios up to roughly 2.3 versus English; the headline 15 and 13 times factors apply to other languages. Replace with the vetted Tagalog range and spell multipliers as words, matching Chapter 2 number style.
3. TLUNIFIED-NER "208,247 tokens" appears in neither the paper text nor the CSV record; Chapter 2 already flagged the same number (its defect item 2). Confirm against the PDF or drop the cell.
4. Verified and safe: HumAID 77,196 tweets, 19 disaster events, ten humanitarian categories, and event-wise splits; Batayan 3,800 test instances across eight tasks; TweetTaglish 21,150 tweets.
5. Encoder parameter counts (XLM-R about 279M, mBERT about 178M, SEA-LION ModernBERT about 313M, RoBERTa-Tagalog about 125M) trace to no local record. Verify against the model cards and Cruz and Cheng (2022), or drop the parenthetical.

### Citation mechanics

6. Chen initials missing; add Z. and S. per the Conventions rule.
7. NDRRMC abbreviation defined twice: once inside the parenthetical citation and again spelled out two sentences later. Define once in narrative at first mention and abbreviate thereafter (APA 8.21).
8. The four unrecorded entries render as empty citations at conversion; see Reference accounting.

### Citation coverage

9. Uncited or misattributed claims. The opening and several Background sentences asserted field-establishing facts without citations: agencies relying on social media, the single-task deployment norm, negative transfer on code-switched text, the borrowed intra-task half-batch baseline, and overwhelmed manual triage. Fixed in `revised-draft.qmd` with imran2016twitter and olteanu2015what (social media reliance), ruder2017overview (single-task norm), adouane2020when (code-switched negative transfer), elich2024examining (half-batch baseline), and ermino2022evaluating (manual triage overwhelm). A second pass caught two subtler cases: the opening stated that extraction "requires" NLP models to execute three tasks, conflating the field's volume-necessity finding with the study's own three-task formulation, and the two-failure-modes taxonomy was uncited. Fixed by splitting necessity from formulation, attributing the task elements to alam2021humaid and wang2021transformerbased, and anchoring the failure modes with chen2018gradnorm and yu2020gradient. Audit standard for sections 1.2 to 1.8: every claim-level borrowing needs attribution, and design decisions must read as the study's formulations, not as field requirements. Two contextual claims remain uncited because no record in either CSV supports them: the commodity-workstation and outage conditions of local government units, and the reliance of modern Transformer fine-tuning on AdamW. The group must either locate sources or consciously accept them as context.

10. Citation-content misattribution (found by the independent two-eye audit). A citation discharges a claim only when the cited record contains it. Fixed instances in `revised-draft.qmd`: HumAID cited for location entities, though HumAID holds sequence-level humanitarian classes only, so the citation moved to the intent element; Ruder cited for the O(K) latency arithmetic, though Ruder predates Transformers, so the arithmetic now stands uncited as the study's own analysis; Elich cited for "AdamW" though the record uses Adam, so the findings sentence says Adam and AdamW survives only as the study's optimizer; Adouane cited for "such as Tagalog-English (Taglish)" though the record covers Algerian Arabic-French text, so the Taglish example moved to the evaluation sentence; Ermino cited for "overwhelmed during peak landfall hours", reworded to the record's actual finding on inefficient manual scanning; and #RescuePH cited to Olteanu, which no local record contains, re-cited to ermino2022evaluating. The auditor's Xin objection ("weights nearly constant" untraceable) was rejected on evidence: the Xin record's Figure 3 caption states "for the majority of the runs, the task weights barely move".

11. Reference infrastructure hazard (also from the audit). `references.bib` and `references.json` are generated from `papers/*/record.json` by `scripts/build-references.py`, so entries appended directly to the bib are silently wiped on rebuild, and Quarto renders the missing keys as a lone (?). Non-paper records now live as record.json files with citekey overrides (`papers/web-pagasa-cyclone/record.json` for pagasaCyclone, `papers/ph-ra-10173/record.json` for republic2012privacy); the Olteanu entry was dropped entirely because the hashtag claim re-cites to a local record. `verify_render.py` now detects the lone (?) and [?] patterns.

### Style mechanics

9. Sentence-length cap: the SOP general paragraph, the four specific problems and objectives, and several definition entries run 40 to 70 words. Split where the paired problem-objective template allows while keeping the four pairs parallel.
10. "Twenty typhoons" becomes "20 typhoons" (numerals for 10 and above); "five" stays spelled.
11. "Optimizers like AdamW" and "agencies like" become "such as".
12. Table 1 needs a general note beginning "Note." that defines nonstandard abbreviations in order of appearance (AGENTS 7.14, 7.15).
13. "Compare to" becomes "compare with" in measurement contexts; check each occurrence.
14. Manual bold-italic table and figure captions become Quarto `tbl-cap` and `fig-cap` at conversion, and caption text must not duplicate the narrative below the figures.
15. Confirm minus signs, en dashes in ranges, dropped leading zeros on r, p, and Delta-m, and percent symbols with numerals; the draft is mostly compliant.

## Cross-chapter consistency

1. Intent taxonomy: Chapter 1 commits to a 4-class intent set (Table 1 and the macro F1 definition). The Chapter 3 outline proposes five action-oriented groups mapped from "11 HumAID classes", while the HumAID record states ten class labels. Settle the count thesis-wide before Chapter 3's statistical treatment freezes.
2. Incident database: Chapter 1 says "SQLite or PostgreSQL"; Chapter 3 Subject 4 commits to PostgreSQL. Align one way.
3. Display numbering: Chapter 1's Table 1 and Figures 1 and 2 collide with Chapter 2's Table 1 under the pending thesis-wide renumbering decision (Chapter 2 open decision 7; Chapter 3 decision 6).
4. Recurring constants are consistent across 1.2, 1.3, 1.6, 1.8, and Chapter 3: seeds 42, 123, and 456; 12 paired event-seed runs; 4-fold LOEO; 30 timed iterations after 10 warm-up; batch sizes 1 and 32; span-level F1 of at least 0.70 agreement threshold.

## Open decisions for the group

1. Typhoon-frequency source: add PAGASA or CFE-DM as the record. Recommendation: PAGASA, the most specific agency for cyclone climatology.
2. Four unrecorded references: add records or replace citations. Recommendation: add all four; each carries content no CSV record covers.
3. Parameter-count parenthetical in 1.8: verify or drop. Recommendation: verify from model cards, since the counts support the commodity-hardware claim.
4. Intent class count (4 versus 5) and HumAID class count (10 versus 11): align with Chapter 3. Recommendation: match Chapter 1's 4-class design and correct the 11 to 10.
5. TLUNIFIED token count: drop unless confirmed against the PDF, mirroring the Chapter 2 resolution.
6. Commission or generate `ipo-diagram-1.png` and `ipo-diagram-2.png` before qmd conversion; the narratives in 1.5 already describe them.
