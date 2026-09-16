# Chapter 1–3 Writing Specifications

**Working title.** *Benchmarking Dynamic Multitask Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets*

**Sources of truth.** Content, variables, and constants are instantiated from `RESEARCH-DESIGN.md` (RD). Section skeleton from `OUTLINE.md` — **binding: all sections are mandatory** (department format, confirmed). Where content within a section conflicts with RD, RD wins (§G2 lists the four deliberate departures: K = 6 LODO, fold-mean inference, confirmatory-only H0/H1 framing, declared-or-rejected instruments). Nothing in this spec may contradict RD. Terminology governance: term usage syncs with `TERMINOLOGY.tsv`, and every citation the spec requires gets a `REFERENCES.tsv` → `manuscript/bibliography.bib` entry — additions here must propagate to both.

**Manuscript targets.** Chapter 1 → `manuscript/01-introduction.qmd`; Chapter 2 → `manuscript/02-literature.qmd`; Chapter 3 → `manuscript/03-methodology.qmd` (Quarto, APA via `apa.csl`).

**Audience.** The examining audience is a Philippine CS panel, not an international venue. Consequence: format is fixed by the outline; substance follows international Applied-NLP norms; and every statistical tool beyond a t-test is introduced in plain language before its formal statement (§1.11, §3.6).

**Reading a section spec.** Each section spec carries: *Scope & focus* (what the section is for and its boundaries), *Content* (explicit material to write), *Variables* (symbols/constructs that must appear), *Artifacts* (tables/figures/equations to produce, with stable IDs referenced in §G4).

---

## Chapter 1 — Introduction (`01-introduction.qmd`)

### 1.1 Title Page
- **Scope & focus.** Institution-formatted title page; title must read as experimental benchmarking + algorithmic optimization (defensible CS focus).
- **Content.** The working title verbatim. No subtitle additions.
- **Variables.** None.
- **Artifacts.** None.

### 1.2 Introduction to the Study
- **Scope & focus.** Executive narrative (1–1.5 pp): subfield → challenge → solution → contribution. No citations overload; 3–5 anchors.
- **Content.**
  1. Crisis informatics context: typhoon response in the Philippines; Twitter/X as real-time distress signal.
  2. The core challenge: joint triage of Taglish (Tagalog–English code-mixed) disaster tweets requires three tasks at once — NER (what/where), intent (what is needed), urgency (how fast) — under domain shift (each typhoon is a new distribution).
  3. The proposed approach: one multilingual encoder, hard parameter sharing, three heads; task balancing as the open question — uniform scalarization vs tuned static weights vs dynamic gradient surgery (PCGrad).
  4. Contribution statement (numbered): (i) a self-annotated Taglish common-corpus benchmark with LODO protocol; (ii) a pre-registered benchmark of dynamic vs static multitask balancing under domain divergence; (iii) an inference-efficiency profile of the unified model on commodity hardware, with edge deployment as future work; (iv) a gradient-conflict diagnostic linking optimization dynamics to per-task transfer.
- **Variables.** Informal first mention only: LODO, Macro-F1, negative transfer, PCGrad.
- **Artifacts.** None required.

### 1.3 Background of the Study
- **Scope & focus.** Technical case file structured by the Macro–Meso–Micro Funnel Model; strictly narrowing.
- **Content.**
  - **Macro (global crisis informatics):** disaster response + social media signal processing worldwide; why automated triage matters operationally (volume, latency of human triage).
  - **Meso (local crisis informatics + technical context):**
    - *Local context — laws, reports, statistics:* legal mandate — RA 10121 (Philippine DRRM Act of 2010) and the NDRRM Framework/Plan; government reports — COA Consolidated Report on the Audit of DRRM Funds (in `references/`; recurring low-utilization findings), NDRRMC situation reports; statistics — PAGASA cyclone frequency (~20 TCs enter PAR yearly, ~8–9 landfall), World Risk Index (PH #1 globally, 2024 score 46.91, 16th consecutive year), Typhoon Yolanda/Haiyan severity benchmark (6,293 dead; ₱571.1B damage/loss). All anchors registered with primary sources in §G5.
    - *Technical context:* local crisis-informatics literature; Taglish code-mixing and low-resource Filipino NLP; multilingual encoders for the region (SEA-LION); MTL with hard parameter sharing; the gradient-conflict/balancing literature (PCGrad through Xin et al. 2022, Kurin et al. 2022, Jiang et al. 2023); inference-efficiency constraints for practical deployment.
  - **Micro:** the specific gap — no LODO benchmark of dynamic vs static task balancing for joint Taglish triage, with inference-efficiency and per-task (not just aggregate) negative-transfer accounting.
- **Variables.** Code-mixing strata; gradient conflict cos(g_i, g_j); negative transfer NT_t (named, not yet formalized).
- **Artifacts.** None required (one funnel summary figure optional).

### 1.4 Statement of the Problem
- **Scope & focus.** One General Problem Paragraph (CS template), then exactly four interrogative questions, labeled RQ1–RQ4 per university format.
- **Content.**
  - General Problem Paragraph: synthesizes domain shift + code-mixing + task interference + deployment constraints into the single problem statement.
  - RQ1–RQ4 of RD restated interrogatively, in order (university label: RQ#, not SOP-Q#): (Q1) comparative OOD generalization aggregate and per stratum; (Q2) operational gains versus sequential (primary) and conditionally concurrent STL execution, and retention on available hardware; (Q3) extent to which PCGrad mitigates per-task NT vs uniform and tuned static scalarization; (Q4) how conflict frequency associates with forward changes in checkpoint-matched per-task F1 (ΔF1_t(c→c+1)) and whether it concentrates at the NER/sequence boundary.
- **Variables.** NT_t, ΔF1_t, cos(g_i,g_j) < 0 conflict definition, strata.
- **Artifacts.** None.

### 1.5 Objectives
- **Scope & focus.** Declarative objectives, one per SOP question, each tagged with its DSR study phase.
- **Content.** RO1–RO4 of RD, abbreviated to declarative form (full operational detail stays in Ch. 3). Phase tags per RD/DSRM mapping (Table 3.1): RO1 → Baseline Analysis + Empirical Benchmarking; RO2 → Empirical Benchmarking + Comparative Validation; RO3 → Design & Artifact Creation + Comparative Validation; RO4 → Design & Artifact Creation (diagnostic instrumentation) + Empirical Benchmarking.
- **Variables.** As in RQs.
- **Artifacts.** **Table 1.1** — Alignment matrix: RQ ↔ RO ↔ Hypotheses ↔ DSR phase ↔ primary metric. This is the chapter's load-bearing artifact; every later chapter inherits its row labels.

### 1.6 Hypothesis
- **Scope & focus.** Null/alternative pairs per template, but partitioned by evidential role exactly as RD declares — do not present descriptive assessments as significance tests.
- **Content.**
  - Confirmatory (H0/H1 stated formally): H1a aggregate superiority; H1b stratified positive transfer (bootstrap-CI form); H3a PCGrad > uniform on aggregate; H4a κ̄_NER,seq > κ̄_seq,seq.
  - Descriptive/secondary: H2a (≥50% footprint reduction, architectural expectation); H2b (bounded degradation, margin ε); H3b (non-superiority of PCGrad vs tuned static, margin δ_opt = 0.01, 90% CI); H4b (median lead–lag correlation < 0).
- **Variables.** μ_MTL − μ_STL; ΣNT identity caveat (state H3a on aggregate per RD); ε; δ_opt; κ.
- **Artifacts.** None (symbol forms deferred to Ch. 3 equations).

### 1.7 Theoretical Framework
- **Scope & focus.** Theories that explain why the design's relationships hold; each theory tied to a hypothesis family.
- **Content.** Inductive transfer / hard parameter sharing (grounds H1); multi-task gradient conflict theory (grounds H3/H4: conflict ≠ transfer, per Jiang et al. 2023 — state the competing hypothesis); code-switching linguistics operationalized via CMI (grounds H1b strata); distribution shift / OOD generalization (grounds LODO); non-inferiority/equivalence testing logic (grounds H2b/H3b margins).
- **Variables.** CMI; strata; cos conflict; ε/δ_opt as practical margins.
- **Artifacts.** **Fig. 1.1** — Theoretical model diagram (constructs → hypotheses).

### 1.8 Conceptual Framework
- **Scope & focus.** Two Input–Process–Output diagrams, per template.
- **Content.**
  - **Fig. 1.2 (Study IPO):** Input = self-annotated Taglish common corpus (K typhoon events, three label sets), LODO folds × S seeds, frozen constants, profiling machine. Process = six training arms (3 STL, uniform, tuned static, PCGrad), pre-registered evaluation protocol, gradient diagnostic logging, inference-efficiency profiling. Output = hypothesis verdicts, stratified transfer profile, optimization contrast, deployment recommendation.
  - **Fig. 1.3 (Software Artifact IPO):** ingestion → pre-processing/CMI assignment → shared multilingual encoder → three heads → orchestrated single-model inference serving (all-resident sequential comparison).
- **Variables.** K, S, T=3, arms enumerated.
- **Artifacts.** Fig. 1.2, Fig. 1.3.

### 1.9 Scope and Delimitations
- **Scope & focus.** Scope named explicitly (datasets, parameters, hardware); delimitations as deliberate boundary choices, not limitations.
- **Content.**
  - Scope: corpus (self-annotated, K = 6 typhoon events, Taglish, common corpus — each tweet carries all three label sets); tasks (NER span-level, Intent, Urgency classification); encoder (one shared SEA-LION-ModernBERT-300M across all six arms, task heads trained from scratch); arms (six); constants (S = 5 and C = 15–20 are RD adjustable defaults, frozen at launch; dev 15% stratified; δ_opt = 0.01; support n ≥ 25; mean-reduced losses; all-resident sequential baseline); hardware (commodity training/profiling machine — specs TBD per §G3; no edge device).
  - Delimitations (each phrased as a choice with reason): deployment estimand without decomposing parameter sharing from cross-task supervision; within-disaster dev leakage accepted (identical across arms); H2a/H2b/H4b/H3b carried as descriptive/non-inferiority assessments rather than confirmatory tests; no cross-hypothesis family-wise control; English-dominant stratum descriptive-only; concurrent execution conditional on no memory contention; N=K resolution acknowledged (wide CIs) rather than corrected; single-platform scope (X/Twitter) over disaster windows for K = 6 historical benchmark typhoon events (2020–2024) selected via pre-registered inclusion criteria — X/Twitter is the only platform satisfying all three corpus requirements (surviving 2020–2024 PH typhoon archive; research-legal release convention for annotated collections; comparable crisis-NLP benchmark literature); Facebook research tooling cannot reconstruct retrospective public-post windows and its terms bar resharing; open alternatives (Bluesky) postdate most of the roster — cross-platform generalization (Facebook, Bluesky, etc.) out of scope, future work; tweet mortality is a release-time concern (IDs + annotations only, §3.5 Step 9), not a collection concern; Taglish-only language scope by inclusion rule — non-Taglish Philippine disaster discourse (Bisaya, Ilocano, Bikol, Waray) is out of scope; CMI strata are interpreted within the inclusion screen's range-restricted distribution.
- **Variables.** All frozen constants of RD, by name and value.
- **Artifacts.** **Table 1.2** — Constants in scope (condensed; full table lives in Ch. 3 / RD).

### 1.10 Significance of the Study
- **Scope & focus.** Primary, secondary, tertiary beneficiaries, each with a concrete use of the outputs.
- **Content.** Primary: DRRM response coordinators (automated triage pre-screening — operationalizes the response pillar mandated by RA 10121; the COA low-utilization findings frame efficiency as an institutional concern, §G5); low-resource/code-mixed NLP researchers (LODO benchmark + corpus). Secondary: ML engineers deploying unified triage services (profiling methodology + footprint data); the regional NLP community (supervised-fine-tuning evidence on SEA-LION encoders for Taglish); future researchers (pre-registered protocol as template). Tertiary: affected communities and public-information officers (faster categorization).
- **Artifacts.** None.

### 1.11 Definitions of Terms
- **Scope & focus.** Operational definitions grounded in pipeline/math behavior (per template), not dictionary entries.
- **Content.** Each term: what it is in *this* system + where it operates: multi-task/single-task learning; hard parameter sharing; uniform scalarization (1:1:1, mean-reduced losses); tuned static scalarization (inner-fold weight search); PCGrad (projection rule, one line); LODO fold; seed; negative transfer NT_t; aggregate Macro-F1; span-level Macro-F1; Code-Mixing Index + strata; tokenizer fertility; conflict frequency κ_{i,j}; κ_t(c); lead–lag correlation; exact paired sign-flip permutation; Holm correction; cluster-paired bootstrap CI; non-superiority margin (practical-equivalence check); ε and δ_opt as frozen margins; collapse (operational: >95% single class on dev); all-resident vs load/unload.
- **Artifacts.** **Table 1.3** — Notation quick reference (subset of §G1 relevant to Ch. 1–3 prose).

---

## Chapter 2 — Literature Review (`02-literature.qmd`)

### 2.1 Crisis Informatics and Joint Triage Tasks
- **Scope & focus.** How the three tasks are formulated and measured in disaster NLP; structured **global → local**; empirical numbers over application summaries.
- **Content.**
  - **2.1.1 Global crisis informatics:** NER over disaster tweets (entity types, span-level F1 conventions); intent/information-type classification; urgency/priority detection; multilingual crisis-response benchmarks (CrisisNLP/HumAID lineage).
  - **2.1.2 Local crisis informatics (Philippines/Taglish):** Philippine typhoon-response and DRRM literature; legal and institutional basis (RA 10121, NDRRMC/LGU mandates) and government reports/statistics as documented anchors (§G5); existing Filipino/Taglish social-media NLP work; what exists locally vs what is missing.
  - For each cited work in both subsections: algorithm, dataset, and the explicit metrics achieved (F1 by class, latency where reported).
- **Variables.** Macro-F1 (both granularities); class inventories as comparison axis.
- **Artifacts.** None (feeds Table 2.2 rows).

### 2.2 Taglish Code-Mixing, Low-Resource NLP, and the SEA-LION Encoder
- **Scope & focus.** Why Taglish is hard for subword models, how it is measured, and why SEA-LION-ModernBERT-300M is the right encoder for this study.
- **Content.** Code-switching measurement (CMI formulations — pick and justify the one to freeze, cf. §G3); language identification incl. borrowing/cognate treatment (flag as open constant); tokenizer fertility as an explanatory variable. Encoder justification: **SEA-LION-ModernBERT-300M** (`aisingapore`, MIT license; base MLM-style pretrain — *not* the contrastive embedding variants; Gemma-3 SentencePiece, 262K vocab; Filipino + English in the pretraining mix at ~2.5% / ~35%; SEA-BED/SEA-HELM evaluations), contrasted against mBERT, DistilmBERT, and XLM-R base on Filipino coverage, parameter count, vocabulary/fertility, license, and tooling (`transformers ≥ 4.48`). Present the English-heavy/Filipino-light mix as motivation for the stratified (H1b) and MTL design, not a defect to hide.
- **Variables.** CMI; fertility (subwords/word); encoder parameter counts.
- **Artifacts.** **Table 2.1** — encoder justification matrix: SEA-LION-ModernBERT-300M vs mBERT/DistilmBERT/XLM-R base on Filipino coverage, parameters, vocab/fertility, license, tooling (mandatory).

### 2.3 Multi-Task Optimization and Balancing (incl. inference efficiency)
- **Scope & focus.** The algorithmic core: how balancing methods function, their computational complexity, and their empirical record.
- **Content.**
  - Static scalarization: L = Σ_t w_t·L_t; uniform vs tuned weighting; the tuning-budget confound.
  - Dynamic gradient surgery: PCGrad projection rule — **Eq. 2.1**: g_i ← g_i − (g_iᵀg_j / ‖g_j‖²)·g_j for cos(g_i,g_j) < 0 — plus its per-step cost (T backward passes) and memory overhead; brief CAGrad/MGDA for context.
  - The critical literature the hypotheses lean on: Xin et al. 2022 (MTO ≈ tuned static under equal budgets); Kurin et al. 2022 (unitary scalarization + regularization matches MTO); Jiang et al. 2023 (conflict does not imply negative transfer; near-orthogonal gradients); Gama & Grassi 2025 (uniform loss vs specialized optimization under matched budgets — recent extension of the debate).
  - Inference efficiency: footprint/latency of shared-encoder multi-head vs multiple STL models; measurement methodology (edge-specific studies cited as background, not tested claims).
- **Variables.** w_t; cos(g_i,g_j); κ; per-step overhead multiplier; peak memory; latency (ms).
- **Artifacts.** **Eq. 2.1** (PCGrad projection); optional **Eq. 2.2** (scalarization).

### 2.4 Algorithmic Literature Taxonomy Matrix
- **Scope & focus.** Structured comparison placing this study against key related work.
- **Content.** **Table 2.2** with columns: Author & Year | Algorithm/Model | Dataset/Testbed | Evaluated Metrics | Identified Limitation/Gap. Rows drawn from **both** the global (2.1.1) and local (2.1.2) literature: PCGrad origin; Xin et al.; Kurin et al.; Jiang et al.; two global crisis-NLP triage works; two Philippine/Taglish NLP works; one inference-efficiency/deployment work; final row = **This study** (fills: Taglish common-corpus joint triage, LODO, dynamic-vs-static balancing, per-task NT, efficiency profile, pre-registered inference).
- **Artifacts.** **Table 2.2** (mandatory).

### 2.5 Synthesis of Related Literature and Studies
- **Scope & focus.** 1–2 pages, three-step critical synthesis; no new citations.
- **Content.**
  - Step 1 Commonalities/Trends: MTL as standard for joint tasks; F1-centric evaluation; gradient conflict as the accepted interference account; encoders as the efficiency lever.
  - Step 2 Gaps/Bottlenecks: (i) balancing evaluated in-distribution, not under event-level domain shift; (ii) aggregate-only reporting masks per-task negative transfer; (iii) conflict↔transfer coupling assumed, not measured (contra Jiang et al.); (iv) code-mixed low-resource strata unexamined; (v) inference efficiency of the unified model unprofiled; (vi) tuning-budget asymmetries confound published MTO comparisons.
  - Step 3 Research Bridge: map each gap to the design element that addresses it (RQ/RO/hypothesis references), ending in the study's positioning statement.
- **Artifacts.** None.

---

## Chapter 3 — Methodology (`03-methodology.qmd`)

### 3.1 Research Methodology
- **Scope & focus.** Governing framework: DSRM 6-phase (Peffers et al.) with quantitative benchmarking as the empirical method inside Design & Development / Demonstration / Evaluation.
- **Content.** Name the artifact triple: (i) the self-annotated Taglish benchmark corpus, (ii) the unified MTL triage system + balancing arms, (iii) the pre-registered evaluation/benchmark protocol. One paragraph reconciling DSRM phases with the OUTLINE's four study phases: Baseline Analysis → problem identification/objectives; Design & Artifact Creation → design & development; Empirical Benchmarking → demonstration; Comparative Validation → evaluation; Communication → thesis defense/publication.
- **Artifacts.** **Table 3.1** — DSRM phase ↔ study phase ↔ activity ↔ output. **Fig. 3.1** — DSRM flow diagram.

### 3.2 Subjects of the Study
- **Scope & focus.** Computational subjects: the corpus and its annotation, not human respondents.
- **Content.**
  - Corpus definition: Taglish typhoon-event tweets, common corpus (every tweet carries all three label sets); K = 6 target events (legal minimum 5); events roster with dates. Corpus size target: N_corpus = 6,000–10,000 Taglish tweets **post-screen** (≈ 1,000–1,700 per event); the harvest quota exceeds N_corpus by a non-Taglish attrition buffer, audited on the historical harvest pool. **Inclusion rule (language, not geography):** a tweet enters the corpus iff it passes the frozen Taglish screen (LID + CMI bands, §G3), regardless of where the event struck — region is a sampling prior for roster selection, never a filter.
  - Collection (retrospective, historical): harvest frozen landfall windows (t₀−24h to t₀+72h; t₀ = official PAGASA landfall declaration) for the locked 2020–2024 event roster via academic archive sharing, research tweet-ID hydration, and targeted historical retrieval; event-specific query design (event name/hashtag, location and distress terms in Tagalog and English); deduplication; PII redaction policy. **Pre-lock acquisition pilot:** one event (Carina 2024) is harvested end-to-end — retrieval → hydration → Taglish screen → pass-rate audit — before the roster freeze; if official retrieval cannot deliver the harvest pool, the documented fallback is officially shared historical tweet-ID collections (e.g., Internet Archive / DocNow event sets), with third-party reseller APIs excluded on platform-ToS grounds. Live prospective streaming is recorded as **considered-and-rejected** due to operational dependency on weather timing and defense schedule risks.
  - Event-roster selection criteria (frozen before the K roster locks): (i) **recency** — 2020–2024 events only (tweet-ID survival, modern PH Twitter penetration); (ii) **salience and volume** — PAGASA-retired names / NDRRMC SitReps with major casualty/damage records as proxy — sufficient for stratum × class support ≥ 25 (target ≥ 1,000 post-screen per event); (iii) **language-composition audit** — skew toward Tagalog-belt events (Metro Manila, CALABARZON, Central Luzon) to protect H1b's tested strata (Tagalog-dominant, Mixed); exactly one VisMin / high-international-salience event (Odette 2021) as deliberately hardest OOD fold, with its expected English-dominant skew noted (empirical basis: Haiyan tweets were 87% English vs 11% Filipino — David, Ong, & Legara 2016, PLOS ONE — vs ~75% Filipino in Luzon-centric disaster studies); (iv) **cluster-overlap control** — at most 1–2 events per storm sequence (e.g., Nov 2020 sequence); (v) **hazard-type spread** (wind, flooding, surge, landslides). **Locked Primary Roster (K = 6):** (1) Ulysses 2020 (NCR/Central Luzon, flood/wind), (2) Rolly 2020 (Bicol/CALABARZON, super typhoon wind/surge), (3) Odette 2021 (VisMin, OOD English/regional skew), (4) Karding 2022 (Central Luzon, rapid intensification/wind/flood), (5) Egay 2023 (Northern Luzon, monsoon enhancement/flooding), (6) Carina 2024 (NCR/CALABARZON, severe urban flooding). **Pre-declared Alternates:** Kristine 2024 and Pepito 2024, activated in order if any primary event fails the ≥ 1,000 post-screen or support threshold.
  - Annotation scheme: NER entity-type inventory; Intent class inventory; Urgency class inventory (inventories are freeze-before-launch items, §G3).
  - Annotation protocol: annotator count and qualifications, guidelines document reference, adjudication procedure, annotation tooling (named and versioned); inter-annotator agreement (IAA), computed on a **pre-registered event-stratified subsample (target ≥ 20% of the corpus) independently re-annotated by a fixed panel of three of the five thesis authors, blind to the primary labels**: classification tasks — Fleiss' κ (three raters); NER — mean pairwise span-level F1 between panel annotators; reported per task and per stratum, with a **pre-specified minimum acceptable IAA threshold** (e.g., κ ≥ 0.6, span-F1 ≥ 0.75) that triggers guideline revision and re-annotation before adjudication. The corpus is annotated by the five thesis authors (no third-party annotators): no human-subjects consent protocol applies, RA 10173 coverage of tweet authors sits in §3.5 Step 1, and institutional REC determination (application or exemption) is documented in the manuscript front matter.
  - Post-hoc stratification: CMI assignment to English-dominant / Mixed / Tagalog-dominant; per-fold stratum composition (feeds the H1b audit). The screen-stratification circularity is acknowledged: the LID + CMI screen is simultaneously the inclusion rule, so the corpus CMI distribution is range-restricted by construction and strata are interpreted within that restricted range (restated as a delimitation, §1.9).
- **Variables.** T = 3; class inventories; K events; strata; IAA statistics.
- **Artifacts.** **Table 3.2** — Corpus statistics: per-event tweet counts across the 6 historical events (Ulysses 2020, Rolly 2020, Odette 2021, Karding 2022, Egay 2023, Carina 2024), **affected region**, strata distribution, per-task class support, with **fold × region and fold × stratum composition reported** (composition confounds handled per RD's audit rule: a stratum confounded with a single disaster → standardize within fold or move to descriptive). **Table 3.3** — Annotation scheme + IAA summary. **Fig. 3.2** — Annotation pipeline.

### 3.3 Sampling Technique
- **Scope & focus.** Leave-one-disaster-out as the sampling design; **overrides the template's "four-fold" wording** (see §G2, item 1).
- **Content.** K folds, one held-out typhoon event per fold; within each fold, 15% stratified dev split of training events (disaster × label), drawn once, identical across arms; S = 5 seeds per arm per fold; seed runs averaged within fold; unit of inference = fold-level paired observations (N = K); K ≥ 5 justification (p_min = 2^−K; at K = 4, p_min = 0.0625 > 0.05 — **Eq. 3.1**); guard clause on disaster exclusion.
- **Variables.** K, S, N = K, p_min, dev fraction.
- **Artifacts.** Eq. 3.1; **Fig. 3.3** — Fold construction diagram; **Table 3.4** — Fold roster (event → role across the 6 historical disasters).

### 3.4 Instruments to Be Used
- **Scope & focus.** Software profilers, benchmarking suites, measurement instruments — each mapped to its measurand and precision.
- **Content.** Training stack (PyTorch/Transformers — versions frozen at launch); per-task gradient extraction (side-effect-free, final shared layer); latency timer (high-resolution monotonic clock, e.g., `perf_counter_ns`, warm-up amortized, batch 1); memory instruments (framework-allocated peak, e.g., `cuda.max_memory_allocated`; system peak RSS via `psutil`/`/proc`); op-level profiling (PyTorch Profiler); state explicitly whether MLPerf/TensorRT are used or considered-and-rejected, with reason (profiling machine per §G3).
- **Variables.** Latency (ms, mean ± SD); peak RAM/VRAM (MB).
- **Artifacts.** **Table 3.5** — Instrument ↔ measurand ↔ precision ↔ tool version.

### 3.5 Procedure of Data Gathering
- **Scope & focus.** Chronological step-by-step protocol ending in automated log generation, per template.
- **Content.**
  1. **Data acquisition (retrospective historical):** Define official PAGASA/NDRRMC landfall timestamps for the K = 6 locked historical events → harvest the frozen window (t₀−24h → t₀+72h from the official landfall declaration t₀; first Philippine landfall for multi-landfall storms) via research archive queries and tweet hydration → dedup → **screen pass-rate audit** (audit Taglish attrition against the initial harvest pool) → PII redaction (compliance statement under the Data Privacy Act of 2012, RA 10173, §G5) → corpus freeze (manifest with hashes).
  2. **Pre-processing:** cleaning rules; tokenization; LID for CMI computation (LID treatment of borrowings/cognates: open constant, §G3); stratum assignment.
  3. **Annotation:** guideline training round → independent annotation → IAA computation → adjudication → final labels.
  4. **Fold construction:** LODO assignment; dev splits; pre-registered stratum × class support audit (unique-example counting; fold × stratum contingency table). Freeze ladder: (1) LID screen + CMI stratum thresholds freeze before annotation; (2) the support audit locks the H1b family after annotation, before any training; (3) ε locks after all K STL runs, before the first MTL run.
  5. **Execution protocol:** all K STL runs first (ε freeze: 1.5 × pooled STL seed-SD) → MTL arms (uniform, tuned static with inner-fold weight search, PCGrad) → gradient logging in the uniform arm (C = 15–20 windows on matched fractions of realized optimizer steps); run-integrity policy (NaN/collapse re-run once; collapse = >95% single class on dev).
  6. **Evaluation:** OOD per-checkpoint F1; pooled out-of-fold stratified predictions (one prediction per example).
  7. **Inference-efficiency profiling:** warm-up runs, N timed trials, batch-1 all-three-tasks workload; all-resident sequential (primary), concurrent (conditional), load/unload (descriptive).
  8. **Logging:** automated CSV/JSON emission per run and per trial.
  9. **Release packaging (post-defense):** the corpus is released as **tweet IDs + annotations only** — never tweet content — following the platform-ToS dehydrate/hydrate convention standard for Twitter/X datasets, together with a Datasheet for Datasets; the unified model is released with a model card; code, configs, seeds, and run manifests are released under open licenses for reproducibility.
- **Variables.** ε; C; collapse threshold; warm-up/trial counts (freeze at launch).
- **Artifacts.** **Fig. 3.4** — Procedure flowchart; **Table 3.6** — Log schema (file → fields → unit).

### 3.6 Statistical Treatment
- **Scope & focus.** The pre-registered inferential machinery; must record which conventional tests were considered and why they are not used at this sample size (template requirement).
- **Content.**
  - Presentation rule: every test gets a one-sentence plain-language explanation before its formal statement, and the rigor is anchored with significance-practice citations (Dror et al. 2018; Dodge et al. 2019; Card et al. 2020) — the machinery is above local thesis norms and must be self-explaining to a general CS panel.
  - Unit of inference: fold means; N = K = 6 (minimum 5).
  - Primary: exact paired sign-flip permutation, one-sided α = 0.05 (H1a, H3a, H4a).
  - Robustness: Wilcoxon signed-rank (reported, never primary).
  - H1b: one-sided cluster-paired bootstrap CIs (resample K fold IDs once per replicate, recompute both arms), Holm across two strata (97.5%/95% bounds); collapsed-family rule; sign-flip robustness check.
  - H3b: 90% two-sided CI non-superiority vs δ_opt = 0.01; TOST-style equivalence extension; K = 6 power note (half-width ≈ 0.82 × SD_diff).
  - H2b: descriptive bound ΔF1_t ≥ −ε with 95% CIs displayed.
  - Conventional tests considered and rejected: paired t-test (normality/SD stability untenable at K = 6); Wilcoxon as primary (exact permutation preferred at identical resolution); pooled-over-observations tests (ignore fold clustering); mixed/hierarchical models (not estimable at K = 6 with pre-registered primaries — acknowledged as limitation).
  - Explicit H0/H1 pairs for every confirmatory hypothesis (from RD).
- **Variables.** α; p_min; δ_opt; ε; SD_diff.
- **Artifacts.** **Eq. 3.2–3.5** (sign-flip statistic; NT_t; ΣNT identity; TOST bounds). **Table 3.7** — Hypothesis ↔ test ↔ statistic ↔ α ↔ decision rule.

### 3.7 Data Analysis
- **Scope & focus.** Every quantitative formula the results chapter will compute; per template, explicit CS performance formulas.
- **Content.** Classification Macro-F1; span-level Macro-F1 (entity types, exact span match); aggregate = unweighted mean over T; NT_t = F1_STL,t − F1_MTL,t; ΔF1_t(c→c+1) checkpoint-matched forward changes; ΔMTL and the ΣNT identity (restate **Eq. 3.6** from RD RO3); κ_{i,j} and κ_t(c); lead–lag Spearman (κ_t(c) vs ΔF1_t(c→c+1)); latency mean ± SD; peak memory; ε computation; tokenizer fertility (subwords/word) by stratum.
- **Variables.** All of §G1.
- **Artifacts.** **Table 3.8** — Metric registry (metric ↔ formula ↔ granularity ↔ hypothesis served).

### 3.8 Software System Description
- **3.8.1 System Structure.** Block architecture: ingestion → pre-processing/CMI → shared **SEA-LION-ModernBERT-300M** encoder (single checkpoint across all arms, frozen hash in the run manifest) → NER head (token classification) + Intent head + Urgency head (sequence classification); training orchestrator (six arms, seed manifest); serving orchestrator (single model, all-resident vs sequential comparison). **Artifacts:** **Fig. 3.5** system block diagram; **Fig. 3.6** neural architecture diagram with parameter counts (note the 262K-vocab embedding matrix's share of parameters and memory).
- **3.8.2 Use Cases.** Actors: response analyst (query triage output), system operator (run arms/profiles), MLOps worker (train/evaluate/log). **Artifacts:** **Fig. 3.7** UML use-case diagram; **Table 3.9** use-case specifications (actor, precondition, main flow, exception).
- **3.8.3 Materials and Equipment.** OS, Python/framework versions (transformers ≥ 4.48 required for ModernBERT), CUDA version, training GPU, and the profiling machine's full specification (same node or a second commodity machine, §G3) with VRAM bounds, power mode, and inference precision (frozen and identical across all profiled arms). Edge deployment is declared out of scope (future work). Also report the **compute budget** per responsible-NLP conventions (ARR Responsible NLP Research Checklist): run count (6 arms × K folds × S seeds ≈ 180 final runs at K = 6, S = 5, plus one-seed-per-point grid-search runs per arm-fold and the tuned arm's inner weight search), GPU type, estimated training/inference hours, and total resources needed to reproduce. **Artifacts:** **Table 3.10** — Environment specification (training node + profiling machine).

### 3.9 Testing and Evaluation
- **Scope & focus.** PoC verification of the artifact before benchmarking; stress behavior; standards mapping where applicable.
- **Content.**
  - Unit tests: tokenizer round-trip; loss-normalization scale parity across arms (mean-reduction invariance check); head output shapes; CMI implementation vs hand-computed fixtures; pooled out-of-fold construction (each example exactly once).
  - Integration tests: end-to-end train → checkpoint → OOD evaluate → log emit; gradient-logging side-effect-freeness (update equality with/without logging on a fixed batch).
  - Empirical stress benchmarking: latency across batch sizes (1 primary, curve descriptive); thermal behavior check on the profiling machine; memory-leak check across sustained trials.
  - ISO/IEC 25010 mapping (applicable subset): performance efficiency, reliability, portability — one characteristic-to-evidence row each; state non-applicable characteristics explicitly.
- **Artifacts.** **Table 3.11** — Test plan (test ↔ level ↔ pass criterion); **Table 3.12** — ISO/IEC 25010 mapping.

---

## G1. Notation registry (single source for all chapters)

| Symbol | Meaning | Defined in |
|---|---|---|
| T = 3 | Tasks: NER, Intent, Urgency | RD constants |
| K | LODO folds = held-out disasters (6 target, ≥ 5) | RD unit of inference |
| S = 5 | Seeds per arm per fold | RD constants |
| N = K | Unit of inference: fold-level paired observations | RD |
| N_corpus | Corpus size (Taglish tweets, post-screen; target 6,000–10,000) | §3.2 |
| t₀ | Official landfall declaration time; harvest window t₀−24h → t₀+72h | §3.5 Step 1 |
| C = 15–20 | Checkpoint windows on realized optimizer steps | RD RO4 |
| F1_t | Per-task Macro-F1 (span-level for NER) | RD aggregate metric |
| agg | Aggregate Macro-F1 = unweighted mean over T | RD |
| NT_t | F1_STL,t − F1_MTL,t | RD RQ3 |
| ΣNT, ΔMTL | Aggregate NT and mean gain; ΔMTL = −(1/T)ΣNT; ΣNT arm-difference = T × aggregate F1 difference | RD RO3 |
| ΔF1_t(c→c+1) | Forward change in per-task F1 between checkpoint windows (windows matched by fraction of realized optimizer steps) | RD RO4 |
| cos(g_i,g_j) | Pairwise gradient cosine, final shared layer | RD RQ4 |
| κ_{i,j}, κ_t(c) | Conflict frequency (window); task-mean conflict | RD RO4 |
| ε | 1.5 × pooled STL seed-SD (all K STL folds) | RD H2b |
| δ_opt = 0.01 | Non-superiority margin (aggregate Macro-F1) | RD H3b |
| p_min = 2^−K | Minimum one-sided p, exact paired tests | RD unit of inference |
| CMI, strata | Code-Mixing Index; EN-dominant / Mixed / TL-dominant | RD RO1 |
| fertility | Subwords per word, by stratum | RD RO1 |
| support n ≥ 25 | Unique examples per stratum × class | RD H1b |

## G2. Template ↔ design conflicts (resolved in RD's favor)

Quoted wording below is the pre-neutralization OUTLINE.md text; OUTLINE.md was neutralized on 2026-09-16 to remove the conflicting prescriptions, and this list remains as the audit trail of each override.

1. **OUTLINE 3.3 "four-fold leave-one-event-out"** → overridden: K = 6 folds (minimum 5). Four folds make p_min = 0.0625 > 0.05, failing RD's own guard; 3.3 must state this explicitly.
2. **OUTLINE 3.6 "paired comparison on per-seed deltas"** → overridden: seeds are averaged within fold; inference is on fold means (N = K), per RD. Per-seed variation feeds ε and descriptive SDs only.
3. **OUTLINE 1.6 "hypothesis: null and alternative"** → satisfied by the confirmatory set only (H1a, H1b, H3a, H4a); H2a/H2b/H3b/H4b are labeled descriptive/non-inferiority assessments, never significance tests.
4. **OUTLINE 3.4 instrument list (MLPerf, TensorRT)** → treated as declared-or-rejected with reasons, not assumed usage.

## G3. Freeze-before-launch registry (TBD slots the chapters must carry visibly)

| Item | Where it lands | Note |
|---|---|---|
| Encoder size variant | 2.2 justification; 3.8.1; Table 3.10 | **Chosen: SEA-LION-ModernBERT-300M base** (`aisingapore`, MIT) — *not* the ModernBERT-Embedding / E5 contrastive variants. 600M fallback only via the pre-lock pilot gate; pin checkpoint hash in run manifest |
| Encoder pilot gate (pre-lock) | 3.5 launch checklist | (a) tokenizer fertility probe: ≥500 sampled tweets per stratum, SEA-LION Gemma-3 tokenizer vs one multilingual baseline; (b) three short STL pilot fine-tunes (one per task) — convergence, no collapse per RD definition; (c) footprint probe: peak VRAM/RAM + step time on training node, batch-1 latency + peak memory on the candidate profiling machine |
| Profiling machine, precision, power mode | 3.8.3, 1.9 | Resolved: commodity workstation, no edge device; precision frozen and identical across arms; specs recorded in Table 3.10 |
| Data acquisition route | 3.2, 3.5 Step 1 | **Resolved: retrospective historical collection (2020–2024 pool)**. Primary roster locked at K = 6 (Ulysses, Rolly, Odette, Karding, Egay, Carina) + 2 alternates (Kristine, Pepito). Frozen: t₀−24h to t₀+72h landfall window; screen pass-rate audit; **pre-lock one-event acquisition pilot (retrieval → hydration → screen) gates the roster freeze**; fallback = officially shared tweet-ID collections (reseller APIs excluded, ToS) |
| Landfall collection window | 3.5 Step 1 | t₀−24h → t₀+72h from the official PAGASA landfall declaration t₀; first Philippine landfall for multi-landfall storms — frozen |
| CMI formula + stratum thresholds | 3.5 Step 2, 2.2, 3.2 | Choose a standard formulation; thresholds frozen **before annotation** (freeze ladder, §3.5 Step 4) |
| LID treatment of borrowings/cognates | 3.5 Step 2, 2.2, 3.2 | Affects stratum assignment **and corpus inclusion** (the LID + CMI screen *is* the inclusion rule, §3.2); frozen before annotation begins |
| Class inventories (NER/Intent/Urgency) | 3.2, Table 3.3 | Freeze with annotation guidelines |
| Minimum IAA acceptance threshold | 3.2 | Pre-specify before annotation; below-threshold agreement triggers guideline revision + re-annotation |
| Class-imbalance / sampling policy | 3.5 execution; RD parity | Any class weighting, oversampling, or decision thresholding must be frozen and applied **identically across all six arms** — otherwise the 1:1:1 uniform-weighting estimand is broken |
| Corpus / code release licenses | 3.5 Step 9 | Tweet IDs + annotations only (not content); open-license choice (e.g., CC BY 4.0 for data, MIT/Apache-2.0 for code) |
| Instrument/framework versions; warm-up/trial counts; gradient-logging subsample | 3.4, 3.5 | Record at launch in run manifest |

## G4. Artifact index

- **Ch. 1:** Table 1.1 alignment matrix; Fig. 1.1 theoretical model; Fig. 1.2 study IPO; Fig. 1.3 software IPO; Table 1.2 constants in scope; Table 1.3 notation.
- **Ch. 2:** Eq. 2.1 PCGrad projection (Eq. 2.2 scalarization optional); Table 2.1 encoder justification matrix (mandatory); Table 2.2 taxonomy matrix, global + local rows (mandatory).
- **Ch. 3:** Table 3.1 DSRM mapping; Fig. 3.1 DSRM flow; Tables 3.2–3.3 corpus + annotation; Eq. 3.1 p_min; Fig. 3.2 annotation pipeline; Fig. 3.3 fold construction; Table 3.4 fold roster; Table 3.5 instruments; Fig. 3.4 procedure flowchart; Table 3.6 log schema; Eqs. 3.2–3.5 statistics; Table 3.7 hypothesis↔test map; Eq. 3.6 ΣNT identity; Table 3.8 metric registry; Fig. 3.5 system blocks; Fig. 3.6 neural architecture; Fig. 3.7 use cases; Table 3.9 use-case specs; Table 3.10 environment; Table 3.11 test plan; Table 3.12 ISO 25010.
- **Appendices** (targets `manuscript/appendix-a.qmd` et seq.): A — annotation guidelines; B — ε computation (promised by RD); C — log schema; D — run manifest template; E — corpus datasheet (**incl. collection windows + per-event screen pass-rate + a current-API hydration-access note for downstream users**) + model card.

## G5. Legal, government-report, and statistical anchors (verified)

All figures verified 2026-09; re-verify currency at manuscript-writing time. Cite primary sources (official gazettes, agency portals), not news aggregators; each anchor gets a `bibliography.bib` entry, using APA legal-reference format for statutes and government reports (e.g., *Republic Act No. 10121*, 2010; Commission on Audit, 2024). Statistics appear only as framing in Chapters 1–2 — never as experimental variables; legal anchors are cited for mandate and ethics, never construed as endorsing the artifact.

| Anchor | Verified content | Where used | Primary source |
|---|---|---|---|
| RA 10121 — Philippine DRRM Act of 2010 | Legal basis of NDRRMC/OCD and the four DRRM pillars incl. disaster response; NDRRMF/LDRRMF | 1.3, 1.10, 2.1.2 | Official Gazette |
| RA 10173 — Data Privacy Act of 2012 | Governs personal data handling; legal basis of the corpus PII-redaction policy | 3.2, 3.5 Step 1 | Official Gazette / NPC |
| NDRRM Framework / National DRRM Plan | Policy articulation of response objectives | 1.3, 2.1.2 | NDRRMC |
| COA Consolidated Report on the Audit of DRRM Funds (2024; already in `references/`) | Recurring low-utilization findings (e.g., 70 LGUs flagged in 2024) — frames efficiency as an institutional concern | 1.3, 1.10 | coa.gov.ph |
| NDRRMC situation reports (SitReps) | Per-event operational reporting; statistics source for the event roster (3.2) | 1.3, 3.2 | NDRRMC / ReliefWeb |
| PAGASA tropical cyclone statistics | ~20 TCs/year enter the PAR; ~8–9 landfall; peak season Jul–Oct | 1.3 | PAGASA-DOST |
| World Risk Index | PH ranked #1 most at-risk: 2024 score 46.91 (16th consecutive year per CPBRD); 2025 #1 (reported 46.56 — verify score against WRR 2025 primary at writing time) | 1.3 | WorldRiskReport; CPBRD FF2024-68 |
| Typhoon Yolanda/Haiyan (2013) severity benchmark | 6,293 dead, 28,689 injured, 1,061 missing (NDRRMC SitRep 108, Apr 2014); ₱571.1B / USD ~12.9B damage-and-loss estimate (NEDA, *Reconstruction Assistance on Yolanda*, 2013; cited secondhand by the World Bank) | 1.3 | NDRRMC SitRep 108; NEDA RAY (2013) |
