# Chapter 2 Outline (v2)

Operating plan for Chapter 2: Review of Related Literature and Studies. Sources: `guidelines.md` (CEU template), `docs/revised-proposal.md`, the Chapter 1 revision skeleton, the 55 records in `literature/primary-literature.csv`, and the paper texts under `papers/{paper_id}/paper.md`. This file carries only the plan and its state.

## Status (state table; a unit is delivered only when all four columns are yes)

| Unit | Drafted | Standardized | Claim-audited | Group-reviewed |
|---|---|---|---|---|
| Opening | yes | yes | yes | partial (roadmap revised 2026-09-10 on group direction; re-read owed) |
| 2.1 (all four subsections) | yes (fresh rewrite 2026-09-10, plan-faithful) | yes | yes (self-audit against paper records; the group banned subagent audits) | no (batch delivered, read pending) |
| 2.2–2.5 | no | – | – | – |

Decision recorded: the trimmed file's editor-history backup was not used, on group direction; Section 2.1 was rewritten fresh from the paragraph plans, the current draft.qmd as raw material, and direct statistic verification against papers/{id}/paper.md. Every statistic in the batch carries a paper anchor recorded in the defect register below. All of Section 2.1 now sits in revised-draft.qmd as one batch for a single full group read.

Standardized means passed through the drafting checklist developed during the Section 2.1 review: claims cited at the claim, statistics traced to source with context, one point per paragraph, given-new handoffs, block structure.

## Content plan

Opening (done): purpose with the three areas and the bridging question, roadmap with logical verbs, sourcing protocol with scoped recency claims.

### 2.1 Joint Multi-Task Modeling and Computational Efficiency in Crisis Informatics (18 papers)

- 2.1.1 Evolution of crisis informatics corpora and models (6 papers: Imran; CrisisBench; HumAID; CrisisTransformers; CrisisSense-LLM; Zguir). What the subsection tells in one sentence: the pipeline matured along two tracks — corpora grew cleaner and larger, models shifted from classical ML to fine-tuned encoders to cloud-scale LLMs — but nothing pairs token-level entity spans with concurrent intent and urgency labels, and nothing reports deployment cost. Four papers establish, two contrast.

  Paragraph plan:

  | ¶ | Job | Tells | Evidence | Hands off |
  |---|---|---|---|---|
  | 1 | Origin — both tracks begin | Domain is tractable: tweets are annotatable and classifiable | 52M/19/~50K; agency-backed schemes; classical AUC ≥ .80 with definition; rare-class weakness; OOV noise | Noise thread to 2.3; imbalance thread to ¶4 caveat; corpus lineage to ¶2 |
  | 2 | Corpus block: consolidation | Fragmentation made results incomparable; consolidation fixed it and paid off | 8 datasets merged; 166,100/141,500; dedup prevents leakage; merged-data generalization payoff | Scale story to ¶3 |
  | 3 | Corpus block: scale and gates | Construction reached largest scale under measured quality control | Largest-corpus claim; 77,196; 19 events; 66% gate; ~70% yield | Quality-gate precedent for the study protocol; benchmarks to ¶4 |
  | 4 | Model block: the takeover | Encoders displaced classical models on standardized benchmarks, with three design premises attached | .872 vs .829; .781/.777; classical <.72; multilingual parity; DistilBERT deployment; two grouped cautions | Backbone, encoder-selection, and deployment premises to ¶7 and the study design |
  | 5 | Model block: domain pretraining | The frontier specialized; domain-adapted encoders win at cluster cost | 15B tokens; 30+ events; all 18 datasets; +17.43%; 6×A100 | Scale-gap evidence to ¶7; domain-adaptation support for the study corpus |
  | 6 | Model block: the LLM era | The generative frontier approaches triage but stays cloud-bound and thinly evaluated | 63.8% vs 4.27% zero-shot; 16×A100/3×A40; hierarchy + QSF; 1,346 GPT-4o + 300 manual | The contrast justifying encoder-only local scope, to ¶7 |
  | 7 | Gap close | Document-level corpora, cluster/cloud systems, no token+sequence pairing, no latency or memory — the void the study fills | Claims, no numbers | Hands the two-track gap to the 2.1.2 thread and the 2.5 synthesis |
- 2.1.2 Joint modeling of token-level and sequence-level triage tasks (5): caruana1997multitask, baxter2000model, weld2022survey, wang2021transformerbased, seeberger2022enhancing. Arc: theory foundation, dialogue precedent, crisis MTL precedents with fixed equal weighting (verified: Wang's lambda = 0.5), gap close on the concurrent pairing.

  Paragraph plan:

  | ¶ | Job | Tells | Evidence | Hands off |
  |---|---|---|---|---|
  | 1 | Theory foundation | Joint modeling improves generalization through shared representations | Caruana inductive transfer; Baxter bias learning across task environments | Precedent to ¶2 |
  | 2 | Dialogue precedent | Token+sequence pairing is standard in spoken-language NLU | Weld survey: intent detection + slot filling as one task pair | Shape parallel to ¶3 |
  | 3 | Crisis precedents | Crisis MTL exists but under fixed, unexamined optimization | Wang on TREC-IS, loss weight fixed at 0.5; Seeberger E-MLM combined design, up to 10% F1 on actionable types | Cross-event thread to 2.1.3 |
  | 4 | Gap close | No crisis study trains entity extraction concurrently with intent and urgency | Claim, no numbers | Optimization question to 2.2 |
- 2.1.3 Cross-event and cross-lingual generalization (3): raychowdhury2020crosslingual, sarioglukayi2020detecting, seeberger2025generalizing. Arc: held-out event evidence, low-resource urgency transfer, spurious-cue debiasing, LOEO adoption.

  Paragraph plan:

  | ¶ | Job | Tells | Evidence | Hands off |
  |---|---|---|---|---|
  | 1 | Held-out events work | Models generalize to unseen disasters when trained for it | Ray Chowdhury: 3 held-out event test sets; 79.36 F1 Philippines Flood (English-only); zero-shot Italian 75.44 / Spanish 85.26; C&A weakest at 70.3 on multilingual mixed | Transfer cost to ¶2 |
  | 2 | Transfer pays a drop | Low-resource targets lose ground without in-language training | Sarioglu Kayi: ensemble 76.5 F1 English vs 63.5 Sinhala / 62.6 Odia; annotator conflation of updates and help requests | In-language motivation to 2.3/2.5 |
  | 3 | Generalization as design target | Spurious cues explain failures; unseen-event testing becomes protocol | Seeberger 2025: causal debiasing, domain-specific experts, up to +1.9% F1 over baselines, three tasks, temporal splits | LOEO definition and adoption close |
- 2.1.4 Complexity analysis, statistical rigor, and design science methodology (4): lin2023libmtl, reimers2017reporting, demsar2006statistical, peffers2007design. Arc: constant-inference-cost derivation (uncited, ours), seed variance, Friedman-Iman-Davenport-Nemenyi protocol, DSRM framing, standards gap close.

  Paragraph plan:

  | ¶ | Job | Tells | Evidence | Hands off |
  |---|---|---|---|---|
  | 1 | Cost axis + feasibility | Shared encoders make latency and memory measurable; the algorithm landscape is benchmarkable | Derivation (ours, uncited); LibMTL: 13 optimization strategies, 8 architectures, one PyTorch framework | Method detail to 2.2 |
  | 2 | Seed variance | Single scores mislead; multi-seed is the standard | Reimers: max differences 2.59 F1 (CoNLL-2003 NER) and 8.23 (ACE 2005); dev-test Spearman rho = .229 | Protocol to ¶3 |
  | 3 | Non-parametric protocol | Cross-dataset comparisons violate parametric assumptions | Demšar: Friedman + post hoc Nemenyi, Iman-Davenport correction | Adoption close |
  | 4 | Study form + standards close | The study is design science; corpus and prototype are the artifacts; no crisis study combines the standards | Peffers six activities; claim close | Gap assembly to 2.5 |

### 2.2 Multi-Task Optimization Dynamics (16 papers)

Arc: gradient surgery family, dynamic loss-weighting family, the audits questioning both, the optimizer confound. Depth tiers to control effort: the seven benchmarked methods (PCGrad, CAGrad, IMTL-G, Nash-MTL, Uncertainty Weighting, GradNorm, FAMO) and the four audits (Kurin, Xin, Elich, Gama) get full metric treatment; MGDA-UB, Aligned-MTL, SAM-GS get one or two context sentences; Loshchilov (AdamW) gets one sentence at first optimizer mention. Subsection gap close: the takeover-versus-audit dispute is unresolved and untested on code-switched text. Liu disambiguation: verify citeproc output at the first render citing two Lius; hand-write initials only if citeproc fails.

### 2.3 Code-Switched Text Processing and Low-Resource Philippine NLP (21 papers)

Arc: challenges and measurement (Winata, Adoptante, Petrov, Gamback for the Code-Mixing Index), multilingual encoder lineage in one chronological block (Devlin, Sanh, Conneau, Chung, Clark, Zhang, Liang, Marone, AI Singapore — one or two sentences each, depth only for the four candidate encoders), multi-task evidence on code-switched text (Adouane deep: the negative-transfer warning), Philippine resources (deep for Cruz, Miranda a and b, Africa, Montalan, Herrera, Ermino), gap close: no concurrent triage annotations on Taglish.

### 2.4 Algorithmic Literature Taxonomy Matrix

Table 1, CEU columns (Author and Year, Algorithm or Model, Dataset/Testbed, Evaluated Metrics, Identified Limitation). Four thematic row clusters plus the proposed study row last. Every cell passes the same tracing rule as prose; cells inherit the defect corrections already applied in prose.

### 2.5 Synthesis of Related Literature and Studies

Three steps in three paragraphs. Step 1, trends, cites CrisisBench .872 and HumAID .781 as converging encoder evidence and carries the weighted-versus-macro F1 bridge. Step 2, four typed gaps assembled from the subsection closes: evidence gap (the MTO dispute untested on code-switched text), methodological gap (no gradient diagnostics in applied crisis MTL), contextual gap (no concurrent Taglish triage corpus), practical gap (no latency or memory measurement). Step 3 maps the study's four contributions one-to-one onto the gaps and closes with the design science artifact.

## Coverage

All 55 primary papers allocated: 2.1 holds 18, 2.2 holds 16, 2.3 holds 21. Depth tiers replace uniform treatment: deep for benchmarked methods, audits, corpus-adjacent resources, and domain precedents; one or two sentences for context papers. Recency framing stays scoped (domain sources mostly last five years; older papers retained as definers of algorithms, encoders, protocol, or corpora); no source totals in chapter prose. Bib cleanup pending before submission: several entries link arXiv where published versions exist (AGENTS 8.5).

## Fixed decisions

Three topical sections with the matrix at 2.4 and synthesis at 2.5. No source totals declared in the chapter; the 55-total and 35-from-2021-to-2026 counts live here only. Liu initials via citeproc, tested not assumed. Year-letter behavior verified in rendered output at first double-citation of a surname. Table numbering deferred to the thesis-wide renumber. Title page fields (course, professor, due date) pending from the group.

Roadmap decisions (2026-09-10, group-approved): the roadmap names every subsection arc, including cross-event generalization, the optimizer confound, and multi-task learning on code-switched text. Chapter prose says "controlled comparisons" for the skeptical MTO studies, never "audits", which stays internal shorthand. The baseline reads "tuned static linear scalarization". The matrix name is capitalized. The roadmap names the tasks as entity, intent, and urgency; 2.1.2 restores the token-level and sequence-level granularity. The opening owns the abbreviation definitions: "multitask optimization (MTO)" is defined in the first sentence, and every later use, including in the 2.1 batch, is bare "MTO" with any body-text definition demoted (6.25). The bridging question and the 2.2 audit question both say "specialized balancing" (4.4); "specialized optimization" is retired.

Defect register, 2.1 rewrite (2026-09-10): the trimmed file's backup exists in editor history but the group directed a fresh rewrite from current sources; nothing was restored. Verification corrections against paper records: CrisisBench corpora are 166,098 and 141,533 tweets (old 166,100/141,500 approximations dropped); HumAID's 70% is the gold-question assignment gate, not a tweet yield, so the "roughly 70% kept" sentence is dropped in favor of 77,196 annotated to 76,466 after cross-event near-duplicate removal; Reimers' "dozens of seeds" and "1.01 F1" replaced by the seed-difference framing (max 2.59 CoNLL-2003 NER, 8.23 ACE 2005); Ray Chowdhury's test sets are three held-out events plus one general mixed set, 79.36 is English-only F1, and 70.3 (caution and advice) sits on the multilingual mixed set; Sarioglu Kayi anchors moved to the ensemble table (76.5 English, 63.5 Sinhala, 62.6 Odia); the Chen 2018 token-versus-sequence premise could not be anchored and the 2.1.2 close now rests on the Weld pairing precedent; LibMTL is cited for 13 optimization strategies and 8 architectures, replacing the unsupported per-task-gradient claim. CrisisSense-LLM supplies 63.8% versus 4.27% and the A100/A40 counts; Zguir supplies the taxonomy, QSF, 1,346 synthetic examples, and 300 hand-annotated tweets.
