# Chapter 2 Outline (v2)

Operating plan for Chapter 2: Review of Related Literature and Studies. Sources: `guidelines.md` (CEU template), `docs/revised-proposal.md`, the Chapter 1 revision skeleton, the 55 records in `literature/primary-literature.csv`, and the paper texts under `papers/{paper_id}/paper.md`. This file carries only the plan and its state.

## Status (state table; a unit is delivered only when all four columns are yes)

| Unit | Drafted | Standardized | Claim-audited | Group-reviewed |
|---|---|---|---|---|
| Opening | yes | yes | yes | partial (roadmap revised 2026-09-10 on group direction; re-read owed) |
| 2.1 (all four subsections) | yes (fresh rewrite 2026-09-10, plan-faithful) | yes | yes (self-audit against paper records; the group banned subagent audits) | partial (group review in progress; three findings resolved, corrections applied) |
| 2.2 | yes (2026-09-10, seven paragraphs) | yes | yes (anchors verified against records; 34 reference entries confirmed) | yes (group review complete; 27 findings resolved, corrections applied) |
| 2.3–2.5 | no | – | – | – |

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

Membership resolved (2026-09-10): the sixteen are the seven methods, the four audits, the three context methods (MGDA-UB, Aligned-MTL, SAM-GS), Loshchilov's AdamW, and Liu et al. (2019), the architecture-side attention response, which supplies one context sentence and the framing point that the benchmark holds the backbone fixed while varying optimization. Caruana and LibMTL sit in the MTO pillar but are spent in 2.1.

Paragraph plan:

| ¶ | Job | Tells | Evidence (verify at write time) | Hands off |
|---|---|---|---|---|
| 1 | Problem framing | One shared network, competing task losses; two algorithmic answers, one architecture answer outside the benchmark's scope | Sener and Koltun: multi-objective framing, MGDA min-norm solver (context tier); Liu 2019 attention as the architecture alternative | Surgery to ¶2–3, weighting to ¶4, architecture premise to the study design |
| 2 | Surgery origin | PCGrad names the problem and the fix: project conflicting gradients | Yu et al. 2020: conflict definition (negative cosine), projection rule, headline results; quote "gradient surgery" at first occurrence (6.7) | Refinements to ¶3 |
| 3 | Surgery refinements | The family diversified along guarantees, impartiality, and game theory | CAGrad common-descent direction with worst-case guarantee; IMTL-G impartial aggregation; Nash-MTL bargaining solution; context: Aligned-MTL, SAM-GS | Weighting to ¶4 |
| 4 | Dynamic loss weighting | Weights learned or adapted instead of gradients edited | Kendall homoscedastic-uncertainty weights; GradNorm gradient-norm balancing; FAMO loss-change balancing at constant memory | Audits to ¶5 |
| 5 | The audits | Equal-weight baselines match specialized methods once budgets are fair | Kurin: unitary scalarization plus single-task regularization matches or improves, per-task-gradient overhead; Xin: no gains beyond traditional optimization across language and vision tasks; Gama: SMTO value re-emerges as task count and feature-space demands grow, not needed on simple problems; Elich: no evidence gradient conflicts are MTL-unique, magnitude differences distinguish | Optimizer confound to ¶6 |
| 6 | Optimizer confound | Optimizer changes ride along with method changes and can masquerade as method gains | Loshchilov AdamW one sentence; Elich's Adam analysis and partial loss-scale invariance; the audits' equal-budget controls as the corrective | Dispute status to ¶7 |
| 7 | Gap close | The dispute is unresolved in the general domain and untested on code-switched, low-resource text under explicit optimization control | Claim, no numbers | Domain to 2.3; evidence gap to 2.5 |

Citation hazards to check at first render: two Liu first-authors with initials (B. Liu for CAGrad 2021 and FAMO 2023, L. Liu for IMTL-G 2021) and the Wang/Navier overlaps do not reach this section; confirm citeproc's rendered forms before polishing.

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

Defect register, 2.1 rewrite (2026-09-10): the trimmed file's backup exists in editor history but the group directed a fresh rewrite from current sources; nothing was restored. Verification corrections against paper records: CrisisBench corpora are 166,098 and 141,533 tweets (old 166,100/141,500 approximations dropped); HumAID's 70% is the gold-question assignment gate, not a tweet yield, so the "roughly 70% kept" sentence is dropped in favor of 77,196 annotated to 76,466 after cross-event near-duplicate removal; Reimers' "dozens of seeds" and "1.01 F1" replaced by the seed-difference framing (max 2.59 CoNLL-2003 NER, 8.23 ACE 2005); Ray Chowdhury's test sets are three held-out events plus one general mixed set, 79.36 is English-only F1, and 70.3 (caution and advice) sits on the multilingual mixed set; Sarioglu Kayi anchors moved to the ensemble table (76.5 English, 63.5 Sinhala, 62.6 Odia); the Chen 2018 token-versus-sequence premise could not be anchored and the 2.1.2 close now rests on the Weld pairing precedent; LibMTL is cited for 13 optimization strategies and 8 architectures, replacing the unsupported per-task-gradient claim. CrisisSense-LLM supplies 63.8% versus 4.27% and the A100/A40 counts (LoRA tuning on the 3-GPU A40 node, section 3.3.1; full-parameter tuning on the 16-GPU A100 node, section 3.3.2); Zguir supplies the taxonomy, QSF, 1,346 synthetic examples, and 300 hand-annotated tweets.

Resolved 2026-09-10, group review of 2.1: the CrisisBench Table 7 consolidated-humanitarian row was re-verified against the original PDF layout (page 8 coordinates, reproduced independently), establishing BERT = .860, DistilBERT = .856, RoBERTa = .872. The text's "BERT trailed RoBERTa by .012" is correct; the earlier "DistilBERT" anchor carried a transposed pair. Lesson recorded: when markdown extraction loses table column labels, verify the value-to-column mapping against PDF coordinates, never against ordering cues from prose or other tables.

Resolved 2026-09-10, second finding: the HumAID results table was re-verified against the original PDF layout (page 9 coordinates), confirming the column order RF, SVM, FastText, BERT, D-BERT, RoBERTa, XLM-R and the combined row .700, .710, .712, .768, .769, .781, .777. The text's "support vector machine, random forest, and fastText classifiers stayed at or below .712" is correct; fastText lands at .712 as a deep-learning classifier. HumAID never evaluated logistic regression — the SVM/LR/RF list that seeded the earlier sentence sits in a related-work passage about prior studies' experiments, not HumAID's own setup, so the fresh draft's "logistic regression" was a hallucinated baseline that the group edit to fastText removed. Second lesson recorded: a model list inside a related-work sentence describes other papers, not the paper at hand; the paper's own experiment set comes from its methods or abstract.

Resolved 2026-09-10, third and fourth findings: the HumAID sentence named its strongest transformer ("RoBERTa reached .781 average weighted F1") using the PDF-coordinate-verified column and recast the scope to cross-event ordering ("Evaluations on HumAID showed the same ordering across events"). The Sarioglu Kayi sentence was recast as active contrast ("scored 76.5% F1 on English but dropped to 63.5% on Sinhala and 62.6% on Odia when transferred without target-language training"), attaching the transfer condition to the two target languages; the comma before "but" stays omitted as a compound predicate (6.3). This register reads as a chronological log: the snapshot paragraph reflects the rewrite, and the Resolved entries reflect the group review that followed.

Resolved 2026-09-10, 2.2 delivery: the literal-citation failure recurred — eight subject-position citations were drafted as plain text and produced no reference entries until converted to @keys; the detector is counting reference URLs against the expected total (34 = 18 from 2.1 plus 16 from 2.2). Standing trap recorded: every subject-position citation is typed as an @key first, never as the rendered form. The Liu pair rendered ambiguously as "(Liu et al., 2021)" twice; citeproc disambiguates narrative citations but not parentheticals (the Alam case behaved identically), so both sentences went narrative and render as "B. Liu et al. (2021)" and "L. Liu et al. (2021)". Standing rule: an ambiguous citation pair goes narrative. "Audit" language was again drafted into prose and replaced with "controlled comparisons"; the group decision applies to every mention, including possessives in section closes. AdamW's entry was initially thought to lack a paper.md record, but was verified under its OpenReview ID at papers/openreview-Bkg6RiCqY7/, where Section 2 is titled "DECOUPLING THE WEIGHT DECAY FROM THE GRADIENT-BASED UPDATE" and the main contribution explicitly states "decoupling the weight decay from the gradient-based update", fully anchoring that phrasing. Coordinate verification also fixes the strongest HumAID transformer as RoBERTa (.781), so the text may name it.

Resolved 2026-09-10, group review of 2.2 (27 findings F1–F27): All seven paragraphs were audited and revised under the 27 review findings. Key corrections: F1 Sener/MGDA-UB objective approximation distinction; F2–F3 acronym expansions on first mention (PCGrad, CAGrad, IMTL-G, Nash-MTL, Aligned-MTL, SAM-GS, GradNorm, FAMO); F4 PCGrad conflict defined via negative cosine similarity; F5 quotes around coined term "gradient surgery"; F6 reinforcement learning without hyphen; F7–F10 surgery refinements restructured with B. Liu / L. Liu citeproc author suppression ([-@key]) to eliminate duplicate author rendering and keep word counts under 30 words; F11 Kendall homoscedastic uncertainty definition added; F12–F13 GradNorm and FAMO mechanism precision; F14 unitary scalarization framing; F15 per-task gradient overhead scaling; F16–F17 Elich and Gama boundary conditions; F18 colon capitalization in boundary condition; F19–F20 AdamW and Elich loss-scale invariance precision; F21 parenthetical consolidation of comparison citations; F22–F27 tense, voice, terminology ("controlled comparisons"), and transition to Section 2.3. Author suffix in references.bib updated to 'Grassi, Jr., Valdir' so in-text renders as 'Gama and Grassi (2025)' per APA 7 §8.17.

Resolved 2026-09-10, Elich invariance refinement: Replaced "proved... showing that constant rescaling cancels out in head updates" with "derived a partial loss-scale invariance under mild assumptions, showing that loss scaling cancels out in head parameters while still affecting the shared backbone." This corrects the epistemological verb to match the source paper's own framing ("theoretically derive") and captures the essential architectural boundary (heads vs. shared backbone).

Resolved 2026-09-10, Kendall gloss refinement: Spot-checked against papers/arxiv-1705.07115/paper.md §3.1 (line 128). The hyphenated compound "input-independent" does not appear verbatim in the paper; the paper states "which is not dependent on the input data... It can therefore be described as task-dependent uncertainty." The parenthetical in Paragraph 4 was trimmed to "(task-dependent)" to match the author's exact phrasing.

Resolved 2026-09-10, multitask spelling normalization: Normalized descriptive occurrences of "multi-task" to solid "multitask" in Paragraph 2 ("multitask supervised and multitask reinforcement learning") and Paragraph 3 ("Nash multitask learning"), maintaining chapter-wide consistency with "multitask optimization", "multitask learning", and "multitask optimizers". Capitalized method acronym expansions that mirror published paper titles (IMTL-G and Aligned-MTL) remain hyphenated as published.



