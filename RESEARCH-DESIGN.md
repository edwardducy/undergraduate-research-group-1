# Research Design

> This document is the study's source of truth: the research questions, objectives, hypotheses, evaluation decisions, and frozen design constants. The manuscript chapters (1–3) present this material for the panel. If a chapter and this document ever disagree, this document wins and the chapter is corrected.

## 1. Study Overview

During a typhoon, response actors face thousands of public tweets. They must sort those messages into what is needed, where it is needed, and how urgently. Much of this traffic is written in Taglish, the code-mixed Tagalog and English widely used in Philippine social media. The exact share varies by platform, audience, and event, so this study measures it for the typhoons it covers rather than assuming it.

Triage models must also work on a typhoon they have never seen. Crisis-informatics research documents a substantial drop in performance when models are evaluated on disasters unseen in training, and held-out-event evaluation is well established for English crisis benchmarks. No equivalent exists for Taglish.

Three things remain unaddressed.

First, we know of no Taglish benchmark or held-out evaluation for disaster triage. Filipino disaster NLP has covered sentiment analysis, content classification, and pre-neural intent extraction. One study classifies urgency using tweets from a single typhoon. Tagalog entity-recognition resources cover news and historical text, not typhoon discourse.

Second, we know of no benchmark that labels all three of named entities, intent, and urgency on the same messages. A fair multitask comparison requires this, since every model must train and be tested on identical inputs. Existing crisis benchmarks pair at most two of the three tasks on the same tweets. The one that does pair span extraction with a message-level task is English-only and classifies event type rather than triage needs.

Third, we know of no study that combines held-out-event evaluation, code-mixing-stratified reporting, a comparison of task-balancing mechanisms, and a deployment cost profile. The closest multitask work on disaster tweets uses two tasks from a single corpus and does not examine efficiency.

Each model in this study is tested on a typhoon it never saw, and on the language mix that typhoon actually produced. The study addresses that gap with three deliverables:

1. **A benchmark corpus** of Taglish typhoon tweets covering six Philippine typhoons from 2020 to 2024. Every tweet carries all three label sets: named entities, intent, and urgency. Every model therefore trains and is tested on exactly the same inputs.
2. **A unified multitask triage system.** One shared encoder (SEA-LION-ModernBERT-300M) feeds three task heads. Uniform 1:1:1 weighting is the system of record. It is evaluated against three dedicated single-task models and against two alternative ways of balancing the tasks during training: tuned static weights and PCGrad.
3. **A pre-registered evaluation protocol** that fixes folds, seeds, metrics, decision rules, and analysis margins before the first training run. Every verdict in the results chapter is then auditable rather than post hoc.

The study follows the Design Science Research Methodology (Peffers et al., 2007), with quantitative benchmarking as the empirical method inside its design-and-development, demonstration, and evaluation phases. Two hypotheses are tested confirmatorily and five further analyses are reported descriptively. The protocol is deliberately simple at the decision level. Every arm sees identical data, folds, seeds, and tuning budget, and every comparison is paired by typhoon.

## 2. Research Questions

**RQ1 — Out-of-disaster generalization.** On typhoons never seen during training, does one multitask model match or beat three dedicated single-task models, in aggregate and across Taglish code-mixing strata (Tagalog-dominant, Mixed, English-dominant)?

**RQ2 — Deployment feasibility.** How much single-request latency and memory does the unified model save against running the three single-task models together, and does it retain every task's performance within a pre-computed tolerance?

**RQ3 — Optimization under domain shift.** Under identical data, schedules, and tuning budgets, does PCGrad (gradient-conflict surgery) reduce task interference relative to equal loss weights and to tuned static weights?

**RQ4 — Gradient-conflict diagnostics.** During training with equal weights, how often do the tasks' gradients conflict; does conflict concentrate between the token-level task (named entities) and the sequence-level tasks (intent, urgency); and is higher conflict followed by worse per-task progress?

## 3. Research Objectives

**RO1 — Benchmark and compare (RQ1).** Build the common corpus and compare the multitask model against the single-task baselines across $K = 6$ leave-one-disaster-out folds $\times$ $S = 5$ seeds: a confirmatory aggregate comparison, plus per-stratum and per-task descriptive reporting.

**RO2 — Profile deployment (RQ2).** Measure batch-1 latency and peak memory (VRAM and system RAM) of every serving configuration on commodity hardware — sequential all-resident as the primary baseline, concurrent where feasible, load/unload descriptively — and assess per-task retention against the tolerance $\epsilon$.

**RO3 — Contrast balancing mechanisms (RQ3).** Train three balancing arms — uniform $1{:}1{:}1$ weights, tuned static weights, PCGrad — under identical conditions, and quantify per-task negative transfer, $\mathrm{NT}_t = \mathrm{F1}_{\mathrm{STL},t} - \mathrm{F1}_{\mathrm{MTL},t}$.

**RO4 — Instrument gradient conflict (RQ4).** Log pairwise task-gradient cosines during the uniform arm across checkpoint windows, and describe conflict frequency, its distribution over task pairs, and its lead–lag relationship with per-task performance changes.

## 4. Hypotheses

### 4.1 Confirmatory

Both are tested with the exact paired sign-flip permutation test on fold-level differences (seeds averaged within each fold, so each typhoon contributes one paired observation; $N = K = 6$; one-sided $\alpha = 0.05$). With six folds, the smallest p-value any result can reach is $2^{-6} \approx 0.016$. This floor is why the design requires $K \ge 5$ and why the departmental template's four-fold wording is overridden: at $K = 4$, even a result where every fold favors the same arm cannot reach $p < 0.05$ (floor $0.0625$).

**H1 — Multitask aggregate superiority (RQ1).** On held-out typhoons, the multitask model achieves higher aggregate Macro-F1 — the unweighted mean of the three per-task Macro-F1 scores — than the single-task baselines' aggregate (the same mean taken over the three dedicated models' scores).
$$H_0\colon \mu_{\mathrm{MTL}} - \mu_{\mathrm{STL}} \le 0 \quad \text{vs.} \quad H_1\colon > 0.$$
If the test fails to reject, the aggregate difference is reported with its confidence interval, and the per-task picture (D2, D4) carries the interpretation.

**H2 — PCGrad aggregate contrast (RQ3).** PCGrad achieves higher aggregate Macro-F1 than uniform scalarization.
$$H_0\colon \mu_{\mathrm{PCGrad}} - \mu_{\mathrm{Uniform}} \le 0 \quad \text{vs.} \quad H_1\colon > 0.$$
The hypothesis is stated on the aggregate directly: the fold-level difference in summed negative transfer between two multitask arms is algebraically $T \times$ the aggregate Macro-F1 difference (the single-task terms cancel), so a transfer-framed test would measure the same quantity under a different label.

### 4.2 Descriptive

These analyses carry no significance claims. Each is reported with 95% confidence intervals, and each has a pre-stated interpretation in §5 so that any outcome is readable.

**D1 — Stratified performance (RQ1).** Macro-F1 per code-mixing stratum on the pooled out-of-fold construction (each tweet scored exactly once, by the model trained without its typhoon). Stratum × class cells with fewer than $n = 25$ unique examples are flagged low-support and read cautiously. Tokenizer fertility (subwords per word) per stratum is reported as explanatory context.

**D2 — Per-task negative transfer (RQ1, RQ3).** $\mathrm{NT}_t$ per fold and task with fold-level confidence intervals: whether an aggregate result conceals a task that still suffers from sharing.

**D3 — PCGrad versus tuned static weights (RQ3).** The fold-level difference (PCGrad − tuned) with its confidence interval. Following Xin et al. (2022) and Kurin et al. (2022), little or no advantage is expected. The comparison is descriptive because a formal non-superiority verdict at $K = 6$ would demand more precision than six folds can deliver.

**D4 — Efficiency and retention (RQ2).** Latency and peak-memory reductions against the all-resident sequential baseline, reported per serving configuration; and retention: every task's $\Delta\mathrm{F1}_t$ (MTL − STL) must clear $-\epsilon$, where $\epsilon$ is frozen at $1.5\times$ the pooled seed-level standard deviation of single-task Macro-F1, computed from all $K$ folds after all single-task runs and before any multitask run.

**D5 — Gradient-conflict structure (RQ4).** Pairwise conflict frequency $\bar\kappa_{i,j}$ (the share of batches with $\cos(g_i, g_j) < 0$), the contrast between the NER–sequence pairs and the sequence–sequence pair, and the per-(run, task) lead–lag correlation between $\kappa_t(c)$ and the forward change in per-task F1 across checkpoint-window boundaries. Conflict can be benign and need not couple with negative transfer (Jiang et al., 2023), so near-zero or positive associations are interpretable outcomes, not null failures.

## 5. Expected Outcomes

**Guaranteed deliverables** — produced regardless of any hypothesis verdict: the annotated corpus (released as tweet identifiers plus labels), the trained models with run manifests, the code and configurations, and complete results tables for every arm, fold, and task. The benchmark itself is a contribution, as the gap analysis of Chapter 2 motivates.

**Predictions.** Each expectation states its basis, and what the opposite outcome would mean:

| Expectation | Basis | If the opposite occurs |
|---|---|---|
| H1: MTL matches or beats STL in aggregate | Multitask learning gains in related low-resource settings; shared supervision on identical inputs | An auditable negative verdict for this setting, with D2 localizing which task suffers — still a complete, reportable result |
| H2: PCGrad shows no large advantage over uniform | Xin et al. (2022); Kurin et al. (2022): tuned simple baselines match complex multitask optimizers | A positive verdict would be evidence that gradient surgery matters specifically under domain shift — also informative |
| D1: strata differ; direction and magnitude unknown | Code-mixing shifts tokenization and label expression across strata | Any pattern is benchmark characterization; low-support cells are flagged, not over-read |
| D4: ≥ 50% memory and latency savings vs. all-resident sequential | Arithmetic: one resident encoder instead of three; the batch-1 workload is encoder-dominated | Near-certain by construction; the interesting number is the magnitude |
| D4: every task retains within $\epsilon$ | Shared-encoder multitask typically retains task performance when training is stable | A task-specific degradation beyond $\epsilon$ is itself a deployment-relevant finding, reported plainly |
| D5: low conflict frequency, weak coupling with transfer | Jiang et al. (2023): task gradients are typically near-orthogonal; conflict can act as regularization | Strong negative coupling would be direct evidence for conflict surgery — the more interesting outcome |

## 6. Study Workflow

Chronological. **Gates** stop the line until passed; **freezes** lock a choice before the evidence that could bias it exists.

1. **Roster lock.** The Carina 2024 event is harvested end-to-end as the pre-lock acquisition pilot; only a pilot that delivers the projected harvest pool locks the roster *(gate)*.
2. **Acquisition.** Per event: the frozen window of 24 hours before to 72 hours after the official PAGASA landfall time; deduplication; redaction of direct identifiers under the Data Privacy Act of 2012 compliance statement; a screen pass-rate audit.
3. **Screening and pre-processing.** Taglish screen (language identification + code-mixing index); stratum assignment. The CMI formula, stratum thresholds, and the borrowing/cognate treatment freeze here — before annotation — because one implementation is both the inclusion rule and the stratification instrument *(freeze)*.
4. **Annotation.** The five thesis authors: guideline calibration round → production annotation → agreement round (a fixed three-author panel re-annotates ≥ 20% of the corpus, blind; acceptance: Fleiss' $\kappa \ge 0.60$ for intent and urgency, mean pairwise span-level Macro-F1 ≥ 0.75 for NER) → adjudication. Below-threshold agreement triggers guideline revision and re-annotation, not a failed corpus.
5. **Corpus freeze.** Folds assigned; the 15% development splits drawn once and frozen; per-stratum × class support counted from corpus-level counts — each tweet is out-of-fold exactly once, so pooled support is known before any training *(freeze: the D1 reporting list)*.
6. **Launch checklist.** Encoder pilot — tokenizer fertility probe, three single-task pilot fine-tunes (one per task), and a footprint probe; all tool versions, trial counts, and sampling intervals recorded in the run manifest *(gate)*.
7. **Training.** All $K$ single-task runs complete first; $\epsilon$ freezes from their pooled seed-level standard deviation *(freeze)*; then the multitask arms (uniform, tuned, PCGrad) in any order. Every arm tunes over the same learning-rate × weight-decay grid with the same budget: grid points run at one seed, the selected configuration at $S$ seeds. Gradient logging rides the uniform arm, side-effect-free.
8. **Evaluation and logging.** Per-checkpoint out-of-domain Macro-F1 and final-model fold-level results; every run emits structured CSV/JSON logs automatically.
9. **Efficiency profiling.** Frozen trial counts after warm-up; sequential all-resident primary; concurrent only if it completes without memory contention; load/unload descriptive.
10. **Analysis.** H1 and H2 tested; D1–D5 reported with confidence intervals; Wilcoxon signed-rank reported alongside each permutation test as a robustness check.
11. **Release (post-defense).** Tweet identifiers plus annotations, a Datasheet for Datasets, a model card, and the code and configurations.

## 7. Evaluation Protocol

### Design and units of inference

- **Leave-one-disaster-out.** One fold per typhoon: that event is the out-of-domain test set; the remaining $K - 1$ train. Every tweet is tested exactly once, on a typhoon its model never saw — deployment's actual condition.
- **$K = 6$ folds, minimum 5.** Pre-declared alternates (Kristine, Pepito) substitute for excluded primaries only, so $K$ can only decrease; $K \ge 5$ is re-verified at roster lock.
- **$S = 5$ seeds per arm per fold**, with identical splits, per-seed batch ordering, schedule, and early-stopping policy across arms. Seeds are averaged within a fold; the typhoon (fold) is the unit of inference, $N = K$. Counting seed replicates as independent observations would be pseudo-replication.

### Development data and model selection

A 15% development split is drawn from the training events only, stratified jointly by event and label, drawn once before any experiment, and applied identically across all arms. Single-task arms early-stop on their own task's dev Macro-F1; multitask arms on the unweighted mean of per-task dev Macro-F1. Held-out-disaster data is never used for model selection. The within-disaster leakage this admits is accepted by design and identical across arms, so comparisons remain fair and absolute numbers read as deployment-realistic.

### Fairness rules (arm parity)

Common corpus — every arm trains on the same inputs; single-task arms use their own task's labels, so gains are attributable to parameter sharing with cross-task supervision, not to data differences. Identical encoder and head construction in every arm. Per-task losses are mean-reduced so that $1{:}1{:}1$ weighting is scale-comparable across tasks. Identical tuning grid and budget for every arm — the tuning budget moves conclusions more than seed choice (Xin et al., 2022). The tuned-scalarization arm searches its weights strictly by inner resampling within training folds.

### Metrics

Intent and urgency: classification Macro-F1. NER: span-level Macro-F1 (exact boundaries and type; macro over entity types). Aggregate: the unweighted mean of the three. Stratified results ride the pooled out-of-fold construction, computed per seed then seed-averaged, with support counted in unique examples. Negative transfer $\mathrm{NT}_t$. Conflict frequency $\kappa_{i,j}(c)$ = the share of a checkpoint window's batches with $\cos(g_i, g_j) < 0$, with window-mean cosines always reported alongside (a sign-only frequency discards magnitude). Checkpoint windows are spaced on matched fractions of each run's realized optimizer steps. Latency: mean ± SD over frozen trials. Memory: peak VRAM and peak system RAM, reported separately.

### Statistical inference

Exact paired sign-flip permutation, one-sided, $\alpha = 0.05$, for H1 and H2; Wilcoxon signed-rank reported alongside each as a robustness check. 95% confidence intervals for every descriptive quantity, clustered by fold where the quantity is computed per-tweet. The resolution floor is $p_{\min} = 2^{-K} \approx 0.016$ at $K = 6$; rejection therefore requires near-unanimity across folds, which the expected-outcomes table accounts for. Rejected alternatives (paired t-test as primary; pooled-over-tweets or over-seeds tests; mixed models at $K = 6$) and their reasons are argued in Chapter 3.

### Run integrity

A run that diverges (non-finite loss) or collapses — more than 95% of a task's dev predictions in one class; for NER, more than 95% of predicted spans of one type or no spans — is re-run once at the same seed and logged; exclusion criteria are fixed in advance. All $K$ single-task runs precede the first multitask run so that $\epsilon$ freezes on schedule (Workflow step 7).

## 8. Design Constants

| Constant | Value | Why |
|---|---|---|
| Tasks $T$ | 3 — NER, intent, urgency | Triage needs what, where, how urgently |
| Folds $K$ | 6 (minimum 5); 2020–2024 pool: Ulysses, Rolly, Odette, Karding, Egay, Carina; alternates (Kristine, Pepito) substitute only | One fold per typhoon; $K \ge 5$ keeps one-sided exact tests viable ($p_{\min} = 2^{-K}$) |
| Seeds $S$ | 5 | Run-to-run variance estimated within each fold |
| Checkpoint windows $C$ | 15–20 | Enough boundaries for lead–lag correlations without logging overhead |
| Dev split | 15% of training events, stratified event × label | Model selection without held-out-disaster leakage |
| Shared encoder | SEA-LION-ModernBERT-300M (base pretrain, MIT license; not the embedding variants) | Filipino + English coverage; base pretrain suits fine-tuning |
| Loss normalization | Per-task mean-reduced losses | Makes $1{:}1{:}1$ weights scale-comparable |
| Architecture parity | Identical encoder and head construction across all arms | Differences attributable to sharing, not capacity |
| Tuning parity | Identical LR × weight-decay grid and budget for every arm | Tuning budget moves conclusions more than seed choice |
| Checkpoint matching | Fraction of realized optimizer steps | Trajectories comparable despite different early stops |
| Gradient logging | Uniform arm, final shared layer, side-effect-free | The instrumented run is the run — no separate experiment |
| Deployment protocol | All-resident sequential primary; concurrent conditional; load/unload descriptive | The deployment-relevant service comparison |
| Workload | Batch 1, every request exercises all three tasks | Matches triage-serving reality |
| Retention tolerance $\epsilon$ | $1.5\times$ pooled seed-level SD of STL Macro-F1 (all $K$ folds), frozen before the first MTL run | Margin scaled to measured training noise |
| Support threshold | $n \ge 25$ unique examples per stratum × class (D1 reporting) | Below this, estimates are too noisy to interpret |
| Primary dynamic method | PCGrad | Representative gradient-conflict method with published evidence on both sides |

**Still to freeze before launch:** CMI formula and stratum thresholds; borrowing/cognate treatment; the concrete LR × weight-decay grid; gradient-log subsample size (if per-step extraction proves costly); profiling precision and power mode.

## 9. Scope and Limitations

- **Language scope.** Taglish only, by the frozen screen. Bisaya, Ilocano, Bikol, and Waray disaster discourse is out of scope by construction (a delimitation stated in Chapter 1).
- **Platform and data availability.** Retrospective Twitter/X data acquired through research-permitted routes; hydration depends on future API availability, acknowledged in the release Datasheet. Released artifacts carry tweet identifiers and annotations, never tweet content.
- **Annotators.** The five thesis authors annotate; there is no external annotator panel. Blind re-annotation by a fixed three-author panel with agreement thresholds guards quality, but single-team annotation remains a limitation.
- **Statistical resolution.** $K = 6$ bounds the strongest possible evidence ($p \approx 0.016$) and requires near-unanimity across folds to reject; the descriptive tier carries the finer-grained readings with their uncertainty rather than over-claiming them.
- **Diagnostics scope.** Conflict is measured on the uniform arm at the final shared encoder layer; PCGrad-arm pre-projection cosines are an optional robustness extra, not a commitment.
- **Hardware.** One commodity GPU workstation; no edge or mobile deployment — that is future work.
- **Dev-split leakage.** Early stopping tunes on within-disaster dev data (identically across arms); absolute numbers are read as deployment-realistic, not event-blind.
