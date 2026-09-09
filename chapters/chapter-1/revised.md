# Chapter 1 Revision Skeleton

Atomic statements for revising Chapter 1 of `draft.md`. Part 1 covers the Introduction, whose four groups recompose into four paragraphs. Part 2 covers the Background of the Study, which recomposes as one section without subheadings. Its groups follow the advisor funnel, macro global research, meso the Philippine Taglish context, micro local bottlenecks, and a synthesis that narrows from global corpora to the Taglish corpus, with a final group stating the study's response. Part 3 covers the Statement of the Problem, which keeps its summary-plus-enumeration shape. Part 4 covers the Objectives of the Study, whose items pair declaratively with the specific problems. Parts 5 to 10 cover the remaining sections in draft order. A Background statement may only revisit an Introduction beat by adding a citation, a mechanism, or a number. Acronyms expand once in the Introduction and appear as short forms afterward. Table 1 stays a table, and the corpus gap group holds its prose takeaway.

## Part 1: Introduction

### Context

1. Natural disasters such as typhoons and severe floods generate high volumes of social media messages (Imran et al., 2016).
2. Emergency responders can use these real-time messages for situational awareness during disaster response (Imran et al., 2016).
3. Disaster triage involves extracting three types of information from these messages.
4. Token-level Named Entity Recognition (NER) extracts critical locations and infrastructure.
5. Sequence-level intent classification detects humanitarian needs.
6. Sequence-level urgency prioritization flags life-threatening distress.

### The Computational Problem

1. Fine-tuning a separate Single-Task Learning (STL) model for each prediction target is a common deployment practice (Alam et al., 2021; Wang et al., 2021).
2. This approach scales inference latency and memory consumption linearly with task count (O(K) for K tasks).
3. This linear cost multiplies the latency and memory needed for real-time triage on modest local hardware.
4. Multi-Task Learning (MTL) with hard parameter sharing generates shared representations from a single multilingual Transformer backbone (Caruana, 1997).
5. All tasks share one forward pass, giving near-constant inference latency (O(1) relative to K).

### The Optimization Problem

1. Joint multi-task training couples heterogeneous loss functions (Sener & Koltun, 2018).
2. This coupling produces two primary gradient failure modes during backpropagation.
3. The first failure mode is gradient magnitude disparity (Chen et al., 2018).
4. The second failure mode is gradient directional conflict (Yu et al., 2020).
5. Joint training on low-resource, noisy, code-switched text benefits some task combinations and induces negative transfer in others (Adouane & Bernardy, 2020).
6. It is unknown whether interference between the tasks grows with the degree of Tagalog-English mixing.

### The Study's Response

1. This study develops an empirical benchmarking framework grounded in the Design Science Research Methodology (DSRM) (Peffers et al., 2007).
2. The framework benchmarks Multi-Task Optimization (MTO) methods (Xin et al., 2022) against Uniform Equal Weighting (EW), tuned Static Linear Scalarization (LS), and isolated STL baselines.
3. The candidate methods span two families, dynamic loss weighting and gradient surgery.
4. All methods are evaluated across four historical Philippine typhoons under a 4-fold Leave-One-Event-Out (LOEO) protocol.
5. PyTorch gradient diagnostic hooks and an intra-task half-batch gradient baseline quantify gradient interference.
6. The framework evaluates out-of-distribution generalization.
7. A stratified analysis reports performance and subword fertility across three language profiles, English-dominant, mixed, and Tagalog-dominant.
8. The framework deploys the fine-tuned model within a prototype web application for disaster triage.

## Part 2: Background of the Study

### Joint architecture for disaster triage

1. Crisis tweets carry humanitarian content categories such as damage reports and urgent needs (Alam et al., 2021).
2. They also convey urgency that priority classifiers can detect (Wang et al., 2021).
3. Separate models for the three triage tasks require three forward passes and three encoder copies per message.
4. MTL with hard parameter sharing places a small classification head for each task on one shared multilingual Transformer backbone (Caruana, 1997).
5. For the three triage tasks, one shared forward pass replaces three separate passes (O(1) relative to K).
6. Sharing also imposes an inductive bias toward representations that support all triage tasks (Caruana, 1997).

### The optimization challenge of shared training

1. Training over shared parameters turns joint learning into a multi-objective optimization problem (Sener & Koltun, 2018).
2. Task gradients on the shared parameters can point in opposing directions (B. Liu et al., 2021; Navon et al., 2022; Yu et al., 2020).
3. An update along one task then increases the loss of another to first order, producing destructive interference and negative transfer (Yu et al., 2020).
4. Gradients can also differ greatly in magnitude, letting dominant tasks suppress secondary objectives (Chen et al., 2018).
5. Joint modeling of token-level and sequence-level tasks offers mutual linguistic reinforcement (Weld et al., 2022).
6. Realizing reliable positive transfer (Δm > 0) under this conflict depends on principled gradient balancing.

### Balancing methods and mixed evidence

1. Uncertainty Weighting weighs each loss by its task's learned uncertainty (Kendall et al., 2018).
2. GradNorm re-weights losses to balance gradient magnitudes across tasks (Chen et al., 2018).
3. Fast Adaptive Multitask Optimization (FAMO) adjusts weights so all task losses decrease at similar rates (B. Liu et al., 2023).
4. Gradient-surgery methods instead edit the gradient vectors themselves.
5. PCGrad projects conflicting gradients away from each other (Yu et al., 2020).
6. CAGrad optimizes the worst-case task improvement within a trust region (B. Liu et al., 2021).
7. IMTL-G combines gradients at equal angles (L. Liu et al., 2021).
8. Nash-MTL formulates this combination as a bargaining game (Navon et al., 2022).
9. Benchmarks disagree on whether dynamic weighting helps at all (Xin et al., 2022).
10. Under adaptive optimizers such as AdamW, EW stays competitive with gradient-surgery methods (Elich et al., 2024).
11. Published comparisons rarely leave homogeneous vision and translation settings.
12. Joint training on low-resource, noisy, code-switched text benefits some task combinations and induces negative transfer in others, with task order and relative training data size shaping the outcome (Adouane & Bernardy, 2020).

### The Philippine disaster setting

1. Around twenty tropical cyclones enter the Philippine Area of Responsibility yearly, and about eight or nine cross the country (Philippine Atmospheric, Geophysical and Astronomical Services Administration [PAGASA], n.d.).
2. During Typhoons Haiyan, Vamco, Rai, and Paeng, citizens posted distress messages on Twitter (Ermino et al., 2022; Imran et al., 2016).
3. Agencies such as the National Disaster Risk Reduction and Management Council (NDRRMC) and the Office of Civil Defense maintain official social media channels for disaster communication.
4. Official situational reports documented the flooding and damage from the recent events (NDRRMC, 2020, 2022).
5. This volume can exceed what duty staff can triage manually in real time.

### Taglish complexity and mixing

1. Taglish messages combine Tagalog affixes and focus markers with English root words such as *ma-evacuate* and *nag-collapse* (Herrera et al., 2022; Montalan et al., 2025).
2. The register also shows non-standard spelling, abbreviations, and informal colloquialisms (Herrera et al., 2022; Montalan et al., 2025).
3. Multilingual tokenizers split these hybrid words into many subword tokens (Petrov et al., 2023).
4. Across evaluated tokenizers, Tagalog needs 1.74 to 3.28 tokens per word, against 1.20 to 2.60 for English (Petrov et al., 2023).
5. Fragmentation lengthens each message and dilutes its token embeddings (Petrov et al., 2023).
6. These sequence changes may amplify the gradient imbalances described above.
7. The degree of Tagalog-English mixing varies across messages.
8. Word-level language tags per tweet make this mixing degree measurable (Herrera et al., 2022).
9. These tags let the study stratify results by mixing level.

### Local hardware and connectivity

1. Five days after Typhoon Rai made landfall, 236 cities and municipalities had power interruptions and 135 areas had disrupted communication services (NDRRMC, 2021).
2. During Typhoon Kristine, 361 cities and municipalities had power interruptions, and 123 still lacked power one week after landfall (United Nations Office for the Coordination of Humanitarian Affairs [OCHA], 2024).
3. During an outage, no configuration can ingest new messages, because the data arrives online.
4. Cloud-hosted Large Language Model (LLM) inference also fails whenever connectivity drops.
5. Cloud services add recurring API costs.
6. Uploading citizen messages to cloud processors adds privacy obligations under the Data Privacy Act of 2012 (Republic Act No. 10173).
7. A compact model on local hardware processes messages without cloud dependence.
8. It keeps citizen data inside agency infrastructure and resumes with restored service.

### The corpus gap

1. HumAID supplies large-scale humanitarian intent labels for 19 disaster events (Alam et al., 2021).
2. It is monolingual English and carries no token-level entity spans (Alam et al., 2021).
3. Batayan evaluates eight Philippine natural language processing (NLP) tasks in isolation (Montalan et al., 2025).
4. TweetTaglish documents language mixing only at the document level (Herrera et al., 2022).
5. TLUNIFIED-NER labels entities in formal news text (Miranda, 2023).
6. No public corpus combines entity, intent, and urgency annotations on Taglish crisis messages under cross-event evaluation.
7. This gap also limits multi-task optimization research on crisis triage.
8. In a shared encoder, NER losses aggregate over all subword tokens, while classification losses use one pooled representation.
9. This asymmetry can let token-level gradients dominate under EW (Chen et al., 2018).
10. Dominant token-level updates can overwrite pooled representations and induce negative transfer.

### The study's response

1. This study curates the Multi-Task Corpus of Taglish Disaster Tweets across four Philippine typhoons.
2. Corpus evaluation follows the 4-fold LOEO protocol.
3. The study benchmarks dynamic loss weighting and gradient surgery against single-task and static baselines.
4. The intended deliverables are empirical guidance and an operational triage artifact.

## Part 3: Statement of the Problem

### The general statement of the problem

1. This study addresses real-time disaster triage of Taglish crisis messages on modest local hardware.
2. The tasks are token-level NER, sequence-level intent classification, and sequence-level urgency classification.
3. Fine-tuning a separate STL model for each prediction target is a common deployment practice (Alam et al., 2021; Wang et al., 2021).
4. This approach scales inference latency and memory consumption linearly with task count (O(K) for K tasks).
5. MTL with hard parameter sharing generates shared representations from a single multilingual Transformer backbone (Caruana, 1997).
6. All tasks share one forward pass, giving near-constant inference latency (O(1) relative to K).
7. Joint multi-task training couples heterogeneous loss functions (Sener & Koltun, 2018).
8. Magnitude disparities and directional conflicts can then induce negative transfer (B. Liu et al., 2021; Yu et al., 2020).
9. Benchmarks disagree on whether dynamic weighting helps at all (Xin et al., 2022).
10. Under adaptive optimizers such as AdamW, EW stays competitive with gradient-surgery methods (Elich et al., 2024).
11. Mini-batch noise can fake gradient conflict (Elich et al., 2024).
12. No public corpus combines entity, intent, and urgency annotations on Taglish crisis messages under cross-event evaluation.
13. The study therefore asks whether principled MTO on a curated corpus produces a triage model that is accurate, statistically validated, and locally deployable.

### Specific Problem 1 (Phase 1 - Baseline Analysis)

1. What are the baseline F1 scores, latency, throughput, and memory of isolated STL models?
2. Under EW, how does the shared encoder compare across LOEO splits in accuracy, gradient norm ratios, and cosine similarities?

### Specific Problem 2 (Phase 2 - Design and Artifact Creation)

1. How can the training pipeline, candidate MTO algorithms, and gradient diagnostics be designed to mitigate interference on Taglish disaster text?
2. How can the fine-tuned model be integrated into a prototype web application for disaster triage?

### Specific Problem 3 (Phase 3 - Empirical Benchmarking)

1. Across LOEO splits, how do the two method families compare against STL, EW, and LS in F1, transfer, efficiency, and runtime?
2. Does the ranking of methods stay stable across the three language profiles?

### Specific Problem 4 (Phase 4 - Comparative Validation)

1. To what extent are performance differences statistically significant under paired bootstrap intervals and Friedman-Nemenyi tests?
2. How strongly does Δm correlate with gradient conflict measured against the intra-task half-batch baseline?

## Part 4: Objectives of the Study

Each specific objective pairs one-to-one with its same-numbered specific problem.

### The general objective

1. The primary objective is to construct the Multi-Task Corpus of Taglish Disaster Tweets, benchmark candidate MTO algorithms with gradient diagnostics, and deploy the fine-tuned model in a prototype web application.

### Specific Objective 1 (Phase 1 - Baseline Analysis)

1. To measure the baseline F1 scores, latency, throughput, and memory of isolated STL models.
2. To compare the shared encoder under EW across LOEO splits in accuracy, gradient norm ratios, and cosine similarities.

### Specific Objective 2 (Phase 2 - Design and Artifact Creation)

1. To design and implement the training pipeline, candidate MTO algorithms, and gradient diagnostics to mitigate interference on Taglish disaster text.
2. To integrate the fine-tuned model into a prototype web application for disaster triage.

### Specific Objective 3 (Phase 3 - Empirical Benchmarking)

1. To benchmark the two method families against STL, EW, and LS across LOEO splits in F1, transfer, efficiency, and runtime.
2. To determine whether the ranking of methods stays stable across the three language profiles.

### Specific Objective 4 (Phase 4 - Comparative Validation)

1. To evaluate the statistical significance of performance differences under paired bootstrap intervals and Friedman-Nemenyi tests.
2. To analyze how strongly Δm correlates with gradient conflict measured against the intra-task half-batch baseline.

## Part 5: Theoretical Framework

Display equations and the six-activity DSRM list stay in `draft.md`, and the statements below carry their claims.

### The three theoretical foundations

1. This study rests on three foundations, DSRM, complexity analysis of shared representations, and multi-objective optimization with gradient dynamics.

### Design Science Research Methodology

1. The study adopts the DSRM of Peffers et al. (2007).
2. Design science produces knowledge through the systematic design, implementation, and evaluation of computational artifacts.
3. The Peffers lifecycle spans six activities, from problem identification and motivation to communication (Peffers et al., 2007).
4. This study structures the lifecycle into four phases, Baseline Analysis, Design and Artifact Creation, Empirical Benchmarking, and Comparative Validation.
5. The primary contribution is the empirical MTO knowledge and the curated corpus, while the prototype serves as the demonstration artifact.

### Complexity of shared representations

1. K separate models require K forward passes and K model copies per input.
2. Hard parameter sharing computes the shared representation once in a single forward pass.
3. The backbone cost dominates the small task heads, so inference time stays independent of K.
4. Task heads also add negligible memory beside the shared backbone.
5. Sharing also imposes an inductive bias toward representations that support all triage tasks (Caruana, 1997).
6. This bias restricts the parameter search space and is expected to reduce sample complexity on code-switched text (Baxter, 2000).

### Multi-objective optimization and gradient dynamics

1. Training over shared parameters turns joint learning into a multi-objective optimization problem (Sener & Koltun, 2018).
2. Multi-task learning seeks a Pareto-optimal configuration where no task loss decreases without increasing another.
3. A first-order expansion shows that a step along one task changes another loss by the gradient inner product (Yu et al., 2020).
4. The first failure mode is gradient magnitude disparity (Chen et al., 2018).
5. Dominant task gradients then dictate the trajectory of parameter updates.
6. The second failure mode is gradient directional conflict (Yu et al., 2020).
7. A negative inner product makes updates along one task increase the loss of another.
8. MTO algorithms respond through dynamic loss weighting or gradient surgery.
9. An intra-task half-batch baseline separates stochastic noise from genuine cross-task conflict (Elich et al., 2024).
10. Statistical validity comes from paired bootstrap intervals with omnibus Friedman-Nemenyi ranking (Demšar, 2006).

## Part 6: Conceptual Framework

Figures 1 and 2 stay as figures in `draft.md`.

### The IPO structure

1. The study adopts the Input-Process-Output (IPO) model to structure the experimental design, implementation, and evaluation.
2. Figure 1 shows the overall research framework, and Figure 2 details the software artifact workflow.

### Input Stage

1. I1 is the Multi-Task Corpus of Taglish Disaster Tweets across four typhoon events, with word-level language tags per tweet.
2. I2 is four candidate encoders, XLM-RoBERTa Base, multilingual BERT Base, SEA-LION ModernBERT, and RoBERTa-Tagalog Base.
3. I3 is the optimization configurations benchmarked against single-task and static baselines.
4. I4 is a local workstation testbed under FP16 mixed precision, sequence length 128, and batch size 32.

### Process Stage

1. P1 covers text normalization, subword tokenization, BIO span alignment, and LOEO splitting.
2. P2 runs multi-task fine-tuning with gradient diagnostic logging against the intra-task half-batch baseline.
3. P3 performs two-tier statistical validation and the stratified analysis across three language profiles.
4. P4 embeds the fine-tuned model into an asynchronous FastAPI service with gazetteer resolution.

### Output Stage

1. O1 is the trained model with concurrent predictions in one forward pass.
2. O2 is benchmark tables covering F1 scores, transfer, efficiency, and gradient dynamics across 12 paired runs.
3. O3 is bootstrap confidence intervals and non-parametric test results.
4. O4 is the prototype web application with four incident lifecycle states.

### The artifact workflow

1. The artifact runs inference through an asynchronous FastAPI service on a relational database, SQLite or PostgreSQL.
2. Replayed event streams and manual ingestion feed Taglish messages to the service.
3. The service performs single-pass inference for entities, intent, and urgency.
4. Extracted entities are resolved against a local Philippine gazetteer, with an unmapped-span fallback.
5. Recognized incidents are stored with timestamps and tracked across New, Acknowledged, Responding, and Resolved.
6. Command centers can triage and monitor reports entirely on local infrastructure.
7. A CPU-only demonstration runs the prototype on hardware without a GPU.

## Part 7: Scope and Delimitations

### Scope of the study

1. The dataset targets 6,000 to 10,000 Taglish crisis tweets, with a verified baseline of at least 6,000.
2. Tweets come from four typhoons, Haiyan, Vamco, Rai, and Paeng.
3. Evaluation uses 4-fold LOEO across seeds 42, 123, and 456.
4. The study evaluates four encoder-only backbones with hard parameter sharing and linear heads.
5. The benchmark covers Uncertainty Weighting, GradNorm, FAMO, PCGrad, CAGrad, IMTL-G, and Nash-MTL against STL, EW, and LS.
6. Evaluation criteria cover F1 scores, transfer, latency, throughput, GPU memory, host RAM, and runtime.
7. Gradient hooks track norm ratios and cosine similarities on a commodity GPU testbed.
8. Statistical testing combines paired bootstrap intervals with Friedman-Nemenyi tests and Spearman correlation.
9. Tweets are assigned to profiles by their Code-Mixing Index, computed from the word-level language tags (Gambäck & Das, 2016).
10. A stratified analysis reports performance and subword fertility across three language profiles, English-dominant, mixed, and Tagalog-dominant.
11. The software deliverable includes a FastAPI service, an incident database, and a dashboard with four lifecycle states.

### Delimitations of the study

1. The study excludes regional languages such as Cebuano, Ilocano, and Hiligaynon.
2. Taglish is retained because it predominates in Philippine disaster social media (Herrera et al., 2022).
3. Multi-language annotation would exceed the operational resource budget.
4. The corpus covers public Twitter or X posts from the four typhoons only.
5. Private platforms, non-crisis threads, and live streaming are excluded to ensure deterministic replay.
6. Pipelines handle text only, excluding imagery, photographs, audio, and video.
7. The architecture is encoder-only with hard parameter sharing, excluding soft sharing, adapters, and cloud-hosted LLMs.
8. Evaluation covers the three triage tasks, excluding sentiment analysis, summarization, and translation.
9. The application stays a research artifact outside dispatch networks and government command centers.

## Part 8: Significance of the Study

### Primary beneficiaries

1. Disaster response agencies gain a locally deployable triage system on commodity workstations.
2. NER, intent classification, and urgency prioritization run in one forward pass.
3. Local inference keeps running on already collected messages and resumes ingestion after service returns.
4. Local deployment removes recurring cloud fees and keeps citizen communications protected under the Data Privacy Act of 2012 (Republic Act No. 10173).
5. The prototype demonstrates entity extraction, gazetteer mapping, and lifecycle tracking.

### Secondary beneficiaries

1. Researchers receive the first documented expert-adjudicated Taglish crisis corpus with concurrent annotations under cross-event evaluation.
2. They also gain benchmark evidence on dynamic weighting, gradient surgery, and static baselines under AdamW.
3. The gradient diagnostics and intra-task half-batch baseline provide a methodology to separate genuine conflict from sampling noise.

### Tertiary beneficiaries

1. Academic institutions and developers receive an open-source artifact with modular training scripts, evaluation pipelines, and baseline checkpoints.
2. Future work can extend to regional languages, post-training 8-bit or 4-bit quantization for edge devices, and multi-modal sensor streams.

## Part 9: Definition of Terms

Existing definitions stay in `draft.md` unchanged, since a glossary is already atomic. The Specific Problem 3 instruments require three new entries.

### Terms to add

1. Add Code-Mixing Index, citing Gambäck and Das (2016).
2. Add Language Profiles for the three stratification groups.
3. Add Subword Fertility, reported per profile.

## Part 10: References

1. Existing reference entries stay unchanged.
2. Add entries for Ermino et al. (2022), PAGASA (n.d.), Gambäck and Das (2016), NDRRMC (2021), and OCHA (2024), cited in Parts 2, 7, and 9.
