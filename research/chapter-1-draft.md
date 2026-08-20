# Chapter 1: The Problem and Its Background (Drafting Outline and Guide)

**Note:** This draft outline is old and out of sync with the current research.

This document provides the outline for Chapter 1 of the undergraduate research proposal. This outline translates the approved title defense script and department rubric into an actionable technical roadmap organized by core arguments, concepts, and evidence.

## 1. Title Page and Research Title

- **Approved Title:** **Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets**
- **Core Research Domain:** Multi-task optimization and crisis natural language processing for low-resource code-switched text.
- **Core Concepts:**
  - *Dynamic Multi-Task Balancing:* The optimization mechanism under evaluation.
  - *Targeted Algorithmic Adaptation:* The designed dynamic balancing optimization formulation tailored to mitigate weak-output degradation on code-switched text.
  - *Multilingual Encoder:* The shared backbone architecture evaluated across candidate multilingual encoders (such as Multilingual ModernBERT, mBERT, and XLM-RoBERTa) to ensure generalizability.
  - *Joint Triage:* The concurrent prediction of intent, urgency, and named entities.
  - *Taglish Disaster Tweets:* The target domain of informal Filipino-English code-switched text.

## 2. Introduction to the Study

### Purpose
This section establishes the crisis response problem, defines the three triage tasks, explains the linguistic properties of Taglish, and motivates the need for a balanced multi-task shared encoder on local hardware.

### Thematic Core Arguments
- **Emergency Crisis Response Context:**
  - High volume and time sensitivity of citizen reports on social media during natural disasters in the Philippines.
  - Inability of emergency responders to process, categorize, and locate thousands of incoming messages manually during active crises.
- **The Three Joint Triage Tasks:**
  - *Intent Classification:* Identifies the specific type of help requested (such as rescue, medical aid, food and water, or infrastructure repair).
  - *Urgency Prediction:* Determines the severity and time-criticality of the report to rank life safety priorities.
  - *Named Entity Recognition (NER):* Extracts span boundaries for critical people, locations, and resources mentioned in the text.
- **Linguistic Complexities of Taglish:**
  - Informal alternation between Tagalog and English within sentences and within word boundaries.
  - Morphological affixation (such as combining Tagalog prefixes with English stems in words like "ma-evacuate") causing subword fragmentation in tokenizers.
- **Engineering and Optimization Trade-Off:**
  - *Separate Pipelines:* Running three independent transformer models sequentially triples memory use and repeats forward computations.
  - *Shared Multi-Task Encoder:* Running one shared encoder processes each message once, enabling practical edge CPU execution.
  - *Optimization Conflict:* Sentence-level tasks (intent, urgency) require holistic semantic representations across languages, while token-level NER requires exact subword representations. During joint training, task gradients conflict, causing dominant tasks to degrade auxiliary tasks.
- **Proposed Research Contribution:**
  - Design of an adapted dynamic multi-task balancing algorithm tailored for code-switched disaster triage, and controlled empirical benchmarking against universal multi-task learning baselines across candidate multilingual encoders on edge CPU hardware.

## 3. Background of the Study

### Purpose
This section establishes the research gap using the Macro-Meso-Micro funnel model, and this section supports the argument with published literature, infrastructural realities, and local hardware constraints.

### 3.1 Macro Level: Global Trends in Multi-Task Optimization and Crisis NLP
- **Viability of Joint Crisis NLP:**
  - Recent research demonstrates that multi-task categorization from crisis text is feasible (such as CrisisSense-LLM by Yin et al., 2026).
- **Code-Switched Multi-Tasking Challenges:**
  - A comprehensive survey of over 300 code-switching studies shows that multi-task learning on mixed-language text risks negative transfer and requires careful task balancing (Sheth et al., 2026).
  - Existing code-switched studies rely on auxiliary tasks rather than dynamic loss balancing (Mazumder et al., 2025) or report conditional findings (Adouane and Bernardy, 2020).
- **Dynamic Optimization Formulations:**
  - *Loss-Balancing Methods:* Uncertainty Weighting (Kendall et al., 2018) and GradNorm (Chen et al., 2018).
  - *Gradient Surgery and Multi-Objective Methods:* PCGrad (Yu et al., 2020), CAGrad (Liu et al., 2021), and Nash-MTL.
- **The Macro Gap:**
  - Universal dynamic balancing algorithms lack adaptations for the severe gradient conflict created by mixed sentence-level and token-level tasks on code-switched text, and no published study designs an adapted dynamic balancing strategy for crisis triage.

### 3.2 Meso Level: Philippine Infrastructure, Linguistic Context, and Edge Constraints
- **Linguistic Bottleneck in Crisis Text:**
  - Morphological blending in Taglish breaks standard tokenizer vocabularies, creating subword fragments that increase sequence length and representation difficulty.
- **Infrastructural Vulnerability:**
  - Severe typhoons frequently disable telecommunication networks and power grids, rendering cloud-hosted language model APIs inaccessible at disaster ground zero.
- **Local Edge Hardware Constraints:**
  - Local disaster management offices (LDRRMOs) operate standard office desktop computers and laptops without dedicated GPU servers.
- **Regulatory Privacy Compliance:**
  - Republic Act 10173 (Philippine Data Privacy Act of 2012) mandates local protection of citizen personal data, requiring on-premise execution for emergency messages containing names and addresses.

### 3.3 Micro Level: Dataset Scarcity, Hardware Testbed, and Baseline Deficits
- **Scarcity of Joint Disaster Annotations:**
  - Existing Philippine language datasets (such as TweetTaglish by Herrera et al., 2022 and Batayan by Montalan et al., 2025) lack disaster triage annotations.
  - Existing Philippine disaster studies focus on single-task analysis without NER or joint triage (Imperial and Orosco, 2019, Ermino et al., 2021, Barba et al., 2021, and Livelo and Cheng, 2018).
- **Hardware Testbed Specifications:**
  - The local testbed consists of two consumer laptops equipped with Intel Core i5-13500HX processors and 32 GB DDR5 RAM running Python on CPU.
- **Baseline Architectural Definitions:**
  - *Baseline 1 (Standalone Pipelines):* Three separate single-task encoders executed sequentially.
  - *Baseline 2 (Static Multi-Task):* One shared encoder trained with static equal loss weights.
  - *Expected Deficit:* Static equal weighting induces gradient interference between sentence-level tasks and token-level NER, leading to negative transfer on the weakest head.

## 4. Statement of the Problem

### General Problem Statement
Automated emergency triage of Taglish disaster tweets requires extracting intent, assessing urgency, and identifying named entities simultaneously on local CPU hardware. While separate single-task models triple computational overhead, a shared multilingual encoder experiences gradient conflict between sentence-level and token-level tasks. Static loss weighting leads to negative transfer, and empirical benchmarks evaluating dynamic multi-task balancing algorithms on code-switched crisis text do not exist.

### Interrogative Research Questions
- **SOP 1 (Baseline Analysis):** What are the predictive performance scores (Intent Macro F1, Urgency Macro F1, NER Span F1) and computational execution profiles (single-pass latency in milliseconds, peak memory in megabytes, throughput in tweets per second) of standalone single-task encoders and a static equally weighted multi-task encoder on Taglish disaster tweets?
- **SOP 2 (Design and Integration):** How can dynamic loss and gradient balancing algorithms (Uncertainty Weighting, GradNorm, PCGrad, CAGrad, and Nash-MTL) be integrated into a unified training pipeline for candidate multilingual Transformer encoders?
- **SOP 3 (Empirical Benchmarking across Event Splits):** To what extent do dynamic multi-task balancing algorithms mitigate negative transfer, reduce gradient conflict angles, and balance gradient scale across intent classification, urgency prediction, and named entity recognition across unseen disaster event splits?
- **SOP 4 (Targeted Adaptation and Edge Validation):** How does a targeted dynamic balancing adaptation designed for the consistently weakest output improve predictive accuracy and maintain edge CPU execution efficiency compared to standalone single-task baselines?

## 5. Research Objectives

### Declarative Objectives Mapped to DSR Phases
- **Phase 1: Baseline Analysis (Paired with SOP 1):**
  - The study measures and establishes quantitative baseline performance benchmarks (Intent Macro F1, Urgency Macro F1, NER Span F1) and edge computational profiles (latency in milliseconds, memory in megabytes, throughput in tweets per second) using standalone single-task encoders and a static equally weighted multi-task baseline on Taglish disaster tweets.
- **Phase 2: Design and Artifact Creation (Paired with SOP 2):**
  - The study constructs an annotated Taglish disaster tweet multi-task dataset labeled jointly for intent, urgency, and named entities, and the study builds a unified multi-task training framework that evaluates dynamic loss and gradient balancing algorithms across candidate multilingual encoders.
- **Phase 3: Empirical Benchmarking (Paired with SOP 3):**
  - The study executes controlled empirical benchmarking of dynamic multi-task balancing methods across disaster event splits, and the study records per-task macro scores, gradient conflict angles, and negative transfer metrics under an identical training protocol.
- **Phase 4: Comparative Validation (Paired with SOP 4):**
  - The study designs a targeted multi-task balancing adaptation that prevents the consistently weakest output from degrading, tests the improvement through statistical significance testing (p < 0.05), and benchmarks execution efficiency within a functional local disaster triage application on edge CPU hardware.

## 6. Theoretical Framework

### Theoretical Anchors
- **Design Science Research Methodology (Peffers et al., 2007):**
  - Guides the structured iteration from problem identification and artifact creation (dataset, training framework, triage application) to empirical evaluation and validation.
- **Multi-Task Optimization and Pareto Optimality Theory (Sener and Koltun, 2018, Chen et al., 2018):**
  - Frames multi-task training as a multi-objective optimization problem where task gradients may point in conflicting directions (cosine similarity < 0).
  - Defines balance as reaching Pareto-stationary points where no task head can improve without degrading another.
- **Computational Complexity and Resource Allocation Theory:**
  - Applies Big-O notation to evaluate memory spatial complexity (O(M)) and inference time complexity (O(T)) of a shared single-pass model (O(1) passes) compared to separate single-task pipelines (O(K) passes for K tasks).

## 7. Conceptual Framework (Input-Process-Output)

### Input
- **Dataset:** Curated Taglish disaster tweet dataset annotated jointly for Intent (categorical aid needs), Urgency (ordinal priority levels), and Named Entities (BIO token spans).
- **Model Parameters:** Pretrained weights and subword tokenizer vocabularies from candidate multilingual Transformer encoders (such as Multilingual ModernBERT, mBERT, and XLM-RoBERTa).
- **Hardware Environment:** Local edge CPU testbed (Intel Core i5-13500HX, 32 GB RAM, standard CPU execution without GPU).

### Process
- **Domain-Adaptive Continued Pre-Training:** Continuing self-supervised pre-training on unlabeled Taglish disaster tweets to adapt candidate encoders to code-switched subword fragmentation.
- **Unified Multi-Task Training Loop:** Passing tokenized tweets through the shared encoder once and predicting all three triage outputs simultaneously.
- **Controlled Balancing Protocol:** Isolating loss computation and applying dynamic balancing algorithms (Uncertainty Weighting, GradNorm, PCGrad, CAGrad, and Nash-MTL).
- **Targeted Weak-Output Adaptation:** Identifying the consistently weakest task across repeated event splits and designing an optimization adaptation that prevents its degradation.
- **Edge Model Optimization:** Optimizing the final checkpoint for local CPU inference execution.

### Output
- **Algorithmic Artifact:** Adapted dynamic multi-task balancing optimization formulation designed to prevent weak-output degradation on code-switched text.
- **Model Checkpoint:** Trained shared multilingual encoder artifact optimized for Taglish disaster triage.
- **Software Artifact:** Functional local disaster triage pipeline and web responder dashboard.
- **Empirical Metrics:**
  - *Per-Task Predictive Scores:* Intent Macro F1, Urgency Macro F1, and Named Entity Recognition Span F1.
  - *Combined Multi-Task Score:* Macro average of the three task F1 scores.
  - *Negative Transfer:* Percentage difference in predictive performance relative to standalone single-task baselines.
  - *Switch-Point Accuracy:* Predictive accuracy on tokens at language switching boundaries.
  - *Gradient Diagnostics:* Gradient scale ratios and gradient conflict angles (cosine similarity).
  - *Computational Metrics:* Single-pass inference latency in milliseconds, peak RAM consumption in megabytes, and processing throughput in tweets per second.
  - *Statistical Significance:* Paired statistical tests across repeated seeds and event splits (p-value threshold alpha = 0.05).

## 8. Scope and Delimitations

### Scope
- **Domain Focus:** Social media posts published during natural disasters in the Philippines (typhoons, floods, and earthquakes).
- **Language Focus:** Informal Taglish (Tagalog-English code-switched text) with intra-sentential switching and morphological blending.
- **Triage Tasks:** Three concurrent triage heads: Intent Classification, Urgency Prediction, and Named Entity Recognition.
- **Model Architecture:** Candidate multilingual Transformer encoders (such as Multilingual ModernBERT, mBERT, and XLM-RoBERTa) sharing all backbone layers across the three task heads.
- **Evaluation Partitioning:** Partitioning datasets by disaster event to evaluate model generalization to unseen crisis events.
- **Hardware Testbed:** Local CPU execution on consumer laptop hardware (Intel Core i5-13500HX with 32 GB RAM).

### Delimitations of the Study
- **Cloud Language Model APIs:** Excluded due to recurring API costs, network latency, internet outage risks during disasters, and data privacy regulations.
- **Non-Disaster Corpora:** Excluded from evaluation to maintain strict focus on the crisis triage domain.
- **Non-Text Modalities:** Image, video, and audio attachments are excluded from the natural language processing pipeline.
- **Auxiliary Syntactic Tasks:** Syntactic dependency parsing and explicit language identification are excluded from the multi-task heads.

## 9. Significance of the Study

### Beneficiary Impact
- **Primary Beneficiaries (Emergency Response Agencies):**
  - Provides the NDRRMC, LDRRMOs, and local response units with an automated, offline-capable triage system.
  - Accelerates situational awareness and resource allocation during active disasters without recurring cloud API fees.
- **Secondary Beneficiaries (Computer Science and NLP Researchers):**
  - Delivers the first public Taglish disaster tweet multi-task dataset with joint intent, urgency, and named entity annotations.
  - Contributes an adapted dynamic multi-task balancing formulation tailored to mitigate gradient conflict and negative transfer on code-switched text.
  - Establishes empirical evidence on dynamic multi-task gradient balancing behavior on informal code-switched text.
- **Tertiary Beneficiaries (Disaster-Affected Communities):**
  - Enables faster emergency dispatch and rescue coordination during disaster events.
  - Protects citizen privacy under Republic Act 10173 by ensuring sensitive reports remain on local on-premise hardware.

## 10. Definition of Terms

### Operational Definitions
- **Code-Switching (Taglish):** The alternation between Tagalog and English within a sentence or word, creating morphological blends (such as "ma-evacuate") that undergo subword fragmentation during tokenization.
- **Disaster Event Split:** An evaluation partitioning method where training, validation, and test sets contain distinct disaster events to prevent event-specific vocabulary leakage.
- **Dynamic Multi-Task Balancing:** Optimization algorithms that automatically adjust loss weights or adjust gradient vectors during training to resolve task competition in a shared encoder.
- **Gradient Conflict:** A training condition where gradient vectors of two task heads point in opposing directions (cosine similarity < 0), causing parameter updates from one task to degrade another.
- **Hard Parameter Sharing:** An architecture where all task heads share identical transformer encoder layers, diverging only at task-specific classification and sequence labeling heads.
- **Intent Classification:** The sequence-level prediction head that classifies the help category requested in a tweet.
- **Joint Triage:** The concurrent prediction of intent, urgency level, and named entities from a single disaster tweet in one forward pass.
- **Named Entity Recognition (NER):** The token-level sequence labeling head that extracts span boundaries for people, locations, and resources.
- **Negative Transfer:** The performance loss experienced when training tasks jointly in a shared network compared to training an isolated single-task baseline model.
- **Peak Memory Footprint (M_RAM):** The peak physical RAM in megabytes consumed by the inference process during model execution on CPU hardware.
- **Single-Pass Inference Latency:** The wall-clock execution time in milliseconds required for the shared model to compute all three outputs for a single tweet on CPU hardware.
- **Throughput:** The number of disaster tweets processed and categorized per second by the system on local CPU hardware.
- **Uncertainty Weighting (UW):** A dynamic multi-task loss-balancing method that scales task losses based on learned homoscedastic uncertainty parameters.
- **Urgency Prediction:** The sequence-level prediction head that evaluates the severity and immediacy of reported emergency situations.
- **Weakest Output:** The specific triage head whose predictive score experiences the greatest degradation relative to its standalone single-task baseline across repeated runs and event splits.
