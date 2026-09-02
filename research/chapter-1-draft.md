# Chapter 1 - The Problem and Its Background

## 1. Proposed Title

<!-- vale off -->
Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets
<!-- vale on -->

## 2. Introduction to the Study

<!-- vale off -->
During natural disasters such as typhoons and severe floods, emergency response agencies rely on real-time social media messages to coordinate rescue, assess infrastructure damage, and distribute humanitarian relief. Extracting critical information from these high-volume message streams requires natural language processing (NLP) models to execute three concurrent tasks, namely token-level Named Entity Recognition (NER) for critical locations and infrastructure, sequence-level intent classification for humanitarian needs, and sequence-level urgency prioritization for life-threatening distress.

Standard NLP deployment pipelines run separate Single-Task Learning (STL) models for each prediction target. This approach scales computational latency and memory consumption linearly with task count ($\mathcal{O}(K)$ for $K$ tasks), exceeding the processing capabilities of local workstation hardware. In contrast, hard-parameter-sharing Multi-Task Learning (MTL) uses a single multilingual Transformer backbone to generate shared representations in a single forward pass, achieving near-constant inference latency ($\mathcal{O}(1)$ relative to $K$). However, joint multi-task training couples heterogeneous loss functions, producing two primary gradient failure modes during backpropagation, namely gradient magnitude disparities and gradient directional conflicts. On low-resource, code-switched text such as Tagalog-English (Taglish), these optimization bottlenecks often trigger negative transfer, causing joint models to underperform isolated single-task baselines.

To resolve these computational and optimization challenges, this study develops an empirical benchmarking framework grounded in the Design Science Research Methodology (DSRM). The investigation benchmarks candidate Multi-Task Optimization (MTO) algorithms (spanning dynamic loss weighting and gradient surgery) against Uniform Equal Weighting (EW), tuned Static Linear Scalarization (LS), and isolated STL baselines across four historical Philippine typhoons under a 4-fold Leave-One-Event-Out (LOEO) evaluation protocol. Using PyTorch gradient diagnostic hooks and an intra-task half-batch gradient baseline, the framework quantifies gradient interference, evaluates out-of-distribution generalization, and deploys the fine-tuned model within a prototype web application for disaster triage.
<!-- vale on -->

## 3. Background of the Study

<!-- vale off -->
### 3.1 Macro Level

Automated disaster triage requires extracting three types of information from incoming crisis messages, namely location entities (Alam et al., 2021), humanitarian intent categories, and urgency levels (Wang et al., 2021). Deploying independent Transformer models for each task multiplies inference latency and parameter storage linearly as $\mathcal{O}(K)$ for $K$ tasks (Ruder, 2017). Conversely, hard parameter sharing passes text through a shared multilingual Transformer encoder to compute intermediate contextual representations, which feed lightweight task-specific linear heads (Chen et al., 2024; Ruder, 2017). This unified architecture maintains near-constant inference latency ($\mathcal{O}(1)$ relative to $K$) on local workstation hardware and introduces an inductive bias that guides shared parameters toward representations supporting all triage tasks simultaneously.

Despite these architectural advantages, joint multi-task training introduces complex multi-objective optimization dynamics across shared parameters (Sener & Koltun, 2018). When task loss gradients point in opposing directions ($\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$), updating shared parameters along task $i$ increases the empirical loss on task $j$ to a first-order approximation, causing destructive interference and negative transfer across shared layers (B. Liu et al., 2021; Navon et al., 2022; Yu et al., 2020). Furthermore, when gradient scales diverge, tasks with larger gradient magnitudes dominate parameter updates and suppress secondary objectives (Chen et al., 2018). While jointly modeling token-level entities and sequence-level intent provides mutual linguistic reinforcement (Weld et al., 2022), realizing this positive transfer ($\Delta_m > 0$) between NER and intent classification requires principled gradient balancing.

To address these optimization bottlenecks, machine learning literature has established two primary Multi-Task Optimization (MTO) approaches, namely dynamic loss weighting and gradient surgery. Dynamic loss-weighting algorithms adjust scalar task loss weights during training using distinct heuristics, where Uncertainty Weighting (Kendall et al., 2018) scales losses by learned homoscedastic uncertainty, GradNorm (Chen et al., 2018) balances gradient magnitudes against relative training paces, and Fast Adaptive Multitask Optimization (FAMO, Liu et al., 2023) equalizes task loss decrease rates. In contrast, gradient-surgery algorithms manipulate gradient vectors directly, where Projecting Conflicting Gradients (PCGrad, Yu et al., 2020) projects conflicting gradients onto orthogonal normal planes, Conflict-Averse Gradient Descent (CAGrad, B. Liu et al., 2021) optimizes the worst-case objective improvement within a trust region, Impartial Multi-Task Learning (IMTL-G, L. Liu et al., 2021) enforces equal-angle descent across objectives, and Nash-MTL (Navon et al., 2022) formulates gradient aggregation as a multi-objective bargaining game.

Despite theoretical motivations, empirical studies show conflicting evidence regarding whether dynamic MTO methods consistently outperform tuned Static Linear Scalarization (LS) or Uniform Equal Weighting (EW). In machine translation and computer vision benchmarks, dynamic loss weights often remain static during training while reducing computational throughput (Xin et al., 2022). Furthermore, under adaptive optimizers like AdamW, gradient magnitude disparities dominate multi-task updates, rendering vector surgery less effective and making Uniform Equal Weighting competitive (Elich et al., 2024). Because modern Transformer fine-tuning relies on AdamW, this study investigates whether these findings hold on low-resource code-switched text with asymmetric token-level and sequence-level objectives. To separate genuine cross-task conflict from stochastic mini-batch noise, the experimental framework computes pairwise gradient angles against an intra-task half-batch baseline (Elich et al., 2024).

### 3.2 Meso Level

In the Philippines, disaster response represents a critical operational priority. The country experiences an average of twenty typhoons annually, with approximately five causing catastrophic damage and flooding (NDRRMC, 2020, 2022). During major typhoon events (such as Typhoon Haiyan in 2013, Typhoon Vamco in 2020, Typhoon Rai in 2021, and Typhoon Paeng in 2022), citizens broadcast thousands of distress messages across social media using civic hashtags such as #RescuePH and #ReliefPH. While disaster agencies like the National Disaster Risk Reduction and Management Council (NDRRMC) and Office of Civil Defense (OCD) monitor these streams, manual triage pipelines become overwhelmed during peak landfall hours.

Tagalog-English code-switching (Taglish) introduces severe linguistic complexity into automated triage. Taglish crisis messages combine Tagalog grammatical affixes and focus markers with English root words (such as *ma-evacuate* or *nag-collapse*), non-standard spelling, abbreviations, and informal colloquialisms (Herrera et al., 2022; Montalan et al., 2025). Multilingual Transformer tokenizers segment these morphologically hybrid words into fragmented subword sequences (inflating sequence length by 2x to 7x), diluting semantic embeddings and elevating gradient variance across shared layers (Petrov et al., 2023). Multi-task training across sequence labeling and sentence classification on code-switched text can trigger severe negative transfer without principled gradient balancing (Adouane & Bernardy, 2020).

Operational constraints in Philippine disaster management further dictate deployment requirements. Local Government Units (LGUs) and municipal command centers frequently operate on commodity workstations subject to power grid failures and telecommunication outages during typhoons. Relying on cloud-hosted Large Language Model (LLM) APIs introduces critical risks, including transmission latency, service loss during internet blackouts, recurring API costs, and compliance risks under the Philippine Data Privacy Act of 2012 (Republic Act No. 10173). Deploying compact, fine-tuned multi-task models locally provides a resilient, self-contained triage capability.

### 3.3 Micro Level

**Table 1**  
*Structural Profile of Related Disaster Informatics and Philippine NLP Corpora*

| **Dataset or Benchmark** | **Domain and Coverage** | **Modality and Corpus Volume** | **Language and Register** | **Annotation Level and Target Tasks** | **Cross-Event Evaluation Protocol** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HumAID (Alam et al., 2021) | Global Crisis Informatics (19 disaster events) | Text only (77,196 tweets) | Monolingual English (Social media) | Sequence-level (10 humanitarian intent categories) | Event-stratified and cross-event splits |
| Batayan (Montalan et al., 2025) | General Philippine NLP Benchmark | Text only (3,800 test instances) | Tagalog and Taglish (Diverse sources) | Task-isolated evaluations across eight separate NLP tasks | Task-specific benchmark splits |
| TweetTaglish (Herrera et al., 2022) | Code-Switching Analysis | Text only (21,150 tweets) | Informal Taglish (Social media) | Document-level language mixing distribution | Random 80/20 train/test split |
| TLUNIFIED-NER (Miranda, 2023) | News and Reference Corpus | Text only (208,247 tokens) | Formal Tagalog (News articles) | Token-level sequence labeling (BIO entity spans) | Standard entity evaluation splits |
| Proposed Multi-Task Corpus | Philippine Disaster Triage (4 typhoon events) | Text only (6,000 to 10,000 tweets) | Informal Taglish (Social media) | Concurrent token-level BIO spans, 4-class Intent, and 3-tier Urgency | 4-fold Leave-One-Event-Out (LOEO) |

As summarized in Table 1, existing benchmarks exhibit clear architectural limitations. Global crisis corpora like HumAID (Alam et al., 2021) provide large-scale intent annotations but remain restricted to monolingual English and lack token-level entity spans. Philippine NLP benchmarks (such as Batayan, TweetTaglish, and TLUNIFIED-NER) evaluate single tasks in isolation or focus on formal news text. No existing public corpus provides concurrent annotations for token-level named entities, sequence-level intent, and sequence-level urgency on Taglish crisis messages under cross-event evaluation.

This data void directly impairs multi-task optimization research for crisis triage. In a shared Transformer encoder, token-level NER aggregates cross-entropy loss across all subword tokens in a sequence, whereas intent and urgency classification compute loss over a single pooled representation. As a result, token labeling gradients systematically dominate backpropagation ($\|\mathbf{g}_{\text{NER}}\|_2 \gg \|\mathbf{g}_{\text{Intent}}\|_2, \|\mathbf{g}_{\text{Urgency}}\|_2$, as characterized by Chen et al., 2018). Under Uniform Equal Weighting, this structural asymmetry allows token-level updates to overwrite pooled sequence representations, inducing negative transfer.

To address these empirical and optimization challenges, this study curates the Multi-Task Corpus of Taglish Disaster Tweets across four Philippine typhoons under a 4-fold LOEO protocol. The study rigorously evaluates dynamic loss weighting and gradient surgery against baseline methods, providing empirical guidance and an operational triage artifact for crisis informatics.
<!-- vale on -->

## 4. Statement of the Problem

<!-- vale off -->
Real-time disaster triage requires models to process text across token-level Named Entity Recognition (NER), sequence-level intent classification, and sequence-level urgency classification. Running separate Single-Task Learning (STL) models scales computational latency and parameter memory linearly ($\mathcal{O}(K)$ for $K$ tasks), exceeding the memory and compute budgets of local workstations (Ruder, 2017). A shared multilingual Transformer encoder processes all three tasks in a single forward pass ($\mathcal{O}(1)$ relative to $K$), but joint optimization couples token-level and sequence-level loss functions. This coupling induces gradient magnitude disparities and directional conflicts ($\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$) that degrade shared representations and cause negative transfer (B. Liu et al., 2021; Yu et al., 2020).

Multi-Task Optimization (MTO) algorithms (spanning dynamic loss weighting and gradient surgery) aim to resolve these gradient conflicts. However, prior empirical studies report conflicting evidence regarding whether dynamic balancing consistently outperforms Static Linear Scalarization (LS) or Uniform Equal Weighting (EW) under adaptive optimizers like AdamW (Elich et al., 2024; Xin et al., 2022). In addition, stochastic variance across mini-batch samples can produce opposing gradient vectors that confound empirical measurements of genuine cross-task conflict (Elich et al., 2024). Furthermore, crisis informatics lacks unified code-switched Taglish corpora providing concurrent token-level and sequence-level annotations under cross-event evaluation.

To resolve these challenges, this study addresses four research questions aligned with the Design Science Research lifecycle.

**Phase 1 (Baseline Analysis):** What are the baseline predictive F1 scores (intent macro F1, urgency macro F1, and span-level NER F1), single-instance latency ($t_{\text{exec}}$ at $B=1$), throughput ($B=32$), and memory footprints ($M_{\text{VRAM}}$, $M_{\text{RAM}}$) of isolated Single-Task Learning models, and how do these metrics compare to a shared multilingual Transformer encoder trained under Uniform Equal Weighting (EW) across 4-fold Leave-One-Event-Out (LOEO) disaster splits in predictive accuracy, gradient norm ratios ($R_{\text{max/min}}$), and pairwise gradient cosine similarities?

**Phase 2 (Design and Artifact Creation):** How can the hard-parameter-sharing pipeline, candidate Multi-Task Optimization algorithms, and PyTorch gradient diagnostic routines be designed to mitigate gradient interference on Taglish disaster text, and how can the fine-tuned model be integrated into a prototype web application for disaster triage?

**Phase 3 (Empirical Benchmarking):** How do candidate Multi-Task Optimization algorithms, comprising dynamic loss weighting (Uncertainty Weighting, GradNorm, and FAMO) and gradient surgery (PCGrad, CAGrad, IMTL-G, and Nash-MTL), compare against isolated Single-Task Learning, Uniform Equal Weighting (EW), and Static Linear Scalarization (LS) across 4-fold Leave-One-Event-Out splits in predictive F1 scores, relative multi-task transfer ($\Delta_m$), computational efficiency ($t_{\text{exec}}$, throughput, $M_{\text{VRAM}}$, $M_{\text{RAM}}$), and training runtime?

**Phase 4 (Comparative Validation):** To what extent do performance differences among candidate Multi-Task Optimization algorithms demonstrate statistical significance under a two-tier non-parametric validation framework (paired bootstrap confidence intervals and omnibus Friedman-Nemenyi tests), and how strongly does relative multi-task transfer ($\Delta_m$) correlate with empirical gradient conflict metrics relative to the intra-task half-batch baseline?
<!-- vale on -->

## 5. Objectives of the Study

<!-- vale off -->
The primary objective of this study is to construct the Multi-Task Corpus of Taglish Disaster Tweets, benchmark candidate Multi-Task Optimization algorithms using PyTorch gradient diagnostics under a 4-fold Leave-One-Event-Out cross-validation protocol, and deploy the fine-tuned multilingual Transformer model into a prototype web application for disaster triage.

Aligned with the four Design Science Research (DSR) phases, this study pursues four specific research objectives.

**Phase 1 (Baseline Analysis):** To establish baseline performance by measuring predictive F1 scores (intent macro F1, urgency macro F1, and span-level NER F1), single-instance latency ($t_{\text{exec}}$ at $B=1$), throughput ($B=32$), and memory footprints ($M_{\text{VRAM}}$, $M_{\text{RAM}}$) across isolated Single-Task Learning models, and to profile the predictive accuracy, gradient norm ratios ($R_{\text{max/min}}$), and pairwise gradient cosine similarities of a shared multilingual Transformer encoder trained under Uniform Equal Weighting (EW) across 4-fold Leave-One-Event-Out (LOEO) disaster splits.

**Phase 2 (Design and Artifact Creation):** To design and implement the hard-parameter-sharing multi-task training pipeline, candidate Multi-Task Optimization algorithms, and PyTorch gradient diagnostic routines to mitigate gradient interference on Taglish disaster text, and to integrate the optimized model into a prototype web application for disaster triage.

**Phase 3 (Empirical Benchmarking):** To benchmark candidate Multi-Task Optimization algorithms, comprising dynamic loss weighting (Uncertainty Weighting, GradNorm, and FAMO) and gradient surgery (PCGrad, CAGrad, IMTL-G, and Nash-MTL), against isolated Single-Task Learning, Uniform Equal Weighting (EW), and Static Linear Scalarization (LS) across 4-fold Leave-One-Event-Out splits in predictive F1 scores, relative multi-task transfer ($\Delta_m$), computational efficiency ($t_{\text{exec}}$, throughput, $M_{\text{VRAM}}$, $M_{\text{RAM}}$), and training runtime.

**Phase 4 (Comparative Validation):** To evaluate the statistical significance of performance differences across candidate Multi-Task Optimization algorithms using a two-tier non-parametric validation framework (paired bootstrap confidence intervals and omnibus Friedman-Nemenyi tests), and to analyze the Spearman rank correlation ($\rho$) between relative multi-task transfer ($\Delta_m$) and empirical gradient conflict metrics relative to the intra-task half-batch baseline.
<!-- vale on -->

## 6. Theoretical Framework

<!-- vale off -->
This study is grounded in three theoretical foundations, namely Design Science Research Methodology, computational complexity analysis of shared representations, and multi-objective optimization with gradient dynamics.

First, this study adopts the Design Science Research Methodology (DSRM) established by Peffers et al. (2007) and the artifact evaluation guidelines formulated by Hevner et al. (2004). Unlike behavioral sciences that focus on observing and predicting natural phenomena, design science in computer science produces new scientific knowledge through the systematic design, implementation, and empirical evaluation of computational artifacts. Hevner et al. (2004) establish that rigorous design science research requires systematic evaluation of artifact utility, problem relevance, and verifiable technical contributions. Peffers et al. (2007) formulated this inquiry as an iterative six-activity lifecycle comprising (1) Problem Identification and Motivation, (2) Definition of the Objectives for a Solution, (3) Design and Development, (4) Demonstration, (5) Evaluation, and (6) Communication. To operationalize this methodology, the study structures research activities into four sequential phases, namely Baseline Analysis (Phase 1), Design and Artifact Creation (Phase 2), Empirical Benchmarking (Phase 3), and Comparative Validation (Phase 4). Under this foundation, the primary intellectual contribution resides in the empirical multi-task optimization knowledge and curated benchmark corpus, while the web application prototype serves as an operational demonstration artifact.

Second, this study applies asymptotic time and space complexity analysis alongside the multi-task representation sharing and inductive bias principles formalized by Ruder (2017). In single-task learning architectures, deploying $K$ independent models to process $K$ natural language processing tasks requires $K$ separate forward passes for each input tweet. The computational inference latency scales linearly with task count as $\mathcal{O}(K \cdot T_{\text{sh}})$, where $T_{\text{sh}}$ denotes the forward pass execution time of the Transformer encoder backbone. Similarly, maintaining $K$ concurrent models in memory requires $\mathcal{O}(K \cdot |\boldsymbol{\theta}_{\text{sh}}|)$ parameter storage, where $|\boldsymbol{\theta}_{\text{sh}}|$ denotes encoder parameter cardinality. In contrast, hard parameter sharing computes intermediate features once in a single shared forward pass. Total forward pass inference complexity is formalized as

$$\mathcal{O}\left(T_{\text{sh}} + \sum_{k=1}^K T_{\text{head}, k}\right)$$

where the backbone forward computation heavily dominates linear classification layers ($T_{\text{sh}} \gg \sum_{k=1}^K T_{\text{head}, k}$), yielding an asymptotic execution time independent of task count $K$ ($\mathcal{O}(1)$ relative to $K$). Parameter memory allocation is restricted to $\mathcal{O}(|\boldsymbol{\theta}_{\text{sh}}| + \sum_{k=1}^K |\boldsymbol{\theta}_{\text{head}, k}|)$, where task-specific linear classification layers contain negligible parameter volume relative to the shared backbone ($|\boldsymbol{\theta}_{\text{head}, k}| \ll |\boldsymbol{\theta}_{\text{sh}}|$). As a result, hard parameter sharing significantly reduces inference latency and memory footprints compared to isolated pipelines. Furthermore, learning shared intermediate representations across related tasks restricts the parameter search space to representations that simultaneously support entity extraction, intent classification, and urgency assessment, formalizing an inductive bias that reduces sample complexity and enhances generalization on code-switched Taglish text.

Third, this study is anchored in Multi-Objective Optimization (Sener & Koltun, 2018) and Gradient Dynamics Principles (Chen et al., 2018; Elich et al., 2024; Kendall et al., 2018; B. Liu et al., 2021; Xin et al., 2022; Yu et al., 2020). In a hard-parameter-sharing neural network with shared encoder parameters $\boldsymbol{\theta}_{\text{sh}}$ and task-specific classification heads $\boldsymbol{\theta}_{\text{head}, k}$, joint training represents a multi-objective optimization problem over vector-valued task losses, formulated as

$$\min_{\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, 1}, \dots, \boldsymbol{\theta}_{\text{head}, K}} \left( \mathcal{L}_1(\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, 1}), \mathcal{L}_2(\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, 2}), \dots, \mathcal{L}_K(\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, K}) \right)$$

Multi-objective optimization theory establishes that multi-task learning seeks a Pareto-optimal parameter configuration where no individual task loss can decrease without increasing the loss of another task. When updating shared parameters $\boldsymbol{\theta}_{\text{sh}}$ through gradient descent with learning rate $\eta$, let $\mathbf{g}_k \equiv \nabla_{\boldsymbol{\theta}_{\text{sh}}} \mathcal{L}_k(\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, k})$ denote the gradient of task loss $k$ with respect to shared parameters. Assuming local smoothness of $\mathcal{L}_j$, applying a first-order Taylor expansion to the loss of task $j$ under a parameter step along task $i$ yields

$$\mathcal{L}_j(\boldsymbol{\theta}_{\text{sh}} - \eta \mathbf{g}_i, \boldsymbol{\theta}_{\text{head}, j}) = \mathcal{L}_j(\boldsymbol{\theta}_{\text{sh}}, \boldsymbol{\theta}_{\text{head}, j}) - \eta \langle \mathbf{g}_i, \mathbf{g}_j \rangle + \mathcal{O}(\eta^2 \|\mathbf{g}_i\|_2^2)$$

This mathematical formulation identifies two primary failure modes in shared parameter optimization, where the linear inner product provides a first-order approximation under bounded learning rates (Yu et al., 2020), with higher-order curvature terms bounded by $\mathcal{O}(\eta^2 \|\mathbf{g}_i\|_2^2)$.

**Disparity in Gradient Magnitudes:** When tasks operate at asymmetric prediction levels (token-level sequence labeling versus sequence-level classification), gradient magnitudes diverge ($\|\mathbf{g}_i\|_2 \gg \|\mathbf{g}_j\|_2$). The dominant task gradient dictates the trajectory of parameter updates, causing tasks with smaller gradient norms to learn slowly or suboptimally (Chen et al., 2018).

**Gradient Directional Conflict:** When the inner product between task gradients is negative ($\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$, or cosine similarity $\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$), the first-order term $-\eta \langle \mathbf{g}_i, \mathbf{g}_j \rangle$ becomes positive. This condition indicates, to a first-order approximation, that parameter updates along task $i$ increase the empirical loss on task $j$, inducing destructive interference and negative transfer across shared layers (Yu et al., 2020).

Specialized Multi-Task Optimization (MTO) algorithms address these challenges through dynamic loss weighting or gradient surgery techniques that project conflicting gradient vectors onto orthogonal hyperplanes or compute impartial descent directions. Furthermore, following the intra-task gradient diagnostic principle of Elich et al. (2024), the framework incorporates an intra-task baseline derived from half-batch gradients computed across disjoint sample subsets to distinguish stochastic mini-batch variance from genuine cross-task conflict. Statistical validity is established through a two-tier non-parametric testing framework (Demšar, 2006) combining per-fold paired bootstrap confidence intervals with omnibus Friedman-Nemenyi ranking. To translate these theoretical principles into an operational experimental workflow, the study adopts the Input-Process-Output conceptual framework.
<!-- vale on -->

## 7. Conceptual Framework

<!-- vale off -->
This study adopts the Input-Process-Output (IPO) model to structure the experimental design, algorithmic implementation, and empirical evaluation across the four Design Science Research phases. Figure 1 outlines the conceptual framework of the study, mapping the input datasets, model architectures, optimization configurations, and testbed specifications through structured preprocessing, multi-task benchmarking, statistical validation, and service integration processes to produce four concrete research deliverables, namely the trained multi-task model, validated empirical benchmark tables, statistical hypothesis test results, and the interactive disaster triage web prototype.

**Figure 1**  
*Input-Process-Output (IPO) Conceptual Framework of the Study*

| **Input** | **Process** | **Output** |
|:---|:---|:---|
| **I1. Datasets and Encoders**<br>• Multi-Task Taglish Disaster Corpus (4-Fold LOEO, BIO Spans, Intent, Urgency)<br>• Multilingual Transformer Encoders (XLM-R, mBERT, SEA-LION ModernBERT, RoBERTa-Tagalog) | **P1. Data and Training Pipeline**<br>• Text Normalization and Multilingual Subword Tokenization<br>• BIO Subword Alignment and LOEO Dataset Partitioning<br>• Hard-Parameter-Sharing Multi-Task Fine-Tuning | **O1. Trained Multi-Task Model**<br>• Fine-Tuned Multilingual Transformer Backbone<br>• Concurrent Multi-Task Heads (NER, Intent, Urgency)<br>• $\mathcal{O}(1)$ Single-Pass Joint Inference |
| **I2. Optimization and Profiling Testbed**<br>• Candidate MTO Algorithms (UW, GradNorm, FAMO, PCGrad, CAGrad, IMTL-G, Nash-MTL)<br>• Baseline Configurations (STL, EW, Static LS)<br>• Dedicated Local Workstation Testbed | **P2. Standardized Benchmarking and Diagnostics**<br>• FP16 Execution (Seq Len 128, Batch 32 Training)<br>• Single-Instance Latency ($B=1$) and Throughput ($B=32$)<br>• Peak VRAM ($M_{\text{VRAM}}$) and Host RAM ($M_{\text{RAM}}$) Profiling<br>• PyTorch Gradient Diagnostic Hooks and Intra-Task Baseline | **O2. Validated Empirical Benchmark Tables**<br>• Task F1 Scores (Intent Macro, Urgency Macro, NER Span)<br>• Relative Multi-Task Transfer ($\Delta_m$)<br>• Latency ($t_{\text{exec}}$), Throughput, Peak VRAM, Host RAM<br>• Gradient Norm Disparities and Cosine Similarity Logs |
| **I3. Empirical Benchmark Observations**<br>• Paired Evaluation Runs ($N=12$: 4 Folds $\times$ 3 Seeds)<br>• Instance-Level Test Predictions per Fold<br>• Gradient Conflict and Intra-Task Baseline Logs | **P3. Two-Tier Non-Parametric Validation**<br>• Tier 1: Paired Bootstrap 95% CIs per Held-Out Fold<br>• Tier 2: Omnibus Friedman and Iman-Davenport Tests<br>• Post-Hoc Nemenyi Critical Difference Ranking<br>• Spearman Rank Correlation ($\rho$) on Transfer Dynamics | **O3. Statistical Test Results**<br>• Primary Inferential Bootstrap 95% Confidence Intervals<br>• Omnibus Statistics ($\chi^2_F$, $F_F$, $p < 0.05$) and Nemenyi Ranks<br>• Monotonic Correlation Coefficients ($\rho$) for Conflict Dynamics |
| **I4. Software Stack and Application Assets**<br>• Fine-Tuned Multi-Task Transformer Model<br>• Local Philippine Gazetteer Mapping Dictionary<br>• Local Application Environment (FastAPI, Relational DB) | **P4. Service Integration and Deployment**<br>• Asynchronous FastAPI Web Service Integration<br>• Disaster Event Replay Stream and Manual Ingestion<br>• Local Spatial Mapping with Unmapped Fallback<br>• 4-Stage Incident Lifecycle State Management | **O4. Prototype Web Application**<br>• Asynchronous Multi-Task Triage Web Backend<br>• Relational Incident Store (SQLite / PostgreSQL)<br>• Interactive 4-Stage Incident Management Dashboard |

`#text(size: 10pt)[#emph[Note.] The framework flows left to right: Input \u{2192} Process \u{2192} Output, aligned across the four sequential Design Science Research phases.]`{=typst}

### 7.1 Input Stage

The input stage incorporates four core resources (I1 to I4), comprising (1) the Multi-Task Corpus of Taglish Disaster Tweets partitioned across four historical typhoon events, (2) four candidate multilingual Transformer encoders (XLM-RoBERTa, multilingual BERT, SEA-LION ModernBERT, and RoBERTa-Tagalog), (3) optimization configurations spanning dynamic loss weighting and gradient surgery benchmarked against single-task and static baselines, and (4) local workstation testbed hardware running under FP16 mixed precision.

### 7.2 Process Stage

The process stage executes four structured workflows (P1 to P4), comprising (1) text normalization, subword tokenization, BIO span alignment, and LOEO data splitting, (2) multi-task fine-tuning and PyTorch gradient diagnostic logging against an intra-task half-batch baseline, (3) two-tier non-parametric statistical validation combining per-fold paired bootstrap confidence intervals with omnibus Friedman-Nemenyi ranking, and (4) embedding the fine-tuned model into an asynchronous FastAPI service with spatial gazetteer resolution.

### 7.3 Output Stage

The output stage delivers four concrete research artifacts (O1 to O4), comprising (1) a trained hard-parameter-sharing multi-task model executing concurrent predictions in $\mathcal{O}(1)$ time, (2) validated empirical benchmark tables reporting task F1 scores, relative transfer ($\Delta_m$), computational efficiency, and gradient conflict dynamics across 12 paired evaluation runs, (3) inferential bootstrap confidence intervals and non-parametric hypothesis test results, and (4) a prototype web application for disaster triage supporting 4-stage incident lifecycle management.
<!-- vale on -->

## 8. Scope and Delimitations

<!-- vale off -->
### 8.1 Scope of the Study

This study focuses on constructing the Multi-Task Corpus of Taglish Disaster Tweets, benchmarking candidate Multi-Task Optimization (MTO) algorithms within a shared multilingual Transformer encoder, and deploying the optimized model into a prototype web application for disaster triage. The experimental dataset comprises a target volume of 6,000 to 10,000 code-switched Taglish crisis tweets (with a verified baseline of at least 6,000 annotated instances) collected across four historical Philippine typhoons (Haiyan, Vamco, Rai, and Paeng), evaluated under a 4-fold Leave-One-Event-Out (LOEO) cross-validation protocol across three random initialization seeds (seeds 42, 123, and 456).

The architectural scope evaluates encoder-only multilingual Transformer backbones (XLM-RoBERTa Base, multilingual BERT Base, SEA-LION ModernBERT, and RoBERTa-Tagalog Base) using hard parameter sharing across intermediate representations, connected to linear heads for token classification and sequence classification. The optimization benchmark evaluates two algorithmic families, comprising dynamic loss-weighting methods (Uncertainty Weighting, GradNorm, and FAMO) and gradient-surgery techniques (PCGrad, CAGrad, IMTL-G, and Nash-MTL), benchmarked against isolated Single-Task Learning (STL), Uniform Equal Weighting (EW), and Static Linear Scalarization (LS).

Evaluation criteria include task F1 scores (intent macro F1, urgency macro F1, and CoNLL span F1 for NER), relative multi-task transfer ($\Delta_m$), unbatched latency ($t_{\text{exec}}$ at $B=1$), batched throughput ($B=32$), peak GPU memory ($M_{\text{VRAM}}$), host RAM ($M_{\text{RAM}}$), and training runtime. PyTorch gradient diagnostic hooks and an intra-task half-batch baseline track gradient norm disparities ($R_{\text{max/min}}$) and pairwise cosine similarities ($\cos(\mathbf{g}_i, \mathbf{g}_j)$) on a local commodity GPU testbed. Statistical testing combines per-fold paired bootstrap 95% confidence intervals with omnibus Friedman-Nemenyi tests and Spearman rank correlation ($\rho$). The software deliverable provides an asynchronous FastAPI service, a relational incident database, and an interactive dashboard for crisis triage across four operational lifecycle states.

### 8.2 Delimitations of the Study

To maintain methodological rigor and computational feasibility, the study establishes clear operational boundaries. Linguistic analysis is strictly confined to code-switched Tagalog-English (Taglish) disaster text. The investigation excludes other Philippine regional languages (such as Cebuano, Ilocano, and Hiligaynon) because Taglish constitutes the predominant register in Philippine disaster social media communications (Herrera et al., 2022), and curating multi-task expert annotations across multiple regional languages exceeds the operational resource budget of this research. Furthermore, the corpus exclusively incorporates public crisis microblogs collected from Twitter/X across four historical Philippine typhoons (Typhoons Haiyan, Vamco, Rai, and Paeng). The study explicitly excludes private messaging platforms, non-crisis social media threads, and live platform streaming APIs, preventing external API rate-limiting constraints and ensuring deterministic experimental reproducibility through replayed historical datasets.

The data processing and inference pipelines handle raw textual content exclusively, excluding multi-modal data streams such as satellite imagery, crowd-sourced photographs, voice recordings, and video feeds. In addition, the architectural scope is delimited to encoder-only multilingual Transformer backbones using hard parameter sharing across intermediate representations. The investigation excludes soft parameter sharing, task-specific adapter modules, and proprietary cloud-hosted generative Large Language Models that require external network access and high-end server clusters.

Model evaluation focuses strictly on the three primary triage tasks, specifically token-level Named Entity Recognition, sequence-level humanitarian intent classification, and sequence-level urgency prioritization. The study excludes auxiliary natural language processing tasks such as sentiment analysis, automatic summarization, and machine translation. Finally, the software deliverable functions strictly as an empirical research artifact and demonstration tool for local triage benchmarking. The application is not integrated into municipal emergency dispatch networks, live 911 infrastructure, or operational government command centers.
<!-- vale on -->

## 9. Significance of the Study

<!-- vale off -->
The outputs of this research deliver computational, methodological, and practical contributions to three distinct stakeholder groups.

**Disaster Response Agencies and Emergency Practitioners (Primary Beneficiaries):**  
Local emergency responders and municipal disaster offices gain a locally deployable triage system powered by a single shared Transformer encoder. By executing Named Entity Recognition, intent classification, and urgency prioritization in a single forward pass, the model runs low-latency local inference on commodity workstations during electrical and telecommunication disruptions. This local deployment eliminates recurring cloud API fees and protects sensitive citizen communications in compliance with the Data Privacy Act of 2012 (Republic Act No. 10173). Furthermore, the web prototype demonstrates automated entity extraction, local gazetteer mapping, and structured incident tracking across four operational lifecycle states.

**Machine Learning and Crisis Informatics Researchers (Secondary Beneficiaries):**  
Natural language processing researchers acquire the Multi-Task Corpus of Taglish Disaster Tweets, providing the first expert-adjudicated dataset with concurrent token-level and sequence-level annotations on Philippine crisis text under cross-event evaluation. In addition, researchers gain empirical benchmark evidence evaluating dynamic loss weighting, gradient surgery, and static weighting under AdamW optimization on asymmetric NLP tasks. The PyTorch gradient diagnostic routines and intra-task half-batch baseline provide a standardized methodology to separate genuine cross-task gradient conflict from mini-batch sample noise.

**Academic Institutions and Future System Developers (Tertiary Beneficiaries):**  
Academic institutions and system developers receive an open-source Design Science Research artifact containing modular training scripts, cross-event evaluation pipelines, and baseline model checkpoints. Future researchers can extend this framework to additional Philippine regional languages, investigate post-training 8-bit or 4-bit quantization for edge devices, or integrate multi-modal sensor streams into the triage pipeline.
<!-- vale on -->

## 10. Definition of Terms

To establish technical precision across this study, the following terms are operationally defined according to their theoretical, mathematical, and algorithmic roles in the experimental framework and software system.

<!-- vale off -->

**AdamW Optimizer:** An adaptive stochastic gradient descent optimization algorithm with decoupled weight decay that maintains running exponential moving averages of first moments (gradient means) and uncentered second moments (uncentered gradient variances) for each parameter. In this study, the AdamW optimizer fine-tunes the shared Transformer backbone and task-specific classification heads, providing per-parameter gradient scale adaptivity across heterogeneous task loss landscapes while applying decoupled weight decay to maintain model generalization.

**BIO Schema for Token Annotation:** A token-level sequence labeling notation that marks the beginning (B-prefix), inside continuation (I-prefix), and outside tokens (O tag without prefix) not belonging to any named entity span. In this study, the schema structures token classification across five distinct tags (`B-LOC`, `I-LOC`, `B-INF`, `I-INF`, and `O`) to extract multi-token location and infrastructure entities from code-switched Taglish disaster tweets.

**Computational Inference Latency ($t_{\text{exec}}$):** The average execution duration in milliseconds required for the multi-task model to execute a single forward pass and output concurrent predictions for all three triage tasks under unbatched single-instance evaluation ($B=1$), computed as
$$t_{\text{exec}} = \frac{1}{N_{\text{iter}}} \sum_{i=1}^{N_{\text{iter}}} \Delta t_i$$
where $\Delta t_i = t_{\text{end}, i} - t_{\text{start}, i}$ represents the elapsed time of iteration $i$ in milliseconds, and $N_{\text{iter}} = 30$ denotes the count of timed evaluation iterations. The benchmarking harness records execution latency on the local GPU testbed using CUDA event synchronization (`torch.cuda.Event(enable_timing=True)` and `torch.cuda.synchronize`) across $N_{\text{iter}} = 30$ timed iterations following 10 preliminary warm-up iterations, reported as the sample mean $\pm$ 95% confidence interval.

**Design Science Research Methodology (DSRM):** An established computer science and information systems research methodology that creates and evaluates new artifacts (a fine-tuned MTL model, LOEO benchmark tables, and a FastAPI triage prototype evaluated against EW/LS/STL baselines) to address practical operational problems. In this study, DSRM organizes the research lifecycle into four sequential phases (Baseline Analysis, Design and Artifact Creation, Empirical Benchmarking, and Comparative Validation), structured under the Input-Process-Output conceptual model.

**Disparity in Gradient Magnitudes ($R_{\text{max/min}}$):** The scalar ratio between the maximum and minimum $\ell_2$ norms of task gradients on shared encoder parameters at training step $t$, expressed as
$$R_{\text{max/min}}^{(t)} = \frac{\max_{k} \|\mathbf{g}_k^{(t)}\|_2}{\min_{k} \|\mathbf{g}_k^{(t)}\|_2}$$
In this study, this metric measures the degree to which dense token-level sequence labeling gradients dominate sparse sequence-level classification gradients during shared parameter updates.

**Dynamic Loss-Weighting Methods:** Multi-task optimization algorithms that dynamically adjust scalar task loss weights $w_k^{(t)} > 0$ at each training step $t$ to minimize the composite multi-task objective, formalized as
$$\mathcal{L}_{\text{total}}^{(t)} = \sum_{k=1}^K w_k^{(t)} \mathcal{L}_k^{(t)}$$
where $\mathcal{L}_k^{(t)}$ is the empirical loss for task $k$ at step $t$. Evaluated algorithms in this family (specifically Uncertainty Weighting, GradNorm, and FAMO) balance task learning rates during backpropagation by scaling scalar loss values without modifying gradient vector trajectories.

**FP16 Mixed Precision Execution:** A computational execution mode that performs forward and backward tensor operations using 16-bit half-precision floating-point arithmetic (`torch.amp.autocast('cuda')` managed via `torch.amp.GradScaler('cuda')`) while maintaining master weights in 32-bit single-precision. In this study, FP16 execution reduces peak GPU memory allocation ($M_{\text{VRAM}}$) and accelerates tensor throughput on the local workstation without compromising numerical gradient stability.

**Gradient Diagnostic Logging:** A diagnostic instrumentation procedure implemented via PyTorch gradient hooks that computes and records per-task gradient norms ($\|\mathbf{g}_k\|_2$), gradient norm disparity ratios ($R_{\text{max/min}}$), and pairwise cosine similarities ($\cos(\mathbf{g}_i, \mathbf{g}_j)$) on shared encoder parameters at every optimization step to track gradient conflict dynamics across training epochs.

**Gradient Directional Conflict:** A geometric condition where gradient vectors for distinct tasks with respect to shared parameters $\boldsymbol{\theta}_{\text{sh}}$ form an obtuse angle, producing a negative inner product, formulated as
$$\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0 \quad \text{or} \quad \cos(\mathbf{g}_i, \mathbf{g}_j) < 0$$
This condition indicates, to a first-order approximation, that parameter updates along task $i$ increase the loss on task $j$.

**Gradient-Surgery and Projection Techniques:** Multi-task optimization algorithms that alter task gradient vectors $\mathbf{g}_k = \nabla_{\boldsymbol{\theta}_{\text{sh}}} \mathcal{L}_k$ prior to parameter updates to eliminate destructive cross-task interference. When pairwise gradient directional conflicts occur ($\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$), these algorithms (specifically PCGrad, CAGrad, IMTL-G, and Nash-MTL) project conflicting vectors onto orthogonal hyperplanes or solve multi-objective subproblems, requiring $K=3$ independent backward passes per optimization step.

**Hard-Parameter-Sharing Multilingual Transformer Encoder:** A deep neural network architecture where all three disaster triage tasks share the hidden representation layers of a single encoder-only multilingual Transformer backbone capable of Taglish subword tokenization (evaluated across XLM-RoBERTa Base at ≈279M parameters, multilingual BERT Base at ≈178M parameters, SEA-LION ModernBERT at ≈313M parameters, and RoBERTa-Tagalog Base at ≈125M parameters). Dedicated task-specific classification heads branch from the shared encoder to output entity spans, intent categories, and urgency tiers in a single forward pass ($\mathcal{O}(1)$ execution time relative to task count $K$).

**Inductive Bias in Representation Sharing:** The explicit optimization constraint imposed on a neural network by training shared encoder parameters $\boldsymbol{\theta}_{\text{sh}}$ across three concurrent loss functions ($\mathcal{L}_{\text{NER}}$, $\mathcal{L}_{\text{Intent}}$, and $\mathcal{L}_{\text{Urgency}}$). This shared objective restricts the parameter search space to representations that generalize across token-level and sequence-level triage tasks simultaneously, improving generalization measured as span-level NER F1 and macro F1 for intent and urgency under 4-fold Leave-One-Event-Out evaluation on held-out typhoon events.

**Intra-Task Baseline Derived from Half-Batch Gradients ($\cos_{\text{intra}, k}$):** A diagnostic noise-floor metric computed as the cosine similarity between gradient vectors from two disjoint half-batches $\mathcal{B}_k^{(1)}$ and $\mathcal{B}_k^{(2)}$ within the same task $k$, calculated as
$$\cos_{\text{intra}, k} = \frac{\langle \mathbf{g}_k^{(1)}, \mathbf{g}_k^{(2)} \rangle}{\|\mathbf{g}_k^{(1)}\|_2 \|\mathbf{g}_k^{(2)}\|_2}$$
In this study, this baseline provides an empirical reference threshold to distinguish genuine cross-task gradient conflict from stochastic variance across mini-batch samples.

**Leave-One-Event-Out (LOEO) Cross-Validation:** A 4-fold cross-validation protocol where models train on tweets from three historical disaster events and evaluate exclusively on the fourth held-out disaster event across three random initialization seeds (seeds 42, 123, and 456). Iterating across all four folds measures out-of-distribution generalization across distinct crisis environments.

**Macro-Averaged F1 Score for Sequence Classification:** The unweighted arithmetic mean of class-specific F1 scores across categorical sequence classification tasks (humanitarian intent with $|C_{\text{intent}}|=4$, and urgency priority with $|C_{\text{urgency}}|=3$), defined as
$$\text{Macro F1}_{\text{Task}} = \frac{1}{|C|} \sum_{c=1}^{|C|} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
where $P_c$ and $R_c$ denote precision and recall for class $c$ computed from the pooled sequence representation $\mathbf{h}_{\text{pool}}$ (such as $[\text{CLS}]$ or $\langle\text{s}\rangle$). This metric weights all target classes equally regardless of class frequency disparities.

**Multi-Task Corpus of Taglish Disaster Tweets:** The primary experimental dataset curated in this study, comprising a target volume of 6,000 to 10,000 code-switched Taglish tweets (with a verified baseline of at least 6,000 instances) collected across four historical Philippine typhoons under a 4-fold Leave-One-Event-Out protocol. Each tweet contains three concurrent annotations, namely token-level BIO spans for location and infrastructure entities, sequence-level humanitarian intent classes, and sequence-level urgency priority tiers, validated through senior expert adjudication with agreement threshold of span-level F1 $\ge 0.70$.

**Multi-Task Inference Throughput:** The processing rate of the multi-task model during batched inference ($B=32$) on the local GPU workstation, expressed in tweets per second, computed as
$$\text{Throughput} = \frac{N_{\text{eval}}}{\sum_{b=1}^{N_{\text{batches}}} t_{\text{batch}, b}}$$
where $N_{\text{eval}}$ represents the total count of evaluated tweets, $N_{\text{batches}}$ is the number of processed batches, and $t_{\text{batch}, b}$ denotes the execution time of batch $b$ in seconds. The benchmarking harness computes throughput across $N_{\text{iter}} = 30$ timed evaluation iterations following 10 warm-up iterations, reported as the sample mean $\pm$ 95% confidence interval.

**Multi-Task Learning (MTL):** A machine learning approach where a single neural network architecture with shared parameters is trained jointly across multiple related task objectives (token-level Named Entity Recognition, sequence-level intent classification, and urgency classification) to learn shared intermediate representations, improve generalization via inductive bias, and reduce inference latency relative to isolated single-task pipelines.

**Multi-Task Optimization (MTO) Algorithms:** Optimization algorithms, comprising dynamic loss-weighting methods and gradient-surgery techniques, that balance task gradients during backpropagation. These algorithms dynamically adjust task loss scalar weights $w_k^{(t)}$ or project shared gradient vectors $\mathbf{g}_k$ to optimize shared encoder parameters $\boldsymbol{\theta}_{\text{sh}}$.

**Negative Transfer:** An empirical condition where joint multi-task optimization degrades generalization performance on task $k$ relative to an isolated Single-Task Learning baseline ($\text{F1}_{\text{MTL}, k} < \text{F1}_{\text{STL}, k}$). When destructive task interference predominates across all evaluated triage objectives, the composite relative transfer metric becomes negative ($\Delta_m < 0$).

**Non-Parametric Hypothesis Testing:** Statistical inference procedures that evaluate performance differences between baseline and candidate models without assuming normally distributed metric distributions. In this study, the two-tier statistical framework establishes primary inferential validity through paired bootstrap 95% confidence intervals computed across test instances within each held-out disaster fold, accompanied by omnibus repeated-measures Friedman tests with Iman-Davenport correction ($p < 0.05$) and post-hoc Nemenyi critical difference ranking across 12 paired event-seed runs (with an $N=4$ fold-level sensitivity analysis).

**Pareto Optimality in Multi-Task Learning:** A parameter state $\boldsymbol{\theta}_{\text{sh}}^*$ where no shared parameter update can decrease the empirical loss on task $i$ without increasing the loss on at least one other task $j$. Multi-task optimization algorithms seek parameter configurations along this Pareto frontier.

**Peak GPU Memory Allocation and Reservation ($M_{\text{VRAM}}$):** The maximum resident video memory in megabytes (MB) consumed by the neural network during training and batched inference on the GPU testbed. Tracking routines reset peak memory counters (`torch.cuda.reset_peak_memory_stats`) before each evaluation fold and record peak memory allocated (`torch.cuda.max_memory_allocated`) and reserved (`torch.cuda.max_memory_reserved`) during the fold.

**Peak Host Memory Usage ($M_{\text{RAM}}$):** The maximum resident set size (RSS) in megabytes (MB) of system RAM consumed by the host Python process during training and inference execution, measured programmatically via `psutil.Process().memory_info().rss / (1024 * 1024)` to quantify workstation deployment feasibility.

**Prototype Web Application for Disaster Triage:** The integrated software artifact developed as a functional research deliverable, comprising (a) an asynchronous FastAPI web service backend connected to a relational database store (SQLite or PostgreSQL) that executes single-pass inference and maintains timestamped crisis records, and (b) an interactive incident dashboard that visualizes extracted entity locations via a local Philippine gazetteer dictionary with unmapped-span fallback and tracks incident lifecycle states across four operational stages (New, Acknowledged, Responding, and Resolved).

**Relative Multi-Task Transfer ($\Delta_m$):** A composite evaluation metric measuring the average percentage gain of multi-task model $m$ over isolated single-task baselines across all $K=3$ triage tasks, defined as
$$\Delta_m = \frac{1}{3} \sum_{k=1}^3 \left( \frac{\text{F1}_{\text{MTL}, k} - \text{F1}_{\text{STL}, k}}{\text{F1}_{\text{STL}, k}} \right) \times 100\%$$
where positive values ($\Delta_m > 0$) indicate positive transfer and negative values ($\Delta_m < 0$) quantify net negative transfer.

**Sequence Labeling versus Sequence Classification:** The structural distinction between token-level dense prediction and sequence-level pooled prediction. Named Entity Recognition performs sequence labeling by aggregating cross-entropy loss over individual subword token hidden states $\mathbf{h}_t$, whereas intent and urgency classification perform sequence classification by predicting class distributions from the pooled sequence representation $\mathbf{h}_{\text{pool}}$. In multi-task training, this structural asymmetry causes gradient norm disparities between token-level and sequence-level tasks.

**Single-Task Learning (STL) Baseline:** A reference training configuration where an independent Transformer model $\boldsymbol{\Theta}_k = \{\boldsymbol{\theta}_{\text{enc}, k}, \boldsymbol{\theta}_{\text{head}, k}\}$ is trained separately on each individual task $k \in \{\text{NER}, \text{Intent}, \text{Urgency}\}$, minimizing
$$\min_{\boldsymbol{\Theta}_k} \mathcal{L}_k(\boldsymbol{\Theta}_k)$$
These models establish empirical performance baselines ($\text{F1}_{\text{STL}, k}$) and define the linear computational complexity baseline for inference latency $\mathcal{O}(K \cdot T_{\text{sh}})$.

**Span-Level F1 Score for Named Entity Recognition (NER):** The harmonic mean of precision and recall computed over entity spans using the CoNLL evaluation standard, computed as
$$\text{Span F1}_{\text{NER}} = \frac{2 \cdot P_{\text{span}} \cdot R_{\text{span}}}{P_{\text{span}} + R_{\text{span}}}$$
A true positive requires an exact match on both the token span boundary $[t_{\text{start}}, t_{\text{end}}]$ and the entity category $E \in \{\text{Location}, \text{Infrastructure}\}$, excluding outside (`O`) tokens.

**Spearman Coefficient of Rank Correlation ($\rho$):** A non-parametric statistic that measures the monotonic association between two ranked variables, defined as
$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$
where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$ denotes the rank difference for observation $i$, and $n$ represents the count of paired observations ($n=12$ across event-seed runs, or $n=4$ for fold-aggregated analysis). In this study, $\rho$ evaluates monotonic relationships between gradient conflict metrics (pairwise cosine similarities and gradient norm disparity ratios) and relative multi-task transfer ($\Delta_m$).

**Static Linear Scalarization (LS):** A baseline multi-task optimization method that minimizes a fixed weighted sum of task losses, formulated as
$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K w_k \mathcal{L}_k$$
where loss weights $\mathbf{w} = (w_1, \dots, w_K)$ remain constant during training and are calibrated via a grid sweep across the simplex $\Delta^{K-1}$ under an equal hyperparameter search budget on an internal validation split carved from training folds.

**Tagalog-English Code-Switching (Taglish):** The target linguistic register of Philippine disaster social media posts, characterized by intra-sentential and intra-word mixing of Tagalog morpho-syntactic elements and English lexical items. In this study, Taglish inputs combine Tagalog grammatical focus markers (*ang*, *ng*, *mga*, *sa*, and *na*) and bound verbal affixes (*nag-*, *mag-*, *um-*, *i-*, and *-an*) with English root verbs and crisis terminology, inducing subword fragmentation in standard multilingual tokenizers.

**Training Runtime:** The total wall-clock execution duration in seconds required to complete model fine-tuning across all training epochs on the GPU testbed. This metric accounts for optimization backward-pass requirements, where gradient-surgery algorithms require $K=3$ independent backward passes per step while dynamic loss-weighting and linear scalarization require a single joint backward pass.

**Uniform Equal Weighting (EW):** The default multi-task optimization baseline that assigns equal weight ($w_k = 1/K$) to all task losses, defined by
$$\mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k, \quad \mathbf{g}_{\text{total}} = \frac{1}{K} \sum_{k=1}^K \nabla_{\boldsymbol{\theta}_{\text{sh}}} \mathcal{L}_k$$
This baseline updates shared encoder parameters using the unweighted arithmetic mean of individual task gradients.
<!-- vale on -->

## 11. References

<!-- vale off -->
Adouane, W., & Bernardy, J.-P. (2020). When is multi-task learning beneficial for low-resource noisy code-switched user-generated Algerian texts? In *Proceedings of the 4th Workshop on Computational Approaches to Code Switching (CALCS@LREC 2020)* (pp. 17-25). European Language Resources Association. https://aclanthology.org/2020.calcs-1.3/

Alam, F., Qazi, U., Imran, M., & Ofli, F. (2021). HumAID: Human-annotated disaster incidents data from Twitter with deep learning benchmarks. In *Proceedings of the International AAAI Conference on Web and Social Media (ICWSM 2021)* (Vol. 15, pp. 914-922). AAAI Press. https://doi.org/10.1609/icwsm.v15i1.18115

Chen, S., Zhang, Y., & Yang, Q. (2024). Multi-task learning in natural language processing: An overview. *ACM Computing Surveys*, *56*(12), 1-32. https://doi.org/10.1145/3663363

Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)* (PMLR Vol. 80, pp. 794-803). PMLR.

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, *7*, 1-30.

Elich, C., Kirchdorfer, L., Köhler, J. M., & Schott, L. (2024). Examining common paradigms in multi-task learning. In *Pattern Recognition: 46th DAGM German Conference, GCPR 2024, Munich, September 10-13, 2024, Proceedings, Part I* (Lecture Notes in Computer Science, Vol. 15297, pp. 131-147). Springer. https://doi.org/10.1007/978-3-031-85181-0_9

Herrera, M., Aich, A., & Parde, N. (2022). TweetTaglish: A dataset for investigating Tagalog-English code-switching. In *Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022)* (pp. 5742-5750). European Language Resources Association.

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, *28*(1), 75-105. https://doi.org/10.2307/25148625

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018)* (pp. 7482-7491). IEEE.

Liu, B., Feng, Y., Stone, P., & Liu, Q. (2023). Fast adaptive multitask optimization. In *Advances in Neural Information Processing Systems (NeurIPS 2023)* (Vol. 36, pp. 24823-24840).

Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-averse gradient descent for multi-task learning. In *Advances in Neural Information Processing Systems (NeurIPS 2021)* (Vol. 34, pp. 1887-1898).

Liu, L., Li, Y., Kuang, Z., Xue, J. H., Chen, Y., Yang, W., Liao, Q., & Zhang, W. (2021). Towards impartial multi-task learning. In *International Conference on Learning Representations (ICLR 2021)*.

Miranda, L. J. V. (2023). Developing a named entity recognition dataset for Tagalog. In *Proceedings of the First Workshop on South East Asian Language Processing (SEALP 2023)* (pp. 13-20). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.sealp-1.2

Montalan, J. R., Layacan, J. P., Africa, D. D., Flores, R. I. S., Lopez, M. T., II, Magsajo, T. D., Cayabyab, A., & Tjhi, W. C. (2025). Batayan: A Filipino NLP benchmark for evaluating large language models. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)*. https://aclanthology.org/2025.acl-long.1509/

National Disaster Risk Reduction and Management Council. (2020). *Situational report on Typhoon Vamco (Ulysses)*. NDRRMC, Republic of the Philippines. https://ndrrmc.gov.ph/

National Disaster Risk Reduction and Management Council. (2022). *Situational report on Typhoon Rai (Odette) and Severe Tropical Storm Paeng (Nalgae)*. NDRRMC, Republic of the Philippines. https://ndrrmc.gov.ph/

Navon, A., Shamsian, A., Achituve, I., Fetaya, E., & Chechik, G. (2022). Multi-task learning as a bargaining game. In *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)* (PMLR Vol. 162, pp. 16428-16446). PMLR.

Olteanu, A., Vieweg, S., & Castillo, C. (2015). What to expect when the unexpected happens: Social media communications across crises. In *Proceedings of the 18th ACM Conference on Computer-Supported Cooperative Work & Social Computing (CSCW 2015)* (pp. 994-1009). ACM. https://doi.org/10.1145/2675133.2675242

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems*, *24*(3), 45-77. https://doi.org/10.2753/MIS0742-1222240302

Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). Language model tokenizers introduce unfairness between languages. In *Advances in Neural Information Processing Systems (NeurIPS 2023)* (Vol. 36, pp. 24841-24856).

Republic of the Philippines. (2012). *Republic Act No. 10173: Data Privacy Act of 2012*. Official Gazette.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.

Sener, O., & Koltun, V. (2018). Multi-task learning as multi-objective optimization. In *Advances in Neural Information Processing Systems (NeurIPS 2018)* (Vol. 31, pp. 527-538).

Wang, C., Nulty, P., & Lillis, D. (2021). Transformer-based multi-task learning for disaster tweet categorisation. In *Proceedings of the 18th International Conference on Information Systems for Crisis Response and Management (ISCRAM 2021)*.

Weld, H., Huang, X., Long, S., Poon, J., & Han, S. C. (2022). A survey of joint intent detection and slot filling models in natural language understanding. *ACM Computing Surveys*, *55*(8), 1-38. https://doi.org/10.1145/3547138

Xin, D., Ghorbani, B., Garg, A., Firat, O., & Gilmer, J. (2022). Do current multi-task optimization methods in deep learning even help? In *Advances in Neural Information Processing Systems (NeurIPS 2022)* (Vol. 35, pp. 24806-24819).

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task learning. In *Advances in Neural Information Processing Systems (NeurIPS 2020)* (Vol. 33, pp. 21824-21835).
<!-- vale on -->
