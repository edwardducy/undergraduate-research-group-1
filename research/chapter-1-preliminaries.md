<!-- vale off -->

# Chapter 1 Preliminaries: CEU CSIT Module M2

**Record:** Pasted copy of the university module "M2. Preliminary & Chapter 1: General Components of a CS Research Proposal", a guide to algorithmic investigations, Design Science Research (DSR), empirical benchmarking, and CEU manuscript protocols. Recorded 2026-08-31.

**Relationship to other research files:**

- This file holds the department rules, templates, and the CS versus software-engineering criteria.
- `research/chapter-1-guidelines.md` holds the specific assignment instructions and the 50-point grading rubric for the Chapter 1 draft.
- Read this file for rules and sentence patterns. Read the guidelines file for what the task asks and how a marker scores the answer.

**Formatting note:** The original module presents the comparison dimensions, the DSR phase matrix, and the IPO model as tables. This record converts those tables to bold-labelled bullet lists, so the file stays readable as raw text and so agents can quote individual entries without table syntax.

## Learning Objectives

After this module, CS research students should be able to:

- Differentiate Computer Science research, which creates new computational knowledge, from software engineering, which builds standard applications with existing APIs.
- Apply CEU CSIT standard manuscript mechanics, APA 7th Edition citations, Zotero reference management, and Boolean search strings.
- Distinguish the Introduction to the Study, an executive domain hook, from the Background of the Study, a technical case file.
- Structure the Background of the Study with the Macro-Meso-Micro model to highlight algorithmic and computational technical gaps.
- Formulate scoped research questions and align the Statement of the Problem (SOP) with Design Science Research objectives.
- Construct an Input-Process-Output (IPO) conceptual framework integrated with quantitative performance metrics, namely execution time, memory footprint, accuracy, precision, and throughput.
- Define algorithmic boundaries, dataset scope, technical delimitations, and operational definitions that match the project codebase and mathematical models.

## 1. Foundation: Software Engineering Versus Computer Science Research

Before drafting Chapter 1, confirm that the project is a Computer Science research project rather than a standard application development effort.

**Core formula for CS research:**

- CS Research Project = New Component (previously untested algorithmic modification compared against baselines with t_exec, M_RAM, accuracy) + Software Artifact + Empirical Metrics

**Comparison by dimension:**

- **Primary Goal**
  - Software Engineering: Build a working, reliable software product for end users.
  - Computer Science Research: Create new computational knowledge, such as an algorithm that reduces inference latency or memory by a stated percent on the target hardware.
- **Nature of Problem**
  - Software Engineering: A known problem solved with standard off-the-shelf tools and APIs.
  - Computer Science Research: An unresolved computational problem or an open algorithmic bottleneck.
  - Tools: Existing frameworks, standard libraries, and commercial SaaS APIs for software engineering, versus custom algorithmic modifications, mathematical models, and empirical testbeds for CS research.
- **Success Metric**
  - Software Engineering: User adoption, business workflow fulfillment, and UI or UX surveys.
  - Computer Science Research: Empirical validation, accuracy and speed metrics, and peer-reviewed proof of concept.

**Not CS research, pure app development:**

- "Mobile App for Plant Disease Detection using a pre-trained AI model from GitHub."

**Defensible CS research project:**

- "Optimization of Lightweight CNN Models for real-time plant disease detection on low-end Android hardware via model quantization."

## 2. General Mechanics And Literature Protocol

All CS proposals submitted to the CEU Department of Computer Science and Information Technology must follow these manuscript rules and reference search procedures.

**Paper and margins:**

- Paper size: Short Bond Paper, 8.5 by 11 inches.
- Left margin: 1.5 inches, reserved for binding allocation.
- Top, bottom, and right margins: 1.0 inch.

**Typography and spacing:**

- Font style: Times New Roman throughout.
- Font size: 12pt for body text and subheadings.
- Spacing: Double-spaced body paragraphs.

**Search and reference strategy:**

- Boolean querying: Construct search strings with AND, OR, and exact quotes in IEEE Xplore and the ACM Digital Library.
- Citation manager: Use Zotero for single-click capture and automated APA 7th Edition formatting.

## 3. Introduction To The Study Versus Background Of The Study

Chapter 1 opens with two distinct sections that build the opening narrative of the proposal.

### 3.1 Introduction To The Study: The Hook And High-Level Overview

- The Introduction is an executive narrative of 1 to 1.5 pages that hooks the reader.
- The Introduction presents the big picture of the field without dense empirical benchmark tables.

**Required content, in paragraph order:**

- **Paragraph 1, the broad domain:** Introduce the high-level Computer Science subfield, such as Edge AI, computer vision, cryptography, or distributed systems, and state its significance in modern computing.
- **Paragraph 2, the general high-level problem:** Describe briefly the practical or computational challenge in current systems.
- **Paragraph 3, the proposed solution and intent:** State what the study aims to do at a high level and highlight the primary CS research contribution, such as optimizing execution speed, reducing memory footprint, or improving accuracy.

### 3.2 Background Of The Study: The Technical Case File And Macro-Meso-Micro Model

- The Background is a rigorous, data-driven technical case file of 2 to 4 pages.
- The Background narrows from broad theoretical frameworks (e.g., DSRM, complexity theory) to concrete computational bottlenecks through the Macro-Meso-Micro funnel model.

**Macro perspective, global algorithmic and CS research trends:**

- Establish the global state of the core CS subfield.
- Cite recent peer-reviewed studies from the last 3 to 5 years, from IEEE Xplore, the ACM Digital Library, or Springer.
- Show how recent peer-reviewed algorithms (2021–2026, IEEE/ACM) perform on accuracy, latency, and memory benchmarks worldwide.

**Meso perspective, national or regional application context:**

- Bring the focus to the Philippine national or regional context.
- Detail infrastructure constraints, such as limited network bandwidth and the prevalence of low-tier Android hardware.
- Detail relevant regulatory frameworks, such as the Data Privacy Act, Republic Act 10173.

**Micro perspective, specific testbed, dataset, and computational bottleneck:**

- Pinpoint the target dataset, local hardware testbed, or operational environment.
- Detail the specific computational inefficiencies in the "As-Is" baseline, such as high memory overhead, excessive latency, or cold-start degradation.

**Technical writing rule, quantitative baseline instead of emotional narrative:**

- Avoid subjective claims such as "Users find the facial recognition slow and annoying."
- State data-driven baselines such as: "Existing LBP algorithms exhibit a 42% drop in recognition accuracy under lighting conditions below 50 lux, with inference times exceeding 1.8 seconds per frame on constrained hardware."
- The numbers in the example sentence are illustrative module content. They are not measured results for this study and must not be cited as such.

## 4. Statement Of The Problem (SOP) And Research Objectives

- The Statement of the Problem defines the precise scientific gap the project resolves.
- The SOP consists of a general problem paragraph followed by specific questions aligned with Design Science Research or experimental methodologies.

**General problem template for CS research:**

- "This study aims to design, implement, and empirically evaluate [Name of Proposed Algorithm / Optimized Model / Heuristic PoC] for [Target Application Domain] to resolve current limitations in [Specific Computational Bottleneck, e.g., inference latency, accuracy degradation, cold-start data sparsity]."

**Research phase alignment matrix, SOP versus research objectives:**

- Every interrogative SOP question must mirror an action-oriented declarative research objective across the four DSR phases.

- **Phase 1: Baseline Analysis**
  - SOP question: "What are the performance limitations and accuracy metrics of existing baseline models under [Target Condition]?"
  - Research objective: "To establish benchmark performance metrics of existing algorithms using [Standard Dataset / Testbed]."
- **Phase 2: Design and Artifact Creation (DSR)**
  - SOP question: "How can [Proposed Algorithm: name the modification, e.g., quantized ensemble with GradNorm] be designed and integrated into a proof-of-concept software artifact?"
  - Research objective: "To design and implement [Proposed Algorithm: specify name, inputs, and tooling] using Python and PyTorch into a functional proof-of-concept artifact."
- **Phase 3: Empirical Benchmarking**
  - SOP question: "What is the empirical performance of the proposed artifact in terms of execution time (texec), memory footprint (MRAM), and accuracy (Acc)?"
  - Research objective: "To empirically measure the artifact's performance across execution speed, memory utilization, and classification accuracy."
- **Phase 4: Comparative Validation**
  - SOP question: "Is there a statistically significant performance difference between the proposed model and existing baseline methods?"
  - Research objective: "To statistically evaluate performance gains of the proposed algorithm against standard baselines using paired t-test or ANOVA."

## 5. Theoretical And Conceptual Frameworks

### A. Theoretical Foundations In Computer Science

The research must anchor in recognized scientific theories and formal frameworks:

- **Design Science Research Methodology, DSRM, Peffers et al.:** Iterative cycle of Problem Identification, Solution Objectives, Artifact Design, Demonstration, Empirical Evaluation, and Communication.
- **Computational Complexity Theory:** Big-O notation, such as O(n) and O(n log n), for time and space trade-off evaluation.
- **Machine Learning and Algorithmic Principles:** Optimization functions, gradient descent, deep neural networks, graph theory, or cryptographic security proofs.

### B. CS Empirical Performance Metrics

- **Processing speed and latency:** Inference time per frame in milliseconds, throughput in frames per second, and execution time (texec).
- **Computational efficiency:** CPU or GPU load from 0 to 100 percent, RAM allocation in megabytes, and FLOPs count.
- **Model precision and quality:** Accuracy, precision, recall, F1-score, and mean average precision (mAP).
- **Stability and edge cases:** Accuracy and latency retention under specified degradations (e.g., <50 lux, 10% packet loss) or sparse data.

### C. Input-Process-Output (IPO) Conceptual Model For CS

Represent the analytical and experimental flow of the project in IPO format:

- **Input**
  - Raw datasets and testbed hardware specifications
  - Baseline algorithmic parameters
  - Hyperparameter constraints
  - DSR evaluation metrics and benchmarks
- **Process**
  - Data preprocessing and feature extraction
  - Algorithm modification or quantization
  - Software proof-of-concept artifact implementation
  - Empirical benchmarking and statistical testing
- **Output**
  - New or optimized artifact that the study measures with execution time, memory footprint, and accuracy (p<0.05)
  - Validated empirical benchmark results
  - Comparative performance proof, p < 0.05
  - Open research codebase or proof of concept

## 6. Scope, Significance, And Terminology

### Scope And Algorithmic Delimitations

- **Scope:** State explicitly the target dataset sizes, model parameters, hardware testbeds, and evaluated environment conditions.
- **Delimitations:** State boundary choices explicitly, for example: "The study is delimited to 8-bit integer model quantization and does not test 4-bit compression schemes."

### Significance Of The Study

- **Primary beneficiaries:** Industry practitioners who need lightweight or fast algorithmic implementations.
- **Secondary beneficiaries:** The Computer Science community and research field, which gain verified empirical benchmarks.
- **Tertiary beneficiaries:** Future CS students who build on the algorithmic methodology or codebase.

### Operational Definitions Rule For CS

- Avoid dictionary definitions.
- Define technical and algorithmic terms operationally, based on how each term functions inside the mathematical formulas, the dataset pipeline, or the execution codebase.

<!-- vale on -->
