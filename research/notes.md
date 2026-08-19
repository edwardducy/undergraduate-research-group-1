# Research Notes

Living notes for the Taglish Disaster Tweets MTL paper.
These notes are separate from original-proposal.md, which stays a clean reference copy of the proposal.

## Title Decision

The title is final and cannot change: Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets.

- **Lightweight dropped:** Dropped "Lightweight" because the term lacks a precise definition.
- **Task list omitted:** Omitted the full task list from the title to keep the title concise. The abstract must name all three heads: Named Entity Recognition, intent classification, and urgency scoring.
- **CPU claim omitted:** Omitted CPU and efficiency claims from the title because small models running on CPU via ONNX or quantization represent engineering rather than science. CPU execution remains as motivation for local deployment by Philippine emergency responders. Sub-RQ 2, covering latency, memory, and throughput, remains as a secondary deployment-context section.
- **Core novelty and method:** The curated Taglish disaster tweet multi-task dataset represents the primary novelty, and the dynamic multi-task balancing comparison represents the methodological contribution.
- **Title components:** The title explains the study in five parts: benchmarking defines the controlled comparison, dynamic multi-task balancing defines the optimization problem, multilingual encoder defines the architecture, joint triage defines the combined prediction goal, and Taglish disaster tweets defines the domain.

## Encoder Selection Strategy

- **Disregard parameter thresholds:** Disregard arbitrary parameter thresholds (such as "<100M parameters" or "sub-100M") because fixed numerical boundaries create unnecessary defense liabilities.
- **Candidate encoder evaluation:** Test a candidate list of multilingual encoders (such as Multilingual ModernBERT, mBERT, and XLM-RoBERTa) to determine empirically which backbone is the most effective and computationally viable on CPU hardware.
- **Generalization:** Evaluate multiple candidate encoders to confirm that dynamic multi-task balancing findings and the targeted adaptation generalize across different multilingual architectures.

## Advisor Guidance and Algorithmic Scope

- **Algorithmic expectation:** The advisor approved and expects an adapted dynamic multi-task balancing algorithm rather than only a passive benchmark of existing methods.
- **Artifact scope:** The planned research artifact is an adaptation of existing balancing methods tailored to code-switched text, rather than a universal general-purpose algorithm. A universal optimization method requires doctoral scope to outperform standard baselines across diverse benchmarks.
- **Scope distinction:** The advisor noted that the project scope exceeds standard benchmarking because the research actively designs a targeted optimization artifact.
- **Targeted protection:** The benchmark systematically identifies the consistently weakest output head, and the team designs an adapted balancing method to prevent that specific head from degrading.
- **Manuscript framing:** The approved defense title retains the term "Benchmarking," while the manuscript text highlights the designed algorithmic adaptation as a core technical contribution.

## Candidate Terms for the Proposed Algorithm

- **Adaptive Dynamic Multi-Task Balancing:** Highlights dynamic weight adjustment guided by training feedback.
- **Weak-Task-Aware Dynamic Optimization:** Emphasizes explicit protection of the lowest-performing triage head.
- **Targeted Dynamic Balancing Adaptation:** Emphasizes evidence-led modification of universal loss or gradient balancing methods.
- **Hybrid Gradient-Loss Balancing Strategy:** Describes combining loss scaling with gradient surgery.

## Refined Four-Part Contribution Hierarchy

- **Dataset Artifact:** The first public multi-task Taglish disaster tweet corpus with joint annotations for intent, urgency, and named entities.
- **Algorithmic Artifact:** An adapted dynamic multi-task balancing algorithm designed to prevent weak-output degradation on code-switched text.
- **Empirical Evaluation:** A controlled experimental benchmark comparing the proposed algorithm against universal MTL methods across disaster event splits.
- **Software Artifact:** A functional, offline-capable disaster triage application executing on edge CPU hardware.

## Method Design Principles

- **Evidence-led design:** Compare existing methods first, observe where they fail, and design the targeted fix based on observed data.
- **Empirical validation:** Avoid assuming that Taglish creates task conflict or that dynamic balancing is automatically superior; both claims require empirical evidence.
- **Standardized metrics:** Measure per-task gradient magnitudes (GradNorm), conflict angles (PCGrad), and per-output score degradation.
- **Honest reporting:** If existing methods perform well, verify whether preventing weak-output degradation harms other outputs; small gains remain honest published results.

## Defense Structure

The central thesis: code-switched Taglish text causes representation difficulties, requiring a shared encoder that balances three competing triage tasks. The study identifies the consistently weakest output and designs a balancing method that prevents that output from degrading.

- **Part 1 (Title and Problem Motivation):** One shared multilingual encoder processes Taglish disaster tweets for intent classification, urgency prediction, and NER. The study evaluates which balancing method trains the encoder reliably for all three tasks.
- **Part 2 (Background and Research Gap):** Disaster tweet benchmarks, Taglish resources, and general balancing methods exist separately. No published study evaluates dynamic balancing on Taglish disaster tweets.
- **Part 3 (Methodology):** Adapt the shared encoder to code-switched text, balance the three outputs under a controlled protocol, and identify the consistently weakest output.
- **Part 4 (Evaluation):** Measure per-output scores for intent, urgency, and NER, overall performance, negative transfer, latency, memory use, and throughput against static equal weights and separate single-task encoders.
- **Part 5 (Expected Contributions):** Deliver the joint triage dataset, the empirical benchmark, the targeted balancing adaptation, and the local edge triage software application.
