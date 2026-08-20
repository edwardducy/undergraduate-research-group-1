# Research Notes

This document records design decisions and technical specifications for the Taglish disaster tweet multi-task learning study.
These working notes remain separate from original-proposal.md, which preserves the unmodified research proposal.

## Title Decision

The title remains fixed for the title defense: Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets. Per advisor guidance, the team can only modify the research title after completing Chapter 1.

- **Omission of "Lightweight":** The title excludes the term "Lightweight" because the descriptor lacks an agreed technical definition and creates unnecessary defense liabilities.
- **Task list scope:** The title omits the exhaustive list of tasks to maintain brevity, while the abstract explicitly names all three prediction heads: named entity recognition, intent classification, and urgency prediction.
- **Edge deployment scope:** The title omits CPU and efficiency claims because model optimization on CPU hardware represents engineering implementation rather than core scientific contribution. CPU execution remains the operational motivation for local deployment by Philippine emergency responders, while secondary research questions address latency, memory footprint, and throughput.
- **Core novelty and methodology:** The curated Taglish disaster tweet multi-task dataset provides the primary empirical resource, whereas dynamic multi-task balancing provides the core methodological contribution.
- **Title components:** The title decomposes the investigation into five core elements: "Benchmarking" denotes the controlled empirical evaluation, "Dynamic Multi-Task Balancing" denotes the optimization strategy, "Multilingual Encoder" denotes the shared architecture, "Joint Triage" denotes the combined prediction objective, and "Taglish Disaster Tweets" denotes the linguistic and operational domain.
- **Diagnostic and algorithmic hierarchy:** The empirical benchmark serves as the diagnostic stage that reveals baseline optimization failures, whereas the proposed balancing algorithm serves as the targeted intervention that resolves those identified limitations.

## Encoder Selection Strategy

- **Parameter threshold exclusion:** The study rejects arbitrary parameter thresholds (such as "<100M parameters" or "sub-100M") because rigid numerical boundaries create unnecessary defense liabilities without providing scientific value.
- **Candidate encoder evaluation:** The benchmark tests a candidate set of multilingual encoders (including mmBERT, mBERT, and XLM-RoBERTa) to determine empirically which backbone is the most effective and computationally viable on CPU hardware.
- **Architectural generalization:** The experimental design evaluates multiple encoder architectures to confirm that dynamic multi-task balancing findings and the proposed adaptation generalize across diverse model families.

## Algorithmic Scope and Contribution Framing

- **Targeted adaptation scope:** The proposed algorithmic artifact adapts existing dynamic multi-task balancing methods specifically for code-switched text, avoiding unsupported claims of universal optimization across arbitrary domains.
- **Intervention beyond passive benchmarking:** The project scope extends beyond observational benchmarking because the research actively develops a targeted optimization algorithm to resolve empirically identified multi-task conflicts.
- **Negative transfer mitigation:** The optimization objective explicitly mitigates negative transfer and gradient interference on tasks degraded relative to isolated single-task performance ceilings.
- **Manuscript and title alignment:** The research title retains the term "Benchmarking" to establish the diagnostic baseline, while the manuscript text highlights the targeted balancing adaptation as the primary methodological contribution.

## Four-Part Contribution Hierarchy

- **Resource Contribution:** The first public multi-task Taglish disaster tweet corpus with joint annotations for intent classification, urgency prediction, and token-level Named Entity Recognition.
- **Methodological Contribution:** An adapted dynamic multi-task balancing algorithm designed to arbitrate gradient conflict and mitigate negative transfer during code-switched multi-task optimization.
- **Empirical Contribution:** A controlled experimental benchmark evaluating universal multi-task learning baselines across disaster event splits to diagnose failure modes on non-standard text.
- **Practical Contribution:** An offline triage system validating execution feasibility, latency, and memory consumption on resource-constrained CPU hardware.

## Methodology

The study structures model training into two distinct adaptation stages:

- **Stage 1:** Domain-adaptive pre-training via Masked Language Modeling on unannotated Taglish disaster text to adjust contextual token representations and reduce dialect shift.
- **Stage 2:** Supervised multi-task optimization combining sequence-level intent classification, sequence-level urgency prediction, and token-level Named Entity Recognition under dynamic loss and gradient balancing methods.

## Diagnostic Protocol and Experimental Principles

- **Evidence-led adaptation:** Evaluate standard baseline models first, observe where negative transfer occurs, and derive the targeted algorithmic intervention from empirical diagnostic observations.
- **Empirical verification:** Avoid assuming that Taglish text inevitably induces task conflict or that dynamic balancing provides universal improvements, because both claims require empirical verification.
- **Objective reporting:** When baseline algorithms achieve strong convergence, examine whether mitigating negative transfer on degraded heads impairs remaining objectives. Modest performance improvements constitute valid scientific results.
