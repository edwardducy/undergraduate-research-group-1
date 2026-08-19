<!-- vale off -->

# Original Proposed Paper (Reference Copy)

- Saved as the original proposal from the project's Activity 1, the title-refinement stage.
- This file is the reference baseline, and we track later revisions against it.

## PART 1: TITLE REFINEMENT

**Original Title (from Activity 1):**

Automated Emergency Triage for Code-Switched Social Media Comments

**Revised Research Title:**

- *Old:* Benchmarking Dynamic Multi-Task Loss Balancing in a Lightweight Transformer Encoder for Joint Intent, Urgency, and Named Entity Recognition on Code-Switched Emergency Comments
- *New:* **Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets**

**Title Revision Rationale (2-3 sentences):**

- The revised title highlights the core algorithmic contribution, which is evaluating dynamic multi-task loss-weighting mechanisms to resolve gradient interference in resource-constrained backbones targeted for local CPU deployment.
- The rationale frames code-switched disaster text as a complex stress test for multi-task optimization on edge hardware.
- It explicitly justifies shifting away from GPU-hosted architectures, and it anchors the work in natural language processing research.

## PART 2: RESEARCH QUESTIONS & FUNNEL

**Identified Research Gap:**

- Existing disaster NLP systems in the Philippines rely on high-latency cloud LLM APIs, or they use separate single-task pipelines that scale linearly in computational footprint.
- Multi-task architectures like Ali et al. (2026) combine sequence classification and entity extraction into a shared backbone, but they rely on large GPU-hosted models and clean text.
- In lightweight encoders (under 100M parameters), a shared backbone that processes informal, code-switched Taglish text creates severe subword fragmentation and conflicting task gradients.
- The conflict happens between the token-level head (NER) and the sequence-level heads (Intent and Urgency).
- No empirical benchmark evaluates how dynamic multi-task loss-balancing algorithms mitigate this negative task transfer on low-resource, code-switched disaster text on local CPU hardware.

**Main Research Question (Main RQ):**

To what extent do dynamic multi-task loss-balancing algorithms mitigate task gradient interference to optimize predictive performance and inference efficiency in a lightweight Transformer encoder performing joint classification and named entity recognition on code-switched Taglish disaster comments executed on CPU hardware?

**Specific Sub-Questions (Sub-RQs):**

- **Sub-RQ 1 (Technical / Performance Metric):** How do dynamic multi-task loss-balancing algorithms (specifically Uncertainty Weighting, PCGrad, and GradNorm) impact Intent Macro F1, Urgency Macro F1, and Token-level NER F1 compared to static equal loss weighting and standalone single-task local baselines?
- **Sub-RQ 2 (Accuracy / System Metric):** What is the single-pass inference latency (ms), memory footprint (MB), and processing throughput (comments/sec) of the optimized joint multi-task model on standard CPU hardware compared to running three separate single-task pipelines sequentially?

<!-- vale on -->
