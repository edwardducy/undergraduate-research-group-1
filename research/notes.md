<!-- vale off -->

# Research Notes

Living notes for the Taglish Disaster Tweets MTL paper.
These notes are separate from original-proposal.md, which stays a clean reference copy of the proposal.

## Title Decision

The title is final and cannot change: Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets.

- We dropped "Lightweight," because "lightweight encoder" is hard to define. The gap statement already defines the constraint as "<100M parameters." If the constraint appears in a title again, "sub-100M" is more precise. The concept may still appear in the abstract.
- No task list in the title. The full list, "Joint Intent, Urgency, and Named Entity Recognition," made the title too long. "Triage" is a signal, not a definition: it points at urgency, but not at NER or intent. The abstract's first sentence must name all three heads: NER, intent classification, and urgency scoring.
- No CPU or efficiency claim in the title. Any small model can run on CPU with ONNX or quantization, and a latency table on one machine is engineering, not science. CPU remains in the paper as motivation for local deployment by Philippine emergency responders. Sub-RQ 2, covering latency, memory, and throughput, remains as a secondary deployment-context section.
- What carries the paper: the new Taglish disaster tweet MTL dataset is the strongest novelty, and the dynamic loss-balancing comparison is the methodological contribution.
- The title explains the study in five parts: benchmarking means a controlled comparison, dynamic multi-task balancing is the optimization problem, multilingual encoder is the model setting, joint triage is the combined prediction goal, and Taglish disaster tweets is the domain.

## Advisor Expectation and Scope

August 2026

- The advisor expects a new MTL optimization: a dynamic balancing algorithm of our own, not only a benchmark of existing methods.
- The deliverable is an adaptation of an existing balancing method, not a novel algorithm. The "new method" wording in earlier notes no longer applies.
- A novel general-purpose MTL optimization is doctoral scope, not undergraduate thesis scope. A general method must beat UW, GradNorm, PCGrad, and similar methods across many settings.
- The dataset remains the strongest contribution, and it is acceptable as the primary contribution if the evidence does not justify a new method.
- The comparison identifies the consistently weakest output, and the team designs a balancing method that prevents it from degrading. The commitment is unconditional, because one output is always the weakest, so the method always exists.

## Method Design Principles

- The method is evidence-led: compare the existing methods first, observe where they fail, and then design the targeted fix. The novelty is the connection between the observed problem and the solution, not an arbitrary combination of existing techniques.
- We must not assume that Taglish creates task conflict or that dynamic balancing is automatically better. Both claims require evidence.
- Measurement follows published metrics: per-task gradient magnitudes (GradNorm), conflict angles (PCGrad), and which output degrades (the Humaid-Ner pattern).
- If the existing methods already work well, the study still checks whether preventing the weakest output's degradation costs anything on the other outputs. A small gain is still an honest published result, and the dataset and the benchmark can stand alone as the contribution.

## Defense Structure

The golden thread: code-switched Taglish text fragments meaning, so the shared encoder must balance three tasks that pull it in different directions, and no study has shown which balancing method prevents all three outputs from degrading. The study finds the consistently weakest output and designs a balancing method that prevents it from degrading.

- Part 1 (title and problem motivation): one shared multilingual encoder processes Taglish disaster tweets and supports intent classification, urgency prediction, and NER. The question is which balancing method trains that encoder reliably for the three tasks in the Taglish disaster setting.
- Part 2 (background and research gap): disaster tweet benchmarks, Taglish resources, and general balancing methods exist, but they remain disconnected. No study has tested dynamic balancing on Taglish disaster tweets.
- Part 3 (methodology): adapt the shared encoder to the fragmented code-switched text, balance the three outputs under a controlled protocol, and let the comparison identify the consistently weakest output.
- Part 4 (evaluation): per-output scores for intent, urgency, and NER, overall performance, negative transfer, latency, memory use, and throughput. The baselines are static equal weights and three separate encoders.
- Part 5 (expected contribution): the dataset with joint triage annotations, the benchmark under a controlled protocol, the balancing adaptation that prevents the weakest output from degrading, and the software artifact, a Taglish disaster triage application.
- The defense presents the study as an evidence-driven investigation, not as a claim that dynamic balancing is already superior.

<!-- vale on -->
