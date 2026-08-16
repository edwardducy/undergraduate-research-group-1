# Project Notes

Living notes for the Taglish Disaster Tweets MTL paper.
These notes are separate from paper_proposal_original.md, which remains a clean reference copy of the proposal.

## Decision Notes

The title is locked. The revised title cannot be changed. These notes explain the title so the team and reviewers do not need to revisit the decision.

- No "Lightweight" in the title.
  We dropped the word because "lightweight encoder" is difficult to define. The definition already exists in the gap statement as "<100M parameters." If the constraint is needed in a title again, "sub-100M" is more precise. The word is gone; the concept may still appear in the abstract.

- No task list in the title.
  The full list, "Joint Intent, Urgency, and Named Entity Recognition," made the title too long. "Triage" is a signal, not a definition: it points at urgency, but not at NER or intent. The abstract's first sentence must name all three heads explicitly: NER, intent classification, and urgency scoring.

- No CPU or efficiency claim in the title.
  We removed it because "it runs on CPU" is not a strong headline contribution. Any small model can run on CPU with ONNX or quantization, and a latency table on one machine is engineering rather than science. CPU remains in the paper as motivation for local deployment by Philippine emergency responders. Sub-RQ 2, covering latency, memory, and throughput, remains as a secondary deployment-context section.

- What carries the paper now:
  The new Taglish disaster tweet MTL dataset is the strongest novelty, and the dynamic loss-balancing comparison is the methodological contribution.

- Consistency warning:
  The Title Revision Rationale in proposal Part 1 still says that the title "explicitly justifies shifting away from GPU-hosted architectures." That was true of the old title, not the final one. When writing the paper, rewrite the introduction's rationale to match the final title: CPU is motivation, and the dataset is a contribution.

## Advisor Requirement: A New MTL Optimization
August 2026

The advisor expects the team to create a new MTL optimization: a dynamic balancing algorithm of our own, rather than only benchmarking existing algorithms.

### Contributions

The paper should have three contributions:

- A new Taglish disaster tweet MTL dataset.
- A new dynamic multi-task balancing algorithm designed by the team.
- A controlled benchmark of the new algorithm against existing methods (UW, PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, and FAMO) and baselines (static equal weights and three separate encoders, one for each task).

### Context and Consequences

- The title now undersells the paper. The locked title, "Benchmarking Dynamic Multi-Task Balancing…," sounds evaluation-only. In the defense, abstract, and introduction, lead with the method: "We propose a new balancing algorithm and benchmark it against existing ones on a new dataset."

- Sub-RQ 1 contradicts the requirement as written. It names only existing algorithms (UW, PCGrad, and GradNorm). When revising the research questions, make the new algorithm the primary subject. For example: "How does our proposed balancing algorithm compare to UW, PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, FAMO, static equal weights, and three separate encoders?"

- The panel will ask, "What exactly is new about your algorithm?" The team must be able to answer in one sentence and distinguish it from Uncertainty Weighting (uncertainty-based reweighting), GradNorm (gradient-magnitude balancing), PCGrad (gradient projection), and the newer strategies in LibMTL: CAGrad, IMTL, Nash-MTL, and FAMO.

- Current status:
  The new algorithm is an advisor expectation, not yet a designed method. The earlier plan was a factorial comparison of known strategies. Designing the algorithm is now a required deliverable. The factorial grid becomes the evaluation of our own method, not just a survey of other methods.

### Eureka: Let the Evidence Lead the Method

The new MTL method should not begin with a complicated formula. It should begin with a demonstrated problem in the Taglish disaster setting.

- Existing MTL methods already provide general ways to train several tasks together.
- Our question is whether those methods work for Taglish disaster triage, not whether MTL exists.
- We must first compare the existing methods and observe where they fail.
- The new method should be designed to address that specific failure.
- The novelty is the connection between an observed problem and a targeted solution, not an arbitrary combination of existing techniques.
- The shared encoder and task heads define the model architecture. The advisor's requirement concerns how the tasks are balanced during training.
- We must not assume that Taglish creates task conflict or that dynamic balancing is automatically better. Both claims require evidence.
- If the existing methods work well, that result is still useful. It may show that dynamic balancing is unnecessary in this setting.

#### Title-Defense Framing

We will first establish whether existing balancing methods are adequate for Taglish disaster triage. The comparison identifies the consistently weakest output. We then design a balancing method that prevents it from degrading, and we benchmark that method against the existing ones. The deliverable is unconditional, because one output is always the weakest, so the method always exists.

### Title Defense Direction

The title should be presented as an evidence-driven investigation, not as a claim that dynamic balancing is already superior.

- Disaster tweet benchmarks, Taglish resources, and MTL balancing methods already exist, but these areas have not been sufficiently studied together.
- The gap is not the absence of MTL for disaster tweets. The gap is the lack of evidence about whether existing balancing methods work reliably for three tasks in low-resource Taglish disaster text.
- The study will compare existing methods before designing a new one.
- If the comparison reveals a consistent weakness, we will adapt an existing method to address that weakness.
- If existing methods perform well, that result will still be useful because it will show that dynamic balancing is not automatically necessary in this setting.

The title can be explained in five parts:

- Benchmarking means that the study will conduct a controlled comparison.
- Dynamic multi-task balancing identifies the optimization problem being studied.
- Multilingual encoder identifies the model setting.
- Joint triage describes the combined prediction goal.
- Taglish disaster tweets defines the domain and research setting.

### Defense Section Mapping

Do not present the whole gap and method as one paragraph. Divide it across the parts below, and let each part end with the question or promise that the next part picks up.

The golden thread runs through all five parts: code-switched Taglish text fragments meaning, so the shared encoder must balance three tasks that pull it in different directions, and no study has shown which balancing method prevents all three outputs from degrading. The study finds the consistently weakest output and designs a balancing method that prevents it from degrading.

#### Part 1: Title and Problem Motivation

The study asks whether one shared multilingual Transformer encoder can process Taglish disaster tweets and support three tasks: intent classification, urgency prediction, and NER. Disaster tweets can contain requests for help, urgent conditions, locations, and other details that responders need to understand. Responders cannot manually interpret large volumes of time-sensitive tweets. Three separate encoders would repeat computation, so one shared encoder can process each tweet once. However, the shared encoder must learn three tasks from code-switched Taglish tweets, where Filipino and English may appear in the same tweet, and the mixing breaks the connection between a word's spelling and its meaning. The three tasks then pull the shared parameters in different directions, so one task can dominate and degrade the others. This leads to the question that carries the whole study: which balancing method can train one shared multilingual Transformer encoder reliably for these three tasks in the Taglish disaster setting?

#### Part 2: Background and Research Gap

The question has no answer yet. Existing studies have developed disaster tweet benchmarks, Taglish resources, and general multi-task balancing methods, but these areas remain disconnected. No study has tested a dynamic multi-task balancing method on Taglish disaster tweets. The few multi-task studies on code-switched text report conditional results and do not use balancing methods. The gaps leave responders with a choice they cannot validate: separate encoders with higher cost, or a shared encoder with an unknown balancing method. Because of these gaps, there is no established baseline for this setting. The study provides that benchmark, and the next part describes how.

#### Part 3: Proposed CS Methodology and Pipeline

The study provides the benchmark with a training procedure that pursues two goals. The first goal is to adapt the shared encoder to the fragmented code-switched text. The second goal is to balance the three outputs so that no output dominates or gets lost. The comparison of balancing methods runs under a controlled protocol and identifies the consistently weakest output. The team then designs a balancing method that prevents it from degrading, and benchmarks that method against the existing ones. The design of the method is evidence-led: it does not start from a formula, it starts from the measured weakness.

#### Part 4: Experimental Evaluation and Metrics Plan

The evaluation measures exactly what part three needs. The comparison runs in a controlled Taglish disaster setting and includes the required baselines: static equal weights and three separate encoders, one for each task. Every method reports per-output performance for intent, urgency, and NER separately, because a single averaged number can hide the imbalance. The evaluation also reports overall performance, negative transfer, latency, memory use, and throughput, and it validates the findings against naive baselines and across held-out disaster events.

#### Part 4 Grounding: TREC Incident Streams

The TREC Incident Streams track (2018-2021) is the protocol precedent: each edition trains on labeled past events and evaluates on new unlabeled events. The study adopts this event-based split. TREC-IS cannot serve as the testbed, because its collections are English-centric with no Taglish data, its 25 information types plus priority differ from the intent, urgency, and NER scheme, and no TREC-IS edition compares balancing methods under a controlled protocol.

The team has not decided the dataset scale. The script holds the placeholders "a number of annotated tweets across a number of disaster events" in Part 4. The team must decide the tweet count and the event count, replace the placeholders, and record the numbers here.

#### Part 5: Expected Contribution

The contribution has four parts that follow from the narrative. First, the new Taglish disaster tweet dataset with joint triage annotations. Second, the benchmark that compares balancing methods on the Taglish disaster tweet dataset under a controlled protocol. Third, the balancing method that prevents the consistently weakest output from degrading in Taglish disaster triage, designed from the measured weakness and validated against the existing methods. Fourth, the software artifact, a Taglish disaster triage application that runs the trained system on incoming tweets. The method is unconditional: one output is always the weakest, so the study always produces it. The team is not proposing an algorithm without a clear reason.

## Scope Check: New Algorithm vs PhD Scope

August 2026

The team realized that a novel general-purpose MTL optimization is doctoral scope, not undergraduate thesis scope.

- A general method must beat UW, GradNorm, PCGrad, and similar methods across many settings. That bar is a PhD thesis.
- The advisor's requirement reads as a targeted adaptation: design a method that addresses a specific failure observed in the Taglish disaster setting, and explain it in one sentence.
- The dataset remains the strongest contribution. It should be acceptable as the primary contribution if the evidence does not justify a new method.

Action items:

- Ask the advisor to confirm the bar: a general method or a targeted adaptation.
- Ask whether the dataset plus a rigorous comparison is sufficient if existing methods already work well.
- Record the answer in writing, because it becomes the defense boundary.
- The team dropped the conditional framing on August 2026. The script now commits unconditionally: the comparison identifies the consistently weakest output, and the team designs a balancing method that prevents it from degrading.

Resolution:

- The deliverable is an adaptation of an existing balancing method, not a novel algorithm. The "new method" wording in the earlier sections of this file is superseded.

## Defense Boundary: Tier-One Precedents for Benchmark Plus Method

August 2026

The team asked whether a new MTL optimization strategy is PhD scope, and whether a benchmark alone can carry the contribution. The literature answers both questions.

- A new general-purpose MTL optimization algorithm is doctoral scope. GradNorm (ICML 2018), IMTL (ICLR 2021), and MOO-MTL (NeurIPS 2018) are each years of work validated across many benchmarks.
- Tier-one venues publish the smaller pattern: a benchmark or analysis, plus a minimal method derived from the findings.

Verified precedents:

- Taskonomy: Disentangling Task Transfer Learning, Zamir et al., CVPR 2018, Best Paper. The paper built a task-affinity benchmark across 26 visual tasks and introduced transfer maps as the new method.
- Which Tasks Should Be Learned Together in Multi-task Learning, Standley, Zamir, Paull, and Savarese, ICML 2020, PMLR 119:9120. The paper ran an empirical task-affinity study and derived a task-grouping method from the measured affinities.
- Gradient Surgery for Multi-Task Learning (PCGrad), Yu et al., NeurIPS 2020. The paper analyzed gradient conflicts and proposed one surgical fix: project conflicting gradients. The method is minimal because it follows from the analysis.
- Multi-Task Deep Neural Networks for Natural Language Understanding (MT-DNN), Liu et al., ACL 2019. The framework's new elements follow from understanding how cross-task training regularizes.

The panel answer in one form: "A new general MTL optimization algorithm is a thesis. We deliver a benchmark, identify the consistently weakest output, and design a balancing method that prevents it from degrading. That is the same pattern as PCGrad at NeurIPS 2020 and the Taskonomy line of work."

What to actually do:

- Run the comparison of existing balancing methods on the new Taglish disaster dataset. This is the core deliverable.
- Measure the failure with published metrics: per-task gradient magnitudes (GradNorm), conflict angles (PCGrad), and which output degrades (the Humaid-Ner pattern).
- Make the minimal targeted change to the consistently weakest output, for example adjust one existing method for the token-level versus sequence-level imbalance. Validate it against all baselines.
- If the existing methods already work well, the method still prevents the weakest output from degrading, and the benchmark measures whether preventing the degradation costs anything on the other outputs. A small gain is still an honest published result.

Prepared follow-up for the panel: "What if the existing methods are already good?" The answer: one output is still the weakest. Our method prevents it from degrading, and the benchmark measures whether preventing the degradation costs anything on the other outputs. If the gain is small, the dataset and the benchmark still stand alone as the contribution.

### Design Candidate: Multi-Stage Continued Pre-Training

August 2026

The team considers a two-stage training procedure as the vehicle for the method.

- Stage 1: continue pre-training the shared multilingual Transformer encoder on unlabeled Taglish disaster tweets. The literature demonstrates the fragmentation problem, so this stage needs no evidence from our own results.
- Stage 2: train the three outputs with the balancing methods. The comparison identifies the consistently weakest output, and the team designs a balancing method that prevents it from degrading.

Literature grounding:

- Don't Stop Pretraining (Gururangan et al., ACL 2020): a second phase of in-domain pre-training improves performance in high- and low-resource settings. The paper defines domain-adaptive pre-training (DAPT) and task-adaptive pre-training (TAPT).
- Improving Pretraining Techniques for Code-Switched NLP (ACL 2023): continued pre-training on code-switched text is an active direction.
- mDAPT: Multilingual Domain Adaptive Pretraining in a Single Model (Findings EMNLP 2021).
- Adapting Multilingual Models to Code-Mixed Tasks via Model Merging (arXiv 2510.19782): continued pre-training on unlabeled code-mixed text.

Two conditions:

- Terminology: use continued pre-training or domain-adaptive pre-training, not pre-finetuning. Pre-finetuning is the embedding literature term (E5 and REZE).
- The staging is the apparatus, not the novelty. The novelty must be the balancing method designed for the output imbalance as it appears after the pre-training stage.

## Git Commit Messages

August 2026

The repository follows Conventional Commits with one format for all commits:

- docs: for content changes. Examples: the script, the glossary, the notes, the literature corpus.
- chore: for tooling and configuration. Examples: pnpm, uv, vale, the dotfiles.
- fix: for corrections to content or tooling.
- The summary is a lowercase imperative sentence. Example: docs: polish the script narrative, add the public release, and expand the glossary.

The three existing commits were rewritten to this format and force-pushed. The repository has no other contributors, so history rewrites stay safe while the commit count is small.
