# Revised Research Title, RQ, and Literature Alignment Sheet

This sheet revises the submitted alignment sheet after the title proposal review. The research title is retained.

## PART 1: TITLE REFINEMENT

### Former Title (as submitted)

Benchmarking Dynamic Multi-Task Loss Balancing in a Lightweight Transformer Encoder for Joint Intent, Urgency, and Named Entity Recognition on Code-Switched Emergency Comments

### Revised Research Title (retained)

Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets

### Title Rationale

The title names the constructs of the study. Dynamic multi-task balancing is the object of study. Joint triage defines the task structure. The study trains one shared encoder for two sequence-level tasks, intent and urgency classification, and one token-level task, named entity recognition. A multilingual encoder is needed because the text mixes English and Tagalog. Taglish disaster tweets define the evaluation domain, which is low-resource code-switched text in a disaster-response setting.

## PART 2: RESEARCH QUESTIONS & FUNNEL

### Identified Research Gap

Dynamic multi-task balancing algorithms such as Uncertainty Weighting, GradNorm, PCGrad, CAGrad, Nash-MTL, and FAMO adapt task weights or gradients during training. These algorithms were formulated and validated on homogeneous task bundles, mostly in computer vision, where the task losses share similar scale and structure. Published comparisons of balancing methods rarely leave that setting. This study examines a task structure that those comparisons do not cover. It combines one token-level task with two sequence-level tasks in a single encoder, so the task losses differ in scale, sparsity, and gradient behavior.

The evaluation domain adds one open question and one practical demand. The open question comes from the text. It is low-resource and code-switched, studies of multi-task learning for code-switched text are rare, and it is unknown whether interference between the tasks grows with the degree of Tagalog and English mixing or whether the ranking of balancing methods changes across mixing levels. The practical demand comes from the deployment setting. Disaster response needs inference under degraded connectivity, so the triage system should run on consumer hardware. The study therefore includes an efficiency comparison between the joint encoder and a cascade of separate single-task models. No public corpus supports this study because none combines the Taglish language pair, the disaster domain, and concurrent intent, urgency, and entity annotations. For this reason, the corpus is built as part of this research and is a contribution on its own.

### Main Research Question (Main RQ)

To what extent do dynamic multi-task balancing algorithms mitigate interference between token-level and sequence-level tasks in a shared multilingual encoder for the joint triage of Taglish disaster tweets, compared with static weighting and single-task baselines?

### Specific Sub-Questions (Sub-RQs)

- **Sub-RQ 1 (Performance).** How do the balancing methods affect per-task performance, measured as intent macro F1, urgency macro F1, and entity-level NER F1, against static equal weights and three single-task encoders? We expect the dynamic methods to improve on static weighting on at least one task, and we expect no single method to dominate all three tasks.
- **Sub-RQ 2 (Task interference).** Does interference differ across task pairs? The study measures the cosine similarity between task gradients on the shared parameters and compares the pair of sequence-level tasks with the pairs that include the token-level task. We expect the token-level pairs to show lower gradient similarity, and the gradient-surgery methods to help the token-level task more than the loss-weighting methods.
- **Sub-RQ 3 (Code-switching).** Does the ranking of balancing methods stay stable across the language profiles of the test tweets? Annotation records a word-level language tag for every tweet, and tweets are grouped into three profiles, English-dominant, mixed, and Tagalog-dominant. Subword fertility is reported per profile, because informal spelling and Tagalog morphology also change tokenization. The stratified evaluation runs on saved predictions and adds no training runs. We expect the gap between static and dynamic balancing to grow with mixing, and we treat this comparison as descriptive rather than as an explanation of mechanism.
- **Sub-RQ 4 (Efficiency).** What are the latency, memory footprint, and throughput of the joint encoder compared with three separate single-task encoders on CPU hardware? We expect the joint encoder to reduce end-to-end latency close to the ratio of encoder passes while keeping the change in per-task accuracy small.

## PART 3: RELATED LITERATURE & BOOLEAN SEARCH

### Boolean Search String Used

("multi-task learning" OR "multi-task balancing" OR "loss weighting" OR "gradient conflict" OR "task interference") AND ("named entity recognition" OR "sequence labeling" OR "intent detection" OR "slot filling") AND ("code-switching" OR "code-switched" OR "Taglish" OR "low-resource") AND ("disaster" OR "crisis" OR "emergency" OR "humanitarian")

### Foundational Article Citation (APA 7th Edition)

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 7482-7491).

### Key Findings Summary

The authors derived task weights from the homoscedastic uncertainty of each task and learned these weights jointly with the network. Across three vision tasks with different scales and units, the uncertainty-weighted objective outperformed equal weighting and manual tuning. It also stayed robust under task-dependent noise.

### Direct Relevance & Justification

This paper is the origin of the research object of the study. It shows that a shared network needs principled balancing when task losses differ, and it provides the reference method of the study. This research asks the same question in a new setting. In this setting the task losses differ because the tasks differ in granularity, the text is low-resource and code-switched, and the model should run on consumer hardware.
