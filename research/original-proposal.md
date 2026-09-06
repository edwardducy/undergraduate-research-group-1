# PRCO191: RESEARCH TITLE, RQ, AND LITERATURE ALIGNMENT SHEET

**Group Number / Name:** 1

**Team Members Present:**

- Almuena, Isaiah Raphael
- Ducyogen, Prinz Edward
- Geronimo, Carlo Miguel
- Lu, Kyle Uri Gabriel
- Patricio, Yngwie Sebastian

## PART 1: TITLE REFINEMENT

### Original Title (from Activity 1)

Automated Emergency Triage for Code-Switched Social Media Comments.

### Revised Research Title

Benchmarking Dynamic Multi-Task Loss Balancing in a Lightweight Transformer Encoder for Joint Intent, Urgency, and Named Entity Recognition on Code-Switched Emergency Comments

### Title Revision Rationale (2-3 sentences)

The revised title highlights the study's core algorithmic contribution by evaluating dynamic multi-task loss-weighting mechanisms to resolve gradient interference in resource-constrained backbones targeted for local CPU deployment. By framing code-switched disaster text as a complex stress test for multi-task optimization on edge hardware, the rationale explicitly justifies shifting away from GPU-hosted architectures while anchoring the work in natural language processing research.

## PART 2: RESEARCH QUESTIONS & FUNNEL

### Identified Research Gap

Existing disaster NLP systems in the Philippines either rely on high-latency cloud LLM APIs or use separate single-task pipelines that scale linearly in computational footprint. While multi-task architectures like Ali et al. (2026) combine sequence classification and entity extraction into a shared backbone, they rely on large GPU-hosted models and clean text. In lightweight encoders (<100M parameters), forcing a shared backbone to process informal, code-switched Taglish text creates severe subword fragmentation and conflicting task gradients between token-level (NER) and sequence-level (Intent and Urgency) heads. There is a lack of empirical benchmarks evaluating how dynamic multi-task loss-balancing algorithms mitigate this negative task transfer on low-resource, code-switched disaster text executing on local CPU hardware.

### Main Research Question (Main RQ)

To what extent do dynamic multi-task loss-balancing algorithms mitigate task gradient interference to optimize predictive performance and inference efficiency in a lightweight Transformer encoder performing joint classification and named entity recognition on code-switched Taglish disaster comments executed on CPU hardware?

### Specific Sub-Questions (Sub-RQs)

- **Sub-RQ 1 (Technical/Performance Metric):** How do dynamic multi-task loss-balancing algorithms (specifically Uncertainty Weighting, PCGrad, and GradNorm) impact Intent Macro F1, Urgency Macro F1, and Token-level NER F1 compared to static equal loss weighting and standalone single-task local baselines?
- **Sub-RQ 2 (Accuracy/System Metric):** What is the single-pass inference latency (ms), memory footprint (MB), and processing throughput (comments/sec) of the optimized joint multi-task model on standard CPU hardware compared to running three separate single-task pipelines sequentially?

## PART 3: RELATED LITERATURE & BOOLEAN SEARCH

### Boolean Search String Used

("multitask learning" OR "multi-task transformer" OR "loss weighting" OR "gradient conflict") AND ("disaster" OR "humanitarian" OR "crisis") AND ("named entity recognition" OR "NER") AND ("classification") AND ("code-switched" OR "Taglish" OR "low-resource")

### Foundational Article Citation (APA 7th Edition)

Ali, A., Basir, N., Nawaz, S., Arain, D. N., & Ali, H. (2026). Humaid-Ner (A Disaster Tweet Dataset for Joint Named Entity Recognition and Event Classification Via Uncertainty-Weighted Multitask Learning). The Asian Bulletin of Big Data Management, 6(1), 138-152.

### Key Findings Summary (2-3 sentences)

The authors demonstrated that a single shared Transformer encoder trained via Uncertainty-Weighted Multi-Task Learning can simultaneously perform token-level entity extraction and sequence-level event classification on crisis social media posts. Their findings showed that joint parameter sharing reduces memory overhead while matching or exceeding the predictive accuracy of standalone single-task baselines.

### Direct Relevance & Justification (2-3 sentences)

While Ali et al. (2026) validated uncertainty-weighted joint learning on large GPU models using monolingual crisis text, they did not evaluate the approach on lightweight encoders operating under severe parameter capacity constraints or on code-switched text. Our research directly extends their work by evaluating how dynamic loss-balancing algorithms perform against negative task transfer and subword fragmentation inherent in informal Taglish comments running on local CPU hardware.

# Advisor Feedback

**1. Is there an existing, open-source dataset of code-switched emergency comments annotated for Intent, Urgency, and NER, or will you have to manually label thousands of posts yourself?**

No public dataset combines Taglish, the disaster domain, and joint intent, urgency, and NER annotations, so we must build our own dataset.

We will manually label our own dataset, but not from zero. We will use a tool to pre-annotate the tweets, and then we will review the labels by hand.

We have not yet decided on the size of the dataset.

**2. Which specific multi-task loss balancing techniques are you comparing?**

We will compare loss-balancing and gradient-balancing methods such as Uncertainty Weighting, PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, and FAMO. The two baselines are static equal weights and three separate encoders, one for each task. This list is not yet final, because we are still deciding on the selection criteria. Most of these techniques were originally applied to computer-vision multi-task benchmarks, and not to informal, code-switched text that carries more than one annotation.

**3. Which base encoder are you using and what are your baseline memory/latency constraints?**

We decided on Multilingual ModernBERT as the leading candidate, but we are still deciding. We will first run an initial test on a Taglish sample, and then we will decide on the encoder and set the baseline memory and latency constraints.

**4. How will success be evaluated? How will you measure overall multi-task performance?**

We will measure how accurately the model identifies the intent of a tweet, its urgency level, and key details such as location names and requested supplies. We will score each of these three tasks separately so we can immediately see if the encoder struggles with one specific area, rather than hiding errors in a single overall average.

We will also compare our single shared encoder directly against three separate standalone encoders to confirm that combining the tasks does not degrade performance.

We are still evaluating how success will be measured in practice, and we will refine our evaluation criteria through an actual test.

**5. Research Title**

Old: Benchmarking Dynamic Multi-Task Loss Balancing in a Lightweight Transformer Encoder for Joint Intent, Urgency, and Named Entity Recognition on Code-Switched Emergency Comments

New: Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets

We replaced the term "lightweight encoder" with "multilingual encoder" to reflect both the language and resource requirements of our research. An English-only model does not understand Tagalog, which would cause errors from a language gap rather than from the balancing methods themselves. While a multilingual encoder supports both English and Tagalog, it still struggles with informal, code-switched Taglish disaster tweets. We also removed the word "lightweight" because modern base encoders can already run on consumer laptops.

We replaced "Code-Switched Emergency Comments" with "Taglish Disaster Tweets" to clarify our target language and domain. The term code-switched applies broadly to any language combination, whereas Taglish specifies the exact mix of Tagalog and English our research evaluates. Additionally, disaster tweets also clarifies the disaster-response context and specifies social media posts, distinguishing our data from general comment sections on video platforms or blogs.
