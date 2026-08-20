# Title Defense Notes

## Defense Structure

The central thesis: code-switched Taglish text causes representation difficulties, requiring a shared encoder that balances three competing triage tasks. The empirical benchmark uncovers baseline limitations and negative transfer relative to isolated single-task performance ceilings, and the proposed algorithm resolves those limitations by mitigating gradient interference.

- **Part 1 (Title and Problem Motivation):** One shared multilingual encoder processes Taglish disaster tweets for intent classification, urgency prediction, and NER. The study evaluates which balancing method trains the encoder reliably for all three tasks.
- **Part 2 (Background and Research Gap):** Disaster tweet benchmarks, Taglish resources, and general balancing methods exist separately. No published study evaluates dynamic balancing on Taglish disaster tweets.
- **Part 3 (Methodology):** Execute a two-stage adaptation pipeline consisting of domain-adaptive pre-training on unannotated Taglish disaster text followed by supervised multi-task optimization across intent, urgency, and NER under a controlled balancing protocol.
- **Part 4 (Evaluation):** Measure per-output scores for intent, urgency, and NER relative to isolated single-task performance ceilings, overall performance, negative transfer, latency, memory use, and throughput against static equal weights and separate single-task encoders.
- **Part 5 (Expected Contributions):** Deliver the joint triage dataset, the empirical benchmark diagnostic, the targeted balancing adaptation, and the local edge triage software application.
