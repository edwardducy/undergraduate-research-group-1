# Thesis Framing (Canonical)

The reference framing for Chapter 2: sections are checked against this paragraph, and its claims trace to the source map below.

## Canonical framing

Philippine disaster response needs rapid triage of Taglish tweets because typhoon message volumes exceed manual review capacity. The study formulates triage as three concurrent tasks: token-level entity extraction, sequence-level intent classification, and sequence-level urgency classification. Citizen messages arrive online, so outages interrupt ingestion under any deployment; the local-versus-cloud choice concerns the rest of the pipeline. Cloud-hosted inference adds recurring cost, raises privacy obligations for citizen messages, and halts processing whenever connectivity drops. A fine-tuned multi-task model on local hardware keeps processing collected messages inside agency infrastructure and resumes with restored service. No public corpus pairs token-level entity spans with concurrent sequence-level intent and urgency labels for Taglish, so the study builds one from four typhoons. The field disputes whether specialized multi-task optimization methods outperform simple loss weighting. The dispute has been tested on vision, translation, and reinforcement learning, never with explicit optimization control on code-switched, low-resource text. The study therefore benchmarks seven multi-task optimization methods against single-task, equal-weight, and tuned scalarization baselines under leave-one-event-out evaluation with gradient diagnostics. The benchmark stratifies results across the three language profiles—English-dominant, mixed, and Tagalog-dominant—to test whether method rankings survive variation in mixing degree. The deliverables are the curated corpus, the benchmark evidence, and a prototype deployment; the corpus and prototype constitute the design science artifacts.

## Terms

Taglish = Tagalog-English code-switching. Language profiles = the three Code-Mixing Index strata (English-dominant, mixed, Tagalog-dominant; Gambäck and Das, 2016). The benchmark = this study's evaluation framework.

## Source map

| Claim | Sources |
|---|---|
| Volumes exceed manual review; outages interrupt ingestion | Chapter 1 (NDRRMC 2021; OCHA 2024); Imran et al. 2016 |
| Privacy obligations for citizen messages | Data Privacy Act of 2012 (Republic Act No. 10173) |
| Dispute over specialized optimizers versus simple weighting; testing domains | Kurin et al. 2022; Xin et al. 2022; Elich et al. 2024; Gama and Grassi Jr 2025 |
| No optimization control on code-switched text | Adouane and Bernardy 2020 (code-switched MTL without balancing) |
| Corpus gap (no token-plus-sequence concurrency for Taglish) | Alam et al. 2021a, 2021b; Herrera et al. 2022; Miranda 2023; Montalan et al. 2025 |
| Seven benchmarked methods; baselines; LOEO; gradient diagnostics; stratification | Chapter 1 scope; Demšar 2006; Elich et al. 2024; Gambäck and Das 2016 |
| Design science artifacts | Peffers et al. 2007 |
