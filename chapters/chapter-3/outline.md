# Chapter 3 Outline

Plan for Chapter 3: Research Methodology. Sources: the CEU template in `guidelines.md`, `docs/revised-proposal.md`, the Chapter 1 revision skeleton in `chapters/chapter-1/revised.md`, and `references.bib` built from the literature CSVs.

## Conventions

- Draft file is `chapters/chapter-3/draft.qmd`. It is in the `_quarto.yml` render list and renders through the apaquarto-pdf format, matching Chapter 2. The chapter title comes from the YAML title field.
- The draft uses active voice with "this study" as the subject, in present tense, matching Chapter 1. AGENTS.md governs all style mechanics and embeds the APA 7 rules. The CEU guideline overrides on structure and formatting.
- Citations use `references.bib` keys generated from the literature CSVs.

## Section plan

### 3.1 Research Methodology

Final in `draft.qmd`. Prose only, no table. Three candidate tables were cut because each duplicated Chapter 1 or the surrounding prose. Tables are reserved for sections with new content.

### 3.2 Subjects of the Study

Status: drafted in `draft.qmd`.

- Four subjects, each defined as an artifact.
- Subject 1 is the Multi-Task Corpus of Taglish Disaster Tweets, 6,000 to 10,000 tweets from four typhoons with word-level language tags, BIO entity spans, intent and urgency labels. Includes the JSONL schema.
- Subject 2 is four candidate encoders with three task heads under hard parameter sharing. The candidates undergo EW fine-tuning first, and one backbone is selected for the full benchmark.
- Subject 3 is ten optimization configurations, the seven MTO methods and the STL, EW, and LS baselines. Both Chapter 1 documents confirm seven. The proposal sheet's six-method list is outdated.
- Subject 4 is the prototype triage service on PostgreSQL.
- Table: subject summary.

### 3.3 Sampling Technique

Status: drafted in `draft.qmd`.

- 4-fold LOEO with fold composition table.
- Seeds 42, 123, and 456, giving 12 runs per configuration.
- No data augmentation.
- Stratification by Code-Mixing Index into three language profiles, computed on saved predictions.

### 3.4 Instruments to Be Used

Status: drafted in `draft.qmd`.

- Training and evaluation, PyTorch, Transformers, NumPy, pandas, scikit-learn.
- Gradient diagnostics, custom hooks plus the intra-task half-batch baseline of Elich et al. (2024).
- Profiling, PyTorch Profiler, pynvml, psutil, timed repetitions with warm-up.
- Logging resolved as a local MLflow server, which also stores exact library versions per run.
- Statistics, SciPy, custom paired bootstrap, Spearman. Annotation on self-hosted Label Studio. Tests on pytest.

### 3.5 Procedure of Data Gathering

Status: drafted in `draft.qmd`.

- Eight numbered stages, acquisition, filtering and deduplication, language tagging, annotation, agreement measurement, preprocessing and BIO alignment, training protocol, efficiency measurement with cache warming.
- 2026 feasibility amendment in stage 1. The free academic track of Twitter/X ended in 2023, and the platform now bills per retrieved post under pay-per-usage full-archive search. The study seeds Haiyan from the CrisisNLP tweet identifiers [@imran2016twitter] and retrieves the three newer typhoons through direct keyword queries. Verified by web search on 2026-09-09.

### 3.6 Statistical Treatment

Status: drafted in `draft.qmd`.

- Hypothesis pairs H01 to H14. The seven pairs decompose the four specific problems of Chapter 1. Specific Problem 1 receives the baseline pair. Specific Problem 3 receives three per-task pairs and the efficiency pair. Specific Problem 4 receives the transfer-conflict pair and the language-profile pair. Specific Problem 2 concerns artifact design and receives none, by stated rationale.
- Equations for the paired bootstrap with instance resampling per fold and run-level resampling for efficiency, Friedman with Iman-Davenport correction over the 12 fold-seed runs with the N=4 sensitivity check, Nemenyi critical difference, and Spearman, citing Demšar (2006).
- The 7-pair count replaces the earlier plan line that said one pair per specific problem.

### 3.7 Data Analysis

Status: drafted in `draft.qmd`.

- Equation blocks for F1 variants, Δm transfer, gradient norms and cosines, Code-Mixing Index, subword fertility, latency, throughput, and memory.
- One paragraph per equation stating which research question it answers, phrased as the specific problem and the hypothesis pairs it feeds. Symbols follow Chapter 1 definitions. The macro F1 for urgency averages the binary conversion classes. The CMI uses equal weights and stays separate from the profile assignment rule of 3.3.

### 3.8 Software System Description

Status: drafted in `draft.qmd`.

- 3.8.1 figures, the end-to-end pipeline, the model layer flow, and the prototype data flow.
- 3.8.2 UML use case diagram and specifications.
- 3.8.3 software versions and hardware specifications, CPU-only deployment, offline serving.
- Figure tooling decision from 2026-09-09. The group edits figures in draw.io (diagrams.net), the leading free tool for thesis diagrams by web consensus. Editable masters live as `.drawio` sources next to the PDFs, generated by `make-drawio.py`. Interim vector renditions come from `make-figures.py` (pymupdf, no extra dependencies) under the same `fig-*.pdf` names, so the chapter renders before the group exports final art. To replace a figure, export PDF from draw.io over the matching `fig-*.pdf` and rerun quarto.

### 3.9 Testing and Evaluation

Status: drafted in `draft.qmd`.

- Unit tests for alignment and balancing math, determinism checks, integration tests.
- Benchmark matrix of 10 configurations, 4 folds, 3 seeds, CPU stress test, reproducibility rerun.
- Derived counts stated in the draft: 48 encoder-selection runs (4 candidates x 4 folds x 3 seeds) before the 120 benchmark runs, and a 12-run reproducibility rerun of one configuration. The CPU stress test is defined as a full replayed fold ingested at peak rate on the CPU-only mode. Table 3.7 consolidates the test groups.

## Defect register

- pymupdf4llm truncates scanned JSTOR PDFs to the cover page. Fall back to per-page text extraction with pymupdf when a converted paper.md is far shorter than the PDF text layer.
- Plain-prose attributions such as "Elich et al. (2024)" without a citeproc key produce no reference entry and stay invisible to the render checker. Borrowed methods always cite through @keys.

## Open decisions for the group

1. Train and validation split, resolved by the HumAID convention of 70/10/20 train, development, and test splits (Alam et al., 2021). Within each LOEO split, 10 percent of the three training typhoons forms a class-stratified validation set.
2. Urgency labels resolved as the four-level Kayi et al. (2020) scale, with a binary conversion for analysis. Intent classes resolved as a triage-oriented merge of the 11 HumAID categories, following the class consolidation practice of HumAID itself, with the mapping documented in the codebook. Word-level tagset resolved following the code-switching shared task convention with mixed and ambiguous tags, described by Gambäck and Das (2016).
3. Profile assignment resolved from the CMI definition in Gambäck and Das (2016), where monolingual utterances have an index of zero. Tweets with zero switched tokens are English-dominant or Tagalog-dominant by majority language, and tweets with any switched content form the mixed profile.
4. Annotation protocol resolved by the HumAID precedent (Alam et al., 2021). It uses three judgments per tweet, approval at a two-of-three majority, and Fleiss' kappa as the agreement statistic. Bands are read against Landis and Koch (1977), pending the new literature entry. Feasibility of three annotators needs group confirmation.
5. Run logging resolved as local MLflow. Final workstation specifications pending.
6. Table numbering, renumber to sequential APA once Chapter 2 is final.
7. Title page fields, course, professor, and due date, plus author note handling.
8. Encoder selection criterion, proposed as mean LOEO macro F1 under EW, to confirm with the group.
9. Gazetteer source for location resolution, proposed as the PSA PSGC masterlist.
10. Tuning procedure resolved by the Xin et al. (2022) findings that untuned scalarization baselines create false MTO gains and that sweeping task weights explores the Pareto frontier. The study sweeps task weights on validation macro F1.
11. Fine-tuning hyperparameters, proposed as a learning rate drawn from the Devlin et al. (2019) fine-tuning range, early stopping on validation macro F1, and the same hyperparameters for every configuration, so that baseline tuning stays fair.
12. Retrieval budget for the paid X API full-archive search. The channel is decided, but the group must approve the funds. A candidate pool of about 100,000 posts costs about 500 US dollars at the 2026 rate of 0.005 dollars per post read, and CrisisNLP seeding keeps the Haiyan reads down to known-relevant identifiers.
