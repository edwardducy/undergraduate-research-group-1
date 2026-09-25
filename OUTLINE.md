# Thesis Outline: PRCO191 Undergraduate Research 1

Requirements come from the Canvas activity descriptions (M. Chapter 1/2/3 - FINAL),
the module pages (M2. The Preliminaries, M2. Preliminary & Chapter 1, M2. Chapter 2,
M2. Chapter 3 - Methodology, M1. CS Research - Software Artifact), and the instructor
feedback on the chapter activities.

# Chapter 1: Introduction

1. **Title Page / Proposed Title.** The title must reflect a defensible CS research
   focus, such as algorithmic optimization, model quantization, or experimental
   benchmarking. Maintain a CS research focus and avoid a standard software
   engineering application effort (rubric criterion). Keep the title descriptive,
   objective, and precise, with no marketing language. This page stands alone and
   comes first in the manuscript.
2. **Introduction to the Study.** A short executive narrative overview. It
   introduces the computer science field, the main problem, and the proposed
   solution and research contribution. Write it as one flowing text. Do not add
   sub-section headings.
3. **Background of the Study.** A detailed technical case file structured with the
   Macro-Meso-Micro Funnel Model as one cohesive narrative. Do not use explicit
   Macro, Meso, or Micro section breaks. Move from general to specific. Start with
   the broad topic, key terms, and real-world context supported by statistics or
   official reports. Then summarize what other researchers have already found.
   Next, point out what is still missing or unclear. End with the exact problem
   this study solves and why it matters. Use recent literature from the last 3 to
   5 years and quantitative baseline metrics.
4. **Statement of the Problem (SOP).** Begin with one General Problem paragraph
   that summarizes the overall research goal, following the CS template. Then
   present exactly 4 interrogative questions. Each question must ask about one
   thing only. The 4 questions align one per phase across the 4 DSR phases in
   order: Baseline Analysis, Design & Artifact Creation, Empirical Benchmarking,
   and Comparative Validation.
5. **Objectives.** One general objective, then 4 declarative objectives paired
   one-to-one with the SOP questions (SO1 pairs with SOP1). Each objective must
   contain only one action and one variable, and each must sit in its matching DSR
   phase (Baseline Analysis, Design & Artifact Creation, Empirical Benchmarking,
   Comparative Validation). Do not combine two actions in one objective. The
   rubric grades the question-to-objective alignment across all 4 phases.
6. **Hypotheses.** Exactly one null hypothesis (H0) and one alternative (H1).
   The pair states the study's central claim. Formal per-test hypotheses live in
   Chapter 3 (3.6).
7. **Theoretical Framework.** Exactly three paragraphs. Paragraph 1 states the
   theory and its authors, and defines only the core concepts used in this paper.
   Paragraph 2 maps the study variables to those concepts. Paragraph 3 explains
   how the theory will be used to analyze the results. Use recognized theoretical
   anchors, such as DSRM (Peffers et al.), Computational Complexity Theory,
   Big-O notation, and machine learning principles.
8. **Conceptual Framework.** Include two IPO diagrams with a short explanation
   below each. The first diagram shows the overall research workflow. The second
   diagram shows the software artifact, which is the prototype application. The
   IPO must detail datasets, preprocessing, hardware specs, baseline parameters,
   and empirical metrics (t_exec, M_RAM, accuracy, throughput, p-value).
9. **Scope and Delimitations.** Scope names what the study covers: datasets,
   parameters, and hardware. Delimitations list the boundary choices you made on
   purpose, with a short reason for each.
10. **Significance of the Study.** Explain who will benefit from the study and
    how. Cover the primary, secondary, and tertiary beneficiaries.
11. **Definitions of Terms.** Define each term by how it behaves in the code,
    pipeline, or mathematical model. Do not use dictionary definitions.

# Chapter 2: Review of Related Literature and Studies

1. **Topical sub-sections (2.1 to 2.3).** Split the literature review into 3
   to 4 sub-headings based on the research variables, mathematical models,
   algorithms, or testbed hardware limits. Organize the review topically. The
   old Local versus Foreign split is strictly deprecated. Explain how the
   algorithms work, how complex they are, and which numbers you measure, such
   as latency, RAM, accuracy, and throughput. Do not write general summaries of
   applications.
2. **Algorithmic Literature Taxonomy Matrix (2.4).** Build one comparison
   table. Put key related studies next to this research. The columns are Author
   and Year, Algorithm or Model, Dataset or Testbed, Evaluated Metrics, and
   Limitation or Gap.
3. **Synthesis (2.5).** Write a critical summary of 1 to 2 pages in three
   steps. Step 1: describe the common trends in current work. Step 2: point
   out the critical gaps and bottlenecks. Step 3: explain how this study
   bridges these gaps.

Source requirements: 15 to 20 peer-reviewed sources from IEEE Xplore, ACM DL,
SpringerLink, or ScienceDirect. At least 80% must be from the last 3 to 5 years.
APA 7th Edition managed with Zotero or Mendeley.

# Chapter 3: Research Methodology

Every subsection must reflect technical precision and complete reproducibility.
CS methodology focuses on artifact construction, computational execution,
mathematical formulations, and empirical benchmarking, not human survey
respondents.

1. **Research Methodology (3.1).** State the main framework of the study. Use
   either the DSRM 6-phase framework or the quantitative experimental and
   benchmarking method.
2. **Subjects of the Study (3.2).** List the computational artifacts, datasets,
   image samples, network packets, or hardware units being tested. Do not use
   human survey respondents. If the study involves expert evaluators or
   annotators, state their number consistently everywhere (review comment).
3. **Sampling Technique (3.3).** Describe the dataset split strategy, such as
   70% train, 15% validation, and 15% test, stratified random sampling, data
   augmentation, or k-fold cross-validation. Give the exact split ratios and
   the unit of evaluation.
4. **Instruments (3.4).** List the software profilers (for example PyTorch
   Profiler), benchmarking suites (for example MLPerf or custom high-resolution
   timers), and hardware measurement tools. Map each tool to what it measures.
   Also list the tools that you considered but did not use, and give the reason.
5. **Procedure of Data Gathering (3.5).** Give the steps in order. First,
   acquire the data. Second, clean and normalize it. Third, run the execution
   protocol with warmup runs and N trials. Fourth, save the results
   automatically as CSV or JSON logs.
6. **Statistical Treatment (3.6).** Define the inferential tests (for example
   paired t-test, ANOVA, Wilcoxon Signed-Rank) and state an explicit null (H0)
   and alternative (H1) pair for each inferential test, with the confidence
   threshold p < 0.05.
7. **Data Analysis (3.7).** Give the formulas that measure success: Accuracy,
   Precision, Recall, F1-Score, Mean Latency, Throughput, and Memory Overhead.
   State units where a metric carries physical units.
8. **Software System Description (3.8).** Describe how the software is
   designed and what it needs to run. Use three sub-sections:
   - **3.8.1 System Structure.** Diagrams of the system. These can be an
     architecture diagram, a block diagram, a data flow diagram (DFD), or a
     neural network layer diagram.
   - **3.8.2 Use Cases.** Draw a UML use case diagram that follows the rules
     of use case diagrams. Add use case specifications for the primary actors
     and system workers.
   - **3.8.3 Materials and Equipment.** Make two separate lists. The first
     list covers the hardware, such as CPU, GPU, RAM, VRAM, and testbeds. The
     second list covers the software, such as OS, Python versions, frameworks,
     and CUDA. Include exact versions and memory bounds.
9. **Testing and Evaluation (3.9).** Describe the proof-of-concept tests.
   Cover unit and integration testing, stress tests across batch sizes,
   thermal and memory-leak checks, and ISO/IEC 25010 evaluation when it
   applies.

# Manuscript Mechanics (graded in every chapter)

Short bond paper (8.5 x 11 inches). Left margin 1.5 inches, other margins 1.0
inch. Times New Roman 12pt. Double-spaced body text. APA 7th Edition citations.
All text colored black.
