# Chapter Draft Guidelines

Consolidated from `chapters/chapter-1/guidelines.md`, `chapters/chapter-2/guidelines.md`, and `chapters/chapter-3/guidelines.md` on 2026-09-11. Each chapter keeps its own instructions and 50-point rubric; the originals remain recoverable in git history.

---

## Module Task: Research Proposal Chapter 1 Draft

Undergraduate Research 1 | CEU Department of Computer Science and IT

Submission Format: PDF document uploaded via Canvas

Total Points: 50 Points

### Task Overview & Instructions

Now that we have covered the foundational components of Computer Science Research, manuscript guidelines, and structural models, it is time to write the initial draft of Chapter 1: The Problem and Its Background for your proposed research project.

Important Reminder: Ensure your proposed project is a defensible Computer Science research project (focused on algorithmic optimization, computational metrics, or novel experimental components) rather than a standard software engineering application development effort.

### Required Manuscript Formatting

- Paper Size: Short Bond Paper (8.5" × 11")
- Margins: 1.5 inches (Left) for binding allocation; 1.0 inch (Top, Bottom, Right)
- Typography: Times New Roman, 12pt font size
- Spacing: Double-spaced body text
- Citations: APA 7th Edition format managed via Zotero or IEEE/ACM academic databases (IEEE Xplore, ACM DL) using Boolean search strategies

### Required Chapter 1 Sections

1. Title Page / Proposed Title: Must reflect a defensible CS project focus (e.g., algorithmic optimization, dataset model quantization, experimental benchmarking).
2. Introduction to the Study: An executive narrative overview that introduces the general CS subfield, highlights the core high-level challenge, and introduces the proposed solution and research contribution.
3. Background of the Study: A detailed technical case file structured using the Macro-Meso-Micro Funnel Model:
    - Macro: Global CS/Algorithmic trends citing peer-reviewed papers from the last 3–5 years (IEEE/ACM).
    - Meso: Regional/Philippine national context and infrastructure/regulatory constraints (e.g., low-tier devices, RA 10173).
    - Micro: Target dataset, local hardware testbed, and specific computational bottlenecks.
    - Rule: Avoid subjective narratives; use quantitative baselines (e.g., latency in ms, RAM footprint in MB, drop in accuracy under specific conditions).
4. Statement of the Problem (SOP)  Provide a General Problem Paragraph following the CS template. Present 4 Interrogative SOP Questions
5. Objectives: Provide the SOP paired declarative Objectives aligned across the 4 DSR phases: Baseline Analysis, Design & Artifact Creation, Empirical Benchmarking, and Comparative Validation.
6. Theoretical Framework: Identify theoretical anchors (e.g., DSRM by Peffers et al., Computational Complexity Theory, Big-O notation, ML principles).
7. Conceptual Framework.  Present an Input-Process-Output (IPO) diagram/table detailing datasets, preprocessing, hardware specs, baseline parameters, and empirical metrics (texec, MRAM, accuracy, throughput, p-value).
8. Scope, Delimitations: Explicitly define Scope (datasets, parameters, hardware) and Delimitations (deliberate boundary choices).
9. Significance of the Study.  Outline primary, secondary, and tertiary beneficiaries.
10. Definitions of Terms: Define key terms operationally based on their behavior within your code, pipeline, or mathematical models—not standard dictionary entries.

### Grading Rubric Matrix (50 Points)

| Evaluation Criteria | Exemplary (100%) | Proficient (80%) | Developing (60%) | Unacceptable (≤40%) | Points |
|---|---|---|---|---|---|
| CS Focus vs. SE Distinction & Title Scope | Focuses clearly on creating computational knowledge, novel algorithms, optimization, or empirical benchmarks. Title reflects a defensible CS scope. | Focuses mostly on CS research, but leans slightly toward standard application development. Title is slightly standard. | Appears mostly as standard application development with minimal algorithmic experimentation. | Pure standard software engineering/app development using off-the-shelf APIs with no CS research dimension. | /10 |
| Introduction & Background of the Study | Introduction clearly hooks domain & intent. Background masterfully executes Macro-Meso-Micro with recent literature (3-5 yrs) and quantitative baseline metrics. | Solid introduction. Background covers Macro, Meso, and Micro levels well with quantitative metrics, but needs stronger literature grounding. | Weak introduction. Background is missing one level of the funnel model or relies heavily on subjective, non-quantitative claims. | Disorganized or merged sections. Lacks Macro-Meso-Micro structure, quantitative baselines, and relevant academic literature. | /10 |
| SOP & Research Objectives Alignment | Flawlessly pairs interrogative SOPs with declarative objectives across all 4 DSR phases (Baseline, Design, Benchmark, Comparative Validation). | SOP and Objectives align across the phases but need minor tweaking in metric specificity or verb precision. | SOPs and Objectives are mismatched, missing 1–2 research phases, or worded too vaguely. | Missing required phases, lack of alignment between questions and objectives, or formulated as simple user-satisfaction surveys. | /10 |
| Theoretical & Conceptual Framework (IPO) | Strong theoretical anchors (DSRM, Big-O, ML principles). IPO framework clearly details data inputs, algorithmic process, and specific performance metrics. | Framework is present with solid theoretical grounding; IPO is clear but missing minor computational metrics. | Weak theoretical anchor. IPO is generic and lacks specific technical/empirical variables. | Framework is incorrect, missing, or copied without application to the proposed algorithmic artifact. | /10 |
| Scope, Delimitations, Terms, & CEU Mechanics | Explicit scope/delimitations; terms defined operationally; strict adherence to CEU layout (margins, Times New Roman 12pt, double space, APA 7th). | Clear scope/delimitations; terms defined mostly operationally; minor formatting or layout errors. | Vague delimitations; terms defined using general dictionary definitions; several formatting errors. | Delimitations missing; dictionary definitions used throughout; disregards CEU layout and manuscript guidelines. | /10 |
| **TOTAL SCORE:** | | | | | /50 |

---

## Module Task: Research Proposal Chapter 2 Draft

Undergraduate Research 1 | CEU Department of Computer Science and Information Technology

Submission Format: PDF document uploaded via Canvas

Total Points: 50 Points

### Task Overview & Instructions

Following our discussion on topical literature structuring, database search protocols, and gap analysis, it is time to write the complete draft of Chapter 2: Review of Related Literature and Studies for your proposed research project.

Important CEU Requirement: Traditional geographical segregation ("Local Literature," "Foreign Literature") is deprecated. Your review must be organized topically / thematically based on sub-concepts, computational paradigms, model architectures, or algorithmic bottlenecks.

### Required Manuscript Formatting

- Paper Size: Short Bond Paper (8.5" × 11")
- Margins: 1.5 inches (Left) for binding allocation; 1.0 inch (Top, Bottom, Right)
- Typography: Times New Roman, 12pt font size
- Spacing: Double-spaced body text
- Citations & Sourcing: APA 7th Edition managed via Zotero/Mendeley. Minimum of 15–20 high-impact, peer-reviewed sources from IEEE Xplore, ACM DL, SpringerLink, or ScienceDirect. At least 80% must be within the last 3–5 years.

### Required Chapter 2 Structure

1. 2.1 to 2.3 Topical / Thematic Sub-sections: Divide your literature review into 3 to 4 logical sub-headings based on your research variables, mathematical models, algorithms, and testbed hardware constraints. Focus on how algorithms function, their computational complexity, and explicit empirical metrics (e.g., latency in ms, RAM footprint, accuracy, throughput) rather than general application summaries.
2. 2.4 Algorithmic Literature Taxonomy Matrix: Construct a structured comparative table placing key related studies alongside your proposed research. The matrix must detail: Author & Year, Algorithm/Model Used, Dataset/Testbed, Evaluated Metrics, and Identified Limitation/Gap.
3. 2.5 Synthesis of Related Literature and Studies: Conclude Chapter 2 with a critical, analytical synthesis (1 to 2 pages) using the 3-step approach:
    - Step 1 (Commonalities & Trends): Summarize established methodologies and accepted state-of-the-art tools.
    - Step 2 (Critical Gap & Bottlenecks): Highlight unaddressed trade-offs, empirical bottlenecks, or hardware constraints in current literature.
    - Step 3 (Research Bridge): Explicitly detail how your proposed study bridges the identified research gap.

### Grading Rubric Matrix (50 Points)

| Evaluation Criteria | Exemplary (100%) | Proficient (80%) | Developing (60%) | Unacceptable (≤40%) | Points |
|---|---|---|---|---|---|
| Topical Structure & Logical Flow | Exclusively topical/thematic. Sub-headings logically map variables, mathematical models, and algorithms seamlessly without geographical segregation. | Topical structure used throughout; sub-headings are mostly logical with minor overlaps between concepts. | Mainly topical, but occasionally lapses into geographical division or generic app summaries. | Strictly uses deprecated geographical formatting (Local vs. Foreign) or lacks thematic organization completely. | /10 |
| Empirical CS Focus & Source Quality | Relies on peer-reviewed repositories (IEEE, ACM, Springer). 80%+ published within 3–5 yrs. Emphasizes explicit empirical benchmark metrics. | Solid peer-reviewed sources; majority published within 3–5 years. Includes performance metrics for most cited works. | Relies heavily on outdated sources (>5 yrs old) or non-peer-reviewed blogs/websites. Lacks empirical benchmark metrics. | Cites non-scientific blogs, tutorials, or generic websites. Completely lacks empirical metrics or computational rigor. | /10 |
| Algorithmic Taxonomy Matrix | Comprehensive matrix clearly comparing key literature against the proposed study across algorithms, datasets, metrics, and explicit limitations. | Matrix is complete and organized, but missing minor benchmark details or 1–2 comparison parameters. | Taxonomy matrix is overly generic, lacks specific computational metrics, or fails to include the proposed study row. | Taxonomy matrix is missing, incomplete, or copied without critical analysis. | /10 |
| Synthesis of Related Literature | Flawlessly executes the 3-step synthesis (Trends → Gap → Research Bridge). Directly justifies the proposed project's computational contribution. | Executes synthesis well; clearly identifies research gaps, but the bridge to the proposed project could be slightly stronger. | Synthesis reads as a generic summary of paragraphs without articulating a clear computational gap or research bridge. | Synthesis is missing, extremely brief (<1 paragraph), or fails to connect cited literature to the project. | /10 |
| APA 7th Citations & CEU Formatting | Flawless APA 7th in-text citations and reference list. Strictly adheres to CEU manuscript mechanics (margins, 12pt Times New Roman, double space). | Minor errors in APA citation style or reference formatting. Adheres closely to CEU layout guidelines. | Frequent APA citation errors. Multiple formatting deviations from CEU manuscript protocols. | Disregards APA citation style entirely. Fails to meet basic CEU manuscript standards. | /10 |
| **TOTAL SCORE:** | | | | | /50 |

---

## Module Task: Research Proposal Chapter 3 Draft

Undergraduate Research 1 | CEU Department of Computer Science and Information Technology

Submission Format: PDF document uploaded via Canvas

Total Points: 50 Points

### Task Overview & Instructions

Following our discussion on Computer Science research methodologies, hardware/software specifications, algorithmic data flows, and empirical evaluation metrics, you are required to submit the complete initial draft of Chapter 3: Methodology for your research proposal.

Replicability Requirement: In CS research, Chapter 3 must be written with technical precision so that an independent researcher can fully replicate your software architecture, training/testing pipeline, experimental testbed, and benchmarking results solely by reading your manuscript.

### Required Manuscript Formatting

- Paper Size: Short Bond Paper (8.5" × 11")
- Margins: 1.5 inches (Left) for binding allocation; 1.0 inch (Top, Bottom, Right)
- Typography: Times New Roman, 12pt font size
- Spacing: Double-spaced body text
- Diagrams & Equations: All UML diagrams, system architectures, and mathematical formulas must be clearly labeled and rendered (use standard equation notation for accuracy/loss/latency equations).

### Required Chapter 3 Outline (CEU CS Template)

Your Chapter 3 submission must strictly include the following sections and subheadings:

1. 3.1 Research Methodology: Explicitly state your framework (Design Science Research Methodology or Quantitative Experimental Benchmarking).
2. 3.2 Subjects of the Study: Define your computational artifacts, dataset instances, file formats, or target system environments.
3. 3.3 Sampling Technique: Detail dataset partition ratios (Train/Val/Test), cross-validation strategy, or data augmentation methods.
4. 3.4 Instruments to Be Used: Detail software profiling tools, benchmarking suites, hardware sensors, or logging frameworks used.
5. 3.5 Procedure of Data Gathering: Chronological protocol for data collection, pre-processing, cache warming, and logging runs.
6. 3.6 Statistical Treatment: Inferential statistical formulas (e.g., paired t-test, ANOVA) and formal null/alternative hypothesis setup.
7. 3.7 Data Analysis: CS performance formulas (Accuracy, Precision, Recall, F1-score, Latency in ms, RAM usage in MB).
8. 3.8 Software System Description:
    1. 3.8.1 System Structure: System architecture diagrams, block diagrams, DFDs, or model layer flowcharts.
    2. 3.8.2 Use Cases: UML Use Case Diagram and detailed use case specifications.
    3. 3.8.3 Materials and Equipment:
        1. Software Requirements: Specific OS, programming languages, libraries, framework versions, and IDEs.
        2. Hardware and Connectivity Requirements: CPU, GPU VRAM limits, system RAM, target testbed devices, and bandwidth conditions.
9. 3.9 Testing and Evaluation: Functional unit/integration test protocols alongside empirical stress/benchmarking evaluation plans.

### Grading Rubric Matrix (50 Points)

| Evaluation Criteria | Exemplary (100%) | Proficient (80%) | Developing (60%) | Unacceptable (≤40%) | Points |
|---|---|---|---|---|---|
| Methodological Framework & Data Pipeline (Sec 3.1–3.3) | DSRM or Experimental design is clearly justified. Dataset subjects, partitioning ratios (Train/Val/Test), and resampling methods are defined with absolute technical clarity. | Methodology is appropriate and subjects are well defined. Minor details missing regarding dataset splitting or data augmentation protocols. | Methodology framework selected is generic or ill-fitting for CS. Dataset details or sampling techniques lack quantitative specifications. | Methodology section is vague, incorrect, or copied from generic social science research templates. | /10 |
| Instruments, Data Gathering & Analysis (Sec 3.4–3.7) | Profiling/benchmarking tools are explicitly named. Procedures ensure repeatable experiments. Statistical hypotheses and CS evaluation metrics (F1, latency, RAM) are fully formulated. | Instruments and data gathering steps are clear. Statistical treatments and CS formulas are provided, but lack minor parameter definitions. | Data gathering steps are too high-level. Missing essential CS evaluation formulas or inferential statistical tests. | Instruments or evaluation metrics are absent or completely irrelevant to Computer Science research. | /10 |
| Software System Design & Use Cases (Sec 3.8.1–3.8.2) | System Architecture/Data Flow diagrams are professional, complete, and technically sound. UML Use Case diagrams and full specifications cover all interactions seamlessly. | System diagrams are clear and logical. Use case diagrams are complete with minor omissions in textual use case details. | System architecture is overly simplistic or unclear. Use case diagrams lack proper UML notation or specifications. | System diagrams or Use Cases are missing, hand-drawn, or completely lack technical depth. | /10 |
| Materials, Equipment & Testing (Sec 3.8.3 & 3.9) | Software/Hardware specs are precise (specific library versions, GPU/CPU models, RAM bounds). Testing protocols rigorously address functional and empirical stress-testing. | Software/Hardware specs are well detailed. Testing protocols cover main system functions and baseline execution performance. | Hardware or software versions are omitted (e.g., just listing "Python" or "Windows"). Testing section is vague or brief. | Fails to specify technical system specifications or includes no testing and evaluation strategy. | /10 |
| CEU Manuscript Formatting & Rigor | Strictly follows all CEU Chapter 3 section titles and sub-headings. Margins (1.5" left, 1" others), typography, spacing, and diagram captions are flawless. | Follows CEU section hierarchy with minor layout, spacing, or figure caption formatting errors. | Deviates from the required CEU template structure or exhibits multiple manuscript formatting errors. | Disregards CEU manuscript guidelines completely. Incomplete template submission. | /10 |
| **TOTAL SCORE:** | | | | | /50 |
