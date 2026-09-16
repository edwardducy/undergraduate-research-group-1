# Chapter 1: Required Sections

1. **Title Page / Proposed Title.** Must reflect a defensible CS project focus, for example algorithmic optimization, dataset model quantization, or experimental benchmarking.
2. **Introduction to the Study.** An executive narrative overview. Introduces the general CS subfield, the core high-level challenge, and the proposed solution and research contribution.
3. **Background of the Study.** A detailed technical case file, structured using the Macro-Meso-Micro Funnel Model.
4. **Statement of the Problem (SOP).** A General Problem Paragraph following the CS template, plus four interrogative SOP questions.
5. **Objectives.** Declarative objectives paired to the SOP and aligned across the four study phases, namely Baseline Analysis, Design and Artifact Creation, Empirical Benchmarking, and Comparative Validation.
6. **Hypothesis.** Null and alternative.
7. **Theoretical Framework.**
8. **Conceptual Framework.** Two Input-Process-Output diagrams. The first, Conceptual Framework of the Study. The second, Software Artifact Conceptual Framework.
9. **Scope and Delimitations.** Scope named explicitly (datasets, parameters, hardware). Delimitations as deliberate boundary choices.
10. **Significance of the Study.** Primary, secondary, and tertiary beneficiaries.
11. **Definitions of Terms.** Operational definitions grounded in how each term behaves in the code, pipeline, or mathematical model, rather than dictionary entries.

# Chapter 2: Required Sections

1. **Topical / Thematic Sub-sections (2.1 to 2.3).** Divide the literature review into 3 to 4 logical sub-headings based on the research variables, mathematical models, algorithms, and testbed hardware constraints. Focus on how algorithms function, their computational complexity, and explicit empirical metrics, for example latency in milliseconds, RAM footprint, accuracy and throughput, rather than general application summaries.
2. **Algorithmic Literature Taxonomy Matrix (2.4).** A structured comparative table placing key related studies alongside the proposed research. The matrix must detail Author and Year, Algorithm or Model Used, Dataset or Testbed, Evaluated Metrics, and Identified Limitation or Gap.
3. **Synthesis of Related Literature and Studies (2.5).** Conclude Chapter 2 with a critical and analytical synthesis of 1 to 2 pages, using the three-step approach. In Step 1, Commonalities and Trends, summarize established methodologies and accepted state-of-the-art tools. In Step 2, Critical Gap and Bottlenecks, highlight unaddressed trade-offs, empirical bottlenecks or hardware constraints in the current literature. In Step 3, Research Bridge, detail explicitly how the proposed study bridges the identified research gap.

# Chapter 3: Required Sections

1. **Research Methodology (3.1).** Specify the overall scientific framework governing the study, being the Design Science Research Methodology (DSRM) 6-phase framework or the Quantitative Experimental or Benchmarking Method.
2. **Subjects of the Study (3.2).** Define computational artifacts, datasets, image samples, network packets, or hardware system instances being tested rather than human respondents.
3. **Sampling Technique (3.3).** Detail the leave-one-disaster-out split, one typhoon event held out per fold, with the fold as the unit of reporting, and the role of the training portion within each fold.
4. **Instruments to Be Used (3.4).** Specify the software profilers, benchmarking suites, and hardware measurement instruments used, each mapped to its measurand, and record instruments that were considered but rejected, with reasons.
5. **Procedure of Data Gathering (3.5).** Provide a chronological step-by-step protocol, being Data Acquisition → Pre-processing and Normalization → Execution Protocol (warmup runs, N trials) → Automated CSV or JSON Log Generation.
6. **Statistical Treatment (3.6).** Define the paired comparison used and its unit of analysis, with interval estimation, at the 0.05 level, alongside the explicit null and alternative pairs, and record which conventional tests were considered and why they were not used at this sample size.
7. **Data Analysis (3.7).** Detail explicit quantitative CS performance formulas (per-task F1 for intent, urgency and span-level named entity recognition, inference latency, peak memory, and the significance result).
8. **Software System Description (3.8).**
   - **(3.8.1) System Structure.** System Architecture, Block, DFD, or Neural Network Layer Diagrams.
   - **(3.8.2) Use Cases.** UML Use Case Diagrams and Use Case Specifications for primary actors and system workers.
   - **(3.8.3) Materials and Equipment.** Complete specs for OS, Python versions, ML frameworks, CUDA versions, CPU/GPU models, VRAM bounds, and target testbed environments.
9. **Testing and Evaluation (3.9).** Describe proof-of-concept (PoC) code unit and integration testing, empirical stress benchmarking across batch sizes, thermal or memory leak behavior, and ISO/IEC 25010 evaluation standards (if applicable).
