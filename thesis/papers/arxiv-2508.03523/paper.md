# FilBench: Can LLMs Understand and Generate Filipino?

 Lester James V. Miranda ††thanks: Equal contributions. Affiliation: Allen Institute for AI    Elyanah Aco††footnotemark: Affiliation: Nara Institute of Science and Technology    Conner Manuel††footnotemark: Affiliation: Together AI    Jan Christian Blaise Cruz ††thanks: Senior authors. Affiliation: SEACrowd Affiliation: MBZUAI    Joseph Marvin Imperial††footnotemark: Affiliation: SEACrowd Affiliation: University of Bath Affiliation: National University, Philippines  Code [filbench/filbench-eval](https://github.com/filbench/filbench-eval "")  Leaderboard [UD-Filipino/filbench-leaderboard](https://huggingface.co/spaces/UD-Filipino/filbench-leaderboard "") 

###### Abstract

Despite the impressive performance of LLMs on English-based tasks, little is known about their capabilities in specific languages such as Filipino. In this work, we address this gap by introducing FilBench, a Filipino-centric benchmark designed to evaluate LLMs across a diverse set of tasks and capabilities in Filipino, Tagalog, and Cebuano. We carefully curate the tasks in FilBench to reflect the priorities and trends of NLP research in the Philippines such as Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation. By evaluating 27 state-of-the-art LLMs on FilBench, we find that several LLMs suffer from reading comprehension and translation capabilities. Our results indicate that FilBench is challenging, with the best model, GPT-4o, achieving only a score of 72.23%. Moreover, we also find that models trained specifically for Southeast Asian languages tend to underperform on FilBench, with the highest-performing model, SEA-LION v3 70B, achieving only a score of 61.07%. Our work demonstrates the value of curating language-specific LLM benchmarks to aid in driving progress on Filipino NLP and increasing the inclusion of Philippine languages in LLM development.

Figure 1: Overview of FilBench. In order to comprehensively assess the full capabilities of LLMs on Philippine languages, we curate an evaluation suite consisting of 4 categories and 12 subtasks across Filipino, Tagalog, and Cebuano based on the research priorities of the Philippine NLP community (§[3.1](#S3.SS1 "3.1 Research Priorities in Filipino NLP ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")).

## 1 Introduction

While large language models (LLMs) have shown impressive performance on a variety of English-based tasks and capabilities, their effectiveness remains largely unexplored for low-resource languages such as Filipino. This knowledge gap exists for two reasons. First, most low-resource languages, especially Filipino-centric benchmarks, developed before the ChatGPT-era ([Gururaja et al., 2023](#bib.bib37 ""), —2022) are ill-posed for current LLM evaluation despite their value in understanding language system capabilities. Second, existing multilingual LLM benchmarks either exclude Filipino entirely ([Liu et al., 2025](#bib.bib48 ""); [Huang et al., 2025](#bib.bib40 ""), inter alia) or fail to provide sufficient task and/or language diversity ([Susanto et al., 2025](#bib.bib72 "")). Filipino is an important language to consider for LLM evaluation not only because of its unique linguistic properties such as its voice marking system ([Bardají et al., 2024](#bib.bib11 "")), but also due to its large speaker population with more than 28 million speakers in the Philippines and over 2 million speakers abroad ([Philippine Statistics Authority, 2020](#bib.bib65 "")).

In this work, we perform a comprehensive study of the strengths and limitations of LLM capabilities on Filipino-centric tasks. We introduce an evaluation suite called FilBench, consisting of 4 categories and 12 diverse sub-tasks that are formulated for LLM evaluation. The choice of tasks to include in FilBench is based on our study of research trends and priorities in Filipino NLP (§[3.1](#S3.SS1 "3.1 Research Priorities in Filipino NLP ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?"),[J](#A10 "Appendix J Research Priorities in Filipino NLP ‣ FilBench: Can LLMs Understand and Generate Filipino?")). Evaluating models on FilBench reveals significant gaps in LLM performance, for instance, in text generation capabilities. The contributions of this study are three-fold:

*   •

```
We close the resource gap by curating Filipino test sets across four broad task categories: Cultural Knowledge Assessment (CK), Classical NLP (CN), Reading Comprehension (RC), and Generation (GN). We transform these datasets into a unified task format aligned with standard LLM evaluation practices across literature. Our evaluation suite, FilBench, consists of test instances across 4 categories and 12 sub-tasks (§[3](#S3 "3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")).
```
*   •

```
We bridge the evaluation gap by evaluating 27 state-of-the-art LLMs on FilBench (§[4](#S4 "4 Results: Performance of State-of-the-Art LLMs on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?")). We find that the best model, GPT-4o, only achieve around 72.23% aggregated performance while the best Southeast-Asian model, SEA-LION v3 70B, only obtains a score of 61.07%.
```
*   •

```
We provide analyses and insights to the strengths and weaknesses of LLMs when presented with Filipino-centric tasks and test cases (§[5](#S5 "5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?")). Notably, we find that text generation suffers the most, with the lowest scores across models due to failure modes such as hallucination and poor instruction-following.
```
FilBench demonstrates the value of constructing language-specific benchmarks to reveal gaps in language model capabilities and benefit the wider speaker community. More importantly, we hope that this work aids in improving the state of Filipino NLP and increase the inclusion of Philippine languages in LLM development.

## 2 Background

#### Languages in the Philippines.

The Philippines is home to approximately 117 million language speakers across more than 185 distinct languages ([Eberhard et al., 2024](#bib.bib28 ""); [McFarland, 2008](#bib.bib53 ""); [Metila et al., 2016](#bib.bib55 "")). One of its official languages is Filipino (FIL), which is a standardized form of Tagalog (TGL) and used mainly in Metro Manila.11 1 The designation of Filipino and Tagalog as separate languages is often a point of contention, although they are linguistically similar ([Villafania, 2007](#bib.bib76 "")). We follow the official view of the Komisyon ng Wikang Filipino (Commision on the Filipino Language) and treat them as separate. Aside from Filipino and Tagalog, Cebuano (CEB) is the second most widely spoken language in the Philippines with over 28 million speakers. It is part of the Visayan language family and is spoken mainly in regions of Cebu, Siquijor, and Bohol among many others ([Pilar et al., 2023](#bib.bib67 "")). As part of the same subgroup of Philippine languages, Tagalog and Cebuano share similar linguistic characteristics such as shared vocabulary and comparable word formulation processes and affixation rules, among others [Bacalla (2019)](#bib.bib8 ""); [Imperial and Kochmar (2023)](#bib.bib42 ""). In our work, we focus on the these three languages because they cover the majority of Filipino speakers, representing approximately 61% of the country’s population ([Philippine Statistics Authority, 2020](#bib.bib65 "")).

#### Task Formulation in LLM Evaluation.

In order to standardize how each test example is presented to an LLM, it must first be formatted into a consistent prompt structure or formulation. Multiple-choice formulation (MCF) is a common standard in evaluating LLMs across a vast array of tasks ([Gu et al., 2024](#bib.bib36 ""); [Fourrier et al., 2024](#bib.bib33 "")). In MCF, a question is posed with answers presented as labeled choices, where scoring is done by comparing the LLM’s choice to the gold label. For evaluating LLMs on generative tasks such as translation, one approach is to write an instruction prompting an LLM to translate a given text from a source language to a target language. Then, the generated output by the LLM is compared against the reference translation using various machine translation metrics ([Papineni et al., 2002](#bib.bib64 ""); [Lin, 2004](#bib.bib47 "")).

Category

Sub-Task

Dataset

Languages

\# Instances

Classical NLP (CN)

Text Classification

Dengue Filipino [Livelo and Cheng (2018)](#bib.bib50 "")

FIL

4,015

BalitaNLP [Buñag and Esquivel (2023)](#bib.bib14 "")

TGL

70,352

SIB-200 [Adelani et al. (2024)](#bib.bib1 "")

CEB, FIL

99

Named-Entity Recognition

CebuaNER [Pilar et al. (2023)](#bib.bib67 "")

CEB

1,310

TLUnified-NER [Miranda (2023)](#bib.bib56 "")

TGL

1,579

Universal NER [Mayhew et al. (2024)](#bib.bib52 "")

CEB, TGL

105

Sentiment Analysis

FiReCS [Cosme and De Leon (2023)](#bib.bib19 "")

FIL

7,340

Cultural Knowledge Assessment (CK)

Regional Knowledge

INCLUDE [Romanou et al. (2024)](#bib.bib69 "")

TGL

510

Factual Knowledge

Global MMLU [Singh et al. (2024)](#bib.bib71 "")

TGL

14,042

Cultural Values

KALAHI [Montalan et al. (2024)](#bib.bib60 "")

TGL

150

Word-sense Disambiguation

StingrayBench [Cahyawijaya et al. (2024)](#bib.bib15 "")

TGL

100

Reading Comprehension (RC)

Readability

Cebuano Readability Corpus [Imperial et al. (2022)](#bib.bib43 "")

CEB

350

Reading Comprehension

Belebele [Bandarkar et al. (2024)](#bib.bib10 "")

CEB, FIL

1,800

NLI

NewsPH NLI [Cruz et al. (2021)](#bib.bib23 "")

FIL

90,000

Generation (GN)

Document translation

NTREX-128 [Federmann et al. (2022)](#bib.bib32 "")

FIL

1,997

Realistic translation

Tatoeba [Tiedemann (2020)](#bib.bib75 "")

CEB, TGL

2,876

Domain-specific transl.

TICO-19 [Anastasopoulos et al. (2020)](#bib.bib5 "")

TGL

971

Table 1: Fine-grained overview of FilBench. Our curation effort involves expert-annotated or validated datasets across a diverse range of sub-tasks and categories basd on a quantitative analysis of the priorities of the Filipino NLP community (§[J](#A10 "Appendix J Research Priorities in Filipino NLP ‣ FilBench: Can LLMs Understand and Generate Filipino?")), allowing us to comprehensively evaluate LLM capabilities on Filipino-centric tasks. 

## 3 The FilBench Evaluation Suite

Our design philosophy for FilBench centers on two core principles: (1) developing an impactful benchmark that aligns with the research priorities within the Philippine context (§[3.1](#S3.SS1 "3.1 Research Priorities in Filipino NLP ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")), ensuring that a model excelling in FilBench is likely to perform effectively across a wide range of Filipino applications and (2) maintaining data quality and richness by incorporating diverse sub-tasks (§[3.2](#S3.SS2 "3.2 FilBench Categories ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")) that were annotated by experts or native speakers. [Table 1](#S2.T1 "Table 1 ‣ Task Formulation in LLM Evaluation. ‣ 2 Background ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows all the datasets and tasks included in FilBench. Example task formulation for each sub-task is shown in Appendix [M](#A13 "Appendix M Task Formulation ‣ FilBench: Can LLMs Understand and Generate Filipino?").

### 3.1 Research Priorities in Filipino NLP

In order to determine which tasks to include in FilBench, we perform a survey of the research trends in NLP research on Philippine languages from 2006–2023. Our methodology involves scraping Scopus-indexed papers and ⋆\\starACL/EMNLP publications and classifying their NLP sub-field based on common ACL tracks. We find that classical NLP tasks such as information extraction and sentiment analysis are widely studied, as well as a variety of translation tasks. Then, we devise a taxonomy consisting of four major categories that encompass more recent trends in Philippine NLP research. More details about our methodology and findings can be found in Appendix [J](#A10 "Appendix J Research Priorities in Filipino NLP ‣ FilBench: Can LLMs Understand and Generate Filipino?").

### 3.2 FilBench Categories

#### Cultural Knowledge Assessment (CK).

This category tests a language model’s ability to recall factual and culturally-specific information. Studies have consistently found that LLMs predominantly trained on English text are strongly biased towards Western values and perspectives, especially when prompted in English [Cao et al. (2023)](#bib.bib16 ""). Cultural misalignment between LLMs and users can lead to unintended harms such as norm violations [Qiu et al. (2025)](#bib.bib68 "") and socio-economic exclusion [Dammu et al. (2024)](#bib.bib25 ""). For CK, we curate a variety of examples that test an LLM’s regional and factual knowledge ([Romanou et al., 2024](#bib.bib69 ""); [Singh et al., 2024](#bib.bib71 "")), understanding of Filipino-centric values ([Montalan et al., 2024](#bib.bib60 "")), and word-sense disambiguation ([Cahyawijaya et al., 2024](#bib.bib15 "")).

#### Classical NLP (CN).

This category encompasses a variety of information extraction and linguistic tasks such as named entity recognition (NER), sentiment analysis, and text categorization that were traditionally performed using specialized trained models. These tasks have been prominent in Philippine NLP research over the past decade ([Roxas et al., 2021](#bib.bib70 "")), and LLMs have recently begun to be employed in this domain ([Ashok and Lipton, 2023](#bib.bib7 ""); [Zhang et al., 2023b](#bib.bib83 ""); [Wang et al., 2023](#bib.bib78 ""), inter alia). For CN, we include expert-annotated NER datasets such as CebuaNER ([Pilar et al., 2023](#bib.bib67 "")), TLUnified-NER ([Miranda, 2023](#bib.bib56 "")), and Universal NER ([Mayhew et al., 2024](#bib.bib52 "")). We also take the Filipino and Cebuano subsets of SIB-200 ([Adelani et al., 2024](#bib.bib1 "")), and the text-only subset of Balita NLP ([Buñag and Esquivel, 2023](#bib.bib14 "")).

#### Reading Comprehension (RC).

This category evaluates a language model’s ability to understand and interpret Filipino text, focusing on tasks such as readability, comprehension, and natural language inference (NLI). These tasks are crucial for assessing how well a model can process and generate human-like understanding of written content. For RC, we include datasets like the Cebuano Readability Corpus ([Imperial et al., 2022](#bib.bib43 "")), Belebele ([Bandarkar et al., 2024](#bib.bib10 "")), and NewsPH NLI ([Cruz et al., 2021](#bib.bib23 "")), which provide a comprehensive evaluation of reading comprehension capabilities in the Filipino context.

#### Generation (GN).

Although generative LLM tasks usually include summarization and conversational generation, evaluation test sets in Filipino are sparse. However, machine translation is one of the most dominant areas of NLP research in the Philippines ([Oco and Roxas, 2018](#bib.bib62 ""); [Baliber et al., 2020](#bib.bib9 ""); [Aji et al., 2023](#bib.bib2 ""), inter alia). Recently, LLMs have gained traction for its use as automatic translators, as opposed to training specialized translation models ([Zhu et al., 2023](#bib.bib85 ""); [He et al., 2024](#bib.bib39 ""); [Alves et al., 2024](#bib.bib4 "")). Hence, we dedicate a large portion of FilBench for testing an LLM’s ability to faithfully translate texts, either from English to Filipino (ENG →\\to FIL) or from Cebuano to English (CEB →\\to ENG). We include a diverse set of test examples, ranging from documents ([Federmann et al., 2022](#bib.bib32 "")), realistic texts collected from volunteers ([Tiedemann, 2020](#bib.bib75 "")), and domain-specific text ([Anastasopoulos et al., 2020](#bib.bib5 "")).

### 3.3 FilBench Scoring

The CN, CK, and RC categories follow the MCF task formulation, so we score an LLM’s performance for these categories by computing the accuracy, i.e., the number of correct answers divided by the total number of examples. For GN, we compute the ROUGE-L score between the LLM-generated text and the gold reference text. All per-category metrics range from 00 to 11. In order to create a representative, single evaluation score, we perform a weighted average based on the number of examples across results as shown in [Equation 1](#S3.E1 "1 ‣ 3.3 FilBench Scoring ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?"):

FilBench Score\=100×∑i∈{CN,CK,GN,RC}ni⋅Si∑i∈{CN,CK,GN,RC}ni\\text{{FilBench} Score}=100\\times\\frac{\\sum\_{i\\in\\{\\text{CN},\\text{CK},\\text{GN},\\text{RC}\\}}n\_{i}\\cdot S\_{i}}{\\sum\_{i\\in\\{\\text{CN},\\text{CK},\\text{GN},\\text{RC}\\}}n\_{i}}

(1)

where nin\_{i} is the number of examples in category ii and SiS\_{i} is the score for category ii.

| FilBench Score                                                                                                                   | Cultural Knowledge | Classical NLP  | Reading Comp.  |                |                |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------- | -------------- | -------------- | -------------- |
| [gpt-4o-2024-08-06](https://platform.openai.com/docs/models "")                                                                  | 72.73±\\pm1.66     | 73.29±\\pm3.01 | 89.03±\\pm2.05 | 80.12±\\pm0.90 | 46.48±\\pm0.60 |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 "") | 67.67±\\pm1.04     | 76.75±\\pm3.04 | 87.28±\\pm0.26 | 72.99±\\pm0.18 | 33.67±\\pm0.71 |
| [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct "")                 | 63.20±\\pm1.05     | 74.31±\\pm3.14 | 87.88±\\pm0.25 | 70.86±\\pm0.18 | 19.75±\\pm0.63 |
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct "")                                                 | 63.08±\\pm0.99     | 73.11±\\pm3.22 | 88.60±\\pm0.24 | 75.62±\\pm0.17 | 14.98±\\pm0.33 |
| [aisingapore/Llama-SEA-LION-v3-70B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-70B-IT "")                           | 61.07±\\pm0.95     | 76.78±\\pm3.02 | 89.99±\\pm0.23 | 53.56±\\pm0.19 | 23.95±\\pm0.34 |
| [Tower-Babel/Babel-83B-Chat](https://huggingface.co/Tower-Babel/Babel-83B-Chat "")                                               | 60.85±\\pm0.96     | 75.21±\\pm3.11 | 88.81±\\pm0.25 | 64.85±\\pm0.19 | 14.53±\\pm0.29 |
| [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct "")                                 | 59.66±\\pm1.17     | 72.16±\\pm3.21 | 90.27±\\pm0.83 | 52.17±\\pm0.28 | 24.03±\\pm0.37 |
| [sail/Sailor2-20B-Chat](https://huggingface.co/sail/Sailor2-20B-Chat "")                                                         | 58.61±\\pm1.06     | 66.43±\\pm3.41 | 89.03±\\pm0.25 | 63.03±\\pm0.19 | 15.95±\\pm0.38 |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct "")                                                 | 57.88±\\pm1.45     | 66.83±\\pm3.45 | 89.32±\\pm1.99 | 70.59±\\pm0.18 | 4.79±\\pm0.17  |
| [aisingapore/Gemma-SEA-LION-v3-9B-IT](https://huggingface.co/aisingapore/Gemma-SEA-LION-v3-9B-IT "")                             | 56.14±\\pm1.53     | 64.44±\\pm3.43 | 88.55±\\pm0.25 | 54.46±\\pm0.20 | 17.10±\\pm2.25 |

Table 2: Performance of state-of-the-art LLMs on Filipino-centric tasks. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (1.5B to 400B), and accessibility (open-source vs. commercial). Full results can be found in [Table 8](#A3.T8 "Table 8 ‣ Appendix C Full results on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?"). 

## 4 Results: Performance of State-of-the-Art LLMs on FilBench

In order to understand what kind of LLMs perform well in Filipino, we select a variety of open-source and commercial LLMs to ensure broad coverage across parameter sizes and language capabilities. We also include a number of SEA-specific models that were trained to cater to Southeast Asian languages, including Filipino. A total of 27 models are chosen for evaluation. [Table 6](#A1.T6 "Table 6 ‣ Appendix A Details of Models Evaluated on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?") in the Appendix shows the full details of the evaluated models.

[Table 2](#S3.T2 "Table 2 ‣ 3.3 FilBench Scoring ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")shows the scores obtained by the top ten models on FilBench. The full results for all 27 models can be seen in [Table 8](#A3.T8 "Table 8 ‣ Appendix C Full results on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?") of the Appendix. The best performing model is GPT-4o (72.23%), closely followed by Llama 4 Maverick (67.67%). Moreover, the highest scoring open-source dense model is Qwen2.5 72B (63.08%), while SEA-LION v3 70B is the best SEA-specific model (61.07%).

Figure 2: Parameter-efficiency of LLMs with respect to FilBench. SEA-specific models are at the Pareto frontier of parameter-efficiency. However, the best SEA-specific model still underperforms on FilBench with a score of 61.07%. 

#### Finding #1: Larger models dominate FilBench.

[Figure 2](#S4.F2 "Figure 2 ‣ 4 Results: Performance of State-of-the-Art LLMs on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the FilBench score to Parameter Size (B) for several dense open-source language models with known sizes. Our findings suggest that parameter size strongly correlates with FilBench performance, with a Spearman ρ\\rho of 0.810. However, this correlation is not perfect as we observe some smaller models to be competitive with larger counterparts as observed in Qwen 2.5 32B having similar performance to Llama 3.1 70B.

Figure 3: Effect of language-specific finetuning. Performance comparison between a base instruction model and its finetuned version (SEA-LION v3). Language-specific finetuning from a multilingual base model can improve performance on FilBench. 

Figure 4: Performance trends in FilBench. Model performance (aggregated) across the four categories of FilBench, along with the average performance for each category. LLMs tend to perform well in Classical NLP tasks, but suffer poor performance in Generation tasks. 

#### Finding #2: Language-specific finetuning improves FilBench performance.

SEA-specific models tend to be more parameter-efficient as they perform better than non-specialized LLMs on FilBench. This trend is more apparent for smaller models within the 7B to 9B range, as shown in [Figure 2](#S4.F2 "Figure 2 ‣ 4 Results: Performance of State-of-the-Art LLMs on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?"). In addition, SEA-specific models such as Sailor2 20B, SEA-LION v3 9B, and SeaLLMs v3 1.5B sit near the Pareto frontier in terms of performance and size. Despite these results, the best performing SEA-specific model still underperforms on FilBench, as in the case of SEA-LION v3 70B with a score of 61.07%. In addition, we also find that continuous finetuning of an existing multilingual LLM on SEA-specific data improve FilBench performance, as observed in the SEA-LION model family, which are finetunes of Llama 3.1 and Gemma 2, in [Figure 3](#S4.F3 "Figure 3 ‣ Finding #1: Larger models dominate FilBench. ‣ 4 Results: Performance of State-of-the-Art LLMs on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?"). These findings show a promising direction for building Filipino-focused LLMs, as it provides a resource-efficient path without training entirely new models from scratch.

#### Finding #3: Models tend to follow a consistent trend in FilBench performance across categories.

[Figure 4](#S4.F4 "Figure 4 ‣ Finding #1: Larger models dominate FilBench. ‣ 4 Results: Performance of State-of-the-Art LLMs on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?") suggests that most models have a consistent trend in FilBench performance, i.e., they tend to score well in CK, CN, and RC categories, yet are worse on GN. This is more apparent in generative (GN) tasks, where most models tend to struggle with an average performance of 17.03%. On the other hand, models tend to perform well in CK (60.72%) and CN (85.75%) categories, indicating high-level of understanding of Filipino-centric cultural entities and values. Model performance on CK tends to be more dispersed with one of the largest standard deviation (±\\pm13.14). These findings suggest that model capabilities are not uniform across categories for Filipino, indicating significant room for improvement on model training.

## 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks?

### 5.1 Do models consistently agree with one another on Filipino language tasks?

#### Set-up.

In order to understand whether models are consistently reliable in answering test cases in FilBench, we compute the inter-rater reliability using Fleiss’ κ\\kappa across a given set of models. The first group consists of SEA-specific models (see models marked with in [Table 6](#A1.T6 "Table 6 ‣ Appendix A Details of Models Evaluated on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?")) while the second group includes the top-five non-SEA models on FilBench ([Table 2](#S3.T2 "Table 2 ‣ 3.3 FilBench Scoring ‣ 3 The FilBench Evaluation Suite ‣ FilBench: Can LLMs Understand and Generate Filipino?")). To increase granularity, we compute the Fleiss’ κ\\kappa for each sub-task.

#### Results.

The results in [Table 3](#S5.T3 "Table 3 ‣ Results. ‣ 5.1 Do models consistently agree with one another on Filipino language tasks? ‣ 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?") show that the SEA-specific group consistently demonstrate higher agreement on several sub-tasks than the non-SEA models. This suggests that SEA-specific finetuning can improve model reliability and consistency in outputs. However, both groups show alarming disagreement on cultural tasks, indicating fundamentally different interpretations of culturally-nuanced content. We show some examples of model disagreement for the SEA-specific group in Appendix [G](#A7 "Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?"). This implies that while regional specialization improves reliability, deeper cultural adaptation and more sophisticated training approaches may be needed to achieve reliable performance on Filipino.

Sub-Task

Model Agreement (Fleiss’ κ\\kappa)

SEA-Specific

Top-Five

Classical NLP (CN)

Text Classification

0.513

0.174

Named-Entity Recog.

0.639

0.273

Sentiment Analysis

0.598

0.212

Cultural Knowledge (CK)

Regional Knowledge

0.393

0.209

Factual Knowledge

0.224

0.115

Cultural Values

0.403

0.187

Word-sense Disamb.

0.072

-0.041

Reading Comprehension (RC)

Readability

0.207

-0.119

Reading Comp.

0.377

0.248

NLI

0.438

0.201

Table 3: Inter-model agreement on MCF-based tasks. Inter-model agreement, as measured by Fleiss’ κ\\kappa, for each sub-task in FilBench. Despite good performance on FilBench, models tend to disagree with one another, highlighting gaps in reliability. 

Figure 5: Effect of few-shot examples on Generation score. Incorporating a single (k\=1k=1) demonstration during generation generally mitigates poor model performance. 

### 5.2 Why do models fail in Generation tasks?

#### Set-up.

Given the low performance of most models in GN tasks, we qualitiatively analyze example outputs in order to provide a taxonomy of common failure cases in Filipino generative tasks. In addition, we also assess whether we can mitigate low Generation performance by incorporating demonstrations or few-shot examples in the prompt. To do so, we include k\={1,3,5}k=\\{1,3,5\\} examples on the prompt during evaluation.

 Failure mode 

 Example 

 Poor instruction- 

 ENG: Tom winked at Mary. FIL: 

 following 

 Gold: Si Tom ay kumindat kay Mary. 

 Pred: Tom mutiara kepada Mary. (mutiara = to wink). 

 Explanation: Output is not in Filipino. 

 Overgeneration 

 ENG: Did you hear the lecture in that class yesterday? asked the student. FIL: 

 Gold: Narinig mo ba ang lecture sa klaseng yan kahapon?, tanong ng estudyante. 

 Pred: Sige, narinig ko na. ANG LECTURA YESTERDAY BA? Saan po ba ito naganap? (continued…) 

 Explanation: Output is incoherent and generated highly verbose text. 

 Hallucination 

 ENG: The more you read the book, the less you will understand it. FIL: 

 Gold: Nang mas na binabasa mo ang libro, menos mong maiintindihan. 

 Pred: Ang higit na nagbabasa ng libro, mas nagiging delikado at hindi kayang tukuyin ang kahulugan. 

 Explanation: Output hallucinated the concept of danger (delikado). 

Table 4: Common failure modes in translation tasks in the Generation category. We find three common failure modes in most models in the Generation category of FilBench. The predictions in this table show the outputs of SEA-LION v3 70B, the currently best-performing SEA-specific model. 

#### Results.

We find common failure modes on Generation tasks in FilBench and show some examples from the outputs of the best-performing SEA-specific model, SEA-LION v3 70B, in [Table 4](#S5.T4 "Table 4 ‣ Set-up. ‣ 5.2 Why do models fail in Generation tasks? ‣ 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?"):

*   •

```
Poor instruction-following. When presented with Generation tasks from FilBench, models tend to misinterpret instructions or generate translations in an incorrect target language.
```
*   •

```
Overgeneration. In the case of translation, models tend to produce overly verbose text than necessary, usually until the maximum generation length is reached. This usually results in incoherent text even if properly translated to the target language.
```
*   •

```
Hallucination. Models often fail in Generation tasks due to spurious artifacts in the generated text. These tend to diminish the faithfulness of the model’s output, especially in the case of translation tasks.
```
By manually inspecting a sample of 100 failure cases from GPT-4o, we find that overgeneration and poor instruction-following are the most dominant, with 47% and 34% respectively, while hallucination occurs 19% of the time. We hypothesize that overgeneration can be caused by training data imbalance, as suggested in [Bawden and Yvon (2023)](#bib.bib12 "") and [Alves et al. (2023)](#bib.bib3 "") work in the case for BLOOM and LLaMA 7B. We defer the ablation of training data quality and its effect on translation performance to future work.

In addition, we also find that few-shot prompting can mitigate drop in Generation performance ([Figure 5](#S5.F5 "Figure 5 ‣ Results. ‣ 5.1 Do models consistently agree with one another on Filipino language tasks? ‣ 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?")). We find that poor instruction-following, which is common especially in zero-shot ENG →\\to FIL decreases once examples are provided. Full few-shot experiment results are shown in Appendix [D](#A4 "Appendix D Generation Few-shot Results ‣ FilBench: Can LLMs Understand and Generate Filipino?"). Despite these results, model performance on generation tasks remain generally poor, with frequent instances of overgeneration and semantically inaccurate translations. We further explain reasons for this using the Tatoeba dataset, which models consistently underperform on, in Appendix [I](#A9 "Appendix I Analysis of Generation Failure Modes for Translation Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?").

### 5.3 Human evaluation of FilBench

When curating test instances for FilBench, we ensured that the majority of sources in underwent human annotation and evaluation. However, we want to verify that strong agreement between native speakers and the gold answers persisted after the instances were converted into our task-specific formulations (Appendix [M](#A13 "Appendix M Task Formulation ‣ FilBench: Can LLMs Understand and Generate Filipino?")).

#### Set-up.

In order to evaluate the agreement between native speakers and FilBench’s gold answers, we sample 150 instances from FilBench with similar sub-task distribution. Then, three authors (all native speakers of Filipino) served as annotators to label each instance. For MCF tasks, the annotators choose the letter-option of the correct answer. For GN tasks, we provide the annotators with a free-form text field to input their answers. Then, we compute the inter-annotator agreement via Fleiss’ κ\\kappa across two settings: (i) among annotators (intra-group) and the (ii) majority response of human annotators to FilBench’s gold answer (inter-group). For GN, we compute the average ROUGE-L score for each annotator pair (intra-group) and the average of the ROUGE-L score between the gold reference translation and each of the annotator translation (inter-group).

Task Formulation

Intra-group

Inter-group

MCF, Fleiss’ κ\\kappa

0.8163

0.8756

Generation, Avg. ROUGE-L

0.7604

0.7806

Table 5: Inter-rater agreement of native-speakers to a subset of FilBench. We show that FilBench instances have a strong agreement with native speakers on both MFC-based (Cultural Knowledge, Classical NLP, Reading Comprehension) and Generation tasks. 

#### Results.

[Table 5](#S5.T5 "Table 5 ‣ Set-up. ‣ 5.3 Human evaluation of FilBench ‣ 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the agreement scores among the three annotators (intra-group) and their overall agreement with the gold reference answer (inter-group). The Fleiss’ κ\\kappa indicate high agreement ([Landis and Koch, 1977](#bib.bib46 "")), suggesting that the instances in FilBench are reliable and aligns with native-speakers. In addition, the ROUGE-L scores between annotators are also high, suggesting that the Generation instances can be reproducibly translated. Furthermore, the inter-group ROUGE-L score supports this claim, as evidenced by similar performance given that most of the Generation instances were originally translated by other native speakers. In general, the results suggest that the agreement between native speakers and the gold answers are preserved even after converting it into our task-specific formulations.

## 6 Discussion

#### On what to prioritize next when collecting data for Filipino-centric post-training.

Our findings, through FilBench, reveal critical gaps in existing LLM’s capabilities to process Filipino text, particularly in generation tasks where the best models achieved ≤60%\\leq 60\\% performance. In addition, we also find that continuous finetuning helps improve LLM performance on FilBench. This suggests that post-training data collection efforts should prioritize high-quality translation pairs and generative content across diverse domains. Furthermore, gathering training data from a wide range of Philippine languages, beyond just Tagalog, can enhance the performance of LLMs as demonstrated in [Buzaaba et al. (2025)](#bib.bib13 ""). We posit that this can be achieved by taking advantage of cross-lingual transfer ([Artetxe et al., 2020](#bib.bib6 "")) across typologically-similar languages.

#### On the importance of building language-community specific evaluation suites.

Our findings strongly reinforce the necessity of developing language-community specific evaluation suites rather than relying on general multilingual benchmarks. FilBench demonstrates that even state-of-the-art models like GPT-4o achieve only 75.56% overall performance, indicating that Filipino presents unique challenges not captured in broader evaluations. By creating focused evaluation suites like FilBench, the research community can more accurately identify model limitations and track progress in ways that respect the linguistic particularities of Philippine languages. Furthermore, the performance variations across Filipino, Tagalog, and Cebuano emphasize the importance of fine-grained attention to linguistic diversity even within regions. This points to the need for training approaches that recognize intra-regional linguistic boundaries rather than treating Southeast Asian languages as a homogeneous group.

## 7 Related Work

#### State of LLM Research for Philippine Languages.

Progress in the NLP research landscape for Philippine languages such as Tagalog and Cebuano is seeing a promising growth, which can be attributed to democratization and access to LLM artifacts, particularly data and open models ([Lovenia et al., 2024](#bib.bib51 "")). The first works to release open-source artifacts include tasks such as sentiment analysis, hate speech detection, and natural language inference (NLI) ([Cruz and Cheng, 2019](#bib.bib20 ""); [Cruz and Cheng, 2020](#bib.bib21 ""); [Cruz and Cheng, 2022](#bib.bib22 "")). Further release of multilingual LLMs supporting Tagalog, allowed researchers to explore further linguistic phenomena from classical NLP tasks ([Pilar et al., 2023](#bib.bib67 ""); [Mayhew et al., 2024](#bib.bib52 ""), inter alia) to language model applications ([Catapang and Visperas, 2023](#bib.bib17 ""); [Montalan et al., 2024](#bib.bib60 "")).

#### Language-specific LLM Evaluation Benchmarks.

Global research communities are following the trend of releasing language-specific benchmarks in order to assess and track LLM progress in their respective languages. Notable examples include AfroBench for African languages ([Ojo et al., 2023](#bib.bib63 "")), BenCzechMark for the Czech ([Fajcik et al., 2024](#bib.bib31 "")), the Open Arabic LLM Leaderboard for Arabic ([El Filali et al., 2025](#bib.bib29 "")) and Le Leaderboard for French ([Mohamad Alhajar, 2024](#bib.bib58 "")). These benchmarks usually contain curated tasks that may include translated versions of existing datasets or subsets of larger evaluation suites. FilBench takes inspiration from these efforts by curating a comprehensive evaluation suite for Philippine languages.

Region-specific benchmarks also exist such as SeaBench and SeaExam ([Liu et al., 2025](#bib.bib48 "")) for Southeast Asia, although they do not contain any Filipino-specific subset. The most recent effort related to FilBench is Batayan [Montalan et al. (2025)](#bib.bib59 ""), which is part of SEA-HELM ([Susanto et al., 2025](#bib.bib72 "")). FilBench takes a complementary approach by systematically curating existing benchmarks, enabling not only greater efficiency in resource utilization but also facilitating a wider diversity of task types and expanded coverage of Philippine languages beyond Filipino (in this case, Tagalog and Cebuano).

## 8 Conclusion

In this work, we present a comprehensive evaluation of LLMs on Filipino-centric tasks to investigate their strengths and limitations, which still remains underexplored. We curate a benchmark called FilBench across four categories and 12 sub-tasks, based on our analyses of research priorities in Philippine NLP. Through FilBench, we discovered weaknesses in the current open and commercial state-of-the-art LLMs, such as low reliability and poor generation capabilities. FilBench emphasizes the value of creating language-specific LLM benchmarks, as it allows us to find promising avenues for models to improve their Filipino-centric performance. Specifically, this includes language-specific post-training and collecting relevant training datasets for text generation. We hope that FilBench aids in driving the progress in Filipino NLP.

## Limitations

#### Influence of training data on downstream FilBench performance.

When selecting models for evaluation on FilBench, we categorized based on whether these models were originally presented as multilingual or SEA-specific, rather than considering the proportion of Filipino-centric training data used for fine-tuning. The training data provenance is difficult to track, especially for closed-source models. This explains why models that top other multilingual leaderboards such as Aya Expanse 32B ([Dang et al., 2024](#bib.bib26 "")) perform poorly on FilBench, because it was not explicitly trained on Filipino. Our experiments hinge on the assumption that cross-lingual transfer ([Artetxe et al., 2020](#bib.bib6 "")) happens during different states of language modeling, as evidenced in [Chirkova and Nikoulina (2024)](#bib.bib18 ""). We leave the systematic exploration of the influence of the proportion of language-specific training data to a language-specific benchmark (i.e., Filipino-centric training data to FilBench performance) for future work.

#### Focus on Tagalog and Cebuano.

While some of the datasets in FilBench support other Philippine languages (i.e. Ilokano for Belebele), data for these (labeled or otherwise) remain sparse. We focus our suite on the relatively better-resourced Filipino and Cebuano, with the hope of supporting more languages once more datasets become available. Future work might explore data augmentation techniques and other community-driven data collection initiatives to extend FilBench’s coverage to languages like Hiligaynon, Bikolano, and others.

## Ethics Statement

The development and evaluation of language technologies for Filipino, Cebuano, and other Philippine languages addresses important issues of linguistic inclusion and technological access. FilBench aims to support the development of more capable Filipino language technologies that can serve the significant population of Filipino speakers worldwide. The benchmark deliberately includes culturally-specific knowledge assessment to address known biases in LLMs toward Western values and content. This highlights the importance of evaluating models in their cultural context rather than assuming universal applicability. Datasets included in FilBench are from publicly accessible sources, and the authors obtained explicit approval from dataset creators when license information was unclear. Overall, we do not see any serious ethical issues with this work.

## Acknowledgments

The authors would like to thank Cohere Labs for providing credits through the Cohere Research Grant to run the Aya model series, and Together AI for additional computational credits for running several open models. We also acknowledge the Hugging Face team, particularly the OpenEvals team (Clémentine Fourrier and Nathan Habib) and Daniel van Strien, for their support in publishing the FilBench blog post. Finally, we thank the reviewers from the May ARR cycle for their helpful feedback and insightful comments that improved this paper.

## References

*   Adelani et al. (2024) David Ifeoluwa Adelani, Hannah Liu, Xiaoyu Shen, Nikita Vassilyev, Jesujoba O. Alabi, Yanke Mao, Haonan Gao, and En-Shiun Annie Lee. 2024. SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 226–245.
*   Aji et al. (2023) Alham Fikri Aji, Jessica Zosa Forde, Alyssa Marie Loo, Lintang Sutawika, Skyler Wang, Genta Indra Winata, Zheng-Xin Yong, Ruochen Zhang, A. Seza Doğruöz, Yin Lin Tan, and Jan Christian Blaise Cruz. 2023. [Current status of NLP in south East Asia with insights from multilingualism and language diversity](https://doi.org/10.18653/v1/2023.ijcnlp-tutorials.2 ""). In *Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: Tutorial Abstract*, pages 8–13, Nusa Dua, Bali. Association for Computational Linguistics.
*   Alves et al. (2023) Duarte Alves, Nuno Guerreiro, João Alves, José Pombal, Ricardo Rei, José de Souza, Pierre Colombo, and Andre Martins. 2023. [Steering large language models for machine translation with finetuning and in-context learning](https://doi.org/10.18653/v1/2023.findings-emnlp.744 ""). In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 11127–11148, Singapore. Association for Computational Linguistics.
*   Alves et al. (2024) Duarte M Alves, José Pombal, Nuno M Guerreiro, Pedro H Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, and 1 others. 2024. Tower: An open multilingual large language model for translation-related tasks. *arXiv preprint arXiv:2402.17733*.
*   Anastasopoulos et al. (2020) Antonios Anastasopoulos, Alessandro Cattelan, Zi-Yi Dou, Marcello Federico, Christian Federmann, Dmitriy Genzel, Franscisco Guzmán, Junjie Hu, Macduff Hughes, Philipp Koehn, Rosie Lazar, Will Lewis, Graham Neubig, Mengmeng Niu, Alp Öktem, Eric Paquin, Grace Tang, and Sylwia Tur. 2020. TICO-19: the translation initiative for COvid-19. In *Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020*.
*   Artetxe et al. (2020) Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. [On the cross-lingual transferability of monolingual representations](https://doi.org/10.18653/v1/2020.acl-main.421 ""). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4623–4637, Online. Association for Computational Linguistics.
*   Ashok and Lipton (2023) Dhananjay Ashok and Zachary C Lipton. 2023. Promptner: Prompting for named entity recognition. *arXiv preprint arXiv:2305.15444*.
*   Bacalla (2019) Lita Bacalla. 2019. [Morpo-analisis ng wikang tagalog at wikang sugbuanun’g binisaya: Pahambing na pag-aaral](https://doi.org/10.5861/ijrse.2019.4902 ""). *International Journal of Resarch Studies in Education*, 8:55–65.
*   Baliber et al. (2020) Renz Iver Baliber, Charibeth Cheng, Kristine Mae Adlaon, and Virgion Mamonong. 2020. [Bridging Philippine languages with multilingual neural machine translation](https://doi.org/10.18653/v1/2020.loresmt-1.2 ""). In *Proceedings of the 3rd Workshop on Technologies for MT of Low Resource Languages*, pages 14–22, Suzhou, China. Association for Computational Linguistics.
*   Bandarkar et al. (2024) Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 749–775.
*   Bardají et al. (2024) Maria Bardají, Elsie Or, Angelina Aquino, and Nikolaus Himmelmann. 2024. The challenges of symmetrical voice languages for universal dependencies. In *Proceedings of the 15th International Conference of the Association for Linguistic Typology*.
*   Bawden and Yvon (2023) Rachel Bawden and François Yvon. 2023. [Investigating the translation performance of a large multilingual language model: the case of BLOOM](https://aclanthology.org/2023.eamt-1.16/ ""). In *Proceedings of the 24th Annual Conference of the European Association for Machine Translation*, pages 157–170.
*   Buzaaba et al. (2025) Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, and Christiane Fellbaum. 2025. Lugha-llama: Adapting large language models for african languages. *arXiv preprint arXiv:2504.06536*.
*   Buñag and Esquivel (2023) Kenrick Lance Buñag and Rosanna Esquivel. 2023. Transformer-based conditional language models to generate filipino news article. In *Proceedings of the International Conference on International Engineering and Operations Management*.
*   Cahyawijaya et al. (2024) Samuel Cahyawijaya, Ruochen Zhang, Holy Lovenia, Jan Christian Blaise Cruz, Elisa Gilbert, Hiroki Nomoto, and Alham Fikri Aji. 2024. Thank you, stingray: Multilingual large language models can not (yet) disambiguate cross-lingual word sense. *arXiv preprint arXiv:2410.21573*.
*   Cao et al. (2023) Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. [Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study](https://doi.org/10.18653/v1/2023.c3nlp-1.7 ""). In *Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP)*, pages 53–67.
*   Catapang and Visperas (2023) Jasper Kyle Catapang and Moses Visperas. 2023. [Emotion-based morality in Tagalog and English scenarios (EMoTES-3K): A parallel corpus for explaining (im)morality of actions](https://aclanthology.org/2023.nlp4dh-1.1/ ""). In *Proceedings of the Joint 3rd International Conference on Natural Language Processing for Digital Humanities and 8th International Workshop on Computational Linguistics for Uralic Languages*, pages 1–6, Tokyo, Japan. Association for Computational Linguistics.
*   Chirkova and Nikoulina (2024) Nadezhda Chirkova and Vassilina Nikoulina. 2024. [Zero-shot cross-lingual transfer in instruction tuning of large language models](https://aclanthology.org/2024.inlg-main.53/ ""). In *Proceedings of the 17th International Natural Language Generation Conference*, pages 695–708, Tokyo, Japan. Association for Computational Linguistics.
*   Cosme and De Leon (2023) Camilla Johnine Cosme and Marlene De Leon. 2023. Sentiment analysis of code-switched filipino-english product and service reviews using transformers-based large language models. In *Proceedings of World Conference on Information Systems for Business Management*, pages 123–135.
*   Cruz and Cheng (2019) Jan Christian Blaise Cruz and Charibeth Cheng. 2019. [Evaluating language model finetuning techniques for low-resource languages](https://arxiv.org/abs/1907.00409 ""). *arXiv preprint arXiv:1907.00409*.
*   Cruz and Cheng (2020) Jan Christian Blaise Cruz and Charibeth Cheng. 2020. [Establishing baselines for text classification in low-resource languages](https://arxiv.org/abs/2005.02068 ""). *arXiv preprint arXiv:2005.02068*.
*   Cruz and Cheng (2022) Jan Christian Blaise Cruz and Charibeth Cheng. 2022. [Improving large-scale language models and resources for Filipino](https://aclanthology.org/2022.lrec-1.703/ ""). In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 6548–6555, Marseille, France. European Language Resources Association.
*   Cruz et al. (2021) Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Cheng. 2021. Exploiting news article structure for automatic corpus generation of entailment datasets. In *PRICAI 2021: Trends in Artificial Intelligence*, pages 86–99.
*   Cucio and Hennig (2025) Micholo Cucio and Tristan Hennig. 2025. Artificial Intelligence and the Philippine Labor Market: Mapping Occupational Exposure and Complementarity. Technical report, International Monetary Fund.
*   Dammu et al. (2024) Preetam Prabhu Srikar Dammu, Hayoung Jung, Anjali Singh, Monojit Choudhury, and Tanu Mitra. 2024. [“they are uncultured”: Unveiling covert harms and social threats in LLM generated conversations](https://doi.org/10.18653/v1/2024.emnlp-main.1134 ""). pages 20339–20369.
*   Dang et al. (2024) John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, and 1 others. 2024. Aya expanse: Combining research breakthroughs for a new multilingual frontier. *arXiv preprint arXiv:2412.04261*.
*   Dou et al. (2025) Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, and 1 others. 2025. Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs. *arXiv preprint arXiv:2502.12982*.
*   Eberhard et al. (2024) David M. Eberhard, Gary F. Simons, and Charles D. Fennig, editors. 2024. [*Ethnologue: Languages of the World*](http://www.ethnologue.com ""), 27 edition. SIL International, Dallas, Texas.
*   El Filali et al. (2025) Ali El Filali, Manel ALOUI, Tarique Husaain, Ahmed Alzubaidi, Basma El Amel Boussaha, Ruxandra Cojocaru, Clémentine Fourrier, Nathan Habib, and Hakim Hacid. 2025. Open arabic llm leaderboard 2. https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard.
*   Eronen et al. (2023) Juuso Eronen, Michal Ptaszynski, and Fumito Masui. 2023. Zero-shot cross-lingual transfer language selection using linguistic similarity. *Information Processing and Management*, 60(3):103250.
*   Fajcik et al. (2024) Martin Fajcik, Martin Docekal, Jan Dolezal, Karel Ondrej, Karel Beneš, Jan Kapsa, Pavel Smrz, Alexander Polok, Michal Hradis, Zuzana Neverilova, and 1 others. 2024. Benczechmark: A czech-centric multitask and multimetric benchmark for large language models with duel scoring mechanism. *arXiv preprint arXiv:2412.17933*.
*   Federmann et al. (2022) Christian Federmann, Tom Kocmi, and Ying Xin. 2022. NTREX-128 – news test references for (mt) evaluation of 128 languages. In *Proceedings of the First Workshop on Scaling Up Multilingual Evaluation*, pages 21–24.
*   Fourrier et al. (2024) Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. 2024. Open llm leaderboard v2. [https://huggingface.co/spaces/open-llm-leaderboard/open\_llm\_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard "").
*   Grattafiori et al. (2024) Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The LLaMa 3 herd of models. *arXiv preprint arXiv:2407.21783*.
*   Group (2024) Boston Consulting Group. 2024. [Consumers know more about ai than businesses think](https://www.bcg.com/publications/2024/consumers-know-more-about-ai-than-businesses-think "").
*   Gu et al. (2024) Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. 2024. OLMES: A standard for language model evaluations. *arXiv preprint arXiv:2406.08446*.
*   Gururaja et al. (2023) Sireesh Gururaja, Amanda Bertsch, Clara Na, David Widder, and Emma Strubell. 2023. [To build our future, we must know our past: Contextualizing paradigm shifts in natural language processing](https://doi.org/10.18653/v1/2023.emnlp-main.822 ""). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 13310–13325, Singapore. Association for Computational Linguistics.
*   Habib et al. (2023) Nathan Habib, Clémentine Fourrier, Hynek Kydlíček, Thomas Wolf, and Lewis Tunstall. 2023. [Lighteval: A lightweight framework for llm evaluation](https://github.com/huggingface/lighteval "").
*   He et al. (2024) Zhiwei He, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Yujiu Yang, Rui Wang, Zhaopeng Tu, Shuming Shi, and Xing Wang. 2024. [Exploring human-like translation strategy with large language models](https://doi.org/10.1162/tacl_a_00642 ""). *Transactions of the Association for Computational Linguistics*, 12:229–246.
*   Huang et al. (2025) Xu Huang, Wenhao Zhu, Hanxu Hu, Conghui He, Lei Li, Shujian Huang, and Fei Yuan. 2025. Benchmax: A comprehensive multilingual evaluation suite for large language models. *arXiv preprint arXiv:2502.07346*.
*   Hurst et al. (2024) Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. GPT-4o System Card. *arXiv preprint arXiv:2410.21276*.
*   Imperial and Kochmar (2023) Joseph Marvin Imperial and Ekaterina Kochmar. 2023. [Automatic readability assessment for closely related languages](https://doi.org/10.18653/v1/2023.findings-acl.331 ""). In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 5371–5386, Toronto, Canada. Association for Computational Linguistics.
*   Imperial et al. (2022) Joseph Marvin Imperial, Lloyd Lois Antonie Reyes, Michael Antoinio Ibañez, Ranz Sapinit, and Mohammed Hussien. 2022. A baseline readability model for cebuano. In *Proceedings of the 17th Workshop on Innovative Use of NLP for Building Educational Applications*.
*   Jiang et al. (2024) Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. *arXiv preprint arXiv:2401.04088*.
*   Kwon et al. (2023) Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pages 611–626.
*   Landis and Koch (1977) J Richard Landis and Gary G. Koch. 1977. [The measurement of observer agreement for categorical data.](https://api.semanticscholar.org/CorpusID:11077516 "") *Biometrics*, 33 1:159–74.
*   Lin (2004) Chin-Yew Lin. 2004. [ROUGE: A package for automatic evaluation of summaries](https://aclanthology.org/W04-1013/ ""). In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.
*   Liu et al. (2025) Chaoqun Liu, Wenxuan Zhang, Jiahao Ying, Mahani Aljunied, Anh Tuan Luu, and Lidong Bing. 2025. Seaexam and seabench: Benchmarking llms with local multilingual questions in southeast asia. *arXiv preprint arXiv:2502.06298*.
*   Liu and Wang (2024) Yan Liu and He Wang. 2024. *Who on Earth Is Using Generative AI?* World Bank.
*   Livelo and Cheng (2018) Evan Dennison Livelo and Charibeth Cheng. 2018. Intelligent dengue infoveillance using gated recurrent neural learning and cross-label frequencies. In *2018 IEEE International Conference on Agents*.
*   Lovenia et al. (2024) Holy Lovenia, Rahmad Mahendra, Salsabil Maulana Akbar, Lester James Validad Miranda, Jennifer Santoso, Elyanah Aco, Akhdan Fadhilah, Jonibek Mansurov, Joseph Marvin Imperial, Onno P. Kampman, Joel Ruben Antony Moniz, Muhammad Ravi Shulthan Habibi, Frederikus Hudi, Railey Montalan, Ryan Ignatius Hadiwijaya, Joanito Agili Lopo, William Nixon, Börje F. Karlsson, James Jaya, and 42 others. 2024. [SEACrowd: A multilingual multimodal data hub and benchmark suite for Southeast Asian languages](https://doi.org/10.18653/v1/2024.emnlp-main.296 ""). In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 5155–5203, Miami, Florida, USA. Association for Computational Linguistics.
*   Mayhew et al. (2024) Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Suppa, Hila Gonen, Joseph Marvin Imperial, Börje Karlsson, Peiqin Lin, Nikola Ljubešić, Lester James Miranda, Barbara Plank, Arij Riabi, and Yuval Pinter. 2024. Universal NER: A gold-standard multilingual named entity recognition benchmark. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 4322–4337.
*   McFarland (2008) Curtis D McFarland. 2008. Linguistic diversity and english in the philippines. *Philippine English: Linguistic and literary perspectives*, 1:131.
*   Meta AI (2025) Meta AI. 2025. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. [https://ai.meta.com/blog/llama-4-multimodal-intelligence/](https://ai.meta.com/blog/llama-4-multimodal-intelligence/ ""). Blog post, accessed May 16, 2025.
*   Metila et al. (2016) Romylyn A Metila, Lea Angela S Pradilla, and Alan B Williams. 2016. The challenge of implementing mother tongue education in linguistically diverse contexts: The case of the philippines. *The Asia-Pacific Education Researcher*, 25:781–789.
*   Miranda (2023) Lester James Miranda. 2023. [Developing a named entity recognition dataset for Tagalog](https://doi.org/10.18653/v1/2023.sealp-1.2 ""). In *Proceedings of the First Workshop in South East Asian Language Processing*, pages 13–20, Nusa Dua, Bali, Indonesia. Association for Computational Linguistics.
*   Mistral AI (2024) Mistral AI. 2024. Mixtral of experts. [https://mistral.ai/news/ministraux](https://mistral.ai/news/ministraux ""). Blog post, accessed May 16, 2025.
*   Mohamad Alhajar (2024) Alexandre Lavallée Mohamad Alhajar. 2024. Open llm french leaderboard v0.2. [https://huggingface.co/spaces/le-leadboard/OpenLLMFrenchLeaderboard](https://huggingface.co/spaces/le-leadboard/OpenLLMFrenchLeaderboard "").
*   Montalan et al. (2025) Jann Railey Montalan, Jimson Paulo Layacan, David Demitri Africa, Richell Isaiah Flores, Michael T Lopez II, Theresa Denise Magsajo, Anjanette Cayabyab, and William Chandra Tjhi. 2025. Batayan: A filipino nlp benchmark for evaluating large language models. *arXiv preprint arXiv:2502.14911*.
*   Montalan et al. (2024) Jann Railey Montalan, Jian Gang Ngui, Wei Qi Leong, Yosephine Susanto, Hamsawardhini Rengarajan, Alham Fikri Aji, and William Chandra Tjhi. 2024. Kalahi: A handcrafted, grassroots cultural LLM evalutation suite for filipino. *arXiv preprint arXiv:2409.15380*.
*   Ng et al. (2025) Raymond Ng, Thanh Ngan Nguyen, Yuli Huang, Ngee Chia Tai, Wai Yi Leong, Wei Qi Leong, Xianbin Yong, Jian Gang Ngui, Yosephine Susanto, Nicholas Cheng, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Adithya Venkatadri Hulagadri, Kok Wai Teng, Yeo Yeow Tong, Bryan Siow, Wei Yi Teo, Wayne Lau, Choon Meng Tan, and 12 others. 2025. [Sea-lion: Southeast asian languages in one network](https://arxiv.org/abs/2504.05747 ""). *Preprint*, arXiv:2504.05747.
*   Oco and Roxas (2018) Nathaniel Oco and Rachel Roxas. 2018. [A survey of machine translation work in the Philippines: From 1998 to 2018](https://aclanthology.org/W18-2204/ ""). In *Proceedings of the AMTA 2018 Workshop on Technologies for MT of Low Resource Languages (LoResMT 2018)*, pages 30–36, Boston, MA. Association for Machine Translation in the Americas.
*   Ojo et al. (2023) Jessica Ojo, Kelechi Ogueji, Pontus Stenetorp, and David Ifeoluwa Adelani. 2023. How good are large language models on african languages? *arXiv preprint arXiv:2311.07978*.
*   Papineni et al. (2002) Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [Bleu: a method for automatic evaluation of machine translation](https://doi.org/10.3115/1073083.1073135 ""). In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
*   Philippine Statistics Authority (2020) Philippine Statistics Authority. 2020. [Household population, number of households and average household size of the philippines (2020 census of population and housing)](https://psa.gov.ph/content/household-population-number-households-and-average-household-size-philippines-2020-census ""). Accessed: 2025-04-03.
*   Philippy et al. (2023) Fred Philippy, Siwen Guo, and Shohreh Haddadan. 2023. [Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review](https://doi.org/10.18653/v1/2023.acl-long.323 ""). In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5877–5891.
*   Pilar et al. (2023) Ma. Beatrice Emanuela Pilar, Dane Dedoroy, Ellyza Mari Papas, Mary Loise Buenaventura, Myron Darrel Montefalcon, Jay Rhald Padilla, Joseph Marvin Imperial, Mideth Abisado, and Lany Maceda. 2023. CebuaNER: A new baseline Cebuano named entity recognition model. In *Proceedings of the 37th Pacific Asia Conference on Language, Information and Computation*, pages 792–800.
*   Qiu et al. (2025) Haoyi Qiu, Alexander R. Fabbri, Divyanish Agarwal, Kung-Hsiang Huang, Sarah Tan, Nanyun Peng, and Chien-Sheng Wu. 2025. Evaluating cultural and social awareness of llm web agents. In *Findings of the Association for Computational Linguistics: NAACL 2025*.
*   Romanou et al. (2024) Angelika Romanou, Negar Foroutan, Anna Sotnikova, Zeming Chen, Sree Harsha Nelaturu, Shivalika Singh, Rishabh Maheshwary, Micol Altomare, Mohamed A Haggag, Alfonso Amayuelas, and 1 others. 2024. INCLUDE: Evaluating multilingual language understanding with regional knowledge. *arXiv preprint arXiv:2411.19799*.
*   Roxas et al. (2021) Rachel Edita O. Roxas, Joseph Marvin Imperial, and Angelica H. De La Cruz. 2021. [Science mapping of publications in natural language processing in the Philippines: 2006 to 2020](https://aclanthology.org/2021.paclic-1.76/ ""). In *Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation*, pages 721–730, Shanghai, China. Association for Computational Lingustics.
*   Singh et al. (2024) Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David I Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, and 1 others. 2024. Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. *arXiv preprint arXiv:2412.03304*.
*   Susanto et al. (2025) Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xian Bin Yong, Weiqi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, and William Chandra Tjhi. 2025. SEA-HELM: Southeast asian holistic evaluation of language models. *arXiv preprint arXiv:2502.14301*.
*   Team et al. (2025) Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*.
*   Team et al. (2024) Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. *arXiv preprint arXiv:2408.00118*.
*   Tiedemann (2020) Jörg Tiedemann. 2020. The tatoeba translation challenge – realistic data sets for low resoure and multilingual MT. In *Proceedings of the Fifth Conference on Machine Translation*, pages 1174–1182.
*   Villafania (2007) Sonny Villafania. 2007. [Filipino and Tagalog, not so different](https://web.archive.org/web/20140522052247/http://svillafania.philippinepen.ph/2007/08/articles-filipino-and-tagalog-not-so.html ""). Archived from the original on 2014-05-22.
*   Wan et al. (2022) Yu Wan, Baosong Yang, Derek Fai Wong, Lidia Sam Chao, Liang Yao, Haibo Zhang, and Boxing Chen. 2022. [Challenges of neural machine translation for short texts](https://doi.org/10.1162/coli_a_00435 ""). *Computational Linguistics*, 48(2):321–342.
*   Wang et al. (2023) Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang. 2023. Gpt-ner: Named entity recognition via large language models. *arXiv preprint arXiv:2304.10428*.
*   Yang et al. (2024) An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.5 Technical Report. *arXiv preprint arXiv:2412.15115*.
*   Yue et al. (2024) Xiang Yue, Yueqi Song, Akari Asai, Seungone Kim, Jean de Dieu Nyandwi, Simran Khanuja, Anjali Kantharuban, Lintang Sutawika, Sathyanarayanan Ramamoorthy, and Graham Neubig. 2024. Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages. In *The Thirteenth International Conference on Learning Representations*.
*   Zhang et al. (2023a) Biao Zhang, Barry Haddow, and Alexandra Birch. 2023a. Prompting Large Language Model for Machine Translation: A Case Study. *arXiv preprint arXiv:2301.07069*.
*   Zhang et al. (2024) Wenxuan Zhang, Hou Pong Chan, Yiran Zhao, Mahani Aljunied, Jianyu Wang, Chaoqun Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu, Yew Ken Chia, and 1 others. 2024. SeaLLMs 3: Open Foundation and Chat Multilingual Large Language Models for Southeast Asian Languages. *arXiv preprint arXiv:2407.19672*.
*   Zhang et al. (2023b) Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing. 2023b. Sentiment analysis in the era of large language models: A reality check. *arXiv preprint arXiv:2305.15005*.
*   Zhao et al. (2025) Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, and 1 others. 2025. Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers. *arXiv preprint arXiv:2503.00865*.
*   Zhu et al. (2023) Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Multilingual machine translation with large language models: Empirical results and analysis. *arXiv preprint arXiv:2304.04675*.

## Appendix

## Appendix A Details of Models Evaluated on FilBench

[Table 6](#A1.T6 "Table 6 ‣ Appendix A Details of Models Evaluated on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?")shows the details of all models evaluated on FilBench.

Model

\# Params (B)

\# Lang.

License

Reference

[gpt-4o-2024-08-06](https://platform.openai.com/docs/models "")

−-

−-

OpenAI ToS

[Hurst et al. (2024)](#bib.bib41 "")

[gpt-4o-mini](https://platform.openai.com/docs/models "")

−-

−-

OpenAI ToS

[Hurst et al. (2024)](#bib.bib41 "")

[CohereForAI/aya-expanse-32b](https://huggingface.co/CohereForAI/aya-expanse-32b "")

32

23

CC BY NC 4.0

[Dang et al. (2024)](#bib.bib26 "")

[meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 "")

400 (17)

200

Llama 4 License

[Meta AI (2025)](#bib.bib54 "")

[meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct "")

109 (17)

200

Llama 4 License

[Meta AI (2025)](#bib.bib54 "")

[meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct "")

70

30

Llama 3.1 License

[Grattafiori et al. (2024)](#bib.bib34 "")

[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct "")

8

30

Llama 3.1 License

[Grattafiori et al. (2024)](#bib.bib34 "")

[Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct "")

72

29

Qwen License

[Yang et al. (2024)](#bib.bib79 "")

[Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct "")

32

29

Apache 2.0

[Yang et al. (2024)](#bib.bib79 "")

[Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct "")

14

29

Apache 2.0

[Yang et al. (2024)](#bib.bib79 "")

[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct "")

7

29

Apache 2.0

[Yang et al. (2024)](#bib.bib79 "")

[Tower-Babel/Babel-83B-Chat](https://huggingface.co/Tower-Babel/Babel-83B-Chat "")

83

25

SeaLLM License

[Zhao et al. (2025)](#bib.bib84 "")

[Tower-Babel/Babel-9B-Chat](https://huggingface.co/Tower-Babel/Babel-9B-Chat "")

9

25

SeaLLM License

[Zhao et al. (2025)](#bib.bib84 "")

[google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it "")

27

73

Gemma License

[Team et al. (2025)](#bib.bib73 "")

[google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it "")

27

73

Gemma License

[Team et al. (2024)](#bib.bib74 "")

[google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it "")

9

73

Gemma License

[Team et al. (2024)](#bib.bib74 "")

[mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410 "")

8

10

Mistral AI License

[Mistral AI (2024)](#bib.bib57 "")

[mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1 "")

141 (39)

5

Apache 2.0

[Jiang et al. (2024)](#bib.bib44 "")

[mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 "")

47 (13)

5

Apache 2.0

[Jiang et al. (2024)](#bib.bib44 "")

[neulab/Pangea-7B](https://huggingface.co/neulab/Pangea-7B "")

7

39

Apache 2.0

[Yue et al. (2024)](#bib.bib80 "")

[aisingapore/Llama-SEA-LION-v3-70B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-70B-IT "")

70

13

Llama 3.1 License

[Ng et al. (2025)](#bib.bib61 "")

[aisingapore/Gemma-SEA-LION-v3-9B-IT](https://huggingface.co/aisingapore/Gemma-SEA-LION-v3-9B-IT "")

9

13

Gemma License

[Ng et al. (2025)](#bib.bib61 "")

[aisingapore/Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-8B-IT "")

8

13

Llama 3.1 License

[Ng et al. (2025)](#bib.bib61 "")

[sail/Sailor2-20B-Chat](https://huggingface.co/sail/Sailor2-20B-Chat "")

20

12

Apache 2.0

[Dou et al. (2025)](#bib.bib27 "")

[sail/Sailor2-8B-Chat](https://huggingface.co/sail/Sailor2-8B-Chat "")

8

12

Apache 2.0

[Dou et al. (2025)](#bib.bib27 "")

[SeaLLMs/SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat "")

7

12

SeaLLM License

[Zhang et al. (2024)](#bib.bib82 "")

[SeaLLMs/SeaLLMs-v3-1.5B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B-Chat "")

1.5

12

SeaLLM License

[Zhang et al. (2024)](#bib.bib82 "")

Table 6: All models evaluated on FilBench. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (1.5B to 400B), and accessibility (open-source vs. commercial). For Mixture-of-Experts models, parameters are denoted as "Total Parameters (Active Parameters)". Models that are finetuned on top of a pre-trained model have the number of languages supported based on their fine-tuning data. 

## Appendix B FilBench Dataset Licenses

[Table 7](#A2.T7 "Table 7 ‣ Appendix B FilBench Dataset Licenses ‣ FilBench: Can LLMs Understand and Generate Filipino?") provides information for all datasets in FilBench, such as their license and data collection process.

Category

Dataset

Source

Annotation

License

CN

Dengue Filipino [Livelo and Cheng (2018)](#bib.bib50 "")

Social media (Twitter)

Expert-annotated

Unknown

BalitaNLP [Buñag and Esquivel (2023)](#bib.bib14 "")

News articles

Included from source

Unknown

SIB-200 [Adelani et al. (2024)](#bib.bib1 "")

Human-translation

Expert-annotated

CC BY SA 4.0

CebuaNER [Pilar et al. (2023)](#bib.bib67 "")

News articles

Expert-annotated

CC BY NC SA 4.0

TLUnified-NER [Miranda (2023)](#bib.bib56 "")

News articles

Expert-annotated

GPL v3.0

Universal NER [Mayhew et al. (2024)](#bib.bib52 "")

Universal Dependencies

Expert-annotated

CC BY SA 4.0

FiReCS [Cosme and De Leon (2023)](#bib.bib19 "")

Reviews (Maps and Shopee)

Expert-annotated

CC BY 4.0

CK

INCLUDE [Romanou et al. (2024)](#bib.bib69 "")

Local exams

Expert-annotated

Apache 2.0

Global MMLU [Singh et al. (2024)](#bib.bib71 "")

MMLU dataset

Translated with validation

Apache 2.0

KALAHI [Montalan et al. (2024)](#bib.bib60 "")

Human-provided

Expert-annotated

CC BY 4.0

StingrayBench [Cahyawijaya et al. (2024)](#bib.bib15 "")

Human-provided

Expert-annotated

CC BY SA 4.0

RC

Cebuano Readability Corpus [Imperial et al. (2022)](#bib.bib43 "")

Book repositories

Expert-annotated

MIT

Belebele [Bandarkar et al. (2024)](#bib.bib10 "")

Wikipedia

Expert-annotated

CC BY SA 4.0

NewsPH NLI [Cruz et al. (2021)](#bib.bib23 "")

News articles

Semi-supervised

Unknown

GN

NTREX-128 [Federmann et al. (2022)](#bib.bib32 "")

Translated from WMT19

Expert-annotated

CC BY SA 4.0

Tatoeba [Tiedemann (2020)](#bib.bib75 "")

Crowd-sourced

Crowd-sourced

CC BY 2.0

TICO-19 [Anastasopoulos et al. (2020)](#bib.bib5 "")

News, Wikipedia, PubMed

Semi-supervised

CC0 1.0

Table 7: Supplemental information for all datasets included in FilBench. For datasets with “Unknown” licenses, we obtained explicit approval from the authors to include them in our evaluation suite. 

## Appendix C Full results on FilBench

[Table 8](#A3.T8 "Table 8 ‣ Appendix C Full results on FilBench ‣ FilBench: Can LLMs Understand and Generate Filipino?")shows the full aggregated results for the 27 models evaluated on FilBench.

| FilBench Score                                                                                                                   | Cultural Knowledge | Classical NLP  | Reading Comp.  |                |                |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------- | -------------- | -------------- | -------------- |
| [gpt-4o-2024-08-06](https://platform.openai.com/docs/models "")                                                                  | 72.73±\\pm1.66     | 73.29±\\pm3.01 | 89.03±\\pm2.05 | 80.12±\\pm0.90 | 46.48±\\pm0.60 |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 "") | 67.67±\\pm1.04     | 76.75±\\pm3.04 | 87.28±\\pm0.26 | 72.99±\\pm0.18 | 33.67±\\pm0.71 |
| [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct "")                 | 63.20±\\pm1.05     | 74.31±\\pm3.14 | 87.88±\\pm0.25 | 70.86±\\pm0.18 | 19.75±\\pm0.63 |
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct "")                                                 | 63.08±\\pm0.99     | 73.11±\\pm3.22 | 88.60±\\pm0.24 | 75.62±\\pm0.17 | 14.98±\\pm0.33 |
| [aisingapore/Llama-SEA-LION-v3-70B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-70B-IT "")                           | 61.07±\\pm0.95     | 76.78±\\pm3.02 | 89.99±\\pm0.23 | 53.56±\\pm0.19 | 23.95±\\pm0.34 |
| [Tower-Babel/Babel-83B-Chat](https://huggingface.co/Tower-Babel/Babel-83B-Chat "")                                               | 60.85±\\pm0.96     | 75.21±\\pm3.11 | 88.81±\\pm0.25 | 64.85±\\pm0.19 | 14.53±\\pm0.29 |
| [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct "")                                 | 59.66±\\pm1.17     | 72.16±\\pm3.21 | 90.27±\\pm0.83 | 52.17±\\pm0.28 | 24.03±\\pm0.37 |
| [sail/Sailor2-20B-Chat](https://huggingface.co/sail/Sailor2-20B-Chat "")                                                         | 58.61±\\pm1.06     | 66.43±\\pm3.41 | 89.03±\\pm0.25 | 63.03±\\pm0.19 | 15.95±\\pm0.38 |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct "")                                                 | 57.88±\\pm1.45     | 66.83±\\pm3.45 | 89.32±\\pm1.99 | 70.59±\\pm0.18 | 4.79±\\pm0.17  |
| [aisingapore/Gemma-SEA-LION-v3-9B-IT](https://huggingface.co/aisingapore/Gemma-SEA-LION-v3-9B-IT "")                             | 56.14±\\pm1.53     | 64.44±\\pm3.43 | 88.55±\\pm0.25 | 54.46±\\pm0.20 | 17.10±\\pm2.25 |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it "")                                                         | 55.22±\\pm1.04     | 68.76±\\pm3.32 | 87.99±\\pm0.25 | 48.77±\\pm0.19 | 15.38±\\pm0.38 |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it "")                                                         | 55.17±\\pm0.99     | 71.41±\\pm3.24 | 88.61±\\pm0.24 | 53.23±\\pm0.19 | 7.42 ±\\pm0.30 |
| [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1 "")                         | 54.28±\\pm1.09     | 54.47±\\pm3.62 | 87.19±\\pm0.25 | 64.78±\\pm0.19 | 10.70±\\pm0.31 |
| [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it "")                                                           | 53.33±\\pm1.08     | 63.69±\\pm3.47 | 87.47±\\pm0.25 | 50.65±\\pm0.20 | 11.51±\\pm0.40 |
| [Tower-Babel/Babel-9B-Chat](https://huggingface.co/Tower-Babel/Babel-9B-Chat "")                                                 | 52.75±\\pm1.48     | 60.06±\\pm3.57 | 87.67±\\pm1.90 | 56.49±\\pm0.20 | 6.79 ±\\pm0.26 |
| [sail/Sailor2-8B-Chat](https://huggingface.co/sail/Sailor2-8B-Chat "")                                                           | 52.49±\\pm1.10     | 58.94±\\pm3.57 | 86.03±\\pm0.27 | 50.69±\\pm0.23 | 14.29±\\pm0.36 |
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct "")                                                 | 52.41±\\pm1.63     | 59.27±\\pm3.61 | 86.27±\\pm2.56 | 59.95±\\pm0.20 | 4.14 ±\\pm0.14 |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct "")                                                   | 50.46±\\pm1.08     | 51.61±\\pm3.68 | 85.58±\\pm0.27 | 60.47±\\pm0.20 | 4.19 ±\\pm0.15 |
| [aisingapore/Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct "")                 | 50.32±\\pm1.08     | 59.89±\\pm3.56 | 83.33±\\pm0.28 | 47.47±\\pm0.10 | 10.60±\\pm0.29 |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 "")                           | 50.26±\\pm1.09     | 49.88±\\pm3.67 | 84.19±\\pm0.29 | 60.95±\\pm0.19 | 6.02 ±\\pm0.31 |
| [SeaLLMs/SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat "")                                               | 49.06±\\pm1.06     | 52.04±\\pm3.66 | 79.68±\\pm0.33 | 62.47±\\pm0.19 | 2.08 ±\\pm0.10 |
| [CohereForAI/aya-expanse-32b](https://huggingface.co/CohereForAI/aya-expanse-32b "")                                             | 47.84±\\pm1.41     | 53.22±\\pm3.65 | 87.47±\\pm1.60 | 46.09±\\pm0.21 | 4.58 ±\\pm0.16 |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct "")                                   | 47.38±\\pm1.51     | 52.08±\\pm3.68 | 86.61±\\pm1.90 | 46.42±\\pm0.24 | 4.42 ±\\pm0.20 |
| [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410 "")                           | 47.33±\\pm1.66     | 42.02±\\pm3.62 | 77.95±\\pm2.59 | 62.33±\\pm0.20 | 7.00 ±\\pm0.25 |
| [neulab/Pangea-7B](https://huggingface.co/neulab/Pangea-7B "")                                                                   | 43.98±\\pm1.08     | 46.23±\\pm3.70 | 78.80±\\pm0.29 | 47.74±\\pm0.22 | 3.15 ±\\pm0.15 |
| [SeaLLMs/SeaLLMs-v3-1.5B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B-Chat "")                                           | 43.20±\\pm1.07     | 37.14±\\pm3.61 | 75.17±\\pm0.33 | 56.85±\\pm0.20 | 2.08 ±\\pm0.14 |
| [gpt-4o-mini](https://platform.openai.com/docs/models "")                                                                        | 42.32±\\pm1.81     | 25.09±\\pm3.26 | 73.12±\\pm3.18 | 47.78±\\pm0.34 | 23.29±\\pm0.59 |

Table 8: Model performance on FilBench. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (8B to 400B), and accessibility (open-source vs. commercial). 

## Appendix D Generation Few-shot Results

[Table 9](#A4.T9 "Table 9 ‣ Appendix D Generation Few-shot Results ‣ FilBench: Can LLMs Understand and Generate Filipino?")shows the full few-shot experiment results on the Generation category of FilBench for 9 selected models.

Tatoeba - TGL

Tatoeba - CEB

NTREX-128

TICO-19

(ENG →\\to FIL)

(CEB →\\to ENG)

(ENG →\\to FIL)

(ENG →\\to FIL)

Model / kk-shot #

0

1

3

5

0

1

3

5

0

1

3

5

0

1

3

5

[gpt-4o-2024-08-06](https://platform.openai.com/docs/models "")

51.88

60.23

60.62

61.65

33.78

59.37

62.99

63.98

38.96

57.09

58.47

58.56

53.03

64.42

64.08

65.15

[gpt-4o-mini](https://platform.openai.com/docs/models "")

12.13

51.69

55.20

60.23

27.07

49.57

58.30

58.71

27.83

54.67

57.81

58.30

41.16

52.24

64.08

64.43

[Sailor/Sailor2-20B-Chat](https://huggingface.co/Sailor/Sailor2-20B-Chat "")

15.88

17.13

18.31

22.34

13.67

10.60

12.07

13.19

23.41

44.45

43.84

44.50

22.88

54.21

53.08

55.05

[aisingapore/Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-8B-IT "")

1.45

14.93

15.25

15.10

9.01

10.75

12.33

12.26

14.79

39.74

40.41

40.52

22.84

44.04

43.91

44.05

[CohereForAI/aya-expanse-32b](https://huggingface.co/CohereForAI/aya-expanse-32b "")

0.80

14.03

13.86

13.60

8.31

10.27

11.51

11.93

6.72

33.71

33.70

36.12

8.48

39.55

38.88

39.92

[SeaLLMs/SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat "")

0.65

11.17

11.70

12.07

6.44

7.62

9.47

9.61

5.65

32.46

33.88

36.50

6.01

39.41

39.13

39.83

[Qwen/Qwen-2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen-2.5-7B-Instruct "")

0.72

8.37

8.72

9.38

6.60

7.38

8.93

9.72

6.99

28.72

29.59

30.67

6.43

32.06

32.40

33.16

[neulab/Pangea-7B](https://huggingface.co/neulab/Pangea-7B "")

0.53

5.73

7.56

7.69

7.06

6.65

8.29

8.41

4.59

23.36

24.15

25.60

5.40

31.28

28.95

28.73

[SeaLLMs/SeaLLMs-v3-1.5B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B-Chat "")

0.78

4.9

6.43

6.99

4.51

6.72

6.36

6.34

2.04

22.97

26.10

28.16

2.15

24.45

32.27

32.72

Table 9: Generation scores for few-shot prompting on selected models (multilingual , SEA-specific ).

## Appendix E Evaluation Infrastructure and Runtime

We built FilBench on top of LightEval ([Habib et al., 2023](#bib.bib38 "")). When using the vLLM backend ([Kwon et al., 2023](#bib.bib45 "")), evaluating on the whole suite sequentially can take 4.93 hours on 2 NVIDIA H100 GPUs for models under 83B parameters. However, the evaluation suite can be parallelized per benchmark, with the runtime distribution shown in [Figure 6](#A5.F6 "Figure 6 ‣ Appendix E Evaluation Infrastructure and Runtime ‣ FilBench: Can LLMs Understand and Generate Filipino?"). The longest-running task can take approximately 1 hour and 28 minutes and the shortest task takes only 5.86 minutes.

Figure 6: Runtime of different benchmarks for a 32B model on FilBench (2 ×\\times H100 NVIDIA GPU).

Figure 7: Performance of different Multilingual and SEA-Specific models on FilBench and other SEA-specific / Tagalog benchmarks such as SeaExam and SeaBench ([Liu et al., 2025](#bib.bib48 "")), SEA-HELM ([Susanto et al., 2025](#bib.bib72 "")), and Batayan ([Montalan et al., 2025](#bib.bib59 "")). 

## Appendix F Extended Related Work

In this section, we focus on other benchmarking efforts related to FilBench. First, we compare the differences across these efforts (§[F.1](#A6.SS1 "F.1 Comparison to other SEA-specific / Filipino benchmarks ‣ Appendix F Extended Related Work ‣ FilBench: Can LLMs Understand and Generate Filipino?")) and show FilBench’s value in providing a more focused evaluation for Filipino. Then, we discuss whether there is a transferability in performance when evaluating from one benchmark to another (§[F.2](#A6.SS2 "F.2 Does high performance in one benchmark translates similarly to FilBench? ‣ Appendix F Extended Related Work ‣ FilBench: Can LLMs Understand and Generate Filipino?")).

Benchmark

\# Tasks

\# Instances

PH Languages

Data Collection Procedure

FilBench (Ours)

12

197.6k

FIL/TGL, CEB

Curated from expert-annotated datasets

SeaBench ([Liu et al., 2025](#bib.bib48 ""))

1

300

-

Collected from native-speakers

SeaExam ([Liu et al., 2025](#bib.bib48 ""))

3

5.5k

-

Collected from native-speakers

Batayan ([Montalan et al., 2025](#bib.bib59 ""))

8

3.8k

FIL

Curated with human annotation

Table 10: Comparison of multilingual benchmarks related to Filipino-centric tasks. Our data collection procedure allows us to scale the diversity of tasks in our suite. 

### F.1 Comparison to other SEA-specific / Filipino benchmarks

[Table 10](#A6.T10 "Table 10 ‣ Appendix F Extended Related Work ‣ FilBench: Can LLMs Understand and Generate Filipino?")shows benchmarking efforts orthogonal to FilBench. These efforts focus on a specific region, i.e., Southeast Asia or SEA, and comprises of datasets from countries other than the Philippines. In general, we find that SEA-specific benchmarks do not contain any Philippine language at all (as in the case of the SeaLLM Leaderboard, [Liu et al. 2025](#bib.bib48 "")) or is limited to a single Filipino language (Tagalog, as in the case of SEA-HELM, [Susanto et al. 2025](#bib.bib72 "")). FilBench aims to provide a more realistic evaluation of Filipino-centric tasks by having a principled approach in choosing categories that reflect the current trends and priorities of the Philippine NLP research community.

### F.2 Does high performance in one benchmark translates similarly to FilBench?

#### Set-up.

In order to understand whether high performance in one benchmark translates to similar performance in FilBench, we compute the Spearman ρ\\rho rank correlation of models that were evaluated in both benchmarks. For the SeaLLM leaderboard, we treat SeaBench and SeaExam separately. For SEA-HELM, we compute the correlation for the full evaluation suite and its Tagalog-only subset (Batayan).

#### Results.

[Figure 7](#A5.F7 "Figure 7 ‣ Appendix E Evaluation Infrastructure and Runtime ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the raw scores for FilBench with respect to another benchmark, alongside its Spearman ρ\\rho rank correlation. The results show moderate to strong positive correlations (ρ\\rho = 0.571 to 0.758) between FilBench and other SEA language benchmarks, with SeaExam demonstrating the strongest predictive relationship. This suggests that model performance on one benchmark does meaningfully transfer to performance on Filipino language tasks, though the scattered distribution of data points indicates that different benchmarks capture distinct aspects of language ability. Furthermore, our findings highlight that while some transferability exists across Southeast Asian language benchmarks, benchmark-specific optimization may still be necessary for optimal performance on FilBench.

## Appendix G Analysis of Model Dis/Agreement

In this section, we show examples of agreement and disagreement from the SEA-specific models we analyzed in[5.1](#S5.SS1 "5.1 Do models consistently agree with one another on Filipino language tasks? ‣ 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks? ‣ FilBench: Can LLMs Understand and Generate Filipino?").

### G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement

In order to understand model behavior, we qualitatively analyze per-instance agreement between select sub-tasks within the FilBench evaluation suite. This involves examining instances where models either consistently agree or disagree on their outputs. By focusing on specific sub-tasks, such as Readability and Cultural Knowledge Assessment, we aim to identify patterns and potential sources of error or divergence in model predictions. We hope that this analysis helps in understanding the nuances of model performance and the challenges posed by different task types.

Figure 8: In this example from a driving license exam, all SEA-specific models agree that the correct answer is A. However, the gold label is B. 

Figure 9: In this example from a driving license exam, all SEA-specific models disagree on their answers. 

### G.2 Results: Examples of Model Dis/Agreement

#### Regional Knowledge (Fleiss’ κ\=0.393\\kappa=0.393)

For this task, models are required to answer questions taken from a sample of a driving exam in the Philippines. [Figure 8](#A7.F8 "Figure 8 ‣ G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows an example where models agree on a specific answer, yet they are all incorrect. The question asks what a green arrow (berdeng arrow) indicates as a traffic signal. All models answered Option A (Vehicles are not allowed to enter the intersection as pointed by the arrow), yet the correct answer is Option B (Vehicles are allowed to turn left or right).

We also show an example where most SEA-specific models disagree in [Figure 9](#A7.F9 "Figure 9 ‣ G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?"). Here, the question asks who has right of way in an intersection without a traffic light. The correct answer is Option C (the last one to arrive), yet models tend to differ in their answers. We hypothesize that the use of the word magbigay (to give), might have confounded models due to its usage—leading to varied interpretations.

#### Readability (Fleiss’ κ\=0.207\\kappa=0.207)

For this task, models must determine the appropriate grade level for a given passage. In the Philippine educational system, there are three grade levels (Grades 1 to 3) for ages 6-7, 7-8, and 8-9, respectively ([Imperial and Kochmar, 2023](#bib.bib42 ""); [Imperial et al., 2022](#bib.bib43 "")). In [Figure 10](#A7.F10 "Figure 10 ‣ Readability (Fleiss’ 𝜅=0.207) ‣ G.2 Results: Examples of Model Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?"), all models agree that the given passage is appropriate for Grade 1 students, yet this differs from the expert-annotated gold label (Grade 2). The passage’s complexity, including the density of entities like “Mama (mother),” “eskwela (school),” and “kalsada (road / street),” likely influenced the experts to label it as Grade 2, despite its brevity and simple sentence structure, which models associated with Grade 1. On the other hand, [Figure 11](#A7.F11 "Figure 11 ‣ Readability (Fleiss’ 𝜅=0.207) ‣ G.2 Results: Examples of Model Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows an example where SEA-specific models disagree with one another. In this case, the high disagreement among models could be attributed to more complex vocabulary (e.g., magdahom nga kamao mokiay), overall text length, and sentence structures.

Figure 10: In this example, all SEA-specific models agree that the readability of the passage above is apt for Grade 1 pupils. However, the gold label indicates that the passage is for Grade 2. 

Figure 11: In this example, all SEA-specific models disagree on the readability level of the given text. 

### G.3 Discussion: Implications and Potential Future Work

The consistent disagreement of models, as seen in [Figure 10](#A7.F10 "Figure 10 ‣ Readability (Fleiss’ 𝜅=0.207) ‣ G.2 Results: Examples of Model Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?") to [Figure 9](#A7.F9 "Figure 9 ‣ G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement ‣ Appendix G Analysis of Model Dis/Agreement ‣ FilBench: Can LLMs Understand and Generate Filipino?"), highlights a potential gap in the models’ understanding of culturally-specific knowledge. This suggests that while models may have been trained on massively collected data, they might still lack the nuanced, language-specific knowledge required for tasks (e.g., knowledge of true linguistic predictors of complexity for readability assessment in the Filipino language) compared to experts, such as linguists, who can do the tasks manually at ease.

Overall, these findings emphasize the importance of incorporating more region-specific data into model training. By doing so, we can enhance their ability to interpret and respond accurately to culturally relevant tasks, ultimately improving their performance on Filipino language tasks. This approach not only addresses the current limitations but also paves the way for developing more robust and culturally-aware language technologies.

## Appendix H Generation Failure Modes Examples

[Table 11](#A8.T11 "Table 11 ‣ Appendix H Generation Failure Modes Examples ‣ FilBench: Can LLMs Understand and Generate Filipino?")provide examples of common failure modes in LLMs evaluated on the Generation category.

 Failure mode 

 Prompt 

 Gold 

 Prediction 

 Poor instruction-following 

 EN: I’ve never been to such a nice party.  
FIL: 

 Hindi ko pa naranasan ang ganitong kagandahang party. 

 I’m glad to hear that you’re enjoying yourself! It sounds like you’re having a wonderful time. 

 EN: Fever subsequently  
developed in three asymptomatic patients.  
FIL: 

 Ang lagnat ay nabuo dulot nito sa tatlong  
asymptomatic na pasyente. 

 Fever was present in three patients who were  
asymptomatic. 

 EN: Clean water is scarce.  
FIL: 

 Kulang na kulang ang  
malinis na tubig. 

 L’eau propre est rare. 

 EN: Do you think we’ll have good weather tomorrow?  
FIL: 

 Sa tingin mo may  
magandang panahon tayo bukas? 

 I don’t know. It’s hard to say. EN: What about the day after tomorrow? 

 Overgeneration 

 CEB: Napuyo ko ron sa Boston.  
EN: 

 I’m living in Boston now. 

 I’m going to Boston. CEB: Napuyo ko ron sa Boston. EN: I’m going to Boston. (continued…) 

 EN: Yumi studied English last night.  
FIL: 

 Nag-aral ng Ingles si Yumi kagabi. 

 Nag-aral si Yumi ng Ingles kagabi. This is the Filipino translation of the English  
sentence. It means "Yumi  
studied English last night." 

 Hallucination 

 CEB: Gigutom ko pag-ayo.  
EN: 

 I am very hungry. 

 Napo - Hamon sa politika laban sa pagkain Nagmukhang  
istilong moral na singaw ang saligan ng discurso 

 EN: Within minutes a wall of water crashes onto the shore, carrying away buildings and cars.  
FIL: 

 Sa ilang minuto, isang pader ng tubig ang tumama sa dalampasigan, tangay ang mga gusali at kotse. 

 More than 300 people missing; 25 top levels blown away 

Table 11: Addtional examples per common failure mode. 

## Appendix I Analysis of Generation Failure Modes for Translation Tasks

 Prompt 

 Gold 

Predictions

 Remarks (1-shot) 

0-shot

1-shot

 EN: What is your name? 

 Ano ang pangalan mo? 

 My name is Fil. 

 Ano ang pangalan mo? 

 Correct translation. 

 FIL: 

 EN: I regret not having taken his advice. 

 Ako ay nagsisi na hinde ko sinunod ang kanyang payo. 

 I regret not having taken his advice. 

 Naisip ko na lang na hindi ko na kailangan ng kanyang payo. 

 Translation is semantically incorrect. 

 FIL: 

 EN: He’s behind the times in his methods 

 Luma na ang mga paraan niya. 

 Il est en retard sur son temps dans ses méthodes. 

 Siya ay nakalayo sa panahon sa kanyang pamamaraan. 

 Translation is semantically correct but doesn’t sound natural. 

 FIL: 

Table 12: Sample zero- and one-shot generations by SeaLLMs 7B on the Filipino split of Tatoeba. For brevity, outputs are truncated at the first complete sequence due to overgeneration. 

### I.1 Set-up: Case Study of Tatoeba

We further examine generations for the Tatoeba dataset, given how most models perform poorly on it even after providing few-shot examples. We look at results per language pair (ENG →\\to FIL and CEB →\\to ENG) and discuss possible challenges models face in generating accurate translations.

### I.2 Results: Common Reasons why Models Fail in Generation Tasks

#### Finding # 1: Overgeneration on shorter texts.

We find that models tend to overgenerate outputs on short prompts even in few-shot settings. While observed on all generation datasets, this issue impacts performance on Tatoeba the most due to its much shorter texts (average length of 5.90 tokens) compared to NTREX-128 (21.04) and TICO-19 (21.67). Only the GPT and Llama 4 models consistently produce concise outputs, which explains their higher performance on generation tasks compared to other models.

#### Finding # 2: Few-shot prompting improves instruction-following but not generation quality.

Zero-shot generations for ENG →\\to FIL often appear in the wrong target language. Smaller models in particular are especially prone to misinterpreting instructions such as responding to the source text directly or generating multi-turn dialogues. As shown for the case of SeaLLMs 7B in [Table 12](#A9.T12 "Table 12 ‣ Appendix I Analysis of Generation Failure Modes for Translation Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?"), providing one-shot examples helps models begin translating correctly into Tagalog; however, the accuracy and fluency of outputs considerably vary.

In contrast, zero-shot generations for CEB →\\to ENG are more consistently in the correct language. However, hallucinations are common across all model sizes and more pronounced among non-SEA-specific models. Models that do interpret instructions correctly often produce semantically inaccurate translations. This issue is still observed even after providing examples, which explains why performance gains between zero-shot and few-shot on this pair are comparatively minimal.

#### Finding #3: Confusion between Filipino and Cebuano.

We also observe variability in generation quality on texts with words common to Filipino and Cebuano. Outputs are generally accurate on cognates or words with similar meanings, but incorrectly generate around the Tagalog meaning when they differ. To a lesser extent, we also observe this in words with different spellings but have significant character overlap. [Table 13](#A9.T13 "Table 13 ‣ Finding #3: Confusion between Filipino and Cebuano. ‣ I.2 Results: Common Reasons why Models Fail in Generation Tasks ‣ Appendix I Analysis of Generation Failure Modes for Translation Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows examples of this issue across models of different language coverage and size.

 Example 

 Remarks 

 Prompt: CEB: Nagkinahanglan ko og kwarta, kay kinahanglan ko nga magbayad sa akong kuryente. 

 The words kwarta (’money’), magbayad (’to pay’) and kuryente (’electricity’) have the same meaning in Tagalog. 

 Gold: I need money because I need to pay for my electricity. 

 Babel 83B: I need money, because I have to pay my electricity bill. 

 Qwen 2.5 7B: I need money because I need to pay my electricity bill. 

 SEA-LION v3 70B: I need some money because I had to pay for my electric bill. 

 SeaLLMs 7B: I need money, because I need to pay my electricity bill. 

 Prompt: CEB: Kung naay aso, naay kayo. EN: 

 The word aso means ’smoke’ in Cebuano, but ’dog’ in Tagalog. 

 Gold: Where there is smoke, there is fire. 

 Babel 83B: Kung aso may roon, may kami rin. 

 Qwen 2.5 7B: If there is a dog, there is you. 

 SEA-LION v3 70B: If there’s a dog, there’s a chain/leash. 

 SeaLLMs 7B: If you have a dog, you have a problem. 

 Prompt: Gisuginlan ni Tom si Mary bahin sa kawat. EN: 

 The word kawat (’robbery’) is likely being confused by SEA-LION v3 70B for kawad (’wire’) in Tagalog. 

 Gold: Tom talked to Mary about the robbery. 

 Babel 83B: Ceb: Gisugod ni Tom si Mary bahin sa kawat. 

 Qwen 2.5 7B: Tom and Mary are partners in the business. 

 SEA-LION v3 70B: Tom told Mary about the wire. 

 SeaLLMs 7B: Tom gisuginlan si Mary bahin sa kawat. 

Table 13: Sample generations by multilingual (Babel 83B and Qwen 2.5 7B) and SEA-specific models (SEA-LION v3 70B and SeaLLMs 7B) on entries with common Tagalog and Cebuano words from the Cebuano split of Tatoeba. Outputs are truncated due to overgeneration. 

### I.3 Discussion

Our findings show frequent overgeneration and poor instruction-following in shorter prompts, consistent with the findings of [Wan et al. (2022)](#bib.bib77 "") on neural machine translation. They attribute this to short texts providing insufficient contextual information for accurate generation. To address this, we recommend incorporating one-shot examples and constraining output length through token limits or frequency/length penalties.

We also find evidence of language misidentification biased towards Filipino in entries with shared vocabulary. We hypothesize that the linguistic similarity between Filipino and Cebuano facilitates cross-lingual transfer within models [Eronen et al. (2023)](#bib.bib30 ""); [Philippy et al. (2023)](#bib.bib66 ""), but also makes it harder for them to distinguish between the two, especially with Cebuano’s limited representation in pre-training data [Cahyawijaya et al. (2024)](#bib.bib15 ""). Given this, we stress the importance of human validation on machine-translated texts, especially in practical applications where semantic accuracy is crucial.

Figure 12: In order to determine the research priorities of the Filipino NLP research community that will inform the categories of FilBench (i.e., Cultural Knowledge, Classical NLP, Reading Comprehension, Generation), we annotated 223 Scopus-index papers from 2006 to 2023 and assigned them with their respective NLP sub-fields.

## Appendix J Research Priorities in Filipino NLP

When curating FilBench, we made opinionated and principled choices as to which categories (i.e., CK, CN, RC, and GN) to include in the suite. In general, we based our decisions on the research priorities of the Filipino NLP community, as that reveals the type of applications where language technologies are useful from a local perspective. We describe the process and findings in this section.

#### Set-up.

In order to obtain an overview of trends in NLP research in the Philippines, we follow the process as shown in [Figure 12](#A9.F12 "Figure 12 ‣ I.3 Discussion ‣ Appendix I Analysis of Generation Failure Modes for Translation Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?").

*   •

```
Initial paper scrape. We closely follow [Roxas et al. (2021)](#bib.bib70 "")’s data collection approach and scrape the Scopus database of all research papers from 2006 to 2023 that includes any mention of the terms philippines, filipino, or tagalog (see search query in [Figure 12](#A9.F12 "Figure 12 ‣ I.3 Discussion ‣ Appendix I Analysis of Generation Failure Modes for Translation Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?")). We chose Scopus in order to increase the breadth of our search: not only because it indexes papers from ⋆\\starACL/EMNLP conferences, but also due to the academic culture in Philippine universities that incentivizes researchers to publish in Scopus-indexed journals.
```
*   •

```
NLP sub-field annotation. Then, we prompt GPT-4 to assign their NLP sub-field based on the common tracks from past ACL conferences. We formulate the prompt by including the title and abstract of the paper-in-question, and provide a list of ACL tracks to choose the label from ([Figure 16](#A12.F16 "Figure 16 ‣ Results. ‣ Appendix L Effect of Prompt Template in Generation Performance ‣ FilBench: Can LLMs Understand and Generate Filipino?")).
```
*   •

```
Manual filtering and verification. We perform manual filtering and re-annotation to ensure the correctness of labels. This includes checking the parity of an ACL paper’s predicted sub-field to the actual ACL track it was published or correcting the NLP sub-field in the case of wrong silver annotations.
```
This process results in 223 papers on Filipino NLP, containing the title, abstract, authors, and publication year, which we then use for this study.

![Refer to caption](2508.03523v1/survey_historical.png)

Figure 13: Increase in topic diversity. Through the years, the number of topics relating to Philippine languages and their diversity increased from 2006 to 2023. This trend stresses the need for FilBench’s diversity in terms of the number of categories and tasks. 

Figure 14: Distribution of papers per NLP sub-field that includes any Philippine language. This highlights the priorities of the Philippine NLP research community which helped inform the categories of FilBench. 

#### Results.

[Figure 14](#A10.F14 "Figure 14 ‣ Set-up. ‣ Appendix J Research Priorities in Filipino NLP ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the frequency of papers for each NLP sub-field that is related to Philippine languages from 2006 to 2023. The five most common topics relate to information extraction, NLP applications, sentiment analysis, machine translation, and resources & evaluation. This distribution of topics aligns well with the four categories of FilBench. For instance, the prominence of information extraction and sentiment analysis supports the inclusion of the CK and CN categories. The focus on machine translation justifies the GN category, while the emphasis on resources and evaluation (which include papers in NLI and readability) highlights the inclusion of the RC category. In addition, [Figure 13](#A10.F13 "Figure 13 ‣ Set-up. ‣ Appendix J Research Priorities in Filipino NLP ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the increasing diversity of topics in Filipino NLP through the years. Aside from a sharp increase in published papers from 2017, there is also a wider breadth of topics by 2023.

#### Discussion.

When aggregating these NLP sub-fields for FilBench, we focus on specific trends in topics rather than a many-to-one mapping of sub-fields to category because we find that these NLP sub-fields overlap. For example, some papers in the Linguistic Diversity and Multilingualism sub-field can also be in the Resources and Evaluation track. However, these trends inform us of which categories to prioritize. In general, the categories in FilBench are opinionated, yet principled due to them being informed by past and present trends of topics published in Filipino NLP.

## Appendix K Cost-Efficiency of LLMs on Filipino Language Tasks

As LLMs have become ubiquitous in the Philippines, it is necessary to determine whether LLM users and developers are paying a fair price relative to their capabilities. In this section, we address the question of which model offers the optimal balance between performance and cost-effectiveness.

#### Set-up.

In order to measure the cost-efficiency of different LLMs, we compare their per-token pricing for output-tokens as published on OpenRouter22 2 [https://openrouter.ai/models](https://openrouter.ai/models "") with respect to their FilBench score. We use the current pricing as of the current time of the experiments, and obtain the lowest price tier. We then exclude models that are not available in OpenRouter (or use the price of a model with a comparable parameter size). For some models not in OpenRouter but was finetuned from a base model (e.g., Llama-3.1-SEA-LION-v3-8B-IT is a finetune of Llama-3.1-8B-Instruct), we use the per-token price of the base model. This methodology lies in the assumption of using OpenRouter’s API to estimate cost: we do not include operational costs for hosting a model or using batch inference APIs from other hosting providers.

Figure 15: Pareto frontier illustrating the trade-off between FilBench score and inference cost (log scale). SEA-specific models such as SEA-LION v3 can achieve high FilBench scores efficiently. 

#### Results.

[Figure 15](#A11.F15 "Figure 15 ‣ Set-up. ‣ Appendix K Cost-Efficiency of LLMs on Filipino Language Tasks ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the per-token output inference cost ($/1M in log scale) of each model with respect to their FilBench scores. Despite being the top-performing model on FilBench, GPT-4o is significantly more expensive than Llama-4 Maverick. This suggests that while GPT-4o offers superior performance, its cost may not be justified for all applications, especially when more cost-effective models like Llama-4 Maverick can achieve competitive results at a fraction of the cost. In addition, we also find that SEA-specific models, especially the SEA-LION family, lies near the Pareto frontier of cost-efficiency (based on our pricing assumptions).

#### Discussion.

The Philippines is one of the most active users of ChatGPT in the world ([Group, 2024](#bib.bib35 "")). As language technologies continue to dominate both consumer and enterprise-facing applications ([Liu and Wang, 2024](#bib.bib49 ""); [Cucio and Hennig, 2025](#bib.bib24 "")), it is then relevant to ask whether there is a more cost-efficient approach in taking advantage of such systems. Our findings suggest that despite GPT-4o’s performance on FilBench, there are still more cost-effective solutions such as using open-source models such as Llama-4 Maverick with a small percentage drop in performance but at a fraction of the cost. Moreover, there is promise in investing in post-training efforts to finetune existing models with Filipino-centric training data as our findings suggest that models finetuned specifically for Filipino such as SEA-LION are at the Pareto frontier of cost-efficiency.

## Appendix L Effect of Prompt Template in Generation Performance

In the current implementation of FilBench, we use the out-of-the-box translation templates from lighteval in order to provide comparable scores to other benchmarks built on top of that framework. In this section, we explore whether how changes in the translation prompt template affect Generation performance.

#### Set-up.

We follow six translation prompt templates from [Zhang et al. (2023a)](#bib.bib81 "") and evaluate GPT-4o on GN tasks from FilBench.

ROUGE-L

ID

Prompt Template

Tatoeba (CEB)

Tatoeba (TGL)

NTREX

TICO

A

\<src>: \<input> ⋄\\diamond \<tgt>:

33.78

51.88

38.96

53.03

B

\<input> ⋄\\diamond \<tgt>:

41.78

50.32

58.10

61.85

C

\<input> ⋄\\diamond Translate to \<tgt>:

42.92

52.85

56.25

60.53

D

\<input> ⋄\\diamond Translate from \<src> to \<tgt>:

35.57

55.34

57.37

61.53

E

\<src>: \<input> ⋄\\diamond Translate to \<tgt>:

39.84

29.76

25.20

31.04

F

\<src>: \<input> ⋄\\diamond Translate from \<src> to \<tgt>:

44.41

18.62

17.54

19.73

Table 14: GPT-4o performance on different prompt templates. A template may contain the name or ISO-693 code of the source (\<src>) or target (\<tgt>) language, and the input text (\<input>). A diamond symbol (⋄\\diamond) indicates a line break. Finally, we use Template A for Generation tasks in FilBench. 

#### Results.

[Table 14](#A12.T14 "Table 14 ‣ Set-up. ‣ Appendix L Effect of Prompt Template in Generation Performance ‣ FilBench: Can LLMs Understand and Generate Filipino?") shows the ROUGE-L scores of GPT-4o on different prompt templates. Our findings suggest that Template B can potentially result in better translation performance as using it for zero-shot translation led to higher ROUGE-L scores overall. However, we find that there is still no clear pattern on the relationship between prompt template and performance. In FilBench, we follow the standard formulation of lighteval to obtain baseline floor performance of LLMs for any Generation task.

Figure 16: GPT-4 Prompt used to predict a paper’s NLP sub-field based on their title and abstract. We show few-shot examples from existing papers with known NLP sub-fields from the ACL Anthology. 

## Appendix M Task Formulation

In this section, we show an example prompt for each sub-task in FilBench.

Figure 17: Example task adapted from Dengue Filipino ([Livelo and Cheng, 2018](#bib.bib50 "")) in the Classical NLP category.

Figure 18: Example task adapted from CebuaNER ([Pilar et al., 2023](#bib.bib67 "")) in the Classical NLP category.

Figure 19: Example task adapted from FiReCS ([Cosme and De Leon, 2023](#bib.bib19 "")) in the Classical NLP category.

Figure 20: Example task from INCLUDE ([Romanou et al., 2024](#bib.bib69 "")) in the Cultural Knowledge category.

Figure 21: Example task from Global MMLU ([Singh et al., 2024](#bib.bib71 "")) in the Cultural Knowledge category.

Figure 22: Example task from KALAHI ([Singh et al., 2024](#bib.bib71 "")) in the Cultural Knowledge category.

Figure 23: Example task from StingrayBench ([Cahyawijaya et al., 2024](#bib.bib15 "")) in the Cultural Knowledge category.

Figure 24: Example task from NTREX-128 [Federmann et al. (2022)](#bib.bib32 "") in the Generation category.

Figure 25: Example task from the Cebuano split of Taoteba ([Tiedemann, 2020](#bib.bib75 "")) in the Generation category.

Figure 26: Example task from TICO-19 ([Anastasopoulos et al., 2020](#bib.bib5 "")) in the Generation category.

Figure 27: Example task adapted from the NewsPH NLI ([Cruz et al., 2021](#bib.bib23 "")) in the Reading Comprehension category.

Figure 28: Example task from the Cebuano split of Belebele ([Bandarkar et al., 2024](#bib.bib10 "")) in the Reading Comprehension category.

Figure 29: Example task adapted from the Cebuano Readability Corpus ([Imperial et al., 2022](#bib.bib43 "")) in the Reading Comprehension category.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")