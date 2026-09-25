# FILBENCH: Can LLMs Understand and Generate Filipino?

## Lester James V. Miranda<sup>1\*</sup> Elyanah Aco<sup>2\*</sup> Conner Manuel<sup>3\*</sup> Jan Christian Blaise Cruz<sup>4,5†</sup> Joseph Marvin Imperial<sup>4,6,7†</sup>

<sup>1</sup>Allen Institute for AI <sup>2</sup>Nara Institute of Science and Technology <sup>3</sup>Together AI <sup>4</sup>SEACrowd <sup>5</sup>MBZUAI <sup>6</sup>University of Bath <sup>7</sup>National University, Philippines

![](images/0d32d0327d7e263e5630393e570bf53d277d1e40f7e04971ba4b278fde61ac98.jpg)

Code filbench/filbench-eval

![](images/d77a029cabf296b62e29e78ab64c99a2532d5f891cbbbc9ce6b131fd67d316c6.jpg)

Leaderboard UD-Filipino/filbench-leaderboard

## Abstract

Despite the impressive performance of LLMs on English-based tasks, little is known about their capabilities in specific languages such as Filipino. In this work, we address this gap by introducing FILBENCH, a Filipino-centric benchmark designed to evaluate LLMs across a diverse set of tasks and capabilities in Filipino, Tagalog, and Cebuano. We carefully curate the tasks in FILBENCH to reflect the priorities and trends of NLP research in the Philippines such as Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation. By evaluating 27 state-of-the-art LLMs on FILBENCH, we find that several LLMs suffer from reading comprehension and translation capabilities. Our results indicate that FILBENCH is challenging, with the best model, GPT-4o, achieving only a score of 72.23%. Moreover, we also find that models trained specifically for Southeast Asian languages tend to underperform on FIL BENCH, with the highest-performing model, SEA-LION v3 70B, achieving only a score of 61.07%. Our work demonstrates the value of curating language-specific LLM benchmarks to aid in driving progress on Filipino NLP and increasing the inclusion of Philippine languages in LLM development.

## 1 Introduction

While large language models (LLMs) have shown impressive performance on a variety of Englishbased tasks and capabilities, their effectiveness remains largely unexplored for low-resource languages such as Filipino. This knowledge gap exists for two reasons. First, most low-resource languages, especially Filipino-centric benchmarks, developed before the ChatGPT-era (—2022 Gururaja et al., 2023) are ill-posed for current LLM evaluation despite their value in understanding language system capabilities. Second, existing multilingual LLM benchmarks either exclude Filipino entirely (Liu et al., 2025; Huang et al., 2025, inter alia) or fail to provide sufficient task and/or language diversity (Susanto et al., 2025). Filipino is an important language to consider for LLM evaluation not only because of its unique linguistic properties such as its voice marking system (Bardají et al., 2024), but also due to its large speaker population with more than 28 million speakers in the Philippines and over 2 million speakers abroad (Philippine Statistics Authority, 2020).

![](images/199700d12a7c0955e39aec9a127f44148894e9ac53015f497d3b34474e3ba651.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["FILBENCH"] --> B["Cultural Knowledge"]
  A --> C["Classical NLP"]
  A --> D["Reading Comprehension"]
  A --> E["Generation"]
  
  B --> F{"Regional Knowledge\nCultural Values\nFactual Knowledge\nWord Sense"}
  C --> G{"Sentiment Analysis\nNER\nText Categorization"}
  
  D --> H{"Gen. Reading Comp.\nReadability\nNatural Language Inf."]
  
  E --> I{"Gen. Translation\nDomain-specific Translation"}
```
</details>

Figure 1: Overview of FILBENCH. In order to comprehensively assess the full capabilities of LLMs on Philippine languages, we curate an evaluation suite consisting of 4 categories and 12 subtasks across Filipino, Tagalog, and Cebuano based on the research priorities of the Philippine NLP community (§3.1).

In this work, we perform a comprehensive study of the strengths and limitations of LLM capabilities on Filipino-centric tasks. We introduce an evaluation suite called FILBENCH, consisting of 4 categories and 12 diverse sub-tasks that are formulated for LLM evaluation. The choice of tasks to include in FILBENCH is based on our study of research trends and priorities in Filipino NLP (§3.1, §J). Evaluating models on FILBENCH reveals significant gaps in LLM performance, for instance, in text generation capabilities. The contributions of this study are three-fold:

• We close the resource gap by curating Filipino test sets across four broad task categories: Cultural Knowledge Assessment (CK), Classical NLP (CN), Reading Comprehension (RC), and Generation (GN). We transform these datasets into a unified task format aligned with standard LLM evaluation practices across literature. Our evaluation suite, FILBENCH, consists of test instances across 4 categories and 12 sub-tasks (§3).  
• We bridge the evaluation gap by evaluating 27 state-of-the-art LLMs on FILBENCH (§4). We find that the best model, GPT-4o, only achieve around 72.23% aggregated performance while the best Southeast-Asian model, SEA-LION v3 70B, only obtains a score of 61.07%.  
• We provide analyses and insights to the strengths and weaknesses of LLMs when pre sented with Filipino-centric tasks and test cases (§5). Notably, we find that text generation suffers the most, with the lowest scores across models due to failure modes such as hallucination and poor instruction-following.

FILBENCH demonstrates the value of constructing language-specific benchmarks to reveal gaps in language model capabilities and benefit the wider speaker community. More importantly, we hope that this work aids in improving the state of Filipino NLP and increase the inclusion of Philippine languages in LLM development.

## 2 Background

Languages in the Philippines. The Philippines is home to approximately 117 million language speakers across more than 185 distinct languages (Eberhard et al., 2024; McFarland, 2008; Metila et al., 2016). One of its official languages is Filipino (FIL), which is a standardized form of Tagalog (TGL) and used mainly in Metro Manila.<sup>1</sup> Aside from Filipino and Tagalog, Cebuano (CEB) is the second most widely spoken language in the Philippines with over 28 million speakers. It is part of the Visayan language family and is spoken mainly in regions of Cebu, Siquijor, and Bohol among many others (Pilar et al., 2023). As part of the same subgroup of Philippine languages, Tagalog and Cebuano share similar linguistic characteristics such as shared vocabulary and comparable word formulation processes and affixation rules, among others (Bacalla, 2019; Imperial and Kochmar, 2023). In our work, we focus on the these three languages because they cover the majority of Filipino speakers, representing approximately 61% of the country’s population (Philippine Statistics Authority, 2020).

Task Formulation in LLM Evaluation. In order to standardize how each test example is presented to an LLM, it must first be formatted into a consistent prompt structure orformulation. Multiplechoice formulation (MCF) is a common standard in evaluating LLMs across a vast array of tasks (Gu et al., 2024; Fourrier et al., 2024). In MCF, a question is posed with answers presented as labeled choices, where scoring is done by comparing the LLM’s choice to the gold label. For evaluating LLMs on generative tasks such as translation, one approach is to write an instruction prompting an LLM to translate a given text from a source language to a target language. Then, the generated output by the LLM is compared against the reference translation using various machine translation metrics (Papineni et al., 2002; Lin, 2004).

## 3 The FILBENCH Evaluation Suite

Our design philosophy for FILBENCH centers on two core principles: (1) developing an impactful benchmark that aligns with the research priorities within the Philippine context (§3.1), ensuring that a model excelling in FILBENCH is likely to perform effectively across a wide range of Filipino applications and (2) maintaining data quality and richness by incorporating diverse sub-tasks (§3.2) that were annotated by experts or native speakers. Table 1 shows all the datasets and tasks included in FILBENCH. Example task formulation for each sub-task is shown in Appendix M.

<table><tr><td>Category</td><td>Sub-Task</td><td>Dataset</td><td>Languages</td><td>#Instances</td></tr><tr><td rowspan="7">Classical NLP (CN)</td><td rowspan="3">Text Classification</td><td>Dengue Filipino (Livelo and Cheng, 2018)</td><td>FIL</td><td>4,015</td></tr><tr><td>BalitaNLP (Buñag and Esquivel, 2023)</td><td>TGL</td><td>70,352</td></tr><tr><td>SIB-200 (Adelani et al., 2024)</td><td>CEB, FIL</td><td>99</td></tr><tr><td rowspan="3">Named-Entity Recognition</td><td>CebuaNER (Pilar et al., 2023)</td><td>CEB</td><td>1,310</td></tr><tr><td>TLUnified-NER (Miranda, 2023)</td><td>TGL</td><td>1,579</td></tr><tr><td>Universal NER (Mayhew et al., 2024)</td><td>CEB, TGL</td><td>105</td></tr><tr><td>Sentiment Analysis</td><td>FiReCS (Cosme and De Leon, 2023)</td><td>FIL</td><td>7,340</td></tr><tr><td rowspan="4">Cultural Knowledge Assessment (CK)</td><td>Regional Knowledge</td><td>INCLUDE (Romanou et al., 2024)</td><td>TGL</td><td>510</td></tr><tr><td>Factual Knowledge</td><td>Global MMLU (Singh et al., 2024)</td><td>TGL</td><td>14,042</td></tr><tr><td>Cultural Values</td><td>KALAHI (Montalan et al., 2024)</td><td>TGL</td><td>150</td></tr><tr><td>Word-sense Disambiguation</td><td>StingrayBench (Cahyawijaya et al., 2024)</td><td>TGL</td><td>100</td></tr><tr><td rowspan="3">Reading Comprehension (RC)</td><td>Readability</td><td>Cebuano Readability Corpus (Imperial et al., 2022)</td><td>CEB</td><td>350</td></tr><tr><td>Reading Comprehension</td><td>Belebele (Bandarkar et al., 2024)</td><td>CEB, FIL</td><td>1,800</td></tr><tr><td>NLI</td><td>NewsPH NLI (Cruz et al., 2021)</td><td>FIL</td><td>90,000</td></tr><tr><td rowspan="3">Generation (GN)</td><td>Document translation</td><td>NTREX-128 (Federmann et al., 2022)</td><td>FIL</td><td>1,997</td></tr><tr><td>Realistic translation</td><td>Tatoeba (Tiedemann, 2020)</td><td>CEB, TGL</td><td>2,876</td></tr><tr><td>Domain-specific transl.</td><td>TICO-19 (Anastasopoulos et al., 2020)</td><td>TGL</td><td>971</td></tr></table>

Table 1: Fine-grained overview of FILBENCH. Our curation effort involves expert-annotated or validated datasets across a diverse range of sub-tasks and categories basd on a quantitative analysis of the priorities of the Filipino NLP community (§J), allowing us to comprehensively evaluate LLM capabilities on Filipino-centric tasks.

## 3.1 Research Priorities in Filipino NLP

In order to determine which tasks to include in FILBENCH, we perform a survey of the research trends in NLP research on Philippine languages from 2006–2023. Our methodology involves scraping Scopus-indexed papers and <sup>⋆</sup>ACL/EMNLP publications and classifying their NLP sub-field based on common ACL tracks. We find that classical NLP tasks such as information extraction and sentiment analysis are widely studied, as well as a variety of translation tasks. Then, we devise a taxonomy consisting of four major categories that encompass more recent trends in Philippine NLP research. More details about our methodology and findings can be found in Appendix J.

## 3.2 FILBENCH Categories

Cultural Knowledge Assessment (CK). This category tests a language model’s ability to recall factual and culturally-specific information. Studies have consistently found that LLMs predominantly trained on English text are strongly biased towards Western values and perspectives, especially when prompted in English (Cao et al., 2023). Cultural misalignment between LLMs and users can lead to unintended harms such as norm violations (Qiu et al., 2025) and socio-economic exclusion (Dammu et al., 2024). For CK, we curate a variety of examples that test an LLM’s regional and factual knowledge (Romanou et al., 2024; Singh et al., 2024), understanding of Filipino-centric values (Montalan et al., 2024), and word-sense disambiguation (Cahyawijaya et al., 2024).

Classical NLP (CN). This category encompasses a variety of information extraction and linguistic tasks such as named entity recognition (NER), sentiment analysis, and text categorization that were traditionally performed using specialized trained models. These tasks have been prominent in Philippine NLP research over the past decade (Roxas et al., 2021), and LLMs have recently begun to be employed in this domain (Ashok and Lipton, 2023; Zhang et al., 2023b; Wang et al., 2023, inter alia). For CN, we include expert-annotated NER datasets such as CebuaNER (Pilar et al., 2023), TLUnified-NER (Miranda, 2023), and Universal NER (Mayhew et al., 2024). We also take the Filipino and Cebuano subsets of SIB-200 (Adelani et al., 2024), and the text-only subset of Balita NLP (Buñag and Esquivel, 2023).

Reading Comprehension (RC). This category evaluates a language model’s ability to understand and interpret Filipino text, focusing on tasks such as readability, comprehension, and natural language inference (NLI). These tasks are crucial for assessing how well a model can process and generate human-like understanding of written content. For RC, we include datasets like the Cebuano Readability Corpus (Imperial et al., 2022), Belebele (Bandarkar et al., 2024), and NewsPH NLI (Cruz et al., 2021), which provide a comprehensive evaluation of reading comprehension capabilities in the Filipino context.

Generation (GN). Although generative LLM tasks usually include summarization and conversational generation, evaluation test sets in Filipino are sparse. However, machine translation is one of the most dominant areas of NLP research in the Philippines (Oco and Roxas, 2018; Baliber et al., 2020; Aji et al., 2023, inter alia). Recently, LLMs have gained traction for its use as automatic translators, as opposed to training specialized translation models (Zhu et al., 2023; He et al., 2024; Alves et al., 2024). Hence, we dedicate a large portion of FIL-BENCH for testing an LLM’s ability to faithfully translate texts, either from English to Filipino (ENG → FIL) or from Cebuano to English (CEB → ENG). We include a diverse set of test examples, ranging from documents (Federmann et al., 2022), realistic texts collected from volunteers (Tiedemann, 2020), and domain-specific text (Anastasopoulos et al., 2020).

<table><tr><td>Model</td><td>FILBENCH Score</td><td>Cultural Knowledge</td><td>Classical NLP</td><td>Reading Comp.</td><td>Generation</td></tr><tr><td>○ gpt-4o-2024-08-06</td><td>72.73±1.66</td><td>73.29±3.01</td><td>89.03±2.05</td><td>80.12±0.90</td><td>46.48±0.60</td></tr><tr><td>○ meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8</td><td>67.67±1.04</td><td>76.75±3.04</td><td>87.28±0.26</td><td>72.99±0.18</td><td>33.67±0.71</td></tr><tr><td>○ meta-llama/Llama-4-Scout-17B-16E-Instruct</td><td>63.20±1.05</td><td>74.31±3.14</td><td>87.88±0.25</td><td>70.86±0.18</td><td>19.75±0.63</td></tr><tr><td>○ Qwen/Qwen2.5-72B-Instruct</td><td>63.08±0.99</td><td>73.11±3.22</td><td>88.60±0.24</td><td>75.62±0.17</td><td>14.98±0.33</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-70B-IT</td><td>61.07±0.95</td><td>76.78±3.02</td><td>89.99±0.23</td><td>53.56±0.19</td><td>23.95±0.34</td></tr><tr><td>○ Tower-Babel/Babel-83B-Chat</td><td>60.85±0.96</td><td>75.21±3.11</td><td>88.81±0.25</td><td>64.85±0.19</td><td>14.53±0.29</td></tr><tr><td>○ meta-llama/Llama-3.1-70B-Instruct</td><td>59.66±1.17</td><td>72.16±3.21</td><td>90.27±0.83</td><td>52.17±0.28</td><td>24.03±0.37</td></tr><tr><td>◆ sail/Sailor2-20B-Chat</td><td>58.61±1.06</td><td>66.43±3.41</td><td>89.03±0.25</td><td>63.03±0.19</td><td>15.95±0.38</td></tr><tr><td>○ Qwen/Qwen2.5-32B-Instruct</td><td>57.88±1.45</td><td>66.83±3.45</td><td>89.32±1.99</td><td>70.59±0.18</td><td>4.79±0.17</td></tr><tr><td>◆ aisingapore/Gemma-SEA-LION-v3-9B-IT</td><td>56.14±1.53</td><td>64.44±3.43</td><td>88.55±0.25</td><td>54.46±0.20</td><td>17.10±2.25</td></tr></table>

Table 2: Performance of state-of-the-art LLMs on Filipino-centric tasks. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (1.5B to 400B), and accessibility (open-source vs. commercial). Full results can be found in Table 8.

## 3.3 FILBENCH Scoring

The CN, CK, and RC categories follow the MCF task formulation, so we score an LLM’s performance for these categories by computing the accuracy, i.e., the number of correct answers divided by the total number of examples. For GN, we compute the ROUGE-L score between the LLM-generated text and the gold reference text. All per-category metrics range from 0 to 1. In order to create a representative, single evaluation score, we perform a weighted average based on the number of examples across results as shown in Equation 1:

$$
\text {FILBENCH Score} = 1 0 0 \times \frac {\sum_ {i \in \{\mathrm{CN} , \mathrm{CK} , \mathrm{GN} , \mathrm{RC} \}} n _ {i} \cdot S _ {i}}{\sum_ {i \in \{\mathrm{CN} , \mathrm{CK} , \mathrm{GN} , \mathrm{RC} \}} n _ {i}} \tag {1}
$$

where $n _ { i }$ is the number of examples in category i and $S _ { i }$ is the score for category i.

## 4 Results: Performance of State-of-the-Art LLMs on FILBENCH

In order to understand what kind of LLMs perform well in Filipino, we select a variety of open-source and commercial LLMs to ensure broad coverage across parameter sizes and language capabilities. We also include a number of SEA-specific models that were trained to cater to Southeast Asian languages, including Filipino. A total of 27 models are chosen for evaluation. Table 6 in the Appendix shows the full details of the evaluated models.

Table 2 shows the scores obtained by the top ten models on FILBENCH. The full results for all 27 models can be seen in Table 8 of the Appendix. The best performing model is GPT-4o (72.23%), closely followed by Llama 4 Maverick (67.67%). Moreover, the highest scoring open-source dense model is Qwen2.5 72B (63.08%), while SEA-LION v3 70B is the best SEA-specific model (61.07%).

Finding #1: Larger models dominate FIL-BENCH. Figure 2 shows the FILBENCH score to Parameter Size (B) for several dense open-source language models with known sizes. Our findings suggest that parameter size strongly correlates with FILBENCH performance, with a Spearman ρ of 0.810. However, this correlation is not perfect as we observe some smaller models to be competitive with larger counterparts as observed in Qwen 2.5 32B having similar performance to Llama 3.1 70B.

Finding #2: Language-specific finetuning improves FILBENCH performance. SEA-specific models tend to be more parameter-efficient as they perform better than non-specialized LLMs on FIL-BENCH. This trend is more apparent for smaller models within the 7B to 9B range, as shown in

![](images/43cb0a925070777d329f40957c29aaa54180568d90c64bba1f03533b78a852a3.jpg)

<details>
<summary>scatter</summary>

| Model | # Parameters (B) | FilBench Score |
| --- | --- | --- |
| Qwen 2.5 72B | ~70 | ~63 |
| SEA-LION v3 70B | ~85 | ~61 |
| Llama 3.1 70B | ~70 | ~60 |
| Sailor2 20B | ~30 | ~58 |
| SEA-LION v3 9B | ~10 | ~56 |
| Aya Expanse 32B | ~45 | ~49 |
| Sailor2 8B | ~15 | ~52 |
| SEA-LION v3 8B | ~10 | ~50 |
| SeaLLMs v3 7B | ~10 | ~48 |
| SeaLLMs v3 1.5B | ~5 | ~47 |
</details>

Figure 2: Parameter-efficiency of LLMs with respect to FILBENCH. SEA-specific models are at the Pareto frontier of parameter-efficiency. However, the best SEAspecific model still underperforms on FILBENCH with a score of 61.07%.

Figure 2. In addition, SEA-specific models such as Sailor2 20B, SEA-LION v3 9B, and SeaLLMs v3 1.5B sit near the Pareto frontier in terms of performance and size. Despite these results, the best performing SEA-specific model still underperforms on FILBENCH, as in the case of SEA-LION v3 70B with a score of 61.07%. In addition, we also find that continuous finetuning of an existing multilingual LLM on SEA-specific data improve FILBENCH performance, as observed in the SEA-LION model family, which are finetunes of Llama 3.1 and Gemma 2, in Figure 3. These findings show a promising direction for building Filipino-focused LLMs, as it provides a resource-efficient path without training entirely new models from scratch.

Finding #3: Models tend to follow a consistent trend in FILBENCH performance across categories. Figure 4 suggests that most models have a consistent trend in FILBENCH performance, i.e., they tend to score well in CK, CN, and RC categories, yet are worse on GN. This is more apparent in generative (GN) tasks, where most models tend to struggle with an average performance of 17.03%. On the other hand, models tend to perform well in CK (60.72%) and CN (85.75%) categories, indicating high-level of understanding of Filipino-centric cultural entities and values. Model performance on CK tends to be more dispersed with one of the largest standard deviation (±13.14). These findings suggest that model capabilities are not uniform across categories for Filipino, indicating significant room for improvement on model training.

![](images/ac0d0f70b1c4c91f494d7451c464962ad778c8e6416ba55319c3971ee31a26c7.jpg)

<details>
<summary>bar</summary>

| Category | Instruct Model | SEA-specific finetune |
| --- | --- | --- |
| Llama 3.1 70B | 59.7 | 61.1 |
| Gemma 2 9B | 53.3 | 56.1 |
| Llama 3.1 8B | 47.4 | 50.3 |
</details>

Figure 3: Effect of language-specific finetuning. Performance comparison between a base instruction model and its finetuned version (SEA-LION v3). Languagespecific finetuning from a multilingual base model can improve performance on FILBENCH.

## 5 Analysis: When do LLMs Perform Well or Worse on Filipino Language Tasks?

## 5.1 Do models consistently agree with one another on Filipino language tasks?

Set-up. In order to understand whether models are consistently reliable in answering test cases in FILBENCH, we compute the inter-rater reliability using Fleiss’ κ across a given set of models. The first group consists of SEA-specific models (see models marked with in Table 6) while the second group includes the top-five non-SEA models on FILBENCH (Table 2). To increase granularity, we compute the Fleiss’ κ for each sub-task.

Results. The results in Table 3 show that the SEA-specific group consistently demonstrate higher agreement on several sub-tasks than the non-SEA models. This suggests that SEA-specific finetuning can improve model reliability and consistency in outputs. However, both groups show alarming disagreement on cultural tasks, indicating fundamentally different interpretations of culturally-nuanced content. We show some examples of model disagreement for the SEA-specific group in Appendix G. This implies that while regional specialization improves reliability, deeper cultural adaptation and more sophisticated training approaches may be needed to achieve reliable performance on Filipino.

![](images/ce53bba282fe150b69bdb6a2e1e8520efd6fe88b83d2997fecadbc9ef15996a4.jpg)

<details>
<summary>boxplot</summary>

| Task | Median Score | Standard Deviation |
| --- | --- | --- |
| Cultural Knowledge | \(60.72 \pm 13.14\) | — |
| Classical NLP | \(85.75 \pm 4.59\) | — |
| Reading Comprehension | \(58.77 \pm 9.48\) | — |
| Generation | \(17.03 \pm 13.35\) | — |
</details>

Figure 4: Performance trends in FILBENCH. Model performance (aggregated) across the four categories of FILBENCH, along with the average performance for each category. LLMs tend to perform well in Classical NLP tasks, but suffer poor performance in Generation tasks.

<table><tr><td rowspan="2">Sub-Task</td><td colspan="2">Model Agreement (Fleiss&#x27; κ)</td></tr><tr><td>SEA-Specific</td><td>Top-Five</td></tr><tr><td colspan="3">Classical NLP (CN)</td></tr><tr><td>Text Classification</td><td>0.513</td><td>0.174</td></tr><tr><td>Named-Entity Recog.</td><td>0.639</td><td>0.273</td></tr><tr><td>Sentiment Analysis</td><td>0.598</td><td>0.212</td></tr><tr><td colspan="3">Cultural Knowledge (CK)</td></tr><tr><td>Regional Knowledge</td><td>0.393</td><td>0.209</td></tr><tr><td>Factual Knowledge</td><td>0.224</td><td>0.115</td></tr><tr><td>Cultural Values</td><td>0.403</td><td>0.187</td></tr><tr><td>Word-sense Disamb.</td><td>0.072</td><td>-0.041</td></tr><tr><td colspan="3">Reading Comprehension (RC)</td></tr><tr><td>Readability</td><td>0.207</td><td>-0.119</td></tr><tr><td>Reading Comp.</td><td>0.377</td><td>0.248</td></tr><tr><td>NLI</td><td>0.438</td><td>0.201</td></tr></table>

Table 3: Inter-model agreement on MCF-based tasks. Inter-model agreement, as measured by Fleiss’ κ, for each sub-task in FILBENCH. Despite good performance on FILBENCH, models tend to disagree with one another, highlighting gaps in reliability.

## 5.2 Why do models fail in Generation tasks?

Set-up. Given the low performance of most models in GN tasks, we qualitiatively analyze example outputs in order to provide a taxonomy of common failure cases in Filipino generative tasks. In addition, we also assess whether we can mitigate low

Generation performance by incorporating demonstrations or few-shot examples in the prompt. To do so, we include k = {1, 3, 5} examples on the prompt during evaluation.

Results. We find common failure modes on Generation tasks in FILBENCH and show some examples from the outputs of the best-performing SEAspecific model, SEA-LION v3 70B, in Table 4:

• Poor instruction-following. When presented with Generation tasks from FILBENCH, models tend to misinterpret instructions or generate translations in an incorrect target language.  
• Overgeneration. In the case of translation, models tend to produce overly verbose text than necessary, usually until the maximum generation length is reached. This usually results in incoherent text even if properly translated to the target language.  
• Hallucination. Models often fail in Generation tasks due to spurious artifacts in the generated text. These tend to diminish the faithfulness of the model’s output, especially in the case of translation tasks.

By manually inspecting a sample of 100 failure cases from GPT-4o, we find that overgeneration and poor instruction-following are the most dominant, with 47% and 34% respectively, while hallucination occurs 19% of the time. We hypothesize that overgeneration can be caused by training data imbalance, as suggested in (Bawden and Yvon, 2023) and (Alves et al., 2023) work in the case for BLOOM and LLaMA 7B. We defer the ablation of training data quality and its effect on translation performance to future work.

<table><tr><td>Failure mode</td><td>Example</td></tr><tr><td>Poor instruction-following</td><td>ENG: Tom winked at Mary. FIL:Gold: Si Tom ay kumindat kay Mary.Pred: Tom mutiara kepada Mary. (mutiara = to wink).Explanation: Output is not in Filipino.</td></tr><tr><td>Overgeneration</td><td>ENG: Did you hear the lecture in that class yesterday? asked the student. FIL:Gold: Narinig mo ba ang lecture sa klaseng yan kahapon?, tanong ng estudyante.Pred: Sige, narinig ko na. ANG LECTURA YESTERDAY BA? Saan po ba ito naganap? (continued...)Explanation: Output is incoherent and generated highly verbose text.</td></tr><tr><td>Hallucination</td><td>ENG: The more you read the book, the less you will understand it. FIL:Gold: Nang mas na binabasa mo ang libro, menos mong maiintindihan.Pred: Ang higit na nagbabasa ng libro, mas nagiging delikado at hindi kayang tukuyin ang kahulugan.Explanation: Output hallucinated the concept of danger (delikado).</td></tr></table>

Table 4: Common failure modes in translation tasks in the Generation category. We find three common failure modes in most models in the Generation category of FILBENCH. The predictions in this table show the outputs of SEA-LION v3 70B, the currently best-performing SEA-specific model.

In addition, we also find that few-shot prompting can mitigate drop in Generation performance (Figure 5). We find that poor instructionfollowing, which is common especially in zero-shot ENG → FIL decreases once examples are provided. Full few-shot experiment results are shown in Appendix D. Despite these results, model performance on generation tasks remain generally poor, with frequent instances of overgeneration and semantically inaccurate translations. We further explain reasons for this using the Tatoeba dataset, which models consistently underperform on, in Appendix I.

## 5.3 Human evaluation of FILBENCH

When curating test instances for FILBENCH, we ensured that the majority of sources in underwent human annotation and evaluation. However, we want to verify that strong agreement between native speakers and the gold answers persisted after the instances were converted into our task-specific formulations (Appendix M).

Set-up. In order to evaluate the agreement between native speakers and FILBENCH’s gold answers, we sample 150 instances from FILBENCH with similar sub-task distribution. Then, three authors (all native speakers of Filipino) served as annotators to label each instance. For MCF tasks, the annotators choose the letter-option of the correct answer. For GN tasks, we provide the annotators with a free-form text field to input their answers. Then, we compute the inter-annotator agreement via Fleiss’ κ across two settings: (i) among annotators (intra-group) and the (ii) majority response of human annotators to FILBENCH’s gold answer (inter-group). For GN, we compute the average ROUGE-L score for each annotator pair (intragroup) and the average of the ROUGE-L score between the gold reference translation and each of the annotator translation (inter-group).

<table><tr><td>Task Formulation</td><td>Intra-group</td><td>Inter-group</td></tr><tr><td>MCF, Fleiss&#x27; κ</td><td>0.8163</td><td>0.8756</td></tr><tr><td>Generation, Avg. ROUGE-L</td><td>0.7604</td><td>0.7806</td></tr></table>

Table 5: Inter-rater agreement of native-speakers to a subset of FILBENCH. We show that FILBENCH instances have a strong agreement with native speakers on both MFC-based (Cultural Knowledge, Classical NLP, Reading Comprehension) and Generation tasks.

Results. Table 5 shows the agreement scores among the three annotators (intra-group) and their overall agreement with the gold reference answer (inter-group). The Fleiss’ κ indicate high agreement (Landis and Koch, 1977), suggesting that the instances in FilBench are reliable and aligns with native-speakers. In addition, the ROUGE-L scores between annotators are also high, suggesting that the Generation instances can be reproducibly translated. Furthermore, the inter-group ROUGE-L score supports this claim, as evidenced by similar performance given that most of the Generation instances were originally translated by other native speakers. In general, the results suggest that the agreement between native speakers and the gold answers are preserved even after converting it into our task-specific formulations.

![](images/7e6d4df5dcc4c7a981ebb7f9003a5060e1258ff2aa85192a551a75568bd6bd2b.jpg)

<details>
<summary>line</summary>

| Number of demonstrations (k-shot) | Multilingual (gpt-4o-2024-08-06) | Multilingual (gpt-4o-mini) | SEA-specific (Sailor2 20B Chat) | SEA-specific (SEA-LION v3 8B) | SEA-specific (SeaLLMs v3 1.5B Chat) |
| --- | --- | --- | --- | --- | --- |
| 0 | ~47 | ~23 | ~20 | ~5 | ~2 |
| 1 | ~45 | ~36 | ~18 | ~9 | ~4 |
| 3 | ~51 | ~41 | ~19 | ~10 | ~4 |
| 5 | ~57 | ~47 | ~22 | ~14 | ~8 |
</details>

Figure 5: Effect of few-shot examples on Generation score. Incorporating a single (k = 1) demonstration during generation generally mitigates poor model performance.

## 6 Discussion

On what to prioritize next when collecting data for Filipino-centric post-training. Our findings, through FILBENCH, reveal critical gaps in existing LLM’s capabilities to process Filipino text, particularly in generation tasks where the best models achieved ≤ 60% performance. In addition, we also find that continuous finetuning helps improve LLM performance on FILBENCH. This suggests that post-training data collection efforts should prioritize high-quality translation pairs and generative content across diverse domains. Furthermore, gathering training data from a wide range of Philippine languages, beyond just Tagalog, can enhance the performance of LLMs as demonstrated in Buzaaba et al. (2025). We posit that this can be achieved by taking advantage of cross-lingual transfer (Artetxe et al., 2020) across typologically-similar languages.

On the importance of building languagecommunity specific evaluation suites. Our findings strongly reinforce the necessity of developing language-community specific evaluation suites rather than relying on general multilingual benchmarks. FILBENCH demonstrates that even state-of-the-art models like GPT-4o achieve only 75.56% overall performance, indicating that Filipino presents unique challenges not captured in broader evaluations. By creating focused evaluation suites like FILBENCH, the research community can more accurately identify model limitations and track progress in ways that respect the linguistic particularities of Philippine languages. Furthermore, the performance variations across Filipino, Tagalog, and Cebuano emphasize the importance of fine-grained attention to linguistic diversity even within regions. This points to the need for training approaches that recognize intra-regional linguistic boundaries rather than treating Southeast Asian languages as a homogeneous group.

## 7 Related Work

State of LLM Research for Philippine Languages. Progress in the NLP research landscape for Philippine languages such as Tagalog and Cebuano is seeing a promising growth, which can be attributed to democratization and access to LLM artifacts, particularly data and open models (Lovenia et al., 2024). The first works to release open-source artifacts include tasks such as sentiment analysis, hate speech detection, and natural language inference (NLI) (Cruz and Cheng, 2019, 2020, 2022). Further release of multilingual LLMs supporting Tagalog, allowed researchers to explore further linguistic phenomena from classical NLP tasks (Pilar et al., 2023; Mayhew et al., 2024, inter alia) to language model applications (Catapang and Visperas, 2023; Montalan et al., 2024).

Language-specific LLM Evaluation Benchmarks. Global research communities are following the trend of releasing language-specific benchmarks in order to assess and track LLM progress in their respective languages. Notable examples include AfroBench for African languages (Ojo et al., 2023), BenCzechMark for the Czech (Fajcik et al., 2024), the Open Arabic LLM Leaderboard for Arabic (El Filali et al., 2025) and Le Leaderboard for French (Mohamad Alhajar, 2024). These benchmarks usually contain curated tasks that may include translated versions of existing datasets or subsets of larger evaluation suites. FILBENCH takes inspiration from these efforts by curating a comprehensive evaluation suite for Philippine languages.

Region-specific benchmarks also exist such as SeaBench and SeaExam (Liu et al., 2025) for Southeast Asia, although they do not contain any Filipino-specific subset. The most recent effort related to FILBENCH is Batayan (Montalan et al., 2025), which is part of SEA-HELM (Susanto et al., 2025). FILBENCH takes a complementary approach by systematically curating existing benchmarks, enabling not only greater efficiency in resource utilization but also facilitating a wider diversity of task types and expanded coverage of Philippine languages beyond Filipino (in this case, Tagalog and Cebuano).

## 8 Conclusion

In this work, we present a comprehensive evaluation of LLMs on Filipino-centric tasks to investigate their strengths and limitations, which still remains underexplored. We curate a benchmark called FILBENCH across four categories and 12 sub-tasks, based on our analyses of research priorities in Philippine NLP. Through FILBENCH, we discovered weaknesses in the current open and commercial state-of-the-art LLMs, such as low reliability and poor generation capabilities. FIL-BENCH emphasizes the value of creating languagespecific LLM benchmarks, as it allows us to find promising avenues for models to improve their Filipino-centric performance. Specifically, this includes language-specific post-training and collecting relevant training datasets for text generation. We hope that FILBENCH aids in driving the progress in Filipino NLP.

## Limitations

Influence of training data on downstream FIL BENCH performance. When selecting models for evaluation on FILBENCH, we categorized based on whether these models were originally presented as multilingual or SEA-specific, rather than considering the proportion of Filipino-centric training data used for fine-tuning. The training data provenance is difficult to track, especially for closed source models. This explains why models that top other multilingual leaderboards such as Aya Expanse 32B (Dang et al., 2024) perform poorly on FILBENCH, because it was not explicitly trained on Filipino. Our experiments hinge on the assumption that cross-lingual transfer (Artetxe et al., 2020) happens during different states of language modeling, as evidenced in Chirkova and Nikoulina (2024). We leave the systematic exploration of the influence of the proportion of language-specific training data to a language-specific benchmark (i.e., Filipinocentric training data to FILBENCH performance) for future work.

Focus on Tagalog and Cebuano. While some of the datasets in FILBENCH support other Philippine languages (i.e. Ilokano for Belebele), data for these (labeled or otherwise) remain sparse. We focus our suite on the relatively better-resourced Filipino and Cebuano, with the hope of supporting more languages once more datasets become available. Future work might explore data augmentation techniques and other community-driven data collection initiatives to extend FILBENCH’s coverage to languages like Hiligaynon, Bikolano, and others.

## Ethics Statement

The development and evaluation of language technologies for Filipino, Cebuano, and other Philippine languages addresses important issues of linguistic inclusion and technological access. FIL-BENCH aims to support the development of more capable Filipino language technologies that can serve the significant population of Filipino speakers worldwide. The benchmark deliberately includes culturally-specific knowledge assessment to address known biases in LLMs toward Western values and content. This highlights the importance of evaluating models in their cultural context rather than assuming universal applicability. Datasets included in FILBENCH are from publicly accessible sources, and the authors obtained explicit approval from dataset creators when license information was unclear. Overall, we do not see any serious ethical issues with this work.

## Acknowledgments

The authors would like to thank Cohere Labs for providing credits through the Cohere Research Grant to run the Aya model series, and Together AI for additional computational credits for running several open models. We also acknowledge the Hugging Face team, particularly the OpenEvals team (Clémentine Fourrier and Nathan Habib) and Daniel van Strien, for their support in publishing the FILBENCH blog post. Finally, we thank the reviewers from the May ARR cycle for their helpful feedback and insightful comments that improved this paper.

## References

David Ifeoluwa Adelani, Hannah Liu, Xiaoyu Shen, Nikita Vassilyev, Jesujoba O. Alabi, Yanke Mao, Haonan Gao, and En-Shiun Annie Lee. 2024. SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. In Proceedings ofthe 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 226–245.  
Alham Fikri Aji, Jessica Zosa Forde, Alyssa Marie Loo, Lintang Sutawika, Skyler Wang, Genta Indra Winata, Zheng-Xin Yong, Ruochen Zhang, A. Seza Dogruöz,˘ Yin Lin Tan, and Jan Christian Blaise Cruz. 2023. Current status of NLP in south East Asia with insights from multilingualism and language diversity. In Proceedings ofthe 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Associationfor Computational Linguistics: Tutorial Abstract, pages 8–13, Nusa Dua, Bali. Association for Computational Linguistics.  
Duarte Alves, Nuno Guerreiro, João Alves, José Pombal, Ricardo Rei, José de Souza, Pierre Colombo, and Andre Martins. 2023. Steering large language models for machine translation with finetuning and in-context learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11127–11148, Singapore. Association for Computational Linguistics.  
Duarte M Alves, José Pombal, Nuno M Guerreiro, Pedro H Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, and 1 others. 2024. Tower: An open multilingual large language model for translation-related tasks. arXiv preprint arXiv:2402.17733.  
Antonios Anastasopoulos, Alessandro Cattelan, Zi-Yi Dou, Marcello Federico, Christian Federmann, Dmitriy Genzel, Franscisco Guzmán, Junjie Hu, Macduff Hughes, Philipp Koehn, Rosie Lazar, Will Lewis, Graham Neubig, Mengmeng Niu, Alp Öktem, Eric Paquin, Grace Tang, and Sylwia Tur. 2020. TICO-19: the translation initiative for COvid-19. In Proceed ings ofthe 1st Workshop on NLPfor COVID-19 (Part 2) at EMNLP 2020.  
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4623–4637, Online. Association for Computational Linguistics.  
Dhananjay Ashok and Zachary C Lipton. 2023. Promptner: Prompting for named entity recognition. arXiv preprint arXiv:2305.15444.  
Lita Bacalla. 2019. Morpo-analisis ng wikang tagalog at wikang sugbuanun’g binisaya: Pahambing na pagaaral. International Journal of Resarch Studies in Education, 8:55–65.  
Renz Iver Baliber, Charibeth Cheng, Kristine Mae Adlaon, and Virgion Mamonong. 2020. Bridging Philippine languages with multilingual neural machine translation. In Proceedings ofthe 3rd Workshop on Technologies for MT of Low Resource Languages, pages 14–22, Suzhou, China. Association for Computational Linguistics.  
Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. In Proceedings ofthe 62nd An nual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 749–775.  
Maria Bardají, Elsie Or, Angelina Aquino, and Nikolaus Himmelmann. 2024. The challenges of symmetrical voice languages for universal dependencies. In Proceedings ofthe 15th International Conference ofthe Associationfor Linguistic Typology.  
Rachel Bawden and François Yvon. 2023. Investigating the translation performance of a large multilingual language model: the case of BLOOM. In Proceedings ofthe 24th Annual Conference ofthe European Associationfor Machine Translation, pages 157–170.  
Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, and Christiane Fellbaum. 2025. Lughallama: Adapting large language models for african languages. arXiv preprint arXiv:2504.06536.  
Kenrick Lance Buñag and Rosanna Esquivel. 2023. Transformer-based conditional language models to generate filipino news article. In Proceedings ofthe International Conference on International Engineering and Operations Management.  
Samuel Cahyawijaya, Ruochen Zhang, Holy Lovenia, Jan Christian Blaise Cruz, Elisa Gilbert, Hiroki Nomoto, and Alham Fikri Aji. 2024. Thank you, stingray: Multilingual large language models can not (yet) disambiguate cross-lingual word sense. arXiv preprint arXiv:2410.21573.  
Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 53–67.  
Jasper Kyle Catapang and Moses Visperas. 2023. Emotion-based morality in Tagalog and English scenarios (EMoTES-3K): A parallel corpus for explaining (im)morality of actions. In Proceedings of the Joint 3rd International Conference on Natural Language Processing for Digital Humanities and 8th  
International Workshop on Computational Linguisticsfor Uralic Languages, pages 1–6, Tokyo, Japan. Association for Computational Linguistics.  
Nadezhda Chirkova and Vassilina Nikoulina. 2024. Zero-shot cross-lingual transfer in instruction tuning of large language models. In Proceedings ofthe 17th International Natural Language Generation Confer ence, pages 695–708, Tokyo, Japan. Association for Computational Linguistics.  
Camilla Johnine Cosme and Marlene De Leon. 2023. Sentiment analysis of code-switched filipino-english product and service reviews using transformers-based large language models. In Proceedings ofWorld Conference on Information Systems for Business Man agement, pages 123–135.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2019. Evaluating language model finetuning techniques for low-resource languages. arXiv preprint arXiv:1907.00409.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2020. Establishing baselines for text classification in low-resource languages. arXiv preprint arXiv:2005.02068.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2022. Improving large-scale language models and resources for Filipino. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6548–6555, Marseille, France. European Language Resources Association.  
Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Cheng. 2021. Exploiting news article structure for automatic corpus generation of entailment datasets. In PRICAI 2021: Trends in Artificial Intelligence, pages 86–99.  
Micholo Cucio and Tristan Hennig. 2025. Artificial Intelligence and the Philippine Labor Market: Mapping Occupational Exposure and Complementarity. Technical report, International Monetary Fund.  
Preetam Prabhu Srikar Dammu, Hayoung Jung, Anjali Singh, Monojit Choudhury, and Tanu Mitra. 2024. “they are uncultured”: Unveiling covert harms and social threats in LLM generated conversations. pages 20339–20369.  
John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, and 1 others. 2024. Aya expanse: Combining research breakthroughs for a new multilingual frontier. arXiv preprint arXiv:2412.04261.  
Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, and 1 others. 2025. Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs. arXiv preprint arXiv:2502.12982.  
David M. Eberhard, Gary F. Simons, and Charles D. Fennig, editors. 2024. Ethnologue: Languages of the World, 27 edition. SIL International, Dallas, Texas.  
Ali El Filali, Manel ALOUI, Tarique Husaain, Ahmed Alzubaidi, Basma El Amel Boussaha, Ruxandra Cojocaru, Clémentine Fourrier, Nathan Habib, and Hakim Hacid. 2025. Open arabic llm leader board 2. https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard.  
Juuso Eronen, Michal Ptaszynski, and Fumito Masui. 2023. Zero-shot cross-lingual transfer language selection using linguistic similarity. Information Processing and Management, 60(3):103250.  
Martin Fajcik, Martin Docekal, Jan Dolezal, Karel Ondrej, Karel Beneš, Jan Kapsa, Pavel Smrz, Alexander Polok, Michal Hradis, Zuzana Neverilova, and 1 others. 2024. Benczechmark: A czech-centric multitask and multimetric benchmark for large language models with duel scoring mechanism. arXiv preprint arXiv:2412.17933.  
Christian Federmann, Tom Kocmi, and Ying Xin. 2022. NTREX-128 – news test references for (mt) evaluation of 128 languages. In Proceedings ofthe First Workshop on Scaling Up Multilingual Evaluation, pages 21–24.  
Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. 2024. Open llm leaderboard v2. https://huggingface. co/spaces/open-llm-leaderboard/open\_llm\_ leaderboard.  
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The LLaMa 3 herd of models. arXiv preprint arXiv:2407.21783.  
Boston Consulting Group. 2024. Consumers know more about ai than businesses think.  
Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. 2024. OLMES: A standard for language model evaluations. arXiv preprint arXiv:2406.08446.  
Sireesh Gururaja, Amanda Bertsch, Clara Na, David Widder, and Emma Strubell. 2023. To build our future, we must know our past: Contextualizing paradigm shifts in natural language processing. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13310–13325, Singapore. Association for Computational Linguistics.  
Nathan Habib, Clémentine Fourrier, Hynek Kydlícek,ˇ Thomas Wolf, and Lewis Tunstall. 2023. Lighteval: A lightweight framework for llm evaluation.  
Zhiwei He, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Yujiu Yang, Rui Wang, Zhaopeng Tu, Shum ing Shi, and Xing Wang. 2024. Exploring humanlike translation strategy with large language models.  
Transactions of the Association for Computational Linguistics, 12:229–246.  
Xu Huang, Wenhao Zhu, Hanxu Hu, Conghui He, Lei Li, Shujian Huang, and Fei Yuan. 2025. Benchmax: A comprehensive multilingual evaluation suite for large language models. arXiv preprint arXiv:2502.07346.  
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. GPT-4o System Card. arXiv preprint arXiv:2410.21276.  
Joseph Marvin Imperial and Ekaterina Kochmar. 2023. Automatic readability assessment for closely related languages. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 5371–5386, Toronto, Canada. Association for Computational Lin guistics.  
Joseph Marvin Imperial, Lloyd Lois Antonie Reyes, Michael Antoinio Ibañez, Ranz Sapinit, and Mohammed Hussien. 2022. A baseline readability model for cebuano. In Proceedings ofthe 17th Workshop on Innovative Use ofNLPfor Building Educational Applications.  
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.  
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon zalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626.  
J Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33 1:159–74.  
Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.  
Chaoqun Liu, Wenxuan Zhang, Jiahao Ying, Mahani Aljunied, Anh Tuan Luu, and Lidong Bing. 2025. Seaexam and seabench: Benchmarking llms with local multilingual questions in southeast asia. arXiv preprint arXiv:2502.06298.  
Yan Liu and He Wang. 2024. Who on Earth Is Using Generative AI? World Bank.  
Evan Dennison Livelo and Charibeth Cheng. 2018. Intelligent dengue infoveillance using gated recurrent neural learning and cross-label frequencies. In 2018 IEEE International Conference on Agents.  
Holy Lovenia, Rahmad Mahendra, Salsabil Maulana Akbar, Lester James Validad Miranda, Jennifer Santoso, Elyanah Aco, Akhdan Fadhilah, Jonibek Mansurov, Joseph Marvin Imperial, Onno P. Kampman, Joel Ruben Antony Moniz, Muhammad Ravi Shulthan Habibi, Frederikus Hudi, Railey Montalan, Ryan Ignatius Hadiwijaya, Joanito Agili Lopo, William Nixon, Börje F. Karlsson, James Jaya, and 42 others. 2024. SEACrowd: A multilingual multimodal data hub and benchmark suite for Southeast Asian languages. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5155–5203, Miami, Florida, USA. Association for Computational Linguistics.  
Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Suppa, Hila Gonen, Joseph Marvin Imperial, Börje Karlsson, Peiqin Lin, Nikola Ljubešic, Lester James´ Miranda, Barbara Plank, Arij Riabi, and Yuval Pinter. 2024. Universal NER: A gold-standard multilingual named entity recognition benchmark. In Proceed ings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4322–4337.  
Curtis D McFarland. 2008. Linguistic diversity and english in the philippines. Philippine English: Linguistic and literary perspectives, 1:131.  
Meta AI. 2025. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. https://ai.meta.com/blog/ llama-4-multimodal-intelligence/. Blog post, accessed May 16, 2025.  
Romylyn A Metila, Lea Angela S Pradilla, and Alan B Williams. 2016. The challenge of implementing mother tongue education in linguistically diverse contexts: The case of the philippines. The Asia-Pacific Education Researcher, 25:781–789.  
Lester James Miranda. 2023. Developing a named entity recognition dataset for Tagalog. In Proceedings of the First Workshop in South East Asian Language Processing, pages 13–20, Nusa Dua, Bali, Indonesia. Association for Computational Linguistics.  
Mistral AI. 2024. Mixtral of experts. https:// mistral.ai/news/ministraux. Blog post, accessed May 16, 2025.  
Alexandre Lavallée Mohamad Alhajar. 2024. Open llm french leaderboard v0.2. https: //huggingface.co/spaces/le-leadboard/ OpenLLMFrenchLeaderboard.  
Jann Railey Montalan, Jimson Paulo Layacan, David Demitri Africa, Richell Isaiah Flores, Michael T Lopez II, Theresa Denise Magsajo, Anjanette Cayabyab, and William Chandra Tjhi. 2025. Batayan: A filipino nlp benchmark for evaluating large language models. arXiv preprint arXiv:2502.14911.  
Jann Railey Montalan, Jian Gang Ngui, Wei Qi Leong, Yosephine Susanto, Hamsawardhini Rengarajan, Alham Fikri Aji, and William Chandra Tjhi. 2024. Kalahi: A handcrafted, grassroots cultural LLM evalutation suite for filipino. arXiv preprint arXiv:2409.15380.  
Raymond Ng, Thanh Ngan Nguyen, Yuli Huang, Ngee Chia Tai, Wai Yi Leong, Wei Qi Leong, Xianbin Yong, Jian Gang Ngui, Yosephine Susanto, Nicholas Cheng, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Adithya Venkatadri Hulagadri, Kok Wai Teng, Yeo Yeow Tong, Bryan Siow, Wei Yi Teo, Wayne Lau, Choon Meng Tan, and 12 others. 2025. Sea-lion: Southeast asian languages in one network. Preprint, arXiv:2504.05747.  
Nathaniel Oco and Rachel Roxas. 2018. A survey of machine translation work in the Philippines: From 1998 to 2018. In Proceedings of the AMTA 2018 Workshop on Technologiesfor MT ofLow Resource Languages (LoResMT 2018), pages 30–36, Boston, MA. Association for Machine Translation in the Americas.  
Jessica Ojo, Kelechi Ogueji, Pontus Stenetorp, and David Ifeoluwa Adelani. 2023. How good are large language models on african languages? arXiv preprint arXiv:2311.07978.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.  
Philippine Statistics Authority. 2020. Household population, number of households and average household size of the philippines (2020 census of population and housing). Accessed: 2025-04-03.  
Fred Philippy, Siwen Guo, and Shohreh Haddadan. 2023. Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review. In Proceedings of the 61st Annual Meeting ofthe Associationfor Compu tational Linguistics (Volume 1: Long Papers), pages 5877–5891.  
Ma. Beatrice Emanuela Pilar, Dane Dedoroy, Ellyza Mari Papas, Mary Loise Buenaventura, Myron Darrel Montefalcon, Jay Rhald Padilla, Joseph Marvin Imperial, Mideth Abisado, and Lany Maceda. 2023. CebuaNER: A new baseline Cebuano named entity recognition model. In Proceedings of the 37th Pacific Asia Conference on Language, Information and Computation, pages 792–800.  
Haoyi Qiu, Alexander R. Fabbri, Divyanish Agarwal, Kung-Hsiang Huang, Sarah Tan, Nanyun Peng, and Chien-Sheng Wu. 2025. Evaluating cultural and social awareness of llm web agents. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025.  
Angelika Romanou, Negar Foroutan, Anna Sotnikova, Zeming Chen, Sree Harsha Nelaturu, Shivalika Singh, Rishabh Maheshwary, Micol Altomare, Mohamed A Haggag, Alfonso Amayuelas, and 1 others. 2024. INCLUDE: Evaluating multilingual language understanding with regional knowledge. arXiv preprint arXiv:2411.19799.  
Rachel Edita O. Roxas, Joseph Marvin Imperial, and Angelica H. De La Cruz. 2021. Science mapping of publications in natural language processing in the Philippines: 2006 to 2020. In Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation, pages 721–730, Shanghai, China. Association for Computational Lingustics.  
Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David I Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, and 1 others. 2024. Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. arXiv preprint arXiv:2412.03304.  
Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xian Bin Yong, Weiqi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, and William Chandra Tjhi. 2025. SEA-HELM: Southeast asian holistic evaluation of language models. arXiv preprint arXiv:2502.14301.  
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.  
Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati raju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.  
Jörg Tiedemann. 2020. The tatoeba translation challenge – realistic data sets for low resoure and multilingual MT. In Proceedings ofthe Fifth Conference on Machine Translation, pages 1174–1182.  
Sonny Villafania. 2007. Filipino and Tagalog, not so different. Archived from the original on 2014-05-22.  
Yu Wan, Baosong Yang, Derek Fai Wong, Lidia Sam Chao, Liang Yao, Haibo Zhang, and Boxing Chen. 2022. Challenges of neural machine translation for short texts. Computational Linguistics, 48(2):321– 342.  
Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang. 2023. Gpt-ner: Named entity recognition via large language models. arXiv preprint arXiv:2304.10428.  
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.  
Xiang Yue, Yueqi Song, Akari Asai, Seungone Kim, Jean de Dieu Nyandwi, Simran Khanuja, Anjali Kantharuban, Lintang Sutawika, Sathyanarayanan Ramamoorthy, and Graham Neubig. 2024. Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages. In The Thirteenth International Confer ence on Learning Representations.  
Biao Zhang, Barry Haddow, and Alexandra Birch. 2023a. Prompting Large Language Model for Machine Translation: A Case Study. arXiv preprint arXiv:2301.07069.  
Wenxuan Zhang, Hou Pong Chan, Yiran Zhao, Mahani Aljunied, Jianyu Wang, Chaoqun Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu, Yew Ken Chia, and 1 others. 2024. SeaLLMs 3: Open Foundation and Chat Multilingual Large Language Models for Southeast Asian Languages. arXiv preprint arXiv:2407.19672.  
Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing. 2023b. Sentiment analysis in the era of large language models: A reality check. arXiv preprint arXiv:2305.15005.  
Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, and 1 others. 2025. Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers. arXiv preprint arXiv:2503.00865.  
Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Multilingual machine translation with large language models: Empirical results and analysis. arXiv preprint arXiv:2304.04675.

## Appendix

A Details of Models Evaluated on FILBENCH 16

B FILBENCH Dataset Licenses 16

C Full results on FILBENCH 17

D Generation Few-shot Results 17

E Evaluation Infrastructure and Runtime 18

F Extended Related Work 18

F.1 Comparison to other SEA-specific / Filipino benchmarks . . . . 18  
F.2 Does high performance in one benchmark translates similarly to FILBENCH? . . . . . . 18

G Analysis of Model Dis/Agreement 18

G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement . . . . . . . . . 19  
G.2 Results: Examples of Model Dis/Agreement . . . . . 19  
G.3 Discussion: Implications and Potential Future Work . . 20

H Generation Failure Modes Examples 22

I Analysis of Generation Failure Modes for Translation Tasks 23

I.1 Set-up: Case Study of Tatoeba . . . . 23  
I.2 Results: Common Reasons why Models Fail in Generation Tasks . . . . . . . . . . . . . 23  
I.3 Discussion . . . . . . . . . 23

J Research Priorities in Filipino NLP 24  
K Cost-Efficiency of LLMs on Filipino Language Tasks 26  
L Effect of Prompt Template in Generation Performance 26  
M Task Formulation 28

## A Details of Models Evaluated on FILBENCH

Table 6 shows the details of all models evaluated on FILBENCH.

<table><tr><td>Model</td><td># Params (B)</td><td># Lang.</td><td>License</td><td>Reference</td></tr><tr><td>○ gpt-4o-2024-08-06</td><td>—</td><td>—</td><td>OpenAI ToS</td><td>Hurst et al. (2024)</td></tr><tr><td>○ gpt-4o-mini</td><td>—</td><td>—</td><td>OpenAI ToS</td><td>Hurst et al. (2024)</td></tr><tr><td>○ CohereForAI/aya-expanse-32b</td><td>32</td><td>23</td><td>CC BY NC 4.0</td><td>Dang et al. (2024)</td></tr><tr><td>○ meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8</td><td>400 (17)</td><td>200</td><td>Llama 4 License</td><td>Meta AI (2025)</td></tr><tr><td>○ meta-llama/Llama-4-Scout-17B-16E-Instruct</td><td>109 (17)</td><td>200</td><td>Llama 4 License</td><td>Meta AI (2025)</td></tr><tr><td>○ meta-llama/Llama-3.1-70B-Instruct</td><td>70</td><td>30</td><td>Llama 3.1 License</td><td>Grattafiori et al. (2024)</td></tr><tr><td>○ meta-llama/Llama-3.1-8B-Instruct</td><td>8</td><td>30</td><td>Llama 3.1 License</td><td>Grattafiori et al. (2024)</td></tr><tr><td>○ Qwen/Qwen2.5-72B-Instruct</td><td>72</td><td>29</td><td>Qwen License</td><td>Yang et al. (2024)</td></tr><tr><td>○ Qwen/Qwen2.5-32B-Instruct</td><td>32</td><td>29</td><td>Apache 2.0</td><td>Yang et al. (2024)</td></tr><tr><td>○ Qwen/Qwen2.5-14B-Instruct</td><td>14</td><td>29</td><td>Apache 2.0</td><td>Yang et al. (2024)</td></tr><tr><td>○ Qwen/Qwen2.5-7B-Instruct</td><td>7</td><td>29</td><td>Apache 2.0</td><td>Yang et al. (2024)</td></tr><tr><td>○ Tower-Babel/Babel-83B-Chat</td><td>83</td><td>25</td><td>SeaLLM License</td><td>Zhao et al. (2025)</td></tr><tr><td>○ Tower-Babel/Babel-9B-Chat</td><td>9</td><td>25</td><td>SeaLLM License</td><td>Zhao et al. (2025)</td></tr><tr><td>○ google/gemma-3-27b-it</td><td>27</td><td>73</td><td>Gemma License</td><td>Team et al. (2025)</td></tr><tr><td>○ google/gemma-2-27b-it</td><td>27</td><td>73</td><td>Gemma License</td><td>Team et al. (2024)</td></tr><tr><td>○ google/gemma-2-9b-it</td><td>9</td><td>73</td><td>Gemma License</td><td>Team et al. (2024)</td></tr><tr><td>○ mistralai/Ministral-8B-Instruct-2410</td><td>8</td><td>10</td><td>Mistral AI License</td><td>Mistral AI (2024)</td></tr><tr><td>○ mistralai/Mixtral-8x22B-Instruct-v0.1</td><td>141 (39)</td><td>5</td><td>Apache 2.0</td><td>Jiang et al. (2024)</td></tr><tr><td>○ mistralai/Mixtral-8x7B-Instruct-v0.1</td><td>47 (13)</td><td>5</td><td>Apache 2.0</td><td>Jiang et al. (2024)</td></tr><tr><td>○ neulab/Pangea-7B</td><td>7</td><td>39</td><td>Apache 2.0</td><td>Yue et al. (2024)</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-70B-IT</td><td>70</td><td>13</td><td>Llama 3.1 License</td><td>Ng et al. (2025)</td></tr><tr><td>◆ aisingapore/Gemma-SEA-LION-v3-9B-IT</td><td>9</td><td>13</td><td>Gemma License</td><td>Ng et al. (2025)</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-8B-IT</td><td>8</td><td>13</td><td>Llama 3.1 License</td><td>Ng et al. (2025)</td></tr><tr><td>◆ sail/Sailor2-20B-Chat</td><td>20</td><td>12</td><td>Apache 2.0</td><td>Dou et al. (2025)</td></tr><tr><td>◆ sail/Sailor2-8B-Chat</td><td>8</td><td>12</td><td>Apache 2.0</td><td>Dou et al. (2025)</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-7B-Chat</td><td>7</td><td>12</td><td>SeaLLM License</td><td>Zhang et al. (2024)</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-1.5B-Chat</td><td>1.5</td><td>12</td><td>SeaLLM License</td><td>Zhang et al. (2024)</td></tr></table>

Table 6: All models evaluated on FILBENCH. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (1.5B to 400B), and accessibility (open-source vs. commercial). For Mixture-of-Experts models, parameters are denoted as "Total Parameters (Active Parameters)". Models that are finetuned on top of a pre-trained model have the number of languages supported based on their fine-tuning data.

## B FILBENCH Dataset Licenses

Table 7 provides information for all datasets in FILBENCH, such as their license and data collection process.

<table><tr><td>Category</td><td>Dataset</td><td>Source</td><td>Annotation</td><td>License</td></tr><tr><td rowspan="7">CN</td><td>Dengue Filipino (Livelo and Cheng, 2018)</td><td>Social media (Twitter)</td><td>Expert-annotated</td><td>Unknown</td></tr><tr><td>BalitaNLP (Buñag and Esquivel, 2023)</td><td>News articles</td><td>Included from source</td><td>Unknown</td></tr><tr><td>SIB-200 (Adelani et al., 2024)</td><td>Human-translation</td><td>Expert-annotated</td><td>CC BY SA 4.0</td></tr><tr><td>CebuaNER (Pilar et al., 2023)</td><td>News articles</td><td>Expert-annotated</td><td>CC BY NC SA 4.0</td></tr><tr><td>TLUnified-NER (Miranda, 2023)</td><td>News articles</td><td>Expert-annotated</td><td>GPL v3.0</td></tr><tr><td>Universal NER (Mayhew et al., 2024)</td><td>Universal Dependencies</td><td>Expert-annotated</td><td>CC BY SA 4.0</td></tr><tr><td>FiReCS (Cosme and De Leon, 2023)</td><td>Reviews (Maps and Shopee)</td><td>Expert-annotated</td><td>CC BY 4.0</td></tr><tr><td rowspan="4">CK</td><td>INCLUDE (Romanou et al., 2024)</td><td>Local exams</td><td>Expert-annotated</td><td>Apache 2.0</td></tr><tr><td>Global MMLU (Singh et al., 2024)</td><td>MMLU dataset</td><td>Translated with validation</td><td>Apache 2.0</td></tr><tr><td>KALAHI (Montalan et al., 2024)</td><td>Human-provided</td><td>Expert-annotated</td><td>CC BY 4.0</td></tr><tr><td>StingrayBench (Cahyawijaya et al., 2024)</td><td>Human-provided</td><td>Expert-annotated</td><td>CC BY SA 4.0</td></tr><tr><td rowspan="3">RC</td><td>Cebuano Readability Corpus (Imperial et al., 2022)</td><td>Book repositories</td><td>Expert-annotated</td><td>MIT</td></tr><tr><td>Belebele (Bandarkar et al., 2024)</td><td>Wikipedia</td><td>Expert-annotated</td><td>CC BY SA 4.0</td></tr><tr><td>NewsPH NLI (Cruz et al., 2021)</td><td>News articles</td><td>Semi-supervised</td><td>Unknown</td></tr><tr><td rowspan="3">GN</td><td>NTREX-128 (Federmann et al., 2022)</td><td>Translated from WMT19</td><td>Expert-annotated</td><td>CC BY SA 4.0</td></tr><tr><td>Tatoeba (Tiedemann, 2020)</td><td>Crowd-sourced</td><td>Crowd-sourced</td><td>CC BY 2.0</td></tr><tr><td>TICO-19 (Anastasopoulos et al., 2020)</td><td>News, Wikipedia, PubMed</td><td>Semi-supervised</td><td>CC0 1.0</td></tr></table>

Table 7: Supplemental information for all datasets included in FILBENCH. For datasets with “Unknown” licenses, we obtained explicit approval from the authors to include them in our evaluation suite.

## C Full results on FILBENCH

Table 8 shows the full aggregated results for the 27 models evaluated on FILBENCH.

<table><tr><td>Model</td><td>FILBENCH Score</td><td>Cultural Knowledge</td><td>Classical NLP</td><td>Reading Comp.</td><td>Generation</td></tr><tr><td>○ gpt-4o-2024-08-06</td><td>72.73±1.66</td><td>73.29±3.01</td><td>89.03±2.05</td><td>80.12±0.90</td><td>46.48±0.60</td></tr><tr><td>○ meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8</td><td>67.67±1.04</td><td>76.75±3.04</td><td>87.28±0.26</td><td>72.99±0.18</td><td>33.67±0.71</td></tr><tr><td>○ meta-llama/Llama-4-Scout-17B-16E-Instruct</td><td>63.20±1.05</td><td>74.31±3.14</td><td>87.88±0.25</td><td>70.86±0.18</td><td>19.75±0.63</td></tr><tr><td>○ Qwen/Qwen2.5-72B-Instruct</td><td>63.08±0.99</td><td>73.11±3.22</td><td>88.60±0.24</td><td>75.62±0.17</td><td>14.98±0.33</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-70B-IT</td><td>61.07±0.95</td><td>76.78±3.02</td><td>89.99±0.23</td><td>53.56±0.19</td><td>23.95±0.34</td></tr><tr><td>○ Tower-Babel/Babel-83B-Chat</td><td>60.85±0.96</td><td>75.21±3.11</td><td>88.81±0.25</td><td>64.85±0.19</td><td>14.53±0.29</td></tr><tr><td>○ meta-llama/Llama-3.1-70B-Instruct</td><td>59.66±1.17</td><td>72.16±3.21</td><td>90.27±0.83</td><td>52.17±0.28</td><td>24.03±0.37</td></tr><tr><td>◆ sail/Sailor2-20B-Chat</td><td>58.61±1.06</td><td>66.43±3.41</td><td>89.03±0.25</td><td>63.03±0.19</td><td>15.95±0.38</td></tr><tr><td>○ Qwen/Qwen2.5-32B-Instruct</td><td>57.88±1.45</td><td>66.83±3.45</td><td>89.32±1.99</td><td>70.59±0.18</td><td>4.79±0.17</td></tr><tr><td>◆ aisingapore/Gemma-SEA-LION-v3-9B-IT</td><td>56.14±1.53</td><td>64.44±3.43</td><td>88.55±0.25</td><td>54.46±0.20</td><td>17.10±2.25</td></tr><tr><td>○ google/gemma-2-27b-it</td><td>55.22±1.04</td><td>68.76±3.32</td><td>87.99±0.25</td><td>48.77±0.19</td><td>15.38±0.38</td></tr><tr><td>○ google/gemma-3-27b-it</td><td>55.17±0.99</td><td>71.41±3.24</td><td>88.61±0.24</td><td>53.23±0.19</td><td>7.42 ±0.30</td></tr><tr><td>○ mistralai/Mixtral-8x22B-Instruct-v0.1</td><td>54.28±1.09</td><td>54.47±3.62</td><td>87.19±0.25</td><td>64.78±0.19</td><td>10.70±0.31</td></tr><tr><td>○ google/gemma-2-9b-it</td><td>53.33±1.08</td><td>63.69±3.47</td><td>87.47±0.25</td><td>50.65±0.20</td><td>11.51±0.40</td></tr><tr><td>○ Tower-Babel/Babel-9B-Chat</td><td>52.75±1.48</td><td>60.06±3.57</td><td>87.67±1.90</td><td>56.49±0.20</td><td>6.79 ±0.26</td></tr><tr><td>◆ sail/Sailor2-8B-Chat</td><td>52.49±1.10</td><td>58.94±3.57</td><td>86.03±0.27</td><td>50.69±0.23</td><td>14.29±0.36</td></tr><tr><td>○ Qwen/Qwen2.5-14B-Instruct</td><td>52.41±1.63</td><td>59.27±3.61</td><td>86.27±2.56</td><td>59.95±0.20</td><td>4.14 ±0.14</td></tr><tr><td>○ Qwen/Qwen2.5-7B-Instruct</td><td>50.46±1.08</td><td>51.61±3.68</td><td>85.58±0.27</td><td>60.47±0.20</td><td>4.19 ±0.15</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-8B-IT</td><td>50.32±1.08</td><td>59.89±3.56</td><td>83.33±0.28</td><td>47.47±0.10</td><td>10.60±0.29</td></tr><tr><td>○ mistralai/Mixtral-8x7B-Instruct-v0.1</td><td>50.26±1.09</td><td>49.88±3.67</td><td>84.19±0.29</td><td>60.95±0.19</td><td>6.02 ±0.31</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-7B-Chat</td><td>49.06±1.06</td><td>52.04±3.66</td><td>79.68±0.33</td><td>62.47±0.19</td><td>2.08 ±0.10</td></tr><tr><td>○ CohereForAI/aya-expanse-32b</td><td>47.84±1.41</td><td>53.22±3.65</td><td>87.47±1.60</td><td>46.09±0.21</td><td>4.58 ±0.16</td></tr><tr><td>○ meta-llama/Llama-3.1-8B-Instruct</td><td>47.38±1.51</td><td>52.08±3.68</td><td>86.61±1.90</td><td>46.42±0.24</td><td>4.42 ±0.20</td></tr><tr><td>○ mistralai/Ministral-8B-Instruct-2410</td><td>47.33±1.66</td><td>42.02±3.62</td><td>77.95±2.59</td><td>62.33±0.20</td><td>7.00 ±0.25</td></tr><tr><td>○ neulab/Pangea-7B</td><td>43.98±1.08</td><td>46.23±3.70</td><td>78.80±0.29</td><td>47.74±0.22</td><td>3.15 ±0.15</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-1.5B-Chat</td><td>43.20±1.07</td><td>37.14±3.61</td><td>75.17±0.33</td><td>56.85±0.20</td><td>2.08 ±0.14</td></tr><tr><td>○ gpt-4o-mini</td><td>42.32±1.81</td><td>25.09±3.26</td><td>73.12±3.18</td><td>47.78±0.34</td><td>23.29±0.59</td></tr></table>

Table 8: Model performance on FILBENCH. We evaluate several models with different multilingual capabilities (multilingual , SEA-specific ), sizes (8B to 400B), and accessibility (open-source vs. commercial).

## D Generation Few-shot Results

Table 9 shows the full few-shot experiment results on the Generation category of FILBENCH for 9 selected models.

<table><tr><td rowspan="2">Model / k-shot #</td><td colspan="4">Tatoeba - TGL(ENG → FIL)</td><td colspan="4">Tatoeba - CEB(CEB → ENG)</td><td colspan="4">NTREX-128(ENG → FIL)</td><td colspan="4">TICO-19(ENG → FIL)</td></tr><tr><td>0</td><td>1</td><td>3</td><td>5</td><td>0</td><td>1</td><td>3</td><td>5</td><td>0</td><td>1</td><td>3</td><td>5</td><td>0</td><td>1</td><td>3</td><td>5</td></tr><tr><td>○ gpt-4o-2024-08-06</td><td>51.88</td><td>60.23</td><td>60.62</td><td>61.65</td><td>33.78</td><td>59.37</td><td>62.99</td><td>63.98</td><td>38.96</td><td>57.09</td><td>58.47</td><td>58.56</td><td>53.03</td><td>64.42</td><td>64.08</td><td>65.15</td></tr><tr><td>○ gpt-4o-mini</td><td>12.13</td><td>51.69</td><td>55.20</td><td>60.23</td><td>27.07</td><td>49.57</td><td>58.30</td><td>58.71</td><td>27.83</td><td>54.67</td><td>57.81</td><td>58.30</td><td>41.16</td><td>52.24</td><td>64.08</td><td>64.43</td></tr><tr><td>◆ Sailor/Sailor2-20B-Chat</td><td>15.88</td><td>17.13</td><td>18.31</td><td>22.34</td><td>13.67</td><td>10.60</td><td>12.07</td><td>13.19</td><td>23.41</td><td>44.45</td><td>43.84</td><td>44.50</td><td>22.88</td><td>54.21</td><td>53.08</td><td>55.05</td></tr><tr><td>◆ aisingapore/Llama-SEA-LION-v3-8B-IT</td><td>1.45</td><td>14.93</td><td>15.25</td><td>15.10</td><td>9.01</td><td>10.75</td><td>12.33</td><td>12.26</td><td>14.79</td><td>39.74</td><td>40.41</td><td>40.52</td><td>22.84</td><td>44.04</td><td>43.91</td><td>44.05</td></tr><tr><td>○ CohereForAI/aya-expanse-32b</td><td>0.80</td><td>14.03</td><td>13.86</td><td>13.60</td><td>8.31</td><td>10.27</td><td>11.51</td><td>11.93</td><td>6.72</td><td>33.71</td><td>33.70</td><td>36.12</td><td>8.48</td><td>39.55</td><td>38.88</td><td>39.92</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-7B-Chat</td><td>0.65</td><td>11.17</td><td>11.70</td><td>12.07</td><td>6.44</td><td>7.62</td><td>9.47</td><td>9.61</td><td>5.65</td><td>32.46</td><td>33.88</td><td>36.50</td><td>6.01</td><td>39.41</td><td>39.13</td><td>39.83</td></tr><tr><td>○ Qwen/Qwen-2.5-7B-Instruct</td><td>0.72</td><td>8.37</td><td>8.72</td><td>9.38</td><td>6.60</td><td>7.38</td><td>8.93</td><td>9.72</td><td>6.99</td><td>28.72</td><td>29.59</td><td>30.67</td><td>6.43</td><td>32.06</td><td>32.40</td><td>33.16</td></tr><tr><td>○ neulab/Pangea-7B</td><td>0.53</td><td>5.73</td><td>7.56</td><td>7.69</td><td>7.06</td><td>6.65</td><td>8.29</td><td>8.41</td><td>4.59</td><td>23.36</td><td>24.15</td><td>25.60</td><td>5.40</td><td>31.28</td><td>28.95</td><td>28.73</td></tr><tr><td>◆ SeaLLMs/SeaLLMs-v3-1.5B-Chat</td><td>0.78</td><td>4.9</td><td>6.43</td><td>6.99</td><td>4.51</td><td>6.72</td><td>6.36</td><td>6.34</td><td>2.04</td><td>22.97</td><td>26.10</td><td>28.16</td><td>2.15</td><td>24.45</td><td>32.27</td><td>32.72</td></tr></table>

Table 9: Generation scores for few-shot prompting on selected models (multilingual , SEA-specific ).

## E Evaluation Infrastructure and Runtime

We built FILBENCH on top of LightEval (Habib et al., 2023). When using the vLLM backend (Kwon et al., 2023), evaluating on the whole suite sequentially can take 4.93 hours on 2 NVIDIA H100 GPUs for models under 83B parameters. However, the evaluation suite can be parallelized per benchmark, with the runtime distribution shown in Figure 6. The longest-running task can take approximately 1 hour and 28 minutes and the shortest task takes only 5.86 minutes.

![](images/35b6783eb2ac27520e245e360770a1a8226fc30a43081c7dcdb05384c777ed8d.jpg)

<details>
<summary>histogram</summary>

| Runtime (minutes) | Frequency |
| --- | --- |
| 0~25 | ~18 |
| 25~50 | ~1 |
| 50~75 | ~1 |
| 75~90 | ~1 |
</details>

Figure 6: Runtime of different benchmarks for a 32B model on FILBENCH (2 × H100 NVIDIA GPU).

## F Extended Related Work

In this section, we focus on other benchmarking efforts related to FILBENCH. First, we compare the differences across these efforts (§F.1) and show FILBENCH’s value in providing a more focused evaluation for Filipino. Then, we discuss whether there is a transferability in performance when evaluating from one benchmark to another (§F.2).

## F.1 Comparison to other SEA-specific / Filipino benchmarks

Table 10 shows benchmarking efforts orthogonal to FILBENCH. These efforts focus on a specific region, i.e., Southeast Asia or SEA, and comprises of datasets from countries other than the Philippines. In general, we find that SEA-specific benchmarks do not contain any Philippine language at all (as in the case of the SeaLLM Leaderboard, Liu et al. 2025) or is limited to a single Filipino language (Tagalog, as in the case of SEA-HELM, Susanto et al. 2025). FILBENCH aims to provide a more realistic evaluation of Filipino-centric tasks by having a principled approach in choosing categories that reflect the current trends and priorities of the Philippine NLP research community.

## F.2 Does high performance in one benchmark translates similarly to FILBENCH?

Set-up. In order to understand whether high performance in one benchmark translates to similar performance in FILBENCH, we compute the Spearman $\rho$ rank correlation of models that were evaluated in both benchmarks. For the SeaLLM leaderboard, we treat SeaBench and SeaExam separately. For SEA-HELM, we compute the correlation for the full evaluation suite and its Tagalog-only subset (Batayan).

Results. Figure 7 shows the raw scores for FIL-BENCH with respect to another benchmark, alongside its Spearman $\rho$ rank correlation. The results show moderate to strong positive correlations $( \rho$ = 0.571 to 0.758) between FilBench and other SEA language benchmarks, with SeaExam demonstrating the strongest predictive relationship. This suggests that model performance on one benchmark does meaningfully transfer to performance on Filipino language tasks, though the scattered distribution of data points indicates that different benchmarks capture distinct aspects of language ability. Furthermore, our findings highlight that while some transferability exists across Southeast Asian language benchmarks, benchmark-specific optimization may still be necessary for optimal performance on FilBench.

## G Analysis of Model Dis/Agreement

In this section, we show examples of agreement and disagreement from the SEA-specific models

<table><tr><td>Benchmark</td><td># Tasks</td><td># Instances</td><td>PH Languages</td><td>Data Collection Procedure</td></tr><tr><td>FILBENCH (OURS)</td><td>12</td><td>197.6k</td><td>FIL/TGL, CEB</td><td>Curated from expert-annotated datasets</td></tr><tr><td>SeaBench (Liu et al., 2025)</td><td>1</td><td>300</td><td>-</td><td>Collected from native-speakers</td></tr><tr><td>SeaExam (Liu et al., 2025)</td><td>3</td><td>5.5k</td><td>-</td><td>Collected from native-speakers</td></tr><tr><td>Batayan (Montalan et al., 2025)</td><td>8</td><td>3.8k</td><td>FIL</td><td>Curated with human annotation</td></tr></table>

Table 10: Comparison of multilingual benchmarks related to Filipino-centric tasks. Our data collection procedure allows us to scale the diversity of tasks in our suite.

![](images/a48fcd1329e3f846f37b47617d0b3cd61f2c78b9514e376655cc2955287318f2.jpg)  
Figure 7: Performance of different Multilingual and SEA-Specific models on FilBench and other SEA-specific / Tagalog benchmarks such as SeaExam and SeaBench (Liu et al., 2025), SEA-HELM (Susanto et al., 2025), and Batayan (Montalan et al., 2025).

we analyzed in §5.1.

## G.1 Set-up: Qualitative Analysis of Model Outputs during Dis/Agreement

In order to understand model behavior, we qualitatively analyze per-instance agreement between select sub-tasks within the FILBENCH evaluation suite. This involves examining instances where models either consistently agree or disagree on their outputs. By focusing on specific sub-tasks, such as Readability and Cultural Knowledge Assessment, we aim to identify patterns and potential sources of error or divergence in model predictions. We hope that this analysis helps in understanding the nuances of model performance and the challenges posed by different task types.

## G.2 Results: Examples of Model Dis/Agreement

Regional Knowledge (Fleiss’ κ = 0.393) For this task, models are required to answer questions taken from a sample of a driving exam in the Philip pines. Figure 8 shows an example where models agree on a specific answer, yet they are all incorrect. The question asks what a green arrow (berdeng arrow) indicates as a traffic signal. All models answered Option A (Vehicles are not allowed to enter the intersection as pointed by the arrow), yet the correct answer is Option B (Vehicles are allowed to turn left or right).

We also show an example where most SEAspecific models disagree in Figure 9. Here, the question asks who has right of way in an intersection without a traffic light. The correct answer is Option C (the last one to arrive), yet models tend to differ in their answers. We hypothesize that the use of the word magbigay (to give), might have

## High agreement among models but incorrect answer

Tanong: Ano ang ibig sabihin ng berdeng arrow sa signal na pang-trapiko?

A. Hindi pinapayagan ang pagpasok sa interseksyong itinuturo ng arrow.  
B. Napapahintulot sa mga sasakyan na kumaliwa o kumanan.  
C. Nagpapahintulot sa pagtawid ng mga taong tatawid.  
D. Wala sa nabanggit.

Sagot:

Model Pred (Majority): A Gold: B

Figure 8: In this example from a driving license exam, all SEA-specific models agree that the correct answer is A. However, the gold label is B.

confounded models due to its usage—leading to varied interpretations.

Readability (Fleiss’ κ = 0.207) For this task, models must determine the appropriate grade level for a given passage. In the Philippine educational system, there are three grade levels (Grades 1 to 3) for ages 6-7, 7-8, and 8-9, respectively (Imperial and Kochmar, 2023; Imperial et al., 2022). In Figure 10, all models agree that the given passage is appropriate for Grade 1 students, yet this differs from the expert-annotated gold label (Grade 2). The passage’s complexity, including the density of entities like “Mama (mother),” “eskwela (school),”

## High disagreement among models

Tanong: Sa isang sangandaan/interseksyon na walang senyas trapiko, dalawang sasakyan ang dumarating sa magkabilang kalye, aling sasakyan ang dapat magbigay?

A. Ang unang dumating  
B. Ang unang nagmarahan  
C. Ang huling dumating  
D. Wala sa nabanggit

Sagot:

Gold: C

Figure 9: In this example from a driving license exam, all SEA-specific models disagree on their answers.

and “kalsada (road / street),” likely influenced the experts to label it as Grade 2, despite its brevity and simple sentence structure, which models associated with Grade 1. On the other hand, Figure 11 shows an example where SEA-specific models disagree with one another. In this case, the high disagreement among models could be attributed to more complex vocabulary (e.g., magdahom nga kamao mokiay), overall text length, and sentence structures.

## G.3 Discussion: Implications and Potential Future Work

The consistent disagreement of models, as seen in Figure 10 to Figure 9, highlights a potential gap in the models’ understanding of culturallyspecific knowledge. This suggests that while models may have been trained on massively collected data, they might still lack the nuanced, languagespecific knowledge required for tasks (e.g., knowledge of true linguistic predictors of complexity for readability assessment in the Filipino language) compared to experts, such as linguists, who can do the tasks manually at ease.

Overall, these findings emphasize the importance of incorporating more region-specific data into model training. By doing so, we can enhance their ability to interpret and respond accurately to culturally relevant tasks, ultimately improving their performance on Filipino language tasks. This approach not only addresses the current limitations but also paves the way for developing more robust and culturally-aware language technologies.

## High agreement among models but incorrect answer

Istorya ni Sue Quirante

Usa, duha, tulo, upat, lima! Lima ka tudlo nga sayo nangmata. Unom, pito, walo, siyam, napulo! Pulo ka tudlo sa banyo naligo.

Usa ka kamot nga naghungit og sula. Duha ka kamot sabunan aron ang hugaw mawala.

Usa ka bata nga nilabang sa kalsada, nagkupot sa kamot ni Mama. Pag-abot sa eskwela, ang bata nagsulat og mga letra.

Usa ka kamot sa wala. Usa ka kamot sa tuo. Duha ka kamot nga nagkaway. Babay mga higala!

Model Pred (Majority): Grade 1 Gold: Grade 2

Figure 10: In this example, all SEA-specific models agree that the readability of the passage above is apt for Grade 1 pupils. However, the gold label indicates that the passage is for Grade 2.

## High disagreement among models

Ang Pagkiay ni Ikay

Gisuwat ni: Juna J. Presbitero

Si Ikay nagtungha sa ikaduhang ang-ang. Kataw-an siya sa iyang kahimsog.

Dili lang niya tagdon ang ilang mga pagsaway. Kay para kaniya gwapa ang iyang dagway.

Sa eskuylahan adunay indigay sa pagsayaw. Walay gustong moapil kay silang tanan maulaw.

Niigon si Ikay nga siya moapil sa indigay. Kay ganahan siya nga mokiay.

Wala sila magdahom nga kamao mokiay si Ikay. Ug nisulting moapil sa maong indigay.

Sa indigay nipakita si Ikay sa iyang pagkiay-kiay. Ang tanan nalingaw sa iyang pagsayaw.

Gihatag ang unang ganti ngadto ni Ikay. Gitawag siya nga batang kusog mokiay.

Malipayon si Ikay sa iyang kadaugan. Sukad niadto gitahod na siya sa iyang mga kauban.

Gold: Grade 2

Figure 11: In this example, all SEA-specific models disagree on the readability level of the given text.

## H Generation Failure Modes Examples

Table 11 provide examples of common failure modes in LLMs evaluated on the Generation category.

<table><tr><td>Failure mode</td><td>Prompt</td><td>Gold</td><td>Prediction</td></tr><tr><td rowspan="4">Poor instruction-following</td><td>EN: I’ve never been to such a nice party.FIL:</td><td>Hindi ko pa naranasan ang ganitong kagandahang party.</td><td>I’m glad to hear that you’re enjoying yourself! It sounds like you’re having a wonderful time.</td></tr><tr><td>EN: Fever subsequently developed in three asymptomatic patients.FIL:</td><td>Ang lagnat ay nabuo dulot nito sa tatlong asymptomatic na pasyente.</td><td>Fever was present in three patients who were asymptomatic.</td></tr><tr><td>EN: Clean water is scarce.FIL:</td><td>Kulang na kulang ang malinis na tubig.</td><td>L’eau propre est rare.</td></tr><tr><td>EN: Do you think we’ll have good weather tomorrow?FIL:</td><td>Sa tingin mo may magandang panahon tayo bukas?</td><td>I don’t know. It’s hard to say.EN: What about the day after tomorrow?</td></tr><tr><td rowspan="2">Overgeneration</td><td>CEB: Napuyo ko ron sa Boston.EN:</td><td>I’m living in Boston now.</td><td>I’m going to Boston. CEB: Napuyo ko ron sa Boston. EN: I’m going to Boston. (continued...)</td></tr><tr><td>EN: Yumi studied English last night.FIL:</td><td>Nag-aral ng Ingles si Yumi kagabi.</td><td>Nag-aral si Yumi ng Ingles kagabi. This is the Filipino translation of the English sentence. It means &quot;Yumi studied English last night.&quot;</td></tr><tr><td rowspan="2">Hallucination</td><td>CEB: Gigutom ko pag-ayo.EN:</td><td>I am very hungry.</td><td>Napo - Hamon sa politika laban sa pagkain Nagmukhang istilong moral na singaw ang saligan ng discurso</td></tr><tr><td>EN: Within minutes a wall of water crashes onto the shore, carrying away buildings and cars.FIL:</td><td>Sa ilang minuto, isang pader ng tubig ang tumama sa dalampasigan, tangay ang mga gusali at kotse.</td><td>More than 300 people missing;25 top levels blown away</td></tr></table>

Table 11: Addtional examples per common failure mode.

## I Analysis of Generation Failure Modes for Translation Tasks

## I.1 Set-up: Case Study of Tatoeba

We further examine generations for the Tatoeba dataset, given how most models perform poorly on it even after providing few-shot examples. We look at results per language pair (ENG → FIL and CEB → ENG) and discuss possible challenges models face in generating accurate translations.

## I.2 Results: Common Reasons why Models Fail in Generation Tasks

Finding # 1: Overgeneration on shorter texts. We find that models tend to overgenerate outputs on short prompts even in few-shot settings. While observed on all generation datasets, this issue impacts performance on Tatoeba the most due to its much shorter texts (average length of 5.90 tokens) compared to NTREX-128 (21.04) and TICO-19 (21.67). Only the GPT and Llama 4 models consistently produce concise outputs, which explains their higher performance on generation tasks compared to other models.

Finding # 2: Few-shot prompting improves instruction-following but not generation quality. Zero-shot generations for ENG → FIL often appear in the wrong target language. Smaller models in particular are especially prone to misinterpreting instructions such as responding to the source text directly or generating multi-turn dialogues. As shown for the case of SeaLLMs 7B in Table 12, providing one-shot examples helps models begin translating correctly into Tagalog; however, the accuracy and fluency of outputs considerably vary.

In contrast, zero-shot generations for CEB → ENG are more consistently in the correct language.

However, hallucinations are common across all model sizes and more pronounced among non-SEA-specific models. Models that do interpret instructions correctly often produce semantically inaccurate translations. This issue is still observed even after providing examples, which explains why performance gains between zero-shot and few-shot on this pair are comparatively minimal.

Finding #3: Confusion between Filipino and Cebuano. We also observe variability in generation quality on texts with words common to Filipino and Cebuano. Outputs are generally accurate on cognates or words with similar meanings, but incorrectly generate around the Tagalog meaning when they differ. To a lesser extent, we also observe this in words with different spellings but have significant character overlap. Table 13 shows examples of this issue across models of different language coverage and size.

## I.3 Discussion

Our findings show frequent overgeneration and poor instruction-following in shorter prompts, consistent with the findings of Wan et al. (2022) on neural machine translation. They attribute this to short texts providing insufficient contextual information for accurate generation. To address this, we recommend incorporating one-shot examples and constraining output length through token limits or frequency/length penalties.

We also find evidence of language misidentification biased towards Filipino in entries with shared vocabulary. We hypothesize that the linguistic similarity between Filipino and Cebuano facilitates cross-lingual transfer within models (Eronen et al., 2023; Philippy et al., 2023), but also makes it harder for them to distinguish between the two, especially with Cebuano’s limited representation in pre-training data (Cahyawijaya et al., 2024). Given this, we stress the importance of human validation on machine-translated texts, especially in practical applications where semantic accuracy is crucial.

<table><tr><td rowspan="2">Prompt</td><td rowspan="2">Gold</td><td colspan="2">Predictions</td><td rowspan="2">Remarks (1-shot)</td></tr><tr><td>0-shot</td><td>1-shot</td></tr><tr><td>EN: What is your name?FIL:</td><td>Ano ang pangalan mo?</td><td>My name is Fil.</td><td>Ano ang pangalan mo?</td><td>Correct translation.</td></tr><tr><td>EN: I regret not having taken his advice.FIL:</td><td>Ako ay nagsisi na hinde ko sinunod ang kanyang payo.</td><td>I regret not having taken his advice.</td><td>Naisip ko na lang na hindi ko na kailangan ng kanyang payo.</td><td>Translation is semantically incorrect.</td></tr><tr><td>EN: He&#x27;s behind the times in his methodsFIL:</td><td>Luma na ang mga paraan niya.</td><td>Il est en retard sur son temps dans ses méthodes.</td><td>Siya ay nakalayo sa pana-hon sa kanyang pamama-raan.</td><td>Translation is semantically correct but doesn&#x27;t sound natural.</td></tr></table>

Table 12: Sample zero- and one-shot generations by SeaLLMs 7B on the Filipino split of Tatoeba. For brevity, outputs are truncated at the first complete sequence due to overgeneration.

<table><tr><td>Example</td><td>Remarks</td></tr><tr><td>Prompt: CEB: Nagkinahanglan ko og kwarta, kay kinahanglan ko nga magbayad sa akong kuryente.Gold: I need money because I need to pay for my electricity.</td><td>The words kwarta (&#x27;money&#x27;), magbayad (&#x27;to pay&#x27;) and kuryente (&#x27;electricity&#x27;) have the same meaning in Tagalog.</td></tr><tr><td>Babel 83B: I need money, because I have to pay my electricity bill.Qwen 2.5 7B: I need money because I need to pay my electricity bill.SEA-LION v3 70B: I need some money because I had to pay for my electric bill.SeaLLMs 7B: I need money, because I need to pay my electricity bill.</td><td></td></tr><tr><td>Prompt: CEB: Kung naay aso, naay kayo. EN:Gold: Where there is smoke, there is fire.</td><td>The word aso means &#x27;smoke&#x27; in Cebuano, but &#x27;dog&#x27; in Tagalog.</td></tr><tr><td>Babel 83B: Kung aso may roon, may kami rin.Qwen 2.5 7B: If there is a dog, there is you.SEA-LION v3 70B: If there&#x27;s a dog, there&#x27;s a chain/leash.SeaLLMs 7B: If you have a dog, you have a problem.</td><td></td></tr><tr><td>Prompt: Gisuginlan ni Tom si Mary bahin sa kawat. EN:Gold: Tom talked to Mary about the robbery.</td><td>The word kawat (&#x27;robbery&#x27;) is likely being confused by SEA-LION v3 70B for kawad (&#x27;wire&#x27;) in Tagalog.</td></tr><tr><td>Babel 83B: Ceb: Gisugod ni Tom si Mary bahin sa kawat.Qwen 2.5 7B: Tom and Mary are partners in the business.SEA-LION v3 70B: Tom told Mary about the wire.SeaLLMs 7B: Tom gisuginlan si Mary bahin sa kawat.</td><td></td></tr></table>

Table 13: Sample generations by multilingual (Babel 83B and Qwen 2.5 7B) and SEA-specific models (SEA-LION v3 70B and SeaLLMs 7B) on entries with common Tagalog and Cebuano words from the Cebuano split of Tatoeba. Outputs are truncated due to overgeneration.

![](images/ad2e808d4bfe27cdcd0af6bde1f9450001d467f715bd36957c0a03535e13ff30.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Initial Paper Scrape\nScopus"] --> B["NLP Sub-Field Annotation"]
  B --> C["Manual Filtering and Verification"]
  
  subgraph Search_Query ["Search Query: (ALL("nlp") OR ALL("natural language processing")) AND (ALL("philippines") OR ALL("filipino") OR ALL("tagalog"))"]
  end
  
  subgraph Prompt ["GPT-4 Prompt"]
    D["Classify the paper below to any of the following labels... {tracks} {title} {abstract}&quot;"]
  end
  
  subgraph Guidelines ["Filtering Guidelines"]
    E{"If published in ACL, check if track is correct."}
    F["Ensure correct metadata"]
  end
  
  style A fill:#f9f9f9,stroke:#333,stroke-width:2px
```
</details>

Figure 12: In order to determine the research priorities of the Filipino NLP research community that will inform the categories of FILBENCH (i.e., Cultural Knowledge, Classical NLP, Reading Comprehension, Generation), we annotated 223 Scopus-index papers from 2006 to 2023 and assigned them with their respective NLP sub-fields.

## J Research Priorities in Filipino NLP

When curating FILBENCH, we made opinionated and principled choices as to which categories (i.e., CK, CN, RC, and GN) to include in the suite. In general, we based our decisions on the research priorities of the Filipino NLP community, as that reveals the type of applications where language technologies are useful from a local perspective. We describe the process and findings in this section.

Set-up. In order to obtain an overview of trends in NLP research in the Philippines, we follow the process as shown in Figure 12.

• Initial paper scrape. We closely follow Roxas et al. (2021)’s data collection approach and scrape the Scopus database of all research papers from 2006 to 2023 that includes any mention of the terms philippines, filipino, or tagalog (see search query in Figure 12). We chose Scopus in order to increase the breadth of our search: not only because it indexes papers from <sup>⋆</sup>ACL/EMNLP conferences, but also due to the academic culture in Philippine universities that incentivizes researchers to publish in Scopusindexed journals.

• NLP sub-field annotation. Then, we prompt GPT-4 to assign their NLP sub-field based on the common tracks from past ACL conferences. We formulate the prompt by including the title and abstract of the paper-in-question, and provide a list of ACL tracks to choose the label from (Figure 16).

![](images/14cc38f22bac64797acecd0d7967438519ab287d905bcf88b751617a1f23181a.jpg)

<details>
<summary>bar_stacked</summary>

| Year | Orange | Brown | Pink | Green | Light Blue | Teal |
| --- | --- | --- | --- | --- | --- | --- |
| 2006 | ~0.5 | ~2.5 | ~0.5 | ~0.5 | ~0.5 | ~0.5 |
| 2008 | ~0.5 | ~1.5 | ~0.5 | ~0.5 | ~0.5 | ~0.5 |
| 2009 | ~1.5 | ~1.5 | ~0.5 | ~3.5 | ~1.5 | ~0.5 |
| 2010 | ~0.5 | ~1.5 | ~0.5 | ~0.5 | ~0.5 | ~0.5 |
| 2011 | ~0.5 | ~1.5 | ~0.5 | ~0.5 | ~3.5 | ~0.5 |
| 2012 | ~2.5 | ~0.5 | ~0.5 | ~0.5 | ~3.5 | ~0.5 |
| 2013 | ~2.5 | ~0.5 | ~0.5 | ~0.5 | ~1.5 | ~0.5 |
| 2014 | ~1.5 | ~1.5 | ~3.5 | ~1.5 | ~3.5 | ~0.5 |
| 2015 | ~1.5 | ~2.5 | ~1.5 | ~0.5 | ~0.5 | ~0.5 |
| 2016 | ~2.5 | ~1.5 | ~1.5 | ~1.5 | ~0.5 | ~0.5 |
| 2017 | ~13.5 | ~4.5 | ~3.5 | ~1.5 | ~3.5 | ~1.5 |
| 2018 | ~9.5 | ~4.5 | ~3.5 | ~1.5 | ~1.5 | ~1.5 |
| 2019 | ~6.5 | ~4.5 | ~3.5 | ~2.5 | ~4.5 | ~1.5 |
| 2020 | ~3.5 | ~3.5 | ~3.5 | ~4.5 | ~3.5 | ~1.5 |
| 2021 | ~11.5 | ~9.5 | ~6.5 | ~3.5 | ~3.5 | ~1.5 |
| 2022 | ~9.5 | ~6.5 | ~4.5 | ~7.5 | ~3.5 | ~1.5 |
| 2023 | ~3.5 | ~3.5 | ~3.5 | ~4.5 | ~4.5 | ~1.5 |
</details>

NLP Sub-Field  
![](images/81084c5a0ef904566066b3323c03bbccf52601dedccd596f2fcdf2a4f9a15a8c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Computational Social Science"] --> B["Large Language Models"]
  C["Dialogue"] --> D["Lexical Semantics"]
  E["Ethics And NLP"] --> F["Linguistic Diversity"]
  G["Information Extraction"] --> H["Linguistic Theories"]
  I["Information Retrieval"] --> J["Machine Learning For NLP"]
  K["Interpretability"] --> L["Machine Translation"]
  M["Multilingualism And Cross-Lingual NLP"] --> N["Multimodality"]
  O["NLP Applications"] --> P["Natural Language Generation"]
  Q["Phonology, Morphology, And Word Segmentation"] --> R["Question Answering"]
  S["Resources And Evaluation"] --> T["Sentiment Analysis"]
  U["Speech Processing"] --> V["Summarization"]
  W["Syntax Tagging, Chunking, And Parsing"] --> X["Query Answering"]
```
</details>

Figure 13: Increase in topic diversity. Through the years, the number of topics relating to Philippine languages and their diversity increased from 2006 to 2023. This trend stresses the need for FILBENCH’s diversity in terms of the number of categories and tasks.

![](images/f1338b55b88b034b227ef5b3d2cc1c6f4106d6399d2b8ec13b901faee50a41b6.jpg)

<details>
<summary>bar</summary>

| NLP Sub-Field | Number of papers published |
| :--- | :--- |
| Information Extraction | ~54 |
| NLP Applications | ~28 |
| Sentiment Analysis | ~21 |
| Resources And Evaluation | ~20 |
| Machine Translation | ~20 |
| Speech Processing | ~17 |
| Computational Social Science | ~13 |
| Machine Learning For NLP | ~11 |
| Natural Language Generation | ~10 |
| Syntax Tagging, Chunking, And Parsing | ~9 |
| Phonology, Morphology, And Word Segmentation | ~8 |
| Multilingualism And Cross-Lingual NLP | ~6 |
| Information Retrieval | ~5 |
| Ethics And NLP | ~4 |
| Linguistic Theories | ~3 |
| Linguistic Diversity | ~3 |
| Multimodality | ~3 |
| Summarization | ~3 |
| Large Language Models | ~3 |
| Dialogue | ~2 |
| Lexical Semantics | ~2 |
| Question Answering | ~2 |
| Interpretability | ~2 |
</details>

Figure 14: Distribution of papers per NLP sub-field that includes any Philippine language. This highlights the priorities of the Philippine NLP research community which helped inform the categories of FILBENCH.

• Manual filtering and verification. We perform manual filtering and re-annotation to ensure the correctness of labels. This includes checking the parity of an ACL paper’s predicted sub-field to the actual ACL track it was published or correcting the NLP sub-field in the case of wrong silver annotations.

This process results in 223 papers on Filipino NLP, containing the title, abstract, authors, and publication year, which we then use for this study.

Results. Figure 14 shows the frequency of papers for each NLP sub-field that is related to Philippine languages from 2006 to 2023. The five most common topics relate to information extraction, NLP applications, sentiment analysis, machine translation, and resources & evaluation. This distribution of topics aligns well with the four categories of FIL-BENCH. For instance, the prominence of information extraction and sentiment analysis supports the inclusion of the CK and CN categories. The focus on machine translation justifies the GN category, while the emphasis on resources and evaluation (which include papers in NLI and readability) highlights the inclusion of the RC category. In addition, Figure 13 shows the increasing diversity of topics in Filipino NLP through the years. Aside from a sharp increase in published papers from 2017, there is also a wider breadth of topics by 2023.

Discussion. When aggregating these NLP subfields for FILBENCH, we focus on specific trends in topics rather than a many-to-one mapping of subfields to category because we find that these NLP sub-fields overlap. For example, some papers in the Linguistic Diversity and Multilingualism subfield can also be in the Resources and Evaluation track. However, these trends inform us of which categories to prioritize. In general, the categories in FILBENCH are opinionated, yet principled due to them being informed by past and present trends of topics published in Filipino NLP.

![](images/3d136c8d44dd425aa950317a45fe656e5556b0df9cf622067a62e7b8269eb757.jpg)

<details>
<summary>scatter</summary>

| Model | Inference Cost ($/1M output tokens), log | FilBench Score |
| --- | --- | --- |
| SEA LION v3 8B | ~0.04 | ~47.5 |
| SEA LION v3 9B | ~0.06 | ~53.5 |
| SEA LION v3 9B | ~0.08 | ~56.2 |
| SEA LION v3 9B | ~0.1 | ~47.5 |
| SEA LION v3 9B | ~0.12 | ~54.2 |
| SEA LION v3 9B | ~0.2 | ~55.2 |
| SEA LION v3 9B | ~0.3 | ~55.2 |
| SEA LION v3 9B | ~0.4 | ~63.2 |
| SEA LION v3 9B | ~0.5 | ~63.2 |
| SEA LION v3 9B | ~0.6 | ~67.8 |
| gpt-4o-mini | ~0.6 | ~49.8 |
| Mixtral 8x22B v0.1 | ~1.2 | ~54.2 |
| gpt-4o-mini | ~10 | ~71.8 |
| Llama 4 Maverick | ~10 | ~71.8 |
| Llama 4 Scout | ~0.3 | ~61.2 |
</details>

Figure 15: Pareto frontier illustrating the trade-off between FILBENCH score and inference cost (log scale). SEA-specific models such as SEA-LION v3 can achieve high FILBENCH scores efficiently.

## K Cost-Efficiency of LLMs on Filipino Language Tasks

As LLMs have become ubiquitous in the Philippines, it is necessary to determine whether LLM users and developers are paying a fair price relative to their capabilities. In this section, we address the question of which model offers the optimal balance between performance and cost-effectiveness.

Set-up. In order to measure the cost-efficiency of different LLMs, we compare their per-token pricing for output-tokens as published on OpenRouter<sup>2</sup> with respect to their FILBENCH score. We use the current pricing as of the current time of the experiments, and obtain the lowest price tier. We then exclude models that are not available in OpenRouter (or use the price of a model with a comparable parameter size). For some models not in Open-Router but was finetuned from a base model (e.g., Llama-3.1-SEA-LION-v3-8B-IT is a finetune of Llama-3.1-8B-Instruct), we use the per-token price of the base model. This methodology lies in the assumption of using OpenRouter’s API to estimate cost: we do not include operational costs for hosting a model or using batch inference APIs from other hosting providers.

Results. Figure 15 shows the per-token output inference cost (\$/1M in log scale) of each model with respect to their FILBENCH scores. Despite being the top-performing model on FILBENCH, GPT-4o is significantly more expensive than Llama-4 Maverick. This suggests that while GPT-4o offers superior performance, its cost may not be justified for all applications, especially when more cost-effective models like Llama-4 Maverick can achieve competitive results at a fraction of the cost. In addition, we also find that SEA-specific models, especially the SEA-LION family, lies near the Pareto frontier of cost-efficiency (based on our pricing assumptions).

Discussion. The Philippines is one of the most active users of ChatGPT in the world (Group, 2024). As language technologies continue to dominate both consumer and enterprise-facing applications (Liu and Wang, 2024; Cucio and Hennig, 2025), it is then relevant to ask whether there is a more costefficient approach in taking advantage of such systems. Our findings suggest that despite GPT-4o’s performance on FILBENCH, there are still more cost-effective solutions such as using open-source models such as Llama-4 Maverick with a small percentage drop in performance but at a fraction of the cost. Moreover, there is promise in investing in post-training efforts to finetune existing models with Filipino-centric training data as our findings suggest that models finetuned specifically for Filipino such as SEA-LION are at the Pareto frontier of cost-efficiency.

## L Effect of Prompt Template in Generation Performance

In the current implementation of FILBENCH, we use the out-of-the-box translation templates from lighteval in order to provide comparable scores to other benchmarks built on top of that framework. In this section, we explore whether how changes in the translation prompt template affect Generation performance.

Set-up. We follow six translation prompt templates from Zhang et al. (2023a) and evaluate GPT-4o on GN tasks from FILBENCH.

Results. Table 14 shows the ROUGE-L scores of GPT-4o on different prompt templates. Our findings suggest that Template B can potentially result in better translation performance as using it for zero-shot translation led to higher ROUGE-L scores overall. However, we find that there is still no clear pattern on the relationship between prompt template and performance. In FILBENCH, we follow the standard formulation of lighteval to obtain baseline floor performance of LLMs for any Generation task.

<table><tr><td rowspan="2">ID</td><td rowspan="2">Prompt Template</td><td colspan="4">ROUGE-L</td></tr><tr><td>Tatoeba (CEB)</td><td>Tatoeba (TGL)</td><td>NTREX</td><td>TICO</td></tr><tr><td>A</td><td>:◇:</td><td>33.78</td><td>51.88</td><td>38.96</td><td>53.03</td></tr><tr><td>B</td><td>◇:</td><td>41.78</td><td>50.32</td><td>58.10</td><td>61.85</td></tr><tr><td>C</td><td>◇ Translate to:</td><td>42.92</td><td>52.85</td><td>56.25</td><td>60.53</td></tr><tr><td>D</td><td>◇ Translate fromto:</td><td>35.57</td><td>55.34</td><td>57.37</td><td>61.53</td></tr><tr><td>E</td><td>:◇ Translate to:</td><td>39.84</td><td>29.76</td><td>25.20</td><td>31.04</td></tr><tr><td>F</td><td>:◇ Translate fromto:</td><td>44.41</td><td>18.62</td><td>17.54</td><td>19.73</td></tr></table>

Table 14: GPT-4o performance on different prompt templates. A template may contain the name or ISO-693 code of the source (<src>) or target (<tgt>) language, and the input text (<input>). A diamond symbol (⋄) indicates a line break. Finally, we use Template A for Generation tasks in FILBENCH.

## GPT-4 Prompt for Classification

System Prompt: You are a helpful and truthful expert text classification system. Your task is to accept Text as input and provide a category for the text based on the predefined labels.

User Prompt: Classify the text below to any of the following labels:

computational social science

dialogue

discourse and pragmatics

ethics and nlp

natural language generation

information extraction

information retrieval

interpretability

language grounding to vision, robotics, and beyond

large language models

linguistic diversity

linguistic theories

cognitive modeling

psycholinguistics

machine learning for nlp

machine translation

multilingualism and cross-lingual nlp

nlp applications

phonology, morphology, and word segmentation

question answering

resources and evaluation

lexical semantics

sentence-level semantics

textual inference

sentiment analysis

stylistic analysis and argument mining

speech processing

multimodality

summarization

syntax tagging, chunking, and parsing

Here are some examples:

{ for example in examples } {{ example.title }}

{{ example.abstract }}

Label: {{ example.label }}

{ endfor }

Here is the paper you need to classify:

{{ paper.title }}

{{ paper.abstract }}

Label:

Figure 16: GPT-4 Prompt used to predict a paper’s NLP sub-field based on their title and abstract. We show few-shot examples from existing papers with known NLP sub-fields from the ACL Anthology.

## M Task Formulation

In this section, we show an example prompt for each sub-task in FILBENCH.

## CN: Text Classification

## Original Prompt (Filipino):

Tungkol ba sa dengue ang sumusunod na pangungusap? Piliin ang tamang sagot:

Not a good time to get sick.

A. Hindi

B. Oo

Sagot:

## Translated Prompt:

Is the following sentence about dengue? Select the correct answer:

Not a good time to get sick.

A. No

B. Yes

Answer:

Figure 17: Example task adapted from Dengue Filipino (Livelo and Cheng, 2018) in the Classical NLP category.

## CN: Named Entity Recognition

## Original Prompt (Cebuano):

Pangutana: Unsa ang ginganlan nga named-entity sa pulong ’Osmeña’ niini nga sentence: Gipasabot ni Osmeña nga makadagiot ang dakbayan sa suhilan sa mga drayber .

A. PERSON

B. ORGANIZATION

C. LOCATION

D. OTHER

Tubag:

## Translated Prompt:

What type of named entity is the term ’Osmeña’ in this sentence: Osmeña explained that the city can save drivers money.

A. PERSON

B. ORGANIZATION

C. LOCATION

D. OTHER

Answer:

Figure 18: Example task adapted from CebuaNER (Pilar et al., 2023) in the Classical NLP category.

## CN: Sentiment Analysis

## Original Prompt (Filipino):

Tanong: Ano ang damdamin o sentimiyento ng sumusunod na pangungusap: im very disappointed kasi di gumana ang dalawa kung order

A. Negatibo

B. Neutral

C. Positibo

Sagot:

## Translated Prompt:

Question: What is the emotion or sentiment of the following sentence: im very disappointed because my two orders didn’t work

A. Negative

B. Neutral

C. Positive

Answer:

Figure 19: Example task adapted from FiReCS (Cosme and De Leon, 2023) in the Classical NLP category.

## CK: Regional Knowledge

## Original Prompt (Tagalog):

Tanong: Niregaluhan ka ng iyong nanay ng laruang babasagin, sa hindi sinsadyang pangyayari nabitawan mo ito at nabasag.

A. Hihingi ulit ng bagong regalo  
B. Pababayaan ang nabasag na laruan para Makita ng nanay  
C. Magdadahilan para di mapagalitan  
D. Ihihingi ngpaumanhin sa nanay ang nangyari.  
Sagot:

## Translated Prompt:

Question: Your mother gifted you a fragile toy, you accidentally dropped it and it shattered.

A. Ask for a new gift  
B. Leave the shattered gift for your mother to see  
C. Make up an excuse so you don’t get scolded  
D. Apologize to your mother for what happened.  
Answer:

Figure 20: Example task from INCLUDE (Romanou et al., 2024) in the Cultural Knowledge category.

## CK: Factual Knowledge

## Original Prompt (Tagalog):

Tanong: Hanapin ang degree para sa ibinigay na field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

A. 0

B. 4

C. 2

D. 6

Sagot:

## Translated Prompt:

Question: Find the degree of the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q

A. 0

B. 4

C. 2

D. 6

Answer:

Figure 21: Example task from Global MMLU (Singh et al., 2024) in the Cultural Knowledge category.

## CK: Cultural Values

## Original Prompt (Tagalog):

Tanong: Nasa hustong gulang na ako.

Nasa lamay ako ngayon. Nakikita kong nagsusugalan ang mga tao doon. Niyaya nila akong sumali.

Anong maaari kong gawin?

A. Nakakabastos sa mga namatayan ang pagsusugal. Pagsabihan mo ang mga nagsusugal na itigil ang kanilang ginagawa.  
B. Maaari kang sumali sa pagsusugal dahil normal ito sa lamay upang manitiling gising sa pagbabantay sa patay.  
C. Mas mainam na ayawan ang kanilang pagyaya dahil makikita ng mga namatayan ang iyong pagbigay-galang.  
D. Huwag kang sumali dahil salungat ito sa iyong paniniwala. Hindi naman nakababastos ang direktang pagtanggi sa pagyaya ng mga nagsusugal.

Sagot:

## Translated Prompt:

Question: I am an adult.

I am currently at a funeral. I see people gambling there. They are inviting me to join.

What should I do?

A. Gambling is disrespectful to the deceased. Tell the gamblers to stop what they are doing.  
B. You can join in gambling because it is normal at the funeral to stay awake to watch over the dead.  
C. It is better to refuse their invitation because the deceased will see your respect.  
D. Don’t join because it goes against your beliefs. It is not disrespectful to directly refuse the gamblers’ invitation.

Answer:

Figure 22: Example task from KALAHI (Singh et al., 2024) in the Cultural Knowledge category.

## CK: Word-sense Disambiguation

## Original Prompt (Tagalog):

Question: Is the usage of Halaman in this sentence correct?

Nagdilig ako ng halaman kaninang umaga.

A. Yes

B. No

Answer:

Sagot:

## Translated Prompt:

Question: Is the usage of "Plant" in this sentence correct?

I watered a plant earlier this morning.

A. Yes

B. No

Answer:

Figure 23: Example task from StingrayBench (Cahyawijaya et al., 2024) in the Cultural Knowledge category.

## GN: Document Translation

## Prompt:

EN: Welsh AMs worried about ’looking like muppets’ FIL:

## Label:

Mga Welsh na AM nangangambang ’magmukhang mga muppet

Figure 24: Example task from NTREX-128 (Federmann et al., 2022) in the Generation category.

## GN: Realistic Translation

## Prompt:

CEB: Ambot unsaon ta ka pagpahibalo. EN:

## Label:

I don’t know how to contact you.

Figure 25: Example task from the Cebuano split of Taoteba (Tiedemann, 2020) in the Generation category.

## GN: Domain-Specific Translation

## Prompt:

EN: and are you having any of the following symptoms with your chest pain FIL:

## Label:

At mayroon ka bang alinman sa mga sumusunod na sintomas kasama ng pananakit ng iyong dibdib

Figure 26: Example task from TICO-19 (Anastasopoulos et al., 2020) in the Generation category.

## RC: Natural Language Inference

## Original Prompt (Filipino):

Dagdag pa ni Corona, bunga ng 45 na taon niyang pagtatrabaho sa private at public sector ang kanyang naipong pera. Tanong: Dahil sa matinding pagbaha dulot ng walang tigil na pag-ulan, isinailalim na sa state of calamity ang isang bayan sa

lalawigan ng Maguindanao.

A. Totoo

B. Hindi totoo

Sagot:

## Translated Prompt:

Corona added that his accumulated money is the result of his 45 years of working in the private and public sectors.

Question: Due to severe flooding caused by incessant rains, a town in

Maguindanao province has been placed under a state of calamity.

A. True

B. False

Answer:

Figure 27: Example task adapted from the NewsPH NLI (Cruz et al., 2021) in the Reading Comprehension category.

## RC: Reading Comprehension

## Original Prompt (Cebuano):

Natawo sa kapital sa Croatia, Zagreb, si Bobek nakaangkon og kabantog samtang nagadula para sa Partizan Belgrade. Miapil siya sa team kaniadtong 1945 ug nagpabilin hangtod 1958. Sa naa pa siya sa kuponon, nakapuntos siya og 403 ka goal sa 468 nga pag-apil. Walay laing nakahimo og mas daghang pagpakita o naka-iskor og mas daghan nga goal para sa grupo kaysa kay Bobek. Kaniadtong 1995, giboto siya nga labing maayo nga magdudula sa kasaysayan sa Partizan.

Pangutana: Hain sa mosunod ang wala tukmang nagpakita sa karera ni Bobek sa Partizan Belgrade?

A. Naka-iskor siya og labaw sa 468 ka goal samtang nagduwa para sa team

B. Naka-iskor siya og mas daghang goal kaysa sa bisan kinsang ubang mga manunuwa

C. Nabotar siya ingong pinakamaayong manunuwa sa kasaysayan sa team

D. Nigawas siya sa mas daghang duwa kaysa sa bisan kinsang ubang manunuwa

Tubag:

## Translated Prompt:

Born in the Croatian capital, Zagreb, Bobek rose to fame while playing for Partizan Belgrade. He joined the team in 1945 and stayed until 1958. During his time on the team, he scored 403 goals in 468 appearances. No one else has made more appearances or scored more goals for the team than Bobek. In 1995, he was voted the best player in Partizan history.

Question: Which of the following does not accurately reflect Bobek’s career at Partizan Belgrade?

A. He scored more than 468 goals while playing for the team

B. He scored more goals than any other player

C. He was voted the best player in the team’s history

D. He appeared in more matches than any other player

Answer:

Figure 28: Example task from the Cebuano split of Belebele (Bandarkar et al., 2024) in the Reading Comprehension category.

## RC: Readability

## Original Prompt (Cebuano):

Pangutana: Unsa ang angay nga lebel sa grado alang sa mosunod nga teksto?

Grade 1 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 6-7.

Grade 2 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 7-8.

Grade 3 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 8-9.

Ang Gatas sa Lata

Sinuwat ni: Milagros Meca

Story Book

Cebuano

Ang baso.

Lata sa gatas.

Gatas sa baso.

Baso ug lata.

Ang baso may gatas.

May gatas ang lata.

## KATAPUSAN

A. Grade 1

B. Grade 2

C. Grade 3

Tubag:

## Translated Prompt:

Question: What is the appropriate grade level for the following text?

Grade 1 - the text can be read by someone between the ages of 6-7.

Grade 2 - the text can be read by someone between the ages of 7-8.

Grade 3 - the text can be read by someone between the ages of 8-9.

The Milk in the Can

Written by: Milagros Meca

Story Book

Cebuano

The glass.

Can for milk.

Milk in the glass.

Glass and can.

The glass has milk.

The can has milk.

## END

A. Grade 1

B. Grade 2

C. Grade 3

Answer:

Figure 29: Example task adapted from the Cebuano Readability Corpus (Imperial et al., 2022) in the Reading Comprehension category.