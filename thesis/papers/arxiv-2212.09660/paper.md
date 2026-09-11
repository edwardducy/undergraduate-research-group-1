# The Decades Progress on Code-Switching Research in NLP:  
A Systematic Survey on Trends and Challenges

 Genta Indra Winata Affiliation: Bloomberg Email: [gwinata@bloomberg.net](mailto:)    Alham Fikri Aji Affiliation: MBZUAI Email: [alham.fikri@mbzuai.ac.ae](mailto:)    Zheng-Xin Yong Affiliation: Brown University Email: [contact.yong@brown.edu](mailto:)    Thamar Solorio ††thanks:  The work was done while at Bloomberg. Affiliation: Bloomberg 

###### Abstract

Code-Switching, a common phenomenon in written text and conversation, has been studied over decades by the natural language processing (NLP) research community. Initially, code-switching is intensively explored by leveraging linguistic theories and, currently, more machine-learning oriented approaches to develop models. We introduce a comprehensive systematic survey on code-switching research in natural language processing to understand the progress of the past decades and conceptualize the challenges and tasks on the code-switching topic. Finally, we summarize the trends and findings and conclude with a discussion for future direction and open questions for further investigation.

 

## 1 Introduction

Code-Switching is the linguistic phenomenon where multilingual speakers use more than one language in the same conversation [Poplack (1978)](#bib.bib289 ""). The fragment of the worldwide population that can be considered multilingual, i.e., speaks more than one language, far outnumbers monolingual speakers [Tucker (2001)](#bib.bib383 ""); [Winata et al. (2021a)](#bib.bib406 ""). This alone makes a compelling argument for developing NLP technology that can successfully process code-switched (CSW) data. However, it was not until the last couple of years that CSW-related research became more popular [Sitaram et al. (2019)](#bib.bib346 ""); [Jose et al. (2020)](#bib.bib188 ""); [Doğruöz et al. (2021)](#bib.bib113 ""), and this increased interest has been motivated to a large extent by: 1) The need to process social media data. Before the proliferation of social media platforms, it was more common to observe code-switching in spoken language and not so much in written language. This is not the case anymore, as multilingual users tend to combine the languages they speak on social media; 2) The increasing release of voice-operated devices. Now that smart assistants are becoming more and more accessible, we have started to realize that assuming users will interact with NLP technology as monolingual speakers is very restrictive and does not fulfill the needs of real-world users. Multilingual speakers also prefer to interact with machines in a CSW manner [Bawa et al. (2020)](#bib.bib55 ""). We show quantitative evidence of the upward trend for CSW-related research in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges").

![Refer to caption](2212.09660v2/images/publication-percent.png) 

![Refer to caption](2212.09660v2/images/ACL_conference_workshop.png) 

Figure 1: Number of publications over time in \*CL and ISCA venues. We collect the papers on October 2022. Top: Relative to all \*CL and ISCA papers. Bottom: absolute number, broken down into conferences vs workshops. It does not include papers published after. The graphs do not show the number of publications published in journals and symposiums.

| Category                                                                                                   |
| ---------------------------------------------------------------------------------------------------------- |
| Bilingual, Trilingual, 4+                                                                                  |
| Conference, Workshop, Symposium, Book                                                                      |
| Theory / Linguistics, Empirical, Analysis, Position/Opinion/Survey, Metric, Corpus, Shared Task, Demo      |
| Social Media, Speech (Recording), Transcription, News, Dialogue, Books, Government Document, Treebank      |
| Rule/Linguistic Constraint, Statistical Model, Neural Network, Pre-trained Model                           |
| Text: Topic Modeling, Semantic Parsing, Dependency Parsing, Sentiment Analysis, Emotion Detection,         |
| Abusive Language Detection, Sarcasm Detection, Humor Detection, Humor Generation, Dialogue State Tracking, |
| Text Generation, Natural Language Understanding, Named Entity Recognition, Part-of-Speech Tagging,         |
| Natural Language Entailment, Language Modeling, Regression, Language Identification, Machine Translation,  |
| Text Normalization, Micro-Dialect Identification, Question Answering, Summarization                        |
| Speech: Acoustic Modeling, Speech Recognition, Text-to-Speech, Speech Synthesis                            |

Table 1: Categories in the annotation scheme.

In this paper, we present the first large-scale comprehensive survey on CSW NLP research in a structured manner by collecting more than 400 papers published on open repositories, such as the ACL Anthology and ISCA proceedings (see[2](#S2 "2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")). We manually coded these papers to collect coarse- and fine-grained information (see[2.1](#S2.SS1 "2.1 Annotation Process ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")) on CSW research in NLP that includes languages covered (see[3](#S3 "3 Language Diversity ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")), NLP tasks that have been explored, and new and emerging trends (see[4](#S4 "4 Tasks and Datasets ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")). In addition, motivated by the fact that fields like linguistics, socio-linguistics, and related fields, have studied CSW since the early 1900s, we also investigate to what extent theoretical frameworks from these fields have influenced NLP approaches (see[5](#S5 "5 From Linguistics to NLP ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")), and how the choice of methods has evolved over time (see[5.4](#S5.SS4 "5.4 Utilizing Neural Networks ‣ 5 From Linguistics to NLP ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")). Finally, we discuss the most pressing research challenges and identify a path forward to continue advancing this exciting line of work (see[6](#S6 "6 Recent Challenges and Future Direction ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")).

The area of NLP for CSW data is thriving, covering an increasing number of language combinations and tasks, and it is clearly advancing from a niche field to a common research topic, thus making our comprehensive survey timely. We expect the survey to provide valuable information to researchers new to the field and motivate more research from researchers already engaging in NLP for CSW data.

![Refer to caption](2212.09660v2/images/language-category.png)

Figure 2: Language Categories. \*NE denotes Non English. We show fine-grained categories in green and blue.

## 2 Exploring Open Proceedings

To develop a holistic understanding of the trends and advances in CSW NLP research, we collect research papers on CSW from the ACL Anthology and ISCA proceedings. We focus on these two sources because they encompass the top venues for publishing in speech and language processing in our field. In addition, we also look into personal repositories from researchers in the community that contains a curated list of CSW-related papers. We discuss below the search process for each venue.

#### ACL Anthology

We crawled the entire the ACL Anthology repository up to October 2022.11 1 [https://github.com/acl-org/acl-anthology](https://github.com/acl-org/acl-anthology "") We then filtered papers by using the following keywords related to CSW: “codeswitch", “code switch", “code-switching", “code-switched", “code-switch", “code-mix", “code-mixed", “code-mixing", “code mix" “mixed-language", “mixed-lingua", “mixed language", “mixed lingua", and “mix language".

#### ISCA Proceedings

We manually reviewed publicly available proceedings on the ISCA website22 2 [https://www.isca-speech.org](https://www.isca-speech.org "") and searched for papers related to CSW using the same set of keywords as above.

![Refer to caption](2212.09660v2/images/number_languages.png) 

![Refer to caption](2212.09660v2/images/number_languages_fine_grained.png) 

Figure 3: (Top): Number of publications across the type of language combination (bilingual, trilingual or 4+. (Bottom): Number of publications on fine-grained bilingual category with English as the L2 language.

![Refer to caption](2212.09660v2/images/number_languages_non_english.png)

Figure 4: Number of publications of bilingual code-switched languages that do not contain English. ∗msa stands for Modern Standard Arabic. The first two are the combination of a language with its dialect.

#### Web Resources

To extend the coverage of paper sources, we also gathered data from existing repositories.33 3 [https://github.com/gentaiscool/code-switching-papers](https://github.com/gentaiscool/code-switching-papers "")44 4 [https://genius1237.github.io/emnlp19\_tut/](https://genius1237.github.io/emnlp19_tut/ "") We can find multiple linguistics papers studying about CSW.

### 2.1 Annotation Process

We have three human annotators to annotate all collected papers based on multiple categories shown in Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges"). All papers are coded by a least one annotator. To extract the specific information we are looking for from the paper, the annotator needs to read through the paper, as most of the information is not contained in the abstract. The full list of the annotations we collected is available in the Appendix (see[A](#A1 "Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")).

To facilitate our analysis, we annotated the following aspects:

*   •

```
Languages: CSW is not restricted to pairs of languages; thus, we divide papers by the number of languages that are covered into bilingual, trilingual, and 4+ (if there are at least four languages). For a more fine-grained classification of languages, we categorize them by geographical location (see Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")).
```
*   •

```
Venues: There are multiple venues for CSW-related publications. We considered the following type of venues: conference, workshop, symposium and book. As we will discuss later, the publication venue is a reasonable data point of the wider spread of CSW research in recent years.
```
*   •

```
Papers: We classify the paper types based on their contribution and nature. We predict that we will have a high distribution of dataset/resource papers, as lack of resources has been a major bottleneck in the past.
```
*   •

```
Datasets: If the paper uses a dataset for the research, we will identify the source and modality (i.e., written text or speech) of the dataset.
```
*   •

```
Methods: We identify the type of methods presented in work.
```
*   •

```
Tasks: We identify the downstream NLP tasks (including the speech processing-related tasks) presented in work.
```
| Languages | \# Publications |     |    |
| --------- | --------------- | --- | -- |
| 111       | 17              | 128 | 30 |
| 78        | 8               | 86  | 40 |
| 20        | 27              | 47  | 5  |
| 37        | 2               | 39  | 17 |
| 23        | 2               | 25  | 13 |

Table 2: Most common code-switching languages in \*CL and ISCA venues. ‡The count does not include the dialect or South East Asian Mandarin - English [Lyu et al. (2010a)](#bib.bib233 "") since they contain more than two languages (i.e., it has words from the Chinese dialect).

## 3 Language Diversity

Here, we show the languages covered in the CSW resources. While focusing on the CSW phenomenon increases diversity of NLP technology, as we will see in this section, future efforts are needed to provide significant coverage of the most common CSW language combinations worldwide.

| Languages | \# Publications |    |
| --------- | --------------- | -- |
| 46        | 17              | 63 |
| 31        | 30              | 61 |
| 17        | 14              | 31 |
| 29        | 1               | 30 |
| 9         | 16              | 25 |
| 20        | 0               | 22 |
| 19        | 1               | 20 |
| 8         | 5               | 13 |

Table 3: Most common tasks in ACL venues. ST denotes shared task.

### 3.1 Variety of Mixed Languages

Figure [3](#S2.F3 "Figure 3 ‣ ISCA Proceedings ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the distribution of languages represented in the NLP for CSW literature. Most of the papers use datasets with two language pairs. However, we did find a few papers that address CSW scenarios with more than two languages. We consider this a relevant future direction in CSW: scaling model abilities to cover n languages, with n≥2n\\geq 2.

#### CSW in two languages

We group the number of publications focusing on bilingual CSW based on world regions in Figure [3](#S2.F3 "Figure 3 ‣ ISCA Proceedings ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") (bottom). We can see that the majority of research in CSW has focused on South Asian-English, especially on Hindi-English, Tamil-English, and Malayalam-English, as shown in Table [2](#S2.T2 "Table 2 ‣ 2.1 Annotation Process ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges"). The other common language pairs are Spanish-English and Chinese-English. That table also shows that many of the publications are shared task papers. This is probably reflecting efforts from a few research groups to motivate more research into CSW, such as that behind the CALCS workshop series.

Looking at the languages covered, we also find that there are many language pairs that come from different language families, such as Turkish-German [Çetinoğlu (2016)](#bib.bib75 ""); [Çetinoğlu and Çöltekin (2019)](#bib.bib76 ""); [Özateş and Çetinoğlu (2021)](#bib.bib273 ""); [Özateş et al. (2022)](#bib.bib272 ""), Turkish-Dutch [Gambäck and Das (2016)](#bib.bib131 ""), French-Arabic [Sankoff (1998)](#bib.bib324 ""); [Lounnas et al. (2021)](#bib.bib228 ""), Russian-Tatar [Taguchi et al. (2021)](#bib.bib376 ""), Russian-Kazakh [Mussakhojayeva et al. (2022a)](#bib.bib257 ""), Hindi-Tamil [Thomas et al. (2018b)](#bib.bib380 ""), Arabic-North African [El-Haj et al. (2018)](#bib.bib122 ""), Basque-Spanish [Aguirre et al. (2022)](#bib.bib18 ""), and Wixarika-Spanish [Mager et al. (2019)](#bib.bib238 ""). There are only very few papers working on Middle Eastern - English language pairs, most of the time, the Middle Eastern languages are mixed with non-English and/or dialects of these languages (see Figure [4](#S2.F4 "Figure 4 ‣ ISCA Proceedings ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")).

#### Trilingual

The number of papers addressing CSW in more than two languages is still small (see [3](#S2.F3 "Figure 3 ‣ ISCA Proceedings ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") top), compared to the papers looking at pairs of languages. Not surprisingly, this smaller number of papers focus on world regions where either the official languages are more than two, or these languages are widely used in the region, for example, Arabic-English-French [Abdul-Mageed et al. (2020)](#bib.bib1 ""), Hindi-Bengali-English [Barman et al. (2016)](#bib.bib51 ""), Tulu-Kannada-English [Hegde et al. (2022)](#bib.bib169 ""), and Darija-English-French [Voss et al. (2014)](#bib.bib390 "").

#### 4+

When looking at the papers that focus on more than three languages, we found that many papers use South East Asian Mandarin-English (SEAME) dataset [Lyu et al. (2010a)](#bib.bib233 ""), which has Chinese dialects and Malay or Indonesian words. Most of the other datasets are machine-generated using rule-based or neural methods.

### 3.2 Language-Dialect Code-Switching

Based on Figure [4](#S2.F4 "Figure 4 ‣ ISCA Proceedings ‣ 2 Exploring Open Proceedings ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges"), we can find some papers with language-dialect CSW, such as Chinese-Taiwanese Dialect [Chu et al. (2007)](#bib.bib101 ""); [Yu et al. (2012)](#bib.bib435 "") and Modern Standard Arabic (MSA)-Arabic Dialect [Elfardy and Diab (2012)](#bib.bib125 ""); [Samih and Maier (2016)](#bib.bib322 ""); [El-Haj et al. (2018)](#bib.bib122 ""). The dialect, in this case, is the variation of the language with a different form that is very specific to the region where the CSW style is spoken.

## 4 Tasks and Datasets

In this section, we summarize our findings, focusing on the CSW tasks and datasets. Table [3](#S3.T3 "Table 3 ‣ 3 Language Diversity ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the distribution of CSW tasks for ACL papers with at least ten publications. The two most popular tasks are language identification and sentiment analysis. Researchers mostly use the shared tasks from 2014 [Solorio et al. (2014)](#bib.bib349 "") and 2016 [Molina et al. (2016)](#bib.bib254 "") for language identification, and the SemEval 2020 shared task [Patwa et al. (2020)](#bib.bib282 "") for sentiment analysis. For ISCA, the most popular tasks are unsurprisingly ASR and TTS. This strong correlation between task and venue shows that the speech processing and \*CL communities remain somehow fragmented and working in isolation from one another, from the most part.

|    | \# Publications |    |
| -- | --------------- | -- |
| 38 | 4               | 42 |
| 54 | 18              | 72 |

Table 4: Publications that introduce new corpus.

| Source              | \*CL | ISCA | Total |
| ------------------- | ---- | ---- | ----- |
| Social Media        | 183  | 3    | 186   |
| Speech (Recording)  | 29   | 102  | 141   |
| Transcription       | 23   | 4    | 27    |
| News                | 19   | 5    | 24    |
| Dialogue            | 16   | 2    | 18    |
| Books               | 7    | 1    | 8     |
| Government Document | 6    | 0    | 6     |
| Treebank            | 5    | 0    | 5     |

Table 5: The source of the CSW dataset in the literature.

#### Public vs. Private Datasets

Public datasets availability also dictates what tasks are being explored in CSW research. Public datasets such as HinGE [Srivastava and Singh (2021b)](#bib.bib369 ""), SEAME [Lyu et al. (2010a)](#bib.bib233 "") and shared task datasets [Solorio et al. (2014)](#bib.bib349 ""); [Molina et al. (2016)](#bib.bib254 ""); [Aguilar et al. (2018)](#bib.bib14 ""); [Patwa et al. (2020)](#bib.bib282 "") have been widely used in many of the papers. Some work, however, used new datasets that are not publicly available, thus hindering adoption (see Table [4](#S4.T4 "Table 4 ‣ 4 Tasks and Datasets ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")). There are two well-known benchmarks in CSW: LinCE [Aguilar et al. (2020)](#bib.bib15 "") and GlueCOS [Khanuja et al. (2020b)](#bib.bib198 ""). These two benchmarks have a handful of tasks, and they are built to encourage transparency and reliability of evaluation since the test set labels are not publicly released. The evaluation is done automatically on their websites. However, their support languages are mostly limited to popular CSW language pairs, such as Spanish-English, Modern Standard Arabic-Egyptian, and Hindi-English, the exception being Nepali-English in LinCE.

#### Dataset Source

Table [5](#S4.T5 "Table 5 ‣ 4 Tasks and Datasets ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the statistics of dataset sources in the CSW literature. We found that most of the ACL papers were working on social media data. This is expected, considering that social media platforms are known to host informal interactions among users, making them reasonable places for users to code-switch. Naturally, most ISCA papers work on speech data, many of which are recordings of conversations and interviews. There are some datasets that come from speech transcription, news, dialogues, books, government documents, and treebanks.

#### Paper Category

Table [6](#S5.T6 "Table 6 ‣ 5 From Linguistics to NLP ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") presents the distribution of CSW papers. Most of the papers are empirical work focusing on the evaluation of downstream tasks. The second largest population is shared tasks. We also notice that many papers introduce new CSW corpus, but they are not released publicly. Some papers only release the URL or id to download the datasets, especially for datasets that come from social media (e.g., Twitter) since redistribution of the actual tweets is not allowed [Solorio et al. (2014)](#bib.bib349 ""); [Molina et al. (2016)](#bib.bib254 "") resulting in making reproducibility harder. Social media users can delete their posts at any point in time, resulting in considerable data attrition rates. There are very few papers working on the demos, theoretical work, position papers, and introducing evaluation metrics.

## 5 From Linguistics to NLP

Notably, papers are working on approaches that are inspired by linguistic theories to enhance the processing of CSW text. In this survey, we find three linguistic constraints that are used in the literature: equivalence constraint, matrix-embedded language Framework (MLF), and Functional Head Constraint. In this section, we will briefly introduce the constraints and list the papers that utilize the constraints.

| Type                    | \*CL | ISCA | Total |
| ----------------------- | ---- | ---- | ----- |
| Empirical               | 205  | 100  | 305   |
| Shared Task             | 82   | 1    | 83    |
| Corpus (Closed)         | 54   | 18   | 62    |
| Corpus (Open)           | 38   | 4    | 42    |
| Analysis                | 34   | 8    | 42    |
| Demo                    | 7    | 2    | 9     |
| Theoretical/Linguistic  | 7    | 0    | 7     |
| Position/Opinion/Survey | 3    | 0    | 3     |
| Metric                  | 2    | 1    | 3     |

Table 6: Paper Type of the CSW papers.

### 5.1 Linguistic-Driven Approaches

#### Equivalence Constraint

In a well-formed code-switched sentence, the switching takes place at those points where the grammatical constraints of both languages are satisfied [Poplack (1980)](#bib.bib290 ""). [Li and Fung (2012)](#bib.bib214 ""); [Li and Fung (2013)](#bib.bib215 "") incorporate this syntactic constraint to a statistical code-switch language model (LM) and evaluate the model on Chinese-English code-switched speech recognition. On the same line of work, [Pratapa et al. (2018a)](#bib.bib293 ""); [Pratapa and Choudhury (2021)](#bib.bib295 "") implement the same constraint to Hindi-English CSW data by producing parse trees of parallel sentences and matching the surface order of child nodes in the trees. [Winata et al. (2019c)](#bib.bib411 "") apply the constraint to generate synthetic CSW text and find that combining the real CSW data with synthetic CSW data can effectively improve the perplexity. They also treat parallel sentences as a linear structure and only allow switching on non-crossing alignments.

#### Matrix-Embedded Language Framework (MLF)

[Myers-Scotton (1997)](#bib.bib259 "") proposed that in bilingual CSW, there exists an asymmetrical relationship between the dominant matrix language and the subordinate embedded language. Matrix language provides the frame of the sentence by governing all or most of the most of the grammatical morphemes as well as word order, whereas syntactic elements that bear no or only limited grammatical function can be provided by the embedded language [Johanson (1999)](#bib.bib187 ""); [Myers-Scotton (2005)](#bib.bib260 ""). [Lee et al. (2019a)](#bib.bib210 "") use augmented parallel data by utilizing MLF to supplement the real code-switch data. [Gupta et al. (2020)](#bib.bib157 "") use MLF to automatically generate the code-mixed text from English to multiple languages without any parallel data.

#### Functional Head Constraint

[Belazi et al. (1994)](#bib.bib58 "") posit that it is impossible to switch languages between a functional head and its complement because of the strong relationship between the two constituents.  [Li and Fung (2014)](#bib.bib216 "") use the constraint of the LM by first expanding the search network with a translation model and then using parsing to restrict paths to those permissible under the constraint.

### 5.2 Learning from Data Distribution

Linguistic constraint theories have been used for decades to generate synthetic CSW sentences to address the lack of data issue. However, the approach requires external word alignments or constituency parsers that create erroneous results instead of applying the linguistic constraints to generate new synthetic CSW data, building a pointer-generator model to learn the real distribution of code-switched data [Winata et al. (2019c)](#bib.bib411 ""). [Chang et al. (2019)](#bib.bib92 "") propose to generate CSW sentences from monolingual sentences using Generative Adversarial Network (GAN) [Goodfellow et al. (2020)](#bib.bib146 "") and the generator learns to predict CSW points without any linguistic knowledge.

### 5.3 The Era of Statistical Methods

The research on CSW is also influenced by the progress and development of machine learning. According to Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Utilizing Neural Networks ‣ 5 From Linguistics to NLP ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges"), starting in 2006, statistical methods have been adapted to CSW research, while before that year, the approaches were mainly rule-based. There are common statistical methods for text classification used in the literature, such as Naive Bayes [Solorio and Liu (2008a)](#bib.bib350 "") and Support Vector Machine (SVM) [Solorio and Liu (2008b)](#bib.bib351 ""). Conditional Random Field (CRF) [Sutton et al. (2012)](#bib.bib373 "") is also widely seen in the literature for sequence labeling, such as Part-of-Speech (POS) tagging [Vyas et al. (2014)](#bib.bib392 ""), Named Entity Recognition (NER), and word-level language identification [Lin et al. (2014)](#bib.bib223 ""); [Chittaranjan et al. (2014)](#bib.bib96 ""); [Jain and Bhat (2014)](#bib.bib180 ""). HMM-based models have been used in speech-related tasks, such as speech recognition [Weiner et al. (2012a)](#bib.bib399 ""); [Li and Fung (2013)](#bib.bib215 "") and text synthesis [Qian et al. (2008)](#bib.bib298 ""); [Shuang et al. (2010)](#bib.bib339 ""); [He et al. (2012)](#bib.bib168 "").

### 5.4 Utilizing Neural Networks

(a) \*CL

(b) ISCA

Figure 5: Methods used for code-mixing NLP.

Following general NLP trends, we see the adoption of neural methods and pre-trained models growing in popularity over time. In contrast, the statistical and rule-based approaches are diminishing. Compared to ISCA, we see more adaptation of the pre-training model. This is because ACL work is more text-based focused, where pre-trained LMs are more widely available.

#### Neural-Based Models

Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Utilizing Neural Networks ‣ 5 From Linguistics to NLP ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows that the trend of using neural-based models started in 2013, and the usage of rule/linguistic constraint and statistical methods diminished gradually through time, but they are still used even with a low percentage. RNN and LSTM architectures are commonly used in sequence modeling, such as language modeling [Adel et al. (2013)](#bib.bib5 ""); [Vu and Schultz (2014)](#bib.bib391 ""); [Adel et al. (2014c)](#bib.bib4 ""); [Winata et al. (2018a)](#bib.bib410 ""); [Garg et al. (2018a)](#bib.bib136 ""); [Winata et al. (2019c)](#bib.bib411 "") and CSW identification [Samih et al. (2016a)](#bib.bib321 ""). DNN-based and hybrid HMM-DNN models are used in speech recognition models [Yilmaz et al. (2018)](#bib.bib427 ""); [Yılmaz et al. (2018)](#bib.bib429 "").

#### Pre-trained Embeddings

Pre-trained embeddings are used to complement neural-based approaches by initializing the embedding layer. Common pre-trained embeddings used in the literature are monolingual subword-based embeddings, FastText [Joulin et al. (2016)](#bib.bib192 ""), and aligned-embeddings MUSE [Conneau et al. (2017)](#bib.bib103 ""). A standard method to utilize monolingual embeddings is to concatenate or sum two or more embeddings from different languages [Trivedi et al. (2018)](#bib.bib382 ""). A more recent approach is to apply an attention mechanism to merge embeddings and form meta-embeddings [Winata et al. (2019a)](#bib.bib407 ""); [Winata et al. (2019b)](#bib.bib408 ""). Character-based embeddings have also been explored in the literature to address the out-of-vocabulary issues on word-embeddings [Winata et al. (2018b)](#bib.bib412 ""); [Attia et al. (2018)](#bib.bib33 ""); [Aguilar et al. (2021)](#bib.bib16 ""). Another approach is to train bilingual embeddings using real and synthetic CSW data [Pratapa et al. (2018b)](#bib.bib296 ""). In the speech domain,  [Lovenia et al. (2022)](#bib.bib229 "") utilize wav2vec 2.0 [Baevski et al. (2020)](#bib.bib35 "") as a starting model before fine-tuning.

#### Language Models

Many pre-trained model approaches utilize multilingual LMs, such as mBERT or XLM-R to deal with CSW data [Khanuja et al. (2020b)](#bib.bib198 ""); [Aguilar and Solorio (2020)](#bib.bib17 ""); [Pant and Dadu (2020)](#bib.bib277 ""); [Patwa et al. (2020)](#bib.bib282 ""); [Winata et al. (2021a)](#bib.bib406 ""). These models are often fine-tuned with the downstream task or with CSW text to better adapt to the languages. Some downstream fine-tuning approaches use synthetic CSW data due to a lack of available datasets.  [Aguilar et al. (2021)](#bib.bib16 "") propose a character-based subword module (char2subword) of the mBERT that learns the subword embedding that is suitable for modeling the noisy CSW text. [Winata et al. (2021a)](#bib.bib406 "") compare the performance of the multilingual LM versus the language-specific LM for CSW context. While XLM-R provides the best result, it is also computationally heavy. There needed to be more exploration of larger models.

We see that pre-trained LMs provide better empirical results on current benchmark tasks and enables an end-to-end approach. Therefore, one can theoretically work on CSW tasks without any linguistic understanding of the language, assuming the dataset for model finetuning is available. However, the downside is that there is little understanding of how and when the LMs would fail, thus we encourage more interpretability work on these LMs in CSW setting.

## 6 Recent Challenges and Future Direction

### 6.1 More Diverse Exploration on Code-Switching Styles and Languages

A handful of languages, such as Spanish-English, Hindi-English, or Chinese-English, dominate research and resource CSW. However, there are still many countries and cultures rich in the use of CSW, which is still under-represented in NLP research [Joshi et al. (2020)](#bib.bib191 ""); [Aji et al. (2022)](#bib.bib20 ""); [Yong et al. (2023)](#bib.bib432 ""), especially on different CSW variations. CSW style can vary in different regions of the world, and it would be interesting to gather more datasets on unexplored and unknown styles, which can be useful for further research and investigation on linguistics and NLP. Therefore, one future direction is to broaden the language scope of CSW research.

### 6.2 Datasets: Access and Sources

According to our findings, there are more than 60% of the datasets are private (see Table [4](#S4.T4 "Table 4 ‣ 4 Tasks and Datasets ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges")), and they are not released to the public. This eventually hampers the progress of CSW research, particularly in the results’ reproducibility, credibility, and transparency. Moreover, many studies in the literature do not release the code to reproduce their work. Therefore, we encourage researchers who build a new corpus to release the datasets publicly. In addition, the fact that some researchers provide urls to download the data is also problematic due to the data attrition issue we raised earlier. Data attrition is bad for reproducibility, but it is also a waste of annotation efforts. Perhaps we should work on identifying alternative means to collect written CSW data in an ecologically valid manner.

### 6.3 Model Scaling

To the best of our knowledge, little work has been done on investigating how well the scaling law holds for code-mixed datasets. [Winata et al. (2021a)](#bib.bib406 "") demonstrate that the XLM-R-large model outperforms smaller pre-trained models on the NER and POS tasks in LinCE benchmark [Aguilar et al. (2020)](#bib.bib15 ""); however, the largest model in the study, which is the XLM-R-large model, only has 355 million parameters. Furthermore, they find that smaller models that combine word, subword, and character embeddings achieve comparable performance as mBERT while being faster in inference. Given the recent release of billion-sized large pre-trained multilingual models such as XGLM and BLOOM [Scao et al. (2022)](#bib.bib327 ""), we urge future research to study the scaling law and performance-compute trade-off in code-mixing tasks.

### 6.4 Zero-Shot and Few-Shot Exploration

The majority of pre-trained model approaches fine-tune their models to the downstream task. On the other hand, CSW data is considerably limited. With the rise of multilingual LMs, especially those that have been fine-tuned with prompt/instruction [Muennighoff et al. (2022)](#bib.bib255 ""); [Ouyang et al. (2022)](#bib.bib271 ""); [Winata et al. (2022)](#bib.bib404 ""), one direction is to see whether these LMs can handle CSW input in a zero-shot fashion. This work might also tie in with model scaling since larger models have shown better capability at zero-shot and few-shot settings [Winata et al. (2021b)](#bib.bib409 ""); [Srivastava et al. (2022)](#bib.bib363 "").

### 6.5 Robustness Evaluation

Since CSW is a widely common linguistic phenomenon, we argue that cross-lingual NLP benchmarks, such as XGLUE [Liang et al. (2020)](#bib.bib221 "") and XTREME-R [Ruder et al. (2021)](#bib.bib319 ""), should incorporate linguistic CSW evaluation [Aguilar et al. (2020)](#bib.bib15 ""); [Khanuja et al. (2020b)](#bib.bib198 ""). The reasons are that CSW is a cognitive ability that multilingual human speakers can perform with ease [Beatty-Martínez et al. (2020)](#bib.bib56 ""). CSW evaluation examines the robustness of multilingual LMs in learning cross-lingual alignment of representations [Conneau et al. (2020)](#bib.bib104 ""); [Libovickỳ et al. (2020)](#bib.bib222 ""); [Pires et al. (2019)](#bib.bib288 ""); [Adilazuarda et al. (2022)](#bib.bib6 ""). On the other hand, catastrophic forgetting is observed in pre-trained models [Shah et al. (2020)](#bib.bib329 ""), and human speakers ([Hirvonen and Lauttamus, 2000](#bib.bib171 ""); [Du Bois, 2009](#bib.bib119 ""), known as language attrition) in a CSW environment. We argue that finetuning LMs on code-mixed data is a form of continual learning to produce a more generalized multilingual LM. Thus, we encourage CSW research to report the performance of finetuned models on both CSW and monolingual texts.

### 6.6 Task Diversity

We encourage creating reasoning-based tasks for CSW texts for two reasons. First, code-mixed datasets for tasks such as NLI, coreference resolution, and question-answering are much fewer in comparison to tasks such as sentiment analysis, parts-of-speech tagging, and named-entity recognition. Second, comprehension tasks with the CSW text present more processing costs for human readers [Bosma and Pablos (2020)](#bib.bib67 "").

### 6.7 Conversational Agents

There has been a recent focus on developing conversational agents with LMs such as ChatGPT55 5 [https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/ ""), Whisper [Radford et al. (2022)](#bib.bib300 ""), SLAM [Bapna et al. (2021)](#bib.bib47 ""), mSLAM [Bapna et al. (2022)](#bib.bib46 ""). We recommend incorporating the capability of synthesizing code-mixed data in human-machine dialogue, as CS is a prevalent communication style among multilingual speakers [Ahn et al. (2020)](#bib.bib19 ""), and humans prefer chatbots with such capability [Bawa et al. (2020)](#bib.bib55 "").

### 6.8 Automatic Evaluation for Generation

With the rise of pre-trained models, generative tasks gained more popularity. However, when generating CSW data, most work used human evaluation for measuring quality of the generated data. Alternative automatic methods for CS text based on word-frequency and temporal distribution are commonly used [Guzmán et al. (2017)](#bib.bib161 ""); [Mave et al. (2018)](#bib.bib250 ""), but we believe there is still much room for improvement in this respect. One possible future direction is to align the evaluation metrics to human judgement of quality [Hamed et al. (2022)](#bib.bib165 "") where we can assess separately the “faithfulness” of the resulting CSW data from other desired properties of language generation. Other nuances here are related to the intricacy of CSW patterns, where ideally the model would mimic the CSW style of the intended users.

## 7 Conclusion

We present a comprehensive systematic survey on code-switching research in natural language processing to explore the progress of the past decades and understand the existing challenges and tasks in the literature. We summarize the trends and findings and conclude with a discussion for future direction and open questions for further investigation. We hope this survey can encourage and lead NLP researchers in a better direction on code-switching research.

## Limitations

The numbers in this survey are limited to papers published in the ACL Anthology and ISCA Proceedings. However, we also included papers as related work from other resources if they are publicly available and accessible. In addition, the category in the survey does not include the code-switching type (i.e., intra-sentential, inter-sentential, etc.) since some papers do not provide such information.

## Ethics Statement

We use publicly available data in our survey with permissive licenses. No potential ethical issues in this work.

## Acknowledgements

Thanks to Igor Malioutov for the insightful discussion on the paper.

## References

*   Abdul-Mageed et al. (2020) Muhammad Abdul-Mageed, Chiyu Zhang, AbdelRahim Elmadany, and Lyle Ungar. 2020. Toward micro-dialect identification in diaglossic and code-switched environments. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5855–5876.
*   Adel et al. (2014a) Heike Adel, Katrin Kirchhoff, Dominic Telaar, Ngoc Thang Vu, Tim Schlippe, and Tanja Schultz. 2014a. Features for factored language models for code-switching speech. In *Proc. 4th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2014)*, pages 32–38.
*   Adel et al. (2014b) Heike Adel, Katrin Kirchhoff, Ngoc Thang Vu, Dominic Telaar, and Tanja Schultz. 2014b. [Comparing approaches to convert recurrent neural networks into backoff language models for efficient decoding](https://doi.org/10.21437/Interspeech.2014-165 ""). In *Proc. Interspeech 2014*, pages 651–655.
*   Adel et al. (2014c) Heike Adel, Dominic Telaar, Ngoc Thang Vu, Katrin Kirchhoff, and Tanja Schultz. 2014c. Combining recurrent neural networks and factored language models during decoding of code-switching speech. In *Fifteenth Annual Conference of the International Speech Communication Association*.
*   Adel et al. (2013) Heike Adel, Ngoc Thang Vu, and Tanja Schultz. 2013. Combination of recurrent neural networks and factored language models for code-switching language modeling. In *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 206–211.
*   Adilazuarda et al. (2022) Muhammad Farid Adilazuarda, Samuel Cahyawijaya, Genta Indra Winata, Pascale Fung, and Ayu Purwarianti. 2022. Indorobusta: Towards robustness against diverse code-mixed indonesian local languages. In *Proceedings of the First Workshop on Scaling Up Multilingual Evaluation*, pages 25–34.
*   Adouane and Bernardy (2020) Wafia Adouane and Jean-Philippe Bernardy. 2020. When is multi-task learning beneficial for low-resource noisy code-switched user-generated algerian texts? In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 17–25.
*   Adouane et al. (2018) Wafia Adouane, Jean-Philippe Bernardy, and Simon Dobnik. 2018. Improving neural network performance by injecting background knowledge: Detecting code-switching and borrowing in algerian texts. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 20–28.
*   Adouane et al. (2020) Wafia Adouane, Samia Touileb, and Jean-Philippe Bernardy. 2020. Identifying sentiments in algerian code-switched user-generated comments. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 2698–2705.
*   Advani et al. (2020) Laksh Advani, Clement Lu, and Suraj Maharjan. 2020. C1 at semeval-2020 task 9: Sentimix: Sentiment analysis for code-mixed social media text using feature engineering. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1227–1232.
*   Agarwal and Narula (2021) Kaustubh Agarwal and Rhythm Narula. 2021. Humor generation and detection in code-mixed hindi-english. In *Proceedings of the Student Research Workshop Associated with RANLP 2021*, pages 1–6.
*   Agarwal et al. (2021) Vibhav Agarwal, Pooja Rao, and Dinesh Babu Jayagopi. 2021. Towards code-mixed hinglish dialogue generation. In *Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI*, pages 271–280.
*   Aggarwal et al. (2020) Akshita Aggarwal, Anshul Wadhawan, Anshima Chaudhary, and Kavita Maurya. 2020. “did you really mean what you said?”: Sarcasm detection in hindi-english code-mixed data using bilingual word embeddings. In *Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)*, pages 7–15.
*   Aguilar et al. (2018) Gustavo Aguilar, Fahad AlGhamdi, Victor Soto, Mona Diab, Julia Hirschberg, and Thamar Solorio. 2018. Named entity recognition on code-switched data: Overview of the calcs 2018 shared task. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 138–147.
*   Aguilar et al. (2020) Gustavo Aguilar, Sudipta Kar, and Thamar Solorio. 2020. Lince: A centralized benchmark for linguistic code-switching evaluation. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 1803–1813.
*   Aguilar et al. (2021) Gustavo Aguilar, Bryan McCann, Tong Niu, Nazneen Rajani, Nitish Shirish Keskar, and Thamar Solorio. 2021. Char2subword: Extending the subword embedding space using robust character compositionality. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 1640–1651.
*   Aguilar and Solorio (2020) Gustavo Aguilar and Thamar Solorio. 2020. From english to code-switching: Transfer learning with strong morphological clues. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8033–8044.
*   Aguirre et al. (2022) Maia Aguirre, Laura García-Sardiña, Manex Serras, Ariane Méndez, and Jacobo López. 2022. Basco: An annotated basque-spanish code-switching corpus for natural language understanding. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 3158–3163.
*   Ahn et al. (2020) Emily Ahn, Cecilia Jimenez, Yulia Tsvetkov, and Alan W Black. 2020. What code-switching strategies are effective in dialog systems? In *Proceedings of the Society for Computation in Linguistics 2020*, pages 254–264.
*   Aji et al. (2022) Alham Aji, Genta Indra Winata, Fajri Koto, Samuel Cahyawijaya, Ade Romadhony, Rahmad Mahendra, Kemal Kurniawan, David Moeljadi, Radityo Eko Prasojo, Timothy Baldwin, et al. 2022. One country, 700+ languages: Nlp challenges for underrepresented languages and dialects in indonesia. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 7226–7249.
*   Al-Badrashiny and Diab (2016) Mohamed Al-Badrashiny and Mona Diab. 2016. The george washington university system for the code-switching workshop shared task 2016. In *Proceedings of The Second Workshop on Computational Approaches to Code Switching*, pages 108–111.
*   AlGhamdi et al. (2016) Fahad AlGhamdi, Giovanni Molina, Mona Diab, Thamar Solorio, Abdelati Hawwari, Victor Soto, and Julia Hirschberg. 2016. Part of speech tagging for code switched data. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 98–107.
*   Ali et al. (2021) Ahmed Ali, Shammur Absar Chowdhury, Amir Hussein, and Yasser Hifny. 2021. [Arabic code-switching speech recognition using monolingual data](https://doi.org/10.21437/Interspeech.2021-2231 ""). In *Proc. Interspeech 2021*, pages 3475–3479.
*   Amazouz et al. (2017) Djegdjiga Amazouz, Martine Adda-Decker, and Lori Lamel. 2017. [Addressing code-switching in french/algerian arabic speech](https://doi.org/10.21437/Interspeech.2017-1373 ""). In *Proc. Interspeech 2017*, pages 62–66.
*   Amin et al. (2022) Saadullah Amin, Noon Pokaratsiri Goldstein, Morgan Wixted, Alejandro Garcia-Rudolph, Catalina Martínez-Costa, and Günter Neumann. 2022. Few-shot cross-lingual transfer for coarse-grained de-identification of code-mixed clinical texts. In *Proceedings of the 21st Workshop on Biomedical Language Processing*, pages 200–211.
*   Andrew (2021) Judith Jeyafreeda Andrew. 2021. Judithjeyafreedaandrew@ dravidianlangtech-eacl2021: offensive language detection for dravidian code-mixed youtube comments. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 169–174.
*   Angel et al. (2020) Jason Angel, Segun Taofeek Aroyehun, Antonio Tamayo, and Alexander Gelbukh. 2020. Nlp-cic at semeval-2020 task 9: Analysing sentiment in code-switching language using a simple deep-learning classifier. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 957–962.
*   Antony et al. (2022) Ansen Antony, Sumanth Reddy Kota, Akhilesh Lade, Spoorthy V, and Shashidhar G. Koolagudi. 2022. [An improved transformer transducer architecture for hindi-english code switched speech recognition](https://doi.org/10.21437/Interspeech.2022-10763 ""). In *Proc. Interspeech 2022*, pages 3123–3127.
*   Aparaschivei et al. (2020) Lavinia Aparaschivei, Andrei Palihovici, and Daniela Gîfu. 2020. Fii-uaic at semeval-2020 task 9: Sentiment analysis for code-mixed social media text using cnn. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 928–933.
*   Appicharla et al. (2021) Ramakrishna Appicharla, Kamal Kumar Gupta, Asif Ekbal, and Pushpak Bhattacharyya. 2021. Iitp-mt at calcs2021: English to hinglish neural machine translation using unsupervised synthetic code-mixed parallel corpus. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 31–35.
*   Arianto and Budi (2020) Dian Arianto and Indra Budi. 2020. Aspect-based sentiment analysis on indonesia’s tourism destinations based on google maps user code-mixed reviews (study case: Borobudur and prambanan temples). In *Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation*, pages 359–367.
*   Asnani and Pawar (2016) Kavita Asnani and Jyoti Pawar. 2016. Use of semantic knowledge base for enhancement of coherence of code-mixed topic-based aspect clusters. In *Proceedings of the 13th International Conference on Natural Language Processing*, pages 259–266.
*   Attia et al. (2018) Mohammed Attia, Younes Samih, and Wolfgang Maier. 2018. Ghht at calcs 2018: Named entity recognition for dialectal arabic using neural networks. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 98–102.
*   Badino et al. (2004) Leonardo Badino, Claudia Barolo, and Silvia Quazza. 2004. [A general approach to tts reading of mixed-language texts](https://doi.org/10.21437/Interspeech.2004-312 ""). In *Proc. Interspeech 2004*, pages 849–852.
*   Baevski et al. (2020) Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems*, 33:12449–12460.
*   Balabel et al. (2020) Mohamed Balabel, Injy Hamed, Slim Abdennadher, Ngoc Thang Vu, and Özlem Çetinoğlu. 2020. Cairo student code-switch (cscs) corpus: An annotated egyptian arabic-english corpus. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 3973–3977.
*   Bali et al. (2014) Kalika Bali, Jatin Sharma, Monojit Choudhury, and Yogarshi Vyas. 2014. “i am borrowing ya mixing?" an analysis of english-hindi code mixing in facebook. In *Proceedings of the first workshop on computational approaches to code switching*, pages 116–126.
*   Ball and Garrette (2018) Kelsey Ball and Dan Garrette. 2018. Part-of-speech tagging for code-switched, transliterated texts without explicit language identification. In *Association for Computational Linguistics*.
*   Balouchzahi et al. (2021) Fazlourrahman Balouchzahi, BK Aparna, and HL Shashirekha. 2021. Mucs@ dravidianlangtech-eacl2021: Cooli-code-mixing offensive language identification. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 323–329.
*   Balouchzahi and Shashirekha (2021) Fazlourrahman Balouchzahi and HL Shashirekha. 2021. La-saco: A study of learning approaches for sentiments analysis incode-mixing texts. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 109–118.
*   Banerjee et al. (2020) Somnath Banerjee, Sahar Ghannay, Sophie Rosset, Anne Vilnat, and Paolo Rosso. 2020. Limsi\_upv at semeval-2020 task 9: Recurrent convolutional neural network for code-mixed sentiment analysis. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1281–1287.
*   Banerjee et al. (2018) Suman Banerjee, Nikita Moghe, Siddhartha Arora, and Mitesh M Khapra. 2018. A dataset for building code-mixed goal oriented conversation systems. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 3766–3780.
*   Bansal et al. (2020a) Neetika Bansal, Vishal Goyal, and Simpel Rani. 2020a. Language identification and normalization of code mixed english and punjabi text. In *Proceedings of the 17th International Conference on Natural Language Processing (ICON): System Demonstrations*, pages 30–31.
*   Bansal et al. (2020b) Shubham Bansal, Arijit Mukherjee, Sandeepkumar Satpal, and Rupeshkumar Mehta. 2020b. [On improving code mixed speech synthesis with mixlingual grapheme-to-phoneme model](https://doi.org/10.21437/Interspeech.2020-2654 ""). In *Proc. Interspeech 2020*, pages 2957–2961.
*   Bansal et al. (2020c) Srijan Bansal, Vishal Garimella, Ayush Suhane, Jasabanta Patro, and Animesh Mukherjee. 2020c. Code-switching patterns can be an effective route to improve performance of downstream nlp applications: A case study of humour, sarcasm and hate speech detection. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 1018–1023.
*   Bapna et al. (2022) Ankur Bapna, Colin Cherry, Yu Zhang, Ye Jia, Melvin Johnson, Yong Cheng, Simran Khanuja, Jason Riesa, and Alexis Conneau. 2022. mslam: Massively multilingual joint pre-training for speech and text. *arXiv preprint arXiv:2202.01374*.
*   Bapna et al. (2021) Ankur Bapna, Yu-an Chung, Nan Wu, Anmol Gulati, Ye Jia, Jonathan H Clark, Melvin Johnson, Jason Riesa, Alexis Conneau, and Yu Zhang. 2021. Slam: A unified encoder for speech and language modeling via speech-text joint pre-training. *arXiv preprint arXiv:2110.10329*.
*   Bar and Dershowitz (2014) Kfir Bar and Nachum Dershowitz. 2014. The tel aviv university system for the code-switching workshop shared task. In *Proceedings of the first workshop on computational approaches to code switching*, pages 139–143.
*   Barman et al. (2014a) Utsab Barman, Amitava Das, Joachim Wagner, and Jennifer Foster. 2014a. Code mixing: A challenge for language identification in the language of social media. In *Proceedings of the first workshop on computational approaches to code switching*, pages 13–23.
*   Barman et al. (2014b) Utsab Barman, Joachim Wagner, Grzegorz Chrupała, and Jennifer Foster. 2014b. Dcu-uvt: Word-level language classification with code-mixed data. In *Proceedings of the first workshop on computational approaches to code switching*, pages 127–132.
*   Barman et al. (2016) Utsab Barman, Joachim Wagner, and Jennifer Foster. 2016. Part-of-speech tagging of code-mixed social media content: Pipeline, stacking and joint modelling. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 30–39.
*   Baroi et al. (2020) Subhra Jyoti Baroi, Nivedita Singh, Ringki Das, and Thoudam Doren Singh. 2020. Nits-hinglish-sentimix at semeval-2020 task 9: Sentiment analysis for code-mixed social media text using an ensemble model. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1298–1303.
*   Baruah et al. (2020) Arup Baruah, Kaushik Das, Ferdous Barbhuiya, and Kuntal Dey. 2020. Iiitg-adbu at semeval-2020 task 12: Comparison of bert and bilstm in detecting offensive language. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1562–1568.
*   Bawa et al. (2018) Anshul Bawa, Monojit Choudhury, and Kalika Bali. 2018. Accommodation of conversational code-choice. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 82–91.
*   Bawa et al. (2020) Anshul Bawa, Pranav Khadpe, Pratik Joshi, Kalika Bali, and Monojit Choudhury. 2020. Do multilingual users prefer chat-bots that code-mix? let’s nudge and find out! *Proceedings of the ACM on Human-Computer Interaction*, 4(CSCW1):1–23.
*   Beatty-Martínez et al. (2020) Anne L Beatty-Martínez, Christian A Navarro-Torres, and Paola E Dussias. 2020. Codeswitching: A bilingual toolkit for opportunistic speech planning. *Frontiers in Psychology*, page 1699.
*   Begum et al. (2016) Rafiya Begum, Kalika Bali, Monojit Choudhury, Koustav Rudra, and Niloy Ganguly. 2016. Functions of code-switching in tweets: An annotation framework and some initial experiments. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 1644–1650.
*   Belazi et al. (1994) Hedi M Belazi, Edward J Rubin, and Almeida Jacqueline Toribio. 1994. Code switching and x-bar theory: The functional head constraint. *Linguistic inquiry*, pages 221–237.
*   Bharathi et al. (2021) B Bharathi et al. 2021. Ssncse\_nlp@ dravidianlangtech-eacl2021: Offensive language identification on multilingual code mixing text. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 313–318.
*   Bhat et al. (2017) Irshad Bhat, Riyaz Ahmad Bhat, Manish Shrivastava, and Dipti Misra Sharma. 2017. Joining hands: Exploiting monolingual treebanks for parsing of code-mixing data. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*, pages 324–330.
*   Bhattacharja (2010) Shishir Bhattacharja. 2010. Benglish verbs: A case of code-mixing in bengali. In *Proceedings of the 24th Pacific Asia Conference on Language, Information and Computation*, pages 75–84.
*   Biradar and Saumya (2022) Shankar Biradar and Sunil Saumya. 2022. Iiitdwd@ tamilnlp-acl2022: Transformer-based approach to classify abusive content in dravidian code-mixed text. In *Proceedings of the second workshop on speech and language technologies for Dravidian languages*, pages 100–104.
*   Biswas et al. (2020) Astik Biswas, Febe De Wet, Thomas Niesler, et al. 2020. Semi-supervised acoustic and language model training for english-isizulu code-switched speech recognition. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 52–56.
*   Biswas et al. (2018a) Astik Biswas, Febe de Wet, Ewald van der Westhuizen, Emre Yilmaz, and Thomas Niesler. 2018a. Multilingual neural network acoustic modelling for asr of under-resourced english-isizulu code-switched speech. In *INTERSPEECH*, pages 2603–2607.
*   Biswas et al. (2018b) Astik Biswas, Ewald van der Westhuizen, Thomas Niesler, and Febe de Wet. 2018b. Improving asr for code-switched speech in under-resourced languages using out-of-domain data. In *SLTU*, pages 122–126.
*   Biswas et al. (2019) Astik Biswas, Emre Yılmaz, Febe de Wet, Ewald van der Westhuizen, and Thomas Niesler. 2019. Semi-supervised acoustic model training for five-lingual code-switched asr. *Proc. Interspeech 2019*, pages 3745–3749.
*   Bosma and Pablos (2020) Evelyn Bosma and Leticia Pablos. 2020. Switching direction modulates the engagement of cognitive control in bilingual reading comprehension: An erp study. *Journal of Neurolinguistics*, 55:100894.
*   Braggaar and van der Goot (2021) Anouck Braggaar and Rob van der Goot. 2021. Challenges in annotating and parsing spoken, code-switched, frisian-dutch data. In *Proceedings of the Second Workshop on Domain Adaptation for NLP*, pages 50–58.
*   Bullock et al. (2018a) Barbara Bullock, Wally Guzmán, Jacqueline Serigos, Vivek Sharath, and Almeida Jacqueline Toribio. 2018a. Predicting the presence of a matrix language in code-switching. In *Proceedings of the third workshop on computational approaches to linguistic code-switching*, pages 68–75.
*   Bullock et al. (2018b) Barbara E. Bullock, Gualberto Guzmán, Jacqueline Serigos, and Almeida Jacqueline Toribio. 2018b. [Should code-switching models be asymmetric?](https://doi.org/10.21437/Interspeech.2018-1284 "") In *Proc. Interspeech 2018*, pages 2534–2538.
*   Calvillo et al. (2020) Jesús Calvillo, Le Fang, Jeremy Cole, and David Reitter. 2020. Surprisal predicts code-switching in chinese-english bilingual text. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4029–4039.
*   Cameron (2020) Marguerite Cameron. 2020. Voice onset time en code-switching anglais-français: une étude des occlusives sourdes en début de mot (voice onset time in english-french code-switching: a study of word-initial voiceless stop consonants). In *Actes de la 6e conférence conjointe Journées d’Études sur la Parole (JEP, 33e édition), Traitement Automatique des Langues Naturelles (TALN, 27e édition), Rencontre des Étudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (RÉCITAL, 22e édition). Volume 1: Journées d’Études sur la Parole*, pages 54–63.
*   Cao et al. (2009) Houwei Cao, P. C. Ching, and Tan Lee. 2009. [Effects of language mixing for automatic recognition of cantonese-english code-mixing utterances](https://doi.org/10.21437/Interspeech.2009-762 ""). In *Proc. Interspeech 2009*, pages 3011–3014.
*   Carpuat (2014) Marine Carpuat. 2014. Mixed language and code-switching in the canadian hansard. In *Proceedings of the first workshop on computational approaches to code switching*, pages 107–115.
*   Çetinoğlu (2016) Özlem Çetinoğlu. 2016. A turkish-german code-switching corpus. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 4215–4220.
*   Çetinoğlu and Çöltekin (2019) Özlem Çetinoğlu and Çağrı Çöltekin. 2019. Challenges of annotating a code-switching treebank. In *Proceedings of the 18th international workshop on Treebanks and Linguistic Theories (TLT, SyntaxFest 2019)*, pages 82–90.
*   Çetinoğlu et al. (2016) Özlem Çetinoğlu, Sarah Schulz, and Ngoc Thang Vu. 2016. Challenges of computational processing of code-switching. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 1–11.
*   Chakravarthi (2020) Bharathi Raja Chakravarthi. 2020. Hopeedi: A multilingual hope speech detection dataset for equality, diversity, and inclusion. In *Proceedings of the Third Workshop on Computational Modeling of People’s Opinions, Personality, and Emotion’s in Social Media*, pages 41–53.
*   Chakravarthi et al. (2020a) Bharathi Raja Chakravarthi, Navya Jose, Shardul Suryawanshi, Elizabeth Sherly, and John Philip McCrae. 2020a. A sentiment analysis dataset for code-mixed malayalam-english. In *Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)*, pages 177–184.
*   Chakravarthi et al. (2020b) Bharathi Raja Chakravarthi, Vigneshwaran Muralidaran, Ruba Priyadharshini, and John Philip McCrae. 2020b. Corpus creation for sentiment analysis in code-mixed tamil-english text. In *Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)*, pages 202–210.
*   Chakravarthy et al. (2020) Sharanya Chakravarthy, Anjana Umapathy, and Alan W Black. 2020. Detecting entailment in code-mixed hindi-english conversations. In *Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)*, pages 165–170.
*   Chan et al. (2009) Joyce YC Chan, Houwei Cao, PC Ching, and Tan Lee. 2009. Automatic recognition of cantonese-english code-mixing speech. In *International Journal of Computational Linguistics & Chinese Language Processing, Volume 14, Number 3, September 2009*.
*   Chan et al. (2005) Joyce YC Chan, PC Ching, and Tan Lee. 2005. Development of a cantonese-english code-mixing speech corpus. In *Ninth European conference on speech communication and technology*.
*   Chan et al. (2006) Joyce YC Chan, PC Ching, Tan Lee, and Houwei Cao. 2006. Automatic speech recognition of cantonese-english code-mixing utterances. In *Ninth International Conference on Spoken Language Processing*.
*   Chan et al. (2004) Joyce YC Chan, PC Ching, Tan Lee, and Helen M Meng. 2004. Detection of language boundary in code-switching utterances by bi-phone probabilities. In *2004 International Symposium on Chinese Spoken Language Processing*, pages 293–296. IEEE.
*   Chanda et al. (2016a) Arunavha Chanda, Dipankar Das, and Chandan Mazumdar. 2016a. Columbia-jadavpur submission for emnlp 2016 code-switching workshop shared task: System description. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 112–115.
*   Chanda et al. (2016b) Arunavha Chanda, Dipankar Das, and Chandan Mazumdar. 2016b. Unraveling the english-bengali code-mixing phenomenon. In *Proceedings of the second workshop on computational approaches to code switching*, pages 80–89.
*   Chandu et al. (2019) Khyathi Chandu, Ekaterina Loginova, Vishal Gupta, Josef van Genabith, Günter Neumann, Manoj Chinnakotla, Eric Nyberg, and Alan W Black. 2019. Code-mixed question answering challenge: Crowd-sourcing data and techniques. In *Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 29–38. Association for Computational Linguistics (ACL).
*   Chandu et al. (2018) Khyathi Chandu, Thomas Manzini, Sumeet Singh, and Alan W Black. 2018. Language informed modeling of code-switched text. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 92–97.
*   Chandu and Black (2020) Khyathi Raghavi Chandu and Alan W. Black. 2020. [Style variation as a vantage point for code-switching](https://doi.org/10.21437/Interspeech.2020-2574 ""). In *Proc. Interspeech 2020*, pages 4761–4765.
*   Chandu et al. (2017) Khyathi Raghavi Chandu, SaiKrishna Rallabandi, Sunayana Sitaram, and Alan W. Black. 2017. [Speech synthesis for mixed-language navigation instructions](https://doi.org/10.21437/Interspeech.2017-1259 ""). In *Proc. Interspeech 2017*, pages 57–61.
*   Chang et al. (2019) Ching-Ting Chang, Shun-Po Chuang, and Hung-Yi Lee. 2019. Code-switching sentence generation by generative adversarial networks and its application to data augmentation. *Proc. Interspeech 2019*, pages 554–558.
*   Chatterjere et al. (2020) Arindam Chatterjere, Vineeth Guptha, Parul Chopra, and Amitava Das. 2020. Minority positive sampling for switching points-an anecdote for the code-mixing language modeling. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 6228–6236.
*   Cheong et al. (2021) Sik Feng Cheong, Hai Leong Chieu, and Jing Lim. 2021. Intrinsic evaluation of language models for code-switching. In *Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)*, pages 81–86.
*   Chinnappa (2021) Dhivya Chinnappa. 2021. dhivya-hope-detection@ lt-edi-eacl2021: multilingual hope speech detection for code-mixed and transliterated texts. In *Proceedings of the First Workshop on Language Technology for Equality, Diversity and Inclusion*, pages 73–78.
*   Chittaranjan et al. (2014) Gokul Chittaranjan, Yogarshi Vyas, Kalika Bali, and Monojit Choudhury. 2014. Word-level language identification using crf: Code-switching shared task report of msr india system. In *Proceedings of The First Workshop on Computational Approaches to Code Switching*, pages 73–79.
*   Cho et al. (2020) Won Ik Cho, Seok Min Kim, and Nam Soo Kim. 2020. Towards an efficient code-mixed grapheme-to-phoneme conversion in an agglutinative language: A case study on to-korean transliteration. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 65–70.
*   Chopra et al. (2021) Parul Chopra, Sai Krishna Rallabandi, Alan W Black, and Khyathi Raghavi Chandu. 2021. Switch point biased self-training: Re-purposing pretrained models for code-switching. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 4389–4397.
*   Choudhury et al. (2017) Monojit Choudhury, Kalika Bali, Sunayana Sitaram, and Ashutosh Baheti. 2017. Curriculum design for code-switching: Experiments with language identification and language modeling with deep neural networks. In *Proceedings of the 14th International Conference on Natural Language Processing (ICON-2017)*, pages 65–74.
*   Chowdhury et al. (2021) Shammur Absar Chowdhury, Amir Hussein, Ahmed Abdelali, and Ahmed Ali. 2021. [Towards one model to rule all: Multilingual strategy for dialectal code-switching arabic asr](https://doi.org/10.21437/Interspeech.2021-1809 ""). In *Proc. Interspeech 2021*, pages 2466–2470.
*   Chu et al. (2007) Chyng-Leei Chu, Dau-cheng Lyu, and Ren-yuan Lyu. 2007. Language identification on code-switching speech. In *Proceedings of ROCLING*.
*   Claeser et al. (2018) Daniel Claeser, Samantha Kent, and Dennis Felske. 2018. Multilingual named entity recognition on spanish-english code-switched tweets using support vector machines. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 132–137.
*   Conneau et al. (2017) Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2017. Word translation without parallel data. *arXiv preprint arXiv:1710.04087*.
*   Conneau et al. (2020) Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Emerging cross-lingual structure in pretrained language models. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6022–6034.
*   Das and Gambäck (2014) Amitava Das and Björn Gambäck. 2014. Identifying languages at the word level in code-mixed indian social media text. In *Proceedings of the 11th International Conference on Natural Language Processing*, pages 378–387.
*   Dave et al. (2021) Bhargav Dave, Shripad Bhat, and Prasenjit Majumder. 2021. Irnlp\_daiict@ dravidianlangtech-eacl2021: offensive language identification in dravidian languages using tf-idf char n-grams and muril. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 266–269.
*   De Leon et al. (2020) Frances Adriana Laureano De Leon, Florimond Guéniat, and Harish Tayyar Madabushi. 2020. Cs-embed at semeval-2020 task 9: The effectiveness of code-switched word embeddings for sentiment analysis. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 922–927.
*   Dey and Fung (2014) Anik Dey and Pascale Fung. 2014. A hindi-english code-switching corpus. In *LREC*, pages 2410–2413.
*   Diab et al. (2016) Mona Diab, Mahmoud Ghoneim, Abdelati Hawwari, Fahad AlGhamdi, Nada Almarwani, and Mohamed Al-Badrashiny. 2016. Creating a large multi-layered representational repository of linguistic code switched arabic data. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 4228–4235.
*   Diab and Kamboj (2011) Mona Diab and Ankit Kamboj. 2011. Feasibility of leveraging crowd sourcing for the creation of a large scale annotated resource for hindi english code switched data: A pilot annotation. In *Proceedings of the 9th Workshop on Asian Language Resources*, pages 36–40.
*   Diwan et al. (2021) Anuj Diwan, Rakesh Vaideeswaran, Sanket Shah, Ankita Singh, Srinivasa Raghavan, Shreya Khare, Vinit Unni, Saurabh Vyas, Akash Rajpuria, Chiranjeevi Yarra, Ashish Mittal, Prasanta Kumar Ghosh, Preethi Jyothi, Kalika Bali, Vivek Seshadri, Sunayana Sitaram, Samarth Bharadwaj, Jai Nanavati, Raoul Nanavati, and Karthik Sankaranarayanan. 2021. [Mucs 2021: Multilingual and code-switching asr challenges for low resource indian languages](https://doi.org/10.21437/Interspeech.2021-1339 ""). In *Proc. Interspeech 2021*, pages 2446–2450.
*   Djegdjiga et al. (2018) Amazouz Djegdjiga, Martine Adda-Decker, and Lori Lamel. 2018. The french-algerian code-switching triggered audio corpus (facst). In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Doğruöz et al. (2021) A Seza Doğruöz, Sunayana Sitaram, Barbara Bullock, and Almeida Jacqueline Toribio. 2021. A survey of code-switching: Linguistic and social perspectives for language technologies. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 1654–1666.
*   dos Santos Neto et al. (2020) Manoel Verissimo dos Santos Neto, Ayrton Amaral, Nádia Silva, and Anderson da Silva Soares. 2020. Deep learning brasil-nlp at semeval-2020 task 9: sentiment analysis of code-mixed tweets using ensemble of language models. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1233–1238.
*   Dowlagar and Mamidi (2021a) Suman Dowlagar and Radhika Mamidi. 2021a. Gated convolutional sequence to sequence based learning for english-hingilsh code-switched machine translation. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 26–30.
*   Dowlagar and Mamidi (2021b) Suman Dowlagar and Radhika Mamidi. 2021b. Graph convolutional networks with multi-headed attention for code-mixed sentiment analysis. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 65–72.
*   Dowlagar and Mamidi (2021c) Suman Dowlagar and Radhika Mamidi. 2021c. Offlangone@ dravidianlangtech-eacl2021: Transformers with the class balanced loss for offensive language identification in dravidian code-mixed text. In *Proceedings of the first workshop on speech and language technologies for dravidian languages*, pages 154–159.
*   Dowlagar and Mamidi (2022) Suman Dowlagar and Radhika Mamidi. 2022. Cmnerone at semeval-2022 task 11: Code-mixed named entity recognition by leveraging multilingual data. In *Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)*, pages 1556–1561.
*   Du Bois (2009) Inke Du Bois. 2009. Language attrition and code-switching among us americans in germany. *Stellenbosch papers in linguistics PLUS*, 39:1–16.
*   Duong et al. (2017) Long Duong, Hadi Afshar, Dominique Estival, Glen Pink, Philip R Cohen, and Mark Johnson. 2017. Multilingual semantic parsing and code-switching. In *Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)*, pages 379–389.
*   Dutta (2022) Aparna Dutta. 2022. Word-level language identification using subword embeddings for code-mixed bangla-english social media data. In *Proceedings of the Workshop on Dataset Creation for Lower-Resourced Languages within the 13th Language Resources and Evaluation Conference*, pages 76–82.
*   El-Haj et al. (2018) Mahmoud El-Haj, Paul Rayson, and Mariam Aboelezz. 2018. Arabic dialect identification in the context of bivalency and code-switching. In *Proceedings of the 11th International Conference on Language Resources and Evaluation, Miyazaki, Japan.*, pages 3622–3627. European Language Resources Association.
*   El Mekki et al. (2022) Abdellah El Mekki, Abdelkader El Mahdaouy, Mohammed Akallouch, Ismail Berrada, and Ahmed Khoumsi. 2022. Um6p-cs at semeval-2022 task 11: Enhancing multilingual and code-mixed complex named entity recognition via pseudo labels using multilingual transformer. In *Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)*, pages 1511–1517.
*   Elfardy et al. (2014) Heba Elfardy, Mohamed Al-Badrashiny, and Mona Diab. 2014. Aida: Identifying code switching in informal arabic text. In *Proceedings of The First Workshop on Computational Approaches to Code Switching*, pages 94–101.
*   Elfardy and Diab (2012) Heba Elfardy and Mona Diab. 2012. Token level identification of linguistic code switching. In *Proceedings of COLING 2012: posters*, pages 287–296.
*   Elmadany et al. (2021) AbdelRahim Elmadany, Muhammad Abdul-Mageed, et al. 2021. Investigating code-mixed modern standard arabic-egyptian to english machine translation. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 56–64.
*   Eskander et al. (2014) Ramy Eskander, Mohamed Al-Badrashiny, Nizar Habash, and Owen Rambow. 2014. Foreign words and the automatic processing of arabic social media text written in roman script. In *Proceedings of The First Workshop on Computational Approaches to Code Switching*, pages 1–12.
*   Fu and Shen (2000) Guokang Fu and Liqin Shen. 2000. Model distance and it’s application on mixed language speech recognition system. *ISCSLP’2000*.
*   Fu et al. (2020) Ruibo Fu, Jianhua Tao, Zhengqi Wen, Jiangyan Yi, Chunyu Qiang, and Tao Wang. 2020. [Dynamic soft windowing and language dependent style token for code-switching end-to-end speech synthesis](https://doi.org/10.21437/Interspeech.2020-1754 ""). In *Proc. Interspeech 2020*, pages 2937–2941.
*   Fung et al. (1999) Pascale Fung, Xiaohu Liu, and Chi-Shun Cheung. 1999. Mixed language query disambiguation. In *Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics*, pages 333–340.
*   Gambäck and Das (2016) Björn Gambäck and Amitava Das. 2016. Comparing the level of code-switching in corpora. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 1850–1855.
*   Ganji and Sinha (2018) Sreeram Ganji and Rohit Sinha. 2018. [A novel approach for effective recognition of the code-switched data on monolingual language model](https://doi.org/10.21437/Interspeech.2018-1259 ""). In *Proc. Interspeech 2018*, pages 1953–1957.
*   Gao et al. (2019) Yingying Gao, Junlan Feng, Ying Liu, Leijing Hou, Xin Pan, and Yong Ma. 2019. [Code-switching sentence generation by bert and generative adversarial networks](https://doi.org/10.21437/Interspeech.2019-2501 ""). In *Proc. Interspeech 2019*, pages 3525–3529.
*   Garain et al. (2020) Avishek Garain, Sainik Mahata, and Dipankar Das. 2020. Junlp at semeval-2020 task 9: Sentiment analysis of hindi-english code mixed data using grid search cross validation. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1276–1280.
*   Garg et al. (2021) Ayush Garg, Sammed Kagi, Vivek Srivastava, and Mayank Singh. 2021. Mipe: A metric independent pipeline for effective code-mixed nlg evaluation. In *Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems*, pages 123–132.
*   Garg et al. (2018a) Saurabh Garg, Tanmay Parekh, and Preethi Jyothi. 2018a. Code-switched language models using dual rnns and same-source pretraining. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3078–3083.
*   Garg et al. (2018b) Saurabh Garg, Tanmay Parekh, and Preethi Jyothi. 2018b. [Dual language models for code switched speech recognition](https://doi.org/10.21437/Interspeech.2018-1343 ""). In *Proc. Interspeech 2018*, pages 2598–2602.
*   Gautam (2022) Akash Kumar Gautam. 2022. Leveraging sub label dependencies in code mixed indian languages for part-of-speech tagging using conditional random fields. In *Proceedings of the WILDRE-6 Workshop within the 13th Language Resources and Evaluation Conference*, pages 13–17.
*   Gautam et al. (2021a) Devansh Gautam, Kshitij Gupta, and Manish Shrivastava. 2021a. Translate and classify: Improving sequence level classification for english-hindi code-mixed data. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 15–25.
*   Gautam et al. (2021b) Devansh Gautam, Prashant Kodali, Kshitij Gupta, Anmol Goel, Manish Shrivastava, and Ponnurangam Kumaraguru. 2021b. Comet: Towards code-mixed translation using parallel monolingual sentences. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 47–55.
*   Geetha et al. (2018) Parvathy Geetha, Khyathi Chandu, and Alan W Black. 2018. Tackling code-switched ner: Participation of cmu. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 126–131.
*   Ghosh et al. (2016) Souvick Ghosh, Satanu Ghosh, and Dipankar Das. 2016. Part-of-speech tagging of code-mixed social media text. In *Proceedings of the second workshop on computational approaches to code switching*, pages 90–97.
*   Ghosh et al. (2019) Urmi Ghosh, Dipti Misra Sharma, and Simran Khanuja. 2019. Dependency parser for bengali-english code-mixed data enhanced with a synthetic treebank. In *Proceedings of the 18th International Workshop on Treebanks and Linguistic Theories (TLT, SyntaxFest 2019)*, pages 91–99.
*   Giwa and Davel (2014) Oluwapelumi Giwa and Marelie H. Davel. 2014. [Language identification of individual words with joint sequence models](https://doi.org/10.21437/Interspeech.2014-344 ""). In *Proc. Interspeech 2014*, pages 1400–1404.
*   Gonen and Goldberg (2019) Hila Gonen and Yoav Goldberg. 2019. Language modeling for code-switching: Evaluation, integration of monolingual data, and discriminative training. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4175–4185.
*   Goodfellow et al. (2020) Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2020. Generative adversarial networks. *Communications of the ACM*, 63(11):139–144.
*   Gopalan and Hopkins (2020) Vinay Gopalan and Mark Hopkins. 2020. Reed at semeval-2020 task 9: Fine-tuning and bag-of-words approaches to code-mixed sentiment analysis. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1304–1309.
*   Goswami et al. (2020) Koustava Goswami, Priya Rani, Bharathi Raja Chakravarthi, Theodorus Fransen, and John Philip McCrae. 2020. Uld@ nuig at semeval-2020 task 9: Generative morphemes with an attention model for sentiment analysis in code-mixed text. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 968–974.
*   Gu et al. (2008) Wen-Tao Gu, Tan Lee, and P. C. Ching. 2008. Prosodic variation in cantonese-english code-mixed speech. In *Proc. International Symposium on Chinese Spoken Language Processing*, pages 342–345.
*   Gundapu and Mamidi (2018) Sunil Gundapu and Radhika Mamidi. 2018. Word level language identification in english telugu code mixed data. In *Proceedings of the 32nd Pacific Asia Conference on Language, Information and Computation*.
*   Gundapu and Mamidi (2020) Sunil Gundapu and Radhika Mamidi. 2020. Gundapusunil at semeval-2020 task 9: Syntactic semantic lstm architecture for sentiment analysis of code-mixed data. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1247–1252.
*   Guo et al. (2018) Pengcheng Guo, Haihua Xu, Lei Xie, and Eng Siong Chng. 2018. [Study of semi-supervised approaches to improving english-mandarin code-switching speech recognition](https://doi.org/10.21437/Interspeech.2018-1974 ""). In *Proc. Interspeech 2018*, pages 1928–1932.
*   Gupta et al. (2021a) Abhirut Gupta, Aditya Vavre, and Sunita Sarawagi. 2021a. Training data augmentation for code-mixed translation. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5760–5766.
*   Gupta et al. (2021b) Akshat Gupta, Sargam Menghani, Sai Krishna Rallabandi, and Alan W Black. 2021b. Unsupervised self-training for sentiment analysis of code-switched data. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 103–112.
*   Gupta et al. (2021c) Akshat Gupta, Sai Krishna Rallabandi, and Alan W Black. 2021c. Task-specific pre-training and cross lingual transfer for sentiment analysis in dravidian code-switched languages. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 73–79.
*   Gupta et al. (2018a) Deepak Gupta, Asif Ekbal, and Pushpak Bhattacharyya. 2018a. A deep neural network based approach for entity extraction in code-mixed indian social media text. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Gupta et al. (2020) Deepak Gupta, Asif Ekbal, and Pushpak Bhattacharyya. 2020. A semi-supervised approach to generate the code-mixed text using pre-trained encoder and transfer learning. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 2267–2280.
*   Gupta et al. (2016) Deepak Gupta, Ankit Lamba, Asif Ekbal, and Pushpak Bhattacharyya. 2016. Opinion mining in a code-mixed environment: A case study with government portals. In *Proceedings of the 13th International Conference on Natural Language Processing*, pages 249–258.
*   Gupta et al. (2018b) Deepak Gupta, Pabitra Lenka, Asif Ekbal, and Pushpak Bhattacharyya. 2018b. Uncovering code-mixed challenges: A framework for linguistically driven question generation and neural based question answering. In *Proceedings of the 22nd Conference on Computational Natural Language Learning*, pages 119–130.
*   Gupta et al. (2018c) Vishal Gupta, Manoj Chinnakotla, and Manish Shrivastava. 2018c. Transliteration better than translation? answering code-mixed questions over a knowledge base. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 39–50.
*   Guzmán et al. (2017) Gualberto A Guzmán, Joseph Ricard, Jacqueline Serigos, Barbara E Bullock, and Almeida Jacqueline Toribio. 2017. Metrics for modeling code-switching across corpora. In *INTERSPEECH*, pages 67–71.
*   Guzman et al. (2016) Gualberto A Guzman, Jacqueline Serigos, Barbara Bullock, and Almeida Jacqueline Toribio. 2016. Simple tools for exploring variation in code-switching for linguists. In *Proceedings of the second workshop on computational approaches to code switching*, pages 12–20.
*   Guzmán et al. (2017) Gualberto Guzmán, Joseph Ricard, Jacqueline Serigos, Barbara E. Bullock, and Almeida Jacqueline Toribio. 2017. [Metrics for modeling code-switching across corpora](https://doi.org/10.21437/Interspeech.2017-1429 ""). In *Proc. Interspeech 2017*, pages 67–71.
*   Hamed et al. (2018) Injy Hamed, Mohamed Elmahdy, and Slim Abdennadher. 2018. Collection and analysis of code-switch egyptian arabic-english speech corpus. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Hamed et al. (2022) Injy Hamed, Amir Hussein, Oumnia Chellah, Shammur Chowdhury, Hamdy Mubarak, Sunayana Sitaram, Nizar Habash, and Ahmed Ali. 2022. Benchmarking evaluation metrics for code-switching automatic speech recognition. *arXiv preprint arXiv:2211.16319*.
*   Hamed et al. (2020) Injy Hamed, Ngoc Thang Vu, and Slim Abdennadher. 2020. Arzen: A speech corpus for code-switched egyptian arabic-english. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 4237–4246.
*   Hartmann et al. (2018) Silvana Hartmann, Monojit Choudhury, and Kalika Bali. 2018. An integrated representation of linguistic and social functions of code-switching. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   He et al. (2012) Ji He, Yao Qian, Frank K Soong, and Sheng Zhao. 2012. Turning a monolingual speaker into multilingual for a mixed-language tts. In *Thirteenth Annual Conference of the International Speech Communication Association*.
*   Hegde et al. (2022) Asha Hegde, Mudoor Devadas Anusha, Sharal Coelho, Hosahalli Lakshmaiah Shashirekha, and Bharathi Raja Chakravarthi. 2022. Corpus creation for sentiment analysis in code-mixed tulu text. In *Proceedings of the 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages*, pages 33–40.
*   Herrera et al. (2022) Megan Herrera, Ankit Aich, and Natalie Parde. 2022. Tweettaglish: A dataset for investigating tagalog-english code-switching. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 2090–2097.
*   Hirvonen and Lauttamus (2000) Pekka Hirvonen and Timo Lauttamus. 2000. Code-switching and language attrition: Evidence from american finnish interview speech. *SKY journal of linguistics*, 13:47–74.
*   Hossain et al. (2021) Eftekhar Hossain, Omar Sharif, and Mohammed Moshiul Hoque. 2021. Nlp-cuet@ lt-edi-eacl2021: Multilingual code-mixed hope speech detection using cross-lingual representation learner. In *Proceedings of the First Workshop on Language Technology for Equality, Diversity and Inclusion*, pages 168–174.
*   Hu et al. (2020) Xinhui Hu, Qi Zhang, Lei Yang, Binbin Gu, and Xinkang Xu. 2020. [Data augmentation for code-switch language modeling by fusing multiple text generation methods](https://doi.org/10.21437/Interspeech.2020-2219 ""). In *Proc. Interspeech 2020*, pages 1062–1066.
*   Huang and Bai (2021) Bo Huang and Yang Bai. 2021. hub at semeval-2021 task 7: Fusion of albert and word frequency information detecting and rating humor and offense. In *Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)*, pages 1141–1145.
*   Huang and Yates (2014) Fei Huang and Alexander Yates. 2014. Improving word alignment using linguistic code switching data. In *Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics*, pages 1–9.
*   Iliescu et al. (2021) Dana-Maria Iliescu, Rasmus Grand, Sara Qirko, and Rob van der Goot. 2021. Much gracias: Semi-supervised code-switch detection for spanish-english: How far can we get? *NAACL 2021*, page 65.
*   Imseng et al. (2010) David Imseng, Hervé Bourlard, and Mathew Magimai Doss. 2010. [Towards mixed language speech recognition systems](https://doi.org/10.21437/Interspeech.2010-110 ""). In *Proc. Interspeech 2010*, pages 278–281.
*   Jaech et al. (2016) Aaron Jaech, George Mulcaire, Mari Ostendorf, and Noah A Smith. 2016. A neural model for language identification in code-switched tweets. In *Proceedings of The Second Workshop on Computational Approaches to Code Switching*, pages 60–64.
*   Jain et al. (2018) Devanshu Jain, Maria Kustikova, Mayank Darbari, Rishabh Gupta, and Stephen Mayhew. 2018. Simple features for strong performance on named entity recognition in code-switched twitter data. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 103–109.
*   Jain and Bhat (2014) Naman Jain and Riyaz Ahmad Bhat. 2014. Language identification in code-switching scenario. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 87–93.
*   Jamatia et al. (2015) Anupam Jamatia, Björn Gambäck, and Amitava Das. 2015. Part-of-speech tagging for code-mixed english-hindi twitter and facebook chat messages. In *Proceedings of the international conference recent advances in natural language processing*, pages 239–248.
*   Janke et al. (2018) Florian Janke, Tongrui Li, Eric Rincón, Gualberto A Guzman, Barbara Bullock, and Almeida Jacqueline Toribio. 2018. The university of texas system submission for the code-switching workshop shared task 2018. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 120–125.
*   Javdan et al. (2020) Soroush Javdan, Behrouz Minaei-Bidgoli, et al. 2020. Iust at semeval-2020 task 9: Sentiment analysis for code-mixed social media text using deep neural networks and linear baselines. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1270–1275.
*   Jawahar et al. (2021) Ganesh Jawahar, Muhammad Abdul-Mageed, VS Laks Lakshmanan, et al. 2021. Exploring text-to-text transformers for english to hinglish machine translation with synthetic code-mixing. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 36–46.
*   Jayanthi et al. (2021) Sai Muralidhar Jayanthi, Kavya Nerella, Khyathi Raghavi Chandu, and Alan W Black. 2021. Codemixednlp: An extensible and open nlp toolkit for code-mixing. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 113–118.
*   Jhamtani et al. (2014) Harsh Jhamtani, Suleep Kumar Bhogi, and Vaskar Raychoudhury. 2014. Word-level language identification in bi-lingual code-switched texts. In *Proceedings of the 28th Pacific Asia Conference on language, information and computing*, pages 348–357.
*   Johanson (1999) Lars Johanson. 1999. The dynamics of code-copying in language encounters. *Language encounters across time and space*, 3762.
*   Jose et al. (2020) Navya Jose, Bharathi Raja Chakravarthi, Shardul Suryawanshi, Elizabeth Sherly, and John P McCrae. 2020. A survey of current datasets for code-switching research. In *2020 6th international conference on advanced computing and communication systems (ICACCS)*, pages 136–141. IEEE.
*   Joshi et al. (2016) Aditya Joshi, Ameya Prabhu, Manish Shrivastava, and Vasudeva Varma. 2016. Towards sub-word level compositions for sentiment analysis of hindi-english code mixed text. In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, pages 2482–2491.
*   Joshi (1982) Aravind Joshi. 1982. Processing of sentences with intra-sentential code-switching. In *Coling 1982: Proceedings of the Ninth International Conference on Computational Linguistics*.
*   Joshi et al. (2020) Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the nlp world. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6282–6293.
*   Joulin et al. (2016) Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, and Tomas Mikolov. 2016. Fasttext.zip: Compressing text classification models. *arXiv preprint arXiv:1612.03651*.
*   Jurgens et al. (2014) David Jurgens, Stefan Dimitrov, and Derek Ruths. 2014. Twitter users# codeswitch hashtags!# moltoimportante# wow. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 51–61.
*   Kevers (2022) Laurent Kevers. 2022. Coswid, a code switching identification method suitable for under-resourced languages. In *Proceedings of the 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages*, pages 112–121.
*   Khan et al. (2021) Humair Raj Khan, Deepak Gupta, and Asif Ekbal. 2021. Towards developing a multilingual and code-mixed visual question answering system by knowledge distillation. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 1753–1767.
*   Khandelwal et al. (2018) Ankush Khandelwal, Sahil Swami, Syed S Akhtar, and Manish Shrivastava. 2018. Humor detection in english-hindi code-mixed social media content: Corpus and baseline system. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Khanuja et al. (2020a) Simran Khanuja, Sandipan Dandapat, Sunayana Sitaram, and Monojit Choudhury. 2020a. A new dataset for natural language inference from code-mixed conversations. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 9–16.
*   Khanuja et al. (2020b) Simran Khanuja, Sandipan Dandapat, Anirudh Srinivasan, Sunayana Sitaram, and Monojit Choudhury. 2020b. Gluecos: An evaluation benchmark for code-switched nlp. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 3575–3585.
*   Khassanov et al. (2019) Yerbolat Khassanov, Haihua Xu, Van Tung Pham, Zhiping Zeng, Eng Siong Chng, Chongjia Ni, and Bin Ma. 2019. [Constrained output embeddings for end-to-end code-switching speech recognition with only monolingual data](https://doi.org/10.21437/Interspeech.2019-1867 ""). In *Proc. Interspeech 2019*, pages 2160–2164.
*   King et al. (2014) Levi King, Eric Baucom, Timur Gilmanov, Sandra Kübler, Daniel Whyatt, Wolfgang Maier, and Paul Rodrigues. 2014. The iucl+ system: Word-level language identification via extended markov models. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 102–106.
*   Klejch et al. (2021) Ondřej Klejch, Electra Wallington, and Peter Bell. 2021. [The cstr system for multilingual and code-switching asr challenges for low resource indian languages](https://doi.org/10.21437/Interspeech.2021-1035 ""). In *Proc. Interspeech 2021*, pages 2881–2885.
*   Knill et al. (2020) Kate M. Knill, Linlin Wang, Yu Wang, Xixin Wu, and Mark J.F. Gales. 2020. [Non-native children’s automatic speech recognition: The interspeech 2020 shared task alta systems](https://doi.org/10.21437/Interspeech.2020-2154 ""). In *Proc. Interspeech 2020*, pages 255–259.
*   Kojima and Tanaka (2003) Hiroaki Kojima and Kazuyo Tanaka. 2003. Mixed-lingual spoken word recognition by using vq codebook sequences of variable length segments. In *Eighth European Conference on Speech Communication and Technology*.
*   Kong et al. (2020) Jun Kong, Jin Wang, and Xuejie Zhang. 2020. Hpcc-ynu at semeval-2020 task 9: A bilingual vector gating mechanism for sentiment analysis of code-mixed text. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 940–945.
*   Kumar et al. (2020) Ayush Kumar, Harsh Agarwal, Keshav Bansal, and Ashutosh Modi. 2020. Baksa at semeval-2020 task 9: Bolstering cnn with self-attention for sentiment analysis of code mixed text. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1221–1226.
*   Kumar et al. (2021) Mari Ganesh Kumar, Jom Kuriakose, Anand Thyagachandran, Arun Kumar A, Ashish Seth, Lodagala V.S.V. Durga Prasad, Saish Jaiswal, Anusha Prakash, and Hema A. Murthy. 2021. [Dual script e2e framework for multilingual and code-switching asr](https://doi.org/10.21437/Interspeech.2021-978 ""). In *Proc. Interspeech 2021*, pages 2441–2445.
*   Kumar et al. (2018) Ritesh Kumar, Aishwarya N Reganti, Akshit Bhatia, and Tushar Maheshwari. 2018. Aggression-annotated corpus of hindi-english code-mixed data. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Lal et al. (2019) Yash Kumar Lal, Vaibhav Kumar, Mrinal Dhar, Manish Shrivastava, and Philipp Koehn. 2019. De-mixing sentiment from code-mixed text. In *Proceedings of the 57th annual meeting of the association for computational linguistics: student research workshop*, pages 371–377.
*   Lee and Li (2020) Grandee Lee and Haizhou Li. 2020. Modeling code-switch languages using bilingual parallel corpus. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 860–870.
*   Lee et al. (2019a) Grandee Lee, Xianghu Yue, and Haizhou Li. 2019a. Linguistically motivated parallel data augmentation for code-switch language modeling. In *Interspeech*, pages 3730–3734.
*   Lee et al. (2019b) Grandee Lee, Xianghu Yue, and Haizhou Li. 2019b. [Linguistically motivated parallel data augmentation for code-switch language modeling](https://doi.org/10.21437/Interspeech.2019-1382 ""). In *Proc. Interspeech 2019*, pages 3730–3734.
*   Li et al. (2022) Chengfei Li, Shuhao Deng, Yaoping Wang, Guangjing Wang, Yaguang Gong, Changbin Chen, and Jinfeng Bai. 2022. [Talcs: An open-source mandarin-english code-switching corpus and a speech recognition baseline](https://doi.org/10.21437/Interspeech.2022-877 ""). In *Proc. Interspeech 2022*, pages 1741–1745.
*   Li and Vu (2020) Chia-Yu Li and Ngoc Thang Vu. 2020. [Improving code-switching language modeling with artificially generated texts using cycle-consistent adversarial networks](https://doi.org/10.21437/Interspeech.2020-2177 ""). In *Proc. Interspeech 2020*, pages 1057–1061.
*   Li and Fung (2012) Ying Li and Pascale Fung. 2012. Code-switch language model with inversion constraints for mixed language speech recognition. In *Proceedings of COLING 2012*, pages 1671–1680.
*   Li and Fung (2013) Ying Li and Pascale Fung. 2013. Language modeling for mixed language speech recognition using weighted phrase extraction. In *Interspeech*, pages 2599–2603.
*   Li and Fung (2014) Ying Li and Pascale Fung. 2014. Language modeling with functional head constraint for code switching speech recognition. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 907–916.
*   Li et al. (2012) Ying Li, Yue Yu, and Pascale Fung. 2012. A mandarin-english code-switching corpus. In *Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12)*, pages 2515–2519.
*   Li (2021) Zichao Li. 2021. Codewithzichao@ dravidianlangtech-eacl2021: Exploring multimodal transformers for meme classification in tamil language. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 352–356.
*   Liang et al. (2007) Hui Liang, Yao Qian, and Frank K Soong. 2007. An hmm-based bilingual (mandarin-english) tts. *Proceedings of SSW6*.
*   Liang et al. (2013) Wei-Bin Liang, Chung-Hsien Wu, and Chun-Shan Hsu. 2013. [Code-switching event detection based on delta-bic using phonetic eigenvoice models](https://doi.org/10.21437/Interspeech.2013-385 ""). In *Proc. Interspeech 2013*, pages 1487–1491.
*   Liang et al. (2020) Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao, et al. 2020. Xglue: A new benchmark datasetfor cross-lingual pre-training, understanding and generation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6008–6018.
*   Libovickỳ et al. (2020) Jindřich Libovickỳ, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 1663–1674.
*   Lin et al. (2014) Chu-Cheng Lin, Waleed Ammar, Lori Levin, and Chris Dyer. 2014. The cmu submission for the shared task on language identification in code-switched data. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 80–86.
*   Lin and Chen (2021) Hou-An Lin and Chia-Ping Chen. 2021. Exploiting low-resource code-switching data to mandarin-english speech recognition systems. In *Proceedings of the 33rd Conference on Computational Linguistics and Speech Processing (ROCLING 2021)*, pages 81–86.
*   Lin and Chen (2020) Wei-Ting Lin and Berlin Chen. 2020. Exploring disparate language model combination strategies for mandarin-english code-switching asr. In *Proceedings of the 32nd Conference on Computational Linguistics and Speech Processing (ROCLING 2020)*, pages 346–358.
*   Liu et al. (2021) Hexin Liu, Leibny Paola García Perera, Xinyi Zhang, Justin Dauwels, Andy W.H. Khong, Sanjeev Khudanpur, and Suzy J. Styles. 2021. [End-to-end language diarization for bilingual code-switching speech](https://doi.org/10.21437/Interspeech.2021-82 ""). In *Proc. Interspeech 2021*, pages 1489–1493.
*   Liu et al. (2020) Jiaxiang Liu, Xuyi Chen, Shikun Feng, Shuohuan Wang, Xuan Ouyang, Yu Sun, Zhengjie Huang, and Weiyue Su. 2020. Kk2018 at semeval-2020 task 9: Adversarial training for code-mixing sentiment classification. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 817–823.
*   Lounnas et al. (2021) Khaled Lounnas, Mourad Abbas, and Mohamed Lichouri. 2021. Towards phone number recognition for code switched algerian dialect. In *Proceedings of The Fourth International Conference on Natural Language and Speech Processing (ICNLSP 2021)*, pages 290–294.
*   Lovenia et al. (2022) Holy Lovenia, Samuel Cahyawijaya, Genta Winata, Peng Xu, Yan Xu, Zihan Liu, Rita Frieske, Tiezheng Yu, Wenliang Dai, Elham J Barezi, et al. 2022. Ascend: A spontaneous chinese-english dataset for code-switching in multi-turn conversation. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 7259–7268.
*   Lovenia et al. (2021) Holy Lovenia, Samuel Cahyawijaya, Genta Indra Winata, Peng Xu, Xu Yan, Zihan Liu, Rita Frieske, Tiezheng Yu, Wenliang Dai, Elham J Barezi, et al. 2021. Ascend: A spontaneous chinese-english dataset for code-switching in multi-turn conversation. *arXiv preprint arXiv:2112.06223*.
*   Lu et al. (2020) Yizhou Lu, Mingkun Huang, Hao Li, Jiaqi Guo, and Yanmin Qian. 2020. [Bi-encoder transformer network for mandarin-english code-switching speech recognition using mixture of experts](https://doi.org/10.21437/Interspeech.2020-2485 ""). In *Proc. Interspeech 2020*, pages 4766–4770.
*   Lyu and Lyu (2008) Dau-Cheng Lyu and Ren-Yuan Lyu. 2008. [Language identification on code-switching utterances using multiple cues](https://doi.org/10.21437/Interspeech.2008-223 ""). In *Proc. Interspeech 2008*, pages 711–714.
*   Lyu et al. (2010a) Dau-Cheng Lyu, Tien-Ping Tan, Eng Siong Chng, and Haizhou Li. 2010a. Seame: a mandarin-english code-switching speech corpus in south-east asia. In *Eleventh Annual Conference of the International Speech Communication Association*.
*   Lyu et al. (2010b) Dau-Cheng Lyu, Tien-Ping Tan, Eng Siong Chng, and Haizhou Li. 2010b. [Seame: a mandarin-english code-switching speech corpus in south-east asia](https://doi.org/10.21437/Interspeech.2010-563 ""). In *Proc. Interspeech 2010*, pages 1986–1989.
*   Lyudovyk and Pylypenko (2014) Tetyana Lyudovyk and Valeriy Pylypenko. 2014. Code-switching speech recognition for closely related languages. In *Proc. 4th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2014)*, pages 188–193.
*   Ma et al. (2020) Yili Ma, Liang Zhao, and Jie Hao. 2020. Xlp at semeval-2020 task 9: Cross-lingual models with focal loss for sentiment analysis of code-mixing language. In *Proceedings of the fourteenth workshop on semantic evaluation*, pages 975–980.
*   Mabokela et al. (2014) Koena Ronny Mabokela, Madimetja Jonas Manamela, and Mabu Manaileng. 2014. Modeling code-switching speech on under-resourced languages for language identification. In *Spoken Language Technologies for Under-Resourced Languages*.
*   Mager et al. (2019) Manuel Mager, Özlem Çetinoğlu, and Katharina Kann. 2019. Subword-level language identification for intra-word code-switching. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2005–2011.
*   Mahata et al. (2021) Sainik Mahata, Dipankar Das, and Sivaji Bandyopadhyay. 2021. Sentiment classification of code-mixed tweets using bi-directional rnn and language tags. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 28–35.
*   Makhija et al. (2020) Piyush Makhija, Ankit Srivastava, and Anuj Gupta. 2020. hinglishnorm-a corpus of hindi-english code mixed sentences for text normalization. In *Proceedings of the 28th International Conference on Computational Linguistics: Industry Track*, pages 136–145.
*   Malmasi et al. (2022a) Shervin Malmasi, Anjie Fang, Besnik Fetahu, Sudipta Kar, and Oleg Rokhlenko. 2022a. Multiconer: A large-scale multilingual dataset for complex named entity recognition. In *Proceedings of the 29th International Conference on Computational Linguistics*, pages 3798–3809.
*   Malmasi et al. (2022b) Shervin Malmasi, Anjie Fang, Besnik Fetahu, Sudipta Kar, and Oleg Rokhlenko. 2022b. Semeval-2022 task 11: Multilingual complex named entity recognition (multiconer). In *Proceedings of the 16th international workshop on semantic evaluation (SemEval-2022)*, pages 1412–1437.
*   Malte et al. (2020) Aditya Malte, Pratik Bhavsar, and Sushant Rathi. 2020. Team\_swift at semeval-2020 task 9: Tiny data specialists through domain-specific pre-training on code-mixed data. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1310–1315.
*   Mandal and Nanmaran (2018) Soumil Mandal and Karthick Nanmaran. 2018. Normalization of transliterated words in code-mixed data using seq2seq model & levenshtein distance. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text*, pages 49–53.
*   Mandal and Singh (2018) Soumil Mandal and Anil Kumar Singh. 2018. Language identification in code-mixed data using multichannel neural networks and context capture. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text*, pages 116–120.
*   Mandalam and Sharma (2021) Asrita Venkata Mandalam and Yashvardhan Sharma. 2021. Sentiment analysis of dravidian code mixed data. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 46–54.
*   Manghat et al. (2020) Sreeja Manghat, Sreeram Manghat, and Tanja Schultz. 2020. [Malayalam-english code-switched: Grapheme to phoneme system](https://doi.org/10.21437/Interspeech.2020-1936 ""). In *Proc. Interspeech 2020*, pages 4133–4137.
*   Manghat et al. (2022) Sreeram Manghat, Sreeja Manghat, and Tanja Schultz. 2022. [Normalization of code-switched text for speech synthesis](https://doi.org/10.21437/Interspeech.2022-10719 ""). In *Proc. Interspeech 2022*, pages 4297–4301.
*   Marcadet et al. (2005) J. C. Marcadet, V. Fischer, and C. Waast-Richard. 2005. [A transformation-based learning approach to language identification for mixed-lingual text-to-speech synthesis](https://doi.org/10.21437/Interspeech.2005-711 ""). In *Proc. Interspeech 2005*, pages 2249–2252.
*   Mave et al. (2018) Deepthi Mave, Suraj Maharjan, and Thamar Solorio. 2018. Language identification and analysis of code-switched social media text. In *Proceedings of the third workshop on computational approaches to linguistic code-switching*, pages 51–61.
*   Mehnaz et al. (2021) Laiba Mehnaz, Debanjan Mahata, Rakesh Gosangi, Uma Sushmitha Gunturi, Riya Jain, Gauri Gupta, Amardeep Kumar, Isabelle G Lee, Anish Acharya, and Rajiv Shah. 2021. Gupshup: Summarizing open-domain code-switched conversations. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6177–6192.
*   Mellado and Lignos (2022) Elena Álvarez Mellado and Constantine Lignos. 2022. Borrowing or codeswitching? annotating for finer-grained distinctions in language mixing. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 3195–3201.
*   Mendels et al. (2018) Gideon Mendels, Victor Soto, Aaron Jaech, and Julia Hirschberg. 2018. Collecting code-switched data from social media. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Molina et al. (2016) Giovanni Molina, Fahad AlGhamdi, Mahmoud Ghoneim, Abdelati Hawwari, Nicolas Rey-Villamizar, Mona Diab, and Thamar Solorio. 2016. Overview for the second shared task on language identification in code-switched data. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 40–49.
*   Muennighoff et al. (2022) Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. *arXiv preprint arXiv:2211.01786*.
*   Mukherjee et al. (2019) Siddhartha Mukherjee, Vinuthkumar Prasan, Anish Nediyanchath, Manan Shah, and Nikhil Kumar. 2019. Robust deep learning based sentiment classification of code-mixed text. In *Proceedings of the 16th International Conference on Natural Language Processing*, pages 124–129.
*   Mussakhojayeva et al. (2022a) Saida Mussakhojayeva, Yerbolat Khassanov, and Huseyin Atakan Varol. 2022a. Kazakhtts2: Extending the open-source kazakh tts corpus with more data, speakers, and topics. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 5404–5411.
*   Mussakhojayeva et al. (2022b) Saida Mussakhojayeva, Yerbolat Khassanov, and Huseyin Atakan Varol. 2022b. [Ksc2: An industrial-scale open-source kazakh speech corpus](https://doi.org/10.21437/Interspeech.2022-421 ""). In *Proc. Interspeech 2022*, pages 1367–1371.
*   Myers-Scotton (1997) Carol Myers-Scotton. 1997. *Duelling languages: Grammatical structure in codeswitching*. Oxford University Press.
*   Myers-Scotton (2005) Carol Myers-Scotton. 2005. *Multiple voices: An introduction to bilingualism*. John Wiley & Sons.
*   Nayak and Joshi (2022) Ravindra Nayak and Raviraj Joshi. 2022. L3cube-hingcorpus and hingbert: A code mixed hindi-english dataset and bert language models. In *Proceedings of the WILDRE-6 Workshop within the 13th Language Resources and Evaluation Conference*, pages 7–12.
*   Nekvinda and Dušek (2020) Tomáš Nekvinda and Ondřej Dušek. 2020. [One model, many languages: Meta-learning for multilingual text-to-speech](https://doi.org/10.21437/Interspeech.2020-2679 ""). In *Proc. Interspeech 2020*, pages 2972–2976.
*   Nguyen and Bryant (2020) Li Nguyen and Christopher Bryant. 2020. Canvec-the canberra vietnamese-english code-switching natural speech corpus. In *Proceedings of the 12th Language Resources and Evaluation Conference*, pages 4121–4129.
*   Niesler and de Wet (2008) Thomas Niesler and Febe de Wet. 2008. Accent identification in the presence of code-mixing. In *Proc. The Speaker and Language Recognition Workshop (Odyssey 2008)*, page paper 27.
*   Niesler et al. (2018) Thomas Niesler et al. 2018. A first south african corpus of multilingual code-switched soap opera speech. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Núñez and Wisniewski (2018) José Carlos Rosales Núñez and Guillaume Wisniewski. 2018. Analyse morpho-syntaxique en présence d’alternance codique (pos tagging of code switching). In *Actes de la Conférence TALN. Volume 1-Articles longs, articles courts de TALN*, pages 473–480.
*   Oco and Roxas (2012) Nathaniel Oco and Rachel Edita Roxas. 2012. Pattern matching refinements to dictionary-based code-switching point detection. In *Proceedings of the 26th Pacific Asia Conference on Language, Information, and Computation*, pages 229–236.
*   Oria and Vetek (2004) Daniela Oria and Akos Vetek. 2004. [Multilingual e-mail text processing for speech synthesis](https://doi.org/10.21437/Interspeech.2004-310 ""). In *Proc. Interspeech 2004*, pages 841–844.
*   Ostapenko et al. (2022) Alissa Ostapenko, Shuly Wintner, Melinda Fricke, and Yulia Tsvetkov. 2022. Speaker information can guide models to better inductive biases: A case study on predicting code-switching. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3853–3867.
*   Otundo and Grice (2022) Billian Khalayi Otundo and Martine Grice. 2022. [Intonation in advice-giving in kenyan english and kiswahili](https://doi.org/10.21437/SpeechProsody.2022-31 ""). In *Proc. Speech Prosody 2022*, pages 150–154.
*   Ouyang et al. (2022) Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*.
*   Özateş et al. (2022) Şaziye Özateş, Arzucan Özgür, Tunga Güngör, and Özlem Çetinoğlu. 2022. Improving code-switching dependency parsing with semi-supervised auxiliary tasks. In *Findings of the Association for Computational Linguistics: NAACL 2022*, pages 1159–1171.
*   Özateş and Çetinoğlu (2021) Şaziye Betül Özateş and Özlem Çetinoğlu. 2021. A language-aware approach to code-switched morphological tagging. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 72–83.
*   Palomino and Ochoa-Luna (2020) Daniel Palomino and José Ochoa-Luna. 2020. Palomino-ochoa at semeval-2020 task 9: Robust system based on transformer for code-mixed sentiment classification. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 963–967.
*   Pandey et al. (2017) Ayushi Pandey, Brij Mohan Lal Srivastava, and Suryakanth Gangashetty. 2017. Towards developing a phonetically balanced code-mixed speech corpus for hindi-english asr. In *Proceedings of the 14th International Conference on Natural Language Processing (ICON-2017)*, pages 95–101.
*   Pandey et al. (2018) Ayushi Pandey, Brij Mohan Lal Srivastava, Rohit Kumar, Bhanu Teja Nellore, Kasi Sai Teja, and Suryakanth V Gangashetty. 2018. Phonetically balanced code-mixed speech corpus for hindi-english automatic speech recognition. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*.
*   Pant and Dadu (2020) Kartikey Pant and Tanvi Dadu. 2020. Towards code-switched classification exploiting constituent language resources. In *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: Student Research Workshop*, pages 37–43.
*   Papalexakis et al. (2014) Evangelos Papalexakis, Dong Nguyen, and A Seza Doğruöz. 2014. Predicting code-switching in multilingual communication for immigrant communities. In *First Workshop on Computational Approaches to Code Switching (EMNLP 2014)*, pages 42–50. Association for Computational Linguistics (ACL).
*   Parekh et al. (2020) Tanmay Parekh, Emily Ahn, Yulia Tsvetkov, and Alan W Black. 2020. Understanding linguistic accommodation in code-switched human-machine dialogues. In *Proceedings of the 24th Conference on Computational Natural Language Learning*, pages 565–577.
*   Parikh et al. (2020) Apurva Parikh, Abhimanyu Singh Bisht, and Prasenjit Majumder. 2020. Irlab\_daiict at semeval-2020 task 9: Machine learning and deep learning methods for sentiment analysis of code-mixed tweets. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1265–1269.
*   Parikh and Solorio (2021) Dwija Parikh and Thamar Solorio. 2021. Normalization and back-transliteration for code-switched data. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 119–124.
*   Patwa et al. (2020) Parth Patwa, Gustavo Aguilar, Sudipta Kar, Suraj Pandey, Srinivas Pykl, Björn Gambäck, Tanmoy Chakraborty, Thamar Solorio, and Amitava Das. 2020. Semeval-2020 task 9: Overview of sentiment analysis of code-mixed tweets. In *Proceedings of the fourteenth workshop on semantic evaluation*, pages 774–790.
*   Peng et al. (2014) Nanyun Peng, Yiming Wang, and Mark Dredze. 2014. Learning polylingual topic models from code-switched social media documents. In *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 674–679.
*   Pfister and Romsdorfer (2003) Beat Pfister and Harald Romsdorfer. 2003. [Mixed-lingual text analysis for polyglot tts synthesis](https://doi.org/10.21437/Eurospeech.2003-585 ""). In *Proc. 8th European Conference on Speech Communication and Technology (Eurospeech 2003)*, pages 2037–2040.
*   Phadte and Thakkar (2017) Akshata Phadte and Gaurish Thakkar. 2017. Towards normalising konkani-english code-mixed social media text. In *Proceedings of the 14th International Conference on Natural Language Processing (ICON-2017)*, pages 85–94.
*   Piccinini and Garellek (2014) Page Piccinini and Marc Garellek. 2014. [Prosodic cues to monolingual versus code-switching sentences in english and spanish](https://doi.org/10.21437/SpeechProsody.2014-166 ""). In *Proc. Speech Prosody 2014*, pages 885–889.
*   Piergallini et al. (2016) Mario Piergallini, Rouzbeh Shirvani, Gauri Shankar Gautam, and Mohamed Chouikha. 2016. Word-level language identification and predicting codeswitching points in swahili-english language data. In *Proceedings of the second workshop on computational approaches to code switching*, pages 21–29.
*   Pires et al. (2019) Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual bert? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4996–5001.
*   Poplack (1978) Shana Poplack. 1978. *Syntactic structure and social function of code-switching*, volume 2. Centro de Estudios Puertorriqueños,\[City University of New York\].
*   Poplack (1980) Shana Poplack. 1980. Sometimes i’ll start a sentence in spanish y termino en espanol: toward a typology of code-switching. *Linguistics*, 18:581–618.
*   Prakash et al. (2019) Anusha Prakash, Anju Leela Thomas, S. Umesh, and Hema A Murthy. 2019. [Building multilingual end-to-end speech synthesisers for indian languages](https://doi.org/10.21437/SSW.2019-35 ""). In *Proc. 10th ISCA Workshop on Speech Synthesis (SSW 10)*, pages 194–199.
*   Prasad et al. (2021) Archiki Prasad, Mohammad Ali Rehan, Shreya Pathak, and Preethi Jyothi. 2021. The effectiveness of intermediate-task training for code-switched natural language understanding. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pages 176–190.
*   Pratapa et al. (2018a) Adithya Pratapa, Gayatri Bhat, Monojit Choudhury, Sunayana Sitaram, Sandipan Dandapat, and Kalika Bali. 2018a. Language modeling for code-mixing: The role of linguistic theory based synthetic data. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1543–1553.
*   Pratapa and Choudhury (2017) Adithya Pratapa and Monojit Choudhury. 2017. Quantitative characterization of code switching patterns in complex multi-party conversations: A case study on hindi movie scripts. In *Proceedings of the 14th International Conference on Natural Language Processing (ICON-2017)*, pages 75–84.
*   Pratapa and Choudhury (2021) Adithya Pratapa and Monojit Choudhury. 2021. Comparing grammatical theories of code-mixing. In *Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)*, pages 158–167.
*   Pratapa et al. (2018b) Adithya Pratapa, Monojit Choudhury, and Sunayana Sitaram. 2018b. Word embeddings for code-mixed language processing. In *Proceedings of the 2018 conference on empirical methods in natural language processing*, pages 3067–3072.
*   Priyanshu et al. (2021) Aman Priyanshu, Aleti Vardhan, Sudarshan Sivakumar, Supriti Vijay, and Nipuna Chhabra. 2021. “something something hota hai!” an explainable approach towards sentiment analysis on indian code-mixed data. In *Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)*, pages 437–444.
*   Qian et al. (2008) Yao Qian, Houwei Cao, and Frank K Soong. 2008. Hmm-based mixed-language (mandarin-english) speech synthesis. In *2008 6th International Symposium on Chinese Spoken Language Processing*, pages 1–4. IEEE.
*   Qiu et al. (2020) Zimeng Qiu, Yiyuan Li, Xinjian Li, Florian Metze, and William M. Campbell. 2020. [Towards context-aware end-to-end code-switching speech recognition](https://doi.org/10.21437/Interspeech.2020-1980 ""). In *Proc. Interspeech 2020*, pages 4776–4780.
*   Radford et al. (2022) Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. *arXiv preprint arXiv:2212.04356*.
*   Raha et al. (2019) Tathagata Raha, Sainik Mahata, Dipankar Das, and Sivaji Bandyopadhyay. 2019. Development of pos tagger for english-bengali code-mixed data. In *Proceedings of the 16th International Conference on Natural Language Processing*, pages 143–149.
*   Rajalakshmi et al. (2021) Ratnavel Rajalakshmi, Yashwant Reddy, and Lokesh Kumar. 2021. Dlrg@ dravidianlangtech-eacl2021: Transformer based approachfor offensive language identification on code-mixed tamil. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 357–362.
*   Rallabandi and Black (2017) SaiKrishna Rallabandi and Alan W. Black. 2017. [On building mixed lingual speech synthesis systems](https://doi.org/10.21437/Interspeech.2017-1244 ""). In *Proc. Interspeech 2017*, pages 52–56.
*   Rallabandi and Black (2019) SaiKrishna Rallabandi and Alan W. Black. 2019. [Variational attention using articulatory priors for generating code mixed speech using monolingual corpora](https://doi.org/10.21437/Interspeech.2019-1103 ""). In *Proc. Interspeech 2019*, pages 3735–3739.
*   Rallabandi et al. (2018) SaiKrishna Rallabandi, Sunayana Sitaram, and Alan W Black. 2018. Automatic detection of code-switching style from acoustics. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 76–81.
*   Ramanarayanan and Suendermann-Oeft (2017) Vikram Ramanarayanan and David Suendermann-Oeft. 2017. [Jee haan, i’d like both, por favor: Elicitation of a code-switched corpus of hindi–english and spanish–english human–machine dialog](https://doi.org/10.21437/Interspeech.2017-1198 ""). In *Proc. Interspeech 2017*, pages 47–51.
*   Rambabu and Gangashetty (2018) Banothu Rambabu and Suryakanth V Gangashetty. 2018. [Development of iiith hindi-english code mixed speech database](https://doi.org/10.21437/SLTU.2018-23 ""). In *Proc. 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2018)*, pages 107–111.
*   Rani et al. (2022) Priya Rani, John Philip McCrae, and Theodorus Fransen. 2022. Mhe: Code-mixed corpora for similar language identification. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 3425–3433.
*   Rani et al. (2020) Priya Rani, Shardul Suryawanshi, Koustava Goswami, Bharathi Raja Chakravarthi, Theodorus Fransen, and John Philip McCrae. 2020. A comparative study of different state-of-the-art hate speech detection methods in hindi-english code-mixed data. In *Proceedings of the second workshop on trolling, aggression and cyberbullying*, pages 42–48.
*   Rao et al. (2018) Preeti Rao, Mugdha Pandya, Kamini Sabu, Kanhaiya Kumar, and Nandini Bondale. 2018. [A study of lexical and prosodic cues to segmentation in a hindi-english code-switched discourse](https://doi.org/10.21437/Interspeech.2018-1600 ""). In *Proc. Interspeech 2018*, pages 1918–1922.
*   Ravikiran and Annamalai (2021) Manikandan Ravikiran and Subbiah Annamalai. 2021. Dosa: Dravidian code-mixed offensive span identification dataset. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 10–17.
*   Ravikiran and Chakravarthi (2022) Manikandan Ravikiran and Bharathi Raja Chakravarthi. 2022. Zero-shot code-mixed offensive span identification through rationale extraction. In *Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages*, pages 240–247.
*   Ravikiran et al. (2022) Manikandan Ravikiran, Bharathi Raja Chakravarthi, S Sangeetha, Ratnavel Rajalakshmi, Sajeetha Thavareesan, Rahul Ponnusamy, Shankar Mahadevan, et al. 2022. Findings of the shared task on offensive span identification fromcode-mixed tamil-english comments. In *Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages*, pages 261–270.
*   Ren et al. (2005) Xiaolin Ren, Xin He, and Yaxin Zhang. 2005. Mandarin/english mixed-lingual name recognition for mobile phone. In *INTERSPEECH*, pages 3373–3376.
*   Rijhwani et al. (2017) Shruti Rijhwani, Royal Sequiera, Monojit Choudhury, Kalika Bali, and Chandra Shekhar Maddila. 2017. Estimating code-switching on twitter with a novel generalized word-level language detection technique. In *Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: long papers)*, pages 1971–1982.
*   Romsdorfer and Pfister (2005) Harald Romsdorfer and Beat Pfister. 2005. [Phonetic labeling and segmentation of mixed-lingual prosody databases](https://doi.org/10.21437/Interspeech.2005-572 ""). In *Proc. Interspeech 2005*, pages 3281–3284.
*   Romsdorfer and Pfister (2006) Harald Romsdorfer and Beat Pfister. 2006. Character stream parsing of mixed-lingual text. In *Proc. Multilingual Language and Speech Processing (MULTILING 2006)*, page paper 021.
*   Rosner and Farrugia (2007) Mike Rosner and Paulseph-John Farrugia. 2007. [A tagging algorithm for mixed language identification in a noisy domain](https://doi.org/10.21437/Interspeech.2007-77 ""). In *Proc. Interspeech 2007*, pages 190–193.
*   Ruder et al. (2021) Sebastian Ruder, Noah Constant, Jan Botha, Aditya Siddhant, Orhan Firat, Jinlan Fu, Pengfei Liu, Junjie Hu, Dan Garrette, Graham Neubig, et al. 2021. Xtreme-r: Towards more challenging and nuanced multilingual evaluation. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 10215–10245.
*   Sabty et al. (2020) Caroline Sabty, Mohamed Islam, and Slim Abdennadher. 2020. Contextual embeddings for arabic-english code-switched data. In *Proceedings of the Fifth Arabic Natural Language Processing Workshop*, pages 215–225.
*   Samih et al. (2016a) Younes Samih, Suraj Maharjan, Mohammed Attia, Laura Kallmeyer, and Thamar Solorio. 2016a. Multilingual code-switching identification via lstm recurrent neural networks. In *Proceedings of the second workshop on computational approaches to code switching*, pages 50–59.
*   Samih and Maier (2016) Younes Samih and Wolfgang Maier. 2016. An arabic-moroccan darija code-switched corpus. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 4170–4175.
*   Samih et al. (2016b) Younes Samih, Wolfgang Maier, and Laura Kallmeyer. 2016b. Sawt: Sequence annotation web tool. In *Proceedings of the second workshop on computational approaches to code switching*, pages 65–70.
*   Sankoff (1998) David Sankoff. 1998. The production of code-mixed discourse. In *COLING 1998 Volume 1: The 17th International Conference on Computational Linguistics*.
*   Santy et al. (2021) Sebastin Santy, Anirudh Srinivasan, and Monojit Choudhury. 2021. Bertologicomix: How does code-mixing interact with multilingual bert? In *Proceedings of the Second Workshop on Domain Adaptation for NLP*, pages 111–121.
*   Saumya et al. (2021) Sunil Saumya, Abhinav Kumar, and Jyoti Prakash Singh. 2021. Offensive language identification in dravidian code mixed social media text. In *Proceedings of the first workshop on speech and language technologies for Dravidian languages*, pages 36–45.
*   Scao et al. (2022) Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b-parameter open-access multilingual language model. *arXiv preprint arXiv:2211.05100*.
*   Sequiera et al. (2015) Royal Sequiera, Monojit Choudhury, and Kalika Bali. 2015. Pos tagging of hindi-english code mixed text from social media: Some machine learning experiments. In *Proceedings of the 12th international conference on natural language processing*, pages 237–246.
*   Shah et al. (2020) Sanket Shah, Basil Abraham, Sunayana Sitaram, Vikas Joshi, et al. 2020. Learning to recognize code-switched speech without forgetting monolingual speech recognition. *arXiv preprint arXiv:2006.00782*.
*   Shah et al. (2019) Sanket Shah, Pratik Joshi, Sebastin Santy, and Sunayana Sitaram. 2019. Cossat: Code-switched speech annotation tool. In *Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*, pages 48–52.
*   Sharif et al. (2021) Omar Sharif, Eftekhar Hossain, and Mohammed Moshiul Hoque. 2021. Nlp-cuet@ dravidianlangtech-eacl2021: Offensive language detection from multilingual code-mixed text using transformers. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 255–261.
*   Sharma et al. (2016) Arnav Sharma, Sakshi Gupta, Raveesh Motlani, Piyush Bansal, Manish Shrivastava, Radhika Mamidi, and Dipti Misra Sharma. 2016. Shallow parsing pipeline-hindi-english code-mixed social media text. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1340–1345.
*   Sharma et al. (2020) Yash Sharma, Basil Abraham, Karan Taneja, and Preethi Jyothi. 2020. [Improving low resource code-switched asr using augmented code-switched tts](https://doi.org/10.21437/Interspeech.2020-2402 ""). In *Proc. Interspeech 2020*, pages 4771–4775.
*   Shen and Guo (2022) Zhijie Shen and Wu Guo. 2022. [An improved deliberation network with text pre-training for code-switching automatic speech recognition](https://doi.org/10.21437/Interspeech.2022-221 ""). In *Proc. Interspeech 2022*, pages 3854–3858.
*   Shirvani et al. (2016) Rouzbeh Shirvani, Mario Piergallini, Gauri Shankar Gautam, and Mohamed Chouikha. 2016. The howard university system submission for the shared task in language identification in spanish-english codeswitching. In *Proceedings of the second workshop on computational approaches to code switching*, pages 116–120.
*   Shoemark et al. (2018) Philippa Shoemark, James Kirby, and Sharon Goldwater. 2018. Inducing a lexicon of sociolinguistic variables from code-mixed text. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text*, pages 1–6.
*   Shrestha (2014) Prajwol Shrestha. 2014. Incremental n-gram approach for language identification in code-switched text. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 133–138.
*   Shrestha (2016) Prajwol Shrestha. 2016. Codeswitching detection via lexical features in conditional random fields. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 121–126.
*   Shuang et al. (2010) Zhiwei Shuang, Shiyin Kang, Yong Qin, Lirong Dai, and Lianhong Cai. 2010. Hmm based tts for mixed language text. In *Eleventh Annual Conference of the International Speech Communication Association*.
*   Sikdar et al. (2018) Utpal Kumar Sikdar, Biswanath Barik, and Björn Gambäck. 2018. Named entity recognition on code-switched data using conditional random fields. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 115–119.
*   Sikdar and Gambäck (2016) Utpal Kumar Sikdar and Björn Gambäck. 2016. Language identification in code-switched text using conditional random fields and babelnet. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 127–131.
*   Singh and Tan (2018) Anand Singh and Tien-Ping Tan. 2018. [Evaluating code-switched malay-english speech using time delay neural networks](https://doi.org/10.21437/SLTU.2018-41 ""). In *Proc. 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2018)*, pages 197–200.
*   Singh et al. (2018) Kushagra Singh, Indira Sen, and Ponnurangam Kumaraguru. 2018. Language identification and named entity recognition in hinglish code mixed tweets. In *Proceedings of ACL 2018, Student Research Workshop*, pages 52–58.
*   Singh and Lefever (2020) Pranaydeep Singh and Els Lefever. 2020. Sentiment analysis for hinglish code-mixed tweets by means of cross-lingual word embeddings. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 45–51.
*   Sitaram and Black (2016) Sunayana Sitaram and Alan W Black. 2016. Speech synthesis of code-mixed text. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 3422–3428.
*   Sitaram et al. (2019) Sunayana Sitaram, Khyathi Raghavi Chandu, Sai Krishna Rallabandi, and Alan W Black. 2019. A survey of code-switched speech and language processing. *arXiv preprint arXiv:1904.00784*.
*   Sitaram et al. (2016) Sunayana Sitaram, Sai Krishna Rallabandi, Shruti Rijhwani, and Alan W. Black. 2016. [Experiments with cross-lingual systems for synthesis of code-mixed text](https://doi.org/10.21437/SSW.2016-13 ""). In *Proc. 9th ISCA Workshop on Speech Synthesis Workshop (SSW 9)*, pages 76–81.
*   Sivasankaran et al. (2018) Sunit Sivasankaran, Brij Mohan Lal Srivastava, Sunayana Sitaram, Kalika Bali, and Monojit Choudhury. 2018. Phone merging for code-switched speech recognition. In *Third Workshop on Computational Approaches to Linguistic Code-switching*.
*   Solorio et al. (2014) Thamar Solorio, Elizabeth Blair, Suraj Maharjan, Steven Bethard, Mona Diab, Mahmoud Ghoneim, Abdelati Hawwari, Fahad AlGhamdi, Julia Hirschberg, Alison Chang, et al. 2014. Overview for the first shared task on language identification in code-switched data. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 62–72.
*   Solorio and Liu (2008a) Thamar Solorio and Yang Liu. 2008a. Learning to predict code-switching points. In *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing*, pages 973–981.
*   Solorio and Liu (2008b) Thamar Solorio and Yang Liu. 2008b. Part-of-speech tagging for english-spanish code-switched text. In *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing*, pages 1051–1060.
*   Song et al. (2022) Tongtong Song, Qiang Xu, Meng Ge, Longbiao Wang, Hao Shi, Yongjie Lv, Yuqin Lin, and Jianwu Dang. 2022. [Language-specific characteristic assistance for code-switching speech recognition](https://doi.org/10.21437/Interspeech.2022-11426 ""). In *Proc. Interspeech 2022*, pages 3924–3928.
*   Sonu et al. (2022) Sanket Sonu, Rejwanul Haque, Mohammed Hasanuzzaman, Paul Stynes, and Pramod Pathak. 2022. Identifying emotions in code mixed hindi-english tweets. In *Proceedings of the WILDRE-6 Workshop within the 13th Language Resources and Evaluation Conference*, pages 35–41.
*   Soto et al. (2018) Victor Soto, Nishmar Cestero, and Julia Hirschberg. 2018. [The role of cognate words, pos tags and entrainment in code-switching](https://doi.org/10.21437/Interspeech.2018-1099 ""). In *Proc. Interspeech 2018*, pages 1938–1942.
*   Soto and Hirschberg (2017) Victor Soto and Julia Hirschberg. 2017. [Crowdsourcing universal part-of-speech tags for code-switching](https://doi.org/10.21437/Interspeech.2017-1663 ""). In *Proc. Interspeech 2017*, pages 77–81.
*   Soto and Hirschberg (2018) Victor Soto and Julia Hirschberg. 2018. Joint part-of-speech and language id tagging for code-switched data. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 1–10.
*   Soto and Hirschberg (2019) Victor Soto and Julia Hirschberg. 2019. [Improving code-switched language modeling performance using cognate features](https://doi.org/10.21437/Interspeech.2019-2681 ""). In *Proc. Interspeech 2019*, pages 3725–3729.
*   SR et al. (2022) Mithun Kumar SR, Lov Kumar, and Aruna Malapati. 2022. Sentiment analysis on code-switched dravidian languages with kernel based extreme learning machines. In *Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages*, pages 184–190.
*   Sravani et al. (2021) Dama Sravani, Lalitha Kameswari, and Radhika Mamidi. 2021. Political discourse analysis: a case study of code mixing and code switching in political speeches. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 1–5.
*   Srinivasan (2020) Anirudh Srinivasan. 2020. Msr india at semeval-2020 task 9: Multilingual models can do code-mixing too. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 951–956.
*   Srinivasan et al. (2020) Anirudh Srinivasan, Sandipan Dandapat, and Monojit Choudhury. 2020. Code-mixed parse trees and how to find them. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 57–64.
*   Srirangam et al. (2019) Vamshi Krishna Srirangam, Appidi Abhinav Reddy, Vinay Singh, and Manish Shrivastava. 2019. Corpus creation and analysis for named entity recognition in telugu-english code-mixed social media data. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, pages 183–189.
*   Srivastava et al. (2022) Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *arXiv preprint arXiv:2206.04615*.
*   Srivastava and Vardhan (2020) Aditya Srivastava and V Harsha Vardhan. 2020. Hcms at semeval-2020 task 9: A neural approach to sentiment analysis for code-mixed texts. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1253–1258.
*   Srivastava and Sitaram (2018) Brij Mohan Lal Srivastava and Sunayana Sitaram. 2018. [Homophone identification and merging for code-switched speech recognition](https://doi.org/10.21437/Interspeech.2018-1171 ""). In *Proc. Interspeech 2018*, pages 1943–1947.
*   Srivastava and Singh (2020a) Vivek Srivastava and Mayank Singh. 2020a. Iit gandhinagar at semeval-2020 task 9: Code-mixed sentiment classification using candidate sentence generation and selection. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1259–1264.
*   Srivastava and Singh (2020b) Vivek Srivastava and Mayank Singh. 2020b. Phinc: A parallel hinglish social media code-mixed corpus for machine translation. In *Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)*, pages 41–49.
*   Srivastava and Singh (2021a) Vivek Srivastava and Mayank Singh. 2021a. Challenges and limitations with the metrics measuring the complexity of code-mixed text. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 6–14.
*   Srivastava and Singh (2021b) Vivek Srivastava and Mayank Singh. 2021b. Hinge: A dataset for generation and evaluation of code-mixed hinglish text. In *Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems*, pages 200–208.
*   Srivastava and Singh (2021c) Vivek Srivastava and Mayank Singh. 2021c. Quality evaluation of the low-resource synthetically generated code-mixed hinglish text. In *Proceedings of the 14th International Conference on Natural Language Generation*, pages 314–319.
*   Stymne et al. (2020) Sara Stymne et al. 2020. Evaluating word embeddings for indonesian–english code-mixed text based on synthetic data. In *Proceedings of the The 4th Workshop on Computational Approaches to Code Switching*, pages 26–35.
*   Sultan et al. (2020) Ahmed Sultan, Mahmoud Salim, Amina Gaber, and Islam El Hosary. 2020. Wessa at semeval-2020 task 9: Code-mixed sentiment analysis using transformers. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1342–1347.
*   Sutton et al. (2012) Charles Sutton, Andrew McCallum, et al. 2012. An introduction to conditional random fields. *Foundations and Trends® in Machine Learning*, 4(4):267–373.
*   Swaminathan et al. (2022) Krithika Swaminathan, K Divyasri, GL Gayathri, Thenmozhi Durairaj, and B Bharathi. 2022. Pandas@ abusive comment detection in tamil code-mixed data using custom embeddings with labse. In *Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages*, pages 112–119.
*   Taguchi et al. (2022) Chihiro Taguchi, Sei Iwata, and Taro Watanabe. 2022. Universal dependencies treebank for tatar: Incorporating intra-word code-switching information. In *Proceedings of the Workshop on Resources and Technologies for Indigenous, Endangered and Lesser-resourced Languages in Eurasia within the 13th Language Resources and Evaluation Conference*, pages 95–104.
*   Taguchi et al. (2021) Chihiro Taguchi, Yusuke Sakai, and Taro Watanabe. 2021. Transliteration for low-resource code-switching texts: Building an automatic cyrillic-to-latin converter for tatar. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 133–140.
*   Taneja et al. (2019) Karan Taneja, Satarupa Guha, Preethi Jyothi, and Basil Abraham. 2019. [Exploiting monolingual speech corpora for code-mixed speech recognition](https://doi.org/10.21437/Interspeech.2019-1959 ""). In *Proc. Interspeech 2019*, pages 2150–2154.
*   Tarunesh et al. (2021) Ishan Tarunesh, Syamantak Kumar, and Preethi Jyothi. 2021. From machine translation to code-switching: Generating high-quality code-switched text. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 3154–3169.
*   Thomas et al. (2018a) Anju Leela Thomas, Anusha Prakash, Arun Baby, and Hema Murthy. 2018a. [Code-switching in indic speech synthesisers](https://doi.org/10.21437/Interspeech.2018-1178 ""). In *Proc. Interspeech 2018*, pages 1948–1952.
*   Thomas et al. (2018b) Anju Leela Thomas, Anusha Prakash, Arun Baby, and Hema A Murthy. 2018b. Code-switching in indic speech synthesisers. In *INTERSPEECH*, pages 1948–1952.
*   Tian et al. (2022) Jinchuan Tian, Jianwei Yu, Chunlei Zhang, Yuexian Zou, and Dong Yu. 2022. [Lae: Language-aware encoder for monolingual and multilingual asr](https://doi.org/10.21437/Interspeech.2022-923 ""). In *Proc. Interspeech 2022*, pages 3178–3182.
*   Trivedi et al. (2018) Shashwat Trivedi, Harsh Rangwani, and Anil Kumar Singh. 2018. Iit (bhu) submission for the acl shared task on named entity recognition on code-switched data. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 148–153.
*   Tucker (2001) G Richard Tucker. 2001. A global perspective on bilingualism and bilingual education. *GEORGETOWN UNIVERSITY ROUND TABLE ON LANGUAGES AND LINGUISTICS 1999*, page 332.
*   van der Westhuizen and Niesler (2017) Ewald van der Westhuizen and Thomas Niesler. 2017. Synthesising isizulu-english code-switch bigrams using word embeddings. In *INTERSPEECH*, pages 72–76.
*   Vasantharajan and Thayasivam (2021) Charangan Vasantharajan and Uthayasanker Thayasivam. 2021. Hypers@ dravidianlangtech-eacl2021: Offensive language identification in dravidian code-mixed youtube comments and posts. In *Proceedings of the first workshop on speech and language technologies for dravidian languages*, pages 195–202.
*   Vijay et al. (2018) Deepanshu Vijay, Aditya Bohra, Vinay Singh, Syed Sarfaraz Akhtar, and Manish Shrivastava. 2018. Corpus creation and emotion prediction for hindi-english code-mixed social media text. In *Proceedings of the 2018 conference of the North American chapter of the Association for Computational Linguistics: student research workshop*, pages 128–135.
*   Vilares et al. (2016) David Vilares, Miguel A Alonso, and Carlos Gómez-Rodríguez. 2016. En-es-cs: An english-spanish code-switching twitter corpus for multilingual sentiment analysis. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 4149–4153.
*   Volk and Clematide (2014) Martin Volk and Simon Clematide. 2014. Detecting code-switching in a multilingual alpine heritage corpus. In *Proceedings of the first workshop on computational approaches to code switching*, pages 24–33.
*   Volk et al. (2022) Martin Volk, Lukas Fischer, Patricia Scheurer, Bernard Silvan Schroffenegger, Raphael Schwitter, Phillip Ströbel, and Benjamin Suter. 2022. Nunc profana tractemus. detecting code-switching in a large corpus of 16th century letters. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 2901–2908.
*   Voss et al. (2014) Clare Voss, Stephen Tratz, Jamal Laoudi, and Douglas Briesch. 2014. Finding romanized arabic dialect in code-mixed tweets. In *Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14)*, pages 2249–2253.
*   Vu and Schultz (2014) Ngoc Thang Vu and Tanja Schultz. 2014. Exploration of the impact of maximum entropy in recurrent neural network language models for code-switching speech. In *Proceedings of The First Workshop on Computational Approaches to Code Switching*, pages 34–41.
*   Vyas et al. (2014) Yogarshi Vyas, Spandana Gella, Jatin Sharma, Kalika Bali, and Monojit Choudhury. 2014. Pos tagging of english-hindi code-mixed social media content. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 974–979.
*   Wadhawan and Aggarwal (2021) Anshul Wadhawan and Akshita Aggarwal. 2021. Towards emotion recognition in hindi-english code-mixed data: A transformer based approach. In *Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*, pages 195–202.
*   Wang et al. (2018) Changhan Wang, Kyunghyun Cho, and Douwe Kiela. 2018. Code-switched named entity recognition with embedding attention. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 154–158.
*   Wang et al. (2020) Jisung Wang, Jihwan Kim, Sangki Kim, and Yeha Lee. 2020. [Exploring lexicon-free modeling units for end-to-end korean and korean-english code-switching speech recognition](https://doi.org/10.21437/Interspeech.2020-2440 ""). In *Proc. Interspeech 2020*, pages 1072–1075.
*   Wang et al. (2019) Qinyi Wang, Emre Yılmaz, Adem Derinel, and Haizhou Li. 2019. [Code-switching detection using asr-generated language posteriors](https://doi.org/10.21437/Interspeech.2019-1161 ""). In *Proc. Interspeech 2019*, pages 3740–3744.
*   Wang et al. (2015) Zhongqing Wang, Sophia Lee, Shoushan Li, and Guodong Zhou. 2015. Emotion detection in code-switching texts via bilingual and sentimental information. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 763–768.
*   Wang et al. (2016) Zhongqing Wang, Yue Zhang, Sophia Lee, Shoushan Li, and Guodong Zhou. 2016. A bilingual attention network for code-switched emotion prediction. In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, pages 1624–1634.
*   Weiner et al. (2012a) Jochen Weiner, Ngoc Thang Vu, Dominic Telaar, Florian Metze, Tanja Schultz, Dau-Cheng Lyu, Eng-Siong Chng, and Haizhou Li. 2012a. Integration of language identification into a recognition system for spoken conversations containing code-switches. In *Spoken Language Technologies for Under-Resourced Languages*.
*   Weiner et al. (2012b) Jochen Weiner, Ngoc Thang Vu, Dominic Telaar, Florian Metze, Tanja Schultz, Dau-Cheng Lyu, Eng-Siong Chng, and Haizhou Li. 2012b. Integration of language identification into a recognition system for spoken conversations containing code-switches. In *Proc. 3rd Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2012)*, pages 76–79.
*   White et al. (2008) Christopher M White, Sanjeev Khudanpur, and James K Baker. 2008. An investigation of acoustic models for multilingual code-switching. In *Ninth Annual Conference of the International Speech Communication Association*.
*   Wiesner et al. (2021) Matthew Wiesner, Mousmita Sarma, Ashish Arora, Desh Raj, Dongji Gao, Ruizhe Huang, Supreet Preet, Moris Johnson, Zikra Iqbal, Nagendra Goel, Jan Trmal, Leibny Paola García Perera, and Sanjeev Khudanpur. 2021. [Training hybrid models on noisy transliterated transcripts for code-switched speech recognition](https://doi.org/10.21437/Interspeech.2021-2127 ""). In *Proc. Interspeech 2021*, pages 2906–2910.
*   Wilkinson et al. (2020) Nick Wilkinson, Astik Biswas, Emre Yilmaz, Febe De Wet, Thomas Niesler, et al. 2020. Semi-supervised acoustic modelling for five-lingual code-switched asr using automatically-segmented soap opera speech. In *Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)*, pages 70–78.
*   Winata et al. (2022) Genta Winata, Shijie Wu, Mayank Kulkarni, Thamar Solorio, and Daniel Preoţiuc-Pietro. 2022. Cross-lingual few-shot learning on unseen languages. In *Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing*, pages 777–791.
*   Winata et al. (2020) Genta Indra Winata, Samuel Cahyawijaya, Zhaojiang Lin, Zihan Liu, Peng Xu, and Pascale Fung. 2020. Meta-transfer learning for code-switched speech recognition. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 3770–3776.
*   Winata et al. (2021a) Genta Indra Winata, Samuel Cahyawijaya, Zihan Liu, Zhaojiang Lin, Andrea Madotto, and Pascale Fung. 2021a. Are multilingual models effective in code-switching? In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 142–153.
*   Winata et al. (2019a) Genta Indra Winata, Zhaojiang Lin, and Pascale Fung. 2019a. Learning multilingual meta-embeddings for code-switching named entity recognition. In *Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019)*, pages 181–186.
*   Winata et al. (2019b) Genta Indra Winata, Zhaojiang Lin, Jamin Shin, Zihan Liu, and Pascale Fung. 2019b. Hierarchical meta-embeddings for code-switching named entity recognition. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3541–3547.
*   Winata et al. (2021b) Genta Indra Winata, Andrea Madotto, Zhaojiang Lin, Rosanne Liu, Jason Yosinski, and Pascale Fung. 2021b. Language models are few-shot multilingual learners. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pages 1–15.
*   Winata et al. (2018a) Genta Indra Winata, Andrea Madotto, Chien-Sheng Wu, and Pascale Fung. 2018a. Code-switching language modeling using syntax-aware multi-task learning. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 62–67.
*   Winata et al. (2019c) Genta Indra Winata, Andrea Madotto, Chien-Sheng Wu, and Pascale Fung. 2019c. Code-switched language models using neural based synthetic data from parallel sentences. In *Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)*, pages 271–280.
*   Winata et al. (2018b) Genta Indra Winata, Chien-Sheng Wu, Andrea Madotto, and Pascale Fung. 2018b. Bilingual character representation for efficiently addressing out-of-vocabulary words in code-switching named entity recognition. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 110–114.
*   Wottawa et al. (2018) Jane Wottawa, Amazouz Djegdjiga, Martine Adda-Decker, and Lori Lamel. 2018. [Studying vowel variation in french-algerian arabic code-switched speech](https://doi.org/10.21437/Interspeech.2018-2381 ""). In *Proc. Interspeech 2018*, pages 2753–2757.
*   Wu et al. (2020) Qi Wu, Peng Wang, and Chenghao Huang. 2020. Meistermorxrc at semeval-2020 task 9: Fine-tune bert and multitask learning for sentiment analysis of code-mixed tweets. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1294–1297.
*   Wu et al. (2011) Yi-Lun Wu, Chaio-Wen Hsieh, Wei-Hsuan Lin, Chun-Yi Liu, and Liang-Chih Yu. 2011. [Unknown word extraction from multilingual code-switching sentences](https://aclanthology.org/O11-2013 ""). In *ROCLING 2011 Poster Papers*, pages 349–360, Taipei, Taiwan. The Association for Computational Linguistics and Chinese Language Processing (ACLCLP).
*   Xia (2016) Meng Xuan Xia. 2016. Codeswitching language identification using subword information enriched word vectors. In *Proceedings of the second workshop on computational approaches to code switching*, pages 132–136.
*   Xia and Cheung (2016) Meng Xuan Xia and Jackie Chi Kit Cheung. 2016. Accurate pinyin-english codeswitched language identification. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 71–79.
*   Xia et al. (2022) Xinyuan Xia, Lu Xiao, Kun Yang, and Yueyue Wang. 2022. Identifying tension in holocaust survivors’ interview: Code-switching/code-mixing as cues. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 1490–1495.
*   Xu et al. (2018) Haihua Xu, Van Tung Pham, Zin Tun Kyaw, Zhi Hao Lim, Eng Siong Chng, and Haizhou Li. 2018. Mandarin-english code-switching speech recognition. In *Proc. Interspeech 2018*, pages 554–555.
*   Xu and Yvon (2021) Jitao Xu and François Yvon. 2021. Can you traducir this? machine translation for code-switched input. In *Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching*, pages 84–94.
*   Xu et al. (2021) Qihui Xu, Magdalena Markowska, Martin Chodorow, and Ping Li. 2021. A network science approach to bilingual code-switching. *Proceedings of the Society for Computation in Linguistics*, 4(1):18–27.
*   Xue et al. (2019) Liumeng Xue, Wei Song, Guanghui Xu, Lei Xie, and Zhizheng Wu. 2019. [Building a mixed-lingual neural tts system with only monolingual data](https://doi.org/10.21437/Interspeech.2019-3191 ""). In *Proc. Interspeech 2019*, pages 2060–2064.
*   Yang et al. (2020) Zhen Yang, Bojie Hu, Ambyera Han, Shen Huang, and Qi Ju. 2020. Csp: Code-switching pre-training for neural machine translation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 2624–2636.
*   Ye et al. (2022) Lingxuan Ye, Gaofeng Cheng, Runyan Yang, Zehui Yang, Sanli Tian, Pengyuan Zhang, and Yonghong Yan. 2022. [Improving recognition of out-of-vocabulary words in e2e code-switching asr by fusing speech generation methods](https://doi.org/10.21437/Interspeech.2022-719 ""). In *Proc. Interspeech 2022*, pages 3163–3167.
*   Yeong and Tan (2010) Yin-Lai Yeong and Tien-Ping Tan. 2010. Language identification of code switching malay-english words using syllable structure information. In *Proc. Spoken Language Technologies for Under-Resourced Languages*, pages 142–145.
*   Yeong and Tan (2014) Yin-Lai Yeong and Tien-Ping Tan. 2014. [Language identification of code switching sentences and multilingual sentences of under-resourced languages by using multi structural word information](https://doi.org/10.21437/Interspeech.2014-612 ""). In *Proc. Interspeech 2014*, pages 3052–3055.
*   Yilmaz et al. (2018) E Yilmaz, H Heuvel, and DA van Leeuwen. 2018. Acoustic and textual data augmentation for improved asr of code-switching speech. In *Proceedings of Interspeech*, pages 1933–1937. Hyderabad, India: ISCA.
*   Yilmaz et al. (2016) Emre Yilmaz, Maaike Andringa, Sigrid Kingma, Jelske Dijkstra, Frits van der Kuip, Hans Van de Velde, Frederik Kampstra, Jouke Algra, Henk van den Heuvel, and David van Leeuwen. 2016. A longitudinal bilingual frisian-dutch radio broadcast database designed for code-switching research. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*, pages 4666–4669.
*   Yılmaz et al. (2018) Emre Yılmaz, Astik Biswas, Ewald van der Westhuizen, Febe de Wet, and Thomas Niesler. 2018. Building a unified code-switching asr system for south african languages. *Proceedings of Interspeech*.
*   Yilmaz et al. (2018) Emre Yilmaz, Henk Van Den Heuvel, and David Van Leeuwen. 2018. [Code-switching detection with data-augmented acoustic and language models](https://doi.org/10.21437/SLTU.2018-27 ""). In *Proc. 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2018)*, pages 127–131.
*   Yirmibeşoğlu and Eryiğit (2018) Zeynep Yirmibeşoğlu and Gülşen Eryiğit. 2018. Detecting code-switching between turkish-english language pair. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text*, pages 110–115.
*   Yong et al. (2023) Zheng-Xin Yong, Ruochen Zhang, Jessica Zosa Forde, Skyler Wang, Samuel Cahyawijaya, Holy Lovenia, Genta Indra Winata, Lintang Sutawika, Jan Christian Blaise Cruz, Long Phan, Yin Lin Tan, and Alham Fikri Aji. 2023. [Prompting multilingual large language models to generate code-mixed texts: The case of south east asian languages](http://arxiv.org/abs/2303.13592 "").
*   You et al. (2004) Shan-Ruei You, Shih-Chieh Chien, Chih-Hsing Hsu, Ke-Shiu Chen, Jia-Jang Tu, Jeng Shien Lin, and Sen-Chia Chang. 2004. Chinese-english mixed-lingual keyword spotting. In *2004 International Symposium on Chinese Spoken Language Processing*, pages 237–240. IEEE.
*   Yu and Chen (2020) Fu-Hao Yu and Kuan-Yu Chen. 2020. A preliminary study on leveraging meta learning technique for code-switching speech recognition. In *Proceedings of the 32nd Conference on Computational Linguistics and Speech Processing (ROCLING 2020)*, pages 136–147.
*   Yu et al. (2012) Liang-Chih Yu, Wei-Cheng He, and Wei-Nan Chien. 2012. A language modeling approach to identifying code-switched sentences and words. In *Proceedings of the Second CIPS-SIGHAN Joint Conference on Chinese Language Processing*, pages 3–8.
*   Yılmaz et al. (2019) Emre Yılmaz, Samuel Cohen, Xianghu Yue, David A. van Leeuwen, and Haizhou Li. 2019. [Multi-graph decoding for code-switching asr](https://doi.org/10.21437/Interspeech.2019-1125 ""). In *Proc. Interspeech 2019*, pages 3750–3754.
*   Yılmaz et al. (2017a) Emre Yılmaz, Jelske Dijkstra, Hans Van de Velde, Frederik Kampstra, Jouke Algra, Henk van den Heuvel, and David Van Leeuwen. 2017a. [Longitudinal speaker clustering and verification corpus with code-switching frisian-dutch speech](https://doi.org/10.21437/Interspeech.2017-301 ""). In *Proc. Interspeech 2017*, pages 37–41.
*   Yılmaz et al. (2016) Emre Yılmaz, Henk van den Heuvel, Jelske Dijkstra, Hans Van de Velde, Frederik Kampstra, Jouke Algra, and David Van Leeuwen. 2016. [Open source speech and language resources for frisian](https://doi.org/10.21437/Interspeech.2016-48 ""). In *Proc. Interspeech 2016*, pages 1536–1540.
*   Yılmaz et al. (2017b) Emre Yılmaz, Henk van den Heuvel, and David Van Leeuwen. 2017b. [Exploiting untranscribed broadcast data for improved code-switching detection](https://doi.org/10.21437/Interspeech.2017-391 ""). In *Proc. Interspeech 2017*, pages 42–46.
*   Yılmaz et al. (2018) Emre Yılmaz, Henk van den Heuvel, and David van Leeuwen. 2018. [Acoustic and textual data augmentation for improved asr of code-switching speech](https://doi.org/10.21437/Interspeech.2018-52 ""). In *Proc. Interspeech 2018*, pages 1933–1937.
*   Zaharia et al. (2020) George-Eduard Zaharia, George-Alexandru Vlad, Dumitru-Clementin Cercel, Traian Rebedea, and Costin Chiru. 2020. Upb at semeval-2020 task 9: Identifying sentiment in code-mixed social media texts using transformers and multi-task learning. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1322–1330.
*   Zeng et al. (2019) Zhiping Zeng, Yerbolat Khassanov, Van Tung Pham, Haihua Xu, Eng Siong Chng, and Haizhou Li. 2019. [On the end-to-end solution to mandarin-english code-switching speech recognition](https://doi.org/10.21437/Interspeech.2019-1429 ""). In *Proc. Interspeech 2019*, pages 2165–2169.
*   Zhang et al. (2020) Haobo Zhang, Haihua Xu, Van Tung Pham, Hao Huang, and Eng Siong Chng. 2020. [Monolingual data selection analysis for english-mandarin hybrid code-switching speech recognition](https://doi.org/10.21437/Interspeech.2020-1582 ""). In *Proc. Interspeech 2020*, pages 2392–2396.
*   Zhang et al. (2019) Shiliang Zhang, Yuan Liu, Ming Lei, Bin Ma, and Lei Xie. 2019. [Towards language-universal mandarin-english speech recognition](https://doi.org/10.21437/Interspeech.2019-1365 ""). In *Proc. Interspeech 2019*, pages 2170–2174.
*   Zhang et al. (2021a) Shuai Zhang, Jiangyan Yi, Zhengkun Tian, Ye Bai, Jianhua Tao, Xuefei Liu, and Zhengqi Wen. 2021a. [End-to-end spelling correction conditioned on acoustic feature for code-switching speech recognition](https://doi.org/10.21437/Interspeech.2021-1242 ""). In *Proc. Interspeech 2021*, pages 266–270.
*   Zhang et al. (2022) Shuai Zhang, Jiangyan Yi, Zhengkun Tian, Jianhua Tao, Yu Ting Yeung, and Liqun Deng. 2022. [reducing multilingual context confusion for end-to-end code-switching automatic speech recognition](https://doi.org/10.21437/Interspeech.2022-10286 ""). In *Proc. Interspeech 2022*, pages 3894–3898.
*   Zhang et al. (2021b) Wenxuan Zhang, Ruidan He, Haiyun Peng, Lidong Bing, and Wai Lam. 2021b. Cross-lingual aspect-based sentiment analysis with aspect term code-switching. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 9220–9230.
*   Zhang and Tao (2008) Yi Zhang and Jian-Hua Tao. 2008. Prosody modification on mixed-language speech synthesis. In *Proc. International Symposium on Chinese Spoken Language Processing*, pages 253–256.
*   Zhao et al. (2020) Shengkui Zhao, Trung Hieu Nguyen, Hao Wang, and Bin Ma. 2020. [Towards natural bilingual and code-switched speech synthesis based on mix of monolingual recordings and cross-lingual voice conversion](https://doi.org/10.21437/Interspeech.2020-1163 ""). In *Proc. Interspeech 2020*, pages 2927–2931.
*   Zhou et al. (2020) Xinyuan Zhou, Emre Yılmaz, Yanhua Long, Yijie Li, and Haizhou Li. 2020. [Multi-encoder-decoder transformer for code-switching speech recognition](https://doi.org/10.21437/Interspeech.2020-2488 ""). In *Proc. Interspeech 2020*, pages 1042–1046.
*   Zhu et al. (2020) Yueying Zhu, Xiaobing Zhou, Hongling Li, and Kunjie Dong. 2020. Zyy1510 team at semeval-2020 task 9: Sentiment analysis for code-mixed social media text with sub-word level representations. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 1354–1359.

## Appendix A Annotation Catalog

We release the annotation of all papers we use in the survey.

### A.1 \*CL Anthology

#### Bilingual

Table [7](#A1.T7 "Table 7 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with African-English. Table [8](#A1.T8 "Table 8 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with East Asian-English languages. Table [9](#A1.T9 "Table 9 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with European-English languages. Table [10](#A1.T10 "Table 10 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with Middle Eastern-English languages. Table [11](#A1.T11 "Table 11 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") and Table [12](#A1.T12 "Table 12 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") show the annotation for papers with South Asian-English languages. Table [13](#A1.T13 "Table 13 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with South East Asian-English languages. Table [14](#A1.T14 "Table 14 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with a combination of a language with a dialect. Table [15](#A1.T15 "Table 15 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with languages in the same family. Table [16](#A1.T16 "Table 16 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with languages in different families.

#### Trilingual

Table [17](#A1.T17 "Table 17 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with three languages.

#### 4+

Table [18](#A1.T18 "Table 18 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with four or more languages.

### A.2 ISCA Proceeding

#### Bilingual

Table [19](#A1.T19 "Table 19 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with African-English. Table [20](#A1.T20 "Table 20 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with East Asian-English languages. Table [21](#A1.T21 "Table 21 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with European-English languages. Table [22](#A1.T22 "Table 22 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with Middle Eastern-English languages. Table [23](#A1.T23 "Table 23 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with South Asian-English languages. Table [24](#A1.T24 "Table 24 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with South East Asian-English languages. Table [25](#A1.T25 "Table 25 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with a combination of a language with a dialect. Table [26](#A1.T26 "Table 26 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with languages in the same family. Table [27](#A1.T27 "Table 27 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with languages in different families.

#### Trilingual

Table [28](#A1.T28 "Table 28 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with three languages.

#### 4+

Table [29](#A1.T29 "Table 29 ‣ 4+ ‣ A.2 ISCA Proceeding ‣ Appendix A Annotation Catalog ‣ The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges") shows the annotation for papers with four or more languages.

| Paper        | Proceeding   | IsiZulu      | Swahili      | isiXhosa     | Setswana | Sesotho |
| ------------ | ------------ | ------------ | ------------ | ------------ | -------- | ------- |
| ✓\\checkmark |              |              |              |              |          |         |
|              | ✓\\checkmark |              |              |              |          |         |
| ✓\\checkmark |              | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |          |         |
| ✓\\checkmark |              |              |              |              |          |         |
| ✓\\checkmark |              | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |          |         |
| ✓\\checkmark |              | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |          |         |

Table 7: \*CL Catalog in African-English.

| Paper                                       | Proceeding | Chinese      | Cantonese    | Korean       |
| ------------------------------------------- | ---------- | ------------ | ------------ | ------------ |
|                                             |            |              |              |              |
| [Fung et al. (1999)](#bib.bib130 "")        | ACL        | ✓\\checkmark |              |              |
| [Chan et al. (2009)](#bib.bib82 "")         | IJCLCLP    |              | ✓\\checkmark |              |
| [Li et al. (2012)](#bib.bib217 "")          | LREC       | ✓\\checkmark |              |              |
| [Peng et al. (2014)](#bib.bib283 "")        | ACL-IJCNLP | ✓\\checkmark |              |              |
| [Li and Fung (2014)](#bib.bib216 "")        | EMNLP      | ✓\\checkmark |              |              |
| [Solorio et al. (2014)](#bib.bib349 "")     | CALCS      | ✓\\checkmark |              |              |
| [Chittaranjan et al. (2014)](#bib.bib96 "") | CALCS      | ✓\\checkmark |              |              |
| [Lin et al. (2014)](#bib.bib223 "")         | CALCS      | ✓\\checkmark |              |              |
| [Jain and Bhat (2014)](#bib.bib180 "")      | CALCS      | ✓\\checkmark |              |              |
| [King et al. (2014)](#bib.bib200 "")        | CALCS      | ✓\\checkmark |              |              |
| [Huang and Yates (2014)](#bib.bib175 "")    | EACL       | ✓\\checkmark |              |              |
| [Wang et al. (2015)](#bib.bib397 "")        | ACL-IJCNLP | ✓\\checkmark |              |              |
| [Gambäck and Das (2016)](#bib.bib131 "")    | LREC       | ✓\\checkmark |              |              |
| [Wang et al. (2016)](#bib.bib398 "")        | COLING     | ✓\\checkmark |              |              |
| [Çetinoğlu et al. (2016)](#bib.bib77 "")    | CALCS      | ✓\\checkmark |              |              |
| [Xia and Cheung (2016)](#bib.bib417 "")     | CALCS      | ✓\\checkmark |              |              |
| [Yang et al. (2020)](#bib.bib423 "")        | EMNLP      | ✓\\checkmark |              |              |
| [Calvillo et al. (2020)](#bib.bib71 "")     | EMNLP      | ✓\\checkmark |              |              |
| [Lin and Chen (2020)](#bib.bib225 "")       | ROCLING    | ✓\\checkmark |              |              |
| [Cho et al. (2020)](#bib.bib97 "")          | CALCS      |              |              | ✓\\checkmark |
| [Lin and Chen (2021)](#bib.bib224 "")       | ROCLING    | ✓\\checkmark |              |              |
| [Lovenia et al. (2021)](#bib.bib230 "")     | LREC       | ✓\\checkmark |              |              |

Table 8: \*CL Catalog in East Asian-English.

| Paper                                            | Proceeding        | Spanish      | French       | Portugese    | Polish       | German       | Dutch        | Finnish      |
| ------------------------------------------------ | ----------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|                                                  |                   |              |              |              |              |              |              |              |
| [Sankoff (1998)](#bib.bib324 "")                 | COLING            |              |              |              |              |              |              | ✓\\checkmark |
| [Solorio and Liu (2008a)](#bib.bib350 "")        | EMNLP             | ✓\\checkmark |              |              |              |              |              |              |
| [Solorio and Liu (2008b)](#bib.bib351 "")        | EMNLP             | ✓\\checkmark |              |              |              |              |              |              |
| [Peng et al. (2014)](#bib.bib283 "")             | ACL-IJCNLP        | ✓\\checkmark |              |              |              |              |              |              |
| [Solorio et al. (2014)](#bib.bib349 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Chittaranjan et al. (2014)](#bib.bib96 "")      | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Lin et al. (2014)](#bib.bib223 "")              | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Jain and Bhat (2014)](#bib.bib180 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [King et al. (2014)](#bib.bib200 "")             | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Carpuat (2014)](#bib.bib74 "")                  | CALCS             |              | ✓\\checkmark |              |              |              |              |              |
| [Barman et al. (2014b)](#bib.bib50 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Shrestha (2014)](#bib.bib337 "")                | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Bar and Dershowitz (2014)](#bib.bib48 "")       | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Gambäck and Das (2016)](#bib.bib131 "")         | LREC              | ✓\\checkmark |              |              |              |              |              |              |
| [Vilares et al. (2016)](#bib.bib387 "")          | LREC              | ✓\\checkmark |              |              |              |              |              |              |
| [Çetinoğlu et al. (2016)](#bib.bib77 "")         | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Guzman et al. (2016)](#bib.bib162 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Molina et al. (2016)](#bib.bib254 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Samih et al. (2016a)](#bib.bib321 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Jaech et al. (2016)](#bib.bib178 "")            | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [AlGhamdi et al. (2016)](#bib.bib22 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Al-Badrashiny and Diab (2016)](#bib.bib21 "")   | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Chanda et al. (2016a)](#bib.bib86 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Shirvani et al. (2016)](#bib.bib335 "")         | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Shrestha (2016)](#bib.bib338 "")                | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Sikdar and Gambäck (2016)](#bib.bib341 "")      | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Xia (2016)](#bib.bib416 "")                     | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Duong et al. (2017)](#bib.bib120 "")            | CoNLL             |              |              |              |              | ✓\\checkmark |              |              |
| [Rijhwani et al. (2017)](#bib.bib315 "")         | ACL               | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |              |              | ✓\\checkmark |              |
| [Choudhury et al. (2017)](#bib.bib99 "")         | ICON              | ✓\\checkmark |              |              |              |              |              |              |
| [Núñez and Wisniewski (2018)](#bib.bib266 "")    | TALN PFIA         | ✓\\checkmark |              |              |              |              |              |              |
| [Pratapa et al. (2018b)](#bib.bib296 "")         | EMNLP             | ✓\\checkmark |              |              |              |              |              |              |
| [Mendels et al. (2018)](#bib.bib253 "")          | LREC              | ✓\\checkmark |              |              |              |              |              |              |
| [Soto and Hirschberg (2018)](#bib.bib356 "")     | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Mave et al. (2018)](#bib.bib250 "")             | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Bullock et al. (2018a)](#bib.bib69 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Rallabandi et al. (2018)](#bib.bib305 "")       | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Bawa et al. (2018)](#bib.bib54 "")              | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Jain et al. (2018)](#bib.bib179 "")             | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Winata et al. (2018b)](#bib.bib412 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Sikdar et al. (2018)](#bib.bib340 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Janke et al. (2018)](#bib.bib182 "")            | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Geetha et al. (2018)](#bib.bib141 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Claeser et al. (2018)](#bib.bib102 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Aguilar et al. (2018)](#bib.bib14 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Trivedi et al. (2018)](#bib.bib382 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Wang et al. (2018)](#bib.bib394 "")             | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Gonen and Goldberg (2019)](#bib.bib145 "")      | EMNLP             | ✓\\checkmark |              |              |              |              |              |              |
| [Yang et al. (2020)](#bib.bib423 "")             | EMNLP             |              | ✓\\checkmark |              |              | ✓\\checkmark |              |              |
| [Khanuja et al. (2020b)](#bib.bib198 "")         | ACL               | ✓\\checkmark |              |              |              |              |              |              |
| [Aguilar and Solorio (2020)](#bib.bib17 "")      | ACL               | ✓\\checkmark |              |              |              |              |              |              |
| [Cameron (2020)](#bib.bib72 "")                  | JEP               |              | ✓\\checkmark |              |              |              |              |              |
| [Ahn et al. (2020)](#bib.bib19 "")               | SCiL              | ✓\\checkmark |              |              |              |              |              |              |
| [Srinivasan et al. (2020)](#bib.bib361 "")       | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Patwa et al. (2020)](#bib.bib282 "")            | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [De Leon et al. (2020)](#bib.bib107 "")          | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Aparaschivei et al. (2020)](#bib.bib29 "")      | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Kong et al. (2020)](#bib.bib204 "")             | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Angel et al. (2020)](#bib.bib27 "")             | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Palomino and Ochoa-Luna (2020)](#bib.bib274 "") | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Ma et al. (2020)](#bib.bib236 "")               | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Kumar et al. (2020)](#bib.bib205 "")            | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Advani et al. (2020)](#bib.bib10 "")            | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Javdan et al. (2020)](#bib.bib183 "")           | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Wu et al. (2020)](#bib.bib414 "")               | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Zaharia et al. (2020)](#bib.bib441 "")          | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Sultan et al. (2020)](#bib.bib372 "")           | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Zhu et al. (2020)](#bib.bib451 "")              | SemEval           | ✓\\checkmark |              |              |              |              |              |              |
| [Parekh et al. (2020)](#bib.bib279 "")           | CoNLL             | ✓\\checkmark |              |              |              |              |              |              |
| [Gupta et al. (2020)](#bib.bib157 "")            | Findings of EMNLP | ✓\\checkmark | ✓\\checkmark |              |              | ✓\\checkmark |              |              |
| [Aguilar et al. (2020)](#bib.bib15 "")           | LREC              | ✓\\checkmark |              |              |              |              |              |              |
| [Iliescu et al. (2021)](#bib.bib176 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Xu and Yvon (2021)](#bib.bib420 "")             | CALCS             | ✓\\checkmark | ✓\\checkmark |              |              |              |              |              |
| [Gupta et al. (2021b)](#bib.bib154 "")           | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Jayanthi et al. (2021)](#bib.bib185 "")         | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Winata et al. (2021a)](#bib.bib406 "")          | CALCS             | ✓\\checkmark |              |              |              |              |              |              |
| [Prasad et al. (2021)](#bib.bib292 "")           | MRL               | ✓\\checkmark |              |              |              |              |              |              |
| [Chopra et al. (2021)](#bib.bib98 "")            | Findings of EMNLP | ✓\\checkmark |              |              |              |              |              |              |
| [Santy et al. (2021)](#bib.bib325 "")            | AdaptNLP          | ✓\\checkmark |              |              |              |              |              |              |
| [Cheong et al. (2021)](#bib.bib94 "")            | W-NUT             | ✓\\checkmark |              |              |              |              |              |              |
| [Pratapa and Choudhury (2021)](#bib.bib295 "")   | W-NUT             | ✓\\checkmark | ✓\\checkmark |              |              | ✓\\checkmark | ✓\\checkmark |              |
| [Xia et al. (2022)](#bib.bib418 "")              | LREC              |              |              |              | ✓\\checkmark | ✓\\checkmark |              |              |
| [Mellado and Lignos (2022)](#bib.bib252 "")      | LREC              | ✓\\checkmark |              |              |              |              |              |              |
| [Ostapenko et al. (2022)](#bib.bib269 "")        | ACL               | ✓\\checkmark |              |              |              |              |              |              |

Table 9: \*CL Catalog in European-English.

| Paper |              |              |              |
| ----- | ------------ | ------------ | ------------ |
|       | 3            | 1            | 2            |
| ACL   |              |              | ✓\\checkmark |
| LREC  | ✓\\checkmark |              |              |
| W-NUT |              |              | ✓\\checkmark |
| WANLP |              | ✓\\checkmark |              |
| LREC  | ✓\\checkmark |              |              |
| LREC  | ✓\\checkmark |              |              |

Table 10: \*CL Catalog in Middle Eastern-English.

Paper

Proceeding

Hindi

Marathi

Konkani

Bengali

Bengali

Nepali

Telugu

Bangla

Gujarati

Punjabi

Tamil

Malayalam

Malayalam

Kannada

intra-word

scripts

111

1

1

12

1

10

7

1

1

2

37

23

1

10

[Joshi (1982)](#bib.bib190 "")

COLING

✓\\checkmark

[Sankoff (1998)](#bib.bib324 "")

COLING

✓\\checkmark

[Bhattacharja (2010)](#bib.bib61 "")

PACLIC

✓\\checkmark

[Diab and Kamboj (2011)](#bib.bib110 "")

ALR

✓\\checkmark

[Dey and Fung (2014)](#bib.bib108 "")

LREC

✓\\checkmark

[Das and Gambäck (2014)](#bib.bib105 "")

ICON

✓\\checkmark

✓\\checkmark

[Vyas et al. (2014)](#bib.bib392 "")

EMNLP

✓\\checkmark

[Jhamtani et al. (2014)](#bib.bib186 "")

PACLIC

✓\\checkmark

[Barman et al. (2014a)](#bib.bib49 "")

CALCS

✓\\checkmark

✓\\checkmark

[Solorio et al. (2014)](#bib.bib349 "")

CALCS

✓\\checkmark

[Chittaranjan et al. (2014)](#bib.bib96 "")

CALCS

✓\\checkmark

[Lin et al. (2014)](#bib.bib223 "")

CALCS

✓\\checkmark

[Jain and Bhat (2014)](#bib.bib180 "")

CALCS

✓\\checkmark

[King et al. (2014)](#bib.bib200 "")

CALCS

✓\\checkmark

[Bali et al. (2014)](#bib.bib37 "")

CALCS

✓\\checkmark

[Barman et al. (2014b)](#bib.bib50 "")

CALCS

✓\\checkmark

[Shrestha (2014)](#bib.bib337 "")

CALCS

✓\\checkmark

[Jamatia et al. (2015)](#bib.bib181 "")

RANLP

✓\\checkmark

[Sequiera et al. (2015)](#bib.bib328 "")

ICON

✓\\checkmark

[Gupta et al. (2016)](#bib.bib158 "")

ICON

✓\\checkmark

[Asnani and Pawar (2016)](#bib.bib32 "")

ICON

✓\\checkmark

[Begum et al. (2016)](#bib.bib57 "")

LREC

✓\\checkmark

[Gambäck and Das (2016)](#bib.bib131 "")

LREC

✓\\checkmark

✓\\checkmark

[Sitaram and Black (2016)](#bib.bib345 "")

LREC

✓\\checkmark

[Sharma et al. (2016)](#bib.bib332 "")

HLT-NAACL

✓\\checkmark

[Joshi et al. (2016)](#bib.bib189 "")

COLING

✓\\checkmark

[Çetinoğlu et al. (2016)](#bib.bib77 "")

CALCS

✓\\checkmark

[Chanda et al. (2016b)](#bib.bib87 "")

CALCS

✓\\checkmark

[Ghosh et al. (2016)](#bib.bib142 "")

CALCS

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Pratapa and Choudhury (2017)](#bib.bib294 "")

ICON

✓\\checkmark

[Phadte and Thakkar (2017)](#bib.bib285 "")

ICON

✓\\checkmark

[Pandey et al. (2017)](#bib.bib275 "")

ICON

✓\\checkmark

[Bhat et al. (2017)](#bib.bib60 "")

EACL

✓\\checkmark

[Banerjee et al. (2018)](#bib.bib42 "")

COLING

✓\\checkmark

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Gundapu and Mamidi (2018)](#bib.bib150 "")

PACLIC

✓\\checkmark

[Ball and Garrette (2018)](#bib.bib38 "")

EMNLP

✓\\checkmark

[Khandelwal et al. (2018)](#bib.bib196 "")

LREC

✓\\checkmark

[Kumar et al. (2018)](#bib.bib207 "")

LREC

✓\\checkmark

[Pandey et al. (2018)](#bib.bib276 "")

LREC

✓\\checkmark

[Hartmann et al. (2018)](#bib.bib167 "")

LREC

✓\\checkmark

[Gupta et al. (2018a)](#bib.bib156 "")

LREC

✓\\checkmark

✓\\checkmark

[Mandal and Nanmaran (2018)](#bib.bib244 "")

W-NUT

✓\\checkmark

[Mandal and Singh (2018)](#bib.bib245 "")

W-NUT

✓\\checkmark

✓\\checkmark

[Vijay et al. (2018)](#bib.bib386 "")

SRW

✓\\checkmark

[Gupta et al. (2018b)](#bib.bib159 "")

CoNLL

✓\\checkmark

[Singh et al. (2018)](#bib.bib343 "")

SRW

✓\\checkmark

[Chandu et al. (2019)](#bib.bib88 "")

CALCS

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Sivasankaran et al. (2018)](#bib.bib348 "")

CALCS

✓\\checkmark

[Chandu et al. (2019)](#bib.bib88 "")

CALCS

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Gupta et al. (2018c)](#bib.bib160 "")

CALCS

✓\\checkmark

[Mave et al. (2018)](#bib.bib250 "")

CALCS

✓\\checkmark

[Rallabandi et al. (2018)](#bib.bib305 "")

CALCS

✓\\checkmark

[Bawa et al. (2018)](#bib.bib54 "")

CALCS

✓\\checkmark

[Chandu et al. (2018)](#bib.bib89 "")

CALCS

✓\\checkmark

[Jain et al. (2018)](#bib.bib179 "")

CALCS

✓\\checkmark

[Mukherjee et al. (2019)](#bib.bib256 "")

ICON

✓\\checkmark

[Raha et al. (2019)](#bib.bib301 "")

ICON

✓\\checkmark

[Shah et al. (2019)](#bib.bib330 "")

AnnoNLP

✓\\checkmark

[Ghosh et al. (2019)](#bib.bib143 "")

TLT, SyntaxFest

✓\\checkmark

✓\\checkmark

[Srirangam et al. (2019)](#bib.bib362 "")

SRW

✓\\checkmark

[Lal et al. (2019)](#bib.bib208 "")

SRW

✓\\checkmark

[Chakravarthi (2020)](#bib.bib78 "")

PEOPLES

✓\\checkmark

✓\\checkmark

[Singh and Lefever (2020)](#bib.bib344 "")

ICON

✓\\checkmark

[Bansal et al. (2020a)](#bib.bib43 "")

ICON

✓\\checkmark

[Bansal et al. (2020c)](#bib.bib45 "")

ACL

✓\\checkmark

[Khanuja et al. (2020b)](#bib.bib198 "")

ACL

✓\\checkmark

[Aguilar and Solorio (2020)](#bib.bib17 "")

ACL

✓\\checkmark

✓\\checkmark

[Rani et al. (2020)](#bib.bib309 "")

TRAC

✓\\checkmark

[Pant and Dadu (2020)](#bib.bib277 "")

SRW

✓\\checkmark

[Khanuja et al. (2020a)](#bib.bib197 "")

CALCS

✓\\checkmark

[Singh and Lefever (2020)](#bib.bib344 "")

CALCS

✓\\checkmark

[Srinivasan et al. (2020)](#bib.bib361 "")

CALCS

✓\\checkmark

[Patwa et al. (2020)](#bib.bib282 "")

SemEval

✓\\checkmark

[Liu et al. (2020)](#bib.bib227 "")

SemEval

✓\\checkmark

[Aparaschivei et al. (2020)](#bib.bib29 "")

SemEval

✓\\checkmark

[Kong et al. (2020)](#bib.bib204 "")

SemEval

✓\\checkmark

[Baruah et al. (2020)](#bib.bib53 "")

SemEval

✓\\checkmark

[Srinivasan (2020)](#bib.bib360 "")

SemEval

✓\\checkmark

[Goswami et al. (2020)](#bib.bib148 "")

SemEval

✓\\checkmark

[Kumar et al. (2020)](#bib.bib205 "")

SemEval

✓\\checkmark

[Advani et al. (2020)](#bib.bib10 "")

SemEval

✓\\checkmark

[dos Santos Neto et al. (2020)](#bib.bib114 "")

SemEval

✓\\checkmark

[Gundapu and Mamidi (2020)](#bib.bib151 "")

SemEval

✓\\checkmark

[Srivastava and Vardhan (2020)](#bib.bib364 "")

SemEval

✓\\checkmark

[Srivastava and Singh (2020a)](#bib.bib366 "")

SemEval

✓\\checkmark

[Parikh et al. (2020)](#bib.bib280 "")

SemEval

✓\\checkmark

[Javdan et al. (2020)](#bib.bib183 "")

SemEval

✓\\checkmark

[Garain et al. (2020)](#bib.bib134 "")

SemEval

✓\\checkmark

[Banerjee et al. (2020)](#bib.bib41 "")

SemEval

✓\\checkmark

Table 11: \*CL Catalog in South Asian-English (1).

[Wu et al. (2020)](#bib.bib414 "")

SemEval

✓\\checkmark

[Baroi et al. (2020)](#bib.bib52 "")

SemEval

✓\\checkmark

[Gopalan and Hopkins (2020)](#bib.bib147 "")

SemEval

✓\\checkmark

[Malte et al. (2020)](#bib.bib243 "")

SemEval

✓\\checkmark

[Zaharia et al. (2020)](#bib.bib441 "")

SemEval

✓\\checkmark

[Zhu et al. (2020)](#bib.bib451 "")

SemEval

✓\\checkmark

[Parekh et al. (2020)](#bib.bib279 "")

CoNLL

✓\\checkmark

[Chakravarthi et al. (2020a)](#bib.bib79 "")

SLTU & CCURL

✓\\checkmark

[Chakravarthi et al. (2020b)](#bib.bib80 "")

SLTU & CCURL

✓\\checkmark

[Gupta et al. (2020)](#bib.bib157 "")

Findings of EMNLP

✓\\checkmark

✓\\checkmark

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Makhija et al. (2020)](#bib.bib240 "")

COLING

✓\\checkmark

[Aguilar et al. (2020)](#bib.bib15 "")

LREC

✓\\checkmark

✓\\checkmark

[Chatterjere et al. (2020)](#bib.bib93 "")

LREC

✓\\checkmark

[Aggarwal et al. (2020)](#bib.bib13 "")

W-NUT

✓\\checkmark

[Srivastava and Singh (2020b)](#bib.bib367 "")

W-NUT

✓\\checkmark

[Chakravarthy et al. (2020)](#bib.bib81 "")

W-NUT

✓\\checkmark

[Chinnappa (2021)](#bib.bib95 "")

LTEDI

✓\\checkmark

✓\\checkmark

[Dave et al. (2021)](#bib.bib106 "")

LTEDI

✓\\checkmark

✓\\checkmark

[Hossain et al. (2021)](#bib.bib172 "")

LTEDI

✓\\checkmark

✓\\checkmark

[Balouchzahi et al. (2021)](#bib.bib39 "")

LTEDI

✓\\checkmark

✓\\checkmark

[Agarwal and Narula (2021)](#bib.bib11 "")

SRW

✓\\checkmark

[Agarwal et al. (2021)](#bib.bib12 "")

NLP4ConvAI

✓\\checkmark

[Garg et al. (2021)](#bib.bib135 "")

Eval4NLP

✓\\checkmark

[Srivastava and Singh (2021b)](#bib.bib369 "")

Eval4NLP

✓\\checkmark

[Tarunesh et al. (2021)](#bib.bib378 "")

ACL

✓\\checkmark

[Srivastava and Singh (2021a)](#bib.bib368 "")

CALCS

✓\\checkmark

[Gautam et al. (2021a)](#bib.bib139 "")

CALCS

✓\\checkmark

[Dowlagar and Mamidi (2021a)](#bib.bib115 "")

CALCS

✓\\checkmark

[Appicharla et al. (2021)](#bib.bib30 "")

CALCS

✓\\checkmark

[Jawahar et al. (2021)](#bib.bib184 "")

CALCS

✓\\checkmark

[Gautam et al. (2021b)](#bib.bib140 "")

CALCS

✓\\checkmark

[Gupta et al. (2021b)](#bib.bib154 "")

CALCS

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Jayanthi et al. (2021)](#bib.bib185 "")

CALCS

✓\\checkmark

✓\\checkmark

[Parikh and Solorio (2021)](#bib.bib281 "")

CALCS

✓\\checkmark

[Winata et al. (2021a)](#bib.bib406 "")

CALCS

✓\\checkmark

[Mehnaz et al. (2021)](#bib.bib251 "")

EMNLP

✓\\checkmark

[Prasad et al. (2021)](#bib.bib292 "")

MRL

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Gupta et al. (2021a)](#bib.bib153 "")

NAACL

✓\\checkmark

[Ravikiran and Annamalai (2021)](#bib.bib311 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

[Mahata et al. (2021)](#bib.bib239 "")

DravidianLangTech

✓\\checkmark

[Saumya et al. (2021)](#bib.bib326 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Mandalam and Sharma (2021)](#bib.bib246 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

[Dowlagar and Mamidi (2021b)](#bib.bib116 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

[Gupta et al. (2021c)](#bib.bib155 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

[Balouchzahi and Shashirekha (2021)](#bib.bib40 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

[Dowlagar and Mamidi (2021c)](#bib.bib117 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Li (2021)](#bib.bib218 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Andrew (2021)](#bib.bib26 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Vasantharajan and Thayasivam (2021)](#bib.bib385 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Huang and Bai (2021)](#bib.bib174 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Sharif et al. (2021)](#bib.bib331 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Bharathi et al. (2021)](#bib.bib59 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Balouchzahi et al. (2021)](#bib.bib39 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Rajalakshmi et al. (2021)](#bib.bib302 "")

DravidianLangTech

✓\\checkmark

[Khan et al. (2021)](#bib.bib195 "")

Findings of EMNLP

✓\\checkmark

[Chopra et al. (2021)](#bib.bib98 "")

Findings of EMNLP

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Srivastava and Singh (2021c)](#bib.bib370 "")

INLG

✓\\checkmark

[Santy et al. (2021)](#bib.bib325 "")

AdaptNLP

✓\\checkmark

[Wadhawan and Aggarwal (2021)](#bib.bib393 "")

WASSA

✓\\checkmark

[Priyanshu et al. (2021)](#bib.bib297 "")

W-NUT

✓\\checkmark

[Dutta (2022)](#bib.bib121 "")

DCLRL

✓\\checkmark

[Biradar and Saumya (2022)](#bib.bib62 "")

DravidianLangTech

✓\\checkmark

[Swaminathan et al. (2022)](#bib.bib374 "")

DravidianLangTech

✓\\checkmark

[SR et al. (2022)](#bib.bib358 "")

DravidianLangTech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Ravikiran and Chakravarthi (2022)](#bib.bib312 "")

DravidianLangTech

✓\\checkmark

[Ravikiran et al. (2022)](#bib.bib313 "")

DravidianLangTech

✓\\checkmark

[Nayak and Joshi (2022)](#bib.bib261 "")

WILDRE-6

✓\\checkmark

[Gautam (2022)](#bib.bib138 "")

WILDRE-6

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Sonu et al. (2022)](#bib.bib353 "")

WILDRE-6

✓\\checkmark

Table 12: \*CL Catalog in South Asian-English (2).

| Paper  |              |              |              |
| ------ | ------------ | ------------ | ------------ |
|        | 1            | 2            | 2            |
| PACLIC |              | ✓\\checkmark |              |
| CALCS  |              |              | ✓\\checkmark |
| LREC   | ✓\\checkmark |              |              |
| PACLIC |              |              | ✓\\checkmark |
| LREC   |              |              | ✓\\checkmark |

Table 13: \*CL Catalog in South East Asian-English.

| Paper       |              |              |              |              |              |              |              |
| ----------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|             | 1            | 15           | 10           | 2            | 2            | 1            | 1            |
|             |              |              | ✓\\checkmark |              |              |              |              |
| CIPS-SIGHAN |              |              |              | ✓\\checkmark |              |              |              |
| COLING      |              | ✓\\checkmark |              |              | ✓\\checkmark |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| LREC        |              | ✓\\checkmark |              |              |              |              |              |
| LREC        | ✓\\checkmark |              |              |              |              |              |              |
| LREC        |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| CALCS       |              |              | ✓\\checkmark |              |              |              |              |
| LREC        |              | ✓\\checkmark |              |              | ✓\\checkmark | ✓\\checkmark |              |
| W-NUT       |              |              |              |              |              |              | ✓\\checkmark |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| LREC        |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |
| CALCS       |              | ✓\\checkmark |              |              |              |              |              |

Table 14: \*CL Catalog in Dialect.

| Paper             |              |              |              |              |              |
| ----------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
|                   | 1            | 1            | 1            | 1            | 3            |
| CALCS             |              | ✓\\checkmark |              |              |              |
| LREC              |              |              |              |              | ✓\\checkmark |
| AdaptNLP          |              |              |              |              | ✓\\checkmark |
| BioNLP            |              |              | ✓\\checkmark |              |              |
| Findings of NAACL | ✓\\checkmark |              |              |              | ✓\\checkmark |
| SIGUL             |              |              |              | ✓\\checkmark |              |

Table 15: \*CL Catalog in Two Languages in the same family.

| Paper                |              |              |              |              |              |              |              |              |              |
| -------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|                      |              | intra-word   |              |              |              |              |              |              | intra-word   |
|                      | 1            | 1            | 7            | 1            | 2            | 2            | 1            | 1            | 1            |
| COLING               |              |              |              |              | ✓\\checkmark |              |              |              |              |
| CALCS                |              |              |              |              |              | ✓\\checkmark |              |              |              |
| LREC                 |              |              |              |              |              | ✓\\checkmark |              |              |              |
| LREC                 |              |              | ✓\\checkmark |              |              |              |              |              |              |
| CALCS                |              |              | ✓\\checkmark |              |              |              |              |              |              |
| LREC                 |              |              |              |              |              |              | ✓\\checkmark |              |              |
| LREC                 |              |              |              | ✓\\checkmark |              |              |              |              |              |
| TLT, SyntaxFest 2019 |              |              | ✓\\checkmark |              |              |              |              |              |              |
| NAACL                |              |              | ✓\\checkmark |              |              |              |              |              | ✓\\checkmark |
| CALCS                |              |              | ✓\\checkmark |              |              |              |              |              |              |
| CALCS                | ✓\\checkmark |              |              |              |              |              |              |              |              |
| ICNLSP               |              |              |              |              | ✓\\checkmark |              |              |              |              |
| LREC                 |              |              |              |              |              |              |              | ✓\\checkmark |              |
| Findings of NAACL    |              |              | ✓\\checkmark |              |              |              |              |              |              |
| EURALI               |              | ✓\\checkmark | ✓\\checkmark |              |              |              |              |              |              |

Table 16: \*CL Catalog in different family.

| Paper |              |              |              |              |              |              |
| ----- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|       | 1            | 1            | 1            | 1            | 1            | 1            |
| LREC  |              |              |              |              |              | ✓\\checkmark |
| CALCS |              |              | ✓\\checkmark |              |              |              |
| CALCS |              | ✓\\checkmark |              |              |              |              |
| EMNLP |              |              |              |              | ✓\\checkmark |              |
| CALCS |              |              |              |              |              |              |
| LREC  |              |              |              | ✓\\checkmark |              |              |
| ELRA  | ✓\\checkmark |              |              |              |              |              |

Table 17: \*CL Catalog in Trilingual.

| Paper   |              |                       |                       |                         |                       |              |              |                             |                   |
| ------- | ------------ | --------------------- | --------------------- | ----------------------- | --------------------- | ------------ | ------------ | --------------------------- | ----------------- |
|         |              | English-Farsi-German- | Latin, French, Greek, | Modern Standard Telugu, | local Algerian Arabic |              | Latin        | local Arabic varieties,     | Taiwanese-English |
|         |              | Hindi-Korean-Russian- | Italian, Hebrew       | English, Hindi, Urdu,   |                       |              |              | Berber, French, and English |                   |
|         |              | Spanish-Turkish       |                       |                         |                       |              |              |                             |                   |
|         | 10           | 4                     | 1                     | 1                       | 2                     | 3            | 1            | 1                           | 1                 |
| ROCLING |              |                       |                       |                         |                       |              |              |                             | ✓\\checkmark      |
| ACL     | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| CALCS   |              |                       |                       |                         |                       |              | ✓\\checkmark |                             |                   |
| CALCS   | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| CALCS   |              |                       |                       |                         |                       | ✓\\checkmark |              |                             |                   |
| EMNLP   | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| CALCS   |              |                       |                       |                         |                       | ✓\\checkmark |              |                             |                   |
| CALCS   | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| CoNLL   | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| ACL     | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| ACL     | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| ROCLING | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| CALCS   |              |                       |                       |                         | ✓\\checkmark          |              |              | ✓\\checkmark                |                   |
| LREC    |              |                       |                       |                         | ✓\\checkmark          |              |              |                             |                   |
| CALCS   |              |                       |                       | ✓\\checkmark            |                       |              |              |                             |                   |
| SCiL    | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| EMNLP   |              |                       |                       |                         |                       | ✓\\checkmark |              |                             |                   |
| W-NUT   | ✓\\checkmark |                       |                       |                         |                       |              |              |                             |                   |
| SemEval |              | ✓\\checkmark          |                       |                         |                       |              |              |                             |                   |
| COLING  |              | ✓\\checkmark          |                       |                         |                       |              |              |                             |                   |
| LREC    |              |                       | ✓\\checkmark          |                         |                       |              |              |                             |                   |
| SemEval |              | ✓\\checkmark          |                       |                         |                       |              |              |                             |                   |
| SemEval |              | ✓\\checkmark          |                       |                         |                       |              |              |                             |                   |

Table 18: \*CL Catalog in 4+ languages.

| Paper                                                   | Proceeding  | isiZulu      | isiXhosa     | Setsawa      | Sesotho      | Sotho        |
| ------------------------------------------------------- | ----------- | ------------ | ------------ | ------------ | ------------ | ------------ |
|                                                         |             |              |              |              |              |              |
| [Niesler and de Wet (2008)](#bib.bib264 "")             | Odyssey     | ✓\\checkmark | ✓\\checkmark |              |              |              |
| [Mabokela et al. (2014)](#bib.bib237 "")                | SLTU        |              |              |              |              | ✓\\checkmark |
| [van der Westhuizen and Niesler (2017)](#bib.bib384 "") | Interspeech | ✓\\checkmark |              |              |              |              |
| [Yılmaz et al. (2018)](#bib.bib429 "")                  | Interspeech | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |              |
| [Biswas et al. (2018a)](#bib.bib64 "")                  | Interspeech | ✓\\checkmark |              |              |              |              |
| [Biswas et al. (2018b)](#bib.bib65 "")                  | SLTU        | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |              |
| [Biswas et al. (2019)](#bib.bib66 "")                   | Interspeech | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark | ✓\\checkmark |              |

Table 19: ISCA Catalog in African-English.

| Paper                                      | Proceeding  | Chinese      | Cantonese    | Korean       | Japanese     |
| ------------------------------------------ | ----------- | ------------ | ------------ | ------------ | ------------ |
|                                            |             |              |              |              |              |
| [Fu and Shen (2000)](#bib.bib128 "")       | ISCSLP      | ✓\\checkmark |              |              |              |
| [Kojima and Tanaka (2003)](#bib.bib203 "") | Eurospeech  |              |              |              | ✓\\checkmark |
| [You et al. (2004)](#bib.bib433 "")        | ISCSLP      | ✓\\checkmark |              |              |              |
| [Chan et al. (2004)](#bib.bib85 "")        | ISCSLP      |              | ✓\\checkmark |              |              |
| [Chan et al. (2005)](#bib.bib83 "")        | Interspeech |              | ✓\\checkmark |              |              |
| [Ren et al. (2005)](#bib.bib314 "")        | Interspeech | ✓\\checkmark |              |              |              |
| [Chan et al. (2006)](#bib.bib84 "")        | Interspeech |              | ✓\\checkmark |              |              |
| [Liang et al. (2007)](#bib.bib219 "")      | SSW         | ✓\\checkmark |              |              |              |
| [White et al. (2008)](#bib.bib401 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Qian et al. (2008)](#bib.bib298 "")       | ISCSLP      | ✓\\checkmark |              |              |              |
| [Gu et al. (2008)](#bib.bib149 "")         | ISCSLP      |              | ✓\\checkmark |              |              |
| [Zhang and Tao (2008)](#bib.bib448 "")     | ISCSLP      | ✓\\checkmark |              |              |              |
| [Cao et al. (2009)](#bib.bib73 "")         | Interspeech |              | ✓\\checkmark |              |              |
| [Shuang et al. (2010)](#bib.bib339 "")     | Interspeech | ✓\\checkmark |              |              |              |
| [He et al. (2012)](#bib.bib168 "")         | Interspeech | ✓\\checkmark |              |              |              |
| [Liang et al. (2013)](#bib.bib220 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Li and Fung (2013)](#bib.bib215 "")       | Interspeech | ✓\\checkmark |              |              |              |
| [Xue et al. (2019)](#bib.bib422 "")        | Interspeech | ✓\\checkmark |              |              |              |
| [Gao et al. (2019)](#bib.bib133 "")        | Interspeech | ✓\\checkmark |              |              |              |
| [Zhang et al. (2019)](#bib.bib444 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Lu et al. (2020)](#bib.bib231 "")         | Interspeech | ✓\\checkmark |              |              |              |
| [Hu et al. (2020)](#bib.bib173 "")         | Interspeech | ✓\\checkmark |              |              |              |
| [Fu et al. (2020)](#bib.bib129 "")         | Interspeech | ✓\\checkmark |              |              |              |
| [Wang et al. (2020)](#bib.bib395 "")       | Interspeech |              |              | ✓\\checkmark |              |
| [Zhang et al. (2020)](#bib.bib443 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Chandu and Black (2020)](#bib.bib90 "")   | Interspeech | ✓\\checkmark |              |              |              |
| [Zhao et al. (2020)](#bib.bib449 "")       | Interspeech | ✓\\checkmark |              |              |              |
| [Zhang et al. (2021a)](#bib.bib445 "")     | Interspeech | ✓\\checkmark |              |              |              |
| [Shen and Guo (2022)](#bib.bib334 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Ye et al. (2022)](#bib.bib424 "")         | Interspeech | ✓\\checkmark |              |              |              |
| [Tian et al. (2022)](#bib.bib381 "")       | Interspeech | ✓\\checkmark |              |              |              |
| [Song et al. (2022)](#bib.bib352 "")       | Interspeech | ✓\\checkmark |              |              |              |
| [Zhang et al. (2022)](#bib.bib446 "")      | Interspeech | ✓\\checkmark |              |              |              |
| [Li et al. (2022)](#bib.bib212 "")         | Interspeech | ✓\\checkmark |              |              |              |

Table 20: ISCA Catalog in East Asian-English.

| Paper                                                       | Proceeding    | Spanish      | French       | German       | Maltese      |
| ----------------------------------------------------------- | ------------- | ------------ | ------------ | ------------ | ------------ |
| [Pfister and Romsdorfer (2003)](#bib.bib284 "")             | Eurospeech    |              |              | ✓\\checkmark |              |
| [Romsdorfer and Pfister (2005)](#bib.bib316 "")             | Interspeech   |              | ✓\\checkmark |              |              |
| [Rosner and Farrugia (2007)](#bib.bib318 "")                | Interspeech   |              |              |              | ✓\\checkmark |
| [Piccinini and Garellek (2014)](#bib.bib286 "")             | SpeechProsody | ✓\\checkmark |              |              |              |
| [Sitaram et al. (2016)](#bib.bib347 "")                     | SSW           |              |              | ✓\\checkmark |              |
| [Soto and Hirschberg (2017)](#bib.bib355 "")                | Interspeech   | ✓\\checkmark |              |              |              |
| [Ramanarayanan and Suendermann-Oeft (2017)](#bib.bib306 "") | Interspeech   | ✓\\checkmark |              |              |              |
| [Guzmán et al. (2017)](#bib.bib163 "")                      | Interspeech   | ✓\\checkmark |              |              |              |
| [Bullock et al. (2018b)](#bib.bib70 "")                     | Interspeech   | ✓\\checkmark |              |              |              |
| [Soto et al. (2018)](#bib.bib354 "")                        | Interspeech   | ✓\\checkmark |              |              |              |
| [Soto and Hirschberg (2019)](#bib.bib357 "")                | Interspeech   | ✓\\checkmark |              |              |              |
| [Chandu and Black (2020)](#bib.bib90 "")                    | Interspeech   | ✓\\checkmark |              |              |              |

Table 21: ISCA Catalog in European-English.

| Paper       | Proceeding   | Modern Standard Arabic |
| ----------- | ------------ | ---------------------- |
| Interspeech | ✓\\checkmark |                        |
| Interspeech | ✓\\checkmark |                        |
| Interspeech | ✓\\checkmark |                        |

Table 22: ISCA Catalog in Middle Eastern-English.

Paper

Proceeding

Hindi

Marathi

Bengali

Telugu

Gujarati

Tamil

Malayalam

Kannada

[Sitaram et al. (2016)](#bib.bib347 "")

SSW

✓\\checkmark

[Ramanarayanan and Suendermann-Oeft (2017)](#bib.bib306 "")

Interspeech

✓\\checkmark

[Ganji and Sinha (2018)](#bib.bib132 "")

Interspeech

✓\\checkmark

[Rao et al. (2018)](#bib.bib310 "")

Interspeech

✓\\checkmark

[Thomas et al. (2018a)](#bib.bib379 "")

Interspeech

✓\\checkmark

✓\\checkmark

[Srivastava and Sitaram (2018)](#bib.bib365 "")

Interspeech

✓\\checkmark

[Rambabu and Gangashetty (2018)](#bib.bib307 "")

SLTU

✓\\checkmark

[Taneja et al. (2019)](#bib.bib377 "")

Interspeech

✓\\checkmark

[Rallabandi and Black (2019)](#bib.bib304 "")

Interspeech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Prakash et al. (2019)](#bib.bib291 "")

SSW

✓\\checkmark

[Sharma et al. (2020)](#bib.bib333 "")

Interspeech

✓\\checkmark

[Manghat et al. (2020)](#bib.bib247 "")

Interspeech

✓\\checkmark

[Bansal et al. (2020b)](#bib.bib44 "")

Interspeech

✓\\checkmark

[Chandu and Black (2020)](#bib.bib90 "")

Interspeech

✓\\checkmark

[Kumar et al. (2021)](#bib.bib206 "")

Interspeech

✓\\checkmark

✓\\checkmark

[Liu et al. (2021)](#bib.bib226 "")

Interspeech

✓\\checkmark

✓\\checkmark

✓\\checkmark

[Diwan et al. (2021)](#bib.bib111 "")

Interspeech

✓\\checkmark

✓\\checkmark

[Klejch et al. (2021)](#bib.bib201 "")

Interspeech

✓\\checkmark

✓\\checkmark

[Wiesner et al. (2021)](#bib.bib402 "")

Interspeech

✓\\checkmark

✓\\checkmark

[Antony et al. (2022)](#bib.bib28 "")

Interspeech

✓\\checkmark

[Manghat et al. (2022)](#bib.bib248 "")

Interspeech

✓\\checkmark

Table 23: ISCA Catalog in South Asian-English.

| Paper       | Proceeding   | Malay |
| ----------- | ------------ | ----- |
| SLTU        | ✓\\checkmark |       |
| Interspeech | ✓\\checkmark |       |
| Interspeech | ✓\\checkmark |       |

Table 24: ISCA Catalog in South East Asian-English.

| Paper       | Proceeding   | Chinese-Taiwanese |
| ----------- | ------------ | ----------------- |
| Interspeech | ✓\\checkmark |                   |

Table 25: ISCA Catalog in Language with Dialects.

| Paper       | Proceeding   | Frisian-Dutch | Russian-Ukrainan |
| ----------- | ------------ | ------------- | ---------------- |
| Interspeech |              | ✓\\checkmark  |                  |
| Interspeech | ✓\\checkmark |               |                  |
| Interspeech | ✓\\checkmark |               |                  |
| Interspeech | ✓\\checkmark |               |                  |
| Interspeech | ✓\\checkmark |               |                  |
| SLTU        | ✓\\checkmark |               |                  |
| Interspeech |              | ✓\\checkmark  |                  |
| Interspeech |              | ✓\\checkmark  |                  |

Table 26: ISCA Catalog in Two Languages in the same family.

| Paper       | Proceeding   | Kazakh-Russian | Hindi-Tamil  | French-Arabic |
| ----------- | ------------ | -------------- | ------------ | ------------- |
| Interspeech |              |                | ✓\\checkmark |               |
| Interspeech |              | ✓\\checkmark   |              |               |
| Interspeech |              |                | ✓\\checkmark |               |
| Interspeech |              |                | ✓\\checkmark |               |
| Interspeech |              |                | ✓\\checkmark |               |
| Interspeech | ✓\\checkmark |                |              |               |

Table 27: ISCA Catalog in Two Languages in different families.

| Paper         | Proceeding   | Italian-German-English | Kiswahili-Shen-English |
| ------------- | ------------ | ---------------------- | ---------------------- |
| Interspeech   | ✓\\checkmark |                        |                        |
| SpeechProsody |              | ✓\\checkmark           |                        |

Table 28: ISCA Catalog in Trilingual.

Paper

Proceeding

SEA Mandarin-English

African Languages-English

Indian Languages-English

Others

17

1

1

7

[Badino et al. (2004)](#bib.bib34 "")

Interspeech

✓\\checkmark

[Oria and Vetek (2004)](#bib.bib268 "")

Interspeech

✓\\checkmark

[Marcadet et al. (2005)](#bib.bib249 "")

Interspeech

✓\\checkmark

[Romsdorfer and Pfister (2006)](#bib.bib317 "")

ML

✓\\checkmark

[Lyu et al. (2010b)](#bib.bib234 "")

Interspeech

✓\\checkmark

[Imseng et al. (2010)](#bib.bib177 "")

Interspeech

✓\\checkmark

[Weiner et al. (2012b)](#bib.bib400 "")

SLTU

✓\\checkmark

[Adel et al. (2014c)](#bib.bib4 "")

Interspeech

✓\\checkmark

[Adel et al. (2014b)](#bib.bib3 "")

Interspeech

✓\\checkmark

[Giwa and Davel (2014)](#bib.bib144 "")

Interspeech

✓\\checkmark

[Adel et al. (2014a)](#bib.bib2 "")

SLTU

✓\\checkmark

[Rallabandi and Black (2017)](#bib.bib303 "")

Interspeech

✓\\checkmark

[Chandu et al. (2017)](#bib.bib91 "")

Interspeech

✓\\checkmark

[Garg et al. (2018b)](#bib.bib137 "")

Interspeech

✓\\checkmark

[Xu et al. (2018)](#bib.bib419 "")

Interspeech

✓\\checkmark

[Guo et al. (2018)](#bib.bib152 "")

Interspeech

✓\\checkmark

[Chang et al. (2019)](#bib.bib92 "")

Interspeech

✓\\checkmark

[Khassanov et al. (2019)](#bib.bib199 "")

Interspeech

✓\\checkmark

[Lee et al. (2019b)](#bib.bib211 "")

Interspeech

✓\\checkmark

[Zeng et al. (2019)](#bib.bib442 "")

Interspeech

✓\\checkmark

[Hu et al. (2020)](#bib.bib173 "")

Interspeech

✓\\checkmark

[Li and Vu (2020)](#bib.bib213 "")

Interspeech

✓\\checkmark

[Zhou et al. (2020)](#bib.bib450 "")

Interspeech

✓\\checkmark

[Nekvinda and Dušek (2020)](#bib.bib262 "")

Interspeech

✓\\checkmark

[Qiu et al. (2020)](#bib.bib299 "")

Interspeech

✓\\checkmark

[Liu et al. (2021)](#bib.bib226 "")

Interspeech

✓\\checkmark

Table 29: ISCA Catalog in 4+.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")