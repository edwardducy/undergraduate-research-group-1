# HUMSET: Dataset of Multilingual Information Extraction and Classification for Humanitarian Crisis Response

Selim Fekih<sup>1</sup> Nicolò Tamagnone<sup>1</sup> Benjamin Minixhofer<sup>3</sup> Ranjan Shrestha<sup>2</sup> Ximena Contla<sup>1</sup> Ewan Oglethorpe<sup>1</sup> Navid Rekabsaz<sup>3</sup>

<sup>1</sup>Data Friendly Space <sup>2</sup>ToggleCorp Solutions

<sup>3</sup>Johannes Kepler University Linz, LIT AI Lab, Austria

{selim, nico, ximena, ewan}@datafriendlyspace.org

ranjan.shrestha@togglecorp.com

{benjamin.minixhofer, navid.rekabsaz}@jku.at

## Abstract

Timely and effective response to humanitarian crises requires quick and accurate analysis of large amounts of text data – a process that can highly benefit from expert-assisted NLP systems trained on validated and annotated data in the humanitarian response domain. To enable creation of such NLP systems, we introduce and release HUMSET, a novel and rich multilingual dataset of humanitarian response documents annotated by experts in the humanitarian response community. The dataset provides documents in three languages (English, French, Spanish) and covers a variety of humanitarian crises from 2018 to 2021 across the globe. For each document, HUMSET provides selected snippets (entries) as well as assigned classes to each entry annotated using common humanitarian information analysis frameworks. H S also provides novel and challenging entry extraction and multi-label entry classification tasks. In this paper, we take a first step towards approaching these tasks and conduct a set of experiments on Pre-trained Language Models (PLM) to establish strong baselines for future research in this domain. The dataset is available at https://blog. thedee .io/humset/

## 1 Introduction

During humanitarian crises caused by reasons ranging from natural disasters, wars ,or epidemics such as COVID-19, a timely and effective humanitarian response highly depends on fast and accurate analysis of relevant data to yield key information. Early in the response phase, namely in the first 72 hours after a disaster strikes, the humanitarian response analysts in international organizations review large amounts of data loosely or strongly relevant to the crisis to gain situational awareness.

A large portion of this data appears in the form of secondary data sources i. e. reports, news, and other forms of text data, and is integral in revealing which type of relief activities to undertake. Analysis in this phase involves extracting key information and organizing it according to sets of pre-defined domain-specific structures and guidelines, referred to as humanitarian analysisframeworks.

While typically only small workforces are available to analyze such information, an automatic document processing system can significantly help analysts save time in the overall humanitarian response cycle. To facilitate such systems, we introduce and release HUMSET, a unique and rich dataset of document analysis in the humanitarian response domain. HUMSET is curated by humanitarian analysts and covers various disasters around the globe that occurred from 2018 to 2021 in 46 humanitarian response projects. The dataset consists of approximately 17K annotated documents in three languages of English, French, and Spanish, originally taken from publicly-available resources.<sup>2</sup> For each document, analysts have identified informative snippets (entries) with respect to common humanitarian frameworks and assigned one or many classes to each entry (details in §2).

HUMSET provides a large dataset for the training and evaluation of entry extraction and classification models, enabling the research and development of further NLP systems in the humanitarian response domain. We take the first step in this direction, by studying the performance of a set of strong baseline models (details in §3). Our released dataset expands the previously provided collection by Yela-Bello et al. (2021) with a more recent and comprehensive set of projects, as well as additional classification labels. Other similar datasets in the humanitarian domain, Imran et al. (2016) present humanannotated Twitter corpora collected during 19 different crises between 2013 and 2015, Alam et al. (2021) provide a combination of various socialmedia crisis-related existing datasets, and Adel and Wang (2020) and later Alharbi and Lee (2021) publish Arabic Twitter classification datasets for crisis events. HUMSET, in contrast to the current resources which mostly originated from social media, is created by humanitarian experts through an annotation process on official documents and news from the most recognized humanitarian agencies, conferring high reliability, continuous updating, and accurate geolocation information.

## 2 HUMSET Dataset

The collection originated from a multiorganizational platform called the Data Entry and Exploration Platform (DEEP),<sup>3</sup> developed and maintained by Data Friendly Space (DFS)<sup>4</sup> The platform facilitates classifying primarily qualitative information with respect to analysis frameworks and allows for collaborative classification and annotation of secondary data. The dataset is available at https://blog.thedeep.io/humset/.

## 2.1 Dataset Overview

HUMSET consists of data used to inform 46 humanitarian response operations across the globe. 24 responses were in Central/South America, 14 in Africa, and 8 in Asia (detailed countries can be found in Table 6 in Appendix).

For each project, documents, referred to as leads, related to a particular humanitarian crisis are collected, analyzed, and annotated. The annotated documents in the dataset mostly consist of recently released information, with 79% of the documents being released in 2020 and 2021 (Table 5 in Appendix), and 90% of all documents being sourced from websites (see Table 4 in Appendix for the most commonly used platforms). Documents are selected from different sources, ranging from official reports by humanitarian organizations to international and national media articles. Overall, documents consist of files in PDF format (70.4%) and HTML pages (29.6%) with an average length of ∼ 2K words. The number of documents analyzed per project varies, ranging from 2 to 2,266.

The relevant snippets of texts, referred to as entries, in each document are annotated by humanitarian experts. The dataset provides an average of ∼10 entries per document, and an average length of ∼ 65 words per entry. Overall, HUMSET is composed of 148,621 tagged entries, selected from 16,857 documents, and in three languages: English (61.3%), French (20.4%) and Spanish (18.3%). The list of projects as well as the number of documents and annotated entries per project is reported in Table 7 in Appendix. Figure 1 shows the distribution of the number of tagged documents per project, as well as the number of tokens per document and entry.

![](images/a87d7d32fd9d41df5f9cdc7d8044d7c3d7c81bb3a2288307a022a49cf7fc8027.jpg)

<details>
<summary>boxplot</summary>

| Metric | Value |
| --- | --- |
| Minimum (Whisker) | ~0 |
| Q1 | ~50 |
| Q2 (Median) | ~180 |
| Q3 | ~400 |
| Maximum (Whisker) | ~700 |
| Outliers | ~1300, ~1450, ~1460, ~1600, ~1750, ~2350 |
</details>

(a)  
![](images/9d1d9c77d780776ac4081dd82d9fc6f67d58e6dd7f5a20dec0e09be30e34558b.jpg)

<details>
<summary>boxplot</summary>

| Metric | Value |
| --- | --- |
| Minimum (Whisker) | ~30 |
| Q1 | ~350 |
| Q2 (Median) | ~600 |
| Q3 | ~1200 |
| Maximum (Whisker) | ~30000 |
| Outliers | >100000 |
</details>

(b)  
![](images/39644757ce6d3854b2fff41cdd1f73c3fd85d4afcef0c223ca42e5c5b53055f2.jpg)

<details>
<summary>boxplot</summary>

| Metric | Value |
| --- | --- |
| Minimum (Whisker) | ~1 |
| Q1 | ~45 |
| Q2 (Median) | ~60 |
| Q3 | ~85 |
| Maximum (Whisker) | ~150 |
| Outliers | >1000 |
</details>

(c)  
Figure 1: (a) Distribution of documents per project. (b) Log-scale distribution of tokens<sup>5</sup> per document. (c) Log-scale distribution of tokens per entry.

## 2.2 Humanitarian Analysis Frameworks and Data Annotation Process

The concept of analyticalframeworks originated in the social sciences (Ragin and Amoroso, 2011), but can be considered foundational and indispensable in numerous research fields. An analytical framework is a set of methodologies and guidelines to facilitate data collection, collation, and analysis, helping to understand what information will be useful and what can be discarded.

In the humanitarian domain, an analytical framework (or analysis framework) not only assists decision-makers to speed up humanitarian response and disaster relief but also enables various groups to share resources (Zhang et al., 2002). When starting a response or project, humanitarian organizations create or more often use an existing analysis framework, which covers the generic but also specific needs of the work. Our data originally contained 11 different frameworks. As there are high similarities across frameworks, we created a common framework, which we refer to as humanitarian analysisframework. This framework covers the framework dimensions of all projects. We build our custom set of tags by mapping the original tags in other frameworks to ours. More specifically, our analysis framework consists of three categories: Sectors (11 tags), Subpillars 1D (33 tags), and Subpillars 2D (18 tags). Pillars/Subpillars 1D, and 2D have a hierarchical structure, consisting of a two-leveled tree hierarchy (Pillars to Subpillars). The list and the number of tags present for each category are reported in Table 1.

<table><tr><td>Categories</td><td>#</td><td>Tags</td></tr><tr><td>Sectors</td><td>11</td><td>Agriculture, Cross-sector, Education, Food Security, Health, Livelihoods, Logistics, Nutrition, Protection, Shelter, WASH (Water, Sanitation &amp; Hygiene)</td></tr><tr><td>Pillars 1D</td><td>7</td><td>Context, COVID-19, Displacement, Humanitarian Access, Information &amp; Communication, Casualties, Shock/Event</td></tr><tr><td>Subpillars 1D</td><td>33</td><td>Details in Table 8 in Appendix</td></tr><tr><td>Pillars 2D</td><td>6</td><td>Capacities &amp; Response, Humanitarian Conditions, Impact, At Risk, Priority Needs, Priority Interventions</td></tr><tr><td>Subpillars 2D</td><td>18</td><td>Details in Table 9 in Appendix</td></tr></table>

Table 1: Overview of humanitarian analysis framework.

For each project, documents relevant to understanding the situation, unmet needs, and underlying factors are captured and uploaded to the DEEP platform. From these sources, entries of text are selected and categorized into an analysis framework. Humanitarian annotators are trained in specific projects to follow analytical standards and thinking to review secondary data.

This process eventually results in annotating and organizing the data according to the humanitarian analysis framework. As the HUMSET dataset is created in a real-world scenario, the distribution of annotated entries is skewed, with 33 tags being present in less than 2% of data. Tables 10, 11, and 12 in Appendix show the detailed number and proportions of the annotated entries in Sectors, Subpillars 1D, and 2D, respectively. Figure 2 in Appendix reports the distribution of tags in dataset.

## 2.3 NLP Tasks

Entry Extraction Task. The first step for humanitarian taggers in analyzing a document is finding entries containing relevant information. A piece of text or information is considered relevant if it meaningfully contains at least one tag present in the given humanitarian analytical framework. Since documents often contain a large amount of information (Figure 1), it is extremely beneficial to automate the process of entry identification, and this is the first task of this research. This can be seen as an extractive summarization task i. e. selecting a subset of passages that contain relevant information from the given document. However, the entries do not necessarily follow the common units of text such as sentence and paragraph and can appear in various lengths. In fact, only 38.8% of entries consist of full sentences, and the rest are snippets that are shorter or longer than sentences. This limits the direct applicability of prior approaches to extractive summarization (Liu and Lapata, 2019; Zhou et al., 2018), and makes the task particularly challenging for NLP research.

Multi-label Entry Classification Task. After selecting the most relevant entries within a document, the next step is to categorize them according to the humanitarian analysis framework (Table 1). An automatic suggestion on which tag to choose from a large number of possibilities can be decisive in speeding up the annotation process. For each category, more than one tag can be assigned to an entry. Hence, we can view this task as multi-label classification.

## 3 Experiments and Results

To conduct a set of baseline experiments on HUM-SET according to the mentioned tasks, we split the data into training, validation, and test sets for all our experiments (80%, 10%, and 10%, respectively). We apply stratified splitting (Szymanski and Kajdanowicz, 2017) to maintain the same distribution of labels for each set. Implementation details of Entry Extraction (Section 3.1) and Entry

<table><tr><td rowspan="2">Model</td><td colspan="2">Sectors</td><td colspan="2">Pillars 1D</td><td colspan="2">Subpillars 1D</td><td colspan="2">Pillars 2D</td><td colspan="2">Subpillars 2D</td></tr><tr><td>Prec.</td><td>F1</td><td>Prec.</td><td>F1</td><td>Prec.</td><td>F1</td><td>Prec.</td><td>F1</td><td>Prec.</td><td>F1</td></tr><tr><td>Random Baseline</td><td>0.09</td><td>0.09</td><td>0.06</td><td>0.06</td><td>0.01</td><td>0.01</td><td>0.13</td><td>0.13</td><td>0.05</td><td>0.05</td></tr><tr><td>FastText</td><td>0.71</td><td>0.61</td><td>0.56</td><td>0.38</td><td>0.58</td><td>0.33</td><td>0.59</td><td>0.45</td><td>0.48</td><td>0.33</td></tr><tr><td> $XtremeDistil_{l6-h256}$ </td><td>0.56</td><td>0.58</td><td>0.35</td><td>0.36</td><td>0.20</td><td>0.20</td><td>0.51</td><td>0.55</td><td>0.28</td><td>0.29</td></tr><tr><td> $XLM-R_{Base}$ </td><td>0.71</td><td>0.73</td><td>0.49</td><td>0.53</td><td>0.45</td><td>0.38</td><td>0.63</td><td>0.63</td><td>0.51</td><td>0.40</td></tr></table>

Table 2: Entry classification results.

Classification (Section 3.2) are available at https: //github.com/the-deep/humset.

<table><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td>LEAD4</td><td>0.32</td><td>0.24</td><td>0.31</td></tr><tr><td>XtremeDistil $_{l6-h256}$ </td><td>0.33</td><td>0.25</td><td>0.33</td></tr><tr><td>XLM-R $_{\text{Base}}$ </td><td>0.42</td><td>0.35</td><td>0.41</td></tr></table>

Table 3: Entry extraction results for ROUGE F1 (R).

## 3.1 Entry Extraction

We evaluate the performance of the entry extraction task using ROUGE-1, ROUGE-2 ,and ROUGE-L F1 score (Lin, 2004). The target text (ground truth) is a concatenation of all relevant entries, and the predicted text is a concatenation of all entries predicted as relevant. We consider a simple heuristic method (LEAD4), as well as Transformerbased (Vaswani et al., 2017) pre-trained language models (PLM) with a multilingual backbone as our baselines as explained in the following:

LEAD4: LEAD-n is a simple baseline where the first n sentences are predicted as being relevant entries. Consistent with prior work (Yela-Bello et al., 2021), we choose $n = 4$ . Transformers: to approach the task using Transformer-based PLM, we formulate the task as a token classification problem. The objective is to distinguish between tokens that are part of relevant entries and tokens which are not. For simplicity, we fine-tune the entire model and do binary classification using a twolayer prediction head on top of the contextualized representation of each token. We conduct our experiments on XtremeDistil <sub>−</sub> (Mukherjee and Hassan Awadallah, 2020) and $\mathbf { X L M - R _ { \mathrm { B a s e } } }$ (Conneau et al., 2019) as the underlying PLM.

The evaluation results of the mentioned methods are reported in Table 3. Among our baselines, the model with $\mathbf { X L M - R _ { \mathrm { B a s e } } }$ shows the best overall performance. However, we should consider these experiments as starting points, and improvements on this task are expected by increasing model capacity and architectural variations.

## 3.2 Entry Classification

We test different multi-label sequence classification models applied to our five categories. We use the Precision and F1-score metrics to assess the performance of the models on each subcategory. We report macro-averages of the metrics, as the tags are unbalanced (see Table 2) and macro-averaging can provide a more nuanced view of the performance especially by supporting the more sparse classes. Finally, we perform threshold tuning of the classification decision boundary with respect to macro-average F1-scores for each label of each category (Pillai et al., 2013). Tuning the threshold is done by finding the optimal results on the validation set, used to make classifications on the test set. We conduct experiments using fastText (Joulin et al., 2016), as well as Transformer-based PLMs as explained below.

fastText: is an Open Source library for text representation and classification that consists of a bag of n-grams representation and a linear classifier. fastText classification is language-agnostic and does not need language-specific pre-trained word vectors, allowing us to train a multilingual classifier as a simple baseline. To handle multiple labels, we trained independent binary classifiers for each label. Transformers: For consistency with the previous task, we fine-tune the same multilingual PLMs and add a dense layer on top for multi-label classification.

Table 2 reports the evaluation results on the mentioned baseline models. For comparison, a random baseline is also reported. The random baseline is a stratified random classification, created based on the distribution of the classes in the training set. Similar to the entry extraction task, the XLM-$\mathrm { R _ { B a s e } }$ outperforms other baselines. Although overall promising results are obtained, we highlight the shortcoming of the models on the categories with many tags (Subpillars 1D and Subpillars 2D), suggesting future research directions for addressing these challenges.

## 4 Conclusion

We presented HUMSET, a new dataset of annotated humanitarian data, containing 148,621 entries with a total of 62 different tags. We have shown two NLP-based tasks that can be applied to it, providing initial experiments of its applications. HUMSET is a multilingual human-annotated humanitarian text dataset not composed of social network data, providing a valuable and highly reliable resource for the development of automation tools regarding crisis response and humanitarian aid activities.

## 5 Limitations

HUMSET is composed of an aggregation of 46 different projects, each with a different contribution in terms of data quantity and topics (Table 7). This can introduce an implicit bias due to the different goals and themes of each project and on respective analysis framework understanding and interpretation by humanitarian annotators. (Röttger et al., 2021) refer to it as persistent subjectivity. This is a complex and challenging limitation and, for example, (Geva et al., 2019) show how this kind of bias can be monitored using annotator identifiers as features in NLP models training when data is produced by crowdsourcing project (Sheng and Zhang, 2019). Since HUMSET is an extension of a real-case application and not the result of crowdsourcing, a more structured analysis on these aspects is needed.

Another complexity lies in the raw data sources. Lead text is the result of a text extraction process from PDF and HTML files (c.f. Section 2.1). In both cases, converting visually-rich graphical text representation into plain text involves errors and limitations. There are several works proposing solutions for digital documents layout-aware text extraction (Ramakrishnan et al., 2012; Zhu and Cole, 2022) but they are often domain-specific, applying only to specific types of documents. (Xu et al., 2020) propose a Transformer-based multi-modal architecture for documents understanding using text, layout, and image data as features. Improvement in document processing could produce better data quality and subsequently improve performance on the entry extraction task (Section 3.1).

Finally, we should point out that HUMSET might contain societal biases and stereotypes and/or overrepresent particular demographics or entities. This case is observed and studied in several other data resources and scenarios (Bolukbasi et al., 2016; Krieg et al., 2022a; Rekabsaz et al., 2021b), which can lead to reflecting or even exaggerating societal biases in the system’s output (Melchiorre et al., 2021; Rekabsaz and Schedl, 2020), and may negatively affect users’ perception and interaction behavior (Krieg et al., 2022b). Hence, when using the dataset (particularly for real-world applications), we strongly recommend first defining and monitoring such potential biases (De-Arteaga et al., 2019; Rekabsaz et al., 2021c), and then mitigating them using the proposed methods in literature (Elazar and Goldberg, 2018; Zmigrod et al., 2019; Rekabsaz et al., 2021a; Zerveas et al., 2022; Ganhör et al., 2022).

## 6 Acknowledgement

We want to thank the humanitarian community users of the Data Entry and Exploration Platform (DEEP) for their openness and interest in sharing their data for this research and to trust that the NLP community can help them make their work better. We want to extend our gratitude, especially to the people working in the USAID’s Bureau for Humanitarian Assistance (BHA) for entrusting Data Friendly Space (DFS) with a grant to create the DEEP and to work on this paper; to the DEEP users and taggers for their work that make this set possible; to the project owners in DEEP that allowed us to use the data to create this set and their organizations; to the DEEP board members Internal Displacement Monitoring Centre (IDMC), International Federation of the Red Cross, iMMAP, Office of the High Commissioner for Human Rights, Okular Analytics, United Nations Office for the Coordination of Humanitarian Affairs (UNOCHA), United Nations High Commissioner for Refugees (UNHCR), United Nations Children’s FUND (UNICEF), United Nations Development Coordination Office (UNDCO), and the Danish Refugee Council (DRC). To the whole DFS team around the world, the ToggleCorp team, our partner institutions ISI Foundation and Johannes Kepler University Linz for their continuous support in making the usage of NLP possible in the humanitarian community.

## References

Ghadah Adel and Yuping Wang. 2020. Detecting and classifying humanitarian crisis in arabic tweets. In 2020 3rd International Conference on Artificial Intelligence and Big Data (ICAIBD), pages 269–274.  
Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli. 2021. Crisisbench: Benchmarking crisisrelated social media datasets for humanitarian information processing. In Proceedings of the International AAAI Conference on Web and Social Media, volume 15, pages 923–932.  
Alaa Alharbi and Mark Lee. 2021. Kawarith: an Arabic Twitter corpus for crisis events. In Proceedings of the Sixth Arabic Natural Language Processing Workshop, pages 42–52, Kyiv, Ukraine (Virtual). Association for Computational Linguistics.  
Steven Bird and Edward Loper. 2004. NLTK: The natural language toolkit. In Proceedings of the ACL Interactive Poster and Demonstration Sessions, pages 214–217, Barcelona, Spain. Association for Computational Linguistics.  
Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in Neural Information Processing Systems.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116.  
Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 120–128.  
Yanai Elazar and Yoav Goldberg. 2018. Adversarial removal of demographic attributes from text data. In Proceedings of the Conference on Empirical Meth ods in Natural Language Processing, pages 11–21.  
Christian Ganhör, David Penz, Navid Rekabsaz, Oleg Lesota, and Markus Schedl. 2022. Mitigating consumer biases in recommendations with adversarial training. In Proceedings of the 45th International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2022. ACM.  
Mor Geva, Yoav Goldberg, and Jonathan Berant. 2019. Are we modeling the task or the annotator? an investigation of annotator bias in natural language understanding datasets. In Proceedings of the 2019 Con ference on Empirical Methods in Natural Language  
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1161–1166, Hong Kong, China. Association for Computational Linguistics.  
Muhammad Imran, Prasenjit Mitra, and Carlos Castillo. 2016. Twitter as a lifeline: Human-annotated twitter corpora for nlp of crisis-related messages. arXiv preprint arXiv:1605.05894.  
Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2016. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759.  
Klara Krieg, Emilia Parada-Cabaleiro, Gertraud Medicus, Oleg Lesota, Markus Schedl, and Navid Rekabsaz. 2022a. Grep-biasir: A dataset for investigating gender representation-bias in information retrieval results. arXiv preprint arXiv:2201.07754.  
Klara Krieg, Emilia Parada-Cabaleiro, Markus Schedl, and Navid Rekabsaz. 2022b. Do perceived gender biases in retrieval results affect relevance judgements? In Proceedings ofthe European Conference on Information Retrieval, Workshop on Algorithmic Bias in Search and Recommendation (ECIR-BIAS 2022).  
Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. Text Summarization Branches Out.  
Yang Liu and Mirella Lapata. 2019. Text summarization with pretrained encoders. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3730–3740, Hong Kong, China. Association for Computational Linguistics.  
Alessandro B. Melchiorre, Navid Rekabsaz, Emilia Parada-Cabaleiro, Stefan Brandl, Oleg Lesota, and Markus Schedl. 2021. Investigating gender fairness of recommendation algorithms in the music domain. Information Processing Managment (IP&M), 58(5):102666.  
Subhabrata Mukherjee and Ahmed Hassan Awadallah. 2020. XtremeDistil: Multi-stage distillation for massive multilingual models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2221–2234, Online. Association for Computational Linguistics.  
Ignazio Pillai, Giorgio Fumera, and Fabio Roli. 2013. Threshold optimisation for multi-label classifiers. Pattern Recognition, 46(7):2055–2065.  
Charles C Ragin and Lisa M Amoroso. 2011. Constructing social research: The unity and diversity of method. Pine Forge Press.  
Cartic Ramakrishnan, Abhishek Patnia, Eduard Hovy, and Gully APC Burns. 2012. Layout-aware text extraction from full-text pdf of scientific articles. Source codefor biology and medicine, 7(1):1–10.  
Navid Rekabsaz, Simone Kopeinik, and Markus Schedl. 2021a. Societal biases in retrieved contents: Measurement framework and adversarial mitigation of BERT rankers. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 306– 316.  
Navid Rekabsaz, Oleg Lesota, Markus Schedl, Jon Brassey, and Carsten Eickhoff. 2021b. TripClick: The Log Files of a Large Health Web Search Engine. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Infor mation Retrieval, pages 2507–2513. Association for Computing Machinery, New York, NY, USA.  
Navid Rekabsaz and Markus Schedl. 2020. Do neural ranking models intensify gender bias? In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2065–2068.  
Navid Rekabsaz, Robert West, James Henderson, and Allan Hanbury. 2021c. Measuring societal biases in text corpora via first-order co-occurrence. Proceedings of the International AAAI Conference on Web and Social Media (ICWSM).  
Paul Röttger, Bertie Vidgen, Dirk Hovy, and Janet B. Pierrehumbert. 2021. Two contrasting data annotation paradigms for subjective NLP tasks. CoRR, abs/2112.07475.  
Victor S Sheng and Jing Zhang. 2019. Machine learning with crowdsourcing: A brief summary of the past research and future directions. In Proceedings ofthe AAAI conference on artificial intelligence, volume 33, pages 9837–9843.  
Piotr Szymanski and Tomasz Kajdanowicz. 2017. A scikit-based python environment for performing multi-label classification. CoRR. abs/1702.01460.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.  
Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. 2020. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. arXiv preprint arXiv:2012.14740.  
Jenny Paola Yela-Bello, Ewan Oglethorpe, and Navid Rekabsaz. 2021. MultiHumES: Multilingual humanitarian dataset for extractive summarization. In Proceedings ofthe 16th Conference ofthe European Chapter of the Association for Computational Linguistics: Main Volume, pages 1713–1717, Online. Association for Computational Linguistics.  
George Zerveas, Navid Rekabsaz, Daniel Cohen, and Carsten Eickhoff. 2022. Mitigating bias in search results through set-based document reranking and neutrality regularization. In Proceedings of the 45th International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2022. ACM.  
Dongsong Zhang, Lina Zhou, and Jay F Nunamaker Jr. 2002. A knowledge management framework for the support of decision making in humanitarian assistance/disaster relief. Knowledge and Information Systems, 4(3):370–385.  
Qingyu Zhou, Nan Yang, Furu Wei, Shaohan Huang, Ming Zhou, and Tiejun Zhao. 2018. Neural document summarization by jointly learning to score and select sentences. In Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 654– 663, Melbourne, Australia. Association for Computational Linguistics.  
Miao Zhu and Jacqueline M Cole. 2022. Pdfdataextractor: A tool for reading scientific text and interpreting metadata from the typeset literature in the portable document format. Journal ofChemical Information and Modeling, 62(7):1633–1643.  
Ran Zmigrod, Sabrina J. Mielke, Hanna Wallach, and Ryan Cotterell. 2019. Counterfactual data augmentation for mitigating gender stereotypes in languages with rich morphology. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1651–1661, Florence, Italy. Association for Computational Linguistics.

## A Additional Statistics

<table><tr><td>Website</td><td>Number</td><td>Prop. (%)</td></tr><tr><td>reliefweb.int</td><td>5,669</td><td>33.6</td></tr><tr><td>dhakatribune.com</td><td>834</td><td>4.9</td></tr><tr><td>redhum.org</td><td>635</td><td>3.8</td></tr><tr><td>humanitarianresponse.info</td><td>612</td><td>3.6</td></tr><tr><td>unb.com.bd</td><td>380</td><td>2.3</td></tr><tr><td>Sum</td><td>8,130</td><td>48.2</td></tr></table>

Table 4: The most frequently sourced websites by number and proportion of documents.

<table><tr><td>Year</td><td>Number</td><td>Proportion (\%)</td></tr><tr><td>Before 2018</td><td>68</td><td>0.4</td></tr><tr><td>2018</td><td>834</td><td>4.9</td></tr><tr><td>2019</td><td>2,639</td><td>15.7</td></tr><tr><td>2020</td><td>6,087</td><td>36.1</td></tr><tr><td>2021</td><td>7,229</td><td>42.9</td></tr><tr><td>Sum</td><td>16,857</td><td>100.0</td></tr></table>

Table 5: Publishing year of documents.

<table><tr><td>Region</td><td>Countries</td></tr><tr><td>Africa</td><td>Burkina Faso, Cameroon, Chad, Libya, Niger, Nigeria, DRC, Somalia, South Sudan, Sudan</td></tr><tr><td>Asia &amp; Middle East</td><td>Afghanistan, Bangladesh, Lebanon, Syria, Yemen</td></tr><tr><td>Central/South America</td><td>Argentina, Aruba, Bolivia, Chile, Colombia, Costa Rica, Curacao, Dominican Republic, Ecuador, El Salvador, Guatemala, Guyana, Honduras, Mexico, Panama, Paraguay, Peru, The Bahamas, Trinidad and Tobago, Uruguay, Venezuela</td></tr></table>

![](images/a5d93ac9d0023fd7079a6fa7553ef6b3b8d556762b327eb588ff0eba246a3f88.jpg)

<details>
<summary>bar</summary>

| Category | Occurrence Probability (%) |
| --- | --- |
| 1 tag | ~22 |
| 2 | ~17.5 |
| 3 | ~17.5 |
| 4 | ~16.5 |
| 5 | ~13.2 |
| 6 | ~10.8 |
| 7 | ~10.5 |
| 8 | ~10.5 |
| 9 | ~10.2 |
| 10 | ~9.8 |
| 11 | ~8.5 |
| 12 | ~8.2 |
| 13 | ~7.2 |
| 14 | ~6.5 |
| 15 | ~6.5 |
| 16 | ~6.2 |
| 17 | ~5.8 |
| 18 | ~5.2 |
| 19 | ~4.2 |
| 20 | ~3.8 |
| 21 | ~3.5 |
| 22 | ~3.5 |
| 23 | ~3.5 |
| 24 | ~3.5 |
| 25 | ~2.2 |
| 26 | ~2.2 |
| 27 | ~2.2 |
| 28 | ~2.2 |
| 29 | ~2.2 |
| 30 | ~2.0 |
| 31 | ~1.8 |
| 32 | ~1.5 |
| 33 | ~1.5 |
| 34 | ~1.5 |
| 35 | ~1.5 |
| 36 | ~1.5 |
| 37 | ~1.5 |
| 38 | ~1.5 |
| 39 | ~1.5 |
| 40 | ~1.5 |
| 41 | ~1.5 |
| 42 | ~1.5 |
| 43 | ~1.5 |
| 44 | ~1.5 |
| 45 | ~1.5 |
| 46 | ~1.5 |
| 47 | ~1.5 |
| 48 | ~1.5 |
| 49 | ~1.5 |
| 50 | ~1.5 |
| 51 | ~1.5 |
| 52 | ~1.5 |
| 53 | ~1.5 |
| 54 | ~1.5 |
| 55 | ~1.5 |
| 56 | ~1.5 |
| 57 | ~1.5 |
| 58 | ~1.5 |
| 59 | ~1.5 |
| 60 | ~1.5 |
| 61 | ~1.5 |
| 62 | ~1.5 |
| 63 | ~1.5 |
| 64 | ~1.5 |
| 65 | ~1.5 |
| 66 | ~1.5 |
| 67 | ~1.5 |
| 68 | ~1.5 |
| 69 | ~1.5 |
| 70 | ~1.5 |
| 71 | ~1.5 |
| 72 | ~1.5 |
| 73 | ~1.5 |
| 74 | ~1.5 |
| 75 | ~1.5 |
| 76 | ~1.5 |
| 77 | ~1.5 |
| 78 | ~1.5 |
| 79 | ~1.5 |
| 80 | ~1.5 |
| 81 | ~1.5 |
| 82 | ~1.5 |
| 83 | ~1.5 |
| 84 | ~1.5 |
| 85 | ~1.5 |
| 86 | ~1.5 |
| 87 | ~1.5 |
| 88 | ~1.5 |
| 89 | ~1.5 |
| 90 | ~1.5 |
| 91 | ~1.5 |
| 92 | ~1.5 |
| 93 | ~1.5 |
| 94 | ~1.5 |
| 95 | ~1.5 |
| 96 | ~1.5 |
| 97 | ~1.5 |
| 98 | ~1.5 |
| 99 | ~1.5 |
| 100 | ~1.5 |
</details>

Figure 2: Proportion of tags in dataset. This figure shows the unbalanced nature of the dataset. Each bar represents a different tag. The y-axis shows the proportion of entries that contain each tag. The horizontal line added is the 2% occurrence line. It is used to visualize the relatively high number of tags with occurrence inferior to 2%.  
Table 6: Countries of projects per region.

<table><tr><td>Project</td><td># Leads</td><td># Entries</td></tr><tr><td>2020 DFS Libya</td><td>354</td><td>1,581</td></tr><tr><td>2020 DFS Nigeria</td><td>496</td><td>2,238</td></tr><tr><td>COVID-19 Americas Region Multi-Sectorial Assessment</td><td>76</td><td>607</td></tr><tr><td>Central America - Dengue Outbreak 2019</td><td>38</td><td>218</td></tr><tr><td>Central America: Hurricanes Eta and Iota</td><td>48</td><td>363</td></tr><tr><td>GIMAC Afghanistan</td><td>248</td><td>7,617</td></tr><tr><td>GIMAC Cameroon</td><td>134</td><td>5,253</td></tr><tr><td>GIMAC Chad</td><td>269</td><td>4,945</td></tr><tr><td>GIMAC Niger</td><td>125</td><td>3,180</td></tr><tr><td>GIMAC Somalia</td><td>157</td><td>5,720</td></tr><tr><td>GIMAC South Sudan</td><td>208</td><td>7,212</td></tr><tr><td>GIMAC Sudan</td><td>99</td><td>3,169</td></tr><tr><td>IMMAP/DFS Bangladesh</td><td>2,266</td><td>14,342</td></tr><tr><td>IMMAP/DFS Burkina Faso</td><td>1,279</td><td>13,443</td></tr><tr><td>IMMAP/DFS Colombia</td><td>1,411</td><td>10,760</td></tr><tr><td>IMMAP/DFS Nigeria</td><td>1,443</td><td>9,620</td></tr><tr><td>IMMAP/DFS RDC</td><td>1,586</td><td>13,065</td></tr><tr><td>IMMAP/DFS Syria</td><td>1,736</td><td>12,043</td></tr><tr><td>Lebanon Situation Analysis</td><td>22</td><td>268</td></tr><tr><td>Libya Situation Analysis (OA)</td><td>681</td><td>2,868</td></tr><tr><td>Nigeria Situation Analysis (OA)</td><td>651</td><td>3,496</td></tr><tr><td>Situation Analysis Generic Libya</td><td>437</td><td>2,355</td></tr><tr><td>Situation Analysis Generic Yemen</td><td>371</td><td>2,256</td></tr><tr><td>The Bahamas - Hurricane Dorian - Early Recovery Assessment</td><td>17</td><td>191</td></tr><tr><td>UNHCR Argentina</td><td>160</td><td>1430</td></tr><tr><td>UNHCR Aruba</td><td>23</td><td>106</td></tr><tr><td>UNHCR Bolivia</td><td>9</td><td>101</td></tr><tr><td>UNHCR Chile</td><td>341</td><td>2,415</td></tr><tr><td>UNHCR Colombia</td><td>649</td><td>6,349</td></tr><tr><td>UNHCR Costa Rica</td><td>85</td><td>603</td></tr><tr><td>UNHCR Curacao</td><td>20</td><td>111</td></tr><tr><td>UNHCR Dominican Republic</td><td>66</td><td>411</td></tr><tr><td>UNHCR Ecuador</td><td>190</td><td>1,648</td></tr><tr><td>UNHCR El Salvador</td><td>64</td><td>349</td></tr><tr><td>UNHCR Guatemala</td><td>65</td><td>348</td></tr><tr><td>UNHCR Guyana</td><td>28</td><td>352</td></tr><tr><td>UNHCR Honduras</td><td>57</td><td>469</td></tr><tr><td>UNHCR Mexico</td><td>16</td><td>96</td></tr><tr><td>UNHCR Panama</td><td>35</td><td>295</td></tr><tr><td>UNHCR Paraguay</td><td>2</td><td>27</td></tr><tr><td>UNHCR Peru</td><td>247</td><td>2,936</td></tr><tr><td>UNHCR Trinidad and Tobago</td><td>67</td><td>574</td></tr><tr><td>UNHCR Uruguay</td><td>29</td><td>272</td></tr><tr><td>UNHCR Venezuela</td><td>293</td><td>1,805</td></tr><tr><td>Venezuela crisis 2019</td><td>176</td><td>714</td></tr><tr><td>Yemen Situation Analysis (OA)</td><td>83</td><td>381</td></tr><tr><td>Total</td><td>16,857</td><td>148,602</td></tr></table>

<table><tr><td>Pillar 1D</td><td>#</td><td>Subpillars 1DLabel</td></tr><tr><td>Context</td><td>7</td><td>Economy, Environment, Demography, Legal &amp; Policy, Politics, Security &amp; Stability, Sociocultural</td></tr><tr><td>COVID-19</td><td>7</td><td>Cases, Contact Tracing, Deaths, Hospitalization &amp; Care, Restriction Measures, Testing, Vaccination</td></tr><tr><td>Displacement</td><td>5</td><td>Intentions, Local Integration, Pull Factors, Push Factors, Type / Numbers / Movements</td></tr><tr><td>Humanitarian Access</td><td>4</td><td>Physical Constraints, Population to Relief, Relief to Population, Number of People Facing Humanitarian Access Constraints / Humanitarian Access Gaps</td></tr><tr><td>Information and Communication</td><td>4</td><td>Communication Means and Preferences, Information Challenges and Barriers, Knowledge and Information Gaps (Hum), Knowledge And Info Gaps (Pop)</td></tr><tr><td>Casualties</td><td>3</td><td>Dead, Injured, Missing</td></tr><tr><td>Shock / Event</td><td>3</td><td>Hazard &amp; Threats, Type and Characteristics Underlying/Aggravating Factors</td></tr><tr><td>Pillar 2D</td><td>#</td><td>Subpillars 2DLabel</td></tr><tr><td>Capacities &amp; Response</td><td>4</td><td>International Response, Local Response, National Response, Number of People Reached / Response Gaps</td></tr><tr><td>Humanitarian Conditions</td><td>4</td><td>Coping Mechanisms, Living Standards, Physical And Mental Well Being, Number of People In Need</td></tr><tr><td>Impact</td><td>4</td><td>Driver/Aggravating Factors, Impact on People, Impact on Systems, Services and Networks, Number of People Affected</td></tr><tr><td>At Risk</td><td>2</td><td>Risk And Vulnerabilities, Number of People at Risk</td></tr><tr><td>Priority Needs</td><td>2</td><td>Expressed by Humanitarian Staff, Expressed by Population</td></tr><tr><td>Priority Interventions</td><td>2</td><td>Expressed by Humanitarian Staff, Expressed by Population</td></tr></table>

Table 8: Pillars and subpillars 1D in humanitarian framework.  
Table 7: Key statistics per project.

Table 9: Pillars and subpillars 2D in humanitarian framework.

<table><tr><td rowspan="2">Sectors</td><td colspan="2">Data Points</td></tr><tr><td>Number</td><td>Proportion (%)</td></tr><tr><td>Agriculture</td><td>2,816</td><td>1.9</td></tr><tr><td>Cross</td><td>24,447</td><td>16.4</td></tr><tr><td>Education</td><td>9,630</td><td>6.5</td></tr><tr><td>Food Security</td><td>14,898</td><td>10.0</td></tr><tr><td>Health</td><td>32,284</td><td>21.7</td></tr><tr><td>Livelihoods</td><td>15,494</td><td>10.4</td></tr><tr><td>Logistics</td><td>2,422</td><td>1.6</td></tr><tr><td>Nutrition</td><td>5,011</td><td>3.4</td></tr><tr><td>Protection</td><td>25,986</td><td>17.5</td></tr><tr><td>Shelter</td><td>8,975</td><td>6.0</td></tr><tr><td>WASH</td><td>10,588</td><td>7.1</td></tr><tr><td>Count:</td><td>115,176</td><td>77.5</td></tr></table>

Table 10: Proportion of sectors in the dataset.

<table><tr><td rowspan="2">Pillar 1D</td><td rowspan="2">Subpillar 1D</td><td colspan="2">Data Points</td></tr><tr><td>Number</td><td>Prop. (%)</td></tr><tr><td rowspan="8">Context</td><td>Demography</td><td>3,041</td><td>2.0</td></tr><tr><td>Economy</td><td>4,969</td><td>3.3</td></tr><tr><td>Environment</td><td>1,124</td><td>0.8</td></tr><tr><td>Legal &amp; Policy</td><td>2,637</td><td>1.8</td></tr><tr><td>Politics</td><td>1,871</td><td>1.3</td></tr><tr><td>Security &amp; Stability</td><td>7,615</td><td>5.1</td></tr><tr><td>Sociocultural</td><td>1,610</td><td>1.1</td></tr><tr><td>Pillar 1D Count:</td><td>21,070</td><td>14.2</td></tr><tr><td rowspan="8">COVID-19</td><td>Cases</td><td>5,501</td><td>3.7</td></tr><tr><td>Contact Tracing</td><td>627</td><td>0.4</td></tr><tr><td>Deaths</td><td>3,095</td><td>2.1</td></tr><tr><td>Hospitalization &amp; Care</td><td>510</td><td>0.3</td></tr><tr><td>Restriction Measures</td><td>5,006</td><td>3.4</td></tr><tr><td>Testing</td><td>1,621</td><td>1.1</td></tr><tr><td>Vaccination</td><td>2,764</td><td>1.9</td></tr><tr><td>Pillar 1D Count:</td><td>14,964</td><td>10.1</td></tr><tr><td rowspan="6">Displacement</td><td>Intentions</td><td>454</td><td>0.3</td></tr><tr><td>Local Integration</td><td>1,657</td><td>1.1</td></tr><tr><td>Pull Factors</td><td>342</td><td>0.2</td></tr><tr><td>Push Factors</td><td>2,047</td><td>1.4</td></tr><tr><td>Type / Numbers / Movements</td><td>8,280</td><td>5.6</td></tr><tr><td>Pillar 1D Count:</td><td>10,678</td><td>7.2</td></tr><tr><td rowspan="4">Humanitarian Access</td><td>Number Of People Facing Humanitarian Access Constraints/Humanitarian Access Gaps</td><td>658</td><td>0.4</td></tr><tr><td>Physical Constraints Population to Relief</td><td>1,483</td><td>1.0</td></tr><tr><td>Relief to Population</td><td>204</td><td>0.1</td></tr><tr><td>Pillar 1D Count:</td><td>2,922</td><td>2.0</td></tr><tr><td rowspan="5">Information and Communication</td><td rowspan="2">Communication Means and Preferences Information Challenges and Barriers</td><td>150</td><td>0.1</td></tr><tr><td>100</td><td>0.1</td></tr><tr><td>Knowledge and Info Gaps (Hum)</td><td>573</td><td>0.4</td></tr><tr><td>Knowledge and Info Gaps (Pop)</td><td>218</td><td>0.1</td></tr><tr><td>Pillar 1D Count:</td><td>993</td><td>0.7</td></tr><tr><td rowspan="4">Casualties</td><td>Dead</td><td>3,147</td><td>2.1</td></tr><tr><td>Injured</td><td>649</td><td>0.4</td></tr><tr><td>Missing</td><td>269</td><td>0.4</td></tr><tr><td>Pillar 1D Count:</td><td>3,497</td><td>2.4</td></tr><tr><td rowspan="5">Shock/Event</td><td rowspan="3">Hazard &amp; Threats Type and Characteristics Underlying/Aggravating Factors</td><td>3,616</td><td>2.4</td></tr><tr><td>1,241</td><td>0.8</td></tr><tr><td>1,589</td><td>1.1</td></tr><tr><td>Pillar 1D Count:</td><td>6,072</td><td>4.1</td></tr><tr><td>Overall Count:</td><td>53,575</td><td>36.0</td></tr><tr><td rowspan="2">Pillar 2D</td><td rowspan="2">Subpillars 2D</td><td colspan="2">Data Points</td></tr><tr><td>Number</td><td>Prop. (%)</td></tr><tr><td rowspan="5">Capacities &amp; Response</td><td>International Response</td><td>15,422</td><td>10.4</td></tr><tr><td>Local Response</td><td>211</td><td>0.1</td></tr><tr><td>National Response Number Of People</td><td>6,596</td><td>4.4</td></tr><tr><td>Reached / Response Gaps</td><td>3,316</td><td>2.2</td></tr><tr><td>Pillar 2D Count:</td><td>20,272</td><td>13.6</td></tr><tr><td rowspan="5">Humanitarian Conditions</td><td>Coping Mechanisms</td><td>4,994</td><td>3.4</td></tr><tr><td>Living Standards</td><td>25,922</td><td>17.4</td></tr><tr><td>Number of People In Need</td><td>1,315</td><td>0.9</td></tr><tr><td>Physical and Mental Well Being</td><td>15,845</td><td>10.7</td></tr><tr><td>Pillar 2D Count:</td><td>43,301</td><td>29.1</td></tr><tr><td rowspan="5">Impact</td><td>Driver/Aggravating Factors</td><td>12,029</td><td>8.1</td></tr><tr><td>Impact on People Impact on Systems,</td><td>12,319</td><td>8.3</td></tr><tr><td>Services And Networks</td><td>14,335</td><td>9.6</td></tr><tr><td>Number of People Affected</td><td>2,293</td><td>1.5</td></tr><tr><td>Pillar 2D Count:</td><td>33,699</td><td>22.7</td></tr><tr><td rowspan="3">At Risk</td><td>Risk and Vulnerabilities</td><td>9,625</td><td>6.5</td></tr><tr><td>Number of People Aa Risk</td><td>261</td><td>0.2</td></tr><tr><td>Pillar 2D Count:</td><td>9,625</td><td>6.5</td></tr><tr><td rowspan="3">Priority Needs</td><td>Expressed by Humanitarian Staff</td><td>1,664</td><td>1.1</td></tr><tr><td>Expressed by Population</td><td>1,669</td><td>1.1</td></tr><tr><td>Pillar 2D Count:</td><td>3,272</td><td>2.2</td></tr><tr><td rowspan="4">Priority Interventions</td><td>Expressed by Humanitarian Staff</td><td>5,575</td><td>3.8</td></tr><tr><td>Expressed by Population</td><td>287</td><td>0.2</td></tr><tr><td>Pillar 2D Count:</td><td>5,844</td><td>3.9</td></tr><tr><td>Overall Count:</td><td>94,429</td><td>63.5</td></tr></table>

Table 11: Proportion of each pillar and subpillar 1D in the dataset.

Table 12: Proportion of each pillar and subpillar 2D in the dataset.