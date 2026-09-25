# calamanCy: A Tagalog Natural Language Processing Toolkit

Lester James V. Miranda

ExplosionAI GmbH

lj@explosion.ai

## Abstract

We introduce calamanCy, an open-source toolkit for constructing natural language processing (NLP) pipelines for Tagalog. It is built on top of spaCy, enabling easy experimentation and integration with other frameworks. calamanCy addresses the development gap by providing a consistent API for building NLP applications and offering general-purpose multitask models with out-of-the-box support for dependency parsing, parts-of-speech (POS) tagging, and named entity recognition (NER). calamanCy aims to accelerate the progress of Tagalog NLP by consolidating disjointed resources in a unified framework. The calamanCy toolkit is available on GitHub: https: //github.com/ljvmiranda921/calamanCy.

## 1 Introduction

Tagalog is a low-resource language from the Austronesian family, with over 28 million speakers in the Philippines (Lewis, 2009). Despite its speaker population, few resources exist for the language (Cruz and Cheng, 2022). For example, Universal Dependencies (UD) treebanks for Tagalog are tiny (≪ 20k words) (Samson, 2018; Aquino and de Leon, 2020), while domain-specific corpora are sparse (Cabasag et al., 2019; Livelo and Cheng, 2018). In addition, Tagalog language models (LMs) (Cruz and Cheng, 2022; Jiang et al., 2021) are few, while most multilingual LMs (Conneau et al., 2020; Devlin et al., 2019) underrepresent the language (Lauscher et al., 2020). Thus, consolidating these disjointed resources in a coherent framework is still an open problem. The lack of such framework hampers model development, experimental workflows, and the overall advancement of Tagalog NLP.

To address this problem, we introduce calamanCy,<sup>1</sup> an open-source toolkit for Tagalog NLP. It is built on top of spaCy (Honnibal et al., 2020)

and offers end-to-end pipelines for NLP tasks such as dependency parsing, parts-of-speech (POS) tagging, and named entity recognition (NER). calamanCy also provides general-purpose pipelines in three different sizes to fit any performance or accuracy requirements. This work has two main contributions: (1) an open-source toolkit with outof-the box support for common NLP tasks, and (2) comprehensive evaluations on several Tagalog benchmarks.

## 2 Related Work

Open-source toolkits for NLP There has been a growing body of work in the development of NLP toolkits in recent years. For example, DaCy (Enevoldsen et al., 2021) and HuSpaCy (Orosz et al., 2022) serve the language-specific needs of Danish and Hungarian respectively. In addition, scispaCy (Neumann et al., 2019) and medspaCy (Eyre et al., 2021) were built to focus on scientific text. These tools employ spaCy (Honnibal et al., 2020), an industrial-strength open-source software for natural language processing. Using spaCy as a foundation is optimal, given its popularity and integration with other frameworks such as Hugging-Face transformers (Wolf et al., 2020). However, no tool has existed for Tagalog until now. We aim to fill this development gap and serve the needs of the Tagalog language community through calamanCy.

Evaluations on Tagalog NLP Tasks Structured evaluations for core NLP tasks, such as dependency parsing, POS tagging, and NER, are meager. However, we have access to a reasonable amount of data to conduct comprehensive benchmarks. For example, TLUnified (Cruz and Cheng, 2022) is a pretraining corpus that combines news reports (Cruz et al., 2020), a preprocessed version of CommonCrawl (Suarez et al., 2019), and several other datasets. However, it was evaluated on domainspecific corpora that may not easily transfer to more general tasks. In addition, Tagalog has two Universal Dependencies (UD) treebanks, Tagalog Reference Grammar (TRG) (Samson, 2018) and Ug nayan (Aquino and de Leon, 2020), both with POS tags and relational structures for parsing grammar. This paper will fill the evaluation gap by providing structured benchmarks on these core tasks.

<table><tr><td>Entity</td><td>Description</td><td>Examples</td></tr><tr><td>Person (PER)</td><td>Person entities limited to humans. It may be a single individual or group.</td><td>Juan de la Cruz, Jose Rizal, Quijano de Manila</td></tr><tr><td>Organization (ORG)</td><td>Organization entities limited to corporations, agencies, and other groups of people defined by an organizational structure.</td><td>Meralco, DPWH, United Nations</td></tr><tr><td>Location (LOC)</td><td>Location entities are geographical regions, areas, and landmasses. Geo-political entities are also included within this group.</td><td>Pilipinas, Manila, CAL-ABARZON, Ilog Pasig</td></tr></table>

Table 1: Entity types used for annotating TLUnified-NER (derived from the TLUnified pretraining corpus of Cruz and Cheng, 2022).

## 3 Implementation

The best way to use calamanCy is through its trained pipelines. After installing the library, users can access the models in a few lines of code:

import calamancy as cl

nlp = cl.load("tl\_calamancy\_md-0.1.0")

doc = nlp("Ako si Juan de la Cruz.")

Here, the variable nlp is a spaCy processing pipeline<sup>2</sup> that contains trained components for POS tagging, dependency parsing, and NER. Applying this pipeline to a text will produce a Doc object with various linguistic features. calamanCy offers three pipelines of varying capacity: two static word vector-based models (md, lg), and one transformerbased model (trf). We will discuss how we developed these pipelines in the following section.

## 3.1 Pipeline development

Data annotation for NER There is no goldstandard corpus for NER, so we built one. To construct the NER corpus, we curated a portion of TLUnified (Cruz and Cheng, 2022) to contain Tagalog news articles. Including the author, we recruited two more annotators with at least a bachelor’s degree and whose native language is Tagalog. The three annotators labeled for four months, given three entity types as seen in Table 1. We chose the entity types to resemble ConLL (Tjong Kim Sang, 2002; Tjong Kim Sang and De Meulder, 2003), a standard NER benchmark. We excluded the MISC label to reduce uncertainty and confusion when labeling. Then, we measured inter-annotator agreement (IAA) by taking the pairwise Cohen’s κ on all tokens and then averaged them for all three pairs. This process resulted in a Cohen’s κ score of 0.81. To avoid confusion with the original TLUnified pretraining corpora, we will refer to this annotated NER dataset as TLUnified-NER. The final dataset statistics can be found in Table 2. For the dependency parser and POS tagger, we merged the TRG (Samson, 2018) and Ugnayan (Aquino and de Leon, 2020) treebanks to leverage their small yet relevant examples.

<table><tr><td>Dataset</td><td>Examples</td><td>PER</td><td>ORG</td><td>LOC</td></tr><tr><td>Training</td><td>6252</td><td>6418</td><td>3121</td><td>3296</td></tr><tr><td>Development</td><td>782</td><td>793</td><td>392</td><td>409</td></tr><tr><td>Test</td><td>782</td><td>818</td><td>423</td><td>438</td></tr></table>

Table 2: Dataset statistics for TLUnified-NER.

Model training We considered three design dimensions when training the calamanCy pipelines: (1) the presence of pretraining, (2) the word representation, and its (3) size or dimension. Model pretraining involves learning vectors from raw text to inform model initialization. Here, the pretraining objective asks the model to predict some number of leading and trailing UTF-8 bytes for the words—a variant of the cloze task (Devlin et al., 2019). A model’s word representation may involve training static word embeddings using floret,<sup>3</sup> an efficient version of fastText (Bojanowski et al., 2017), or using context-sensitive vectors from a transformer (Vaswani et al., 2017). Finally, a model’s dimension is our way to tune the tradeoff between performance and accuracy.

<table><tr><td>Pipeline</td><td>Pretraining objective</td><td>Word embeddings</td><td>Dimensions</td></tr><tr><td>Medium-sized pipeline (tl_calamancy_md)</td><td>Predict some number of leading and trailing UTF-8 bytes for the words.</td><td>Uses floret vectors trained on the TLUnified corpora.</td><td>50k unique vectors (200 dimensions), Size: 77 MB</td></tr><tr><td>Large-sized pipeline (tl_calamancy_lg)</td><td>Same pretraining objective as the medium-sized pipeline.</td><td>Uses fastText vectors trained on Common-Crawl corpora.</td><td>714k unique vectors (300 dimensions), Size: 455 MB</td></tr><tr><td>Transformer-based pipeline (tl_calamancy_trf)</td><td>No separate pretraining because there&#x27;s no token-to-vector component.</td><td>Context-sensitive vectors from a transformer network.</td><td>Uses roberta-tagalog-base. Size: 813 MB</td></tr></table>

Table 3: Language pipelines available in calamanCy (v0.1.0). The pretraining method for the word-vector models is a variant of the cloze task. All pipelines have a tagger, parser, morphologizer, and ner spaCy component.

<table><tr><td>Dataset</td><td>Task / Labels</td><td>Description</td></tr><tr><td>Hatespeech (Cabasag et al., 2019)</td><td>Binary text classification (hate speech, not hate speech)</td><td>Contains 10k tweets collected during the 2016 Philippine Presidential Elections labeled as hate speech or non-hate speech.</td></tr><tr><td>Dengue (Livelo and Cheng, 2018)</td><td>Multilabel text classification (absent, dengue, health, sick, mosquito)</td><td>Contains 4k dengue-related tweets collected for a health infoveillance application that classifies text into dengue subtopics.</td></tr><tr><td>TLUnified-NER (Cruz and Cheng, 2022)</td><td>Named entity recognition (Person, Organization, Location)</td><td>A held-out test split from the annotated TLUnified corpora containing news reports and other articles. See Table 2.</td></tr><tr><td>Merged UD (Samson, 2018; Aquino and de Leon, 2020)</td><td>Dependency parsing and POS tagging</td><td>Merged version of the Ugnayan and TRG treebanks from the Universal Dependencies framework.</td></tr></table>

Table 4: Datasets for benchmarking calamanCy.

The general process involves pretraining a filtered version of TLUnified, constructing static word embeddings if necessary, and training the downstream components. We used TLUnified-NER to train the NER component, and then trained the dependency parser and POS tagger using the combined treebanks. Ultimately, we devised three language pipelines as seen in Table 3.

## 4 Evaluation

Architectures We used spaCy’s built-in architectures for each component in the calamanCy pipeline. The token-to-vector layer uses the multihash embedding trick (Miranda et al., 2022) to reduce the representation size. For the parser and named entity recognizer, we used a transition-based parser that maps text representations into a series of state transitions. As for the text categorizer, we utilized an ensemble of a bag-of-words model and a feed-forward network.

Experimental set-up We assessed the calamanCy pipelines on various Tagalog benchmarks as detailed in Table 4. We also tested on text categorization, an unseen task, for robustness. For NER evaluation, we used a held-out test split from TLUnified-NER. We measured their performance across five trials and then reported the average and standard deviation. For treebank-related benchmarks (POS tagging and dependency parsing), we followed UD’s data split guidelines (Nivre et al., 2022) and performed 10-fold cross-validation to compensate for the size of the corpora (≪ 20k tokens).

We also tested a cross-lingual transfer learning approach, i.e., finetuning a model from a source language closely related to Tagalog. According to

<table><tr><td rowspan="2">Model</td><td colspan="2">Text categorization</td><td>NER</td><td colspan="2">Dep. pars. &amp; POS tag.</td></tr><tr><td>Hatespeech (binary)</td><td>Dengue (multilabel)</td><td>TLUnified-NER</td><td>Merged UD, UAS / LAS</td><td>Merged UD, POS Acc.</td></tr><tr><td>Monolingual (Ours)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>tl_calamancy_md</td><td>74.40±0.05</td><td>65.32±0.04</td><td>87.67±0.03</td><td>76.47 / 54.40</td><td>96.70</td></tr><tr><td>tl_calamancy_lg</td><td>75.62±0.02</td><td>68.42±0.01</td><td>88.90±0.01</td><td>82.13 / 70.32</td><td>97.20</td></tr><tr><td>tl_calamancy_trf</td><td>78.25±0.06</td><td>72.45±0.02</td><td>90.34±0.02</td><td>92.48 / 80.90</td><td>97.80</td></tr><tr><td>Cross-lingual transfer</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>uk_core_news_trf</td><td>75.24±0.03</td><td>65.57±0.01</td><td>51.11±0.02</td><td>54.77 / 37.68</td><td>82.86</td></tr><tr><td>ro_core_news_lg</td><td>69.01±0.01</td><td>59.10±0.01</td><td>02.01±0.00</td><td>84.65 / 65.30</td><td>82.80</td></tr><tr><td>ca_core_news_trf</td><td>70.01±0.02</td><td>59.42±0.03</td><td>14.58±0.02</td><td>91.17 / 79.30</td><td>83.09</td></tr><tr><td>Multilingual finetuning</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>xlm-roberta-base</td><td>77.57±0.01</td><td>67.20±0.01</td><td>88.03±0.03</td><td>88.34 / 76.07</td><td>94.29</td></tr><tr><td>bert-base-multilingual</td><td>76.40±0.02</td><td>71.07±0.04</td><td>87.40±0.02</td><td>90.79 / 78.52</td><td>95.30</td></tr></table>

Table 5: Benchmark evaluation scores for monolingual, cross-lingual, and multilingual pipelines across a variety of tasks and datasets. We evaluated the text categorization and NER tasks across five trials, and then conducted 10-fold cross-validation for dependency parsing. F1-scores are reported on the text categorization and NER tasks.

Aquino and de Leon (2020), the closest languages to Tagalog are Indonesian (id), Ukrainian (uk), Vietnamese (vi), Romanian (ro), and Catalan (ca). They obtained these results via a distance metric (Agic´, 2017) based on the World Atlas for Language Structures (Haspelmath et al., 2005). However, only uk, ro, and ca have equivalent spaCy pipelines, so we only compared against those three. Finally, we also compared against multilingual language models by finetuning on XLM RoBERTa (Conneau et al., 2020) and an uncased version of multilingual BERT (Devlin et al., 2019). These LMs contain Tagalog in their training pool and are common alternatives for building Tagalog NLP applications.

## 5 Discussion

Table 5 shows the F1-scores for the text categorization and NER tasks, the unlabeled (UAS) and labeled attachment scores (LAS) for the dependency parsing task, and the tag accuracy for POS tagging.

The calamanCy pipelines are competitive across all core NLP tasks while maintaining a smaller compute footprint. As shown in the text catego rization and NER results, users with low compute budgets can attain similar performance to multilingual LMs by using medium- or large-sized calamanCy models. The transformer-based calamanCy pipeline is the best option for users who prioritize accuracy. However, we were surprised that most alternative approaches perform better in dependency parsing. We attribute this performance to the added strength of multilingual and crosslingual information, which we don’t have when training solely on a smaller treebank. We plan to improve dependency parsing performance by building a larger treebank within the Universal Dependencies framework. For practical applications, we recommend users to start with a medium- or largesized calamanCy model before trying out GPUintensive pipelines. Only then can they switch to a transformer-based pipeline to get accuracy gains.

## 6 Conclusion

In this paper, we introduced calamanCy, a natural language processing toolkit for Tagalog. Our work has two main contributions: (1) an opensource toolkit containing general-purpose multitask pipelines with out-of-the-box support for common NLP tasks, and (2) comprehensive benchmarks that compare against alternative approaches, such as cross-lingual or multilingual finetuning. We hope that calamanCy is a step forward to improving the state of Tagalog NLP. As a low-resource language, consolidating resources into a unified framework is crucial to advance research and improve collaboration. In the future, we plan to create a more fine-grained NER benchmark corpus and extend calamanCy to natural language understanding (NLU) tasks. Finally, the project is hosted on GitHub (https://github.com/ljvmiranda921/ calamanCy) and we are happy to receive community feedback and contributions.

## Limitations

The TLUnified-NER corpus utilized for training the NER component of calamanCy comprises of new articles from early 2000s to the present. In addition the Universal Dependencies (UD) corpora for the POS tagger and dependency parser components are relatively modest in size, containing fewer than 10k tokens. Hence, the performance for these tasks during test-time could potentially be constrained by these factors.

Finally, reproducing the transformer pipelines may require a T4 or V100 GPU. The biggest bottleneck for reproduction is pretraining on the whole TLUnified corpus. In a 64vCPU machine with 256GB of RAM, the pretraining process can take three full days for 20 epochs.

## References

Željko Agic. 2017.´ Cross-lingual parser selection for low-resource languages. In Proceedings of the NoDaLiDa 2017 Workshop on Universal Dependencies (UDW 2017), pages 1–10, Gothenburg, Sweden. Association for Computational Linguistics.  
Angelina Aquino and Franz de Leon. 2020. Parsing in the absence of related languages: Evaluating lowresource dependency parsers on Tagalog. In Proceedings of the Fourth Workshop on Universal Dependencies (UDW 2020), pages 8–15, Barcelona, Spain (Online). Association for Computational Linguistics.  
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. Transactions of the Associationfor Computational Linguistics, 5:135–146.  
Alex Brandsen, Suzan Verberne, Milco Wansleeben, and Karsten Lambers. 2020. Creating a dataset for named entity recognition in the archaeology domain. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 4573–4577, Marseille, France. European Language Resources Association.  
Neil Vicente P. Cabasag, Vicente Raphael C. Chan, Sean Christian Y. Lim, Mark Edward M. Gonzales, and Charibeth K. Cheng. 2019. Hate Speech in Philippine Election-Related Tweets: Automatic Detection and Classification Using Natural Language Processing. Philippine Computing Journal Dedicated Issue on Natural Language Processing, pages 1–14.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440–  
8451, Online. Association for Computational Linguistics.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2022. Improving large-scale language models and resources for Filipino. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6548–6555, Marseille, France. European Language Resources Association.  
Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Ko Cheng. 2020. Exploiting News Article Structure for Automatic Corpus Generation of Entailment Datasets. In Pacific Rim International Conference on Artificial Intelligence.  
Louise Deleger, Qi Li, Todd Lingren, Megan Kaiser, Katalin Molnar, Laura Stoutenborough, Michal Kouril, Keith Marsolo, and Imre Solti. 2012. Building gold standard corpora for medical natural language processing tasks. In AMIA Annual Symposium Proceedings, pages 144–53.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.  
Kenneth C. Enevoldsen, L M Hansen, and Kristoffer Laigaard Nielbo. 2021. DaCy: A Unified Framework for Danish NLP. In Workshop on Computational Humanities Research.  
Hannah Eyre, Alec B. Chapman, Kelly S. Peterson, Jianlin Shi, Patrick R. Alba, Makoto M. Jones, Tamára L Box, Scott L Duvall, and Olga V Patterson. 2021. Launching into clinical space with medspaCy: a new clinical text processing toolkit in Python. Proceedings ofthe AMIA Annual Symposium, 2021:438–447.  
Martin Haspelmath, Matthew Dryer, David Gil, and Comrie Bernard. 2005. The World Atlas of Language Structures. In Oxford University Press.  
Matthew Honnibal, Ines Montani, Sofie Van Lan deghem, and Adriane Boyd. 2020. spaCy: Industrialstrength Natural Language Processing in Python.  
Shengyi Jiang, Yingwen Fu, Xiaotian Lin, and Nankai Lin. 2021. Pre-trained Language Models for Tagalog with Multi-source Data. In Natural Language Processing and Chinese Computing.  
Anne Lauscher, Vinit Ravishankar, Ivan Vulic, and´ Goran Glavaš. 2020. From zero to hero: On the limitations of zero-shot language transfer with mul tilingual Transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4483–4499, Online. Association for Computational Linguistics.  
Paul M. A. Lewis. 2009. Ethnologue: languages of the world. https://ethnologue.com/language/tgl. Accessed: June 2023.  
Evan Dennison S. Livelo and Charibeth Ko Cheng. 2018. Intelligent Dengue Infoveillance Using Gated Recurrent Neural Learning and Cross-Label Frequencies. 2018 IEEE International Conference on Agents (ICA), pages 2–7.  
Lester James V. Miranda, Ákos Kádár, Adriane Boyd, Sofie Van Landeghem, Anders Søgaard, and Matthew Honnibal. 2022. Multi hash embeddings in spaCy. ArXiv, abs/2212.09255.  
Mark Neumann, Daniel King, Iz Beltagy, and Waleed Ammar. 2019. ScispaCy: Fast and robust models for biomedical natural language processing. In Proceedings ofthe 18th BioNLP Workshop and Shared Task, pages 319–327, Florence, Italy. Association for Computational Linguistics.  
Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajivc, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis M. Tyers, and Daniel Zeman. 2022. Data Release Checklist - Universal Dependencies. https://universaldependencies.org/ release\_checklist.html#data-split. Accessed: June 2023.  
György Orosz, Zsolt Szántó, Péter Berkecz, Gergo Szabó, and Richárd Farkas. 2022. HuSpaCy: an industrial-strength Hungarian natural language processing toolkit. ArXiv, abs/2201.01956.  
Nils Reiter. 2017. How to develop annotation guidelines. https://sharedtasksinthedh.github. io/2017/10/01/howto-annotation/. Accessed: June 2023.  
Stephanie Dawn Samson. 2018. A treebank prototype of Tagalog. Bachelor’s thesis, University of Tübingen, Germany.  
Pedro Ortiz Suarez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures. In 7th Workshop on the Challenges in the Management ofLarge Corpora.  
Erik F. Tjong Kim Sang. 2002. Introduction to the CoNLL-2002 shared task: Language-independent named entity recognition. In COLING-02: The 6th Conference on Natural Language Learning 2002 (CoNLL-2002).  
Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, pages 142– 147.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

## A Appendix

## A.1 Reproducibility

All the experiments and models in this paper are available publicly. Readers can head over to https://github.com/ljvmiranda921/ calamanCy for all related software. Note that the XLM-RoBERTa and multilingual BERT experiments may at least require a T4 or V100 GPU.

To reproduce the calamanCy models, head over to models/v0.1.0. To reproduce the benchmarking experiments, head over to the report/benchmark directory. Readers who are interested in the training set-up (e.g., hyperparameters, architectures used, etc.) can check the configuration (.cfg) files in the respective project’s configs/ directory.

## A.2 Building the TLUnified-NER corpus

The TLUnified-NER dataset is a named entity recognition corpus containing the Person (PER), Organization (ORG), and Location (LOC) entities. It includes news articles and other texts in Tagalog from 2009 to 2020. It was based on the TLUnified pretraining corpora by (Cruz and Cheng, 2022). The author, together with two more annotators, annotated TLUnified in the course of four months. We employed an iterative approach as recommended by Reiter (2017), which included resolving disagreements and updating the annotation guidelines.

To get the inter-annotator agreement (IAA) score, we took Brandsen et al. (2020)’s work on the Archaeology dataset as inspiration. We computed Cohen’s κ for all tokens, and only annotated tokens. In addition, we also measured the (3) pairwise F1 score without the ‘O’ label (Deleger et al., 2012).

![](images/cc28fbad0f3240558500796433409ae472e0d98ccb50c540740ed1db7c2daeff.jpg)

<details>
<summary>line</summary>

| Number of examples (cumulative) | Cohen's \(\kappa\) (all tokens) | Cohen's \(\kappa\) (annotated) | F1-score |
| --- | --- | --- | --- |
| 500 | ~0.42 | ~0.25 | ~0.64 |
| 1,000 | ~0.46 | ~0.29 | ~0.68 |
| 2,000 | ~0.63 | ~0.42 | ~0.80 |
| 3,500 | ~0.70 | ~0.53 | ~0.89 |
| 4,000 | ~0.77 | ~0.60 | ~0.92 |
| 5,500 | ~0.81 | ~0.65 | ~0.94 |
</details>

Figure 1: Inter-annotator agreement measurement after each annotation round. Each mark represents the end of a round. For each round, the annotators discuss disagreements, update the annotation guidelines, and evaluate the current set of annotations.

<table><tr><td>Metric</td><td>IAA</td></tr><tr><td>Cohen&#x27;s κ on all tokens</td><td>0.81</td></tr><tr><td>Cohen&#x27;s κ on annotated tokens only</td><td>0.65</td></tr><tr><td>F1 score</td><td>0.91</td></tr></table>

Table 6: Inter-annotator agreement (IAA) measurements. We obtained these values by computing for the pairwise comparisons between all annotator-pairs and averaging the results.

Table 6 shows the IAA measurements while Figure 1 shows their growth after each annotation round.