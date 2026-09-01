# Developing a Named Entity Recognition Dataset for Tagalog

Lester James V. Miranda

ljvmiranda@gmail.com

## Abstract

We present the development of a Named Entity Recognition (NER) dataset for Tagalog. This corpus helps fill the resource gap present in Philippine languages today, where NER resources are scarce. The texts were obtained from a pretraining corpora containing news reports, and were labeled by native speakers in an iterative fashion. The resulting dataset contains ∼7.8k documents across three entity types: Person, Organization, and Location. The inter-annotator agreement, as measured by Cohen's κ, is 0.81. We also conducted extensive empirical evaluation of state-of-the-art methods across supervised and transfer learning settings. Finally, we released the data and processing code publicly to inspire future work on Tagalog NLP.

## 1 Introduction

Tagalog (tl) is one of the major languages in the Philippines with over 28 million speakers in the country [\(Lewis,](#page-6-0) [2009\)](#page-6-0). It constitutes the bulk of Filipino, the country's official language, by sharing its lexical items and grammatical structure. Despite this fact, there are little to no resources for Tagalog [\(Cruz and Cheng,](#page-5-0) [2022\)](#page-5-0), hampering the development of reliable language technologies.

In this paper, we present TLUNIFIED-NER, [1](#page-0-0) a Tagalog dataset for Named Entity Recognition (NER). The texts were obtained from TLUnified [\(Cruz and Cheng,](#page-5-0) [2022\)](#page-5-0), a pretraining corpora containing news reports and other types of text. We focused on NER because of its foundational role in several NLP tasks [\(Tjong Kim Sang and](#page-6-1) [De Meulder,](#page-6-1) [2003;](#page-6-1) [Lample et al.,](#page-6-2) [2016\)](#page-6-2), especially in problems that require the extraction of structured information. TLUNIFIED-NER consists of ∼7.8k documents across three entity types (*Person*, *Organization*, *Location*), modeled closely to

the CoNLL Shared Tasks [\(Tjong Kim Sang,](#page-6-3) [2002;](#page-6-3) [Tjong Kim Sang and De Meulder,](#page-6-1) [2003\)](#page-6-1). Three native speakers conducted the annotation process, resulting to an inter-annotator agreement (IAA) score of 0.81.

We hope that TLUNIFIED-NER will allow researchers to build better NER classifiers for Tagalog, and thereby inspire future research on Tagalog NLP through the following contributions:

- 1. We curated and annotated texts from a large pretraining corpora to represent the modern usage of Tagalog in the news domain.
- 2. We provided performance baselines across a variety of supervised and transfer learning settings.

## 2 Related Work

Tagalog language Tagalog is an agglutinative language within the Austronesian family [\(Kroeger,](#page-6-4) [1992\)](#page-6-4). It uses the Latin script for its writing system with 28 letters in its alphabet. Twenty-six letters are the same as in English, with the addition of Ñ/ñ and Ng/ng. Tagalog typically follows the VSO word order, but VOS and SVO are also accepted [\(Schachter and Otanes,](#page-6-5) [1973\)](#page-6-5). Although Filipino is the country's official language, it has little to no linguistic differences with Tagalog.

Tagalog NER datasets Unfortunately, resources for Tagalog NER are meager. One major resource is WikiANN [\(Pan et al.,](#page-6-6) [2017\)](#page-6-6), a silver-standard corpora based on a framework designed for 282 other languages. However, the Tagalog portion of WikiANN is full of annotation errors, often misconstruing one entity type as another. Another NER dataset is the Filipino Storytelling corpora [\(Cos](#page-5-1)[tiniano et al.,](#page-5-1) [2022\)](#page-5-1). Although gold-standard, its entity labels (e.g., *Humans & Body*, *Natural Environment*, etc.) are too domain-specific for general use. Finally, the LORELEI project also provides

<span id="page-0-0"></span><sup>1</sup>The dataset is accessible at [https://huggingface.co/](https://huggingface.co/datasets/ljvmiranda921/tlunified-ner) [datasets/ljvmiranda921/tlunified-ner](https://huggingface.co/datasets/ljvmiranda921/tlunified-ner)

<span id="page-1-2"></span>

Table 1: Entity types used for annotating TLUNIFIED-NER (derived from the TLUnified pretraining corpus of [Cruz and Cheng,](#page-5-0) [2022\)](#page-5-0).

<span id="page-1-1"></span>![](_page_1_Picture_2.jpeg)

language packs for Tagalog [\(Strassel and Tracey,](#page-6-7) [2016\)](#page-6-7), but they're not publicly-accessible.

TLUNIFIED-NER aims to fill this resource gap by providing a publicly-assessible gold standard resource for Tagalog NER.

## 3 Dataset Collection

The texts were obtained from [Cruz and Cheng](#page-5-0) [\(2022\)](#page-5-0)'s TLUnified pretraining corpora. It combines news reports [\(Cruz et al.,](#page-5-2) [2020\)](#page-5-2), a preprocessed version of CommonCrawl [\(Suarez et al.,](#page-6-8) [2019\)](#page-6-8), and several other datasets. We manually filtered this dataset to contain news reports so as to resemble the CoNLL Shared Tasks [\(Tjong Kim Sang,](#page-6-3) [2002;](#page-6-3) [Tjong Kim Sang and De Meulder,](#page-6-1) [2003\)](#page-6-1).

The texts are diverse. It contains articles from different news sites online that ran a published print media or news channel in Metro Manila from 2009 to early 2020. The topics range from politics, weather, and popular science among others.

## <span id="page-1-3"></span>4 Annotation Setup

We used Prodigy as our annotation tool.[<sup>2</sup>](#page-1-0) We set up a web server on the Google Cloud Platform and routed the examples through Prodigy's builtin task router. Figure [1](#page-1-1) shows the labeling interface as seen by the annotator. Finally, we used the ner.manual recipe to highlight spans during the annotation process. We used three entity labels for TLUNIFIED-NER as shown in Table [1.](#page-1-2) Unlike CoNLL, we decided to exclude the *Miscellaneous (MISC)* tag to reduce confusion.

Annotation Process The annotation process was done iteratively with three annotators (including the author) who are native Tagalog speakers. Given

Figure 1: Prodigy's annotation interface for a given text. (Translation: *MANILA - The owner of the illegal billboards that fell on EDSA this Monday, injuring five people and damaging property, should be caught and imprisoned according to Senator Miriam Defensor Santiago.*)

a set annotation budget, we paid the annotators above the country's minimum daily wage. Each annotation round spans for two to three weeks, for a total of six rounds (18 weeks). The annotators labeled the same batch of examples to ensure high overlap.

After each round, the annotators hold a retrospective meeting and discussed examples they found confusing, inconsistent with the annotation guidelines, and noteworthy. This process continued until we reached ∼10k examples or if we exhausted our annotation budget. In addition, we also tracked the training curve to determine the quality of the collected annotations. If the F1-score improved within the last 25% of the training data, then it is a good sign that obtaining more labels will result to better accuracy.

Annotation Guidelines We developed the annotation guidelines in an iterative fashion.

<span id="page-1-0"></span><sup>2</sup> <https://prodigy.ai>

<span id="page-2-0"></span>

Table 2: Dataset statistics for TLUNIFIED-NER. It shows the number of examples, number of tokens, and spanlevel statistics. SD stands for span distinctiveness whereas BD is boundary distinctiveness [\(Papay et al.,](#page-6-9) [2020\)](#page-6-9).

<span id="page-2-1"></span>

Table 3: Inter-annotator agreement (IAA) measurements. We obtained these values by computing for the pairwise comparisons on all annotator-pairs and averaging the results.

The Automatic Content Extraction (ACE 2004/05) annotation document [\(Doddington](#page-6-10) [et al.,](#page-6-10) [2004\)](#page-6-10) heavily inspired our initial draft. We co-developed the guidelines after each annotation round to improve clarity and reduce disagreements. These guidelines are accessible on GitHub: [https://github.com/ljvmiranda921/](https://github.com/ljvmiranda921/calamanCy/tree/master/datasets/tl_calamancy_gold_corpus/guidelines) [calamanCy/tree/master/datasets/tl\\_](https://github.com/ljvmiranda921/calamanCy/tree/master/datasets/tl_calamancy_gold_corpus/guidelines) [calamancy\\_gold\\_corpus/guidelines](https://github.com/ljvmiranda921/calamanCy/tree/master/datasets/tl_calamancy_gold_corpus/guidelines)

## 5 Corpus Statistics and Evaluation

Table [2](#page-2-0) shows the final dataset statistics for TLUNIFIED-NER. We also included span- (SD) and boundary-distinctiveness (BD) metrics [\(Papay](#page-6-9) [et al.,](#page-6-9) [2020\)](#page-6-9). They measure the KL-divergence of the unigram word distributions between the span (or its boundaries) and the rest of the corpora. These metrics can be used to gauge the difficulty of the span labeling task, (e.g., more distinct spans means it's "easier" to detect them in the text).

### 5.1 Inter-annotator Agreement (IAA)

Similar to [Brandsen et al.](#page-5-3) [\(2020\)](#page-5-3), we measured two types of Cohen's κ. The first metric calculates κ for tokens where at least one annotator has made an annotation. The second metric computes for all tokens while ignoring the 'O' label. In addition, we had a third measure: the F1-score using one set of annotations as reference [\(Deleger et al.,](#page-5-4) [2012\)](#page-5-4). We did these computations for each annotator-pair and averaged the results as shown in Table [3.](#page-2-1)

Finally, Figure [2](#page-2-2) shows the growth of IAA for

<span id="page-2-2"></span>![](_page_2_Figure_3.jpeg)

Figure 2: Growth of IAA for each annotation round.

each annotation round. Because of our annotation process, we were able to label the same batch of documents and track the agreement every round.

### <span id="page-2-3"></span>5.2 Benchmark results

We trained several NER models using spaCy's transition-based parser [\(Honnibal et al.,](#page-6-11) [2020\)](#page-6-11). The state transitions are based on the BILUO sequence encoding scheme and the actions are decided by a convolutional neural network with a maxout [\(Good](#page-6-12)[fellow et al.,](#page-6-12) [2013\)](#page-6-12) activation function.

While keeping the NER classifier constant, we experimented with various word embeddings that led to the following configurations:

- Baseline: we trained the transition-based parser "from scratch" without additional information from static or context-sensitive vectors.
- Static vectors: we used Tagalog fastText vectors [\(Bojanowski et al.,](#page-5-5) [2017\)](#page-5-5) and included a simple pretraining process to initialize the

<span id="page-3-0"></span>

Table 4: Benchmark results on TLUNIFIED-NER across different word embeddings using spaCy's transitionbased parser [\(Honnibal et al.,](#page-6-11) [2020\)](#page-6-11). Reported results are F1-scores on the test set across three trials.

<span id="page-3-1"></span>![](_page_3_Figure_3.jpeg)

Figure 3: Development set confusion matrix of the Baseline model predictions in the IOB format.

weights of the model. The pretraining objective asks the model to predict some number of leading and trailing UTF-8 bytes for the words—a variant of the cloze task.

- Transformer-based vectors (monolingual): we used RoBERTa Tagalog [\(Cruz and Cheng,](#page-5-0) [2022\)](#page-5-0), the only pretrained language model for Tagalog, and finetuned it with our annotations.
- Transformer-based vectors (multilingual): we tested on XLM-RoBERTa [\(Conneau et al.,](#page-5-6) [2020\)](#page-5-6) and multilingual BERT [\(Devlin et al.,](#page-5-7) [2019\)](#page-5-7) for transfer learning. These models include Tagalog in their training pool albeit underrepresented.

This experimental setup allows us to see the expected performance when training Tagalog NER classifiers using standard techniques. Table [4](#page-3-0) reports the F1-score on the test set across three trials.

<span id="page-3-2"></span>

Table 5: Relative error reduction (with respect to the Baseline) for classifying ORG and LOC entities. Reported results are F1-scores on the development set.

### 5.3 Error analysis

From our benchmark results, we noticed that most models are having trouble predicting the *Location* or *Organization* tags. Figure [3](#page-3-1) shows the confusion matrix of the Baseline model on the development set in the IOB format.

Most of the mistakes came from incorrectly tagging a token with the outside 'O' label. However, we also noticed instances where the model confuses between the lexical and semantic tag of an entity. For example, in the span, *". . . panukala ng Ombudsman. . . "* (". . . proposed by the Ombudsman. . . "), the token Ombudsman might be a Person or Organization depending on the context. We hypothesize that including context-sensitive training, which the baseline model lacks, can help mitigate this issue.

To test this hypothesis, we experimented on two training configurations. First, we trained a POS tagger together with our transition-based NER with shared weights. This process may help provide extra information to the transition-based parser so it can disambiguate between entities. Second, we finetuned context-sensitive vectors from RoBERTa Tagalog [\(Cruz and Cheng,](#page-5-0) [2022\)](#page-5-0) for NER. Table [5](#page-3-2) shows the relative error reduction between LOC and ORG entities. Given these results, we encourage researchers to utilize context-sensitive vectors such as RoBERTa Tagalog (or other BERT variants) when training models from this corpora.

<span id="page-4-0"></span>

Table 6: Cross-dataset comparison between WikiANN [\(Pan et al.,](#page-6-6) [2017\)](#page-6-6) and TLUNIFIED-NER. We trained a model from WikiANN then applied it to TLUNIFIED-NER (and vice-versa). Reported results are F1-scores on the test set across three trials.

<span id="page-4-1"></span>

Table 7: Comparing the overlap between the original (silver-standard) WikiANN annotations against our reannotated version.

### 5.4 Comparison to WikiANN

The WikiANN dataset [\(Pan et al.,](#page-6-6) [2017\)](#page-6-6) is another resource for Tagalog NER. However, we found many annotation errors in the dataset, from misclassifications to fragmented sentences. We investigated how TLUNIFIED-NER fares against WikiANN's silver-standard annotations.

We finetuned several models similar to Section [5.2](#page-2-3) on the Tagalog portion of WikiANN's training set and tested it on TLUNIFIED-NER's test set (and vice-versa). In order to properly evaluate on WikiANN, we reannotated the test dataset using the same annotation guidelines described in Section [4.](#page-1-3)

Our results in Table [6](#page-4-0) suggest that models built from the TLUNIFIED-NER corpus are more performant than with WikiANN. Additionally, the gap between WikiANN's silver-standard annotations and our corrections is large, as shown in Table [7.](#page-4-1) We then posit that the gold-standard nature of TLUNIFIED-NER led to better performance than WikiANN, which predominantly consists of text fragments and low-quality annotations.

### 5.5 Experiments on large language models

Large language models (LLMs) have been shown to exhibit multilingual capabilities—incidental or not [\(Briakou et al.,](#page-5-8) [2023\)](#page-5-8). We investigated this property by performing a zero-shot prompting ap-

<span id="page-4-2"></span>

Table 8: Benchmark results on TLUNIFIED-NER across a variety of open-source and commercial LLMs. We used the 7B-parameter variants for models denoted with an asterisk (∗) due to budget constraints.

proach on TLUNIFIED-NER's test set across a variety of commercial and open-source LLMs. Table [8](#page-4-2) reports the F1-score across three trials.

Our results suggest that supervised learning reliably outperforms zero-shot prompting for TLUNIFIED-NER given our prompt (see Appendix [A.1\)](#page-6-16). However, we acknowledge that these results are not a definitive comparison between two methods as prompt engineering is unstable with high variance [\(Webson and Pavlick,](#page-6-17) [2022;](#page-6-17) [Zhao et al.,](#page-6-18) [2021\)](#page-6-18). In the future, we plan to explore different prompting techniques such as PromptNER [\(Ashok](#page-5-13) [and Lipton,](#page-5-13) [2023\)](#page-5-13) and chain-of-thought [\(Wei et al.,](#page-6-19) [2023\)](#page-6-19) to uncover the language models' full capabilities.

### 6 Conclusion

In this paper, we introduced TLUNIFIED-NER, a Named Entity Recognition dataset for Tagalog. Unlike other Tagalog NER datasets, TLUNIFIED-NER is publicly-accessible and gold standard. Our iterative annotation process, together with our interannotator agreement, shows that the corpus is of high quality. In addition, our benchmarking results suggest that the task is learnable even with a simple baseline method. We hope that TLUNIFIED-NER fills the resource gap present in Tagalog NLP today. In the future, we plan to create a more finegrained (and perhaps, overlapping) NER tag set similar to the ACE project and expand on other major Philippine languages. Finally, the dataset is available online ([https://huggingface.co/](https://huggingface.co/datasets/ljvmiranda921/tlunified-ner) [datasets/ljvmiranda921/tlunified-ner](https://huggingface.co/datasets/ljvmiranda921/tlunified-ner)) and we encourage researchers to improve upon our benchmark results. Limitations The TLUNIFIED-NER corpora is comprised mostly by news reports. Although the texts demonstrate the standard usage of Tagalog, its domain is limited. In addition, we only trained a transitionbased parser model for our NER classifier. In the future, we plan to extend these benchmarks and include CRFs or other tools such as Stanford Stanza. Acknowledgements We would like to express our gratitute to all those who contributed to the completion of this resource. We extend our appreciation to the anonymous reviewers for their constructive comments, which greatly improved the quality of this paper. References Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. Falcon-40B: an open large language model with state-of-the-art performance. Anthropic. 2023. [Model card and evaluations for](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf) [claude models.](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf) Dhananjay Ashok and Zachary C. Lipton. 2023. [Promptner: Prompting for named entity recognition.](http://arxiv.org/abs/2305.15444) Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. [Enriching Word Vectors with](https://doi.org/10.1162/tacl_a_00051) [Subword Information.](https://doi.org/10.1162/tacl_a_00051) *Transactions of the Association for Computational Linguistics*, 5:135–146. Alex Brandsen, Suzan Verberne, Milco Wansleeben, and Karsten Lambers. 2020. [Creating a Dataset](https://aclanthology.org/2020.lrec-1.562) [for Named Entity Recognition in the Archaeology](https://aclanthology.org/2020.lrec-1.562) [Domain.](https://aclanthology.org/2020.lrec-1.562) In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 4573– 4577, Marseille, France. European Language Resources Association. Eleftheria Briakou, Colin Cherry, and George Foster. 2023. [Searching for needles in a haystack: On](https://doi.org/10.18653/v1/2023.acl-long.524) [the role of incidental bilingualism in PaLM's trans](https://doi.org/10.18653/v1/2023.acl-long.524)[lation capability.](https://doi.org/10.18653/v1/2023.acl-long.524) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9432– 9452, Toronto, Canada. Association for Computational Linguistics. Cohere. 2023. [Command Model: The AI-Powered So](https://cohere.com/models/command)[lution for the Enterprise.](https://cohere.com/models/command) Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. [Unsupervised](https://doi.org/10.18653/v1/2020.acl-main.747) [Cross-lingual Representation Learning at Scale.](https://doi.org/10.18653/v1/2020.acl-main.747) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8440– 8451, Online. Association for Computational Linguistics. Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. [Free dolly:](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) [Introducing the world's first truly open instruction](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)[tuned llm.](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) Sherwyne Costiniano, Rose Ann Mae Santos, Julius Simon Mendoza, and Allen Jay Gale. 2022. [Custom](https://api.semanticscholar.org/CorpusID:255117774) [Coarse Grained Named Entity Recognition for Fil](https://api.semanticscholar.org/CorpusID:255117774)[ipino Storytelling Data Using Uncased Transformer](https://api.semanticscholar.org/CorpusID:255117774) [Models.](https://api.semanticscholar.org/CorpusID:255117774) *SSRN Electronic Journal*. Jan Christian Blaise Cruz and Charibeth Cheng. 2022. [Improving Large-scale Language Models and Re](https://aclanthology.org/2022.lrec-1.703)[sources for Filipino.](https://aclanthology.org/2022.lrec-1.703) In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 6548–6555, Marseille, France. European Language Resources Association. Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Ko Cheng. 2020. Exploiting News Article Structure for Automatic Corpus Generation of Entailment Datasets. In *Pacific Rim International Conference on Artificial Intelligence*. Louise Deleger, Qi Li, Todd Lingren, Megan Kaiser, Katalin Molnar, Laura Stoutenborough, Michal Kouril, Keith Marsolo, and Imre Solti. 2012. Building gold standard corpora for medical natural language processing tasks. In *AMIA Annual Symposium Proceedings*, pages 144–53. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. [BERT: Pre-training of](https://doi.org/10.18653/v1/N19-1423) [deep bidirectional transformers for language under](https://doi.org/10.18653/v1/N19-1423)[standing.](https://doi.org/10.18653/v1/N19-1423) In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language*

<span id="page-5-8"></span>

<span id="page-5-10"></span><span id="page-5-6"></span>

<span id="page-5-13"></span><span id="page-5-12"></span><span id="page-5-11"></span><span id="page-5-9"></span><span id="page-5-7"></span><span id="page-5-5"></span><span id="page-5-4"></span><span id="page-5-3"></span><span id="page-5-2"></span><span id="page-5-1"></span><span id="page-5-0"></span>

- <span id="page-6-15"></span><span id="page-6-12"></span><span id="page-6-11"></span><span id="page-6-10"></span><span id="page-6-4"></span><span id="page-6-2"></span>*Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. George Doddington, Alexis Mitchell, Mark Przybocki, Lance Ramshaw, Stephanie Strassel, and Ralph Weischedel. 2004. [The automatic content extraction](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf) [\(ACE\) program – tasks, data, and evaluation.](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf) In *Proceedings of the Fourth International Conference on Language Resources and Evaluation (LREC'04)*, Lisbon, Portugal. European Language Resources Association (ELRA). Xinyang Geng and Hao Liu. 2023. [OpenLLaMA: An](https://github.com/openlm-research/open_llama) [Open Reproduction of LLaMA.](https://github.com/openlm-research/open_llama) Ian Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. 2013. [Maxout](https://proceedings.mlr.press/v28/goodfellow13.html) [networks.](https://proceedings.mlr.press/v28/goodfellow13.html) In *Proceedings of the 30th International Conference on Machine Learning*, volume 28 of *Proceedings of Machine Learning Research*, pages 1319–1327, Atlanta, Georgia, USA. PMLR. Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. [spaCy:](https://doi.org/10.5281/zenodo.1212303) [Industrial-strength Natural Language Processing in](https://doi.org/10.5281/zenodo.1212303) [Python.](https://doi.org/10.5281/zenodo.1212303) Paul R. Kroeger. 1992. [Phrase Structure and Grammat](https://api.semanticscholar.org/CorpusID:60973484)[ical Relations in Tagalog.](https://api.semanticscholar.org/CorpusID:60973484) Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. [Neural architectures for named entity recognition.](https://doi.org/10.18653/v1/N16-1030) In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 260–270, San Diego, California. Association for Computational Linguistics. Paul M. A. Lewis. 2009. Ethnologue: languages of the world. <https://ethnologue.com/language/tgl>. Accessed: June 2023. OpenAI. 2023. [GPT-4 Technical Report.](http://arxiv.org/abs/2303.08774) Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. [Cross](https://doi.org/10.18653/v1/P17-1178)[lingual name tagging and linking for 282 languages.](https://doi.org/10.18653/v1/P17-1178) In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1946–1958, Vancouver, Canada. Association for Computational Linguistics. Sean Papay, Roman Klinger, and Sebastian Padó. 2020. [Dissecting span identification tasks with per](https://doi.org/10.18653/v1/2020.emnlp-main.396)[formance prediction.](https://doi.org/10.18653/v1/2020.emnlp-main.396) In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4881–4895, Online. Association for Computational Linguistics. Paul Schachter and Fe T. Otanes. 1973. [Tagalog refer](https://api.semanticscholar.org/CorpusID:162393304)[ence grammar.](https://api.semanticscholar.org/CorpusID:162393304) *The Journal of Asian Studies*, 32:760
- <span id="page-6-8"></span><span id="page-6-7"></span><span id="page-6-3"></span><span id="page-6-1"></span>– 761. Stephanie Strassel and Jennifer Tracey. 2016. [LORELEI language packs: Data, tools, and](https://aclanthology.org/L16-1521) [resources for technology development in low](https://aclanthology.org/L16-1521) [resource languages.](https://aclanthology.org/L16-1521) In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)*, pages 3273–3280, Portorož, Slovenia. European Language Resources Association (ELRA). Pedro Ortiz Suarez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures. In *7th Workshop on the Challenges in the Management of Large Corpora*. Erik F. Tjong Kim Sang. 2002. [Introduction to the](https://aclanthology.org/W02-2024) [CoNLL-2002 shared task: Language-independent](https://aclanthology.org/W02-2024) [named entity recognition.](https://aclanthology.org/W02-2024) In *COLING-02: The 6th Conference on Natural Language Learning 2002 (CoNLL-2002)*. Erik F. Tjong Kim Sang and Fien De Meulder. 2003. [Introduction to the CoNLL-2003 shared task:](https://aclanthology.org/W03-0419) [Language-independent named entity recognition.](https://aclanthology.org/W03-0419) In *Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003*, pages 142–147. Albert Webson and Ellie Pavlick. 2022. [Do prompt](https://doi.org/10.18653/v1/2022.naacl-main.167)[based models really understand the meaning of their](https://doi.org/10.18653/v1/2022.naacl-main.167) [prompts?](https://doi.org/10.18653/v1/2022.naacl-main.167) In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2300–2344, Seattle, United States. Association for Computational Linguistics. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. [Chain-of-thought prompt](http://arxiv.org/abs/2201.11903)[ing elicits reasoning in large language models.](http://arxiv.org/abs/2201.11903) Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. [Calibrate before use: Improv](https://proceedings.mlr.press/v139/zhao21c.html)[ing few-shot performance of language models.](https://proceedings.mlr.press/v139/zhao21c.html) In *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 12697–12706. PMLR. A Appendix A.1 Zero-shot prompt template *You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entities for the set of predefined entity labels. From the Text input provided, extract named entities for each label in the following format:*
  - *PER: <comma delimited list of strings>*
  - *ORG: <comma delimited list of strings>*
  - *LOC: <comma delimited list of strings> Below are definitions of each label to help aid*

<span id="page-6-19"></span><span id="page-6-18"></span><span id="page-6-17"></span><span id="page-6-16"></span><span id="page-6-13"></span><span id="page-6-9"></span><span id="page-6-6"></span><span id="page-6-0"></span>

<span id="page-6-14"></span><span id="page-6-5"></span>*you in what kinds of named entities to extract for*

*each label. Assume these definitions are written by an expert and follow them closely.*

- *PER: PERSON*
- *ORG: ORGANIZATION*
- *LOC: LOCATION OR GEOPOLITICAL EN-TITY Text: {{ text }}*

### A.2 Reproducibility

All the experiments and models in this paper are available publicly. Readers can head over to [https://github.com/ljvmiranda921/](https://github.com/ljvmiranda921/calamanCy/reports/aacl) [calamanCy/reports/aacl](https://github.com/ljvmiranda921/calamanCy/reports/aacl) for all related code and assets. Note that the XLM-RoBERTa and multilingual BERT experiments may at least require a T4 or V100 GPU.