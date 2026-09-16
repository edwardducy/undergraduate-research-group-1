# Developing a Named Entity Recognition Dataset for Tagalog

Lester James V. Miranda

ljvmiranda@gmail.com

## Abstract

We present the development of a Named Entity Recognition (NER) dataset for Tagalog. This corpus helps fill the resource gap present in Philippine languages today, where NER resources are scarce. The texts were obtained from a pretraining corpora containing news reports, and were labeled by native speakers in an iterative fashion. The resulting dataset contains ∼7.8k documents across three entity types: Person, Organization, and Location. The inter-annotator agreement, as measured by Cohen’s κ, is 0.81. We also conducted extensive empirical evaluation of state-of-the-art methods across supervised and transfer learning settings. Finally, we released the data and processing code publicly to inspire future work on Tagalog NLP.

## 1 Introduction

Tagalog (tl) is one of the major languages in the Philippines with over 28 million speakers in the country (Lewis, 2009). It constitutes the bulk of Filipino, the country’s official language, by sharing its lexical items and grammatical structure. Despite this fact, there are little to no resources for Tagalog (Cruz and Cheng, 2022), hampering the development of reliable language technologies.

In this paper, we present TLUNIFIED-NER,<sup>1</sup> a Tagalog dataset for Named Entity Recognition (NER). The texts were obtained from TLUnified (Cruz and Cheng, 2022), a pretraining corpora containing news reports and other types of text. We focused on NER because of its foundational role in several NLP tasks (Tjong Kim Sang and De Meulder, 2003; Lample et al., 2016), especially in problems that require the extraction of structured information. TLUNIFIED-NER consists of ∼7.8k documents across three entity types (Person, Organization, Location), modeled closely to the CoNLL Shared Tasks (Tjong Kim Sang, 2002; Tjong Kim Sang and De Meulder, 2003). Three native speakers conducted the annotation process, resulting to an inter-annotator agreement (IAA) score of 0.81.

We hope that TLUNIFIED-NER will allow researchers to build better NER classifiers for Tagalog, and thereby inspire future research on Tagalog NLP through the following contributions:

1. We curated and annotated texts from a large pretraining corpora to represent the modern usage of Tagalog in the news domain.  
2. We provided performance baselines across a variety of supervised and transfer learning settings.

## 2 Related Work

Tagalog language Tagalog is an agglutinative language within the Austronesian family (Kroeger, 1992). It uses the Latin script for its writing system with 28 letters in its alphabet. Twenty-six letters are the same as in English, with the addition of Ñ/ñ and Ng/ng. Tagalog typically follows the VSO word order, but VOS and SVO are also accepted (Schachter and Otanes, 1973). Although Filipino is the country’s official language, it has little to no linguistic differences with Tagalog.

Tagalog NER datasets Unfortunately, resources for Tagalog NER are meager. One major resource is WikiANN (Pan et al., 2017), a silver-standard corpora based on a framework designed for 282 other languages. However, the Tagalog portion of WikiANN is full of annotation errors, often misconstruing one entity type as another. Another NER dataset is the Filipino Storytelling corpora (Costiniano et al., 2022). Although gold-standard, its entity labels (e.g., Humans & Body, Natural Environment, etc.) are too domain-specific for general use. Finally, the LORELEI project also provides language packs for Tagalog (Strassel and Tracey, 2016), but they’re not publicly-accessible.

<table><tr><td>Entity</td><td>Short Description</td><td>Examples</td></tr><tr><td>Person (PER)</td><td>Person entities limited to humans. It may be a single individual or group.</td><td>Juan de la Cruz, Jose Rizal, Quijano de Manila</td></tr><tr><td>Organization (ORG)</td><td>Organization entities limited to corporations, agencies, and other groups of people defined by an organizational structure.</td><td>Meralco, DPWH, United Nations</td></tr><tr><td>Location (LOC)</td><td>Location entities are geographical regions, areas, and landmasses. Geo-political entities are also included within this group.</td><td>Pilipinas, Manila, CAL-ABARZON, Ilog Pasig</td></tr></table>

Table 1: Entity types used for annotating TLUNIFIED-NER (derived from the TLUnified pretraining corpus of Cruz and Cheng, 2022).

TLUNIFIED-NER aims to fill this resource gap by providing a publicly-assessible gold standard resource for Tagalog NER.

## 3 Dataset Collection

The texts were obtained from Cruz and Cheng (2022)’s TLUnified pretraining corpora. It combines news reports (Cruz et al., 2020), a preprocessed version of CommonCrawl (Suarez et al., 2019), and several other datasets. We manually filtered this dataset to contain news reports so as to resemble the CoNLL Shared Tasks (Tjong Kim Sang, 2002; Tjong Kim Sang and De Meulder, 2003).

The texts are diverse. It contains articles from different news sites online that ran a published print media or news channel in Metro Manila from 2009 to early 2020. The topics range from politics, weather, and popular science among others.

## 4 Annotation Setup

We used Prodigy as our annotation tool.<sup>2</sup> We set up a web server on the Google Cloud Platform and routed the examples through Prodigy’s builtin task router. Figure 1 shows the labeling interface as seen by the annotator. Finally, we used the ner.manual recipe to highlight spans during the annotation process. We used three entity labels for TLUNIFIED-NER as shown in Table 1. Unlike CoNLL, we decided to exclude the Miscellaneous (MISC) tag to reduce confusion.

Annotation Process The annotation process was done iteratively with three annotators (including the author) who are native Tagalog speakers. Given a set annotation budget, we paid the annotators above the country’s minimum daily wage. Each annotation round spans for two to three weeks, for a total of six rounds (18 weeks). The annotators labeled the same batch of examples to ensure high overlap.

![](images/a11523034ec1e6d6fc4e30521222a12f47973cfc681a998089fe05a466bacbec.jpg)

<details>
<summary>text_image</summary>

LOC 1 ORG 2 PER 3
MANILA LOC - Dapat umanong hulihin at ikulong ang may-ari ng mga illegal billboard na bumagsak sa EDSA LOC nitong Lunes na ikinasugat ng lima katao at nakasira ng ari-arian, ayon kay Senador Miriam Defensor Santiago PER .
</details>

Figure 1: Prodigy’s annotation interface for a given text. (Translation: MANILA - The owner ofthe illegal billboards thatfell on EDSA this Monday, injuringfive people and damaging property, should be caught and imprisoned according to Senator Miriam Defensor Santiago.)

After each round, the annotators hold a retrospective meeting and discussed examples they found confusing, inconsistent with the annotation guidelines, and noteworthy. This process continued until we reached ∼10k examples or if we exhausted our annotation budget. In addition, we also tracked the training curve to determine the quality of the collected annotations. If the F1-score improved within the last 25% of the training data, then it is a good sign that obtaining more labels will result to better accuracy.

Annotation Guidelines We developed the annotation guidelines in an iterative fashion.

<table><tr><td>Dataset</td><td>Examples</td><td>Tokens</td><td>PER</td><td>ORG</td><td>LOC</td><td>Length</td><td>SD</td><td>BD</td></tr><tr><td>Training</td><td>6252</td><td>198579</td><td>6418</td><td>3121</td><td>3296</td><td>1.49</td><td>2.66</td><td>1.26</td></tr><tr><td>Development</td><td>782</td><td>25069</td><td>793</td><td>392</td><td>409</td><td>1.51</td><td>2.77</td><td>1.37</td></tr><tr><td>Test</td><td>782</td><td>25100</td><td>818</td><td>423</td><td>438</td><td>1.48</td><td>2.77</td><td>1.34</td></tr></table>

Table 2: Dataset statistics for TLUNIFIED-NER. It shows the number of examples, number of tokens, and spanlevel statistics. SD stands for span distinctiveness whereas BD is boundary distinctiveness (Papay et al., 2020).

<table><tr><td>Metric</td><td>IAA</td></tr><tr><td>Cohen&#x27;s κ on all tokens</td><td>0.81</td></tr><tr><td>Cohen&#x27;s κ on annotated tokens only</td><td>0.65</td></tr><tr><td>F1 score</td><td>0.91</td></tr></table>

Table 3: Inter-annotator agreement (IAA) measurements. We obtained these values by computing for the pairwise comparisons on all annotator-pairs and averaging the results.

The Automatic Content Extraction (ACE 2004/05) annotation document (Doddington et al., 2004) heavily inspired our initial draft. We co-developed the guidelines after each annotation round to improve clarity and reduce disagreements. These guidelines are accessible on GitHub: https://github.com/ljvmiranda921/ calamanCy/tree/master/datasets/tl\_ calamancy\_gold\_corpus/guidelines

## 5 Corpus Statistics and Evaluation

Table 2 shows the final dataset statistics for TLUNIFIED-NER. We also included span- (SD) and boundary-distinctiveness (BD) metrics (Papay et al., 2020). They measure the KL-divergence of the unigram word distributions between the span (or its boundaries) and the rest of the corpora. These metrics can be used to gauge the difficulty of the span labeling task, (e.g., more distinct spans means it’s “easier” to detect them in the text).

## 5.1 Inter-annotator Agreement (IAA)

Similar to Brandsen et al. (2020), we measured two types of Cohen’s κ. The first metric calculates κ for tokens where at least one annotator has made an annotation. The second metric computes for all tokens while ignoring the ‘O’ label. In addition, we had a third measure: the F1-score using one set of annotations as reference (Deleger et al., 2012). We did these computations for each annotator-pair and averaged the results as shown in Table 3.

Finally, Figure 2 shows the growth of IAA for each annotation round. Because of our annotation process, we were able to label the same batch of documents and track the agreement every round.

![](images/be916391d7bce521db8b25d9d95d60664be729e88e65da1d67f26a9a41dec5cb.jpg)

<details>
<summary>line</summary>

| Number of examples | F1-score | Cohen's Kappa (all tokens) | Cohen's Kappa (annotated only) |
| --- | --- | --- | --- |
| 500 | ~0.64 | ~0.42 | ~0.25 |
| 1000 | ~0.68 | ~0.46 | ~0.29 |
| 2000 | ~0.80 | ~0.63 | ~0.42 |
| 3500 | ~0.89 | ~0.70 | ~0.53 |
| 4000 | ~0.92 | ~0.77 | ~0.60 |
| 6000 | ~0.96 | ~0.84 | ~0.67 |
</details>

Figure 2: Growth of IAA for each annotation round.

## 5.2 Benchmark results

We trained several NER models using spaCy’s transition-based parser (Honnibal et al., 2020). The state transitions are based on the BILUO sequence encoding scheme and the actions are decided by a convolutional neural network with a maxout (Goodfellow et al., 2013) activation function.

While keeping the NER classifier constant, we experimented with various word embeddings that led to the following configurations:

• Baseline: we trained the transition-based parser “from scratch” without additional information from static or context-sensitive vectors.  
• Static vectors: we used Tagalog fastText vectors (Bojanowski et al., 2017) and included a simple pretraining process to initialize the

<table><tr><td>Word Embeddings</td><td>Person</td><td>Organization</td><td>Location</td><td>Overall</td></tr><tr><td>Baseline (no additional embeddings)</td><td>87.85±0.01</td><td>74.80±0.02</td><td>81.03±0.01</td><td>84.57±0.02</td></tr><tr><td>fastText (Bojanowski et al., 2017)</td><td>91.20±0.02</td><td>85.39±0.03</td><td>88.38±0.01</td><td>88.90±0.01</td></tr><tr><td>RoBERTa Tagalog (Cruz and Cheng, 2022)</td><td>92.18±0.01</td><td>87.30±0.00</td><td>90.01±0.02</td><td>90.34±0.02</td></tr><tr><td>XLM-RoBERTa (Conneau et al., 2020)</td><td>91.95±0.04</td><td>84.84±0.02</td><td>88.92±0.01</td><td>88.03±0.03</td></tr><tr><td>Multilingual BERT (Devlin et al., 2019)</td><td>90.78±0.03</td><td>85.08±0.01</td><td>88.45±0.03</td><td>87.40±0.02</td></tr></table>

Table 4: Benchmark results on TLUNIFIED-NER across different word embeddings using spaCy’s transitionbased parser (Honnibal et al., 2020). Reported results are F1-scores on the test set across three trials.

![](images/c914d72f69793cec0b5cf99553265e151d4fe866fbe52be92904f519d12e315b.jpg)

<details>
<summary>confusion</summary>

| Reference annotations | B-PER | I-PER | B-ORG | I-ORG | B-LOC | I-LOC | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-PER | 0.90 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.09 |
| I-PER | 0.02 | 0.90 | 0.00 | 0.01 | 0.00 | 0.00 | 0.06 |
| B-ORG | 0.01 | 0.00 | 0.82 | 0.01 | 0.01 | 0.00 | 0.16 |
| I-ORG | 0.00 | 0.01 | 0.01 | 0.86 | 0.00 | 0.01 | 0.11 |
| B-LOC | 0.01 | 0.00 | 0.02 | 0.00 | 0.85 | 0.01 | 0.10 |
| I-LOC | 0.00 | 0.01 | 0.00 | 0.03 | 0.04 | 0.78 | 0.14 |
| O | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.99 |
</details>

Figure 3: Development set confusion matrix of the Baseline model predictions in the IOB format.

weights of the model. The pretraining objective asks the model to predict some number of leading and trailing UTF-8 bytes for the words—a variant of the cloze task.

• Transformer-based vectors (monolingual): we used RoBERTa Tagalog (Cruz and Cheng, 2022), the only pretrained language model for Tagalog, and finetuned it with our annotations.  
• Transformer-based vectors (multilingual): we tested on XLM-RoBERTa (Conneau et al., 2020) and multilingual BERT (Devlin et al., 2019) for transfer learning. These models include Tagalog in their training pool albeit underrepresented.

This experimental setup allows us to see the expected performance when training Tagalog NER classifiers using standard techniques. Table 4 reports the F1-score on the test set across three trials.

<table><tr><td rowspan="2">Embeddings set-up</td><td colspan="2">Rel. error reduction</td></tr><tr><td>ORG</td><td>LOC</td></tr><tr><td>Shared</td><td>+5%</td><td>+3%</td></tr><tr><td>Context-sensitive</td><td>+12%</td><td>+18%</td></tr></table>

Table 5: Relative error reduction (with respect to the Baseline) for classifying ORG and LOC entities. Reported results are F1-scores on the development set.

## 5.3 Error analysis

From our benchmark results, we noticed that most models are having trouble predicting the Location or Organization tags. Figure 3 shows the confusion matrix of the Baseline model on the development set in the IOB format.

Most of the mistakes came from incorrectly tagging a token with the outside ‘O’ label. However, we also noticed instances where the model confuses between the lexical and semantic tag of an entity. For example, in the span, “. . .panukala ng Ombudsman. . . ” (“. . . proposed by the Ombudsman. . . ”), the token Ombudsman might be a Person or Organization depending on the context. We hypothesize that including context-sensitive training, which the baseline model lacks, can help mitigate this issue.

To test this hypothesis, we experimented on two training configurations. First, we trained a POS tagger together with our transition-based NER with shared weights. This process may help provide extra information to the transition-based parser so it can disambiguate between entities. Second, we finetuned context-sensitive vectors from RoBERTa Tagalog (Cruz and Cheng, 2022) for NER. Table 5 shows the relative error reduction between LOC and ORG entities. Given these results, we encourage researchers to utilize context-sensitive vectors such as RoBERTa Tagalog (or other BERT variants) when training models from this corpora.

<table><tr><td rowspan="2">Model</td><td colspan="2">Training dataset</td></tr><tr><td>WikiANN</td><td>TLUNIFIED-NER</td></tr><tr><td>Baseline (no additional embeddings)</td><td>19.92±0.03</td><td>30.24±0.02</td></tr><tr><td>fastText (Bojanowski et al., 2017)</td><td>24.41±0.01</td><td>45.09±0.02</td></tr><tr><td>RoBERTa Tagalog (Cruz and Cheng, 2022)</td><td>23.38±0.02</td><td>58.90±0.03</td></tr><tr><td>XLM-RoBERTa (Conneau et al., 2020)</td><td>31.28±0.01</td><td>57.67±0.01</td></tr><tr><td>Multilingual BERT (Devlin et al., 2019)</td><td>29.20±0.03</td><td>59.26±0.03</td></tr></table>

Table 6: Cross-dataset comparison between WikiANN (Pan et al., 2017) and TLUNIFIED-NER. We trained a model from WikiANN then applied it to TLUNIFIED-NER (and vice-versa). Reported results are F1-scores on the test set across three trials.

<table><tr><td>Entity label</td><td>F1-score</td></tr><tr><td>Person (PER)</td><td>67.95</td></tr><tr><td>Organization (ORG)</td><td>00.59</td></tr><tr><td>Location (LOC)</td><td>35.17</td></tr></table>

Table 7: Comparing the overlap between the original (silver-standard) WikiANN annotations against our reannotated version.

## 5.4 Comparison to WikiANN

The WikiANN dataset (Pan et al., 2017) is another resource for Tagalog NER. However, we found many annotation errors in the dataset, from misclassifications to fragmented sentences. We investigated how TLUNIFIED-NER fares against WikiANN’s silver-standard annotations.

We finetuned several models similar to Section 5.2 on the Tagalog portion of WikiANN’s training set and tested it on TLUNIFIED-NER’s test set (and vice-versa). In order to properly evaluate on WikiANN, we reannotated the test dataset using the same annotation guidelines described in Section 4.

Our results in Table 6 suggest that models built from the TLUNIFIED-NER corpus are more performant than with WikiANN. Additionally, the gap between WikiANN’s silver-standard annotations and our corrections is large, as shown in Table 7. We then posit that the gold-standard nature of TLUNIFIED-NER led to better performance than WikiANN, which predominantly consists of text fragments and low-quality annotations.

## 5.5 Experiments on large language models

Large language models (LLMs) have been shown to exhibit multilingual capabilities—incidental or not (Briakou et al., 2023). We investigated this property by performing a zero-shot prompting approach on TLUNIFIED-NER’s test set across a variety of commercial and open-source LLMs. Table 8 reports the F1-score across three trials.

<table><tr><td>Model</td><td>F1-score</td></tr><tr><td>GPT-4 (OpenAI, 2023)</td><td>65.89±0.44</td></tr><tr><td>GPT-3.5-turbo</td><td>53.05±0.42</td></tr><tr><td>Claude v1 (Anthropic, 2023)</td><td>58.88±0.03</td></tr><tr><td>Command (Cohere, 2023)</td><td>25.48±0.11</td></tr><tr><td>Dolly v2* (Conover et al., 2023)</td><td>13.07±0.14</td></tr><tr><td>Falcon* (Almazrouei et al., 2023)</td><td>8.65±0.04</td></tr><tr><td>StableLM v2* (Stability-AI, 2023)</td><td>0.25±0.03</td></tr><tr><td>OpenLLaMa* (Geng and Liu, 2023)</td><td>15.09±0.48</td></tr></table>

Table 8: Benchmark results on TLUNIFIED-NER across a variety of open-source and commercial LLMs. We used the 7B-parameter variants for models denoted with an asterisk (∗) due to budget constraints.

Our results suggest that supervised learning reliably outperforms zero-shot prompting for TLUNIFIED-NER given our prompt (see Appendix A.1). However, we acknowledge that these results are not a definitive comparison between two methods as prompt engineering is unstable with high variance (Webson and Pavlick, 2022; Zhao et al., 2021). In the future, we plan to explore different prompting techniques such as PromptNER (Ashok and Lipton, 2023) and chain-of-thought (Wei et al., 2023) to uncover the language models’ full capabilities.

## 6 Conclusion

In this paper, we introduced TLUNIFIED-NER, a Named Entity Recognition dataset for Tagalog. Unlike other Tagalog NER datasets, TLUNIFIED-NER is publicly-accessible and gold standard. Our iterative annotation process, together with our interannotator agreement, shows that the corpus is of high quality. In addition, our benchmarking results suggest that the task is learnable even with a simple baseline method. We hope that TLUNIFIED-NER fills the resource gap present in Tagalog NLP today. In the future, we plan to create a more finegrained (and perhaps, overlapping) NER tag set similar to the ACE project and expand on other major Philippine languages. Finally, the dataset is available online (https://huggingface.co/ datasets/ljvmiranda921/tlunified-ner) and we encourage researchers to improve upon our benchmark results.

## Limitations

The TLUNIFIED-NER corpora is comprised mostly by news reports. Although the texts demonstrate the standard usage of Tagalog, its domain is limited. In addition, we only trained a transitionbased parser model for our NER classifier. In the future, we plan to extend these benchmarks and include CRFs or other tools such as Stanford Stanza.

## Acknowledgements

We would like to express our gratitute to all those who contributed to the completion of this resource. We extend our appreciation to the anonymous reviewers for their constructive comments, which greatly improved the quality of this paper.

## References

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. Falcon-40B: an open large language model with state-of-the-art performance.  
Anthropic. 2023. Model card and evaluations for claude models.  
Dhananjay Ashok and Zachary C. Lipton. 2023. Promptner: Prompting for named entity recognition.  
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching Word Vectors with Subword Information. Transactions ofthe Associationfor Computational Linguistics, 5:135–146.  
Alex Brandsen, Suzan Verberne, Milco Wansleeben, and Karsten Lambers. 2020. Creating a Dataset for Named Entity Recognition in the Archaeology Domain. In Proceedings of the Twelfth Language  
Resources and Evaluation Conference, pages 4573– 4577, Marseille, France. European Language Resources Association.  
Eleftheria Briakou, Colin Cherry, and George Foster. 2023. Searching for needles in a haystack: On the role of incidental bilingualism in PaLM’s translation capability. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9432–9452, Toronto, Canada. Association for Computational Linguistics.  
Cohere. 2023. Command Model: The AI-Powered Solution for the Enterprise.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learning at Scale. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.  
Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world’s first truly open instructiontuned llm.  
Sherwyne Costiniano, Rose Ann Mae Santos, Julius Simon Mendoza, and Allen Jay Gale. 2022. Custom Coarse Grained Named Entity Recognition for Filipino Storytelling Data Using Uncased Transformer Models. SSRN Electronic Journal.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2022. Improving Large-scale Language Models and Resources for Filipino. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6548–6555, Marseille, France. European Language Resources Association.  
Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Ko Cheng. 2020. Exploiting News Article Structure for Automatic Corpus Generation of Entailment Datasets. In Pacific Rim International Conference on Artificial Intelligence.  
Louise Deleger, Qi Li, Todd Lingren, Megan Kaiser, Katalin Molnar, Laura Stoutenborough, Michal Kouril, Keith Marsolo, and Imre Solti. 2012. Building gold standard corpora for medical natural language processing tasks. In AMIA Annual Symposium Proceedings, pages 144–53.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages  
4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.  
George Doddington, Alexis Mitchell, Mark Przybocki, Lance Ramshaw, Stephanie Strassel, and Ralph Weischedel. 2004. The automatic content extraction (ACE) program – tasks, data, and evaluation. In Proceedings ofthe Fourth International Conference on Language Resources and Evaluation (LREC’04), Lisbon, Portugal. European Language Resources Association (ELRA).  
Xinyang Geng and Hao Liu. 2023. OpenLLaMA: An Open Reproduction of LLaMA.  
Ian Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. 2013. Maxout networks. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings ofMachine Learning Research, pages 1319–1327, Atlanta, Georgia, USA. PMLR.  
Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrial strength Natural Language Processing in Python.  
Paul R. Kroeger. 1992. Phrase Structure and Grammati cal Relations in Tagalog.  
Guillaume Lample, Miguel Ballesteros, Sandeep Sub ramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In Proceedings ofthe 2016 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 260–270, San Diego, California. Association for Computational Linguistics.  
Paul M. A. Lewis. 2009. Ethnologue: languages of the world. https://ethnologue.com/language/tgl. Accessed: June 2023  
OpenAI. 2023. GPT-4 Technical Report.  
Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Noth man, Kevin Knight, and Heng Ji. 2017. Cross-lingual name tagging and linking for 282 languages. In Proceedings ofthe 55th Annual Meeting ofthe Associa tionfor Computational Linguistics (Volume 1: Long Papers), pages 1946–1958, Vancouver, Canada. Association for Computational Linguistics.  
Sean Papay, Roman Klinger, and Sebastian Padó. 2020. Dissecting span identification tasks with performance prediction. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4881–4895, Online. Association for Computational Linguistics.  
Paul Schachter and Fe T. Otanes. 1973. Tagalog reference grammar. The Journal ofAsian Studies, 32:760 – 761.  
Stability-AI. 2023. StableLM-Alpha v2.  
Stephanie Strassel and Jennifer Tracey. 2016. LORELEI language packs: Data, tools, and resources for technology development in low resource languages. In Proceedings ofthe Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 3273–3280, Portorož, Slovenia. European Language Resources Association (ELRA).  
Pedro Ortiz Suarez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures. In 7th Workshop on the Challenges in the Management ofLarge Corpora.  
Erik F. Tjong Kim Sang. 2002. Introduction to the CoNLL-2002 shared task: Language-independent named entity recognition. In COLING-02: The 6th Conference on Natural Language Learning 2002 (CoNLL-2002).  
Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, pages 142– 147.  
Albert Webson and Ellie Pavlick. 2022. Do promptbased models really understand the meaning of their prompts? In Proceedings ofthe 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2300–2344, Seattle, United States. Association for Computational Linguistics.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.  
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12697–12706. PMLR.

## A Appendix

## A.1 Zero-shot prompt template

You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entitiesfor the set ofpredefined entity labels. From the Text input provided, extract named entitiesfor each label in thefollowingformat:

• PER: <comma delimited list ofstrings>  
• ORG: <comma delimited list ofstrings>  
• LOC: <comma delimited list ofstrings>

Below are definitions of each label to help aid you in what kinds ofnamed entities to extractfor each label. Assume these definitions are written by an expert andfollow them closely.

• PER: PERSON  
• ORG: ORGANIZATION  
• LOC: LOCATION OR GEOPOLITICAL EN-TITY

Text: {{ text }}

## A.2 Reproducibility

All the experiments and models in this paper are available publicly. Readers can head over to https://github.com/ljvmiranda921/ calamanCy/reports/aacl for all related code and assets. Note that the XLM-RoBERTa and multilingual BERT experiments may at least require a T4 or V100 GPU.