# Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning

 Philipp Seeberger    Korbinian Riedhammer Affiliation: Technische Hochschule Nürnberg Georg Simon Ohm Email: [{philipp.seeberger,korbinian.riedhammer}@th-nuernberg.de](mailto:) 

###### Abstract

Social media has become an important information source for crisis management and provides quick access to ongoing developments and critical information. However, classification models suffer from event-related biases and highly imbalanced label distributions which still poses a challenging task. To address these challenges, we propose a combination of entity-masked language modeling and hierarchical multi-label classification as a multi-task learning problem. We evaluate our method on tweets from the TREC-IS dataset and show an absolute performance gain w.r.t. F1-score of up to 10% for actionable information types. Moreover, we found that entity-masking reduces the effect of overfitting to in-domain events and enables improvements in cross-event generalization. Our source code is publicly available on GitHub.11 1 [https://github.com/th-nuernberg/crisis-tapt-hmc](https://github.com/th-nuernberg/crisis-tapt-hmc "")

 

## 1 Introduction

Messages on social media during disaster events have become an important information source in crisis management ([Reuter et al., 2018](#bib.bib31 "")). In contrast to traditional sources (e.g., official news), social media posts immediately provide details about developments, first-party observations, and affected people in an ongoing emergency situation ([Sakaki et al., 2010](#bib.bib32 "")). Having access to this information is crucial for developing situational awareness and supporting relief providers, government agencies, and other official institutions ([Kruspe et al., 2021](#bib.bib19 "")).

One key challenge poses the information refinement of high-volume social media streams which requires automatic methods for reliable detection of relevant content ([Kaufhold, 2021](#bib.bib16 "")). Most recent work has focused on binary, multi-class, and multi-label text classification techniques to classify posts into coarse (e.g., Relevant, Irrelevant) or fine-grained (e.g., InfrastructureDamage, MissingPeople) categories composed of flattened or hierarchical structures ([Alam et al., 2018b](#bib.bib2 ""); [Alam et al., 2021](#bib.bib3 ""); [Buntain et al., 2021](#bib.bib8 "")).

Another challenge in Natural Language Processing (NLP) is the nature of data prevalent in social media and microblogging platforms. For example, most works in the crisis-related domain focus on Twitter data ([Kruspe et al., 2021](#bib.bib19 "")) which inherits properties such as short texts (280 characters limitation per tweet), less contextual information, hashtags, and noise (e.g., misspellings, emojis) ([Wiegmann et al., 2020](#bib.bib39 ""); [Zahera et al., 2021](#bib.bib42 "")). According to [Sarmiento and Poblete (2021)](#bib.bib33 ""), different types of disasters (e.g., flood, wildfire) can be identified by only a few text-based features. However, event-related biases and entities as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") prevent models from generalizing to unseen disaster events and therefore degrade w.r.t. detection performance.

Figure 1: Example tweets of several disasters over time, annotated with entitites. The short posts are mostly biased towards specific events.

To circumvent this problem, approaches such as adversarial training ([Medina Maza et al., 2020](#bib.bib24 "")), domain adaptation ([Alam et al., 2018a](#bib.bib1 "")), and hierarchical label embeddings ([Miyazaki et al., 2019](#bib.bib28 "")) have been proposed but suffer from mixed event types, assume unlabeled data or require semantic label descriptions. Contrary to this work, we aim to enhance the detection of rare actionable information for unseen events by masking out entities, applying adaptive pre-training, and incorporating the hierarchical structure of labels.

#### Contributions

Our main contributions are as follows: (1) We introduce an adaptive pre-training strategy based on entity-masking. (2) We incorporate the hierarchical structure of labels as multi-task learning (MTL) problem. (3) We empirically show that our approach improves generalization to new events and increases detection performance for actionable information types.

## 2 Related Work

#### Crisis Tweet Classification

Besides conventional detection approaches such as filtering ([Kumar et al., 2011](#bib.bib20 "")) or crowdsourcing ([Poblet et al., 2014](#bib.bib30 "")), machine learning has received much attention in this area. Researchers experimented with several methods such as Naive Bayes, Support Vector Machines, and Decision Trees either with term-frequency features ([Habdank et al., 2017](#bib.bib14 "")) or static embeddings ([Kejriwal and Zhou, 2019](#bib.bib17 "")). More recently, the combination of Word2Vec [Mikolov et al. (2013)](#bib.bib27 "") with Convolutional and Recurrent Neural Networks achieved remarkable improvement in this field ([Kersten et al., 2019](#bib.bib18 ""); [Snyder et al., 2019](#bib.bib35 "")). Due to the success of Transformers ([Vaswani et al., 2017](#bib.bib36 "")) and the follow-up language models [Devlin et al. (2019)](#bib.bib10 ""), most works have been built upon this and outperformed previous approaches ([Alam et al., 2021](#bib.bib3 ""); [Wang et al., 2021](#bib.bib37 "")).

|        | Train  | Test |
| ------ | ------ | ---- |
|        |        |      |
| 30,389 | 16,059 |      |
| 32,105 | 10,709 |      |
| 1,458  | 389    |      |
| 683    | 144    |      |

Table 1: Overview of the dataset split; the values within the brackets of the upper classes corresponds to the number of unique low-level information types.

#### Adaptive Pre-Training

Transfer learning with language models essentially contributes to state-of-the-art results in a variety of NLP tasks ([Devlin et al., 2019](#bib.bib10 ""); [Liu et al., 2019](#bib.bib22 ""); [Clark et al., 2020](#bib.bib9 "")). Typically, such language models follow the three training steps ([Howard and Ruder, 2018](#bib.bib15 ""); [Ben-David et al., 2020](#bib.bib5 "")): (1) Pre-training on massive corpora; (2) Optional pre-training on task-specific data; (3) Supervised fine-tuning on target tasks. However, the second step is often neglected due to computational constraints whereby adaptive pre-training has shown to be effective ([Howard and Ruder, 2018](#bib.bib15 "")). Hence, [Gururangan et al. (2020)](#bib.bib13 "") introduced domain-adaptive pre-training (DAPT) and task-adaptive pre-training (TAPT) which cover continual pre-training on corpora tailored for a specific task. Moreover, strategies such as adding special tokens for tweets ([Nguyen et al., 2020](#bib.bib29 ""); [Wiegmann et al., 2020](#bib.bib39 "")) or additional masked language modeling (MLM) approaches ([Ben-David et al., 2020](#bib.bib5 "")) have been proven beneficial.

(a) Entity-masked language modeling

(b) Multi-task classification heads

Figure 2: Illustration of the concepts E-MLM with named entity recognition (NER) and MTL. FCMLM represents the prediction head for MLM. The classification heads will be placed on top of the pre-trained encoder. The building blocks Pooler, LT, LB and LG are fully connected layers and use the CLS token as sentence embedding.

#### Hierarchical Multi-Label Classification

Hierarchical multi-label classification (HMC) covers local and global approaches and the combination of both worlds ([Wehrmann et al., 2018](#bib.bib38 "")). A popular categorization of local methods is the subdivision into local classifier per parent node (LCPN) ([Dumais and Chen, 2000](#bib.bib11 "")), local classifier per node (LCN) ([Banerjee et al., 2019](#bib.bib4 "")), and local classifier per level (LCL) ([Wehrmann et al., 2018](#bib.bib38 "")). Hybrid approaches integrate the global part as a particular constraint such as hierarchical softmax ([Brinkmann and Bizer, 2021](#bib.bib7 "")) or combine multiple local and global prediction heads ([Wehrmann et al., 2018](#bib.bib38 "")). Recent work in information type classification introduced label embeddings which utilize the hierarchical structure ([Miyazaki et al., 2019](#bib.bib28 "")). Finally, the classification can also be viewed as MTL by combining certain loss functions ([Yu et al., 2021](#bib.bib41 ""); [Wang et al., 2021](#bib.bib37 "")).

## 3 TREC-IS

In this work, we mainly focus on the dataset of the shared-task TREC-IS, which represents a collection of annotated crisis-related tweets ([Buntain et al., 2021](#bib.bib8 "")). Each tweet belongs to a disaster event and is annotated with high-level information types which are derived from an ontology composed of hierarchical stages. However, information type labels are only shipped as a two-level hierarchy with four upper classes LT and 25 lower classes LB. Thus, both hierarchy levels represent a multi-label classification task. Following the TREC-IS track design, we split the dataset into train and test events which corresponds to the TREC-IS 2020B task. This split poses a challenging setup due to the requirement of cross-event generalization ([Wiegmann et al., 2020](#bib.bib39 "")). Table [1](#S2.T1 "Table 1 ‣ Crisis Tweet Classification ‣ 2 Related Work ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") gives an overview of each split; obviously, the information type distribution is highly imbalanced. For example, information types with low criticality such as MultimediaShare (31.7%) and News (25.4%) are prevalent. In contrast, the highly critical information types MovePeople (0.9%) and SearchAndRescue (0.4%) occur only rarely ([McCreadie et al., 2019](#bib.bib23 "")).22 2 We provide an overview of the labels with some example posts in Appendix [A](#A1 "Appendix A Overview of Information Types ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning").

## 4 Method

As depicted in Figure [2](#S2.F2 "Figure 2 ‣ Adaptive Pre-Training ‣ 2 Related Work ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") our approach combines the two concepts entity-masked language modeling (E-MLM) and MTL. In the following, we briefly describe our method as a combination of those two.

### 4.1 Entity-Masked Language Modeling

Based on adaptive pre-training, we extend on masked language modeling of a transformer encoder pre-trained on a large corpus such as BERT ([Devlin et al., 2019](#bib.bib10 "")). Here, the mitigation of event-related biases is facilitated by replacing entities – which are prone to be event-specific – with special tokens (see Figure [2(a)](#S2.F2.sf1 "In Figure 2 ‣ Adaptive Pre-Training ‣ 2 Related Work ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning")). This way we intend to capture disaster-related language patterns independently of the concrete entities. Following [Ben-David et al. (2020)](#bib.bib5 ""), we further introduce a masking probability α\\alpha tailored to entities in addition to the standard word masking with probability β\\beta. That is, with a typically higher probability α\\alpha we select random entity-tokens such as locations and lower probability β\\beta random standard subword-tokens. Finally, these selected tokens will be replaced by \[MASK\], random tokens or the unchanged tokens in order to learn the linguistic patterns related to those entities. For the rest of this paper, we rely on the pre-trained BERTBASE as the encoder model and the corresponding default MLM setup for pre-training (\[MASK\] with 80%, random tokens with 10%, and unchanged tokens with 10%).

### 4.2 Multi-Task Learning

The next step represents the fine-tuning of a classification head. We implement four basic hierarchical multi-label classification approaches as shown in Figure [2(b)](#S2.F2.sf2 "In Figure 2 ‣ Adaptive Pre-Training ‣ 2 Related Work ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning"). The LCL classification head jointly trains a flattened classification layer for each of the two hierarchy levels. In contrast, the LCPN model consists of a classification layer for each parent node. The hierarchical multi-label classification network (HMCN) is adapted from [Wehrmann et al. (2018)](#bib.bib38 "") and introduces a pooling layer on top of the preceding pooling layer. We experiment with a local and a global variant, whereas the global one additionally consists of a global classification layer. All pooling and classification layers are composed of a single feed-forward layer with tanh and sigmoid as activation functions, respectively. Finally, we minimize the binary cross-entropy ℒM​T​L\=λ​ℒLT+(1−λ)​ℒLB\\mathcal{L}\_{MTL}=\\lambda\\mathcal{L}\_{L\_{T}}+(1-\\lambda)\\mathcal{L}\_{L\_{B}} as a weighted loss function whereby ℒLT\\mathcal{L}\_{L\_{T}} represents the upper classes and ℒLB\\mathcal{L}\_{L\_{B}} the lower classes loss.

## 5 Experiments

### 5.1 Evaluation Metric

We follow the TREC-IS evaluation scheme: macro-averaged F1-score across information types for the two hierarchy levels in addition to the actionable information types (AIT) ([McCreadie et al., 2019](#bib.bib23 "")). The latter include rare information types with high priority consisting of: MovePeople, EmergingThreats, NewSubEvent, ServiceAvailable, GoodsServices, and SearchAndRescue.

### 5.2 Named Entity Recognition

As event-specific entities, we use the special tokens hashtag, url, person, location, organization, event, address, phone number, date, and number. All entities except the tokens hashtag and url are extracted with the Natural Language API of the Google Cloud Platform.33 3 We extracted the entities on 29 March 2022. We manually annotated 300 tweets and calculated a strict F1-score ([Segura-Bedmar et al., 2013](#bib.bib34 "")) of 0.692 which represents a reasonable good result for tweets.

### 5.3 Baseline and Hyper-Parameters

As baseline, we use TF-IDF with Logistic Regression (TF-IDF+LR) and BERTBASE with a single-task classification head. Furthermore, we apply the standard MLM of BERT in contrast to E-MLM in order to validate the effect of masking entities. Lastly, we train the MTL model (MTLprio) from [Wang et al. (2021)](#bib.bib37 "") which combines lower classes as classification and priority scores as regression task. We choose the best hyper-parameters for each model based on a stratified split with a ratio of 90% for train and 10% for development data, respectively. In terms of hyper-parameters, we set α\=0.5\\alpha=0.5 and β\=0.1\\beta=0.1 for E-MLM; other parameters were set according to other work, including learning rate of 5​e−55e-5, batch size of 32, and λ\=0.1\\lambda=0.1 for fine-tuning. The detailed hyper-parameter selection process is shown in Appendix [B](#A2 "Appendix B Hyper-Parameters ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning").

| Model | LT    | LB    | AIT |
| ----- | ----- | ----- | --- |
|       |       |       |     |
| 0.657 | 0.499 | 0.462 |     |
| 0.717 | 0.531 | 0.513 |     |
| 0.714 | 0.551 | 0.546 |     |
| 0.701 | 0.481 | 0.444 |     |

Table 2: Overall results on the development set.

| Model | LT    | LB    | AIT |
| ----- | ----- | ----- | --- |
|       |       |       |     |
| 0.460 | 0.201 | 0.168 |     |
| 0.553 | 0.269 | 0.236 |     |
| 0.524 | 0.245 | 0.229 |     |
| 0.553 | 0.307 | 0.306 |     |
|       |       |       |     |
| 0.548 | 0.314 | 0.309 |     |
| 0.548 | 0.305 | 0.307 |     |
| 0.546 | 0.310 | 0.320 |     |
| 0.558 | 0.312 | 0.335 |     |

Table 3: Overall results of information type classification; bold and underlined values indicate the best and second-best results, respectively. ∗We fine-tuned the approach of [Wang et al. (2021)](#bib.bib37 "") with BERTBASE and without ensembling.

Figure 3: Absolute performance differences w.r.t. F1-score between the single-task and HMCNlocal model.

### 5.4 Results

In the following, we report the performance for the upper classes LT, lower classes LB, and AIT. However, for our evaluation we do not focus on LT since the experiments did not show large differences across all BERT models. The MTL models are only reported with BERTE-MLM.

#### E-MLM

Table [3](#S5.T3 "Table 3 ‣ 5.3 Baseline and Hyper-Parameters ‣ 5 Experiments ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") displays the results of all single-task and MTL runs. For E-MLM, we observe an absolute performance gain w.r.t. F1-score for both LB and AIT by up to 4% and 7%, respectively. To validate the event-generalization effect, we additionally analyzed the development set, as a proxy to estimate the in-domain event performance as shown in Table [2](#S5.T2 "Table 2 ‣ 5.3 Baseline and Hyper-Parameters ‣ 5 Experiments ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning"). Contrary to the test set, standard MLM increases the absolute LB performance by 2% whereas the E-MLM approach drops by 5% which is a confirmation of our assumption about event-related overfitting.

#### Multi-Task Learning

In terms of MTL, the HMCNlocal model achieved the best results for AIT. Overall the MTL classification outperforms the single-task models for actionable categories and in addition the LB classes except for LCPN. We assume that the LT classification objective implicitly clusters the internal representation w.r.t. the high-level information types and therefore mitigates overfitting towards the major classes. As depicted in Figure [3](#S5.F3 "Figure 3 ‣ 5.3 Baseline and Hyper-Parameters ‣ 5 Experiments ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning"), the HMCNlocal model improves the detection of rare actionable information types over the single-task model while at the same time decreasing the performance on the category with the most information types. This can be caused by the ambiguous label definitions and semantic similarities with other information types ([Mehrotra et al., 2022](#bib.bib26 "")).

Figure 4: Comparison across event types w.r.t. F1-score between the BERTBASE and HMCNlocal model. We plot the mean and standard deviation for multiple events within a event type.

### 5.5 Analysis of Events

In Figure [4](#S5.F4 "Figure 4 ‣ Multi-Task Learning ‣ 5.4 Results ‣ 5 Experiments ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") we illustrate the model performance for LB across different event types. For multiple events, we report the mean and standard deviation, respectively. We observe an increase in performance for the event types covid, shooting, typhoon, storm, tornado, and flood and a small decrease for the event types fire, hostage, and explosion. As shown by the variance for multiple events, the performance highly differs across specific events. Surprisingly, the event type covid achieved the worst performance for both models despite the existence of three covid events within the train data. These results indicate that even regional differences about the same global event predominantly affect the generalization performance across events.

### 5.6 Ablation Study

As ablation study we removed several proposed components to assess the performance impact of our model. Thereby, the component entities represents the additional special tokens and replacement within the input text. As shown in Table [4](#S5.T4 "Table 4 ‣ 5.6 Ablation Study ‣ 5 Experiments ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning"), we started with the HMCNlocal model and demonstrate that entities, MLM and MTL contribute to an increase w.r.t. F1-score for both LB and AIT. The results indicate that the variant which removes the hierarchical component only degrades the performance for the low-resource actionable information types. Removing the E-MLM mechanism degrades the model’s performance most in our experiments.

| Method | LT    | LB    | AIT |
| ------ | ----- | ----- | --- |
| 0.548  | 0.314 | 0.309 |     |
| 0.553  | 0.307 | 0.306 |     |
| 0.529  | 0.276 | 0.242 |     |
| 0.553  | 0.269 | 0.236 |     |

Table 4: Overall results of the ablation study.

## 6 Conclusion and Future Work

In this work, we identified shortcomings in the field of crisis tweet classification for unseen events. For the TREC-IS data, we found contrasting effects in terms of pre-training and observed an absolute improvement of up to 3% w.r.t. F1-score for actionable information types by incorporating the hierarchical structure. Furthermore, we confirmed the effectiveness of our method based on the shared-task TREC-IS. Future work includes pre-training on a larger corpus, the mitigation of the trade-off between major and minor classes performances, and to analyse the influence of label semantics.

## Ethical and Societal Implications

Open Source Intelligence (OSINT) has become a significant role for various authorities and NGOs for advancing struggles in global health, human rights, and crisis management ([Bernard et al., 2018](#bib.bib6 ""); [Evangelista et al., 2021](#bib.bib12 ""); [Kaufhold, 2021](#bib.bib16 "")). Following the view of OSINT as a tool, our work pursues the goal to support relief providers, government agencies, and other disaster-response stakeholders during ongoing and evolving crisis events.

We argue that NLP for disaster response can have a positive impact on comprehensive situational awareness and in decision-making processes such as coordination of particular services or physical goods. In the context of this work, positive impact means to supplement traditional information sources with social media streams that enable faster access to ongoing developments, first-party observations, and more fine-grained information content. For example, NLP for social media can enrich the information with the public as co-producers which may reveal critical subevents like missed or trapped people ([Li et al., 2018](#bib.bib21 "")). Retrieving this kind of information could positively affect disaster management strategies and relief efforts during natural and human-made disasters.

In contrast, relying on social media as an information source runs the risk of introducing mis- and disinformation. This can cause adverse effects on relief efforts and requires tailored strategies and particular care before the deployment of such models. Furthermore, data privacy issues may arise due to the inherited properties of social media data. Various anonymization processes should be taken into account for identifying and neutralizing sensitive references ([Medlock, 2006](#bib.bib25 "")). In this work, the use of entity tokens as categorization can be seen as one kind of anonymization procedure. However, model training with such entities could be task-specific and prone to error propagation by named entity recognition systems.

## Acknowledgments

The authors acknowledge the financial support by the Federal Ministry of Education and Research of Germany in the project ISAKI (project number 13N15572). We also would like to thank the anonymous reviewers for their constructive feedback.

## References

*   Alam et al. (2018a) Firoj Alam, Shafiq Joty, and Muhammad Imran. 2018a. [Domain adaptation with adversarial training and graph embeddings](https://doi.org/10.18653/v1/P18-1099 ""). In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1077–1087, Melbourne, Australia. Association for Computational Linguistics.
*   Alam et al. (2018b) Firoj Alam, Ferda Ofli, and Muhammad Imran. 2018b. Crisismmd: Multimodal twitter datasets from natural disasters. In *Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)*.
*   Alam et al. (2021) Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli. 2021. Crisisbench: Benchmarking crisis-related social media datasets for humanitarian information processing. In *15th International Conference on Web and Social Media (ICWSM)*.
*   Banerjee et al. (2019) Siddhartha Banerjee, Cem Akkaya, Francisco Perez-Sorrosal, and Kostas Tsioutsiouliklis. 2019. [Hierarchical transfer learning for multi-label text classification](https://doi.org/10.18653/v1/P19-1633 ""). In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6295–6300, Florence, Italy. Association for Computational Linguistics.
*   Ben-David et al. (2020) Eyal Ben-David, Carmel Rabinovitz, and Roi Reichart. 2020. [PERL: Pivot-based domain adaptation for pre-trained deep contextualized embedding models](https://doi.org/10.1162/tacl_a_00328 ""). *Transactions of the Association for Computational Linguistics*, 8:504–521.
*   Bernard et al. (2018) Rose Bernard, G. Bowsher, C. Milner, P. Boyle, P. Patel, and R. Sullivan. 2018. [Intelligence and global health: assessing the role of open source and social media intelligence analysis in infectious disease outbreaks](https://doi.org/10.1007/s10389-018-0899-3 ""). *Journal of Public Health*, 26(5):509–514.
*   Brinkmann and Bizer (2021) Alexander Brinkmann and Christian Bizer. 2021. [Improving hierarchical product classification using domain-specific language modelling](https://madoc.bib.uni-mannheim.de/60102/ ""). *Bulletin of the Technical Committee on Data Engineering / IEEE Computer Society*, 44(2):14–25.
*   Buntain et al. (2021) Cody L. Buntain, Richard McCreadie, and Ian Soboroff. 2021. Incident Streams 2020: TREC-IS in the Time of COVID-19. In *ISCRAM 2021: 18th International Conference on Information Systems for Crisis Response and Management*.
*   Clark et al. (2020) Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://openreview.net/pdf?id=r1xMH1BtvB ""). In *ICLR*.
*   Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423 ""). In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, Minneapolis, Minnesota. Association for Computational Linguistics.
*   Dumais and Chen (2000) Susan Dumais and Hao Chen. 2000. [Hierarchical classification of web content](https://doi.org/10.1145/345508.345593 ""). In *Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR ’00, page 256–263, New York, NY, USA. Association for Computing Machinery.
*   Evangelista et al. (2021) João Rafael Gonçalves Evangelista, Renato José Sassi, Márcio Romero, and Domingos Napolitano. 2021. [Systematic Literature Review to Investigate the Application of Open Source Intelligence (OSINT) with Artificial Intelligence](https://doi.org/10.1080/19361610.2020.1761737 ""). *Journal of Applied Security Research*, 16(3):345–369. Publisher: Routledge \_eprint: https://doi.org/10.1080/19361610.2020.1761737.
*   Gururangan et al. (2020) Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. [Don’t stop pretraining: Adapt language models to domains and tasks](https://doi.org/10.18653/v1/2020.acl-main.740 ""). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8342–8360, Online. Association for Computational Linguistics.
*   Habdank et al. (2017) Matthias Habdank, Nikolai Rodehutskors, and Rainer Koch. 2017. [Relevancy assessment of tweets using supervised learning techniques: Mining emergency related tweets for automated relevancy classification](https://doi.org/10.1109/ICT-DM.2017.8275670 ""). In *2017 4th International Conference on Information and Communication Technologies for Disaster Management (ICT-DM)*, pages 1–8.
*   Howard and Ruder (2018) Jeremy Howard and Sebastian Ruder. 2018. [Universal language model fine-tuning for text classification](https://doi.org/10.18653/v1/P18-1031 ""). In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 328–339, Melbourne, Australia. Association for Computational Linguistics.
*   Kaufhold (2021) Marc-André Kaufhold. 2021. [*Information Refinement Technologies for Crisis Informatics: User Expectations and Design Principles for Social Media and Mobile Apps*](https://doi.org/10.1007/978-3-658-33341-6 ""). Springer Fachmedien Wiesbaden, Wiesbaden.
*   Kejriwal and Zhou (2019) M. Kejriwal and P. Zhou. 2019. [Low-supervision urgency detection and transfer in short crisis messages](https://doi.org/10.1145/3341161.3342936 ""). In *2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)*, pages 353–356, Los Alamitos, CA, USA. IEEE Computer Society.
*   Kersten et al. (2019) Jens Kersten, Anna Kruspe, Matti Wiegmann, and Friederike Klan. 2019. Robust filtering of crisis-related tweets. In *ISCRAM 2019: 16th International Conference on Information Systems for Crisis Response and Management*.
*   Kruspe et al. (2021) Anna Kruspe, Jens Kersten, and Friederike Klan. 2021. [Review article: Detection of actionable tweets in crisis events](https://doi.org/10.5194/nhess-21-1825-2021 ""). *Natural Hazards and Earth System Sciences*, 21(6):1825–1845.
*   Kumar et al. (2011) Shamanth Kumar, Geoffrey Barbier, Mohammad Abbasi, and Huan Liu. 2011. [Tweettracker: An analysis tool for humanitarian and disaster relief](https://ojs.aaai.org/index.php/ICWSM/article/view/14079 ""). *Proceedings of the International AAAI Conference on Web and Social Media*, 5(1):661–662.
*   Li et al. (2018) Lifang Li, Qingpeng Zhang, Jun Tian, and Haolin Wang. 2018. [Characterizing information propagation patterns in emergencies: A case study with Yiliang Earthquake](https://doi.org/10.1016/j.ijinfomgt.2017.08.008 ""). *International Journal of Information Management*, 38(1):34–41.
*   Liu et al. (2019) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. [Roberta: A robustly optimized bert pretraining approach](https://doi.org/10.48550/ARXIV.1907.11692 "").
*   McCreadie et al. (2019) Richard McCreadie, Cody L. Buntain, and Ian Soboroff. 2019. TREC Incident Streams: Finding Actionable Information on Social Media. In *ISCRAM 2019: 16th International Conference on Information Systems for Crisis Response and Management*.
*   Medina Maza et al. (2020) Salvador Medina Maza, Evangelia Spiliopoulou, Eduard Hovy, and Alexander Hauptmann. 2020. [Event-related bias removal for real-time disaster events](https://doi.org/10.18653/v1/2020.findings-emnlp.344 ""). In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3858–3868, Online. Association for Computational Linguistics.
*   Medlock (2006) Ben Medlock. 2006. [An introduction to NLP-based textual anonymisation](http://www.lrec-conf.org/proceedings/lrec2006/pdf/200_pdf.pdf ""). In *Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC’06)*, Genoa, Italy. European Language Resources Association (ELRA).
*   Mehrotra et al. (2022) Harshit Mehrotra, Akanksha Mishra, and Sukomal Pal. 2022. [A Multi-stage Classification Framework for Disaster-Specific Tweets](https://doi.org/10.1007/s42979-021-00930-z ""). *SN Computer Science*, 3(1):24.
*   Mikolov et al. (2013) Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. [Distributed representations of words and phrases and their compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf ""). In *Advances in Neural Information Processing Systems*, volume 26. Curran Associates, Inc.
*   Miyazaki et al. (2019) Taro Miyazaki, Kiminobu Makino, Yuka Takei, Hiroki Okamoto, and Jun Goto. 2019. [Label embedding using hierarchical structure of labels for Twitter classification](https://doi.org/10.18653/v1/D19-1660 ""). In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 6317–6322, Hong Kong, China. Association for Computational Linguistics.
*   Nguyen et al. (2020) Dat Quoc Nguyen, Thanh Vu, and Anh Tuan Nguyen. 2020. [BERTweet: A pre-trained language model for English tweets](https://doi.org/10.18653/v1/2020.emnlp-demos.2 ""). In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 9–14, Online. Association for Computational Linguistics.
*   Poblet et al. (2014) Marta Poblet, Esteban García-Cuesta, and Pompeu Casanovas. 2014. Crowdsourcing tools for disaster management: A review of platforms and methods. In *AI Approaches to the Complexity of Legal Systems*, pages 261–274, Berlin, Heidelberg. Springer Berlin Heidelberg.
*   Reuter et al. (2018) Christian Reuter, Amanda Lee Hughes, and Marc-André Kaufhold. 2018. [Social Media in Crisis Management: An Evaluation and Analysis of Crisis Informatics Research](https://doi.org/10.1080/10447318.2018.1427832 ""). *International Journal of Human–Computer Interaction*, 34(4):280–294.
*   Sakaki et al. (2010) Takeshi Sakaki, Makoto Okazaki, and Yutaka Matsuo. 2010. [Earthquake shakes twitter users: Real-time event detection by social sensors](https://doi.org/10.1145/1772690.1772777 ""). In *Proceedings of the 19th International Conference on World Wide Web*, WWW ’10, page 851–860, New York, NY, USA. Association for Computing Machinery.
*   Sarmiento and Poblete (2021) Hernan Sarmiento and Barbara Poblete. 2021. [Crisis communication: A comparative study of communication patterns across crisis events in social media](https://doi.org/10.1145/3412841.3442044 ""). In *Proceedings of the 36th Annual ACM Symposium on Applied Computing*, SAC ’21, page 1711–1720, New York, NY, USA. Association for Computing Machinery.
*   Segura-Bedmar et al. (2013) Isabel Segura-Bedmar, Paloma Martínez, and María Herrero-Zazo. 2013. [SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)](https://aclanthology.org/S13-2056 ""). In *Second Joint Conference on Lexical and Computational Semantics (\*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013)*, pages 341–350, Atlanta, Georgia, USA. Association for Computational Linguistics.
*   Snyder et al. (2019) Luke S. Snyder, Yi-Shan Lin, Morteza Karimzadeh, Dan Goldwasser, and David S. Ebert. 2019. [Interactive Learning for Identifying Relevant Tweets to Support Real-time Situational Awareness](https://doi.org/10.1109/TVCG.2019.2934614 ""). *IEEE Transactions on Visualization and Computer Graphics*, pages 1–1.
*   Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf ""). In *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc.
*   Wang et al. (2021) Congcong Wang, Paul Nulty, and David Lillis. 2021. Transformer-based Multi-task Learning for Disaster Tweet Categorisation. In *ISCRAM 2021: 18th International Conference on Information Systems for Crisis Response and Management*.
*   Wehrmann et al. (2018) Jonatas Wehrmann, Ricardo Cerri, and Rodrigo Barros. 2018. [Hierarchical multi-label classification networks](https://proceedings.mlr.press/v80/wehrmann18a.html ""). In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 5075–5084. PMLR.
*   Wiegmann et al. (2020) Matti Wiegmann, Jens Kersten, Friederike Klan, Martin Potthast, and Benno Stein. 2020. [Analysis of Detection Models for Disaster-Related Tweets](https://doi.org/10.5281/ZENODO.3713920 ""). In *ISCRAM 2020: 17th International Conference on Information Systems for Crisis Response and Management*.
*   Wolf et al. (2020) Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. [Transformers: State-of-the-art natural language processing](https://doi.org/10.18653/v1/2020.emnlp-demos.6 ""). In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics.
*   Yu et al. (2021) Yipeng Yu, Zixun Sun, Chi Sun, and Wenqiang Liu. 2021. [Hierarchical multilabel text classification via multitask learning](https://doi.org/10.1109/ICTAI52525.2021.00180 ""). In *2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)*, pages 1138–1143.
*   Zahera et al. (2021) Hamada M. Zahera, Rricha Jalota, Mohamed Ahmed Sherif, and Axel-Cyrille Ngonga Ngomo. 2021. [I-aid: Identifying actionable information from disaster-related tweets](https://doi.org/10.1109/ACCESS.2021.3107812 ""). *IEEE Access*, 9:118861–118870.

## Appendix A Overview of Information Types

We list all information types of the TREC-IS dataset in Table [5](#A2.T5 "Table 5 ‣ Appendix B Hyper-Parameters ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning"). The value in the last column indicates the number of Twitter posts to which the corresponding labels were assigned. Table [6](#A2.T6 "Table 6 ‣ Appendix B Hyper-Parameters ‣ Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning") displays example tweets for various events with the corresponding labels from the TREC-IS dataset.

## Appendix B Hyper-Parameters

The search space for TF-IDF+LR included ngram-range, max features and regularization strength. In terms of BERT fine-tuning, we manually experimented with the same parameters as in [Wang et al. (2021)](#bib.bib37 "") and selected in line with this work the learning rate 5​e−55e-5 and batch size 3232. Due to computational constraints, we used for BERT pre-training the TAPT parameters of [Gururangan et al. (2020)](#bib.bib13 ""). Similar to [Ben-David et al. (2020)](#bib.bib5 ""), we experimented with the MLM probabilities α∈{0.1,0.3,0.5,0.8}\\alpha\\in\\{0.1,0.3,0.5,0.8\\} and β∈{0.1,0.3,0.5,0.8}\\beta\\in\\{0.1,0.3,0.5,0.8\\} and found the setup α\=0.5\\alpha=0.5 and β\=0.1\\beta=0.1 to perform best. This is in line with [Ben-David et al. (2020)](#bib.bib5 "") which empirically show good results. For MTL we tuned λ∈{0.1,0.5,0.9}\\lambda\\in\\{0.1,0.5,0.9\\} and finally set λ\=0.1\\lambda=0.1. We trained all transformer models with the Transformers library ([Wolf et al., 2020](#bib.bib40 "")) and AdamW for up to 50 (pre-training) and 15 (fine-tuning) epochs, evaluated the performance each 1000 steps on the development set and selected the best performing checkpoint. If not other mentioned, we used for the rest of the hyper-parameters the default setup of BERTBASE from the Transformers library.

| Id     | Upper Class (LT) | Lower Class (LB)      | Actionable (AIT) | \# tweets |
| ------ | ---------------- | --------------------- | ---------------- | --------- |
| RQ 01  | Request          | GoodsServices         | ✓                | 194       |
| RQ 02  | Request          | InformationWanted     |                  | 395       |
| RQ 03  | Request          | SearchAndRescue       | ✓                | 274       |
| CTA 01 | CallToAction     | Donations             |                  | 986       |
| CTA 02 | CallToAction     | MovePeople            | ✓                | 679       |
| CTA 03 | CallToAction     | Volunteer             |                  | 242       |
| O 01   | Other            | Advice                |                  | 3,277     |
| O 02   | Other            | ContextualInformation |                  | 4,583     |
| O 03   | Other            | Discussion            |                  | 5,303     |
| O 04   | Other            | Irrelevant            |                  | 23,053    |
| O 05   | Other            | Sentiment             |                  | 11,101    |
| RP 01  | Report           | CleanUp               |                  | 493       |
| RP 02  | Report           | EmergingThreats       | ✓                | 6,930     |
| RP 03  | Report           | Factoid               |                  | 10,224    |
| RP 04  | Report           | NewSubEvent           | ✓                | 2,806     |
| RP 05  | Report           | FirstPartyObservation |                  | 5,290     |
| RP 06  | Report           | Hashtags              |                  | 15,787    |
| RP 07  | Report           | Location              |                  | 23,676    |
| RP 08  | Report           | MultimediaShare       |                  | 22,976    |
| RP 09  | Report           | News                  |                  | 18,374    |
| RP 10  | Report           | Official              |                  | 2,836     |
| RP 11  | Report           | OriginalEvent         |                  | 4,148     |
| RP 12  | Report           | ServiceAvailable      | ✓                | 2,184     |
| RP 13  | Report           | ThirdPartyObservation |                  | 17,223    |
| RP 14  | Report           | Weather               |                  | 7,655     |

Table 5: Information types and hierarchical structure of labels.

| Event                                                   | Labels                                                                                                                          | Tweet |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Irrelevant                                              | From the train, showing the smoke filled sky from the #Lithgow #nswfires                                                        |       |
| ThirdPartyObservation, Factoid, Advice                  | FIRE UPDATE: Families told to be ready to run as a massive 300km wall of fire sweeps through Blue Mtns. #nswfires               |       |
| News                                                    | New this morning: At least 6 people are dead after the massive M8.2 quake in #Chile                                             |       |
| Location, Factoid, OriginalEvent, ContextualInformation | At least 25 dead and more than 2,500 injured as a result of the Beirut Port explosion according to the Lebanese Health Ministry |       |
| Factoid                                                 | 5 people confirmed dead in Colorado flooding, and 1,254 people unaccounted for statewide, official says                         |       |
| Weather, Location, Hashtags                             | We have 2.5 inches here 2.6 miles north-west of Downtown awake Forest. #FlorenceHurricane2018                                   |       |

Table 6: Example tweets and labels for different events.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")