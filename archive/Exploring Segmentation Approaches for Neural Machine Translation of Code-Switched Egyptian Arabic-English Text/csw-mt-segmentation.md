# <span id="page-0-0"></span>Exploring Segmentation Approaches for Neural Machine Translation of Code-Switched Egyptian Arabic-English Text

Marwa Gaser,<sup>1</sup> Manuel Mager,2[∗](#page-0-0) Injy Hamed,3,<sup>4</sup>

Nizar Habash,<sup>4</sup> Slim Abdennadher,<sup>1</sup> Ngoc Thang Vu<sup>3</sup>

<sup>1</sup>The German University in Cairo, <sup>2</sup>AWS AI Labs

<sup>3</sup>University of Stuttgart, <sup>4</sup>New York University Abu Dhabi

{marwa.saleh,slim.abdennadher}@guc.edu.eg, magerlm@amazon.com,

{injy.hamed,nizar.habash}@nyu.edu, thang.vu@ims.uni-stuttgart.de

# Abstract

Data sparsity is one of the main challenges posed by code-switching (CS), which is further exacerbated in the case of morphologically rich languages. For the task of machine translation (MT), morphological segmentation has proven successful in alleviating data sparsity in monolingual contexts; however, it has not been investigated for CS settings. In this paper, we study the effectiveness of different segmentation approaches on MT performance, covering morphology-based and frequency-based segmentation techniques. We experiment on MT from code-switched Arabic-English to English. We provide detailed analysis, examining a variety of conditions, such as data size and sentences with different degrees of CS. Empirical results show that morphology-aware segmenters perform the best in segmentation tasks but under-perform in MT. Nevertheless, we find that the choice of the segmentation setup to use for MT is highly dependent on the data size. For extreme low-resource scenarios, a combination of frequency and morphology-based segmentations is shown to perform the best. For more resourced settings, such a combination does not bring significant improvements over the use of frequency-based segmentation.

# 1 Introduction

Code-switching (CS), i.e. the alternation of language in text or speech, has been gaining worldwide popularity, due to several reasons, including globalization and immigration. While this has been met with a growing interest in the NLP field to build systems that can handle such mixed input, work on CS machine translation (MT) is still considered in its infancy, where only a few language pairs have been investigated [\(Sinha and Thakur,](#page-11-0) [2005;](#page-11-0) [Dhar et al.,](#page-9-0) [2018;](#page-9-0) [Menacer et al.,](#page-11-1) [2019;](#page-11-1) [Xu](#page-12-0) [and Yvon,](#page-12-0) [2021;](#page-12-0) [Hamed et al.,](#page-10-0) [2022c\)](#page-10-0).

<span id="page-0-1"></span>

| situation بصراحة بالنسبالي ع ال depends it CS Sentence |                                                                             |
|--------------------------------------------------------|-----------------------------------------------------------------------------|
| بصراحة                                                 | ب#صراحة                                                                     |
| بالنسبالي                                              | ب#النسبا#ل#ي                                                                |
| ع                                                      | ع                                                                           |
| it depends Translation                                 | situation bSrAHp bAlnsbAly E Al for me it honestly depends on the situation |
| it                                                     | it it                                                                       |
| depends depends                                        | depend#s                                                                    |
| situation situation                                    | situation                                                                   |
| bSrAHp                                                 | b#SrAHp                                                                     |
| bAlnsbAly                                              | b#AlnsbA#ly                                                                 |
| E                                                      | E                                                                           |
| Al                                                     | Al                                                                          |
| Word Translation ال ال                                 | Segmentation honestly for me on the                                         |

Figure 1: An example sentence with code-switching (CS) between English and Egyptian Arabic. The words are contrasted with their segmentations and English translations. Arabic words are paired with their transliterations in the Buckwalter scheme [\(Habash et al.,](#page-10-1) [2007\)](#page-10-1).

In this work, we focus on the CS Egyptian Arabic (EGY)-English (EN) language pair, as we observe its usage is becoming more common. Besides being prevalent amongst Egyptian migrant communities, it is also commonly used in Egypt due to the increase in international schooling systems and educational advancements. We identify three main challenges for CS MT. First is data sparsity, a challenge common to many CS language pairs because of limited parallel corpora containing commissioned translations of CS text [\(Çetinoglu et al.](#page-9-1) ˘ , [2016;](#page-9-1) [Srivastava and Singh,](#page-11-2) [2020;](#page-11-2) [Tarunesh et al.,](#page-11-3) [2021;](#page-11-3) [Hamed et al.,](#page-10-2) [2022b;](#page-10-2) [Chen et al.,](#page-9-2) [2022\)](#page-9-2). Second is Egyptian Arabic morphological richness, which further exacerbates the data sparsity situation [\(Habash et al.,](#page-10-3) [2012a](#page-10-3)[,b\)](#page-10-4). Third, since the matrix language (EGY) is morphologically rich, CS occurs at three CS levels: on the boundaries of sentences (inter-sentential CS), between words (intrasentential CS), and within words, i.e., morphological code-switching (MCS). This mix of types of CS raises the question of how to handle them all in the same system. These challenges are further illustrated in Figure [1.](#page-0-1)

<sup>∗</sup>Work done while at the University of Stuttgart.

A common solution to handle data sparsity for MT of morphologically rich languages is morphological segmentation [\(Oudah et al.,](#page-11-4) [2019;](#page-11-4) [Ata](#page-9-3)[man et al.,](#page-9-3) [2017;](#page-9-3) [Grönroos et al.,](#page-10-5) [2020\)](#page-10-5). However, this has not been investigated for CS. In this paper, we explore a wide range of segmentation approaches, covering unsupervised morphologybased segmenters, unsupervised frequency-based segmenters, and supervised morphology-based segmenters. This work aims to answer the following research questions (RQs):

- RQ1: Which segmentation setup performs the best in the downstream MT task across different training sizes?
- RQ2: Does the effectiveness of the different segmenters in the MT task differ according to the CS type of the source sentence?
- RQ3: Is there a correlation between a more morphologically correct segmentation and MT performance?

While our results show that there is no correlation between correct morphological segmentation and MT performance, we find that the performance ranking between the MT systems varies across different training data sizes and sentence types (monolingual vs. code-switched). We show that applying a combination of supervised morphology-based and unsupervised frequency-based segmentations consistently gives best results, with statistical significance under low data sizes. While commonwisdom suggests that Byte-Pair Encoding (BPE) is the best approach, our experiments highlight the importance of integrating morphological knowledge in the case of extreme low-resource settings. We believe that the insights and methodology we follow will be useful to researchers working with low-resource languages. An additional contribution of our research is the creation of a gold standard morphologically annotated CS Egyptian Arabic-English dataset which we make publicly available.[<sup>1</sup>](#page-1-0)

The paper is organized as follows. Section [2](#page-1-1) discusses related work. Section [3](#page-2-0) describes the dataset we annotated. Section [4](#page-2-1) describes and evaluates the different segmenters used. Section [5](#page-5-0) describes and evaluates the various MT systems. In Section [6,](#page-7-0) we answer our research questions.

# <span id="page-1-1"></span>2 Related Work

Several researchers have investigated the effect of applying different morphological and agnostic segmentation approaches on the MT performance for monolingual languages. [Roest et al.](#page-11-5) [\(2020\)](#page-11-5); [Saleva and Lignos](#page-11-6) [\(2021\)](#page-11-6) show that unsupervised morphology-based segmentation like Linguistically Motivated Vocabulary Reduction (LMVR) [\(Ata](#page-9-3)[man et al.,](#page-9-3) [2017\)](#page-9-3), Morfessor [\(Smit et al.,](#page-11-7) [2014\)](#page-11-7), and FlatCat [\(Grönroos et al.,](#page-10-6) [2014\)](#page-10-6) for Nepali–English, Sinhala–English, Kazakh–English, and Inuktitut– English language pairs show either no improvement or no significant improvement over the agnostic BPE segmentation [\(Sennrich et al.,](#page-11-8) [2016\)](#page-11-8) in translation tasks. Meanwhile, [Mager et al.](#page-10-7) [\(2022\)](#page-10-7) and [Ataman et al.](#page-9-3) [\(2017\)](#page-9-3) show that for polysynthetic and highly agglutinative languages, unsupervised morphology-based segmentation outperforms BPEs [\(Sennrich et al.,](#page-11-8) [2016\)](#page-11-8) in MT tasks in both directions. Nevertheless, applying BPEs on top of morphology-based segmentation for Turkish– English, Uyghur–Chinese, and Arabic–English has shown to bring improvements over solely using BPEs or morphology-based segmentation for neural MT task [\(Pan et al.,](#page-11-9) [2020;](#page-11-9) [Tawfik et al.,](#page-12-1) [2019\)](#page-12-1). A similar result was achieved by [\(Ortega et al.,](#page-11-10) [2020\)](#page-11-10), using a morphological guided BPE for polysynthetic languages. However, [Oudah et al.](#page-11-4) [\(2019\)](#page-11-4) show that such an approach is beneficial in the case of statistical machine translation (SMT), and does not improve results for neural machine translation (NMT). For other natural language processing (NLP) tasks, [Al-Thubaity and Al-Subaie](#page-9-4) [\(2015\)](#page-9-4) show that utilizing word segmented Arabic dataset leads to improvements in text classification task over utilizing unsegmented dataset in terms of accuracy, precision, recall, and F-measure.

As for work on CS MT, there are many efforts [\(Sinha and Thakur,](#page-11-0) [2005;](#page-11-0) [Dhar et al.,](#page-9-0) [2018;](#page-9-0) [Mahata](#page-11-11) [et al.,](#page-11-11) [2019;](#page-11-11) [Menacer et al.,](#page-11-1) [2019;](#page-11-1) [Song et al.,](#page-11-12) [2019;](#page-11-12) [Tarunesh et al.,](#page-11-3) [2021;](#page-11-3) [Xu and Yvon,](#page-12-0) [2021;](#page-12-0) [Chen](#page-9-2) [et al.,](#page-9-2) [2022;](#page-9-2) [Hamed et al.,](#page-10-0) [2022c\)](#page-10-0). To the best of our knowledge, none of these efforts presented an extensive comparison covering different segmentation techniques. With regards to the languages covered, only [Menacer et al.](#page-11-1) [\(2019\)](#page-11-1) worked on CS Arabic-English. However, since they used carefully edited UN documents, the text only included the Modern Standard Arabic variety, and contained limited types of CS.

With regards to similar corpora, [Balabel et al.](#page-9-5)

<span id="page-1-0"></span><sup>1</sup> <http://arzen.camel-lab.com/>

<span id="page-2-2"></span>

| Case      | Stem       | Ending     |             | Example   |
|-----------|------------|------------|-------------|-----------|
| Irregular | modified   | Irregular: | es          | monki+es  |
| Irregular | modified   | Regular:   | s,ed,ing,en | car+ing   |
| Irregular | modified   | Irregular: | <nil>       | went      |
| Irregular | unmodified | Irregular: | es          | church+es |
| Regular   | unmodified | Regular:   | s,ed,ing,en | car+s     |

Table 1: The ordered list of rules we follow to segment the English words.

[\(2020\)](#page-9-5) annotated CS Egyptian Arabic-English data [\(Hamed et al.,](#page-10-8) [2018\)](#page-10-8) with tokenization (canonical segmentation), lemmatization, and POS tags. However, their corpus does not contain translations.

### <span id="page-2-0"></span>3 Data

#### 3.1 Pre-existing Datasets

We use the *ArzEn* parallel corpus [\(Hamed et al.,](#page-10-9) [2020,](#page-10-9) [2022b\)](#page-10-2), which consists of speech transcriptions gathered through informal interviews with bilingual Egyptian Arabic-English speakers, as well as their English translations. The corpus consists of 6,213 sentences, where 4,154 (66.9%) are code-mixed, 1,865 (30.0%) are monolingual Arabic, and 194 (3.1%) are monolingual English. Among the code-mixed sentences, there are 1,781 (28.7%) sentences with morphological code-switching. We follow the predefined dataset splits, containing 3,341 (53.8%), 1,402 (22.6%), and 1,470 (23.7%) sentences for train, dev, and test sets, respectively. For training purposes, we also use 308k monolingual parallel sentences obtained from MADAR [\(Bouamor et al.,](#page-9-6) [2018\)](#page-9-6) and the following *LDC* corpora: [\(Gadalla et al.,](#page-9-7) [1997;](#page-9-7) [LDC,](#page-10-10) [2002b](#page-10-10)[,a;](#page-10-11) [Chen et al.,](#page-9-8) [2017;](#page-9-8) [Tracey et al.,](#page-12-2) [2021;](#page-12-2) [BBN Technologies et al.,](#page-9-9) [2012;](#page-9-9) [Chen et al.,](#page-9-10) [2019\)](#page-9-10). The preprocessing steps we apply are outlined in Appendix [A.](#page-13-0) We use *ArzEn* train set as well as the monolingual parallel corpora to train both the segmenters and MT systems. For tuning and testing the MT systems, we use the *ArzEn* dev and test sets. For tuning and testing the segmenters, we annotated a new dataset, discussed next.

#### 3.2 A New Dataset: ArzEn Surface Segmentation (ArzEnSEG) Corpus

To facilitate our research, we created a codeswitched Egyptian Arabic-English morphologically annotated dataset which we use for tuning and testing. The dataset comprises the first 500 lines of *ArzEn* dev set. Unlike [Balabel et al.](#page-9-5) [\(2020\)](#page-9-5), we opt for surface form segmentation to allow for

<span id="page-2-3"></span>

Table 2: Statistics on *ArzEnSEG* corpus.

evaluating the segmenters. We also opt for extending *ArzEn* dataset as it contains translations and is used in our MT experiments.

For Arabic word segmentation, we use the Arabic Treebank (ATB) segmentation scheme [\(Maamouri et al.,](#page-10-12) [2004;](#page-10-12) [Habash,](#page-10-13) [2010\)](#page-10-13). We choose this scheme as it is the standard tokenization scheme used in different treebanks [\(Maamouri](#page-10-12) [et al.,](#page-10-12) [2004,](#page-10-12) [2012;](#page-10-14) [Taji et al.,](#page-11-13) [2017;](#page-11-13) [Habash et al.,](#page-10-15) [2022\)](#page-10-15). It has also shown to be efficient in [Oudah](#page-11-4) [et al.](#page-11-4) [\(2019\)](#page-11-4) and has demonstrated its competitiveness in [Habash et al.](#page-10-16) [\(2013\)](#page-10-16).

For English word segmentation, we follow five rules in sequential order depending on whether the word has a regular or irregular stem and whether the word has a regular or irregular ending. Table [1](#page-2-2) exhibits the five English rules we follow in order.

All annotation decisions were made in context by two bilingual speakers who collaborated on initial annotations and quality checks. Figure [1](#page-0-1) presents an annotation example. We divide the sentences randomly into dev and test sets (250 sentences each). In Table [2,](#page-2-3) we display statistics about *ArzEnSEG*.

# <span id="page-2-1"></span>4 Segmentation Experiments

#### 4.1 Experimental Setup

We explore three categories of segmenters: unsupervised morphology-based, unsupervised frequency-based, and supervised morphologybased segmentation. For the unsupervised morphology-based segmenters, we use MorphA-Gram in addition to three segmenters from the Morfessor family: Morfessor, LMVR, and FlatCat. For unsupervised frequency-based segmenters, we use BPE. Figure [2](#page-3-0) summarizes the process of training these segmenters. For the supervised morphologybased segmenters, we use MADAMIRA [\(Pasha](#page-11-14) [et al.,](#page-11-14) [2014\)](#page-11-14), where we exploit the segmentation schemes designed for Egyptian Arabic.

<span id="page-3-0"></span>![](_page_3_Diagram_0.jpeg)

Figure 2: The unsupervised segmentation models we study in this paper and their training data dependencies. We use four systems: Morfessor (MORF), FlatCat (FC), LMVR, and MorphAGram (MorphA). The subscripts specify the training data: source (src), target (tgt), source+target (joint), and source without English, i.e., Egyptian, (src/egy).

### <span id="page-3-1"></span>4.2 Segmentation Systems

In this section, we introduce the segmentation systems used for the study. Details about the hyperparameter tuning for each system family can be found in Appendix [B.](#page-13-1) The different segmentation models and their training dataset are displayed in Figure [2.](#page-3-0)

Morfessor Family We exploit three Morfessor family tools for unsupervised morphology-based segmentation in this research: Morfessor, [\(Smit](#page-11-7) [et al.,](#page-11-7) [2014\)](#page-11-7), FlatCat [\(Grönroos et al.,](#page-10-6) [2014\)](#page-10-6), and LMVR [\(Ataman et al.,](#page-9-3) [2017\)](#page-9-3).

Morfessor is a morphological-based segmentation model which we train in an unsupervised manner. Three components form the system: the model, the cost function, and the training and decoding algorithms [\(Virpioja et al.,](#page-12-3) [2013\)](#page-12-3). The model is mainly concerned with the grammar and lexicon where the latter holds the attributes of the subwords and the grammar controls how these subwords are combined to form the word. Morfessor's grammar assumes that the subwords that form the word are independent of each other and that a word has at least one subword.

FlatCat is a variant of Morfessor which we also train in an unsupervised manner. Even though Flat-Cat builds on Morfessor and shares the same model component, they differ in their morphotactics (the set of rules that determine how the word's morphemes are arranged). FlatCat morphotactics is based on the Hidden Markov model [\(Baum and](#page-9-11) [Petrie,](#page-9-11) [1966\)](#page-9-11) which considers context. On the contrary, Morfessor's morphotactics algorithm is based on a unigram model which is not context-sensitive.

LMVR is a morphology-based segmenter that is built upon FlatCat and we train in an unsupervised manner. Nonetheless, LMVR takes into consideration the desired segmentation output vocabulary size during training.

For each tool, two models are generated; one trained on the source side; thus capable of segmenting CS data, and the other trained on the target side of the training data; thus capable of segmenting English data only. We add a *src* and *tgt* subscript to the segmenters' names to distinguish between both settings. Hence, MORFsrc, FCsrc, and LMVRsrc resemble Morfessor, FlatCat, and LMVR respectively, where the segmenters are trained on the source side. MORFtgt, FCtgt, and LMVRtgt resemble the segmenters trained on the target side.

MorphAGram We also include in this study the unsupervised morphology segmenter MorphA-Gram [\(Eskander et al.,](#page-9-12) [2020\)](#page-9-12) which is based on Adaptor Grammars. We use the *PrStSu+SM* grammar, which represents a word as a sequence of prefixes followed by a stem then a sequence of suffixes, in the unsupervised *Standard* learning setting to train the segmenters.

BPE The SentencePiece [\(Kudo and Richardson,](#page-10-17) [2018\)](#page-10-17) implementation of BPE [\(Gage,](#page-10-18) [1994;](#page-10-18) [Sen](#page-11-8)[nrich et al.,](#page-11-8) [2016\)](#page-11-8) is a frequency-based unsupervised segmenter. We train the BPE model jointly, on the concatenation of the source and target sides of the training parallel corpus. Previous work has shown that this approach is better suited for low resource settings [\(Guzmán et al.,](#page-10-19) [2019\)](#page-10-19). We refer to our joint BPE segmenter as BPEjoint.

MADAMIRA For supervised morphology-based segmenters, we use MADAMIRA's Egyptian Arabic model [\(Pasha et al.,](#page-11-14) [2014\)](#page-11-14), which was trained on the Egyptian Arabic Treebank (parts 1 through 6) [\(Maamouri et al.,](#page-10-14) [2012\)](#page-10-14). Specifically, we use MADAMIRA's *ATB\_BWFORM* and *D3\_BWFORM* schemes, henceforth MDMRAT B and MDMRD3, respectively. Both schemes apply Alif/Ya normaliza-

<span id="page-4-0"></span>

|        | Segmenter | EGY   | EMMA F1 EN | Score All |
|--------|-----------|-------|------------|-----------|
| raw    |           | 0.806 | 0.953      | 0.838     |
| MorphA | src       | 0.682 | 0.942      | 0.737     |
| MORF   | src       | 0.814 | 0.888      | 0.832     |
| FC     | src       | 0.821 | 0.961      | 0.851     |
| LMVR   | src       | 0.836 | 0.961      | 0.863     |
| LMVR   | src/egy   | 0.838 | 0.953      | 0.863     |
| MorphA | tgt       | 0.806 | 0.953      | 0.838     |
| MORF   | tgt       | 0.147 | 0.951      | 0.327     |
| FC     | tgt       | 0.806 | 0.952      | 0.838     |
| LMVR   | tgt       | 0.806 | 0.966      | 0.842     |
| LMVR   | joint     | 0.841 | 0.963      | 0.868     |
| BPE    | joint     | 0.678 | 0.814      | 0.707     |
| MDMR   | AT B      | 0.935 | 0.953      | 0.939     |
| MDMR   | D 3       | 0.868 | 0.953      | 0.887     |

Table 3: EMMA F1 score calculated on *ArzEnSEG* test set for the raw data as well as the segmented data using the different segmenters. The Arabic gold segmentation is based on the ATB segmentation scheme. We show the overall score (All) and language-specific scores calculated on the Egyptian Arabic (EGY) and English (EN) words separately. Segmenter names with a *src*, *tgt*, and *joint* subscripts represent segmenters that are trained on the source, target, and source+target sides respectively. The best performing segmenters from each category are highlighted in bold.

tion and segment the Arabic clitics. MDMRD<sup>3</sup> splits the Arabic definite article È@ *Al* (the), while MDMRAT B does not.

#### <span id="page-4-2"></span>4.3 Segmentation Results

To evaluate the performance of the segmenters, we use EMMA F1 score [\(Spiegler and Monson,](#page-11-15) [2010\)](#page-11-15). Results in Table [3,](#page-4-0) reported on *ArzEnSEG* test set, show overall and language-specific scores.

Unsupervised morphology-based segmentation Results show that LMVR outperforms the other unsupervised morphology-based segmenters in terms of segmenting Arabic and English words. We perform further experiments where we train 2 additional models: i) a model trained jointly on the concatenation of the source and target sides of the parallel corpus, and ii) a model trained on the Arabic words only in the source side (where English words are dropped). Therefore, the former model is capable of segmenting both languages, while the latter is only tailored for segmenting Arabic words. We perform these experiments using LMVR, given that it outperforms the other segmenters. We refer to these models as LMVRjoint and LMVRsrc/egy respectively, as outlined in Figure [2.](#page-3-0) Results show that joint training provides best EMMA scores.

Supervised morphology-based segmentation As shown in Table [3,](#page-4-0) both supervised morphologybased segmenters MDMRAT B and MDMRD<sup>3</sup> outperform all other segmenters. Their superiority in segmenting Arabic is expected, as they are trained on human-annotated data and hence are capable of generating infrequent morphemes. Additionally, MADAMIRA has a morphological analyzer embedded in it, which in turn enriches the inspection of Arabic words prior to segmentation. Higher EMMA scores are reported for MDMRAT B over MDMRD3, which is also expected, as *ArzEnSEG* is segmented following the ATB scheme.

Unsupervised frequency-based segmentation As expected, BPEjoint performs the worst in the morphology-based segmentation task, as it is designed for agnostic segmentation for the purpose of improving downstream tasks.

Further analysis We surprisingly find that MorphAtgt outperforms MorphAsrc on Arabic words and FCsrc outperforms FCtgt on English words. Therefore, we conduct an internal analysis where we look into the percentage of over and under segmentations.[<sup>2</sup>](#page-4-1) In Appendix [C,](#page-13-2) we present the number of under and over segmented words for each segmentation approach. Our analysis shows that MorphAsrc over segments 25% of the Arabic words. We observe that in 20% of these over segmentation cases, the Arabic definite article is segmented. For example, the word I. JºË@ *Alktb* 'the books' is segmented to I. J»#È@ *Al#ktb* which is considered valid in segmentation schemes like D3. However, since we use the ATB scheme in *ArzEnSEG* annotation, the EMMA system penalizes the MorphAsrc segmenter and rewards the MorphAtgt segmenter which leaves most of the Arabic words and the definite article unsegmented. Another case is the segmentation of affixes, which is not done in ATB. For example, 16% of the over segmentation cases are separation of the Ta-Marbuta (feminine nominal ending) in Arabic words. The rest of the cases are grammatically incorrect segmentations. FCtgt is also shown to under segment around 17% more English words compared to FCsrc which can contribute to worse scores. We also observe that MORFtgt performs

<span id="page-4-1"></span><sup>2</sup>Over segmentation is a term we use to indicate that the word gets segmented to more morphemes compared to the gold standard segmentation. Meanwhile, under segmentation is a term we use to convey that the word is segmented into fewer morphemes than the gold standard segmentation.

significantly worse than the other segmenters when segmenting Arabic words, despite the fact that 81% of the Arabic words do not require segmentation. Internal analysis shows that MORFtgt over segments the Arabic words to the character level in an attempt to extract the underlying morphology of Egyptian Arabic, which it was not trained on.

# <span id="page-5-0"></span>5 Machine Translation Experiments

Since no previous research investigates the best segmentation technique for NMT of the code-switched Egyptian Arabic–English language pair, we explore training NMT models using the various segmentation setups discussed in Section [4](#page-2-1) to answer RQ1. Moreover, we analyze the performance of the topperforming MT systems on different types of CS sentences to answer RQ2. Afterward, we compare the MT scores against the EMMA F1 scores discussed in Section [4.3](#page-4-2) to answer RQ3.

#### 5.1 Experimental Setup

We train Transformer models for our MT systems using Fairseq [\(Ott et al.,](#page-11-16) [2019\)](#page-11-16) on a single GeForce RTX 3090 GPU. We use the hyperparameters from the FLORES[<sup>3</sup>](#page-5-1) benchmark for low-resource MT [Guzmán et al.](#page-10-19) [\(2019\)](#page-10-19), which we list in Appendix [D.](#page-13-3) Afterwards, we evaluate the MT models on *ArzEn*'s dev and test sets using chrF2 [\(Popovic´,](#page-11-17) [2015\)](#page-11-17).[<sup>4</sup>](#page-5-2) We choose chrF2 over BLEU [\(Papineni et al.,](#page-11-18) [2002\)](#page-11-18) as it rewards partially correct translations which makes it a convenient choice for our research, and because chrF has shown to have higher correlation with human judgments over BLEU [\(Kocmi et al.,](#page-10-20) [2021\)](#page-10-20).

### 5.2 Machine Translation Systems

We experiment with different categories of segmentation setups. Table [4](#page-5-3) shows all the different setups that we explore. See Table [10](#page-15-0) in Appendix [D](#page-13-3) for training time.

For the unsupervised morphology-based segmentations, we use MorphAGram, Morfessor, FlatCat, and LMVR to segment the source/target sides of the parallel corpus, where the segmenters were trained on each side separately (see Figure [2\)](#page-3-0). For the best performing segmenter, we further investigate the best training setting, where we investigate using segmenters trained only on Arabic

<span id="page-5-3"></span>

|                 | Segmentation     |            |            | chrF2 |      |
|-----------------|------------------|------------|------------|-------|------|
| Source          |                  | Target     |            |       |      |
| EGY             |                  | EN         | EN         | dev   | test |
|                 | raw              |            | raw        | 47.1  | 49.9 |
| Unsupervised    | Morphology-based |            | Segmenters |       |      |
| MorphA          | src              | MorphA     | tgt        | 47.0  | 49.7 |
| MORF            | src              | MORF       | tgt        | 47.4  | 50.8 |
| FC              | src              | FC         | tgt        | 47.2  | 50.6 |
| LMVR            | src              | LMVR       | tgt        | 48.3  | 51.7 |
| LMVR            | joint            | LMVR       | joint      | 48.8  | 52.5 |
| LMVR src/egy    | LMVR             | tgt LMVR   | tgt        | 48.9  | 52.9 |
| LMVR src        | LMVR             | tgt LMVR   | tgt        | 48.8  | 52.9 |
| LMVR src/egy    | LMVR             | src LMVR   | tgt        | 48.5  | 52.0 |
| Frequency-based |                  | Segmenters |            |       |      |
| BPE             | joint            | BPE        | joint      | 50.1  | 53.7 |
| BPE             | joint            |            | raw        | 47.4  | 50.8 |
|                 | raw              | BPE        | joint      | 44.3  | 46.9 |
| Supervised      | Morphology-based | Segmenters |            |       |      |
| MDMR AT B       |                  | raw        | raw        | 48.8  | 52.1 |
| MDMR D          | 3                | raw        | raw        | 47.9  | 51.1 |
|                 | Combination      | Segmenters |            |       |      |
| MDMR AT B +BPE  | joint BPE        | joint      | raw        | 46.5  | 50.1 |
| MDMR AT B +BPE  | joint BPE        | joint BPE  | joint      | 50.2  | 53.8 |
| MDMR D 3 +BPE   | joint BPE        | joint      | raw        | 46.9  | 50.7 |
| MDMR D 3 +BPE   | joint BPE        | joint BPE  | joint      | 49.8  | 53.3 |

Table 4: The chrF2 results of our NMT systems with different segmentation combinations on *ArzEn*'s dev and test sets. Numbers highlighted in bold show the best performing system in each category.

words on the source side as well as segmenters that are trained jointly on both sides.

For the supervised morphology-based segmentations, we only follow one approach and that is segmenting the source side using MDMRAT B or MDMRD<sup>3</sup> segmenters. This causes the English words to be left unsegmented.

For the unsupervised frequency-based segmentations, we exploit the jointly trained model, BPEjoint, to segment the source side only, target side only, or both sides of the parallel corpus.

Finally, inspired by the work of [Oudah et al.](#page-11-4) [\(2019\)](#page-11-4), we explore combinations between BPE and supervised morphology-based segmenters. As shown in Table [4,](#page-5-3) for the source side, we apply BPEjoint on top of segmentations provided by either MDMRAT B or MDMRD3. For the target side, we either leave it in the raw format or apply BPEjoint .

# 5.3 Machine Translation Results

Table [4](#page-5-3) shows the different MT systems and their performance on *ArzEn*'s dev and test sets.

Amongst the unsupervised morphology-based segmenters, LMVR outperforms the other segmenters. We find that training language-specific

<span id="page-5-1"></span><sup>3</sup> FLORES hyperparameters outperform [Vaswani et al.](#page-12-4) [\(2017\)](#page-12-4) for our code-switched pair by +0.4 chrF2 points.

<span id="page-5-2"></span><sup>4</sup>We use sacreBLEU's [\(Post,](#page-11-19) [2018\)](#page-11-19) implementation of chrF2.

segmenters (using LMVRsrc/egy for Arabic words and LMVRtgt for English words) outperforms training the segmenter jointly (LMVRjoint). This setup gives the best performing model, referred to as MTLMV R.

Amongst the supervised morphology-based segmenters, the setup with MDMRAT B is the best, which we refer to as MTAT B. The finding is consistent with [Oudah et al.](#page-11-4) [\(2019\)](#page-11-4)'s results.

For unsupervised frequency-based segmenters, using BPEjoint to segment both source and target sides outperforms MTLMV R by +0.8 chrF2 points and MTAT B by +1.6 chrF2 points, which we refer to as MTBP E. We observe that the ranking of these segmenters in MT performance is in reverse order compared to their ranking in segmentation task performance. We discuss this later in Section [6.](#page-7-0)

Most interestingly, contrary to [\(Oudah et al.,](#page-11-4) [2019\)](#page-11-4), we find that applying BPEjoint on top of MDMRAT B, which we refer to as MTAT B+BP E, slightly improves over MTBP E but without statistical significance. However, MTAT B+BP E outperforms MTAT B and MTLMV R with statistical significance.[<sup>5</sup>](#page-6-0) We further investigate the effectiveness and statistical significance achieved by this approach in a learning curve with varying the training data size in Section [5.4.](#page-6-1)

Finally, we note that segmenting English words on the source and target sides consistently, while controlling all other conditions, is always advantageous, as shown in Table [4.](#page-5-3)

# <span id="page-6-1"></span>5.4 Analysis

We further analyze the performance of the top MT systems from each segmentation setup (MTAT B+BP E, MTBP E, MTLMV R, and MTAT B). We first look into the number of Out-of-Vocabulary (OOV) tokens associated with each of the top-performing MT systems to examine whether it has an impact on their final ranking. Secondly, we investigate whether the ranking of the systems is consistent across the different types of sentences. We evaluate the systems against varying morphological richness, English percentages, and CS types. Thirdly, we further investigate the effectiveness of applying BPE over ATB compared to using each segmenter on its own.

<span id="page-6-2"></span>![](_page_6_Figure_0.jpeg)

Figure 3: The percentage of the OOV words generated from each of the top-performing MT systems from each segmentation setup on *ArzEn*'s dev set.

We conduct this analysis across different CS types and sizes of training data.

OOV To further study the reason behind MTBP E and MTAT B+BP E top performance, we observe if the top-performing MT systems' ranking is linked with the percentage of OOV in the different MT systems. As shown in Figure [3,](#page-6-2) we find that for MTAT B+BP E and MTBP E , the OOV percentage is 0%. However, for MTLMV R and MTAT B , the percentage rises to 4.90% and 9.70%, respectively, which we believe contributes to worsening the MT systems.

Evaluating Systems Under Different Sentence Categories We evaluate the performance of the MT systems for sentences falling under different ranges of (i) morphological richness, (ii) percentage of CS English words, and (iii) sentence CS types. Morphological richness of a sentence is calculated as the quotient of the number of tokens in the segmented sentence and unsegmented original sentence. As expected, the performance of all the MT models decreases as the morphological richness increases and there is a boost in performance across all systems when the percentage of English words increases (see Appendix [E\)](#page-14-0). We observe that the MTAT B+BP E and MTBP E perform the best across all ranges for the first two features. We then evaluate the performance of the MT systems across sentences according to CS types: purely monolingual Arabic, CS, and CS having MCS [\(Hamed](#page-10-21) [et al.,](#page-10-21) [2022a;](#page-10-21) [Mager et al.,](#page-10-22) [2019\)](#page-10-22). We observe that for all systems, the performance across CS sentences is higher than across monolingual Arabic sentences. We also observe that among CS sentences, the performance is reduced in the case of morphologically code-switched sentences. We believe that the following two factors can be contributing to these results. Firstly, the complex MCS

<span id="page-6-0"></span><sup>5</sup>We use Paired Significance Tests for Multi System Evaluation provided by SacreBLEU for the significance tests [https://github.com/mjpost/sacrebleu#](https://github.com/mjpost/sacrebleu#paired-bootstrap-resampling---paired-bs) [paired-bootstrap-resampling---paired-bs](https://github.com/mjpost/sacrebleu#paired-bootstrap-resampling---paired-bs).

constructions might impose challenges to the MT system. Secondly, we observe that the average length of MCS sentences is higher than that of CS sentences in general. This is partially due to the fact that the tokens in MCS words are space-separated during the data preprocessing step. We report that on average, CS sentences contain 21.1 words (21.4 tokens), while MCS sentences contain 25.0 words (26.3 tokens).

Further Investigating the Effectiveness of MTAT B+BP E over MTBP E and MTAT B We study whether the ranking of MTAT B+BP E, MTBP E, and MTAT B is altered when going from a low-resource to an extreme low-resource setting across different sentence types. We achieve this by varying the MT training data to 25% and 50% of its original size. The results are shown in Table [5.](#page-7-1)

We observe that the effectiveness of the MTAT B+BP E varies under constrained conditions. For monolingual Arabic sentences, when training the MT systems on 100% of data, we see that MTAT B+BP E is not statistically significant over MTBP E and MTAT B. Moreover, MTAT B+BP E was outperformed by MTBP E. However, when training with 25% and 50% of data, MTAT B+BP E outperforms MTBP E and MTAT B with statistical significance across all sentence types. We further exhibit this in Figure [4](#page-8-0) when all sentence categories are considered during analysis under different data sizes. This finding highlights the importance of combining morphology-based and frequency-based segmentations in extremely low-resource scenarios.

We also observe that across all data sizes, MTAT B performs the worst on CS sentences. Our first hypothesis is that this is due to English words left unsegmented. However, results in Table [3](#page-4-0) contradict this hypothesis. Our second hypothesis is that since MDMRAT B takes into consideration the context of the word prior to segmentation, the English words in the CS sentences might break the flow of the sentence, hence negatively impacting the context of the word, thus worsening the score.

System Selection As per our findings, MTAT B+BP E is always the best choice across all sentence types in extreme low-resource settings. However, when training on 100% of the data, MTBP E improves slightly over MTAT B+BP E on monolingual Arabic sentences. Therefore, we create a system selection setup

<span id="page-7-1"></span>

| Size MT | System | All       |     | EGY  |     | CS   |     | MCS  |     |
|---------|--------|-----------|-----|------|-----|------|-----|------|-----|
| 25% MT  | AT B + | BP E 39.8 | (1) | 36.6 | (1) | 40.6 | (1) | 40.0 | (1) |
| MT      | BP E   | 38.4      | (2) | 35.6 | (3) | 39.1 | (2) | 38.5 | (2) |
| MT      | AT B   | 36.9      | (3) | 35.9 | (2) | 37.0 | (3) | 36.0 | (3) |
| 50% MT  | AT B + | BP E 45.9 | (1) | 42.1 | (1) | 46.8 | (1) | 46.4 | (1) |
| MT      | BP E   | 44.5      | (2) | 40.7 | (3) | 45.5 | (2) | 44.8 | (2) |
| MT      | AT B   | 44.0      | (3) | 41.4 | (2) | 44.7 | (3) | 44.0 | (3) |
| 100% MT | AT B + | BP E 50.2 | (1) | 44.4 | (2) | 51.5 | (1) | 51.3 | (1) |
| MT      | BP E   | 50.1      | (2) | 44.6 | (1) | 51.3 | (2) | 51.1 | (2) |
| MT      | AT B   | 48.8      | (3) | 44.2 | (3) | 49.8 | (3) | 49.4 | (3) |

Table 5: We compare the results of the best performing MT system (MTAT B+BP E) which utilizes BPE on top of ATB segmentation against the MT systems that utilize BPE (MTBP E) or ATB (MTAT B) only on *ArzEn*'s dev set. We report chrF2 results when training on 25%, 50%, and 100% of the training data. Results are shown for different types of sentences: monolingual Egyptian Arabic (EGY), code-switched (CS), and morphologically code-switched (MCS), as well as all sentences (All). The ranking of the MT systems with respect to each other is represented by the numbers between parentheses, where (1) is the best rank and (3) is the worst.

which uses both, MTAT B+BP E and MTBP E, to investigate if it would lead to further improvements. In this setup, the CS and monolingual English sentences are translated using MTAT B+BP E, while monolingual Arabic sentences are translated using MTBP E. Despite the hybrid system showing an overall improvement of +0.1 chrF2 points over MTAT B+BP E, the improvement is not statistically significant.

### <span id="page-7-0"></span>6 Discussion

We revisit the RQs we outlined in our introduction.

RQ1 - Which segmentation setup performs the best in the downstream MT task across different training sizes? Results show that frequencybased segmentation applied on top of morphologybased segmentation outperforms the other segmentation techniques, with statistical significance on lower resource settings. The superiority of this approach is seen across sentences with varying morphological richness, percentage of English words, and across sentences with different CS types. We believe the strength of the combination is because it exploits complementarity of both methods. On one hand, supervised morphology-based segmenters bring in high correctness; however, they are not always robust, having high OOV rates. On the other hand, while BPE segmentation is not necessarily morphologically correct, it achieves high robustness. The robustness of BPE is consistent with the findings in [Banerjee and Bhattacharyya](#page-9-13) [\(2018\)](#page-9-13).

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 4: Demonstrates the effectiveness of applying BPE on top of ATB segmentation (MTAT B+BP E) as opposed to using either approaches separately (MTBP E and MTAT B), which is confirmed when reducing the amount of training data. Results are reported on *ArzEn*'s dev set.

RQ2 - Does the effectiveness of the different segmenters in the MT task differ according to the CS type of the source sentence? We observe that the effectiveness of the different segmenters on MT performance is consistent across two categories of CS sentences; those with and without MCS. However, when comparing their effectiveness on monolingual Arabic vs. CS sentences, we observe that the rankings between segmenters are not consistent. In the case of constrained data size settings (25% and 50% of data), we observe a clear pattern where MTAT B outperforms MTBP E on monolingual sentences, while MTBP E outperforms MTAT B on CS. In the case of using 100% of the training data, MTAT B+BP E outperforms MTBP E on CS sentences; however, MTBP E outperforms MTAT B+BP E on monolingual Arabic sentences. Since our test and dev sets are dominated by CS sentences (61.5% and 63.8%, respectively), we believe that the overall ranking is more greatly affected by the systems' performance on CS sentences, thus reflecting the same ranking on the overall evaluation set as that across CS sentences.

RQ3 - Is there a correlation between a more morphologically correct segmentation and MT performance? For unsupervised morphologybased segmenters, a segmenter with a better segmentation EMMA F1 score also scores better in the downstream MT task. However, we cannot hypothesize that a better segmentation score implies a better translation system, as counter examples exist. For example, while we notice that

MDMRAT B gives the best segmentation in terms of EMMA F1 score, it does not outperform any of the top-performing MT systems. We hypothesize that despite MDMRAT B's capability of generating morphologically correct segmentations, it can generate infrequent morphemes due to the out-ofdomain data which it is trained on. This may not only increase the sentence length which worsens MT performance as shown in [Mager et al.](#page-10-7) [\(2022\)](#page-10-7), but may also be one of the contributing factors to the 9.70% OOV percentage found in MTAT B. On the contrary, BPEjoint performs the worst in the segmentation task as we expect, since it is designated for agnostic-based segmentations; however, it surpasses the top-performing MT models. We believe this is due to its capability to generate semicorrect segmentation and to reduce OOV rates.

# 7 Conclusion and Future Work

In this paper, we study the impact of a comprehensive set of morphological and frequency-based segmentation methods on MT, where the source is a code-switched text. The experiments are performed on code-switched Arabic-English to English. We found that the supervised morphological segmenter achieved the best performance on the segmentation task, followed by unsupervised morphological methods, and finally, unsupervised frequency-based. Afterward, we train 18 different MT systems with different source and target side segmentations. We find that the rank of the segmenters is reversed, as BPE's could not be outperformed (significantly) by any morphologicalinspired segmentation method. However, combining morphology-based and frequency-based segmentations has shown to give improvements, which are statistically significant in lower resource settings, where the training data size is reduced to 25% and 50%. For future work, we plan to apply our different MT setups on other low resource and code-switched language pairs. Specifically, we plan to explore languages with different typologies, to study whether or not the relation between the data size and choice of the segmentation setup (frequency-based, morphology-based, or a mix) is based on morphological features and data size rather than the language itself. Moreover, we plan to extend our annotated dataset, *ArzEnSEG*, by adding further details to allow evaluating different schemes.

# Limitations

The first challenge we face in this work is the computational power needed to tune the Morfessor family segmenters. Therefore, in an attempt to overcome this challenge, for the Morfessor family, the choice of the optimal hyperparameters is dependent on the parent tool. For instance, the optimal hyperparameters for Morfessor are directly used in its FlatCat variant and the hyperparameters specific to FlatCat are then tuned. The same applies for LMVR which is a variant of FlatCat. Moreover, we cannot verify whether or not our results will hold for languages with different typologies, specifically those that are low resource and codeswitched. Therefore, the results of this research must be seen in light of these limitations. Ethics Statement We could not find any potential harm that might derive from this work. However, we understand that translation as a whole can impact the cultural and social life of the people that use it. This has been used in the past in a harmful way, i.e., to spread colonial views [\(Mbuwayesango,](#page-11-20) [2018\)](#page-11-20). Therefore, we call the final user to use this work ethically. Regarding the annotation process, all manual annotations were made by a subset of the authors of this paper. Therefore, no hiring of external workers was necessary. Acknowledgements We want to thank all the anonymous reviewers for their helpful comments and suggestions. This project has benefited from financial support to Marwa Gaser and Manuel Mager by the DAAD (German Academic Exchange Service). References Abdulmohsen Al-Thubaity and Abdullah Al-Subaie. 2015. Effect of word segmentation on Arabic text classification. In *Proceedings of the International Conference on Asian Language Processing (IALP)*, pages 127–131. Duygu Ataman, Matteo Negri, Marco Turchi, and Marcello Federico. 2017. Linguistically motivated vocabulary reduction for neural machine translation from Turkish to English. *The Prague Bulletin of Mathematical Linguistics*, 108(1):331–342. Mohamed Balabel, Injy Hamed, Slim Abdennadher, Ngoc Thang Vu, and Özlem Çetinoglu. 2020. Cairo ˘ student code-switch (CSCS) corpus: An annotated Egyptian Arabic-English corpus. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 3973–3977. Tamali Banerjee and Pushpak Bhattacharyya. 2018. Meaningless yet meaningful: Morphology grounded subword-level NMT. In *Proceedings of the workshop on subword/character level models*, pages 55–60. Leonard E Baum and Ted Petrie. 1966. Statistical inference for probabilistic functions of finite state Markov chains. *The annals of mathematical statistics*, 37(6):1554–1563. Raytheon BBN Technologies, Linguistic Data Consortium, and Sakhr Software. 2012. Arabic-Dialect/English parallel text – LDC2012T09. Web Download. Philadelphia: Linguistic Data Consortium. Houda Bouamor, Nizar Habash, Mohammad Salameh, Wajdi Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa, Fadhl Eryani, Alexander Erdmann, and Kemal Oflazer. 2018. The MADAR Arabic Dialect Corpus and Lexicon. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, Miyazaki, Japan. Özlem Çetinoglu, Sarah Schulz, and Ngoc Thang Vu. ˘ 2016. Challenges of computational processing of code-switching. In *Proceedings of the Workshop on Computational Approaches to Code Switching*, pages 1–11. Shuguang Chen, Gustavo Aguilar, Anirudh Srinivasan, Mona Diab, and Thamar Solorio. 2022. CALCS 2021 shared task: Machine translation for code-switched data. *arXiv preprint arXiv:2202.09625*. Song Chen, Dana Fore, Stephanie Strassel, Haejoong Lee, and Jonathan Wright. 2017. BOLT Egyptian Arabic SMS/Chat and Transliteration LDC2017T07. Philadelphia: Linguistic Data Consortium. Song Chen, Jennifer Tracey, Christopher Walker, and Stephanie Strassel. 2019. BOLT Arabic discussion forum parallel training data. Linguistic Data Consortium (LDC) catalog number LDC2019T01, ISBN 1-58563-871-4. Mrinal Dhar, Vaibhav Kumar, and Manish Shrivastava. 2018. Enabling code-mixed translation: Parallel corpus creation and MT augmentation approach. In *Proceedings of the Workshop on Linguistic Resources for Natural Language Processing*, pages 131–140. Ramy Eskander, Francesca Callejas, Elizabeth Nichols, Judith L Klavans, and Smaranda Muresan. 2020. Morphagram, evaluation and framework for unsupervised morphological segmentation. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 7112–7122. Hassan Gadalla, Hanaa Kilany, Howaida Arram, Ashraf Yacoub, Alaa El-Habashi, Amr Shalaby, Krisjanis Karins, Everett Rowson, Robert MacIntyre, Paul Kingsbury, David Graff, and Cynthia McLemore. 1997. CALLHOME Egyptian Arabic transcripts – LDC97T19. Web Download. Philadelphia: Linguistic Data Consortium.

<span id="page-9-13"></span><span id="page-9-11"></span><span id="page-9-9"></span>

<span id="page-9-6"></span><span id="page-9-2"></span><span id="page-9-1"></span>

<span id="page-9-12"></span><span id="page-9-10"></span><span id="page-9-8"></span><span id="page-9-7"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-0"></span>

<span id="page-10-22"></span><span id="page-10-21"></span><span id="page-10-20"></span><span id="page-10-19"></span><span id="page-10-18"></span><span id="page-10-17"></span><span id="page-10-16"></span><span id="page-10-15"></span><span id="page-10-14"></span><span id="page-10-13"></span><span id="page-10-12"></span><span id="page-10-11"></span><span id="page-10-10"></span><span id="page-10-9"></span><span id="page-10-8"></span><span id="page-10-7"></span><span id="page-10-6"></span><span id="page-10-5"></span><span id="page-10-4"></span><span id="page-10-3"></span><span id="page-10-2"></span><span id="page-10-1"></span><span id="page-10-0"></span>Philip Gage. 1994. A new algorithm for data compression. *C Users Journal*, 12(2):23–38. Stig-Arne Grönroos, Sami Virpioja, and Mikko Kurimo. 2020. Morfessor EM+Prune: Improved subword segmentation with expectation maximization and pruning. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 3944–3953. Stig-Arne Grönroos, Sami Virpioja, Peter Smit, and Mikko Kurimo. 2014. Morfessor flatcat: An HMMbased method for unsupervised and semi-supervised learning of morphology. In *Proceedings of the International Conference on Computational Linguistics (COLING): Technical Papers*, pages 1177–1185. Francisco Guzmán, Peng-Jen Chen, Myle Ott, Juan Pino, Guillaume Lample, Philipp Koehn, Vishrav Chaudhary, and Marc'Aurelio Ranzato. 2019. The FLORES evaluation datasets for low-resource machine translation: Nepali–English and Sinhala– English. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 6098– 6111, Hong Kong, China. Nizar Habash, Muhammed AbuOdeh, Dima Taji, Reem Faraj, Jamila El Gizuli, and Omar Kallas. 2022. Camel treebank: An open multi-genre Arabic dependency treebank. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 2672–2681. Nizar Habash, Mona Diab, and Owen Rambow. 2012a. Conventional Orthography for Dialectal Arabic. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 711–718, Istanbul, Turkey. Nizar Habash, Ramy Eskander, and Abdelati Hawwari. 2012b. A Morphological Analyzer for Egyptian Arabic. In *Proceedings of the Workshop of the Special Interest Group on Computational Morphology and Phonology (SIGMORPHON)*, pages 1–9, Montréal, Canada. Nizar Habash, Ryan Roth, Owen Rambow, Ramy Eskander, and Nadi Tomeh. 2013. Morphological Analysis and Disambiguation for Dialectal Arabic. In *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, Atlanta, Georgia. Nizar Habash, Abdelhadi Soudi, and Tim Buckwalter. 2007. On Arabic Transliteration. In A. van den Bosch and A. Soudi, editors, *Arabic Computational Morphology: Knowledge-based and Empirical Methods*, pages 15–22. Springer, Netherlands. Nizar Y Habash. 2010. Introduction to Arabic natural language processing. *Synthesis lectures on human language technologies*, 3(1):1–187. Injy Hamed, Pavel Denisov, Chia-Yu Li, Mohamed Elmahdy, Slim Abdennadher, and Ngoc Thang Vu. 2022a. Investigations on speech recognition systems for low-resource dialectal Arabic–English codeswitching speech. *Computer Speech & Language*, 72:101278. Injy Hamed, Mohamed Elmahdy, and Slim Abdennadher. 2018. Collection and analysis of code-switch Egyptian Arabic-English speech corpus. In *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*. Injy Hamed, Nizar Habash, Slim Abdennadher, and Ngoc Thang Vu. 2022b. ArzEn-ST: A three-way speech translation corpus for code-switched Egyptian Arabic-English. In *Proceedings of the Workshop for Arabic Natural Language Processing (WANLP)*. Injy Hamed, Nizar Habash, Slim Abdennadher, and Ngoc Thang Vu. 2022c. Investigating lexical replacements for Arabic-English code-switched data augmentation. *arXiv preprint arXiv:2205.12649*. Injy Hamed, Ngoc Thang Vu, and Slim Abdennadher. 2020. ArzEn: A speech corpus for code-switched Egyptian Arabic-English. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 4237–4246. Tom Kocmi, Christian Federmann, Roman Grundkiewicz, Marcin Junczys-Dowmunt, Hitokazu Matsushita, and Arul Menezes. 2021. To ship or not to ship: An extensive evaluation of automatic metrics for machine translation. In *Proceedings of the Conference on Machine Translation*, pages 478–494. Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations*, pages 66–71, Brussels, Belgium. LDC. 2002a. 1997 hub5 Arabic transcripts – LDC2002T39. Web Download. Philadelphia: Linguistic Data Consortium. LDC. 2002b. CALLHOME Egyptian Arabic transcripts supplement – LDC2002T38. Web Download. Philadelphia: Linguistic Data Consortium. Mohamed Maamouri, Ann Bies, Tim Buckwalter, and Wigdan Mekki. 2004. The Penn Arabic Treebank: Building a Large-Scale Annotated Arabic Corpus. In *Proceedings of the International Conference on Arabic Language Resources and Tools*, pages 102– 109, Cairo, Egypt. Mohamed Maamouri, Ann Bies, Seth Kulick, Dalila Tabessi, and Sondos Krouna. 2012. Egyptian Arabic Treebank DF Parts 1-8 V2.0 - LDC catalog numbers LDC2012E93, LDC2012E98, LDC2012E89, LDC2012E99, LDC2012E107, LDC2012E125, LDC2013E12, LDC2013E21. Manuel Mager, Özlem Çetinoglu, and Katharina Kann. ˘ 2019. Subword-level language identification for intra-word code-switching. In *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2005–2011. Manuel Mager, Arturo Oncevay, Elisabeth Maier, Katharina Kann, and Thang Vu. 2022. BPE vs. morphological segmentation: A case study on machine translation of four polysynthetic languages. In *Findings of*

- <span id="page-11-21"></span><span id="page-11-20"></span><span id="page-11-19"></span><span id="page-11-18"></span><span id="page-11-17"></span><span id="page-11-16"></span><span id="page-11-15"></span><span id="page-11-14"></span><span id="page-11-13"></span><span id="page-11-12"></span><span id="page-11-11"></span><span id="page-11-10"></span><span id="page-11-9"></span><span id="page-11-8"></span><span id="page-11-7"></span><span id="page-11-6"></span><span id="page-11-5"></span><span id="page-11-4"></span><span id="page-11-3"></span><span id="page-11-2"></span><span id="page-11-1"></span><span id="page-11-0"></span>*the Association for Computational Linguistics (ACL)*, pages 961–971. Sainik Kumar Mahata, Soumil Mandal, Dipankar Das, and Sivaji Bandyopadhyay. 2019. Code-mixed to monolingual translation framework. In *Proceedings of the Forum for Information Retrieval Evaluation*, pages 30–35. Dora Rudo Mbuwayesango. 2018. The bible as tool of colonization. *Colonialism and the Bible: Contemporary Reflections from the Global South*, page 31. Mohamed Amine Menacer, David Langlois, Denis Jouvet, Dominique Fohr, Odile Mella, and Kamel Smaïli. 2019. Machine translation on a parallel codeswitched corpus. In *Canadian Conference on Artificial Intelligence*, pages 426–432. Ossama Obeid, Nasser Zalmout, Salam Khalifa, Dima Taji, Mai Oudah, Bashar Alhafni, Go Inoue, Fadhl Eryani, Alexander Erdmann, and Nizar Habash. 2020. CAMeL tools: An open source python toolkit for Arabic natural language processing. In *Proceedings of the Language Resources and Evaluation Conference (LREC)*, pages 7022–7032, Marseille, France. John E Ortega, Richard Castro Mamani, and Kyunghyun Cho. 2020. Neural machine translation with a polysynthetic low resource language. *Machine Translation*, 34(4):325–346. Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. FAIRSEQ: A fast, extensible toolkit for sequence modeling. In *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)*, pages 48–53, Minneapolis, Minnesota. Mai Oudah, Amjad Almahairi, and Nizar Habash. 2019. The impact of preprocessing on Arabic-English statistical and neural machine translation. In *Proceedings of Machine Translation Summit XVII Volume 1: Research Track*, pages 214–221. Yirong Pan, Xiao Li, Yating Yang, and Rui Dong. 2020. Morphological word segmentation on agglutinative languages for neural machine translation. *arXiv preprint arXiv:2001.01589*. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a method for automatic evaluation of machine translation. In *Proceedings of the Conference of the Association for Computational Linguistics (ACL)*, pages 311–318. Arfath Pasha, Mohamed Al-Badrashiny, Mona Diab, Ahmed El Kholy, Ramy Eskander, Nizar Habash, Manoj Pooleery, Owen Rambow, and Ryan Roth. 2014. MADAMIRA: A fast, comprehensive tool for morphological analysis and disambiguation of Arabic. In *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, pages 1094–1101, Reykjavik, Iceland. Maja Popovic. 2015. chrF: character n-gram F-score ´ for automatic MT evaluation. In *Proceedings of the Workshop on Statistical Machine Translation*, pages 392–395, Lisbon, Portugal. Matt Post. 2018. A call for clarity in reporting BLEU scores. In *Proceedings of the Conference on Machine Translation: Research Papers*, pages 186–191, Brussels, Belgium. Christian Roest, Lukas Edman, Gosse Minnema, Kevin Kelly, Jennifer Spenader, and Antonio Toral. 2020. Machine translation for English–Inuktitut with segmentation, data acquisition and pre-training. In *Proceedings of the Conference on Machine Translation*, pages 274–281. Jonne Saleva and Constantine Lignos. 2021. The effectiveness of morphology-aware segmentation in lowresource neural machine translation. In *Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL): Student Research Workshop*, pages 164–174. Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In *Proceedings of the Conference of the Association for Computational Linguistics (ACL)*, pages 1715–1725, Berlin, Germany.
  - R. Mahesh K. Sinha and Anil Thakur. 2005. Machine translation of bi-lingual Hindi-English (Hinglish) text. In *Proceedings of Machine Translation Summit X: Papers*, pages 149–156, Phuket, Thailand. Peter Smit, Sami Virpioja, Stig-Arne Grönroos, and Mikko Kurimo. 2014. Morfessor 2.0: Toolkit for statistical morphological segmentation. In *Proceedings of the Demonstrations at the Conference of the European Chapter of the Association for Computational Linguistics (EACL)*, pages 21–24, Gothenburg, Sweden. Kai Song, Yue Zhang, Heng Yu, Weihua Luo, Kun Wang, and Min Zhang. 2019. Code-switching for enhancing NMT with pre-specified translation. In *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, pages 449–459. Sebastian Spiegler and Christian Monson. 2010. EMMA: A novel evaluation metric for morphological analysis. In *Proceedings of the International Conference on Computational Linguistics (COLING)*, pages 1029–1037. Vivek Srivastava and Mayank Singh. 2020. PHINC: A parallel Hinglish social media code-mixed corpus for machine translation. In *Proceedings of the Workshop on Noisy User-generated Text (W-NUT 2020)*, pages 41–49. Dima Taji, Nizar Habash, and Daniel Zeman. 2017. Universal dependencies for Arabic. In *Proceedings of the Workshop for Arabic Natural Language Processing (WANLP)*, pages 166–176. Ishan Tarunesh, Syamantak Kumar, and Preethi Jyothi. 2021. From machine translation to code-switching: Generating high-quality code-switched text. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 3154– 3169.

- <span id="page-12-4"></span><span id="page-12-2"></span><span id="page-12-1"></span>Ahmed Tawfik, Mahitab Emam, Khaled Essam, Robert Nabil, and Hany Hassan. 2019. Morphology-aware word-segmentation in dialectal Arabic adaptation of neural machine translation. In *Proceedings of the Workshop for Arabic Natural Language Processing (WANLP)*, pages 11–17. Jennifer Tracey et al. 2021. BOLT Egyptian Arabic sms/chat parallel training data LDC2021T15. Web Download. Philadelphia: Linguistic Data Consortium. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems*, 30. Sami Virpioja, Peter Smit, Stig-Arne Grönroos, Mikko Kurimo, et al. 2013. Morfessor 2.0: Python implementation and extensions for morfessor baseline. Technical report, Aalto University, Finland. Jitao Xu and François Yvon. 2021. Can you traducir this? machine translation for code-switched input. In *Proceedings of the Workshop on Computational Approaches to Linguistic Code-Switching*, pages 84–
- <span id="page-12-5"></span><span id="page-12-3"></span><span id="page-12-0"></span>94. Nasser Zalmout and Nizar Habash. 2017. Optimizing tokenization choice for machine translation across multiple target languages. *The Prague Bulletin of Mathematical Linguistics*, 108:257–270.

### <span id="page-13-0"></span>A Data Preprocessing

We use the same preprocessing pipeline for all the corpora, where we start by removing any corpusrelated annotations. Afterward, we remove URLs and emoticons, through *tweet-preprocessor*, [6](#page-13-4) remove trailing and leading spaces, and tokenize numbers. Finally, *Moses Tokenizer*[<sup>7</sup>](#page-13-5) is applied for tokenization and empty lines are removed from the parallel corpora. For LDC2017T07 [\(Chen et al.,](#page-9-8) [2017\)](#page-9-8) and LDC2019T01 [\(Chen et al.,](#page-9-10) [2019\)](#page-9-10), some sentences have literal and intended translations for some words. Hence, we opt for one translation having all literal translations and another having all intended translations. Once all the preprocessing steps are done, we concatenate the nine corpora collectively and pass the resulting training corpus to MADAMIRA [\(Pasha et al.,](#page-11-14) [2014\)](#page-11-14) to obtain two different supervised morphological segmentations of the corpus, namely *ATB\_BWFORM* and *D3\_BWFORM* which we discuss in Section [4.2.](#page-3-1) Additionally, we obtain a raw training corpus by further tokenizing punctuation and removing emojis using MADAMIRA's D0 scheme [\(Zalmout and](#page-12-5) [Habash,](#page-12-5) [2017\)](#page-12-5). Nonetheless, we normalize the Arabic letters ø and @ to ø and @ respectively through CAMeL Tools [\(Obeid et al.,](#page-11-21) [2020\)](#page-11-21) since D0's output is not normalized.

# <span id="page-13-1"></span>B Segmenters' Hyperparameters

Morfessor family Since all Morfessor family segmenters are morphology inspired, the hyperparameters are tuned on *ArzEnSEG*'s dev set. For LMVRsrc and LMVRtgt setting the vocabulary sizes to 64k and 16k respectively outperform 3k, 5k, 8k, 16k, 32k, 100k. For LMVRjoint setting the vocabulary size to 32k outperforms 3k, 5k, 8k, 16k, 64k, and 100k. Meanwhile, For LMVRsrc/egy setting the vocabulary size to 64k outperforms 3k, 5k, 8k, 16k, 32k, and 100k.

Table [6](#page-13-6) shows the possible values used during the optimal hyperparameter search for each Morfessor tool. For Morfessor, FlatCat, and LMVR 18, 360, and 7 different segmentation models are generated. These are a result of the combination of the possible hyperparameter values. The hyperparameter combination which yields the highest EMMA score on *ArzEnSEG*'s dev set for each Morfessor

<span id="page-13-6"></span>

| Segmenters Hyperparameter -F -d -a | Hyperparameters Values Bound Morfessor [0.003, 0.005, 0.007] [log, ones, none] [recursive, viterbi] FlatCat |
|------------------------------------|-------------------------------------------------------------------------------------------------------------|
| -p                                 | [50, 60, 70, 80, 90, 100, 200, 300]                                                                         |
| –min-perplexity-length             | [1, 2, 3, 4, 5]                                                                                             |
| –min-shift-remainder               | [1, 2, 3]                                                                                                   |
| –length-threshold                  | [2, 3, 4] LMVR                                                                                              |
| –lexicon-size                      | [3k, 5k, 8k, 16k, 32k, 64k, 100k]                                                                           |

Table 6: The values bound we use during the best hyperparameter combination search for the Morfessor tools.

tool is used to segment the MT training data. The best combination values are reported in Table [7.](#page-14-1)

MorphAGram Akin to the Morfessor family, we tune the hyperparameters on *ArzEnSEG*'s dev set and train two models: one on the source side and the other on the target side of the training parallel corpus which we refer to as MorphAsrc and MorphAtgt, respectively (see Figure [2\)](#page-3-0). Tuning results show that setting the vocabulary size to 3k for MorphAsrc outperforms 5k, 8k, 16k, 32k, and 50k, while setting the vocabulary size to 50k for MorphAtgt outperforms 5k, 3k, 8k, 16k, and 32k. Nevertheless, it is worth noting that the vocabulary size on the target side is < 50k which shows that MorphAtgt performs the best when no segmentations are applied on the English words.

BPE Since BPE is a segmentation technique that is designated for agnostic segmentation for MT tasks, we tune the vocabulary size on *ArzEn*'s dev set in an NMT task. We apply a vocabulary size of 8k, which outperforms 5k, 16k, 32k, 64k.

# <span id="page-13-2"></span>C Segmenters Performance Analysis

Table [8](#page-14-2) shows the error analysis we perform on the segmenters with regards to over segmentation, under segmentation, or generating the correct number of segmentations per word.

# <span id="page-13-3"></span>D MT Hyperparameters

The MT hyperparameters are shown in Table [9.](#page-14-3) We follow the FLORES hyperparameters for lowresource language pairs. The full train command can be found on FLORES GitHub.[<sup>8</sup>](#page-13-7) The training

<span id="page-13-5"></span><span id="page-13-4"></span><sup>6</sup> <https://pypi.org/project/tweet-preprocessor/>

<sup>7</sup> [https://github.com/moses-smt/mosesdecoder/](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) [blob/master/scripts/tokenizer/tokenizer.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)

<span id="page-13-7"></span><sup>8</sup> [https://github.com/facebookresearch/flores/](https://github.com/facebookresearch/flores/blob/6641ec0e23d173906dd2e01551a430884b1dba31/floresv1/README.md#train-a-baseline-transformer-model) [blob/6641ec0e23d173906dd2e01551a430884b1dba31/](https://github.com/facebookresearch/flores/blob/6641ec0e23d173906dd2e01551a430884b1dba31/floresv1/README.md#train-a-baseline-transformer-model) [floresv1/README.md#train-a-baseline-transformer](https://github.com/facebookresearch/flores/blob/6641ec0e23d173906dd2e01551a430884b1dba31/floresv1/README.md#train-a-baseline-transformer-model)[model](https://github.com/facebookresearch/flores/blob/6641ec0e23d173906dd2e01551a430884b1dba31/floresv1/README.md#train-a-baseline-transformer-model)

<span id="page-14-1"></span>

| Data    | -F    | Morfessor -d | -a        | -p  | –min -perplexity -length | FlatCat –min -shift -remainder | –length -threshold | LMVR –lexicon -size |
|---------|-------|--------------|-----------|-----|--------------------------|--------------------------------|--------------------|---------------------|
| src     | 0.003 | log          | recursive | 200 | 1                        | 1                              | 4                  | 64k                 |
| tgt     | 0.003 | log          | recursive | 100 | 4                        | 2                              | 4                  | 16k                 |
| src/egy | 0.007 | log          | recursive | 300 | 1                        | 1                              | 2                  | 64k                 |
| joint   | 0.007 | log          | recursive | 300 | 4                        | 2                              | 4                  | 32k                 |

Table 7: The different hyperparameters used for each Morfessor family segmenter depending on whether the model is trained on the source (src), target (tgt), source without English, i.e., Egyptian, (src/egy), or source+target (joint) side(s).

<span id="page-14-2"></span>

|        | Segmenter | under | over  | EGY correct | seg. | unseg. | under | over | EN correct | seg | unseg. |
|--------|-----------|-------|-------|-------------|------|--------|-------|------|------------|-----|--------|
| raw    |           | 634   | 0     | 2,780       | 0    | 2,780  | 71    | 0    | 430        | 0   | 430    |
| MorphA | src       | 249   | 855   | 2,310       | 385  | 1,925  | 70    | 45   | 386        | 1   | 385    |
| MORF   | src       | 466   | 299   | 2,649       | 148  | 2,501  | 15    | 103  | 383        | 42  | 341    |
| FC     | src       | 592   | 8     | 2,814       | 42   | 2,772  | 56    | 7    | 438        | 15  | 423    |
| LMVR   | src       | 520   | 47    | 2,847       | 111  | 2,736  | 43    | 7    | 451        | 28  | 423    |
| MorphA | tgt       | 634   | 35    | 2,745       | 0    | 2,745  | 6     | 148  | 347        | 65  | 282    |
| MORF   | tgt       | 0     | 3,150 | 264         | 3    | 261    | 21    | 37   | 443        | 49  | 394    |
| FC     | tgt       | 634   | 0     | 2,780       | 0    | 2,780  | 66    | 8    | 427        | 5   | 422    |
| LMVR   | tgt       | 634   | 0     | 2,780       | 0    | 2,780  | 23    | 19   | 459        | 48  | 411    |
| LMVR   | joint     | 485   | 79    | 2,850       | 144  | 2,706  | 20    | 32   | 449        | 51  | 398    |
| BPE    | joint     | 338   | 368   | 2,708       | 230  | 2,478  | 28    | 132  | 341        | 30  | 311    |
| MDMR   | AT B      | 38    | 62    | 3,314       | 581  | 2,733  | 71    | 0    | 430        | 0   | 430    |
| MDMR   | D 3       | 38    | 293   | 3,083       | 561  | 2,522  | 71    | 0    | 430        | 0   | 430    |

Table 8: The table shows the number of under segmented words (under), over segmented words (over), and the number of cases where the segmenter generates the correct count of morphemes (correct) for English (EN) and Arabic (EGY) words in *ArzEnSEG* test set. Additionally, out of the correct count of morphemes (correct), we report the words which originally require segmentation (seg.) and those which do not (unseg.).

<span id="page-14-3"></span>time for MT model the training time is exhibited in Table [10.](#page-15-0)

| Hyperparameter        | Value  |
|-----------------------|--------|
| encoder-layers        | 5      |
| decoder-layer         | 5      |
| encoder-embed-dim     | 512    |
| decoder-embed-dim     | 512    |
| encoder-ffn-embed-dim | 2      |
| decoder-ffn-embed-dim | 2      |
| dropout               | 0.4    |
| attention-dropout     | 0.2    |
| relu-dropout          | 0.2    |
| weight-decay          | 0.0001 |
| label-smoothing       | 0.2    |
| warmup-updates        | 4000   |
| warmup-init-lr        | 1e-9   |

Table 9: FLORES hyperparameters for low-resource language pairs.

#### <span id="page-14-0"></span>E Evaluating Systems Under Different Sentence Categories

Figure [5](#page-15-1) shows the performance of the top MT systems from each segmentation setup across sentences of different morphological richness ratios and different percentages of English words in

*ArzEn*'s dev set. Results show that there is a general decrease in performance as the morphological richness increases. However, as the percentage of English words in the sentences increases, the performance of the systems generally improves. It is also shown that MTAT B+BP E and MTBP E achieve overall comparable performances and outperform the other systems.

<span id="page-15-1"></span>![](_page_15_Figure_0.jpeg)

Figure 5: The average chrF2 scores for the top performing MT systems from each segmentation setup across sentences with various (a) morphological richness ratios and (b) percentage of English words in *ArzEn*'s dev set. Morphological richness of a sentence is calculated as the quotient of the number of tokens in the segmented sentence and unsegmented original sentence. The bar width is indicative of the number of sentences in each bin.

<span id="page-15-0"></span>

|                 | Segmentation     |            |            | Training  |
|-----------------|------------------|------------|------------|-----------|
| Source          |                  | Target     |            | Time      |
| EGY             |                  | EN         | EN         | (seconds) |
|                 | raw              |            | raw        | 13,522    |
| Unsupervised    | Morphology-based |            | Segmenters |           |
| MorphA          | src              | MorphA     | tgt        | 24,731    |
| MORF            | src              | MORF       | tgt        | 18,916    |
| FC              | src              | FC         | tgt        | 18,225    |
| LMVR            | src              | LMVR       | tgt        | 4,476     |
| LMVR            | joint            | LMVR       | joint      | 18,019    |
| LMVR src/egy    | LMVR             | tgt LMVR   | tgt        | 22,462    |
| LMVR src        | LMVR             | tgt LMVR   | tgt        | 4,181     |
| LMVR src/egy    | LMVR             | src LMVR   | tgt        | 4,526     |
| Frequency-based |                  | Segmenters |            |           |
| BPE             | joint            | BPE        | joint      | 18,279    |
| BPE             | joint            |            | raw        | 23,193    |
|                 | raw              | BPE        | joint      | 17,905    |
| Supervised      | Morphology-based | Segmenters |            |           |
| MDMR AT B       |                  | raw        | raw        | 18,280    |
| MDMR D          | 3                | raw        | raw        | 18,519    |
|                 | Combination      | Segmenters |            |           |
| MDMR AT B +BPE  | joint BPE        | joint      | raw        | 17,629    |
| MDMR AT B +BPE  | joint BPE        | joint BPE  | joint      | 27,088    |
| MDMR D 3 +BPE   | joint BPE        | joint      | raw        | 24,256    |
| MDMR D 3 +BPE   | joint BPE        | joint BPE  | joint      | 23,611    |

Table 10: The training time in seconds of our different NMT systems.