# CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts

Journal: Knowledge-Based Systems

 Rabindra Lamsal Affiliation: School of Computing and Information Systems, The University of Melbourne, Australia Corresponding author: rlamsal@student.unimelb.edu.au    Maria Rodriguez Read Affiliation: School of Computing and Information Systems, The University of Melbourne, Australia    Shanika Karunasekera Affiliation: School of Computing and Information Systems, The University of Melbourne, Australia 

###### Abstract

Social media platforms play an essential role in crisis communication, but analyzing crisis-related social media texts is challenging due to their informal nature. Transformer-based pre-trained models like BERT and RoBERTa have shown success in various NLP tasks, but they are not tailored for crisis-related texts. Furthermore, general-purpose sentence encoders are used to generate sentence embeddings, regardless of the textual complexities in crisis-related texts. Advances in applications like text classification, semantic search, and clustering contribute to the effective processing of crisis-related texts, which is essential for emergency responders to gain a comprehensive view of a crisis event, whether historical or real-time. To address these gaps in crisis informatics literature, this study introduces CrisisTransformers, an ensemble of pre-trained language models and sentence encoders trained on an extensive corpus of over 15 billion word tokens from tweets associated with more than 30 crisis events, including disease outbreaks, natural disasters, conflicts, and other critical incidents. We evaluate existing models and CrisisTransformers on 18 crisis-specific public datasets. Our pre-trained models outperform strong baselines across all datasets in classification tasks, and our best-performing sentence encoder improves the state-of-the-art by 17.43% in sentence encoding tasks. Additionally, we investigate the impact of model initialization on convergence and evaluate the significance of domain-specific models in generating semantically meaningful sentence embeddings. The models are publicly available at: [https://huggingface.co/crisistransformers](https://huggingface.co/crisistransformers "")

###### Keywords: 

classification models , sentence encoding models , crisis informatics , social media analytics , social computing 

## 1 Introduction

Social media platforms, such as Facebook and Twitter, have become an essential medium for information sharing and communication during times of crisis \[[1](#bib.bib1 ""), [2](#bib.bib2 "")\]. Particularly during disasters, such as wildfires, earthquakes, hurricanes, tsunamis, floods, cyclones, and epidemics, social media platforms play a pivotal role in the timely dissemination of information \[[3](#bib.bib3 ""), [4](#bib.bib4 ""), [5](#bib.bib5 ""), [6](#bib.bib6 ""), [7](#bib.bib7 "")\]. These platforms are critical information sources for affected individuals and emergency responders, enabling real-time updates on evolving situations \[[8](#bib.bib8 ""), [9](#bib.bib9 ""), [10](#bib.bib10 ""), [11](#bib.bib11 "")\] and providing firsthand accounts from those directly and indirectly impacted \[[1](#bib.bib1 ""), [2](#bib.bib2 "")\]. The enormous amount of user-generated content on social media platforms acts as a rich source of historical as well as real-time data. However, the volume \[[12](#bib.bib12 "")\] and textual complexity of crisis-related social media texts give rise to multiple challenges for effective analysis and understanding. The volume necessitates automated analysis as the number of conversations increases exponentially during a crisis, and the textual complexity involves dealing with informally written texts with a significant presence of acronyms, misspellings, hashtags, mentions, etc.

Transformer-based \[[13](#bib.bib13 "")\] domain-specific pre-trained language models have helped produce state-of-the-art results for numerous NLP tasks in various areas such as biomedical research \[[14](#bib.bib14 "")\], scientific literature analysis \[[15](#bib.bib15 "")\], clinical text analysis \[[16](#bib.bib16 "")\] and financial text analysis \[[17](#bib.bib17 "")\]. Trained on massive amounts of domain-specific texts, these models produce contextual text representations within their respective domains. Despite the broad array of domains in which pre-trained models have been employed, a notable gap exists, i.e., the absence of pre-trained models explicitly tailored for crisis-related social media texts. Furthermore, pre-trained language models do not produce semantically rich sentence embeddings11 1 Semantically rich sentence embeddings position semantically similar sentences close together in the vector space., critical for tasks like semantic search and clustering \[[18](#bib.bib18 "")\]. Currently, the generation of semantically meaningful sentence embeddings, regardless of the domain, relies on general-purpose sentence embedding models (sentence encoders) \[[18](#bib.bib18 ""), [19](#bib.bib19 "")\]. These models utilize pre-trained models that have been trained on corpora comprising texts from broad and general domains. Studies have consistently found BERT \[[20](#bib.bib20 "")\] and its variants to be effective in various crisis-related tasks, including informativeness and humanitarian classification \[[21](#bib.bib21 "")\], health and figurative mentions identification \[[22](#bib.bib22 "")\], emotion classification \[[23](#bib.bib23 "")\], COVID-19 data analysis \[[24](#bib.bib24 ""), [7](#bib.bib7 "")\], and more. Hence, there exists a necessity to investigate the efficacy of utilizing domain-specific pre-trained language models and sentence encoders for processing crisis-related social media texts.

To address the above-discussed gaps in the crisis informatics literature, this study proposes CrisisTransformers, an ensemble of pre-trained language models and sentence encoders trained on hundreds of millions of crisis-related tweets from over 30 different crisis events, including the COVID-19 pandemic. CrisisTransformers’ embeddings can be utilized in various tasks, including text classification \[[25](#bib.bib25 ""), [26](#bib.bib26 ""), [27](#bib.bib27 ""), [28](#bib.bib28 "")\], semantic search \[[29](#bib.bib29 "")\], clustering \[[25](#bib.bib25 ""), [30](#bib.bib30 ""), [2](#bib.bib2 "")\], and topic modelling \[[31](#bib.bib31 "")\]. Advancements in these applications contribute to a more comprehensive understanding of crisis-related social media texts, thereby aiding decision-making processes and facilitating targeted interventions and communication strategies during times of crisis \[[1](#bib.bib1 "")\].

This study contributes the following to the crisis informatics literature:

*   1.

    We provide the first set of experiments relative to domain-specific pre-training to address the following research questions:

    *   (a)

        How does the choice of model initialization impact pre-training in terms of loss convergence?

    *   (b)

        With BERTweet \[[32](#bib.bib32 "")\] and other strong baselines in place, can yet another domain-specific pre-trained model demonstrate superior performance in crisis-related social media text classification?

    *   (c)

        To what extent do domain-specific pre-trained models help generate sentence embeddings with semantic richness, in comparison to current pre-trained models and sentence encoders?
*   2.

```
We introduce CrisisTransformers, the first pre-trained language models and sentence encoders designed for processing crisis-related social media texts. The pre-training of CrisisTransformers was done on 6 NVIDIA A100 GPUs over a period of 2 months.
```
*   3.

```
Our pre-trained models outperform existing models across all 18 crisis-related datasets in classification tasks, and our best-performing sentence encoder improves the current state-of-the-art by 17.43% in sentence encoding tasks. Results confirm that CrisisTransformers can capture distinct linguistic nuances, informal language structures, and unique contextual cues present in crisis contexts.
```
*   4.

```
We publicly release CrisisTransformers. The released models can be used with the Transformers \[[33](#bib.bib33 "")\] library. We anticipate that these models will serve as a robust baseline for tasks involving the analysis of crisis-related social media texts.
```
The rest of the paper is organized as follows: Section [2](#S2 "2 Related Work ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") discusses related work, Section [3](#S3 "3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") details the materials and methods used in designing CrisisTransformers, Section [4](#S4 "4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") presents evaluation results and discussions, and Section [5](#S5 "5 Conclusion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") concludes the paper.

## 2 Related Work

The current landscape of crisis informatics heavily relies on Transformer-based models (BERT-family \[[20](#bib.bib20 "")\]) trained on general domain texts (we discuss the relevant literature on BERT-family later in this section). For instance, \[[21](#bib.bib21 "")\] conducted classification tasks on 8 human-annotated crisis datasets and reported BERT and RoBERTa as the best classifiers. \[[22](#bib.bib22 "")\] demonstrated that BERT performs better in the correct classification of health and figurative mentions on Twitter. Similarly, the BERT family has also been employed in the identification and classification of transportation disaster tweets \[[34](#bib.bib34 "")\], classifying informative tweets \[[35](#bib.bib35 "")\], identifying location mentions on disaster tweets \[[36](#bib.bib36 "")\], and emotion classification in crisis-related tweets \[[23](#bib.bib23 "")\]. Likewise, the BERT family has also been extensively utilized in COVID-19 data to create state-of-the-art classifiers for vaccine-related stance \[[37](#bib.bib37 ""), [38](#bib.bib38 ""), [39](#bib.bib39 "")\], inferring the origin locations of tweets \[[40](#bib.bib40 "")\], distinguishing tweets that self-report potential cases \[[24](#bib.bib24 "")\]. Furthermore, the state-of-the-art sentence embeddings from Sentence-Transformers are based on the BERT family.

Generally, transformer-based models fall into three main categories: encoder-decoder models, decoder-only models, and encoder-only models. Auto-regressive models, such as GPT-like \[[41](#bib.bib41 "")\] models (CTRL, GPT, GPT-2, Transformer XL), utilize only the decoder component of the Transformer architecture. They focus on predicting the subsequent word in a sentence, making them optimal for text generation tasks. In such models, the attention mechanism operates such that it can only access preceding words, making them autoregressive in nature. On the other hand, BART/T5-like \[[42](#bib.bib42 ""), [43](#bib.bib43 "")\] models (BART, mBART, Marian, T5), known as sequence-to-sequence models \[[44](#bib.bib44 "")\], use both the encoder and decoder components of the Transformer architecture. These models are best suited for generating new sentences based on provided input sequences, such as in text summarization, translation, and question answering tasks. In such models, the attention mechanism in the encoder accesses all words in the input sequence, while in the decoder, it can only access preceding words. Lastly, there are BERT-like models \[[20](#bib.bib20 "")\] (BERT, RoBERTa, ALBERT, ELECTRA), which are auto-encoding models that use only the encoder component of the Transformer architecture. These models are ideal for tasks that require the entire input sequence to make decisions, such as text classification and named-entity recognition. In these models, the attention mechanism accesses all words in the input sequence, a feature commonly referred to as bi-directional attention. This characteristic makes encoder models well-suited for tasks requiring contextual embeddings. Next, we review the state-of-the-art encoder-only Transformer models, which are the focus of this study.

BERT \[[20](#bib.bib20 "")\] has become a ubiquitous baseline in NLP tasks. BERT uses two pre-training objectives — masked language modelling (MLM) and next sentence prediction (NSP). The MLM objective involves randomly masking specific tokens of an input sentence and training the model to predict the original masked tokens based on the context (surrounding words). Through this objective, BERT learns relationships between words and captures rich contextualized representations. Since the introduction of BERT, MLM has become a standard pre-training objective for many transformer-based models. Various improvements in training approaches and variants of MLM have been explored in subsequent research. RoBERTa, proposed in \[[45](#bib.bib45 "")\], outperformed BERT in various downstream tasks with some changes in the pre-training process: large batch size, longer training, more training data, and removal of the NSP objective. In \[[46](#bib.bib46 "")\], ALBERT was introduced, which offered competitive results with reduced parameters through factorized embedding parameterization and cross-layer parameter sharing. MPNet was introduced in \[[47](#bib.bib47 "")\] combining MLM and permuted language modelling (PLM). In PLM \[[48](#bib.bib48 "")\], a sequence is randomly permuted, and the model autoregressively predicts the tokens. In \[[49](#bib.bib49 "")\], XLM-RoBERTa was trained to confirm the usefulness of pre-training multilingual language models on large-scale data for cross-lingual transfer tasks. ELECTRA, introduced in \[[50](#bib.bib50 "")\], proposed a pre-training objective where two models (generator and discriminator) are involved — the generator replaces tokens in a sequence, and the discriminator predicts which tokens are originals and which are the ones replaced by the generator. The above-discussed models were pre-trained on datasets such as Wikipedia, BooksCorpus, OpenWebText, CC-News, etc., which contain general domain texts. Researchers have also introduced domain-specific pre-trained models; we discuss some of those models next.

BERTweet \[[32](#bib.bib32 "")\] is a transformer-based model specifically designed for processing Twitter data and other social media texts. It leverages the BERT model configuration and incorporates RoBERTa’s pre-training approach. During pre-training, it was exposed to a massive corpus containing 16 billion word tokens. BioBERT, which was introduced in \[[14](#bib.bib14 "")\], was pre-trained on biomedical texts, including PubMed abstracts (PubMed) and PubMed Central full-text articles (PMC), using the same architecture as BERT. Similarly, SciBERT \[[15](#bib.bib15 "")\] also shared the architecture of BERT but was pre-trained on a random sample of over 1 million papers. Its pre-training corpus consisted of 18% computer science and 82% biomedical domain full-text papers. Additionally, a variant of the BERT architecture called ClinicalBERT \[[16](#bib.bib16 "")\] was developed by pre-training on electronic health records. This specific pre-training made ClinicalBERT suitable for processing clinical text and medical data. BERT’s application has also been extended to the finance domain. FinBERT, introduced in \[[17](#bib.bib17 "")\], is a pre-trained model trained on an extensive financial communication corpus containing 4.9 billion tokens.

When the pre-trained models utilize either the embeddings of the CLS token or the mean-pooling of all tokens to generate sentence embeddings and subsequently undergo fine-tuning, they produce state-of-the-art results in text classification/regression tasks. However, previous research shows that such sentence embeddings lack semanticity and are actually worse than averaging GloVe embeddings \[[18](#bib.bib18 "")\]. For effective semantic search and clustering tasks, it is critical to have semantically meaningful embeddings that position sentences in a vector space, such that semantically similar sentences are located closely together. Generating such sentence embeddings is an extensively researched area, and various methods have been proposed, which we discuss next.

In \[[51](#bib.bib51 "")\], an encoder-decoder was trained to reconstruct the surrounding sentences of an encoded sequence so that the sentences that share semantic properties are mapped to similar vector representations. In \[[52](#bib.bib52 "")\], a siamese BiLSTM network was trained with max-pooling on the Stanford Natural Language Inference (SNLI) dataset which outperformed previous unsupervised methods \[[51](#bib.bib51 ""), [53](#bib.bib53 "")\]. In \[[54](#bib.bib54 "")\], a transformer network was trained and unsupervised learning was extended with training on the SNLI dataset. Additionally, \[[55](#bib.bib55 "")\] presented an unsupervised learning approach to sentence-level semantic similarity based on conversational data. Until this period, the sentence encoding approaches involved training the respective networks from scratch. After the introduction of BERT in 2018, replacing the unsupervised training part of designing sentence encoders became possible. In \[[18](#bib.bib18 "")\], BERT was finetuned through siamese and triplet networks on SNLI and Multi-Genre natural language inference (MultiNLI) datasets, with softmax classifier over “contradiction", “entailment", and “neutral" labels. Similarly, \[[19](#bib.bib19 "")\] proposed a contrastive approach (SimCSE) to finetune pre-trained models with natural language inference datasets using “contradiction" pairs as hard negatives. Following \[[19](#bib.bib19 "")\], \[[18](#bib.bib18 "")\] fine-tuned multiple pre-trained models using the contrastive training objective on over 1 billion sentence pairs and publicly released all their models as Sentence-Transformers.

The substantial computational resources large language models require pose challenges for real-time processing, especially in contexts like analyzing social media posts where rapidity is critical. Models with smaller footprints, typically comprising millions rather than billions of parameters, emerge as promising alternatives \[[56](#bib.bib56 "")\] as they offer viable solutions for scenarios where computational resources are limited. Furthermore, the closed-source nature of some large models, which are accessible only through APIs, introduces obstacles in terms of transparency and adaptability to specific research or application requirements. Smaller models not only address computational constraints but also promote transparency and flexibility in model usage. Therefore, in this study, we consider the base architectures of MPNet, BERTweet, BERT, RoBERTa, XLM-RoBERTa, ALBERT, and ELECTRA, as baselines for the classification task, and Sentence-Transformers and SimCSE as baselines for the sentence encoding task. These baselines (except ALBERT) share similar parameter counts with the models proposed in this study.

## 3 Materials and methods

![Refer to caption](2309.05494v3/method-overview.png)

Figure 1: A high-level methodological view for developing pre-trained models and sentence encoders.

In this section, we detail the curation process of the pre-training corpus (Section [3.1](#S3.SS1 "3.1 The crisis corpus ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")) and the development of pre-trained models (Section [3.2](#S3.SS2 "3.2 Unsupervised pre-training ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")) and sentence encoders (Section [3.4](#S3.SS4 "3.4 Enriching sentence encoding ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")). A high-level methodological view is preovided in Figure [1](#S3.F1 "Figure 1 ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

### 3.1 The crisis corpus

A large-scale social media text corpus was curated for unsupervised pre-training, with Twitter serving as the primary data source. Our main objective was to create a comprehensive corpus containing texts discussing a diverse range of crisis events, such as disease outbreaks, natural disasters, terrorist attacks, conflicts, and other critical incidents. In general, as illustrated in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 The crisis corpus ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"), the corpus underwent curation across three distinct stages: (i) consideration of an in-house dataset consisting of billions of tweets, (ii) hydration of Twitter identifiers collected from various data repositories, and (iii) utilization of Twitter’s full-archive endpoint to search historical tweets. We maintained an in-house billion-scale COVID-19 tweets dataset from the onset of the COVID-19 outbreak until March 2023. The initial version of the dataset, COV19Tweets \[[57](#bib.bib57 "")\], comprised more than 2.2 billion tweets. Subsequently, we created the second version, BillionCOV \[[58](#bib.bib58 "")\], by filtering out unavailable tweets, resulting in over 1.4 billion tweets. For this study, we considered all the tweets present in BillionCOV, excluding retweets. Although BillionCOV contains COVID-19-related tweets, the COVID-19 discourse was not solely limited to discussions about the virus. Numerous other events unfolded worldwide along with the pandemic, including economic crises, natural disasters, humanitarian crises, social unrest, mental health concerns, and social issues.

![Refer to caption](2309.05494v3/pretraining_corpus.png)

Figure 2: The pre-training corpus curation process.

Next, we collected tweet identifiers from multiple data repositories such as CrisisNLP \[[59](#bib.bib59 "")\] and DocNow Catalog22 2 https://catalog.docnow.io/. Tweets collected from these sources needed to be hydrated to re-create the datasets locally, as Twitter’s data re-distribution policy restricts sharing data other than tweet identifiers. At this stage, the corpus had texts related to more than 30 crisis events that occurred after 2014. Furthermore, to fill the temporal gap in the corpus, we utilized Twitter’s full-archive endpoint to search for historical tweets created between 2006 and 2013. We applied lang:en condition and used the following keywords (along with their #hashtag and plural variants): crisis, disaster, earthquake, typhoon, volcano, flood, landslide, hurricane, tornado, cyclone, wildfire, famine, drought, tsunami, avalanche, epidemic, hailstorm, storm, protest, virus, war, and riot. Below are some of the crisis events covered in the corpus.

*   1.

```
Disease Outbreaks: COVID-19, Middle East Respiratory Syndrome, Ebola Virus Outbreak.
```
*   2.

```
Natural Disasters: Hurricanes Harvey, Irma, Florence, Dorian, Odile, Cyclone PAM, Typhoon Hagupit, California Earthquake, Pakistan Earthquake, Chile Earthquake, Nepal Earthquake, Pakistan Floods, India Floods, Iceland Volcano, Tropical Storm Imelda.
```
*   3.

```
Terrorist Attacks: Paris Attacks, Stockholm Attack, Catalonia Attacks, Peshawar School Attack.
```
*   4.

```
Protests and Activism: #J20, Tyendinaga protests.
```
*   5.

```
Shootings: Dallas Police Shooting, Las Vegas Shooting.
```
*   6.

```
Landslides: Landslides worldwide.
```
*   7.

```
Conflicts: Gaza, Palestine Conflict.
```
*   8.

```
Civil War: Fall of Aleppo.
```
*   9.

```
Missing Flight: Flight MH370.
```
#### 3.1.1 Text pre-processing

Each tweet in the corpus was pre-processed as follows: We (i) replaced URLs with “HTTPURL" token, (ii) replaced mentions (usernames) with “@USER" token (iii) decoded HTML entities to their original form (e.g., &amp; to &), (iv) removed newline characters and replaced multiple consecutive whitespaces with a single space, (v) fixed text encoding to correct various encoding issues and improve consistency in text representation, and (vi) replaced emojis with their textual representation, as their descriptive text counterparts are meaningful. We considered only the tweets with more than ten tokens. Refer to Table [1](#S3.T1 "Table 1 ‣ 3.1.1 Text pre-processing ‣ 3.1 The crisis corpus ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") for the descriptive statistics of the corpus.

|               | count        |
| ------------- | ------------ |
| tokens        | 15 billion   |
| sentences     | 997 million  |
| unique tokens | 36.7 million |

Table 1: Descriptive statistics of the preprocessed corpus. Note: A tweet can have multiple sentences.

### 3.2 Unsupervised pre-training

#### 3.2.1 Architecture and pre-training procedure

CrisisTransformers use the same architecture as B​E​R​TB​A​S​EBERT\_{BASE}. In contrast to existing studies \[[14](#bib.bib14 ""), [17](#bib.bib17 ""), [16](#bib.bib16 ""), [32](#bib.bib32 "")\], we adopted a more versatile approach to selecting a pre-training procedure for our models. Instead of starting with a specific pre-training procedure, we experimented with multiple state-of-the-art models, namely MPNet, BERTweet, BERT, RoBERTa, XLM-RoBERTa, ALBERT, and ELECTRA, on classification tasks using 18 crisis-related labelled datasets (detailed in Section [3.3.1](#S3.SS3.SSS1 "3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")). We observed RoBERTa and BERTweet emerging as the top-performing models on average. Therefore, we selected RoBERTa’s pre-training procedure for training CrisisTransformers. Due to the extensive adoption of BERT and RoBERTa, we do not provide an in-depth explanation of the architecture in this paper; for more comprehensive insights, please refer to \[[20](#bib.bib20 ""), [45](#bib.bib45 "")\]. The configurations of the proposed models are provided in [B](#A2 "Appendix B Configurations of CrisisTransformers ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

#### 3.2.2 Pre-training data

We trained a Byte-Level BPE (Byte-Pair Encoding) tokenizer using the Tokenizers library \[[33](#bib.bib33 "")\] for our domain, utilizing the pre-processed crisis corpus discussed in Section [3.1](#S3.SS1 "3.1 The crisis corpus ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"). Acknowledging the nuanced nature of social media texts (the crisis corpus had 36 million unique tokens), we also set the vocabulary size to 64k \[[32](#bib.bib32 "")\]. Next, we used the trained tokenizer to tokenize the crisis corpus, thus generating sequence blocks of size 128128, on which we trained the CrisisTransformers. Table [2](#S3.T2 "Table 2 ‣ 3.2.2 Pre-training data ‣ 3.2 Unsupervised pre-training ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") provides a comparative analysis of token counts in the vocabularies of established pre-trained models and CrisisTransformers. Among the existing models, RoBERTa and BERTweet share the highest similarity in vocabulary with CrisisTransformers.

| model       | intersection | unique  |
| ----------- | ------------ | ------- |
| RoBERTa     | 37,338       | 12,927  |
| BERTweet    | 15,121       | 48,880  |
| BERT        | 7,905        | 21,091  |
| XLM-RoBERTa | 6,431        | 243,571 |
| MPNet       | 5,754        | 24,773  |
| ELECTRA     | 5,749        | 24,773  |
| ALBERT      | 4,394        | 25,606  |

Table 2: Vocabulary similarity between existing pre-trained models and CrisisTransformers. Note: intersection denotes the number of tokens shared between the existing models and CrisisTransformers, while unique indicates the tokens exclusive to the vocabulary of the existing models.

#### 3.2.3 Optimization

![Refer to caption](2309.05494v3/pre_training.png)

Figure 3: Pre-training of CrisisTransformers. Note: “\*" represents different checkpoints, which will be discussed later in Section [4](#S4 "4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

We pre-trained three models (as shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.2.3 Optimization ‣ 3.2 Unsupervised pre-training ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")), utilizing 6 NVIDIA A100 GPUs (each with 80GB of memory). The training configurations for these models were as follows: CT-M1 (or CrisisTransformer-Model1) was pre-trained from scratch with randomly initialized weights; CT-M2 had weights initialized with pre-trained RoBERTa’s weights; and CT-M3 had weights initialized with pre-trained BERTweet’s weights. CT-M1 was trained for 40 epochs, while CT-M2 and CT-M3 were trained for 20 epochs each. We used the Transformers library \[[33](#bib.bib33 "")\] to implement these models.

For optimization, we employed the AdamW optimizer with a peak learning rate set to 0.0004. To utilize the available GPU memory efficiently, we used a batch size of 8k with gradient accumulation steps of 16. Additionally, we set 5% of the total training steps for warming up the learning rate. All three models finished training in two months.

### 3.3 Fine-tuning

For fine-tuning the baselines and CrisisTransformers for text classification, as outlined in \[[32](#bib.bib32 "")\], we added a linear prediction layer to the pooled output. We implemented mean pooling over the token embeddings of an input sequence while considering the attention mask. Both baselines and CrisisTransformers were fine-tuned under identical conditions. Each model was fine-tuned across 18 labelled crisis-related datasets for a maximum of 30 epochs, a batch size of 32, a learning rate of 1e-5, and AdamW as an optimizer. During each epoch, classification performance was assessed on a validation set. Early stopping was configured with a patience of 5 and a threshold of 0.0001. The final checkpoint was then used for evaluation on a test set. The fine-tuning procedure was repeated 5 times per model and dataset, with average performance scores being reported at a 95% confidence interval.

#### 3.3.1 Labelled crisis-related datasets

Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") lists the datasets we considered to evaluate both baselines and CrisisTransformers. Evaluating the performance across such diverse datasets from the crisis informatics domain was essential to test the robustness of our proposed models. CrisisBench \[[21](#bib.bib21 "")\] provided the train/validation/test splits for datasets D-01 through D-06. For the remaining datasets, we implemented stratified sampling, allocating 70% for training, 10% for validation, and 20% for testing, using scikit-learn’s train-test split33 3 https://scikit-learn.org with a random state of 42.

| Id   | Dataset                                         | Description – (# of Classes)                                      | Samples |
| ---- | ----------------------------------------------- | ----------------------------------------------------------------- | ------- |
| D-01 | CrisisMMD \[[60](#bib.bib60 "")\]               | Tweets from 7 disaster events from 2017 – (6)                     | 10,070  |
| D-02 | CrisisLex \[[61](#bib.bib61 "")\]               | Tweets from 26 different crisis events in 2012–13 – (6)           | 10,041  |
| D-03 | AIDR \[[62](#bib.bib62 "")\]                    | Tweets collected by AIDR system – (9)                             | 5,169   |
| D-04 | ISCRAM2013 \[[27](#bib.bib27 "")\]              | Tweets from 2 different events in 2011 – (5)                      | 810     |
| D-05 | SWDM2013 \[[63](#bib.bib63 "")\]                | Tweets related to Joplin tornado and Hurricane Sandy – (4)        | 346     |
| D-06 | CrisisNLP \[[59](#bib.bib59 "")\]               | Tweets from 19 different disaster events in 2013–15 – (8)         | 10,214  |
| D-07 | Poddar et al. (2022) \[[37](#bib.bib37 "")\]    | Tweets related to stance towards COVID-19 vaccines – (3)          | 3,300   |
| D-08 | SAD Stressor \[[64](#bib.bib64 "")\]            | SMS-like sentences mentioning everyday stressors                  | 6,850   |
| D-09 | SAD Stress \[[64](#bib.bib64 "")\]              | Stress and non-stress SMS-like sentences – (2)                    | 6,850   |
| D-10 | SAD COVID \[[64](#bib.bib64 "")\]               | COVID and non-COVID SMS-like sentences – (2)                      | 6,850   |
| D-11 | LocBERT \[[40](#bib.bib40 "")\]                 | COVID-19 tweets with origin and non-origin locations – (2)        | 2,800   |
| D-12 | HMC (a) \[[22](#bib.bib22 "")\]                 | Figurative versus literal health reports on Twitter – (3)         | 13,017  |
| D-13 | Cotfas et al. (2021) \[[38](#bib.bib38 "")\]    | Twitter opinions regarding COVID-19 vaccination – (3)             | 2,393   |
| D-14 | HMC (b) \[[22](#bib.bib22 "")\]                 | Disease mentions on tweets – (10)                                 | 13,017  |
| D-15 | PHM \[[65](#bib.bib65 "")\]                     | Health mentions in social media – (4)                             | 4,419   |
| D-16 | Klein et al. (2021) (a) \[[24](#bib.bib24 "")\] | Tweets about actual and potential COVID-19 patients – (3)         | 4,266   |
| D-17 | Klein et al. (2021) (b) \[[24](#bib.bib24 "")\] | Tweets about groups of potential COVID-19 positive contacts – (8) | 4,266   |
| D-18 | ANTiVax \[[39](#bib.bib39 "")\]                 | Tweets on vaccine misinformation – (2)                            | 11,518  |

Table 3: Labelled crisis datasets considered in this study for evaluating the performance of baselines and CrisisTransformers.

### 3.4 Enriching sentence encoding

By default, CrisisTransformers do not produce semantically rich embeddings, even though they were trained on a domain-specific corpus. Such pre-trained models require additional fine-tuning to learn to represent semantically similar sentences closer together within the vector space. These enhanced embeddings, capable of capturing semantic meanings, can then be effectively compared using cosine similarity. Their significance becomes particularly evident in tasks involving semantic search and clustering.

Our sentence encoders (CT-M1-\*-SE, CT-M2-\*-SE, and CT-M3-\*-SE — where, “SE" stands for Sentence Encoder) are built upon the recent success of utilizing siamese and triplet networks on sentence pairs \[[52](#bib.bib52 "")\] with pre-trained transformers \[[18](#bib.bib18 "")\] while leveraging the idea that adding corresponding contradicting pairs as “hard negatives" alongside in-batch negatives further improves the performance \[[19](#bib.bib19 "")\]. Expanding upon the method introduced in \[[19](#bib.bib19 "")\], we adapt it to utilize domain-specific pre-trained models instead of the existing general pre-trained models like BERT and RoBERTa. We used the following contrastive learning objectives to train our sentence encoders:

*   1.

```
Multiple Negative Ranking (MNR): This loss incorporates the (anchor, positive) pairs. Given a batch of pairs (a1,a1+),(a2,a2+),…,(an,an+)(a\_{1},a^{+}\_{1}),(a\_{2},a^{+}\_{2}),\\ldots,(a\_{n},a^{+}\_{n}) where (ai,ai+)(a\_{i},a^{+}\_{i}) are positive pairs and (ai,aj+)(a\_{i},a^{+}\_{j}) for i≠ji\\neq j are considered negative pairs. The training objective for (ai,ai+)(a\_{i},a^{+}\_{i}) with mini-batch NN is:

li\=−log⁡(esimilarity​(ri,ri+)/τ∑j\=1Nesimilarity​(ri,rj+)/τ)l\_{i}=-\\log\\left(\\frac{e^{\\text{similarity}(r\_{i},r^{+}\_{i})/\\tau}}{\\sum\_{j=1}^{N}e^{\\text{similarity}(r\_{i},r^{+}\_{j})/\\tau}}\\right)

(1)

where, rir\_{i} and ri+r^{+}\_{i} are embeddings of aia\_{i} and ai+a^{+}\_{i} generated by our CrisisTransformers, similarity(ri,ri+)(r\_{i},r^{+}\_{i}) is cosine similarity, and τ\\tau is temperature hyperparameter.
```
*   2.

```
MNR with hard negatives: This loss incorporates the (anchor, positive, hard negative) pairs, i.e., (an,an+,an−)(a\_{n},a^{+}\_{n},a^{-}\_{n}). The training objective in Equation [1](#S3.E1 "In item 1 ‣ 3.4 Enriching sentence encoding ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") can be modified to:

li\=−log⁡(esimilarity​(ri,ri+)/τ∑j\=1N(esimilarity​(ri,rj+)/τ+esimilarity​(ri,rj−)/τ))l\_{i}=-\\log\\left(\\frac{e^{\\text{similarity}(r\_{i},r^{+}\_{i})/\\tau}}{\\sum\_{j=1}^{N}(e^{\\text{similarity}(r\_{i},r^{+}\_{j})/\\tau}+e^{\\text{similarity}(r\_{i},r^{-}\_{j})/\\tau})}\\right)

(2)
```
![Refer to caption](2309.05494v3/se_training.png)

Figure 4: Training of our sentence encoders.

The MNR loss maximizes the similarity between an anchor sentence and its positive sentence while considering all other positives in a batch as negatives. In MNR with hard negatives, the similarity between an anchor sentence and its positive sentence is maximized while using its hard negative and all other positive sentences in the same batch as negatives. We include the MNR training objective in the experiments for comparison purposes, even though MNR with hard negatives has been shown to outperform it \[[19](#bib.bib19 "")\]. We train our sentence encoders (as shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.4 Enriching sentence encoding ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")) with these two objectives on (Question, Answer) pairs from GooAQ \[[66](#bib.bib66 "")\], (anchor, positive, hard negative) triplets from QQP44 4 https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs \[[67](#bib.bib67 "")\] and (anchor, entailment, contradiction) triplets from AllNLI \[[67](#bib.bib67 ""), [68](#bib.bib68 ""), [69](#bib.bib69 "")\] with a large batch size of 512 for a maximum of 20 epochs. We utilize a learning rate of 2e-05 and allocate 1% of the total training steps for warm-up.

We implemented mean pooling over the token embeddings with attention to generate sentence embeddings.

### 3.5 Evaluation setup

#### 3.5.1 Classification task

In line with prior research \[[32](#bib.bib32 ""), [15](#bib.bib15 "")\], we evaluate baselines and CrisisTransformers for the classification task using F1-macro, which considers the precision and recall of each class and provides an overall evaluation of the models’ classification performance. For each dataset, we compute the F1-macro score as follows:

Pclassi\\displaystyle P\_{\\text{class}\_{i}}

\=T​PclassiT​Pclassi+F​Pclassi\\displaystyle=\\frac{TP\_{\\text{class}\_{i}}}{TP\_{\\text{class}\_{i}}+FP\_{\\text{class}\_{i}}}

Rclassi\\displaystyle R\_{\\text{class}\_{i}}

\=T​PclassiT​Pclassi+F​Nclassi\\displaystyle=\\frac{TP\_{\\text{class}\_{i}}}{TP\_{\\text{class}\_{i}}+FN\_{\\text{class}\_{i}}}

F​1classi\\displaystyle F1\_{\\text{class}\_{i}}

\=2⋅Pclassi⋅RclassiPclassi+Rclassi\\displaystyle=\\frac{2\\cdot P\_{\\text{class}\_{i}}\\cdot R\_{\\text{class}\_{i}}}{P\_{\\text{class}\_{i}}+R\_{\\text{class}\_{i}}}

F1-macro

\=1Nclasses​∑i\=1NclassesF​1classi\\displaystyle=\\frac{1}{N\_{\\text{classes}}}\\sum\_{i=1}^{N\_{\\text{classes}}}F1\_{\\text{class}\_{i}}

where, T​PclassiTP\_{\\text{class}\_{i}} is the number of true positive predictions for class ii, F​PclassiFP\_{\\text{class}\_{i}} is the number of false positive predictions for class ii, F​NclassiFN\_{\\text{class}\_{i}} is the number of false negative predictions for class ii, and Nc​l​a​s​s​e​sN\_{classes} is the total number of classes in the dataset.

#### 3.5.2 Sentence encoding task

There is an absence of standard benchmark datasets in the crisis informatics domain to assess the semantic quality of the generated embeddings. In agreement with \[[18](#bib.bib18 "")\] and \[[19](#bib.bib19 "")\] that the primary objective of the embeddings is to capture and represent semantic relationships in text data, we designed an alternative task. Our evaluation method involved calculating the weighted average cosine similarity among encoded tweets within individual classes in a labelled dataset, thereby measuring the semantic consistency of tweets belonging to the same class. This approach allowed us to capture the complexities and semantics of crisis-related content, resulting in a more insightful evaluation of the sentence embeddings.

Task definition: Let NN represent the total number of crisis-related tweets in a dataset and KK denote the number of unique classes within the dataset. Let EE be a matrix of sentence embeddings, where each row 𝐞i\\mathbf{e}\_{i} corresponds to the normalized embedding of the ii-th tweet. Additionally, let yy be a vector containing the class labels associated with each tweet.

For each unique class ckc\_{k}, the class weight wkw\_{k} is computed as the inverse of the count of tweets belonging to that class:

wk\=1count​(ck)w\_{k}=\\frac{1}{\\text{count}(c\_{k})}

These class weights are then normalized to obtain w^k\\hat{w}\_{k}:

w^k\=wk∑i\=1Kwi\\hat{w}\_{k}=\\frac{w\_{k}}{\\sum\_{i=1}^{K}w\_{i}}

For each unique class ckc\_{k}, the intra-class cosine similarity dkd\_{k} is computed. For each tweet 𝐞i\\mathbf{e}\_{i} within class ckc\_{k}, the average cosine similarity to other tweets within the same class is determined:

dk\=1|{i:yi\=ck}|∑i:yi\=cksimilarity(𝐞i,𝐞j)d\_{k}=\\frac{1}{|\\{i:y\_{i}=c\_{k}\\}|}\\sum\_{i:y\_{i}=c\_{k}}\\text{similarity}(\\mathbf{e}\_{i},\\mathbf{e}\_{j})

Here, similarity​(𝐞i,𝐞j)\\text{similarity}(\\mathbf{e}\_{i},\\mathbf{e}\_{j}) calculates the cosine similarity between tweet embeddings 𝐞i\\mathbf{e}\_{i} and 𝐞j\\mathbf{e}\_{j}, where 𝐞j\\mathbf{e}\_{j} is a tweet within the same class as 𝐞i\\mathbf{e}\_{i}.

The weighted average distance DavgD\_{\\text{avg}} is computed across all classes, considering their respective normalized class weights w^k\\hat{w}\_{k}:

Davg\=∑k\=1Kw^k⋅dkD\_{\\text{avg}}=\\sum\_{k=1}^{K}\\hat{w}\_{k}\\cdot d\_{k}

DavgD\_{\\text{avg}} quantifies the average within-class semantic similarity of crisis-related tweets while accounting for the distribution of class weights.

The cosine similarity between sentence embeddings reflects how semantically similar or related the sentences are. If the embeddings are better at capturing the semantic content of crisis-related tweets within each class, the cosine similarity values within a class would be high. A higher cosine similarity within each class indicates that the embeddings effectively represent tweets that share similar content or context related to a specific crisis-related class. In summary, the higher the value of Da​v​gD\_{avg}, the better the performance of a sentence encoder. We considered all the datasets listed in Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") for this task.

## 4 Results and Discussion

### 4.1 Checkpoints and convergence

After the pre-training, we were interested in multiple checkpoints of CrisisTransformers: CT-M1-\*, CT-M2-\*, and CT-M3-\*. CT-M1 was built from scratch and had two variants, CT-M1-BestLoss, representing the model at its lowest loss during training, and CT-M1-Complete, representing the model after 40 epochs. On the other hand, CT-M2 and CT-M3 were initialized using weights from pre-trained RoBERTa and BERTweet, respectively, and were trained up to 20 epochs each. CT-M2-OneLook represents the model after 1 epoch, while CT-M2-BestLoss and CT-M2-Complete represent the model at its lowest loss and the model after 20 epochs, respectively. The same setup was applied to CT-M3 models. In total, CrisisTransformers has 8 variants based on different checkpoints of CT-M1, CT-M2, and CT-M3 models.

Figure [5](#S4.F5 "Figure 5 ‣ 4.1 Checkpoints and convergence ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") visualizes the validation loss versus epoch for each model. The graph provides insights into the impact of different initialization on the models’ convergence. The loss patterns of the three models revealed distinct behaviours. CT-M1 demonstrated a gradual and consistent reduction in loss throughout the training period, suggesting steady convergence. CT-M2, on the other hand, exhibited a sharp initial drop in the loss within a few training steps, indicating rapid convergence and a smoother decline. Similarly, CT-M3 also displayed a significant initial loss drop. While CT-M3 initially shared a sharp loss drop with CT-M2, its convergence pattern aligned more with CT-M1 in the later epochs. The final loss of CT-M3 ultimately converged closer to that of CT-M1. All models seemed to plateau in their loss during the later epochs, indicating a potential convergence point. These loss patterns highlight the influence of different initializations on the time and trajectory of loss convergence; the pre-trained models seem to leverage their existing knowledge for a more efficient initial convergence than the model whose weights were randomly initialized.

Figure 5: Validation loss versus epoch for CrisisTransformers’ CT-M1-\*, CT-M2-\*, and CT-M3-\* checkpoints, showing the impact of different initializations. The loss for CT-M1 at Epoch 0 was 9.8419.841, and it achieved its lowest loss at the 26th epoch. For CT-M2, the loss at Epoch 0 was 2.26, and it achieved its lowest loss at the 8th epoch. Lastly, CT-M3 started with a loss of 2.856 at Epoch 0 and reached its lowest loss at the 15th epoch. The y-axis is truncated to a maximum value of 3 for clarity. Although the data extends to 9.8419.841 on the y-axis, focusing on the range up to 3 enhances the visibility of differences between the plots, which may otherwise be overshadowed by the scale.

### 4.2 Evaluations

For the classification task, we considered MPNet, BERTweet, BERT, RoBERTa, XLM-RoBERTa, ALBERT, and Electra as baselines for CrisisTransformers. As discussed in Section [3.3](#S3.SS3 "3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"), we finetuned the baselines and CrisisTransformers for the classification task across 18 different crisis-related datasets, each identified by a unique identifier (D-01 through D-18) (refer to Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")). Results from the experiments are summarized in Table [4](#S4.T4 "Table 4 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

Amongst the baselines, RoBERTa consistently outperforms other models with high F1-macro scores across several datasets. However, with the introduction of CrisisTransformers, the checkpoints of CT-M1 and CT-M2 stand out; overall, CrisisTransformers outperform the existing pre-trained models across all 18 datasets. The following models outperformed others in the respective number of datasets: CT-M1-BestLoss (in 4 datasets), CT-M1-Complete (in 3 datasets), CT-M2-OneLook (in 4 datasets), CT-M2-BestLoss (in 1 dataset), CT-M2-Complete (in 4 datasets), and CT-M3-OneLook (in 2 datasets). These results confirm the potential of CrisisTransformers for generalization and applicability in various crisis text classification tasks, which is particularly valuable for real-world applications such as disaster response, emergency communication, and crisis management.

Next, we performed sentence encoding tasks across all 18 datasets with the existing pre-trained models, CrisisTransformers, Sentence-Transformers, SimCSE, and CrisisTransformers-based sentence encoders. The results from the sentence encoding task are summarized in Tables [5](#S4.T5 "Table 5 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")–[7](#S4.T7 "Table 7 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

The pre-trained models do not yield semantically meaningful sentence embeddings out-of-the-box. Nevertheless, one of our objectives was to investigate how well domain-adapted models perform in generating semantically meaningful sentence embeddings. Results show that (refer to Table [5](#S4.T5 "Table 5 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")), within the existing pre-trained models, BERTweet emerged as a robust performer, consistently achieving competitive weighted average cosine similarity scores. However, CrisisTransformers, particularly the CT-M3 variants, invariably achieved the highest or second-highest scores regardless of the specific configuration (OneLook, BestLoss, or Complete). The performance of BERTweet and CT-M3 variants can be attributed to BERTweet’s pre-training on an extensive corpus of tweets. The results further indicate that the performance of the pre-trained RoBERTa is subpar. Consequently, the CT-M2 variants do not notably enhance performance. In contrast, the CT-M1 variants demonstrate a significant performance advantage over the CT-M2 variants. These findings suggest that further pre-training a domain-specific model on a sub-domain corpus (where “tweets" reflect “domain" and “crisis-related tweets" indicate “sub-domain") leads to improved performance in generating better sentence embeddings.

Furthermore, we trained CrisisTransformers using siamese and triplet networks with MNR and MNR with hard negatives training objectives, as discussed in Section [3.4](#S3.SS4 "3.4 Enriching sentence encoding ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"), to create sentence encoders specifically designed for crisis-related social media texts. We used GooAQ (Question, Answer) pairs for MNR and QQP (anchor, positive, hard negative) triplets for MNR with hard negatives. As baselines for our sentence encoders, we utilized Sentence-Transformers and SimCSE. We considered the ‘‘all-mpnet-base-v2" model55 5 https://huggingface.co/sentence-transformers/all-mpnet-base-v2, which is the highest-performing model in Sentence-Transformers, and the ‘‘sup-simcse-roberta-base" model66 6 https://huggingface.co/princeton-nlp/sup-simcse-roberta-base, a high-performing base architecture model for SimCSE. We used only the first 10k pairs from GooAQ and QQP for both training objectives. We explored different sample sizes and ultimately found that using 10k pairs balanced model performance and having fewer training samples. This contrasts our baselines, where Sentence-Transformers was trained on over 1 billion samples, and SimCSE was trained on 314k samples.

Table [6](#S4.T6 "Table 6 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") and Table [7](#S4.T7 "Table 7 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") summarize the performance of the baselines and our sentence encoders in terms of the weighted average cosine similarity, and Table [8](#S4.T8 "Table 8 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") reports the overall performance. Across all 18 datasets, our sentence encoders outperform both Sentence-Transformers and SimCSE. Notably, CT-M1-Complete-SE (MNR) and CT-M2-Complete-SE (hard negatives) each achieved the best performances across 4 datasets, and CT-M1-BestLoss-SE (MNR) and CT-M2-BestLoss-SE (hard negatives) each in 3 datasets. Overall, CT-M1 variants performed better in 11 datasets, CT-M2 variants in 6 datasets, and CT-M3 in 1 dataset. Considering training objectives, models trained with hard negatives achieved the highest scores across 11 datasets. CT-M1-Complete-SE (hard negatives), although trained on 10k samples, achieved an average score of 0.7140, surpassing the current state-of-the-art by 12% while outperforming Sentence-Transformers’ average score of 0.6374. These results highlight the adaptability and effectiveness of CrisisTransformers-based sentence encoders in capturing semantic similarity within sentences, particularly in crisis-related contexts. This reinforces the idea that tailoring models to specific domains, like crisis situations, can yield significant improvements over more general-purpose models in sentence encoding tasks, even when trained with less data. Among the baselines, Sentence-Transformers performed better compared to SimCSE across all datasets. In fact, our CT-M3 variants (avg. scores ranging from 0.2663 to 0.2792) outperformed SimCSE (avg. score of 0.1765). The noticeable performance advantage of Sentence-Transformers over SimCSE can be attributed to the comprehensive training of its “all-mpnet-base-v2” model, which involved training on more than 1 billion sentence pairs/triplets. This extensive training likely provided the model with a broader and richer understanding of general language nuances, thus contributing to its superior performance.

Motivated to study the effect of training samples, we re-trained CT-M1-Complete-SE (hard negatives) while increasing the training samples from 10k to 102k samples (complete QQP) and further augmented the AllNLI dataset to create a training size of 378k. After this re-training, we observed an improvement of approx. 3.56% with complete QQP and approx. 4.83% with QQP+AllNLI. Overall, our best-performing sentence encoder improved the current state-of-the-art by around 17.43%. This observation sets the stage for potential enhancements to our sentence encoder. Going forward, our future objectives include training our sentence encoders on a scale similar to Sentence-Transformers for an even more substantial improvement.

| model         |               |               |               |               |               |               |               |               |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.6559 ±0.005 | 0.7571 ±0.004 | 0.4286 ±0.036 | 0.7359 ±0.000 | 0.5706 ±0.168 | 0.7776 ±0.002 | 0.5810 ±0.014 | 0.6934 ±0.022 | 0.6758 ±0.000 |
| 0.6597 ±0.024 | 0.7569 ±0.001 | 0.5735 ±0.022 | 0.6979 ±0.024 | 0.6574 ±0.183 | 0.7871 ±0.002 | 0.5890 ±0.010 | 0.7148 ±0.002 | 0.6940 ±0.004 |
| 0.6728 ±0.026 | 0.7297 ±0.000 | 0.5956 ±0.008 | 0.6089 ±0.084 | 0.6495 ±0.078 | 0.7782 ±0.001 | 0.5197 ±0.013 | 0.7040 ±0.002 | 0.6770 ±0.000 |
| 0.6891 ±0.023 | 0.7603 ±0.002 | 0.6240 ±0.000 | 0.7650 ±0.021 | 0.7683 ±0.014 | 0.7919 ±0.000 | 0.5808 ±0.010 | 0.7171 ±0.002 | 0.7122 ±0.008 |
| 0.6419 ±0.006 | 0.7505 ±0.003 | 0.5908 ±0.020 | 0.6321 ±0.049 | 0.3849 ±0.127 | 0.7918 ±0.001 | 0.5187 ±0.015 | 0.7232 ±0.001 | 0.6702 ±0.000 |
| 0.6548 ±0.038 | 0.7281 ±0.006 | 0.5571 ±0.022 | 0.5589 ±0.070 | 0.4484 ±0.023 | 0.7648 ±0.005 | 0.4963 ±0.020 | 0.6982 ±0.016 | 0.6989 ±0.000 |
| 0.6427 ±0.008 | 0.7409 ±0.000 | 0.5766 ±0.049 | 0.5422 ±0.036 | 0.2685 ±0.045 | 0.7874 ±0.003 | 0.5699 ±0.009 | 0.7183 ±0.005 | 0.7191 ±0.010 |
| 0.6555 ±0.002 | 0.7539 ±0.002 | 0.6174 ±0.041 | 0.6730 ±0.033 | 0.8510 ±0.018 | 0.7927 ±0.004 | 0.7118 ±0.009 | 0.7121 ±0.003 | 0.7023 ±0.012 |
| 0.6567 ±0.000 | 0.7613 ±0.000 | 0.6291 ±0.003 | 0.6870 ±0.004 | 0.8160 ±0.018 | 0.7703 ±0.032 | 0.7035 ±0.009 | 0.709 ±0.008  | 0.6980 ±0.005 |
| 0.6916 ±0.032 | 0.757 ±0.000  | 0.6651 ±0.009 | 0.7744 ±0.040 | 0.8274 ±0.024 | 0.7862 ±0.000 | 0.6504 ±0.005 | 0.7250 ±0.000 | 0.7046 ±0.003 |
| 0.6646 ±0.013 | 0.7643 ±0.001 | 0.6637 ±0.007 | 0.7669 ±0.014 | 0.7874 ±0.074 | 0.7860 ±0.006 | 0.6721 ±0.003 | 0.7207 ±0.000 | 0.6969 ±0.000 |
| 0.6606 ±0.008 | 0.7656 ±0.000 | 0.6569 ±0.013 | 0.7453 ±0.019 | 0.8317 ±0.006 | 0.7796 ±0.005 | 0.6621 ±0.012 | 0.7015 ±0.014 | 0.6889 ±0.013 |
| 0.6494 ±0.002 | 0.7592 ±0.000 | 0.5383 ±0.052 | 0.7021 ±0.015 | 0.8048 ±0.034 | 0.7779 ±0.005 | 0.6587 ±0.010 | 0.7291 ±0.002 | 0.7445 ±0.005 |
| 0.6546 ±0.006 | 0.7585 ±0.000 | 0.5555 ±0.054 | 0.7165 ±0.030 | 0.5725 ±0.285 | 0.7874 ±0.000 | 0.6729 ±0.003 | 0.7173 ±0.004 | 0.7200 ±0.000 |
| 0.6547 ±0.004 | 0.7590 ±0.002 | 0.5726 ±0.053 | 0.7011 ±0.011 | 0.6806 ±0.189 | 0.7898 ±0.002 | 0.6814 ±0.004 | 0.7156 ±0.005 | 0.7061 ±0.032 |

| model         |               |               |               |               |               |               |               |               |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.9208 ±0.002 | 0.7590 ±0.000 | 0.8882 ±0.003 | 0.8119 ±0.006 | 0.9905 ±0.000 | 0.8029 ±0.000 | 0.8040 ±0.009 | 0.5260 ±0.008 | 0.9829 ±0.000 |
| 0.9358 ±0.004 | 0.7727 ±0.005 | 0.9009 ±0.002 | 0.8562 ±0.000 | 0.9933 ±0.000 | 0.8209 ±0.010 | 0.8108 ±0.016 | 0.5274 ±0.001 | 0.9830 ±0.001 |
| 0.9001 ±0.009 | 0.7230 ±0.008 | 0.8745 ±0.000 | 0.7595 ±0.000 | 0.9899 ±0.000 | 0.8106 ±0.003 | 0.7662 ±0.010 | 0.5941 ±0.018 | 0.9748 ±0.002 |
| 0.9125 ±0.002 | 0.7665 ±0.005 | 0.8904 ±0.000 | 0.8338 ±0.005 | 0.9927 ±0.000 | 0.8310 ±0.006 | 0.7998 ±0.005 | 0.6182 ±0.050 | 0.9837 ±0.000 |
| 0.9167 ±0.004 | 0.7703 ±0.003 | 0.8835 ±0.003 | 0.7975 ±0.000 | 0.9930 ±0.000 | 0.8101 ±0.004 | 0.7902 ±0.008 | 0.5280 ±0.041 | 0.9773 ±0.003 |
| 0.8759 ±0.000 | 0.7623 ±0.006 | 0.8750 ±0.001 | 0.7625 ±0.018 | 0.9917 ±0.000 | 0.8050 ±0.016 | 0.8011 ±0.011 | 0.5709 ±0.029 | 0.9760 ±0.002 |
| 0.9005 ±0.003 | 0.7820 ±0.007 | 0.8878 ±0.001 | 0.8268 ±0.004 | 0.9914 ±0.000 | 0.8119 ±0.009 | 0.8008 ±0.006 | 0.5027 ±0.016 | 0.9810 ±0.000 |
| 0.9336 ±0.007 | 0.7855 ±0.006 | 0.8988 ±0.002 | 0.8654 ±0.011 | 0.9929 ±0.000 | 0.8260 ±0.011 | 0.8537 ±0.007 | 0.5662 ±0.051 | 0.9885 ±0.000 |
| 0.9400 ±0.007 | 0.7817 ±0.008 | 0.9043 ±0.000 | 0.8641 ±0.006 | 0.9916 ±0.000 | 0.8343 ±0.004 | 0.8551 ±0.004 | 0.5573 ±0.060 | 0.9893 ±0.000 |
| 0.9337 ±0.003 | 0.7775 ±0.006 | 0.8988 ±0.003 | 0.8613 ±0.003 | 0.9928 ±0.000 | 0.8427 ±0.005 | 0.8407 ±0.008 | 0.7080 ±0.000 | 0.9851 ±0.000 |
| 0.9392 ±0.004 | 0.7875 ±0.002 | 0.8952 ±0.004 | 0.8490 ±0.005 | 0.9940 ±0.000 | 0.8265 ±0.003 | 0.8244 ±0.002 | 0.6304 ±0.044 | 0.9861 ±0.001 |
| 0.9476 ±0.000 | 0.7903 ±0.014 | 0.8978 ±0.000 | 0.8491 ±0.007 | 0.9930 ±0.000 | 0.8400 ±0.007 | 0.8207 ±0.002 | 0.7254 ±0.000 | 0.9881 ±0.001 |
| 0.9386 ±0.000 | 0.7861 ±0.003 | 0.8946 ±0.001 | 0.8268 ±0.014 | 0.9938 ±0.000 | 0.8235 ±0.005 | 0.8319 ±0.004 | 0.6026 ±0.006 | 0.9865 ±0.000 |
| 0.9439 ±0.000 | 0.7829 ±0.003 | 0.8968 ±0.000 | 0.8594 ±0.008 | 0.9938 ±0.000 | 0.8388 ±0.003 | 0.8328 ±0.005 | 0.6235 ±0.064 | 0.9864 ±0.000 |
| 0.9398 ±0.003 | 0.7785 ±0.005 | 0.8961 ±0.000 | 0.8510 ±0.002 | 0.9933 ±0.000 | 0.8103 ±0.003 | 0.8338 ±0.000 | 0.6470 ±0.046 | 0.9894 ±0.000 |

Table 4: Performance of the existing pre-trained models and CrisisTransformers (CT-\*) on classification task across 18 crisis datasets (D-01 through D-18), with average F1-macro (at 95% confidence interval) being reported. For the corresponding dataset names of each dataset identifier, please refer to Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"). The best scores are shown in bold.

| model  |        |        |        |        |        |        |        |        |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0.0709 | 0.0777 | 0.0783 | 0.0795 | 0.073  | 0.0739 | 0.0618 | 0.0938 | 0.1719 |
| 0.2532 | 0.2508 | 0.2687 | 0.2349 | 0.218  | 0.2382 | 0.2255 | 0.2465 | 0.299  |
| 0.0897 | 0.1182 | 0.1108 | 0.121  | 0.1079 | 0.1076 | 0.1171 | 0.1904 | 0.2429 |
| 0.0235 | 0.0273 | 0.0272 | 0.0272 | 0.0252 | 0.0254 | 0.0276 | 0.037  | 0.0502 |
| 0.0026 | 0.0027 | 0.0028 | 0.0026 | 0.0024 | 0.0025 | 0.0024 | 0.0041 | 0.0074 |
| 0.1129 | 0.1249 | 0.1262 | 0.1286 | 0.1088 | 0.1183 | 0.1225 | 0.1854 | 0.272  |
| 0.0646 | 0.0713 | 0.0805 | 0.0713 | 0.063  | 0.0665 | 0.0554 | 0.0898 | 0.1702 |
| 0.1129 | 0.1238 | 0.1216 | 0.1117 | 0.1094 | 0.1243 | 0.1303 | 0.1137 | 0.1407 |
| 0.1078 | 0.1186 | 0.1164 | 0.1064 | 0.1052 | 0.1199 | 0.127  | 0.1111 | 0.1368 |
| 0.0309 | 0.036  | 0.034  | 0.0346 | 0.0328 | 0.032  | 0.0378 | 0.0456 | 0.0625 |
| 0.0527 | 0.0581 | 0.0548 | 0.0561 | 0.0515 | 0.052  | 0.0676 | 0.0766 | 0.1019 |
| 0.0541 | 0.0584 | 0.0552 | 0.0564 | 0.0521 | 0.0529 | 0.0713 | 0.0731 | 0.097  |
| 0.2712 | 0.2962 | 0.2636 | 0.2694 | 0.2578 | 0.2648 | 0.2921 | 0.273  | 0.3179 |
| 0.2758 | 0.2935 | 0.2697 | 0.2579 | 0.2468 | 0.2673 | 0.2867 | 0.253  | 0.3008 |
| 0.2732 | 0.2853 | 0.2679 | 0.2492 | 0.2404 | 0.2661 | 0.2858 | 0.2432 | 0.2874 |

| model  |        |        |        |        |        |        |        |        |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0.1119 | 0.053  | 0.0961 | 0.0629 | 0.0962 | 0.0813 | 0.0654 | 0.054  | 0.0665 |
| 0.2475 | 0.1997 | 0.3069 | 0.2261 | 0.3039 | 0.2776 | 0.2283 | 0.1949 | 0.2452 |
| 0.1839 | 0.102  | 0.1684 | 0.1076 | 0.162  | 0.1587 | 0.1423 | 0.1154 | 0.1309 |
| 0.0354 | 0.0246 | 0.0372 | 0.0262 | 0.0378 | 0.0331 | 0.0292 | 0.0239 | 0.0302 |
| 0.0042 | 0.0021 | 0.004  | 0.0026 | 0.004  | 0.0032 | 0.0033 | 0.0023 | 0.003  |
| 0.1933 | 0.1109 | 0.1666 | 0.1246 | 0.1671 | 0.1516 | 0.1291 | 0.1069 | 0.1366 |
| 0.1134 | 0.0439 | 0.0922 | 0.0596 | 0.0916 | 0.0796 | 0.0665 | 0.0416 | 0.0648 |
| 0.1188 | 0.1168 | 0.1454 | 0.1279 | 0.1451 | 0.1324 | 0.1173 | 0.0986 | 0.136  |
| 0.1161 | 0.1135 | 0.1409 | 0.1248 | 0.1403 | 0.1285 | 0.115  | 0.0977 | 0.1328 |
| 0.0445 | 0.0317 | 0.0505 | 0.0353 | 0.0509 | 0.0444 | 0.0372 | 0.0314 | 0.0407 |
| 0.078  | 0.0503 | 0.0791 | 0.0631 | 0.0786 | 0.0697 | 0.0627 | 0.0587 | 0.0724 |
| 0.0744 | 0.0528 | 0.0791 | 0.0651 | 0.0788 | 0.0691 | 0.0641 | 0.0601 | 0.0751 |
| 0.2831 | 0.2644 | 0.3284 | 0.2814 | 0.3194 | 0.3028 | 0.2418 | 0.2107 | 0.2886 |
| 0.2586 | 0.2537 | 0.3224 | 0.2782 | 0.3162 | 0.2938 | 0.2315 | 0.1996 | 0.2829 |
| 0.25   | 0.2519 | 0.3122 | 0.2795 | 0.3071 | 0.2844 | 0.2283 | 0.1965 | 0.2856 |

Table 5: Performance of the existing pre-trained models and CrisisTransformers on sentence encoding task across 18 crisis datasets (D-01 through D-18), with weighted average cosine similarity being reported. The best scores are shown in bold, and the second-best scores are underlined. Note that results reported in this table are intended for comparative purposes only; embeddings generated by pre-trained models out-of-the-box do not produce semantically meaningful sentence embeddings.

| model                                                                     | D-1    | D-2    | D-3    | D-4    | D-5    | D-6    | D-7    | D-8    | D-9    |
| ------------------------------------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Sentence-Transformers                                                     | 0.6103 | 0.6809 | 0.5407 | 0.5632 | 0.487  | 0.5698 | 0.6019 | 0.7528 | 0.8749 |
| SimCSE                                                                    | 0.18   | 0.18   | 0.15   | 0.17   | 0.16   | 0.17   | 0.17   | 0.19   | 0.23   |
| CT-M1-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.6201 | 0.7647 | 0.5936 | 0.59   | 0.6276 | 0.6344 | 0.8147 | 0.7117 | 0.8438 |
| CT-M1-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6278 | 0.7628 | 0.5964 | 0.5904 | 0.6229 | 0.6382 | 0.8212 | 0.7156 | 0.8469 |
| CT-M2-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.5535 | 0.7057 | 0.5418 | 0.5652 | 0.5297 | 0.5594 | 0.7277 | 0.7149 | 0.8667 |
| CT-M2-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.587  | 0.735  | 0.535  | 0.5968 | 0.5542 | 0.5794 | 0.8034 | 0.7136 | 0.8705 |
| CT-M2-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.5834 | 0.7366 | 0.5391 | 0.5955 | 0.5573 | 0.5834 | 0.8002 | 0.71   | 0.8721 |
| CT-M3-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.5672 | 0.7155 | 0.5484 | 0.5756 | 0.5459 | 0.5709 | 0.7593 | 0.701  | 0.8275 |
| CT-M3-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.5971 | 0.7265 | 0.5528 | 0.5864 | 0.5378 | 0.5922 | 0.778  | 0.6924 | 0.8215 |
| CT-M3-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6002 | 0.7262 | 0.5556 | 0.582  | 0.5382 | 0.5951 | 0.785  | 0.6937 | 0.825  |
| CT-M1-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6465 | 0.7338 | 0.6059 | 0.615  | 0.6016 | 0.6649 | 0.7874 | 0.7657 | 0.871  |
| CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6432 | 0.7385 | 0.6097 | 0.6151 | 0.6053 | 0.6626 | 0.7912 | 0.7685 | 0.8735 |
| CT-M2-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.5702 | 0.6753 | 0.5536 | 0.5924 | 0.5465 | 0.5709 | 0.7063 | 0.7554 | 0.8733 |
| CT-M2-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.5855 | 0.7145 | 0.5436 | 0.6123 | 0.6163 | 0.5903 | 0.8158 | 0.7784 | 0.8833 |
| CT-M2-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.5972 | 0.7208 | 0.5466 | 0.6116 | 0.6118 | 0.6049 | 0.8261 | 0.7769 | 0.8883 |
| CT-M3-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.605  | 0.7022 | 0.5583 | 0.6147 | 0.5674 | 0.595  | 0.7504 | 0.7301 | 0.828  |
| CT-M3-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6307 | 0.715  | 0.5687 | 0.6199 | 0.5529 | 0.6151 | 0.7652 | 0.7334 | 0.8289 |
| CT-M3-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6314 | 0.7144 | 0.5684 | 0.6171 | 0.5518 | 0.6161 | 0.7682 | 0.7307 | 0.8274 |

Table 6: (Part 1/2) Performance of baselines and our sentence encoders in the sentence encoding task. The best scores are shown in bold, and the second-best scores are underlined.

| model                                                                     | D-10   | D-11   | D-12   | D-13   | D-14   | D-15   | D-16   | D-17   | D-18   |
| ------------------------------------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Sentence-Transformers                                                     | 0.6674 | 0.7297 | 0.8211 | 0.4939 | 0.7259 | 0.7579 | 0.532  | 0.5192 | 0.545  |
| SimCSE                                                                    | 0.19   | 0.18   | 0.2    | 0.15   | 0.19   | 0.19   | 0.1683 | 0.15   | 0.16   |
| CT-M1-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.6878 | 0.8553 | 0.8426 | 0.7094 | 0.7609 | 0.7624 | 0.6788 | 0.5874 | 0.727  |
| CT-M1-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6904 | 0.8584 | 0.8439 | 0.7146 | 0.7599 | 0.7567 | 0.6814 | 0.5918 | 0.728  |
| CT-M2-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.685  | 0.7972 | 0.823  | 0.6105 | 0.7357 | 0.7403 | 0.6182 | 0.5505 | 0.6392 |
| CT-M2-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.7075 | 0.838  | 0.8375 | 0.7005 | 0.7551 | 0.7424 | 0.6047 | 0.5555 | 0.6978 |
| CT-M2-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.7035 | 0.8425 | 0.8383 | 0.6952 | 0.7597 | 0.7432 | 0.6124 | 0.5727 | 0.6824 |
| CT-M3-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.6633 | 0.8135 | 0.8097 | 0.6797 | 0.7247 | 0.7391 | 0.6155 | 0.536  | 0.6602 |
| CT-M3-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.6659 | 0.8137 | 0.8107 | 0.6831 | 0.7346 | 0.7421 | 0.6508 | 0.5655 | 0.6778 |
| CT-M3-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6644 | 0.8175 | 0.8134 | 0.6907 | 0.7347 | 0.7464 | 0.652  | 0.5657 | 0.6857 |
| CT-M1-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7424 | 0.7961 | 0.8232 | 0.7455 | 0.744  | 0.7867 | 0.6526 | 0.5346 | 0.727  |
| CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7417 | 0.7912 | 0.8205 | 0.7496 | 0.7441 | 0.7868 | 0.6547 | 0.531  | 0.7261 |
| CT-M2-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.7205 | 0.7013 | 0.7833 | 0.626  | 0.7207 | 0.7593 | 0.5885 | 0.5022 | 0.6529 |
| CT-M2-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7478 | 0.7925 | 0.8013 | 0.7532 | 0.7453 | 0.7759 | 0.6333 | 0.5363 | 0.7323 |
| CT-M2-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7399 | 0.7851 | 0.8094 | 0.7632 | 0.7564 | 0.7815 | 0.6416 | 0.5417 | 0.743  |
| CT-M3-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.704  | 0.7404 | 0.7825 | 0.695  | 0.7138 | 0.7482 | 0.6241 | 0.524  | 0.6728 |
| CT-M3-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7092 | 0.7419 | 0.7885 | 0.7032 | 0.7235 | 0.7641 | 0.6355 | 0.5378 | 0.6883 |
| CT-M3-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7071 | 0.7442 | 0.7894 | 0.7049 | 0.7241 | 0.7635 | 0.6361 | 0.543  | 0.6903 |

Table 7: (Part 2/2) Performance of baselines and our sentence encoders in the sentence encoding task. The best scores are shown in bold, and the second-best scores are underlined.

| sentence encoder                                                            | avg. score |
| --------------------------------------------------------------------------- | ---------- |
| Sentence-Transformers                                                       | 0.6374     |
| SimCSE                                                                      | 0.1765     |
| 10k training samples of GooAQ                                               |            |
| – CT-M1-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.7117     |
| – CT-M1-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.7137     |
| – CT-M2-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.6646     |
| – CT-M2-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.6896     |
| – CT-M2-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6904     |
| – CT-M3-OneLook-SEMNR{}\_{\\text{MNR}}                                      | 0.6696     |
| – CT-M3-BestLoss-SEMNR{}\_{\\text{MNR}}                                     | 0.6793     |
| – CT-M3-Complete-SEMNR{}\_{\\text{MNR}}                                     | 0.6817     |
| 10k training samples of QQP                                                 |            |
| – CT-M1-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7135     |
| – CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7140     |
| – CT-M2-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.661      |
| – CT-M2-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7032     |
| – CT-M2-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7081     |
| – CT-M3-OneLook-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}  | 0.6753     |
| – CT-M3-BestLoss-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6845     |
| – CT-M3-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.6848     |
| Complete QQP (102k training samples)                                        |            |
| – CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7394     |
| Complete QQP and AllNLI (378k training samples)                             |            |
| – CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}} | 0.7485     |

Table 8: Overall performance of the evaluated sentence encoders across 18 datasets.

Furthermore, we investigated different pooling methods in our best-performing sentence encoder, CT-M1-Complete-SEMNR w/ hard negatives{}\_{\\text{MNR~w/~hard~negatives}}. We considered four different strategies: (i) mean pooling with attention, (ii) \[CLS\] embedding, (iii) max-pooling, and (iv) mean pooling without attention. We utilized the 18 datasets listed in Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") for this task. Results are summarized in Table [9](#S4.T9 "Table 9 ‣ 4.2 Evaluations ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"). Across all datasets, mean pooling with attention achieved the highest scores (0.7485 on avg.). If attention is not considered while mean pooling, it seems to degrade the performance (0.5969 on avg.) and fall behind using \[CLS\] embedding (0.6612 on avg.). Max pooling seems to be the worst (0.3372 on avg.) among these four pooling strategies.

| Dataset | Mean (w/ atten.) | CLS    | Max    | Mean (w/o atten.) |
| ------- | ---------------- | ------ | ------ | ----------------- |
| 0.7191  | 0.641            | 0.3484 | 0.4905 |                   |
| 0.8070  | 0.7048           | 0.3114 | 0.5461 |                   |
| 0.6128  | 0.5313           | 0.2793 | 0.4256 |                   |
| 0.6872  | 0.6082           | 0.2810 | 0.5481 |                   |
| 0.6744  | 0.5825           | 0.2677 | 0.5390 |                   |
| 0.6960  | 0.6010           | 0.2821 | 0.4769 |                   |
| 0.8742  | 0.7602           | 0.2840 | 0.7223 |                   |
| 0.7027  | 0.6420           | 0.4780 | 0.6061 |                   |
| 0.9021  | 0.8136           | 0.6086 | 0.7532 |                   |
| 0.7167  | 0.6449           | 0.4864 | 0.5757 |                   |
| 0.8982  | 0.7781           | 0.2259 | 0.7946 |                   |
| 0.8342  | 0.7414           | 0.3948 | 0.6506 |                   |
| 0.7874  | 0.6856           | 0.3064 | 0.6472 |                   |
| 0.7762  | 0.6799           | 0.356  | 0.6083 |                   |
| 0.7708  | 0.6816           | 0.3635 | 0.5831 |                   |
| 0.6549  | 0.5921           | 0.2875 | 0.5865 |                   |
| 0.5874  | 0.5331           | 0.2226 | 0.5406 |                   |
| 0.7722  | 0.6814           | 0.2873 | 0.6507 |                   |
| 0.7485  | 0.6612           | 0.3372 | 0.5969 |                   |

Table 9: Performance of different pooling methods in our best-performing sentence encoder.

### 4.3 Inference time analysis

| Tokenization (in milliseconds)    |        |        |        |        |        |        |
| --------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                   |        |        |        |        |        |        |
| microsoft/mpnet-base              | 0.0796 | 0.0224 | 0.0467 | 0.0656 | 0.0755 | 0.0969 |
| vinai/bertweet-covid19-base-cased | 0.1868 | 0.0725 | 0.0818 | 0.1457 | 0.1823 | 0.2291 |
| bert-base-cased                   | 0.0772 | 0.0183 | 0.0459 | 0.0670 | 0.0743 | 0.0919 |
| roberta-base                      | 0.0743 | 0.0193 | 0.0420 | 0.0643 | 0.0679 | 0.0910 |
| xlm-roberta-base                  | 0.0854 | 0.0230 | 0.0471 | 0.0736 | 0.0847 | 0.1016 |
| albert-base-v2                    | 0.0966 | 0.0263 | 0.0579 | 0.0836 | 0.0883 | 0.1145 |
| google/electra-base-discriminator | 0.0875 | 0.0206 | 0.0591 | 0.0710 | 0.0851 | 0.1032 |
| Sentence-Transformers             | 0.0791 | 0.0215 | 0.0461 | 0.0642 | 0.0773 | 0.0943 |
| SimCSE                            | 0.0744 | 0.0203 | 0.0411 | 0.0644 | 0.0699 | 0.0909 |
| CT-M1                             | 0.0759 | 0.0202 | 0.0429 | 0.0662 | 0.0713 | 0.0930 |
| CT-M2                             | 0.0800 | 0.0229 | 0.0486 | 0.0674 | 0.0754 | 0.0930 |
| CT-M3                             | 0.1873 | 0.0711 | 0.0880 | 0.1487 | 0.1834 | 0.2268 |

| Embeddings generation (in milliseconds) |        |        |        |        |        |        |
| --------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                         |        |        |        |        |        |        |
| microsoft/mpnet-base                    | 0.0047 | 0.0039 | 0.0025 | 0.0032 | 0.0036 | 0.0045 |
| vinai/bertweet-covid19-base-cased       | 0.0031 | 0.0046 | 0.0009 | 0.0011 | 0.0016 | 0.0026 |
| bert-base-cased                         | 0.0031 | 0.0044 | 0.0013 | 0.0013 | 0.0016 | 0.0025 |
| roberta-base                            | 0.0031 | 0.0045 | 0.0009 | 0.0013 | 0.0016 | 0.0025 |
| xlm-roberta-base                        | 0.0032 | 0.0047 | 0.0009 | 0.0011 | 0.0016 | 0.0026 |
| albert-base-v2                          | 0.0917 | 0.1302 | 0.0015 | 0.0024 | 0.0040 | 0.2190 |
| google/electra-base-discriminator       | 0.0031 | 0.0042 | 0.0012 | 0.0014 | 0.0018 | 0.0026 |
| Sentence-Transformers                   | 0.0046 | 0.0039 | 0.0024 | 0.0032 | 0.0036 | 0.0043 |
| SimCSE                                  | 0.0031 | 0.0046 | 0.0009 | 0.0011 | 0.0015 | 0.0026 |
| CT-M1                                   | 0.0032 | 0.0046 | 0.0009 | 0.0011 | 0.0017 | 0.0029 |
| CT-M2                                   | 0.0031 | 0.0045 | 0.0009 | 0.0012 | 0.0016 | 0.0026 |
| CT-M3                                   | 0.0032 | 0.0047 | 0.0009 | 0.0011 | 0.0015 | 0.0025 |

Table 10: Inference times of the baselines and CrisisTransformers.

Given that each model utilizes a distinct set of tokens for tokenization and considering the varying number of parameters in each model, we conducted a comprehensive analysis of the inference time for both the baselines and CrisisTransformers, focusing on two key tasks: tokenization and embedding generation. Tokenization entails the creation of input identifiers and attention masks, while embedding generation encompasses feeding the outputs of tokenization into the model and producing mean token embeddings while considering the attention masks. These analyses were carried out using an Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz alongside an 80GB A100 Nvidia GPU. We utilized native tokenizers from the latest release of each model. For every dataset listed in Table [3](#S3.T3 "Table 3 ‣ 3.3.1 Labelled crisis-related datasets ‣ 3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts"), we executed tokenization and embedding generation processes. We report the average processing times (measured in milliseconds) along with the standard deviation, minimum time, and Quartiles in Table [10](#S4.T10 "Table 10 ‣ 4.3 Inference time analysis ‣ 4 Results and Discussion ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts").

In terms of tokenization, CT-M1 outperforms MPNet, BERTweet, BERT, XLM-RoBERTa, ALBERT, and Sentence-Transformers in average performance, despite having the largest vocabulary size. It also exhibits similar performance to RoBERTa and SimCSE. Among CrisisTransformers, CT-M3, which is based on BERTweet, demonstrates the slowest tokenization speed performance, comparable to BERTweet. Regarding embedding generation, CrisisTransformers exhibit nearly identical performance (0.0031–0.0032ms) to BERTweet, BERT, RoBERTa, XLM-RoBERTa, and SimCSE. MPNet, ALBERT, and Sentence-Transformers show much higher embedding generation times.

### 4.4 Intended uses and limitations

CrisisTransformers offers a selection of 8 pre-trained models alongside a sentence encoder. Similar to BERT and RoBERTa, the pre-trained models are designed for fine-tuning in downstream tasks (that require an entire sentence to make decisions), such as sequence classification and token classification. Additionally, the sentence encoder, similar to Sentence-Transformers, is intended for generating semantically meaningful sentence embeddings for tasks such as semantic search, clustering, and topic modeling.

The training corpus utilized by CrisisTransformers comprised a substantial volume of unfiltered tweets, inherently containing non-neutral content. Consequently, similar to RoBERTa and BERTweet, both the pre-trained models and their fine-tuned versions are prone to biased predictions. Biased predictions in this context refer to the tendency of the models to produce outputs that favour or exhibit partiality towards certain groups, perspectives, or sentiments present in the training data. Moreover, these models are specifically designed for processing crisis-related social media texts. Despite this focus, we observed that the CT-M2 and CT-M3 variants, which are built upon RoBERTa and BERTweet, respectively, also exhibit strong performance when applied to tweets from diverse domains (refer [A](#A1 "Appendix A Performance of baselines and CrisisTransformers on tweets from general domains ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts")). This efficacy can be attributed to the robustness of their original base models, supplemented by additional training data provided during this study. Furthermore, at this stage, CrisisTransformers can process only English-language tweets. As a part of future work, we plan to release their multi-lingual versions \[[70](#bib.bib70 "")\].

## 5 Conclusion

In this study, we introduced CrisisTransformers, an ensemble of pre-trained language models and sentence encoders designed for processing crisis-related social media texts. The pre-trained models were trained on a large-scale corpus of over 15 billion word tokens sourced from tweets associated with more than 30 crisis events that occurred between 2006 and 2023. Additionally, we fine-tuned the pre-trained models using siamese and triplet networks to create sentence encoders. Existing models and CrisisTransformers were evaluated on 18 crisis-specific datasets for classification and sentence encoding tasks. Our pre-trained models outperform strong baselines across all 18 datasets in classification tasks, and our best-performing sentence encoder improves the state-of-the-art by 17.43% in sentence encoding tasks. We publicly release CrisisTransformers, which include 8 variants of pre-trained models and the best-performing sentence encoder, hoping that they will serve as a robust baseline for tasks that involve processing crisis-related social media texts.

CrisisTransformers offers checkpoints of models trained from scratch (CT-M1) and those initialized with RoBERTa’s weights (CT-M2) and BERTweet’s weights (CT-M3). During experimentations, we observed that pre-trained models (CT-M2 and CT-M3), which undergo further pre-training, leverage existing knowledge for efficient initial convergence, unlike randomly initialized CT-M1. CT-M2 and CT-M3 exhibited rapid initial drops in loss; CT-M3 later aligned with CT-M1 in terms of final loss. All models plateaued, implying convergence. In classification, CT-M1 performed best on 7 datasets, CT-M2 on 9, and CT-M3 on 2. Regarding sentence encoding, CT-M1 outperformed in 11 datasets, CT-M2 on 6, and CT-M3 on 1. Considering the training objectives, models trained with hard negatives achieved the highest scores across 11 datasets, which remains in line with what has been reported in the literature. We noticed that the CT-M1 at the lowest loss utilizing only 10k training samples with the MNR with hard negatives training objective outperformed the state-of-the-art Sentence-Transformers (trained on 1 billion samples) by a significant margin of 12%. By increasing the training samples to 378k using the QQP+AllNLI datasets, the performance improved further to 17.43%. This observation confirmed that domain-specific pre-trained models demonstrate significant improvements over general-purpose models in sentence encoding tasks. Going forward, our future objectives include training the sentence encoders on a scale similar to Sentence-Transformers. Also, the proposed models process only English-language tweets. As a future task, we aim to release their multi-lingual versions.

## Acknowledgements

This study is supported by the Melbourne Research Scholarship from the University of Melbourne, Australia. This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne, which was established with the assistance of LIEF Grant LE170100200. The cloud infrastructure required to maintain COV19Tweets over the last three years was provided by DigitalOcean. We appreciate the insights provided by Dat Quoc Nguyen (BERTweet’s co-author) during the pre-training phase of CrisisTransformers.

## CRediT authorship contribution statement

Rabindra Lamsal performed Conceptualization, Data curation, Methodology, Software, Visualization, Writing–first draft. Maria Rodriguez Read and Shanika Karunasekera performed Conceptualization, Supervision, Writing–Review & Editing.

## Declaration of competing interest

All authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

*   \[1\] M. Imran, C. Castillo, F. Diaz, S. Vieweg, Processing social media messages in mass emergency: A survey, ACM Computing Surveys (CSUR) 47 (4) (2015) 1–38.
*   \[2\] R. Lamsal, A. Harwood, M. R. Read, Socially enhanced situation awareness from microblogs using artificial intelligence: A survey, ACM Computing Surveys 55 (4) (2022) 1–38.
*   \[3\] K. Starbird, L. Palen, Pass it on?: Retweeting in mass emergency, in: I, 2010.
*   \[4\] R. Thomson, N. Ito, H. Suda, F. Lin, Y. Liu, R. Hayasaka, R. Isochi, Z. Wang, Trusting tweets: The fukushima disaster and information source credibility on twitter., in: ISCRAM, 2012.
*   \[5\] F. Alam, F. Ofli, M. Imran, M. Aupetit, A twitter tale of three hurricanes: Harvey, irma, and maria, arXiv preprint arXiv:1805.05144 (2018).
*   \[6\] N. Pourebrahim, S. Sultana, J. Edwards, A. Gochanour, S. Mohanty, Understanding communication dynamics on twitter during natural disasters: A case study of hurricane sandy, International journal of disaster risk reduction 37 (2019) 101176.
*   \[7\] R. Lamsal, M. R. Rodriguez, S. Karunasekera, A twitter narrative of the covid-19 pandemic in australia, in: Proceedings of the International ISCRAM Conference, 2023, pp. 353–370.
*   \[8\] A. Sarcevic, L. Palen, J. White, K. Starbird, M. Bagdouri, K. Anderson, " beacons of hope" in decentralized coordination: learning from on-the-ground medical twitterers during the 2010 haiti earthquake, in: CSCW, 2012, pp. 47–56.
*   \[9\] A. L. Hughes, L. Palen, Twitter adoption and use in mass convergence and emergency events, International journal of emergency management 6 (3-4) (2009) 248–260.
*   \[10\] S. Vieweg, A. L. Hughes, K. Starbird, L. Palen, Microblogging during two natural hazards events: what twitter may contribute to situational awareness, in: Proceedings of the SIGCHI conference on human factors in computing systems, 2010, pp. 1079–1088.
*   \[11\] S. Vieweg, Situational awareness in mass emergency: A behavioral and linguistic analysis of microblogged communications, Ph.D. thesis, University of Colorado at Boulder (2012).
*   \[12\] S. Stieglitz, M. Mirbabaie, B. Ross, C. Neuberger, Social media analytics–challenges in topic discovery, data collection, and data preparation, International journal of information management 39 (2018) 156–168.
*   \[13\] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).
*   \[14\] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, J. Kang, Biobert: a pre-trained biomedical language representation model for biomedical text mining, Bioinformatics 36 (4) (2020) 1234–1240.
*   \[15\] I. Beltagy, K. Lo, A. Cohan, SciBERT: A pretrained language model for scientific text, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, Hong Kong, China, 2019, pp. 3615–3620.
*   \[16\] K. Huang, J. Altosaar, R. Ranganath, Clinicalbert: Modeling clinical notes and predicting hospital readmission, arXiv preprint arXiv:1904.05342 (2019).
*   \[17\] Y. Yang, M. C. S. Uy, A. Huang, Finbert: A pretrained language model for financial communications, arXiv preprint arXiv:2006.08097 (2020).
*   \[18\] N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using Siamese BERT-networks, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, Hong Kong, China, 2019, pp. 3982–3992.
*   \[19\] T. Gao, X. Yao, D. Chen, SimCSE: Simple contrastive learning of sentence embeddings, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 2021, pp. 6894–6910.
*   \[20\] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Association for Computational Linguistics, Minneapolis, Minnesota, 2019, pp. 4171–4186.
*   \[21\] F. Alam, H. Sajjad, M. Imran, F. Ofli, Crisisbench: Benchmarking crisis-related social media datasets for humanitarian information processing, in: Proceedings of the International AAAI Conference on Web and Social Media, Vol. 15, 2021, pp. 923–932.
*   \[22\] R. Biddle, A. Joshi, S. Liu, C. Paris, G. Xu, Leveraging sentiment distributions to distinguish figurative from literal health reports on twitter, in: Proceedings of the web conference 2020, 2020, pp. 1217–1227.
*   \[23\] P. Y. W. Myint, S. L. Lo, Y. Zhang, Unveiling the dynamics of crisis events: Sentiment and emotion analysis via multi-task learning with attention mechanism and subject-based intent prediction, Information Processing & Management 61 (4) (2024) 103695.
*   \[24\] A. Z. Klein, A. Magge, K. O’Connor, J. I. Flores Amaro, D. Weissenbacher, G. Gonzalez Hernandez, Toward using twitter for tracking covid-19: a natural language processing pipeline and exploratory data set, Journal of medical Internet research 23 (1) (2021) e25314.
*   \[25\] Z. Ashktorab, C. Brown, M. Nandi, A. Culotta, Tweedr: Mining twitter to inform disaster response., in: ISCRAM, 2014, pp. 269–272.
*   \[26\] C. Caragea, N. J. McNeese, A. R. Jaiswal, G. Traylor, H.-W. Kim, P. Mitra, D. Wu, A. H. Tapia, C. L. Giles, B. J. Jansen, et al., Classifying text messages for the haiti earthquake., in: ISCRAM, Citeseer, 2011.
*   \[27\] M. Imran, S. Elbassuoni, C. Castillo, F. Diaz, P. Meier, Extracting information nuggets from disaster-related messages in social media., Iscram 201 (3) (2013) 791–801.
*   \[28\] H. Li, X. Li, D. Caragea, C. Caragea, Comparison of word embeddings and sentence encodings as generalized representations for crisis tweet classification tasks, ISCRAM Asia Pacific (2018).
*   \[29\] R. Dutt, M. Basu, K. Ghosh, S. Ghosh, Utilizing microblogs for assisting post-disaster relief operations via matching resource needs and availabilities, Information Processing & Management 56 (5) (2019) 1680–1697.
*   \[30\] S. A. Curiskis, B. Drake, T. R. Osborn, P. J. Kennedy, An evaluation of document clustering and topic modelling in two online social networks: Twitter and reddit, Information Processing & Management 57 (2) (2020) 102034.
*   \[31\] M. Grootendorst, Bertopic: Neural topic modeling with a class-based tf-idf procedure, arXiv preprint arXiv:2203.05794 (2022).
*   \[32\] D. Q. Nguyen, T. Vu, A. Tuan Nguyen, BERTweet: A pre-trained language model for English tweets, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Association for Computational Linguistics, Online, 2020, pp. 9–14.
*   \[33\] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, A. Rush, Transformers: State-of-the-art natural language processing, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Association for Computational Linguistics, Online, 2020, pp. 38–45.
*   \[34\] R. Prasad, A. U. Udeme, S. Misra, H. Bisallah, Identification and classification of transportation disaster tweets using improved bidirectional encoder representations from transformers, International Journal of Information Management Data Insights 3 (1) (2023) 100154.
*   \[35\] R. Koshy, S. Elango, Multimodal tweet classification in disaster response systems using transformer-based bidirectional attention model, Neural Computing and Applications 35 (2) (2023) 1607–1627.
*   \[36\] R. Suwaileh, T. Elsayed, M. Imran, Idrisi-re: A generalizable dataset with benchmarks for location mention recognition on disaster tweets, Information Processing & Management 60 (3) (2023) 103340.
*   \[37\] S. Poddar, M. Mondal, J. Misra, N. Ganguly, S. Ghosh, Winds of change: Impact of covid-19 on vaccine-related opinions of twitter users, in: Proceedings of the International AAAI Conference on Web and Social Media, Vol. 16, 2022, pp. 782–793.
*   \[38\] L.-A. Cotfas, C. Delcea, I. Roxin, C. Ioanăş, D. S. Gherai, F. Tajariol, The longest month: analyzing covid-19 vaccination opinions dynamics from tweets in the month following the first vaccine announcement, Ieee Access 9 (2021) 33203–33223.
*   \[39\] K. Hayawi, S. Shahriar, M. A. Serhani, I. Taleb, S. S. Mathew, Anti-vax: a novel twitter dataset for covid-19 vaccine misinformation detection, Public health 203 (2022) 23–30.
*   \[40\] R. Lamsal, A. Harwood, M. R. Read, Where did you tweet from? inferring the origin locations of tweets based on contextual information, in: 2022 IEEE International Conference on Big Data (Big Data), IEEE, 2022, pp. 3935–3944.
*   \[41\] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., Language models are few-shot learners, Advances in neural information processing systems 33 (2020) 1877–1901.
*   \[42\] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, L. Zettlemoyer, Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, arXiv preprint arXiv:1910.13461 (2019).
*   \[43\] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, P. J. Liu, Exploring the limits of transfer learning with a unified text-to-text transformer, Journal of machine learning research 21 (140) (2020) 1–67.
*   \[44\] Q. Zhong, L. Ding, J. Liu, B. Du, D. Tao, E2s2: Encoding-enhanced sequence-to-sequence pretraining for language understanding and generation, IEEE Transactions on Knowledge and Data Engineering (2023).
*   \[45\] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, V. Stoyanov, Roberta: A robustly optimized bert pretraining approach, arXiv preprint arXiv:1907.11692 (2019).
*   \[46\] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut, Albert: A lite bert for self-supervised learning of language representations, arXiv preprint arXiv:1909.11942 (2019).
*   \[47\] K. Song, X. Tan, T. Qin, J. Lu, T.-Y. Liu, Mpnet: Masked and permuted pre-training for language understanding, Advances in Neural Information Processing Systems 33 (2020) 16857–16867.
*   \[48\] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, Q. V. Le, Xlnet: Generalized autoregressive pretraining for language understanding, Advances in neural information processing systems 32 (2019).
*   \[49\] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, V. Stoyanov, Unsupervised cross-lingual representation learning at scale, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, Online, 2020, pp. 8440–8451.
*   \[50\] K. Clark, M.-T. Luong, Q. V. Le, C. D. Manning, Electra: Pre-training text encoders as discriminators rather than generators, arXiv preprint arXiv:2003.10555 (2020).
*   \[51\] R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, S. Fidler, Skip-thought vectors, Advances in neural information processing systems 28 (2015).
*   \[52\] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, Supervised learning of universal sentence representations from natural language inference data, in: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Copenhagen, Denmark, 2017, pp. 670–680.
*   \[53\] F. Hill, K. Cho, A. Korhonen, Learning distributed representations of sentences from unlabelled data, in: Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, San Diego, California, 2016, pp. 1367–1377.
*   \[54\] D. Cer, Y. Yang, S.-y. Kong, N. Hua, N. Limtiaco, R. S. John, N. Constant, M. Guajardo-Cespedes, S. Yuan, C. Tar, et al., Universal sentence encoder, arXiv preprint arXiv:1803.11175 (2018).
*   \[55\] Y. Yang, S. Yuan, D. Cer, S.-y. Kong, N. Constant, P. Pilar, H. Ge, Y.-H. Sung, B. Strope, R. Kurzweil, Learning semantic textual similarity from conversations, in: Proceedings of the Third Workshop on Representation Learning for NLP, Association for Computational Linguistics, Melbourne, Australia, 2018, pp. 164–174.
*   \[56\] Q. Zhong, L. Ding, J. Liu, B. Du, D. Tao, Can chatgpt understand too? a comparative study on chatgpt and fine-tuned bert, arXiv preprint arXiv:2302.10198 (2023).
*   \[57\] R. Lamsal, Design and analysis of a large-scale covid-19 tweets dataset, applied intelligence 51 (2021) 2790–2804.
*   \[58\] R. Lamsal, M. R. Read, S. Karunasekera, Billioncov: An enriched billion-scale collection of covid-19 tweets for efficient hydration, Data in Brief 48 (2023) 109229.
*   \[59\] M. Imran, P. Mitra, C. Castillo, Twitter as a lifeline: Human-annotated Twitter corpora for NLP of crisis-related messages, in: Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), European Language Resources Association (ELRA), Portorož, Slovenia, 2016, pp. 1638–1643.
*   \[60\] F. Alam, F. Ofli, M. Imran, Crisismmd: Multimodal twitter datasets from natural disasters, in: Proceedings of the international AAAI conference on web and social media, Vol. 12, 2018.
*   \[61\] A. Olteanu, C. Castillo, F. Diaz, S. Vieweg, Crisislex: A lexicon for collecting and filtering microblogged communications in crises, in: Proceedings of the international AAAI conference on web and social media, Vol. 8, 2014, pp. 376–385.
*   \[62\] M. Imran, C. Castillo, J. Lucas, P. Meier, S. Vieweg, Aidr: Artificial intelligence for disaster response, in: Proceedings of the 23rd international conference on world wide web, 2014, pp. 159–162.
*   \[63\] M. Imran, S. Elbassuoni, C. Castillo, F. Diaz, P. Meier, Practical extraction of disaster-relevant information from social media, in: Proceedings of the 22nd international conference on world wide web, 2013, pp. 1021–1024.
*   \[64\] M. L. Mauriello, T. Lincoln, G. Hon, D. Simon, D. Jurafsky, P. Paredes, Sad: A stress annotated dataset for recognizing everyday stressors in sms-like conversational systems, in: Extended abstracts of the 2021 CHI conference on human factors in computing systems, 2021, pp. 1–7.
*   \[65\] P. Karisani, E. Agichtein, Did you really just have a heart attack? towards robust detection of personal health mentions in social media, in: Proceedings of the 2018 World Wide Web Conference, 2018, pp. 137–146.
*   \[66\] D. Khashabi, A. Ng, T. Khot, A. Sabharwal, H. Hajishirzi, C. Callison-Burch, GooAQ: Open question answering with diverse answer types, in: Findings of the Association for Computational Linguistics: EMNLP 2021, Association for Computational Linguistics, Punta Cana, Dominican Republic, 2021, pp. 421–433.
*   \[67\] Datasets at hugging face: Training data for text embedding models, [https://huggingface.co/datasets/sentence-transformers/embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data "").
*   \[68\] S. R. Bowman, G. Angeli, C. Potts, C. D. Manning, A large annotated corpus for learning natural language inference, in: Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Lisbon, Portugal, 2015, pp. 632–642.
*   \[69\] A. Williams, N. Nangia, S. Bowman, A broad-coverage challenge corpus for sentence understanding through inference, in: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), Association for Computational Linguistics, 2018, pp. 1112–1122.
*   \[70\] N. Reimers, I. Gurevych, Making monolingual sentence embeddings multilingual using knowledge distillation, arXiv preprint arXiv:2004.09813 (2020).
*   \[71\] F. Barbieri, J. Camacho-Collados, F. Ronzano, L. Espinosa-Anke, M. Ballesteros, V. Basile, V. Patti, H. Saggion, Semeval 2018 task 2: Multilingual emoji prediction, in: Proceedings of The 12th International Workshop on Semantic Evaluation, 2018, pp. 24–33.
*   \[72\] S. Mohammad, F. Bravo-Marquez, M. Salameh, S. Kiritchenko, Semeval-2018 task 1: Affect in tweets, in: Proceedings of the 12th international workshop on semantic evaluation, 2018, pp. 1–17.
*   \[73\] V. Basile, C. Bosco, E. Fersini, D. Nozza, V. Patti, F. M. Rangel Pardo, P. Rosso, M. Sanguinetti, [SemEval-2019 task 5: Multilingual detection of hate speech against immigrants and women in Twitter](https://www.aclweb.org/anthology/S19-2007 ""), in: Proceedings of the 13th International Workshop on Semantic Evaluation, Association for Computational Linguistics, Minneapolis, Minnesota, USA, 2019, pp. 54–63. [doi:10.18653/v1/S19-2007](https://doi.org/10.18653/v1/S19-2007 "").
```
URL [https://www.aclweb.org/anthology/S19-2007](https://www.aclweb.org/anthology/S19-2007 "")
```
*   \[74\] C. Van Hee, E. Lefever, V. Hoste, Semeval-2018 task 3: Irony detection in english tweets, in: Proceedings of The 12th International Workshop on Semantic Evaluation, 2018, pp. 39–50.
*   \[75\] M. Zampieri, S. Malmasi, P. Nakov, S. Rosenthal, N. Farra, R. Kumar, Semeval-2019 task 6: Identifying and categorizing offensive language in social media (offenseval), in: Proceedings of the 13th International Workshop on Semantic Evaluation, 2019, pp. 75–86.
*   \[76\] S. Rosenthal, N. Farra, P. Nakov, Semeval-2017 task 4: Sentiment analysis in twitter, in: Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017), 2017, pp. 502–518.
*   \[77\] S. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, C. Cherry, Semeval-2016 task 6: Detecting stance in tweets, in: Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), 2016, pp. 31–41.

## Appendix A Performance of baselines and CrisisTransformers on tweets from general domains

The table below summarizes the performance of baselines and CrisisTransformers on tweets from non-crisis domains. The same training configurations discussed in Section [3.3](#S3.SS3 "3.3 Fine-tuning ‣ 3 Materials and methods ‣ CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts") were applied for fine-tuning. We report average F1-macro scores from the five runs.

|        |        |        |        |        |        |        |        |        |        |        |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0.2542 | 0.7878 | 0.4973 | 0.7753 | 0.7974 | 0.6992 | 0.5916 | 0.6707 | 0.5454 | 0.622  | 0.5898 |
| 0.2939 | 0.8038 | 0.5391 | 0.8277 | 0.7917 | 0.7133 | 0.5738 | 0.6904 | 0.5613 | 0.5923 | 0.5736 |
| 0.3043 | 0.7583 | 0.5    | 0.6569 | 0.7999 | 0.6862 | 0.5764 | 0.598  | 0.4763 | 0.523  | 0.5515 |
| 0.3230 | 0.7854 | 0.479  | 0.4181 | 0.7853 | 0.7102 | 0.5842 | 0.7081 | 0.5698 | 0.6393 | 0.6687 |
| 0.3177 | 0.7676 | 0.5013 | 0.7009 | 0.7825 | 0.7007 | 0.5973 | 0.6871 | 0.5339 | 0.5598 | 0.5736 |
| 0.2360 | 0.7119 | 0.5125 | 0.6873 | 0.7939 | 0.6846 | 0.4914 | 0.5885 | 0.4891 | 0.442  | 0.4447 |
| 0.2414 | 0.7793 | 0.4781 | 0.7088 | 0.8081 | 0.7115 | 0.5575 | 0.5927 | 0.5134 | 0.5256 | 0.5184 |
| 0.3279 | 0.7808 | 0.4962 | 0.7457 | 0.8134 | 0.7112 | 0.6    | 0.6694 | 0.5719 | 0.5845 | 0.6527 |
| 0.3294 | 0.7869 | 0.5077 | 0.6175 | 0.8069 | 0.7214 | 0.5856 | 0.6776 | 0.5641 | 0.5951 | 0.6415 |
| 0.3323 | 0.794  | 0.5002 | 0.5422 | 0.8112 | 0.7031 | 0.6179 | 0.679  | 0.5555 | 0.6074 | 0.6155 |
| 0.3323 | 0.7841 | 0.4844 | 0.5337 | 0.8107 | 0.7133 | 0.592  | 0.688  | 0.5522 | 0.6076 | 0.6526 |
| 0.3333 | 0.7856 | 0.4896 | 0.5554 | 0.8068 | 0.7071 | 0.5946 | 0.6949 | 0.5545 | 0.6014 | 0.6432 |
| 0.2854 | 0.7729 | 0.5512 | 0.8267 | 0.8009 | 0.7147 | 0.626  | 0.6593 | 0.5727 | 0.5954 | 0.6416 |
| 0.2975 | 0.7838 | 0.5223 | 0.8256 | 0.8142 | 0.7215 | 0.6272 | 0.6154 | 0.562  | 0.6249 | 0.6388 |
| 0.2940 | 0.7904 | 0.5362 | 0.8157 | 0.8103 | 0.7239 | 0.624  | 0.5827 | 0.5643 | 0.6127 | 0.652  |

∗stance detection \[[77](#bib.bib77 "")\]

## Appendix B Configurations of CrisisTransformers

|                    | CT-M1              | CT-M2              | CT-M3 |
| ------------------ | ------------------ | ------------------ | ----- |
| RobertaForMaskedLM | RobertaForMaskedLM | RobertaForMaskedLM |       |
| 0.1                | 0.1                | 0.1                |       |
| gelu               | gelu               | gelu               |       |
| 0.1                | 0.1                | 0.1                |       |
| 768                | 768                | 768                |       |
| 3072               | 3072               | 3072               |       |
| 1e-12              | 1e-05              | 1e-05              |       |
| 130                | 514                | 130                |       |
| 12                 | 12                 | 12                 |       |
| 12                 | 12                 | 12                 |       |
| 64,000             | 50,265             | 64001              |       |

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")