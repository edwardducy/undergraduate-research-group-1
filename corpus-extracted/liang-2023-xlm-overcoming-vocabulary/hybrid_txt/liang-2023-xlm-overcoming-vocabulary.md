# XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models

Davis Liang

Hila Gonen

Yuning Mao

Rui Hou

Naman Goyal

Marjan Ghazvininejad

Luke Zettlemoyer

Madian Khabsa

Meta AI

## Abstract

Large multilingual language models typically rely on a single vocabulary shared across 100+ languages. As these models have increased in parameter count and depth, vocabulary size has remained largely unchanged. This vocabulary bottleneck limits the representational capabil ities of multilingual models like XLM-R. In this paper, we introduce a new approach for scaling to very large multilingual vocabularies by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language. Tokenizations using our vocabulary are typically more semantically meaningful and shorter compared to XLM-R. Leveraging this improved vocabulary, we train XLM-V, a multilingual language model with a one million token vocabulary. XLM-V outperforms XLM-R on every task we tested on ranging from natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), to named entity recognition (WikiAnn). XLM-V is particularly effective on low-resource language tasks and outperforms XLM-R by 11.2% and 5.8% absolute on MasakhaNER and Americas NLI, respectively.

## 1 Introduction

While multilingual language models have increased in parameter count and depth over time, vocabulary size has largely remained unchanged: mBART (680M parameters; Liu et al. 2020), XGLM (7.5B parameters, Lin et al. 2021), XLM-R XXL (10.7B parameters; Goyal et al. 2021), mT5 XXL (13B parameters; Xue et al. 2020); and BLOOM (176B parameters; Scao et al. 2022) all share the same 250K token vocabulary size as XLM-R base (Conneau et al., 2019), a 250M parameter model.

For models like mT5 and XLM-R, this 250K vocabulary is shared across 100+ languages. Discounting shared tokens, this results in an average of

2,500 unique tokens per language, calling into question the vocabulary’s ability to represent the diverse selection of languages that it was intended to model. For example, there are 8,105 characters in the Table of General Standard Chinese characters and over 100,000 unique characters in total; the number of commonly used Chinese words (consisting of multiple characters) is even larger (Wikipedia, 2023). In fact, prior work has already shown that this vocabulary bottleneck hinders the performance of multilingual models on question answering and sequence labeling where in-depth token-level and sequence-level understanding is essential (Wang et al., 2019).

In this paper, we construct a large multilingual vocabulary by attending to two core principles: (1) vocabularies can be improved by de-emphasizing token sharing between languages with little lexical overlap and (2) proper vocabulary capacity allocation for individual languages is crucial for ensuring that diverse languages are well-represented. Then, we show that our new vocabulary exhibits favorable characteristics including the ability to frequently output semantically meaningful tokenizations while reducing over-tokenization for lowresource languages. Finally, we present XLM-V, the first multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl (Conneau et al., 2019).

Our main contributions are as follows:

• In Section 3, we present our method for constructing large multilingual vocabularies. Specifically, we improve upon the language clustering algorithm from Chung et al. (2020) by constructing better vector representations for individual languages and leverage Zheng et al. (2021) to improve the vocabulary capacity assignments for each cluster.  
• In Section 5, we demonstrate that XLM-V outperforms comparable baselines that have

the same vocabulary size on XNLI. Additionally, XLM-V outperforms XLM-R on every multilingual language understanding task we tested on (including XNLI, WikiAnn, MLQA, XQuAD, and TyDiQA) by an average of 3.5 points absolute. XLM-V performs especially well on low-resource evaluation datasets like AmericasNLI and MasakhaNER, outperforming XLM-R by 5.8% absolute accuracy and 11.2% absolute F1, respectively.

• Finally, in Section 6, we provide examples and quantitative analysis to compare our new vocabulary to various baselines. Most notably, we provide evidence showing that expanding the vocabulary beyond 1M tokens can degrade performance on downstream tasks.

## 2 Background

## 2.1 Sentencepiece

The Unigram Language Model (ULM) from Kudo and Richardson (2018) is a popular subword segmentation algorithm used to construct vocabularies. ULM begins with a large initial vocabulary that is iteratively pruned to maximize the likelihood of the training corpus (under a unigram language model of the tokens) until the number of tokens falls below some pre-determined vocabulary size threshold, |V|. During tokenization, ULM decodes the most probable segmentation of a sequence through the Viterbi algorithm (Viterbi, 1967). This method is used by both XLM-R and our work.

## 2.2 Clustering

Chung et al. (2020) proposed an approach to multilingual vocabulary construction that balances the trade-off between optimizing for cross-lingual subword sharing and the need for robust representation of individual languages.

Their procedure for building a multilingual vocabulary contains several steps. First, the authors train individual sentencepiece models for each language: for each language l in the set of languages $L$ , a vocabulary $V ^ { l }$ is generated. Then, they create the shared lexicon $\bar { V } ^ { L }$ by taking the union of each language-specific vocabulary, $\bar { V } ^ { L } = \cup _ { l \in L } V ^ { l }$ Next, for each language $l ,$ they construct a binary vector $v ^ { l }$ of dimension $| V ^ { L } |$ which represents the lexicon of l. Each component of $v ^ { l }$ corresponds to a subword in $V ^ { L }$ . In other words, the binary vector $v ^ { l }$ contains a 1 corresponding to each subword present in the vocabulary of l. An illustration of this step is shown in Figure 1. Then, the authors cluster the binary vectors to group lexically similar languages together. Finally, they construct a vocabulary for each cluster and combine the per-cluster vocabularies together to form a unified multilingual vocabulary.

## 2.3 Vocabulary allocation

Zheng et al. (2021) proposed the average log probability (ALP) to evaluate the ability of a vocabulary to represent a particular language. Specifically, given a monolingual corpus composed of sentences $\mathcal { D } _ { i } = \{ s _ { 1 } , . . . , s _ { | \mathcal { D } _ { i } | } \}$ from the i-th language and tokenized with vocabulary $V$ , the average log probability is defined as;

$$
A L P (\mathcal {D} _ {i}, V) = \frac {1}{| \mathcal {D} _ {i} |} \sum_ {j = 1} ^ {| \mathcal {D} _ {i} |} \sum_ {k = 1} ^ {| s _ {j} |} \log p _ {u n i} (s _ {j} ^ {k}) \tag {1}
$$

where $s _ { j } ^ { k }$ is the k-th subword of the sentence $s _ { j }$ and $p _ { u n i } ( \cdot )$ is the unigram distribution counted on the monolingual corpus $\mathcal { D } _ { i }$ . The authors first show that ALP is highly correlated with downstream task performance and then propose a greedy algorithm to determine the desired vocabulary capacity for individual languages in the multilingual vocabulary.

## 3 Methodology

## 3.1 Building the vocabulary

In this subsection, we describe our method for constructing multilingual vocabularies. At a high level, we (1) train individual monolingual sentencepiece models (SPM) for each language in our dataset using the Unigram Language Model (ULM) algorithm (Kudo and Richardson, 2018), (2) use the per-language vocabularies to construct lexical representation vectors for each language, (3) cluster the lexical representation vectors using K-Means, assign vocabulary capacities for each cluster using the ALP, and then construct per-cluster vocabularies using the ULM algorithm, and (4) create the final multilingual vocabulary by taking the union of the vocabularies for each cluster.

Training monolingual SPMs To acquire the data for building the vocabulary, we perform sampling with temperature t = 2 to sample 1 billion lines of text from CC100 (up-sampling lower-resource and down-sampling data from high resource languages). Then, for each language in CC100, we train a language-specific sentencepiece model with a vocabulary size of 30,000 (per language) using this data.

<table><tr><td>Cluster</td><td> $|V^c|$ </td><td>Languages</td></tr><tr><td> $c_1$ </td><td>174,504</td><td>fa, pa, sa, ka, ur, lo, my, ne, am, te, my, th, ta, ko, bn, ml, he, sd, as, hi, km, gu, kn, si, yi, mr, ps, or, xh, ar, ug</td></tr><tr><td> $c_2$ </td><td>102,722</td><td>ja, zh-TW, zh-CN</td></tr><tr><td> $c_3$ </td><td>186,881</td><td>fi, sk, om, sw, ln, az, lg, uz, so, hy, ss, hu, la, ff, et, ta, wo, lv, ku, te, sc, el, pl, lt, tr</td></tr><tr><td> $c_4$ </td><td>110,148</td><td>pt, eu, gl, gn, it, ca, qu, es</td></tr><tr><td> $c_5$ </td><td>24,752</td><td>af, li, nl, fy</td></tr><tr><td> $c_6$ </td><td>19,801</td><td>hr, sl, bs</td></tr><tr><td> $c_7$ </td><td>101,485</td><td>bg, ky, uk, be, kk, sr, mk, ru, mn</td></tr><tr><td> $c_8$ </td><td>279,702</td><td>su, jv, tl, sv, tn, no, id, ig, bn, ns, mg, cs, ms, ro, ur, rm, ha, ga, ht, is, eo, gd, br, hi, en, cy, fr, vi, da, yo, de, sq</td></tr></table>

Table 1: Lexical clustering results for XLM-V with number of clusters k = 8 and a total vocabulary capacity of 1M.

![](images/c3f25b9a976f5deb0a7d678fdeb88ce795e81ee17ae0ffde629f46a7a0f49151.jpg)  
Figure 1: Similar to Chung et al. (2020), we also leverage the per-language sentencepiece vocabularies as a “lexical fingerprint” for clustering. However, instead of using binary vectors, we use the unigram log probability instead.

Constructing lexical fingerprints We then construct a vector representation of each language using the vocabularies of each language as shown in Figure 1. Unlike Chung et al. (2020), where a language is represented by a binary vector containing a 1 corresponding to each subword present in the vocabulary of that language, we instead use the negative log probability that each token appears in the respective language’s monolingual corpus. We hypothesize that weighting each token by its likelihood of occurring better represents the lexical fingerprint of a language.

Clustering and capacity allocation Next, we construct language clusters and train sentencepiece models for each cluster in order to discourage the vocabulary sharing between lexically dissimilar languages. Before training per-cluster sentencepiece models, we need to first decide on the vocabulary size, or vocabulary capacity, to allocate to each cluster. Unfortunately, we found that the method for assigning vocabulary capacities used by Chung et al. (2020) (i.e. proportionally to the set union of the per-language vocabularies in each cluster) resulted in several clusters with deficient vocabulary capacity. For example, cluster c<sub>2</sub> in Table 1 (a smaller cluster that contains lexically diverse languages: Chinese Simplified, Chinese Traditional, and Japanese), was assigned a capacity of just 28,593 tokens.

We instead use the per-language vocabulary capacity allocations (Zheng et al., 2021) optimized for the CC100 dataset. By doing so, the vocabulary capacity assigned to c was increased to 102,722. For each tail-end (low-resource) language that was not covered in Zheng et al. (2021), we allocate a 2,000 token vocabulary budget. Rather than use the vocabulary allocations directly, we take their relative values and rescale them to sum up to the vocabulary capacity of our choosing (e.g. 1M, 2M, etc.). Finally, we perform K-Means clustering with k = 8, based on experiments from Chung et al. (2020) showing that k = 8 results in the best performance on downstream tasks. We expect the ideal number of clusters to vary not based on the number of languages but rather on the identity of those languages and their respective similarities to one another.

The final vocabulary For each resulting cluster, we train per-cluster sentencepiece models and combine the vocabularies of each cluster into a single multilingual vocabulary. The final vocabulary consists of 901,629 tokens (remaining 98,371 tokens overlapped between the 8 clusters), meaning that on average over 90% of the tokens learned in each cluster are unique.

## 3.2 Training the model

To pretrain our model, we follow the same training procedure from XLM-R (Conneau et al., 2019). Specifically, we use the CC100 dataset with a sampling temperature of 0.3 to increase the amount of low- and medium-resource language examples seen during training. We use the Adam optimizer (Kingma and Ba, 2014) with the default $( \beta _ { 1 } , \beta _ { 2 } )$ and ϵ parameters of (0.9, 0.98) and 1e-6, respectively. We use a learning rate of 6e-4, a warmup of 15,000 steps, a batch size of 8,192 distributed across 256 A100 GPUs, and train for a total of 1.5M iterations. Each batch consists of examples concatenated up to the maximum sequence length of 512. We pretrain the model using the Masked Language Model (MLM) task (Devlin et al., 2018) with the standard masking rate of 15%.

Increasing the vocabulary size can significantly increase pretraining time due to the computationally intensive softmax layer. To address this, prior works have leveraged approximation tricks such as adaptive softmax (Baevski and Auli, 2018) and adaptive inputs (Joulin et al., 2017). However, we found that these tricks require non-trivial amounts of tuning and resulted in slower convergence and increased training instability. In this paper, we perform pretraining without any approximation tricks noting that this method may not be feasible when the vocabulary is scaled beyond 2M.<sup>1</sup>

## 4 Experiment setup

## 4.1 Baselines

Aside from training XLM-V, we also construct several baselines to compare our model against. To construct our baselines, we first create the respective vocabularies and then pretrain transformer encoders (12-layers, equivalent to XLM-R base) using these vocabularies. For the rest of the paper, we will use the following names to refer to the vocabulary and the model interchangeably.

XLM-R (250K) The XLM-R vocabulary is created using the same procedure from (Conneau et al., 2019) by applying the ULM algorithm described in Section 2 on a corpus of 1B lines of text sampled from CC100. The result is a multilingual vocabulary with 250,002 tokens. For our experiments, we simply re-use the publicly available XLM-R sentencepiece model and pretrained model checkpoint from fairseq (Ott et al., 2019).

XLM-R (1M) We construct a 1M token vocabulary by following the same approach as XLM-R (250K) with an increased vocabulary capacity.

Chung et al. (2020) (1M) We create a 1M token vocabulary using the lexical clustering approach from Chung et al. 2020 as described in Section 2.

## 4.2 Datasets

CC100 (Conneau et al., 2019) is a multilingual corpus created from one Common Crawl dump for English and twelve dumps for all other languages. The resulting corpus contains 2.5 TB of data split between 116 languages. We use this dataset exclusively for constructing vocabularies and pretraining our models.

FLoRes-200 (Goyal et al., 2022) is an evaluation corpus consisting of 3,001 sentences extracted from 842 English Wikipedia articles and covering a variety of different topics and domains. These sentences have been translated into 200 languages by professional translators through a carefully controlled process.

XNLI (Conneau et al., 2018) asks whether a premise sentence entails, contradicts, or is neutral toward a hypothesis sentence. Crowd-sourced English data is translated to 10 other languages by professional human translators and used for evaluation, while the Multi-Genre Natural Language Inference Corpus (MultiNLI) (Williams et al., 2018) data is used for training.

MLQA (Lewis et al., 2019) <sup>2</sup> is a QA evaluation dataset created by mining target language sentences that are parallel to sentences in English from Wikipedia, crowd-sourcing annotations in English, and translating the question and aligning the answer spans in one of the 6 target languages. It consists of over 12K QA instances in English and 5K in each other language. The training set of MLQA is SQuAD v1.1 (Rajpurkar et al., 2016).

XQuAD (Artetxe et al., 2019) translates the dev set of SQuAD v1.1 into 10 other languages through professional translators. The resulting dataset is used for evaluation. The training set of XQuAD is SQuAD v1.1.

TyDiQA-GoldP (Clark et al., 2020) is a question answering (QA) dataset covering 11 typologically diverse languages with 200K QA pairs. Questions in TyDiQA are written without seeing the answers leading to significantly less lexical overlap than XQuAD or MLQA. The languages of TyDiQA are selected to be diverse with regard to their typology. We use the gold passage version of the Typologically Diverse Question Answering dataset.

<table><tr><td>Model</td><td>XNLI Acc.</td><td>NER Acc.</td><td>MLQA EM / F1</td><td>TyDiQA EM / F1</td><td>XQuAD EM / F1</td><td>ANLI F1</td><td colspan="2">MNER Average F1</td></tr><tr><td>XLM</td><td>69.1</td><td>-</td><td>32.6 / 48.5</td><td>29.1 / 43.6</td><td>44.3 / 59.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>XLM-R</td><td>76.2</td><td>-</td><td>46.3 / 63.7</td><td>- / -</td><td>- / -</td><td>38.5</td><td>-</td><td>-</td></tr><tr><td>XLM-R reimpl.</td><td>74.9</td><td>61.3</td><td>46.7 / 64.4</td><td>38.3 / 56.0</td><td>56.0 / 71.3</td><td>39.6</td><td>20.9</td><td>55.5</td></tr><tr><td>XLM-V</td><td>76.0</td><td>64.7</td><td>47.7 / 66.0</td><td>39.7 / 56.9</td><td>56.3 / 71.9</td><td>45.4</td><td>32.1</td><td>59.0</td></tr></table>

Table 2: Overall results across multiple multilingual datasets comparing our model against the XLM and XLM-R baselines. All results are based on crosslingual transfer after fine-tuning on English data. We computed the average result using the accuracy or F1 of each task. “reimpl” is our re-implementation of finetuning, used by both XLM-R and XLM-V. Please refer to the appendix for specific hyperparameters to reproduce each result. EM stands for exact match. ANLI refers to AmericasNLI and MNER refers to MasakhaNER.

NER (Pan et al., 2017) consists of 48 languages and is based on the WikiAnn (PAN-X) dataset. Named entities were automatically annotated with LOC, PER, and ORG tags through knowledge base properties, crosslingual and anchor links, selftraining, and data selection. Similar to (Hu et al., 2020), we use the balanced dev and test splits from Rahimi et al. (2019).

Americas NLI (Ebrahimi et al., 2021) is an extension of XNLI to 10 indigenous languages of the Americas constructed by translating a subset of XNLI using human translators. These languages contain interesting linguistic features such as a rich system of applicative suffixes (Asháninka), directional verbs (Bribri), and nominal incorporation (Wixarika). Presently, these languages are written, spoken, and used in an official capacity by tens of thousands to several million people in Central and Southern America. The training set of Americas NLI is MultiNLI.

MasakhaNER (Adelani et al., 2021) is the first large, publicly available, and high-quality dataset for named entity recognition (NER) in ten African languages including Amharic, Hausa, Igbo, and others. The languages covered in this dataset have varied scripts and range from 4M to 98M speakers in regions across East, West, Central, and Northwest Africa.

## 5 Results

## 5.1 Comparisons using partial training

We first perform a study to measure the impact of our new vocabulary on downstream performance. Specifically, we pretrain a 12-layer transformer encoder model using Masked Language Modeling on the CC100 corpus for each baseline as well as for our proposed method. Because pretraining is expensive, we limit the batch size to 2,048 and the number of total steps to 300,000 for these experiments. The results in Figure 2 show that our model outperforms all baselines on XNLI including XLM-R (1M) by 1.34% and Chung et al. (2020) by 1.11% absolute accuracy.

![](images/b38974bd01f3ae00301bdf1e56e92e774b67d46a103a4330a7206e12343f55e9.jpg)

<details>
<summary>bar</summary>

| Sentencepiece Model | XNLI Accuracy (dev) |
| --- | --- |
| XLM-R (250k) | 70.13 |
| XLM-R (1M) | 71.78 |
| Chung et al. 2020 (1M) | 72.01 |
| XLM-V (1.5M) | 72.72 |
| XLM-V (1M) | 73.12 |
</details>

Figure 2: We compare the performance of the same model trained with different sentencepiece vocabularies. The models are all trained for 300K iterations with a batch size of 2,048 on the CC100 corpus.

## 5.2 Fully trained model

We evaluate an XLM-V (1M) model, trained on CC100 for 1.5M iterations with a batch size of 8,192, on several tasks including natural language inference (XNLI), question answering (MLQA,

<table><tr><td>Model</td><td>en</td><td>fr</td><td>es</td><td>de</td><td>el</td><td>bg</td><td>ru</td><td>tr</td><td>ar</td><td>vi</td><td>th</td><td>zh</td><td>hi</td><td>sw</td><td>ur</td><td>AVG</td></tr><tr><td colspan="17">Finetune multilingual model on English training set (Cross-lingual Transfer)</td></tr><tr><td>XLM-R reimpl.</td><td>85.4</td><td>78.5</td><td>79.1</td><td>77.7</td><td>76.1</td><td>78.1</td><td>76.3</td><td>73.9</td><td>72.3</td><td>75.6</td><td>73.0</td><td>74.9</td><td>70.5</td><td>65.8</td><td>66.5</td><td>74.9</td></tr><tr><td>XLM-V</td><td>85.6</td><td>79.6</td><td>79.5</td><td>78.4</td><td>76.9</td><td>79.6</td><td>76.6</td><td>74.0</td><td>73.1</td><td>76.2</td><td>73.0</td><td>75.1</td><td>72.0</td><td>70.5</td><td>69.4</td><td>76.0</td></tr><tr><td colspan="17">Finetune multilingual model on all training sets (Translate-Train-All)</td></tr><tr><td>XLM-R reimpl.</td><td>85.4</td><td>81.5</td><td>82.0</td><td>80.7</td><td>80.2</td><td>81.2</td><td>78.9</td><td>78.4</td><td>77.6</td><td>79.9</td><td>77.6</td><td>79.5</td><td>75.8</td><td>73.4</td><td>72.3</td><td>79.0</td></tr><tr><td>XLM-V</td><td>85.6</td><td>81.5</td><td>82.1</td><td>81.5</td><td>80.7</td><td>81.5</td><td>79.6</td><td>78.7</td><td>77.6</td><td>80.0</td><td>77.7</td><td>79.5</td><td>77.0</td><td>74.3</td><td>73.9</td><td>79.4</td></tr></table>

Table 3: XLM-V outperforms XLM-R on cross-lingual transfer on every language in XNLI with outsized improvements on the lower-resource languages, Swahili and Urdu. We observe similar improvements on translate-train-all. The model is trained for 12 epochs (2 epochs for translate-train-all) on 8 A100 GPUs with float16 precision. We use a learning rate of 7.5e-6 with a max sequence length of 256, a batch size of 16, no weight decay, and no warmup.

<table><tr><td>Model</td><td>aym</td><td>bzd</td><td>cni</td><td>gn</td><td>hch</td><td>nah</td><td>oto</td><td>quy</td><td>shp</td><td>tar</td><td>AVG</td></tr><tr><td>XLM-R reimpl.</td><td>36.6</td><td>39.6</td><td>40.5</td><td>41.6</td><td>38.8</td><td>40.2</td><td>39.4</td><td>38.7</td><td>42.7</td><td>37.6</td><td>39.6</td></tr><tr><td>XLM-V</td><td>39.9</td><td>41.5</td><td>41.7</td><td>58.8</td><td>40.7</td><td>44.7</td><td>42.1</td><td>56.9</td><td>46.5</td><td>41.2</td><td>45.4</td></tr><tr><td>Tok. Length (rel.)</td><td>-10.8%</td><td>-11.6%</td><td>-11.9%</td><td>-16.5%</td><td>-6.5%</td><td>-10.7%</td><td>-8.4%</td><td>-18.4%</td><td>-10.9%</td><td>-9.1%</td><td>-11.5%</td></tr></table>

Table 4: We show the zero-shot cross-lingual transfer results on Americas NLI (trained on English and evaluated on the unseen languages). Our model, XLM-V, outperforms XLM-R by a wide margin with outsized improvements on Quechua and Guaraní. Tok. Length (rel.) refers to the relative difference in the average number of tokens (post-tokenization) between XLM-R and XLM-V. XLM-V consistently outputs shorter sequences post-tokenization. The model is trained for 12 epochs on 8 A100 GPUs with float16 precision. We use a learning rate of 7.5e-6 with a max sequence length of 256, batch size of 16, no weight decay, and no warmup.

TyDiQA, and XQuAD), named enitity recognition (WikiAnn), and low resource language tasks (AmericasNLI, MasakhaNER). All tasks leverage crosslingual transfer from English-only finetuning and are trained using float16 precision with the AdamW optimizer (Loshchilov and Hutter, 2017). We use hyperparameters selected based on the best English performance on the dev set,<sup>3</sup> and finally evaluate on the test set. We compile all of our results in Table 2 for XLM-V and XLM-R. We also include results for XLM (Lample and Conneau, 2019) for additional context.

Table 2 shows that XLM-V outperforms our reimplementation of XLM-R on all datasets by an average of 3.5 points absolute (we compute the average result using either the accuracy or F1 of each task). In Table 3, we show that XLM-V outperforms XLM-R on all languages in cross-lingual transfer (training on English and evaluating on other languages) with similar improvements on translate-train-all (finetuning the model on both the English and translated training sets). In particular, we find that XLM-V consistently outperforms XLM-R on low-resource languages. For example, in Table 3, we observe a 4.7% and 2.9% accuracy improvement on Swahili (sw) and Urdu (ur) on XNLI. Similarly, we show an average gain of 11.2% F1 on MasakhaNER, a low-resource African language NER dataset.

In Table 4 we show that XLM-V not only consistently outperforms XLM-R on Americas NLI in zero-shot crosslingual transfer but is able to obtain 18.2% absolute F1 improvement on Quechua (quy) and 17.2% absolute improvement on Guaraní (gn). Interestingly, Quechua and Guaraní are also the two languages with the largest relative drop in average token count per sentence – suggesting that these languages are over-tokenized by XLM-R.

## 6 Analysis

## 6.1 The Zipf ceiling

We explored training models with vocabulary sizes greater than 1M tokens but found that these models perform comparatively worse on downstream tasks. We visualize the diminishing utility of increasing the vocabulary size in Figure 3. Specifically, we create vocabularies with 500K, 1M, 1.5M, and 2M tokens using our methodology. Then, we use these vocabularies to tokenize the FLoRes-200 dataset. For vocabulary sizes of 500K, 1M, and 2M, we find that 99% of the content is covered by just 140,337, 197,817, and 243,832 unique tokens, respectively.

<table><tr><td>Language</td><td>Tokenizer</td><td>Tokenized Output</td></tr><tr><td rowspan="5">zh</td><td>Original Sentence</td><td>剑桥大学本科生和研究生</td></tr><tr><td>XLM-R (250K)</td><td>[&#x27;剑&#x27;, &#x27;桥&#x27;, &#x27;大学&#x27;, &#x27;本科&#x27;, &#x27;生&#x27;, &#x27;和&#x27;, &#x27;研究生&#x27;]</td></tr><tr><td>XLM-R (1M)</td><td>[&#x27;剑&#x27;, &#x27;桥&#x27;, &#x27;大学&#x27;, &#x27;本&#x27;, &#x27;科&#x27;, &#x27;生&#x27;, &#x27;和&#x27;, &#x27;研究&#x27;, &#x27;生&#x27;]</td></tr><tr><td>Chung et al. (2020) (1M)</td><td>[&#x27;剑桥&#x27;, &#x27;大学本科&#x27;, &#x27;生&#x27;, &#x27;和&#x27;, &#x27;研究生&#x27;]</td></tr><tr><td>XLM-V (1M)</td><td>[&#x27;剑桥大学&#x27;, &#x27;本科生&#x27;, &#x27;和&#x27;, &#x27;研究生&#x27;]</td></tr><tr><td rowspan="5">en, fr, es</td><td>Original Sentence</td><td>narcolepsy narcolepsie narcolepsia</td></tr><tr><td>XLM-R (250K)</td><td>[&#x27;na&#x27;, &#x27;r&#x27;, &#x27;cole&#x27;, &#x27;psy&#x27;] [&#x27;na&#x27;, &#x27;r&#x27;, &#x27;cole&#x27;, &#x27;psi&#x27;, &#x27;e&#x27;] [&#x27;na&#x27;, &#x27;r&#x27;, &#x27;cole&#x27;, &#x27;psi&#x27;, &#x27;a&#x27;]</td></tr><tr><td>XLM-R (1M)</td><td>[&#x27;na&#x27;, &#x27;rcole&#x27;, &#x27;psy&#x27;] [&#x27;na&#x27;, &#x27;rcole&#x27;, &#x27;psie&#x27;] [&#x27;na&#x27;, &#x27;rcole&#x27;, &#x27;psia&#x27;]</td></tr><tr><td>Chung et al. (2020) (1M)</td><td>[&#x27;na&#x27;, &#x27;rcole&#x27;, &#x27;psy&#x27;] [&#x27;narco&#x27;, &#x27;lepsi&#x27;, &#x27;e&#x27;] [&#x27;na&#x27;, &#x27;rcole&#x27;, &#x27;psia&#x27;]</td></tr><tr><td>XLM-V (1M)</td><td>[&#x27;narco&#x27;, &#x27;le&#x27;, &#x27;psy&#x27;] [&#x27;narco&#x27;, &#x27;lepsi&#x27;, &#x27;e&#x27;] [&#x27;narco&#x27;, &#x27;lepsi&#x27;, &#x27;a&#x27;]</td></tr><tr><td rowspan="5">de</td><td>Original Sentence</td><td>Betäubungsmittelverschreibungsverordnung</td></tr><tr><td>XLM-R (250K)</td><td>[&#x27;Be&#x27;, &#x27;tä&#x27;, &#x27;ub&#x27;, &#x27;ungs&#x27;, &#x27;mittel&#x27;, &#x27;ver&#x27;, &#x27;schreibung&#x27;, &#x27;s&#x27;, &#x27;ver&#x27;, &#x27;ordnung&#x27;]</td></tr><tr><td>XLM-R (1M)</td><td>[&#x27;Be&#x27;, &#x27;tä&#x27;, &#x27;ub&#x27;, &#x27;ungsmittel&#x27;, &#x27;ver&#x27;, &#x27;schreibung&#x27;, &#x27;s&#x27;, &#x27;verordnung&#x27;]</td></tr><tr><td>Chung et al. (2020) (1M)</td><td>[&#x27;Bet&#x27;, &#x27;äub&#x27;, &#x27;ungsmittel&#x27;, &#x27;ver&#x27;, &#x27;schreibung&#x27;, &#x27;sverordnung&#x27;]</td></tr><tr><td>XLM-V (1M)</td><td>[&#x27;Bet&#x27;, &#x27;äub&#x27;, &#x27;ungsmittel&#x27;, &#x27;ver&#x27;, &#x27;schreibung&#x27;, &#x27;sverordnung&#x27;]</td></tr></table>

Table 5: We provide examples comparing tokenization using the XLM-V vocabulary against baselines. We find that our sentencepiece model reduces overtokenization and can be surprisingly good at splitting sentences into pseudo-meaningful segments out-of-the-box.

<table><tr><td>Model</td><td>vi</td><td>zh</td><td>fr</td><td>de</td><td>en</td><td>xho</td><td>tel</td><td>AVG</td></tr><tr><td>XLM-R (250K)</td><td>34.3</td><td>28.5</td><td>37.5</td><td>33.9</td><td>29.1</td><td>43.9</td><td>38.8</td><td>43.6</td></tr><tr><td>XLM-R (1M)</td><td>33.5</td><td>31.7</td><td>34.8</td><td>31</td><td>26.8</td><td>40.3</td><td>41.6</td><td>41.4</td></tr><tr><td>Chung et al. (2020) (1M)</td><td>32.7</td><td>24.4</td><td>32.9</td><td>29.1</td><td>27.8</td><td>29.2</td><td>25.7</td><td>37.7</td></tr><tr><td>XLM-V (1M)</td><td>32.4</td><td>23.4</td><td>32.2</td><td>28.3</td><td>25.5</td><td>37.4</td><td>33.2</td><td>38.6</td></tr></table>

Table 6: Average number of tokens after tokenization on the FLoRes-200 dataset for several high, medium, and low resource languages. AVG denotes the average tokenized lengths per sentence across all 200 languages in Flores-200.

![](images/32e087f199f55e297386bb380cce495c9832affea9fc510d6c523c21889528c0.jpg)

<details>
<summary>scatter</summary>

| Token Index (sorted by frequency) | 2M Token Vocab (Log Frequency) | 1.5M Token Vocab (Log Frequency) | 1M Token Vocab (Log Frequency) | 500k Token Vocab (Log Frequency) |
| --- | --- | --- | --- | --- |
| ~0 | ~12.3 | ~12.3 | ~12.3 | ~12.6 |
| ~10000 | ~4.5 | ~4.5 | ~4.5 | ~4.5 |
| ~50000 | ~2.8 | ~2.8 | ~2.8 | ~2.8 |
| ~100000 | ~1.9 | ~1.9 | ~1.9 | ~1.9 |
| ~150000 | ~1.4 | ~1.4 | ~1.4 | ~1.4 |
| ~200000 | ~1.1 | ~1.1 | ~1.1 | ~1.1 |
| ~250000 | ~0.7 | ~0.7 | ~0.7 | ~0.7 |
| ~300000 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
</details>

Figure 3: We compare the token utilization of each sentencepiece vocabulary on the FLoRes-200 dataset. We see diminishing returns as the size of the vocabulary is increased beyond 1M tokens.

We hypothesize that since the Unigram LM (Kudo and Richardson, 2018) algorithm used to construct the vocabulary iteratively prunes a large initial set, as discussed in Section 2, further expanding the vocabulary is equivalent to inheriting tokens from the long tail of a Zipfian distribution. These token embeddings are problematic because they are trained on significantly less data during the course of MLM pretraining and will learn sub-optimal representations as a result. As a consequence, vocabularies past a certain size will cease to improve model performance and can potentially degrade it. A clear example of this is shown in Figure 2 where our model with a 1M token vocabulary outperforms its 1.5M token counterpart trained using an equivalent amount of data.

## 6.2 Qualitative improvements in tokenization

Table 5 shows a few tokenized examples from Chinese (zh), English (en), French (fr), Spanish (es), and German (de). For languages in the same cluster (en, fr, es), our method can separate shared roots (e.g. narco) from the same word in different languages. Notably, our method demonstrates a surprising ability to segment Chinese out-of-the-box, parsing out individual entities in the original phrase. For example, the XLM-V tokenizer is able to meaningfully break down <sub>the phrase</sub> 剑 桥 大 学 本 科 生 和 研 究 生<sub>, trans-</sub> lated as Cambridge University undergraduates and postgraduates. Specifically, the output of the <sub>XLM-V tokenizer is</sub> 剑桥大学<sub>(Cambridge Univer-</sub> <sub>sity),</sub> 本科生<sub>(undergraduates),</sub> 和<sub>(and), and</sub> 研究 生(postgraduates). Qualitatively, our tokenizer frequently performs tokenizations that are semantically meaningful, one possible contributor to the improved downstream performance.

## 6.3 Over-tokenization

Representing input data with fewer tokens can speed up inference, allow the model to make use of longer context, and help with over-tokenization for low-resource languages (Rust et al., 2020). Table 6 shows the average number of resulting tokens (post-tokenization) for several languages in FLoRes-200. On average, the XLM-V tokenizer returns fewer tokens for high and medium resource languages while Chung et al. (2020) returns the fewest tokens for low-resource languages. Overall, XLM-V returns 11.5% fewer tokens compared to the baseline XLM-R tokenizer, meaning that input sequences are on average 11.5% shorter.

![](images/110b2575570591e42d297b0cb8629ed881e2a8029cc8bb9752d9f06657ce8996.jpg)

<details>
<summary>scatter</summary>

| Number of parameters (in millions) | Time per iteration (seconds) |
| --- | --- |
| 250K | ~0.15 |
| 500K | ~0.16 |
| 1M | ~0.19 |
| 1.5M | ~0.22 |
</details>

Figure 4: We track training speed vs. vocabulary size using a typical training setup on XNLI: one A100 GPU, a batch size of 16, sequence length of 128, and float16 precision. The text above each point denotes the vocabulary size.

## 6.4 Speed vs. size

For XLM-R, which has a vocabulary size of 250K tokens, the vocabulary embedding matrix contains 77% of the model’s trainable parameters. For XLM-V, the 1M token vocabulary accounts for 93% of the model’s trainable parameters. While scaling the vocabulary can markedly increase the number of trainable parameters in a model, we can treat it as an efficient form of conditional compute (Bengio et al., 2015): only a small fraction of the embedding matrix is used for any given input. We illustrate the relationship between the vocabulary size and training speed in Figure 4. By increasing the vocabulary from 250K to 1M tokens, we can increase the number of trainable parameters by 3.3x with just a 25% increase in training time.

## 7 Related work

## 7.1 Vocabulary-free models

In recent years, vocabulary-free models like ByT4 (Xue et al., 2022) and CANINE (Clark et al., 2022) have demonstrated on-par or better performance compared to their subword tokenizationbased counterparts. However, one consistent drawback of these models is slower training and inference speed. For example, ByT5 is 6.4 to 9.5 times slower than mT5 (Xue et al., 2020) on classification tasks like XNLI. CANINE fares better, leveraging optimizations like lower input character dimensions and heavy down sampling, but still remains approximately 1.6 times slower than a comparable BERT baseline. On the other hand, simply using a larger sentencepiece vocabulary can improve downstream performance, increase the capacity of the model, and reduce the over-tokenization and coverage of low-resource languages all with a smaller impact on inference latency. We believe that both directions are useful areas of research and can be explored simultaneously.

## 7.2 Building larger vocabularies

Prior work on vocabulary expansion (Wang et al., 2019) sought to augment the vocabulary of existing models to address out-of-vocabulary (OOV) problems in multilingual settings. While these results are potentially useful in augmenting subword models like BERT, sentencepiece models by nature encounter significantly fewer OOVs.

More recent work on building larger vocabularies (Chung et al., 2020; Zheng et al., 2021) leverage tricks like lexical clustering and more principled methods for vocabulary allocation have tackled issues with over-tokenization and vocabulary coverage for low-resource languages. While compelling, these works are unfortunately limited by data (the models are trained on Wikipedia, a relatively small pretraining corpus) and scale (the largest vocabulary explored was 500K, only twice the size of the vocabulary in XLM-R). As such, the resulting models significantly under-perform the public XLM-R baseline. Our work seeks to combine and improve upon existing methods for building large-scale vocabularies, pretrain with substantially bigger datasets, and explore vocabularies of 1M tokens and beyond.

## 8 Conclusion

In this paper, we presented XLM-V, a multilingual language model with a 1M token vocabulary. We showed that our model outperforms XLM-R, has outsized gains on tasks in low-resource languages, results in semantically meaningful tokenizations, reduces average sequence length, and serves as an efficient form of conditional compute. In the future, we would like to further investigate the Zipf ceiling discussed in Section 6 by increasing the vocabulary beyond 2M tokens while also using more data. Another possible direction for future work is to explore larger multilingual vocabularies for autoregressive language models. Finally, further exploration with different clustering methods such as hierarchical clustering may prove both interesting and effective.

## Limitations

While the strengths of XLM-V are clear, there remains several scalability issues that are notable. First, while scaling the vocabulary is an efficient form of conditional compute, it can result in increased pre-training times due to the computational complexity of the softmax over the entire vocabulary. We believe these issues can be solved by adopting approximation techniques like adaptive softmax (Joulin et al., 2017) and adaptive inputs (Baevski and Auli, 2018). Additionally, scaling the vocabulary can also significantly increase the memory footprint of a model. However, we believe memory-related issues become less of a problem as we begin to work with larger models, where the number of non-embedding parameters vastly outweigh the size of the vocabulary embedding matrix.

## References

David Ifeoluwa Adelani, Jade Abbott, Graham Neubig, Daniel D’souza, Julia Kreutzer, Constantine Lignos, Chester Palen-Michel, Happy Buzaaba, Shruti Rijhwani, Sebastian Ruder, et al. 2021. Masakhaner: named entity recognition for african languages. Transactions of the Association for Computational Linguistics, 9:1116–1131.  
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2019. On the cross-lingual transferability of monolingual representations. arXiv preprint arXiv:1910.11856.  
Alexei Baevski and Michael Auli. 2018. Adaptive input  
representations for neural language modeling. arXiv preprint arXiv:1809.10853.  
Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. 2015. Conditional computation in neural networks for faster models. arXiv preprint arXiv:1511.06297.  
Hyung Won Chung, Dan Garrette, Kiat Chuan Tan, and Jason Riesa. 2020. Improving multilingual models with language-clustered vocabularies. EMNLP.  
Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Associationfor Computational Linguistics, 8:454–470.  
Jonathan H Clark, Dan Garrette, Iulia Turc, and John Wieting. 2022. Canine: Pre-training an efficient tokenization-free encoder for language representation. Transactions ofthe Associationfor Computational Linguistics, 10:73–91.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116.  
Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. arXiv preprint arXiv:1809.05053.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.  
Abteen Ebrahimi, Manuel Mager, Arturo Oncevay, Vishrav Chaudhary, Luis Chiruzzo, Angela Fan, John Ortega, Ricardo Ramos, Annette Rios, Ivan Vladimir, et al. 2021. Americasnli: Evaluating zero-shot natural language understanding of pretrained multilingual models in truly low-resource languages. arXiv preprint arXiv:2104.08726.  
Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, and Alexis Conneau. 2021. Larger-scale transformers for multilingual masked language modeling. arXiv preprint arXiv:2105.00572.  
Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzman, and Angela Fan. 2022. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions ofthe Associationfor Computational Linguistics, 10:522–538.  
Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In International Conference on Machine Learning, pages 4411–4421. PMLR.  
Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou, et al. 2017. Efficient softmax approximation for gpus. In International conference on machine learning, pages 1302–1310. PMLR.  
Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.  
Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226.  
Guillaume Lample and Alexis Conneau. 2019. Crosslingual language model pretraining. arXiv preprint arXiv:1901.07291.  
Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian˘ Riedel, and Holger Schwenk. 2019. Mlqa: Eval uating cross-lingual extractive question answering. arXiv preprint arXiv:1910.07475.  
Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. 2021. Few-shot learning with multilingual language models. arXiv preprint arXiv:2112.10668.  
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions ofthe Associationfor Computational Linguistics, 8:726–742.  
Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.  
Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. Fairseq: A fast, extensible toolkit for sequence modeling. arXiv preprint arXiv:1904.01038.  
Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. Cross-lingual name tagging and linking for 282 languages. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1946–1958.  
Afshin Rahimi, Yuan Li, and Trevor Cohn. 2019. Massively multilingual transfer for ner. arXiv preprint arXiv:1902.00193.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250.  
Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian´ Ruder, and Iryna Gurevych. 2020. How good is your tokenizer? on the monolingual performance of multilingual language models. arXiv preprint arXiv:2012.15613.  
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman´ Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.  
Andrew Viterbi. 1967. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE transactions on Information Theory, 13(2):260–269.  
Hai Wang, Dian Yu, Kai Sun, Janshu Chen, and Dong Yu. 2019. Improving pre-trained multilingual models with vocabulary expansion. arXiv preprint arXiv:1909.12440.  
Wikipedia. 2023. Table of general standard chinese characters — wikipedia, the free encyclopedia. http://en.wikipedia.org/w/index.php? title=Table%20of%20General%20Standard% 20Chinese%20Characters&oldid=1123968033. [Online; accessed 05-January-2023].  
Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018. The multi-genre nli corpus.  
Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022. Byt5: Towards a token-free future with pre-trained byte-to-byte models. Transactions ofthe Associationfor Computational Linguistics, 10:291–306.  
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2020. mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934.  
Bo Zheng, Li Dong, Shaohan Huang, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, and Furu Wei. 2021. Allocating large vocabulary capacity for crosslingual language model pre-training. EMNLP.

## A Appendix

## B Appendix

We show the per-language results for each task we tested on. For the sake of reproducibility, we also provide the hyperparameters that we used to finetune the model for each task.

<table><tr><td>Model</td><td>en</td><td>es</td><td>de</td><td>ar</td><td>hi</td><td>vi</td><td>zh</td><td>AVG</td></tr><tr><td>XLM-R reimpl.</td><td>65.9 / 78.7</td><td>50.4 / 67.7</td><td>47.6 / 62.2</td><td>36.8 / 55.8</td><td>42.1 / 59.3</td><td>45.2 / 65.2</td><td>37.8 / 60.7</td><td>46.5 / 64.2</td></tr><tr><td>XLM-V</td><td>67.5 / 80.4</td><td>51.1 / 69.4</td><td>49.8 / 64.3</td><td>38.1 / 58.2</td><td>44.5 / 62.7</td><td>46.4 / 67.2</td><td>36.3 / 59.9</td><td>47.7 / 66.0</td></tr></table>

Table 7: MLQA results (EM/F1). The model is trained for 2 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.

<table><tr><td>Model</td><td>en</td><td>ar</td><td>bn</td><td>fi</td><td>id</td><td>ko</td><td>ru</td><td>sw</td><td>te</td><td>AVG</td></tr><tr><td>XLM-R reimpl.</td><td>55.5/68.6</td><td>42.0/63.9</td><td>18.6/37.6</td><td>42.8/61.6</td><td>54.7/73.1</td><td>23.6/39.9</td><td>31.5/59.9</td><td>30.7/54.1</td><td>27.5/44.3</td><td>36.3/55.9</td></tr><tr><td>XLM-V</td><td>52.3/66.9</td><td>45.4/65.5</td><td>27.4/42.7</td><td>46.0/63.6</td><td>56.1/72.3</td><td>22.8/37.4</td><td>31.5/59.3</td><td>43.1/61.4</td><td>32.4/43.2</td><td>39.7/56.9</td></tr></table>

Table 8: TyDiQA-GoldP results (EM/F1). The model is trained for 8 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.

<table><tr><td>Model</td><td>en</td><td>es</td><td>de</td><td>el</td><td>ru</td><td>tr</td></tr><tr><td>XLM-R reimpl.</td><td>72.1 / 83.5</td><td>58.5 / 76.5</td><td>57.6 / 73.0</td><td>55.4 / 72.2</td><td>56.6 / 73.1</td><td>52.2 / 68.3</td></tr><tr><td>XLM-V</td><td>72.9 / 84.2</td><td>60.3 / 78.1</td><td>57.3 / 75.1</td><td>53.5 / 72.4</td><td>56.0 / 73.2</td><td>51.8 / 67.5</td></tr><tr><td></td><td>ar</td><td>vi</td><td>th</td><td>zh</td><td>hi</td><td>AVG</td></tr><tr><td>XLM-R reimpl.</td><td>49.2 / 65.9</td><td>53.5 / 72.9</td><td>55.7 / 66.3</td><td>55.5 / 65.3</td><td>49.8 / 57.7</td><td>56.0 / 71.3</td></tr><tr><td>XLM-V</td><td>51.2 / 67.5</td><td>53.7 / 73.1</td><td>56.9 / 67.0</td><td>53.5 / 63.1</td><td>51.9 / 69.4</td><td>56.3 / 71.9</td></tr></table>

Table 9: XQuAD Results (EM/F1). The model is trained for 2 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.

<table><tr><td>Model</td><td>ro</td><td>gu</td><td>pa</td><td>lt</td><td>az</td><td>uk</td><td>pl</td><td>qu</td><td>hu</td><td>fi</td><td>et</td><td>tr</td><td>kk</td><td>zh</td><td>my</td><td>yo</td><td>sw</td></tr><tr><td>XLM-R reimpl.</td><td>73.5</td><td>62.9</td><td>53.6</td><td>72.7</td><td>61.0</td><td>72.4</td><td>77.5</td><td>60.4</td><td>75.8</td><td>74.4</td><td>71.2</td><td>75.4</td><td>42.2</td><td>25.3</td><td>48.9</td><td>33.6</td><td>66.3</td></tr><tr><td>XLM-V</td><td>73.8</td><td>66.4</td><td>48.7</td><td>75.6</td><td>66.7</td><td>65.7</td><td>79.5</td><td>70.0</td><td>79.5</td><td>78.7</td><td>75.0</td><td>77.3</td><td>50.4</td><td>30.2</td><td>61.5</td><td>54.2</td><td>72.4</td></tr><tr><td></td><td>th</td><td>ko</td><td>ka</td><td>ja</td><td>ru</td><td>bg</td><td>es</td><td>pt</td><td>it</td><td>fr</td><td>fa</td><td>ur</td><td>mr</td><td>hi</td><td>bn</td><td>el</td><td>de</td></tr><tr><td>XLM-R reimpl.</td><td>5.2</td><td>49.4</td><td>65.4</td><td>21.0</td><td>63.1</td><td>76.1</td><td>70.2</td><td>77.0</td><td>76.9</td><td>76.5</td><td>44.6</td><td>51.4</td><td>61.5</td><td>67.2</td><td>69.0</td><td>73.8</td><td>74.4</td></tr><tr><td>XLM-V</td><td>3.3</td><td>53.0</td><td>69.5</td><td>22.4</td><td>68.1</td><td>79.8</td><td>74.5</td><td>80.5</td><td>78.7</td><td>77.6</td><td>50.6</td><td>48.9</td><td>59.8</td><td>67.3</td><td>72.6</td><td>76.7</td><td>76.8</td></tr><tr><td></td><td>en</td><td>nl</td><td>af</td><td>te</td><td>ta</td><td>ml</td><td>eu</td><td>tl</td><td>ms</td><td>jv</td><td>id</td><td>vi</td><td>he</td><td>ar</td><td colspan="3">AVG</td></tr><tr><td>XLM-R reimpl.</td><td>83.0</td><td>80.0</td><td>75.83</td><td>49.2</td><td>56.3</td><td>61.9</td><td>57.2</td><td>69.8</td><td>68.3</td><td>59.4</td><td>48.6</td><td>67.7</td><td>53.2</td><td>43.8</td><td colspan="3">61.3</td></tr><tr><td>XLM-V</td><td>83.4</td><td>81.4</td><td>78.3</td><td>51.8</td><td>54.9</td><td>63.1</td><td>67.1</td><td>75.6</td><td>70.0</td><td>67.5</td><td>52.6</td><td>67.1</td><td>60.1</td><td>45.8</td><td colspan="3">64.7</td></tr></table>

Table 10: NER Results. The model is trained for 10 epochs on a single A100 GPU with float16 precision. We use a learning rate of 2e-5 with a max sequence length of 128, batch size of 32, no weight decay, and no warmup.

<table><tr><td>Model</td><td>amh</td><td>hau</td><td>ibo</td><td>kin</td><td>lug</td><td>luo</td><td>pcm</td><td>swa</td><td>wol</td><td>yor</td><td>AVG</td></tr><tr><td>XLM-R reimpl.</td><td>25.1</td><td>43.5</td><td>11.6</td><td>9.4</td><td>9.5</td><td>8.4</td><td>36.8</td><td>48.9</td><td>5.3</td><td>10.0</td><td>20.9</td></tr><tr><td>XLM-V</td><td>20.6</td><td>35.9</td><td>45.9</td><td>25.0</td><td>48.7</td><td>10.4</td><td>38.2</td><td>44.0</td><td>16.7</td><td>35.8</td><td>32.1</td></tr></table>

Table 11: We show the zero-shot cross-lingual transfer results on MasakhaNER (trained on English and evaluated on the unseen languages). The model is trained for 10 epochs on a single A100 GPU with float16 precision. We use a learning rate of 2e-5 with a max sequence length of 128, batch size of 32, no weight decay, and no warmup.