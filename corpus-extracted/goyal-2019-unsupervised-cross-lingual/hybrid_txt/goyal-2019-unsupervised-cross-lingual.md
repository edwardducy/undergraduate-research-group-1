# Unsupervised Cross-lingual Representation Learning at Scale Alexis Conneau<sup>∗</sup> Kartikay Khandelwal

Naman Goyal Vishrav Chaudhary Guillaume Wenzek Francisco Guzman´

Edouard Grave Myle Ott Luke Zettlemoyer Veselin Stoyanov

## Facebook AI

## Abstract

This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks. We train a Transformerbased masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6% average accuracy on XNLI, +13% average F1 score on MLQA, and +2.4% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7% in XNLI accuracy for Swahili and 11.4% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing perlanguage performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code, data and models publicly available.<sup>1</sup>

## 1 Introduction

The goal of this paper is to improve cross-lingual language understanding (XLU), by carefully studying the effects of training unsupervised crosslingual representations at a very large scale. We present XLM-R a transformer-based multilingual masked language model pre-trained on text in 100 languages, which obtains state-of-the-art performance on cross-lingual classification, sequence labeling and question answering.

Multilingual masked language models (MLM) like mBERT (Devlin et al., 2018) and XLM (Lample and Conneau, 2019) have pushed the stateof-the-art on cross-lingual understanding tasks by jointly pretraining large Transformer models (Vaswani et al., 2017) on many languages. These models allow for effective cross-lingual transfer, as seen in a number of benchmarks including cross-lingual natural language inference (Bowman et al., 2015; Williams et al., 2017; Conneau et al., 2018), question answering (Rajpurkar et al., 2016; Lewis et al., 2019), and named entity recognition (Pires et al., 2019; Wu and Dredze, 2019). However, all of these studies pre-train on Wikipedia, which provides a relatively limited scale especially for lower resource languages.

In this paper, we first present a comprehensive analysis of the trade-offs and limitations of multilingual language models at scale, inspired by recent monolingual scaling efforts (Liu et al., 2019). We measure the trade-off between high-resource and low-resource languages and the impact of language sampling and vocabulary size. The experiments expose a trade-off as we scale the number of languages for a fixed model capacity: more languages leads to better cross-lingual performance on low-resource languages up until a point, after which the overall performance on monolingual and cross-lingual benchmarks degrades. We refer to this tradeoff as the curse of multilinguality, and show that it can be alleviated by simply increasing model capacity. We argue, however, that this remains an important limitation for future XLU systems which may aim to improve performance with more modest computational budgets.

Our best model XLM-RoBERTa (XLM-R) outperforms mBERT on cross-lingual classification by up to 23% accuracy on low-resource languages. It outperforms the previous state of the art by 5.1% average accuracy on XNLI, 2.42% average F1-score on Named Entity Recognition, and 9.1% average F1-score on cross-lingual Question Answering. We also evaluate monolingual fine tuning on the GLUE and XNLI benchmarks, where XLM-R obtains results competitive with state-of-the-art monolingual models, including RoBERTa (Liu et al., 2019). These results demonstrate, for the first time, that it is possible to have a single large model for all languages, without sacrificing per-language performance. We will make our code, models and data publicly available, with the hope that this will help research in multilingual NLP and low-resource language understanding.

## 2 Related Work

From pretrained word embeddings (Mikolov et al., 2013b; Pennington et al., 2014) to pretrained contextualized representations (Peters et al., 2018; Schuster et al., 2019) and transformer based language models (Radford et al., 2018; Devlin et al., 2018), unsupervised representation learning has significantly improved the state of the art in natural language understanding. Parallel work on cross-lingual understanding (Mikolov et al., 2013a; Schuster et al., 2019; Lample and Conneau, 2019) extends these systems to more languages and to the cross-lingual setting in which a model is learned in one language and applied in other languages.

Most recently, Devlin et al. (2018) and Lample and Conneau (2019) introduced mBERT and XLM - masked language models trained on multiple languages, without any cross-lingual supervision. Lample and Conneau (2019) propose translation language modeling (TLM) as a way to leverage parallel data and obtain a new state of the art on the cross-lingual natural language inference (XNLI) benchmark (Conneau et al., 2018). They further show strong improvements on unsupervised machine translation and pretraining for sequence generation. Wu et al. (2019) shows that monolingual BERT representations are similar across languages, explaining in part the natural emergence of multilinguality in bottleneck architectures. Separately, Pires et al. (2019) demonstrated the effectiveness of multilingual models like mBERT on sequence labeling tasks. Huang et al. (2019) showed gains over XLM using cross-lingual multi-task learning, and Singh et al. (2019) demonstrated the efficiency of cross-lingual data augmentation for cross-lingual NLI. However, all of this work was at a relatively modest scale, in terms of the amount of training data, as compared to our approach.

The benefits of scaling language model pretraining by increasing the size of the model as well as the training data has been extensively studied in the literature. For the monolingual case, Jozefowicz et al. (2016) show how large-scale LSTM models can obtain much stronger performance on language modeling benchmarks when trained on billions of tokens. GPT (Radford et al., 2018) also highlights the importance of scaling the amount of data and RoBERTa (Liu et al., 2019) shows that training BERT longer on more data leads to significant boost in performance. Inspired by RoBERTa, we show that mBERT and XLM are undertuned, and that simple improvements in the learning procedure of unsupervised MLM leads to much better performance. We train on cleaned CommonCrawls (Wenzek et al., 2019), which increase the amount of data for low-resource languages by two orders of magnitude on average. Similar data has also been shown to be effective for learning high quality word embeddings in multiple languages (Grave et al., 2018).

Several efforts have trained massively multilingual machine translation models from large parallel corpora. They uncover the high and low resource trade-off and the problem of capacity dilution (Johnson et al., 2017; Tan et al., 2019). The work most similar to ours is Arivazhagan et al. (2019), which trains a single model in 103 languages on over 25 billion parallel sentences. Siddhant et al. (2019) further analyze the representations obtained by the encoder of a massively multilingual machine translation system and show that it obtains similar results to mBERT on cross-lingual NLI. Our work, in contrast, focuses on the unsupervised learning of cross-lingual representations and their transfer to discriminative tasks.

## 3 Model and Data

In this section, we present the training objective, languages, and data we use. We follow the XLM approach (Lample and Conneau, 2019) as closely as possible, only introducing changes that improve performance at scale.

Masked Language Models. We use a Transformer model (Vaswani et al., 2017) trained with the multilingual MLM objective (Devlin et al., 2018; Lample and Conneau, 2019) using only monolingual data. We sample streams of text from each language and train the model to predict the masked tokens in the input. We apply subword tokenization directly on raw text data using Sentence Piece (Kudo and Richardson, 2018) with a unigram language model (Kudo, 2018). We sample batches from different languages using the same sampling distribution as Lample and Conneau (2019), but with α = 0.3. Unlike Lample and Conneau (2019), we do not use language embeddings, which allows our model to better deal with code-switching. We use a large vocabulary size of 250K with a full softmax and train two different models: XLM-R (L = 12, H = 768, A = 12, 270M params) and XLM-R (L = 24, H = 1024, A = 16, 550M params). For all of our ablation studies, we use a BERT architecture with a vocabulary of 150K tokens. Appendix B goes into more details about the architecture of the different models referenced in this paper.

![](images/56cc5d85820b180af0e3c809f94917152214d7eb9782ae9744c71f9d75d2fda7.jpg)

<details>
<summary>bar_stacked</summary>

| Language | CommonCrawl (GB) | Wikipedia (GB) |
| --- | --- | --- |
| en | ~250 | ~150 |
| ru | ~250 | ~45 |
| id | ~130 | ~0.4 |
| vi | ~120 | ~0.7 |
| fa | ~100 | ~0.6 |
| uk | ~80 | ~2 |
| sv | ~70 | ~2 |
| th | ~70 | ~0.4 |
| ja | ~65 | ~2.5 |
| de | ~60 | ~4.5 |
| ro | ~60 | ~0.3 |
| bu | ~55 | ~0.7 |
| bg | ~55 | ~0.6 |
| tr | ~55 | ~3.5 |
| fi | ~55 | ~0.6 |
| ko | ~55 | ~0.5 |
| es | ~55 | ~3 |
| no | ~50 | ~0.5 |
| pt | ~50 | ~1.2 |
| el | ~50 | ~0.6 |
| zh | ~50 | ~1 |
| da | ~50 | ~0.3 |
| pl | ~45 | ~1.5 |
| he | ~30 | ~0.9 |
| lt | ~30 | ~2.5 |
| nl | ~30 | ~1.5 |
| ar | ~28 | ~1.2 |
| sk | ~25 | ~0.2 |
| hi | ~22 | ~0.3 |
| br | ~20 | ~0.2 |
| tr | ~20 | ~0.4 |
| cs | ~18 | ~0.7 |
| lt | ~15 | ~0.2 |
| ta | ~12 | ~0.4 |
| ca | ~12 | ~0.9 |
| sl | ~10 | ~0.2 |
| ka | ~9 | ~0.3 |
| sr | ~8 | ~0.7 |
| lv | ~8 | ~0.1 |
| bn | ~8 | ~0.25 |
| ms | ~8 | ~0.2 |
| ml | ~7 | ~0.2 |
| az | ~6 | ~0.25 |
| kk | ~6 | ~0.2 |
| ct | ~6 | ~0.15 |
| ur | ~6 | — |
| hy | ~6 | — |
| sq | ~6 | — |
| mk | ~5 | — |
| te | ~5 | — |
| be | ~4.5 | — |
| ne | ~4.5 | — |
| si | ~4.5 | — |
| is | ~4.5 | — |
| kn | ~4.5 | — |
| tl | ~4.5 | — |
| gi | ~4.5 | — |
| mn | ~4.5 | — |
| mr | ~4.5 | — |
| la | ~4.5 | — |
| etu | ~3.5 | — |
| gu | ~3.5 | — |
| sw | ~3.5 | — |
| km | ~3.5 | — |
| af | ~3.5 | — |
| ky | ~3.5 | — |
| eo | ~3.5 | — |
| am | ~3.5 | — |
| pa | ~3.5 | — |
| cy | ~3.5 | — |
| ps | ~3.5 | — |
| uz | ~3.5 | — |
| or | ~3.5 | — |
| ga | ~3.5 | — |
| myi | ~3.5 | — |
| ku | ~3.5 | — |
| so | ~3.5 | — |
| ug | ~3.5 | — |
| sa | ~3.5 | — |
| yi | ~3.5 | — |
| mgg | ~1.5 | — |
</details>

Figure 1: Amount of data in GiB (log-scale) for the 88 languages that appear in both the Wiki-100 corpus used for mBERT and XLM-100, and the CC-100 used for XLM-R. CC-100 increases the amount of data by several orders of magnitude, in particular for low-resource languages.

Scaling to a hundred languages. XLM-R is trained on 100 languages; we provide a full list of languages and associated statistics in Appendix A. Figure 1 specifies the iso codes of 88 languages that are shared across XLM-R and XLM-100, the model from Lample and Conneau (2019) trained on Wikipedia text in 100 languages.

Compared to previous work, we replace some languages with more commonly used ones such as romanized Hindi and traditional Chinese. In our ablation studies, we always include the 7 languages for which we have classification and sequence labeling evaluation benchmarks: English, French, German, Russian, Chinese, Swahili and Urdu. We chose this set as it covers a suitable range of language families and includes low-resource languages such as Swahili and Urdu. We also consider larger sets of 15, 30, 60 and all 100 languages. When reporting results on high-resource and lowresource, we refer to the average of English and French results, and the average of Swahili and Urdu results respectively.

Scaling the Amount of Training Data. Following Wenzek et al. (2019) <sup>2</sup>, we build a clean CommonCrawl Corpus in 100 languages. We use an internal language identification model in combination with the one from fastText (Joulin et al., 2017). We train language models in each language and use it to filter documents as described in Wenzek et al. (2019). We consider one CommonCrawl dump for English and twelve dumps for all other languages, which significantly increases dataset sizes, especially for low-resource languages like Burmese and Swahili.

Figure 1 shows the difference in size between the Wikipedia Corpus used by mBERT and XLM-100, and the CommonCrawl Corpus we use. As we show in Section 5.3, monolingual Wikipedia corpora are too small to enable unsupervised representation learning. Based on our experiments, we found that a few hundred MiB of text data is usually a minimal size for learning a BERT model.

## 4 Evaluation

We consider four evaluation benchmarks. For crosslingual understanding, we use cross-lingual natural language inference, named entity recognition, and question answering. We use the GLUE benchmark to evaluate the English performance of XLM-R and compare it to other state-of-the-art models.

Cross-lingual Natural Language Inference (XNLI). The XNLI dataset comes with groundtruth dev and test sets in 15 languages, and a ground-truth English training set. The training set has been machine-translated to the remaining 14 languages, providing synthetic training data for these languages as well. We evaluate our model on cross-lingual transfer from English to other languages. We also consider three machine translation baselines: (i) translate-test: dev and test sets are machine-translated to English and a single English model is used (ii) translate-train (per-language): the English training set is machine-translated to each language and we fine-tune a multiligual model on each training set (iii) translate-train-all (multi-language): we fine-tune a multilingual model on the concatenation of all training sets from translate-train. For the translations, we use the official data provided by the XNLI project.

Named Entity Recognition. For NER, we consider the CoNLL-2002 (Sang, 2002) and CoNLL 2003 (Tjong Kim Sang and De Meulder, 2003) datasets in English, Dutch, Spanish and German. We fine-tune multilingual models either (1) on the English set to evaluate cross-lingual transfer, (2) on each set to evaluate per-language performance, or (3) on all sets to evaluate multilingual learning. We report the F1 score, and compare to baselines from Lample et al. (2016) and Akbik et al. (2018).

Cross-lingual Question Answering. We use the MLQA benchmark from Lewis et al. (2019), which extends the English SQuAD benchmark to Spanish, German, Arabic, Hindi, Vietnamese and Chinese. We report the F1 score as well as the exact match (EM) score for cross-lingual transfer from English.

GLUE Benchmark. Finally, we evaluate the English performance of our model on the GLUE benchmark (Wang et al., 2018) which gathers multiple classification tasks, such as MNLI (Williams et al., 2017), SST-2 (Socher et al., 2013), or QNLI (Rajpurkar et al., 2018). We use BERT<sub>Large</sub> and RoBERTa as baselines.

## 5 Analysis and Results

In this section, we perform a comprehensive analysis of multilingual masked language models. We conduct most of the analysis on XNLI, which we found to be representative of our findings on other tasks. We then present the results of XLM-R on cross-lingual understanding and GLUE. Finally, we compare multilingual and monolingual models, and present results on low-resource languages.

## 5.1 Improving and Understanding Multilingual Masked Language Models

Much of the work done on understanding the crosslingual effectiveness of mBERT or XLM (Pires et al., 2019; Wu and Dredze, 2019; Lewis et al.,

2019) has focused on analyzing the performance of fixed pretrained models on downstream tasks. In this section, we present a comprehensive study of different factors that are important to pretraining large scale multilingual models. We highlight the trade-offs and limitations of these models as we scale to one hundred languages.

Transfer-dilution Trade-off and Curse of Multilinguality. Model capacity (i.e. the number of parameters in the model) is constrained due to practical considerations such as memory and speed during training and inference. For a fixed sized model, the per-language capacity decreases as we increase the number of languages. While low-resource language performance can be improved by adding similar higher-resource languages during pretraining, the overall downstream performance suffers from this capacity dilution (Arivazhagan et al., 2019). Positive transfer and capacity dilution have to be traded off against each other.

We illustrate this trade-off in Figure 2, which shows XNLI performance vs the number of languages the model is pretrained on. Initially, as we go from 7 to 15 languages, the model is able to take advantage of positive transfer which improves performance, especially on low resource languages. Beyond this point the curse ofmultilinguality kicks in and degrades performance across all languages. Specifically, the overall XNLI accuracy decreases from 71.8% to 67.7% as we go from XLM-7 to XLM-100. The same trend can be observed for models trained on the larger CommonCrawl Corpus.

The issue is even more prominent when the capacity of the model is small. To show this, we pretrain models on Wikipedia Data in 7, 30 and 100 languages. As we add more languages, we make the Transformer wider by increasing the hidden size from 768 to 960 to 1152. In Figure 4, we show that the added capacity allows XLM-30 to be on par with XLM-7, thus overcoming the curse of multilinguality. The added capacity for XLM-100, however, is not enough and it still lags behind due to higher vocabulary dilution (recall from Section 3 that we used a fixed vocabulary size of 150K for all models).

High-resource vs Low-resource Trade-off. The allocation of the model capacity across languages is controlled by several parameters: the training set size, the size of the shared subword vocabulary, and the rate at which we sample training examples from each language. We study the effect of sampling on the performance of highresource (English and French) and low-resource (Swahili and Urdu) languages for an XLM-100 model trained on Wikipedia (we observe a similar trend for the construction of the subword vocab). Specifically, we investigate the impact of varying the α parameter which controls the exponential smoothing of the language sampling rate. Similar to Lample and Conneau (2019), we use a sampling rate proportional to the number of sentences in each corpus. Models trained with higher values of α see batches of high-resource languages more often. Figure 5 shows that the higher the value of α, the better the performance on high-resource languages, and vice-versa. When considering overall performance, we found 0.3 to be an optimal value for α, and use this for XLM-R.

![](images/72836a902d3a82135e45eb8c825397745da7f5b571d53a619f3dec36e64e36b9.jpg)

<details>
<summary>bar</summary>

| Number of languages | Low res. | High res. | All |
| --- | --- | --- | --- |
| 7 | ~63 | ~80 | ~72 |
| 15 | ~65 | ~79 | ~72 |
| 30 | ~64 | ~78 | ~71 |
| 60 | ~61 | ~77 | ~69 |
| 100 | ~58 | ~74 | ~67 |
</details>

Figure 2: The transferinterference trade-off: Lowresource languages benefit from scaling to more languages, until dilution (interference) kicks in and degrades overall performance.

![](images/6ceddcdaebb5d6b1ee91a17fa7a128d61ba704ca93db9a633e6ed9cbcd952556.jpg)

<details>
<summary>bar</summary>

| Category | Wikipedia | CommonCrawl |
| --- | --- | --- |
| Low res. | ~63 | ~69 |
| High res. | ~80 | ~82 |
| All | ~72 | ~76 |
</details>

Figure 3: Wikipedia versus CommonCrawl: An XLM-7 obtains significantly better performance when trained on CC, in particular on low-resource languages.

![](images/5966c57226cd5a10ec8aedf88bc699bef3ce179849edcb174f741221ab2e98a8.jpg)

<details>
<summary>bar</summary>

| Number of languages | Fixed capacity | Increased capacity |
| --- | --- | --- |
| 7 | ~72.8 | — |
| 30 | ~71.8 | ~72.4 |
| 100 | ~67.5 | ~69.7 |
</details>

Figure 4: Adding more capacity to the model alleviates the curse of multilinguality, but remains an issue for models of moderate size.

![](images/92275117db5ae483d9e0401c2e34460a54b6b6c6221b3356de46d0e1313602dc.jpg)

<details>
<summary>bar</summary>

| Language sampling | Low res. | High res. | All |
| --- | --- | --- | --- |
| 0.01 | ~58 | ~72 | ~65 |
| 0.3 | ~57 | ~74 | ~66 |
| 0.7 | ~52 | ~75 | ~65 |
| 1.0 | ~49 | ~75 | ~64 |
</details>

Figure 5: On the high-resource versus low-resource trade-off: impact of batch language sampling for XLM-100.

![](images/24fc092ee98303795a468469d3ee215df2b8dd6d050e1126b029bddd541ee06a.jpg)

<details>
<summary>bar</summary>

| Vocabulary size | Fixed capacity | Increased capacity |
| --- | --- | --- |
| 32k | ~61.9 | — |
| 64k | ~62.5 | — |
| 128k | ~63.6 | ~63.6 |
| 256k | ~64.7 | ~66.6 |
| 512k | ~63.3 | ~67.7 |
</details>

Figure 6: On the impact of vocabulary size at fixed capacity and with increasing capacity for XLM-100.

![](images/5fa1f115b48411f91c559edd9ab35a7e925d48a3591508e4d97cd5bb6c518b30.jpg)

<details>
<summary>bar</summary>

| Category | Accuracy |
| --- | --- |
| 2048 (Batch size) | ~64.7 |
| 4096 (Batch size) | ~65.5 |
| 8192 (Batch size) | ~67.0 |
| BPE (Preproc.) | ~64.0 |
| SPM (Preproc.) | ~63.8 |
</details>

Figure 7: On the impact of largescale training, and preprocessing simplification from BPE with tokenization to SPM on raw text data.

Importance of Capacity and Vocabulary. In previous sections and in Figure 4, we showed the importance of scaling the model size as we increase the number of languages. Similar to the overall model size, we argue that scaling the size of the shared vocabulary (the vocabulary capacity) can improve the performance of multilingual models on downstream tasks. To illustrate this effect, we train XLM-100 models on Wikipedia data with different vocabulary sizes. We keep the overall number of parameters constant by adjusting the width of the transformer. Figure 6 shows that even with a fixed capacity, we observe a 2.8% increase in XNLI average accuracy as we increase the vocabulary size from 32K to 256K. This suggests that multilingual models can benefit from allocating a higher proportion of the total number of parameters to the embedding layer even though this reduces the size of the Transformer. For simplicity and given the softmax computational constraints, we use a vocabulary of 250k for XLM-R.

We further illustrate the importance of this parameter, by training three models with the same transformer architecture $\mathbf { ( B E R T _ { B a s e } ) }$ but with different vocabulary sizes: 128K, 256K and 512K. We observe more than 3% gains in overall accuracy on XNLI by simply increasing the vocab size from 128k to 512k.

Larger-scale Datasets and Training. As shown in Figure 1, the CommonCrawl Corpus that we collected has significantly more monolingual data than the previously used Wikipedia corpora. Figure 3 shows that for the same $\mathbf { B E R T _ { B a s e } }$ architecture, all models trained on CommonCrawl obtain significantly better performance.

Apart from scaling the training data, Liu et al. (2019) also showed the benefits of training MLMs longer. In our experiments, we observed similar effects of large-scale training, such as increasing batch size (see Figure 7) and training time, on model performance. Specifically, we found that using validation perplexity as a stopping criterion for pretraining caused the multilingual MLM in Lample and Conneau (2019) to be under-tuned. In our experience, performance on downstream tasks continues to improve even after validation perplexity has plateaued. Combining this observation with our implementation of the unsupervised XLM-MLM objective, we were able to improve the performance of Lample and Conneau (2019) from 71.3% to more than 75% average accuracy on XNLI, which was on par with their supervised translation language modeling (TLM) objective. Based on these results, and given our focus on unsupervised learning, we decided to not use the supervised TLM objective for training our models.

Simplifying Multilingual Tokenization with Sentence Piece. The different language-specific tokenization tools used by mBERT and XLM-100 make these models more difficult to use on raw text. Instead, we train a Sentence Piece model (SPM) and apply it directly on raw text data for all languages. We did not observe any loss in performance for models trained with SPM when compared to models trained with language-specific preprocessing and byte-pair encoding (see Figure 7) and hence use SPM for XLM-R.

## 5.2 Cross-lingual Understanding Results

Based on these results, we adapt the setting of Lample and Conneau (2019) and use a large Transformer model with 24 layers and 1024 hidden states, with a 250k vocabulary. We use the multi lingual MLM loss and train our XLM-R model for 1.5 Million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192. We leverage the SPM-preprocessed text data from Common-Crawl in 100 languages and sample languages with $\alpha \ = \ 0 . 3$ . In this section, we show that it outperforms all previous techniques on cross-lingual benchmarks while getting performance on par with RoBERTa on the GLUE benchmark.

XNLI. Table 1 shows XNLI results and adds some additional details: (i) the number of models the approach induces (#M), (ii) the data on which the model was trained (D), and (iii) the number of languages the model was pretrained on (#lg). As we show in our results, these parameters significantly impact performance. Column #M specifies whether model selection was done separately on the dev set of each language (N models), or on the joint dev set of all the languages (single model). We observe a 0.6 decrease in overall accuracy when we go from N models to a single model - going from 71.3 to 70.7. We encourage the community to adopt this setting. For cross-lingual transfer, while this approach is not fully zero-shot transfer, we argue that in real applications, a small amount of supervised data is often available for validation in each language.

XLM-R sets a new state of the art on XNLI. On cross-lingual transfer, XLM-R obtains 80.9% accuracy, outperforming the XLM-100 and mBERT open-source models by 10.2% and 14.6% average accuracy. On the Swahili and Urdu lowresource languages, XLM-R outperforms XLM-100 by 15.7% and 11.4%, and mBERT by 23.5% and 15.8%. While XLM-R handles 100 languages, we also show that it outperforms the former state of the art Unicoder (Huang et al., 2019) and XLM (MLM+TLM), which handle only 15 languages, by 5.5% and 5.8% average accuracy respectively. Using the multilingual training of translate-train-all, XLM-R further improves performance and reaches 83.6% accuracy, a new overall state of the art for XNLI, outperforming Unicoder by 5.1%. Multilingual training is similar to practical applications where training sets are available in various languages for the same task. In the case of XNLI, datasets have been translated, and translate-trainall can be seen as some form of cross-lingual data augmentation (Singh et al., 2019), similar to backtranslation (Xie et al., 2019).

Named Entity Recognition. In Table 2, we report results of XLM-R and mBERT on CoNLL-2002 and CoNLL-2003. We consider the LSTM + CRF approach from Lample et al. (2016) and the Flair model from Akbik et al. (2018) as baselines. We evaluate the performance of the model on each of the target languages in three different settings: (i) train on English data only (en) (ii) train on data in target language (each) (iii) train on data in all languages (all). Results of mBERT are reported from Wu and Dredze (2019). Note that we do not use a linear-chain CRF on top of XLM-R and mBERT representations, which gives an advantage to Akbik et al. (2018). Without the CRF, our XLM-R model still performs on par with the state of the art, outperforming Akbik et al. (2018) on Dutch by 2.09 points. On this task, XLM-R also outperforms mBERT by 2.42 F1 on average for cross-lingual transfer, and 1.86 F1 when trained on each language. Training on all languages leads to an average F1 score of 89.43%, outperforming cross-lingual transfer approach by 8.49%.

<table><tr><td>Model</td><td>D</td><td>#M</td><td>#lg</td><td>en</td><td>fr</td><td>es</td><td>de</td><td>el</td><td>bg</td><td>ru</td><td>tr</td><td>ar</td><td>vi</td><td>th</td><td>zh</td><td>hi</td><td>sw</td><td>ur</td><td>Avg</td></tr><tr><td colspan="20">Fine-tune multilingual model on English training set (Cross-lingual Transfer)</td></tr><tr><td>Lample and Conneau (2019)</td><td>Wiki+MT</td><td>N</td><td>15</td><td>85.0</td><td>78.7</td><td>78.9</td><td>77.8</td><td>76.6</td><td>77.4</td><td>75.3</td><td>72.5</td><td>73.1</td><td>76.1</td><td>73.2</td><td>76.5</td><td>69.6</td><td>68.4</td><td>67.3</td><td>75.1</td></tr><tr><td>Huang et al. (2019)</td><td>Wiki+MT</td><td>N</td><td>15</td><td>85.1</td><td>79.0</td><td>79.4</td><td>77.8</td><td>77.2</td><td>77.2</td><td>76.3</td><td>72.8</td><td>73.5</td><td>76.4</td><td>73.6</td><td>76.2</td><td>69.4</td><td>69.7</td><td>66.7</td><td>75.4</td></tr><tr><td>Devlin et al. (2018)</td><td>Wiki</td><td>N</td><td>102</td><td>82.1</td><td>73.8</td><td>74.3</td><td>71.1</td><td>66.4</td><td>68.9</td><td>69.0</td><td>61.6</td><td>64.9</td><td>69.5</td><td>55.8</td><td>69.3</td><td>60.0</td><td>50.4</td><td>58.0</td><td>66.3</td></tr><tr><td>Lample and Conneau (2019)</td><td>Wiki</td><td>N</td><td>100</td><td>83.7</td><td>76.2</td><td>76.6</td><td>73.7</td><td>72.4</td><td>73.0</td><td>72.1</td><td>68.1</td><td>68.4</td><td>72.0</td><td>68.2</td><td>71.5</td><td>64.5</td><td>58.0</td><td>62.4</td><td>71.3</td></tr><tr><td>Lample and Conneau (2019)</td><td>Wiki</td><td>1</td><td>100</td><td>83.2</td><td>76.7</td><td>77.7</td><td>74.0</td><td>72.7</td><td>74.1</td><td>72.7</td><td>68.7</td><td>68.6</td><td>72.9</td><td>68.9</td><td>72.5</td><td>65.6</td><td>58.2</td><td>62.4</td><td>70.7</td></tr><tr><td> $XLM-R_{Base}$ </td><td>CC</td><td>1</td><td>100</td><td>85.8</td><td>79.7</td><td>80.7</td><td>78.7</td><td>77.5</td><td>79.6</td><td>78.1</td><td>74.2</td><td>73.8</td><td>76.5</td><td>74.6</td><td>76.7</td><td>72.4</td><td>66.5</td><td>68.3</td><td>76.2</td></tr><tr><td>XLM-R</td><td>CC</td><td>1</td><td>100</td><td>89.1</td><td>84.1</td><td>85.1</td><td>83.9</td><td>82.9</td><td>84.0</td><td>81.2</td><td>79.6</td><td>79.8</td><td>80.8</td><td>78.1</td><td>80.2</td><td>76.9</td><td>73.9</td><td>73.8</td><td>80.9</td></tr><tr><td colspan="20">Translate everything to English and use English-only model (TRANSLATE-TEST)</td></tr><tr><td>BERT-en</td><td>Wiki</td><td>1</td><td>1</td><td>88.8</td><td>81.4</td><td>82.3</td><td>80.1</td><td>80.3</td><td>80.9</td><td>76.2</td><td>76.0</td><td>75.4</td><td>72.0</td><td>71.9</td><td>75.6</td><td>70.0</td><td>65.8</td><td>65.8</td><td>76.2</td></tr><tr><td>RoBERTa</td><td>Wiki+CC</td><td>1</td><td>1</td><td>91.3</td><td>82.9</td><td>84.3</td><td>81.2</td><td>81.7</td><td>83.1</td><td>78.3</td><td>76.8</td><td>76.6</td><td>74.2</td><td>74.1</td><td>77.5</td><td>70.9</td><td>66.7</td><td>66.8</td><td>77.8</td></tr><tr><td colspan="20">Fine-tune multilingual model on each training set (TRANSLATE-TRAIN)</td></tr><tr><td>Lample and Conneau (2019)</td><td>Wiki</td><td>N</td><td>100</td><td>82.9</td><td>77.6</td><td>77.9</td><td>77.9</td><td>77.1</td><td>75.7</td><td>75.5</td><td>72.6</td><td>71.2</td><td>75.8</td><td>73.1</td><td>76.2</td><td>70.4</td><td>66.5</td><td>62.4</td><td>74.2</td></tr><tr><td colspan="20">Fine-tune multilingual model on all training sets (TRANSLATE-TRAIN-ALL)</td></tr><tr><td>Lample and Conneau (2019) $^†$ </td><td>Wiki+MT</td><td>1</td><td>15</td><td>85.0</td><td>80.8</td><td>81.3</td><td>80.3</td><td>79.1</td><td>80.9</td><td>78.3</td><td>75.6</td><td>77.6</td><td>78.5</td><td>76.0</td><td>79.5</td><td>72.9</td><td>72.8</td><td>68.5</td><td>77.8</td></tr><tr><td>Huang et al. (2019)</td><td>Wiki+MT</td><td>1</td><td>15</td><td>85.6</td><td>81.1</td><td>82.3</td><td>80.9</td><td>79.5</td><td>81.4</td><td>79.7</td><td>76.8</td><td>78.2</td><td>77.9</td><td>77.1</td><td>80.5</td><td>73.4</td><td>73.8</td><td>69.6</td><td>78.5</td></tr><tr><td>Lample and Conneau (2019)</td><td>Wiki</td><td>1</td><td>100</td><td>84.5</td><td>80.1</td><td>81.3</td><td>79.3</td><td>78.6</td><td>79.4</td><td>77.5</td><td>75.2</td><td>75.6</td><td>78.3</td><td>75.7</td><td>78.3</td><td>72.1</td><td>69.2</td><td>67.7</td><td>76.9</td></tr><tr><td> $XLM-R_{Base}$ </td><td>CC</td><td>1</td><td>100</td><td>85.4</td><td>81.4</td><td>82.2</td><td>80.3</td><td>80.4</td><td>81.3</td><td>79.7</td><td>78.6</td><td>77.3</td><td>79.7</td><td>77.9</td><td>80.2</td><td>76.1</td><td>73.1</td><td>73.0</td><td>79.1</td></tr><tr><td>XLM-R</td><td>CC</td><td>1</td><td>100</td><td>89.1</td><td>85.1</td><td>86.6</td><td>85.7</td><td>85.3</td><td>85.9</td><td>83.5</td><td>83.2</td><td>83.1</td><td>83.7</td><td>81.5</td><td>83.7</td><td>81.6</td><td>78.0</td><td>78.1</td><td>83.6</td></tr></table>

Table 1: Results on cross-lingual classification. We report the accuracy on each of the 15 XNLI languages and the average accuracy. We specify the dataset D used for pretraining, the number of models #M the approach requires and the number of languages #lg the model handles. Our XLM-R results are averaged over five different seeds. We show that using the translate-train-all approach which leverages training sets from multiple languages, XLM-R obtains a new state of the art on XNLI of 83.6% average accuracy. Results with <sup>†</sup> are from Huang et al. (2019).

<table><tr><td>Model</td><td>train</td><td>#M</td><td>en</td><td>nl</td><td>es</td><td>de</td><td>Avg</td></tr><tr><td>Lample et al. (2016)</td><td>each</td><td>N</td><td>90.74</td><td>81.74</td><td>85.75</td><td>78.76</td><td>84.25</td></tr><tr><td>Akbik et al. (2018)</td><td>each</td><td>N</td><td>93.18</td><td>90.44</td><td>-</td><td>88.27</td><td>-</td></tr><tr><td rowspan="2">mBERT $^{\dagger}$ </td><td>each</td><td>N</td><td>91.97</td><td>90.94</td><td>87.38</td><td>82.82</td><td>88.28</td></tr><tr><td>en</td><td>1</td><td>91.97</td><td>77.57</td><td>74.96</td><td>69.56</td><td>78.52</td></tr><tr><td rowspan="3">XLM-R $_{\text{Base}}$ </td><td>each</td><td>N</td><td>92.25</td><td>90.39</td><td>87.99</td><td>84.60</td><td>88.81</td></tr><tr><td>en</td><td>1</td><td>92.25</td><td>78.08</td><td>76.53</td><td>69.60</td><td>79.11</td></tr><tr><td>all</td><td>1</td><td>91.08</td><td>89.09</td><td>87.28</td><td>83.17</td><td>87.66</td></tr><tr><td rowspan="3">XLM-R</td><td>each</td><td>N</td><td>92.92</td><td>92.53</td><td>89.72</td><td>85.81</td><td>90.24</td></tr><tr><td>en</td><td>1</td><td>92.92</td><td>80.80</td><td>78.64</td><td>71.40</td><td>80.94</td></tr><tr><td>all</td><td>1</td><td>92.00</td><td>91.60</td><td>89.52</td><td>84.60</td><td>89.43</td></tr></table>

Table 2: Results on named entity recognition on CoNLL-2002 and CoNLL-2003 (F1 score). Results with † are from Wu and Dredze (2019). Note that mBERT and XLM-R do not use a linear-chain CRF, as opposed to Akbik et al. (2018) and Lample et al. (2016).

Question Answering. We also obtain new state of the art results on the MLQA cross-lingual question answering benchmark, introduced by Lewis et al. (2019). We follow their procedure by training on the English training data and evaluating on the 7 languages of the dataset. We report results in Table 3. XLM-R obtains F1 and accuracy scores of 70.7% and 52.7% while the previous state of the art was 61.6% and 43.5%. XLM-R also outperforms mBERT by 13.0% F1-score and 11.1% accuracy. It even outperforms BERT-Large on English, confirming its strong monolingual performance.

## 5.3 Multilingual versus Monolingual

In this section, we present results of multilingual XLM models against monolingual BERT models.

GLUE: XLM-R versus RoBERTa. Our goal is to obtain a multilingual model with strong performance on both, cross-lingual understanding tasks as well as natural language understanding tasks for each language. To that end, we evaluate XLM-R on the GLUE benchmark. We show in Table 4, that XLM-R obtains better average dev performance than $\mathrm { \Delta B E R T _ { L a r g e } }$ by 1.6% and reaches performance on par with $\mathrm { X L N e t _ { L a r g e } }$ . The RoBERTa model outperforms XLM-R by only 1.0% on average. We believe future work can reduce this gap even further by alleviating the curse of multilinguality and vocabulary dilution. These results demonstrate the possibility of learning one model for many languages while maintaining strong performance on per-language downstream tasks.

<table><tr><td>Model</td><td>train</td><td>#lgs</td><td>en</td><td>es</td><td>de</td><td>ar</td><td>hi</td><td>vi</td><td>zh</td><td>Avg</td></tr><tr><td>BERT-Large $^{\dagger}$ </td><td>en</td><td>1</td><td>80.2 / 67.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>mBERT $^{\dagger}$ </td><td>en</td><td>102</td><td>77.7 / 65.2</td><td>64.3 / 46.6</td><td>57.9 / 44.3</td><td>45.7 / 29.8</td><td>43.8 / 29.7</td><td>57.1 / 38.6</td><td>57.5 / 37.3</td><td>57.7 / 41.6</td></tr><tr><td>XLM-15 $^{\dagger}$ </td><td>en</td><td>15</td><td>74.9 / 62.4</td><td>68.0 / 49.8</td><td>62.2 / 47.6</td><td>54.8 / 36.3</td><td>48.8 / 27.3</td><td>61.4 / 41.8</td><td>61.1 / 39.6</td><td>61.6 / 43.5</td></tr><tr><td>XLM-RBase</td><td>en</td><td>100</td><td>77.1 / 64.6</td><td>67.4 / 49.6</td><td>60.9 / 46.7</td><td>54.9 / 36.6</td><td>59.4 / 42.9</td><td>64.5 / 44.7</td><td>61.8 / 39.3</td><td>63.7 / 46.3</td></tr><tr><td>XLM-R</td><td>en</td><td>100</td><td>80.6 / 67.8</td><td>74.1 / 56.0</td><td>68.5 / 53.6</td><td>63.1 / 43.5</td><td>69.2 / 51.6</td><td>71.3 / 50.9</td><td>68.0 / 45.4</td><td>70.7 / 52.7</td></tr></table>

Table 3: Results on MLQA question answering We report the F1 and EM (exact match) scores for zero-shot classification where models are fine-tuned on the English Squad dataset and evaluated on the 7 languages of MLQA. Results with † are taken from the original MLQA paper Lewis et al. (2019).

<table><tr><td>Model</td><td>#Igs</td><td>MNLI-m/mm</td><td>QNLI</td><td>QQP</td><td>SST</td><td>MRPC</td><td>STS-B</td><td>Avg</td></tr><tr><td>BERT $_{Large}^{\dagger}$ </td><td>1</td><td>86.6/-</td><td>92.3</td><td>91.3</td><td>93.2</td><td>88.0</td><td>90.0</td><td>90.2</td></tr><tr><td>XLNet $_{Large}^{\dagger}$ </td><td>1</td><td>89.8/-</td><td>93.9</td><td>91.8</td><td>95.6</td><td>89.2</td><td>91.8</td><td>92.0</td></tr><tr><td>RoBERTa $^{\dagger}$ </td><td>1</td><td>90.2/90.2</td><td>94.7</td><td>92.2</td><td>96.4</td><td>90.9</td><td>92.4</td><td>92.8</td></tr><tr><td>XLM-R</td><td>100</td><td>88.9/89.0</td><td>93.8</td><td>92.3</td><td>95.0</td><td>89.5</td><td>91.2</td><td>91.8</td></tr></table>

Table 4: GLUE dev results. Results with <sup>†</sup> are from Liu et al. (2019). We compare the performance of XLM-R to BERT<sub>Large</sub>, XLNet and RoBERTa on the English GLUE benchmark.

XNLI: XLM versus BERT. A recurrent criticism against multilingual models is that they obtain worse performance than their monolingual counterparts. In addition to the comparison of XLM-R and RoBERTa, we provide the first comprehensive study to assess this claim on the XNLI benchmark. We extend our comparison between multilingual XLM models and monolingual BERT models on 7 languages and compare performance in Table 5. We train 14 monolingual BERT models on Wikipedia and CommonCrawl (capped at 60 GiB), and two XLM-7 models. We increase the vocabulary size of the multilingual model for a better comparison. We found that multilingual models can outperform their monolingual BERT counterparts. Specifically, in Table 5, we show that for cross-lingual transfer, monolingual baselines outperform XLM-7 for both Wikipedia and CC by 1.6% and 1.3% average accuracy. However, by making use of multilingual training (translate-trainall) and leveraging training sets coming from multiple languages, XLM-7 can outperform the BERT models: our XLM-7 trained on CC obtains 80.0% average accuracy on the 7 languages, while the average performance of BERT models trained on CC is 77.5%. This is a surprising result that shows that the capacity of multilingual models to leverage training data coming from multiple languages for a particular task can overcome the capacity dilution problem to obtain better overall performance.

<table><tr><td>Model</td><td>D</td><td>#vocab</td><td>en</td><td>fr</td><td>de</td><td>ru</td><td>zh</td><td>sw</td><td>ur</td><td>Avg</td></tr><tr><td colspan="11">Monolingual baselines</td></tr><tr><td rowspan="2">BERT</td><td>Wiki</td><td>40k</td><td>84.5</td><td>78.6</td><td>80.0</td><td>75.5</td><td>77.7</td><td>60.1</td><td>57.3</td><td>73.4</td></tr><tr><td>CC</td><td>40k</td><td>86.7</td><td>81.2</td><td>81.2</td><td>78.2</td><td>79.5</td><td>70.8</td><td>65.1</td><td>77.5</td></tr><tr><td colspan="11">Multilingual models (cross-lingual transfer)</td></tr><tr><td rowspan="2">XLM-7</td><td>Wiki</td><td>150k</td><td>82.3</td><td>76.8</td><td>74.7</td><td>72.5</td><td>73.1</td><td>60.8</td><td>62.3</td><td>71.8</td></tr><tr><td>CC</td><td>150k</td><td>85.7</td><td>78.6</td><td>79.5</td><td>76.4</td><td>74.8</td><td>71.2</td><td>66.9</td><td>76.2</td></tr><tr><td colspan="11">Multilingual models (translate-train-all)</td></tr><tr><td rowspan="2">XLM-7</td><td>Wiki</td><td>150k</td><td>84.6</td><td>80.1</td><td>80.2</td><td>75.7</td><td>78</td><td>68.7</td><td>66.7</td><td>76.3</td></tr><tr><td>CC</td><td>150k</td><td>87.2</td><td>82.5</td><td>82.9</td><td>79.7</td><td>80.4</td><td>75.7</td><td>71.5</td><td>80.0</td></tr></table>

Table 5: Multilingual versus monolingual models (BERT-BASE). We compare the performance of monolingual models (BERT) versus multilingual models (XLM) on seven languages, using a BERT-BASE architecture. We choose a vocabulary size of 40k and 150k for monolingual and multilingual models.

## 5.4 Representation Learning for Low-resource Languages

We observed in Table 5 that pretraining on Wikipedia for Swahili and Urdu performed similarly to a randomly initialized model; most likely due to the small size of the data for these languages. On the other hand, pretraining on CC improved performance by up to 10 points. This confirms our assumption that mBERT and XLM-100 rely heavily on cross-lingual transfer but do not model the low-resource languages as well as XLM-R. Specifically, in the translate-train-all setting, we observe that the biggest gains for XLM models trained on CC, compared to their Wikipedia counterparts, are on low-resource languages; 7% and 4.8% improvement on Swahili and Urdu respectively.

## 6 Conclusion

In this work, we introduced XLM-R, our new state of the art multilingual masked language model trained on 2.5 TB of newly created clean CommonCrawl data in 100 languages. We show that it provides strong gains over previous multilingual models like mBERT and XLM on classification, sequence labeling and question answering. We exposed the limitations of multilingual MLMs, in particular by uncovering the high-resource versus low-resource trade-off, the curse of multilinguality and the importance of key hyperparameters. We also expose the surprising effectiveness of multilingual models over monolingual models, and show strong improvements on low-resource languages.

## References

Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence labeling. In COLING, pages 1638–1649.  
Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, et al. 2019. Massively multilingual neural machine translation in the wild: Findings and challenges. arXiv preprint arXiv:1907.05019.  
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In EMNLP.  
Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. In EMNLP. Association for Computational Linguistics.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. NAACL.  
Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, and Tomas Mikolov. 2018. Learning word vectors for 157 languages. In LREC.  
Haoyang Huang, Yaobo Liang, Nan Duan, Ming Gong, Linjun Shou, Daxin Jiang, and Ming Zhou. 2019. Unicoder: A universal language encoder by pretraining with multiple cross-lingual tasks. ACL.  
Melvin Johnson, Mike Schuster, Quoc V Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viegas, Martin Wattenberg, Greg Corrado,´ et al. 2017. Googles multilingual neural machine translation system: Enabling zero-shot translation. TACL, 5:339–351.  
Armand Joulin, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. 2017. Bag of tricks for efficient text classification. EACL 2017, page 427.  
Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 2016. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410.  
Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple sub word candidates. In ACL, pages 66–75.  
Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. EMNLP.  
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In NAACL, pages 260–270, San Diego, California. Association for Computational Linguistics.  
Guillaume Lample and Alexis Conneau. 2019. Crosslingual language model pretraining. NeurIPS.  
Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian˘ Riedel, and Holger Schwenk. 2019. Mlqa: Evaluating cross-lingual extractive question answering. arXiv preprint arXiv:1910.07475.  
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.  
Tomas Mikolov, Quoc V Le, and Ilya Sutskever. 2013a. Exploiting similarities among languages for machine translation. arXiv preprint arXiv:1309.4168.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013b. Distributed representations of words and phrases and their compositionality. In NIPS, pages 3111–3119.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In EMNLP, pages 1532–1543.  
Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. NAACL.  
Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual bert? In ACL.  
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. URL https://s3-us-west-2.amazonaws.com/openaiassets/research-covers/languageunsupervised/language understanding paper.pdf.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI Blog, 1(8).  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683.  
Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for squad. ACL.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In EMNLP, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.  
Erik F Sang. 2002. Introduction to the conll-2002 shared task: Language-independent named entity recognition. CoNLL.  
Tal Schuster, Ori Ram, Regina Barzilay, and Amir Globerson. 2019. Cross-lingual alignment of contextual word embeddings, with applications to zeroshot dependency parsing. NAACL.  
Aditya Siddhant, Melvin Johnson, Henry Tsai, Naveen Arivazhagan, Jason Riesa, Ankur Bapna, Orhan Firat, and Karthik Raman. 2019. Evaluating the crosslingual effectiveness of massively multilingual neural machine translation. AAAI.  
Jasdeep Singh, Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. 2019. Xlda: Cross-lingual data augmentation for natural language inference and question answering. arXiv preprint arXiv:1905.11471.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, pages 1631–1642.  
Xu Tan, Yi Ren, Di He, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2019. Multilingual neural machine translation with knowledge distillation. ICLR.  
Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: languageindependent named entity recognition. In CoNLL, pages 142–147. Association for Computational Linguistics.  
hi h i h iki k b Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.  
Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461.  
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzman, Armand Joulin, and Edouard Grave. 2019. Ccnet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359.  
Adina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A broad-coverage challenge corpus for sentence understanding through inference. Proceedings ofthe 2nd Workshop on Evaluating Vector-Space Representationsfor NLP.  
Shijie Wu, Alexis Conneau, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Emerging cross-lingual structure in pretrained language models. ACL.  
Shijie Wu and Mark Dredze. 2019. Beto, bentz, becas: The surprising cross-lingual effectiveness of bert. EMNLP.  
Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. 2019. Unsupervised data augmentation for consistency training. arXiv preprint arXiv:1904.12848.

## Appendix

## A Languages and statistics for CC-100 used by XLM-R

In this section we present the list of languages in the CC-100 corpus we created for training XLM-R. We also report statistics such as the number of tokens and the size of each monolingual corpus.

<table><tr><td>ISO code</td><td>Language</td><td>Tokens (M)</td><td>Size (GiB)</td><td>ISO code</td><td>Language</td><td>Tokens (M)</td><td>Size (GiB)</td></tr><tr><td>af</td><td>Afrikaans</td><td>242</td><td>1.3</td><td>lo</td><td>Lao</td><td>17</td><td>0.6</td></tr><tr><td>am</td><td>Amharic</td><td>68</td><td>0.8</td><td>lt</td><td>Lithuanian</td><td>1835</td><td>13.7</td></tr><tr><td>ar</td><td>Arabic</td><td>2869</td><td>28.0</td><td>lv</td><td>Latvian</td><td>1198</td><td>8.8</td></tr><tr><td>as</td><td>Assamese</td><td>5</td><td>0.1</td><td>mg</td><td>Malagasy</td><td>25</td><td>0.2</td></tr><tr><td>az</td><td>Azerbaijani</td><td>783</td><td>6.5</td><td>mk</td><td>Macedonian</td><td>449</td><td>4.8</td></tr><tr><td>be</td><td>Belarusian</td><td>362</td><td>4.3</td><td>ml</td><td>Malayalam</td><td>313</td><td>7.6</td></tr><tr><td>bg</td><td>Bulgarian</td><td>5487</td><td>57.5</td><td>mn</td><td>Mongolian</td><td>248</td><td>3.0</td></tr><tr><td>bn</td><td>Bengali</td><td>525</td><td>8.4</td><td>mr</td><td>Marathi</td><td>175</td><td>2.8</td></tr><tr><td>-</td><td>Bengali Romanized</td><td>77</td><td>0.5</td><td>ms</td><td>Malay</td><td>1318</td><td>8.5</td></tr><tr><td>br</td><td>Breton</td><td>16</td><td>0.1</td><td>my</td><td>Burmese</td><td>15</td><td>0.4</td></tr><tr><td>bs</td><td>Bosnian</td><td>14</td><td>0.1</td><td>my</td><td>Burmese</td><td>56</td><td>1.6</td></tr><tr><td>ca</td><td>Catalan</td><td>1752</td><td>10.1</td><td>ne</td><td>Nepali</td><td>237</td><td>3.8</td></tr><tr><td>cs</td><td>Czech</td><td>2498</td><td>16.3</td><td>nl</td><td>Dutch</td><td>5025</td><td>29.3</td></tr><tr><td>cy</td><td>Welsh</td><td>141</td><td>0.8</td><td>no</td><td>Norwegian</td><td>8494</td><td>49.0</td></tr><tr><td>da</td><td>Danish</td><td>7823</td><td>45.6</td><td>om</td><td>Oromo</td><td>8</td><td>0.1</td></tr><tr><td>de</td><td>German</td><td>10297</td><td>66.6</td><td>or</td><td>Oriya</td><td>36</td><td>0.6</td></tr><tr><td>el</td><td>Greek</td><td>4285</td><td>46.9</td><td>pa</td><td>Punjabi</td><td>68</td><td>0.8</td></tr><tr><td>en</td><td>English</td><td>55608</td><td>300.8</td><td>pl</td><td>Polish</td><td>6490</td><td>44.6</td></tr><tr><td>eo</td><td>Esperanto</td><td>157</td><td>0.9</td><td>ps</td><td>Pashto</td><td>96</td><td>0.7</td></tr><tr><td>es</td><td>Spanish</td><td>9374</td><td>53.3</td><td>pt</td><td>Portuguese</td><td>8405</td><td>49.1</td></tr><tr><td>et</td><td>Estonian</td><td>843</td><td>6.1</td><td>ro</td><td>Romanian</td><td>10354</td><td>61.4</td></tr><tr><td>eu</td><td>Basque</td><td>270</td><td>2.0</td><td>ru</td><td>Russian</td><td>23408</td><td>278.0</td></tr><tr><td>fa</td><td>Persian</td><td>13259</td><td>111.6</td><td>sa</td><td>Sanskrit</td><td>17</td><td>0.3</td></tr><tr><td>fi</td><td>Finnish</td><td>6730</td><td>54.3</td><td>sd</td><td>Sindhi</td><td>50</td><td>0.4</td></tr><tr><td>fr</td><td>French</td><td>9780</td><td>56.8</td><td>si</td><td>Sinhala</td><td>243</td><td>3.6</td></tr><tr><td>fy</td><td>Western Frisian</td><td>29</td><td>0.2</td><td>sk</td><td>Slovak</td><td>3525</td><td>23.2</td></tr><tr><td>ga</td><td>Irish</td><td>86</td><td>0.5</td><td>sl</td><td>Slovenian</td><td>1669</td><td>10.3</td></tr><tr><td>gd</td><td>Scottish Gaelic</td><td>21</td><td>0.1</td><td>so</td><td>Somali</td><td>62</td><td>0.4</td></tr><tr><td>gl</td><td>Galician</td><td>495</td><td>2.9</td><td>sq</td><td>Albanian</td><td>918</td><td>5.4</td></tr><tr><td>gu</td><td>Gujarati</td><td>140</td><td>1.9</td><td>sr</td><td>Serbian</td><td>843</td><td>9.1</td></tr><tr><td>ha</td><td>Hausa</td><td>56</td><td>0.3</td><td>su</td><td>Sundanese</td><td>10</td><td>0.1</td></tr><tr><td>he</td><td>Hebrew</td><td>3399</td><td>31.6</td><td>sv</td><td>Swedish</td><td>77.8</td><td>12.1</td></tr><tr><td>hi</td><td>Hindi</td><td>1715</td><td>20.2</td><td>sw</td><td>Swahili</td><td>275</td><td>1.6</td></tr><tr><td>-</td><td>Hindi Romanized</td><td>88</td><td>0.5</td><td>ta</td><td>Tamil</td><td>595</td><td>12.2</td></tr><tr><td>hr</td><td>Croatian</td><td>3297</td><td>20.5</td><td>-</td><td>Tamil Romanized</td><td>36</td><td>0.3</td></tr><tr><td>hu</td><td>Hungarian</td><td>7807</td><td>58.4</td><td>te</td><td>Telugu</td><td>249</td><td>4.7</td></tr><tr><td>hy</td><td>Armenian</td><td>421</td><td>5.5</td><td>-</td><td>Telugu Romanized</td><td>39</td><td>0.3</td></tr><tr><td>id</td><td>Indonesian</td><td>22704</td><td>148.3</td><td>th</td><td>Thai</td><td>1834</td><td>71.7</td></tr><tr><td>is</td><td>Icelandic</td><td>505</td><td>3.2</td><td>tl</td><td>Filipino</td><td>556</td><td>3.1</td></tr><tr><td>it</td><td>Italian</td><td>4983</td><td>30.2</td><td>tr</td><td>Turkish</td><td>2736</td><td>20.9</td></tr><tr><td>ja</td><td>Japanese</td><td>530</td><td>69.3</td><td>ug</td><td>Uyghur</td><td>27</td><td>0.4</td></tr><tr><td>jv</td><td>Javanese</td><td>24</td><td>0.2</td><td>uk</td><td>Ukrainian</td><td>6.5</td><td>84.6</td></tr><tr><td>ka</td><td>Georgian</td><td>469</td><td>9.1</td><td>ur</td><td>Urdu</td><td>730</td><td>5.7</td></tr><tr><td>kk</td><td>Kazakh</td><td>476</td><td>6.4</td><td>-</td><td>Urdu Romanized</td><td>85</td><td>0.5</td></tr><tr><td>km</td><td>Khmer</td><td>36</td><td>1.5</td><td>uz</td><td>Uzbek</td><td>91</td><td>0.7</td></tr><tr><td>kn</td><td>Kannada</td><td>169</td><td>3.3</td><td>vi</td><td>Vietnamese</td><td>24757</td><td>137.3</td></tr><tr><td>ko</td><td>Korean</td><td>5644</td><td>54.2</td><td>xh</td><td>Xhosa</td><td>13</td><td>0.1</td></tr><tr><td>ku</td><td>Kurdish (Kurmanji)</td><td>66</td><td>0.4</td><td>yi</td><td>Yiddish</td><td>34</td><td>0.3</td></tr><tr><td>ky</td><td>Kyrgyz</td><td>94</td><td>1.2</td><td>zh</td><td>Chinese (Simplified)</td><td>259</td><td>46.9</td></tr><tr><td>la</td><td>Latin</td><td>390</td><td>2.5</td><td>zh</td><td>Chinese (Traditional)</td><td>176</td><td>16.6</td></tr></table>

Table 6: Languages and statistics of the CC-100 corpus. We report the list of 100 languages and include the number of tokens (Millions) and the size of the data (in GiB) for each language. Note that we also include romanized variants of some non latin languages such as Bengali, Hindi, Tamil, Telugu and Urdu.

## B Model Architectures and Sizes

As we showed in section 5, capacity is an important parameter for learning strong cross-lingual representations. In the table below, we list multiple monolingual and multilingual models used by the research community and summarize their architectures and total number of parameters.

<table><tr><td>Model</td><td>#lgs</td><td>tokenization</td><td>L</td><td> $H_m$ </td><td> $H_{ff}$ </td><td>A</td><td>V</td><td>#params</td></tr><tr><td> $BERT_{Base}$ </td><td>1</td><td>WordPiece</td><td>12</td><td>768</td><td>3072</td><td>12</td><td>30k</td><td>110M</td></tr><tr><td> $BERT_{Large}$ </td><td>1</td><td>WordPiece</td><td>24</td><td>1024</td><td>4096</td><td>16</td><td>30k</td><td>335M</td></tr><tr><td>mBERT</td><td>104</td><td>WordPiece</td><td>12</td><td>768</td><td>3072</td><td>12</td><td>110k</td><td>172M</td></tr><tr><td> $RoBERTa_{Base}$ </td><td>1</td><td>bBPE</td><td>12</td><td>768</td><td>3072</td><td>8</td><td>50k</td><td>125M</td></tr><tr><td>RoBERTa</td><td>1</td><td>bBPE</td><td>24</td><td>1024</td><td>4096</td><td>16</td><td>50k</td><td>355M</td></tr><tr><td>XLM-15</td><td>15</td><td>BPE</td><td>12</td><td>1024</td><td>4096</td><td>8</td><td>95k</td><td>250M</td></tr><tr><td>XLM-17</td><td>17</td><td>BPE</td><td>16</td><td>1280</td><td>5120</td><td>16</td><td>200k</td><td>570M</td></tr><tr><td>XLM-100</td><td>100</td><td>BPE</td><td>16</td><td>1280</td><td>5120</td><td>16</td><td>200k</td><td>570M</td></tr><tr><td>Unicoder</td><td>15</td><td>BPE</td><td>12</td><td>1024</td><td>4096</td><td>8</td><td>95k</td><td>250M</td></tr><tr><td> $XLM-R_{Base}$ </td><td>100</td><td>SPM</td><td>12</td><td>768</td><td>3072</td><td>12</td><td>250k</td><td>270M</td></tr><tr><td>XLM-R</td><td>100</td><td>SPM</td><td>24</td><td>1024</td><td>4096</td><td>16</td><td>250k</td><td>550M</td></tr><tr><td>GPT2</td><td>1</td><td>bBPE</td><td>48</td><td>1600</td><td>6400</td><td>32</td><td>50k</td><td>1.5B</td></tr><tr><td>wide-mmNMT</td><td>103</td><td>SPM</td><td>12</td><td>2048</td><td>16384</td><td>32</td><td>64k</td><td>3B</td></tr><tr><td>deep-mmNMT</td><td>103</td><td>SPM</td><td>24</td><td>1024</td><td>16384</td><td>32</td><td>64k</td><td>3B</td></tr><tr><td>T5-3B</td><td>1</td><td>WordPiece</td><td>24</td><td>1024</td><td>16384</td><td>32</td><td>32k</td><td>3B</td></tr><tr><td>T5-11B</td><td>1</td><td>WordPiece</td><td>24</td><td>1024</td><td>65536</td><td>32</td><td>32k</td><td>11B</td></tr></table>

Table 7: Details on model sizes. We show the tokenization used by each Transformer model, the number of lavers L, the number of hidden states of the model $H _ { m } ,$ , the dimension of the feed-forward layer $H _ { f f } ,$ the number of attention heads A, the size of the vocabulary V and the total number of parameters #params. For Transformer encoders, the number of parameters can be approximated by $4 L H _ { m } ^ { 2 } + 2 L H _ { m } H _ { f f } + V H _ { m }$ GPT2 numbers are from Radford et al. (2019), mm-NMT models are from the work of Arivazhagan et al. (2019) on massively multilingual neural machine translation (mmNMT), and T5 numbers are from Raffel et al. (2019). While XLM-R is among the largest models partly due to its large embedding layer, it has a similar number of parameters than XLM-100, and remains significantly smaller that recently introduced Transformer models for multilingual MT and transfer learning. While this table gives more hindsight on the difference of capacity of each model, note it does not highlight other critical differences between the models.