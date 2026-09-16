# Improving Large-scale Language Models and Resources for Filipino

Jan Christian Blaise Cruz and Charibeth Cheng

Center for Language Technologies

College of Computer Studies

De La Salle University, Manila

{jan\_christian\_cruz, charibeth.cheng}@dlsu.edu.ph

## Abstract

In this paper, we improve on existing language resources for the low-resource Filipino language in two ways. First, we outline the construction of the TLUnified dataset, a largescale pretraining corpus that serves as an improvement over smaller existing pretraining datasets for the language in terms of scale and topic variety. Second, we pretrain new Transformer language models following the RoBERTa pretraining technique to supplant existing models trained with small corpora. Our new RoBERTa models show significant improvements over existing Filipino models in three benchmark datasets with an average gain of 4.47% test accuracy across the three classification tasks of varying difficulty.

## 1 Introduction

Unlike High-resource Languages (HRL) such as English, German, and French, Low-resource Languages (LRL) suffer from a lack of benchmark datasets, databases, linguistic tools, and pretrained models that impede the progress of research within those languages.

Despite the growing success of methods that intrinsically learn from little data (Deng et al., 2020; Lee et al., 2021), creating “more data” remains a very significant fundamental task in NLP. Given the data-hungry nature of the neural networks that are prevalent in NLP today, creating new datasets to train from is the most efficient way to improve model performance. In addition, cleverlyconstructed datasets also reveal new insights into the models we commonly use, letting us gauge their true performance and expose hidden weaknesses (Maudslay and Cotterell, 2021).

In this paper, we improve upon the existing resources for Filipino, a low-resource language spoken in the Philippines. We create a larger, more topically-varied large-scale pretraining dataset that improves upon the existing WikiText-TL-39 (Cruz and Cheng, 2019) that is too small and too topicallynarrow to create robust models that perform well in modern NLP. We also produce new RoBERTa pretrained models using our pretraining dataset that supplant existing models trained with less data (Cruz and Cheng, 2020).

## 2 Resource Creation

In this section, we outline our full methodology for resource creation. First, we introduce the construction of our new large-scale pretraining dataset. Next, we detail the pretraining steps for our new RoBERTa models. Lastly, we introduce the task datasets that we use to benchmark performance for our new pretrained models.

## 2.1 The TLUnified Dataset

To effectively pretrain a large transformer for downstream tasks, we require an equally large pretraining corpus of high-quality Filipino text. We construct our pretraining corpus by combining a number of available Filipino corpora, including:

• Bilingual Text Data – Bitext datasets are used for training Machine Translation models and contain crawled and aligned data from multiple sources. We collected multiple bitexts, extracted Filipino text, then deduplicated the extracted data to add to our pretraining corpus. Datasets we collected from include bible-uedin (Christodouloupoulos and Steedman, 2015), CCAligned (El-Kishky et al., 2020), ELRC 2922<sup>1</sup>, MultiC-CAligned (El-Kishky et al., 2020),ParaCrawl <sup>2</sup>, TED2020 (Reimers and Gurevych, 2020), WikiMatrix (Schwenk et al., 2019), tico-19, Ubuntu, OpenSubtitles, QED, Tanzil, Tatoeba, GlobalVoices, KDE4, and WikiMedia (Tiedemann, 2012).

• OSCAR – The Open Super-Large Crawled Aggregated Corpus (OSCAR) (Ortiz Suárez et al., 2019) is a massive dataset obtained from language identification and filtering of the Common Crawl dataset. We use the deduplicated version of the Filipino (Tagalog) portion of OSCAR and add it to our pretraining corpus.  
• NewsPH – The NewsPH (Cruz et al., 2021) corpus is a large-scale crawled corpus of Filipino news articles, originally used in automatically creating the NewsPH-NLI benchmark dataset. Since we plan on using an NLI dataset derived from NewsPH for benchmarking in this paper, we opted to only use a 60% subset of the NewsPH corpus to add to TLUnified.

Since a large portion of our corpus is crawled and artificially aligned, we expect that out-of-the-box data quality would be low. To clean our dataset, we apply a number of preprocessing filters to it, including:

1. Non-latin Filter – We filter out sentences whose characters are composed of more than 15% non-latin letters.  
2. Length Filter – We remove sentences that have a number of tokens N where 4 <= N <= 150.  
3. Puncutation Filter – All sentences that have tokens composed of too many succeeding punctuations (eg. “///”) are all removed.  
4. Average Word Length Filter – If a sentence has tokens that are significantly longer than the other tokens in the sentence, we remove the sentence entirely. We first take the sum of the character lengths of each token, then divide it by the number of tokens to get a ratio r. Only sentences with ratio 3 <= r <= 18 are kept in the corpus.  
5. HTML Filter – All sentences with HTML and URL-related tokens (e.g. “.com” or “http://”) are removed.

After filtering the dataset, we perform one additional deduplication step to ensure that no identical lines are found in the dataset. The final result is a large-scale pretraining dataset we call TLUnified.

We then train tokenizers using TLUnified, limiting our vocabulary to a fixed 32,000 BPE subwords (Sennrich et al., 2015). Our tokenizers are trained with a character coverage of 1.0. We also do not remove casing to ensure that capitalization is kept after tokenization.

<table><tr><td></td><td>Base</td><td>Large</td></tr><tr><td>Hidden Size</td><td>768</td><td>1024</td></tr><tr><td>Feedforward Size</td><td>3072</td><td>4096</td></tr><tr><td>Max Sequence Length</td><td>512</td><td>512</td></tr><tr><td>Attention Heads</td><td>12</td><td>16</td></tr><tr><td>Hidden Layers</td><td>12</td><td>24</td></tr><tr><td>Droput</td><td>0.1</td><td>0.1</td></tr></table>

Table 1: Base and Large RoBERTa hyperparameters.

## 2.2 Pretraining

We then pretrain transformer language models that can serve as bases for a variety of downstream tasks later on. For this purpose, we use the RoBERTa (Liu et al., 2019) pretraining technique. Previous pretrained transformers in Filipino (Cruz and Cheng, 2020; Cruz et al., 2021) used BERT (Devlin et al., 2018), and ELECTRA (Clark et al., 2020) as their method of choice.

We choose RoBERTa as it retains state-of-the-art performance on multiple NLP tasks while keeping its pretraining task simple unlike methods such as ELECTRA. As a reproduction study of BERT, RoBERTa optimizes and builds up on the BERT pretraining scheme to improve training efficiency and downstream performance.

Two size variants are trained in this study following the original RoBERTa paper: a Base model (110M parameters) and a Large model (330M parameters). Both size variants use the same BPE tokenizer trained with TLUnified. Our hyperparameter choices also follow the original RoBERTa paper closely. A summary of our models’ hyperparameters can be found in Table 1.

During training, we construct batches by continually filling them with tokens until we reach a maximum batch size of 8192 tokens. Both variants are trained using the Adafactor (Shazeer and Stern, 2018) optimizer with β = 0.98 and a weight decay of 0.01. The base model is trained for 100,000 steps with a learning rate of 6e-4, while the large variant is trained for 300,000 steps with a learning rate of 4e-4. We also use a learning rate schedule that linearly warms up for 25,000 steps, then linearly decays for the rest of training. All experiments are done on a server with 8x NVIDIA Tesla

P100 GPUs.

## 2.3 Benchmark Datasets

We test the efficacy of our RoBERTa models on three Filipino benchmark datasets:

• Filipino Hatespeech Dataset – 10,000 tweets labelled as “hate” and “non-hate” collected during the 2016 Philippine Presidential Elections. Originally published in Cabasag et al. (2019) and benchmarked with modern Transformers in Cruz and Cheng (2020).  
• Filipino Dengue Dataset – Low-resource multiclass classification dataset with 4000 samples that can be one or many of five labels. Originally published in Livelo and Cheng (2018) and benchmarked in Cruz and Cheng (2020) using pretrained Transformers.  
• NewsPH-NLI – An automatically-generated dataset constructed by exploiting the “inverted-pyramid” structure of news articles, causing every sentence to naturally entail the sentence that came before it. Originally created in Cruz et al. (2021).

For this study, we do not use the original NewsPH-NLI created in Cruz et al. (2021) as it has significant overlap with the subset of the NewsPH corpus that we used for pretraining. We instead re-generated a version of NewsPH-NLI (which we call “NewsPH-NLI Medium”) using 40% of the NewsPH corpus, using the other 60% as part of the TLUnified pretraining data. This ensures that no test data is present in the training data, which will significantly inflate the benchmark scores.

Preprocessing for the downstream benchmark datasets is kept simple and non-destructive to preserve the linguistic structures and information present in the original data.

For the Hatespeech and the Dengue datasets, we follow the preprocessing used in Cruz and Cheng (2020), with a number of changes. Since both are datasets composed mainly of tweet data, the following preprocessing steps are done:

• Moses detokenization (Koehn and Hoang, 2010) was applied on all Moses-tokenized text.  
• All HTML meta text and link texts are collapsed into a special [LINK] token. This is to reduce the noise in the dataset as images in the tweets are naturally converted into links.

• All substrings that start with an @ character that are greater than length 1 are automatically treated as a “mention” and are replaced with a [MENTION] special token.  
• All substrings that start with a # character that are greater than length 1 are automatically treated as a “hashtag” and are replaced with a [HASHTAG] special token.  
• We renormalize apostrophes (e.g. it ’s → it’s) and punctuation that were spaced out (e.g. one - two → one-two) during the preprocessing in the Cruz and Cheng (2020) paper.  
• Characters that were converted into unicode (e.g. &amp;) are converted back into their encoded form (e.g. &).

For the Dengue dataset, we transform the multilabel, multiclass classification setup into a multiclass classification problem by concatenating an example’s labels and converting the resulting binary number into an integer. For example, a sentence with the labels 1, 1, 0, 1, 1 for absent, dengue, healthclasses, mosquito, and sick will be converted into 27 (11011 → 27). This results in 32 possible labels and increases the difficulty of the task.

For the NewsPH-NLI Medium dataset, we opted to not do any further preprocessing as the released data from Cruz et al. (2021) is already preprocessed and clean.

## 2.4 Finetuning Setups

We then finetune for the downstream benchmark tasks using our pretrained RoBERTa models. Since the NewsPH-NLI version and the setup of the Dengue dataset task is different from the previous benchmarking paper, we also finetuned Tagalog BERT (Cruz and Cheng, 2019) and Tagalog ELEC-TRA (Cruz et al., 2021) to serve as baseline models against the new RoBERTa model.

All models are trained using the Adafactor (Shazeer and Stern, 2018) optimizer with a learning rate scheduler that linearly increases from zero after a ratio of steps-to-total-training-steps has reached, then linearly decays afterwards. We use a batch size of 32 sentences for all models and use a weight decay of 0.1. We opted to use a larger maximum sequence length for the Large RoBERTa models as it has more capacity due to its deeper encoder stack.

<table><tr><td rowspan="2">Model</td><td colspan="2">Hatespeech</td><td colspan="2">Dengue</td><td colspan="2">NewsPH-NLI Med.</td></tr><tr><td>Val. Acc</td><td>Test Acc.</td><td>Val. Acc</td><td>Test Acc.</td><td>Val. Acc</td><td>Test Acc.</td></tr><tr><td>BERT Base Cased</td><td>0.7479</td><td>0.7417</td><td>0.7720</td><td>0.7580</td><td>0.8838</td><td>0.8874</td></tr><tr><td>ELECTRA Base Cased</td><td>0.7491</td><td>0.7250</td><td>0.7400</td><td>0.6920</td><td>0.9094</td><td>0.9106</td></tr><tr><td>RoBERTa Base</td><td>0.7866</td><td>0.7807</td><td>0.8180</td><td>0.8020</td><td>0.9492</td><td>0.9501</td></tr><tr><td>RoBERTa Large</td><td>0.7897</td><td>0.7824</td><td>0.8281</td><td>0.8110</td><td>0.9499</td><td>0.9510</td></tr></table>

Table 2: Finetuning results for all Transformer variants on the three benchmark datasets.

<table><tr><td></td><td>Base</td><td>Large</td></tr><tr><td>Max. Seq. Length</td><td>128</td><td>256</td></tr><tr><td>Learning Rate</td><td>2e-5</td><td>1e-5</td></tr><tr><td>Warmup Ratio</td><td>0.1</td><td>0.06</td></tr></table>

Table 3: Unique finetuning hyperparameters for Base and Large transformer variants.

Hyperparameters that are different between Base and Large variants of the pretrained Transformers used are found in Table 3.

We add the [LINK], [MENTION], and [HASHTAG] special tokens during finetuning for the Hatespeech and Dengue datasets to the vocabularies of the Transformers used, averaging the vectors of all subword embeddings in the embedding layer to serve as initialization for the three added tokens.

Despite our RoBERTa having a full maximum sequence length allowance of 512, we opted to use smaller maximum sequence lengths during finetuning. This speeds up training (approximately 4x for the Base models and 2x for the Large models) while losing zero information since no sentence or sentence pair in any task reaches 256 subwords in length.

All experiments are done on a server with 8x NVIDIA Tesla P100 GPUs.

## 3 Results

We report the results for our finetuning for the three benchmark datasets in terms of validation and test accuracy. A summary of the results can be found on Table 2.

Our RoBERTa models outperformed both the BERT and the ELECTRA models across all tasks. For the Hatespeech task, RoBERTa Large outperformed the best previous model (BERT Base) by +4.07% test accuracy. RoBERTa large also had a gain in performance in the Dengue dataset (+5.3% test accuracy over BERT Base) and the NewsPH-NLI Medium dataset (+4.04% test accuracy over

ELECTRA Base).

While marginally inferior to the Large variant, the Base RoBERTa variant still outperforms the baseline models in all tasks. RoBERTa Base has an improvement of +3.9% against BERT Base on the Hatespeech task, +4.4% against BERT Base on the Dengue task, and +3.95% against ELECTRA Base on the NewsPH-NLI Medium task.

The difference in performance between the Base and Large RoBERTa variants is marginal in the current benchmarks. Large outperforms Base only by +0.17% for Hatespeech, +0.9% for Dengue, and +0.09% for NewsPH-NLI Medium. We hypothesize that this is due to the size of the pretraining dataset. While the size of TLUnified is much larger than the previous WikiText-TL-39, it may still not be enough to make full use of the capacity of a Large-variant Transformer. We surmise that RoBERTa Large may need to be trained with more data to show significant, non-marginal improvements in performance.

Overall, our new models show significant improvements over older pretrained Filipino Transformer models. This is likely due to the improved pretraining corpus, with TLUnified being larger and of more varied topics and sources than the previous WikiText-TL-39.

## 4 Conclusion

Our work has two main contributions in terms of language resources for the Filipino language. First, we construct TLUnified, a new large-scale pretraining corpus for Filipino. This is an improvement over the much smaller pretraining corpora currently available, boasting much larger scale and topic variety. Second, we release new pretrained Transformers using the RoBERTa pretraining method. Our new models outperform existing baselines on three different classification tasks, with significant improvements of +4.07%, +5.03%, and +4.04% test accuracy for the Hatespeech, Dengue, and NewsPH-NLI Medium datasets respectively.

## References

Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng. 2019. Hate speech in philippine election-related tweets: Automatic detection and classification using natural language processing. Philippine Computing Journal, XIV No, 1.  
Christos Christodouloupoulos and Mark Steedman. 2015. A massively parallel corpus: the bible in 100 languages. Language resources and evaluation, 49(2):375–395.  
Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. 2020. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2019. Evaluating language model finetuning techniques for low-resource languages. arXiv preprint arXiv:1907.00409.  
Jan Christian Blaise Cruz and Charibeth Cheng. 2020. Establishing baselines for text classification in low-resource languages. arXiv preprint arXiv:2005.02068.  
Jan Christian Blaise Cruz, Jose Kristian Resabal, James Lin, Dan John Velasco, and Charibeth Cheng. 2021. Exploiting news article structure for automatic corpus generation of entailment datasets. In Pacific Rim International Conference on Artificial Intelligence, pages 86–99. Springer.  
Shumin Deng, Ningyu Zhang, Zhanlin Sun, Jiaoyan Chen, and Huajun Chen. 2020. When low resource nlp meets unsupervised language model: Metapretraining then meta-learning for few-shot text classification (student abstract). In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 13773–13774.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understand ing. arXiv preprint arXiv:1810.04805.  
Ahmed El-Kishky, Vishrav Chaudhary, Francisco Guzmán, and Philipp Koehn. 2020. CCAligned: A massive collection of cross-lingual web-document pairs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pages 5960–5969, Online. Association for Computational Linguistics.  
Philipp Koehn and Hieu Hoang. 2010. Moses. Statistical Machine Translation System, User Manual and Code Guide, page 245.  
Hung-yi Lee, Ngoc Thang Vu, and Shang-Wen Li. 2021. Meta learning and its applications to natural language processing. In Proceedings ofthe 59th Annual Meeting of the Association for Computational  
Linguistics and the 11th International Joint Conference on Natural Language Processing: Tutorial Abstracts, pages 15–20.  
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.  
Evan Dennison Livelo and Charibeth Cheng. 2018. Intelligent dengue infoveillance using gated recurrent neural learning and cross-label frequencies. In 2018 IEEE International Conference on Agents (ICA), pages 2–7. IEEE.  
Rowan Hall Maudslay and Ryan Cotterell. 2021. Do syntactic probes probe syntax? experiments with jabberwocky probing. arXiv preprint arXiv:2106.02559.  
Pedro Javier Ortiz Suárez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures. Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019, pages 9 – 16, Mannheim. Leibniz-Institut für Deutsche Sprache.  
Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.  
Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzmán. 2019. Wikimatrix: Mining 135m parallel sentences in 1620 language pairs from wikipedia. arXiv preprint arXiv·1907 05791  
Rico Sennrich, Barry Haddow, and Alexandra Birch. 2015. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.  
Noam Shazeer and Mitchell Stern. 2018. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR.  
Jörg Tiedemann. 2012. Parallel data, tools and interfaces in opus. In Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC’12), Istanbul, Turkey. European Language Resources Association (ELRA).