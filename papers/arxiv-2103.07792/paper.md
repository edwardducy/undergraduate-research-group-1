# Multilingual Code-Switching for Zero-Shot Cross-Lingual  
Intent Prediction and Slot Filling

 Jitin Krishnan    Antonios Anastasopoulos    Hemant Purohit    Huzefa Rangwala Affiliation: George Mason University Affiliation: Fairfax, VA, USA Email: [{jkrishn2,antonis,hpurohit,rangwala}@gmu.edu](mailto:) 

###### Abstract

Predicting user intent and detecting the corresponding slots from text are two key problems in Natural Language Understanding (NLU). In the context of zero-shot learning, this task is typically approached by either using representations from pre-trained multilingual transformers such as mBERT, or by machine translating the source data into the known target language and then fine-tuning. Our work focuses on a particular scenario where the target language is unknown during training. To this goal, we propose a novel method to augment the monolingual source data using multilingual code-switching via random translations to enhance a transformer’s language neutrality when fine-tuning it for a downstream task. This method also helps discover novel insights on how code-switching with different language families around the world impact the performance on the target language. Experiments on the benchmark dataset of MultiATIS++ yielded an average improvement of +4.2%+4.2\\% in accuracy for intent task and +1.8%+1.8\\% in F1 for slot task using our method over the state-of-the-art across 88 different languages11 1 Languages that have different morphological structures compared to English, such as Hindi, Turkish, Chinese, and Japanese, yielded higher benefits.. Furthermore, we present an application of our method for crisis informatics using a new human-annotated tweet dataset of slot filling in English and Haitian Creole, collected during Haiti earthquake disaster22 2 Dataset and implementation available at [https://github.com/jitinkrishnan/Multilingual-ZeroShot-SlotFilling](https://github.com/jitinkrishnan/Multilingual-ZeroShot-SlotFilling "")..

 

![Refer to caption](2103.07792v2/mbert_rep.png)

Figure 1: t-SNE plot of embeddings across the 1212 multi-head attention layers of multilingual BERT. Parallelly translated sentences of MutiATIS++ dataset are still clustered according to the languages: English (black), Chinese (cyan), French (blue), German (green), and Japanese (red).

![Refer to caption](2103.07792v2/chunk3.png)

Figure 2: An original example in English from MultiATIS++ dataset and its multilingually code-switched version. In the above code-switching example, the chunks are in Chinese, Punjabi, Spanish, English, Arabic, and Russian. ‘atis\_airfare’ represents an intent class where the user seeks price of a ticket.

## 1 Introduction

A cross-lingual setting is typically described as a scenario in which a model trained for a particular task in one language (e.g. English) should be able to generalize well to a different language (e.g. Japanese). While a semi-supervised solution [Xiao and Guo (2013)](#bib.bib60 ""); [Muis et al. (2018)](#bib.bib34 "") assumes some target language data is available, a zero-shot solution [Eriguchi et al. (2018)](#bib.bib16 ""); [Srivastava et al. (2018)](#bib.bib50 ""); [Xu et al. (2020)](#bib.bib62 "") assumes none is available at training time. This is particularly significant in real world problems such as extracting relevant information during a new disaster [Nguyen et al. (2017)](#bib.bib35 ""); [Krishnan et al. (2020)](#bib.bib28 "") and hate speech detection [Pamungkas and Patti (2019)](#bib.bib37 ""); [Stappen et al. (2020)](#bib.bib51 ""), where the target language might be of low-resource or unknown. In such scenarios, it is crucial that models can generalize well to unseen languages.

Intent prediction and slot filling are important NLU tasks and significant for real world problems. They are studied extensively for goal-oriented dialogue systems currently, such as Amazon’s Alexa, Apple’s Siri, Google Assistant, and Microsoft’s Cortana. Finding the ‘intent’ behind the user’s query and identifying relevant ‘slots’ in the sentence to engage in a dialogue are essential for an effective conversational assistance. For example, users might want to ‘play music’ given the slot labels ‘year’ and ‘artist’ [Coucke et al. (2018)](#bib.bib11 ""), or they may want to ‘book a flight’ given the slot labels ‘airport’ and ‘locations’ [Price (1990)](#bib.bib41 ""). A strong correlation between the two tasks has made jointly trained models successful [Goo et al. (2018)](#bib.bib18 ""); [Haihong et al. (2019)](#bib.bib19 ""); [Hardalov et al. (2020)](#bib.bib21 ""); [Chen et al. (2019)](#bib.bib5 ""). In a cross-lingual setting, the model should be able to learn this joint task in one language and transfer knowledge to another [Upadhyay et al. (2018)](#bib.bib54 ""); [Schuster et al. (2019)](#bib.bib46 ""); [Xu et al. (2020)](#bib.bib62 ""). This is the premise of our work.

Highly effective multilingual models such as mBERT [Devlin et al. (2019)](#bib.bib12 "") and XLM-R [Conneau et al. (2020a)](#bib.bib7 "") have shown success across several multilingual tasks in recent years. In the zero-shot cross-lingual transfer setting with an unknown target language, a typical solution is to use pre-trained transformer models and fine-tune to the downstream task using the monolingual source data [Xu et al. (2020)](#bib.bib62 ""). However, previous work [Pires et al. (2019)](#bib.bib39 "") has shown that existing transformer-based representations may exhibit systematic deficiencies for certain language pairs.Previous work [Pires et al. (2019)](#bib.bib39 "") has shown that existing transformer-based representations may exhibit systematic deficiencies for certain language pairs. Figure [1](#S0.F1 "Figure 1 ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") shows that the representations across the 1212 multi-head attention layers of mBERT are still clustered according to the languages. This leads to a fundamental challenge that we address in this work: enhancing the language neutrality so that the fine-tuned model is generalizable across languages for the downstream task. To this goal, we introduce a data augmentation method via multilingual code-switching, where the original sentence in English is code-switched into randomly selected languages. For example, chunk-level code-switching creates sentences with phrases in multiple languages as shown in Figure [2](#S0.F2 "Figure 2 ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). We show that this can lead to a better performance in the zero-shot setting such that mBERT can be fine-tuned for all languages (not just one) with a monolingual source data.

Further, we show how code-switching with different language families impact the model’s performance on individual target languages. Cross-lingual study of language families largely remains unexplored for NLU tasks. For instance, while it might be intuitive that Sino-Tibetan language family can aid a task in Hindi, results indicating that Turkic language family may help Japanese can reveal intriguing inter-family relationships and how they are aligned in the underlying language model’s vector space.

Contributions: a) We present a data augmentation method via multilingual code-switching to enhance the language neutrality of mBERT for fine-tuning to a downstream NLU task of intent prediction and slot filling. b) By code-switching into different language families, we show that potential relationships between a family and a target language can be identified and studied; which could help foster zero-shot cross-lingual research in low-resource languages. c) We release a new human-annotated tweet dataset, collected during Haiti earthquake disaster, for intent prediction and slot filling in English and Haitian Creole.

Advantages: With enhanced generalizability, our model can be deployed with an out-of-the-box functionality. Previous methods of first machine translation of the source data into the known target language, followed by fine-tuning (referred ‘translate-train’) [Xu et al. (2020)](#bib.bib62 ""); [Yarowsky et al. (2001)](#bib.bib64 ""); [Shah et al. (2010)](#bib.bib48 ""); [Ni et al. (2017)](#bib.bib36 "") require a separate model to be trained for each language.

## 2 Related Work

### 2.1 Cross-Lingual Transfer Learning

Researchers have studied cross-lingual tasks in various settings such as sentiment/sequence classification [Wan (2009)](#bib.bib56 ""); [Eriguchi et al. (2018)](#bib.bib16 ""); [Yu et al. (2018)](#bib.bib66 ""), named entity recognition [Zirikly and Hagiwara (2015)](#bib.bib67 ""); [Tsai et al. (2016)](#bib.bib53 ""); [Xie et al. (2018)](#bib.bib61 ""), parts-of-speech tagging [Yarowsky et al. (2001)](#bib.bib64 ""); [Täckström et al. (2013)](#bib.bib52 ""); [Plank and Agić (2018)](#bib.bib40 ""), and natural language understanding [He et al. (2013)](#bib.bib22 ""); [Upadhyay et al. (2018)](#bib.bib54 ""); [Xu et al. (2020)](#bib.bib62 ""). The methodology for most of the current approaches for cross-lingual tasks fall into the following three categories: a) multilingual representations from pre-trained or fine-tuned models such as mBERT [Devlin et al. (2019)](#bib.bib12 "") or XLM-R [Conneau et al. (2020a)](#bib.bib7 ""), b) machine translation followed by alignment [Shah et al. (2010)](#bib.bib48 ""); [Yarowsky et al. (2001)](#bib.bib64 ""); [Ni et al. (2017)](#bib.bib36 ""), or c) a combination of both [Xu et al. (2020)](#bib.bib62 ""). Before transformer models, effective approaches included domain adversarial training to extract language-agnostic features [Ganin et al. (2016)](#bib.bib17 ""); [Chen et al. (2018)](#bib.bib6 "") and word alignment methods such as MUSE [Conneau et al. (2017)](#bib.bib8 "") to align fastText word vectors [Bojanowski et al. (2017)](#bib.bib3 ""). Recently, [Conneau et al.](#bib.bib10 ""), [2020b](#bib.bib10 "") has shown that having shared parameters in the top layers of the multi-lingual encoders can be used to align different languages quite effectively on tasks such as XNLI [Conneau et al. (2018)](#bib.bib9 "").

Monolingual models for joint slot filling and intent prediction have used methods such as attention-based RNN [Liu and Lane (2016)](#bib.bib29 "") and attention-based BiLSTM with a slot gate [Goo et al. (2018)](#bib.bib18 "") on benchmark datasets such as ATIS [Price (1990)](#bib.bib41 "") and SNIPS [Coucke et al. (2018)](#bib.bib11 ""). These methods have shown that a joint method can enhance both tasks and slot filling can be conditioned on the learned intent. An interrelated mechanism was introduced [Haihong et al. (2019)](#bib.bib19 "") to iteratively learn the relationship between the two tasks. Recently, BERT-based approaches [Hardalov et al. (2020)](#bib.bib21 ""); [Chen et al. (2019)](#bib.bib5 "") have shown improved results. On the other hand, cross-lingual versions of this joint task include a low-supervision based approach for Hindi and Turkish [Upadhyay et al. (2018)](#bib.bib54 ""), new dataset for Spanish and Thai [Schuster et al. (2019)](#bib.bib46 ""), and the most recent work of MultiATIS++ [Xu et al. (2020)](#bib.bib62 "") creating a comprehensive dataset in 9 languages; which is used to benchmark our results.

The joint task mentioned above in a pure zero-shot learning is the motivation of our work. Zero-shot is described as the setting where the model sees a new distribution of examples during test time [Xian et al. (2017)](#bib.bib59 ""); [Srivastava et al. (2018)](#bib.bib50 ""); [Romera-Paredes and Torr (2015)](#bib.bib44 ""). It is common for machine translation based methods to translate source data to the target language before training. We assume that target language is unknown during training, so that our model is generalizable across languages.

### 2.2 Code-Switching

Linguistic code-switching is a phenomenon where multilingual speakers alternate between languages. Recently, monolingual models have been adapted to code-switched text in several tasks such as entity recognition [Aguilar and Solorio (2019)](#bib.bib1 ""), part-of-speech tagging [Soto and Hirschberg (2018)](#bib.bib49 ""); [Ball and Garrette (2018)](#bib.bib2 ""), sentiment analysis [Joshi et al. (2016)](#bib.bib25 ""), and language identification [Mave et al. (2018)](#bib.bib32 ""); [Yirmibeşoğlu and Eryiğit (2018)](#bib.bib65 ""); [Mager et al. (2019)](#bib.bib30 ""). Recently, [KhudaBukhsh et al.](#bib.bib26 ""), [2020](#bib.bib26 "") have proposed a pipeline to sample code-mixed documents using minimal supervision. [Qin et al.](#bib.bib42 ""), [2020](#bib.bib42 "") allows randomized code-switching to include the target language. In our context, if the target language is German, we ensure that there is no code-switching to German during training. We consider this distinction essential to evaluate a true zero-shot learning scenario and prevent any bias. Another recent work by [Yang et al.](#bib.bib63 ""), [2020](#bib.bib63 "") presents a non-zero-shot approach that performs code-switching to target languages. [Jiang et al.](#bib.bib23 ""), [2020](#bib.bib23 "") presents presents a code-switching based method to improve the ability of multilingual language models for factual knowledge retrieval. Code-switching is usually done at the word-level. However, our results favor chunk-level switching over word-level as the latter may bring more noise to the code-switched version when compared to the original meaning of the sentence. Code-switching and other data augmentation techniques have been applied to the pre-training stage in recent works [Chaudhary et al. (2020)](#bib.bib4 ""); [Dufter and Schütze (2020)](#bib.bib13 ""), however we do not address pre-training in this work. Pre-trained models such as XLM-R is also likely to be exposed to code-switched data, as it is trained using common-crawl. In this work, we specifically focus on mBERT which largely remain monolingual at the sentence level to identify the impact of code-switching during fine-tuning, in addition to study the impact of language-family-based augmentations.

## 3 Methodology

This section first describes our problem for zero-shot cross-lingual transfer setting, followed by a novel data augmentation method using multilingual code-switching of monolingual source to enhance language neutrality. We then describe language families, followed by the joint training setup.

### 3.1 Problem Definition

Given a source (S) and a set of target (T) languages, the goal is to train a classifier using data only in the source language and predict examples from the completely unseen target languages. We assume the target language is unknown during training time, which makes direct translation to target infeasible. In this context, we use code-switching (c​scs) to augment the monolingual source data. Thus, the input and output of our problem can be defined as:  
Input: Xu​tSX\_{ut}^{S}, yiSy\_{i}^{S}, ys​lSy\_{sl}^{S}  
Code-Switched Input: Xu​tc​sX\_{ut}^{cs}, yic​sy\_{i}^{cs}, ys​lc​sy\_{sl}^{cs}  
Output: yiTy\_{i}^{T}, ys​lTy\_{sl}^{T} ←p​r​e​d​i​c​t​(Xu​tT)\\leftarrow predict(X\_{ut}^{T})  
 where Xu​tX\_{ut} represents sentences, yiy\_{i} their ground truth intent classes, and ys​ly\_{sl} the slot labels for the words in those sentences. An example sentence, its intent class, and slot labels are shown in Figure [2](#S0.F2 "Figure 2 ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling").

### 3.2 Multilingual Code-Switching

Multilingual masked language models, such as mBERT [Devlin et al. (2019)](#bib.bib12 ""), are trained using large datasets of publicly available unlabeled corpora such as Wikipedia. Such corpora largely remain monolingual at the sentence level because the presence of intra-sentence code-switched data in written texts is likely scarce. The masked words that needed to be predicted usually are in the same language as their surrounding words. We study how code-switching can enhance the language neutrality of such language models by augmenting it with artificially code-switched data for fine-tuning it to a downstream task. Algorithm [1](#algorithm1 "In 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") explains this code-switching process at the chunk-level. When using slot filling datasets, slot labels that are grouped by BIO [Ramshaw and Marcus (1999)](#bib.bib43 "") tags constitute natural chunks. To summarize the algorithm, we take a sentence, take each chunk from that sentence, perform a translation into a random language using Google’s NMT system [Wu et al. (2016)](#bib.bib58 ""), and align the slot labels to fit the translation. At the chunk-level, we use a direct alignment. i.e., the BIO-tagged labels are recreated for the translated phrase based on the word tokens. More complex methods can be applied here to improve the alignment of the slot labels such as fast-align [Dyer et al. (2013)](#bib.bib14 "") or soft-align [Xu et al. (2020)](#bib.bib62 ""). Code-Switching at the word-level essentially translates every word randomly, while at the sentence-level translates the entire sentence. During the experimental evaluation process, to build a language neutral model using monolingual source of English data, all 88 target languages are excluded from the code-switching procedure to avoid unfair model comparisons, i.e. remove target languages from l​s​e​tlset in Algorithm [1](#algorithm1 "In 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling").

Complexity: The augmentation process is repeated kk times per sentence producing a new augmented dataset of size k×nk\\times n, where nn is the size of the original dataset, i.e. space complexity of 𝒪⁡(k×n)\\mathcal{O}(k\\times n). Algorithm [1](#algorithm1 "In 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") has a runtime complexity of 𝒪⁡(k×n×t​r​a​n​s​l​a​t​i​o​n​s/s​e​n​t​e​n​c​e)\\mathcal{O}(k\\times n\\times translations/sentence) steps assuming constant time for alignment. Word-level requires as many translations as the number of words but sentence-level requires only one. An increase in the dataset size also increases the training time, but an advantage is one model fits all languages.

Algorithm 1 Data Augmentation via Multilingual Code-Switching (Chunk-Level)

Input: Xu​te​nX\_{ut}^{en}, yie​ny\_{i}^{en}, ys​le​ny\_{sl}^{en} 

Output: Xu​tc​sX\_{ut}^{cs}, yic​sy\_{i}^{cs}, ys​lc​sy\_{sl}^{cs} 

Xu​tc​s←∅X\_{ut}^{cs}\\leftarrow\\emptyset, yic​s←∅y\_{i}^{cs}\\leftarrow\\emptyset, ys​lc​s←∅y\_{sl}^{cs}\\leftarrow\\emptyset 

l​s​e​t\=g​o​o​g​l​e​t​r​a​n​s.l​a​n​g​u​a​g​e​s−lTlset=googletrans.languages-l\_{T} 

for *i∈1..ki\\in 1..\\ k* do 

   for *i∈1..l​e​n​(Xu​te​n)i\\in 1..\\ len(X\_{ut}^{en})* do 

      Gc​s←∅G^{cs}\\leftarrow\\emptyset, Lc​s←∅L^{cs}\\leftarrow\\emptyset 

      c​h​u​n​k​s\=s​l​o​t​\_​c​h​u​n​k​s​(Xu​te​n​\[i\],ys​le​n​\[i\])chunks=slot\\\_chunks(X\_{ut}^{en}\[i\],\\ y\_{sl}^{en}\[i\]) 

      for *c∈c​h​u​n​k​sc\\in chunks* do 

         l←r​a​n​d​o​m.c​h​o​i​c​e​(l​s​e​t)l\\leftarrow random.choice(lset) 

         t←t​r​a​n​s​l​a​t​e​(c,l)t\\leftarrow translate(c,l) 

         Gc​s←Gc​s∪tG^{cs}\\leftarrow G^{cs}\\cup t 

         Lc​s←Lc​s∪a​l​i​g​n​\_​l​a​b​e​l​(c,t)L^{cs}\\leftarrow L^{cs}\\cup align\\\_label(c,t) 

      end for 

      Xu​tc​s←Xu​tc​s∪Gc​sX\_{ut}^{cs}\\leftarrow X\_{ut}^{cs}\\ \\cup\\ G^{cs} 

      yic​s←yic​s∪yic​s​\[i\]y\_{i}^{cs}\\leftarrow y\_{i}^{cs}\\ \\cup\\ y\_{i}^{cs}\[i\] 

      ys​lc​s←ys​lc​s∪Lc​sy\_{sl}^{cs}\\leftarrow y\_{sl}^{cs}\\ \\cup\\ L^{cs} 

   end for 

end for 

| Group Name             | Languages                                                                        |
| ---------------------- | -------------------------------------------------------------------------------- |
| Afro-Asiatic           | Arabic (ar), Amharic (am), Hebrew (he), Somali (so)                              |
| Germanic               | German (de), Dutch (nl), Danish (da), Swedish (sv), Norwegian (no)               |
| Indo-Aryan             | Hindi (hi), Bengali (bn), Marathi (mr), Nepali (ne), Gujarati (gu), Punjabi (pa) |
| Romance                | Spanish (es), Portuguese (pt), French (fr), Italian (it), Romanian (ro)          |
| Sino-Tibetan & Japonic | Chinese (zh-cn), Japanese (ja), Korean (ko)                                      |
| Turkic                 | Turkish (tr), Azerbaijani (az), Uyghur (ug), Kazakh (kk)                         |

Table 1: Selected language families to evaluate their impact on a target language.

| Language |     |      |        |       |       |    |    |
| -------- | --- | ---- | ------ | ----- | ----- | -- | -- |
| train    | dev | test | train  | dev   | test  |    |    |
| 4488     | 490 | 893  | 50755  | 5445  | 9164  | 18 | 84 |
| 4488     | 490 | 893  | 55197  | 5927  | 10338 | 18 | 84 |
| 4488     | 490 | 893  | 55052  | 5909  | 10228 | 18 | 84 |
| 4488     | 490 | 893  | 51111  | 5517  | 9383  | 18 | 84 |
| 4488     | 490 | 893  | 55909  | 5769  | 10511 | 18 | 84 |
| 4488     | 490 | 893  | 88194  | 9652  | 16710 | 18 | 84 |
| 4488     | 490 | 893  | 133890 | 14416 | 25939 | 18 | 84 |
| 1440     | 160 | 893  | 16422  | 1753  | 9755  | 17 | 75 |
| 578      | 60  | 715  | 6132   | 686   | 7683  | 17 | 71 |
| 3518     | 490 | -    | 16369  | 4242  | -     | 2  | 5  |
| -        | -   | 520  | -      | -     | 2834  | 2  | 5  |

Table 2: Datasets and statistics.

### 3.3 Language Families

A language family is defined as a group of related languages that are likely coming from the same parent. For example, Portuguese, Spanish, French, Italian, and Romanian are daughter languages derived from Latin [Rowe and Levine (2017)](#bib.bib45 ""). We use language families to study their impact on the target languages. We augment the source language with code-switching from a particular language family. For instance, code-switching the English dataset with Turkic language family and testing on Japanese can reveal how closely the two are aligned in the vector space of a pre-trained multilingual model. From a set of 5 distinct language families, we select a total of 6 groups of languages: Afro-Asiatic [Voegelin and Voegelin (1976)](#bib.bib55 ""), Germanic [Harbert (2006)](#bib.bib20 ""), Indo-Aryan [Masica (1993)](#bib.bib31 ""), Romance [Elcock and Green (1960)](#bib.bib15 ""), Sino-Tibetan and Japonic [Shafer (1955)](#bib.bib47 ""); [Miller (1967)](#bib.bib33 ""), and Turkic [Johanson and Johanson (2015)](#bib.bib24 ""). Germanic, Romance, and Indo-Aryan are branches of the Indo-European language family. Language groups and their selected daughter languages are shown in Table [1](#S3.T1 "Table 1 ‣ 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). Each group is selected based on a target language in the dataset and Afro-Asiatic family is added as an extra group. In experiments, l​s​e​tlset in Algorithm [1](#algorithm1 "In 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") will be assigned languages from a specific family.

| Intent Acc. | mm    |        |        |        |        |        |        |       |
| ----------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ----- |
| 94.42       | 94.29 | 79.53  | 73.75  | 92.90  | 93.86  | 67.06  | 69.71  | 83.19 |
| 95.03       | 94.51 | 80.54  | 73.57  | 93.48  | 93.33  | 73.53  | 71.05  | 84.38 |
| 94.18       | 93.92 | 81.67  | 75.48  | 92.54  | 94.18  | 81.19  | 74.22  | 85.92 |
| 94.60       | 93.53 | 81.21  | 75.01  | 93.10  | 93.24  | 82.37  | 75.11  | 86.02 |
| 95.12       | 95.27 | 83.88  | 74.27  | 94.20  | 93.48  | 82.73  | 77.51  | 87.06 |
| 95.48       | 94.51 | 84.43♠ | 76.48♠ | 94.15♠ | 94.89♠ | 85.37♠ | 78.04♠ | 87.92 |
| 94.02       | 93.84 | 90.21  | 84.19  | 95.66  | 94.54  | 85.08  | 85.79  | 90.42 |
| 94.16       | 94.24 | 91.56  | 85.98  | 95.75  | 95.01  | 86.45  | 84.95  | 91.01 |
| 95.48       | 95.41 | 91.60  | 87.17  | 95.34  | 94.60  | 87.94  | 85.93  | 91.68 |
| es          | de    | zh     | ja     | pt     | fr     | hi     | tr     | AVG   |
| 96.16       | 96.73 | 83.12  | 78.81  | 95.63  | 95.40  | 77.05  | 88.09  | 88.87 |
| 96.12       | 96.76 | 84.95  | 79.60  | 95.76  | 95.76  | 77.63  | 88.92  | 89.44 |
| 95.81       | 96.33 | 85.46  | 79.33  | 96.27  | 95.08  | 79.10  | 86.86  | 89.28 |
| 96.57       | 96.92 | 86.32  | 79.52  | 96.65  | 95.84  | 81.94  | 89.84  | 90.45 |
| 96.68       | 96.82 | 87.10  | 80.00  | 96.46  | 96.31  | 80.95  | 91.60  | 90.51 |
| 96.09       | 96.56 | 88.61♠ | 82.28♠ | 96.01  | 95.94  | 82.28♠ | 90.45♠ | 91.03 |
| 96.89       | 96.04 | 93.48  | 85.29  | 96.35  | 96.02  | 82.03  | 91.21  | 92.16 |
| 96.92       | 95.66 | 93.64  | 87.84  | 96.11  | 95.95  | 82.98  | 91.15  | 92.53 |
| 96.98       | 96.27 | 93.37  | 85.87  | 95.88  | 95.44  | 82.00  | 91.31  | 92.14 |

Table 3: Performance evaluation of code-switching with setting k\=5k=5. C​SCS: Code-Switching. Reported scores are average of 5 independent runs (including a separate code-switched data for each run). mm = number of distinct models to be trained. \*: modified BERT-based implementations [Chen et al. (2019)](#bib.bib5 ""); [Xu et al. (2020)](#bib.bib62 "").  
♠: The difference is significant with p < 0.05 using Tukey HSD (conducted between J​o​i​n​te​n−o​n​l​yJoint\_{en-only} + CCS versus J​o​i​n​te​n−o​n​l​yJoint\_{en-only} Baseline for each language). 

### 3.4 Joint Training

Joint training is traditionally used for intent prediction and slot filling to exploit the correlation between the two tasks. This is done by feeding the feature vectors of one model to another or by sharing layers of a neural network followed by training the tasks together. So, a standard joint model loss can be defined as a combination of intent (LiL\_{i}) and slot (Ls​lL\_{sl}) losses. i.e., L\=α​Li+β​Ls​lL=\\alpha L\_{i}+\\beta L\_{sl}, where α\\alpha and β\\beta are corresponding task weights. Prior works [Goo et al. (2018)](#bib.bib18 ""); [Schuster et al. (2019)](#bib.bib46 ""); [Liu and Lane (2016)](#bib.bib29 ""); [Haihong et al. (2019)](#bib.bib19 "") that use BiLSTM or RNN are now modified to BERT-based implementations explored in more recent works [Chen et al. (2019)](#bib.bib5 ""); [Hardalov et al. (2020)](#bib.bib21 ""); [Xu et al. (2020)](#bib.bib62 ""). A standard J​o​i​n​tJoint model consists of BERT outputs from the final hidden state (classification (CLS) token for intent and mm word tokens for slots) fed to linear layers to get intent and slot predictions. Assuming hc​l​sh\_{cls} represents the CLS token and hmh\_{m} represents a token from the remaining word-level tokens, the BERT model outputs are defined as [Chen et al. (2019)](#bib.bib5 ""); [Xu et al. (2020)](#bib.bib62 ""):

pi\=s​o​f​t​m​a​x​(Wi​hc​l​s+bi)pms​l\=softmax(Ws​lhm+bs​l)∀m\\displaystyle\\begin{split}p^{i}&=softmax(W^{i}h\_{cls}+b^{i})\\\\ p^{sl}\_{m}&=softmax(W^{sl}h\_{m}+b^{sl})\\ \\ \\forall m\\end{split}

(1)

with a multi-class cross-entropy loss33 3 L\=−1n∑i\=1n\[ylogy^\]L=-\\frac{1}{n}\\sum\_{i=1}^{n}\[y\\log\\hat{y}\] for both intent (LiL\_{i}) and slots (Ls​lL\_{sl}). We will use this model as our baseline for joint training. Our goal will be to show that code-switching on top of joint training improves the performance. The output of Algorithm [1](#algorithm1 "In 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") will be the input used for joint training on BERT for code-switched experiments.

| Intent Acc.                                  | ht    |
| -------------------------------------------- | ----- |
| English-Only Baseline\*                      | 56.12 |
| Chunk-level CS (CCS)                         | 63.15 |
| J​o​i​n​te​n−o​n​l​yJoint\_{en-only}\* + CCS | 63.73 |
| Translate-Train (TT)\*                       | 62.58 |
| Slot F1                                      | ht    |
| English-Only Baseline\*                      | 68.72 |
| Chunk-level CS (CCS)                         | 70.27 |
| J​o​i​n​te​n−o​n​l​yJoint\_{en-only}\* + CCS | 70.02 |
| Translate-Train (TT)\*                       | 69.96 |

Table 4: Performance on disaster data in Haitian Creole (ht). C​SCS = Code-Switching. Reported scores are average of 5 independent runs (\*: modified BERT-based).

## 4 Datasets

### 4.1 Benchmark Dataset

We use the latest multilingual benchmark dataset of MultiATIS++ [Xu et al. (2020)](#bib.bib62 ""), which was created by manually translating the original ATIS [Price (1990)](#bib.bib41 "") dataset from English (en) to 8 other languages: Spanish (es), Portuguese (pt), German (de), French (fr), Chinese (zh), Japanese (ja), Hindi (hi), and Turkish (tr). The dataset consists of utterances for each language with an ‘intent’ label for ‘flight intent’ and ‘slot’ labels for the word tokens in BIO [Ramshaw and Marcus (1999)](#bib.bib43 "") format. A sample datapoint in English is shown in Figure [2](#S0.F2 "Figure 2 ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling").

### 4.2 New Dataset for Disaster NLU

We construct a new intent and slot filling dataset of tweets collected during natural disasters, in two languages: English and Haitian Creole. The tweets originally were released by Appen44 4 [https://appen.com/datasets/combined-disaster-response-data/](https://appen.com/datasets/combined-disaster-response-data/ ""). For English, a language expert coded the tweets, and for Haitian Creole, we used Amazon Mechanical Turk with five annotators. Intent classes include: ‘request’ and ‘others’. Slot filling consists of 55 labels: ‘medical\_help’, ‘food’, ‘water’, ‘shelter’, and ‘other\_aid’. Table [2](#S3.T2 "Table 2 ‣ 3.2 Multilingual Code-Switching ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") provides the dataset statistics.

![Refer to caption](2103.07792v2/group.png)

Figure 3: Impact of different language groups on the target languages.

## 5 Experimental Setup

We use the traditional cross-lingual task setting where each experiment consists of a source language and a target language. A model is trained on the source data (English) and evaluated on the target data (8 other languages). For code-switching experiments, an English text is augmented with multilingual code-switching before training. Our implementation is in PyTorch [Paszke et al. (2019)](#bib.bib38 "") and we use the pre-trained bert-base-multilingual-uncased [Devlin et al. (2019)](#bib.bib12 "") with BertForSequenceClassification [Wolf et al. (2020)](#bib.bib57 "") as the mBERT model. Maximum epoch is set to 2525 with an early stopping patience of 55, batch size of 3232, and Adam optimizer [Kingma and Ba (2014)](#bib.bib27 "") with a learning rate of 5​e−55\\mathrm{e}{-5}. We select the best model on the validation set. Consistent with the metrics reported for intent prediction and slot filling evaluation in the past, we also use accuracy for intent and micro F1 to measure slot performance.

### 5.1 Baselines & Upper Bound

Since we assume that target language is not known before hand, Translate-Train (TT) [Xu et al. (2020)](#bib.bib62 "") method is not a suitable baseline. Rather, we set this to be an upper bound, i.e. translating to the target language and fine-tuning the model should intuitively outperform a generic model. Additionally, we add code-switching to this TT model to assess if augmentation negatively impacts its performance. The zero-shot baselines for the code-switching experiments use an English-Only [Xu et al. (2020)](#bib.bib62 "") model, which is fine-tuned over the pre-trained mBERT separately for each task and an English-only Joint model [Chen et al. (2019)](#bib.bib5 "").

| CS (k=5) | MTT      | 𝐉𝐨𝐢𝐧𝐭𝐞𝐧\\mathbf{Joint\_{en}} | 𝐉𝐨𝐢𝐧𝐭𝐜𝐬\\mathbf{Joint\_{cs}} | 𝐉𝐨𝐢𝐧𝐭𝐓𝐓\\mathbf{Joint\_{TT}} |
| -------- | -------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 00:11:50 | 01:06:50 | 00:11:04                     |                              |                              |

Table 5: Runtime on Google Colab (K80 GPU for training joint models). M​T​TMTT: Machine Translation to Target. Note that M​T​TMTT and JT​TJ\_{TT} are for one target language (averaged).

## 6 Results & Discussion

### 6.1 Effect of Multilingual Code-Switching

Table [3](#S3.T3 "Table 3 ‣ 3.3 Language Families ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") describes performance evaluation on the MultiATIS++ dataset. When compared to the state-of-the-art jointly trained English-only baseline, we see a +4.2%+4.2\\% boost in intent accuracy and +1.8%+1.8\\% boost in slot F1 scores on average by augmenting the dataset via multilingual code-switching without requiring the target language. From the significance tests, except for Spanish and German, all other languages were helped by code-switching for intent detection. For slot filling, improvement on Portuguese and French went insignificant. This suggests that code-switching primarily helped languages that are morphologically more different as compared to the source language (English). For example, Hindi and Turkish have the highest intent performance improvement of +16.1%+16.1\\% and +9.8%+9.8\\% respectively. And for slots, Hindi and Chinese with +6.0%+6.0\\% and +4.3%+4.3\\% respectively. Japanese showed +4%+4\\% improvement for intent and +3.4%+3.4\\% for slots.

The running time of the models in Table [5](#S5.T5 "Table 5 ‣ 5.1 Baselines & Upper Bound ‣ 5 Experimental Setup ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") show that code-switching is expensive which can take up to 55 hours for k\=5k=5. Its training is also expensive because there is kk times more data as compared to the monolingual source data. Increasing the number of code-switchings (kk) for a sentence from 55 to 5050 improved the performance by +1%+1\\%, while increasing the run-time by a large margin. So, parameter kk should be picked appropriately. Albeit this time cost is for training, with benefits at the prediction stage for real world problems.

In the translate-train (upper bound) scenario, it is not immediately clear if augmentation can help, because data in the same language as the target is always preferred over other languages, or code-switched. However, we show in Table [3](#S3.T3 "Table 3 ‣ 3.3 Language Families ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") that augmentation did not hinder the performance.

For both intent and slot performance, chunk-level model remained robust across the languages. For intent, difference between word-level and sentence-level was insignificant. For slot, sentence-level was in par with chunk-level on average. Thus, we think that code-switching at chunk-level is safer for avoiding semantic discrepancies (as in the word-level) while also capturing better intra-sentence language neutrality.

![Refer to caption](2103.07792v2/intent_errw2.png)

Figure 4: Impact of code-switching on intent classes.

![Refer to caption](2103.07792v2/slot_errw2.png)

Figure 5: Impact of code-switching on slot labels.

![Refer to caption](2103.07792v2/freeze.png)

Figure 6: Freezing earlier layers and unfreezing a few at the top of the transformer appear to be most optimal.

### 6.2 Evaluation on Disaster Dataset

We found that disaster data is more challenging when compared to the ATIS dataset for transfer learning in NLU. The predictive performance is shown in Table [4](#S3.T4 "Table 4 ‣ 3.4 Joint Training ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). Code-Switching improved intent accuracy by +12.5%+12.5\\% and slot F1 by +2.3%+2.3\\%, which is promising considering that they are tweets. Joint training added +0.9%+0.9\\% improvement to intent accuracy, however did not seem to help slot F1. This might imply a lack of strong correlation between the two tasks, i.e. a mention of ‘food’ or ‘shelter’ in a tweet may not always mean that it is a ‘request’ or vice-versa. The upper bound of translate-train method did not perform any better than the randomly code-switched model which seemed counter-intuitive. This might be due to the lack of strong representation for Haitian Creole in the pre-trained model, although it is similar to French.

### 6.3 Impact of Language Families

Results of language family analysis are shown in Figure [3](#S4.F3 "Figure 3 ‣ 4.2 New Dataset for Disaster NLU ‣ 4 Datasets ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). The input in English is independently code-switched using 6 different language families. Note that the target language is always excluded from the group when evaluating on the same, i.e. Hindi is excluded from Indo-Aryan family when that family is being evaluated on it. Translate-train model is provided as a frame of reference and upper bound. We dropped French and Portuguese from the chart as they fall in to Romance family similar to Spanish. Results show the language families helped their corresponding languages, i.e. Romance helped Spanish, Germanic helped German, and so on; with the exception of Chinese and Japanese. In both cases, Turkic language family helped better than others.

### 6.4 Error Analysis

Selecting intent classes with support \>10\>10, Figure [4](#S6.F4 "Figure 4 ‣ 6.1 Effect of Multilingual Code-Switching ‣ 6 Results & Discussion ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling") shows how each class is positively or negatively impacted by code-switching. Improvement was primarily on ‘airfare’, ‘distance’ ‘capacity’, ‘airline’, and ‘ground\_service’ which had longer sentences such as ‘Please tell me which airline has the most departures from Atlanta’ when compared to ‘abbreviations’ and ‘airport’ classes that included very short phrases like ‘What does EA mean?’ However, note that, Spanish and German did not improve much; aligning with our results in Table [3](#S3.T3 "Table 3 ‣ 3.3 Language Families ‣ 3 Methodology ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). For slot labels in Figure [5](#S6.F5 "Figure 5 ‣ 6.1 Effect of Multilingual Code-Switching ‣ 6 Results & Discussion ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"), we selected the ones with support \>50\>50 and that have different characteristics, e.g. ‘name’, ‘code’, etc. The overall trend in slot performance shows improvements for labels such as ‘day\_name’, ‘airport\_code’, and ‘city\_name’ and slight variations in labels such as ‘fight\_number’ and ‘period\_of\_day’; implying textual slots benefiting over numeric ones.

### 6.5 Hyperparameter Tuning

For joint training with same task weights, we tuned α\\alpha and β\\beta using grid search to see the strength of correlation between the tasks. For intent, the (α,β\\alpha,\\beta) combination of (1.0,0.61.0,0.6) performed well, while (1.0,1.01.0,1.0) for slots. This suggests that intent benefiting slot might be slightly more than slot benefiting intent. Additionally, during fine-tuning, freezing the layers of the transformer affected the model performance as shown in Figure [6](#S6.F6 "Figure 6 ‣ 6.1 Effect of Multilingual Code-Switching ‣ 6 Results & Discussion ‣ Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling"). Keeping the first 88 layers frozen gave the best performance. By freezing the earlier layers, the transformer can retain its most fundamental feature information gained from the massive pre-training step, and by unfreezing some top layers, it can undergo fine-tuning.

## 7 Conclusion & Future Work

This study shows that augmenting the monolingual input data with multilingual code-switching via random translations helps a zero-shot model to be more language neutral when evaluated on unseen languages. This approach enhanced the generalizability of pre-trained mBERT when fine-tuning for downstream tasks of intent detection and slot filling. We presented an application of this method using a new annotated dataset of disaster tweets. Further, we studied code-switching with language families and their impact on specific target languages, which can be used to enhance the zero-shot generalizability of models created for low-resource languages. Expanding to XLM-R and similar approaches to improve masked language model training by addressing code-switching during pre-training and releasing a larger dataset of annotated disaster tweets in more languages are planned for future work.  

## 8 Acknowledgement

We thank U.S. National Science Foundation grants IIS-1815459 and IIS-1657379 for partially supporting this research. We also thank Ming Sun and Alexis Conneau for giving valuable insights on multilingual model training. We also acknowledge ARGO team as the experiments were run on ARGO, a research computing cluster provided by the Office of Research Computing at George Mason University.

## References

*   Aguilar and Solorio (2019) Gustavo Aguilar and Thamar Solorio. 2019. From english to code-switching: Transfer learning with strong morphological clues. *arXiv preprint arXiv:1909.05158*.
*   Ball and Garrette (2018) Kelsey Ball and Dan Garrette. 2018. Part-of-speech tagging for code-switched, transliterated texts without explicit language identification. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3084–3089.
*   Bojanowski et al. (2017) Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics*, 5:135–146.
*   Chaudhary et al. (2020) Aditi Chaudhary, Karthik Raman, Krishna Srinivasan, and Jiecao Chen. 2020. Dict-mlm: Improved multilingual pre-training using bilingual dictionaries. *arXiv preprint arXiv:2010.12566*.
*   Chen et al. (2019) Qian Chen, Zhu Zhuo, and Wen Wang. 2019. Bert for joint intent classification and slot filling. *arXiv preprint arXiv:1902.10909*.
*   Chen et al. (2018) Xilun Chen, Yu Sun, Ben Athiwaratkun, Claire Cardie, and Kilian Weinberger. 2018. Adversarial deep averaging networks for cross-lingual sentiment classification. *Transactions of the Association for Computational Linguistics*, 6:557–570.
*   Conneau et al. (2020a) Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020a. Unsupervised cross-lingual representation learning at scale. *arXiv preprint arXiv:1911.02116*.
*   Conneau et al. (2017) Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2017. Word translation without parallel data. *arXiv preprint arXiv:1710.04087*.
*   Conneau et al. (2018) Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating cross-lingual sentence representations. *arXiv preprint arXiv:1809.05053*.
*   Conneau et al. (2020b) Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020b. [Emerging cross-lingual structure in pretrained language models](https://doi.org/10.18653/v1/2020.acl-main.536 ""). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6022–6034, Online. Association for Computational Linguistics.
*   Coucke et al. (2018) Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, et al. 2018. Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces. *arXiv preprint arXiv:1805.10190*.
*   Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186.
*   Dufter and Schütze (2020) Philipp Dufter and Hinrich Schütze. 2020. Identifying elements essential for bert’s multilinguality. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4423–4437.
*   Dyer et al. (2013) Chris Dyer, Victor Chahuneau, and Noah A Smith. 2013. A simple, fast, and effective reparameterization of ibm model 2. In *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 644–648.
*   Elcock and Green (1960) William Denis Elcock and John N Green. 1960. *The romance languages*. Faber & Faber London.
*   Eriguchi et al. (2018) Akiko Eriguchi, Melvin Johnson, Orhan Firat, Hideto Kazawa, and Wolfgang Macherey. 2018. Zero-shot cross-lingual classification using multilingual neural machine translation. *arXiv preprint arXiv:1809.04686*.
*   Ganin et al. (2016) Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. *The Journal of Machine Learning Research*, 17(1):2096–2030.
*   Goo et al. (2018) Chih-Wen Goo, Guang Gao, Yun-Kai Hsu, Chih-Li Huo, Tsung-Chieh Chen, Keng-Wei Hsu, and Yun-Nung Chen. 2018. Slot-gated modeling for joint slot filling and intent prediction. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, pages 753–757.
*   Haihong et al. (2019) E Haihong, Peiqing Niu, Zhongfu Chen, and Meina Song. 2019. A novel bi-directional interrelated model for joint intent detection and slot filling. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 5467–5471.
*   Harbert (2006) Wayne Harbert. 2006. *The Germanic Languages*. Cambridge University Press.
*   Hardalov et al. (2020) Momchil Hardalov, Ivan Koychev, and Preslav Nakov. 2020. Enriched pre-trained transformers for joint slot filling and intent detection. *arXiv preprint arXiv:2004.14848*.
*   He et al. (2013) Xiaodong He, Li Deng, Dilek Hakkani-Tur, and Gokhan Tur. 2013. Multi-style adaptive training for robust cross-lingual spoken language understanding. In *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*, pages 8342–8346. IEEE.
*   Jiang et al. (2020) Zhengbao Jiang, Antonios Anastasopoulos, Jun Araki, Haibo Ding, and Graham Neubig. 2020. Multilingual factual knowledge retrieval from pretrained language models. *arXiv preprint arXiv:2010.06189*.
*   Johanson and Johanson (2015) Lars Johanson and Éva Ágnes Csató Johanson. 2015. *The Turkic Languages*. Routledge.
*   Joshi et al. (2016) Aditya Joshi, Ameya Prabhu, Manish Shrivastava, and Vasudeva Varma. 2016. Towards sub-word level compositions for sentiment analysis of hindi-english code mixed text. In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, pages 2482–2491.
*   KhudaBukhsh et al. (2020) Ashiqur R KhudaBukhsh, Shriphani Palakodety, and Jaime G Carbonell. 2020. Harnessing code switching to transcend the linguistic barrier. *arXiv preprint arXiv:2001.11258*.
*   Kingma and Ba (2014) Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
*   Krishnan et al. (2020) Jitin Krishnan, Hemant Purohit, and Huzefa Rangwala. 2020. Unsupervised and interpretable domain adaptation to rapidly filter social web data for emergency services. In *ASONAM*.
*   Liu and Lane (2016) Bing Liu and Ian Lane. 2016. Attention-based recurrent neural network models for joint intent detection and slot filling. *arXiv preprint arXiv:1609.01454*.
*   Mager et al. (2019) Manuel Mager, Özlem Çetinoğlu, and Katharina Kann. 2019. Subword-level language identification for intra-word code-switching. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2005–2011.
*   Masica (1993) Colin P Masica. 1993. *The indo-aryan languages*. Cambridge University Press.
*   Mave et al. (2018) Deepthi Mave, Suraj Maharjan, and Thamar Solorio. 2018. Language identification and analysis of code-switched social media text. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 51–61.
*   Miller (1967) Roy Andrew Miller. 1967. *The Japanese Language*. University of Chicago Press Chicago.
*   Muis et al. (2018) Aldrian Obaja Muis, Naoki Otani, Nidhi Vyas, Ruochen Xu, Yiming Yang, Teruko Mitamura, and Eduard Hovy. 2018. Low-resource cross-lingual event type detection via distant supervision with minimal effort. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 70–82.
*   Nguyen et al. (2017) Dat Tien Nguyen, Kamla Al-Mannai, Shafiq R Joty, Hassan Sajjad, Muhammad Imran, and Prasenjit Mitra. 2017. Robust classification of crisis-related data on social networks using convolutional neural networks. *ICWSM*, 31(3):632–635.
*   Ni et al. (2017) Jian Ni, Georgiana Dinu, and Radu Florian. 2017. Weakly supervised cross-lingual named entity recognition via effective annotation and representation projection. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1470–1480.
*   Pamungkas and Patti (2019) Endang Wahyu Pamungkas and Viviana Patti. 2019. Cross-domain and cross-lingual abusive language detection: A hybrid approach with deep learning and a multilingual lexicon. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, pages 363–370.
*   Paszke et al. (2019) Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. [Pytorch: An imperative style, high-performance deep learning library](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf ""). In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems 32*, pages 8024–8035. Curran Associates, Inc.
*   Pires et al. (2019) Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual bert? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4996–5001.
*   Plank and Agić (2018) Barbara Plank and Željko Agić. 2018. Distant supervision from disparate sources for low-resource part-of-speech tagging. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 614–620.
*   Price (1990) Patti Price. 1990. Evaluation of spoken language systems: The atis domain. In *Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27, 1990*.
*   Qin et al. (2020) Libo Qin, Minheng Ni, Yue Zhang, and Wanxiang Che. 2020. Cosda-ml: Multi-lingual code-switching data augmentation for zero-shot cross-lingual nlp. *arXiv preprint arXiv:2006.06402*.
*   Ramshaw and Marcus (1999) Lance A Ramshaw and Mitchell P Marcus. 1999. Text chunking using transformation-based learning. In *Natural language processing using very large corpora*, pages 157–176. Springer.
*   Romera-Paredes and Torr (2015) Bernardino Romera-Paredes and Philip Torr. 2015. An embarrassingly simple approach to zero-shot learning. In *International Conference on Machine Learning*, pages 2152–2161.
*   Rowe and Levine (2017) Bruce Rowe and Diane Levine. 2017. A concise introduction to linguistics. *Routledge. pp. 340–341*.
*   Schuster et al. (2019) Sebastian Schuster, Sonal Gupta, Rushin Shah, and Mike Lewis. 2019. Cross-lingual transfer learning for multilingual task oriented dialog. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 3795–3805.
*   Shafer (1955) Robert Shafer. 1955. Classification of the sino-tibetan languages. *Word*, 11(1):94–111.
*   Shah et al. (2010) Rushin Shah, Bo Lin, Anatole Gershman, and Robert Frederking. 2010. Synergy: a named entity recognition system for resource-scarce languages such as swahili using online machine translation. In *Proceedings of the Second Workshop on African Language Technology (AfLaT 2010)*, pages 21–26.
*   Soto and Hirschberg (2018) Victor Soto and Julia Hirschberg. 2018. Joint part-of-speech and language id tagging for code-switched data. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 1–10.
*   Srivastava et al. (2018) Shashank Srivastava, Igor Labutov, and Tom Mitchell. 2018. Zero-shot learning of classifiers from natural language quantification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 306–316.
*   Stappen et al. (2020) Lukas Stappen, Fabian Brunn, and Björn Schuller. 2020. Cross-lingual zero-and few-shot hate speech detection utilising frozen transformer language models and axel. *arXiv preprint arXiv:2004.13850*.
*   Täckström et al. (2013) Oscar Täckström, Dipanjan Das, Slav Petrov, Ryan McDonald, and Joakim Nivre. 2013. Token and type constraints for cross-lingual part-of-speech tagging. *Transactions of the Association for Computational Linguistics*, 1:1–12.
*   Tsai et al. (2016) Chen-Tse Tsai, Stephen Mayhew, and Dan Roth. 2016. Cross-lingual named entity recognition via wikification. In *Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning*, pages 219–228.
*   Upadhyay et al. (2018) Shyam Upadhyay, Manaal Faruqui, Gokhan Tür, Hakkani-Tür Dilek, and Larry Heck. 2018. (almost) zero-shot cross-lingual spoken language understanding. In *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 6034–6038. IEEE.
*   Voegelin and Voegelin (1976) Charles Frederick Voegelin and Florence Marie Voegelin. 1976. Classification and index of the world’s languages.
*   Wan (2009) Xiaojun Wan. 2009. Co-training for cross-lingual sentiment classification. In *Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP*, pages 235–243.
*   Wolf et al. (2020) Thomas Wolf, Julien Chaumond, Lysandre Debut, Victor Sanh, Clement Delangue, Anthony Moi, Pierric Cistac, Morgan Funtowicz, Joe Davison, Sam Shleifer, et al. 2020. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45.
*   Wu et al. (2016) Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*.
*   Xian et al. (2017) Yongqin Xian, Bernt Schiele, and Zeynep Akata. 2017. Zero-shot learning-the good, the bad and the ugly. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 4582–4591.
*   Xiao and Guo (2013) Min Xiao and Yuhong Guo. 2013. Semi-supervised representation learning for cross-lingual text classification. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1465–1475.
*   Xie et al. (2018) Jiateng Xie, Zhilin Yang, Graham Neubig, Noah A Smith, and Jaime G Carbonell. 2018. Neural cross-lingual named entity recognition with minimal resources. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 369–379.
*   Xu et al. (2020) Weijia Xu, Batool Haider, and Saab Mansour. 2020. End-to-end slot alignment and recognition for cross-lingual nlu. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5052–5063.
*   Yang et al. (2020) Jian Yang, Shuming Ma, Dongdong Zhang, ShuangZhi Wu, Zhoujun Li, and Ming Zhou. 2020. Alternating language modeling for cross-lingual pre-training. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 9386–9393.
*   Yarowsky et al. (2001) David Yarowsky, Grace Ngai, and Richard Wicentowski. 2001. Inducing multilingual text analysis tools via robust projection across aligned corpora. Technical report, Johns Hopkins Univ Baltimore MD Dept of Computer Science.
*   Yirmibeşoğlu and Eryiğit (2018) Zeynep Yirmibeşoğlu and Gülşen Eryiğit. 2018. Detecting code-switching between turkish-english language pair. In *Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text*, pages 110–115.
*   Yu et al. (2018) Katherine Yu, Haoran Li, and Barlas Oguz. 2018. Multilingual seq2seq training with similarity loss for cross-lingual document classification. In *Proceedings of The Third Workshop on Representation Learning for NLP*, pages 175–179.
*   Zirikly and Hagiwara (2015) Ayah Zirikly and Masato Hagiwara. 2015. Cross-lingual transfer of named entity recognizers without parallel corpora. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 390–396.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")