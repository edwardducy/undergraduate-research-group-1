# Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling

Jitin Krishnan Antonios Anastasopoulos Hemant Purohit Huzefa Rangwala

George Mason University

Fairfax, VA, USA

{jkrishn2,antonis,hpurohit,rangwala}@gmu.edu

## Abstract

Predicting user intent and detecting the corresponding slots from text are two key problems in Natural Language Understanding (NLU). In the context of zero-shot learning, this task is typically approached by either using representations from pre-trained multilingual transformers such as mBERT, or by machine translating the source data into the known target language and then fine-tuning. Our work focuses on a particular scenario where the target language is unknown during training. To this goal, we propose a novel method to augment the monolingual source data using multilingual code-switching via random translations to enhance a transformer’s language neutrality when fine-tuning it for a downstream task. This method also helps discover novel insights on how code-switching with different language families around the world impact the performance on the target language. Experiments on the benchmark dataset of MultiATIS++ yielded an average improvement of +4.2% in accuracy for intent task and +1.8% in F1 for slot task using our method over the state-of-the-art across 8 different languages<sup>1</sup>. Furthermore, we present an application of our method for crisis informatics using a new human-annotated tweet dataset of slot filling in English and Haitian Creole, collected during Haiti earthquake disaster<sup>2</sup>.

## 1 Introduction

A cross-lingual setting is typically described as a scenario in which a model trained for a particular task in one language (e.g. English) should be able to generalize well to a different language (e.g.

Japanese). While a semi-supervised solution (Xiao and Guo, 2013; Muis et al., 2018) assumes some target language data is available, a zero-shot solution (Eriguchi et al., 2018; Srivastava et al., 2018; Xu et al., 2020) assumes none is available at training time. This is particularly significant in real world problems such as extracting relevant information during a new disaster (Nguyen et al., 2017; Krishnan et al., 2020) and hate speech detection (Pamungkas and Patti, 2019; Stappen et al., 2020), where the target language might be of low-resource or unknown. In such scenarios, it is crucial that models can generalize well to unseen languages.

Intent prediction and slot filling are important NLU tasks and significant for real world problems. They are studied extensively for goal-oriented dialogue systems currently, such as Amazon’s Alexa, Apple’s Siri, Google Assistant, and Microsoft’s Cortana. Finding the ‘intent’ behind the user’s query and identifying relevant ‘slots’ in the sentence to engage in a dialogue are essential for an effective conversational assistance. For example, users might want to ‘play music’ given the slot labels ‘year’ and ‘artist’ (Coucke et al., 2018), or they may want to ‘book a flight’ given the slot labels ‘airport’ and ‘locations’ (Price, 1990). A strong correlation between the two tasks has made jointly trained models successful (Goo et al., 2018; Haihong et al., 2019; Hardalov et al., 2020; Chen et al., 2019). In a cross-lingual setting, the model should be able to learn this joint task in one language and transfer knowledge to another (Upadhyay et al., 2018; Schuster et al., 2019; Xu et al., 2020). This is the premise of our work.

Highly effective multilingual models such as mBERT (Devlin et al., 2019) and XLM-R (Conneau et al., 2020a) have shown success across several multilingual tasks in recent years. In the zero-shot cross-lingual transfer setting with an unknown target language, a typical solution is to use pre-trained transformer models and finetune to the downstream task using the monolingual source data (Xu et al., 2020). However, previous work (Pires et al., 2019) has shown that existing transformer-based representations may exhibit systematic deficiencies for certain language pairs.Previous work (Pires et al., 2019) has shown that existing transformer-based representations may exhibit systematic deficiencies for certain language pairs. Figure 1 shows that the representations across the 12 multi-head attention layers of mBERT are still clustered according to the languages. This leads to a fundamental challenge that we address in this work: enhancing the language neutrality so that the fine-tuned model is generalizable across languages for the downstream task. To this goal, we introduce a data augmentation method via multilingual code-switching, where the original sentence in English is code-switched into randomly selected languages. For example, chunk-level codeswitching creates sentences with phrases in multiple languages as shown in Figure 2. We show that this can lead to a better performance in the zeroshot setting such that mBERT can be fine-tuned for all languages (not just one) with a monolingual source data.

![](images/5b91266895d9cbd34eaae75399c2b645e56cecdafe8b10dc81c2c14e627fde4b.jpg)

<details>
<summary>scatter</summary>

| Category | X (range) | Y (range) |
| --- | --- | --- |
| 0 | 0~100 | 0~100 |
| 2 | 0~100 | 0~100 |
| 4 | 0~100 | 0~100 |
| 6 | 0~100 | 0~100 |
| 8 | 0~100 | 0~100 |
| 10 | 0~100 | 0~100 |
| 12 | 0~100 | 0~100 |
</details>

Figure 1: t-SNE plot of embeddings across the 12 multi-head attention layers of multilingual BERT. Parallelly translated sentences of MutiATIS++ dataset are still clustered according to the languages: English (black), Chinese (cyan), French (blue), German (green), and Japanese (red).

<table><tr><td>Intent</td><td colspan="10">atis_airfare</td></tr><tr><td colspan="11">Original (English)</td></tr><tr><td>Words</td><td>Show</td><td>me</td><td>round</td><td>trip</td><td>fares</td><td>from</td><td>Denver</td><td>to</td><td>Philadelphia</td><td></td></tr><tr><td>Slots</td><td>O</td><td>O</td><td>B-round_trip</td><td>I-round_trip</td><td>O</td><td>O</td><td>B-fromloc.city_name</td><td>O</td><td>B-fromloc.city_name</td><td></td></tr><tr><td colspan="11">Chunk-level Code-Switched</td></tr><tr><td>Words</td><td>给</td><td>我</td><td>看</td><td>看</td><td>tarifas</td><td>desde</td><td>Denver</td><td>إلى</td><td>Филадельфия</td><td></td></tr><tr><td>Slots</td><td>O</td><td>O</td><td>O</td><td>O</td><td>B-round_trip</td><td>O</td><td>O</td><td>B-fromloc.city_name</td><td>O</td><td>B-fromloc.city_name</td></tr></table>

Figure 2: An original example in English from MultiATIS++ dataset and its multilingually code-switched version. In the above code-switching example, the chunks are in Chinese, Punjabi, Spanish, English, Arabic, and Russian. ‘atis\_airfare’ represents an intent class where the user seeks price of a ticket.

Further, we show how code-switching with different language families impact the model’s performance on individual target languages. Crosslingual study of language families largely remains unexplored for NLU tasks. For instance, while it might be intuitive that Sino-Tibetan language family can aid a task in Hindi, results indicating that Turkic language family may help Japanese can reveal intriguing inter-family relationships and how they are aligned in the underlying language model’s vector space.

Contributions: a) We present a data augmentation method via multilingual code-switching to enhance the language neutrality of mBERT for fine-tuning to a downstream NLU task of intent prediction and slot filling. b) By code-switching into different language families, we show that potential relationships between a family and a target language can be identified and studied; which could help foster zero-shot cross-lingual research in low-resource languages. c) We release a new human-annotated tweet dataset, collected during Haiti earthquake disaster, for intent prediction and slot filling in English and Haitian Creole.

Advantages: With enhanced generalizability, our model can be deployed with an out-of-thebox functionality. Previous methods of first machine translation of the source data into the known target language, followed by fine-tuning (referred ‘translate-train’) (Xu et al., 2020; Yarowsky et al.,

2001; Shah et al., 2010; Ni et al., 2017) require a separate model to be trained for each language.

## 2 Related Work

## 2.1 Cross-Lingual Transfer Learning

Researchers have studied cross-lingual tasks in various settings such as sentiment/sequence classifi cation (Wan, 2009; Eriguchi et al., 2018; Yu et al., 2018), named entity recognition (Zirikly and Hagiwara, 2015; Tsai et al., 2016; Xie et al., 2018), parts-of-speech tagging (Yarowsky et al., 2001; Täckström et al., 2013; Plank and Agic´, 2018), and natural language understanding (He et al., 2013; Upadhyay et al., 2018; Xu et al., 2020). The methodology for most of the current approaches for cross-lingual tasks fall into the following three categories: a) multilingual representations from pre-trained or fine-tuned models such as mBERT (Devlin et al., 2019) or XLM-R (Conneau et al., 2020a), b) machine translation followed by alignment (Shah et al., 2010; Yarowsky et al., 2001; Ni et al., 2017), or c) a combination of both (Xu et al., 2020). Before transformer models, effective approaches included domain adversarial training to extract language-agnostic features (Ganin et al., 2016; Chen et al., 2018) and word alignment methods such as MUSE (Conneau et al., 2017) to align fastText word vectors (Bojanowski et al., 2017). Recently, Conneau et al., 2020b has shown that having shared parameters in the top layers of the multi-lingual encoders can be used to align different languages quite effectively on tasks such as XNLI (Conneau et al., 2018).

Monolingual models for joint slot filling and intent prediction have used methods such as attentionbased RNN (Liu and Lane, 2016) and attentionbased BiLSTM with a slot gate (Goo et al., 2018) on benchmark datasets such as ATIS (Price, 1990) and SNIPS (Coucke et al., 2018). These methods have shown that a joint method can enhance both tasks and slot filling can be conditioned on the learned intent. An interrelated mechanism was introduced (Haihong et al., 2019) to iteratively learn the relationship between the two tasks. Recently, BERT-based approaches (Hardalov et al., 2020; Chen et al., 2019) have shown improved results. On the other hand, cross-lingual versions of this joint task include a low-supervision based approach for Hindi and Turkish (Upadhyay et al., 2018), new dataset for Spanish and Thai (Schuster et al., 2019), and the most recent work of MultiATIS++ (Xu et al., 2020) creating a comprehensive dataset in 9 languages; which is used to benchmark our results.

The joint task mentioned above in a pure zeroshot learning is the motivation of our work. Zeroshot is described as the setting where the model sees a new distribution of examples during test time (Xian et al., 2017; Srivastava et al., 2018; Romera-Paredes and Torr, 2015). It is common for machine translation based methods to translate source data to the target language before training. We assume that target language is unknown during training, so that our model is generalizable across languages.

## 2.2 Code-Switching

Linguistic code-switching is a phenomenon where multilingual speakers alternate between languages. Recently, monolingual models have been adapted to code-switched text in several tasks such as entity recognition (Aguilar and Solorio, 2019), part-of speech tagging (Soto and Hirschberg, 2018; Ball and Garrette, 2018), sentiment analysis (Joshi et al., 2016), and language identification (Mave et al., 2018; Yirmibe¸soglu and Eryi˘ git˘ , 2018; Mager et al., 2019). Recently, KhudaBukhsh et al., 2020 have proposed a pipeline to sample code-mixed documents using minimal supervision. Qin et al., 2020 allows randomized code-switching to include the target language. In our context, if the target lan guage is German, we ensure that there is no codeswitching to German during training. We consider this distinction essential to evaluate a true zero-shot learning scenario and prevent any bias. Another recent work by Yang et al., 2020 presents a nonzero-shot approach that performs code-switching to target languages. Jiang et al., 2020 presents presents a code-switching based method to im prove the ability of multilingual language models for factual knowledge retrieval. Code-switching is usually done at the word-level. However, our results favor chunk-level switching over word-leve as the latter may bring more noise to the code switched version when compared to the original meaning of the sentence. Code-switching and other data augmentation techniques have been applied to the pre-training stage in recent works (Chaudhary et al., 2020; Dufter and Schütze, 2020), however we do not address pre-training in this work. Pre trained models such as XLM-R is also likely to be exposed to code-switched data, as it is trained using common-crawl. In this work, we specifically focus on mBERT which largely remain monolingual at the sentence level to identify the impact of code-switching during fine-tuning, in addition to study the impact of language-family-based augmentations.

## 3 Methodology

This section first describes our problem for zeroshot cross-lingual transfer setting, followed by a novel data augmentation method using multilingual code-switching of monolingual source to enhance language neutrality. We then describe language families, followed by the joint training setup.

## 3.1 Problem Definition

Given a source (S) and a set of target (T) languages, the goal is to train a classifier using data only in the source language and predict examples from the completely unseen target languages. We assume the target language is unknown during training time, which makes direct translation to target infeasible. In this context, we use code-switching (cs) to augment the monolingual source data. Thus, the input and output of our problem can be defined as:

Input: $\bar { X } _ { u t } ^ { S } , y _ { i } ^ { S } , y _ { s l } ^ { \bar { S } }$

Code-Switched Input: $X _ { u t } ^ { c s } , y _ { i } ^ { c s } , y _ { s l } ^ { c s }$

Output: $y _ { i } ^ { T } , y _ { s l } ^ { T } \gets p r e d i c t ( X _ { u t } ^ { T } )$

where $X _ { u t }$ represents sentences, $y _ { i }$ their ground truth intent classes, and $y _ { s l }$ the slot labels for the words in those sentences. An example sentence, its intent class, and slot labels are shown in Figure 2.

## 3.2 Multilingual Code-Switching

Multilingual masked language models, such as mBERT (Devlin et al., 2019), are trained using large datasets of publicly available unlabeled corpora such as Wikipedia. Such corpora largely remain monolingual at the sentence level because the presence of intra-sentence code-switched data in written texts is likely scarce. The masked words that needed to be predicted usually are in the same language as their surrounding words. We study how code-switching can enhance the language neutrality of such language models by augmenting it with artificially code-switched data for fine-tuning it to a downstream task. Algorithm 1 explains this codeswitching process at the chunk-level. When using slot filling datasets, slot labels that are grouped by BIO (Ramshaw and Marcus, 1999) tags constitute natural chunks. To summarize the algorithm, we take a sentence, take each chunk from that sentence, perform a translation into a random language using

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Data Augmentation via Multilingual Code-Switching (Chunk-Level)
Input: $X_{ut}^{en}$, $y_i^{en}$, $y_{sl}^{en}$
Output: $X_{ut}^{cs}$, $y_i^{cs}$, $y_{sl}^{cs}$ $X_{ut}^{cs} \leftarrow \emptyset$, $y_i^{cs} \leftarrow \emptyset$, $y_{sl}^{cs} \leftarrow \emptyset$
lset = googletrans.languages - $l_T$
for $i \in 1.. k$ do
    for $i \in 1.. len(X_{ut}^{en})$ do
        $G^{cs} \leftarrow \emptyset$, $L^{cs} \leftarrow \emptyset$
        chunks =
            slot_chunks($X_{ut}^{en}[i]$, $y_{sl}^{en}[i]$)
        for $c \in chunks$ do
            $l \leftarrow random.choice(lset)$ $t \leftarrow translate(c, l)$ $G^{cs} \leftarrow G^{cs} \cup t$ $L^{cs} \leftarrow L^{cs} \cup align\_label(c, t)$
        end
        $X_{ut}^{cs} \leftarrow X_{ut}^{cs} \cup G^{cs}$ $y_i^{cs} \leftarrow y_i^{cs} \cup y_i^{cs}[i]$ $y_{sl}^{cs} \leftarrow y_{sl}^{cs} \cup L^{cs}$
    end
end
</div>

Google’s NMT system (Wu et al., 2016), and align the slot labels to fit the translation. At the chunklevel, we use a direct alignment. i.e., the BIOtagged labels are recreated for the translated phrase based on the word tokens. More complex methods can be applied here to improve the alignment of the slot labels such as fast-align (Dyer et al., 2013) or soft-align (Xu et al., 2020). Code-Switching at the word-level essentially translates every word randomly, while at the sentence-level translates the entire sentence. During the experimental evaluation process, to build a language neutral model using monolingual source of English data, all 8 target languages are excluded from the code-switching procedure to avoid unfair model comparisons, i.e. remove target languages from lset in Algorithm 1. Complexity: The augmentation process is repeated k times per sentence producing a new augmented dataset of size $k \times n .$ , where n is the size of the original dataset, i.e. space complexity of ${ \mathcal { O } } ( k \times n )$ . Algorithm 1 has a runtime complexity of $\mathcal { O } ( k \times n \times t r a n s l a t i o n s / s e n t e n c e )$ steps assuming constant time for alignment. Word-level requires as many translations as the number of words but sentence-level requires only one. An increase in the dataset size also increases the training time, but an advantage is one model fits all languages.

<table><tr><td>Group Name</td><td>Languages</td></tr><tr><td>Afro-Asiatic</td><td>Arabic (ar), Amharic (am), Hebrew (he), Somali (so)</td></tr><tr><td>Germanic</td><td>German (de), Dutch (nl), Danish (da), Swedish (sv), Norwegian (no)</td></tr><tr><td>Indo-Aryan</td><td>Hindi (hi), Bengali (bn), Marathi (mr), Nepali (ne), Gujarati (gu), Punjabi (pa)</td></tr><tr><td>Romance</td><td>Spanish (es), Portuguese (pt), French (fr), Italian (it), Romanian (ro)</td></tr><tr><td>Sino-Tibetan &amp; Japonic</td><td>Chinese (zh-cn), Japanese (ja), Korean (ko)</td></tr><tr><td>Turkic</td><td>Turkish (tr), Azerbaijani (az), Uyghur (ug), Kazakh (kk)</td></tr></table>

Table 1: Selected language families to evaluate their impact on a target language.

<table><tr><td rowspan="2">Language</td><td colspan="3">Utterances</td><td colspan="3">Tokens</td><td rowspan="2">Intents</td><td rowspan="2">Slots</td></tr><tr><td>train</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td></tr><tr><td colspan="9">MultiATIS++ (Xu et al., 2020)</td></tr><tr><td>English</td><td>4488</td><td>490</td><td>893</td><td>50755</td><td>5445</td><td>9164</td><td>18</td><td>84</td></tr><tr><td>Spanish</td><td>4488</td><td>490</td><td>893</td><td>55197</td><td>5927</td><td>10338</td><td>18</td><td>84</td></tr><tr><td>Portuguese</td><td>4488</td><td>490</td><td>893</td><td>55052</td><td>5909</td><td>10228</td><td>18</td><td>84</td></tr><tr><td>German</td><td>4488</td><td>490</td><td>893</td><td>51111</td><td>5517</td><td>9383</td><td>18</td><td>84</td></tr><tr><td>French</td><td>4488</td><td>490</td><td>893</td><td>55909</td><td>5769</td><td>10511</td><td>18</td><td>84</td></tr><tr><td>Chinese</td><td>4488</td><td>490</td><td>893</td><td>88194</td><td>9652</td><td>16710</td><td>18</td><td>84</td></tr><tr><td>Japanese</td><td>4488</td><td>490</td><td>893</td><td>133890</td><td>14416</td><td>25939</td><td>18</td><td>84</td></tr><tr><td>Hindi</td><td>1440</td><td>160</td><td>893</td><td>16422</td><td>1753</td><td>9755</td><td>17</td><td>75</td></tr><tr><td>Turkish</td><td>578</td><td>60</td><td>715</td><td>6132</td><td>686</td><td>7683</td><td>17</td><td>71</td></tr><tr><td colspan="9">Disaster Tweets (New Dataset)</td></tr><tr><td>English</td><td>3518</td><td>490</td><td>-</td><td>16369</td><td>4242</td><td>-</td><td>2</td><td>5</td></tr><tr><td>Haitian Creole</td><td>-</td><td>-</td><td>520</td><td>-</td><td>-</td><td>2834</td><td>2</td><td>5</td></tr></table>

Table 2: Datasets and statistics.

## 3.3 Language Families

A language family is defined as a group of related languages that are likely coming from the same parent. For example, Portuguese, Spanish, French, Italian, and Romanian are daughter languages derived from Latin (Rowe and Levine, 2017). We use language families to study their impact on the target languages. We augment the source language with code-switching from a particular language family. For instance, code-switching the English dataset with Turkic language family and testing on Japanese can reveal how closely the two are aligned in the vector space of a pre-trained multilingual model. From a set of 5 distinct language families, we select a total of 6 groups of languages: Afro-Asiatic (Voegelin and Voegelin, 1976), Germanic (Harbert, 2006), Indo-Aryan (Masica, 1993), Romance (Elcock and Green, 1960), Sino-Tibetan and Japonic (Shafer, 1955; Miller, 1967), and Turkic (Johanson and Johanson, 2015). Germanic, Romance, and Indo-Aryan are branches of the Indo-European language family. Language groups and their selected daughter languages are shown in Table 1. Each group is selected based on a target language in the dataset and Afro-Asiatic family is added as an extra group. In experiments, lset in Algorithm 1 will be assigned languages from a specific family.

## 3.4 Joint Training

Joint training is traditionally used for intent prediction and slot filling to exploit the correlation between the two tasks. This is done by feeding the feature vectors of one model to another or by sharing layers of a neural network followed by training the tasks together. So, a standard joint model loss can be defined as a combination of intent $( L _ { i } )$ and slot $( L _ { s l } )$ losses. i.e., $L = \alpha L _ { i } + \beta L _ { s l }$ , where α and β are corresponding task weights. Prior works (Goo et al., 2018; Schuster et al., 2019; Liu and Lane, 2016; Haihong et al., 2019) that use BiL-STM or RNN are now modified to BERT-based implementations explored in more recent works (Chen et al., 2019; Hardalov et al., 2020; Xu et al., 2020). A standard Joint model consists of BERT outputs from the final hidden state (classification (CLS) token for intent and m word tokens for slots) fed to linear layers to get intent and slot predictions.

<table><tr><td>Intent Acc.</td><td>m</td><td>es</td><td>de</td><td>zh</td><td>ja</td><td>pt</td><td>fr</td><td>hi</td><td>tr</td><td>AVG</td></tr><tr><td>English-Only Baseline*</td><td>1</td><td>94.42</td><td>94.29</td><td>79.53</td><td>73.75</td><td>92.90</td><td>93.86</td><td>67.06</td><td>69.71</td><td>83.19</td></tr><tr><td> $Joint_{en-only}$  Baseline*</td><td>1</td><td>95.03</td><td>94.51</td><td>80.54</td><td>73.57</td><td>93.48</td><td>93.33</td><td>73.53</td><td>71.05</td><td>84.38</td></tr><tr><td>Word-level CS</td><td>1</td><td>94.18</td><td>93.92</td><td>81.67</td><td>75.48</td><td>92.54</td><td>94.18</td><td>81.19</td><td>74.22</td><td>85.92</td></tr><tr><td>Sentence-level CS</td><td>1</td><td>94.60</td><td>93.53</td><td>81.21</td><td>75.01</td><td>93.10</td><td>93.24</td><td>82.37</td><td>75.11</td><td>86.02</td></tr><tr><td>Chunk-level CS (CCS)</td><td>1</td><td>95.12</td><td>95.27</td><td>83.88</td><td>74.27</td><td>94.20</td><td>93.48</td><td>82.73</td><td>77.51</td><td>87.06</td></tr><tr><td> $Joint_{en-only}$ * + CCS</td><td>1</td><td>95.48</td><td>94.51</td><td>84.43♠</td><td>76.48♠</td><td>94.15♠</td><td>94.89♠</td><td>85.37♠</td><td>78.04♠</td><td>87.92</td></tr><tr><td colspan="11">Upper Bound</td></tr><tr><td>Translate-Train (TT)*</td><td>8</td><td>94.02</td><td>93.84</td><td>90.21</td><td>84.19</td><td>95.66</td><td>94.54</td><td>85.08</td><td>85.79</td><td>90.42</td></tr><tr><td> $Joint_{TT}$ *</td><td>8</td><td>94.16</td><td>94.24</td><td>91.56</td><td>85.98</td><td>95.75</td><td>95.01</td><td>86.45</td><td>84.95</td><td>91.01</td></tr><tr><td> $Joint_{TT}$ * + CCS</td><td>8</td><td>95.48</td><td>95.41</td><td>91.60</td><td>87.17</td><td>95.34</td><td>94.60</td><td>87.94</td><td>85.93</td><td>91.68</td></tr><tr><td>Slot F1</td><td>m</td><td>es</td><td>de</td><td>zh</td><td>ja</td><td>pt</td><td>fr</td><td>hi</td><td>tr</td><td>AVG</td></tr><tr><td>English-Only Baseline*</td><td>1</td><td>96.16</td><td>96.73</td><td>83.12</td><td>78.81</td><td>95.63</td><td>95.40</td><td>77.05</td><td>88.09</td><td>88.87</td></tr><tr><td> $Joint_{en-only}$  Baseline*</td><td>1</td><td>96.12</td><td>96.76</td><td>84.95</td><td>79.60</td><td>95.76</td><td>95.76</td><td>77.63</td><td>88.92</td><td>89.44</td></tr><tr><td>Word-level CS</td><td>1</td><td>95.81</td><td>96.33</td><td>85.46</td><td>79.33</td><td>96.27</td><td>95.08</td><td>79.10</td><td>86.86</td><td>89.28</td></tr><tr><td>Sentence-level CS</td><td>1</td><td>96.57</td><td>96.92</td><td>86.32</td><td>79.52</td><td>96.65</td><td>95.84</td><td>81.94</td><td>89.84</td><td>90.45</td></tr><tr><td>Chunk-level CS (CCS)</td><td>1</td><td>96.68</td><td>96.82</td><td>87.10</td><td>80.00</td><td>96.46</td><td>96.31</td><td>80.95</td><td>91.60</td><td>90.51</td></tr><tr><td> $Joint_{en-only}$ * + CCS</td><td>1</td><td>96.09</td><td>96.56</td><td>88.61♠</td><td>82.28♠</td><td>96.01</td><td>95.94</td><td>82.28♠</td><td>90.45♠</td><td>91.03</td></tr><tr><td colspan="11">Upper Bound</td></tr><tr><td>Translate-Train (TT)*</td><td>8</td><td>96.89</td><td>96.04</td><td>93.48</td><td>85.29</td><td>96.35</td><td>96.02</td><td>82.03</td><td>91.21</td><td>92.16</td></tr><tr><td> $Joint_{TT}$ *</td><td>8</td><td>96.92</td><td>95.66</td><td>93.64</td><td>87.84</td><td>96.11</td><td>95.95</td><td>82.98</td><td>91.15</td><td>92.53</td></tr><tr><td> $Joint_{TT}$ * + CCS</td><td>8</td><td>96.98</td><td>96.27</td><td>93.37</td><td>85.87</td><td>95.88</td><td>95.44</td><td>82.00</td><td>91.31</td><td>92.14</td></tr></table>

Table 3: Performance evaluation of code-switching with setting $k = 5 .$ CS: Code-Switching. Reported scores are average of 5 independent runs (including a separate code-switched data for each run). m = number of distinc models to be trained. \*: modified BERT-based implementations (Chen et al., 2019; Xu et al., 2020).  
<sup>♠</sup>: The difference is significant with $\mathrm { p < 0 . 0 5 }$ using Tukey HSD (conducted between $J o i n t _ { e n - o n l y } + \mathrm { C C S }$ versus $J o i n t _ { e n - o n l y }$ Baseline for each language).

Assuming $h _ { c l s }$ represents the CLS token and $h _ { m }$ represents a token from the remaining word-level tokens, the BERT model outputs are defined as (Chen et al., 2019; Xu et al., 2020):

$$
p ^ {i} = \text {softmax} \left(W ^ {i} h _ {c l s} + b ^ {i}\right) \tag {1}
$$

$$
p _ {m} ^ {s l} = s o f t m a x (W ^ {s l} h _ {m} + b ^ {s l}) \forall m
$$

with a multi-class cross-entropy loss<sup>3</sup> for both intent $( L _ { i } )$ and slots $( L _ { s l } )$ . We will use this model as our baseline for joint training. Our goal will be to show that code-switching on top of joint training improves the performance. The output of Algorithm 1 will be the input used for joint training on BERT for code-switched experiments.

## 4 Datasets

## 4.1 Benchmark Dataset

We use the latest multilingual benchmark dataset of MultiATIS++ (Xu et al., 2020), which was created by manually translating the original ATIS (Price, 1990) dataset from English (en) to 8 other languages: Spanish (es), Portuguese (pt), German (de), French (fr), Chinese (zh), Japanese (ja), Hindi (hi), and Turkish (tr). The dataset consists of utterances for each language with an ‘intent’ label for ‘flight intent’ and ‘slot’ labels for the word tokens in BIO (Ramshaw and Marcus, 1999) format. A sample datapoint in English is shown in Figure 2.

<table><tr><td>Intent Acc.</td><td>ht</td></tr><tr><td>English-Only Baseline*</td><td>56.12</td></tr><tr><td>Chunk-level CS (CCS)</td><td>63.15</td></tr><tr><td> $Joint_{en-only}^{*} + \text{CCS}$ </td><td>63.73</td></tr><tr><td>Translate-Train (TT)*</td><td>62.58</td></tr><tr><td>Slot F1</td><td>ht</td></tr><tr><td>English-Only Baseline*</td><td>68.72</td></tr><tr><td>Chunk-level CS (CCS)</td><td>70.27</td></tr><tr><td> $Joint_{en-only}^{*} + \text{CCS}$ </td><td>70.02</td></tr><tr><td>Translate-Train (TT)*</td><td>69.96</td></tr></table>

Table 4: Performance on disaster data in Haitian Creole (ht). CS = Code-Switching. Reported scores are average of 5 independent runs (\*: modified BERT-based).

## 4.2 New Dataset for Disaster NLU

We construct a new intent and slot filling dataset of tweets collected during natural disasters, in two languages: English and Haitian Creole. The tweets originally were released by Appen<sup>4</sup>. For English, a language expert coded the tweets, and for Haitian Creole, we used Amazon Mechanical Turk with five annotators. Intent classes include:

$$
^ 3 L = - \frac {1}{n} \sum_ {i = 1} ^ {n} [ y \log \hat {y} ]
$$

<table><tr><td>CS (k=5)</td><td>MTT</td><td>Jointen</td><td>Jointcs</td><td>JointTT</td></tr><tr><td>05:04:49</td><td>1:31:32</td><td>00:11:50</td><td>01:06:50</td><td>00:11:04</td></tr></table>

Table 5: Runtime on Google Colab (K80 GPU for training joint models). MTT: Machine Translation to Target. Note that MT T and $J _ { T T }$ are for one target language (averaged).

‘request’ and ‘others’. Slot filling consists of 5 labels: ‘medical\_help’, ‘food’, ‘water’, ‘shelter’, and ‘other\_aid’. Table 2 provides the dataset statistics.

## 5 Experimental Setup

We use the traditional cross-lingual task setting where each experiment consists of a source language and a target language. A model is trained on the source data (English) and evaluated on the target data (8 other languages). For codeswitching experiments, an English text is augmented with multilingual code-switching before training. Our implementation is in PyTorch (Paszke et al., 2019) and we use the pre-trained bert-basemultilingual-uncased (Devlin et al., 2019) with BertForSequenceClassification (Wolf et al., 2020) as the mBERT model. Maximum epoch is set to 25 with an early stopping patience of 5, batch size of 32, and Adam optimizer (Kingma and Ba, 2014) with a learning rate of 5e−5. We select the best model on the validation set. Consistent with the metrics reported for intent prediction and slot filling evaluation in the past, we also use accuracy for intent and micro F1 to measure slot performance.

## 5.1 Baselines & Upper Bound

Since we assume that target language is not known before hand, Translate-Train (TT) (Xu et al., 2020) method is not a suitable baseline. Rather, we set this to be an upper bound, i.e. translating to the target language and fine-tuning the model should intuitively outperform a generic model. Additionally, we add code-switching to this TT model to assess if augmentation negatively impacts its performance. The zero-shot baselines for the codeswitching experiments use an English-Only (Xu et al., 2020) model, which is fine-tuned over the pre-trained mBERT separately for each task and an English-only Joint model (Chen et al., 2019).

## 6 Results & Discussion

## 6.1 Effect of Multilingual Code-Switching

Table 3 describes performance evaluation on the MultiATIS++ dataset. When compared to the stateof-the-art jointly trained English-only baseline, we see a +4.2% boost in intent accuracy and +1.8% boost in slot F1 scores on average by augmenting the dataset via multilingual code-switching without requiring the target language. From the significance tests, except for Spanish and German, all other languages were helped by code-switching for intent detection. For slot filling, improvement on Portuguese and French went insignificant. This suggests that code-switching primarily helped languages that are morphologically more different as compared to the source language (English). For example, Hindi and Turkish have the highest intent performance improvement of +16.1% and +9.8% respectively. And for slots, Hindi and Chinese with +6.0% and +4.3% respectively. Japanese showed +4% improvement for intent and +3.4% for slots.

The running time of the models in Table 5 show that code-switching is expensive which can take up to 5 hours for k = 5. Its training is also expensive because there is k times more data as compared to the monolingual source data. Increasing the number of code-switchings (k) for a sentence from 5 to 50 improved the performance by +1%, while increasing the run-time by a large margin. So, parameter k should be picked appropriately. Albeit this time cost is for training, with benefits at the prediction stage for real world problems.

In the translate-train (upper bound) scenario, it is not immediately clear if augmentation can help, because data in the same language as the target is always preferred over other languages, or codeswitched. However, we show in Table 3 that augmentation did not hinder the performance.

For both intent and slot performance, chunklevel model remained robust across the languages. For intent, difference between word-level and sentence-level was insignificant. For slot, sentencelevel was in par with chunk-level on average. Thus, we think that code-switching at chunk-level is safer for avoiding semantic discrepancies (as in the wordlevel) while also capturing better intra-sentence language neutrality.

## 6.2 Evaluation on Disaster Dataset

We found that disaster data is more challenging when compared to the ATIS dataset for transfer learning in NLU. The predictive performance is shown in Table 4. Code-Switching improved intent accuracy by +12.5% and slot F1 by +2.3%, which is promising considering that they are tweets. Joint training added +0.9% improvement to intent accuracy, however did not seem to help slot F1. This might imply a lack of strong correlation between the two tasks, i.e. a mention of ‘food’ or ‘shelter in a tweet may not always mean that it is a ‘request or vice-versa. The upper bound of translate-train method did not perform any better than the randomly code-switched model which seemed counterintuitive. This might be due to the lack of strong representation for Haitian Creole in the pre-trained model, although it is similar to French.

![](images/3419cbf7e1a49701e8b7ec9c2997aa7b8b219ae2ad5364298fe3014c1c8c9df9.jpg)

<details>
<summary>bar</summary>

| Target Language | Translate Train (Upper Bound) | English-Only | Romance | Germanic | Sino-Tibetan & Japonic | Indo-Aryan | Turkic | Afro-Asiatic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spanish | ~94.2 | ~94.5 | ~95.8 | ~95.0 | ~94.9 | ~94.3 | ~94.9 | ~94.2 |
| German | ~93.9 | ~94.4 | ~94.2 | ~95.3 | ~94.5 | ~93.0 | ~92.6 | ~92.7 |
| Chinese | ~90.3 | ~79.6 | ~79.4 | ~82.1 | ~80.9 | ~80.4 | ~82.7 | ~79.6 |
| Japanese | ~84.3 | ~73.8 | ~72.6 | ~74.4 | ~76.8 | ~73.0 | ~77.2 | ~72.2 |
| Hindi | ~85.1 | ~67.1 | ~75.1 | ~65.0 | ~82.5 | ~84.6 | ~69.8 | ~79.4 |
| Turkish | ~85.9 | ~69.8 | ~73.9 | ~72.8 | ~74.7 | ~75.7 | ~78.6 | ~73.1 |
</details>

Figure 3: Impact of different language groups on the target languages.

![](images/0ed9859007eead347f9809032fac46a0a154e5d86a7ebe63ad497e9d357052c1.jpg)

<details>
<summary>bar</summary>

| Category | hi | tr | zh | ja | es | de |
| --- | --- | --- | --- | --- | --- | --- |
| capacity | ~0.25 | ~0.50 | ~0.20 | ~0.40 | ~0.25 | ~-0.01 |
| distance | ~0.26 | ~0.22 | ~0.43 | ~0.49 | ~0.01 | ~-0.03 |
| airfare | ~0.48 | ~0.22 | ~0.44 | ~0.06 | ~-0.03 | 0 |
| airline | ~0.49 | ~0.10 | ~0.21 | ~0.20 | 0 | ~0.01 |
| ground_service | ~0.17 | ~0.05 | ~0.31 | ~0.23 | ~0.06 | ~0.01 |
| flight | ~0.04 | ~0.01 | ~0.02 | ~0.03 | ~0.01 | 0 |
| abbreviation | ~0.02 | ~0.18 | ~-0.01 | ~-0.15 | 0 | ~0.05 |
| airport | ~-0.04 | 0 | ~0.03 | ~-0.07 | 0 | ~0.06 |
</details>

Figure 4: Impact of code-switching on intent classes.

![](images/03862e0fe4104dc75ac0f6ed2c6f2d736652adf41e020e4b4f857efc9cbc636d.jpg)

<details>
<summary>bar</summary>

| Category | hi | zh | ja | tr | es | de |
| --- | --- | --- | --- | --- | --- | --- |
| fl.apc | ~0.21 | ~0.68 | ~0.49 | ~0.57 | ~0.67 | ~0.66 |
| dd.dn | ~0.59 | ~0.27 | ~0.52 | ~0.23 | ~0.55 | ~0.33 |
| cn | ~0.12 | ~0.32 | ~0.09 | ~0.20 | ~0.55 | ~0.57 |
| fn | ~0.30 | ~-0.06 | ~0.06 | ~0.09 | ~0.59 | ~0.68 |
| dt.t | ~0.02 | ~0.58 | ~0.20 | ~0.15 | ~0.26 | ~0.31 |
| acc | ~0.24 | ~0.24 | ~-0.11 | ~-0.18 | ~0.59 | ~0.63 |
| an | ~0.14 | ~0.18 | ~0.08 | ~0.11 | ~0.22 | ~0.11 |
| dt.pd | ~0.04 | ~0.34 | ~-0.04 | ~0.09 | ~0.10 | ~-0.01 |
</details>

Figure 5: Impact of code-switching on slot labels.

## 6.3 Impact of Language Families

Results of language family analysis are shown in Figure 3. The input in English is independently code-switched using 6 different language families. Note that the target language is always excluded from the group when evaluating on the same, i.e. Hindi is excluded from Indo-Aryan family when that family is being evaluated on it. Translate-train model is provided as a frame of reference and upper bound. We dropped French and Portuguese from the chart as they fall in to Romance family similar to Spanish. Results show the language families helped their corresponding languages, i.e. Romance helped Spanish, Germanic helped German, and so on; with the exception of Chinese and Japanese. In both cases, Turkic language family helped better than others.

![](images/476718ef4b59c4346dd77b8a29f6e88166ab2366d28d088b9f34d5180d348d6c.jpg)

<details>
<summary>line</summary>

| Layers Frozen | AVG | es | ja | hi |
| --- | --- | --- | --- | --- |
| 0 | ~84 | ~92 | ~72 | ~79.5 |
| 2 | ~87 | ~95 | ~76 | ~84.5 |
| 4 | ~87 | ~95.5 | ~75.5 | ~84.5 |
| 6 | ~87.5 | ~95 | ~78 | ~85 |
| 8 | ~88 | ~95 | ~78.5 | ~86 |
| 10 | ~86 | ~94 | ~74.5 | ~83 |
| 12 | ~75 | ~76 | ~74.5 | ~74.5 |
</details>

Figure 6: Freezing earlier layers and unfreezing a few at the top of the transformer appear to be most optimal.

## 6.4 Error Analysis

Selecting intent classes with support > 10, Figure 4 shows how each class is positively or negatively impacted by code-switching. Improvement was primarily on ‘airfare’, ‘distance’ ‘capacity’, ‘airline’, and ‘ground\_service’ which had longer sentences such as ‘Please tell me which airline has the most departuresfrom Atlanta’ when compared to ‘abbreviations’ and ‘airport’ classes that included very short phrases like ‘What does EA mean?’ However, note that, Spanish and German did not improve much; aligning with our results in Table 3. For slot labels in Figure 5, we selected the ones with support > 50 and that have different characteristics, e.g. ‘name’, ‘code’, etc. The overall trend in slot performance shows improvements for labels such as ‘day\_name’, ‘airport\_code’, and ‘city\_name’ and slight variations in labels such as ‘fight\_number’ and ‘period\_of\_day’; implying textual slots benefiting over numeric ones.

## 6.5 Hyperparameter Tuning

For joint training with same task weights, we tuned α and $\beta$ using grid search to see the strength of correlation between the tasks. For intent, the $( \alpha , \beta )$ combination of (1.0, 0.6) performed well, while (1.0, 1.0) for slots. This suggests that intent benefiting slot might be slightly more than slot benefiting intent. Additionally, during fine-tuning, freezing the layers of the transformer affected the model performance as shown in Figure 6. Keeping the first 8 layers frozen gave the best performance. By freezing the earlier layers, the transformer can retain its most fundamental feature information gained from the massive pre-training step, and by unfreezing some top layers, it can undergo fine-tuning.

## 7 Conclusion & Future Work

This study shows that augmenting the monolingual input data with multilingual code-switching via random translations helps a zero-shot model to be more language neutral when evaluated on unseen languages. This approach enhanced the generalizability of pre-trained mBERT when finetuning for downstream tasks of intent detection and slot filling. We presented an application of this method using a new annotated dataset of disaster tweets. Further, we studied code-switching with language families and their impact on specific target languages, which can be used to enhance the zero-shot generalizability of models created for low-resource languages. Expanding to XLM-R and similar approaches to improve masked language model training by addressing code-switching during pre-training and releasing a larger dataset of annotated disaster tweets in more languages are planned for future work.

## 8 Acknowledgement

We thank U.S. National Science Foundation grants IIS-1815459 and IIS-1657379 for partially supporting this research. We also thank Ming Sun and Alexis Conneau for giving valuable insights on multilingual model training. We also acknowledge ARGO team as the experiments were run on ARGO, a research computing cluster provided by the Office of Research Computing at George Mason University.

## References

Gustavo Aguilar and Thamar Solorio. 2019. From english to code-switching: Transfer learning with strong morphological clues. arXiv preprint arXiv:1909.05158.  
Kelsey Ball and Dan Garrette. 2018. Part-of-speech tagging for code-switched, transliterated texts without explicit language identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3084–3089.  
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. Transactions of the Associationfor Computational Linguistics, 5:135–146.  
Aditi Chaudhary, Karthik Raman, Krishna Srinivasan, and Jiecao Chen. 2020. Dict-mlm: Improved multilingual pre-training using bilingual dictionaries. arXiv preprint arXiv:2010.12566.  
Qian Chen, Zhu Zhuo, and Wen Wang. 2019. Bert for joint intent classification and slot filling. arXiv preprint arXiv:1902.10909.  
Xilun Chen, Yu Sun, Ben Athiwaratkun, Claire Cardie, and Kilian Weinberger. 2018. Adversarial deep averaging networks for cross-lingual sentiment classification. Transactions ofthe Associationfor Computational Linguistics, 6:557–570.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020a. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116.  
Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2017. Word translation without parallel data. arXiv preprint arXiv:1710.04087.  
Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. arXiv preprint arXiv:1809.05053.  
Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020b. Emerging cross-lingual structure in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6022–6034, Online. Association for Computational Linguistics.  
Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, et al. 2018. Snips voice plat form: an embedded spoken language understanding system for private-by-design voice interfaces. arXiv preprint arXiv:1805.10190.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Tech nologies, Volume 1 (Long and Short Papers), pages 4171–4186.  
Philipp Dufter and Hinrich Schütze. 2020. Identifying elements essential for bert’s multilinguality. In Proceedings of the 2020 Conference on Empirical  
Methods in Natural Language Processing (EMNLP), pages 4423–4437.  
Chris Dyer, Victor Chahuneau, and Noah A Smith. 2013. A simple, fast, and effective reparameterization of ibm model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648.  
William Denis Elcock and John N Green. 1960. The romance languages. Faber & Faber London.  
Akiko Eriguchi, Melvin Johnson, Orhan Firat, Hideto Kazawa, and Wolfgang Macherey. 2018. Zeroshot cross-lingual classification using multilingual neural machine translation. arXiv preprint arXiv:1809.04686.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. The Journal of Machine Learning Research, 17(1):2096–2030.  
Chih-Wen Goo, Guang Gao, Yun-Kai Hsu, Chih-Li Huo, Tsung-Chieh Chen, Keng-Wei Hsu, and Yun-Nung Chen. 2018. Slot-gated modeling for joint slot filling and intent prediction. In Proceedings of the 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 753–757.  
E Haihong, Peiqing Niu, Zhongfu Chen, and Meina Song. 2019. A novel bi-directional interrelated model for joint intent detection and slot filling. In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 5467– 5471.  
Wayne Harbert. 2006. The Germanic Languages. Cambridge University Press.  
Momchil Hardalov, Ivan Koychev, and Preslav Nakov. 2020. Enriched pre-trained transformers for joint slot filling and intent detection. arXiv preprint arXiv:2004.14848.  
Xiaodong He, Li Deng, Dilek Hakkani-Tur, and Gokhan Tur. 2013. Multi-style adaptive training for robust cross-lingual spoken language understanding. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 8342– 8346. IEEE.  
Zhengbao Jiang, Antonios Anastasopoulos, Jun Araki, Haibo Ding, and Graham Neubig. 2020. Multilingual factual knowledge retrieval from pretrained language models. arXiv preprint arXiv:2010.06189.  
Lars Johanson and Éva Ágnes Csató Johanson. 2015. The Turkic Languages. Routledge.  
Aditya Joshi, Ameya Prabhu, Manish Shrivastava, and Vasudeva Varma. 2016. Towards sub-word level compositions for sentiment analysis of hindi-english code mixed text. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 2482–2491.  
Ashiqur R KhudaBukhsh, Shriphani Palakodety, and Jaime G Carbonell. 2020. Harnessing code switching to transcend the linguistic barrier. arXiv preprint arXiv:2001.11258.  
Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.  
Jitin Krishnan, Hemant Purohit, and Huzefa Rangwala. 2020. Unsupervised and interpretable domain adaptation to rapidly filter social web data for emergency services. In ASONAM.  
Bing Liu and Ian Lane. 2016. Attention-based recurrent neural network models for joint intent detection and slot filling. arXiv preprint arXiv:1609.01454.  
Manuel Mager, Özlem Çetinoglu, and Katharina Kann.˘ 2019. Subword-level language identification for intra-word code-switching. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2005–2011.  
Colin P Masica. 1993. The indo-aryan languages. Cambridge University Press.  
Deepthi Mave, Suraj Maharjan, and Thamar Solorio. 2018. Language identification and analysis of codeswitched social media text. In Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching, pages 51–61.  
Roy Andrew Miller. 1967. The Japanese Language. University of Chicago Press Chicago.  
Aldrian Obaja Muis, Naoki Otani, Nidhi Vyas, Ruochen Xu, Yiming Yang, Teruko Mitamura, and Eduard Hovy. 2018. Low-resource cross-lingual event type detection via distant supervision with minimal effort. In Proceedings of the 27th Inter national Conference on Computational Linguistics, pages 70–82.  
Dat Tien Nguyen, Kamla Al-Mannai, Shafiq R Joty, Hassan Sajjad, Muhammad Imran, and Prasenjit Mi tra. 2017. Robust classification of crisis-related data on social networks using convolutional neural networks. ICWSM, 31(3):632–635.  
Jian Ni, Georgiana Dinu, and Radu Florian. 2017. Weakly supervised cross-lingual named entity recognition via effective annotation and representation projection. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1470–1480.  
Endang Wahyu Pamungkas and Viviana Patti. 2019. Cross-domain and cross-lingual abusive language detection: A hybrid approach with deep learning and a multilingual lexicon. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 363–370.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc.  
Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual bert? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001.  
Barbara Plank and Željko Agic. 2018. Distant super-´ vision from disparate sources for low-resource partof-speech tagging. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 614–620.  
Patti Price. 1990. Evaluation of spoken language systems: The atis domain. In Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27, 1990.  
Libo Qin, Minheng Ni, Yue Zhang, and Wanxiang Che. 2020. Cosda-ml: Multi-lingual code-switching data augmentation for zero-shot cross-lingual nlp. arXiv preprint arXiv:2006.06402.  
Lance A Ramshaw and Mitchell P Marcus. 1999. Text chunking using transformation-based learning. In Natural language processing using very large corpora, pages 157–176. Springer.  
Bernardino Romera-Paredes and Philip Torr. 2015. An embarrassingly simple approach to zero-shot learning. In International Conference on Machine Learning, pages 2152–2161.  
Bruce Rowe and Diane Levine. 2017. A concise introduction to linguistics. Routledge. pp. 340–341.  
Sebastian Schuster, Sonal Gupta, Rushin Shah, and Mike Lewis. 2019. Cross-lingual transfer learning for multilingual task oriented dialog. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3795–3805.  
Robert Shafer. 1955. Classification of the sino-tibetan languages. Word, 11(1):94–111.  
Rushin Shah, Bo Lin, Anatole Gershman, and Robert Frederking. 2010. Synergy: a named entity recognition system for resource-scarce languages such as swahili using online machine translation. In Proceedings of the Second Workshop on African Language Technology (AfLaT 2010), pages 21–26.  
Victor Soto and Julia Hirschberg. 2018. Joint part-ofspeech and language id tagging for code-switched data. In Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching, pages 1–10.  
Shashank Srivastava, Igor Labutov, and Tom Mitchell. 2018. Zero-shot learning of classifiers from natural language quantification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 306–316.  
Lukas Stappen, Fabian Brunn, and Björn Schuller. 2020. Cross-lingual zero-and few-shot hate speech detection utilising frozen transformer language models and axel. arXiv preprint arXiv:2004.13850.  
Oscar Täckström, Dipanjan Das, Slav Petrov, Ryan Mc-Donald, and Joakim Nivre. 2013. Token and type constraints for cross-lingual part-of-speech tagging. Transactions of the Association for Computational Linguistics, 1:1–12.  
Chen-Tse Tsai, Stephen Mayhew, and Dan Roth. 2016. Cross-lingual named entity recognition via wikification. In Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning, pages 219–228.  
Shyam Upadhyay, Manaal Faruqui, Gokhan Tür, Hakkani-Tür Dilek, and Larry Heck. 2018. (almost) zero-shot cross-lingual spoken language understand ing. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6034–6038. IEEE.  
Charles Frederick Voegelin and Florence Marie Voegelin. 1976. Classification and index of the world’s languages.  
Xiaojun Wan. 2009. Co-training for cross-lingual sentiment classification. In Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natu ral Language Processing ofthe AFNLP, pages 235– 243.  
Thomas Wolf, Julien Chaumond, Lysandre Debut, Victor Sanh, Clement Delangue, Anthony Moi, Pierric Cistac, Morgan Funtowicz, Joe Davison, Sam Shleifer, et al. 2020. Transformers: State-of-theart natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.  
Yongqin Xian, Bernt Schiele, and Zeynep Akata. 2017. Zero-shot learning-the good, the bad and the ugly. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4582–4591.  
Min Xiao and Yuhong Guo. 2013. Semi-supervised representation learning for cross-lingual text classification. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1465–1475.  
Jiateng Xie, Zhilin Yang, Graham Neubig, Noah A Smith, and Jaime G Carbonell. 2018. Neural crosslingual named entity recognition with minimal resources. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 369–379.  
Weijia Xu, Batool Haider, and Saab Mansour. 2020. End-to-end slot alignment and recognition for crosslingual nlu. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5052–5063.  
Jian Yang, Shuming Ma, Dongdong Zhang, ShuangZhi Wu, Zhoujun Li, and Ming Zhou. 2020. Alternating language modeling for cross-lingual pre-training. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 9386–9393.  
David Yarowsky, Grace Ngai, and Richard Wicentowski. 2001. Inducing multilingual text analysis tools via robust projection across aligned corpora. Technical report, Johns Hopkins Univ Baltimore MD Dept of Computer Science.  
Zeynep Yirmibe¸soglu and Gül¸sen Eryi˘ git. 2018. De-˘ tecting code-switching between turkish-english language pair. In Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy Usergenerated Text, pages 110–115.  
Katherine Yu, Haoran Li, and Barlas Oguz. 2018. Multilingual seq2seq training with similarity loss for cross-lingual document classification. In Proceedings of The Third Workshop on Representation Learningfor NLP, pages 175–179.  
Ayah Zirikly and Masato Hagiwara. 2015. Crosslingual transfer of named entity recognizers without parallel corpora. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 390–396.