# Learning Multilingual Meta-Embeddings for Code-Switching Named Entity Recognition

Genta Indra Winata, Zhaojiang Lin, Pascale Fung

Center for Artificial Intelligence Research (CAiRE)

Department of Electronic and Computer Engineering

The Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong

{giwinata,zlinao}@connect.ust.hk, pascale@ece.ust.hk

# Abstract

In this paper, we propose Multilingual Meta-Embeddings (MME), an effective method to learn multilingual representations by leveraging monolingual pre-trained embeddings. MME learns to utilize information from these embeddings via a self-attention mechanism without explicit language identification. We evaluate the proposed embedding method on the code-switching English-Spanish Named Entity Recognition dataset in a multilingual and cross-lingual setting. The experimental results show that our proposed method achieves state-of-the-art performance on the multilingual setting, and it has the ability to generalize to an unseen language task.

# 1 Introduction

Learning a representation through embedding is a fundamental technique to capture latent word semantics [\(Clark,](#page-4-0) [2015\)](#page-4-0). Practically, word-level representation has been extensively explored to improve many downstream natural language processing (NLP) tasks [\(Mikolov et al.,](#page-5-0) [2013;](#page-5-0) [Pennington](#page-5-1) [et al.,](#page-5-1) [2014;](#page-5-1) [Grave et al.,](#page-4-1) [2018\)](#page-4-1). A new wave of "meta-embeddings" research aims to learn how to effectively combine pre-trained word embeddings in supervised training into a single dense representation [\(Yin and Schütze,](#page-5-2) [2016;](#page-5-2) [Muromägi et al.,](#page-5-3) [2017;](#page-5-3) [Bollegala et al.,](#page-4-2) [2018;](#page-4-2) [Coates and Bollegala,](#page-4-3) [2018;](#page-4-3) [Kiela et al.,](#page-4-4) [2018\)](#page-4-4). This method is known to be effective to overcome domain and modality limitations. However, the generalization ability of previous works has been limited to monolingual tasks, so we aim to extend the method to multilingual contexts which benefits the processing of code-switching text.

In multilingual societies, speakers tend to move back and forth from one language to another during the same conversation, which is commonly

![](_page_0_Diagram_7.jpeg)

Figure 1: Multilingual Meta-Embeddings. The inputs are word embeddings and the output is a single word representation.

called "*code-switching*". Code-Switching is produced in both written text and speech in a discourse. Recent studies in code-switching has been mainly focused on natural language tasks, such as language modeling [\(Winata et al.,](#page-5-4) [2018a;](#page-5-4) [Prat](#page-5-5)[apa et al.,](#page-5-5) [2018;](#page-5-5) [Garg et al.,](#page-4-5) [2018\)](#page-4-5), named entity recognition [\(Aguilar et al.,](#page-4-6) [2018\)](#page-4-6), and language identification [\(Solorio et al.,](#page-5-6) [2014;](#page-5-6) [Molina et al.,](#page-5-7) [2016;](#page-5-7) [Barman et al.,](#page-4-7) [2014\)](#page-4-7). Code-Switching is considered as a challenging task because words from different languages may co-exist within a sequence, and models are required to recognize the context of mixed-language sentences. Meanwhile, some words with the same spelling may have entirely different meanings (e.g., cola in English and Spanish) [\(Winata et al.,](#page-5-8) [2018b\)](#page-5-8). Language identifiers were commonly used to solve the word ambiguity issue in mixed-language sentences. However, it may not reliably cover all code-switching cases, and it creates a bottleneck that would require large-scale crowdsourcing to annotate language identifiers in code-switching data correctly.

To overcome the code-switching problem, we introduce a multilingual meta-embedding model learned from different languages. Our approach can be seen as a method to create a universal multilingual meta-embedding learned in a supervised way with code-switching contexts by gathering information from monolingual sources. Concurrently, this is a language-agnostic approach where it does not require any language information of each word. We show the possibility of transferring information from multiple languages to unseen languages, and this approach can also be useful for a low-resource setting. To effectively leverage the embeddings, we use FastText subwords information to solve out-of-vocabulary (OOV) issues. By applying this method, our model can align the words with the corresponding languages. Our contributions are two-fold:

- We propose to generate multilingual metarepresentations from pre-trained monolingual word embeddings. The model can learn how to construct the best word representation by mixing multiple sources without explicit language identification.
- We evaluate our multilingual metaembedding on English-Spanish codeswitching Named Entity Recognition (NER). The result shows the effectiveness of the method on multilingual setting and demonstrates that our meta-embedding can generalize to unseen languages in a cross-lingual setting.

# 2 Meta-Embeddings

Word embedding pre-training is a well-known method to transfer the knowledge from previous tasks to a target task that has fewer high-quality training data. Word embeddings are commonly used as features in supervised learning problems. We propose to generate a single word representation by extracting information from different pretrained embeddings. We extend the idea of metaembeddings from [Kiela et al.](#page-4-4) [\(2018\)](#page-4-4) to solve a multilingual task. We define a sentence that consists of m words {xj} m <sup>j</sup>=1, and {wi,j} n <sup>j</sup>=1 word vectors from n pre-trained word embeddings.

#### 2.1 Baselines

We compare our method to two baselines: (1) concatenation and (2) linear ensembles.

Concatenation We concatenate word embeddings by merging the dimensions of word representations. This is the simplest way to utilize all sources of information; however, it is very inefficient due to the high-dimensional input:

wCONCAT <sup>i</sup> = [wi,1, ..., wi,n]. (1)

Linear Ensembles We sum all word embeddings into a single word vector with an equal weight. This method is efficient since it does not increase the dimensionality of the input. We apply a projection layer through wi,j to have equal dimension before we sum:

wLINEAR <sup>i</sup> = Xn j=0 w0 i,j , (2)

w0 i,j = a<sup>j</sup> · wi,j + b<sup>j</sup> , (3)

where a<sup>j</sup> ∈ <sup>R</sup> l×d and b<sup>j</sup> ∈ <sup>R</sup> d are trainable parameters, and l and d are the original dimensions of the pre-trained embeddings and projected dimensions respectively.

## 2.2 Multilingual Meta-Embedding

We generate a multilingual vector representation for each word by taking a weighted sum of monolingual embeddings. Each embedding wi,j is projected with a fully connected layer with a nonlinear scoring function φ (e.g., tanh) into a ddimensional vector, and an attention mechanism to calculate attention weight αi,j ∈ <sup>R</sup> d :

wMME <sup>i</sup> = Xn j=1 αi,jw<sup>0</sup> i,j , (4)

αi,j = e φ(w 0 i,j ) P<sup>n</sup> <sup>j</sup>=1 e φ(w 0 i,j )) . (5)

## 3 Named Entity Recognition

Our proposed model is based on a self-attention mechanism from a transformer encoder [\(Vaswani](#page-5-9) [et al.,](#page-5-9) [2017\)](#page-5-9) followed by a Conditional Random Field (CRF) layer [\(Lafferty et al.,](#page-4-8) [2001\)](#page-4-8).

Encoder Architecture We apply a multi-layer transformer encoder as our sentence encoder:

h<sup>0</sup> = Concat(w0, w1, . . . , wm)W<sup>t</sup> + Wp, (6)

h<sup>l</sup> = Transformer\_blocks(h0), (7)

o = hlW<sup>o</sup> + bo, (8)

where W<sup>t</sup> is the projection matrix, W<sup>p</sup> is the positional encoding matrix, W<sup>o</sup> is the output layer, h<sup>0</sup> is the first layer hidden states, and h<sup>l</sup> is the output representation from the final transformer layer. The output of the final layer is logits o.

Conditional Random Field This model calculates the dependencies across tag labels. NER requires a stronger constraint where I-PERSON should follow only after B-PERSON. We use CRF to learn the correlations between the current label and its neighbors [\(Lafferty et al.,](#page-4-8) [2001\)](#page-4-8). We consider A ∈ R (k+2)×(k+2) as a trainable matrix, transition scores of the tags, where k is the number of tags. Ai,j denotes the transition score from tag i to tag j. We include a start tag and an end tag in the matrix, and calculate the score of a tag sequence y given o as follows:

<sup>s</sup>(o, <sup>y</sup>) = <sup>X</sup><sup>n</sup> i=0 Ayi,yi+1 + Xn i=0 Pi,y<sup>i</sup> , (9)

where Pi,y<sup>i</sup> ∈ <sup>R</sup> n×k represents the output probability of the tags. We use the Viterbi algorithm to select the best sequence.

# 4 Experiments

### 4.1 Dataset

For our experiment, we use English-Spanish tweets data provided by [Aguilar et al.](#page-4-6) [\(2018\)](#page-4-6). There are nine entity labels. The labels use IOB format, where every token is labeled as a B-label in the beginning and then an I-label if it is a named entity, or O otherwise.

#### 4.2 Experimental Setup

We use pre-trained FastText [<sup>1</sup>](#page-2-0) English *(EN)* and Spanish *(ES)* word embeddings [\(Grave et al.,](#page-4-1) [2018\)](#page-4-1) as our primary language embeddings, and pre-trained FastText Catalan *(CA)* and Portuguese *(PT)* word embeddings as our auxiliary language embeddings. We opt for *CA* and *PT* because they come from the same Romance language family as Spanish. We also include GloVe Twitter English embedding *(GLOVE\_EN)* [\(Pennington et al.,](#page-5-1) [2014\)](#page-5-1).[<sup>2</sup>](#page-2-1) Experiments are conducted in two different settings. In the multilingual setting, we learn our meta-embedding from primary languages and auxiliary languages, while in the cross-lingual setting only auxiliary languages are used. We run all experiments five times and calculate the average and standard deviation. To improve our final predictions, we ensemble all five experiments and take the results from a majority consensus.

<span id="page-2-2"></span>

| Approaches |      |        |         |               | F1           |
|------------|------|--------|---------|---------------|--------------|
| Trivedi    |      | et al. | (2018)  | (Single)      | 61.89        |
| Wang       | et   | al.    | (2018)  | (Single)      | 62.39        |
| Wang       | et   | al.    | (2018)  | (Ensemble)    | 62.67        |
| Winata     |      | et al. | (2018b) | (Single)      | 62.76        |
| Trivedi    |      | et al. | (2018)  | (Ensemble)    | 63.76        |
| EN         |      |        |         |               | 62.75 ± 0.66 |
| ES         |      |        |         |               | 62.91 ± 1.07 |
| EN         | + ES |        |         |               | 65.30 ± 0.38 |
| EN         | + ES | +      | CA      |               | 65.36 ± 0.85 |
| EN         | + ES | +      | PT      |               | 65.53 ± 0.79 |
| EN         | + ES | +      | CA +    | PT            | 64.99 ± 1.06 |
| EN         | + ES | +      | CA +    | PT (Single)   | 65.33 ± 0.87 |
| EN         | + ES | +      | CA +    | PT (Ensemble) | 67.03        |
| EN         | + ES |        |         |               | 65.43 ± 0.67 |
| EN         | + ES | +      | CA      |               | 65.69 ± 0.83 |
| EN         | + ES | +      | PT      |               | 65.65 ± 0.48 |
| EN         | + ES | +      | CA +    | PT (Single)   | 66.63 ± 0.94 |
| EN         | + ES | +      | CA +    | PT (Ensemble) | 68.34        |

Table 1: Multilingual results (mean and standard deviation from five experiments). *EN*: both English FastText and GloVe word embeddings.

Implementation Details Our model is trained using a Noam optimizer with a dropout of 0.1 for multilingual setting and 0.3 for the crosslingual setting. Our model contains four layers of transformer blocks with a hidden size of 200 and four heads. We start the training with a learning rate of 0.1. We replace user hashtags (#user) and mentions (@user) with <USR>, and URL (https://domain.com) with <URL>, similarly to [Winata et al.](#page-5-8) [\(2018b\)](#page-5-8).

### 5 Results

Multilingual experimental results are shown in Table [1.](#page-2-2) Interestingly, both concatenation and linear ensemble are strong baselines since they can achieve higher performance compared to any existing works that use more complicated features, such as character-based features using a bidirectional long short-term memory (LSTM) [\(Winata](#page-5-8) [et al.,](#page-5-8) [2018b;](#page-5-8) [Wang et al.,](#page-5-11) [2018\)](#page-5-11) or a convolutional neural network (CNN) with additional gazetteers [\(Trivedi et al.,](#page-5-10) [2018\)](#page-5-10). Overall, our transformer encoder using a single word embedding achieves better performance compared to the LSTM encoder

<span id="page-2-0"></span><sup>1</sup> https://fasttext.cc/docs/en/crawl-vectors.html

<span id="page-2-1"></span>https://nlp.stanford.edu/projects/glove/

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 2: An example of attention weights on a development sample evaluated from a multilingual model (top) and a cross-lingual model (bottom). Darker color shows higher attention scores.

<span id="page-3-0"></span>

| Approaches         | F1           |
|--------------------|--------------|
| CA                 | 53.96 ± 1.42 |
| PT                 | 54.86 ± 4.10 |
| CA + PT            | 58.28 ± 2.66 |
| CA + PT (Single)   | 60.72 ± 0.84 |
| CA + PT (Ensemble) | 62.9         |
| CA + PT (Single)   | 61.75 ± 0.56 |
| CA + PT (Ensemble) | 63.66        |

Table 2: Cross-lingual results (mean and standard deviation from five experiments).

structure used by [Winata et al.](#page-5-8) [\(2018b\)](#page-5-8); [Trivedi](#page-5-10) [et al.](#page-5-10) [\(2018\)](#page-5-10); [Wang et al.](#page-5-11) [\(2018\)](#page-5-11). More importantly, MME outperforms the two baselines on different language combinations, which shows its effectiveness. The results also show that the two baselines cannot effectively exploit the information from auxiliary languages. Here we note that the main advantage of MME is that it dynamically weights the different language pre-trained embeddings for each input token, while the concatenation and linear ensemble approaches always score the weights equally.

In the cross-lingual setting, our model does not perform well when we only use one auxiliary language, as seen in Table [2.](#page-3-0) A significant improvement is shown after we combine both languages, and MME shows a similar performance to the previous state-of-the-art result [\(Trivedi et al.,](#page-5-10) [2018\)](#page-5-10). This implies that our approach can effectively generalize word representations on an unseen language task by transferring information from languages that come from the same root as the primary languages.

We inspect the assigned weights on word embeddings to see which embedding our model attends. Figure [2](#page-3-1) visualizes the weights for the multilingual and cross-lingual cases. It appears that our model can align words to their languages (e.g., Spanish words, such as "*ti*", "*te*", and "*ponen*" attend to *ES*) with strong confidences. In most cases, our model strongly attends to a single language and takes a small proportion of information from other languages. It shows the potential to automatically learn how to construct a multilingual embedding from semantically similar embeddings without requiring any language labels.

# 6 Related Work

Early studies on named entity recognition heavily relied on language-specific knowledge resources, such as hand-crafted features or gazetteers [\(Laf](#page-4-8)[ferty et al.,](#page-4-8) [2001;](#page-4-8) [Ratinov and Roth,](#page-5-12) [2009;](#page-5-12) [Tsai](#page-5-13) [et al.,](#page-5-13) [2016\)](#page-5-13). However, this approach was costly for new languages and domains. Thus, end-toend approaches that do not rely on any external knowledge were proposed. [Sobhana et al.](#page-5-14) [\(2010\)](#page-5-14) proposed to use a CRF without any external resources, to leverage the label dependencies. Then, neural-based approaches, such as LSTM with a CRF [\(Lample et al.,](#page-4-9) [2016;](#page-4-9) [Lin et al.,](#page-4-10) [2017;](#page-4-10) [Green](#page-4-11)[berg et al.,](#page-4-11) [2018\)](#page-4-11) and LSTM with a CNN [\(Chiu](#page-4-12) [and Nichols,](#page-4-12) [2016\)](#page-4-12) showed a significant improvement in performance. [Liu et al.](#page-4-13) [\(2018\)](#page-4-13); [Trivedi](#page-5-10) [et al.](#page-5-10) [\(2018\)](#page-5-10) proposed a character-level LSTM to capture the underlying style and structure, such as word boundaries and spellings. Finally, wordembedding ensemble techniques and preprocessing techniques, such as tokenization and normalization have been introduced to reduce OOV is-

- <span id="page-4-3"></span>sues [\(Winata et al.,](#page-5-8) [2018b;](#page-5-8) [Wang et al.,](#page-5-11) [2018\)](#page-5-11). 7 Conclusion In this paper, we propose a novel approach to learn multilingual representations by leveraging monolingual pre-trained embeddings. MME solves the dependencies on the language identification in code-switching Named Entity Recognition task since it utilizes more information from semantically similar embeddings. The experiment results show that our method surpasses previous works and baselines, achieving the state-of-the-art performance. Moreover, cross-lingual setting experiments demonstrate the generalization ability of MME to an unseen language task. Acknowledgments We want to thank Samuel Cahyawijaya for insightful discussions about this project. This work has been partially funded by ITF/319/16FP and MRP/055/18 of the Innovation Technology Commission, the Hong Kong SAR Government, and School of Engineering Ph.D. Fellowship Award, the Hong Kong University of Science and Technology, and RDC 1718050-0 of EMOS.AI. References Gustavo Aguilar, Fahad AlGhamdi, Victor Soto, Mona Diab, Julia Hirschberg, and Thamar Solorio. 2018. [Named entity recognition on code-switched data:](https://www.aclweb.org/anthology/W18-3219) [Overview of the calcs 2018 shared task.](https://www.aclweb.org/anthology/W18-3219) In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 138– 147, Melbourne, Australia. Association for Computational Linguistics. Utsab Barman, Amitava Das, Joachim Wagner, and Jennifer Foster. 2014. Code mixing: A challenge for language identification in the language of social media. In *Proceedings of the first workshop on computational approaches to code switching*, pages 13–
- <span id="page-4-0"></span>23. Danushka Bollegala, Kohei Hayashi, and Ken-Ichi Kawarabayashi. 2018. Think globally, embed locally: locally linear meta-embedding of words. In *Proceedings of the 27th International Joint Conference on Artificial Intelligence*, pages 3970–3976. AAAI Press. Jason PC Chiu and Eric Nichols. 2016. Named entity recognition with bidirectional lstm-cnns. *Transactions of the Association for Computational Linguistics*, 4:357–370. Stephen Clark. 2015. Vector space models of lexical meaning. *Handbook of Contemporary Semantics*, 10:9781118882139. Joshua Coates and Danushka Bollegala. 2018. Frustratingly easy meta-embedding–computing metaembeddings by averaging source word embeddings. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, pages 194–198. Saurabh Garg, Tanmay Parekh, and Preethi Jyothi. 2018. Code-switched language models using dual rnns and same-source pretraining. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3078–3083. Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, and Tomas Mikolov. 2018. Learning word vectors for 157 languages. In *Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)*. Nathan Greenberg, Trapit Bansal, Patrick Verga, and Andrew McCallum. 2018. Marginal likelihood training of bilstm-crf for biomedical named entity recognition from disjoint label sets. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2824–2829. Douwe Kiela, Changhan Wang, and Kyunghyun Cho. 2018. Dynamic meta-embeddings for improved sentence representations. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1466–1477. John D. Lafferty, Andrew McCallum, and Fernando Pereira. 2001. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In *ICML*. Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 260–270. Bill Y Lin, Frank Xu, Zhiyi Luo, and Kenny Zhu. 2017. Multi-channel bilstm-crf model for emerging named entity recognition in social media. In *Proceedings of the 3rd Workshop on Noisy User-generated Text*, pages 160–165. Liyuan Liu, Jingbo Shang, Xiang Ren, Frank Fangzheng Xu, Huan Gui, Jian Peng, and Jiawei Han. 2018. Empower sequence labeling with task-aware neural language model. In *Thirty-Second AAAI Conference on Artificial Intelligence*.

<span id="page-4-5"></span>

<span id="page-4-13"></span><span id="page-4-12"></span><span id="page-4-11"></span><span id="page-4-10"></span><span id="page-4-9"></span><span id="page-4-8"></span><span id="page-4-7"></span><span id="page-4-6"></span><span id="page-4-4"></span><span id="page-4-2"></span><span id="page-4-1"></span>

<span id="page-5-14"></span><span id="page-5-13"></span><span id="page-5-12"></span><span id="page-5-11"></span><span id="page-5-10"></span><span id="page-5-9"></span><span id="page-5-8"></span><span id="page-5-7"></span><span id="page-5-6"></span><span id="page-5-5"></span><span id="page-5-4"></span><span id="page-5-3"></span><span id="page-5-2"></span><span id="page-5-1"></span><span id="page-5-0"></span>Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, pages 3111–3119. Giovanni Molina, Fahad AlGhamdi, Mahmoud Ghoneim, Abdelati Hawwari, Nicolas Rey-Villamizar, Mona Diab, and Thamar Solorio. 2016. Overview for the second shared task on language identification in code-switched data. In *Proceedings of the Second Workshop on Computational Approaches to Code Switching*, pages 40–49. Avo Muromägi, Kairit Sirts, and Sven Laur. 2017. Linear ensembles of word embedding models. In *Proceedings of the 21st Nordic Conference on Computational Linguistics*, pages 96–104. Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 1532–1543. Adithya Pratapa, Gayatri Bhat, Monojit Choudhury, Sunayana Sitaram, Sandipan Dandapat, and Kalika Bali. 2018. Language modeling for code-mixing: The role of linguistic theory based synthetic data. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, volume 1, pages 1543–1553. Lev Ratinov and Dan Roth. 2009. Design challenges and misconceptions in named entity recognition. In *Proceedings of the thirteenth conference on computational natural language learning*, pages 147–155. Association for Computational Linguistics. N Sobhana, Pabitra Mitra, and SK Ghosh. 2010. Conditional random field based named entity recognition in geological text. *International Journal of Computer Applications*, 1(3):143–147. Thamar Solorio, Elizabeth Blair, Suraj Maharjan, Steven Bethard, Mona Diab, Mahmoud Ghoneim, Abdelati Hawwari, Fahad AlGhamdi, Julia Hirschberg, Alison Chang, et al. 2014. Overview for the first shared task on language identification in code-switched data. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 62–72. Shashwat Trivedi, Harsh Rangwani, and Anil Kumar Singh. 2018. Iit (bhu) submission for the acl shared task on named entity recognition on codeswitched data. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 148–153. Chen-Tse Tsai, Stephen Mayhew, and Dan Roth. 2016. Cross-lingual named entity recognition via wikification. In *Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning*, pages 219–228. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *Advances in neural information processing systems*, pages 5998–6008. Changhan Wang, Kyunghyun Cho, and Douwe Kiela. 2018. Code-switched named entity recognition with embedding attention. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 154–158. Genta Indra Winata, Andrea Madotto, Chien-Sheng Wu, and Pascale Fung. 2018a. Code-switching language modeling using syntax-aware multi-task learning. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 62–67. Genta Indra Winata, Chien-Sheng Wu, Andrea Madotto, and Pascale Fung. 2018b. Bilingual character representation for efficiently addressing outof-vocabulary words in code-switching named entity recognition. In *Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching*, pages 110–114. Wenpeng Yin and Hinrich Schütze. 2016. Learning word meta-embeddings. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, volume 1, pages 1351–1360.