# Multi-Task Learning in Natural Language Processing: An Overview

 Shijie Chen Note: This work was done when the first author worked as a research assistant at Southern University of Science and Technology. email: [chen.10216@osu.edu](mailto:chen.10216@osu.edu) Affiliation: The Ohio State University, USA , Yu Zhang Note: Corresponding author email: [yu.zhang.ust@gmail.com](mailto:yu.zhang.ust@gmail.com) Affiliation: Southern University of Science and Technology, China and Qiang Yang email: [qyang@cse.ust.hk](mailto:qyang@cse.ust.hk) Affiliation: Hong Kong University of Science and Technology, China 

###### Abstract.

Deep learning approaches have achieved great success in the field of Natural Language Processing (NLP). However, directly training deep neural models often suffer from overfitting and data scarcity problems that are pervasive in NLP tasks. In recent years, Multi-Task Learning (MTL), which can leverage useful information of related tasks to achieve simultaneous performance improvement on these tasks, has been used to handle these problems. In this paper, we give an overview of the use of MTL in NLP tasks. We first review MTL architectures used in NLP tasks and categorize them into four classes, including parallel architecture, hierarchical architecture, modular architecture, and generative adversarial architecture. Then we present optimization techniques on loss construction, gradient regularization, data sampling, and task scheduling to properly train a multi-task model. After presenting applications of MTL in a variety of NLP tasks, we introduce some benchmark datasets. Finally, we make a conclusion and discuss several possible research directions in this field.

## 1\. Introduction

In recent years, data-driven neural models have achieved great success in machine learning problems. In the field of Natural Language Processing (NLP), the introduction of transformers ([Vaswani et al., 2017](#bib.bib130 "")) and pre-trained language models (PLMs) such as BERT ([Devlin et al., 2019](#bib.bib27 "")), T5 ([Raffel et al., 2020](#bib.bib103 "")) and GPT-3 ([Brown et al., 2020](#bib.bib9 "")) has led to a huge leap in the performance on multiple downstream tasks. While pre-training equips PLMs with general encyclopedic and linguistic knowledge, using PLMs on downstream tasks still requires task-specific adaptation. However, sufficiently training such models usually require a large amount of labeled training samples, which is often expensive for NLP tasks. With the increasing size of neural models, training them on downstream datasets also demands immense computing power as well as huge time and storage budget. To further improve model performance, combat the data scarcity problem, and facilitate cost-efficient task adaptation, researchers have adopted Multi-Task Learning (MTL) ([Caruana, 1997](#bib.bib10 ""); [Zhang and Yang, 2021](#bib.bib165 "")) for NLP tasks. More recently, with the uprising of generative pre-trained models ([Raffel et al., 2020](#bib.bib103 ""); [Brown et al., 2020](#bib.bib9 "")), notably large language models (LLMs), researchers have generalized the notion of performing tasks into following instructions ([Mishra et al., 2022](#bib.bib88 ""); [Xie et al., 2022](#bib.bib150 "")), which virtually makes any NLP task a text-to-text task. This further allows to fine-tune a language model on a huge collection of tasks in a unified sequence-to-sequence framework. As a result, contemporary LLMs set new state-of-the-art on a variety of tasks and demonstrate an impressive ability in adapting to new tasks under few-shot and zero-shot settings ([Wei et al., 2022](#bib.bib143 ""); [Sanh et al., 2022](#bib.bib110 "")), highlighting the instrumental role of multi-task learning in building strong models for natural language processing.

MTL trains machine learning models from multiple related tasks simultaneously or enhances the model for a specific task using auxiliary tasks. Learning from multiple tasks makes it possible for models to capture generalized and complementary knowledge from the tasks at hand besides task-specific features. Tasks in MTL can be tasks with assumed relatedness ([Collobert and Weston, 2008](#bib.bib22 ""); [de Souza et al., 2015](#bib.bib25 ""); [Gupta et al., 2016](#bib.bib43 ""); [Vijayaraghavan et al., 2017](#bib.bib131 ""); [Lan et al., 2017](#bib.bib62 "")), tasks with different styles of supervision (e.g., supervised and unsupervised tasks ([Luong et al., 2016](#bib.bib80 ""); [Hai et al., 2016](#bib.bib45 ""); [Lim et al., 2020](#bib.bib71 ""))), tasks with different types of goals (e.g., classification and generation ([Nishino et al., 2019](#bib.bib91 ""))), tasks with different levels of features (e.g., token-level and sentence-level features ([Søgaard and Goldberg, 2016](#bib.bib119 ""); [Lauscher et al., 2018](#bib.bib63 ""))), and even tasks in different modalities (e.g., text and image data ([Liu et al., 2016c](#bib.bib73 ""); [Suglia et al., 2020](#bib.bib125 ""))). Alternatively, we can treat the same task in multiple domains or languages as multiple tasks, which is also known as multi-domain learning ([Yang and Hospedales, 2015](#bib.bib154 "")) in some literature, and learn an MTL model from them.

MTL naturally aggregates training samples from datasets of multiple tasks and alleviates the data scarcity problem. The benefit is escalated when unsupervised or self-supervised tasks, such as language modeling, are included. This is especially meaningful for low-resource tasks and languages whose labeled dataset is sometimes too small to sufficiently train a model. In most cases, the enlarged training dataset reduces the risk of the overfitting and leads to more robust models. From this perspective, MTL acts similarly to data augmentation techniques ([Guo et al., 2018a](#bib.bib40 "")). However, MTL provides additional performance gain compared to data augmentation approaches, due to its ability to learn common knowledge shared by different tasks.

While the thirst for better performance has driven people to build increasingly large models, developing more compact and efficient models with competitive performance has also received a growing interest. Through implicit knowledge sharing during the training process, MTL models could match or even exceed the performance of their single-task counterparts using much less training samples ([Domhan and Hieber, 2017](#bib.bib29 ""); [Singla et al., 2018](#bib.bib118 "")). Besides, multi-task adapters ([Stickland and Murray, 2019](#bib.bib123 ""); [Pfeiffer et al., 2020](#bib.bib100 "")) transfer large pre-trained models to new tasks and languages by adding a modest amount of task-specific parameters. In this way, the costly fine-tuning of the entire model is avoided, which is important for real-world applications such as mobile computing and latency-sensitive services. Many NLP models leverage additional features, including hand-crafted features and those produced by automatic NLP tools. Through MTL on various linguistic tasks, such as chunking, Part-Of-Speech (POS) tagging, Named Entity Recognition (NER), and dependency parsing, we can reduce the reliance on external knowledge and prevent error propagation, which results in simpler models with potentially better performance ([Luan et al., 2018](#bib.bib79 ""); [Zhou et al., 2019](#bib.bib170 ""); [Sanh et al., 2019](#bib.bib111 ""); [Song et al., 2020b](#bib.bib121 "")).

This paper reviews the application of MTL in recent NLP research. We focus on the ways in which researchers apply MTL to downstream NLP tasks, including model architectures, training processes, and data sources. While most pre-trained language models take advantage of MTL during pre-training, they are not designed for specific down-stream tasks, and thus they are not in the focus of this paper. Depending on the objective of applying MTL, we denote by auxiliary MTL the case where auxiliary tasks are introduced to improve the performance of primary tasks and by joint MTL the case where multiple tasks are equally important.

We first introduce popular MTL architectures used in NLP tasks and categorize them into four classes, including parallel architecture, hierarchical architecture, modular architecture, and generative adversarial architecture (Section [2](#S2 "2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview")). Then we review optimization techniques of MTL for NLP tasks in terms of loss construction, data sampling, and task scheduling (Section [3](#S3 "3. Optimization for MTL Models ‣ Multi-Task Learning in Natural Language Processing: An Overview")). After that, we present applications of MTL, categorized into auxiliary MTL and joint MTL, in a variety of NLP tasks (Section [4](#S4 "4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview")), and introduce some MTL benchmark datasets used in NLP (Section [5](#S5 "5. Data Source and Benchmarks for Multi-task Learning ‣ Multi-Task Learning in Natural Language Processing: An Overview")). Finally, we conclude the whole paper and discuss several possible research topics in this field.

Notations. In this paper, we use lowercase letters, such as tt, to denote scalars and use lowercase letters in boldface, such as 𝐱\\mathbf{x}, to denote vectors. Uppercase letters, such as MM and TT, are used for constants and uppercase letters in boldface are used to represent matrices, including feature matrices like 𝐗\\mathbf{X} and weight matrices like 𝐖\\mathbf{W}. In general, a multi-task learning model, parametrized by θ\\theta, handles MM tasks on a dataset 𝒟\\mathcal{D} with a loss function ℒ\\mathcal{L}.

## 2\. MTL Architectures for NLP Tasks

The architectures of MTL models depend on the characteristics of the indented tasks as well as the design of the base models. When training generative models on instruction following, people usually train the entire model and focus more on data curation. We refer interested readers to another survey paper on instruction tuning ([Zhang et al., 2023](#bib.bib163 "")). In this work, we mainly focus on reviewing MTL architectures with task-specifc trainable parameters.

Based on how the relatedness between tasks are utilized, we categorize MTL architectures into the following classes: parallel architecture, hierarchical architecture, modular architecture, and generative adversarial architecture. The parallel architecture shares the bulk of the model among multiple tasks while each task has its own task-specific output layer. The hierarchical architecture models the hierarchical relationships between tasks. Such architecture can hierarchically combine features from different tasks, take the output of one task as the input of another task, or explicitly model the interaction between tasks. The modular architecture decomposes the whole model into shared components and task-specific components that learn task-invariant and task-specific features, respectively. Different from the above three architectures, the generative adversarial architecture borrows the idea of the generative adversarial network ([Goodfellow et al., 2014](#bib.bib38 "")) to improve capabilities of existing models. Note that the boundaries between different categories are not always solid and hence a specific model may fit into multiple classes. Still, we believe that this taxonomy could illustrate important ideas behind the design of MTL architectures.

Before introducing MTL architectures, we would like to clarify the definitions of hard and soft parameter sharing. In this paper, hard parameter sharing refers to sharing the same model parameters among multiple tasks, and it is the most widely used approach in multi-task learning models. Soft parameter sharing, on the other hand, constrains a distance metric between the intended parameters, such as the Euclidean distance ([Guo et al., 2018b](#bib.bib41 "")) and correlation matrix penalty ([Hai et al., 2016](#bib.bib45 "")), to force certain parameters of models for different tasks to be similar. Alternatively, [Le et al. (2020)](#bib.bib64 "") add a regularization term to ensure the outputs of encoders of each task to be close for similar input instances. Differently, some researchers use hard parameter sharing to design a multi-task learning model that shares all the hidden layers except the final task-specific output layers and use soft parameter sharing to establish a multi-task model that partially shares its parameters ([Dankers et al., 2019](#bib.bib24 "")), such as embedding layers and low-level encoders. In this paper, such models fall into the ‘parallel architecture’ category.

### 2.1. Parallel Architectures

As its name suggests, the model for each task run in parallel under the parallel architecture, which is implemented by sharing certain intermediate layers. In this case, there is no dependency other than layer sharing among tasks. Therefore, there is no constraint on the order of training samples from each task. During training, the shared parameters receive gradients from samples of each task, enabling knowledge sharing among tasks. Fig. [1](#S2.F1 "Figure 1 ‣ 2.1.2. Parallel Feature Fusion ‣ 2.1. Parallel Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview") illustrates different forms of parallel architectures.

#### 2.1.1. Parallel Feature Sharing.

The simplest form of parallel architecture is a parallel feature sharing architecture (Fig. [1(a)](#S2.F1.sf1 "In Figure 1 ‣ 2.1.2. Parallel Feature Fusion ‣ 2.1. Parallel Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview")), where the models for different tasks share a base feature extractor (i.e., the trunk) followed by task-specific encoders and output layers (i.e., the branches). A shallow trunk can be simply the word representation layer ([Singla et al., 2018](#bib.bib118 "")) while a deep trunk can be the entire model except output layers. The tree-like architecture was proposed by [Caruana (1997)](#bib.bib10 "") and has been widely used in MTL ([Li and Zong, 2008](#bib.bib66 ""); [Wu and Huang, 2015](#bib.bib144 ""); [Luong et al., 2016](#bib.bib80 ""); [Bollmann and Søgaard, 2016](#bib.bib7 ""); [Gupta et al., 2016](#bib.bib43 ""); [Cummins et al., 2016](#bib.bib23 ""); [Liu et al., 2016b](#bib.bib76 ""); [Augenstein and Søgaard, 2017](#bib.bib5 ""); [Vijayaraghavan et al., 2017](#bib.bib131 ""); [Hashimoto et al., 2017](#bib.bib47 ""); [Masumura et al., 2018b](#bib.bib86 ""); [Tafreshi and Diab, 2018](#bib.bib126 ""); [Fares et al., 2018](#bib.bib31 ""); [Luan et al., 2018](#bib.bib79 ""); [Cerisara et al., 2018](#bib.bib11 ""); [Kochkina et al., 2018](#bib.bib59 ""); [Guo et al., 2018b](#bib.bib41 ""); [Liu et al., 2018b](#bib.bib74 ""); [Zhang et al., 2018a](#bib.bib164 ""); [Fei et al., 2019](#bib.bib32 ""); [Rawat et al., 2019](#bib.bib104 ""); [Nishida et al., 2019](#bib.bib90 ""); [Zhao et al., 2019](#bib.bib167 ""); [Pasunuru and Bansal, 2019](#bib.bib96 ""); [Shimura et al., 2019](#bib.bib117 ""); [Ye et al., 2019](#bib.bib155 ""); [Zalmout and Habash, 2019](#bib.bib157 ""); [Nishino et al., 2019](#bib.bib91 ""); [Shen et al., 2019](#bib.bib116 ""); [Watanabe et al., 2019](#bib.bib142 ""); [Cheng et al., 2020](#bib.bib18 ""); [Zhao et al., 2020](#bib.bib166 ""); [Jin et al., 2020](#bib.bib55 ""); [Song et al., 2020b](#bib.bib121 ""); [Chang et al., 2020](#bib.bib12 ""); [Chauhan et al., 2020](#bib.bib15 ""); [Wang et al., 2020a](#bib.bib136 ""); [Wang et al., 2020e](#bib.bib137 ""); [Liu et al., 2016a](#bib.bib78 ""); [Peng et al., 2017](#bib.bib97 ""); [Wang et al., 2018](#bib.bib138 ""); [Zheng et al., 2018](#bib.bib169 ""); [Kurita and Søgaard, 2019](#bib.bib60 ""); [Aminian et al., 2020](#bib.bib3 "")). In some literature, this architecture is also known as hard sharing architecture or multi-head architecture, where each head corresponds to the combination of a task-specific encoder and the corresponding output layer or just a branch.

Parallel feature sharing uses a single trunk to force all tasks to share the same low-level feature representation, which may limit the expressive power of the model for each task. A solution is to equip the shared trunk with task-specific encoders ([Xing et al., 2018](#bib.bib151 ""); [Hershcovich et al., 2018](#bib.bib51 ""); [Le et al., 2020](#bib.bib64 "")). For example, [Lin et al. (2018)](#bib.bib72 "") combine a shared character embedding layer and language-specific word embedding layers for different languages. Another way is to make different groups of tasks share different parts of the trunk ([Pasunuru and Bansal, 2017](#bib.bib95 ""); [Guo et al., 2018a](#bib.bib40 ""); [Masumura et al., 2018b](#bib.bib86 "")). This idea can also be applied to the decoder. For instance, [Wang et al. (2020d)](#bib.bib139 "") share the trunk encoder with a source-side language model and shares the decoder with a target-side denoising autoencoder.

#### 2.1.2. Parallel Feature Fusion

Different from learning shared features implicitly by sharing model parameters in the trunk, MTL models can actively combine features from different tasks, including shared and task-specific features, to form representations for each task. As shown in Fig. [1(b)](#S2.F1.sf2 "In Figure 1 ‣ 2.1.2. Parallel Feature Fusion ‣ 2.1. Parallel Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"), such models can use a globally shared encoder to produce shared representations that can be used as additional features for each task-specific model ([Liu et al., 2016b](#bib.bib76 "")). The shared representations can also be used indirectly as the key for attention layers in each task-specific model ([Tian et al., 2019](#bib.bib128 "")).

(a) Parallel Feature Sharing

(b) Parallel Feature Fusion

(c) Parallel Multi-level Supervision

Figure 1. Illustration for parallel architectures. For task tt, ht(i)h\_{t}^{(i)} represents the latent representation at the ii-th layer and yty\_{t} represents the corresponding label (hsh\_{s} are shared latent representations). The green blocks represent shared parameters and the orange blocks are task-specific parameters. Red circles represent feature fusion mechanism ff.

However, simply aggregating features of different tasks via weighted sum ([Li and Lam, 2017](#bib.bib67 "")) or attention ([Zheng et al., 2018](#bib.bib169 "")) is sub-optimal since these features might actually hurt the performance of other tasks, also known as inter-task interference. Researchers have proposed to do more fine-grained feature sharing between tasks to counter this issue. One approach is to directly aggregate shared and task-specific features using learnable feed-forward layers ([Zhang et al., 2017](#bib.bib162 ""); [Gupta et al., 2019](#bib.bib42 "")) or gating mechanisms ([Lan et al., 2017](#bib.bib62 ""); [Dankers et al., 2019](#bib.bib24 "")). Additionally, feature sharing can be indirectly performed by maintaining memory units that are shared among different tasks either globally or in pairs ([Liu et al., 2016b](#bib.bib76 ""); [Wu et al., 2019](#bib.bib146 "")).

A more generalized approach for inter-task feature sharing is modeling task relatedness and sharing features accordingly. As an example, Sluice network ([Ruder et al., 2019](#bib.bib108 "")) controls feature transfer by a learned task relatedness matrix. Instead of using a fixed relatedness matrix, LK-MTL ([Xiao et al., 2018a](#bib.bib148 "")) uses leaky units to dynamically control pairwise feature flow based on input features, and similar to RNN cells, it modulates information flow by two gates. Specifically, given two tasks mm and nn, the leaky gate 𝐫m​n\\mathbf{r}\_{mn} determines how much knowledge should be transferred from task nn to task mm and emits a feature map 𝐡\~m​n\\tilde{\\mathbf{h}}\_{mn}. The update gate 𝐳m​n\\mathbf{z}\_{mn} determines how much information should be maintained from task mm and emits the final output 𝐡\~m\\tilde{\\mathbf{h}}\_{m} for task mm. Mathematically, the feature sharing process is formulated as:

𝐫m​n\\displaystyle\\mathbf{r}\_{mn}

\=σ⁡(𝐖r⋅\[𝐡m,𝐡n\])\\displaystyle=\\sigma(\\mathbf{W}\_{r}\\cdot\[\\mathbf{h}\_{m},\\mathbf{h}\_{n}\])

𝐡\~m​n\\displaystyle\\tilde{\\mathbf{h}}\_{mn}

\=tanh⁡(𝐔⋅𝐡m+𝐖⋅(𝐫m​n⊙𝐡n))\\displaystyle=\\mathrm{tanh}(\\mathbf{U}\\cdot\\mathbf{h}\_{m}+\\mathbf{W}\\cdot(\\mathbf{r}\_{mn}\\odot\\mathbf{h}\_{n}))

𝐳m​n\\displaystyle\\mathbf{z}\_{mn}

\=σ⁡(𝐖z⋅\[𝐡m,𝐡n\])\\displaystyle=\\sigma(\\mathbf{W}\_{z}\\cdot\[\\mathbf{h}\_{m},\\mathbf{h}\_{n}\])

𝐡\~m\\displaystyle\\tilde{\\mathbf{h}}\_{m}

\=𝐳m​n⋅𝐡m+(1−𝐳m​n)⋅𝐡\~m​n,\\displaystyle=\\mathbf{z}\_{mn}\\cdot\\mathbf{h}\_{m}+(1-\\mathbf{z}\_{mn})\\cdot\\tilde{\\mathbf{h}}\_{mn}, where σ⁡(⋅)\\sigma(\\cdot) denotes the sigmoid function and tanh⁡(⋅)\\mathrm{tanh}(\\cdot) denotes the hyperbolic tangent function. When considering all pairwise directions, the output for each task is given by the sum of each row in

\[∑k\=1M𝐳1​k(1−𝐳12)⋯(1−𝐳1​M)(1−𝐳21)∑k\=1M𝐳2​k⋅OPEN1−𝐳2​M)⋮⋮⋱⋮(1−𝐳M​1)(1−𝐳M​2)⋯∑k\=1M𝐳M​k\]⋅\[𝐡1𝐡12⋯𝐡1​M𝐡\~21i𝐡2⋯𝐡\~2​Mi⋮⋮⋱⋮𝐡\~M​1i𝐡\~M​2i⋯𝐡\~Mi\]/M.\\left\[\\begin{array}\[\]{cccc}\\sum\_{k=1}^{M}\\mathbf{z}\_{1k}&\\left(1-\\mathbf{z}\_{12}\\right)&\\cdots&\\left(1-\\mathbf{z}\_{1M}\\right)\\\\ \\left(1-\\mathbf{z}\_{21}\\right)&\\sum\_{k=1}^{M}\\mathbf{z}\_{2k}&\\cdot&\\left.1-\\mathbf{z}\_{2M}\\right)\\\\ \\vdots&\\vdots&\\ddots&\\vdots\\\\ \\left(1-\\mathbf{z}\_{M1}\\right)&\\left(1-\\mathbf{z}\_{M2}\\right)&\\cdots&\\sum\_{k=1}^{M}\\mathbf{z}\_{Mk}\\end{array}\\right\]\\cdot\\left\[\\begin{array}\[\]{cccc}\\mathbf{h}\_{1}&\\mathbf{h}\_{12}&\\cdots&\\mathbf{h}\_{1M}\\\\ \\tilde{\\mathbf{h}}\_{21}^{i}&\\mathbf{h}\_{2}&\\cdots&\\tilde{\\mathbf{h}}\_{2M}^{i}\\\\ \\vdots&\\vdots&\\ddots&\\vdots\\\\ \\tilde{\\mathbf{h}}\_{M1}^{i}&\\tilde{\\mathbf{h}}\_{M2}^{i}&\\cdots&\\tilde{\\mathbf{h}}\_{M}^{i}\\end{array}\\right\]/M.

Task routing is another method for dynamic feature fusion, where the paths that samples go through in the model differ by their tasks. Given MM tasks, the routing network in ([Zaremoodi et al., 2018](#bib.bib158 "")) splits RNN cells into several shared blocks with MM task-specific blocks (one for each task) and then modulates the input to as well as output from each RNN block by a learned weight. MCapsNet ([Xiao et al., 2018b](#bib.bib149 "")), which adapts CapsNet ([Sabour et al., 2017](#bib.bib109 "")) to NLP tasks, replaces dynamic routing in CapsNet with task routing to build different feature spaces for each task. In MCapsNet, similar to dynamic routing, task routing computes task coupling coefficients ci​j(k)c\_{ij}^{(k)} for capsule ii in the current layer and capsule jj in the next layer for task kk. Due to the fine-grained dynamic control of information flow between tasks, LK-MTL and MCapsNet outperform other feature fusion methods and obtain state-of-the-art performance.

#### 2.1.3. Parallel Multi-level Supervision.

While models using the parallel architecture handle multiple tasks in parallel, these tasks may concern features at different abstraction levels. For NLP tasks, such levels can be character-level, token-level, sentence-level, paragraph-level, and document-level. Due to the compositional nature of language, both syntactically and semantically, it is natural to give supervision signals at different depths of an MTL model for tasks at different levels ([Collobert and Weston, 2008](#bib.bib22 ""); [Søgaard and Goldberg, 2016](#bib.bib119 ""); [Mishra et al., 2018](#bib.bib87 ""); [Sanh et al., 2019](#bib.bib111 "")) as illustrated in Fig. [1(c)](#S2.F1.sf3 "In Figure 1 ‣ 2.1.2. Parallel Feature Fusion ‣ 2.1. Parallel Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"). For example, in ([Lauscher et al., 2018](#bib.bib63 ""); [Farag and Yannakoudakis, 2019](#bib.bib30 "")), token-level tasks receive supervisions at lower-layers while sentence-level tasks receive supervision at higher layers. [Rawat et al. (2019)](#bib.bib104 "") supervise a higher-level QA task on both sentence and document-level features in addition to a sentence similarity prediction task that only relies on sentence-level features. In addition, [Gong et al. (2019)](#bib.bib36 ""); [Perera et al. (2018)](#bib.bib99 "") add skip connections so that signals from higher-level tasks are amplified. [Chaplot et al. (2020)](#bib.bib14 "") learn semantic goal navigation at a lower level and learns the task of embodied question answering at a higher level.

In some settings where MTL is used to improve the performance of a primary task, the introduction of auxiliary tasks at different levels could be helpful. Several works integrate a language modeling task on lower-level encoders for better performance on simile detection ([Rei, 2017](#bib.bib105 "")), sequence labeling ([Liu et al., 2018a](#bib.bib75 "")), question generation ([Zhou et al., 2019](#bib.bib170 "")), and task-oriented dialogue generation ([Zhou et al., 2019](#bib.bib170 "")). [Li and Caragea (2019)](#bib.bib69 "") add sentence-level sentiment classification and attention-level supervision to assist the primary stance detection task. [Nishino et al. (2019)](#bib.bib91 "") add attention-level supervision to improve consistency of the two primary language generation tasks. [Chuang et al. (2020)](#bib.bib19 "") minimize an auxiliary cosine softmax loss based on the audio encoder to learn more accurate speech-to-semantic mappings.

### 2.2. Hierarchical Architectures

The hierarchical architecture considers hierarchical relationships among multiple tasks. The features and output of one task can be used by another task as an extra input or additional control signals. The design of hierarchical architectures depends on the tasks at hand and is usually more complicated than parallel architectures. Fig. [2](#S2.F2 "Figure 2 ‣ 2.2. Hierarchical Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview") illustrates different hierarchical architectures. We notice that parallel MTL architectures usually assume the features shared are in the same feature space. Thus they should be processed by similar model architectures. In contrast, Hierarchical MTL architectures allow independent processing for each task and could accommodate tasks with data in heterogeneous feature spaces such as text, knowledge graphs, images, and audio.

(a) Hierarchical Feature Fusion

(b) Hierarchical Pipeline

(c) Hierarchical Interactive MTL

Figure 2. Illustration for hierarchical architectures. hh represents different hidden states and y^t\\hat{y}\_{t} represents the predicted output distribution for task tt. Red boxes stand for hierarchical feature fusion mechanisms. The purple block and blue circle in (b) stand for hierarchical feature and signal pipeline unit respectively.

#### 2.2.1. Hierarchical Feature Fusion

Different from parallel feature fusion that combines features of different tasks at the same depth, hierarchical feature fusion can explicitly combine features at different depths and allow different processing for different features. To solve the Twitter demographic classification problem, [Vijayaraghavan et al. (2017)](#bib.bib131 "") encode the name, following network, profile description, and profile picture features of each user by different neural models and combines the outputs using an attention mechanism. [Liu et al. (2018a)](#bib.bib75 "") take the hidden states for tokens in simile extraction as an extra feature in the sentence-level simile classification task. For knowledge base question answering, [Deng et al. (2019)](#bib.bib26 "") combine lower level word and knowledge features with more abstract semantic and knowledge semantic features by a weighted sum. ([Wang et al., 2020c](#bib.bib135 "")) fuses topic features of different roles into the main model via a gating mechanism. In ([Chauhan et al., 2020](#bib.bib15 "")), text and video features are combined through inter-modal attention mechanisms of different granularity to improve performance of sarcasm detection.

#### 2.2.2. Hierarchical Pipeline

Instead of aggregating features from different tasks as in feature fusion architectures, pipeline architectures treat the output of a task as an extra input of another task and form a hierarchical pipeline between tasks. In this section, we refer to *output* as the final result for a task, including the final output distribution and hidden states before the last output layer. The extra input can be used directly as input features or used indirectly as control signals to enhance the performance of other tasks. Therefore, we further divide hierarchical pipeline architectures into hierarchical feature pipeline and hierarchical signal pipeline.

In hierarchical feature pipeline, the output of one task is used as extra features for another task. The tasks are assumed to be directly related so that outputs instead of hidden feature representations are helpful to other tasks. For example, [Chen et al. (2019)](#bib.bib17 "") feed the output of a question-review pair recognition model to the question answering model. [He et al. (2019)](#bib.bib48 "") feed the output of aspect term extraction to aspect-term sentiment classification. Targeting community question answering, [Yang et al. (2019)](#bib.bib153 "") use the result of question category prediction to enhance document representations. [Song and Park (2019)](#bib.bib120 "") feed the result of morphological tagging to a POS tagging model and the two models are further tied by skip connections.

Hierarchical feature pipeline is especially useful for tasks at different abstraction levels. [Fei et al. (2019)](#bib.bib32 "") use the output of neighboring word semantic type prediction as extra features for neighboring word prediction. [Hashimoto et al. (2017)](#bib.bib47 "") use skip connections to forward predictions of lower-level POS tagging, chunking, and dependency parsing tasks to higher-level entailment and relatedness classification tasks. In addition, deep cascade MTL ([Gong et al., 2019](#bib.bib36 "")) adds both residual connections and cascade connections to a single-trunk parallel MTL model with supervision at different levels, where residual connections forward hidden representations and cascade connections forward output distributions of a task to the prediction layer of another task. [Song et al. (2020a)](#bib.bib122 "") include the output of the low-level discourse element identification task in the organization grid, which consists of sentence-level, phrase-level, and document-level features of an essay, for the primary essay organization evaluation task. In ([Shimura et al., 2019](#bib.bib117 "")), the word predominant sense prediction task and the text categorization task share a transformer-based embedding layer and embeddings of certain words in the text categorization task could be replaced by prediction results of the predominant sense prediction task.

The direction of hierarchical pipelines is not necessarily always from low-level tasks to high-level tasks. For example, in ([Alqahtani et al., 2020](#bib.bib2 "")), the outputs of word-level tasks are fed to the char-level primary task. [Rivas Rojas et al. (2020)](#bib.bib107 "") feed the output of more general classification models to more specific classification models during training, and the more general classification results are used to optimize beam search of more specific models at test time.

In hierarchical signal pipeline, the outputs of tasks are used indirectly as external signals to help improve the performance of other tasks. For example, the predicted probability of the sentence extraction task can be used to weigh sentence embeddings for a document-level classification task ([Isonuma et al., 2017](#bib.bib54 "")). For the hashtag segmentation task, [Maddela et al. (2019)](#bib.bib81 "") first predict the probability of a hashtag being single-token or multi-token as an auxiliary task and further use the output to combine single-token and multi-token features. In ([Shen et al., 2019](#bib.bib116 "")), the output of an auxiliary entity type prediction task is used to disambiguate candidate entities for logical form prediction. The outputs of a task can also be used for post-processing. For instance, [Zeng et al. (2020b)](#bib.bib159 "") use the output of NER to help extract multi-token entities.

#### 2.2.3. Hierarchical Interactive MTL

Different from most machine learning models that give predictions in a single pass, hierarchical interactive MTL explicitly models the interactions between tasks via a multi-turn prediction mechanism which allows a model to refine its predictions over multiple steps with the help of the previous outputs from other tasks in a way similar to recurrent neural networks. [He et al. (2019)](#bib.bib48 "") maintain a shared latent representation which is updated by TT iterations. In cyclic MTL ([Zeng et al., 2020a](#bib.bib160 "")), the output of one task is used as an extra input to its successive lower-level task and the output of the last task is fed to the first one, forming a loop. Most hierarchical interactive MTL models as introduced above report that performance converges quickly at T\=2T=2 steps, showing the benefit and efficiency of doing multi-step prediction.

### 2.3. Modular Architectures

The idea behind the modular MTL architecture is simple: breaking an MTL model into shared modules and task-specific modules. The shared modules learn shared features from multiple tasks. Since the shared modules can learn from many tasks, they can be sufficiently trained and can generalize better, which is particularly meaningful for low-resource scenarios. On the other hand, task-specific modules learn features that are specific to a certain task. Compared with shared modules, task-specific modules are usually much smaller and thus less likely to suffer from overfitting caused by insufficient training data. The robustness of shared modules and the flexibility of task-specific modules makes modular architectures suitable for learning different tasks efficiently.

The simplest form of modular architectures is a single shared module coupled with task-specific modules as in parallel feature sharing described in Section [2.1.1](#S2.SS1.SSS1 "2.1.1. Parallel Feature Sharing. ‣ 2.1. Parallel Architectures ‣ 2. MTL Architectures for NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"). Besides, another common practice is to share the first embedding layers across tasks ([Zhuang and Liu, 2019](#bib.bib172 ""); [Le et al., 2020](#bib.bib64 "")). [Alqahtani et al. (2020)](#bib.bib2 "") share word and character embedding matrices and combines them differently for different tasks. [Sarwar et al. (2019)](#bib.bib112 "") share two encoding layers and a vocabulary lookup table between the primary neural machine translation task and the auxiliary representation learning task. Shared embeddings can be used alongside task-specific embeddings ([Li et al., 2019](#bib.bib65 ""); [Yadav et al., 2019](#bib.bib152 "")) as well. In addition to word embeddings, ([Zhang et al., 2018b](#bib.bib161 "")) shares label embeddings between tasks. Researchers have also developed modular architectures at a finer granularity. For example, [Tong et al. (2018)](#bib.bib129 "") split the model into task-specific encoders and language-specific encoders for multilingual dialogue evaluation. In ([Deng et al., 2019](#bib.bib26 "")), each task has its own encoder and decoder, while all tasks share a representation learning layer and a joint encoding layer. [Pentyala et al. (2019)](#bib.bib98 "") create encoder modules on different levels, including task level, task group level, and universal level.

(a) Bert and PALs ([Stickland and Murray, 2019](#bib.bib123 ""))

(b) MAD-X ([Pfeiffer et al., 2020](#bib.bib100 ""))

Figure 3. Illustration for multi-task adapters.

When adapting large pre-trained models to down-stream tasks, a common practice is to fine-tune a separate model for each task. While this approach usually attains good performance, it poses heavy computational and storage costs. A more cost-efficient way is to add lightweight task-specific trainable modules into a single shared frozen backbone model. A special case is prefix-tuning for adapting pre-trained generative language models ([Li and Liang, 2021](#bib.bib68 "")), where learnable prefix vectors are prepended to inputs to frozen language models as context. Several works train task-specific prompt vectors for MTL ([Vu et al., 2022](#bib.bib132 ""); [Asai et al., 2022](#bib.bib4 "")). [Wang et al. (2023)](#bib.bib140 "") further improve multi-task prefix-tuning by decomposing the task prompts into a task-shared prompt and smaller task-specific prompts.

Multi-task adapters adapt single-task models to multiple tasks by adding extra task-specific parameters (adapters). [Stickland and Murray (2019)](#bib.bib123 "") add task-specific Projected Attention Layers (PALs) in parallel with self-attention operations in a pre-trained BERT model. Here PALs in different layers share the same parameters to reduce model capacity and improve training speed. In Multiple ADapters for Cross-lingual transfer (MAD-X) ([Pfeiffer et al., 2020](#bib.bib100 "")), the model is decomposed into four types of adapters: language adapters, task adapters, invertible adapters, and its counterpart inversed adapters, where language adapters learn language-specific task-invariant features, task adapters learn language-invariant task-specific features, invertible adapters conversely map input embeddings from different tasks into a shared feature space, and inversed adapters map hidden states into domain-specific embeddings. MAD-X can perform quick domain adaptation by directly switching corresponding language and task adapters instead of training new models from the scratch.

Further, task adaptation modules can also be dynamically generated by a meta-network. As an example, Hypergrid transformer ([Tay et al., 2020](#bib.bib127 "")) scales the weight matrix HH of the second feed forward layer in each transformer block by the multiplication of two vectors as

𝐇⁡(𝐱)\=ϕ⁡(σ⁡((𝐋r​o​w⋅𝐱)​(𝐋c​o​l⋅𝐱)))⊙𝐖,\\mathbf{H}(\\mathbf{x})=\\phi(\\sigma((\\mathbf{L}\_{row}\\cdot\\mathbf{x})(\\mathbf{L}\_{col}\\cdot\\mathbf{x})))\\odot\\mathbf{W}, where 𝐋r​o​w\\mathbf{L}\_{row} and 𝐋c​o​l\\mathbf{L}\_{col} are either globally shared task feature vectors or local instance-wise feature vectors, ϕ\\phi is a scaling operation, 𝐱\\mathbf{x} is an input vector, and 𝐖\\mathbf{W} is a learnable weight matrix. Similarly, Hyperformer ([Karimi Mahabadi et al., 2021](#bib.bib57 "")) inserts feed-forward adapter modules, which are generated by a task-aware hypernetwork, between pre-trained Transformer layers for efficient adaptation. Differently, Conditionally Adaptive MTL (CA-MTL) ([Pilault et al., 2021](#bib.bib102 "")) implements task adapters in the self-attention operation of each transformer block based on task representations {𝐳i}\\{\\mathbf{z}\_{i}\\} as

Attention⁡(𝐐,𝐊,𝐕,𝐳i)\=softmax⁡(𝐌⁡(𝐳i)+𝐐𝐊Td)​𝐕\\mathrm{Attention}\\left(\\mathbf{Q},\\mathbf{K},\\mathbf{V},\\mathbf{z}\_{i}\\right)=\\mathrm{softmax}\\left(\\mathbf{M}\\left(\\mathbf{z}\_{i}\\right)+\\frac{\\mathbf{Q}\\mathbf{K}^{T}}{\\sqrt{d}}\\right)\\mathbf{V}

where 𝐌⁡(𝐳i)\=diag⁡(𝐀1′​(𝐳i),…,𝐀N′​(𝐳i))\\mathbf{M}(\\mathbf{z}\_{i})=\\mathrm{diag}(\\mathbf{A}^{\\prime}\_{1}(\\mathbf{z}\_{i}),\\dots,\\mathbf{A}^{\\prime}\_{N}(\\mathbf{z}\_{i})) is a diagonal block matrix consisting of NN learnable linear transformations over 𝐳i\\mathbf{z}\_{i}. Therefore, 𝐌⁡(𝐳i)\\mathbf{M}(\\mathbf{z}\_{i}) injects task-specific bias into the attention map in the self-attention mechanism. Similar adaptation operations are used in input alignment and layer normalization as well. Impressively, a single jointly trained Hypergrid transformer, Hyperformer, or CA-MTL model could match or outperform single-task fine-tuned models on multi-task benchmark datasets while only adding a negligible amount of parameters. Instead of generating adaptation parameters with hypernetworks, Mixture-of-Expert (MoE) models ([Shazeer et al., 2017](#bib.bib115 "")) adjust computation by routing input to different trainable expert modules and show performance improvement on MTL ([Kim et al., 2021](#bib.bib58 ""); [Gao et al., 2022](#bib.bib35 ""); [Zhao et al., 2023](#bib.bib168 "")). More recently, task-specific information has been introduced to the routing algorithm for further performance improvement ([Gupta et al., 2022](#bib.bib44 ""); [Pham et al., 2023](#bib.bib101 "")).

### 2.4. Generative Adversarial Architectures

Generative Adversarial Networks (GANs) have achieved great success in generative tasks for computer vision. The basic idea of GANs is to train a discriminator model that distinguishes generated images from ground truth ones and train the generator model to fool the discriminator. By jointly optimizing both models, we can obtain a generator that can produce more vivid images and a discriminator that is better at spotting synthesized images. A similar idea can be used in MTL for NLP tasks. By introducing a discriminator GG that predicts which task a given training instance comes from, the shared feature extractor EE is forced to produce more generalized task-invariant features ([Liu et al., 2017](#bib.bib77 ""); [Wang et al., 2018](#bib.bib138 ""); [Masumura et al., 2018a](#bib.bib85 ""); [Tong et al., 2018](#bib.bib129 ""); [Yadav et al., 2019](#bib.bib152 "")) and therefore improve the performance and robustness of the entire MTL model. In the training process of such models, the adversarial objective is usually formulated as

ℒa​d​v\=minθE⁡maxθD​∑t\=1M∑i\=1|𝒟t|dit​log⁡\[D⁡(E⁡(𝐗))\],\\mathcal{L}\_{adv}=\\min\_{\\theta\_{E}}\\max\_{\\theta\_{D}}\\sum\_{t=1}^{M}\\sum\_{i=1}^{|\\mathcal{D}\_{t}|}d\_{i}^{t}\\log\[D(E(\\mathbf{X}))\], where θE\\theta\_{E} and θD\\theta\_{D} denote model parameters for the feature extractor and discriminator, respectively, and ditd\_{i}^{t} denotes the one-hot task label.

An additional benefit of generative adversarial architectures is that unlabeled data can be fully utilized. [Wang et al. (2020c)](#bib.bib135 "") add an auxiliary generative model that reconstructs documents from document representations learned by the primary model and improves the quality of document representations by training the generative model on unlabeled documents. To improve the performance of an extractive machine reading comprehension model, [Ren et al. (2020)](#bib.bib106 "") use a self-supervised approach. First, a discriminator that rates the quality of candidate answers is trained on labeled samples. Then, during unsupervised adversarial training, the answer extractor tries to obtain a high score from the discriminator.

## 3\. Optimization for MTL Models

Optimization techniques of training MTL models are equally as important as the design of model architectures. In this section, we summarize optimization techniques for MTL models used in recent research literatures targeting NLP tasks, including loss construction, data sampling, and task scheduling.

### 3.1. Loss Construction

The most common approach to train an MTL model is to linearly combine loss functions of different tasks into a single global loss function. In this way, the entire objective function of the MTL model can be optimized through conventional learning techniques such as stochastic gradient descent with back-propagation. Different tasks may use different types of loss functions. For example, in ([Ye et al., 2019](#bib.bib155 "")), the cross-entropy loss for the relation identification task and the ranking loss for the relation classification task are linearly combined, which performs better than single-task learning. Specifically, given MM tasks each associated with a loss function ℒi\\mathcal{L}\_{i} and a weight λt\\lambda\_{t}, the overall loss ℒ\\mathcal{L} is defined as

ℒ\=∑t\=1Mλt​ℒt+∑λa​ℒa​d​a​p+∑λr​ℒr​e​g,\\mathcal{L}=\\sum\_{t=1}^{M}\\lambda\_{t}\\mathcal{L}\_{t}+\\sum\\lambda\_{a}\\mathcal{L}\_{adap}+\\sum\\lambda\_{r}\\mathcal{L}\_{reg}, where ℒt\\mathcal{L}\_{t}, ℒa​d​a​p\\mathcal{L}\_{adap}, and ℒr\\mathcal{L}\_{r} denotes loss functions of different tasks, adaptive losses, and regularization terms, with λt\\lambda\_{t}, λa\\lambda\_{a}, and λr​e​g\\lambda\_{reg} being their respective weights. For cases where the tasks are optimized in turns rather than joint training ([Subramanian et al., 2018](#bib.bib124 "")), λt\\lambda\_{t} is equivalent to the sampling weight ptp\_{t} for task tt, which will be discussed in Section [3.3](#S3.SS3 "3.3. Data Sampling ‣ 3. Optimization for MTL Models ‣ Multi-Task Learning in Natural Language Processing: An Overview").

An important question is how to assign a proper weight λt\\lambda\_{t} to each task. The simplest way is to set them equally ([Peng et al., 2017](#bib.bib97 ""); [Zhuang and Liu, 2019](#bib.bib172 ""); [Wang et al., 2020e](#bib.bib137 "")), i.e., λt\=1M\\lambda\_{t}=\\frac{1}{M}. As a generalization, the weights are usually viewed as hyper-parameters and set based on experience or through grid search ([Liu et al., 2016b](#bib.bib76 ""); [Lan et al., 2017](#bib.bib62 ""); [Liu et al., 2018b](#bib.bib74 ""); [Zhang et al., 2018b](#bib.bib161 ""); [Chen et al., 2018](#bib.bib16 ""); [Zhang et al., 2018a](#bib.bib164 ""); [Luan et al., 2018](#bib.bib79 ""); [Liu et al., 2018a](#bib.bib75 ""); [Shao et al., 2019](#bib.bib114 ""); [Gupta et al., 2019](#bib.bib42 ""); [Maddela et al., 2019](#bib.bib81 ""); [Nishida et al., 2019](#bib.bib90 ""); [Sarwar et al., 2019](#bib.bib112 ""); [Yadav et al., 2019](#bib.bib152 ""); [Farag and Yannakoudakis, 2019](#bib.bib30 ""); [Dankers et al., 2019](#bib.bib24 ""); [Zhou et al., 2019](#bib.bib170 ""); [Zhu et al., 2019](#bib.bib171 ""); [Shen et al., 2019](#bib.bib116 ""); [Xia et al., 2019](#bib.bib147 ""); [Deng et al., 2019](#bib.bib26 ""); [Wu et al., 2019](#bib.bib146 ""); [Zeng et al., 2020a](#bib.bib160 ""); [Zeng et al., 2020a](#bib.bib160 ""); [Zeng et al., 2020b](#bib.bib159 ""); [Wang et al., 2020c](#bib.bib135 ""); [Ren et al., 2020](#bib.bib106 ""); [Chang et al., 2020](#bib.bib12 ""); [Cheng et al., 2020](#bib.bib18 ""); [Zhao et al., 2020](#bib.bib166 ""); [Song et al., 2020b](#bib.bib121 "")). For example, to prevent large datasets from dominating training, [Perera et al. (2018)](#bib.bib99 "") set the weights as

λt∝1|𝒟t|,\\lambda\_{t}\\propto\\frac{1}{|\\mathcal{D}\_{t}|}\\ , where |𝒟t||\\mathcal{D}\_{t}| denotes the size of the training dataset for task tt. The weights can also be adjusted dynamically during the training process based on certain metrics. Through adjusting weights, we can purposely emphasize different tasks in different training stages. For instance, since dynamically assigning smaller weights to more uncertain tasks usually leads to good performance for MTL ([Cipolla et al., 2018](#bib.bib20 "")), ([Lauscher et al., 2018](#bib.bib63 "")) assigns weights based on the homoscedasticity of training losses from different tasks as

λt\=12​σt2,\\lambda\_{t}=\\frac{1}{2\\sigma\_{t}^{2}}, where σt\\sigma\_{t} measures the variance of the training loss for task tt. In ([Lim et al., 2020](#bib.bib71 "")), the weight of an unsupervised task is set to a confidence score that measures how much a prediction resembles the corresponding self-supervised label. To ensure that a student model could receive enough supervision during knowledge distillation, BAM! ([Clark et al., 2019](#bib.bib21 "")) combines the supervised loss ℒs​u​p\\mathcal{L}\_{sup} with the distillation loss ℒd​i​s​s\\mathcal{L}\_{diss} as

ℒ\=λ​ℒd​i​s​s+(1−λ)​ℒs​u​p,\\mathcal{L}=\\lambda\\mathcal{L}\_{diss}+(1-\\lambda)\\mathcal{L}\_{sup}, where λ\\lambda increases linearly from 0 to 1 in the training process. In ([Song et al., 2020a](#bib.bib122 "")), three tasks are jointly optimized, including the primary essay organization evaluation (OE) task as well as the auxiliary sentence function identification (SFI) and paragraph function identification (PFI) tasks. The two lower-level auxiliary tasks are assumed to be equally important with weights set to 1 (i.e., λS​F​I\=λP​F​I\=1\\lambda\_{SFI}=\\lambda\_{PFI}=1) and the weight of the OE task is set as

λO​E\=max⁡(min⁡(ℒO​EℒS​F​I⋅λO​E,1),0.01),\\lambda\_{OE}=\\max\\left(\\min\\left(\\frac{\\mathcal{L}\_{OE}}{\\mathcal{L}\_{SFI}}\\cdot\\lambda\_{OE},1\\right),0.01\\right), where λO​E\\lambda\_{OE} is initialized to 0.1 and then dynamically updated during training, so that the model focuses on the lower-level tasks at first before λO​E\\lambda\_{OE} becomes larger when ℒS​F​I\\mathcal{L}\_{SFI} gets relatively smaller. [Nishino et al. (2019)](#bib.bib91 "") guide the model to focus on easy tasks by setting weights as

(1)

λt​(e)\=λtc​o​n​s​t1+exp⁡((et′−e)/α),\\lambda\_{t}(e)=\\frac{\\lambda\_{t}^{const}}{1+\\exp((e\_{t}^{\\prime}-e)/\\alpha)}, where ee denotes the number of epochs, λtc​o​n​s​t\\lambda\_{t}^{const} and et′e\_{t}^{\\prime} are hyperparameters for each task, and α\\alpha denotes temperature.

In addition to combining loss functions from different tasks, researchers also use additional adaptive loss functions ℒa​d​a​p​t\\mathcal{L}\_{adapt} to enhance MTL models. In ([Li and Caragea, 2019](#bib.bib69 "")), the alignment between an attention vector and a hand-crafted lexicon feature vector is normalized to encourage the model to attend to important words in the input. [Chen et al. (2019)](#bib.bib17 "") penalize the similarity between attention vectors from two tasks and the Euclidean distance between the resulting feature representations to enforce the models to focus on different task-specific features. To learn domain-invariant features, [Xing et al. (2018)](#bib.bib151 "") minimize a distance function g⁡(⋅)g(\\cdot) between a pair of learned representations from different tasks. Candidates of g⁡(⋅)g(\\cdot) include the KL divergence, maximum mean discrepancy (MMD), and central moment discrepancy (CMD). Extensive experiments show that KL divergence gives overall stable improvements on all experiments while CMD hits more best scores.

The L1L\_{1} metric linearly combines different loss functions and optimizes all tasks simultaneously. However, when we view multi-task learning as a multi-objective optimization problem, this type of objective functions cannot guarantee optimality in obtaining Pareto-optimal models when each loss function is non-convex. To address this issue, Tchebycheff loss ([Mao et al., 2020](#bib.bib82 "")) optimizes an MTL model by an L∞L\_{\\infty} objective, which is formulated as

ℒc​h​e​b\=maxt⁡{λ1​ℒ1​(θs​h,θ1),…,λM​ℒM​(θs​h,θM)}{\\mathcal{L}}\_{cheb}=\\max\_{t}\\left\\{\\lambda\_{1}{\\mathcal{L}}\_{1}\\left(\\theta^{sh},\\theta^{1}\\right),\\ldots,\\lambda\_{M}{\\mathcal{L}}\_{M}\\left(\\theta^{sh},\\theta^{M}\\right)\\right\\}

where ℒt\\mathcal{L}\_{t} denotes the training loss for task tt, θs​h\\theta^{sh} denotes the shared model parameters, θi\\theta^{i} denotes task-specific model parameters for task ii, ltl\_{t} denotes the empirical loss of task tt, and λt\=1l¯t​∑i\=1T1l¯i\\lambda\_{t}=\\frac{1}{\\bar{l}\_{t}\\sum\_{i=1}^{T}\\frac{1}{\\bar{l}\_{i}}}. The Tchebycheff loss can be combined with aforementioned adversarial MTL as well ([Liu et al., 2017](#bib.bib77 "")).

Note that adjusting loss weight λt\\lambda\_{t} of each task could guide the model to focus on different tasks during training while still learning multiple tasks at the same time, which can be seen as implicit task scheduling, compared to explicit task scheduling, which will be discussed in Section [3.4](#S3.SS4 "3.4. Task Scheduling ‣ 3. Optimization for MTL Models ‣ Multi-Task Learning in Natural Language Processing: An Overview"). In general, auxiliary MTL models are often bootstrapped with easier or lower-level tasks. For joint MTL, one would want to emphasize difficult tasks or tasks with lower homoscedasticity.

### 3.2. Gradient Regularization

Aside from studying how to combine loss functions of different tasks, some studies optimize the training process by manipulating gradients. When jointly learning multiple tasks, the gradients from different tasks may be in conflict with each other, causing inter-task interference that harms performance. PCGrad ([Yu et al., 2020](#bib.bib156 "")) resolves such conflict using gradient projections. Specifically, given two conflicting gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j} from tasks ii and jj, respectively, PCGrad projects 𝐠i\\mathbf{g}\_{i} onto the normal plane of 𝐠j\\mathbf{g}\_{j} as

𝐠i′\=𝐠i−𝐠i⋅𝐠j‖𝐠j‖2​𝐠j.\\mathbf{g}\_{i}^{\\prime}=\\mathbf{g}\_{i}-\\frac{\\mathbf{g}\_{i}\\cdot\\mathbf{g}\_{j}}{\\left\\|\\mathbf{g}\_{j}\\right\\|^{2}}\\mathbf{g}\_{j}.

Based on the observation that gradient similarity correlates well with language similarity and model performance, GradVac ([Wang et al., 2020b](#bib.bib141 "")), which targets at optimization of multilingual models, regulates parameter updates according to geometry similarities between gradients. That is, GradVac alters both the direction and magnitude of gradients so that they are aligned with the cosine similarity between gradient vectors by modifying 𝐠i\\mathbf{g}\_{i} as

𝐠i′\=𝐠i+‖𝐠i‖​(ϕi​jT​1−ϕi​j2−ϕi​j​1−(ϕi​jT)2)‖𝐠j‖​1−(ϕi​jT)2⋅𝐠j\\mathbf{g}\_{i}^{\\prime}=\\mathbf{g}\_{i}+\\frac{\\left\\|\\mathbf{g}\_{i}\\right\\|\\left(\\phi\_{ij}^{T}\\sqrt{1-\\phi\_{ij}^{2}}-\\phi\_{ij}\\sqrt{1-\\left(\\phi\_{ij}^{T}\\right)^{2}}\\right)}{\\left\\|\\mathbf{g}\_{j}\\right\\|\\sqrt{1-\\left(\\phi\_{ij}^{T}\\right)^{2}}}\\cdot\\mathbf{g}\_{j}

where ϕi​j∈\[−1,1\]\\phi\_{ij}\\in\[-1,1\] is the cosine distance between gradients 𝐠i\\mathbf{g}\_{i} and 𝐠j\\mathbf{g}\_{j}. Notice that PCGrad is a special case of GradVac when ϕi​jT\=0\\phi\_{ij}^{T}=0. While PCGrad does not modify positively associated gradients, GradVac aligns both positively and negatively associated gradients, leading to a consist performance improvement for multilingual models.

### 3.3. Data Sampling

Machine learning models often suffer from imbalanced data distributions. MTL further complicates this issue in that training datasets of multiple tasks with potentially different sizes and data distributions are involved. Various data sampling techniques have been proposed to properly construct training datasets. In practice, given MM tasks and their datasets {𝒟1,…,𝒟M}\\{\\mathcal{D}\_{1},\\dots,\\mathcal{D}\_{M}\\}, a sampling weight ptp\_{t} is assigned to task tt to control the probability of sampling a data batch from 𝒟t\\mathcal{D}\_{t} in each training step.

In general, ptp\_{t} takes the form of:

pt∝|𝒟t|1αp\_{t}\\propto|\\mathcal{D}\_{t}|^{\\frac{1}{\\alpha}}

where α{\\alpha} is the sampling temperature. When α\>1\\alpha>1, the divergence of sampling probabilities between tasks is reduced and vice versa. α\\alpha can be either a constant hyperparameter or can be changed dynamically during training. Similar to task loss weights, researchers have proposed various techniques to adjust α\\alpha. For example, the annealed sampling method ([Stickland and Murray, 2019](#bib.bib123 "")) adjusts α\\alpha as training proceeds. Given a total number of EE epochs, α\\alpha at epoch ee is set to

α⁡(e)\=11−0.8​(e−1)E−1.\\alpha(e)=\\frac{1}{1-\\frac{0.8(e-1)}{E-1}}.

In this way, the model is trained more evenly for different tasks towards the end of the training process to reduce inter-task interference. [Wang et al. (2020d)](#bib.bib139 "") define α\\alpha as

α⁡(e)\=min⁡(αm,(e−1)​αm−α0M+α0),\\alpha(e)=\\min\\left(\\alpha\_{m},(e-1)\\frac{\\alpha\_{m}-\\alpha\_{0}}{M}+\\alpha\_{0}\\right), where α0\\alpha\_{0} and αm\\alpha\_{m} denote initial and maximum values of α\\alpha. The noise level of the self-supervised denoising autoencoding task is scheduled similarly, increasing difficulty after a warm-up period. In both works, temperature α\\alpha increases during training which encourages up-sampling of low-resource tasks and alleviates overfitting.

### 3.4. Task Scheduling

Task scheduling determines the order of tasks on which an MTL model is trained. A naive way is to train all tasks together. [Zhang et al. (2017)](#bib.bib162 "") take this way to train an MTL model, where data batches are organized as four-dimensional tensors of size N×M×T×dN\\times M\\times T\\times d, where NN denotes the number of samples, MM denotes the number of tasks, TT denotes sequence length, and dd represents embedding dimensions. Similarly, [Zalmout and Habash (2019)](#bib.bib157 "") put labeled data and unlabeled data together to form a batch and [Xia et al. (2019)](#bib.bib147 "") learn the dependency parsing and semantic role labeling tasks together. In the case of auxiliary MTL, [Augenstein and Søgaard (2017)](#bib.bib5 "") train the primary task and one of the auxiliary tasks together at each step. Conversely, [Song et al. (2020b)](#bib.bib121 "") train one of the primary tasks and the auxiliary task together and shuffles between the two primary tasks.

Alternatively, we can train an MTL model on different tasks at different steps. Similar to data sampling techniques, we can assign a task sampling weight rtr\_{t} for task tt, which is also called mixing ratio, to control the frequency of data batches from task tt. The most common task scheduling technique is to shuffle between different tasks ([Collobert and Weston, 2008](#bib.bib22 ""); [Luong et al., 2016](#bib.bib80 ""); [Bollmann and Søgaard, 2016](#bib.bib7 ""); [Søgaard and Goldberg, 2016](#bib.bib119 ""); [Liu et al., 2016a](#bib.bib78 ""); [Pasunuru and Bansal, 2017](#bib.bib95 ""); [Subramanian et al., 2018](#bib.bib124 ""); [Masumura et al., 2018b](#bib.bib86 ""); [Guo et al., 2018b](#bib.bib41 ""); [Singla et al., 2018](#bib.bib118 ""); [Mishra et al., 2018](#bib.bib87 ""); [Perera et al., 2018](#bib.bib99 ""); [Fei et al., 2019](#bib.bib32 ""); [He et al., 2019](#bib.bib48 ""); [Sanh et al., 2019](#bib.bib111 ""); [Gong et al., 2019](#bib.bib36 ""); [Tian et al., 2019](#bib.bib128 ""); [Jin et al., 2020](#bib.bib55 ""); [Rivas Rojas et al., 2020](#bib.bib107 "")), either randomly or according to a pre-defined schedule. While random shuffling is widely adopted, introducing more heuristics into scheduling could help further improving the performance of MTL models. For example, according to the similarity between each task and the primary task in a multilingual multi-task scenario, [Lin et al. (2018)](#bib.bib72 "") define rtr\_{t} as

rt\=μt​ζt​|𝒟t|12,r\_{t}=\\mu\_{t}\\zeta\_{t}|\\mathcal{D}\_{t}|^{\\frac{1}{2}}, where μt\\mu\_{t} or ζt\\zeta\_{t} is set to 11 if the corresponding task or language is the same as the primary task and 0.10.1 otherwise.

Instead of using a fixed mixing ratio designed by hand, some researchers explore using a dynamic mixing ratio during the training process. [Gupta et al. (2016)](#bib.bib43 "") schedule tasks by a state machine that switches between the two tasks and updates learning rate when validation loss rises. [Guo et al. (2018a)](#bib.bib40 "") develop a controller meta-network that dynamically schedules tasks based on multi-armed bandits. The controller has MM arms and optimizes a control policy πe\\pi\_{e} for arm (task) tt at step ee based on an estimated action value Qe,tQ\_{e,t} defined as

πe​(t)\\displaystyle\\pi\_{e}(t)

\=exp⁡(Qe,t/τ)/∑i\=1Mexp⁡(Qe,i/τ)\\displaystyle=\\exp(Q\_{e,t}/\\tau)/\\sum\_{i=1}^{M}\\exp(Q\_{e,i}/\\tau)

Qe,t\\displaystyle Q\_{e,t}

\=(1−α)e​Q0,t+∑k\=1eα​(1−α)e−k​Rk\\displaystyle=(1-\\alpha)^{e}Q\_{0,t}+\\sum\_{k=1}^{e}\\alpha(1-\\alpha)^{e-k}R\_{k}

where τ\\tau denotes the temperature, α\\alpha is the decay rate, and RkR\_{k} is the observed reward at step kk that is defined as the negative validation loss of the primary task. Analysis shows that the bandit assigns a higher probability to the primary task at first and then more evenly switches between all tasks, which echos the dynamic data sampling techniques introduced in Section [3.3](#S3.SS3 "3.3. Data Sampling ‣ 3. Optimization for MTL Models ‣ Multi-Task Learning in Natural Language Processing: An Overview").

Besides probabilistic approaches, task scheduling could also use heuristics based on certain performance metrics. By optimizing the Tchebycheff loss, [Mao et al. (2020)](#bib.bib82 "") learn from the task which has the worst validation performance at each step. The CA-MTL model ([Pilault et al., 2021](#bib.bib102 "")) introduces an uncertainty-based sampling strategy based on Shannon entropy for joint learning of classification tasks. Specifically, given a batch size bb and MM tasks, a pool of b×Mb\\times M samples are first sampled. Then, the uncertainty measure 𝒰⁡(x)\\mathcal{U}(x) for a sample 𝐱\\mathbf{x} from task ii is defined as

𝒰⁡(𝐱)\=Si​(𝐱)S^×S′\\mathcal{U}\\left(\\mathbf{x}\\right)=\\frac{S\_{i}\\left(\\mathbf{x}\\right)}{\\hat{S}\\times S^{\\prime}}

where SS denotes the Shannon entropy of the model’s prediction on 𝐱\\mathbf{x}, S^\\hat{S} is the model’s maximum average entropy over the bb samples from each task. S′S^{\\prime} denotes the entropy of a uniform distribution and is used to normalize the variance of the number of classes in each task. At last, bb samples with the highest uncertainty measures are used for training at the current step. Experiments show that this uncertainty-based sampling strategy could effectively avoid catastrophic forgetting and inter-task interference when jointly learning multiple tasks, outperforming the aforementioned annealed sampling ([Stickland and Murray, 2019](#bib.bib123 "")).

In some cases, multiple tasks are learned sequentially. Such tasks usually form a clear dependency relationship or are of different difficulty levels. For instance, [Isonuma et al. (2017)](#bib.bib54 ""); [Nishino et al. (2019)](#bib.bib91 "") train MTL models on different tasks in the order of increasing difficulties. Similarly, [Hashimoto et al. (2017)](#bib.bib47 "") train a multi-task model in the order of low-level tasks, high-level tasks, and at last mixed-level batches. Unicoder ([Huang et al., 2019](#bib.bib53 "")) trains its five pre-training objectives sequentially in each step. [Pfeiffer et al. (2020)](#bib.bib100 "") first pre-train language and invertible adapters on language modeling before training task adapters on different down-stream tasks, where the language and invertible adapters can also receive gradient when training task adapters. To stabilize the training process when alternating between tasks with imbalanced dataset sizes, successive regularization ([Hashimoto et al., 2017](#bib.bib47 ""); [Fei et al., 2019](#bib.bib32 "")) can be added to loss functions as a regularization term, which is defined as ℒs​r\=δ​‖θe−θe′‖2\\mathcal{L}\_{sr}=\\delta\\left\\|\\theta\_{e}-\\theta\_{e}^{\\prime}\\right\\|^{2}, where θe\\theta\_{e} and θe′\\theta^{\\prime}\_{e} are model parameters before and after the update in the previous training step and δ\\delta is a hyperparameter.

To sum up, task scheduling for MTL aims at alleviate overfitting and negative transfer caused by imbalanced dataset size. For auxiliary MTL, depending on the relationship between tasks, we can either start with the primary task before training primary and auxiliary tasks together or adopt a pre-train then fine-tune approach ([Lamprinidis et al., 2018](#bib.bib61 ""); [He et al., 2019](#bib.bib48 ""); [Wang et al., 2020a](#bib.bib136 ""); [Chen et al., 2019](#bib.bib17 "")), which bootstraps the model with auxiliary tasks that are often easier or more data-rich. For joint MTL, we would like to choose tasks that are more likely to benefit the model. Generally, dynamic scheduling approaches like CA-MTL performs better than using a fixed mixing ratio.

## 4\. Application in NLP Tasks

In this section, we summarize the application of multi-task learning in NLP tasks, including applying MTL to optimize certain primary tasks (i.e., Auxiliary MTL), to jointly learn multiple tasks (i.e., Joint MTL), and to improve the performance in multilingual multi-task and multimodal scenarios. Existing research works have also explored different ways to improve the performance and efficiency of MTL models, as well as using MTL to study the relatedness of different tasks.

### 4.1. Auxiliary MTL

Table 1. A summary of auxiliary MTL studies according to types of primary and auxiliary tasks involved. ‘W’, ‘S’, and ‘D’ in the three rightmost columns represent word-level, sentence-level, and document-level tasks for auxiliary tasks, respectively. ‘LM’ denotes language modeling tasks and ‘Gen’ denotes text generation tasks. The ‘Architecture’ column denotes the architecture used, where PFS denotes Parallel Feature Sharing, PFF denotes Parallel Feature Fusion, PMS denotes Parallel Multi-level Supervision, HP denotes Hierarchical Pipeline, and GAA denotes Generative Adversarial Architecture.

Primary Task

Reference

W

S

D

Architecture

Tagging

Parsing

Chunking

LM

Gen

Classification

Classification

 Sequence Tagging 

([Augenstein and Søgaard, 2017](#bib.bib5 ""))

✓\\checkmark

✓\\checkmark

PFS

([Cheng et al., 2020](#bib.bib18 ""))

✓\\checkmark

PFS

([Le et al., 2020](#bib.bib64 ""))

✓\\checkmark

PFS

([Wang et al., 2020a](#bib.bib136 ""))

✓\\checkmark

✓\\checkmark

PFS

([Li and Lam, 2017](#bib.bib67 ""))

✓\\checkmark

✓\\checkmark

PFF

([Rei, 2017](#bib.bib105 ""))

✓\\checkmark

PMS

([Watanabe et al., 2019](#bib.bib142 ""))

✓\\checkmark

PMS

([Isonuma et al., 2017](#bib.bib54 ""))

✓\\checkmark

HP

([Xia et al., 2019](#bib.bib147 ""))

✓\\checkmark

HP

([Nishida et al., 2019](#bib.bib90 ""))

✓\\checkmark

HP

([Alqahtani et al., 2020](#bib.bib2 ""))

✓\\checkmark

✓\\checkmark

HP

Classification

([Lamprinidis et al., 2018](#bib.bib61 ""))

✓\\checkmark

✓\\checkmark

PFS

([Liu et al., 2018b](#bib.bib74 ""))

✓\\checkmark

PFS

([Wu et al., 2019](#bib.bib146 ""))

✓\\checkmark

PFF

([Kochkina et al., 2018](#bib.bib59 ""))

✓\\checkmark

✓\\checkmark

PFS

([Yadav et al., 2019](#bib.bib152 ""))

✓\\checkmark

PFF

([Li et al., 2019](#bib.bib65 ""))

✓\\checkmark

PFF

([Li and Caragea, 2019](#bib.bib69 ""))

✓\\checkmark

PMS

([Mishra et al., 2018](#bib.bib87 ""))

✓\\checkmark

PMS

([Rawat et al., 2019](#bib.bib104 ""))

✓\\checkmark

PMS

([Farag and Yannakoudakis, 2019](#bib.bib30 ""))

✓\\checkmark

PMS

([Maddela et al., 2019](#bib.bib81 ""))

✓\\checkmark

HP

([Shimura et al., 2019](#bib.bib117 ""))

✓\\checkmark

HP

([Yang et al., 2019](#bib.bib153 ""))

✓\\checkmark

HP

([Song et al., 2020a](#bib.bib122 ""))

✓\\checkmark

✓\\checkmark

HP

([Ren et al., 2020](#bib.bib106 ""))

✓\\checkmark

GAA

 Text Generation 

([Domhan and Hieber, 2017](#bib.bib29 ""))

✓\\checkmark

PFS

([Luong et al., 2016](#bib.bib80 ""))

✓\\checkmark

✓\\checkmark

PFS

([Wang et al., 2020d](#bib.bib139 ""))

✓\\checkmark

✓\\checkmark

PFS

([Guo et al., 2018a](#bib.bib40 ""))

✓\\checkmark

PFS

([Guo et al., 2018b](#bib.bib41 ""))

✓\\checkmark

PFS

([Shao et al., 2019](#bib.bib114 ""))

✓\\checkmark

✓\\checkmark

✓\\checkmark

PFS

([Zhu et al., 2019](#bib.bib171 ""))

✓\\checkmark

PFS

([Zaremoodi et al., 2018](#bib.bib158 ""))

✓\\checkmark

✓\\checkmark

PFF

([Chang et al., 2020](#bib.bib12 ""))

✓\\checkmark

PMS

([Zhou et al., 2019](#bib.bib170 ""))

✓\\checkmark

HP

([Rivas Rojas et al., 2020](#bib.bib107 ""))

✓\\checkmark

HP

 Representation Learning 

([Subramanian et al., 2018](#bib.bib124 ""))

✓\\checkmark

✓\\checkmark

✓\\checkmark

PFS

([Wang et al., 2020e](#bib.bib137 ""))

✓\\checkmark

✓\\checkmark

✓\\checkmark

✓\\checkmark

PFS

Auxiliary MTL aims to improve the performance of certain primary tasks by introducing auxiliary tasks and is widely used in the NLP field for different types of primary tasks, such as sequence tagging, classification, text generation, and representation learning. Table [1](#S4.T1 "Table 1 ‣ 4.1. Auxiliary MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview") summarizes the types of auxiliary tasks used along with different types of primary tasks. As shown in Table [1](#S4.T1 "Table 1 ‣ 4.1. Auxiliary MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"), auxiliary tasks are usually closely related to primary tasks.

Targeting sequence tagging tasks, [Rei (2017)](#bib.bib105 "") adds a language modeling objective into a sequence labeling model to counter the sparsity of named entities and make full use of training data. [Augenstein and Søgaard (2017)](#bib.bib5 "") add five auxiliary tasks for scientific keyphrase boundary classification, including syntactic chunking, frame target annotation, hyperlink prediction, multi-word expression identification, and semantic super-sense tagging. [Li and Lam (2017)](#bib.bib67 "") use opinion word extraction and sentence-level sentiment identification to assist aspect term extraction. [Isonuma et al. (2017)](#bib.bib54 "") train an extractive summarization model together with an auxiliary document-level classification task. [Xing et al. (2018)](#bib.bib151 "") transfer knowledge from a large open-domain corpus to the data-scarce medical domain for Chinese word segmentation using a parallel MTL architecture. HanPaNE ([Watanabe et al., 2019](#bib.bib142 "")) improves NER for chemical compounds by jointly training a chemical compound paraphrase model. [Xia et al. (2019)](#bib.bib147 "") enhance Chinese semantic role labeling by adding a dependency parsing model and uses the output of dependency parsing as additional features. [Nishida et al. (2019)](#bib.bib90 "") improve the evidence extraction capability of an explainable multi-hop QA model by viewing evidence extraction as an auxiliary summarization task. [Alqahtani et al. (2020)](#bib.bib2 "") improve character-level diacritic restoration with word-level syntactic diacritization, POS tagging, and word segmentation. In ([Cheng et al., 2020](#bib.bib18 "")), the performance of argument mining is improved by the argument pairing task on review and rebuttal pairs of scientific papers. [Le et al. (2020)](#bib.bib64 "") make use of the similarity between word sense disambiguation and metaphor detection to improve the performance of the latter task. To handle the primary disfluency detection task, [Wang et al. (2020a)](#bib.bib136 "") pre-train two self-supervised tasks using constructed pseudo training data before fine-tuning on the primary task.

Researchers have also applied auxiliary MTL to classification tasks, such as explicit ([Liu et al., 2016a](#bib.bib78 "")) and implicit ([Lan et al., 2017](#bib.bib62 "")) discourse relation classification. To improve automatic rumor identification, [Kochkina et al. (2018)](#bib.bib59 "") jointly train on the stance classification and veracity prediction tasks. [Lamprinidis et al. (2018)](#bib.bib61 "") learn a headline popularity prediction model with the help of POS tagging and domain prediction. [Li et al. (2019)](#bib.bib65 "") enhance a rumor detection model with user credibility features. [Farag and Yannakoudakis (2019)](#bib.bib30 "") add a low-level grammatical role prediction task into a discourse coherence assessment model to help improve its performance. [Maddela et al. (2019)](#bib.bib81 "") enhance the hashtag segmentation task by introducing an auxiliary task which predicts whether a given hashtag is single-token or multi-token. In ([Shimura et al., 2019](#bib.bib117 "")), text classification is boosted by learning the predominant sense of words. [Wu et al. (2019)](#bib.bib146 "") assist the fake news detection task by stance classification. [Chen et al. (2019)](#bib.bib17 "") jointly learn the answer identification task with an auxiliary question answering task. To improve slot filling performance for online shopping assistants, [Gong et al. (2019)](#bib.bib36 "") add NER and segment tagging tasks as auxiliary tasks. In ([Song et al., 2020a](#bib.bib122 "")), the organization evaluation for student essays is learned together with the sentence and paragraph discourse element identification tasks. [Li and Caragea (2019)](#bib.bib69 "") model the stance detection task with the help of the sentiment classification and self-supervised stance lexicon tasks. Generative adversarial MTL architectures are used to improve classification tasks as well. Targeting pharmacovigilance mining, [Yadav et al. (2019)](#bib.bib152 "") treat mining on different data sources as different tasks and applies self-supervised adversarial training as an auxiliary task to help the model combat the variation of data sources and produce more generalized features. Differently, [Ren et al. (2020)](#bib.bib106 "") enhance a feature extractor through unsupervised adversarial training with a discriminator that is pre-trained with supervised data. Sentiment classification models can be enhanced by POS tagging and gaze prediction ([Mishra et al., 2018](#bib.bib87 "")), label distribution learning ([Zhang et al., 2018a](#bib.bib164 "")), unsupervised topic modeling ([Wang et al., 2020c](#bib.bib135 "")), or domain adversarial training ([Wang et al., 2018](#bib.bib138 "")). In ([Wu and Huang, 2016](#bib.bib145 "")), besides the shared base model, a separate model is built for each Microblog user as an auxiliary task. [Rawat et al. (2019)](#bib.bib104 "") estimate causality scores via Naranjo questionnaire, consisting of 10 multiple-choice questions, with sentence relevance classification as an auxiliary task. [Liu et al. (2018b)](#bib.bib74 "") introduce an auxiliary task of selecting the passages containing the answers to assist a multi-answer question answering task. [Yang et al. (2019)](#bib.bib153 "") improve a community question answering model with an auxiliary question category classification task. To counter data scarcity in the multi-choice question answering task, [Jin et al. (2020)](#bib.bib55 "") propose a multi-stage MTL model that is first coarsely pre-trained using a large out-of-domain natural language inference dataset and then fine-tuned on an in-domain dataset.

For text generation tasks, MTL is brought in to improve the quality of the generated text. It is observed in ([Domhan and Hieber, 2017](#bib.bib29 "")) that adding a target-side language modeling task on the decoder of a neural machine translation (NMT) model brings moderate but consistent performance gain. [Luong et al. (2016)](#bib.bib80 "") learn a multilingual NMT model with constituency parsing and image caption generation as two auxiliary tasks. Similarly, [Zaremoodi et al. (2018)](#bib.bib158 "") learn an NMT model together with the help of NER, syntactic parsing, and semantic parsing tasks. To make an NMT model aware of the vocabulary distribution of the retrieval corpus for query translation, [Sarwar et al. (2019)](#bib.bib112 "") add an unsupervised auxiliary task that learns continuous bag-of-words embeddings on the retrieval corpus in addition to the sentence-level parallel data. [Wang et al. (2020d)](#bib.bib139 "") build a multilingual NMT system with source-side language modeling and target-side denoising autoencoder. For the sentence simplification task, [Guo et al. (2018a)](#bib.bib40 "") use paraphrase generation and entailment generation as two auxiliary tasks. [Guo et al. (2018b)](#bib.bib41 "") build an abstractive summarization model with the question and entailment generation tasks as auxiliary tasks. By improving a language modeling task through MTL, we can generate more natural and coherent text for question generation ([Zhou et al., 2019](#bib.bib170 "")) or task-oriented dialogue generation ([Zhu et al., 2019](#bib.bib171 "")). [Shao et al. (2019)](#bib.bib114 "") implement a semantic parser that jointly learns question type classification, entity mention detection, as well as a weakly supervised objective via question paraphrasing. [Chang et al. (2020)](#bib.bib12 "") enhance a text-to-SQL semantic parser by adding explicit condition value detection and value-column mapping as auxiliary tasks. [Rivas Rojas et al. (2020)](#bib.bib107 "") view hierarchical text classification, where each text may have several labels on different levels, as a generation task by generating from more general labels to more specific ones, and an auxiliary task of generating in the opposite order is introduced to guide the model to treat high-level and low-level labels more equally and therefore learn more robust representations.

Besides tackling specific tasks, some researchers aim at building general-purpose text representations for future use in downstream tasks. For example, [Subramanian et al. (2018)](#bib.bib124 "") learn sentence representations through multiple weakly related tasks, including learning skip-thought vectors, neural machine translation, constituency parsing, and natural language inference tasks. [Wang et al. (2020e)](#bib.bib137 "") train multi-role dialogue representations via unsupervised multi-task pre-training on reference prediction, word prediction, role prediction, and sentence generation. As existing pre-trained models impose huge storage cost for the deployment, PinText ([Zhuang and Liu, 2019](#bib.bib172 "")) learns user profile representations through learning custom word embeddings, which are obtained by minimizing the distance between positive engagement pairs based on user behaviors, including homefeed, related pins, and search queries, by sharing the embedding lookup table.

### 4.2. Joint MTL

Different from auxiliary MTL, joint MTL models optimize its performance on several tasks simultaneously. Similar to auxiliary MTL, tasks in joint MTL are usually related to or complementary to each other. Table [2](#S4.T2 "Table 2 ‣ 4.2. Joint MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview") gives an overview of task combinations used in joint MTL models. In certain scenarios, we can even convert models following the traditional pipeline architecture as in single-task learning to joint MTL models so that different tasks can adapt to each other. For example, [Perera et al. (2018)](#bib.bib99 "") convert the parsing of Alexa meaning representation language into three independent tagging tasks for intents, types, and properties, respectively. [Song and Park (2019)](#bib.bib120 "") transform the pipeline relation between POS tagging and morphological tagging into a parallel relation and further builds a joint MTL model.

Joint MTL has been proven to be an effective way to improve the performance of standard NLP tasks. For instance, [Hashimoto et al. (2017)](#bib.bib47 "") train six tasks of different levels jointly, including POS tagging, chunking, dependency parsing, relatedness classification, and entailment classification. [Zhang et al. (2017)](#bib.bib162 "") apply parallel feature fusion to learn multiple classification tasks, including sentiment classification on movie and product reviews. Different from traditional pipeline methods, [Luan et al. (2018)](#bib.bib79 "") jointly learn identification and classification of entities, relations, and coreference clusters in scientific literatures. [Sanh et al. (2019)](#bib.bib111 "") optimize four semantic tasks together, including NER, entity mention detection (EMD), coreference resolution (CR), and relation extraction (RE) tasks. [Gupta et al. (2016)](#bib.bib43 ""); [Zeng et al. (2020b)](#bib.bib159 ""); [Ye et al. (2019)](#bib.bib155 "") learn entity extraction alongside relation extraction. For sentiment analysis tasks, [Cerisara et al. (2018)](#bib.bib11 "") jointly learn dialogue act and sentiment recognition using the parallel feature sharing MTL architecture. [He et al. (2019)](#bib.bib48 "") learn the aspect term extraction and aspect sentiment classification tasks jointly to facilitate aspect-based sentiment analysis. [Zhao et al. (2020)](#bib.bib166 "") build a joint aspect term, opinion term, and aspect-opinion pair extraction model through MTL and shows that the joint model outperforms single-task and pipeline baselines by a large margin.

Besides well-studied NLP tasks, joint MTL is also widely applied in various downstream tasks. One major problem of such tasks is the lack of sufficient labeled data. Through joint MTL, one could take advantage of data-rich domains via implicit knowledge sharing. In addition, abundant unlabeled data could be utilized via unsupervised learning techniques. [Zhao et al. (2019)](#bib.bib167 "") develop a joint MTL model for the NER and entity name normalization tasks in the medical field. [Liu et al. (2018a)](#bib.bib75 ""); [Zeng et al. (2020a)](#bib.bib160 "") use MTL to perform simile detection, which includes simile sentence classification and simile component extraction. To analyze Twitter demographic data, [Vijayaraghavan et al. (2017)](#bib.bib131 "") jointly learn classification models for genders, ages, political orientations, and locations. The SLUICE network ([Ruder et al., 2019](#bib.bib108 "")) is used to learn four different non-literal language detection tasks in English and German ([Do Dinh et al., 2018](#bib.bib28 "")). [Niu et al. (2018)](#bib.bib92 "") jointly train a monolingual formality transfer model and a formality sensitive machine translation model between English and French. For community question answering, [Joty et al. (2018)](#bib.bib56 "") build an MTL model that extracts existing questions related to the current one and looks for question-comment threads that could answer the question at the same time. To analyze the argumentative structure of scientific publications, [Lauscher et al. (2018)](#bib.bib63 "") optimize argumentative component identification, discourse role classification, citation context identification, subjective aspect classification, and summary relevance classification together with a dynamic weighting mechanism. Considering the connection between sentence emotions and the use of the metaphor, [Dankers et al. (2019)](#bib.bib24 "") jointly train a metaphor identification model with an emotion detection model. To ensure the consistency between generated key phrases (short text) and headlines (long text), [Nishino et al. (2019)](#bib.bib91 "") train the two generative models jointly with a document category classification model and adds a hierarchical consistency loss based on the attention mechanism. An MTL model is proposed in ([Song et al., 2020b](#bib.bib121 "")) to jointly perform zero pronoun detection, recovery, and resolution, and unlike previous works, it does not require external syntactic parsing tools.

Table 2. A summary of joint MTL studies according to types of tasks involved. ‘W’, ‘S’, ‘D’, and ‘O’ in the four rightmost columns represent the word-level, sentence-level, and document-level tasks, and tasks of other abstract levels such as RE, respectively. A single checkmark could mean joint learning of multiple tasks of the same type. The ‘Architecture’ column denotes the architecture used, where PFS denotes Parallel Feature Sharing, PFF denotes Parallel Feature Fusion, PMS denotes Parallel Multi-level Supervision, HFF denotes Hierarchical Feature Fusion, HP denotes Hierarchical Pipeline, and HIM denotes Hierarchical Interactive MTL.

| Reference    |              |                |                |                |     |
| ------------ | ------------ | -------------- | -------------- | -------------- | --- |
| Tagging      | Generation   | Classification | Classification | Classification |     |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | PFS |
| ✓\\checkmark |              |                |                |                | PFS |
|              | ✓\\checkmark |                |                |                | PFS |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | PFS |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | PFS |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | PFS |
| ✓\\checkmark |              |                |                |                | PFS |
|              |              |                | ✓\\checkmark   |                | PFS |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | PFS |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | PFF |
|              |              | ✓\\checkmark   | ✓\\checkmark   |                | PFF |
|              | ✓\\checkmark |                | ✓\\checkmark   |                | PMS |
| ✓\\checkmark |              |                |                |                | PMS |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | PMS |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | PMS |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | PMS |
|              |              |                | ✓\\checkmark   |                | HFF |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | HP  |
| ✓\\checkmark |              |                |                |                | HP  |
| ✓\\checkmark |              |                |                | ✓\\checkmark   | HP  |
| ✓\\checkmark |              | ✓\\checkmark   | ✓\\checkmark   |                | HP  |
| ✓\\checkmark |              |                |                |                | HP  |
| ✓\\checkmark |              | ✓\\checkmark   |                |                | HIM |

Moreover, joint MTL is suitable for multi-domain or multi-formalism NLP tasks. Multi-domain tasks share the same problem definition and label space among tasks, but have different data distributions. Applications in multi-domain NLP tasks include sentiment classification ([Li and Zong, 2008](#bib.bib66 ""); [Wu and Huang, 2015](#bib.bib144 "")), dialog state tracking ([Mrkšić et al., 2015](#bib.bib89 "")), essay scoring ([Cummins et al., 2016](#bib.bib23 "")), deceptive review detection ([Hai et al., 2016](#bib.bib45 "")), multi-genre emotion detection and classification ([Tafreshi and Diab, 2018](#bib.bib126 "")), RST discourse parsing ([Braud et al., 2016](#bib.bib8 "")), historical spelling normalization ([Bollmann and Søgaard, 2016](#bib.bib7 "")), and document classification ([Tian et al., 2019](#bib.bib128 "")). Multi-formalism tasks have the same problem definition but may have different while structurally similar label spaces. [Peng et al. (2017)](#bib.bib97 ""); [Kurita and Søgaard (2019)](#bib.bib60 "") model three different formalisms of semantic dependency parsing (i.e., DELPH-IN MRS (DM) ([Flickinger et al., 2012](#bib.bib34 "")), Predicate-Argument Structures (PAS) ([Marcus et al., 1994](#bib.bib83 "")), and Prague Semantic Dependencies (PSD) ([Hajic et al., 2012](#bib.bib46 ""))) jointly. In ([Hershcovich et al., 2018](#bib.bib51 "")), a transition-based semantic parsing system is trained jointly on different parsing tasks, including Abstract Meaning Representation (AMR) ([Banarescu et al., 2013](#bib.bib6 "")), Semantic Dependency Parsing (SDP) ([Oepen et al., 2016](#bib.bib94 "")), and Universal Dependencies (UD) ([Nivre et al., 2016](#bib.bib93 "")), and it shows that joint training improves performance on the testing UCCA dataset. [Liu et al. (2016a)](#bib.bib78 "") jointly model discourse relation classification on two distinct datasets: PDTB and RST-DT. [Fares et al. (2018)](#bib.bib31 "") show the dual annotation and joint learning of two distinct sets of relations for noun-noun compounds could improve the performance of both tasks. In ([Zalmout and Habash, 2019](#bib.bib157 "")), an adversarial MTL model is proposed for morphological modeling for high-resource modern standard Arabic and its low-resource dialect Egyptian Arabic, to enable knowledge between the two domains.

### 4.3. Multilingual and Multimodal Tasks

Multilingual machine learning has always been a hot topic in the NLP field with a representative example of NMT systems mentioned in Section [4.1](#S4.SS1 "4.1. Auxiliary MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"). Since monolingual data source may be limited and biased, leveraging data from multiple languages through MTL can benefit multilingual machine learning models, such as language intent learning in Japanese and English ([Masumura et al., 2018b](#bib.bib86 "")) and sentiment classification in Chinese and English ([Wang et al., 2018](#bib.bib138 "")). Another use of MTL is cross-lingual knowledge transfer, where knowledge learned in one language can be used in tasks in another language. For example, ([Niu et al., 2018](#bib.bib92 "")) develops a formality-sensitive translation system from English to French where formality labels are only available in English. Besides, effort has also been made to learn unified cross-lingual language representations ([Singla et al., 2018](#bib.bib118 ""); [Huang et al., 2019](#bib.bib53 "")). Such cross-lingual representations could substantially boost performance under low-resource settings ([Lin et al., 2018](#bib.bib72 "")).

One step further from multilingual learning, multimodal learning has attracted an increasing interest in recent years. Researchers have incorporated features from multiple modalities, such as auditory and visual features, to text-related cross-modal tasks. To this end, MTL is a natural choice for learning generalized multimodal features by shaping a shared cross-modal feature space. One example is end-to-end speech translation ([Chuang et al., 2020](#bib.bib19 "")) where speech recognition and text translation are learned jointly. Similarly for video captioning ([Pasunuru and Bansal, 2017](#bib.bib95 "")), the video prediction task and text entailment generation task are used to enhance the encoder and decoder of the model, respectively. A multimodal representation space also makes it possible to build natural language interfaces to different systems. One example is semantic navigation ([Chaplot et al., 2020](#bib.bib14 "")), where an agent acts according to navigation commands in a 3-D environment. The key is learning a one-to-one mapping, also known as knowledge grounding, between visual feature maps and text tokens via joint learning of object detection and visual question answering tasks. A multi-task evaluation framework ([Suglia et al., 2020](#bib.bib125 "")) is proposed to evaluate knowledge grounding of such vision-language models.

### 4.4. Task Relatedness in MTL

A key issue that affects the performance of MTL is how to properly choose a set of tasks for joint training. Generally, tasks that are similar and complementary to each other are suitable for multi-task learning, and there are some works that studies this issue for NLP tasks. For semantic sequence labeling tasks, [Martínez Alonso and Plank (2017)](#bib.bib84 "") report that MTL works best when the label distribution of auxiliary tasks has low kurtosis and high entropy. This finding also holds for rumor verification ([Kochkina et al., 2018](#bib.bib59 "")). Similarly, [Liu et al. (2016a)](#bib.bib78 "") report that tasks with major differences, such as implicit and explicit discourse classification, may not benefit much from each other. To quantitatively estimate the likelihood of two tasks benefiting from joint training, [Schröder and Biemann (2020)](#bib.bib113 "") propose a dataset similarity metric which considers both tokens and their labels. The proposed metric is based on the normalized mutual information of the confusion matrix between label clusters of two datasets. Such similarity metrics could help identify helpful tasks and improve the performance of MTL models that are empirically hard to achieve through manual selection.

As MTL assumes certain relatedness and complementarity between the chosen tasks, the performance gain brought by MTL can in turn reveal the strength of such relatedness. [Changpinyo et al. (2018)](#bib.bib13 "") study the pairwise impact of joint training among 11 tasks under 3 different MTL schemes and show that MTL on a set of properly selected tasks outperforms MTL on all tasks. The harmful tasks either are totally unrelated to other tasks or possess a small dataset that is prone to overfitting. For dependency parsing problems, [Peng et al. (2017)](#bib.bib97 ""); [Kurita and Søgaard (2019)](#bib.bib60 "") claim that MTL works best for formalisms that are more similar. [Dankers et al. (2019)](#bib.bib24 "") model the interplay of the metaphor and emotion via MTL and reports that metaphorical features are beneficial to sentiment analysis tasks. Unicoder ([Huang et al., 2019](#bib.bib53 "")) presents results of jointly fine-tuning on different sets of languages as well as pairwise cross-language transfer among 15 languages, and finds that knowledge transfer between English, Spanish, and French is easier than other combinations of languages.

## 5\. Data Source and Benchmarks for Multi-task Learning

In this section, we introduce the ways of preparing datasets for training MTL models and some benchmark datasets.

### 5.1. Data Source

Given MM tasks with corresponding datasets 𝒟t\={𝐗t,𝐘t},t\=1,…,M\\mathcal{D}\_{t}=\\{\\mathbf{X}\_{t},\\mathbf{Y}\_{t}\\},t=1,\\ldots,M, where 𝐗t\\mathbf{X}\_{t} denotes the set of data instances in task tt and 𝐘t\\mathbf{Y}\_{t} denotes the corresponding labels, we denote the entire dataset for the MM tasks by 𝒟\={𝐗,𝐘}\\mathcal{D}=\\{\\mathbf{X},\\mathbf{Y}\\}. We describe different forms of 𝒟\\mathcal{D} in the following sections.

#### 5.1.1. Disjoint Datasets

In most multi-task learning literature, the datasets of different tasks have distinct label spaces, i.e. ∀i≠j,𝐘i∩𝐘j\=∅\\forall i\\neq j,\\;\\mathbf{Y}\_{i}\\cap\\mathbf{Y}\_{j}=\\emptyset. In this case, 𝒟\={𝒟1,…,𝒟M}\\mathcal{D}=\\{\\mathcal{D}\_{1},\\dots,\\mathcal{D}\_{M}\\}. The most popular way to train MTL models on such tasks is to alternate between different tasks ([Collobert and Weston, 2008](#bib.bib22 ""); [Luong et al., 2016](#bib.bib80 ""); [Bollmann and Søgaard, 2016](#bib.bib7 ""); [Søgaard and Goldberg, 2016](#bib.bib119 ""); [Gupta et al., 2016](#bib.bib43 ""); [Liu et al., 2016b](#bib.bib76 ""); [Liu et al., 2016a](#bib.bib78 ""); [Pasunuru and Bansal, 2017](#bib.bib95 ""); [Domhan and Hieber, 2017](#bib.bib29 ""); [Hashimoto et al., 2017](#bib.bib47 ""); [Masumura et al., 2018b](#bib.bib86 ""); [Xiao et al., 2018b](#bib.bib149 ""); [Zheng et al., 2018](#bib.bib169 ""); [Fei et al., 2019](#bib.bib32 ""); [Rawat et al., 2019](#bib.bib104 "")), either randomly or by a schedule, as previously discussed in Section [3](#S3 "3. Optimization for MTL Models ‣ Multi-Task Learning in Natural Language Processing: An Overview").

#### 5.1.2. Multi-label Datasets

Instances in multi-label datasets share one feature space for all tasks, i.e. ∀i≠j,𝐗\=𝐗i\=𝐗j\\forall i\\neq j,\\;\\mathbf{X}=\\mathbf{X}\_{i}=\\mathbf{X}\_{j}, which makes it possible to optimize all task-specific components at the same time. In this case, 𝒟\={𝐗,𝐘^}\\mathcal{D}=\\{\\mathbf{X},\\hat{\\mathbf{Y}}\\} where 𝐘^\=∪i\=1M𝐘i\\hat{\\mathbf{Y}}=\\cup\_{i=1}^{M}\\mathbf{Y}\_{i}.

Multi-label datasets can be created by giving extra annotations to existing data. For example, [Peng et al. (2017)](#bib.bib97 ""); [Kurita and Søgaard (2019)](#bib.bib60 "") annotate dependency parse trees of three different formalisms for each text input. [Vijayaraghavan et al. (2017)](#bib.bib131 "") label Twitter posts with 4 demographic labels. [Fares et al. (2018)](#bib.bib31 "") annotate two distinct sets of relations over the same set of underlying chemical compounds.

The extra annotations can be created automatically as well, resulting in a self-supervised multi-label dataset. Extra labels can be obtained using pre-defined rules ([Rei, 2017](#bib.bib105 ""); [Li and Caragea, 2019](#bib.bib69 "")). In ([Lan et al., 2017](#bib.bib62 "")), to synthesize unlabeled dataset for the auxiliary unsupervised implicit discourse classification task, explicit discourse connectives (e.g., because, but, etc.) are removed from a large corpus and used as implicit relation labels. [Niu et al. (2018)](#bib.bib92 "") combine an English corpus with formality labels and an unlabeled English-French parallel corpus by random selection and concatenation to facilitate the joint training of formality style transfer and formality-sensitive translation. [Tafreshi and Diab (2018)](#bib.bib126 "") use hashtags to represent genres of tweet posts. [Watanabe et al. (2019)](#bib.bib142 "") generate sentence pairs by replacing chemical named entities with their paraphrases in the PubChemDic database. Unicoder ([Huang et al., 2019](#bib.bib53 "")) uses translated text from the source language to fine-tune on the target language. [Wang et al. (2020a)](#bib.bib136 "") create disfluent sentences by randomly repeating or inserting nn-grams. Besides annotating in the aforementioned ways, some researchers create self-supervised labels with the help of external tools or previously trained models. [Shimura et al. (2019)](#bib.bib117 "") obtain dominant word sense labels from WordNet ([Fellbaum, 2010](#bib.bib33 "")). [Deng et al. (2019)](#bib.bib26 "") apply entity linking for QA data over databases through an entity linker. [Gong et al. (2019)](#bib.bib36 "") assign NER and segmentation labels for three tasks using an unsupervised dynamic programming method. [Lim et al. (2020)](#bib.bib71 "") use the output of a meta-network as labels for unsupervised training data. As a special case of multi-label dataset, mask orchestration ([Wang et al., 2020e](#bib.bib137 "")) provides different parts of an instance to different tasks by applying different masks. That is, labels for one task may become the input for another task.

### 5.2. Multi-task Benchmark Datasets

Table 3. Statistics of multi-task benchmark datasets for NLP tasks.

| Dataset | \# Tasks | \# Languages                         | \# Samples | Topic |
| ------- | -------- | ------------------------------------ | ---------- | ----- |
| 1 (en)  | 2157k    | Language Understanding               |            |       |
| 1 (en)  | 160k     | Language Understanding               |            |       |
| 1 (en)  | -        | Language Understanding               |            |       |
| 40      | 597k     | Multilingual Learning                |            |       |
| 100     | 2747G    | Cross-lingual Pre-training           |            |       |
| 1 (en)  | 51k      | Semantic Parsing                     |            |       |
| 1 (cn)  | 28k      | Language Processing                  |            |       |
| 1 (en)  | 5k       | Anti-reflexive Gender Bias Detection |            |       |
| 1 (en)  | 66k      | Grounded Language Learning           |            |       |
| 1 (en)  | 500      | Scientific Literature Understanding  |            |       |

As summarized in Table [3](#S5.T3 "Table 3 ‣ 5.2. Multi-task Benchmark Datasets ‣ 5. Data Source and Benchmarks for Multi-task Learning ‣ Multi-Task Learning in Natural Language Processing: An Overview"), we list a few public multi-task benchmark datasets for NLP tasks.

*   •

```
GLUE ([Wang et al., 2019b](#bib.bib134 "")) is a benchmark dataset for evaluating natural language understanding (NLU) models. The main benchmark consists of 8 sentence and sentence-pair classification tasks as well as a regression task. The tasks cover a diverse range of genres, dataset sizes, and difficulties. Besides, a diagnostic dataset is provided to evaluate the ability of NLU models on capturing a pre-defined set of language phenomena.
```
*   •

```
SuperGLUE ([Wang et al., 2019a](#bib.bib133 "")) is a generalization of GLUE. As the performance of state-of-the-art models has exceeded non-expert human baselines on GLUE, SuperGLUE contains a set of 8 more challenging NLU tasks along with comprehensive human baselines. Besides retaining the two hardest tasks in GLUE, 6 tasks are added with two new question formats: coreference resolution and question answering (QA).
```
*   •

```
Measuring Massive Multitask Language Understanding (MMMLU) ([Hendrycks et al., 2020](#bib.bib50 "")) is a multi-task few-shot learning dataset for world knowledge and problem solving abilities of language processing models. This dataset covers 57 subjects including 19 in STEM, 13 in humanities, 12 in social sciences, and 13 in other subjects. This dataset is split into a few-shot development set that has 5 questions for each subject, a validation set for tuning hyper-parameters containing 1540 questions, and a test set with 14079 questions.
```
*   •

```
Xtreme ([Hu et al., 2020](#bib.bib52 "")) is a multi-task benchmark dataset for evaluating cross-lingual generalization capabilities of multilingual representations covering 9 tasks in 40 languages. The tasks include 2 classification tasks, 2 structure prediction tasks, 3 question answering tasks, and 2 sentence retrieval tasks. Out of the 40 languages involved, 19 languages appear in at least 3 datasets and the rest 21 languages appear in at least one dataset.
```
*   •

```
XGLUE ([Liang et al., 2020](#bib.bib70 "")) is a benchmark dataset that supports the development and evaluation of large cross-lingual pre-trained language models. The XGLUE dataset includes 11 downstream tasks, including 3 single-input understanding tasks, 6 pair-input understanding tasks, and 2 generation tasks. The pre-training corpus consists of a small corpus that includes a 101G multilingual corpus covering 100 languages and a 146G bilingual corpus covering 27 languages, and a large corpus with 2,500G multilingual data covering 89 languages.
```
*   •

```
LSParD ([Shao et al., 2019](#bib.bib114 "")) is a multi-task semantic parsing dataset with 3 tasks, including question type classification, entity mention detection, and question semantic parsing. Each logical form is associated with a question and multiple human annotated paraphrases. This dataset contains 51,164 questions in 9 categories, 3361 logical form patterns, and 23,144 entities.
```
*   •

```
ECSA ([Gong et al., 2019](#bib.bib36 "")) is a dataset for slot filling, named entity recognition, and segmentation to evaluate online shopping assistant systems in Chinese. The training part contains 24,892 pairs of input utterances and their corresponding slot labels, named entity labels, and segment labels. The testing part includes 2,723 such pairs with an Out-of-Vocabulary (OOV) rate of 85.3%, which is much higher than the ATIS dataset ([Hemphill et al., 1990](#bib.bib49 "")) whose OOV rate is smaller than 1%.
```
*   •

```
ABC ([Gonzalez et al., 2020](#bib.bib37 "")), the Anti-reflexive Bias Challenge, is a multi-task benchmark dataset designed for evaluating gender assumptions in NLP models. ABC consists of 4 tasks, including language modeling, natural language inference (NLI), coreference resolution, and machine translation. A total of 4,560 samples are collected by a template-based method. The language modeling task is to predict the pronoun of a sentence. For NLI and coreference resolution, three variations of each sentence are used to construct entailment pairs. For machine translation, sentences with two variations of third-person pronouns in English are used as source sentences.
```
*   •

```
CompGuessWhat?! ([Suglia et al., 2020](#bib.bib125 "")) is a dataset for grounded language learning with 65,700 collected dialogues. It is an instance of the Grounded Language Learning with Attributes (GROLLA) framework. The evaluation process includes three parts: goal-oriented evaluation (e.g., Visual QA and Visual NLI), object attribute prediction, and zero-shot evaluation.
```
*   •

```
SCIERC ([Luan et al., 2018](#bib.bib79 "")) is a multi-label dataset for identifying entities, relations, and cross-sentence coreference clusters from abstracts of research papers. SCIERC contains 500 scientific abstracts collected from proceedings in 12 conferences and workshops in artificial intelligence.
```
## 6\. Conclusion and Discussions

In this paper, we give an overview of the application of multi-task learning in recent natural language processing research, focusing on deep learning approaches. We first present different architectures of MTL used in recent research literature, including parallel architecture, hierarchical architecture, modular architecture, and generative adversarial architectures. After that, optimization techniques, including loss construction, data sampling, and task scheduling are discussed. After briefly summarizing the application of MTL in different down-stream tasks, we describe the ways to manage data sources in MTL as well as some MTL benchmark datasets for NLP research.

There are several directions worth further investigations for future studies. Firstly, given multiple NLP tasks, how to find a set of tasks that could take advantage of MTL remains a challenge. Besides improving performance of MTL models, a deeper understanding of task relatedness could also help expanding the application of MTL to more tasks. Though there are some works studying this issue, as discussed in Section [4.4](#S4.SS4 "4.4. Task Relatedness in MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"), they are far from mature.

Secondly, current NLP models often rely on a large or even huge amount of labeled data. However, in many real-world applications, where large-scale data annotation is costly, this requirement cannot be easily satisfied. In this case, we may consider to leverage abundant unlabeled data in MTL by using self-supervised or unsupervised learning techniques.

Thirdly, we are curious about whether we can create more powerful Pre-trained Language Models (PLMs) via more advanced MTL techniques. PLMs have become an essential part of NLP pipeline. Though most PLMs are trained on multiple tasks, the MTL architectures used are mostly simple feature sharing architectures. A better MTL architecture might be the key for the next breakthrough for PLMs.

At last, it would be interesting to extend the use of MTL to more NLP tasks. Though there are many NLP tasks that can be jointly learned by MTL, most NLP tasks are well-studied tasks, such as classification, sequence labeling, and text generation, as shown in Tables [1](#S4.T1 "Table 1 ‣ 4.1. Auxiliary MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview") and [2](#S4.T2 "Table 2 ‣ 4.2. Joint MTL ‣ 4. Application in NLP Tasks ‣ Multi-Task Learning in Natural Language Processing: An Overview"). We would like to see how MTL could benefit more challenging NLP tasks, such as building dialogue systems and multi-modal learning tasks.

## Acknowledgements

This work is supported by NSFC key grant under grant no. 62136005, NSFC general grant under grant no. 62076118, and Shenzhen fundamental research program JCYJ20210324105000003.

## References

*   Alqahtani et al. (2020) Sawsan Alqahtani, Ajay Mishra, and Mona Diab. 2020. A Multitask Learning Approach for Diacritic Restoration. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 8238–8247.
*   Aminian et al. (2020) Maryam Aminian, Mohammad Sadegh Rasooli, and Mona Diab. 2020. Mutlitask Learning for Cross-Lingual Transfer of Semantic Dependencies. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. arXiv:2004.14961
*   Asai et al. (2022) Akari Asai, Mohammadreza Salehi, Matthew Peters, and Hannaneh Hajishirzi. 2022. ATTEMPT: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 6655–6672. [https://doi.org/10.18653/v1/2022.emnlp-main.446](https://doi.org/10.18653/v1/2022.emnlp-main.446 "")
*   Augenstein and Søgaard (2017) Isabelle Augenstein and Anders Søgaard. 2017. Multi-Task Learning of Keyphrase Boundary Classification. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 341–346.
*   Banarescu et al. (2013) Laura Banarescu, Claire Bonial, Shu Cai, Madalina Georgescu, Kira Griffitt, Ulf Hermjakob, Kevin Knight, Philipp Koehn, Martha Palmer, and Nathan Schneider. 2013. Abstract Meaning Representation for Sembanking. In *Proceedings of the 7th Linguistic Annotation Workshop and Interoperability with Discourse*. Association for Computational Linguistics, 178–186.
*   Bollmann and Søgaard (2016) Marcel Bollmann and Anders Søgaard. 2016. Improving Historical Spelling Normalization with Bi-Directional LSTMs and Multi-Task Learning. In *Proceedings of the 26th International Conference on Computational Linguistics*. The COLING 2016 Organizing Committee, 131–139.
*   Braud et al. (2016) Chloé Braud, Barbara Plank, and Anders Søgaard. 2016. Multi-View and Multi-Task Training of RST Discourse Parsers. In *Proceedings of the 26th International Conference on Computational Linguistics*. 1903–1913.
*   Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In *Advances in Neural Information Processing Systems*, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 1877–1901. [https://proceedings.neurips.cc/paper\_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf "")
*   Caruana (1997) Rich Caruana. 1997. Multitask Learning. *Machine Learning* 28, 1 (1997), 41–75.
*   Cerisara et al. (2018) Christophe Cerisara, Somayeh Jafaritazehjani, Adedayo Oluokun, and Hoa T. Le. 2018. Multi-Task Dialog Act and Sentiment Recognition on Mastodon. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 745–754.
*   Chang et al. (2020) Shuaichen Chang, Pengfei Liu, Yun Tang, Jing Huang, Xiaodong He, and Bowen Zhou. 2020. Zero-Shot Text-to-SQL Learning with Auxiliary Task. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 7488–7495.
*   Changpinyo et al. (2018) Soravit Changpinyo, Hexiang Hu, and Fei Sha. 2018. Multi-Task Learning for Sequence Tagging: An Empirical Study. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 2965–2977.
*   Chaplot et al. (2020) Devendra Singh Chaplot, Lisa Lee, Ruslan Salakhutdinov, Devi Parikh, and Dhruv Batra. 2020. Embodied Multimodal Multitask Learning. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 2442–2448.
*   Chauhan et al. (2020) Dushyant Singh Chauhan, Dhanush S R, Asif Ekbal, and Pushpak Bhattacharyya. 2020. Sentiment and Emotion Help Sarcasm? A Multi-Task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 4351–4360.
*   Chen et al. (2018) Junkun Chen, Xipeng Qiu, Pengfei Liu, and Xuanjing Huang. 2018. Meta Multi-Task Learning for Sequence Modeling. In *Proceedings of the AAAI Conference on Artificial Intelligence*.
*   Chen et al. (2019) Long Chen, Ziyu Guan, Wei Zhao, Wanqing Zhao, Xiaopeng Wang, Zhou Zhao, and Huan Sun. 2019. Answer Identification from Product Reviews for User Questions by Multi-Task Attentive Networks. *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 2019), 45–52.
*   Cheng et al. (2020) Liying Cheng, Lidong Bing, Qian Yu, Wei Lu, and Luo Si. 2020. APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-Task Learning. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics, 7000–7011.
*   Chuang et al. (2020) Shun-Po Chuang, Tzu-Wei Sung, Alexander H. Liu, and Hung-yi Lee. 2020. Worse WER, but Better BLEU? Leveraging Word Embedding as Intermediate in Multitask End-to-End Speech Translation. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 5998–6003.
*   Cipolla et al. (2018) Roberto Cipolla, Yarin Gal, and Alex Kendall. 2018. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*. IEEE, 7482–7491.
*   Clark et al. (2019) Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D. Manning, and Quoc V. Le. 2019. BAM! Born-Again Multi-Task Networks for Natural Language Understanding. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 5931–5937.
*   Collobert and Weston (2008) Ronan Collobert and Jason Weston. 2008. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. In *Proceedings of the 25th International Conference on Machine Learning* *(ICML ’08)*. Association for Computing Machinery, 160–167.
*   Cummins et al. (2016) Ronan Cummins, Meng Zhang, and Ted Briscoe. 2016. Constrained Multi-Task Learning for Automated Essay Scoring. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 789–799.
*   Dankers et al. (2019) Verna Dankers, Marek Rei, Martha Lewis, and Ekaterina Shutova. 2019. Modelling the Interplay of Metaphor and Emotion through Multitask Learning. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 2218–2229.
*   de Souza et al. (2015) José G. C. de Souza, Matteo Negri, Elisa Ricci, and Marco Turchi. 2015. Online Multitask Learning for Machine Translation Quality Estimation. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*. Association for Computational Linguistics, 219–228.
*   Deng et al. (2019) Yang Deng, Yuexiang Xie, Yaliang Li, Min Yang, Nan Du, Wei Fan, Kai Lei, and Ying Shen. 2019. Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering. *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 2019), 6318–6325.
*   Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Association for Computational Linguistics, 4171–4186.
*   Do Dinh et al. (2018) Erik-Lân Do Dinh, Steffen Eger, and Iryna Gurevych. 2018. Killing Four Birds with Two Stones: Multi-Task Learning for Non-Literal Language Detection. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 1558–1569.
*   Domhan and Hieber (2017) Tobias Domhan and Felix Hieber. 2017. Using Target-Side Monolingual Data for Neural Machine Translation through Multi-Task Learning. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1500–1505.
*   Farag and Yannakoudakis (2019) Youmna Farag and Helen Yannakoudakis. 2019. Multi-Task Learning for Coherence Modeling. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 629–639.
*   Fares et al. (2018) Murhaf Fares, Stephan Oepen, and Erik Velldal. 2018. Transfer and Multi-Task Learning for Noun–Noun Compound Interpretation. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1488–1498.
*   Fei et al. (2019) Hongliang Fei, Shulong Tan, and Ping Li. 2019. Hierarchical Multi-Task Word Embedding Learning for Synonym Prediction. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 834–842.
*   Fellbaum (2010) Christiane Fellbaum. 2010. WordNet. In *Theory and Applications of Ontology: Computer Applications*. Springer, 231–243.
*   Flickinger et al. (2012) Dan Flickinger, Yi Zhang, and Valia Kordoni. 2012. DeepBank. A Dynamically Annotated Treebank of the Wall Street Journal. In *Proceedings of the 11th International Workshop on Treebanks and Linguistic Theories*. 85–96.
*   Gao et al. (2022) Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen. 2022. Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models. In *Proceedings of the 29th International Conference on Computational Linguistics*. International Committee on Computational Linguistics, Gyeongju, Republic of Korea, 3263–3273. [https://aclanthology.org/2022.coling-1.288](https://aclanthology.org/2022.coling-1.288 "")
*   Gong et al. (2019) Yu Gong, Xusheng Luo, Yu Zhu, Wenwu Ou, Zhao Li, Muhua Zhu, Kenny Q. Zhu, Lu Duan, and Xi Chen. 2019. Deep Cascade Multi-Task Learning for Slot Filling in Online Shopping Assistant. *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 2019), 6465–6472.
*   Gonzalez et al. (2020) Ana Valeria Gonzalez, Maria Barrett, Rasmus Hvingelby, Kellie Webster, and Anders Søgaard. 2020. Type B Reflexivization as an Unambiguous Testbed for Multilingual Multi-Task Gender Bias. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. arXiv:2009.11982
*   Goodfellow et al. (2014) Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In *Advances in Neural Information Processing Systems*, Vol. 27. Curran Associates, Inc.
*   Gottumukkala et al. (2020) Ananth Gottumukkala, Dheeru Dua, Sameer Singh, and Matt Gardner. 2020. Dynamic Sampling Strategies for Multi-Task Reading Comprehension. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 920–924.
*   Guo et al. (2018a) Han Guo, Ramakanth Pasunuru, and Mohit Bansal. 2018a. Dynamic Multi-Level Multi-Task Learning for Sentence Simplification. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 462–476.
*   Guo et al. (2018b) Han Guo, Ramakanth Pasunuru, and Mohit Bansal. 2018b. Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 687–697.
*   Gupta et al. (2019) Divam Gupta, Tanmoy Chakraborty, and Soumen Chakrabarti. 2019. GIRNet: Interleaved Multi-Task Recurrent State Sequence Models. *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 2019), 6497–6504.
*   Gupta et al. (2016) Pankaj Gupta, Hinrich Schütze, and Bernt Andrassy. 2016. Table Filling Multi-Task Recurrent Neural Network for Joint Entity and Relation Extraction. In *Proceedings of the 26th International Conference on Computational Linguistics*. The COLING 2016 Organizing Committee, 2537–2547.
*   Gupta et al. (2022) Shashank Gupta, Subhabrata Mukherjee, Krishan Subudhi, Eduardo Gonzalez, Damien Jose, Ahmed H. Awadallah, and Jianfeng Gao. 2022. Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners. arXiv:2204.07689 \[cs.LG\]
*   Hai et al. (2016) Zhen Hai, Peilin Zhao, Peng Cheng, Peng Yang, Xiao-Li Li, and Guangxia Li. 2016. Deceptive Review Spam Detection via Exploiting Task Relatedness and Unlabeled Data. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1817–1826.
*   Hajic et al. (2012) Jan Hajic, Eva Hajicová, Jarmila Panevová, Petr Sgall, Ondrej Bojar, Silvie Cinková, Eva Fucíková, Marie Mikulová, Petr Pajas, Jan Popelka, et al. 2012. Announcing Prague Czech-English Dependency Treebank 2.0.. In *LREC*. 3153–3160.
*   Hashimoto et al. (2017) Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. 2017. A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1923–1933.
*   He et al. (2019) Ruidan He, Wee Sun Lee, Hwee Tou Ng, and Daniel Dahlmeier. 2019. An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 504–515.
*   Hemphill et al. (1990) Charles T. Hemphill, John J. Godfrey, and George R. Doddington. 1990. The ATIS Spoken Language Systems Pilot Corpus. In *Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27,1990*.
*   Hendrycks et al. (2020) Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring Massive Multitask Language Understanding. In *International Conference on Learning Representations*.
*   Hershcovich et al. (2018) Daniel Hershcovich, Omri Abend, and Ari Rappoport. 2018. Multitask Parsing Across Semantic Representations. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 373–385.
*   Hu et al. (2020) Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A Massively Multilingual Multi-Task Benchmark for Evaluating Cross-Lingual Generalization. *In Proceedings of the 37th International Conference on Machine Learning (ICML). July 2020* (July 2020). arXiv:2003.11080
*   Huang et al. (2019) Haoyang Huang, Yaobo Liang, Nan Duan, Ming Gong, Linjun Shou, Daxin Jiang, and Ming Zhou. 2019. Unicoder: A Universal Language Encoder by Pre-Training with Multiple Cross-Lingual Tasks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 2485–2494.
*   Isonuma et al. (2017) Masaru Isonuma, Toru Fujino, Junichiro Mori, Yutaka Matsuo, and Ichiro Sakata. 2017. Extractive Summarization Using Multi-Task Learning with Document Classification. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 2101–2110.
*   Jin et al. (2020) Di Jin, Shuyang Gao, Jiun-Yu Kao, Tagyoung Chung, and Dilek Hakkani-tur. 2020. MMM: Multi-Stage Multi-Task Learning for Multi-Choice Reading Comprehension. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 8010–8017.
*   Joty et al. (2018) Shafiq Joty, Lluís Màrquez, and Preslav Nakov. 2018. Joint Multitask Learning for Community Question Answering Using Task-Specific Embeddings. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 4196–4207.
*   Karimi Mahabadi et al. (2021) Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, and James Henderson. 2021. Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks. In *Annual Meeting of the Association for Computational Linguistics*.
*   Kim et al. (2021) Young Jin Kim, Ammar Ahmad Awan, Alexandre Muzio, Andres Felipe Cruz Salinas, Liyang Lu, Amr Hendy, Samyam Rajbhandari, Yuxiong He, and Hany Hassan Awadalla. 2021. Scalable and Efficient MoE Training for Multitask Multilingual Models. arXiv:2109.10465 \[cs.CL\]
*   Kochkina et al. (2018) Elena Kochkina, Maria Liakata, and Arkaitz Zubiaga. 2018. All-in-One: Multi-Task Learning for Rumour Verification. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 3402–3413.
*   Kurita and Søgaard (2019) Shuhei Kurita and Anders Søgaard. 2019. Multi-Task Semantic Dependency Parsing with Policy Gradient for Learning Easy-First Strategies. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2420–2430.
*   Lamprinidis et al. (2018) Sotiris Lamprinidis, Daniel Hardt, and Dirk Hovy. 2018. Predicting News Headline Popularity with Syntactic and Semantic Knowledge Using Multi-Task Learning. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 659–664.
*   Lan et al. (2017) Man Lan, Jianxiang Wang, Yuanbin Wu, Zheng-Yu Niu, and Haifeng Wang. 2017. Multi-Task Attention-Based Neural Networks for Implicit Discourse Relationship Representation and Identification. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1299–1308.
*   Lauscher et al. (2018) Anne Lauscher, Goran Glavaš, Simone Paolo Ponzetto, and Kai Eckert. 2018. Investigating the Role of Argumentation in the Rhetorical Analysis of Scientific Publications with Neural Multi-Task Learning Models. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 3326–3338.
*   Le et al. (2020) Duong Le, My Thai, and Thien Nguyen. 2020. Multi-Task Learning for Metaphor Detection with Graph Convolutional Neural Networks and Word Sense Disambiguation. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 8139–8146.
*   Li et al. (2019) Quanzhi Li, Qiong Zhang, and Luo Si. 2019. Rumor Detection by Exploiting User Credibility Information, Attention and Multi-Task Learning. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 1173–1179.
*   Li and Zong (2008) Shoushan Li and Chengqing Zong. 2008. Multi-Domain Sentiment Classification. In *Proceedings of the 46th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 257–260.
*   Li and Lam (2017) Xin Li and Wai Lam. 2017. Deep Multi-Task Learning for Aspect Term Extraction with Memory Interaction. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 2886–2892.
*   Li and Liang (2021) Xiang Lisa Li and Percy Liang. 2021. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (Eds.). Association for Computational Linguistics, Online, 4582–4597. [https://doi.org/10.18653/v1/2021.acl-long.353](https://doi.org/10.18653/v1/2021.acl-long.353 "")
*   Li and Caragea (2019) Yingjie Li and Cornelia Caragea. 2019. Multi-Task Stance Detection with Sentiment and Stance Lexicons. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 6299–6305.
*   Liang et al. (2020) Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao, Xiaodong Fan, Ruofei Zhang, Rahul Agrawal, Edward Cui, Sining Wei, Taroon Bharti, Ying Qiao, Jiun-Hung Chen, Winnie Wu, Shuguang Liu, Fan Yang, Daniel Campos, Rangan Majumder, and Ming Zhou. 2020. XGLUE: A New Benchmark Datasetfor Cross-Lingual Pre-Training, Understanding and Generation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics, 6008–6018.
*   Lim et al. (2020) KyungTae Lim, Jay Yoon Lee, Jaime Carbonell, and Thierry Poibeau. 2020. Semi-Supervised Learning on Meta Structure: Multi-Task Tagging and Parsing in Low-Resource Scenarios. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Association for the Advancement of Artificial Intelligence (Ed.). Association for the Advancement of Artificial Intelligence.
*   Lin et al. (2018) Ying Lin, Shengqi Yang, Veselin Stoyanov, and Heng Ji. 2018. A Multi-Lingual Multi-Task Architecture for Low-Resource Sequence Labeling. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 799–809.
*   Liu et al. (2016c) Changsong Liu, Shaohua Yang, Sari Saba-Sadiya, Nishant Shukla, Yunzhong He, Song-Chun Zhu, and Joyce Chai. 2016c. Jointly Learning Grounded Task Structures from Language Instruction and Visual Demonstration. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1482–1492.
*   Liu et al. (2018b) Jiahua Liu, Wan Wei, Maosong Sun, Hao Chen, Yantao Du, and Dekang Lin. 2018b. A Multi-Answer Multi-Task Framework for Real-World Machine Reading Comprehension. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 2109–2118.
*   Liu et al. (2018a) Lizhen Liu, Xiao Hu, Wei Song, Ruiji Fu, Ting Liu, and Guoping Hu. 2018a. Neural Multitask Learning for Simile Recognition. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 1543–1553.
*   Liu et al. (2016b) Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2016b. Deep Multi-Task Learning with Shared Memory for Text Classification. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 118–127.
*   Liu et al. (2017) Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial Multi-Task Learning for Text Classification. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 1–10.
*   Liu et al. (2016a) Yang Liu, Sujian Li, Xiaodong Zhang, and Zhifang Sui. 2016a. Implicit Discourse Relation Classification via Multi-Task Neural Networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*.
*   Luan et al. (2018) Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 3219–3232.
*   Luong et al. (2016) Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. 2016. Multi-Task Sequence to Sequence Learning. *International Conference on Learning Representations 2016* (March 2016). arXiv:1511.06114
*   Maddela et al. (2019) Mounica Maddela, Wei Xu, and Daniel Preoţiuc-Pietro. 2019. Multi-Task Pairwise Neural Ranking for Hashtag Segmentation. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2538–2549.
*   Mao et al. (2020) Yuren Mao, Shuang Yun, Weiwei Liu, and Bo Du. 2020. Tchebycheff Procedure for Multi-Task Text Classification. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 4217–4226.
*   Marcus et al. (1994) Mitchell Marcus, Grace Kim, Mary Ann Marcinkiewicz, Robert MacIntyre, Ann Bies, Mark Ferguson, Karen Katz, and Britta Schasberger. 1994. The Penn Treebank: Annotating Predicate Argument Structure. In *Human Language Technology: Proceedings of a Workshop Held at Plainsboro, New Jersey, March 8-11, 1994*.
*   Martínez Alonso and Plank (2017) Héctor Martínez Alonso and Barbara Plank. 2017. When Is Multitask Learning Effective? Semantic Sequence Prediction under Varying Data Conditions. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers*. Association for Computational Linguistics, 44–53.
*   Masumura et al. (2018a) Ryo Masumura, Yusuke Shinohara, Ryuichiro Higashinaka, and Yushi Aono. 2018a. Adversarial Training for Multi-Task and Multi-Lingual Joint Modeling of Utterance Intent Classification. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 633–639.
*   Masumura et al. (2018b) Ryo Masumura, Tomohiro Tanaka, Ryuichiro Higashinaka, Hirokazu Masataki, and Yushi Aono. 2018b. Multi-Task and Multi-Lingual Joint Learning of Neural Lexical Utterance Classification Based on Partially-Shared Modeling. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 3586–3596.
*   Mishra et al. (2018) Abhijit Mishra, Srikanth Tamilselvam, Riddhiman Dasgupta, Seema Nagar, and Kuntal Dey. 2018. Cognition-Cognizant Sentiment Analysis With Multitask Subjectivity Summarization Based on Annotators’ Gaze Behavior. In *Proceedings of the AAAI Conference on Artificial Intelligence*.
*   Mishra et al. (2022) Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-Task Generalization via Natural Language Crowdsourcing Instructions. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 3470–3487. [https://doi.org/10.18653/v1/2022.acl-long.244](https://doi.org/10.18653/v1/2022.acl-long.244 "")
*   Mrkšić et al. (2015) Nikola Mrkšić, Diarmuid Ó Séaghdha, Blaise Thomson, Milica Gašić, Pei-Hao Su, David Vandyke, Tsung-Hsien Wen, and Steve Young. 2015. Multi-Domain Dialog State Tracking Using Recurrent Neural Networks. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*. Association for Computational Linguistics, 794–799.
*   Nishida et al. (2019) Kosuke Nishida, Kyosuke Nishida, Masaaki Nagata, Atsushi Otsuka, Itsumi Saito, Hisako Asano, and Junji Tomita. 2019. Answering While Summarizing: Multi-Task Learning for Multi-Hop QA with Evidence Extraction. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2335–2345.
*   Nishino et al. (2019) Toru Nishino, Shotaro Misawa, Ryuji Kano, Tomoki Taniguchi, Yasuhide Miura, and Tomoko Ohkuma. 2019. Keeping Consistency of Sentence Generation and Document Classification with Multi-Task Learning. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 3195–3205.
*   Niu et al. (2018) Xing Niu, Sudha Rao, and Marine Carpuat. 2018. Multi-Task Neural Models for Translating Between Styles Within and Across Languages. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 1008–1021.
*   Nivre et al. (2016) Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Yoav Goldberg, Jan Hajič, Christopher D. Manning, Ryan McDonald, Slav Petrov, Sampo Pyysalo, Natalia Silveira, Reut Tsarfaty, and Daniel Zeman. 2016. Universal Dependencies v1: A Multilingual Treebank Collection. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*. European Language Resources Association (ELRA), 1659–1666.
*   Oepen et al. (2016) Stephan Oepen, Marco Kuhlmann, Yusuke Miyao, Daniel Zeman, Silvie Cinková, Dan Flickinger, Jan Hajič, Angelina Ivanova, and Zdeňka Urešová. 2016. Towards Comparability of Linguistic Graph Banks for Semantic Parsing. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)*. European Language Resources Association (ELRA), 3991–3995.
*   Pasunuru and Bansal (2017) Ramakanth Pasunuru and Mohit Bansal. 2017. Multi-Task Video Captioning with Video and Entailment Generation. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 1273–1283.
*   Pasunuru and Bansal (2019) Ramakanth Pasunuru and Mohit Bansal. 2019. Continual and Multi-Task Architecture Search. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 1911–1922.
*   Peng et al. (2017) Hao Peng, Sam Thomson, and Noah A. Smith. 2017. Deep Multitask Learning for Semantic Dependency Parsing. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 2037–2048.
*   Pentyala et al. (2019) Shiva Pentyala, Mengwen Liu, and Markus Dreyer. 2019. Multi-Task Networks with Universe, Group, and Task Feature Learning. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 820–830.
*   Perera et al. (2018) Vittorio Perera, Tagyoung Chung, Thomas Kollar, and Emma Strubell. 2018. Multi-Task Learning For Parsing The Alexa Meaning Representation Language. In *Proceedings of the AAAI Conference on Artificial Intelligence*.
*   Pfeiffer et al. (2020) Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. arXiv:2005.00052
*   Pham et al. (2023) Hai Pham, Young Jin Kim, Subhabrata Mukherjee, David P. Woodruff, Barnabas Poczos, and Hany Hassan. 2023. Task-Based MoE for Multitask Multilingual Machine Translation. In *Proceedings of the 3rd Workshop on Multi-lingual Representation Learning (MRL)*. Association for Computational Linguistics, Singapore, 164–172. [https://doi.org/10.18653/v1/2023.mrl-1.13](https://doi.org/10.18653/v1/2023.mrl-1.13 "")
*   Pilault et al. (2021) Jonathan Pilault, Amine El hattami, and Christopher Pal. 2021. Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data. In *International Conference on Learning Representations*.
*   Raffel et al. (2020) Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research* 21, 140 (2020), 1–67. [http://jmlr.org/papers/v21/20-074.html](http://jmlr.org/papers/v21/20-074.html "")
*   Rawat et al. (2019) Bhanu Pratap Singh Rawat, Fei Li, and Hong Yu. 2019. Naranjo Question Answering Using End-to-End Multi-Task Learning Model. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 2547–2555.
*   Rei (2017) Marek Rei. 2017. Semi-Supervised Multitask Learning for Sequence Labeling. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, 2121–2130.
*   Ren et al. (2020) Qiyu Ren, Xiang Cheng, and Sen Su. 2020. Multi-Task Learning with Generative Adversarial Training for Multi-Passage Machine Reading Comprehension. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 8705–8712.
*   Rivas Rojas et al. (2020) Kervy Rivas Rojas, Gina Bustamante, Arturo Oncevay, and Marco Antonio Sobrevilla Cabezudo. 2020. Efficient Strategies for Hierarchical Text Classification: External Knowledge and Auxiliary Tasks. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2252–2257.
*   Ruder et al. (2019) Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. 2019. Latent Multi-Task Architecture Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* 33, 01 (July 2019), 4822–4829.
*   Sabour et al. (2017) Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. 2017. Dynamic Routing between Capsules. In *Advances in Neural Information Processing Systems*, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc.
*   Sanh et al. (2022) Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. 2022. Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*. [https://openreview.net/forum?id=9Vrb9D0WI4](https://openreview.net/forum?id=9Vrb9D0WI4 "")
*   Sanh et al. (2019) Victor Sanh, Thomas Wolf, and Sebastian Ruder. 2019. A Hierarchical Multi-Task Approach for Learning Embeddings from Semantic Tasks. *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 2019), 6949–6956.
*   Sarwar et al. (2019) Sheikh Muhammad Sarwar, Hamed Bonab, and James Allan. 2019. A Multi-Task Architecture on Relevance-Based Neural Query Translation. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 6339–6344.
*   Schröder and Biemann (2020) Fynn Schröder and Chris Biemann. 2020. Estimating the Influence of Auxiliary Tasks for Multi-Task Learning of Sequence Tagging Tasks. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2971–2985.
*   Shao et al. (2019) Bo Shao, Yeyun Gong, Junwei Bao, Jianshu Ji, Guihong Cao, Xiaola Lin, and Nan Duan. 2019. Weakly Supervised Multi-Task Learning for Semantic Parsing. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 3375–3381.
*   Shazeer et al. (2017) Noam Shazeer, \*Azalia Mirhoseini, \*Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In *International Conference on Learning Representations*. [https://openreview.net/forum?id=B1ckMDqlg](https://openreview.net/forum?id=B1ckMDqlg "")
*   Shen et al. (2019) Tao Shen, Xiubo Geng, Tao Qin, Daya Guo, Duyu Tang, Nan Duan, Guodong Long, and Daxin Jiang. 2019. Multi-Task Learning for Conversational Question Answering over a Large-Scale Knowledge Base. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 2442–2451.
*   Shimura et al. (2019) Kazuya Shimura, Jiyi Li, and Fumiyo Fukumoto. 2019. Text Categorization by Learning Predominant Sense of Words as Auxiliary Task. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 1109–1119.
*   Singla et al. (2018) Karan Singla, Dogan Can, and Shrikanth Narayanan. 2018. A Multi-Task Approach to Learning Multilingual Representations. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 214–220.
*   Søgaard and Goldberg (2016) Anders Søgaard and Yoav Goldberg. 2016. Deep Multi-Task Learning with Low Level Tasks Supervised at Lower Layers. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 231–235.
*   Song and Park (2019) Hyun-Je Song and Seong-Bae Park. 2019. Korean Morphological Analysis with Tied Sequence-to-Sequence Multi-Task Model. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 1436–1441.
*   Song et al. (2020b) Linfeng Song, Kun Xu, Yue Zhang, Jianshu Chen, and Dong Yu. 2020b. ZPR2: Joint Zero Pronoun Recovery and Resolution Using Multi-Task Learning and BERT. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 5429–5434.
*   Song et al. (2020a) Wei Song, Ziyao Song, Lizhen Liu, and Ruiji Fu. 2020a. Hierarchical Multi-Task Learning for Organization Evaluation of Argumentative Student Essays. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 3875–3881.
*   Stickland and Murray (2019) Asa Cooper Stickland and Iain Murray. 2019. BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. In *In Proceedings of the 36th International Conference on Machine Learning (ICML)*. PMLR, 5986–5995.
*   Subramanian et al. (2018) Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J Pal. 2018. Learning General Purpose Distributed Sentence Representations via Large Scale Multi-Task Learning. In *International Conference on Learning Representations*.
*   Suglia et al. (2020) Alessandro Suglia, Ioannis Konstas, Andrea Vanzo, Emanuele Bastianelli, Desmond Elliott, Stella Frank, and Oliver Lemon. 2020. CompGuessWhat?!: A Multi-Task Evaluation Framework for Grounded Language Learning. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 7625–7641.
*   Tafreshi and Diab (2018) Shabnam Tafreshi and Mona Diab. 2018. Emotion Detection and Classification in a Multigenre Corpus with Joint Multi-Task Deep Learning. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 2905–2913.
*   Tay et al. (2020) Yi Tay, Zhe Zhao, Dara Bahri, Donald Metzler, and Da-Cheng Juan. 2020. HyperGrid Transformers: Towards A Single Model for Multiple Tasks. In *International Conference on Learning Representations*.
*   Tian et al. (2019) Bing Tian, Yong Zhang, Jin Wang, and Chunxiao Xing. 2019. Hierarchical Inter-Attention Network for Document Classification with Multi-Task Learning. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 3569–3575.
*   Tong et al. (2018) Xiaowei Tong, Zhenxin Fu, Mingyue Shang, Dongyan Zhao, and Rui Yan. 2018. One "Ruler" for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning. In *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 4432–4438.
*   Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. *arXiv:1706.03762 \[cs\]* (Dec. 2017). arXiv:1706.03762 \[cs\]
*   Vijayaraghavan et al. (2017) Prashanth Vijayaraghavan, Soroush Vosoughi, and Deb Roy. 2017. Twitter Demographic Classification Using Deep Multi-Modal Multi-Task Learning. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 478–483.
*   Vu et al. (2022) Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou’, and Daniel Cer. 2022. SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 5039–5059. [https://doi.org/10.18653/v1/2022.acl-long.346](https://doi.org/10.18653/v1/2022.acl-long.346 "")
*   Wang et al. (2019a) Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019a. SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. In *Advances in Neural Information Processing Systems*, Vol. 32. Curran Associates, Inc.
*   Wang et al. (2019b) Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019b. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In *International Conference on Learning Representations 2019*. Association for Computational Linguistics, 353–355.
*   Wang et al. (2020c) Jiancheng Wang, Jingjing Wang, Changlong Sun, Shoushan Li, Xiaozhong Liu, Luo Si, Min Zhang, and Guodong Zhou. 2020c. Sentiment Classification in Customer Service Dialogue with Topic-Aware Multi-Task Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 9177–9184.
*   Wang et al. (2020a) Shaolei Wang, Wangxiang Che, Qi Liu, Pengda Qin, Ting Liu, and William Yang Wang. 2020a. Multi-Task Self-Supervised Learning for Disfluency Detection. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 9193–9200.
*   Wang et al. (2020e) Tianyi Wang, Yating Zhang, Xiaozhong Liu, Changlong Sun, and Qiong Zhang. 2020e. Masking Orchestration: Multi-Task Pretraining for Multi-Role Dialogue Representation Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 9217–9224.
*   Wang et al. (2018) Weichao Wang, Shi Feng, Wei Gao, Daling Wang, and Yifei Zhang. 2018. Personalized Microblog Sentiment Classification via Adversarial Cross-Lingual Multi-Task Learning. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 338–348.
*   Wang et al. (2020d) Yiren Wang, ChengXiang Zhai, and Hany Hassan Awadalla. 2020d. Multi-Task Learning for Multilingual Neural Machine Translation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. arXiv:2010.02523
*   Wang et al. (2023) Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, and Yoon Kim. 2023. Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning. In *The Eleventh International Conference on Learning Representations*. [https://openreview.net/forum?id=Nk2pDtuhTq](https://openreview.net/forum?id=Nk2pDtuhTq "")
*   Wang et al. (2020b) Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. 2020b. Gradient Vaccine: Investigating and Improving Multi-Task Optimization in Massively Multilingual Models. In *International Conference on Learning Representations*.
*   Watanabe et al. (2019) Taiki Watanabe, Akihiro Tamura, Takashi Ninomiya, Takuya Makino, and Tomoya Iwakura. 2019. Multi-Task Learning for Chemical Named Entity Recognition with Chemical Compound Paraphrasing. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 6244–6249.
*   Wei et al. (2022) Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022. Finetuned Language Models are Zero-Shot Learners. In *International Conference on Learning Representations*. [https://openreview.net/forum?id=gEZrGCozdqR](https://openreview.net/forum?id=gEZrGCozdqR "")
*   Wu and Huang (2015) Fangzhao Wu and Yongfeng Huang. 2015. Collaborative Multi-Domain Sentiment Classification. In *2015 IEEE International Conference on Data Mining*. 459–468.
*   Wu and Huang (2016) Fangzhao Wu and Yongfeng Huang. 2016. Personalized Microblog Sentiment Classification via Multi-Task Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* (2016), 7.
*   Wu et al. (2019) Lianwei Wu, Yuan Rao, Haolin Jin, Ambreen Nazir, and Ling Sun. 2019. Different Absorption from the Same Sharing: Sifted Multi-Task Learning for Fake News Detection. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 4644–4653.
*   Xia et al. (2019) Qingrong Xia, Zhenghua Li, and Min Zhang. 2019. A Syntax-Aware Multi-Task Learning Framework for Chinese Semantic Role Labeling. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 5382–5392.
*   Xiao et al. (2018a) Liqiang Xiao, Honglun Zhang, Wenqing Chen, Yongkun Wang, and Yaohui Jin. 2018a. Learning What to Share: Leaky Multi-Task Network for Text Classification. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 2055–2065.
*   Xiao et al. (2018b) Liqiang Xiao, Honglun Zhang, Wenqing Chen, Yongkun Wang, and Yaohui Jin. 2018b. MCapsNet: Capsule Network for Text with Multi-Task Learning. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 4565–4574.
*   Xie et al. (2022) Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I. Wang, Victor Zhong, Bailin Wang, Chengzu Li, Connor Boyle, Ansong Ni, Ziyu Yao, Dragomir Radev, Caiming Xiong, Lingpeng Kong, Rui Zhang, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2022. UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 602–631. [https://doi.org/10.18653/v1/2022.emnlp-main.39](https://doi.org/10.18653/v1/2022.emnlp-main.39 "")
*   Xing et al. (2018) Junjie Xing, Kenny Zhu, and Shaodian Zhang. 2018. Adaptive Multi-Task Transfer Learning for Chinese Word Segmentation in Medical Text. In *Proceedings of the 27th International Conference on Computational Linguistics*. Association for Computational Linguistics, 3619–3630.
*   Yadav et al. (2019) Shweta Yadav, Asif Ekbal, Sriparna Saha, and Pushpak Bhattacharyya. 2019. A Unified Multi-Task Adversarial Learning Framework for Pharmacovigilance Mining. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 5234–5245.
*   Yang et al. (2019) Min Yang, Lei Chen, Xiaojun Chen, Qingyao Wu, Wei Zhou, and Ying Shen. 2019. Knowledge-Enhanced Hierarchical Attention for Community Question Answering with Multi-Task and Adaptive Learning. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 5349–5355.
*   Yang and Hospedales (2015) Yongxin Yang and Timothy M Hospedales. 2015. A Unified Perspective on Multi-Domain and Multi-Task Learning. (2015), 9.
*   Ye et al. (2019) Wei Ye, Bo Li, Rui Xie, Zhonghao Sheng, Long Chen, and Shikun Zhang. 2019. Exploiting Entity BIO Tag Embeddings and Multi-Task Learning for Relation Extraction with Imbalanced Data. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 1351–1360.
*   Yu et al. (2020) Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient Surgery for Multi-Task Learning. *Advances in Neural Information Processing Systems* 33 (2020), 5824–5836.
*   Zalmout and Habash (2019) Nasser Zalmout and Nizar Habash. 2019. Adversarial Multitask Learning for Joint Multi-Feature and Multi-Dialect Morphological Modeling. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 1775–1786.
*   Zaremoodi et al. (2018) Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari. 2018. Adaptive Knowledge Sharing in Multi-Task Learning: Improving Low-Resource Neural Machine Translation. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 656–661.
*   Zeng et al. (2020b) Daojian Zeng, Haoran Zhang, and Qianying Liu. 2020b. CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 9507–9514.
*   Zeng et al. (2020a) Jiali Zeng, Linfeng Song, Jinsong Su, Jun Xie, Wei Song, and Jiebo Luo. 2020a. Neural Simile Recognition with Cyclic Multitask Learning and Local Attention. *Proceedings of the AAAI Conference on Artificial Intelligence* 34, 05 (April 2020), 9515–9522.
*   Zhang et al. (2018b) Honglun Zhang, Liqiang Xiao, Wenqing Chen, Yongkun Wang, and Yaohui Jin. 2018b. Multi-Task Label Embedding for Text Classification. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 4545–4553.
*   Zhang et al. (2017) Honglun Zhang, Liqiang Xiao, Yongkun Wang, and Yaohui Jin. 2017. A Generalized Recurrent Neural Architecture for Text Classification with Multi-Task Learning. In *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 3385–3391.
*   Zhang et al. (2023) Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. 2023. Instruction Tuning for Large Language Models: A Survey. *arXiv preprint arXiv:2308.10792* (2023).
*   Zhang et al. (2018a) Yuxiang Zhang, Jiamei Fu, Dongyu She, Ying Zhang, Senzhang Wang, and Jufeng Yang. 2018a. Text Emotion Distribution Learning via Multi-Task Convolutional Neural Network. In *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 4595–4601.
*   Zhang and Yang (2021) Yu Zhang and Qiang Yang. 2021. A Survey on Multi-Task Learning. *IEEE Transactions on Knowledge and Data Engineering* (2021).
*   Zhao et al. (2020) He Zhao, Longtao Huang, Rong Zhang, Quan Lu, and Hui Xue. 2020. SpanMlt: A Span-Based Multi-Task Learning Framework for Pair-Wise Aspect and Opinion Terms Extraction. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 3239–3248.
*   Zhao et al. (2019) Sendong Zhao, Ting Liu, Sicheng Zhao, and Fei Wang. 2019. A Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization. *Proceedings of the AAAI Conference on Artificial Intelligence* 33, 01 (July 2019), 817–824.
*   Zhao et al. (2023) Xin Zhao, Kun Zhou, Beichen Zhang, Zheng Gong, Zhipeng Chen, Yuanhang Zhou, Ji-Rong Wen, Jing Sha, Shijin Wang, Cong Liu, and Guoping Hu. 2023. JiuZhang 2.0: A Unified Chinese Pre-Trained Language Model for Multi-Task Mathematical Problem Solving. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (, Long Beach, CA, USA,) *(KDD ’23)*. Association for Computing Machinery, New York, NY, USA, 5660–5672. [https://doi.org/10.1145/3580305.3599850](https://doi.org/10.1145/3580305.3599850 "")
*   Zheng et al. (2018) Renjie Zheng, Junkun Chen, and Xipeng Qiu. 2018. Same Representation, Different Attentions: Shareable Sentence Representation Learning from Multiple Tasks. In *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization, 4616–4622.
*   Zhou et al. (2019) Wenjie Zhou, Minghua Zhang, and Yunfang Wu. 2019. Multi-Task Learning with Language Modeling for Question Generation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 3394–3399.
*   Zhu et al. (2019) Chenguang Zhu, Michael Zeng, and Xuedong Huang. 2019. Multi-Task Learning for Natural Language Generation in Task-Oriented Dialogue. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*. Association for Computational Linguistics, 1261–1266.
*   Zhuang and Liu (2019) Jinfeng Zhuang and Yu Liu. 2019. PinText: A Multitask Text Embedding System in Pinterest. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 2653–2661.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")