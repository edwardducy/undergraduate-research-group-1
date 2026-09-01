# *Smarter, Better, Faster, Longer*: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

Benjamin Warner1† Antoine Chaffin2† Benjamin Clavié1† Orion Weller<sup>3</sup> Oskar Hallström<sup>2</sup> Said Taghadouini<sup>2</sup> Alexis Gallagher<sup>1</sup> Raja Biswas<sup>1</sup> Faisal Ladhak4\* Tom Aarsen<sup>5</sup> Nathan Cooper<sup>1</sup> Griffin Adams<sup>1</sup> Jeremy Howard<sup>1</sup> Iacopo Poli<sup>2</sup>

<sup>1</sup>Answer.AI <sup>2</sup>LightOn <sup>3</sup> Johns Hopkins University <sup>4</sup>NVIDIA <sup>5</sup>Hugging Face

*†: core authors, \*: work done while at Answer.AI*

Correspondence: {bw,bc}@answer.ai, antoine.chaffin@lighton.ai

# Abstract

Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-ofthe-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, Modern-BERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs.

# 1 Introduction

After the release of BERT [\(Devlin et al.,](#page-10-0) [2019\)](#page-10-0), encoder-only transformer-based [\(Vaswani et al.,](#page-13-0) [2017\)](#page-13-0) language models dominated most applications of modern Natural Language Processing (NLP). Despite the rising popularity of Large Language Models (LLMs) such as GPT [\(Radford et al.,](#page-12-0) [2018,](#page-12-0) [2019;](#page-12-1) [Brown et al.,](#page-9-0) [2020\)](#page-9-0), Llama [\(Touvron](#page-13-1) [et al.,](#page-13-1) [2023;](#page-13-1) [Dubey et al.,](#page-10-1) [2024\)](#page-10-1), and Qwen [\(Bai](#page-9-1) [et al.,](#page-9-1) [2023;](#page-9-1) [Yang et al.,](#page-14-0) [2024\)](#page-14-0), encoder-only models remain widely used in a variety of nongenerative downstream applications[.](#page-0-0)

The encoder's popularity is largely due to their modest inference requirements, enabling them to efficiently process corpora of documents at scale for retrieval and quickly perform discriminative tasks. Encoder models offer a compelling tradeoff in quality versus size, making them a popular

option against encoder-decoder and decoder-only language models when dealing with substantial amounts of data [\(Penedo et al.,](#page-12-2) [2024\)](#page-12-2).

Encoder models are particularly popular in Information Retrieval (IR) applications, e.g., semantic search, with notable progress on leveraging encoders for this task [\(Karpukhin et al.,](#page-11-0) [2020;](#page-11-0) [Khat](#page-12-3)[tab and Zaharia,](#page-12-3) [2020\)](#page-12-3). While LLMs have taken the spotlight in recent years, they have also motivated a renewed interest in encoder-only models for IR. Indeed, encoder-based semantic search is a core component of Retrieval-Augmented Generation (RAG) pipelines [\(Lewis et al.,](#page-12-4) [2020\)](#page-12-4), where encoder models are used to retrieve and feed LLMs with context relevant to user queries.

Encoder-only models are also still frequently used for a variety of discriminative tasks such as classification [\(Tunstall et al.,](#page-13-2) [2022\)](#page-13-2) or Named Entity Recognition (NER) [\(Zaratiana et al.,](#page-14-1) [2024\)](#page-14-1), where they often match the performance of specialized LLMs. Here again, they can be used in conjunction with LLMs, for example detecting toxic prompts [\(Ji et al.,](#page-11-1) [2023;](#page-11-1) [Jiang et al.,](#page-11-2) [2024b\)](#page-11-2) and preventing responses, or routing queries in an agentic framework [\(Yao et al.,](#page-14-2) [2023;](#page-14-2) [Schick et al.,](#page-13-3) [2023\)](#page-13-3).

Surprisingly, these pipelines currently rely on older models, and quite often on the original BERT itself as their backbone [\(Wang et al.,](#page-14-3) [2022;](#page-14-3) [Xiao](#page-14-4) [et al.,](#page-14-4) [2023\)](#page-14-4), without leveraging improvements developed in recent years. Practitioners face many drawbacks: sequence lengths limited to 512 tokens, suboptimal model design [\(Anthony et al.,](#page-9-2) [2024\)](#page-9-2) and vocabulary sizes [\(Karpathy,](#page-11-3) [2023\)](#page-11-3), and generally inefficient architectures, whether in terms of downstream performance or computational efficiency. Finally, training data is limited in volume and restricted to narrow domains (especially lacking code data) or lacking knowledge of recent events.

Recent modernization efforts have only partially addressed the shortcomings of encoder-only mod-

<span id="page-0-0"></span><https://github.com/AnswerDotAI/ModernBERT>

els due to limited breadth. MosaicBERT [\(Portes](#page-12-5) [et al.,](#page-12-5) [2023\)](#page-12-5), CrammingBERT [\(Geiping and Gold](#page-10-2)[stein,](#page-10-2) [2023\)](#page-10-2), and AcademicBERT [\(Izsak et al.,](#page-11-4) [2021\)](#page-11-4) focused on matching BERT performance with better training efficiency. NomicBERT [\(Nuss](#page-12-6)[baum et al.,](#page-12-6) [2024\)](#page-12-6) and GTE-en-MLM [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5) (developed concurrently to this work) introduced longer-context encoder models focused on retrieval applications, but did not optimize for efficiency or classification performance, and re-used older training data mixtures which is especially apparent in programming-related tasks.

Contributions We present ModernBERT, a modernized encoder-only transformer model, with an improved architecture designed to increase downstream performance and efficiency, especially over longer sequence lengths. We also bring encoderonly models to modern, larger data scales, by training on 2 trillion tokens, with a data mixture including code data. We release two models, [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) and [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), which reach state-of-the-art overall performance against all existing encoder models on a wide variety of downstream tasks. These results are achieved with considerably higher inference efficiency, processing sequences of 8192 tokens almost two times faster than previous models.

To support future research on encoder-only models, we release [FlexBERT](https://github.com/AnswerDotAI/ModernBERT)[<sup>1</sup>](#page-1-0) , our modular architecture framework allowing easy experimentation, and inspired by Pythia [\(Biderman et al.,](#page-9-3) [2023\)](#page-9-3), all intermediate training checkpoints (further detailed in Section [2.2.2\)](#page-2-0).

# 2 Methods

## 2.1 Architectural Improvements

Our model architecture extends the standard transformer architecture [\(Vaswani et al.,](#page-13-0) [2017\)](#page-13-0) by incorporating extensively tested recent advances (Section [2.1.1\)](#page-1-1). We introduce additional efficiencyoriented modifications, through both architectural and implementation improvements (Section [2.1.2\)](#page-1-2) and a GPU optimized model design (Section [2.1.3\)](#page-2-1). All of our architectural decisions were informed by ablations, which we detail in Appendix [D.](#page-17-0)

## <span id="page-1-1"></span>2.1.1 Modern Transformer

Bias Terms Following [\(Dayma et al.,](#page-10-3) [2021\)](#page-10-3), we disable bias terms in all linear layers except for the final decoder linear layer[<sup>2</sup>](#page-1-3) . We also disable all bias terms in Layer Norms [\(Xu et al.,](#page-14-6) [2019\)](#page-14-6). These two changes allow us to spend more of our parameter budget in linear layers.

Positional Embeddings We use rotary positional embeddings (RoPE) [\(Su et al.,](#page-13-4) [2024\)](#page-13-4) instead of absolute positional embeddings. This choice is motivated by the proven performance of RoPE in short- and long-context language models [\(Black](#page-9-4) [et al.,](#page-9-4) [2022;](#page-9-4) [Dubey et al.,](#page-10-1) [2024;](#page-10-1) [Gemma Team et al.,](#page-10-4) [2024\)](#page-10-4), efficient implementations in most frameworks, and ease of context extension.

Normalization We use a pre-normalization block [\(Xiong et al.,](#page-14-7) [2020\)](#page-14-7) with the standard layer normalization [\(Lei Ba et al.,](#page-12-7) [2016\)](#page-12-7), which is known to help stabilize training [\(Xiong et al.,](#page-14-7) [2020\)](#page-14-7). Similar to CrammingBERT [\(Geiping and Goldstein,](#page-10-2) [2023\)](#page-10-2) which also uses pre-normalization, we add a LayerNorm after the embedding layer. To avoid repetition, we remove the first LayerNorm in the first attention layer.

Activation We adopt GeGLU [\(Shazeer,](#page-13-5) [2020\)](#page-13-5), a Gated-Linear Units (GLU)-based [\(Dauphin et al.,](#page-10-5) [2017\)](#page-10-5) activation function built on top of the original BERT's GeLU [\(Hendrycks and Gimpel,](#page-11-5) [2016\)](#page-11-5) activation function. This is in line with recent work showing consistent empirical improvements when using GLU variants [\(Shazeer,](#page-13-5) [2020;](#page-13-5) [Geiping and](#page-10-2) [Goldstein,](#page-10-2) [2023\)](#page-10-2).

### <span id="page-1-2"></span>2.1.2 Efficiency Improvements

Alternating Attention Following recent work on efficient long context models [\(Gemma Team et al.,](#page-10-4) [2024\)](#page-10-4), attention layers in ModernBERT alternate between global attention, where every token within a sequence attends to every other token, and local attention, where tokens only attend to each other within a small sliding window [\(Beltagy et al.,](#page-9-5) [2020\)](#page-9-5). In ModernBERT, every third layer employs global attention with a RoPE theta of 160,000 and the remaining layers use a 128 token, local sliding window attention with a RoPE theta of 10,000.

Unpadding ModernBERT follows MosaicBERT [\(Portes et al.,](#page-12-5) [2023\)](#page-12-5) and GTE [\(Zhang](#page-14-5) [et al.,](#page-14-5) [2024\)](#page-14-5) in employing unpadding [\(Zeng et al.,](#page-14-8) [2022\)](#page-14-8) for both training and inference. Encoderonly language models typically use padding tokens to ensure a uniform sequence length in a batch,

<span id="page-1-0"></span><sup>1</sup> FlexBERT is built on top of a revised MosaicBERT [\(Portes et al.,](#page-12-5) [2023\)](#page-12-5) codebase.

<span id="page-1-3"></span><sup>2</sup>While many efficient BERT training recipes disable the bias term in the decoder, e.g. [Geiping and Goldstein](#page-10-2) [\(2023\)](#page-10-2), we hypothesized a decoder bias might help alleviate weight tying's negative effects [\(Gao et al.,](#page-10-6) [2019;](#page-10-6) [Welch et al.,](#page-14-9) [2020\)](#page-14-9).

wasting compute on semantically empty tokens. Unpadding avoids this inefficiency by removing padding tokens, concatenating all sequences from a minibatch into a single sequence, and processing it as a batch of one. Prior unpadding implementations unpad and repad sequences internally for different model layers, wasting compute and memory bandwidth. We use Flash Attention's variable length attention and RoPE implementations, allowing jagged attention masks and RoPE applications on one unpadded sequence. ModernBERT unpads inputs before the token embedding layer and optionally repads model outputs leading to a 10-to-20 percent performance improvement over other unpadding methods.

Flash Attention Flash Attention [\(Dao et al.,](#page-10-7) [2022\)](#page-10-7) is a core component of modern transformerbased models, providing memory and compute efficient attention kernels. At the start of this work, Flash Attention 3 [\(Shah et al.,](#page-13-6) [2024\)](#page-13-6), the most recent iteration for Nvidia H100 GPUs, did not include support for sliding window attention. ModernBERT uses a mixture of Flash Attention 3 for global attention layers and Flash Attention 2 [\(Dao,](#page-10-8) [2023\)](#page-10-8) for local attention layers.

torch.compile We leverage PyTorch's built-in compiling [\(Ansel et al.,](#page-9-6) [2024\)](#page-9-6) to improve the training efficiency by compiling all compatible modules. This yields a 10 percent improvement in throughput with negligible compilation overhead.

## <span id="page-2-1"></span>2.1.3 Model Design

At the same parameter count, models with more narrow layers (*Deep & Narrow*) have different learning patterns than models with fewer wide layers (*Shallow & Wide*) [\(Nguyen et al.,](#page-12-8) [2021\)](#page-12-8). [Tay](#page-13-7) [et al.](#page-13-7) [\(2022\)](#page-13-7) and [\(Liu et al.,](#page-12-9) [2024\)](#page-12-9) have shown that *Deep & Narrow* language models have better downstream performance than their shallower counterparts, at the expense of slower inference.

[Anthony et al.](#page-9-2) [\(2024\)](#page-9-2) highlighted that large runtime gains can be unlocked by designing models in a *hardware-aware* way, which had previously been anecdotally observed by many practitioners [\(Shoeybi et al.,](#page-13-8) [2019;](#page-13-8) [Karpathy,](#page-11-3) [2023;](#page-11-3) [Black et al.,](#page-9-4) [2022\)](#page-9-4). ModernBERT was designed through many small-scale ablations to maximize the utilization of a basket of common GPUs[<sup>3</sup>](#page-2-2) , while

aiming to be as *Deep & Narrow* as possible without a significant inference slowdown.

ModernBERT has 22 and 28 layers for the base and large models, for a total parameter count of 149 and 395 million, respectively, striking the balance between downstream performance and hardware efficiency. ModernBERT base has a hidden size of 768 with a GLU expansion of 2,304, while large has a hidden size of 1,024 and GLU expansion of 5,248. These ratios allow optimal tiling across tensor cores and the most efficient tiling across the differing number of streaming multiprocessors on our target basket of GPUs. More details on model design are provided in Appendix [B.](#page-16-0)

## 2.2 Training

# 2.2.1 Data

Mixture Both ModernBERT models are trained on 2 trillion tokens of primarily English data from a variety of data sources, including web documents, code, and scientific literature, following common modern data mixtures. We choose the final data mixture based on a series of ablations.

Tokenizer Unlike the majority of recent encoders which reuse the original BERT tokenizer [\(Nussbaum et al.,](#page-12-6) [2024;](#page-12-6) [Portes et al.,](#page-12-5) [2023;](#page-12-5) [Zhang et al.,](#page-14-5) [2024\)](#page-14-5), we opt to use a modern BPE tokenizer. We use a modified version of the OLMo tokenizer [\(Groeneveld et al.,](#page-11-6) [2024\)](#page-11-6) which provides better token efficiency and performance on coderelated tasks. The ModernBERT tokenizer uses the same special tokens (e.g., [CLS] and [SEP]) and templating as the original BERT model [\(Devlin](#page-10-0) [et al.,](#page-10-0) [2019\)](#page-10-0), facilitating backwards compatibility. To ensure optimal GPU utilization [\(Anthony et al.,](#page-9-2) [2024;](#page-9-2) [Karpathy,](#page-11-3) [2023\)](#page-11-3), the vocabulary is set to 50,368, a multiple of 64 and includes 83 unused tokens to support downstream applications.

Sequence Packing In order to avoid high minibatch-size variance within our training batches as a result of unpadding, we adopt sequence packing [\(Raffel et al.,](#page-13-9) [2020;](#page-13-9) [Krell et al.,](#page-12-10) [2022\)](#page-12-10) with a greedy algorithm, which resulted in a sequence packing efficiency of over 99 percent, ensuring batch size uniformity.

# <span id="page-2-0"></span>2.2.2 Training Settings

MLM We follow the Masked Language Modeling (MLM) setup used by MosaicBERT [\(Portes et al.,](#page-12-5) [2023\)](#page-12-5). We remove the Next-Sentence Prediction objective which introduces noticeable overhead for no performance improvement [\(Liu et al.,](#page-12-11) [2019a;](#page-12-11)

<span id="page-2-2"></span><sup>3</sup>Which, at the time of this work, are server GPUs: NVIDIA T4, A10, L4, A100, and H100 and consumer GPUs: NVIDIA RTX 3090 and 4090. Prioritization was given to inference GPUs (excluding A100 & H100).

[Izsak et al.,](#page-11-4) [2021\)](#page-11-4), and use a masking rate of 30 percent, as the original rate of 15 percent has since been shown to be sub-optimal [\(Wettig et al.,](#page-14-10) [2023\)](#page-14-10).

Optimizer We use the StableAdamW optimizer [\(Wortsman et al.,](#page-14-11) [2023\)](#page-14-11), which improves upon AdamW [\(Loshchilov and Hutter,](#page-12-12) [2019\)](#page-12-12) by adding Adafactor-style [\(Shazeer and Stern,](#page-13-10) [2018\)](#page-13-10) update clipping as a per-parameter learning rate adjustment. StableAdamW's learning rate clipping outperformed standard gradient clipping on downstream tasks and led to more stable training. Hyperparameters details are given in Appendix [A.](#page-15-0)

Learning Rate Schedule During pretraining, we use a modified trapezoidal Learning Rate (LR) schedule [\(Xing et al.,](#page-14-12) [2018\)](#page-14-12), also known as Warmup-Stable-Decay (WSD) [\(Zhai et al.,](#page-14-13) [2022;](#page-14-13) [Hu et al.,](#page-11-7) [2024\)](#page-11-7). After a short LR warmup, the trapezoidal schedule holds the LR constant for the majority of training, followed by a short LR decay. This schedule has been shown to match the performance of cosine scheduling [\(Hägele et al.,](#page-11-8) [2024;](#page-11-8) [Hallström et al.,](#page-11-9) [2024\)](#page-11-9) with the benefit of enabling continual training on any checkpoint without cold restart issues [\(Ash and Adams,](#page-9-7) [2019\)](#page-9-7). Unlike most trapezoidal schedules, we use a <sup>1</sup> − sqrt LR decay [\(Hägele et al.,](#page-11-10) [2024\)](#page-11-10), as we found it to outperform linear and cosine decay.

We trained ModernBERT-base at a constant LR of 8e-4 for 1.7 trillion tokens following a 3 billion token warmup. After a 2 billion token warmup, we trained ModernBERT-large at a LR of 5e-4 for 900 billion tokens. We rolled back and restarted training at 5e-5 for the remaining 800 billion tokens after large's loss plateaued for a few hundred billion tokens at 5e-4. Full loss curves are available in Appendi[xG.](#page-20-0)

Batch Size Schedule Batch size scheduling starts with smaller gradient accumulated batches, increasing over time to the full batch size. In ablations, this schedule accelerated training progress. We warmup the batch size from 768 to 4,608 over 50 billion tokens and from 448 to 4,928 over 10 billion tokens, for ModernBERT-base and -large, respectively, with an uneven token schedule so each batch size has the same number of update steps. Details are provided in Appendix [A.1.](#page-15-1)

Weight Initialization and Tiling We initialize ModernBERT-base with random weights following the Megatron initialization [\(Shoeybi et al.,](#page-13-8) [2019\)](#page-13-8). For ModernBERT-large, we follow the Phi model

family [\(Li et al.,](#page-12-13) [2023;](#page-12-13) [Javaheripi et al.,](#page-11-11) [2023\)](#page-11-11) [4](#page-3-0) and initialize -large's weights from ModernBERT-base. In ablation runs, this consistently matched Phi's improved training results and greatly speed up the initial loss decrease of our model training[<sup>5</sup>](#page-3-1) . Details are provided in Appendix [A.3.](#page-15-2)

Context Length Extension After training on 1.7 trillion tokens at a 1024 sequence length and RoPE theta of 10,000, we extend the native context length of ModernBERT to 8192 tokens by increasing the global attention layer's RoPE theta to 160,000 and train for an additional 300 billion tokens. We first train at a constant lower learning rate[<sup>6</sup>](#page-3-2) of 3e-4 for 250 billion tokens on an 8192 token mixture of the original pretraining dataset sampled following [Fu](#page-10-9) [et al.](#page-10-9) [\(2024\)](#page-10-9). Next, we upsample higher-quality sources following [Gao et al.](#page-10-10) [\(2024\)](#page-10-10) and conduct the decay phase with a <sup>1</sup> − sqrt LR schedule over 50 billion tokens. This context extension process yielded the most balanced model on downstream tasks, as most of our ablations using only one of these strategies resulted in a performance loss on either retrieval or classification tasks.

Training Ablations Discussion of the training and architecture ablations can be found in Appendix [D](#page-17-0)

# 3 Downstream Evaluation

We performed an extensive set of evaluations, across a large range of tasks, aiming to demonstrate the versatility of ModernBERT in common scenarios.

For all tasks, ModernBERT is evaluated against existing encoders of similar size. The BASE size, conventionally defined as under 150 million parameters, includes [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) [\(Devlin et al.,](#page-10-0) [2019\)](#page-10-0), [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) [\(He et al.,](#page-11-12) [2023\)](#page-11-12), [RoBERTa](https://huggingface.co/FacebookAI/roberta-base)[base](https://huggingface.co/FacebookAI/roberta-base) [\(Liu et al.,](#page-12-11) [2019a\)](#page-12-11), as well as the more recent 8192 context [NomicBERT](https://huggingface.co/nomic-ai/NomicBERT-2048) [\(Nussbaum et al.,](#page-12-6) [2024\)](#page-12-6) and [GTE-en-MLM-base](https://huggingface.co/Alibaba-NLP/GTE-en-MLM-base) [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5). The LARGE size, conventionally defined as above 300 million and under 500 million parameters, includes [BERT-large-uncased](https://huggingface.co/google-bert/bert-large-uncased) [\(Devlin et al.,](#page-10-0) [2019\)](#page-10-0), [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) [\(He et al.,](#page-11-12) [2023\)](#page-11-12) and [RoBERTa](https://huggingface.co/FacebookAI/roberta-large)[large](https://huggingface.co/FacebookAI/roberta-large) [\(Liu et al.,](#page-12-11) [2019a\)](#page-12-11) and [GTE-en-MLM](https://huggingface.co/Alibaba-NLP/GTE-en-MLM-large)[large](https://huggingface.co/Alibaba-NLP/GTE-en-MLM-large) [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5).

<span id="page-3-1"></span><span id="page-3-0"></span><sup>4</sup>As detailed in their 2023 NeurIPS presentation.

<sup>5</sup>This initialization reduced the amount of batch size and LR warmup needed for ModernBERT-large

<span id="page-3-2"></span><sup>6</sup>We only lowered the LR for ModernBERT-base, as large already decreased LR during the 1024 token training phase.

#### 3.1 Evaluation Setting

## <span id="page-4-4"></span>3.1.1 Natural Language Understanding

The General Language Understanding Evaluation (GLUE) benchmark [\(Wang et al.,](#page-14-14) [2018\)](#page-14-14) is the standard Natural Language Understanding (NLU) benchmark for encoder models, aiming to measure how well a model performs across a range of sentence or sentence-pair understanding tasks, such as sentiment detection [\(Liu et al.,](#page-12-14) [2019b\)](#page-12-14) or language entailment, through tasks such as MNLI [\(Williams](#page-14-15) [et al.,](#page-14-15) [2018\)](#page-14-15). Although GLUE is often regarded as saturated by the best-performing models, such as large language models [\(Zhao et al.,](#page-14-16) [2023\)](#page-14-16), it remains one of the most commonly used evaluation suites for smaller encoder-based models, and provides a good impression of a model's performance on common classification tasks [\(Portes et al.,](#page-12-5) [2023;](#page-12-5) [Zhang et al.,](#page-14-5) [2024;](#page-14-5) [He et al.,](#page-11-12) [2023\)](#page-11-12).

We follow the practice of previous studies [\(De](#page-10-0)[vlin et al.,](#page-10-0) [2019;](#page-10-0) [Liu et al.,](#page-12-11) [2019a;](#page-12-11) [He et al.,](#page-11-12) [2023\)](#page-11-12) and conduct a hyperparameter search on each GLUE subset (detailed in Appendix [E.1\)](#page-18-0) in order to provide values comparable to other models.[<sup>7</sup>](#page-4-0)

## <span id="page-4-3"></span>3.1.2 Text Retrieval

Information Retrieval (IR) is one of the most common applications of encoder-only models,[<sup>8</sup>](#page-4-1) where they are used to represent documents and queries in semantic search [\(Karpukhin et al.,](#page-11-0) [2020\)](#page-11-0). This domain has recently seen considerable growth and interest following the spread of LLMs where semantic search powered by lightweight models is used to provide relevant context to LLMs as part of Retrieval-Augmented Generation pipelines.

We evaluate models in both the single-vector Dense Passage Retrieval (DPR) [\(Karpukhin et al.,](#page-11-0) [2020\)](#page-11-0) setting and the multi-vector ColBERT [\(Khat](#page-12-3)[tab and Zaharia,](#page-12-3) [2020\)](#page-12-3) setting.

We report retrieval results on the popular BEIR evaluation suite [\(Thakur et al.,](#page-13-11) [2021\)](#page-13-11), the common standard for evaluating retrieval performance across a variety of tasks and domains, using the nDCG@10 metric. For each setting detailed below, we conduct a learning rate sweep based on results

over a subset of the BEIR benchmarks to select the final model, detailed in Appendix [E.2.](#page-18-1)

Single vector retrieval One of the most common approaches to neural retrieval using encoders is DPR [\(Karpukhin et al.,](#page-11-0) [2020\)](#page-11-0), where a singlevector is used to represent an entire document. The similarity between a query and a document can then be computed through distance operations, such as cosine similarity. Models are finetuned using contrastive learning to create representations which are close if a document is relevant to a query, and distant if not [\(van den Oord et al.,](#page-13-12) [2018\)](#page-13-12).

We train every base model using the MS-MARCO [\(Bajaj et al.,](#page-9-8) [2016\)](#page-9-8) dataset with [mined](https://huggingface.co/datasets/sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1) [hard negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1) [\(Xuan et al.,](#page-14-17) [2020\)](#page-14-17) on 1.25M samples using [sentence-transformers](https://sbert.net/) [\(Reimers and](#page-13-13) [Gurevych,](#page-13-13) [2019\)](#page-13-13). Hyperparameters are detailed in Appendix [E.2.](#page-18-1)

Multi vector retrieval Multi-vector retrieval, championed by ColBERT [\(Khattab and Zaharia,](#page-12-3) [2020\)](#page-12-3), seeks to mitigate lost information from compressing an entire sequence into a single vector. In multi-vector retrieval, each document is represented by all of its individual token vectors, and the similarity between a query and a document is computed using the MaxSim[<sup>9</sup>](#page-4-2) operator.

We adopt the training setup of JaCol-BERTv2.5 [\(Clavié,](#page-10-11) [2024\)](#page-10-11), an update on the ColBERTv2 [\(Santhanam et al.,](#page-13-14) [2022\)](#page-13-14) training procedure. We train all models by distilling the knowledge of a teacher model by using the KL-Divergence between the normalized teacher and student scores. Models are trained on [810k](https://huggingface.co/datasets/lightonai/ms-marco-en-bge) [samples from MS-Marco](https://huggingface.co/datasets/lightonai/ms-marco-en-bge) [\(Bajaj et al.,](#page-9-8) [2016\)](#page-9-8) and teacher scores from BGE-M3 [\(Chen et al.,](#page-9-9) [2024\)](#page-9-9), using the [PyLate](https://github.com/lightonai/pylate) library [\(Chaffin and](#page-9-10) [Sourty,](#page-9-10) [2024\)](#page-9-10). Hyperparameters are detailed in Appendix [E.2.](#page-18-1)

## 3.1.3 Long-Context Text Retrieval

With a native 8192 context length, ModernBERT improves long-context performance over most existing encoders. However, there are relatively few standardized long-context benchmarks for encoder-only models, and most benchmarks, such as Needle-in-a-haystack [\(Kamradt,](#page-11-13) [2023\)](#page-11-13) and RULER [\(Hsieh et al.,](#page-11-14) [2024\)](#page-11-14) are geared towards generative tasks. Given this limitation, we demonstrate improved long-context performance on the English subset of [MLDR](https://huggingface.co/datasets/Shitao/MLDR) [\(Chen et al.,](#page-9-9) [2024\)](#page-9-9), a long-context

<span id="page-4-0"></span><sup>7</sup>As [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5) do not explicitly mention a parameter sweep, we initially ran the same hyperparameter sweep as we did for ModernBERT, but observed inconsistencies in the results. To avoid under-representing GTE-en-MLM's capabilities, we choose to use their reported GLUE results.

<span id="page-4-1"></span><sup>8</sup>At the time of this paper's writing, over half of the 100 most downloaded models on the [HuggingFace Model Hub](https://huggingface.co/models) were encoder-based retrieval models.

<span id="page-4-2"></span><sup>9</sup>The sum for every query token of its similarity with the most similar document token

<span id="page-5-2"></span>

Table 1: Results for all models across an overview of all tasks. CSN refers to CodeSearchNet and SQA to StackQA. MLDRID refers to in-domain (fine-tuned on the training set) evaluation, and MLDROOD to out-of-domain.

retrieval benchmark comprised of over 200,000 long documents. We evaluate three settings:

Single Vector – Out-Of-Domain Models are trained on short-context MS-MARCO as described above, and is evaluated on long context MLDR without any further fine-tuning.

Single Vector – In Domain Models trained on MS-MARCO are further fine-tuned on longcontext MLDR training set before being evaluated.

Multi-Vector – Out-Of-Domain Due to its token-level MaxSim mechanism, ColBERT models are able to generalize to long-context without any specific training [\(Bergum,](#page-9-11) [2024\)](#page-9-11). We directly evaluate the best checkpoints from Section [3.1.2](#page-4-3) without any further fine-tuning on MLDR.

#### 3.1.4 Code Retrieval

Fueled by increasingly good code completion models [\(Jiang et al.,](#page-11-15) [2024a\)](#page-11-15), downstream applications have quickly grown in popularity following the emergence of code assistants.[<sup>10</sup>](#page-5-0) Encoder-only models are used to process and retrieve large quantities of code-related information under resource constraints, increasing the importance of measuring and improving code capabilities of encoder models [\(Li et al.,](#page-12-15) [2024\)](#page-12-15). Unlike most previous encoders which were largely trained only on textual data [\(De](#page-10-0)[vlin et al.,](#page-10-0) [2019;](#page-10-0) [Liu et al.,](#page-12-11) [2019a;](#page-12-11) [Portes et al.,](#page-12-5) [2023;](#page-12-5) [Zhang et al.,](#page-14-5) [2024;](#page-14-5) [Nussbaum et al.,](#page-12-6) [2024\)](#page-12-6), ModernBERT is pre-trained on code and uses a

code-aware tokenizer[<sup>11</sup>](#page-5-1) .

To measure programming-related performance, we evaluate all models on CodeSearchNet [\(Hu](#page-11-16)[sain et al.,](#page-11-16) [2019\)](#page-11-16), a code-to-text benchmark where the model must identify relevant docstring or comments for code blocks, and StackOverflow-QA [\(Li](#page-12-15) [et al.,](#page-12-15) [2024\)](#page-12-15), where the model must identify relevant responses to StackOverflow questions, in a "hybrid" setting where documents contain both text and code. The latter benchmark also leverages longcontext capabilities, as its queries and documents respectively contain 1,400 and 1,200 words on average, leading to average token counts of over 2000.

We evaluate these benchmarks using the CoIR (CodeIR) framework [\(Li et al.,](#page-12-15) [2024\)](#page-12-15), as singlevector retrieval tasks. All models are trained by re-using the best hyper-parameters identified in Section [3.1.2.](#page-4-3)

## 3.2 Downstream Results and Discussion

Aggregated results for all evaluations are presented in Table [1.](#page-5-2) For BEIR and GLUE, the two common evaluation suites, we follow existing practice in reporting the average results. Detailed results are provided in Appendix [E.](#page-18-2)

In terms of downstream performance, Modern-BERT is the strongest overall model at both the BASE and LARGE model sizes. ModernBERT represents a Pareto improvement on all tasks over the original BERT and RoBERTA models, with better performance on every evaluation category.

<span id="page-5-0"></span><sup>10</sup>Spearheaded by [GitHub Copilot](https://github.com/features/copilot) in 2021

<span id="page-5-1"></span><sup>11</sup>Avoiding issues such as the ones seen in T5 [\(Raffel et al.,](#page-13-9) [2020\)](#page-13-9), whose vocabulary did not include curly braces.

<span id="page-6-0"></span>

Table 2: Memory (max batch size, *BS*) and Inference (in thousands of tokens per second) efficiency results on an NVIDIA RTX 4090, averaged over 10 runs. Dashes indicate unsupported configurations.

Short-Context Retrieval On BEIR, both variants of ModernBERT outperform existing encoders in both the DPR and ColBERT settings, including the recent GTE-en-MLM and NomicBERT models designed to serve as better backbones for retrieval [\(Zhang et al.,](#page-14-5) [2024;](#page-14-5) [Nussbaum et al.,](#page-12-6) [2024\)](#page-12-6).

While ModernBERT-base only narrowly edges out GTE-en-MLM-base on DPR evaluations, ModernBERT-large increases its lead despite having comparatively fewer parameters at 395M to GTE-en-MLM-large's 435M.

Long-Context Retrieval - Single Vector In the DPR setting, ModernBERT achieves impressive performance on MLDR, a long-context text retrieval task. However, these results also highlight an interesting phenomenon: without long-context finetuning ModernBERT outperforms both shortercontext models and the long-context NomicBERT but performs noticeably worse than GTE-en-MLM. The performance gap narrows considerably when evaluated in-domain, with both models performing similarly. This suggests that ModernBERT can effectively process long context sequences as a dense encoder but may require more adapted tuning. We plan to explore multiple potential explanations for this phenomenon in future work, including the impact of local attention or GTE-en-MLM having spent a larger part of its pretraining compute budget on longer sequence lengths [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5).

Long-Context Retrieval - Multi-Vector In the ColBERT setting, long-context models (GTE- en-MLM, NomicBERT, and ModernBERT) all outperform short-context models by at least 40 NDCG@10 points without requiring any specific finetuning. These results confirm the findings of [Bergum](#page-9-11) [\(2024\)](#page-9-11), who showed that ColBERT models are particularly well-suited to long-context retrieval tasks. Among the long-context models, Modern-BERT outperforms other long-context models, with at least a 9 NDCG@10 point lead on both model sizes. We theorize that these sizable gains could be explained by our long pretraining ensuring few, if any, tokens are under-trained, as well as a potentially synergistic effect of local attention with ColBERT-style retrieval, but leave further exploration of this phenomenon to future work.

Natural Language Understanding Both ModernBERT models demonstrate exceptional NLU results, as measured by GLUE. ModernBERTbase surpasses all existing base models, including DeBERTaV3-base, becoming the first MLMtrained model to do so. This is surprising, as DeBERTaV3 was trained with the Replaced-Token-Detection objective, which was previously thought to yield stronger downstream NLU performance [\(Clark et al.,](#page-10-12) [2020;](#page-10-12) [He et al.,](#page-11-12) [2023\)](#page-11-12). ModernBERT-large is the second-best large encoder on GLUE, almost matching DeBERTaV3 large with one-tenth fewer parameters while processing tokens in half the time (see Section [4\)](#page-7-0).

Code On programming tasks, in both code-totext (CodeSearchNet) and longer-context hybrid

settings (StackQA), ModernBERT outperforms all other models. This result was expected, as it is the only evaluated encoder to be trained on a data mixture including programming data. These results, combined with ModernBERT's strong showings on other tasks, indicates that ModernBERT has improved understanding of code at no detriment to its ability to process natural text.

### <span id="page-7-0"></span>4 Efficiency

## 4.1 Evaluation Setting

To measure inference efficiency across multiple sequence lengths, we create 4 synthetic sets of 8192 documents[12](#page-7-1). The first two document sets are fixed-length: in *fixed short-context*, all documents contain 512 tokens and in *fixed long-context* all documents contain 8192 tokens[13](#page-7-2). To account for the impact of unpadding, we also create two varying-length document sets, where the number of tokens in each set are defined by a normal distribution centered on half the maximum sequence length, 256 and 4096 tokens, respectively. Full data statistics are provided in Appendix [F.](#page-19-0)

We then evaluate all models based on the number of tokens they can process per second, averaged over ten runs. All efficiency evaluations are ran on a single NVIDIA RTX 4090, one of the target GPUs of ModernBERT outlined in Section [2.1.3](#page-2-1) We evaluate the GTE-en-MLM models under two settings: out-of-the box, and with the use of the xformers [\(Lefaudeux et al.,](#page-12-16) [2022\)](#page-12-16) library, which enables efficiency enhancements such as unpadding.

## 4.2 Results

All tokens-per-second efficiency results are presented in Table [2,](#page-6-0) with absolute run-times provided in Appendix [F.](#page-19-0) ModernBERT stands out as the most efficient model overall. On short context, it processes fixed-length 512 token inputs faster than all other recent encoders, although slower than the original BERT and RoBERTa models[14](#page-7-3). On longcontext, ModernBERT is faster than all competing encoders, processing documents 2.65 and 3 times faster than the next-fastest encoder at the BASE and LARGE sizes, respectively. ModernBERT-large's

processing speed at length 8192 (46,801 tokens per second) is closer to that of GTE-en-MLM base (47,507 tokens per second) than it is to GTE-en-MLM-large (16,532 tokens per second).

On variable-length inputs, both GTE-en-MLM and ModernBERT models are considerably faster than all other models, largely due to unpadding. However, ModernBERT remains noticeably more efficient than GTE-en-MLM, processing 14.5-30.9 percent more tokens per second at low context lengths and 98.8-118.8 percent more at longer context lengths, thanks to its use of local attention.

ModernBERT is the overall most memory efficient model on both model sizes. ModernBERTbase is able to process batch sizes twice as large as every other model on both input lengths. ModernBERT-large is slightly less memory efficient than the original BERT-large on short-context inputs, but can process batches at least 60 percent bigger than every other large model.

# 5 Conclusion

We present ModernBERT, an open family of encoder-only models which set a new state of the art over existing encoder models on a wide range of classification and retrieval tasks. We show that encoders benefit from both recent pretraining data scales and architecture improvements from autoregressive LLMs.

ModernBERT has a native sequence length of 8,192 tokens and incorporates recent architecture improvements, such as GeGLU layers, RoPE positional embeddings, and alternating local-global attention. ModernBERT is the first open model to feature full model unpadding and is the first encoder designed in a hardware-aware way to maximize inference efficiency.

ModernBERT pushes the encoder state of the art forward across a wide range of benchmarks. On GLUE, ModernBERT-base is the first encoder to beat DeBERTaV3-base since its release in 2021. ModernBERT is in a class of its own in code and ColBERT-style long-context retrieval benchmarks, scoring at least 6.85 and 9.1 percentage points higher than the closest model, respectively, while remaining state-of-the-art on short-context retrieval in both single and multi-vector settings.

At the same time, ModernBERT processes short context inputs twice as fast as DeBERTaV3 and long-context inputs two times faster than the next fastest model with best-in-class memory efficiency.

<span id="page-7-1"></span><sup>12</sup>Many common benchmarks are biased towards low and uniform sequence lengths, which is unrepresentative of many real-world situations.

<span id="page-7-2"></span><sup>13</sup>512 being the maximum length of most existing encoders, while 8192 is the maximum length of all long-context ones.

<span id="page-7-3"></span><sup>14</sup>This is partially due to the relatively low parameter count of BERT and RoBERTa compared to more recent encoders.

ModernBERT is a generational leap over the original encoder models, with notable performance improvements over BERT and RoBERTa on both classification and retrieval tasks. ModernBERT is one of the few encoders to support long-context and programming applications, while simultaneously setting a new record in encoder inference efficiency.

## 6 Limitations

Language This study focuses exclusively on the English language, and trains on a very large number of tokens. As such, a major limitation of our work is that it is not directly applicable to other languages, and potentially even less-so to lower resources languages. Exploration of modernizing encoder models in multilingual [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5) and monolingual but non-English [\(Antoun et al.,](#page-9-12) [2024\)](#page-9-12) settings is a promising avenue.

Biases Our model is trained largely on web data, as a result, all of its representations are subject to the biases present in such data.

Harmful Content Generation The MLM objective gives the model some ability to generate text by suggesting a given token to replace the [MASK] token [\(Samuel,](#page-13-15) [2024\)](#page-13-15), which could result in the generation of harmful content. However, Modern-BERT is not, primarily, a generative model, and as such, has not been trained to and therefore cannot generate longer sequences of text. As a result, it is considerably less likely to be at risk of generating harmful content of any kind.

MLM-only objective Given the strong results of DeBERTav3 on classification tasks but weak ones on retrieval, it seems that a training leveraging both MLM and RTD might be better suited to achieve best results on classification. Extending our work to RTD is thus a promising line of research.

Scaling Besides the architectural modifications, a key aspect of our studies is data scaling. However, other scaling axes, notably in terms of model parameters are left unexplored.

# 7 Acknowledgements

The authors would like to acknowledge & thank the many people who assisted, supported, or offered insights useful for the completion of this project.

We are particularly thankful for the one-off implementation or evaluation work conducted by Jack Cook, Mark Tenenholtz, Johno Whitaker, and Wayde Gilliam. We also extend similar thanks to

Zach Nussbaum for assisting in resolving issues we encountered with NomicBERT during evaluation.

We would like to acknowledge Enrico Shippole, Daniel Han, Colin Raffel, Pierre-Carl Langlais, Omar Khattab, Urchade Zaratiana, Aurélien Lac, Amélie Chatelain, and Raphaël Sourty, for their helpful contributions to discussions.

We also thank Weights&Biases for providing free access to their platform, in particular Morgan McGuire and Thomas Capelle for their support.

We thank HuggingFace's Arthur Zucker, Cyril Vallez, and Pedro Cuenca for assisting with dayone HuggingFace support.

Finally, we acknowledge Orange Business Cloud Avenue as compute provider and their hardware support throughout the project and thank LightOn for sponsoring the compute.

## 8 Contribution Statement

BW, AC, and BC jointly led the project and contributed to all parts of it.

BW worked on all aspects of the project and contributed to all major decisions. He led model design, model training, implemented the majority of the model architecture, and assisted with data selection, elevations, and paper writing.

AC co-initiated the project and worked on all aspects of it, including project coordination. Notably, he contributed to monitoring training runs and coled ablations, final evaluations and paper writing.

BC initiated the project and worked on all aspects of it. He contributed to model design and co-led final evaluations, led paper writing, and contributed

to the context extension data processing. OW led and conducted the majority of the data selection, processing, and discussion, for all stages of training. He also contributed valuable inputs throughout all stages of the project.

OH and ST contributed to a majority of the stages of the project, in particular model architecture and training, with both discussions, implementations and paper writing. Other contributions include pretraining monitoring, final traditional evaluations, and ablations. ST specifically worked on adapting the RoPE kernel for unpadded sequences and running the final GLUE benchmarks. OH additionally conducted a thorough investigation into complex issues that arose during training.

RB contributed greatly to the initial evaluation work, focusing on ablations and in-training evals. AG and FL contributed to training efficiency, espe-

<span id="page-9-14"></span><span id="page-9-13"></span><span id="page-9-12"></span><span id="page-9-11"></span><span id="page-9-10"></span><span id="page-9-9"></span><span id="page-9-8"></span><span id="page-9-7"></span><span id="page-9-6"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>

#### <span id="page-15-0"></span>A Training Settings

Detailed training settings can be found in Table [3.](#page-16-1)

During training we used MNLI as a live evaluation, along with validation loss and token accuracy metrics on a 500 million randomly sampled sequences from the source datasets.

We use [Composer](https://github.com/mosaicml/composer) [\(Mosaic ML Team,](#page-12-17) [2021\)](#page-12-17) as our training framework and [optim¯ı](https://github.com/search?q=optimi&type=repositories) [\(Warner,](#page-14-18) [2023\)](#page-14-18) for our optimizer implementations.

#### <span id="page-15-1"></span>A.1 Batch Size Schedule

Batch size warmup is a common-knowledge trick to speed up model training when working with medium to large batch sizes. Instead of "wasting" a full batch on updating the suboptimal initial weight distribution, we update the model weights on a gradually increasing batch size. Batch size warmup is usually longer than learning rate warmup, and can be thought of as providing a higher initial learning rate with a mini-learning rate decay to the defined learning rate schedule. We warmup ModernBERT's batch size from 768 to 4,608 over 50 billion tokens and from 448 to 4,928 over 10 billion tokens, for -base and -large, respectively, with an uneven token schedule so each batch size has the same number of update steps.

#### A.2 Full Model Unpadding & Sequence Packing

Naively padding short sequences to a fixed length wastes compute on padding tokens. "Mixed packing" strategies pack multiple sequences with separator tokens to improve token efficiency, but at a cost to model quality due to cross-contamination where tokens from one sequence attend to another [\(Krell](#page-12-10) [et al.,](#page-12-10) [2022\)](#page-12-10). "Structured packing" strategies can mitigate but not eliminate this risk by packing sequences which are semantically similar or from the same source [\(Staniszewski et al.,](#page-13-16) [2025\)](#page-13-16).

Anti-contamination strategies (also known as "unpadding methods") modify the model itself to enforce sequence independence [\(Zeng et al.,](#page-14-8) [2022\)](#page-14-8). These methods either involve delicate integration with the model architecture. Intra-document masking requires adjusting the attention mask or using a boundary-aware attention kernel, like FlashAttention [\(Zhao et al.,](#page-14-19) [2024;](#page-14-19) [Dao,](#page-10-8) [2023\)](#page-10-8), or only apply to a subset of model layers.

ModernBERT uses full, sequence independent model unpadding, combined with online sequence packing. Unlike existing approaches which repeat-

edly unpad and repad tensors at different layers (causing "padding thrashing"), we unpad once before the tokenizer embedding layer, for any unpacked data, and process everything through unpadded paths, only repadding when required by specific model heads. This full model unpadding alone provided approximately 10-20% training speedup.

For sequence independent packing, we developed a Numba [\(Lam et al.,](#page-12-18) [2015\)](#page-12-18)-optimized, greedy best-fit, online sequence packing algorithm that runs dynamically within the training loop, leveraging FlashAttention's jagged tensor support. This increased token utilization efficiency from approximately 60% to over 99%, providing an additional 10% training improvement. The combination enables efficient, contamination-free training while maintaining the computational benefits of packed sequences.

#### <span id="page-15-2"></span>A.3 Weight Tiling

Following the Phi family of models [\(Li et al.,](#page-12-13) [2023;](#page-12-13) [Javaheripi et al.,](#page-11-11) [2023\)](#page-11-11), we initialized ModernBERT-large directly from ModernBERTbase's pretraining weights using center tiling and Gopher layer scaling [\(Rae et al.,](#page-12-19) [2022\)](#page-12-19). Since Base's weight matrices are smaller than Large's, we centered Base' weights, accounting for each token embedding and attention head, then filled rest the of the weights using wraparound. Like Phi, we tested center initialization with random edge values and tiling from an edge, but both of these underperformed center tiling with wraparound. This weight initialization strategy greatly accelerates ModernBERT-large's initial training.

### A.4 Weight Decay

We did not apply weight decay to the bias terms or normalization layers. Instead of PyTorch-style decoupled weight decay, we applied fully decoupled weight decay following [Loshchilov and Hutter](#page-12-12) [\(2019\)](#page-12-12).

## A.5 Final Checkpoints

Inspired by recent work showing that checkpoint averaging yields stronger final models [\(Dubey](#page-10-1) [et al.,](#page-10-1) [2024;](#page-10-1) [Clavié,](#page-10-11) [2024\)](#page-10-11), we selected our final checkpoints by experimenting with various averaging methods and evaluating them on a subset of evaluation tasks. In no cases did Exponential Moving Average during annealing, as used by [Dubey et al.](#page-10-1) [\(2024\)](#page-10-1), result in stronger performance. ModernBERT-base is the result of averaging the 3

<span id="page-16-1"></span>

Table 3: ModernBERT training settings. Dropout and below are shared across all phases.

best performing annealing checkpoints with the final one. Averaging did not yield successful results on the large size, ModernBERT-Large model is the best performing annealing checkpoint.

# <span id="page-16-0"></span>B Model Design

From [Anthony et al.](#page-9-2) [\(2024\)](#page-9-2), in addition to setting attention heads as multiples of 64 and setting the embedding matrix as a power of 2 or multiple of 64, there are three model design choices to maximize performance (assuming float16 or bfloat16 computation):

- Tensor Core Requirement: Weight matrix dimensions should be divisible by 64
- Tile Quantization: Weight matrix is divisible into 128 × 256 blocks.
- Wave Quantization: Number of blocks is divisible by the number of streaming multiprocessors (SM).

Given that we wanted to target good performance across multiple GPUs with a wide variety of SM counts, wave quantization is an impossible ask. So we selected a basket of GPUs (NVIDIA T4, A10, L4, RTX 3090, RTX 4090, A100, and H100) and calculated the approximate SM utilization for each

by dividing the modulus blocks by the number of SMs. This appeared to be a decent performance heuristic in our spot checking. We then designed our models to maximize performance on the basket of GPUs, putting more weight on inference GPUs.

## C Training Log

## C.1 Sampling Issue

Our first pretraining run of ModernBERT-base ended in disaster as the loss exhibited a slow seesaw pattern before slowly diverging. Despite using PyTorch's distributed random sampler, training metrics suggested that the model was training on the dataset in a non-random order. Like the Olmo authors[15](#page-16-2), we determined that the PyTorch random sampler returns sequentially biased samples when the number of samples is somewhere between 500 million and 1 billion samples[16](#page-16-3). We resolved this issue by replacing the PyTorch sampler with NumPy's PCG64DXSM random sampler.

<span id="page-16-2"></span><sup>15</sup>We found a comment and GitHub issue about this in the Olmo codebase after resolving the issue ourselves.

<span id="page-16-3"></span><sup>16</sup>We did not conduct a rigorous statistical analysis to determine exactly when this happens.

Table 4: ModernBERT model design

### C.2 Large Rollback

We rolled back and restarted ModernBERT-large training at a lower learning rate of 5e-5 and lower weight decay of 1e-6 for the last 800 billion tokens. Prior to restarting training, large's training loss, validation metrics, and live evaluations on MNLI had plateaued for a few hundred billion tokens at the higher 5e-4 learning rate. In contrast, ModernBERT-base showed a continuous, but diminishing, improvement on training loss, validation metrics, and live evaluations through the entire 1.719 trillion token training phase. This highlights one of the risks of training with a constant learning rate, other learning rate schedules can mitigate selecting a too high learning rate (or too small batch size) by lowering the learning rate throughout training.

## <span id="page-17-0"></span>D Ablations

To select the training settings and updates to add to the ModernBERT architecture, we performed multiple ablations. Except where stated, ablations were ran at the 8-20 billion token scale using a 150M parameter or smaller model.

# D.1 Architecture ablations

- We compared two GLU layers, GeGLU and SwiGLU. We find close to no difference between the two and choose to use GeGLU layers.
- Using different percentage of the head dimension for the RoPE dimension (50, 75, 100). Lower percentages gave slightly better results.

However, the observed difference was minimal. As the ablations were conducted at a considerably smaller scale than the final training, we choose to err on the side of caution and opt to keep the dimension at 100 % to avoid potentially hindering the capabilities of the fully trained models.

- Both LayerNorm and RMSNorm yielded similar results. While RMSNorm is theoretically faster, at the time this work was conducted, PyTorch did not have a native RMSNorm implementation, leading to eager-mode RM-SNorm being the default implementation used for many users. To ensure ModernBERT has the highest possible out-of-the-box efficiency, we choose to use LayerNorm in the final models.
- We investigated using parallel attention to compute the MLP and attention matrices at the same time, including a mixture of parallel and prenorm, which has been shown to increase processing speeds for larger model sizes [\(Chowdhery et al.,](#page-9-13) [2023\)](#page-9-13). However, for models within our targe sizes and pre-training sequence length, the speed-up we observed was minimal while we encountered significant degradation in downstream performance. As such, we do not use parallel attention. However, it is possible that larger encoders and/or larger sequence lengths might see a different trade-off.
- We explored the use of alternating global/local

attention, with global attention every 3 layers and local attention over a 128 token sliding window otherwise. This setup yielded identical downstream performance when compared to the use of global attention in every layer, even at 100 billion tokens, providing a major speedup.

- We tried multiple variations global-local RoPE θ, from 500 to 5K local, and 2K to 160K global. We found no discernible difference and selected our unified 2K RoPE θ for pretraining and increased the global RoPE θ to 160K during context extension.
- We observed a potential improvement with Gemma-2 tanh softcapping [\(Gemma Team](#page-10-4) [et al.,](#page-10-4) [2024\)](#page-10-4), but elected to forgo softcapping due to downstream compatibility concerns.
- We experimented with multiple tokenizers, before selecting our final one, based on a modified OLMo [\(Groeneveld et al.,](#page-11-6) [2024\)](#page-11-6) tokenizer, which performed the best of the recent tokenizers evaluated. Tokenizers from the BERT and RoBERTa generation of encoder models had competitive downstream performance on MNLI, but we theorized that their lack of recent training data and lack of code support would hinder downstream applications. We observed significant downstream performance degradation when using the Llama 2 [\(Touvron et al.,](#page-13-1) [2023\)](#page-13-1) tokenizer.

## D.2 Training ablations

- We compared AdamW [\(Loshchilov and Hut](#page-12-12)[ter,](#page-12-12) [2019\)](#page-12-12) and StableAdamW [\(Wortsman et al.,](#page-14-11) [2023\)](#page-14-11) optimizers across a variety of ablation settings. We discovered that while StableAdamW initially underperformed AdamW, it avoided loss spikes better than AdamW with gradient clipping, and after a few billion tokens of training the two were nearly identical. We selected StableAdamW due to its increased training stability over AdamW.
- We experimented with a range of batch sizes, from 4K to 16K, and found that 4-5K appeared to be a sweet spot between wallclock and downstream performance, Exact batch sizes were selected due to microbatch memory constraints.

- We tested both the trapezoidal Learning Rate (LR) schedule [\(Xing et al.,](#page-14-12) [2018\)](#page-14-12) and the Cosine Inverse Square Root LR schedule introduced by Stable LM 2 [\(Bellagente et al.,](#page-9-14) [2024\)](#page-9-14) up to 100B tokens. We found that after decay, the final results were near indistinguishable and selected the trapezoidal schedule due it's potential ease of continual pretraining.
- Following OLMo [\(Groeneveld et al.,](#page-11-6) [2024\)](#page-11-6), we tested multiple model initialization methods, including PyTorch's default, normal, Kaiming normal [\(He et al.,](#page-11-17) [2015\)](#page-11-17), fan-in variance scaling, a truncated normal distribution with an adaptive standard deviation [\(Groen](#page-11-6)[eveld et al.,](#page-11-6) [2024\)](#page-11-6), and Megatron initialization [\(Shoeybi et al.,](#page-13-8) [2019\)](#page-13-8). Details for each method can be seen in our code. We selected Megatron initialization for ModernBERT-base as it performed best during our ablations. For ModernBERT-large we found that Phi-style weight initialization outperformed random init on 50B token ablations.
- We experimented with training with exponential moving average (EMA) of our model weights, but did not find improvement with EMA windows of 1K or 5K steps.

## <span id="page-18-2"></span>E Extended results

# <span id="page-18-0"></span>E.1 Full GLUE results

The results for all the models each GLUE subsets are presented in Table [5.](#page-19-1) The values for prior models are extracted from the literature. As mentioned in Section [3.1.1,](#page-4-4) we follow standard practice [\(Liu](#page-12-11) [et al.,](#page-12-11) [2019a;](#page-12-11) [Portes et al.,](#page-12-5) [2023;](#page-12-5) [He et al.,](#page-11-12) [2023\)](#page-11-12) and conduct an hyperparameter search on each subset. More specifically, we perform a sweep over learning rates in [1e−5, <sup>3</sup>e−5, <sup>5</sup>e−5, <sup>8</sup>e−5], weight decay in [1e−6, <sup>5</sup>e−6, <sup>8</sup>e−6, <sup>1</sup>e−5], and number of epochs in [1, 2, 3] for tasks in SST-2, MNLI, and RTE, and [2, 5, 10] for tasks in QNLI, QQP, CoLA, MRPC, and STS-B. The final values are detailed in Table [6.](#page-19-2) Early stopping is used for all the fine-tuning runs which reduces the overall fine-tuning time considerably. RTE MRPC and STS-B checkpoints are trained starting from the MNLI checkpoint. The batch size is set to 64.

# <span id="page-18-1"></span>E.2 Full BEIR results

In the main body, we only report the average score over the 15 very diverse datasets of BEIR. We

<span id="page-19-1"></span>

<span id="page-19-2"></span>Table 5: GLUE [\(Wang et al.,](#page-14-14) [2018\)](#page-14-14) dev set scores. <sup>α</sup> taken from Table 8 of [\(Liu et al.,](#page-12-11) [2019a\)](#page-12-11), <sup>β</sup> taken from Table S3 of [\(Portes et al.,](#page-12-5) [2023\)](#page-12-5), <sup>γ</sup> from Table 2 of [\(Nussbaum et al.,](#page-12-6) [2024\)](#page-12-6), <sup>δ</sup> from Table 21 of [\(Zhang et al.,](#page-14-5) [2024\)](#page-14-5), <sup>ϵ</sup> from Table 2 of [\(Qiang et al.,](#page-12-20) [2024\)](#page-12-20) and <sup>ζ</sup> from Table 3 of [\(He et al.,](#page-11-12) [2023\)](#page-11-12)

Table 6: Fine-tuning hyperparameters for ModernBERT on GLUE tasks. LR: Learning Rate, WD: Weight Decay, Ep: Epochs.

report the results on every subsets for both single and multi-vector retrieval in Table [7](#page-20-1) and Table [8](#page-20-2) respectively. For both settings and for every model, we perform a sweep for learning rates in [1e−5, <sup>2</sup>e−5, <sup>3</sup>e−5, <sup>5</sup>e−5, <sup>8</sup>e−5, <sup>1</sup>e−4] with a learning rate warmup for 5% of the training and choose the model obtaining the best average result over a subset of datasets composed of NFCorpus, SciFact, TREC-Covid and FiQA as the final model. Best learning rates for every setting are reported in Table [9.](#page-20-3) For the single-vector setup, the batch size is set to 64 and 16 and gradient accumulation to 8 and 32 for base and large sizes respectively. For the multi-vector setup, the batch size is set to 8 and 4 and gradient accumulation to 2 and 4 for base and large sizes respectively. Although ModernBERT showcase strong results across the board, it should be noted that an important factor in its performance is TREC-COVID [\(Voorhees et al.,](#page-13-17)

[2021\)](#page-13-17), potentially showcasing the benefits of ModernBERT being trained with a more recent knowledge cutoff than most existing encoders. However, NomicBERT and GTE have also been trained on updated data, so the cutoff cannot be the only factor affecting the performance.

# <span id="page-19-0"></span>F Efficiency

Full statistics of the synthetic datasets used to evaluate the efficiency of the models in Section [4](#page-7-0) are given in Table [10.](#page-21-0) The detailed runtimes, alongside with the maximum batch size for every model is detailed in Table [11.](#page-21-1)

The high maximum batch-size achieved by ModernBERT models, considerably higher than any other models, highlight the strong memory efficiency of the model at both sizes. Inversely, it is worth noting that while DeBERTaV3 has competitive GLUE performance, it stands out as par-

<span id="page-20-1"></span>

Table 7: BEIR [\(Thakur et al.,](#page-13-11) [2021\)](#page-13-11) nDCG@10 scores for single-vector retrieval models.

<span id="page-20-2"></span>

<span id="page-20-3"></span>Table 8: BEIR [\(Thakur et al.,](#page-13-11) [2021\)](#page-13-11) nDCG@10 scores for multi-vector retrieval models.

Table 9: Learning rate used for reported results on BEIR [\(Thakur et al.,](#page-13-11) [2021\)](#page-13-11) for both single and multi vector retrieval

ticularly inefficient, both in its memory use and processing speed. Indeed, on both model sizes, De-BERTaV3's memory use is 5-to-7 times higher than ModernBERT's, and it processes inputs two times slower, even in the most favorable scenario where all sequences are at the maximum possible length, thus negating any advantage from unpadding.

## <span id="page-20-0"></span>G Loss Curves

Figure [1](#page-21-2) presents the loss curves for both models, across all three phases of training.

## H Licensing

We release the ModernBERT model architecture, model weights, and training codebase under the Apache 2.0 license.

<span id="page-21-0"></span>

Table 10: Token statistics for the synthetic datasets used in efficiency evaluations.

<span id="page-21-1"></span>

Table 11: Inference runtime for all models. Bold indicates the best for the column within two SDs.

<span id="page-21-2"></span>![](_page_21_Figure_4.jpeg)

Figure 1: Training loss curves.