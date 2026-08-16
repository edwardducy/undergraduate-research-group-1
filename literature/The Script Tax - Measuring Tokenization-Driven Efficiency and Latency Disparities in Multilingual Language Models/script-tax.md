# The Script Tax: Measuring Tokenization-Driven Efficiency and Latency Disparities in Multilingual Language Models

Aradhya Dixit

Wake Technical Community College adixit1@my.waketech.edu

Shreem Dixit

University of North Carolina Charlotte sdixit6@charlotte.edu

# Abstract

Pretrained multilingual language models are often assumed to be script-agnostic, yet their tokenizers can impose systematic costs on certain writing systems. We quantify this *script tax* by comparing two orthographic variants with identical linguistic content. Across mBERT and XLM-R, the higher-fragmentation orthography shows a ∼3.4× increase in fertility (6.73– 6.85 vs. 2.10–2.35 tokens/word), leading to a 16.5× inference slowdown (0.23 vs. 3.8 sentences/second) on identical hardware. Using bits per character (BPC) to avoid the "NLL paradox" from subword fragmentation, we find a substantial increase in information cost: +19.7% for mBERT (8.06→9.65) and +47.1% for XLM-R (12.19→17.94). A round-trip conversion check (CERrt=0.31) suggests these gaps reflect orthography-conditioned processing rather than mapping noise. Our results highlight tokenization as a key source of inequity in multilingual NLP and motivate script-aware tokenization and pretraining.

# 1 Introduction

Large multilingual language models are often treated as *script-agnostic*: if two inputs express the same linguistic content, we might expect comparable model quality and comparable inference cost regardless of the writing system [\(Ahia et al.,](#page-4-0) [2023;](#page-4-0) [Petrov et al.,](#page-4-1) [2023\)](#page-4-1). In practice, most multilingual models rely on a single pretrained subword tokenizer whose vocabulary and segmentation behavior can differ drastically across scripts [\(Lim](#page-4-2)[isiewicz et al.,](#page-4-2) [2023;](#page-4-2) [Petrov et al.,](#page-4-1) [2023\)](#page-4-1). When a tokenizer fragments one script into many more pieces than another, the model is forced to process longer sequences for the same content, increasing memory use, compute cost, and latency—and potentially degrading modeling efficiency by obscuring word- and morpheme-level structure [\(Ahia](#page-4-0) [et al.,](#page-4-0) [2023;](#page-4-0) [Limisiewicz et al.,](#page-4-2) [2023\)](#page-4-2).

We refer to this systematic penalty as a *script tax*: a writing-system-dependent increase in (i) tokenization cost, (ii) computational cost, and (iii) information cost [\(Ahia et al.,](#page-4-0) [2023;](#page-4-0) [Petrov et al.,](#page-4-1) [2023\)](#page-4-1). The core mechanism is tokenization fragmentation, which we quantify with *fertility* (tokens per word). Higher fertility inflates the effective sequence length, which can induce disproportionate slowdowns in Transformer inference due to attention costs that scale superlinearly with sequence length [\(Ahia et al.,](#page-4-0) [2023;](#page-4-0) [Petrov et al.,](#page-4-1) [2023\)](#page-4-1). Importantly, standard log-likelihood metrics can be misleading under severe fragmentation: predicting many small subword fragments can yield deceptively low token-level loss even when the model is less efficient at representing the underlying information [\(Limisiewicz et al.,](#page-4-2) [2023;](#page-4-2) [Petrov et al.,](#page-4-1) [2023\)](#page-4-1). To address this, we evaluate modeling efficiency using bits per character (BPC), normalizing the loss by the number of characters rather than the number of tokens.

In this work, we instantiate the script tax using paired sentence sets that preserve identical linguistic content across two orthographic variants and evaluate two widely used multilingual masked language models (mBERT and XLM-R). We observe a large fertility gap: 6.73–6.85 tokens/word for the higher-fragmentation orthography versus 2.10–2.35 for the lower-fragmentation orthography (a ∼3.4× increase). This tokenization disparity translates into a substantial runtime penalty on identical hardware, reducing throughput from ∼3.8 to ∼0.23 sentences/second (16.5× slower). After normalizing by characters, we find a pronounced information-cost increase: BPC rises by +19.7% for mBERT (8.06→9.65) and +47.1% for XLM-R (12.19→17.94). A round-trip conversion robustness check (CERrt=0.31) suggests these differences are not artifacts of mapping instability, but reflect orthography-conditioned processing induced by pretrained tokenizers.

<span id="page-1-0"></span>![](_page_1_Diagram_0.jpeg)

Figure 1: Evaluation pipeline used to measure the *script tax*. We compare paired sentences across orthographic variants and compute (i) tokenization fertility (tokens/word), (ii) modeling efficiency via BPC (loss normalized by character count), and (iii) inference latency/throughput on identical hardware.

Our contributions are:

- We define and operationalize the script tax as a joint disparity in fertility, latency, and character-normalized information cost (BPC) under controlled, content-matched evaluation.
- We show that tokenization fragmentation is a first-order driver of both compute overhead (16.5× slowdown) and reduced modeling efficiency (up to +47.1% BPC), even when tokenlevel NLL appears favorable.
- We provide a simple robustness check (roundtrip CER) that helps separate script-induced effects from conversion noise.

### <span id="page-1-1"></span>2 Methodology

Figure [1](#page-1-0) summarizes our pipeline. We quantify the *script tax* along three axes: tokenization fragmentation, inference overhead, and character-normalized information cost.

# 2.1 Paired setup

Let D = {(x (n) A , x (n) B )} N <sup>n</sup>=1 be paired sentences with identical content in orthographies A and B. For model m with tokenizer τm(·), define token length Lm(x) = |τm(x)| and word count W(x) (whitespace tokens).

### 2.2 Fragmentation (fertility)

We measure fragmentation via fertility (tokens/word):

<sup>F</sup>m(x) = <sup>L</sup>m(x) W(x) , ∆F<sup>m</sup> = 1 N X n Fm(x (n) B ) − Fm(x (n) A ) . (1)

#### 2.3 Information cost (BPC)

Token-level NLL can be biased by subword fragmentation, so we normalize by characters. Let C(x) be the number of Unicode characters (excluding spaces). For a fixed masking scheme, let NLLm(x) be the average masked-token loss in nats. We report bits per character:

BPCm(x) = NLLm(x) log 2 · |M(x)| C(x) , δBPC<sup>m</sup> = BPC(B) <sup>m</sup> <sup>−</sup> BPC(A) m BPC(A) m . (2)

### 2.4 Compute cost (latency)

Let Tm(x) be measured inference time under fixed hardware and settings. We report median latency and the latency tax:

Lat(s) <sup>m</sup> = median<sup>n</sup> Tm(x (n) s ), ρlat <sup>m</sup> = Lat(B) m Lat(A) m . (3)

Sequence inflation increases attention cost roughly quadratically:

Cost(x) ∝ Lm(x) <sup>2</sup> ⇒ Cost(xB) Cost(xA) ≈ Lm(xB) Lm(xA) 2 . (4)

# 2.5 Robustness

We validate conversion stability using round-trip CER. With maps πA→<sup>B</sup> and πB→A, let xˆ<sup>A</sup> = πB→A(πA→B(xA)). Using edit distance ED:

CERrt = 1 N X n ED(x (n) A , xˆ (n) A ) C(x (n) A ) . (5)

We summarize the script tax per model as (∆Fm, ρlat <sup>m</sup> , δBPCm).

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

<span id="page-2-1"></span>![](_page_2_Figure_1.jpeg)

Figure 2: Tokenization bottleneck (fertility, tokens/word). The higher-fragmentation orthography requires substantially more tokens per word across both mBERT and XLM-R.

# 3 Results

We report results for two multilingual masked language models (mBERT and XLM-R) under the paired evaluation described in Section [2.](#page-1-1) All measurements are computed on matched sentence pairs and identical inference settings.

### 3.1 Tokenization bottleneck: fertility gap

Figure [2](#page-2-0) shows a large and consistent fragmentation disparity across models. The higherfragmentation orthography requires 6.73–6.85 tokens/word, while the lower-fragmentation orthography requires 2.10–2.35 tokens/word, yielding a gap of +4.38 to +4.75 tokens/word (∼3.4× longer sequences). This indicates that the tokenizer is effectively operating closer to character-level segmentation for one orthography, which inflates sequence length and drives downstream compute overhead.

# 3.2 Computational overhead: latency tax

Sequence inflation yields a substantial runtime penalty. Measured on identical hardware, throughput drops from ∼3.8 to ∼0.23 sentences/second, corresponding to a 16.5× slowdown for the higherfragmentation orthography. This is consistent with the superlinear dependence of Transformer inference cost on sequence length (Section [2\)](#page-1-1), where longer token sequences amplify attention and memory overhead.

Figure 3: The script tax as a joint disparity in information cost and runtime. The higher-fragmentation orthography exhibits both higher BPC (worse; ↓ better) and substantially higher inference latency.

### 3.3 Information cost: BPC "script tax"

Figure [3](#page-2-1) plots the joint effect of information cost (BPC; lower is better) and inference latency. Character-normalized efficiency degrades substantially for the higher-fragmentation orthography: BPC increases by +19.7% for mBERT (8.06→9.65) and by +47.1% for XLM-R (12.19→17.94). This confirms that the apparent token-level "ease" sometimes observed under fragmentation is an artifact of predicting smaller units; when normalized to characters, the model is less efficient at representing the same underlying content.

#### 3.4 Summary

Across both models, tokenization fragmentation is the dominant upstream driver: a ∼3.4× increase in sequence length produces a 16.5× slowdown and a large rise in character-normalized information cost (up to +47.1% BPC). Together, these results demonstrate that pretrained tokenizers can introduce structural disparities in both computational accessibility and modeling efficiency.

# 4 Discussion and Implications

Our results show that tokenization is not a neutral preprocessing choice in multilingual LMs: it can create a large, systematic *cost and quality disparity* between orthographic variants with identical linguistic content [\(Ahia et al.,](#page-4-0) [2023;](#page-4-0) [Petrov et al.,](#page-4-1) [2023;](#page-4-1) [Limisiewicz et al.,](#page-4-2) [2023\)](#page-4-2).

#### 4.1 One upstream cause, multiple downstream taxes

The fertility gap (Figure [2\)](#page-2-0) is the dominant upstream driver: because the higher-fragmentation orthography expands sequence length by ∼3.4×, even modest superlinear scaling in the forward pass can translate into large slowdowns, which we observe as a 16.5× throughput drop. This gap is not a "deployment detail": it directly affects user experience (latency) and provider cost (compute).

#### 4.2 Why token-level loss can be misleading (and BPC fixes it)

Methodologically, fragmentation can produce an *NLL paradox*: predicting many short fragments can reduce per-token loss while still representing the same content less efficiently. BPC counters this by normalizing by characters, making the comparison invariant to how aggressively the tokenizer splits; the observed BPC increases (+19.7% for mBERT and +47.1% for XLM-R; Figure [3\)](#page-2-1) therefore indicate a genuine degradation in information efficiency, not a metric artifact.

# 4.3 Robustness and interpretation

The round-trip CER check (CERrt = 0.31) is not perfect reconstruction, but it provides evidence that the paired setup is not dominated by conversion noise; importantly, the same orthography that yields higher fertility also yields both higher latency and higher BPC, supporting the interpretation that the observed gaps are primarily orthographyconditioned processing effects rather than dataset artifacts.

# 4.4 Practical implications

Practically, these findings suggest two lessons [\(Ahia et al.,](#page-3-0) [2024;](#page-3-0) [Liang et al.,](#page-4-3) [2023;](#page-4-3) [Tao](#page-5-0) [et al.,](#page-5-0) [2024\)](#page-5-0): (i) script-aware preprocessing/tokenization matters—for underrepresented scripts, adapting the tokenizer (e.g., vocabulary augmentation or re-tokenization) may provide large gains in both inference cost and modeling efficiency without changing the model architecture [\(Ahia et al.,](#page-3-0) [2024;](#page-3-0) [Liang et al.,](#page-4-3) [2023;](#page-4-3) [Tao et al.,](#page-5-0) [2024\)](#page-5-0); and (ii) report compute-aware metrics evaluations that only report token-level perplexity/NLL risk hiding real inequities, whereas reporting fertility, character-normalized loss (BPC), and latency gives a more faithful picture of accessibility and performance [\(Ahia et al.,](#page-4-0) [2023;](#page-4-0) [Petrov et al.,](#page-4-1) [2023;](#page-4-1) [Ramesh et al.,](#page-4-4) [2023\)](#page-4-4). Overall, the script tax

is best understood as a structural property of pretrained tokenizers: it creates a predictable mapping from sequence inflation → compute penalty → degraded character-normalized efficiency, raising both fairness and efficiency concerns for multilingual NLP.

### 5 Conclusion

We introduced the *script tax*, a tokenization-driven disparity in both compute cost and modeling efficiency between orthographic variants with the same linguistic content. Across mBERT and XLM-R, we observe a large fertility gap (6.73–6.85 vs. 2.10– 2.35 tokens/word) that induces a 16.5× inference slowdown on identical hardware, and a substantial increase in character-normalized information cost (BPC): +19.7% for mBERT and +47.1% for XLM-R. A round-trip CER check suggests these effects are not dominated by conversion noise.

Practically, our findings imply that reporting only token-level loss can obscure real disparities created by tokenizer fragmentation, and that character-normalized metrics such as BPC should be paired with compute measures (latency/throughput) when assessing multilingual model quality. More broadly, these results motivate script-aware interventions—such as tokenizer adaptation or script-balanced pretraining—as straightforward ways to reduce both the inference burden and the representational inefficiency imposed on higher-fragmentation scripts [\(Ahia et al.,](#page-3-0) [2024;](#page-3-0) [Liang et al.,](#page-4-3) [2023;](#page-4-3) [Tao et al.,](#page-5-0) [2024;](#page-5-0) [Xue et al.,](#page-5-1) [2022\)](#page-5-1).

# 6 Limitations

Our analysis is limited to two orthographic variants and two multilingual masked language models (mBERT and XLM-R), so the magnitude of the script tax may differ for other scripts, tokenizers, and model families. The paired setup relies on an orthography conversion pipeline; while round-trip CER (CERrt) offers a robustness check, residual conversion artifacts may still influence results. Finally, latency numbers depend on a specific hardware/inference configuration, though the relative penalties from sequence inflation should generalize qualitatively.

# References

<span id="page-3-0"></span>Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Valentin Hofmann, Tomasz Limisiewicz, Yulia Tsvetkov, and

<span id="page-4-4"></span><span id="page-4-3"></span><span id="page-4-2"></span><span id="page-4-1"></span><span id="page-4-0"></span>Noah A. Smith. 2024. MAGNET: Improving the multilingual fairness of language models with adaptive gradient-based tokenization. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David Mortensen, Noah Smith, and Yulia Tsvetkov. 2023. [Do all languages cost the same?](https://doi.org/10.18653/v1/2023.emnlp-main.614) [tokenization in the era of commercial language mod](https://doi.org/10.18653/v1/2023.emnlp-main.614)[els.](https://doi.org/10.18653/v1/2023.emnlp-main.614) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 9904–9923, Singapore. Association for Computational Linguistics. Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2023. When is multilinguality a curse? language modeling for 250 high- and low-resource languages. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. Thao Anh Dang, Limor Raviv, and Lukas Galke. 2025. Tokenization and morphology in multilingual language models: A comparative analysis of mT5 and ByT5. In *Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP 2025)*, pages 242–257, Odense, Denmark. Association for Computational Linguistics. Lukas Edman, Gabriele Sarti, Antonio Toral, Gertjan van Noord, and Arianna Bisazza. 2024. [Are](https://doi.org/10.1162/tacl_a_00651) [character-level translations worth the wait? compar](https://doi.org/10.1162/tacl_a_00651)[ing ByT5 and mT5 for machine translation.](https://doi.org/10.1162/tacl_a_00651) *Transactions of the Association for Computational Linguistics*, 12:392–410. Masahiro Kaneko, Aizhan Imankulova, Danushka Bollegala, and Naoaki Okazaki. 2022. [Gender bias in](https://doi.org/10.18653/v1/2022.naacl-main.197) [masked language models for multiple languages.](https://doi.org/10.18653/v1/2022.naacl-main.197) In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2740–2750, Seattle, United States. Association for Computational Linguistics. Sharon Levy, Neha A. John, Ling Liu, Yogarshi Vyas, Jie Ma, Yoshinari Fujinuma, Miguel Ballesteros, Vittorio Castelli, and Dan Roth. 2023. Comparing biases and the impact of multilingual training across multiple languages. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 10260–10280. Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa. 2023. [XLM-V: Over](https://doi.org/10.18653/v1/2023.emnlp-main.813)[coming the vocabulary bottleneck in multilingual](https://doi.org/10.18653/v1/2023.emnlp-main.813) [masked language models.](https://doi.org/10.18653/v1/2023.emnlp-main.813) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 13142–13152, Singapore. Association for Computational Linguistics. Tomasz Limisiewicz, Jiˇrí Balhar, and David Marecek. ˇ 2023. [Tokenization impacts multilingual language](https://doi.org/10.18653/v1/2023.findings-acl.350) [modeling: Assessing vocabulary allocation and over](https://doi.org/10.18653/v1/2023.findings-acl.350)[lap across languages.](https://doi.org/10.18653/v1/2023.findings-acl.350) In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 5661–5681, Toronto, Canada. Association for Computational Linguistics. Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Üstün, Sara Hooker, and Sebastian Ruder. 2024a. How does quantization affect multilingual LLMs? In *Findings of the Association for Computational Linguistics: EMNLP 2024*. Kelly Marchisio, Wei-Yin Ko, Alexandre Bérard, Théo Dehaze, and Sebastian Ruder. 2024b. Understanding and mitigating language confusion in LLMs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 6653– 6677. Kurt Micallef, Fadhl Eryani, Nizar Habash, Houda Bouamor, and Claudia Borg. 2023. Exploring the impact of transliteration on NLP performance: Treating Maltese as an Arabic dialect. In *Proceedings of the Workshop on Computation and Written Language (CAWL 2023)*, pages 22–32, Toronto, Canada. Association for Computational Linguistics. Gabriel Nicholas and Aliya Bhatia. 2023. Lost in translation: Large language models in non-english content analysis. *arXiv:2306.07377*. Elizabeth Nielsen, Christo Kirov, and Brian Roark. 2023. Distinguishing Romanized Hindi from Romanized Urdu. In *Proceedings of the Workshop on Computation and Written Language (CAWL 2023)*, pages 33–42, Toronto, Canada. Association for Computational Linguistics. Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. 2023. Language model tokenizers introduce unfairness between languages. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. Krithika Ramesh, Sunayana Sitaram, and Monojit Choudhury. 2023. Fairness in language models beyond english: Gaps and challenges. In *Findings of the Association for Computational Linguistics: EACL 2023*, pages 2061–2074. Chenglei Si, Zhengyan Zhang, Yingfa Chen, Fanchao Qi, Xiaozhi Wang, Zhiyuan Liu, Yasheng Wang, Qun Liu, and Maosong Sun. 2023. [Sub-character tok](https://doi.org/10.1162/tacl_a_00560)[enization for Chinese pretrained language models.](https://doi.org/10.1162/tacl_a_00560) *Transactions of the Association for Computational Linguistics*, 11:469–487. Jimin Sun, Patrick Fernandes, Xinyi Wang, and Graham Neubig. 2023. [A multi-dimensional evaluation](https://doi.org/10.18653/v1/2023.findings-eacl.128) [of tokenizer-free multilingual pretrained models.](https://doi.org/10.18653/v1/2023.findings-eacl.128) In *Findings of the Association for Computational Linguistics: EACL 2023*, pages 1725–1735, Dubrovnik, Croatia. Association for Computational Linguistics. Zeerak Talat, Aurélie Névéol, Stella Biderman, Miruna Clinciu, Manan Dey, Shayne Longpre, Sasha Luccioni, Maraim Masoud, Margaret Mitchell, Dragomir Radev, Shanya Sharma, Arjun Subramonian, Jaesung

Tae, Samson Tan, Deepak Tunuguntla, and Oskar Van Der Wal. 2022. You reap what you sow: On the challenges of bias evaluation under multilingual settings. In *Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models*, pages 26–41, virtual+Dublin. Association for Computational Linguis-

<span id="page-5-1"></span><span id="page-5-0"></span>tics. Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. 2024. Scaling laws with vocabulary: Larger models deserve larger vocabularies. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, and Donald Metzler. 2022. Charformer: Fast character transformers via gradientbased subword tokenization. In *Proceedings of the 10th International Conference on Learning Representations (ICLR 2022)*. Aniket Vashishtha, Kabir Ahuja, and Sunayana Sitaram. 2023. On evaluating and mitigating gender biases in multilingual settings. In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 307–318. Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022. [ByT5: Towards a token-free](https://doi.org/10.1162/tacl_a_00461) [future with pre-trained byte-to-byte models.](https://doi.org/10.1162/tacl_a_00461) *Transactions of the Association for Computational Linguistics*, 10:291–306. Zheng-Xin Yong, Yanis Tan, Po-Chun Huang, and Amos Azaria. 2023. Multilingual large language models are not (yet) code-switchers. *arXiv:2309.14635*.