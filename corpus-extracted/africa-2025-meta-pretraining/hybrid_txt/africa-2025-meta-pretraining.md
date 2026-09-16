# Meta-Pretraining for Zero-Shot Cross-Lingual Named Entity Recognition in Low-Resource Philippine Languages

David Demitri Africa\* Suchir Salhan Yuval Weiss

Paula Buttery Richard Diehl Martinez

University of Cambridge

## Abstract

Named-entity recognition (NER) in lowresource languages is usually tackled by finetuning very large multilingual LMs, an option that is often infeasible in memory- or latencyconstrained settings. We ask whether small decoder LMs can be pretrained so that they adapt quickly and transfer zero-shot to languages unseen during pretraining. To this end we replace part of the autoregressive objective with firstorder model-agnostic meta-learning (MAML). Tagalog and Cebuano are typologically similar yet structurally different in their actor/non-actor voice systems, and hence serve as a challenging test-bed. Across four model sizes (11 M – 570 M) MAML lifts zero-shot micro-F<sub>1</sub> by 2–6 pp under head-only tuning and 1–3 pp after full tuning, while cutting convergence time by up to 8%. Gains are largest for single-token person entities that co-occur with Tagalog case particles si/ni, highlighting the importance of surface anchors.

## 1 Introduction

Named-entity recognition (NER) locates and categorises Persons (PER), Organisations (ORG) and Locations (LOC) in unstructured text (Chinchor and Robinson, 1997). It is used in a variety of important domains such as healthcare (Kundeti et al., 2016; Polignano et al., 2021; Shafqat et al., 2022) and law (Leitner et al., 2019; Au et al., 2022; Naik et al., 2023), yet progress remains concentrated in a handful of well-resourced languages. Cross-lingual named-entity recognition is therefore important to better serve underserved communities, yet recent advancements remain unevenly distributed since NER performance in many languages remains poor due to limited training resources.

A key challenge is that entity boundaries and categories are not universal: languages differ in their morphosyntactic cues, word order, and orthographic conventions. Models trained primarily on Indo-European data thus fail to generalize reliably to underrepresented settings. In this paper, we address this problem through meta-pretraining: shaping language model initializations to adapt rapidly to new linguistic conditions. Unlike standard pretraining, which minimizes average loss over a static corpus, episodic meta-pretraining (e.g. via MAML; Finn et al. 2017) explicitly optimizes for fast transfer. For low-resource NER, this offers two potential benefits: (i) rapid adaptation to languages with typologically distinct cues (e.g. case particles, voice systems, code-switching), and (ii) stronger zero-shot prototypes for common entity types, even without in-language exposure. While meta-learning has been explored for classification tasks in English or cross-lingually at BERT scale (Wu et al., 2020; Li et al., 2020; de Lichy et al., 2021), its efficacy for small decoder LMs and morphologically rich languages is underexplored.

<table><tr><td>Typological Feature</td><td>Tagalog</td><td>Cebuano</td></tr><tr><td>Voice system</td><td>√ Four-way</td><td>√ Reduced two-way</td></tr><tr><td>Case marking</td><td>√ Obligatory</td><td>✗ Often dropped</td></tr><tr><td>Borrowing / code-switch</td><td>√ High density</td><td>✗ More conservative</td></tr><tr><td>Morphological richness</td><td>√ Productive affixation</td><td>√ Regular affixation</td></tr><tr><td>Word order flexibility</td><td>√</td><td>√</td></tr><tr><td>Pronominal systems</td><td>√ Rich clitic pronouns</td><td>√ Similar</td></tr><tr><td>Reduplication</td><td>√ Common</td><td>√ Widespread</td></tr><tr><td>Orthography variation</td><td>√ Multiple conventions</td><td>✗ Multiple conventions</td></tr><tr><td>Pivot marking</td><td>√ Consistently overt</td><td>√ Overt but less consistent</td></tr></table>

Table 1: A selection of Typological Features of Tagalog and Cebuano relevant for NER. ✓ indicates strong presence, ✗ indicates reduced/less overt presence in each language. We highlight high divergence features, moderate divergence and similar features compared to Indo-European Languages, motivating these languages as a case-study for low-resourced NER. We provide a more detailed comparison along with an illustrative gloss in Appendix A.

As a case study, we focus on NER in Tagalog and

Cebuano, the two most widely spoken Philippine languages (Miranda, 2023). Typologically, both languages combine Austronesian features such as voice alternations, case particles, and reduplication with pervasive borrowing and code-switching (Figure 8; Table 1). These languages stress-test whether meta-pretraining can yield more adaptable NER representations than vanilla pretraining alone. We ask the following research questions:

RQ1 Efficacy. How much does first-order MAML improve zero-shot NER on Tagalog and Cebuano relative to vanilla autoregressive pretraining?

RQ2 What transfers? Which entity classes, morphological cues, and lexical patterns (especially those tied to Tagalog/Cebuano typology) explain the observed gains or failures?

We answer these questions by systematically comparing first-order MAML and vanilla pretraining on LLaMa-style Pico Decoders across scales, analyzing both downstream performance and representation dynamics (Diehl Martinez, 2025; Martinez et al., 2025). This allows us to investigate:

RQ3 How does the effect of meta-pretraining vary with model size? Are benefits stronger at small scales, or do they persist as capacity increases?

## 1.1 Contributions.

We provide the following contributions:

• A systematic evaluation of meta-pretrained small decoder LMs for zero-shot NER in Tagalog and Cebuano, comparing against strong vanilla pretraining baselines across four model scales.  
• Quantitative and qualitative evidence that MAML-based meta-pretraining produces sharper single-token entity prototypes, improving zero-shot NER, especially for person entities and Tagalog’s particle-rich syntax.  
• An analysis of failure modes and learning dynamics, showing the capacity-dependent nature of meta-learning gains and the tradeoff between prototype sharpening and contextual generalization.

## 2 Method

## 2.1 Motivation

Why these two languages? Tagalog and Cebuano are used every day by well over 100 million people. However, they occupy only a small fraction of the web text that current language models are pretrained on, which makes them both socially important and under-served by existing NLP tools (Miranda, 2023). Linguistically, these languages also offer complementary typological challenges for NER, which we summarise in Figure 1. Tagalog and Cebuano combine Austronesian voice systems, case particles, reduplication, and discourse-driven topic marking in ways that are rare in widely studied NLP benchmarks. In particular, Tagalog offers more overt morphosyntactic cues than Cebuano: it retains a four-way actor/non-actor voice paradigm, while Cebuano reduces this to two (Tanangkingsing, 2011) and marks syntactic roles with case particles (si/ni/ang/ng/sa). These languages offer a test bed for multilingual NER models that must generalize beyond Indo-European NER cues – where entities are typically identifiable through fixed word order and stable orthography– to handle the interaction of morphological marking, argument interaction and code-switching. Tagalog contains more Spanish loans and code-switching into English, while Cebuano maintains a more conservative Austronesian lexicon (Bautista, 2004; Baklanova, 2019). We provide a more detailed comparison of Tagalog and Cebuano typological features in Table 3.

Why Meta-learning? Being underrepresented in natural language processing (NLP) corpora (Cajote et al., 2024; Quakenbush, 2005; Dita et al., 2009; Bandarkar et al., 2024), Philippine language datasets suffer from size and quality issues. In lowresource settings, where pretraining data is scarce or absent, it is important to ask the question: will a given checkpoint finetune or transfer rapidly when exposed to a novel language (such as in deployment)?

Meta-learning addresses this by shaping initializations for quick adaptation. Model-Agnostic Meta-Learning (MAML) optimizes an LM backbone so that a few gradient steps yield high performance on a new task (Finn et al., 2017). We ask whether such an initialization, learned entirely without Tagalog/Cebuano exposure, can transfer to these languages’ distinct morphological and lexical cues for NER. Our working hypothesis is that a pretraining routine that is itself optimized for rapid adaptation will induce representations that generalize more readily across languages. Prior NLP studies have tested this mostly on English or on “BERT-scale” encoder models (Wu et al., 2020; Ma et al., 2022; Li et al., 2020; de Lichy et al., 2021); we explore whether episodic meta-pretraining of small decoder LMs, without any exposure to Tagalog or Cebuano, can still yield zero-shot gains for NER. We do not evaluate a multilingual languagemodel baseline, as our objective is to isolate the effect of episodic meta-pretraining under a matched corpus and schedule; training a competitive multilingual baseline would require different data and budgets, confounding a like-for-like comparison.

Our working hypothesis is that a pretraining routine that is itself optimized for rapid adaptation will induce representations that generalize more readily across languages, so that a model exposed only to high-resource sources can still zero-shot transfer to typologically distant, low-resource targets.

## 2.2 Architecture

We build upon the PICO decoder stack (Diehl Martinez, 2025), a LLaMa-style causal Transformer implemented in PyTorch. Four capacity tiers (tiny (11 M), small (65 M), medium (181 M) and large (570 M)) share all hyper-parameters except hidden width d ∈ {96, 384, 768, 1536}. Each model comprises L=12 RMS-normalised decoder blocks (Zhang and Sennrich, 2019) with grouped-query self-attention (Ainslie et al., 2023), RoPE positions (Su et al., 2024) and SwiGLU feed-forwards (Shazeer, 2020) that expand to 4d.

## 2.3 Hybrid pretraining objective

Training alternates between two outer-loop updates:

1. Autoregressive LM step. Standard nexttoken prediction on a pre-tokenized version of Dolma (Soldaini et al., 2024) released by the Pico library (Diehl Martinez, 2025).  
2. First-order MAML episode. A 32-way, 4-shot Subset-Masked LM Task (SMLMT; Bansal et al., 2020) is sampled, where the model predicts a masked token from the corpus on the fly. The inner loop finetunes a lightweight MLP head for ten SGD steps (α= 10<sup>−3</sup>) and the outer loop back-propagates the query loss through the frozen backbone.

The branch decision is a Bernoulli draw with probability $\rho { = } 0 . 5 ,$ , synchronised across four A100- 80 GB GPUs. The pseudocode for both can be found in Appendix C.

## 2.4 Optimisation and monitoring

We run 6,000 outer updates with AdamW $( \eta _ { \mathrm { p e a k } } =$ $3 { \times } 1 0 ^ { - 4 } , 2 . 5$ k warm-up, cosine decay), accumulating eight micro-batches of 256 sequences to reach an effective batch of 2048 sequences (1024 for tiny). Every 100 steps we log: Paloma perplexity (Magnusson et al., 2024), singular-value spectra of three attention and three feed-forward weight matrices, from which we compute proportional effective rank (PER; Diehl Martinez et al., 2024), and support and query accuracy within MAML episodes.

## 2.5 Finetuning on High-Resourced Languages

We deliberately choose high-resource languages as the finetuning sources because, in realistic deployments, these are the languages for which sizable, high-quality NER data already exists. They therefore form the most natural setting for cross-lingual transfer into low-resource settings.

After pretraining we attach an untrained linear conditional random field head (Lafferty et al., 2001), which is a well-known method used often for NER (Bundschus et al., 2008; Ma and Hovy, 2016). We finetune on a high-resource language (Danish, English, Croatian, Portuguese, Slovak, Serbian, Swedish, Chinese, Chinese-Simplified, and a mixture of all languages) before zero-shot evaluation on Tagalog (tl\_trg, tl\_ugnayan) and Cebuano (ceb\_gja) from Universal NER v1 (Mayhew et al., 2024). Results are later broken down by finetuning language. Further, two finetuning regimes are compared: head-only, where the transformer is frozen and only the classifier learns, and full, where all parameters are freed to update.

Finetuning uses AdamW $( 3 \times 1 0 ^ { - 5 } )$ for up to ten epochs with early stopping on development $\mathrm { F _ { 1 } }$ . We report micro-F<sub>1</sub>, with full details in Appendix D.

## 2.6 Baselines

For each capacity tier we also evaluate a "vanilla" Pico model (no MAML, pure autoregressive loss) under identical data, schedule and compute. Pretraining results can be found in Appendix E with model configuration details in Appendix F. A more detailed discussion of pretraining results and overall methodology can be found in Africa et al. (2025).

![](images/8ad92612dc83bd25abdb841928bbae3f003ae9c5189a123787e526bb55957596.jpg)

<details>
<summary>bar_line</summary>

| Model size | Training::maml (Micro-F1) | Training::vanilla (Micro-F1) | Delta F1 (pp) |
| --- | --- | --- | --- |
| tiny | ~0.11 | ~0.11 | ~0.003 |
| small | ~0.16 | ~0.145 | ~0.017 |
| medium | ~0.23 | ~0.215 | ~0.017 |
| large | ~0.28 | ~0.26 | ~0.015 |
</details>

Figure 1: Scale curve. Zero-shot $\mathbf { M i c r o - F _ { 1 } }$ on Cebuano & Tagalog versus parameter count. Bars compare PICO-MAML (blue) to vanilla pretraining (green); the overlaid line shows the relative gain of MAML (Delta F1, right axis). Meta-pretraining helps at every scale, but the relative lift shrinks from +38 % (11 M) to +6 % (570 M), revealing a capacity threshold below which the inner loop cannot extract reusable features.

## 3 Zero-Shot Transfer Results

Zero-shot evaluation. Unless stated otherwise, all scores are obtained without seeing any Tagalog/Cebuano data during finetune, relying solely on the UNER test sets (§ 2.4).

Figure 1 shows that PICO-MAML improves Cebuano/Tagalog micro-F at every parameter budget. The relative lift is largest for moderate sizes and tapers with scale (+6% at 570M). These results indicate that adding a single outer-loop meta-update per batch yields a cross-lingual prior not captured by vanilla pretraining under our setup.

Comparison of head-only tuning and full tuning. Decomposing by finetuning regime (Fig. 2), MAML yields 1–2 pp gains when only the CRF head is trained, implying that the frozen weights already embeds better entity cues. Full tuning narrows the gap to 0.5–1.3 pp, indicating that the lift persists even when the optimiser is free to overwrite the initialisation.

Further, results indicate that the benefit provided by the meta-objective is scale-dependent. For the 11 M (tiny) model, MAML moves the overall score by < 1 pp and yields no gain under head-only tuning. From 65 M parameters upward the benefit becomes clearer with larger head-only lifts, sug gesting a threshold at which meta-gradients can provide reusable entity features without crowding out the LM signal.

Sensitivity to finetuning language. Figure 3 profiles performance after adapting on nine highresource languages. Eight of nine languages exhibit positive deltas; the largest relative lifts occur for

Zero-Shot F1 on Cebuano & Tagalog by Regime  
![](images/f6d530757ec6166018b8956cccaecb1eb0d7011f8c4e0dfe48cd8b69dd7292e7.jpg)

<details>
<summary>bar</summary>

| Model Size | Finetune::Head::manl | Finetune::Head::vanilla | Finetune::Full::manl | Finetune::Full::vanilla |
| --- | --- | --- | --- | --- |
| tiny | ~0.005 | ~0.008 | ~0.225 | ~0.215 |
| small | ~0.045 | ~0.035 | ~0.275 | ~0.260 |
| medium | ~0.165 | ~0.145 | ~0.300 | ~0.290 |
| large | ~0.250 | ~0.240 | ~0.310 | ~0.290 |
</details>

Figure 2: Impact of finetuning regime. Head-only tuning (left) magnifies the meta-learning advantage up to +2.5 pp at 570 M, likely because the backbone must already encode entity cues. Full tuning (right) reduces but does not erase the gap, suggesting that MAML primarily accelerates convergence rather than acting as a regulariser.

Slovak (+18 %) and Croatian (+13 %). Gain in Slovak might be due to fixed case endings that consistently bracket entity names, providing a clear surface boundary signal for the model (similar in function to Tagalog’s case particles but realised morphologically rather than syntactically.) The sole regression (–2 pp on Simplified Chinese) is most likely due to a known issue in poor crossscript transfer to Chinese, but it may also be due to subword sparsity in the shared vocabulary rather than a failure of the meta-objective. (Mayhew et al., 2024).

Overall, MAML appears to teach the model to exploit shallow lexical anchors (particles, affixes) that generalise well across Indo-European languages while still transferring to more typologically distant Austronesian targets. To better understand the mechanisms underlying these gains, we conduct a focused qualitative analysis on a representative configuration.

## 4 Analysis of MAML Pretrained Models

In order to analyze the learning process, rather than just the last checkpoint, we focus our qualitative study on a MEDIUM-sized model (181 M parameters) finetuned in a head-only regime on Slovak (sk\_snk), finetuning on all 61 checkpoints from step 0 of pretraining to step 6000. We restrict our analysis to this slice because while finetuning 9760 (2 pretraining regimes x 2 finetuning regimes x 4 model sizes x 10 finetuning languages x 61 checkpoints) models would be prohibitively expensive, this configuration at least offers a reasonable signalto-cost trade-off. This is for a few reasons: (i) the medium tier is the smallest model that still exhibits a clear 2–3 pp head-only lift (Figure 1) yet is threetimes cheaper to run than the 570 M variant, (ii) Slovak delivers one of the largest relative gains without vocabulary sparsity issues and, as a Slavic language, should produce transfer errors that differ sharply from those in Tagalog and Cebuano, and (iii) freezing the backbone during head-only finetuning ensures that any performance delta must stem from representations learned during metapretraining rather than from subsequent weight updates. In the next subsection, we inspect how pretraining affects finetuning performance across checkpoints.

![](images/f1b1999963430be55718f08677badb50c79955d233ee4762858a3a548e76b7eb.jpg)  
Figure 3: Sensitivity to finetuning language. Grid of zero-shot $\mathrm { F _ { 1 } }$ curves after adapting on nine high-resource languages plus an All-languages mixture. Eight of nine languages show positive deltas; the largest relative gains occur for Slovak and Croatian, while Simplified Chinese is the lone outlier $( - 2 \mathsf { p p } )$ . This pattern indicates that the meta-objective encourages reliance on surface affixes and particles that generalise well across Indo-European sources yet still transfer to Austronesian targets.

## 4.1 Checkpoint Analysis

Does the head-only learner actually learn? Figure 4 overlays the complete finetuning trajectories for every Slovak head-only run (61 checkpoints, maml\_s0000–maml\_s6000). Viridis traces show the individual runs (getting darker the later the model checkpoint was taken), while the bold line and ribbon denote the median and inter-quartile range (IQR). The train-loss fan collapses to its asymptote within the first ≈ 800 steps and stays flat thereafter; in parallel the evaluation $\mathrm { F _ { 1 } }$ rises smoothly to 0.14 and plateaus with a narrow ±0.01 IQR. Crucially, no run diverges or oscillates, confirming that freezing the backbone and training only a linear chain CRF head is both stable and something is learned. This satisfies the prerequisite for using the configuration as a clean test-bed: any downstream difference between MAML and vanilla is likely to stem from the initial representations, not from optimisation quirks or training instabilities.

Does meta-pretraining yield transfer-relevant representations? The checkpoint sweep in Figure 5 confirms the other prerequisite for this qualitative analysis: that meta-pretraining produces representations which become increasingly helpful for zero-shot transfer. First, the top panel shows that, regardless of which MAML snapshot we freeze, the linear chain CRF head always converges to essentially the same narrow band of train loss (0.10-0.15); optimisation is therefore stable and predictable, satisfying our first prerequisite. More importantly, the bottom panel reveals a very different story for cross-lingual evaluation: while Slovak dev $\mathrm { F } _ { 1 }$ plateaus early (by around step 1k), Tagalog and Cebuano $\mathrm { F } _ { 1 }$ continue to climb for another four thousand meta-updates, ending 0.15 and 0.12 points higher than at the initial checkpoint. In other words, additional MAML steps learn features that are invisible to the in-language dev set yet directly benefit unseen Austronesian targets. Tagalog improves earlier and peaks higher than Cebuano, hinting that the meta-objective is capturing surface cues (e.g. case particles) that are more diagnostic in Tagalog. Taken together with the “fan” plot of learning curves, the sweep demonstrates that meta-pretraining yields encoder states that are both optimisation-friendly and transfer-relevant, justifying the focus on this snapshot for deeper qualitative inspection. As such, we deepen the analysis in the next subsection by inspecting the behavior of our models on the level of the NER tags predicted.

Learning Curves (Medium, Head-only, Slovak)  
![](images/74fa2a3a43b1741fd26244c10c2f5872ed6c1995fd9823746eb362c4e566bb6d.jpg)

<details>
<summary>line</summary>

| Step | Train Loss (Median) | Train Loss (IQR range) | Eval F1 (Median) |
| --- | --- | --- | --- |
| 0 | ~0.45 | ~0.42 ~0.48 | 0 |
| 500 | ~0.45 | ~0.42 ~0.48 | 0 |
| 1000 | ~0.18 | ~0.16 ~0.22 | ~0.03 |
| 2000 | ~0.15 | ~0.13 ~0.19 | ~0.07 |
| 3000 | ~0.14 | ~0.12 ~0.18 | ~0.10 |
| 4000 | ~0.13 | ~0.11 ~0.17 | ~0.12 |
| 5000 | ~0.12 | ~0.10 ~0.16 | ~0.13 |
</details>

Figure 4: Learning curves for the Slovak head-only setting. Top: train loss; bottom: eval micro-F . Faint green lines = all individual checkpoints; bold line = median; shaded band = 25–75 % IQR. Both metrics converge monotonically and remain tightly bunched, indicating a stable optimisation surface for the linear head.

## 4.2 Tag-level Analysis

Per-tag behaviour. Figure 6 reports per-entity $\mathrm { F } _ { 1 }$ obtained after head-only finetuning the Slovak CRF head on each MAML checkpoint. PER climbs to 0.6-0.7 while LOC and ORG remain at zero. This is not a case of the classifier “over-fitting” in the usual sense—i.e. collapsing to always predicting a single label. A linear-chain CRF is free to emit any BIO tag at any position; if it were truly degenerate we would see train loss stagnate near the log-uniform baseline and the PER curve itself would also be flat. Instead, train loss converges to the same narrow band for every checkpoint (Fig.4) and PER performance tracks the amount of metapretraining, so the head is learning a genuine decision boundary. It simply has informative features for people but none for locations or organisations.

![](images/dd211c8fa4b12fc2222c356c52630312da51389030d69857a60938c91309681b.jpg)

<details>
<summary>line</summary>

| Pretraining Checkpoint Step | Final Train Loss | Dev F1 | Tagalog F1 | Cebuano F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.30 | ~0.04 | ~0.05 | ~0.07 |
| 250 | ~0.18 | ~0.01 | ~0.01 | ~0.01 |
| 500 | ~0.16 | ~0.01 | ~0.01 | ~0.01 |
| 750 | ~0.15 | ~0.01 | ~0.01 | ~0.01 |
| 1000 | ~0.14 | ~0.02 | ~0.24 | ~0.03 |
| 1250 | ~0.14 | ~0.06 | ~0.14 | ~0.02 |
| 1500 | ~0.13 | ~0.12 | ~0.14 | ~0.02 |
| 1750 | ~0.13 | ~0.10 | ~0.14 | ~0.02 |
| 2000 | ~0.13 | ~0.10 | ~0.15 | ~0.05 |
| 2250 | ~0.20 | ~0.12 | ~0.01 | ~0.02 |
| 2500 | ~0.13 | ~0.08 | ~0.16 | ~0.08 |
| 2750 | ~0.12 | ~0.12 | ~0.26 | ~0.12 |
| 3000 | ~0.12 | ~0.12 | ~0.33 | ~0.17 |
| 3250 | ~0.12 | ~0.13 | ~0.37 | ~0.20 |
| 3500 | ~0.12 | ~0.13 | ~0.33 | ~0.18 |
| 3750 | ~0.12 | ~0.13 | ~0.28 | ~0.18 |
| 4000 | ~0.12 | ~0.13 | ~0.26 | ~0.18 |
| 4250 | ~0.12 | ~0.15 | ~0.37 | ~0.19 |
| 4500 | ~0.12 | ~0.14 | ~0.33 | ~0.19 |
| 4750 | ~0.12 | ~0.15 | ~0.26 | ~0.20 |
| 5000 | ~0.11 | ~0.14 | ~0.25 | ~0.20 |
| 5250 | ~0.11 | ~0.17 | ~0.37 | ~0.21 |
| 5500 | ~0.11 | ~0.16 | ~0.27 | ~0.21 |
| 5750 | ~0.11 | ~0.16 | ~0.30 | ~0.21 |
| 6000 | ~0.11 | ~0.16 | ~0.27 | ~0.21 |
</details>

Figure 5: Final metrics vs. pretraining checkpoint for the MEDIUM MAML backbone frozen during headonly finetuning on Slovak. Top: final train loss of the CRF head, every run converges to the same narrow range. Bottom: final micro-F on Slovak dev (blue), Tagalog (green) and Cebuano (yellow). Although inlanguage performance saturates early, cross-lingual $\mathrm { F _ { 1 } }$ keeps improving up to step 6000, indicating that later meta-updates learn representations useful specifically for zero-shot transfer.

![](images/7cb46366755ad97af1be42e6dd741182f6a117e4317f8e297a99b1cc910fc3fa.jpg)

<details>
<summary>line</summary>

| Pretraining Checkpoint Step | Tagalog F1::PER | Tagalog F1::LOC | Tagalog F1::ORG | Cebuano F1::PER | Cebuano F1::LOC | Cebuano F1::ORG |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.2 | 0 | 0 | ~0.25 | 0 | 0 |
| 100 | ~0.2 | 0 | 0 | ~0.15 | 0 | 0 |
| 200 | — | 0 | 0 | ~0.1 | 0 | 0 |
| 300 | — | 0 | 0 | ~0.08 | 0 | 0 |
| 400 | — | 0 | 0 | ~0.05 | 0 | 0 |
| 900 | — | 0 | 0 | ~0.05 | 0 | 0 |
| 1000 | ~0.5 | 0 | 0 | ~0.15 | 0 | 0 |
| 1100 | ~0.45 | 0 | 0 | ~0.12 | 0 | 0 |
| 1200 | ~0.35 | 0 | 0 | ~0.12 | 0 | 0 |
| 1300 | ~0.32 | 0 | 0 | ~0.12 | 0 | 0 |
| 1400 | ~0.32 | 0 | 0 | ~0.12 | 0 | 0 |
| 1500 | ~0.32 | 0 | 0 | ~0.12 | 0 | 0 |
| 1600 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 1700 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 1800 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 1900 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2000 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2100 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2200 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2300 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2400 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2500 | ~0.28 | 0 | 0 | ~0.12 | 0 | 0 |
| 2600 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 2700 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 2800 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 2900 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3000 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3100 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3200 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3300 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3400 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3500 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3600 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3700 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3800 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 3900 | ~-0.15 | 0 | 0 | ~-0.15 | 0 | 0 |
| 4000 | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 415* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 425* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 435* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 445* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 455* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 465* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 475* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 485* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 495* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 515* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 535* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* | ~-4.25* |
| 545* | ~-4.25* | ~-4.25* | ~ -4.25* | ~-4.25* | ~ -4.25* | ~ -4.25* |
</details>

Figure 6: Per-entity $\mathbf { F } _ { 1 }$ across MAML checkpoints. PER (dark viridis) improves steadily with more metasteps; LOC and ORG curves remain at chance level, indicating that the frozen backbone provides transferable features for single-token personal names but little for multi-token locations or organisations. Tagalog benefits earlier than Cebuano, consistent with its obligatory case particles.

Observed imbalance and potential causes. First, the Slovak finetune set is intrinsically personheavy. As Table 4 shows, PER spans outnumber LOC by roughly 8:1 and ORG by 15:1. Under head-only training, every gradient step passes through the frozen encoder unchanged and the CRF receives thousands of positive updates for persons but only a few hundred for the other classes. This likely leads to only the PER decision boundary sharpening. Second, 87.6% of Slovak person mentions are single tokens compared with 75.1 % for locations and 56.9% for organisations. A singletoken span can be captured by one weight vec tor, whereas multi-word spans require the head to model boundaries and label transitions—a capacity it simply does not have when the encoder cannot adapt. Third, Tagalog still offers a comparatively reliable surface cue. The case particles si and ni precede roughly 11% of gold PER spans, almost double the $^ { 5 - 6 }$ % rate observed in Cebuano (Ta ble 5). The earlier lift and higher ceiling of the Tagalog PER curve are therefore consistent with the backbone having learned to map the pattern "particle + token" to the PER label, a cue that is in formative in Tagalog but is sparser in Cebuano. Fi nally, cross-lingual lexical overlap is likely higher for personal names, many of which (e.g. Obama, Manuel) appear verbatim in English corpora used during pretraining; locations and organisations, by contrast, are often translated or abbreviated. All four factors act in the same direction, favouring PER. Disentangling their individual contributions would require targeted ablations (particle masking, balanced resampling, controlled name substitution, etc.) which we leave for future work. In the next subsection, we assess behaviors on the level of words and tokens to relate NER performance to the low-resource languages being transferred to.

## 4.3 Word-level Analysis

Figures $7 { \mathrm { a } } { - } 7 { \mathrm { d } }$ visualise the checkpoint-by checkpoint evolution of token-level confidence (p(correct tag)) for the ten most frequent surface words in each evaluation set. Entities and non-entities are split so the dynamic range is not drowned out by O tokens. Two qualitative patterns emerge.

Fast confidence in frequent tokens. Non-entity function words such as ng, ang, sa in Tagalog and the Cebuano clitic -ng start with high confi dence and barely budge after the first 200 metaupdates (Fig. 7b, 7d). As these tokens domi nate the language-model loss, autoregressive training achieves a high confidence in them early and MAML has little head-room to improve over check points.

Monotonic gains for high-overlap proper names. In the Tagalog set, international names (City, Maynila, Maria) and locations transliterated from English (Pasay) become steadily brighter (lower loss) until about step 3000 (Fig. 7a). Similar be haviour appears for Maria, Cebu, Mary in Cebuano (Fig. 7c). These words either appear verbatim in the English Dolma corpus or share sub-tokens (Ma\_, Ceb\_) with it, so the meta-objective can reuse pro totypes that happen to be used by the Austronesian targets. The timing matches the checkpoint-sweep (Fig. 5): cross-lingual $\mathrm { F } _ { 1 }$ continues to climb long after Slovak dev has saturated likely because the back-bone is still lowering loss on these anchor words. We illustrate these mechanisms further in two case studies in Appendix B.

<table><tr><td>Size</td><td>Regime</td><td> $\Delta t_{90}$ </td><td> $\Delta AUC$ </td><td> $\Delta slope$ </td></tr><tr><td rowspan="2">large</td><td>full</td><td>-111.1</td><td>-0.004</td><td>0.0e-05</td></tr><tr><td>head</td><td>-55.6</td><td>-0.012</td><td>1.0e-05</td></tr><tr><td rowspan="2">medium</td><td>full</td><td>0.0</td><td>-0.005</td><td>0.0e-05</td></tr><tr><td>head</td><td>55.6</td><td>-0.011</td><td>0.0e-05</td></tr><tr><td rowspan="2">small</td><td>full</td><td>0.0</td><td>0.003</td><td>0.0e-05</td></tr><tr><td>head</td><td>-55.6</td><td>0.003</td><td>-0.0e-05</td></tr><tr><td rowspan="2">tiny</td><td>full</td><td>-111.1</td><td>0.004</td><td>-0.0e-05</td></tr><tr><td>head</td><td>55.6</td><td>-0.023</td><td>5.0e-05</td></tr></table>

Table 2: Finetuning convergence speed metrics ∆ (MAML-Vanilla) averaged over nine in-language tasks. The largest and smallest models enjoy the most pro nounced speed-ups from full MAML meta-initialization, while medium and tiny models show negligible $\Delta t _ { 9 0 }$ under full-model tuning. Under head-only tuning, large and small decoders still benefit modestly, whereas medium and tiny decoders actually slow down. Across all settings, slope remains near zero, indicating that meta-training primarily accelerates mid-to-late conver gence rather than the very first gradient steps.

## 5 Finetuning Speed of Meta-Pretraining

Finally, we assess finetuning speed using convergence time (measuring time to achieve 90% of final loss $t _ { 9 0 } )$ , normalized area under the loss curve (measuring aggregate convergence behavior over the curve), and initial slope (measuring the initial speed of learning in the first few steps), as seen in Table 2. Across nine in-language tasks, full-model finetuning shows the clearest acceleration for the largest and smallest models: MAML cuts t by roughly 8% (≈ 111 steps) and modestly reduces loss AUC. Medium and small models show negligi ble or inconsistent speed-ups under full tuning, suggesting that the effect depends strongly on model capacity. In head-only tuning, large and small mod els again benefit slightly, while medium and tiny models slow down, likely due to underpowered or collapsed meta-dynamics.

Initial slopes remain effectively unchanged across all settings, indicating that MAML does not alter the very first gradient steps but instead reorganizes the loss landscape to make mid- to late-stage convergence more efficient. These results align with earlier findings that MAML’s main benefit lies in providing sharper, more reusable token-level features for high-capacity backbones, with limited or negative effects when capacity is insufficient to retain both language modeling and episodic priors.

![](images/30a3cee3c84843f85fbf874890e3af61212039c3b64776297e8a909864945406.jpg)

<details>
<summary>heatmap</summary>

| Juan | ~0.45 ~0.75 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 | ~0.15 ~0.38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linda | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | ~0.15 ~0.35 | **~0.25** | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 | ~0.25 ~0.45 |
</details>

(a) Tagalog — Entities only

![](images/cde739f2f151e3ad0d144b9ff432daecc30ff262f3ef290e244e410989479b6f.jpg)

<details>
<summary>heatmap</summary>

| ? | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0.75 | ~0..75 | ~0..75 | ~0..75 | ~0..75 | ~0..75 | ~0..75 | ~0..75 | ~1. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bata | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
</details>

(b) Tagalog — Non-entities

![](images/21007804a70ba6d137f916862fb30c012f5a19bc8759a4615f5d5d6b579af9ec.jpg)

<details>
<summary>heatmap</summary>

| Word (Top 10 by freq) | Pretraining Checkpoint Step 0 | Pretraining Checkpoint Step 200 | Pretraining Checkpoint Step 400 | Pretraining Checkpoint Step 600 | Pretraining Checkpoint Step 800 | Pretraining Checkpoint Step 1000 | Pretraining Checkpoint Step 1200 | Pretraining Checkpoint Step 1400 | Pretraining Checkpoint Step 1600 | Pretraining Checkpoint Step 1800 | Pretraining Checkpoint Step 2000 | Pretraining Checkpoint Step 2200 | Pretraining Checkpoint Step 2400 | Pretraining Checkpoint Step 2600 | Pretraining Checkpoint Step 2800 | Pretraining Checkpoint Step 3000 | Pretraining Checkpoint Step 3200 | Pretraining Checkpoint Step 3400 | Pretraining Checkpoint Step 3600 | Pretraining Checkpoint Step 3800 | Pretraining Checkpoint Step 4000 | Pretraining Checkpoint Step 4200 | Pretraining Checkpoint Step 4400 | Pretraining Checkpoint Step 4600 | Pretraining Checkpoint Step 4800 | Pretraining Checkpoint Step 5000 | Pretraining Checkpoint Step 5200 | Pretraining Checkpoint Step 5400 | Pretraining Checkpoint Step 5600 | Pretraining Checkpoint Step 5800 | Pretraining Checkpoint Step 6000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tom | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| Ditang | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0 .25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 | ~0.25 |
| Juan | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 | ~0..25 |
| Carmen | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 | ~0 ..25 |
| Ruben | ~0.15 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Pedro | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
</details>

(c) Cebuano — Entities only

![](images/ad17cfebcff25774def39d2d9ae91dbc3655faf34fc4cb12081808d60f892fc9.jpg)

<details>
<summary>heatmap</summary>

| Word (Top 10 by freq) | Pretraining Checkpoint Step | p(correct tag) |
| --- | --- | --- |
| g-ngug ka koang si | 0~200 | ~0.4 ~0.6 |
| g-ngug ka koang si | 200~400 | ~0.6 ~0.8 |
| g-ngug ka koang si | 400~600 | ~0.8 ~0.9 |
| g-ngug ka koang si | 600~800 | ~0.9 ~1.0 |
| g-ngug ka koang si | 800~1000 | ~1.0 |
| g-ngug ka koang si | 1000~1200 | ~1.0 |
| g-ngug ka koang si | 1200~1400 | ~1.0 |
| g-ngug ka koang si | 1400~1600 | ~1.0 |
| g-ngug ka koang si | 1600~1800 | ~1.0 |
| g-ngug ka koang si | 1800~2000 | ~1.0 |
| g-ngug ka koang si | 2000~2200 | ~1.0 |
| g-ngug ka koang si | 2200~2400 | ~1.0 |
| g-ngug ka koang si | 2400~2600 | ~1.0 |
| g-ngug ka koang si | 2600~2800 | ~1.0 |
| g-ngug ka koang si | 2800~3000 | ~1.0 |
| g-ngug ka koang si | 3000~3200 | ~1.0 |
| g-ngug ka koang si | 3200~3400 | ~1.0 |
| g-ngug ka koang si | 3400~3600 | ~1.0 |
| g-ngug ka koang si | 3600~3800 | ~1.0 |
| g-ngug ka koang si | 3800~4000 | ~1.0 |
| g-ngug ka koang si | 4000~4200 | ~1.0 |
| g-ngug ka koang si | 4200~4400 | ~1.0 |
| g-ngug ka koang si | 4400~4600 | ~1.0 |
| g-ngug ka koang si | 4600~4800 | ~1.0 |
| g-ngug ka koang si | 4800~5000 | ~1.0 |
| g-ngug ka koang si | 5000~5200 | ~1.0 |
| g-ngug ka koang si | 5200~5400 | ~1.0 |
| g-ngug ka koang si | 5400~5600 | ~1.0 |
| g-ngug ka koang si | 5600~5800 | ~1.0 |
| g-ngug ka koang si | 5800~6000 | ~1.0 |
</details>

(d) Cebuano — Non-entities  
Figure 7: Evolution of token-level confidence (p(correct tag)) across pretraining checkpoints. Top row: Tagalog; bottom row: Cebuano. Left: entities only. Right: non-entities.

## 6 Related Work

NER in Filipino, Tagalog, and Cebuano. NER for Philippine languages remains underexplored, with most work focusing on resource construction rather than cross-lingual modeling. Recent corpora include TLUnified-NER (Miranda, 2023), TF-NERD (Ramos and Vergara, 2023), CebuaNER (Pilar et al., 2023), and UniversalNER (Mayhew et al., 2024). Modeling efforts in this area primarily use NER-specific systems (Sagum and Sagum, 2025; Eboña et al., 2013; Dela Cruz et al., 2018) incorporating a simpler backbone such as a support vector machine (Castillo et al., 2013) or an LSTM (Chan et al., 2023). Most recently, FilBench (Miranda et al., 2025) and Batayan (Montalan et al., 2025) support Filipino evaluation on NLP tasks for LLMs.

Meta-learning for Pretraining. Although most work applies meta-learning at fine-tuning time, a growing line of research embeds meta-objectives directly into pretraining. (Raghu et al., 2021) showed that framing parameter-efficient adapter learning as a bilevel problem yields representations that fine-tune more effectively than standard PEFT. (Hou et al., 2022) extend this to full transformers. (Miranda et al., 2023) argue that explicit MAML objectives can outperform fixed pretraining on highly diverse task distributions. (Ke et al., 2021) integrate a MAML-style inner loop into a multi-criteria Chinese Word Segmentation pretraining task.

## 7 Conclusion

This paper shows that MAML-based metapretraining, even when applied to small decoderonly language models, can meaningfully improve zero-shot transfer to low-resource languages, as demonstrated on Tagalog and Cebuano NER. The gains are most pronounced for person entities and head-only finetuning, and scale best with larger model capacities. Our qualitative and word-level analyses reveal that the mechanism of improvement centers on the sharpening of lexical prototypes and better anchoring to surface cues like Tagalog case particles. Hence, we do not expect these improvements to fully generalize to multi-token or highly contextual entity types.

These findings suggest that meta-learning can provide a principled route to more adaptable small models, but also highlight key limitations: the benefits are capacity- and task-dependent, and the current approach struggles with richer entity structures. Future work should explore alternative metalearning objectives, extend to more diverse tasks and languages, and investigate the dynamics of pro totype formation in even lower-resource settings.

## Limitations

The gains are most pronounced for person entities and head-only finetuning, and scale best with larger model capacities. All training runs stop at exactly six thousand outer steps, a horizon that may be too short for the largest model, so the conclusions derived only cover a fraction of the training budget a corporate setup might have. A more diverse and multilingual corpus may alter both quantitative and qualitative conclusions, and varying languages in the meta-task is a natural way to extend this work. Qualitative analysis was conducted on a single configuration and single seed due to cost and GPU constraints. Qualitative analysis was conducted by a native Tagalog speaker with a register typical of Manila, and a wide variety of perspectives would improve the robustness of the analysis. Finally (and most naturally), our focus on only two Austronesian languages controls for certain lexical and syntactic divergences but limits the generality of the typological conclusions; extending to a broader set of Philippine and Malayo-Polynesian languages is a natural next step.

## References

David Demitri Africa, Yuval Weiss, Paula Buttery, and Richard Diehl Martinez. 2025. Learning dynamics of meta-learning in small model pretraining. Preprint, arXiv:2508.02189  
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895– 4901.  
Ting Wai Terence Au, Ingemar J Cox, and Vasileios Lampos. 2022. E-ner–an annotated named entity recognition corpus of legal text. arXiv preprint arXiv:2212.09306.  
Ekaterina Baklanova. 2019. The impact of spanish and english hybrids on contemporary tagalog.  
Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. In Proceedings ofthe 62nd An nual Meeting ofthe Associationfor Computational  
Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.  
Trapit Bansal, Rishikesh Jha, Tsendsuren Munkhdalai, and Andrew McCallum. 2020. Self-supervised metalearning for few-shot natural language classification tasks. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 522–534, Online. Association for Computational Linguistics.  
Maria Lourdes S Bautista. 2004. Tagalog-english code switching as a mode of discourse. Asia Pacific Education Review, 5(2):226–233.  
Markus Bundschus, Mathaeus Dejori, Martin Stetter, Volker Tresp, and Hans-Peter Kriegel. 2008. Extraction of semantic biomedical relations from text using conditional random fields. BMC bioinformatics, 9(1):207.  
Rhandley D Cajote, Rowena Cristina L Guevara, Michael Gringo Angelo R Bayona, and Crisron Rudolf G Lucas. 2024. Philippine Languages Database: A Multilingual Speech Corpora for Developing Systems for Philippine Spoken Languages. LREC-COLING 2024, page 264.  
Jonalyn M Castillo, Marck Augustus L Mateo, Antonio DC Paras, Ria A Sagum, and Vina Danica F Santos. 2013. Named entity recognition using support vector machine for filipino text documents. International Journal ofFuture Computer and Communication, 2(5):530.  
Kyle Chan, Kaye Ann De Las Alas, Charles Orcena, Dan John Velasco, Qyle John San Juan, and Charibeth Cheng. 2023. Practical approaches for lowresource named entity recognition of filipino telecommunications domain. In Proceedings ofthe 37th Pacific Asia Conference on Language, Information and Computation, pages 234–242.  
Nancy Chinchor and Patricia Robinson. 1997. Muc-7 named entity task definition. In Proceedings of the 7th Conference on Message Understanding, volume 29, pages 1–21.  
Cyprien de Lichy, Hadrien Glaude, and William Campbell. 2021. Meta-learning for few-shot named entity recognition. In Proceedings ofthe 1st Workshop on Meta Learning and Its Applications to Natural Language Processing, pages 44–58.  
Bern Maris Dela Cruz, Cyril Montalla, Allysa Manansala, Ramon Rodriguez, Manolito Octaviano, and Bernie S. Fabito. 2018. Named-entity recognition for disaster related filipino news articles. In TEN-CON 2018 - 2018 IEEE Region 10 Conference, pages 1633–1636.  
Richard Diehl Martinez. 2025. Pico: A lightweight framework for studying language model learning dynamics.  
Richard Diehl Martinez, Pietro Lesci, and Paula Buttery. 2024. Tending towards stability: Convergence challenges in small language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 3275–3286, Miami, Florida, USA. Association for Computational Linguistics.  
Shirley N Dita, Rachel Edita O Roxas, and Paul Inventado. 2009. Building online corpora of Philippine languages. In Proceedings ofthe 23rd Pacific Asia Conference on Language, Information and Computation, pages 646–653. Waseda University.  
Karen Mae L Eboña, Orlando S Llorca Jr, Genrev P Perez, Jhustine M Roldan, Iluminda Vivien R Domingo, and Ria A Sagum. 2013. Named-entity recognizer (ner) for filipino novel excerpts using maximum entropy approach. Journal ofIndustrial and Intelligent Information Vol, 1(1).  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1126–1135. PMLR.  
Zejiang Hou, Julian Salazar, and George Polovets. 2022. Meta-learning the difference: Preparing large language models for efficient adaptation. Transactions of the Association for Computational Linguistics, 10:1249–1265.  
Zhen Ke, Liang Shi, Songtao Sun, Erli Meng, Bin Wang, and Xipeng Qiu. 2021. Pre-training with meta learning for Chinese word segmentation. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5514–5523, Online. Association for Computational Linguistics.  
Srinivasa Rao Kundeti, J Vijayananda, Srikanth Mujjiga, and M Kalyan. 2016. Clinical named entity recognition: Challenges and opportunities. In 2016 IEEE International Conference on Big Data (Big Data), pages 1937–1945. IEEE.  
John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. 2001. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In Proceedings of the Eighteenth International Conference on Machine Learning, ICML ’01, page 282–289, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc.  
Elena Leitner, Georg Rehm, and Julian Moreno-Schneider. 2019. Fine-grained named entity recogni tion in legal documents. In International conference on semantic systems, pages 272–287. Springer.  
Jing Li, Billy Chiu, Shanshan Feng, and Hao Wang. 2020. Few-shot named entity recognition via metalearning. IEEE Transactions on Knowledge and Data Engineering, 34(9):4245–4256.  
Tingting Ma, Huiqiang Jiang, Qianhui Wu, Tiejun Zhao, and Chin-Yew Lin. 2022. Decomposed metalearning for few-shot named entity recognition. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 1584–1596.  
Xuezhe Ma and Eduard Hovy. 2016. End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1064–1074, Berlin, Germany. Association for Computational Linguistics.  
Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Walsh, Yanai Elazar, Kyle Lo, and 1 others. 2024. Paloma: A benchmark for evaluating language model fit. Advances in Neural Information Processing Systems, 37:64338–64376.  
Richard Diehl Martinez, David Demitri Africa, Yuval Weiss, Suchir Salhan, Ryan Daniels, and Paula Buttery. 2025. Pico: A lightweight framework for studying language model learning dynamics. Under Review.  
Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Suppa, Hila Gonen, Joseph Marvin Imperial, Börje Karlsson, Peiqin Lin, Nikola Ljubešic, Lester James´ Miranda, Barbara Plank, Arij Riabi, and Yuval Pinter. 2024. Universal NER: A gold-standard multilingual named entity recognition benchmark. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4322–4337, Mexico City, Mexico. Association for Computational Linguistics.  
Brando Miranda, Patrick Yu, Saumya Goyal, Yu-Xiong Wang, and Sanmi Koyejo. 2023. Is pretraining truly better than meta-learning? Preprint, arXiv:2306.13841.  
Lester James V. Miranda. 2023. Developing a named entity recognition dataset for Tagalog. In Proceedings ofthe First Workshop in South East Asian Language Processing, pages 13–20, Nusa Dua, Bali, Indonesia. Association for Computational Linguistics.  
Lester James V. Miranda, Elyanah Aco, Conner Manuel, Jan Christian Blaise Cruz, and Joseph Marvin Imperial. 2025. Filbench: Can llms understand and generate filipino? Preprint, arXiv:2508.03523.  
Jann Railey Montalan, Jimson Paulo Layacan, David Demitri Africa, Richell Isaiah S. Flores, Michael T. Lopez Ii, Theresa Denise Magsajo, Anjanette Cayabyab, and William Chandra Tjhi. 2025. Batayan: A Filipino NLP benchmark for evaluating large language models. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31239–31273, Vienna, Austria. Association for Computational Linguistics.  
Varsha Naik, Purvang Patel, and Rajeswari Kannan. 2023. Legal entity extraction: An experimental study of ner approach for legal documents. International Journal ofAdvanced Computer Science and Applications, 14(3).  
Ma. Beatrice Emanuela Pilar, Dane Dedoroy, Ellyza Mari Papas, Mary Loise Buenaventura, Myron Darrel Montefalcon, Jay Rhald Padilla, Joseph Marvin Imperial, Mideth Abisado, and Lany Maceda. 2023. CebuaNER: A new baseline Cebuano named entity recognition model. In Proceedings of the 37th Pacific Asia Conference on Language, Information and Computation, pages 792–800, Hong Kong, China. Association for Computational Linguistics.  
Marco Polignano, Marco de Gemmis, Giovanni Semeraro, and 1 others. 2021. Comparing transformerbased ner approaches for analysing textual medical diagnoses. In CLEF (Working Notes), pages 818– 833.  
J Stephen Quakenbush. 2005. Philippine linguistics from an SIL perspective: Trends and prospects. Current issues in Philippine linguistics and anthropology: Parangal kay Lawrence A. Reid, pages 3–27.  
Aniruddh Raghu, Jonathan Lorraine, Simon Kornblith, Matthew McDermott, and David K Duvenaud. 2021. Meta-learning to improve pre-training. Advances in Neural Information Processing Systems, 34:23231– 23244.  
Robin Kamille Ramos and John Paul Vergara. 2023. Tfnerd: Tagalog fine-grained named entity recognition dataset. In Proceedings ofthe 2023 7th International Conference on Natural Language Processing and Information Retrieval, pages 222–227.  
Ria A. Sagum and Janelle Kyra A. Sagum. 2025. Paral lel ensemble approach for named entity recognition in filipino text. In Proceedings ofthe 2024 7th Artifi cial Intelligence and Cloud Computing Conference, AICCC ’24, page 409–413, New York, NY, USA. Association for Computing Machinery.  
Sarah Shafqat, Hammad Majeed, Qaisar Javaid, and Hafiz Farooq Ahmad. 2022. Standard ner tagging scheme for big data healthcare analytics built on unified medical corpora. Journal of Artificial Intelligence and Technology, 2(4):152–157.  
Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202.  
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, and 1 others. 2024. Dolma: an open corpus of three trillion tokens for language model pretraining research. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15725–15788.  
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.  
Michael Tanangkingsing. 2011. A Functional Reference Grammar ofCebuano: A Discourse-Based Perspective. Peter Lang, Berlin.  
Qianhui Wu, Zijia Lin, Guoxin Wang, Hui Chen, Börje F Karlsson, Biqing Huang, and Chin-Yew Lin. 2020. Enhanced meta-learning for cross-lingual named entity recognition with minimal resources. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 9274–9281.  
Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32.

## A NER-Relevant Typological Features of Cebuano and Tagalog

This extended table highlights how morphosyntactic and discourse-level differences between the two languages interact with the challenges of named entity recognition (NER). We lay out feature-by-feature contrasts to illustrate that even closely related Philippine languages present distinct hurdles for tasks like NER. The table emphasizes that while Tagalog offers overt morphosyntactic cues (e.g., case particles, topic marking), Cebuano relies more heavily on discourse inference, thereby requiring different modeling strategies for effective NER.

<table><tr><td>Typological Feature</td><td>Tagalog</td><td>Cebuano</td><td>Challenge for NER</td></tr><tr><td>Voice system</td><td>Four-way actor/non-actor voice paradigm</td><td>Reduced two-way system</td><td>Tagalog&#x27;s rich voice alternations encode argument roles morphologically, complicating alignment of entities with semantic roles. Cebuano&#x27;s reduced system lowers redundancy, making cues for role identification less explicit.</td></tr><tr><td>Case marking</td><td>Obligatory case particles (si, ni, ang, ng, sa)</td><td>Case particles often dropped or fused</td><td>Tagalog provides reliable morphosyntactic signals for entity boundaries/roles. Cebuano forces reliance on discourse, requiring coreference and contextual inference.</td></tr><tr><td>Lexical borrowing / code-switching</td><td>High density of Spanish loans and English code-switching</td><td>More conservative Austronesian lexicon</td><td>Tagalog NER must cope with OOV issues, language-mixing, and orthographic variation. Cebuano NER must handle morphologically complex Austronesian stems, underrepresented in multilingual embeddings.</td></tr><tr><td>Morphological richness</td><td>Productive affixation (focus, aspect, causatives)</td><td>Similarly rich, but slightly more regular</td><td>Surface forms for named entities may be inflected or derivationally complex, increasing sparsity for training data.</td></tr><tr><td>Word order flexibility</td><td>Relatively free (voice and particles constrain roles)</td><td>Even freer, especially without explicit case markers</td><td>Named entities may appear in non-canonical positions, reducing the utility of positional cues.</td></tr><tr><td>Pronominal systems</td><td>Rich system of clitic pronouns that attach to verbs or particles</td><td>Similar system but with different distributions</td><td>Entities can be referred to obliquely or dropped entirely; clitic attachment blurs tokenization boundaries, confusing NER pipelines.</td></tr><tr><td>Reduplication</td><td>Common for aspect, plurality, intensification</td><td>Widespread and productive</td><td>Reduplicated forms of named entities (nicknames, reductive roots) may not be recognized as related to the canonical form.</td></tr><tr><td>Orthography &amp; variation</td><td>Spanish-influenced orthography, multiple spelling conventions</td><td>More phonologically consistent, but dialectal spelling variation persists</td><td>Orthographic inconsistency makes lexicon-based NER brittle, especially in noisy social media text.</td></tr><tr><td>Discourse prominence / topic marking</td><td>Ang-marked topic influences salience</td><td>Topic is often inferred from discourse, less explicit marking</td><td>Tagalog gives overt topic marking, aiding salience detection; Cebuano relies on pragmatics, requiring discourse-level modeling.</td></tr></table>

Table 3: Detailed typological contrasts between Tagalog and Cebuano and their implications for NER.

<table><tr><td>Tag.</td><td>Pumunta</td><td>si</td><td>Maria</td><td>sa</td><td>Cebu.</td></tr><tr><td>Gloss</td><td>go.PFV</td><td>NOM</td><td>Maria</td><td>OBL</td><td>Cebu</td></tr><tr><td>NER</td><td>O</td><td>O</td><td>B-PER</td><td>O</td><td>B-LOC</td></tr><tr><td>Ceb. (with marker)</td><td>Miadto</td><td>si</td><td>Juan</td><td>sa</td><td>Sugbo.</td></tr><tr><td>Gloss</td><td>go.PST</td><td>NOM</td><td>Juan</td><td>OBL</td><td>Cebu</td></tr><tr><td>NER</td><td>O</td><td>O</td><td>B-PER</td><td>O</td><td>B-LOC</td></tr><tr><td>Ceb. (zero-marked)</td><td>Miadto</td><td>Juan</td><td>sa</td><td>Sugbo.</td><td></td></tr><tr><td>Gloss</td><td>go.PST</td><td>Juan</td><td>OBL</td><td>Cebu</td><td></td></tr><tr><td>NER</td><td>O</td><td>B-PER</td><td>O</td><td>B-LOC</td><td></td></tr></table>

Figure 8: Surface cues for named entities. Tagalog typically provides an overt personal article (si/ni) before names; Cebuano may show the same article, but zero-marked variants also occur in some registers/contexts, reducing overt anchors.

## B Case Studies

To illustrate the mechanisms underlying MAML’s improvements, we present two contrasting examples that demonstrate how meta-pretraining affects different types of linguistic patterns in Tagalog NER. We measure $\Delta$ log-prob as the change in surprisal (−! log p) for the gold label between the vanilla and MAML model. A negative $\Delta$ means the model is more confident after MAML; a positive $\Delta$ means less confident.

Case 1: Prototype Amplification. Sentence: “Inahit ni John ang sarili niya.” (Gloss: “John shaved himself.”)

The first case study demonstrates how MAML strengthens recognition of cross-linguistically common proper names. In this example, MAML sharply reduces surprisal on “John,” indicating stronger prototype activation.

We suspect improvement operates at two levels: (1) lexical level, in the sense that the token "John" becomes more strongly associated with person entities through meta-learning’s emphasis on rapid adapta tion to new entities, and (2) contextual level, in the sense that the ni + proper-name pattern gets reinforced as a reliable PER indicator during meta-training episodes.

Case 2: Contextual Suppression (Loss). Sentence: “Malapit kay Maria si Juan.” (Gloss: “Juan is close to Maria.”)

The second case study reveals MAML’s limitations with complex multi-token constructions. Here, ∆ is positive for key tokens, showing that MAML reduces confidence in the correct label. In "Malapit kay Maria si Juan" (Juan is close to Maria), both the locative adverb "Malapit" (close/near) and the oblique case marker "kay" show substantially decreased confidence for location labeling under MAML (combined decrease of approximately −3.3 log-probability points).

We suspect this occurs due to: (1) capacity constraints, in the sense that the frozen backbone has limited representational capacity, and strengthening PER features may crowd out LOC/ORG representations, and (2) training signal imbalance, in the sense that finetuning contained more person-like entities than complex locative expressions, biasing the learned representations toward single-token person recognition.

## ∆ log-prob by Entity Class (Example 4)

∆ log-prob by Entity Class (Example 7)  
![](images/3777d94272c114115081996c7b4ed17205c086428a1da21b977af68a4b11ea0f.jpg)

<details>
<summary>heatmap</summary>

| Language | PER | LOC | ORG |
| :--- | :--- | :--- | :--- |
| Inahit | ~0.1 | ~0.1 | ~0.1 |
| ki | ~1.2 | ~0.1 | ~0.1 |
| John | ~0.1 | ~0.1 | ~0.1 |
| ang | ~0.1 | ~-1.8 | ~0.1 |
| sarili | ~0.1 | ~0.1 | ~0.1 |
| niya | ~0.1 | ~0.1 | ~0.1 |
| Malapit | ~0.1 | ~0.1 | ~1.2 |
| kay | ~0.1 | ~0.1 | ~0.1 |
| Maria | ~0.1 | ~0.1 | ~0.1 |
| si | ~0.1 | ~-0.5 | ~0.1 |
| Juan | ~0.1 | ~0.1 | ~0.1 |
</details>

(a) Prototype Amplification.  
(b) Contextual Suppression.

Figure 9: MAML’s impact on (a) single-token prototype confidence and (b) multi-token contextual cue sensitivity.

Below is the pseudocode for the MAML and vanilla pretraining setup.

## Distributed Subset Masked Language Modeling Tasks (SMLMT) Training

Algorithm 1 Distributed SMLMT Loop

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
// Initialization: same as Alg. 2, plus
initialize inner-optimizer SGD on head $h_{\phi}$
step $\leftarrow 0$
for each sub_batch in dataloader do
    // gather across GPUs
    $X \leftarrow$ fabric.all_gather(sub_batch["input_ids"])
    // sync random branch decision
    $r \leftarrow$ Uniform(0, 1); $r \leftarrow$ fabric.broadcast(r)
    if $r &lt; \rho$ then
        // Meta-learning episode
        $(S, Q)$, labels$_{S}$, labels$_{Q} \leftarrow$ mask_tokens($X$)
        $\phi_{0} \leftarrow \phi$ $\triangleright$ snapshot head params
        for $t = 1$ to $T_{inner}$ do
            $\ell_{S} \leftarrow \text{CE}(h_{\phi_{t-1}}(f_{\theta}(S)), \text{labels}_{S})$ $\phi_{t} \leftarrow \phi_{t-1} - \alpha \nabla \ell_{S}$ $\triangleright$ inner SGD
        end for
        $\ell_{Q} \leftarrow \text{CE}(h_{\phi_{T}}(f_{\theta}(Q)), \text{labels}_{Q})$ $\phi \leftarrow \phi_{0}$ $\triangleright$ restore head
        fabric.backward($\ell_{Q}$/accum_steps)
    else
        // Standard AR
        $X_{\text{in}}, Y \leftarrow X[; : -1], X[; 1:]$ $\ell_{\text{AR}} \leftarrow \text{CE}(f_{\theta}(X_{\text{in}}), Y)$
        fabric.backward($\ell_{\text{AR}}$/accum_steps)
    end if
    // outer-step and logging
    if (step+1) % accum_steps == 0 then
        opt.step(); scheduler.step(); opt.zero_grad()
        // aggregate metrics across GPUs
        log_loss $\leftarrow$ fabric.all_reduce($\ell$)
        fabric.log(...)
        fabric.barrier()
    end if
    step += 1
end for
</div>

Algorithm 2 Distributed AR Loop

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
// Initialization (in Trainer.__init__):
Load configs; initialize Fabric, tokenizer, model $f_\theta$
(model, opt) ← fabric.setup($f_\theta$, AdamW)
dl ← base dataloader; dl ← fabric.setup_dataloaders(dl)
step ← 0; zero gradients
for each sub_batch in dl do
    // Gather full batch across GPUs if needed:
        X ← fabric.all_gather(sub_batch["input_ids"])
        $X_{\text{in}}$, Y ← X[;; -1], X[; 1:]
    // forward + loss
    $\ell \leftarrow \text{CE}(f_\theta(X_{\text{in}}), Y)$
    // backward (handles synchronization)
    fabric.backward($\ell/\text{accum\_steps}$)
    // outer-step when accumulated
    if (step+1) % accum_steps == 0 then
        opt.step(); scheduler.step(); opt.zero_grad()
        // optional barrier
        fabric.barrier()
    end if
    step += 1
end for
</div>

## C.1 Multi-GPU processing

Pico already uses Lightning-Fabric data parallelism but meta-learning introduces various demands that make multi-GPU processing complicated. A Bernoulli draw is done on one GPU and broadcast so all ranks choose the same objective. Support and query tensors are constructed on rank 0 then scattered, because per-rank random masks would destroy gradient equivalence. Every GPU performs the same ten head updates before any gradient is communicated. A stray early all\_reduce would mix gradients from different inner steps, so we place an explicit barrier between inner and outer phases.

## D Universal NER Datasets

To comprehensively evaluate the pretraining method, each permutation of finetuning setup ({head-only, full}, finetuning dataset ({da\_ddt, . . . , zh\_gsdsimp, all}) (where all consists of all available training sets), model size ({tiny, small, medium, large}), and pretraining setup ({vanilla, MAML}) is evaluated, for a total of 160 evaluation runs.

• Publicly Available In-language treebanks (9 langs): full train/dev/test splits, identical to the official UD partitions.

– da\_ddt, en\_ewt, hr\_set, pt\_bosque, sk\_snk, sr\_set, sv\_talbanken, zh\_gsd, zh\_gsdsimp

• Parallel UD (PUD) evaluation (6 langs): single test.txt files, all sentence-aligned across German, English, Portuguese, Russian, Swedish and Chinese.

– de\_pud, en\_pud, pt\_pud, ru\_pud, sv\_pud, zh\_pud

• Other eval-only sets (3 langs): small test splits for low-resource languages.

– ceb\_gja (Cebuano), tl\_trg (Tagalog TRG), tl\_ugnayan (Tagalog Ugnayan)

## D.1 Slovak Fine-Tune Token Statistics

<table><tr><td>Entity</td><td># spans</td><td>% single-token</td></tr><tr><td>PER</td><td>2 277</td><td>87.6 %</td></tr><tr><td>LOC</td><td>277</td><td>75.1 %</td></tr><tr><td>ORG</td><td>153</td><td>56.9 %</td></tr></table>

Table 4: Span statistics for the Slovak finetune set (sk\_snk train). The data are strongly person-heavy and person spans are almost always single words, whereas locations and organisations are both rarer and more often multi-token.

## D.2 Tagalog and Cebuano Particle and Out-of-Vocabulary Statistics

<table><tr><td>Language</td><td>Particle recall</td><td>OOV rate</td></tr><tr><td>Tagalog</td><td>0.113 ± 0.000</td><td>0.523 ± 0.000</td></tr><tr><td>Cebuano</td><td>0.058 ± 0.000</td><td>0.534 ± 0.000</td></tr></table>

Table 5: Mean $( \pm \mathrm { s . d . }$ . across checkpoints) of particle–preceding-span recall and token out-of-vocabulary rate, measured on the zero-shot evaluation sets after Slovak head-only tuning. “Particle recall” is the fraction of gold PER entities whose left context token is a Filipino case particle recognised by the model.

## E Pretraining Results

We present the unedited pretraining indicators for each pico-maml-decoder model below, as logged on WandB.

![](images/bc9d2a0d640786a04524dd3b32cf47aff9fcfd68f3632fbaeb42f61ccbc0e24b.jpg)

<details>
<summary>line</summary>

| Global Step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| 0 | ~6.8 | ~6.8 | ~7.4 | ~7.4 |
| 2k | ~3.8 | ~4.0 | ~4.2 | ~5.2 |
| 4k | ~3.6 | ~3.7 | ~3.9 | ~4.6 |
| 6k | ~3.5 | ~3.6 | ~3.8 | ~4.5 |
</details>

Figure 10: Pretraining training loss curve.

![](images/04f4ea330ff825e3a7a8b514f3a2d955fd8ce11feaa75752f3025069a0946f00.jpg)

<details>
<summary>line</summary>

| trainer/global_step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| 0 | ~58000 | ~58000 | ~58000 | ~58000 |
| ~200 | ~3000 | ~4000 | ~5000 | ~55000 |
| ~400 | ~2000 | ~2500 | ~3000 | ~15000 |
| ~600 | ~1500 | ~1800 | ~2000 | ~6000 |
| ~800 | ~1200 | ~1400 | ~1500 | ~4000 |
| ~1k | ~1000 | ~1200 | ~1200 | ~3500 |
| ~2k | ~800 | ~1100 | ~1100 | ~2500 |
| ~2.3k | — | ~4500 | — | — |
| ~2.7k | ~4500 | — | — | — |
| ~3.8k | ~2500 | — | — | — |
| ~6.5k | — | — | — | — |
| ~7.5k | — | — | — | ~2500 |
</details>

Figure 11: PALOMA score over pretraining steps.

![](images/ac69e60eb613f95fa23a2e2fb5c1c5894a4e255c709ae5c1fd645336c1919a87.jpg)

<details>
<summary>line</summary>

| trainer/global_step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| ~6500 | ~0.52 | ~0.48 | ~0.47 | ~0.07 |
</details>

Figure 12: Query accuracy during pretraining.

![](images/c61e973d8acbb29e42b3bf7b9e2d8e9ffbf672a2d5de2999d93b5263d9db7740.jpg)

<details>
<summary>line</summary>

| trainer/global_step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| ~6.8k | — | ~0.055 | ~0.063 | ~0.053 |
</details>

Figure 13: Support accuracy over pretraining.

![](images/54959ed7a705ef310120e492859b068c2f9fbb9f37de7d6e7036805c0ba58672.jpg)

<details>
<summary>line</summary>

| trainer/global_step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| ~6.5k | ~0.0005 | ~0.0006 | ~-0.0004 | ~-0.0075 |
</details>

Figure 14: Mean of weights in classifier head over pretraining.

![](images/d955f570815ddd11a0f0b45b5cec88251d2d2c2c7a40da66c1c94634c4e04c12.jpg)

<details>
<summary>line</summary>

| trainer/global_step | pico-maml-decoder-large-1 | pico-maml-decoder-medium-2 | pico-maml-decoder-small-1 | pico-maml-decoder-tiny-1 |
| --- | --- | --- | --- | --- |
| ~6.5k | 0.0143 | 0.0147 | 0.0181 | 0.0165 |
</details>

Figure 15: Standard deviation of weights in classifier head over pretraining.

F Default pico-maml-train Configurations

<table><tr><td>Category</td><td>Parameter</td><td>Default Value</td></tr><tr><td rowspan="10">Model</td><td>Model Type</td><td>pico_decoder</td></tr><tr><td>Hidden Dimension ( $d_{model}$ )</td><td>768</td></tr><tr><td>Number of Layers ( $n_{layers}$ )</td><td>12</td></tr><tr><td>Vocabulary Size</td><td>50,304</td></tr><tr><td>Sequence Length</td><td>2,048</td></tr><tr><td>Attention Heads</td><td>12</td></tr><tr><td>Key/Value Heads</td><td>4</td></tr><tr><td>Activation Hidden Dim</td><td>3,072</td></tr><tr><td>Normalization Epsilon</td><td> $1 \times 10^{-6}$ </td></tr><tr><td>Positional Embedding Theta</td><td>10,000.0</td></tr><tr><td rowspan="7">Training</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>LR Scheduler</td><td>Linear w/ Warmup</td></tr><tr><td>Warmup Steps</td><td>2,500</td></tr><tr><td>Gradient Accumulation Steps</td><td>128</td></tr><tr><td>Max Training Steps</td><td>200,000</td></tr><tr><td>Precision</td><td>BF16 Mixed</td></tr><tr><td rowspan="3">Data</td><td>Dataset Name</td><td>pico-lm/pretokenized-dolma</td></tr><tr><td>Batch Size</td><td>1,024</td></tr><tr><td>Tokenizer</td><td>allenai/OLMo-7B-0724-hf</td></tr><tr><td rowspan="4">Checkpointing</td><td>Auto Resume</td><td>True</td></tr><tr><td>Save Every N Steps</td><td>100</td></tr><tr><td>Learning Dynamics Layers</td><td>&quot;attention.v_proj&quot;, &quot;attention.o_proj&quot;, &quot;swiglu.w_2&quot;</td></tr><tr><td>Learning Dynamics Eval Data</td><td>pico-lm/pretokenized-paloma-tinsy</td></tr><tr><td rowspan="3">Evaluation</td><td>Metrics</td><td>[&quot;paloma&quot;]</td></tr><tr><td>Paloma Dataset Name</td><td>pico-lm/pretokenized-paloma-tinsy</td></tr><tr><td>Eval Batch Size</td><td>16</td></tr><tr><td rowspan="2">Monitoring</td><td>Logging Level</td><td>INFO</td></tr><tr><td>Log Every N Steps</td><td>100</td></tr><tr><td rowspan="10">Meta-Learning</td><td>Enabled</td><td>True</td></tr><tr><td>Hybrid Ratio</td><td>0.5</td></tr><tr><td>Inner Steps ( $k$ )</td><td>10</td></tr><tr><td>Inner Learning Rate</td><td>0.001</td></tr><tr><td>Support Shots ( $k$ )</td><td>4</td></tr><tr><td>Query Ways ( $n$ )</td><td>32</td></tr><tr><td>Classifier Head Layers</td><td>4</td></tr><tr><td>Classifier Head Hidden Dim</td><td>128</td></tr><tr><td>Classifier Head Dropout</td><td>0.1</td></tr><tr><td>Classifier Head Init Method</td><td>xavier</td></tr><tr><td rowspan="2">Monitoring</td><td>Logging Level</td><td>INFO</td></tr><tr><td>Log Every N Steps</td><td>100</td></tr></table>

Table 6: Default configuration settings used in pico-maml-train.

<table><tr><td colspan="5">Pico-MAML-Decoder Model Comparison</td></tr><tr><td>Attribute</td><td>tiny</td><td>small</td><td>medium</td><td>large</td></tr><tr><td>Parameter Count</td><td>11M</td><td>65M</td><td>181M</td><td>570M</td></tr><tr><td>Hidden Dimension ( $d_{model}$ )</td><td>96</td><td>384</td><td>768</td><td>1536</td></tr><tr><td>Feed-forward Dim</td><td>384</td><td>1536</td><td>3072</td><td>6144</td></tr><tr><td>Training Time (6k steps)</td><td>10h</td><td>15h</td><td>16h</td><td>25h</td></tr></table>

Table 7: Comparison of pico-maml-decoder model variants trained with default pico-maml-train configurations. Except for hidden and feed-forward dimension, all models share the training settings detailed in 6. Models were trained for 6000 training steps on 4 NVIDIA A100-SXM4-80GB GPUs; the listed training times correspond to the initial 6000 steps.