# <span id="page-0-0"></span>Meta-Pretraining for Zero-Shot Cross-Lingual Named Entity Recognition in Low-Resource Philippine Languages

David Demitri Afric[a](#page-0-0)\* Suchir Salhan Yuval Weiss Paula Buttery Richard Diehl Martinez University of Cambridge

# Abstract

Named-entity recognition (NER) in lowresource languages is usually tackled by finetuning very large multilingual LMs, an option that is often infeasible in memory- or latencyconstrained settings. We ask whether small decoder LMs can be pretrained so that they adapt quickly and transfer zero-shot to languages unseen during pretraining. To this end we replace part of the autoregressive objective with firstorder model-agnostic meta-learning (MAML). Tagalog and Cebuano are typologically similar yet structurally different in their actor/non-actor voice systems, and hence serve as a challenging test-bed. Across four model sizes (11 M – 570 M) MAML lifts zero-shot micro-F<sup>1</sup> by 2–6 pp under head-only tuning and 1–3 pp after full tuning, while cutting convergence time by up to 8%. Gains are largest for single-token person entities that co-occur with Tagalog case particles *si*/*ni*, highlighting the importance of surface anchors.

![](_page_0_Picture_6.jpeg)

[davidafrica/pico-maml](https://huggingface.co/collections/davidafrica/pico-maml-68e7703c4a40343e852db50f)

[DavidDemitriAfrica/pico-maml-train](https://github.com/DavidDemitriAfrica/pico-maml-train/tree/main)

# 1 Introduction

Named-entity recognition (NER) locates and categorises Persons (PER), Organisations (ORG) and Locations (LOC) in unstructured text [\(Chinchor and](#page-9-0) [Robinson,](#page-9-0) [1997\)](#page-9-0). It is used in a variety of important domains such as healthcare [\(Kundeti et al.,](#page-9-1) [2016;](#page-9-1) [Polignano et al.,](#page-10-0) [2021;](#page-10-0) [Shafqat et al.,](#page-10-1) [2022\)](#page-10-1) and law [\(Leitner et al.,](#page-9-2) [2019;](#page-9-2) [Au et al.,](#page-8-0) [2022;](#page-8-0) [Naik](#page-10-2) [et al.,](#page-10-2) [2023\)](#page-10-2), yet progress remains concentrated in a handful of well-resourced languages. Cross-lingual named-entity recognition is therefore important to better serve underserved communities, yet recent advancements remain unevenly distributed since

NER performance in many languages remains poor due to limited training resources.

A key challenge is that entity boundaries and categories are not universal: languages differ in their morphosyntactic cues, word order, and orthographic conventions. Models trained primarily on Indo-European data thus fail to generalize reliably to underrepresented settings. In this paper, we address this problem through meta-pretraining: shaping language model initializations to adapt rapidly to new linguistic conditions. Unlike standard pretraining, which minimizes average loss over a static corpus, episodic meta-pretraining (e.g. via MAML; [Finn et al.](#page-9-3) [2017\)](#page-9-3) explicitly optimizes for fast transfer. For low-resource NER, this offers two potential benefits: (i) rapid adaptation to languages with typologically distinct cues (e.g. case particles, voice systems, code-switching), and (ii) stronger zero-shot prototypes for common entity types, even without in-language exposure. While meta-learning has been explored for classification tasks in English or cross-lingually at BERT scale [\(Wu et al.,](#page-10-3) [2020;](#page-10-3) [Li et al.,](#page-9-4) [2020;](#page-9-4) [de Lichy et al.,](#page-9-5) [2021\)](#page-9-5), its efficacy for small decoder LMs and morphologically rich languages is underexplored.

As a case study, we focus on NER in Tagalog and Cebuano, the two most widely spoken Philippine languages [\(Miranda,](#page-10-4) [2023\)](#page-10-4). Typologically, both languages combine Austronesian features such as voice alternations, case particles, and reduplication with pervasive borrowing and code-switching (Figure [8;](#page-11-0) Table [1\)](#page-1-0). These languages stress-test whether meta-pretraining can yield more adaptable NER representations than vanilla pretraining alone. We ask the following research questions:

RQ1 Efficacy. How much does first-order MAML improve zero-shot NER on Tagalog and Cebuano relative to vanilla autoregressive pretraining?

<sup>\*</sup>Corresponding Author:

[david.demitri.africa@gmail.com](mailto:david.demitri.africa@gmail.com)

<span id="page-1-0"></span>

Table 1: A selection of Typological Features of Tagalog and Cebuano relevant for NER. ✓ indicates strong presence, ✗ indicates reduced/less overt presence in each language. We highlight high divergence features, moderate divergence and similar features compared to Indo-European Languages, motivating these languages as a case-study for low-resourced NER. We provide a more detailed comparison along with an illustrative gloss in Appendix [A.](#page-11-1)

phological cues, and lexical patterns (especially those tied to Tagalog/Cebuano typology) explain the observed gains or failures?

We answer these questions by systematically comparing first-order MAML and vanilla pretraining on LLaMa-style Pico Decoders across scales, analyzing both downstream performance and representation dynamics [\(Diehl Martinez,](#page-9-6) [2025;](#page-9-6) [Mar](#page-9-7)[tinez et al.,](#page-9-7) [2025\)](#page-9-7). This allows us to investigate:

RQ3 How does the effect of meta-pretraining vary with model size? Are benefits stronger at small scales, or do they persist as capacity increases?

## 1.1 Contributions.

We provide the following contributions:

- A systematic evaluation of meta-pretrained small decoder LMs for zero-shot NER in Tagalog and Cebuano, comparing against strong vanilla pretraining baselines across four model scales.
- Quantitative and qualitative evidence that MAML-based meta-pretraining produces sharper single-token entity prototypes, improving zero-shot NER, especially for person entities and Tagalog's particle-rich syntax.
- An analysis of failure modes and learning dynamics, showing the capacity-dependent nature of meta-learning gains and the tradeoff between prototype sharpening and contextual generalization.

## 2 Method

### 2.1 Motivation

Why these two languages? Tagalog and Cebuano are used every day by well over 100 million people. However, they occupy only a small fraction of the web text that current language models are pretrained on, which makes them both socially important and under-served by existing NLP tools [\(Miranda,](#page-10-4) [2023\)](#page-10-4). Linguistically, these languages also offer complementary typological challenges for NER, which we summarise in Figure [1.](#page-1-0) Tagalog and Cebuano combine Austronesian voice systems, case particles, reduplication, and discourse-driven topic marking in ways that are rare in widely studied NLP benchmarks. In particular, Tagalog offers more overt morphosyntactic cues than Cebuano: it retains a four-way actor/non-actor voice paradigm, while Cebuano reduces this to two [\(Tanangkings](#page-10-5)[ing,](#page-10-5) [2011\)](#page-10-5) and marks syntactic roles with case particles (*si/ni/ang/ng/sa*). These languages offer a test bed for multilingual NER models that must generalize beyond Indo-European NER cues – where entities are typically identifiable through fixed word order and stable orthography– to handle the interaction of morphological marking, argument interaction and code-switching. Tagalog contains more Spanish loans and code-switching into English, while Cebuano maintains a more conservative Austronesian lexicon [\(Bautista,](#page-8-1) [2004;](#page-8-1) [Bak](#page-8-2)[lanova,](#page-8-2) [2019\)](#page-8-2). We provide a more detailed comparison of Tagalog and Cebuano typological features in Table [3.](#page-11-2)

Why Meta-learning? Being underrepresented in natural language processing (NLP) corpora [\(Ca](#page-8-3)[jote et al.,](#page-8-3) [2024;](#page-8-3) [Quakenbush,](#page-10-6) [2005;](#page-10-6) [Dita et al.,](#page-9-8) [2009;](#page-9-8) [Bandarkar et al.,](#page-8-4) [2024\)](#page-8-4), Philippine language datasets suffer from size and quality issues. In lowresource settings, where pretraining data is scarce or absent, it is important to ask the question: will a given checkpoint finetune or transfer rapidly when exposed to a novel language (such as in deployment)?

Meta-learning addresses this by shaping initializations for quick adaptation. Model-Agnostic Meta-Learning (MAML) optimizes an LM backbone so that a few gradient steps yield high performance on a new task [\(Finn et al.,](#page-9-3) [2017\)](#page-9-3). We ask whether such an initialization, learned entirely without Tagalog/Cebuano exposure, can transfer to these languages' distinct morphological and lexical

cues for NER. Our working hypothesis is that a pretraining routine that is itself optimized for rapid adaptation will induce representations that generalize more readily across languages. Prior NLP studies have tested this mostly on English or on "BERT-scale" encoder models [\(Wu et al.,](#page-10-3) [2020;](#page-10-3) [Ma](#page-9-9) [et al.,](#page-9-9) [2022;](#page-9-9) [Li et al.,](#page-9-4) [2020;](#page-9-4) [de Lichy et al.,](#page-9-5) [2021\)](#page-9-5); we explore whether episodic meta-pretraining of small decoder LMs, without any exposure to Tagalog or Cebuano, can still yield zero-shot gains for NER. We do not evaluate a multilingual languagemodel baseline, as our objective is to isolate the effect of episodic meta-pretraining under a matched corpus and schedule; training a competitive multilingual baseline would require different data and budgets, confounding a like-for-like comparison.

Our working hypothesis is that a pretraining routine that is itself optimized for rapid adaptation will induce representations that generalize more readily across languages, so that a model exposed only to high-resource sources can still zero-shot transfer to typologically distant, low-resource targets.

# 2.2 Architecture

We build upon the PICO decoder stack [\(Diehl Mar](#page-9-6)[tinez,](#page-9-6) [2025\)](#page-9-6), a LLaMa-style causal Transformer implemented in PyTorch. Four capacity tiers (tiny (11 M), small (65 M), medium (181 M) and large (570 M)) share all hyper-parameters except hidden width <sup>d</sup> <sup>∈</sup> {96, <sup>384</sup>, <sup>768</sup>, <sup>1536</sup>}. Each model comprises L=12 RMS-normalised decoder blocks [\(Zhang and](#page-10-7) [Sennrich,](#page-10-7) [2019\)](#page-10-7) with grouped-query self-attention [\(Ainslie et al.,](#page-8-5) [2023\)](#page-8-5), RoPE positions [\(Su et al.,](#page-10-8) [2024\)](#page-10-8) and SwiGLU feed-forwards [\(Shazeer,](#page-10-9) [2020\)](#page-10-9) that expand to 4d.

# 2.3 Hybrid pretraining objective

Training alternates between two outer-loop updates:

- 1. Autoregressive LM step. Standard nexttoken prediction on a pre-tokenized version of Dolma [\(Soldaini et al.,](#page-10-10) [2024\)](#page-10-10) released by the Pico library [\(Diehl Martinez,](#page-9-6) [2025\)](#page-9-6).
- 2. First-order MAML episode. A 32-way, 4-shot Subset-Masked LM Task (SMLMT; [Bansal et al.,](#page-8-6) [2020\)](#page-8-6) is sampled, where the model predicts a masked token from the corpus on the fly. The inner loop finetunes a lightweight MLP head for ten SGD steps (α=10−<sup>3</sup> ) and the outer loop back-propagates the query loss through the frozen backbone.

The branch decision is a Bernoulli draw with probability ρ=0.5, synchronised across four A100- 80 GB GPUs. The pseudocode for both can be found in Appendix [C.](#page-14-0)

## 2.4 Optimisation and monitoring

We run 6,000 outer updates with AdamW (ηpeak = <sup>3</sup>×10−<sup>4</sup> , 2.5 k warm-up, cosine decay), accumulating eight micro-batches of 256 sequences to reach an effective batch of 2048 sequences (1024 for tiny). Every 100 steps we log: Paloma perplexity [\(Magnusson et al.,](#page-9-10) [2024\)](#page-9-10), singular-value spectra of three attention and three feed-forward weight matrices, from which we compute proportional effective rank (PER; [Diehl Martinez et al.,](#page-9-11) [2024\)](#page-9-11), and support and query accuracy within MAML episodes.

# 2.5 Finetuning on High-Resourced Languages

We deliberately choose high-resource languages as the finetuning sources because, in realistic deployments, these are the languages for which sizable, high-quality NER data already exists. They therefore form the most natural setting for cross-lingual transfer into low-resource settings.

After pretraining we attach an untrained linear conditional random field head [\(Lafferty et al.,](#page-9-12) [2001\)](#page-9-12), which is a well-known method used often for NER [\(Bundschus et al.,](#page-8-7) [2008;](#page-8-7) [Ma and Hovy,](#page-9-13) [2016\)](#page-9-13). We finetune on a high-resource language (Danish, English, Croatian, Portuguese, Slovak, Serbian, Swedish, Chinese, Chinese-Simplified, and a mixture of all languages) before zero-shot evaluation on Tagalog (tl\_trg, tl\_ugnayan) and Cebuano (ceb\_gja) from Universal NER v1 [\(May](#page-9-14)[hew et al.,](#page-9-14) [2024\)](#page-9-14). Results are later broken down by finetuning language. Further, two finetuning regimes are compared: head-only, where the transformer is frozen and only the classifier learns, and full, where all parameters are freed to update.

Finetuning uses AdamW (3×10−<sup>5</sup> ) for up to ten epochs with early stopping on development F1. We report micro-F1, with full details in Appendix [D.](#page-15-0)

# 2.6 Baselines

For each capacity tier we also evaluate a "vanilla" Pico model (no MAML, pure autoregressive loss) under identical data, schedule and compute. Pretraining results can be found in Appendix [E](#page-17-0) with model configuration details in Appendix [F.](#page-20-0) A more detailed discussion of pretraining results and overall methodology can be found in [Africa et al.](#page-8-8) [\(2025\)](#page-8-8).

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 1: Scale curve. Zero-shot Micro-F<sup>1</sup> on Cebuano & Tagalog versus parameter count. Bars compare PICO-MAML (blue) to vanilla pretraining (green); the overlaid line shows the relative gain of MAML (Delta F1, right axis). Meta-pretraining helps at every scale, but the relative lift shrinks from +38 % (11 M) to +6 % (570 M), revealing a capacity threshold below which the inner loop cannot extract reusable features.

# 3 Zero-Shot Transfer Results

Zero-shot evaluation. Unless stated otherwise, all scores are obtained without seeing any Tagalog/Cebuano data during finetune, relying solely on the UNER test sets (§ 2.4).

Figure [1](#page-3-0) shows that PICO-MAML improves Cebuano/Tagalog micro-F<sup>1</sup> at every parameter budget. The relative lift is largest for moderate sizes and tapers with scale (+6% at 570M). These results indicate that adding a single outer-loop meta-update per batch yields a cross-lingual prior not captured by vanilla pretraining under our setup.

Comparison of head-only tuning and full tuning. Decomposing by finetuning regime (Fig. [2\)](#page-3-1), MAML yields 1–2 pp gains when only the CRF head is trained, implying that the frozen weights already embeds better entity cues. Full tuning narrows the gap to 0.5–1.3 pp, indicating that the lift persists even when the optimiser is free to overwrite the initialisation.

Further, results indicate that the benefit provided by the meta-objective is scale-dependent. For the 11 M (tiny) model, MAML moves the overall score by < 1 pp and yields no gain under head-only tuning. From 65 M parameters upward the benefit becomes clearer with larger head-only lifts, suggesting a threshold at which meta-gradients can provide reusable entity features without crowding out the LM signal.

Sensitivity to finetuning language. Figure [3](#page-4-0) profiles performance after adapting on nine highresource languages. Eight of nine languages exhibit positive deltas; the largest relative lifts occur for

<span id="page-3-1"></span>![](_page_3_Figure_1.jpeg)

Figure 2: Impact of finetuning regime. Head-only tuning (left) magnifies the meta-learning advantage up to +2.5 pp at 570 M, likely because the backbone must already encode entity cues. Full tuning (right) reduces but does not erase the gap, suggesting that MAML primarily accelerates convergence rather than acting as a regulariser.

Slovak (+18 %) and Croatian (+13 %). Gain in Slovak might be due to fixed case endings that consistently bracket entity names, providing a clear surface boundary signal for the model (similar in function to Tagalog's case particles but realised morphologically rather than syntactically.) The sole regression (–2 pp on Simplified Chinese) is most likely due to a known issue in poor crossscript transfer to Chinese, but it may also be due to subword sparsity in the shared vocabulary rather than a failure of the meta-objective. [\(Mayhew et al.,](#page-9-14) [2024\)](#page-9-14).

Overall, MAML appears to teach the model to exploit shallow lexical anchors (particles, affixes) that generalise well across Indo-European languages while still transferring to more typologically distant Austronesian targets. To better understand the mechanisms underlying these gains, we conduct a focused qualitative analysis on a representative configuration.

# 4 Analysis of MAML Pretrained Models

In order to analyze the learning process, rather than just the last checkpoint, we focus our qualitative study on a MEDIUM-sized model (181 M parameters) finetuned in a head-only regime on Slovak (sk\_snk), finetuning on all 61 checkpoints from step 0 of pretraining to step 6000. We restrict our analysis to this slice because while finetuning 9760 (2 pretraining regimes x 2 finetuning regimes x 4 model sizes x 10 finetuning languages x 61 checkpoints) models would be prohibitively expensive, this configuration at least offers a reasonable signalto-cost trade-off. This is for a few reasons: (i) the medium tier is the smallest model that still exhibits a clear 2–3 pp head-only lift (Figure [1\)](#page-3-0) yet is three-

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 3: Sensitivity to finetuning language. Grid of zero-shot F<sup>1</sup> curves after adapting on nine high-resource languages plus an *All*-languages mixture. Eight of nine languages show positive deltas; the largest relative gains occur for Slovak and Croatian, while Simplified Chinese is the lone outlier (–2 pp). This pattern indicates that the meta-objective encourages reliance on surface affixes and particles that generalise well across Indo-European sources yet still transfer to Austronesian targets.

times cheaper to run than the 570 M variant, (ii) Slovak delivers one of the largest relative gains without vocabulary sparsity issues and, as a Slavic language, should produce transfer errors that differ sharply from those in Tagalog and Cebuano, and (iii) freezing the backbone during head-only finetuning ensures that any performance delta must stem from representations learned during metapretraining rather than from subsequent weight updates. In the next subsection, we inspect how pretraining affects finetuning performance across checkpoints.

## 4.1 Checkpoint Analysis

Does the head-only learner actually learn? Figure [4](#page-5-0) overlays the complete finetuning trajectories for every Slovak head-only run (61 checkpoints, maml\_s0000–maml\_s6000). Viridis traces show the individual runs (getting darker the later the model checkpoint was taken), while the bold line and ribbon denote the median and inter-quartile range (IQR). The train-loss fan collapses to its asymptote within the first ≈ 800 steps and stays flat thereafter; in parallel the evaluation F<sup>1</sup> rises smoothly to 0.14 and plateaus with a narrow ±0.01 IQR. Crucially, no run diverges or oscillates, confirming that freezing the backbone and training only a linear chain CRF head is both stable and something is learned. This satisfies the prerequisite for using the configuration as a clean test-bed: any downstream difference between MAML and vanilla is likely to stem from the initial representations, not from optimisation quirks or training instabilities.

Does meta-pretraining yield transfer-relevant representations? The checkpoint sweep in Figure [5](#page-5-1) confirms the other prerequisite for this qualitative analysis: that meta-pretraining produces representations which become increasingly helpful for zero-shot transfer. First, the top panel shows that, regardless of which MAML snapshot we freeze, the linear chain CRF head always converges to essentially the same narrow band of train loss (0.10-0.15); optimisation is therefore stable and predictable, satisfying our first prerequisite. More importantly, the bottom panel reveals a very different story for cross-lingual evaluation: while Slovak dev F<sup>1</sup> plateaus early (by around step 1k), Tagalog and Cebuano F<sup>1</sup> continue to climb for another four thousand meta-updates, ending 0.15 and 0.12 points higher than at the initial checkpoint. In

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 4: Learning curves for the Slovak head-only setting. Top: train loss; bottom: eval micro-F1. Faint green lines = all individual checkpoints; bold line = median; shaded band = 25–75 % IQR. Both metrics converge monotonically and remain tightly bunched, indicating a stable optimisation surface for the linear head.

other words, additional MAML steps learn features that are invisible to the in-language dev set yet directly benefit unseen Austronesian targets. Tagalog improves earlier and peaks higher than Cebuano, hinting that the meta-objective is capturing surface cues (e.g. case particles) that are more diagnostic in Tagalog. Taken together with the "fan" plot of learning curves, the sweep demonstrates that meta-pretraining yields encoder states that are both optimisation-friendly and transfer-relevant, justifying the focus on this snapshot for deeper qualitative inspection. As such, we deepen the analysis in the next subsection by inspecting the behavior of our models on the level of the NER tags predicted.

### 4.2 Tag-level Analysis

Per-tag behaviour. Figure [6](#page-5-2) reports per-entity F<sup>1</sup> obtained after head-only finetuning the Slovak CRF head on each MAML checkpoint. PER climbs to 0.6-0.7 while LOC and ORG remain at zero. This is not a case of the classifier "over-fitting" in the usual sense—i.e. collapsing to always predicting a single label. A linear-chain CRF is free to emit any BIO tag at any position; if it were truly degenerate we would see train loss stagnate near the log-uniform baseline and the PER curve itself would also be flat. Instead, train loss converges to the same narrow band for every checkpoint (Fig[.4\)](#page-5-0) and PER performance tracks the amount of metapretraining, so the head is learning a genuine decision boundary. It simply has informative features

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 5: Final metrics vs. pretraining checkpoint for the MEDIUM MAML backbone frozen during headonly finetuning on Slovak. Top: final train loss of the CRF head, every run converges to the same narrow range. Bottom: final micro-F<sup>1</sup> on Slovak dev (blue), Tagalog (green) and Cebuano (yellow). Although inlanguage performance saturates early, cross-lingual F<sup>1</sup> keeps improving up to step 6000, indicating that later meta-updates learn representations useful specifically for zero-shot transfer.

<span id="page-5-2"></span>![](_page_5_Figure_5.jpeg)

Figure 6: Per-entity F<sup>1</sup> across MAML checkpoints. PER (dark viridis) improves steadily with more metasteps; LOC and ORG curves remain at chance level, indicating that the frozen backbone provides transferable features for single-token personal names but little for multi-token locations or organisations. Tagalog benefits earlier than Cebuano, consistent with its obligatory case particles.

for people but none for locations or organisations.

Observed imbalance and potential causes. First, the Slovak finetune set is intrinsically personheavy. As Table [4](#page-16-0) shows, PER spans outnumber LOC by roughly 8∶1 and ORG by 15∶1. Under head-only training, every gradient step passes through the frozen encoder unchanged and the CRF receives thousands of positive updates for persons but only a few hundred for the other classes. This likely leads to only the PER decision boundary sharpening. Second, 87.6% of Slovak person mentions are single tokens compared with 75.1 % for locations and 56.9% for organisations. A singletoken span can be captured by one weight vector, whereas multi-word spans require the head to model boundaries and label transitions—a capacity it simply does not have when the encoder cannot adapt. Third, Tagalog still offers a comparatively reliable surface cue. The case particles *si* and *ni* precede roughly 11% of gold PER spans, almost double the 5–6 % rate observed in Cebuano (Table [5\)](#page-16-1). The earlier lift and higher ceiling of the Tagalog PER curve are therefore consistent with the backbone having learned to map the pattern "particle + token" to the PER label, a cue that is informative in Tagalog but is sparser in Cebuano. Finally, cross-lingual lexical overlap is likely higher for personal names, many of which (e.g. *Obama*, *Manuel*) appear verbatim in English corpora used during pretraining; locations and organisations, by contrast, are often translated or abbreviated. All four factors act in the same direction, favouring PER. Disentangling their individual contributions would require targeted ablations (particle masking, balanced resampling, controlled name substitution, etc.) which we leave for future work. In the next subsection, we assess behaviors on the level of words and tokens to relate NER performance to the low-resource languages being transferred to.

### 4.3 Word-level Analysis

Figures [7a–7d](#page-7-0) visualise the checkpoint-bycheckpoint evolution of token-level confidence (p(correct tag)) for the ten most frequent surface words in each evaluation set. Entities and non-entities are split so the dynamic range is not drowned out by O tokens. Two qualitative patterns emerge.

Fast confidence in frequent tokens. Non-entity function words such as *ng, ang, sa* in Tagalog and the Cebuano clitic *-ng* start with high confidence and barely budge after the first 200 metaupdates (Fig. [7b,](#page-7-0) [7d\)](#page-7-0). As these tokens dominate the language-model loss, autoregressive training achieves a high confidence in them early and MAML has little head-room to improve over checkpoints.

Monotonic gains for high-overlap proper names. In the Tagalog set, international names (*City, Maynila, Maria*) and locations transliterated from English (*Pasay*) become steadily brighter (lower loss) until about step 3000 (Fig. [7a\)](#page-7-0). Similar behaviour appears for *Maria, Cebu, Mary* in Cebuano (Fig. [7c\)](#page-7-0). These words either appear verbatim in

<span id="page-6-0"></span>

Table 2: Finetuning convergence speed metrics ∆ (MAML-Vanilla) averaged over nine in-language tasks. The largest and smallest models enjoy the most pronounced speed-ups from full MAML meta-initialization, while medium and tiny models show negligible ∆t<sup>90</sup> under full-model tuning. Under head-only tuning, large and small decoders still benefit modestly, whereas medium and tiny decoders actually slow down. Across all settings, slope remains near zero, indicating that meta-training primarily accelerates mid-to-late convergence rather than the very first gradient steps.

the English Dolma corpus or share sub-tokens (Ma\_, Ceb\_) with it, so the meta-objective can reuse prototypes that happen to be used by the Austronesian targets. The timing matches the checkpoint-sweep (Fig. [5\)](#page-5-1): cross-lingual F<sup>1</sup> continues to climb long after Slovak dev has saturated likely because the back-bone is still lowering loss on these anchor words. We illustrate these mechanisms further in two case studies in Appendix [B.](#page-12-0)

# 5 Finetuning Speed of Meta-Pretraining

Finally, we assess finetuning speed using convergence time (measuring time to achieve 90% of final loss t90), normalized area under the loss curve (measuring aggregate convergence behavior over the curve), and initial slope (measuring the initial speed of learning in the first few steps), as seen in Table [2.](#page-6-0) Across nine in-language tasks, full-model finetuning shows the clearest acceleration for the largest and smallest models: MAML cuts t<sup>90</sup> by roughly 8% (≈ 111 steps) and modestly reduces loss AUC. Medium and small models show negligible or inconsistent speed-ups under full tuning, suggesting that the effect depends strongly on model capacity. In head-only tuning, large and small models again benefit slightly, while medium and tiny models slow down, likely due to underpowered or collapsed meta-dynamics.

Initial slopes remain effectively unchanged across all settings, indicating that MAML does not alter the very first gradient steps but instead reorga-

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 7: Evolution of token-level confidence (p(correct tag)) across pretraining checkpoints. Top row: Tagalog; bottom row: Cebuano. Left: entities only. Right: non-entities.

nizes the loss landscape to make mid- to late-stage convergence more efficient. These results align with earlier findings that MAML's main benefit lies in providing sharper, more reusable token-level features for high-capacity backbones, with limited or negative effects when capacity is insufficient to retain both language modeling and episodic priors.

# 6 Related Work

NER in Filipino, Tagalog, and Cebuano. NER for Philippine languages remains underexplored, with most work focusing on resource construction rather than cross-lingual modeling. Recent corpora include TLUnified-NER [\(Miranda,](#page-10-4) [2023\)](#page-10-4), TF-NERD [\(Ramos and Vergara,](#page-10-11) [2023\)](#page-10-11), CebuaNER [\(Pilar et al.,](#page-10-12) [2023\)](#page-10-12), and UniversalNER [\(Mayhew](#page-9-14) [et al.,](#page-9-14) [2024\)](#page-9-14). Modeling efforts in this area primarily use NER-specific systems [\(Sagum and Sagum,](#page-10-13) [2025;](#page-10-13) [Eboña et al.,](#page-9-15) [2013;](#page-9-15) [Dela Cruz et al.,](#page-9-16) [2018\)](#page-9-16) incorporating a simpler backbone such as a support vector machine [\(Castillo et al.,](#page-8-9) [2013\)](#page-8-9) or an LSTM [\(Chan et al.,](#page-8-10) [2023\)](#page-8-10). Most recently, FilBench [\(Mi](#page-10-14)[randa et al.,](#page-10-14) [2025\)](#page-10-14) and Batayan [\(Montalan et al.,](#page-10-15) [2025\)](#page-10-15) support Filipino evaluation on NLP tasks for LLMs.

Meta-learning for Pretraining. Although most work applies meta-learning at fine-tuning time, a growing line of research embeds meta-objectives directly into pretraining. [\(Raghu et al.,](#page-10-16) [2021\)](#page-10-16) showed that framing parameter-efficient adapter learning as a bilevel problem yields representations that fine-tune more effectively than standard PEFT. [\(Hou et al.,](#page-9-17) [2022\)](#page-9-17) extend this to full transformers. [\(Miranda et al.,](#page-9-18) [2023\)](#page-9-18) argue that explicit MAML objectives can outperform fixed pretraining on highly diverse task distributions. [\(Ke et al.,](#page-9-19) [2021\)](#page-9-19) integrate a MAML-style inner loop into a multi-criteria Chinese Word Segmentation pretraining task.

## 7 Conclusion

This paper shows that MAML-based metapretraining, even when applied to small decoderonly language models, can meaningfully improve zero-shot transfer to low-resource languages, as demonstrated on Tagalog and Cebuano NER. The gains are most pronounced for person entities and head-only finetuning, and scale best with larger model capacities. Our qualitative and word-level analyses reveal that the mechanism of improvement centers on the sharpening of lexical prototypes and better anchoring to surface cues like Tagalog case particles. Hence, we do not expect these improvements to fully generalize to multi-token or highly contextual entity types.

These findings suggest that meta-learning can provide a principled route to more adaptable small models, but also highlight key limitations: the benefits are capacity- and task-dependent, and the current approach struggles with richer entity structures. Future work should explore alternative metalearning objectives, extend to more diverse tasks

- and languages, and investigate the dynamics of prototype formation in even lower-resource settings. Limitations The gains are most pronounced for person entities and head-only finetuning, and scale best with larger model capacities. All training runs stop at exactly six thousand outer steps, a horizon that may be too short for the largest model, so the conclusions derived only cover a fraction of the training budget a corporate setup might have. A more diverse and multilingual corpus may alter both quantitative and qualitative conclusions, and varying languages in the meta-task is a natural way to extend this work. Qualitative analysis was conducted on a single configuration and single seed due to cost and GPU constraints. Qualitative analysis was conducted by a native Tagalog speaker with a register typical of Manila, and a wide variety of perspectives would improve the robustness of the analysis. Finally (and most naturally), our focus on only two Austronesian languages controls for certain lexical and syntactic divergences but limits the generality of the typological conclusions; extending to a broader set of Philippine and Malayo-Polynesian languages is a natural next step. Acknowledgments This work was supported by a grant from the Accelerate Programme for Scientific Discovery, made possible by a donation from Schmidt Futures. David Demitri Africa is supported by the Cambridge Trust and the Jardine Foundation. Suchir Salhan is supported by Cambridge University Press & Assessment. Richard Diehl Martinez is supported by the Gates Cambridge Trust (grant OPP1144 from the Bill & Melinda Gates Foundation). It was performed using resources provided by the Cambridge Service for Data Driven Discovery (CSD3) operated by the University of Cambridge Research Computing Service, provided by Dell EMC and Intel using Tier-2 funding from the Engineering and Physical Sciences Research Council (capital grant EP/T022159/1), and DiRAC funding from the Science and Technology Facilities Coun-
- <span id="page-8-5"></span>cil. References David Demitri Africa, Yuval Weiss, Paula Buttery, and Richard Diehl Martinez. 2025. [Learning dynamics of](https://arxiv.org/abs/2508.02189) [meta-learning in small model pretraining.](https://arxiv.org/abs/2508.02189) *Preprint*, arXiv:2508.02189. Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 4895– 4901. Ting Wai Terence Au, Ingemar J Cox, and Vasileios Lampos. 2022. E-ner–an annotated named entity recognition corpus of legal text. *arXiv preprint arXiv:2212.09306*. Ekaterina Baklanova. 2019. The impact of spanish and english hybrids on contemporary tagalog. Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. [The Belebele Benchmark:](https://aclanthology.org/2024.acl-long.44) [a Parallel Reading Comprehension Dataset in 122](https://aclanthology.org/2024.acl-long.44) [Language Variants.](https://aclanthology.org/2024.acl-long.44) In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 749–775, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics. Trapit Bansal, Rishikesh Jha, Tsendsuren Munkhdalai, and Andrew McCallum. 2020. [Self-supervised meta](https://doi.org/10.18653/v1/2020.emnlp-main.38)[learning for few-shot natural language classification](https://doi.org/10.18653/v1/2020.emnlp-main.38) [tasks.](https://doi.org/10.18653/v1/2020.emnlp-main.38) In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 522–534, Online. Association for Computational Linguistics. Maria Lourdes S Bautista. 2004. Tagalog-english code switching as a mode of discourse. *Asia Pacific Education Review*, 5(2):226–233. Markus Bundschus, Mathaeus Dejori, Martin Stetter, Volker Tresp, and Hans-Peter Kriegel. 2008. Extraction of semantic biomedical relations from text using conditional random fields. *BMC bioinformatics*, 9(1):207. Rhandley D Cajote, Rowena Cristina L Guevara, Michael Gringo Angelo R Bayona, and Crisron Rudolf G Lucas. 2024. Philippine Languages Database: A Multilingual Speech Corpora for Developing Systems for Philippine Spoken Languages. *LREC-COLING 2024*, page 264. Jonalyn M Castillo, Marck Augustus L Mateo, Antonio DC Paras, Ria A Sagum, and Vina Danica F Santos. 2013. Named entity recognition using support vector machine for filipino text documents. *International Journal of Future Computer and Communication*, 2(5):530. Kyle Chan, Kaye Ann De Las Alas, Charles Orcena, Dan John Velasco, Qyle John San Juan, and Charibeth Cheng. 2023. Practical approaches for lowresource named entity recognition of filipino telecom-

<span id="page-8-4"></span><span id="page-8-2"></span><span id="page-8-0"></span>

<span id="page-8-10"></span><span id="page-8-9"></span><span id="page-8-8"></span><span id="page-8-7"></span><span id="page-8-6"></span><span id="page-8-3"></span><span id="page-8-1"></span>

# <span id="page-11-1"></span>A NER-Relevant Typological Features of Cebuano and Tagalog

This extended table highlights how morphosyntactic and discourse-level differences between the two languages interact with the challenges of named entity recognition (NER). We lay out feature-by-feature contrasts to illustrate that even closely related Philippine languages present distinct hurdles for tasks like NER. The table emphasizes that while Tagalog offers overt morphosyntactic cues (e.g., case particles, topic marking), Cebuano relies more heavily on discourse inference, thereby requiring different modeling strategies for effective NER.

<span id="page-11-2"></span>

Table 3: Detailed typological contrasts between Tagalog and Cebuano and their implications for NER.

<span id="page-11-0"></span>

Figure 8: Surface cues for named entities. Tagalog typically provides an overt personal article (*si*/*ni*) before names; Cebuano may show the same article, but zero-marked variants also occur in some registers/contexts, reducing overt anchors.

# <span id="page-12-0"></span>B Case Studies

To illustrate the mechanisms underlying MAML's improvements, we present two contrasting examples that demonstrate how meta-pretraining affects different types of linguistic patterns in Tagalog NER. We measure ∆ log-prob as the change in surprisal (−! log p) for the gold label between the vanilla and MAML model. A negative ∆ means the model is more confident after MAML; a positive ∆ means less confident.

Case 1: Prototype Amplification. Sentence: "Inahit ni John ang sarili niya." (Gloss: "John shaved himself.")

The first case study demonstrates how MAML strengthens recognition of cross-linguistically common proper names. In this example, MAML sharply reduces surprisal on "John," indicating stronger prototype activation.

We suspect improvement operates at two levels: (1) lexical level, in the sense that the token "John" becomes more strongly associated with person entities through meta-learning's emphasis on rapid adaptation to new entities, and (2) contextual level, in the sense that the *ni* + proper-name pattern gets reinforced as a reliable PER indicator during meta-training episodes.

Case 2: Contextual Suppression (Loss). Sentence: "Malapit kay Maria si Juan." (Gloss: "Juan is close to Maria.")

The second case study reveals MAML's limitations with complex multi-token constructions. Here, ∆ is positive for key tokens, showing that MAML reduces confidence in the correct label. In "Malapit kay Maria si Juan" (Juan is close to Maria), both the locative adverb "Malapit" (close/near) and the oblique case marker "kay" show substantially decreased confidence for location labeling under MAML (combined decrease of approximately −3.3 log-probability points).

We suspect this occurs due to: (1) capacity constraints, in the sense that the frozen backbone has limited representational capacity, and strengthening PER features may crowd out LOC/ORG representations, and (2) training signal imbalance, in the sense that finetuning contained more person-like entities than complex locative expressions, biasing the learned representations toward single-token person recognition.

![](_page_13_Figure_0.jpeg)

Figure 9: MAML's impact on (a) single-token prototype confidence and (b) multi-token contextual cue sensitivity.

<span id="page-14-0"></span>

### Distributed Autoregressive (AR) Training

### <span id="page-15-1"></span>Algorithm 2 Distributed AR Loop

1: // *Initialization (in* Trainer.\_\_init\_\_*):* 2: Load configs; initialize Fabric, tokenizer, model f<sup>θ</sup> 3: (model, opt) <sup>←</sup> fabric.setup(<sup>f</sup>θ, AdamW) 4: dl ← base dataloader; dl ← fabric.setup\_dataloaders(dl) 5: step ← 0; zero gradients 6: for each sub\_batch in dl do 7: // *Gather full batch across GPUs if needed:* 8: X ← fabric.all\_gather(sub\_batch["input\_ids"]) 9: <sup>X</sup>in, Y <sup>←</sup> <sup>X</sup>[ <sup>∶</sup>, ∶ −<sup>1</sup> ], X[ <sup>∶</sup>, <sup>1</sup> <sup>∶</sup> ] 10: // forward + loss 11: <sup>ℓ</sup> <sup>←</sup> CE(<sup>f</sup>θ(<sup>X</sup>in), Y ) 12: // backward (handles synchronization) 13: fabric.backward(ℓ/accum\_steps) 14: // outer-step when accumulated 15: if (step+1) % accum\_steps == 0 then 16: opt.step(); scheduler.step(); opt.zero\_grad() 17: // optional barrier 18: fabric.barrier() 19: end if 20: step + = 1 21: end for

## C.1 Multi-GPU processing

Pico already uses Lightning-Fabric data parallelism but meta-learning introduces various demands that make multi-GPU processing complicated. A Bernoulli draw is done on one GPU and broadcast so all ranks choose the same objective. Support and query tensors are constructed on rank 0 then scattered, because per-rank random masks would destroy gradient equivalence. Every GPU performs the same ten head updates before any gradient is communicated. A stray early all\_reduce would mix gradients from different inner steps, so we place an explicit barrier between inner and outer phases.

# <span id="page-15-0"></span>D Universal NER Datasets

To comprehensively evaluate the pretraining method, each permutation of finetuning setup ({head-only, full}, finetuning dataset ({da\_ddt, . . . , zh\_gsdsimp, all}) (where all consists of all available training sets), model size ({tiny, small, medium, large}), and pretraining setup ({vanilla, MAML}) is evaluated, for a total of 160 evaluation runs.

- Publicly Available In-language treebanks (9 langs): full train/dev/test splits, identical to the official UD partitions.
  - da\_ddt, en\_ewt, hr\_set, pt\_bosque, sk\_snk, sr\_set, sv\_talbanken, zh\_gsd, zh\_gsdsimp
- Parallel UD (PUD) evaluation (6 langs): single test.txt files, all sentence-aligned across German, English, Portuguese, Russian, Swedish and Chinese.
  - de\_pud, en\_pud, pt\_pud, ru\_pud, sv\_pud, zh\_pud
- Other eval-only sets (3 langs): small test splits for low-resource languages.
  - ceb\_gja (Cebuano), tl\_trg (Tagalog TRG), tl\_ugnayan (Tagalog Ugnayan)

## <span id="page-16-0"></span>D.1 Slovak Fine-Tune Token Statistics

Table 4: Span statistics for the Slovak finetune set (sk\_snk train). The data are strongly person-heavy and person spans are almost always single words, whereas locations and organisations are both rarer and more often multi-token.

## <span id="page-16-1"></span>D.2 Tagalog and Cebuano Particle and Out-of-Vocabulary Statistics

Table 5: Mean ( ± s.d. across checkpoints) of particle–preceding-span recall and token out-of-vocabulary rate, measured on the zero-shot evaluation sets after Slovak head-only tuning. "Particle recall" is the fraction of gold PER entities whose left context token is a Filipino case particle recognised by the model.

# <span id="page-17-0"></span>E Pretraining Results

We present the unedited pretraining indicators for each pico-maml-decoder model below, as logged on WandB.

![](_page_17_Figure_2.jpeg)

Figure 10: Pretraining training loss curve.

![](_page_17_Figure_4.jpeg)

Figure 11: PALOMA score over pretraining steps.

![](_page_18_Figure_0.jpeg)

Figure 12: Query accuracy during pretraining.

![](_page_18_Figure_2.jpeg)

Figure 13: Support accuracy over pretraining.

![](_page_19_Figure_0.jpeg)

Figure 14: Mean of weights in classifier head over pretraining.

![](_page_19_Figure_2.jpeg)

Figure 15: Standard deviation of weights in classifier head over pretraining.

# <span id="page-20-0"></span>F Default **pico-maml-train** Configurations

<span id="page-20-1"></span>

Table 6: Default configuration settings used in pico-maml-train.

Table 7: Comparison of pico-maml-decoder model variants trained with default pico-maml-train configurations. Except for hidden and feed-forward dimension, all models share the training settings detailed in [6.](#page-20-1) Models were trained for 6000 training steps on 4 NVIDIA A100-SXM4-80GB GPUs; the listed training times correspond to the initial 6000 steps.