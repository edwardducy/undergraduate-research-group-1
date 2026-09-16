# Missing from index.json

Recorded 2026-09-16. Papers referenced or verified during the methodology review and gap verification that are **not** in `corpus-extracted/index.json`.

## A. Already cited in 03-methodology.qmd — add first

- **Tweeting Supertyphoon Haiyan: Evolving Functions of Twitter during and after a Disaster Event** — Clarissa C. David, Jonathan C. Ong, Erika Fille T. Legara (2016). PLOS ONE 11(3): e0150190. [Journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150190) — cited as `@david2016tweeting` (Odette rationale; 87% English / 11% Filipino statistic, verified).
- **ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning** — Junguang Jiang et al. (2023). NeurIPS 2023. [arXiv:2301.12618](https://arxiv.org/abs/2301.12618) — cited as `@jiang2023forkmerge` (near-orthogonality App. B.1; benign-conflict Finding 1, both verified in full text).
- **The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing** — Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart (2018). ACL 2018. [Anthology](https://aclanthology.org/P18-1128/) — cited as `@dror2018hitchhiker`.
- **Show Your Work: Improved Reporting of Experimental Results** — Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith (2019). EMNLP 2019. [Anthology](https://aclanthology.org/D19-1509/) — cited as `@dodge2019show`.
- **With Little Power Comes Great Responsibility** — Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowal, Daniel W. Jurafsky (2020). EMNLP 2020. [Anthology](https://aclanthology.org/2020.emnlp-main.745/) — cited as `@card2020power`.

## B. Used by the methodology but not yet cited anywhere

- **Measuring Nominal Scale Agreement Among Many Raters** — Joseph L. Fleiss (1971). Psychological Bulletin 76(5), 378–382. [DOI](https://doi.org/10.1037/h0031619) — source of Fleiss' κ, the IAA statistic in the annotation protocol. (Index has Landis & Koch 1977 benchmarks only.)
- **Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Efficient and Accurate Long-Context Processing (ModernBERT)** — Benjamin Warner, Antoine Chaffin, Benjamin Wixom, et al. (2024). [arXiv:2412.13663](https://arxiv.org/abs/2412.13663) — the encoder architecture; referenced throughout Chapter 3 with no citation.
- **SEA-LION-ModernBERT-300M model card** — AI Singapore (2026). Hugging Face, MIT license. [Model card](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) — cited as `@aisingapore2026sealion`; specs card-verified 2026-09-16 (312.5M params, 22 layers, d_model 768, vocab 262,144, EN 35% / TL 2.5%).

## C. Gap-verification sources (for Chapter 2 / RD §1 gap paragraph)

- **Disaster Tweet Corpus 2020** — Matti Wiegmann et al. (2020). Zenodo dataset with event-based cross-event splits. [Zenodo](https://zenodo.org/records/3713920)
- **CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises** — Alexandra Olteanu, Carlos Castillo, Fernando Diaz, Sarah Vieweg (2014). ICWSM 2014. [Paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14538) · [Data portal](http://crisislex.org/data-collections.html) — CrisisLexT26 contains Yolanda (38,951 tweets), Pablo, Manila floods, Bohol earthquake (verified).
- **Evaluating the Generalisability of Neural Rumour Verification** — Elena Kochkina et al. (2023). Information Processing & Management. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306457322002175) — leave-one-event-out protocol precedent.
- **Cross-Lingual Query-Based Summarization of Crisis Events** — Yury Vitiugin et al. (2022). SIGIR 2022. [arXiv:2204.10230](https://arxiv.org/abs/2204.10230) — leave-one-event-out protocol precedent.
- **Verifying Information with Multimedia Content on Twitter: A Comparative Analysis** — Christina Boididou et al. (2017). MediaEval. [PDF](https://mever.gr/publications/Verifying%2520information%2520with%2520multimedia%2520content%2520on%2520twitter.pdf) — leave-one-event-out protocol precedent.
- **TriggerCit: Learning Early Detection of Emergencies from Social Media** — Guglielmo Bono et al. (2023). [PDF](https://re.public.polimi.it/bitstream/11311/1247097/1/TriggerCit_v2%20%2811%29.pdf) — leave-one-event-out protocol precedent.
- **Cross-Lingual and Cross-Domain Crisis Message Classification** — Andrés Sánchez et al. (2023). ICWSM 2023. [Paper](https://ojs.aaai.org/index.php/ICWSM/article/view/22185/21964)
- **Humaid-Ner: A Disaster Tweet Dataset for Joint Named Entity Recognition and Event Classification via Uncertainty-Weighted Multitask Learning** — Aijaz Ali, Nazish Basir, et al. (2026). Asian Bulletin of Big Data Management. [Journal](https://abbdm.com/index.php/Journal/article/view/463) — closest prior multi-task disaster benchmark; English-only (verified), cite to support the "first for Taglish" claim.
- **FILIET: An Information Extraction System for Filipino Disaster-Related Tweets** — Ralph Vincent J. Regalado, et al. (2015). DLSU Research Congress. [PDF](http://www.dlsu.edu.ph/conferences/dlsu_research_congress/2015/proceedings/SEE/010-HCT_Regalado_RJ.pdf) — prior Filipino disaster intent work.

## Non-paper items referenced but not in index

- Republic Act No. 10173 (Data Privacy Act of 2012) — cited as `@ra10173`; index has the NPC 2021-02 circular only.
- NDRRMC situation reports and PAGASA landfall bulletins — flagged in a Chapter 3 `<!-- CITE -->` comment as unregistered.
- ISO/IEC 25010 — flagged in a Chapter 3 `<!-- CITE -->` comment as unregistered.

## Confirmed already in index (do not re-add)

Peffers et al. 2007 · Yu et al. 2020 (PCGrad) · Xin et al. 2022 · Kurin et al. 2022 · Alam et al. CrisisBench 2021 · Alam et al. HumAID 2021 · Imran et al. 2016 (Twitter as a Lifeline) · Imperial et al. 2019 · Herrera et al. TweetTaglish 2022 · Cruz & Cheng TLUnified 2022 · Miranda TLUnified-NER 2023 · Seeberger et al. 2022 and 2025 · Wang & Nulty 2021 (Transformer-based MTL disaster) · Montalan et al. Batayan 2025 · Demšar 2006 · Landis & Koch 1977 · Gambäck & Das 2016 (CMI)
