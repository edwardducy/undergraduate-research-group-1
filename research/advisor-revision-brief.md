# Revision Brief for the Revised Alignment Sheet

This brief accompanies the revised alignment sheet. It lists what changed since the submitted sheet, why each change was made, and what we ask you to approve.

## What changed and why

1. **The research object is now stated explicitly.** The study benchmarks dynamic multi-task balancing across heterogeneous task granularities, one token-level task alongside two sequence-level tasks. The previous sheet described the setting but not the object of study.
2. **The connection to code-switching is now measured, not asserted.** The review asked how multi-task optimization relates to code-switching. The revised sheet adds one stratified analysis that answers this with data.
3. **The foundational article was replaced.** The venue of the previous foundational article has ceased to exist, so the citation no longer resolves for readers. Kendall et al. (2018) now fills the role because it is the origin of the balancing question and the reference method of the benchmark.
4. **The writing was revised to a formal academic register.** Claims are now stated so that each one is checkable.

## The one addition we ask you to review

Sub-RQ 3 asks whether the degree of Tagalog and English mixing changes the effect of balancing.

- **Provenance.** It responds directly to the review comment about the connection to code-switching.
- **Instrument.** Annotation records a word-level language tag per tweet, and each tweet receives a mixing ratio under the Code-Mixing Index convention. Both follow established practice. The word-level language identification method and its released code come from TweetTaglish (Herrera et al., 2022), which Chapter 1 already cites in its benchmark table. The mixing metric is the Code-Mixing Index (Gambäck and Das, 2016).
- **Cost.** Zero additional training runs. The stratified evaluation runs after training, on the saved predictions of all methods. It also adds only one automatic field to the annotation workflow, which the pre-annotation tool computes and reviewers spot-check.
- **Risk controls.** Three language profiles instead of a single ratio, subword fertility reported per profile so tokenization effects stay visible, an exploratory rather than confirmatory gradient analysis, and honest power caveats in the results.

## What we ask you to approve

1. The reinstatement of Sub-RQ 3 in the scoped form above.
2. The hardware wording for the efficiency study. Local consumer hardware, with a GPU workstation used for training and profiling, and a CPU demonstration for deployment. The current documents disagree on this point.
3. One primary encoder for the full benchmark grid, with the remaining encoder candidates used in a smaller ablation. Without this decision the benchmark grows to roughly 480 training runs across four encoders.
4. Adding two or three citations to the literature base. The Code-Mixing Index paper (Gambäck and Das, 2016) is confirmed. Two candidate papers on multilingual interference and code-switching benchmarks still need verification before we commit them.

## What we did not change

The research title, the task triplet, the Design Science Research phase structure in Chapter 1, and the four historical typhoon events all remain as approved.
