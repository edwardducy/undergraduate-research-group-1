# Manuscript Changes from Panel Feedback

# Chapter 1. Introduction

## 1.1 Introduction to the Study

- [x] Follow the TIOC format, presenting the trend, the issue, the objective, and
  the contribution in that order. (P19)
- [ ] State that the study trains one model to perform both tasks, rather than
  combining two existing models. The model does not correct or rewrite posts.
  It only sorts each post into a humanitarian category and marks key details,
  such as places and requested supplies. (C1)
  Partly done. Section 1.7 shows one model performing both tasks, but no
  section states that the model does not correct or rewrite posts.
- [x] Explain that responders need the posts classified because actionable
  posts are slow to find among the many posted during a disaster (Somido &
  Reyes, 2023). Classification shows which posts to act on first, and the key
  details show where help is needed and what supplies are asked for. (C9)

## 1.2 Background of the Study

- [x] Explain negative transfer in plain words, and give a short explanation
  of each technical term, such as "task gradient," at its first use. Negative
  transfer is when a model that learns both tasks does worse on one of them
  than a model that learns that task alone. (B1)

## 1.3 Statement of the Problem

No changes.

## 1.4 Objectives

No changes.

## 1.5 Hypotheses

- [x] Check that the hypotheses test whether the final model performs
  differently from the single-task models, not better. The feedback places this
  change in Section 3.6, but the hypotheses are written here. (P14)

## 1.6 Theoretical Framework

No changes.

## 1.7 Conceptual Framework

- [x] Update Figure 1.2 to show how the final model processes a post, replacing
  its outdated IPO model. (P22)

## 1.8 Scope, Limitations, and Delimitations

- [x] Remove Facebook as a fallback data source. (P4)
- [x] Change the period from 2020–2026 to January–November 2026. (P9)
- [x] Remove the claim that Taglish changes over time. The feedback does not
  say which section had it. (P10)
- [x] Change the dataset size from 3,900 to about 2,500 posts, based on
  annotation capacity. (P12, B8)
- [x] State that all posts, including the test posts, come from January to
  November 2026, taken from one day before each tropical cyclone enters the
  Philippine Area of Responsibility to three days after it leaves, according to
  PAGASA. The window includes the warnings posted before a tropical cyclone
  arrives and the requests and relief efforts that continue after it leaves.
  (P21)
  Section 1.8.1 states the window but not the reason for it.
- [x] State that training uses posts from past disasters and does not wait for
  a new disaster. The posts are collected before labeling begins in December.
  (C7)
- [x] State that posts from January to November 2026 are enough, because
  training depends on the number of labeled posts, not the number of years.
  (B5)
- [x] Narrow the scope to Taglish posts on X about the tropical cyclones of
  January to November 2026. Posts from other dates or about other disasters are
  out of scope. (B7)

### 1.8.1 Scope of the Study

- [x] State that the study does not create new optimizers. It tests 13
  optimizers from earlier studies and selects one for the final model. (P2)
  Stated in Section 1.8.3 instead.

### 1.8.2 Limitations of the Study

- [x] Add this section, grouping the limitations by the data, the experiments,
  and the results. (P20)
  - Data. X users may not represent everyone affected by tropical cyclones or
    the Taglish of every region, the 2,500 posts may be too few, rare
    categories may have too few test posts for precise F1-scores, the entity
    annotations from GLiNER are not checked by the annotators, and disputed
    categories are decided by one adjudicator.
  - Experiments. Five runs per model make the comparison of the optimizers
    descriptive and the paired t-tests unable to detect small differences,
    training time and peak memory may differ on other computers, and freezing
    the embedding layer may lower the F1-scores.
  - Results. The findings may not hold for tropical cyclones of other years or
    for other tasks and encoders.
- [x] Add a regional limitation. X does not reliably show the region of a
  user, so posts cannot be counted by region, and the dataset may not represent
  the Taglish of every region equally. (P13, B7)

### 1.8.3 Delimitations of the Study

- [x] State that the posts come only from X, because the developer policy of X
  allows posts collected through its API to be stored, provided that posts
  later deleted on X are also deleted, and because earlier disaster datasets
  were built on Twitter posts. (B9)
- [x] Find a source for the terms of Meta if the reason for excluding Facebook
  is to be stated. Without one, 1.8.3 gives only the reason for using X. (P4)
  No source was found, so 1.8.3 gives only the reason for using X.
- [x] State that the study compares existing optimizers rather than creating a
  new one. (P2)
- [ ] Replace the reason for omitting other optimizers. Section 1.8.3 says they
  are omitted "to keep the total number of training runs feasible," but the 13
  are all the optimizers that meet the two criteria in Section 3.2.
- [x] State that the model does not translate posts, since its pretrained
  transformer encoder was trained on both English and Filipino. (B10)
- [x] State that the study does not verify whether posts are true. (B11)
- [x] State that the study does not rate the urgency or priority of posts. (B14)
- [x] State that the study does not classify hazard types. (B15)
- [x] State that the study does not maintain the final model or the prototype
  after it ends. The final model, the code, and the post IDs are released on
  GitHub. (C12)

## 1.9 Significance of the Study

- [x] Make researchers the main beneficiaries, in four groups. Researchers in
  NLP and crisis informatics gain the annotated dataset. Researchers in
  multi-task learning gain evidence on negative transfer and training cost.
  Local disaster offices gain one model for both tasks, the F1-score comparison
  with the single-task models, and the prototype. Future researchers gain the
  released model, code, guidelines, and post IDs. Do not claim that the model
  runs on office computers or compare it with GPT, since the study tests
  neither. (C8)

## 1.10 Definitions of Terms

- [x] Revise the definition of negative transfer in plain words, as in 1.2.
  (B1)
- [x] Define "typhoon" and "Tweets" as used in the title. "Typhoon" refers to
  any tropical cyclone, following its common use in the Philippines, and
  "Tweets" refers to posts on X. Moved from 1.8.1.

# Chapter 2. Related Literature and Studies

Rewritten as two themes (2.1 and 2.2), a matrix (2.3), and a synthesis (2.4).
The section on multi-task optimizers and training cost was merged into 2.2.

## 2.1 Natural Language Processing of Disaster Posts

- [x] State that each Philippine study in the review trained a model for one
  task (Ermino et al., 2022). The one Philippine system that classifies urgency
  and extracts place names sends comments to GPT instead of training a model
  (Caoile & Aviles, 2025). (C5)

## 2.2 Multi-Task Learning

- [x] State that Zou et al. (2025) is one of the reviewed studies and not a
  model for this study. Only its finding on how training the two tasks together
  affected each task is used, so the flooding and areas of China are not
  compared. The study labels about 2,500 posts, not the 6,000 of Zou et al.
  (P3)
  Not stated directly. Sections 2.1, 2.2, and 2.3 use only the findings and
  dataset of Zou et al. and do not present it as a model for this study.
- [x] State that the studies behind the 13 optimizers were conducted outside the
  Philippines, but the optimizers still apply because they do not depend on the
  language of the posts. How well each works on Taglish posts is not yet known,
  so the study tests all 13. (C4)
- [x] State that the optimizers do not depend on the place or language of the
  data, because they act only on the numbers each task sends the model during
  training. Earlier studies applied the same optimizers to images and text (Xin
  et al., 2022). Moved from 1.8.3, since it is not an exclusion. (B4)

## 2.3 Algorithmic Literature Taxonomy Matrix

- [x] Rebuild the matrix with 18 rows grouped by kind of study, ending with
  this study, and a closing sentence on the gap it shows.

## 2.4 Synthesis of Related Literature and Studies

- [x] Rewrite the synthesis as three trends and the two gaps of Section 1.2,
  each tied to the research questions that address it.

# Chapter 3. Methodology

## 3.1 Research Methodology

No changes.

## 3.2 Subjects of the Study

- [x] State the two criteria for choosing the optimizers. The optimizer was
  compared in at least two of the comparison studies reviewed in Chapter 2,
  following Kurin et al. (2022), who chose well-established optimizers. It also
  uses the same model and training setup as the others, so that only the
  optimizer differs. (P1, B3)
- [x] State that no study, standard, or formula sets how many optimizers to
  test, and that the cited studies did not explain their own numbers. (P1)
  Dropped while writing. Research papers do not include such a statement, and
  the criteria already explain the number.
- [x] State that the 13 optimizers come from the group's own count of how many
  earlier studies compared each optimizer, not from the list of any one study.
  (C3)
- [x] State that the number 13 follows from the criteria, so leaving any out
  would need a supported reason. (B2)
- [x] Separate the choice of the 13 optimizers, made before training, from the
  choice of the final optimizer, made after training (see 3.6). (B16)
- [x] Remove Facebook as a fallback data source. (P4)
- [x] Change the period from 2020–2026 to January–November 2026. (P9)
- [x] Change the dataset size from 3,900 to about 2,500 posts, based on
  annotation capacity (Lakens, 2022). (P12, B8)
  Section 3.2 gives only the number. The reason is given in Section 3.3.

- [x] Apply the rule to the 13 optimizers. They are the optimizers evaluated in
  at least two of seven comparison studies (Vandenhende et al., 2022; Xin et
  al., 2022; Kurin et al., 2022; Lin et al., 2022; Lin et al., 2023; Royer et
  al., 2023; Gama & Grassi, 2026) that can be trained with the same model and
  setup, which excludes RotoGrad. DWA and GradVac replace FAMO and PCGrad with
  uncertainty weighting. Rename tuned static weighting to linear
  scalarization. Update Sections 3.5 and 3.8.1 to match.
- [x] Move the encoder, task heads, and losses here from 3.8.1, so that the
  models are described where they are introduced.

## 3.3 Sampling Technique

- [x] Remove Facebook as a fallback data source. (P4)
- [x] Change the period to January–November 2026. Earlier years are excluded,
  and December is left for labeling. (P9)
- [x] Remove the allocation of posts across years and tropical cyclones. (P11)
- [x] Change the dataset size to about 2,500 posts. Two annotators labeling 2
  to 3 hours a day on the 38 working days from December 2026 to January 2027
  can label about this many. (P12, B8)

## 3.4 Instruments to Be Used

- [x] Replace the absolute change in F1-score with the relative change of
  Vandenhende et al. (2022), $\Delta_{\text{MTL}}$, which Gama and Grassi
  (2026) also use.
- [x] Move the reason for using F1-scores instead of accuracy here from 3.7,
  next to the F1-score formula. (P16, P18)
- [x] Move the GPU monitoring to 3.5, with the other run conditions.

## 3.5 Procedure of Data Gathering

### Data collection

- [x] Remove Facebook as a fallback data source, including from Table 3.1. Meta
  does not allow automated collection without permission, and the Meta Content
  Library does not allow downloads for annotation. If X has too few posts, the
  smaller dataset is stated as a limitation. (P4)
  Removed from 3.5. Facebook is still mentioned in Section 3.8.3.
- [x] State that posts are collected only through the X API, with permission
  from X. Each post comes with its ID, which links to the original post. No
  screenshots or copying. (P5)
  The screenshot statement was dropped because methods sections state what was
  done. The post IDs are in the terms paragraph.
- [x] Describe the collection protocol in the text and in Table 3.1. The study
  searches the X API with search terms and the days around each tropical
  cyclone, and X returns the post ID with each post. (P6)
  The search terms are chosen from real posts in a pilot search on two of the
  study's tropical cyclones, and their recall is measured on the second one.
- [x] State that users keep ownership of their posts but accept the X terms,
  which let X license the posts. The study therefore does not ask each user for
  consent. It may store the posts but removes posts that users delete. (P7)
- [ ] State the cost. X charges $0.005 (about ₱0.32) per post, so 2,500 posts
  cost at least $12.50 (about ₱800). The final total is known only after
  cleaning. (P8)
  Stated in 3.5 as US$0.005 per returned post, with the cost depending on the
  size of the sampling frame, since X charges for every returned post and not
  only the 2,500 sampled. The P8 response still says at least $12.50.
- [x] Describe the near-duplicate removal stated in Section 3.3, following
  HumAID (Alam et al., 2021). Remove links and user mentions, represent each
  post by the counts of its words and word pairs, and remove one post from each
  pair with a cosine similarity above 0.75.

### Labeling

- [x] Add a table of annotator qualifications, the labeling timeline, and the
  rules for settling disagreements. (P17)
  The table has placeholders for the details of each person.
- [x] Describe how disagreements are settled. The two annotators discuss first,
  and a third annotator from outside the group, with disaster management or
  crisis informatics experience, decides if they still disagree. The guidelines
  are revised when agreement in a batch falls below its target. (P17, B13)
- [ ] Add the complete labeling guidelines, with definitions, inclusion rules,
  and Taglish examples for each category and entity type. (B17)
- [ ] Define *Requests or urgent needs* in the guidelines as posts asking for
  rescue, supplies, or immediate help. (B14)
- [ ] State in the guidelines that posts unrelated to disasters are kept and
  annotated *Not humanitarian*, as defined by HumAID (Alam et al., 2021).
  Moved from 1.8.3, since it is not an exclusion. (B15)

### Training

- [x] Train every model five times, with seeds 1 to 5, within the two to ten
  runs used in the reviewed comparisons of multi-task optimizers. The final
  model no longer gets extra seeds after the selection. (B6)
- [x] Move the optimizer implementation here from 3.8.1. Take the task
  gradients at the output of the encoder where LibMTL allows it.
- [x] Move the learning-curve runs here from 3.7, and the keywords and
  near-duplicate removal from here to 3.3.
- [ ] C3 is left to another group member. Its response in panel-feedback.md
  was removed. The pilot run in 3.5 only sets the batch size and the frozen
  layers. (C3)

## 3.6 Statistical Treatment

- [x] State that results are reported as they are. The hypotheses test whether
  the final model performs differently from the single-task models, not
  better. (P14)
- [x] State how the final optimizer is chosen. The final model uses the
  optimizer whose candidate model has the highest mean $\Delta_{\text{MTL}}$
  on the validation set. The ANOVA and Dunnett's test are dropped, so the
  comparison of the 13 optimizers is descriptive. (B16)

## 3.7 Data Analysis

- [x] State that there is no minimum acceptable F1-score, because no earlier
  study reports scores for both tasks on Taglish typhoon posts. The difference
  from the single-task models is reported instead. (P15)
- [x] Add the check of whether 2,500 posts are enough. Train the single-task
  models on 25%, 50%, 75%, and 100% of the training posts, and check whether
  the F1-scores stop improving before 100%. (P12, B8)
  The runs are in 3.5, and the rule for reading the curve is here.
- [x] Add an error analysis. The test posts the final model gets wrong are
  reviewed for what they have in common. Hard posts are not added on purpose,
  since the test posts are random. (C10)

## 3.8 Software System Description

### 3.8.1 System Structure

- [x] Move the model architecture to 3.2 and the optimizer implementation to
  3.5. Move the description of the prototype parts here from 3.8.2.
- [x] Update Figure 3.1. Remove Facebook and the ANOVA and Dunnett's test, and
  show that GLiNER annotates the entity spans.

### 3.8.2 Use Cases

No changes. Section 3.9 uses these for its test cases. (P18)

### 3.8.3 Materials and Equipment

- [x] Remove Facebook as a fallback data source. (P4)
- [x] Check that the development tools and their versions are listed: Python,
  PyTorch, Hugging Face Transformers, and LibMTL for the model, and React,
  FastAPI, and PostgreSQL for the prototype. The response says these are
  already there. (C6)

## 3.9 Testing and Evaluation

- [x] Give each use case at least one test case, which passes when the actual
  result matches the expected result. (P18)
  The test cases are in Appendix F, and the unit tests in Appendix E.
- [x] Define speed as the processing time, the mean time to label one post,
  compared with the combined time of the two single-task models. The optimizer
  does not affect speed. (C11)
  Measured in the performance testing, which replaces the stress testing.
- [ ] Measure the time the final model and the single-task models take to label
  one post, on the processor alone without the graphics card. (C8)
  Not added. The performance testing runs on the GPU.
- [x] Add a table of roles assigning each part of the study to a member,
  covering data collection, cleaning, annotation, benchmark code, prototyping,
  and testing. (B19)
