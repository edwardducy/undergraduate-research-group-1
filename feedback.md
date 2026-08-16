# Advisor Feedback on the Title Proposal

Date: August 2026

The advisor asked for a revised proposal based on the first of the three submitted titles. The proposal was too vague. The advisor asked four questions and set one requirement for the final system deliverable. This file records the questions, the grounded answers, and the open decisions.

## The Advisor's Questions

1. Is there an existing, open-source dataset of code-switched emergency comments annotated for Intent, Urgency, and NER, or will we have to manually label thousands of posts ourselves?
2. Which specific multi-task loss balancing techniques are we comparing?
3. Which base encoder are we using, and what are our baseline memory and latency constraints?
4. How do we evaluate success? How do we measure overall multi-task performance?

## Question 1: Is There an Existing Dataset?

No public dataset combines Taglish, the disaster domain, and joint intent, urgency, and NER annotations, so we must build our own dataset.

We will manually label our own dataset, but not from zero. We will use a tool to pre-annotate the tweets, and then we will review the labels by hand.

We have not yet decided on the size of the dataset.

## Question 2: Which Balancing Techniques?

We will compare loss-balancing and gradient-balancing methods such as Uncertainty Weighting (UW), PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, and FAMO. The two baselines are static equal weights and three separate encoders, one for each task. This list is not yet final, because we are still deciding on the selection criteria. Most of these techniques were originally applied to computer-vision multi-task benchmarks, and not to informal, code-switched text that carries more than one annotation.

## Question 3: Which Base Encoder and Which Constraints?

We have not yet made the final decision on the base encoder. We decided on Multilingual ModernBERT as the leading candidate, but we are still deciding. We will first run an initial test on a Taglish sample, and then we will decide on the encoder and set the baseline memory and latency constraints.

## Question 4: How Is Success Evaluated?

- Per output: Intent Macro-F1, Urgency Macro-F1, and token-level NER span F1 with exact boundaries. We report the three scores separately, because one averaged number hides which output lost.
- Overall multi-task performance: the average of the three scores, plus the standard multi-task aggregate from the Taskonomy and Standley line of work. That aggregate is the mean relative improvement of each shared-encoder output over its single-task baseline. It answers whether sharing helped overall.
- Negative transfer: we measure each output against its single-task baseline, so a drop is visible per output.
- Rigor: event-based splits (the TREC Incident Streams precedent), repeated seeds, significance tests, naive baselines, and scoring at language switch points.
- Success criteria: first, the shared encoder reaches parity with or beats the three separate encoders on every output. Second, our balancing method raises the consistently weakest output above the best existing method without significant cost to the other two. Third, the latency, memory, and throughput constraints hold on the laptops.

## The System Deliverable Requirement

The advisor's requirement, in plain words:

- A single-page testing dashboard such as Streamlit or Gradio is not enough for the final system deliverable.
- We are not expected to build a full social media platform.
- We must build a fully functional emergency management and triage pipeline system that consumes emergency data and applies the trained model in a realistic deployment context.

The system must have three parts:

1. Database and backend API: real-time processing and logging of incoming code-switched comments.
2. Operational interface: a functional responder dashboard, an interactive map visualization based on parsed NER locations, or a mobile or bot integration.
3. Workflow state: the ability for an operator to view, filter, and update the status of incoming alerts based on the model's intent and urgency classifications.

Our plan follows the Humaid-Ner precedent of a real-time web dashboard:

- Backend and database: a FastAPI service with SQLite or PostgreSQL. One endpoint receives tweets from a simulated live stream of a held-out event, plus manual entry. The service runs the trained encoder and logs every tweet with its predictions and timestamps.
- Operational interface: a web responder dashboard with an interactive map. Each tweet becomes an alert card that shows intent, urgency, and entities. Each NER location becomes a map pin.
- Workflow state: every alert carries a status of New, Acknowledged, Responding, or Resolved. The operator filters by intent, urgency, location, or status, updates statuses, and the system logs every change.

Defense demo: a live feed of held-out Taglish tweets flows through the pipeline. The dashboard and the map update in real time. The operator moves one alert through the whole workflow, from New to Resolved. This shows the trained model inside a realistic deployment context with no social platform.

## Open Decisions

Four numbers wait for our confirmation before the answers enter the proposal:

1. Dataset scale: 6,000 tweets across 6 Philippine disaster events.
2. Primary encoder: Multilingual ModernBERT (leading candidate, pending confirmation from the initial Taglish sample test).
3. Constraints: 1.5 GB of memory, one second of latency per tweet, and 2 tweets per second of throughput.
4. The method list: UW, PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, and FAMO, plus the two baselines.

We must also confirm that the first of the three submitted titles is the locked title, Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets.

## Integration Plan

Where each answer lands in the proposal:

- Part 2: add Humaid-Ner as the closest dataset precedent and as the annotation-pipeline precedent. The script already cites TweetTaglish, Batayan, and CALCS 2020.
- Part 3: name the seven methods and the two baselines. Add the encoder-selection measurement.
- Part 4: add the overall relative-improvement aggregate and the concrete constraint numbers.
- Part 5: replace the artifact sentence with the three-part pipeline system and the defense demo.
- notes.md: record the scale, encoder, constraint, and system decisions.
