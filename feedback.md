# Advisor Feedback on the Title Proposal

Date: August 2026

## Research Title

**Old:** Benchmarking Dynamic Multi-Task Loss Balancing in a Lightweight Transformer Encoder for Joint Intent, Urgency, and Named Entity Recognition on Code-Switched Emergency Comments

**New:** Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets

Why we replaced "lightweight encoder" with "multilingual encoder":

- The new term reflects both the language and the resource requirements of the research.
- An English-only encoder does not understand Tagalog, so it would make errors from a language gap rather than from the balancing methods themselves.
- A multilingual encoder supports both English and Tagalog, but it still struggles with informal, code-switched Taglish disaster tweets.
- We removed the word "lightweight" because modern base encoders can already run on consumer laptops.

Why we replaced "Code-Switched Emergency Comments" with "Taglish Disaster Tweets":

- The new term clarifies the target language and the domain.
- "Code-switched" applies broadly to any language combination, while "Taglish" specifies the exact mix of Tagalog and English that the research evaluates.
- "Disaster tweets" clarifies the disaster-response context.
- It specifies social media posts, and it distinguishes our data from general comment sections on video platforms or blogs.

## The Advisor's Questions

The advisor asked four questions:

1. Is there an existing, open-source dataset of code-switched emergency comments annotated for Intent, Urgency, and NER, or will we have to manually label thousands of posts ourselves?
2. Which specific multi-task loss balancing techniques are we comparing?
3. Which base encoder are we using, and what are our baseline memory and latency constraints?
4. How do we evaluate success? How do we measure overall multi-task performance?

## The Advisor's Requirement

The final deliverable must be a fully functional emergency management and triage pipeline:

- The pipeline consumes emergency data and applies the trained model in a realistic deployment context.
- A single-page testing dashboard, such as Streamlit or Gradio, is not enough.
- A full social media platform is not required.

## Decisions

### Question 1: Existing Dataset

- No public dataset combines Taglish, the disaster domain, and joint intent, urgency, and NER annotations.
- We build our own dataset.
- A tool pre-annotates the tweets, and then we review the labels by hand.

### Question 2: Balancing Techniques

- We compare loss-balancing and gradient-balancing methods: Uncertainty Weighting (UW), PCGrad, GradNorm, CAGrad, IMTL, Nash-MTL, and FAMO.
- Baseline 1: static equal weights.
- Baseline 2: three separate encoders, one for each task.
- Most of these techniques were originally applied to computer-vision multi-task benchmarks, not to informal, code-switched text that carries more than one annotation.
- The list is not yet final, and we are still choosing the selection criteria.

### Question 3: Base Encoder and Constraints

- Multilingual ModernBERT is the leading candidate.
- An initial test on a Taglish sample decides the encoder.
- The same test sets the baseline memory and latency constraints.

### Question 4: Evaluation

- We score each of the three tasks separately.
- A single overall average would hide a weakness in one area.
- We compare the shared encoder directly against three separate standalone encoders.
- The comparison confirms that combining the tasks does not degrade performance.

## The System Deliverable

The system has three parts:

1. **Database and backend API:** a FastAPI service with SQLite or PostgreSQL.
   - One endpoint receives tweets from a simulated live stream of a held-out event.
   - The endpoint also receives tweets from manual entry.
   - The service runs the trained encoder and logs every tweet with its predictions and timestamps.
2. **Operational interface:** a web responder dashboard with an interactive map.
   - Each tweet becomes an alert card that shows intent, urgency, and entities.
   - Each NER location becomes a map pin.
3. **Workflow state:** every alert carries a status of New, Acknowledged, Responding, or Resolved.
   - The operator filters by intent, urgency, location, or status.
   - The operator updates statuses, and the system logs every change.

The plan follows the Humaid-Ner precedent of a real-time web dashboard.

### Defense Demo

- A live feed of held-out Taglish tweets flows through the pipeline.
- The dashboard and the map update in real time.
- The operator moves one alert from New to Resolved.
