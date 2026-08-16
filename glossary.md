# Glossary

Use these terms consistently in the title defense script and proposal.

### Taglish

A form of communication that combines Filipino and English within the same message or tweet.

### Taglish Disaster Tweets

Disaster-related tweets written in Taglish. These tweets are the primary data units in the study.

### Code-Switching

Using Filipino and English within the same message or tweet. In this study, code-switched Taglish tweets may contain both languages in one message.

### Filipino and English

The two languages that the multilingual Transformer encoder is designed to process. Taglish refers to their use together.

### Transformer Encoder

A neural network structure that learns relationships among words in a tweet.

### Multilingual Transformer Encoder

A Transformer encoder designed to process both Filipino and English. In this study, it is the main text-processing component for intent classification, urgency prediction, and NER.

### Separate Multilingual Transformer Encoders

A baseline design that uses one multilingual Transformer encoder for each task. Each tweet passes through all three encoders, which repeats computation and increases memory use.

### Shared Multilingual Transformer Encoder

One multilingual Transformer encoder used by all three tasks. It processes each tweet once, and that single pass provides the basis for the intent, urgency, and NER predictions.

### Intent Classification

The task of identifying what a person needs or is trying to communicate, such as requesting evacuation assistance or reporting a disaster condition.

### Urgency Prediction

The task of estimating how urgently a message should be reviewed or acted upon.

### Named Entity Recognition (NER)

The task of identifying words or phrases that belong to defined entity categories, such as locations, people, organizations, resources, or disaster-related terms. In the example tweet, Marikina is a location and lolo is a person, so both are named entities. Baha is not an entity, because it describes the disaster-related condition and belongs to the urgency output.

### Task

One learning objective performed by the system. This study has three tasks: intent classification, urgency prediction, and named entity recognition.

### Outputs

The results produced by the system for the three tasks: the intent result, the urgency result, and the named entities identified through NER.

### Predictions

The results produced when the trained encoder and its task components analyze a tweet.

### Joint Triage

The application goal of the study. The system analyzes one tweet for intent, urgency, and named entities at the same time so that responders receive several useful signals for review and prioritization.

### Taglish Disaster Triage Application

The software artifact of the study. It processes incoming tweets with the trained system, so responders can run the triage locally. UR2 ends with the release of the annotated dataset, the trained system, and the code.

### Multi-Task Learning

A training approach in which one shared multilingual Transformer encoder learns several related tasks at the same time.

### Multi-Task Balancing

The process of controlling how much influence each task has while the shared encoder is being trained.

### Dynamic Multi-Task Balancing

A multi-task balancing approach that adjusts task influence during training instead of keeping the task contributions fixed.

### Balancing Methods

The family name used in the script for the methods that control task influence during training. The benchmark compares the established ones, such as Uncertainty Weighting and PCGrad, and the newer ones, such as CAGrad, IMTL, Nash-MTL, and FAMO. Each method replaces line 5 of the training loop with its own rule.

### Static Equal Weights

The simplest balancing method, used as a baseline. It gives the three losses the same influence at every training step. It does not adjust during training.

### Loss

The number that measures how wrong one task's predictions are on the current batch. Each task produces its own loss, and the balancing method decides how hard each loss pulls on the shared encoder.

### Task Heads

The three small components that sit on top of the shared encoder and produce the three outputs. The intent head reads what the person needs, the urgency head reads how urgent the situation is, and the NER head reads the named entities.

### Subwords and Subword Tokenizer

The tokenizer splits each tweet into small pieces called subwords. In Taglish, the split can break one word into pieces that come from both languages. For example, "ma-evacuate" splits at the hyphen, which sits between the Tagalog prefix and the English stem.

### Continued Pre-Training

The first goal of the training procedure. The encoder keeps training on unlabeled Taglish disaster tweets as part of the same procedure, so it adapts to the fragmented code-switched text. Gururangan and colleagues showed that this step helps in low-resource settings (2020.acl-main.740).

### Benchmarking

Conducting a controlled comparison of different methods using the same data, encoder setting, training conditions, and evaluation measures.

### Event Split

The dataset split for evaluation. Each method trains on past disaster events and tests on events it has never seen. TREC Incident Streams follows the same rule. Random splits inflate the scores, because event-specific words leak between the training and the test sets.

### Single-Task Baseline

The score of one task when it is trained alone. Negative transfer measures how far an output falls below its single-task baseline, and the weakest-output rule compares each output against it.

### Per-Output Score

One score for each output, reported separately. One averaged number can hide the imbalance among the three outputs, so the study reports all three.

### Language Switch Points

The words and phrases where Filipino and English meet, such as "ma-evacuate" from the example tweet. The evaluation scores the outputs at these points, because they are the hardest part of the Taglish text.

### Statistical Significance

A test of whether the difference between two methods is real, and not random variation from the seeds and splits. The repeated runs provide the data for this test.

### Latency and Throughput

The deployment cost metrics. Latency is the time one tweet takes on the laptops, and throughput is the number of tweets the system processes per second.

### Resource-Constrained Deployment

A deployment setting where memory, computation, latency, or hardware availability must be considered carefully, such as deployment on locally available infrastructure.

### Negative Transfer

A situation in which training several tasks together makes one task perform worse than it would if trained alone. In this study, one task may interfere with another while the shared encoder learns intent, urgency, and NER from code-switched Taglish tweets, for example by predicting urgency well but failing to identify locations.

### Task Interference

The mechanism behind negative transfer: updates that improve one task make another task worse. In this study, NER is token-level while intent and urgency are sequence-level, so their training signals can conflict.

### Task Weighting (Loss Weighting)

The general approach of assigning an importance weight to each task's loss during training. Multi-task balancing methods, including the ones compared in this study, are task weighting methods.

### Consistently Weakest Output

The output whose per-output score falls the most below its single-task baseline, and the loss repeats across the repeated runs and the event splits. The comparison identifies this output, and the targeted method is designed to prevent it from degrading.

### Targeted Adaptation

What the study designs. It is a targeted change to an existing balancing method, built from the measured weakness. It is not a new general-purpose algorithm.

### Uncertainty Weighting

A task weighting method proposed by Kendall and colleagues in 2018. It learns each task's weight from the network's own uncertainty about that task. It is a standard baseline in multi-task balancing studies.

### GradNorm

A task balancing method proposed by Chen and colleagues in 2018. It adjusts task weights so that the gradients of all tasks have similar magnitudes during training.

### DWA (Dynamic Weight Averaging)

A task balancing method proposed by Liu and colleagues in 2019. It sets each task's weight from how quickly that task's loss changed in recent training steps.

### PCGrad

A task balancing method proposed by Yu and colleagues in 2020. It removes conflicting components between task gradients before updating the shared parameters.

### NTK-MTL

A task balancing method proposed by Navon and colleagues in 2022. It explains task imbalance using the neural tangent kernel and rebalances task losses accordingly.

### CAGrad

A task balancing method proposed by Liu and colleagues in 2021. It finds one update direction that improves all the tasks together and avoids conflicts between their gradients.

### IMTL

A task balancing method proposed by Liu and colleagues in 2021. It finds the update direction that treats all task losses impartially, so no task's loss dominates.

### Nash-MTL

A task balancing method proposed by Navon and colleagues in 2022. It views task balancing as a bargaining game and uses the Nash solution as the update direction.

### FAMO

A task balancing method proposed by Liu and colleagues in 2023. It combines all task losses into one fast adaptive number, so balancing stays practical when the task count grows.

### Main Task

The task that a model is primarily designed to perform. In some multi-task setups, one main task receives support from auxiliary tasks. In this study, intent classification, urgency prediction, and NER are all main tasks.

### Auxiliary Task

An extra task trained alongside the main task to support it, without being a goal by itself. Example: the supporting task in the study by Mazumder and colleagues at EMNLP 2025.

### Humanitarian Information Needs

The standard label scheme for crisis social media data, used by CrisisLex and HumAID. The categories include caution and advice, casualties and damage, donations and volunteering, help requests, and other useful information.

### Informativeness

The label that separates informative crisis tweets from uninformative ones, used in crisis datasets such as CrisisBench. Informativeness filtering is the first step in crisis tweet processing.

### Situational Awareness

The goal of crisis informatics: building a shared understanding of what is happening during a disaster from social media reports, so that responders can act on verified information.

### Inter-Annotator Agreement

A measure of how consistently different annotators label the same data. Cohen's kappa is used for two annotators, and Fleiss' kappa for more than two. It is reported when a dataset is created, including the dataset in this study.

### Low-Resource Language

A language with little labeled data and few available NLP resources. Tagalog and Filipino are low-resource languages, which is one reason Taglish disaster data is scarce.

### Token-Level and Sequence-Level Tasks

Token-level tasks predict a label for each word in the tweet, such as NER. Sequence-level tasks predict one label for the whole tweet, such as intent classification and urgency prediction. The difference is one reason the tasks interfere. The script calls the sequence-level tasks sentence-level tasks.

### UR1 and UR2

The two research semesters. UR1 covers the first three chapters, the annotation, and the first training runs. UR2 covers the comparison, the targeted method, the remaining two chapters, the software artifact, and the public release.
