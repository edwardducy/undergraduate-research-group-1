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

The task of identifying words or phrases that belong to defined entity categories, such as locations, people, organizations, resources, or disaster-related terms. In the example tweet, Marikina is a location. Lolo provides context about who is at risk and is not treated as a named entity simply because it refers to a person.

### Task

One learning objective performed by the system. This study has three tasks: intent classification, urgency prediction, and named entity recognition.

### Outputs

The results produced by the system for the three tasks: the intent result, the urgency result, and the named entities identified through NER.

### Predictions

The results produced when the trained encoder and its task components analyze a tweet.

### Joint Triage

The application goal of the study. The system analyzes one tweet for intent, urgency, and named entities at the same time so that responders receive several useful signals for review and prioritization.

### Multi-Task Learning

A training approach in which one shared multilingual Transformer encoder learns several related tasks at the same time.

### Multi-Task Balancing

The process of controlling how much influence each task has while the shared encoder is being trained.

### Dynamic Multi-Task Balancing

A multi-task balancing approach that adjusts task influence during training instead of keeping the task contributions fixed.

### Benchmarking

Conducting a controlled comparison of different methods using the same data, model setting, training conditions, and evaluation measures.

### Resource-Constrained Deployment

A deployment setting where memory, computation, latency, or hardware availability must be considered carefully, such as deployment on locally available infrastructure.

### Negative Transfer

A situation in which training several tasks together makes one task perform worse than it would if trained alone. In this study, one task may interfere with another while the shared encoder learns intent, urgency, and NER from code-switched Taglish tweets, for example by predicting urgency well but failing to identify locations.

### Task Interference

The mechanism behind negative transfer: updates that improve one task make another task worse. In this study, NER is token-level while intent and urgency are tweet-level, so their training signals can conflict.

### Task Weighting (Loss Weighting)

The general approach of assigning an importance weight to each task's loss during training. Multi-task balancing methods, including the ones compared in this study, are task weighting methods.

### Uncertainty Weighting

A task weighting method proposed by Kendall and colleagues in 2018. It learns each task's weight from the model's uncertainty about that task. It is a standard baseline in multi-task balancing studies.

### GradNorm

A task balancing method proposed by Chen and colleagues in 2018. It adjusts task weights so that the gradients of all tasks have similar magnitudes during training.

### DWA (Dynamic Weight Averaging)

A task balancing method proposed by Liu and colleagues in 2019. It sets each task's weight from how quickly that task's loss changed in recent training steps.

### PCGrad

A task balancing method proposed by Yu and colleagues in 2020. It removes conflicting components between task gradients before updating the shared parameters.

### NTK-MTL

A task balancing method proposed by Navon and colleagues in 2022. It explains task imbalance using the neural tangent kernel and rebalances task losses accordingly.

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

Token-level tasks predict a label for each word in the tweet, such as NER. Sequence-level tasks predict one label for the whole tweet, such as intent classification and urgency prediction. The difference is one reason the tasks interfere.
