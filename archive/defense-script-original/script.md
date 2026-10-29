# Script

## Slide 1. Title and Group Details

Good [morning or afternoon] to the panel. The members of this group are Isaiah Raphael Almuena, Prinz Edward Ducyogen, Carlo Miguel Geronimo, Kyle Uri Gabriel Lu, and Yngwie Sebastian Patricio, with Engr. Ma Christina A. Florentino as adviser.

The title of the study is "Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets." The title contains several technical terms, so each one is explained in plain words, starting from the end of the title.

**[Taglish Typhoon Tweets]** The data of the study are social media posts. "Taglish" is the mix of Tagalog and English that many Filipinos use in everyday writing. "Typhoon" is used the way Filipinos commonly use it, for any tropical cyclone. "Tweets" refers to posts on X, the platform formerly called Twitter, which is the main source of the posts.

**[Humanitarian Classification]** The study gives a computer two tasks to do on each post. Take this made-up post as an example. "Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!" The first task, humanitarian classification, reads the whole post and gives it one category that tells responders what kind of report it is. This post belongs to the category "requests or urgent needs."

**[Named Entity Recognition]** The second task, named entity recognition, or NER, looks inside the same post and marks the words that name specific details. In the example, "Marikina" is marked as a location, and "need ng tubig" is marked as a request. The category alone does not say where help is needed or what is needed, so the two tasks complement each other.

**[Multi-Task]** A "multi-task" model is one model that does both tasks at the same time. Instead of two separate models, both tasks share one pretrained encoder, a language model that has already learned from large amounts of text. The shared model reads each post only once, so it is expected to need less storage and less processing time than two models. The risk is that sharing can lower the accuracy of one task, compared with a separate model trained on that task alone. This drop is called negative transfer.

**[Optimizers]** A multi-task optimizer is a training method that decides how strongly each task influences the model while it learns. The simplest one, equal weighting, gives both tasks the same influence throughout training. Others adjust this balance during training to reduce negative transfer, but they usually need more training time and memory. The study compares 13 of them.

**[Benchmarking]** "Benchmarking" means testing all 13 optimizers under the same conditions, so that any difference comes from the optimizer alone. They are compared on two things, how much negative transfer each one leaves and how much training costs in time and memory. The optimizer that wins this comparison is used in the final model of the study.

**[One-sentence summary]** Put simply, the study builds one model that sorts Taglish typhoon posts by type of report and picks out key details such as places and requests. It tests 13 ways of training this model and keeps the one that works best.

To explain why this problem matters, Isaiah presents the background of the study.

## Slide 2. Background and Problem Statement

**[Real world Context]** The Philippines has the highest disaster risk among 193 countries. During tropical cyclones, Filipinos often post calls for help on social media, many of them in Taglish, a mix of Tagalog and English. These posts arrive faster than responders can review them.

**[Bottlenecks]** Two NLP tasks can sort these posts. Humanitarian classification assigns each post a category, such as urgent needs or damaged infrastructure. Named entity recognition, or NER, finds details such as locations and requested supplies. A recent system for Urdaneta City does both tasks, but it sends comments to OpenAI for processing. This study instead builds a model that runs on one laptop.

**[Computational problem]** The multi-task model from the title answers these bottlenecks by sharing one pretrained encoder between both tasks, so it stores one encoder instead of two.

The difficulty is negative transfer. During training, each task sends the model a signal, called a task gradient, that shows how the model should change. On English text, Zhu et al. found that the two tasks sent the model conflicting signals. Ali et al. argued that NER may overpower classification, because NER makes a decision for every word, while classification makes one for the whole post. In a post of 30 words, that is 30 decisions against one.

**[Disagreement and cost]** Multi-task optimizers are meant to reduce this problem, but studies disagree. Kurin et al. and others found that equal weighting did as well or better, while Gama and Grassi found that most other optimizers did better on more complex problems. The optimizers also differ in training cost, the time and memory that training needs, and this cost returns every time the model is retrained after new disasters. No reviewed study tested these optimizers on Taglish posts, so which one works best here is still an open question.

**[Main research question]** This leads to the main research question. How can one model perform both humanitarian classification and NER on Taglish typhoon posts, and which multi-task optimizer should it use, given negative transfer and training cost? Carlo will present the four research questions next, together with the objectives.

## Slide 3. Objectives and Scope

**[General objectives and Specific Objectives]**

The general objective of this study is to develop a multi-task model that performs both humanitarian classification and named entity recognition on Taglish typhoon posts, supported by four specific objectives: measuring single-task baseline scores, implementing the shared model, selecting the best optimizer from 13 candidates through benchmarking, and validating the final model against single-task baselines on performance and training cost

Prinz will now present the scope and exclusions of the study.

**[Scope and Exclusions]**

## Slide 4. Literature and Baseline Algorithms (Kyle's part)

**[Comparison table]** This table puts the three studies next to this study. Each row is a study, and each column is something the problem needs.

Ali et al. labels the same English posts for humanitarian classification and named entity recognition. However, only two weighting methods were compared, and training time and memory were not measured.

The Urdaneta City system handles Taglish and gives both the type of report and its details. However, it relies on a large language model run by OpenAI, so every comment is sent outside the city's computers.

Kurin et al. compared several optimizers and reported training time. However, the tests used only images and robot tasks, with no text, and memory was not measured.

**[Research gap]** Each study covers only part of the problem. No reviewed study brings together Taglish text, humanitarian classification and named entity recognition on the same data, several multi-task optimizers, and measured training time and memory. This study fills that gap.

**[Hand-off]** Next, Yngwie presents what this study is expected to contribute.

## Slide 5. Software Artifact and System Design

**[Opening]** This study is expected to add three things to computer science.

**[Optimizer debate]** First, it gives new evidence on whether the specialized optimizers, which change the balance between the tasks during training, beat equal weighting. The evidence comes from a new language, a new domain, and a task pair that labels both the whole post and each word.

**[Inductive transfer]** Second, it tests inductive transfer, the idea that learning one task can help a related task. The result may be a gain, called positive transfer, a loss, or no clear effect.

**[Training cost]** Third, it is the first reviewed comparison to measure both training time and peak memory, the most graphics memory used at one time, for a set of optimizers on a pretrained language encoder.

**[Hand-off]** Isaiah now presents who benefits and what the study delivers.

**[Practical Benefits]** This study has practical benefits for three groups. Local government units can use the measured training time, memory, storage, and processing time as a reference when planning hardware for a model that runs on a laptop. Developers of similar disaster tools can use these measurements to estimate their own hardware needs, although the results apply to this encoder, dataset, and laptop. First responders could then use the model to sort incoming posts more quickly, once it has been tested with their staff.

## Slide 6. Software Artifact and System Design

**[Research method, system architecture, and model design]**

**[Tech Stack]**

For our tech stack, all model training runs locally on an Acer Predator Helios Neo 16 laptop equipped with an 8 GB RTX 4060 GPU. We use Python, PyTorch, Hugging Face Transformers, and LibMTL. Running completely offline ensures that sensitive disaster posts are never sent to outside third-party services.

**[UI/UX]**

For the user interface, we built a local prototype dashboard using React, FastAPI, and PostgreSQL. It allows disaster response staff to import posts, view automatic category tags and highlighted entity spans, manually correct wrong labels, and export summary reports.

Kyle will now present our testing methods and metrics

## Slide 7a. Data and Metrics

**[Data split]** The models are tested on the Taglish dataset built in this study. The dataset is split into three parts. The training set, with 72 percent of the posts, is used to train the models. The validation set, with 14 percent, is used to compare options and pick the best one, such as the best learning speed for each model and the best of the 13 optimizers. The test set, with the last 14 percent, is used only once, at the end, so that the final scores come from posts that played no part in any decision.

**[Accuracy]** Accuracy is measured with the F1-score, a number from 0 to 1 that is high only when a model finds most of the correct answers and makes few wrong ones. For humanitarian classification, the score counts rare categories as much as common ones. For named entity recognition, an entity counts as correct only when the whole phrase and its type are right. Negative transfer is the F1-score of the multi-task model minus that of the single-task model. A negative number means the shared model lost accuracy.

**[Label quality]** The labels are also checked. Two annotators label every post separately, and their agreement is measured after each batch. If it falls below a set target, the labeling guidelines are revised before the next batch.

**[Cost]** Cost is measured as training time, peak memory on the graphics card, the size of the saved model, and the time needed to process each post.

## Slide 7b. Choosing and Testing the Final Model

**[Runs]** Training involves some chance, so each candidate model is trained three times, and the final model and the single-task models five times. This shows whether a difference between models is real or only due to chance.

**[Selection]** The optimizer is chosen step by step, as this flowchart shows. First, an analysis of variance, or ANOVA, checks whether negative transfer differs at all among the 13 optimizers. If it does, Dunnett's test compares each optimizer with equal weighting, the simplest one. An optimizer qualifies only if it meets two conditions. It must have significantly less negative transfer than equal weighting. It must also score no lower than equal weighting on classification and on named entity recognition. Among the optimizers that qualify, the one with the shortest training time is chosen. If none qualifies, the final model keeps equal weighting.

**[Hypotheses]** Last, the final model is compared with the single-task models on the test posts. For classification and for named entity recognition separately, a paired t-test compares the five runs of each model in matched pairs. The null hypothesis is that sharing one model does not change the average score.

## Slide 7c. Software and Prototype Testing

**[Software]** Before training, each part of the code, including every optimizer, is tested on small examples whose correct answers are worked out by hand. Training starts only after every test passes. During training, any run that slows down from heat or runs short of graphics memory is repeated.

**[Prototype]** The prototype is stress-tested for one hour, and each of its use cases is checked against an expected result. As stated in the scope, it is not tested for usability with disaster office staff.

**[Hand-off]** Next, Yngwie presents the development and testing schedule.

## Slide 8. Immediate Next Steps

**[Gantt chart]** This Gantt chart, a timeline in which each bar shows when a task starts and ends, covers November 2026 to April 2027.

**[Bars, from top to bottom]** Chapters 1 to 3 are revised in the first two weeks of November. The longest task, collecting, cleaning, and labeling the posts, runs from mid-November to early February. While labeling continues, the benchmark code and its tests are written in December and January, and the prototype is built and tested from December to February. From late January to early March, all 143 runs are done, 92 for tuning and 51 main runs. Chapters 4 and 5 are written from February to March.

**[Proof of completion]** Every bar ends before the final defense in the first week of April 2027, and the last two weeks of March are kept free as a buffer for repeated runs or late labeling.

**[Hand-off]** Isaiah now explains why this workload is feasible.

**[Feasibility]** This study is feasible for three reasons. First, the training load is known in advance. The learning-rate and task-weight search needs 92 tuning runs, and the comparison needs 51 main runs, for 143 training runs on one laptop. Runs that hit a laptop slowdown are repeated, so the total can be higher. To keep the schedule safe, each run starts only after the runs it depends on are complete: tuning runs finish before the optimizer is selected, and the final model's extra seeds run after that selection.

## Slide 9. Immediate Next Steps

**[Right after approval]**

Right after proposal approval, the group will revise Chapters 1 to 3 based on the panel's feedback, secure official full-archive search access to the X API, and request the dataset post IDs from Ermino et al

Prinz will now discuss our targets for the next two to four weeks and the milestones before full data gathering

**[Next two to four weeks.]**

**[Milestones before full data gathering.]**

## Slide 10. Technical Risks and Backup Plans

**[Coding bugs]** The main coding risk is an optimizer that does not work as intended. Eleven of the 13 optimizers come ready-made from a library called LibMTL. The other two are existing methods that the library does not include, so the group codes them from their published descriptions. The tests described earlier check that every optimizer behaves as its original paper describes, and training starts only after every test passes.

**[Hardware]** The main hardware limit is the 8 gigabytes of memory on the laptop's graphics card. A pilot run has already confirmed that every model fits, once the first layer of the encoder is kept fixed in all models.

**[Prototype]** For the prototype, any function that fails its test is fixed and tested again before the final defense.

**[Delays]** The largest risk to the schedule is the data. If X does not provide enough Taglish posts, public Facebook posts are added. If the dataset is still short, the study uses the posts it has and reports the smaller dataset as a limitation. If labeling or training runs late, the buffer weeks in late March absorb the delay.

**[Hand-off]** The last slide lists the references.
