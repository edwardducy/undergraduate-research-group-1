# Mock Defense Script

## Title and Group Details

### Kyle

Good morning, panel. We are Isaiah Raphael Almuena, Prinz Edward Ducyogen, Carlo Miguel Geronimo, Kyle Uri Gabriel Lu, and Yngwie Sebastian Patricio.

Our study is titled "Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets." Let me briefly explain what that means.

During typhoons, many Filipinos post about their situation on X, often in Taglish. Take this post as an example. "Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!"

Our study will give a model two tasks on posts like this. Humanitarian classification assigns each post one category. This one falls under requests or urgent needs. Named entity recognition, or NER, picks out the key details inside the post. In this post, it marks "Marikina" as a location and "need ng tubig" as a request.

A multi-task model does both tasks at once, instead of using two separate models. This can save storage and processing time. However, the two tasks can pull the model in different directions while it learns, and this can make it worse at one of them. This problem is called negative transfer. A multi-task optimizer reduces this problem by controlling how much each task influences the model during training.

Finally, benchmarking means we will test 13 of these optimizers under the same conditions and keep the one that best reduces negative transfer without needing too much training time or memory.

Isaiah will now explain why this problem matters.

## Background and Problem Statement

### Isaiah

Thank you, Kyle. According to the WorldRiskReport 2025, the Philippines has the highest disaster risk in the world. Part of this risk comes from how often extreme weather hits the country. About eight or nine tropical cyclones cross the Philippines every year, and during each one, Filipinos post their needs and calls for help on social media. These posts arrive in such large numbers that responders cannot read them one by one. Responders therefore need a tool that can sort these posts for them.

Despite this, most of the Philippine papers we reviewed trained a model for only one of them. A recent system for Urdaneta City does both, but it does so by sending the posts to a large language model. Another option is a smaller model trained on labeled examples, which can sort posts about as well. However, training one for both tasks needs posts labeled for both tasks. An English dataset called HUMAID-NER, from Ali et al., has this kind of labeling, as the English post on the slide shows. None of the datasets we reviewed has it for Taglish, so the Taglish post on the slide has no labels yet.

So the first step is to build that dataset. But even with it, training one model for both tasks raises a computing problem, which Yngwie will now explain.

### Yngwie

Thank you, Isaiah. As Kyle mentioned, a model that does both tasks can become worse at one of them, which is negative transfer. There are reasons to expect it with our two tasks. When Zhu et al. trained classification and NER together on English text, the two tasks sent the model conflicting signals. Both tasks also scored slightly lower than models trained on just one of them. Ali et al. also argued that NER may overpower classification, since NER makes a decision for every word in a post, while classification makes only one for the whole post.

Multi-task optimizers are meant to reduce negative transfer, but studies disagree on whether they help. Some studies, including Kurin et al., found that the simplest approach works just as well. This approach is equal weighting, which treats both tasks as equally important. Gama and Grassi, however, found that most advanced optimizers did better, but only on harder problems.

The optimizers also differ in training cost. Many of them need more time and memory, and this cost is paid again each time the model needs retraining after a new disaster. So an advanced optimizer is worth using only if it reduces negative transfer enough to justify that cost. Since Gama and Grassi's results show that the answer depends on the problem, we cannot tell in advance which side holds for our two tasks. We have to test it.

This brings us to the main problem of our study. How can one model perform both tasks on Taglish typhoon posts while keeping negative transfer and training cost low? Carlo will now present the objectives that answer this question.

## Objectives and Scope

### Carlo

Thank you, Yngwie. To answer this question, the general objective of our study is to develop a multi-task model that performs humanitarian classification and NER on Taglish typhoon posts. As Isaiah mentioned, this model cannot be trained without the right data, so we will first build a Taglish dataset labeled for both tasks.

Our study has four specific objectives. First is the baseline analysis. We will train one single-task model for each task and measure how well it performs. Second is design and artifact creation. We will design and implement the multi-task model. Third is empirical benchmarking. We will train 13 versions of the multi-task model, one for each optimizer, and select the optimizer by comparing their negative transfer and training cost. The version with the selected optimizer will become our final multi-task model. Fourth is comparative validation. We will compare this final model with the single-task models in accuracy, speed, and memory.

Prinz will now present the scope of our study.

### Prinz

Thank you, Carlo. Our study covers Taglish posts about tropical cyclones that crossed the Philippines from 2020 to 2026, mainly from X. We will label about 3,900 of these posts for both tasks, using the categories of HumAID and the entity types of HUMAID-NER. All candidate models will share the same encoder, data, and training process, so any difference between them comes from the optimizer. We will also build a prototype web application that runs the final model, to show how disaster office staff could use it.

There are three things our study will not do. We will not compare our model with large language models, because they differ from our models in size, training, and cost, so the comparison would answer a different question. We will not deploy the model in real disasters, and we will not test the prototype with disaster office staff. Our research questions are about how the model is built, trained, and evaluated, not about how people use it.

With our scope set, let me now turn to the related studies.

## Literature and Baseline Algorithms

### Prinz

Earlier studies related to ours come from three areas, and we chose the closest study from each.

The first area is multi-task models for disaster posts. The closest is a 2026 study by Ali et al. They used one pretrained encoder for both tasks, with a separate part for humanitarian classification and another for NER, and trained it on English disaster posts. Our model will follow the same design.

The second area is Philippine disaster systems. The closest here is the Urdaneta City system that Caoile and Aviles built in 2025. It sends Filipino, Ilocano, and Taglish comments to a large language model, which sorts each comment by its purpose, such as a request for help, and picks out details like barangay names.

The third area is optimizer benchmarks. Kurin et al., in 2022, ran one of the studies in the debate Yngwie described. They trained the same model with equal weighting and with several advanced optimizers, then compared their scores and training time. This is the same kind of benchmark we will run.

Kyle will now show how these three studies compare with ours.

### Kyle

Thank you, Prinz. This table compares these three studies with ours. More importantly, each one points to work that our study will do.

Ali et al. worked on English posts and suggested extending their work to other languages. They also suggested combining PCGrad with uncertainty weighting. Our study will extend the same design to Taglish and include that combined optimizer among our 13.

Caoile and Aviles showed that a local disaster office needs both outputs from Taglish comments, and their system gets them from a large language model. Our study will test the other option Isaiah mentioned, a smaller model trained on labeled Taglish posts.

Kurin et al. found that equal weighting often did as well as advanced optimizers on images and robot tasks. However, they noted that it may do worse in other settings, and that finding those settings is future work. Our study will test a setting they did not cover, Taglish text with one task that labels the whole post and one that labels each word.

Together, these form the gap our study fills. Yngwie will now present what our study contributes by filling it.

## Expected Scientific Contribution

### Yngwie

Thank you, Kyle. Earlier, I said that an advanced optimizer is worth using only if it reduces negative transfer enough to justify its cost, and that we have to test this for our two tasks. Our benchmark will test it in three steps, and each step adds something new.

First, it will show whether training classification and NER together helps, hurts, or makes no difference to each task on Taglish posts. Earlier studies found small changes that went both ways, depending on the dataset, so this has to be measured on Taglish posts. No study we reviewed could do this, because no Taglish dataset labels the same posts for both tasks.

Second, it will show whether any advanced optimizer reduces negative transfer more than equal weighting, and either outcome adds evidence to the debate. If one does, our tasks become one of the settings Kurin et al. asked future work to find. If none does, our results will support their finding that equal weighting is often enough, now on Taglish text as well as on images and robot tasks.

Third, it will measure the training time and memory of all 13 optimizers on the same laptop, so researchers can weigh any gain against its cost.

Isaiah will now explain the practical benefits and what we will deliver.

### Isaiah

Thank you, Yngwie. In practice, our study will benefit three groups. The main beneficiaries are LGUs and their disaster offices. They will get one model that gives both the type of report and the key details of each post, and it could run locally instead of through a paid large language model service. Our comparison will also show them what they gain or give up by using one model instead of two. Local universities will benefit too, because our training time and memory results will show what they need to build and update such a model. In the end, communities hit by typhoons could also benefit, since responders who use the model may find urgent requests sooner.

By the end of our study, we will deliver the labeled Taglish dataset with its labeling guidelines, the final model with our benchmark results, and our code, including the prototype web application. As X's terms require, posts from X will be shared as post IDs with their labels.

Prinz will now present our software artifact.

## Software Artifact and System Design

### Prinz

Thank you, Isaiah. The main product of our study is the final multi-task model. This diagram shows how our system is laid out. It has two parts. The first is the benchmarking pipeline. It will prepare the labeled dataset, train every model, and select the optimizer. Because every candidate model will pass through this same pipeline, the benchmark stays fair. The second part is the prototype web application, which will run the final model.

This figure shows the model itself. The pretrained encoder reads each post once, and two separate parts, one for each task, use its output. One gives the category, and the other marks the entities. The optimizer is not shown, because it works only during training.

Carlo will now present our tools and the prototype's UI.

### Carlo

Thank you, Prinz. We will train the models in Python, using PyTorch and Hugging Face Transformers, the standard tools for this kind of model. Eleven of our 13 optimizers come from LibMTL, a published library, so most of them run on the same tested code instead of code we write ourselves. We will label the posts in doccano. The prototype will use React for the UI, FastAPI for the server, and PostgreSQL for the database. All training and measurement will run on one laptop, which serves as our testbed.

The prototype shows how disaster office staff could use the model. Posts arrive on their own through a simulated live stream that replays collected typhoon posts, and the model analyzes each one as it comes in. Each post appears with its category and its key details highlighted. If a label is wrong, the staff can correct it, and the correction is saved for training future models. They can also filter for urgent requests, mark the ones forwarded to a response team, and view or export a summary of the posts.

Kyle will now explain how we will test and evaluate the model.

## Testing Method and Metrics

### Kyle

Thank you, Carlo. Our dataset will come from the tropical cyclones that crossed the Philippines from 2020 to 2026. Following Seeberger et al., we will keep only the cyclones with at least 400 Taglish posts after cleaning, which should give us about 3,900 posts. Before any model is trained, we will split these labeled posts once into training, validation, and test posts. The models will learn only from the training posts. We will use the validation posts to make every choice, such as which optimizer to keep. The test posts will be used only once, at the very end, to compare the final model with the single-task models. This way, the final scores come from posts that played no part in training or in any of our choices, so they are not inflated.

The labels themselves must also be trustworthy, so we will measure how much our two annotators agree after each batch. Our targets are a Cohen's kappa of at least .61 for the categories and an F1-score of at least .74 for the entities. If a batch falls below either target, we will revise the labeling guidelines before the next one.

Each model will be measured on the accuracy, speed, and memory Carlo mentioned. For accuracy, we will use the F1-score. To see how it works, take one post. For NER, the model marks the phrases it thinks are key details. We then check each mark against the answer key from our annotators. Some marks are correct, some are wrong, and some key details are missed completely. We do this for every post. Now imagine an eager model that marks almost every word. It finds the key details, but most of its answers are wrong, so its F1-score is low. A careful model does the opposite. It marks only what it is sure of, so its answers are correct, but it misses many key details, and its F1-score is also low. The F1-score is high only when the model's answers are correct and it also finds the correct answers. The same idea applies to the categories in classification.

From these F1-scores, we will compute negative transfer as the multi-task score minus the single-task score. If the multi-task model scores lower than the single-task model, the result falls below zero, and this lost accuracy is negative transfer. For speed, we will first record the training time, from the first training post until the model is saved. We will also record the time the model takes to label each post, from the moment one post goes in until its labels come out, with each post handled one at a time. For memory, we will record the peak memory, which is the highest amount of memory used at any point during training. Finally, we will record the storage size of the saved model, which is the space a computer needs to keep the model and run it.

These measures will decide the optimizer. First, an analysis of variance will check whether the 13 optimizers differ in negative transfer at all. If they do, Dunnett's test will compare each one with equal weighting. An optimizer qualifies only if it reduces negative transfer significantly without lowering the score on either task. Among those that qualify, the one with the shortest training time becomes the final optimizer. If none qualifies, we will keep equal weighting, since it adds no extra cost.

Finally, we will compare the final model with the single-task models on the test posts. Each will be trained with the same five random seeds, so every run of one model has a matching run of the other. A paired t-test will then check whether their mean F1-scores differ for each task, and we will report each difference with a 95% confidence interval.

Since usability testing is outside our scope, we will test the prototype with functional test cases instead. Each use case will have test cases that state an input and the expected result, and it passes only if the actual result matches in every case. For example, when a staff member corrects a label, the new label must replace the old one and be saved.

Yngwie will now present our schedule.

## Development and Testing Schedule

### Yngwie

Thank you, Kyle. This Gantt chart shows our plan from November 2026 to April 2027. Right after this defense, we will revise Chapters 1 to 3 based on the panel's comments. From mid-November, we will collect the posts and prepare for labeling, and full labeling will run from December to late January, including the Christmas break. We will not wait for labeling to finish. While it continues, we will write the benchmark code and its tests, and build and test the prototype, so both are ready when the data is. Training needs the finished dataset, so it will start in late January and run until late February. During February, we will also write Chapters 4 and 5 as the results come in. Every task will end by the end of February, which leaves all of March as a buffer before the final defense in the first week of April 2027.

Isaiah will now explain why this plan is realistic.

### Isaiah

Thank you, Yngwie. Two things make this plan realistic. We have already run a training pilot, which confirmed that, after one change Kyle will explain, every model fits in our laptop's memory. The workload is also fixed in advance, at 92 tuning runs and 51 main runs, so the training time can be planned.

Carlo will now present our immediate next steps.

## Immediate Next Steps

### Carlo

Thank you, Isaiah. As soon as our proposal is approved, we will revise the chapters, as Yngwie mentioned. Alongside this, we will secure access to the full-archive search of the X API, since most of our posts will come from it.

Prinz will now present our targets for the coming weeks.

### Prinz

Thank you, Carlo. In the next two to four weeks, we will start with a pilot search. It will show how many posts remain after cleaning, which sets a limit for each cyclone. We will then collect and clean the posts and keep only the cyclones with at least 400 Taglish posts. At the same time, we will prepare for labeling by drafting our guidelines with Taglish examples, setting up doccano, and labeling a practice batch.

Full labeling will start only after two milestones. The first is about the data. The selected cyclones must give us about 3,900 posts. If X gives us too few Taglish posts, we will add public posts from Facebook. The second is about the labels. Our practice batch must reach the agreement targets Kyle described. If it falls short, we will revise the guidelines and label another practice batch. Once both are met, full labeling starts in December, as shown on our schedule.

Kyle will now present our technical risks and backup plans.

## Technical Risks and Backup Plans

### Kyle

Thank you, Prinz. We have planned for four risks.

The first is bugs in our code. Before any training, we will test each part, including every optimizer, on small examples with known answers, and then run the whole pipeline on a small sample.

The second is our hardware, and we have already faced it. In our training pilot, four optimizers needed more memory than our 8 GB graphics processor. We solved this by freezing the embedding layer in every model. Earlier work found that this layer changes little during fine-tuning, so freezing it should cost little accuracy.

The third is the prototype. If a use case fails, we will fix it and test it again.

The last is data and delays. If the X API fails or gives too few Taglish posts, we will turn to public Facebook posts. If we are still short, we will report the smaller dataset as a limitation. Any other delay falls into the month of buffer Yngwie described.

## References

### Kyle

To close, our study will build one model that both sorts Taglish typhoon posts by type of report and picks out their key details, and it will find the optimizer that best reduces negative transfer without needing too much training time or memory. The works we cited are listed here. Thank you for your time, and we welcome your questions.
