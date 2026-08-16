# Title Defense Script

## Part 1: Title and Problem Motivation

Estimated speaking time: 3 to 4 minutes

Our study is titled:

Benchmarking Dynamic Multi-Task Balancing in a Multilingual Encoder for Joint Triage of Taglish Disaster Tweets

We ask whether one multilingual Transformer encoder can process Taglish disaster tweets and produce three outputs:

- What the person needs
- How urgent the situation is
- Which important people, places, or resources are mentioned

These outputs matter because during disasters, people use social media to request assistance, report casualties, identify locations, and describe dangerous conditions.

Taglish is important because Filipino-English code-switching is common in informal online communication. A disaster-response system should be evaluated in the language people may use to report emergencies.

Consider this example:

"Pa-help naman po, stranded si lolo sa Marikina. Pataas ng pataas yung baha, di namin siya ma-evacuate."

A useful system should understand that this tweet is a request for evacuation or rescue assistance. It should recognize that the situation is urgent because an elderly person is stranded and the flood is continuing to rise. It should identify "Marikina" as a location. The word "lolo" also indicates who is at risk, while "baha" describes the disaster-related condition.

This is what we mean by joint triage. The system performs three tasks for each tweet: intent classification, urgency prediction, and named entity recognition (NER). Together, these outputs help responders understand what is needed, how urgent the situation is, and which details require attention.

Because responders cannot manually interpret large volumes of time-sensitive tweets, the system needs automation.

To automate this process, the study needs a multilingual Transformer encoder. It learns relationships among words in a tweet and is designed to process Filipino and English.

To run these three tasks, a straightforward design would be three separate multilingual Transformer encoders, one for each task. Each tweet would pass through all three, repeating computation and increasing memory use.

Our study instead uses one shared multilingual Transformer encoder. It processes each tweet once, and that single pass provides the basis for the intent, urgency, and NER predictions. This shared design could make local or resource-constrained deployment more practical.

However, sharing one encoder creates a multi-task learning challenge. Taglish makes the challenge harder, because Filipino and English mix within the same sentence, and even within the same word. The example tweet's "ma-evacuate" combines a Tagalog prefix with an English stem.

This mixing breaks the connection between a word's spelling and its meaning. Intent classification and urgency prediction need the meaning of the tweet across both languages, so they must understand that "ma-evacuate" means evacuation. In contrast, NER needs the words exactly as written, so it must track "ma-evacuate" with its exact spelling. During training, these different needs make the three tasks adjust the same shared encoder in different directions. When their adjustments conflict, one task can dominate and degrade the others. Which task loses is an open question. Suppose the NER output is the one that loses. In the example tweet, the shared encoder might recognize that the situation is urgent but fail to identify Marikina as the location. Responders would know that someone is stranded, but not where. In a disaster, this mistake can delay the rescue.

This is where dynamic multi-task balancing becomes relevant. It adjusts how much influence each task has during training. Which balancing method can train one shared multilingual Transformer encoder reliably for these three tasks in the Taglish disaster setting? Our study answers this question.

The next section explains what existing research has already accomplished and what remains unaddressed.

## Part 2: Background and Research Gap

Estimated speaking time: 3 to 4 minutes

Research already exists for each part of this study: disaster tweets, code-switched text, and Taglish. But no study combines these parts. We examine each part in turn.

We consider first the research on disaster tweets. Recent studies use a single model to predict crisis categories for one tweet at the same time. One example is CrisisSense-LLM, published in 2026 by Yin and colleagues (arXiv 2406.15477). It uses a single large language model to classify disaster tweets into multiple categories simultaneously. These studies demonstrate that joint triage is a feasible and practical direction. The question is whether this multi-task approach also works effectively for code-switched text.

The answer remains unknown for code-switched text. A 2026 survey of code-switched text research by Sheth and colleagues reviewed over three hundred studies. The survey reports that multi-task learning, which trains one model on many tasks at once, can introduce negative transfer. This risk makes careful task balancing necessary (DOI 10.18653/v1/2026.acl-long.386).

However, existing multi-task studies on code-switched text do not use balancing methods. For example, a recent study by Mazumder and colleagues at EMNLP 2025 adds an auxiliary task, an extra task that supports the main one, rather than a loss-balancing method, which is one family of balancing methods (arXiv 2412.12761). Beyond that, results in this area are conditional. A study on code-switched Algerian Arabic found that joint training helps some tasks in particular settings, and that the outcome depends on the choice of tasks, their order, and the data sizes (CALCS 2020). Consequently, in our search of the literature, no published study has systematically benchmarked dynamic multi-task balancing on code-switched data.

We found the same limitation in Taglish resources. Existing Taglish datasets do not support disaster triage. For example, TweetTaglish by Herrera and colleagues labels language proportions between Tagalog and English, but not disaster needs (2022.lrec-1.225). Similarly, the Batayan benchmark by Montalan and colleagues evaluates language models on Filipino tasks, but does not cover disaster response (arXiv 2502.14911). To date, no public dataset combines Taglish, the disaster domain, and joint intent, urgency, and named entity recognition annotations (no counterexample found in literature search, August 2026). Our study must therefore construct its own annotated dataset as part of its contribution.

We also checked Philippine disaster research. It does not provide this either. Existing local studies focus on single-task models. For instance, Imperial and colleagues analyze sentiment in typhoon tweets (arXiv 1908.01765). Other local works estimate urgency through sentiment, classify disaster needs in isolation, or detect disease outbreaks (Ermino et al. 2021; Barba et al. 2021; Livelo and Cheng 2018). None uses named entity recognition, multi-task learning, or joint triage labels.

The gaps leave responders with two options for multilingual Transformer encoders, and each has a cost. The first option is to use three separate encoders, which are slower and more resource-heavy than the other option. The second option is to use one shared encoder, and with it comes the question of which balancing method to use. No study we found has tested a dynamic multi-task balancing method on Taglish disaster tweets, so any choice would be a scientific guess or a copy from research on other languages. With static equal weights, training can lose one of the three outputs. Which one is an open question.

Because of these gaps, there is currently no established baseline or unified workflow for dynamic multi-task balancing in a shared multilingual Transformer encoder for Taglish disaster triage. Our study provides that benchmark. It introduces a new Taglish disaster tweet dataset and compares existing balancing methods under a controlled protocol.

The next section explains our methodology.

## Part 3: Proposed Methodology and Pipeline

Estimated speaking time: 5 to 7 minutes

The benchmark has three pieces: the system pipeline, the training procedure with its two goals, and the plan that turns the comparison into a method designed from the measurement. We explain each piece in turn.

The system has one shared multilingual Transformer encoder and three task heads. The encoder processes the tweet once, and that single pass provides the basis for all three outputs.

```
        Taglish disaster tweet
                 |
                 v
        subword tokenizer
                 |
                 v
   +-----------------------------+
   | shared multilingual         |
   | Transformer encoder         |
   | (one pass per tweet)        |
   +-----------------------------+
        |          |          |
        v          v          v
   intent      urgency      NER
    head        head        head
        |          |          |
        v          v          v
   what the   how urgent  people,
   person     the         places,
   needs      situation   resources
              is
```

Consider the path of the example tweet through this diagram. The tokenizer splits the tweet into small pieces called subwords. In Taglish, this step can split one word into pieces that come from both languages. For example, the word "ma-evacuate" does not survive as one piece, because the tokenizer splits at the hyphen, and the hyphen sits between the Tagalog prefix and the English stem. The two pieces still carry the meaning, but the encoder must work harder to connect them back into one word with its exact spelling. The encoder then processes all the pieces in one pass. From that single pass, the intent head reads the request for rescue, the urgency head reads how urgent the situation is, and the NER head reads Marikina and lolo as entities. The single pass is what makes the shared design cheaper than using three separate encoders.

The training procedure pursues two goals at the same time. The first goal is to adapt the shared encoder to the fragmented code-switched text. To pursue it, we continue pre-training the encoder on unlabeled Taglish disaster tweets. Existing evidence supports this step. Gururangan and colleagues showed that continued pre-training on in-domain text improves performance in low-resource settings (2020.acl-main.740), and the same pattern appears for code-switched text (2023.acl-long.66). The fragmentation problem is already demonstrated, so this goal needs no new evidence from our own results.

The second goal is to balance the three outputs so that no output dominates the others or gets lost. To pursue it, we train the three heads together with existing balancing methods under a controlled protocol. The three-task training runs on the Taglish disaster tweets, so the balancing operates on code-switched text. Both goals belong to the same procedure. The methods include the established ones and the newer ones, so the benchmark covers both generations of balancing strategies. The protocol also includes two baselines: static equal weights and three separate encoders, one for each task.

This is how balancing works. Each task produces its own loss during training. A balancing method decides how hard each loss pulls on the shared encoder. The simplest choice is static equal weights. It gives the three losses the same influence at every step. In contrast, the dynamic methods adjust the influence of each loss while training runs.

The training loop has six lines:

```
for each batch of labeled Taglish disaster tweets:
    1. Tokenize each tweet into subword pieces.
    2. Pass the pieces through the shared encoder once.
    3. Produce the three outputs with the three heads.
    4. Compute the intent, urgency, and NER losses.
    5. Combine the losses with the balancing method.
    6. Update the shared encoder and the three heads.
```

The one line that changes between balancing methods is line 5. Static equal weights combine the losses with fixed weights. Each balancing method replaces line 5 with its own rule. The controlled protocol keeps every other line identical, so the comparison isolates the balancing method.

The example tweet runs through the loop like this. Line 1 splits it into subword pieces, and "ma-evacuate" fragments. Line 2 passes the pieces through the shared encoder once. Line 3 produces the three outputs with the three heads. The intent head reads a request for rescue. The urgency head reads a stranded elderly person and a rising flood. The NER head reads Marikina and lolo.

Line 4 is where the conflict appears. The intent and urgency losses reward the encoder for understanding the meaning across the pieces. The NER loss rewards it for keeping the exact spelling of every entity. Under static equal weights, both pressures pull the same encoder. Related work in disaster tweet multi-task learning shows the pattern: a token-level task produces thousands of training signals for every single signal from a sentence-level task. If NER dominates, intent or urgency degrades. Which one loses is open, and the comparison measures it.

The comparison identifies the output that is consistently weakest across repeated runs and settings. We then design a balancing method that prevents that output from degrading, and we benchmark that method against the existing ones under the same protocol. Our method is a targeted adaptation of an existing balancing method, not a new general-purpose algorithm. It does not start from a formula. It starts from the measured weakness.

The next section explains how we will evaluate the methods, on which data, and with which metrics.

## Part 4: Experimental Evaluation and Metrics Plan

Estimated speaking time: 3 to 4 minutes

The evaluation has three parts: the data, the test conditions, and the metrics. We describe each part in turn.

The evaluation runs on the new Taglish disaster tweet dataset, which contains a number of annotated tweets across a number of disaster events. We split the dataset by disaster event, so each method trains on past events and tests on events it has never seen. This split is the realistic one, because a deployment must work on the next disaster, not on the one it trained for. Random splits inflate the scores, because event-specific words leak between the training and the test sets (ICDMW 2025). The standard evaluation track for crisis tweets, TREC Incident Streams, follows the same rule.

Every method runs under the same conditions: the same splits, the same machines, the same amount of training, and repeated runs across different seeds. The repeated runs let us test whether the differences between methods are statistically significant, and not just random variation. The comparison includes the two baselines: static equal weights, and three separate encoders, one for each task. Between the balancing methods, the only difference is line 5 of the training loop.

The metrics come next. The main numbers are one score per output. For intent and urgency, the system gets a separate score for each possible answer, and then we average the scores. Suppose a report of damage appears often in the data, and a request for rescue appears rarely. If we counted all correct predictions at once, the common answer would dominate the score, and the system could ignore the rare answer and still score high. Separate scores prevent that, so a rare answer counts as much as a common one. For named entity recognition, the score counts how many of the people, places, and resources the system identifies correctly, with the exact start and end of each mention. We report the three scores separately, because one averaged number can hide the imbalance among the three outputs. Established multi-task benchmarks follow this rule. Taskonomy, the CVPR 2018 Best Paper, reports every task separately, and PCGrad does the same (NeurIPS 2020).

We also report the average of the three scores, so our results can compare with studies that report one number, and the negative transfer of each output. Negative transfer measures how far an output falls below its single-task baseline. We also check every method against naive baselines, such as always predicting the most common class.

We also score the outputs at the language switch points, on the words and phrases where Filipino and English meet, such as "ma-evacuate" from the example tweet. This check shows whether each method handles the hardest part of the Taglish text.

The cost metrics follow. Training runs on one machine with a graphics card, where we report the extra memory and the extra computation that each balancing method adds per step. Deployment runs on our two consumer-grade laptops, each with an Intel i5-13500HX processor and 32 gigabytes of DDR5 memory, using Python. On these laptops, we report the latency of one tweet and the throughput in tweets per second. These numbers matter because the shared encoder must stay practical for local hardware.

During training, we also record how hard each task pulls on the shared encoder, and in which direction. These numbers show how each method handles the conflict, and they follow the published diagnostics from GradNorm (ICML 2018) and PCGrad (NeurIPS 2020).

From these per-output scores, the comparison identifies the consistently weakest output with a simple rule. The weakest output is the one whose per-output score falls the most below its single-task baseline, and the loss repeats across the repeated runs and the event splits. This rule turns the comparison into a decision.

Our method then runs under the same protocol, and we report the same numbers for it. The method succeeds when the weakest output improves over the best score from the existing methods, and we measure whether preventing the degradation costs anything on the other two outputs.

The next section explains the feasibility, the timeline, the risks, and the expected contributions.

## Part 5: Feasibility, Timeline and Expected Contributions

Estimated speaking time: 2 to 3 minutes

The final part covers four things: the feasibility of the study, the timeline, the risks, and the expected contributions. We cover each in turn.

The resources already exist. Training runs on one machine with a graphics card, and the deployment measurements run on our two laptops. The balancing methods are available in published libraries. The main costs are time spent annotating the tweets and reading the source papers. The annotation protocol is standard. Multiple annotators label the same tweets, and we measure how well their labels agree. We use the labels only after the agreement reaches the accepted level, and we report the agreement number together with the dataset. We are still reading the source papers and writing notes on each one, so every claim in this defense stands on a source we can show.

The timeline spans the two research semesters. In UR1, we do three things. We complete the first three chapters: the problem and its background, including the scope and delimitations, the review of related literature, and the methodology. We collect and annotate the Taglish disaster tweets, and we measure how well the annotators agree. We set up the training procedure, and we run the two baselines and the first balancing methods. In UR2, we finish the comparison and identify the consistently weakest output. We design the targeted method that prevents it from degrading, and we validate that method under the same protocol. We then write the remaining two chapters. Chapter 4 reports the results and discussion. Chapter 5 closes with the summary, conclusions, and recommendations. We also build the software artifact, a Taglish disaster triage application that processes incoming tweets with the trained system. We end UR2 by releasing the annotated dataset, the trained system, and the code to the public.

We know the risks in advance. The first risk is the dataset. The annotation effort depends on the tweet count, which we have not fixed yet, and the labels must stay reliable across annotators. We control this risk with the standard annotation protocol and agreement checks before we use any label. The second risk is that the existing methods may already balance the three outputs well. Even then, one output is still the weakest, so the designed method still has a target, and the dataset and the benchmark stand on their own. The third risk is compute. The comparison multiplies methods, seeds, and event splits, so we fix the training budget, reuse the published implementations, and keep the text tweet-length, which one graphics card can handle.

The study contributes four things. First, the new Taglish disaster tweet dataset with joint intent, urgency, and NER annotations. Second, the benchmark that compares existing balancing methods on the Taglish disaster tweet dataset under a controlled protocol. Third, the balancing method that prevents the consistently weakest output from degrading in Taglish disaster triage, designed from the measured weakness and validated against the existing methods. Fourth, the software artifact, a Taglish disaster triage application that lets responders run the trained system on incoming tweets.

The study started with one question. Which balancing method can train one shared multilingual Transformer encoder reliably for the three tasks in the Taglish disaster setting? This plan answers that question in two steps. First, the benchmark measures the three outputs under a controlled protocol and identifies the consistently weakest output. Second, we design a targeted balancing method that prevents that output from degrading, and we measure whether preventing the degradation costs anything on the other two outputs. We do not answer with a guess. We answer with a measurement.
