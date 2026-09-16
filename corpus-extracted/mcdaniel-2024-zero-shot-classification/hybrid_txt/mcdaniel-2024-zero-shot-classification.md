# Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models\*

Emma L. McDaniel<sup>∗†</sup>, Samuel Scheele<sup>∗‡</sup>, Jeffrey Liu<sup>§</sup>

∗ indicates equal contribution

<sup>†</sup>Computer Science Department

Georgia State University

Atlanta, GA, USA

<sup>†‡§</sup>Humanitarian Assistance and Disaster Relief Systems

MIT Lincoln Laboratory

Lexington, MA, USA

<sup>†</sup>emcdaniel10@gsu.edu, <sup>‡</sup>samuel.scheele@ll.mit.edu, <sup>§</sup>jeffrey.liu@ll.mit.edu

Abstract—Social media posts are frequently identified as a valuable source of open-source intelligence for disaster response, and pre-LLM NLP techniques have been evaluated on datasets of crisis tweets. We assess three commercial large language models (OpenAI GPT-4o, Gemini 1.5-flash-001 and Anthropic Claude-3-5 Sonnet) capabilities in zero-shot classification of short social media posts. In one prompt, the models are asked to perform two classification tasks: 1) identify if the post is informative in a humanitarian context; and 2) rank and provide probabilities for the post in relation to 16 possible humanitarian classes. The posts being classified are from the consolidated crisis tweet dataset, CrisisBench. Results are evaluated using macro, weighted, and binary F1-scores. The informative classification task, generally performed better without extra information, while for the humanitarian label classification providing the event that occurred during which the tweet was mined, resulted in better performance. Further, we found that the models have significantly varying performance by dataset, which raises questions about dataset quality.

Index Terms—Large Language Models, Zero-Shot Classification, Crisis Classification, Social Media

## I. INTRODUCTION

In crisis scenarios, such as natural hazard-induced disasters or humanitarian emergencies, timely and accurate information is crucial to decision makers. Social media posts can provide valuable information in real time; however, the sheer speed and quantity of data coming from social media can be overwhelming for human analysts to process. As such, Natural Language Processing (NLP) techniques have been used to automate the processing of social media data in order to classify and extract the most relevant information. CrisisBench [1] provides a benchmark dataset to evaluate the performance

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force. <sup>©</sup> 2024 Massachusetts Institute of Technology. Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

of NLP solutions for classifying crisis-related social media posts.

Recently, Large Language Models (LLMs) and Large Multimodal Models (LMMs) have shown impressive performance on a wide range of NLP tasks without needing task-specific training or fine-tuning. Large Language Models, such as GPT [2], [3], are trained on massive text datasets to predict the next word in a sequence, and can be used to generate answers to questions. They can thus be used as zero-shot text classifiers by inputting the relevant text, followed by a question asking which of a set of given labels or classes apply to the text. Large Multimodal Models can be trained and utilized similarly, except they are configured to accept other data modalities, such as images, in addition to text.

Due to the recent popularity of LLMs, we expect that humanitarian practitioners will try to use them to help automate the process of extracting relevant information from social media during crises. As a first step towards characterizing the performance of LLM/LMMs on such tasks and identifying which ones provide the best performance, we evaluate various open-access and commercial LLMs and LMMs on zero-shot classification of social media posts using the CrisisBench dataset. In addition, we compare the performance of the zeroshot classifiers to the existing benchmarks from purpose-built classifiers for crisis-related social media text classification.

## A. Related Work

CrisisBench [1] combines a number of crisis datasets [4]–[9] through cleaning and standardizing labels in order to create a benchmark for measuring performance of NLP classification of crisis-related social media posts. CrisisBench defines two tasks. The “informativeness” task, a binary classification task that seeks to identify whether a provided tweet contains valuable information regarding a disaster or crisis event. The “humanitarian information type,” a multi-class classification task that seeks to categorize a tweet into one of 16 classes (e.g donation and volunteering, displaced and evacuations).

Previous work in classifying crisis-related social media posts has involved conventional machine learning methodologies and non-transformer-based neural networks [10]–[15], fine-tuned transformer-based models for multimodal classification using images embedded within social media messages [16], and fine-tuned transformer-based models focusing solely on text [11], [17]–[20].

In contrast to developing task-specific models, a growing trend involves leveraging instruction-tuned LLMs and LMMs for zero-shot classification [21]–[28]. While zero-shot classification circumvents the need for extensive labeled training data for fine-tuning, it is essential to understand various models’ limitations within specific domains. Our work attempts to address this need in the realm of humanitarian assistance by providing performance statistics for a range of commercial models.

## II. METHODOLOGY

## A. CrisisBench Task Descriptions

In this paper, we focus on the “informativeness” task from the CrisisBench consolidated dataset, and provide incidental analysis of the “humanitarian information type” task for those data points that also had “humanitarian information” labels. Roughly 5,000 of the examples in the informativeness test set are also in the humanitarian information type test set - these examples are the only ones considered for the analysis of the humanitarian information type task.

The motivation for this is that the classes in the multi-class task are often amalgamations of classes from the constituent datasets, and most constituent datasets used only a few of the classes. In the Discussion, we will provide preliminary analysis of the multi-class task where semantic differences in definitions across constituent datasets substantially impacted performance. The “informativeness” task does not suffer from the same ambiguity as the “humanitarian information type” task and results are therefore easier to obtain and interpret.

A subset of tweets in the CrisisBench dataset are from CrisisMMD dataset, which contains only tweets which include images [5]. The CrisisBench authors also include an event type annotation that indicates the type of crisis event that was contemporaneous with the timestamp of the tweet. We evaluate each task both with/without event awareness, and with/without images for four configurations per task.

## B. Models Evaluated

We evaluate three commercial models: OpenAI’s GPT-4o [29], Google’s Gemini 1.5 Flash [30], and Anthropic’s Claude Sonnet 3.5 [31], and accessed them through their respective APIs. The models were chosen based on several considerations, including prominence, performance on other benchmarks, and availability.

## C. Prompt Structure

We used the same base prompt for all models in the CrisisBench dataset, in which we requested that the model return a JSON string with a specified schema. We used Pydantic to validate the JSON. In the case where the model did not return valid JSON, we simply re-submitted the prompt and retried up to a set patience of three attempts. Responses that were not valid were omitted from analysis.

We asked the models to complete both the “informativeness” task as well as the multi-class “humanitarian information type” classification task in the same prompt. For the informativeness task the model provides a true or false The base prompt is provided below,

```python
Provide classifications of the following tweet
    based on its relevance to a humanitarian
    event and a classification of its content.
    "\
        "{img_str}"\
        "{event_str}"\
        "The tweet follows:\n{tweet_str} \n"\
        "{field_descriptions}"
```

where the fields {img\_str}, {event\_str}, {tweet\_str}, and {field\_descriptions} are placeholders.

The placeholder {img\_str} was filled in with the text "Use the images, if present, to help you make a your determinations related to the informativeness and category of the tweet." if an image was associated with the tweet; otherwise, it was left blank. Images were resized to fit within 768 × 768 pixels while maintaining aspect ratio, encoded in base64, and appended to the prompt in accordance to the respective LMM’s specifications.

The placeholder {event\_str} was filled in with "While it may still be irrelevant or uninformative, this tweet was created around the time of a disaster with description: {event\_type}.", where {event\_type} corresponds to the event that was occurring during the time of the tweet if we were evaluating the tweet in the “event-aware” configuration; otherwise, {event\_str} was left blank. The {tweet\_str} placeholder contained the actual text of the tweet.

The {field\_descriptions} placeholder contained descriptions of the classes as well as the desired JSON format for the output. We requested two fields: is\_informative and humanitarian\_label. For the is\_informative field, the prompt was "Does the tweet contain information pertinent to a humanitarian event or natural disaster? Respond with a boolean true/false". For the humanitarian\_label field, the prompt is provided below:

```txt
For a given tweet, determine which of the
    humanitarian labels are most relevant:
The humanitarian labels and their descriptions
    are provided below:
    "not_humanitarian" - The tweet is not
        humanitarian in nature and does not fit
        into any other class.
    "donation_and_volunteering" - The tweet
        relates to directing, accepting, or
        distributing donations or volunteer
        effort.
```

"requests\_or\_needs" - The tweet describes a request or need of an individual or community.  
"sympathy\_and\_support" - The tweet expresses sympathy or support for disaster victims.  
"infrastructure\_and\_utilities\_damage" - The tweet relates to the construction or destruction of infrastructure, utilities, or structures.  
"affected\_individual" - The tweet contains information on a particular individual affected by a disaster.  
"caution\_and\_advice" - The tweet contains caution or advice for victims, responders, or others.  
"injured\_or\_dead\_people" - The tweet notes the presence of injured or dead people.  
"response\_efforts" - The tweet pertains to the response effort.  
"missing\_and\_found\_people" - The tweet discusses missing persons, including in the context of finding them.  
"displaced\_and\_evacuations" - The tweet relates to displaced people or an evacuation process.  
"personal\_updates" - The tweet relates to a personal opinion or a status update about the tweet author or their close relations.  
"physical\_landslide" - The tweet is related to a physical landslide.  
"disease\_related" - The tweet reports on disease transmissions, symptoms, treatment, prevention, or affected people.  
"terrorism\_related" - The tweet reports possible terrorism or terrorist acts.  
"other\_relevant\_information" - The tweet is humanitarian in nature, but does not fit in any other class.  
The output should be formatted as a dictionary whose keys are the humanitarian labels, and the values are two-element arrays whose entries correspond to the following:  
The first element of the array should be a ranking: an integer from 1 to 16 representing the relative relevance of the humanitarian label compared to the others . The most relevant label should be ranked 1, and least relevant should be ranked 16.  
The second element should be a likelihood score: a floating point number between 0 and 1 representing the likelihood that the label applies to the tweet.  
The dictionary should have an entry for every humanitarian label, even if it is not relevant. The rankings for each label should be unique---that is, no two labels should have the same ranking, even if they are both not relevant: you must rank one higher than the other---and the likelihoods should sum to 1

At the end of the prompt, we provided an example of the JSON format to address errors encountered during models’ JSON construction and subsequent validation using Langchain/Pydantic:

For example, a correctly formatted answer would be: {"is\_informative": false, " humanitarian\_label": {"not\_humanitarian": [1, 0.95], "donation\_and\_volunteering": [16, 0.005], "requests\_or\_needs": [15, 0.005], "sympathy\_and\_support": [2, 0.01], "infrastructure\_and\_utilities\_damage": [14, 0.002], "affected\_individual": [13, 0.002], "caution\_and\_advi $\begin{array} { r l r } { \mathsf { C e } ^ { \mathsf { m } } : } & { { } [ 1 2 , } & { 0 . 0 0 2 ] ~ . } \end{array}$ "injured\_or\_dead\_people": [11, 0.002], " response\_efforts": [10, 0.002], II missing\_and\_found\_people": [9, 0.002], displaced\_and\_evacuations": [8, 0.002], II personal\_updates": [3, 0.008], " physical\_landslide": [7, 0.002], II disease\_related": [6, 0.002], " terrorism\_related": [5, 0.002], " other\_relevant\_information": [4, 0.007]}

## D. Evaluation

For both the informativeness and humanitarian label tasks, we calculate F1 scores to facilitate comparison with existing evaluations of datasets within CrisisBench. For the informativeness classification, we calculate macro (unweighted), weighted, and binary (only the positive class) F1 scores; for the humanitarian classification, we calculate weighted and macro (unweighted) class-averaged F1 scores. Given that some datasets in CrisisBench may cover only a subset of the 16 labels specified in the prompt, we maintain consistency in the prompt by including all 16 labels in our evaluation framework. However, the performance assessment of each dataset is based exclusively on the rankings of the labels present within that dataset. Class-specific precision for class i is computed as $\begin{array} { r } { { \sf P } _ { i } ~ = ~ \frac { \mathrm { T P } _ { i } } { \mathrm { T P } _ { i } + \mathrm { F P } _ { i } } } \end{array}$ , where $\mathrm { T P } _ { i } , \mathrm { F P } _ { i }$ stand for the count of true positives and false positives for class i, respectively. Classspecific recall is defined as $\begin{array} { r } { { \bf R } _ { i } = \frac { { \bf T } { \bf P } _ { i } } { { \bf T } { \bf P } _ { i } + { \bf F } { \bf N } _ { i } } } \end{array}$ , where $\mathrm { F N } _ { i }$ is the count of false negatives for class i. Class-specific F1 is defined as

$$
F 1 _ {i} = \frac {2 \mathrm{P} _ {i} \mathrm{R} _ {i}}{\mathrm{P} _ {i} + \mathrm{R} _ {i}}
$$

In the case where there is only one class, the class-specific F1 is equivalent to the binary F1.

TABLE I SUMMARY OF F1 SCORES FOR INFORMATIVENESS TASK

<table><tr><td rowspan="2">Model</td><td colspan="2">Event Aware</td><td rowspan="2">x</td></tr><tr><td colspan="2">Metric</td></tr><tr><td rowspan="3">Claude-3-5 Sonnet</td><td>macro</td><td>0.800</td><td>0.796</td></tr><tr><td>binary</td><td>0.836</td><td>0.836</td></tr><tr><td>weighted</td><td>0.808</td><td>0.805</td></tr><tr><td rowspan="3">Gemini 1.5-flash-001</td><td>macro</td><td>0.802</td><td>0.799</td></tr><tr><td>binary</td><td>0.855</td><td>0.848</td></tr><tr><td>weighted</td><td>0.814</td><td>0.810</td></tr><tr><td rowspan="3">GPT-4o</td><td>macro</td><td>0.819</td><td>0.801</td></tr><tr><td>binary</td><td>0.860</td><td>0.837</td></tr><tr><td>weighted</td><td>0.828</td><td>0.809</td></tr></table>

TABLE II  
F1 SCORES FOR INFORMATIVENESS TASK BY EACH DATASET ACROSS COMMERCIAL MODELS WITH AND WITHOUT EVENT AWARENESS

<table><tr><td rowspan="2">Model</td><td rowspan="2">Event Aware Metric</td><td colspan="2">CrisisLex6</td><td colspan="2">CrisisLex26</td><td colspan="2">CrisisNLP-cf</td><td colspan="2">CrisisNLP-vol</td><td colspan="2">AIDR</td><td colspan="2">DSM</td><td colspan="2">DRD</td><td colspan="2">ISCRAM2013</td><td colspan="2">SWDM13</td></tr><tr><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td></tr><tr><td rowspan="3">Claude-3-5 Sonnet</td><td>macro</td><td>0.840</td><td>0.825</td><td>0.603</td><td>0.617</td><td>0.759</td><td>0.739</td><td>0.667</td><td>0.687</td><td>0.746</td><td>0.739</td><td>0.728</td><td>0.716</td><td>0.740</td><td>0.743</td><td>0.498</td><td>0.481</td><td>0.555</td><td>0.557</td></tr><tr><td>binary</td><td>0.812</td><td>0.796</td><td>0.916</td><td>0.932</td><td>0.936</td><td>0.921</td><td>0.640</td><td>0.655</td><td>0.707</td><td>0.693</td><td>0.617</td><td>0.599</td><td>0.839</td><td>0.841</td><td>0.892</td><td>0.871</td><td>0.765</td><td>0.777</td></tr><tr><td>weighted</td><td>0.842</td><td>0.827</td><td>0.886</td><td>0.902</td><td>0.896</td><td>0.881</td><td>0.676</td><td>0.697</td><td>0.752</td><td>0.746</td><td>0.753</td><td>0.743</td><td>0.782</td><td>0.785</td><td>0.869</td><td>0.849</td><td>0.689</td><td>0.697</td></tr><tr><td rowspan="3">Gemini 1.5-flash-001</td><td>macro</td><td>0.860</td><td>0.834</td><td>0.590</td><td>0.600</td><td>0.744</td><td>0.738</td><td>0.656</td><td>0.689</td><td>0.718</td><td>0.725</td><td>0.787</td><td>0.774</td><td>0.687</td><td>0.695</td><td>0.560</td><td>0.536</td><td>0.588</td><td>0.598</td></tr><tr><td>binary</td><td>0.842</td><td>0.814</td><td>0.943</td><td>0.940</td><td>0.941</td><td>0.930</td><td>0.639</td><td>0.658</td><td>0.706</td><td>0.699</td><td>0.731</td><td>0.708</td><td>0.849</td><td>0.851</td><td>0.951</td><td>0.927</td><td>0.848</td><td>0.846</td></tr><tr><td>weighted</td><td>0.861</td><td>0.835</td><td>0.909</td><td>0.908</td><td>0.897</td><td>0.887</td><td>0.662</td><td>0.699</td><td>0.719</td><td>0.729</td><td>0.800</td><td>0.789</td><td>0.756</td><td>0.761</td><td>0.929</td><td>0.905</td><td>0.753</td><td>0.756</td></tr><tr><td rowspan="3">GPT-4o</td><td>macro</td><td>0.879</td><td>0.822</td><td>0.622</td><td>0.610</td><td>0.786</td><td>0.748</td><td>0.670</td><td>0.726</td><td>0.738</td><td>0.741</td><td>0.744</td><td>0.732</td><td>0.753</td><td>0.735</td><td>0.524</td><td>0.456</td><td>0.591</td><td>0.543</td></tr><tr><td>binary</td><td>0.862</td><td>0.789</td><td>0.940</td><td>0.922</td><td>0.950</td><td>0.923</td><td>0.645</td><td>0.682</td><td>0.702</td><td>0.701</td><td>0.644</td><td>0.624</td><td>0.855</td><td>0.847</td><td>0.926</td><td>0.836</td><td>0.822</td><td>0.743</td></tr><tr><td>weighted</td><td>0.880</td><td>0.824</td><td>0.910</td><td>0.892</td><td>0.913</td><td>0.884</td><td>0.678</td><td>0.740</td><td>0.743</td><td>0.747</td><td>0.766</td><td>0.756</td><td>0.796</td><td>0.782</td><td>0.903</td><td>0.815</td><td>0.738</td><td>0.670</td></tr></table>

TABLE III F1 SCORES FOR INFORMATIVENESS TASK ON CRISISMMD WITH/WITHOUT EVENT AWARENESS AND USE OF IMAGES

<table><tr><td rowspan="2">Model</td><td rowspan="2">Event AwareImage UsedMetric</td><td colspan="2"></td><td colspan="2">x</td></tr><tr><td></td><td>x</td><td></td><td>x</td></tr><tr><td rowspan="3">Claude-3-5 Sonnet</td><td>macro</td><td>0.760</td><td>0.712</td><td>0.743</td><td>0.731</td></tr><tr><td>binary</td><td>0.877</td><td>0.869</td><td>0.871</td><td>0.871</td></tr><tr><td>weighted</td><td>0.809</td><td>0.776</td><td>0.795</td><td>0.788</td></tr><tr><td rowspan="3">Gemini 1.5-flash-001</td><td>macro</td><td>0.701</td><td>0.703</td><td>0.745</td><td>0.729</td></tr><tr><td>binary</td><td>0.873</td><td>0.872</td><td>0.870</td><td>0.869</td></tr><tr><td>weighted</td><td>0.771</td><td>0.772</td><td>0.796</td><td>0.786</td></tr><tr><td rowspan="3">GPT-4o</td><td>macro</td><td>0.733</td><td>0.699</td><td>0.761</td><td>0.728</td></tr><tr><td>binary</td><td>0.876</td><td>0.870</td><td>0.868</td><td>0.870</td></tr><tr><td>weighted</td><td>0.791</td><td>0.769</td><td>0.805</td><td>0.786</td></tr></table>

The weighted class-average F1 is defined as

$$
F 1 _ {\text {weighted}} = \sum_ {i} \frac {n _ {i}}{N} \mathrm{F} 1 _ {i}
$$

where $n _ { i }$ is the number of instances where the true class is $i ,$ and N is the total number of instances. The macro classaverage F1 is defined as

$$
F 1 _ {\text {macro}} = \sum_ {i} \frac {1}{m} F 1 _ {i}
$$

where $m$ is the number of classes.

## III. RESULTS

In our experiment, the classification tasks for the crisis tweets on informativeness and type of humanitarian label are combined into one prompt. The model was prompted to: 1) determine if the social media post is informative in a humanitarian context, and 2) rank and assign probabilities to 16 potential humanitarian labels in how it fits the post. We report macro, weighted, and binary F1 scores for the informativeness task. For the humanitarian task, we use macro and weighted F1 scores.

Results for the informativeness task across the three models for all tested crisis tweets are in Table I. Weighted, binary and macro F1 scores are reported with and without event awareness. “Event-aware” indicates whether the name of a disaster contemporaneous with the tweet is included in the prompt. For each version of the prompt, the highest F1 scores are bolded. Models performed slightly better without eventawareness. OpenAI’s GPT-4o performed the best without event-awareness at a macro F1 score of 0.819, a binary F1 score at 0.860, and weighted F1 score at 0.828.

The results by dataset are in two tables, Table II contains all datasets not including CrisisMMD, and Table III contains the CrisisMMD results. These results are also separated by whether extra information (event and/or image) was included in the prompt. Across most datasets, OpenAI GPT-4o outperformed the other evaluated LLMs in informativeness classifications.

Based on reported metrics, the best LLMs compares moderately lower to existing benchmarks on the consolidated dataset [1], with a weighted F1 of 0.828 (LLM: GPT-4o) vs. 0.883 (fine-tuned RoBERTa). When looking at specific datasets where benchmarks were available, the LLMs also underperform, sometimes by a large margin: on CrisisMMD [10], (weighted F1 of .638 vs .842), on CrisisLexT26 [11] (macro F1 of .492 vs .848), and CrisisLexT6 [11] (macro F1 of .882 vs .947).

As a note, we contend that binary F1 (class-wise F1 for single class) as the most appropriate metric for this task. Macro and weighted F1 are most appropriate metrics for multi-class classification, in which F1 scores for multiple classes are condensed into a single metric. For this binary classification problem, the “informative” class serves as a foreground and the “not informative” class as a background; binary F1 appropriately privileges the foreground class as being the one which is useful to distinguish. The use of macro and weighted F1 are significantly impacted by the number of negative examples in the dataset, which is not desirable when the task is trying to find positive instances in a haystack of negative examples. The LLMs obtain substantially better binary F1 scores than macro F1. Unfortunately, binary F1 is not reported for the informativeness task in any literature we found, making direct comparison on the metric difficult.

For the informativeness task, the inclusion of images (in CrisisMMD) and event context (all datasets) did not significantly affect F1 scores, with changes dependent on the dataset. For humanitarian classification, including event context slightly improved scores, while including images did not.

For the humanitarian classification task, we compute both weighted and macro F1 scores, and treat the highest-ranked label within the subset of labels from the constituent dataset as the predicted label for evaluation. These results are reported in Table IV. The highest F1 score for both macro and weighted for each dataset are bolded for each variation of the prompt. Anthropic Claude-3-5 Sonnet generally performed better than other tested LLMs in both weighted and macro f1-scores. The F1 scores of the LLMs are broadly low, but inconsistencies in the CrisisBench dataset are likely substantially responsible for this fact, which we will explore further in the discussion. We also note that this evaluation of humanitarian labels is preliminary, as it only includes the subset of tweets in the informativeness task that also happened to have humanitarian label annotations.

TABLE IV  
MACRO AND WEIGHTED F1 SCORES FOR HUMANITARIAN CLASS IFICATION BY EACH DATASET

<table><tr><td rowspan="3"></td><td>Dataset</td><td colspan="4">CrisisMMD</td><td colspan="2">CrisisLex6</td><td colspan="2">CrisisLex26</td><td colspan="2">CrisisNLP-cf</td><td colspan="2">CrisisNLP-vol</td><td colspan="2">AIDR</td><td colspan="2">DRD</td><td colspan="2">ISCRAM2013</td><td colspan="2">SWDM13</td></tr><tr><td>Event Aware</td><td colspan="2"></td><td>x</td><td></td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td></tr><tr><td>Image Used Metric</td><td></td><td>x</td><td></td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="22">Model</td></tr><tr><td rowspan="2">Claude-3-5 Sonnet</td><td>macro</td><td>0.537</td><td>0.508</td><td>0.554</td><td>0.493</td><td>0.836</td><td>0.832</td><td>0.461</td><td>0.492</td><td>0.564</td><td>0.553</td><td>0.263</td><td>0.250</td><td>0.345</td><td>0.333</td><td>0.700</td><td>0.680</td><td>0.530</td><td>0.547</td><td>0.486</td><td>0.540</td></tr><tr><td>weighted</td><td>0.625</td><td>0.557</td><td>0.638</td><td>0.558</td><td>0.837</td><td>0.834</td><td>0.467</td><td>0.497</td><td>0.583</td><td>0.570</td><td>0.546</td><td>0.558</td><td>0.623</td><td>0.612</td><td>0.837</td><td>0.823</td><td>0.653</td><td>0.618</td><td>0.590</td><td>0.666</td></tr><tr><td rowspan="2">Gemini 1.5-flash-001</td><td>macro</td><td>0.500</td><td>0.466</td><td>0.533</td><td>0.472</td><td>0.869</td><td>0.838</td><td>0.439</td><td>0.445</td><td>0.546</td><td>0.540</td><td>0.193</td><td>0.219</td><td>0.363</td><td>0.346</td><td>0.488</td><td>0.501</td><td>0.541</td><td>0.526</td><td>0.440</td><td>0.440</td></tr><tr><td>weighted</td><td>0.531</td><td>0.510</td><td>0.565</td><td>0.527</td><td>0.869</td><td>0.839</td><td>0.438</td><td>0.452</td><td>0.565</td><td>0.549</td><td>0.540</td><td>0.598</td><td>0.637</td><td>0.626</td><td>0.617</td><td>0.633</td><td>0.610</td><td>0.581</td><td>0.549</td><td>0.549</td></tr><tr><td rowspan="2">GPT-4o</td><td>macro</td><td>0.513</td><td>0.485</td><td>0.531</td><td>0.504</td><td>0.882</td><td>0.822</td><td>0.448</td><td>0.464</td><td>0.603</td><td>0.583</td><td>0.254</td><td>0.262</td><td>0.341</td><td>0.300</td><td>0.627</td><td>0.611</td><td>0.532</td><td>0.538</td><td>0.505</td><td>0.433</td></tr><tr><td>weighted</td><td>0.568</td><td>0.545</td><td>0.586</td><td>0.563</td><td>0.882</td><td>0.824</td><td>0.451</td><td>0.468</td><td>0.626</td><td>0.585</td><td>0.596</td><td>0.644</td><td>0.625</td><td>0.605</td><td>0.773</td><td>0.754</td><td>0.645</td><td>0.612</td><td>0.624</td><td>0.537</td></tr></table>

## IV. DISCUSSION

Overall, LLMs perform reasonably well on the informativeness task, achieving zero-shot performance on the consolidated dataset within 6% of that of pretrained classifiers in [1].

However, performance was substantially worse on the multiclass humanitarian label task. The LLMs broadly underperformed the models trained in CrisisTransformers [20]. We note several limitations and challenges that may have contributed to this. The task performance of the LLMs tested may have been limited by a number of factors, including the absence of optimizations like prompt engineering or fine-tuning. Further, fine-tuned models may perform better because they were finetuned on each dataset individually and were able to fit the base rates at which various classes occur. More fundamentally, however, we note that methods in the construction of Crisis-Bench dataset had a substantial impact on the multi-class task performance for the LLMs.

Whereas the zero-shot classification technique for LLMs uses natural language prompting to describe the criteria for each class based on the name and description of the label, traditional models are trained on the training dataset. This raises a potential issue if there is misalignment between the labeled examples and the semantic understanding of the label. We identified methods related to the construction of CrisisBench which may contribute to such misalignment.

TABLE VACCURACY RATES OF TWO CLASSES FOR ANTHROPIC’S CLAUDE-3.5SONNET AND MANUALLY ASS IGNED LABELS COMPARED TOGROUND-TRUTH

<table><tr><td>Label</td><td>Accuracy (ground-truth) %</td><td>Accuracy (manual) %</td></tr><tr><td>affected_individual</td><td>13.3</td><td>76.0</td></tr><tr><td>caution_and_advice</td><td>14.7</td><td>48.0</td></tr><tr><td>disease_related</td><td>56.0</td><td>74.7</td></tr><tr><td>affected_individual (ground-truth)</td><td>100</td><td>29.3</td></tr><tr><td>caution_and_advice (ground-truth)</td><td>100</td><td>61.3</td></tr><tr><td>disease_related (ground-truth)</td><td>100</td><td>65.3</td></tr></table>

CrisisBench draws from a number of datasets, aggregating labels with potentially different definitions into single classes. For example, the infrastructure\_and\_utilities\_damage class is defined in CrisisNLP-volunteers to be the destruction of houses, buildings, or roads, or the interruption of utilities, but CrisisNLP-CF defines it to include restoration of utilities as well [1]. Furthermore, the classes of constituent datasets are sometimes defined such that they cannot be mapped onto a single CrisisBench label. The CrisisLexT26 affected\_individual class contributes all of the affected\_individual examples in CrisisBench. But in CrisisLexT26, this class is defined to include personal updates, which is a separate CrisisBench class. The only way for an LLM to correctly categorize a personal update is to correctly guess whether it came from CrisisLexT26 or not. As CrisisLexT26 is one of the largest datasets labeled for humanitarian class, it is unsurprising that personal\_update and affected\_individual are the two lowest-performing classes for the evaluated LLMs.

To better understand the impact of ambiguous label mappings on performance, we performed manual binary annotation on two of the lowest-performing classes, affected\_individual and caution\_and\_advice. We examined 75 randomly sampled tweets which were assigned to the two classes (for a total of 150 tweets) by either the ground truth label or the maximum likelihood estimate of the LLM, without knowledge of the ground-truth or predicted label of any particular tweet. We manually performed binary classification on each tweet as either matching or not matching the description of its reference class given in the prompt (for example, we might look at a tweet with the knowledge that at least one of the ground truth label or the predicted label was affected\_individual, and assign a binary label based on whether we believed the tweet matched the definition we gave for that class). The results of this experiment are in Table V. This experiment suggests that semantic differences in the labels, which would not have affected models trained on the training data [20], had a substantial impact on the performance of the LLMs. It also showed generally low agreement between our manual labels and the ground truth, with our manual labels matching the LLM labels more often than the ground truth on the affected\_individuals class. While the LLMs performed better on our manual labels than on the ground truth in general, the difference in performance is much larger in the cases where agreement between our labels and the ground truth was relatively weaker. To verify that we were not observing a regression to the mean by injecting noise into the labels, we also analyzed a further 75 tweets from a high-performing class, disease\_related. Even on this higher-performing class, accuracy is substantially better when comparing against manually labeled examples as opposed to the ground truth labels. We also observe that accuracy as measured against the ground truth is higher on labels where the ground truth and manual labeling have higher agreement.

The significant level of variation in F1 scores between datasets across both classification tasks merits further investigation. For example, the binary F1 Score for OpenAI’s GPT-4o of 0.950 suggests strong performance for the positive label on the CrisisNLP-cf dataset (labeled by paid crowd workers), and much worse performance with 0.682 on the CrisisNLP-vol dataset (labeled by unpaid volunteers). One possible explanation for this is that data quality varies substantially between constituent datasets.

We also note a couple additional challenges during implementation that practitioners and researchers using LLMs should be aware of. Occasionally, LLMs would refuse to classify a tweet due to objectionable subject material (pornographic content, hate speech, and foul language). In addition, LLMs sometimes struggled to output correctly formatted JSON—this was usually able to be resolved by resubmitting the prompt, but on rare occasion, the request failed past our patience threshold and had to be omitted. There were also a small number of tweets that were misconstrued as part of the prompt, leading the LLM to respond that it did not detect a tweet to classify. A stronger distinction between the prompt and tweet to be classified, perhaps using a special token, would help ameliorate this. The total number of refused/omitted tweets were small (on the order of 10 per model), and thus should not affect the evaluation scores.

## V. CONCLUSION

In this paper, we present the performance of commercial large language models on zero-shot classification for two tasks on short social media posts on CrisisBench. We find that overall performance on the binary informativeness task is strong, even relative to models fine-tuned on the evaluation datasets. Incorporating extra information in the form of possible event context and images did not substantially impact the model’s performance on the task.

For the second task, humanitarian classification, an ambiguous multi-class task performance rapidly declined, emphasizing the need for careful deployment of these tools to the humanitarian space. Based on small-scale experiments with manual labeling, we attribute most of the LLMs’ declining performance to semantic ambiguity in social media posts and their labels rather than a latent inability to parse and classify natural language.

In future work, we plan to include open-source models in our classification assessments. We will substantially reduce problems associated with the dataset aggregation performed by CrisisBench by changing the prompt and label definitions based on source dataset. This will also provide an avenue for further analysis of each dataset’s quality. Further, we plan to analyze the results by language to better understand the multilingual components of the LLMs in relation to humanitarian classification tasks. Another avenue of future research is to assess the impact of prompt engineering more broadly. For example, we prompt for both classification tasks in the same prompt, but it would be of interest to look into the extent to which the dual classification task in one prompt impacts model performance.

## REFERENCES

[1] F. Alam, H. Sajjad, M. Imran, and F. Ofli, “CrisisBench: Benchmarking crisis-related social media datasets for humanitarian information processing,” in Proceedings of the International AAAI Conference on Web and Social Media, ser. ICWSM ’21, vol. 15, May 2021, pp. 923–932. [Online]. Available: https://ojs.aaai.org/index.php/ICWSM/article/view/18115.  
[2] T. B. Brown, B. Mann, N. Ryder, et al., Language models are few-shot learners, 2020. arXiv: 2005.14165 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2005.14165.  
[3] OpenAI, J. Achiam, S. Adler, et al., GPT-4 technical report, 2024. arXiv: 2303.08774 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2303.08774.  
[4] M. Imran, P. Mitra, and C. Castillo, “Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages,” in Proc. of the LREC, 2016, Paris, France: ELRA, May 2016, ISBN: 978-2-9517408- 9-1.  
[5] F. Alam, F. Ofli, and M. Imran, “CrisisMMD: Multimodal twitter datasets from natural disasters,” in Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM), USA, Jun. 2018.  
[6] A. Olteanu, S. Vieweg, and C. Castillo, “What to expect when the unexpected happens: Social media communications across crises,” in Proc. of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing, ACM, 2015, pp. 994–1009.  
[7] A. Olteanu, C. Castillo, F. Diaz, and S. Vieweg, “CrisisLex: A lexicon for collecting and filtering microblogged communications in crises.,” in Proc. of the 8th ICWSM, 2014, AAAI press, 2014.  
[8] M. Imran, S. Elbassuoni, C. Castillo, F. Diaz, and P. Meier, “Practical extraction of disaster-relevant information from social media,” in Proc. of the 22nd WWW, ACM, 2013, pp. 1021–1024.  
[9] M. Imran, S. M. Elbassuoni, C. Castillo, F. Diaz, and P. Meier, “Extracting information nuggets from disasterrelated messages in social media,” in Proc. of the 12th ISCRAM, 2013.  
[10] F. Ofli, F. Alam, and M. Imran, “Analysis of social media data using multimodal deep learning for disaster response,” in The 17th International Conference on In-  
formation Systems for Crisis Response and Management (ISCRAM 2020), 2020.  
[11] H. Li, D. Caragea, and C. Caragea, “Combining selftraining with deep learning for disaster tweet classification,” in The 18th International Conference on Information Systems for Crisis Response and Management (ISCRAM 2021), 2021.  
[12] A. Mondal, A. Kesan, A. Rodrigues, and J. George, “An efficient multi-modal classification approach for disaster-related tweets,” in 2022 IEEE International Conference on Distributed Computing and Electrical Circuits and Electronics (ICDCECE), IEEE, 2022, pp. 1–4. DOI: 10.1109/ICDCECE53908.2022.9792951.  
[13] S. K. Dasari, S. Gorla, and P. R. PVGD, “A stacking ensemble approach for identification of informative tweets on twitter data,” International Journal of Information Technology, vol. 15, no. 5, pp. 2651–2662, 2023.  
[14] K. Asinthara, M. Jayan, and L. Jacob, “Categorizing disaster tweets using learning based models for emergency crisis management,” in 2023 9th International conference on advanced computing and communication systems (ICACCS), IEEE, vol. 1, 2023, pp. 1133–1138. DOI: 10.1109/ICACCS57279.2023.10113105.  
[15] D. S. Krishna, G. Srinivas, and P. P. Reddy, “A deep parallel hybrid fusion model for disaster tweet classification on twitter data,” Decision Analytics Journal, vol. 11, p. 100 453, 2024. DOI: 10.1016/j.dajour.2024.100453.  
[16] J. Li, Y. Wang, and W. Li, “MHRN: A multimodal hierarchical reasoning network for topic detection,” IEEE Transactions on Multimedia, 2024. DOI: 10.1109/TMM.2024.3358696.  
[17] M. Franc¸ois and P. Gay, “Active learning with few shot learning for crisis management,” in Proceedings of the 20th International Conference on Contentbased Multimedia Indexing, 2023, pp. 233–237. DOI: 10.1145/3617233.  
[18] A. Dahou, A. Mabrouk, A. A. Ewees, M. A. Gaheen, and M. Abd Elaziz, “A social media event detection framework based on transformers and swarm optimization for public notification of crises and emergency management,” Technological Forecasting and Social Change, vol. 192, p. 122 546, 2023. DOI: 10.1016/j.techfore.2023.122546.  
[19] M. S. I. Malik, M. Z. Younas, M. M. Jamjoom, and D. I. Ignatov, “Categorization of tweets for damages: Infrastructure and human damage assessment using finetuned bert model,” PeerJ Computer Science, vol. 10, e1859, 2024. DOI: 10.7717/peerj-cs.1859.  
[20] R. Lamsal, M. R. Read, and S. Karunasekera, “CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts,” Knowledge-Based Systems, vol. 296, p. 111 916, Jul. 2024, ISSN: 0950-7051. DOI: 10.1016/j.knosys.2024.111916. [Online]. Available: http://dx.doi.org/10.1016/j.knosys.2024.111916.  
[21] Y. Xian, C. H. Lampert, B. Schiele, and Z. Akata, “Zero-shot learning: A comprehensive evaluation of the good, the bad and the ugly,” IEEE transactions on pattern analysis and machine intelligence, vol. 41, no. 9, pp. 2251–2265, 2018. DOI: 10.1109/TPAMI.2018.2857768.  
[22] T. Kojima, S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, et al., Eds., vol. 35, Curran Associates, Inc., 2022, pp. 22 199–22 213.  
[23] G. Li, P. Wang, and W. Ke, “Revisiting large language models as zero-shot relation extractors,” in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds., Singapore: Association for Computational Linguistics, Dec. 2023, pp. 6877–6892. [Online]. Available: https://aclanthology.org/2023.findings-emnlp.459.  
[24] X. Wu, X. He, T. Liu, N. Liu, and X. Zhai, “Matching Exemplar as Next Sentence Prediction (MeNSP): Zero-Shot Prompt Learning for Automatic Scoring in Science Education,” in International Conference on Artificial Intelligence in Education, Springer, 2023, pp. 401–413. DOI: 10.1007/978-3-031-36272-9 33.  
[25] H. Yu, P. Guo, and A. Sano, “Zero-shot ecg diagnosis with large language models and retrieval-augmented generation,” in Proceedings of the 3rd Machine Learning for Health Symposium, ser. Proceedings of Machine Learning Research, vol. 225, PMLR, Oct. 2023, pp. 650–663.  
[26] L. Zheng, W.-L. Chiang, Y. Sheng, et al., “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, et al., Eds., vol. 36, Curran Associates, Inc., 2023, pp. 46 595–46 623.  
[27] E. Latif and X. Zhai, “Fine-tuning chatgpt for automatic scoring,” Computers and Education: Artificial Intelligence, vol. 6, p. 100 210, 2024. DOI: 10.1016/j.caeai.2024.100210.  
[28] M. Sushil, T. Zack, D. Mandair, et al., “A comparative study of large language model-based zero-shot inference and task-specific supervised classification of breast cancer pathology reports,” Journal of the American Medical Informatics Association, ocae146, 2024. DOI: 10.1093/jamia/ocae146.  
[29] OpenAI, Hello GPT-4o, 2024. [Online]. Available: https://openai.com/index/hello-gpt-4o/.  
[30] Google, Gemini breaks new ground with a faster model, longer context, AI agents and more, 2024. [Online]. Available: https://blog.google/technology/ai/google-gemini-update-flash-ai-ass  
[31] Anthropic, Claude 3.5 Sonnet, 2024. [Online]. Available: https://www.anthropic.com/news/claude-3-5-sonnet.

This figure "fig1.png" is available in "png" format from:

http://arxiv.org/ps/2410.00182v1