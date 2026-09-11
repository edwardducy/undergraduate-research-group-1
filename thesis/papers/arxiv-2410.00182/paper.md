# Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models\*  
 Thanks: This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force. © 2024 Massachusetts Institute of Technology. Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

 Emma L. McDaniel12, Samuel Scheele13, Jeffrey Liu4 Affiliation: 1 indicates equal contribution  
2Computer Science Department  
Georgia State University  Atlanta, GA, USA Affiliation: 234Humanitarian Assistance and Disaster Relief Systems  
MIT Lincoln Laboratory  Lexington, MA, USA  
2emcdaniel10@gsu.edu, 3samuel.scheele@ll.mit.edu, 4jeffrey.liu@ll.mit.edu 

###### Abstract

Social media posts are frequently identified as a valuable source of open-source intelligence for disaster response, and pre-LLM NLP techniques have been evaluated on datasets of crisis tweets. We assess three commercial large language models (OpenAI GPT-4o, Gemini 1.5-flash-001 and Anthropic Claude-3-5 Sonnet) capabilities in zero-shot classification of short social media posts. In one prompt, the models are asked to perform two classification tasks: 1) identify if the post is informative in a humanitarian context; and 2) rank and provide probabilities for the post in relation to 16 possible humanitarian classes. The posts being classified are from the consolidated crisis tweet dataset, CrisisBench. Results are evaluated using macro, weighted, and binary F1-scores. The informative classification task, generally performed better without extra information, while for the humanitarian label classification providing the event that occurred during which the tweet was mined, resulted in better performance. Further, we found that the models have significantly varying performance by dataset, which raises questions about dataset quality.

###### Index Terms: 

Large Language Models, Zero-Shot Classification, Crisis Classification, Social Media 

## I Introduction

In crisis scenarios, such as natural hazard-induced disasters or humanitarian emergencies, timely and accurate information is crucial to decision makers. Social media posts can provide valuable information in real time; however, the sheer speed and quantity of data coming from social media can be overwhelming for human analysts to process. As such, Natural Language Processing (NLP) techniques have been used to automate the processing of social media data in order to classify and extract the most relevant information. CrisisBench \[[1](#bib.bibx1 "")\] provides a benchmark dataset to evaluate the performance of NLP solutions for classifying crisis-related social media posts.

Recently, Large Language Models (LLMs) and Large Multimodal Models (LMMs) have shown impressive performance on a wide range of NLP tasks without needing task-specific training or fine-tuning. Large Language Models, such as GPT \[[2](#bib.bibx2 ""), [3](#bib.bibx3 "")\], are trained on massive text datasets to predict the next word in a sequence, and can be used to generate answers to questions. They can thus be used as zero-shot text classifiers by inputting the relevant text, followed by a question asking which of a set of given labels or classes apply to the text. Large Multimodal Models can be trained and utilized similarly, except they are configured to accept other data modalities, such as images, in addition to text.

Due to the recent popularity of LLMs, we expect that humanitarian practitioners will try to use them to help automate the process of extracting relevant information from social media during crises. As a first step towards characterizing the performance of LLM/LMMs on such tasks and identifying which ones provide the best performance, we evaluate various open-access and commercial LLMs and LMMs on zero-shot classification of social media posts using the CrisisBench dataset. In addition, we compare the performance of the zero-shot classifiers to the existing benchmarks from purpose-built classifiers for crisis-related social media text classification.

### I-A Related Work

CrisisBench \[[1](#bib.bibx1 "")\] combines a number of crisis datasets \[[4](#bib.bibx4 ""), [5](#bib.bibx5 ""), [6](#bib.bibx6 ""), [7](#bib.bibx7 ""), [8](#bib.bibx8 ""), [9](#bib.bibx9 "")\] through cleaning and standardizing labels in order to create a benchmark for measuring performance of NLP classification of crisis-related social media posts. CrisisBench defines two tasks. The “informativeness” task, a binary classification task that seeks to identify whether a provided tweet contains valuable information regarding a disaster or crisis event. The “humanitarian information type,” a multi-class classification task that seeks to categorize a tweet into one of 16 classes (e.g donation and volunteering, displaced and evacuations).

Previous work in classifying crisis-related social media posts has involved conventional machine learning methodologies and non-transformer-based neural networks \[[10](#bib.bibx10 ""), [11](#bib.bibx11 ""), [12](#bib.bibx12 ""), [13](#bib.bibx13 ""), [14](#bib.bibx14 ""), [15](#bib.bibx15 "")\], fine-tuned transformer-based models for multimodal classification using images embedded within social media messages \[[16](#bib.bibx16 "")\], and fine-tuned transformer-based models focusing solely on text \[[11](#bib.bibx11 ""), [17](#bib.bibx17 ""), [18](#bib.bibx18 ""), [19](#bib.bibx19 ""), [20](#bib.bibx20 "")\].

In contrast to developing task-specific models, a growing trend involves leveraging instruction-tuned LLMs and LMMs for zero-shot classification \[[21](#bib.bibx21 ""), [22](#bib.bibx22 ""), [23](#bib.bibx23 ""), [24](#bib.bibx24 ""), [25](#bib.bibx25 ""), [26](#bib.bibx26 ""), [27](#bib.bibx27 ""), [28](#bib.bibx28 "")\]. While zero-shot classification circumvents the need for extensive labeled training data for fine-tuning, it is essential to understand various models’ limitations within specific domains. Our work attempts to address this need in the realm of humanitarian assistance by providing performance statistics for a range of commercial models.

## II Methodology

### II-A CrisisBench Task Descriptions

In this paper, we focus on the “informativeness” task from the CrisisBench consolidated dataset, and provide incidental analysis of the “humanitarian information type” task for those data points that also had “humanitarian information” labels. Roughly 5,000 of the examples in the informativeness test set are also in the humanitarian information type test set - these examples are the only ones considered for the analysis of the humanitarian information type task.

The motivation for this is that the classes in the multi-class task are often amalgamations of classes from the constituent datasets, and most constituent datasets used only a few of the classes. In the Discussion, we will provide preliminary analysis of the multi-class task where semantic differences in definitions across constituent datasets substantially impacted performance. The “informativeness” task does not suffer from the same ambiguity as the “humanitarian information type” task and results are therefore easier to obtain and interpret.

A subset of tweets in the CrisisBench dataset are from CrisisMMD dataset, which contains only tweets which include images \[[5](#bib.bibx5 "")\]. The CrisisBench authors also include an event type annotation that indicates the type of crisis event that was contemporaneous with the timestamp of the tweet. We evaluate each task both with/without event awareness, and with/without images for four configurations per task.

### II-B Models Evaluated

We evaluate three commercial models: OpenAI’s GPT-4o \[[29](#bib.bibx29 "")\], Google’s Gemini 1.5 Flash \[[30](#bib.bibx30 "")\], and Anthropic’s Claude Sonnet 3.5 \[[31](#bib.bibx31 "")\], and accessed them through their respective APIs. The models were chosen based on several considerations, including prominence, performance on other benchmarks, and availability.

### II-C Prompt Structure

We used the same base prompt for all models in the CrisisBench dataset, in which we requested that the model return a JSON string with a specified schema. We used Pydantic to validate the JSON. In the case where the model did not return valid JSON, we simply re-submitted the prompt and retried up to a set patience of three attempts. Responses that were not valid were omitted from analysis.

We asked the models to complete both the “informativeness” task as well as the multi-class “humanitarian information type” classification task in the same prompt. For the informativeness task the model provides a true or false The base prompt is provided below, [⬇](data:text/plain;base64,UHJvdmlkZSBjbGFzc2lmaWNhdGlvbnMgb2YgdGhlIGZvbGxvd2luZyB0d2VldCBiYXNlZCBvbiBpdHMgcmVsZXZhbmNlIHRvIGEgaHVtYW5pdGFyaWFuIGV2ZW50IGFuZCBhIGNsYXNzaWZpY2F0aW9uIG9mIGl0cyBjb250ZW50LiAiXAogICAgICAgICJ7aW1nX3N0cn0iXAogICAgICAgICJ7ZXZlbnRfc3RyfSJcCiAgICAgICAgIlRoZSB0d2VldCBmb2xsb3dzOlxue3R3ZWV0X3N0cn0gXG4iXAogICAgICAgICJ7ZmllbGRfZGVzY3JpcHRpb25zfSI=)

Provide classifications of the following tweet based on its relevance to a humanitarian event and a classification of its content. "\\ 

 "{img\_str}"\\ 

 "{event\_str}"\\ 

 "The tweet follows:\\n{tweet\_str} \\n"\\ 

 "{field\_descriptions}" 

where the fields {img\_str}, {event\_str}, {tweet\_str}, and {field\_descriptions} are placeholders.

The placeholder {img\_str} was filled in with the text "Use the images, if present, to help you make a your determinations related to the informativeness and category of the tweet." if an image was associated with the tweet; otherwise, it was left blank. Images were resized to fit within 768×768768\\times 768 pixels while maintaining aspect ratio, encoded in base64, and appended to the prompt in accordance to the respective LMM’s specifications.

The placeholder {event\_str} was filled in with "While it may still be irrelevant or uninformative, this tweet was created around the time of a disaster with description: {event\_type}.", where {event\_type} corresponds to the event that was occurring during the time of the tweet if we were evaluating the tweet in the “event-aware” configuration; otherwise, {event\_str} was left blank. The {tweet\_str} placeholder contained the actual text of the tweet.

The {field\_descriptions} placeholder contained descriptions of the classes as well as the desired JSON format for the output. We requested two fields: is\_informative and humanitarian\_label. For the is\_informative field, the prompt was "Does the tweet contain information pertinent to a humanitarian event or natural disaster? Respond with a boolean true/false". For the humanitarian\_label field, the prompt is provided below: [⬇](data:text/plain;base64,Rm9yIGEgZ2l2ZW4gdHdlZXQsIGRldGVybWluZSB3aGljaCBvZiB0aGUgaHVtYW5pdGFyaWFuIGxhYmVscyBhcmUgbW9zdCByZWxldmFudDoKVGhlIGh1bWFuaXRhcmlhbiBsYWJlbHMgYW5kIHRoZWlyIGRlc2NyaXB0aW9ucyBhcmUgcHJvdmlkZWQgYmVsb3c6CiAgICAibm90X2h1bWFuaXRhcmlhbiIgLSBUaGUgdHdlZXQgaXMgbm90IGh1bWFuaXRhcmlhbiBpbiBuYXR1cmUgYW5kIGRvZXMgbm90IGZpdCBpbnRvIGFueSBvdGhlciBjbGFzcy4KICAgICJkb25hdGlvbl9hbmRfdm9sdW50ZWVyaW5nIiAtIFRoZSB0d2VldCByZWxhdGVzIHRvIGRpcmVjdGluZywgYWNjZXB0aW5nLCBvciBkaXN0cmlidXRpbmcgZG9uYXRpb25zIG9yIHZvbHVudGVlciBlZmZvcnQuCiAgICAicmVxdWVzdHNfb3JfbmVlZHMiIC0gVGhlIHR3ZWV0IGRlc2NyaWJlcyBhIHJlcXVlc3Qgb3IgbmVlZCBvZiBhbiBpbmRpdmlkdWFsIG9yIGNvbW11bml0eS4KICAgICJzeW1wYXRoeV9hbmRfc3VwcG9ydCIgLSBUaGUgdHdlZXQgZXhwcmVzc2VzIHN5bXBhdGh5IG9yIHN1cHBvcnQgZm9yIGRpc2FzdGVyIHZpY3RpbXMuCiAgICAiaW5mcmFzdHJ1Y3R1cmVfYW5kX3V0aWxpdGllc19kYW1hZ2UiIC0gVGhlIHR3ZWV0IHJlbGF0ZXMgdG8gdGhlIGNvbnN0cnVjdGlvbiBvciBkZXN0cnVjdGlvbiBvZiBpbmZyYXN0cnVjdHVyZSwgdXRpbGl0aWVzLCBvciBzdHJ1Y3R1cmVzLgogICAgImFmZmVjdGVkX2luZGl2aWR1YWwiIC0gVGhlIHR3ZWV0IGNvbnRhaW5zIGluZm9ybWF0aW9uIG9uIGEgcGFydGljdWxhciBpbmRpdmlkdWFsIGFmZmVjdGVkIGJ5IGEgZGlzYXN0ZXIuCiAgICAiY2F1dGlvbl9hbmRfYWR2aWNlIiAtIFRoZSB0d2VldCBjb250YWlucyBjYXV0aW9uIG9yIGFkdmljZSBmb3IgdmljdGltcywgcmVzcG9uZGVycywgb3Igb3RoZXJzLgogICAgImluanVyZWRfb3JfZGVhZF9wZW9wbGUiIC0gVGhlIHR3ZWV0IG5vdGVzIHRoZSBwcmVzZW5jZSBvZiBpbmp1cmVkIG9yIGRlYWQgcGVvcGxlLgogICAgInJlc3BvbnNlX2VmZm9ydHMiIC0gVGhlIHR3ZWV0IHBlcnRhaW5zIHRvIHRoZSByZXNwb25zZSBlZmZvcnQuCiAgICAibWlzc2luZ19hbmRfZm91bmRfcGVvcGxlIiAtIFRoZSB0d2VldCBkaXNjdXNzZXMgbWlzc2luZyBwZXJzb25zLCBpbmNsdWRpbmcgaW4gdGhlIGNvbnRleHQgb2YgZmluZGluZyB0aGVtLgogICAgImRpc3BsYWNlZF9hbmRfZXZhY3VhdGlvbnMiIC0gVGhlIHR3ZWV0IHJlbGF0ZXMgdG8gZGlzcGxhY2VkIHBlb3BsZSBvciBhbiBldmFjdWF0aW9uIHByb2Nlc3MuCiAgICAicGVyc29uYWxfdXBkYXRlcyIgLSBUaGUgdHdlZXQgcmVsYXRlcyB0byBhIHBlcnNvbmFsIG9waW5pb24gb3IgYSBzdGF0dXMgdXBkYXRlIGFib3V0IHRoZSB0d2VldCBhdXRob3Igb3IgdGhlaXIgY2xvc2UgcmVsYXRpb25zLgogICAgInBoeXNpY2FsX2xhbmRzbGlkZSIgLSBUaGUgdHdlZXQgaXMgcmVsYXRlZCB0byBhIHBoeXNpY2FsIGxhbmRzbGlkZS4KICAgICJkaXNlYXNlX3JlbGF0ZWQiIC0gVGhlIHR3ZWV0IHJlcG9ydHMgb24gZGlzZWFzZSB0cmFuc21pc3Npb25zLCBzeW1wdG9tcywgdHJlYXRtZW50LCBwcmV2ZW50aW9uLCBvciBhZmZlY3RlZCBwZW9wbGUuCiAgICAidGVycm9yaXNtX3JlbGF0ZWQiIC0gVGhlIHR3ZWV0IHJlcG9ydHMgcG9zc2libGUgdGVycm9yaXNtIG9yIHRlcnJvcmlzdCBhY3RzLgogICAgIm90aGVyX3JlbGV2YW50X2luZm9ybWF0aW9uIiAtIFRoZSB0d2VldCBpcyBodW1hbml0YXJpYW4gaW4gbmF0dXJlLCBidXQgZG9lcyBub3QgZml0IGluIGFueSBvdGhlciBjbGFzcy4KVGhlIG91dHB1dCBzaG91bGQgYmUgZm9ybWF0dGVkIGFzIGEgZGljdGlvbmFyeSB3aG9zZSBrZXlzIGFyZSB0aGUgaHVtYW5pdGFyaWFuIGxhYmVscywgYW5kIHRoZSB2YWx1ZXMgYXJlIHR3by1lbGVtZW50IGFycmF5cyB3aG9zZSBlbnRyaWVzIGNvcnJlc3BvbmQgdG8gdGhlIGZvbGxvd2luZzoKVGhlIGZpcnN0IGVsZW1lbnQgb2YgdGhlIGFycmF5IHNob3VsZCBiZSBhIHJhbmtpbmc6IGFuIGludGVnZXIgZnJvbSAxIHRvIDE2IHJlcHJlc2VudGluZyB0aGUgcmVsYXRpdmUgcmVsZXZhbmNlIG9mIHRoZSBodW1hbml0YXJpYW4gbGFiZWwgY29tcGFyZWQgdG8gdGhlIG90aGVycy4gVGhlIG1vc3QgcmVsZXZhbnQgbGFiZWwgc2hvdWxkIGJlIHJhbmtlZCAxLCBhbmQgbGVhc3QgcmVsZXZhbnQgc2hvdWxkIGJlIHJhbmtlZCAxNi4KVGhlIHNlY29uZCBlbGVtZW50IHNob3VsZCBiZSBhIGxpa2VsaWhvb2Qgc2NvcmU6IGEgZmxvYXRpbmcgcG9pbnQgbnVtYmVyIGJldHdlZW4gMCBhbmQgMSByZXByZXNlbnRpbmcgdGhlIGxpa2VsaWhvb2QgdGhhdCB0aGUgbGFiZWwgYXBwbGllcyB0byB0aGUgdHdlZXQuClRoZSBkaWN0aW9uYXJ5IHNob3VsZCBoYXZlIGFuIGVudHJ5IGZvciBldmVyeSBodW1hbml0YXJpYW4gbGFiZWwsIGV2ZW4gaWYgaXQgaXMgbm90IHJlbGV2YW50LiBUaGUgcmFua2luZ3MgZm9yIGVhY2ggbGFiZWwgc2hvdWxkIGJlIHVuaXF1ZS0tLXRoYXQgaXMsIG5vIHR3byBsYWJlbHMgc2hvdWxkIGhhdmUgdGhlIHNhbWUgcmFua2luZywgZXZlbiBpZiB0aGV5IGFyZSBib3RoIG5vdCByZWxldmFudDogeW91IG11c3QgcmFuayBvbmUgaGlnaGVyIHRoYW4gdGhlIG90aGVyLS0tYW5kIHRoZSBsaWtlbGlob29kcyBzaG91bGQgc3VtIHRvIDE=)

For a given tweet, determine which of the humanitarian labels are most relevant: 

The humanitarian labels and their descriptions are provided below: 

 "not\_humanitarian" - The tweet is not humanitarian in nature and does not fit into any other class. 

 "donation\_and\_volunteering" - The tweet relates to directing, accepting, or distributing donations or volunteer effort. 

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

The first element of the array should be a ranking: an integer from 1 to 16 representing the relative relevance of the humanitarian label compared to the others. The most relevant label should be ranked 1, and least relevant should be ranked 16. 

The second element should be a likelihood score: a floating point number between 0 and 1 representing the likelihood that the label applies to the tweet. 

The dictionary should have an entry for every humanitarian label, even if it is not relevant. The rankings for each label should be unique---that is, no two labels should have the same ranking, even if they are both not relevant: you must rank one higher than the other---and the likelihoods should sum to 1 

At the end of the prompt, we provided an example of the JSON format to address errors encountered during models’ JSON construction and subsequent validation using Langchain/Pydantic: [⬇](data:text/plain;base64,Rm9yIGV4YW1wbGUsIGEgY29ycmVjdGx5IGZvcm1hdHRlZCBhbnN3ZXIgd291bGQgYmU6IHsiaXNfaW5mb3JtYXRpdmUiOiBmYWxzZSwgImh1bWFuaXRhcmlhbl9sYWJlbCI6IHsibm90X2h1bWFuaXRhcmlhbiI6IFsxLCAwLjk1XSwgImRvbmF0aW9uX2FuZF92b2x1bnRlZXJpbmciOiBbMTYsIDAuMDA1XSwgInJlcXVlc3RzX29yX25lZWRzIjogWzE1LCAwLjAwNV0sICJzeW1wYXRoeV9hbmRfc3VwcG9ydCI6IFsyLCAwLjAxXSwgImluZnJhc3RydWN0dXJlX2FuZF91dGlsaXRpZXNfZGFtYWdlIjogWzE0LCAwLjAwMl0sICJhZmZlY3RlZF9pbmRpdmlkdWFsIjogWzEzLCAwLjAwMl0sICJjYXV0aW9uX2FuZF9hZHZpY2UiOiBbMTIsIDAuMDAyXSwgImluanVyZWRfb3JfZGVhZF9wZW9wbGUiOiBbMTEsIDAuMDAyXSwgInJlc3BvbnNlX2VmZm9ydHMiOiBbMTAsIDAuMDAyXSwgIm1pc3NpbmdfYW5kX2ZvdW5kX3Blb3BsZSI6IFs5LCAwLjAwMl0sICJkaXNwbGFjZWRfYW5kX2V2YWN1YXRpb25zIjogWzgsIDAuMDAyXSwgInBlcnNvbmFsX3VwZGF0ZXMiOiBbMywgMC4wMDhdLCAicGh5c2ljYWxfbGFuZHNsaWRlIjogWzcsIDAuMDAyXSwgImRpc2Vhc2VfcmVsYXRlZCI6IFs2LCAwLjAwMl0sICJ0ZXJyb3Jpc21fcmVsYXRlZCI6IFs1LCAwLjAwMl0sICJvdGhlcl9yZWxldmFudF9pbmZvcm1hdGlvbiI6IFs0LCAwLjAwN119)

For example, a correctly formatted answer would be: {"is\_informative": false, "humanitarian\_label": {"not\_humanitarian": \[1, 0.95\], "donation\_and\_volunteering": \[16, 0.005\], "requests\_or\_needs": \[15, 0.005\], "sympathy\_and\_support": \[2, 0.01\], "infrastructure\_and\_utilities\_damage": \[14, 0.002\], "affected\_individual": \[13, 0.002\], "caution\_and\_advice": \[12, 0.002\], "injured\_or\_dead\_people": \[11, 0.002\], "response\_efforts": \[10, 0.002\], "missing\_and\_found\_people": \[9, 0.002\], "displaced\_and\_evacuations": \[8, 0.002\], "personal\_updates": \[3, 0.008\], "physical\_landslide": \[7, 0.002\], "disease\_related": \[6, 0.002\], "terrorism\_related": \[5, 0.002\], "other\_relevant\_information": \[4, 0.007\]} 

TABLE I: Summary of F1 Scores for Informativeness Task

|          |       |       |
| -------- | ----- | ----- |
| Metric   |       |       |
| macro    | 0.800 | 0.796 |
| binary   | 0.836 | 0.836 |
| weighted | 0.808 | 0.805 |
| macro    | 0.802 | 0.799 |
| binary   | 0.855 | 0.848 |
| weighted | 0.814 | 0.810 |
| macro    | 0.819 | 0.801 |
| binary   | 0.860 | 0.837 |
| weighted | 0.828 | 0.809 |

TABLE II: F1 Scores for Informativeness Task by Each Dataset Across Commercial Models with and without Event Awareness

|          |       | CrisisLex6 | CrisisLex26 | CrisisNLP-cf | CrisisNLP-vol | AIDR  | DSM   | DRD   | ISCRAM2013 | SWDM13 |       |       |       |       |       |       |       |       |
| -------- | ----- | ---------- | ----------- | ------------ | ------------- | ----- | ----- | ----- | ---------- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Metric   |       |            |             |              |               |       |       |       |            |        |       |       |       |       |       |       |       |       |
| macro    | 0.840 | 0.825      | 0.603       | 0.617        | 0.759         | 0.739 | 0.667 | 0.687 | 0.746      | 0.739  | 0.728 | 0.716 | 0.740 | 0.743 | 0.498 | 0.481 | 0.555 | 0.557 |
| binary   | 0.812 | 0.796      | 0.916       | 0.932        | 0.936         | 0.921 | 0.640 | 0.655 | 0.707      | 0.693  | 0.617 | 0.599 | 0.839 | 0.841 | 0.892 | 0.871 | 0.765 | 0.777 |
| weighted | 0.842 | 0.827      | 0.886       | 0.902        | 0.896         | 0.881 | 0.676 | 0.697 | 0.752      | 0.746  | 0.753 | 0.743 | 0.782 | 0.785 | 0.869 | 0.849 | 0.689 | 0.697 |
| macro    | 0.860 | 0.834      | 0.590       | 0.600        | 0.744         | 0.738 | 0.656 | 0.689 | 0.718      | 0.725  | 0.787 | 0.774 | 0.687 | 0.695 | 0.560 | 0.536 | 0.588 | 0.598 |
| binary   | 0.842 | 0.814      | 0.943       | 0.940        | 0.941         | 0.930 | 0.639 | 0.658 | 0.706      | 0.699  | 0.731 | 0.708 | 0.849 | 0.851 | 0.951 | 0.927 | 0.848 | 0.846 |
| weighted | 0.861 | 0.835      | 0.909       | 0.908        | 0.897         | 0.887 | 0.662 | 0.699 | 0.719      | 0.729  | 0.800 | 0.789 | 0.756 | 0.761 | 0.929 | 0.905 | 0.753 | 0.756 |
| macro    | 0.879 | 0.822      | 0.622       | 0.610        | 0.786         | 0.748 | 0.670 | 0.726 | 0.738      | 0.741  | 0.744 | 0.732 | 0.753 | 0.735 | 0.524 | 0.456 | 0.591 | 0.543 |
| binary   | 0.862 | 0.789      | 0.940       | 0.922        | 0.950         | 0.923 | 0.645 | 0.682 | 0.702      | 0.701  | 0.644 | 0.624 | 0.855 | 0.847 | 0.926 | 0.836 | 0.822 | 0.743 |
| weighted | 0.880 | 0.824      | 0.910       | 0.892        | 0.913         | 0.884 | 0.678 | 0.740 | 0.743      | 0.747  | 0.766 | 0.756 | 0.796 | 0.782 | 0.903 | 0.815 | 0.738 | 0.670 |

TABLE III: F1 Scores for Informativeness Task on CrisisMMD with/without Event Awareness and Use of Images

|            |       |       |       |       |
| ---------- | ----- | ----- | ----- | ----- |
| Image Used |       | x     |       | x     |
| Metric     |       |       |       |       |
| macro      | 0.760 | 0.712 | 0.743 | 0.731 |
| binary     | 0.877 | 0.869 | 0.871 | 0.871 |
| weighted   | 0.809 | 0.776 | 0.795 | 0.788 |
| macro      | 0.701 | 0.703 | 0.745 | 0.729 |
| binary     | 0.873 | 0.872 | 0.870 | 0.869 |
| weighted   | 0.771 | 0.772 | 0.796 | 0.786 |
| macro      | 0.733 | 0.699 | 0.761 | 0.728 |
| binary     | 0.876 | 0.870 | 0.868 | 0.870 |
| weighted   | 0.791 | 0.769 | 0.805 | 0.786 |

TABLE IV: Macro and Weighted F1 Scores for Humanitarian Classification by each Dataset

|       | Dataset |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
| ----- | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|       | x       |       | x     |       | x     |       | x     |       | x     |       | x     |       | x     |       | x     |       | x     |       |       |
|       | x       |       | x     |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
|       |         |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
| 0.537 | 0.508   | 0.554 | 0.493 | 0.836 | 0.832 | 0.461 | 0.492 | 0.564 | 0.553 | 0.263 | 0.250 | 0.345 | 0.333 | 0.700 | 0.680 | 0.530 | 0.547 | 0.486 | 0.540 |
| 0.625 | 0.557   | 0.638 | 0.558 | 0.837 | 0.834 | 0.467 | 0.497 | 0.583 | 0.570 | 0.546 | 0.558 | 0.623 | 0.612 | 0.837 | 0.823 | 0.653 | 0.618 | 0.590 | 0.666 |
| 0.500 | 0.466   | 0.533 | 0.472 | 0.869 | 0.838 | 0.439 | 0.445 | 0.546 | 0.540 | 0.193 | 0.219 | 0.363 | 0.346 | 0.488 | 0.501 | 0.541 | 0.526 | 0.440 | 0.440 |
| 0.531 | 0.510   | 0.565 | 0.527 | 0.869 | 0.839 | 0.438 | 0.452 | 0.565 | 0.549 | 0.540 | 0.598 | 0.637 | 0.626 | 0.617 | 0.633 | 0.610 | 0.581 | 0.549 | 0.549 |
| 0.513 | 0.485   | 0.531 | 0.504 | 0.882 | 0.822 | 0.448 | 0.464 | 0.603 | 0.583 | 0.254 | 0.262 | 0.341 | 0.300 | 0.627 | 0.611 | 0.532 | 0.538 | 0.505 | 0.433 |
| 0.568 | 0.545   | 0.586 | 0.563 | 0.882 | 0.824 | 0.451 | 0.468 | 0.626 | 0.585 | 0.596 | 0.644 | 0.625 | 0.605 | 0.773 | 0.754 | 0.645 | 0.612 | 0.624 | 0.537 |

### II-D Evaluation

For both the informativeness and humanitarian label tasks, we calculate F1 scores to facilitate comparison with existing evaluations of datasets within CrisisBench. For the informativeness classification, we calculate macro (unweighted), weighted, and binary (only the positive class) F1 scores; for the humanitarian classification, we calculate weighted and macro (unweighted) class-averaged F1 scores. Given that some datasets in CrisisBench may cover only a subset of the 16 labels specified in the prompt, we maintain consistency in the prompt by including all 16 labels in our evaluation framework. However, the performance assessment of each dataset is based exclusively on the rankings of the labels present within that dataset. Class-specific precision for class ii is computed as Pi\=TPiTPi+FPi\\text{P}\_{i}=\\frac{\\text{TP}\_{i}}{\\text{TP}\_{i}+\\text{FP}\_{i}}, where TPi,FPi\\text{TP}\_{i},\\text{FP}\_{i} stand for the count of true positives and false positives for class ii, respectively. Class-specific recall is defined as Ri\=TPiTPi+FNi\\text{R}\_{i}=\\frac{\\text{TP}\_{i}}{\\text{TP}\_{i}+\\text{FN}\_{i}}, where FNi\\text{FN}\_{i} is the count of false negatives for class ii. Class-specific F1 is defined as

F​1i\=2​Pi​RiPi+RiF1\_{i}=\\frac{2\\text{P}\_{i}\\text{R}\_{i}}{\\text{P}\_{i}+\\text{R}\_{i}}

In the case where there is only one class, the class-specific F1 is equivalent to the binary F1.

The weighted class-average F1 is defined as

F​1weighted\=∑iniN​F1iF1\_{\\text{weighted}}=\\sum\_{i}\\frac{n\_{i}}{N}\\text{F1}\_{i}

where nin\_{i} is the number of instances where the true class is ii, and NN is the total number of instances. The macro class-average F1 is defined as

F​1macro\=∑i1m​F1iF1\_{\\text{macro}}=\\sum\_{i}\\frac{1}{m}\\text{F1}\_{i}

where mm is the number of classes.

## III Results

In our experiment, the classification tasks for the crisis tweets on informativeness and type of humanitarian label are combined into one prompt. The model was prompted to: 1) determine if the social media post is informative in a humanitarian context, and 2) rank and assign probabilities to 16 potential humanitarian labels in how it fits the post. We report macro, weighted, and binary F1 scores for the informativeness task. For the humanitarian task, we use macro and weighted F1 scores.

Results for the informativeness task across the three models for all tested crisis tweets are in Table [I](#S2.T1 "TABLE I ‣ II-C Prompt Structure ‣ II Methodology ‣ Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models*"). Weighted, binary and macro F1 scores are reported with and without event awareness. “Event-aware” indicates whether the name of a disaster contemporaneous with the tweet is included in the prompt. For each version of the prompt, the highest F1 scores are bolded. Models performed slightly better without event-awareness. OpenAI’s GPT-4o performed the best without event-awareness at a macro F1 score of 0.819, a binary F1 score at 0.860, and weighted F1 score at 0.828.

The results by dataset are in two tables, Table [II](#S2.T2 "TABLE II ‣ II-C Prompt Structure ‣ II Methodology ‣ Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models*") contains all datasets not including CrisisMMD, and Table [III](#S2.T3 "TABLE III ‣ II-C Prompt Structure ‣ II Methodology ‣ Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models*") contains the CrisisMMD results. These results are also separated by whether extra information (event and/or image) was included in the prompt. Across most datasets, OpenAI GPT-4o outperformed the other evaluated LLMs in informativeness classifications.

Based on reported metrics, the best LLMs compares moderately lower to existing benchmarks on the consolidated dataset \[[1](#bib.bibx1 "")\], with a weighted F1 of 0.828 (LLM: GPT-4o) vs. 0.883 (fine-tuned RoBERTa). When looking at specific datasets where benchmarks were available, the LLMs also underperform, sometimes by a large margin: on CrisisMMD \[[10](#bib.bibx10 "")\], (weighted F1 of .638 vs .842), on CrisisLexT26 \[[11](#bib.bibx11 "")\] (macro F1 of .492 vs .848), and CrisisLexT6 \[[11](#bib.bibx11 "")\] (macro F1 of .882 vs .947).

As a note, we contend that binary F1 (class-wise F1 for single class) as the most appropriate metric for this task. Macro and weighted F1 are most appropriate metrics for multi-class classification, in which F1 scores for multiple classes are condensed into a single metric. For this binary classification problem, the “informative” class serves as a foreground and the “not informative” class as a background; binary F1 appropriately privileges the foreground class as being the one which is useful to distinguish. The use of macro and weighted F1 are significantly impacted by the number of negative examples in the dataset, which is not desirable when the task is trying to find positive instances in a haystack of negative examples. The LLMs obtain substantially better binary F1 scores than macro F1. Unfortunately, binary F1 is not reported for the informativeness task in any literature we found, making direct comparison on the metric difficult.

For the informativeness task, the inclusion of images (in CrisisMMD) and event context (all datasets) did not significantly affect F1 scores, with changes dependent on the dataset. For humanitarian classification, including event context slightly improved scores, while including images did not.

For the humanitarian classification task, we compute both weighted and macro F1 scores, and treat the highest-ranked label within the subset of labels from the constituent dataset as the predicted label for evaluation. These results are reported in Table [IV](#S2.T4 "TABLE IV ‣ II-C Prompt Structure ‣ II Methodology ‣ Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models*"). The highest F1 score for both macro and weighted for each dataset are bolded for each variation of the prompt. Anthropic Claude-3-5 Sonnet generally performed better than other tested LLMs in both weighted and macro f1-scores. The F1 scores of the LLMs are broadly low, but inconsistencies in the CrisisBench dataset are likely substantially responsible for this fact, which we will explore further in the discussion. We also note that this evaluation of humanitarian labels is preliminary, as it only includes the subset of tweets in the informativeness task that also happened to have humanitarian label annotations.

TABLE V: Accuracy rates of two classes for Anthropic’s Claude-3.5 Sonnet and manually assigned labels compared to ground-truth

| Label |      |
| ----- | ---- |
| 13.3  | 76.0 |
| 14.7  | 48.0 |
| 56.0  | 74.7 |
| 100   | 29.3 |
| 100   | 61.3 |
| 100   | 65.3 |

## IV Discussion

Overall, LLMs perform reasonably well on the informativeness task, achieving zero-shot performance on the consolidated dataset within 6% of that of pretrained classifiers in \[[1](#bib.bibx1 "")\].

However, performance was substantially worse on the multi-class humanitarian label task. The LLMs broadly underperformed the models trained in CrisisTransformers \[[20](#bib.bibx20 "")\]. We note several limitations and challenges that may have contributed to this. The task performance of the LLMs tested may have been limited by a number of factors, including the absence of optimizations like prompt engineering or fine-tuning. Further, fine-tuned models may perform better because they were fine-tuned on each dataset individually and were able to fit the base rates at which various classes occur. More fundamentally, however, we note that methods in the construction of CrisisBench dataset had a substantial impact on the multi-class task performance for the LLMs.

Whereas the zero-shot classification technique for LLMs uses natural language prompting to describe the criteria for each class based on the name and description of the label, traditional models are trained on the training dataset. This raises a potential issue if there is misalignment between the labeled examples and the semantic understanding of the label. We identified methods related to the construction of CrisisBench which may contribute to such misalignment.

CrisisBench draws from a number of datasets, aggregating labels with potentially different definitions into single classes. For example, the infrastructure\_and\_utilities\_damage class is defined in CrisisNLP-volunteers to be the destruction of houses, buildings, or roads, or the interruption of utilities, but CrisisNLP-CF defines it to include restoration of utilities as well \[[1](#bib.bibx1 "")\]. Furthermore, the classes of constituent datasets are sometimes defined such that they cannot be mapped onto a single CrisisBench label. The CrisisLexT26 affected\_individual class contributes all of the affected\_individual examples in CrisisBench. But in CrisisLexT26, this class is defined to include personal updates, which is a separate CrisisBench class. The only way for an LLM to correctly categorize a personal update is to correctly guess whether it came from CrisisLexT26 or not. As CrisisLexT26 is one of the largest datasets labeled for humanitarian class, it is unsurprising that personal\_update and affected\_individual are the two lowest-performing classes for the evaluated LLMs.

To better understand the impact of ambiguous label mappings on performance, we performed manual binary annotation on two of the lowest-performing classes, affected\_individual and caution\_and\_advice. We examined 75 randomly sampled tweets which were assigned to the two classes (for a total of 150 tweets) by either the ground truth label or the maximum likelihood estimate of the LLM, without knowledge of the ground-truth or predicted label of any particular tweet. We manually performed binary classification on each tweet as either matching or not matching the description of its reference class given in the prompt (for example, we might look at a tweet with the knowledge that at least one of the ground truth label or the predicted label was affected\_individual, and assign a binary label based on whether we believed the tweet matched the definition we gave for that class). The results of this experiment are in Table [V](#S3.T5 "TABLE V ‣ III Results ‣ Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models*"). This experiment suggests that semantic differences in the labels, which would not have affected models trained on the training data \[[20](#bib.bibx20 "")\], had a substantial impact on the performance of the LLMs. It also showed generally low agreement between our manual labels and the ground truth, with our manual labels matching the LLM labels more often than the ground truth on the affected\_individuals class. While the LLMs performed better on our manual labels than on the ground truth in general, the difference in performance is much larger in the cases where agreement between our labels and the ground truth was relatively weaker. To verify that we were not observing a regression to the mean by injecting noise into the labels, we also analyzed a further 75 tweets from a high-performing class, disease\_related. Even on this higher-performing class, accuracy is substantially better when comparing against manually labeled examples as opposed to the ground truth labels. We also observe that accuracy as measured against the ground truth is higher on labels where the ground truth and manual labeling have higher agreement.

The significant level of variation in F1 scores between datasets across both classification tasks merits further investigation. For example, the binary F1 Score for OpenAI’s GPT-4o of 0.950 suggests strong performance for the positive label on the CrisisNLP-cf dataset (labeled by paid crowd workers), and much worse performance with 0.682 on the CrisisNLP-vol dataset (labeled by unpaid volunteers). One possible explanation for this is that data quality varies substantially between constituent datasets.

We also note a couple additional challenges during implementation that practitioners and researchers using LLMs should be aware of. Occasionally, LLMs would refuse to classify a tweet due to objectionable subject material (pornographic content, hate speech, and foul language). In addition, LLMs sometimes struggled to output correctly formatted JSON—this was usually able to be resolved by resubmitting the prompt, but on rare occasion, the request failed past our patience threshold and had to be omitted. There were also a small number of tweets that were misconstrued as part of the prompt, leading the LLM to respond that it did not detect a tweet to classify. A stronger distinction between the prompt and tweet to be classified, perhaps using a special token, would help ameliorate this. The total number of refused/omitted tweets were small (on the order of 10 per model), and thus should not affect the evaluation scores.

## V Conclusion

In this paper, we present the performance of commercial large language models on zero-shot classification for two tasks on short social media posts on CrisisBench. We find that overall performance on the binary informativeness task is strong, even relative to models fine-tuned on the evaluation datasets. Incorporating extra information in the form of possible event context and images did not substantially impact the model’s performance on the task.

For the second task, humanitarian classification, an ambiguous multi-class task performance rapidly declined, emphasizing the need for careful deployment of these tools to the humanitarian space. Based on small-scale experiments with manual labeling, we attribute most of the LLMs’ declining performance to semantic ambiguity in social media posts and their labels rather than a latent inability to parse and classify natural language.

In future work, we plan to include open-source models in our classification assessments. We will substantially reduce problems associated with the dataset aggregation performed by CrisisBench by changing the prompt and label definitions based on source dataset. This will also provide an avenue for further analysis of each dataset’s quality. Further, we plan to analyze the results by language to better understand the multi-lingual components of the LLMs in relation to humanitarian classification tasks. Another avenue of future research is to assess the impact of prompt engineering more broadly. For example, we prompt for both classification tasks in the same prompt, but it would be of interest to look into the extent to which the dual classification task in one prompt impacts model performance.

## References

*   \[1\] Firoj Alam, Hassan Sajjad, Muhammad Imran and Ferda Ofli “CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing” In *Proceedings of the International AAAI Conference on Web and Social Media* 15.1, ICWSM ’21, 2021, pp. 923–932 URL: [https://ojs.aaai.org/index.php/ICWSM/article/view/18115](https://ojs.aaai.org/index.php/ICWSM/article/view/18115 "")
*   \[2\] Tom. Brown et al. “Language Models are Few-Shot Learners”, 2020 arXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165 "")
*   \[3\] OpenAI et al. “GPT-4 Technical Report”, 2024 arXiv: [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774 "")
*   \[4\] Muhammad Imran, Prasenjit Mitra and Carlos Castillo “Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages” In *Proc. of the LREC, 2016* Paris, France: ELRA, 2016
*   \[5\] Firoj Alam, Ferda Ofli and Muhammad Imran “CrisisMMD: Multimodal Twitter Datasets from Natural Disasters” In *Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)*, 2018
*   \[6\] Alexandra Olteanu, Sarah Vieweg and Carlos Castillo “What to expect when the unexpected happens: Social media communications across crises” In *Proc. of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing*, 2015, pp. 994–1009 ACM
*   \[7\] Alexandra Olteanu, Carlos Castillo, Fernando Diaz and Sarah Vieweg “CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises.” In *Proc. of the 8th ICWSM, 2014* AAAI press, 2014
*   \[8\] Muhammad Imran et al. “Practical extraction of disaster-relevant information from social media” In *Proc. of the 22nd WWW*, 2013, pp. 1021–1024 ACM
*   \[9\] Muhammad Imran et al. “Extracting information nuggets from disaster-related messages in social media” In *Proc. of the 12th ISCRAM*, 2013
*   \[10\] Ferda Ofli, Firoj Alam and Muhammad Imran “Analysis of social media data using multimodal deep learning for disaster response” In *The 17th International Conference on Information Systems for Crisis Response and Management (ISCRAM 2020)*, 2020
*   \[11\] Hongmin Li, Doina Caragea and Cornelia Caragea “Combining self-training with deep learning for disaster tweet classification” In *The 18th International Conference on Information Systems for Crisis Response and Management (ISCRAM 2021)*, 2021
*   \[12\] Ankur Mondal, Athishay Kesan, Andrea Rodrigues and Jossy George “An Efficient Multi-Modal Classification Approach for Disaster-related Tweets” In *2022 IEEE International Conference on Distributed Computing and Electrical Circuits and Electronics (ICDCECE)*, 2022, pp. 1–4 IEEE DOI: [10.1109/ICDCECE53908.2022.9792951](https://dx.doi.org/10.1109/ICDCECE53908.2022.9792951 "")
*   \[13\] Siva Dasari, Srinivas Gorla and Prasad PVGD “A stacking ensemble approach for identification of informative tweets on twitter data” In *International Journal of Information Technology* 15.5 Springer, 2023, pp. 2651–2662
*   \[14\] K Asinthara, Meghna Jayan and Lija Jacob “Categorizing disaster tweets using learning based models for emergency crisis management” In *2023 9th International conference on advanced computing and communication systems (ICACCS)* 1, 2023, pp. 1133–1138 IEEE DOI: [10.1109/ICACCS57279.2023.10113105](https://dx.doi.org/10.1109/ICACCS57279.2023.10113105 "")
*   \[15\] Dasari Krishna, Gorla Srinivas and PVGD Reddy “A Deep Parallel Hybrid Fusion Model for disaster tweet classification on Twitter data” In *Decision Analytics Journal* 11 Elsevier, 2024, pp. 100453 DOI: [10.1016/j.dajour.2024.100453](https://dx.doi.org/10.1016/j.dajour.2024.100453 "")
*   \[16\] Jiankai Li, Yunhong Wang and Weixin Li “MHRN: A Multimodal Hierarchical Reasoning Network for Topic Detection” In *IEEE Transactions on Multimedia* IEEE, 2024 DOI: [10.1109/TMM.2024.3358696](https://dx.doi.org/10.1109/TMM.2024.3358696 "")
*   \[17\] Matthieu François and Paul Gay “Active Learning with few shot learning for crisis management” In *Proceedings of the 20th International Conference on Content-based Multimedia Indexing*, 2023, pp. 233–237 DOI: [10.1145/3617233](https://dx.doi.org/10.1145/3617233 "")
*   \[18\] Abdelghani Dahou et al. “A social media event detection framework based on transformers and swarm optimization for public notification of crises and emergency management” In *Technological Forecasting and Social Change* 192 Elsevier, 2023, pp. 122546 DOI: [10.1016/j.techfore.2023.122546](https://dx.doi.org/10.1016/j.techfore.2023.122546 "")
*   \[19\] Muhammad Malik, Muhammad Younas, Mona Jamjoom and Dmitry Ignatov “Categorization of tweets for damages: infrastructure and human damage assessment using fine-tuned BERT model” In *PeerJ Computer Science* 10 PeerJ Inc., 2024, pp. e1859 DOI: [10.7717/peerj-cs.1859](https://dx.doi.org/10.7717/peerj-cs.1859 "")
*   \[20\] Rabindra Lamsal, Maria Read and Shanika Karunasekera “CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts” In *Knowledge-Based Systems* 296 Elsevier BV, 2024, pp. 111916 DOI: [10.1016/j.knosys.2024.111916](https://dx.doi.org/10.1016/j.knosys.2024.111916 "")
*   \[21\] Yongqin Xian, Christoph Lampert, Bernt Schiele and Zeynep Akata “Zero-Shot Learning: A Comprehensive Evaluation of the Good, the Bad and the Ugly” In *IEEE transactions on pattern analysis and machine intelligence* 41.9 IEEE, 2018, pp. 2251–2265 DOI: [10.1109/TPAMI.2018.2857768](https://dx.doi.org/10.1109/TPAMI.2018.2857768 "")
*   \[22\] Takeshi Kojima et al. “Large Language Models are Zero-Shot Reasoners” In *Advances in Neural Information Processing Systems* 35 Curran Associates, Inc., 2022, pp. 22199–22213
*   \[23\] Guozheng Li, Peng Wang and Wenjun Ke “Revisiting Large Language Models as Zero-shot Relation Extractors” In *Findings of the Association for Computational Linguistics: EMNLP 2023* Singapore: Association for Computational Linguistics, 2023, pp. 6877–6892 URL: [https://aclanthology.org/2023.findings-emnlp.459](https://aclanthology.org/2023.findings-emnlp.459 "")
*   \[24\] Xuansheng Wu et al. “Matching Exemplar as Next Sentence Prediction (MeNSP): Zero-Shot Prompt Learning for Automatic Scoring in Science Education” In *International Conference on Artificial Intelligence in Education*, 2023, pp. 401–413 Springer DOI: [10.1007/978-3-031-36272-9˙33](https://dx.doi.org/10.1007/978-3-031-36272-9_33 "")
*   \[25\] Han Yu, Peikun Guo and Akane Sano “Zero-Shot ECG Diagnosis with Large Language Models and Retrieval-Augmented Generation” In *Proceedings of the 3rd Machine Learning for Health Symposium* 225, Proceedings of Machine Learning Research PMLR, 2023, pp. 650–663
*   \[26\] Lianmin Zheng et al. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena” In *Advances in Neural Information Processing Systems* 36 Curran Associates, Inc., 2023, pp. 46595–46623
*   \[27\] Ehsan Latif and Xiaoming Zhai “Fine-tuning ChatGPT for automatic scoring” In *Computers and Education: Artificial Intelligence* 6 Elsevier, 2024, pp. 100210 DOI: [10.1016/j.caeai.2024.100210](https://dx.doi.org/10.1016/j.caeai.2024.100210 "")
*   \[28\] Madhumita Sushil et al. “A comparative study of large language model-based zero-shot inference and task-specific supervised classification of breast cancer pathology reports” In *Journal of the American Medical Informatics Association* Oxford University Press, 2024, pp. ocae146 DOI: [10.1093/jamia/ocae146](https://dx.doi.org/10.1093/jamia/ocae146 "")
*   \[29\] OpenAI “Hello GPT-4o”, 2024 URL: [https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/ "")
*   \[30\] Google “Gemini breaks new ground with a faster model, longer context, AI agents and more”, 2024 URL: [https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/](https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/ "")
*   \[31\] Anthropic “Claude 3.5 Sonnet”, 2024 URL: [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet "")

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")