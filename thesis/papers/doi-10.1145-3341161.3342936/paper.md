# Low-supervision urgency detection and transfer in short crisis messages

Mayank Kejriwal Affiliation: Information Sciences Institute  
University of Southern California  Marina del Rey, CA  
kejriwal@isi.edu   Peilin Zhou Affiliation: Information Sciences Institute  
University of Southern California  Marina del Rey, CA  
zpeilin@isi.edu

###### Abstract

Humanitarian disasters have been on the rise in recent years due to the effects of climate change and socio-political situations such as the refugee crisis. Technology can be used to best mobilize resources such as food and water in the event of a natural disaster, by semi-automatically flagging tweets and short messages as indicating an urgent need. The problem is challenging not just because of the sparseness of data in the immediate aftermath of a disaster, but because of the varying characteristics of disasters in developing countries (making it difficult to train just one system) and the noise and quirks in social media. In this paper, we present a robust, low-supervision social media urgency system that adapts to arbitrary crises by leveraging both labeled and unlabeled data in an ensemble setting. The system is also able to adapt to new crises where an unlabeled background corpus may not be available yet by utilizing a simple and effective transfer learning methodology. Experimentally, our transfer learning and low-supervision approaches are found to outperform viable baselines with high significance on myriad disaster datasets.

###### Index Terms: 

Urgency detection, social media, machine learning, Twitter, crisis informatics

## I Introduction

TABLE I: Urgent and non-urgent examples from three real-world datasets that we describe further in Section [V](#S5 "V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages").

| Dataset   | Urgent Sentences                                                                                              | Non-urgent Sentences                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Nepal     | Anyone who speaks about Balochistan in provinces other than Punjab either ends up dead or missing             | Today’s earthquake data for Nepal                                                                                                      |
|           | EMERGENCY: 4 locals trapped in this rubble INSIDE PALTANGHAR                                                  | Wow. ndtv just showed the same Philippines earthquake picture and said it’s from Kathmandu on TV.                                      |
| Macedonia | Some people are trapped in the marketplace need help.                                                         | the streets are filled with fecal and water no water                                                                                   |
|           | We re trapped at the national commissioner s house the first floor s loaded with the kids have begun scared.  | I’m about to walk with bicite but the rain that fell before s been blocking the roads that the channels are from the time of the rock. |
| Kerala    | 8 people no food survivin on dry cornflakes for the last 3 days east kadungalloor two families.               | I’m from kerala and the situation here is very very bad, thousands have lost.                                                          |
|           | At least 324 people have been killed in flooding and landslides in the indian state of while more than 200000 | there has been floods in kerala india, more than 70 have lost their lives may ”Make it easy for all”.                                  |

The United Nations Office for the Coordination of Human Affairs (OCHA) reported11 1 [https://www.unocha.org/sites/unocha/files/WHDT2018\_web\_final\_spread.pdf](https://www.unocha.org/sites/unocha/files/WHDT2018_web_final_spread.pdf "") that in 2018, more than 141 million people were in need of humanitarian assistance, with over 9 billion dollars of unmet requirements. Using technology to address this shortfall by assisting aid agencies and first responders mobilize and send resources where they are needed the most is an important problem with the potential for widespread long-lasting social impact \[[1](#bib.bib1 "")\], \[[2](#bib.bib2 "")\].

To achieve this goal, the problem of semi-automatic *urgency detection* needs to be solved, especially on short message streams like social media that support real-time news feeds and micro-updates from citizens on the ground. Put intuitively, the urgency detection problem can be framed in terms of probabilistic binary classification, a common machine learning paradigm involving other related tasks like sentiment analysis \[[3](#bib.bib3 "")\]. Although urgency detection has some similarity with sentiment analysis, the core problem is different, since the goal is to flag messages that *express urgency*, which is almost always a negative or panic-ridden emotion. However, it can be difficult to distinguish urgency-related tweets from just negative tweets. We provide an illustrative set of real-world examples22 2 A description of the datasets will be provided in Section [V-A](#S5.SS1 "V-A Data ‣ V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages"). in Table I.

In this paper, we present practical approaches for crisis-specific minimally supervised urgency detection on short message streams such as Twitter. The presented approaches cover two scenarios that often emerge in the real world. In the first scenario, a small amount (a few hundred messages) of training data labeled as urgent or non-urgent is available, along with a copious ‘unlabeled’ background corpus. In the second scenario, similar data is available for a ‘source’ domain but not for the target domain (expressing a ‘new crisis’) for which the urgency detection needs to be deployed. In other words, as messages are streaming in for this new domain, investigators label a few samples, but cannot rely on the availability of a background corpus since urgency needs to be tagged in real time before the crisis has fully subsided. To accomplish this challenging goal, our approach relies on a simple and robust transfer learning methodology \[[4](#bib.bib4 "")\]. Experimental results on three real-world datasets and several performance metrics validate our methods. To the best of our knowledge, this is the first such paper investigating the problem of urgency detection in social media, both algorithmically and empirically, for arbitrary disasters in low-supervision and transfer learning settings.

The rest of this paper is structured as follows. Section [II](#S2 "II Related Work ‣ Low-supervision urgency detection and transfer in short crisis messages") describes some related work, Section [III](#S3 "III Research Questions ‣ Low-supervision urgency detection and transfer in short crisis messages") specifies our two research questions, and Section [IV](#S4 "IV Approach ‣ Low-supervision urgency detection and transfer in short crisis messages") describes our approaches in support of answering those questions. Section [V](#S5 "V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages") covers the experiments, and Section [VI](#S6 "VI Conclusion and Future Work ‣ Low-supervision urgency detection and transfer in short crisis messages") concludes the paper.

## II Related Work

*Crisis informatics* is emerging as an important field for both data scientists and policy analysts. A good introduction to the field was provided in a recent Science policy forum article \[[1](#bib.bib1 "")\]. The field draws on interdisciplinary strands of research, especially with respect to collecting, processing and analyzing real-world data. Particularly, social media platforms like Twitter have emerged as important channels (‘social sensors’ \[[2](#bib.bib2 "")\]) for *situational awareness* in support of crisis informatics. Although situational awareness is a broad notion extending beyond crisis informatics (e.g., military situational awareness), urgency detection is a special kind of situational awareness that tends to arise mainly in the crisis domain. A direct application is to help first responders and aid agencies assess needs in crisis-stricken areas and mobilize resources effectively (i.e. where needs are most urgent). NLP methods have been widely used in extracting situational awareness from Twitter e.g., see the work by Verma et al. \[[5](#bib.bib5 "")\]. Another important line of work is in analyzing events other than natural disasters (such as mass convergence and disruption events), but still relevant to crisis informatics. For example Stabird et al. presented a collaborative filtering system for identifying on-the-ground ‘Twitterers’ during mass disruptions \[[6](#bib.bib6 "")\]. Similar techniques could be employed to supplement the work in this paper.

More generally, projects like CrisisLex, Crisis Computing33 3 [https://crisiscomputing.qcri.org/](https://crisiscomputing.qcri.org/ "") and EPIC (Empowering the Public with Information in Crisis) have emerged as major efforts in the crisis informatics space due to two reasons: first, the abundance and fine granularity of social media data implies that mining such data during crises can lead to robust, real-time responses; second, the recognition that any technology that is thus developed must also address the inherent challenges (including problems of noise, scale and irrelevance) in working with such datasets. CrisisLex provides a repository of crisis-related social media data and tools, including collections of crisis data and lexicons of crisis terms \[[7](#bib.bib7 "")\]. It also includes tools to help users create their own collections and lexicons. In contrast, Project EPIC, launched in 2009 and supported by a US National Science Foundation grant, is a multi-disciplinary effort involving several universities and languages with the goal of utilizing behavioral and technical knowledge of computer mediated communication for better crisis study and emergency response. Since its founding, Project EPIC has led to several advances in the crisis informatics space; see for example \[[8](#bib.bib8 ""), [9](#bib.bib9 ""), [10](#bib.bib10 ""), [11](#bib.bib11 ""), [12](#bib.bib12 "")\]. The work presented in this article is intended to be compatible with these efforts.

Other lines of work relevant to this paper involve minimally supervised machine learning, representation learning and transfer learning. Concerning minimally supervised machine learning (ML), in general, ML techniques where there are few, and in the case of zero-shot learning \[[13](#bib.bib13 ""), [14](#bib.bib14 "")\], no observed instances for a label has been a popular research agenda for many years \[[15](#bib.bib15 ""), [16](#bib.bib16 "")\]. In addition to weak supervision approaches \[[16](#bib.bib16 "")\], both semi-supervised and active learning have also been studied in great depth, with surveys provided by \[[17](#bib.bib17 ""), [18](#bib.bib18 "")\]. However, to the best of our knowledge, a successful systems-level conjunction of various minimally supervised ML techniques has not been achieved for the task of short-text urgency detection. Such as empirical assessment is an important goal of this paper.

Due to the current renaissance of neural networks \[[19](#bib.bib19 "")\], *embedding* and *representation learning* methods have become more popular due to the advent of fast and effective models like skip-gram. Recent work has used such embeddings in numerous NLP and graph-theoretic applications \[[20](#bib.bib20 "")\], including information extraction \[[21](#bib.bib21 "")\], named entity recognition \[[22](#bib.bib22 "")\] and entity linking \[[23](#bib.bib23 "")\]. The most well-known example is word2vec (for words) \[[24](#bib.bib24 "")\], followed by similar models like paragraph2vec (for multi-word text) and fasttext \[[25](#bib.bib25 ""), [26](#bib.bib26 "")\], the last two being most relevant for the work in this paper. For a recent evaluation study on representation learning for text, including potential problems, we refer the reader to \[[27](#bib.bib27 "")\]. Finally, transfer learning is a central agenda in this paper; an excellent survey of dominant techniques may be found in \[[4](#bib.bib4 "")\]. More recent work on domain adaptation may be found in \[[28](#bib.bib28 "")\], with the work in \[[29](#bib.bib29 "")\] applied specifically to the disaster response problem. Pedrood and Purohit \[[29](#bib.bib29 "")\] also applied transfer learning to the problem of mining help intent on Twitter. Other relevant work in crisis informatics, both in terms of defining ‘actionable information’ problems like urgency and need mining, as well as providing multimodal Twitter datasets from natural disasters, may be found in \[[30](#bib.bib30 "")\], \[[31](#bib.bib31 "")\] and \[[32](#bib.bib32 "")\]. An alternate way of looking at the problem is as an ‘event detection’ problem e.g., in \[[33](#bib.bib33 "")\] Zheng et al. study semi-supervised event-related tweet identification which also tries to identify the urgent tweets related to earthquakes and floods. These works are complementary to the minimally supervised, low-resource setting in this paper.

## III Research Questions

We briefly enumerate below the research questions under consideration in this paper. While the first question captures the classical low-supervision setting, the second question introduces an element of transfer learning.

1.  1.

```
Low-supervision Training for Urgency Detection: How do we build an urgency detection system for a specific crisis when given as training input both a small number of manually labeled tweets, and a large number of unlabeled tweets (background corpus), for that crisis?
```
2.  2.

```
Low-supervision Transfer Learning for Urgency Detection: How do we build an urgency detection system for a specific crisis when given as training input a small number of manually labeled tweets for that crisis, as well as ‘auxiliary’ training input of (a small number of) manually labeled tweets and unlabeled background tweets from a *different* crisis?
```
Unlike the first scenario, the second scenario applies to a very short period (hours, or even minutes) after the crisis has struck; this is why a background corpus is not available (yet) for that crisis. Instead, only a few manually labeled messages that have been acquired till that point are available.

## IV Approach

### IV-A Low-supervision urgency detection

The approach for addressing the first research question is schematized in Figure 1. The first step in the workflow involves data preprocessing of the corpus. We follow a standard set of preprocessing steps. First, we apply a tokenizer to split the sentences into lists of words and delete words with special prefixes (including @ and RT, which are particularly prevalent in Twitter), and special suffixes. We also remove non-alphanumeric characters and convert the entire sentence to lowercase. Next, similar to traditional machine learning pipelines, we extract a set of manual features for expressing prior human knowledge about urgency detection. Our manual features are thus called because they are primarily keyword-based and binary, with keywords selected based on data exploration and domain knowledge. We consider ten such keywords, namely *hit, help, kill, injure, strand, miss, urgent, die, need, food*. If any of these keywords are present44 4 Possibly as stems, for example, the word ‘helping’ would trigger the ‘help’ keyword feature, which would be consequently set to 1., the corresponding feature is set to 1. Note that these keywords are associated with situations that are generally urgent, like people who have been attacked or affected by a crisis and need urgent help, but some are noisier than others55 5 For example ‘help’ could be associated with a more trivial situation like someone needing help with their dog.. Additionally, we also utilize an eleventh feature that checks to see if any numeric digits are present in the dataset. The rationale behind this feature is that, in more urgent tweets, numbers are often present e.g., ‘15 climbers are currently trapped on Everest due to the avalanche’.

In the experimental section, we show that the manual features are not adequate for addressing low-supervision urgency detection. Besides, it is prudent to utilize the large number of unlabeled tweets (background corpus) if it serves a useful purpose in improving performance. To that end, we train a skip-gram based word embedding model based on the ‘bag of tricks’ model released by researchers from Facebook in a package called *fastText* \[[26](#bib.bib26 "")\]. The reason behind using fastText, as opposed to alternate word embedding models like GloVe and word2vec \[[24](#bib.bib24 "")\], is several-fold. First, fastText is very fast and easy to execute, and is well-maintained. Second, preliminary analyses showed that it does quite well on social media tasks and because of the bag of tricks methodology (that uses character and sub-word embeddings to gracefully deal with OOVs66 6 Out of Vocabulary words. and misspellings), it is able to generalize much better. Finally, fastText’s APIs include a way to get sentence embeddings directly after training the word embedding model. By training fastText on the background corpus, we are able to train a robust embedding model. In both the training and test phase, we use this model to get feature vectors for our messages besides the 11-dimensional manual feature vector described earlier.

However, given that the background corpus might not be as extensive or representative as a ‘general’ corpus like Wikipedia, we try to smooth the feature space by also using a pre-trained embedding model trained over the English Wikipedia corpus and publicly available77 7 [https://fasttext.cc/docs/en/pretrained-vectors.html](https://fasttext.cc/docs/en/pretrained-vectors.html ""). The vectors obtained from this model have 300 dimensions and were trained using skip gram with default parameters.

As Figure 1 illustrates, we use all of these feature sets to build an ensemble by combining local embedding features, manual features and Wikipedia pre-trained word embedding features. The final score of the ensemble model is achieved by weighting the scores of the three Linear Regression models (one for each feature-set), with weights adding to 1. The weights are set using a held-out validation set.

When the urgency of a new ‘test’ message needs to be determined, we preprocess the message, extract all three feature-sets88 8 In the case of the two trained embedding models, by getting the respective sentence embeddings for the test message, and get the weighted score from the three regression models. If the score falls above a pre-determined threshold (again, determined through validation), then the message is flagged as urgent, otherwise it is not.

![Refer to caption](1907.06745v1/RQ1diagram2.png) 

Fig. 1: Training for Urgency Detection.

### IV-B Urgency detection using transfer learning

In this section, we describe our approach for ‘urgency detection transfer’ whereby a *source* dataset is given (similar to RQ1, where both an unlabeled background corpus, as well as a small manually labeled training set, are available) along with a *target* dataset (only a small manually labeled training set and no background corpus), representing the crisis under investigation. Our approach for urgency transfer is captured in Algorithm 1. Many of the steps are similar to those for RQ1, including preprocessing, but there are some important differences. For example, while the Wiki embedding model remains the same as earlier, the manual features are obviously extracted over the target domain (since they do not require a background corpus) and importantly, the ‘local’ embedding model is now trained over the source domain corpus, since there is no target domain unlabeled background corpus available.

To ‘sync’ the source and target domains, we consider a simple, but empirically effective, approach. Rather than use just the labeled target domain data for training the three linear regression models, we combine the labeled training data from both the source and target domains, but the target training data is up-sampled to allow its properties to emerge more concretely in the training. The up-sampling margin is a parameter in Algorithm 1; in practice, a factor of 6 (meaning the target labeled dataset is up-sampled by 6x) has been found to work well. To maximize training dataset utility, we do not use a validation set for classifier weight optimization, but consider the average of all three classifiers as the final score.

Algorithm 1 Transfer Learning for Urgency Detection.

 Input :

*   •

```
Labeled dataset in target domain: DtD\_{t}
```
*   •

```
Labeled dataset in source domain: Ds​lD\_{sl}
```
*   •

```
Unlabeled corpus in source domain: Ds​uD\_{su}
```
*   •

```
Pre-trained Wikipedia Embedding Model: WwW\_{w}
```
*   •

```
Up-sampling parameter: uu
```
 Output :

*   •

```
Classifier for Urgency Detection: 𝒞\\mathcal{C}
```
 Method :

1.  1.

```
Train word embedding WsW\_{s} on text in Ds​u∪Ds​lD\_{su}\\cup D\_{sl} ;
```
2.  2.

```
Up-sample DtD\_{t} by factor uu and ‘mix’ with Ds​lD\_{sl} to get expanded training set, Dt​r​a​i​n:Dt​u∪Ds​lD\_{train}:D\_{tu}\\cup D\_{s}l
```
3.  3.

```
Extract manual feature set FmF\_{m}, source embedding feature set FsF\_{s} (using WsW\_{s}), and Wiki feature set FwF\_{w} (using WwW\_{w}) from each message in Dt​r​a​i​nD\_{train};
```
4.  4.

```
Train linear regression models CsC\_{s}, CmC\_{m} and CwC\_{w} on FsF\_{s}, FmF\_{m} and FwF\_{w} resp. to get classifier;
```
5.  5.

```
Return final classifier model 𝒞:a​v​g​\_​s​c​o​r​e​(Cs,Cm,Cw)\\mathcal{C}:avg\\\_score(C\_{s},C\_{m},C\_{w});
```
## V Experiments

### V-A Data

TABLE II: Details on datasets used for experiments.

 Dataset 

 Unlabeled / Labeled Messages 

 Urgent / Non-urgent Messages 

 Unique Tokens 

 Avg. Tokens / Message 

 Time Range 

 Nepal 

 6,063/400 

 201/199 

 1,641 

 14 

 04/05/2015-05/06/2015 

 Macedonia 

 0/205 

 92/113 

 129 

 18 

 09/18/2018-09/21/2018 

 Kerala 

 92,046/400 

 125/275 

 19,393 

 15 

 08/17/2018-08/22/2018 

For evaluating the approaches laid out in Section [IV](#S4 "IV Approach ‣ Low-supervision urgency detection and transfer in short crisis messages"), we consider three real-world datasets described in Table [II](#S5.T2 "TABLE II ‣ V-A Data ‣ V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages").

Two of the datasets (Nepal and Macedonia) were made available to us through the DARPA LORELEI program, under which this project is funded. The Nepal dataset comprises a collection of tweets collected in the aftermath of the 2015 Nepal earthquake (also called the Gorkha earthquake), while Macedonia was not an actual disaster but a realistic live-action simulation (of a disaster) conduced in Macedonia towards the end of 2018. Macedonia does not have much noise and is ‘information-dense’, but small. As such, it provides a good test of the transfer learning abilities of the approach presented. Kerala describes tweets in the aftermath of the Kerala floods in South India in 2018, and is the largest dataset, with many relevant and irrelevant tweets.

Originally, all the raw messages for the datasets described in Table [II](#S5.T2 "TABLE II ‣ V-A Data ‣ V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages") were unlabeled, in that their urgency status was unknown. Since the Macedonia dataset only contains 205 messages, and is a small but information-dense dataset, we labeled all messages in Macedonia as urgent or non-urgent (hence, there are no unlabeled messages in Macedonia per Table [II](#S5.T2 "TABLE II ‣ V-A Data ‣ V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages")). For the two other Twitter-based datasets, we used active learning to compose a labeled set that would contain challenging examples. The basic process was to do data preprocessing as described in Section [IV](#S4 "IV Approach ‣ Low-supervision urgency detection and transfer in short crisis messages"), followed by training the local fastText-based word embedding model on all messages in the corpus. Next, we randomly labeled 50 urgent and non-urgent tweets and fed them into a classifier. The classifier was applied on the rest of the unlabeled data to obtain ‘ambiguous’ examples (where the classifier’s probability of the positive label was closest to 50%). We labeled another 100 samples this way, and continued to re-train and apply the classifier for two more iterations till we obtained a total of 400 labeled points. Note that the final labeled dataset may not be balanced in terms of urgent and non-urgent messages. Table [II](#S5.T2 "TABLE II ‣ V-A Data ‣ V Experiments ‣ Low-supervision urgency detection and transfer in short crisis messages") shows that Nepal is roughly balanced, while Kerala is imbalanced. We used stratified sampling therefore to split the labeled pool into a training and testing dataset for evaluating the two research questions. We used 90% for training and 10% for testing.

### V-B Metrics

We consider four standard metrics, namely *Accuracy, Precision, Recall and F-Measure*. Accuracy is simply the ratio of correctly labeled messages to the size of test set, precision is the ratio of the true positives to the sum of true positives and false positives, recall is the ratio of true positives to the sum of true positives and false negatives, and finally, F-Measure is the harmonic mean of precision and recall and captures their trade-off.

### V-C Methodology

#### V-C1 Protocol

Concerning RQ1, for datasets, we use *Nepal* and *Kerala* since *Macedonia* does not have a large unlabeled corpus available, which is an assumption made per RQ1. Recall that we used stratified random sampling to split the labeled data for each dataset into training (90%) and test (10%) sets. Of the 90% training set, a further split was done, with 90% kept for ‘training’ and 10% for setting optimal weights for the 3 linear classifiers99 9 The hyperparameters of the linear regression itself were optimized through 5-fold cross-validation on this ‘inner’ (i.e. 90% of the original 90% training set) training set. trained in Section IV. To account for the effects of randomness, each experiment was conducted across ten trials, with averages reported on all four metrics described previously for all baselines described below and our approach. Among the different machine learning classifiers in the sklearn package tested, the linear regression was found to work well and used as the classifier of choice where applicable.

#### V-C2 Baselines for Low-supervision Training for Urgency Detection

We use six baselines to evaluate the approach for RQ1 described in Section [IV](#S4 "IV Approach ‣ Low-supervision urgency detection and transfer in short crisis messages"). Note that statistical significance is tested using the one-sided Student’s paired t-test by comparing the best system (on each metric) against the *Local* baseline, which is a reasonable choice since in a high-supervision (or even normal-supervision) setting, this baseline has been found to perform quite well. Significance at the 90% level is indicated with a \*, at the 95% level with a \*\*, and at the 99% level with a \*\*\*.

TABLE III: Description For Each Baseline On Research Question 1.

| Baseline                                                                 | Description                                                                                                                                                                                                                |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local Embedding (Local)                                                  | The features for a single linear classifier are sentence embeddings (with each pre-processed message treated as a ‘sentence’) trained using the 5-gram skip gram-based fastText model with vector dimensionality set to 20 |
| Manual Feature-based (Manual)                                            | This baseline only considers the 11 manual features described earlier in Section IV                                                                                                                                        |
| Wikipedia Word Embedding (Wiki)                                          | This baselinse only considers the linear classifier trained on the pre-trained Wikipedia Embedding model                                                                                                                   |
| Local Embedding and Manual Feature-based Ensemble (Local-Manual)         | This baseline combines *Local* and *Manual* by training two Linear Regression classifiers and weighting their probabilities to get the final result (using the validation set).                                            |
| Local Embedding and Wikipedia Word Embedding Ensemble (Wiki-Local)       | This baseline combines *Local* and *Wiki* using the same methodology as for *Local-Manual*.                                                                                                                                |
| Wikipedia Word Embedding and Manual Feature-based Ensemble (Wiki-Manual) | This baseline combines *Manual* and *Wiki* using the same methodology as for *Local-Manual*.                                                                                                                               |

#### V-C3 Baselines for Low-supervision Transfer Learning for Urgency Detection

For RQ2, we consider three baselines besides our own approach:

Target-only Local (Target Local): This baseline is essentially the *Wiki-Manual* baseline described in the previous section and trained on the target dataset (i.e. no transfer learning is used, and no source is assumed). This baseline is used to illustrate the benefits of transfer learning, since this baseline sets the minimum benchmark that has to be bested by a transfer learning baseline.

Locally Supervised with Source Embedding (Embedding Transform): Similar to our approach on RQ1, manual features, source embeddings and pre-trained Wikipedia embeddings are used to train three classifiers (but on the labeled target domain), and average their probabilities as the final result. While the local embeddings are trained on the source domain (since unlabeled data is not available for the target domain), all classifier training is always done on the target.

Locally Supervised with Up-sampling and Source Embedding (Upsample): This baseline is the same as Embedding Transform, except to boost the power of the baseline, we upsample the labeled data (in the target dataset) by 6x. Thus, this baseline tries to mitigate source bias and concept drift by giving more importance to the transfer domain. This baseline is also more appropriate for the case where the target training data is extremely limited.

### V-D Results and Discussion

TABLE IV: Results investigating RQ1 on the Nepal and Kerala datasets.

(a) Nepal

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 63.97%63.97\% 

 64.27%64.27\% 

 64.50%64.50\% 

 63.93%63.93\% 

 Manual 

 64.25%64.25\% 

 70.84%∗∗ 

 48.50%48.50\% 

 57.11%57.11\% 

 Wiki 

 67.25%67.25\% 

 66.51%66.51\% 

 69.50%69.50\% 

 67.76%67.76\% 

 Local-Manual 

 65.75%65.75\% 

 67.96%67.96\% 

 59.50%59.50\% 

 62.96%62.96\% 

 Wiki-Local 

 67.40%67.40\% 

 65.54%65.54\% 

 68.50%68.50\% 

 66.80%66.80\% 

 Wiki-Manual 

 67.75%67.75\% 

 70.38%70.38\% 

 63.00%63.00\% 

 65.79%65.79\% 

 *Our Approach* 

 69.25%∗∗∗ 

 68.76%68.76\% 

 70.50%∗∗ 

 69.44%∗∗∗ 

(b) Kerala

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 56.25%56.25\% 

 37.17%37.17\% 

 55.71%55.71\% 

 44.33%44.33\% 

 Manual 

 65.00%65.00\% 

 47.82%47.82\% 

 55.77%\\bf 55.77\% 

 50.63%50.63\% 

 Wiki 

 63.25%63.25\% 

 42.07%42.07\% 

 46.67%46.67\% 

 44.00%44.00\% 

 Local-Manual 

 64.50%64.50\% 

 46.90%46.90\% 

 51.86%51.86\% 

 48.47%48.47\% 

 Wiki-Manual 

 62.25%62.25\% 

 43.56%43.56\% 

 52.63%52.63\% 

 46.93%46.93\% 

 Wiki-Manual 

 68.75%∗∗∗ 

 51.04%51.04\% 

 54.29%54.29\% 

 52.20%∗∗ 

 *Our Approach* 

 68.50%68.50\% 

 51.39%∗∗∗ 

 52.76%52.76\% 

 51.62%51.62\% 

TABLE V: Results investigating RQ2 using the Nepal dataset as source and Macedonia dataset as target.

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 58.76%58.76\% 

 52.96%52.96\% 

 59.19%59.19\% 

 54.95%54.95\% 

 Transform 

 58.62%58.62\% 

 51.40%51.40\% 

 60.32%∗\%^\{\*} 

 55.34%55.34\% 

 Upsample 

 59.38%59.38\% 

 52.35%52.35\% 

 57.58%57.58\% 

 54.76%54.76\% 

 *Our Approach* 

 61.79%∗ 

 55.08%\\% 

 59.19%59.19\% 

 56.90%\\% 

TABLE VI: Results investigating RQ2 using the Kerala dataset as source and Macedonia dataset as target.

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 58.76%58.76\% 

 52.96%52.96\% 

 59.19%59.19\% 

 54.95%54.95\% 

 Transform 

 62.07%62.07\% 

 55.45%55.45\% 

 64.52%64.52\% 

 59.09%59.09\% 

 Upsample 

 64.90%∗∗∗\\%^\{\*\*\*} 

 57.98%∗\\%^\{\*} 

 65.48%∗∗∗\\%^\{\*\*\*} 

 61.30%∗∗∗\\%^\{\*\*\*} 

 *Our Approach* 

 62.90%62.90\% 

 56.28%56.28\% 

 62.42%62.42\% 

 58.91%58.91\% 

TABLE VII: Results investigating RQ2 using the Nepal dataset as source and Kerala dataset as target.

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 58.65%58.65\% 

 42.40%\\% 

 47.47%47.47\% 

 36.88%36.88\% 

 Transform 

 53.74%53.74\% 

 32.89%32.89\% 

 57.47%∗\\%^\{\*} 

 41.42%41.42\% 

 Upsample 

 53.88%53.88\% 

 31.71%31.71\% 

 56.32%56.32\% 

 40.32%40.32\% 

 *Our Approach* 

 58.79%\\% 

 35.26%35.26\% 

 55.89%55.89\% 

 43.03%∗\\%^\{\*} 

TABLE VIII: Results investigating RQ2 using the Kerala dataset as source and Nepal dataset as target.

 System 

 Accuracy 

 Precision 

 Recall 

 F-Measure 

 Local 

 60.26%60.26\% 

 61.80%\\% 

 59.94%59.94\% 

 59.88%59.88\% 

 Transform 

 61.18%∗\\%^\{\*} 

 61.04%61.04\% 

 63.63%63.63\% 

 62.08%62.08\% 

 Upsample 

 60.29%60.29\% 

 59.44%59.44\% 

 66.02%∗\\%^\{\*} 

 62.50%∗\\%^\{\*} 

 *Our Approach* 

 60.06%60.06\% 

 59.54%59.54\% 

 63.98%63.98\% 

 61.64%61.64\% 

Table IV illustrate the result for RQ1 on the Nepal and Kerala datasets. The results illustrate the viability of urgency detection in low-supervision settings (with our approach yielding 69.44% F-Measure on Nepal, at 99% significance compared to the Local baseline), with different feature sets contributing differently to the four metrics. While the local embedding model can reduce precision, for example, it can help the system to improve and accuracy and recall. Similarly, manual features reduce recall, but help the system to improve accuracy and precision (sometimes considerably). To truly address the urgency problem, therefore, a multi-pronged ensemble approach is justified, as also argued intuitively in Section IV. We also note that the pre-trained Wikipedia embedding model proved to be an important tool in improving the generalization ability of the model and not requiring any labeled or unlabeled data; in essence, serving as a free resource that could be helped to regularize and stabilize models that would otherwise be uncertain in low-supervision settings.

Concerning transfer learning experiments (RQ2), we note that source domain embedding model can improve the performance for target model, and upsampling has a generally positive effect (Tables V-VIII). As expected, transfer learning performance (RQ2) is generally lower compared to the low-supervision urgency detection on a *single* dataset1010 10 The best F-Measure achieved on Nepal in Table IV was more than 69%, but when using Kerala as source, only 62.5% F-Measure could be achieved (Table VIII). (RQ1). Note that at least one of the transfer learning methods always bests the *Local* baseline on all metrics (except precision in Table VII, a result not found to be significant even at the 90% level). Our approach shows a slight improvement over the upsampling baseline on two of the four scenarios (Tables V and VII) by 2-2.7% on the F-Measure metric, which shows the diminishing returns from mixing source and target labeled training data. Further improving performance by high margins will require a radically new approach left for future work.

## VI Conclusion and Future Work

This paper presented minimally supervised urgency detection approaches for short texts (such as tweets) in the aftermath of an arbitrary humanitarian crisis such as the 2015 Nepal earthquake. The presented systems covered two scenarios that often emerge in the real world. In the first scenario, a small amount (a few hundred messages) of training data labeled as urgent or non-urgent is available, along with a copious background corpus. In the second scenario, similar data is available for a ‘source’ domain but not for the target domain (expressing a ‘new crisis’) for which the urgency detection needs to be deployed. As messages are streaming in for this new domain, investigators label a few samples, but cannot rely on the availability of a background corpus since urgency needs to be tagged in real time before the crisis has fully subsided. To accomplish this challenging goal, our approach relies on a simple but robust transfer learning methodology. Experimental results on three real-world datasets validate our methods.

Some of the obvious avenues for future work are to improve the existing approach incrementally by (for example) adding more manual features and using more sophisticated local embedding model, possibly with more advanced tuning of hyperparameters like the learning rate and vector dimensionality. For improving transfer learning, we are considering using a deep learning model with priors to truly leverage the presence of a source, albeit one covering a domain that is different from the target. Deep learning for transfer learning is still in its infancy in the machine learning community, and has not been demonstrated for difficult and irregular social media datasets. However, we believe that this presents an opportunity for further study.

## Acknowledgements

The authors gratefully acknowledge the ongoing support and funding of the DARPA LORELEI program, and our partner collaborators in providing detailed analysis. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA, AFRL, or the U.S. Government.

## References

*   \[1\] L. Palen and K. M. Anderson, “Crisis informatics—new data for extraordinary times,” *Science*, vol. 353, no. 6296, pp. 224–225, 2016.
*   \[2\] T. Sakaki, M. Okazaki, and Y. Matsuo, “Earthquake shakes twitter users: real-time event detection by social sensors,” in *Proceedings of the 19th international conference on World wide web*.  ACM, 2010, pp. 851–860.
*   \[3\] B. Pang, L. Lee *et al.*, “Opinion mining and sentiment analysis,” *Foundations and Trends® in Information Retrieval*, vol. 2, no. 1–2, pp. 1–135, 2008.
*   \[4\] S. J. Pan and Q. Yang, “A survey on transfer learning,” *IEEE Transactions on knowledge and data engineering*, vol. 22, no. 10, pp. 1345–1359, 2010.
*   \[5\] S. Verma, S. Vieweg, W. J. Corvey, L. Palen, J. H. Martin, M. Palmer, A. Schram, and K. M. Anderson, “Natural language processing to the rescue? extracting” situational awareness” tweets during mass emergency,” in *Fifth International AAAI Conference on Weblogs and Social Media*, 2011.
*   \[6\] K. Starbird, G. Muzny, and L. Palen, “Learning from the crowd: collaborative filtering techniques for identifying on-the-ground twitterers during mass disruptions,” in *Proceedings of 9th International Conference on Information Systems for Crisis Response and Management, ISCRAM*, 2012, pp. 1–10.
*   \[7\] A. Olteanu, C. Castillo, F. Diaz, and S. Vieweg, “CrisisLex: A lexicon for collecting and filtering microblogged communications in crises.” in *Proc. Int. Conf. Weblogs and Social Media (ICWSM)*, Oxford, UK, 2014.
*   \[8\] M. Barrenechea, K. M. Anderson, A. A. Aydin, M. Hakeem, and S. Jambi, “Getting the query right: User interface design of analysis platforms for crisis research,” in *Engineering the Web in the Big Data Era*, P. Cimiano, F. Frasincar, G.-J. Houben, and D. Schwabe, Eds.  Cham: Springer International Publishing, 2015, pp. 547–564.
*   \[9\] L. Palen, R. Soden, T. J. Anderson, and M. Barrenechea, “Success &#38; scale in a data-producing organization: The socio-technical evolution of openstreetmap in response to humanitarian events,” in *Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems*, ser. CHI ’15.  New York, NY, USA: ACM, 2015, pp. 4113–4122. \[Online\]. Available: [http://doi.acm.org/10.1145/2702123.2702294](http://doi.acm.org/10.1145/2702123.2702294 "")
*   \[10\] M. Kogan, L. Palen, and K. M. Anderson, “Think local, retweet global: Retweeting by the geographically-vulnerable during hurricane sandy,” in *Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work &#38; Social Computing*, ser. CSCW ’15.  New York, NY, USA: ACM, 2015, pp. 981–993. \[Online\]. Available: [http://doi.acm.org/10.1145/2675133.2675218](http://doi.acm.org/10.1145/2675133.2675218 "")
*   \[11\] K. M. Anderson, A. Schram, A. Alzabarah, and L. Palen, “Architectural implications of social media analytics in support of crisis informatics research,” *IEEE Data Eng. Bull.*, vol. 36, pp. 13–20, 2013.
*   \[12\] R. Soden, N. Budhathoki, and L. Palen, “Resilience-building and the crisis informatics agenda: Lessons learned from open cities kathmandu,” in *ISCRAM*, 2014.
*   \[13\] M. Palatucci, D. Pomerleau, G. E. Hinton, and T. M. Mitchell, “Zero-shot learning with semantic output codes,” in *Advances in neural information processing systems*, 2009, pp. 1410–1418.
*   \[14\] B. Romera-Paredes and P. Torr, “An embarrassingly simple approach to zero-shot learning,” in *International Conference on Machine Learning*, 2015, pp. 2152–2161.
*   \[15\] H. Uszkoreit, F. Xu, and H. Li, “Analysis and improvement of minimally supervised machine learning for relation extraction.” in *NLDB*.  Springer, 2009, pp. 8–23.
*   \[16\] C. C. Aggarwal and C. Zhai, *Mining text data*.  Springer Science & Business Media, 2012.
*   \[17\] X. Zhu, “Semi-supervised learning literature survey,” 2005.
*   \[18\] B. Settles, “Active learning literature survey,” *University of Wisconsin, Madison*, vol. 52, no. 55-66, p. 11, 2010.
*   \[19\] M. Sahlgren, “An introduction to random indexing,” 2005.
*   \[20\] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa, “Natural language processing (almost) from scratch,” *Journal of Machine Learning Research*, vol. 12, no. Aug, pp. 2493–2537, 2011.
*   \[21\] M. Kejriwal and P. Szekely, “Information extraction in illicit web domains,” in *Proceedings of the 26th International Conference on World Wide Web*.  International World Wide Web Conferences Steering Committee, 2017, pp. 997–1006.
*   \[22\] D. Nadeau and S. Sekine, “A survey of named entity recognition and classification,” *Lingvisticae Investigationes*, vol. 30, no. 1, pp. 3–26, 2007.
*   \[23\] A. Moro, A. Raganato, and R. Navigli, “Entity linking meets word sense disambiguation: a unified approach,” *Transactions of the Association for Computational Linguistics*, vol. 2, pp. 231–244, 2014.
*   \[24\] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,” in *Advances in neural information processing systems*, 2013, pp. 3111–3119.
*   \[25\] A. M. Dai, C. Olah, and Q. V. Le, “Document embedding with paragraph vectors,” *arXiv preprint arXiv:1507.07998*, 2015.
*   \[26\] A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, “Bag of tricks for efficient text classification,” *arXiv preprint arXiv:1607.01759*, 2016.
*   \[27\] M. Faruqui, Y. Tsvetkov, P. Rastogi, and C. Dyer, “Problems with evaluation of word embeddings using word similarity tasks,” *arXiv preprint arXiv:1605.02276*, 2016.
*   \[28\] F. Alam, S. Joty, and M. Imran, “Domain adaptation with adversarial training and graph embeddings,” *arXiv preprint arXiv:1805.05151*, 2018.
*   \[29\] B. Pedrood and H. Purohit, “Mining help intent on twitter during disasters via transfer learning with sparse coding,” in *International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation*.  Springer, 2018, pp. 141–153.
*   \[30\] X. He, D. Lu, D. Margolin, M. Wang, S. E. Idrissi, and Y.-R. Lin, “The signals and noise: actionable information in improvised social media channels during a disaster,” in *Proceedings of the 2017 ACM on Web Science Conference*.  ACM, 2017, pp. 33–42.
*   \[31\] H. Purohit, C. Castillo, M. Imran, and R. Pandey, “Social-eoc: Serviceability model to rank social media requests for emergency operation centers,” in *2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)*.  IEEE, 2018, pp. 119–126.
*   \[32\] F. Alam, F. Ofli, and M. Imran, “Crisismmd: Multimodal twitter datasets from natural disasters,” in *Twelfth International AAAI Conference on Web and Social Media*, 2018.
*   \[33\] X. Zheng, A. Sun, S. Wang, and J. Han, “Semi-supervised event-related tweet identification with dynamic keyword generation,” in *Proceedings of the 2017 ACM on Conference on Information and Knowledge Management*.  ACM, 2017, pp. 1619–1628.
