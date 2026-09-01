# Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages

Muhammad Imran<sup>1</sup> , Prasenjit Mitra<sup>1</sup> , Carlos Castillo<sup>2</sup>

<sup>1</sup>Qatar Computing Research Institute (HBKU), Doha, Qatar

<sup>2</sup>Eurecat, Barcelona, Spain

mimran@qf.org.qa, pmitra@qf.org.qa, chato@acm.org

#### Abstract

Microblogging platforms such as Twitter provide active communication channels during mass convergence and emergency events such as earthquakes, typhoons. During the sudden onset of a crisis situation, affected people post useful information on Twitter that can be used for situational awareness and other humanitarian disaster response efforts, if processed timely and effectively. Processing social media information pose multiple challenges such as parsing noisy, brief and informal messages, learning information categories from the incoming stream of messages and classifying them into different classes among others. One of the basic necessities of many of these tasks is the availability of data, in particular human-annotated data. In this paper, we present human-annotated Twitter corpora collected during 19 different crises that took place between 2013 and 2015. To demonstrate the utility of the annotations, we train machine learning classifiers. Moreover, we publish first largest word2vec word embeddings trained on 52 million crisis-related tweets. To deal with tweets language issues, we present human-annotated normalized lexical resources for different lexical variations.

Keywords: Natural language processing, Twitter, Disaster response, Supervised classification

## 1. Introduction

Twitter has been extensively used as an active communication channel, especially during mass convergence events such as natural disasters like earthquakes, floods, typhoons [\(Imran et al., 2015;](#page-5-0) [Hughes and Palen, 2009\)](#page-5-1). During the onset of a crisis, a variety of information is posted in real-time by affected people; by people who are in need of help (e.g., food, shelter, medical assistance, etc.) or by people who are willing to donate or offer volunteering services [\(Vieweg, 2012\)](#page-5-2). Moreover, humanitarian and formal crisis response organizations such as government agencies, public health care NGOs, and military are tasked with responsibilities to save lives, reach people who need help, etc. [\(Vieweg et al., 2014\)](#page-5-3). Situation-sensitive requirements arise during such events and formal disaster response agencies look for actionable and tactical information in realtime to effectively estimate early damage assessment, and to launch relief efforts accordingly.

Recent studies have shown the importance of social media messages to enhance situational awareness and also indicate that these messages contain significant actionable and tactical information [\(Cameron et al., 2012;](#page-5-4) [Imran et](#page-5-5) [al., 2013;](#page-5-5) [Purohit et al., 2013\)](#page-5-6). Many Natural-Language-Processing (NLP) techniques such as automatic summarization, information classification, named-entity recognition, information extraction can be used to process such social media messages [\(Bontcheva et al., 2013;](#page-5-7) [Imran et](#page-5-0) [al., 2015\)](#page-5-0). However, many social media messages are very brief, informal, and often contain slangs, typograpical errors, abbreviations, and incorrect grammar [\(Han et al.,](#page-5-8) [2013\)](#page-5-8). These issues degrade the performance of many NLP techniques when used down the processing pipeline [\(Ritter](#page-5-9) [et al., 2010;](#page-5-9) [Foster et al., 2011\)](#page-5-10).

We present Twitter corpora consisting of more than 52 million crisis-related messages collected during 19 different crises. We provide human annotations (volunteers and crowd-sourced workers) of two types. First, the tweets are annotated with a set of categories such as displaced people, financial needs, infrastructure, etc. These annotation schemes were built using input taken from formal crisis response agencies such as United Nations Office for the Coordination of Humanitarian Affairs (UN OCHA). Second, the tweets are annotated to identify out-of-vocabulary(OOV) terms, such as slangs, places names, abbreviations, misspellings, etc. and their corrections and normalized forms. This dataset can form the basis for research in text classification for short messages and for research on normalizing informal language.

Creating large corpora for training supervised machinelearning models is hard because it requires time and money that may not be available. However, since our dataset was used for disaster relief efforts, volunteers were willing to annotate it; this work can now be leveraged to improve text classification and language processing tasks. Our work provides annotations for around 50,000 thousand messages, which is a significant corpus, that will enable research into applied machine learning and consequently benefit the disaster relief (and other) research communities. Our dataset has been collected from various countries and during various times of the year. This diversity would make it an interesting dataset that if used would be a foil to solutions that only work for specific language "dialects", e.g., American English and would fail or suffer from degradation of quality if applied to variations, such as Indian English. Our work shows that when a dataset is used for a real application, we could obtain larger number of annotations than otherwise. These can then be used to improve text processing as a byproduct.

The annotated data is also used to train machine-learning classifiers. In this case, we use three well-known learning algorithms: Naive Bayes, Random Forest, and Support Vector Machines (SVM). We remark that these classifiers are useful for formal crisis response organizations as well as for the research community to build more effective computational methods [\(Pak and Paroubek, 2010;](#page-5-11) [Imran et al.,](#page-5-0) [2015\)](#page-5-0) on top. We also train word2vec word embeddings from all 52 million messages and make them available to research community.

### 1.1. Contributions

The contributions of this paper are as follows:

- 1. We present human-annotated crisis-related messages collected during 19 different crises
- 2. We use human-annotations to built machine-learning classifiers in a multiclass classification setting to classify messages that are useful for humanitarian efforts
- 3. We provide first largest word2vec word embeddings trained using 52 million crisis-related messages
- 4. We use the collected data to identify OOV (out-ofvocabulary) words and provide human-annotated normalized lexical resources for different lexical variations

#### 1.2. Paper organization

The rest of the paper is organized as follows. In the next section, we describe datasets details and annotation schemes. Section 3 describes supervised classification task and word2vec word embeddings. Section 4 provides details of text normalization and we present related work in section 5. We conclude the paper in section 6.

### 2. Crises Corpora Collection and Annotation

### 2.1. Data collection

We collected crisis-related messages from Twitter posted during 19 different crises that took place from 2013 to 2015. Table [1](#page-2-0) shows the list of crisis events along with their names, crisis type (e.g. earthquake, flood), countries where they took place, and the number of tweets each crisis contains. We collected these messages using our AIDR (Artificial Intelligence for Disaster Response) platform [\(Imran](#page-5-12) [et al., 2014\)](#page-5-12). AIDR is an open source platform to collect and classify Twitter messages during the onset of a humanitarian crisis. AIDR has been used by UN OCHA during many major disasters such as Nepal Earthquake, Typhoon Hagupit.

AIDR provides different convenient ways to collect messages from Twitter using the Twitter's streaming API. One can use different data collection strategies. For example, collecting tweets that contain some keywords and are specifically from a particular geographical area/region/city (e.g. New York). The detailed data collection strategies used to collect the datasets shown in Table [1](#page-2-0) are included in each dataset folder.

## 2.2. Data annotation

Messages posted on social media vary greatly in terms of information they contain. For example, users post messages of personal nature, messages useful for situational awareness (e.g. infrastructure damage, causalities, individual needs), or not related to the crisis at all. Depending on their information needs, different humanitarian organizations use different annotation schemes to categories these messages. In this work, we use a subset of the annotations used by the United Nations Office for the Coordination of Humanitarian Affairs (UN OCHA). The 9 category types (including two catch-all classes: "Other Useful Information" and "Irrelevant") used by the UN OCHA are shown in the below-presented annotation scheme. For most of the datasets we have performed annotations by employing volunteers and paid workers.

To perform volunteered-based annotations, messages were collected from Twitter in real-time and passed through a deduplication process. Only unique messages were considered for human-annotation. We use Stand-By-Task-Force (SBTF)[<sup>1</sup>](#page-1-0) volunteers to annotate messages using our MicroMappers platform.[<sup>2</sup>](#page-1-1) The real-time annotation process helps train machine learning classifiers rapidly, which are then used to classify new incoming messages. This process helps address time-critical information needs requirement of many humanitarian organizations.

After the first round of annotations, we found that some categories are small in terms of number of labels thus showing high class-imbalance. A dataset is said to be *imbalanced* if at least one of the classes has significantly fewer annotated instances than the others. The class imbalance problem has been known to hinder the learning performance of classification algorithms. In this case, we performed another round of annotations for datasets that have high class imbalance using the paid crowdsourcing platform CrowdFlower.[<sup>3</sup>](#page-1-2)

In both annotation processes, an annotation task consists of a tweet and the list of categories listed below. A paid worker or volunteer reads the message and selects one of the categories most suitable for the message. Messages that do not belong to any category but contain some important information are categorized as "Other Useful Information".

A task is finalized (i.e. a category is assigned) when three different volunteers/paid workers agree on a category. According to the Twitter's data distribution policy, we are not allowed to publish actual contents of more than 50k tweets. For this reason, we publish all annotated tweets, which are less than 50k, along with tweet-ids of all the unannotated messages at [https://github.com/](https://github.com/CrisisNLP/lrec16_tools) [CrisisNLP/lrec16\\_tools](https://github.com/CrisisNLP/lrec16_tools). We also provide a tweets retrieval tool implemented in Java, which can be used to get full tweets content from Twitter.

In below we show the annotation scheme used for crisis events caused by natural disasters. For other events, details regarding their annotations are available with the published data.

#### Annotation scheme: Categorizing messages by information types

- *Injured or dead people*: Reports of casualties and/or injured people due to the crisis

<span id="page-1-0"></span><sup>1</sup><http://blog.standbytaskforce.com/>

<span id="page-1-1"></span><sup>2</sup><http://micromappers.org/>

<span id="page-1-2"></span><sup>3</sup><http://crowdflower.com/>

<span id="page-2-0"></span>Table 1: Crises datasets details including crisis type, name, year, language of messages, country, # of tweets.

- *Missing, trapped, or found people*: Reports and/or questions about missing or found people
- *Displaced people and evacuations*: People who have relocated due to the crisis, even for a short time (includes evacuations)
- *Infrastructure and utilities damage*: Reports of damaged buildings, roads, bridges, or utilities/services interrupted or restored
- *Donation needs or offers or volunteering services*: Reports of urgent needs or donations of shelter and/or supplies such as food, water, clothing, money, medical supplies or blood; and volunteering services
- *Caution and advice*: Reports of warnings issued or lifted, guidance and tips
- *Sympathy and emotional support*: Prayers, thoughts, and emotional support
- *Other useful information*: Other useful information that helps understand the situation
- *Not related or irrelevant*: Unrelated to the situation or irrelevant

## 3. Classification of Messages

To make sense of huge amounts of Twitter messages posted during crises, we consider a basic operation, that is, the automatic categorization of messages into the categories of interest. This is a multiclass categorization problem in which instances are categorized into one of several classes. Specifically, we aim at learning a predictor h : X → Y, where X is the set of messages and Y is a finite set of categories. For this purpose, we use three well-known learning algorithms i.e. Naive Bayes (NB), Support Vector Machines (SVM), and Random Forest (RF).

#### 3.1. Preprocessing and feature extraction

Prior to learning a classifier, we perform the following preprocessing steps. First, stop-words, URLs, and usermentions are removed from the Twitter messages. We perform stemming using the Lovins stemmer. We use Unigrams and bi-grams as our features. Previous studies found these two features outperform when used for similar classification tasks [\(Imran et al., 2013\)](#page-5-5). Finally, we used the information gain, a well-know feature selection method to select top 1k features. The labeled data we used in this task was annotated by the paid workers.

### 3.2. Evaluation and Results

We trained all three different kinds of classifiers using the preprocessed data. For the evaluation of the trained models, we used 10-folds cross-validation technique. Table [2](#page-3-0) shows the results of the classification task in terms of Area Under ROC curve[<sup>4</sup>](#page-2-1) for all classes of the 8 different disaster datasets. We also show the proportion of each class in each dataset.

Given the complexity of the task i.e. multiclass classification of short messages, we can see that all three classifiers have pretty decedent results. In this case, a random classifier represents an AUC = 0.50 and higher values are preferable. Other than the "missing trapped or found people" class, which is the smallest class in term of proportion across all the datasets, results for most of the other classes are at the acceptable level (i.e. ≥ 0.80).

### 3.3. Crisis word embeddings

Many applications of machine learning and computational linguistics rely on semantic representations and relationships between words of a text document. Many different types of methods have been proposed that use continuous representations of words such as Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA). How-

<span id="page-2-1"></span><sup>4</sup>[https://en.wikipedia.org/wiki/Receiver\\_](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) [operating\\_characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

<span id="page-3-0"></span>Table 2: Classification results in terms of Area Under ROC Curve for selected datasets across all classes using Support Vector Machines (SVM), Naive Bayes (NB), and Random Forest (RF).

ever, recently models based on distributional representations of words become more famous. In this work, we train word embeddings (i.e. distributed word representations) using the 52 million Twitter messages in our datasets and make it available to research community. To the best of our knowledge this is the first largest word embeddings that are trained on crisis-related tweets.

We use word2vec, a very popular software to train word embedding [\(Mikolov et al., 2013\)](#page-5-13). As preprocessing, we replaced URLs, digits, and usernames with fixed constants and removed special characters. Finally, the word embeddings are generated using Continuous Bag Of Words (CBOW) architecture with negative sampling along with 300 word representation dimensionality.

## 4. Twitter Text Normalization

### 4.1. Language issues in Twitter messages

The quality—in terms of readability, grammar, sentence structure etc.—of Twitter messages vary significantly. Typically, Twitter messages are brief, informal, noisy, unstructured, and often contain misspellings and grammatical mistakes. Moreover, due to Twitter's 140 character limit restriction, Twitter users intentionally shorten words by using abbreviations, acronyms, slangs, and sometimes words without spaces. The accuracy of natural language processing techniques would improve if we can identify the informal nature of the language in tweets and normalize OOV terms [\(Han et al., 2013\)](#page-5-8). We divide these lexical variations into the following five categories:

- 1. *Typos/misspellings:* e.g. earthquak (earthquake), missin (missing), ovrcme (overcome)

- 2. *Single-word abbreviation/slangs:* e.g. pls (please), srsly (seriously), govt (government), msg (message)
- 3. *Multi-word abbreviation/slangs:* e.g. imo (in my opinion), im (i am), brb (be right back)
- 4. *Phonetics substitutions:* e.g. 2morrow (tomorrow), 4ever (forever), 4g8 (forget), w8 (wait)
- 5. *Words without spaces:* e.g. prayfornepal (pray for nepal), wehelp (we help), weneedshelter (we need shelter)

#### 4.2. Identification of candidate OOV words

To identify candidate OOV words that require normalization, we first build initial vocabularies consisting of lexical variations mentioned in the previous section. We use a dictionary available on the web to normalize abbreviations, chat shortcuts, and slang.[<sup>5</sup>](#page-3-1) We also use the SCOWL (Spell Checker Oriented Word Lists) aspell English dictionary [<sup>6</sup>](#page-3-2) that consists of 349,554 English words. The SCOWL dictionary is suitable for English spell checkers for most of English dialects. Although, the SCOWL dictionary contains places names (e.g. names of countries and famous cities), after testing it on Nepal Earthquake data, we found that its coverage is not complete and a large number of cities/towns of Nepal are missing. To overcome this issue, we use the

<span id="page-3-1"></span><sup>5</sup>[http://www.innocentenglish.com/news/](http://www.innocentenglish.com/news/texting-abbreviations-collection-texting-slang.html) [texting-abbreviations-collection-texting](http://www.innocentenglish.com/news/texting-abbreviations-collection-texting-slang.html)[slang.html](http://www.innocentenglish.com/news/texting-abbreviations-collection-texting-slang.html)

<span id="page-3-2"></span><sup>6</sup><http://wordlist.aspell.net/>

<span id="page-4-3"></span>Figure 1: Crowdsourcing task for Twitter out-of-vocabulary words normalization

MaxMind [<sup>7</sup>](#page-4-0) world cities database that consists of 3,173,959 cities.

Using the above resources, we try to find OOV words in the dataset. However, we observed that a large number of OOVs consist of misspelled words for which a correct form can be obtained using one edit-distance change (i.e. by performing one insertion, deletion, or substitution operation). For this purpose, we train a language model using lists of most frequent words from Wiktionary,[<sup>8</sup>](#page-4-1) the British National Corpus,[<sup>9</sup>](#page-4-2) and words in our SCOWL dictionary. For a given misspelled word w, we aim to find a correction c out of all possible corrections where the probability of c given w is maximum, i.e., argmaxcP(c|w) By Bayes Theorem this

is equivalent to:

argmaxcP(c|w) = argmaxcP(w|c)P(c)/P(w)

or it can be written as:

argmaxcP(c|w) = argmaxcP(w|c)P(c)

where P(c) is the probability that c is the correct word and P(w|c) is the probability that the author typed w when c was intended. We then restrict the language model to predict corrections within one edit-distance range and from those choose the one with highest probability. Misspellings for which more than one change is required, we consider them as OOVs to be corrected by human workers.

<span id="page-4-0"></span><sup>7</sup>[https://www.maxmind.com/en/free-world](https://www.maxmind.com/en/free-world-cities-database)[cities-database](https://www.maxmind.com/en/free-world-cities-database)

<span id="page-4-2"></span><span id="page-4-1"></span><sup>8</sup><http://en.wiktionary.org/wiki/Wiktionary>

<sup>9</sup>[http://www.kilgarriff.co.uk/bnc-readme.](http://www.kilgarriff.co.uk/bnc-readme.html) [html](http://www.kilgarriff.co.uk/bnc-readme.html)

- 4.3. Normalization of OOV words To normalize the identified OOV words, we used the CrowdFlower crowdsourcing platform. A crowdsourcing task in this case consists of a Twitter message that contains one or more OOV words and a set of instructions shown in Figure [1.](#page-4-3) The workers were asked to read the instructions and examples carefully before providing an answer. A worker reads the given message and provides a correct OOV tag (i.e. slang/abbreviation/acronym, a location name, an organization name, a misspelled word, or a person name). If an OOV is a misspelled word, the worker also provides its corrected form. We provide all the resources and the results of crowdsoucing to research community.
- <span id="page-5-8"></span><span id="page-5-5"></span><span id="page-5-1"></span>5. Related Work The use of microblogging platforms such as Twitter during the sudden onset of a crisis situation has been increased in the last few years. Thousands of crisis-related messages that are posted online contain important information that can also be useful to humanitarian organizations for disaster response efforts, if processed timely and effectively [\(Hughes and Palen, 2009;](#page-5-1) [Imran et al., 2015\)](#page-5-0). Many different types of processing approaches based on techniques ranging from machine learning to natural language processing to computational linguistics have been developed [\(Corvey et al., 2010\)](#page-5-14). Despite there exists some resources e.g. [\(Temnikova et al., 2015;](#page-5-15) [Olteanu et al.,](#page-5-16) [2015\)](#page-5-16), however, due to the scarcity of relevant data, in particular human-annotated data, crisis informatics researchers still cannot fully utilize the capabilities of different computational methods. To overcome these issues, we present to research community a corpora consisting of labeled and unlabeled crisis-related Twitter messages. Moreover, we also provide normalized lexical resources useful for linguistic analysis of Twitter messages.
- <span id="page-5-16"></span><span id="page-5-13"></span><span id="page-5-11"></span><span id="page-5-6"></span>6. Conclusions We present Twitter corpora consisting of over 52 million crisis-related tweets collected during 19 crisis events. We provide two sets of annotations related to topiccategorization of the tweets and tagging out-of-vocabulary words and their normalizations. We build machine-learning classifiers to empirically validate the effectiveness of the annotated datasets. We also provide word2vec word embeddings trained on 52 million messages. We believe that these resources and the tools built using them will help improve automatic natural language processing of crisisrelated messages and eventually be useful for humanitarian organizations.
- <span id="page-5-15"></span><span id="page-5-9"></span><span id="page-5-7"></span><span id="page-5-3"></span>7. References Bontcheva, K., Derczynski, L., Funk, A., Greenwood,
- <span id="page-5-4"></span><span id="page-5-2"></span>M. A., Maynard, D., and Aswani, N. (2013). Twitie: An open-source information extraction pipeline for microblog text. In *RANLP*, pages 83–90. Cameron, M. A., Power, R., Robinson, B., and Yin, J. (2012). Emergency situation awareness from twitter for crisis management. In *Proc. of the 21st international conference companion on World Wide Web*, pages 695–
- <span id="page-5-14"></span><span id="page-5-10"></span>698. ACM. Corvey, W. J., Vieweg, S., Rood, T., and Palmer, M. (2010). Twitter in mass emergency: what nlp techniques can contribute. In *Proc. of the NAACL HLT 2010 Workshop on Computational Linguistics in a World of Social Media*, pages 23–24. Association for Computational Linguistics. Foster, J., C¸ etinoglu, O., Wagner, J., Le Roux, J., Hogan, ¨ S., Nivre, J., Hogan, D., and Van Genabith, J. (2011). # hardtoparse: Pos tagging and parsing the twitterverse. In *AAAI 2011 Workshop on Analyzing Microtext*, pages 20–25. Han, B., Cook, P., and Baldwin, T. (2013). Lexical normalization for social media text. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 4(1):5. Hughes, A. L. and Palen, L. (2009). Twitter adoption and use in mass convergence and emergency events. *International Journal of Emergency Management*, 6(3-4):248–
  - 260. Imran, M., Elbassuoni, S. M., Castillo, C., Diaz, F., and Meier, P. (2013). Extracting information nuggets from disaster-related messages in social media. *Proc. of IS-CRAM, Baden-Baden, Germany*. Imran, M., Castillo, C., Lucas, J., Meier, P., and Vieweg,
  - S. (2014). AIDR: Artificial intelligence for disaster response. In *Proc. of the companion publication of the 23rd international conference on World wide web companion*, pages 159–162. International World Wide Web Conferences Steering Committee. Imran, M., Castillo, C., Diaz, F., and Vieweg, S. (2015). Processing social media messages in mass emergency: A survey. *ACM Computing Surveys (CSUR)*, 47(4):67. Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*. Olteanu, A., Vieweg, S., and Castillo, C. (2015). What to expect when the unexpected happens: Social media communications across crises. In *Proc. of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing*, pages 994–1009. ACM. Pak, A. and Paroubek, P. (2010). Twitter as a corpus for sentiment analysis and opinion mining. In *LREC*, volume 10, pages 1320–1326. Purohit, H., Castillo, C., Diaz, F., Sheth, A., and Meier,
  - P. (2013). Emergency-relief coordination on social media: Automatically matching resource requests and offers. *First Monday*, 19(1). Ritter, A., Cherry, C., and Dolan, B. (2010). Unsupervised modeling of twitter conversations. In *Proc of NAACL*. Temnikova, I., Castillo, C., and Vieweg, S. (2015). Emterms 1.0. In *Information Systems for Crisis Response and Management, ISCRAM*. Vieweg, S., Castillo, C., and Imran, M. (2014). Integrating social media communications into the rapid assessment of sudden onset disasters. In *Social Informatics*, pages 444–461. Springer. Vieweg, S. E. (2012). Situational awareness in mass emergency: A behavioral and linguistic analysis of microblogged communications.

<span id="page-5-12"></span><span id="page-5-0"></span>