# **Robust Classification of Crisis-Related Data on Social Networks Using Convolutional Neural Networks** 

**Dat Tien Nguyen,**<sup>*****</sup> **Kamela Ali Al Mannai,**<sup>*****</sup> **Shafiq Joty,**<sup>*****</sup> **Hassan Sajjad,**<sup>*****</sup> **Muhammad Imran,**<sup>*****</sup> **Prasenjit Mitra**<sup>******</sup> 

> *Qatar Computing Research Institute, HBKU Doha, Qatar 

**Pennsylvania State University, University Park, USA 

_{_ ndat, kamlmannai, sjoty, hsajjad, mimran _}_ @hbku.edu.qa, pmitra@ist.psu.edu 

#### **Abstract** 

The role of social media, in particular microblogging platforms such as Twitter, as a conduit for actionable and tactical information during disasters is increasingly acknowledged. However, time-critical analysis of big crisis data on social media streams brings challenges to machine learning techniques, especially the ones that use supervised learning. The scarcity of labeled data, particularly in the early hours of a crisis, delays the learning process. Existing classification methods require a significant amount of labeled data specific to a particular event for training plus a lot of feature engineering to achieve best results. In this work, we introduce neural network based classification methods for identifying useful tweets during a crisis situation. At the onset of a disaster when no labeled data is available, our proposed method makes the best use of the out-of-event data and achieves good results. 

## **Introduction** 

Time-critical analysis of social media data streams is important for many application areas (Lee, Agrawal, and Choudhary 2013; Rudra et al. 2016). During the onset of a crisis situation, people use social media platforms to post situational updates, look for useful information, and ask for help (Imran et al. 2015). Rapid analysis of messages posted on microblogging platforms such as Twitter can help humanitarian organizations gain situational awareness, learn about urgent needs of affected people, critical infrastructure damage, and medical emergencies (Nguyen et al. 2016). 

Automatic identification of useful tweets is a challenging task because: ( _i_ ) tweets are short – only 140 characters – and therefore, hard to understand without enough context; ( _ii_ ) they often contain abbreviations, informal language, spelling variations and are ambiguous; and, ( _iii_ ) judging a tweet’s utility is a subjective exercise. Despite advances in natural language processing (NLP), interpreting short informal texts automatically remains a hard problem. 

Supervised machine learning algorithms are dependent on labeled data from the event for training. The performance of models trained using data from previous events ( _out-of-event data_ ) is poor due to discrete word representations and the variety across events from which the historical data was collected. Second, training a classifier from scratch every time 

Copyright _⃝_ c 2017, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

a disaster occurs is infeasible due to labeled data scarcity. Third, traditional approaches require manually engineered features like cue words and TF-IDF vectors (Imran et al. 2015) for learning. Moreover, manual adaptation of models to changes in features and their importance is undesirable (and often infeasible) because of the effort it takes to do so. 

Deep neural networks (DNNs) are ideally suited for classifying a stream of crisis-related tweets. They are usually trained with online methods and have the flexibility to learn and adapt from new batches of labeled data without requiring to retrain from scratch. Due to their distributed word representation, they generalize well and make better use of the previously labeled data from other events to speed up the learning process in the beginning of a disaster. DNNs obviate the need of manually crafting features and automatically learn latent features as distributed dense vectors, which have shown to benefit various NLP tasks (Collobert et al. 2011). 

In this paper, we use a convolutional neural network (CNN) to classify tweets. CNNs capture the most salient _n_ -gram information by means of its convolution and maxpooling operations.We present a series of experiments using different variations of the training data – event data only, out-of-event data only, and both. Experiments results show that neural network models perform better than non-neural models.They can be used reliably with the already available out-of-event data. DNNs are better suited for classification of crisis data than conventional classifiers because they can learn features automatically and can be adopted to online settings. 

## **Convolutional Neural Network** 

Figure 1 shows our CNN model for classifying tweets into _useful_ vs. _not useful_ for a crisis event. The architecture of our model is similar to the one proposed in (Kim 2014). 

For distributed representation of words, we first construct a vocabulary _V_ from the training set by selecting _T_ most frequent words. Each word in the vocabulary is then represented by a _D_ dimensional vector in a shared look-up table _L ∈_ R<sup>_|V |×D_</sup> , which is considered a model parameter to learn. We can initialize _L_ randomly or using pretrained word embedding vectors like word2vec (Mikolov et al. 2013a). 

Given an input tweet **s** = ( _w_ 1 _, · · · , wT_ ), we first transform it into a feature sequence by mapping each word token _wt ∈_ **s** to an index in _L_ . The look-up layer then creates an 

632 



�����������������<br>���������������������<br>���� x 1 U ��������������<br>���������<br>��� x 2 h 1�<br>m 1� V<br>w<br>���� m 2 y<br>h 2�<br>Output�<br>x T� � mN layer�<br>����� Dense�<br>h N�<br>layer�<br>x T Max�<br>����� pooling�<br>Feature�<br>Input� Look-up�<br>maps�<br>layer� layer�<br>���<br>��� ������<br>���<br>��� �<br>���<br>������ � ����� ��<br>��� ��� ���<br>������<br>���<br>
Figure 1: Convolutional neural network on a tweet: _“guys if know any medical emergency around balaju area you can reach umesh HTTP doctor at HTTP”_ 

input vector **xt** _∈_ R<sup>_D_</sup> for each token _wt_ , which are passed through a sequence of convolution and pooling operations to learn high-level feature representations. 

A convolution operation involves applying a _filter_ **u** _∈_ R<sup>_L.D_</sup> (i.e., a vector of parameters) to a window of _L_ words to produce a new feature 



where **x** _t_ : _t_ + _L−_ 1 denotes the concatenation of _L_ input vectors, _bt_ is a bias term, and _f_ is a nonlinear activation function (e.g., sig _,_ tanh). We apply this filter to each possible _L_ -word window in the tweet to generate a _feature map_ **h**<sup>_i_</sup> = [ _h_ 1 _, · · · , hT_ + _L−_ 1]. We repeat this process _N_ times with _N_ different filters to get _N_ different feature maps. We use a _wide_ convolution (Kalchbrenner, Grefenstette, and Blunsom 2014) (as opposed to _narrow_ ), which ensures that the filters reach the entire sentence, including the boundary words. This is done by performing _zero-padding_ , where outof-range (i.e., _t<_ 1 or _t>T_ ) vectors are assumed to be zero. 

After the convolution, we apply a max-pooling operation to each feature map 



where _μp_ ( **h**<sup>_i_</sup> ) refers to the max operation applied to each window of _p_ features in the feature map **h**<sup>_i_</sup> . For instance, with _p_ = 2, this pooling gives the same number of features as in the feature map (because of the zero-padding). Intuitively, the filters compose local _n_ -grams into higherlevel representations in the feature maps, and max-pooling reduces the output dimensionality while keeping the most important aspects from each feature map. 

Since each convolution-pooling operation is performed independently, the features extracted become invariant in locations (i.e., where they occur in the tweet), thus acts like bag-of- _n_ -grams. However, keeping the _order_ information could be important for modeling sentences. In order to model interactions between the features picked up by the 

filters and the pooling, we include a _dense_ layer of hidden nodes on top of the pooling layer 



where _V_ is the weight matrix, **bh** is a bias vector, and _f_ is a non-linear activation. The dense layer naturally deals with variable sentence lengths by producing fixed size output vectors **z** , which are fed to the output layer for classification. The output layer defines a Bernoulli distribution: 



where sig refers to the sigmoid function, and **w** are the weights from the dense layer to the output layer and _b_ is a bias term. We fit the model by minimizing the cross-entropy between the predicted distributions _y_ ˆ _nθ_ = _p_ ( _yn|_ **s** _n, θ_ ) and the target distributions _yn_ (i.e., the gold labels).<sup>1</sup> 

### **Word Embedding and Fine-tuning** 

We avoid manual engineering of features and use word embeddings instead as the only features. We use pre-trained word embeddings to better initialize our models, and we fine-tune them for our task which turns out to be beneficial. We experimented with two types of pre-trained embeddings. **(** **_i_ ) Google Embedding:** We use the pre-trained 300dimensional Google word embeddings released by Mikolov et al. (2013a). These vectors were trained by their skip-gram model on part of Google news dataset containing about 100 billion words with a vocabulary size of 3 millions words.<sup>2</sup> **(** **_ii_ ) Crisis Embedding:** Since we work on disaster related tweets, which are quite different from news, we have also trained 300-dimensional _domain-specific_ word embeddings (vocabulary size of 20 millions) using the skip-gram model of word2vec from a large corpus of disaster related tweets. The corpus contains 57 _,_ 908 tweets and 9 _._ 4 million tokens. 

## **Datasets and Experimental Settings** 

We use data from multiple sources: (1) CrisisNLP<sup>3</sup> (Imran, Mitra, and Castillo 2016), (2) CrisisLex (Olteanu et al. 2014), and (3) AIDR (Imran et al. 2014). The first two sources have tweets posted during several humanitarian crises and labeled by paid workers. The AIDR data consists of tweets from several crises events labeled by volunteers. Table 1 shows statistics about the data. 

**Data Preprocessing:** We normalize all characters to their lower-cased forms and tokenize the tweets using the CMU TweetNLP tool (Gimpel et al. 2011). 

**Data Settings:** Given a particular event (e.g. Nepal earthquake), we use data from all other events plus _All others_ (see Table 1) as _out-of-event_ data. We divide each event dataset into train (70%), validation (10%) and test sets (20%) using ski-learn toolkit’s module (Pedregosa et al. 2011) which ensured that the class distribution remains reasonably balanced in each subset. 

> 1Other loss functions (e.g., hinge) yielded similar results. 

> 2https://code.google.com/p/word2vec/ 

> 3http://crisisnlp.qcri.org/ 

633 

|**EVENT**|**Nepal Earthquake**|**Typhoon Hagupit**|**California Earthquake**|**Cyclone PAM**|**All Others**|
|---|---|---|---|---|---|
|**Aff**ected individual|756|204|227|235|4624|
|**Don**ations and volunteering|1021|113|83|389|1752|
|<br>**Inf**rastructure and utilities|351|352|351|233|1972|
|**Sym**pathy and support|983|290|83|164|4546|
|<br>**Oth**er Useful Information|1505|732|1028|679|7709|
|**Not**related or irrelevant|6698|290|157|718|418|
|Grand Total|11314|1981|1929|2418|21021|



Table 1: Class distribution of events under consideration and all other crises (i.e. data used as part of _out-of-event_ data) 

**Feature Extraction:** We extracted word-level unigrams, bigrams and trigrams from tweets. They are converted to TFIDF vectors by considering each tweet as a document. Note that these features are used only in non-neural models. The neural models take tweets and their labels as input. For the SVM classifier, we implemented feature selection using a Chi-Squared test to improve the estimator’s accuracy scores. 

### **Models Settings** 

**Settings for Non-Neural Models:** We experimented with ( _i_ ) Support Vector Machine ( _SVM_ ), a discriminative maxmargin model; ( _ii_ ) Logistic Regression ( _LR_ ), a discriminative probabilistic model; and ( _iii_ ) Random Forest ( _RF_ ), an ensemble model of decision trees. We use the implementation from the scikit-learn toolkit (Pedregosa et al. 2011). **Settings for Convolutional Neural Network Models:** Our CNN model is implemented in Theano (Theano Development Team 2016). We train CNN models by optimizing the cross entropy using the gradient-based online learning algorithm ADADELTA (Zeiler 2012).<sup>4</sup> The learning rate and parameters were set to the values as suggested by the authors. The maximum number of epochs was set to 25. To avoid overfitting, we use dropout (Srivastava et al. 2014) of hidden units and _early stopping_ based on the accuracy on the validation set.<sup>5</sup> We experimented with _{_ 0 _._ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 5 _}_ dropout rates and _{_ 32 _,_ 64 _,_ 128 _}_ minibatch sizes. We limit the vocabulary ( _V_ ) to the most frequent _P_ % ( _P ∈{_ 80 _,_ 85 _,_ 90 _}_ ) words in the training corpus. The word vectors in _L_ were initialized using two types of pre-trained word embeddings: ( _i_ ) _Crisis embeddings (CNNI )_ and ( _ii_ ) _Google embeddings (CNNII )_ . 

We use rectified linear units (ReLU) for the activation functions ( _f_ ), _{_ 100 _,_ 150 _,_ 200 _}_ filters each having window size ( _L_ ) of _{_ 2 _,_ 3 _,_ 4 _}_ , pooling length ( _p_ ) of _{_ 2 _,_ 3 _,_ 4 _}_ , and _{_ 100 _,_ 150 _,_ 200 _}_ dense layer units. All the hyperparameters are tuned on the development set. 

## **Results** 

For each event under consideration, we train classifiers on the event data only, on the out-of-event data only, and on a combination of both. We evaluate them on the binary classification task. We merge all informative classes (Table 1) to create one general _Useful_ or _Relevant_ class. 

Table 2 presents the results of binary classification comparing several non-neural classifiers with the CNN-based 

> 4Other algorithms (SGD, Adagrad) gave similar results. 

> 5 _l_ 1 and _l_ 2 regularization on weights did not work well. 

Table 2: The AUC scores of non-neural and neural networkbased classifiers. _event_ , _out_ and _event+out_ represents the three different settings of the training data – event only, outof-event only and a concatenation of both. 

|SYS|RF|LR|SVM|CNN_I_|CNN_II_|
|---|---|---|---|---|---|
||N<br>|epal Eart<br>|hquake<br>|||
|B_event_|82.70|85.47|85.34|**86.89**|85.71|
|B_out_|74.63|78.58|78.93|81.14|78.72|
|B_event_+_out_|81.92|82.68|83.62|84.82|84.91|
||Cal<br>|ifornia E<br>|arthquake<br>|<br>||
|B_event_|75.64|79.57|78.95|**81.21**|78.82|
|B_out_|56.12|50.37|50.83|62.08|68.82|
|B_event_+_out_|77.34|75.50|74.67|78.32|79.75|
||T<br>|yphoon<br>|Hagupit<br>|||
|B_event_|82.05|82.36|78.08|87.83|**90.17**|
|B_out_|73.89|71.14|71.86|82.35|84.48|
|B_event_+_out_|78.37|75.90|77.64|85.84|87.71|
|||Cyclone|PAM|||
|B_event_|90.26|90.64|90.82|**94.17**|93.11|
|B_out_|80.24|79.22|80.83|85.62|87.48|
|B_event_+_out_|89.38|90.61|90.74|92.64|91.20|



classifier. CNNs performed better than all non-neural classifiers for all events under consideration. The improvements are substantial in the case of training with the out-of-event data only. In this case, CNN outperformed SVM by a margin of up to 11%. This result has a significant impact to a situation involving early hours of a crisis, where though a lot of data pours in, but performing data labeling using experts or volunteers to get a substantial amount of training data takes a lot of time. Our result shows that the CNN model handles this situation robustly by making use of the out-of-event data and provides reasonable performance. 

When trained using both the event and out-of-event data, CNNs also performed better than the non-neural models. Comparing different training settings, we saw a drop in performance when compared to the event-only training. This drop is because of the inherent variety in the crisis data. The large size of the out-of-event data down-weights the benefits of the event data, and skewed the probability distribution of the training data towards the out-of-event data. 

To summarize, the neural network based classifier outperformed non-neural classifiers in all data settings. The performance of the models trained on out-of-event data are (as 

634 

expected) lower than that in the other two training settings. However, in case of the CNN models, the results are reasonable to the extent that out-of-event data can be used to predict tweets informativeness when no event data is available. Comparing CNN _I_ with CNN _II_ , we did not see any system consistently better than the other. In the rest of our experiments below, we only consider the CNN _I_ trained on crisis embeddings because on the average, the crisis embeddings work slightly better than the rest of the alternatives. 

## **Related Work** 

Studies have analyzed how big crisis data can be useful during major disasters so as to gain insights into the situation as it unfolds (Acar and Muraki 2011). A number of systems have been developed to classify, extract, and summarize crisis-relevant information from social media; for a detailed survey see Imran, et al. (Imran et al. 2015). DNNs and word embeddings have been applied successfully to address NLP problems (Collobert et al. 2011; Caragea, Silvescu, and Tapia 2016). The emergence of tools such as word2vec (Mikolov et al. 2013b) and GloVe (Pennington, Socher, and Manning 2014) have enabled NLP researchers to learn word embeddings efficiently and use them to train better models. As opposed to previous works, we address the cold-start problem and show how out-of-event data can be used when there is not enough labeled training data at the beginning of a disaster. 

## **Conclusions** 

We addressed the problem of rapid classification of crisisrelated data posted on microblogging platforms like Twitter. Specifically, we addressed the challenges using deep neural network models the classification of crisis-related tweets and showed that one can reliably use out-of-event data for the classification of new event when no event-specific data is available. The performance of the classifiers degraded from event data when out-of-event training samples were added to training samples. Thus, we recommend using out-of-event training data during the first few hours of a disaster only after which the training data related to the event should be used. In the future, we will explore and perform experimentation to determine even more robust domain adaptation techniques. 

## **References** 

Acar, A., and Muraki, Y. 2011. Twitter for crisis communication: lessons learned from japan’s tsunami disaster. _International Journal of Web Based Communities_ 7(3):392–402. 

Caragea, C.; Silvescu, A.; and Tapia, A. H. 2016. Identifying informative messages in disaster events using convolutional neural networks. _International Conference on Information Systems for Crisis Response and Management_ . 

Collobert, R.; Weston, J.; Bottou, L.; Karlen, M.; Kavukcuoglu, K.; and Kuksa, P. 2011. Natural language processing (almost) from scratch. _The Journal of Machine Learning Research_ 12:2493– 2537. 

Gimpel, K.; Schneider, N.; O’Connor, B.; Das, D.; Mills, D.; Eisenstein, J.; Heilman, M.; Yogatama, D.; Flanigan, J.; and Smith, N. A. 2011. Part-of-speech tagging for twitter: Annotation, features, and 

experiments. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ , 42–47. 

Imran, M.; Castillo, C.; Lucas, J.; Meier, P.; and Vieweg, S. 2014. AIDR: Artificial intelligence for disaster response. In _Proceedings of the 23rd international conference on WWW_ , 159–162. 

Imran, M.; Castillo, C.; Diaz, F.; and Vieweg, S. 2015. Processing social media messages in mass emergency: a survey. _ACM Computing Surveys (CSUR)_ 47(4):67. 

Imran, M.; Mitra, P.; and Castillo, C. 2016. Twitter as a lifeline: Human-annotated twitter corpora for NLP of crisis-related messages. In _Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC)_ . 

Kalchbrenner, N.; Grefenstette, E.; and Blunsom, P. 2014. A convolutional neural network for modelling sentences. In _Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . 

Kim, Y. 2014. Convolutional neural networks for sentence classification. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 1746–1751. Lee, K.; Agrawal, A.; and Choudhary, A. 2013. Real-time disease surveillance using twitter data: demonstration on flu and cancer. In _Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining_ , 1474–1477. ACM. 

Mikolov, T.; Chen, K.; Corrado, G.; and Dean, J. 2013a. Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_ . 

Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J. 2013b. Distributed representations of words and phrases and their compositionality. In _Advances in Neural Information Processing Systems_ , 3111–3119. 

Nguyen, D. T.; Joty, S.; Imran, M.; Sajjad, H.; and Mitra, P. 2016. Applications of online deep learning for crisis response using social media information. _arXiv preprint arXiv:1610.01030_ . 

Olteanu, A.; Castillo, C.; Diaz, F.; and Vieweg, S. 2014. Crisislex: A lexicon for collecting and filtering microblogged communications in crises. In _In Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM” 14)_ , number EPFL-CONF-203561. 

Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; Vanderplas, J.; Passos, A.; Cournapeau, D.; Brucher, M.; Perrot, M.; and Duchesnay, E. 2011. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ 12:2825–2830. Pennington, J.; Socher, R.; and Manning, C. 2014. Glove: Global vectors for word representation. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 1532–1543. 

Rudra, K.; Banerjee, S.; Ganguly, N.; Goyal, P.; Imran, M.; and Mitra, P. 2016. Summarizing situational tweets in crisis scenario. In _Proceedings of the 27th ACM Conference on Hypertext and Social Media_ , 137–147. ACM. 

Srivastava, N.; Hinton, G.; Krizhevsky, A.; Sutskever, I.; and Salakhutdinov, R. 2014. Dropout: A simple way to prevent neural networks from overfitting. _Journal of Machine Learning Research_ 15:1929–1958. 

Theano Development Team. 2016. Theano: A Python framework for fast computation of mathematical expressions. _arXiv e-prints_ abs/1605.02688. 

Zeiler, M. D. 2012. ADADELTA: an adaptive learning rate method. _CoRR_ abs/1212.5701. 

635 

