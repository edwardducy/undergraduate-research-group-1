# A Survey on Multi-Task Learning

 Yu Zhang    Qiang Yang ††thanks: Y. Zhang is with Department of Computer Science and Engineering, Southern University of Science and Technology and Peng Cheng Laboratory. Q. Yang is with the Department of Computer Science and Engineering, Hong Kong University of Science and Technology.  
E-mail: yu.zhang.ust@gmail.com, qyang@cse.ust.hk  
Corresponding author: Yu Zhang.

###### Abstract

Multi-Task Learning (MTL) is a learning paradigm in machine learning and its aim is to leverage useful information contained in multiple related tasks to help improve the generalization performance of all the tasks. In this paper, we give a survey for MTL from the perspective of algorithmic modeling, applications and theoretical analyses. For algorithmic modeling, we give a definition of MTL and then classify different MTL algorithms into five categories, including feature learning approach, low-rank approach, task clustering approach, task relation learning approach and decomposition approach as well as discussing the characteristics of each approach. In order to improve the performance of learning tasks further, MTL can be combined with other learning paradigms including semi-supervised learning, active learning, unsupervised learning, reinforcement learning, multi-view learning and graphical models. When the number of tasks is large or the data dimensionality is high, we review online, parallel and distributed MTL models as well as dimensionality reduction and feature hashing to reveal their computational and storage advantages. Many real-world applications use MTL to boost their performance and we review representative works in this paper. Finally, we present theoretical analyses and discuss several future directions for MTL.

###### Index Terms: 

Multi-Task Learning, Machine Learning, Artificial Intelligence 

## I Introduction

Human can learn multiple tasks simultaneously and during this learning process, human can use the knowledge learned in a task to help the learning of another task. For example, according to our experience in learning to play tennis and squash, we find that the skill of playing tennis can help learn to play squash and vice versa. Inspired by such human learning ability, Multi-Task Learning (MTL) \[[1](#bib.bib1 "")\], a learning paradigm in machine learning, aims to learn multiple related tasks jointly so that the knowledge contained in a task can be leveraged by other tasks, with the hope of improving the generalization performance of all the tasks at hand.

At its early stage, an important motivation of MTL is to alleviate the data sparsity problem where each task has a limited number of labeled data. In the data sparsity problem, the number of labeled data in each task is insufficient to train an accurate learner, while MTL aggregates the labeled data in all the tasks in the spirit of data augmentation to obtain a more accurate learner for each task. From this perspective, MTL can help reuse existing knowledge and reduce the cost of manual labeling for learning tasks. When the era of “big data” comes in some areas such as computer vision and Natural Language Processing (NLP), it is found that deep MTL models can achieve better performance than their single-task counterparts. One reason that MTL is effective is that it utilizes more data from different learning tasks when compared with single-task learning. With more data, MTL can learn more robust and universal representations for multiple tasks and more powerful models, leading to better knowledge sharing among tasks, better performance of each task and low risk of overfitting in each task.

![Refer to caption](1707.08114v3/fig_diff_TL.png)

(a) MTL vs. transfer learning

![Refer to caption](1707.08114v3/fig_diff_MLL.png)

(b) MTL vs. multi-label learning/multi-output regression

![Refer to caption](1707.08114v3/fig_diff_MVL.png)

(c) MTL vs. multi-view learning

Fig. 1: Illustrations for differences between MTL and other learning paradigms.

MTL is related to other learning paradigms in machine learning, including transfer learning \[[2](#bib.bib2 "")\], multi-label learning \[[3](#bib.bib3 "")\] and multi-output regression. The setting of MTL is similar to that of transfer learning but with significant differences. In MTL, there is no distinction among different tasks and the objective is to improve the performance of all the tasks. However, transfer learning is to improve the performance of a target task with the help of source tasks, hence the target task plays a more important role than source tasks. In a word, MTL treats all the tasks equally but in transfer learning the target task attracts most attentions. From the perspective of the knowledge flow, flows of knowledge transfer in transfer learning are from source task(s) to the target task, but in multi-task learning, there are flows of knowledge sharing between any pair of tasks, which is illustrated in Fig. [1(a)](#S1.F1.sf1 "In Fig. 1 ‣ I Introduction ‣ A Survey on Multi-Task Learning"). Continual learning \[[4](#bib.bib4 "")\], in which tasks come sequentially, learns tasks one by one, while MTL is to learn multiple tasks together. In multi-label learning and multi-output regression, each data point is associated with multiple labels which can be categorical or numeric. If we treat each of all the possible labels as a task, multi-label learning and multi-output regression can be viewed in some sense as a special case of multi-task learning where different tasks always share the same data during both the training and testing phrases. On the one hand, such characteristic in multi-label learning and multi-output regression leads to different research issues from MTL. For example, the ranking loss, which enforces the scores (e.g., the classification probability) of labels associated with a data point to be larger than those of absent labels, can be used for multi-label learning but it does not fit MTL where different tasks possess different data. On the other hand, this characteristic in multi-label learning and multi-output regression is invalid in MTL problems. For example, in a MTL problem discussed in Section [II-G](#S2.SS7 "II-G Benchmark Datasets and Performance Comparison ‣ II MTL Models ‣ A Survey on Multi-Task Learning") where each task is to predict the disease symptom score of Parkinson for a patient based on 19 bio-medical features, different patients/tasks should not share the bio-medical data. In a word, multi-label learning and multi-output regression are different from multi-task learning as illustrated in Fig. [1(b)](#S1.F1.sf2 "In Fig. 1 ‣ I Introduction ‣ A Survey on Multi-Task Learning") and hence we will not survey literature on multi-label learning and multi-output regression. Moreover, multi-view learning is another learning paradigm in machine learning, where each data point is associated with multiple views, each of which consists of a set of features. Even though different views have different sets of features, all the views are used together to learn for the same task and hence multi-view learning belongs to single-task learning with multiple sets of features, which is different from MTL as shown in Fig. [1(c)](#S1.F1.sf3 "In Fig. 1 ‣ I Introduction ‣ A Survey on Multi-Task Learning").

Over past decades, MTL has attracted many attentions in the artificial intelligence and machine learning communities. Many MTL models have been devised and many MTL applications in other areas have been exploited. Moreover, many analyses have been conducted to study theoretical problems in MTL. This paper serves as a survey on MTL from the perspective of algorithmic modeling, applications and theoretical analyses. For algorithmic modeling, we first give a definition for MTL and then classify different MTL algorithms into five categories: feature learning approach which can be further categorized into feature transformation and feature selection approaches, low-rank approach, task clustering approach, task relation learning approach and decomposition approach. After that, we discuss the combination of MTL with other learning paradigms, including semi-supervised learning, active learning, unsupervised learning, reinforcement learning, multi-view learning and graphical models. To handle a large number of tasks, we review online, parallel and distributed MTL models. For data in a high-dimensional space, feature selection, dimensionality reduction and feature hashing are introduced as vital tools to process them. As a promising learning paradigm, MTL has many applications in various areas and here we briefly review its applications in computer vision, bioinformatics, health informatics, speech, NLP, web, etc. From the perspective of theoretical analyses on MTL, we review relevant works. At last, we discuss several future directions for MTL.11 1 For an introduction to MTL without technical details, please refer to \[[5](#bib.bib5 "")\].

## II MTL Models

In order to fully characterize MTL, we first give the definition of MTL.

###### Definition (Multi-Task Learning).

Given mm learning tasks {𝒯i}i\=1m\\{\\mathcal{T}\_{i}\\}\_{i=1}^{m} where all the tasks or a subset of them are related, multi-task learning aims to learn the mm tasks together to improve the learning of a model for each task 𝒯i\\mathcal{T}\_{i} by using the knowledge contained in all or some of other tasks.

Based on the definition of MTL, we focus on supervised learning tasks in this section since most MTL studies fall in this setting and for other types of tasks, we review them in the next section. In the setting of supervised learning tasks, a task 𝒯i\\mathcal{T}\_{i} is usually accompanied by a training dataset 𝒟i\\mathcal{D}\_{i} consisting of nin\_{i} training samples, i.e., 𝒟i\={𝐱ji,yji}j\=1ni\\mathcal{D}\_{i}=\\{\\mathbf{x}^{i}\_{j},y^{i}\_{j}\\}\_{j=1}^{n\_{i}}, where 𝐱ji∈ℝdi\\mathbf{x}^{i}\_{j}\\in\\mathbb{R}^{d\_{i}} is the jjth training instance in 𝒯i\\mathcal{T}\_{i} and yjiy^{i}\_{j} is its label. We denote by 𝐗i\\mathbf{X}^{i} the training data matrix for 𝒯i\\mathcal{T}\_{i}, i.e., 𝐗i\=(𝐱1i,…,𝐱nii)\\mathbf{X}^{i}=(\\mathbf{x}^{i}\_{1},\\ldots,\\mathbf{x}^{i}\_{n\_{i}}). When different tasks lie in the same feature space implying that did\_{i} equals djd\_{j} for any i≠ji\\neq j, this setting is the homogeneous-feature MTL, and otherwise it corresponds to heterogeneous-feature MTL. Without special explanation, the default MTL setting is the homogeneous-feature MTL. Here we need to distinguish the heterogeneous-feature MTL from the heterogeneous MTL. In \[[6](#bib.bib6 "")\], the heterogeneous MTL is considered to consist of different types of supervised tasks including classification and regression problems, and here we generalize it to a more general setting that the heterogeneous MTL consists of tasks with different types including supervised learning, unsupervised learning, semi-supervised learning, reinforcement learning, multi-view learning and graphical models. The opposite to the heterogeneous MTL is the homogeneous MTL which consists of tasks with only one type. In a word, the homogeneous and heterogeneous MTL differ in the type of learning tasks while the homogeneous-feature MTL is different from the heterogeneous-feature MTL in terms of the original feature representations. Similarly, without special explanation, the default MTL setting is the homogeneous MTL.

In order to characterize the relatedness in the definition of MTL, there are three issues to be addressed: when to share, what to share and how to share.

The ‘when to share’ issue is to make choices between single-task and multi-task models for a multi-task problem. Currently such decision is made by human experts and there are few works to study it. A simple solution is to formulate such decision as a model selection problem and then use model selection techniques, e.g., cross validation, to make decisions, but this solution is usually computational heavy and may require much more training data. An advanced solution we think is to use multi-task models which can degenerate to their single-task counterparts given some form of model parameters, for example, problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) presented in Section [II-H](#S2.SS8 "II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning") which can reduce to multiple single-task models with the learning of different tasks decoupled when a parameter 𝚺\\bm{\\Sigma} becomes diagonal. In this case, we can let the training data determine the form of 𝚺\\bm{\\Sigma} to make an implicit choice.

‘What to share’ needs to determine the form through which knowledge sharing among all the tasks could occur. Usually, there are three forms for ‘what to share’, including feature, instance and parameter. Feature-based MTL aims to learn common features among different tasks as a way to share knowledge. Instance-based MTL identifies useful data instances in a task for other tasks and then shares knowledge via the identified instances. Parameter-based MTL uses model parameters (e.g., coefficients in linear models or weights in deep models) in a task to help learn model parameters in other tasks in some ways, for example, the regularization. Existing MTL studies mainly focus on feature-based and parameter-based methods, and only a few works belong to the instance-based method. A representative instance-based method is the multi-task distribution matching method proposed in \[[7](#bib.bib7 "")\], which first estimates density ratios between probabilities that each instance as well as its label belongs to both its own task and a mixture of all the tasks and then uses all the weighted training data from all the tasks based on the estimated density ratios to learn model parameters for each task. Since the studies on instance-based MTL are few, we mainly review feature-based and parameter-based MTL models.

After determining ‘what to share’, ‘how to share’ specifies concrete ways to share knowledge among tasks. In feature-based MTL, there is a primary approach: feature learning approach. The feature learning approach focuses on learning common feature representations for multiple tasks based on shallow or deep models, where the learned common feature representation can be a subset or a transformation of the original feature representation. In parameter-based MTL, there are four main approaches: low-rank approach, task clustering approach, task relation learning approach and decomposition approach. The low-rank approach interprets the relatedness of multiple tasks as the low rankness of the parameter matrix of these tasks. The task clustering approach is to identify task clusters, each of which contains similar tasks. The task relation learning approach aims to learn quantitative relations between tasks from data automatically. The decomposition approach decomposes the model parameters of all the tasks into two or more components, which are penalized by different regularizers.

In summary, there are mainly five approaches in the feature-based and parameter-based MTL. In the following sections, we review these approaches in a chronological order to reveal the relations and evolutions among different models.

### II-A Feature Learning Approach

Since tasks are related, it is intuitive to assume that different tasks share a common feature representation based on the original features. One reason to learn common feature representations instead of directly using the original ones is that the original representation may not have enough expressive power for multiple tasks. With the training data in all the tasks, a more powerful representation can be learned for all the tasks and this representation can bring the improvement on the performance.

Based on the relationship between the original feature representation and the learned one, we can further classify this category into two sub-categories. The first sub-category is the feature transformation approach where the learned representation is a linear or nonlinear transformation of the original representation and in this approach, each feature in the learned representation is different from the original features. Different from this approach, the feature selection approach, the second sub-category, selects a subset of the original features as the learned representation and hence the learned representation is similar to the original one by eliminating useless features based on different criteria. In the following, we introduce these two approaches.

#### II-A1 Feature Transformation Approach

The multi-layer feedforward neural network \[[1](#bib.bib1 "")\], which belongs to the feature transformation approach, is one of the earliest model for multi-task learning. To see how the multi-layer feedforward neural network is constructed for MTL, in Figure [2](#S2.F2 "Fig. 2 ‣ II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning") we show an example with an input layer, a hidden layer and an output layer. The input layer receives training instances from all the tasks and the output layer has mm output units with one for each task. Here the outputs of the hidden layer can be viewed as the common feature representation learned for the mm tasks and the transformation from the original representation to the learned one depends on the weights connecting the input and hidden layers as well as the activation function adopted in the hidden units. Hence, if the activation function in the hidden layer is linear, then the transformation is a linear function and otherwise it is nonlinear. Compared with multi-layer feedforward neural networks used for single-task learning, the difference in the network architecture lies in the output layers where in single-task learning, there is only one output unit while in MTL, there are mm ones. In \[[8](#bib.bib8 "")\], the radial basis function network, which has only one hidden layer, is extended to MTL by greedily determining the structure of the hidden layer. Different from these neural network models, Silver et al. \[[9](#bib.bib9 "")\] propose a context-sensitive multi-task neural network which has only one output unit shared by different tasks but has a task-specific context as an additional input.

![Refer to caption](1707.08114v3/fig_MTLNN_sample.png)

Fig. 2: An example for the multi-task feedforward neural network with an input layer, a hidden layer and an output layer. 

Different from multi-layer feedforward neural networks which are connectionist models, the multi-task feature learning (MTFL) method \[[10](#bib.bib10 "")\] is formulated under the regularization framework with the objective function as

min𝐀,𝐔,𝐛\\displaystyle\\min\_{\\mathbf{A},\\mathbf{U},\\mathbf{b}}\\ 

∑i\=1m1ni​∑j\=1nil⁡(yji,(𝐚i)T​𝐔T​𝐱ji+bi)+λ​‖𝐀‖2,12\\displaystyle\\sum\_{i=1}^{m}\\frac{1}{n\_{i}}\\sum\_{j=1}^{n\_{i}}l(y^{i}\_{j},(\\mathbf{a}^{i})^{T}\\mathbf{U}^{T}\\mathbf{x}^{i}\_{j}+b\_{i})+\\lambda\\|\\mathbf{A}\\|\_{2,1}^{2}

s.t.\\displaystyle\\mathrm{s.t.}\\ 

𝐔𝐔T\=𝐈,\\displaystyle\\mathbf{U}\\mathbf{U}^{T}=\\mathbf{I},

(1)

where l⁡(⋅,⋅)l(\\cdot,\\cdot) denotes a loss function such as the hinge loss or square loss, 𝐛\=(b1,…,bm)T\\mathbf{b}=(b\_{1},\\ldots,b\_{m})^{T} is a vector of offsets in all the tasks, 𝐔∈ℝd×d\\mathbf{U}\\in\\mathbb{R}^{d\\times d} is a square transformation matrix, 𝐀∈ℝd×m\\mathbf{A}\\in\\mathbb{R}^{d\\times m} contains model parameters of all the tasks with its iith column 𝐚i\\mathbf{a}^{i} as model parameters for the iith task after the transformation, the ℓ2,1\\ell\_{2,1} norm of a matrix 𝐀\\mathbf{A} denoted by ‖𝐀‖2,1\\|\\mathbf{A}\\|\_{2,1} equals the sum of the ℓ2\\ell\_{2} norm of rows in 𝐀\\mathbf{A}, 𝐈\\mathbf{I} denotes an identity matrix with an appropriate size, and λ\\lambda is a positive regularization parameter. The first term in the objective function of problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) measures the empirical loss on the training sets of all the tasks and the second one is to enforce 𝐀\\mathbf{A} to be row-sparse via the ℓ2,1\\ell\_{2,1} norm which is equivalent to selecting features after the transformation, while the constraint enforces 𝐔\\mathbf{U} to be orthogonal. Different from the multi-layer feedforward neural network whose hidden representations may be redundant, the orthogonality of 𝐔\\mathbf{U} can prevent the MTFL method from it. As proved in \[[10](#bib.bib10 "")\], problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is equivalent to

min𝐖,𝐃,𝐛⁡L⁡(𝐖,𝐛)+λ​tr​(𝐖T​𝐃−1​𝐖)​s.t.𝐃⪰𝟎,tr⁡(𝐃)≤1,\\displaystyle\\min\_{\\mathbf{W},\\mathbf{D},\\mathbf{b}}L(\\mathbf{W},\\mathbf{b})+\\lambda\\mathrm{tr}(\\mathbf{W}^{T}\\mathbf{D}^{-1}\\mathbf{W})\\ \\mathrm{s.t.}\\ \\mathbf{D}\\succeq\\mathbf{0},\\ \\mathrm{tr}(\\mathbf{D})\\leq 1,

(2)

where L⁡(𝐖,𝐛)\=∑i\=1m1ni​∑j\=1nil⁡(yji,(𝐰i)T​𝐱ji+bi)L(\\mathbf{W},\\mathbf{b})=\\sum\_{i=1}^{m}\\frac{1}{n\_{i}}\\sum\_{j=1}^{n\_{i}}l(y^{i}\_{j},(\\mathbf{w}^{i})^{T}\\mathbf{x}^{i}\_{j}+b\_{i}) denotes the total training loss, tr⁡(⋅)\\mathrm{tr}(\\cdot) denotes the trace of a square matrix, 𝐰i\=𝐔𝐚i\\mathbf{w}^{i}=\\mathbf{U}\\mathbf{a}^{i} is the model parameter for 𝒯i\\mathcal{T}\_{i}, 𝐖\=(𝐰1,…,𝐰m)\\mathbf{W}=(\\mathbf{w}^{1},\\ldots,\\mathbf{w}^{m}), 𝟎\\mathbf{0} denotes a zero vector or matrix with an appropriate size, 𝐌−1\\mathbf{M}^{-1} for any square matrix 𝐌\\mathbf{M} denotes its inverse when it is nonsingular or otherwise its pseudo inverse, and 𝐁⪰𝐂\\mathbf{B}\\succeq\\mathbf{C} means that 𝐁−𝐂\\mathbf{B}-\\mathbf{C} is positive semidefinite. Based on this formulation, we can see that the MTFL method is to learn a feature covariance 𝐃\\mathbf{D} for all the tasks, which will be interpreted in Section [II-H](#S2.SS8 "II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning") from a probabilistic perspective. Given 𝐃\\mathbf{D}, the learning of different tasks can be decoupled and this can facilitate the parallel computing. When given 𝐖\\mathbf{W}, 𝐃\\mathbf{D} has an analytical solution as 𝐃\=(𝐖T​𝐖)12/tr⁡((𝐖T​𝐖)12)\\mathbf{D}=(\\mathbf{W}^{T}\\mathbf{W})^{\\frac{1}{2}}/\\mathrm{tr}\\left((\\mathbf{W}^{T}\\mathbf{W})^{\\frac{1}{2}}\\right) and by plugging this solution into problem ([2](#S2.E2 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we can see that the regularizer on 𝐖\\mathbf{W} is the squared trace norm. Then Argyriou et al. \[[11](#bib.bib11 "")\] extend problem ([2](#S2.E2 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) to a general formulation where the second term in the objective function becomes λ​tr​(𝐖T​f​(𝐃)​𝐖)\\lambda\\mathrm{tr}(\\mathbf{W}^{T}f(\\mathbf{D})\\mathbf{W}) with f⁡(𝐃)f(\\mathbf{D}) operating on the spectrum of 𝐃\\mathbf{D} and discuss the condition on f⁡(⋅)f(\\cdot) to make the whole problem convex.

Similar to the MTFL method, the multi-task sparse coding method \[[12](#bib.bib12 "")\] is to learn a linear transformation on features with the objective function formulated as

min𝐀,𝐔,𝐛⁡L⁡(𝐔𝐀,𝐛)​s.t.‖𝐚i‖1≤λ​∀i∈\[m\],‖𝐮j‖2≤1​∀j∈\[D\],\\displaystyle\\hskip-10.84006pt\\min\_{\\mathbf{A},\\mathbf{U},\\mathbf{b}}L(\\mathbf{UA},\\mathbf{b})\\ \\mathrm{s.t.}\\ \\|\\mathbf{a}^{i}\\|\_{1}\\leq\\lambda\\ \\forall i\\in\[m\],\\|\\mathbf{u}^{j}\\|\_{2}\\leq 1\\ \\forall j\\in\[D\],

(3)

where 𝐚i\\mathbf{a}^{i}, the iith column of 𝐀\\mathbf{A}, contains model parameters of the iith task, 𝐮j\\mathbf{u}^{j} is the jjth column in 𝐔\\mathbf{U}, \[c\]\[c\] for an integer cc denotes a set of integers from 1 to cc, ∥⋅∥1\\|\\cdot\\|\_{1} denotes the ℓ1\\ell\_{1} norm of a vector or matrix and equals the sum of the absolute value of its entries, and ∥⋅∥2\\|\\cdot\\|\_{2} denotes the ℓ2\\ell\_{2} norm of a vector. Here the transformation 𝐔∈ℝd×D\\mathbf{U}\\in\\mathbb{R}^{d\\times D} is also called the dictionary in sparse coding and shared by all the tasks. Compared with the MTFL method where 𝐔\\mathbf{U} in problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is a d×dd\\times d orthogonal matrix, 𝐔\\mathbf{U} in problem ([3](#S2.E3 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is overcomplete, which implies that DD is larger than dd, with each column having a bounded ℓ2\\ell\_{2} norm. Another difference is that in problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) 𝐀\\mathbf{A} is enforced to be row-sparse but in problem ([3](#S2.E3 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) it is only sparse via the first constraint. With a similar idea to the multi-task sparse coding method, Zhu et al. \[[13](#bib.bib13 "")\] propose a multi-task infinite support vector machine via the Indian buffet process and the difference is that in \[[13](#bib.bib13 "")\] the dictionary is sparse and model parameters are non-sparse. In \[[14](#bib.bib14 "")\], the spike and slab prior is used to learn sparse model parameters for multi-output regression problems where transformed features are induced by Gaussian processes and shared by different outputs.

Recently deep learning becomes popular due to its capacity to learn nonlinear features, which facilitates the learning of invariant features for multiple tasks, and hence many deep multi-task models belonging to this approach have been proposed with each task modeled by a deep neural network. Here we classify deep multi-task models in this approach into three main categories. The first category \[[15](#bib.bib15 ""), [16](#bib.bib16 ""), [17](#bib.bib17 ""), [18](#bib.bib18 ""), [19](#bib.bib19 "")\] is to learn a common feature representation for multiple tasks by sharing first several layers in a similar architecture to Fig. [2](#S2.F2 "Fig. 2 ‣ II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"). However, different from Fig. [2](#S2.F2 "Fig. 2 ‣ II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"), deep MTL models in this category have a large number of shared layers, which have general structures such as convolutional layers and pooling layers. Building on the first category, the second category is to use adversarial learning, which is inspired by generative adversarial networks, to learn a common feature representation for MTL as did in \[[20](#bib.bib20 ""), [21](#bib.bib21 "")\]. Specifically, there are three networks in such adversarial multi-task models, including a feature network NfN\_{f}, a classification network NcN\_{c} and a domain network NdN\_{d}. Based on NfN\_{f}, NcN\_{c} is to minimize the training loss for all the tasks, while NdN\_{d} aims to distinguish which task a data instance is from. The objective function of such models is usually formulated as

minθf,θc⁡max⁡∑i\=1mθd⁡1ni​∑j\=1ni(l⁡(yji,Nc​(Nf​(𝐱ji)))−lc​e​(dji,Nd​(Nf​(𝐱ji)))),\\min\_{\\theta\_{f},\\theta\_{c}}\\max\_{\\theta\_{d}}\\sum\_{i=1}^{m}\\frac{1}{n\_{i}}\\sum\_{j=1}^{n\_{i}}\\left(l(y^{i}\_{j},N\_{c}(N\_{f}(\\mathbf{x}^{i}\_{j})))-l\_{ce}(d^{i}\_{j},N\_{d}(N\_{f}(\\mathbf{x}^{i}\_{j})))\\right), where θf\\theta\_{f}, θc\\theta\_{c}, θd\\theta\_{d} denote parameters of three networks NfN\_{f}, NcN\_{c}, NdN\_{d}, respectively, dji∈{1,…,m}d^{i}\_{j}\\in\\{1,\\ldots,m\\} denotes the task index/label of 𝐱ji\\mathbf{x}^{i}\_{j}, and lc​e​(⋅,⋅)l\_{ce}(\\cdot,\\cdot) denotes the cross-entropy loss. Based on this minimax problem, NfN\_{f} is to minimize the training loss for all the tasks and maximize the cross-entropy loss to fool the domain network to make the learned feature representation indistinguishable to all the tasks. When there is no domain network, this category can reduce to the first category. Moreover, in \[[21](#bib.bib21 "")\], each task can learn its specific feature representation to increase the expressive power of the whole model. The last category is to learn different but related feature representations for different tasks with the cross-stitch network \[[22](#bib.bib22 "")\] as a representative model. Specifically, given two tasks AA and BB with an identical network architecture, xAi,jx^{i,j}\_{A} (xBi,jx^{i,j}\_{B}) denotes the hidden feature outputted by the jjth unit of the iith hidden layer for task AA (BB). Then we can define the cross-stitch operation on xAi,jx^{i,j}\_{A} and xBi,jx^{i,j}\_{B} as (x\~Ai,jx\~Bi,j)\=(αA​AαA​BαB​AαB​B)​(xAi,jxBi,j)\\left(\\begin{array}\[\]{c}\\tilde{x}^{i,j}\_{A}\\\\ \\tilde{x}^{i,j}\_{B}\\end{array}\\right)=\\left(\\begin{array}\[\]{cc}\\alpha\_{AA}&\\alpha\_{AB}\\\\ \\alpha\_{BA}&\\alpha\_{BB}\\end{array}\\right)\\left(\\begin{array}\[\]{ccc}x^{i,j}\_{A}\\\\ x^{i,j}\_{B}\\end{array}\\right), where x\~Ai,j\\tilde{x}^{i,j}\_{A} and x\~Bi,j\\tilde{x}^{i,j}\_{B} are new hidden features after learning the two tasks jointly. When both αA​B\\alpha\_{AB} and αB​A\\alpha\_{BA} equal 0, training the two networks jointly is equivalent to training them independently. The network architecture of the cross-stitch network is shown in Fig. [3](#S2.F3 "Fig. 3 ‣ II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"). Here matrix 𝜶\\bm{\\alpha}, which is defined as 𝜶≡(αA​AαA​BαB​AαB​B)\\bm{\\alpha}\\equiv\\left(\\begin{array}\[\]{cc}\\alpha\_{AA}&\\alpha\_{AB}\\\\ \\alpha\_{BA}&\\alpha\_{BB}\\end{array}\\right), encodes feature-level task relations between the two tasks and it can be learned via the backpropagation method.

![Refer to caption](1707.08114v3/fig_CSN.png)

Fig. 3: The architecture for the cross-stitch network.

#### II-A2 Feature Selection Approach

One way to do feature selection in MTL is to use the ℓp,q\\ell\_{p,q} norm denoted by ‖𝐖‖p,q≡‖(‖𝐰1‖p,…,‖𝐰d‖p)‖q\\|\\mathbf{W}\\|\_{p,q}\\equiv\\|(\\|\\mathbf{w}\_{1}\\|\_{p},\\ldots,\\|\\mathbf{w}\_{d}\\|\_{p})\\|\_{q}, where 𝐰i\\mathbf{w}\_{i} denotes the iith row of 𝐖\\mathbf{W} and ∥⋅∥p\\|\\cdot\\|\_{p} denotes the ℓp\\ell\_{p} norm of a vector, to achieve the group sparsity. Obozinski et al. \[[23](#bib.bib23 "")\] are among the first to study the multi-task feature selection (MTFS) problem based on the ℓ2,1\\ell\_{2,1} norm with the objective function formulated as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​‖𝐖‖2,1.\\min\_{\\mathbf{W},\\mathbf{b}}\\ \\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\|\\mathbf{W}\\|\_{2,1}.

(4)

The regularizer on 𝐖\\mathbf{W} in problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is to enforce 𝐖\\mathbf{W} to be row-sparse, which in turn helps select important features. In \[[23](#bib.bib23 "")\], a path-following algorithm is proposed to solve problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and then Liu et al. \[[24](#bib.bib24 "")\] employ an optimal first-order optimization method to solve it. Compared with problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we can see that problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is similar to the MTFL method without learning the transformation 𝐔\\mathbf{U}. Lee et al. \[[25](#bib.bib25 "")\] propose a weighted ℓ2,1\\ell\_{2,1} norm for multi-task feature selection where the weights can be learned as well and problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is extended in \[[26](#bib.bib26 "")\] to a general case where feature groups can overlap with each other. In order to make problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) more robust to outliers, a square-root loss function is investigated in \[[27](#bib.bib27 "")\]. Moreover, in order to make speedup, a safe screening method is proposed in \[[28](#bib.bib28 "")\] to filter out useless features corresponding to zero rows in 𝐖\\mathbf{W} before optimizing problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). Liu et al. \[[29](#bib.bib29 "")\] propose to use the ℓ∞,1\\ell\_{\\infty,1} norm to select features with the objective function formulated as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​‖𝐖‖∞,1.\\min\_{\\mathbf{W},\\mathbf{b}}\\ \\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\|\\mathbf{W}\\|\_{\\infty,1}.

(5)

A block coordinate descent method is proposed to solve problem ([5](#S2.E5 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). In general, we can use the ℓp,q\\ell\_{p,q} norm to select features for MTL.

In order to attain a more sparse subset of features, Gong et al. \[[30](#bib.bib30 "")\] propose a capped-ℓp,1\\ell\_{p,1} regularizer for multi-task feature selection where p\=1p=1 or 2 and the objective function is formulated as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​∑i\=1dmin⁡(‖𝐰i‖p,θ),\\min\_{\\mathbf{W},\\mathbf{b}}\\ \\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\sum\_{i=1}^{d}\\min(\\|\\mathbf{w}\_{i}\\|\_{p},\\theta),

(6)

where 𝐰i\\mathbf{w}\_{i} denotes the iith row of 𝐖\\mathbf{W}. With a given threshold θ\\theta, the capped-ℓp,1\\ell\_{p,1} regularizer (i.e., the second term in problem ([6](#S2.E6 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))) focuses on rows with smaller ℓp\\ell\_{p} norms than θ\\theta, which is more likely to be sparse. When θ\\theta becomes large enough, the capped-ℓp,1\\ell\_{p,1} regularizer becomes ‖𝐖‖p,1\\|\\mathbf{W}\\|\_{p,1} and hence problem ([6](#S2.E6 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) degenerates to problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) or ([5](#S2.E5 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) when pp equals 2 or ∞\\infty.

Lozano and Swirszcz \[[31](#bib.bib31 "")\] propose a multi-level Lasso for MTL where the (j,i)(j,i)th entry in the parameter matrix 𝐖\\mathbf{W} is defined as wj​i\=θj​w^j​iw\_{ji}=\\theta\_{j}\\hat{w}\_{ji}. When θj\\theta\_{j} is equal to 0, wj​iw\_{ji} becomes 0 for i∈\[m\]i\\in\[m\] and hence the jjth feature is not selected by the model. In this sense, θj\\theta\_{j} controls the global sparsity for the jjth feature among the mm tasks. Moreover, when w^j​i\\hat{w}\_{ji} becomes 0, wj​iw\_{ji} is also 0 for ii only, implying that the jjth feature is not useful for task 𝒯i\\mathcal{T}\_{i}, and so w^j​i\\hat{w}\_{ji} is a local indicator for the sparsity in task 𝒯j\\mathcal{T}\_{j}. Based on these observations, θj\\theta\_{j} and w^j​i\\hat{w}\_{ji} are expected to be sparse, leading to the objective function formulated as

min𝜽,𝐖^,𝐛⁡L⁡(𝐖,𝐛)+λ1​‖𝜽‖1+λ2​‖𝐖^‖1​s.t.wj​i\=θj​w^j​i,θj≥0,\\displaystyle\\hskip-10.84006pt\\min\_{\\bm{\\theta},\\mathbf{\\hat{W}},\\mathbf{b}}L(\\mathbf{W},\\mathbf{b})+\\lambda\_{1}\\|\\bm{\\theta}\\|\_{1}+\\lambda\_{2}\\|\\mathbf{\\hat{W}}\\|\_{1}\\ \\mathrm{s.t.}\\ w\_{ji}=\\theta\_{j}\\hat{w}\_{ji},\\theta\_{j}\\geq 0,

(7)

where 𝜽\=(θ1,…,θd)T\\bm{\\theta}=(\\theta\_{1},\\ldots,\\theta\_{d})^{T}, 𝐖^\=(𝐰^1,…,𝐰^m)\\mathbf{\\hat{W}}=(\\mathbf{\\hat{w}}^{1},\\ldots,\\mathbf{\\hat{w}}^{m}), and the nonnegative constraint on θj\\theta\_{j} is to keep the model identifiability. It has been proved in \[[31](#bib.bib31 "")\] that problem ([7](#S2.E7 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) leads to a regularizer ∑j\=1d‖𝐰j‖1\\sum\_{j=1}^{d}\\sqrt{\\|\\mathbf{w}\_{j}\\|\_{1}}, the square root of the ℓ1,12\\ell\_{1,\\frac{1}{2}} norm regularization. Moreover, Wang et al. \[[32](#bib.bib32 "")\] extend problem ([7](#S2.E7 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) to a general situation where the regularizer becomes λ1​∑i\=1m‖𝐰^i‖pp+λ2​‖𝜽‖qq\\lambda\_{1}\\sum\_{i=1}^{m}\\|\\mathbf{\\hat{w}}^{i}\\|\_{p}^{p}+\\lambda\_{2}\\|\\bm{\\theta}\\|\_{q}^{q}. By utilizing a priori information describing the task relations in a hierarchical structure, Han et al. \[[33](#bib.bib33 "")\] propose a multi-component product based decomposition for wi​jw\_{ij} where the number of components in the decomposition can be arbitrary instead of only 2 in \[[31](#bib.bib31 ""), [32](#bib.bib32 "")\]. Similar to \[[31](#bib.bib31 "")\], Jebara \[[34](#bib.bib34 "")\] proposes to learn a binary indicator vector to do multi-task feature selection based on the maximum entropy discrimination formalism.

Similar to \[[33](#bib.bib33 "")\] where a priori information is given to describe task relations in a hierarchical/tree structure, Kim and Xing \[[35](#bib.bib35 "")\] utilize the given tree structure to design a regularizer on 𝐖\\mathbf{W} as f⁡(𝐖)\=∑i\=1d∑v∈Vλv​‖𝐰i,Gv‖2f(\\mathbf{W})=\\sum\_{i=1}^{d}\\sum\_{v\\in V}\\lambda\_{v}\\|\\mathbf{w}\_{i,G\_{v}}\\|\_{2}, where VV denotes the set of nodes in the given tree structure, GvG\_{v} denotes the set of leaf nodes (i.e., tasks) in a sub-tree rooted at node vv, and 𝐰i,Gv\\mathbf{w}\_{i,G\_{v}} denotes a subvector of the iith row of 𝐖\\mathbf{W} indexed by GvG\_{v}. This regularizer not only enforces each row of 𝐖\\mathbf{W} to be sparse as the ℓ2,1\\ell\_{2,1} norm did in problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), but also induces the sparsity in subsets of each row in 𝐖\\mathbf{W} based on the tree structure.

Different from conventional multi-task feature selection methods which assume that different tasks share a set of original features, Zhou et al. \[[36](#bib.bib36 "")\] consider a different scenario where useful features in different tasks have no overlapping. In order to achieve this, an exclusive Lasso model is proposed with the objective function formulated as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​‖𝐖‖1,22,\\min\_{\\mathbf{W},\\mathbf{b}}\\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\|\\mathbf{W}\\|\_{1,2}^{2}, where the regularizer is the squared ℓ1,2\\ell\_{1,2} norm on 𝐖\\mathbf{W}.

Another way to select common features for MTL is to use sparse priors to design probabilistic or Bayesian models. For ℓp,1\\ell\_{p,1}-regularized multi-task feature selection, Zhang et al. \[[37](#bib.bib37 "")\] propose a probabilistic interpretation where the ℓp,1\\ell\_{p,1} regularizer corresponds to a generalized normal prior: wj​i∼𝒢𝒩(⋅|0,ρj,p)w\_{ji}\\sim\\mathcal{GN}(\\cdot|0,\\rho\_{j},p), where ⋅\\cdot denotes a (random) variable when we do not want to introduce it explicitly. Based on this interpretation, Zhang et al. \[[37](#bib.bib37 "")\] further propose a probabilistic framework for multi-task feature selection, in which task relations and outlier tasks can be identified, based on the matrix-variate generalized normal prior.

In \[[38](#bib.bib38 "")\], a generalized horseshoe prior is proposed to do feature selection for MTL as:

ℙ⁡(𝐰i)\=∫∏j\=1d𝒩⁡(wj​i|0,uj​ivj​i)​𝒩​(𝐮i|0,ρ2​𝐂)​𝒩​(𝐯i|0,γ2​𝐂)​d​𝐮i​d​𝐯i,\\mathbb{P}(\\mathbf{w}^{i})=\\int\\prod\_{j=1}^{d}\\mathcal{N}(w\_{ji}|0,\\frac{u\_{ji}}{v\_{ji}})\\mathcal{N}(\\mathbf{u}^{i}|0,\\rho^{2}\\mathbf{C})\\mathcal{N}(\\mathbf{v}^{i}|0,\\gamma^{2}\\mathbf{C})\\mathrm{d}\\mathbf{u}^{i}\\mathrm{d}\\mathbf{v}^{i}, where 𝒩(⋅|𝐦,𝝈)\\mathcal{N}(\\cdot|\\mathbf{m},\\bm{\\sigma}) denotes a univariate or multivariate normal distribution with 𝐦\\mathbf{m} as the mean and 𝝈\\bm{\\sigma} as the variance or covariance matrix, uj​iu\_{ji} and vj​iv\_{ji} are the jjth entries in 𝐮i\\mathbf{u}^{i} and 𝐯i\\mathbf{v}^{i}, respectively, and ρ,γ\\rho,\\gamma are hyperparameters. Here 𝐂\\mathbf{C} shared by all the tasks denotes the feature correlation matrix to be learned from data and it encodes an assumption that different tasks share identical feature correlations. When 𝐂\\mathbf{C} becomes an identity matrix which means that features are independent, this prior degenerates to the horseshoe prior.

Hernández-Lobato et al. \[[39](#bib.bib39 "")\] propose a probabilistic model based on the horseshoe prior as

ℙ⁡(wj​i)\=\\displaystyle\\mathbb{P}(w\_{ji})=

\[π​(wj​i)ηj​i​δ01−ηj​i\]zj​\[π​(wj​i)τj​i​δ01−τj​i\]ωi​(1−zj)\\displaystyle\\left\[\\pi(w\_{ji})^{\\eta\_{ji}}\\delta\_{0}^{1-\\eta\_{ji}}\\right\]^{z\_{j}}\\left\[\\pi(w\_{ji})^{\\tau\_{ji}}\\delta\_{0}^{1-\\tau\_{ji}}\\right\]^{\\omega\_{i}(1-z\_{j})}

\[π​(wj​i)γj​δ01−γj\](1−ωi)​(1−zj),\\displaystyle\\left\[\\pi(w\_{ji})^{\\gamma\_{j}}\\delta\_{0}^{1-\\gamma\_{j}}\\right\]^{(1-\\omega\_{i})(1-z\_{j})},

(8)

where δ0\\delta\_{0} is the probability mass function at zero and π⁡(⋅)\\pi(\\cdot) denotes the density function of non-zero coefficients. In Eq. ([8](#S2.E8 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), zjz\_{j} indicates whether feature jj is an outlier (zj\=1z\_{j}=1) or not (zj\=0z\_{j}=0) and ωi\\omega\_{i} indicates whether task 𝒯i\\mathcal{T}\_{i} is an outlier (ωi\=1\\omega\_{i}=1) or not (ωi\=0\\omega\_{i}=0). Moreover, ηj​i\\eta\_{ji} and τj​i\\tau\_{ji} indicate whether feature jj is relevant for the prediction in 𝒯i\\mathcal{T}\_{i} (ηj​i,τj​i\=1\\eta\_{ji},\\tau\_{ji}=1) or not (ηj​i,τj​i\=0\\eta\_{ji},\\tau\_{ji}=0), and γj\\gamma\_{j} indicates whether a non-outlier feature jj is relevant (γj\=1\\gamma\_{j}=1) for the prediction or not (γj\=0\\gamma\_{j}=0) in all non-outlier tasks. Based on the above definitions, the three terms in the right-hand side of Eq. ([8](#S2.E8 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) specify probability density functions of wj​iw\_{ji} based on different situations of features and tasks. So this model can also handle outlier tasks but in a different way from \[[37](#bib.bib37 "")\].

#### II-A3 Comparison between Two Sub-categories

The two sub-categories have different characteristics where the feature transformation approach learns a transformation of the original features as the new representation but the feature selection approach selects a subset of the original features as the new representation for all the tasks. Based on the characteristics of those two approaches, the feature selection approach can be viewed as a special case of the feature transformation approach when the transformation matrix is a diagonal 0/10/1 matrix where the diagonal entries with value 1 correspond to the selected features. By selecting a subset of the original features as the new representation, the feature selection approach has a better interpretability.

### II-B Low-Rank Approach

The relatedness among multiple tasks can imply the low-rank of 𝐖\\mathbf{W}, leading to the low-rank approach. For example, if the iith, jjth and kkth tasks are related in that the model parameter 𝐰i\\mathbf{w}^{i} of the iith task is a linear combination of those of the other two tasks, then it is easy to show that the rank of 𝐖\\mathbf{W} is at most m−1m-1 and hence of low rank. From this perspective, the more the relatedness is, the lower the rank of 𝐖\\mathbf{W} is.

Ando and Zhang \[[40](#bib.bib40 "")\] assume that model parameters of different tasks share a low-rank subspace in part and specifically, 𝐰i\\mathbf{w}^{i} takes the following form as

𝐰i\=𝐮i+𝚯T​𝐯i.\\mathbf{w}^{i}=\\mathbf{u}^{i}+\\bm{\\Theta}^{T}\\mathbf{v}^{i}.

(9)

Here 𝚯∈ℝh×d\\bm{\\Theta}\\in\\mathbb{R}^{h\\times d} is the shared low-rank subspace by multiple tasks where h<dh\<d. Then we can write in a matrix form as 𝐖\=𝐔+𝚯T​𝐕\\mathbf{W}=\\mathbf{U}+\\bm{\\Theta}^{T}\\mathbf{V}. Based on the form of 𝐖\\mathbf{W}, the objective function proposed in \[[40](#bib.bib40 "")\] is formulated as

min𝐔,𝐕,𝚯,𝐛⁡L⁡(𝐔+𝚯T​𝐕,𝐛)+λ​‖𝐔‖F2​s.t.𝚯​𝚯T\=𝐈,\\displaystyle\\min\_{\\mathbf{U},\\mathbf{V},\\bm{\\Theta},\\mathbf{b}}L(\\mathbf{U}+\\bm{\\Theta}^{T}\\mathbf{V},\\mathbf{b})+\\lambda\\|\\mathbf{U}\\|\_{F}^{2}\\ \\ \\mathrm{s.t.}\\ \\bm{\\Theta}\\bm{\\Theta}^{T}=\\mathbf{I},

(10)

where ∥⋅∥F\\|\\cdot\\|\_{F} denotes the Frobenius norm. The orthonormal constraint on 𝚯\\bm{\\Theta} in problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) makes the subspace non-redundant. When λ\\lambda is large enough, the optimal 𝐔\\mathbf{U} can become a zero matrix and hence problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is very similar to problem ([1](#S2.E1 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) except that there is no regularization on 𝐕\\mathbf{V} in problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and that 𝚯\\bm{\\Theta} has a smaller number of rows than columns. Chen et al. \[[41](#bib.bib41 "")\] generalize problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) as

min𝐔,𝐕,𝚯,𝐛\\displaystyle\\hskip-19.5132pt\\min\_{\\mathbf{U},\\mathbf{V},\\bm{\\Theta},\\mathbf{b}}

L⁡(𝐖,𝐛)+λ1​‖𝐔‖F2+λ2​‖𝐖‖F2\\displaystyle L(\\mathbf{W},\\mathbf{b})+\\lambda\_{1}\\|\\mathbf{U}\\|\_{F}^{2}+\\lambda\_{2}\\|\\mathbf{W}\\|\_{F}^{2}

s.t.\\displaystyle\\mathrm{s.t.}

𝐖\=𝐔+𝚯T​𝐕,𝚯​𝚯T\=𝐈.\\displaystyle\\mathbf{W}=\\mathbf{U}+\\bm{\\Theta}^{T}\\mathbf{V},\\ \\bm{\\Theta}\\bm{\\Theta}^{T}=\\mathbf{I}.

(11)

When setting λ2\\lambda\_{2} to be 0, problem ([11](#S2.E11 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) reduces to problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). Even though problem ([11](#S2.E11 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is non-convex, with some convex relaxation technique, it can be relaxed to the following convex problem as

min𝐖,𝐛,𝐌⁡L⁡(𝐖,𝐛)+λ​tr​(𝐖T​(𝐌+η​𝐈)−1​𝐖)​s.t.tr⁡(𝐌)\=h𝟎⪯𝐌⪯𝐈,\\displaystyle\\hskip-10.84006pt\\min\_{\\mathbf{W},\\mathbf{b},\\mathbf{M}}L(\\mathbf{W},\\mathbf{b})+\\lambda\\mathrm{tr}\\big(\\mathbf{W}^{T}\\left(\\mathbf{M}+\\eta\\mathbf{I}\\right)^{-1}\\mathbf{W}\\big)\\ \\mathrm{s.t.}\\ {\\mathrm{tr}(\\mathbf{M})=h\\atop\\mathbf{0}\\preceq\\mathbf{M}\\preceq\\mathbf{I}},

(12)

where η\=λ2/λ1\\eta=\\lambda\_{2}/\\lambda\_{1} and λ\=λ1​η​(η+1)\\lambda=\\lambda\_{1}\\eta(\\eta+1). One advantage of problem ([12](#S2.E12 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) over problem ([11](#S2.E11 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is that the global optimum of the convex problem ([12](#S2.E12 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is much easier to be obtained than that of the non-convex problem ([11](#S2.E11 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). Compared with the alternative objective function ([2](#S2.E2 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) in the MTFL method, problem ([12](#S2.E12 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) has a similar formulation where 𝐌\\mathbf{M} models the feature covariance for all the tasks. Problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is extended in \[[42](#bib.bib42 "")\] to a general case where different 𝐰i\\mathbf{w}^{i}’s lie in a manifold instead of a subspace. Moreover, in \[[43](#bib.bib43 "")\], a latent variable model is proposed for 𝐖\\mathbf{W} with the same decomposition as Eq. ([9](#S2.E9 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and it can provide a framework for MTL by modeling more cases than problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) such as task clustering, sharing sparse representation, duplicate tasks and evolving tasks.

It is well known that using the trace norm as a regularizer can make a matrix have low rank and hence this regularization is suitable for MTL. Specifically, an objective function with the trace norm regularization is proposed in \[[44](#bib.bib44 "")\] as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​‖𝐖‖S⁡(1),\\min\_{\\mathbf{W},\\mathbf{b}}\\ \\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\|\\mathbf{W}\\|\_{S(1)},

(13)

where μi​(𝐖)\\mu\_{i}(\\mathbf{W}) denotes the iith smallest singular value of 𝐖\\mathbf{W} and ‖𝐖‖S⁡(1)\=∑i\=1min⁡(m,d)μi​(𝐖)\\|\\mathbf{W}\\|\_{S(1)}=\\sum\_{i=1}^{\\min(m,d)}\\mu\_{i}(\\mathbf{W}) denotes the trace norm of matrix 𝐖\\mathbf{W}. Based on the trace norm, Han and Zhang \[[45](#bib.bib45 "")\] propose a capped trace regularizer with the objective function formulated as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​∑i\=1min⁡(m,d)min⁡(μi​(𝐖),θ).\\min\_{\\mathbf{W},\\mathbf{b}}\\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\sum\_{i=1}^{\\min(m,d)}\\min(\\mu\_{i}(\\mathbf{W}),\\theta).

(14)

With the use of the threshold θ\\theta, the capped trace regularizer only penalizes small singular values of 𝐖\\mathbf{W}, which is related to the determination of the rank of 𝐖\\mathbf{W}. When θ\\theta is large enough, the capped trace regularizer will become the trace norm and hence problem ([14](#S2.E14 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) will reduce to problem ([13](#S2.E13 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). Moreover, a spectral kk-support norm is proposed in \[[46](#bib.bib46 "")\] as an improvement over the trace norm regularization.

The trace norm regularization has been extended to regularize model parameters in deep multi-task models. Specifically, the weights in the last several fully connected layers of deep multi-task neural networks can be viewed as the parameters of learners for all the tasks. In this view, the weights connecting two consecutive layers for one task can be organized in a matrix and hence the weights of all the tasks can form a tensor. Based on such tensor representations, several tensor trace norms, which are based on the trace norm, are used in \[[47](#bib.bib47 "")\] as regularizers to identify the low-rank structure of the parameter tensor.

### II-C Task Clustering Approach

The task clustering approach assumes that different tasks form several clusters, each of which consists of similar tasks. As indicated by its name, this approach has a close connection with clustering algorithms and it can be viewed as an extension of clustering algorithms to the task level while the conventional clustering algorithms are on the data level.

Thrun and Sullivan \[[48](#bib.bib48 "")\] propose the first task clustering algorithm by using a weighted nearest neighbor classifier for each task, where the initial weights to define the weighted Euclidean distance are learned by minimizing pairwise within-class distances and maximizing pairwise between-class distances simultaneously within each task. Then a task transfer matrix 𝐀\\mathbf{A} is defined with its (i,j)(i,j)th entry ai​ja\_{ij} recording the generalization accuracy obtained for task 𝒯i\\mathcal{T}\_{i} by using task 𝒯j\\mathcal{T}\_{j}’s distance metric. Based on 𝐀\\mathbf{A}, mm tasks can be grouped into rr clusters {𝒞i}i\=1r\\{\\mathcal{C}\_{i}\\}\_{i=1}^{r} by maximizing ∑t\=1r1|𝒞t|​∑i,j∈𝒞tai​j\\sum\_{t=1}^{r}\\frac{1}{|\\mathcal{C}\_{t}|}\\sum\_{i,j\\in\\mathcal{C}\_{t}}a\_{ij}, where |⋅||\\cdot| denotes the cardinality of a set. After obtaining the cluster structure among all the tasks, the training data of tasks in a cluster will be pooled together to learn the final weighted nearest neighbor classifier. This approach has been extended to an iterative learning process \[[49](#bib.bib49 "")\] in a similar way to kk-means clustering.

Bakker and Heskes \[[50](#bib.bib50 "")\] propose a multi-task Bayesian neural network model with the network structure similar to Fig. [2](#S2.F2 "Fig. 2 ‣ II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning") where input-to-hidden weights are shared by all the tasks but hidden-to-output weights are task-specific. By defining 𝐰i\\mathbf{w}^{i} as the vector of hidden-to-output weights for task 𝒯i\\mathcal{T}\_{i}, the multi-task Bayesian neural network assigns a mixture of Gaussian prior to it: 𝐰i∼∑j\=1rπj𝒩(⋅|𝐦j,𝚺j)\\mathbf{w}^{i}\\sim\\sum\_{j=1}^{r}\\pi\_{j}\\mathcal{N}(\\cdot|\\mathbf{m}\_{j},\\bm{\\Sigma}\_{j}), where πj\\pi\_{j}, 𝐦j\\mathbf{m}\_{j} and 𝚺j\\bm{\\Sigma}\_{j} specify the prior, the mean and the covariance in the jjth cluster. For tasks in a cluster, they will share a Gaussian distribution. When rr equals 1, this model degenerates to a case where model parameters of different tasks share a prior, which is similar to several Bayesian MTL models such as \[[51](#bib.bib51 ""), [52](#bib.bib52 ""), [53](#bib.bib53 "")\] that are based on Gaussian processes and tt processes.

Xue et al. \[[54](#bib.bib54 "")\] deploy the Dirichlet process to do clustering on task level. Specifically, it defines the prior on 𝐰i\\mathbf{w}^{i} as

𝐰i∼G,G∼𝒟​𝒫​(α,G0)​∀i∈\[m\],\\mathbf{w}^{i}\\sim G,\\ G\\sim\\mathcal{DP}(\\alpha,G\_{0})\\ \\forall i\\in\[m\], where 𝒟​𝒫​(α,G0)\\mathcal{DP}(\\alpha,G\_{0}) denotes a Dirichlet process with α\\alpha as a positive scaling parameter and G0G\_{0} a base distribution. To see the clustering effect, by integrating out GG, the conditional distribution of 𝐰i\\mathbf{w}^{i}, given model parameters of other tasks 𝐖−i\={⋯,𝐰i−1,𝐰i+1,⋯}\\mathbf{W}\_{-i}=\\{\\cdots,\\mathbf{w}^{i-1},\\mathbf{w}^{i+1},\\cdots\\}, is

ℙ⁡(𝐰i|𝐖−i,α,G0)\=αm−1+α​G0+1m−1+α​∑j\=1,j≠imδ𝐰j,\\mathbb{P}(\\mathbf{w}^{i}|\\mathbf{W}\_{-i},\\alpha,G\_{0})=\\frac{\\alpha}{m-1+\\alpha}G\_{0}+\\frac{1}{m-1+\\alpha}\\sum\_{j=1,j\\neq i}^{m}\\delta\_{\\mathbf{w}^{j}}, where δ𝐰j\\delta\_{\\mathbf{w}^{j}} denotes the distribution concentrated at a single point 𝐰j\\mathbf{w}^{j}. So 𝐰i\\mathbf{w}^{i} can be equal to either 𝐰j\\mathbf{w}^{j} (j≠ij\\neq i) with probability 1m−1+α\\frac{1}{m-1+\\alpha}, which corresponds to the case that those two tasks lie in the same cluster, or a new sample from G0G\_{0} with probability αm−1+α\\frac{\\alpha}{m-1+\\alpha}, which is the case that task 𝒯i\\mathcal{T}\_{i} forms a new task cluster. When α\\alpha is large, the chance to form a new task cluster is large and so α\\alpha affects the number of task clusters. This model is extended in \[[55](#bib.bib55 ""), [56](#bib.bib56 "")\] to a case where different tasks in a task cluster share useful features via a matrix stick-breaking process and a beta-Bernoulli hierarchical prior, respectively, and in \[[57](#bib.bib57 "")\] where each task is a compressive sensing task. Moreover, a nested Dirichlet process is proposed in \[[58](#bib.bib58 ""), [59](#bib.bib59 "")\] to use Dirichlet processes to learn both task clusters and the state structure of an infinite hidden Markov model, which handles sequential data in each task. In \[[60](#bib.bib60 "")\], 𝐰i\\mathbf{w}^{i} is decomposed as 𝐰i\=𝐮i+𝚯iT​𝐯i\\mathbf{w}^{i}=\\mathbf{u}^{i}+\\bm{\\Theta}\_{i}^{T}\\mathbf{v}^{i} similar to Eq. ([9](#S2.E9 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), where 𝐮i\\mathbf{u}^{i} and 𝚯i\\bm{\\Theta}\_{i} are sampled according to a Dirichlet process.

Different from \[[50](#bib.bib50 ""), [54](#bib.bib54 "")\], Jacob et al. \[[61](#bib.bib61 "")\] aim to learn task clusters under the regularization framework by considering three orthogonal aspects, including a global penalty to measure on average how large the parameters, a measure of between-cluster variance to quantify the distance among different clusters and a measure of within-cluster variance to quantify the compactness of task clusters. By combining these three aspects and adopting some convex relaxation technique, a convex objective function is formulated as

min𝐖,𝐛,𝚺\\displaystyle\\min\_{\\mathbf{W},\\mathbf{b},\\bm{\\Sigma}}\\ 

L⁡(𝐖,𝐛)+λ​tr​(𝐖𝟏𝟏T​𝐖T)+tr⁡(𝐖\~​𝚺−1​𝐖\~T)\\displaystyle L(\\mathbf{W},\\mathbf{b})+\\lambda\\mathrm{tr}(\\mathbf{W}\\mathbf{1}\\mathbf{1}^{T}\\mathbf{W}^{T})+\\mathrm{tr}(\\mathbf{\\tilde{W}}\\bm{\\Sigma}^{-1}\\mathbf{\\tilde{W}}^{T})

s.t.\\displaystyle\\mathrm{s.t.}\\ 

𝐖\~\=𝐖​𝚷,α​𝐈⪯𝚺⪯β​𝐈,tr⁡(𝚺)\=γ,\\displaystyle\\mathbf{\\tilde{W}}=\\mathbf{W}\\bm{\\Pi},\\ \\alpha\\mathbf{I}\\preceq\\bm{\\Sigma}\\preceq\\beta\\mathbf{I},\\ \\mathrm{tr}(\\bm{\\Sigma})=\\gamma,

(15)

where 𝚷\\bm{\\Pi} denotes the m×mm\\times m centering matrix, 𝟏\\mathbf{1} denotes a column vector of all ones with its size depending on the context, and α,β,γ\\alpha,\\beta,\\gamma are hyperparameters.

Kang et al. \[[62](#bib.bib62 "")\] extend the MTFL method \[[10](#bib.bib10 "")\] to the case with multiple task clusters and aim to minimize the squared trace norm in each cluster. A diagonal matrix, 𝐐i∈ℝm×m\\mathbf{Q}\_{i}\\in\\mathbb{R}^{m\\times m}, is defined as a cluster indicator matrix for the iith cluster. The jjth diagonal entry of 𝐐i\\mathbf{Q}\_{i} is equal to 1 if task 𝒯j\\mathcal{T}\_{j} lies in the iith cluster and otherwise 0. Since each task can belong to only one cluster, it is easy to see that ∑i\=1r𝐐i\=𝐈\\sum\_{i=1}^{r}\\mathbf{Q}\_{i}=\\mathbf{I}. Based on these considerations, the objective function is formulated as

min𝐖,𝐛,{𝐐i}⁡L⁡(𝐖,𝐛)+λ​∑i\=1r‖𝐖𝐐i‖S⁡(1)2​s.t.𝐐i∈{0,1}m×m∑i\=1r𝐐i\=𝐈.\\displaystyle\\min\_{\\mathbf{W},\\mathbf{b},\\{\\mathbf{Q}\_{i}\\}}L(\\mathbf{W},\\mathbf{b})+\\lambda\\sum\_{i=1}^{r}\\|\\mathbf{W}\\mathbf{Q}\_{i}\\|\_{S(1)}^{2}\\ \\mathrm{s.t.}{\\mathbf{Q}\_{i}\\in\\{0,1\\}^{m\\times m}\\atop\\sum\_{i=1}^{r}\\mathbf{Q}\_{i}=\\mathbf{I}}.

When rr equals 1, this method reduces to the MTFL method.

Han and Zhang \[[63](#bib.bib63 "")\] devise a structurally sparse regularizer to cluster tasks with the objective function as

min𝐖,𝐛⁡L⁡(𝐖,𝐛)+λ​∑j\>i‖𝐰i−𝐰j‖2.\\min\_{\\mathbf{W},\\mathbf{b}}\\ L(\\mathbf{W},\\mathbf{b})+\\lambda\\sum\_{j>i}\\|\\mathbf{w}^{i}-\\mathbf{w}^{j}\\|\_{2}.

(16)

Problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is a special case of the method proposed in \[[63](#bib.bib63 "")\] with only one level of task clusters. The regularizer on 𝐖\\mathbf{W} enforces any pair of columns in 𝐖\\mathbf{W} to have a chance to be identical and after solving problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), the cluster structure can be discovered by comparing columns in 𝐖\\mathbf{W}. One advantage of this structurally sparse regularizer is that the convex problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can automatically determine the number of task clusters.

Barzilai and Crammer \[[64](#bib.bib64 "")\] propose a task clustering method by defining 𝐖\\mathbf{W} as 𝐖\=𝐅𝐆\\mathbf{W}=\\mathbf{FG} where 𝐅∈ℝd×r\\mathbf{F}\\in\\mathbb{R}^{d\\times r} and 𝐆∈{0,1}r×m\\mathbf{G}\\in\\{0,1\\}^{r\\times m}. With an assumption that each task belongs to only one cluster, the objective function is formulated as

min𝐅,𝐆,𝐛⁡L⁡(𝐅𝐆,𝐛)+λ​‖𝐅‖F2​s.t.𝐆∈{0,1}r×m‖𝐠i‖2\=1​∀i∈\[m\],\\displaystyle\\min\_{\\mathbf{F},\\mathbf{G},\\mathbf{b}}L(\\mathbf{FG},\\mathbf{b})+\\lambda\\|\\mathbf{F}\\|\_{F}^{2}\\ \\mathrm{s.t.}\\ {\\mathbf{G}\\in\\{0,1\\}^{r\\times m}\\atop\\|\\mathbf{g}^{i}\\|\_{2}=1\\ \\forall i\\in\[m\]},

(17)

where 𝐠i\\mathbf{g}^{i} denotes the iith column of 𝐆\\mathbf{G}. When using the hinge loss or logistic loss, this non-convex problem can be relaxed to a min-max problem, which has a global optimum, by utilizing the dual problem with respect to 𝐖\\mathbf{W} and 𝐛\\mathbf{b} and discarding some non-convex constraints.

Zhou and Zhao \[[65](#bib.bib65 "")\] aim to cluster tasks by identifying representative tasks which are a subset of the given mm tasks. If task 𝒯i\\mathcal{T}\_{i} is selected by task 𝒯j\\mathcal{T}\_{j} as a representative task, then it is expected that model parameters for 𝒯j\\mathcal{T}\_{j} are similar to those of 𝒯i\\mathcal{T}\_{i}. zi​jz\_{ij} is defined as the probability that task 𝒯j\\mathcal{T}\_{j} selects task 𝒯i\\mathcal{T}\_{i} as its representative task. Then based on a matrix 𝐙\\mathbf{Z} whose (i,j)(i,j)th entry is zi​jz\_{ij}, the objective function is formulated as

min𝐖,𝐛,𝐙\\displaystyle\\hskip-3.61371pt\\min\_{\\mathbf{W},\\mathbf{b},\\mathbf{Z}}

L⁡(𝐖,𝐛)+λ1​‖𝐖‖F2+λ2​∑i\=1m∑j\=1mzi​j​‖𝐰i−𝐰j‖22+λ3​‖𝐙‖2,1\\displaystyle L(\\mathbf{W},\\mathbf{b})+\\lambda\_{1}\\|\\mathbf{W}\\|\_{F}^{2}+\\lambda\_{2}\\sum\_{i=1}^{m}\\sum\_{j=1}^{m}z\_{ij}\\|\\mathbf{w}^{i}-\\mathbf{w}^{j}\\|\_{2}^{2}+\\lambda\_{3}\\|\\mathbf{Z}\\|\_{2,1}

s.t.\\displaystyle\\mathrm{s.t.}\\ 

𝐙≥𝟎,𝐙T​𝟏\=𝟏.\\displaystyle\\mathbf{Z}\\geq\\mathbf{0},\\ \\mathbf{Z}^{T}\\mathbf{1}=\\mathbf{1}.

(18)

The third term in the objective function of problem ([18](#S2.E18 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) enforces the closeness of each pair of tasks based on 𝐙\\mathbf{Z} and the last term employs the ℓ2,1\\ell\_{2,1} norm to enforce the row sparsity of 𝐙\\mathbf{Z} which implies that the number of representative tasks is limited. The constraints in problem ([18](#S2.E18 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) guarantee that entries in 𝐙\\mathbf{Z} define valid probabilities. Problem ([18](#S2.E18 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is related to problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) since the regularizer in problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can be reformulated as 2​∑j\>i‖𝐰i−𝐰j‖2\=min⁡∑j\>i𝐙^≥𝟎⁡(z^i​j​‖𝐰i−𝐰j‖22+1z^i​j)2\\sum\_{j>i}\\|\\mathbf{w}^{i}-\\mathbf{w}^{j}\\|\_{2}=\\min\_{\\mathbf{\\hat{Z}}\\geq\\mathbf{0}}\\sum\_{j>i}\\left(\\hat{z}\_{ij}\\|\\mathbf{w}^{i}-\\mathbf{w}^{j}\\|\_{2}^{2}+\\frac{1}{\\hat{z}\_{ij}}\\right), where both the regularizer and constraint on 𝐙^\\mathbf{\\hat{Z}} are different from those on 𝐙\\mathbf{Z} in problem ([18](#S2.E18 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

Previous studies assume that each task can belong to only one task cluster and this assumption seems too restrictive. In \[[66](#bib.bib66 "")\], a GO-MTL method relaxes this assumption by allowing a task to belong to more than one cluster and defines a decomposition of 𝐖\\mathbf{W} similar to problem ([17](#S2.E17 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) as 𝐖\=𝐋𝐒\\mathbf{W}=\\mathbf{LS} where 𝐋∈ℝd×r\\mathbf{L}\\in\\mathbb{R}^{d\\times r} denotes the latent basis with r<mr\<m and 𝐒∈ℝr×m\\mathbf{S}\\in\\mathbb{R}^{r\\times m} contains linear combination coefficients for all the tasks. 𝐒\\mathbf{S} is assumed to be sparse since each task is generated from only a few columns in 𝐋\\mathbf{L} or equivalently belongs to a small number of clusters. The objective function is formulated as

min𝐋,𝐒,𝐛⁡L⁡(𝐋𝐒,𝐛)+λ1​‖𝐒‖1+λ2​‖𝐋‖F2.\\min\_{\\mathbf{L},\\mathbf{S},\\mathbf{b}}L(\\mathbf{LS},\\mathbf{b})+\\lambda\_{1}\\|\\mathbf{S}\\|\_{1}+\\lambda\_{2}\\|\\mathbf{L}\\|\_{F}^{2}.

(19)

Compared with the objective function of multi-task sparse coding, i.e., problem ([3](#S2.E3 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we can see that when the regularization parameters take appropriate values, these two problems are almost equivalent except that in multi-task sparse coding, the dictionary 𝐔\\mathbf{U} is overcomplete, while here the number of columns in 𝐒\\mathbf{S} is smaller than that of its rows. This method has been extended in \[[67](#bib.bib67 "")\] to decompose the parameter tensor in the fully connected layers of deep neural networks.

Among the aforementioned methods, the method in \[[48](#bib.bib48 "")\] first identifies the cluster structure and then learns the model parameters of all the tasks separately, which is not preferred since the cluster structure learned may be suboptimal for the model parameters, hence follow-up works learn model parameters and the cluster structure together. An important problem in clustering is to determine the number of clusters and this is also important for this approach. Out of the above methods, only methods in \[[54](#bib.bib54 ""), [63](#bib.bib63 "")\] can automatically determine the number of task clusters, where the method in \[[54](#bib.bib54 "")\] depends on the capacity of the Dirichlet process while the method in \[[63](#bib.bib63 "")\] relies on the use of a structurally sparse regularizer. Among all those models, some belong to Bayesian learning, i.e., \[[50](#bib.bib50 ""), [54](#bib.bib54 "")\], while the rest models are regularized models. Among those regularized methods, only the objective function proposed in \[[63](#bib.bib63 "")\] is convex while others are originally non-convex.

The task clustering approach is related to the low-rank approach. To see that, suppose that there are rr task clusters (r<mr\<m) and all the tasks in a cluster share the same model parameters, making the parameter matrix 𝐖\\mathbf{W} low-rank with the rank at most rr. From the perspective of modeling, by setting 𝐮i\\mathbf{u}^{i} to be a zero vector in Eq. ([9](#S2.E9 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we can see that the decomposition of 𝐖\\mathbf{W} in \[[40](#bib.bib40 "")\] becomes similar to those in \[[66](#bib.bib66 ""), [64](#bib.bib64 "")\], which in some sense shows the relation between those two approaches. Moreover, the equivalence between problems ([12](#S2.E12 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and ([15](#S2.E15 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), two typical methods in the low-rank and task clustering approaches, has been proved in \[[68](#bib.bib68 "")\]. The task clustering approach can visualize the learned cluster structure, which is an advantage over the low-rank approach.

### II-D Task Relation Learning Approach

In MTL, tasks are related and the task relatedness can be quantitated via task similarity, task correlation, task covariance and so on. Here we use task relations to include all the quantitative relatedness.

In earlier studies on MTL, task relations are assumed to be known as a priori information. In \[[69](#bib.bib69 ""), [70](#bib.bib70 "")\], each task is assumed to be similar to any other task and so model parameters of each task will be enforced to approach the average model parameters of all the tasks. In \[[71](#bib.bib71 ""), [72](#bib.bib72 "")\], task similarities for each pair of tasks are given and these studies utilize the task similarities to design regularizers to guide the learning of multiple tasks in a principle that the more similar two tasks are, the closer the corresponding model parameters are expected to be. A similar formulation to \[[71](#bib.bib71 "")\] is proposed in \[[73](#bib.bib73 "")\] to estimate the mean of multiple distributions by learning pairwise task relations and another similar formulation is proposed in \[[74](#bib.bib74 "")\] for log-density gradient estimation. Given a tree structure describing relations among tasks in \[[75](#bib.bib75 "")\], model parameters of a task corresponding to a node in the tree are enforced to be similar to those of its parent node.

However, in most applications, task relations are not available. In this case, learning task relations from data automatically is a good option. Bonilla et al. \[[76](#bib.bib76 "")\] propose a multi-task Gaussian process (MTGP) by defining a prior on fjif^{i}\_{j}, the functional value for 𝐱ji\\mathbf{x}^{i}\_{j}, as 𝐟∼𝒩(⋅|𝟎,𝚺)\\mathbf{f}\\sim\\mathcal{N}(\\cdot|\\mathbf{0},\\bm{\\Sigma}), where 𝐟\=(f11,…,fnmm)T\\mathbf{f}=(f^{1}\_{1},\\ldots,f^{m}\_{n\_{m}})^{T} contains the functional values for all the training data. 𝚺\\bm{\\Sigma}, the covariance matrix, defines the covariance between fjif^{i}\_{j} and fqpf^{p}\_{q} as σ⁡(fji,fqp)\=ωi​p​k​(𝐱ji,𝐱qp)\\sigma(f^{i}\_{j},f^{p}\_{q})=\\omega\_{ip}k(\\mathbf{x}^{i}\_{j},\\mathbf{x}^{p}\_{q}), where k⁡(⋅,⋅)k(\\cdot,\\cdot) denotes a kernel function and ωi​p\\omega\_{ip} describes the covariance between tasks 𝒯i\\mathcal{T}\_{i} and 𝒯p\\mathcal{T}\_{p}. In order to keep 𝚺\\bm{\\Sigma} positive definite, a matrix 𝛀\\bm{\\Omega} containing ωi​p\\omega\_{ip} as its (i,p)(i,p)th entry is also required to be positive definite, which makes 𝛀\\bm{\\Omega} the task covariance to describe the similarities between tasks. Then based on the Gaussian likelihood for labels given 𝐟\\mathbf{f}, the analytically marginal likelihood by integrating out 𝐟\\mathbf{f} can be used to learn 𝛀\\bm{\\Omega} from data. In \[[77](#bib.bib77 "")\], the learning curve and generalization bound of the MTGP are studied. Since 𝛀\\bm{\\Omega} in MTGP has a point estimation which may lead to the overfitting, based on a proposed weight-space view of MTGP, Zhang and Yeung \[[78](#bib.bib78 "")\] propose a multi-task generalized tt process by placing an inverse-Wishart prior on 𝛀\\bm{\\Omega} as 𝛀∼ℐ𝒲(⋅|ν,𝚿)\\bm{\\Omega}\\sim\\mathcal{IW}(\\cdot|\\nu,\\bm{\\Psi}), where ν\\nu denotes the degree of freedom and 𝚿\\bm{\\Psi} is the base covariance for generating 𝛀\\bm{\\Omega}. Since 𝚿\\bm{\\Psi} models the covariance between pairs of tasks, it can be determined based on the maximum mean discrepancy (MMD).

Different from \[[76](#bib.bib76 ""), [78](#bib.bib78 "")\] which are Bayesian models, Zhang and Yeung \[[79](#bib.bib79 ""), [80](#bib.bib80 "")\] propose a regularized multi-task model called multi-task relationship learning (MTRL) by placing a matrix-variate normal prior on 𝐖\\mathbf{W} as

𝐖∼ℳ𝒩(⋅|𝟎,𝐈,𝛀),\\mathbf{W}\\sim\\mathcal{MN}(\\cdot|\\mathbf{0},\\mathbf{I},\\bm{\\Omega}),

(20)

where ℳ𝒩(⋅|𝐌,𝐀,𝐁)\\mathcal{MN}(\\cdot|\\mathbf{M},\\mathbf{A},\\mathbf{B}) denotes a matrix-variate normal distribution with 𝐌\\mathbf{M} as the mean, 𝐀\\mathbf{A} the row covariance, and 𝐁\\mathbf{B} the column covariance. Based on this prior as well as some likelihood function, the objective function for a modified maximum a posterior solution is formulated as

min𝐖,𝐛,𝛀\\displaystyle\\min\_{\\mathbf{W},\\mathbf{b},\\bm{\\Omega}}\\ 

L⁡(𝐖,𝐛)+λ1​‖𝐖‖F2+λ2​tr​(𝐖​𝛀−1​𝐖T)\\displaystyle L(\\mathbf{W},\\mathbf{b})+\\lambda\_{1}\\|\\mathbf{W}\\|\_{F}^{2}+\\lambda\_{2}\\mathrm{tr}(\\mathbf{W}\\bm{\\Omega}^{-1}\\mathbf{W}^{T})

s.t.\\displaystyle\\mathrm{s.t.}\\ 

𝛀≻𝟎,tr⁡(𝛀)≤1,\\displaystyle\\bm{\\Omega}\\succ\\mathbf{0},\\ \\mathrm{tr}(\\bm{\\Omega})\\leq 1,

(21)

where the second term in the objective function is to penalize the complexity of 𝐖\\mathbf{W}, the last term is due to the matrix-variate normal prior, and the constraints control the complexity of the positive definite covariance matrix 𝛀\\bm{\\Omega}. It has been proved in \[[79](#bib.bib79 ""), [80](#bib.bib80 "")\] that problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is jointly convex with respect to 𝐖\\mathbf{W}, 𝐛\\mathbf{b} and 𝛀\\bm{\\Omega}. Problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) has been extended to multi-task boosting \[[81](#bib.bib81 "")\] and multi-label learning \[[82](#bib.bib82 "")\] by learning label correlations. Problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can also been interpreted from the perspective of reproducing kernel Hilbert spaces for vector-valued functions \[[83](#bib.bib83 ""), [84](#bib.bib84 ""), [85](#bib.bib85 ""), [86](#bib.bib86 "")\]. Moreover, Problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is extended to learn sparse task relations in \[[87](#bib.bib87 "")\] via the ℓ1\\ell\_{1} regularization on 𝛀\\bm{\\Omega} when the number of tasks is large. A model similar to problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is proposed in \[[88](#bib.bib88 "")\] via a matrix-variate normal prior on 𝐖\\mathbf{W}: 𝐖∼ℳ𝒩(⋅|𝟎,𝛀1,𝛀2)\\mathbf{W}\\sim\\mathcal{MN}(\\cdot|\\mathbf{0},\\bm{\\Omega}\_{1},\\bm{\\Omega}\_{2}), where 𝛀1−1\\bm{\\Omega}\_{1}^{-1} and 𝛀2−1\\bm{\\Omega}\_{2}^{-1} are assumed to be sparse. The MTRL model is extended in \[[89](#bib.bib89 "")\] to use the symmetric matrix-variate generalized hyperbolic distribution to learn block sparse structure in 𝐖\\mathbf{W} and in \[[90](#bib.bib90 "")\] to use the matrix generalized inverse Gaussian prior to learn low-rank 𝛀1\\bm{\\Omega}\_{1} and 𝛀2\\bm{\\Omega}\_{2}. Moreover, the MTRL model is generalized to the multi-task feature selection problem \[[37](#bib.bib37 "")\] by learning task relations via the matrix-variate generalized normal distribution. Since the prior defined in Eq. ([20](#S2.E20 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) implies that 𝐖T​𝐖\\mathbf{W}^{T}\\mathbf{W} follows a Wishart distribution as 𝒲(⋅|𝟎,𝛀)\\mathcal{W}(\\cdot|\\mathbf{0},\\bm{\\Omega}), Zhang and Yeung \[[91](#bib.bib91 "")\] generalize it as

(𝐖T𝐖)t∼𝒲(⋅|𝟎,𝛀),(\\mathbf{W}^{T}\\mathbf{W})^{t}\\sim\\mathcal{W}(\\cdot|\\mathbf{0},\\bm{\\Omega}),

(22)

where tt is a positive integer to model high-order task relationships. Eq. ([22](#S2.E22 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can induce a new prior, which is a generalization of the matrix-variate normal distribution, on 𝐖\\mathbf{W} and based on this new prior, a regularized method is devised to learn high-order task relations in \[[91](#bib.bib91 "")\]. The MTRL model has been extended to multi-output regression \[[92](#bib.bib92 ""), [90](#bib.bib90 ""), [93](#bib.bib93 ""), [94](#bib.bib94 "")\] by modeling the structure contained in noises via some matrix-variate priors. For deep neural networks, the MTRL method has been extended in \[[95](#bib.bib95 "")\] by placing a tensor-variate normal distribution as a prior on the parameter tensor in the fully connected layers.

Different from the aforementioned methods which investigate the use of global learning models in MTL, Zhang \[[96](#bib.bib96 "")\] aims to learn the task relations in local learning methods such as the kk-nearest-neighbor (kkNN) classifier by defining the learning function as a weighted voting of neighbors:

f⁡(𝐱ji)\=∑(p,q)∈Nk​(i,j)σi​p​s​(𝐱ji,𝐱qp)​yqp,f(\\mathbf{x}^{i}\_{j})=\\sum\_{(p,q)\\in N\_{k}(i,j)}\\sigma\_{ip}s(\\mathbf{x}^{i}\_{j},\\mathbf{x}^{p}\_{q})y^{p}\_{q},

(23)

where Nk​(i,j)N\_{k}(i,j) denotes the set of task indices and instance indices for the kk nearest neighbors of 𝐱ji\\mathbf{x}^{i}\_{j}, i.e., (p,q)∈Nk​(i,j)(p,q)\\in N\_{k}(i,j) meaning that 𝐱qp\\mathbf{x}^{p}\_{q} is one of the kk nearest neighbors of 𝐱ji\\mathbf{x}^{i}\_{j}, s⁡(𝐱ji,𝐱qp)s(\\mathbf{x}^{i}\_{j},\\mathbf{x}^{p}\_{q}) defines the similarity between 𝐱ji\\mathbf{x}^{i}\_{j} and 𝐱qp\\mathbf{x}^{p}\_{q}, and σi​p\\sigma\_{ip} represents the contribution of task 𝒯p\\mathcal{T}\_{p} to 𝒯i\\mathcal{T}\_{i} when 𝒯p\\mathcal{T}\_{p} has some data points to be neighbors of a data point in 𝒯i\\mathcal{T}\_{i}. σi​p\\sigma\_{ip} can be viewed as the similarity from 𝒯p\\mathcal{T}\_{p} to 𝒯i\\mathcal{T}\_{i}. When σi​p\=1\\sigma\_{ip}=1 for all ii and pp, Eq. ([23](#S2.E23 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) reduces to the decision function of the kkNN classifier for all the tasks. Then the objective function to learn 𝚺\\bm{\\Sigma}, which is a m×mm\\times m matrix with σi​p\\sigma\_{ip} as its (i,p)(i,p)th entry, can be formulated as

min𝚺\\displaystyle\\min\_{\\bm{\\Sigma}}

∑i\=1m1ni​∑j\=1nil⁡(yji,f⁡(𝐱ji))+λ14​‖𝚺−𝚺T‖F2+λ22​‖𝚺‖F2\\displaystyle\\sum\_{i=1}^{m}\\frac{1}{n\_{i}}\\sum\_{j=1}^{n\_{i}}l(y^{i}\_{j},f(\\mathbf{x}^{i}\_{j}))+\\frac{\\lambda\_{1}}{4}\\|\\bm{\\Sigma}-\\bm{\\Sigma}^{T}\\|\_{F}^{2}+\\frac{\\lambda\_{2}}{2}\\|\\bm{\\Sigma}\\|\_{F}^{2}

s.t.\\displaystyle\\mathrm{s.t.}

σi​i≥0​∀i∈\[m\],−σi​i≤σi​j≤σi​i​∀i≠j.\\displaystyle\\sigma\_{ii}\\geq 0\\ \\forall i\\in\[m\],-\\sigma\_{ii}\\leq\\sigma\_{ij}\\leq\\sigma\_{ii}\\ \\forall i\\neq j.

(24)

The first regularizer in problem ([24](#S2.E24 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) enforces 𝚺\\bm{\\Sigma} to be nearly symmetric and the second one is to penalize the complexity of 𝚺\\bm{\\Sigma}. The constraints in problem ([24](#S2.E24 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) guarantee that the similarity from one task to itself is positive and also the largest. Similarly, a multi-task kernel regression is proposed in \[[96](#bib.bib96 "")\] for regression tasks.

While the aforementioned methods whose task relations are symmetric except \[[96](#bib.bib96 "")\], Lee et al. \[[97](#bib.bib97 "")\] focus on learning asymmetric task relations. Since different tasks are assumed to be related, 𝐰i\\mathbf{w}\_{i} can lie in the space spanned by 𝐖\\mathbf{W}, i.e., 𝐰i≈𝐖𝐚i\\mathbf{w}\_{i}\\approx\\mathbf{W}\\mathbf{a}\_{i}, and hence we have 𝐖≈𝐖𝐀\\mathbf{W}\\approx\\mathbf{W}\\mathbf{A}. Here matrix 𝐀\\mathbf{A} can be viewed as asymmetric task relations between pairs of tasks. By assuming that 𝐀\\mathbf{A} is sparse, the objective function is formulated as

min𝐖,𝐛,𝐀\\displaystyle\\hskip-7.22743pt\\min\_{\\mathbf{W},\\mathbf{b},\\mathbf{A}}\\ 

∑i\=1m(1+λ1​‖𝐚^i‖1)​∑j\=1nil⁡(yji,(𝐰i)T​𝐱ji+bi)+λ2​‖𝐖−𝐖𝐀‖F2\\displaystyle\\sum\_{i=1}^{m}(1+\\lambda\_{1}\\|\\mathbf{\\hat{a}}\_{i}\\|\_{1})\\sum\_{j=1}^{n\_{i}}l(y^{i}\_{j},(\\mathbf{w}^{i})^{T}\\mathbf{x}^{i}\_{j}+b\_{i})+\\lambda\_{2}\\|\\mathbf{W}-\\mathbf{W}\\mathbf{A}\\|\_{F}^{2}

s.t.\\displaystyle\\mathrm{s.t.}\\ 

ai​j≥0​∀i,j∈\[m\],\\displaystyle a\_{ij}\\geq 0\\ \\forall i,j\\in\[m\],

(25)

where 𝐚^i\\mathbf{\\hat{a}}\_{i} denotes the iith row of 𝐀\\mathbf{A} by deleting ai​ia\_{ii}. The term before the training loss of each task, i.e., 1+λ1​‖𝐚^i‖11+\\lambda\_{1}\\|\\mathbf{\\hat{a}}\_{i}\\|\_{1}, not only enforces 𝐀\\mathbf{A} to be sparse but also allows asymmetric information sharing from easier tasks to difficult ones. The regularizer in problem ([25](#S2.E25 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can make 𝐖\\mathbf{W} approach 𝐖𝐀\\mathbf{W}\\mathbf{A} with the closeness depending on λ2\\lambda\_{2}. To see the connection between problems ([25](#S2.E25 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we rewrite the regularizer in problem ([25](#S2.E25 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) as ‖𝐖−𝐖𝐀‖F2\=tr⁡(𝐖⁡(𝐈−𝐀)​(𝐈−𝐀)T​𝐖T)\\|\\mathbf{W}-\\mathbf{W}\\mathbf{A}\\|\_{F}^{2}=\\mathrm{tr}\\left(\\mathbf{W}(\\mathbf{I}-\\mathbf{A})(\\mathbf{I}-\\mathbf{A})^{T}\\mathbf{W}^{T}\\right). Based on this reformulation, the regularizer in problem ([25](#S2.E25 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is a special case of that in problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) by assuming 𝛀−1\=(𝐈−𝐀)​(𝐈−𝐀)T\\bm{\\Omega}^{-1}=(\\mathbf{I}-\\mathbf{A})(\\mathbf{I}-\\mathbf{A})^{T}. Though 𝐀\\mathbf{A} is asymmetric, from the perspective of the regularizer, the task relations here are symmetric and act as the task precision matrix with a restrictive form.

### II-E Decomposition Approach

The decomposition approach assumes that the parameter matrix 𝐖\\mathbf{W} can be decomposed into two or more component matrices {𝐖k}k\=1h\\{\\mathbf{W}\_{k}\\}\_{k=1}^{h} where h≥2h\\geq 2, i.e., 𝐖\=∑k\=1h𝐖k\\mathbf{W}=\\sum\_{k=1}^{h}\\mathbf{W}\_{k}. The objective functions of most methods in this approach can be unified as

min{𝐖i}∈𝒞W,𝐛⁡L⁡(∑k\=1h𝐖k,𝐛)+∑k\=1hgk​(𝐖k),\\min\_{\\{\\mathbf{W}\_{i}\\}\\in\\mathcal{C}\_{W},\\mathbf{b}}\\ L\\Big(\\sum\_{k=1}^{h}\\mathbf{W}\_{k},\\mathbf{b}\\Big)+\\sum\_{k=1}^{h}g\_{k}(\\mathbf{W}\_{k}),

(26)

where the regularizer is decomposable with respect to 𝐖k\\mathbf{W}\_{k}’s and 𝒞W\\mathcal{C}\_{W} denotes a set of constraints for component matrices. To help understand problem ([26](#S2.E26 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), we introduce several instantiations as follows.

In \[[98](#bib.bib98 "")\] where hh equals 2 and 𝒞W\=∅\\mathcal{C}\_{W}=\\emptyset is an empty set, g1​(⋅)g\_{1}(\\cdot) and g2​(⋅)g\_{2}(\\cdot) are defined as

g1​(𝐖1)\=λ1​‖𝐖1‖∞,1,g2​(𝐖2)\=λ2​‖𝐖2‖1,g\_{1}(\\mathbf{W}\_{1})=\\lambda\_{1}\\|\\mathbf{W}\_{1}\\|\_{\\infty,1},\\ g\_{2}(\\mathbf{W}\_{2})=\\lambda\_{2}\\|\\mathbf{W}\_{2}\\|\_{1}, where λ1\\lambda\_{1} and λ2\\lambda\_{2} are positive regularization parameters. Similar to problem ([5](#S2.E5 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), each row of 𝐖1\\mathbf{W}\_{1} is likely to be a zero row and hence g1​(𝐖1)g\_{1}(\\mathbf{W}\_{1}) can help select important features. Due to the ℓ1\\ell\_{1} norm regularization, g2​(𝐖2)g\_{2}(\\mathbf{W}\_{2}) makes 𝐖2\\mathbf{W}\_{2} sparse. Because of the characteristics of two regularizers, the parameter matrix 𝐖\\mathbf{W} can eliminate unimportant features for all the tasks when the corresponding rows in both 𝐖1\\mathbf{W}\_{1} and 𝐖2\\mathbf{W}\_{2} are sparse. Moreover, 𝐖2\\mathbf{W}\_{2} can identify features for tasks which have their own useful features that may be outliers for other tasks. Hence this model can be viewed as a ‘robust’ version of problem ([5](#S2.E5 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

With two component matrices, Chen et al. \[[99](#bib.bib99 "")\] define

g2​(𝐖2)\=λ2​‖𝐖2‖1,𝒞W\={𝐖1|‖𝐖1‖S⁡(1)≤λ1},\\displaystyle g\_{2}(\\mathbf{W}\_{2})=\\lambda\_{2}\\|\\mathbf{W}\_{2}\\|\_{1},\\ \\mathcal{C}\_{W}=\\{\\mathbf{W}\_{1}|\\|\\mathbf{W}\_{1}\\|\_{S(1)}\\leq\\lambda\_{1}\\},

(27)

where g1​(𝐖1)\=0g\_{1}(\\mathbf{W}\_{1})=0. Similar to problem ([13](#S2.E13 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), 𝒞W\\mathcal{C}\_{W} makes 𝐖1\\mathbf{W}\_{1} low-rank. With a sparse regularizer g2​(𝐖2)g\_{2}(\\mathbf{W}\_{2}), 𝐖2\\mathbf{W}\_{2} makes the entire model matrix 𝐖\\mathbf{W} more robust to outlier tasks in a way similar to the previous model. When λ2\\lambda\_{2} is large enough, 𝐖2\\mathbf{W}\_{2} will become a zero matrix and then problem ([27](#S2.E27 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) will act similarly to problem ([13](#S2.E13 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

gi​(⋅)g\_{i}(\\cdot)’s in \[[100](#bib.bib100 "")\] where 𝒞W\=∅\\mathcal{C}\_{W}=\\emptyset are defined as

g1​(𝐖1)\=λ1​‖𝐖1‖S⁡(1),g2​(𝐖2)\=λ2​‖𝐖2T‖2,1.g\_{1}(\\mathbf{W}\_{1})=\\lambda\_{1}\\|\\mathbf{W}\_{1}\\|\_{S(1)},\\hskip 9.24994ptg\_{2}(\\mathbf{W}\_{2})=\\lambda\_{2}\\|\\mathbf{W}\_{2}^{T}\\|\_{2,1}.

(28)

Different from the above two models which assume that 𝐖2\\mathbf{W}\_{2} is sparse, here g2​(𝐖2)g\_{2}(\\mathbf{W}\_{2}) enforces 𝐖2\\mathbf{W}\_{2} to be column-sparse. For related tasks, their columns in 𝐖1\\mathbf{W}\_{1} are correlated via the trace norm regularization and the corresponding columns in 𝐖2\\mathbf{W}\_{2} are zero. For outlier tasks which are unrelated to other tasks, the corresponding columns in 𝐖2\\mathbf{W}\_{2} can take arbitrary values and hence model parameters in 𝐖\\mathbf{W} for them have no low-rank structure even though those in 𝐖1\\mathbf{W}\_{1} may have.

In \[[101](#bib.bib101 "")\], these functions are defined as

g1​(𝐖1)\=λ1​‖𝐖1‖2,1,g2​(𝐖2)\=λ2​‖𝐖2T‖2,1,𝒞W\=∅.g\_{1}(\\mathbf{W}\_{1})=\\lambda\_{1}\\|\\mathbf{W}\_{1}\\|\_{2,1},\\ g\_{2}(\\mathbf{W}\_{2})=\\lambda\_{2}\\|\\mathbf{W}\_{2}^{T}\\|\_{2,1},\\ \\mathcal{C}\_{W}=\\emptyset.

(29)

Similar to problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), g1​(𝐖1)g\_{1}(\\mathbf{W}\_{1}) makes 𝐖1\\mathbf{W}\_{1} row-sparse. Here g2​(𝐖2)g\_{2}(\\mathbf{W}\_{2}) is identical to that in \[[100](#bib.bib100 "")\] and it makes 𝐖2\\mathbf{W}\_{2} column-sparse. Hence 𝐖1\\mathbf{W}\_{1} helps select useful features while non-zero columns in 𝐖2\\mathbf{W}\_{2} capture outlier tasks.

With h\=2h=2, Zhong and Kwok \[[102](#bib.bib102 "")\] define

g1​(𝐖1)\=λ1​c​(𝐖1)+λ2​‖𝐖1‖F2,g2​(𝐖2)\=λ3​‖𝐖2‖F2,𝒞W\=∅,g\_{1}(\\mathbf{W}\_{1})=\\lambda\_{1}c(\\mathbf{W}\_{1})+\\lambda\_{2}\\|\\mathbf{W}\_{1}\\|\_{F}^{2},g\_{2}(\\mathbf{W}\_{2})=\\lambda\_{3}\\|\\mathbf{W}\_{2}\\|\_{F}^{2},\\mathcal{C}\_{W}=\\emptyset, where c⁡(𝐔)\=∑i\=1d∑k\>j|ui​j−ui​k|c(\\mathbf{U})=\\sum\_{i=1}^{d}\\sum\_{k>j}|u\_{ij}-u\_{ik}| with ui​ju\_{ij} as the (i,j)(i,j)th entry in a matrix 𝐔\\mathbf{U}. Due to the sparse nature of the ℓ1\\ell\_{1} norm, c⁡(𝐖1)c(\\mathbf{W}\_{1}) enforces corresponding entries in different columns of 𝐖1\\mathbf{W}\_{1} to be identical, which is equivalent to clustering tasks in terms of individual model parameters. Both the squared Frobenius norm regularizations in g1​(𝐖1)g\_{1}(\\mathbf{W}\_{1}) and g2​(𝐖2)g\_{2}(\\mathbf{W}\_{2}) penalize the complexities of 𝐖1\\mathbf{W}\_{1} and 𝐖2\\mathbf{W}\_{2}. The use of 𝐖2\\mathbf{W}\_{2} improves the model flexibility when not all the tasks exhibit a clear cluster structure.

Different from the aforementioned methods which have only two component matrices, an arbitrary number of component matrices are considered in \[[103](#bib.bib103 "")\] with

gk​(𝐖k)\=λ⁡\[(h−k)​‖𝐖k‖2,1+(k−1)​‖𝐖k‖1\]/(h−1),\\displaystyle g\_{k}(\\mathbf{W}\_{k})=\\lambda\\big\[(h-k)\\|\\mathbf{W}\_{k}\\|\_{2,1}+(k-1)\\|\\mathbf{W}\_{k}\\|\_{1}\\big\]/(h-1),

(30)

where 𝒞W\=∅\\mathcal{C}\_{W}=\\emptyset. According to Eq. ([30](#S2.E30 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), 𝐖k\\mathbf{W}\_{k} is assumed to be both sparse and row-sparse for all k∈\[h\]k\\in\[h\]. Based on different regularization parameters on the regularizer of 𝐖k\\mathbf{W}\_{k}, we can see that when kk increases, 𝐖k\\mathbf{W}\_{k} is more likely to be sparse than to be row-sparse. Even though each 𝐖k\\mathbf{W}\_{k} is sparse or row-sparse, the entire parameter matrix 𝐖\\mathbf{W} can be non-sparse and hence this model can discover the latent sparse structure among tasks.

In the above methods, different component matrices have no direct connection. When there is a dependency among component matrices, problem ([26](#S2.E26 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) can model more complex structure among tasks. For example, Han and Zhang \[[104](#bib.bib104 "")\] define

gk(𝐖k)\=λ∑i\>j∥𝐰ki−𝐰kj∥2/ηk−1∀k∈\[h\]\\displaystyle g\_{k}(\\mathbf{W}\_{k})=\\lambda\\sum\_{i>j}\\|\\mathbf{w}\_{k}^{i}-\\mathbf{w}\_{k}^{j}\\|\_{2}/\\eta^{k-1}\\ \\forall k\\in\[h\]

𝒞W\={{𝐖k}||𝐰k−1i−𝐰k−1j|≥|𝐰ki−𝐰kj|∀k≥2,∀i\>j},\\displaystyle\\mathcal{C}\_{W}=\\{\\{\\mathbf{W}\_{k}\\}|\\ |\\mathbf{w}\_{k-1}^{i}-\\mathbf{w}\_{k-1}^{j}|\\geq|\\mathbf{w}\_{k}^{i}-\\mathbf{w}\_{k}^{j}|\\ \\forall k\\geq 2,\\ \\forall i>j\\}, where 𝐰ki\\mathbf{w}\_{k}^{i} denotes the iith column of 𝐖k\\mathbf{W}\_{k}. Note that the constraint set 𝒞W\\mathcal{C}\_{W} relates component matrices and the regularizer gk​(𝐖k)g\_{k}(\\mathbf{W}\_{k}) makes each pair of 𝐰ki\\mathbf{w}\_{k}^{i} and 𝐰kj\\mathbf{w}\_{k}^{j} have a chance to become identical. Once this happens for some ii, jj, kk, then based on the constraint set 𝒞W\\mathcal{C}\_{W}, 𝐰k′i\\mathbf{w}\_{k^{\\prime}}^{i} and 𝐰k′j\\mathbf{w}\_{k^{\\prime}}^{j} will always have the same value for k′≥kk^{\\prime}\\geq k. This corresponds to sharing all the ancestor nodes for two internal nodes in a tree and hence this method can learn a hierarchical structure to characterize task relations. When the constraints are removed, this method reduces to the multi-level task clustering method \[[63](#bib.bib63 "")\], which is a generalization of problem ([16](#S2.E16 "In II-C Task Clustering Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

Another way to relate different component matrices is to use a non-decomposable regularizer as \[[105](#bib.bib105 "")\] did, which is slightly different from problem ([26](#S2.E26 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) in terms of the regularizer. Specifically, given mm tasks, there are 2m−12^{m}-1 possible and non-empty task clusters. All the task clusters can be organized in a tree, where the root node represents a dummy node, nodes in the second level represent groups with a single task, and the parent-child relations are the ‘subset of’ relation. In total, there are h≡2mh\\equiv 2^{m} component matrices each of which corresponds to a node in the tree and hence an index aa is used to denote both a level and the corresponding node in the tree. The objective function is formulated as

min{𝐖i},𝐛\\displaystyle\\min\_{\\{\\mathbf{W}\_{i}\\},\\mathbf{b}}\\ 

L⁡(∑k\=1h𝐖k,𝐛)+(∑v∈Vλv​(∑a∈D⁡(v)r​(𝐖a)p)1p)2\\displaystyle L\\left(\\sum\_{k=1}^{h}\\mathbf{W}\_{k},\\mathbf{b}\\right)+\\left(\\sum\_{v\\in V}\\lambda\_{v}\\left(\\sum\_{a\\in D(v)}r(\\mathbf{W}\_{a})^{p}\\right)^{\\frac{1}{p}}\\right)^{2}

s.t.\\displaystyle\\mathrm{s.t.}\\ 

𝐰ai\=𝟎​∀i∉t⁡(a),\\displaystyle\\mathbf{w}\_{a}^{i}=\\mathbf{0}\\ \\forall i\\notin t(a),

(31)

where pp takes a value between 1 and 2, D⁡(a)D(a) denotes the set of all the descendants of aa, t⁡(a)t(a) denotes the set of tasks contained in node aa, 𝐰ai\\mathbf{w}\_{a}^{i} denotes the iith column of 𝐖a\\mathbf{W}\_{a}, and r⁡(𝐖a)r(\\mathbf{W}\_{a}) reflects relations among tasks in node aa based on 𝐖a\\mathbf{W}\_{a}. The regularizer in problem ([31](#S2.E31 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is used to prune the subtree rooted at each node vv based on the ℓp\\ell\_{p} norm. The constraint in problem ([31](#S2.E31 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) implies that for tasks not contained in a node aa, the corresponding columns in 𝐖a\\mathbf{W}\_{a} are zero. In \[[105](#bib.bib105 "")\], r⁡(𝐖a)r(\\mathbf{W}\_{a}) adopts the regularizer proposed in \[[69](#bib.bib69 "")\] which enforces the parameters of all the tasks to approach their average.

Different from deep MTL models which are deep in terms of layers of feature representations, the decomposition approach can be viewed as a ‘deep’ approach in terms of model parameters while most of previous approaches are just shallow ones, making this approach have more powerful capacity. Moreover, the decomposition approach can reduce to other approaches such as the feature learning, low-rank and task clustering approaches when there is only one component matrix and hence it can be considered as an improved version of those approaches.

TABLE I: The performance comparison of representative MTL models in the five approaches on benchmark datasets in terms of some evaluation metric. nMSE stands for ‘normalized mean squared error’, RMSE is for ‘root mean squared error’, and AUC stands for ‘Area Under Curve’. ↑\\uparrow after the evaluation metric implies that the larger value the better performance and ↓\\downarrow indicates the opposite case.

| Dataset (Reference)                         | Evaluation Metric                | STL    | Feature Learning | Low-Rank | Task Clustering   | Task Relation Learning | Decomposition         |
| ------------------------------------------- | -------------------------------- | ------ | ---------------- | -------- | ----------------- | ---------------------- | --------------------- |
| School (\[[104](#bib.bib104 "")\])          | nMSE↓\\downarrow                 | —      | 0.4393           | —        | 0.4374/-/0.6466/- | —                      | 0.4445/-/-/-/0.4169   |
| SARCOS (\[[100](#bib.bib100 "")\])          | nMSE↓\\downarrow                 | 0.1821 | 0.1568           | 0.1531   | —                 | —                      | 0.1495/0.1456/-/-/-   |
| Computer Survey (\[[102](#bib.bib102 "")\]) | RMSE↓\\downarrow                 | 2.381  | —                | —        | 2.072/-/-/-       | 2.110/-/-              | 2.138/2.052/2.074/-/- |
| Parkinson (\[[86](#bib.bib86 "")\])         | Explained Variance↑\\uparrow     | 2.8%   | —                | —        | 2.7%/33.6%/-/-    | 12.0%/27.0%/-          | -/-/-/16.8%/-         |
| Sentiment (\[[79](#bib.bib79 "")\])         | Classification Error↓\\downarrow | 0.2779 | 0.2756           | —        | —                 | 0.2324/-/-             | —                     |
| MHC-I (\[[65](#bib.bib65 "")\])             | Classification Error↓\\downarrow | 0.2010 | —                | —        | 0.1890/0.2050/-/- | 0.1870/-/-             | 0.2030/-/0.2070/-/-   |
| Landmine (\[[86](#bib.bib86 "")\])          | AUC↑\\uparrow                    | 74.6%  | —                | —        | 75.9%/76.7%/-/-   | 76.1%/76.8%/-          | -/-/-/76.4%/-         |
| Office-Caltech (\[[95](#bib.bib95 "")\])    | Classification Error↓\\downarrow | 0.0920 | 0.0740           | —        | -/-/-/0.0670      | 0.0690/-/0.0450        | -/-/0.0760/-/-        |
| Office-Home (\[[95](#bib.bib95 "")\])       | Classification Error↓\\downarrow | 0.3430 | 0.4170           | —        | -/-/-/0.3350      | 0.4070/-/0.3310        | -/-/0.4140/-/-        |
| ImageCLEF (\[[95](#bib.bib95 "")\])         | Classification Error↓\\downarrow | 0.3640 | 0.3440           | —        | -/-/-/0.2780      | 0.3350/-/0.2470        | -/-/0.3510/-/-        |

### II-F Comparisons among Different Approaches

Based on the above introduction, we can see that different approaches exhibit their own characteristics. Specifically, the feature learning approach can learn common features, which are generic and invariant to all the tasks at hand and even new tasks, for all the tasks. When there exist outlier tasks which are unrelated to other tasks, the learned features can be influenced by outlier tasks significantly and they may cause the performance deterioration. By assuming that the parameter matrix is low-rank, the low-rank approach can explicitly learn the subspace of the parameter matrix or implicitly achieve that via some convex or non-convex regularizer. This approach is powerful but it seems applicable to only linear models, making nonlinear extensions non-trivial to be devised. The task clustering approach performs clustering on the task level in terms of model parameters and it can identify task clusters each of which consists of similar tasks. A major limitation of the task clustering approach is that it can capture positive correlations among tasks in the same cluster but ignore negative correlations among tasks in different clusters. Moreover, even though some methods in this category can automatically determine the number of clusters, most of them still need a model selection method such as cross validation to determine it, which may bring additional computational costs. The task relation learning approach can learn model parameters and pairwise task relations simultaneously. The learned task relations can give us insights about the relations between tasks and hence they improve the interpretability. The decomposition approach can be viewed as extensions of other parameter-based approaches by equipping multi-level parameters and hence they can model more complex task structure, e.g., tree structure. The number of components in the decomposition approach is important to the performance and needs to be carefully determined.

### II-G Benchmark Datasets and Performance Comparison

In this section, we introduce some benchmark datasets for MTL and compare the performance of different MTL models on them.

Some benchmark datasets for MTL are listed as follows.

*   •

```
School dataset \[[50](#bib.bib50 "")\]: This dataset is to estimate examination scores of 15,362 students from 139 secondary schools in London from 1985 to 1987 where each school is treated as a task. The input consists of four school-specific and three student-specific attributes.
```
*   •

```
SARCOS dataset22 2 [http://www.gaussianprocess.org/gpml/data/](http://www.gaussianprocess.org/gpml/data/ ""): This dataset studies a multi-output problem of learning the inverse dynamics of 7 SARCOS anthropomorphic robot arms, each of which corresponds to a task, based on 21 features, including seven joint positions, seven joint velocities and seven joint accelerations. This dataset contains 48,933 data points.
```
*   •

```
Computer Survey dataset \[[11](#bib.bib11 "")\]: This dataset is taken from a survey of 180 persons/tasks who rated the likelihood of purchasing one of 20 different personal computers, resulting in 36,000 data points in all the tasks. The features contain 13 different computer characteristics (e.g., price, CPU and RAM) while the output is an integer rating on the scale 0-10.
```
*   •

```
Parkinson dataset \[[105](#bib.bib105 "")\]: This dataset is to predict the disease symptom score of Parkinson for patients at different times using 19 bio-medical features. This dataset has 5,875 data points for 42 patients, each of whom is treated as a task.
```
*   •

```
Sentiment dataset33 3 [http://www.cs.jhu.edu/~mdredze/datasets/sentiment/](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/ ""): This dataset is to classify reviews of four products/tasks, i.e., books, DVDs, electronics and kitchen appliances, from Amazon into two classes: positive and negative reviews. For each task, there are 1,000 positive and 1,000 negative reviews, respectively.
```
*   •

```
MHC-I dataset \[[61](#bib.bib61 "")\]: This databset contains binding affinities of 15,236 peptides with 35 MHC-I molecules. Each MHC-I molecule is considered as a task and the goal is to predict whether a peptide binds a molecule.
```
*   •

```
Landmine dataset \[[54](#bib.bib54 "")\]: This dataset consists of 9-dimensional data points, whose features are extracted from radar images, from 29 landmine fields/tasks. Each task is to classify a data point into two classes (landmine or clutter). There are 14,820 data points in total.
```
*   •

```
Office-Caltech dataset \[[106](#bib.bib106 "")\]: The dataset contains data from 10 common categories shared in the Caltech-256 dataset and the Office dataset which consists of images collected from three distinct domains/tasks: Amazon, Webcam and DSLR, making this dataset contain 4 tasks. There are 2,533 images in all the tasks.
```
*   •

```
Office-Home dataset44 4 [http://hemanthdv.org/OfficeHome-Dataset](http://hemanthdv.org/OfficeHome-Dataset ""): This dataset consists of images from 4 different domains/tasks: artistic images, clip art, product images and real-world images. Each task contains images of 65 object categories collected in the office and home settings. In total, there are about 15,500 images in all the tasks.
```
*   •

```
ImageCLEF dataset55 5 [http://imageclef.org/2014/adaptation](http://imageclef.org/2014/adaptation ""): This dataset contains 12 common categories shared by four tasks: Caltech-256, ImageNet ILSVRC 2012, Pascal VOC 2012 and Bing. There are about 2,400 images in all the tasks.
```
In the above benchmark datasets, the first four datasets consist of regression tasks while the other datasets are classification tasks, where each task in the Sentiment, MHC-I and Landmine datasets is a binary classification problem and that in the other three image datasets is a multi-class classification problem. In order to compare different MTL approaches on those benchmark datasets, we select some representative MTL methods from each of the five approaches introduced in the previous sections and list in Table [I](#S2.T1 "TABLE I ‣ II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning") their performance reported in the MTL literature. We also include the performance of Single-Task Learning (STL), which trains a learning model for each task separately, for comparison. It is easy to see that MTL models perform better than STL counterparts in most cases, which verifies the effectiveness of MTL. Usually, different datasets have their own characteristics, making them more suitable for some MTL approach. For example, according to the studies in \[[50](#bib.bib50 ""), [71](#bib.bib71 ""), [102](#bib.bib102 "")\], different tasks in the School dataset are found to be very similar to each other. According to \[[54](#bib.bib54 "")\], the Landmine dataset can have two task clusters, where the first cluster consisting of the first 15 tasks corresponds to regions that are relatively highly foliated and the rest tasks belong to another cluster with regions that are bare earth or deserts. According to \[[61](#bib.bib61 "")\], it is well known in the vaccine design community that some molecules/tasks in the MHC-I dataset can be grouped into empirically defined supertypes known to have similar binding behaviors. For those three datasets, according to Table [I](#S2.T1 "TABLE I ‣ II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning") we can see that the task clustering, task relation learning and decomposition approaches have better performance since they can identify the cluster structure contained in the data in a plain or hierarchical way. For other datasets, they do not have so obvious structure among tasks but some MTL models can learn task correlations, which can bring more insights for model design and the interpretation of experimental results. For example, the task correlations in the SARCOS and Sentiment datasets are shown in Tables 2 and 3 of \[[79](#bib.bib79 "")\], and the task similarities in the Office-Caltech dataset are shown in Figure 3(b) of \[[95](#bib.bib95 "")\]. Moreover, for image datasets (i.e., Office-Caltech, Office-Home and ImageCLEF), deep MTL models (e.g., \[[67](#bib.bib67 ""), [95](#bib.bib95 "")\]) achieve better performance than shallow models since they can learn powerful feature representations, while the rest datasets are from diverse areas, making shallow models perform well on them.

### II-H Another Taxonomy for Regularized MTL Methods

Regularized methods form a main methodology for MTL. Here we classify many regularized MTL algorithms into two main categories: learning with feature covariance and learning with task relations. The former can be viewed as a representative formulation in feature-based MTL, while the latter is for parameter-based MTL.

Objective functions in the first category can be unified as

min𝐖,𝐛,𝚯⁡L⁡(𝐖,𝐛)+λ2​tr​(𝐖T​𝚯−1​𝐖)+f⁡(𝚯),\\hskip-7.22743pt\\min\_{\\mathbf{W},\\mathbf{b},\\bm{\\Theta}}L(\\mathbf{W},\\mathbf{b})+\\frac{\\lambda}{2}\\mathrm{tr}(\\mathbf{W}^{T}\\bm{\\Theta}^{-1}\\mathbf{W})+f(\\bm{\\Theta}),

(32)

where f⁡(⋅)f(\\cdot) denotes a regularizer or constraint on 𝚯\\bm{\\Theta}. From the perspective of probabilistic modeling, the regularizer λ2​tr​(𝐖T​𝚯−1​𝐖)\\frac{\\lambda}{2}\\mathrm{tr}(\\mathbf{W}^{T}\\bm{\\Theta}^{-1}\\mathbf{W}) corresponds to a matrix-variate normal distribution on 𝐖\\mathbf{W} as 𝐖∼ℳ​𝒩​(𝟎,1λ​𝚯⊗𝐈)\\mathbf{W}\\sim\\mathcal{MN}(\\mathbf{0},\\frac{1}{\\lambda}\\bm{\\Theta}\\otimes\\mathbf{I}). Based on this probabilistic prior, 𝚯\\bm{\\Theta} models the covariance between the features since 1λ​𝚯\\frac{1}{\\lambda}\\bm{\\Theta} is the row covariance matrix with each row in 𝐖\\mathbf{W} corresponding to a feature and different tasks share the feature covariance. All the models in this category differ in the choice of the function f⁡(⋅)f(\\cdot) on 𝚯\\bm{\\Theta}. For example, methods in \[[40](#bib.bib40 ""), [10](#bib.bib10 ""), [41](#bib.bib41 "")\] use f⁡(⋅)f(\\cdot) to restrict the trace of 𝚯\\bm{\\Theta} as shown in problems ([2](#S2.E2 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and ([12](#S2.E12 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). Moreover, multi-task feature selection methods based on the ℓ2,1\\ell\_{2,1} norm such as \[[23](#bib.bib23 ""), [24](#bib.bib24 ""), [25](#bib.bib25 "")\] can be reformulated as instances of problem ([32](#S2.E32 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

Different from the first category, methods in the second category have a unified objective function as

min𝐖,𝐛,𝚺⁡L⁡(𝐖,𝐛)+λ2​tr​(𝐖​𝚺−1​𝐖T)+g⁡(𝚺),\\hskip-7.22743pt\\min\_{\\mathbf{W},\\mathbf{b},\\bm{\\Sigma}}L(\\mathbf{W},\\mathbf{b})+\\frac{\\lambda}{2}\\mathrm{tr}(\\mathbf{W}\\bm{\\Sigma}^{-1}\\mathbf{W}^{T})+g(\\bm{\\Sigma}),

(33)

where g⁡(⋅)g(\\cdot) denotes a regularizer or constraint on 𝚺\\bm{\\Sigma}. The regularizer λ2​tr​(𝐖​𝚺−1​𝐖T)\\frac{\\lambda}{2}\\mathrm{tr}(\\mathbf{W}\\bm{\\Sigma}^{-1}\\mathbf{W}^{T}) corresponds to a matrix-variate normal prior on 𝐖\\mathbf{W} as 𝐖∼ℳ​𝒩​(𝟎,𝐈⊗1λ​𝚺)\\mathbf{W}\\sim\\mathcal{MN}(\\mathbf{0},\\mathbf{I}\\otimes\\frac{1}{\\lambda}\\bm{\\Sigma}), where 𝚺\\bm{\\Sigma} is to model the task relations since 1λ​𝚺\\frac{1}{\\lambda}\\bm{\\Sigma} is the column covariance with each column in 𝐖\\mathbf{W} corresponding to a task. From this perspective, the two regularizers for 𝐖\\mathbf{W} in problems ([32](#S2.E32 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) and ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) have different meanings even though the formulations seem a bit similar. All the methods in this category use different functions g⁡(⋅)g(\\cdot) to learn 𝚺\\bm{\\Sigma} with different functionalities. For example, the methods in \[[69](#bib.bib69 ""), [71](#bib.bib71 ""), [72](#bib.bib72 ""), [70](#bib.bib70 "")\], which utilize a priori information on task relations, directly learn 𝐖\\mathbf{W} and 𝐛\\mathbf{b} by defining g⁡(𝚺)\=0g(\\bm{\\Sigma})=0. Some task clustering methods \[[61](#bib.bib61 ""), [65](#bib.bib65 "")\] identify task clusters by assuming that 𝚺\\bm{\\Sigma} has a block structure. Several task relation learning methods including \[[79](#bib.bib79 ""), [107](#bib.bib107 ""), [80](#bib.bib80 ""), [97](#bib.bib97 ""), [87](#bib.bib87 "")\] directly learn 𝚺\\bm{\\Sigma} as a covariance matrix by constraining its trace or sparsity in g⁡(𝚺)g(\\bm{\\Sigma}). The trace norm regularization \[[44](#bib.bib44 "")\] can be formulated as an instance of problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")).

Even though this taxonomy cannot cover all the regularized MTL methods, it can bring insights to understand regularized MTL methods better and help devise more MTL models. For example, a learning framework is proposed in \[[108](#bib.bib108 "")\] to learn a suitable multi-task model for a given multi-task problem under problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) by utilizing 𝚺\\bm{\\Sigma} to represent the corresponding multi-task model.

### II-I Other Settings in MTL

Instead of assuming that different tasks share an identical feature representation, Zhang and Yeung \[[109](#bib.bib109 "")\] consider a multi-database face recognition problem where face recognition in a database is treated as a task. Since different face databases have different image sizes, here naturally all the tasks do not lie in the same feature space in this application, leading to a heterogeneous-feature MTL problem. To tackle this problem, a multi-task discriminant analysis (MTDA) is proposed in \[[109](#bib.bib109 "")\] by first projecting data in different tasks into a common subspace and then learning a common projection in this subspace to discriminate different classes in different tasks. In \[[110](#bib.bib110 "")\], a latent probit model is proposed to generate data of different tasks in different feature spaces via sparse transformations on a shared latent space and then to generate labels based on this latent space.

In many MTL classification problems, each task is explicitly or implicitly assumed to be a binary classification problem as each column in the parameter matrix 𝐖\\mathbf{W} contains model parameters for the corresponding task. It is not difficult to see that many methods in the feature learning approach, low-rank approach and decomposition approach can be directly extended to a general setting where each classification task can be a multi-class classification problem and correspondingly multiple columns in 𝐖\\mathbf{W} contains model parameters of a multi-class classification task. Such direct extension is applicable since those methods only rely on the entire 𝐖\\mathbf{W} or its rows but not columns as a media to share knowledge among tasks. However, to the best of our knowledge, there is no theoretical or empirical study to investigate such direct extension. For most methods in the task clustering and task relation learning approaches, such direct extension does not work since for multiple columns in 𝐖\\mathbf{W} corresponding to one task, we do not know which one(s) can be used to represent this task. Therefore, the direct extension may not be the best solution to the general setting. In the following, we introduce four main approaches other than the direct extension to tackle the general setting in MTL where each classification task can be a multi-class classification problem. The first method is to transform the multi-class classification problem in each task into a binary classification problem. For example, multi-task metric learning \[[70](#bib.bib70 ""), [111](#bib.bib111 "")\] can do that by treating a pair of data points from the same class as positive and that from different classes as negative. The second recipe is to utilize the characteristics of learners. For example, the linear discriminant analysis can handle binary and multi-class classification problems in a unified formulation and hence MTDA \[[109](#bib.bib109 "")\] can naturally handle them without changing the formulation. The third approach is to directly learn label correspondence among different tasks. In \[[112](#bib.bib112 "")\], two learning tasks, which share the training data, aim to maximize the mutual information to identify the correspondence between labels in different tasks. By assuming that all the tasks share the same label space, the last approach including \[[67](#bib.bib67 ""), [47](#bib.bib47 ""), [95](#bib.bib95 "")\] organizes the model parameters of all the tasks in a tensor where the model parameters of each task form a slice. Then the parameter tensor can be regularized by tensor trace norms \[[47](#bib.bib47 "")\] and a tensor-variate normal prior \[[95](#bib.bib95 "")\], or factorized as a product of several low-rank matrices or tensors \[[67](#bib.bib67 "")\].

Most MTL methods assume that the training data in each task are stored in a data matrix. In some case, the training data in each task exhibit a multi-modal structure and hence they are represented in a tensor instead of a matrix. Multilinear multi-task methods proposed in \[[113](#bib.bib113 ""), [114](#bib.bib114 "")\] can handle this situation by employing tensor trace norms as a generalization of the trace norm to perform the regularization.

### II-J Optimization Techniques in MTL

Optimization techniques used in MTL can be categorized into three main classes as follows.

*   •

```
Gradient descent method and its variants: The gradient descent method can be used to optimize smooth unconstrained objective functions possessed by many MTL models. If the unconstrained objective function is non-smooth, the subgradient can be used instead and then the gradient descent method can also be used. When there are some constraints in the objective function of MTL models \[[44](#bib.bib44 ""), [64](#bib.bib64 "")\], the projected gradient descent method can be used to project the updated solution in each step to the space defined by constraints. For deep MTL models, stochastic gradient descent methods can be used. Moreover, the GradNorm \[[115](#bib.bib115 "")\] is devised to normalize gradients to balance the learning of multiple tasks and \[[116](#bib.bib116 "")\] proposes the gradient surgery to avoid the interference between task gradients. Differently, \[[117](#bib.bib117 "")\] studies MTL from the perspective of multi-objective optimization by learning dynamic loss weights.
```
*   •

```
Block Coordinate Descent (BCD) method: The parameters in many MTL models can be divided into several blocks. For example, parameters in learning functions of all the tasks form a block and parameters to represent task relations are from another block. Directly optimizing the objective function of such a MTL model with respect to parameters in all blocks together is not easy. The BCD method, which is also known as the alternating method, is widely used in the MTL literature, e.g., \[[40](#bib.bib40 ""), [10](#bib.bib10 ""), [61](#bib.bib61 ""), [41](#bib.bib41 ""), [79](#bib.bib79 ""), [62](#bib.bib62 ""), [31](#bib.bib31 ""), [66](#bib.bib66 ""), [12](#bib.bib12 ""), [96](#bib.bib96 ""), [80](#bib.bib80 ""), [65](#bib.bib65 ""), [97](#bib.bib97 "")\], to alternatively optimize each block of parameters while fixing parameters in other blocks. Hence, each step of the BCD method will solve several subproblems, each of which is to optimize with respect to a block of parameters. Compared with the original objective function, each subproblem is easier to be solved and so the BCD method can help reduce the optimization complexity.
```
*   •

```
Proximal method \[[118](#bib.bib118 "")\]: For a nonsmooth objective function, which is the sum of smooth and nonsmooth functions, in an MTL model, the proximal method is frequently used (e.g., \[[24](#bib.bib24 ""), [99](#bib.bib99 ""), [100](#bib.bib100 ""), [101](#bib.bib101 ""), [102](#bib.bib102 ""), [103](#bib.bib103 ""), [27](#bib.bib27 ""), [63](#bib.bib63 ""), [104](#bib.bib104 ""), [119](#bib.bib119 ""), [120](#bib.bib120 ""), [121](#bib.bib121 "")\]) to construct a proximal problem by replacing the smooth function with a quadratic function that may be constructed based on its Taylor series in various ways and the resulting proximal problem is usually easier to be solved than the original problem. The proximal method can accelerate the convergence rate of the optimization process or facilitate the design of distributed optimization algorithms.
```
## III MTL with Other Learning Paradigms

In the previous section, we review different MTL approaches for supervised learning tasks. In this section, we overview some works on the combination of MTL with other learning paradigms in machine learning, including unsupervised learning such as clustering, semi-supervised learning, active learning, reinforcement learning, multi-view learning and graphical models, to either improve the performance of supervised MTL further via additional information such as unlabeled data or use MTL to help improve the performance of other learning paradigms.

In most applications, labeled data are expensive to collect but unlabeled data are abundant. So in some MTL applications, the training dataset of each task consists of both labeled and unlabeled data, hence we hope to exploit useful information contained in the unlabeled data to further improve the performance of supervised learning tasks. In machine learning, semi-supervised learning and active learning are two ways to utilize unlabeled data but in different ways. Semi-supervised learning aims to exploit geometrical information contained in the unlabeled data, while active learning selects representative unlabeled data to query an oracle with the hope of increasing the labeling cost as little as possible. Hence semi-supervised learning and active learning can be combined with MTL, leading to three new learning paradigms including semi-supervised multi-task learning \[[122](#bib.bib122 ""), [123](#bib.bib123 ""), [124](#bib.bib124 "")\], multi-task active learning \[[125](#bib.bib125 ""), [126](#bib.bib126 ""), [127](#bib.bib127 "")\] and semi-supervised multi-task active learning \[[128](#bib.bib128 "")\]. Specifically, a semi-supervised multi-task classification model is proposed in \[[122](#bib.bib122 ""), [123](#bib.bib123 "")\] to use random walk to exploit unlabeled data in each task and then cluster multiple tasks via a relaxed Dirichlet process. In \[[124](#bib.bib124 "")\], a semi-supervised multi-task Gaussian process for regression tasks, where different tasks are related via the hyperprior on the kernel parameters in Gaussian processes of all the tasks, is proposed to incorporate unlabeled data into the design of the kernel function in each task to achieve the smoothness in the corresponding functional spaces. Different from these semi-supervised multi-task methods, multi-task active learning adaptively selects informative unlabeled data for multi-task learners and hence the selection criterion is the core research issue. Reichart et al. \[[125](#bib.bib125 "")\] believe that data instances to be selected should be as informative as possible for a set of tasks instead of only one task and hence they propose two protocols for multi-task active learning. In \[[126](#bib.bib126 "")\], the expected error reduction is used as a criterion where each task is modeled by a supervised latent Dirichlet allocation model. Inspired by multi-armed bandits which balance the trade-off between the exploitation and exploration, a selection strategy is proposed in \[[127](#bib.bib127 "")\] to consider both the risk of a multi-task learner based on the trace norm regularization and the corresponding confidence bound. In \[[129](#bib.bib129 "")\], the MTRL method (i.e., problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))) is extended to the interactive setting where a human expert is enquired about partial orderings of pairwise task covariances based an inconsistency criterion. In \[[130](#bib.bib130 "")\], a proposed generalization bound is used to select a subset from multiple unlabeled tasks to acquire labels to improve the generalization performance of all the tasks. For semi-supervised multi-task active learning, Li et al. \[[128](#bib.bib128 "")\] propose a model to use the Fisher information as a criterion to select unlabeled data to acquire their labels with the semi-supervised multi-task classification model \[[122](#bib.bib122 ""), [123](#bib.bib123 "")\] as the classifier for each task.

MTL achieves the performance improvement in not only supervised learning tasks but also unsupervised learning tasks such as clustering. In \[[131](#bib.bib131 "")\], a multi-task Bregman clustering method is proposed based on single-task Bregman clustering by using the earth mover distance to minimize distances between any pair of tasks in terms of cluster centers and then in \[[132](#bib.bib132 ""), [133](#bib.bib133 "")\], an improved version of \[[131](#bib.bib131 "")\] and its kernel extension are proposed to avoid the negative effect caused by the regularizer in \[[131](#bib.bib131 "")\] via choosing the better one between single-task and multi-task Bregman clustering. In \[[134](#bib.bib134 "")\], a multi-task kernel kk-means method is proposed by learning the kernel matrix via both MMD between any pair of tasks and the Laplacian regularization that helps identify a smooth kernel space. In \[[135](#bib.bib135 "")\], two proposed multi-task clustering methods are extensions of the MTFL and MTRL methods by treating labels as cluster indicators to be learned. In \[[136](#bib.bib136 "")\], the principle of MTL is incorporated into the subspace clustering by capturing correlations between data instances. In \[[137](#bib.bib137 "")\], a multi-task clustering method belonging to instance-based MTL is proposed to share data instances among different tasks. In \[[138](#bib.bib138 "")\], a multi-task spectral clustering algorithm, which can handle the out-of-sample issue via a linear function to learn the cluster assignment, is proposed to achieve the feature selection among tasks via the ℓ2,1\\ell\_{2,1} regularization \[[139](#bib.bib139 "")\]. \[[140](#bib.bib140 "")\] proposes to identify the task cluster structure and learn task relations together.

Reinforcement Learning (RL) is a promising area in machine learning and has shown superior performance in many applications such as game playing (e.g., Atari and Go) and robotics. MTL can help boost the performance of reinforcement learning, leading to Multi-task Reinforcement Learning (MRL). Some works \[[141](#bib.bib141 ""), [142](#bib.bib142 ""), [143](#bib.bib143 ""), [144](#bib.bib144 ""), [145](#bib.bib145 ""), [146](#bib.bib146 ""), [147](#bib.bib147 ""), [148](#bib.bib148 ""), [149](#bib.bib149 ""), [150](#bib.bib150 "")\] adapt the ideas introduced in Section [II](#S2 "II MTL Models ‣ A Survey on Multi-Task Learning") to MRL. Specifically, in \[[141](#bib.bib141 "")\] where a task solves a sequence of Markov Decision Processes (MDPs), a hierarchical Bayesian infinite mixture model is used to model the distribution over MDPs and for each new MDP, previously learned distributions are used as an informative prior. In \[[142](#bib.bib142 "")\], a regionalized policy representation is introduced to characterize the behavior of an agent in each task and a Dirichlet process is placed over regionalized policy representations across multiple tasks to cluster tasks. In \[[143](#bib.bib143 "")\], a Gaussian process temporal-difference value function model is used for each task and a hierarchical Bayesian approach is to model the distribution over value functions in different tasks. Calandriello et al. \[[144](#bib.bib144 "")\] assume that parameter vectors of value functions in different tasks are jointly sparse and then extend the MTFS method with the ℓ2,1\\ell\_{2,1} regularization as well as the MTFL method to learn value functions in multiple tasks together. In \[[145](#bib.bib145 "")\], a model associating each subtask with a modular subpolicy is proposed to learn from policy sketches, which annotate tasks with sequences of named subtasks and provide information about high-level structural relationships among tasks. In \[[146](#bib.bib146 "")\], a multi-task contextual bandit is introduced to leverage or learn similarities in contexts among arms to improve the prediction of rewards from contexts. In \[[147](#bib.bib147 "")\], a multi-task linearly solvable MDP, whose task basis matrix contains a library of component tasks shared by all the tasks, is proposed to maintain a parallel distributed representation of tasks each of which enables an agent to draw on macro actions simultaneously. In \[[148](#bib.bib148 "")\], a multi-task deep RL model based on the attention can automatically group tasks into sub-networks on a state-level granularity. In \[[149](#bib.bib149 "")\], a sharing experience framework is introduced to use task-specific rewards to identify similar parts defined as shared-regions which can guide the experience sharing of task policies. In \[[150](#bib.bib150 "")\], multi-task soft option learning, a hierarchical framework based on planning as inference, is regularized by a shared prior to avoid training instabilities and allow the fine-tuning of options for new tasks without forgetting learned policies. The idea of compression and distillation have been incorporated into MRL as in \[[151](#bib.bib151 ""), [152](#bib.bib152 ""), [153](#bib.bib153 ""), [154](#bib.bib154 "")\]. For example, in \[[151](#bib.bib151 "")\], the proposed Actor-Mimic method combines both deep reinforcement learning and model compression techniques to train a policy network which can learn to act for multiple tasks. In \[[152](#bib.bib152 "")\], a policy distillation method is proposed to not only train an efficient network to learn the policy of an agent but also consolidate multiple task-specific policies into a single policy. In \[[153](#bib.bib153 "")\], the problem of multi-task multi-agent reinforcement learning under the partial observability is addressed by distilling decentralized single-task policies into a unified policy across multiple tasks. In \[[154](#bib.bib154 "")\], each task has its own policy which is constrained to be close to a shared policy that is trained by the distillation. Some works \[[155](#bib.bib155 ""), [156](#bib.bib156 ""), [157](#bib.bib157 ""), [158](#bib.bib158 ""), [159](#bib.bib159 ""), [160](#bib.bib160 "")\] in MRL focus on online and distributed settings. Specifically, in \[[155](#bib.bib155 "")\], a distributed MRL framework is devised to model it as an instance of general consensus and an efficient decentralized solver is developed. In \[[156](#bib.bib156 ""), [157](#bib.bib157 "")\], multiple goal-directed tasks are learned in an online setup without the need for expert supervision by actively sampling harder tasks. In \[[158](#bib.bib158 "")\], a distributed agent is developed to not only use resources more efficiently in single-machine training but also scale to thousands of machines without sacrificing data efficiency or resource utilization. \[[159](#bib.bib159 "")\] formulates MRL from a perspective of variational inference and it proposes a novel distributed solver with quadratic convergence guarantees. In \[[160](#bib.bib160 "")\], an online learning algorithm is proposed to dynamically combine different auxiliary tasks which provide gradient directions to speed up the training of the main reinforcement learning task. Some works study the theoretical foundation of MRL. For example, in \[[161](#bib.bib161 "")\], sharing representations among tasks is analyzed with theoretical guarantees to highlight conditions to share representations and finite-time bounds of approximated value-iteration are extended to the multi-task setting. Moreover, there are some works to design novel MRL methods. For example, in \[[162](#bib.bib162 "")\], a MRL framework is proposed to train agent to employ hierarchical policies that decide when to use a previously learned policy and when to learn a new skill with a temporal grammar that helps the agent learn complex temporal dependencies. \[[163](#bib.bib163 "")\] studies the problem of parallel learning of multiple sequential-decision tasks and proposes to automatically adapt the contribution of each task to the updates of the agent to make all tasks have comparable impacts on the learning dynamics. In \[[164](#bib.bib164 "")\], a self-supervised representation learning algorithm is proposed for multi-task deep RL to capture structured information about environment dynamics based on multi-step predictive representations of future observations.

Multi-view learning assumes that each data point is associated with multiple sets of features where each set corresponds to a view and it usually exploits information contained in multiple views for supervised or semi-supervised learning tasks. Multi-task multi-view learning extends multi-view learning to the MTL setting where each task is a multi-view learning problem. Specifically, in \[[165](#bib.bib165 "")\], a graph-based method is proposed for multi-task multi-view classification problems. In a task, each view is enforced to be consistent with both other views and labels, while different tasks are expected to have similar predictions on views they share, making views as a bridge to construct the task relatedness. In \[[166](#bib.bib166 "")\], both a regularized MTL method \[[71](#bib.bib71 "")\] and the MTRL method are applied to each view of different tasks and different views in a task are expected to achieve an agreement on unlabeled data. Different from \[[165](#bib.bib165 ""), [166](#bib.bib166 "")\] which study the multi-task multi-view classification problem, in \[[167](#bib.bib167 ""), [168](#bib.bib168 "")\], two multi-task multi-view clustering methods are proposed and both methods consider three factors: within-view-task clustering which conducts clustering on each view in a task, view relation learning which minimizes the disagreement among views in a task, and low-rank structure learning which aims to learn a shared subspace for different tasks under a common view. The difference between these two methods is that the first method uses a bipartite graph co-clustering method for nonnegative data while the other one adopts a semi-nonnegative matrix tri-factorization to cluster general data. In \[[169](#bib.bib169 "")\], a multi-label multi-view algorithm is proposed to not only learn common features via the ℓ2,1\\ell\_{2,1} regularization but also identify useless views via the Frobenius norm. In multi-task multi-view learning, each task is usually supplied with both labeled and unlabeled data, hence this paradigm can also be viewed as another way to utilize unlabeled information for MTL. A deep multi-task multi-view model is proposed in \[[170](#bib.bib170 "")\] to fuse all the views based on the cross-stitch network.

MTL can help learn more accurate structure in graphical models. In \[[171](#bib.bib171 "")\], an algorithm is proposed to learn Bayes network structures by assuming that different networks/tasks share similar structures via a common prior and then a heuristic search is used to find structures with high scores for all the tasks. With a similar idea, multiple Gaussian graphical models are jointly learned in \[[172](#bib.bib172 "")\] by assuming joint sparsity among precision matrices via the ℓ∞,1\\ell\_{\\infty,1} norm regularization. In \[[173](#bib.bib173 "")\], some domain knowledge about task relations is incorporated into the learning of multiple Bayesian networks. By viewing the feature interaction matrix as a form of graphical models to model pairwise relations between features, two models are proposed in \[[174](#bib.bib174 "")\] to learn a quadratical function, where the feature interaction matrix defines the quadratic term, for each task based on the ℓ2,1\\ell\_{2,1} and tensor trace norm regularization, respectively.

According to the above discussions, we can see that most research works discussed in this section follow the spirits of MTL approaches introduced in Section [II](#S2 "II MTL Models ‣ A Survey on Multi-Task Learning") and adapt to their own settings.

## IV Handling Big Data

When the number of tasks is large, the total number of training data in all the tasks can be very big and hence a ‘big’ aspect in MTL denotes the number of tasks. In this case, we can either devise online, parallel, or distributed MTL models to accelerate the learning process. Another ‘big’ aspect in MTL lies in the data dimensionality which can be very high. In this situation, we can speedup the learning via feature selection, dimensionality reduction and feature hashing to reduce the dimension without losing too much useful information. In this section, we review some relevant works.

When the number of tasks is very big, we can devise some parallel MTL methods to speedup the learning process on multi-CPU or multi-GPU devices. As a representative formulation in feature-based MTL, problem ([32](#S2.E32 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) is easy to parallelize since when given the feature covariance matrix 𝚯\\bm{\\Theta}, the learning of different tasks can be decoupled. However, for problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) in parameter-based MTL, the situation is totally different since even given the task covariance matrix 𝚺\\bm{\\Sigma}, different tasks are still coupled, making the direct parallelization fail. In order to parallelize problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")), Zhang \[[175](#bib.bib175 "")\] uses the FISTA algorithm to design a surrogate function for problem ([33](#S2.E33 "In II-H Another Taxonomy for Regularized MTL Methods ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) with a given 𝚺\\bm{\\Sigma}, where the surrogate function is decomposable with respect to tasks, leading to a parallel design for MTL based on different loss functions including the hinge, ϵ\\epsilon-insensitive and square losses.

Online multi-task learning is also capable of handling a big number of tasks. In \[[176](#bib.bib176 ""), [177](#bib.bib177 "")\], under a setting where all the tasks contribute toward a common goal, the relation between tasks is measured via a global loss function and several online algorithms are proposed to use absolute norms as the global loss function. In \[[178](#bib.bib178 "")\], online MTL algorithms are devised by modeling the task relatedness via hard constraints that the mm-tuple of actions for tasks satisfies. In \[[179](#bib.bib179 "")\], perceptron-based online algorithms are proposed for multi-task binary classification problems where task similarities are measured based on either the geometric closeness of the task reference vectors or the dimension of their spanned subspace. In \[[180](#bib.bib180 "")\], a recursive Bayesian online algorithm based on Gaussian processes is devised to update both estimations and confidence intervals when data instances arrive sequentially. In \[[181](#bib.bib181 "")\], an online version of the MTRL method is proposed to update both the model parameters and task covariance in a sequential way. An online multi-task learning algorithm is proposed in \[[182](#bib.bib182 "")\] to jointly learn the per-task model and the task relations by smoothing the loss function of each task w.r.t. a task distribution and adaptively refining this distribution over time. In \[[183](#bib.bib183 "")\], an online multi-task model is proposed to learn both a low-rank component and a group sparse component to characterize task relations. In \[[184](#bib.bib184 "")\], a multi-task passive-aggressive method is proposed to learn multiple relative similarity learning tasks, each of which is to learn a similarity function from data with relative constraints. In \[[185](#bib.bib185 "")\], a Gaussian distribution, whose mean or covariance consists of a local component for each task and a global component shared by all the tasks, over each task is used as a confidence measure to guide the online MTL process.

Training data can locate at different devices, making the design of distributed MTL models important. In \[[186](#bib.bib186 "")\], a communication-efficient distributed MTL algorithm, where each machine learns a task, based on the debiased Lasso is proposed to learn jointly sparse features in a high-dimensional space. In \[[187](#bib.bib187 "")\], the MTRL method (i.e., problem ([21](#S2.E21 "In II-D Task Relation Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))) is extended to the distributed setting based on a stochastic dual coordinate ascent method. In \[[188](#bib.bib188 "")\], to protect the privacy of data, a privacy-preserving distributed MTL method is proposed based on a privacy-preserving proximal gradient algorithm with asynchronous updates. In \[[189](#bib.bib189 "")\], federated multi-task learning is proposed as an extension of distributed multi-task learning to consider both stragglers and fault tolerance. In \[[190](#bib.bib190 "")\], a distributed multi-task algorithm is proposed under the online MTL setting.

For high-dimensional data in MTL, we can use multi-task feature selection methods to reduce the dimension or extend single-task dimension reduction techniques to the multi-task setting as did in \[[109](#bib.bib109 "")\]. Another option is to use the feature hashing and in \[[191](#bib.bib191 "")\], multiple hashing functions are proposed to accelerate the joint learning of multiple tasks.

TABLE II: The classification of works about MTL applications in different areas according to different MTL approaches.

| Approach                                                         |                                                                   |                                                                     |                                                   |                                                           |
| ---------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| Health Informatics                                               |                                                                   |                                                                     |                                                   |                                                           |
| Visual tracking \[[192](#bib.bib192 ""), [193](#bib.bib193 "")\] | Protein subcellular location prediction \[[194](#bib.bib194 "")\] | Speech synthesis \[[195](#bib.bib195 ""), [196](#bib.bib196 "")\]   | Learning to rank \[[197](#bib.bib197 "")\]        | Stock prediction \[[198](#bib.bib198 "")\]                |
| Action recognition \[[199](#bib.bib199 "")\]                     | Protein interaction prediction \[[200](#bib.bib200 "")\]          | Speech recognition \[[201](#bib.bib201 ""), [202](#bib.bib202 "")\] |                                                   | Localization \[[203](#bib.bib203 "")\]                    |
| Facial landmark detection \[[15](#bib.bib15 "")\]                | Biological image classification \[[17](#bib.bib17 "")\]           | Jointly learning of NLP tasks \[[204](#bib.bib204 "")\]             |                                                   |                                                           |
| Scene classification \[[205](#bib.bib205 "")\]                   |                                                                   | Dialog state tracking \[[18](#bib.bib18 "")\]                       |                                                   |                                                           |
| Attribute prediction \[[206](#bib.bib206 "")\]                   |                                                                   | Machine translation \[[207](#bib.bib207 "")\]                       |                                                   |                                                           |
| Image rotation \[[208](#bib.bib208 "")\]                         |                                                                   | Syntactic parsing \[[207](#bib.bib207 "")\]                         |                                                   |                                                           |
| Immediacy prediction \[[209](#bib.bib209 "")\]                   |                                                                   |                                                                     |                                                   |                                                           |
| Pose estimation \[[19](#bib.bib19 "")\]                          |                                                                   |                                                                     |                                                   |                                                           |
| Thumbnail selection \[[16](#bib.bib16 "")\]                      |                                                                   |                                                                     |                                                   |                                                           |
| Face verification \[[210](#bib.bib210 "")\]                      |                                                                   |                                                                     |                                                   |                                                           |
| Face and object recognition \[[211](#bib.bib211 "")\]            | siRNA efficacy prediction \[[212](#bib.bib212 "")\]               | Microblog analysis \[[119](#bib.bib119 "")\]                        | Behavioral targeting \[[213](#bib.bib213 "")\]    |                                                           |
| Brain imaging \[[214](#bib.bib214 "")\]                          | Genetic marker detection \[[215](#bib.bib215 "")\]                |                                                                     |                                                   |                                                           |
|                                                                  | Mental state examination \[[216](#bib.bib216 "")\]                |                                                                     |                                                   |                                                           |
|                                                                  | Predict cognitive outcome \[[217](#bib.bib217 "")\]               |                                                                     |                                                   |                                                           |
|                                                                  | Survival analysis \[[120](#bib.bib120 "")\]                       |                                                                     |                                                   |                                                           |
|                                                                  | Genetic trait prediction \[[218](#bib.bib218 "")\]                |                                                                     |                                                   |                                                           |
|                                                                  | Gene expression association \[[219](#bib.bib219 "")\]             |                                                                     |                                                   |                                                           |
| Image segmentation \[[220](#bib.bib220 "")\]                     | Identification of longitudinal                                    |                                                                     |                                                   | Climate prediction \[[221](#bib.bib221 "")\]              |
| Saliency detection\[[222](#bib.bib222 "")\]                      | phenotypic markers \[[223](#bib.bib223 "")\]                      |                                                                     |                                                   |                                                           |
| Image segmentation \[[224](#bib.bib224 "")\]                     | Brain-computer interfaces \[[225](#bib.bib225 "")\]               |                                                                     |                                                   |                                                           |
| Age estimation \[[226](#bib.bib226 "")\]                         | Personalized medical                                              |                                                                     |                                                   |                                                           |
| Facial action unit                                               | treatment \[[227](#bib.bib227 "")\]                               |                                                                     |                                                   |                                                           |
| prediction \[[228](#bib.bib228 "")\]                             | Genetic trait prediction \[[218](#bib.bib218 "")\]                |                                                                     |                                                   |                                                           |
| Action recognition \[[229](#bib.bib229 "")\]                     |                                                                   |                                                                     |                                                   |                                                           |
|                                                                  | Protein subcellular location prediction \[[194](#bib.bib194 "")\] | Sentiment classification \[[230](#bib.bib230 "")\]                  | Web search ranking \[[231](#bib.bib231 "")\]      | Localization \[[203](#bib.bib203 "")\]                    |
|                                                                  | Organism modeling \[[232](#bib.bib232 "")\]                       |                                                                     | Collaborative filtering \[[233](#bib.bib233 "")\] | Robotics \[[234](#bib.bib234 ""), [235](#bib.bib235 "")\] |
|                                                                  | MHC-I binding prediction \[[236](#bib.bib236 "")\]                |                                                                     | Display advertising \[[237](#bib.bib237 "")\]     | Trajectory regression \[[238](#bib.bib238 "")\]           |
|                                                                  | Splice-site prediction \[[236](#bib.bib236 "")\]                  |                                                                     |                                                   | Traffic sign recognition \[[239](#bib.bib239 "")\]        |
|                                                                  | Prioritization of disease genes \[[240](#bib.bib240 "")\]         |                                                                     |                                                   | Soil moisture forecasts \[[241](#bib.bib241 "")\]         |
|                                                                  | Protein interaction prediction \[[242](#bib.bib242 "")\]          |                                                                     |                                                   |                                                           |
|                                                                  | Identifying antigenic variants \[[243](#bib.bib243 "")\]          |                                                                     |                                                   |                                                           |
| Multi-view tracking \[[244](#bib.bib244 "")\]                    | Genetic trait prediction \[[218](#bib.bib218 "")\]                |                                                                     |                                                   |                                                           |
| Pose estimation \[[245](#bib.bib245 "")\]                        | Protein interaction prediction \[[246](#bib.bib246 "")\]          |                                                                     |                                                   |                                                           |
| Person re-identification \[[247](#bib.bib247 "")\]               |                                                                   |                                                                     |                                                   |                                                           |

## V Applications

MTL has many applications in various areas including computer vision, bioinformatics, health informatics, speech, NLP, web, and so on. In Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning"), we categorize different MTL problems in each application area according to MTL approaches they used, where the classification of MTL approaches has been already introduced in Section [II](#S2 "II MTL Models ‣ A Survey on Multi-Task Learning"). In the last column of Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning"), we list some problems in various application areas which are different from other columns. For application problems listed in Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning"), application-dependent MTL models have been proposed to solve them.66 6 For details of those models, refer to an arXiv version \[[248](#bib.bib248 "")\] of this paper. Though these models are different from each other, there are some characteristics in respective areas. For example, in computer vision, deep MTL models, most of which belong to the feature transformation approach, exhibit good performance, making this approach popular in computer vision. In bioinformatics and health informatics, the interpretability of learning models is more important in some sense. Therefore, the feature selection and task relation learning approaches are widely used in this area as the former approach can identify useful features and the latter one can quantitatively show task relations. In speech and NLP, the data exhibit a sequential structure, which makes recurrent-neural-network-based deep MTL models in the feature transformation approach play an important role. As the data in web applications is of a large scale, this area favors simple models such as linear models or their ensembles based on boosting. Among all the MTL approaches, the feature transformation, feature selection and task relation learning approaches are among the most widely used MTL approaches in different application areas according to Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning").

When encountering a new application problem which can be modeled as a MTL problem, we need to judge whether tasks in this problem are related in terms of either low-level features or high-level concepts. If so, by treating Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning") as a look-up table. we can identify a problem in Table [II](#S4.T2 "TABLE II ‣ IV Handling Big Data ‣ A Survey on Multi-Task Learning") similar to the new problem and then adapt the corresponding MTL model to solve the new problem. Otherwise, we can try popular MTL approaches in the respective area.

## VI Theoretical Analyses

As well as designing MTL models and exploiting MTL applications, there are some works to study theoretical aspects of MTL and here we review them.

The generalization bound, which is to upper-bound the generalization loss in terms of the training loss, model complexity and confidence, is core in learning theory since it can identify the learnability and induce the sample complexity. There are several works \[[249](#bib.bib249 ""), [250](#bib.bib250 ""), [40](#bib.bib40 ""), [251](#bib.bib251 ""), [252](#bib.bib252 ""), [253](#bib.bib253 ""), [254](#bib.bib254 ""), [49](#bib.bib49 ""), [255](#bib.bib255 ""), [256](#bib.bib256 ""), [257](#bib.bib257 ""), [258](#bib.bib258 ""), [259](#bib.bib259 ""), [260](#bib.bib260 "")\] to study the generalization bound of different MTL models. In Table [III](#S6.T3 "TABLE III ‣ VI Theoretical Analyses ‣ A Survey on Multi-Task Learning"), we compare those works in terms of the analyzed MTL model, analysis tool and the convergence rate of the corresponding bound which is based on the number of tasks (i.e., mm) and the average number of data points per task (i.e., n0n\_{0}). According to Table [III](#S6.T3 "TABLE III ‣ VI Theoretical Analyses ‣ A Survey on Multi-Task Learning"), we can see that some works (i.e., \[[249](#bib.bib249 ""), [250](#bib.bib250 ""), [253](#bib.bib253 ""), [254](#bib.bib254 ""), [49](#bib.bib49 ""), [257](#bib.bib257 ""), [258](#bib.bib258 "")\]) analyze different MTL models based on various analysis tools and the best convergence rate is O⁡(1m​n0)O(\\frac{1}{\\sqrt{mn\_{0}}}). Though MTL models analyzed in \[[40](#bib.bib40 ""), [251](#bib.bib251 ""), [12](#bib.bib12 ""), [259](#bib.bib259 "")\] are not the same, those MTL models exhibit similar objectives to learn a linear or nonlinear feature transformation shared by all the tasks, and the convergence rates are O⁡(1m​n0)O(\\frac{1}{\\sqrt{mn\_{0}}}) except \[[12](#bib.bib12 "")\]. The trace norm regularization (i.e., Problem ([13](#S2.E13 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))) is analyzed in \[[258](#bib.bib258 ""), [255](#bib.bib255 ""), [256](#bib.bib256 ""), [260](#bib.bib260 "")\], among which \[[260](#bib.bib260 "")\] has the best convergence rate based on the local Rademacher complexity. A related MTL model based on the Schatten norm regularization is analyzed in \[[252](#bib.bib252 "")\] with an O⁡(1m​n0)O(\\frac{1}{\\sqrt{mn\_{0}}}) convergence rate. For the graph regularization \[[69](#bib.bib69 ""), [71](#bib.bib71 "")\] analyzed in \[[252](#bib.bib252 ""), [260](#bib.bib260 "")\], the local Rademacher complexity leads to a better convergence rate (i.e., O⁡(1(m​n0)α)O(\\frac{1}{(mn\_{0})^{\\alpha}}) for some constant α∈(0.5,1)\\alpha\\in(0.5,1)) and a similar observation holds for problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")). In a word, various analysis tools can be used to analyze MTL models and among them, the local Rademacher complexity can derive tighter generalization bounds than others.

Besides generalization bounds, there are some works to study other theoretical problems in MTL. For example, Argyriou et al. \[[261](#bib.bib261 ""), [262](#bib.bib262 "")\] discuss conditions where representer theorems hold for regularized MTL algorithms. Several studies \[[263](#bib.bib263 ""), [264](#bib.bib264 ""), [265](#bib.bib265 "")\] investigate conditions to well recover true features for multi-task feature selection models.

TABLE III: Comparison of generalization bounds derived in different works in terms of the analyzed MTL model, analysis tool and the convergence rate of the bound. mm denotes the number of tasks and n0n\_{0} denotes the average number of data points per task. 

MTL Model

Reference

Analysis Tool

Convergence rate

Tasks from an environment

\[[249](#bib.bib249 ""), [250](#bib.bib250 "")\]

VC dimension & Covering number

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Task distributions can be transformed

\[[254](#bib.bib254 "")\]

VC dimension

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Task clustering

\[[49](#bib.bib49 "")\]

VC dimension

O⁡(m​ln⁡(n0/m)/n0)O(m\\ln(n\_{0}/m)/n\_{0})

Multi-task kernel classifier

\[[257](#bib.bib257 "")\]

Covering number

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Problem ([26](#S2.E26 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) with Eq. ([27](#S2.E27 "In II-E Decomposition Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))

\[[258](#bib.bib258 "")\]

Multi-task stability

O⁡(m​m/n0)O(m\\sqrt{m}/n\_{0})

Multi-task data compression

\[[253](#bib.bib253 "")\]

Kolmogorov complexity

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))

\[[40](#bib.bib40 "")\]

Covering number

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Problem ([10](#S2.E10 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning")) without 𝐔\\mathbf{U}

\[[251](#bib.bib251 "")\]

Rademacher complexity

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Problem ([3](#S2.E3 "In II-A1 Feature Transformation Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))

\[[12](#bib.bib12 "")\]

Rademacher complexity

O⁡(1/n0)O(1/\\sqrt{n\_{0}})

Learn a common feature transformation

\[[259](#bib.bib259 "")\]

Gaussian average

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Problem ([13](#S2.E13 "In II-B Low-Rank Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))

\[[258](#bib.bib258 "")\]

Multi-task stability

O⁡(m/n0)O(m/n\_{0})

\[[255](#bib.bib255 "")\]

Rademacher complexity

O⁡(ln⁡(m)/n0)O(\\ln(m)/\\sqrt{n\_{0}})

\[[256](#bib.bib256 "")\]

Rademacher complexity

O⁡(max⁡(ln⁡(m​n0)/m​n0,1/n0))O(\\max(\\sqrt{\\ln(mn\_{0})/mn\_{0}},1/\\sqrt{n\_{0}}))

\[[260](#bib.bib260 "")\]

Local Rademacher complexity

O⁡(1/(m​n0)α)O(1/(mn\_{0})^{\\alpha}), 0.5<α<10.5<\\alpha<1

Schatten-norm-regularized MTL models

\[[252](#bib.bib252 "")\]

Rademacher complexity

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

Graph regularizers \[[69](#bib.bib69 ""), [71](#bib.bib71 "")\]

\[[252](#bib.bib252 "")\]

Rademacher complexity

O⁡(1/m​n0)O(1/\\sqrt{mn\_{0}})

\[[260](#bib.bib260 "")\]

Local Rademacher complexity

O⁡(1/(m​n0)α)O(1/(mn\_{0})^{\\alpha}), 0.5<α<10.5<\\alpha<1

Problem ([4](#S2.E4 "In II-A2 Feature Selection Approach ‣ II-A Feature Learning Approach ‣ II MTL Models ‣ A Survey on Multi-Task Learning"))

\[[255](#bib.bib255 "")\]

Rademacher complexity

O⁡(1/n0)O(1/\\sqrt{n\_{0}})

\[[260](#bib.bib260 "")\]

Local Rademacher complexity

O⁡(1/(m​n0)α)O(1/(mn\_{0})^{\\alpha}), 0.5<α<10.5<\\alpha<1

## VII Conclusions and Discussions

In this paper, we survey different aspects of MTL. First, after giving the definition of MTL, we give a classification of supervised MTL models into five main approaches and discuss their characteristics. Then we review the combinations of MTL with other learning paradigms. The online, parallel and distributed MTL models as well as dimensionality reduction and feature hashing are discussed to speedup the learning process. The applications of MTL in various areas are introduced to show the usefulness of MTL and theoretical aspects of MTL are discussed.

In future studies, there are several issues to be addressed. Firstly, outlier tasks, which are unrelated to other tasks, are well known to hamper the performance of all the tasks when learning them jointly. There are some methods to alleviate negative effects that outlier tasks bring. However, there lacks principled ways and theoretical analyses to study the resulting negative effects. In order to make MTL safe to be used by human, this is an important issue and needs more studies.

Secondly, deep learning has become a dominant approach in many areas and several multi-task deep models belonging to the feature transformation, low-rank, task clustering and task relation learning approaches have been proposed as reviewed in Sections [II](#S2 "II MTL Models ‣ A Survey on Multi-Task Learning"), [III](#S3 "III MTL with Other Learning Paradigms ‣ A Survey on Multi-Task Learning") and [V](#S5 "V Applications ‣ A Survey on Multi-Task Learning"). As discussed, most of them only share hidden layers. This approach is powerful when all the tasks are related, but it is vulnerable to noisy and outlier tasks that can deteriorate the performance dramatically. We believe that it is desirable to design flexible and robust deep multi-task models.

Lastly, existing studies mainly focus on supervised learning tasks, and only a few ones are on other tasks such as unsupervised learning, semi-supervised learning, active learning, multi-view learning and reinforcement learning tasks. It is natural to adapt or extend the five approaches introduced in Section [II](#S2 "II MTL Models ‣ A Survey on Multi-Task Learning") to those non-supervised learning tasks. We think that such adaptation and extension require more efforts to design appropriate models. Moreover, it is worth trying to apply MTL to other areas in artificial intelligence such as logic and planning to broaden its application scopes.

Acknowledgments This work is supported by NSFC 62076118.

## References

*   \[1\] R. Caruana, “Multitask learning,” MLJ, 1997.
*   \[2\] Q. Yang, Y. Zhang, W. Dai, and S. J. Pan, Transfer Learning. Cambridge University Press, 2020.
*   \[3\] M.-L. Zhang and Z.-H. Zhou, “A review on multi-label learning algorithms,” IEEE TKDE, 2014.
*   \[4\] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.
*   \[5\] Y. Zhang and Q. Yang, “An overview of multi-task learning,” National Science Review, 2018.
*   \[6\] X. Yang, S. Kim, and E. P. Xing, “Heterogeneous multitask learning with joint sparsity constraints,” in NIPS, 2009.
*   \[7\] S. Bickel, J. Bogojeska, T. Lengauer, and T. Scheffer, “Multi-task learning for HIV therapy screening,” in ICML, 2008.
*   \[8\] X. Liao and L. Carin, “Radial basis function network for multi-task learning,” in NIPS, 2005.
*   \[9\] D. L. Silver, R. Poirier, and D. Currie, “Inductive transfer with context-sensitive neural networks,” MLJ, 2008.
*   \[10\] A. Argyriou, T. Evgeniou, and M. Pontil, “Convex multi-task feature learning,” MLJ, 2008.
*   \[11\] A. Argyriou, C. A. Micchelli, M. Pontil, and Y. Ying, “A spectral regularization framework for multi-task structure learning,” in NIPS, 2007.
*   \[12\] A. Maurer, M. Pontil, and B. Romera-Paredes, “Sparse coding for multitask and transfer learning,” in ICML, 2013.
*   \[13\] J. Zhu, N. Chen, and E. P. Xing, “Infinite latent SVM for classification and multi-task learning,” in NIPS, 2011.
*   \[14\] M. K. Titsias and M. Lázaro-Gredilla, “Spike and slab variational inference for multi-task and multiple kernel learning,” in NIPS, 2011.
*   \[15\] Z. Zhang, P. Luo, C. C. Loy, and X. Tang, “Facial landmark detection by deep multi-task learning,” in ECCV, 2014.
*   \[16\] W. Liu, T. Mei, Y. Zhang, C. Che, and J. Luo, “Multi-task deep visual-semantic embedding for video thumbnail selection,” in CVPR, 2015.
*   \[17\] W. Zhang, R. Li, T. Zeng, Q. Sun, S. Kumar, J. Ye, and S. Ji, “Deep model based transfer and multi-task learning for biological image analysis,” in KDD, 2015.
*   \[18\] N. Mrksic, D. Ó. Séaghdha, B. Thomson, M. Gasic, P. Su, D. Vandyke, T. Wen, and S. J. Young, “Multi-domain dialog state tracking using recurrent neural networks,” in ACL, 2015.
*   \[19\] S. Li, Z. Liu, and A. B. Chan, “Heterogeneous multi-task learning for human pose estimation with deep convolutional neural network,” IJCV, 2015.
*   \[20\] Y. Shinohara, “Adversarial multi-task learning of deep neural networks for robust speech recognition,” in Interspeech, 2016.
*   \[21\] P. Liu, X. Qiu, and X. Huang, “Adversarial multi-task learning for text classification,” in ACL, 2017.
*   \[22\] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert, “Cross-stitch networks for multi-task learning,” in CVPR, 2016.
*   \[23\] G. Obozinski, B. Taskar, and M. Jordan, “Multi-task feature selection,” tech. rep., University of California, Berkeley, 2006.
*   \[24\] J. Liu, S. Ji, and J. Ye, “Multi-task feature learning via efficient l2,1l\_{2,1}-norm minimization,” in UAI, 2009.
*   \[25\] S. Lee, J. Zhu, and E. P. Xing, “Adaptive multi-task lasso: With application to eQTL detection,” in NIPS, 2010.
*   \[26\] N. S. Rao, C. R. Cox, R. D. Nowak, and T. T. Rogers, “Sparse overlapping sets lasso for multitask learning and its application to fMRI analysis,” in NIPS, 2013.
*   \[27\] P. Gong, J. Zhou, W. Fan, and J. Ye, “Efficient multi-task feature learning with calibration,” in KDD, 2014.
*   \[28\] J. Wang and J. Ye, “Safe screening for multi-task feature learning with multiple data matrices,” in ICML, 2015.
*   \[29\] H. Liu, M. Palatucci, and J. Zhang, “Blockwise coordinate descent procedures for the multi-task lasso, with applications to neural semantic basis discovery,” in ICML, 2009.
*   \[30\] P. Gong, J. Ye, and C. Zhang, “Multi-stage multi-task feature learning,” JMLR, 2013.
*   \[31\] A. C. Lozano and G. Swirszcz, “Multi-level lasso for sparse multi-task regression,” in ICML, 2012.
*   \[32\] X. Wang, J. Bi, S. Yu, and J. Sun, “On multiplicative multitask feature learning,” in NIPS, 2014.
*   \[33\] L. Han, Y. Zhang, G. Song, and K. Xie, “Encoding tree sparsity in multi-task learning: A probabilistic framework,” in AAAI, 2014.
*   \[34\] T. Jebara, “Multi-task feature and kernel selection for SVMs,” in ICML, 2004.
*   \[35\] S. Kim and E. P. Xing, “Tree-guided group lasso for multi-task regression with structured sparsity,” in ICML, 2010.
*   \[36\] Y. Zhou, R. Jin, and S. C. H. Hoi, “Exclusive lasso for multi-task feature selection,” in AISTATS, 2010.
*   \[37\] Y. Zhang, D.-Y. Yeung, and Q. Xu, “Probabilistic multi-task feature selection,” in NIPS, 2010.
*   \[38\] D. Hernández-Lobato and J. M. Hernández-Lobato, “Learning feature selection dependencies in multi-task learning,” in NIPS, 2013.
*   \[39\] D. Hernández-Lobato, J. M. Hernández-Lobato, and Z. Ghahramani, “A probabilistic model for dirty multi-task feature selection,” in ICML, 2015.
*   \[40\] R. K. Ando and T. Zhang, “A framework for learning predictive structures from multiple tasks and unlabeled data,” JMLR, 2005.
*   \[41\] J. Chen, L. Tang, J. Liu, and J. Ye, “A convex formulation for learning shared structures from multiple tasks,” in ICML, 2009.
*   \[42\] A. Agarwal, H. Daumé III, and S. Gerber, “Learning multiple tasks using manifold regularization,” in NIPS, 2010.
*   \[43\] J. Zhang, Z. Ghahramani, and Y. Yang, “Learning multiple related tasks using latent independent component analysis,” in NIPS, 2005.
*   \[44\] T. K. Pong, P. Tseng, S. Ji, and J. Ye, “Trace norm regularization: Reformulations, algorithms, and multi-task learning,” SIAM Journal on Optimization, 2010.
*   \[45\] L. Han and Y. Zhang, “Multi-stage multi-task learning with reduced rank,” in AAAI, 2016.
*   \[46\] A. M. McDonald, M. Pontil, and D. Stamos, “Spectral kk-support norm regularization,” in NIPS, 2014.
*   \[47\] Y. Yang and T. M. Hospedales, “Trace norm regularised deep multi-task learning,” in ICLR, Workshop Track, 2017.
*   \[48\] S. Thrun and J. O’Sullivan, “Discovering structure in multiple learning tasks: The TC algorithm,” in ICML, 1996.
*   \[49\] K. Crammer and Y. Mansour, “Learning multiple tasks using shared hypotheses,” in NIPS, 2012.
*   \[50\] B. Bakker and T. Heskes, “Task clustering and gating for Bayesian multitask learning,” JMLR, 2003.
*   \[51\] K. Yu, V. Tresp, and A. Schwaighofer, “Learning Gaussian processes from multiple tasks,” in ICML, 2005.
*   \[52\] S. Yu, V. Tresp, and K. Yu, “Robust multi-task learning with *t*-processes,” in ICML, 2007.
*   \[53\] W. Lian, R. Henao, V. Rao, J. E. Lucas, and L. Carin, “A multitask point process predictive model,” in ICML, 2015.
*   \[54\] Y. Xue, X. Liao, L. Carin, and B. Krishnapuram, “Multi-task learning for classification with Dirichlet process priors,” JMLR, 2007.
*   \[55\] Y. Xue, D. B. Dunson, and L. Carin, “The matrix stick-breaking process for flexible multi-task learning,” in ICML, 2007.
*   \[56\] H. Li, X. Liao, and L. Carin, “Nonparametric Bayesian feature selection for multi-task learning,” in ICASSP, 2011.
*   \[57\] Y. Qi, D. Liu, D. B. Dunson, and L. Carin, “Multi-task compressive sensing with Dirichlet process priors,” in ICML, 2008.
*   \[58\] K. Ni, L. Carin, and D. B. Dunson, “Multi-task learning for sequential data via iHMMs and the nested Dirichlet process,” in ICML, 2007.
*   \[59\] K. Ni, J. W. Paisley, L. Carin, and D. B. Dunson, “Multi-task learning for analyzing and sorting large databases of sequential data,” IEEE TSP, 2008.
*   \[60\] A. Passos, P. Rai, J. Wainer, and H. Daumé III, “Flexible modeling of latent task structures in multitask learning,” in ICML, 2012.
*   \[61\] L. Jacob, F. R. Bach, and J.-P. Vert, “Clustered multi-task learning: A convex formulation,” in NIPS, 2008.
*   \[62\] Z. Kang, K. Grauman, and F. Sha, “Learning with whom to share in multi-task feature learning,” in ICML, 2011.
*   \[63\] L. Han and Y. Zhang, “Learning multi-level task groups in multi-task learning,” in AAAI, 2015.
*   \[64\] A. Barzilai and K. Crammer, “Convex multi-task learning by clustering,” in AISTATS, 2015.
*   \[65\] Q. Zhou and Q. Zhao, “Flexible clustered multi-task learning by learning representative tasks,” IEEE TPAMI, 2016.
*   \[66\] A. Kumar and H. Daumé III, “Learning task grouping and overlap in multi-task learning,” in ICML, 2012.
*   \[67\] Y. Yang and T. M. Hospedales, “Deep multi-task representation learning: A tensor factorisation approach,” in ICLR, 2017.
*   \[68\] J. Zhou, J. Chen, and J. Ye, “Clustered multi-task learning via alternating structure optimization,” in NIPS, 2011.
*   \[69\] T. Evgeniou and M. Pontil, “Regularized multi-task learning,” in KDD, 2004.
*   \[70\] S. Parameswaran and K. Q. Weinberger, “Large margin multi-task metric learning,” in NIPS, 2010.
*   \[71\] T. Evgeniou, C. A. Micchelli, and M. Pontil, “Learning multiple tasks with kernel methods,” JMLR, 2005.
*   \[72\] T. Kato, H. Kashima, M. Sugiyama, and K. Asai, “Multi-task learning via conic programming,” in NIPS, 2007.
*   \[73\] S. Feldman, M. R. Gupta, and B. A. Frigyik, “Revisiting Stein’s paradox: Multi-task averaging,” JMLR, 2014.
*   \[74\] I. Yamane, H. Sasaki, and M. Sugiyama, “Regularized multitask learning for multidimensional log-density gradient estimation,” Neural Computation, 2016.
*   \[75\] N. Görnitz, C. Widmer, G. Zeller, A. Kahles, S. Sonnenburg, and G. Rätsch, “Hierarchical multitask structured output learning for large-scale sequence segmentation,” in NIPS, 2011.
*   \[76\] E. V. Bonilla, K. M. A. Chai, and C. K. I. Williams, “Multi-task Gaussian process prediction,” in NIPS, 2007.
*   \[77\] K. M. A. Chai, “Generalization errors and learning curves for regression with multi-task Gaussian processes,” in NIPS, 2009.
*   \[78\] Y. Zhang and D.-Y. Yeung, “Multi-task learning using generalized tt process,” in AISTATS, 2010.
*   \[79\] Y. Zhang and D.-Y. Yeung, “A convex formulation for learning task relationships in multi-task learning,” in UAI, 2010.
*   \[80\] Y. Zhang and D.-Y. Yeung, “A regularization approach to learning task relationships in multitask learning,” ACM TKDD, 2014.
*   \[81\] Y. Zhang and D.-Y. Yeung, “Multi-task boosting by exploiting task relationships,” in ECMLPKDD, 2012.
*   \[82\] Y. Zhang and D.-Y. Yeung, “Multilabel relationship learning,” ACM TKDD, 2013.
*   \[83\] F. Dinuzzo, C. S. Ong, P. V. Gehler, and G. Pillonetto, “Learning output kernels with block coordinate descent,” in ICML, 2011.
*   \[84\] C. Ciliberto, Y. Mroueh, T. A. Poggio, and L. Rosasco, “Convex learning of multiple tasks and their structure,” in ICML, 2015.
*   \[85\] C. Ciliberto, L. Rosasco, and S. Villa, “Learning multiple visual tasks while discovering their structure,” in CVPR, 2015.
*   \[86\] P. Jawanpuria, M. Lapin, M. Hein, and B. Schiele, “Efficient output kernel learning for multiple tasks,” in NIPS, 2015.
*   \[87\] Y. Zhang and Q. Yang, “Learning sparse task relations in multi-task learning,” in AAAI, 2017.
*   \[88\] Y. Zhang and J. G. Schneider, “Learning multiple tasks with a sparse matrix-normal penalty,” in NIPS, 2010.
*   \[89\] C. Archambeau, S. Guo, and O. Zoeter, “Sparse Bayesian multi-task learning,” in NIPS, 2011.
*   \[90\] M. Yang, Y. Li, and Z. Zhang, “Multi-task learning with Gaussian matrix generalized inverse Gaussian model,” in ICML, 2013.
*   \[91\] Y. Zhang and D.-Y. Yeung, “Learning high-order task relationships in multi-task learning,” in IJCAI, 2013.
*   \[92\] P. Rai, A. Kumar, and H. Daumé III, “Simultaneously leveraging output and task structures for multiple-output regression,” in NIPS, 2012.
*   \[93\] B. Rakitsch, C. Lippert, K. M. Borgwardt, and O. Stegle, “It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals,” in NIPS, 2013.
*   \[94\] A. R. Goncalves, F. J. V. Zuben, and A. Banerjee, “Multi-task sparse structure learning with Gaussian copula models,” JMLR, 2016.
*   \[95\] M. Long, Z. Cao, J. Wang, and P. S. Yu, “Learning multiple tasks with multilinear relationship networks,” in NIPS, 2017.
*   \[96\] Y. Zhang, “Heterogeneous-neighborhood-based multi-task local learning algorithms,” in NIPS, 2013.
*   \[97\] G. Lee, E. Yang, and S. J. Hwang, “Asymmetric multi-task learning based on task relatedness and loss,” in ICML, 2016.
*   \[98\] A. Jalali, P. Ravikumar, S. Sanghavi, and C. Ruan, “A dirty model for multi-task learning,” in NIPS, 2010.
*   \[99\] J. Chen, J. Liu, and J. Ye, “Learning incoherent sparse and low-rank patterns from multiple tasks,” in KDD, 2010.
*   \[100\] J. Chen, J. Zhou, and J. Ye, “Integrating low-rank and group-sparse structures for robust multi-task learning,” in KDD, 2011.
*   \[101\] P. Gong, J. Ye, and C. Zhang, “Robust multi-task feature learning,” in KDD, 2012.
*   \[102\] W. Zhong and J. T. Kwok, “Convex multitask learning with flexible task clusters,” in ICML, 2012.
*   \[103\] A. Zweig and D. Weinshall, “Hierarchical regularization cascade for joint learning,” in ICML, 2013.
*   \[104\] L. Han and Y. Zhang, “Learning tree structure in multi-task learning,” in KDD, 2015.
*   \[105\] P. Jawanpuria and J. S. Nath, “A convex feature learning formulation for latent task structure discovery,” in ICML, 2012.
*   \[106\] B. Gong, Y. Shi, F. Sha, and K. Grauman, “Geodesic flow kernel for unsupervised domain adaptation,” in CVPR, 2012.
*   \[107\] M. Solnon, S. Arlot, and F. R. Bach, “Multi-task regression using minimal penalties,” JMLR, 2012.
*   \[108\] Y. Zhang, Y. Wei, and Q. Yang, “Learning to multitask,” in NIPS, 2018.
*   \[109\] Y. Zhang and D.-Y. Yeung, “Multi-task learning in heterogeneous feature spaces,” in AAAI, 2011.
*   \[110\] S. Han, X. Liao, and L. Carin, “Cross-domain multitask learning with latent probit models,” in ICML, 2012.
*   \[111\] P. Yang, K. Huang, and C. Liu, “Geometry preserving multi-task metric learning,” MLJ, 2013.
*   \[112\] N. Quadrianto, A. J. Smola, T. S. Caetano, S. V. N. Vishwanathan, and J. Petterson, “Multitask learning without label correspondences,” in NIPS, 2010.
*   \[113\] B. Romera-Paredes, H. Aung, N. Bianchi-Berthouze, and M. Pontil, “Multilinear multitask learning,” in ICML, 2013.
*   \[114\] K. Wimalawarne, M. Sugiyama, and R. Tomioka, “Multitask learning meets tensor factorization: Task imputation via convex optimization,” in NIPS, 2014.
*   \[115\] O. Sener and V. Koltun, “Multi-task learning as multi-objective optimization,” in NeurIPS, 2018.
*   \[116\] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” in NeurIPS, 2020.
*   \[117\] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich, “Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks,” in ICML, 2018.
*   \[118\] N. Parikh and S. P. Boyd, “Proximal algorithms,” Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127–239, 2014.
*   \[119\] L. Zhao, Q. Sun, J. Ye, F. Chen, C. Lu, and N. Ramakrishnan, “Multi-task learning for spatio-temporal event forecasting,” in KDD, 2015.
*   \[120\] Y. Li, J. Wang, J. Ye, and C. K. Reddy, “A multi-task learning formulation for survival analysis,” in KDD, 2016.
*   \[121\] L. Zhao, Q. Sun, J. Ye, F. Chen, C. Lu, and N. Ramakrishnan, “Feature constrained multi-task learning models for spatiotemporal event forecasting,” IEEE TKDE, 2017.
*   \[122\] Q. Liu, X. Liao, and L. Carin, “Semi-supervised multitask learning,” in NIPS, 2007.
*   \[123\] Q. Liu, X. Liao, H. Li, J. R. Stack, and L. Carin, “Semisupervised multitask learning,” IEEE TPAMI, 2009.
*   \[124\] Y. Zhang and D. Yeung, “Semi-supervised multi-task regression,” in ECMLPKDD, 2009.
*   \[125\] R. Reichart, K. Tomanek, U. Hahn, and A. Rappoport, “Multi-task active learning for linguistic annotations,” in ACL, 2008.
*   \[126\] A. Acharya, R. J. Mooney, and J. Ghosh, “Active multitask learning using both latent and supervised shared topics,” in SDM, 2014.
*   \[127\] M. Fang and D. Tao, “Active multi-task learning via bandits,” in SDM, 2015.
*   \[128\] H. Li, X. Liao, and L. Carin, “Active learning for semi-supervised multi-task learning,” in ICASSP, 2009.
*   \[129\] K. Lin and J. Zhou, “Interactive multi-task relationship learning,” in ICDM, 2016.
*   \[130\] A. Pentina and C. H. Lampert, “Multi-task learning with labeled and unlabeled tasks,” in ICML, 2017.
*   \[131\] J. Zhang and C. Zhang, “Multitask Bregman clustering,” in AAAI, 2010.
*   \[132\] X. Zhang and X. Zhang, “Smart multi-task Bregman clustering and multi-task kernel clustering,” in AAAI, 2013.
*   \[133\] X. Zhang, X. Zhang, and H. Liu, “Smart multitask Bregman clustering and multitask kernel clustering,” ACM TKDD, 2015.
*   \[134\] Q. Gu, Z. Li, and J. Han, “Learning a kernel for multi-task clustering,” in AAAI, 2011.
*   \[135\] X. Zhang, “Convex discriminative multitask clustering,” IEEE TPAMI, 2015.
*   \[136\] Y. Wang, D. P. Wipf, Q. Ling, W. Chen, and I. J. Wassell, “Multi-task learning for subspace segmentation,” in ICML, 2015.
*   \[137\] X. Zhang, X. Zhang, and H. Liu, “Self-adapted multi-task clustering,” in IJCAI, 2016.
*   \[138\] Y. Yang, Z. Ma, Y. Yang, F. Nie, and H. T. Shen, “Multitask spectral clustering by exploring intertask correlation,” IEEE TCYB, 2015.
*   \[139\] X. Zhu, X. Li, S. Zhang, C. Ju, and X. Wu, “Robust joint graph sparse coding for unsupervised spectral feature selection,” IEEE TNNLS, 2017.
*   \[140\] X. Zhang, X. Zhang, H. Liu, and J. Luo, “Multi-task clustering with model relation learning,” in IJCAI, 2018.
*   \[141\] A. Wilson, A. Fern, S. Ray, and P. Tadepalli, “Multi-task reinforcement learning: A hierarchical Bayesian approach,” in ICML, 2007.
*   \[142\] H. Li, X. Liao, and L. Carin, “Multi-task reinforcement learning in partially observable stochastic environments,” JMLR, 2009.
*   \[143\] A. Lazaric and M. Ghavamzadeh, “Bayesian multi-task reinforcement learning,” in ICML, 2010.
*   \[144\] D. Calandriello, A. Lazaric, and M. Restelli, “Sparse multi-task reinforcement learning,” in NIPS, 2014.
*   \[145\] J. Andreas, D. Klein, and S. Levine, “Modular multitask reinforcement learning with policy sketches,” in ICML, 2017.
*   \[146\] A. A. Deshmukh, Ü. Dogan, and C. Scott, “Multi-task learning for contextual bandits,” in NIPS, 2017.
*   \[147\] A. M. Saxe, A. C. Earle, and B. Rosman, “Hierarchy through composition with multitask LMDPs,” in ICML, 2017.
*   \[148\] T. Bräm, G. Brunner, O. Richter, and R. Wattenhofer, “Attentive multi-task deep reinforcement learning,” in ECMLPKDD, 2019.
*   \[149\] T. Vuong, D. V. Nguyen, T. Nguyen, C. Bui, H. Kieu, V. Ta, Q. Tran, and T. H. Le, “Sharing experience in multitask reinforcement learning,” in IJCAI, 2019.
*   \[150\] M. Igl, A. Gambardella, N. Nardelli, N. Siddharth, W. Böhmer, and S. Whiteson, “Multitask soft option learning,” in UAI, 2020.
*   \[151\] E. Parisotto, J. Ba, and R. Salakhutdinov, “Actor-mimic: Deep multitask and transfer reinforcement learning,” in ICLR, 2016.
*   \[152\] A. A. Rusu, S. G. Colmenarejo, Ç. Gülçehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” in ICLR, 2016.
*   \[153\] S. Omidshafiei, J. Pazis, C. Amato, J. P. How, and J. Vian, “Deep decentralized multi-task multi-agent reinforcement learning under partial observability,” in ICML, 2017.
*   \[154\] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu, “Distral: Robust multitask reinforcement learning,” in NIPS, 2017.
*   \[155\] S. E. Bsat, H. Bou-Ammar, and M. E. Taylor, “Scalable multitask policy gradient reinforcement learning,” in AAAI, 2017.
*   \[156\] S. Sharma and B. Ravindran, “Online multi-task learning using active sampling,” in ICLR Workshop, 2017.
*   \[157\] S. Sharma, A. K. Jha, P. Hegde, and B. Ravindran, “Learning to multi-task by active sampling,” in ICLR, 2018.
*   \[158\] L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu, “IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures,” in ICML, 2018.
*   \[159\] R. Tutunov, D. Kim, and H. Bou-Ammar, “Distributed multitask reinforcement learning with quadratic convergence,” in NeurIPS, 2018.
*   \[160\] X. Lin, H. S. Baweja, G. Kantor, and D. Held, “Adaptive auxiliary task weighting for reinforcement learning,” in NeurIPS, 2019.
*   \[161\] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Sharing knowledge in multi-task deep reinforcement learning,” in ICLR, 2020.
*   \[162\] T. Shu, C. Xiong, and R. Socher, “Hierarchical and interpretable skill acquisition in multi-task reinforcement learning,” in ICLR, 2018.
*   \[163\] M. Hessel, H. Soyer, L. Espeholt, W. Czarnecki, S. Schmitt, and H. van Hasselt, “Multi-task deep reinforcement learning with PopArt,” in AAAI, 2019.
*   \[164\] Z. Guo, B. A. Pires, M. G. Azar, B. Piot, F. Altché, J.-B. Grill, and R. Munos, “Bootstrap latent-predictive representations for multitask reinforcement learning,” in ICML, 2020.
*   \[165\] J. He and R. Lawrence, “A graph-based framework for multi-task multi-view learning,” in ICML, 2011.
*   \[166\] J. Zhang and J. Huan, “Inductive multi-task learning with multiple view data,” in KDD, 2012.
*   \[167\] X. Zhang, X. Zhang, and H. Liu, “Multi-task multi-view clustering for non-negative data,” in IJCAI, 2015.
*   \[168\] X. Zhang, X. Zhang, H. Liu, and X. Liu, “Multi-task multi-view clustering,” IEEE TKDE, 2016.
*   \[169\] X. Zhu, X. Li, and S. Zhang, “Block-row sparse multiview multilabel learning for image classification,” IEEE TCYB, 2016.
*   \[170\] L. Zheng, Y. Cheng, and J. He, “Deep multimodality model for multi-task multi-view learning,” in SDM, 2019.
*   \[171\] A. Niculescu-Mizil and R. Caruana, “Inductive transfer for Bayesian network structure learning,” in AISTATS, 2007.
*   \[172\] J. Honorio and D. Samaras, “Multi-task learning of Gaussian graphical models,” in ICML, 2010.
*   \[173\] D. Oyen and T. Lane, “Leveraging domain knowledge in multitask Bayesian network structure learning,” in AAAI, 2012.
*   \[174\] K. Lin, J. Xu, I. M. Baytas, S. Ji, and J. Zhou, “Multi-task feature interaction learning,” in KDD, 2016.
*   \[175\] Y. Zhang, “Parallel multi-task learning,” in ICDM, 2015.
*   \[176\] O. Dekel, P. M. Long, and Y. Singer, “Online multitask learning,” in COLT, 2006.
*   \[177\] O. Dekel, P. M. Long, and Y. Singer, “Online learning of multiple tasks with a shared loss,” JMLR, 2007.
*   \[178\] G. Lugosi, O. Papaspiliopoulos, and G. Stoltz, “Online multi-task learning with hard constraints,” in COLT, 2009.
*   \[179\] G. Cavallanti, N. Cesa-Bianchi, and C. Gentile, “Linear algorithms for online multitask classification,” JMLR, 2010.
*   \[180\] G. Pillonetto, F. Dinuzzo, and G. D. Nicolao, “Bayesian online multitask learning of Gaussian processes,” IEEE TPAMI, 2010.
*   \[181\] A. Saha, P. Rai, H. Daumé III, and S. Venkatasubramanian, “Online learning of multiple tasks and their relationships,” in AISTATS, 2011.
*   \[182\] K. Murugesan, H. Liu, J. G. Carbonell, and Y. Yang, “Adaptive smoothed online multi-task learning,” in NIPS, 2016.
*   \[183\] P. Yang, P. Zhao, and X. Gao, “Robust online multi-task learning with correlative and personalized structures,” IEEE TKDE, 2017.
*   \[184\] S. Hao, P. Zhao, Y. Liu, S. C. H. Hoi, and C. Miao, “Online multitask relative similarity learning,” in IJCAI, 2017.
*   \[185\] P. Yang, P. Zhao, J. Zhou, and X. Gao, “Confidence weighted multitask learning,” in AAAI, 2019.
*   \[186\] J. Wang, M. Kolar, and N. Srebro, “Distributed multi-task learning,” in AISTATS, 2016.
*   \[187\] S. Liu, S. J. Pan, and Q. Ho, “Distributed multi-task relationship learning,” in KDD, 2017.
*   \[188\] L. Xie, I. M. Baytas, K. Lin, and J. Zhou, “Privacy-preserving distributed multi-task learning with asynchronous updates,” in KDD, 2017.
*   \[189\] V. Smith, C. Chiang, M. Sanjabi, and A. S. Talwalkar, “Federated multi-task learning,” in NIPS, 2017.
*   \[190\] C. Zhang, P. Zhao, S. Hao, Y. C. Soh, B. Lee, C. Miao, and S. C. H. Hoi, “Distributed multi-task classification: A decentralized online learning approach,” MLJ, 2018.
*   \[191\] K. Q. Weinberger, A. Dasgupta, J. Langford, A. J. Smola, and J. Attenberg, “Feature hashing for large scale multitask learning,” in ICML, 2009.
*   \[192\] T. Zhang, B. Ghanem, S. Liu, and N. Ahuja, “Robust visual tracking via multi-task sparse learning,” in CVPR, 2012.
*   \[193\] T. Zhang, B. Ghanem, S. Liu, and N. Ahuja, “Robust visual tracking via structured multi-task sparse learning,” IJCV, 2013.
*   \[194\] Q. Xu, S. J. Pan, H. H. Xue, and Q. Yang, “Multitask learning for protein subcellular location prediction,” IEEE/ACM TCBB, 2011.
*   \[195\] Z. Wu, C. Valentini-Botinhao, O. Watts, and S. King, “Deep neural networks employing multi-task learning and stacked bottleneck features for speech synthesis,” in ICASSP, 2015.
*   \[196\] Q. Hu, Z. Wu, K. Richmond, J. Yamagishi, Y. Stylianou, and R. Maia, “Fusion of multiple parameterisations for DNN-based sinusoidal speech synthesis with multi-task learning,” in InterSpeech, 2015.
*   \[197\] J. Bai, K. Zhou, G. Xue, H. Zha, G. Sun, B. L. Tseng, Z. Zheng, and Y. Chang, “Multi-task learning for learning to rank in web search,” in CIKM, 2009.
*   \[198\] J. Ghosn and Y. Bengio, “Multi-task learning for stock selection,” in NIPS, 1996.
*   \[199\] C. Yuan, W. Hu, G. Tian, S. Yang, and H. Wang, “Multi-task sparse learning with Beta process prior for action recognition,” in CVPR, 2013.
*   \[200\] Y. Qi, O. Tastan, J. G. Carbonell, J. Klein-Seetharaman, and J. Weston, “Semi-supervised multi-task learning for predicting interactions between HIV-1 and human proteins,” Bioinformatics, 2010.
*   \[201\] P. Bell and S. Renals, “Regularization of context-dependent deep neural networks with context-independent multi-task training,” in ICASSP, 2015.
*   \[202\] Z. Chen, S. Watanabe, H. Erdogan, and J. R. Hershey, “Speech enhancement and recognition using multi-task learning of long short-term memory recurrent neural networks,” in InterSpeech, 2015.
*   \[203\] V. W. Zheng, S. J. Pan, Q. Yang, and J. J. Pan, “Transferring multi-device localization models using latent multi-task learning,” in AAAI, 2008.
*   \[204\] R. Collobert and J. Weston, “A unified architecture for natural language processing: Deep neural networks with multitask learning,” in ICML, 2008.
*   \[205\] M. Lapin, B. Schiele, and M. Hein, “Scalable multitask representation learning for scene classification,” in CVPR, 2014.
*   \[206\] A. H. Abdulnabi, G. Wang, J. Lu, and K. Jia, “Multi-task CNN model for attribute prediction,” IEEE TMM, 2015.
*   \[207\] M. Luong, Q. V. Le, I. Sutskever, O. Vinyals, and L. Kaiser, “Multi-task sequence to sequence learning,” in ICLR, 2016.
*   \[208\] J. Yim, H. Jung, B. Yoo, C. Choi, D. Park, and J. Kim, “Rotating your face using multi-task deep neural network,” in CVPR, 2015.
*   \[209\] X. Chu, W. Ouyang, W. Yang, and X. Wang, “Multi-task recurrent neural network for immediacy prediction,” in ICCV, 2015.
*   \[210\] X. Wang, C. Zhang, and Z. Zhang, “Boosted multi-task learning for face verification with applications to web image and video search,” in CVPR, 2009.
*   \[211\] X. Yuan and S. Yan, “Visual classification with multi-task joint sparse representation,” in CVPR, 2010.
*   \[212\] Q. Liu, Q. Xu, V. W. Zheng, H. Xue, Z. Cao, and Q. Yang, “Multi-task learning for cross-platform siRNA efficacy prediction: An in-silico study,” BMC Bioinformatics, 2010.
*   \[213\] A. Ahmed, M. Aly, A. Das, A. J. Smola, and T. Anastasakos, “Web-scale multi-task feature selection for behavioral targeting,” in CIKM, 2012.
*   \[214\] H. Wang, F. Nie, H. Huang, S. L. Risacher, C. H. Q. Ding, A. J. Saykin, and L. Shen, “Sparse multi-task regression and feature selection to identify brain imaging predictors for memory performance,” in ICCV, 2011.
*   \[215\] K. Puniyani, S. Kim, and E. P. Xing, “Multi-population GWA mapping via multi-task regularized regression,” Bioinformatics, 2010.
*   \[216\] J. Zhou, L. Yuan, J. Liu, and J. Ye, “A multi-task learning formulation for predicting disease progression,” in KDD, 2011.
*   \[217\] J. Wan, Z. Zhang, J. Yan, T. Li, B. D. Rao, S. Fang, S. Kim, S. L. Risacher, A. J. Saykin, and L. Shen, “Sparse Bayesian multi-task learning for predicting cognitive outcomes from neuroimaging measures in Alzheimer’s disease,” in CVPR, 2012.
*   \[218\] D. He, D. Kuhn, and L. Parida, “Novel applications of multitask learning and multiple output regression to multiple genetic trait prediction,” Bioinformatics, 2016.
*   \[219\] K. Zhang, J. W. Gray, and B. Parvin, “Sparse multitask regression for identifying common mechanism of response to therapeutic targets,” Bioinformatics, 2010.
*   \[220\] B. Cheng, G. Liu, J. Wang, Z. Huang, and S. Yan, “Multi-task low-rank affinity pursuit for image segmentation,” in ICCV, 2011.
*   \[221\] J. Xu, P. Tan, L. Luo, and J. Zhou, “GSpartan: a geospatio-temporal multi-task learning framework for multi-location prediction,” in SDM, 2016.
*   \[222\] C. Lang, G. Liu, J. Yu, and S. Yan, “Saliency detection by multitask sparsity pursuit,” IEEE TIP, 2012.
*   \[223\] H. Wang, F. Nie, H. Huang, J. Yan, S. Kim, S. L. Risacher, A. J. Saykin, and L. Shen, “High-order multi-task feature learning to identify longitudinal phenotypic markers for Alzheimer’s disease progression prediction,” in NIPS, 2012.
*   \[224\] Q. An, C. Wang, I. Shterev, E. Wang, L. Carin, and D. B. Dunson, “Hierarchical kernel stick-breaking process for multi-task image analysis,” in ICML, 2008.
*   \[225\] M. Alamgir, M. Grosse-Wentrup, and Y. Altun, “Multitask learning for brain-computer interfaces,” in AISTATS, 2010.
*   \[226\] Y. Zhang and D.-Y. Yeung, “Multi-task warped Gaussian process for personalized age estimation,” in CVPR, 2010.
*   \[227\] J. Xu, J. Zhou, and P. Tan, “FORMULA: FactORized MUlti-task LeArning for task discovery in personalized medical models,” in SDM, 2015.
*   \[228\] T. R. Almaev, B. Martínez, and M. F. Valstar, “Learning to transfer: Transferring latent task structures and its application to person-specific facial action unit detection,” in ICCV, 2015.
*   \[229\] A. Liu, Y. Su, W. Nie, and M. S. Kankanhalli, “Hierarchical clustering multi-task learning for joint human action grouping and recognition,” IEEE TPAMI, 2017.
*   \[230\] F. Wu and Y. Huang, “Collaborative multi-domain sentiment classification,” in ICDM, 2015.
*   \[231\] O. Chapelle, P. K. Shivaswamy, S. Vadrevu, K. Q. Weinberger, Y. Zhang, and B. L. Tseng, “Multi-task learning for boosting with application to web search ranking,” in KDD, 2010.
*   \[232\] C. Widmer, J. Leiva, Y. Altun, and G. Rätsch, “Leveraging sequence classification by taxonomy-based multitask learning,” in RECOMB, 2010.
*   \[233\] Y. Zhang, B. Cao, and D.-Y. Yeung, “Multi-domain collaborative filtering,” in UAI, 2010.
*   \[234\] K. M. A. Chai, C. K. I. Williams, S. Klanke, and S. Vijayakumar, “Multi-task Gaussian process learning of robot inverse dynamics,” in NIPS, 2008.
*   \[235\] D.-Y. Yeung and Y. Zhang, “Learning inverse dynamics by Gaussian process regression under the multi-task learning framework,” in The Path to Autonomous Robots, Springer, 2009.
*   \[236\] C. Widmer, N. C. Toussaint, Y. Altun, and G. Rätsch, “Inferring latent task structure for multitask learning by multiple kernel learning,” BMC Bioinformatics, 2010.
*   \[237\] A. Ahmed, A. Das, and A. J. Smola, “Scalable hierarchical multitask learning algorithms for conversion optimization in display advertising,” in WSDM, 2014.
*   \[238\] J. Zheng and L. M. Ni, “Time-dependent trajectory regression on road networks via multi-task learning,” in AAAI, 2013.
*   \[239\] X. Lu, Y. Wang, X. Zhou, Z. Zhang, and Z. Ling, “Traffic sign recognition via multi-modal tree-structure embedded multi-task learning,” IEEE TITS, 2017.
*   \[240\] F. Mordelet and J. Vert, “ProDiGe: Prioritization of disease genes with multitask machine learning from positive and unlabeled examples,” BMC Bioinformatics, 2011.
*   \[241\] J. Xu, P. Tan, J. Zhou, and L. Luo, “Online multi-task learning framework for ensemble forecasting,” IEEE TKDE, 2017.
*   \[242\] M. Kshirsagar, J. G. Carbonell, and J. Klein-Seetharaman, “Multitask learning for host-pathogen protein interactions,” Bioinformatics, 2013.
*   \[243\] L. Han, L. Li, F. Wen, L. Zhong, T. Zhang, and X. Wan, “Graph-guided multi-task sparse learning model: a method for identifying antigenic variants of influenza A(H3N2) virus,” Bioinformatics, 2019.
*   \[244\] Z. Hong, X. Mei, D. V. Prokhorov, and D. Tao, “Tracking via robust multi-task multi-view joint sparse representation,” in ICCV, 2013.
*   \[245\] Y. Yan, E. Ricci, S. Ramanathan, O. Lanz, and N. Sebe, “No matter where you are: Flexible graph-guided multi-task learning for multi-view head pose classification under target motion,” in ICCV, 2013.
*   \[246\] M. Kshirsagar, K. Murugesan, J. G. Carbonell, and J. Klein-Seetharaman, “Multitask matrix completion for learning protein interactions across diseases,” Journal of Computational Biology, 2017.
*   \[247\] C. Su, F. Yang, S. Zhang, Q. Tian, L. S. Davis, and W. Gao, “Multi-task learning with low rank attribute embedding for person re-identification,” in ICCV, 2015.
*   \[248\] Y. Zhang and Q. Yang, “A survey on multi-task learning,” CoRR, 2017.
*   \[249\] J. Baxter, “Learning internal representations,” in COLT, 1995.
*   \[250\] J. Baxter, “A model of inductive bias learning,” JAIR, 2000.
*   \[251\] A. Maurer, “Bounds for linear multi-task learning,” JMLR, 2006.
*   \[252\] A. Maurer, “The Rademacher complexity of linear transformation classes,” in COLT, 2006.
*   \[253\] B. Juba, “Estimating relatedness via data compression,” in ICML, 2006.
*   \[254\] S. Ben-David and R. S. Borbely, “A notion of task relatedness yielding provable multiple-task learning guarantees,” MLJ, 2008.
*   \[255\] S. M. Kakade, S. Shalev-Shwartz, and A. Tewari, “Regularization techniques for learning with matrices,” JMLR, 2012.
*   \[256\] M. Pontil and A. Maurer, “Excess risk bounds for multitask learning with trace norm regularization,” in COLT, 2013.
*   \[257\] A. Pentina and S. Ben-David, “Multi-task and lifelong learning of kernels,” in ALT, 2015.
*   \[258\] Y. Zhang, “Multi-task learning and algorithmic stability,” in AAAI, 2015.
*   \[259\] A. Maurer, M. Pontil, and B. Romera-Paredes, “The benefit of multitask representation learning,” JMLR, 2016.
*   \[260\] N. Yousefi, Y. Lei, M. Kloft, M. Mollaghasemi, and G. Anagnastapolous, “Local Rademacher complexity-based learning guarantees for multi-task learning,” JMLR, 2018.
*   \[261\] A. Argyriou, C. A. Micchelli, and M. Pontil, “When is there a representer theorem? vector versus matrix regularizers,” JMLR, 2009.
*   \[262\] A. Argyriou, C. A. Micchelli, and M. Pontil, “On spectral learning,” JMLR, 2010.
*   \[263\] K. Lounici, M. Pontil, A. B. Tsybakov, and S. A. van de Geer, “Taking advantage of sparsity in multi-task learning,” in COLT, 2009.
*   \[264\] G. Obozinski, M. J. Wainwright, and M. I. Jordan, “Support union recovery in high-dimensional multivariate regression,” Annals of Statistics, 2011.
*   \[265\] M. Kolar, J. D. Lafferty, and L. A. Wasserman, “Union support recovery in multi-task learning,” JMLR, 2011.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")