# A Survey on Multi-Task Learning

Yu Zhang and Qiang Yang

Abstract—Multi-Task Learning (MTL) is a learning paradigm in machine learning and its aim is to leverage useful information contained in multiple related tasks to help improve the generalization performance of all the tasks. In this paper, we give a survey for MTL from the perspective of algorithmic modeling, applications and theoretical analyses. For algorithmic modeling, we give a definition of MTL and then classify different MTL algorithms into five categories, including feature learning approach, low-rank approach, task clustering approach, task relation learning approach and decomposition approach as well as discussing the characteristics of each approach. In order to improve the performance of learning tasks further, MTL can be combined with other learning paradigms including semi-supervised learning, active learning, unsupervised learning, reinforcement learning, multi-view learning and graphical models When the number of tasks is large or the data dimensionality is high, we review online, parallel and distributed MTL models as well as dimensionality reduction and feature hashing to reveal their computational and storage advantages. Many real-world applications use MTL to boost their performance and we review representative works in this paper. Finally, we present theoretical analyses and discuss several future directions for MTL.

Index Terms—Multi-Task Learning, Machine Learning, Artificial Intelligence

## 1 INTRODUCTION

H <sup>UMAN</sup> <sup>can</sup> <sup>learn</sup> <sup>multiple</sup> <sup>tasks</sup> <sup>simultaneously</sup> <sup>and</sup> <sup>during</sup>this learning process, human can use the knowledge learned this learning process, human can use the knowledge learned in a task to help the learning of another task. For example, according to our experience in learning to play tennis and squash, we find that the skill of playing tennis can help learn to play squash and vice versa. Inspired by such human learning ability, Multi-Task Learning (MTL) [1], a learning paradigm in machine learning, aims to learn multiple related tasks jointly so that the knowledge contained in a task can be leveraged by other tasks, with the hope of improving the generalization performance of all the tasks at hand.

At its early stage, an important motivation of MTL is to alleviate the data sparsity problem where each task has a limited number of labeled data. In the data sparsity problem, the number of labeled data in each task is insufficient to train an accurate learner, while MTL aggregates the labeled data in all the tasks in the spirit of data augmentation to obtain a more accurate learner for each task. From this perspective, MTL can help reuse existing knowledge and reduce the cost of manual labeling for learning tasks. When the era of “big data” comes in some areas such as computer vision and Natural Language Processing (NLP), it is found that deep MTL models can achieve better performance than their single-task counterparts. One reason that MTL is effective is that it utilizes more data from different learning tasks when compared with single-task learning. With more data, MTL can learn more robust and universal representations for multiple tasks and more powerful models, leading to better knowledge sharing among tasks, better performance of each task and low risk of overfitting in each task.

MTL is related to other learning paradigms in machine learning, including transfer learning [2], multi-label learning [3] and multi-output regression. The setting of MTL is similar to that of transfer learning but with significant differences. In MTL, there is no distinction among different tasks and the objective is to improve the performance of all the tasks. However, transfer learning is to improve the performance of a target task with the help of source tasks, hence the target task plays a more important role than source tasks. In a word, MTL treats all the tasks equally but in transfer learning the target task attracts most attentions. From the perspective of the knowledge flow, flows of knowledge transfer in transfer learning are from source task(s) to the target task, but in multi-task learning, there are flows of knowledge sharing between any pair of tasks, which is illustrated in Fig. 1(a). Continual learning [4], in which tasks come sequentially, learns tasks one by one, while MTL is to learn multiple tasks together. In multi-label learning and multi-output regression, each data point is associated with multiple labels which can be categorical or numeric. If we treat each of all the possible labels as a task, multi-label learning and multi-output regression can be viewed in some sense as a special case of multi-task learning where different tasks always share the same data during both the training and testing phrases. On the one hand, such characteristic in multi-label learning and multi-output regression leads to different research issues from MTL. For example, the ranking loss, which enforces the scores (e.g., the classification probability) of labels associated with a data point to be larger than those of absent labels, can be used for multi-label learning but it does not fit MTL where different tasks possess different data. On the other hand, this characteristic in multi-label learning and multi-output regression is invalid in MTL problems. For example, in a MTL problem discussed in Section 2.7 where each task is to predict the disease symptom score of Parkinson for a patient based on 19 bio-medical features, different patients/tasks should not share the bio-medical data. In a word, multi-label learning and multi-output regression are different from multi-task learning as illustrated in Fig. 1(b) and hence we will not survey literature on multi-label learning and multi-output regression. Moreover, multi-view learning is another learning paradigm in machine learning, where each data point is associated with multiple views, each of which consists of a set of features. Even though different views have different sets of features, all the views are used together to learn for the same task and hence multi-view learning belongs to single-task learning with multiple sets of features, which is different from MTL as shown in Fig. 1(c).

![](images/b8420b3ae9f66841bcafecdc8190165d581e4859efa5cd063385f1ab8aa6a380.jpg)  
(a) MTL vs. transfer learning

![](images/8ef0f8f710724162a816d7f53a723bdf2ff0ec3424c808a0acc66636e9087a7f.jpg)  
(b) MTL vs. multi-label learning/multioutput regression

![](images/46d1dc0073a4c173a81f1abaeff5bb8c7b212987b4a5f681cf5001dec690c164.jpg)  
(c) MTL vs. multi-view learning  
Fig. 1. Illustrations for differences between MTL and other learning paradigms.

Over past decades, MTL has attracted many attentions in the artificial intelligence and machine learning communities. Many MTL models have been devised and many MTL applications in other areas have been exploited. Moreover, many analyses have been conducted to study theoretical problems in MTL. This paper serves as a survey on MTL from the perspective of algorithmic modeling, applications and theoretical analyses. For algorithmic modeling, we first give a definition for MTL and then classify different MTL algorithms into five categories: feature learning approach which can be further categorized into feature transformation and feature selection approaches, lowrank approach, task clustering approach, task relation learning approach and decomposition approach. After that, we discuss the combination of MTL with other learning paradigms, including semi-supervised learning, active learning, unsupervised learning, reinforcement learning, multi-view learning and graphical models. To handle a large number of tasks, we review online, parallel and distributed MTL models. For data in a high-dimensional space, feature selection, dimensionality reduction and feature hashing are introduced as vital tools to process them. As a promising learning paradigm, MTL has many applications in various areas and here we briefly review its applications in computer vision, bioinformatics, health informatics, speech, NLP, web, etc. From the perspective of theoretical analyses on MTL, we review relevant works. At last, we discuss several future directions for MTL.<sup>1</sup>

## 2 MTL MODELS

In order to fully characterize MTL, we first give the definition of MTL.

Definition (Multi-Task Learning). Given m learning tasks $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { m }$ where all the tasks or a subset of them are related, multi-task learning aims to learn the m tasks together to improve the learning of a model for each task $\mathcal { T } _ { i }$ by using the knowledge contained in all or some of other tasks.

1. For an introduction to MTL without technical details, please refer to [5].

Based on the definition of MTL, we focus on supervised learning tasks in this section since most MTL studies fall in this setting and for other types of tasks, we review them in the next section. In the setting of supervised learning tasks, a task $\mathcal { T } _ { i }$ is usually accompanied by a training dataset $\mathcal { D } _ { i }$ consisting of $n _ { i }$ training samples, i.e., $\mathcal { D } _ { i } = \{ \mathbf { x } _ { j } ^ { i } , y _ { j } ^ { i } \} _ { j = 1 } ^ { n _ { i } }$ , where $\mathbf { x } _ { j } ^ { i } \in \mathbb { R } ^ { d _ { i } }$ is the jth training instance in $\mathcal { T } _ { i }$ and $y _ { j } ^ { i }$ is its label. We denote by $\mathbf { X } ^ { i }$ the training data matrix for $\mathcal { T } _ { i } ,$ i.e., $\mathbf { X } ^ { i } = ( \mathbf { x } _ { 1 } ^ { i } , \ldots , \mathbf { x } _ { n _ { i } } ^ { i } )$ When different tasks lie in the same feature space implying that $d _ { i }$ equals $d _ { j }$ for any $i \neq j$ , this setting is the homogeneousfeature MTL, and otherwise it corresponds to heterogeneousfeature MTL. Without special explanation, the default MTL setting is the homogeneous-feature MTL. Here we need to distinguish the heterogeneous-feature MTL from the heterogeneous MTL. In [6], the heterogeneous MTL is considered to consist of different types of supervised tasks including classification and regression problems, and here we generalize it to a more general setting that the heterogeneous MTL consists of tasks with different types including supervised learning, unsupervised learning, semisupervised learning, reinforcement learning, multi-view learning and graphical models. The opposite to the heterogeneous MTL is the homogeneous MTL which consists of tasks with only one type. In a word, the homogeneous and heterogeneous MTL differ in the type of learning tasks while the homogeneous-feature MTL is different from the heterogeneous-feature MTL in terms of the original feature representations. Similarly, without special explanation, the default MTL setting is the homogeneous MTL.

In order to characterize the relatedness in the definition of MTL, there are three issues to be addressed: when to share, what to share and how to share.

The ‘when to share’ issue is to make choices between singletask and multi-task models for a multi-task problem. Currently such decision is made by human experts and there are few works to study it. A simple solution is to formulate such decision as a model selection problem and then use model selection techniques, e.g., cross validation, to make decisions, but this solution is usually computational heavy and may require much more training data. An advanced solution we think is to use multi-task models which can degenerate to their single-task counterparts given some form of model parameters, for example, problem (33) presented in Section 2.8 which can reduce to multiple single-task models with the learning of different tasks decoupled when a parameter Σ becomes diagonal. In this case, we can let the training data determine the form of Σ to make an implicit choice.

‘What to share’ needs to determine the form through which knowledge sharing among all the tasks could occur. Usually, there are three forms for ‘what to share’, including feature, instance and parameter. Feature-based MTL aims to learn common features among different tasks as a way to share knowledge. Instancebased MTL identifies useful data instances in a task for other tasks and then shares knowledge via the identified instances. Parameter-based MTL uses model parameters (e.g., coefficients in linear models or weights in deep models) in a task to help learn model parameters in other tasks in some ways, for example, the regularization. Existing MTL studies mainly focus on featurebased and parameter-based methods, and only a few works belong to the instance-based method. A representative instance-based method is the multi-task distribution matching method proposed in [7], which first estimates density ratios between probabilities that each instance as well as its label belongs to both its own task and a mixture of all the tasks and then uses all the weighted training data from all the tasks based on the estimated density ratios to learn model parameters for each task. Since the studies on instance-based MTL are few, we mainly review feature-based and parameter-based MTL models.

After determining ‘what to share’, ‘how to share’ specifies concrete ways to share knowledge among tasks. In feature-based MTL, there is a primary approach: feature learning approach. The feature learning approach focuses on learning common feature representations for multiple tasks based on shallow or deep models, where the learned common feature representation can be a subset or a transformation of the original feature representation. In parameter-based MTL, there are four main approaches: lowrank approach, task clustering approach, task relation learning approach and decomposition approach. The low-rank approach interprets the relatedness of multiple tasks as the low rankness of the parameter matrix of these tasks. The task clustering approach is to identify task clusters, each of which contains similar tasks. The task relation learning approach aims to learn quantitative relations between tasks from data automatically. The decomposition approach decomposes the model parameters of all the tasks into two or more components, which are penalized by different regularizers.

In summary, there are mainly five approaches in the featurebased and parameter-based MTL. In the following sections, we review these approaches in a chronological order to reveal the relations and evolutions among different models.

## 2.1 Feature Learning Approach

Since tasks are related, it is intuitive to assume that different tasks share a common feature representation based on the original features. One reason to learn common feature representations instead of directly using the original ones is that the original representation may not have enough expressive power for multiple tasks. With the training data in all the tasks, a more powerful representation can be learned for all the tasks and this representation can bring the improvement on the performance.

Based on the relationship between the original feature representation and the learned one, we can further classify this category into two sub-categories. The first sub-category is the feature transformation approach where the learned representation is a linear or nonlinear transformation of the original representation and in this approach, each feature in the learned representation is different from the original features. Different from this approach, the feature selection approach, the second sub-category, selects a subset of the original features as the learned representation and hence the learned representation is similar to the original one by eliminating useless features based on different criteria. In the following, we introduce these two approaches.

## 2.1.1 Feature Transformation Approach

The multi-layer feedforward neural network [1], which belongs to the feature transformation approach, is one of the earliest model for multi-task learning. To see how the multi-layer feedforward neural network is constructed for MTL, in Figure 2 we show an example with an input layer, a hidden layer and an output layer. The input layer receives training instances from all the tasks and the output layer has m output units with one for each task. Here the outputs of the hidden layer can be viewed as the common feature representation learned for the m tasks and the transformation from the original representation to the learned one depends on the weights connecting the input and hidden layers as well as the activation function adopted in the hidden units. Hence, if the activation function in the hidden layer is linear, then the transformation is a linear function and otherwise it is nonlinear. Compared with multi-layer feedforward neural networks used for single-task learning, the difference in the network architecture lies in the output layers where in single-task learning, there is only one output unit while in MTL, there are m ones. In [8], the radial basis function network, which has only one hidden layer, is extended to MTL by greedily determining the structure of the hidden layer. Different from these neural network models, Silver et al. [9] propose a context-sensitive multi-task neural network which has only one output unit shared by different tasks but has a task-specific context as an additional input.

![](images/9160ef85f5d6e42ae3677629bc87ab13c4381824fbfb77a0f1511022a51e15e5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  InputLayer["Input Layer"] --> HiddenLayer["Hidden Layer"]
  HiddenLayer --> OutputLayer["Output Layer"]
  Feature1["Feature 1"] --> HiddenLayer
  HiddenLayer --> Output1["Output for task 1"]
  Featured["Feature d"] --> HiddenLayer
  HiddenLayer --> Output2["Output for task m"]
  Output1 --> Feature1
  Output1 --> HiddenLayer
  Output1 --> Output2
  Featured --> HiddenLayer
  HiddenLayer --> Output2
```
</details>

Fig. 2. An example for the multi-task feedforward neural network with an input layer, a hidden layer and an output layer.

Different from multi-layer feedforward neural networks which are connectionist models, the multi-task feature learning (MTFL) method [10] is formulated under the regularization framework with the objective function as

$$
\min _ {\mathbf {A}, \mathbf {U}, \mathbf {b}} \sum_ {i = 1} ^ {m} \frac {1}{n _ {i}} \sum_ {j = 1} ^ {n _ {i}} l (y _ {j} ^ {i}, (\mathbf {a} ^ {i}) ^ {T} \mathbf {U} ^ {T} \mathbf {x} _ {j} ^ {i} + b _ {i}) + \lambda \| \mathbf {A} \| _ {2, 1} ^ {2}
$$

$$
\mathrm{s.t.} \mathbf {U} \mathbf {U} ^ {T} = \mathbf {I}, \tag {1}
$$

where $l ( \cdot , \cdot )$ denotes a loss function such as the hinge loss or square loss, $\mathbf { b } _ { . } = ( b _ { 1 } , \ldots , b _ { m } ) ^ { T }$ is a vector of offsets in all the tasks, $\mathbf { U } \in \mathbb { R } ^ { d \times d }$ is a square transformation matrix, $\mathbf { A } \in \mathbb { R } ^ { d \times m }$ contains model parameters of all the tasks with its ith column $\mathbf { a } ^ { i }$ as model parameters for the ith task after the transformation, the $\ell _ { 2 , 1 }$ norm of a matrix A denoted by $\| \mathbf { A } \| _ { 2 , 1 }$ equals the sum of the $\ell _ { 2 }$ norm of rows in A, I denotes an identity matrix with an appropriate size, and λ is a positive regularization parameter. The first term in the objective function of problem (1) measures the empirical loss on the training sets of all the tasks and the second one is to enforce A to be row-sparse via the $\ell _ { 2 , 1 }$ norm which is equivalent to selecting features after the transformation, while the constraint enforces U to be orthogonal. Different from the multilayer feedforward neural network whose hidden representations may be redundant, the orthogonality of U can prevent the MTFL method from it. As proved in [10], problem (1) is equivalent to

$$
\min _ {\mathbf {W}, \mathbf {D}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \mathrm{tr} (\mathbf {W} ^ {T} \mathbf {D} ^ {- 1} \mathbf {W}) \text {s.t.} \mathbf {D} \succeq \mathbf {0}, \mathrm{tr} (\mathbf {D}) \leq 1, \tag {2}
$$

where $\begin{array} { r } { L ( \mathbf { W } , \mathbf { b } ) = \sum _ { i = 1 } ^ { m } \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } l ( y _ { j } ^ { i } , ( \mathbf { w } ^ { i } ) ^ { T } \mathbf { x } _ { j } ^ { i } + b _ { i } ) } \end{array}$ denotes the total training loss, $\operatorname { t r } ( \cdot )$ denotes the trace of a square matrix, $\mathbf { w } ^ { i } = \mathbf { U } \mathbf { a } ^ { i }$ is the model parameter for $\mathcal { T } _ { i } , \mathbf { W } = ( \mathbf { w } ^ { 1 } , \dots , \mathbf { w } ^ { m } )$ 0 denotes a zero vector or matrix with an appropriate size, ${ { \bf { M } } ^ { - 1 } }$ for any square matrix M denotes its inverse when it is nonsingular or otherwise its pseudo inverse, and $\mathbf { B } \succeq \mathbf { C }$ means that $\mathbf { B } - \mathbf { C }$ is positive semidefinite. Based on this formulation, we can see that the MTFL method is to learn a feature covariance D for all the tasks, which will be interpreted in Section 2.8 from a probabilistic perspective. Given D, the learning of different tasks can be decoupled and this can facilitate the parallel computing. When given W, D has an analytical solution as $\mathbf { D } = ( \mathbf { W } ^ { T } \mathbf { W } ) ^ { \frac { 1 } { 2 } } / \mathrm { t r } \left( ( \mathbf { W } ^ { T } \mathbf { W } ) ^ { \frac { 1 } { 2 } } \right)$ and by plugging this solution into problem (2), we can see that the regularizer on W is the squared trace norm. Then Argyriou et al. [11] extend problem (2) to a general formulation where the second term in the objective function becomes $\lambda \mathrm { t r } ( \mathbf { W } ^ { T } f ( \mathbf { D } ) \mathbf { W } )$ with $f ( \mathbf { D } )$ operating on the spectrum of D and discuss the condition on $f ( \cdot )$ to make the whole problem convex.

Similar to the MTFL method, the multi-task sparse coding method [12] is to learn a linear transformation on features with the objective function formulated as

$$
\min _ {\mathbf {A}, \mathbf {U}, \mathbf {b}} L (\mathbf {U A}, \mathbf {b}) \text {s.t.} \| \mathbf {a} ^ {i} \| _ {1} \leq \lambda \forall i \in [ m ], \| \mathbf {u} ^ {j} \| _ {2} \leq 1 \forall j \in [ D ], \tag {3}
$$

where $\mathbf { a } ^ { i }$ , the ith column of A, contains model parameters of the ith task, $\mathbf { u } ^ { j }$ is the jth column in U, [c] for an integer c denotes a set of integers from 1 to $c , \parallel \cdot \parallel _ { 1 }$ denotes the $\ell _ { 1 }$ norm of a vector or matrix and equals the sum of the absolute value of its entries, and $\| \cdot \| _ { 2 }$ denotes the $\ell _ { 2 }$ norm of a vector. Here the transformation $\textbf { U } \in \overset { \because } { \mathbb { R } } ^ { d \times D }$ is also called the dictionary in sparse coding and shared by all the tasks. Compared with the MTFL method where U in problem (1) is a $d \times d$ orthogonal matrix, U in problem (3) is overcomplete, which implies that D is larger than $d ,$ with each column having a bounded $\ell _ { 2 }$ norm. Another difference is that in problem (1) A is enforced to be row-sparse but in problem (3) it is only sparse via the first constraint. With a similar idea to the multi-task sparse coding method, Zhu et al. [13] propose a multi-task infinite support vector machine via the Indian buffet process and the difference is that in [13] the dictionary is sparse and model parameters are non-sparse. In [14], the spike and slab prior is used to learn sparse model parameters for multi-output regression problems where transformed features are induced by Gaussian processes and shared by different outputs.

Recently deep learning becomes popular due to its capacity to learn nonlinear features, which facilitates the learning of invariant features for multiple tasks, and hence many deep multi-task models belonging to this approach have been proposed with each task modeled by a deep neural network. Here we classify deep multitask models in this approach into three main categories. The first category [15], [16], [17], [18], [19] is to learn a common feature representation for multiple tasks by sharing first several layers in a similar architecture to Fig. 2. However, different from Fig. 2, deep

MTL models in this category have a large number of shared layers, which have general structures such as convolutional layers and pooling layers. Building on the first category, the second category is to use adversarial learning, which is inspired by generative adversarial networks, to learn a common feature representation for MTL as did in [20], [21]. Specifically, there are three networks in such adversarial multi-task models, including a feature network $N _ { f }$ , a classification network $N _ { c }$ and a domain network $N _ { d }$ . Based on $N _ { f } , N _ { c }$ is to minimize the training loss for all the tasks, while $N _ { d }$ aims to distinguish which task a data instance is from. The objective function of such models is usually formulated as

$$
\min _ {\theta_ {f}, \theta_ {c}} \max _ {\theta_ {d}} \sum_ {i = 1} ^ {m} \frac {1}{n _ {i}} \sum_ {j = 1} ^ {n _ {i}} \left(l (y _ {j} ^ {i}, N _ {c} (N _ {f} (\mathbf {x} _ {j} ^ {i}))) - l _ {c e} (d _ {j} ^ {i}, N _ {d} (N _ {f} (\mathbf {x} _ {j} ^ {i})))\right),
$$

where $\theta _ { f } , \theta _ { c } , \theta _ { d }$ denote parameters of three networks $N _ { f } , N _ { c } , N _ { d }$ respectively, $d _ { j } ^ { i } \in \{ 1 , \dots , m \}$ denotes the task index/label of $\mathbf { x } _ { j } ^ { i } ,$ and $l _ { c e } ( \cdot , \cdot )$ denotes the cross-entropy loss. Based on this minimax problem, $N _ { f }$ is to minimize the training loss for all the tasks and maximize the cross-entropy loss to fool the domain network to make the learned feature representation indistinguishable to all the tasks. When there is no domain network, this category can reduce to the first category. Moreover, in [21], each task can learn its specific feature representation to increase the expressive power of the whole model. The last category is to learn different but related feature representations for different tasks with the cross-stitch network [22] as a representative model. Specifically, given two tasks A and B with an identical network architecture, $\bar { x } _ { A } ^ { i , \bar { j } } ( x _ { B } ^ { i , j } )$ ) denotes the hidden feature outputted by the jth unit of the ith hidden layer for task A (B). Then we can define the cross-stitch operation on $x _ { A } ^ { i , j }$ and $x _ { B } ^ { i , j }$ as $\left( \begin{array} { l } { { \tilde { x } _ { A } ^ { i , j } } } \\ { { \tilde { x } _ { B } ^ { i , j } } } \end{array} \right) ~ = ~ \left( \begin{array} { l l } { { \alpha _ { A A } } } & { { \alpha _ { A B } } } \\ { { \alpha _ { B A } } } & { { \alpha _ { B B } } } \end{array} \right) \left( \begin{array} { l } { { { \bf \bar { \alpha } } _ { A } ^ { i , j } } } \\ { { x _ { B } ^ { i , j } } } \end{array} \right)$ where $\tilde { x } _ { A } ^ { i , j }$ and $\tilde { x } _ { B } ^ { i , j }$ are new hidden features after learning the two tasks jointly. When both $\alpha _ { A B }$ and $\alpha _ { B A }$ equal 0, training the two networks jointly is equivalent to training them independently. The network architecture of the cross-stitch network is shown in Fig. 3. Here matrix α, which is defined as $\pmb { \alpha } \equiv \left( \begin{array} { l l } { { \alpha _ { A A } } } & { { \alpha _ { A B } } } \\ { { \alpha _ { B A } } } & { { \alpha _ { B B } } } \end{array} \right)$ encodes feature-level task relations between the two tasks and it can be learned via the backpropagation method.

![](images/6d3e0a95a2c4d0d8d04a6866d6f01a1b5723e54651fb0fd8949794f8a454ee6e.jpg)

<details>
<summary>flowchart</summary>

This diagram illustrates a neural network architecture with convolutional layers (conv1-conv5) and fully connected layers (fc6-fc8), showing the flow of inputs from A to B through alpha nodes.
</details>

Fig. 3. The architecture for the cross-stitch network.

## 2.1.2 Feature Selection Approach

One way to do feature selection in MTL is to use the $\ell _ { p , q }$ norm denoted by $\lVert \mathbf { W } \rVert _ { p , q } \equiv \lVert ( \lVert \mathbf { w } _ { 1 } \rVert _ { p } , \dots , \lVert \mathbf { w } _ { d } \rVert _ { p } ) \rVert _ { q } ,$ where w denotes the ith row of W and $\| \cdot \| _ { p } ^ { - }$ denotes the $\ell _ { p }$ norm of a vector, to achieve the group sparsity. Obozinski et al. [23] are among the first to study the multi-task feature selection (MTFS) problem based on the $\ell _ { 2 , 1 }$ norm with the objective function formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \| \mathbf {W} \| _ {2, 1}. \tag {4}
$$

The regularizer on W in problem (4) is to enforce W to be rowsparse, which in turn helps select important features. In [23], a path-following algorithm is proposed to solve problem (4) and then Liu et al. [24] employ an optimal first-order optimization method to solve it. Compared with problem (1), we can see that problem (4) is similar to the MTFL method without learning the transformation U. Lee et al. [25] propose a weighted $\ell _ { 2 , 1 }$ norm for multi-task feature selection where the weights can be learned as well and problem (4) is extended in [26] to a general case where feature groups can overlap with each other. In order to make problem (4) more robust to outliers, a square-root loss function is investigated in [27]. Moreover, in order to make speedup, a safe screening method is proposed in [28] to filter out useless features corresponding to zero rows in W before optimizing problem (4). Liu et al. [29] propose to use the $\ell _ { \infty , 1 }$ norm to select features with the objective function formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \| \mathbf {W} \| _ {\infty , 1}. \tag {5}
$$

A block coordinate descent method is proposed to solve problem (5). In general, we can use the $\ell _ { p , q }$ norm to select features for MTL.

In order to attain a more sparse subset of features, Gong et al. [30] propose a capped $\boldsymbol { \mathbf { \ell } } _ { - } \ell _ { p , 1 }$ regularizer for multi-task feature selection where $p = \bar { 1 }$ or 2 and the objective function is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \sum_ {i = 1} ^ {d} \min (\| \mathbf {w} _ {i} \| _ {p}, \theta), \tag {6}
$$

where $\mathbf { w } _ { i }$ denotes the ith row of W. With a given threshold $\theta ,$ the $\mathrm { c a p p e d } { \cdot } \ell _ { p , 1 }$ regularizer (i.e., the second term in problem (6)) focuses on rows with smaller $\ell _ { p }$ norms than θ, which is more likely to be sparse. When θ becomes large enough, the capped- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { p , 1 }$ regularizer becomes $\| \mathbf { W } \| _ { p , 1 }$ and hence problem (6) degenerates to problem (4) or (5) when p equals 2 or ∞.

Lozano and Swirszcz [31] propose a multi-level Lasso for MTL where the (j, i)th entry in the parameter matrix W is defined as $w _ { j i } = \theta _ { j } \hat { w } _ { j i }$ . When $\theta _ { j }$ is equal to 0, $w _ { j i }$ becomes 0 for $i \in [ m ]$ and hence the jth feature is not selected by the model. In this sense, $\theta _ { j }$ controls the global sparsity for the jth feature among the m tasks. Moreover, when $\hat { w } _ { j i }$ becomes $0 , w _ { j i }$ is also 0 for i only, implying that the jth feature is not useful for task $\mathcal { T } _ { i } ,$ and so $\hat { w } _ { j i }$ is a local indicator for the sparsity in task $\tau _ { j }$ . Based on these observations, $\theta _ { j }$ and $\hat { w } _ { j i }$ are expected to be sparse, leading to the objective function formulated as

$$
\min _ {\boldsymbol {\theta}, \hat {\mathbf {W}}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda_ {1} \| \boldsymbol {\theta} \| _ {1} + \lambda_ {2} \| \hat {\mathbf {W}} \| _ {1} \text {s.t.} w _ {j i} = \theta_ {j} \hat {w} _ {j i}, \theta_ {j} \geq 0, \tag {7}
$$

where $\begin{array} { r c l } { \pmb { \theta } } & { = } & { ( \theta _ { 1 } , \ldots , \theta _ { d } ) ^ { T } , \hat { \textbf { W } } = \mathbf { \Pi } ( \hat { \mathbf { w } } ^ { 1 } , \ldots , \hat { \mathbf { w } } ^ { m } ) } \end{array}$ , and the nonnegative constraint on $\theta _ { j }$ is to keep the model identifiability. It has been proved in [31] that problem (7) leads to a regularizer $\textstyle \sum _ { j = 1 } ^ { d } { \sqrt { \| \mathbf { w } _ { j } \| _ { 1 } } }$ , the square root of the $\ell _ { 1 , \frac { 1 } { 9 } }$ norm regularization. Moreover, Wang et al. [32] extend problem (7) to a general situation where the regularizer becomes $\begin{array} { r } { \lambda _ { 1 } \sum _ { i = 1 } ^ { m } \| \hat { \mathbf { w } } ^ { i } \| _ { p } ^ { \bar { p } } + \lambda _ { 2 } \| \pmb { \theta } \| _ { q } ^ { q } } \end{array}$ By utilizing a priori information describing the task relations in a hierarchical structure, Han et al. [33] propose a multi-component product based decomposition for $w _ { i j }$ where the number of components in the decomposition can be arbitrary instead of only 2 in [31], [32]. Similar to [31], Jebara [34] proposes to learn a binary indicator vector to do multi-task feature selection based on the maximum entropy discrimination formalism.

Similar to [33] where a priori information is given to describe task relations in a hierarchical/tree structure, Kim and Xing [35] utilize the given tree structure to design a regularizer on W as $\begin{array} { r } { f ( \mathbf { W } ) = \sum _ { i = 1 } ^ { d } \sum _ { v \in V } \lambda _ { v } \| \mathbf { w } _ { i , G _ { v } } \| _ { 2 } , } \end{array}$ , where $\bar { V }$ denotes the set of nodes in the given tree structure, $G _ { v }$ denotes the set of leaf nodes $( \mathrm { i . e . }$ , tasks) in a sub-tree rooted at node $v ,$ and ${ \bf w } _ { i , G _ { \imath } }$ denotes a subvector of the ith row of W indexed by $G _ { v }$ . This regularizer not only enforces each row of W to be sparse as the $\ell _ { 2 , 1 }$ norm did in problem (4), but also induces the sparsity in subsets of each row in W based on the tree structure.

Different from conventional multi-task feature selection methods which assume that different tasks share a set of original features, Zhou et al. [36] consider a different scenario where useful features in different tasks have no overlapping. In order to achieve this, an exclusive Lasso model is proposed with the objective function formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \| \mathbf {W} \| _ {1, 2} ^ {2},
$$

where the regularizer is the squared $\ell _ { 1 , 2 }$ norm on W.

Another way to select common features for MTL is to use sparse priors to design probabilistic or Bayesian models. For $\ell _ { p , \cdot }$ <sub>1</sub>-regularized multi-task feature selection, Zhang et al. [37] propose a probabilistic interpretation where the $\ell _ { p , 1 }$ regularizer corresponds to a generalized normal prior: $w _ { j i } \sim \mathcal { G N } ( \cdot | 0 , \rho _ { j } , p )$ where · denotes a (random) variable when we do not want to introduce it explicitly. Based on this interpretation, Zhang et al. [37] further propose a probabilistic framework for multi-task feature selection, in which task relations and outlier tasks can be identified, based on the matrix-variate generalized normal prior.

In [38], a generalized horseshoe prior is proposed to do feature selection for MTL as:

$$
\mathbb {P} (\mathbf {w} ^ {i}) = \int \prod_ {j = 1} ^ {d} \mathcal {N} (w _ {j i} | 0, \frac {u _ {j i}}{v _ {j i}}) \mathcal {N} (\mathbf {u} ^ {i} | 0, \rho^ {2} \mathbf {C}) \mathcal {N} (\mathbf {v} ^ {i} | 0, \gamma^ {2} \mathbf {C}) \mathrm{d} \mathbf {u} ^ {i} \mathrm{d} \mathbf {v} ^ {i},
$$

where $\mathcal { N } ( \cdot | \mathbf { m } , \pmb { \sigma } )$ denotes a univariate or multivariate normal distribution with m as the mean and σ as the variance or covariance matrix, $u _ { j i }$ and $v _ { j i }$ are the jth entries in $\mathrm { \mathbf { u } } ^ { i }$ and $\mathbf { v } ^ { i }$ , respectively, and $\rho , \gamma$ are hyperparameters. Here C shared by all the tasks denotes the feature correlation matrix to be learned from data and it encodes an assumption that different tasks share identical feature correlations. When C becomes an identity matrix which means that features are independent, this prior degenerates to the horseshoe prior.

Hernandez-Lobato et al. [39] propose a probabilistic model´ based on the horseshoe prior as

$$
\begin{array}{l} \mathbb {P} (w _ {j i}) = \left[ \pi (w _ {j i}) ^ {\eta_ {j i}} \delta_ {0} ^ {1 - \eta_ {j i}} \right] ^ {z _ {j}} \left[ \pi (w _ {j i}) ^ {\tau_ {j i}} \delta_ {0} ^ {1 - \tau_ {j i}} \right] ^ {\omega_ {i} (1 - z _ {j})} \\ \left[ \pi (w _ {j i}) ^ {\gamma_ {j}} \delta_ {0} ^ {1 - \gamma_ {j}} \right] ^ {(1 - \omega_ {i}) (1 - z _ {j})}, \tag {8} \\ \end{array}
$$

where $\delta _ { 0 }$ is the probability mass function at zero and $\pi ( \cdot )$ denotes the density function of non-zero coefficients. In $\mathrm { E q . ~ } ( 8 ) , \ z _ { j }$ indicates whether feature $j$ is an outlier $( z _ { j } = 1 )$ or not $( z _ { j } = 0 )$ and $\omega _ { i }$ indicates whether task $\mathcal { T } _ { i }$ is an outlier $( \omega _ { i } = 1 )$ or not $( \omega _ { i } = 0 )$ Moreover, $\eta _ { j i }$ and $\tau _ { j i }$ indicate whether feature $j$ is relevant for the prediction in $\mathcal { T } _ { i } \overset { \cdot } { ( } \eta _ { j i } , \tau _ { j i } = 1 )$ or not $( \eta _ { j i } , \tau _ { j i } = 0 )$ , and $\gamma _ { j }$ indicates whether a non-outlier feature $j$ is relevant $( \gamma _ { j } = 1 )$ for the prediction or not $( \gamma _ { j } = 0 )$ in all non-outlier tasks. Based on the above definitions, the three terms in the right-hand side of Eq. (8) specify probability density functions of $w _ { j i }$ based on different situations of features and tasks. So this model can also handle outlier tasks but in a different way from [37].

## 2.1.3 Comparison between Two Sub-categories

The two sub-categories have different characteristics where the feature transformation approach learns a transformation of the original features as the new representation but the feature selection approach selects a subset of the original features as the new representation for all the tasks. Based on the characteristics of those two approaches, the feature selection approach can be viewed as a special case of the feature transformation approach when the transformation matrix is a diagonal 0/1 matrix where the diagonal entries with value 1 correspond to the selected features. By selecting a subset of the original features as the new representation, the feature selection approach has a better interpretability.

## 2.2 Low-Rank Approach

The relatedness among multiple tasks can imply the low-rank of W, leading to the low-rank approach. For example, if the ith, jth and kth tasks are related in that the model parameter $\mathbf { w } ^ { i }$ of the ith task is a linear combination of those of the other two tasks, then it is easy to show that the rank of W is at most m − 1 and hence of low rank. From this perspective, the more the relatedness is, the lower the rank of W is.

Ando and Zhang [40] assume that model parameters of different tasks share a low-rank subspace in part and specifically, $\mathbf { w } ^ { i }$ takes the following form as

$$
\mathbf {w} ^ {i} = \mathbf {u} ^ {i} + \boldsymbol {\Theta} ^ {T} \mathbf {v} ^ {i}. \tag {9}
$$

Here $\Theta \in \mathbb { R } ^ { h \times d }$ is the shared low-rank subspace by multiple tasks where $h < d .$ Then we can write in a matrix form as $\mathbf { W } =$ $\mathbf { U } + \mathbf { \Theta } \Theta ^ { T } \mathbf { V }$ . Based on the form of W, the objective function proposed in [40] is formulated as

$$
\min _ {\mathbf {U}, \mathbf {V}, \boldsymbol {\Theta}, \mathbf {b}} L (\mathbf {U} + \boldsymbol {\Theta} ^ {T} \mathbf {V}, \mathbf {b}) + \lambda \| \mathbf {U} \| _ {F} ^ {2} \text {s.t.} \boldsymbol {\Theta} \boldsymbol {\Theta} ^ {T} = \mathbf {I}, \tag {10}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. The orthonormal constraint on Θ in problem (10) makes the subspace non-redundant. When λ is large enough, the optimal U can become a zero matrix and hence problem (10) is very similar to problem (1) except that there is no regularization on V in problem (10) and that Θ has a smaller number of rows than columns. Chen et al. [41] generalize problem (10) as

$$
\min _ {\mathbf {U}, \mathbf {V}, \boldsymbol {\Theta}, \mathbf {b}} \quad L (\mathbf {W}, \mathbf {b}) + \lambda_ {1} \| \mathbf {U} \| _ {F} ^ {2} + \lambda_ {2} \| \mathbf {W} \| _ {F} ^ {2}
$$

$$
\mathrm{s.t.} \quad \mathbf {W} = \mathbf {U} + \boldsymbol {\Theta} ^ {T} \mathbf {V}, \boldsymbol {\Theta} \boldsymbol {\Theta} ^ {T} = \mathbf {I}. \tag {11}
$$

When setting $\lambda _ { 2 }$ to be 0, problem (11) reduces to problem (10). Even though problem (11) is non-convex, with some convex relaxation technique, it can be relaxed to the following convex problem as

$$
\min _ {\mathbf {W}, \mathbf {b}, \mathbf {M}} L (\mathbf {W}, \mathbf {b}) + \lambda \operatorname{tr} \left(\mathbf {W} ^ {T} \left(\mathbf {M} + \eta \mathbf {I}\right) ^ {- 1} \mathbf {W}\right) \text {s.t.} \begin{array}{l} \operatorname{tr} (\mathbf {M}) = h \\ \mathbf {0} \preceq \mathbf {M} \preceq \mathbf {I} \end{array} , \tag {12}
$$

where $\eta ~ = ~ \lambda _ { 2 } / \lambda _ { 1 }$ and $\lambda = \lambda _ { 1 } \eta ( \eta + 1 )$ . One advantage of problem (12) over problem (11) is that the global optimum of the convex problem (12) is much easier to be obtained than that of the non-convex problem (11). Compared with the alternative objective function (2) in the MTFL method, problem (12) has a similar formulation where M models the feature covariance for all the tasks. Problem (10) is extended in [42] to a general case where different $\mathbf { w } ^ { i } \mathbf { \bar { s } }$ lie in a manifold instead of a subspace. Moreover, in [43], a latent variable model is proposed for $\bar { \bf W }$ with the same decomposition as Eq. (9) and it can provide a framework for MTL by modeling more cases than problem (10) such as task clustering, sharing sparse representation, duplicate tasks and evolving tasks.

It is well known that using the trace norm as a regularizer can make a matrix have low rank and hence this regularization is suitable for MTL. Specifically, an objective function with the trace norm regularization is proposed in [44] as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \| \mathbf {W} \| _ {S (1)}, \tag {13}
$$

where $\mu _ { i } ( \mathbf { W } )$ denotes the ith smallest singular value of W and $\begin{array} { r } { \| \mathbf { W } \| _ { S ( 1 ) } = \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , d ) } \mu _ { i } ( \mathbf { W } ) } \end{array}$ denotes the trace norm of matrix W. Based on the trace norm, Han and Zhang [45] propose a capped trace regularizer with the objective function formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \sum_ {i = 1} ^ {\min (m, d)} \min (\mu_ {i} (\mathbf {W}), \theta). \tag {14}
$$

With the use of the threshold θ, the capped trace regularizer only penalizes small singular values of W, which is related to the determination of the rank of W. When θ is large enough, the capped trace regularizer will become the trace norm and hence problem (14) will reduce to problem (13). Moreover, a spectral k-support norm is proposed in [46] as an improvement over the trace norm regularization.

The trace norm regularization has been extended to regularize model parameters in deep multi-task models. Specifically, the weights in the last several fully connected layers of deep multi-task neural networks can be viewed as the parameters of learners for all the tasks. In this view, the weights connecting two consecutive layers for one task can be organized in a matrix and hence the weights of all the tasks can form a tensor. Based on such tensor representations, several tensor trace norms, which are based on the trace norm, are used in [47] as regularizers to identify the low-rank structure of the parameter tensor.

## 2.3 Task Clustering Approach

The task clustering approach assumes that different tasks form several clusters, each of which consists of similar tasks. As indicated by its name, this approach has a close connection with clustering algorithms and it can be viewed as an extension of clustering algorithms to the task level while the conventional clustering algorithms are on the data level.

Thrun and Sullivan [48] propose the first task clustering algorithm by using a weighted nearest neighbor classifier for each task, where the initial weights to define the weighted Euclidean distance are learned by minimizing pairwise within-class distances and maximizing pairwise between-class distances simultaneously within each task. Then a task transfer matrix A is defined with its $( i , j ) \mathrm { t h }$ entry $a _ { i j }$ recording the generalization accuracy obtained for task $\mathcal { T } _ { i }$ by using task $\mathcal { T } _ { j } ^ { \bullet }$ distance metric. Based on A, m tasks can be grouped into r clusters $\{ \mathcal { C } _ { i } \} _ { i = 1 } ^ { r }$ by maximizing $\begin{array} { r } { \sum _ { t = 1 } ^ { r } \frac { 1 } { | \mathcal { C } _ { t } | } \sum _ { i , j \in \mathcal { C } _ { t } } \bar { a } _ { i j } } \end{array}$ , where | · | denotes the cardinality of a set. After obtaining the cluster structure among all the tasks, the training data of tasks in a cluster will be pooled together to learn the final weighted nearest neighbor classifier. This approach has been extended to an iterative learning process [49] in a similar way to k-means clustering.

Bakker and Heskes [50] propose a multi-task Bayesian neural network model with the network structure similar to Fig. 2 where input-to-hidden weights are shared by all the tasks but hiddento-output weights are task-specific. By defining $\mathbf { w } ^ { i }$ as the vector of hidden-to-output weights for task $\mathcal { T } _ { i } ,$ , the multi-task Bayesian neural network assigns a mixture of Gaussian prior to it: $\mathbf { w } ^ { i } \sim$ $\begin{array} { r } { \sum _ { j = 1 } ^ { r } \pi _ { j } \mathcal { N } ( \cdot | \mathbf { m } _ { j } , \bar { \mathbf { \Theta } } _ { j } ) } \end{array}$ , where $\pi _ { j } , \mathbf { m } _ { j }$ and $\Sigma _ { j }$ specify the prior, the mean and the covariance in the jth cluster. For tasks in a cluster, they will share a Gaussian distribution. When r equals 1, this model degenerates to a case where model parameters of different tasks share a prior, which is similar to several Bayesian MTL models such as [51], [52], [53] that are based on Gaussian processes and t processes.

Xue et al. [54] deploy the Dirichlet process to do clustering on task level. Specifically, it defines the prior on $\mathbf { w } ^ { i }$ as

$$
\mathbf {w} ^ {i} \sim G, G \sim \mathcal {D P} (\alpha , G _ {0}) \forall i \in [ m ],
$$

where $\mathcal { D P } ( \alpha , G _ { 0 } )$ denotes a Dirichlet process with α as a positive scaling parameter and $G _ { 0 }$ a base distribution. To see the clustering effect, by integrating out $G ,$ the conditional distribution of $\bar { \mathbf { w } } ^ { i }$ , given model parameters of other tasks ${ \bf W } _ { - i } =$ $\{ \cdots , \mathbf { w } ^ { i - 1 } , \mathbf { w } ^ { i + 1 } , \cdots \}$ , is

$$
\mathbb {P} (\mathbf {w} ^ {i} | \mathbf {W} _ {- i}, \alpha , G _ {0}) = \frac {\alpha}{m - 1 + \alpha} G _ {0} + \frac {1}{m - 1 + \alpha} \sum_ {j = 1, j \neq i} ^ {m} \delta_ {\mathbf {w} ^ {j}},
$$

where $\delta _ { \mathbf { w } ^ { j } }$ denotes the distribution concentrated at a single point $\mathbf { w } ^ { j }$ . So $\mathbf { w } ^ { i }$ can be equal to either $\mathbf { w } ^ { j } \mathbf { \Xi } ( j \neq i )$ with probability ${ \frac { 1 } { m - 1 + \alpha } } ,$ , which corresponds to the case that those two tasks lie in the same cluster, or a new sample from $G _ { 0 }$ with probability $\frac { \alpha } { m - 1 + \alpha }$ , which is the case that task $\mathcal { T } _ { i }$ forms a new task cluster. When α is large, the chance to form a new task cluster is large and so α affects the number of task clusters. This model is extended in [55], [56] to a case where different tasks in a task cluster share useful features via a matrix stick-breaking process and a beta-Bernoulli hierarchical prior, respectively, and in [57] where each task is a compressive sensing task. Moreover, a nested Dirichlet process is proposed in [58], [59] to use Dirichlet processes to learn both task clusters and the state structure of an infinite hidden Markov model, which handles sequential data in each task. In [60], $\mathbf { w } ^ { i }$ is decomposed as $\mathbf { w } ^ { i } = \mathbf { u } ^ { i } + \mathbf { \Theta } \Theta _ { i } ^ { T } \mathbf { v } ^ { i }$ similar to Eq. (9), where $\mathrm { \mathbf { u } } ^ { i }$ and $\Theta _ { i }$ are sampled according to a Dirichlet process.

Different from [50], [54], Jacob et al. [61] aim to learn task clusters under the regularization framework by considering three orthogonal aspects, including a global penalty to measure on average how large the parameters, a measure of between-cluster variance to quantify the distance among different clusters and a measure of within-cluster variance to quantify the compactness of task clusters. By combining these three aspects and adopting some convex relaxation technique, a convex objective function is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}, \boldsymbol {\Sigma}} L (\mathbf {W}, \mathbf {b}) + \lambda \mathrm{tr} (\mathbf {W 1 1} ^ {T} \mathbf {W} ^ {T}) + \mathrm{tr} (\tilde {\mathbf {W}} \boldsymbol {\Sigma} ^ {- 1} \tilde {\mathbf {W}} ^ {T})
$$

$$
\text {s.t.} \tilde {\mathbf {W}} = \mathbf {W} \boldsymbol {\Pi}, \alpha \mathbf {I} \preceq \boldsymbol {\Sigma} \preceq \beta \mathbf {I}, \operatorname{tr} (\boldsymbol {\Sigma}) = \gamma , \tag {15}
$$

where Π denotes the m×m centering matrix, 1 denotes a column vector of all ones with its size depending on the context, and $\alpha , \beta , \gamma$ are hyperparameters.

Kang et al. [62] extend the MTFL method [10] to the case with multiple task clusters and aim to minimize the squared trace norm in each cluster. A diagonal matrix, $\mathbf { Q } _ { i } \in \mathbb { R } ^ { m \times \dot { m } }$ , is defined as a cluster indicator matrix for the ith cluster. The jth diagonal entry of $\mathbf { Q } _ { i }$ is equal to 1 if task $\tau _ { j }$ lies in the ith cluster and otherwise 0. Since each task can belong to only one cluster, it is easy to see that $\textstyle \sum _ { i = 1 } ^ { r } \mathbf { Q } _ { i } = \mathbf { I }$ . Based on these considerations, the objective function is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}, \{\mathbf {Q} _ {i} \}} L (\mathbf {W}, \mathbf {b}) + \lambda \sum_ {i = 1} ^ {r} \| \mathbf {W Q} _ {i} \| _ {S (1)} ^ {2} \text {s.t.} \begin{array}{c} \mathbf {Q} _ {i} \in \{0, 1 \} ^ {m \times m} \\ \sum_ {i = 1} ^ {r} \mathbf {Q} _ {i} = \mathbf {I} \end{array} .
$$

When $r$ equals 1, this method reduces to the MTFL method.

Han and Zhang [63] devise a structurally sparse regularizer to cluster tasks with the objective function as

$$
\min _ {\mathbf {W}, \mathbf {b}} L (\mathbf {W}, \mathbf {b}) + \lambda \sum_ {j > i} \| \mathbf {w} ^ {i} - \mathbf {w} ^ {j} \| _ {2}. \tag {16}
$$

Problem (16) is a special case of the method proposed in [63] with only one level of task clusters. The regularizer on W enforces any pair of columns in W to have a chance to be identical and after solving problem (16), the cluster structure can be discovered by comparing columns in W. One advantage of this structurally sparse regularizer is that the convex problem (16) can automatically determine the number of task clusters.

Barzilai and Crammer [64] propose a task clustering method by defining W as $\mathbf { W } \ = \ \bar { \mathbf { F } } \bar { \mathbf { G } }$ where $\textbf { F } \in \ \mathbb { R } ^ { d \times r }$ and ${ \textbf { G } } \in$ $\{ \mathbf { \bar { 0 } } , 1 \} ^ { r \times m }$ . With an assumption that each task belongs to only one cluster, the objective function is formulated as

$$
\min _ {\mathbf {F}, \mathbf {G}, \mathbf {b}} L (\mathbf {F} \mathbf {G}, \mathbf {b}) + \lambda \| \mathbf {F} \| _ {F} ^ {2} \text {s.t.} \begin{array}{c} \mathbf {G} \in \{0, 1 \} ^ {r \times m} \\ \| \mathbf {g} ^ {i} \| _ {2} = 1 \forall i \in [ m ], \end{array} \tag {17}
$$

where $\mathbf { g } ^ { i }$ denotes the ith column of G. When using the hinge loss or logistic loss, this non-convex problem can be relaxed to a min-max problem, which has a global optimum, by utilizing the dual problem with respect to W and b and discarding some non-convex constraints.

Zhou and Zhao [65] aim to cluster tasks by identifying representative tasks which are a subset of the given m tasks. If task T is selected by task $\tau _ { j }$ as a representative task, then it is expected that model parameters for $\tau _ { j }$ are similar to those of $\mathcal { T } _ { i } . ~ z _ { i j }$ is defined as the probability that task $\tau _ { j }$ selects task $\mathcal { T } _ { i }$ as its representative task. Then based on a matrix $\mathbf { Z }$ whose $( i , j ) \mathrm { t h }$ entry is $z _ { i j }$ , the objective function is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}, \mathbf {Z}} L (\mathbf {W}, \mathbf {b}) + \lambda_ {1} \| \mathbf {W} \| _ {F} ^ {2} + \lambda_ {2} \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {m} z _ {i j} \| \mathbf {w} ^ {i} - \mathbf {w} ^ {j} \| _ {2} ^ {2} + \lambda_ {3} \| \mathbf {Z} \| _ {2, 1}
$$

$$
\text {s.t.} \mathbf {Z} \geq \mathbf {0}, \mathbf {Z} ^ {T} \mathbf {1} = \mathbf {1}. \tag {18}
$$

The third term in the objective function of problem (18) enforces the closeness of each pair of tasks based on Z and the last term employs the $\ell _ { 2 , 1 }$ norm to enforce the row sparsity of Z which implies that the number of representative tasks is limited. The constraints in problem (18) guarantee that entries in Z define valid probabilities. Problem (18) is related to problem (16) since the regularizer in problem (16) can be reformulated as $2 \textstyle \sum _ { j > i } \| \mathbf { w } ^ { i } - $ $\begin{array} { r } { { \bf w } ^ { j } \| _ { 2 } = \operatorname* { m i n } _ { \hat { \bf Z } \geq { \bf 0 } } \sum _ { j > i } \left( \hat { z } _ { i j } \| { \bf w } ^ { i } - { \bf w } ^ { j } \| _ { 2 } ^ { 2 } + \frac { 1 } { \hat { z } _ { i j } } \right) } \end{array}$ , where both the regularizer and constraint on $\hat { \mathbf { Z } }$ are different from those on Z in problem (18).

Previous studies assume that each task can belong to only one task cluster and this assumption seems too restrictive. In [66], a GO-MTL method relaxes this assumption by allowing a task to belong to more than one cluster and defines a decomposition of W similar to problem (17) as W = LS where $\mathbf { L } \doteq \mathbb { R } ^ { d \times r }$ denotes the latent basis with $r \ < \ m$ and $\mathbf { S } \in \mathbb { R } ^ { r \times m }$ contains linear combination coefficients for all the tasks. S is assumed to be sparse since each task is generated from only a few columns in L or equivalently belongs to a small number of clusters. The objective function is formulated as

$$
\min _ {\mathbf {L}, \mathbf {S}, \mathbf {b}} L (\mathbf {L} \mathbf {S}, \mathbf {b}) + \lambda_ {1} \| \mathbf {S} \| _ {1} + \lambda_ {2} \| \mathbf {L} \| _ {F} ^ {2}. \tag {19}
$$

Compared with the objective function of multi-task sparse coding, i.e., problem (3), we can see that when the regularization parameters take appropriate values, these two problems are almost equivalent except that in multi-task sparse coding, the dictionary U is overcomplete, while here the number of columns in S is smaller than that of its rows. This method has been extended in [67] to decompose the parameter tensor in the fully connected layers of deep neural networks.

Among the aforementioned methods, the method in [48] first identifies the cluster structure and then learns the model parameters of all the tasks separately, which is not preferred since the cluster structure learned may be suboptimal for the model parameters, hence follow-up works learn model parameters and the cluster structure together. An important problem in clustering is to determine the number of clusters and this is also important for this approach. Out of the above methods, only methods in [54], [63] can automatically determine the number of task clusters, where the method in [54] depends on the capacity of the Dirichlet process while the method in [63] relies on the use of a structurally sparse regularizer. Among all those models, some belong to Bayesian learning, i.e., [50], [54], while the rest models are regularized models. Among those regularized methods, only the objective function proposed in [63] is convex while others are originally non-convex.

The task clustering approach is related to the low-rank approach. To see that, suppose that there are r task clusters $( r < m )$ and all the tasks in a cluster share the same model parameters, making the parameter matrix W low-rank with the rank at most $r .$ From the perspective of modeling, by setting $\mathbf { u } ^ { i }$ to be a zero vector in Eq. (9), we can see that the decomposition of W in [40] becomes similar to those in [64], [66], which in some sense shows the relation between those two approaches. Moreover, the equivalence between problems (12) and (15), two typical methods in the low-rank and task clustering approaches, has been proved in [68]. The task clustering approach can visualize the learned cluster structure, which is an advantage over the low-rank approach.

## 2.4 Task Relation Learning Approach

In MTL, tasks are related and the task relatedness can be quantitated via task similarity, task correlation, task covariance and so on. Here we use task relations to include all the quantitative relatedness.

In earlier studies on MTL, task relations are assumed to be known as a priori information. In [69], [70], each task is assumed to be similar to any other task and so model parameters of each task will be enforced to approach the average model parameters of all the tasks. In [71], [72], task similarities for each pair of tasks are given and these studies utilize the task similarities to design regularizers to guide the learning of multiple tasks in a principle that the more similar two tasks are, the closer the corresponding model parameters are expected to be. A similar formulation to [71] is proposed in [73] to estimate the mean of multiple distributions by learning pairwise task relations and another similar formulation is proposed in [74] for log-density gradient estimation. Given a tree structure describing relations among tasks in [75], model parameters of a task corresponding to a node in the tree are enforced to be similar to those of its parent node.

However, in most applications, task relations are not available. In this case, learning task relations from data automatically is a good option. Bonilla et al. [76] propose a multi-task Gaussian process (MTGP) by defining a prior on $f _ { j } ^ { i }$ , the functional value for $\mathbf { x } _ { j } ^ { i } ,$ , as $\mathbf { f } \ \sim \ \mathcal { N } ( \cdot | \mathbf { 0 } , \pmb { \Sigma } )$ , where $\mathbf { f } ~ = ~ ^ { \prime } ( f _ { 1 } ^ { 1 } , \ldots , f _ { n _ { m } } ^ { m } ) ^ { T }$ contains the functional values for all the training data. Σ, the covariance matrix, defines the covariance between $f _ { j } ^ { i }$ and $f _ { q } ^ { p }$ as $\sigma ( f _ { j } ^ { i } , f _ { q } ^ { p } ) \ = \ \omega _ { i p } k ( \mathbf { x } _ { j } ^ { i } , \mathbf { x } _ { q } ^ { p } )$ , where $k ( \cdot , \cdot )$ denotes a kernel function and $\omega _ { i p }$ describes the covariance between tasks $\mathcal { T } _ { i }$ and $\mathcal { T } _ { p } .$ . In order to keep Σ positive definite, a matrix Ω containing $\omega _ { i p }$ as its $( i , p )$ )th entry is also required to be positive definite, which makes Ω the task covariance to describe the similarities between tasks. Then based on the Gaussian likelihood for labels given f, the analytically marginal likelihood by integrating out f can be used to learn Ω from data. In [77], the learning curve and generalization bound of the MTGP are studied. Since Ω in MTGP has a point estimation which may lead to the overfitting, based on a proposed weight-space view of MTGP, Zhang and Yeung [78] propose a multi-task generalized t process by placing an inverse-Wishart prior on Ω as $\Omega \sim \mathcal { T W } ( \cdot | \nu , \Psi )$ , where ν denotes the degree of freedom and Ψ is the base covariance for generating Ω. Since Ψ models the covariance between pairs of tasks, it can be determined based on the maximum mean discrepancy (MMD).

Different from [76], [78] which are Bayesian models, Zhang and Yeung [79], [80] propose a regularized multi-task model called multi-task relationship learning (MTRL) by placing a matrixvariate normal prior on W as

$$
\mathbf {W} \sim \mathcal {M N} (\cdot | \mathbf {0}, \mathbf {I}, \boldsymbol {\Omega}), \tag {20}
$$

where $\mathcal { M N } ( \cdot | \mathbf { M } , \mathbf { A } , \mathbf { B } )$ denotes a matrix-variate normal distribution with M as the mean, A the row covariance, and B the column covariance. Based on this prior as well as some likelihood function, the objective function for a modified maximum a posterior solution is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}, \boldsymbol {\Omega}} L (\mathbf {W}, \mathbf {b}) + \lambda_ {1} \| \mathbf {W} \| _ {F} ^ {2} + \lambda_ {2} \mathrm{tr} (\mathbf {W} \boldsymbol {\Omega} ^ {- 1} \mathbf {W} ^ {T})
$$

$$
\text {s.t.} \boldsymbol {\Omega} \succ \mathbf {0}, \operatorname{tr} (\boldsymbol {\Omega}) \leq 1, \tag {21}
$$

where the second term in the objective function is to penalize the complexity of W, the last term is due to the matrix-variate normal prior, and the constraints control the complexity of the positive definite covariance matrix Ω. It has been proved in [79], [80] that problem (21) is jointly convex with respect to W, b and Ω. Problem (21) has been extended to multi-task boosting [81] and multi-label learning [82] by learning label correlations. Problem (21) can also been interpreted from the perspective of reproducing kernel Hilbert spaces for vector-valued functions [83], [84], [85], [86]. Moreover, Problem (21) is extended to learn sparse task relations in [87] via the $\ell _ { 1 }$ regularization on Ω when the number of tasks is large. A model similar to problem (21) is proposed in [88] via a matrix-variate normal prior on W: $\mathbf { W } \sim \mathcal { M N } ( \cdot | \mathbf { 0 } , \Omega _ { 1 } , \Omega _ { 2 } )$ , where $\Omega _ { 1 } ^ { - 1 }$ and $\Omega _ { 2 } ^ { - 1 }$ are assumed to be sparse. The MTRL model is extended in [89] to use the symmetric matrix-variate generalized hyperbolic distribution to learn block sparse structure in W and in [90] to use the matrix generalized inverse Gaussian prior to learn low-rank $\pmb { \Omega } _ { 1 }$ and $\mathbf { \bar { \Omega } }$ . Moreover, the MTRL model is generalized to the multitask feature selection problem [37] by learning task relations via the matrix-variate generalized normal distribution. Since the prior defined in Eq. (20) implies that $\mathbf { W } ^ { T } \mathbf { W }$ follows a Wishart distribution as $\mathcal { W } ( \cdot | \mathbf { 0 } , \pmb { \Omega } )$ , Zhang and Yeung [91] generalize it as

$$
(\mathbf {W} ^ {T} \mathbf {W}) ^ {t} \sim \mathcal {W} (\cdot | \mathbf {0}, \boldsymbol {\Omega}), \tag {22}
$$

where t is a positive integer to model high-order task relationships. Eq. (22) can induce a new prior, which is a generalization of the matrix-variate normal distribution, on W and based on this new prior, a regularized method is devised to learn high-order task relations in [91]. The MTRL model has been extended to multioutput regression [90], [92], [93], [94] by modeling the structure contained in noises via some matrix-variate priors. For deep neural networks, the MTRL method has been extended in [95] by placing a tensor-variate normal distribution as a prior on the parameter tensor in the fully connected layers.

Different from the aforementioned methods which investigate the use of global learning models in MTL, Zhang [96] aims to learn the task relations in local learning methods such as the k-nearest-neighbor (kNN) classifier by defining the learning function as a weighted voting of neighbors:

$$
f (\mathbf {x} _ {j} ^ {i}) = \sum_ {(p, q) \in N _ {k} (i, j)} \sigma_ {i p} s (\mathbf {x} _ {j} ^ {i}, \mathbf {x} _ {q} ^ {p}) y _ {q} ^ {p}, \tag {23}
$$

where $N _ { k } ( i , j )$ denotes the set of task indices and instance indices for the k nearest neighbors of $\mathbf { x } _ { j } ^ { i } , \mathrm { i . e . , } ( p , q ) \in N _ { k } ( i , j )$ meaning that $\mathbf { x } _ { q } ^ { p }$ is one of the k nearest neighbors of $\mathbf { x } _ { j } ^ { i } , ~ s ( \mathbf { x } _ { j } ^ { i } , \mathbf { x } _ { q } ^ { p } )$ defines the similarity between $\mathbf { x } _ { i } ^ { i }$ and $\mathbf { x } _ { a } ^ { p } .$ , and $\sigma _ { i p }$ represents the contribution of task $\dot { \mathcal { T } } _ { p }$ to $\mathcal { T } _ { i }$ when $\mathcal { T } _ { p }$ has some data points to be neighbors of a data point in T. $\sigma _ { i p }$ can be viewed as the similarity from $\mathcal { T } _ { p } \mathrm { ~ t o ~ } \mathcal { T } _ { i }$ . When $\sigma _ { i p } = 1$ for all i and p, Eq. (23) reduces to the decision function of the kNN classifier for all the tasks. Then the objective function to learn $\Sigma ,$ , which is a m × m matrix with $\sigma _ { i p }$ as its $( i , p ) \mathrm { \bf t }$ h entry, can be formulated as

$$
\min _ {\boldsymbol {\Sigma}} \quad \sum_ {i = 1} ^ {m} \frac {1}{n _ {i}} \sum_ {j = 1} ^ {n _ {i}} l (y _ {j} ^ {i}, f (\mathbf {x} _ {j} ^ {i})) + \frac {\lambda_ {1}}{4} \| \boldsymbol {\Sigma} - \boldsymbol {\Sigma} ^ {T} \| _ {F} ^ {2} + \frac {\lambda_ {2}}{2} \| \boldsymbol {\Sigma} \| _ {F} ^ {2}
$$

$$
\text {s.t.} \quad \sigma_ {i i} \geq 0 \forall i \in [ m ], - \sigma_ {i i} \leq \sigma_ {i j} \leq \sigma_ {i i} \forall i \neq j. \tag {24}
$$

The first regularizer in problem (24) enforces Σ to be nearly symmetric and the second one is to penalize the complexity of $\dot { \Sigma } .$ The constraints in problem (24) guarantee that the similarity from one task to itself is positive and also the largest. Similarly, a multi-task kernel regression is proposed in [96] for regression tasks.

While the aforementioned methods whose task relations are symmetric except [96], Lee et al. [97] focus on learning asymmetric task relations. Since different tasks are assumed to be related, $\mathbf { w } _ { i }$ can lie in the space spanned by W, i.e., $\begin{array} { r } { \mathbf { w } _ { i } \approx \mathbf { W } \mathbf { a } _ { i } , } \end{array}$ , and hence we have W ≈ WA. Here matrix A can be viewed as asymmetric task relations between pairs of tasks. By assuming that A is sparse, the objective function is formulated as

$$
\min _ {\mathbf {W}, \mathbf {b}, \mathbf {A}} \sum_ {i = 1} ^ {m} (1 + \lambda_ {1} \| \hat {\mathbf {a}} _ {i} \| _ {1}) \sum_ {j = 1} ^ {n _ {i}} l (y _ {j} ^ {i}, (\mathbf {w} ^ {i}) ^ {T} \mathbf {x} _ {j} ^ {i} + b _ {i}) + \lambda_ {2} \| \mathbf {W} - \mathbf {W A} \| _ {F} ^ {2}
$$

$$
\text {s.t.} a _ {i j} \geq 0 \forall i, j \in [ m ], \tag {25}
$$

where $\hat { \mathbf { a } } _ { i }$ denotes the ith row of A by deleting $a _ { i i }$ . The term before the training loss of each task, $\mathrm { i . e . , 1 } + \lambda _ { 1 } \| \hat { { \bf a } } _ { i } \| _ { 1 }$ , not only enforces A to be sparse but also allows asymmetric information sharing from easier tasks to difficult ones. The regularizer in problem (25) can make W approach WA with the closeness depending on $\lambda _ { 2 }$ . To see the connection between problems (25) and (21), we rewrite the regularizer in problem (25) as $\| \mathbf { W } - \mathbf { W } \mathbf { A } \| _ { F } ^ { 2 } = \operatorname { t r } \left( \mathbf { W } ( \mathbf { I } - \mathbf { A } ) ( \mathbf { \bar { I } } - \mathbf { A } ) ^ { T } \mathbf { W } ^ { \hat { T } } \right)$ . Based on this reformulation, the regularizer in problem (25) is a special case of that in problem (21) by assuming $\Omega ^ { - 1 } = ( \mathbf { I } - \hat { \mathbf { A } } ) ( \mathbf { I } - \mathbf { A } ) ^ { T }$ Though A is asymmetric, from the perspective of the regularizer, the task relations here are symmetric and act as the task precision matrix with a restrictive form.

## 2.5 Decomposition Approach

The decomposition approach assumes that the parameter matrix W can be decomposed into two or more component matrices $\{ \mathbf { W } _ { k } \} _ { k = 1 } ^ { h }$ where $\bar { h } \geq 2 , \mathrm { i . e . , } \mathbf { W } = \sum _ { k = 1 } ^ { h } \mathbf { W } _ { k } ^ { }$ . The objective functions of most methods in this approach can be unified as

$$
\min _ {\{\mathbf {W} _ {i} \} \in \mathcal {C} _ {W}, \mathbf {b}} L \left(\sum_ {k = 1} ^ {h} \mathbf {W} _ {k}, \mathbf {b}\right) + \sum_ {k = 1} ^ {h} g _ {k} (\mathbf {W} _ {k}), \tag {26}
$$

where the regularizer is decomposable with respect to $\mathbf { W } _ { k } \mathbf { \dot { \Omega } } \mathbf { s }$ and $\mathcal { C } _ { W }$ denotes a set of constraints for component matrices. To help understand problem (26), we introduce several instantiations as follows.

In [98] where h equals 2 and ${ \mathcal { C } } _ { W } = \emptyset$ is an empty set, $g _ { 1 } ( \cdot )$ and $g _ { 2 } ( \cdot )$ are defined as

$$
g _ {1} (\mathbf {W} _ {1}) = \lambda_ {1} \| \mathbf {W} _ {1} \| _ {\infty , 1}, g _ {2} (\mathbf {W} _ {2}) = \lambda_ {2} \| \mathbf {W} _ {2} \| _ {1},
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are positive regularization parameters. Similar to problem (5), each row of $\mathbf { W } _ { 1 }$ is likely to be a zero row and hence $g _ { 1 } ( \mathbf { W } _ { 1 } )$ can help select important features. Due to the $\ell _ { 1 }$ norm regularization, $g _ { 2 } ( \mathbf { W } _ { 2 } )$ makes $\mathbf { W } _ { 2 }$ sparse. Because of the characteristics of two regularizers, the parameter matrix W can eliminate unimportant features for all the tasks when the corresponding rows in both $\mathbf { W } _ { 1 }$ and $\mathbf { W } _ { 2 }$ are sparse. Moreover, $\mathbf { W } _ { 2 }$ can identify features for tasks which have their own useful features that may be outliers for other tasks. Hence this model can be viewed as a ‘robust’ version of problem (5).

With two component matrices, Chen et al. [99] define

$$
g _ {2} (\mathbf {W} _ {2}) = \lambda_ {2} \| \mathbf {W} _ {2} \| _ {1}, \mathcal {C} _ {W} = \{\mathbf {W} _ {1} | \| \mathbf {W} _ {1} \| _ {S (1)} \leq \lambda_ {1} \}, \tag {27}
$$

where $g _ { 1 } ( \mathbf { W } _ { 1 } ) = 0$ . Similar to problem (13), $\mathcal { C } _ { W }$ makes $\mathbf { W } _ { 1 }$ low-rank. With a sparse regularizer $g _ { 2 } ( \mathbf { W } _ { 2 } ) , \mathbf { W } _ { 2 }$ makes the entire model matrix W more robust to outlier tasks in a way similar to the previous model. When $\lambda _ { 2 }$ is large enough, $\mathbf { W } _ { 2 }$ will become a zero matrix and then problem (27) will act similarly to problem (13).

$g _ { i } ( \cdot ) \mathbf { \dot { s } }$ in [100] where ${ \mathcal { C } } _ { W } = \emptyset$ are defined as

$$
g _ {1} (\mathbf {W} _ {1}) = \lambda_ {1} \| \mathbf {W} _ {1} \| _ {S (1)}, \quad g _ {2} (\mathbf {W} _ {2}) = \lambda_ {2} \| \mathbf {W} _ {2} ^ {T} \| _ {2, 1}. \tag {28}
$$

Different from the above two models which assume that $\mathbf { W } _ { 2 }$ is sparse, here $g _ { 2 } ( \mathbf { W } _ { 2 } )$ enforces $\mathbf { W } _ { 2 }$ to be column-sparse. For related tasks, their columns in $\mathbf { W } _ { 1 }$ are correlated via the trace norm regularization and the corresponding columns in $\mathbf { W } _ { 2 }$ are zero. For outlier tasks which are unrelated to other tasks, the corresponding columns in $\mathbf { W } _ { 2 }$ can take arbitrary values and hence model parameters in W for them have no low-rank structure even though those in $\mathbf { W } _ { 1 }$ may have.

In [101], these functions are defined as

$$
g _ {1} (\mathbf {W} _ {1}) = \lambda_ {1} \| \mathbf {W} _ {1} \| _ {2, 1}, g _ {2} (\mathbf {W} _ {2}) = \lambda_ {2} \| \mathbf {W} _ {2} ^ {T} \| _ {2, 1}, \mathcal {C} _ {W} = \emptyset . \tag {29}
$$

Similar to problem (4), $g _ { 1 } ( \mathbf { W } _ { 1 } )$ makes $\mathbf { W } _ { 1 }$ row-sparse. Here $g _ { 2 } ( \mathbf { W } _ { 2 } )$ is identical to that in [100] and it makes $\mathbf { W } _ { 2 }$ columnsparse. Hence $\mathbf { W } _ { 1 }$ helps select useful features while non-zero columns in $\mathbf { W } _ { 2 }$ capture outlier tasks.

With $h = 2 ,$ , Zhong and Kwok [102] define

$$
g _ {1} (\mathbf {W} _ {1}) = \lambda_ {1} c (\mathbf {W} _ {1}) + \lambda_ {2} \| \mathbf {W} _ {1} \| _ {F} ^ {2}, g _ {2} (\mathbf {W} _ {2}) = \lambda_ {3} \| \mathbf {W} _ {2} \| _ {F} ^ {2}, \mathcal {C} _ {W} = \emptyset ,
$$

where $\begin{array} { r } { c ( \mathbf { U } ) = \sum _ { i = 1 } ^ { d } \sum _ { k > j } | u _ { i j } - u _ { i k } | } \end{array}$ with $u _ { i j }$ as the $( i , j ) \mathrm { t h }$ entry in a matrix U. Due to the sparse nature of the $\ell _ { 1 }$ norm, $c ( \mathbf { W } _ { 1 } )$ enforces corresponding entries in different columns of $\mathbf { W } _ { 1 }$ to be identical, which is equivalent to clustering tasks in terms of individual model parameters. Both the squared Frobenius norm regularizations in $g _ { 1 } ( \mathbf { W } _ { 1 } )$ and $g _ { 2 } ( \mathbf { W } _ { 2 } )$ penalize the complexities of $\mathbf { W } _ { 1 }$ and $\mathbf { W } _ { 2 }$ . The use of $\mathbf { W } _ { 2 }$ improves the model flexibility when not all the tasks exhibit a clear cluster structure.

Different from the aforementioned methods which have only two component matrices, an arbitrary number of component matrices are considered in [103] with

$$
g _ {k} (\mathbf {W} _ {k}) = \lambda \big [ (h - k) \| \mathbf {W} _ {k} \| _ {2, 1} + (k - 1) \| \mathbf {W} _ {k} \| _ {1} \big ] / (h - 1), \tag {30}
$$

where $\mathcal { C } _ { W } ~ = ~ \varnothing$ . According to Eq. $( 3 0 ) , \mathbf { W } _ { k }$ is assumed to be both sparse and row-sparse for all $k \in [ h ]$ . Based on different regularization parameters on the regularizer of $\mathbf { W } _ { k }$ , we can see that when k increases, $\mathbf { W } _ { k }$ is more likely to be sparse than to be row-sparse. Even though each $\mathbf { W } _ { k }$ is sparse or row-sparse, the entire parameter matrix W can be non-sparse and hence this model can discover the latent sparse structure among tasks.

In the above methods, different component matrices have no direct connection. When there is a dependency among component matrices, problem (26) can model more complex structure among tasks. For example, Han and Zhang [104] define

$$
g _ {k} (\mathbf {W} _ {k}) = \lambda \sum_ {i > j} \| \mathbf {w} _ {k} ^ {i} - \mathbf {w} _ {k} ^ {j} \| _ {2} / \eta^ {k - 1} \forall k \in [ h ]
$$

$$
\mathcal {C} _ {W} = \{\{\mathbf {W} _ {k} \} | | \mathbf {w} _ {k - 1} ^ {i} - \mathbf {w} _ {k - 1} ^ {j} | \geq | \mathbf {w} _ {k} ^ {i} - \mathbf {w} _ {k} ^ {j} | \forall k \geq 2, \forall i > j \},
$$

where $ { \mathbf { w } } _ { k } ^ { i }$ denotes the ith column of $\mathbf { W } _ { k }$ . Note that the constraint set $\mathcal { C } _ { W }$ relates component matrices and the regularizer $g _ { k } ( \mathbf { W } _ { k } )$ makes each pair of $\mathbf { w } _ { k } ^ { i }$ and $\mathbf { w } _ { k } ^ { j }$ have a chance to become identical. Once this happens for some $i , j ,$ k, then based on the constraint set $\mathcal { C } _ { W } ,  { \mathbf { w } } _ { k } ^ { i } ,$ and $\mathbf { w } _ { k } ^ { j } ,$ will always have the same value for $k ^ { \prime } \geq k$ This corresponds to sharing all the ancestor nodes for two internal nodes in a tree and hence this method can learn a hierarchical structure to characterize task relations. When the constraints are removed, this method reduces to the multi-level task clustering method [63], which is a generalization of problem (16).

Another way to relate different component matrices is to use a non-decomposable regularizer as [105] did, which is slightly different from problem (26) in terms of the regularizer. Specifically, given m tasks, there are $2 ^ { m } - 1$ possible and non-empty task clusters. All the task clusters can be organized in a tree, where the root node represents a dummy node, nodes in the second level represent groups with a single task, and the parent-child relations are the ‘subset of’ relation. In total, there are $h \equiv 2 ^ { m }$ component matrices each of which corresponds to a node in the tree and hence an index a is used to denote both a level and the corresponding node in the tree. The objective function is formulated as

$$
\begin{array}{l} \min _ {\{\mathbf {W} _ {i} \}, \mathbf {b}} L \left(\sum_ {k = 1} ^ {h} \mathbf {W} _ {k}, \mathbf {b}\right) + \left(\sum_ {v \in V} \lambda_ {v} \left(\sum_ {a \in D (v)} r (\mathbf {W} _ {a}) ^ {p}\right) ^ {\frac {1}{p}}\right) ^ {2} \\ \text {s.t.} \mathbf {w} _ {a} ^ {i} = \mathbf {0} \forall i \notin t (a), \tag {31} \\ \end{array}
$$

where $p$ takes a value between 1 and $2 , D ( a )$ denotes the set of all the descendants of $a , t ( a )$ denotes the set of tasks contained in node $a , \mathbf { w } _ { a } ^ { i }$ denotes the ith column of $\mathbf { W } _ { a } ,$ and $r ( \mathbf { W } _ { a } )$ reflects relations among tasks in node a based on ${ \bf W } _ { a }$ . The regularizer in problem (31) is used to prune the subtree rooted at each node v based on the $\ell _ { p }$ norm. The constraint in problem (31) implies that for tasks not contained in a node a, the corresponding columns in ${ \bf W } _ { a }$ are zero. In [105], $r ( \mathbf { W } _ { a } )$ adopts the regularizer proposed in [69] which enforces the parameters of all the tasks to approach their average.

Different from deep MTL models which are deep in terms of layers of feature representations, the decomposition approach can be viewed as a ‘deep’ approach in terms of model parameters while most of previous approaches are just shallow ones, making this approach have more powerful capacity. Moreover, the decomposition approach can reduce to other approaches such as the feature learning, low-rank and task clustering approaches when there is only one component matrix and hence it can be considered as an improved version of those approaches.

## 2.6 Comparisons among Different Approaches

Based on the above introduction, we can see that different approaches exhibit their own characteristics. Specifically, the feature learning approach can learn common features, which are generic and invariant to all the tasks at hand and even new tasks, for all the tasks. When there exist outlier tasks which are unrelated to other tasks, the learned features can be influenced by outlier tasks significantly and they may cause the performance deterioration.

By assuming that the parameter matrix is low-rank, the lowrank approach can explicitly learn the subspace of the parameter matrix or implicitly achieve that via some convex or non-convex regularizer. This approach is powerful but it seems applicable to only linear models, making nonlinear extensions non-trivial to be devised. The task clustering approach performs clustering on the task level in terms of model parameters and it can identify task clusters each of which consists of similar tasks. A major limitation of the task clustering approach is that it can capture positive correlations among tasks in the same cluster but ignore negative correlations among tasks in different clusters. Moreover, even though some methods in this category can automatically determine the number of clusters, most of them still need a model selection method such as cross validation to determine it, which may bring additional computational costs. The task relation learning approach can learn model parameters and pairwise task relations simultaneously. The learned task relations can give us insights about the relations between tasks and hence they improve the interpretability. The decomposition approach can be viewed as extensions of other parameter-based approaches by equipping multi-level parameters and hence they can model more complex task structure, e.g., tree structure. The number of components in the decomposition approach is important to the performance and needs to be carefully determined.

## 2.7 Benchmark Datasets and Performance Comparison

In this section, we introduce some benchmark datasets for MTL and compare the performance of different MTL models on them.

Some benchmark datasets for MTL are listed as follows.

• School dataset [50]: This dataset is to estimate examination scores of 15,362 students from 139 secondary schools in London from 1985 to 1987 where each school is treated as a task. The input consists of four school-specific and three student-specific attributes.  
• SARCOS dataset<sup>2</sup>: This dataset studies a multi-output problem of learning the inverse dynamics of 7 SARCOS anthropomorphic robot arms, each of which corresponds to a task, based on 21 features, including seven joint positions, seven joint velocities and seven joint accelerations. This dataset contains 48,933 data points.  
• Computer Survey dataset [11]: This dataset is taken from a survey of 180 persons/tasks who rated the likelihood of purchasing one of 20 different personal computers, resulting in 36,000 data points in all the tasks. The features contain 13 different computer characteristics (e.g., price, CPU and RAM) while the output is an integer rating on the scale 0-10.  
• Parkinson dataset [105]: This dataset is to predict the disease symptom score of Parkinson for patients at different times using 19 bio-medical features. This dataset has 5,875 data points for 42 patients, each of whom is treated as a task.  
• Sentiment dataset<sup>3</sup>: This dataset is to classify reviews of four products/tasks, i.e., books, DVDs, electronics and kitchen appliances, from Amazon into two classes: positive and negative reviews. For each task, there are 1,000 positive and 1,000 negative reviews, respectively.  
• MHC-I dataset [61]: This databset contains binding affinities of 15,236 peptides with 35 MHC-I molecules. Each MHC-I

2. http://www.gaussianprocess.org/gpml/data/  
3. http://www.cs.jhu.edu/<sup>∼</sup>mdredze/datasets/sentiment

TABLE 1 The performance comparison of representative MTL models in the five approaches on benchmark datasets in terms of some evaluation metric. nMSE stands for ‘normalized mean squared error’, RMSE is for ‘root mean squared error’, and AUC stands for ‘Area Under Curve’. ↑ after the evaluation metric implies that the larger value the better performance and ↓ indicates the opposite case.

<table><tr><td rowspan="2">Dataset (Reference)</td><td rowspan="2">Evaluation Metric</td><td rowspan="2">STL</td><td>Feature Learning</td><td>Low-Rank</td><td>Task Clustering</td><td>Task Relation Learning</td><td>Decomposition</td></tr><tr><td>[10]</td><td>[44]</td><td>[61]/ [62]/ [66]/ [67]</td><td>[79]/ [86]/ [95]</td><td>[98]/ [99]/ [100]/ [105]/ [104]</td></tr><tr><td>School ( [104])</td><td>nMSE↓</td><td>—</td><td>0.4393</td><td>—</td><td>0.4374/-/0.6466/-</td><td>—</td><td>0.4445/-/-/0.4169</td></tr><tr><td>SARCOS ( [100])</td><td>nMSE↓</td><td>0.1821</td><td>0.1568</td><td>0.1531</td><td>—</td><td>—</td><td>0.1495/0.1456/-/-</td></tr><tr><td>Computer Survey ( [102])</td><td>RMSE↓</td><td>2.381</td><td>—</td><td>—</td><td>2.072/-/-/-</td><td>2.110/-/-</td><td>2.138/2.052/2.074/-/-</td></tr><tr><td>Parkinson ( [86])</td><td>Explained Variance↑</td><td>2.8%</td><td>—</td><td>—</td><td>2.7%/33.6%/-/-</td><td>12.0%/27.0%/-</td><td>-/-/-16.8%/-</td></tr><tr><td>Sentiment ( [79])</td><td>Classification Error↓</td><td>0.2779</td><td>0.2756</td><td>—</td><td>—</td><td>0.2324/-/-</td><td>—</td></tr><tr><td>MHC-I ( [65])</td><td>Classification Error↓</td><td>0.2010</td><td>—</td><td>—</td><td>0.1890/0.2050/-/-</td><td>0.1870/-/-</td><td>0.2030/-/0.2070/-/-</td></tr><tr><td>Landmine ( [86])</td><td>AUC↑</td><td>74.6%</td><td>—</td><td>—</td><td>75.9%/76.7%/-/-</td><td>76.1%/76.8%/-</td><td>-/-/-76.4%/-</td></tr><tr><td>Office-Caltech ( [95])</td><td>Classification Error↓</td><td>0.0920</td><td>0.0740</td><td>—</td><td>-/-/-/0.0670</td><td>0.0690/-/0.0450</td><td>-/-/0.0760/-/-</td></tr><tr><td>Office-Home ( [95])</td><td>Classification Error↓</td><td>0.3430</td><td>0.4170</td><td>—</td><td>-/-/-/0.3350</td><td>0.4070/-/0.3310</td><td>-/-/0.4140/-/-</td></tr><tr><td>ImageCLEF ( [95])</td><td>Classification Error↓</td><td>0.3640</td><td>0.3440</td><td>—</td><td>-/-/-/0.2780</td><td>0.3350/-/0.2470</td><td>-/-/0.3510/-/-</td></tr></table>

molecule is considered as a task and the goal is to predict whether a peptide binds a molecule.

• Landmine dataset [54]: This dataset consists of 9-dimensional data points, whose features are extracted from radar images, from 29 landmine fields/tasks. Each task is to classify a data point into two classes (landmine or clutter). There are 14,820 data points in total.  
• Office-Caltech dataset [106]: The dataset contains data from 10 common categories shared in the Caltech-256 dataset and the Office dataset which consists of images collected from three distinct domains/tasks: Amazon, Webcam and DSLR, making this dataset contain 4 tasks. There are 2,533 images in all the tasks.  
• Office-Home dataset<sup>4</sup>: This dataset consists of images from 4 different domains/tasks: artistic images, clip art, product images and real-world images. Each task contains images of 65 object categories collected in the office and home settings. In total, there are about 15,500 images in all the tasks.  
• ImageCLEF dataset<sup>5</sup>: This dataset contains 12 common categories shared by four tasks: Caltech-256, ImageNet ILSVRC 2012, Pascal VOC 2012 and Bing. There are about 2,400 images in all the tasks.

In the above benchmark datasets, the first four datasets consist of regression tasks while the other datasets are classification tasks, where each task in the Sentiment, MHC-I and Landmine datasets is a binary classification problem and that in the other three image datasets is a multi-class classification problem. In order to compare different MTL approaches on those benchmark datasets, we select some representative MTL methods from each of the five approaches introduced in the previous sections and list in Table 1 their performance reported in the MTL literature. We also include the performance of Single-Task Learning (STL), which trains a learning model for each task separately, for comparison. It is easy to see that MTL models perform better than STL counterparts in most cases, which verifies the effectiveness of MTL. Usually, different datasets have their own characteristics, making them more suitable for some MTL approach. For example, according to the studies in [50], [71], [102], different tasks in the School dataset are found to be very similar to each other. According to [54], the Landmine dataset can have two task clusters, where the first cluster consisting of the first 15 tasks corresponds to regions that are relatively highly foliated and the rest tasks belong to another cluster with regions that are bare earth or deserts. According to [61], it is well known in the vaccine design community that some molecules/tasks in the MHC-I dataset can be grouped into empirically defined supertypes known to have similar binding behaviors. For those three datasets, according to Table 1 we can see that the task clustering, task relation learning and decomposition approaches have better performance since they can identify the cluster structure contained in the data in a plain or hierarchical way. For other datasets, they do not have so obvious structure among tasks but some MTL models can learn task correlations, which can bring more insights for model design and the interpretation of experimental results. For example, the task correlations in the SARCOS and Sentiment datasets are shown in Tables 2 and 3 of [79], and the task similarities in the Office-Caltech dataset are shown in Figure 3(b) of [95]. Moreover, for image datasets (i.e., Office-Caltech, Office-Home and ImageCLEF), deep MTL models (e.g., [67], [95]) achieve better performance than shallow models since they can learn powerful feature representations, while the rest datasets are from diverse areas, making shallow models perform well on them.

## 2.8 Another Taxonomy for Regularized MTL Methods

Regularized methods form a main methodology for MTL. Here we classify many regularized MTL algorithms into two main categories: learning with feature covariance and learning with task relations. The former can be viewed as a representative formulation in feature-based MTL, while the latter is for parameter-based MTL.

Objective functions in the first category can be unified as

$$
\min _ {\mathbf {W}, \mathbf {b}, \boldsymbol {\Theta}} L (\mathbf {W}, \mathbf {b}) + \frac {\lambda}{2} \mathrm{tr} (\mathbf {W} ^ {T} \boldsymbol {\Theta} ^ {- 1} \mathbf {W}) + f (\boldsymbol {\Theta}), \tag {32}
$$

where $f ( \cdot )$ denotes a regularizer or constraint on Θ. From the perspective of probabilistic modeling, the regularizer $\begin{array} { r } { \frac { \lambda } { 2 } \mathrm { t r } ( \mathbf { \hat { W } } ^ { T } \mathbf { \hat { \Theta } } ^ { - 1 } \mathbf { W } ) } \end{array}$ corresponds to a matrix-variate normal distribution on W as $\mathbf { W } \ \sim \ { \mathcal { M N } } ( \mathbf { 0 } , { \frac { 1 } { \lambda } } \mathbf { \Theta } \otimes \ \otimes \mathbf { I } )$ . Based on this probabilistic prior, Θ models the covariance between the features since $\textstyle { \frac { 1 } { \lambda } } \Theta$ is the row covariance matrix with each row in W corresponding to a feature and different tasks share the feature covariance. All the models in this category differ in the choice of the function $f ( \cdot )$ on Θ. For example, methods in [10], [40], [41] use $f ( \cdot )$ to restrict the trace of Θ as shown in problems (2) and (12). Moreover, multi-task feature selection methods based on the $\ell _ { 2 , 1 }$ norm such as [23], [24], [25] can be reformulated as instances of problem (32).

Different from the first category, methods in the second category have a unified objective function as

$$
\min _ {\mathbf {W}, \mathbf {b}, \boldsymbol {\Sigma}} L (\mathbf {W}, \mathbf {b}) + \frac {\lambda}{2} \mathrm{tr} (\mathbf {W} \boldsymbol {\Sigma} ^ {- 1} \mathbf {W} ^ {T}) + g (\boldsymbol {\Sigma}), \tag {33}
$$

where $g ( \cdot )$ denotes a regularizer or constraint on Σ. The regularizer $\begin{array} { r } { \frac { \lambda } { 2 } \mathrm { t r } ( \mathbf { W } \mathbf { \Sigma } ^ { - 1 } \mathbf { W } ^ { T } ) } \end{array}$ corresponds to a matrix-variate normal prior on W as $\mathbf { W } \sim { \mathcal { M N } } ( \mathbf { 0 } , \mathbf { I } \otimes { \frac { 1 } { \lambda } } \pmb { \Sigma } )$ , where Σ is to model the task relations since $\scriptstyle { \frac { 1 } { \lambda } } \Sigma$ is the column covariance with each column in W corresponding to a task. From this perspective, the two regularizers for W in problems (32) and (33) have different meanings even though the formulations seem a bit similar. All the methods in this category use different functions $g ( \cdot )$ to learn Σ with different functionalities. For example, the methods in [69], [70], [71], [72], which utilize a priori information on task relations, directly learn W and b by defining $g ( \pmb { \Sigma } ) = 0$ Some task clustering methods [61], [65] identify task clusters by assuming that Σ has a block structure. Several task relation learning methods including [79], [80], [87], [97], [107] directly learn Σ as a covariance matrix by constraining its trace or sparsity in $g ( \Sigma )$ . The trace norm regularization [44] can be formulated as an instance of problem (33).

Even though this taxonomy cannot cover all the regularized MTL methods, it can bring insights to understand regularized MTL methods better and help devise more MTL models. For example, a learning framework is proposed in [108] to learn a suitable multi-task model for a given multi-task problem under problem (33) by utilizing Σ to represent the corresponding multitask model.

## 2.9 Other Settings in MTL

Instead of assuming that different tasks share an identical feature representation, Zhang and Yeung [109] consider a multi-database face recognition problem where face recognition in a database is treated as a task. Since different face databases have different image sizes, here naturally all the tasks do not lie in the same feature space in this application, leading to a heterogeneous-feature MTL problem. To tackle this problem, a multi-task discriminant analysis (MTDA) is proposed in [109] by first projecting data in different tasks into a common subspace and then learning a common projection in this subspace to discriminate different classes in different tasks. In [110], a latent probit model is proposed to generate data of different tasks in different feature spaces via sparse transformations on a shared latent space and then to generate labels based on this latent space.

In many MTL classification problems, each task is explicitly or implicitly assumed to be a binary classification problem as each column in the parameter matrix W contains model parameters for the corresponding task. It is not difficult to see that many methods in the feature learning approach, low-rank approach and decomposition approach can be directly extended to a general setting where each classification task can be a multi-class classification problem and correspondingly multiple columns in W contains model parameters of a multi-class classification task. Such direct extension is applicable since those methods only rely on the entire W or its rows but not columns as a media to share knowledge among tasks. However, to the best of our knowledge, there is no theoretical or empirical study to investigate such direct extension. For most methods in the task clustering and task relation learning approaches, such direct extension does not work since for multiple columns in W corresponding to one task, we do not know which one(s) can be used to represent this task. Therefore, the direct extension may not be the best solution to the general setting. In the following, we introduce four main approaches other than the direct extension to tackle the general setting in MTL where each classification task can be a multi-class classification problem. The first method is to transform the multi-class classification problem in each task into a binary classification problem. For example, multi-task metric learning [70], [111] can do that by treating a pair of data points from the same class as positive and that from different classes as negative. The second recipe is to utilize the characteristics of learners. For example, the linear discriminant analysis can handle binary and multi-class classification problems in a unified formulation and hence MTDA [109] can naturally handle them without changing the formulation. The third approach is to directly learn label correspondence among different tasks. In [112], two learning tasks, which share the training data, aim to maximize the mutual information to identify the correspondence between labels in different tasks. By assuming that all the tasks share the same label space, the last approach including [47], [67], [95] organizes the model parameters of all the tasks in a tensor where the model parameters of each task form a slice. Then the parameter tensor can be regularized by tensor trace norms [47] and a tensor-variate normal prior [95], or factorized as a product of several low-rank matrices or tensors [67].

Most MTL methods assume that the training data in each task are stored in a data matrix. In some case, the training data in each task exhibit a multi-modal structure and hence they are represented in a tensor instead of a matrix. Multilinear multi-task methods proposed in [113], [114] can handle this situation by employing tensor trace norms as a generalization of the trace norm to perform the regularization.

## 2.10 Optimization Techniques in MTL

Optimization techniques used in MTL can be categorized into three main classes as follows.

• Gradient descent method and its variants: The gradient descent method can be used to optimize smooth unconstrained objective functions possessed by many MTL models. If the unconstrained objective function is non-smooth, the subgradient can be used instead and then the gradient descent method can also be used. When there are some constraints in the objective function of MTL models [44], [64], the projected gradient descent method can be used to project the updated solution in each step to the space defined by constraints. For deep MTL models, stochastic gradient descent methods can be used. Moreover, the GradNorm [115] is devised to normalize gradients to balance the learning of multiple tasks and [116] proposes the gradient surgery to avoid the interference between task gradients. Differently, [117] studies MTL from the perspective of multi-objective optimization by learning dynamic loss weights.

• Block Coordinate Descent (BCD) method: The parameters in many MTL models can be divided into several blocks. For example, parameters in learning functions of all the tasks form a block and parameters to represent task relations are from another block. Directly optimizing the objective function of such a MTL model with respect to parameters in all blocks together is not easy. The BCD method, which is also known as the alternating method, is widely used in the MTL literature, e.g., [10], [12], [31], [40], [41], [61], [62], [65], [66], [79], [80], [96], [97], to alternatively optimize each block of parameters while fixing parameters in other blocks. Hence, each step of the BCD method will solve several subproblems, each of which is to optimize with respect to a block of parameters. Compared with the original objective function, each subproblem is easier to be solved and so the BCD method can help reduce the optimization complexity.

• Proximal method [118]: For a nonsmooth objective function, which is the sum of smooth and nonsmooth functions, in an MTL model, the proximal method is frequently used (e.g., [24], [27], [63], [99], [100], [101], [102], [103], [104], [119], [120], [121]) to construct a proximal problem by replacing the smooth function with a quadratic function that may be constructed based on its Taylor series in various ways and the resulting proximal problem is usually easier to be solved than the original problem. The proximal method can accelerate the convergence rate of the optimization process or facilitate the design of distributed optimization algorithms.

## 3 MTL WITH OTHER LEARNING PARADIGMS

In the previous section, we review different MTL approaches for supervised learning tasks. In this section, we overview some works on the combination of MTL with other learning paradigms in machine learning, including unsupervised learning such as clustering, semi-supervised learning, active learning, reinforcement learning, multi-view learning and graphical models, to either improve the performance of supervised MTL further via additional information such as unlabeled data or use MTL to help improve the performance of other learning paradigms.

In most applications, labeled data are expensive to collect but unlabeled data are abundant. So in some MTL applications, the training dataset of each task consists of both labeled and unlabeled data, hence we hope to exploit useful information contained in the unlabeled data to further improve the performance of supervised learning tasks. In machine learning, semi-supervised learning and active learning are two ways to utilize unlabeled data but in different ways. Semi-supervised learning aims to exploit geometrical information contained in the unlabeled data, while active learning selects representative unlabeled data to query an oracle with the hope of increasing the labeling cost as little as possible. Hence semi-supervised learning and active learning can be combined with MTL, leading to three new learning paradigms including semi-supervised multi-task learning [122], [123], [124], multi-task active learning [125], [126], [127] and semi-supervised multi-task active learning [128]. Specifically, a semi-supervised multi-task classification model is proposed in [122], [123] to use random walk to exploit unlabeled data in each task and then cluster multiple tasks via a relaxed Dirichlet process. In [124], a semi-supervised multi-task Gaussian process for regression tasks, where different tasks are related via the hyperprior on the kernel parameters in Gaussian processes of all the tasks, is proposed to incorporate unlabeled data into the design of the kernel function in each task to achieve the smoothness in the corresponding functional spaces. Different from these semi-supervised multi-task methods, multi-task active learning adaptively selects informative unlabeled data for multi-task learners and hence the selection criterion is the core research issue. Reichart et al. [125] believe that data instances to be selected should be as informative as possible for a set of tasks instead of only one task and hence they propose two protocols for multi-task active learning. In [126], the expected error reduction is used as a criterion where each task is modeled by a supervised latent Dirichlet allocation model. Inspired by multi-armed bandits which balance the tradeoff between the exploitation and exploration, a selection strategy is proposed in [127] to consider both the risk of a multi-task learner based on the trace norm regularization and the corresponding confidence bound. In [129], the MTRL method (i.e., problem (21))

is extended to the interactive setting where a human expert is enquired about partial orderings of pairwise task covariances based an inconsistency criterion. In [130], a proposed generalization bound is used to select a subset from multiple unlabeled tasks to acquire labels to improve the generalization performance of all the tasks. For semi-supervised multi-task active learning, Li et al. [128] propose a model to use the Fisher information as a criterion to select unlabeled data to acquire their labels with the semi-supervised multi-task classification model [122], [123] as the classifier for each task.

MTL achieves the performance improvement in not only supervised learning tasks but also unsupervised learning tasks such as clustering. In [131], a multi-task Bregman clustering method is proposed based on single-task Bregman clustering by using the earth mover distance to minimize distances between any pair of tasks in terms of cluster centers and then in [132], [133], an improved version of [131] and its kernel extension are proposed to avoid the negative effect caused by the regularizer in [131] via choosing the better one between single-task and multitask Bregman clustering. In [134], a multi-task kernel k-means method is proposed by learning the kernel matrix via both MMD between any pair of tasks and the Laplacian regularization that helps identify a smooth kernel space. In [135], two proposed multitask clustering methods are extensions of the MTFL and MTRL methods by treating labels as cluster indicators to be learned. In [136], the principle of MTL is incorporated into the subspace clustering by capturing correlations between data instances. In [137], a multi-task clustering method belonging to instance-based MTL is proposed to share data instances among different tasks. In [138], a multi-task spectral clustering algorithm, which can handle the out-of-sample issue via a linear function to learn the cluster assignment, is proposed to achieve the feature selection among tasks via the $\ell _ { 2 , 1 }$ regularization [139]. [140] proposes to identify the task cluster structure and learn task relations together.

Reinforcement Learning (RL) is a promising area in machine learning and has shown superior performance in many applications such as game playing (e.g., Atari and Go) and robotics. MTL can help boost the performance of reinforcement learning, leading to Multi-task Reinforcement Learning (MRL). Some works [141], [142], [143], [144], [145], [146], [147], [148], [149], [150] adapt the ideas introduced in Section 2 to MRL. Specifically, in [141] where a task solves a sequence of Markov Decision Processes (MDPs), a hierarchical Bayesian infinite mixture model is used to model the distribution over MDPs and for each new MDP, previously learned distributions are used as an informative prior. In [142], a regionalized policy representation is introduced to characterize the behavior of an agent in each task and a Dirichlet process is placed over regionalized policy representations across multiple tasks to cluster tasks. In [143], a Gaussian process temporal-difference value function model is used for each task and a hierarchical Bayesian approach is to model the distribution over value functions in different tasks. Calandriello et al. [144] assume that parameter vectors of value functions in different tasks are jointly sparse and then extend the MTFS method with the $\ell _ { 2 , 1 }$ regularization as well as the MTFL method to learn value functions in multiple tasks together. In [145], a model associating each subtask with a modular subpolicy is proposed to learn from policy sketches, which annotate tasks with sequences of named subtasks and provide information about high-level structural relationships among tasks. In [146], a multi-task contextual bandit is introduced to leverage or learn similarities in contexts among arms to improve the prediction of rewards from contexts. In [147], a multi-task linearly solvable MDP, whose task basis matrix contains a library of component tasks shared by all the tasks, is proposed to maintain a parallel distributed representation of tasks each of which enables an agent to draw on macro actions simultaneously. In [148], a multi-task deep RL model based on the attention can automatically group tasks into sub-networks on a state-level granularity. In [149], a sharing experience framework is introduced to use taskspecific rewards to identify similar parts defined as shared-regions which can guide the experience sharing of task policies. In [150], multi-task soft option learning, a hierarchical framework based on planning as inference, is regularized by a shared prior to avoid training instabilities and allow the fine-tuning of options for new tasks without forgetting learned policies. The idea of compression and distillation have been incorporated into MRL as in [151], [152], [153], [154]. For example, in [151], the proposed Actor-Mimic method combines both deep reinforcement learning and model compression techniques to train a policy network which can learn to act for multiple tasks. In [152], a policy distillation method is proposed to not only train an efficient network to learn the policy of an agent but also consolidate multiple task-specific policies into a single policy. In [153], the problem of multi-task multi-agent reinforcement learning under the partial observability is addressed by distilling decentralized single-task policies into a unified policy across multiple tasks. In [154], each task has its own policy which is constrained to be close to a shared policy that is trained by the distillation. Some works [155], [156], [157], [158], [159], [160] in MRL focus on online and distributed settings. Specifically, in [155], a distributed MRL framework is devised to model it as an instance of general consensus and an efficient decentralized solver is developed. In [156], [157], multiple goal-directed tasks are learned in an online setup without the need for expert supervision by actively sampling harder tasks. In [158], a distributed agent is developed to not only use resources more efficiently in singlemachine training but also scale to thousands of machines without sacrificing data efficiency or resource utilization. [159] formulates MRL from a perspective of variational inference and it proposes a novel distributed solver with quadratic convergence guarantees. In [160], an online learning algorithm is proposed to dynamically combine different auxiliary tasks which provide gradient directions to speed up the training of the main reinforcement learning task. Some works study the theoretical foundation of MRL. For example, in [161], sharing representations among tasks is analyzed with theoretical guarantees to highlight conditions to share representations and finite-time bounds of approximated value-iteration are extended to the multi-task setting. Moreover, there are some works to design novel MRL methods. For example, in [162], a MRL framework is proposed to train agent to employ hierarchical policies that decide when to use a previously learned policy and when to learn a new skill with a temporal grammar that helps the agent learn complex temporal dependencies. [163] studies the problem of parallel learning of multiple sequentialdecision tasks and proposes to automatically adapt the contribution of each task to the updates of the agent to make all tasks have comparable impacts on the learning dynamics. In [164], a selfsupervised representation learning algorithm is proposed for multi task deep RL to capture structured information about environment dynamics based on multi-step predictive representations of future observations.

Multi-view learning assumes that each data point is associated with multiple sets of features where each set corresponds to a view and it usually exploits information contained in multiple views for supervised or semi-supervised learning tasks. Multi-task multiview learning extends multi-view learning to the MTL setting where each task is a multi-view learning problem. Specifically, in [165], a graph-based method is proposed for multi-task multiview classification problems. In a task, each view is enforced to be consistent with both other views and labels, while different tasks are expected to have similar predictions on views they share, making views as a bridge to construct the task relatedness. In [166], both a regularized MTL method [71] and the MTRL method are applied to each view of different tasks and different views in a task are expected to achieve an agreement on unlabeled data. Different from [165], [166] which study the multi-task multi-view classification problem, in [167], [168], two multi-task multi-view clustering methods are proposed and both methods consider three factors: within-view-task clustering which conducts clustering on each view in a task, view relation learning which minimizes the disagreement among views in a task, and lowrank structure learning which aims to learn a shared subspace for different tasks under a common view. The difference between these two methods is that the first method uses a bipartite graph coclustering method for nonnegative data while the other one adopts a semi-nonnegative matrix tri-factorization to cluster general data. In [169], a multi-label multi-view algorithm is proposed to not only learn common features via the $\ell _ { 2 , 1 }$ regularization but also identify useless views via the Frobenius norm. In multi-task multiview learning, each task is usually supplied with both labeled and unlabeled data, hence this paradigm can also be viewed as another way to utilize unlabeled information for MTL. A deep multi-task multi-view model is proposed in [170] to fuse all the views based on the cross-stitch network.

MTL can help learn more accurate structure in graphical models. In [171], an algorithm is proposed to learn Bayes network structures by assuming that different networks/tasks share similar structures via a common prior and then a heuristic search is used to find structures with high scores for all the tasks. With a similar idea, multiple Gaussian graphical models are jointly learned in [172] by assuming joint sparsity among precision matrices via the $\ell _ { \infty , 1 }$ norm regularization. In [173], some domain knowledge about task relations is incorporated into the learning of multiple Bayesian networks. By viewing the feature interaction matrix as a form of graphical models to model pairwise relations between features, two models are proposed in [174] to learn a quadratical function, where the feature interaction matrix defines the quadratic term, for each task based on the $\ell _ { 2 , 1 }$ and tensor trace norm regularization, respectively.

According to the above discussions, we can see that most research works discussed in this section follow the spirits of MTL approaches introduced in Section 2 and adapt to their own settings.

## 4 HANDLING BIG DATA

When the number of tasks is large, the total number of training data in all the tasks can be very big and hence a ‘big’ aspect in MTL denotes the number of tasks. In this case, we can either devise online, parallel, or distributed MTL models to accelerate the learning process. Another ‘big’ aspect in MTL lies in the data dimensionality which can be very high. In this situation, we can speedup the learning via feature selection, dimensionality reduction and feature hashing to reduce the dimension without losing too much useful information. In this section, we review some relevant works.

When the number of tasks is very big, we can devise some parallel MTL methods to speedup the learning process on multi-CPU or multi-GPU devices. As a representative formulation in feature-based MTL, problem (32) is easy to parallelize since when given the feature covariance matrix Θ, the learning of different tasks can be decoupled. However, for problem (33) in parameterbased MTL, the situation is totally different since even given the task covariance matrix Σ, different tasks are still coupled, making the direct parallelization fail. In order to parallelize problem (33), Zhang [175] uses the FISTA algorithm to design a surrogate function for problem (33) with a given Σ, where the surrogate function is decomposable with respect to tasks, leading to a parallel design for MTL based on different loss functions including the hinge, -insensitive and square losses.

Online multi-task learning is also capable of handling a big number of tasks. In [176], [177], under a setting where all the tasks contribute toward a common goal, the relation between tasks is measured via a global loss function and several online algorithms are proposed to use absolute norms as the global loss function. In [178], online MTL algorithms are devised by modeling the task relatedness via hard constraints that the mtuple of actions for tasks satisfies. In [179], perceptron-based online algorithms are proposed for multi-task binary classification problems where task similarities are measured based on either the geometric closeness of the task reference vectors or the dimension of their spanned subspace. In [180], a recursive Bayesian online algorithm based on Gaussian processes is devised to update both estimations and confidence intervals when data instances arrive sequentially. In [181], an online version of the MTRL method is proposed to update both the model parameters and task covariance in a sequential way. An online multi-task learning algorithm is proposed in [182] to jointly learn the per-task model and the task relations by smoothing the loss function of each task w.r.t. a task distribution and adaptively refining this distribution over time. In [183], an online multi-task model is proposed to learn both a lowrank component and a group sparse component to characterize task relations. In [184], a multi-task passive-aggressive method is proposed to learn multiple relative similarity learning tasks, each of which is to learn a similarity function from data with relative constraints. In [185], a Gaussian distribution, whose mean or covariance consists of a local component for each task and a global component shared by all the tasks, over each task is used as a confidence measure to guide the online MTL process.

Training data can locate at different devices, making the design of distributed MTL models important. In [186], a communicationefficient distributed MTL algorithm, where each machine learns a task, based on the debiased Lasso is proposed to learn jointly sparse features in a high-dimensional space. In [187], the MTRL method (i.e., problem (21)) is extended to the distributed setting based on a stochastic dual coordinate ascent method. In [188], to protect the privacy of data, a privacy-preserving distributed MTL method is proposed based on a privacy-preserving proximal gradient algorithm with asynchronous updates. In [189], federated multi-task learning is proposed as an extension of distributed multi-task learning to consider both stragglers and fault tolerance. In [190], a distributed multi-task algorithm is proposed under the online MTL setting.

For high-dimensional data in MTL, we can use multi-task feature selection methods to reduce the dimension or extend singletask dimension reduction techniques to the multi-task setting as did in [109]. Another option is to use the feature hashing and in [191], multiple hashing functions are proposed to accelerate the joint learning of multiple tasks.

## 5 APPLICATIONS

MTL has many applications in various areas including computer vision, bioinformatics, health informatics, speech, NLP, web, and so on. In Table 2, we categorize different MTL problems in each application area according to MTL approaches they used, where the classification of MTL approaches has been already introduced in Section 2. In the last column of Table 2, we list some problems in various application areas which are different from other columns. For application problems listed in Table 2, application-dependent MTL models have been proposed to solve them.<sup>6</sup> Though these models are different from each other, there are some characteristics in respective areas. For example, in computer vision, deep MTL models, most of which belong to the feature transformation approach, exhibit good performance, making this approach popular in computer vision. In bioinformatics and health informatics, the interpretability of learning models is more important in some sense. Therefore, the feature selection and task relation learning approaches are widely used in this area as the former approach can identify useful features and the latter one can quantitatively show task relations. In speech and NLP, the data exhibit a sequential structure, which makes recurrent-neuralnetwork-based deep MTL models in the feature transformation approach play an important role. As the data in web applications is of a large scale, this area favors simple models such as linear models or their ensembles based on boosting. Among all the MTL approaches, the feature transformation, feature selection and task relation learning approaches are among the most widely used MTL approaches in different application areas according to Table 2.

When encountering a new application problem which can be modeled as a MTL problem, we need to judge whether tasks in this problem are related in terms of either low-level features or high-level concepts. If so, by treating Table 2 as a look-up table. we can identify a problem in Table 2 similar to the new problem and then adapt the corresponding MTL model to solve the new problem. Otherwise, we can try popular MTL approaches in the respective area.

## 6 THEORETICAL ANALYSES

As well as designing MTL models and exploiting MTL applications, there are some works to study theoretical aspects of MTL and here we review them.

The generalization bound, which is to upper-bound the generalization loss in terms of the training loss, model complexity and confidence, is core in learning theory since it can identify the learnability and induce the sample complexity. There are several works [40], [49], [249], [250], [251], [252], [253], [254], [255], [256], [257], [258], [259], [260] to study the generalization bound of different MTL models. In Table 3, we compare those works in terms of the analyzed MTL model, analysis tool and the convergence rate of the corresponding bound which is based on the number of tasks (i.e., m) and the average number of data points per task (i.e., n<sub>0</sub>). According to Table 3, we can see that some works (i.e., [49], [249], [250], [253], [254], [257], [258])

TABLE 2 The classification of works about MTL applications in different areas according to different MTL approaches.

<table><tr><td>Approach</td><td>Computer Vision</td><td>Bioinformatics &amp; Health Informatics</td><td>Speech &amp; NLP</td><td>Web</td><td>Miscellaneous</td></tr><tr><td>Feature Transformation</td><td>Visual tracking [192], [193]Action recognition [199]Facial landmark detection [15]Scene classification [205]Attribute prediction [206]Image rotation [208]Immediacy prediction [209]Pose estimation [19]Thumbnail selection [16]Face verification [210]</td><td>Protein subcellular location prediction [194]Protein interaction prediction [200]Biological image classification [17]</td><td>Speech synthesis [195], [196]Speech recognition [201], [202]Jointly learning of NLP tasks [204]Dialog state tracking [18]Machine translation [207]Syntactic parsing [207]</td><td>Learning to rank [197]</td><td>Stock prediction [198]Localization [203]</td></tr><tr><td>Feature Selection</td><td>Face and object recognition [211]Brain imaging [214]</td><td>siRNA efficacy prediction [212]Genetic marker detection [215]Mental state examination [216]Predict cognitive outcome [217]Survival analysis [120]Genetic trait prediction [218]Gene expression association [219]</td><td>Microblog analysis [119]</td><td>Behavioral targeting [213]</td><td></td></tr><tr><td>Low-Rank</td><td>Image segmentation [220]Saliency detection [222]</td><td>Identification of longitudinal phenotypic markers [223]</td><td></td><td></td><td>Climate prediction [221]</td></tr><tr><td>Task Clustering</td><td>Image segmentation [224]Age estimation [226]Facial action unit prediction [228]Action recognition [229]</td><td>Brain-computer interfaces [225]Personalized medical treatment [227]Genetic trait prediction [218]</td><td></td><td></td><td></td></tr><tr><td>Task Relation Learning</td><td></td><td>Protein subcellular location prediction [194]Organism modeling [232]MHC-I binding prediction [236]Splice-site prediction [236]Prioritization of disease genes [240]Protein interaction prediction [242]Identifying antigenic variants [243]</td><td>Sentiment classification [230]</td><td>Web search ranking [231]Collaborative filtering [233]Display advertising [237]</td><td>Localization [203]Robotics [234], [235]Trajectory regression [238]Traffic sign recognition [239]Soil moisture forecasts [241]</td></tr><tr><td>Decomposition</td><td>Multi-view tracking [244]Pose estimation [245]Person re-identification [247]</td><td>Genetic trait prediction [218]Protein interaction prediction [246]</td><td></td><td></td><td></td></tr></table>

analyze different MTL models based on various analysis tools and the best convergence rate is $O ( \frac { 1 } { \sqrt { m n _ { 0 } } } )$ . Though MTL models analyzed in [12], [40], [251], [259] are not the same, those MTL models exhibit similar objectives to learn a linear or nonlinear feature transformation shared by all the tasks, and the convergence rates are $\begin{array} { r } { O ( \frac { 1 } { \sqrt { m n _ { 0 } } } ) } \end{array}$ except [12]. The trace norm regularization (i.e., Problem (13)) is analyzed in [255], [256], [258], [260], among which [260] has the best convergence rate based on the local Rademacher complexity. A related MTL model based on the Schatten norm regularization is analyzed in [252] with an $O ( \frac { 1 } { \sqrt { m n _ { 0 } } } )$ convergence rate. For the graph regularization [69], [71] analyzed in [252], [260], the local Rademacher complexity leads to a better convergence rate (i.e., $O \big ( \frac { 1 } { ( m n _ { 0 } ) ^ { \alpha } } \big )$ for some constant $\alpha \in ( 0 . 5 , 1 ) _ { \AA } ^ { \cdot }$ ) and a similar observation holds for problem (4). In a word, various analysis tools can be used to analyze MTL models and among them, the local Rademacher complexity can derive tighter generalization bounds than others.

Besides generalization bounds, there are some works to study other theoretical problems in MTL. For example, Argyriou et al. [261], [262] discuss conditions where representer theorems hold for regularized MTL algorithms. Several studies [263], [264], [265] investigate conditions to well recover true features for multitask feature selection models.

TABLE 3 Comparison of generalization bounds derived in different works in terms of the analyzed MTL model, analysis tool and the convergence rate of the bound. m denotes the number of tasks and n denotes the average number of data points per task.

<table><tr><td>MTL Model</td><td>Reference</td><td>Analysis Tool</td><td>Convergence rate</td></tr><tr><td>Tasks from an environment</td><td>[249], [250]</td><td>VC dimension &amp; Covering number</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Task distributions can be transformed</td><td>[254]</td><td>VC dimension</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Task clustering</td><td>[49]</td><td>VC dimension</td><td> $O(m\ln(n_0/m)/n_0)$ </td></tr><tr><td>Multi-task kernel classifier</td><td>[257]</td><td>Covering number</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Problem (26) with Eq. (27)</td><td>[258]</td><td>Multi-task stability</td><td> $O(m\sqrt{m/n_0})$ </td></tr><tr><td>Multi-task data compression</td><td>[253]</td><td>Kolmogorov complexity</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Problem (10)</td><td>[40]</td><td>Covering number</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Problem (10) without U</td><td>[251]</td><td>Rademacher complexity</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>Problem (3)</td><td>[12]</td><td>Rademacher complexity</td><td> $O(1/\sqrt{n_0})$ </td></tr><tr><td>Learn a common feature transformation</td><td>[259]</td><td>Gaussian average</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td rowspan="4">Problem (13)</td><td>[258]</td><td>Multi-task stability</td><td> $O(m/n_0)$ </td></tr><tr><td>[255]</td><td>Rademacher complexity</td><td> $O(\ln(m)/\sqrt{n_0})$ </td></tr><tr><td>[256]</td><td>Rademacher complexity</td><td> $O(\max(\sqrt{\ln(mn_0)/mn_0}, 1/\sqrt{n_0}))$ </td></tr><tr><td>[260]</td><td>Local Rademacher complexity</td><td> $O(1/(mn_0)^{\alpha}), 0.5 < \alpha < 1$ </td></tr><tr><td>Schatten-norm-regularized MTL models</td><td>[252]</td><td>Rademacher complexity</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td rowspan="2">Graph regularizers [69], [71]</td><td>[252]</td><td>Rademacher complexity</td><td> $O(1/\sqrt{mn_0})$ </td></tr><tr><td>[260]</td><td>Local Rademacher complexity</td><td> $O(1/(mn_0)^{\alpha}), 0.5 < \alpha < 1$ </td></tr><tr><td rowspan="2">Problem (4)</td><td>[255]</td><td>Rademacher complexity</td><td> $O(1/\sqrt{n_0})$ </td></tr><tr><td>[260]</td><td>Local Rademacher complexity</td><td> $O(1/(mn_0)^{\alpha}), 0.5 < \alpha < 1$ </td></tr></table>

## 7 CONCLUSIONS AND DISCUSSIONS

In this paper, we survey different aspects of MTL. First, after giving the definition of MTL, we give a classification of supervised MTL models into five main approaches and discuss their characteristics. Then we review the combinations of MTL with other learning paradigms. The online, parallel and distributed MTL models as well as dimensionality reduction and feature hashing are discussed to speedup the learning process. The applications of MTL in various areas are introduced to show the usefulness of MTL and theoretical aspects of MTL are discussed.

In future studies, there are several issues to be addressed. Firstly, outlier tasks, which are unrelated to other tasks, are well known to hamper the performance of all the tasks when learning them jointly. There are some methods to alleviate negative effects that outlier tasks bring. However, there lacks principled ways and theoretical analyses to study the resulting negative effects. In order to make MTL safe to be used by human, this is an important issue and needs more studies.

Secondly, deep learning has become a dominant approach in many areas and several multi-task deep models belonging to the feature transformation, low-rank, task clustering and task relation learning approaches have been proposed as reviewed in Sections 2, 3 and 5. As discussed, most of them only share hidden layers. This approach is powerful when all the tasks are related, but it is vulnerable to noisy and outlier tasks that can deteriorate the performance dramatically. We believe that it is desirable to design flexible and robust deep multi-task models.

Lastly, existing studies mainly focus on supervised learning tasks, and only a few ones are on other tasks such as unsupervised learning, semi-supervised learning, active learning, multi-view learning and reinforcement learning tasks. It is natural to adapt or extend the five approaches introduced in Section 2 to those non-supervised learning tasks. We think that such adaptation and extension require more efforts to design appropriate models. Moreover, it is worth trying to apply MTL to other areas in artificial intelligence such as logic and planning to broaden its application scopes.

Acknowledgments This work is supported by NSFC 62076118.

## REFERENCES

[1] R. Caruana, “Multitask learning,” MLJ, 1997.  
[2] Q. Yang, Y. Zhang, W. Dai, and S. J. Pan, Transfer Learning. Cambridge University Press, 2020.  
[3] M.-L. Zhang and Z.-H. Zhou, “A review on multi-label learning algorithms,” IEEE TKDE, 2014.  
[4] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.  
[5] Y. Zhang and Q. Yang, “An overview of multi-task learning,” National Science Review, 2018.  
[6] X. Yang, S. Kim, and E. P. Xing, “Heterogeneous multitask learning with joint sparsity constraints,” in NIPS, 2009.  
[7] S. Bickel, J. Bogojeska, T. Lengauer, and T. Scheffer, “Multi-task learning for HIV therapy screening,” in ICML, 2008.  
[8] X. Liao and L. Carin, “Radial basis function network for multi-task learning,” in NIPS, 2005.  
[9] D. L. Silver, R. Poirier, and D. Currie, “Inductive transfer with contextsensitive neural networks,” MLJ, 2008.  
[10] A. Argyriou, T. Evgeniou, and M. Pontil, “Convex multi-task feature learning,” MLJ, 2008.  
[11] A. Argyriou, C. A. Micchelli, M. Pontil, and Y. Ying, “A spectral regularization framework for multi-task structure learning,” in NIPS, 2007.  
[12] A. Maurer, M. Pontil, and B. Romera-Paredes, “Sparse coding for multitask and transfer learning,” in ICML, 2013.  
[13] J. Zhu, N. Chen, and E. P. Xing, “Infinite latent SVM for classification and multi-task learning,” in NIPS, 2011.  
[14] M. K. Titsias and M. Lazaro-Gredilla, “Spike and slab variational´ inference for multi-task and multiple kernel learning,” in NIPS, 2011.  
[15] Z. Zhang, P. Luo, C. C. Loy, and X. Tang, “Facial landmark detection by deep multi-task learning,” in ECCV, 2014.  
[16] W. Liu, T. Mei, Y. Zhang, C. Che, and J. Luo, “Multi-task deep visualsemantic embedding for video thumbnail selection,” in CVPR, 2015.  
[17] W. Zhang, R. Li, T. Zeng, Q. Sun, S. Kumar, J. Ye, and S. Ji, “Deep model based transfer and multi-task learning for biological image analysis,” in KDD, 2015.  
[18] N. Mrksic, D. O. S<sup>´</sup> eaghdha, B. Thomson, M. Gasic, P. Su, D. Vandyke,´ T. Wen, and S. J. Young, “Multi-domain dialog state tracking using recurrent neural networks,” in ACL, 2015.  
[19] S. Li, Z. Liu, and A. B. Chan, “Heterogeneous multi-task learning for human pose estimation with deep convolutional neural network,” IJCV, 2015.  
[20] Y. Shinohara, “Adversarial multi-task learning of deep neural networks for robust speech recognition,” in Interspeech, 2016.  
[21] P. Liu, X. Qiu, and X. Huang, “Adversarial multi-task learning for text classification,” in ACL, 2017.  
[22] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert, “Cross-stitch networks for multi-task learning,” in CVPR, 2016.  
[23] G. Obozinski, B. Taskar, and M. Jordan, “Multi-task feature selection,” tech. rep., University of California, Berkeley, 2006.  
[24] J. Liu, S. Ji, and J. Ye, “Multi-task feature learning via efficient $l _ { 2 , 1 ^ { - } }$ norm minimization,” in UAI, 2009.  
[25] S. Lee, J. Zhu, and E. P. Xing, “Adaptive multi-task lasso: With application to eQTL detection,” in NIPS, 2010.  
[26] N. S. Rao, C. R. Cox, R. D. Nowak, and T. T. Rogers, “Sparse overlapping sets lasso for multitask learning and its application to fMRI analysis,” in NIPS, 2013.  
[27] P. Gong, J. Zhou, W. Fan, and J. Ye, “Efficient multi-task feature learning with calibration,” in KDD, 2014.  
[28] J. Wang and J. Ye, “Safe screening for multi-task feature learning with multiple data matrices,” in ICML, 2015.  
[29] H. Liu, M. Palatucci, and J. Zhang, “Blockwise coordinate descent procedures for the multi-task lasso, with applications to neural semantic basis discovery,” in ICML, 2009.  
[30] P. Gong, J. Ye, and C. Zhang, “Multi-stage multi-task feature learning,” JMLR, 2013.  
[31] A. C. Lozano and G. Swirszcz, “Multi-level lasso for sparse multi-task regression,” in ICML, 2012.  
[32] X. Wang, J. Bi, S. Yu, and J. Sun, “On multiplicative multitask feature learning,” in NIPS, 2014.  
[33] L. Han, Y. Zhang, G. Song, and K. Xie, “Encoding tree sparsity in multi-task learning: A probabilistic framework,” in AAAI, 2014.  
[34] T. Jebara, “Multi-task feature and kernel selection for SVMs,” in ICML, 2004.  
[35] S. Kim and E. P. Xing, “Tree-guided group lasso for multi-task regression with structured sparsity,” in ICML, 2010.  
[36] Y. Zhou, R. Jin, and S. C. H. Hoi, “Exclusive lasso for multi-task feature selection,” in AISTATS, 2010.  
[37] Y. Zhang, D.-Y. Yeung, and Q. Xu, “Probabilistic multi-task feature selection,” in NIPS, 2010.  
[38] D. Hernandez-Lobato and J. M. Hern ´ andez-Lobato, “Learning feature ´ selection dependencies in multi-task learning,” in NIPS, 2013.  
[39] D. Hernandez-Lobato, J. M. Hern´ andez-Lobato, and Z. Ghahramani,´ “A probabilistic model for dirty multi-task feature selection,” in ICML, 2015.  
[40] R. K. Ando and T. Zhang, “A framework for learning predictive structures from multiple tasks and unlabeled data,” JMLR, 2005.  
[41] J. Chen, L. Tang, J. Liu, and J. Ye, “A convex formulation for learning shared structures from multiple tasks,” in ICML, 2009.  
[42] A. Agarwal, H. Daume III, and S. Gerber, “Learning multiple tasks´ using manifold regularization,” in NIPS, 2010.  
[43] J. Zhang, Z. Ghahramani, and Y. Yang, “Learning multiple related tasks using latent independent component analysis,” in NIPS, 2005.  
[44] T. K. Pong, P. Tseng, S. Ji, and J. Ye, “Trace norm regularization: Reformulations, algorithms, and multi-task learning,” SIAM Journal on Optimization, 2010.  
[45] L. Han and Y. Zhang, “Multi-stage multi-task learning with reduced rank,” in AAAI, 2016.  
[46] A. M. McDonald, M. Pontil, and D. Stamos, “Spectral k-support norm regularization,” in NIPS, 2014.  
[47] Y. Yang and T. M. Hospedales, “Trace norm regularised deep multi-task learning,” in ICLR, Workshop Track, 2017.  
[48] S. Thrun and J. O’Sullivan, “Discovering structure in multiple learning tasks: The TC algorithm,” in ICML, 1996.  
[49] K. Crammer and Y. Mansour, “Learning multiple tasks using shared hypotheses,” in NIPS, 2012.  
[50] B. Bakker and T. Heskes, “Task clustering and gating for Bayesian multitask learning,” JMLR, 2003.  
[51] K. Yu, V. Tresp, and A. Schwaighofer, “Learning Gaussian processes from multiple tasks,” in ICML, 2005.  
[52] S. Yu, V. Tresp, and K. Yu, “Robust multi-task learning with tprocesses,” in ICML, 2007.  
[53] W. Lian, R. Henao, V. Rao, J. E. Lucas, and L. Carin, “A multitask point process predictive model,” in ICML, 2015.  
[54] Y. Xue, X. Liao, L. Carin, and B. Krishnapuram, “Multi-task learning for classification with Dirichlet process priors,” JMLR, 2007.  
[55] Y. Xue, D. B. Dunson, and L. Carin, “The matrix stick-breaking process for flexible multi-task learning,” in ICML, 2007.  
[56] H. Li, X. Liao, and L. Carin, “Nonparametric Bayesian feature selection for multi-task learning,” in ICASSP, 2011.  
[57] Y. Qi, D. Liu, D. B. Dunson, and L. Carin, “Multi-task compressive sensing with Dirichlet process priors,” in ICML, 2008.  
[58] K. Ni, L. Carin, and D. B. Dunson, “Multi-task learning for sequential data via iHMMs and the nested Dirichlet process,” in ICML, 2007.  
[59] K. Ni, J. W. Paisley, L. Carin, and D. B. Dunson, “Multi-task learning for analyzing and sorting large databases of sequential data,” IEEE TSP, 2008.  
[60] A. Passos, P. Rai, J. Wainer, and H. Daume III, “Flexible modeling of´ latent task structures in multitask learning,” in ICML, 2012.  
[61] L. Jacob, F. R. Bach, and J.-P. Vert, “Clustered multi-task learning: A convex formulation,” in NIPS, 2008.  
[62] Z. Kang, K. Grauman, and F. Sha, “Learning with whom to share in multi-task feature learning,” in ICML, 2011.  
[63] L. Han and Y. Zhang, “Learning multi-level task groups in multi-task learning,” in AAAI, 2015.  
[64] A. Barzilai and K. Crammer, “Convex multi-task learning by clustering,” in AISTATS, 2015.  
[65] Q. Zhou and Q. Zhao, “Flexible clustered multi-task learning by learning representative tasks,” IEEE TPAMI, 2016.  
[66] A. Kumar and H. Daume III, “Learning task grouping and overlap in´ multi-task learning,” in ICML, 2012.  
[67] Y. Yang and T. M. Hospedales, “Deep multi-task representation learning: A tensor factorisation approach,” in ICLR, 2017.  
[68] J. Zhou, J. Chen, and J. Ye, “Clustered multi-task learning via alternating structure optimization,” in NIPS, 2011.  
[69] T. Evgeniou and M. Pontil, “Regularized multi-task learning,” in KDD, 2004.  
[70] S. Parameswaran and K. Q. Weinberger, “Large margin multi-task metric learning,” in NIPS, 2010.  
[71] T. Evgeniou, C. A. Micchelli, and M. Pontil, “Learning multiple tasks with kernel methods,” JMLR, 2005.  
[72] T. Kato, H. Kashima, M. Sugiyama, and K. Asai, “Multi-task learning via conic programming,” in NIPS, 2007.  
[73] S. Feldman, M. R. Gupta, and B. A. Frigyik, “Revisiting Stein’s paradox: Multi-task averaging,” JMLR, 2014.  
[74] I. Yamane, H. Sasaki, and M. Sugiyama, “Regularized multitask learning for multidimensional log-density gradient estimation,” Neural Computation, 2016.  
[75] N. Gornitz, C. Widmer, G. Zeller, A. Kahles, S. Sonnenburg, and¨ G. Ratsch, “Hierarchical multitask structured output learning for large-¨ scale sequence segmentation,” in NIPS, 2011.  
[76] E. V. Bonilla, K. M. A. Chai, and C. K. I. Williams, “Multi-task Gaussian process prediction,” in NIPS, 2007.  
[77] K. M. A. Chai, “Generalization errors and learning curves for regression with multi-task Gaussian processes,” in NIPS, 2009.  
[78] Y. Zhang and D.-Y. Yeung, “Multi-task learning using generalized t process,” in AISTATS, 2010.  
[79] Y. Zhang and D.-Y. Yeung, “A convex formulation for learning task relationships in multi-task learning,” in UAI, 2010.  
[80] Y. Zhang and D.-Y. Yeung, “A regularization approach to learning task relationships in multitask learning,” ACM TKDD, 2014.  
[81] Y. Zhang and D.-Y. Yeung, “Multi-task boosting by exploiting task relationships,” in ECMLPKDD, 2012.  
[82] Y. Zhang and D.-Y. Yeung, “Multilabel relationship learning,” ACM TKDD, 2013.  
[83] F. Dinuzzo, C. S. Ong, P. V. Gehler, and G. Pillonetto, “Learning output kernels with block coordinate descent,” in ICML, 2011.  
[84] C. Ciliberto, Y. Mroueh, T. A. Poggio, and L. Rosasco, “Convex learning of multiple tasks and their structure,” in ICML, 2015.  
[85] C. Ciliberto, L. Rosasco, and S. Villa, “Learning multiple visual tasks while discovering their structure,” in CVPR, 2015.  
[86] P. Jawanpuria, M. Lapin, M. Hein, and B. Schiele, “Efficient output kernel learning for multiple tasks,” in NIPS, 2015.  
[87] Y. Zhang and Q. Yang, “Learning sparse task relations in multi-task learning,” in AAAI, 2017.  
[88] Y. Zhang and J. G. Schneider, “Learning multiple tasks with a sparse matrix-normal penalty,” in NIPS, 2010.  
[89] C. Archambeau, S. Guo, and O. Zoeter, “Sparse Bayesian multi-task learning,” in NIPS, 2011.  
[90] M. Yang, Y. Li, and Z. Zhang, “Multi-task learning with Gaussian matrix generalized inverse Gaussian model,” in ICML, 2013.  
[91] Y. Zhang and D.-Y. Yeung, “Learning high-order task relationships in multi-task learning,” in IJCAI, 2013.  
[92] P. Rai, A. Kumar, and H. Daume III, “Simultaneously leveraging output´ and task structures for multiple-output regression,” in NIPS, 2012.  
[93] B. Rakitsch, C. Lippert, K. M. Borgwardt, and O. Stegle, “It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals,” in NIPS, 2013.  
[94] A. R. Goncalves, F. J. V. Zuben, and A. Banerjee, “Multi-task sparse structure learning with Gaussian copula models,” JMLR, 2016.  
[95] M. Long, Z. Cao, J. Wang, and P. S. Yu, “Learning multiple tasks with multilinear relationship networks,” in NIPS, 2017.  
[96] Y. Zhang, “Heterogeneous-neighborhood-based multi-task local learning algorithms,” in NIPS, 2013.  
[97] G. Lee, E. Yang, and S. J. Hwang, “Asymmetric multi-task learning based on task relatedness and loss,” in ICML, 2016.  
[98] A. Jalali, P. Ravikumar, S. Sanghavi, and C. Ruan, “A dirty model for multi-task learning,” in NIPS, 2010.  
[99] J. Chen, J. Liu, and J. Ye, “Learning incoherent sparse and low-rank patterns from multiple tasks,” in KDD, 2010.  
[100] J. Chen, J. Zhou, and J. Ye, “Integrating low-rank and group-sparse structures for robust multi-task learning,” in KDD, 2011.  
[101] P. Gong, J. Ye, and C. Zhang, “Robust multi-task feature learning,” in KDD, 2012.  
[102] W. Zhong and J. T. Kwok, “Convex multitask learning with flexible task clusters,” in ICML, 2012.  
[103] A. Zweig and D. Weinshall, “Hierarchical regularization cascade for joint learning,” in ICML, 2013.  
[104] L. Han and Y. Zhang, “Learning tree structure in multi-task learning,” in KDD, 2015.  
[105] P. Jawanpuria and J. S. Nath, “A convex feature learning formulation for latent task structure discovery,” in ICML, 2012.  
[106] B. Gong, Y. Shi, F. Sha, and K. Grauman, “Geodesic flow kernel for unsupervised domain adaptation,” in CVPR, 2012.  
[107] M. Solnon, S. Arlot, and F. R. Bach, “Multi-task regression using minimal penalties,” JMLR, 2012.  
[108] Y. Zhang, Y. Wei, and Q. Yang, “Learning to multitask,” in NIPS, 2018.  
[109] Y. Zhang and D.-Y. Yeung, “Multi-task learning in heterogeneous feature spaces,” in AAAI, 2011.  
[110] S. Han, X. Liao, and L. Carin, “Cross-domain multitask learning with latent probit models,” in ICML, 2012.  
[111] P. Yang, K. Huang, and C. Liu, “Geometry preserving multi-task metric learning,” MLJ, 2013.  
[112] N. Quadrianto, A. J. Smola, T. S. Caetano, S. V. N. Vishwanathan, and J. Petterson, “Multitask learning without label correspondences,” in NIPS, 2010.  
[113] B. Romera-Paredes, H. Aung, N. Bianchi-Berthouze, and M. Pontil, “Multilinear multitask learning,” in ICML, 2013.  
[114] K. Wimalawarne, M. Sugiyama, and R. Tomioka, “Multitask learning meets tensor factorization: Task imputation via convex optimization,” in NIPS, 2014.  
[115] O. Sener and V. Koltun, “Multi-task learning as multi-objective optimization,” in NeurIPS, 2018.  
[116] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” in NeurIPS, 2020.  
[117] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich, “Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks,” in ICML, 2018.  
[118] N. Parikh and S. P. Boyd, “Proximal algorithms,” Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127–239, 2014.  
[119] L. Zhao, Q. Sun, J. Ye, F. Chen, C. Lu, and N. Ramakrishnan, “Multitask learning for spatio-temporal event forecasting,” in KDD, 2015.  
[120] Y. Li, J. Wang, J. Ye, and C. K. Reddy, “A multi-task learning formulation for survival analysis,” in KDD, 2016.  
[121] L. Zhao, Q. Sun, J. Ye, F. Chen, C. Lu, and N. Ramakrishnan, “Feature constrained multi-task learning models for spatiotemporal event forecasting,” IEEE TKDE, 2017.  
[122] Q. Liu, X. Liao, and L. Carin, “Semi-supervised multitask learning,” in NIPS, 2007.  
[123] Q. Liu, X. Liao, H. Li, J. R. Stack, and L. Carin, “Semisupervised multitask learning,” IEEE TPAMI, 2009.  
[124] Y. Zhang and D. Yeung, “Semi-supervised multi-task regression,” in ECMLPKDD, 2009.  
[125] R. Reichart, K. Tomanek, U. Hahn, and A. Rappoport, “Multi-task active learning for linguistic annotations,” in ACL, 2008.  
[126] A. Acharya, R. J. Mooney, and J. Ghosh, “Active multitask learning using both latent and supervised shared topics,” in SDM, 2014.  
[127] M. Fang and D. Tao, “Active multi-task learning via bandits,” in SDM, 2015.  
[128] H. Li, X. Liao, and L. Carin, “Active learning for semi-supervised multitask learning,” in ICASSP, 2009.  
[129] K. Lin and J. Zhou, “Interactive multi-task relationship learning,” in ICDM, 2016.  
[130] A. Pentina and C. H. Lampert, “Multi-task learning with labeled and unlabeled tasks,” in ICML, 2017.  
[131] J. Zhang and C. Zhang, “Multitask Bregman clustering,” in AAAI, 2010.  
[132] X. Zhang and X. Zhang, “Smart multi-task Bregman clustering and multi-task kernel clustering,” in AAAI, 2013.  
[133] X. Zhang, X. Zhang, and H. Liu, “Smart multitask Bregman clustering and multitask kernel clustering,” ACM TKDD, 2015.  
[134] Q. Gu, Z. Li, and J. Han, “Learning a kernel for multi-task clustering,” in AAAI, 2011.  
[135] X. Zhang, “Convex discriminative multitask clustering,” IEEE TPAMI, 2015.  
[136] Y. Wang, D. P. Wipf, Q. Ling, W. Chen, and I. J. Wassell, “Multi-task learning for subspace segmentation,” in ICML, 2015.  
[137] X. Zhang, X. Zhang, and H. Liu, “Self-adapted multi-task clustering,” in IJCAI, 2016.  
[138] Y. Yang, Z. Ma, Y. Yang, F. Nie, and H. T. Shen, “Multitask spectral clustering by exploring intertask correlation,” IEEE TCYB, 2015.  
[139] X. Zhu, X. Li, S. Zhang, C. Ju, and X. Wu, “Robust joint graph sparse coding for unsupervised spectral feature selection,” IEEE TNNLS, 2017.  
[140] X. Zhang, X. Zhang, H. Liu, and J. Luo, “Multi-task clustering with model relation learning,” in IJCAI, 2018.  
[141] A. Wilson, A. Fern, S. Ray, and P. Tadepalli, “Multi-task reinforcement learning: A hierarchical Bayesian approach,” in ICML, 2007.  
[142] H. Li, X. Liao, and L. Carin, “Multi-task reinforcement learning in partially observable stochastic environments,” JMLR, 2009.  
[143] A. Lazaric and M. Ghavamzadeh, “Bayesian multi-task reinforcement learning,” in ICML, 2010.  
[144] D. Calandriello, A. Lazaric, and M. Restelli, “Sparse multi-task reinforcement learning,” in NIPS, 2014.  
[145] J. Andreas, D. Klein, and S. Levine, “Modular multitask reinforcement learning with policy sketches,” in ICML, 2017.  
[146] A. A. Deshmukh, U. Dogan, and C. Scott, “Multi-task learning for<sup>¨</sup> contextual bandits,” in NIPS, 2017.  
[147] A. M. Saxe, A. C. Earle, and B. Rosman, “Hierarchy through composition with multitask LMDPs,” in ICML, 2017.  
[148] T. Bram, G. Brunner, O. Richter, and R. Wattenhofer, “Attentive multi-¨ task deep reinforcement learning,” in ECMLPKDD, 2019.  
[149] T. Vuong, D. V. Nguyen, T. Nguyen, C. Bui, H. Kieu, V. Ta, Q. Tran, and T. H. Le, “Sharing experience in multitask reinforcement learning,” in IJCAI, 2019.  
[150] M. Igl, A. Gambardella, N. Nardelli, N. Siddharth, W. Bohmer, and¨ S. Whiteson, “Multitask soft option learning,” in UAI, 2020.  
[151] E. Parisotto, J. Ba, and R. Salakhutdinov, “Actor-mimic: Deep multitask and transfer reinforcement learning,” in ICLR, 2016.  
[152] A. A. Rusu, S. G. Colmenarejo, C¸ . Gulc¸ehre, G. Desjardins, J. Kirk-¨ patrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” in ICLR, 2016.  
[153] S. Omidshafiei, J. Pazis, C. Amato, J. P. How, and J. Vian, “Deep decentralized multi-task multi-agent reinforcement learning under partial observability,” in ICML, 2017.  
[154] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu, “Distral: Robust multitask reinforcement learning,” in NIPS, 2017.  
[155] S. E. Bsat, H. Bou-Ammar, and M. E. Taylor, “Scalable multitask policy gradient reinforcement learning,” in AAAI, 2017.  
[156] S. Sharma and B. Ravindran, “Online multi-task learning using active sampling,” in ICLR Workshop, 2017.  
[157] S. Sharma, A. K. Jha, P. Hegde, and B. Ravindran, “Learning to multitask by active sampling,” in ICLR, 2018.  
[158] L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu, “IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures,” in ICML, 2018.  
[159] R. Tutunov, D. Kim, and H. Bou-Ammar, “Distributed multitask reinforcement learning with quadratic convergence,” in NeurIPS, 2018.  
[160] X. Lin, H. S. Baweja, G. Kantor, and D. Held, “Adaptive auxiliary task weighting for reinforcement learning,” in NeurIPS, 2019.  
[161] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Sharing knowledge in multi-task deep reinforcement learning,” in ICLR, 2020.  
[162] T. Shu, C. Xiong, and R. Socher, “Hierarchical and interpretable skill acquisition in multi-task reinforcement learning,” in ICLR, 2018.  
[163] M. Hessel, H. Soyer, L. Espeholt, W. Czarnecki, S. Schmitt, and H. van Hasselt, “Multi-task deep reinforcement learning with PopArt,” in AAAI, 2019.  
[164] Z. Guo, B. A. Pires, M. G. Azar, B. Piot, F. Altche, J.-B. Grill, and´ R. Munos, “Bootstrap latent-predictive representations for multitask reinforcement learning,” in ICML, 2020.  
[165] J. He and R. Lawrence, “A graph-based framework for multi-task multiview learning,” in ICML, 2011.  
[166] J. Zhang and J. Huan, “Inductive multi-task learning with multiple view data,” in KDD, 2012.  
[167] X. Zhang, X. Zhang, and H. Liu, “Multi-task multi-view clustering for non-negative data,” in IJCAI, 2015.  
[168] X. Zhang, X. Zhang, H. Liu, and X. Liu, “Multi-task multi-view clustering,” IEEE TKDE, 2016.  
[169] X. Zhu, X. Li, and S. Zhang, “Block-row sparse multiview multilabel learning for image classification,” IEEE TCYB, 2016.  
[170] L. Zheng, Y. Cheng, and J. He, “Deep multimodality model for multitask multi-view learning,” in SDM, 2019.  
[171] A. Niculescu-Mizil and R. Caruana, “Inductive transfer for Bayesian network structure learning,” in AISTATS, 2007.  
[172] J. Honorio and D. Samaras, “Multi-task learning of Gaussian graphical models,” in ICML, 2010.  
[173] D. Oyen and T. Lane, “Leveraging domain knowledge in multitask Bayesian network structure learning,” in AAAI, 2012.  
[174] K. Lin, J. Xu, I. M. Baytas, S. Ji, and J. Zhou, “Multi-task feature interaction learning,” in KDD, 2016.  
[175] Y. Zhang, “Parallel multi-task learning,” in ICDM, 2015.  
[176] O. Dekel, P. M. Long, and Y. Singer, “Online multitask learning,” in COLT, 2006.  
[177] O. Dekel, P. M. Long, and Y. Singer, “Online learning of multiple tasks with a shared loss,” JMLR, 2007.  
[178] G. Lugosi, O. Papaspiliopoulos, and G. Stoltz, “Online multi-task learning with hard constraints,” in COLT, 2009.  
[179] G. Cavallanti, N. Cesa-Bianchi, and C. Gentile, “Linear algorithms for online multitask classification,” JMLR, 2010.  
[180] G. Pillonetto, F. Dinuzzo, and G. D. Nicolao, “Bayesian online multitask learning of Gaussian processes,” IEEE TPAMI, 2010.  
[181] A. Saha, P. Rai, H. Daume III, and S. Venkatasubramanian, “Online´ learning of multiple tasks and their relationships,” in AISTATS, 2011.  
[182] K. Murugesan, H. Liu, J. G. Carbonell, and Y. Yang, “Adaptive smoothed online multi-task learning,” in NIPS, 2016.  
[183] P. Yang, P. Zhao, and X. Gao, “Robust online multi-task learning with correlative and personalized structures,” IEEE TKDE, 2017.  
[184] S. Hao, P. Zhao, Y. Liu, S. C. H. Hoi, and C. Miao, “Online multitask relative similarity learning,” in IJCAI, 2017.  
[185] P. Yang, P. Zhao, J. Zhou, and X. Gao, “Confidence weighted multitask learning,” in AAAI, 2019.  
[186] J. Wang, M. Kolar, and N. Srebro, “Distributed multi-task learning,” in AISTATS, 2016.  
[187] S. Liu, S. J. Pan, and Q. Ho, “Distributed multi-task relationship learning,” in KDD, 2017.  
[188] L. Xie, I. M. Baytas, K. Lin, and J. Zhou, “Privacy-preserving distributed multi-task learning with asynchronous updates,” in KDD, 2017.  
[189] V. Smith, C. Chiang, M. Sanjabi, and A. S. Talwalkar, “Federated multitask learning,” in NIPS, 2017.  
[190] C. Zhang, P. Zhao, S. Hao, Y. C. Soh, B. Lee, C. Miao, and S. C. H. Hoi, “Distributed multi-task classification: A decentralized online learning approach,” MLJ, 2018.  
[191] K. Q. Weinberger, A. Dasgupta, J. Langford, A. J. Smola, and J. Attenberg, “Feature hashing for large scale multitask learning,” in ICML, 2009.  
[192] T. Zhang, B. Ghanem, S. Liu, and N. Ahuja, “Robust visual tracking via multi-task sparse learning,” in CVPR, 2012.  
[193] T. Zhang, B. Ghanem, S. Liu, and N. Ahuja, “Robust visual tracking via structured multi-task sparse learning,” IJCV, 2013.  
[194] Q. Xu, S. J. Pan, H. H. Xue, and Q. Yang, “Multitask learning for protein subcellular location prediction,” IEEE/ACM TCBB, 2011.  
[195] Z. Wu, C. Valentini-Botinhao, O. Watts, and S. King, “Deep neural networks employing multi-task learning and stacked bottleneck features for speech synthesis,” in ICASSP, 2015.  
[196] Q. Hu, Z. Wu, K. Richmond, J. Yamagishi, Y. Stylianou, and R. Maia, “Fusion of multiple parameterisations for DNN-based sinusoidal speech synthesis with multi-task learning,” in InterSpeech, 2015.  
[197] J. Bai, K. Zhou, G. Xue, H. Zha, G. Sun, B. L. Tseng, Z. Zheng, and Y. Chang, “Multi-task learning for learning to rank in web search,” in CIKM, 2009.  
[198] J. Ghosn and Y. Bengio, “Multi-task learning for stock selection,” in NIPS, 1996.  
[199] C. Yuan, W. Hu, G. Tian, S. Yang, and H. Wang, “Multi-task sparse learning with Beta process prior for action recognition,” in CVPR, 2013.  
[200] Y. Qi, O. Tastan, J. G. Carbonell, J. Klein-Seetharaman, and J. Weston, “Semi-supervised multi-task learning for predicting interactions between HIV-1 and human proteins,” Bioinformatics, 2010.  
[201] P. Bell and S. Renals, “Regularization of context-dependent deep neural networks with context-independent multi-task training,” in ICASSP, 2015.  
[202] Z. Chen, S. Watanabe, H. Erdogan, and J. R. Hershey, “Speech enhancement and recognition using multi-task learning of long short-term memory recurrent neural networks,” in InterSpeech, 2015.  
[203] V. W. Zheng, S. J. Pan, Q. Yang, and J. J. Pan, “Transferring multidevice localization models using latent multi-task learning,” in AAAI, 2008.  
[204] R. Collobert and J. Weston, “A unified architecture for natural language processing: Deep neural networks with multitask learning,” in ICML, 2008.  
[205] M. Lapin, B. Schiele, and M. Hein, “Scalable multitask representation learning for scene classification,” in CVPR, 2014.  
[206] A. H. Abdulnabi, G. Wang, J. Lu, and K. Jia, “Multi-task CNN model for attribute prediction,” IEEE TMM, 2015.  
[207] M. Luong, Q. V. Le, I. Sutskever, O. Vinyals, and L. Kaiser, “Multi-task sequence to sequence learning,” in ICLR, 2016.  
[208] J. Yim, H. Jung, B. Yoo, C. Choi, D. Park, and J. Kim, “Rotating your face using multi-task deep neural network,” in CVPR, 2015.  
[209] X. Chu, W. Ouyang, W. Yang, and X. Wang, “Multi-task recurrent neural network for immediacy prediction,” in ICCV, 2015.  
[210] X. Wang, C. Zhang, and Z. Zhang, “Boosted multi-task learning for face verification with applications to web image and video search,” in CVPR, 2009.  
[211] X. Yuan and S. Yan, “Visual classification with multi-task joint sparse representation,” in CVPR, 2010.  
[212] Q. Liu, Q. Xu, V. W. Zheng, H. Xue, Z. Cao, and Q. Yang, “Multitask learning for cross-platform siRNA efficacy prediction: An in-silico study,” BMC Bioinformatics, 2010.  
[213] A. Ahmed, M. Aly, A. Das, A. J. Smola, and T. Anastasakos, “Webscale multi-task feature selection for behavioral targeting,” in CIKM, 2012.  
[214] H. Wang, F. Nie, H. Huang, S. L. Risacher, C. H. Q. Ding, A. J. Saykin, and L. Shen, “Sparse multi-task regression and feature selection to identify brain imaging predictors for memory performance,” in ICCV, 2011.  
[215] K. Puniyani, S. Kim, and E. P. Xing, “Multi-population GWA mapping via multi-task regularized regression,” Bioinformatics, 2010.  
[216] J. Zhou, L. Yuan, J. Liu, and J. Ye, “A multi-task learning formulation for predicting disease progression,” in KDD, 2011.  
[217] J. Wan, Z. Zhang, J. Yan, T. Li, B. D. Rao, S. Fang, S. Kim, S. L. Risacher, A. J. Saykin, and L. Shen, “Sparse Bayesian multi-task learning for predicting cognitive outcomes from neuroimaging measures in Alzheimer’s disease,” in CVPR, 2012.  
[218] D. He, D. Kuhn, and L. Parida, “Novel applications of multitask learning and multiple output regression to multiple genetic trait prediction,” Bioinformatics, 2016.  
[219] K. Zhang, J. W. Gray, and B. Parvin, “Sparse multitask regression for identifying common mechanism of response to therapeutic targets,” Bioinformatics, 2010.  
[220] B. Cheng, G. Liu, J. Wang, Z. Huang, and S. Yan, “Multi-task low-rank affinity pursuit for image segmentation,” in ICCV, 2011.  
[221] J. Xu, P. Tan, L. Luo, and J. Zhou, “GSpartan: a geospatio-temporal multi-task learning framework for multi-location prediction,” in SDM, 2016.  
[222] C. Lang, G. Liu, J. Yu, and S. Yan, “Saliency detection by multitask sparsity pursuit,” IEEE TIP, 2012.  
[223] H. Wang, F. Nie, H. Huang, J. Yan, S. Kim, S. L. Risacher, A. J. Saykin, and L. Shen, “High-order multi-task feature learning to identify longitudinal phenotypic markers for Alzheimer’s disease progression prediction,” in NIPS, 2012.  
[224] Q. An, C. Wang, I. Shterev, E. Wang, L. Carin, and D. B. Dunson, “Hierarchical kernel stick-breaking process for multi-task image analysis,” in ICML, 2008.  
[225] M. Alamgir, M. Grosse-Wentrup, and Y. Altun, “Multitask learning for brain-computer interfaces,” in AISTATS, 2010.  
[226] Y. Zhang and D.-Y. Yeung, “Multi-task warped Gaussian process for personalized age estimation,” in CVPR, 2010.  
[227] J. Xu, J. Zhou, and P. Tan, “FORMULA: FactORized MUlti-task LeArning for task discovery in personalized medical models,” in SDM, 2015.  
[228] T. R. Almaev, B. Mart´ınez, and M. F. Valstar, “Learning to transfer: Transferring latent task structures and its application to person-specific facial action unit detection,” in ICCV, 2015.  
[229] A. Liu, Y. Su, W. Nie, and M. S. Kankanhalli, “Hierarchical clustering multi-task learning for joint human action grouping and recognition,” IEEE TPAMI, 2017.  
[230] F. Wu and Y. Huang, “Collaborative multi-domain sentiment classification,” in ICDM, 2015.  
[231] O. Chapelle, P. K. Shivaswamy, S. Vadrevu, K. Q. Weinberger, Y. Zhang, and B. L. Tseng, “Multi-task learning for boosting with application to web search ranking,” in KDD, 2010.  
[232] C. Widmer, J. Leiva, Y. Altun, and G. Ratsch, “Leveraging sequence¨ classification by taxonomy-based multitask learning,” in RECOMB, 2010.  
[233] Y. Zhang, B. Cao, and D.-Y. Yeung, “Multi-domain collaborative filtering,” in UAI, 2010.  
[234] K. M. A. Chai, C. K. I. Williams, S. Klanke, and S. Vijayakumar, “Multi-task Gaussian process learning of robot inverse dynamics,” in NIPS, 2008.  
[235] D.-Y. Yeung and Y. Zhang, “Learning inverse dynamics by Gaussian process regression under the multi-task learning framework,” in The Path to Autonomous Robots, Springer, 2009.  
[236] C. Widmer, N. C. Toussaint, Y. Altun, and G. Ratsch, “Inferring latent¨ task structure for multitask learning by multiple kernel learning,” BMC Bioinformatics, 2010.  
[237] A. Ahmed, A. Das, and A. J. Smola, “Scalable hierarchical multitask learning algorithms for conversion optimization in display advertising,” in WSDM, 2014.  
[238] J. Zheng and L. M. Ni, “Time-dependent trajectory regression on road networks via multi-task learning,” in AAAI, 2013.  
[239] X. Lu, Y. Wang, X. Zhou, Z. Zhang, and Z. Ling, “Traffic sign recognition via multi-modal tree-structure embedded multi-task learning,” IEEE TITS, 2017.  
[240] F. Mordelet and J. Vert, “ProDiGe: Prioritization of disease genes with multitask machine learning from positive and unlabeled examples,” BMC Bioinformatics, 2011.  
[241] J. Xu, P. Tan, J. Zhou, and L. Luo, “Online multi-task learning framework for ensemble forecasting,” IEEE TKDE, 2017.  
[242] M. Kshirsagar, J. G. Carbonell, and J. Klein-Seetharaman, “Multitask learning for host-pathogen protein interactions,” Bioinformatics, 2013.  
[243] L. Han, L. Li, F. Wen, L. Zhong, T. Zhang, and X. Wan, “Graph-guided multi-task sparse learning model: a method for identifying antigenic variants of influenza A(H3N2) virus,” Bioinformatics, 2019.  
[244] Z. Hong, X. Mei, D. V. Prokhorov, and D. Tao, “Tracking via robust multi-task multi-view joint sparse representation,” in ICCV, 2013.  
[245] Y. Yan, E. Ricci, S. Ramanathan, O. Lanz, and N. Sebe, “No matter where you are: Flexible graph-guided multi-task learning for multi-view head pose classification under target motion,” in ICCV, 2013.  
[246] M. Kshirsagar, K. Murugesan, J. G. Carbonell, and J. Klein-Seetharaman, “Multitask matrix completion for learning protein interactions across diseases,” Journal of Computational Biology, 2017.  
[247] C. Su, F. Yang, S. Zhang, Q. Tian, L. S. Davis, and W. Gao, “Multi-task learning with low rank attribute embedding for person re-identification,” in ICCV, 2015.  
[248] Y. Zhang and Q. Yang, “A survey on multi-task learning,” CoRR, 2017.  
[249] J. Baxter, “Learning internal representations,” in COLT, 1995.  
[250] J. Baxter, “A model of inductive bias learning,” JAIR, 2000.  
[251] A. Maurer, “Bounds for linear multi-task learning,” JMLR, 2006.  
[252] A. Maurer, “The Rademacher complexity of linear transformation classes,” in COLT, 2006.  
[253] B. Juba, “Estimating relatedness via data compression,” in ICML, 2006.  
[254] S. Ben-David and R. S. Borbely, “A notion of task relatedness yielding provable multiple-task learning guarantees,” MLJ, 2008.  
[255] S. M. Kakade, S. Shalev-Shwartz, and A. Tewari, “Regularization techniques for learning with matrices,” JMLR, 2012.  
[256] M. Pontil and A. Maurer, “Excess risk bounds for multitask learning with trace norm regularization,” in COLT, 2013.  
[257] A. Pentina and S. Ben-David, “Multi-task and lifelong learning of kernels,” in ALT, 2015.  
[258] Y. Zhang, “Multi-task learning and algorithmic stability,” in AAAI, 2015.  
[259] A. Maurer, M. Pontil, and B. Romera-Paredes, “The benefit of multitask representation learning,” JMLR, 2016.  
[260] N. Yousefi, Y. Lei, M. Kloft, M. Mollaghasemi, and G. Anagnastapolous, “Local Rademacher complexity-based learning guarantees for multi-task learning,” JMLR, 2018.  
[261] A. Argyriou, C. A. Micchelli, and M. Pontil, “When is there a representer theorem? vector versus matrix regularizers,” JMLR, 2009.  
[262] A. Argyriou, C. A. Micchelli, and M. Pontil, “On spectral learning,” JMLR, 2010.  
[263] K. Lounici, M. Pontil, A. B. Tsybakov, and S. A. van de Geer, “Taking advantage of sparsity in multi-task learning,” in COLT, 2009.  
[264] G. Obozinski, M. J. Wainwright, and M. I. Jordan, “Support union recovery in high-dimensional multivariate regression,” Annals of Statistics, 2011.  
[265] M. Kolar, J. D. Lafferty, and L. A. Wasserman, “Union support recovery in multi-task learning,” JMLR, 2011.