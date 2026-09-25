# MetaWeighting: Learning to Weight Tasks in Multi-Task Learning

Yuren Mao<sup>1</sup>, Zekai Wang<sup>2</sup>, Weiwei Liu<sup>2</sup>∗, Xuemin Lin<sup>1</sup>, Pengtao Xie<sup>3</sup>

<sup>1</sup>School of Computer Science and Engineering, University of New South Wales <sup>2</sup>School of Computer Science, Wuhan University

<sup>3</sup>Department of Electrical and Computer Engineering, University of California San Diego

yuren.mao@unsw.edu.au, {wzekai99,liuweiwei863}@gmail.com lxue@cse.unsw.edu.au, pengtaoxie2008@gmail.com

## Abstract

Task weighting, which assigns weights on the including tasks during training, significantly matters the performance of Multi-task Learning (MTL); thus, recently, there has been an explosive interest in it. However, existing task weighting methods assign weights only based on the training loss, while ignoring the gap between the training loss and generalization loss. It degenerates MTL’s performance. To address this issue, the present paper proposes a novel task weighting algorithm, which automatically weights the tasks via a learning-to-learn paradigm, referred to as MetaWeighting. Extensive experiments are conducted to validate the superiority of our proposed method in multitask text classification.

## 1 Introduction

Multi-task Learning (MTL) simultaneously learns multiple related tasks and aims to achieve better performance than learning each task independently (Caruana, 1993; Baxter, 2000). It has achieved great success in various applications; especially, in the text classification context, MTL can significantly outperform single task learning (Liu et al., 2017; Mao et al., 2021).

In MTL, it is common for the including tasks to be competing. If we cannot properly balance these tasks, some tasks might dominate the training process and hurt the performance of other tasks, a phenomenon known as task imbalance. To address the task imbalance, the most widely used method is task weighting, which adaptively assigns weights on the tasks during training to balance their impacts. Various task weighting methods have been proposed and can be used in multi-task text classification, such as (Kendall et al., 2018; Sener and Koltun, 2018; Chen et al., 2018).

However, existing task weighting methods compute the task weights only based on training losses or corresponding gradients. They ignore the gap between the training loss and generalization loss. To illustrate this gap, we report observations of our four-task topic classification experiment in Figure 1. The detailed experimental settings are introduced in the experiment section. Figure 1 demonstrates that the training losses and generalization losses (estimated by the test losses) have different magnitudes; moreover, they have different patterns, such as a task might have the largest training loss but the lowest generalization loss among the tasks.

![](images/54cf2ecadac2025faaa593f46dea95d1af70a941aafb1b2ce344d698989bfe30.jpg)

<details>
<summary>radar</summary>

| Task | Epoch | Training Loss | Generalization Loss |
| --- | --- | --- | --- |
| task 1 | 500th epoch | ~0.5 | ~0.5 |
| task 1 | 1000th epoch | ~0.2 | ~0.4 |
| task 1 | 1500th epoch | ~0.1 | ~0.3 |
| task 2 | 500th epoch | ~0.4 | ~0.4 |
| task 2 | 1000th epoch | ~0.2 | ~0.3 |
| task 2 | 1500th epoch | ~0.1 | ~0.2 |
| task 3 | 500th epoch | ~0.3 | ~0.4 |
| task 3 | 1000th epoch | ~0.2 | ~0.3 |
| task 3 | 1500th epoch | ~0.1 | ~0.2 |
| task 4 | 500th epoch | ~0.4 | ~0.5 |
| task 4 | 1000th epoch | ~0.2 | ~0.3 |
| task 4 | 1500th epoch | ~0.1 | ~0.2 |
</details>

Figure 1: Illustration of the gap between training loss and generalization loss in the training process of a fourtask topic classification experiment (500<sup>th</sup> , 1000<sup>th</sup>, 1500<sup>th</sup> epochs respectively).

This gap causes a mismatch between the task weights and tasks’ generalization performance, which reduces effectiveness of the task weighting. To tackle this issue, this paper proposes a novel task weighting method based on a bi-level optimization problem, which aims to find task weights that explicitly optimize the generalization performance. Our proposed method computes task weights by solving this bi-level optimization problem and performs in a learning-to-learn manner; thus, dubbed MetaWeighting. MetaWeighting can improve the performance of multi-task text classification.

To verify our theoretical analysis and validate the superiority of MetaWeighting, we conduct experiments on two classical text classification problems: sentiment analysis (on reviews) and topic classification (on news). The results demonstrate that MetaWeighting outperforms several state-of-the-art multi-task text classification methods.

## 2 Related Works

Existing task weighting strategies can be divided into two categories: weight adaptation methods and Pareto Optimization (PO)-based methods. The weight adaptation methods adaptively adjust the tasks’ weights during training based on pre-defined heuristic, such as uncertainty (Kendall et al., 2018), task difficulty prioritization (Guo et al., 2018), gradient normalization (Chen et al., 2018), weight average (Liu et al., 2019) and task variance regularization (Mao et al., 2021). These methods only use training losses or their gradients to compute task weights while ignores the gap between the training loss and generalization loss.

Besides, the PO-based methods formulate MTL as a multi-objective optimization problem and aim to find an arbitrary Pareto stationary solution (Sener and Koltun, 2018; Lin et al., 2019; Mahapatra and Rajan, 2020; Lin et al., 2020; Ma et al., 2020; Mao et al., 2020). However, in these methods, the learning objectives only involve training losses; thus, they can only achieve Pareto stationary points w.r.t training losses. They also ignore the gap between the training loss and generalization loss. Moreover, (Lin et al., 2019) proposes that the PO-based methods can be also regarded as weight adaptation methods for they optimize the weighted sum of training losses as well.

Overlooking the gap between the training loss and generalization loss would degenerate the performance of MTL. This paper proposes a novel meta learning-based task weighting method to solve this issue. There are some works adopt meta learning-based weighting methods in multilingual learning, e.g., (Wang et al., 2020) and (Tarunesh et al., 2021). However, these works cannot solve multi-objective optimization problems. By contrast, this paper proposes a novel method which can solve multi-objective optimization problems.

## 3 Preliminaries

Consider a multi-task learning problem with $T$ tasks over an input space $\mathcal { X }$ and a collection of task spaces $\{ \mathcal { V } _ { t } \} _ { t = 1 } ^ { T }$ . For each task, we have a set of i.i.d. training samples $D _ { t } \ = \ \{ x _ { t } ^ { i } , y _ { t } ^ { i } \} _ { i = 1 } ^ { n }$ The training samples are sampled from an identical distribution $\mathcal { P } _ { t }$ . Based on the training sets $\{ D _ { t } \} _ { t = 1 } ^ { T }$ , we learn an MTL model from a parameterized hypothesis class , which shares some parameters across tasks. Let $\theta _ { s }$ represent the parameters shared between tasks (task-sharing parameters), while $\theta _ { t }$ represent the task-specific parameters. $h ( \cdot , \theta _ { s } , \theta _ { 1 } , . . . , \theta _ { T } ) : \mathcal { X } \to \{ \mathcal { V } _ { t } \} _ { t = 1 } ^ { T } \in \mathcal { H }$ denotes an MTL model that learns from $\mathcal { H } .$ , while $h ( \cdot , \theta _ { s } , \theta _ { t } ) : \mathcal { X }  \mathcal { Y } _ { t }$ denotes the task-specific module in the MTL model.

The loss function is represented by $l ( \cdot , \cdot ) ~ :$ $\mathcal { V } ^ { t } \times \mathcal { Y } ^ { t }  [ 0 , 1 ] ^ { T }$ . For each task, the generalization loss is $\mathscr { L } _ { t } ( \theta ) = \mathbb { E } _ { ( x _ { t } , y _ { t } ) \sim \mathcal { P } _ { t } } l ( h ( x _ { t } , \theta _ { s } , \theta _ { t } ) , y _ { t } )$ and the training loss is defined as $\mathcal { L } _ { t } ^ { t r } ( \theta , D _ { t } ) =$ $\begin{array} { r } { \frac { 1 } { | D _ { t } | } \sum _ { ( x _ { t } , y _ { t } ) \in D _ { t } } l ( h ( x _ { t } , \theta _ { s } , \theta _ { t } ) , y _ { t } ) } \end{array}$ . In this paper, -each training set $D _ { t }$ is randomly divided into two subsets: support set $D _ { t } ^ { s }$ and query set $D _ { t } ^ { q }$ . Correspondingly; moreover, the support loss is defined as $\begin{array} { r } { \mathcal { L } _ { t } ^ { s } ( \theta , D _ { t } ^ { s } ) = \frac { 1 } { | D _ { t } ^ { s } | } \sum _ { ( x _ { t } , y _ { t } ) \in D _ { t } ^ { s } } l ( h ( x _ { t } , \theta _ { s } , \theta _ { t } ) , y _ { t } ) } \end{array}$ and the query loss is defined as $\begin{array} { r l } { \mathcal { L } _ { t } ^ { q } ( \theta , D _ { t } ^ { q } ) } & { { } = } \end{array}$ $\frac { 1 } { | D _ { t } ^ { q } | } \sum _ { ( x _ { t } , y _ { t } ) \in D _ { t } ^ { q } } l \big ( h ( x _ { t } , \theta _ { s } , \theta _ { t } ) , y _ { t } \big )$

## 3.1 Hypergradient Descent

Hypergradient Descent (HD) (Almeida et al., 1998; Baydin et al., 2018) provides an efficient way to apply gradient descent on hyper-parameters. Here, we take learning rate’s HD as an example to introduce the basic form of HD. Given an objective function $f ( \theta )$ and previous parameters $\theta ^ { k - 1 }$ , gradient descent-based learning typically evaluates the gradient $\nabla f ( \theta ^ { k - 1 } )$ and moves against it to arrive at updated parameters

$$
\theta^ {k} = \theta^ {k - 1} - \eta \nabla f (\theta^ {k - 1}), \tag {1}
$$

where $\eta$ is the learning rate. HD derives an update rule for the learning rate η itself. Based on Eq. (1) and the chain rule, we have

$$
\begin{array}{l} \frac {\partial f \left(\theta^ {k}\right)}{\partial \eta} = \nabla f \left(\theta^ {k}\right) \cdot \frac {\partial \left(\theta^ {k - 1} - \eta \nabla f \left(\theta^ {k - 1}\right)\right)}{\partial \eta} \tag {2} \\ = \nabla f (\theta^ {k}) \cdot (- \nabla f (\dot {\theta} ^ {k - 1})), \\ \end{array}
$$

with which we construct a update rule for $\eta \colon$

$$
\eta^ {k + 1} = \eta^ {k} + \beta \nabla f (\theta^ {k}) \cdot \nabla f (\theta^ {k - 1}), \tag {3}
$$

introducing $\beta$ as the hypergradient step size. In this paper, we extend HD to a bi-level multi-objective optimization problem.

## 3.2 Common Descent Direction for Multiple Objectives

When using gradient descent to jointly optimize multiple optimization objectives, we need to find a descent direction common to all the objectives. Based on the descent direction for each objective, (Désidéri, 2012) proposes a way to obtain the common descent direction, as in Theorem 1. This paper proposes a method to simultaneously optimize the tasks’ generalization loss based on Theorem 1.

Theorem 1 ((Désidéri, 2012)). Let be a Hilbert space offinite or infinite dimension N. Let $f _ { i } ( z )$ $( 1 \leq i \leq n \leq N )$ be n smoothfunctions ofthe vector z . and $z ^ { 0 }$ a particular admissible designpoint, at which the gradient-vectors are denoted $g _ { i } = \nabla f _ { i } ( z ^ { 0 } )$ , and

$$
\mathcal {U} = \{a \in \mathcal {A} | a = \sum_ {i = 1} ^ {n} \lambda_ {i} g _ {i}; \lambda_ {i} > 0 (\forall i); \sum_ {i = 1} ^ {n} \lambda_ {i} = 1 \}. \tag {4}
$$

Let $\begin{array} { r } { a ^ { * } = \arg \operatorname* { m i n } _ { a \in \bar { \mathcal { U } } } \| \ a \ \| , } \end{array}$ , where consists of the convex hull and closure of . Then, $i f a ^ { * } \neq 0 ,$ $a ^ { * }$ is a descent direction common to all the objectives.

## 4 MetaWeighting for MTL

In this section, we demonstrate the gap between existing task weighting strategies and the generalization performance of MTL in Section 4.1. This gap motivates us to proposed a MetaWeighting problem, which aims to automatically learn a task weighting strategy that can narrow this gap, in Section 4.2. Moreover, we propose an algorithm to solve the MetaWeighting problem in Section 4.3.

## 4.1 Gap Between Task Weighting and Generalization Performance

MTL aims to improve the generalization performance of all the including tasks, which can be formulated via the following optimization problem.

$$
\min _ {\theta} \mathbf {L} (\theta) = (\mathcal {L} _ {1} (\theta),..., \mathcal {L} _ {T} (\theta)) ^ {\top}. \tag {5}
$$

By contrast, existing task weighting strategies train an MTL model via the following objective.

$$
\min _ {\theta} \frac {1}{T} w _ {t} \mathcal {L} _ {t} ^ {t r} (\theta , D _ {t}), \tag {6}
$$

where the $w _ { t }$ is adaptive during training and only depends on the training losses or their gradients. $\mathbf { A } \mathbf { s }$ the neural networks are usually heavily overparameterized (Allen-Zhu et al., 2019), the training losses cannot properly estimate the generalization losses. Thus, existing task weighting strategies, which tunes weights only based on the training losses, overlook the generalization losses. Obviously, there is a gap between these task weighting strategies and the generalization performance of MTL.

## 4.2 MetaWeighting Problem

To narrow the gap between task weighting strategies and generalization performance, we propose to automatically learn task weights that can reduce the generalization losses, namely learning to weight. This learning to weight problem is formlated via the following bi-level optimization problem, dubbed MetaWeighting.

## Problem 1.

$$
\begin{array}{l} \min _ {\boldsymbol {w}} \left(\mathcal {L} _ {1} (\theta^ {*} (\boldsymbol {w})),..., \mathcal {L} _ {T} (\theta^ {*} (\boldsymbol {w}))\right) ^ {\top} \\ s. t. \theta^ {*} (\boldsymbol {w}) = \arg \min _ {\theta} \frac {1}{T} \sum_ {t = 1} ^ {T} w _ {t} \mathcal {L} _ {t} ^ {t r} (\theta , D _ {t}) \tag {7} \\ \end{array}
$$

where $\mathbf { w } = ( w _ { 1 } , w _ { 2 } , . . . , w _ { T } )$ . This bi-level optimization problem combines (5) and (6) together, by solving which we can obtain task weights that benefit the generalization performance of MTL.

However, the generalization loss is agnostic. To properly estimate the generalization loss, we randomly divide the training set $D _ { t }$ into two subsets: support set $D _ { t } ^ { s }$ and query set $D _ { t } ^ { q }$ , where $D _ { t } ^ { s }$ is used to train an MTL model, and $D _ { t } ^ { q }$ is used to estimate generalization loss of the MTL model. In Section 5, we theoretically demonstrate that query loss is a good estimator for the generalization loss; besides, in Section $6 . 7 ,$ experimental analysis also supports that query loss is a good estimator.

Based on the support-query split, the MetaWeighting problem is transformed into the following form.

## Problem 2.

$$
\begin{array}{l} \min _ {\boldsymbol {w}} \left(\mathcal {L} _ {1} ^ {q} (\theta^ {*} (\boldsymbol {w}), D _ {1} ^ {q}), \dots , \mathcal {L} _ {T} ^ {q} (\theta^ {*} (\boldsymbol {w}), D _ {T} ^ {q})\right) ^ {\top} \\ s. t. \quad \theta^ {*} (\boldsymbol {w}) = \arg \min _ {\theta} \frac {1}{T} \sum_ {t = 1} ^ {T} w _ {t} \mathcal {L} _ {t} ^ {s} (\theta , D _ {t} ^ {s}) \tag {8} \\ \end{array}
$$

## 4.3 MetaWeighting Algorithm

In the MetaWeighting problem, the inner optimization objective is embedded within the outer optimization objective. In MTL, the inner optimization objective is to minimize the weighted sum of task-specific training losses, which is typically optimized by means of iterative gradient descent; thus, Problem 2 can be formulated by the following problem in the $k ^ { t h }$ learning iteration.

## Problem 3.

$$
\min _ {\boldsymbol {w}} \left(\mathcal {L} _ {1} ^ {q} (\theta^ {k}, D _ {1} ^ {q}),..., \mathcal {L} _ {T} ^ {q} (\theta^ {k}, D _ {T} ^ {q})\right) ^ {\top}
$$

$$
s. t. \quad \theta^ {k} = \theta^ {k - 1} - \frac {\eta}{T} \sum_ {t = 1} ^ {T} w _ {t} \nabla_ {\theta} \mathcal {L} _ {t} ^ {s} (\theta^ {k - 1}, D _ {t} ^ {s}) \tag {9}
$$

To solve Problem 3, we adopt the Hypergradient Descent (HD) method. However, the original HD method (Almeida et al., 1998; Baydin et al., 2018) is proposed for single objective optimization, which can not used in our problem where a multiobjective optimization problem involves. In this section, this paper proposes a novel HD method for the multi-objective optimization setting, as in the following sections.

## 4.3.1 Task-Specific Descent Direction

The learning objective of Problem 3 involves $T$ objectives. We aim to find a gradient direction, moving against which all the objective can be optimized. To find this gradient direction, we first find the hypergradient direction w.r.t w (denoted as $d _ { t } )$ for each task. $d _ { t }$ is computed by the following equation.

$$
\begin{array}{l} d _ {t} = \frac {\partial \mathcal {L} _ {t} ^ {q} \left(\theta^ {k} , D _ {t} ^ {q}\right)}{\partial \mathbf {w}} = \nabla_ {\theta} \mathcal {L} _ {t} ^ {q} \left(\theta^ {k}, D _ {t} ^ {q}\right) \cdot \frac {\partial \theta^ {k}}{\partial \mathbf {w}} \tag {10} \\ = - \frac {\eta}{T} \nabla_ {\theta} \mathcal {L} _ {t} ^ {q} (\theta^ {k}, D _ {t} ^ {q}) \nabla_ {\theta} \mathbf {L} ^ {s} (\theta^ {k - 1}, D ^ {s}). \\ \end{array}
$$

where L<sup>s</sup>(θ<sup>k</sup>−<sup>1</sup>, D<sup>s</sup>) 二 ( <sub>θ</sub> <sup>s</sup>(θ<sup>k</sup>−<sup>1</sup>, D<sup>s</sup>), ..., <sub>θ</sub> <sup>s</sup> (θ<sup>k</sup>−<sup>1</sup>, D<sup>s</sup> )).

Moving against $d _ { t }$ , the generalization loss of task t can be optimized.

## 4.3.2 Common Descent Direction

Base on $d _ { t }$ , in this section, we find a common gradient direction, moving against which all the objective can be optimized. Let $\mathbf { d } = ( d _ { 1 } ^ { \top } , d _ { 2 } ^ { \top } , . . . , d _ { T } ^ { \top } )$ and $d _ { c }$ be the common gradient direction. Theorem 1 presents that the following Eq. (11) is a common descent direction.

$$
d _ {c} = \lambda^ {*} \mathbf {d} ^ {\top} \tag {11}
$$

where

$$
\lambda^ {*} = \arg \min _ {\lambda} \{\| \lambda \mathbf {d} ^ {\top} \| _ {2} ^ {2} | \lambda \mathbf {1} ^ {\top} = 1, \lambda \succeq \mathbf {0} \}, \tag {12}
$$

where $\mathbf { 1 } = ( 1 , 1 , . . . , 1 )$ . Eq. (12) is a typical minimum Euclidean-norm point problem. We here adopt the widely used Frank-Wolfe optimization algorithm (Jaggi, 2013), a minimum-norm-point algorithm, to solve it. The Frank-Wolfe optimization algorithm is presented in Algorithm 2.

Algorithm 1: MetaWeighting Algorithm

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: data $\{D_t^s\}_{t=1}^T$ and $\{D_t^q\}_{t=1}^T$, Number of learning iterations $K$, step size $\alpha$ for updating $\mathbf{w}$.
Initialize: $w^0 = (1, 1, ..., 1)$, $\theta^0$, $\eta$.
for $k = 1$ to $K$ do
    $\theta^k = \theta^{k-1} - \frac{\eta}{T} \sum_{t=1}^T w_t \nabla_\theta \mathcal{L}_t^s (\theta^{k-1}, D_t^s)$.
    for $t = 1$ to $T$ do
        $d_t = -\frac{\eta}{T} \nabla_\theta \mathcal{L}_t^q (\theta^k, D_t^q) \nabla_\theta \mathbf{L}^s (\theta^{k-1}, D^s)$.
    end for
    $\mathbf{d} = (d_1^\top, d_2^\top, ..., d_T^\top)$ $\lambda^* = \arg \min_\lambda \{ \| \lambda \mathbf{d}^\top \|_2^2 | \lambda \mathbf{1}^\top = 1, \lambda \succeq \mathbf{0} \}$
    (calls Algorithm 2).
    $d_c = \lambda^* \mathbf{d}^\top$.
    $\mathbf{w}^{k+1} = \mathbf{w}^k - \alpha d_c$.
end for
</div>

Algorithm 2: Frank-Wolfe Algorithm

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: Number of Iterations $N$.
Initialize: $\lambda_0 = [\frac{1}{T}, ..., \frac{1}{T}]$.
$B = \mathbf{d}^\top \mathbf{d}$.
for $i = 0$ to $N$ do
    $v = \arg \min_{v \in \{v^\top \mathbf{1} = 1, v \succeq \mathbf{0}\}} v^\top B\lambda$.
    $\gamma = \arg \min_{\gamma \in [0,1]} (\lambda_i + \gamma(v - \lambda_i))^^\top B(\lambda_i + \gamma(v - \lambda_i)).$ $\lambda_{i+1} = (1 - \gamma)\lambda_i + \gamma v$.
end for
return: $\lambda_N$
</div>

## 4.3.3 MetaWeighting

Moving against $d _ { c } ,$ , all the objective can be optimized; thus, the update rule of w is

$$
\mathbf {w} ^ {k + 1} = \mathbf {w} ^ {k} - \alpha d _ {c}, \tag {13}
$$

where $\alpha$ is the step size. Based on this update rule, the task weights are automatically learnt oriented by optimizing the generalization losses.

Overall, we propose the MetaWeighting algorithm, which is presented in algorithmic form in Algorithm 1. Our proposed method bridges the gap between task weighting and generalization performance of MTL.

## 5 Theoretical Analysis

In this section, we study the generalization error bound for MTL; furthermore, we compare the bound w.r.t training loss and the bound w.r.t the query loss. The comparison presents that the query loss is a more accurate estimation of the generalization loss than the training loss.

Firstly, we derive the generalization error bound w.r.t training loss for MTL.

Theorem 2. Assume we have n training samples for each task. Let $\sigma ~ = ~ \{ \{ \sigma _ { i } ^ { t } \} _ { i = 1 } ^ { n } \} _ { t = 1 } ^ { T }$ be a sequence ofbinary random variables such that each $\sigma _ { i } ^ { t } = \pm 1$ is independent with probability 1/2. Then, $\forall \delta \in [ 0 , 1 ]$ , for all $h ( \cdot , \theta ^ { s } , \theta ^ { 1 } , . . . , \theta ^ { T } ) \in \mathcal { H } .$ , with probability ofat least $1 - \delta .$

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \left(\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} (\theta , D _ {t})\right) \\ \leq 2 R (l \circ \mathcal {H} \circ D) + 4 \sqrt {\frac {2 \log (4 / \delta)}{T n}}. \tag {14} \\ \end{array}
$$

where

$$
R (l \circ \mathcal {H} \circ D) = \mathbb {E} _ {\sigma} \sup _ {\theta} (\frac {1}{T n} \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {n} \sigma_ {i} ^ {t} l (h (x _ {i} ^ {t}, \theta), y _ {i} ^ {t}). \tag {15}
$$

is the Rademacher complexityfor MTL.

Proof. The proof is provided in Appendix A.

Next, we derive the generalization error bound w.r.t query loss for MTL.

Theorem 3. Assume we have m training samples for each task. $\forall \delta \in [ 0 , 1 ]$ , with probability of at least 1 δ,for all $h ( \cdot , \theta ^ { s } , \theta ^ { 1 } , . . . , \theta ^ { T } ) \in \mathcal { H } ,$ , we have

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} (\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t} ^ {q})) \leq \sqrt {\frac {\log (2 / \delta)}{2 m}}. \tag {16}
$$

Proof. The proof is provided in Appendix A.

Comparing the bound (14) and (16), we can find that the upper bound for the query loss is tighter than that for the training loss. Taking m to be order of n, the query loss is a more accurate estimate of the generalization loss than the training loss by a factor that depends on the Rademacher complexity.

## 6 Experiments

In this section, we perform experimental studies on sentiment analysis to evaluate the performance of our proposed MetaWeighting and verify our theoretical analysis.

## 6.1 Datasets

Sentiment Analysis <sup>1</sup>. We evaluate our algorithm on product reviews from Amazon. The dataset (Blitzer et al., 2007) contains product reviews from 14 domains, including books, DVDs, electronics, kitchen appliances and so on. We consider each domain as a binary classification task. Reviews with rating > 3 were labeled positive, those with rating < 3 were labeled negative, reviews with rating <sub>=</sub> 3 are discarded as the sentiments were ambiguous and hard to predict.

Topic Classification <sup>2</sup>. We select 16 newsgroups from the 20 Newsgroup dataset, which is a collection of approximately 20,000 newsgroup documents that is partitioned (nearly) evenly across 20 different newsgroups, then formulate them into four 4-class classification tasks (as shown in Table 1) to evaluate the performance of our algorithm on topic classification.

Table 1: Data Allocation for Topic Classification Tasks.

<table><tr><td>TASKS</td><td>NEWSGROUPS</td></tr><tr><td>COMP</td><td>OS.MS-WINDOWS.MISC, SYS.MAC.HARDWARE, GRAPHICS, WINDOWS.X</td></tr><tr><td>REC</td><td>SPORT.BASEBALL, SPORT.HOCKEY AUTOS, MOTORCYCLES</td></tr><tr><td>SCI</td><td>CRYPT, ELECTRONICS, MED, SPACE</td></tr><tr><td>TALK</td><td>POLITICS.MIDEAST, RELIGION.MISC, POLITICS.MISC, POLITICS.GUNS</td></tr></table>

## 6.2 Baselines

We compare MetaWeighting with methods:

Single-Task Learning (STL): learning each task independently.

Uniform: learning tasks simultaneously using uniform task weights.

Uncertainty: using the uncertainty weighting method proposed by (Kendall et al., 2018).

GradNorm: using the gradient normalization method proposed by (Chen et al., 2018).

MGDA: using the MGDA-UB method proposed by (Sener and Koltun, 2018).

AdvMTL: using the adversarial Multi-task Learning method proposed by (Liu et al., 2017).

TchebycheffAdv: using the Adversarial Tchebycheff procedure proposed by (Mao et al., 2020).

BanditMTL: using the BanditMTL method proposed by (Mao et al., 2021).

![](images/2ca04bc90cb590c7eec031c1a4fc64fbba999e55169e6e636d2acd5c0f3ea489.jpg)

Figure 2: Classification accuracy of Single Task Learning, Uniform Scaling, AdvMTL, MGDA, GradNorm, Uncertainty, TchebycheffAdv, BanditMTL and MetaWeighting on TextCNN for the sentiment analysis dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on ten of the fourteen tasks; besides, its average performance is superior to that of all baselines.  
![](images/f173a382718b18786aef2de09bcf99fa3fa1fb46e20dab85bafc19e9311438f7.jpg)  
Figure 3: Classification accuracy of Single Task Learning, Uniform Scaling, AdvMTL, MGDA, GradNorm, Uncertainty, TchebycheffAdv, BanditMTL and MetaWeighting on TextCNN for the topic classification dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines in all tasks.

## 6.3 Experimental Settings

We adopt the hard parameter-sharing MTL framework (Mao et al., 2021), where the shared representation extractor is built with TextCNN or BERT; besides, the task-specific module is formulated by means of one fully connected layer ending with a softmax function. The detailed experimental settings are introduced in the Appendix B.

## 6.4 Classification Performance

We compare the proposed MetaWeighting with the baselines and report the results over 10 runs by plotting the classification accuracy of each task for both sentiment analysis and topic classification. The results on TextCNN are shown in Fig. 2 and 3. Due to space limitations, we provide the results for BERT in the Appendix C. All experimental results show that our proposed MetaWeighting outperforms all baselines and achieves state-of-the-art performance.

![](images/56b9de95ce41abe10d371bf74a13ee66754f7704d779d204700a2c18cb2e65a9.jpg)

<details>
<summary>boxplot</summary>

| Metric | \(\rho\) | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- | --- |
| Sentiment (%) | 0.01 | ~89.22 | ~89.30 | ~89.38 |
| Sentiment (%) | 0.05 | ~89.58 | ~89.65 | ~89.75 |
| Sentiment (%) | 0.1 | ~89.85 | ~89.90 | ~89.95 |
| Sentiment (%) | 0.2 | ~89.78 | ~89.88 | ~89.90 |
| Sentiment (%) | 0.3 | ~89.52 | ~89.62 | ~89.70 |
| Sentiment (%) | 0.4 | ~89.32 | ~89.40 | ~89.48 |
| Topic (%) | 0.01 | ~91.52 | ~91.62 | ~91.68 |
| Topic (%) | 0.05 | ~91.85 | ~91.92 | ~91.98 |
| Topic (%) | 0.1 | ~91.92 | ~91.98 | ~92.02 |
| Topic (%) | 0.2 | ~91.82 | ~91.88 | ~91.95 |
| Topic (%) | 0.3 | ~91.70 | ~91.75 | ~91.82 |
| Topic (%) | 0.4 | ~91.40 | ~91.50 | ~91.60 |
</details>

Figure 4: Task-average classification accuracy w.r.t different value of $\rho$ (query-split radio) for sentiment analysis and topic classification.  
![](images/6e295dc180e48f57a7ffb73c71dab2b4c7dbe7dd33125d3ed8b8db6cb8779bb9.jpg)

<details>
<summary>boxplot</summary>

| Metric | \(\alpha\) | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- | --- |
| Sentiment (%) | 0.01 | ~89.45 | ~89.50 | ~89.56 |
| Sentiment (%) | 0.05 | ~89.70 | ~89.75 | ~89.81 |
| Sentiment (%) | 0.1 | ~89.85 | ~89.90 | ~89.95 |
| Sentiment (%) | 0.5 | ~89.58 | ~89.63 | ~89.68 |
| Sentiment (%) | 1 | ~89.28 | ~89.35 | ~89.42 |
| Topic (%) | 0.01 | ~91.35 | ~91.40 | ~91.48 |
| Topic (%) | 0.05 | ~91.67 | ~91.72 | ~91.76 |
| Topic (%) | 0.1 | ~91.83 | ~91.88 | ~91.93 |
| Topic (%) | 0.5 | ~91.93 | ~91.97 | ~92.02 |
| Topic (%) | 1 | ~91.58 | ~91.65 | ~91.72 |
</details>

Figure 5: Task-average classification accuracy w.r.t different value of α (step size) for sentiment analysis and topic classification.

## 6.5 The Impact of Query-Split Radio

Let n be the size of the entire training set and m be the size of the query set. We define the querysplit radio as $\textstyle \rho = { \frac { m } { n } }$ to indicate the radio of query samples to the entire training samples. From the theoretical analysis of Section 5, we can see that the query loss can estimate generalization loss more accurately when $\rho$ increases, but increasing $\rho$ would hurt the training process for the size of support set decreases. Therefore, $\rho$ faces a trade-off between the performance estimation of generalization loss and training performance.

To investigate the impact of $\rho ,$ we record the changes in MetaWeighting’s average classification accuracy w.r.t different values of $\rho$ in Fig. 4, where each boxplot visually illustrates the distribution of results over ten runs through displaying the data quartiles (first quartile and third quartile), minimum/maximum value and median. These experiments are conducted based on TextCNN. In this figure, as $\rho$ increases, the average accuracy of MetaWeighting first increases and then decreases. It verifies our theoretical analysis. For both sentiment analysis and topic classification, setting $\rho = 0 . 1$ provides satisfactory results.

![](images/a8a88210cd95e0b0a4ee8b7a4c4b954671e8c05c61cf1cffdec19fbb1e98b44a.jpg)

<details>
<summary>radar</summary>

| Epoch | Time Step | Training Loss | Generalization Loss | Query Loss |
| --- | --- | --- | --- | --- |
| 500th epoch | t1 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t2 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t3 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t4 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t5 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t6 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t7 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t8 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t9 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t10 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t11 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t12 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t13 | ~0.25 | ~0.35 | ~0.25 |
| 500th epoch | t14 | ~0.25 | ~0.35 | ~0.25 |
| 1000th epoch | t1 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t2 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t3 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t4 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t5 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t6 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t7 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t8 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t9 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t10 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t11 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t12 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t13 | ~0.35 | ~0.45 | ~0.35 |
| 1000th epoch | t14 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t1 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t2 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t3 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t4 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t5 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t6 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t7 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t8 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t9 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t10 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t11 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t12 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t13 | ~0.35 | ~0.45 | ~0.35 |
| 1500th epoch | t14 | ~0.35 | ~0.45 | ~0.35 |
</details>

Figure 6: Illustration of the gap between training loss, query loss and generalization loss in the training process of sentiment analysis $( 5 0 0 ^ { t h }$ $1 0 0 0 ^ { t h }$ ， $1 5 0 0 ^ { t \bar { h } }$ epochs respectively).  
![](images/c8776b11a0e931fe8cf20025accaeda10cfcedc9940e8485fc376917c153df00.jpg)

<details>
<summary>radar</summary>

| Task | Epoch | Training Loss | Generalization Loss | Query Loss |
| --- | --- | --- | --- | --- |
| task 1 | 500th epoch | ~0.6 | ~0.7 | ~0.6 |
| task 1 | 1000th epoch | ~0.2 | ~0.4 | ~0.3 |
| task 1 | 1500th epoch | ~0.2 | ~0.4 | ~0.3 |
| task 3 | 500th epoch | ~0.5 | ~0.6 | ~0.5 |
| task 3 | 1000th epoch | ~0.2 | ~0.3 | ~0.2 |
| task 3 | 1500th epoch | ~0.2 | ~0.3 | ~0.2 |
| task 4 | 500th epoch | ~0.6 | ~0.7 | ~0.6 |
| task 4 | 1000th epoch | ~0.2 | ~0.4 | ~0.3 |
| task 4 | 1500th epoch | ~0.2 | ~0.4 | ~0.3 |
</details>

Figure 7: Illustration of the gap between training loss, query loss and generalization loss in the training process of topic classification $( 5 0 0 ^ { t h }$ $1 0 0 0 ^ { t h }$ $1 5 0 0 ^ { t \overline { { h } } }$ epochs respectively).

## 6.6 Sensitive Study on α

In MetaWeighting, the step size α is a hyperparameter. To determine whether the performance of MetaWeighting is sensitive to $\alpha ,$ we conduct experiments on the classification accuracy performance of MetaWeighting w.r.t different values of α based on the TextCNN model. The results of these experiments are presented in Figure 5 (boxplots over ten runs). As the figure shows, the performance of our proposed method is not very sensitive to α when α is within the range of 0.05 to 0.1 for sentiment analysis and 0.1 to 0.5 for topic classification. The results demonstrate that MetaWeighting can work well in a wide range of α values.

## 6.7 The Gap between the Training Loss, Query Loss and Generalization Loss

To experimentally verify that the query loss is a good estimator for generalization loss, we record the generalization loss (estimated by test loss), query loss and training loss for each task during training and report the results in Fig. 6 and 7 for sentiment analysis and topic classification respectively. From these figures, we can see that there is a large gap between the training and generalization loss, while the gap between the query and generalization loss is smaller than that between the training and generalization loss. The results verify our theoretical analysis in Section 5; furthermore, they experimentally support our motivation for MetaWeighting.

![](images/680127b0bec288f3b14cb2214f3119e8e63352c41523d26857e0dfd922fcf8ab.jpg)  
Figure 8: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for sentiment analysis.

![](images/3999a5b1897ca8511f93320e168e3abe4a6c01dfb399517cbd1ddfba1dbb7e89.jpg)  
Figure 9: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for topic classification.

In this section, TextCNN is used, and tasks have uniform weights during training. Fig. 1 is obtained under this setting as well.

## 6.8 The Evolution of Task Weights

In this section, we observe the changes in task weights in the training process of MetaWeighting and compare these changes with four baselines (Uncertainty, Gradnorm, MGDA and BanditMTL). The results for sentiment analysis and topic classification are reported in Fig. 8 and 9 respectively. Due to space limitations, for sentiment analysis, we only report the results of the first four tasks here, and the results of the other ten tasks are presented in the Appendix D.

From these figures, we can see that the weight adaption process of MetaWeighting is different with that of Uncertainty, Gradnorm, MGDA and BanditMTL. In MetaWeighting, the task weights are automatically learnt, and there is no pre-defined heuristic involved. It is verified by the evolution curves of task weights for MetaWeighting illustrated in Fig. 8 and 9, which fluctuate without any regular patterns.

## 7 Conclusion

This paper presents that the gap between the training loss and the generalization loss, which is overlooked by existing task weighting methods, is nonnegligible; furthermore, to narrow this gap, a novel task weighting method (dubbed MetaWeighting) is proposed. In MetaWeighting, multi-task text classification is formulated as a multi-objective bilevel programming problem, and then solved in a learning-to-learn manner. MetaWeighting automatically learns the task weights without any predefined heuristic and achieves state-of-the-art performance. It has the potential to forge new trends in task weighting research.

## References

Jon Wellner Aad van der Vaart. 1996. Weak convergence and empirical processes. Springer.  
Zeyuan Allen-Zhu, Yuanzhi Li, and Yingyu Liang. 2019. Learning and generalization in overparameterized neural networks, going beyond two layers. In NeurIPS.  
Luís B Almeida, Thibault Langlois, José D Amaral, and Alexander Plakhov. 1998. Parameter adaptation in stochastic optimization. In On-Line Learning in Neural Networks, pages 111–134. Cambridge University Press.  
Jonathan Baxter. 2000. A model of inductive bias learning. Journal ofartificial intelligence research, 12:149–198.  
Atilim Gunes Baydin, Robert Cornish, David Martínez-Rubio, Mark Schmidt, and Frank Wood. 2018. Online learning rate adaptation with hypergradient de scent. In ICLR.  
John Blitzer, Mark Dredze, and Fernando Pereira. 2007. Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In ACL.  
Rich Caruana. 1993. Multitask learning: A knowledgebased source of inductive bias. In ICML.  
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML.  
Jean-Antoine Désidéri. 2012. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318.  
Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. 2018. Dynamic task prioritization for multitask learning. In ECCV.  
Martin Jaggi. 2013. Revisiting frank-wolfe: Projectionfree sparse convex optimization. In ICML.  
Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR.  
Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In ICLR.  
Xi Lin, Zhiyuan Yang, Qingfu Zhang, and Sam Kwong. 2020. Controllable pareto multi-task learning. CoRR.  
Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, and Sam Kwong. 2019. Pareto multi-task learning. In NIPS.  
Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial multi-task learning for text classification. In ACL.  
Shikun Liu, Edward Johns, and Andrew J. Davison. 2019. End-to-end multi-task learning with attention. In CVPR.  
Pingchuan Ma, Tao Du, and Wojciech Matusik. 2020. Efficient continuous pareto exploration in multi-task learning. In ICML.  
Debabrata Mahapatra and Vaibhav Rajan. 2020. Multitask learning with user preferences: Gradient descent with controlled ascent in pareto optimization. In ICML.  
Yuren Mao, Zekai Wang, Weiwei Liu, Xuemin Lin, and Wenbin Hu. 2021. Banditmtl: Bandit-based multi task learning for text classification. In ACL.  
Yuren Mao, Shuang Yun, Weiwei Liu, and Bo Du. 2020. Tchebycheff procedure for multi-task text classification. In ACL.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In EMNLP.  
Ozan Sener and Vladlen Koltun. 2018. Multi-task learning as multi-objective optimization. In NeurIPS.  
Ishan Tarunesh, Sushil Khyalia, Vishwajeet Kumar, Ganesh Ramakrishnan, and Preethi Jyothi. 2021. Meta-learning for effective multi-task and multilin gual modelling. In EACL.  
Xinyi Wang, Yulia Tsvetkov, and Graham Neubig. 2020. Balancing training for multilingual neural machine translation. In ACL.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In EMNLP

## A Proof of the Theorem 2 and Theorem 3

Lemma 1 (McDiarmid’s Inequality). Let V be some set and let $f : V ^ { n } \to \mathbb { R } b e a$ function of n variables such thatfor some $c > 0$ , for all $i \in [ n ]$ andfor all $z _ { 1 } , . . . , z _ { n } , z _ { i } ^ { \prime } \in V$ we have

$$
\left| f (z _ {1},..., z _ {n}) - f (z _ {1},..., z _ {i - 1}, z _ {i} ^ {\prime}, z _ {i + 1},..., z _ {n}) \right| \leq c. \tag {17}
$$

Let $Z _ { 1 } , . . . , Z _ { n }$ be n independent random variables taking values in V . Then, with probability of at least $1 - \delta$ we have

$$
| f (Z _ {1},..., Z _ {n}) - \mathbb {E} [ f (Z _ {1},..., Z _ {n}) ] | \leq c \sqrt {\frac {n \log (2 / \delta)}{2}}. \tag {18}
$$

Lemma 2 (Hoeffding’s Inequality). Let $z _ { 1 } , . . . , z _ { m }$ be a a sequence of i.i.d. random variables and assume thatfor all i, $\mathbb { E } ( z _ { i } ) = \mu$ and $P ( a \leq z _ { i } \leq$ $b ) = 1$ . Then,for any $\epsilon > 0$

$$
P \left[ \left| \frac {1}{m} \sum_ {i = 1} ^ {m} z _ {i} - \mu \right| > \epsilon \right] \leq 2 e x p (\frac {- 2 m \epsilon^ {2}}{(b - a) ^ {2}}). \tag {19}
$$

Lemma 3. Assume that $\begin{array} { r l } { \forall ( x _ { t } ^ { i } , y _ { t } ^ { i } ) , ( x _ { t } ^ { j } , y _ { t } ^ { j } ) } & { { } : } \end{array}$ $| l ( h ( x _ { t } ^ { i } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { i } ) - l ( h ( x _ { t } ^ { j } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { j } ) | \leq c .$ Let

$$
R e p (\mathcal {H}, D) = \sup _ {h \in \mathcal {H}} \frac {1}{T} \sum_ {t = 1} ^ {T} (\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} (\theta , D _ {t})), \tag {20}
$$

then $\forall \delta \in [ 0 , 1 ]$ , with probability ofat least $1 - \delta .$

$$
R e p (\mathcal {H}, D) \leq \mathbb {E} _ {D} R e p (\mathcal {H}, D) + c \sqrt {\frac {2 \log (2 / \delta)}{T n}}. \tag {21}
$$

Proof. Let $\begin{array} { r l r l r l } { s _ { t } ^ { i } } & { { } } & { = } & { { } } & { ( x _ { t } ^ { i } , y _ { t } ^ { i } ) } \end{array}$ The

training set for MTL is D $\{ \{ ( s _ { 1 } ^ { 1 } , . . . , s _ { 1 } ^ { n } \} , . . . , \{ s _ { t } ^ { 1 } , . . . , s _ { t } ^ { n } \} , . . . , \{ s _ { T } ^ { 1 } , . . . , s _ { T } ^ { n } \} \}$

For $\forall t , i$ , replace $s _ { t } ^ { i }$ with $\begin{array} { r c c l } { u _ { t } ^ { i } } & { = } & { ( x _ { t } ^ { * } , y _ { t } ^ { * } ) } & { \in } \end{array}$ D and create a new dataset D <sup>h</sup> $\{ \{ ( s _ { 1 } ^ { 1 } , . . . , s _ { 1 } ^ { n } \} , . . . , \{ s _ { t } ^ { 1 } , . . . , u _ { t } ^ { i } , . . . , s _ { t } ^ { n } \} , . . . , \{ s _ { T } ^ { 1 } , . . . , s _ { T } ^ { n } \} \}$

Let $h _ { t } ( \cdot ) \ = \ h ( \cdot , \theta ^ { s } , \theta ^ { t } ) . \mathrm { A s } \ \forall ( x _ { t } ^ { i } , y _ { t } ^ { i } ) , ( x _ { t } ^ { j } , y _ { t } ^ { j } )$ : $| l ( h ( x _ { t } ^ { i } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { i } ) - l ( h ( x _ { t } ^ { j } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { j } ) | \leq c ,$ we have

$$
\begin{array}{l} R e p (\mathcal {H}, D) - R e p (\mathcal {H}, \overline {{D}}) \\ \leq \sup _ {h \in \mathcal {H}} \frac {1}{T n} \left| l (h _ {t} (x _ {t} ^ {n}), y _ {t} ^ {n}) - l (h _ {t} (x _ {t} ^ {*}), y _ {t} ^ {*})) \right| \leq \frac {c}{T n}. \tag {22} \\ \end{array}
$$

Using the McDiarmid’s Inequality (Lemma 1), we have

$$
\begin{array}{l} R e p (\mathcal {H}, D) \leq \mathbb {E} _ {D} R e p (\mathcal {H}, D) + \frac {2 c}{T n} \sqrt {\frac {T n \log (2 / \delta)}{2}} \\ = \mathbb {E} _ {D} \operatorname{Rep} (\mathcal {H}, D) + c \sqrt {\frac {2 \log (2 / \delta)}{T n}}. \tag {23} \\ \end{array}
$$

We conclude our proof.

![](images/356b4f6cfc7b6309504511bb0eb1f820ff223f37bb2a13a19f1d5591458af102.jpg)

## Proof of Theorem 2.

Proof. Using the standard symmetrization argument (for example, see Lemma 2.3.1 of (Aad van der Vaart, 1996) ), we have

$$
\mathbb {E} _ {D} R e p (\mathcal {H}, D) \leq 2 \mathbb {E} _ {D} R (l \circ \mathcal {H} \circ D). \tag {24}
$$

Combining Eq. (21) and Eq. (24), with probability $1 - \delta / 2 \colon$

$$
\begin{array}{l} \sup _ {h \in \mathcal {H}} \frac {1}{T} \sum_ {t = 1} ^ {T} \left(\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} \left(\theta , D _ {t}\right)\right) (25) \\ \leq 2 \mathbb {E} _ {D} R (l \circ \mathcal {H} \circ D) + c \sqrt {\frac {2 \log (4 / \delta)}{T n}}. (25) \\ \end{array}
$$

Obviously, with probability of at least $1 - \delta / 2$ , for all $h \in \mathcal H$ , we have

$$
\begin{array}{l} \begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \left(\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} \left(\theta , D _ {t}\right)\right) \\ + 2 T - D (1 - \delta t - D) = \sqrt {2 \log (4 / \delta)} \end{array} \tag {26} \\ \begin{array}{l} \leq 2 \mathbb {E} _ {D} R (l \circ \mathcal {H} \circ D) + c \sqrt {\frac {2 \log (4 / \delta)}{T n}}. \end{array} \\ \end{array}
$$

Let $\begin{array} { r l r } { s _ { t } ^ { i } } & { { } = } & { ( x _ { t } ^ { i } , y _ { t } ^ { i } ) } \end{array}$ The training set for MTL is D 二 $\{ \{ ( s _ { 1 } ^ { 1 } , . . . , s _ { 1 } ^ { n } \} , . . . , \{ s _ { t } ^ { 1 } , . . . , s _ { t } ^ { n } \} , . . . , \{ s _ { T } ^ { 1 } , . . . , s _ { T } ^ { n } \} \}$

For $\forall t , i ,$ replace $s _ { t } ^ { i }$ with $u _ { t } ^ { i } = ( x _ { t } ^ { * } , y _ { t } ^ { * } ) \in D _ { t }$ and create a new dataset $\overline { { D } } = \{ \{ ( s _ { 1 } ^ { 1 } , . . . , s _ { 1 } ^ { n } \} , . . . ,$ $\{ s _ { t } ^ { 1 } , . . . , u _ { t } ^ { i } , . . . , s _ { t } ^ { n } \} , . . . , \{ s _ { T } ^ { 1 } , . . . , s _ { T } ^ { n } \} \}$

Let $h _ { t } ( \cdot ) = h ( \cdot , \theta ^ { s } , \theta ^ { t } ) . \mathrm { ~ A s ~ } \forall ( x _ { t } ^ { i } , y _ { t } ^ { i } ) , ( x _ { t } ^ { j } , y _ { t } ^ { j } ) :$ $| l ( h ( x _ { t } ^ { i } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { i } ) - l ( h ( x _ { t } ^ { j } , \theta ^ { s } , \theta ^ { t } ) , y _ { t } ^ { j } ) | \leq c ,$ we have

$$
\begin{array}{l} R e p (\mathcal {H}, D) - R e p (\mathcal {H}, \overline {{D}}) \leq \\ \sup _ {h \in \mathcal {H}} \frac {1}{T n} | l (h _ {t} (x _ {t} ^ {n}), y _ {t} ^ {n}) - l (h _ {t} (x _ {t} ^ {*}), y _ {t} ^ {*})) | \leq \frac {c}{T n} \\ \end{array}
$$

(27)

Using the McDiarmid’s Inequality (Lemma 1), we ave that: with probability of at least $1 - \delta / 2 \colon$

$$
\mathbb {E} _ {D} R (l \circ \mathcal {H} \circ D) \leq R (l \circ \mathcal {H} \circ D) + 2 c \sqrt {\frac {2 \log (4 / \delta)}{T n}}. \tag {28}
$$

Based on Eq. (28) and the union bound, we have that - with probability of at least $1 - \delta \colon$

$$
\begin{array}{l} \begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} \left(\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} \left(\theta , D _ {t}\right)\right) \\ 6. 2 B (1 - \beta^ {\prime} - D) + 4 = \sqrt {2 \log (4 / \delta)} \end{array} \tag {29} \\ \leq 2 R (l \circ \mathcal {H} \circ D) + 4 c \sqrt {\frac {2 \log (4 / \delta)}{T n}}. \\ \end{array}
$$

In our setting, $l ( \cdot , \cdot ) : \mathcal { V } ^ { t } \times \mathcal { V } ^ { t }  [ 0 , 1 ]$ , then $c = 1$ We have

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 1} ^ {T} (\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {t r} (\theta , D _ {t})) \tag {30} \\ \leq 2 R (l \circ \mathcal {H} \circ D) + 4 \sqrt {\frac {2 \log (4 / \delta)}{T n}}. \\ \end{array}
$$

We conclude our proof.

![](images/90bd1ab17efab540bfd328d4adf8ae6fc3edf28cb7c843c3ab4ee64c30e05aa7.jpg)

Based on the Hoeffding’s Inequality (Lemma 2), we have the following theorem.

## Proof of Theorem 3.

Proof. Based on the Hoeffding’s Inequality (Lemma 2) and $l ( \cdot , \cdot ) : \mathcal { V } ^ { t } \times \mathcal { V } ^ { t }  [ 0 , 1 ]$ , for each $h ( \cdot , \theta ^ { s } , \theta ^ { t } ) \in \mathcal { H } ^ { t }$ , we have

$$
P \left[ | \mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t}) | > \epsilon \right] \leq 2 e x p (- 2 m \epsilon^ {2}). \tag {31}
$$

Then, with probability of at least $1 { - } 2 e x p ( - 2 m \epsilon ^ { 2 } )$ we have

$$
\left| \mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t}) \right| \leq \epsilon . \tag {32}
$$

Let $\delta = 2 e x p ( - 2 m \epsilon ^ { 2 } )$ , we have that with probability of at least $1 - \delta .$

$$
\left| \mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t}) \right| \leq \sqrt {\frac {\log (2 / \delta)}{2 m}}. \tag {33}
$$

Thus, for each task,

$$
\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t}) \leq \sqrt {\frac {\log (2 / \delta)}{2 m}}. \tag {34}
$$

Since the bound for each task are independent, we have

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} (\mathcal {L} _ {t} (\theta) - \mathcal {L} _ {t} ^ {q} (\theta , D _ {t})) \leq \sqrt {\frac {\log (2 / \delta)}{2 m}}. \tag {35}
$$

We conclude our proof.

![](images/e7130eb5901f1d7d8a7904aa1b9ae4325a5d934d4914a8f43a6787a4177c7b95.jpg)

## B Detailed Experimental Settings

We adopt the hard parameter-sharing MTL framework (Mao et al., 2021), where the shared representation extractor is built with TextCNN or BERT; besides, the task-specific module is formulated by means of one fully connected layer ending with a softmax function. The TextCNN module is structured with three parallel convolutional layers with kernels size of 3, 5, 7 respectively. For TextCNN, we adopt Pre-trained GloVe (Pennington et al., 2014) word embeddings. By contrast, the BERT module is formulated via a pre-trained BERT-base model provided by Hugging Face(Wolf et al., 2020), with a hidden size of 768, 12 Transformer blocks and 12 self-attention heads.

We train the deep MTL network model in line with Algorithm 1. We set α to be 0.1 and 0.5 for sentiment analysis and topic classification respectively, and the query-split radio (radio of query samples to entire training samples) to be 0.1 for both sentiment analysis and topic classification. We use the Adam optimizer (Kingma and Ba, 2015). We train over 3000 epochs for TextCNN and finetune over 50 epochs for BERT. For TextCNN, the learning rate is $1 e - 3$ and the batch size is 256. For BERT, the learning rate is $2 e - 5$ , the batch size is 32, and the max sequence length is 256. For the baselines, we search over the set $\{ 1 e - 5 , 2 e - 5 , 5 e - 5 , 1 e - 4 , 5 e - 4 , 1 e - 3 , 5 e - 3 \}$ learning rates and choose the model with best performance.

## C Classification Performance on BERT

For the BERT-based MTL model, we compare the proposed MetaWeighting with the baselines and report the results over 10 runs by plotting the classification accuracy of each task for both sentiment analysis and topic classification in Fig. 10 and 11. AdvMTL and TchebycheffAdv are not available for BERT; thus, we do not compare with AdvMTL and compare with Tchebycheff which is TchebycheffAdv without aversarial module (Mao et al., 2021). From these figures, we can see that our proposed MetaWeighting outperforms all baselines and achieves state-of-the-art performance.

## D The Evolution of Task Weights for Sentiment Analysis

Fig. 12 illustrates the changes in task weights in the training process of MetaWeighting for all the tasks of sentiment analysis.

![](images/c92a2db985ec9be893ef018f4bfe035f2a30d884d3ca9070faf8722a3d7ace25.jpg)  
Figure 10: Classification accuracy of Single Task Learning, Uniform Scaling, MGDA, TchebycheffAdv, Uncertainty, GradNorm, BanditMTL and MetaWeighting on BERT for the sentiment analysis dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on eleven of the fourteen tasks; besides, its average performance is superior to that of all baselines.

![](images/065b1fb94b55de26045ffc89fb7fd149d3f44bf1d91e551730423133569e5968.jpg)  
Figure 11: Classification accuracy of Single Task Learning, Uniform Scaling, MGDA, TchebycheffAdv, Uncertainty, GradNorm, BanditMTL and MetaWeighting on BERT for the topic classification dataset. Each colored cluster illustrates the classification accuracy performance of a method over 10 runs. Our proposed MetaWeighting outperforms all baselines on three of the four tasks; besides, its average performance is superior to that of all baselines.

![](images/ecf297f4a33a3302b77b0af04176ad6cd8d849f428d6d59784949da62b2dc761.jpg)

<details>
<summary>line</summary>

| Dataset | Metric | Epoch 0 | Epoch 500 | Epoch 1000 | Epoch 1500 | Epoch 2000 | Epoch 2500 | Epoch 3000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apparel | MGDA | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| apparel | GradNorm | ~1.0 | ~0.8 | ~0.8 | ~0.8 | ~0.8 | ~0.8 | ~0.8 |
| apparel | Uncertainty | ~0.1 | ~0.6 | ~0.9 | ~1.1 | ~1.2 | ~1.4 | ~1.4 |
| apparel | BanditMTL | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| apparel | MetaWeighting | ~1.2 | ~1.3 | ~1.4 | ~1.3 | ~1.2 | ~1.3 | ~1.5 |
| baby | MGDA | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| baby | GradNorm | ~1.0 | ~1.1 | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| baby | Uncertainty | ~0.1 | ~0.7 | ~0.9 | ~1.0 | ~1.2 | ~1.3 | ~1.4 |
| baby | BanditMTL | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| baby | MetaWeighting | ~1.2 | ~1.2 | ~1.4 | ~1.3 | ~1.2 | ~1.2 | ~1.2 |
| books | MGDA | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| books | GradNorm | ~1.0 | ~1.1 | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| books | Uncertainty | ~0.2 | ~0.4 | ~0.5 | ~0.5 | ~0.5 | ~0.5 | ~0.6 |
| books | BanditMTL | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| books | MetaWeighting | ~1.2 | ~1.4 | ~1.3 | ~1.2 | ~1.2 | ~1.4 | ~1.3 |
| camera_photo | MGDA | ~0.1 | ~0.6 | ~1.2 | ~1.3 | ~1.4 | ~1.6 | ~1.8 |
| camera_photo | GradNorm | ~1.2 | ~1.2 | ~1.3 | ~1.3 | ~1.3 | ~1.3 | ~1.3 |
| camera_photo | Uncertainty | ~0.1 | ~0.8 | ~1.2 | ~1.3 | ~1.4 | ~1.6 | ~1.8 |
| camera_photo | BanditMTL | ~0.2 | ~0.2 | ~0.2 | ~0.2 | ~0.2 | ~0.2 | ~0.2 |
| camera_photo | MetaWeighting | ~1.2 | ~1.4 | ~1.6 | ~1.7 | ~1.8 | ~2.0 | ~2.2 |
</details>

Figure 12: Comparison of task weight adaption processes between MetaWeighting, Uncertainty, Gradnorm, MGDA and BanditMTL for sentiment analysis.