# Multi-Task Learning as Multi-Objective Optimization

Ozan Sener

Intel Labs

Vladlen Koltun

Intel Labs

## Abstract

In multi-task learning, multiple tasks are solved jointly, sharing inductive bias between them. Multi-task learning is inherently a multi-objective problem because different tasks may conflict, necessitating a trade-off. A common compromise is to optimize a proxy objective that minimizes a weighted linear combination of pertask losses. However, this workaround is only valid when the tasks do not compete, which is rarely the case. In this paper, we explicitly cast multi-task learning as multi-objective optimization, with the overall objective of finding a Pareto optimal solution. To this end, we use algorithms developed in the gradient-based multiobjective optimization literature. These algorithms are not directly applicable to large-scale learning problems since they scale poorly with the dimensionality of the gradients and the number of tasks. We therefore propose an upper bound for the multi-objective loss and show that it can be optimized efficiently. We further prove that optimizing this upper bound yields a Pareto optimal solution under realistic assumptions. We apply our method to a variety of multi-task deep learning problems including digit classification, scene understanding (joint semantic segmentation, instance segmentation, and depth estimation), and multi label classification. Our method produces higher-performing models than recent multi-task learning formulations or per-task training.

## 1 Introduction

One of the most surprising results in statistics is Stein’s paradox. Stein (1956) showed that it is better to estimate the means of three or more Gaussian random variables using samples from all of them rather than estimating them separately, even when the Gaussians are independent. Stein’s paradox was an early motivation for multi-task learning (MTL) (Caruana, 1997), a learning paradigm in which data from multiple tasks is used with the hope to obtain superior performance over learning each task independently. Potential advantages of MTL go beyond the direct implications of Stein’s paradox, since even seemingly unrelated real world tasks have strong dependencies due to the shared processes that give rise to the data. For example, although autonomous driving and object manipulation are seemingly unrelated, the underlying data is governed by the same laws of optics, material properties, and dynamics. This motivates the use of multiple tasks as an inductive bias in learning systems.

A typical MTL system is given a collection of input points and sets of targets for various tasks per point. A common way to set up the inductive bias across tasks is to design a parametrized hypothesis class that shares some parameters across tasks. Typically, these parameters are learned by solving an optimization problem that minimizes a weighted sum of the empirical risk for each task. However, the linear-combination formulation is only sensible when there is a parameter set that is effective across all tasks. In other words, minimization of a weighted sum of empirical risk is only valid if tasks are not competing, which is rarely the case. MTL with conflicting objectives requires modeling of the trade-off between tasks, which is beyond what a linear combination achieves.

An alternative objective for MTL is finding solutions that are not dominated by any others. Such solutions are said to be Pareto optimal. In this paper, we cast the objective of MTL in terms of finding Pareto optimal solutions.

The problem of finding Pareto optimal solutions given multiple criteria is called multi-objective optimization. A variety of algorithms for multi-objective optimization exist. One such approach is the multiple-gradient descent algorithm (MGDA), which uses gradient-based optimization and provably converges to a point on the Pareto set (Désidéri, 2012). MGDA is well-suited for multi-task learning with deep networks. It can use the gradients of each task and solve an optimization problem to decide on an update over the shared parameters. However, there are two technical problems that hinder the applicability of MGDA on a large scale. (i) The underlying optimization problem does not scale gracefully to high-dimensional gradients, which arise naturally in deep networks. (ii) The algorithm requires explicit computation of gradients per task, which results in linear scaling of the number of backward passes and roughly multiplies the training time by the number of tasks.

In this paper, we develop a Frank-Wolfe-based optimizer that scales to high-dimensional problems. Furthermore, we provide an upper bound for the MGDA optimization objective and show that it can be computed via a single backward pass without explicit task-specific gradients, thus making the computational overhead of the method negligible. We prove that using our upper bound yields a Pareto optimal solution under realistic assumptions. The result is an exact algorithm for multi-objective optimization of deep networks with negligible computational overhead.

We empirically evaluate the presented method on three different problems. First, we perform an extensive evaluation on multi-digit classification with MultiMNIST (Sabour et al., 2017). Second, we cast multi-label classification as MTL and conduct experiments with the CelebA dataset (Liu et al., 2015b). Lastly, we apply the presented method to scene understanding; specifically, we perform joint semantic segmentation, instance segmentation, and depth estimation on the Cityscapes dataset (Cordts et al., 2016). The number of tasks in our evaluation varies from 2 to 40. Our method clearly outperforms all baselines.

## 2 Related Work

Multi-task learning. We summarize the work most closely related to ours and refer the interested reader to reviews by Ruder (2017) and Zhou et al. (2011b) for additional background. Multi-task learning (MTL) is typically conducted via hard or soft parameter sharing. In hard parameter sharing, a subset of the parameters is shared between tasks while other parameters are task-specific. In soft parameter sharing, all parameters are task-specific but they are jointly constrained via Bayesian priors (Xue et al., 2007; Bakker and Heskes, 2003) or a joint dictionary (Argyriou et al., 2007; Long and Wang, 2015; Yang and Hospedales, 2016; Ruder, 2017). We focus on hard parameter sharing with gradient-based optimization, following the success of deep MTL in computer vision (Bilen and Vedaldi, 2016; Misra et al., 2016; Rudd et al., 2016; Yang and Hospedales, 2016; Kokkinos, 2017; Zamir et al., 2018), natural language processing (Collobert and Weston, 2008; Dong et al., 2015; Liu et al., 2015a; Luong et al., 2015; Hashimoto et al., 2017), speech processing (Huang et al., 2013; Seltzer and Droppo, 2013; Huang et al., 2015), and even seemingly unrelated domains over multiple modalities (Kaiser et al., 2017).

Baxter (2000) theoretically analyze the MTL problem as interaction between individual learners and a meta-algorithm. Each learner is responsible for one task and a meta-algorithm decides how the shared parameters are updated. All aforementioned MTL algorithms use weighted summation as the meta-algorithm. Meta-algorithms that go beyond weighted summation have also been explored. Li et al. (2014) consider the case where each individual learner is based on kernel learning and utilize multi-objective optimization. Zhang and Yeung (2010) consider the case where each learner is a linear model and use a task affinity matrix. Zhou et al. (2011a) and Bagherjeiran et al. (2005) use the assumption that tasks share a dictionary and develop an expectation-maximization-like metaalgorithm. de Miranda et al. (2012) and Zhou et al. (2017b) use swarm optimization. None of these methods apply to gradient-based learning of high-capacity models such as modern deep networks. Kendall et al. (2018) and Chen et al. (2018) propose heuristics based on uncertainty and gradient magnitudes, respectively, and apply their methods to convolutional neural networks. Another recent work uses multi-agent reinforcement learning (Rosenbaum et al., 2017).

Multi-objective optimization. Multi-objective optimization addresses the problem of optimizing a set of possibly contrasting objectives. We recommend Miettinen (1998) and Ehrgott (2005) for surveys of this field. Of particular relevance to our work is gradient-based multi-objective optimization, as developed by Fliege and Svaiter (2000), Schäffler et al. (2002), and Désidéri (2012). These methods use multi-objective Karush-Kuhn-Tucker (KKT) conditions (Kuhn and Tucker, 1951) and find a descent direction that decreases all objectives. This approach was extended to stochastic gradient descent by Peitz and Dellnitz (2018) and Poirion et al. (2017). In machine learning, these methods have been applied to multi-agent learning (Ghosh et al., 2013; Pirotta and Restelli, 2016; Parisi et al., 2014), kernel learning (Li et al., 2014), sequential decision making (Roijers et al., 2013), and Bayesian optimization (Shah and Ghahramani, 2016; Hernández-Lobato et al., 2016). Our work applies gradient-based multi-objective optimization to multi-task learning.

## 3 Multi-Task Learning as Multi-Objective Optimization

Consider a multi-task learning (MTL) problem over an input space $\mathcal { X }$ and a collection of task spaces $\{ \mathcal { V } ^ { t } \} _ { t \in [ T ] }$ , such that a large dataset of i.i.d. data points $\mathbf { \bar { \{ x } }  _ { i } , \mathbf { \bar { { y } } } _ { i } ^ { 1 } , \dots , { \bf { { y } } } _ { i } ^ { T } \mathbf  \} _ { i \in [ N ] }$ is given where $T$ is the number of tasks, N is the number of data points, and $y _ { i } ^ { t }$ is the label of the $t ^ { \mathrm { { t h } } }$ task for the $i ^ { \mathrm { { t h } } }$ data point.<sup>1</sup> We further consider a parametric hypothesis class per task as $f ^ { t } ( \mathbf { x } ; \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) : \mathcal { X }  \mathcal { Y } ^ { t }$ such that some parameters $( \pmb \theta ^ { s h } )$ are shared between tasks and some $( \pmb \theta ^ { t } )$ are task-specific. We also consider task-specific loss functions $\mathcal { L } ^ { t } ( \cdot , \cdot ) : \mathcal { V } ^ { t } \times \mathcal { V } ^ { t }  \mathbb { R } ^ { + }$

Although many hypothesis classes and loss functions have been proposed in the MTL literature, they generally yield the following empirical risk minimization formulation:

$$
\min_ {\substack {\boldsymbol {\theta} ^ {s h}, \\ \boldsymbol {\theta} ^ {1}, \dots , \boldsymbol {\theta} ^ {T}}} \quad \sum_ {t = 1} ^ {T} c ^ {t} \hat {\mathcal {L}} ^ {t} (\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}) \tag{1}
$$

for some static or dynamically computed weights $c ^ { t }$ per task, where $\hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } )$ is the empirical loss of the task $t ,$ defined as $\begin{array} { r l } & { \hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) \triangleq \frac { 1 } { N } \sum _ { i } \mathcal { L } \big ( f ^ { t } ( \mathbf { x } _ { i } ; \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) , y _ { i } ^ { t } \big ) } \end{array}$

Although the weighted summation formulation (1) is intuitively appealing, it typically either requires an expensive grid search over various scalings or the use of a heuristic (Kendall et al., 2018; Chen et al., 2018). A basic justification for scaling is that it is not possible to define global optimality in the MTL setting. Consider two sets of solutions $\pmb { \theta }$ and $\bar { \pmb { \theta } }$ such that $\hat { \mathcal { L } } ^ { t _ { 1 } } ( { \pmb { \theta } } ^ { s h } , { \pmb { \theta } } ^ { t _ { 1 } } ) < \hat { \mathcal { L } } ^ { t _ { 1 } } ( \bar { \pmb { \theta } } ^ { s h } , \bar { \pmb { \theta } } ^ { t _ { 1 } } )$ and $\hat { \mathcal { L } } ^ { t _ { 2 } } ( { \pmb \theta } ^ { s h } , { \pmb \theta } ^ { t _ { 2 } } ) > \hat { \mathcal { L } } ^ { t _ { 2 } } ( \bar { \pmb \theta } ^ { s h } , \bar { \pmb \theta } ^ { t _ { 2 } } )$ , for some tasks $t _ { 1 }$ and $t _ { 2 }$ . In other words, solution $\pmb \theta$ is better for task $t _ { 1 }$ whereas $\pmb \theta$ is better for $t _ { 2 }$ . It is not possible to compare these two solutions without a pairwise importance of tasks, which is typically not available.

Alternatively, MTL can be formulated as multi-objective optimization: optimizing a collection of possibly conflicting objectives. This is the approach we take. We specify the multi-objective optimization formulation of MTL using a vector-valued loss L:

$$
\min_ {\substack {\boldsymbol {\theta} ^ {s h}, \\ \boldsymbol {\theta} ^ {1}, \dots , \dot {\boldsymbol {\theta}} ^ {T}}} \mathbf {L} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {1}, \dots , \boldsymbol {\theta} ^ {T}\right) = \min_ {\substack {\boldsymbol {\theta} ^ {s h}, \\ \boldsymbol {\theta} ^ {1}, \dots , \dot {\boldsymbol {\theta}} ^ {T}}} \left(\hat {\mathcal {L}} ^ {1} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {1}\right), \dots , \hat {\mathcal {L}} ^ {T} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {T}\right)\right) ^ {\intercal}. \tag{2}
$$

The goal of multi-objective optimization is achieving Pareto optimality.

Definition 1 (Pareto optimality for MTL)

(a) A solution θ dominates a solution θ<sup>¯</sup> $i f ~ \hat { \mathcal { L } } ^ { t } ( { \pmb { \theta } } ^ { s h } , { \pmb { \theta } } ^ { t } ) \leq \hat { \mathcal { L } } ^ { t } ( \bar { \pmb { \theta } } ^ { s h } , \bar { \pmb { \theta } } ^ { t } )$ for all tasks t and ${ \bf L } ( \theta ^ { s h } , \theta ^ { 1 } , \dots , \theta ^ { T } ) \neq { \bf L } ( \bar { \theta } ^ { s h } , \bar { \theta } ^ { 1 } , \dots , \bar { \theta } ^ { T } )$

(b) A solution $\pmb { \theta } ^ { \star }$ is called Pareto optimal ifthere exists no solution θ that dominates $\pmb { \theta } ^ { \star }$

The set of Pareto optimal solutions is called the Pareto set $( { \mathcal { P } } _ { \theta } )$ and its image is called the Pareto front $( \mathcal { P } _ { \mathbf { L } } = \{ { \mathbf { L } } ( \pmb { \theta } ) \} _ { \pmb { \theta } \in \mathcal { P } _ { \pmb { \theta } } } )$ . In this paper, we focus on gradient-based multi-objective optimization due to its direct relevance to gradient-based MTL.

In the rest of this section, we first summarize in Section 3.1 how multi-objective optimization can be performed with gradient descent. Then, we suggest in Section 3.2 a practical algorithm for performing multi-objective optimization over very large parameter spaces. Finally, in Section 3.3 we propose an efficient solution for multi-objective optimization designed directly for high-capacity deep networks. Our method scales to very large models and a high number of tasks with negligible overhead.

## 3.1 Multiple Gradient Descent Algorithm

As in the single-objective case, multi-objective optimization can be solved to local optimality via gradient descent. In this section, we summarize one such approach, called the multiple gradient descent algorithm (MGDA) (Désidéri, 2012). MGDA leverages the Karush-Kuhn-Tucker (KKT) conditions, which are necessary for optimality (Fliege and Svaiter, 2000; Schäffler et al., 2002; Désidéri, 2012). We now state the KKT conditions for both task-specific and shared parameters:

• There exist $\alpha ^ { 1 } , \ldots , \alpha ^ { T } \ge 0$ such that $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \alpha ^ { t } = 1 \mathrm { a n d } \sum _ { t = 1 } ^ { T } \alpha ^ { t } \nabla _ { \theta ^ { s h } } \hat { \mathcal { L } } ^ { t } ( \theta ^ { s h } , \theta ^ { t } ) = 0 } \end{array}$  
• For all tasks $t , \nabla _ { \pmb { \theta } ^ { t } } \hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) = 0$

Any solution that satisfies these conditions is called a Pareto stationary point. Although every Pareto optimal point is Pareto stationary, the reverse may not be true. Consider the optimization problem

$$
\left. \min _ {\alpha^ {1}, \dots , \alpha^ {T}} \left\{\left\| \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {t} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}\right) \right\| _ {2} ^ {2} \right\rvert \sum_ {t = 1} ^ {T} \alpha^ {t} = 1, \alpha^ {t} \geq 0 \quad \forall t \right\} \tag {3}
$$

Désidéri (2012) showed that either the solution to this optimization problem is 0 and the resulting point satisfies the KKT conditions, or the solution gives a descent direction that improves all tasks. Hence, the resulting MTL algorithm would be gradient descent on the task-specific parameters followed by solving (3) and applying the solution $\begin{array} { r l } {  { ( \sum _ { t = 1 } ^ { T } \alpha ^ { t } \nabla _ { \theta ^ { s h } } ) } } \end{array}$ as a gradient update to shared parameters. We discuss how to solve (3) for an arbitrary model in Section 3.2 and present an efficient solution when the underlying model is an encoder-decoder in Section 3.3.

## 3.2 Solving the Optimization Problem

The optimization problem defined in (3) is equivalent to finding a minimum-norm point in the convex hull of the set of input points. This problem arises naturally in computational geometry: it is equivalent to finding the closest point within a convex hull to a given query point. It has been studied extensively (Makimoto et al., 1994; Wolfe, 1976; Sekitani and Yamamoto, 1993). Although many algorithms have been proposed, they do not apply in our setting because the assumptions they make do not hold. Algorithms proposed in the computational geometry literature address the problem of finding minimum-norm points in the convex hull of a large number of points in a low-dimensional space (typically of dimensionality 2 or 3). In our setting, the number of points is the number of tasks and is typically low; in contrast, the dimensionality is the number of shared parameters and can be in the millions. We therefore use a different approach based on convex optimization, since (3) is a convex quadratic problem with linear constraints.

Before we tackle the general case, let’s consider the case of two tasks. The optimization problem can be defined as mi $\begin{array} { r } { 1 _ { \alpha \in [ 0 , 1 ] } \| \alpha \nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal { L } } ^ { 1 } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { 1 } ) + ( 1 - \alpha ) \nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal { L } } ^ { 2 } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { 2 } ) \| _ { 2 } ^ { 2 } } \end{array}$ , which is a onedimensional quadratic function of α with an analytical solution:

$$
\hat {\alpha} = \left[ \frac {\left(\nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {2} \left(\boldsymbol {\theta} ^ {s h} , \boldsymbol {\theta} ^ {2}\right) - \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {1} \left(\boldsymbol {\theta} ^ {s h} , \boldsymbol {\theta} ^ {1}\right)\right) ^ {\intercal} \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {2} \left(\boldsymbol {\theta} ^ {s h} , \boldsymbol {\theta} ^ {2}\right)}{\| \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {1} \left(\boldsymbol {\theta} ^ {s h} , \boldsymbol {\theta} ^ {1}\right) - \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {2} \left(\boldsymbol {\theta} ^ {s h} , \boldsymbol {\theta} ^ {2}\right) \| _ {2} ^ {2}} \right] _ {+, \frac {1}{\tau}} \tag {4}
$$

where $[ \cdot ] _ { + , \frac { 1 } { \tau } }$ represents clipping to $[ 0 , 1 ]$ as $[ a ] _ { + , \tiny { \frac { 1 } { \tau } } } = \operatorname* { m a x } ( \operatorname* { m i n } ( a , 1 ) , 0 )$ . We further visualize this solution in Figure 1. Although this is only applicable when $T = 2$ , this enables efficient application of the Frank-Wolfe algorithm (Jaggi, 2013) since the line search can be solved analytically. Hence, we use Frank-Wolfe to solve the constrained optimization problem, using (4) as a subroutine for the line search. We give all the update equations for the Frank-Wolfe solver in Algorithm 2.

![](images/fd187f5de6d9e31613d9fd5623e93576b6887c93b3d472c11cb151216c433871.jpg)  
Figure 1: Visualisation of the min-norm point in the convex hull of two points $\begin{array} { r l }  {  { ( \operatorname* { m i n } _ { \gamma \in [ 0 , 1 ] } \| \gamma \pmb \theta + ( 1 - \gamma ) \pmb \theta \| _ { 2 } ^ { 2 } ) } } \end{array}$ . As the geometry suggests, the solution is either an edge case or a perpendicular vector.

## Algorithm 1

$$
\min _ {\gamma \in [ 0, 1 ]} \| \gamma \boldsymbol {\theta} + (1 - \gamma) \bar {\boldsymbol {\theta}} \| _ {2} ^ {2}
$$

1: if θ<sup>|</sup>θ<sup>¯</sup> ≥ θ<sup>|</sup>θ then  
2: γ = 1  
3: else if θ<sup>|</sup>θ<sup>¯</sup> ≥ θ<sup>¯|</sup>θ<sup>¯</sup> then  
4: γ = 0  
5: else  
6: $\begin{array} { r } { \gamma = \frac { ( \bar { \theta } - \theta ) ^ { \intercal } \bar { \theta } } { | | \pmb { \theta } - \pmb { \theta } | | _ { 2 } ^ { 2 } } } \end{array}$  
7: end if

Algorithm 2 Update Equations for MTL

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
for $t = 1$ to $T$ do
    $\boldsymbol{\theta}^t = \boldsymbol{\theta}^t - \eta \nabla_{\boldsymbol{\theta}^t} \hat{\mathcal{L}}^t(\boldsymbol{\theta}^{sh}, \boldsymbol{\theta}^t)$ $\triangleright$ Gradient descent on task-specific parameters
end for
$\alpha^1, \ldots, \alpha^T = \text{FRANKWOLFESOLVER}(\boldsymbol{\theta})$ $\triangleright$ Solve (3) to find a common descent direction
$\boldsymbol{\theta}^{sh} = \boldsymbol{\theta}^{sh} - \eta \sum_{t=1}^{T} \alpha^t \nabla_{\boldsymbol{\theta}^{sh}} \hat{\mathcal{L}}^t(\boldsymbol{\theta}^{sh}, \boldsymbol{\theta}^t)$ $\triangleright$ Gradient descent on shared parameters

procedure FRANKWOLFESOLVER($\boldsymbol{\theta}$)
    Initialize $\boldsymbol{\alpha} = (\alpha^1, \ldots, \alpha^T) = (\frac{1}{T}, \ldots, \frac{1}{T})$
    Precompute M st. $\mathbf{M}_{i,j} = \left( \nabla_{\boldsymbol{\theta}^{sh}} \hat{\mathcal{L}}^i(\boldsymbol{\theta}^{sh}, \boldsymbol{\theta}^i) \right)^{\mathsf{T}} \left( \nabla_{\boldsymbol{\theta}^{sh}} \hat{\mathcal{L}}^j(\boldsymbol{\theta}^{sh}, \boldsymbol{\theta}^j) \right)$
    repeat
        $\hat{t} = \arg \min_r \sum_t \alpha^t \mathbf{M}_{rt}$ $\hat{\gamma} = \arg \min_\gamma ((1 - \gamma)\boldsymbol{\alpha} + \gamma \boldsymbol{e}_\hat{t})^\mathsf{T} \mathbf{M}((1 - \gamma)\boldsymbol{\alpha} + \gamma \boldsymbol{e}_\hat{t})$ $\triangleright$ Using Algorithm 1
        $\boldsymbol{\alpha} = (1 - \hat{\gamma})\boldsymbol{\alpha} + \hat{\gamma} \boldsymbol{e}_\hat{t}$
    until $\hat{\gamma} \sim 0$ or Number of Iterations Limit
    return $\alpha^1, \ldots, \alpha^T$
end procedure
</div>

## 3.3 Efficient Optimization for Encoder-Decoder Architectures

The MTL update described in Algorithm 2 is applicable to any problem that uses optimization based on gradient descent. Our experiments also suggest that the Frank-Wolfe solver is efficient and accurate as it typically converges in a modest number of iterations with negligible effect on training time. However, the algorithm we described needs to compute $\nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } )$ for each task t, which requires a backward pass over the shared parameters for each task. Hence, the resulting gradient computation would be the forward pass followed by $T$ backward passes. Considering the fact that computation of the backward pass is typically more expensive than the forward pass, this results in linear scaling of the training time and can be prohibitive for problems with more than a few tasks.

We now propose an efficient method that optimizes an upper bound of the objective and requires only a single backward pass. We further show that optimizing this upper bound yields a Pareto optimal solution under realistic assumptions. The architectures we address conjoin a shared representation function with task-specific decision functions. This class of architectures covers most of the existing deep MTL models and can be formally defined by constraining the hypothesis class as

$$
f ^ {t} (\mathbf {x}; \boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}) = (f ^ {t} (\cdot ; \boldsymbol {\theta} ^ {t}) \circ g (\cdot ; \boldsymbol {\theta} ^ {s h})) (\mathbf {x}) = f ^ {t} (g (\mathbf {x}; \boldsymbol {\theta} ^ {s h}); \boldsymbol {\theta} ^ {t}) \tag {5}
$$

where $g$ is the representation function shared by all tasks and $f ^ { t }$ are the task-specific functions that take this representation as input. If we denote the representations as ${ \bf Z } = ( { \bf z } _ { 1 } , \ldots , { \bf z } _ { N } )$ , where $\mathbf { z } _ { i } = g ( \mathbf { x } _ { i } ; \pmb { \theta } ^ { s h } )$ , we can state the following upper bound as a direct consequence of the chain rule:

$$
\left\| \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {t} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}\right) \right\| _ {2} ^ {2} \leq \left\| \frac {\partial \mathbf {Z}}{\partial \boldsymbol {\theta} ^ {s h}} \right\| _ {2} ^ {2} \left\| \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}\right) \right\| _ {2} ^ {2} \tag {6}
$$

where $\left\| \frac { \partial \mathbf Z } { \partial \pmb { \theta } ^ { s h } } \right\| _ { 2 }$ is the matrix norm of the Jacobian of Z with respect to $\pmb { \theta } ^ { s h }$ . Two desirable properties of this upper bound are that (i) $\nabla _ { \mathbf Z } \hat { \mathcal L } ^ { t } ( \pmb \theta ^ { s h } , \pmb \theta ^ { t } )$ can be computed in a single backward pass for all tasks and $( \operatorname { i i } ) \left\| { \frac { \partial \mathbf { Z } } { \partial \pmb { \theta } ^ { s h } } } \right\| _ { 2 } ^ { 2 }$ is not a function of $\alpha ^ { 1 } , \dots , \alpha ^ { T }$ , hence it can be removed when it is used as an optimization objective. We replace the $\begin{array} { r l } {  { \biggl \| \sum _ { t = 1 } ^ { T } \alpha ^ { t } \nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) \biggr \| _ { 2 } ^ { 2 } } \qquad } & { { } } \end{array}$ term with the upper bound we have just derived in order to obtain the approximate optimization problem and drop the $\left\| \frac { \partial \mathbf Z } { \partial \pmb { \theta } ^ { s h } } \right\| _ { 2 } ^ { 2 }$ term since it does not affect the optimization. The resulting optimization problem is

$$
\min _ {\alpha^ {1}, \dots , \alpha^ {T}} \left\{\left\| \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t} \left(\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}\right) \right\| _ {2} ^ {2} \mid \sum_ {t = 1} ^ {T} \alpha^ {t} = 1, \alpha^ {t} \geq 0 \quad \forall t \right\} \tag {MGDA-UB}
$$

We refer to this problem as MGDA-UB (Multiple Gradient Descent Algorithm – Upper Bound). In practice, MGDA-UB corresponds to using the gradients of the task losses with respect to the representations instead of the shared parameters. We use Algorithm 2 with only this change as the final method.

Although MGDA-UB is an approximation of the original optimization problem, we now state a theorem that shows that our method produces a Pareto optimal solution under mild assumptions. The proof is given in the supplement.

Theorem 1 Assume $\frac { \partial \mathbf { Z } } { \partial \theta ^ { s h } }$ is full-rank. $I f \alpha ^ { 1 , \dots , T }$ is the solution of MGDA-UB, one of the following is true:

(a) $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \alpha ^ { t } \nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal { L } } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) = 0 } \end{array}$ and the current parameters are Pareto stationary.  
(b) $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \alpha ^ { t } \nabla _ { \pmb { \theta } ^ { s h } } \hat { \mathcal L } ^ { t } ( \pmb { \theta } ^ { s h } , \pmb { \theta } ^ { t } ) } \end{array}$ is a descent direction that decreases all objectives.

This result follows from the fact that as long as $\frac { \partial \mathbf { Z } } { \partial \pmb { \theta } ^ { s h } }$ is full rank, optimizing the upper bound corresponds to minimizing the norm of the convex combination of the gradients using the Mahalonobis norm defined by $\frac { \partial \mathbf { Z } } { \partial \pmb { \theta } ^ { s h } } \overset { \intercal } { \frac { \partial \mathbf { Z } } { \partial \pmb { \theta } ^ { s h } } }$ . The non-singularity assumption is reasonable as singularity implies that tasks are linearly related and a trade-off is not necessary. In summary, our method provably finds a Pareto stationary point with negligible computational overhead and can be applied to any deep multi-objective problem with an encoder-decoder model.

## 4 Experiments

We evaluate the presented MTL method on a number of problems. First, we use MultiMNIST (Sabour et al., 2017), an MTL adaptation of MNIST (LeCun et al., 1998). Next, we tackle multi-label classification on the CelebA dataset (Liu et al., 2015b) by considering each label as a distinct binary classification task. These problems include both classification and regression, with the number of tasks ranging from 2 to 40. Finally, we experiment with scene understanding, jointly tackling the tasks of semantic segmentation, instance segmentation, and depth estimation on the Cityscapes dataset (Cordts et al., 2016). We discuss each experiment separately in the following subsections.

The baselines we consider are (i) uniform scaling: minimizing a uniformly weighted sum of loss functions ${ \frac { 1 } { T } } \sum _ { t } { \mathcal { L } } ^ { t }$ , (ii) single task: solving tasks independently, (iii) grid search: exhaustively trying various values from $\begin{array} { r } { \big \{ c ^ { t } \in [ 0 , 1 ] | \sum _ { t } c ^ { t } = 1 \big \} } \end{array}$ and optimizing for ${ \begin{array} { l } { { \frac { 1 } { T } } \sum _ { t } c ^ { t } { \mathcal { L } } ^ { t } } \end{array} }$ (iv) Kendall et al. (2018): using the uncertainty weighting proposed by Kendall et al. (2018), and (v) GradNorm: using the normalization proposed by Chen et al. (2018).

## 4.1 MultiMNIST

Our initial experiments are on MultiMNIST, an MTL version of the MNIST dataset (Sabour et al., 2017). In order to convert digit classification into a multi-task problem, Sabour et al. (2017) overlaid multiple images together. We use a similar construction. For each image, a different one is chosen uniformly in random. Then one of these images is put at the top-left and the other one is at the bottom-right. The resulting tasks are: classifying the digit on the top-left (task-L) and classifying the digit on the bottom-right (task-R). We use 60K examples and directly apply existing single-task MNIST models. The MultiMNIST dataset is illustrated in the supplement.

We use the LeNet architecture (LeCun et al., 1998). We treat all layers except the last as the representation function g and put two fully-connected layers as task-specific functions (see the supplement for details). We visualize the performance profile as a scatter plot of accuracies on task-L and task-R in Figure 3, and list the results in Table 3.

![](images/b6edccc3b0fed7442c535bd12a69bf93abb98a04a94cbe222a9e04e0c874f954.jpg)  
Figure 2: Radar charts of percentage error per attribute on CelebA (Liu et al., 2015b). Lower is better. We divide attributes into two sets for legibility: easy on the left, hard on the right. Zoom in for details.

In this setup, any static scaling results in lower accuracy than solving each task separately (the single task baseline). The two tasks appear to compete for model capacity, since increase in the accuracy of one task results in decrease in the accuracy of the other. Uncertainty weighting (Kendall et al., 2018) and GradNorm (Chen et al., 2018) find solutions that are slightly better than grid search but distinctly worse than the single-task baseline. In contrast, our method finds a solution that efficiently utilizes the model capacity and yields accuracies that are as good as the single-task solutions. This experiment demonstrates the effectiveness of our method as well as the necessity of treating MTL as multi-objective optimization. Even after a large hyper-parameter search, any scaling of tasks does not approach the effectiveness of our method.

## 4.2 Multi-Label Classification

Next, we tackle multi-label classification. Given a set of attributes, multi-label classification calls for deciding whether each attribute holds for the input. We use the CelebA dataset (Liu et al., 2015b), which includes 200K face images annotated with 40 attributes. Each attribute gives rise to a binary classification task and we cast this as a 40-way MTL problem. We use ResNet-18 (He et al., 2016) without the final layer as a shared representation function, and attach a linear layer for each attribute (see the supplement for further details).

We plot the resulting error for each binary classification task as a radar chart in Figure 2. The average over them is listed in Table 1. We skip grid search since it is not feasible over 40 tasks. Although uniform scaling is the norm in the multi-label classification literature, single-task performance is significantly better. Our method outperforms baselines for significant majority of tasks and achieves comparable performance in rest. This experiment also shows that our method remains effective when the number of tasks is high.

Table 1: Mean of error per category of MTL algorithms in multi-label classification on CelebA (Liu et al., 2015b).

<table><tr><td></td><td>Average error</td></tr><tr><td>Single task</td><td>8.77</td></tr><tr><td>Uniform scaling</td><td>9.62</td></tr><tr><td>Kendall et al. 2018</td><td>9.53</td></tr><tr><td>GradNorm</td><td>8.44</td></tr><tr><td>Ours</td><td>8.25</td></tr></table>

## 4.3 Scene Understanding

To evaluate our method in a more realistic setting, we use scene understanding. Given an RGB image, we solve three tasks: semantic segmentation (assigning pixel-level class labels), instance segmentation (assigning pixel-level instance labels), and monocular depth estimation (estimating continuous disparity per pixel). We follow the experimental procedure of Kendall et al. (2018) and use an encoder-decoder architecture. The encoder is based on ResNet-50 (He et al., 2016) and is shared by all three tasks. The decoders are task-specific and are based on the pyramid pooling module (Zhao et al., 2017) (see the supplement for further implementation details).

Table 2: Effect of the MGDA-UB approximation. We report the final accuracies as well as training times for our method with and without the approximation.

<table><tr><td rowspan="2"></td><td colspan="4">Scene understanding (3 tasks)</td><td colspan="2">Multi-label (40 tasks)</td></tr><tr><td>Training time</td><td>Segmentation mIoU [%]</td><td>Instance error [px]</td><td>Disparity error [px]</td><td>Training time (hour)</td><td>Average error</td></tr><tr><td>Ours (w/o approx.)</td><td>38.6</td><td>66.13</td><td>10.28</td><td>2.59</td><td>429.9</td><td>8.33</td></tr><tr><td>Ours</td><td>23.3</td><td>66.63</td><td>10.25</td><td>2.54</td><td>16.1</td><td>8.25</td></tr></table>

Since the output space of instance segmentation is unconstrained (the number of instances is not known in advance), we use a proxy problem as in Kendall et al. (2018). For each pixel, we estimate the location of the center of mass of the instance that encompasses the pixel. These center votes can then be clustered to extract the instances. In our experiments, we directly report the MSE in the proxy task. Figure 4 shows the performance profile for each pair of tasks, although we perform all experiments on all three tasks jointly. The pairwise performance profiles shown in Figure 4 are simply 2D projections of the three-dimensional profile, presented this way for legibility. The results are also listed in Table 4.

MTL outperforms single-task accuracy, indicating that the tasks cooperate and help each other. Our method outperforms all baselines on all tasks.

## 4.4 Role of the Approximation

In order to understand the role of the approximation proposed in Section 3.3, we compare the final performance and training time of our algorithm with and without the presented approximation in Table 2 (runtime measured on a single Titan Xp GPU). For a small number of tasks (3 for scene understanding), training time is reduced by 40%. For the multi-label classification experiment (40 tasks), the presented approximation accelerates learning by a factor of 25.

On the accuracy side, we expect both methods to perform similarly as long as the full-rank assumption is satisfied. As expected, the accuracy of both methods is very similar. Somewhat surprisingly, our approximation results in slightly improved accuracy in all experiments. While counter-intuitive at first, we hypothesize that this is related to the use of SGD in the learning algorithm. Stability analysis in convex optimization suggests that if gradients are computed with an error $\hat { \nabla } _ { \pmb { \theta } } \mathcal { L } ^ { t } = \nabla _ { \pmb { \theta } } \mathcal { L } ^ { t } + \mathbf { e } ^ { t }$ (θ corresponds to $\theta ^ { s h }$ in (3)), as opposed to Z in the approximate problem in (MGDA-UB), the error in the solution is bounded as $\| \hat { \boldsymbol { \alpha } } - \boldsymbol { \alpha } \| _ { 2 } \leq \mathcal { O } ( \operatorname* { m a x } _ { t } \| \dot { \mathbf { e } } ^ { t } \| _ { 2 } )$ . Considering the fact that the gradients are computed over the full parameter set (millions of dimensions) for the original problem and over a smaller space for the approximation (batch size times representation which is in the thousands), the dimension of the error vector is significantly higher in the original problem. We expect the $l _ { 2 }$ norm of such a random vector to depend on the dimension.

In summary, our quantitative analysis of the approximation suggests that (i) the approximation does not cause an accuracy drop and (ii) by solving an equivalent problem in a lower-dimensional space, our method achieves both better computational efficiency and higher stability.

## 5 Conclusion

We described an approach to multi-task learning. Our approach is based on multi-objective optimiza tion. In order to apply multi-objective optimization to MTL, we described an efficient algorithm as well as specific approximations that yielded a deep MTL algorithm with almost no computational overhead. Our experiments indicate that the resulting algorithm is effective for a wide range of multi-task scenarios.

![](images/3c10b3f6afd629e92719a62b2dc4953f49e91113896ac12a12806a8280979724.jpg)

<details>
<summary>scatter</summary>

| Series | Accuracy L | Accuracy R |
| --- | --- | --- |
| Grid Search | ~0.908 | ~0.959 |
| Grid Search | ~0.938 | ~0.957 |
| Grid Search | ~0.945 | ~0.956 |
| Grid Search | ~0.950 | ~0.955 |
| Grid Search | ~0.953 | ~0.955 |
| Grid Search | ~0.956 | ~0.952 |
| Grid Search | ~0.959 | ~0.951 |
| Grid Search | ~0.961 | ~0.951 |
| Grid Search | ~0.963 | ~0.949 |
| Grid Search | ~0.964 | ~0.948 |
| Grid Search | ~0.965 | ~0.947 |
| Grid Search | ~0.966 | ~0.946 |
| Grid Search | ~0.967 | ~0.944 |
| Grid Search | ~0.968 | ~0.940 |
| Grid Search | ~0.969 | ~0.935 |
| Grid Search | ~0.970 | ~0.930 |
| Grid Search | ~0.971 | ~0.911 |
| Grid Search | ~0.972 | ~0.894 |
| Uniform Scaling | ~0.964 | ~0.949 |
| Kendall et al. 2018 | ~0.965 | ~0.953 |
| GradNorm | ~0.964 | ~0.948 |
| Ours | ~0.973 | ~0.959 |
</details>

Figure 3: MultiMNIST accuracy profile. We plot the obtained accuracy in detecting the left and right digits for all baselines. The grid-search results suggest that the tasks compete for model capacity. Our method is the only one that finds a solution that is as good as training a dedicated model for each task. Top-right is better.

Table 3: Performance of MTL algorithms on MultiMNIST. Single-task baselines solve tasks separately, with dedicated models, but are shown in the same row for clarity.

<table><tr><td></td><td>Left digit accuracy [%]</td><td>Right digit accuracy [%]</td></tr><tr><td>Single task</td><td>97.23</td><td>95.90</td></tr><tr><td>Uniform scaling</td><td>96.46</td><td>94.99</td></tr><tr><td>Kendall et al. 2018</td><td>96.47</td><td>95.29</td></tr><tr><td>GradNorm</td><td>96.27</td><td>94.84</td></tr><tr><td>Ours</td><td>97.26</td><td>95.90</td></tr></table>

Table 4: Performance of MTL algorithms in joint semantic segmentation, instance segmentation, and depth estimation on Cityscapes. Singletask baselines solve tasks separately but are shown in the same row for clarity.

<table><tr><td></td><td>Segmentation mIoU [%]</td><td>Instance error [px]</td><td>Disparity error [px]</td></tr><tr><td>Single task</td><td>60.68</td><td>11.34</td><td>2.78</td></tr><tr><td>Uniform scaling</td><td>54.59</td><td>10.38</td><td>2.96</td></tr><tr><td>Kendall et al. 2018</td><td>64.21</td><td>11.54</td><td>2.65</td></tr><tr><td>GradNorm</td><td>64.81</td><td>11.31</td><td>2.57</td></tr><tr><td>Ours</td><td>66.63</td><td>10.25</td><td>2.54</td></tr></table>

![](images/9b5c3ce650bd2b497fcfdb16723c20cd295f24893f889edd2126e4b8e9c72141.jpg)  
Figure 4: Cityscapes performance profile. We plot the performance of all baselines for the tasks of semantic segmentation, instance segmentation, and depth estimation. We use mIoU for semantic segmentation, error of per-pixel regression (normalized to image size) for instance segmentation, and disparity error for depth estimation. To convert errors to performance measures, we use 1 − instance error and 1/disparity error. We plot 2D projections of the performance profile for each pair of tasks. Although we plot pairwise projections for visualization, each point in the plots solves all tasks. Top-right is better.

## References

A. Argyriou, T. Evgeniou, and M. Pontil. Multi-task feature learning. In NIPS, 2007.  
A. Bagherjeiran, R. Vilalta, and C. F. Eick. Content-based image retrieval through a multi-agent meta-learning framework. In International Conference on Tools with Artificial Intelligence, 2005.  
B. Bakker and T. Heskes. Task clustering and gating for Bayesian multitask learning. JMLR, 4:83–99, 2003.  
J. Baxter. A model of inductive bias learning. Journal of Artificial Intelligence Research, 12:149–198, 2000.  
H. Bilen and A. Vedaldi. Integrated perception with recurrent multi-task neural networks. In NIPS, 2016.  
R. Caruana. Multitask learning. Machine Learning, 28(1):41–75, 1997.  
Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML, 2018.  
R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In ICML, 2008.  
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.  
P. B. C. de Miranda, R. B. C. Prudêncio, A. C. P. L. F. de Carvalho, and C. Soares. Combining a multi-objective optimization approach with meta-learning for SVM parameter selection. In International Conference on Systems, Man, and Cybernetics, 2012.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.  
J.-A. Désidéri. Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. Comptes Rendus Mathematique, 350(5):313–318, 2012.  
D. Dong, H. Wu, W. He, D. Yu, and H. Wang. Multi-task learning for multiple language translation. In ACL, 2015.  
M. Ehrgott. Multicriteria Optimization (2. ed.). Springer, 2005.  
J. Fliege and B. F. Svaiter. Steepest descent methods for multicriteria optimization. Mathematical Methods of Operations Research, 51(3):479–494, 2000.  
S. Ghosh, C. Lovell, and S. R. Gunn. Towards Pareto descent directions in sampling experts for multiple tasks in an on-line learning paradigm. In AAAI Spring Symposium: Lifelong Machine Learning, 2013.  
K. Hashimoto, C. Xiong, Y. Tsuruoka, and R. Socher. A joint many-task model: Growing a neural network for multiple NLP tasks. In EMNLP, 2017.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.  
D. Hernández-Lobato, J. M. Hernández-Lobato, A. Shah, and R. P. Adams. Predictive entropy search for multi-objective bayesian optimization. In ICML, 2016.  
J.-T. Huang, J. Li, D. Yu, L. Deng, and Y. Gong. Cross-language knowledge transfer using multilingual deep neural network with shared hidden layers. In ICASSP, 2013.  
Z. Huang, J. Li, S. M. Siniscalchi, I.-F. Chen, J. Wu, and C.-H. Lee. Rapid adaptation for deep neural networks through multi-task learning. In Interspeech, 2015.  
M. Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In ICML, 2013.  
L. Kaiser, A. N. Gomez, N. Shazeer, A. Vaswani, N. Parmar, L. Jones, and J. Uszkoreit. One model to learn them all. arXiv:1706.05137, 2017.  
A. Kendall, Y. Gal, and R. Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, 2018.  
I. Kokkinos. UberNet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In CVPR, 2017.  
H. W. Kuhn and A. W. Tucker. Nonlinear programming. In Proceedings ofthe Second Berkeley Symposium on Mathematical Statistics and Probability, Berkeley, Calif., 1951. University of California Press.  
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.  
C. Li, M. Georgiopoulos, and G. C. Anagnostopoulos. Pareto-path multi-task multiple kernel learning. arXiv:1404.3190, 2014.  
X. Liu, J. Gao, X. He, L. Deng, K. Duh, and Y.-Y. Wang. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In NAACL HLT, 2015a.  
Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In ICCV, 2015b.  
M. Long and J. Wang. Learning multiple tasks with deep relationship networks. arXiv:1506.02117, 2015.  
M.-T. Luong, Q. V. Le, I. Sutskever, O. Vinyals, and L. Kaiser. Multi-task sequence to sequence learning. arXiv:1511.06114, 2015.  
N. Makimoto, I. Nakagawa, and A. Tamura. An efficient algorithm for finding the minimum norm point in the convex hull of a finite point set in the plane. Operations Research Letters, 16(1):33–40, 1994.  
K. Miettinen. Nonlinear Multiobjective Optimization. Springer, 1998.  
I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Cross-stitch networks for multi-task learning. In CVPR, 2016.  
S. Parisi, M. Pirotta, N. Smacchia, L. Bascetta, and M. Restelli. Policy gradient approaches for multi-objective sequential decision making. In IJCNN, 2014.  
A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. In NIPS Workshops, 2017  
S. Peitz and M. Dellnitz. Gradient-based multiobjective optimization with uncertainties. In NEO, 2018.  
M. Pirotta and M. Restelli. Inverse reinforcement learning through policy gradient minimization. In AAAI, 2016.  
F. Poirion, Q. Mercier, and J. Désidéri. Descent algorithm for nonsmooth stochastic multiobjective optimization. Computational Optimization and Applications, 68(2):317–331, 2017.  
D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley. A survey of multi-objective sequential decision-making. Journal ofArtificial Intelligence Research, 48:67–113, 2013.  
C. Rosenbaum, T. Klinger, and M. Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv:1711.01239, 2017.  
E. M. Rudd, M. Günther, and T. E. Boult. MOON: A mixed objective optimization network for the recognition of facial attributes. In ECCV, 2016.  
S. Ruder. An overview of multi-task learning in deep neural networks. arXiv:1706.05098, 2017.  
S. Sabour, N. Frosst, and G. E. Hinton. Dynamic routing between capsules. In NIPS, 2017.  
S. Schäffler, R. Schultz, and K. Weinzierl. Stochastic method for the solution of unconstrained vector optimization problems. Journal of Optimization Theory and Applications, 114(1):209–222, 2002.  
K. Sekitani and Y. Yamamoto. A recursive algorithm for finding the minimum norm point in a polytope and a pair of closest points in two polytopes. Mathematical Programming, 61(1-3):233–249, 1993.  
M. L. Seltzer and J. Droppo. Multi-task learning in deep neural networks for improved phoneme recognition. In ICASSP, 2013.  
A. Shah and Z. Ghahramani. Pareto frontier learning with expensive correlated objectives. In ICML, 2016.  
C. Stein. Inadmissibility of the usual estimator for the mean of a multivariate normal distribution. Technical report, Stanford University, US, 1956.  
P. Wolfe. Finding the nearest point in a polytope. Mathematical Programming, 11(1):128–149, 1976.  
Y. Xue, X. Liao, L. Carin, and B. Krishnapuram. Multi-task learning for classification with dirichlet process priors. JMLR, 8:35–63, 2007.  
Y. Yang and T. M. Hospedales. Trace norm regularised deep multi-task learning. arXiv:1606.04038, 2016.  
A. R. Zamir, A. Sax, W. B. Shen, L. J. Guibas, J. Malik, and S. Savarese. Taskonomy: Disentangling task transfer learning. In CVPR, 2018.  
Y. Zhang and D. Yeung. A convex formulation for learning task relationships in multi-task learning. In UAI, 2010.  
H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. In CVPR, 2017.  
B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba. Scene parsing through ADE20K dataset. In CVPR, 2017a.  
D. Zhou, J. Wang, B. Jiang, H. Guo, and Y. Li. Multi-task multi-view learning based on cooperative multiobjective optimization. IEEE Access, 2017b.  
J. Zhou, J. Chen, and J. Ye. Clustered multi-task learning via alternating structure optimization. In NIPS, 2011a.  
J. Zhou, J. Chen, and J. Ye. MALSAR: Multi-task learning via structural regularization. Arizona State University, 2011b.

## A Proof of Theorem 1

Proof. We begin by showing that if the optimum value of (MGDA-UB) is 0, so is the optimum value of (3). This shows the first case of the theorem. Then, we will show the second part.

If the optimum value of (MGDA-UB) is 0,

$$
\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\boldsymbol {\theta} ^ {s h}} \hat {\mathcal {L}} ^ {t} (\boldsymbol {\theta} ^ {s h}, \boldsymbol {\theta} ^ {t}) = \frac {\partial \mathbf {Z}}{\partial \theta^ {s h}} \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t} = \sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\theta^ {s h}} \hat {\mathcal {L}} ^ {t} = 0 \tag {7}
$$

Hence $\alpha ^ { 1 } , \dots , \alpha ^ { T }$ is the solution of (3) and the optimal value of (3) is 0. This proves the first case of the theorem. Before we move to the second case, we state a straightforward corollary. Since $\frac { \partial \mathbf { Z } } { \partial \theta ^ { s h } }$ is full rank, this equivalence is bi-directional. In other words, if $\alpha ^ { 1 } , \dots , \alpha ^ { T }$ is the solution of (3), it is the solution of (MGDA-UB) as well. Hence, both formulations completely agree on Pareto stationarity.

In order to prove the second case, we need to show that the resulting descent direction computed by solving (MGDA-UB) does not increase any of the loss functions. Formally, we need to show that

$$
\left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\theta^ {s h}} \hat {\mathcal {L}} ^ {t}\right) ^ {\intercal} \left(\nabla_ {\theta^ {s h}} \hat {\mathcal {L}} ^ {t ^ {\prime}}\right) \geq 0 \quad \forall t ^ {\prime} \in \{1, \dots , T \} \tag {8}
$$

This condition is equivalent to

$$
\left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) ^ {\intercal} \mathbf {M} \left(\nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t ^ {\prime}}\right) \geq 0 \quad \forall t ^ {\prime} \in \{1, \dots , T \} \tag {9}
$$

where $\begin{array} { r } { \mathbf { M } = \left( \frac { \partial \mathbf { Z } } { \partial \theta ^ { s h } } \right) ^ { \mathsf { T } } \left( \frac { \partial \mathbf { Z } } { \partial \theta ^ { s h } } \right) } \end{array}$ . Since M is positive definite (following the assumption), this is further equivalent to

$$
\left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) ^ {\intercal} \left(\nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t ^ {\prime}}\right) \geq 0 \quad \forall t ^ {\prime} \in \{1, \dots , T \} \tag {10}
$$

We show that this follows from the optimality conditions for (MGDA-UB). The Lagrangian of (MGDA-UB) is

$$
\left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) ^ {\intercal} \left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) - \lambda \left(\sum_ {i} \alpha^ {i} - 1\right) \text {where} \lambda \geq 0. \tag {11}
$$

The KKT condition for this Lagrangian yields the desired result as

$$
\left(\sum_ {t = 1} ^ {T} \alpha^ {t} \nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) ^ {\intercal} \left(\nabla_ {\mathbf {Z}} \hat {\mathcal {L}} ^ {t}\right) = \frac {\lambda}{2} \geq 0 \tag {12}
$$

![](images/e67924f97a0753f37b8ab94a6b25f6ab89fd363b6e48edf1ab493f263194830c.jpg)

## B Additional Results on Multi-label Classification

In this section, we present the experimental results we did not include in the main text.

In the main text, we plotted a radar chart of the binary attribute classification errors. However, we did not include the tabulated results due to the space limitations. Here we list the binary classification error of each attribute for each algorithm in Table 5.

Table 5: Multi-label classification error per attribute for all algorithms.

<table><tr><td></td><td>Uniform scaling</td><td>Single task</td><td>Kendall et al.</td><td>Grad Norm</td><td>Ours</td><td></td><td>Uniform scaling</td><td>Single task</td><td>Kendall et al.</td><td>Grad Norm</td><td>Ours</td></tr><tr><td>Attr. 0</td><td>7.11</td><td>7.16</td><td>7.18</td><td>6.54</td><td>6.17</td><td>Attr. 5</td><td>4.91</td><td>4.75</td><td>4.95</td><td>4.19</td><td>4.13</td></tr><tr><td>Attr. 1</td><td>17.30</td><td>14.38</td><td>16.77</td><td>14.80</td><td>14.87</td><td>Attr. 6</td><td>20.97</td><td>14.24</td><td>15.17</td><td>14.07</td><td>14.08</td></tr><tr><td>Attr. 2</td><td>20.99</td><td>19.25</td><td>20.56</td><td>18.97</td><td>18.35</td><td>Attr. 7</td><td>18.53</td><td>17.74</td><td>18.84</td><td>17.33</td><td>17.25</td></tr><tr><td>Attr. 3</td><td>17.82</td><td>16.79</td><td>18.45</td><td>16.47</td><td>16.06</td><td>Attr. 8</td><td>10.22</td><td>8.87</td><td>10.19</td><td>8.67</td><td>8.42</td></tr><tr><td>Attr. 4</td><td>1.25</td><td>1.20</td><td>1.17</td><td>1.13</td><td>1.08</td><td>Attr. 9</td><td>5.29</td><td>5.09</td><td>5.44</td><td>4.68</td><td>4.60</td></tr><tr><td>Attr. 10</td><td>4.14</td><td>4.02</td><td>4.33</td><td>3.77</td><td>3.60</td><td>Attr. 15</td><td>0.81</td><td>0.52</td><td>0.62</td><td>0.56</td><td>0.56</td></tr><tr><td>Attr. 11</td><td>16.22</td><td>15.34</td><td>16.64</td><td>14.73</td><td>14.56</td><td>Attr. 16</td><td>4.00</td><td>3.94</td><td>3.99</td><td>3.72</td><td>3.46</td></tr><tr><td>Attr. 12</td><td>8.42</td><td>7.68</td><td>8.85</td><td>7.23</td><td>7.41</td><td>Attr. 17</td><td>2.39</td><td>2.66</td><td>2.35</td><td>2.09</td><td>2.16</td></tr><tr><td>Attr. 13</td><td>5.17</td><td>5.15</td><td>5.26</td><td>4.75</td><td>4.52</td><td>Attr. 18</td><td>8.79</td><td>9.01</td><td>8.84</td><td>8.00</td><td>7.83</td></tr><tr><td>Attr. 14</td><td>4.14</td><td>4.13</td><td>4.17</td><td>3.73</td><td>3.54</td><td>Attr. 19</td><td>13.78</td><td>12.27</td><td>13.86</td><td>11.79</td><td>11.29</td></tr><tr><td>Attr. 20</td><td>1.61</td><td>1.61</td><td>1.58</td><td>1.42</td><td>1.43</td><td>Attr. 25</td><td>27.59</td><td>24.82</td><td>26.94</td><td>24.26</td><td>23.87</td></tr><tr><td>Attr. 21</td><td>7.18</td><td>6.20</td><td>7.73</td><td>6.91</td><td>6.26</td><td>Attr. 26</td><td>3.54</td><td>3.40</td><td>3.78</td><td>3.22</td><td>3.16</td></tr><tr><td>Attr. 22</td><td>4.38</td><td>4.14</td><td>4.08</td><td>3.88</td><td>3.81</td><td>Attr. 27</td><td>26.74</td><td>22.74</td><td>26.21</td><td>23.12</td><td>22.45</td></tr><tr><td>Attr. 23</td><td>8.32</td><td>6.57</td><td>8.80</td><td>6.54</td><td>6.47</td><td>Attr. 28</td><td>6.14</td><td>5.82</td><td>6.17</td><td>5.43</td><td>5.16</td></tr><tr><td>Attr. 24</td><td>5.01</td><td>5.38</td><td>5.12</td><td>4.63</td><td>4.23</td><td>Attr. 29</td><td>5.55</td><td>5.18</td><td>5.40</td><td>5.13</td><td>4.87</td></tr><tr><td>Attr. 30</td><td>3.29</td><td>3.79</td><td>3.24</td><td>2.94</td><td>3.03</td><td>Attr. 35</td><td>1.15</td><td>1.13</td><td>1.08</td><td>0.94</td><td>1.08</td></tr><tr><td>Attr. 31</td><td>8.05</td><td>7.18</td><td>8.40</td><td>7.21</td><td>6.92</td><td>Attr. 36</td><td>7.91</td><td>7.56</td><td>8.06</td><td>7.47</td><td>7.18</td></tr><tr><td>Attr. 32</td><td>18.21</td><td>17.25</td><td>18.15</td><td>15.93</td><td>15.93</td><td>Attr. 37</td><td>13.27</td><td>11.90</td><td>13.47</td><td>11.61</td><td>11.19</td></tr><tr><td>Attr. 33</td><td>16.53</td><td>15.55</td><td>16.19</td><td>13.93</td><td>13.80</td><td>Attr. 38</td><td>3.80</td><td>3.29</td><td>4.04</td><td>3.57</td><td>3.51</td></tr><tr><td>Attr. 34</td><td>11.12</td><td>9.76</td><td>11.46</td><td>10.17</td><td>9.73</td><td>Attr. 39</td><td>13.25</td><td>13.40</td><td>13.78</td><td>12.26</td><td>11.95</td></tr></table>

## C Implementation Details

## C.1 MultiMNIST

We use the MultiMNIST dataset, which overlays multiple images together (Sabour et al., 2017). For each image, a different one is chosen uniformly in random. One of these images is placed at the top-left and the other at the bottom-right. We show sample MultiMNIST images in Figure 6.

![](images/bbcaf9d79dd0a71f8ac965954542d63d0801b145eda99cd0908e0a92cf671a54.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  InputImage["Input Image\n32 x 32"] --> ConvLayer1["Conv (Kernel: 5, Filter: 10)"]
  ConvLayer1 --> ReLU["ReLU"]
  ReLU --> MaxPool1["Max Pool (2 x 2)"]
  MaxPool1 --> ConvLayer2["Conv (Kernel: 5, Filter: 20)"]
  ConvLayer2 --> ReLU2["ReLU"]
  ReLU2 --> MaxPool2["Max Pool (2 x 2)"]
  MaxPool2 --> FullyConnectedLayer["Fully Connected Layer (320 x 50)"]
  FullyConnectedLayer --> LeftDigit["Left Digit (10 class)"]
  FullyConnectedLayer --> RightDigit["Right Digit (10 class)"]
```
</details>

Figure 5: Architecture used for MultiMNIST experiments.

For the MultiMNIST experiments, we use an architecture based on LeNet (LeCun et al., 1998). We use all layers except the final one as a shared encoder. We use the fully-connected layer as a task-specific function for the left and right tasks by simply adding two independent fully-connected layers, each taking the output of the shared encoder as input. As a task-specific loss function, we use the cross-entropy loss with a softmax for both tasks. The architecture is visualized in Figure 5.

The implementation uses PyTorch (Paszke et al., 2017). For all baselines, we searched over the set $L R = \{ 1 \mathrm { e } { - } 4 , 5 \mathrm { e } { - } 4 , 1 \mathrm { e } { - } 3 , 5 \mathrm { e } { - } 3 , 1 \mathrm { e } { - } 2 , 5 \mathrm { e } { - } 2 \}$ of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 256 and train for 100 epochs. We report test accuracy.

## C.2 Multi-label classification

For multi-label classification experiments, we use ResNet-18 (He et al., 2016) without the final layer as a shared representation function. Since there are 40 attributes, we add 40 separate $2 0 4 8 \times 2$ dimensional fully-connected layers as task-specific functions. The final two-dimensional output is passed through a 2-class softmax to get binary attribute classification probabilities. We use cross-entropy as a task-specific loss. The architecture is visualized in Figure 7.

The implementation uses PyTorch (Paszke et al., 2017). We resize each CelebA image (Liu et al., 2015b) to 64 × 64 × 3. For all experiments, we searched over the set $\begin{array} { r l } { L R } & { { } = } \end{array}$ $\{ 1 \mathrm { e } { - } 4 , 5 \mathrm { e } { - } 4 , 1 \mathrm { e } { - } 3 , 5 \mathrm { e } { - } 3 , 1 \mathrm { e } { - } 2 , 5 \mathrm { e } { - } 2 \}$ of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 256 and train for 100 epochs. We report attribute-wise binary accuracies on the test set as well as the average accuracy.

## C.3 Scene understanding

For scene understanding experiments, we use the Cityscapes dataset (Cordts et al., 2016). We resize all images to resolution $2 5 6 \times 5 1 2$ for computational efficiency. As a shared representation function (encoder), we use the ResNet-50 architecture (He et al., 2016) in fully-convolutional fashion. We take the ResNet-50 architecture and only use layers prior to average pooling that are fully convolutional. As a decoder, we use the pyramid pooling module (Zhao et al., 2017) and set the output sizes to $2 5 6 \times 5 1 2 \times 1 9$ for semantic segmentation (19 classes), $2 5 6 \times 5 1 2 \times 2$ for instance segmentation (one output channel for the x-offset of the center location and another channel for the y-offset), and $2 5 6 \times 5 1 2 \times 1$ for monocular depth estimation. For instance segmentation, we use the proxy task of estimating the offset for the center location of the instance that encompasses the pixel. We directly estimate disparity instead of depth and later convert it to depth using the provided camera intrinsics. As a loss function, we use cross-entropy with a softmax for semantic segmentation, and MSE for depth and instance segmentation. We visualize the architecture in Figure 8.

We initialize the encoder with a model pretrained on ImageNet (Deng et al., 2009). We use the implementation of the pyramidal pooling network with bilinear interpolation shared by Zhou et al. (2017a). Ground-truth results for the Cityscapes test set are not publicly available. Therefore, we report numbers on the validation set. As a validation set for hyperparameter search, we randomly choose 275 images from the training set. After the best hyperparameters are chosen, we retrain with the full training set and report the metrics on the Cityscapes validation set, which our algorithm never sees during training or hyperparameter search. As metrics, we use mean intersection over union (mIoU) for semantic segmentation, MSE for instance segmentation, and MSE for disparities (depth estimation). We directly report the metric in the proxy task for instance segmentation instead of performing a further clustering operation. For all experiments, we searched over the set LR = {1e−4, 5e−4, 1e−3, 5e−3, 1e−2, 5e−2} of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 8 and train for 250 epochs.

![](images/dd24507da83e117059a209cf353390a7276365313f2a04cce40682516329a37c.jpg)  
Figure 6: Sample MultiMNIST images. In each image, one task (task-L) is classifying the digit on the top-left and the second task (task-R) is classifying the digit on the bottom-right.

![](images/f7f37e61b4edebca8087f45b85412c2ec4a9adefd39307c24a3c0ccb67d84a59.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  InputImage["Input Image\n64 x 64 x 3"] --> ResNet18["ResNet 18:\nResNet 18 without the final fully connected layer."]
  ResNet18 --> FullyConnectedLayer["Fully Connected Layer (2048 x 2)"]
  ResNet18 --> Attribute0["Attribute 0: (binary)"]
  ResNet18 --> Attribute39["Attribute 39: (binary)"]
  Attribute0 -.-> FullyConnectedLayer
  Attribute39 -.-> FullyConnectedLayer
```
</details>

Figure 7: Architecture used for multi-label classification experiments.

![](images/db2fb2a5d3142862a6b7e67a8bb6dae9ca697158387a2f565ac54f4b13b0fceb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  InputImage["Input Image\n256 x 512 x 3"] --> ConvNet["Convolutional ResNet 50:\nResNet 50 without final average poling and fully connected layer."]
  ConvNet --> Pool1["Pyramid Pooling Network"]
  ConvNet --> Pool2["Pyramid Pooling Network"]
  ConvNet --> Pool3["Pyramid Pooling Network"]
  Pool1 --> Seg1["Semantic Segmentation\n(256 x 512 x 19)"]
  Pool2 --> Seg2["Instance Segmentation\n(256 x 512 x 2)"]
  Pool3 --> Seg3["Disparity Estimation\n(256 x 512 x 1)"]
```
</details>

Figure 8: Architecture used for scene understanding experiments.