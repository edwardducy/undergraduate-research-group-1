# Multi-Task Learning as Multi-Objective Optimization

 Ozan Sener Affiliation: Intel Labs    Vladlen Koltun Affiliation: Intel Labs 

###### Abstract

In multi-task learning, multiple tasks are solved jointly, sharing inductive bias between them. Multi-task learning is inherently a multi-objective problem because different tasks may conflict, necessitating a trade-off. A common compromise is to optimize a proxy objective that minimizes a weighted linear combination of per-task losses. However, this workaround is only valid when the tasks do not compete, which is rarely the case. In this paper, we explicitly cast multi-task learning as multi-objective optimization, with the overall objective of finding a Pareto optimal solution. To this end, we use algorithms developed in the gradient-based multi-objective optimization literature. These algorithms are not directly applicable to large-scale learning problems since they scale poorly with the dimensionality of the gradients and the number of tasks. We therefore propose an upper bound for the multi-objective loss and show that it can be optimized efficiently. We further prove that optimizing this upper bound yields a Pareto optimal solution under realistic assumptions. We apply our method to a variety of multi-task deep learning problems including digit classification, scene understanding (joint semantic segmentation, instance segmentation, and depth estimation), and multi-label classification. Our method produces higher-performing models than recent multi-task learning formulations or per-task training.

## 1 Introduction

One of the most surprising results in statistics is Stein’s paradox. [Stein (1956)](#bib.bib50 "") showed that it is better to estimate the means of three or more Gaussian random variables using samples from all of them rather than estimating them separately, even when the Gaussians are independent. Stein’s paradox was an early motivation for multi-task learning (MTL) ([Caruana, 1997](#bib.bib6 "")), a learning paradigm in which data from multiple tasks is used with the hope to obtain superior performance over learning each task independently. Potential advantages of MTL go beyond the direct implications of Stein’s paradox, since even seemingly unrelated real world tasks have strong dependencies due to the shared processes that give rise to the data. For example, although autonomous driving and object manipulation are seemingly unrelated, the underlying data is governed by the same laws of optics, material properties, and dynamics. This motivates the use of multiple tasks as an inductive bias in learning systems.

A typical MTL system is given a collection of input points and sets of targets for various tasks per point. A common way to set up the inductive bias across tasks is to design a parametrized hypothesis class that shares some parameters across tasks. Typically, these parameters are learned by solving an optimization problem that minimizes a weighted sum of the empirical risk for each task. However, the linear-combination formulation is only sensible when there is a parameter set that is effective across all tasks. In other words, minimization of a weighted sum of empirical risk is only valid if tasks are not competing, which is rarely the case. MTL with conflicting objectives requires modeling of the trade-off between tasks, which is beyond what a linear combination achieves.

An alternative objective for MTL is finding solutions that are not dominated by any others. Such solutions are said to be Pareto optimal. In this paper, we cast the objective of MTL in terms of finding Pareto optimal solutions.

The problem of finding Pareto optimal solutions given multiple criteria is called multi-objective optimization. A variety of algorithms for multi-objective optimization exist. One such approach is the multiple-gradient descent algorithm (MGDA), which uses gradient-based optimization and provably converges to a point on the Pareto set ([Désidéri, 2012](#bib.bib12 "")). MGDA is well-suited for multi-task learning with deep networks. It can use the gradients of each task and solve an optimization problem to decide on an update over the shared parameters. However, there are two technical problems that hinder the applicability of MGDA on a large scale. (i) The underlying optimization problem does not scale gracefully to high-dimensional gradients, which arise naturally in deep networks. (ii) The algorithm requires explicit computation of gradients per task, which results in linear scaling of the number of backward passes and roughly multiplies the training time by the number of tasks.

In this paper, we develop a Frank-Wolfe-based optimizer that scales to high-dimensional problems. Furthermore, we provide an upper bound for the MGDA optimization objective and show that it can be computed via a single backward pass without explicit task-specific gradients, thus making the computational overhead of the method negligible. We prove that using our upper bound yields a Pareto optimal solution under realistic assumptions. The result is an exact algorithm for multi-objective optimization of deep networks with negligible computational overhead.

We empirically evaluate the presented method on three different problems. First, we perform an extensive evaluation on multi-digit classification with MultiMNIST ([Sabour et al., 2017](#bib.bib45 "")). Second, we cast multi-label classification as MTL and conduct experiments with the CelebA dataset ([Liu et al., 2015b](#bib.bib30 "")). Lastly, we apply the presented method to scene understanding; specifically, we perform joint semantic segmentation, instance segmentation, and depth estimation on the Cityscapes dataset ([Cordts et al., 2016](#bib.bib9 "")). The number of tasks in our evaluation varies from 2 to 40. Our method clearly outperforms all baselines.

## 2 Related Work

Multi-task learning. We summarize the work most closely related to ours and refer the interested reader to reviews by [Ruder (2017)](#bib.bib44 "") and [Zhou et al. (2011b)](#bib.bib60 "") for additional background. Multi-task learning (MTL) is typically conducted via hard or soft parameter sharing. In hard parameter sharing, a subset of the parameters is shared between tasks while other parameters are task-specific. In soft parameter sharing, all parameters are task-specific but they are jointly constrained via Bayesian priors ([Xue et al., 2007](#bib.bib52 ""); [Bakker and Heskes, 2003](#bib.bib3 "")) or a joint dictionary ([Argyriou et al., 2007](#bib.bib1 ""); [Long and Wang, 2015](#bib.bib31 ""); [Yang and Hospedales, 2016](#bib.bib53 ""); [Ruder, 2017](#bib.bib44 "")). We focus on hard parameter sharing with gradient-based optimization, following the success of deep MTL in computer vision ([Bilen and Vedaldi, 2016](#bib.bib5 ""); [Misra et al., 2016](#bib.bib35 ""); [Rudd et al., 2016](#bib.bib43 ""); [Yang and Hospedales, 2016](#bib.bib53 ""); [Kokkinos, 2017](#bib.bib25 ""); [Zamir et al., 2018](#bib.bib54 "")), natural language processing ([Collobert and Weston, 2008](#bib.bib8 ""); [Dong et al., 2015](#bib.bib13 ""); [Liu et al., 2015a](#bib.bib29 ""); [Luong et al., 2015](#bib.bib32 ""); [Hashimoto et al., 2017](#bib.bib17 "")), speech processing ([Huang et al., 2013](#bib.bib20 ""); [Seltzer and Droppo, 2013](#bib.bib48 ""); [Huang et al., 2015](#bib.bib21 "")), and even seemingly unrelated domains over multiple modalities ([Kaiser et al., 2017](#bib.bib23 "")).

[Baxter (2000)](#bib.bib4 "") theoretically analyze the MTL problem as interaction between individual learners and a meta-algorithm. Each learner is responsible for one task and a meta-algorithm decides how the shared parameters are updated. All aforementioned MTL algorithms use weighted summation as the meta-algorithm. Meta-algorithms that go beyond weighted summation have also been explored. [Li et al. (2014)](#bib.bib28 "") consider the case where each individual learner is based on kernel learning and utilize multi-objective optimization. [Zhang and Yeung (2010)](#bib.bib55 "") consider the case where each learner is a linear model and use a task affinity matrix. [Zhou et al. (2011a)](#bib.bib59 "") and [Bagherjeiran et al. (2005)](#bib.bib2 "") use the assumption that tasks share a dictionary and develop an expectation-maximization-like meta-algorithm. [de Miranda et al. (2012)](#bib.bib10 "") and [Zhou et al. (2017b)](#bib.bib58 "") use swarm optimization. None of these methods apply to gradient-based learning of high-capacity models such as modern deep networks. [Kendall et al. (2018)](#bib.bib24 "") and [Chen et al. (2018)](#bib.bib7 "") propose heuristics based on uncertainty and gradient magnitudes, respectively, and apply their methods to convolutional neural networks. Another recent work uses multi-agent reinforcement learning ([Rosenbaum et al., 2017](#bib.bib42 "")).

Multi-objective optimization. Multi-objective optimization addresses the problem of optimizing a set of possibly contrasting objectives. We recommend [Miettinen (1998)](#bib.bib34 "") and [Ehrgott (2005)](#bib.bib14 "") for surveys of this field. Of particular relevance to our work is gradient-based multi-objective optimization, as developed by [Fliege and Svaiter (2000)](#bib.bib15 ""), [Schäffler et al. (2002)](#bib.bib46 ""), and [Désidéri (2012)](#bib.bib12 ""). These methods use multi-objective Karush-Kuhn-Tucker (KKT) conditions ([Kuhn and Tucker, 1951](#bib.bib26 "")) and find a descent direction that decreases all objectives. This approach was extended to stochastic gradient descent by [Peitz and Dellnitz (2018)](#bib.bib38 "") and [Poirion et al. (2017)](#bib.bib40 ""). In machine learning, these methods have been applied to multi-agent learning ([Ghosh et al., 2013](#bib.bib16 ""); [Pirotta and Restelli, 2016](#bib.bib39 ""); [Parisi et al., 2014](#bib.bib36 "")), kernel learning ([Li et al., 2014](#bib.bib28 "")), sequential decision making ([Roijers et al., 2013](#bib.bib41 "")), and Bayesian optimization ([Shah and Ghahramani, 2016](#bib.bib49 ""); [Hernández-Lobato et al., 2016](#bib.bib19 "")). Our work applies gradient-based multi-objective optimization to multi-task learning.

## 3 Multi-Task Learning as Multi-Objective Optimization

Consider a multi-task learning (MTL) problem over an input space 𝒳\\mathcal{X} and a collection of task spaces {𝒴t}t∈\[T\]\\{\\mathcal{Y}^{t}\\}\_{t\\in\[T\]}, such that a large dataset of i.i.d. data points {𝐱i,yi1,…,yiT}i∈\[N\]\\{\\mathbf{x}\_{i},y\_{i}^{1},\\ldots,y^{T}\_{i}\\}\_{i\\in\[N\]} is given where TT is the number of tasks, NN is the number of data points, and yity^{t}\_{i} is the label of the ttht^{\\textup{th}} task for the ithi^{\\textup{th}} data point.11 1 This definition can be extended to the partially-labelled case by extending 𝒴t\\mathcal{Y}^{t} with a null label. We further consider a parametric hypothesis class per task as ft​(𝐱,𝜽s​h,𝜽t):𝒳→𝒴tf^{t}(\\mathbf{x};{\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}):\\mathcal{X}\\rightarrow\\mathcal{Y}^{t}, such that some parameters (𝜽s​h{\\bm{\\theta}}^{sh}) are shared between tasks and some (𝜽t{\\bm{\\theta}}^{t}) are task-specific. We also consider task-specific loss functions ℒt​(⋅,⋅):𝒴t×𝒴t→ℝ+\\mathcal{L}^{t}(\\cdot,\\cdot):\\mathcal{Y}^{t}\\times\\mathcal{Y}^{t}\\rightarrow{\\mathbb{R}}^{+}.

Although many hypothesis classes and loss functions have been proposed in the MTL literature, they generally yield the following empirical risk minimization formulation:

min𝜽s​h,𝜽1,…,𝜽T∑t\=1Tct​ℒ^t​(𝜽s​h,𝜽t)\\min\_{\\begin{subarray}{c}{\\bm{\\theta}}^{sh},\\\\ {\\bm{\\theta}}^{1},\\ldots,{\\bm{\\theta}}^{T}\\end{subarray}}\\quad\\sum\_{t=1}^{T}c^{t}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})

(1)

for some static or dynamically computed weights ctc^{t} per task, where ℒ^t​(𝜽s​h,𝜽t)\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) is the empirical loss of the task tt, defined as ℒ^t​(𝜽s​h,𝜽t)≜1N​∑iℒ⁡(ft​(𝐱i,𝜽s​h,𝜽t),yit)\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\triangleq\\frac{1}{N}\\sum\_{i}\\mathcal{L}\\big(f^{t}(\\mathbf{x}\_{i};{\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}),y\_{i}^{t}\\big).

Although the weighted summation formulation ([1](#S3.E1 "In 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) is intuitively appealing, it typically either requires an expensive grid search over various scalings or the use of a heuristic ([Kendall et al., 2018](#bib.bib24 ""); [Chen et al., 2018](#bib.bib7 "")). A basic justification for scaling is that it is not possible to define global optimality in the MTL setting. Consider two sets of solutions 𝜽{\\bm{\\theta}} and 𝜽¯\\bar{{\\bm{\\theta}}} such that ℒ^t1​(𝜽s​h,𝜽t1)<ℒ^t1​(𝜽¯s​h,𝜽¯t1)\\hat{\\mathcal{L}}^{t\_{1}}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t\_{1}})<\\hat{\\mathcal{L}}^{t\_{1}}(\\bar{{\\bm{\\theta}}}^{sh},\\bar{{\\bm{\\theta}}}^{t\_{1}}) and ℒ^t2​(𝜽s​h,𝜽t2)\>ℒ^t2​(𝜽¯s​h,𝜽¯t2)\\hat{\\mathcal{L}}^{t\_{2}}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t\_{2}})>\\hat{\\mathcal{L}}^{t\_{2}}(\\bar{{\\bm{\\theta}}}^{sh},\\bar{{\\bm{\\theta}}}^{t\_{2}}), for some tasks t1t\_{1} and t2t\_{2}. In other words, solution 𝜽{\\bm{\\theta}} is better for task t1t\_{1} whereas 𝜽¯\\bar{{\\bm{\\theta}}} is better for t2t\_{2}. It is not possible to compare these two solutions without a pairwise importance of tasks, which is typically not available.

Alternatively, MTL can be formulated as multi-objective optimization: optimizing a collection of possibly conflicting objectives. This is the approach we take. We specify the multi-objective optimization formulation of MTL using a vector-valued loss 𝐋\\mathbf{L}:

min𝜽s​h,𝜽1,…,𝜽T⁡𝐋⁡(𝜽s​h,𝜽1,…,𝜽T)\=min𝜽s​h,𝜽1,…,𝜽T⁡(ℒ^1​(𝜽s​h,𝜽1),…,ℒ^T​(𝜽s​h,𝜽T))⊺.\\min\_{\\begin{subarray}{c}{\\bm{\\theta}}^{sh},\\\\ {\\bm{\\theta}}^{1},\\ldots,{\\bm{\\theta}}^{T}\\end{subarray}}\\mathbf{L}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1},\\ldots,{\\bm{\\theta}}^{T})=\\min\_{\\begin{subarray}{c}{\\bm{\\theta}}^{sh},\\\\ {\\bm{\\theta}}^{1},\\ldots,{\\bm{\\theta}}^{T}\\end{subarray}}\\big(\\hat{\\mathcal{L}}^{1}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1}),\\ldots,\\hat{\\mathcal{L}}^{T}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{T})\\big)^{\\intercal}.

(2)

The goal of multi-objective optimization is achieving Pareto optimality.

###### Definition 1 (Pareto optimality for MTL).

1.  *(a)*

```
A solution 𝜽{\\bm{\\theta}} dominates a solution 𝜽¯\\bar{{\\bm{\\theta}}} if ℒ^t​(𝛉s​h,𝛉t)≤ℒ^t​(𝛉¯s​h,𝛉¯t)\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\leq\\hat{\\mathcal{L}}^{t}(\\bar{{\\bm{\\theta}}}^{sh},\\bar{{\\bm{\\theta}}}^{t}) for all tasks tt and 𝐋⁡(𝛉s​h,𝛉1,…,𝛉T)≠𝐋⁡(𝛉¯s​h,𝛉¯1,…,𝛉¯T)\\mathbf{L}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1},\\ldots,{\\bm{\\theta}}^{T})\\neq\\mathbf{L}(\\bar{{\\bm{\\theta}}}^{sh},\\bar{{\\bm{\\theta}}}^{1},\\ldots,\\bar{{\\bm{\\theta}}}^{T}).
```
2.  *(b)*

```
A solution 𝜽⋆{\\bm{\\theta}}^{\\star} is called Pareto optimal if there exists no solution 𝜽{\\bm{\\theta}} that dominates 𝜽⋆{\\bm{\\theta}}^{\\star}.
```
The set of Pareto optimal solutions is called the Pareto set (𝒫𝜽\\mathcal{P}\_{{\\bm{\\theta}}}) and its image is called the Pareto front (𝒫𝐋\={𝐋⁡(𝜽)}𝜽∈𝒫𝜽\\mathcal{P}\_{\\mathbf{L}}=\\{\\mathbf{L}({\\bm{\\theta}})\\}\_{{\\bm{\\theta}}\\in\\mathcal{P}\_{{\\bm{\\theta}}}}). In this paper, we focus on gradient-based multi-objective optimization due to its direct relevance to gradient-based MTL.

In the rest of this section, we first summarize in Section [3.1](#S3.SS1 "3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") how multi-objective optimization can be performed with gradient descent. Then, we suggest in Section [3.2](#S3.SS2 "3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") a practical algorithm for performing multi-objective optimization over very large parameter spaces. Finally, in Section [3.3](#S3.SS3 "3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") we propose an efficient solution for multi-objective optimization designed directly for high-capacity deep networks. Our method scales to very large models and a high number of tasks with negligible overhead.

### 3.1 Multiple Gradient Descent Algorithm

As in the single-objective case, multi-objective optimization can be solved to local optimality via gradient descent. In this section, we summarize one such approach, called the multiple gradient descent algorithm (MGDA) ([Désidéri, 2012](#bib.bib12 "")). MGDA leverages the Karush-Kuhn-Tucker (KKT) conditions, which are necessary for optimality ([Fliege and Svaiter, 2000](#bib.bib15 ""); [Schäffler et al., 2002](#bib.bib46 ""); [Désidéri, 2012](#bib.bib12 "")). We now state the KKT conditions for both task-specific and shared parameters:

*   •

```
There exist α1,…,αT≥0\\alpha^{1},\\ldots,\\alpha^{T}\\geq 0 such that ∑t\=1Tαt\=1\\sum\_{t=1}^{T}\\alpha^{t}=1 and ∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)\=0\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})=0
```
*   •

```
For all tasks tt, ∇𝜽tℒ^t​(𝜽s​h,𝜽t)\=0\\nabla\_{{\\bm{\\theta}}^{t}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})=0
```
Any solution that satisfies these conditions is called a Pareto stationary point. Although every Pareto optimal point is Pareto stationary, the reverse may not be true. Consider the optimization problem

minα1,…,αT{∥∑t\=1Tαt∇𝜽s​hℒ^t(𝜽s​h,𝜽t)∥22|∑t\=1Tαt\=1,αt≥0∀t}\\min\_{\\alpha^{1},\\ldots,\\alpha^{T}}\\Bigg\\{\\bigg\\|\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\bigg\\|\_{2}^{2}\\bigg|\\sum\_{t=1}^{T}\\alpha^{t}=1,\\alpha^{t}\\geq 0\\quad\\forall t\\Bigg\\}

(3)

[Désidéri (2012)](#bib.bib12 "") showed that either the solution to this optimization problem is 00 and the resulting point satisfies the KKT conditions, or the solution gives a descent direction that improves all tasks. Hence, the resulting MTL algorithm would be gradient descent on the task-specific parameters followed by solving ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) and applying the solution (∑t\=1Tαt​∇𝜽s​h\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}) as a gradient update to shared parameters. We discuss how to solve ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) for an arbitrary model in Section [3.2](#S3.SS2 "3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") and present an efficient solution when the underlying model is an encoder-decoder in Section [3.3](#S3.SS3 "3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization").

### 3.2 Solving the Optimization Problem

The optimization problem defined in ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) is equivalent to finding a minimum-norm point in the convex hull of the set of input points. This problem arises naturally in computational geometry: it is equivalent to finding the closest point within a convex hull to a given query point. It has been studied extensively ([Makimoto et al., 1994](#bib.bib33 ""); [Wolfe, 1976](#bib.bib51 ""); [Sekitani and Yamamoto, 1993](#bib.bib47 "")). Although many algorithms have been proposed, they do not apply in our setting because the assumptions they make do not hold. Algorithms proposed in the computational geometry literature address the problem of finding minimum-norm points in the convex hull of a large number of points in a low-dimensional space (typically of dimensionality 2 or 3). In our setting, the number of points is the number of tasks and is typically low; in contrast, the dimensionality is the number of shared parameters and can be in the millions. We therefore use a different approach based on convex optimization, since ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) is a convex quadratic problem with linear constraints.

Before we tackle the general case, let’s consider the case of two tasks. The optimization problem can be defined as minα∈\[0,1\]⁡‖α​∇𝜽s​hℒ^1​(𝜽s​h,𝜽1)+(1−α)​∇𝜽s​hℒ^2​(𝜽s​h,𝜽2)‖22\\min\_{\\alpha\\in\[0,1\]}\\|\\alpha\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{1}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1})+(1-\\alpha)\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{2}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{2})\\|\_{2}^{2}, which is a one-dimensional quadratic function of α\\alpha with an analytical solution:

α^\=\[(∇𝜽s​hℒ^2​(𝜽s​h,𝜽2)−∇𝜽s​hℒ^1​(𝜽s​h,𝜽1))⊺​∇𝜽s​hℒ^2​(𝜽s​h,𝜽2)‖∇𝜽s​hℒ^1​(𝜽s​h,𝜽1)−∇𝜽s​hℒ^2​(𝜽s​h,𝜽2)‖22\]+,1⊺\\hat{\\alpha}=\\left\[\\frac{\\big(\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{2}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{2})-\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{1}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1})\\big)^{\\intercal}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{2}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{2})}{\\|\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{1}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{1})-\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{2}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{2})\\|\_{2}^{2}}\\right\]\_{+,{1\\atop\\intercal}}

(4)

where \[⋅\]+,1⊺\[\\cdot\]\_{+,{1\\atop\\intercal}} represents clipping to \[0,1\]\[0,1\] as \[a\]+,1⊺\=max⁡(min⁡(a,1),0)\[a\]\_{+,{1\\atop\\intercal}}=\\max(\\min(a,1),0). We further visualize this solution in Figure 1. Although this is only applicable when T\=2T=2, this enables efficient application of the Frank-Wolfe algorithm ([Jaggi, 2013](#bib.bib22 "")) since the line search can be solved analytically. Hence, we use Frank-Wolfe to solve the constrained optimization problem, using ([4](#S3.E4 "In 3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) as a subroutine for the line search. We give all the update equations for the Frank-Wolfe solver in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization").

 ![[Uncaptioned image]](1810.04650v2/min_norm2point.png) Figure 1: Visualisation of the min-norm point in the convex hull of two points (minγ∈\[0,1\]⁡‖γ​𝜽+(1−γ)​𝜽¯‖22\\min\_{\\gamma\\in\[0,1\]}\\|\\gamma{\\bm{\\theta}}+(1-\\gamma)\\bar{{\\bm{\\theta}}}\\|\_{2}^{2}). As the geometry suggests, the solution is either an edge case or a perpendicular vector. 

 Algorithm 1 minγ∈\[0,1\]⁡‖γ​𝜽+(1−γ)​𝜽¯‖22\\min\_{\\gamma\\in\[0,1\]}\\|\\gamma{\\bm{\\theta}}+(1-\\gamma)\\bar{{\\bm{\\theta}}}\\|\_{2}^{2} 1: if 𝜽⊺​𝜽¯≥𝜽⊺​𝜽{\\bm{\\theta}}^{\\intercal}\\bar{{\\bm{\\theta}}}\\geq{\\bm{\\theta}}^{\\intercal}{\\bm{\\theta}} then 2:   γ\=1\\gamma=1 3: else if 𝜽⊺​𝜽¯≥𝜽¯⊺​𝜽¯{\\bm{\\theta}}^{\\intercal}\\bar{{\\bm{\\theta}}}\\geq\\bar{{\\bm{\\theta}}}^{\\intercal}\\bar{{\\bm{\\theta}}} then 4:   γ\=0\\gamma=0 5: else 6:   γ\=(𝜽¯−𝜽)⊺​𝜽¯‖𝜽−𝜽¯‖22\\gamma=\\frac{(\\bar{{\\bm{\\theta}}}-{\\bm{\\theta}})^{\\intercal}\\bar{{\\bm{\\theta}}}}{\\|{\\bm{\\theta}}-\\bar{{\\bm{\\theta}}}\\|\_{2}^{2}} 7: end if 

Algorithm 2 Update Equations for MTL

1: for t\=1t=1 to TT do 

2:   𝜽t\=𝜽t−η​∇𝜽tℒ^t​(𝜽s​h,𝜽t){\\bm{\\theta}}^{t}={\\bm{\\theta}}^{t}-\\eta\\nabla\_{{\\bm{\\theta}}^{t}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) ⊳\\triangleright Gradient descent on task-specific parameters 

3: end for 

4: α1,…,αT\\alpha^{1},\\ldots,\\alpha^{T} = FrankWolfeSolver(𝜽{\\bm{\\theta}}) ⊳\\triangleright Solve (3) to find a common descent direction 

5: 𝜽s​h\=𝜽s​h−η​∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t){\\bm{\\theta}}^{sh}={\\bm{\\theta}}^{sh}-\\eta\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) ⊳\\triangleright Gradient descent on shared parameters 

6: 

7: procedure FrankWolfeSolver(𝜽{\\bm{\\theta}}) 

8:   Initialize 𝜶\=(α1,…,αT)\=(1T,…,1T)\\bm{\\alpha}=(\\alpha^{1},\\ldots,\\alpha^{T})=(\\frac{1}{T},\\ldots,\\frac{1}{T}) 

9:   Precompute 𝐌\\mathbf{M} st. 𝐌i,j\=(∇𝜽s​hℒ^i​(𝜽s​h,𝜽i))⊺​(∇𝜽s​hℒ^j​(𝜽s​h,𝜽j))\\mathbf{M}\_{i,j}=\\big(\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{i}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{i})\\big)^{\\intercal}\\big(\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{j}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{j})\\big) 

10:   repeat 

11:    t^\=arg​minr∑tαt𝐌rt\\hat{t}=\\argmin\_{r}\\sum\_{t}\\alpha^{t}\\mathbf{M}\_{rt} 

12:    γ^\=arg​minγ⁡((1−γ)​𝜶+γ​𝐞t^)⊺​𝐌​((1−γ)​𝜶+γ​𝐞t^)\\hat{\\gamma}=\\argmin\_{\\gamma}\\big((1-\\gamma)\\bm{\\alpha}+\\gamma\\bm{e}\_{\\hat{t}}\\big)^{\\intercal}\\mathbf{M}\\big((1-\\gamma)\\bm{\\alpha}+\\gamma\\bm{e}\_{\\hat{t}}\\big) ⊳\\triangleright Using Algorithm 1 

13:    𝜶\=(1−γ^)​𝜶+γ^​𝒆t^\\bm{\\alpha}=(1-\\hat{\\gamma})\\bm{\\alpha}+\\hat{\\gamma}\\bm{e}\_{\\hat{t}} 

14:   until γ^∼0\\hat{\\gamma}\\sim 0 or Number of Iterations Limit 

15:   return α1,…,αT\\alpha^{1},\\ldots,\\alpha^{T} 

16: end procedure 

### 3.3 Efficient Optimization for Encoder-Decoder Architectures

The MTL update described in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") is applicable to any problem that uses optimization based on gradient descent. Our experiments also suggest that the Frank-Wolfe solver is efficient and accurate as it typically converges in a modest number of iterations with negligible effect on training time. However, the algorithm we described needs to compute ∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) for each task tt, which requires a backward pass over the shared parameters for each task. Hence, the resulting gradient computation would be the forward pass followed by TT backward passes. Considering the fact that computation of the backward pass is typically more expensive than the forward pass, this results in linear scaling of the training time and can be prohibitive for problems with more than a few tasks.

We now propose an efficient method that optimizes an upper bound of the objective and requires only a single backward pass. We further show that optimizing this upper bound yields a Pareto optimal solution under realistic assumptions. The architectures we address conjoin a shared representation function with task-specific decision functions. This class of architectures covers most of the existing deep MTL models and can be formally defined by constraining the hypothesis class as

ft​(𝐱,𝜽s​h,𝜽t)\=(ft​(⋅,𝜽t)∘g⁡(⋅,𝜽s​h))​(𝐱)\=ft​(g⁡(𝐱,𝜽s​h),𝜽t)f^{t}(\\mathbf{x};{\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})=(f^{t}(\\cdot;{\\bm{\\theta}}^{t})\\circ g(\\cdot;{\\bm{\\theta}}^{sh}))(\\mathbf{x})=f^{t}(g(\\mathbf{x};{\\bm{\\theta}}^{sh});{\\bm{\\theta}}^{t})

(5)

where gg is the representation function shared by all tasks and ftf^{t} are the task-specific functions that take this representation as input. If we denote the representations as 𝐙\=(𝐳1,…,𝐳N)\\mathbf{Z}=\\big(\\mathbf{z}\_{1},\\ldots,\\mathbf{z}\_{N}\\big), where 𝐳i\=g⁡(𝐱i,𝜽s​h)\\mathbf{z}\_{i}=g(\\mathbf{x}\_{i};{\\bm{\\theta}}^{sh}), we can state the following upper bound as a direct consequence of the chain rule:

‖∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)‖22≤‖∂𝐙∂𝜽s​h‖22​‖∑t\=1Tαt​∇𝐙ℒ^t​(𝜽s​h,𝜽t)‖22\\Bigg\\|\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\Bigg\\|\_{2}^{2}\\leq\\Bigg\\|\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}\\Bigg\\|\_{2}^{2}\\Bigg\\|\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\Bigg\\|\_{2}^{2}

(6)

where ‖∂𝐙∂𝜽s​h‖2\\left\\|\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}\\right\\|\_{2} is the matrix norm of the Jacobian of 𝐙\\mathbf{Z} with respect to 𝜽s​h{\\bm{\\theta}}^{sh}. Two desirable properties of this upper bound are that (i) ∇𝐙ℒ^t​(𝜽s​h,𝜽t)\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) can be computed in a single backward pass for all tasks and (ii) ‖∂𝐙∂𝜽s​h‖22\\left\\|\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}\\right\\|\_{2}^{2} is not a function of α1,…,αT\\alpha^{1},\\ldots,\\alpha^{T}, hence it can be removed when it is used as an optimization objective. We replace the ‖∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)‖22\\left\\|\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\right\\|\_{2}^{2} term with the upper bound we have just derived in order to obtain the approximate optimization problem and drop the ‖∂𝐙∂𝜽s​h‖22\\left\\|\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}\\right\\|\_{2}^{2} term since it does not affect the optimization. The resulting optimization problem is

minα1,…,αT{∥∑t\=1Tαt∇𝐙ℒ^t(𝜽s​h,𝜽t)∥22|∑t\=1Tαt\=1,αt≥0∀t}\\min\_{\\alpha^{1},\\ldots,\\alpha^{T}}\\Bigg\\{\\bigg\\|\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})\\bigg\\|\_{2}^{2}\\bigg|\\sum\_{t=1}^{T}\\alpha^{t}=1,\\alpha^{t}\\geq 0\\quad\\forall t\\Bigg\\}\\\\ 

(MGDA-UB)

We refer to this problem as MGDA-UB (Multiple Gradient Descent Algorithm – Upper Bound). In practice, MGDA-UB corresponds to using the gradients of the task losses with respect to the representations instead of the shared parameters. We use Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Solving the Optimization Problem ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") with only this change as the final method.

Although MGDA-UB is an approximation of the original optimization problem, we now state a theorem that shows that our method produces a Pareto optimal solution under mild assumptions. The proof is given in the supplement.

###### Theorem 1.

Assume ∂𝐙∂θs​h\\frac{\\partial\\mathbf{Z}}{\\partial\\mathbf{\\theta}^{sh}} is full-rank. If α1,…,T\\alpha^{1,\\ldots,T} is the solution of MGDA-UB, one of the following is true:

1.  *(a)*

```
∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)\=0\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})=0 and the current parameters are Pareto stationary.
```
2.  *(b)*

```
∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t}) is a descent direction that decreases all objectives.
```
This result follows from the fact that as long as ∂𝐙∂𝜽s​h\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}} is full rank, optimizing the upper bound corresponds to minimizing the norm of the convex combination of the gradients using the Mahalonobis norm defined by ∂𝐙∂𝜽s​h⊺​∂𝐙∂𝜽s​h\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}^{\\intercal}\\frac{\\partial\\mathbf{Z}}{\\partial{\\bm{\\theta}}^{sh}}. The non-singularity assumption is reasonable as singularity implies that tasks are linearly related and a trade-off is not necessary. In summary, our method provably finds a Pareto stationary point with negligible computational overhead and can be applied to any deep multi-objective problem with an encoder-decoder model.

## 4 Experiments

We evaluate the presented MTL method on a number of problems. First, we use MultiMNIST ([Sabour et al., 2017](#bib.bib45 "")), an MTL adaptation of MNIST ([LeCun et al., 1998](#bib.bib27 "")). Next, we tackle multi-label classification on the CelebA dataset ([Liu et al., 2015b](#bib.bib30 "")) by considering each label as a distinct binary classification task. These problems include both classification and regression, with the number of tasks ranging from 2 to 40. Finally, we experiment with scene understanding, jointly tackling the tasks of semantic segmentation, instance segmentation, and depth estimation on the Cityscapes dataset ([Cordts et al., 2016](#bib.bib9 "")). We discuss each experiment separately in the following subsections.

The baselines we consider are (i) uniform scaling: minimizing a uniformly weighted sum of loss functions 1T​∑tℒt\\frac{1}{T}\\sum\_{t}\\mathcal{L}^{t}, (ii) single task: solving tasks independently, (iii) grid search: exhaustively trying various values from {ct∈\[0,1\]|∑tct\=1}\\{c^{t}\\in\[0,1\]|\\sum\_{t}c^{t}=1\\} and optimizing for 1T​∑tct​ℒt\\frac{1}{T}\\sum\_{t}c^{t}\\mathcal{L}^{t}, (iv) [Kendall et al. (2018)](#bib.bib24 ""): using the uncertainty weighting proposed by [Kendall et al. (2018)](#bib.bib24 ""), and (v) GradNorm: using the normalization proposed by [Chen et al. (2018)](#bib.bib7 "").

### 4.1 MultiMNIST

Our initial experiments are on MultiMNIST, an MTL version of the MNIST dataset ([Sabour et al., 2017](#bib.bib45 "")). In order to convert digit classification into a multi-task problem, [Sabour et al. (2017)](#bib.bib45 "") overlaid multiple images together. We use a similar construction. For each image, a different one is chosen uniformly in random. Then one of these images is put at the top-left and the other one is at the bottom-right. The resulting tasks are: classifying the digit on the top-left (task-L) and classifying the digit on the bottom-right (task-R). We use 60K examples and directly apply existing single-task MNIST models. The MultiMNIST dataset is illustrated in the supplement.

We use the LeNet architecture ([LeCun et al., 1998](#bib.bib27 "")). We treat all layers except the last as the representation function gg and put two fully-connected layers as task-specific functions (see the supplement for details). We visualize the performance profile as a scatter plot of accuracies on task-L and task-R in Figure [3](#S5.F3 "Figure 3 ‣ 5 Conclusion ‣ Multi-Task Learning as Multi-Objective Optimization"), and list the results in Table [5](#S5 "5 Conclusion ‣ Multi-Task Learning as Multi-Objective Optimization").

In this setup, any static scaling results in lower accuracy than solving each task separately (the single-task baseline). The two tasks appear to compete for model capacity, since increase in the accuracy of one task results in decrease in the accuracy of the other. Uncertainty weighting ([Kendall et al., 2018](#bib.bib24 "")) and GradNorm ([Chen et al., 2018](#bib.bib7 "")) find solutions that are slightly better than grid search but distinctly worse than the single-task baseline. In contrast, our method finds a solution that efficiently utilizes the model capacity and yields accuracies that are as good as the single-task solutions. This experiment demonstrates the effectiveness of our method as well as the necessity of treating MTL as multi-objective optimization. Even after a large hyper-parameter search, *any* scaling of tasks does not approach the effectiveness of our method.

### 4.2 Multi-Label Classification

Figure 2: Radar charts of percentage error per attribute on CelebA ([Liu et al., 2015b](#bib.bib30 "")). Lower is better. We divide attributes into two sets for legibility: easy on the left, hard on the right. Zoom in for details.

  

|                    |
| ------------------ |
| error              |
| 8.778.77           |
| 9.629.62           |
| 9.539.53           |
| 8.448.44           |
| 8.25\\mathbf{8.25} |

Table 1: Mean of error per category of MTL algorithms in multi-label classification on CelebA ([Liu et al., 2015b](#bib.bib30 "")).

Next, we tackle multi-label classification. Given a set of attributes, multi-label classification calls for deciding whether each attribute holds for the input. We use the CelebA dataset ([Liu et al., 2015b](#bib.bib30 "")), which includes 200K face images annotated with 40 attributes. Each attribute gives rise to a binary classification task and we cast this as a 40-way MTL problem. We use ResNet-18 ([He et al., 2016](#bib.bib18 "")) without the final layer as a shared representation function, and attach a linear layer for each attribute (see the supplement for further details).

We plot the resulting error for each binary classification task as a radar chart in Figure [2](#S4.F2 "Figure 2 ‣ 4.2 Multi-Label Classification ‣ 4 Experiments ‣ Multi-Task Learning as Multi-Objective Optimization"). The average over them is listed in Table [4.2](#S4.SS2 "4.2 Multi-Label Classification ‣ 4 Experiments ‣ Multi-Task Learning as Multi-Objective Optimization"). We skip grid search since it is not feasible over 40 tasks. Although uniform scaling is the norm in the multi-label classification literature, single-task performance is significantly better. Our method outperforms baselines for significant majority of tasks and achieves comparable performance in rest. This experiment also shows that our method remains effective when the number of tasks is high.

### 4.3 Scene Understanding

To evaluate our method in a more realistic setting, we use scene understanding. Given an RGB image, we solve three tasks: semantic segmentation (assigning pixel-level class labels), instance segmentation (assigning pixel-level instance labels), and monocular depth estimation (estimating continuous disparity per pixel). We follow the experimental procedure of [Kendall et al. (2018)](#bib.bib24 "") and use an encoder-decoder architecture. The encoder is based on ResNet-50 ([He et al., 2016](#bib.bib18 "")) and is shared by all three tasks. The decoders are task-specific and are based on the pyramid pooling module ([Zhao et al., 2017](#bib.bib56 "")) (see the supplement for further implementation details).

Since the output space of instance segmentation is unconstrained (the number of instances is not known in advance), we use a proxy problem as in [Kendall et al. (2018)](#bib.bib24 ""). For each pixel, we estimate the location of the center of mass of the instance that encompasses the pixel. These center votes can then be clustered to extract the instances. In our experiments, we directly report the MSE in the proxy task. Figure [4](#S5.F4 "Figure 4 ‣ 5 Conclusion ‣ Multi-Task Learning as Multi-Objective Optimization") shows the performance profile for each pair of tasks, although we perform all experiments on all three tasks jointly. The pairwise performance profiles shown in Figure [4](#S5.F4 "Figure 4 ‣ 5 Conclusion ‣ Multi-Task Learning as Multi-Objective Optimization") are simply 2D projections of the three-dimensional profile, presented this way for legibility. The results are also listed in Table [5](#S5 "5 Conclusion ‣ Multi-Task Learning as Multi-Objective Optimization").

MTL outperforms single-task accuracy, indicating that the tasks cooperate and help each other. Our method outperforms all baselines on all tasks.

### 4.4 Role of the Approximation

In order to understand the role of the approximation proposed in Section [3.3](#S3.SS3 "3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization"), we compare the final performance and training time of our algorithm with and without the presented approximation in Table [2](#S4.T2 "Table 2 ‣ 4.4 Role of the Approximation ‣ 4 Experiments ‣ Multi-Task Learning as Multi-Objective Optimization") (runtime measured on a single Titan Xp GPU). For a small number of tasks (3 for scene understanding), training time is reduced by 40%. For the multi-label classification experiment (40 tasks), the presented approximation accelerates learning by a factor of 25.

On the accuracy side, we expect both methods to perform similarly as long as the full-rank assumption is satisfied. As expected, the accuracy of both methods is very similar. Somewhat surprisingly, our approximation results in slightly improved accuracy in all experiments. While counter-intuitive at first, we hypothesize that this is related to the use of SGD in the learning algorithm. Stability analysis in convex optimization suggests that if gradients are computed with an error ∇^𝜽​ℒt\=∇𝜽ℒt+𝐞t\\hat{\\nabla}\_{\\bm{\\theta}}\\mathcal{L}^{t}=\\nabla\_{\\bm{\\theta}}\\mathcal{L}^{t}+\\mathbf{e}^{t} (𝜽{\\bm{\\theta}} corresponds to 𝜽s​h{\\bm{\\theta}}^{sh} in ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization"))), as opposed to 𝐙\\mathbf{Z} in the approximate problem in [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization"), the error in the solution is bounded as ‖α^−α‖2≤𝒪⁡(maxt⁡‖𝐞t‖2)\\|\\hat{\\mathbf{\\alpha}}-\\mathbf{\\alpha}\\|\_{2}\\leq\\mathcal{O}(\\max\_{t}\\|\\mathbf{e}^{t}\\|\_{2}). Considering the fact that the gradients are computed over the full parameter set (millions of dimensions) for the original problem and over a smaller space for the approximation (batch size times representation which is in the thousands), the dimension of the error vector is significantly higher in the original problem. We expect the l2l\_{2} norm of such a random vector to depend on the dimension.

In summary, our quantitative analysis of the approximation suggests that (i) the approximation does not cause an accuracy drop and (ii) by solving an equivalent problem in a lower-dimensional space, our method achieves both better computational efficiency and higher stability.

Table 2: Effect of the MGDA-UB approximation. We report the final accuracies as well as training times for our method with and without the approximation.

|                    |                      |                      |                    |  |                    |                    |
| ------------------ | -------------------- | -------------------- | ------------------ |  | ------------------ | ------------------ |
| Training           | Segmentation         | Instance             | Disparity          |  | Training           | Average            |
| time               | mIoU \[%\]           | error \[px\]         | error \[px\]       |  | time (hour)        | error              |
| 38.638.6           | 66.1366.13           | 10.2810.28           | 2.592.59           |  | 429.9429.9         | 8.338.33           |
| 23.3\\mathbf{23.3} | 66.63\\mathbf{66.63} | 10.25\\mathbf{10.25} | 2.54\\mathbf{2.54} |  | 16.1\\mathbf{16.1} | 8.25\\mathbf{8.25} |

## 5 Conclusion

We described an approach to multi-task learning. Our approach is based on multi-objective optimization. In order to apply multi-objective optimization to MTL, we described an efficient algorithm as well as specific approximations that yielded a deep MTL algorithm with almost no computational overhead. Our experiments indicate that the resulting algorithm is effective for a wide range of multi-task scenarios.

 Figure 3: MultiMNIST accuracy profile. We plot the obtained accuracy in detecting the left and right digits for all baselines. The grid-search results suggest that the tasks compete for model capacity. Our method is the only one that finds a solution that is as good as training a dedicated model for each task. Top-right is better. Table 3: Performance of MTL algorithms on MultiMNIST. Single-task baselines solve tasks separately, with dedicated models, but are shown in the same row for clarity. Left digit Right digit accuracy \[%\] accuracy \[%\] Single task 97.23\\mathbf{97.23} 95.90\\mathbf{95.90} Uniform scaling 96.4696.46 94.9994.99 [Kendall et al. 2018](#bib.bib24 "") 96.4796.47 95.2995.29 GradNorm 96.2796.27 94.8494.84 Ours 97.26\\mathbf{97.26} 95.90\\mathbf{95.90} Table 4: Performance of MTL algorithms in joint semantic segmentation, instance segmentation, and depth estimation on Cityscapes. Single-task baselines solve tasks separately but are shown in the same row for clarity. Segmentation Instance Disparity mIoU \[%\] error \[px\] error \[px\] Single task 60.6860.68 11.3411.34 2.782.78 Uniform scaling 54.5954.59 10.3810.38 2.962.96 [Kendall et al. 2018](#bib.bib24 "") 64.2164.21 11.5411.54 2.652.65 GradNorm 64.8164.81 11.3111.31 2.57\\mathbf{2.57} Ours 66.63\\mathbf{66.63} 10.25\\mathbf{10.25} 2.54\\mathbf{2.54}     

 Figure 4: Cityscapes performance profile. We plot the performance of all baselines for the tasks of semantic segmentation, instance segmentation, and depth estimation. We use mIoU for semantic segmentation, error of per-pixel regression (normalized to image size) for instance segmentation, and disparity error for depth estimation. To convert errors to performance measures, we use 1 −- instance error and 1/disparity error. We plot 2D projections of the performance profile for each pair of tasks. Although we plot pairwise projections for visualization, each point in the plots solves all tasks. Top-right is better. 

## References

*   Argyriou et al. (2007) A. Argyriou, T. Evgeniou, and M. Pontil. Multi-task feature learning. In *NIPS*, 2007.
*   Bagherjeiran et al. (2005) A. Bagherjeiran, R. Vilalta, and C. F. Eick. Content-based image retrieval through a multi-agent meta-learning framework. In *International Conference on Tools with Artificial Intelligence*, 2005.
*   Bakker and Heskes (2003) B. Bakker and T. Heskes. Task clustering and gating for Bayesian multitask learning. *JMLR*, 4:83–99, 2003.
*   Baxter (2000) J. Baxter. A model of inductive bias learning. *Journal of Artificial Intelligence Research*, 12:149–198, 2000.
*   Bilen and Vedaldi (2016) H. Bilen and A. Vedaldi. Integrated perception with recurrent multi-task neural networks. In *NIPS*, 2016.
*   Caruana (1997) R. Caruana. Multitask learning. *Machine Learning*, 28(1):41–75, 1997.
*   Chen et al. (2018) Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *ICML*, 2018.
*   Collobert and Weston (2008) R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In *ICML*, 2008.
*   Cordts et al. (2016) M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The Cityscapes dataset for semantic urban scene understanding. In *CVPR*, 2016.
*   de Miranda et al. (2012) P. B. C. de Miranda, R. B. C. Prudêncio, A. C. P. L. F. de Carvalho, and C. Soares. Combining a multi-objective optimization approach with meta-learning for SVM parameter selection. In *International Conference on Systems, Man, and Cybernetics*, 2012.
*   Deng et al. (2009) J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In *CVPR*, 2009.
*   Désidéri (2012) J.-A. Désidéri. Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5):313–318, 2012.
*   Dong et al. (2015) D. Dong, H. Wu, W. He, D. Yu, and H. Wang. Multi-task learning for multiple language translation. In *ACL*, 2015.
*   Ehrgott (2005) M. Ehrgott. *Multicriteria Optimization (2. ed.)*. Springer, 2005.
*   Fliege and Svaiter (2000) J. Fliege and B. F. Svaiter. Steepest descent methods for multicriteria optimization. *Mathematical Methods of Operations Research*, 51(3):479–494, 2000.
*   Ghosh et al. (2013) S. Ghosh, C. Lovell, and S. R. Gunn. Towards Pareto descent directions in sampling experts for multiple tasks in an on-line learning paradigm. In *AAAI Spring Symposium: Lifelong Machine Learning*, 2013.
*   Hashimoto et al. (2017) K. Hashimoto, C. Xiong, Y. Tsuruoka, and R. Socher. A joint many-task model: Growing a neural network for multiple NLP tasks. In *EMNLP*, 2017.
*   He et al. (2016) K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In *CVPR*, 2016.
*   Hernández-Lobato et al. (2016) D. Hernández-Lobato, J. M. Hernández-Lobato, A. Shah, and R. P. Adams. Predictive entropy search for multi-objective bayesian optimization. In *ICML*, 2016.
*   Huang et al. (2013) J.-T. Huang, J. Li, D. Yu, L. Deng, and Y. Gong. Cross-language knowledge transfer using multilingual deep neural network with shared hidden layers. In *ICASSP*, 2013.
*   Huang et al. (2015) Z. Huang, J. Li, S. M. Siniscalchi, I.-F. Chen, J. Wu, and C.-H. Lee. Rapid adaptation for deep neural networks through multi-task learning. In *Interspeech*, 2015.
*   Jaggi (2013) M. Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In *ICML*, 2013.
*   Kaiser et al. (2017) L. Kaiser, A. N. Gomez, N. Shazeer, A. Vaswani, N. Parmar, L. Jones, and J. Uszkoreit. One model to learn them all. *arXiv:1706.05137*, 2017.
*   Kendall et al. (2018) A. Kendall, Y. Gal, and R. Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *CVPR*, 2018.
*   Kokkinos (2017) I. Kokkinos. UberNet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In *CVPR*, 2017.
*   Kuhn and Tucker (1951) H. W. Kuhn and A. W. Tucker. Nonlinear programming. In *Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability*, Berkeley, Calif., 1951. University of California Press.
*   LeCun et al. (1998) Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
*   Li et al. (2014) C. Li, M. Georgiopoulos, and G. C. Anagnostopoulos. Pareto-path multi-task multiple kernel learning. *arXiv:1404.3190*, 2014.
*   Liu et al. (2015a) X. Liu, J. Gao, X. He, L. Deng, K. Duh, and Y.-Y. Wang. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In *NAACL HLT*, 2015a.
*   Liu et al. (2015b) Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In *ICCV*, 2015b.
*   Long and Wang (2015) M. Long and J. Wang. Learning multiple tasks with deep relationship networks. *arXiv:1506.02117*, 2015.
*   Luong et al. (2015) M.-T. Luong, Q. V. Le, I. Sutskever, O. Vinyals, and L. Kaiser. Multi-task sequence to sequence learning. *arXiv:1511.06114*, 2015.
*   Makimoto et al. (1994) N. Makimoto, I. Nakagawa, and A. Tamura. An efficient algorithm for finding the minimum norm point in the convex hull of a finite point set in the plane. *Operations Research Letters*, 16(1):33–40, 1994.
*   Miettinen (1998) K. Miettinen. *Nonlinear Multiobjective Optimization*. Springer, 1998.
*   Misra et al. (2016) I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Cross-stitch networks for multi-task learning. In *CVPR*, 2016.
*   Parisi et al. (2014) S. Parisi, M. Pirotta, N. Smacchia, L. Bascetta, and M. Restelli. Policy gradient approaches for multi-objective sequential decision making. In *IJCNN*, 2014.
*   Paszke et al. (2017) A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. In *NIPS Workshops*, 2017.
*   Peitz and Dellnitz (2018) S. Peitz and M. Dellnitz. Gradient-based multiobjective optimization with uncertainties. In *NEO*, 2018.
*   Pirotta and Restelli (2016) M. Pirotta and M. Restelli. Inverse reinforcement learning through policy gradient minimization. In *AAAI*, 2016.
*   Poirion et al. (2017) F. Poirion, Q. Mercier, and J. Désidéri. Descent algorithm for nonsmooth stochastic multiobjective optimization. *Computational Optimization and Applications*, 68(2):317–331, 2017.
*   Roijers et al. (2013) D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley. A survey of multi-objective sequential decision-making. *Journal of Artificial Intelligence Research*, 48:67–113, 2013.
*   Rosenbaum et al. (2017) C. Rosenbaum, T. Klinger, and M. Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. *arXiv:1711.01239*, 2017.
*   Rudd et al. (2016) E. M. Rudd, M. Günther, and T. E. Boult. MOON: A mixed objective optimization network for the recognition of facial attributes. In *ECCV*, 2016.
*   Ruder (2017) S. Ruder. An overview of multi-task learning in deep neural networks. *arXiv:1706.05098*, 2017.
*   Sabour et al. (2017) S. Sabour, N. Frosst, and G. E. Hinton. Dynamic routing between capsules. In *NIPS*, 2017.
*   Schäffler et al. (2002) S. Schäffler, R. Schultz, and K. Weinzierl. Stochastic method for the solution of unconstrained vector optimization problems. *Journal of Optimization Theory and Applications*, 114(1):209–222, 2002.
*   Sekitani and Yamamoto (1993) K. Sekitani and Y. Yamamoto. A recursive algorithm for finding the minimum norm point in a polytope and a pair of closest points in two polytopes. *Mathematical Programming*, 61(1-3):233–249, 1993.
*   Seltzer and Droppo (2013) M. L. Seltzer and J. Droppo. Multi-task learning in deep neural networks for improved phoneme recognition. In *ICASSP*, 2013.
*   Shah and Ghahramani (2016) A. Shah and Z. Ghahramani. Pareto frontier learning with expensive correlated objectives. In *ICML*, 2016.
*   Stein (1956) C. Stein. Inadmissibility of the usual estimator for the mean of a multivariate normal distribution. Technical report, Stanford University, US, 1956.
*   Wolfe (1976) P. Wolfe. Finding the nearest point in a polytope. *Mathematical Programming*, 11(1):128–149, 1976.
*   Xue et al. (2007) Y. Xue, X. Liao, L. Carin, and B. Krishnapuram. Multi-task learning for classification with dirichlet process priors. *JMLR*, 8:35–63, 2007.
*   Yang and Hospedales (2016) Y. Yang and T. M. Hospedales. Trace norm regularised deep multi-task learning. *arXiv:1606.04038*, 2016.
*   Zamir et al. (2018) A. R. Zamir, A. Sax, W. B. Shen, L. J. Guibas, J. Malik, and S. Savarese. Taskonomy: Disentangling task transfer learning. In *CVPR*, 2018.
*   Zhang and Yeung (2010) Y. Zhang and D. Yeung. A convex formulation for learning task relationships in multi-task learning. In *UAI*, 2010.
*   Zhao et al. (2017) H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene parsing network. In *CVPR*, 2017.
*   Zhou et al. (2017a) B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba. Scene parsing through ADE20K dataset. In *CVPR*, 2017a.
*   Zhou et al. (2017b) D. Zhou, J. Wang, B. Jiang, H. Guo, and Y. Li. Multi-task multi-view learning based on cooperative multi-objective optimization. *IEEE Access*, 2017b.
*   Zhou et al. (2011a) J. Zhou, J. Chen, and J. Ye. Clustered multi-task learning via alternating structure optimization. In *NIPS*, 2011a.
*   Zhou et al. (2011b) J. Zhou, J. Chen, and J. Ye. MALSAR: Multi-task learning via structural regularization. *Arizona State University*, 2011b.

## Appendix A Proof of Theorem 1

###### Proof.

We begin by showing that if the optimum value of [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") is 0, so is the optimum value of ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")). This shows the first case of the theorem. Then, we will show the second part.

If the optimum value of [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") is 00, ∑t\=1Tαt​∇𝜽s​hℒ^t​(𝜽s​h,𝜽t)\=∂𝐙∂θs​h​∑t\=1Tαt​∇𝐙ℒt^\=∑t\=1Tαt​∇θs​hℒt^\=0\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{{\\bm{\\theta}}^{sh}}\\hat{\\mathcal{L}}^{t}({\\bm{\\theta}}^{sh},{\\bm{\\theta}}^{t})=\\frac{\\partial\\mathbf{Z}}{\\partial\\mathbf{\\theta}^{sh}}\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}^{t}}=\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{\\theta}^{sh}}\\hat{\\mathcal{L}^{t}}=0

(7)

Hence α1,…,αT\\alpha^{1},\\ldots,\\alpha^{T} is the solution of ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) and the optimal value of ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")) is 00. This proves the first case of the theorem. Before we move to the second case, we state a straightforward corollary. Since ∂𝐙∂θs​h\\frac{\\partial\\mathbf{Z}}{\\partial\\mathbf{\\theta}^{sh}} is full rank, this equivalence is bi-directional. In other words, if α1,…,αT\\alpha^{1},\\ldots,\\alpha^{T} is the solution of ([3](#S3.E3 "In 3.1 Multiple Gradient Descent Algorithm ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization")), it is the solution of [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") as well. Hence, both formulations completely agree on Pareto stationarity.

In order to prove the second case, we need to show that the resulting descent direction computed by solving [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") does not increase any of the loss functions. Formally, we need to show that

(∑t\=1Tαt​∇θs​hℒ^t)⊺​(∇θs​hℒ^t′)≥0∀t′∈{1,…,T}\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{\\theta}^{sh}}\\hat{\\mathcal{L}}^{t}\\right)^{\\intercal}\\left(\\nabla\_{\\mathbf{\\theta}^{sh}}\\hat{\\mathcal{L}}^{t^{\\prime}}\\right)\\geq 0\\quad\\forall~t^{\\prime}\\in\\{1,\\ldots,T\\}

(8)

This condition is equivalent to

(∑t\=1Tαt​∇𝐙ℒ^t)⊺​𝐌​(∇𝐙ℒ^t′)≥0∀t′∈{1,…,T}\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)^{\\intercal}\\mathbf{M}\\left(\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t^{\\prime}}\\right)\\geq 0\\quad\\forall~t^{\\prime}\\in\\{1,\\ldots,T\\}

(9)

where 𝐌\=(∂𝐙∂θs​h)⊺​(∂𝐙∂θs​h)\\mathbf{M}=\\big(\\frac{\\partial\\mathbf{Z}}{\\partial\\mathbf{\\theta}^{sh}}\\big)^{\\intercal}\\big(\\frac{\\partial\\mathbf{Z}}{\\partial\\mathbf{\\theta}^{sh}}\\big). Since 𝐌\\mathbf{M} is positive definite (following the assumption), this is further equivalent to

(∑t\=1Tαt​∇𝐙ℒ^t)⊺​(∇𝐙ℒ^t′)≥0∀t′∈{1,…,T}\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)^{\\intercal}\\left(\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t^{\\prime}}\\right)\\geq 0\\quad\\forall~t^{\\prime}\\in\\{1,\\ldots,T\\}

(10)

We show that this follows from the optimality conditions for [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization"). The Lagrangian of [(MGDA-UB)](#S3.Ex1 "In 3.3 Efficient Optimization for Encoder-Decoder Architectures ‣ 3 Multi-Task Learning as Multi-Objective Optimization ‣ Multi-Task Learning as Multi-Objective Optimization") is

(∑t\=1Tαt​∇𝐙ℒ^t)⊺​(∑t\=1Tαt​∇𝐙ℒ^t)−λ⁡(∑iαi−1)​ where ​λ≥0.\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)^{\\intercal}\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)-\\lambda\\left(\\sum\_{i}\\alpha^{i}-1\\right)\\text{ where }\\lambda\\geq 0.

(11)

The KKT condition for this Lagrangian yields the desired result as

(∑t\=1Tαt​∇𝐙ℒ^t)⊺​(∇𝐙ℒ^t)\=λ2≥0\\left(\\sum\_{t=1}^{T}\\alpha^{t}\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)^{\\intercal}\\left(\\nabla\_{\\mathbf{Z}}\\hat{\\mathcal{L}}^{t}\\right)=\\frac{\\lambda}{2}\\geq 0

(12)

∎

## Appendix B Additional Results on Multi-label Classification

In this section, we present the experimental results we did not include in the main text.

In the main text, we plotted a radar chart of the binary attribute classification errors. However, we did not include the tabulated results due to the space limitations. Here we list the binary classification error of each attribute for each algorithm in Table [5](#A2.T5 "Table 5 ‣ Appendix B Additional Results on Multi-label Classification ‣ Multi-Task Learning as Multi-Objective Optimization").

Table 5: Multi-label classification error per attribute for all algorithms.

| Uniform  | Single     | Kendall    | Grad       |            |            |  | Uniform  | Single     | Kendall    | Grad       |            |            |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- |  | -------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|          |            |            |            |            |            |  |          |            |            |            |            |            |
| Attr. 0  | 7.117.11   | 7.167.16   | 7.187.18   | 6.546.54   | 6.17       |  | Attr. 5  | 4.914.91   | 4.754.75   | 4.954.95   | 4.194.19   | 4.13       |
| Attr. 1  | 17.3017.30 | 14.38      | 16.7716.77 | 14.8014.80 | 14.8714.87 |  | Attr. 6  | 20.9720.97 | 14.2414.24 | 15.1715.17 | 14.07      | 14.0814.08 |
| Attr. 2  | 20.9920.99 | 19.2519.25 | 20.5620.56 | 18.9718.97 | 18.35      |  | Attr. 7  | 18.5318.53 | 17.7417.74 | 18.8418.84 | 17.3317.33 | 17.25      |
| Attr. 3  | 17.8217.82 | 16.7916.79 | 18.4518.45 | 16.4716.47 | 16.06      |  | Attr. 8  | 10.2210.22 | 8.878.87   | 10.1910.19 | 8.678.67   | 8.42       |
| Attr. 4  | 1.251.25   | 1.201.20   | 1.171.17   | 1.131.13   | 1.08       |  | Attr. 9  | 5.295.29   | 5.095.09   | 5.445.44   | 4.684.68   | 4.60       |
| Attr. 10 | 4.144.14   | 4.024.02   | 4.334.33   | 3.773.77   | 3.60       |  | Attr. 15 | 0.810.81   | 0.52       | 0.620.62   | 0.560.56   | 0.560.56   |
| Attr. 11 | 16.2216.22 | 15.3415.34 | 16.6416.64 | 14.7314.73 | 14.56      |  | Attr. 16 | 4.004.00   | 3.943.94   | 3.993.99   | 3.723.72   | 3.46       |
| Attr. 12 | 8.428.42   | 7.687.68   | 8.858.85   | 7.23       | 7.417.41   |  | Attr. 17 | 2.392.39   | 2.662.66   | 2.352.35   | 2.09       | 2.162.16   |
| Attr. 13 | 5.175.17   | 5.155.15   | 5.265.26   | 4.754.75   | 4.52       |  | Attr. 18 | 8.798.79   | 9.019.01   | 8.848.84   | 8.008.00   | 7.83       |
| Attr. 14 | 4.144.14   | 4.134.13   | 4.174.17   | 3.733.73   | 3.54       |  | Attr. 19 | 13.7813.78 | 12.2712.27 | 13.8613.86 | 11.7911.79 | 11.29      |
| Attr. 20 | 1.611.61   | 1.611.61   | 1.581.58   | 1.42       | 1.431.43   |  | Attr. 25 | 27.5927.59 | 24.8224.82 | 26.9426.94 | 24.2624.26 | 23.87      |
| Attr. 21 | 7.187.18   | 6.20       | 7.737.73   | 6.916.91   | 6.266.26   |  | Attr. 26 | 3.543.54   | 3.403.40   | 3.783.78   | 3.223.22   | 3.16       |
| Attr. 22 | 4.384.38   | 4.144.14   | 4.084.08   | 3.883.88   | 3.81       |  | Attr. 27 | 26.7426.74 | 22.7422.74 | 26.2126.21 | 23.1223.12 | 22.45      |
| Attr. 23 | 8.328.32   | 6.576.57   | 8.808.80   | 6.546.54   | 6.47       |  | Attr. 28 | 6.146.14   | 5.825.82   | 6.176.17   | 5.435.43   | 5.16       |
| Attr. 24 | 5.015.01   | 5.385.38   | 5.125.12   | 4.634.63   | 4.23       |  | Attr. 29 | 5.555.55   | 5.185.18   | 5.405.40   | 5.135.13   | 4.87       |
| Attr. 30 | 3.293.29   | 3.793.79   | 3.243.24   | 2.94       | 3.033.03   |  | Attr. 35 | 1.151.15   | 1.131.13   | 1.081.08   | 0.94       | 1.081.08   |
| Attr. 31 | 8.058.05   | 7.187.18   | 8.408.40   | 7.217.21   | 6.92       |  | Attr. 36 | 7.917.91   | 7.567.56   | 8.068.06   | 7.477.47   | 7.18       |
| Attr. 32 | 18.2118.21 | 17.2517.25 | 18.1518.15 | 15.93      | 15.93      |  | Attr. 37 | 13.2713.27 | 11.9011.90 | 13.4713.47 | 11.6111.61 | 11.19      |
| Attr. 33 | 16.5316.53 | 15.5515.55 | 16.1916.19 | 13.9313.93 | 13.80      |  | Attr. 38 | 3.803.80   | 3.29       | 4.044.04   | 3.573.57   | 3.513.51   |
| Attr. 34 | 11.1211.12 | 9.769.76   | 11.4611.46 | 10.1710.17 | 9.73       |  | Attr. 39 | 13.2513.25 | 13.4013.40 | 13.7813.78 | 12.2612.26 | 11.95      |

## Appendix C Implementation Details

### C.1 MultiMNIST

We use the MultiMNIST dataset, which overlays multiple images together ([Sabour et al., 2017](#bib.bib45 "")). For each image, a different one is chosen uniformly in random. One of these images is placed at the top-left and the other at the bottom-right. We show sample MultiMNIST images in Figure [6](#A3.F6 "Figure 6 ‣ C.1 MultiMNIST ‣ Appendix C Implementation Details ‣ Multi-Task Learning as Multi-Objective Optimization").

![Refer to caption](1810.04650v2/arch_multi_mnist.png)

Figure 5: Architecture used for MultiMNIST experiments.

![Refer to caption](1810.04650v2/multi_mnist_vis_.png)

Figure 6: Sample MultiMNIST images. In each image, one task (task-L) is classifying the digit on the top-left and the second task (task-R) is classifying the digit on the bottom-right.

For the MultiMNIST experiments, we use an architecture based on LeNet ([LeCun et al., 1998](#bib.bib27 "")). We use all layers except the final one as a shared encoder. We use the fully-connected layer as a task-specific function for the left and right tasks by simply adding two independent fully-connected layers, each taking the output of the shared encoder as input. As a task-specific loss function, we use the cross-entropy loss with a softmax for both tasks. The architecture is visualized in Figure [5](#A3.F5 "Figure 5 ‣ C.1 MultiMNIST ‣ Appendix C Implementation Details ‣ Multi-Task Learning as Multi-Objective Optimization").

The implementation uses PyTorch ([Paszke et al., 2017](#bib.bib37 "")). For all baselines, we searched over the set L​R\={1×10−4,5×10−4,1×10−3,5×10−3,1×10−2,5×10−2}LR=\\{$1\\text{\\times}{10}^{-4}$,$5\\text{\\times}{10}^{-4}$,$1\\text{\\times}{10}^{-3}$,$5\\text{\\times}{10}^{-3}$,$1\\text{\\times}{10}^{-2}$,$5\\text{\\times}{10}^{-2}$\\} of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 256256 and train for 100100 epochs. We report test accuracy.

### C.2 Multi-label classification

For multi-label classification experiments, we use ResNet-18 ([He et al., 2016](#bib.bib18 "")) without the final layer as a shared representation function. Since there are 40 attributes, we add 40 separate 2048×22048\\times 2 dimensional fully-connected layers as task-specific functions. The final two-dimensional output is passed through a 2-class softmax to get binary attribute classification probabilities. We use cross-entropy as a task-specific loss. The architecture is visualized in Figure [7](#A3.F7 "Figure 7 ‣ C.2 Multi-label classification ‣ Appendix C Implementation Details ‣ Multi-Task Learning as Multi-Objective Optimization").

The implementation uses PyTorch ([Paszke et al., 2017](#bib.bib37 "")). We resize each CelebA image ([Liu et al., 2015b](#bib.bib30 "")) to 64×64×364\\times 64\\times 3. For all experiments, we searched over the set L​R\={1×10−4,5×10−4,1×10−3,5×10−3,1×10−2,5×10−2}LR=\\{$1\\text{\\times}{10}^{-4}$,$5\\text{\\times}{10}^{-4}$,$1\\text{\\times}{10}^{-3}$,$5\\text{\\times}{10}^{-3}$,$1\\text{\\times}{10}^{-2}$,$5\\text{\\times}{10}^{-2}$\\} of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 256256 and train for 100100 epochs. We report attribute-wise binary accuracies on the test set as well as the average accuracy.

![Refer to caption](1810.04650v2/arch_multi_label.png)

Figure 7: Architecture used for multi-label classification experiments.

### C.3 Scene understanding

For scene understanding experiments, we use the Cityscapes dataset ([Cordts et al., 2016](#bib.bib9 "")). We resize all images to resolution 256×512256\\times 512 for computational efficiency. As a shared representation function (encoder), we use the ResNet-50 architecture ([He et al., 2016](#bib.bib18 "")) in fully-convolutional fashion. We take the ResNet-50 architecture and only use layers prior to average pooling that are fully convolutional. As a decoder, we use the pyramid pooling module ([Zhao et al., 2017](#bib.bib56 "")) and set the output sizes to 256×512×19256\\times 512\\times 19 for semantic segmentation (1919 classes), 256×512×2256\\times 512\\times 2 for instance segmentation (one output channel for the x-offset of the center location and another channel for the y-offset), and 256×512×1256\\times 512\\times 1 for monocular depth estimation. For instance segmentation, we use the proxy task of estimating the offset for the center location of the instance that encompasses the pixel. We directly estimate disparity instead of depth and later convert it to depth using the provided camera intrinsics. As a loss function, we use cross-entropy with a softmax for semantic segmentation, and MSE for depth and instance segmentation. We visualize the architecture in Figure [8](#A3.F8 "Figure 8 ‣ C.3 Scene understanding ‣ Appendix C Implementation Details ‣ Multi-Task Learning as Multi-Objective Optimization").

We initialize the encoder with a model pretrained on ImageNet ([Deng et al., 2009](#bib.bib11 "")). We use the implementation of the pyramidal pooling network with bilinear interpolation shared by [Zhou et al. (2017a)](#bib.bib57 ""). Ground-truth results for the Cityscapes test set are not publicly available. Therefore, we report numbers on the validation set. As a validation set for hyperparameter search, we randomly choose 275275 images from the training set. After the best hyperparameters are chosen, we retrain with the full training set and report the metrics on the Cityscapes validation set, which our algorithm never sees during training or hyperparameter search. As metrics, we use mean intersection over union (mIoU) for semantic segmentation, MSE for instance segmentation, and MSE for disparities (depth estimation). We directly report the metric in the proxy task for instance segmentation instead of performing a further clustering operation. For all experiments, we searched over the set L​R\={1×10−4,5×10−4,1×10−3,5×10−3,1×10−2,5×10−2}LR=\\{$1\\text{\\times}{10}^{-4}$,$5\\text{\\times}{10}^{-4}$,$1\\text{\\times}{10}^{-3}$,$5\\text{\\times}{10}^{-3}$,$1\\text{\\times}{10}^{-2}$,$5\\text{\\times}{10}^{-2}$\\} of learning rates and chose the model with the highest validation accuracy. We used SGD with momentum, halving the learning rate every 30 epochs. We use batch size 88 and train for 250250 epochs.

![Refer to caption](1810.04650v2/arch_cityscapes.png)

Figure 8: Architecture used for scene understanding experiments.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")