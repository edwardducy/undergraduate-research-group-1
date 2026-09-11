# Do Current Multi-Task Optimization Methods in Deep Learning Even Help?

 Derrick Xin ††thanks: Equal contribution Affiliation: Google Research Affiliation: Mountain View, CA Email: [dxin@google.com](mailto:)    Behrooz Ghorbani\* Affiliation: Google Research Affiliation: Mountain View, CA Email: [ghorbani@google.com](mailto:)    Ankush Garg Affiliation: Google Research Affiliation: Mountain View, CA Email: [ankugarg@google.com](mailto:)    Orhan Firat Affiliation: Google Research Affiliation: Mountain View, CA Email: [orhanf@google.com](mailto:)    Justin Gilmer Affiliation: Google Research Affiliation: Mountain View, CA Email: [gilmer@google.com](mailto:) 

###### Abstract

Recent research has proposed a series of specialized optimization algorithms for deep multi-task models. It is often claimed that these multi-task optimization (MTO) methods yield solutions that are superior to the ones found by simply optimizing a weighted average of the task losses. In this paper, we perform large-scale experiments on a variety of language and vision tasks to examine the empirical validity of these claims. We show that, despite the added design and computational complexity of these algorithms, MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches. We highlight alternative strategies that consistently yield improvements to the performance profile and point out common training pitfalls that might cause suboptimal results. Finally, we outline challenges in reliably evaluating the performance of MTO algorithms and discuss potential solutions.

## 1 Introduction

Multitask models are ubiquitous in deep learning \[[1](#bib.bib1 ""), [2](#bib.bib2 ""), [17](#bib.bib17 "")\]. This popularity stems from the fact that these models can potentially leverage transfer learning in between different tasks and modalities. Moreover, by reducing the number of the models that need to be maintained, multitask models greatly simplify serving users.

Multitask models come with their own challenges and downsides. Different tasks often compete with each other for model capacity, leading to *the task interference* problem. Finding the training setting that strikes the right balance between different tasks is an engineering intensive endeavor that requires extensive trial and error for most realistic setups.

Over the past few years, a vast number of multi-task optimization (MTO) algorithms have been proposed in the literature that claim to alleviate the task interference problem \[[5](#bib.bib5 ""), [9](#bib.bib9 ""), [20](#bib.bib20 ""), [21](#bib.bib21 ""), [26](#bib.bib26 ""), [30](#bib.bib30 ""), [31](#bib.bib31 "")\]. These algorithms typically leverage clever intuitions about the training process to dynamically balance the different tasks throughout training. However, in exchange for this, these algorithms often drastically add to the computational and design complexity of the training process.

The goal of this paper is not to provide another MTO algorithm. Instead, we provide a large-scale empirical study of the algorithms presented in the literature; we examine to what degree the improvements presented in the literature are reproducible and whether these algorithms really reduce loss interference between different tasks. As such, our study contributes to the growing body of literature that aims to provide a reality check on recent algorithmic proposals in the machine learning community \[[11](#bib.bib11 ""), [12](#bib.bib12 ""), [23](#bib.bib23 ""), [28](#bib.bib28 "")\]. We provide the following observations:

*   •

```
Despite the added complexity, MTO algorithms fail to improve the interference profile beyond what is achievable by simple static weighting of the tasks (Section [4](#S4 "4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")).
```
*   •

```
The performance of multi-task models is sensitive to basic optimization parameters such as learning rate and weight-decay. Insufficient tuning of these hyper-parameters in the baselines, along with the complexity of evaluating multi-task models, can create a false perception of performance improvement (Section [4](#S4 "4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")).
```
*   •

```
In some instances, the gains reported in the MTO literature are due to flaws in the experimental design. Often times these reported gains disappear with better tuning of the baseline hyperparameters. In addition, in a handful of cases, we were unable to reproduce the reported results (Section [4.2](#S4.SS2 "4.2 Benchmarks from the Literature ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")).
```
*   •

```
Finally, we discuss the implications for the community and the potential steps that need to be taken to standardize evaluation for multi-task models (Section [5](#S5 "5 Conclusions ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")).
```
## 2 Setting

We focus our discussion on the supervised learning setup, where the model parameters, 𝜽∈ℝp\\boldsymbol{\\theta}\\in\\mathbb{R}^{p}, are trained on KK different tasks. We denote the loss associated with task ii with ℒi​(𝜽)\\mathcal{L}\_{i}(\\boldsymbol{\\theta}).

For some problem instances, the parameter space contains a globally optimal point that achieves the best possible performance on all tasks. Figure [1](#S2.F1 "Figure 1 ‣ 2 Setting ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") (left) provides a cartoon example of one such scenario. However, for most realistic setups, a globally optimal 𝜽\\boldsymbol{\\theta} doesn’t exist. In these cases, different tasks compete with each other for model capacity. In these scenarios, the concept of Pareto optimality is used to capture the optimal trade-off in between the tasks:

###### Definition (Pareto Optimality).

𝜽∈ℝp\\boldsymbol{\\theta}\\in\\mathbb{R}^{p} Pareto dominates another 𝛉′\\boldsymbol{\\theta}^{\\prime} if ∀1≤i≤K\\forall 1\\leq i\\leq K, ℒi​(𝛉)≤ℒi​(𝛉′)\\mathcal{L}\_{i}(\\boldsymbol{\\theta})\\leq\\mathcal{L}\_{i}(\\boldsymbol{\\theta}^{\\prime}) and there exists a task jj where ℒj​(𝛉)<ℒj​(𝛉′)\\mathcal{L}\_{j}(\\boldsymbol{\\theta})<\\mathcal{L}\_{j}(\\boldsymbol{\\theta}^{\\prime}). 𝛉\\boldsymbol{\\theta} is Pareto optimal if it is not dominated by any other point. The collection of the Pareto optimal points is denoted as Pareto front.

Figure [1](#S2.F1 "Figure 1 ‣ 2 Setting ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") (center) provides a cartoon representation of the Pareto front for a two-task setup. The Pareto front represents the collection of parameters that achieve the best possible trade-off profile between the tasks. A practitioner can aim to land on a particular point on this trade-off curve depending on their (implicit or explicit) utility function. The location and the curvature of the Pareto curve represent the severity of the interference problem. Ideally, one would like to identify training protocols that push the trade-off curve towards the origin as much as possible (Figure [1](#S2.F1 "Figure 1 ‣ 2 Setting ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")-right).

Figure 1: A cartoon representation of the achievable trade-offs in a two-task setup. 

The traditional approach to optimize multi-task models is via *scalarization* \[[3](#bib.bib3 "")\]:

𝜽^​(𝒘)\=arg⁡min𝜽⁡ℒ⁡(𝜽,𝒘)whereℒ⁡(𝜽,𝒘)≡∑i\=1K𝒘i​ℒi​(𝜽),𝒘\>0,∑i𝒘i\=1.\\displaystyle\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w})=\\arg\\min\_{\\boldsymbol{\\theta}}\\mathcal{L}(\\boldsymbol{\\theta};\\boldsymbol{w})\\quad\\mbox{where}\\quad\\mathcal{L}(\\boldsymbol{\\theta};\\boldsymbol{w})\\equiv\\sum\_{i=1}^{K}\\boldsymbol{w}\_{i}\\mathcal{L}\_{i}(\\boldsymbol{\\theta}),\\quad\\boldsymbol{w}>0,\\quad\\sum\_{i}\\boldsymbol{w}\_{i}=1.

(1)

Here, 𝒘\\boldsymbol{w} is a fixed vector of task weights determined by the practitioner beforehand. The algorithmic and computational simplicity of this approach has made scalarization highly popular in practice.

Scalarization comes with certain theoretical guarantees. It can be easily shown that any solution to problem ([1](#S2.E1 "In 2 Setting ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")) is guaranteed to be Pareto optimal. In addition, when {ℒi}i\=1K\\{\\mathcal{L}\_{i}\\}\_{i=1}^{K} are convex, there exists a partial converse:

###### Theorem (Informal).

Let 𝛉#\\boldsymbol{\\theta}^{\\#} be a point on the Pareto front. Then there exists 𝐰#≥0\\boldsymbol{w}^{\\#}\\geq 0 such that scalarization with 𝐰#\\boldsymbol{w}^{\\#} yields 𝛉#\\boldsymbol{\\theta}^{\\#}. 11 1 The precise statements and their proofs are provided in the appendix.

These results suggest that, at least for convex setups, sweeping the task weights should be sufficient for full exploration of the Pareto frontier. In particular, in the convex setting it is provable that no algorithm can outperform properly chosen scalarization that has been trained to convergence.

The results above raise a series of questions. Where are the reported improvements of MTO algorithms coming from? Is non-convexity adding additional complexity which makes scalarization insufficient for tracing out the Pareto front? Is it the case that neural networks trained via a combination of scalarization and standard first-order optimization methods are not able to reach the Pareto Frontier? Do MTO algorithms achieve a better performance trade-off curve? In following sections, we empirically examine these questions for several popular deep learning workloads.

## 3 Prior Work

There has been a flurry of research on MTO algorithms over the past few years. \[[13](#bib.bib13 ""), [4](#bib.bib4 "")\] argue that finding the appropriate scalarization weights is often costly. To alleviate this, they provide algorithms that aim to automatically find a reasonable set of task weights. Sener & Koltun (2019) \[[26](#bib.bib26 "")\] approach multi-task learning from a multi-objective optimization view point and suggest Multiple Gradient Descent Algorithm (MGDA) for efficiently finding Pareto optimal solutions. \[[18](#bib.bib18 ""), [21](#bib.bib21 ""), [30](#bib.bib30 ""), [31](#bib.bib31 "")\] hypothesize that negative interactions between the gradients of different tasks is a significant contributor to the interference problem. As such, these studies put forward various suggestions for projecting out conflicting gradients in order to improve the optimization dynamics. Finally, \[[5](#bib.bib5 ""), [20](#bib.bib20 "")\] propose algorithms that inject randomization into the training pipeline and argue that this added randomness improves the training dynamics by allowing the optimization trajectory to escape poor local minima.

It is important to note that MTO algorithms often come with substantial computational overhead. Chen et al. (2020) report 2-5 fold increase in the training time for a 40-task benchmark \[[5](#bib.bib5 "")\]. Similarly, Kurin et al. (2022) observe that on some benchmarks MTO algorithms can train as much as 3535 times slower compared to scalarization \[[16](#bib.bib16 "")\].

Recently, there has been a number of studies that critically question the benefits of MTO algorithms. The closest such study to ours is Kurin et al. (2022) ([16](#bib.bib16 "")) that appeared on Arxiv during the preparation of this manuscript. The paper argues that MTO algorithms implicitly regularize the model and shows that with careful regularization, scalarization with equal weights can match the performance of MTOs on various popular benchmarks.

In contrast, we argue that MTO algorithms yield different solutions on the same trade-off curve (See Figure [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for an example). In most cases, these solutions tend to be different from equal weighting scalarization solution. When performance on popular benchmarks is concerned, we argue that scalarization baselines are often under-tuned. With additional tuning of the hyper-parameters, we find that most optimizers yield comparable results.

## 4 Experiments

### 4.1 Multilingual Machine Translation

In this section, we examine the effect of MTO algorithms on multilingual neural machine translation (NMT). In particular, we focus on translation out of English as prior work has reported significant task interference in this translation direction \[[1](#bib.bib1 "")\].

We start off by examining models trained jointly on English→\\rightarrow{French, Chinese} translation tasks. The two-task setup allows us to effectively visualize the performance trade-off curves. French and Chinese are specifically chosen due to the large difference in their semantic and syntactic structures. Here, we anticipate a large degree of interference among the tasks—a setting where MTOs claim to improve upon scalarization. We repeat our experiments for English→\\rightarrow{French, German} and English→\\rightarrow{French, Romanian} translation tasks to ensure that our observations generalize across different task setups with different levels of data imbalance. See Table [1](#S4.T1 "Table 1 ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for an overview of data sources. All models use (pre-LN) Transformer architecture \[[29](#bib.bib29 "")\] and have been trained using early stopping. See Appendix [A](#A1 "Appendix A NMT Training Setup ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for training details.

Table 1: Overview of data sources used in our NMT experiments. 

| Language Pair        | Dataset    | \# Train Examples | \# Eval Examples |
| -------------------- | ---------- | ----------------- | ---------------- |
| 40,853,29840,853,298 | 4,5034,503 |                   |                  |
| 25,986,43625,986,436 | 3,9813,981 |                   |                  |
| 4,548,8854,548,885   | 2,1692,169 |                   |                  |
| 610,320610,320       | 1,9991,999 |                   |                  |

We compare the performance trade-offs achieved by various popular MTO algorithms with the Pareto frontier of scalarization. Following the NMT literature’s convention, we implement scalarization via proportional sampling. Here, the average number of observations in the batch corresponding to task ii is proportional to 𝒘i\\boldsymbol{w}\_{i}. In this setup, the expected training loss is equal to

ℒ⁡(𝜽)\=𝔼𝒙​\[ℓ⁡(𝒙,𝜽)\]\=∑i\=1Kℙ⁡(𝒙∈task ​i)​𝔼𝒙​\[ℓ⁡(𝒙,𝜽)|𝒙∈task ​i\]\=∑i\=1K𝒘i​ℒi​(𝜽).\\displaystyle\\mathcal{L}(\\boldsymbol{\\theta})=\\mathbb{E}\_{\\boldsymbol{x}}\[\\ell(\\boldsymbol{x};\\boldsymbol{\\theta})\]=\\sum\_{i=1}^{K}\\mathbb{P}(\\boldsymbol{x}\\in\\mbox{task }i)\\mathbb{E}\_{\\boldsymbol{x}}\[\\ell(\\boldsymbol{x};\\boldsymbol{\\theta})|\\boldsymbol{x}\\in\\mbox{task }i\]=\\sum\_{i=1}^{K}\\boldsymbol{w}\_{i}\\mathcal{L}\_{i}(\\boldsymbol{\\theta}).

We compare scalarization with a series of popular MTO algorithms: Multiple Gradient Descent (MGDA) \[[26](#bib.bib26 ""), [7](#bib.bib7 ""), [18](#bib.bib18 "")\], GradNorm \[[4](#bib.bib4 "")\], Gradient Surgery (PCGrad) \[[31](#bib.bib31 "")\], IMTL \[[21](#bib.bib21 "")\], and Random Loss Weighting (RLW) \[[20](#bib.bib20 "")\]. For GradNorm’s α\\alpha hyper-parameter, we perform a grid search and report all non-Pareto dominated models. To give an apples-to-apples comparison, all models have been trained with the same batch-size for the same number of training steps. All models use Adam \[[14](#bib.bib14 "")\] as the base optimizer. For all these optimizer categories, we tune the learning rate on a grid from 5×10−25\\times 10^{-2} to 55 and report all non-Pareto dominated models. Details of the training and hyper-parameters are presented in Appendix [A](#A1 "Appendix A NMT Training Setup ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?").

The overview of our experimental findings are presented in Figures [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), [4](#S4.F4 "Figure 4 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), and [5](#S4.F5 "Figure 5 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"). The blue dashed line corresponds to the Pareto front achieved via proportional sampling with English→\\rightarrowFrench sampling rate ranging from 10% to 90%. Several observations are in order:

##### No Improvements from MTO Algorithms

Despite the promise to alleviate interference among the tasks, all of the MTO algorithms in our study simply yield performance trade-off points on the scalarization Pareto front. As such, their performance can be fully replicated by simply optimizing a weighted average of the losses. To understand this phenomenon better, in Figure [3](#S4.F3 "Figure 3 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), we plot the evolution of the task weights for PCGrad, MGDA, GradNorm, and IMTL during training. We observe that for the majority of the training runs, the dynamically assigned task weights do not move significantly. As such, in effect, these MTO algorithms behave similar to static weighting.

Figure 2: Performance trade-off behavior for En→\\rightarrow{Fr, Zh} models. Each point corresponds to the final performance of a model. We observe no improvements in terms of final performance or training behavior from MTO algorithms. See Appendix [B](#A2 "Appendix B Additional Results ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for more comparisons. 

Figure 3: The evolution of task weights during training. All models are trained with the same (near optimal) base learning rate of 0.50.5. For the majority of the runs, the task weights barely move. 

##### Other Language Pairs

En→\\rightarrowFr and En→\\rightarrowZh are both high-resource tasks with O⁡(107)O(10^{7}) training examples. In these experiments, we observe minimal overfitting and excellent agreement between train and test behaviors. One might argue that MTO algorithms possess transfer learning and regularization capabilities En→\\rightarrow{Zh, Fr} experiments downplay 22 2 See \[[16](#bib.bib16 "")\] for an overview of proposals on how MTO algorithms can perform implicit regularization.. To address this, we repeat our experiments in two new task setups where En→\\rightarrowZh task is substituted with En→\\rightarrowDe (mid-resource) and En→\\rightarrowRo (low-resource).

Figures [4](#S4.F4 "Figure 4 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") & [5](#S4.F5 "Figure 5 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") present the results of these experiments. En→\\rightarrow{De, Fr} experiments closely resemble En→\\rightarrow{Zh, Fr} ones: MTO algorithms simply achieve different trade-off points on the scalarization Pareto front. The results for En→\\rightarrow{Ro, Fr} are more interesting. We still observe a clear Pareto front for the training performance; different MTO algorithms achieve different points on this curve. For the generalization behavior however, the Pareto frontier ceases to exist. Instead, we observe models that are *globally optimal*. Interestingly, these globally optimal solutions are found only by scalarization (with sampling rates close to (0.3,0.7)(0.3,0.7)). In contrast, MTO algorithms find solutions with near equal task weights which yield significantly worse generalization performances. As the generalization performance in this setup is primarily driven by the amount of regularization applied to the low-resource task during training, our results cast doubt on the ability of MTO algorithms to effectively regularize the model.

Figure 4: Performance trade-off behavior for En→\\rightarrow{De, Fr} models. The results closely resemble our observations on En→\\rightarrow{Zh, Fr} experiments. 

Figure 5: Performance trade-off behavior for En→\\rightarrow{Ro, Fr} models. Left: We do not observe a Pareto frontier for the test performance. Instead, scalarization with weights (0.3,0.7)(0.3,0.7) achieves a globally optimal trade-off and outperforms MTO algorithms. Right: For the training loss, a clear Pareto frontier appears and MTO algorithms simply selects a point on the performance profile traced out by the scalarization sweep. 

##### Evaluation Challenges

Our experiments suggest that the model performances are highly sensitive to the choice of hyper-parameters. Even subtle choices regarding the hyper-parameter grid can drastically change the results. For example, it is common practice to tune the learning rate on a sparse grid, say sweeping η∈\[10−3,10−2,10−1\]\\eta\\in\[10^{-3},10^{-2},10^{-1}\]. How much do our reported metrics suffer from such a sparse sweep, and how much performance can be gained on average from further tuning of just this one hyperparameter? To answer this question, we simulate running multiple instances of sparse grid searches of the form {k×10−3,k×10−2,k×10−1}\\{k\\times 10^{-3},k\\times 10^{-2},k\\times 10^{-1}\\} for 1≤k≤91\\leq k\\leq 9; each choice of kk produces a tuning study sweeping over 3 learning rates. We then measure the variance in the optimal performance of each 3 trial study, as kk is varied from 1 to 9. The results are shown in Figure [6](#S4.F6 "Figure 6 ‣ Evaluation Challenges ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") (left). For comparison, we plot the variance in performance resulting from running the best η\\eta multiple times with different seeds. Notably, the effective standard deviation resulting from sparse learning rate tuning is 6 to 7 times the standard deviation observed from varying the random seed for a fixed hyperparameter point. The upshot is, estimating trial variance by rerunning multiple seeds is insufficient for concluding that performance gains from a new algorithm are significant when the hyperparameters are sampled on a sparse grid.

The established convention in the literature for ranking MTO performances is to compare some kind of average of the per-task performances. The specific average used is fixed somewhat arbitrarily for the purposes of benchmarking. However, in practice, the utility function for ranking algorithms may vary dramatically depending on the goals of the practitioner. Thus, useful MTOs need to be robust to changes in the utility function. Either they need to improve upon the performance profile curve traced out by a sampling sweep, or reliably find better points on the profile curve with minimal tuning as the utility function is varied. Unfortunately, the current practice to consider just one (arbitrary) weighting scheme will bias the evaluation towards algorithms that perform well on that specific scheme but are not robust to changes in the utility function. For example, Figure [6](#S4.F6 "Figure 6 ‣ Evaluation Challenges ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") (right) ranks 3 MTOs as the evaluation per-task weighting is varied. As the En→\\rightarrowZh weight is varied, the ranking shifts from MGDA being the best MTO to PCGrad being the best. This is a natural consequence of Figure [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") which shows that different MTOs find different points on the same Pareto front traced out by a sampling sweep. Notably, no algorithm outperforms sampling with a well-chosen sampling ratio.

![Refer to caption](2209.11379v1/ZH_FR_Eval_6.png)

Figure 6: Left: Sparse sampling of learning rates has a significantly larger affect on reported performance than varying the random seed of a particular hyperparameter point. Right: Rankings between algorithms can depend on how tasks are weighted at evaluation time. For example, if the eval performance is ranked according to .9.9 En-Fr + .1.1 En-Zh, then MGDA is preferred to GradNorm and PCGrad. At equal weighting, GradNorm and PCGrad outperform MGDA. For any weighting, all MTOs underperform an optimally chosen sampling scheme.

##### Alternative Approaches

As discussed in Section [3](#S3 "3 Prior Work ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), MTO algorithms often drastically increase the algorithmic and computational complexity of the training process. In our experiments, we observed that the requirement to compute per-task gradients (which is necessary for many MTO algorithms) led to a significant reduction in the number of training steps per second (from ≈12\\approx 12 to ≈5\\approx 5). Given these observations, it is natural to wonder if there are more effective ways to spend this extra compute budget. Figure [7](#S4.F7 "Figure 7 ‣ Alternative Approaches ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") examines how scaling the model size changes the performance trade-off behavior. We examine increasing the model depth by a factor of {1,2,3,4}\\{1,2,3,4\\}. Our largest model achieves on average 5.45.4 training steps per second, which is comparable with the models trained using per-task gradients. Our results suggest that, unlike the observed behavior with MTO algorithms, allocating more compute to scale the model yields consistent improvements across the board. Larger models achieve Pareto fronts strictly to the lower-left of the base model, which corresponds to a performance improvement for all utility functions.

Figure 7: The effect of model size on the Pareto frontier for En→\\rightarrow{Zh, Fr} models. 

Table 2: Overview of models used in the scaling experiment. 

| Model   | Optimizer  | \# Parameters | \# Steps/Sec |
| ------- | ---------- | ------------- | ------------ |
| 120120M | 11.5211.52 |               |              |
| 120120M | 4.814.81   |               |              |
| 142142M | 8.288.28   |               |              |
| 165165M | 6.686.68   |               |              |
| 187187M | 5.485.48   |               |              |

### 4.2 Benchmarks from the Literature

The observations of Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), run contrary to many recent influential studies proposing MTOs for multi-task models \[[4](#bib.bib4 ""), [5](#bib.bib5 ""), [9](#bib.bib9 ""), [26](#bib.bib26 ""), [31](#bib.bib31 "")\]. These papers often compare the performance of their proposed algorithm with traditional training strategies and report significant gains. In this section, we attempt to reproduce these results on a number of supervised-learning benchmarks. We present comparisons for CityScapes \[[6](#bib.bib6 "")\] and CelebA \[[22](#bib.bib22 "")\] datasets in the main text.33 3 See Appendix [C](#A3 "Appendix C Vision Benchmarks ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for more comparisons and details.

For these experiments, we closely follow the experimental setup and the publicly available code from \[[26](#bib.bib26 "")\]. We modified the code sparingly to address bugs, update deprecated libraries, and speed up the data loader. We perform an extensive grid search for learning rate, weight decay, and dropout. All models use early stopping. Our implementation details are presented in the appendix.

### 4.3 CityScapes

CityScapes \[[6](#bib.bib6 "")\] is a dataset for understanding urban street scenes. It is constructed via stereo video sequences from different cities and contains 29752975 training and 500500 validation images. In the multi-task optimization literature, this dataset is popularly cast as a two-task problem with one task being 7-class semantic segmentation and the other being depth estimation. In our experiments, we choose 595595 random samples from the training data to serve as our validation set. This validation set is used for tuning hyper-parameters such as learning rate and weight decay (See appendix for details). We use the original validation set as our test set.

Figure [8](#S4.F8 "Figure 8 ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") provides an overview of our experimental results. Similar to Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), we observe that scalarization solutions form the generalization performance Pareto front. This frontier is observable both for test loss (left) and task specific generalization metrics (right). In both cases, MTO solutions significantly under-perform scalarization.

Figure 8: The generalization performance of different optimizers for CityScapes benchmark. Left: Test segmentation loss vs test depth loss. Points on the lower left side represent better solutions. Right: Test pixel accuracy vs test absolute depth estimation error. Here, points on the upper left side are better solution. In both cases, scalarization solutions form the Pareto frontier.

For CityScapes models, the segmentation task loss is an order of magnitude larger than the depth estimation task loss. This severe loss imbalance causes interesting behaviors to emerge that are worth noting. Figure [9](#S4.F9 "Figure 9 ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") examines the train / test behavior of the different scalarization solutions. Contrary to recent results reported in the literature \[[16](#bib.bib16 "")\], we observe that appropriately balancing the different losses is crucial in achieving a desirable generalization behavior: in Figure [9](#S4.F9 "Figure 9 ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), the majority of the generalization Pareto frontier is populated by models with segmentation task weight less than 0.20.2.

![Refer to caption](2209.11379v1/CityScape_ParetoCurve_Train_Test_2.png)

Figure 9: Scalarization test (left) and train (right) Pareto frontiers for CityScapes dataset. We plot all non-dominated experimental runs per scalarization weight mixture. 

#### 4.3.1 CelebA

CelebA dataset \[[22](#bib.bib22 "")\] is a collection of 200K face images annotated with 4040 attributes. This dataset is a popular benchmark for MTO research where each attribute is treated as a separate binary classification task. In Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), we identified a number of evaluation challenges for MTO algorithms, namely the significance of exact hyper-parameter tuning and the difficulty of comparing models via average performances. These evaluation challenges become highly visible for CelebA.

Figure [10](#S4.F10 "Figure 10 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") presents the overview of our results. We report average performance across the tasks. Our results suggest that scalarization performance is comparable with the performance of popular MTO algorithms. This is in line with recent findings in the literature \[[16](#bib.bib16 "")\]. More importantly, Figure [10](#S4.F10 "Figure 10 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") shows the importance of careful hyper-parameter tuning: Even in the presence of early-stopping, there is significant variation in the final performance of the models that is drastically larger than the effect of the MTO algorithm choice.

Figure 10: For CelebA, the effect of tuning hyper-parameters is much more significant than the effect of the MTO algorithm choice. Each point here corresponds to the performance of an early-stopped model. We vary the learning rate from 10−410^{-4} to 5×10−15\\times 10^{-1} and weight decay from 00 to 5×10−35\\times 10^{-3}. 

Figure 11: Average test error as a function of learning rate for each MTO on CelebA. The performance ranking of the different MTOs is highly dependent on the learning rate. Due to this high variability, studies with sparse learning rate grids can yield misleading conclusions. 

This acute hyperparameter sensitivity can lead to a scenario where insufficient tuning of baseline hyper-parameters gives illusions of significant performance gains. We suspect such evaluation challenges play a prominent role in the significant disagreements we observe in the literature on the effect and ranking of MTO algorithms. Table [3](#S4.T3 "Table 3 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") presents an overview of the results presented in the literature. As the table suggests different papers report wildly different performance for the same algorithm. A large fraction of the reported statistics resemble the quantities we observe on the *validation* dataset. This is concerning as validation performance on CelebA tends to be noisy. If only the validation performance is reported, a combination of early stopping and high evaluation frequency can artificially boost the scores. This artificial boost in scores is clearly visible between the left and the right sides of Figure [10](#S4.F10 "Figure 10 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?").

Table 3: An overview of reported results in the literature for CelebA benchmark. There is significant disagreement between reported statistics from different studies. We suspect improper tuning of baseline hyper-parameters is a likely culprit (compare reported statistics with Figure [10](#S4.F10 "Figure 10 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")). 

| Study         | \[[5](#bib.bib5 "")\] | \[[26](#bib.bib26 "")\] | \[[31](#bib.bib31 "")\] | \[[21](#bib.bib21 "")\] | \[[16](#bib.bib16 "")\] | Ours (Test) | Ours (Validation) |
| ------------- | --------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------- | ----------------- |
| Scalarization | 8.718.71              | 9.629.62                | –                       | 9.999.99                | 9.19.1                  | 9.239.23    | 8.738.73          |
| MGDA          | 10.8210.82            | 8.258.25                | 8.958.95                | 9.969.96                | 9.789.78                | 9.539.53    | 9.249.24          |
| GradNorm      | 8.688.68              | 8.448.44                | –                       | 10.0810.08              | –                       | –           | –                 |
| PCGrad        | 8.728.72              | –                       | 8.698.69                | 10.0110.01              | 9.079.07                | 9.359.35    | 8.898.89          |
| GradDrop      | 8.528.52              | –                       | –                       | –                       | 9.029.02                | –           | –                 |

## 5 Conclusions

In this paper, we presented a large-scale empirical study examining the effects of multi-task optimization methods. It is often assumed that these algorithms enhance the optimization dynamics of multi-task models and yield desirable solutions that cannot be achieved via scalarization. Our results suggest the contrary. Across a variety of language and vision tasks, we showed that scalarization, with appropriate weights, can match both the optimization and the generalization behaviors of MTO algorithms. As such, in effect, scalarization solutions form a superset for MTO solutions. Our experimental results suggest effective exploration of the scalarization solution set might be a more reliable and effective strategy for boosting the model performance (See Fig [5](#S4.F5 "Figure 5 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")).

Our observations suggest the final performance of multi-task models is highly sensitive to the choice of training hyper-parameters. Often times, the effect size associated with subtle design decisions in the choice of the hyper-parameter grid is orders of magnitude larger than the MTO effect size (See Fig [10](#S4.F10 "Figure 10 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")). As such, researchers can unknowingly create the illusion of significant performance gains by simply under-tuning the competing baselines. The fact that different studies are reporting drastically different numbers for the same dataset-algorithm pair (Table [3](#S4.T3 "Table 3 ‣ 4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")) suggests this phenomenon is prevalent in this literature.

##### Limitations and Future Research

Our results suggest that by exploring the scalarization solution space, one can attain performance on par with (or better than) many MTO algorithms. However, the grid search approach we used for computing the scalarization Pareto frontier is computationally prohibitive. Examining strategies for efficiently searching this solution space (such as \[[10](#bib.bib10 ""), [15](#bib.bib15 "")\]) is a fruitful future research direction.

In the paper, we pointed out worrying concerns regarding faulty evaluation and under-tuned baselines. A natural solution to alleviate these problems is to adopt the Common Task Framework (CTF) \[[8](#bib.bib8 ""), [19](#bib.bib19 "")\] to reliably identify and measure algorithmic improvements in multi-task optimization. With the creation of a commonly used competitive benchmark with a proper validation / test split, baselines will naturally become stronger as subsequent papers progressively improve performance—this makes substantial gains more convincing than current practice where baselines rerun by the authors themselves. We postpone the development of such pipeline to future work.

Finally, to keep the discussion tractable, we focused our analysis to supervised learning benchmarks. Whether the same behavior holds for reinforcement learning and self-supervised learning setups is still an open question.

## Acknowledgments and Disclosure of Funding

We thank George E. Dahl, Wolfgang Macherey, and Macduff Hughes for their constructive comments on the initial version of this manuscript. Additionally, we thank Sourabh Medapati and Zachary Nado for their help in debugging our code base. Moreover, we are grateful to Soham Ghosh and Mojtaba Seyedhosseini for valuable discussions regarding the role of MTOs in large-scale models.

## References

*   \[1\] Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, et al. Massively multilingual neural machine translation in the wild: Findings and challenges. arXiv preprint arXiv:1907.05019, 2019.
*   \[2\] Ankur Bapna, Colin Cherry, Yu Zhang, Ye Jia, Melvin Johnson, Yong Cheng, Simran Khanuja, Jason Riesa, and Alexis Conneau. mslam: Massively multilingual joint pre-training for speech and text. arXiv preprint arXiv:2202.01374, 2022.
*   \[3\] Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
*   \[4\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pages 794–803. PMLR, 2018.
*   \[5\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.
*   \[6\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
*   \[7\] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012.
*   \[8\] David Donoho. 50 years of data science. Journal of Computational and Graphical Statistics, 26(4):745–766, 2017.
*   \[9\] Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34, 2021.
*   \[10\] Alex Graves, Marc G Bellemare, Jacob Menick, Remi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. In international conference on machine learning, pages 1311–1320. PMLR, 2017.
*   \[11\] Klaus Greff, Rupesh K Srivastava, Jan Koutník, Bas R Steunebrink, and Jürgen Schmidhuber. Lstm: A search space odyssey. IEEE transactions on neural networks and learning systems, 28(10):2222–2232, 2016.
*   \[12\] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. arXiv preprint arXiv:2007.01434, 2020.
*   \[13\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
*   \[14\] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
*   \[15\] Julia Kreutzer, David Vilar, and Artem Sokolov. Bandits don’t follow rules: Balancing multi-facet machine translation with multi-armed bandits. arXiv preprint arXiv:2110.06997, 2021.
*   \[16\] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and M Pawan Kumar. In defense of the unitary scalarization for deep multi-task learning. arXiv preprint arXiv:2201.04122, 2022.
*   \[17\] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.
*   \[18\] Xian Li and Hongyu Gong. Robust optimization for multilingual translation with imbalanced data. Advances in Neural Information Processing Systems, 34, 2021.
*   \[19\] Mark Liberman. Obituary: Fred jelinek. Computational Linguistics, 36(4):595–599, 2010.
*   \[20\] Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. arXiv preprint arXiv:2111.10603, 2021.
*   \[21\] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. ICLR, 2021.
*   \[22\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.
*   \[23\] Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. Unsupervised domain adaptation: A reality check. arXiv preprint arXiv:2111.15672, 2021.
*   \[24\] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
*   \[25\] Matt Post. A call for clarity in reporting bleu scores. arXiv preprint arXiv:1804.08771, 2018.
*   \[26\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 525–536, 2018.
*   \[27\] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.
*   \[28\] Xin Su, Yiyun Zhao, and Steven Bethard. A comparison of strategies for source-free domain adaptation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8352–8367, 2022.
*   \[29\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
*   \[30\] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. arXiv preprint arXiv:2010.05874, 2020.
*   \[31\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.

## Checklist

1.  1.

    For all authors…

    1.  (a)

        Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? \[Yes\] See Sections [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")

    2.  (b)

        Did you describe the limitations of your work? \[Yes\] See Section [5](#S5 "5 Conclusions ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")

    3.  (c)

        Did you discuss any potential negative societal impacts of your work? \[N/A\]

    4.  (d)

        Have you read the ethics review guidelines and ensured that your paper conforms to them? \[N/A\]
2.  2.

    If you are including theoretical results…

    1.  (a)

        Did you state the full set of assumptions of all theoretical results? \[Yes\] See Appendix

    2.  (b)

        Did you include complete proofs of all theoretical results? \[Yes\] See Appendix
3.  3.

    If you ran experiments…

    1.  (a)

        Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? \[Yes\] See appendix for implementation details. The code will be made public after the review period

    2.  (b)

        Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? \[Yes\] See Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")

    3.  (c)

        Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? \[Yes\] See Figure [6](#S4.F6 "Figure 6 ‣ Evaluation Challenges ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")

    4.  (d)

        Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? \[Yes\] Compute details are provided in the appendix
4.  4.

    If you are using existing assets (e.g., code, data, models) or curating/releasing new assets…

    1.  (a)

        If your work uses existing assets, did you cite the creators? \[Yes\] See Section [4.3.1](#S4.SS3.SSS1 "4.3.1 CelebA ‣ 4.3 CityScapes ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")

    2.  (b)

        Did you mention the license of the assets? \[N/A\]

    3.  (c)

        Did you include any new assets either in the supplemental material or as a URL? \[N/A\]

    4.  (d)

        Did you discuss whether and how consent was obtained from people whose data you’re using/curating? \[N/A\]

    5.  (e)

        Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? \[N/A\]
5.  5.

    If you used crowdsourcing or conducted research with human subjects…

    1.  (a)

        Did you include the full text of instructions given to participants and screenshots, if applicable? \[N/A\]

    2.  (b)

        Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? \[N/A\]

    3.  (c)

        Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? \[N/A\]
## Appendix A NMT Training Setup

In this appendix, we provide full details of our experimental setup for Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"). All models use pre-LN encoder-decoder transformer architecture. The base model, used for the majority of the experiments of this section, has 33 encoder layers and 33 decoder layers. Note that we intentionally chose a small model to exacerbate interference among the tasks and make our experimental setup more favorable to MTO algorithms. Following the NMT literature convention, our models are trained with 0.10.1 label smoothing and 0.10.1 dropout \[[27](#bib.bib27 "")\] for feed-forward and attention layers. We use a sentence piece vocabulary of size 6464K for our models. Table [4](#A1.T4 "Table 4 ‣ Appendix A NMT Training Setup ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") provides the architecture details.

Table 4: Overview of network and optimizer hyper-parameters. 

| Hyper-parameter   |          |
| ----------------- | -------- |
| Feed-forward dim  | 20482048 |
| Model dim         | 512512   |
| Attention heads   | 88       |
| Attention QKV dim | 512512   |
| Label smoothing   | 0.10.1   |
| Dropout           | 0.10.1   |
| Batch-size        | 10241024 |
| Warm-up steps     | 4040K    |

Models are trained using Adam optimizer \[[14](#bib.bib14 "")\] with a fixed batch-size of 10241024. En→\\rightarrow{Zh, Fr} models are trained for 530038530038 steps while the rest of models (due to smaller training data size) are trained for 397529397529 steps. For all the runs, we use 4040K steps of linear warm-up and then use a learning rate schedule of the form

ηt,η:base learning rate,t:training step.\\displaystyle\\frac{\\eta}{\\sqrt{t}},\\qquad\\eta:\\mbox{base learning rate,}\\qquad t:\\mbox{training step.}

For each model run, we sweep for η\\eta in the grid {0.05,0.1,0.5,1.0,2.5,5.0}\\{0.05,0.1,0.5,1.0,2.5,5.0\\}. Often times, η\=0.5\\eta=0.5 yields the optimal performance and η\=5.0\\eta=5.0 diverges. For sampling experiments, we sweep the rate for En→\\rightarrowFr in the grid {i/10}i\=19\\{i/10\\}\_{i=1}^{9}. This determines the rate for the other language pair automatically. As such, to derive each scalarization front, we train a total of 5454 models.

Some of the MTO algorithms under our investigation have algorithm-specific hyper-parameters. In particular, RLW \[[20](#bib.bib20 "")\] requires specifying the task weight distribution and GradNorm \[[4](#bib.bib4 "")\] requires specifying a parameter α\\alpha. For RLW, we examined Gaussian and Dirichlet distributions and presented the results separately in our plots. For GradNorm, we sweep for α\\alpha in the grid {0.25,0.5,0.75,1.0,1.25,1.5}\\{0.25,0.5,0.75,1.0,1.25,1.5\\} and present all non-Pareto dominated models.

When examining the generalization performance (left hand side of Figures [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), [4](#S4.F4 "Figure 4 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), and [5](#S4.F5 "Figure 5 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")) we use early stopping: we evaluate the model every 50005000 steps and use the step that yields the smallest average validation loss for the two tasks. For En→\\rightarrow{Zh, Fr} and En→\\rightarrow{De, Fr} models, it is often the case that the final step is the optimal step. As such early stopping doesn’t significantly change the picture. For En→\\rightarrow{Ro, Fr}, performance statistics change noticeably with early stopping but the overall qualitative picture remains the same. For the training performance (right hand side of Figures [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), [4](#S4.F4 "Figure 4 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), and [5](#S4.F5 "Figure 5 ‣ Other Language Pairs ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?")) we report the final step training statistics.

## Appendix B Additional Results

In this appendix section, we provide additional performance comparisons for NMT models trained in Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?").

Figure 12: The full generalization / optimization performance overview for En→\\rightarrow{Zh, Fr} models. Small dots correspond to Pareto dominated models excluded from Figure [2](#S4.F2 "Figure 2 ‣ No Improvements from MTO Algorithms ‣ 4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") to avoid clutter. Pareto dominated trade-off curves correspond to models trained with suboptimal base learning rate. 

Figure 13: Observations of Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") generalize across the choice of performance metrics. *Left:* Next token prediction error rate evaluated on the validation data. *Right:* Next token prediction error rate evaluated on the training data. 

In order to avoid the artifacts and complexities decoding, in the main text, we used cross-entropy loss as the main evaluation metric for models in Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"). To complete the picture, Figure [14](#A2.F14 "Figure 14 ‣ Appendix B Additional Results ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") examines the quality of generated translations (as measured by (Sacre-)BLEU score \[[24](#bib.bib24 ""), [25](#bib.bib25 "")\]). All translations are generated via Beam-Search with beam size of 44. Note that, for the sake of computational tractability, we do not optimize the decoding algorithm hyper-parameters for each model. As such the performance trade-off frontier is more noisy.

Figure 14: Translation quality of our models as measured by BLEU score. For En→\\rightarrow{Ro, Fr} models (right) scalarization clearly outperforms the rest of the optimizers. 

## Appendix C Vision Benchmarks

We analyze results on three main vision benchmarks used in multi-task optimization, Multi-MNIST \[[26](#bib.bib26 "")\], CelebA \[[22](#bib.bib22 "")\] and CityScapes \[[6](#bib.bib6 "")\]. Multi-MNIST is a two task dataset, which uses the handwritten digits of MNIST but overlays a right digit and a left digit over each other. CelebA is a dataset of celebrity faces and is cast as a 40-task classification problem; each task predicts a different attribute of the face. Finally, CityScapes is a dataset for understanding urban street scenes. In our setting, it is a two task problem with one task being 7-class semantic segmentation and the other being depth estimation.

We would like to thank Lin et al. (2021) \[[20](#bib.bib20 "")\] and Sener et al. (2018) \[[26](#bib.bib26 "")\] for publicly releasing their code. Our CelebA and Multi-MNIST experiments heavily utilize code from Sener et al. and our CityScapes experiments heavily utilize code from Lin et al. For CelebA and Multi-MNIST, our primary changes to the code base include integrating more optimization algorithms, speeding up the dataloaders via the Tensorflow datasets library and creating a validation set for Multi-MNIST by partitioning the training set. Our validation set for Multi-MNIST is 1200012000 images, while our training set is 4800048000 images. We use the original MNIST testing set as our test set, but transformed to a multi-task setting. For CityScapes, we primarily changed the dataloader to have it pre-load images into memory, added statistic tracking for the validation set, and integrated other optimizers.

### C.1 Hyper-Parameter and Experiment Details

##### Multi-MNIST

For all optimizers, we searched through all combinations of learning rate η∈\[0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0\]\\eta\\in\[0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0\], and dropout rate γ∈\[0.1,0.2,0.3,0.4,0.5\]\\gamma\\in\[0.1,0.2,0.3,0.4,0.5\]. We use a LeNet architecture detailed in Sener et al. (2018) \[[26](#bib.bib26 "")\] with two fully-connected layers devoted for each task. For GradNorm specifically, we also search through α∈\[0.5,1.0,1.5,2.0\]\\alpha\\in\[0.5,1.0,1.5,2.0\]. Our learning rate follows a step-wise scheduler with a multiplicative factor of 0.850.85 every 3030 epochs. To create our dataset, we follow steps outlined in Sener et al. (2018), overlaying two random digits on top of each other, one positioned at the top left, and the other at the bottom left. We then resize the image to 28×2828\\times 28. We use batch size of 256256 and SGD with momentum of 0.80.8.

##### CelebA

Similarly our hyper-parameter search for CelebA included all combinations of learning rate η∈\[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0\]\\eta\\in\[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0\] and weight decay λ∈\[0,10−5,5×10−5,10−4,5×10−4,10−3,5×10−3\]\\lambda\\in\[0,10^{-5},5\\times 10^{-5},10^{-4},5\\times 10^{-4},10^{-3},5\\times 10^{-3}\]. For GradNorm, we search through α∈\[0.5,1.0,1.5,2.0\]\\alpha\\in\[0.5,1.0,1.5,2.0\]. Our learning rate schedule was the same as the one for Multi-MNIST and we use a batch size of 256256. For CelebA we also use SGD with momentum of 0.80.8. The model follows the one detailed in Sener et al. (2018).

##### Cityscapes

Here our hyper-parameter search implements something slightly different. We search through all combinations of learning rates η∈\[10−5,10−4.5,10−4,10−3.5,10−3,10−2.5,10−2\]\\eta\\in\[10^{-5},10^{-4.5},10^{-4},10^{-3.5},10^{-3},10^{-2.5},10^{-2}\] and weight decay λ∈\[0,10−6,10−5.5,10−5,10−4.5,10−4,10−3.5,10−3,10−2.5,10−2\]\\lambda\\in\[0,10^{-6},10^{-5.5},10^{-5},10^{-4.5},10^{-4},10^{-3.5},10^{-3},10^{-2.5},10^{-2}\]. For GradNorm, we search through α∈\[0.5,1.0,1.5,2.0\]\\alpha\\in\[0.5,1.0,1.5,2.0\]. We use a batch size of 6464 for all optimizers. We split the training data set of 29752975 images into a validation set of 595595 with the rest being our actual training set and we use the original validation set of 500500 images as our test set. All images are resized to 128×256128\\times 256 and we use Adam \[[14](#bib.bib14 "")\] as our base optimizer. For model we use the architecture utilizing ResNet-50 as a shared encoder detailed in Lin et al. (2021) \[[20](#bib.bib20 "")\].

### C.2 Additional Comparisons

We present additional metrics from Section [4.2](#S4.SS2 "4.2 Benchmarks from the Literature ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") for Cityscapes dataset. The results are presented in Figure [15](#A3.F15 "Figure 15 ‣ C.2 Additional Comparisons ‣ Appendix C Vision Benchmarks ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"). We compute mIOU for segmentation, and for depth estimation we compute absolute error. All models are trained with early stopping on validation data. The experimental results align closely with our findings in Sections [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") and [4.2](#S4.SS2 "4.2 Benchmarks from the Literature ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?").

In figure [16](#A3.F16 "Figure 16 ‣ C.2 Additional Comparisons ‣ Appendix C Vision Benchmarks ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") we also present our results on the Multi-Mnist data set, whose results also align with our previous findings. We see in figure [16](#A3.F16 "Figure 16 ‣ C.2 Additional Comparisons ‣ Appendix C Vision Benchmarks ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?") the performance of MTO algorithms on this benchmark again do not out perform scalarization.

Figure 15: Additional metrics for the generalization performance of different optimizers on the Cityscapes benchmark. We have test segmentation mIOU (y-axis) and test depth absolute error (x-axis).

Figure 16: Test accuracy behavior on Multi-MNIST dataset aligns with our observation of Section [4.1](#S4.SS1 "4.1 Multilingual Machine Translation ‣ 4 Experiments ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"). 

## Appendix D Theorem Statements and Proofs

In Section [2](#S2 "2 Setting ‣ Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"), we briefly discussed theoretical guarantees for Scalarization. In this appendix section, we make these statements explicit. The theorem statements and their proofs closely mirror the discussion in Section 4.7 of \[[3](#bib.bib3 "")\].

###### Theorem 1.

Let 𝛉^​(𝐰)∈arg⁡min𝛉⁡ℒ⁡(𝛉,𝐰)\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w})\\in\\arg\\min\_{\\boldsymbol{\\theta}}\\mathcal{L}(\\boldsymbol{\\theta};\\boldsymbol{w}) for 𝐰\>0\\boldsymbol{w}>0. Then 𝛉^​(𝐰)\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w}) is Pareto optimal.

###### Proof.

Let’s assume the contrary. In this case, by the definition of Pareto optimality, ∃𝜽′\\exists\\boldsymbol{\\theta}^{\\prime} s.t. ∀1≤i≤K\\forall 1\\leq i\\leq K, ℒi​(𝜽′)≤ℒi​(𝜽^​(𝒘))\\mathcal{L}\_{i}(\\boldsymbol{\\theta}^{\\prime})\\leq\\mathcal{L}\_{i}(\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w})) and for at least one task jj, ℒj​(𝜽′)<ℒj​(𝜽^​(𝒘))\\mathcal{L}\_{j}(\\boldsymbol{\\theta}^{\\prime})<\\mathcal{L}\_{j}(\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w})). As such, given that 𝒘\>0\\boldsymbol{w}>0, we have

ℒ⁡(𝜽′,𝒘)\=∑i\=1K𝒘i​ℒi​(𝜽′)<∑i\=1K𝒘i​ℒi​(𝜽^​(𝒘))\=ℒ⁡(𝜽^​(𝒘),𝒘)\\displaystyle\\mathcal{L}(\\boldsymbol{\\theta}^{\\prime};\\boldsymbol{w})=\\sum\_{i=1}^{K}\\boldsymbol{w}\_{i}\\mathcal{L}\_{i}(\\boldsymbol{\\theta}^{\\prime})<\\sum\_{i=1}^{K}\\boldsymbol{w}\_{i}\\mathcal{L}\_{i}(\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w}))=\\mathcal{L}(\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w});\\boldsymbol{w})

which contradicts our assumption that 𝜽^​(𝒘)\\hat{\\boldsymbol{\\theta}}(\\boldsymbol{w}) is a minimizer of the problem. ∎

###### Theorem 2.

Let {ℒi}i\=1K\\{\\mathcal{L}\_{i}\\}\_{i=1}^{K} be convex. Also let 𝛉#\\boldsymbol{\\theta}^{\\#} be an arbitrary point on the Pareto frontier. Then ∃𝐰≥0\\exists\\boldsymbol{w}\\geq 0, 𝐰≠0\\boldsymbol{w}\\not=0 such that 𝛉#∈arg⁡min𝛉⁡ℒ⁡(𝛉,𝐰)\\boldsymbol{\\theta}^{\\#}\\in\\arg\\min\_{\\boldsymbol{\\theta}}\\mathcal{L}(\\boldsymbol{\\theta};\\boldsymbol{w}).

###### Proof.

See Section 4.7.4 of \[[3](#bib.bib3 "")\]. ∎

## Appendix E Compute Resources

For the NMT experiments, we trained a total of 589589 models. Each experiment was trained on Google Cloud Platform v3 TPUs for a period of 12-28 hours. For the vision benchmarks we trained a total of 19601960 models for CityScapes, 10081008 models for CelebA, and 720720 models for Multi-Mnist. Each being trained on an Nvidia A100 GPU.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")