# Examining Common Paradigms in  
Multi-Task Learning

 Cathrin Elich Affiliation: Bosch Center for Artificial Intelligence Affiliation: Max Planck Institute for Intelligent Systems, Tübingen, Germany Affiliation: Max Planck ETH Center for Learning Systems    Lukas Kirchdorfer Affiliation: Bosch Center for Artificial Intelligence Affiliation: University of Mannheim  
cathrin.elich@tuebingen.mpg.de, {jan.koehler,lukas.schott}@bosch.com  
‡Work done during an internship at Bosch.      ∗Joint senior authors.    Jan M. Köhler Affiliation: Bosch Center for Artificial Intelligence    Lukas Schott Affiliation: Bosch Center for Artificial Intelligence 

###### Abstract

While multi-task learning (MTL) has gained significant attention in recent years, its underlying mechanisms remain poorly understood. Recent methods did not yield consistent performance improvements over single task learning (STL) baselines, underscoring the importance of gaining more profound insights about challenges specific to MTL. In our study, we investigate paradigms in MTL in the context of STL: First, the impact of the choice of optimizer has only been mildly investigated in MTL. We show the pivotal role of common STL tools such as the Adam optimizer in MTL empirically in various experiments. To further investigate Adam’s effectiveness, we theoretical derive a partial loss-scale invariance under mild assumptions. Second, the notion of gradient conflicts has often been phrased as a specific problem in MTL. We explore the role of gradient conflicts in MTL and compare it to STL. For angular gradient alignment we find no evidence that this is a unique problem in MTL. We emphasize differences in gradient magnitude as the main distinguishing factor. Overall, we find surprising similarities between STL and MTL suggesting to consider methods from both fields in a broader context.

###### Keywords: 

Multi-task learning Deep Learning Computer Vision 

## 1 Introduction

Multi-task learning (MTL) is gaining significance in the deep learning literature and in industry applications. Especially, tasks like autonomous driving and robotics necessitate real-time execution of neural networks while obeying constraints of limited computational resources. Consequently, there is a demand for neural networks capable of simultaneously inferring multiple tasks \[[19](#bib.bib19 ""), [26](#bib.bib26 "")\].

In a seminal study, Caruana \[[4](#bib.bib4 "")\] highlights both advantages and challenges in MTL. On the one hand, certain tasks can exhibit a symbiotic relationship, resulting in a mutual performance enhancement when trained together. On the other hand, conflicts between tasks can arise and decrease the performance when trained jointly, also known as negative transfer.

Several approaches have been suggested to mitigate the issue of negative transfer among tasks during network training. Our study focuses on two main branches in the literature: First, *gradient magnitude* methods which incorporate weights to scale task-specific losses to achieve an adequate balance between tasks. Second, *gradient alignment* methods which aim to resolve conflicts in gradient vectors that may arise between tasks within a shared network backbone.

The effectiveness of the proposed MTL methods remain uncertain in the literature. Upon comparing various studies, it becomes evident that there is no definitive approach that consistently performs well across different settings \[[48](#bib.bib48 "")\]. This observation has been reinforced in more recent studies where competitive performance was achieved through plain unitary scaling in combination with common regularization methods \[[25](#bib.bib25 "")\] or tuned task weighting \[[50](#bib.bib50 "")\].

The current understanding of MTL still lacks a deeper comprehension of its underlying mechanisms. To address this gap, our study aims to examine commonly held paradigms, such as the choice of optimizer, as well as the notion of gradient alignment and gradient magnitudes. Our contributions are:

*   •

```
The impact of off-the-shelf optimizers has received little attention in MTL benchmarks. We evaluate the Adam \[[22](#bib.bib22 "")\] optimizer and demonstrate its favorable performance over SGD+momentum in various experiments.
```
*   •

```
We provide a potential explanation for Adam’s effectiveness in MTL by theoretically demonstrating a partial invariance w.r.t. to different loss scalings. Similarly, we derive a full invariance for an optimal variation of the well-established used method of uncertainty weighting \[[21](#bib.bib21 "")\].
```
*   •

```
So far gradient *alignment* conflicts have mostly been considered between different tasks \[[56](#bib.bib56 ""), [31](#bib.bib31 ""), [20](#bib.bib20 ""), [7](#bib.bib7 "")\]. We present empirical evidence that conflicts arising from gradient alignment between tasks are not exclusive and can even be more pronounced between different samples within a task.
```
*   •

```
Corroborating the methods proposed to balance gradient *magnitude* conflicts in MTL \[[21](#bib.bib21 ""), [34](#bib.bib34 ""), [32](#bib.bib32 ""), [50](#bib.bib50 "")\], we confirm that gradient magnitudes pose a challenge between tasks and is less pronounced between samples within a task.
```
*   •

```
We examine the presumption of increased robustness on corrupted data as a result of MTL \[[24](#bib.bib24 ""), [37](#bib.bib37 "")\]. We find light evidence that a higher number of tasks can result in improved transferability. Due to page limitations, we moved these results to [App. A6](#Pt0.A6 "Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning"), focusing the more compelling findings in the main text.
```
Overall, we provide a vast set of experiments and theoretical insights which contribute to a more comprehensive understanding of MTL in computer vision.

## 2 Related Work

Work in multi-task learning (MTL) can be roughly divided into three fields:

Network architectures focus on the question of how features should be shared across tasks, e.g. \[[38](#bib.bib38 ""), [34](#bib.bib34 ""), [51](#bib.bib51 ""), [36](#bib.bib36 "")\]. Multi-task optimization (MTO) aims to resolve imbalances and conflicts of tasks during MTL. Task affinities examine a grouping of tasks that should be learned together to benefit from the joint training \[[47](#bib.bib47 ""), [12](#bib.bib12 "")\]. A general overview of recent works in MTL can be found in \[[48](#bib.bib48 ""), [43](#bib.bib43 "")\]. Our work focuses on MTO, which we review more thoroughly in the following.

Gradient magnitude methods prevent the dominance of individual tasks by balancing them with task-specific weights. One line of works are loss-weighting methods. Here, weights are determined before any (task-wise) gradient computation and are used for a weighted aggregation of the tasks’ losses. These methods consider either the task uncertainty (UW) \[[21](#bib.bib21 "")\], rate of change of the losses (DWA, FAMO) \[[34](#bib.bib34 ""), [30](#bib.bib30 "")\], the tasks’ difficulty (DTP) \[[14](#bib.bib14 "")\], validation performance by applying meta-learning (MOML, Auto-λ\\lambda) \[[53](#bib.bib53 ""), [33](#bib.bib33 "")\], or randomly chosen task weights (RLW) \[[28](#bib.bib28 "")\]. In line with these, the geometric mean of task losses has been used to handle the different convergence rates of the tasks \[[8](#bib.bib8 "")\]. An advantage of theses methods is their computational efficiency as the gradient needs to be computed only once for the aggregated loss. Alternatively, other methods consider the task-specific gradients directly, e.g., by normalizing them (GradNorm) \[[6](#bib.bib6 "")\] or propose a hybrid balancing between task-wise loss and gradient scaling (IMTL, DB-MTL) \[[32](#bib.bib32 ""), [27](#bib.bib27 "")\]. Furthermore, there are several adaptions for the multiple-gradient descent algorithm (MGDA) \[[10](#bib.bib10 "")\], e.g. for applying it efficiently in deep learning setups \[[44](#bib.bib44 "")\] or by introducing a stochastic gradient correction \[[11](#bib.bib11 "")\]. Recently, task-wise gradient weights have been estimated by treating MTL as a bargaining problem (Nash-MTL) \[[40](#bib.bib40 "")\], or considering a stability criterion (Aligned-MTL) \[[45](#bib.bib45 "")\]. Crucially, all *gradient magnitude* methods consider scalar weightings of task-wise gradients within the backbone and/or heads. They do not modify the alignment of task-specific gradient vectors.

Gradient alignment methods perform more profound vector manipulations on the task-wise gradients w.r.t. to the network weights of a shared backbone before aggregating them. The underlying assumption indicates conflicting gradients as a major problem in MTL. To address this, GradDrop \[[7](#bib.bib7 "")\] randomly drops gradient components in the case of opposing signs. PCGrad \[[56](#bib.bib56 "")\] proposes to circumvent problems of conflicting gradients by projecting them onto each other’s normal plane. Following this idea, Liu et al. \[[31](#bib.bib31 "")\] propose CAGrad to converge to a minimum of the average loss instead of any point on the Pareto front. RotoGrad \[[20](#bib.bib20 "")\] rotates gradients at the intersection of the heads and backbone to improve their alignment. Shi et al. \[[46](#bib.bib46 "")\] propose to alter the network architecture based on the occurrence of layer-wise gradient conflicts. Lastly, \[[41](#bib.bib41 "")\] use separate optimizers such as SGD and SGD+momentum per task. This is extended to AdaGrad, RMSProp and Adam in AdaTask \[[52](#bib.bib52 "")\].

Recent studies question the effectiveness of optimization-based methods in MTL. Xin et al. \[[50](#bib.bib50 "")\] execute an extensive hyperparameter search to show that simple scalar task-weighting performs equivalent or superior to many aforementioned multi-task optimization methods. Their hyperparameter search not only include the task-weights, but also common deep learning parameters such as the learning rate and regularization. Concurrently, Kurin et al. \[[25](#bib.bib25 "")\] empirically show that fixed task-weights combined with regularization and stabilization techniques yield to equivalent performance compared to sophisticated multi-task optimization methods. Following these, Royer et al. \[[42](#bib.bib42 "")\] examine the role of model capacity for MTL performance as well as the occurrences of gradient conflicts. We extend these critical studies. In particular, we theoretically and empirically demonstrate that the choice of optimizer is crucial and could potentially help to explain discrepancies found in prior studies ([4.1](#S4.SS1 "4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")). We further specifically distinguish between gradient conflicts between tasks and samples ([4.2](#S4.SS2 "4.2 Investigating gradient conflicts between tasks and samples ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")).

## 3 Problem Statement

Multi-task learning addresses the problem of learning a set of TT tasks simultaneously (see e.g. \[[48](#bib.bib48 ""), [4](#bib.bib4 "")\]). It is noteworthy that this setup is occasionally also referred to as multi-label or multi-target learning \[[57](#bib.bib57 "")\] Importantly, this study does not incorporate multi-input data. We consider a supervised learning setup, use a shared backbone architecture, and learn all tasks together. Formally, given input data 𝒳\\mathcal{X}, the goal is to learn a function f𝜽​(𝐱)f\_{\\bm{\\theta}}(\\mathbf{x}) which maps a point 𝐱∈𝒳\\mathbf{x}\\in\\mathcal{X} to each task label yty\_{t} with t\=1,..,Tt=1,..,T. The trainable parameters 𝜽\={ϕ,ψ1:T}\\bm{\\theta}=\\{\\phi,\\psi\_{1:T}\\} consist of shared parameters ϕ\\phi and task-specific parameters ψt\\psi\_{t}. Training a task tt is associated with the loss ℒt​(f𝜽​(x),𝜽)\\mathcal{L}\_{t}(f\_{\\bm{\\theta}}(x);\\bm{\\theta}), e.g., a regression or classification loss. We denote respective gradients on the shared and task-specific parameters with 𝐠tϕ\=∇ϕℒt\\mathbf{g}^{\\phi}\_{t}=\\nabla\_{\\phi}\\mathcal{L}\_{t}, and 𝐠tψ\=∇ψℒt\\mathbf{g}^{\\psi}\_{t}=\\nabla\_{\\psi}\\mathcal{L}\_{t}. When training on multiple tasks, the shared parameters ϕ\\phi needs to be updated w.r.t. all task-wise gradients 𝐠tϕ\\mathbf{g}^{\\phi}\_{t} which requires an appropriate aggregation. A simple solution is to uniformly sum up the task losses ℒ\=∑tℒt\\mathcal{L}=\\sum\_{t}\\mathcal{L}\_{t} which is referred to as Equal Weighting (EW). However, as tasks might be competing against each other, this can result in negative transfer and thus sub-optimal solutions. One way to deal with this difficulty is to adapt the magnitude of task-specific gradients. This can be achieved by weighting tasks during training, e.g., by scaling different losses ℒ\=∑tαt​ℒt\\mathcal{L}=\\sum\_{t}\\alpha\_{t}\\mathcal{L}\_{t}, where αt≥0\\alpha\_{t}\\geq 0. Note that the αt\\alpha\_{t} can change during training. Furthermore, the weighing can also be performed on gradient level to distinguish between shared and task-specific gradients. We refer to those approaches as *gradient magnitude* methods. Interestingly, the relationship between loss weights, network updates and learning rate also depends on the optimizer. We show a derivation for SGD and Adam in [Section A1.2](#Pt0.A1.SS2 "A1.2 SGD: No loss-scale invariance and relationship of learning rate and task weights on a gradient level ‣ Appendix A1 Theoretical insights into multi-task learning dynamics ‣ Examining Common Paradigms in Multi-Task Learning"). Additionally to adapting the gradient magnitude, one can directly adapt the alignment of task-wise gradient vectors within the shared backbone 𝐠\~ϕ\=𝐡⁡(𝐠𝟏ϕ,…,𝐠𝐓ϕ)\\tilde{\\mathbf{g}}^{\\phi}=\\mathbf{h}(\\mathbf{g\_{1}}^{\\phi},...,\\mathbf{g\_{T}}^{\\phi}).

In practice, an optimum for 𝜽\\bm{\\theta} that yields best performance on all tasks often does not exist. Instead, improving performance on some task often yields a performance decrease in another task. To still enable a comparison across network instances in MTL, an instance 𝜽∗\\bm{\\theta}^{\\ast} is called to be Pareto optimal, if there is no other 𝜽′\\bm{\\theta}^{\\prime} such that ℒt​(𝜽′)≤ℒt​(𝜽∗)​∀t\\mathcal{L}\_{t}(\\bm{\\theta}^{\\prime})\\leq\\mathcal{L}\_{t}(\\bm{\\theta}^{\\ast})\\;\\forall t with strict inequality in at least one task. The Pareto front consists of the Pareto optimal solutions.

## 4 Experiments and results

In this section we perform several experiments to gain a more profound understanding of multi-task learning (MTL) in computer vision by questioning common paradigms. We compare the impact of Adam and SGD in MTL in [Sec. 4.1](#S4.SS1 "4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning") and examine the process of gradient similarity in different settings in [Sec. 4.2](#S4.SS2 "4.2 Investigating gradient conflicts between tasks and samples ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning"). Throughout this evaluation, we repeatedly make use of common setups, which we will specify as follows and in more detail in [App. A3](#Pt0.A3 "Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning").

Datasets:  For our experiments, we consider three different datasets that are commonly used for evaluating MTL in computer vision: Cityscapes \[[9](#bib.bib9 "")\] contains images of urban street scenes. In line with previous work, we consider the tasks of semantic segmentation (7 classes) and depth estimation. NYUv2 \[[39](#bib.bib39 "")\] is an indoor dataset for scene understanding which was recorded over 464 different scenes across three different cities. Besides semantic segmentation (13-class) and depth estimation, it also contains the task of surface normal prediction. CelebA \[[35](#bib.bib35 "")\] consists of 200K face images which are labeled with 40 binary attributes.

Networks:  We use network architectures with hard-parameter sharing which consist of a shared backbone and task-specific heads. For the dense prediction tasks on Cityscapes and NYUv2, we compare SegNet\[[1](#bib.bib1 "")\] and DeepLabV3+\[[5](#bib.bib5 "")\]. Experiments on CelebA are performed on a ResNet-18  \[[15](#bib.bib15 "")\] with an additional single linear layer for each head.

Training:  For each method, we follow the loss or gradient aggregation as described in the related work, e.g., for equal weighting all task-specific losses are simply summed up to compute the joint network gradients. The learning rate is tuned separately for each approach. We use the validation set performance of the Δm\\Delta\_{m} metric as early stopping criteria. The Δm\\Delta\_{m} metric \[[36](#bib.bib36 "")\] measures the average relative task performance drop of a method mm compared to the single-task baseline bb using the same backbone and is computed as Δm\=1T​∑t\=1T(−1)lt​(Mm,t−Mb,t)/Mb,t\\Delta\_{m}=\\frac{1}{T}\\sum\_{t=1}^{T}(-1)^{l\_{t}}(M\_{m,t}-M\_{b,t})/M\_{b,t} where lt\=1l\_{t}=1 if a higher value means better for measure M⋅,tM\_{\\cdot,t} of some task metric tt, and 0 otherwise.

### 4.1 Effectiveness of Adam in multi-task learning

Examined paradigm: The impact of the choice of standard optimizer is often disregarded and varies across studies (overview in [Tab. A1](#Pt0.A3.T1 "In CelebA ‣ A3.1 Datasets ‣ Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning")) when comparing MTL methods. For instance, Adam \[[22](#bib.bib22 "")\] was successfully used to show that random/constant weighting of tasks’ losses performs competitive compared to MTO methods \[[28](#bib.bib28 ""), [25](#bib.bib25 ""), [50](#bib.bib50 "")\]. In contrast, many methods proposing adaptive, task-specific weighting methods \[[32](#bib.bib32 ""), [21](#bib.bib21 "")\] use stochastic gradient descent with momentum (SGD+mom). In recent works, the optimizer choice converged to Adam and a fixed learning rate schedule \[[34](#bib.bib34 ""), [56](#bib.bib56 ""), [31](#bib.bib31 ""), [45](#bib.bib45 "")\] without a comparison to SGD+mom.

In this part of our study, we investigate the impact of Adam and SGD+mom in conjunction with common MTO methods. We identify the choice of optimizer as a crucial confounder in the experimental setup. Compared to SGD+mom, we find that the Adam optimizer itself is a quite effective baseline in MTL and can be regarded as a loss weighting method from a theoretical viewpoint.

#### Toy Task Experiment

To get a first impression of the impact of the optimizer and common hyperparameters such as the learning rate, we investigate the impact of Adam and plain gradient descent (GD) in a simple toy task.

GD

Adam

 learning rate 

1.0

0.05

0.001

1.0

0.05

0.001

 EW 

setup from \[[31](#bib.bib31 "")\]

 CAGrad 

Figure 1: Toy task experiment from CAGrad \[[31](#bib.bib31 "")\] for different learning rates and optimizers. Consistent with results from \[[50](#bib.bib50 "")\], we observe that the choice of the learning rate is crucial even for this toy optimization problem. Moreover, it becomes apparent, that selecting Adam over simple gradient decent (GD) yields superior results. The contour lines depict the 2D loss landscape; the optimization trajectories are colored from red to yellow for 100k iteration steps from three different starting points (seeds). 

Approach: We repeat the experiment of Liu et al. \[[31](#bib.bib31 "")\] using their original implementation but further test different learning rates and optimizers. They motivate their gradient alignment method CAGrad with a simple toy optimization problem in which their method reliably converges to the minimum of the average loss, while other MTO approaches would either get stuck (e.g., EW) or only converge to any point on the Pareto front (e.g., PCGrad \[[56](#bib.bib56 "")\], MGDA \[[44](#bib.bib44 "")\]).

Table 1: Maximum number of iterations for all seeds in the toy task experiment from \[[31](#bib.bib31 "")\] to reach the global minimum for varying MTO method, learning rate, and optimizer combination. EW+Adam often shows the fastest convergence to the global minimum. ’-’ denotes that not all seeds converged within 100k iterations. As reported in \[[31](#bib.bib31 "")\], PCGrad often only converges to a point on the Pareto Front. We highlight the best run for each learning rate over all MTO methods. 

|    |        |    | learning rate |     |       |        |   |   | learning rate |        |  |    |    |        |        |        |       |   |
| -- | ------ | -- | ------------- | --- | ----- | ------ | - | - | ------------- | ------ |  | -- | -- | ------ | ------ | ------ | ----- | - |
|    |        |    |               |     |       |        |   |   |               |        |  |    |    |        |        |        |       |   |
| GD |        | EW |               | -   | -     | -      | - | - |               | Adam   |  | EW |    | 26     | 22     | 709    | 9,015 | - |
|    | PCGrad |    | -             | -   | -     | -      | - |   |               | PCGrad |  | 25 | 56 | 34,175 | -      | -      |       |   |
|    | CAGrad |    | 644           | 213 | 8,069 | 20,418 | - |   |               | CAGrad |  | 27 | 32 | 802    | 11,239 | 57,700 |       |   |

Result: For higher learning rates with Adam optimizer, even equal weighting (EW) reaches the global optimum (cf. [Fig. 1](#S4.F1 "In Toy Task Experiment ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning"), e.g., EW+Adam, lr=0.05) and often converges even faster than dedicated MTO methods ([Tab. 1](#S4.T1 "In Toy Task Experiment ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")). Note, original results were shown for learning rate 0.001 using Adam and were, thus, in favor of CAGrad. Results for additional learning rates are reported in [Tab. A3](#Pt0.A4.T3 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning").

Conclusion: The choice of optimizer appears to be more important on the success of the outcome of this experiment than the choice of MTO method, as Adam converges considerably faster and more reliably than GD. Also, tuning the learning rate is a relevant factor, however, especially in MTL with differently scaled losses, a single suitable learning rate for all tasks does often not exist.

#### Experiments on Cityscapes and NYUv2

We test the effectiveness of Adam and its role as a confounder in common MTL datasets for various MTO methods.

Approach: We compare Adam and SGD+mom in combination with any MTO method from equal weighting (EW), uncertainty weighting (UW) \[[21](#bib.bib21 "")\], random loss weighting (RLW) \[[28](#bib.bib28 "")\], PCGrad \[[56](#bib.bib56 "")\], CAGrad \[[31](#bib.bib31 "")\], IMTL \[[32](#bib.bib32 "")\] and Aligned-MTL \[[45](#bib.bib45 "")\], for which we used the implementation from  \[[29](#bib.bib29 "")\], as well as MTL-IO \[[41](#bib.bib41 "")\] and AdaTask \[[52](#bib.bib52 "")\]. We distinguish between any combination of dataset {Cityscapes \[[9](#bib.bib9 "")\], NYUv2 \[[39](#bib.bib39 "")\]} and network architecture {SegNet \[[1](#bib.bib1 "")\], DeepLabV3 \[[5](#bib.bib5 "")\]}. We run experiments for ten different initial learning rates from \[0.5,0.1,0.05,…,0.00001\]\[0.5,0.1,0.05,...,0.00001\] and select the best one w.r.t. to the validation performance. More details are described in [Sec. A3.2](#Pt0.A3.SS2.SSS0.Px1 "Effectiveness of Adam in MTL. ‣ A3.2 Training ‣ Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning"). As different models and parameter setups can show preference towards different tasks and metrics, we are interested in those models which are Pareto optimal (PO).

Table 2: Number of Pareto optimal (PO) experiments using either Adam or SGD+mom. as optimizer. Models trained with Adam are consistently more often on the Pareto front compared to those trained with SGD+mom. The number of Adam-based runs that are not dominated by any SGD-based run (PO w.r.t. SGD) is even higher, while the reverse does not apply. 

|            |           | Adam |    | SGD+mom. |  |   |   |
| ---------- | --------- | ---- | -- | -------- |  | - | - |
|            |           |      |    |          |  |   |   |
| Cityscapes | SegNet    |      | 5  | 24       |  | 0 | 0 |
| Cityscapes | DeepLabV3 |      | 10 | 24       |  | 0 | 0 |
| NYUv2      | SegNet    |      | 11 | 21       |  | 1 | 1 |
| NYUv2      | DeepLabV3 |      | 16 | 21       |  | 6 | 6 |

![Refer to caption](2311.04698v5/adamVSsgd_parallelcoordplot_DatasetCityScapes_AugmentationTrue_ArchitectureSegNet_weightdecay1e-05_BS64__new.png)

(a) SegNet

![Refer to caption](2311.04698v5/adamVSsgd_parallelcoordplot_DatasetCityScapes_AugmentationTrue_ArchitectureDeepLabV3_resnet50_weightdecay1e-05_BS64__new.png)

(b) DeepLabV3

Figure 2: Parallel coordinate plot over all experiments on Cityscapes. We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation. We observe that Adam clearly outperforms the usage of SGD+mom. 

Results: We observe over all experimental setups that Adam performs favorably over SGD+mom ([Tab. 2](#S4.T2 "In Experiments on Cityscapes and NYUv2 ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")). This especially holds true for experiments on Cityscapes where the Pareto front for both network architectures only consists of Adam-based models. Moreover, an even larger number of Adam-based models is not dominated by any model trained with SGD+mom (PO w.r.t. SGD). For NYUv2, Adam still performs stronger but SGD+mom. also occasionally delivers a PO result. For the individual metrics, the predominance of Adam is further visualized in a parallel coordinate plot in [Figs. 2](#S4.F2 "In Experiments on Cityscapes and NYUv2 ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning") and [A4](#Pt0.A4.F4 "Figure A4 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"). Bold lines indicate the overall Pareto optimal experiments (PO full).

In [App. A4](#Pt0.A4 "Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), we further report best Δm\\Delta\_{m} results for common MTO methods in combination with Adam or SGD+mom ([Tabs. A5](#Pt0.A4.T5 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A6](#Pt0.A4.T6 "Table A6 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") and [A7](#Pt0.A4.T7 "Table A7 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning")). Again, Adam boosts the overall performance across methods. Furthermore, when comparing the ranking of MTO methods w.r.t. the Δm\\Delta\_{m} metric, we see that the order can change based on the choice of optimizer, e.g., for Cityscapes with SegNet the best method with Adam is UW but with SGD+mom it is CAGrad. This underlines the importance of the choice of optimizer as a confounder in the experimental setup. Noteworthy, EW with Adam yields Pareto optimal results in three of the four setups (cf. [Tab. A2](#Pt0.A4.T2 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning")) and is not dominated by any specialized MTO method trained in combination with SGD+mom for all dataset and network combinations. This supports claims questioning the effectiveness of specific MTO methods \[[25](#bib.bib25 ""), [50](#bib.bib50 "")\]. Nonetheless, looking at the Δm\\Delta\_{m} metric and individual metrics, we see that sometimes with a small relative performance drop on one metric, significant gains on another metric can be achieved (e.g., Cityscapes+sem.seg. and depth for UW vs EW).

Conclusion: Not only a well-tuned learning rate but also the optimizer is crucial for MTL performance. In a fair and extensive experimental comparison, we were able to show that Adam shows superior performance in MTL setup compared to SGD+mom.

#### The reasonable effectiveness of Adam in the context of uncertainty weighting

We show that Adam’s mechanism to estimate a parameter-specific learning rate is partially loss-scale invariant and hypothesize that this could contribute to Adams effectiveness in MTL. We demonstrate this partial invariance theoretically and empirically. Furthermore, a full loss-scale invariance can also be shown under mild assumptions for UW \[[21](#bib.bib21 "")\], which is among the most prevalent loss weighting method in the literature, and related similar variant \[[27](#bib.bib27 "")\].

The loss-scale invariance of UW can be shown by assuming an optimal solution for the σ\\sigma values similar to \[[23](#bib.bib23 "")\]. This assumption is mild as this is a 1-dimensional convex optimization problem for each σ\\sigma. The invariance can be demonstrated by inserting the analytical solution starting from UW. For example, assuming a Laplacian distribution (this can be shown for other distributions as well), we have

minσt⁡1σt​ℒt+log⁡σt⇒σt\=ℒt\\min\_{\\sigma\_{t}}\\frac{1}{\\sigma\_{t}}\\mathcal{L}\_{t}+\\log\\sigma\_{t}\\;\\Rightarrow\\;\\sigma\_{t}=\\mathcal{L}\_{t}

(1)

The left hand side shows the typical form of UW, as shown for a Gaussian in \[[21](#bib.bib21 ""), eq.(5)\]. Here, ℒt\\mathcal{L}\_{t} is a task-specific loss and σt\\sigma\_{t} is a scalar parameter that is usually learned. Plugging back the optimal solution for σt\\sigma\_{t}, we get

ℒ\=∑tℒts​g​\[ℒt\]+c,\\mathcal{L}=\\sum\_{t}\\frac{\\mathcal{L}\_{t}}{sg\[\\mathcal{L}\_{t}\]}+c,

(2)

where s​gsg is the stop-gradient operator and cc is a constant that can be omitted during optimization. Given this, we directly see the invariance w.r.t. loss-scalings. For instance, with ℒ1→α1​ℒ1\\mathcal{L}\_{1}\\rightarrow\\alpha\_{1}\\mathcal{L}\_{1} and ℒ2→α2​ℒ2\\mathcal{L}\_{2}\\rightarrow\\alpha\_{2}\\mathcal{L}\_{2}, the derivative of the total loss ℒ\\mathcal{L} remains unchanged. As this invariance is shown on the loss-level, it holds for all gradient updates w.r.t. the head and backbone. Intuitively, this could explain why UW performs strongly in the context of various loss scalings such as measuring depth in centimeters or meters. Further details, are in [App. A1](#Pt0.A1 "Appendix A1 Theoretical insights into multi-task learning dynamics ‣ Examining Common Paradigms in Multi-Task Learning").

Similarly, for Adam, we can prove a partial scale invariance of losses in MTL that holds for the parameters of network heads. As before, we assume a hydra-like network architecture with a shared backbone and task-specific heads. We start with the parameter-update rule from Adam and scale the corresponding losses ℒt→αt​ℒt\\mathcal{L}\_{t}\\rightarrow\\alpha\_{t}\\mathcal{L}\_{t}. When only considering the parameters of the corresponding heads ψt\\psi\_{t}, the scalings αt\\alpha\_{t} cancel out

ψt,i\=ψt,i−1−γαt2​vt′^​αt​mt′^.\\psi\_{t,i}=\\psi\_{t,i-1}-\\frac{\\gamma}{\\sqrt{\\alpha\_{t}^{2}\\hat{v^{\\prime}\_{t}}}}\\alpha\_{t}\\hat{m^{\\prime}\_{t}}.

(3)

Thus, for the network heads, we see a similar effect as for optimal UW that different scalings do not impact the network update. However, this does not hold for the backbone. The full derivation is shown in [App. A1](#Pt0.A1 "Appendix A1 Theoretical insights into multi-task learning dynamics ‣ Examining Common Paradigms in Multi-Task Learning"). We confirm empirically in a handcrafted loss-scaling experiment in [App. A2](#Pt0.A2 "Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting ‣ Examining Common Paradigms in Multi-Task Learning") and [Figs. A1](#Pt0.A2.F1 "In Results for fixed backbone ‣ Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting ‣ Examining Common Paradigms in Multi-Task Learning") and [A2](#Pt0.A2.F2 "Figure A2 ‣ Results for free backbone ‣ Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting ‣ Examining Common Paradigms in Multi-Task Learning") that SGD does not offer any scaling invariance, whereas Adam involves the invariance property for the heads. The optimal UW demonstrates a scaling invariance for the heads and the backbone.

We would like to note that our derivation for Adam is only valid for constant αt\\alpha\_{t}, e.g., measuring depth in different units or unitary weightings \[[50](#bib.bib50 ""), [25](#bib.bib25 "")\]. In case of dynamic loss weights that are not constant (e.g., UW), the weights do not cancel out fully due to the accumulation of gradient histories within Adam. Nonetheless, this has profound implications for loss weighting methods that are used in conjunction with Adam. For instance, when turning off the history within Adam (by setting β1,2\=0\\beta\_{1,2}=0) and having a fixed backbone, all loss weighting methods, such as UW, RLW, and others, become equivalent to equal weighting.

Additional ablations to our previous experiments suppport the relevance of invariance in MTL (cf. [Tabs. A6](#Pt0.A4.T6 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A7](#Pt0.A4.T7 "Table A7 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") and [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning")). First, we compare to signSGD+mom \[[3](#bib.bib3 "")\] which only updates on the sign of gradients and is therefore trivially scale-invariant in the heads. In a direct comparison with SGD+mom, we observe a superiority of signSGD for a majority of tested setups. Next, we applied task-specific Adam optimizers as in AdaTask \[[52](#bib.bib52 "")\] for a full loss-scale invariance and to allow an estimate of task-specific momentum and squared gradient accumulation. This is Pareto dominant over plain Adam+EW in almost all cases and significantly improves the Δm\\Delta\_{m} metric.

Conclusion: In the context of MTL, we derive and measure a full loss-scale invariance for an optimal UW and a partial invariance for Adam. This partial invariance does not hold for SGD+mom and could explain, among other properties, the effectiveness of Adam in MTL. Furthermore, when comparing different loss weighting methods, it is crucial to be aware of the influence of the optimizer.

### 4.2 Investigating gradient conflicts between tasks and samples

Examined paradigm: The field of MTL strongly focuses on resolving conflicts between tasks, especially from a perspective of gradient conflicts \[[31](#bib.bib31 ""), [56](#bib.bib56 ""), [20](#bib.bib20 "")\]. In computer vision, tasks are often defined on a conceptional level such as segmentation and depth (Cityscapes), or recognizing multiple attributes (CelebA). However, in principle, conflicts can not only occur between tasks but also between samples within a task.

![Refer to caption](2311.04698v5/cat_collage.png)

Figure 3: High intra-task diversity can mimic MTL.

We argue that in an extreme case, even recognizing a single cat in multiple images could be considered MTL. For instance, in one image, the cat could be hiding behind a plant and only revealing its eyes, requiring a neural network to recognize the cat solely based on the eyes. In other images, the cat might only reveal its paws, front of a bright window, or might be tired and curled up into a furry ball because we took so many pictures. This would require a paw, shape or fur classifier. Thus, a neural network is required to recognize multiple attributes to reliably recognize our cat. We note that this is conceptually similar to the commonly \[[44](#bib.bib44 ""), [56](#bib.bib56 "")\] considered MTL dataset CelebA \[[35](#bib.bib35 "")\] which requires attribute detection such as wavy hair, mustache or hat, but within one image.

Motivated by this example, we would like to quantify inter-task and inter-sample conflicts in common datasets from a perspective of the MTO literature, which inspects gradient conflicts in neural networks. In particular, we challenge the sole focus on *inter-task* gradients conflicts in MTL. While several works follow the idea of overcoming gradient conflicts in MTL  \[[31](#bib.bib31 ""), [20](#bib.bib20 ""), [46](#bib.bib46 "")\], their appearance has only been mildly investigated so far.

Prerequisite: We compare gradients w.r.t. network weights for different tasks tt and samples 𝐱i\\mathbf{x}\_{i}. The alignment of two gradients 𝐠,𝐠′\\mathbf{g},\\mathbf{g^{\\prime}} on the shared parameters, e.g., of task aa and task bb, is compared with the cosine similarity

Sc​o​s​(𝐠,𝐠′)\=cos⁡(ϕ)\=𝐠⋅𝐠′‖𝐠‖​‖𝐠′‖.S\_{cos}(\\mathbf{g},\\mathbf{g^{\\prime}})=\\cos(\\phi)=\\frac{\\mathbf{g}\\cdot\\mathbf{g^{\\prime}}}{\\|\\mathbf{g}\\|\\|\\mathbf{g^{\\prime}}\\|}.

(4)

Thus, two gradients are in conflict, if their cosine similarity is smaller than zero \[[56](#bib.bib56 "")\]. In particular, Sc​o​sS\_{cos} is 11/−1-1 if gradients point in the same/opposite direction and 00 in case of orthogonal directions. The gradient magnitude similarity

Sm​a​g​(𝐠,𝐠′)\=2​‖𝐠‖2⋅‖𝐠′‖2‖𝐠‖22+‖𝐠′‖22S\_{mag}(\\mathbf{g},\\mathbf{g^{\\prime}})=\\frac{2\\|\\mathbf{g}\\|\_{2}\\cdot\\|\\mathbf{g^{\\prime}}\\|\_{2}}{\\|\\mathbf{g}\\|\_{2}^{2}+\\|\\mathbf{g^{\\prime}}\\|\_{2}^{2}}

(5)

as defined in \[[56](#bib.bib56 "")\], yields values close to 1 for gradients of similar magnitude, or close to 0 for large discrepancies in magnitude. High dissimilarity in both gradient direction and magnitude is presumed to be a common MTL problem.

| SegNet |
| ------ |
|        |

(a) Cityscapes

| SegNet |
| ------ |
|        |

(b) NYUv2

ResNet50

(c) CelebA

Figure 4: Gradient similarities and conflicts for different datasets and network architectures over training epochs. For each dataset and network combination, we report (from left to right) gradient cosine similarity, gradient magnitude similarity, and the ratio of conflicting gradient parameters w.r.t. gradient pairs corresponding to either inter-samples (fixed task) or inter-tasks (fixed sample). We report mean (solid line), standard deviation (shaded area), upper (97.5%97.5\\%) and lower (2.5%2.5\\%) percentile (dotted line) within an epoch. Overall, the direction conflicts are similar (first / last column), whereas the magnitude differences are more pronounced in MTL (middle column). 

Approach: During the training on aforementioned datasets, we examine gradient similarity across two different setups: (1) between gradients of different tasks with respect to a single sample (inter-task), e.g., 𝐠\=∇ϕL0​(f𝜽​(𝐱i))\\mathbf{g}=\\nabla\_{\\phi}L\_{0}\\left(f\_{\\bm{\\theta}}(\\mathbf{x}\_{i})\\right) and 𝐠′\=∇ϕL1​(f𝜽​(𝐱i))\\mathbf{g^{\\prime}}=\\nabla\_{\\phi}L\_{1}\\left(f\_{\\bm{\\theta}}(\\mathbf{x}\_{i})\\right); and (2) between gradients corresponding to the same task but different samples within a batch (inter-sample), e.g., 𝐠\=∇ϕLt​(f𝜽​(𝐱0))\\mathbf{g}=\\nabla\_{\\phi}L\_{t}\\left(f\_{\\bm{\\theta}}(\\mathbf{x}\_{0})\\right) and 𝐠′\=∇ϕLt​(f𝜽​(𝐱1))\\mathbf{g^{\\prime}}=\\nabla\_{\\phi}L\_{t}\\left(f\_{\\bm{\\theta}}(\\mathbf{x}\_{1})\\right). For both setups, we compute the gradient cosine similarity and gradient magnitude similarity as well as the ratio of conflicting gradient parameters. We are aware that our comparison between samples and tasks is not direct. Nonetheless, it serves as a coarse indicator to estimate their impact during network training. Implementation details are in [App. A3](#Pt0.A3 "Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning").

Results: We show the evolvement of the gradient similarity measures over epochs in [Fig. 4](#S4.F4 "In 4.2 Investigating gradient conflicts between tasks and samples ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning"). Surprisingly, when comparing inter-sample (red line) and inter-task (blue line), we find no consistent evidence for gradient alignment conflicts (left column) to be an exclusive problem of having multiple tasks. For instance, for Cityscapes, the variation of gradient alignment is fully encapsulated within the spread we observe in inter-sample variation (task is fixed). For CelebA, the converse seems to be mostly the case. Furthermore, the choice of network architecture and distribution of task-specific and shared parameters (SegNet vs. DeepLabV3) can have a large influence on the spread of the cosine-similarity. Both architectures have roughly a similar number of shared-parameters. However, DeepLabV3 has a higher number of task-specific parameters which seems to reduce the variance in conflicts for both inter-sample and inter-task (row one vs. two). In line with these observations, we found a similar number of conflicting gradient parameters (third column) for both inter-sample and inter-task comparisons among all experiments.

For gradient magnitude similarities (middle column), we observe a clearer pattern. The similarity in magnitudes are continuously (in the mean) less pronounced for the inter-task setup compared to inter-samples (blue line is below red one in all settings). Interestingly, the relative difference between the two setups remains similar over training which justifies the choice of fixed scalar task weightings as done in \[[50](#bib.bib50 "")\]. Further measures can be found in [Figures A6](#Pt0.A5.F6 "In Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning"), [A5](#Pt0.A5.F5 "Figure A5 ‣ Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning") and [A7](#Pt0.A5.F7 "Figure A7 ‣ Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning").

Conclusion: We find that the difficulty of MTL (inter-task and inter-sample) as opposed to STL (inter-sample only) is predominantly due to differences in gradient *magnitudes*. Balancing different magnitudes is tackled in the literature, e.g., \[[21](#bib.bib21 ""), [50](#bib.bib50 "")\] The problem of conflicting gradients has been typically associated with task-specific conflicts \[[56](#bib.bib56 ""), [31](#bib.bib31 ""), [20](#bib.bib20 "")\], here, we find that gradient *alignment* conflicts can actually be even more pronounced between samples. On the one hand, these observations are along the same lines as findings by Royer et al. \[[42](#bib.bib42 "")\] who reason that ’correcting conflicting gradients \[between tasks\] at every training iteration can be superfluous’. On the other hand, gradient-alignment methods in MTL could be considered not only in the context of task-specific conflicts but also for conflicts between samples. Interestingly, previous work has explored the potential benefit of not only learning weights per task but also per sample in the dataset \[[49](#bib.bib49 "")\]. While our experiments show a relatively high similarity in gradient magnitude across samples and, thus, don’t motivate a sample-wise loss weighting, this could, however, be due to only little disruptive noise within data samples which has been the main motivation of \[[49](#bib.bib49 "")\].

## 5 Conclusion and outlook

This study aims to enhance our understanding of multi-task learning (MTL) in computer vision, providing valuable insights for future research as well as guidance for implementations of real-world applications.

We show that common optimization methods from single task learning (STL) like the Adam optimizer are effective in MTL problems. Next, we compare gradient conflicts during training between tasks and samples. While gradient *magnitudes* are a specific problem between tasks (MTL) and thus justify the need for multi-task specific methods for automatic loss weighting, we find the variability in gradient *alignment* to be similar between samples and tasks. Thus, we encourage a more unified viewpoint in which specific MTO methods are also considered in single-task problems and vice versa.

Beyond our work, we encourage to improve the understanding of challenges and paradigms specific to MTL. For instance, our understanding of task (and sample) specific capacity allocation within a network and how best to tune it to custom requirements, is still not thoroughly understood. Often task-weights are increased to assign more importance to a task which is in contrast to tuning the learning rate per task where a smaller learning rate can be beneficial. Thus, we require further investigations and disentanglement of these two concepts.

## Acknowledgments

We thank Claudia Blaiotta, Martin Rapp, Frank R. Schmidt, Leonhard Hennicke, and Bastian Bischoff for their feedback and valuable discussions. Cathrin Elich thanks her supervisors, Jörg Stückler and Marc Pollefeys, for enabling the opportunity to pursue an internship during her Ph.D. studies.

The Bosch Group is carbon neutral. Administration, manufacturing and research activities do no longer leave a carbon footprint. This also includes GPU clusters on which the experiments have been performed.

## References

*   \[1\] Badrinarayanan, V., Kendall, A., Cipolla, R.: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2017)
*   \[2\] Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European conference on computer vision (ECCV). pp. 456–473 (2018)
*   \[3\] Bernstein, J., Wang, Y.X., Azizzadenesheli, K., Anandkumar, A.: signSGD: Compressed optimisation for non-convex problems. In: Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 560–569. PMLR (10–15 Jul 2018)
*   \[4\] Caruana, R.: Multitask learning. Machine learning 28, 41–75 (1997)
*   \[5\] Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Computer Vision – ECCV 2018 (2018)
*   \[6\] Chen, Z., Badrinarayanan, V., Lee, C., Rabinovich, A.: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: Dy, J.G., Krause, A. (eds.) Proc. of ICML. Proceedings of Machine Learning Research, vol. 80, pp. 793–802. PMLR (2018)
*   \[7\] Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., Anguelov, D.: Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)
*   \[8\] Chennupati, S., Sistu, G., Yogamani, S., A Rawashdeh, S.: Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)
*   \[9\] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3213–3223. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.350
*   \[10\] Désidéri, J.A.: Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique 350, 313–318 (2012)
*   \[11\] Fernando, H.D., Shen, H., Liu, M., Chaudhury, S., Murugesan, K., Chen, T.: Mitigating gradient bias in multi-objective learning: A provably convergent approach. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 (2023)
*   \[12\] Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., Finn, C.: Efficiently identifying task groupings for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 27503–27516 (2021)
*   \[13\] Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)
*   \[14\] Guo, M., Haque, A., Huang, D.A., Yeung, S., Fei-Fei, L.: Dynamic task prioritization for multitask learning. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)
*   \[15\] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 770–778. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.90
*   \[16\] Hendrycks, D., Dietterich, T.G.: Benchmarking neural network robustness to common corruptions and perturbations. In: Proc. of ICLR. OpenReview.net (2019)
*   \[17\] Hu, Z., Zhao, Z., Yi, X., Yao, T., Hong, L., Sun, Y., Chi, E.: Improving multi-task generalization via regularizing spurious correlation. Advances in Neural Information Processing Systems 35, 11450–11466 (2022)
*   \[18\] Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., Madry, A.: Adversarial examples are not bugs, they are features. In: Wallach, H.M., Larochelle, H., Beygelzimer, A., d’Alché-Buc, F., Fox, E.B., Garnett, R. (eds.) Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada. pp. 125–136 (2019)
*   \[19\] Ishihara, K., Kanervisto, A., Miura, J., Hautamäki, V.: Multi-task learning with attention for end-to-end autonomous driving. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2021, virtual, June 19-25, 2021. pp. 2902–2911. Computer Vision Foundation / IEEE (2021). https://doi.org/10.1109/CVPRW53098.2021.00325
*   \[20\] Javaloy, A., Valera, I.: Rotograd: Gradient homogenization in multitask learning. In: Proc. of ICLR. OpenReview.net (2022)
*   \[21\] Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 7482–7491. IEEE Computer Society (2018). https://doi.org/10.1109/CVPR.2018.00781
*   \[22\] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) Proc. of ICLR (2015)
*   \[23\] Kirchdorfer, L., Elich, C., Kutsche, S., Stuckenschmidt, H., Schott, L., Köhler: Analytical uncertainty-based loss weighting in multi-task learning. In: German Conference on Pattern Recognition (2024)
*   \[24\] Klingner, M., Bar, A., Fingscheidt, T.: Improved noise and attack robustness for semantic segmentation by using multi-task training with self-supervised depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 320–321 (2020)
*   \[25\] Kurin, V., De Palma, A., Kostrikov, I., Whiteson, S., Kumar, M.P.: In Defense of the Unitary Scalarization for Deep Multi-Task Learning. In: Neural Information Processing Systems (2022)
*   \[26\] Lee, D.G.: Fast drivable areas estimation with multi-task learning for real-time autonomous driving assistant. Applied Sciences 11(22), 10713 (2021)
*   \[27\] Lin, B., Jiang, W., Ye, F., Zhang, Y., Chen, P., Chen, Y.C., Liu, S., Kwok, J.T.: Dual-balancing for multi-task learning (2023)
*   \[28\] Lin, B., YE, F., Zhang, Y., Tsang, I.: Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning. Transactions on Machine Learning Research (2022)
*   \[29\] Lin, B., Zhang, Y.: LibMTL: A Python Library for Multi-Task Learning. ArXiv preprint abs/2203.14338 (2022)
*   \[30\] Liu, B., Feng, Y., Stone, P., Liu, Q.: Famo: Fast adaptive multitask optimization. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural Information Processing Systems. vol. 36, pp. 57226–57243. Curran Associates, Inc. (2023)
*   \[31\] Liu, B., Liu, X., Jin, X., Stone, P., Liu, Q.: Conflict-averse gradient descent for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 18878–18890 (2021)
*   \[32\] Liu, L., Li, Y., Kuang, Z., Xue, J., Chen, Y., Yang, W., Liao, Q., Zhang, W.: Towards impartial multi-task learning. In: Proc. of ICLR. OpenReview.net (2021)
*   \[33\] Liu, S., James, S., Davison, A.J., Johns, E.: Auto-Lambda: Disentangling Dynamic Task Relationships. Transactions on Machine Learning Research (2022)
*   \[34\] Liu, S., Johns, E., Davison, A.J.: End-to-end multi-task learning with attention. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1871–1880. Computer Vision Foundation / IEEE (2019). https://doi.org/10.1109/CVPR.2019.00197
*   \[35\] Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015. pp. 3730–3738. IEEE Computer Society (2015). https://doi.org/10.1109/ICCV.2015.425
*   \[36\] Maninis, K., Radosavovic, I., Kokkinos, I.: Attentive single-tasking of multiple tasks. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1851–1860. Computer Vision Foundation / IEEE (2019). https://doi.org/10.1109/CVPR.2019.00195
*   \[37\] Mao, C., Gupta, A., Nitin, V., Ray, B., Song, S., Yang, J., Vondrick, C.: Multitask learning strengthens adversarial robustness. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 158–174. Springer (2020)
*   \[38\] Misra, I., Shrivastava, A., Gupta, A., Hebert, M.: Cross-stitch networks for multi-task learning. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3994–4003. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.433
*   \[39\] Nathan Silberman, Derek Hoiem, P.K., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: ECCV (2012)
*   \[40\] Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., Fetaya, E.: Multi-task learning as a bargaining game. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesvári, C., Niu, G., Sabato, S. (eds.) International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA. Proceedings of Machine Learning Research, vol. 162, pp. 16428–16446. PMLR (2022)
*   \[41\] Pascal, L., Michiardi, P., Bost, X., Huet, B., Zuluaga, M.A.: Improved optimization strategies for deep multi-task networks. ArXiv preprint abs/2109.11678 (2021)
*   \[42\] Royer, A., Blankevoort, T., Bejnordi, B.E.: Scalarization for multi-task and multi-domain learning at scale. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)
*   \[43\] Ruder, S.: An overview of multi-task learning in deep neural networks. ArXiv preprint abs/1706.05098 (2017)
*   \[44\] Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. In: Bengio, S., Wallach, H.M., Larochelle, H., Grauman, K., Cesa-Bianchi, N., Garnett, R. (eds.) Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada. pp. 525–536 (2018)
*   \[45\] Senushkin, D., Patakin, N., Kuznetsov, A., Konushin, A.: Independent component alignment for multi-task learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE (2023)
*   \[46\] Shi, G., Li, Q., Zhang, W., Chen, J., Wu, X.M.: Recon: Reducing Conflicting Gradients From the Root For Multi-Task Learning. In: The Eleventh International Conference on Learning Representations (2023)
*   \[47\] Standley, T., Zamir, A.R., Chen, D., Guibas, L.J., Malik, J., Savarese, S.: Which tasks should be learned together in multi-task learning? In: Proc. of ICML. Proceedings of Machine Learning Research, vol. 119, pp. 9120–9132. PMLR (2020)
*   \[48\] Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., Van Gool, L.: Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021). https://doi.org/10.1109/TPAMI.2021.3054719
*   \[49\] Vasu, P.K.A., Saxena, S., Tuzel, O.: Instance-level task parameters: A robust multi-task weighting framework. CoRR abs/2106.06129 (2021)
*   \[50\] Xin, D., Ghorbani, B., Garg, A., Firat, O., Gilmer, J.: Do Current Multi-Task Optimization Methods in Deep Learning Even Help? In: Neural Information Processing Systems (2022)
*   \[51\] Xu, D., Ouyang, W., Wang, X., Sebe, N.: Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 675–684. IEEE Computer Society (2018). https://doi.org/10.1109/CVPR.2018.00077
*   \[52\] Yang, E., Pan, J., Wang, X., Yu, H., Shen, L., Chen, X., Xiao, L., Jiang, J., Guo, G.: Adatask: A task-aware adaptive learning rate approach to multi-task learning. Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 10745–10753 (2023)
*   \[53\] Ye, F., Lin, B., Yue, Z., Guo, P., Xiao, Q., Zhang, Y.: Multi-objective meta learning. In: Advances in Neural Information Processing Systems. vol. 34 (2021)
*   \[54\] Yeo, T., Kar, O.F., Zamir, A.: Robustness via cross-domain ensembles. In: 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. pp. 12169–12179. IEEE (2021). https://doi.org/10.1109/ICCV48922.2021.01197
*   \[55\] Yu, F., Koltun, V., Funkhouser, T.A.: Dilated residual networks. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017. pp. 636–644. IEEE Computer Society (2017). https://doi.org/10.1109/CVPR.2017.75
*   \[56\] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)
*   \[57\] Zhang, Y., Yang, Q.: A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering 34(12), 5586–5609 (2022). https://doi.org/10.1109/TKDE.2021.3070203
*   \[58\] Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017. pp. 6230–6239. IEEE Computer Society (2017). https://doi.org/10.1109/CVPR.2017.660

Examining Common Paradigms in Multi-Task Learning  
-Supplementary Material-

## Appendix A1 Theoretical insights into multi-task learning dynamics

In this section, we aim to explain the success of the Adam optimizer \[[22](#bib.bib22 "")\] by relating it to uncertainty weighting \[[21](#bib.bib21 "")\]. We show partial invariances w.r.t. prior task-weights for the Adam optimizer and full invariances for the uncertainty weighting under mild assumptions. We further show that for SGD + momentum no invariance can be observed. Instead, the loss-weight can be seen as a task-specific learning rate which is not the case for the Adam optimizer. Previous literature on weighting methods in MTL did not explicitly show how task-weighting methods are affected by different optimizers.

### A1.1 Uncertainty weighting (UW): Full loss-scale invariance

In UW \[[21](#bib.bib21 "")\], the homoscedastic uncertainty11 1 In Kendall et al., this is termed the aleatoric homoscedastic uncertainty. However, as the task weights vary over the course of training and also with respect to the model capacity, it is technically not only the aleatoric uncertainty but also encapsulates further components such as model capacity and amount of data seen. σt\\sigma\_{t} to weight task tt is learned by gradient descent. However, we can also analytically compute the optimal uncertainty weights in each iteration instead of learning them using gradient descent as done in \[[23](#bib.bib23 "")\]. The minimization objective depends on the underlying loss function and likelihood. For simplicity, we show the derivation exemplary for the L1L\_{1} loss. It is straight-forward to derive the same for a Gaussian and other distributions. The objective of uncertainty weighting is given as

minσt⁡1σt​ℒt+log⁡σt\\min\_{\\sigma\_{t}}\\frac{1}{\\sigma\_{t}}\\mathcal{L}\_{t}+\\log\\sigma\_{t}

(6)

with ℒt\=|y−fW​(x)|\\mathcal{L}\_{t}=|y-f^{W}(x)| which can be derived from a log likelihood of a Laplace distribution p⁡(y|fW​(x),σ)\=12​σ​e​x​p​(−|y−fW​(x)|σ)p(y|f^{W}(x),\\sigma)=\\frac{1}{2\\sigma}exp(-\\frac{|y-f^{W}(x)|}{\\sigma}). Taking the derivative and solving for σt\\sigma\_{t} results in an analytically optimal solution:

∂∂σt​1σt​ℒt+log⁡σt\=−1σt2​ℒt+1σt\\frac{\\partial}{\\partial\\sigma\_{t}}\\frac{1}{\\sigma\_{t}}\\mathcal{L}\_{t}+\\log\\sigma\_{t}=-\\frac{1}{\\sigma\_{t}^{2}}\\mathcal{L}\_{t}+\\frac{1}{\\sigma\_{t}}

(7)

−1σt2​ℒt+1σt​\=!​0⇒σt\=ℒt-\\frac{1}{\\sigma\_{t}^{2}}\\mathcal{L}\_{t}+\\frac{1}{\\sigma\_{t}}\\overset{!}{=}0~~~~\\Rightarrow~~~~\\sigma\_{t}=\\mathcal{L}\_{t}

(8)

with σt\>0\\sigma\_{t}>0. As the optimization problem is convex and just one dimensional, assuming an optimal log-sigma is a mild assumption. Plugging the optimal solution back into the original uncertainty weighting, we get

ℒ\=∑t1s​g​\[ℒt\]​ℒt+log⁡s​g​\[ℒt\],\\mathcal{L}=\\sum\_{t}\\frac{1}{sg\[\\mathcal{L}\_{t}\]}\\mathcal{L}\_{t}+\\log\\sqrt{sg\[\\mathcal{L}\_{t}\]},

(9)

where we denote s​gsg as the stopgradient operator.

Since there is no gradient for the second part of the loss, it can be simplified such that

ℒ\=∑tℒts​g​\[ℒt\].\\mathcal{L}=\\sum\_{t}\\frac{\\mathcal{L}\_{t}}{sg\[\\mathcal{L}\_{t}\]}.

(10)

Assuming task-specific weights αt\\alpha\_{t}, we get

ℒ\=∑tαt​ℒtαt​s​g​\[ℒt\]\=∑tℒts​g​\[ℒt\]\\begin{split}\\mathcal{L}&=\\sum\_{t}\\frac{\\alpha\_{t}\\mathcal{L}\_{t}}{\\alpha\_{t}sg\[\\mathcal{L}\_{t}\]}\\\\ &=\\sum\_{t}\\frac{\\mathcal{L}\_{t}}{sg\[\\mathcal{L}\_{t}\]}\\end{split}

(11)

Therefore, the optimal uncertainty weighting is invariant w.r.t. task-specific loss-scalings, as each scaling cancels out.

### A1.2 SGD: No loss-scale invariance and relationship of learning rate and task weights on a gradient level

Unlike for optimal UW, we show that the SGD update rule does not show any invariances and that task-weights are essentially task-specific learning rates. Instead, task-weights and learning rate are interacting hyperparameters and thus cannot be viewed in isolation.

The parameter update rule in neural networks optimized with SGD is

θi\=θi−1−γ​∂∂θi−1​ℒ,\\theta\_{i}=\\theta\_{i-1}-\\gamma\\frac{\\partial}{\\partial\\theta\_{i-1}}\\mathcal{L},

(12)

where the network parameters in iteration ii are defined as θi\\theta\_{i}, γ\\gamma is the learning rate and ℒ\=∑tαi​ℒt\\mathcal{L}=\\sum\_{t}\\alpha\_{i}\\mathcal{L}\_{t} .

In the case of uniform task weights (EW), α\=αi​∀i\\alpha=\\alpha\_{i}\\forall i, we have

θi\=θi−1−γ​∂∂θi−1​∑iα​ℒt\=θi−1−γ​α​∂∂θi−1​∑iℒt\\begin{split}\\theta\_{i}&=\\theta\_{i-1}-\\gamma\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{i}\\alpha\\mathcal{L}\_{t}\\\\ &=\\theta\_{i-1}-\\gamma\\alpha\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{i}\\mathcal{L}\_{t}\\\\ \\end{split}

(13)

Here, task weight and learning rate are interchangeable. In particular, increasing the weight α\\alpha by a constant factor cc has the same effect as increasing the learning rate by a factor cc.

In the case of non-uniform task weights αi\\alpha\_{i}, the parameter update is

θi\=θi−1−γ​∂∂θi−1​∑iαt​ℒt\=θi−1−∂∂θi−1​∑iγ​αt​ℒt\\begin{split}\\theta\_{i}&=\\theta\_{i-1}-\\gamma\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{i}\\alpha\_{t}\\mathcal{L}\_{t}\\\\ &=\\theta\_{i-1}-\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{i}\\gamma\\alpha\_{t}\\mathcal{L}\_{t}\\\\ \\end{split}

(14)

As the learning rate can be included in the task-specific weight, it follows that task weighting is interchangeable to assigning task-specific learning rates. Tasks with a higher weight αi\\alpha\_{i} have a proportionally higher parameter update step and vice versa.

While this holds for SGD and SGD + momentum, it does not apply to optimizers such as Adam, Adagrad, or RMSProp. We demonstrate this for Adam in the following subsection.

### A1.3 Adam: Partial loss-scale invariance

Similarly to the invariance demonstrated for optimal UW, we derive a partial invariance for Adam. In their work, Kingma and Ba \[[22](#bib.bib22 "")\] have already shown that the magnitudes of the parameter updates using Adam are invariant to rescaling the gradients. Our novelty lies in demonstrating this invariance property in the context of MTL and its impact on different MTO methods. For Adam, we claim that the magnitude of task-specific weights only affects the backbone and cancels out for the heads.

We consider the standard MTL model setting with a shared backbone and task-specific heads. In this analysis, we assume a frozen backbone and only look at the task-specific parameters ψt\\psi\_{t} of task tt whose loss ℒt\\mathcal{L}\_{t} is scaled by αt\\alpha\_{t}, such that ℒt→αt​ℒt\\mathcal{L}\_{t}\\rightarrow\\alpha\_{t}\\mathcal{L}\_{t}. The parameter update of one head is independent of the other heads as the derivative of the losses w.r.t. the other tasks is 00:

∂∂ψt,i−1​ℒj\=0​for​t≠j.\\frac{\\partial}{\\partial\\psi\_{t,i-1}}\\mathcal{L}\_{j}=0\\;\\;\\mathrm{for\\>t\\neq j}.

(15)

The general update rule for parameters ψ\\psi at time step ii using Adam is

ψi\=ψi−1−γvi^+ϵ​mi^,\\psi\_{i}=\\psi\_{i-1}-\\frac{\\gamma}{\\sqrt{\\hat{v\_{i}}}+\\epsilon}\\hat{m\_{i}},

(16)

where mi\=β1​mi−1+(1−β1)​gim\_{i}=\\beta\_{1}m\_{i-1}+(1-\\beta\_{1})g\_{i} and vi\=β2​vi−1+(1−β2)​gi2v\_{i}=\\beta\_{2}v\_{i-1}+(1-\\beta\_{2})g\_{i}^{2}. To counteract the bias towards 0, the moments are corrected as mi^\=mi1−β1i\\hat{m\_{i}}=\\frac{m\_{i}}{1-\\beta\_{1}^{i}} and vi^\=vi1−β2i\\hat{v\_{i}}=\\frac{v\_{i}}{1-\\beta\_{2}^{i}}.

For task-specific parameters ψt\\psi\_{t}, task weights αt\\alpha\_{t} linearly scale the first moment mt,im\_{t,i}

mt,i\=β1​mt,i−1+(1−β1)​gt,i\=β1​mt,i−1+(1−β1)​∂∂ψt,i−1​αt​ℒt\=β1​mt,i−1+(1−β1)​αt​∂∂ψt,i−1​ℒt\=β1​mt,i−1+(1−β1)​αt​gt,i′\\begin{split}m\_{t,i}&=\\beta\_{1}m\_{t,i-1}+(1-\\beta\_{1})g\_{t,i}\\\\ &=\\beta\_{1}m\_{t,i-1}+(1-\\beta\_{1})\\frac{\\partial}{\\partial\\psi\_{t,i-1}}\\alpha\_{t}\\mathcal{L}\_{t}\\\\ &=\\beta\_{1}m\_{t,i-1}+(1-\\beta\_{1})\\alpha\_{t}\\frac{\\partial}{\\partial\\psi\_{t,i-1}}\\mathcal{L}\_{t}\\\\ &=\\beta\_{1}m\_{t,i-1}+(1-\\beta\_{1})\\alpha\_{t}g^{\\prime}\_{t,i}\\\\ \\end{split}

(17)

and quadratically scale the second moment vt,iv\_{t,i}

vt,i\=β2​vt,i−1+(1−β2)​gt,i2\=β2​vt,i−1+(1−β2)​(∂∂ψt,i−1​αt​ℒt)2\=β2​vt,i−1+(1−β2)​αt2​(∂∂ψt,i−1​ℒt)2\=β2​vt,i−1+(1−β2)​αt2​g′t,i2,\\begin{split}v\_{t,i}&=\\beta\_{2}v\_{t,i-1}+(1-\\beta\_{2})g\_{{t,i}}^{2}\\\\ &=\\beta\_{2}v\_{t,i-1}+(1-\\beta\_{2})(\\frac{\\partial}{\\partial\\psi\_{t,i-1}}\\alpha\_{t}\\mathcal{L}\_{t})^{2}\\\\ &=\\beta\_{2}v\_{t,i-1}+(1-\\beta\_{2})\\alpha\_{t}^{2}(\\frac{\\partial}{\\partial\\psi\_{t,i-1}}\\mathcal{L}\_{t})^{2}\\\\ &=\\beta\_{2}v\_{t,i-1}+(1-\\beta\_{2})\\alpha\_{t}^{2}{g^{\\prime}}\_{t,i}^{2},\\\\ \\end{split}

(18)

where g′t,i{g^{\\prime}}\_{t,i} is the gradient of the unscaled loss ℒt\\mathcal{L}\_{t} w.r.t. the task-specific parameters for task tt. As this holds for iteration ii and because we have mt,1\=α​g′t,1+0m\_{t,1}=\\alpha{g^{\\prime}}\_{t,1}+0 respectively vt,1\=αt2​g′t,12+0v\_{t,1}=\\alpha\_{t}^{2}{g^{\\prime}}^{2}\_{t,1}+0 with mt,0\=0m\_{t,0}=0, vt,0\=0v\_{t,0}=0 at the first iteration, this holds for any iteration step. We can thus rewrite m^t,i\=αt​m^t,i′\\hat{m}\_{t,i}=\\alpha\_{t}\\hat{m}^{\\prime}\_{t,i} and v^t,i\=αt2​v′^t,i\\hat{v}\_{t,i}=\\alpha\_{t}^{2}\\hat{v^{\\prime}}\_{t,i}.

Plugging this back into the update rule, we get

ψt,i\=ψt,i−1−γv^t​i​m^t,i\=ψt,i−1−γαt2​v′^t,i​αt​m′^t,i\\begin{split}\\psi\_{t,i}&=\\psi\_{t,i-1}-\\frac{\\gamma}{\\sqrt{\\hat{v}\_{t}i}}\\hat{m}\_{t,i}\\\\ &=\\psi\_{t,i-1}-\\frac{\\gamma}{\\sqrt{\\cancel{\\alpha\_{t}^{2}}\\hat{v^{\\prime}}\_{t,i}}}\\cancel{\\alpha\_{t}}\\hat{m^{\\prime}}\_{t,i}\\end{split}

(19)

where the loss-scaling αt\\alpha\_{t} cancels out. Therefore, the parameters of the task-specific heads are invariant to loss-scalings using Adam.

This partial invariance is a highly desired property as there is a fundamental trade-off between tuning the learning rate and manual task weights. Given Adams invariance for the head, the weighting only affects the backbone. Thus the learning rate can be set for the parameters of the head independent of the loss weights. With the loss weights, we can prioritize tasks in the backbone and therefore walk along the Pareto front as empirically shown by \[[50](#bib.bib50 "")\].

The invariance, however, does not hold anymore when the backbone parameters θ\\theta are updated as well. As we have

mi\=β1​mi−1+(1−β1)​∂∂θi−1​∑tαt​ℒt\=β1​mi−1+(1−β1)​∑tαt​gt,i′\\begin{split}m\_{i}&=\\beta\_{1}m\_{i-1}+(1-\\beta\_{1})\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{t}\\alpha\_{t}\\mathcal{L}\_{t}\\\\ &=\\beta\_{1}m\_{i-1}+(1-\\beta\_{1})\\sum\_{t}\\alpha\_{t}g^{\\prime}\_{t,i}\\end{split}

(20)

and

vi\=β1​vi−1+(1−β1)​(∂∂θi−1​∑tαt​ℒt)2\=β1​vi−1+(1−β1)​(∑tαt​gt,i′)2\\begin{split}v\_{i}&=\\beta\_{1}v\_{i-1}+(1-\\beta\_{1})(\\frac{\\partial}{\\partial\\theta\_{i-1}}\\sum\_{t}\\alpha\_{t}\\mathcal{L}\_{t})^{2}\\\\ &=\\beta\_{1}v\_{i-1}+(1-\\beta\_{1})(\\sum\_{t}\\alpha\_{t}g^{\\prime}\_{t,i})^{2}\\end{split}

(21)

we conclude that the task weights ata\_{t} linearly affect the first moment mim\_{i}, while having a quadratic effect on the update of the second moment viv\_{i}.

Note that for both task-heads only as well as the backbone, we have a full invariance in case of independent optimizers, e.g., one Adam optimizer per task similar to \[[41](#bib.bib41 ""), [52](#bib.bib52 "")\]. However, naive implementations scale poorly (in terms of computational complexity) with the number of tasks here.  

In the following experiments, we provide empirical evidence for our finding that a) Adam offers loss-scale invariance for the parameters of the task-specific heads, and b) Adam offers loss-scale invariance for all network parameters (backbone and heads) if β1,2\=0\\beta\_{1,2}=0.

## Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting

In the prior section, we derived theoretical results for loss-scale (partial) invariance within multi-task learning for the Adam optimizer and uncertainty weighting. In this section, we confirm this invariance empirically with a toy task.

##### Experimental Setup

We consider a two-task toy experiment in which we look at the gradient magnitudes with different combinations of Adam, SGD, EW, optimal uncertainty weighting (UW-O), and loss-scalings. To generate the data, we sample scalar input values from a uniform distribution; the outputs are just scalings of the input. We apply a simple neural network which consists of a shared backbone (two layers with LeakyReLU as non-linearity and 20 neurons per hidden layer) and two heads for the two tasks, each consisting again of two layers. Both task measure the depth but in different units using the L1L\_{1}-loss.

We provide two settings: In the first one, depth is measured on the same scale. In the second setting, one depth loss is scaled by 10x (e.g., measured in cm instead of deci-meters) and one other loss is scaled by 0.1 (e.g., measured in meters instead of deci-meters). For each setting, we test various combinations of loss weighting and optimizer combinations.

The 8 different experiments are:

*   •

```
EW using SGD
```
*   •

```
EW using SGD with scalings 10⋅Ls​e​g10\\cdot L\_{seg}, 0.01⋅Ld​e​p0.01\\cdot L\_{dep}
```
*   •

```
EW using Adam
```
*   •

```
EW using Adam with scalings 10⋅Ls​e​g10\\cdot L\_{seg} and 0.01⋅Ld​e​p0.01\\cdot L\_{dep}
```
*   •

```
UW-O) using SGD
```
*   •

```
UW-O) using SGD with scalings 10⋅Ls​e​g10\\cdot L\_{seg} and 0.01⋅Ld​e​p0.01\\cdot L\_{dep}
```
*   •

```
EW using separate Adam optimizers per task
```
*   •

```
EW using separate Adam optimizers per task with scalings 10⋅Ls​e​g,0.01⋅Ld​e​p10\\cdot L\_{seg},0.01\\cdot L\_{dep}
```
To better control for different factors of influence, we first perform the first 6 of the listed experiments with a fixed backbone, i.e., we do not update the parameters in the backbone but only in the heads. Afterward, we show all 8 experiments trained with a network where all parameters (including the backbone) are updated. This allows us to verify if our theoretical derivations regarding the (partial) loss-scaling invariance of Adam and UW-O also hold in practice, and compare this to the SGD optimizer.

Note that we only care about the invariance and did not tune any hyperparameters for performance.

##### Results for fixed backbone

Figure [A1](#Pt0.A2.F1 "Figure A1 ‣ Results for fixed backbone ‣ Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting ‣ Examining Common Paradigms in Multi-Task Learning") shows the losses, the scaled losses (by loss weighting method), the gradient magnitudes as well as the gradient update magnitudes for both heads along the 100 epochs of training with a fixed backbone. Regarding SGD, we can observe that the equal weighting experiment differs from its scaled variant along all 8 dimensions. This is because SGD does not offer any loss-scaling invariance. As expected, at the beginning of the training the gradient update magnitude of the first depth head parameters with the scaled loss (dotted line) is by a factor of 1010 higher than the unscaled (solid line) one. The same effect applies to the gradient update magnitude of the second depth head parameters, but with a factor of 0.010.01.

In contrast, Adam is loss-scale invariant. We can observe that the unscaled (solid line) and the scaled version ( dotted line) have equal gradient update magnitudes in the last row. Note that practically due to an e​p​s​i​l​o​n\=10−8epsilon=10^{-8} parameter in the denominator and float precision a slight divergence would occur with larger number of epochs. This result confirms our theoretical finding in equation [19](#Pt0.A1.E19 "Equation 19 ‣ A1.3 Adam: Partial loss-scale invariance ‣ Appendix A1 Theoretical insights into multi-task learning dynamics ‣ Examining Common Paradigms in Multi-Task Learning"). We skip the experiment of separated Adam optimizers per task because it would be equivalent to this version given a fixed backbone.

Lastly, we want to investigate the invariance properties of UW-O. We compare the scaled (dotted line) and unscaled (solid line) version of UW-O with the SGD optimizer. As expected, the gradients, as well as the gradient updates, match in both heads.

In the following, let’s investigate whether the observed results still hold if we also consider the update of the backbone parameters.

Figure A1: Invariances within the neural network for a frozen backbone. Comparing the effect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the the two task heads and keep the backbone frozen. While SGD does not offer any loss-scaling invariance, Adam makes the gradient updates of the head parameters invariant to scales confirming our derivation (red lines overlap in lowest row). Equivalently, for UW-O we also observe the theoretically derived invariances (green lines overlap in lowest row)

##### Results for free backbone

Figure [A2](#Pt0.A2.F2 "Figure A2 ‣ Results for free backbone ‣ Appendix A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting ‣ Examining Common Paradigms in Multi-Task Learning") shows the scaled losses, the gradient magnitudes as well as the gradient update magnitudes in the backbone and the depth heads along the 100 epochs of training with a free backbone. Again, the loss-scalings affect the gradient magnitudes using SGD. This applies to both backbone and heads.

When looking at the Adam experiments, we can observe that it is partly loss-scale invariant by looking at the first iteration in the heads. However, due to different updates in the backbone, the networks behave different in both settings (scaled and unscaled loses). Furthermore, when implementing task-specific optimizers, we can observe that not only the gradient update magnitudes in the task heads, but also in the backbone match between the scaled (dotted line) and the unscaled (solid line) variant. Thus, all network parameters are invariant to loss-scalings when using separate Adam optimizers. This confirms our theoretical results.

Along the lines of our theoretical findings, we can observe that UW-O offers scaling-invariance across the whole network as the gradients as well as the gradient updates match among the two variants in the backbone and in both heads. This empirical observation matches our theoretical derivation in equation [11](#Pt0.A1.E11 "Equation 11 ‣ A1.1 Uncertainty weighting (UW): Full loss-scale invariance ‣ Appendix A1 Theoretical insights into multi-task learning dynamics ‣ Examining Common Paradigms in Multi-Task Learning").

Figure A2: Invariances within the neural network for a learnable backbone. Comparing the effect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. Additionally, we implement independent Adam optimizer per task. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the backbone(first row) and the the two task heads (2nd and 3rd row). Neither Adam, nor SGD show invariances if the backbone is trained as well. UW-O is still invriant (green lines are overlapping). We revoke Adam’s inveriance by implementing separate optimizers per task (lowerst black lines are overlapping).

## Appendix A3 Implementation Details

In this section, we explain the applied settings used for the reported experiments in more detail. In particular, we describe the handling of the different datasets in [Sec. A3.1](#Pt0.A3.SS1 "A3.1 Datasets ‣ Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning") and provide further information on the applied training procedures in [Sec. A3.2](#Pt0.A3.SS2 "A3.2 Training ‣ Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning"). Our chosen experimental setups are designed to follow previous work and mainly inspirited by \[[31](#bib.bib31 ""), [50](#bib.bib50 ""), [29](#bib.bib29 "")\]. However, we found that the experimental setup would vary widely across different works in the field of multi-task learning as can be seen in [Table A1](#Pt0.A3.T1 "In CelebA ‣ A3.1 Datasets ‣ Appendix A3 Implementation Details ‣ Examining Common Paradigms in Multi-Task Learning"). We use a uniform setup for each dataset independent of the choice of network and MTO.

### A3.1 Datasets

##### Cityscapes

\[[9](#bib.bib9 "")\] We make use of the official split of the dataset which consists of 2975 training and 500 validation scenes. Similar to  \[[50](#bib.bib50 "")\], we denote 595 random samples from the training split as validation data and report test results on the original validation split. We further follow the pre-processing scheme from  \[[34](#bib.bib34 "")\] of re-scaling images to 128x256 pixels and use inverse depth labels. During training, we apply random scaling and cropping for data augmentation22 2 https://github.com/Cranial-XIX/CAGrad. Following previous work \[[31](#bib.bib31 "")\] for number of epochs and learning rate schedule, we train for 300 epochs and decrease the learning rate by a factor of 0.5 every 100 epochs. The batch size is setto 64, similar to \[[50](#bib.bib50 "")\]. We only consider a fixed weight decay of 10−510^{-5} for all datasets and experiments as we found varying this parameter had only little influence in initial experiments.

##### NYUv2

\[[39](#bib.bib39 "")\] From the 795 official training images we use 159 for our validation split as in \[[29](#bib.bib29 "")\] and report test performance on the official 654 test images. Similar to \[[34](#bib.bib34 "")\], we re-size the images to 288x384 pixels. Training is run for 200 epochs with a batch size of 8. We apply the same data augmentation and learning rate schedule as for Cityscapes.

##### CelebA

\[[35](#bib.bib35 "")\] We re-size images to 64x64 pixels as done in \[[28](#bib.bib28 "")\] and consider the original split of 162,770/19,867/19,962 for training, validation, and testing. We set the batch size to 512, train for 100 epochs, and halve the learning rates every 30 epochs.

Table A1: Original experiment setup as reported in respective papers. We note a high variation regarding the choice of network, optimizer, and other hyper-parameters among the different works. 

| Data                                  |                                                                            |                              |                                                                                         |                       |     |            |
| ------------------------------------- | -------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------- | --------------------- | --- | ---------- |
| UW    \[[21](#bib.bib21 "")\]         | DeepLabV3 \[[5](#bib.bib5 "")\]   with ResNet101 \[[55](#bib.bib55 "")\]   | SGD + Nesterov updates, Mom. | init.: 2.5⋅10−32.5\\cdot 10^{-3};   polynomial lr decay                                 | 1⋅10−41\\cdot 10^{-4} | 8   | 100k iter. |
| RLW    \[[28](#bib.bib28 "")\]        | DeepLabV3 \[[5](#bib.bib5 "")\]   with ResNet50 \[[55](#bib.bib55 "")\]    | Adam                         | 1⋅10−41\\cdot 10^{-4}                                                                   | 1⋅10−51\\cdot 10^{-5} | 64  |            |
| IMTL    \[[32](#bib.bib32 "")\]       | ResNet50 \[[55](#bib.bib55 "")\]   \+ PSPNet \[[58](#bib.bib58 "")\] heads | SGD+Mom.                     | init.: 0.020.02;   polynomial lr decay                                                  | 1⋅10−41\\cdot 10^{-4} | 32  | 200 epochs |
| PCGrad    \[[56](#bib.bib56 "")\]     | MTAN \[[34](#bib.bib34 "")\]                                               | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr after 40k iter.                                 | -                     | 8   | 80k iter   |
| CAGrad    \[[31](#bib.bib31 "")\]     | MTAN \[[34](#bib.bib34 "")\]                                               | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr every 100 epochs                                | -                     | 8   | 200 epochs |
| AlignedMTL  \[[45](#bib.bib45 "")\]   | MTAN \[[34](#bib.bib34 "")\] /   PSPNet \[[58](#bib.bib58 "")\]            | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr every 100 epochs                                | -                     | 8   | 200 epochs |
| RLW    \[[28](#bib.bib28 "")\]        | DeepLabV3 \[[5](#bib.bib5 "")\]   with ResNet50 \[[55](#bib.bib55 "")\]    | Adam                         | 1⋅10−41\\cdot 10^{-4}                                                                   | 1⋅10−51\\cdot 10^{-5} | 8   |            |
| IMTL    \[[32](#bib.bib32 "")\]       | ResNet50 \[[55](#bib.bib55 "")\]   \+ PSPNet \[[58](#bib.bib58 "")\] heads | SGD+Mom.                     | init.: 0.030.03                                                                         | -                     | 48  | 200 epochs |
| PCGrad    \[[56](#bib.bib56 "")\]     | MTAN \[[34](#bib.bib34 "")\]                                               | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr after 40k iter.                                 | -                     | 2   | 80k iter   |
| CAGrad    \[[31](#bib.bib31 "")\]     | MTAN \[[34](#bib.bib34 "")\]                                               | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr after 100 epochs                                | -                     | 2   | 200 epochs |
| AlignedMTL    \[[45](#bib.bib45 "")\] | MTAN \[[34](#bib.bib34 "")\] /   PSPNet \[[58](#bib.bib58 "")\]            | Adam                         | init.: 1⋅10−41\\cdot 10{-4};   halve lr after 100 epochs                                | -                     | 2   | 200 epochs |
| RLW    \[[28](#bib.bib28 "")\]        | ResNet17 \[[15](#bib.bib15 "")\]   \+ lin. classifier                      | Adam                         | 1⋅10−31\\cdot 10^{-3}                                                                   | -                     | 512 |            |
| IMTL    \[[32](#bib.bib32 "")\]       | ResNet17 \[[15](#bib.bib15 "")\]   \+ lin. classifier                      | Adam                         | 0.0030.003                                                                              | -                     | 256 | 100 epochs |
| PCGrad    \[[56](#bib.bib56 "")\]     | ResNet17 \[[15](#bib.bib15 "")\]   \+ lin. classifier                      | Adam                         | init. from {10−4,…,5⋅10−2}\\{10^{-4},...,5\\cdot 10^{-2}\\};   halve lr every 30 epochs | -                     | 256 | 100 epochs |

### A3.2 Training

##### Effectiveness of Adam in MTL.

All presented results are based on performing early stopping w.r.t. Δm\\Delta\_{m} metric on the validation set. For this, we further trained single-task learning (STL) models for each experiment combination (dataset and network) using the respective network architecture except for the missing head(s). We trained the models using Adam and any learning rate from {0.01,0.005,…,0.00005}\\{0.01,0.005,...,0.00005\\}. The training was stopped early based on the validation loss. Reported scores in [Tabs. A6](#Pt0.A4.T6 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A7](#Pt0.A4.T7 "Table A7 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") and [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") are computed as the mean of the models’ performance that were initialized with the three different seeds.

Our implementation for all experiments is based on the LibMTL library \[[29](#bib.bib29 "")\].

##### Gradient Similarity.

Our gradient similarity experiments were conducted on the best performing hyper-parameter configuration for EW from the previous extensive evaluation. Over the full training, gradient similarity measures are computed every five iteration steps and summarized per epoch. To make the computation effort more feasible in case of settings with large batch size or high number of tasks, we randomly select eight samples or tasks respectively and consider corresponding gradients in these cases.

### A3.3 Loss functions

##### Cityscapes

For the task of semantic segmentation, we employ a pixel-wise cross-entropy loss:

ℒC​E\=∑c\=1Cyc⋅log⁡(pc)\\mathcal{L}\_{CE}=\\sum\_{c=1}^{C}y^{c}\\cdot\\log(p^{c})

(22)

where CC is the number of classes, yc∈{0,1}y^{c}\\in\\{0,1\\} indicates the ground truth class, and pcp^{c} is the predicted probability for class cc which results from computing the softmax for output logits zcz^{c}, pc\=exp⁡(zc)∑c\=1Cexp⁡(zc)p^{c}=\\frac{\\exp(z^{c})}{\\sum\_{c=1}^{C}\\exp(z^{c})}. This loss is averaged over the image.

For depth estimation, we utilize the L1L\_{1} loss:

ℒd​e​p​t​h\=‖𝒚−𝒚^‖1\\mathcal{L}\_{depth}=\\|\\bm{y}-\\hat{\\bm{y}}\\|\_{1}

(23)

where 𝒚,𝒚^\\bm{y},\\hat{\\bm{y}} indicate ground truth and prediction, respectively. Pixels with invalid depth value in the ground truth data are ignored. It is noteworthy that these two types of losses are not balanced when used directly without modification.

##### NYUv2

The tasks of semantic segmentation and depth estimation are trained using the same loss functions as described for Cityscapes. In addition, we compute the cosine loss on the (normalized) surface normal maps:

ℒn​o​r​m​a​l\=1−cos⁡θ\=1−𝒚⋅𝒚^‖𝒚‖​‖𝒚^‖\\mathcal{L}\_{normal}=1-\\cos{\\theta}=1-\\frac{\\bm{y}\\cdot\\hat{\\bm{y}}}{\\|\\bm{y}\\|\\|\\hat{\\bm{y}}\\|}

(24)

where 𝒚,𝒚^\\bm{y},\\hat{\\bm{y}} are the ground truth and predicted normal maps. Similar to the Cityscapes setup, the combination of these loss functions is not balanced per default.

##### CelebA

To learn to predict multiple attributes, we use a binary cross entropy loss for the individual classes:

ℒC​E,b​i​n\=−\[y​log⁡(p)+(1−y)​log⁡(1−p)\]\\mathcal{L}\_{CE,bin}=-\[y\\log(p)+(1-y)\\log(1-p)\]

(25)

Although all these losses have a similar scale, their impact varies based on the difficulty of the individual tasks and the number of available samples displaying the respective attribute.

### A3.4 Evaluation criteria

In this study, we primarily focus on Pareto optimal solution to acknowledge that different configurations may lead to varying preferences for the learned tasks. However, it is important to note that not every point on the Pareto front is relevant in practice, especially when one metric significantly dominates while others are close to chance level. Moreover, specific real-world applications can have a stronger, pre-defined prioritization of one or a few sub-tasks which requires a relative weighting of the tasks’ performances.

Additionally, we further employ the Δm\\Delta\_{m} metric which offers a simple option to directly compare the performance of two models using a single scalar. This metric further indicates the relative performance compared to the single-tasks models.

Note that for our initial toy task experiment ([Sec. 4.1](#S4.SS1 "4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")), we consider the original setting from \[[31](#bib.bib31 "")\] which optimzes the global minimum of the two loss functions.

## Appendix A4 Additional results on comparison between Adam and SGD

We present additional evaluation results for our comparison between optimizers for MTL. In [Figure A3](#Pt0.A4.F3 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), we compare the Δm\\Delta\_{m} metric performance between the usage of Adam and SGD+mom. [Fig. A4](#Pt0.A4.F4 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") shows additional parallel coordinate plots for NYUv2 and both choices of networks. In [Table A2](#Pt0.A4.T2 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), we count for each used MTO method the number of experiment runs that are located on the Pareto front w.r.t. each setup. Best performing quantitative results for all MTOs can be found in [Tabs. A5](#Pt0.A4.T5 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A5](#Pt0.A4.T5 "Table A5 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning"), [A6](#Pt0.A4.T6 "Table A6 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning") and [A7](#Pt0.A4.T7 "Table A7 ‣ Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning").

We further show extended results on the toy task by Liu et al. \[[31](#bib.bib31 "")\] for more learning rates in [Tab. A3](#Pt0.A4.T3 "In Appendix A4 Additional results on comparison between Adam and SGD ‣ Examining Common Paradigms in Multi-Task Learning").

Figure A3: Mean Δm\\Delta\_{m} metric for experiments run on Cityscapes and NYUv2 with SegNet and DeepLabV3. We compare the performance of the best hyperparameter setting for every MTO method using either Adam (left) or SGD+Momentum (right) (lower is better). Every MTO is associated with a different line color/style. On Cityscapes, there is a large difference for the Δm\\Delta\_{m} score for Adam compared to SGD+Momentum, especially for UW, IMTL, and CAGrad. Therefore, for this setup, the result depends more on the optimizer than on the MTO method. On the NYUv2 dataset this observation weakens. Adam still achieves the lowest Δm\\Delta\_{m} scores across different MTO methods (except for SegNet with UW and IMTL), though, besides chosing Adam, it is also important to select the appropriate MTO method. 

![Refer to caption](2311.04698v5/adamVSsgd_parallelcoordplot_DatasetNYUv2_AugmentationTrue_ArchitectureSegNet_weightdecay1e-05_BS8__new.png)

(a) SegNet

![Refer to caption](2311.04698v5/adamVSsgd_parallelcoordplot_DatasetNYUv2_AugmentationTrue_ArchitectureDeepLabV3_resnet50_weightdecay1e-05_BS8__new.png)

(b) DeepLabV3

Figure A4: Parallel coordinate plot over all experiments on NYUv2 We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation.  
Similiar to results on Cityscapes in the main paper ([Fig. 2](#S4.F2 "In Experiments on Cityscapes and NYUv2 ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning")) we observe a dominance of Adam albeit, here, we also have some experiments using SGD+Mom. on the overall Pareto front. 

Table A2: Count of Pareto optimal experiments for each MTO method. We found no single MTO method to be clearly superior over all combinations of dataset and networks. Total numbers can be compared to [Table 2](#S4.T2 "In Experiments on Cityscapes and NYUv2 ‣ 4.1 Effectiveness of Adam in multi-task learning ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning") 

| Data       | Network   | Optimizer |  | EW | UW | RLW | IMTL | PCGrad | CAGrad | AlignedMTL | AdaTask |  | Total |
| ---------- | --------- | --------- |  | -- | -- | --- | ---- | ------ | ------ | ---------- | ------- |  | ----- |
| Cityscapes | SegNet    | Adam      |  | -  | 3  | -   | -    | -      | 1      | 1          | -       |  | 5     |
| Cityscapes | SegNet    | SGD+Mom.  |  | -  | -  | -   | -    | -      | -      | -          | -       |  | -     |
| Cityscapes | DeepLabV3 | Adam      |  | 2  | 2  | 1   | 2    | 2      | 1      | -          | -       |  | 15    |
| Cityscapes | DeepLabV3 | SGD+Mom.  |  | -  | -  | -   | -    | -      | -      | -          | -       |  | -     |
| NYUv2      | SegNet    | Adam      |  | 1  | 2  | -   | 1    | 1      | 1      | 3          | 2       |  | 11    |
| NYUv2      | SegNet    | SGD+Mom.  |  | -  | -  | -   | 1    | -      | -      | -          | -       |  | 1     |
| NYUv2      | DeepLabV3 | Adam      |  | 2  | -  | 1   | 3    | 3      | 2      | 2          | 3       |  | 16    |
| NYUv2      | DeepLabV3 | SGD+Mom.  |  | -  | 1  | -   | -    | -      | 3      | 2          | -       |  | 6     |

Table A3: Number of iterations after which all seeds in toy task experiment from CAGrad \[[31](#bib.bib31 "")\] have reached the global minimum for different learning rates and optimizer. We show results for additional learning rates compared to the main paper. The maximum iteration number over all three seeds for each MTO method / learning rate / optimizer combination is reported. If not all seeds converged to the global minimum within 100k iteration steps, we denote it as ’-’. In several setups, EW+Adam converges fastest to the global minimum. Especially for small learning rates, CAGrad performs advantageous compared to EW. As reported in previous work, we found that PCGrad often would converge only to some point on the Pareto Front. The best and second best run for each learning rate over all MTO methods are indicated via font type. 

learning rate

method

10.0

5.0

1.0

0.5

0.1

0.05

0.01

0.005

0.001\*

 GD 

EW

-

103

-

-

-

-

-

-

-

PCGrad

-

-

-

-

-

-

-

-

-

CAGrad

644

-

213

621

8,069

5,732

20,418

34,405

-

 Adam 

EW

26

37

22

58

709

2,135

9,015

16,005

-

PCGrad

25

4,960

56

15,741

34,175

41,438

-

-

-

CAGrad

27

30

32

106

802

7,109

11,239

14,323

57,700

\*LR used for results in \[[31](#bib.bib31 "")\] with Adam

Table A4: Results for different MTO methods and optimizers on Cityscapes\[[9](#bib.bib9 "")\] using SegNet \[[1](#bib.bib1 "")\]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. While different MTO methods perform best over the distinct metrics, models trained with Adam outperform those based on SGD+mom in most direct comparisons. On the overall Δm\\Delta\_{m} metric, Adam shows superior performance for all MTO methods, in some cases even with a high margin. Best performance for each metric was also achieved by using Adam. 

Sem.Seg.

Depth

MTO

Optimizer

lr

mIoU ↑\\uparrow

pixAcc ↑\\uparrow

AbsErr ↓\\downarrow

RelErr ↓\\downarrow

DeltaM ↓\\downarrow

STL

adam

0.7122

0.9221

0.0134

29.88

EW

adam

0.005

0.6898

0.9165

0.0196

109.84

79.43 ±\\pm3.68

EW

signSGD

0.001

0.7013

0.9174

0.0210

115.67

86.52 ±\\pm6.06

EW

sgd

0.1

0.6967

0.9179

0.0216

113.82

86.24 ±\\pm1.97

UW

adam

0.001

0.7052

0.9202

0.0136

35.69

 5.44 ±\\pm2.38

UW

signSGD

0.001

0.6506

0.8997

0.0166

53.61

28.54 ±\\pm7.83

UW

sgd

0.01

0.6750

0.9110

0.0219

114.67

88.39 ±\\pm1.35

RLW

adam

0.001

0.7013

0.9196

0.0197

103.61

73.91 ±\\pm7.63

RLW

signSGD

0.0005

0.6962

0.9169

0.0204

112.46

82.90 ±\\pm5.32

RLW

sgd

0.1

0.6918

0.9156

0.0227

113.59

88.16 ±\\pm0.67

IMTL

adam

0.005

0.6963

0.9170

0.0148

45.63

16.55 ±\\pm1.52

IMTL

signSGD

0.005

0.6659

0.9070

0.0197

101.90

74.03 ±\\pm14.62

IMTL

sgd

0.01

0.6716

0.9107

0.0230

114.38

90.21 ±\\pm0.74

PCGrad

adam

0.01

0.6770

0.9135

0.0226

103.88

80.56 ±\\pm3.70

PCGrad

signSGD

0.001

0.6929

0.9170

0.0210

113.32

84.74 ±\\pm3.46

PCGrad

sgd

0.1

0.6972

0.9176

0.0235

107.06

84.09 ±\\pm0.93

CAGrad

adam

0.001

0.7088

0.9208

0.0162

66.39

35.81 ±\\pm14.91

CAGrad

signSGD

0.001

0.6883

0.9138

0.0182

107.06

74.68 ±\\pm18.91

CAGrad

sgd

0.1

0.6896

0.9156

0.0205

115.52

85.88 ±\\pm0.31

AlignedMTL

adam

0.0005

0.7164

0.9246

0.0154

64.89

32.76 ±\\pm5.27

AlignedMTL

signSGD

0.001

0.7010

0.9189

0.0180

102.41

69.77 ±\\pm24.91

AlignedMTL

sgd

0.1

0.6684

0.9103

0.0224

113.11

88.37 ±\\pm0.53

Adatask

adam

0.0005

0.7039

0.9196

0.0146

52.17

21.28 ±\\pm0.35

MTL-IO

sgd

0.1

0.6957

0.9172

0.0233

110.19

86.46 ±\\pm1.63

Table A5: Results for different MTO methods and optimizers on Cityscapes \[[9](#bib.bib9 "")\] using DeepLabV3+ \[[5](#bib.bib5 "")\]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Adam is Pareto dominant over SGD+mom in a direct pairwise comparison across all MTO methods. Here, we would also like to highlight that MTL can outperform STL as suggested by \[[4](#bib.bib4 "")\]. 

Sem.Seg.

Depth

MTO

Optimizer

lr

mIoU ↑\\uparrow

pixAcc ↑\\uparrow

AbsErr ↓\\downarrow

RelErr ↓\\downarrow

DeltaM ↓\\downarrow

STL

adam

0.7203

0.9253

0.0132

47.37

EW

adam

0.001

0.7247

0.9268

0.0128

47.73

-0.80 ±\\pm0.69

EW

signSGD

0.0005

0.7188

0.9249

0.0134

52.47

3.07 ±\\pm2.70

EW

sgd

0.05

0.7100

0.9217

0.0174

120.34

46.88 ±\\pm2.02

UW

adam

0.001

0.7224

0.9259

0.0122

44.37

-3.65±\\pm0.61

UW

signSGD

0.001

0.7172

0.9244

0.0127

48.31

-0.35 ±\\pm2.23

UW

sgd

0.005

0.7003

0.9187

0.0171

116.09

44.47 ±\\pm1.91

RLW

adam

0.001

0.7230

0.9263

0.0133

47.76

0.26 ±\\pm1.08

RLW

signSGD

0.001

0.7185

0.9244

0.0133

54.75

4.09 ±\\pm0.39

RLW

sgd

0.05

0.7070

0.9205

0.0176

123.13

48.84 ±\\pm1.79

IMTL

adam

0.001

0.7226

0.9259

0.0121

45.25

-3.38 ±\\pm0.92

IMTL

signSGD

0.001

0.7177

0.9243

0.0128

49.61

0.57 ±\\pm3.75

IMTL

sgd

0.005

0.7027

0.9192

0.0185

122.37

50.33 ±\\pm3.30

PCGrad

adam

0.001

0.7247

0.9272

0.0130

47.24

-0.58 ±\\pm0.56

PCGrad

signSGD

0.001

0.7187

0.9248

0.0132

53.35

3.28 ±\\pm1.41

PCGrad

sgd

0.05

0.7083

0.9212

0.0173

122.52

47.90 ±\\pm3.68

CAGrad

adam

0.001

0.7245

0.9264

0.0124

45.56

-2.59 ±\\pm0.68

CAGrad

signSGD

0.001

0.7154

0.9243

0.0128

49.06

0.37 ±\\pm0.41

CAGrad

sgd

0.1

0.7096

0.9220

0.0172

120.24

46.58 ±\\pm2.21

AlignedMTL

adam

0.0005

0.7225

0.9259

0.0122

47.70

-1.79 ±\\pm0.70

AlignedMTL

signsgd

0.001

0.7169

0.9241

0.0127

47.56

-0.70 ±\\pm0.61

AlignedMTL

sgd

0.1

0.7096

0.9218

0.0174

120.38

46.94 ±\\pm1.19

Adatask

adam

0.001

0.7234

0.9260

0.0124

47.21

–1.77 ±\\pm0.86

MTL-IO

sgd

0.05

0.7098

0.9217

0.0187

120.61

49.57 ±\\pm2.28

Table A6: Results for different MTO methods and optimizers on NYUv2\[[39](#bib.bib39 "")\] using SegNet \[[1](#bib.bib1 "")\]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Using Adam yields in superior performance in the majority of cases, both when considering the individual metrics and the overall Δm\\Delta\_{m} metric. We note that Δm\\Delta\_{m} is more effected by the normal task due to the higher number of corresponding metrics as can be observed in the case of UW. 

Sem.Seg.

Depth

Normal

MTO

Optimizer

lr

mIoU↑\\uparrow

pixAcc↑\\uparrow

AbsErr↓\\downarrow

RelErr↓\\downarrow

Mean↓\\downarrow

Median↓\\downarrow

<11.25↑<11.25\\uparrow

<22.5↑<22.5\\uparrow

<30.0↑<30.0\\uparrow

DeltaM↓\\downarrow

STL

adam

0.392

0.646

0.607

0.258

24.74

18.49

0.308

0.582

0.700

EW

adam

0.0001

0.398

0.650

0.530

0.212

29.53

25.02

0.216

0.454

0.584

10.08 ±\\pm2.84

EW

signSGD

0.0001

0.389

0.647

0.543

0.220

30.18

25.93

0.204

0.438

0.569

12.75 ±\\pm2.24

EW

sgd

0.01

0.384

0.644

0.551

0.227

30.12

25.72

0.212

0.443

0.571

12.84 ±\\pm0.65

UW

adam

0.0001

0.401

0.652

0.522

0.213

27.93

22.80

0.243

0.494

0.623

5.42 ±\\pm1.04

UW

signSGD

0.0001

0.384

0.634

0.535

0.218

28.36

23.45

0.233

0.482

0.612

8.05 ±\\pm1.18

UW

sgd

0.05

0.383

0.642

0.551

0.218

27.07

21.66

0.259

0.516

0.644

4.46 ±\\pm2.13

RLW

adam

0.0001

0.390

0.638

0.537

0.218

30.83

26.78

0.196

0.424

0.554

14.28 ±\\pm2.38

RLW

signSGD

0.0001

0.375

0.636

0.544

0.226

29.38

24.77

0.217

0.458

0.589

11.42 ±\\pm1.02

RLW

sgd

0.05

0.369

0.633

0.572

0.233

31.21

27.17

0.198

0.420

0.546

16.82 ±\\pm1.16

IMTL

adam

0.0001

0.380

0.644

0.524

0.216

26.35

20.77

0.270

0.534

0.660

2.06 ±\\pm1.31

IMTL

signSGD

0.0001

0.374

0.639

0.527

0.212

26.26

20.65

0.272

0.537

0.663

1.97 ±\\pm0.832

IMTL

sgd

0.05

0.396

0.656

0.532

0.215

26.16

20.38

0.277

0.542

0.666

0.72 ±\\pm0.68

PCGrad

adam

0.0001

0.406

0.654

0.529

0.215

28.60

23.75

0.231

0.477

0.606

7.39±\\pm0.86

PCGrad

signSGD

0.0001

0.394

0.649

0.556

0.219

28.98

24.25

0.223

0.467

0.598

9.59 ±\\pm1.02

PCGrad

sgd

0.01

0.389

0.644

0.547

0.223

29.76

25.22

0.214

0.451

0.580

11.61±\\pm0.36

CAGrad

adam

0.0001

0.405

0.661

0.527

0.211

25.85

20.02

0.282

0.550

0.674

-0.65 ±\\pm0.96

CAGrad

signSGD

0.0001

0.393

0.648

0.546

0.217

26.21

20.58

0.272

0.538

0.665

1.67 ±\\pm0.61

CAGrad

sgd

0.05

0.400

0.656

0.547

0.223

26.39

20.78

0.268

0.534

0.662

2.06 ±\\pm0.65

AlignedMTL

adam

0.0001

0.385

0.648

0.519

0.212

25.51

19.66

0.288

0.557

0.680

-0.86 ±\\pm0.14

AlignedMTL

signSGD

0.0001

0.363

0.637

0.528

0.211

25.86

20.02

0.282

0.550

0.674

0.90 ±\\pm0.54

AlignedMTL

sgd

0.05

0.371

0.642

0.531

0.214

26.41

20.71

0.272

0.535

0.661

2.33 ±\\pm1.00

adatask

adam

0.0001

0.394

0.650

0.518

0.211

26.29

20.67

0.273

0.536

0.661

1.00±\\pm0.8

MTL-IO

sgd

0.01

0.377

0.637

0.551

0.225

30.05

25.50

0.215

0.447

0.574

12.69±\\pm1.21

Table A7: Results for different MTO methods and optimizers on NYUv2\[[39](#bib.bib39 "")\] using DeepLabV3+ \[[5](#bib.bib5 "")\]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. We note a full dominance of Adam over SGD+mom on both the depth and normal tasks as well as on the Δm\\Delta\_{m} metric. Overall, best results for all metrics were also achieved using Adam as optimizer. 

Sem.Seg.

Depth

Normal

MTO

Optimizer

lr

mIoU↑\\uparrow

pixAcc↑\\uparrow

AbsErr↓\\downarrow

RelErr↓\\downarrow

Mean↓\\downarrow

Median↓\\downarrow

<11.25↑<11.25\\uparrow

<22.5↑<22.5\\uparrow

<30.0↑<30.0\\uparrow

DeltaM↓\\downarrow

STL

adam

0.552

0.767

0.365

0.152

21.16

14.52

0.402

0.668

0.768

EW

adam

0.0001

0.552

0.770

0.352

0.143

22.54

16.05

0.366

0.634

0.742

2.70 ±\\pm0.12

EW

signSGD

0.0001

0.554

0.766

0.362

0.147

22.69

16.15

0.364

0.630

0.739

3.60 ±\\pm0.38

EW

sgd

0.01

0.556

0.769

0.365

0.147

23.35

16.76

0.352

0.617

0.727

5.21 ±\\pm0.18

UW

adam

0.0005

0.532

0.755

0.358

0.141

22.01

15.30

0.383

0.650

0.753

1.59 ±\\pm0.14

UW

signSGD

0.0005

0.526

0.748

0.366

0.146

22.18

15.46

0.380

0.645

0.749

2.90 ±\\pm0.79

UW

sgd

0.01

0.557

0.769

0.364

0.148

22.37

15.69

0.374

0.641

0.746

2.57 ±\\pm0.56

RLW

adam

0.0001

0.556

0.768

0.360

0.150

22.37

15.87

0.370

0.638

0.745

2.96 ±\\pm0.55

RLW

signSGD

0.0001

0.557

0.768

0.363

0.146

22.40

15.84

0.371

0.638

0.745

2.74 ±\\pm0.30

RLW

sgd

0.05

0.541

0.762

0.368

0.151

23.00

16.44

0.357

0.625

0.734

5.15 ±\\pm0.54

IMTL

adam

0.0005

0.532

0.756

0.353

0.140

21.34

14.61

0.399

0.666

0.766

-0.39 ±\\pm0.05

IMTL

signSGD

0.0001

0.552

0.767

0.359

0.146

21.81

15.22

0.385

0.652

0.756

1.12 ±\\pm0.16

IMTL

sgd

0.05

0.545

0.762

0.367

0.150

21.63

14.98

0.390

0.658

0.760

1.30 ±\\pm0.40

PCGrad

adam

0.0001

0.558

0.772

0.357

0.146

22.33

15.81

0.371

0.639

0.746

2.34 ±\\pm0.48

PCGrad

signSGD

0.0001

0.553

0.766

0.360

0.148

22.61

16.11

0.365

0.631

0.739

3.55 ±\\pm0.13

PCGrad

sgd

0.01

0.556

0.769

0.361

0.147

23.36

16.79

0.351

0.617

0.728

5.10 ±\\pm0.28

CAGrad

adam

0.0005

0.531

0.755

0.357

0.143

21.42

14.67

0.399

0.664

0.764

0.10 ±\\pm0.55

CAGrad

signSGD

0.0001

0.550

0.766

0.361

0.146

21.94

15.27

0.383

0.650

0.754

1.49 ±\\pm0.12

CAGrad

sgd

0.05

0.556

0.768

0.360

0.144

21.77

14.99

0.391

0.657

0.759

0.42 ±\\pm0.07

AlignedMTL

adam

0.0005

0.531

0.752

0.355

0.144

21.15

14.49

0.403

0.669

0.768

-0.33 ±\\pm0.41

AlignedMTL

signSGD

0.0005

0.527

0.751

0.360

0.144

21.35

14.60

0.400

0.666

0.765

0.25 ±\\pm0.21

AlignedMTL

sgd

0.1

0.550

0.766

0.367

0.150

21.59

14.88

0.393

0.659

0.761

0.95 ±\\pm0.62

adatask

adam

0.0001

0.553

0.768

0.353

0.142

21.44

14.83

0.395

0.661

0.763

-0.39 ±\\pm0.21

MTL-IO

sgd

0.01

0.554

0.768

0.362

0.146

23.31

16.75

0.352

0.617

0.728

5.01 ±\\pm0.38

## Appendix A5 Additional gradient alignment results

We report extended evaluation on gradient simlarity in MTL and STL.

Alternatively to a single sample/ task, we consider the average gradient in [Fig. A5](#Pt0.A5.F5 "In Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning"). In [Fig. A6](#Pt0.A5.F6 "In Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning"), we differentiate between conflicting and supporting gradient pairs when evaluating the cosine similarity. [Figure A7](#Pt0.A5.F7 "In Appendix A5 Additional gradient alignment results ‣ Examining Common Paradigms in Multi-Task Learning") shows the evaluation of the scalar product as an combined measure of similarity in gradient direction and magnitude.

(a) Cityscapes,  
SegNet

(b) Cityscapes,  
  DeepLabV3

(c) NYUv2,  
  SegNet

(d) NYUv2,  
    DeepLabV3

Figure A5: Gradient similarities when averaging over batch/ losses. In contrast to results shown in [Fig. 4](#S4.F4 "In 4.2 Investigating gradient conflicts between tasks and samples ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning"), we compute either the average gradient over all tasks when comparing inter-samples or the average gradient over all samples within the batch for the comparison between inter-tasks. We observed a lower variance in some cases (e.g. Cityscapes+Segnet, grad. cosine similarity) which we trace back on noisy gradients being averaged out. Overall, we obtain the same findings as for a direct gradient comparison. 

![Refer to caption](2311.04698v5/gradsim__combi_posneg__CityScapes+aug__EW__HPS-SegNet__adam-lr0.001-wd1e-05__bs-64__seed-0.png) 

![Refer to caption](2311.04698v5/x1.png)

(a) Cityscapes,  
SegNet

![Refer to caption](2311.04698v5/gradsim__combi_posneg__CityScapes+aug__EW__HPS-resnet_dilated-DeepLabHead__adam-lr0.001-wd1e-05__bs-64__seed-0.png)

(b) Cityscapes,  
DeepLabV3

(c) NYUv2,  
SegNet

(d) NYUv2,  
DeepLabV3

(e) CelebA,  
ResNet50

Figure A6: Differentiation between conflicting and supportive gradients. We report mean (solid line), standard deviation (shaded area), upper (97.5%97.5\\%) and lower (2.5%2.5\\%) percentile (dotted line) of the gradient cosine similarity between either inter-samples gradients or inter-tasks gradients within an epoch. While showing overall results over all respective gradient pairs (Top) as can be also found in [Figure 4](#S4.F4 "In 4.2 Investigating gradient conflicts between tasks and samples ‣ 4 Experiments and results ‣ Examining Common Paradigms in Multi-Task Learning"), we also show the course of cosine similarity for either gradients that are conflicting (\[neg\], bottom) or those which have cosine similarity greater than zero (\[pos\], middle). 

(a) Cityscapes,  
SegNet

(b) Cityscapes,  
DeepLabV3

(c) NYUv2,  
SegNet

(d) NYUv2,  
DeepLabV3

(e) CelebA,  
ResNet50

Figure A7: Scalar product between pairs of gradients. We report mean (solid line), standard deviation (shaded area), upper (97.5%97.5\\%) and lower (2.5%2.5\\%) percentile (dotted line) of the gradient cosine similarity between either gradients of inter-samples or inter-tasks within an epoch. We observe an overall decrease of the variance of the scalar product for both Cityscapes setups and the NYUv2+DeepLabV3 experiment over the training which we explain with evenly smaller overall gradients. Surprisingly, this does not apply for NYUv2 with SegNet or CelebA. Similar to previous results, we do not see any indication for inter-sample gradients being better aligned than inter-tasks gradients. 

## Appendix A6 Robustness of multi-task representations on corrupted data

In this additonal part of our analysis, we investigate whether features learned for multiple tasks generalize better to corrupted data compared to those learned for single tasks only.

Motivation: In his seminal paper, Caruana gives preliminary evidence that MTL provides stronger features and avoids spurious correlations (referred to better *attribute selection*) \[[4](#bib.bib4 "")\]. More recently, spurious correlations have often been directly connected with robustness \[[13](#bib.bib13 ""), [18](#bib.bib18 "")\]. Results from current literature on the robustness of MTL features are mixed. While MTL is stated to increase the adversarial- and noise-robustness over STL\[[24](#bib.bib24 ""), [37](#bib.bib37 ""), [54](#bib.bib54 "")\], others argue features selected by MTL could be more likely to be non-causal and, therefore, less robust \[[17](#bib.bib17 ""), [2](#bib.bib2 "")\]. Here, we further examine whether MTL features lead to better robustness. We would like to nuance that we do not consider the transferability of representations, e.g., to new tasks, but solely focus on the claim that the MTL trained features are more robust w.r.t. different inputs.

Approach: In our experiment, we treat the common corruptions \[[16](#bib.bib16 "")\] as downstream task and compare the performance after fine-tuning the heads on corrupted data while freezing the pre-trained STL/MTL backbone. While this differs from the typical OOD setup, here, it allows us to explore whether MTL or STL yield more robust representation for corrupted data.

We select models trained on clean data with the best performing hyperparameter configuration from previous experiments and fine-tune their heads on corrupted data. Following this, we compare the test performance of models trained in the multi-task setup to those that were learned for a single-task only. We use the perturbation modes proposed by Hendrycks et al. \[[16](#bib.bib16 "")\] which include different variants of noise, blur, and weather conditions and apply five levels of severity. We randomly select corruption and severity level for each data sample during fine-tuning and create a full corrupted version of the test data considering all proposed corruptions and perturbation levels.

To quantify the robustness of single- and multi-task models, we first compute the individual task metrics MM (e.g., mIoU) per task tt for a STL and MTL network. Next, we compute the relative performance when each model is faced with corrupted data. Lastly, we calculate the difference of relative performances of the MTL compared to the STL model. In detail, over all corruption modes CC and levels of severity SS we have

δt\\displaystyle\\delta\_{t}

\=1|C|⋅|S|​∑c∈C∑s∈S(−1)p⁡(t)​δt,c,s\\displaystyle=\\frac{1}{|C|\\cdot|S|}\\sum\_{c\\in C}\\sum\_{s\\in S}(-1)^{p(t)}\\delta\_{t,c,s}

(26)

withδt,c,s\=Mt,c,sM​T​L,c​o​r​r​u​p​t​e​dMtM​T​L,c​l​e​a​n−Mt,c,sS​T​L,c​o​r​r​u​p​t​e​dMtS​T​L,c​l​e​a​n\\displaystyle\\text{with}\\quad\\delta\_{t,c,s}=\\frac{M^{MTL,corrupted}\_{t,c,s}}{M^{MTL,clean}\_{t}}-\\frac{M^{STL,corrupted}\_{t,c,s}}{M^{STL,clean}\_{t}}

where p⁡(t)\=1p(t)=1 if a higher value on task tt corresponds to better performance and p⁡(t)\=0p(t)=0 otherwise. This metric yields δt<0\\delta\_{t}<0 if the MTL model was able to handle data corruption better. If the STL model is less impacted, we get δt\>0\\delta\_{t}>0.

(a) Cityscapes, SegNet

(b) Cityscapes, DeepLabV3

(c) NYUv2, SegNet

(d) NYUv2, DeepLabV3

Figure A8: Transfer to out-of-distribution data for MTL and STL. For every task and respective metrics, we show the difference over relative performance decrease over all corruption modes averaged over five levels of severity and three runs. EW was used to train the MTL model on uncorrupted data. We color blocks in case either STL or MTL is able to handle the respective corruption better for all metrics of one task. Regarding the Cityscapes dataset, the performance on both task would strongly benefit in MTL setup. A similar behavior can be seen for NYUv2+DeepLabV3. Using SegNet on NYUv2, however, shows preferences towards STL features. Overall, we see a minor indication that MTL result in features that would generalize better to corrupted data. 

Table A8: Out-Of Distribution transfer on corrupted Cityscapes \[[9](#bib.bib9 "")\] dataset for different networks and MTO methods. We report difference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. [Equation 26](#Pt0.A6.E26 "In Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning")). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. Overall, we observe a slight benefit in performance for the depth task when training for multiple tasks. 

|           |        | Sem.Seg. |        | Depth  |  | Mean    |         |  |         |
| --------- | ------ | -------- | ------ | ------ |  | ------- | ------- |  | ------- |
|           |        |          |        |        |  |         |         |  |         |
|           | UW     |          | 0.0854 | 0.0421 |  | 0.7799  | -0.2118 |  | 0.1739  |
|           | RLW    |          | 0.0699 | 0.0307 |  | 0.6553  | -0.5652 |  | 0.0477  |
|           | IMTL   |          | 0.0696 | 0.0243 |  | 0.5384  | 0.1108  |  | 0.1858  |
|           | PCGrad |          | 0.0824 | 0.0316 |  | 0.4106  | -0.6392 |  | -0.0286 |
|           | CAGrad |          | 0.0561 | 0.0213 |  | 0.3860  | -0.3100 |  | 0.0383  |
| DeepLabV3 | EW     |          | 0.0087 | 0.0017 |  | -0.0108 | -0.1826 |  | -0.0457 |
|           | UW     |          | 0.0110 | 0.0018 |  | 0.1179  | 0.0861  |  | 0.0542  |
|           | RLW    |          | 0.0263 | 0.0084 |  | -0.0809 | -0.3595 |  | -0.1015 |
|           | IMTL   |          | 0.0172 | 0.0041 |  | 0.1747  | 0.0857  |  | 0.0704  |
|           | PCGrad |          | 0.0090 | 0.0019 |  | 0.0394  | -0.2240 |  | -0.0434 |
|           | CAGrad |          | 0.0224 | 0.0067 |  | 0.0898  | -0.2841 |  | -0.0413 |

Table A9: Out-Of Distribution transfer on corrupted NYUv2 \[[39](#bib.bib39 "")\] dataset for different multi-task optimization methods. We report difference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. [Equation 26](#Pt0.A6.E26 "In Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning")). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. While for DeepLabV3 largely benefits from MTL, this is not the case for SegNet. Over both networks, EW profits shows lowest relative performance decrease among all MTO methods. Interestingly we found that even for different metrics corresponding to the same task, either the multi-task or single-task learning model would show lower decrease in performance on the corrupted data. 

|           |        | Sem.Seg. |         | Depth   |  | Normal  |         | Mean |         |         |         |         |         |  |         |
| --------- | ------ | -------- | ------- | ------- |  | ------- | ------- | ---- | ------- | ------- | ------- | ------- | ------- |  | ------- |
|           |        |          |         |         |  |         |         |      |         |         |         |         |         |  |         |
|           | UW     |          | 0.0166  | 0.0149  |  | 0.1146  | 0.1225  |      | -0.0564 | -0.0984 | 0.0313  | 0.0207  | 0.0164  |  | 0.0204  |
|           | RLW    |          | 0.0174  | 0.0223  |  | -0.0062 | 0.0281  |      | -0.1048 | -0.1857 | 0.0229  | 0.0089  | 0.0055  |  | -0.0213 |
|           | IMTL   |          | 0.0073  | 0.0123  |  | 0.1108  | 0.1247  |      | -0.0186 | -0.0390 | 0.0361  | 0.0274  | 0.0228  |  | 0.0315  |
|           | PCGrad |          | 0.0186  | 0.0174  |  | 0.0720  | 0.0817  |      | -0.0702 | -0.1271 | 0.0264  | 0.0158  | 0.0121  |  | 0.0052  |
|           | CAGrad |          | 0.0061  | 0.0116  |  | 0.1172  | 0.1428  |      | -0.0082 | -0.0202 | 0.0465  | 0.0286  | 0.0223  |  | 0.0385  |
| DeepLabV3 | EW     |          | -0.0289 | -0.0187 |  | -0.1194 | -0.1249 |      | -0.1182 | -0.1947 | -0.0409 | -0.0288 | -0.0238 |  | -0.0776 |
|           | UW     |          | -0.0268 | -0.0165 |  | -0.0263 | -0.0219 |      | -0.0465 | -0.0701 | -0.0237 | -0.0139 | -0.0113 |  | -0.0286 |
|           | RLW    |          | -0.0084 | -0.0054 |  | -0.0075 | -0.0187 |      | -0.0390 | -0.0597 | -0.0051 | 0.0009  | 0.0005  |  | -0.0158 |
|           | IMTL   |          | -0.0211 | -0.0167 |  | 0.0130  | 0.0059  |      | -0.0153 | -0.0219 | -0.0072 | -0.0045 | -0.0038 |  | -0.0079 |
|           | PCGrad |          | -0.0102 | -0.0081 |  | -0.0242 | -0.0327 |      | -0.0504 | -0.0800 | -0.0236 | -0.0107 | -0.0076 |  | -0.0275 |
|           | CAGrad |          | -0.0139 | -0.0129 |  | 0.0022  | -0.0123 |      | -0.0096 | -0.0063 | 0.0023  | 0.0047  | 0.0034  |  | -0.0047 |

|           |        | Sem.Seg. |        | Depth  |  |        |        |
| --------- | ------ | -------- | ------ | ------ |  | ------ | ------ |
|           |        |          |        |        |  |        |        |
|           | UW     |          | 0.3689 | 0.7273 |  | 0.0503 | 77.86  |
|           | RLW    |          | 0.3744 | 0.7321 |  | 0.0526 | 96.26  |
|           | IMTL   |          | 0.3684 | 0.7363 |  | 0.0479 | 78.92  |
|           | PCGrad |          | 0.3678 | 0.7310 |  | 0.0515 | 94.44  |
|           | CAGrad |          | 0.3769 | 0.7397 |  | 0.0479 | 92.75  |
| DeepLabV3 | STL    |          | 0.5732 | 0.8593 |  | 0.0244 | 98.97  |
|           | EW     |          | 0.5769 | 0.8619 |  | 0.0253 | 104.37 |
|           | UW     |          | 0.5801 | 0.8632 |  | 0.0240 | 99.67  |
|           | RLW    |          | 0.5634 | 0.8557 |  | 0.0270 | 103.22 |
|           | IMTL   |          | 0.5769 | 0.8617 |  | 0.0244 | 101.26 |
|           | PCGrad |          | 0.5742 | 0.8609 |  | 0.0263 | 105.46 |
|           | CAGrad |          | 0.5690 | 0.8576 |  | 0.0259 | 104.90 |

Table A10: Results for evaluating on corrupted Cityscapes \[[9](#bib.bib9 "")\]. Scores are averaged over all corruption modes, level of severity and three seeds. 

|           |        | Sem.Seg. |        | Depth  |  | Normal |        |  |       |       |        |        |        |
| --------- | ------ | -------- | ------ | ------ |  | ------ | ------ |  | ----- | ----- | ------ | ------ | ------ |
|           |        |          |        |        |  |        |        |  |       |       |        |        |        |
|           | UW     |          | 0.2266 | 0.4750 |  | 0.7847 | 0.2913 |  | 34.90 | 31.25 | 0.1445 | 0.3538 | 0.4814 |
|           | RLW    |          | 0.2182 | 0.4587 |  | 0.7718 | 0.2957 |  | 37.09 | 34.39 | 0.1165 | 0.3045 | 0.4298 |
|           | IMTL   |          | 0.2190 | 0.4743 |  | 0.7804 | 0.2885 |  | 33.89 | 29.67 | 0.1609 | 0.3812 | 0.5085 |
|           | PCGrad |          | 0.2279 | 0.4751 |  | 0.7790 | 0.2900 |  | 35.34 | 31.90 | 0.1388 | 0.3423 | 0.4696 |
|           | CAGrad |          | 0.2324 | 0.4850 |  | 0.7822 | 0.2897 |  | 33.50 | 29.10 | 0.1648 | 0.3908 | 0.5183 |
| DeepLabV3 | STL    |          | 0.3793 | 0.6285 |  | 0.5342 | 0.2111 |  | 27.32 | 21.25 | 0.2853 | 0.5286 | 0.6414 |
|           | EW     |          | 0.3644 | 0.6132 |  | 0.5594 | 0.2181 |  | 29.88 | 24.80 | 0.2359 | 0.4657 | 0.5851 |
|           | UW     |          | 0.3836 | 0.6321 |  | 0.5137 | 0.2003 |  | 27.75 | 21.89 | 0.2738 | 0.5169 | 0.6324 |
|           | RLW    |          | 0.3861 | 0.6338 |  | 0.5290 | 0.2081 |  | 28.33 | 23.00 | 0.2566 | 0.4964 | 0.6162 |
|           | IMTL   |          | 0.3773 | 0.6323 |  | 0.5208 | 0.1991 |  | 27.27 | 21.34 | 0.2830 | 0.5273 | 0.6412 |
|           | PCGrad |          | 0.3932 | 0.6409 |  | 0.5168 | 0.2025 |  | 28.05 | 22.44 | 0.2659 | 0.5063 | 0.6238 |
|           | CAGrad |          | 0.3738 | 0.6289 |  | 0.5266 | 0.2022 |  | 27.68 | 21.86 | 0.2754 | 0.5177 | 0.6324 |

Table A11: Results for evaluating on corrupted NYUv2 \[[39](#bib.bib39 "")\]. Scores are averaged over all corruption modes, level of severity and three seeds. 

Results: [Figure A8](#Pt0.A6.F8 "In Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning") shows δt\\delta\_{t} for different corruption types for all network architectures and datasets with EW averaged over five corruption levels and three seeds. Considering the combination DeepLabV3 and Cityscapes, on the semantic segmentation tasks, the STL models show a slightly lower decrease in performance on the corrupted data than MTL (δt\>0\\delta\_{t}>0 more often; shaded in red), indicating that the features learned for these respective tasks can better generalize to corrupted data. In contrast, the MTL model shows significant better relative performance on the depth task (δt<0\\delta\_{t}<0 more often; shaded green). Comparing these observations to other dataset+network combinations, we find a strong robustness of MTL features for both setups of CityScape+SegNet and NYUv2+DeepLabV3 over all tasks. However, evaluation on NYUv2+SegNet resulted on average in better performance on STL features, especially for the depth task. Furthermore, we see little evidence for general higher robustness against certain types of corruption (e.g., higher robustness against weather conditions) for either MTL or STL across all setups.

The results of other MTO methods ([Tabs. A8](#Pt0.A6.T8 "In Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning") and [A9](#Pt0.A6.T9 "Table A9 ‣ Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning")) indicate that it depends less on the used method but more on the choice of dataset and network architecture whether some tasks would benefit from MTL for learning more robust features. Averaged absolute scores can be found in [Tabs. A10](#Pt0.A6.T10 "In Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning") and [A11](#Pt0.A6.T11 "Table A11 ‣ Appendix A6 Robustness of multi-task representations on corrupted data ‣ Examining Common Paradigms in Multi-Task Learning").

Conclusion: Our experiments show that MTL can result in learning more robust features, either for a subset of tasks or even all. However, we could not observe a uniform pattern whether certain tasks consistently benefit from MTL. Instead, it depends on the task, the type of corruption, the network, and the dataset whether MTL or STL is superior towards corrupted data. Whether there is a general pattern, we leave to further research. We further cannot fully confirm the outcome of \[[24](#bib.bib24 "")\] as only two of our setups have indicated that the segmentation task can be more robust in the MTL setting. Controversial to the claim of \[[37](#bib.bib37 "")\], our evaluation shows that none of the MTL approaches, even IMTL, PCGrad or CAGrad which adjust the gradients, yields consistent values of δt<0\\delta\_{t}<0 which would have shown an advantage of certain MTO methods over STL.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")