# Examining Common Paradigms in Multi-Task Learning

Cathrin Elich<sup>‡,1,2,3</sup>, Lukas Kirchdorfer<sup>‡,1,4</sup>, Jan M. K¨ohler<sup>∗,1</sup>, and Lukas Schott<sup>∗,1</sup>

<sup>1</sup> Bosch Center for Artificial Intelligence <sup>2</sup> Max Planck Institute for Intelligent Systems, T¨ubingen, Germany  
3 Max Planck ETH Center for Learning Systems  
<sup>4</sup> University of Mannheim  
cathrin.elich@tuebingen.mpg.de, {jan.koehler,lukas.schott}@bosch.com  
<sup>‡</sup>Work done during an internship at Bosch. <sup>∗</sup>Joint senior authors.

Abstract. While multi-task learning (MTL) has gained significant attention in recent years, its underlying mechanisms remain poorly understood. Recent methods did not yield consistent performance improvements over single task learning (STL) baselines, underscoring the importance of gaining more profound insights about challenges specific to MTL. In our study, we investigate paradigms in MTL in the context of STL: First, the impact of the choice of optimizer has only been mildly investigated in MTL. We show the pivotal role of common STL tools such as the Adam optimizer in MTL empirically in various experiments. To further investigate Adam’s efectiveness, we theoretical derive a partial loss-scale invariance under mild assumptions. Second, the notion of gradient conflicts has often been phrased as a specific problem in MTL. We explore the role of gradient conflicts in MTL and compare it to STL. For angular gradient alignment we find no evidence that this is a unique problem in MTL. We emphasize diferences in gradient magnitude as the main distinguishing factor. Overall, we find surprising similarities between STL and MTL suggesting to consider methods from both fields in a broader context.

Keywords: Multi-task learning · Deep Learning · Computer Vision

## 1 Introduction

Multi-task learning (MTL) is gaining significance in the deep learning literature and in industry applications. Especially, tasks like autonomous driving and robotics necessitate real-time execution of neural networks while obeying constraints of limited computational resources. Consequently, there is a demand for neural networks capable of simultaneously inferring multiple tasks [19, 26].

In a seminal study, Caruana [4] highlights both advantages and challenges in MTL. On the one hand, certain tasks can exhibit a symbiotic relationship, resulting in a mutual performance enhancement when trained together. On the other hand, conflicts between tasks can arise and decrease the performance when trained jointly, also known as negative transfer.

Several approaches have been suggested to mitigate the issue of negative transfer among tasks during network training. Our study focuses on two main branches in the literature: First, gradient magnitude methods which incorporate weights to scale task-specific losses to achieve an adequate balance between tasks. Second, gradient alignment methods which aim to resolve conflicts in gradient vectors that may arise between tasks within a shared network backbone.

The efectiveness of the proposed MTL methods remain uncertain in the literature. Upon comparing various studies, it becomes evident that there is no definitive approach that consistently performs well across diferent settings [48]. This observation has been reinforced in more recent studies where competitive performance was achieved through plain unitary scaling in combination with common regularization methods [25] or tuned task weighting [50].

The current understanding of MTL still lacks a deeper comprehension of its underlying mechanisms. To address this gap, our study aims to examine commonly held paradigms, such as the choice of optimizer, as well as the notion of gradient alignment and gradient magnitudes. Our contributions are:

– The impact of of-the-shelf optimizers has received little attention in MTL benchmarks. We evaluate the Adam [22] optimizer and demonstrate its favorable performance over SGD+momentum in various experiments.  
– We provide a potential explanation for Adam’s efectiveness in MTL by theoretically demonstrating a partial invariance w.r.t. to diferent loss scalings. Similarly, we derive a full invariance for an optimal variation of the wellestablished used method of uncertainty weighting [21].  
– So far gradient alignment conflicts have mostly been considered between different tasks [7, 20, 31, 56]. We present empirical evidence that conflicts arising from gradient alignment between tasks are not exclusive and can even be more pronounced between diferent samples within a task.  
– Corroborating the methods proposed to balance gradient magnitude conflicts in MTL [21, 32, 34, 50], we confirm that gradient magnitudes pose a challenge between tasks and is less pronounced between samples within a task.  
– We examine the presumption of increased robustness on corrupted data as a result of MTL [24, 37]. We find light evidence that a higher number of tasks can result in improved transferability. Due to page limitations, we moved these results to App. A6, focusing the more compelling findings in the main text.  
Overall, we provide a vast set of experiments and theoretical insights which contribute to a more comprehensive understanding of MTL in computer vision.

## 2 Related Work

Work in multi-task learning (MTL) can be roughly divided into three fields:

Network architectures focus on the question of how features should be shared across tasks, e.g. [34, 36, 38, 51]. Multi-task optimization (MTO) aims to resolve imbalances and conflicts of tasks during MTL. Task afinities examine a grouping of tasks that should be learned together to benefit from the joint training [12,47].

A general overview of recent works in MTL can be found in [43, 48]. Our work focuses on MTO, which we review more thoroughly in the following.

Gradient magnitude methods prevent the dominance of individual tasks by balancing them with task-specific weights. One line of works are loss-weighting methods. Here, weights are determined before any (task-wise) gradient computation and are used for a weighted aggregation of the tasks’ losses. These methods consider either the task uncertainty (UW) [21], rate of change of the losses (DWA, FAMO) [30, 34], the tasks’ dificulty (DTP) [14], validation performance by applying meta-learning (MOML, Auto-λ) [33, 53], or randomly chosen task weights (RLW) [28]. In line with these, the geometric mean of task losses has been used to handle the diferent convergence rates of the tasks [8]. An advantage of theses methods is their computational eficiency as the gradient needs to be computed only once for the aggregated loss. Alternatively, other methods consider the task-specific gradients directly, e.g., by normalizing them (Grad-Norm) [6] or propose a hybrid balancing between task-wise loss and gradient scaling (IMTL, DB-MTL) [27, 32]. Furthermore, there are several adaptions for the multiple-gradient descent algorithm (MGDA) [10], e.g. for applying it eficiently in deep learning setups [44] or by introducing a stochastic gradient correction [11]. Recently, task-wise gradient weights have been estimated by treating MTL as a bargaining problem (Nash-MTL) [40], or considering a stability criterion (Aligned-MTL) [45]. Crucially, all gradient magnitude methods consider scalar weightings of task-wise gradients within the backbone and/or heads. They do not modify the alignment of task-specific gradient vectors.

Gradient alignment methods perform more profound vector manipulations on the task-wise gradients w.r.t. to the network weights of a shared backbone before aggregating them. The underlying assumption indicates conflicting gradients as a major problem in MTL. To address this, GradDrop [7] randomly drops gradient components in the case of opposing signs. PCGrad [56] proposes to circumvent problems of conflicting gradients by projecting them onto each other’s normal plane. Following this idea, Liu et al. [31] propose CAGrad to converge to a minimum of the average loss instead of any point on the Pareto front. RotoGrad [20] rotates gradients at the intersection of the heads and backbone to improve their alignment. Shi et al. [46] propose to alter the network architecture based on the occurrence of layer-wise gradient conflicts. Lastly, [41] use separate optimizers such as SGD and SGD+momentum per task. This is extended to AdaGrad, RMSProp and Adam in AdaTask [52].

Recent studies question the efectiveness of optimization-based methods in MTL. Xin et al. [50] execute an extensive hyperparameter search to show that simple scalar task-weighting performs equivalent or superior to many aforementioned multi-task optimization methods. Their hyperparameter search not only include the task-weights, but also common deep learning parameters such as the learning rate and regularization. Concurrently, Kurin et al. [25] empirically show that fixed task-weights combined with regularization and stabilization techniques yield to equivalent performance compared to sophisticated multi-task optimization methods. Following these, Royer et al. [42] examine the role of model capacity for MTL performance as well as the occurrences of gradient conflicts. We extend these critical studies. In particular, we theoretically and empirically demonstrate that the choice of optimizer is crucial and could potentially help to explain discrepancies found in prior studies (4.1). We further specifically distinguish between gradient conflicts between tasks and samples (4.2).

## 3 Problem Statement

Multi-task learning addresses the problem of learning a set of $T$ tasks simultaneously (see e.g. [4, 48]). It is noteworthy that this setup is occasionally also referred to as multi-label or multi-target learning [57] Importantly, this study does not incorporate multi-input data. We consider a supervised learning setup, use a shared backbone architecture, and learn all tasks together. Formally, given input data $x ,$ the goal is to learn a function $f _ { \pmb { \theta } } ( \mathbf { x } )$ which maps a point $\mathbf { x } \in \mathcal { X }$ to each task label $y _ { t }$ with $t = 1 , . . , T$ . The trainable parameters $\pmb { \theta } = \{ \phi , \psi _ { 1 : T } \}$ consist of shared parameters ϕ and task-specific parameters $\psi _ { t }$ . Training a task t is associated with the loss $\mathcal { L } _ { t } ( f _ { \theta } ( x ) ; \theta )$ , e.g., a regression or classification loss. We denote respective gradients on the shared and task-specific parameters with $\mathbf { g } _ { t } ^ { \phi } = \nabla _ { \phi } { \mathcal { L } } _ { t }$ , and $\mathbf { g } _ { t } ^ { \psi } = \nabla _ { \psi } \mathcal { L } _ { t }$ . When training on multiple tasks, the shared parameters ϕ needs to be updated w.r.t. all task-wise gradients $\mathbf { g } _ { t } ^ { \phi }$ which requires an appropriate aggregation. A simple solution is to uniformly sum up the task losses $\textstyle { \mathcal { L } } = \sum _ { t } { \mathcal { L } } _ { t }$ which is referred to as Equal Weighting (EW). However, as tasks might be competing against each other, this can result in negative transfer and thus sub-optimal solutions. One way to deal with this dificulty is to adapt the magnitude of task-specific gradients. This can be achieved by weighting tasks during training, $\mathrm { e . g . }$ , by scaling diferent losses $\begin{array} { r } { \mathcal { L } = \sum _ { t } \alpha _ { t } \mathcal { L } _ { t } , } \end{array}$ where $\alpha _ { t } \geq 0$ . Note that the $\alpha _ { t }$ can change during training. Furthermore, the weighing can also be performed on gradient level to distinguish between shared and task-specific gradients. We refer to those approaches as gradient magnitude methods. Interestingly, the relationship between loss weights, network updates and learning rate also depends on the optimizer. We show a derivation for SGD and Adam in Appendix A1.2. Additionally to adapting the gradient magnitude, one can directly adapt the alignment of task-wise gradient vectors within the shared backbone $\tilde { \mathbf { g } } ^ { \phi } = \mathbf { h } ( \mathbf { g _ { 1 } } ^ { \phi } , . . . , \mathbf { g _ { T } } ^ { \phi } )$

In practice, an optimum for θ that yields best performance on all tasks often does not exist. Instead, improving performance on some task often yields a performance decrease in another task. To still enable a comparison across network instances in MTL, an instance $\pmb { \theta } ^ { * }$ is called to be Pareto optimal, if there is no other $\pmb { \theta } ^ { \prime }$ such that $\mathcal { L } _ { t } ( { \pmb \theta } ^ { \prime } ) \leq \mathcal { L } _ { t } ( { \pmb \theta } ^ { * } )$ ∀t with strict inequality in at least one task. The Pareto front consists of the Pareto optimal solutions.

## 4 Experiments and results

In this section we perform several experiments to gain a more profound understanding of multi-task learning (MTL) in computer vision by questioning common paradigms. We compare the impact of Adam and SGD in MTL in Sec. 4.1 and examine the process of gradient similarity in diferent settings in Sec. 4.2. Throughout this evaluation, we repeatedly make use of common setups, which we will specify as follows and in more detail in App. A3.

Datasets: For our experiments, we consider three diferent datasets that are commonly used for evaluating MTL in computer vision: Cityscapes [9] contains images of urban street scenes. In line with previous work, we consider the tasks of semantic segmentation (7 classes) and depth estimation. NYUv2 [39] is an indoor dataset for scene understanding which was recorded over 464 diferent scenes across three diferent cities. Besides semantic segmentation (13-class) and depth estimation, it also contains the task of surface normal prediction. CelebA [35] consists of 200K face images which are labeled with 40 binary attributes.

Networks: We use network architectures with hard-parameter sharing which consist of a shared backbone and task-specific heads. For the dense prediction tasks on Cityscapes and NYUv2, we compare SegNet [1] and DeepLabV3+ [5]. Experiments on CelebA are performed on a ResNet-18 [15] with an additional single linear layer for each head.

Training: For each method, we follow the loss or gradient aggregation as described in the related work, $\mathrm { e . g . } $ for equal weighting all task-specific losses are simply summed up to compute the joint network gradients. The learning rate is tuned separately for each approach. We use the validation set performance of the $\varDelta _ { m }$ metric as early stopping criteria. The $\varDelta _ { m }$ metric [36] measures the average relative task performance drop of a method m compared to the single-task baseline b using the same backbone and is computed as $\varDelta _ { m } =$ $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( - 1 ) ^ { l _ { t } } ( M _ { m , t } - M _ { b , t } ) / \bar { M _ { b , t } } } \end{array}$ where $l _ { t } = 1 \mathrm { i f } \mathrm { a }$ higher value means better for measure $M _ { \cdot , t }$ of some task metric t, and 0 otherwise.

## 4.1 Efectiveness of Adam in multi-task learning

Examined paradigm: The impact of the choice of standard optimizer is often disregarded and varies across studies (overview in Tab. A1) when comparing MTL methods. For instance, Adam [22] was successfully used to show that random/constant weighting of tasks’ losses performs competitive compared to MTO methods [25,28,50]. In contrast, many methods proposing adaptive, taskspecific weighting methods [21, 32] use stochastic gradient descent with momentum (SGD+mom). In recent works, the optimizer choice converged to Adam and a fixed learning rate schedule [31,34,45,56] without a comparison to SGD+mom.

In this part of our study, we investigate the impact of Adam and SGD+mom in conjunction with common MTO methods. We identify the choice of optimizer as a crucial confounder in the experimental setup. Compared to SGD+mom, we find that the Adam optimizer itself is a quite efective baseline in MTL and can be regarded as a loss weighting method from a theoretical viewpoint.

Toy Task Experiment To get a first impression of the impact of the optimizer and common hyperparameters such as the learning rate, we investigate the impact of Adam and plain gradient descent (GD) in a simple toy task.

![](images/6480a0f85498c3859686019c1b6262d9355f7080c43f29f7d42c5108529d418e.jpg)

<details>
<summary>heatmap</summary>

| Method | Learning Rate | GD::GD::1.0 | GD::GD::0.05 | GD::GD::0.001 | Adam::1.0 | Adam::0.05 | Adam::0.001 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EW | 1.0 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
| EW | 0.05 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
| EW | 0.001 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
| CAGrad | 1.0 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
| CAGrad | 0.05 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
| CAGrad | 0.001 | [Image] | [Image] | [Image] | [Image] | [Image] | [Image] |
</details>

Fig. 1: Toy task experiment from CAGrad [31] for diferent learning rates and optimizers. Consistent with results from [50], we observe that the choice of the learning rate is crucial even for this toy optimization problem. Moreover, it becomes apparent, that selecting Adam over simple gradient decent (GD) yields superior results. The contour lines depict the 2D loss landscape; the optimization trajectories are colored from red to yellow for 100k iteration steps from three diferent starting points (seeds).

Table 1: Maximum number of iterations for all seeds in the toy task experiment from [31] to reach the global minimum for varying MTO method, learning rate, and optimizer combination. EW+Adam often shows the fastest convergence to the global minimum. ’-’ denotes that not all seeds converged within 100k iterations. As reported in [31], PCGrad often only converges to a point on the Pareto Front. We highlight the best run for each learning rate over all MTO methods.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">learning rate</td><td rowspan="2"></td><td rowspan="2">method</td><td colspan="5">learning rate</td></tr><tr><td>10.0</td><td>1.0</td><td>0.1</td><td>0.01</td><td>0.001*</td><td>10.0</td><td>1.0</td><td>0.1</td><td>0.01</td><td>0.001*</td></tr><tr><td rowspan="3">GD</td><td>EW</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td rowspan="3">Adam</td><td>EW</td><td>26</td><td>22</td><td>709</td><td>9,015</td><td>-</td></tr><tr><td>PCGrad</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>PCGrad</td><td>25</td><td>56</td><td>34,175</td><td>-</td><td>-</td></tr><tr><td>CAGrad</td><td>644</td><td>213</td><td>8,069</td><td>20,418</td><td>-</td><td>CAGrad</td><td>27</td><td>32</td><td>802</td><td>11,239</td><td>57,700</td></tr></table>

Approach: We repeat the experiment of Liu et al. [31] using their original implementation but further test diferent learning rates and optimizers. They motivate their gradient alignment method CAGrad with a simple toy optimization problem in which their method reliably converges to the minimum of the average loss, while other MTO approaches would either get stuck (e.g., EW) or only converge to any point on the Pareto front (e.g., PCGrad [56], MGDA [44]).

Result: For higher learning rates with Adam optimizer, even equal weighting (EW) reaches the global optimum (cf. Fig. 1, e.g., EW+Adam, lr=0.05) and often converges even faster than dedicated MTO methods (Tab. 1). Note, original results were shown for learning rate 0.001 using Adam and were, thus, in favor of CAGrad. Results for additional learning rates are reported in Tab. A3.

Conclusion: The choice of optimizer appears to be more important on the success of the outcome of this experiment than the choice of MTO method, as Adam converges considerably faster and more reliably than GD. Also, tuning the learning rate is a relevant factor, however, especially in MTL with diferently scaled losses, a single suitable learning rate for all tasks does often not exist.

Table 2: Number of Pareto optimal (PO) experiments using either Adam or SGD+mom. as optimizer. Models trained with Adam are consistently more often on the Pareto front compared to those trained with SGD+mom. The number of Adam-based runs that are not dominated by any SGD-based run (PO w.r.t. SGD) is even higher, while the reverse does not apply.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">Adam</td><td colspan="2">SGD+mom.</td></tr><tr><td>PO (full)</td><td>PO w.r.t. SGD</td><td>PO (full)</td><td>PO w.r.t. Adam</td></tr><tr><td>Cityscapes</td><td>SegNet</td><td>5</td><td>24</td><td>0</td><td>0</td></tr><tr><td>Cityscapes</td><td>DeepLabV3</td><td>10</td><td>24</td><td>0</td><td>0</td></tr><tr><td>NYUv2</td><td>SegNet</td><td>11</td><td>21</td><td>1</td><td>1</td></tr><tr><td>NYUv2</td><td>DeepLabV3</td><td>16</td><td>21</td><td>6</td><td>6</td></tr></table>

![](images/4d68d83fb38af20bfbeffad6b03ba46c53f9194aaa0b7571c8af28ed3e772bfd.jpg)

<details>
<summary>line</summary>

| Category | Value |
| --- | --- |
| SemSeg/mIoU | 0.72 |
| SemSeg/pixAcc | 0.927 |
| Depth/AbsErr | 0.012 |
| Depth/RelErr | 30 |
</details>

(a) SegNet

![](images/3a83784a282d862220db1eb7c3cc66cc6aa82c166c121e5edf5a7da24fed36b8.jpg)

<details>
<summary>line</summary>

| Metric | Value |
| --- | --- |
| SemSeg/mIoU | 0.73 |
| SemSeg/pixAcc | 0.93 |
| Depth/AbsErr | 0.01 |
| Depth/RelErr | 30 |
</details>

(b) DeepLabV3  
Fig. 2: Parallel coordinate plot over all experiments on Cityscapes. We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation. We observe that Adam clearly outperforms the usage of SGD+mom.

Experiments on Cityscapes and NYUv2 We test the efectiveness of Adam and its role as a confounder in common MTL datasets for various MTO methods.

Approach: We compare Adam and SGD+mom in combination with any MTO method from equal weighting (EW), uncertainty weighting (UW) [21], random loss weighting (RLW) [28], PCGrad [56], CAGrad [31], IMTL [32] and Aligned-MTL [45], for which we used the implementation from [29], as well as MTL-IO [41] and AdaTask [52]. We distinguish between any combination of dataset {Cityscapes [9], NYUv2 [39]} and network architecture {SegNet [1], DeepLabV3 [5]}. We run experiments for ten diferent initial learning rates from [0.5, 0.1, 0.05, ..., 0.00001] and select the best one w.r.t. to the validation performance. More details are described in App. A3.2. As diferent models and parameter setups can show preference towards diferent tasks and metrics, we are interested in those models which are Pareto optimal (PO).

Results: We observe over all experimental setups that Adam performs favorably over SGD+mom (Tab. 2). This especially holds true for experiments on Cityscapes where the Pareto front for both network architectures only consists of Adam-based models. Moreover, an even larger number of Adam-based models is not dominated by any model trained with SGD+mom (PO w.r.t. SGD). For

NYUv2, Adam still performs stronger but SGD+mom. also occasionally delivers a PO result. For the individual metrics, the predominance of Adam is further visualized in a parallel coordinate plot in Figs. 2 and A4. Bold lines indicate the overall Pareto optimal experiments (PO full).

In App. A4, we further report best $\varDelta _ { m }$ results for common MTO methods in combination with Adam or SGD+mom (Tabs. A4 to A7). Again, Adam boosts the overall performance across methods. Furthermore, when comparing the ranking of MTO methods w.r.t. the $\varDelta _ { m }$ metric, we see that the order can change based on the choice of optimizer, e.g., for Cityscapes with SegNet the best method with Adam is UW but with SGD+mom it is CAGrad. This underlines the importance of the choice of optimizer as a confounder in the experimental setup. Noteworthy, EW with Adam yields Pareto optimal results in three of the four setups (cf. Tab. A2) and is not dominated by any specialized MTO method trained in combination with SGD+mom for all dataset and network combinations. This supports claims questioning the efectiveness of specific MTO methods [25, 50]. Nonetheless, looking at the $\varDelta _ { m }$ metric and individual metrics, we see that sometimes with a small relative performance drop on one metric, significant gains on another metric can be achieved (e.g., Cityscapes+sem.seg. and depth for UW vs EW).

Conclusion: Not only a well-tuned learning rate but also the optimizer is crucial for MTL performance. In a fair and extensive experimental comparison, we were able to show that Adam shows superior performance in MTL setup compared to SGD+mom.

The reasonable efectiveness of Adam in the context of uncertainty weighting We show that Adam’s mechanism to estimate a parameter-specific learning rate is partially loss-scale invariant and hypothesize that this could contribute to Adams efectiveness in MTL. We demonstrate this partial invariance theoretically and empirically. Furthermore, a full loss-scale invariance can also be shown under mild assumptions for UW [21], which is among the most prevalent loss weighting method in the literature, and related similar variant [27].

The loss-scale invariance of UW can be shown by assuming an optimal solution for the σ values similar to [23]. This assumption is mild as this is a 1-dimensional convex optimization problem for each σ. The invariance can be demonstrated by inserting the analytical solution starting from UW. For example, assuming a Laplacian distribution (this can be shown for other distributions as well), we have 1

$$
\min _ {\sigma_ {t}} \frac {1}{\sigma_ {t}} \mathcal {L} _ {t} + \log \sigma_ {t} \Rightarrow \sigma_ {t} = \mathcal {L} _ {t} \tag {1}
$$

The left hand side shows the typical form of UW, as shown for a Gaussian in [21, eq.(5)]. Here, $\mathcal { L } _ { t }$ is a task-specific loss and $\sigma _ { t }$ is a scalar parameter that is usually learned. Plugging back the optimal solution for $\sigma _ { t } .$ , we get

$$
\mathcal {L} = \sum_ {t} \frac {\mathcal {L} _ {t}}{s g [ \mathcal {L} _ {t} ]} + c, \tag {2}
$$

where $s g$ is the stop-gradient operator and c is a constant that can be omitted during optimization. Given this, we directly see the invariance w.r.t. loss-scalings. For instance, with ${ \mathcal L } _ { 1 } \to \alpha _ { 1 } { \mathcal L } _ { 1 }$ and ${ \mathcal L } _ { 2 } \to \alpha _ { 2 } { \mathcal L } _ { 2 }$ , the derivative of the total loss $\mathcal { L }$ remains unchanged. As this invariance is shown on the loss-level, it holds for all gradient updates w.r.t. the head and backbone. Intuitively, this could explain why UW performs strongly in the context of various loss scalings such as measuring depth in centimeters or meters. Further details, are in App. A1.

Similarly, for Adam, we can prove a partial scale invariance of losses in MTL that holds for the parameters of network heads. As before, we assume a hydralike network architecture with a shared backbone and task-specific heads. We start with the parameter-update rule from Adam and scale the corresponding losses ${ \mathcal { L } } _ { t } \to \alpha _ { t } { \mathcal { L } } _ { t }$ . When only considering the parameters of the corresponding heads $\psi _ { t }$ , the scalings $\alpha _ { t }$ cancel out

$$
\psi_ {t, i} = \psi_ {t, i - 1} - \frac {\gamma}{\sqrt {\alpha_ {t} ^ {2} \hat {v _ {t} ^ {\prime}}}} \alpha_ {t} \hat {m _ {t} ^ {\prime}}. \tag {3}
$$

Thus, for the network heads, we see a similar efect as for optimal UW that diferent scalings do not impact the network update. However, this does not hold for the backbone. The full derivation is shown in App. A1. We confirm empirically in a handcrafted loss-scaling experiment in App. A2 and Figs. A1 and A2 that SGD does not ofer any scaling invariance, whereas Adam involves the invariance property for the heads. The optimal UW demonstrates a scaling invariance for the heads and the backbone.

We would like to note that our derivation for Adam is only valid for constant $\alpha _ { t } , \ \mathrm { e . g . }$ , measuring depth in diferent units or unitary weightings [25, 50]. In case of dynamic loss weights that are not constant (e.g., UW), the weights do not cancel out fully due to the accumulation of gradient histories within Adam. Nonetheless, this has profound implications for loss weighting methods that are used in conjunction with Adam. For instance, when turning of the history within Adam (by setting $\beta _ { 1 , 2 } ~ = ~ 0 )$ and having a fixed backbone, all loss weighting methods, such as UW, RLW, and others, become equivalent to equal weighting.

Additional ablations to our previous experiments suppport the relevance of invariance in MTL (cf. Tabs. A4 to A7). First, we compare to signSGD+mom [3] which only updates on the sign of gradients and is therefore trivially scaleinvariant in the heads. In a direct comparison with SGD+mom, we observe a superiority of signSGD for a majority of tested setups. Next, we applied taskspecific Adam optimizers as in AdaTask [52] for a full loss-scale invariance and to allow an estimate of task-specific momentum and squared gradient accumulation. This is Pareto dominant over plain Adam+EW in almost all cases and significantly improves the $\varDelta _ { m }$ metric.

Conclusion: In the context of MTL, we derive and measure a full loss-scale invariance for an optimal UW and a partial invariance for Adam. This partial invariance does not hold for SGD+mom and could explain, among other properties, the efectiveness of Adam in MTL. Furthermore, when comparing diferent loss weighting methods, it is crucial to be aware of the influence of the optimizer.

## 4.2 Investigating gradient conflicts between tasks and samples

Examined paradigm: The field of MTL strongly focuses on resolving conflicts between tasks, especially from a perspective of gradient conflicts [20, 31, 56]. In computer vision, tasks are often defined on a conceptional level such as segmentation and depth (Cityscapes), or recognizing multiple attributes (CelebA). However, in principle, conflicts can not only occur between tasks but also between samples within a task.

We argue that in an extreme case, even recognizing a single cat in multiple images could be considered MTL. For instance, in one image, the cat could be hiding behind a plant and only revealing its eyes, requiring a neural network to recognize the cat solely based on the eyes. In other images, the cat might only reveal its paws, front of a bright window, or might be tired and curled up into a furry ball because we took so many pictures. This would require a paw, shape or fur classifier. Thus, a neural network is required to recognize multiple attributes to reliably recognize our cat. We

note that this is conceptually similar to the commonly [44, 56] considered MTL dataset CelebA [35] which requires attribute detection such as wavy hair, mustache or hat, but within one image.

Motivated by this example, we would like to quantify inter-task and intersample conflicts in common datasets from a perspective of the MTO literature, which inspects gradient conflicts in neural networks. In particular, we challenge the sole focus on inter-task gradients conflicts in MTL. While several works follow the idea of overcoming gradient conflicts in MTL [20, 31, 46], their appearance has only been mildly investigated so far.

Prerequisite: We compare gradients w.r.t. network weights for diferent tasks t and samples $\mathbf { x } _ { i }$ . The alignment of two gradients g, g<sup>′</sup> on the shared parameters, e.g., of task a and task $b ,$ is compared with the cosine similarity

$$
S _ {c o s} (\mathbf {g}, \mathbf {g} ^ {\prime}) = \cos (\phi) = \frac {\mathbf {g} \cdot \mathbf {g} ^ {\prime}}{\| \mathbf {g} \| \| \mathbf {g} ^ {\prime} \|}. \tag {4}
$$

Thus, two gradients are in conflict, if their cosine similarity is smaller than zero [56]. In particular, $S _ { c o s }$ is 1/−1 if gradients point in the same/opposite direction and 0 in case of orthogonal directions. The gradient magnitude similarity

$$
S _ {m a g} (\mathbf {g}, \mathbf {g} ^ {\prime}) = \frac {2 \| \mathbf {g} \| _ {2} \cdot \| \mathbf {g} ^ {\prime} \| _ {2}}{\| \mathbf {g} \| _ {2} ^ {2} + \| \mathbf {g} ^ {\prime} \| _ {2} ^ {2}} \tag {5}
$$

as defined in [56], yields values close to 1 for gradients of similar magnitude, or close to 0 for large discrepancies in magnitude. High dissimilarity in both gradient direction and magnitude is presumed to be a common MTL problem.

Approach: During the training on aforementioned datasets, we examine gradient similarity across two diferent setups: (1) between gradients of diferent tasks with respect to a single sample (inter-task), e.g., $\mathbf { g } = \nabla _ { \phi } L _ { 0 } \left( f _ { \theta } ( \mathbf { x } _ { i } ) \right)$ and $\mathbf { g } ^ { \prime } = \nabla _ { \phi } L _ { 1 } \left( f _ { \pmb { \theta } } ( \mathbf { x } _ { i } ) \right)$ ; and (2) between gradients corresponding to the same task but diferent samples within a batch (inter-sample), e $\mathrm { . g . , } \ \mathbf { g } = \nabla _ { \phi } L _ { t } \left( f _ { \theta } ( \mathbf { x } _ { 0 } ) \right)$ and $\mathbf { g } ^ { \prime } = \nabla _ { \phi } L _ { t } \left( f _ { \pmb { \theta } } ( \mathbf { x } _ { 1 } ) \right)$ . For both setups, we compute the gradient cosine similarity and gradient magnitude similarity as well as the ratio of conflicting gradient parameters. We are aware that our comparison between samples and tasks is not direct. Nonetheless, it serves as a coarse indicator to estimate their impact during network training. Implementation details are in App. A3.

![](images/95510daad98161066f136e86081d9daddaa93a551c1bbdd96aad11c82af8cf7b.jpg)  
Fig. 3: High intra-task diversity can mimic MTL.

![](images/aca3b9bbcb288f9f87e0c6110fafecbdb640c11527e34612883125488419b15c.jpg)  
<sup>enu.js</sup>(a) Cityscapes

![](images/11fbf2714004f66ac6026568e63d4b7e4439a37c599ec54358e9fc7c9308d56e.jpg)  
<sup>s</sup>(b) NYUv2

![](images/71379bd39fdf6a442bcf2c2e6dca5d2bd2f1c172b1b3b6536c23d1ac4d9f89a7.jpg)

<details>
<summary>line</summary>

| X | Grad. Cosine Similarity (Blue) | Grad. Cosine Similarity (Red) | Grad. Magnitude Similarity (Blue) | Grad. Magnitude Similarity (Red) | ratio(#Conflicts) (Blue) | ratio(#Conflicts) (Red) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.45 | ~0.45 |
| 20 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.05 | ~0.05 |
| 40 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.02 | ~0.02 |
| 60 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.01 | ~0.01 |
| 80 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.01 | ~0.01 |
</details>

(c) CelebA  
Fig. 4: Gradient similarities and conflicts for diferent datasets and network architectures over training epochs. For each dataset and network combination, we report (from left to right) gradient cosine similarity, gradient magnitude similarity, and the ratio of conflicting gradient parameters w.r.t. gradient pairs corresponding to either inter-samples (fixed task) or inter-tasks (fixed sample). We report mean (solid line), standard deviation (shaded area), upper (97.5%) and lower (2.5%) percentile (dotted line) within an epoch. Overall, the direction conflicts are similar (first / last column), whereas the magnitude diferences are more pronounced in MTL (middle column).

Results: We show the evolvement of the gradient similarity measures over epochs in Fig. 4. Surprisingly, when comparing inter-sample (red line) and intertask (blue line), we find no consistent evidence for gradient alignment conflicts (left column) to be an exclusive problem of having multiple tasks. For instance, for Cityscapes, the variation of gradient alignment is fully encapsulated within the spread we observe in inter-sample variation (task is fixed). For CelebA, the converse seems to be mostly the case. Furthermore, the choice of network architecture and distribution of task-specific and shared parameters (SegNet vs. DeepLabV3) can have a large influence on the spread of the cosine-similarity. Both architectures have roughly a similar number of shared-parameters. However, DeepLabV3 has a higher number of task-specific parameters which seems to reduce the variance in conflicts for both inter-sample and inter-task (row one vs. two). In line with these observations, we found a similar number of conflicting gradient parameters (third column) for both inter-sample and inter-task comparisons among all experiments.

For gradient magnitude similarities (middle column), we observe a clearer pattern. The similarity in magnitudes are continuously (in the mean) less pronounced for the inter-task setup compared to inter-samples (blue line is below red one in all settings). Interestingly, the relative diference between the two setups remains similar over training which justifies the choice of fixed scalar task weightings as done in [50]. Further measures can be found in Figures A5 to A7.

Conclusion: We find that the dificulty of MTL (inter-task and inter-sample) as opposed to STL (inter-sample only) is predominantly due to diferences in gradient magnitudes. Balancing diferent magnitudes is tackled in the literature, e.g., [21, 50] The problem of conflicting gradients has been typically associated with task-specific conflicts [20, 31, 56], here, we find that gradient alignment conflicts can actually be even more pronounced between samples. On the one hand, these observations are along the same lines as findings by Royer et al. [42] who reason that ’correcting conflicting gradients [between tasks] at every training iteration can be superfluous’. On the other hand, gradient-alignment methods in MTL could be considered not only in the context of task-specific conflicts but also for conflicts between samples. Interestingly, previous work has explored the potential benefit of not only learning weights per task but also per sample in the dataset [49]. While our experiments show a relatively high similarity in gradient magnitude across samples and, thus, don’t motivate a sample-wise loss weighting, this could, however, be due to only little disruptive noise within data samples which has been the main motivation of [49].

## 5 Conclusion and outlook

This study aims to enhance our understanding of multi-task learning (MTL) in computer vision, providing valuable insights for future research as well as guidance for implementations of real-world applications.

We show that common optimization methods from single task learning (STL) like the Adam optimizer are efective in MTL problems. Next, we compare gradient conflicts during training between tasks and samples. While gradient magnitudes are a specific problem between tasks (MTL) and thus justify the need for multi-task specific methods for automatic loss weighting, we find the variability in gradient alignment to be similar between samples and tasks. Thus, we encourage a more unified viewpoint in which specific MTO methods are also considered in single-task problems and vice versa.

Beyond our work, we encourage to improve the understanding of challenges and paradigms specific to MTL. For instance, our understanding of task (and sample) specific capacity allocation within a network and how best to tune it to custom requirements, is still not thoroughly understood. Often task-weights are increased to assign more importance to a task which is in contrast to tuning the learning rate per task where a smaller learning rate can be beneficial. Thus, we require further investigations and disentanglement of these two concepts.

## Acknowledgments

We thank Claudia Blaiotta, Martin Rapp, Frank R. Schmidt, Leonhard Hennicke, and Bastian Bischof for their feedback and valuable discussions. Cathrin Elich thanks her supervisors, J¨org St¨uckler and Marc Pollefeys, for enabling the opportunity to pursue an internship during her Ph.D. studies.

The Bosch Group is carbon neutral. Administration, manufacturing and research activities do no longer leave a carbon footprint. This also includes GPU clusters on which the experiments have been performed.

## References

1. Badrinarayanan, V., Kendall, A., Cipolla, R.: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2017)  
2. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European conference on computer vision (ECCV). pp. 456–473 (2018)  
3. Bernstein, J., Wang, Y.X., Azizzadenesheli, K., Anandkumar, A.: signSGD: Compressed optimisation for non-convex problems. In: Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 560–569. PMLR (10–15 Jul 2018)  
4. Caruana, R.: Multitask learning. Machine learning 28, 41–75 (1997)  
5. Chen, L.C., Zhu, Y., Papandreou, G., Schrof, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Computer Vision – ECCV 2018 (2018)  
6. Chen, Z., Badrinarayanan, V., Lee, C., Rabinovich, A.: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: Dy, J.G., Krause, A. (eds.) Proc. of ICML. Proceedings of Machine Learning Research, vol. 80, pp. 793–802. PMLR (2018)  
7. Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., Anguelov, D.: Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)  
8. Chennupati, S., Sistu, G., Yogamani, S., A Rawashdeh, S.: Multinet++: Multistream feature aggregation and geometric loss strategy for multi-task learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)  
9. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3213–3223. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.350  
10. D´esid´eri, J.A.: Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique 350, 313–318 (2012)  
11. Fernando, H.D., Shen, H., Liu, M., Chaudhury, S., Murugesan, K., Chen, T.: Mitigating gradient bias in multi-objective learning: A provably convergent approach.  
In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 (2023)  
12. Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., Finn, C.: Eficiently identifying task groupings for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 27503–27516 (2021)  
13. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)  
14. Guo, M., Haque, A., Huang, D.A., Yeung, S., Fei-Fei, L.: Dynamic task prioritization for multitask learning. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)  
15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 770–778. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.90  
16. Hendrycks, D., Dietterich, T.G.: Benchmarking neural network robustness to common corruptions and perturbations. In: Proc. of ICLR. OpenReview.net (2019)  
17. Hu, Z., Zhao, Z., Yi, X., Yao, T., Hong, L., Sun, Y., Chi, E.: Improving multi-task generalization via regularizing spurious correlation. Advances in Neural Information Processing Systems 35, 11450–11466 (2022)  
18. Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., Madry, A.: Adversarial examples are not bugs, they are features. In: Wallach, H.M., Larochelle, H., Beygelzimer, A., d’Alch´e-Buc, F., Fox, E.B., Garnett, R. (eds.) Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada. pp. 125–136 (2019)  
19. Ishihara, K., Kanervisto, A., Miura, J., Hautam¨aki, V.: Multi-task learning with attention for end-to-end autonomous driving. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2021, virtual, June 19-25, 2021. pp. 2902–2911. Computer Vision Foundation / IEEE (2021). https://doi.org/10.1109/CVPRW53098.2021.00325  
20. Javaloy, A., Valera, I.: Rotograd: Gradient homogenization in multitask learning. In: Proc. of ICLR. OpenReview.net (2022)  
21. Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 7482–7491. IEEE Computer Society (2018). https://doi.org/10.1109/CVPR.2018.00781  
22. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) Proc. of ICLR (2015)  
23. Kirchdorfer, L., Elich, C., Kutsche, S., Stuckenschmidt, H., Schott, L., K¨ohler: Analytical uncertainty-based loss weighting in multi-task learning. In: German Conference on Pattern Recognition (2024)  
24. Klingner, M., Bar, A., Fingscheidt, T.: Improved noise and attack robustness for semantic segmentation by using multi-task training with self-supervised depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 320–321 (2020)  
25. Kurin, V., De Palma, A., Kostrikov, I., Whiteson, S., Kumar, M.P.: In Defense of the Unitary Scalarization for Deep Multi-Task Learning. In: Neural Information Processing Systems (2022)  
26. Lee, D.G.: Fast drivable areas estimation with multi-task learning for real-time autonomous driving assistant. Applied Sciences 11(22), 10713 (2021)  
27. Lin, B., Jiang, W., Ye, F., Zhang, Y., Chen, P., Chen, Y.C., Liu, S., Kwok, J.T.: Dual-balancing for multi-task learning (2023)  
28. Lin, B., YE, F., Zhang, Y., Tsang, I.: Reasonable Efectiveness of Random Weighting: A Litmus Test for Multi-Task Learning. Transactions on Machine Learning Research (2022)  
29. Lin, B., Zhang, Y.: LibMTL: A Python Library for Multi-Task Learning. ArXiv preprint abs/2203.14338 (2022)  
30. Liu, B., Feng, Y., Stone, P., Liu, Q.: Famo: Fast adaptive multitask optimization. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural Information Processing Systems. vol. 36, pp. 57226–57243. Curran Associates, Inc. (2023)  
31. Liu, B., Liu, X., Jin, X., Stone, P., Liu, Q.: Conflict-averse gradient descent for multi-task learning. In: Ranzato, M., Beygelzimer, A., Dauphin, Y.N., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual. pp. 18878–18890 (2021)  
32. Liu, L., Li, Y., Kuang, Z., Xue, J., Chen, Y., Yang, W., Liao, Q., Zhang, W.: Towards impartial multi-task learning. In: Proc. of ICLR. OpenReview.net (2021)  
33. Liu, S., James, S., Davison, A.J., Johns, E.: Auto-Lambda: Disentangling Dynamic Task Relationships. Transactions on Machine Learning Research (2022)  
34. Liu, S., Johns, E., Davison, A.J.: End-to-end multi-task learning with attention. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1871–1880. Computer Vision Foundation / IEEE (2019). https://doi.org/10.1109/CVPR.2019.00197  
35. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015. pp. 3730–3738. IEEE Computer Society (2015). https://doi.org/10.1109/ICCV.2015.425  
36. Maninis, K., Radosavovic, I., Kokkinos, I.: Attentive single-tasking of multiple tasks. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. pp. 1851–1860. Computer Vision Foundation / IEEE (2019). https://doi.org/10.1109/CVPR.2019.00195  
37. Mao, C., Gupta, A., Nitin, V., Ray, B., Song, S., Yang, J., Vondrick, C.: Multitask learning strengthens adversarial robustness. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 158–174. Springer (2020)  
38. Misra, I., Shrivastava, A., Gupta, A., Hebert, M.: Cross-stitch networks for multitask learning. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. pp. 3994–4003. IEEE Computer Society (2016). https://doi.org/10.1109/CVPR.2016.433  
39. Nathan Silberman, Derek Hoiem, P.K., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: ECCV (2012)  
40. Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., Fetaya, E.: Multi-task learning as a bargaining game. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesv´ari, C., Niu, G., Sabato, S. (eds.) International Conference on  
Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA. Proceedings of Machine Learning Research, vol. 162, pp. 16428–16446. PMLR (2022)  
41. Pascal, L., Michiardi, P., Bost, X., Huet, B., Zuluaga, M.A.: Improved optimization strategies for deep multi-task networks. ArXiv preprint abs/2109.11678 (2021)  
42. Royer, A., Blankevoort, T., Bejnordi, B.E.: Scalarization for multi-task and multidomain learning at scale. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)  
43. Ruder, S.: An overview of multi-task learning in deep neural networks. ArXiv preprint abs/1706.05098 (2017)  
44. Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. In: Bengio, S., Wallach, H.M., Larochelle, H., Grauman, K., Cesa-Bianchi, N., Garnett, R. (eds.) Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada. pp. 525–536 (2018)  
45. Senushkin, D., Patakin, N., Kuznetsov, A., Konushin, A.: Independent component alignment for multi-task learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE (2023)  
46. Shi, G., Li, Q., Zhang, W., Chen, J., Wu, X.M.: Recon: Reducing Conflicting Gradients From the Root For Multi-Task Learning. In: The Eleventh International Conference on Learning Representations (2023)  
47. Standley, T., Zamir, A.R., Chen, D., Guibas, L.J., Malik, J., Savarese, S.: Which tasks should be learned together in multi-task learning? In: Proc. of ICML. Proceedings of Machine Learning Research, vol. 119, pp. 9120–9132. PMLR (2020)  
48. Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., Van Gool, L.: Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021). https://doi.org/10.1109/TPAMI.2021.3054719  
49. Vasu, P.K.A., Saxena, S., Tuzel, O.: Instance-level task parameters: A robust multitask weighting framework. CoRR abs/2106.06129 (2021)  
50. Xin, D., Ghorbani, B., Garg, A., Firat, O., Gilmer, J.: Do Current Multi-Task Optimization Methods in Deep Learning Even Help? In: Neural Information Processing Systems (2022)  
51. Xu, D., Ouyang, W., Wang, X., Sebe, N.: Pad-net: Multi-tasks guided predictionand-distillation network for simultaneous depth estimation and scene parsing. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. pp. 675–684. IEEE Computer Society (2018). https://doi.org/10.1109/CVPR.2018.00077  
52. Yang, E., Pan, J., Wang, X., Yu, H., Shen, L., Chen, X., Xiao, L., Jiang, J., Guo, G.: Adatask: A task-aware adaptive learning rate approach to multi-task learning. Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 10745–10753 (2023)  
53. Ye, F., Lin, B., Yue, Z., Guo, P., Xiao, Q., Zhang, Y.: Multi-objective meta learning. In: Advances in Neural Information Processing Systems. vol. 34 (2021)  
54. Yeo, T., Kar, O.F., Zamir, A.: Robustness via cross-domain ensembles. In: 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. pp. 12169–12179. IEEE (2021). https://doi.org/10.1109/ICCV48922.2021.01197  
55. Yu, F., Koltun, V., Funkhouser, T.A.: Dilated residual networks. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Hon-  
olulu, HI, USA, July 21-26, 2017. pp. 636–644. IEEE Computer Society (2017). https://doi.org/10.1109/CVPR.2017.75  
56. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)  
57. Zhang, Y., Yang, Q.: A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering 34(12), 5586–5609 (2022). https://doi.org/10.1109/TKDE.2021.3070203  
58. Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017. pp. 6230–6239. IEEE Computer Society (2017). https://doi.org/10.1109/CVPR.2017.660

# Examining Common Paradigms in Multi-Task Learning -Supplementary Material-

## A1 Theoretical insights into multi-task learning dynamics

In this section, we aim to explain the success of the Adam optimizer [22] by relating it to uncertainty weighting [21]. We show partial invariances w.r.t. prior task-weights for the Adam optimizer and full invariances for the uncertainty weighting under mild assumptions. We further show that for SGD + momentum no invariance can be observed. Instead, the loss-weight can be seen as a taskspecific learning rate which is not the case for the Adam optimizer. Previous literature on weighting methods in MTL did not explicitly show how task-weighting methods are afected by diferent optimizers.

## A1.1 Uncertainty weighting (UW): Full loss-scale invariance

In UW [21], the homoscedastic uncertainty<sup>5</sup> $\sigma _ { t }$ to weight task t is learned by gradient descent. However, we can also analytically compute the optimal uncertainty weights in each iteration instead of learning them using gradient descent as done in [23]. The minimization objective depends on the underlying loss function and likelihood. For simplicity, we show the derivation exemplary for the $L _ { 1 }$ loss. It is straight-forward to derive the same for a Gaussian and other distributions. The objective of uncertainty weighting is given as

$$
\min _ {\sigma_ {t}} \frac {1}{\sigma_ {t}} \mathcal {L} _ {t} + \log \sigma_ {t} \tag {6}
$$

with $\mathcal { L } _ { t } ~ = ~ | \boldsymbol { y } - \boldsymbol { f } ^ { W } ( \boldsymbol { x } ) |$ which can be derived from a log likelihood of a Laplace distribution $\begin{array} { r } { p ( y | f ^ { W } ( x ) , \sigma ) = \frac { 1 } { 2 \sigma } e x p ( - \frac { | y - f ^ { W } ( x ) | } { \sigma } ) } \end{array}$ . Taking the derivative and solving for $\sigma _ { t }$ results in an analytically optimal solution:

$$
\frac {\partial}{\partial \sigma_ {t}} \frac {1}{\sigma_ {t}} \mathcal {L} _ {t} + \log \sigma_ {t} = - \frac {1}{\sigma_ {t} ^ {2}} \mathcal {L} _ {t} + \frac {1}{\sigma_ {t}} \tag {7}
$$

$$
- \frac {1}{\sigma_ {t} ^ {2}} \mathcal {L} _ {t} + \frac {1}{\sigma_ {t}} \stackrel {!} {=} 0 \quad \Rightarrow \quad \sigma_ {t} = \mathcal {L} _ {t} \tag {8}
$$

with $\sigma _ { t } > 0$ . As the optimization problem is convex and just one dimensional, assuming an optimal log-sigma is a mild assumption. Plugging the optimal solution back into the original uncertainty weighting, we get

$$
\mathcal {L} = \sum_ {t} \frac {1}{s g [ \mathcal {L} _ {t} ]} \mathcal {L} _ {t} + \log \sqrt {s g [ \mathcal {L} _ {t} ]}, \tag {9}
$$

where we denote sg as the stopgradient operator.

Since there is no gradient for the second part of the loss, it can be simplified such that

$$
\mathcal {L} = \sum_ {t} \frac {\mathcal {L} _ {t}}{s g [ \mathcal {L} _ {t} ]}. \tag {10}
$$

Assuming task-specific weights $\alpha _ { t } ,$ , we get

$$
\begin{array}{l} \mathcal {L} = \sum_ {t} \frac {\alpha_ {t} \mathcal {L} _ {t}}{\alpha_ {t} s g [ \mathcal {L} _ {t} ]} \tag {11} \\ = \sum_ {t} \frac {\mathcal {L} _ {t}}{s g [ \mathcal {L} _ {t} ]} \\ \end{array}
$$

Therefore, the optimal uncertainty weighting is invariant w.r.t. task-specific loss-scalings, as each scaling cancels out.

## A1.2 SGD: No loss-scale invariance and relationship of learning rate and task weights on a gradient level

Unlike for optimal UW, we show that the SGD update rule does not show any invariances and that task-weights are essentially task-specific learning rates. Instead, task-weights and learning rate are interacting hyperparameters and thus cannot be viewed in isolation.

The parameter update rule in neural networks optimized with SGD is

$$
\theta_ {i} = \theta_ {i - 1} - \gamma \frac {\partial}{\partial \theta_ {i - 1}} \mathcal {L}, \tag {12}
$$

where the network parameters in iteration i are defined as $\theta _ { i }$ , γ is the learning rate and $\begin{array} { r } { \mathcal { L } = \sum _ { t } \alpha _ { i } \mathcal { L } _ { t } } \end{array}$

In the case of uniform task weights (EW), $\alpha = \alpha _ { i } \forall i$ , we have

$$
\begin{array}{l} \theta_ {i} = \theta_ {i - 1} - \gamma \frac {\partial}{\partial \theta_ {i - 1}} \sum_ {i} \alpha \mathcal {L} _ {t} \tag {13} \\ = \theta_ {i - 1} - \gamma \alpha \frac {\partial}{\partial \theta_ {i - 1}} \sum_ {i} \mathcal {L} _ {t} \\ \end{array}
$$

Here, task weight and learning rate are interchangeable. In particular, increasing the weight α by a constant factor c has the same efect as increasing the learning rate by a factor c.

In the case of non-uniform task weights $\alpha _ { i } .$ the parameter update is

$$
\begin{array}{l} \theta_ {i} = \theta_ {i - 1} - \gamma \frac {\partial}{\partial \theta_ {i - 1}} \sum_ {i} \alpha_ {t} \mathcal {L} _ {t} \tag {14} \\ = \theta_ {i - 1} - \frac {\partial}{\partial \theta_ {i - 1}} \sum_ {i} \gamma \alpha_ {t} \mathcal {L} _ {t} \\ \end{array}
$$

As the learning rate can be included in the task-specific weight, it follows that task weighting is interchangeable to assigning task-specific learning rates. Tasks with a higher weight $\alpha _ { i }$ have a proportionally higher parameter update step and vice versa.

While this holds for SGD and SGD + momentum, it does not apply to optimizers such as Adam, Adagrad, or RMSProp. We demonstrate this for Adam in the following subsection.

## A1.3 Adam: Partial loss-scale invariance

Similarly to the invariance demonstrated for optimal UW, we derive a partial invariance for Adam. In their work, Kingma and Ba [22] have already shown that the magnitudes of the parameter updates using Adam are invariant to rescaling the gradients. Our novelty lies in demonstrating this invariance property in the context of MTL and its impact on diferent MTO methods. For Adam, we claim that the magnitude of task-specific weights only afects the backbone and cancels out for the heads.

We consider the standard MTL model setting with a shared backbone and task-specific heads. In this analysis, we assume a frozen backbone and only look at the task-specific parameters $\psi _ { t }$ of task t whose loss $\mathcal { L } _ { t }$ is scaled by $\alpha _ { t } .$ , such that $\mathcal { L } _ { t } \to \alpha _ { t } \mathcal { L } _ { t }$ . The parameter update of one head is independent of the other heads as the derivative of the losses w.r.t. the other tasks is 0:

$$
\frac {\partial}{\partial \psi_ {t , i - 1}} \mathcal {L} _ {j} = 0 \text {for} t \neq j. \tag {15}
$$

The general update rule for parameters ψ at time step i using Adam is

$$
\psi_ {i} = \psi_ {i - 1} - \frac {\gamma}{\sqrt {\hat {v _ {i}}} + \epsilon} \hat {m} _ {i}, \tag {16}
$$

where $m _ { i } = \beta _ { 1 } m _ { i - 1 } + ( 1 - \beta _ { 1 } ) g _ { i }$ and $v _ { i } = \beta _ { 2 } v _ { i - 1 } + ( 1 - \beta _ { 2 } ) g _ { i } ^ { 2 }$ . To counteract the bias towards $0 ,$ the moments are corrected as $\begin{array} { r } { \hat { m } _ { i } = \frac { m _ { i } } { 1 - \beta _ { 1 } ^ { i } } } \end{array}$ and $\begin{array} { r } { \hat { v _ { i } } = \frac { v _ { i } } { 1 - \beta _ { 2 } ^ { i } } } \end{array}$

For task-specific parameters $\psi _ { t }$ , task weights $\alpha _ { t }$ linearly scale the first moment $m _ { t , \ast }$ i

$$
\begin{array}{l} m _ {t, i} = \beta_ {1} m _ {t, i - 1} + (1 - \beta_ {1}) g _ {t, i} \\ = \beta_ {1} m _ {t, i - 1} + (1 - \beta_ {1}) \frac {\partial}{\partial \psi_ {t , i - 1}} \alpha_ {t} \mathcal {L} _ {t} \tag {17} \\ = \beta_ {1} m _ {t, i - 1} + (1 - \beta_ {1}) \alpha_ {t} \frac {\partial}{\partial \psi_ {t , i - 1}} \mathcal {L} _ {t} \\ = \beta_ {1} m _ {t, i - 1} + (1 - \beta_ {1}) \alpha_ {t} g _ {t, i} ^ {\prime} \\ \end{array}
$$

and quadratically scale the second moment $v _ { t , \cdot }$ i

$$
\begin{array}{l} v _ {t, i} = \beta_ {2} v _ {t, i - 1} + (1 - \beta_ {2}) g _ {t, i} ^ {2} \\ = \beta_ {2} v _ {t, i - 1} + \left(1 - \beta_ {2}\right) \left(\frac {\partial}{\partial \psi_ {t , i - 1}} \alpha_ {t} \mathcal {L} _ {t}\right) ^ {2} \tag {18} \\ = \beta_ {2} v _ {t, i - 1} + (1 - \beta_ {2}) \alpha_ {t} ^ {2} (\frac {\partial}{\partial \psi_ {t , i - 1}} \mathcal {L} _ {t}) ^ {2} \\ = \beta_ {2} v _ {t, i - 1} + (1 - \beta_ {2}) \alpha_ {t} ^ {2} g _ {t, i} ^ {\prime 2}, \\ \end{array}
$$

where $g ^ { \prime } { } _ { t , i }$ is the gradient of the unscaled loss $\scriptstyle { \mathcal { L } } _ { t }$ w.r.t. the task-specific parameters for task t. As this holds for iteration i and because we have $m _ { t , 1 } = \alpha g _ { ~ t , 1 } ^ { \prime } + 0$ respectively $v _ { t , 1 } = \alpha _ { t } ^ { 2 } { g ^ { \prime } } _ { t , 1 } ^ { 2 } + 0$ with $m _ { t , 0 } = 0 , v _ { t , 0 } = 0$ at the first iteration, this holds for any iteration step. We can thus rewrite $\hat { m } _ { t , i } = \alpha _ { t } \hat { m } _ { t , i } ^ { \prime }$ and $\hat { v } _ { t , i } = \alpha _ { t } ^ { 2 } \hat { v ^ { \prime } } _ { t , i }$

Plugging this back into the update rule, we get

$$
\begin{array}{l} \psi_ {t, i} = \psi_ {t, i - 1} - \frac {\gamma}{\sqrt {\hat {v} _ {t} i}} \hat {m} _ {t, i} \\ = \psi_ {t, i - 1} - \frac {\gamma}{\sqrt {\alpha_ {t} ^ {2} \hat {v} _ {t , i} ^ {\prime}}} \alpha_ {t} ^ {\prime} \hat {m} _ {t, i} ^ {\prime} \tag {19} \\ \end{array}
$$

where the loss-scaling $\alpha _ { t }$ cancels out. Therefore, the parameters of the taskspecific heads are invariant to loss-scalings using Adam.

This partial invariance is a highly desired property as there is a fundamental trade-of between tuning the learning rate and manual task weights. Given Adams invariance for the head, the weighting only afects the backbone. Thus the learning rate can be set for the parameters of the head independent of the loss weights. With the loss weights, we can prioritize tasks in the backbone and therefore walk along the Pareto front as empirically shown by [50].

The invariance, however, does not hold anymore when the backbone parameters $\theta$ are updated as well. As we have

$$
\begin{array}{l} m _ {i} = \beta_ {1} m _ {i - 1} + \left(1 - \beta_ {1}\right) \frac {\partial}{\partial \theta_ {i - 1}} \sum_ {t} \alpha_ {t} \mathcal {L} _ {t} \tag {20} \\ = \beta_ {1} m _ {i - 1} + (1 - \beta_ {1}) \sum_ {t} \alpha_ {t} g _ {t, i} ^ {\prime} \\ \end{array}
$$

and

$$
\begin{array}{l} v _ {i} = \beta_ {1} v _ {i - 1} + (1 - \beta_ {1}) \left(\frac {\partial}{\partial \theta_ {i - 1}} \sum_ {t} \alpha_ {t} \mathcal {L} _ {t}\right) ^ {2} \tag {21} \\ = \beta_ {1} v _ {i - 1} + (1 - \beta_ {1}) (\sum_ {t} \alpha_ {t} g _ {t, i} ^ {\prime}) ^ {2} \\ \end{array}
$$

we conclude that the task weights $a _ { t }$ linearly afect the first moment $m _ { i }$ , while having a quadratic efect on the update of the second moment $v _ { i }$

Note that for both task-heads only as well as the backbone, we have a full invariance in case of independent optimizers, $\mathrm { e . g . }$ , one Adam optimizer per task similar to [41,52]. However, naive implementations scale poorly (in terms of computational complexity) with the number of tasks here.

In the following experiments, we provide empirical evidence for our finding that a) Adam ofers loss-scale invariance for the parameters of the task-specific heads, and b) Adam ofers loss-scale invariance for all network parameters (backbone and heads) if $\beta _ { 1 , 2 } = 0$

## A2 Empirical Confirmation of scale invariances in Adam and Optimal Uncertainty Weighting

In the prior section, we derived theoretical results for loss-scale (partial) invariance within multi-task learning for the Adam optimizer and uncertainty weighting. In this section, we confirm this invariance empirically with a toy task.

Experimental Setup We consider a two-task toy experiment in which we look at the gradient magnitudes with diferent combinations of Adam, SGD, EW, optimal uncertainty weighting (UW-O), and loss-scalings. To generate the data, we sample scalar input values from a uniform distribution; the outputs are just scalings of the input. We apply a simple neural network which consists of a shared backbone (two layers with LeakyReLU as non-linearity and 20 neurons per hidden layer) and two heads for the two tasks, each consisting again of two layers. Both task measure the depth but in diferent units using the $L _ { 1 }$ -loss.

We provide two settings: In the first one, depth is measured on the same scale. In the second setting, one depth loss is scaled by 10x (e.g., measured in cm instead of deci-meters) and one other loss is scaled by 0.1 (e.g., measured in meters instead of deci-meters). For each setting, we test various combinations of loss weighting and optimizer combinations.

The 8 diferent experiments are:

– EW using SGD  
– EW using SGD with scalings $1 0 \cdot L _ { s e g } , 0 . 0 1 \cdot L _ { d e p }$

– EW using Adam

– EW using Adam with scalings $1 0 \cdot L _ { s e g }$ and $0 . 0 1 \cdot L _ { d e p }$

– UW-O) using SGD

– UW-O) using SGD with scalings $1 0 \cdot L _ { s e g }$ and $0 . 0 1 \cdot L _ { d e p }$

– EW using separate Adam optimizers per task

– EW using separate Adam optimizers per task with scalings $1 0 { \cdot } L _ { s e g } , 0 { . } 0 1 { \cdot } L _ { d e p }$

To better control for diferent factors of influence, we first perform the first 6 of the listed experiments with a fixed backbone, i.e., we do not update the parameters in the backbone but only in the heads. Afterward, we show all 8 experiments trained with a network where all parameters (including the backbone) are updated. This allows us to verify if our theoretical derivations regarding the (partial) loss-scaling invariance of Adam and UW-O also hold in practice, and compare this to the SGD optimizer.

Note that we only care about the invariance and did not tune any hyperparameters for performance.

Results for fixed backbone Figure A1 shows the losses, the scaled losses (by loss weighting method), the gradient magnitudes as well as the gradient update magnitudes for both heads along the 100 epochs of training with a fixed backbone. Regarding SGD, we can observe that the equal weighting experiment difers from its scaled variant along all 8 dimensions. This is because SGD does not ofer any loss-scaling invariance. As expected, at the beginning of the training the gradient update magnitude of the first depth head parameters with the scaled loss (dotted line) is by a factor of 10 higher than the unscaled (solid line) one. The same efect applies to the gradient update magnitude of the second depth head parameters, but with a factor of 0.01.

In contrast, Adam is loss-scale invariant. We can observe that the unscaled (solid line) and the scaled version (dotted line) have equal gradient update magnitudes in the last row. Note that practically due to an epsilon = 10<sup>−8</sup> parameter in the denominator and float precision a slight divergence would occur with larger number of epochs. This result confirms our theoretical finding in equation 19. We skip the experiment of separated Adam optimizers per task because it would be equivalent to this version given a fixed backbone.

Lastly, we want to investigate the invariance properties of UW-O. We compare the scaled (dotted line) and unscaled (solid line) version of UW-O with the SGD optimizer. As expected, the gradients, as well as the gradient updates, match in both heads.

In the following, let’s investigate whether the observed results still hold if we also consider the update of the backbone parameters.

Results for free backbone Figure A2 shows the scaled losses, the gradient magnitudes as well as the gradient update magnitudes in the backbone and the depth heads along the 100 epochs of training with a free backbone. Again, the loss-scalings afect the gradient magnitudes using SGD. This applies to both backbone and heads.

When looking at the Adam experiments, we can observe that it is partly loss-scale invariant by looking at the first iteration in the heads. However, due to diferent updates in the backbone, the networks behave diferent in both settings (scaled and unscaled loses). Furthermore, when implementing task-specific optimizers, we can observe that not only the gradient update magnitudes in the task heads, but also in the backbone match between the scaled (dotted line) and the unscaled (solid line) variant. Thus, all network parameters are invariant to loss-scalings when using separate Adam optimizers. This confirms our theoretical results.

Along the lines of our theoretical findings, we can observe that UW-O ofers scaling-invariance across the whole network as the gradients as well as the gradient updates match among the two variants in the backbone and in both heads. This empirical observation matches our theoretical derivation in equation 11.

![](images/5f59e85229f2614d0c774884afb51858b39fd8060bda78aad73de3551163438e.jpg)  
Fig. A1: Invariances within the neural network for a frozen backbone. Comparing the efect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the the two task heads and keep the backbone frozen. While SGD does not ofer any loss-scaling invariance, Adam makes the gradient updates of the head parameters invariant to scales confirming our derivation (red lines overlap in lowest row). Equivalently, for UW-O we also observe the theoretically derived invariances (green lines overlap in lowest row)

![](images/c77b6fb7d54b2eddfb7ab2daa7250fd27037cb6a402b4023c338976f9261948b.jpg)  
Fig. A2: Invariances within the neural network for a learnable backbone. Comparing the efect of loss-scalings in a toy experiment with two tasks. For each optimizer and loss weighting combination, we run two settings with a) loss L1 and loss L2 are equally weighted or b) L1 is scaled by 10x and L2 by 0.1. For each setting, we measure the SGD + momentum and Adam optimizer with no post weighting (EW) and SGD + momentum with optimimal uncertainty weighting. Additionally, we implement independent Adam optimizer per task. We show the scaled losses, gradient magnitudes, and gradient update magnitudes in the backbone(first row) and the the two task heads (2nd and 3rd row). Neither Adam, nor SGD show invariances if the backbone is trained as well. UW-O is still invriant (green lines are overlapping). We revoke Adam’s inveriance by implementing separate optimizers per task (lowerst black lines are overlapping).

## A3 Implementation Details

In this section, we explain the applied settings used for the reported experiments in more detail. In particular, we describe the handling of the diferent datasets in App. A3.1 and provide further information on the applied training procedures in App. A3.2. Our chosen experimental setups are designed to follow previous work and mainly inspirited by [29, 31, 50]. However, we found that the experimental setup would vary widely across diferent works in the field of multi-task learning as can be seen in Table A1. We use a uniform setup for each dataset independent of the choice of network and MTO.

## A3.1 Datasets

Cityscapes [9] We make use of the oficial split of the dataset which consists of 2975 training and 500 validation scenes. Similar to [50], we denote 595 random samples from the training split as validation data and report test results on the original validation split. We further follow the pre-processing scheme from [34] of re-scaling images to 128x256 pixels and use inverse depth labels. During training, we apply random scaling and cropping for data augmentation<sup>6</sup>. Following previous work [31] for number of epochs and learning rate schedule, we train for 300 epochs and decrease the learning rate by a factor of 0.5 every 100 epochs. The batch size is setto 64, similar to [50]. We only consider a fixed weight decay of $1 0 ^ { - 5 }$ for all datasets and experiments as we found varying this parameter had only little influence in initial experiments.

NYUv2 [39] From the 795 oficial training images we use 159 for our validation split as in [29] and report test performance on the oficial 654 test images. Similar to [34], we re-size the images to 288x384 pixels. Training is run for 200 epochs with a batch size of 8. We apply the same data augmentation and learning rate schedule as for Cityscapes.

CelebA [35] We re-size images to 64x64 pixels as done in [28] and consider the original split of 162,770/19,867/19,962 for training, validation, and testing. We set the batch size to 512, train for 100 epochs, and halve the learning rates every 30 epochs.

## A3.2 Training

Efectiveness of Adam in MTL. All presented results are based on performing early stopping w.r.t. $\varDelta _ { m }$ metric on the validation set. For this, we further trained single-task learning (STL) models for each experiment combination (dataset and network) using the respective network architecture except for the missing head(s). We trained the models using Adam and any learning rate from {0.01, 0.005, ..., 0.00005}. The training was stopped early based on the validation loss. Reported scores in Tabs. A4 to A7 are computed as the mean of the models’ performance that were initialized with the three diferent seeds.

Table A1: Original experiment setup as reported in respective papers. We note a high variation regarding the choice of network, optimizer, and other hyper-parameters among the diferent works.

<table><tr><td colspan="2">Data MTO</td><td>Network</td><td>Optimizer</td><td>learning rate</td><td>weight decay</td><td>batch size</td><td>#train. iterations</td></tr><tr><td rowspan="6">Cityscapes [9]</td><td>UW [21]</td><td>DeepLabV3 [5] with ResNet101 [55]</td><td>SGD + Nesterov updates, Mom.</td><td>init.:  $2.5 \cdot 10^{-3}$ ; polynomial lr decay</td><td> $1 \cdot 10^{-4}$ </td><td>8</td><td>100k iter.</td></tr><tr><td>RLW [28]</td><td>DeepLabV3 [5] with ResNet50 [55]</td><td>Adam</td><td> $1 \cdot 10^{-4}$ </td><td> $1 \cdot 10^{-5}$ </td><td>64</td><td></td></tr><tr><td>IMTL [32]</td><td>ResNet50 [55] + PSPNet [58] heads</td><td>SGD+Mom.</td><td>init.: 0.02; polynomial lr decay</td><td> $1 \cdot 10^{-4}$ </td><td>32</td><td>200 epochs</td></tr><tr><td>PCGrad [56]</td><td>MTAN [34]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr after 40k iter.</td><td>-</td><td>8</td><td>80k iter</td></tr><tr><td>CAGrad [31]</td><td>MTAN [34]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr every 100 epochs</td><td>-</td><td>8</td><td>200 epochs</td></tr><tr><td>AlignedMTL [45]</td><td>MTAN [34] / PSPNet [58]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr every 100 epochs</td><td>-</td><td>8</td><td>200 epochs</td></tr><tr><td rowspan="5">NYuv2 [39]</td><td>RLW [28]</td><td>DeepLabV3 [5] with ResNet50 [55]</td><td>Adam</td><td> $1 \cdot 10^{-4}$ </td><td> $1 \cdot 10^{-5}$ </td><td>8</td><td></td></tr><tr><td>IMTL [32]</td><td>ResNet50 [55] + PSPNet [58] heads</td><td>SGD+Mom.</td><td>init.: 0.03</td><td>-</td><td>48</td><td>200 epochs</td></tr><tr><td>PCGrad [56]</td><td>MTAN [34]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr after 40k iter.</td><td>-</td><td>2</td><td>80k iter</td></tr><tr><td>CAGrad [31]</td><td>MTAN [34]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr after 100 epochs</td><td>-</td><td>2</td><td>200 epochs</td></tr><tr><td>AlignedMTL [45]</td><td>MTAN [34] / PSPNet [58]</td><td>Adam</td><td>init.:  $1 \cdot 10-4$ ; halve lr after 100 epochs</td><td>-</td><td>2</td><td>200 epochs</td></tr><tr><td rowspan="3">CelebA [35]</td><td>RLW [28]</td><td>ResNet17 [15] + lin. classifier</td><td>Adam</td><td> $1 \cdot 10^{-3}$ </td><td>-</td><td>512</td><td></td></tr><tr><td>IMTL [32]</td><td>ResNet17 [15] + lin. classifier</td><td>Adam</td><td>0.003</td><td>-</td><td>256</td><td>100 epochs</td></tr><tr><td>PCGrad [56]</td><td>ResNet17 [15] + lin. classifier</td><td>Adam</td><td>init. from  $\{10^{-4}, ..., 5 \cdot 10^{-2}\}$ ; halve lr every 30 epochs</td><td></td><td>256</td><td>100 epochs</td></tr></table>

Our implementation for all experiments is based on the LibMTL library [29].

Gradient Similarity. Our gradient similarity experiments were conducted on the best performing hyper-parameter configuration for EW from the previous extensive evaluation. Over the full training, gradient similarity measures are computed every five iteration steps and summarized per epoch. To make the computation efort more feasible in case of settings with large batch size or high number of tasks, we randomly select eight samples or tasks respectively and consider corresponding gradients in these cases.

## A3.3 Loss functions

Cityscapes For the task of semantic segmentation, we employ a pixel-wise crossentropy loss: C

$$
\mathcal {L} _ {C E} = \sum_ {c = 1} ^ {C} y ^ {c} \cdot \log (p ^ {c}) \tag {22}
$$

where C is the number of classes, $y ^ { c } \in \{ 0 , 1 \}$ indicates the ground truth class, and $p ^ { c }$ is the predicted probability for class c which results from computing the softmax for output logits $\begin{array} { r } { z ^ { c } , p ^ { c } = \frac { \exp ( z ^ { c } ) } { \sum _ { c = 1 } ^ { C } \exp ( z ^ { c } ) } } \end{array}$ . This loss is averaged over the image.

For depth estimation, we utilize the $L _ { 1 }$ loss:

$$
\mathcal {L} _ {\text {depth}} = \| \boldsymbol {y} - \hat {\boldsymbol {y}} \| _ {1} \tag {23}
$$

where $\mathbf { \mu } _ { y , \hat { \mathbf { \mu } } }$ indicate ground truth and prediction, respectively. Pixels with invalid depth value in the ground truth data are ignored. It is noteworthy that these two types of losses are not balanced when used directly without modification.

NYUv2 The tasks of semantic segmentation and depth estimation are trained using the same loss functions as described for Cityscapes. In addition, we compute the cosine loss on the (normalized) surface normal maps:

$$
\mathcal {L} _ {\text {normal}} = 1 - \cos \theta = 1 - \frac {\boldsymbol {y} \cdot \hat {\boldsymbol {y}}}{\| \boldsymbol {y} \| \| \hat {\boldsymbol {y}} \|} \tag {24}
$$

where $\mathbf { \mu } _ { y , \hat { \mathbf { \mu } } }$ are the ground truth and predicted normal maps. Similar to the Cityscapes setup, the combination of these loss functions is not balanced per default.

CelebA To learn to predict multiple attributes, we use a binary cross entropy loss for the individual classes:

$$
\mathcal {L} _ {C E, b i n} = - [ y \log (p) + (1 - y) \log (1 - p) ] \tag {25}
$$

Although all these losses have a similar scale, their impact varies based on the dificulty of the individual tasks and the number of available samples displaying the respective attribute.

## A3.4 Evaluation criteria

In this study, we primarily focus on Pareto optimal solution to acknowledge that diferent configurations may lead to varying preferences for the learned tasks. However, it is important to note that not every point on the Pareto front is relevant in practice, especially when one metric significantly dominates while others are close to chance level. Moreover, specific real-world applications can have a stronger, pre-defined prioritization of one or a few sub-tasks which requires a relative weighting of the tasks’ performances.

Additionally, we further employ the $\varDelta _ { m }$ metric which ofers a simple option to directly compare the performance of two models using a single scalar. This metric further indicates the relative performance compared to the single-tasks models.

Note that for our initial toy task experiment (Sec. 4.1), we consider the original setting from [31] which optimzes the global minimum of the two loss functions.

## A4 Additional results on comparison between Adam and SGD

We present additional evaluation results for our comparison between optimizers for MTL. In Figure A3, we compare the $\varDelta _ { m }$ metric performance between the usage of Adam and SGD+mom. Fig. A4 shows additional parallel coordinate plots for NYUv2 and both choices of networks. In Table A2, we count for each used MTO method the number of experiment runs that are located on the Pareto front w.r.t. each setup. Best performing quantitative results for all MTOs can be found in Tabs. A4 to A7.

We further show extended results on the toy task by Liu et al. [31] for more learning rates in Tab. A3.

![](images/0afe7b9b050d9f47eea78bf15e6e95959319d83b1f371025c6017ec319002e30.jpg)  
Fig. A3: Mean $\varDelta _ { m }$ metric for experiments run on Cityscapes and NYUv2 with SegNet and DeepLabV3. We compare the performance of the best hyperparameter setting for every MTO method using either Adam (left) or SGD+Momentum (right) (lower is better). Every MTO is associated with a diferent line color/style. On Cityscapes, there is a large diference for the $\varDelta _ { m }$ score for Adam compared to SGD+Momentum, especially for UW, IMTL, and CAGrad. Therefore, for this setup, the result depends more on the optimizer than on the MTO method. On the NYUv2 dataset this observation weakens. Adam still achieves the lowest $\varDelta _ { m }$ scores across diferent MTO methods (except for SegNet with UW and IMTL), though, besides chosing Adam, it is also important to select the appropriate MTO method.

![](images/39fd168f340f84edac1d965b2387230218f1adda152e9403adba81536d4476df.jpg)

<details>
<summary>line</summary>

| Category | Value |
| --- | --- |
| SemSeg/mIoU | 0.42 |
| SemSeg/pixAcc | 0.68 |
| Depth/AbsErr | 0.5 |
| Depth/RelErr | 0.2 |
| Normal/Mean | 25 |
| Normal/Median | 18 |
| Normal/<11.25 | 0.3 |
| Normal/<22.5 | 0.6 |
| Normal/<30.0 | 0.7 |
</details>

(a) SegNet

![](images/61eff97100c6a38b811dbb2c39367f548d63d516f6e3b21a30d7aaa919241256.jpg)

<details>
<summary>line</summary>

| Metric | Blue Line Value | Red Line Value |
| :--- | :--- | :--- |
| SemSeg/mIoU | 0.565 | 0.775 |
| SemSeg/pixAcc | 0.35 | 0.35 |
| Depth/AbsErr | 0.14 | 0.138 |
| Depth/RelErr | 21 | 21 |
| Normal/Mean | 14 | 14 |
| Normal/Median | 0.41 | 0.41 |
| Normal/<11.25 | 0.68 | 0.68 |
| Normal/<22.5 | 0.78 | 0.78 |
| Normal/<30.0 | 0.78 | 0.78 |
| SemSeg/mIoU (Blue) | ~0.535 | ~0.555 |
| SemSeg/mIoU (Red) | ~0.535 | ~0.555 |
| SemSeg/pixAcc (Blue) | ~0.535 | ~0.555 |
| SemSeg/pixAcc (Red) | ~0.535 | ~0.555 |
| Depth/AbsErr (Blue) | ~0.56 | ~0.56 |
| Depth/AbsErr (Red) | ~0.56 | ~0.56 |
| Depth/RelErr (Blue) | ~0.56 | ~0.56 |
| Depth/RelErr (Red) | ~0.56 | ~0.56 |
| Normal/Mean (Blue) | ~0.565 | ~0.565 |
| Normal/Mean (Red) | ~0.565 | ~0.565 |
| Normal/Median (Blue) | ~0.56 | ~0.56 |
| Normal/Median (Red) | ~0.56 | ~0.56 |
| Normal/<11.25 (Blue) | ~0.56 | ~0.56 |
| Normal/<11.25 (Red) | ~0.56 | ~0.56 |
| Normal/<22.5 (Blue) | ~0.56 | ~0.56 |
| Normal/<22.5 (Red) | ~0.56 | ~0.56 |
| Normal/<30.0 (Blue) | ~0.56 | ~0.56 |
| Normal/<30.0 (Red) | ~0.56 | ~0.56 |
| SemSeg/mIoU (Blue) | 0.77 | 0.77 |
| SemSeg/mIoU (Red) | 0.77 | 0.77 |
| SemSeg/pixAcc (Blue) | 0.76 | 0.76 |
| SemSeg/pixAcc (Red) | 0.76 | 0.76 |
| Depth/AbsErr (Blue) | 0.37 | 0.37 |
| Depth/AbsErr (Red) | 0.37 | 0.37 |
| Depth/RelErr (Blue) | 0.15 | 0.15 |
| Depth/RelErr (Red) | 0.15 | 0.15 |
| Normal/Mean (Blue) | 23 | 23 |
| Normal/Mean (Red) | 23 | 23 |
| Normal/Median (Blue) | 17 | 17 |
| Normal/Median (Red) | 17 | 17 |
| Normal/<11.25 (Blue) | 0.35 | 0.35 |
| Normal/<11.25 (Red) | 0.35 | 0.35 |
| Normal/<22.5 (Blue) | 0.74 | 0.74 |
| Normal/<22.5 (Red) | 0.74 | 0.74 |
| Normal/<30.0 (Blue) | 0.78 | 0.78 |
| Normal/<30.0 (Red) | 0.78 | 0.78 |
| SemSeg/mIoU (Blue) | N/A | N/A |
| SemSeg/mIoU (Red) | N/A | N/A |
| SemSeg/pixAcc (Blue) | N/A | N/A |
| SemSeg/pixAcc (Red) | N/A | N/A |
| Depth/AbsErr (Blue) | N/A | N/A |
| Depth/AbsErr (Red) | N/A | N/A |
| Depth/RelErr (Blue) | N/A | N/A |
| Depth/RelErr (Red) | N/A | N/A |
| Normal/Mean (Blue) | N/A | N/A |
| Normal/Mean (Red) | N/A | N/A |
| Normal/Median (Blue) | N/A | N/A |
| Normal/Median (Red) | N/A | N/A |
| Normal/<11.25 (Blue) | N/A | N/A |
| Normal/<11.25 (Red) | N/A | N/A |
| Normal/<22.5 (Blue) | N/A | N/A |
| Normal/<22.5 (Red) | N/A | N/A |
| Normal/<30.0 (Blue) | N/A | N/A |
| Normal/<30.0 (Red) | N/A | N/A |
| SemSeg/mIoU (Blue) | N/A | N/A |
| SemSeg/mIoU (Red) | N/A | N/A |
| SemSeg/pixAcc (Blue) | N/A | N/A |
| SemSeg/pixAcc (Red) | N/A | N/A |
| Depth/AbsErr (Blue) | N/A | N/A |
| Depth/AbsErr (Red) | N/A | N/A |
| Depth/Relerr (Blue) | N/A | N/A |
| Depth/Relerr (Red) | N/A | N/A |
| Normal/Mean (Blue) | N/A | N/A |
| Normal/Mean (Red) | N/A | N/A |
| Normal/Median (Blue) | N/A | N/A |
| Normal/Median (Red) | N/A | N/A |
| Normal/<11.25 (Blue) | N/A | N/A |
| Normal/<11.25 (Red) | N/A | — |
| Normal/<22.5 (Blue) | N/A | N/A |
| Normal/<22.5 (Red) | N/A | — |
| Normal/<30.0 (Blue) | N/A | N/A |
| Normal/<30.0 (Red) | N/A | — |

*Note: Values are estimated from gridlines where exact labels are not present.*
</details>

(b) DeepLabV3  
Fig. A4: Parallel coordinate plot over all experiments on NYUv2 We distinguish between experiments using SGD+mom and Adam optimizer. Experiments that reached Pareto front performance are drawn with higher saturation. Similiar to results on Cityscapes in the main paper (Fig. 2) we observe a dominance of Adam albeit, here, we also have some experiments using SGD+Mom. on the overall Pareto front.

Table A2: Count of Pareto optimal experiments for each MTO method. We found no single MTO method to be clearly superior over all combinations of dataset and networks. Total numbers can be compared to Table 2

<table><tr><td>Data</td><td>Network</td><td>Optimizer</td><td>EW</td><td>UW</td><td>RLW</td><td>IMTL</td><td>PCGrad</td><td>CAGrad</td><td>AlignedMTL</td><td>AdaTask</td><td>Total</td></tr><tr><td>Cityscapes</td><td>SegNet</td><td>Adam</td><td>-</td><td>3</td><td>-</td><td>-</td><td>-</td><td>1</td><td>1</td><td>-</td><td>5</td></tr><tr><td>Cityscapes</td><td>SegNet</td><td>SGD+Mom.</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Cityscapes</td><td>DeepLabV3</td><td>Adam</td><td>2</td><td>2</td><td>1</td><td>2</td><td>2</td><td>1</td><td>-</td><td>-</td><td>15</td></tr><tr><td>Cityscapes</td><td>DeepLabV3</td><td>SGD+Mom.</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>NYUv2</td><td>SegNet</td><td>Adam</td><td>1</td><td>2</td><td>-</td><td>1</td><td>1</td><td>1</td><td>3</td><td>2</td><td>11</td></tr><tr><td>NYUv2</td><td>SegNet</td><td>SGD+Mom.</td><td>-</td><td>-</td><td>-</td><td>1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1</td></tr><tr><td>NYUv2</td><td>DeepLabV3</td><td>Adam</td><td>2</td><td>-</td><td>1</td><td>3</td><td>3</td><td>2</td><td>2</td><td>3</td><td>16</td></tr><tr><td>NYUv2</td><td>DeepLabV3</td><td>SGD+Mom.</td><td>-</td><td>1</td><td>-</td><td>-</td><td>-</td><td>3</td><td>2</td><td>-</td><td>6</td></tr></table>

Table A3: Number of iterations after which all seeds in toy task experiment from CAGrad [31] have reached the global minimum for diferent learning rates and optimizer. We show results for additional learning rates compared to the main paper. The maximum iteration number over all three seeds for each MTO method / learning rate / optimizer combination is reported. If not all seeds converged to the global minimum within 100k iteration steps, we denote it as ’-’. In several setups, EW+Adam converges fastest to the global minimum. Especially for small learning rates, CAGrad performs advantageous compared to EW. As reported in previous work, we found that PCGrad often would converge only to some point on the Pareto Front. The best and second best run for each learning rate over all MTO methods are indicated via font type.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="9">learning rate</td><td></td></tr><tr><td>method</td><td>10.0</td><td>5.0</td><td>1.0</td><td>0.5</td><td>0.1</td><td>0.05</td><td>0.01</td><td>0.005</td><td>0.001*</td></tr><tr><td rowspan="3">GD</td><td>EW</td><td>-</td><td>103</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PCGrad</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CAGrad</td><td>644</td><td>-</td><td>213</td><td>621</td><td>8,069</td><td>5,732</td><td>20,418</td><td>34,405</td><td>-</td><td></td></tr><tr><td rowspan="3">Adam</td><td>EW</td><td>26</td><td>37</td><td>22</td><td>58</td><td>709</td><td>2,135</td><td>9,015</td><td>16,005</td><td>-</td><td></td></tr><tr><td>PCGrad</td><td>25</td><td>4,960</td><td>56</td><td>15,741</td><td>34,175</td><td>41,438</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>CAGrad</td><td>27</td><td>30</td><td>32</td><td>106</td><td>802</td><td>7,109</td><td>11,239</td><td>14,323</td><td>57,700</td><td></td></tr></table>

\*LR used for results in [31] with Adam

Table A4: Results for diferent MTO methods and optimizers on Cityscapes [9] using SegNet [1]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. While different MTO methods perform best over the distinct metrics, models trained with Adam outperform those based on SGD+mom in most direct comparisons. On the overall $\varDelta _ { m }$ metric, Adam shows superior performance for all MTO methods, in some cases even with a high margin. Best performance for each metric was also achieved by using Adam.

<table><tr><td rowspan="2">MTO</td><td rowspan="2">Optimizer</td><td rowspan="2">lr</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td rowspan="2">DeltaM ↓</td></tr><tr><td>mIoU ↑</td><td>pixAcc ↑</td><td>AbsErr ↓</td><td>RelErr ↓</td></tr><tr><td>STL</td><td>adam</td><td></td><td>0.7122</td><td>0.9221</td><td>0.0134</td><td>29.88</td><td></td></tr><tr><td>EW</td><td>adam</td><td>0.005</td><td>0.6898</td><td>0.9165</td><td>0.0196</td><td>109.84</td><td> $79.43 \pm 3.68$ </td></tr><tr><td>EW</td><td>signSGD</td><td>0.001</td><td>0.7013</td><td>0.9174</td><td>0.0210</td><td>115.67</td><td> $86.52 \pm 6.06$ </td></tr><tr><td>EW</td><td>sgd</td><td>0.1</td><td>0.6967</td><td>0.9179</td><td>0.0216</td><td>113.82</td><td> $86.24 \pm 1.97$ </td></tr><tr><td>UW</td><td>adam</td><td>0.001</td><td>0.7052</td><td>0.9202</td><td>0.0136</td><td>35.69</td><td> $5.44 \pm 2.38$ </td></tr><tr><td>UW</td><td>signSGD</td><td>0.001</td><td>0.6506</td><td>0.8997</td><td>0.0166</td><td>53.61</td><td> $28.54 \pm 7.83$ </td></tr><tr><td>UW</td><td>sgd</td><td>0.01</td><td>0.6750</td><td>0.9110</td><td>0.0219</td><td>114.67</td><td> $88.39 \pm 1.35$ </td></tr><tr><td>RLW</td><td>adam</td><td>0.001</td><td>0.7013</td><td>0.9196</td><td>0.0197</td><td>103.61</td><td> $73.91 \pm 7.63$ </td></tr><tr><td>RLW</td><td>signSGD</td><td>0.0005</td><td>0.6962</td><td>0.9169</td><td>0.0204</td><td>112.46</td><td> $82.90 \pm 5.32$ </td></tr><tr><td>RLW</td><td>sgd</td><td>0.1</td><td>0.6918</td><td>0.9156</td><td>0.0227</td><td>113.59</td><td> $88.16 \pm 0.67$ </td></tr><tr><td>IMTL</td><td>adam</td><td>0.005</td><td>0.6963</td><td>0.9170</td><td>0.0148</td><td>45.63</td><td> $16.55 \pm 1.52$ </td></tr><tr><td>IMTL</td><td>signSGD</td><td>0.005</td><td>0.6659</td><td>0.9070</td><td>0.0197</td><td>101.90</td><td> $74.03 \pm 14.62$ </td></tr><tr><td>IMTL</td><td>sgd</td><td>0.01</td><td>0.6716</td><td>0.9107</td><td>0.0230</td><td>114.38</td><td> $90.21 \pm 0.74$ </td></tr><tr><td>PCGrad</td><td>adam</td><td>0.01</td><td>0.6770</td><td>0.9135</td><td>0.0226</td><td>103.88</td><td> $80.56 \pm 3.70$ </td></tr><tr><td>PCGrad</td><td>signSGD</td><td>0.001</td><td>0.6929</td><td>0.9170</td><td>0.0210</td><td>113.32</td><td> $84.74 \pm 3.46$ </td></tr><tr><td>PCGrad</td><td>sgd</td><td>0.1</td><td>0.6972</td><td>0.9176</td><td>0.0235</td><td>107.06</td><td> $84.09 \pm 0.93$ </td></tr><tr><td>CAGrad</td><td>adam</td><td>0.001</td><td>0.7088</td><td>0.9208</td><td>0.0162</td><td>66.39</td><td> $35.81 \pm 14.91$ </td></tr><tr><td>CAGrad</td><td>signSGD</td><td>0.001</td><td>0.6883</td><td>0.9138</td><td>0.0182</td><td>107.06</td><td> $74.68 \pm 18.91$ </td></tr><tr><td>CAGrad</td><td>sgd</td><td>0.1</td><td>0.6896</td><td>0.9156</td><td>0.0205</td><td>115.52</td><td> $85.88 \pm 0.31$ </td></tr><tr><td>AlignedMTL</td><td>adam</td><td>0.0005</td><td>0.7164</td><td>0.9246</td><td>0.0154</td><td>64.89</td><td> $32.76 \pm 5.27$ </td></tr><tr><td>AlignedMTL</td><td>signSGD</td><td>0.001</td><td>0.7010</td><td>0.9189</td><td>0.0180</td><td>102.41</td><td> $69.77 \pm 24.91$ </td></tr><tr><td>AlignedMTL</td><td>sgd</td><td>0.1</td><td>0.6684</td><td>0.9103</td><td>0.0224</td><td>113.11</td><td> $88.37 \pm 0.53$ </td></tr><tr><td>Adatask</td><td>adam</td><td>0.0005</td><td>0.7039</td><td>0.9196</td><td>0.0146</td><td>52.17</td><td> $21.28 \pm 0.35$ </td></tr><tr><td>MTL-IO</td><td>sgd</td><td>0.1</td><td>0.6957</td><td>0.9172</td><td>0.0233</td><td>110.19</td><td> $86.46 \pm 1.63$ </td></tr></table>

Table A5: Results for diferent MTO methods and optimizers on Cityscapes [9] using DeepLabV3+ [5]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Adam is Pareto dominant over SGD+mom in a direct pairwise comparison across all MTO methods. Here, we would also like to highlight that MTL can outperform STL as suggested by [4].

<table><tr><td rowspan="2">MTO</td><td rowspan="2">Optimizer</td><td rowspan="2">lr</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td rowspan="2">DeltaM ↓</td></tr><tr><td>mIoU ↑</td><td>pixAcc ↑</td><td>AbsErr ↓</td><td>RelErr ↓</td></tr><tr><td>STL</td><td>adam</td><td></td><td>0.7203</td><td>0.9253</td><td>0.0132</td><td>47.37</td><td></td></tr><tr><td>EW</td><td>adam</td><td>0.001</td><td>0.7247</td><td>0.9268</td><td>0.0128</td><td>47.73</td><td>-0.80 ±0.69</td></tr><tr><td>EW</td><td>signSGD</td><td>0.0005</td><td>0.7188</td><td>0.9249</td><td>0.0134</td><td>52.47</td><td>3.07 ±2.70</td></tr><tr><td>EW</td><td>sgd</td><td>0.05</td><td>0.7100</td><td>0.9217</td><td>0.0174</td><td>120.34</td><td>46.88 ±2.02</td></tr><tr><td>UW</td><td>adam</td><td>0.001</td><td>0.7224</td><td>0.9259</td><td>0.0122</td><td>44.37</td><td>-3.65 ±0.61</td></tr><tr><td>UW</td><td>signSGD</td><td>0.001</td><td>0.7172</td><td>0.9244</td><td>0.0127</td><td>48.31</td><td>-0.35 ±2.23</td></tr><tr><td>UW</td><td>sgd</td><td>0.005</td><td>0.7003</td><td>0.9187</td><td>0.0171</td><td>116.09</td><td>44.47 ±1.91</td></tr><tr><td>RLW</td><td>adam</td><td>0.001</td><td>0.7230</td><td>0.9263</td><td>0.0133</td><td>47.76</td><td>0.26 ±1.08</td></tr><tr><td>RLW</td><td>signSGD</td><td>0.001</td><td>0.7185</td><td>0.9244</td><td>0.0133</td><td>54.75</td><td>4.09 ±0.39</td></tr><tr><td>RLW</td><td>sgd</td><td>0.05</td><td>0.7070</td><td>0.9205</td><td>0.0176</td><td>123.13</td><td>48.84 ±1.79</td></tr><tr><td>IMTL</td><td>adam</td><td>0.001</td><td>0.7226</td><td>0.9259</td><td>0.0121</td><td>45.25</td><td>-3.38 ±0.92</td></tr><tr><td>IMTL</td><td>signSGD</td><td>0.001</td><td>0.7177</td><td>0.9243</td><td>0.0128</td><td>49.61</td><td>0.57 ±3.75</td></tr><tr><td>IMTL</td><td>sgd</td><td>0.005</td><td>0.7027</td><td>0.9192</td><td>0.0185</td><td>122.37</td><td>50.33 ±3.30</td></tr><tr><td>PCGrad</td><td>adam</td><td>0.001</td><td>0.7247</td><td>0.9272</td><td>0.0130</td><td>47.24</td><td>-0.58 ±0.56</td></tr><tr><td>PCGrad</td><td>signSGD</td><td>0.001</td><td>0.7187</td><td>0.9248</td><td>0.0132</td><td>53.35</td><td>3.28 ±1.41</td></tr><tr><td>PCGrad</td><td>sgd</td><td>0.05</td><td>0.7083</td><td>0.9212</td><td>0.0173</td><td>122.52</td><td>47.90 ±3.68</td></tr><tr><td>CAGrad</td><td>adam</td><td>0.001</td><td>0.7245</td><td>0.9264</td><td>0.0124</td><td>45.56</td><td>-2.59 ±0.68</td></tr><tr><td>CAGrad</td><td>signSGD</td><td>0.001</td><td>0.7154</td><td>0.9243</td><td>0.0128</td><td>49.06</td><td>0.37 ±0.41</td></tr><tr><td>CAGrad</td><td>sgd</td><td>0.1</td><td>0.7096</td><td>0.9220</td><td>0.0172</td><td>120.24</td><td>46.58 ±2.21</td></tr><tr><td>AlignedMTL</td><td>adam</td><td>0.0005</td><td>0.7225</td><td>0.9259</td><td>0.0122</td><td>47.70</td><td>-1.79 ±0.70</td></tr><tr><td>AlignedMTL</td><td>signsgd</td><td>0.001</td><td>0.7169</td><td>0.9241</td><td>0.0127</td><td>47.56</td><td>-0.70 ±0.61</td></tr><tr><td>AlignedMTL</td><td>sgd</td><td>0.1</td><td>0.7096</td><td>0.9218</td><td>0.0174</td><td>120.38</td><td>46.94 ±1.19</td></tr><tr><td>Adatask</td><td>adam</td><td>0.001</td><td>0.7234</td><td>0.9260</td><td>0.0124</td><td>47.21</td><td>-1.77 ±0.86</td></tr><tr><td>MTL-IO</td><td>sgd</td><td>0.05</td><td>0.7098</td><td>0.9217</td><td>0.0187</td><td>120.61</td><td>49.57 ±2.28</td></tr></table>

Table A6: Results for diferent MTO methods and optimizers on NYUv2 [39] using SegNet [1]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. Using Adam yields in superior performance in the majority of cases, both when considering the individual metrics and the overall $\varDelta _ { m }$ metric. We note that ∆x $\varDelta _ { m }$ is more efected by the normal task due to the higher number of corresponding metrics as can be observed in the case of UW.

<table><tr><td rowspan="2">MTO</td><td rowspan="2">Optimizer</td><td rowspan="2">lr</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td colspan="5">Normal</td><td rowspan="2">DeltaM↓</td></tr><tr><td colspan="2">mIoU↑pixAcc↑</td><td colspan="2">AbsErr↓RelErr↓</td><td colspan="5">Mean↓Median↓&lt; 11.25 ↑&lt; 22.5 ↑&lt; 30.0 ↑</td></tr><tr><td>STL</td><td>adam</td><td></td><td>0.392</td><td>0.646</td><td>0.607</td><td>0.258</td><td>24.74</td><td>18.49</td><td>0.308</td><td>0.582</td><td>0.700</td><td></td></tr><tr><td>EW</td><td>adam</td><td>0.0001</td><td>0.398</td><td>0.650</td><td>0.530</td><td>0.212</td><td>29.53</td><td>25.02</td><td>0.216</td><td>0.454</td><td>0.584</td><td> $10.08 \pm 2.84$ </td></tr><tr><td>EW</td><td>signSGD</td><td>0.0001</td><td>0.389</td><td>0.647</td><td>0.543</td><td>0.220</td><td>30.18</td><td>25.93</td><td>0.204</td><td>0.438</td><td>0.569</td><td> $12.75 \pm 2.24$ </td></tr><tr><td>EW</td><td>sgd</td><td>0.01</td><td>0.384</td><td>0.644</td><td>0.551</td><td>0.227</td><td>30.12</td><td>25.72</td><td>0.212</td><td>0.443</td><td>0.571</td><td> $12.84 \pm 0.65$ </td></tr><tr><td>- UW</td><td>adam</td><td>0.0001</td><td>0.401</td><td>0.652</td><td>0.522</td><td>0.213</td><td>27.93</td><td>22.80</td><td>0.243</td><td>0.494</td><td>0.623</td><td> $5.42 \pm 1.04$ </td></tr><tr><td>UW</td><td>signSGD</td><td>0.0001</td><td>0.384</td><td>0.634</td><td>0.535</td><td>0.218</td><td>28.36</td><td>23.45</td><td>0.233</td><td>0.482</td><td>0.612</td><td> $8.05 \pm 1.18$ </td></tr><tr><td>UW</td><td>sgd</td><td>0.05</td><td>0.383</td><td>0.642</td><td>0.551</td><td>0.218</td><td>27.07</td><td>21.66</td><td>0.259</td><td>0.516</td><td>0.644</td><td> $4.46 \pm 2.13$ </td></tr><tr><td>- RLW</td><td>adam</td><td>0.0001</td><td>0.390</td><td>0.638</td><td>0.537</td><td>0.218</td><td>30.83</td><td>26.78</td><td>0.196</td><td>0.424</td><td>0.554</td><td> $14.28 \pm 2.38$ </td></tr><tr><td>RLW</td><td>signSGD</td><td>0.0001</td><td>0.375</td><td>0.636</td><td>0.544</td><td>0.226</td><td>29.38</td><td>24.77</td><td>0.217</td><td>0.458</td><td>0.589</td><td> $11.42 \pm 1.02$ </td></tr><tr><td>RLW</td><td>sgd</td><td>0.05</td><td>0.369</td><td>0.633</td><td>0.572</td><td>0.233</td><td>31.21</td><td>27.17</td><td>0.198</td><td>0.420</td><td>0.546</td><td> $16.82 \pm 1.16$ </td></tr><tr><td>- IMTL</td><td>adam</td><td>0.0001</td><td>0.380</td><td>0.644</td><td>0.524</td><td>0.216</td><td>26.35</td><td>20.77</td><td>0.270</td><td>0.534</td><td>0.660</td><td> $2.06 \pm 1.31$ </td></tr><tr><td>IMTL</td><td>signSGD</td><td>0.0001</td><td>0.374</td><td>0.639</td><td>0.527</td><td>0.212</td><td>26.26</td><td>20.65</td><td>0.272</td><td>0.537</td><td>0.663</td><td> $1.97 \pm 0.832$ </td></tr><tr><td>IMTL</td><td>sgd</td><td>0.05</td><td>0.396</td><td>0.656</td><td>0.532</td><td>0.215</td><td>26.16</td><td>20.38</td><td>0.277</td><td>0.542</td><td>0.666</td><td> $0.72 \pm 0.68$ </td></tr><tr><td>- PCGrad</td><td>adam</td><td>0.0001</td><td>0.406</td><td>0.654</td><td>0.529</td><td>0.215</td><td>28.60</td><td>23.75</td><td>0.231</td><td>0.477</td><td>0.606</td><td> $7.39 \pm 0.86$ </td></tr><tr><td>PCGrad</td><td>signSGD</td><td>0.0001</td><td>0.394</td><td>0.649</td><td>0.556</td><td>0.219</td><td>28.98</td><td>24.25</td><td>0.223</td><td>0.467</td><td>0.598</td><td> $9.59 \pm 1.02$ </td></tr><tr><td>PCGrad</td><td>sgd</td><td>0.01</td><td>0.389</td><td>0.644</td><td>0.547</td><td>0.223</td><td>29.76</td><td>25.22</td><td>0.214</td><td>0.451</td><td>0.580</td><td> $11.61 \pm 0.36$ </td></tr><tr><td>- CAGrad</td><td>adam</td><td>0.0001</td><td>0.405</td><td>0.661</td><td>0.527</td><td>0.211</td><td>25.85</td><td>20.02</td><td>0.282</td><td>0.550</td><td>0.674</td><td> $-0.65 \pm 0.96$ </td></tr><tr><td>CAGrad</td><td>signSGD</td><td>0.0001</td><td>0.393</td><td>0.648</td><td>0.546</td><td>0.217</td><td>26.21</td><td>20.58</td><td>0.272</td><td>0.538</td><td>0.665</td><td> $1.67 \pm 0.61$ </td></tr><tr><td>CAGrad</td><td>sgd</td><td>0.05</td><td>0.400</td><td>0.656</td><td>0.547</td><td>0.223</td><td>26.39</td><td>20.78</td><td>0.268</td><td>0.534</td><td>0.662</td><td> $2.06 \pm 0.65$ </td></tr><tr><td>AlignedMTL</td><td>adam</td><td>0.0001</td><td>0.385</td><td>0.648</td><td>0.519</td><td>0.212</td><td>25.51</td><td>19.66</td><td>0.288</td><td>0.557</td><td>0.680</td><td> $-0.86 \pm 0.14$ </td></tr><tr><td>AlignedMTL</td><td>signSGD</td><td>0.0001</td><td>0.363</td><td>0.637</td><td>0.528</td><td>0.211</td><td>25.86</td><td>20.02</td><td>0.282</td><td>0.550</td><td>0.674</td><td> $0.90 \pm 0.54$ </td></tr><tr><td>AlignedMTL</td><td>sgd</td><td>0.05</td><td>0.371</td><td>0.642</td><td>0.531</td><td>0.214</td><td>26.41</td><td>20.71</td><td>0.272</td><td>0.535</td><td>0.661</td><td> $2.33 \pm 1.00$ </td></tr><tr><td>adatask</td><td>adam</td><td>0.0001</td><td>0.394</td><td>0.650</td><td>0.518</td><td>0.211</td><td>26.29</td><td>20.67</td><td>0.273</td><td>0.536</td><td>0.661</td><td> $1.00 \pm 0.8$ </td></tr><tr><td>MTL-IO</td><td>sgd</td><td>0.01</td><td>0.377</td><td>0.637</td><td>0.551</td><td>0.225</td><td>30.05</td><td>25.50</td><td>0.215</td><td>0.447</td><td>0.574</td><td> $12.69 \pm 1.21$ </td></tr></table>

Table A7: Results for diferent MTO methods and optimizers on NYUv2 [39] using DeepLabV3+ [5]. The best score per metric is highlighted for each MTO method as well as over all methods and optimizers. We note a full dominance of Adam over SGD+mom on both the depth and normal tasks as well as on the $\varDelta _ { m }$ metric. Overall, best results for all metrics were also achieved using Adam as optimizer.

<table><tr><td rowspan="2">MTO</td><td rowspan="2">Optimizer</td><td rowspan="2">lr</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td colspan="5">Normal</td><td rowspan="2">DeltaM↓</td></tr><tr><td colspan="2">mIoU↑pixAcc↑</td><td colspan="2">AbsErr↓RelErr↓</td><td colspan="5">Mean↓Median↓&lt; 11.25 ↑&lt; 22.5 ↑&lt; 30.0 ↑</td></tr><tr><td>STL</td><td>adam</td><td></td><td>0.552</td><td>0.767</td><td>0.365</td><td>0.152</td><td>21.16</td><td>14.52</td><td>0.402</td><td>0.668</td><td>0.768</td><td></td></tr><tr><td>EW</td><td>adam</td><td>0.0001</td><td>0.552</td><td>0.770</td><td>0.352</td><td>0.143</td><td>22.54</td><td>16.05</td><td>0.366</td><td>0.634</td><td>0.742</td><td> $2.70 \pm 0.12$ </td></tr><tr><td>EW</td><td>signSGD</td><td>0.0001</td><td>0.554</td><td>0.766</td><td>0.362</td><td>0.147</td><td>22.69</td><td>16.15</td><td>0.364</td><td>0.630</td><td>0.739</td><td> $3.60 \pm 0.38$ </td></tr><tr><td>EW</td><td>sgd</td><td>0.01</td><td>0.556</td><td>0.769</td><td>0.365</td><td>0.147</td><td>23.35</td><td>16.76</td><td>0.352</td><td>0.617</td><td>0.727</td><td> $5.21 \pm 0.18$ </td></tr><tr><td>UW</td><td>adam</td><td>0.0005</td><td>0.532</td><td>0.755</td><td>0.358</td><td>0.141</td><td>22.01</td><td>15.30</td><td>0.383</td><td>0.650</td><td>0.753</td><td> $1.59 \pm 0.14$ </td></tr><tr><td>UW</td><td>signSGD</td><td>0.0005</td><td>0.526</td><td>0.748</td><td>0.366</td><td>0.146</td><td>22.18</td><td>15.46</td><td>0.380</td><td>0.645</td><td>0.749</td><td> $2.90 \pm 0.79$ </td></tr><tr><td>UW</td><td>sgd</td><td>0.01</td><td>0.557</td><td>0.769</td><td>0.364</td><td>0.148</td><td>22.37</td><td>15.69</td><td>0.374</td><td>0.641</td><td>0.746</td><td> $2.57 \pm 0.56$ </td></tr><tr><td>RLW</td><td>adam</td><td>0.0001</td><td>0.556</td><td>0.768</td><td>0.360</td><td>0.150</td><td>22.37</td><td>15.87</td><td>0.370</td><td>0.638</td><td>0.745</td><td> $2.96 \pm 0.55$ </td></tr><tr><td>RLW</td><td>signSGD</td><td>0.0001</td><td>0.557</td><td>0.768</td><td>0.363</td><td>0.146</td><td>22.40</td><td>15.84</td><td>0.371</td><td>0.638</td><td>0.745</td><td> $2.74 \pm 0.30$ </td></tr><tr><td>RLW</td><td>sgd</td><td>0.05</td><td>0.541</td><td>0.762</td><td>0.368</td><td>0.151</td><td>23.00</td><td>16.44</td><td>0.357</td><td>0.625</td><td>0.734</td><td> $5.15 \pm 0.54$ </td></tr><tr><td>IMTL</td><td>adam</td><td>0.0005</td><td>0.532</td><td>0.756</td><td>0.353</td><td>0.140</td><td>21.34</td><td>14.61</td><td>0.399</td><td>0.666</td><td>0.766</td><td> $-0.39 \pm 0.05$ </td></tr><tr><td>IMTL</td><td>signSGD</td><td>0.0001</td><td>0.552</td><td>0.767</td><td>0.359</td><td>0.146</td><td>21.81</td><td>15.22</td><td>0.385</td><td>0.652</td><td>0.756</td><td> $1.12 \pm 0.16$ </td></tr><tr><td>IMTL</td><td>sgd</td><td>0.05</td><td>0.545</td><td>0.762</td><td>0.367</td><td>0.150</td><td>21.63</td><td>14.98</td><td>0.390</td><td>0.658</td><td>0.760</td><td> $1.30 \pm 0.40$ </td></tr><tr><td>PCGrad</td><td>adam</td><td>0.0001</td><td>0.558</td><td>0.772</td><td>0.357</td><td>0.146</td><td>22.33</td><td>15.81</td><td>0.371</td><td>0.639</td><td>0.746</td><td> $2.34 \pm 0.48$ </td></tr><tr><td>PCGrad</td><td>signSGD</td><td>0.0001</td><td>0.553</td><td>0.766</td><td>0.360</td><td>0.148</td><td>22.61</td><td>16.11</td><td>0.365</td><td>0.631</td><td>0.739</td><td> $3.55 \pm 0.13$ </td></tr><tr><td>PCGrad</td><td>sgd</td><td>0.01</td><td>0.556</td><td>0.769</td><td>0.361</td><td>0.147</td><td>23.36</td><td>16.79</td><td>0.351</td><td>0.617</td><td>0.728</td><td> $5.10 \pm 0.28$ </td></tr><tr><td>CAGrad</td><td>adam</td><td>0.0005</td><td>0.531</td><td>0.755</td><td>0.357</td><td>0.143</td><td>21.42</td><td>14.67</td><td>0.399</td><td>0.664</td><td>0.764</td><td> $0.10 \pm 0.55$ </td></tr><tr><td>CAGrad</td><td>signSGD</td><td>0.0001</td><td>0.550</td><td>0.766</td><td>0.361</td><td>0.146</td><td>21.94</td><td>15.27</td><td>0.383</td><td>0.650</td><td>0.754</td><td> $1.49 \pm 0.12$ </td></tr><tr><td>CAGrad</td><td>sgd</td><td>0.05</td><td>0.556</td><td>0.768</td><td>0.360</td><td>0.144</td><td>21.77</td><td>14.99</td><td>0.391</td><td>0.657</td><td>0.759</td><td> $0.42 \pm 0.07$ </td></tr><tr><td>AlignedMTL</td><td>adam</td><td>0.0005</td><td>0.531</td><td>0.752</td><td>0.355</td><td>0.144</td><td>21.15</td><td>14.49</td><td>0.403</td><td>0.669</td><td>0.768</td><td> $-0.33 \pm 0.41$ </td></tr><tr><td>AlignedMTL</td><td>signSGD</td><td>0.0005</td><td>0.527</td><td>0.751</td><td>0.360</td><td>0.144</td><td>21.35</td><td>14.60</td><td>0.400</td><td>0.666</td><td>0.765</td><td> $0.25 \pm 0.21$ </td></tr><tr><td>AlignedMTL</td><td>sgd</td><td>0.1</td><td>0.550</td><td>0.766</td><td>0.367</td><td>0.150</td><td>21.59</td><td>14.88</td><td>0.393</td><td>0.659</td><td>0.761</td><td> $0.95 \pm 0.62$ </td></tr><tr><td>adatask</td><td>adam</td><td>0.0001</td><td>0.553</td><td>0.768</td><td>0.353</td><td>0.142</td><td>21.44</td><td>14.83</td><td>0.395</td><td>0.661</td><td>0.763</td><td> $-0.39 \pm 0.21$ </td></tr><tr><td>MTL-IO</td><td>sgd</td><td>0.01</td><td>0.554</td><td>0.768</td><td>0.362</td><td>0.146</td><td>23.31</td><td>16.75</td><td>0.352</td><td>0.617</td><td>0.728</td><td> $5.01 \pm 0.38$ </td></tr></table>

## A5 Additional gradient alignment results

We report extended evaluation on gradient simlarity in MTL and STL.

Alternatively to a single sample/ task, we consider the average gradient in Fig. A5. In Fig. A6, we diferentiate between conflicting and supporting gradient pairs when evaluating the cosine similarity. Figure A7 shows the evaluation of the scalar product as an combined measure of similarity in gradient direction and magnitude.

![](images/5afc63da5baf7494d55b034d93cda1437c5c345c29bddb96d2b4e49020ce97c0.jpg)  
Fig. A5: Gradient similarities when averaging over batch/ losses. In contrast to results shown in Fig. 4, we compute either the average gradient over all tasks when comparing inter-samples or the average gradient over all samples within the batch for the comparison between inter-tasks. We observed a lower variance in some cases (e.g. Cityscapes+Segnet, grad. cosine similarity) which we trace back on noisy gradients being averaged out. Overall, we obtain the same findings as for a direct gradient comparison.

![](images/16ec20819cc0c1cdb4ef8f0c973275bbea5dd0484df6ba7b7e57a1843e861164.jpg)

<details>
<summary>line</summary>

| Epoch | Overall (range) | pos (range) | neg (range) |
| --- | --- | --- | --- |
| 0~250 | -1.0~1.0 | -0.3~0.8 | -1.0~0.0 |
</details>

(a) Cityscapes, SegNet

![](images/8da803172bf51e8fa765ff311d06f6557727ce35e267b35bcfd0f2b0997e3e9f.jpg)

<details>
<summary>line</summary>

| Epoch | Series::Red (Value) | Series::Blue (Value) |
| --- | --- | --- |
| 0 | ~0.1 | ~0.1 |
| 100 | ~-0.1 | ~-0.1 |
| 200 | ~-0.1 | ~-0.1 |
| 250 | ~-0.1 | ~-0.1 |
</details>

(b) Cityscapes, DeepLabV3

![](images/459588e7ea52004f652eb22899833b508b583e3bf6c317380dc7dca4eb8bfc41.jpg)

<details>
<summary>line</summary>

| Epoch | Model 1 (Blue) | Model 2 (Red) | Model 3 (Blue) |
| --- | --- | --- | --- |
| 0 | ~0.25 | ~-0.15 | ~0.25 |
| 50 | ~0.25 | ~-0.15 | ~0.25 |
| 100 | ~0.25 | ~-0.15 | ~0.25 |
| 150 | ~0.25 | ~-0.15 | ~0.25 |
</details>

(c) NYUv2, SegNet

![](images/aebf02dda123ebf040184c33a60faeaa3418fde61862d1210ef7a0b6f04e5120.jpg)

<details>
<summary>line</summary>

| Epoch | Model 1 (Blue) | Model 2 (Red) | Model 3 (Dark Blue) |
| --- | --- | --- | --- |
| 0 | ~0.8 | ~-0.1 | ~0.1 |
| 50 | ~0.4 | ~-0.1 | ~0.1 |
| 100 | ~0.3 | ~-0.1 | ~0.1 |
| 150 | ~0.3 | ~-0.1 | ~0.1 |
</details>

(d) NYUv2, DeepLabV3

![](images/96a6bd2fd0f70622d919ee1b5b759ebc10d39ee6262ad1055a78227d40d313bb.jpg)

<details>
<summary>line</summary>

| Epoch | Top Panel (Blue) | Top Panel (Red) | Middle Panel (Blue) | Middle Panel (Red) | Bottom Panel (Blue) | Bottom Panel (Red) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.15 | ~-0.15 | ~0.35 | ~0.15 | ~-0.15 | ~-0.15 |
| 20 | ~0.15 | ~-0.15 | ~0.40 | ~0.15 | ~-0.15 | ~-0.15 |
| 40 | ~0.15 | ~-0.15 | ~0.40 | ~0.15 | ~-0.15 | ~-0.15 |
| 60 | ~0.15 | ~-0.15 | ~0.40 | ~0.15 | ~-0.15 | ~-0.15 |
| 80 | ~0.15 | ~-0.15 | ~0.40 | ~0.15 | ~-0.15 | ~-0.15 |
| 90 | ~0.15 | ~-0.15 | ~0.40 | ~0.15 | ~-0.15 | ~-0.15 |
</details>

(e) CelebA, ResNet50  
Fig. A6: Diferentiation between conflicting and supportive gradients. We report mean (solid line), standard deviation (shaded area), upper (97.5%) and lower (2.5%) percentile (dotted line) of the gradient cosine similarity between either inter-samples gradients or inter-tasks gradients within an epoch. While showing overall results over all respective gradient pairs (Top) as can be also found in Figure 4, we also show the course of cosine similarity for either gradients that are conflicting ([neg], bottom) or those which have cosine similarity greater than zero ([pos], middle).

![](images/d1b1113c38dd3e7bb6fa969f207bf69ccbd0215b6db1326c686cac31146f1e60.jpg)  
(a) Cityscapes, SegNet

![](images/54c499ca264846d10265acf8bf0bc9862feef377de69421e3632acef48de38d6.jpg)  
(b) Cityscapes, DeepLabV3

![](images/3992e0497bd16353af90d8671f4fa78e979484f9bf4007faa3a1fe5de06732a2.jpg)  
(c) NYUv2, SegNet

![](images/6eab29c54944335e7525e0e6bdd4172b5f4ce34433d96fd5a8b3e1d7b0c7fc5f.jpg)  
(d) NYUv2, DeepLabV3

![](images/921db429ec18e07f0f19322535166e689be408155e4d39b19b9c72bace09035b.jpg)  
(e) CelebA, ResNet50  
Fig. A7: Scalar product between pairs of gradients. We report mean (solid line), standard deviation (shaded area), upper (97.5%) and lower (2.5%) percentile (dotted line) of the gradient cosine similarity between either gradients of inter-samples or inter-tasks within an epoch. We observe an overall decrease of the variance of the scalar product for both Cityscapes setups and the NYUv2+DeepLabV3 experiment over the training which we explain with evenly smaller overall gradients. Surprisingly, this does not apply for NYUv2 with Seg-Net or CelebA. Similar to previous results, we do not see any indication for inter-sample gradients being better aligned than inter-tasks gradients.

## A6 Robustness of multi-task representations on corrupted data

In this additonal part of our analysis, we investigate whether features learned for multiple tasks generalize better to corrupted data compared to those learned for single tasks only.

Motivation: In his seminal paper, Caruana gives preliminary evidence that MTL provides stronger features and avoids spurious correlations (referred to better attribute selection) [4]. More recently, spurious correlations have often been directly connected with robustness [13, 18]. Results from current literature on the robustness of MTL features are mixed. While MTL is stated to increase the adversarial- and noise-robustness over STL [24, 37, 54], others argue features selected by MTL could be more likely to be non-causal and, therefore, less robust [2, 17]. Here, we further examine whether MTL features lead to better robustness. We would like to nuance that we do not consider the transferability of representations, e.g., to new tasks, but solely focus on the claim that the MTL trained features are more robust w.r.t. diferent inputs.

Approach: In our experiment, we treat the common corruptions [16] as downstream task and compare the performance after fine-tuning the heads on corrupted data while freezing the pre-trained STL/MTL backbone. While this differs from the typical OOD setup, here, it allows us to explore whether MTL or STL yield more robust representation for corrupted data.

We select models trained on clean data with the best performing hyperparameter configuration from previous experiments and fine-tune their heads on corrupted data. Following this, we compare the test performance of models trained in the multi-task setup to those that were learned for a single-task only. We use the perturbation modes proposed by Hendrycks et al. [16] which include diferent variants of noise, blur, and weather conditions and apply five levels of severity. We randomly select corruption and severity level for each data sample during fine-tuning and create a full corrupted version of the test data considering all proposed corruptions and perturbation levels.

To quantify the robustness of single- and multi-task models, we first compute the individual task metrics M (e.g., mIoU) per task t for a STL and MTL network. Next, we compute the relative performance when each model is faced with corrupted data. Lastly, we calculate the diference of relative performances of the MTL compared to the STL model. In detail, over all corruption modes C and levels of severity $S$ we have

$$
\delta_ {t} = \frac {1}{| C | \cdot | S |} \sum_ {c \in C} \sum_ {s \in S} (- 1) ^ {p (t)} \delta_ {t, c, s} \tag {26}
$$

$$
\mathrm{with} \quad \delta_ {t, c, s} = \frac {M _ {t , c , s} ^ {M T L , c o r r u p t e d}}{M _ {t} ^ {M T L , c l e a n}} - \frac {M _ {t , c , s} ^ {S T L , c o r r u p t e d}}{M _ {t} ^ {S T L , c l e a n}}
$$

where $p ( t ) = 1$ if a higher value on task t corresponds to better performance and $p ( t ) = 0$ otherwise. This metric yields $\delta _ { t } < 0$ if the MTL model was able to handle data corruption better. If the STL model is less impacted, we get $\delta _ { t } > 0$

![](images/79de9506f804bdbcd141b50226e6924ed5a6bb2a3492466070d07a7d721a41c3.jpg)

<details>
<summary>line</summary>

| Category | \(\delta(\)mIoU) | \(\delta(\)pixAcc) | \(\delta(\)AbsErr) | \(\delta(\)RelErr) |
| --- | --- | --- | --- | --- |
| Gaussian Noise | ~-0.2 | ~-0.15 | ~-0.2 | ~-1.0 |
| Snot Noise | ~-0.2 | ~-0.15 | ~-0.2 | ~-1.05 |
| Impulse Noise | ~-0.25 | ~-0.15 | ~-0.3 | ~-1.05 |
| Defocus Blur | ~-0.4 | ~-0.15 | ~-0.45 | ~-0.7 |
| Glass Blur | ~-0.4 | ~-0.15 | ~-0.45 | ~-0.6 |
| Motion Blur | ~-0.35 | ~-0.15 | ~-0.6 | ~-0.2 |
| Zoom Blur | ~-0.45 | ~-0.15 | ~-0.55 | ~-0.35 |
| Snow | ~-0.25 | ~-0.25 | ~-0.2 | ~-0.85 |
| Frost | ~-0.2 | ~-0.2 | ~-0.25 | ~-0.65 |
| Fog | ~-0.35 | ~-0.25 | ~-0.35 | ~-0.75 |
| Brightness | ~-0.4 | ~-0.2 | ~-0.1 | ~0.1 |
| Contrast | ~-0.25 | ~-0.15 | ~-0.35 | ~-0.4 |
| Elastic | ~-0.35 | ~-0.15 | ~-0.35 | ~-0.3 |
| Pixelate | ~-0.35 | ~-0.15 | ~-0.3 | ~0.25 |
| JPEG | ~-0.35 | ~-0.15 | ~-0.4 | ~-0.35 |
</details>

(a) Cityscapes, SegNet

![](images/79faa42f4c1f47e9a91b2083e5e14e1be9a98d1f42def01d4bf75aef8b86dc6f.jpg)

<details>
<summary>line</summary>

| Category | \(\delta(\)mIoU) | \(\delta(\)pixAcc) | \(\delta(\)AbsErr) | \(\delta(\)RelErr) |
| --- | --- | --- | --- | --- |
| Gaussian Noise | ~0.01 | 0.0 | ~-0.04 | ~-0.33 |
| Shot Noise | ~0.01 | 0.0 | ~-0.02 | ~-0.33 |
| Impulse Noise | ~0.01 | 0.0 | ~-0.17 | ~-0.33 |
| Deforce Blur | ~0.01 | 0.0 | ~0.18 | ~-0.52 |
| Glass Blur | ~0.01 | 0.0 | ~-0.16 | ~-0.34 |
| Motion Blur | ~0.01 | 0.0 | ~-0.10 | ~-0.25 |
| Zoom Blur | ~0.01 | 0.0 | ~-0.10 | ~-0.30 |
| Snow | ~0.01 | 0.0 | ~-0.11 | ~-0.12 |
| Frost | ~0.01 | 0.0 | ~0.04 | ~-0.15 |
| Fog | ~0.01 | 0.0 | ~0.11 | ~0.32 |
| Brightness | ~0.01 | 0.0 | ~0.07 | ~-0.02 |
| Contrast | ~0.01 | 0.0 | ~-0.11 | ~-0.15 |
| Elastic | ~0.01 | 0.0 | ~-0.11 | ~-0.22 |
| Pixelate | ~0.01 | 0.0 | ~0.14 | ~-0.02 |
| JPEG | ~0.01 | 0.0 | ~0.07 | ~-0.26 |
</details>

(b) Cityscapes, DeepLabV3

![](images/2e1219cdf8ef769799875ed5db517150594dab6cc9e16c072a5d7794f816c686.jpg)

<details>
<summary>line</summary>

| Category | \(\delta(\)mIoU) | \(\delta(\)pixAcc) | \(\delta(\)AbsErr) | \(\delta(\)RelErr) | \(\delta(<11.25)\) | \(\delta(<22.5)\) | \(\delta(<30.0)\) | \(\delta(\)Mean) | \(\delta(\)Median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian Noise | ~0.03 | ~0.03 | 0.0 | ~0.09 | ~0.04 | ~0.04 | ~-0.01 | ~-0.14 | ~-0.28 |
| Snot Noise | ~0.03 | ~0.03 | ~0.04 | ~0.09 | ~0.05 | ~0.05 | ~-0.01 | ~-0.13 | ~-0.26 |
| Impulse Noise | ~0.07 | ~0.07 | 0.0 | ~0.15 | ~0.08 | ~0.08 | ~-0.01 | ~-0.12 | ~-0.28 |
| Defocus Blur | ~0.08 | ~0.08 | ~0.29 | ~0.24 | ~0.11 | ~0.11 | ~-0.01 | ~-0.06 | ~-0.12 |
| Glass Blur | ~0.04 | ~0.04 | ~0.23 | ~0.19 | ~0.05 | ~0.05 | ~-0.01 | ~-0.06 | ~-0.12 |
| Motion Blur | 0.0 | 0.0 | 0.05 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Zoom Blur | 0.02 | 0.02 | ~0.22 | ~0.18 | 0.01 | 0.01 | 0.01 | -0.16 | -0.21 |
| Snow | 0.02 | 0.02 | ~0.16 | ~0.13 | 0.02 | 0.02 | 0.02 | -0.15 | -0.26 |
| Frost | -0.03 | -0.03 | -0.06 | 0.07 | 0.11 | 0.11 | 0.11 | -0.13 | -0.19 |
| Fog | -0.05 | -0.05 | -0.06 | 0.04 | 0.11 | 0.11 | 0.11 | -0.13 | -0.19 |
| Brightness | -0.02 | -0.02 | -0.04 | 0.04 | -0.15 | -0.15 | -0.15 | -0.13 | -0.19 |
| Contrast | -0.04 | -0.04 | -0.05 | 0.05 | 0.13 | 0.13 | 0.13 | -0.13 | -0.19 |
| Elastic | 0.02 | 0.02 | 0.08 | 0.15 | -0.15 | -0.15 | -0.15 | -0.13 | -0.21 |
| Pevulate | 0.03 | 0.03 | 0.15 | 1.15 | -0.15 | -1.15 | -1.15 | -1.15 | -1.15 |
| JPEG | 0.03 | 0.03 | 1.15 | 1.15 | -1.15 | -1.15 | -1.15 | -1.15 | -1.15 |
</details>

(c) NYUv2, SegNet

![](images/908911220835aa32b0451c8ed987e8982b67364c23933936dd596c27257748bf.jpg)

<details>
<summary>line</summary>

| Category | \(\delta(\)mIoU) | \(\delta(\)pixAcc) | \(\delta(\)AbsErr) | \(\delta(\)RelErr) | \(\delta(<11.25)\) | \(\delta(<22.5)\) | \(\delta(<30.0)\) | \(\delta(\)Mean) | \(\delta(\)Median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian Noise | ~-0.08 | ~-0.05 | ~-0.10 | ~-0.22 | ~-0.10 | ~-0.10 | ~-0.10 | ~-0.18 | ~-0.40 |
| Shot Noise | ~-0.09 | ~-0.06 | ~-0.12 | ~-0.24 | ~-0.12 | ~-0.12 | ~-0.12 | ~-0.18 | ~-0.35 |
| Impulse Noise | ~-0.10 | ~-0.07 | ~-0.14 | ~-0.26 | ~-0.14 | ~-0.14 | ~-0.14 | ~-0.20 | ~-0.45 |
| Defocus Blur | 0.00 | 0.00 | -0.23 | -0.18 | -0.18 | -0.18 | -0.18 | -0.18 | -0.18 |
| Glass Blur | 0.00 | 0.00 | -0.18 | -0.12 | -0.18 | -0.18 | -0.18 | -0.18 | -0.18 |
| Motion Blur | 0.00 | 0.00 | -0.12 | -0.08 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 |
| Zoom Blur | 0.00 | 0.00 | -0.16 | -0.12 | -0.16 | -0.16 | -0.16 | -0.16 | -0.16 |
| Snow | 0.00 | 0.00 | -0.14 | -0.18 | -0.14 | -0.14 | -0.14 | -0.18 | -0.26 |
| Frost | 0.00 | 0.00 | -0.16 | -0.16 | -0.16 | -0.16 | -0.16 | -0.18 | -0.32 |
| Fog | 0.02 | 0.02 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 |
| Brightness | 0.02 | 0.02 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 |
| Contrast | 0.05 | 0.05 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 | -0.12 |
| Elastic | 0.02 | 0.02 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.24 |
| Pixelate | 0.02 | 0.02 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 |
| JPEG | 0.02 | 0.02 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 | -0.24 |
</details>

(d) NYUv2, DeepLabV3  
Fig. A8: Transfer to out-of-distribution data for MTL and STL. For every task and respective metrics, we show the diference over relative performance decrease over all corruption modes averaged over five levels of severity and three runs. EW was used to train the MTL model on uncorrupted data. We color blocks in case either STL or MTL is able to handle the respective corruption better for all metrics of one task. Regarding the Cityscapes dataset, the performance on both task would strongly benefit in MTL setup. A similar behavior can be seen for NYUv2+DeepLabV3. Using SegNet on NYUv2, however, shows preferences towards STL features. Overall, we see a minor indication that MTL result in features that would generalize better to corrupted data.

Results: Figure A8 shows $\delta _ { t }$ for diferent corruption types for all network architectures and datasets with EW averaged over five corruption levels and three seeds. Considering the combination DeepLabV3 and Cityscapes, on the semantic segmentation tasks, the STL models show a slightly lower decrease in performance on the corrupted data than MTL $( \delta _ { t } > 0$ more often; shaded in red), indicating that the features learned for these respective tasks can better generalize to corrupted data. In contrast, the MTL model shows significant better relative performance on the depth task $( \delta _ { t } < 0$ more often; shaded green). Comparing these observations to other dataset+network combinations, we find a strong robustness of MTL features for both setups of CityScape+SegNet and NYUv2+DeepLabV3 over all tasks. However, evaluation on NYUv2+SegNet resulted on average in better performance on STL features, especially for the depth task. Furthermore, we see little evidence for general higher robustness against certain types of corruption (e.g., higher robustness against weather conditions) for either MTL or STL across all setups.

Table A8: Out-Of Distribution transfer on corrupted Cityscapes [9] dataset for diferent networks and MTO methods. We report diference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. Equation (26)). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. Overall, we observe a slight benefit in performance for the depth task when training for multiple tasks.

<table><tr><td rowspan="2">Network</td><td rowspan="2">MTO</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td rowspan="2">Mean</td></tr><tr><td> $\delta_{\text{mIoU}}$ </td><td> $\delta_{\text{pixAcc}}$ </td><td> $\delta_{\text{AbsErr}}$ </td><td> $\delta_{\text{RelErr}}$ </td></tr><tr><td rowspan="6">SegNet</td><td>EW</td><td>-0.2922</td><td>-0.1386</td><td>-0.3175</td><td>-0.4949</td><td>-0.3108</td></tr><tr><td>UW</td><td>0.0854</td><td>0.0421</td><td>0.7799</td><td>-0.2118</td><td>0.1739</td></tr><tr><td>RLW</td><td>0.0699</td><td>0.0307</td><td>0.6553</td><td>-0.5652</td><td>0.0477</td></tr><tr><td>IMTL</td><td>0.0696</td><td>0.0243</td><td>0.5384</td><td>0.1108</td><td>0.1858</td></tr><tr><td>PCGrad</td><td>0.0824</td><td>0.0316</td><td>0.4106</td><td>-0.6392</td><td>-0.0286</td></tr><tr><td>CAGrad</td><td>0.0561</td><td>0.0213</td><td>0.3860</td><td>-0.3100</td><td>0.0383</td></tr><tr><td rowspan="6">DeepLabV3</td><td>EW</td><td>0.0087</td><td>0.0017</td><td>-0.0108</td><td>-0.1826</td><td>-0.0457</td></tr><tr><td>UW</td><td>0.0110</td><td>0.0018</td><td>0.1179</td><td>0.0861</td><td>0.0542</td></tr><tr><td>RLW</td><td>0.0263</td><td>0.0084</td><td>-0.0809</td><td>-0.3595</td><td>-0.1015</td></tr><tr><td>IMTL</td><td>0.0172</td><td>0.0041</td><td>0.1747</td><td>0.0857</td><td>0.0704</td></tr><tr><td>PCGrad</td><td>0.0090</td><td>0.0019</td><td>0.0394</td><td>-0.2240</td><td>-0.0434</td></tr><tr><td>CAGrad</td><td>0.0224</td><td>0.0067</td><td>0.0898</td><td>-0.2841</td><td>-0.0413</td></tr></table>

The results of other MTO methods (Tabs. A8 and A9) indicate that it depends less on the used method but more on the choice of dataset and network architecture whether some tasks would benefit from MTL for learning more robust features. Averaged absolute scores can be found in Tabs. A10 and A11.

Conclusion: Our experiments show that MTL can result in learning more robust features, either for a subset of tasks or even all. However, we could not observe a uniform pattern whether certain tasks consistently benefit from MTL. Instead, it depends on the task, the type of corruption, the network, and the dataset whether MTL or STL is superior towards corrupted data. Whether there is a general pattern, we leave to further research. We further cannot fully confirm the outcome of [24] as only two of our setups have indicated that the segmentation task can be more robust in the MTL setting. Controversial to the claim of [37], our evaluation shows that none of the MTL approaches, even IMTL, PC-Grad or CAGrad which adjust the gradients, yields consistent values of $\delta _ { t } < 0$ which would have shown an advantage of certain MTO methods over STL.

Table A9: Out-Of Distribution transfer on corrupted NYUv2 [39] dataset for diferent multi-task optimization methods. We report difference between relative performance decrease for single-task and multi-task learning averaged over all modes of corruption and all levels of severity (cf. Equation (26)). A value lower than zero indicates a better generalization capability of the MTL backbone, a positive value displays that the STL backbone shows a lower decrease when evaluated on the corrupted data. Results are averaged over runs for three seeds for both multi-task and single-task models. While for DeepLabV3 largely benefits from MTL, this is not the case for SegNet. Over both networks, EW profits shows lowest relative performance decrease among all MTO methods. Interestingly we found that even for diferent metrics corresponding to the same task, either the multi-task or single-task learning model would show lower decrease in performance on the corrupted data.

<table><tr><td rowspan="2">Network</td><td rowspan="2">MTO</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td colspan="5">Normal</td><td rowspan="2">Mean</td></tr><tr><td> $\delta_{\text{mIoU}}$ </td><td> $\delta_{\text{pixAcc}}$ </td><td> $\delta_{\text{AbsErr}}$ </td><td> $\delta_{\text{RelErr}}$ </td><td> $\delta_{\text{Mean}}$ </td><td> $\delta_{\text{Median}}$ </td><td> $\delta_{<11.25}$ </td><td> $\delta_{<22.5}$ </td><td> $\delta_{<30.0}$ </td></tr><tr><td rowspan="6">SegNet</td><td>EW</td><td>0.0139</td><td>0.0134</td><td>0.0715</td><td>0.0922</td><td>-0.0836</td><td>-0.1492</td><td>0.0293</td><td>0.0168</td><td>0.0116</td><td>0.0018</td></tr><tr><td>UW</td><td>0.0166</td><td>0.0149</td><td>0.1146</td><td>0.1225</td><td>-0.0564</td><td>-0.0984</td><td>0.0313</td><td>0.0207</td><td>0.0164</td><td>0.0204</td></tr><tr><td>RLW</td><td>0.0174</td><td>0.0223</td><td>-0.0062</td><td>0.0281</td><td>-0.1048</td><td>-0.1857</td><td>0.0229</td><td>0.0089</td><td>0.0055</td><td>-0.0213</td></tr><tr><td>IMTL</td><td>0.0073</td><td>0.0123</td><td>0.1108</td><td>0.1247</td><td>-0.0186</td><td>-0.0390</td><td>0.0361</td><td>0.0274</td><td>0.0228</td><td>0.0315</td></tr><tr><td>PCGrad</td><td>0.0186</td><td>0.0174</td><td>0.0720</td><td>0.0817</td><td>-0.0702</td><td>-0.1271</td><td>0.0264</td><td>0.0158</td><td>0.0121</td><td>0.0052</td></tr><tr><td>CAGrad</td><td>0.0061</td><td>0.0116</td><td>0.1172</td><td>0.1428</td><td>-0.0082</td><td>-0.0202</td><td>0.0465</td><td>0.0286</td><td>0.0223</td><td>0.0385</td></tr><tr><td rowspan="6">DeepLabV3</td><td>EW</td><td>-0.0289</td><td>-0.0187</td><td>-0.1194</td><td>-0.1249</td><td>-0.1182</td><td>-0.1947</td><td>-0.0409</td><td>-0.0288</td><td>-0.0238</td><td>-0.0776</td></tr><tr><td>UW</td><td>-0.0268</td><td>-0.0165</td><td>-0.0263</td><td>-0.0219</td><td>-0.0465</td><td>-0.0701</td><td>-0.0237</td><td>-0.0139</td><td>-0.0113</td><td>-0.0286</td></tr><tr><td>RLW</td><td>-0.0084</td><td>-0.0054</td><td>-0.0075</td><td>-0.0187</td><td>-0.0390</td><td>-0.0597</td><td>-0.0051</td><td>0.0009</td><td>0.0005</td><td>-0.0158</td></tr><tr><td>IMTL</td><td>-0.0211</td><td>-0.0167</td><td>0.0130</td><td>0.0059</td><td>-0.0153</td><td>-0.0219</td><td>-0.0072</td><td>-0.0045</td><td>-0.0038</td><td>-0.0079</td></tr><tr><td>PCGrad</td><td>-0.0102</td><td>-0.0081</td><td>-0.0242</td><td>-0.0327</td><td>-0.0504</td><td>-0.0800</td><td>-0.0236</td><td>-0.0107</td><td>-0.0076</td><td>-0.0275</td></tr><tr><td>CAGrad</td><td>-0.0139</td><td>-0.0129</td><td>0.0022</td><td>-0.0123</td><td>-0.0096</td><td>-0.0063</td><td>0.0023</td><td>0.0047</td><td>0.0034</td><td>-0.0047</td></tr></table>

<table><tr><td rowspan="2">Network</td><td rowspan="2">MTO</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td></tr><tr><td>mIoU</td><td>pixAcc</td><td>AbsErr</td><td>RelErr</td></tr><tr><td rowspan="7">SegNet</td><td>STL</td><td>0.3774</td><td>0.7241</td><td>0.0592</td><td>87.06</td></tr><tr><td>EW</td><td>0.3783</td><td>0.7399</td><td>0.0533</td><td>95.17</td></tr><tr><td>UW</td><td>0.3689</td><td>0.7273</td><td>0.0503</td><td>77.86</td></tr><tr><td>RLW</td><td>0.3744</td><td>0.7321</td><td>0.0526</td><td>96.26</td></tr><tr><td>IMTL</td><td>0.3684</td><td>0.7363</td><td>0.0479</td><td>78.92</td></tr><tr><td>PCGrad</td><td>0.3678</td><td>0.7310</td><td>0.0515</td><td>94.44</td></tr><tr><td>CAGrad</td><td>0.3769</td><td>0.7397</td><td>0.0479</td><td>92.75</td></tr><tr><td rowspan="7">DeepLabV3</td><td>STL</td><td>0.5732</td><td>0.8593</td><td>0.0244</td><td>98.97</td></tr><tr><td>EW</td><td>0.5769</td><td>0.8619</td><td>0.0253</td><td>104.37</td></tr><tr><td>UW</td><td>0.5801</td><td>0.8632</td><td>0.0240</td><td>99.67</td></tr><tr><td>RLW</td><td>0.5634</td><td>0.8557</td><td>0.0270</td><td>103.22</td></tr><tr><td>IMTL</td><td>0.5769</td><td>0.8617</td><td>0.0244</td><td>101.26</td></tr><tr><td>PCGrad</td><td>0.5742</td><td>0.8609</td><td>0.0263</td><td>105.46</td></tr><tr><td>CAGrad</td><td>0.5690</td><td>0.8576</td><td>0.0259</td><td>104.90</td></tr></table>

Table A10: Results for evaluating on corrupted Cityscapes [9]. Scores are averaged over all corruption modes, level of severity and three seeds.

<table><tr><td rowspan="2">Network</td><td rowspan="2">MTO</td><td colspan="2">Sem.Seg.</td><td colspan="2">Depth</td><td colspan="5">Normal</td></tr><tr><td>mIoU</td><td>pixAcc</td><td>AbsErr</td><td>RelErr</td><td>Mean</td><td>Median</td><td>&lt; 11.25</td><td>&lt; 22.5</td><td>&lt; 30.0</td></tr><tr><td rowspan="7">SegNet</td><td>STL</td><td>0.2282</td><td>0.4783</td><td>0.8389</td><td>0.3156</td><td>32.05</td><td>26.60</td><td>0.1994</td><td>0.4390</td><td>0.5608</td></tr><tr><td>EW</td><td>0.2253</td><td>0.4750</td><td>0.7702</td><td>0.2886</td><td>36.06</td><td>32.90</td><td>0.1306</td><td>0.3263</td><td>0.4529</td></tr><tr><td>UW</td><td>0.2266</td><td>0.4750</td><td>0.7847</td><td>0.2913</td><td>34.90</td><td>31.25</td><td>0.1445</td><td>0.3538</td><td>0.4814</td></tr><tr><td>RLW</td><td>0.2182</td><td>0.4587</td><td>0.7718</td><td>0.2957</td><td>37.09</td><td>34.39</td><td>0.1165</td><td>0.3045</td><td>0.4298</td></tr><tr><td>IMTL</td><td>0.2190</td><td>0.4743</td><td>0.7804</td><td>0.2885</td><td>33.89</td><td>29.67</td><td>0.1609</td><td>0.3812</td><td>0.5085</td></tr><tr><td>PCGrad</td><td>0.2279</td><td>0.4751</td><td>0.7790</td><td>0.2900</td><td>35.34</td><td>31.90</td><td>0.1388</td><td>0.3423</td><td>0.4696</td></tr><tr><td>CAGrad</td><td>0.2324</td><td>0.4850</td><td>0.7822</td><td>0.2897</td><td>33.50</td><td>29.10</td><td>0.1648</td><td>0.3908</td><td>0.5183</td></tr><tr><td rowspan="7">DeepLabV3</td><td>STL</td><td>0.3793</td><td>0.6285</td><td>0.5342</td><td>0.2111</td><td>27.32</td><td>21.25</td><td>0.2853</td><td>0.5286</td><td>0.6414</td></tr><tr><td>EW</td><td>0.3644</td><td>0.6132</td><td>0.5594</td><td>0.2181</td><td>29.88</td><td>24.80</td><td>0.2359</td><td>0.4657</td><td>0.5851</td></tr><tr><td>UW</td><td>0.3836</td><td>0.6321</td><td>0.5137</td><td>0.2003</td><td>27.75</td><td>21.89</td><td>0.2738</td><td>0.5169</td><td>0.6324</td></tr><tr><td>RLW</td><td>0.3861</td><td>0.6338</td><td>0.5290</td><td>0.2081</td><td>28.33</td><td>23.00</td><td>0.2566</td><td>0.4964</td><td>0.6162</td></tr><tr><td>IMTL</td><td>0.3773</td><td>0.6323</td><td>0.5208</td><td>0.1991</td><td>27.27</td><td>21.34</td><td>0.2830</td><td>0.5273</td><td>0.6412</td></tr><tr><td>PCGrad</td><td>0.3932</td><td>0.6409</td><td>0.5168</td><td>0.2025</td><td>28.05</td><td>22.44</td><td>0.2659</td><td>0.5063</td><td>0.6238</td></tr><tr><td>CAGrad</td><td>0.3738</td><td>0.6289</td><td>0.5266</td><td>0.2022</td><td>27.68</td><td>21.86</td><td>0.2754</td><td>0.5177</td><td>0.6324</td></tr></table>

Table A11: Results for evaluating on corrupted NYUv2 [39]. Scores are averaged over all corruption modes, level of severity and three seeds.