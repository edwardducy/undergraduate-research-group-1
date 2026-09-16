# Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective

Zhi Zhang<sup>1</sup>, Jiayi Shen<sup>1</sup>, Congfeng Cao<sup>1</sup>, Gaole Dai<sup>2</sup>, Shiji Zhou<sup>3</sup>, Qizhe Zhang<sup>2</sup>, Shanghang Zhang<sup>2</sup>\*, Ekaterina Shutova<sup>1</sup>\*

<sup>1</sup>ILLC, University of Amsterdam, Netherlands

<sup>2</sup> State Key Laboratory of Multimedia Information Processing,

School of Computer Science, Peking University, China

<sup>3</sup>Department of Automation, Tsinghua University, China

zhangzhizz2626@gmail.com, j.shen@uva.nl, shanghang@pku.edu.cn, e.shutova@uva.nl

## Abstract

Advancing towards generalist agents necessitates the concurrent processing of multiple tasks using a unified model, thereby underscoring the growing significance of simultaneous model training on multiple downstream tasks. A common issue in multi-task learning is the occurrence of gradient conflict, which leads to potential competition among different tasks during joint training. This competition often results in improvements in one task at the expense of deterioration in another. Although several optimization methods have been developed to address this issue by manipulating task gradients for better task balancing, they cannot decrease the incidence ofgradient conflict. In this paper, we systematically investigate the occurrence of gradient conflict across different methods and propose a strategy to reduce such conflicts through sparse training (ST), wherein only a portion of the model’s parameters are updated during training while keeping the rest unchanged. Our extensive experiments demonstrate that ST effectively mitigates conflicting gradients and leads to superior performance. Furthermore, ST can be easily integrated with gradient manipulation techniques, thus enhancing their effectiveness.

## 1. Introduction

Attaining the status of a generalist agent necessitates addressing multiple tasks within a unified architecture, thereby emphasizing the significance of multi-task learning (MTL) [37], which involves concurrently acquiring proficiency in multiple tasks and striving for superior overall performance compared to learning these tasks separately.

The primary concern for MTL lies in the phenomenon of task competition when the model is jointly trained by optimizing the average loss across all tasks. As a result, a subset of tasks demonstrates superior performance while others remain sub-optimized compared to their individual learning counterparts. One of the reasons behind it, from an optimization perspective, is gradient conflict (GC) [35], wherein the direction and magnitude of gradients between tasks differ significantly. This can result in the average gradient biasing towards optimizing one task while providing relatively smaller and sometimes even negative optimization for other tasks when updating the network [18, 35].

Numerous works have employed the gradient manipulation method to directly or indirectly adjust the gradients of tasks to mitigate the issue of gradient conflict in tasks. The former involves direct alteration of task gradients through manually designed criteria when conflicts arise [4, 17, 35], while the latter modifies task gradients by adjusting weights of loss for each task [18, 19, 26, 28]. Although these methods effectively modify the gradients conflicting with each other, they do not decrease the occurrence of conflicting gradients during training [29].

A simple approach to mitigate the occurrence of conflicting gradients is to convert those layers in which gradient conflict frequently arises into task-specific layers, thereby reducing the likelihood of gradient conflicts within the remaining shared layers [29]. However, this strategy introduces additional modules and disrupts the internal structure of the original model, resulting in increased computational costs. Furthermore, identifying frequently conflicting layers adds extra computational costs. This becomes prohibitively expensive as the model size continues to expand, and thus prompting our fundamental inquiry:

(Q) Is there a universally applicable approach to proactively mitigate the occurrence of gradient conflicts as well as preserve architectural integrityfor MTL?

![](images/218bdee40419f1ffb230df2cb837d55059224e058e2decdbd7ad6e80f58de858.jpg)

<details>
<summary>bar</summary>

| Category | Baseline (All epochs) (%) | Baseline w/ ST (All epochs) (%) | Baseline (Last 50% epochs) (%) | Baseline w/ ST (Last 50% epochs) (%) |
| --- | --- | --- | --- | --- |
| Joint Train | ~32 | ~26 | ~36 | ~29 |
| PCGrad | ~34 | ~30 | ~39 | ~33 |
| CAGrad | ~34 | ~31 | ~40 | ~35 |
| GradDrop | ~34 | ~31 | ~38 | ~34 |
| MGDA | ~40 | ~40 | ~45 | ~42 |
| IMTL-G | ~32 | ~28 | ~37 | ~31 |
| NashMTL | ~37 | ~35 | ~40 | ~35 |
</details>

Figure 1. The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on the SAM model with NYUv2 datasets is evaluated using various methods, including joint training and gradient manipulation techniques.

To tackle this issue, we propose a novel perspective on mitigating gradient conflict in MTL, termed Sparse Training (ST), wherein a subset of parameters from the original model are selected to learn multiple tasks simultaneously while keeping the remaining parameters frozen. The intuition behind this lies in the reduction of a high-dimensional optimization problem to a low-dimensional one, which effectively alleviates the optimization complexity. Moreover, restricting the gradient updates of individual tasks to influence only a subset of parameters, rather than all parameters, effectively reduces potential interference between tasks.

Our key findings demonstrate that ST can effectively reduce the incidence of gradient conflict, particularly during the later stages of training, as illustrated in Fig. 1. A summary of our contributions is as follows: i) We provide a novel perspective, sparse training, for proactively reducing the incidence of gradient conflict during training while keeping the architecture intact; ii) Sparse training can be easily applied to improve various gradient manipulation methods by reducing the occurrence the gradient conflict over different datasets and architectures; iii) In addition to conventional research that primarily focuses on smaller models (MTAN [20] and SegNet [1]), we provide a comprehensive assessment of larger pre-trained models, including SAM [3], ViT [8], Swin Transformer [22], using various gradient manipulation techniques, such as PCGrad [35], CAGrad [17], GradDrop [4], MGDA [28], IMTL-G [19] and NashMTL [26], to stimulate research in the field of sparse training for MTL. Our findings demonstrate that as the model size increases, the issue of gradient conflict becomes more exacerbated, as shown in Fig. 5a, underscoring the significance of investigating the gradient conflict in large-scale models.

## 2. Related work

Multi-task optimization for MTL The recent works [4, 17–19, 26, 28, 35] have achieved impressive results in addressing task imbalance issues in MTL by directly or indirectly modifying conflicting task gradients. Specifically, some works [4, 17, 35] propose to form a new update gradient at each training step by directly altering gradients based on certain criteria. Other works [14, 18, 19, 26, 28] learn dynamic loss scale to balance different tasks during training, and thus indirectly altering the gradient of tasks. However, these methods only address GC when it occurs and do not proactively prevent it. In this paper, we sparsely train an MTL model, effectively reducing the incidence of GC.

Training with subset of parameters Several methods have already been proposed in single-task learning. Some of them select a subset of parameters based on a certain pre-defined rule, such as gradient [11, 40] and magnitude of parameters [16]. In addition to selecting parameters by hand design, the works in [25, 27, 34] automatically select the subset of parameters through optimization. Although sparse training has been extensively investigated in singletask learning, its application in MTL remains relatively unexplored. Sun et al. [30] and Calandriello et al. [2] learn to share information between tasks using a sparse model instead of sparse training. Differently, we research the gradient conflict via the sparse training perspective.

## 3. Approach

## 3.1. Background

Multi-task learning (MTL) aims to learn multiple tasks simultaneously within a single model. Formally, given $\left\{ \mathcal { T } _ { t } \right\} _ { t = 1 } ^ { T }$ tasks $\left( \geq 2 \right)$ and a model Θ with parameters $\Theta =$ $( \theta _ { \mathrm { { s h a } } } , \theta _ { \mathrm { { s e p } } } )$ , where $\theta _ { \mathrm { s h a } }$ and $\theta _ { \mathrm { s e p } }$ are shared parameter with all tasks and task-specific parameters $\theta _ { \mathrm { s e p } } ~ = ~ \left\{ \theta _ { \mathrm { s e p } } ^ { t } \right\} _ { t = 1 } ^ { T }$ respectively, the commonly used optimization method for MTL (referred to as Joint Train) is based on computing the average loss across all tasks with equal weights:

$$
\Theta^ {*} = \arg \min _ {\Theta} \mathcal {L} (\Theta), \tag {1}
$$

![](images/181fdf847defbff2257cbc112e8eaeef5a0078150f73a7ab9970c86ba904a27f.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(a) Joint Train

![](images/78bdab353d19f31be58e5b199f2ef20eba2ce388c0eb15d79666c4669f0017f6.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(b) Joint Train w/ ST

![](images/84be2e1c5859cf14ad001fbb6eeeb9b3cf4d45f3b9e20fc5fe6bf6973b211ad4.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(c) PCGrad

![](images/81e6d6e35e3eebbfc62840efef94a80def067a0df9e0b9182847511df86dc17c.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(d) PCGrad w/ ST  
Figure 2. Visualization of gradients change for different methods. $g _ { i }$ and $g _ { j }$ are two conflicting gradients, and the green arrow is the actual update vector. The process of sparse training can be interpreted as performing an orthographic/coordinate projection of conflicting gradients onto the subspace defined by the selected parameters, resulting in better alignment of the projected gradients.

$$
\mathcal {L} (\Theta) = \mathcal {L} (\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}}) = \frac {1}{T} \sum_ {t = 1} ^ {T} \mathcal {L} _ {t} (\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {t}) \quad (2)
$$

where each task t is associated with a corresponding loss function $\mathcal { L } _ { t } ( \theta _ { \mathrm { s h a } } , \theta _ { \mathrm { s e p } } ^ { t } )$

Gradient conflict (GC) However, optimizing all tasks by aggregating their losses indiscriminately (Eq. (2)) may lead to task competition, wherein certain tasks demonstrate improvement while others exhibit a decline compared to training them separately. From an optimization perspective, one of the reasons stems from conflicts in gradients. Formally, the update of task $\mathcal { T } _ { i }$ may potentially exert a detrimental impact on another task $\mathcal { T } _ { j }$ , namely:

$$
\Delta \mathcal {L} _ {j} = \mathcal {L} _ {j} (\hat {\theta} _ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {j}) - \mathcal {L} _ {j} (\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {j}), \tag {3}
$$

$$
\hat {\theta} _ {\mathrm{sha}} = \theta_ {\mathrm{sha}} - \alpha \mathbf {g} _ {i} \tag {4}
$$

where $\mathbf { g } _ { i } = \nabla _ { \theta _ { \mathrm { s h a } } } \mathcal { L } _ { i } ( \theta _ { \mathrm { s h a } } , \theta _ { \mathrm { s e p } } ^ { i } )$ is the gradient of loss on task $\mathcal { T } _ { i }$ with respect to $\theta _ { \mathrm { s h a } }$ and α is the learning rate. After the first-order Taylor approximation, Eq. (3) can be expressed as $- \alpha \mathbf { g } _ { i } \cdot \mathbf { g } _ { j } + o ( \alpha )$ . Gradient conflict arises when $\mathbf { g } _ { i } \cdot \mathbf { g } _ { j } < 0$ , leading to $\Delta \mathcal { L } _ { j } > 0$ , indicating that task $\mathcal { T } _ { i }$ has a detrimental impact on task $\tau _ { j }$ . Following [35], we provide the definition of gradient conflict:

Definition 1 (Gradient Conflict) Ifcos $\phi _ { i j } < 0 ,$ , where $\phi _ { i j }$ is the angle between gradients of two tasks $\mathbf { g } _ { i }$ and ${ \bf g } _ { j } \left( i \neq \right.$ $j )$ , then $\mathbf { g } _ { i }$ and ${ \bf { g } } _ { j }$ are deemed to exhibit gradient conflict.

Gradient manipulation To alleviate the issue of gradient conflict, gradient manipulation methods adjust conflicting gradients based on specific criteria and utilize these modified gradients for model updating. Instead of updating the model on the average gradient in Eq. (1) and Eq. (2):

$$
\nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} (\Theta) = \frac {1}{T} \sum_ {t = 1} ^ {T} \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {t} (\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {t}), \tag {5}
$$

the gradients of all tasks in gradient manipulation methods are modified as follows:

$$
\nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {\mathrm{gm}} (\Theta) = \frac {1}{T} \sum_ {t = 1} ^ {T} \boldsymbol {w} _ {t} \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {t} (\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {t}), \tag {6}
$$

$$
\boldsymbol {w} _ {t} = f \left(\nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {1}, \dots , \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {T}\right) \tag {7}
$$

where ${ \pmb w } _ { t }$ can be either pre-defined or dynamically computed for tasks via $f$ and thus achieve the aim of adjusting the task gradient [4, 17–19, 26, 28, 35]. However, the results of our experiment suggest that these methods can only modify gradients when conflicts occur, rather than proactively reducing the occurrence of GC during training, compared with Joint Train, as shown in Fig. 1.

## 3.2. Sparse training for multi-task learning

In this study, we investigate the gradient conflict commonly observed in multi-task learning from a novel perspective: sparse training, which selectively trains only a subset of the model parameters as opposed to full parameter training. This perspective is based on the intuition that by converting a high-dimensional space optimization problem into a lower-dimensional one, the complexity of optimization can be effectively reduced. Additionally, by limiting the impact of gradient updates to only a subset of parameters for each task instead of all parameters, potential interference between tasks can be mitigated.

Sparse training (ST) entails the initial parameter selection from the original model, and then updating only these parameters while keeping other parameters fixed during model training. To clarify potential misunderstandings regarding ST—often confused with sparse networks, where parameters are abandoned for model compression—we provide the following definition to ensure consistency and ease of understanding throughout this paper.

Definition 2 (Sparse Training) Given a model Θ and a binary mask matrix M indicating whether parameters in Θ are selected, where $M \ \in \ \mathbb { R } ^ { | \Theta | \times | \Theta | } , \ M _ { i i } \ \in \ \{ 0 , 1 \}$ and $M _ { i j } = 0 ~ ( \forall i \neq j )$ , the model is updated by $\hat { \Theta } = \Theta -$ $\alpha M \nabla _ { \Theta } \mathcal { L } ( \Theta )$ . We define this training strategy as sparse training.

Typically, the model architecture in multi-task learning includes a shared encoder as a feature extractor with taskspecific decoders for multiple tasks. Therefore, sparse training is used in the encoder, and full parameters training for the decoders. We detail how the mask is computed in section Sec. 3.4. We now apply sparse training for multi-task learning (Joint Train). The visualization of the gradient change can be viewed in Fig. 2 and the update with the reformulated gradient from Eq. (5) is as follows

$$
\begin{array}{l} \hat {\theta} _ {\mathrm{sha}} = \theta_ {\mathrm{sha}} - \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} (\Theta) \\ = \theta_ {\mathrm{sha}} - M \frac {1}{T} \sum_ {t = 1} ^ {T} \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {t} \left(\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {t}\right). \tag {8} \\ \end{array}
$$

Combination with gradient manipulation methods The application of sparse training can be seamlessly and effectively extended to improve various gradient manipulation methods in MTL. The update with the reformulated gradient from Eq. (6) is as follows

$$
\begin{array}{l} \hat {\theta} _ {\mathrm{sha}} = \theta_ {\mathrm{sha}} - \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {\mathrm{gm}} (\Theta) \\ = \theta_ {\mathrm{sha}} - M \frac {1}{T} \sum_ {t = 1} ^ {T} \boldsymbol {w} _ {t} \nabla_ {\theta_ {\mathrm{sha}}} \mathcal {L} _ {t} \left(\theta_ {\mathrm{sha}}, \theta_ {\mathrm{sep}} ^ {t}\right). \tag {9} \\ \end{array}
$$

## 3.3. Theoretical analysis for sparse training

After introducing sparse training into MTL, the optimization objective in Eq. (1) can be formed:

$$
\Theta^ {*} = \arg \min _ {\Theta} \mathcal {L} (\Theta), s. t. \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2} = 0, \tag {10}
$$

where $\theta _ { \mathrm { s h a } } ^ { \mathrm { i n } }$ is the initialized original model for θ and I is identity matrix. According to Lagrangian duality, Eq. (10) can be reformulated as:

$$
L = \min _ {\Theta} \max _ {\lambda} \mathcal {L} (\Theta) + \lambda \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2}. \tag {11}
$$

This can be transformed to optimize the upper bound L of regularized problem:

$$
L _ {r} = \min _ {\Theta} \mathcal {L} (\Theta) + \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2} \leq L. \tag {12}
$$

Please see the supplemental material for proof. Fu et al. [11] demonstrates that Eq. (12) has better stability and smaller generalization bound than only optimizing Eq. (1), resulting in better performance.

![](images/359fe0ebdd0f84ff7177bb602822b8f31b5aaeace1e7cc9c4883b0f563727416.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Node 1"] --> B["Node 2"]
  C["Node 3"] --> D["Node 4"]
  E["Node 5"] --> F["Node 6"]
  G["Node 7"] --> H["Node 8"]
  I["Node 9"] --> J["Node 10"]
  K["Node 11"] --> L["Node 12"]
  M["Node 13"] --> N["Node 14"]
  O["Node 15"] --> P["Node 16"]
  Q["Node 17"] --> R["Node 18"]
  S["Node 19"] --> T["Node 20"]
  U["Node 21"] --> V["Node 22"]
  W["Node 23"] --> X["Node 24"]
  Y["Node 25"] --> Z["Node 26"]
  AA["Node 27"] --> AB["Node 28"]
  AC["Node 29"] --> AD["Node 30"]
  AE["Node 31"] --> AF["Node 32"]
  AG["Node 33"] --> AH["Node 34"]
  AI["Node 35"] --> AJ["Node 36"]
  AK["Node 37"] --> AL["Node 38"]
  AM["Node 39"] --> AN["Node 40"]
  AO["Node 41"] --> AP["Node 42"]
  AQ["Node 43"] --> AR["Node 44"]
  AS["Node 45"] --> AT["Node 46"]
  AU["Node 47"] --> AV["Node 48"]
  AW["Node 49"] --> AX["Node 50"]
  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  F --> G
  G --> H
  H --> I
  I --> J
  J --> K
  K --> L
  L --> M
  M --> N
  N --> O
  O --> P
  P --> Q
  Q --> R
  R --> S
  S --> T
  T --> U
  U --> V
  V --> W
  W --> X
  X --> Y
  Y --> Z
  Z --> AA
  AA --> AB
  AB --> AC
  AC --> AD
  AD --> AE
  AE --> AF
  AF --> AG
  AG --> AH
  AH --> AI
  AI --> AJ
  AJ --> AK
  AK --> AL
  AL --> AM
  AM --> AN
  AN --> AO
  AO --> AP
  AP --> AQ
```
</details>

Figure 3. PSN. Top-1 highest-magnitude parameter among all input connections of each neuron is selected.

## 3.4. Parameter selection per neuron (PSN)

Several promising sparse training methods exist for singletask learning, but they are either time-consuming, requiring mask updates at each iteration [25, 27, 34], or memoryintensive due to gradient calculations for all parameters [11, 40]. In MTL, where multiple tasks are trained simultaneously, time efficiency is crucial. Thus, we adopt a onetime selection method, choosing parameters before training and keeping the mask fixed throughout. We consider the following two aspects for selection, magnitude of the parameter and involvement of all neurons in the network.

The magnitude of parameters Several studies have focused on model compression through the elimination of parameters with lower magnitudes [10, 13]. This highlights the significance of parameters with larger magnitudes in neural networks, which is consistent with our experimental findings (See Fig. 5c). The intuition behind this phenomenon lies in the fact that parameters with larger magnitudes exert a greater influence on altering neuron activation states through the activation function, wherein a neuron becomes active once the input surpasses a predefined threshold. Therefore, we exclusively select parameters with the highest magnitude for training multiple tasks.

The involvement of all neurons A simple idea is to select a certain proportion of parameters with the highest magnitude from the neural network (NN), but this may prevent some neurons from being engaged during training and hinder effective model training due to the dependence of the NN state on neuron activation. Motivated by studies highlighting distinct roles for different components in NN [9, 32, 40], we posit that engaging all neurons is crucial for effective model training. The rationale is that each neuron within the network possesses the inherent capability to finely adjust its activation state, thereby effectively adapting the overall NN state to the tasks, especially for learning multiple tasks simultaneously. Our experiments further substantiate this assertion, as shown in Fig. 5c.

PSN By integrating the two aspects, we select the top-K connections (weight/parameters) with the highest magnitude among all input connections for each neuron in the network (Please see Fig. 3 for top-1 example). This approach facilitates the training process for fitting tasks by ensuring that every neuron possesses activation potential, while parameters with higher magnitudes facilitate easier activation of neurons. In this paper, sparse training refers to using this method to select parameters and training the selected parameter, unless otherwise specified.

## 4. Experiments

Our experiments are conducted on comprehensive MTL benchmarks to evaluate the effectiveness of sparse training. First, we investigate if sparse training reduces gradient conflict. Subsequently, we examine its impact on performance across various MTL setups. The more details of the experiment are provided in Appendix D.

## 4.1. EXPERIMENTAL SETUP

Dateset Our MTL datasets are categorized into three groups: i) Dense prediction tasks: NYUv2 [6]: An indoor scene understanding dataset containing 1449 RGBD images with per-pixel labels across 13 classes, including semantic segmentation, depth estimation, and surface normal prediction. CityScapes [5]: 5000 street-view RGBD images with per-pixel annotations for 7-class semantic segmentation and depth estimation. ii) Multiple binary-classification tasks: CelebA [21]: 200,000 facial images of 10,000 celebrities, each with 40 binary attributes for facial features. We use the first 10 attributes for 10 binary classification tasks due to limited computation. iii) Multiple multi-class classification tasks: VTAB [36]: Containing 24 image understanding tasks with 1000 training examples per task. We use four tasks from it to create two multi-task benchmarks: Clevr: Simple 3D shapes with counting and depth prediction tasks. SmallNORB: Artificial objects with object azimuth and camera elevation prediction tasks.

Baseline We evaluate our approach using various baselines including i) single-task learning (STL): Each task is trained independently; ii) Joint Train: Training all tasks with average task loss; and 6 gradient manipulation methods including 3 direct and 3 indirect modification techniques. The former includes: iii) PCGrad: Projecting each task gradient onto the normal plane of other tasks [35]; iv) CAGrad: Enhancing the optimization of average loss by explicitly regulating the minimum decrease across tasks [17]; and v) GradDrop: Stochastically dropping specific dimensions of the gradients based on their level of conflict. The latter includes vi) MGDA: Identifying the same descent direction for each task [28]; vii) IMTL-G: Determining the update direction by ensuring equal projections on gradients [19]; viii) NashMTL: Treating MTL as a bargaining game to optimize all tasks [26].

Model We experiment with several architectures including: i) CNN-based: MTAN [20] incorporates an attention mechanism into the SegNet [1]. ii) Transformer-based. SAM [15] is a strong visual foundation model for segmentation. ViT-B/16 [8] and Swin Transformer [22] are vision classification models pre-trained on ImageNet21K [7]. All experiments were conducted on pre-trained SAM, ViT and Swin (except for randomly initialized MTAN), unless otherwise specified.

Evaluation i) Relative task drop $( \Delta m \% )$ Following [23], we evaluate the MTL overall performance for a baseline b by computing the average performance drop against STL s over $\{ \mathcal { T } _ { t } \} _ { t = 1 } ^ { T }$ tasks and $K _ { T _ { t } }$ metrics for each $\textstyle { \mathcal { T } } _ { t } \colon$ $\begin{array} { r } { \Delta m \% = \big ( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { K _ { \mathcal { T } _ { t } } } \sum _ { k = 1 } ^ { K _ { \mathcal { T } _ { t } } } ( - 1 ) ^ { \delta _ { k } } ( M _ { b } ^ { k } - M _ { s } ^ { k } ) / M _ { s } ^ { k } \big ) \times } \end{array}$ 100 where $M _ { b } ^ { k } , M _ { s } ^ { k }$ are the value of metrics k evaluated with b and s respectively. $\delta _ { k } = 1$ if the $M ^ { k }$ is higher the better and 0 otherwise. ii) Average incidence of $G C \left( p \% \right)$ We evaluate the extent of gradient conflict for a baseline by calculating the average incidence of GC over epochs during training. Given T tasks, E epochs, and I iterations per epoch, $\begin{array} { r } { p \bar { \% } \ = \ \frac { 1 } { E I } \sum _ { e = 1 } ^ { E } \sum _ { i = 1 } ^ { I } \bar { ( } N _ { g c } / N _ { a l l } ) \times 1 0 0 \ } \end{array}$ , where $N _ { g c }$ and $N _ { a l l }$ represent the number of occurrence of gradient conflicts between two tasks for all task combinations $\binom { T } { 2 }$ and the number of the combinations in each iteration during training, respectively.

## 4.2. Incidence of gradient conflict

We train a MTL model using the Joint Train and 6 stateof-the-art gradient manipulation techniques including PC-Grad, CAGrad, GradDrop, MGDA, IMTL-G and NashMTL and then introduce our sparse training strategy to these methods. Throughout the training process, we record instances of GC between any two tasks among all tasks for each training iteration and then calculate the average incidence of GC both over all epochs and the last 50% epochs. The observations of the SAM model on the NYU-v2 dataset are provided below. Similar results on other datasets and models are shown in Appendix F.3, Appendix F.5, Appendix F.7 and Appendix F.6.

Gradient manipulation methods cannot effectively reduce the incidence of gradient conflict The gradient manipulation methods [4, 17–19, 26, 28, 35] aim to modify conflicting gradients that are prevalent during the joint training of MTL. As shown in Tab. 1, the average incidence of GC using Joint train is 31.89% across all training epochs and 35.85% over the last 50% epochs. The incidence of GC cannot be effectively reduced by any gradient magnitude methods compared with the Joint train, as shown in Fig. 1 and Tab. 1. The reason is that these methods can only make the conflicting gradients not conflict when the GC occurs, rather than proactively prevent the occurrence of GC. The incidence of GC is even exacerbated by these methods, particularly MGDA showing a significant increase of 8.55% compared to Joint Train. Notably, these findings are consistent with [29], where they provide the distribution of the angles between the two task gradients.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average incidence of GC (%)</td></tr><tr><td>All epochs</td><td>Last 50% epochs</td></tr><tr><td>Joint Train</td><td>31.89</td><td>35.85</td></tr><tr><td>w/ ST</td><td>26.33 (5.56)</td><td>29.14 (6.71)</td></tr><tr><td>PCGrad</td><td>33.69</td><td>38.70</td></tr><tr><td>w/ ST</td><td>30.33 (3.36)</td><td>33.46 (5.24)</td></tr><tr><td>CAGrad</td><td>34.26</td><td>39.97</td></tr><tr><td>w/ ST</td><td>31.50 (2.76)</td><td>34.68 (5.29)</td></tr><tr><td>GradDrop</td><td>33.56</td><td>38.45</td></tr><tr><td>w/ ST</td><td>30.95 (2.61)</td><td>33.93 (4.52)</td></tr><tr><td>MGDA</td><td>40.44</td><td>44.77</td></tr><tr><td>w/ ST</td><td>40.05 (0.39)</td><td>42.34 (2.43)</td></tr><tr><td>IMTL-G</td><td>32.15</td><td>37.13</td></tr><tr><td>w/ ST</td><td>28.45 (3.70)</td><td>31.34 (5.79)</td></tr><tr><td>NashMTL</td><td>36.67</td><td>39.58</td></tr><tr><td>w/ ST</td><td>35.51 (1.16)</td><td>35.48 (4.10)</td></tr></table>

Table 1. Average incidence of GC between tasks for different methods. We compute the average incidence of GC over all epochs and the last 50% epochs during training SAM on NYUv2. The improvement by sparse training is provided in (•).

Sparse training effectively decreases the occurrence of gradient conflict As shown in Tab. 1, after combining sparse training with all methods, including Joint Train and gradient manipulation methods, the average incidence of gradient conflict is effectively reduced over all epochs. For example, ST in Joint Train reduced the incidence over all epochs by 5.56%. The phenomenon of gradient conflict reduction is consistently observed in nearly every training epoch, as illustrated in Fig. 4, which further demonstrates the effectiveness of ST for decreasing gradient conflict. In addition, all methods with ST exhibit a greater improvement in the average incidence of gradient conflict during the last 50% epochs compared to all epochs, which implies a greater level of prevention of gradient conflict with the progress of sparse training. For instance of NashMTL, there is a threefold improvement in the average incidence of gradient conflict during the last 50% epochs compared to all epochs.

## 4.3. Performance on diverse benchmarks

It is natural to investigate whether reducing gradient conflict during training through sparsity can enhance performance on common benchmarks. In this section, we present diverse benchmarks to demonstrate the effectiveness of ST.

![](images/627acf3a4efe35609ec8c7baff91f27ae885895aac24294cef25ce70f1b087b3.jpg)

<details>
<summary>line</summary>

| Epoch | Joint Train (%) | Joint Train w/ ST (%) |
| --- | --- | --- |
| 0 | ~52 | ~52 |
| 10 | ~24 | ~20 |
| 20 | ~28 | ~23 |
| 30 | ~27 | ~24 |
| 40 | ~29 | ~23 |
| 50 | ~31 | ~26 |
| 60 | ~33 | ~28 |
| 70 | ~35 | ~29 |
| 80 | ~38 | ~30 |
| 90 | ~40 | ~33 |
| 100 | ~41 | ~29 |
</details>

(a) Joint Train

![](images/5671921e1bd5380bbab7c4ecf2aa35b43cd27e83d9496853ee6ac056a454467c.jpg)

<details>
<summary>line</summary>

| Epoch | PCGrad (%) | PCGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~40 | ~38 |
| 5 | ~28 | ~29 |
| 10 | ~24 | ~26 |
| 15 | ~27 | ~27 |
| 20 | ~28 | ~25 |
| 25 | ~29 | ~26 |
| 30 | ~28 | ~27 |
| 35 | ~30 | ~28 |
| 40 | ~31 | ~29 |
| 45 | ~33 | ~31 |
| 50 | ~36 | ~30 |
| 55 | ~34 | ~32 |
| 60 | ~37 | ~31 |
| 65 | ~38 | ~33 |
| 70 | ~39 | ~34 |
| 75 | ~40 | ~35 |
| 80 | ~41 | ~36 |
| 85 | ~42 | ~38 |
| 90 | ~43 | ~40 |
| 95 | ~44 | ~41 |
| 100 | ~42 | ~37 |
</details>

(b) PCGrad  
Figure 4. The incidence of GC between tasks during training SAM on NYUv2 dataset. The top and bottom figures are Joint Train and PCGrad respectively. Please see Fig. 7 in Appendix F.2 for more results on other gradient manipulation methods.

Sparse training improves the performance for all stateof-the-art methods The performance of Joint Train and all gradient manipulation methods is consistently improved by sparse training, as demonstrated in Tab. 2 for NYUv2 benchmarks. Specifically, sparse training not only enhances overall task performance but also improves individual task performance for the majority of methods. For example, in Tab. 2, Joint Train demonstrates improvements across all individual tasks through sparse training. Similarly, as shown in Tab. 3, all methods exhibit notable improvements by sparse training on CelebA, Clevr, Small-NORB and CityScapes benchmarks.

Effectiveness on both pre-trained and randomly initialized models Our study primarily focuses on the sparse training for large pre-trained models, because leveraging prior knowledge from these models can be beneficial for MTL and our experimental results demonstrate that larger models exhibit a more severe gradient conflict, as shown in Fig. 5a. However, in order to ensure a fair comparison with related works that manipulate gradients in small and randomly initialized models, we also conduct experiments under the same setting as theirs to further demonstrate the effectiveness of sparse training. As shown in Tab. 3, we observe that even for the small randomly initialized models, the performance of joint training and all gradient manipulation methods is improved by sparse training. Please see Tab. 7 and Tab. 12 for the detailed results in the Appendix.

Generalization on different architectures and MTL tasks To evaluate the generalization across diverse architectures and MTL tasks, we conducted experiments on both CNN-based models and transformer-based models with varying visual MTL capabilities. Specifically, our MTL tasks encompassed visual classification (CelebA, Clevr and

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m\% \downarrow$ </td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within  $t^{\circ}$  ↑</td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>STL</td><td>58.62</td><td>79.20</td><td>0.3810</td><td>0.1553</td><td>19.29</td><td>12.64</td><td>46.37</td><td>72.19</td><td>80.73</td><td>-</td></tr><tr><td rowspan="2">Joint Trainw/ ST</td><td>59.09</td><td>79.61</td><td>0.3348</td><td>0.1360</td><td>22.34</td><td>16.33</td><td>35.46</td><td>64.02</td><td>75.20</td><td>6.763</td></tr><tr><td>60.03</td><td>79.96</td><td>0.3320</td><td>0.1353</td><td>21.98</td><td>15.92</td><td>36.69</td><td>64.92</td><td>75.82</td><td>5.314</td></tr><tr><td rowspan="2">PCGradw/ ST</td><td>59.18</td><td>80.12</td><td>0.3258</td><td>0.1323</td><td>21.81</td><td>15.72</td><td>36.92</td><td>65.49</td><td>76.26</td><td>4.584</td></tr><tr><td>59.37</td><td>80.33</td><td>0.3272</td><td>0.1330</td><td>21.53</td><td>15.39</td><td>38.02</td><td>66.09</td><td>76.71</td><td>3.741</td></tr><tr><td rowspan="2">CAGradw/ ST</td><td>59.78</td><td>80.16</td><td>0.3215</td><td>0.1305</td><td>19.92</td><td>13.40</td><td>43.87</td><td>70.47</td><td>79.59</td><td>-1.816</td></tr><tr><td>60.33</td><td>80.20</td><td>0.3232</td><td>0.1306</td><td>19.74</td><td>13.20</td><td>44.42</td><td>71.04</td><td>80.02</td><td>-2.423</td></tr><tr><td rowspan="2">GradDropw/ ST</td><td>59.02</td><td>79.80</td><td>0.3283</td><td>0.1321</td><td>22.03</td><td>15.95</td><td>36.42</td><td>64.90</td><td>75.79</td><td>5.323</td></tr><tr><td>59.74</td><td>80.32</td><td>0.3278</td><td>0.1322</td><td>21.81</td><td>15.63</td><td>37.40</td><td>65.50</td><td>76.15</td><td>4.329</td></tr><tr><td rowspan="2">MGDAw/ ST</td><td>37.43</td><td>67.58</td><td>0.4427</td><td>0.1810</td><td>19.23</td><td>12.61</td><td>46.43</td><td>72.35</td><td>80.87</td><td>9.162</td></tr><tr><td>41.60</td><td>69.96</td><td>0.4414</td><td>0.1778</td><td>19.22</td><td>12.61</td><td>46.44</td><td>72.29</td><td>80.80</td><td>7.791</td></tr><tr><td rowspan="2">IMTL-Gw/ ST</td><td>60.64</td><td>80.29</td><td>0.3324</td><td>0.1348</td><td>19.85</td><td>13.37</td><td>43.92</td><td>70.78</td><td>79.9</td><td>-1.537</td></tr><tr><td>60.35</td><td>80.18</td><td>0.3347</td><td>0.1350</td><td>19.65</td><td>13.15</td><td>44.66</td><td>71.25</td><td>80.20</td><td>-1.955</td></tr><tr><td rowspan="2">NashMTLw/ ST</td><td>59.42</td><td>80.20</td><td>0.3303</td><td>0.1341</td><td>19.90</td><td>13.39</td><td>43.86</td><td>70.65</td><td>79.72</td><td>-1.295</td></tr><tr><td>59.36</td><td>79.98</td><td>0.3278</td><td>0.1323</td><td>19.63</td><td>13.02</td><td>45.06</td><td>71.31</td><td>80.15</td><td>-2.384</td></tr></table>

Table 2. The test performance on NYU-v2 dataset training on SAM model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td>CelebA</td><td colspan="3">Clevr</td><td>SmallNORB</td><td>NYU-v2</td><td>CityScapes</td></tr><tr><td> $\Delta m\% \downarrow$ (F1)</td><td>Counting(Top 1 ↑)</td><td>Depth(Top 1 ↑)</td><td> $\Delta m\% \downarrow$ </td><td> $\Delta m\% \downarrow$ </td><td> $\Delta m\% \downarrow$ </td><td> $\Delta m\% \downarrow$ </td></tr><tr><td>STL</td><td>-</td><td>58.64</td><td>57.68</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="2">Joint Trainw/ ST</td><td>3.12</td><td>54.86</td><td>54.68</td><td>5.84</td><td>10.70</td><td>5.59</td><td>26.87</td></tr><tr><td>2.03</td><td>61.80</td><td>54.81</td><td>-0.21</td><td>10.11</td><td>2.49</td><td>17.48</td></tr><tr><td rowspan="2">PCGradw/ ST</td><td>1.70</td><td>49.01</td><td>53.39</td><td>11.93</td><td>9.99</td><td>3.97</td><td>19.96</td></tr><tr><td>1.42</td><td>59.01</td><td>55.29</td><td>1.75</td><td>9.71</td><td>1.98</td><td>19.22</td></tr><tr><td rowspan="2">CAGradw/ ST</td><td>1.96</td><td>49.33</td><td>53.67</td><td>11.41</td><td>10.50</td><td>0.20</td><td>16.26</td></tr><tr><td>1.23</td><td>58.51</td><td>55.27</td><td>2.19</td><td>10.22</td><td>-2.76</td><td>8.88</td></tr><tr><td rowspan="2">GradDropw/ ST</td><td>1.18</td><td>49.02</td><td>52.88</td><td>12.36</td><td>11.73</td><td>3.58</td><td>20.34</td></tr><tr><td>0.83</td><td>58.87</td><td>54.07</td><td>2.94</td><td>10.76</td><td>1.38</td><td>17.45</td></tr><tr><td rowspan="2">MGDAw/ ST</td><td>-0.41</td><td>49.56</td><td>55.97</td><td>9.22</td><td>10.15</td><td>1.38</td><td>6.91</td></tr><tr><td>-1.08</td><td>58.23</td><td>56.91</td><td>1.02</td><td>9.79</td><td>-3.18</td><td>3.17</td></tr><tr><td rowspan="2">IMTL-Gw/ ST</td><td>0.97</td><td>54.99</td><td>54.51</td><td>5.87</td><td>10.19</td><td>-0.76</td><td>10.65</td></tr><tr><td>0.19</td><td>61.05</td><td>56.73</td><td>-1.24</td><td>10.15</td><td>-3.18</td><td>7.10</td></tr><tr><td rowspan="2">NashMTLw/ ST</td><td>3.59</td><td>47.04</td><td>53.07</td><td>13.89</td><td>10.84</td><td>-4.04</td><td>6.68</td></tr><tr><td>3.22</td><td>58.61</td><td>54.97</td><td>2.37</td><td>9.57</td><td>-5.11</td><td>3.99</td></tr></table>

Table 3. The test performance on CelebA, Clevr, SmallNORB, NYU-v2 and CityScapes dataset. CelebA is trained on Swin Transformer. Clevr and SmallNORB are trained on ViT. NYU-v2 and CityScapes are trained on MTAN. We only present ∆m% for limited space. Please see Tab. 7, Tab. 12 and Tab. 11 for detailed results in supplemental materials. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

SmallNORB) and visual dense prediction (NYU-v2 and CityScapes). For the former, we utilized Swin Transformer and ViT as backbones for multiple binary classification tasks (Tab. 3) and two multi-class classification tasks (Tab. 3, and Tab. 11 in Appendix), respectively. The latter involved predicting dense masks for each task, necessitating an encoder-decoder structure to generate corresponding masks. We explored two types of structures: a symmetrical encoder-decoder structure with a CNN-based model, e.g. MTAN (Tab. 3, and Tabs. 7 and 12 in Appendix) and an asymmetric structure with a heavy-weight encoder and a light-weight decoder using a transformer-based model, e.g.

![](images/5845e8e7b8a28ba7e4bb29057970c0b966066a0c8eb2a33cb629a4e130458a90.jpg)

<details>
<summary>line</summary>

| Epoch | Swin / Tiny (%) | Swin / Base (%) | Swin / Large (%) |
| --- | --- | --- | --- |
| 0 | ~50.0 | ~49.5 | ~52.0 |
| 10 | ~38.0 | ~40.5 | ~42.0 |
| 20 | ~37.5 | ~40.0 | ~41.5 |
| 30 | ~36.0 | ~39.5 | ~41.0 |
| 40 | ~35.5 | ~39.5 | ~41.0 |
| 50 | ~36.0 | ~39.5 | ~41.5 |
| 60 | ~36.5 | ~39.5 | ~41.0 |
| 70 | ~37.0 | ~40.0 | ~41.5 |
| 80 | ~38.0 | ~41.0 | ~42.0 |
| 90 | ~38.5 | ~41.5 | ~42.0 |
| 100 | ~39.5 | ~42.5 | ~42.5 |
</details>

(a) Different model size

![](images/ac66e215c80b1d39ebbe14bb15cd721ce2b0973c964ef54b6c745355141010bd.jpg)

<details>
<summary>line</summary>

| Trainable parameters (%) | ST (MTAN) | Joint Train (MTAN) | ST (SAM) | Joint Train (SAM) |
| --- | --- | --- | --- | --- |
| 10 | — | ~5.7 | ~11.6 | ~6.8 |
| 15 | ~18.2 | ~5.7 | ~7.7 | ~6.8 |
| 22 | ~13.7 | ~5.7 | ~5.7 | ~6.8 |
| 27 | ~12.1 | ~5.7 | ~5.6 | ~6.8 |
| 30 | ~9.1 | ~5.7 | ~5.4 | ~6.8 |
| 33 | ~9.5 | ~5.7 | ~5.6 | ~6.8 |
| 37 | ~7.6 | ~5.7 | ~5.8 | ~6.8 |
| 41 | ~6.0 | ~5.7 | ~5.9 | ~6.8 |
| 45 | ~5.8 | ~5.7 | ~6.0 | ~6.8 |
| 50 | ~6.2 | ~5.7 | ~6.2 | ~6.8 |
| 53 | ~6.2 | ~5.7 | ~6.4 | ~6.8 |
| 58 | — | ~5.7 | ~7.8 | ~6.8 |
| 63 | ~2.5 | ~5.7 | — | ~6.8 |
| 68 | ~3.0 | ~5.7 | — | ~6.8 |
| 73 | — | ~5.7 | — | ~6.8 |
| 76 | — | ~5.7 | — | ~6.8 |
| 84 | ~4.7 | ~5.7 | — | ~6.8 |
</details>

(b) Different num. of parameters

![](images/8bdf20ef42b515046f4dd7e30a581f7d1d2b9a14c9225164187297c749f7b6fa.jpg)

<details>
<summary>radar</summary>

| Dimension | Random | Global | Reverse | Ours | Joint Train |
| --- | --- | --- | --- | --- | --- |
| Dep AbsErr | ~0.8 | ~0.4 | ~0.2 | ~0.9 | ~0.6 |
| Seq-ByAcc | ~0.7 | ~0.2 | ~0.2 | ~0.9 | ~0.2 |
| Seq-modL | ~0.6 | ~0.2 | ~0.2 | ~0.9 | ~0.2 |
| Ncr-30 | ~0.3 | ~0.1 | ~0.1 | ~0.9 | ~0.1 |
| Ncr-22.5 | ~0.1 | ~0.1 | ~0.1 | ~0.9 | ~0.1 |
| Ncr-11.25 | ~0.4 | ~0.1 | ~0.4 | ~0.9 | ~0.1 |
| Avg Median | ~0.4 | ~0.1 | ~0.4 | ~0.9 | ~0.1 |
| Nor mean | ~0.6 | ~0.1 | ~0.1 | ~0.9 | ~0.1 |
| Dep RelErr | ~0.7 | ~0.4 | ~0.7 | ~0.9 | ~0.7 |
</details>

(c) Different sparse methods  
Figure 5. Ablation study for Joint Train with NYU-v2 dataset. (a) The average incidence of GC during joint training on different sizes of Swin transformers. Please see the numerical statics for all epochs in Tab. 8 in Appendix F.4. (b) The different number of trainable parameters for MTAN and SAM models. (C) Different sparse methods training on SAM. Metrics for all tasks are min-max normalized. Please see Tab. 5 for detailed results in Appendix F.1.

SAM (Tab. 2 in Appendix). As shown in these tables, the efficacy of sparse training in improving all baselines across various architectures and MTL tasks underscores its robust generalization capability.

## 4.4. Ablation study

The larger the model, the more severe gradient conflicts. In this paper, we focus more on investigating the gradient conflict in the pre-trained large models as larger models demonstrated a more severe phenomenon of gradient conflict. This can be observed in Fig. 5a, where Swin/Tiny demonstrates significantly less gradient conflict compared to Swin/Base and Swin/Large. It is worth noting that although larger models tend to experience more severe gradient conflicts, this does not necessarily lead to inferior performance compared to smaller models with milder gradient conflicts. This discrepancy can be attributed to differences in model capacity and the prior knowledge embedded through pre-training. Nevertheless, this observation underscores the importance of exploring methods to mitigate gradient conflicts in larger models. Within the same model architecture and size, reducing gradient conflicts has been shown to improve performance, as evidenced by works such as [17, 35]. Addressing severe gradient conflicts in larger models may thus unlock their full potential, enabling better utilization of their capacity and capabilities.

Effortless search for the number of trainable parameters. We explore the effect of trainable parameter numbers for ST. The results in Fig. 5b show that the pre-trained model (SAM) and the randomly initialized model (MTAN) have different optimal trainable parameter numbers. MTAN requires ∼60% of the parameters, while SAM needs only ∼30%, leveraging information from the pre-trained model. In our paper, most of the experiments use these proportions for ST and achieve better results (please see Tab. 4 in Appendix D.1 for the detailed number). Additionally, ST offers a wide range of trainable parameter options that outperform Joint Train, which implies that hyperparameter search for the number of trainable parameters becomes effortless. Specifically, both models have a ∼40% probability of yielding superior outcomes.

Effectiveness for both higher magnitude and neurallevel selection. We investigate various parameter selection approaches: Random: Randomly selecting parameters from the network; Global: Choosing parameters with the highest magnitude from the whole network instead of the input connections of each neuron in the network (Ours); Reverse: Selecting parameters with the lowest magnitude among input connections of each neuron. For a fair comparison, we maintain the same selected number. The results in Fig. 5c indicate that higher magnitude values are superior to lower ones (Ours > Reverse). Furthermore, it is crucial to evenly select parameters from the entire network (Ours > Random > Global), as Ours ensure that the parameters of input connection for each neuron are selected, and Random guarantees an equal proportion of parameters is selected in each block of the network, whereas this is not the case for Global (see Fig. 6 for detailed statistics in Appendix).

## 5. Conclusion

In this paper, the occurrence of gradient conflict in multitask learning is extensively investigated from a novel perspective: sparse training. Extensive experiments demonstrate that sparse training transferring high-dimensional space into low-dimensional space effectively reduces the incidence of gradient conflict during training while preserving the integrity of the original model. Furthermore, combining sparse training with other gradient manipulation methods significantly improves performance for multi-task learning.

## References

[1] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017. 2, 5  
[2] Daniele Calandriello, Alessandro Lazaric, and Marcello Restelli. Sparse multi-task reinforcement learning. Advances in neural information processing systems, 27, 2014. 2  
[3] Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Shangzhan Zhang, Yan Wang, Zejian Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam fails to segment anything? – samadapter: Adapting sam in underperformed scenes: Camouflage, shadow, and more, 2023. 2  
[4] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020. 1, 2, 3, 5  
[5] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 5  
[6] Camille Couprie, Clement Farabet, Laurent Najman, and´ Yann LeCun. Indoor semantic segmentation using depth information, 2013. 5  
[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5  
[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 5  
[9] Fenglei Fan, Jinjun Xiong, Mengzhou Li, and Ge Wang. On interpretability of artificial neural networks: A survey. IEEE Transactions on Radiation and Plasma Medical Sciences, 5: 741–760, 2020. 4  
[10] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018. 4  
[11] Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, and Nigel Collier. On the effectiveness of parameter-efficient fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12799–12807, 2023. 2, 4  
[12] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International conference on machine learning, pages 3854–3863. PMLR, 2020. 2  
[13] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural net-  
work. Advances in neural information processing systems, 28, 2015. 4  
[14] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018. 2  
[15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 5  
[16] Franc¸ois Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush. Block pruning for faster transformers. arXiv preprint arXiv:2109.04838, 2021. 2  
[17] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021. 1, 2, 3, 5, 8  
[18] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization, 2023. 1, 2, 8  
[19] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2021. 1, 2, 3, 5  
[20] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019. 2, 5  
[21] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015. 5  
[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 2, 5  
[23] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1851–1860, 2019. 5  
[24] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3994–4003, 2016. 2  
[25] Hesham Mostafa and Xin Wang. Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. In International Conference on Machine Learning, pages 4646–4655. PMLR, 2019. 2, 4  
[26] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multitask learning as a bargaining game, 2022. 1, 2, 3, 5  
[27] Victor Sanh, Thomas Wolf, and Alexander Rush. Movement pruning: Adaptive sparsity by fine-tuning. Advances in neural information processing systems, 33:20378–20389, 2020. 2, 4  
[28] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in Neural Information Processing Systems, 31, 2018. 1, 2, 3, 5  
[29] Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. Recon: Reducing conflicting gradients from the root for multi-task learning. arXiv preprint arXiv:2302.11289, 2023. 1, 6  
[30] Tianxiang Sun, Yunfan Shao, Xiaonan Li, Pengfei Liu, Hang Yan, Xipeng Qiu, and Xuanjing Huang. Learning sparse sharing architectures for multiple tasks. In Proceedings of the AAAI conference on artificial intelligence, pages 8936– 8943, 2020. 2  
[31] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3614–3633, 2021. 2  
[32] Zifeng Wang, Shao-Lun Huang, Ercan E Kuruoglu, Jimeng Sun, Xi Chen, and Yefeng Zheng. Pac-bayes information bottleneck. arXiv preprint arXiv:2109.14509, 2021. 4  
[33] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 675–684, 2018. 2  
[34] Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. Raise a child in large language model: Towards effective and generalizable fine-tuning. arXiv preprint arXiv:2109.05687, 2021. 2, 4  
[35] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020. 1, 2, 3, 5, 8  
[36] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019. 5  
[37] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 34 (12):5586–5609, 2021. 1, 2  
[38] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Zequn Jie, Xiang Li, and Jian Yang. Joint task-recursive learning for semantic segmentation and depth estimation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 235–251, 2018. 2  
[39] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4106–4115, 2019. 2  
[40] Zhi Zhang, Qizhe Zhang, Zijun Gao, Renrui Zhang, Ekaterina Shutova, Shiji Zhou, and Shanghang Zhang. Gradientbased parameter selection for efficient fine-tuning. arXiv preprint arXiv:2312.10136, 2023. 2, 4

# Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective

Supplementary Material

In this supplemental material, we provide extra details about the content in the main body of the paper. First, we provide detailed proof for Equation (12) in Appendix A. Then, we discuss the limitations of our work in Appendix B. Moreover, the broader impacts of our research are discussed in Appendix C. In addition, we present all hyperparameters and experiment settings in Appendix F for a better understanding of the experiments and reproduction to the readers. We also provide the extended related works in Appendix E. Finally, the additional experiment results are demonstrated in Appendix F, which further indicate the effectiveness of our proposed method and the consistency with the claim in the main body of the paper.

## A. Proof for Equation (12)

According to Lagrangian duality, Eq. (10) can be reformulated as:

$$
\begin{array}{l} L = \min _ {\Theta} \max _ {\lambda} \mathcal {L} (\Theta) + \lambda \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2} \\ \geq \max _ {\lambda} \min _ {\Theta} \mathcal {L} (\Theta) + \lambda \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2} \\ \geq \min _ {\Theta} \mathcal {L} (\Theta) + \| (I - M) (\theta_ {\mathrm{sha}} - \theta_ {\mathrm{sha}} ^ {\mathrm{in}}) \| ^ {2} \\ \end{array}
$$

where λ is the Lagrangian multiplier.

## B. Limitations

Due to the limited computational resources, we employ grid searches in the Joint train method to determine the optimal hyperparameter for the number of trainable parameters, which is then utilized across all gradient manipulation methods. However, it is possible that these methods may benefit from a more optimized hyperparameter selection for the number of trainable parameters. Furthermore, sparse training can effectively mitigate gradient conflicts between tasks in MTL by reducing the dimensionality of parameter space and limiting their impact on updates between tasks. The regularization constitutes one of the theory’s reasons. Nevertheless, we anticipate that our future research will contribute to a deeper comprehension of multi-task learning and subsequently enhance the performance of MTL.

## C. Broader Impacts

The nature of our research does not directly contribute to societal impact; however, like any machine learning paper, it has the potential to adversely affect society through automation and job displacement. While it is challenging to predict specific risks, similar to any technology, inadequate regulation may lead to an exacerbation of social and economic inequality. The positive aspect lies in the potential environmental impact of our work, as multi-task learning enables information sharing among tasks, thereby reducing data requirements and further minimizing energy consumption during training.

## D. Detailed experiment setting

## D.1. Number of trainable parameters

We provide the number of trainable parameters for all experiments conducted in our paper. As shown in Table 4, most of them have the same percentage of trainable parameters within a model across different methods. In addition, in general, we can observe that sparse training for the pretrained model needs ∼30% while that for random initialized model needs ∼60%.

## D.2. Implementation details

Following the work of Nash [26], we apply all gradient manipulation techniques to the gradients of the shared weights. We set the hyperparameter c of CAGrad to 0.4, as it has been reported to yield optimal performance for NYUv2 and Cityscapes datasets [17]. The experiments were conducted on the A100 80G GPU. Typically, training with SAM using NYU-v2 and Swin with CelebA requires approximately 1 day for a gradient manipulation method. Training ViT with SmallNORB takes around 18 minutes for a gradient manipulation method, while training ViT with Clevr takes about 30 minutes. On the other hand, training MTAN with NYUv2 demands roughly 18 hours for a gradient manipulation method, whereas training MTAN with CityScapes necessitates approximately 12 hours.

SAM, ViT, Swin For all methods, including single-task learning, the gradient manipulation method, and our sparse training, we employed a batch size of 3 and searched for the optimal learning rate from the set {2e-4, 5e-5}, and then the best results are reported. The reason is that we find the optimal learning rate for sparse training is bigger than that for full parameters training. Therefore, for most methods, the optimal learning rate for sparse training is 2e-4 and that for the full parameters training is 5e-5. we also use data augmentations for all methods, following [17]. The batch size used is set to be 3 for NYUv2 dataset, and 256 for CelebA, and 128 for SmallNORB and Clevr.

<table><tr><td rowspan="3">Method</td><td colspan="4">Pre-trained model</td><td colspan="2">Random initialized model</td></tr><tr><td>SAM</td><td>Swin</td><td colspan="2">ViT</td><td colspan="2">MTAN</td></tr><tr><td>NYU-v2</td><td>CelebA</td><td>Clevr</td><td>SmallNORB</td><td>NYU-v2</td><td>CityScapes</td></tr><tr><td>Joint Train w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>29.38</td><td>62.19</td><td>76.02</td></tr><tr><td>PCGrad w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>19.63</td><td>62.19</td><td>76.02</td></tr><tr><td>CAGrad w/ ST</td><td>30.97</td><td>72.85</td><td>29.38</td><td>29.38</td><td>62.19</td><td>76.02</td></tr><tr><td>GradDrop w/ ST</td><td>30.97</td><td>49.58</td><td>29.38</td><td>29.38</td><td>62.19</td><td>76.02</td></tr><tr><td>MGDA w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>29.38</td><td>62.19</td><td>62.19</td></tr><tr><td>IMTL-G w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>29.38</td><td>62.19</td><td>62.19</td></tr><tr><td>NashMTL w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>29.38</td><td>62.19</td><td>83.48</td></tr><tr><td>FAMO w/ ST</td><td>30.97</td><td>37.60</td><td>29.38</td><td>29.38</td><td>62.19</td><td>62.19</td></tr></table>

Table 4. Number of trainable parameters. The values in the table are expressed as percentages (%). As we select Top-K input parameters among all input connections for each neuron, therefore the same K might lead to different percentages of trainable parameters for different models. For example, K=300 results in 30.97% in SAM, 37.60% in Swin, and 29.38% in ViT for the pre-trained model.

MTAN Following the works in [17, 26], we incorporate data augmentations during training for both Joint Train method and all gradient manipulation methods. Each method is trained for 200 epochs with an initial learning rate of 0.0001, which is then reduced to 0.00005 after 100 epochs. For Multi-Task Learning (MTL) methods, we utilize a Multi-Task Attention Network (MTAN) [20] based on SegNet architecture proposed by [1]. Similar to previous studies [17], the STL baseline refers to training task-specific SegNet models. The batch size used is set to be 2 for NYUv2 dataset and 8 for CityScapes dataset respectively. To align with prior research on MTL including [1, 17, 35], we report the test performance averaged over the last 10 epochs.

## E. Extended related work

Multi-task learning Multi-task learning [37] aims to improve the overall performance of all tasks. In this work, we focus on a conventional setup of multi-task learning [31]: given a single input, multi-task models perform different and related predictions, such as segmentation, depth and surface normal. In other words, the input is shared by different tasks. In this paper, we roughly divide existing MTL into two categories:

i) Multi-task optimization. Recent works [4, 17–19, 26, 28, 35] provide impressive results in solving the task imbalance during optimization. The rationale behind these works is that re-weighting all task gradients or losses helps multi-task models reduce conflicting gradients among tasks [17, 28]. Specifically, some works [4, 19, 28] propose to form a new update gradient at each optimization by linearly combining task gradients. Other works [14, 18] learn dynamic loss scale to balance different tasks during training. However, it is challenging to scale up most existing optimization works to giant foundation models due to non-trivial computational and memory costs. In this paper, we propose a neuron-based parameter selection to sparsely fine-tune the pre-trained model, which boosts the performance of most optimization methods.

ii) Multi-task architecture In this branch, multi-task methods design different architectures to improve the exchanging or sharing of information among tasks [31]. Regarding where tasks interact, multi-task architectures are separated into encoder-focused and decoder-focused. The former shares the information in the encoder by the transformation of activations among tasks [24], learnable taskspecific attention modules [20], branching networks for similar tasks [12] and so on. The latter recursively uses task predictions to improve overall performance [33, 38, 39]. However, these architectures still suffer from the task imbalance issue during multi-task optimization. In this paper, our work focuses on boosting multi-task optimization. As one of the multi-task optimization methods, our method can seamlessly generalize to different backbone models.

Training with subset of parameters several methods already proposed in single-task learning. several methods select a subset of parameters based on a certain pre-defined rule, such as gradient [11, 40] and magnitude of parameters [16]. In addition to selecting parameters by hand design, [25, 27, 34] automatically select the subset of parameters through optimization. Although sparse training has been extensively investigated in single-task learning, its application in multi-task learning remains relatively unexplored. [2, 30] learning to share information between tasks using a sparse model. Differing from them, in this paper, we systematically research the gradient conflict via the sparse training perspective.

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta \mathrm{m}\% \downarrow$ </td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within  $t^{\circ}$  ↑</td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>Random</td><td>59.85</td><td>80.09</td><td>0.3357</td><td>0.1359</td><td>22.17</td><td>16.12</td><td>36.08</td><td>64.50</td><td>75.56</td><td>6.014</td></tr><tr><td>Global</td><td>59.53</td><td>79.38</td><td>0.3380</td><td>0.1373</td><td>22.32</td><td>16.32</td><td>35.62</td><td>63.93</td><td>75.16</td><td>6.855</td></tr><tr><td>Reverse</td><td>59.35</td><td>79.57</td><td>0.3417</td><td>0.1396</td><td>22.31</td><td>16.25</td><td>35.98</td><td>64.07</td><td>75.16</td><td>6.960</td></tr><tr><td>Ours</td><td>60.03</td><td>79.96</td><td>0.3320</td><td>0.1353</td><td>21.98</td><td>15.92</td><td>36.69</td><td>64.92</td><td>75.82</td><td>5.314</td></tr></table>

Table 5. Different sparse training methods on SAM model with NYU-v2 datasets.

![](images/555a29815e0f24b8a72c3b2b684d5f4db94a8bbad038465bc20189a0afb4a13e.jpg)

<details>
<summary>pie</summary>

| Category | Value (%) |
| --- | --- |
| Block 0 | 8.33 |
| Block 1 | 8.33 |
| Block 2 | 8.33 |
| Block 3 | 8.33 |
| Block 4 | 8.33 |
| Block 5 | 8.33 |
| Block 6 | 8.33 |
| Block 7 | 8.33 |
| Block 8 | 8.33 |
| Block 9 | 8.33 |
| Block 10 | 8.33 |
| Block 11 | 8.33 |
</details>

(a) Ours

![](images/6dba5670f2f34c9c3efe18c410cd6ad39dfdf8910c254178fce17e76431dd778.jpg)

<details>
<summary>pie</summary>

| Category | Value (%) |
| --- | --- |
| Block 8 | 8.34 |
| Block 11 | 8.34 |
| Block 1 | 8.34 |
| Block 6 | 8.34 |
| Block 4 | 8.34 |
| Block 7 | 8.33 |
| Block 0 | 8.33 |
| Block 5 | 8.33 |
| Block 2 | 8.33 |
| Block 3 | 8.33 |
| Block 9 | 8.33 |
| Block 10 | 8.32 |
</details>

(b) Random

![](images/0d8edbc75c390c838ac6297d7a0f86a00c211e329f5dbe0e170bf06f434fe546.jpg)

<details>
<summary>pie</summary>

| Category | Value (%) |
| --- | --- |
| Block 1 | 11.18 |
| Block 2 | 11.03 |
| Block 3 | 10.39 |
| Block 4 | 9.70 |
| Block 5 | 8.83 |
| Block 11 | 8.72 |
| Block 0 | 8.42 |
| Block 10 | 7.24 |
| Block 6 | 7.21 |
| Block 7 | 6.00 |
| Block 9 | 5.76 |
| Block 8 | 5.52 |
</details>

(c) Global  
Figure 6. The distribution of selected trainable parameters for different sparse training methods over different blocks. The experiments are conducted on SAM model with NYU-v2 dataset.

## F. Detailed experiment results

In this section, we provide the detailed experiment results conducted in the main body of our paper, including the average incident of gradient conflict, the incident of gradient conflict for all epochs, and visualization of the gradient conflict for Joint Train and all gradient manipulation methods.

## F.1. Ablation study

The detailed results for various sparse methods are provided in Tab. 5, which is the full version of Fig. 5c. It can be observed that, with the exception of Pix Acc in segmentation, our sparse method outperforms other methods. In addition, we provide the distribution of the selected parameters using different sparse training over different blocks of the model. As shown in Fig. 6, the parameters selected by our sparse training method and Random are evenly distributed over the whole network. As for Global selecting the parameters with the highest magnitude, the distribution of selected parameters is largely different over different blocks

## F.2. NYU-v2 on SAM

The incidence of gradient conflict for Joint Train and gradient manipulation method over all epochs are shown in

Fig. 7, which is the full version of Fig. 4 in the main body of the paper.

## F.3. NYU-v2 on MTAN

We also conduct experiments on MTAN with NYU-v2 dataset. MTAN is a random initialized model. As we can see in Tab. 6, even for the random initialized model, sparse training can also reduce the incidence of gradient conflict. The visualization of the occurrence of gradient conflict for each epoch is shown in Fig. 9 and the average incidence of gradient conflict across all epochs for different methods is shown in Fig. 8. As for the performance of the overall tasks on NYU-v2, the sparse training improves not only the overall performance (∆m%) but also the performance of each task for all methods including Joint Train and all gradient manipulation methods, as shown in Tab. 7. In addition, following [26], we conduct the experiments three times with three different seeds. The mean ± std is presented in Tab. 7, we can observe that the sparse training is robust to the random seed.

## F.4. NYU-v2 on Swin

In order to investigate how the incidence of gradient conflict changes with varying model sizes, we conduct experiments on Swin/Tiny, Swin/Base and Swin/Large through the Joint Train. As depicted in Tab. 8, there is an observed increase in the incidence of gradient conflict as the model size increases. Additionally, the performance of tasks improves as the model size increases Tab. 9.

![](images/5907fdc400f89328f382dc6e183661ffb66fb171723dea63b50ca068ae6b12bd.jpg)

<details>
<summary>line</summary>

| Epoch | Joint Train (%) | Joint Train w/ ST (%) |
| --- | --- | --- |
| 0 | ~40 | ~51 |
| 10 | ~24 | ~20 |
| 20 | ~33 | ~28 |
| 30 | ~26 | ~24 |
| 40 | ~29 | ~22 |
| 50 | ~34 | ~27 |
| 60 | ~31 | ~28 |
| 70 | ~38 | ~30 |
| 80 | ~44 | ~33 |
| 90 | ~39 | ~36 |
| 100 | ~42 | ~29 |
</details>

(a) Joint Train

![](images/bfc080eaab2036fb1fb8891d7a4eb2d048ef6a63ab512e470ecb666de40166e5.jpg)

<details>
<summary>line</summary>

| Epoch | GradDrop (%) | GradDrop w/ ST (%) |
| --- | --- | --- |
| 0 | ~40 | ~37.5 |
| 10 | ~23 | ~26 |
| 20 | ~22 | ~28 |
| 30 | ~30 | ~29 |
| 40 | ~33 | ~27 |
| 50 | ~33 | ~31 |
| 60 | ~38 | ~31 |
| 70 | ~38 | ~33 |
| 80 | ~42 | ~37 |
| 90 | ~44 | ~38 |
| 100 | ~39 | ~37 |
</details>

(b) GradDrop

![](images/7b094d725e965689c2aa12bd3afee33b94d2829d78054638be189faa3743a7b9.jpg)

<details>
<summary>line</summary>

| Epoch | IMTL-G (%) | IMTL-G w/ ST (%) |
| --- | --- | --- |
| 0 | ~36 | ~54 |
| 10 | ~23 | ~24 |
| 20 | ~27 | ~28 |
| 30 | ~29 | ~27 |
| 40 | ~28 | ~26 |
| 50 | ~33 | ~27 |
| 60 | ~34 | ~28 |
| 70 | ~38 | ~31 |
| 80 | ~38 | ~33 |
| 90 | ~43 | ~36 |
| 100 | ~43 | ~38 |
</details>

(c) IMTL-G

![](images/78f7742060374ce29d26a377c6291784cbe910f4a554dc25715fa50fa0303d4f.jpg)

<details>
<summary>line</summary>

| Epoch | CAGrad (%) | CAGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~39 | ~39 |
| 5 | ~23 | ~28 |
| 10 | ~20 | ~26 |
| 15 | ~27 | ~27 |
| 20 | ~24 | ~25 |
| 25 | ~28 | ~27 |
| 30 | ~31 | ~29 |
| 35 | ~33 | ~28 |
| 40 | ~30 | ~29 |
| 45 | ~34 | ~31 |
| 50 | ~36 | ~29 |
| 55 | ~35 | ~30 |
| 60 | ~41 | ~32 |
| 65 | ~37 | ~34 |
| 70 | ~44 | ~36 |
| 75 | ~38 | ~35 |
| 80 | ~45 | ~37 |
| 85 | ~41 | ~38 |
| 90 | ~46 | ~40 |
| 95 | ~48 | ~42 |
| 100 | ~46 | ~39 |
</details>

(d) CAGrad

![](images/004c2fbabb2a528cf1131d85c7f8deabc437d6d563aed3631312f5842b2138f0.jpg)

<details>
<summary>line</summary>

| Epoch | PCGrad (%) | PCGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~40 | ~38 |
| 10 | ~25 | ~26 |
| 20 | ~29 | ~27 |
| 30 | ~28 | ~26 |
| 40 | ~31 | ~29 |
| 50 | ~34 | ~28 |
| 60 | ~36 | ~31 |
| 70 | ~38 | ~33 |
| 80 | ~41 | ~34 |
| 90 | ~42 | ~38 |
| 100 | ~43 | ~37 |
</details>

(e) PCGrad

![](images/1bf6290969686148bf72b554eb643ba798ddfcc56f57bda748d280ae8ea29191.jpg)

<details>
<summary>line</summary>

| Epoch | NashMTL (%) | NashMTL w/ ST (%) |
| --- | --- | --- |
| 0 | ~38 | ~39 |
| 5 | ~31 | ~41 |
| 10 | ~38 | ~52 |
| 15 | ~30 | ~27 |
| 20 | ~26 | ~33 |
| 25 | ~39 | ~48 |
| 30 | ~35 | ~32 |
| 35 | ~34 | ~34 |
| 40 | ~36 | ~50 |
| 45 | ~41 | ~32 |
| 50 | ~45 | ~34 |
| 55 | ~38 | ~37 |
| 60 | ~43 | ~31 |
| 65 | ~38 | ~35 |
| 70 | ~42 | ~32 |
| 75 | ~39 | ~37 |
| 80 | ~41 | ~36 |
| 85 | ~41 | ~40 |
| 90 | ~43 | ~42 |
| 95 | ~46 | ~44 |
| 100 | ~41 | ~39 |
</details>

(f) NashMTL

![](images/c183122475ea8eeda4b18300c68f978c72c31aaf4ecddd0d5702662dcd44723f.jpg)

<details>
<summary>line</summary>

| Epoch | MGDA (%) | MGDA w/ ST (%) |
| --- | --- | --- |
| 0 | ~39 | ~40 |
| 5 | ~27 | ~31 |
| 10 | ~26 | ~32 |
| 15 | ~31 | ~34 |
| 20 | ~38 | ~41 |
| 25 | ~30 | ~37 |
| 30 | ~38 | ~39 |
| 35 | ~37 | ~35 |
| 40 | ~41 | ~39 |
| 45 | ~44 | ~40 |
| 50 | ~38 | ~35 |
| 55 | ~45 | ~47 |
| 60 | ~37 | ~48 |
| 65 | ~51 | ~42 |
| 70 | ~46 | ~41 |
| 75 | ~52 | ~43 |
| 80 | ~41 | ~43 |
| 85 | ~49 | ~48 |
| 90 | ~43 | ~43 |
| 95 | ~51 | ~46 |
| 100 | ~47 | ~43 |
</details>

(g) MGDA

Figure 7. The number of occurrence gradient conflictions between tasks during training SAM on NYUv2 dataset.  
![](images/2583b7ca319c90fd46ef8c16c44db374ddded4be8ad57b3f597118783a061eff.jpg)

<details>
<summary>bar</summary>

| Category | Baseline (All epochs) | Baseline w/ ST (All epochs) | Baseline (Last 50% epochs) | Baseline w/ ST (Last 50% epochs) |
| --- | --- | --- | --- | --- |
| Joint Train | ~36.0 | ~33.8 | ~39.8 | ~36.4 |
| PCGrad | ~35.7 | ~34.0 | ~39.5 | ~37.2 |
| CAGrad | ~37.2 | ~34.1 | ~40.9 | ~37.0 |
| GradDrop | ~36.3 | ~34.4 | ~39.7 | ~37.0 |
| MGDA | ~37.7 | ~37.1 | ~42.1 | ~41.2 |
| IMTL-G | ~37.1 | ~35.8 | ~41.2 | ~39.1 |
| NashMTL | ~37.1 | ~35.8 | ~40.7 | ~38.9 |
</details>

Figure 8. The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on MTAN model with NYU-v2 datasets was evaluated using various methods, including joint training and gradient manipulation techniques.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average incidence of GC (%)</td></tr><tr><td>All epochs</td><td>Last 50% epochs</td></tr><tr><td>Joint Train</td><td>36.01</td><td>39.87</td></tr><tr><td>w/ ST</td><td>33.86 (2.15)</td><td>36.45 (3.42)</td></tr><tr><td>PCGrad</td><td>35.71</td><td>39.51</td></tr><tr><td>w/ ST</td><td>34.05 (1.66)</td><td>37.25 (2.26)</td></tr><tr><td>CAGrad</td><td>37.21</td><td>40.93</td></tr><tr><td>w/ ST</td><td>34.14 (3.07)</td><td>37.04 (3.89)</td></tr><tr><td>GradDrop</td><td>36.37</td><td>39.71</td></tr><tr><td>w/ ST</td><td>34.42 (1.95)</td><td>37.10(2.61)</td></tr><tr><td>MGDA</td><td>37.76</td><td>42.1</td></tr><tr><td>w/ ST</td><td>37.15 (0.61)</td><td>41.25 (0.85)</td></tr><tr><td>IMTL-G</td><td>37.14</td><td>41.22</td></tr><tr><td>w/ ST</td><td>35.81 (1.33)</td><td>39.17 (2.05)</td></tr><tr><td>NashMTL</td><td>37.19</td><td>40.79</td></tr><tr><td>w/ ST</td><td>35.83 (1.36)</td><td>39.0 (1.79)</td></tr></table>

Table 6. Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (•). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training MTAN on NYUv2.

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta \mathrm{m}\% \downarrow$ </td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within  $t^{\circ} \uparrow$ </td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>STL</td><td>38.30</td><td>63.76</td><td>0.6754</td><td>0.2780</td><td>25.01</td><td>19.21</td><td>30.14</td><td>57.20</td><td>69.15</td><td>-</td></tr><tr><td rowspan="2">Joint Train w/ ST</td><td>39.29</td><td>65.33</td><td>0.5493</td><td>0.2263</td><td>28.15</td><td>23.96</td><td>22.09</td><td>47.50</td><td>61.08</td><td>5.59</td></tr><tr><td>41.04 (±0.28)</td><td>66.05 (±0.12)</td><td>0.5417 (±0.0008)</td><td>0.2232 (±0.0011)</td><td>27.40(±0.05)</td><td>22.90(±0.12)</td><td>23.58(±0.13)</td><td>49.59(±0.14)</td><td>63.01(±0.09)</td><td>2.49(±0.11)</td></tr><tr><td rowspan="2">PCGrad w/ ST</td><td>38.06</td><td>64.64</td><td>0.5550</td><td>0.2325</td><td>27.41</td><td>22.80</td><td>23.86</td><td>49.83</td><td>63.14</td><td>3.97</td></tr><tr><td>40.49 (±0.32)</td><td>66.17(±0.23)</td><td>0.5441 (±0.0023)</td><td>0.2264 (±0.0030)</td><td>27.09 (±0.08)</td><td>22.55(±0.03)</td><td>24.22(±0.12)</td><td>50.34(±0.17)</td><td>63.63(±0.12)</td><td>1.98(±0.12)</td></tr><tr><td rowspan="2">CAGrad w/ ST</td><td>39.79</td><td>65.49</td><td>0.5486</td><td>0.2250</td><td>26.31</td><td>21.58</td><td>25.61</td><td>52.36</td><td>65.58</td><td>0.20</td></tr><tr><td>39.93(±0.33)</td><td>66.19(±0.16)</td><td>0.5299(±0.0025)</td><td>0.2097(±0.0038)</td><td>25.71(±0.02)</td><td>20.70(±0.03)</td><td>26.86(±0.13)</td><td>54.22(±0.15)</td><td>67.30(±0.13)</td><td>-2.76(±0.10)</td></tr><tr><td rowspan="2">GradDrop w/ ST</td><td>39.39</td><td>65.12</td><td>0.5455</td><td>0.2279</td><td>27.48</td><td>22.96</td><td>23.38</td><td>49.44</td><td>62.87</td><td>3.58</td></tr><tr><td>40.84(±0.35)</td><td>66.84(±0.24)</td><td>0.5288(±0.0021)</td><td>0.2209(±0.0021)</td><td>27.18(±0.03)</td><td>22.56(±0.07)</td><td>24.10(±0.11)</td><td>50.33(±0.14)</td><td>63.67(±0.13)</td><td>1.38(±0.12)</td></tr><tr><td rowspan="2">MGDA w/ ST</td><td>30.47</td><td>59.90</td><td>0.6070</td><td>0.2555</td><td>24.88</td><td>19.45</td><td>29.18</td><td>56.88</td><td>69.36</td><td>1.38</td></tr><tr><td>32.42(±0.41)</td><td>61.61(±0.21)</td><td>0.5851(±0.0015)</td><td>0.2239 (±0.0032)</td><td>24.35(±0.02)</td><td>18.61(±0.03)</td><td>31.14(±0.12)</td><td>58.63(±0.15)</td><td>70.62(±0.13)</td><td>-3.09(±0.14)</td></tr><tr><td rowspan="2">IMTL-G w/ ST</td><td>39.35</td><td>65.60</td><td>0.5426</td><td>0.2256</td><td>26.02</td><td>21.19</td><td>26.20</td><td>53.13</td><td>66.24</td><td>-0.76</td></tr><tr><td>40.73(±0.33)</td><td>66.00(±0.17)</td><td>0.5219(±0.0015)</td><td>0.2100(±0.0021)</td><td>25.6(±0.05)</td><td>20.64(±0.04)</td><td>26.81(±0.16)</td><td>54.38(±0.15)</td><td>67.49(±0.12)</td><td>-3.18(±0.11)</td></tr><tr><td rowspan="2">NashMTL w/ ST</td><td>40.13</td><td>65.93</td><td>0.5261</td><td>0.2171</td><td>25.26</td><td>20.08</td><td>28.40</td><td>55.47</td><td>68.15</td><td>-4.04</td></tr><tr><td>39.75(±0.21)</td><td>66.45(±0.05)</td><td>0.5156(±0.0006)</td><td>0.2121(±0.0009)</td><td>24.96(±0.01)</td><td>19.80(±0.05)</td><td>28.80(±0.11)</td><td>56.20(±0.10)</td><td>68.93(±0.09)</td><td>-5.11(±0.07)</td></tr></table>

Table 7. The test performance on NYU-v2 dataset training on MTAN model, involving three tasks: semantic segmentation, depth estimation and surface normal. The result is the mean over three random seeds (std is presented in (± •). The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

## F.5. CelebA on Swin

Following [26], we train CelebA on Swin for only 30 epochs, because there are many more tasks in this dataset compared with other datasets, which leads to a significant increase in computation. As we can observe in Tab. 10, most of the methods including Joint Train and gradient manipulation methods can be improved by sparse training in terms of average incidence of gradient conflict between tasks over epochs. It is noted that the improvement by sparse training here is not significant, which is because of the limited training epoch. Specifically, as shown in Tab. 1 and Tab. 6, our sparse training improves more for later epochs. As for the performance of CelebA on Swin, please refer to Tab. 3. The visualization for the occurrence of gradient conflict for each epoch and average incidence of gradient conflict over all epochs for different methods, including Joint Train and all gradient manipulation methods, are shown in Fig. 10 and Fig. 11

![](images/f873315a2358cae1827cb972d60d1a97bd230a3c2faa5e3d22da44a298b2d65b.jpg)

<details>
<summary>line</summary>

| Epoch | Joint Train (%) | Joint Train w/ ST (%) |
| --- | --- | --- |
| 0 | ~34.5 | ~33.5 |
| 25 | ~28.5 | ~29.5 |
| 50 | ~34.5 | ~33.5 |
| 75 | ~36.5 | ~35.5 |
| 100 | ~39.5 | ~37.5 |
| 125 | ~40.5 | ~38.5 |
| 150 | ~41.5 | ~39.5 |
| 175 | ~40.5 | ~37.5 |
| 200 | ~39.5 | ~36.5 |
</details>

(a) Joint Train

![](images/37d11ce1d1d806019d32813306081040039481b9c7a7b8475591149349090cbc.jpg)

<details>
<summary>line</summary>

| Epoch | GradDrop (%) | GradDrop w/ ST (%) |
| --- | --- | --- |
| 0 | ~36.5 | ~35.5 |
| 25 | ~31.5 | ~31.0 |
| 50 | ~34.0 | ~33.5 |
| 75 | ~36.0 | ~35.0 |
| 100 | ~38.0 | ~37.0 |
| 125 | ~40.0 | ~38.0 |
| 150 | ~40.0 | ~38.0 |
| 175 | ~40.0 | ~38.0 |
| 200 | ~40.0 | ~38.0 |
</details>

(b) GradDrop

![](images/745539955dc48b2a8c285f1e881235636789861f1b7907eba815d322305f11dd.jpg)

<details>
<summary>line</summary>

| Epoch | IMTL-G (%) | IMTL-G w/ ST (%) |
| --- | --- | --- |
| 0 | ~36.5 | ~33.5 |
| 25 | ~27.5 | ~28.5 |
| 50 | ~34.5 | ~33.5 |
| 75 | ~37.5 | ~36.5 |
| 100 | ~40.5 | ~39.5 |
| 125 | ~41.5 | ~40.5 |
| 150 | ~41.5 | ~40.5 |
| 175 | ~42.5 | ~41.5 |
| 200 | ~41.5 | ~41.5 |
</details>

(c) IMTL-G

![](images/fdfba037f21d5857108ea75dda2f1b42ade0cabdd07f34ccd197cae1cf41efde.jpg)

<details>
<summary>line</summary>

| Epoch | CAGrad (%) | CAGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~34 | ~34 |
| 25 | ~28 | ~28 |
| 50 | ~35 | ~33 |
| 75 | ~37 | ~36 |
| 100 | ~40 | ~36 |
| 125 | ~41 | ~38 |
| 150 | ~41 | ~37 |
| 175 | ~42 | ~37 |
| 200 | ~39 | ~39 |
</details>

(d) CAGrad

![](images/0c1703ce7769918ee1c8a097c20bad60a30f415890b4d02fb1f7814c21771e09.jpg)

<details>
<summary>line</summary>

| Epoch | PCGrad (%) | PCGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~36 | ~34 |
| 25 | ~28 | ~29 |
| 50 | ~34 | ~33 |
| 75 | ~36 | ~35 |
| 100 | ~38 | ~37 |
| 125 | ~40 | ~38 |
| 150 | ~39 | ~37 |
| 175 | ~40 | ~38 |
| 200 | ~40 | ~38 |
</details>

(e) PCGrad

![](images/80d7baaeb40f598800b322c1601e635ce81cf0d51c7b44b71062657bea2c1646.jpg)

<details>
<summary>line</summary>

| Epoch | NashMTL (%) | NashMTL w/ ST (%) |
| --- | --- | --- |
| 0 | ~24 | ~34 |
| 25 | ~28 | ~29 |
| 50 | ~36 | ~35 |
| 75 | ~37 | ~37 |
| 100 | ~38 | ~38 |
| 125 | ~41 | ~40 |
| 150 | ~40 | ~39 |
| 175 | ~41 | ~40 |
| 200 | ~42 | ~41 |
</details>

(f) NashMTL

![](images/915ed03c8016cb244ec906f1bfa4249719b2e34c197053db0739640b8f42d2f9.jpg)

<details>
<summary>line</summary>

| Epoch | MGDA (%) | MGDA w/ ST (%) |
| --- | --- | --- |
| 0 | ~34.5 | ~26.0 |
| 25 | ~29.0 | ~28.5 |
| 50 | ~33.0 | ~32.5 |
| 75 | ~36.0 | ~36.5 |
| 100 | ~38.5 | ~38.0 |
| 125 | ~41.5 | ~41.0 |
| 150 | ~42.0 | ~41.5 |
| 175 | ~43.5 | ~42.5 |
| 200 | ~42.0 | ~41.5 |
</details>

(g) MGDA  
Figure 9. The number of occurrence gradient conflictions between tasks during tuning MTAN on NYUv2 dataset.

## F.6. SmallNORB on ViT

SmallNORB is a much more difficult benchmark compared to other benchmarks in this paper. It comprises artificial objects observed under varying conditions and includes two tasks: object azimuth and camera-elevation prediction. As shown in Tab. 11, even for the STL, the Top 1 accuracy only achieves ∼30%, therefore, we use Top 5 as an extra metric here. We observed that even for this difficult task, sparse training can still achieve better performance compared with Joint Train and all gradient manipulation methods.

![](images/5e4e575c71e57ba6bd02f03e3e71076eb0eb1a24209b829a4bc288ad6fd87180.jpg)

<details>
<summary>line</summary>

| Epoch | Joint Train (%) | Joint Train w/ ST (%) |
| --- | --- | --- |
| 0 | ~48.7 | ~47.8 |
| 1 | ~47.9 | ~46.0 |
| 2 | ~47.1 | ~46.0 |
| 3 | ~46.3 | ~45.2 |
| 4 | ~45.4 | ~44.9 |
| 5 | ~45.4 | ~43.6 |
| 6 | ~44.6 | ~43.6 |
| 7 | ~45.5 | ~44.1 |
| 8 | ~45.9 | ~44.8 |
| 9 | ~46.1 | ~45.2 |
| 10 | ~46.5 | ~45.2 |
| 11 | ~46.0 | ~45.9 |
| 12 | ~47.1 | ~46.0 |
| 13 | ~47.6 | ~46.7 |
| 14 | ~47.2 | ~47.1 |
| 15 | ~47.1 | ~46.7 |
| 16 | ~47.9 | ~47.5 |
| 17 | ~48.2 | ~47.6 |
| 18 | ~48.5 | ~47.7 |
| 19 | ~48.5 | ~48.3 |
| 20 | ~48.9 | ~48.3 |
| 21 | ~48.6 | ~49.2 |
| 22 | ~49.7 | ~48.9 |
| 23 | ~49.8 | ~48.9 |
| 24 | ~49.5 | ~48.9 |
| 25 | ~49.7 | ~49.3 |
| 26 | ~49.3 | ~49.3 |
| 27 | ~49.8 | ~48.9 |
| 28 | ~49.3 | ~49.9 |
| 29 | ~50.0 | N/A |
</details>

(a) Joint Train

![](images/a4ca5b842e12e5e7da3e271e9f11a37f52bc0582359c821b16b8f101a11979de.jpg)

<details>
<summary>line</summary>

| Epoch | GradDrop (%) | GradDrop w/ ST (%) |
| --- | --- | --- |
| 0 | ~47.8 | ~48.2 |
| 1 | ~46.0 | ~46.0 |
| 2 | ~45.9 | ~45.8 |
| 3 | ~45.7 | ~45.4 |
| 4 | ~45.2 | ~44.1 |
| 5 | ~45.3 | ~44.4 |
| 6 | ~45.2 | ~44.5 |
| 7 | ~45.4 | ~45.0 |
| 8 | ~45.7 | ~44.8 |
| 9 | ~45.1 | ~45.0 |
| 10 | ~46.2 | ~45.9 |
| 11 | ~45.9 | ~46.0 |
| 12 | ~46.7 | ~46.6 |
| 13 | ~46.5 | ~46.5 |
| 14 | ~47.3 | ~47.3 |
| 15 | ~47.3 | ~47.3 |
| 16 | ~47.7 | ~46.8 |
| 17 | ~47.8 | ~47.6 |
| 18 | ~48.1 | ~47.6 |
| 19 | ~48.0 | ~47.7 |
| 20 | ~48.5 | ~48.3 |
| 21 | ~48.5 | ~48.9 |
| 22 | ~48.5 | ~49.3 |
| 23 | ~49.5 | ~49.1 |
| 24 | ~49.0 | ~48.8 |
| 25 | ~49.3 | ~49.3 |
| 26 | ~49.3 | ~49.3 |
| 27 | ~49.5 | ~49.3 |
| 28 | ~49.6 | ~49.3 |
| 29 | >50.0 | ~49.5 |
</details>

(b) GradDrop

![](images/23ca6d964a709f57a6280d6ef31d6119389f339cecef51d1974d493d0680dcea.jpg)

<details>
<summary>line</summary>

| Epoch | IMTL-G (%) | IMTL-G w/ ST (%) |
| --- | --- | --- |
| 0 | ~49.5 | ~47.2 |
| 1 | ~48.6 | ~46.9 |
| 2 | ~47.1 | ~45.8 |
| 3 | ~47.1 | ~44.8 |
| 4 | ~46.1 | ~43.8 |
| 5 | ~45.5 | ~44.2 |
| 6 | ~45.7 | ~43.7 |
| 7 | ~45.3 | ~42.6 |
| 8 | ~44.9 | ~41.9 |
| 9 | ~45.1 | ~42.6 |
| 10 | ~45.8 | ~40.7 |
| 11 | ~45.0 | ~42.7 |
| 12 | ~45.5 | ~43.7 |
| 13 | ~45.3 | ~43.3 |
| 14 | ~44.7 | ~42.0 |
| 15 | ~45.5 | ~42.1 |
| 16 | ~45.1 | ~44.0 |
| 17 | ~44.7 | ~42.7 |
| 18 | ~46.0 | ~45.2 |
| 19 | ~47.6 | ~44.0 |
| 20 | ~45.2 | ~45.1 |
| 21 | ~48.4 | ~46.0 |
| 22 | ~47.7 | ~46.7 |
| 23 | ~47.5 | ~46.5 |
| 24 | ~48.8 | ~47.8 |
| 25 | ~49.7 | ~48.3 |
| 26 | ~50.1 | ~48.9 |
| 27 | ~50.3 | ~49.3 |
| 28 | ~50.0 | ~49.7 |
| 29 | — | ~49.6 |
</details>

(c) IMTL-G

![](images/d00148cbc6854970395a3a99e935d538e5da0c2cdd5fb83842d0e76aa48806e0.jpg)

<details>
<summary>line</summary>

| Epoch | CAGrad (%) | CAGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~48.3 | ~48.1 |
| 2 | ~45.4 | ~45.6 |
| 3 | ~45.3 | ~45.6 |
| 4 | ~44.7 | ~44.2 |
| 5 | ~44.9 | ~45.2 |
| 6 | ~44.0 | ~44.3 |
| 7 | ~44.6 | ~44.7 |
| 8 | ~45.1 | ~44.7 |
| 9 | ~45.8 | ~45.3 |
| 10 | ~47.2 | ~46.5 |
| 11 | ~47.3 | ~46.6 |
| 12 | ~47.4 | ~46.7 |
| 13 | ~48.2 | ~49.1 |
| 14 | ~47.4 | ~49.6 |
| 15 | ~48.6 | ~49.0 |
| 16 | ~48.3 | ~49.6 |
| 17 | ~49.3 | ~50.9 |
| 18 | ~49.6 | ~50.0 |
| 19 | ~48.9 | ~50.3 |
| 20 | ~49.5 | ~50.7 |
| 21 | ~50.0 | ~51.4 |
| 22 | ~50.5 | ~51.5 |
| 23 | ~50.9 | ~50.7 |
| 24 | ~51.3 | ~50.3 |
| 25 | ~52.2 | ~50.1 |
| 26 | ~51.7 | ~50.3 |
| 27 | ~52.0 | ~51.1 |
| 28 | N/A | ~51.7 |
</details>

(d) CAGrad

![](images/e1b6ba781161af063cc8a4d3e3a7fc3cd569f033c7d2d9128b53394211e71184.jpg)

<details>
<summary>line</summary>

| Epoch | PCGrad (%) | PCGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~48.2 | ~47.9 |
| 1 | ~47.5 | ~47.9 |
| 2 | ~45.7 | ~46.5 |
| 3 | ~45.8 | ~45.8 |
| 4 | ~44.7 | ~45.0 |
| 5 | ~44.6 | ~44.6 |
| 6 | ~45.1 | ~43.7 |
| 7 | ~44.6 | ~43.9 |
| 8 | ~45.5 | ~45.2 |
| 9 | ~45.9 | ~45.9 |
| 10 | ~47.0 | ~46.0 |
| 11 | ~46.3 | ~45.5 |
| 12 | ~47.4 | ~46.3 |
| 13 | ~47.2 | ~46.2 |
| 14 | ~47.6 | ~46.8 |
| 15 | ~47.3 | ~47.3 |
| 16 | ~48.2 | ~47.8 |
| 17 | ~49.3 | ~48.0 |
| 18 | ~49.0 | ~47.7 |
| 19 | ~49.2 | ~48.2 |
| 20 | ~50.1 | ~48.5 |
| 21 | ~50.8 | ~48.8 |
| 22 | ~51.3 | ~49.2 |
| 23 | ~51.8 | ~49.0 |
| 24 | ~52.1 | ~49.8 |
| 25 | ~52.3 | ~49.9 |
| 26 | ~52.2 | ~49.7 |
| 27 | ~53.3 | ~49.7 |
| 28 | ~52.7 | ~49.8 |
| 29 | ~53.1 | ~50.3 |
</details>

(e) PCGrad

![](images/2b0f18c25fe629d521a647213da98a3fec3780db61e0791940f23cc062b817d5.jpg)

<details>
<summary>line</summary>

| Epoch | NashMTL (%) | NashMTL w/ ST (%) |
| --- | --- | --- |
| 0 | ~88.3 | ~90.5 |
| 1 | ~87.6 | ~88.2 |
| 2 | ~86.1 | ~87.1 |
| 3 | ~86.7 | ~86.5 |
| 4 | ~85.7 | ~86.2 |
| 5 | ~86.2 | ~86.1 |
| 6 | ~85.7 | ~85.5 |
| 7 | ~85.9 | ~86.0 |
| 8 | ~84.8 | ~85.9 |
| 9 | ~86.5 | ~84.0 |
| 10 | ~85.9 | ~85.9 |
| 11 | ~86.3 | ~85.1 |
| 12 | ~85.1 | ~86.5 |
| 13 | ~86.4 | ~84.7 |
| 14 | ~83.0 | ~85.0 |
| 15 | ~86.5 | ~85.9 |
| 16 | ~85.9 | ~84.6 |
| 17 | ~84.9 | ~85.7 |
| 18 | ~86.3 | ~83.9 |
| 19 | ~86.1 | ~86.5 |
| 20 | ~87.7 | ~87.9 |
| 21 | ~88.2 | ~88.1 |
| 22 | ~88.2 | ~88.3 |
| 23 | ~88.7 | ~89.0 |
| 24 | ~88.7 | ~87.9 |
| 25 | ~87.6 | ~88.3 |
| 26 | ~88.5 | ~89.2 |
| 27 | ~89.0 | ~89.3 |
| 28 | ~89.0 | ~90.0 |
| 29 | ~89.3 | ~89.3 |
</details>

(f) NashMTL

![](images/5d7c7bd8af65968c7899e2eb74f18a435eb5f472c0484ab5371e41d8ad710173.jpg)

<details>
<summary>line</summary>

| Epoch | MGDA (%) | MGDA w/ ST (%) |
| --- | --- | --- |
| 0 | ~48.4 | ~48.4 |
| 1 | ~46.9 | ~47.0 |
| 2 | ~45.9 | ~46.7 |
| 3 | ~45.1 | ~46.2 |
| 4 | ~43.3 | ~46.0 |
| 5 | ~43.0 | ~43.8 |
| 6 | ~43.6 | ~45.0 |
| 7 | ~44.1 | ~43.3 |
| 8 | ~43.7 | ~44.6 |
| 9 | ~41.8 | ~44.6 |
| 10 | ~43.0 | ~44.1 |
| 11 | ~42.5 | ~41.8 |
| 12 | ~40.8 | ~43.7 |
| 13 | ~42.9 | ~42.0 |
| 14 | ~42.3 | ~42.7 |
| 15 | ~42.3 | ~41.5 |
| 16 | ~45.0 | ~43.6 |
| 17 | ~46.5 | ~42.3 |
| 18 | ~44.2 | ~44.2 |
| 19 | ~44.1 | ~41.3 |
| 20 | ~43.8 | ~45.4 |
| 21 | ~43.8 | ~43.8 |
| 22 | ~43.3 | ~44.1 |
| 23 | ~46.4 | ~41.9 |
| 24 | ~41.3 | ~44.6 |
| 25 | ~46.7 | ~42.9 |
| 26 | ~46.7 | ~41.6 |
| 27 | ~47.7 | ~45.9 |
| 28 | ~48.8 | ~48.2 |
| 29 | ~48.2 | ~47.5 |
</details>

(g) MGDA

Figure 10. The number of occurrence gradient conflictions between tasks during tuning Swin on CelebA dataset.  
![](images/c4cb3d2ad56e4aed4bdf148cfe5ec61e8630e281c5f1fbc68a992d5330dbd162.jpg)

<details>
<summary>bar</summary>

| Method | Baseline (All epochs) | Baseline w/ ST (All epochs) | Baseline (Last 50% epochs) | Baseline w/ ST (Last 50% epochs) |
| --- | --- | --- | --- | --- |
| Joint Train | ~47.6 | ~46.9 | ~48.8 | ~48.5 |
| PCGrad | ~48.5 | ~47.3 | ~50.8 | ~48.9 |
| CAGrad | ~48.2 | ~48.3 | ~50.2 | ~50.4 |
| GradDrop | ~47.4 | ~47.1 | ~48.7 | ~48.6 |
| MGDA | ~44.5 | ~44.3 | ~45.6 | ~44.3 |
| IMTL-G | ~46.9 | ~45.0 | ~47.7 | ~46.3 |
| NashMTL | ~46.8 | ~46.8 | ~47.6 | ~47.3 |
</details>

Figure 11. The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on Swin mode with CelebA datasets was evaluated using various methods, including joint training and gradient manipulation techniques.

<table><tr><td>Model / Size</td><td>Average incidence of GC (%)</td></tr><tr><td>Swin / Tiny</td><td>37.42</td></tr><tr><td>Swin / Base</td><td>40.34</td></tr><tr><td>Swin / Large</td><td>41.84</td></tr></table>

Table 8. The average incidence of gradient conflict across all epochs during joint training with NYU-v2 on different sizes of Swin transformer.

## F.7. CityScapes on MTAN

We also conduct experiments on MTAN with CityScapes dataset. MTAN is a random initialized model. As we can see in Tab. 13, even for the random initialized model, sparse training can also reduce the incidence of gradient conflict. The reduction in the incidence of gradient conflict for CityScapes is observed to be comparatively smaller than that for NYU-v2. This discrepancy can be attributed to the fact that CityScapes, which involves only two tasks, has a lower likelihood of encountering gradient conflicts between tasks compared to NYU-v2, which encompasses three tasks. The visualization of the occurrence of gradient conflict for each epoch is shown in Fig. 12 and the average incidence of gradient conflict across all epochs for different methods is shown in Fig. 13 . As for the performance of the overall tasks on CityScapes, the sparse training improves all methods including Joint Train and all gradient manipulation methods, as shown in Tab. 12.

## F.8. FAMO

FAMO [18] is an approximation method for gradient manipulation by using the history of loss to compute the current task weight. We also try our sparse training with FAMO on NYU-v2, CelebA, Clevr, SmallORB datasets with ViT, SAM, MTAN and Swin models. As shown in Tab. 14, Tab. 16, Tab. 17 and Tab. 15, even for the approximation method, sparse training method achieves the best results and further show the effectiveness of our sparse training methods.

<table><tr><td rowspan="3">Model</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within t° ↑</td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>Swin/Tiny</td><td>55.22</td><td>76.54</td><td>0.3746</td><td>0.1542</td><td>27.47</td><td>21.70</td><td>27.81</td><td>52.40</td><td>64.05</td></tr><tr><td>Swin/Base</td><td>59.60</td><td>79.16</td><td>0.3419</td><td>0.1388</td><td>25.88</td><td>19.74</td><td>31.23</td><td>56.24</td><td>67.32</td></tr><tr><td>Swin/Large</td><td>61.34</td><td>80.28</td><td>0.3321</td><td>0.1345</td><td>25.09</td><td>18.73</td><td>33.05</td><td>58.12</td><td>68.86</td></tr></table>

Table 9. The test performance on NYU-v2 dataset jointly training on Swin models.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average incidence of GC (%)</td></tr><tr><td>All epochs</td><td>Last 50% epochs</td></tr><tr><td>Joint Train</td><td>47.61</td><td>48.78</td></tr><tr><td>w/ ST</td><td>46.96 (0.65)</td><td>48.48 (0.30)</td></tr><tr><td>PCGrad</td><td>48.48</td><td>50.83</td></tr><tr><td>w/ ST</td><td>47.24 (1.24)</td><td>48.88 (1.95)</td></tr><tr><td>CAGrad</td><td>48.21</td><td>50.23</td></tr><tr><td>w/ ST</td><td>48.33 (-0.12)</td><td>50.40(-0.17)</td></tr><tr><td>GradDrop</td><td>47.36</td><td>48.72</td></tr><tr><td>w/ ST</td><td>47.13 (0.23)</td><td>48.57 (0.15)</td></tr><tr><td>MGDA</td><td>44.56</td><td>45.65</td></tr><tr><td>w/ ST</td><td>44.30 (0.26)</td><td>44.26(1.39)</td></tr><tr><td>IMTL-G</td><td>46.89</td><td>47.77</td></tr><tr><td>w/ ST</td><td>45.03 (1.86)</td><td>46.32(1.45)</td></tr><tr><td>NashMTL</td><td>46.83</td><td>47.67</td></tr><tr><td>w/ ST</td><td>46.78(0.05)</td><td>47.34(0.33)</td></tr></table>

Table 10. Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (•). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training Swin on CelebA.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Object Azimuth</td><td colspan="2">Camera Elevation</td><td rowspan="2"> $\Delta m\% \downarrow$ </td></tr><tr><td>Top 1 ↑</td><td>Top 5 ↑</td><td>Top 1 ↑</td><td>Top 5 ↑</td></tr><tr><td>STL</td><td>32.92</td><td>70.06</td><td>36.56</td><td>94.67</td><td>-</td></tr><tr><td rowspan="2">Joint Trainw/ ST</td><td>28.01</td><td>67.05</td><td>29.84</td><td>89.75</td><td>10.70</td></tr><tr><td>27.33</td><td>68.35</td><td>30.73</td><td>89.87</td><td>10.11</td></tr><tr><td rowspan="2">PCGradw/ ST</td><td>28.79</td><td>67.85</td><td>30.10</td><td>88.44</td><td>9.99</td></tr><tr><td>27.539</td><td>67.92</td><td>31.18</td><td>90.18</td><td>9.71</td></tr><tr><td rowspan="2">CAGradw/ ST</td><td>28.72</td><td>68.42</td><td>29.33</td><td>87.93</td><td>10.50</td></tr><tr><td>28.59</td><td>68.21</td><td>29.82</td><td>88.37</td><td>10.22</td></tr><tr><td rowspan="2">GradDropw/ ST</td><td>27.50</td><td>66.13</td><td>29.86</td><td>88.50</td><td>11.73</td></tr><tr><td>28.34</td><td>67.79</td><td>29.52</td><td>88.38</td><td>10.76</td></tr><tr><td rowspan="2">MGDAw/ ST</td><td>30.82</td><td>70.13</td><td>27.29</td><td>86.16</td><td>10.15</td></tr><tr><td>28.28</td><td>68.88</td><td>30.01</td><td>89.47</td><td>9.79</td></tr><tr><td rowspan="2">IMTL-Gw/ ST</td><td>29.57</td><td>69.92</td><td>28.51</td><td>86.74</td><td>10.19</td></tr><tr><td>27.65</td><td>69.09</td><td>30.01</td><td>89.66</td><td>10.15</td></tr><tr><td rowspan="2">NashMTLw/ ST</td><td>27.02</td><td>66.88</td><td>30.83</td><td>89.74</td><td>10.84</td></tr><tr><td>28.17</td><td>67.93</td><td>31.01</td><td>89.35</td><td>9.57</td></tr></table>

Table 11. The test performance on SmallNORB dataset trained on ViT. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="2"> $\Delta m\% \downarrow$ </td></tr><tr><td>mIoU ↑</td><td>Pix Acc ↑</td><td>Abs Err ↓</td><td>Rel Err ↓</td></tr><tr><td>STL</td><td>77.61</td><td>94.15</td><td>0.0122</td><td>35.68</td><td>-</td></tr><tr><td>Joint Train</td><td>78.14</td><td>94.29</td><td>0.0174</td><td>59.21</td><td>26.87</td></tr><tr><td>w/ ST</td><td>78.34</td><td>94.34</td><td>0.0143</td><td>55.00</td><td>17.48</td></tr><tr><td>PCGrad</td><td>77.79</td><td>94.21</td><td>0.0155</td><td>51.99</td><td>19.96</td></tr><tr><td>w/ ST</td><td>77.79</td><td>94.26</td><td>0.0160</td><td>51.99</td><td>19.22</td></tr><tr><td>CAGrad</td><td>76.82</td><td>93.70</td><td>0.0138</td><td>53.74</td><td>16.26</td></tr><tr><td>w/ ST</td><td>77.20</td><td>94.01</td><td>0.0150</td><td>39.85</td><td>8.88</td></tr><tr><td>GradDrop</td><td>77.91</td><td>94.28</td><td>0.0154</td><td>55.58</td><td>20.34</td></tr><tr><td>w/ ST</td><td>78.34</td><td>94.38</td><td>0.0163</td><td>48.95</td><td>17.45</td></tr><tr><td>MGDA</td><td>69.91</td><td>92.17</td><td>0.0124</td><td>40.68</td><td>6.91</td></tr><tr><td>w/ ST</td><td>68.38</td><td>91.91</td><td>0.0128</td><td>33.19</td><td>3.17</td></tr><tr><td>IMTL-G</td><td>77.55</td><td>94.10</td><td>0.0135</td><td>47.17</td><td>10.65</td></tr><tr><td>w/ ST</td><td>75.75</td><td>93.98</td><td>0.0138</td><td>40.16</td><td>7.10</td></tr><tr><td>NashMTL</td><td>77.51</td><td>94.22</td><td>0.0152</td><td>36.36</td><td>6.68</td></tr><tr><td>w/ ST</td><td>76.87</td><td>94.09</td><td>0.0148</td><td>33.30</td><td>3.99</td></tr></table>

Table 12. The test performance on CityScapes dataset training on MTAN model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average incidence of GC (%)</td></tr><tr><td>All epochs</td><td>Last 50% epochs</td></tr><tr><td>Joint Train</td><td>39.72</td><td>40.99</td></tr><tr><td>w/ ST</td><td>38.79 (0.93)</td><td>40.02 (0.97)</td></tr><tr><td>PCGrad</td><td>39.98</td><td>41.06</td></tr><tr><td>w/ ST</td><td>38.66(1.32)</td><td>39.97(1.09)</td></tr><tr><td>CAGrad</td><td>39.39</td><td>40.94</td></tr><tr><td>w/ ST</td><td>37.77(1.62)</td><td>39.42(1.52)</td></tr><tr><td>GradDrop</td><td>39.32</td><td>40.72</td></tr><tr><td>w/ ST</td><td>39.03(0.29)</td><td>40.12(0.60)</td></tr><tr><td>MGDA</td><td>36.37</td><td>39.69</td></tr><tr><td>w/ ST</td><td>36.14(0.23)</td><td>39.38(0.31)</td></tr><tr><td>IMTL-G</td><td>37.72</td><td>39.51</td></tr><tr><td>w/ ST</td><td>36.83(0.89)</td><td>38.72(0.79)</td></tr><tr><td>NashMTL</td><td>38.40</td><td>40.69</td></tr><tr><td>w/ ST</td><td>38.04(0.36)</td><td>40.26(0.43)</td></tr></table>

Table 13. Average incidence of gradient conflict between tasks over epochs for different methods. The improvement by sparse training is provided in (•). We calculate the average incidence of gradient conflict over all epochs and the last 50% epochs during training MTAN on CityScapes.

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3">Δm% ↓</td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within t° ↑</td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>FAMO</td><td>57.64</td><td>78.59</td><td>0.3574</td><td>0.1463</td><td>19.396</td><td>12.846</td><td>45.61</td><td>71.87</td><td>80.59</td><td>-0.5669</td></tr><tr><td>w/ ST</td><td>57.68</td><td>78.79</td><td>0.3520</td><td>0.1430</td><td>19.279</td><td>12.711</td><td>46.12</td><td>72.06</td><td>80.72</td><td>-1.353</td></tr></table>

Table 14. The test performance on NYU-v2 dataset training on SAM model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td>CelebA</td><td colspan="3">Clevr</td><td>NYU-v2</td></tr><tr><td> $\Delta m\% \downarrow$ (F1)</td><td>Counting(Top 1 ↑)</td><td>Depth(Top 1 ↑)</td><td> $\Delta m\% \downarrow$ </td><td> $\Delta m\% \downarrow$ </td></tr><tr><td>FAMO</td><td>2.35</td><td>55.83</td><td>56.80</td><td>3.16</td><td>-4.10</td></tr><tr><td>w/ ST</td><td>2.32</td><td>62.57</td><td>56.04</td><td>-1.93</td><td>-4.46</td></tr></table>

Table 15. The test performance on CelebA, Clevr and NYU-v2 dataset. CelebA is trained on Swin Transformer and Clevr is trained on ViT. NYU-v2 is trained on MTAN. The green cell color indicates that sparse training improves the performance ofjoint training or gradient manipulation methods. The best result is highlighted in bold.

![](images/bf4e1eaeb34fc53c6fef8683f2a29deb381404d95816b5bb68d937e202421221.jpg)

<details>
<summary>line</summary>

| Epoch | Joint Train (%) | Joint Train w/ ST (%) |
| --- | --- | --- |
| 0 | ~53 | ~48 |
| 10 | ~38 | ~37 |
| 25 | ~39 | ~30 |
| 50 | ~37 | ~44 |
| 75 | ~39 | ~36 |
| 100 | ~38 | ~38 |
| 125 | ~41 | ~42 |
| 150 | ~40 | ~41 |
| 175 | ~41 | ~42 |
| 200 | ~42 | ~38 |
</details>

(a) Joint Train

![](images/c13bce3fd482d6982ce4d81defb1f42551854a423527b05542892f4d99e2aed9.jpg)

<details>
<summary>line</summary>

| Epoch | GradDrop (%) | GradDrop w/ ST (%) |
| --- | --- | --- |
| 0 | ~50.0 | ~50.0 |
| 10 | ~36.0 | ~38.0 |
| 25 | ~37.0 | ~39.0 |
| 50 | ~38.0 | ~38.0 |
| 75 | ~37.0 | ~38.0 |
| 100 | ~38.0 | ~40.0 |
| 125 | ~40.0 | ~42.0 |
| 150 | ~40.0 | ~42.0 |
| 175 | ~38.0 | ~40.0 |
| 200 | ~36.0 | ~37.0 |
</details>

(b) GradDrop

![](images/be7aa293fbef2650f6ba9d72b1f38828a2d8e9ede8a149f95167139f144f6ffe.jpg)

<details>
<summary>line</summary>

| Epoch | IMTL-G (%) | IMTL-G w/ ST (%) |
| --- | --- | --- |
| 0 | ~32 | ~54 |
| 25 | ~33 | ~36 |
| 50 | ~35 | ~37 |
| 75 | ~38 | ~35 |
| 100 | ~40 | ~39 |
| 125 | ~42 | ~41 |
| 150 | ~39 | ~38 |
| 175 | ~40 | ~39 |
| 200 | ~33 | ~40 |
</details>

(c) IMTL-G

![](images/8ce7c0c62b53072a31b1dbb44eb0c460ef82476e915dcdb19fd8749a1541cf75.jpg)

<details>
<summary>line</summary>

| Epoch | CAGrad (%) | CAGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~53 | ~53 |
| 10 | ~36 | ~34 |
| 20 | ~37 | ~38 |
| 30 | ~36 | ~36 |
| 40 | ~37 | ~37 |
| 50 | ~38 | ~38 |
| 60 | ~39 | ~39 |
| 70 | ~38 | ~38 |
| 80 | ~39 | ~39 |
| 90 | ~40 | ~40 |
| 100 | ~41 | ~41 |
| 110 | ~42 | ~42 |
| 120 | ~43 | ~43 |
| 130 | ~44 | ~44 |
| 140 | ~45 | ~45 |
| 150 | ~46 | ~46 |
| 160 | ~47 | ~47 |
| 170 | ~48 | ~48 |
| 180 | ~49 | ~49 |
| 190 | ~50 | ~50 |
| 200 | ~51 | ~51 |
</details>

(d) CAGrad

![](images/7e496712e48bfe9e23d6ae4d72a9894bce7032a2e793332263db00745d06d6d2.jpg)

<details>
<summary>line</summary>

| Epoch | PCGrad (%) | PCGrad w/ ST (%) |
| --- | --- | --- |
| 0 | ~52 | ~49 |
| 25 | ~38 | ~37 |
| 50 | ~41 | ~38 |
| 75 | ~36 | ~35 |
| 100 | ~43 | ~42 |
| 125 | ~41 | ~40 |
| 150 | ~40 | ~39 |
| 175 | ~41 | ~40 |
| 200 | ~38 | ~43 |
</details>

(e) PCGrad

![](images/58c4bd7d4fbe63dc361deee594a1020564b59a67210a46f91bf0a9b78d5562e4.jpg)

<details>
<summary>line</summary>

| Epoch | NashMTL (%) | NashMTL w/ ST (%) |
| --- | --- | --- |
| 0 | ~45.5 | ~44.5 |
| 10 | ~32.5 | ~31.5 |
| 20 | ~37.5 | ~36.5 |
| 30 | ~32.5 | ~33.5 |
| 40 | ~37.5 | ~36.5 |
| 50 | ~35.5 | ~36.5 |
| 60 | ~39.5 | ~38.5 |
| 70 | ~39.5 | ~43.5 |
| 80 | ~39.5 | ~39.5 |
| 90 | ~45.0 | ~39.5 |
| 100 | ~41.5 | ~41.5 |
| 110 | ~45.5 | ~44.5 |
| 120 | ~41.5 | ~41.5 |
| 130 | ~46.0 | ~46.0 |
| 140 | ~41.5 | ~41.5 |
| 150 | ~46.0 | ~46.0 |
| 160 | ~41.5 | ~41.5 |
| 170 | ~46.0 | ~46.0 |
| 180 | ~41.5 | ~41.5 |
| 190 | ~46.0 | ~46.0 |
| 200 | ~44.5 | ~42.5 |
</details>

(f) NashMTL

![](images/69afb7c6698cc178043e59cc1517f0244a68817c26bc9edb6970b08593f7c53f.jpg)

<details>
<summary>line</summary>

| Epoch | MGDA (%) | MGDA w/ ST (%) |
| --- | --- | --- |
| 0 | ~60 | ~60 |
| 25 | ~30 | ~30 |
| 50 | ~30 | ~30 |
| 75 | ~35 | ~35 |
| 100 | ~38 | ~38 |
| 125 | ~40 | ~40 |
| 150 | ~40 | ~40 |
| 175 | ~40 | ~40 |
| 200 | ~40 | ~40 |
</details>

(g) MGDA

Figure 12. The number of occurrence gradient conflictions between tasks during tuning MTAN on CityScapes dataset.  
![](images/134eca3f25c20cc74225f33f054c7cda3024948602cd4eb7c29922c82a2c82bc.jpg)

<details>
<summary>bar</summary>

| Method | Baseline (All epochs) | Baseline w/ ST (All epochs) | Baseline (Last 50% epochs) | Baseline w/ ST (Last 50% epochs) |
| :--- | :--- | :--- | :--- | :--- |
| Joint Train | ~39.7 | ~38.7 | ~41.0 | ~40.0 |
| PCGrad | ~40.0 | ~38.6 | ~41.1 | ~40.0 |
| CAGrad | ~39.4 | ~37.7 | ~41.0 | ~39.4 |
| GradDrop | ~39.3 | ~39.0 | ~40.7 | ~40.1 |
| MGDA | ~36.3 | ~36.1 | ~39.7 | ~39.4 |
| IMTL-G | ~37.7 | ~36.8 | ~39.5 | ~38.7 |
| NashMTL | ~38.4 | ~38.0 | ~40.7 | ~40.2 |
</details>

Figure 13. The average occurrence percentage of gradient conflict over epochs (all epochs/last 50% epochs) during training on MTAN model with CityScapes datasets was evaluated using various methods, including joint training and gradient manipulation techniques.

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m\% \downarrow$ </td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within  $t^{\circ} \uparrow$ </td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>FAMO</td><td>38.88</td><td>64.90</td><td>0.5474</td><td>0.2194</td><td>25.06</td><td>19.57</td><td>29.21</td><td>56.61</td><td>68.98</td><td>-4.10</td></tr><tr><td>w/ ST</td><td>37.85</td><td>65.27</td><td>0.5543</td><td>0.2215</td><td>25.09</td><td>19.15</td><td>30.03</td><td>57.49</td><td>69.52</td><td>-4.46</td></tr></table>

Table 16. The test performance on NYU-v2 dataset training on MTAN model. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Object Azimuth</td><td colspan="2">Camera Elevation</td><td rowspan="2"> $\Delta m\% \downarrow$ </td></tr><tr><td>Top 1 ↑</td><td>Top 5 ↑</td><td>Top 1 ↑</td><td>Top 5 ↑</td></tr><tr><td>FAMO</td><td>24.68</td><td>63.69</td><td>34.35</td><td>92.13</td><td>10.71</td></tr><tr><td>w/ ST</td><td>26.38</td><td>66.54</td><td>32.05</td><td>91.02</td><td>10.27</td></tr></table>

Table 17. The test performance on SmallNORB dataset trained on ViT. The green cell color indicates that sparse training improves the performance of joint training or gradient manipulation methods. The best result is highlighted in bold.