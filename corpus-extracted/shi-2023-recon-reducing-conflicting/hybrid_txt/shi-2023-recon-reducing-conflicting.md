# RECON: REDUCING CONFLICTING GRADIENTS FROM THE ROOT FOR MULTI-TASK LEARNING

Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, Xiao-Ming Wu<sup></sup> Department of Computing, The Hong Kong Polytechnic University, Hong Kong S.A.R., China {guang-yuan.shi, qee-mai.li, wenlong.zhang}@connect.polyu.hk, jiax.chen@connect.polyu.hk, xiao-ming.wu@polyu.edu.hk

## ABSTRACT

A fundamental challenge for multi-task learning is that different tasks may conflict with each other when they are solved jointly, and a cause of this phenomenon is conflicting gradients during optimization. Recent works attempt to mitigate the influence of conflicting gradients by directly altering the gradients based on some criteria. However, our empirical study shows that “gradient surgery” cannot effectively reduce the occurrence of conflicting gradients. In this paper, we take a different approach to reduce conflicting gradients from the root. In essence, we investigate the task gradients w.r.t. each shared network layer, select the layers with high conflict scores, and turn them to task-specific layers. Our experiments show that such a simple approach can greatly reduce the occurrence of conflicting gradients in the remaining shared layers and achieve better performance, with only a slight increase in model parameters in many cases. Our approach can be easily applied to improve various state-of-the-art methods including gradient manipulation methods and branched architecture search methods. Given a network architecture (e.g., ResNet18), it only needs to search for the conflict layers once, and the network can be modified to be used with different methods on the same or even different datasets to gain performance improvement. The source code is available at https://github.com/moukamisama/Recon.

## 1 INTRODUCTION

Multi-task learning (MTL) is a learning paradigm in which multiple different but correlated tasks are jointly trained with a shared model (Caruana, 1997), in the hope of achieving better performance with an overall smaller model size than learning each task independently. By discovering shared structures across tasks and leveraging domain-specific training signals of related tasks, MTL can achieve efficiency and effectiveness. Indeed, MTL has been successfully applied in many domains including natural language processing (Hashimoto et al., 2017), reinforcement learning (Parisotto et al., 2016; D’Eramo et al., 2020) and computer vision (Vandenhende et al., 2021).

A major challenge for multi-task learning is negative transfer (Ruder, 2017), which refers to the performance drop on a task caused by the learning of other tasks, resulting in worse overall performance than learning them separately. This is caused by task conflicts, i.e., tasks compete with each other and unrelated information of individual tasks may impede the learning of common structures. From the optimization point of view, a cause of negative transfer is conflicting gradients (Yu et al., 2020), which refers to two task gradients pointing away from each other and the update of one task will have a negative effect on the other. Conflicting gradients make it difficult to optimize the multitask objective, since task gradients with larger magnitude may dominate the update vector, making the optimizer prioritize some tasks over others and struggle to converge to a desirable solution.

Prior works address task/gradient conflicts mainly by balancing the tasks via task reweighting or gradient manipulation. Task reweighting methods adaptively re-weight the loss functions by homoscedastic uncertainty (Kendall et al., 2018), balancing the pace at which tasks are learned Chen et al. (2018); Liu et al. (2019), or learning a loss weight parameter (Liu et al., 2021b). Gradient manipulation methods reduce the influence of conflicting gradients by directly altering the gradients based on different criteria (Sener & Koltun, 2018; Yu et al., 2020; Chen et al., 2020; Liu et al.,

2021a) or rotating the shared features (Javaloy & Valera, 2022). While these methods have demonstrated effectiveness in different scenarios, in our empirical study, we find that they cannot reduce the occurrence of conflicting gradients (see Sec. 3.3 for more discussion).

We propose a different approach to reduce conflicting gradients for MTL. Specifically, we investigate layer-wise conflicting gradients, i.e., the task gradients w.r.t. each shared network layer. We first train the network with a regular MTL algorithm (e.g., joint-training) for a number of iterations, compute the conflict scores for all shared layers, and select those with highest conflict scores (indicating severe conflicts). We then set the selected shared layers task-specific and train the modified network from scratch. As demonstrated by comprehensive experiments and analysis, our simple approach Recon has the following key advantages: (1) Recon can greatly reduce conflicting gradients with only a slight increase in model parameters (less than 1% in some cases) and lead to significantly better performance. (2) Recon can be easily applied to improve various gradient manipulation methods and branched architecture search methods. Given a network architecture, it only needs to search for the conflict layers once, and the network can be modified to be used with different methods and even on different datasets to gain performance improvement. (3) Recon can achieve better performance than branched architecture search methods with a much smaller model.

## 2 RELATED WORKS

In this section, we briefly review related works in multi-task learning in four categories: tasks clustering, architecture design, architecture search, and task balancing. Tasks clustering methods mainly focus on identifying which tasks should be learned together (Thrun & O’Sullivan, 1996; Zamir et al., 2018; Standley et al., 2020; Shen et al., 2021; Fifty et al., 2021).

Architecture design methods include hard parameter sharing methods (Kokkinos, 2017; Long et al., 2017; Bragman et al., 2019), which learn a shared feature extractor and task-specific decoders, and soft parameters sharing methods (Misra et al., 2016; Ruder et al., 2019; Gao et al., 2019; 2020; Liu et al., 2019), where some parameters of each task are assigned to do cross-task talk via a sharing mechanism. Compared with soft parameters sharing methods, our approach Recon has much better scalability when dealing with a large number of tasks

Instead of designing a fixed network structure, some methods (Rosenbaum et al., 2018; Meyerson & Miikkulainen, 2018; Yang et al., 2020) propose to dynamically self-organize the network for different tasks. Among them, branched architecture search (Guo et al., 2020; Bruggemann et al., 2020) methods are more related to our work. They propose an automated architecture search algorithm to build a tree-structured network by learning where to branch. In contrast, our method Recon decides which layers to be shared across tasks by considering the severity of layer-wise conflicting gradients, resulting in a more compact architecture with lower time cost and better performance.

Another line of research is task balancing methods. To address task/gradient conflicts, some methods attempt to re-weight the multi-task loss function using homoscedastic uncertainty (Kendall et al., 2018), task prioritization (Guo et al., 2018), or similar learning pace (Liu et al., 2019; 2021b). GradNorm (Chen et al., 2018) learns task weights by dynamically tuning gradient magnitudes. MGDA (Sener & Koltun, 2018) find the weights by minimizing the norm of the weighted sum of task gradients. To reduce the influence of conflicting gradients, PCGrad (Yu et al., 2020) projects each gradient onto the normal plane of another gradient and uses the average of projected gradients for update. Graddrop (Chen et al., 2020) randomly drops some elements of gradients based on element-wise conflict. CAGrad (Liu et al., 2021a) ensures convergence to a minimum of the average loss across tasks by gradient manipulation. RotoGrad (Javaloy & Valera, 2022) re-weights task gradients and rotates the shared feature space. Instead of manipulating gradients, our method Recon leverages gradient information to modify network structure to mitigate task conflicts from the root.

## 3 PILOT STUDY: TASK CONFLICTS IN MULTI-TASK LEARNING

## 3.1 MULTI-TASK LEARNING: PROBLEM DEFINITION

Multi-task learning (MTL) aims to learn a set of correlated tasks $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { T }$ simultaneously. For each task $\mathcal { T } _ { i }$ , the empirical loss function is $\mathcal { L } _ { i } ( \theta _ { \mathrm { s h } } , \theta _ { i } )$ , where $\theta _ { \mathrm { s h } }$ are parameters shared among all tasks and $\theta _ { i }$ are task-specific parameters. The goal is to find optimal parameters $\boldsymbol { \theta } = \{ \theta _ { \mathrm { s h } } , \theta _ { 1 } , \theta _ { 2 } , \cdot \cdot \cdot , \theta _ { T } \}$ to achieve high performance across all tasks. Formally, it aims to minimize a multi-task objective:

![](images/9e8e82614726c8451c9c2904b97edbb5131eb88080e6a1f5ec2965a19249bfa0.jpg)

<details>
<summary>bar</summary>

| \(cos\phi ij\) | Joint-train | Graddrop | PCGrad | CAGrad | MGDA |
| --- | --- | --- | --- | --- | --- |
| [1.00, 0.03] | ~0.025 | ~0.025 | ~0.025 | ~0.025 | ~0.025 |
| (0.03, 0.02] | ~0.035 | ~0.04 | ~0.04 | ~0.04 | ~0.04 |
| (0.02, 0.01] | ~0.145 | ~0.145 | ~0.145 | ~0.14 | ~0.135 |
| (0.01, 0.00] | ~0.355 | ~0.365 | ~0.355 | ~0.36 | ~0.365 |
| (0.00, -0.01] | ~0.31 | ~0.31 | ~0.315 | ~0.315 | ~0.325 |
| (-0.01, -0.02] | ~0.095 | ~0.09 | ~0.09 | ~0.095 | ~0.09 |
| (-0.02, -0.03] | ~0.02 | ~0.02 | ~0.02 | ~0.02 | ~0.02 |
| (-0.03, -1.00] | ~0.015 | ~0.015 | ~0.015 | ~0.015 | ~0.01 |
</details>

Figure 1: The distributions of gradient conflicts (in terms of cos $\phi _ { i j } )$ of the joint-training baseline and state-of-the-art gradient manipulation methods on Multi-Fashion+MNIST benchmark.

$$
\theta^ {*} = \arg \min _ {\theta} \sum_ {i} ^ {T} w _ {i} \mathcal {L} _ {i} (\theta_ {\mathrm{sh}}, \theta_ {i}), \tag {1}
$$

where $w _ { i }$ are pre-defined or dynamically computed weights for different tasks. A popular choice is to use the average loss $( \mathrm { i . e . , }$ equal weights). However, optimizing the multi-task objective is difficult, and a known cause is conflicting gradients.

## 3.2 CONFLICTING GRADIENTS

Let $\mathbf { g } _ { i } = \nabla _ { \boldsymbol { \theta } _ { \mathrm { s h } } } \mathcal { L } _ { i } ( \boldsymbol { \theta } _ { \mathrm { s h } } , \boldsymbol { \theta } _ { i } )$ denote the gradient of task $\mathcal { T } _ { i }$ w.r.t. the shared parameters $\theta _ { \mathrm { s h } }$ (i.e., a vector of the partial derivatives of $\mathcal { L } _ { i }$ w.r.t. $\theta _ { \mathrm { s h } } )$ and $g _ { i } ^ { \mathrm { t s } } = \nabla _ { \theta _ { i } } \mathcal { L } _ { i } ( \theta _ { \mathrm { s h } } , \theta _ { i } )$ denote the gradient w.r.t. the task-specific parameters $\theta _ { i }$ . A small change of $\theta _ { \mathrm { s h } }$ in the direction of negative $\mathbf { g } _ { i } \mathrm { i s } \theta _ { \mathrm { s h } }  \theta _ { \mathrm { s h } } - \alpha \mathbf { g } _ { i }$ with a sufficiently small step size α. The effect of this change on the performance of another task $\tau _ { j }$ is measured by:

$$
\Delta \mathcal {L} _ {j} = \mathcal {L} _ {j} (\theta_ {\mathrm{sh}} - \alpha \mathbf {g} _ {i}, \theta_ {j}) - \mathcal {L} _ {j} (\theta_ {\mathrm{sh}}, \theta_ {j}) = - \alpha \mathbf {g} _ {i} \cdot \mathbf {g} _ {j} + o (\alpha), \tag {2}
$$

where the second equality is obtained by first order Taylor approximation. Likewise, the effect of a small update of $\theta _ { \mathrm { s h } }$ in the direction of the negative gradient of task $\mathcal { T } _ { j } ( \mathrm { i } . \mathrm { e } . , - \mathbf { g } _ { j } )$ on the performance of task $\mathcal { T } _ { i }$ is $\Delta \mathcal { L } _ { i } = - \alpha \mathbf { g } _ { i } \cdot \mathbf { g } _ { j } + o ( \alpha )$ ). Notably, the model update for task $\mathcal { T } _ { i }$ is considered to have a negative effect on task $\tau _ { i }$ when $\mathbf { g } _ { i } \cdot \mathbf { g } _ { j } < 0$ , since it increases the loss of task $\tau _ { j }$ , and vice versa. A formal definition of conflicting gradients is given as follows (Yu et al., 2020).

Definition 1 (Conflicting Gradients). The gradients g and $\mathbf { g } _ { j } ( i \neq j )$ are said to be conflicting with each other ifcos $\phi _ { i j } < 0 $ , where $\phi _ { i j }$ is the angle between g<sub>i</sub> and $\mathbf { g } _ { j }$ .

As shown in Yu et al. (2020), conflicts in gradient pose serious challenges for optimizing the multitask objective (Eq. 1). Using the average gradient $\begin{array} { r } { ( \mathrm { i . e . , } \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \mathbf { g } _ { i } ) } \end{array}$ for gradient decent may hurt the performance of individual tasks, especially when there is a large difference in gradient magnitudes, which will make the optimizer struggle to converge to a desirable solution.

## 3.3 GRADIENT SURGERY CANNOT EFFECTIVELY REDUCE CONFLICTING GRADIENTS

To mitigate the influence of conflicting gradients, several methods (Yu et al., 2020; Chen et al., 2020; Liu et al., 2021a) have been proposed to perform “gradient surgery”. Instead of following the average gradient direction, they alter conflicting gradients based on some criteria and use the modified gradients for model update. We conduct a pilot study to investigate whether gradient manipulation can effectively reduce the occurrence of conflicting gradients. For each training iteration, we first calculate the task gradients of all tasks w.r.t. the shared parameters $( \mathrm { i } . \mathrm { e } . , \mathbf { g } _ { i }$ for any task i) and compute the conflict angle between any two task gradients $\mathbf { g } _ { i }$ and ${ \bf { g } } _ { j }$ in terms of $c o s \phi _ { i j }$ . We then count and draw the distribution of $c o s \phi _ { i j }$ in all training iterations. We provide the statistics of the joint-training baseline (i.e., training all tasks jointly with equal loss weights and all parameters shared) and several state-of-the-art gradient manipulation methods including GradDrop (Chen et al., 2020), PCGrad (Yu et al., 2020), CAGrad (Liu et al., 2021a), and MGDA (Sener & Koltun, 2018) on Multi-Fashion+MNIST (Lin et al., 2019), CityScapes, NYUv2, and PASCAL-Context datasets.

![](images/e650866a0e7dce5e27f8e0812dacf9bdcc44e77b4cb50942f5d232fb34090862.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(a) Joint-train

![](images/adb2f9173fa1694e1fd0e52eb3ed1c7d67c1a1933f1c84ab456ba30c5cf387ec.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
θ_1
θ_2
θ_3
</details>

(b) PCGrad

![](images/f55f7b7c3209d97934c7a6b229de301275120fabe2c614d27b80304b4c6f20a4.jpg)

<details>
<summary>text_image</summary>

g_i
g_j
g_i^fix
g_j^fix
θ_1
θ_2
θ_3
</details>

(c) Recon

![](images/c2d3154ca0724c36cf76307e5d24563fd58e40a2e4c477a842b66b6acf5ddbcc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  subgraph High_conflict_score ["High conflict score"]
  A["Start"] --> B["Process Block 1"]
  B --> C["Output"]
  C --> D["Task1"]
  C --> E["Task2"]
  end

  subgraph Turned_task_specific ["Turned task-specific"]
  F["Start"] --> G["Process Block 2"]
  G --> H["Output"]
  H --> I["Task1"]
  H --> J["Task2"]
  end
```
</details>

(d) Recon  
Figure 2: Illustration of the differences between joint-training, gradient manipulation, and our approach. (a) In joint-training, the update vector (in green) is the average gradient $\begin{array} { r } { \frac { 1 } { 2 } ( { \bf g } _ { i } + { \bf g } _ { j } ) } \end{array}$ . Due to the conflict between $\mathbf { g } _ { i }$ and $\mathbf { g } _ { j }$ , the update vector is dominated by $\mathbf { g } _ { i }$ (in red). (b) PCGrad (Yu et al., 2020) projects each gradient onto the normal plane of the other one and uses the average of the projected gradients (indicated by dashed grey arrows) as the update vector (in green). As such, the update vector is less dominated by $\mathbf { g } _ { i }$ . (c) Our approach Recon finds the parameters contributing most $( \mathrm { e . g . , } \theta _ { 3 } )$ to gradient conflicts and turns them into task specific ones. In effect, it performs an orthographic/coordinate projection of conflicting gradients to the space of the rest parameters (e.g., $\theta _ { 1 }$ and $\theta _ { 2 } )$ such that the projected gradients $\mathbf { g } _ { i } ^ { \mathrm { { f i x } } }$ and $\mathbf { g } _ { j } ^ { \mathrm { { f i x } } }$ are better aligned. (d) Illustration of Recon turning a shared layer with high conflict score to task-specific layers.

The results are provided in Fig. 1, Fig. 5, Fig. 6, Fig. 7, Table 6, and Tables 8-10. It can be seen that gradient manipulation methods can only slightly reduce the occurrence of conflicting gradients (compared to joint-training) in some cases, and in some other cases they even increase it.

## 4 OUR APPROACH: REDUCING CONFLICTING GRADIENTS FROM THE ROOT

Our pilot study shows that adjusting gradients for model update cannot effectively prevent the occurrence of conflicting gradients in MTL, which suggests that the root causes of this phenomenon may be closely related to the nature of different tasks and the way how model parameters are shared among them. Therefore, to mitigate task conflicts for MTL, in this paper, we take a different approach to reduce the occurrence of conflicting gradients from the root.

## 4.1 RECON: REMOVING LAYER-WISE CONFLICTING GRADIENTS

Our approach is extremely simple and intuitive. We first identify the shared network layers where conflicts occur most frequently and then turn them into task-specific parameters. Suppose the shared model parameters $\theta _ { \mathrm { s h } }$ are composed of n layers, i.e., $\theta _ { \mathrm { s h } } = \{ \theta _ { \mathrm { s h } } ^ { ( k ) } \} _ { k = 1 } ^ { n }$ , where $\theta _ { \mathrm { s h } } ^ { ( k ) }$ is the $k ^ { \mathrm { t h } }$ shared layer. Let $\mathbf { g } _ { i } ^ { ( k ) }$ denote the gradient of task $\mathcal { T } _ { i }$ w.r.t. the $k ^ { \mathrm { t h } }$ shared layer $\theta _ { \mathrm { s h } } ^ { ( k ) }$ , i.e., $\mathbf { g } _ { i } ^ { ( k ) }$ is a vector of the partial derivatives of $\mathcal { L } _ { i }$ w.r.t. the parameters of $\theta _ { \mathrm { s h } } ^ { ( k ) }$ . Let $\phi _ { i j } ^ { ( k ) }$ denote the angle between $\mathbf { g } _ { i } ^ { ( k ) }$ and $\mathbf { g } _ { j } ^ { ( k ) }$ . We define layer-wise conflicting gradients and S-conflict score as follows.

Definition 2 (Layer-wise Conflicting Gradients). The gradients $\mathbf { g } _ { i } ^ { ( k ) }$ and $\mathbf { g } _ { j } ^ { ( k ) } \left( i \neq j \right)$ are said to be conflicting with each other $i f \cos \phi _ { i j } ^ { ( k ) } < 0$

Definition 3 (S-Conflict Score). For an $\begin{array} { r } { v - 1 < S \le 0 } \end{array}$ , the S-conflict score for the $k ^ { \mathrm { t h } }$ shared layer is the number of different pairs $( i , j ) ( i \neq j )$ s.t. cos $\phi _ { i j } ^ { ( k ) } < S _ { : }$ , denoted as $s ^ { ( k ) }$

S indicates the severity of conflicts, and setting S smaller means we care about cases of more severe conflicts. The S-conflict score $s ^ { ( k ) }$ indicates the occurrence of conflicting gradients at severity level S for the $k ^ { \mathrm { t h } }$ shared layer. $\begin{array} { r } { \operatorname { I f } s ^ { ( k ) } = { \binom { T } { 2 } } } \end{array}$ , it means that for any two different tasks, there is a conflict in their gradients w.r.t. the $k ^ { \mathrm { t h } }$ shared layer. By computing S-conflict scores, we can identify the shared layers where conflicts occur most frequently.

We describe our method Recon in Algorithm 1. First, we train the network for I iterations and compute S-conflict scores for each shared layer $\theta ^ { ( k ) }$ in every iteration, denoted by $\{ s _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { I }$ . Then, we sum up the scores in all iterations, i.e., $\begin{array} { r } { s ^ { ( k ) } = \sum _ { i = 1 } ^ { I } s _ { i } ^ { ( k ) } } \end{array}$ , and find the layers with highest $s ^ { ( k ) }$ scores. Next, we set these layers to be task-specific and train the modified network from scratch. We demonstrate the effectiveness of Recon by a theoretical analysis in Sec. 4.2 and comprehensive experiments in Sec. 5. The results show that Recon can effectively reduce the occurrence of conflicting gradients in the remaining shared layers and lead to substantial improvements over state-of-the-art.

Algorithm 1: Recon: Removing Layer-wise Conflicting Gradients

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: Model parameters $\theta$, learning rate $\alpha$, a set of tasks $\{\mathcal{T}_i\}_{i=1}^T$, number of iterations $I$ for computing conflict scores, conflict severity level $S$, number of selected layers $K$.
// Train the network and compute conflict scores for all layers
for iteration $i = 1, 2, \ldots, I$ do
    for $i = 1, 2, \ldots, T$ do
        Compute the gradients of task $\mathcal{T}_i$ w.r.t. all shared layers, i.e., $\{\mathbf{g}_i^{(k)}\}_{k=1}^n$;
    end
    Calculate the $S$-conflict scores for all shared layers in the current iteration, i.e., $\{s_i^{(k)}\}_{k=1}^n$;
    Update $\theta$ with joint-training or any gradient manipulation method;
</div>

end

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
// Set layers with top conflict scores task-specific
For each layer $k$, calculate the sum of $S$-conflict scores in all iterations, i.e., $s^{(k)} = \sum_{i=1}^{I} s_i^{(k)}$;
Select the top $K$ layers with highest $s^{(k)}$ and set them task-specific;
// Train the modified network from scratch
for iteration $i = 1, 2, \ldots$ do
    Update $\theta$ with joint-training or any gradient manipulation method;
end
Output: Model parameters $\theta$.
</div>

## 4.2 THEORETICAL ANALYSIS

Here, we provide a theoretical analysis of Recon. Let $\theta _ { \mathrm { s h } } = \{ \theta _ { \mathrm { s h } } ^ { \mathrm { f i x } } , \theta _ { \mathrm { s h } } ^ { \mathrm { c f } } \}$ , where $\theta _ { \mathrm { s h } } ^ { \mathrm { f i x } }$ are the remaining shared parameters, and $\theta _ { \mathrm { s h } } ^ { \mathrm { c f } }$ are those that will be turned to task-specific parameters $\theta _ { 1 } ^ { \mathrm { c f } } , \theta _ { 2 } ^ { \mathrm { c f } } , \cdot \cdot \cdot , \theta _ { T } ^ { \mathrm { c f } }$ Notice that $\theta _ { 1 } ^ { \mathrm { c f } } , \theta _ { 2 } ^ { \mathrm { c f } } , \cdot \cdot \cdot , \overline { { \theta _ { T } ^ { \mathrm { c f } } } }$ will all be initialized with $\theta _ { \mathrm { s h } } ^ { \mathrm { c f } }$ . Therefore, after applying Recon, the model parameters are $\theta _ { r } = \{ \theta _ { \mathrm { s h } } ^ { \mathrm { f i x } } , \theta _ { 1 } ^ { \mathrm { c f } } , \dots , \theta _ { T } ^ { \mathrm { c f } } , \theta _ { 1 } ^ { \mathrm { t s } } , \dots , \theta _ { T } ^ { \mathrm { t s } } \}$ . An one-step gradient update of $\theta _ { r }$ is:

$$
\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}} = \theta_ {\mathrm{sh}} ^ {\mathrm{fix}} - \alpha \sum_ {i = 1} ^ {T} w _ {i} \mathbf {g} _ {i} ^ {\mathrm{fix}}, \quad \hat {\theta} _ {i} ^ {\mathrm{cf}} = \theta_ {i} ^ {\mathrm{cf}} - \alpha \mathbf {g} _ {i} ^ {\mathrm{cf}}, \quad \hat {\theta} _ {i} ^ {\mathrm{ts}} = \theta_ {i} ^ {\mathrm{ts}} - \alpha \mathbf {g} _ {i} ^ {\mathrm{ts}}, \quad i = 1, \dots , T, \tag {3}
$$

where $w _ { i }$ are weight parameters, $\mathbf { g } _ { i } ^ { \mathrm { t s } } = \nabla _ { \theta _ { i } ^ { \mathrm { t s } } } \mathcal { L } _ { i } , \mathbf { g } _ { i } ^ { \mathrm { c f } } = \nabla _ { \theta _ { \mathrm { e } , \mathrm { l } } ^ { \mathrm { c f } } } \mathcal { L } _ { i }$ and $\mathbf { g } _ { i } ^ { \mathrm { f i x } } = \nabla _ { \theta _ { \mathrm { c h } } ^ { \mathrm { f i x } } } \mathcal { L } _ { i }$ . Notice that different methods such as joint-training, MGDA Sener & Koltun (2018), PCGrad Yu et al. (2020), and CAGrad Liu et al. (2021a) choose different $w _ { i }$ dynamically.

Without applying Recon, the model parameters are $\theta = \{ \theta _ { \mathrm { { s h } } } ^ { \mathrm { { f i x } } } , \theta _ { \mathrm { { s h } } } ^ { \mathrm { { c f } } } , \theta _ { 1 } ^ { \mathrm { { t s } } } , . . . , \theta _ { T } ^ { \mathrm { { t s } } } \}$ . An one-step gradient update of θ is given by

$$
\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}} = \theta_ {\mathrm{sh}} ^ {\mathrm{fix}} - \alpha \sum_ {i = 1} ^ {T} w _ {i} \mathbf {g} _ {i} ^ {\mathrm{fix}}, \quad \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}} = \theta_ {\mathrm{sh}} ^ {\mathrm{cf}} - \alpha \sum_ {i = 1} ^ {T} w _ {i} \mathbf {g} _ {i} ^ {\mathrm{cf}}, \quad \hat {\theta} _ {i} ^ {\mathrm{ts}} = \theta_ {i} ^ {\mathrm{ts}} - \alpha \mathbf {g} _ {i} ^ {\mathrm{ts}}, \quad i = 1, \dots , T. \tag {4}
$$

After the one-step updates, the loss functions with the updated parameters $\hat { \theta } _ { r }$ and $\hat { \theta }$ respectively are:

$$
\mathcal {L} \left(\hat {\theta} _ {r}\right) = \sum_ {i = 1} ^ {T} \mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\text {fix}}, \hat {\theta} _ {i} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right), \text {and,} \mathcal {L} (\hat {\theta}) = \sum_ {i = 1} ^ {T} \mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\text {fix}}, \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right), \tag {5}
$$

where $\mathcal { L } _ { i }$ is the loss function of task $\mathcal { T } _ { i } .$ . Denote the set of indices of the layers turned task-specific by P, then $\theta _ { \mathrm { s h } } ^ { \mathrm { c f } } = \{ \theta _ { \mathrm { s h } } ^ { ( k ) } \} , k \in \mathbb { P }$ . Assume that $\textstyle \sum _ { i = 1 } ^ { T } w _ { i } = 1$ , then we have the following theorem.

Table 1: Multi-task learning results on Multi-Fashion+MNIST dataset. All experiments are repeated over 3 random seeds and the mean values are reported. ∆m% denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

<table><tr><td>Method</td><td>Single-task</td><td>RotoGrad</td><td>BMTAS</td><td>Joint-train</td><td>w/ Recon</td><td>MGDA</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td><td>GradDrop</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td><td>MMoE</td><td>w/ Recon</td></tr><tr><td>T1 Acc↑</td><td>98.37</td><td>98.10</td><td>98.20</td><td>97.42</td><td>98.13</td><td>95.19</td><td>98.33</td><td>97.37</td><td>98.30</td><td>97.38</td><td>98.25</td><td>97.47</td><td>98.28</td><td>98.27</td><td>98.25</td></tr><tr><td>T2 Acc↑</td><td>89.63</td><td>88.25</td><td>89.71</td><td>88.82</td><td>89.26</td><td>89.46</td><td>89.28</td><td>88.68</td><td>89.77</td><td>88.57</td><td>89.51</td><td>88.85</td><td>89.65</td><td>89.51</td><td>89.67</td></tr><tr><td> $\Delta m\%^{\uparrow}$ </td><td>-</td><td>-0.91</td><td>-0.04</td><td>-0.94</td><td>-0.33</td><td>-1.71</td><td>-0.22</td><td>-1.04</td><td>0.04</td><td>-1.10</td><td>-0.13</td><td>-0.90</td><td>-0.04</td><td>-0.12</td><td>-0.04</td></tr><tr><td>#P.</td><td>85.62</td><td>42.81</td><td>85.61</td><td>42.81</td><td>43.43</td><td>42.81</td><td>43.43</td><td>42.81</td><td>43.43</td><td>42.81</td><td>43.43</td><td>42.81</td><td>43.43</td><td>85.62</td><td>105.70</td></tr></table>

Table 2: Multi-task learning results on CelebA dataset. All experiments are repeated over 3 random seeds and the mean values are reported. $\Delta m \%$ denotes the average relative improvement of all tasks. #P denotes model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

<table><tr><td>Method</td><td>Single-task</td><td>Joint-train</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td><td>Graddrop</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td></tr><tr><td>Average Error</td><td>8.38</td><td>8.33</td><td>8.22</td><td>8.31</td><td>8.23</td><td>8.33</td><td>8.20</td><td>8.64</td><td>8.36</td></tr><tr><td> $\Delta m\% \uparrow$ </td><td>-</td><td>0.55</td><td>1.92</td><td>0.79</td><td>1.74</td><td>0.23</td><td>2.13</td><td>-3.14</td><td>0.24</td></tr><tr><td>#P.</td><td>1706.03</td><td>43.26</td><td>68.03</td><td>43.26</td><td>68.03</td><td>43.26</td><td>68.03</td><td>43.26</td><td>68.03</td></tr></table>

Theorem 4.1. Assume that $\mathcal { L }$ is differentiable andfor any two different tasks $\mathcal { T } _ { i }$ and $\tau _ { j } ,$ it satisfies

$$
\cos \phi_ {i j} ^ {(k)} \| \mathbf {g} _ {i} ^ {(k)} \| <   \| \mathbf {g} _ {j} ^ {(k)} \|, \quad \forall k \in \mathbb {P} \tag {6}
$$

then for any sufficiently small learning rate α $> 0 ,$

$$
\mathcal {L} (\hat {\theta} _ {r}) <   \mathcal {L} (\hat {\theta}). \tag {7}
$$

The theorem indicates that a single gradient update on the model parameters of Recon achieves lower loss than that on the original model parameters. The proof is provided in Appendix A

## 5 EXPERIMENTS

In this section, we conduct extensive experiments to evaluate our approach Recon for multi-task learning and demonstrate its effectiveness, efficiency and generality.

## 5.1 EXPERIMENTAL SETUP

Datasets. We evaluate Recon on 4 multi-task datasets, namely Multi-Fashion+MNIST (Lin et al., 2019), CityScapes (Cordts et al., 2016), NYUv2 (Couprie et al., 2013), PASCAL-Context (Mottaghi et al., 2014), and CelebA (Liu et al., 2015). The tasks of each dataset are described as follows. 1) Multi-Fashion+MNIST contains two image classification tasks. Each image consists of an item from FashionMNIST and an item from MNIST. 2) CityScapes contains 2 vision tasks: 7-class semantic segmentation and depth estimation. 3) NYUv2 contains 3 tasks: 13-class semantic segmentation, depth estimation and normal prediction. 4) PASCAL-Context consists of 5 tasks: semantic segmentation, human parts segmentation and saliency estimation, surface normal estimation, and edge detection. 5) CelebA contains 40 binary classification tasks.

Baselines. The baselines include 1) single-task learning (single-task): training all tasks independently; 2) joint-training (joint-train): training all tasks together with equal loss weights and all parameters shared; 3) gradient manipulation methods: MGDA (Sener & Koltun, 2018), PCGrad (Yu et al., 2020), GradDrop (Chen et al., 2020), CAGrad (Liu et al., 2021a), RotoGrad (Javaloy & Valera, 2022); 4) branched architecture search methods: BMTAS (Bruggemann et al., 2020); 5) Architecture design methods: Cross-Stitch (Misra et al., 2016), MMoE (Ma et al., 2018). Following Liu et al. (2021a), we implement Cross-Stitch based on SegNet (Badrinarayanan et al., 2017). For a fair comparison, all methods use same configurations and random seeds. We run all experiments 3 times with different random seeds. More experimental details are provided in Appendix B.

Relative task improvement. Following Maninis et al. (2019), we compute the relative task improvement with respect to the single-task baseline for each task. Given a task $\tau _ { j }$ , the relative task improvement is $\begin{array} { r } { \Delta m _ { \mathcal T _ { j } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } ( - 1 ) ^ { l _ { i } } ( M _ { i } - S _ { i } ) / S _ { i } } \end{array}$ , where $M _ { i } , S _ { i }$ refer to metrics for the $i ^ { \mathrm { { t h } } }$ criterion obtained by objective model and single-task model respectively, $l _ { i } = 1$ if a lower value for the criterion is better and 0 otherwise. The average relative task improvement is $\begin{array} { r } { \Delta m = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } \Delta m \tau _ { j } } \end{array}$

Table 3: Multi-task learning results on CityScapes dataset. All experiments are repeated over 3 random seeds and the mean values are reported. $\Delta m \%$ denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="3"> $\Delta m\% \uparrow$ </td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td></tr><tr><td>Single-task</td><td>74.36</td><td>93.22</td><td>0.0128</td><td>29.98</td><td></td><td>190.59</td></tr><tr><td>Cross-Stitch</td><td>74.05</td><td>93.17</td><td>0.0162</td><td>116.66</td><td>-79.04</td><td>190.59</td></tr><tr><td>RotoGrad</td><td>73.38</td><td>92.97</td><td>0.0147</td><td>82.31</td><td>-47.81</td><td>103.43</td></tr><tr><td>Joint-train</td><td>74.13</td><td>93.13</td><td>0.0166</td><td>116.00</td><td>-79.32</td><td>95.43</td></tr><tr><td>w/ Recon</td><td>74.17</td><td>93.21</td><td>0.0136</td><td>43.18</td><td>-12.63</td><td>108.44</td></tr><tr><td>MGDA</td><td>70.74</td><td>92.19</td><td>0.0130</td><td>47.09</td><td>-16.22</td><td>95.43</td></tr><tr><td>w/ Recon</td><td>71.01</td><td>92.17</td><td>0.0129</td><td>33.41</td><td>-4.46</td><td>108.44</td></tr><tr><td>Graddrop</td><td>74.08</td><td>93.08</td><td>0.0173</td><td>115.79</td><td>-80.48</td><td>95.43</td></tr><tr><td>w/ Recon</td><td>74.17</td><td>93.11</td><td>0.0134</td><td>41.37</td><td>-10.69</td><td>108.44</td></tr><tr><td>PCGrad</td><td>73.98</td><td>93.08</td><td>0.02</td><td>114.50</td><td>-78.39</td><td>95.43</td></tr><tr><td>w/ Recon</td><td>74.18</td><td>93.14</td><td>0.0136</td><td>46.02</td><td>-14.92</td><td>108.44</td></tr><tr><td>CAGrad</td><td>73.81</td><td>93.02</td><td>0.0153</td><td>88.29</td><td>-53.81</td><td>95.43</td></tr><tr><td>w/ Recon</td><td>74.22</td><td>93.10</td><td>0.0130</td><td>38.27</td><td>-7.38</td><td>108.44</td></tr></table>

![](images/9a380c920d3e70b1e94421a2509df217eb1c275b54f481aea9de8edede5b920b.jpg)

<details>
<summary>line</summary>

| K | \(\Delta m %\) |
| --- | --- |
| 5 | ~-0.67 |
| 10 | ~-0.27 |
| 15 | ~-0.31 |
| 20 | ~-0.24 |
| 25 | ~-0.03 |
| 30 | ~-0.02 |
| 35 | ~-0.07 |
</details>

(a)

![](images/9e165dd5c956df597c856e4f0e37d90a9006236bfcdc1235a40a2b047f77d499.jpg)

<details>
<summary>line</summary>

| S | Performance \((\Delta m\) %) |
| --- | --- |
| -0.15 | ~-0.14 |
| -0.10 | ~-0.03 |
| -0.05 | ~-0.07 |
| 0.00 | ~-0.37 |
</details>

(b)  
Figure 3: The performance of CAGrad combined with Recon on the Multi-Fashion+MNIST benchmark with (a) different number of selected layers K (b) different severity value S for computing conflict scores.

## 5.2 COMPARISON WITH THE STATE-OF-THE-ART

Recon improves the performance of all base models. The main results on Multi-Fashion+MNIST, and CelebA, CityScapes, PASCAL-Context, and NYUv2, are presented in Table 1, Table 2, Table 3, Table 4, and Table 5 respectively. (1) Compared to gradient manipulation methods, Recon consistently improves their performance in most evaluation metrics, and achieve comparable performance on the rest of evaluation metrics. (2) Compared with branched architecture search methods and architecture design methods, Recon can further improve the performance of BMTAS and MMoE. Besides, Recon combined with other gradient manipulation methods with small model size can achieve better results than branched architecture search methods with much bigger models.

Small increases in model parameters can lead to good performance gains. Note that Recon only changes a small portion of shared parameters to task-specific. As shown in Table 1-5, Recon increases the model size by 0.52% to 57.25%. Recon turns 1.42%, 1.46%, 12.77%, 0.26%, 9.80% shared parameters to task-specific on Multi-Fashion+MNIST, CelebA, CityScapes, NYUv2 and PASCAL-Context respectively. The results suggest that the gradient conflicts in a small portion (less than 13%) of shared parameters impede the training of the model for multi-task learning.

Recon is compatible with various neural network architectures. We use ResNet18 on Multi-Fashion+MNIST, SegNet (Badrinarayanan et al., 2017) on CityScapes, MTAN (Liu et al., 2019) on NYUv2, and MobileNetV2 (Sandler et al., 2018) on PASCAL-Context. Recon improves the performance of baselines with different neural network architectures, including the architecture search method BMTAS (Bruggemann et al., 2020) which finds a tree-like structure for multi-task learning.

Only one search of conflict layers is needed for the same network architecture. An interesting observation from our experiments is that network architecture seems to be the deciding factor for the conflict layers found by Recon. With the same network architecture (e.g., ResNet18), the found conflict layers are quite consistent w.r.t. (1) different training stages (e.g., the first 25% iterations, or the middle or last ones) (see Table 12 and Table 13 and discussion in Appendix C), (2) different MTL methods (e.g., joint-training or gradient manipulation methods) (see Table 14 and discussion in Appendix C), and (3) different datasets (see Table 15 and Table 16 and discussion in Appendix C).

Table 4: Multi-task learning results on PASCAL-Context dataset with 4-task setting. All experiments are repeated over 3 random seeds and the mean values are reported. $\Delta m \%$ denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates Recon improves the result of the base model. The best average result is marked in bold.

<table><tr><td rowspan="3">Method</td><td colspan="2">SemSeg</td><td colspan="2">PartSeg</td><td>saliency</td><td colspan="4">Surface Normal</td><td rowspan="3"> $\Delta m\% \uparrow$ </td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td>(Higher Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="2">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>mIoU</td><td>Pix Acc</td><td>mIoU</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td></tr><tr><td>Single-task</td><td>65.00</td><td>90.53</td><td>59.59</td><td>92.61</td><td>65.61</td><td>14.55</td><td>12.36</td><td>46.51</td><td>81.29</td><td></td><td>30.09</td></tr><tr><td rowspan="2">Joint-trainw/ Recon</td><td>64.06</td><td>90.45</td><td>57.91</td><td>92.17</td><td>62.71</td><td>16.40</td><td>14.23</td><td>39.38</td><td>75.93</td><td>-4.82</td><td>8.04</td></tr><tr><td>64.73</td><td>90.50</td><td>59.00</td><td>92.44</td><td>66.17</td><td>14.99</td><td>12.68</td><td>44.82</td><td>80.11</td><td>-0.66</td><td>10.20</td></tr><tr><td rowspan="2">MGDAw/ Recon</td><td>46.05</td><td>86.62</td><td>54.82</td><td>91.39</td><td>64.76</td><td>15.77</td><td>13.54</td><td>41.98</td><td>77.82</td><td>-7.67</td><td>8.04</td></tr><tr><td>55.82</td><td>87.73</td><td>56.31</td><td>91.67</td><td>64.91</td><td>15.12</td><td>12.88</td><td>44.36</td><td>79.81</td><td>-4.14</td><td>10.20</td></tr><tr><td rowspan="2">PCGradw/ Recon</td><td>63.91</td><td>90.45</td><td>58.01</td><td>92.19</td><td>63.09</td><td>16.34</td><td>14.19</td><td>39.62</td><td>76.06</td><td>-4.59</td><td>8.04</td></tr><tr><td>65.02</td><td>90.45</td><td>59.22</td><td>92.46</td><td>66.14</td><td>14.95</td><td>12.73</td><td>44.96</td><td>80.22</td><td>-0.55</td><td>10.20</td></tr><tr><td rowspan="2">Graddropw/ Recon</td><td>64.14</td><td>90.34</td><td>57.62</td><td>92.12</td><td>62.64</td><td>16.46</td><td>14.28</td><td>39.29</td><td>75.71</td><td>-5.00</td><td>8.04</td></tr><tr><td>64.48</td><td>90.45</td><td>59.08</td><td>92.46</td><td>66.23</td><td>14.94</td><td>12.72</td><td>45.03</td><td>80.25</td><td>-0.63</td><td>10.20</td></tr><tr><td rowspan="2">CAGradw/ Recon</td><td>63.37</td><td>90.17</td><td>57.49</td><td>92.07</td><td>64.16</td><td>16.30</td><td>14.12</td><td>39.80</td><td>76.23</td><td>-4.37</td><td>8.04</td></tr><tr><td>64.60</td><td>90.40</td><td>59.27</td><td>92.47</td><td>65.67</td><td>14.92</td><td>12.71</td><td>45.10</td><td>80.33</td><td>-0.76</td><td>10.20</td></tr><tr><td rowspan="2">BMTASw/ Recon</td><td>64.89</td><td>90.44</td><td>58.87</td><td>92.36</td><td>63.42</td><td>15.66</td><td>13.44</td><td>42.29</td><td>78.14</td><td>-2.89</td><td>15.18</td></tr><tr><td>64.78</td><td>90.46</td><td>59.96</td><td>92.58</td><td>65.96</td><td>14.74</td><td>12.57</td><td>45.62</td><td>80.84</td><td>-0.19</td><td>16.83</td></tr></table>

Table 5: Multi-task learning results on NYUv2 dataset with MTAN as backbone. All experiments are repeated over 3 random seeds and the mean values are reported. $\Delta m \%$ denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon improves the result of the base model. The best average result is marked in bold.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m\% \uparrow$ </td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="3">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>Single-task</td><td>38.67</td><td>64.27</td><td>0.6881</td><td>0.2788</td><td>24.87</td><td>18.99</td><td>30.43</td><td>57.81</td><td>69.70</td><td></td><td>285.88</td></tr><tr><td>Cross-Stitch</td><td>40.45</td><td>66.15</td><td>0.5051</td><td>0.2134</td><td>27.58</td><td>23.00</td><td>24.69</td><td>49.47</td><td>62.36</td><td>4.16</td><td>285.88</td></tr><tr><td>Joint-train</td><td>39.48</td><td>65.23</td><td>0.5491</td><td>0.2235</td><td>27.87</td><td>23.76</td><td>22.68</td><td>47.91</td><td>61.58</td><td>0.75</td><td>168.72</td></tr><tr><td>w/ Recon</td><td>39.54</td><td>65.20</td><td>0.5312</td><td>0.2234</td><td>26.55</td><td>21.40</td><td>26.53</td><td>52.60</td><td>65.31</td><td>4.14</td><td>169.59</td></tr><tr><td>MGDA</td><td>29.28</td><td>60.30</td><td>0.6027</td><td>0.2515</td><td>24.89</td><td>19.32</td><td>29.85</td><td>57.18</td><td>69.38</td><td>-2.26</td><td>168.72</td></tr><tr><td>w/ Recon</td><td>32.82</td><td>61.26</td><td>0.5884</td><td>0.2295</td><td>25.17</td><td>19.72</td><td>28.18</td><td>56.49</td><td>68.96</td><td>0.53</td><td>169.59</td></tr><tr><td>Graddrop</td><td>38.70</td><td>64.97</td><td>0.5565</td><td>0.2333</td><td>27.41</td><td>23.00</td><td>23.79</td><td>49.45</td><td>62.87</td><td>0.49</td><td>168.72</td></tr><tr><td>w/ Recon</td><td>40.14</td><td>66.08</td><td>0.5265</td><td>0.2241</td><td>26.51</td><td>21.45</td><td>26.51</td><td>52.48</td><td>65.26</td><td>4.67</td><td>169.59</td></tr><tr><td>PCGrad</td><td>38.55</td><td>65.07</td><td>0.54</td><td>0.23</td><td>26.90</td><td>22.05</td><td>24.98</td><td>51.36</td><td>64.41</td><td>2.02</td><td>168.72</td></tr><tr><td>w/ Recon</td><td>38.61</td><td>65.48</td><td>0.5350</td><td>0.2271</td><td>26.31</td><td>21.11</td><td>26.90</td><td>53.21</td><td>65.95</td><td>3.87</td><td>169.59</td></tr><tr><td>CAGrad</td><td>39.89</td><td>66.47</td><td>0.5496</td><td>0.2281</td><td>26.36</td><td>21.47</td><td>25.50</td><td>52.68</td><td>65.90</td><td>3.74</td><td>168.72</td></tr><tr><td>w/ Recon</td><td>39.92</td><td>66.07</td><td>0.5320</td><td>0.2200</td><td>25.80</td><td>20.59</td><td>27.60</td><td>54.31</td><td>67.05</td><td>5.80</td><td>169.59</td></tr></table>

Hence, in our experiments, we only search for the conflict layers once with the joint-training baseline in the first 25% training iterations and modify the network to improve various methods on the same dataset. We also find that the conflict layers found on one dataset can be used to modify the network to be directly applied on another dataset to gain performance improvement.

## 5.3 ABLATION STUDY AND ANALYSIS

Recon greatly reduces the occurrence of conflicting gradients. In Fig. 4 and Table 6, we compare the distribution of cos $\phi _ { i j }$ before and after applying Recon on Multi-Fashion+MNIST (the results on other datasets are provided in Appendix C). It can be seen that Recon greatly reduces the numbers of gradient pairs with severe conflicts (cos $\phi _ { i j } ~ \in ~ ( - 0 . 0 1 , - 1 ] )$ by at least 67% and up to 79% when compared with joint-training, while gradient manipulation methods only slightly reduce the percentage and some even increases it. Similar observations can be made from Tables 8-10.

Randomly selecting conflict layers does not work. To show that the performance gain of Recon comes from selecting the layers with most severe conflicts instead of merely increasing model parameters, we further compare Recon with the following two baselines. RSL: randomly selecting

![](images/f03535154e1051a3525ad2b43e7d6297cf9b75c841cb60cd53a345a7f55d135a.jpg)  
Figure 4: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ of baselines and baselines with Recon on Multi-Fashion+MNIST dataset.

Table 6: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ w.r.t. the shared parameters on Multi-Fashion+MNIST dataset. “Reduction” means the percentage of conflicting gradients in the interval of $( - 0 . 0 1 , - 1 . 0 ]$ reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some method even increases it.

<table><tr><td> $\cos\phi_{ij}$ </td><td>Joint-train</td><td>w/ RSL</td><td>w/ RSP</td><td>w/ Recon</td><td>MGDA</td><td>w/ Recon</td><td>Graddrop</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td></tr><tr><td>[1.0, 0)</td><td>56.56</td><td>53.44</td><td>58.15</td><td>58.53</td><td>56.06</td><td>56.50</td><td>57.26</td><td>57.61</td><td>56.72</td><td>57.75</td><td>56.18</td><td>59.06</td></tr><tr><td>(0, -0.01]</td><td>31.25</td><td>27.35</td><td>34.33</td><td>37.67</td><td>32.36</td><td>40.93</td><td>31.06</td><td>38.28</td><td>31.19</td><td>38.76</td><td>31.25</td><td>37.84</td></tr><tr><td>(-0.01, -0.02]</td><td>9.26</td><td>13.45</td><td>6.38</td><td>3.04</td><td>8.87</td><td>2.12</td><td>8.93</td><td>3.32</td><td>9.09</td><td>2.87</td><td>9.37</td><td>2.44</td></tr><tr><td>(-0.02, -0.03]</td><td>2.05</td><td>4.18</td><td>0.8</td><td>0.5</td><td>1.71</td><td>0.26</td><td>1.72</td><td>0.54</td><td>1.90</td><td>0.42</td><td>2.00</td><td>0.41</td></tr><tr><td>(-0.03, -1.0]</td><td>1.25</td><td>1.58</td><td>0.34</td><td>0.25</td><td>1.0</td><td>0.18</td><td>1.03</td><td>0.26</td><td>1.10</td><td>0.2</td><td>1.20</td><td>0.25</td></tr><tr><td>Reduction (%)</td><td>-</td><td>-52.94</td><td>40.13</td><td>69.82</td><td>7.80</td><td>79.62</td><td>7.01</td><td>67.20</td><td>3.74</td><td>72.21</td><td>-0.08</td><td>75.32</td></tr></table>

Table 7: Comparison of Recon with RSL and RSP. PD: performance drop compared to Recon.

<table><tr><td rowspan="3">Seed</td><td rowspan="3">w/ RSL</td><td rowspan="3">w/ RSP</td><td rowspan="3">w/ Recon</td><td colspan="5">CAGrad</td><td colspan="5">PCGrad</td></tr><tr><td colspan="2">Task 1</td><td colspan="2">Task2</td><td rowspan="2">#P.</td><td colspan="2">Task 1</td><td colspan="2">Task2</td><td rowspan="2">#P.</td></tr><tr><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td></tr><tr><td>0</td><td>✓</td><td></td><td></td><td>97.60</td><td>0.68</td><td>64.39</td><td>25.26</td><td>73.02</td><td>97.43</td><td>0.87</td><td>65.57</td><td>24.21</td><td>73.02</td></tr><tr><td>1</td><td>✓</td><td></td><td></td><td>97.11</td><td>1.18</td><td>87.61</td><td>2.04</td><td>83.63</td><td>94.92</td><td>3.39</td><td>87.31</td><td>2.46</td><td>83.63</td></tr><tr><td>2</td><td>✓</td><td></td><td></td><td>94.62</td><td>3.66</td><td>87.68</td><td>1.96</td><td>76.33</td><td>92.90</td><td>5.40</td><td>87.41</td><td>2.36</td><td>76.33</td></tr><tr><td>0</td><td></td><td>✓</td><td></td><td>97.11</td><td>1.18</td><td>85.57</td><td>4.07</td><td>52.25</td><td>96.93</td><td>1.38</td><td>88.16</td><td>1.62</td><td>52.25</td></tr><tr><td>1</td><td></td><td>✓</td><td></td><td>97.81</td><td>0.47</td><td>88.28</td><td>1.36</td><td>51.96</td><td>97.63</td><td>0.68</td><td>88.55</td><td>1.22</td><td>51.96</td></tr><tr><td>2</td><td></td><td>✓</td><td></td><td>81.18</td><td>17.10</td><td>76.56</td><td>13.09</td><td>47.50</td><td>88.71</td><td>9.59</td><td>84.51</td><td>5.27</td><td>47.50</td></tr><tr><td>-</td><td>-</td><td>-</td><td>✓</td><td>98.28</td><td>0</td><td>89.65</td><td>0</td><td>43.42</td><td>98.30</td><td>0</td><td>89.77</td><td>0</td><td>43.42</td></tr></table>

same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific. The results in Table 7 show that both RSL and RSP lead to significant performance drops, which verifies the effectiveness of the selection strategy of Recon. We compare Recon with the baselines that selects the first or last K layers in Appendix C.

Ablation study on hyperparameters. We study the influence of the conflict severity S and the number of selected layers K on the performance of CAGrad w/ Recon on Multi-Fashion+MNIST. As shown in Fig. 3, a small K leads to a significant performance drop, which indicates that there are still some shared network layers suffering from severe gradient conflicts, while a large K will not lead to further performance improvement since severe conflicts have been resolved. For the conflict severity S, we find that a high value of S (e.g., 0.0) leads to performance drops since it includes too many gradient pairs with small conflicts, while some of them are helpful for learning common structures and should not be removed. In the meantime, a too small $S \left( \mathrm { e . g . , - 0 . 1 5 } \right)$ also leads to performance degradation because it ignores too many gradient pairs with large conflicts, which may be detrimental to learning. While K and S are sensitive, we may only need to tune them once for a given network architecture, as discussed in Sec. 5.2.

## 6 CONCLUSION

We have proposed a very simple yet effective approach, namely Recon, to reduce the occurrence of conflicting gradients for multi-task learning. By considering layer-wise gradient conflicts and identifying the shared layers with severe conflicts and setting them task-specific, Recon can significantly reduce the occurrence of severe conflicting gradients and boost the performance of existing methods with only a reasonable increase in model parameters. We have demonstrated the effectiveness, efficiency, and generality of Recon via extensive experiments and analysis.

## ACKNOWLEDGMENTS

The authors would like to thank Lingzi Jin for checking the proof of Theorem A.1 and the anonymous reviewers for their insightful and helpful comments.

## REFERENCES

Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoderdecoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1385–1394, 2019.  
David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. British Machine Vision Conference (BMVC), 2020.  
Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.  
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pp. 794–803. PMLR, 2018.  
Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2016.  
Camille Couprie, Clement Farabet, Laurent Najman, and Yann LeCun. Indoor semantic segmenta-´ tion using depth information. CoRR, abs/1301.3572, 2013.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.  
Carlo D’Eramo, Davide Tateo, Andrea Bonarini, Marcello Restelli, Jan Peters, et al. Sharing knowledge in multi-task deep reinforcement learning. In International Conference on Learning Representations, pp. 1–11. OpenReview. net, 2020.  
Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34:27503–27516, 2021.  
Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3205–3214, 2019.  
Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In Proceedings ofthe IEEE/CVF Conference on computer vision and pattern recognition, pp. 11543–11552, 2020.  
Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pp. 270–287, 2018.  
Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International Conference on Machine Learning, pp. 3854–3863. PMLR, 2020.  
Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. Empirical Methods in Natural Language Processing (EMNLP), 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 770–778, 2016.  
Adrian Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In´ International Conference on Learning Representations, 2022.  
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7482–7491, 2018.  
Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6129–6138, 2017.  
Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. Pareto multi-task learning. Advances in Neural Information Processing Systems, 32, 2019.  
Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021a.  
Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2021b.  
Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1871–1880, 2019.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pp. 3730–3738, 2015.  
Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. Learning multiple tasks with multilinear relationship networks. Advances in Neural Information Processing Systems, 30, 2017.  
Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1930–1939, 2018.  
Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1851–1860, 2019.  
Elliot Meyerson and Risto Miikkulainen. Beyond shared hierarchies: Deep multitask learning through soft layer ordering. In International Conference on Learning Representations, 2018.  
Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3994–4003, 2016.  
Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 891–898, 2014.  
Emilio Parisotto, Lei Jimmy Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In International Conference on Learning Representations, 2016.  
Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. In International Conference on Learning Representations, 2018.  
Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.  
Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pp. 4822–4829, 2019.  
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4510–4520, 2018.  
Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in Neural Information Processing Systems, 31, 2018.  
Jiayi Shen, Xiantong Zhen, Marcel Worring, and Ling Shao. Variational multi-task learning with gumbel-softmax priors. Advances in Neural Information Processing Systems, 34:21031–21042, 2021.  
Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International Conference on Machine Learning, pp. 9120–9132. PMLR, 2020.  
Sebastian Thrun and Joseph O’Sullivan. Discovering structure in multiple learning tasks: The tc algorithm. In International Conference on Machine Learning, volume 96, pp. 489–497, 1996.  
Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE transactions on pattern analysis and machine intelligence, 2021.  
Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. Advances in Neural Information Processing Systems, 33:4767–4777, 2020.  
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.  
Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3712–3722, 2018.

## A PROOF OF THEOREM A.1

Theorem A.1. Assume that L is differentiable and for any two different tasks $\mathcal { T } _ { i }$ and ${ \mathcal { T } } _ { j } ,$ , it satisfies

$$
\cos \phi_ {i j} ^ {(k)} \| \mathbf {g} _ {i} ^ {(k)} \| <   \| \mathbf {g} _ {j} ^ {(k)} \|, \quad \forall k \in \mathbb {P} \tag {8}
$$

thenfor any sufficiently small learning rate α $\begin{array} { r } { \iota > 0 , } \end{array}$

$$
\mathcal {L} (\hat {\theta} _ {r}) <   \mathcal {L} (\hat {\theta}). \tag {9}
$$

Proof. We consider the first order Taylor approximation of $\mathcal { L } _ { i }$ . For normal update, we have

$$
\mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) = \mathcal {L} _ {i} \left(\theta_ {\mathrm{sh}} ^ {\mathrm{fix}}, \theta_ {\mathrm{sh}} ^ {\mathrm{cf}}, \theta_ {i} ^ {\mathrm{ts}}\right) + (\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}} - \theta_ {\mathrm{sh}} ^ {\mathrm{fix}}) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{fix}} \tag {10}
$$

$$
+ (\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}} - \theta_ {\mathrm{sh}} ^ {\mathrm{cf}}) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{cf}} + (\hat {\theta} _ {i} ^ {\mathrm{ts}} - \theta_ {i} ^ {\mathrm{ts}}) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{ts}} + o (\alpha). \tag {11}
$$

For Recon update, we have

$$
\mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {i} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) = \mathcal {L} _ {i} \left(\theta_ {\mathrm{sh}} ^ {\mathrm{fix}}, \theta_ {\mathrm{sh}} ^ {\mathrm{cf}}, \theta_ {i} ^ {\mathrm{ts}}\right) + \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}} - \theta_ {\mathrm{sh}} ^ {\mathrm{fix}}\right) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{ts}} \tag {12}
$$

$$
+ \left(\hat {\theta} _ {i} ^ {\mathrm{cf}} - \theta_ {\mathrm{sh}} ^ {\mathrm{cf}}\right) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{cf}} + \left(\hat {\theta} _ {i} ^ {\mathrm{ts}} - \theta_ {i} ^ {\mathrm{ts}}\right) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{ts}} + o (\alpha). \tag {13}
$$

The difference between the two loss functions after the update is

$$
\mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {i} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) - \mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) = (\hat {\theta} _ {i} ^ {\mathrm{cf}} - \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}}) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{cf}} + o (\alpha) \tag {14}
$$

$$
= - \alpha \left(\mathbf {g} _ {i} ^ {\mathrm{cf}} - \sum_ {j = 1} ^ {T} w _ {j} \mathbf {g} _ {j} ^ {\mathrm{cf}}\right) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{cf}} + o (\alpha) \tag {15}
$$

$$
= - \alpha \sum_ {j = 1} ^ {T} w _ {j} \left(\mathbf {g} _ {i} ^ {\mathrm{cf}} - \mathbf {g} _ {j} ^ {\mathrm{cf}}\right) ^ {\top} \mathbf {g} _ {i} ^ {\mathrm{cf}} + o (\alpha) \tag {16}
$$

$$
= - \alpha \sum_ {j = 1} ^ {T} w _ {j} \left(\| \mathbf {g} _ {i} ^ {\mathrm{cf}} \| ^ {2} - \mathbf {g} _ {j} ^ {\mathrm{cf} \top} \mathbf {g} _ {i} ^ {\mathrm{cf}}\right) + o (\alpha). \tag {17}
$$

Assume, without loss of generality, that $\| \mathbf { g } _ { i } ^ { c f } \| \neq 0 ;$ , then

$$
\left\| \mathbf {g} _ {i} ^ {\mathrm{cf}} \right\| ^ {2} - \mathbf {g} _ {j} ^ {\mathrm{cf} \top} \mathbf {g} _ {i} ^ {\mathrm{cf}} = \sum_ {k \in \mathbb {P}} \left(\left\| \mathbf {g} _ {i} ^ {(k)} \right\| ^ {2} - \mathbf {g} _ {i} ^ {(k) \top} \mathbf {g} _ {j} ^ {(k)}\right) \tag {18}
$$

$$
= \sum_ {k \in \mathbb {P}} \left\| \mathbf {g} _ {i} ^ {(k)} \right\| \left(\left\| \mathbf {g} _ {i} ^ {(k)} \right\| - \cos \phi_ {i j} ^ {(k)} \left\| \mathbf {g} _ {j} ^ {(k)} \right\|\right) \tag {19}
$$

$$
> 0. \tag {20}
$$

Hence, the above difference is negative, if α is sufficiently small. As such, the difference between the multi-task loss functions is also negative, if α is sufficiently small.

$$
\mathcal {L} (\hat {\theta} _ {r}) - \mathcal {L} (\hat {\theta}) = \sum_ {i = 1} ^ {T} \mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {i} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) - \sum_ {i = 1} ^ {T} \mathcal {L} _ {i} \left(\hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{fix}}, \hat {\theta} _ {\mathrm{sh}} ^ {\mathrm{cf}}, \hat {\theta} _ {i} ^ {\mathrm{ts}}\right) <   0 \tag {21}
$$

![](images/bcde0edefceaca78acc51bc85a21ec952e5752c7c3bd346ead11051a7732e4f5.jpg)

## B EXPERIMENTAL SETUP

## B.1 MULTI-FASHION+MNIST

Model. We adopt ResNet18 (He et al., 2016) without pre-training as the backbone and modify the dimension of the output features to 100 for the last linear layer. For the task-specific heads, we define two linear layers followed by a ReLU function.

Tasks, losses, and metrics. Each task is a classification problem with 10 classes and we use the cross-entropy loss as the classification loss. For evaluation, we use the classification accuracy as the metric for each task.

Model hyperparameters. We train the model for 120 epochs with the batch size of 256. We adopt SGD with an initial learning rate of 0.1 and decay the learning rate by 0.1 at the $6 0 ^ { \mathrm { t h } }$ and $9 0 ^ { \mathrm { t h } }$ epoch.

Baseline hyperparameters. For CAGrad, we set $\alpha = 0 . 2 .$ . For BMTAS, we set the resource loss weight to 1.0, and we search the architecture for 100 epochs. For RotoGrad, we set $R _ { k } = 1 0 0$ which is equal to the dimension of shared features and set the learning rate of rotation parameters as learning rate of the neural networks. For MMoE, the initial learning rate of expert networks and gates are 0.1 and $1 e - 3$ respectively.

Recon hyperparameters. We use CAGrad to train the model for 30 epochs and compute the conflict score of each shared layer. We set $S = - 0 . 1$ for computing the scores. We select 25 layers with the highest conflict scores and turn them into task-specific layers.

## B.2 CITYSCAPES

Model. We adopt SegNet (Badrinarayanan et al., 2017) as the backbone where the decoder is split into two convolutional heads.

Model hyperparameters. We train the model for 200 epochs with the batch size of 8. We adopt Adam with an initial learning rate of 5e − 5 and decay the learning rate by 0.5 at the $1 0 0 ^ { \mathrm { t h } }$ epoch.

Baselines hyperparameters. For CAGrad, we set $\alpha = 0 . 2$ . For RotoGrad, we set $R _ { k } = $ 1024 and set the learning rate of rotation parameters as 10 times less than the learning rate of the neural networks.

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set $S = 0 . 0$ for computing the scores. We select 39 layers with the highest conflict scores and turn them into task-specific layers.

## B.3 NYUV2

Model. We adopt MTAN (Liu et al., 2019) – the SegNet combined with task-specific attention modules on the encoder.

Model hyperparameters. We train the model for 200 epochs with the batch size of 2. We adopt Adam with an initial learning rate of 1e − 4 and decay the learning rate by 0.5 at the $1 0 0 ^ { \mathrm { t h } }$ epoch.

Baseline hyperparameters. For CAGrad, we set $\alpha = 0 . 4$ similar with Liu et al. (2021a).

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set $S = - 0 . 0 2$ for computing the scores. We select 22 layers with the highest conflict scores and turn them into task-specific layers.

## B.4 PASCAL-CONTEXT

Model. Following Bruggemann et al. (2020), we employ MobileNetv2 Sandler et al. (2018) as the backbone with a reduced design of the ASPP module (R-ASPP) (Sandler et al., 2018). We pre-train the model on ImageNet (Deng et al., 2009).

Model hyperparameters. We train the model for 130 epochs with the batch size of 6. We adopt Adam with an initial learning rate of $1 e - 4$ and decay the learning rate by 0.1 at the $7 0 ^ { \mathrm { t h } }$ and $1 0 0 ^ { \mathrm { { \dot { \mathrm { t h } } } } }$ epoch.

Baselines hyperparameters. For CAGrad, we set $\alpha = 0 . 1$ . For BMTAS, we set the resoure loss weight to 0.1, and we search the architecture for 130 epochs.

Recon hyperparameters. We use joint-train to train the model for 40 epochs and compute the conflict score of each shared layer. We set $S = - 0 . 0 2$ for computing the scores. We select 85 layers with the highest conflict scores and turn them into task-specific layers.

## B.5 CELEBA

Model. Following Sener & Koltun (2018), we use ResNet18 (He et al., 2016) as the backbone network. We pre-train the model on ImageNet (Deng et al., 2009).

Model hyperparameters. We train the model for 5 epochs. We adopt Adam with an initial learning rate of 5e − 5 and decay the learning rate by 0.5 at the 3<sup>th</sup> epoch.

Baselines hyperparameters. For CAGrad, we set $\alpha = 0 . 1$

Recon hyperparameters. We use joint-train to train the model for 2 epochs and compute the conflict score of each shared layer. We set $S = - 0 . 0 5$ . We select 25 layers with the highest conflict scores and turn them into task-specific layers.

## C ADDITIONAL ABLATION STUDY

The distribution of gradient conflicts. In addition to the statistics on Multi-Fashion+MNIST, we further show the distributions of gradient conflicts of various baselines on CityScapes, NYUv2, and PASCAL-Context in Fig 5, Fig 6, and Fig 7 respectively. We compare the distributions with those of baselines w/ Recon on the three datasets in Fig. 8, Fig. 9, and Fig. 10 respectively. The detailed statistics are provided in Tables 8-10.

![](images/b14b00f8f7bfc758b69192cafe9e40bd5b9ae1db81101359eb00389b33d2a126.jpg)

<details>
<summary>bar</summary>

| \(cos\phi ij\) Range | Joint-train | Graddrop | PCGrad | CAGrad | MGDA |
| --- | --- | --- | --- | --- | --- |
| [1.00, 0.08] | ~0.205 | ~0.220 | ~0.235 | ~0.190 | ~0.198 |
| (0.08, 0.06] | ~0.078 | ~0.077 | ~0.074 | ~0.075 | ~0.069 |
| (0.06, 0.04] | ~0.095 | ~0.091 | ~0.090 | ~0.098 | ~0.100 |
| (0.04, 0.02] | ~0.108 | ~0.102 | ~0.101 | ~0.119 | ~0.127 |
| (0.02, 0.00] | ~0.108 | ~0.106 | ~0.103 | ~0.124 | ~0.142 |
| (0.00, -0.02] | ~0.102 | ~0.096 | ~0.096 | ~0.111 | ~0.124 |
| (-0.02, -0.04] | ~0.085 | ~0.082 | ~0.081 | ~0.089 | ~0.087 |
| (-0.04, -0.06] | ~0.064 | ~0.066 | ~0.063 | ~0.062 | ~0.055 |
| (-0.06, -0.08] | ~0.048 | ~0.047 | ~0.043 | ~0.039 | ~0.033 |
| (-0.08, -1.00] | ~0.105 | ~0.113 | ~0.121 | ~0.092 | ~0.065 |
</details>

Figure 5: The distributions of gradient conflicts (in terms of cos $\phi _ { i j } )$ of the joint-training baseline and state-of-the-art gradient manipulation methods on CityScapes dataset.

![](images/9a28323017947348cef268d3b0f29ef418aae084fd4b2388dff6b2eb248eb611.jpg)

<details>
<summary>bar</summary>

| \(cos\phi_ij\) | Joint-train | Graddrop | PCGrad | CAGrad | MGDA |
| --- | --- | --- | --- | --- | --- |
| [1.00,0.08] | ~0.47 | ~0.47 | ~0.48 | ~0.47 | ~0.49 |
| (0.08,0.06] | ~0.04 | ~0.04 | ~0.04 | ~0.03 | ~0.03 |
| (0.06,0.04] | ~0.04 | ~0.04 | ~0.04 | ~0.04 | ~0.03 |
| (0.04,0.02] | ~0.04 | ~0.04 | ~0.04 | ~0.03 | ~0.03 |
| (0.02,0.00] | ~0.04 | ~0.04 | ~0.04 | ~0.03 | ~0.03 |
| (0.00,-0.02] | ~0.04 | ~0.04 | ~0.03 | ~0.03 | ~0.03 |
| (-0.02,-0.04] | ~0.03 | ~0.03 | ~0.03 | ~0.03 | ~0.03 |
| (-0.04,-0.06] | ~0.03 | ~0.03 | ~0.03 | ~0.03 | ~0.03 |
| (-0.06,-0.08] | ~0.03 | ~0.03 | ~0.03 | ~0.03 | ~0.03 |
| (-0.08,-1.00] | ~0.24 | ~0.23 | ~0.23 | ~0.25 | ~0.28 |
</details>

Figure 6: The distributions of gradient conflicts (in terms of cos $\phi _ { i j } )$ of the joint-training baseline and state-of-the-art gradient manipulation methods on NYUv2 dataset.

![](images/2f73a08e3fbf2a2ca54255feb05e02aa826cad5049f28800b04633a8e43c8041.jpg)

<details>
<summary>bar</summary>

| \(cos\phi_ij\) Range | Joint-train | Graddrop | PCGrad | CAGrad | MGDA |
| --- | --- | --- | --- | --- | --- |
| [1.00, 0.08] | ~0.24 | ~0.245 | ~0.24 | ~0.235 | ~0.27 |
| (0.08, 0.06] | ~0.06 | ~0.06 | ~0.06 | ~0.06 | ~0.055 |
| (0.06, 0.04] | ~0.075 | ~0.075 | ~0.075 | ~0.075 | ~0.07 |
| (0.04, 0.02] | ~0.09 | ~0.09 | ~0.095 | ~0.095 | ~0.08 |
| (0.02, 0.00] | ~0.145 | ~0.145 | ~0.145 | ~0.15 | ~0.13 |
| (0.00, -0.02] | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.085 |
| (-0.02, -0.04] | ~0.08 | ~0.08 | ~0.08 | ~0.08 | ~0.075 |
| (-0.04, -0.06] | ~0.06 | ~0.06 | ~0.06 | ~0.06 | ~0.055 |
| (-0.06, -0.08] | ~0.04 | ~0.045 | ~0.04 | ~0.045 | ~0.045 |
| (-0.08, -1.00] | ~0.115 | ~0.115 | ~0.115 | ~0.11 | ~0.135 |
</details>

Figure 7: The distributions of gradient conflicts (in terms of cos $\phi _ { i j } )$ of the joint-training baseline and state-of-the-art gradient manipulation methods on PASCAL-Context dataset.

![](images/e2b1576810acb81e5d7b6ee9941548bf504688688f75d1c2e1c63664ec2a57e0.jpg)

Figure 8: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ w.r.t. the shared parameters on CityScapes. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.  
![](images/cb6c4147d609e9f61200056aae5d1094c4c0aa8a3b953cd0483cae57591c6edb.jpg)

Figure 9: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ of baselines and baselines with Recon on NYUv2. RSL: randomly selecting same number of layers as Recon and set them taskspecific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.  
![](images/e3e47bd6b41c4291947ee294019f26de247488f5073689d06d39c2370707406b.jpg)  
Figure 10: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ of baselines and baselines with Recon on PASCAL-Context. RSL: randomly selecting same number of layers as Recon and set them task-specific. RSP: randomly selecting similar amount of parameters as Recon and set them task-specific.

Table 8: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ w.r.t. the shared parameters on CityScapes dataset. “Reduction” means the percentage of conflicting gradients in the interval of $( - \mathrm { { 0 . 0 2 } , - 1 . 0 ] }$ reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only moderately decrease their occurrence (MGDA deceases it by 22%), and some methods even increase it.

<table><tr><td> $\cos\phi_{ij}$ </td><td>Joint-train</td><td>w/ RSL</td><td>w/ RSP</td><td>w/ Recon</td><td>MGDA</td><td>w/ Recon</td><td>Graddrop</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td></tr><tr><td>[1.0, 0)</td><td>59.55</td><td>53.16</td><td>58.29</td><td>73.62</td><td>63.9</td><td>78.27</td><td>59.56</td><td>73.82</td><td>59.85</td><td>74.52</td><td>60.79</td><td>74.54</td></tr><tr><td>(0, -0.02]</td><td>10.14</td><td>9.01</td><td>10.77</td><td>20.13</td><td>12.51</td><td>12.54</td><td>9.61</td><td>19.75</td><td>9.58</td><td>19.43</td><td>11.13</td><td>19.77</td></tr><tr><td>(-0.02, -0.04]</td><td>8.52</td><td>7.34</td><td>8.72</td><td>5.13</td><td>8.59</td><td>5.54</td><td>8.19</td><td>5.17</td><td>7.94</td><td>4.89</td><td>8.83</td><td>4.62</td></tr><tr><td>(-0.04, -0.06]</td><td>6.45</td><td>5.69</td><td>6.48</td><td>0.94</td><td>5.39</td><td>2.23</td><td>6.49</td><td>1.05</td><td>6.24</td><td>0.96</td><td>6.05</td><td>0.89</td></tr><tr><td>(-0.06, -0.08]</td><td>4.79</td><td>4.53</td><td>4.61</td><td>0.14</td><td>3.29</td><td>0.85</td><td>4.76</td><td>0.16</td><td>4.41</td><td>0.15</td><td>4.06</td><td>0.13</td></tr><tr><td>(-0.08, -1.0]</td><td>10.54</td><td>20.26</td><td>11.13</td><td>0.03</td><td>6.33</td><td>0.56</td><td>11.38</td><td>0.05</td><td>11.98</td><td>0.06</td><td>9.13</td><td>0.04</td></tr><tr><td>Reduction (%)</td><td>-</td><td>-24.82</td><td>-2.11</td><td>79.41</td><td>22.11</td><td>69.70</td><td>-1.72</td><td>78.78</td><td>-0.89</td><td>80.03</td><td>7.36</td><td>81.22</td></tr></table>

Table 9: The distribution of gradient conflicts (in terms of cos $\phi _ { i j } )$ w.r.t. the shared parameters on NYUv2 dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.04, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it.

<table><tr><td> $\cos\phi_{ij}$ </td><td>Joint-train</td><td>w/ RSL</td><td>w/ RSP</td><td>w/ Recon</td><td>MGDA</td><td>w/ Recon</td><td>Graddrop</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td></tr><tr><td>[1.0, 0)</td><td>61.96</td><td>52.61</td><td>59.70</td><td>73.99</td><td>61.28</td><td>74.08</td><td>62.93</td><td>75.35</td><td>63.25</td><td>75.54</td><td>61.95</td><td>74.49</td></tr><tr><td>(0, -0.02]</td><td>3.85</td><td>3.75</td><td>3.47</td><td>14.17</td><td>2.97</td><td>13.38</td><td>3.83</td><td>13.50</td><td>3.61</td><td>12.66</td><td>3.53</td><td>14.20</td></tr><tr><td>(-0.02, -0.04]</td><td>3.63</td><td>3.60</td><td>3.41</td><td>7.07</td><td>2.77</td><td>7.21</td><td>3.70</td><td>6.71</td><td>3.62</td><td>6.66</td><td>3.39</td><td>6.96</td></tr><tr><td>(-0.04, -0.06]</td><td>3.39</td><td>3.43</td><td>3.11</td><td>2.89</td><td>2.81</td><td>3.19</td><td>3.45</td><td>2.71</td><td>3.26</td><td>2.98</td><td>3.21</td><td>2.71</td></tr><tr><td>(-0.06, -0.08]</td><td>3.11</td><td>3.30</td><td>2.94</td><td>1.13</td><td>2.64</td><td>1.28</td><td>3.16</td><td>1.03</td><td>3.06</td><td>1.25</td><td>3.05</td><td>1.01</td></tr><tr><td>(-0.08, -1.0]</td><td>24.05</td><td>33.31</td><td>27.37</td><td>0.76</td><td>27.53</td><td>0.87</td><td>22.92</td><td>0.70</td><td>23.20</td><td>0.90</td><td>24.88</td><td>0.63</td></tr><tr><td>Reduction (%)</td><td>-</td><td>-31.06</td><td>-9.39</td><td>84.35</td><td>-7.95</td><td>82.52</td><td>3.34</td><td>85.47</td><td>3.37</td><td>83.21</td><td>-1.93</td><td>85.76</td></tr></table>

Table 10: The distribution of gradient conflicts (in terms of $\cos \phi _ { i j } )$ w.r.t. the shared parameters on PASCAL-Context dataset. “Reduction” means the percentage of conflicting gradients in the interval of (−0.02, −1.0] reduced by the model compared with joint-training. The grey cell color indicates Recon greatly reduces the conflicting gradients (more than 50%). In contrast, gradient manipulation methods only slightly decrease their occurrence, and some methods even increase it.

<table><tr><td> $\cos\phi_{ij}$ </td><td>Joint-train</td><td>w/ RSL</td><td>w/ RSP</td><td>w/ Recon</td><td>MGDA</td><td>w/ Recon</td><td>Graddrop</td><td>w/ Recon</td><td>PCGrad</td><td>w/ Recon</td><td>CAGrad</td><td>w/ Recon</td></tr><tr><td>[1.0, 0)</td><td>61.26</td><td>59.20</td><td>60.47</td><td>63.99</td><td>60.40</td><td>63.61</td><td>61.18</td><td>63.76</td><td>61.35</td><td>63.83</td><td>60.99</td><td>63.78</td></tr><tr><td>(0, -0.02]</td><td>9.66</td><td>21.01</td><td>18.25</td><td>23.57</td><td>8.51</td><td>33.53</td><td>9.66</td><td>23.41</td><td>9.83</td><td>23.61</td><td>9.95</td><td>24.04</td></tr><tr><td>(-0.02, -0.04]</td><td>7.90</td><td>9.91</td><td>9.10</td><td>7.65</td><td>7.27</td><td>2.04</td><td>7.89</td><td>7.83</td><td>7.90</td><td>7.65</td><td>8.03</td><td>7.53</td></tr><tr><td>(-0.04, -0.06]</td><td>5.85</td><td>3.05</td><td>3.88</td><td>2.59</td><td>5.68</td><td>0.45</td><td>5.80</td><td>2.71</td><td>5.82</td><td>2.66</td><td>5.91</td><td>2.51</td></tr><tr><td>(-0.06, -0.08]</td><td>4.16</td><td>1.32</td><td>1.79</td><td>1.07</td><td>4.35</td><td>0.17</td><td>4.21</td><td>1.12</td><td>4.13</td><td>1.10</td><td>4.23</td><td>1.04</td></tr><tr><td>(-0.08, -1.0]</td><td>11.16</td><td>1.30</td><td>2.29</td><td>1.13</td><td>13.80</td><td>0.20</td><td>11.24</td><td>1.16</td><td>10.97</td><td>1.16</td><td>10.88</td><td>1.08</td></tr><tr><td>Reduction (%)</td><td>-</td><td>46.41</td><td>41.31</td><td>57.21</td><td>-6.98</td><td>90.16</td><td>-0.24</td><td>55.90</td><td>0.86</td><td>56.76</td><td>0.07</td><td>58.20</td></tr></table>

Table 11: Multi-task learning results on Multi-Fashion+MNIST dataset. LSK refers to turning the fist K layers into task-specific layers. FSK refers to turning the last K layers into task-specific layers. PD denotes the performance drop compared with Recon.

<table><tr><td rowspan="3">LSK</td><td rowspan="3">FSK</td><td rowspan="3">w/ Recon</td><td colspan="4">CAGrad</td><td rowspan="3">#P.</td><td colspan="5">PCGrad</td></tr><tr><td colspan="2">Task 1</td><td colspan="2">Task2</td><td colspan="2">Task 1</td><td colspan="2">Task2</td><td rowspan="2">#P.</td></tr><tr><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td><td>Acc↑</td><td>PD</td></tr><tr><td>✓</td><td></td><td></td><td>97.63</td><td>0.66</td><td>89.14</td><td>0.50</td><td>84.17</td><td>97.63</td><td>0.65</td><td>88.98</td><td>0.66</td><td>84.17</td></tr><tr><td></td><td>✓</td><td></td><td>98.21</td><td>0.07</td><td>89.15</td><td>0.50</td><td>48.90</td><td>98.19</td><td>0.09</td><td>89.51</td><td>0.13</td><td>48.90</td></tr><tr><td></td><td>-</td><td>✓</td><td>98.28</td><td>0</td><td>89.65</td><td>0</td><td>43.42</td><td>98.30</td><td>0</td><td>89.77</td><td>0</td><td>43.42</td></tr></table>

Selecting the first K layers and the last K Layers as conflict layers does not work. To further support the conclusion that the selection of parameters with higher probability of conflicting gradients contributes most to the performance gain rather than the increase in model capacity. We compare Recon with two baselines: (1) Select the first K neural network layers and turn them into task-specific layers. (2) Select the last K neural network layers and turn them into task-specific layers. The multi-task learning results on the Multi-Fashion+MNIST benchmark are presented in Table 11. The results show that if we directly turn the top or the bottom of the neural network into task-specific parameters, it still will lead to performance degradation compared to Recon.

Recon finds similar layers in different training stages. Recon ranks the network layers according to the computed S-conflict scores. The ranking result can be represented as a layer permutation, denoted as π, and $\pi ( l )$ is the position of layer l. The similarity between two rankings $\pi _ { i }$ and $\pi _ { j }$ can be measured as:

$$
d (\pi_ {i}, \pi_ {j}) = \frac {1}{| \mathbb {L} |} \sum_ {l \in \mathbb {L}} | \pi_ {i} (l) - \pi_ {j} (l) |, \tag {22}
$$

where L denotes the set of neural network layers. In Table 12, we measure the differences in rankings obtained in different training stages (e.g., in the first 25% iterations or the second 25% iterations)

Table 12: The distance between the layer permutations (rankings) obtained in different training stages on Multi-Fashion+MNIST dataset. “Iter.” denotes iterations.

<table><tr><td>Training Stage</td><td>1st 25% Iter.</td><td>2nd 25% Iter.</td><td>3rd 25% Iter.</td><td>4th 25% Iter.</td><td>All Iter.</td></tr><tr><td>1st 25% Iter.</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>2nd 25% Iter.</td><td>2.39</td><td>0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>3rd 25% Iter.</td><td>1.85</td><td>2.14</td><td>0</td><td>-</td><td>-</td></tr><tr><td>4th 25% Iter.</td><td>1.95</td><td>2.24</td><td>0.68</td><td>0</td><td>-</td></tr><tr><td>All Iter.</td><td>1.36</td><td>1.95</td><td>0.82</td><td>0.97</td><td>0</td></tr></table>

Table 13: Performance of the networks modified by Recon with conflict layers found in different training stages of joint-training on CityScapes dataset. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The best result is marked in bold.

<table><tr><td rowspan="3">Model</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="3">Δm%</td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td></tr><tr><td>Single-task</td><td>74.36</td><td>93.22</td><td>0.0128</td><td>29.98</td><td></td><td>190.59</td></tr><tr><td>1st 25% Iterations</td><td>74.17</td><td>93.21</td><td>0.0136</td><td>43.18</td><td>-12.63</td><td>108.439</td></tr><tr><td>2nd 25% Iterations</td><td>74.20</td><td>93.19</td><td>0.0135</td><td>42.45</td><td>-11.83</td><td>108.440</td></tr><tr><td>3rd 25% Iterations</td><td>74.80</td><td>93.19</td><td>0.0136</td><td>41.34</td><td>-10.90</td><td>109.567</td></tr><tr><td>4th 25% Iterations</td><td>74.80</td><td>93.19</td><td>0.0136</td><td>41.34</td><td>-10.90</td><td>109.567</td></tr><tr><td>All Iterations</td><td>74.80</td><td>93.19</td><td>0.0136</td><td>41.34</td><td>-10.90</td><td>109.567</td></tr></table>

on Multi-Fashion+MNIST by Eq. 22. The small distances (less than 2.4) indicate that the layers found in different training stages are quite similar. In Table 13, we compare the performance of the networks modified by Recon with conflict layers found in different training stages on CityScapes. It can be seen that the results of the last three rows are the same, which is because the layers found in the 3rd 25% iterations, 4th 25% iterations, and all iterations are exactly the same (the rankings may be slightly different though). The layers found in the later stages lead to slightly better performance than those found in the early stages (i.e., 1st 25% iterations and 2nd 25% iterations), indicating the conflict scores in early iterations might be a little noisy. However, since the performance gaps are acceptably small, to save time, we use the initial 25% training iterations to find conflict layers.

Table 14: The distance between the layer permutations (rankings) obtained by Recon with different methods on Multi-Fashion+MNIST dataset.

<table><tr><td>Method</td><td>Joint-train</td><td>CAGrad</td><td>PCGrad</td><td>Gradrop</td><td>MGDA</td></tr><tr><td>Joint-train</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CAGrad</td><td>1.07</td><td>0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PCGrad</td><td>0.78</td><td>1.17</td><td>0</td><td>-</td><td>-</td></tr><tr><td>Gradrop</td><td>0.59</td><td>0.83</td><td>0.68</td><td>0</td><td>-</td></tr><tr><td>MGDA</td><td>1.71</td><td>1.32</td><td>1.90</td><td>1.56</td><td>0</td></tr></table>

Recon finds similar layers with different MTL methods. In Table 14, we measure the differences in layer permutations (rankings) obtained by Recon with different methods (e.g., CAGrad and PC-Grad) on Multi-Fashion+MNIST by Eq. 22. The small distances (less than 1.9) indicate that the layers found by Recon with different methods are quite similar. Therefore, in our experiments, we only use joint-training to search for the conflict layers once, and directly apply the modified network to improve different gradient manipulation methods as shown in Tables 1-5.

The conflict layers found by Recon with the same architecture are transferable between different datasets. We conduct experiments with three different architectures: ResNet18, SegNet, and MTAN. (1) For Resnet18, we find that the layers found by Recon on CelebA and those found on Multi-Fashion+MNIST are exactly the same. (2) For SegNet, we find that 95% layers (38 out of 40)

Table 15: Multi-task learning results on NYUv2 dataset with SegNet as backbone. Recon<sup>∗</sup> denotes setting the layers found on CityScapes to task-specific. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon<sup>∗</sup> improves the result of the base model.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m\% \uparrow$ </td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="3">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>Single-task</td><td>38.67</td><td>64.27</td><td>0.6881</td><td>0.2788</td><td>24.8683</td><td>18.9919</td><td>30.43</td><td>57.81</td><td>69.7</td><td></td><td>285.88</td></tr><tr><td>Joint-train</td><td>38.62</td><td>65.36</td><td>0.5378</td><td>0.2273</td><td>29.92</td><td>25.82</td><td>20.79</td><td>44.29</td><td>57.36</td><td>-1.62</td><td>95.58</td></tr><tr><td>w/ Recon</td><td>40.68</td><td>66.12</td><td>0.5786</td><td>0.2558</td><td>26.72</td><td>21.41</td><td>26.58</td><td>52.58</td><td>65.20</td><td>2.15</td><td>139.59</td></tr><tr><td>w/ Recon*</td><td>38.81</td><td>63.69</td><td>0.5637</td><td>0.2413</td><td>26.75</td><td>21.73</td><td>26.16</td><td>51.80</td><td>64.64</td><td>1.59</td><td>121.59</td></tr><tr><td>MGDA</td><td>25.71</td><td>57.72</td><td>0.6033</td><td>0.2358</td><td>24.53</td><td>18.65</td><td>31.22</td><td>58.46</td><td>70.21</td><td>-2.15</td><td>95.58</td></tr><tr><td>w/ Recon</td><td>36.64</td><td>62.36</td><td>0.5613</td><td>0.2255</td><td>24.66</td><td>18.66</td><td>31.30</td><td>58.47</td><td>70.16</td><td>5.37</td><td>139.59</td></tr><tr><td>w/ Recon*</td><td>36.85</td><td>63.51</td><td>0.5760</td><td>0.2362</td><td>24.89</td><td>18.96</td><td>30.53</td><td>57.94</td><td>69.82</td><td>4.34</td><td>121.59</td></tr><tr><td>Graddrop</td><td>39.01</td><td>66.13</td><td>0.5462</td><td>0.2296</td><td>29.72</td><td>25.51</td><td>19.87</td><td>44.68</td><td>58.12</td><td>-1.52</td><td>95.58</td></tr><tr><td>w/ Recon</td><td>39.78</td><td>65.63</td><td>0.5460</td><td>0.2280</td><td>26.42</td><td>21.16</td><td>26.89</td><td>53.16</td><td>65.84</td><td>4.45</td><td>139.59</td></tr><tr><td>w/ Recon*</td><td>39.97</td><td>65.71</td><td>0.5544</td><td>0.2261</td><td>26.52</td><td>21.37</td><td>26.65</td><td>52.65</td><td>65.46</td><td>4.21</td><td>121.59</td></tr><tr><td>PCGrad</td><td>40.01</td><td>65.77</td><td>0.5349</td><td>0.2227</td><td>28.53</td><td>24.08</td><td>22.33</td><td>47.42</td><td>60.69</td><td>1.43</td><td>95.58</td></tr><tr><td>w/ Recon</td><td>40.03</td><td>65.92</td><td>0.5523</td><td>0.2384</td><td>26.24</td><td>20.89</td><td>27.30</td><td>53.66</td><td>66.25</td><td>4.19</td><td>139.59</td></tr><tr><td>w/ Recon*</td><td>39.93</td><td>65.46</td><td>0.5494</td><td>0.2315</td><td>26.82</td><td>21.70</td><td>26.34</td><td>52.04</td><td>64.74</td><td>3.53</td><td>121.59</td></tr><tr><td>CAGrad</td><td>38.87</td><td>66.54</td><td>0.5331</td><td>0.2289</td><td>25.85</td><td>20.60</td><td>27.50</td><td>54.41</td><td>67.10</td><td>5.60</td><td>95.58</td></tr><tr><td>w/ Recon</td><td>40.68</td><td>66.12</td><td>0.5372</td><td>0.2266</td><td>25.44</td><td>19.87</td><td>28.96</td><td>56.00</td><td>68.28</td><td>6.99</td><td>139.59</td></tr><tr><td>w/ Recon*</td><td>39.97</td><td>65.92</td><td>0.5298</td><td>0.2273</td><td>25.56</td><td>20.11</td><td>28.69</td><td>55.37</td><td>67.75</td><td>6.47</td><td>121.59</td></tr></table>

Table 16: Multi-task learning results on CityScapes dataset with MTAN as backbone. Recon<sup>∗</sup> denotes setting the layers found on NYUv2 to task-specific. ∆m% denotes the average relative improvement of all tasks. #P denotes the model size (MB). The grey cell color indicates that Recon or Recon<sup>∗</sup> improves the result of the base model.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="3"> $\Delta m\% \uparrow$ </td><td rowspan="3">#P.</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td></tr><tr><td>Single-task</td><td>73.74</td><td>93.05</td><td>0.0129</td><td>27.71</td><td></td><td>190.58</td></tr><tr><td>Joint-train</td><td>75.35</td><td>93.55</td><td>0.0169</td><td>45.64</td><td>-23.26</td><td>157.19</td></tr><tr><td>w/ Recon</td><td>75.72</td><td>93.74</td><td>0.0130</td><td>40.90</td><td>-11.36</td><td>196.32</td></tr><tr><td>w/ Recon*</td><td>76.32</td><td>93.76</td><td>0.0132</td><td>46.40</td><td>-16.44</td><td>159.19</td></tr><tr><td>MGDA</td><td>70.46</td><td>91.75</td><td>0.0224</td><td>34.33</td><td>-26.02</td><td>157.19</td></tr><tr><td>w/ Recon</td><td>72.23</td><td>92.60</td><td>0.0122</td><td>26.93</td><td>1.37</td><td>196.32</td></tr><tr><td>w/ Recon*</td><td>70.83</td><td>92.14</td><td>0.0125</td><td>25.69</td><td>1.31</td><td>159.19</td></tr><tr><td>Graddrop</td><td>75.19</td><td>93.53</td><td>0.0168</td><td>46.35</td><td>-23.90</td><td>157.19</td></tr><tr><td>w/ Recon</td><td>75.60</td><td>93.72</td><td>0.0127</td><td>38.55</td><td>-8.71</td><td>196.32</td></tr><tr><td>w/ Recon*</td><td>76.49</td><td>93.82</td><td>0.0129</td><td>47.54</td><td>-16.81</td><td>159.19</td></tr><tr><td>PCGrad</td><td>75.64</td><td>93.54</td><td>0.02</td><td>43.53</td><td>-23.60</td><td>157.19</td></tr><tr><td>w/ Recon</td><td>75.89</td><td>93.71</td><td>0.0129</td><td>40.05</td><td>-10.35</td><td>196.32</td></tr><tr><td>w/ Recon*</td><td>76.24</td><td>93.69</td><td>0.0128</td><td>45.24</td><td>-14.66</td><td>159.19</td></tr><tr><td>CAGrad</td><td>75.26</td><td>93.50</td><td>0.0176</td><td>44.23</td><td>-23.40</td><td>157.19</td></tr><tr><td>w/ Recon</td><td>75.65</td><td>93.71</td><td>0.0125</td><td>36.23</td><td>-6.15</td><td>196.32</td></tr><tr><td>w/ Recon*</td><td>76.25</td><td>93.74</td><td>0.0123</td><td>40.05</td><td>-8.99</td><td>159.19</td></tr></table>

found on NYUv2 are identical to those found on CityScapes. On NYUv2, we compare the performance of using conflict layers found on NYUv2 (baselines w/ Recon) to that of using conflict layers found on CityScapes (i.e., baselines w/ Recon<sup>∗</sup>), as shown in Table 15. (3) For MTAN (SegNet with attention), we find that 68% layers (17 out of 25) found on CityScapes are identical to those found on NYUv2. On CityScapes, we compare the performance of using conflict layers found on CityScapes (baselines w/ Recon) to that of using conflict layers found on NYUv2 (i.e., baselines w/ Recon<sup>∗</sup>), as shown in Table 16. The results show that the conflict layers found on one dataset can be used to modify the network to be directly used on another dataset to consistently improve the performance of various baselines, while searching for the conflict layers again on the new dataset may lead to better performance.

![](images/6d3eda882e8e92647f0b29cf3542250590d11f5919112a1cd38e2fee30a067c7.jpg)

<details>
<summary>line</summary>

| Number of tasks | Recon (s/iter) | Joint-train (s/iter) | CAGrad (s/iter) | Graddrop (s/iter) | PCGrad (s/iter) |
| --- | --- | --- | --- | --- | --- |
| 2 | ~0.05 | ~0.03 | ~0.05 | ~0.05 | ~0.06 |
| 12 | ~0.20 | ~0.03 | ~0.14 | ~0.14 | ~0.28 |
| 24 | ~0.29 | ~0.03 | ~0.27 | ~0.30 | ~0.75 |
| 30 | ~0.37 | ~0.04 | ~0.34 | ~0.38 | ~1.22 |
| 40 | ~0.51 | ~0.04 | ~0.47 | ~0.51 | ~2.23 |
</details>

Figure 11: Comparison of running time (one iteration, excludes data fetching) on CelebA dataset.

Analysis of running time. We evaluate how Recon scales with the number of tasks on CelebA dataset, by comparing the running time of one iteration used by Recon in computing gradient conflict scores (the most time-consuming part of Recon) to that of the baselines. The results in Fig. 11 show that Recon is as fast as other gradient manipulation methods such as CAGrad (Liu et al., 2021a) and Graddrop (Chen et al., 2020), but much slower than joint-training especially when the number of tasks is large, which is natural since Recon needs to compute pariwise cosine similarity of task gradients. However, since Recon only needs to search for the conflict layers once for a given network architecture, as discussed above, the running time is not a problem.