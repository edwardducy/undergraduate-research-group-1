# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning

Wooseong Jeong KAIST

stk14570@kaist.ac.kr

Kuk-Jin Yoon KAIST

kjyoon@kaist.ac.kr

## Abstract

Multi-Task Learning (MTL) enables multiple tasks to be learned within a shared network, but differences in objectives across tasks can cause negative transfer, where the learning of one task degrades another task’s performance. While pre-trained transformers significantly improve MTL performance, their fixed network capacity and rigid structure limit adaptability. Previous dynamic network archi tectures attempt to address this but are inefficient as they directly convert shared parameters into task-specific ones. We propose Dynamic Token Modulation and Expansion (DTME-MTL), aframework applicable to any transformerbased MTL architecture. DTME-MTL enhances adaptability and reduces overfitting by identifying gradient conflicts in token space and applying adaptive solutions based on conflict type. Unlike prior methods that mitigate negative transfer by duplicating network parameters, DTME-MTL operates entirely in token space, enabling efficient adaptation without excessive parameter growth. Extensive experiments demonstrate that DTME-MTL consistently improves multi-task performance with minimal computational overhead, offering a scalable and effective solutionfor enhancing transformer-based MTL models.

## 1. Introduction

Multi-Task Learning (MTL) enables multiple tasks to be learned simultaneously within a shared network, improving generalization, efficiency, and convergence speed compared to training separate models [6]. However, conflicting objectives among tasks can lead to negative transfer, where learning one task degrades the performance of another [10]. The key challenge lies in designing architectures that effectively balance shared and task-specific representations to mitigate negative transfer.

Transformer-based MTL architectures [63, 64, 67] leverage the strong generalization capabilities of large-scale pretrained networks such as Vision Transformers (ViTs) [14]. By utilizing pre-trained transformers trained on large opensource datasets, these architectures demonstrate improved generalization compared to conventional CNN-based MTL methods [11, 16, 41, 53, 57, 61, 71, 72]. However, they typically rely on predefined modules, such as Task Prompter mechanisms [63, 64, 67] and Mixture of Experts (MoE) [9, 18, 44, 48, 70], to separate shared and task-specific components. These rigid structures struggle to adapt to dynamic task relationships, leading to inefficient information sharing and suboptimal performance. The degree of task specialization required varies across different network depths [15]: high-level tasks such as semantic segmentation demand greater capacity in deeper layers, while low-level tasks like surface normal estimation rely more on shallower layers. Ideally, MTL architectures should dynamically adjust the allocation of shared and task-specific representations to accommodate these variations. However, existing transformer-based MTL frameworks are inherently constrained by their fixed network capacity, limiting their ability to adapt to evolving task dependencies and effectively mitigate negative transfer.

A straightforward approach to addressing these limitations is to increase the size of the transformer backbone. While this expands network capacity, it has a major drawback: it prevents the use of open-source pre-trained networks, which provide strong initialization and generalization capabilities across multiple tasks. Training a larger network from scratch requires massive computational resources and large-scale datasets, making this approach impractical for many applications. Instead, an effective MTL framework should refine existing architectures to retain the advantages of pre-trained transformers while improving adaptability to task-specific needs.

To achieve this, efficient adaptation methods for pretrained transformer-based MTL architectures are needed. Unlike approaches that build MTL frameworks from scratch or rely on task optimization within a fixed network capacity, we focus on adapting and enhancing predefined architectures while preserving their core design. This allows existing MTL models to be improved efficiently while dynamically adjusting task-specific representations. Despite its potential, the challenge of how to adaptively expand existing multi-task networks remains an underexplored problem.

One possible approach for adapting models during finetuning is the use of multi-task optimization techniques [13, 23, 33, 35–38, 45, 49, 50, 68], which mitigate negative transfer by adjusting task loss weights or modifying gradients. While these methods help balance task performance, they remain limited by a fixed network design and cannot expand model capacity.

A more direct approach involves dynamic network architectures, such as Recon [22], which directly expand network capacity to mitigate negative transfer. Recon measures conflicting gradients [68] in each layer—where gradients from different tasks point in opposing directions—and transforms the most conflicting layers into task-specific ones. While this increases flexibility by expanding the capacity of predefined architectures, directly converting shared parameters into task-specific ones in transformers leads to parameter inefficiency, excessive computational overhead, and a higher risk of overfitting. Consequently, its scalability to large transformer-based architectures is limited.

To address these challenges, we propose Dynamic Token Modulation and Expansion (DTME-MTL), a novel framework designed to improve pre-trained transformerbased MTL architectures. Unlike previous methods that directly manipulate network parameters, our approach mitigates negative transfer by restructuring the token space of multi-task networks. We treat transformer tokens as learnable parameters and analyze their structure using singular value decomposition (SVD) to identify gradient conflicts in token space. These conflicts are categorized into two types: range space conflicts, which are addressed through modulation via affine transformation of existing tokens, and null space conflicts, which are resolved by introducing new taskspecific tokens through expansion.

In our experiments, we demonstrate that DTME-MTL effectively enhances multi-task performance with minimal parameter overhead. Additionally, our results highlight that resolving task conflicts in the token space improves adaptability while mitigating overfitting.

Our main contributions are summarized as follows:

• We propose DTME-MTL which dynamically modulates and expands token spaces to mitigate negative transfer in transformer-based multi-task architectures.  
• We introduce a structured approach to resolving gradient conflicts in token space by categorizing them into range and null space conflicts, demonstrating how this improves multi-task performance.  
• DTME-MTL is an off-the-shelf solution that seamlessly integrates with existing state-of-the-art transformer-based

MTL architectures, enhancing performance with minimal computational overhead.

## 2. Related Works

Multi-Task Learning in Vision Transformers. Originally designed for NLP tasks, transformers have outperformed existing CNN models in various computer vision tasks. Attempts have been made to incorporate Vision Transformer [14, 39, 58–60, 65] in MTL. MTFormer [62] employs a shared transformer encoder and decoder with a cross-task attention mechanism. MulT [1] utilizes a shared attention mechanism to model task dependencies based on the Swin transformer. InvPT [66] focuses on global spatial position and multi-task context for dense prediction tasks through multi-scale feature aggregation. Mixture of Experts (MoE), inspired by the NLP domain, divides the model into predefined expert groups, adaptively shared or devoted to specific tasks during the learning phase [9, 18, 27, 44, 48, 70]. Task prompter [63, 64, 67] uses task-specific tokens to encapsulate task-specific information and employs cross-task interactions to enhance multi-task performance.

Dynamic Network Architectures for MTL. Dynamic networks adapt their structure during training to improve efficiency and task specialization. Several methods have explored dynamic architectures for MTL. Channel-wise dynamic allocation [2] assigns different convolutional channels to different tasks, but this method is not directly applicable to transformer-based architectures. Neural Architecture Search (NAS) for MTL [3, 5, 21, 24, 34, 47, 52] explores optimal network configurations but is computationally expensive and incompatible with large pre-trained backbone models such as ViTs [14]. Recon [22] transforms shared parameters directly into task-specific ones to handle conflicting gradients. Unlike most dynamic network architectures, our approach focuses on a dynamic system that can be directly applied to transformer-based multi-task architectures, leveraging pre-trained backbones while maintaining a reasonable computational cost.

Multi-Task Optimization. Optimizing the MTL aims to address negative transfer by adjusting the relative weighting of task losses or directly manipulating gradients. Taskdependent uncertainty [33] is utilized to weigh the loss of multiple tasks. Liu et al. [38] considers the rate of loss descent, while [23] prioritizes tasks based on difficulty. Recently, Liu et al. [36] proposed updating task weights based on the loss history. In contrast, approaches like [13, 35, 37, 45, 49, 50, 68] directly modify task gradients to achieve the desired balance. PCGrad [68] analyzes negative transfer by identifying conflicting gradients in the shared parameters of the network. Jiang et al. [31] suggests a positive link between negative transfer and conflicting gradients in auxiliary task learning. However, the conventional view in MTL considers conflicting gradients a key factor contributing to negative transfer in joint multi-task learning optimization [13, 29, 30, 35, 37, 45, 49, 50, 68], where tasks are learned together rather than serving as auxiliary tasks. Therefore, we adopt a similar perspective. Normalized gradients are employed to prevent spillover between tasks [7], whereas Chen et al. [8] introduce stochasticity to the network’s parameters based on the consistency in the sign of gradients. RotoGrad [28] rotates the feature space of the network to narrow the gap between tasks.

![](images/9eaf3c0c5751d0bf39ce54c2e97857e62e6a9d3c58ff4f560afa87150f7fb628.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  Input["L_i"] --> Sum(( ))
  Lj["L_j"] --> Sum
  Sum --> A["A"]
  A --> Wq["W_q"]
  A --> Wk["W_k"]
  A --> Wv["W_v"]
  Wq --> T_s1["T_{s,1}"]
  Wk --> T_s2["T_{s,2}"]
  Wv --> T_s3["T_{s,3}"]
  T_s1 --> T_sN["T_{s,N}"]
  T_s2 --> T_sN
  T_s3 --> T_sN
  T_sN --> TaskSpecificGradient["Task-specific gradients\nin shared token space"]
  subgraph RangeSpace["Range space\nof T_s \mathcal{R}(T̃_s^d)"]
    direction TB
    G1["g_i = \nabla_{T_s} \mathcal{L}_i"]
    Gj["g_j = \nabla_{T_s} \mathcal{L}_j"]
    Output["Output: R(T̃_s^d)"]
  end
```
</details>

(a) SVD in shared token space

![](images/9dbffc664bcc85b6790198357801d2869a5bb1b1d3fbe1f3c1fad1ca6eb40b55.jpg)

<details>
<summary>flowchart</summary>

This diagram illustrates a system architecture for detecting conflicts in range space, showing data flow from input sources through processing blocks (A, Wq, Wk, Wv), and finally to a target space with associated graph representations.
</details>

(b) Token modulation by detecting range space gradient conflicts

![](images/3a7c108051989d3a255e32abd127d649416ca7be5d22a432a7b84a30a2424ee0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  subgraph Process
  A["A"] --> B{"Decision"}
  C["Wq"] --> A
  D["Wk"] --> A
  E["Wv"] --> A
  F["Ts"] --> A
  G["T_i"] --> H["T_j"]
  I["Ts,1"] --> J["Ts,2"]
  K["Ts,3"] --> L["Ts,N"]
  M["..."] --> N["Ts,N"]
  end

  %% Details
  B --> C
  B --> D
  B --> E
  C --> F
  D --> G
  E --> H
  F --> I
  G --> J
  H --> K
  I --> L
  J --> M
  K --> M
  L --> M
  M --> N

  %% Details of Conflicts
  G -.->|"Detecting conflicts g_{N,i} \cdot g_{N,j} \le 0 in null space N(\tilde{T}_s^d)""| R["R(\tilde{T}_s^d)"]
  R --> S["g_i"]
  R --> T["g_{N,i}"]
  R --> U["g_{N,j}"]
  STU["S & T & U"] --> V["g_j"]
```
</details>

(c) Token expansion by detecting null space gradient conflicts  
Figure 1. Framework overview of the proposed DTME-MTL. (a) At each network layer, we compute the input token’s range space $\mathcal { R } ( \widetilde { T } _ { s } ^ { d } )$ and their task-specific gradients, determining principal vectors from the uncentered covariance of $\mathcal { T } _ { s }$ . (b) In cases where task-specific gradients conflict in the range space of $\widetilde { \mathcal { T } } _ { s } ^ { d } \left( \mathbf { e . g . } \ g _ { \mathcal { R } , i } \cdot \ g _ { \mathcal { R } , j } \leq 0 \right)$ , modulation is applied to $\mathcal { T } _ { s }$ by introducing $\mathcal { M } _ { i }$ and $\mathcal { M } _ { j }$ . (c) When task-specific gradients conflict within the null space of $\widetilde { \mathcal { T } } _ { s } ^ { d } \left( \mathbf { e . g . } g _ { \mathcal { N } , i } \cdot g _ { \mathcal { N } , j } \leq 0 \right)$ , task-specific tokens T<sub>i</sub> and $\mathcal { T } _ { j }$ are added.

## 3. Preliminaries

In multi-task learning, the network learns a set of tasks $\{ \tau _ { i } \} _ { i = 1 } ^ { K }$ jointly, where $\kappa$ is the number of tasks. Each task $\tau _ { i }$ has its own loss function $\mathcal { L } _ { i }$ . The network parameter Θ can be classified into $\boldsymbol { \Theta } = \{ \boldsymbol { \Theta } _ { s } , \boldsymbol { \Theta } _ { 1 } , . . . , \boldsymbol { \Theta } _ { K } \}$ where $\Theta _ { s }$ is shared parameter across all tasks and $\Theta _ { i }$ is task-specific parameters devoted to task $\tau _ { i }$ . Then, the objective function of multi-task learning is to minimize the weighted sum of all tasks’ losses: $\begin{array} { r } { \Theta ^ { * } \mathbf { \bar { \Psi } } = \arg \operatorname* { m i n } _ { \Theta } \sum _ { i = 1 } ^ { K } w _ { i } \mathcal { \bar { L } } _ { i } ( \Theta _ { s } , \Theta _ { i } ) } \end{array}$ where $w _ { i }$ represents the scale of the task-specific loss $\mathcal { L } _ { i } . ~ \mathrm { A }$ phenomenon called conflicting gradients [68], where the gradients of each objective point in different directions, has been identified as a main cause of negative transfer.

Definition 1 (Conflicting gradients). Define g<sub>i</sub> as the gradient of task $\tau _ { i }$ with respect to the shared parameters $\Theta _ { s }$ as $g _ { i } = \nabla _ { \Theta _ { s } } \mathcal { L } _ { i } ( \Theta _ { s } , \Theta _ { i } )$ . Let $g _ { i }$ and $g _ { j }$ represent the gradients for a pair of tasks $\tau _ { i }$ and $\tau _ { j }$ where i ̸= j. $I f g _ { i } \cdot g _ { j } \leq 0 ,$ , these two gradients are termed conflicting gradients.

However, the role of conflicting gradients remains a topic of debate. While conventional MTL optimization studies [13, 29, 35, 37, 45, 49, 50, 68] consider conflicting gradients as a main cause of negative transfer, Jiang et al. [31] argue that they can serve as a form of regularization that improves generalization when present in network parameters. Our findings align with Jiang et al. [31] in that directly resolving conflicting gradients by converting shared parameters into task-specific ones [22] leads to overfitting when applied to transformers. In contrast, we propose a tokenbased network expansion approach that categorizes gradient conflicts within token space and adapts accordingly, mitigating negative transfer while maintaining generalization.

## 4. Method

In order to mitigate negative transfer by ensuring sufficient space for tasks, we adopt token-based network expansion. Initially, we define the token space as the output of each layer in the transformer block through singular value decomposition (SVD). Subsequently, we categorize conflicts in task-specific gradients into two types: conflicts in the range space of tokens and conflicts in the null space of tokens. Finally, based on the type of conflict, we introduce efficient token modulation and expansion techniques for transformer-based multi-task architectures.

![](images/ee8b71cc5cff15b2b969600bedb5029a84ca8c924390a83882573208540faff0.jpg)

<details>
<summary>flowchart</summary>

This diagram illustrates a matrix decomposition algorithm for eigenvalue integration using the SVD(τ_s^d) function, showing how the input Λ is divided into sub-diagonal elements to derive an output diag(Λ).
</details>

Figure 2. The process approximates the range and null spaces of $\tilde { \mathcal { T } } _ { s } ^ { d }$ based on the proportion of total variance, r. These eigenvalues are arranged in descending order, satisfying $\lambda _ { i } \geq \lambda _ { j } \mathrm { i f } i < j .$ . If r is greater than the sum up to $\lambda _ { m }$ and smaller than the sum up to $\lambda _ { m + 1 }$ , then we select the set $\{ \lambda _ { i } \} _ { i = 1 } ^ { m }$ as $\Lambda _ { \mathcal { R } }$ , and the remaining set $\{ \lambda _ { i } \} _ { i = m + 1 } ^ { p }$ as $\Lambda _ { \mathcal { N } }$

## 4.1. Defining Token Space using SVD

In this section, we create a vector space consisting of shared tokens in a transformer, aiming to classify the types of conflicting gradients. More specifically, we approximate the range space and null space of the uncentered covariance of the tokens before applying our methods.

Let’s consider a dataset $\{ \mathcal { X } _ { l } , \mathcal { N } _ { l } \} _ { l = 1 } ^ { n }$ , where $\mathcal { X } _ { l }$ represents the input, $\mathcal { \ V } _ { l }$ denotes the label, and n is the number of samples. Denote input shared token for a layer d as $\mathcal { T } _ { s } ^ { l , d } = [ \mathcal { T } _ { s , 1 } ^ { l , d } , \mathcal { T } _ { s , 2 } ^ { l , d } , . . . , \dot { \mathcal { T } } _ { s , N } ^ { l , d } ]$ where N is the total number of shared tokens in that layer. Every token $\mathcal { T } _ { s , k } ^ { l , d } \in \mathbb { R } ^ { p }$ represents the output of the transformer layer d − 1 for the corresponding input data $\mathcal { X } _ { l } .$ , where p is the hidden dimension of the token embedding. Let’s consider a total of $D$ transformer layers. Next, the uncentered covariance of the token in layer d (where $1 \leq d \leq D )$ is as follows:

$$
\widetilde {\mathcal {T}} _ {s} ^ {d} = \frac {1}{n} \sum_ {l = 1} ^ {n} (\mathcal {T} _ {s} ^ {l, d}) (\mathcal {T} _ {s} ^ {l, d}) ^ {T} \tag {1}
$$

$\widetilde { \mathcal { T } } _ { s } ^ { d }$ is a square matrix of dimensions $p \times p .$ To define the token space, we apply SVD to ${ \widetilde { \mathcal { T } } } _ { s } ^ { d } .$ . Following this, we can divide vector space formed by $\widetilde { \mathcal { T } } _ { s } ^ { d }$ into its range space $\mathcal { R } ( \widetilde { \mathcal { T } } _ { s } ^ { d } )$ and null space $\mathcal { N } ( \widetilde { \mathcal { T } } _ { s } ^ { d } )$ depending on the magnitude of eigenvalue Λ. The process is illustrated below:

$$
\widetilde {\mathcal {T}} _ {s} ^ {d} = \mathcal {U} \Lambda \mathcal {V} ^ {T}, \quad \Lambda = \left[ \begin{array}{c c} \Lambda_ {\mathcal {R}} & 0 \\ 0 & \Lambda_ {\mathcal {N}} \end{array} \right] \tag {2}
$$

where $\Lambda$ is a diagonal matrix. Each $\Lambda _ { \mathcal { R } }$ and $\Lambda _ { \mathcal { N } }$ represent submatrices of $\Lambda$ containing the eigenvalues of the range space and null space, respectively. Both U and V are square matrices, each with dimensions $p \times p$

From Eq. (2), we obtain a mathematical tool to define the range and null space of the covariance of the token, $\widetilde { \mathcal { T } } _ { s } ^ { d }$ To approximate the range space, we choose the eigenvalue $\Lambda _ { \mathcal { R } }$ along with their corresponding eigenvectors from $\mathcal { U } _ { \mathcal { R } }$

On the other hand, when approximating the null space, we should select the eigenvalues $\Lambda _ { \mathcal { N } }$ and their corresponding eigenvectors from $\mathcal { U } _ { \mathcal { N } }$ . Ideally, we should choose eigenvalues that are exactly zero to form the null space. However, in practice, Λ can not be precisely zero. Therefore, it’s essential to establish a criterion for selecting the eigenvalue to distinguish between these two spaces.

Instead of introducing a new manually designed rule for approximating each range and null space of $\widetilde { \mathcal { T } } _ { s } ^ { d }$ , we opt to directly employ the evaluation tool for the SVD [32] as criteria for determining the range and null space of tokens. In assessing the accuracy of the SVD approximation, the proportion of total variance, denoted as $r ,$ has been employed:

$$
r = \frac {\sum_ {\lambda \in d i a g (\Lambda_ {\mathcal {N}})} \lambda}{\sum_ {\lambda \in d i a g (\Lambda_ {\mathcal {R}})} \lambda} \tag {3}
$$

The diag function serves as an operator, returning a set containing the diagonal entries of the input matrix. In our approach, we employ Eq. (3) to directly divide the range and null space of $\widetilde { \mathcal { T } } _ { s } ^ { d }$ . As depicted in Fig. 2, the diagonal elements of the matrix $\Lambda _ { ; }$ , obtained through the SVD of $\tilde { \mathcal { T } } _ { s } ^ { d }$ are arranged in descending order based on their magnitudes. We can select the index of the eigenvalue m such that the sum of eigenvalues up to order m is smaller than $r ,$ and the sum up to $m + 1$ is larger than r. This index serves as a boundary to divide the range space and null space of $\tilde { \mathcal { T } } _ { s } ^ { d }$

## 4.2. Types of Gradient Conflicts

In Section 4.1, we create a p-dimensional vector space using the uncentered covariance of the shared token $\widetilde { \mathcal { T } } _ { s } ^ { d }$ , linked to the input data set $\{ \mathcal { X } \} _ { l = 1 } ^ { n }$ . This vector space is divided into the range and null space, with each space spanned by eigenvectors corresponding to singular values selected based on a specified ratio $^ r \cdot$ In the upcoming sections, we pinpoint the types of gradient conflict within the vector space we’ve constructed. We then address these conflicts adaptively by introducing token modulation and expansion techniques.

Using Eq. (2) and Eq. (3), we can partition the eigenvectors of the p-dimensional vector space into its range and null space. Now, let’s consider the shared tokens $\bar { \mathcal { T } } _ { s } ^ { l , d } =$ $[ \mathcal { T } _ { s , 1 } ^ { l , d } , \dots , \mathcal { T } _ { s , N } ^ { l , d } ]$ , omitting the explicit notation of $l ,$ d for simplicity. For example, we write $\mathcal { T } _ { s } ^ { l , d } \to \mathcal { T } _ { s } , \mathcal { T } _ { s , k } ^ { l , d } \to \mathcal { T } _ { s , k }$ and $\tilde { \mathcal { T } } _ { s } ^ { d }  \tilde { \mathcal { T } } _ { s }$ . We treat $\mathcal { T } _ { s }$ as network parameters, for which gradients can be computed during the backpropagation process. Then, for each loss $\mathcal { L } _ { i } .$ the task-specific gradient for $\mathcal { T } _ { s , k }$ is denoted as $g _ { i } ~ = ~ \nabla _ { T _ { s , k } } { \mathcal { L } } _ { i }$ . Consequently, we obtain task-specific gradients $\{ g _ { i } \} _ { i = : } ^ { \kappa }$ corresponding to a set of losses $\{ \mathcal { L } _ { i } \} _ { i = } ^ { \mathcal { K } }$ <sub>1</sub> for $\mathcal { T } _ { s }$ as shown in Fig. 1-(a).

Each task-specific gradient $g _ { i }$ can be decomposed into two components, $g _ { \mathcal { R } , i }$ and $g _ { \mathcal { N } , i }$ , through projection onto the range and null space of $\tilde { \mathcal { T } } _ { s } ^ { d }$ , respectively. This breakdown is expressed as follows:

$$
g _ {\mathcal {R}, i} = (\mathcal {U} _ {\mathcal {R}} \mathcal {U} _ {\mathcal {R}} ^ {T}) \nabla_ {\mathcal {T} _ {s, k}} \mathcal {L} _ {i} \quad g _ {\mathcal {N}, i} = (\mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T}) \nabla_ {\mathcal {T} _ {s, k}} \mathcal {L} _ {i} \tag {4}
$$

$\mathcal { U } _ { \mathcal { R } }$ and $\mathcal { U } _ { \mathcal { N } }$ are orthogonal matrices that consist of eigenvectors of the range space and null space, respectively, with each column representing one eigenvector. Then, the matrices $( { \mathcal { U } } _ { { \mathcal { R } } } { \mathcal { U } } _ { { \mathcal { R } } } ^ { T } )$ and $( \mathcal { U } _ { \mathcal { N } } \mathcal { U } _ { \mathcal { N } } ^ { \breve { T } } )$ function as projection operators onto the range and null spaces, respectively.

Building upon the concept of conflicting gradients outlined in Definition 1, we classify conflicts into two types based on the space in which they occur: range space conflicts and null space conflicts. Specifically, conflicts in the range space of tokens occur when $g _ { \mathcal { R } , i } \cdot g _ { \mathcal { R } , j } \leq 0$ for any pair of i and $j$ where $i \neq j$ . Likewise, conflicts in the null space of tokens emerge when $g _ { \mathcal { N } , i } \cdot g _ { \mathcal { N } , j } \le 0$

## 4.3. Token Modulation and Expansion

Building on the gradient conflict types defined in Sec. 4.2, we propose adaptive strategies to mitigate task interference. Specifically, if gradient conflicts occur in the range space, we apply an affine transformation to modulate tokens, while conflicts in the null space are addressed by introducing additional tokens to expand the feature space. This distinction is particularly relevant in the transfer learning setting, where a pretrained transformer backbone is used, and task interference must be handled during fine-tuning. According to [46], training from pretrained weights constrains the model within the same basin of the loss landscape, preserving a feature space similar to that of the pretrained network. This insight guides our separation of token space into range and null spaces: conflicts in the row space indicate that the network already has relevant interpretative capabilities and can be resolved through rotation or scaling, whereas conflicts in the null space suggest the need for additional features, requiring token expansion to enhance the model’s capacity.

Token Modulation. In situations where task-specific gradients conflict within the range space of $\widetilde { \mathcal { T } } _ { s } .$ such as $\operatorname { \it g } _ { \mathcal { R } , i } \cdot \mathrm {  ~ \Omega ~ }$ $g _ { \mathcal { R } , j } ~ \leq ~ 0 .$ , modulators $\mathcal { M } _ { i }$ and $\mathcal { M } _ { j }$ are added after the shared token $\mathcal { T } _ { s }$ as shown in Fig. 1-(b). The token modulator $\mathcal { M }$ is a straightforward affine transformation that modulates the shared token $\mathcal { T } _ { s }$ along the channel dimension. To elaborate, considering the embedding dimension of the transformer $p$ and the number of shared tokens is $N ,$ we can arrange $\mathcal { T } _ { s }$ in the form $[ \mathcal { T } _ { s , 1 } , \ldots , \mathcal { T } _ { s , N } ]$ . This arrangement turns $\mathcal { T } _ { s }$ into a $p \times N$ matrix. The modulator ${ \mathcal { M } } .$ which incorporates weight and bias $W , b \in \mathbb { R } ^ { p }$ , performs the transformation $W \odot T _ { s , i } + b ,$ where ⊙ denotes elementwise multiplication. When the gradient lies in the row space of $\tilde { \mathcal { T } } _ { s }$ , Proposition 1 demonstrates that applying token modulation can effectively resolve gradient conflicts, lowering the multi-task loss.

Algorithm 1: DTME-MTL

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Data: Task $\{\tau_i\}_{i=1}^{\mathcal{K}}$, Loss function $\{\mathcal{L}_i\}_{i=1}^{\mathcal{K}}$,
Dataset $\{\mathcal{X}_l, \mathcal{Y}_l\}_{l=1}^n$, Shared tokens
$\mathcal{T}_s^{l,d} = \{\mathcal{T}_{s,i}^{l,d}\}_{i=1}^N$, Depth of the Network $D$
for each layer of the network ($d \leftarrow 1$ to $D$) do
    Get tokens $\{\mathcal{T}_s^{l,d}\}_{l=1}^n$ for the layer $d$
    corresponding to input $\{\mathcal{X}_l\}_{l=1}^n$
    Calculate uncentered covariance.
    $\widetilde{\mathcal{T}}_s^d = \frac{1}{n} \sum_{l=1}^n (\mathcal{T}_s^{l,d})(\mathcal{T}_s^{l,d})^T$
    Singular value decomposition.
    $\mathcal{U}, \Lambda, \mathcal{V} = SVD(\widetilde{\mathcal{T}}_s^d)$
    Divide range and null space. $\mathcal{U} = [\mathcal{U}_{\mathcal{R}}, \mathcal{U}_{\mathcal{N}}]$
    Projection to range space.
    $\{g_{\mathcal{R},i}\}_{i=1}^{\mathcal{K}} = \{(\mathcal{U}_{\mathcal{R}}\mathcal{U}_{\mathcal{R}}^T) \nabla_{\mathcal{T}_{s,k}^{l,d}} \mathcal{L}_i\}_{i=1}^{\mathcal{K}}$
    Projection to null space.
    $\{g_{\mathcal{N},i}\}_{i=1}^{\mathcal{K}} = \{(\mathcal{U}_{\mathcal{N}}\mathcal{U}_{\mathcal{N}}^T) \nabla_{\mathcal{T}_{s,k}^{l,d}} \mathcal{L}_i\}_{i=1}^{\mathcal{K}}$
    if $g_{\mathcal{R},i} \cdot g_{\mathcal{R},j} \leq 0$ then
        Insert token modulators $\mathcal{M}_i$ and $\mathcal{M}_j$
    if $g_{\mathcal{N},i} \cdot g_{\mathcal{N},j} \leq 0$ then
        Insert task-specific tokens $\mathcal{T}_i$ and $\mathcal{T}_j$
</div>

Proposition 1. When the input token $\mathcal { T } _ { i n }$ for input sample $\mathcal { X } _ { l }$ spans the range space of $\cdot _ { \tilde { \mathcal { T } } _ { s } , }$ , optimizing the token modulators $\{ \mathcal { M } _ { i } \} _ { i = 1 } ^ { \mathcal { K } }$ reduces gradient conflicts in the row space $o f \tilde { \mathcal { T } } _ { s }$ and leads to a reduction in the multi-task loss.

Token Expansion. Similarly, in cases where task-specific gradients conflict within the null space of $\widetilde { \mathcal { T } } _ { s } .$ , such as $g _ { \mathcal { N } , i } \cdot g _ { \mathcal { N } , j } \leq 0$ , task-specific tokens $\mathcal { T } _ { i }$ and $\tau _ { j }$ are added alongside shared tokens $\mathcal { T } _ { s }$ as shown in Fig. 1-(c). The task-specific tokens $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { \kappa }$ are concatenated with shared tokens before entering the transformer block. Consequently, each task-specific token acquires task-specific information within that layer. Specifically, in a standard transformer block, self-attention is performed for each pair of tokens in the form of $[ \mathcal { T } _ { s , 1 } , . . . , \mathcal { T } _ { s , N } ] \times [ \mathcal { T } _ { s , 1 } , . . . , \mathcal { T } _ { s , N } ]$ With token expansion, attention is extended to include $[ \mathcal { T } _ { s , 1 } , \ldots , \mathcal { T } _ { s , N } ] \times [ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { K } ]$ on the output. Proposition 2 explains how expanding the token space to address gradient conflicts in the null space of $\tilde { \mathcal { T } } _ { s }$ leads to a reduction in multi-task loss when the gradient lies in this null space. All proofs are provided in Supple E.

Table 1. We conduct an ablation study on dynamic token modulation and expansion, evaluating the multi-task performance of our method on NYUD-v2 and PASCAL-Context. The results of TE, TM, and their combination, TE+TM are presented. We employ a shared encode and multiple decoders, using ViT-T as the backbone network.

<table><tr><td rowspan="2">Model</td><td colspan="4">NYUD-v2</td><td colspan="5">PASCAL-Context</td></tr><tr><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td></tr><tr><td>Baseline (ST)</td><td>39.35</td><td>0.6611</td><td>22.14</td><td>59.68</td><td>67.96</td><td>58.90</td><td>83.76</td><td>15.65</td><td>47.70</td></tr><tr><td>Baseline (MT)</td><td>34.13</td><td>0.6732</td><td>22.51</td><td>55.30</td><td>54.47</td><td>51.48</td><td>82.04</td><td>16.22</td><td>41.28</td></tr><tr><td>TM</td><td>37.85</td><td>0.6490</td><td>21.75</td><td>56.92</td><td>64.28</td><td>55.10</td><td>83.02</td><td>15.40</td><td>45.80</td></tr><tr><td>TE</td><td>37.25</td><td>0.6553</td><td>21.87</td><td>57.00</td><td>60.51</td><td>54.00</td><td>82.85</td><td>15.55</td><td>44.98</td></tr><tr><td>TM+TE</td><td>38.27</td><td>0.6370</td><td>21.64</td><td>57.90</td><td>66.18</td><td>56.29</td><td>83.41</td><td>15.26</td><td>47.00</td></tr><tr><td>Gain (vs. MT)</td><td> $\triangle 4.14$ </td><td> $\triangle 0.0362$ </td><td> $\triangle 0.87$ </td><td> $\triangle 2.60$ </td><td> $\triangle 11.71$ </td><td> $\triangle 4.81$ </td><td> $\triangle 1.37$ </td><td> $\triangle 0.96$ </td><td> $\triangle 5.72$ </td></tr><tr><td> $\triangle m \uparrow$ </td><td colspan="4">0.044</td><td colspan="5">-1.289</td></tr><tr><td>#Param ↑ (%)</td><td colspan="4">0.24</td><td colspan="5">0.30</td></tr></table>

![](images/89e42a7bab168643a28ad808935b7c0c0e4db9e4a2aad894c08ffa08fd3219dc.jpg)  
Figure 3. Task performance varies based on when we expand the network. To determine the optimal timing, we assess expansions at the beginning of training and at the end of each quarter iteration, monitoring the corresponding changes in performance.

Proposition 2. When the input token $\mathcal { T } _ { i n }$ for input sample $\mathcal { X } _ { l }$ spans the null space of $\tilde { \mathcal { T } } _ { s } ,$ token expansion using $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { \mathcal { K } }$ alleviates the increase in multi-task loss caused by gradient conflicts in the null space of $\cdot \tilde { \tau } _ { s } .$

The complete procedure for DTME-MTL is outlined in Alg.1. Handling gradient conflicts in token space improves adaptability and reduces overfitting while being more efficient than addressing conflicts at the parameter level.

## 5. Experiments

## 5.1. Experimental Settings

Datasets and Evaluation. Our method is evaluated on multi-task datasets: NYUD-v2 [51], PASCAL-Context [43] and Taskonomy [69]. Each of them with 4, 5, 11 tasks. To evaluate the performance of tasks, we employed widely used metrics. To evaluate the multi-task performance, we utilize the metric proposed by Maninis et al. [42]. It measures the per-task performance $M _ { m , i }$ by averaging it with respect to the single-task baseline $M _ { b , i } .$ , as shown in $\triangle _ { m } =$ $( 1 / \mathcal { K } ) \sum _ { i = 1 } ^ { \mathcal { K } } ( - 1 ) ^ { l _ { i } } ( M _ { m , i } - M _ { b , i } ) / M _ { b , i \cdot } l _ { i } = 1$ if a lower value of the measure $M _ { i }$ indicates better performance for task i, and 0 otherwise.

Baselines and Model Variants. For a comprehensive analysis of the proposed DTME-MTL framework, we adopt a typical experimental setup for MTL in our experiments. In Tab. 1, ‘Baseline (MT)’ refers to a simple multi-task architecture consisting of a shared transformer backbone and basic task-specific decoders. Each decoder comprises one $3 \times 3 \mathrm { C o n v  – B N – R e L U }$ block. ‘Baseline (ST)’ has the same structure as ‘Baseline (MT)’ but is trained with only a single task. We assess the proposed DTME-MTL framework by expanding the network from ‘Baseline (MT)’ and measure the performance gains achieved by the proposed methods. ‘TM’ (Token Modulation) signifies the addition of the proposed token modulator to ‘baseline (MT)’, while ‘TE’ (Token Expansion) indicates the incorporation of task-specific tokens onto ‘Baseline (MT)’. Finally, ‘TM+TE’ combines both proposed methods. To show how effectively our approach reduces negative transfer, we also compare it with previous multi-task optimization, though our methods can be used alongside them. We include simple gradient descent (GD), GradDrop [8], MGDA [49], PCGrad [68], CAGrad [35], IMTL [37], Nash-MTL [45], and Aligned-MTL [50], as well as loss balancing methods such as UW [33], DWA [38], and FAMO [36]. We also compare our results with dynamic network architecture such as Recon [22]. Further experimental details are summarized in Supple B.

## 5.2. Experimental Results

Effectiveness of Token Modulation and Expansion. We assess the effectiveness of the proposed methods on the NYUD-v2 and PASCAL-Context datasets, with results detailed in Tab. 1. In the last three rows of the table, we depict the performance gains compared to the two baselines and the increased number of parameters in “#P aram ↑ (%)”. Compared to the Baseline (MT), our methods demonstrate significant performance improvements across all tasks in both datasets. Particularly noteworthy is the substantial increase in multi-task performance achieved with just a 0.2% to 0.3% increase in the total network parameters. Additionally, our approach exhibits comparable performance to Baseline (ST) in a multi-task scenario. This implies that reducing negative transfer is achievable by simply integrating token modulators and task tokens, without complex modules.

Table 2. Performance comparison based on the degree of conflicts in reversed order (Reversed) and randomly selected layers (Random).

<table><tr><td rowspan="2">Model</td><td colspan="4">NYUD-v2</td><td colspan="5">PASCAL-Context</td></tr><tr><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td></tr><tr><td>TM+TE</td><td>38.27</td><td>0.6370</td><td>21.64</td><td>57.90</td><td>66.18</td><td>56.29</td><td>83.21</td><td>15.26</td><td>47.00</td></tr><tr><td>TM+TE (Random)</td><td>36.88</td><td>0.6567</td><td>22.27</td><td>56.30</td><td>62.12</td><td>54.43</td><td>82.95</td><td>15.55</td><td>45.80</td></tr><tr><td>TM+TE (Reverse)</td><td>34.71</td><td>0.6898</td><td>22.59</td><td>55.80</td><td>57.84</td><td>52.82</td><td>82.75</td><td>15.74</td><td>43.20</td></tr></table>

![](images/b6a07a21f37b76575d2ec3c4a8d4170e53cd6564b9216f32b810a67d662e76a1.jpg)

<details>
<summary>bar</summary>

| Cosine Similarity \((cos\phi ij)\) | Baseline (%) | TM (%) | TE (%) | TM+TE (%) |
| --- | --- | --- | --- | --- |
| (1.00, 0.40) | — | — | — | — |
| (0.40, 0.20) | ~0.065 | ~0.065 | ~0.09 | ~0.07 |
| (0.20, 0.00) | ~0.46 | ~0.51 | ~0.505 | ~0.53 |
| (0.00, -0.20) | ~0.42 | ~0.395 | ~0.365 | ~0.37 |
| (-0.20, -0.40) | ~0.055 | ~0.025 | ~0.035 | ~0.03 |
| (-0.40, -1.00) | — | — | — | — |
</details>

(a) NYUD-v2

![](images/d7d8d309c9a31952f458880bf2e4d1d17d0530d56e921f187520c864ef4d1d7e.jpg)

<details>
<summary>bar</summary>

| Cosine Similarity \((cos\phi ij)\) | Baseline (%) | TM (%) | TE (%) | TM+TE (%) |
| --- | --- | --- | --- | --- |
| (1.00, 0.40) | ~0.02 | ~0.02 | ~0.025 | ~0.02 |
| (0.40, 0.20) | ~0.235 | ~0.238 | ~0.255 | ~0.25 |
| (0.20, 0.00) | ~0.315 | ~0.318 | ~0.305 | ~0.315 |
| (0.00, -0.20) | ~0.295 | ~0.285 | ~0.28 | ~0.278 |
| (-0.20, -0.40) | ~0.1 | ~0.11 | ~0.11 | ~0.105 |
| (-0.40, -1.00) | ~0.04 | ~0.038 | ~0.038 | ~0.038 |
</details>

(b) PASCAL-Context  
Figure 4. We evaluate the distribution of gradient conflicts by measuring the cosine similarity between task-specific gradients across all shared parameters throughout the optimization process. This is represented as cosϕ<sub>ij</sub> in (a) for NYUD-v2 and in (b) for PASCAL-Context.

Analysis of the Timing of Network Expansion. In Fig. 3, we analyze the performance of each task according to the timing of network expansion using the proposed DTME MTL. Specifically, the timing for expansion refers to the point at which token modulation and expansion are performed based on calculations of the token space using Singular Value Decomposition and measurement of gradient conflicts. The figure illustrates the performance results when network expansion is conducted at the beginning of training $( 0 ^ { t h } )$ and after each quarter of the entire training process (i<sup>th</sup> 25% Iter). To ensure fair comparisons, we trained the network using the same number of iterations after the expansion. The results indicate that the optimal timing for expansion may vary across tasks. However, overall, early-stage expansion during network training tends to yield better performance. This aligns with the fact that our approach builds on pre-trained backbone networks.

Analysis of Gradient Conflicts in Parameters. We focus on resolving gradient conflicts in token space. While our primary method operates in token-level representations, we also monitor gradient conflicts in parameter space to better understand training dynamics. In Fig. 4, we visualize the distribution of angles between task-specific gradients of network parameters, categorizing them into different ranges and tracking their frequency over the course of training. When applied to the baseline model, both Token Modulation (TM) and Token Expansion (TE) reduce gradient conflicts in parameter space to some extent. However, as shown in Tab. 7, where our methods are compared with Recon [22], we observe important differences. Recon explicitly suppresses conflicts by modifying network parameters whenever the cosine similarity between task gradients becomes negative. Although this reduces gradient conflicts in parameter space, it often results in severe overfitting and degraded multi-task performance. These findings suggest that conflicts in parameter space are not always reliable indicators of negative transfer. Instead, resolving conflicts in token space offers an alternative strategy that avoids overfitting while still mitigating interference. Additional analysis of token-level conflicts is provided in Appendix D.

Computational Cost of DTME-MTL. In Tab. 6, we report the time consumption for each process of DTME-MTL on PASCAL-Context using a single NVIDIA RTX A6000. We measure the time required for calculating the token space with SVD and for computing gradient conflicts in the token space of the transformer. The time required for each process increases with the size of the transformer. However, the proposed methods are computationally efficient, requiring approximately 1 hour with ViT-L. Considering that typical multi-task architectures require at least one day of training on a single GPU, the computational cost of DTME-MTL is relatively low. Proposed DTME-MTL increases inference time of each task about 13.4% with ViT-B.

Comparing Performance based on Layer Selection Criteria. In Tab. 2, we applied TM+TE to layers with the highest gradient conflicts between tasks. Results are also shown for randomly chosen layers (Random) or layers with the lowest gradient conflicts (Reverse). The network expansion system, using conflict detection, outperforms random selection across all tasks. Particularly, applying TM+TE to layers with severe conflict levels consistently outperforms its application in layers with lower conflict levels, validating the effectiveness of the strategy.

Table 3. Comparison of multi-task optimization methods on Taskonomy across 11 tasks. Non-converged results are indicated with a dash.

<table><tr><td>Task Metric</td><td>DE L1 Dist. ↓</td><td>DZ L1 Dist. ↓</td><td>EO L1 Dist. ↓</td><td>ET L1 Dist. ↓</td><td>Key2D L1 Dist. ↓</td><td>Key3D L1 Dist. ↓</td><td>N L1 Dist.</td><td>PC RMSE ↓</td><td>R L1 Dist. ↓</td><td>S2D L1 Dist. ↓</td><td>S25D L1 Dist. ↓</td><td> $\triangle_m \uparrow (\%)$ </td></tr><tr><td>ST</td><td>0.0199</td><td>0.0195</td><td>0.1085</td><td>0.1714</td><td>0.1633</td><td>0.0872</td><td>0.2715</td><td>0.7586</td><td>0.1503</td><td>0.1742</td><td>0.1504</td><td>0.00</td></tr><tr><td>GD</td><td>0.0187</td><td>0.0188</td><td>0.1301</td><td>0.1757</td><td>0.1733</td><td>0.0942</td><td>0.3076</td><td>0.7991</td><td>0.1826</td><td>0.1902</td><td>0.1652</td><td>- 7.83</td></tr><tr><td>GradDrop [8]</td><td>0.0315</td><td>0.0242</td><td>0.1390</td><td>0.1776</td><td>0.1778</td><td>0.0976</td><td>0.4564</td><td>0.8644</td><td>0.2088</td><td>0.1995</td><td>0.1752</td><td>- 26.11</td></tr><tr><td>MGDA [49]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UW [33]</td><td>0.0190</td><td>0.0190</td><td>0.1308</td><td>0.1758</td><td>0.1734</td><td>0.0945</td><td>0.3109</td><td>0.8009</td><td>0.1840</td><td>0.1906</td><td>0.1657</td><td>- 8.43</td></tr><tr><td>DWA [38]</td><td>0.0186</td><td>0.0187</td><td>0.1294</td><td>0.1759</td><td>0.1735</td><td>0.0938</td><td>0.2788</td><td>0.7943</td><td>0.1805</td><td>0.1902</td><td>0.1640</td><td>- 6.45</td></tr><tr><td>PCGrad [68]</td><td>0.0217</td><td>0.0192</td><td>0.1298</td><td>0.1775</td><td>0.1714</td><td>0.0939</td><td>0.2856</td><td>0.7985</td><td>0.1817</td><td>0.1927</td><td>0.1595</td><td>- 8.29</td></tr><tr><td>CAGrad [35]</td><td>0.0219</td><td>0.0203</td><td>0.1314</td><td>0.1800</td><td>0.1665</td><td>0.0932</td><td>0.3039</td><td>0.8121</td><td>0.1874</td><td>0.1953</td><td>0.1673</td><td>- 10.57</td></tr><tr><td>IMTL [37]</td><td>0.0210</td><td>0.0192</td><td>0.1282</td><td>0.1772</td><td>0.1719</td><td>0.0936</td><td>0.2468</td><td>0.7784</td><td>0.1734</td><td>0.1943</td><td>0.1647</td><td>- 6.17</td></tr><tr><td>Align-MTL [50]</td><td>0.0189</td><td>0.0193</td><td>0.1254</td><td>0.1728</td><td>0.1664</td><td>0.0914</td><td>0.3524</td><td>0.8640</td><td>0.1938</td><td>0.1889</td><td>0.1582</td><td>- 9.41</td></tr><tr><td>Nash-MTL [45]</td><td>0.0201</td><td>0.0184</td><td>0.1248</td><td>0.1764</td><td>0.1701</td><td>0.0921</td><td>0.2658</td><td>0.7793</td><td>0.1706</td><td>0.1914</td><td>0.1624</td><td>- 5.01</td></tr><tr><td>FAMO [36]</td><td>0.0188</td><td>0.0188</td><td>0.1300</td><td>0.1758</td><td>0.1733</td><td>0.0942</td><td>0.3058</td><td>0.7986</td><td>0.1826</td><td>0.1904</td><td>0.1654</td><td>- 7.87</td></tr><tr><td>DTME-MTL</td><td>0.0150</td><td>0.0154</td><td>0.1193</td><td>0.1733</td><td>0.1668</td><td>0.0891</td><td>0.2038</td><td>0.7373</td><td>0.1567</td><td>0.1773</td><td>0.1517</td><td>+ 4.67</td></tr></table>

Table 4. Adaptation of DTME-MTL to other state-of-the-art MTL methods on NYUD-v2.

<table><tr><td>Task Metric</td><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td></tr><tr><td>MTformer [62]</td><td>50.04</td><td>0.490</td><td>-</td><td>-</td></tr><tr><td>InvPT [66]</td><td>53.56</td><td>0.5183</td><td>18.81</td><td>78.10</td></tr><tr><td>+ DTME-MTL</td><td>54.38</td><td>0.5020</td><td>18.51</td><td>78.20</td></tr><tr><td>Taskprompter [67]</td><td>55.30</td><td>0.5152</td><td>18.47</td><td>78.20</td></tr><tr><td>+ DTME-MTL</td><td>56.36</td><td>0.5122</td><td>18.38</td><td>78.40</td></tr></table>

Table 5. Adaptation of DTME-MTL to other state-of-the-art MTL methods on PASCAL-Context.

<table><tr><td>TaskMetric</td><td>SemsegmIoU ↑</td><td>ParsingmIoU ↑</td><td>SaliencymaxF ↑</td><td>NormalmErr ↓</td><td>EdgeodsF ↑</td></tr><tr><td>MTformer [62]</td><td>73.51</td><td>64.26</td><td>67.24</td><td>-</td><td>-</td></tr><tr><td>InvPT [66]</td><td>79.03</td><td>67.61</td><td>84.81</td><td>14.15</td><td>73.00</td></tr><tr><td>+ DTME-MTL</td><td>81.91</td><td>71.13</td><td>84.96</td><td>13.73</td><td>73.80</td></tr><tr><td>Taskprompter [67]</td><td>80.89</td><td>68.89</td><td>84.83</td><td>13.72</td><td>73.50</td></tr><tr><td>+ DTME-MTL</td><td>81.01</td><td>69.08</td><td>84.75</td><td>13.65</td><td>73.60</td></tr></table>

Comparison with Multi-Task Optimization. In Tab. 3, we compare DTME-MTL with previous multi-task optimization approaches to demonstrate its effectiveness in reducing negative transfer between tasks on the Taskonomy benchmark using ViT-B. DTME-MTL achieves the best multitask performance, improving each task by an average of 4.67% with only a 0.118% increase in the number of parameters. Although DTME-MTL introduces additional parameters to address negative transfer, making direct comparisons with optimization methods less straightforward, it consistently improves multi-task performance. However, using more task-specific parameters does not always lead to better performance. As as shown in Tab. 7, Recon shows poor results with the vision transformer on NYUD-v2. This comparison highlights that previous multi-task expansion approaches, which naively duplicate network branches, are not only parameter-inefficient but also prone to overfitting due to the increased complexity of transformers.

Adapting to Multi-Task Architectures. In Tabs. 4 and 5, we compare DTME-MTL with leading multi-task architectures on the NYUD-v2 and PASCAL-Context datasets. We evaluate its multi-task performance against transformerbased approaches. Our method is compatible with any transformer-based multi-task architecture, enabling us to assess its effectiveness by integrating it into two leading models: InvPT and TaskPrompter. DTME-MTL seamlessly enhances these architectures, significantly boosting performance with only a minimal increase in parameters — just 0.048% for InvPT and 0.046% for TaskPrompter.

Table 6. Time consumption of each process in DTME-MTL across different backbone sizes, recorded in minutes.

<table><tr><td>Process (min)</td><td>ViT-T</td><td>ViT-S</td><td>ViT-B</td><td>ViT-L</td></tr><tr><td>Calculate Token Space (SVD)</td><td>3.61</td><td>3.74</td><td>11.54</td><td>11.96</td></tr><tr><td>Calculate Gradient Conflict</td><td>8.25</td><td>16.43</td><td>21.94</td><td>58.66</td></tr></table>

Table 7. Comparison with Recon on NYUD-v2

<table><tr><td>Method</td><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td><td>#Param ↑ (%)</td></tr><tr><td>Joint</td><td>34.13</td><td>0.673</td><td>22.51</td><td>56.38</td><td>0.0</td></tr><tr><td>Recon [22]</td><td>31.92</td><td>0.693</td><td>23.35</td><td>52.80</td><td>23.34</td></tr><tr><td>Ours</td><td>38.27</td><td>0.6370</td><td>21.64</td><td>57.90</td><td>0.24</td></tr></table>

## 6. Conclusion

We introduced Dynamic Token Modulation and Expansion for Multi-Task Learning, an efficient approach for improving transformer-based MTL architectures. By categorizing gradient conflicts into range space and null space, it adaptively applies token modulation and expansion to mitigate negative transfer and reduce overfitting. DTME-MTL seamlessly integrates with existing transformer-based MTL frameworks, requiring minimal additional parameters. By refining encoded token space, it provides a lightweight and scalable solution for enhancing multi-task performance.

## Acknowledgment

This research was supported by the Challengeable Future Defense Technology Research and Development Program through the Agency For Defense Development(ADD) funded by the Defense Acquisition Program Administration(DAPA) in 2025(No.915102201).

## References

[1] Deblina Bhattacharjee, Tong Zhang, Sabine Susstrunk, and¨ Mathieu Salzmann. Mult: an end-to-end multitask learning transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12031– 12041, 2022. 2  
[2] Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1385– 1394, 2019. 2  
[3] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resourceefficient branched multi-task networks. arXiv preprint arXiv:2008.10292, 2020. 2  
[4] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resourceefficient branched multi-task networks. arXiv preprint arXiv:2008.10292, 2020. 12  
[5] Ronghong Cai and Jianping Luo. Multi-task learning for multi-objective evolutionary neural architecture search. In 2021 IEEE Congress on Evolutionary Computation (CEC), pages 1680–1687. IEEE, 2021. 2  
[6] Rich Caruana. Multitask learning. Machine learning, 28: 41–75, 1997. 1  
[7] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018. 3  
[8] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020. 3, 6, 8  
[9] Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik G Learned-Miller, and Chuang Gan. Mod-squad: Designing mixtures of experts as modular multi-task learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11828–11837, 2023. 1, 2  
[10] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796, 2020. 1  
[11] Jifeng Dai, Kaiming He, and Jian Sun. Instance-aware semantic segmentation via multi-task network cascades. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3150–3158, 2016. 1, 12  
[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 12  
[13] Jean-Antoine Desid´ eri. Multiple-gradient descent algorithm´ (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012. 2, 3  
[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 2, 12  
[15] Kshitij Dwivedi and Gemma Roig. Representation similarity analysis for efficient task taxonomy & transfer learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12387–12396, 2019. 1  
[16] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE international conference on computer vision, pages 2650–2658, 2015. 1, 12  
[17] Mark Everingham and John Winn. The pascal visual object classes challenge 2012 (voc2012) development kit. Pattern Anal. Stat. Model. Comput. Learn., Tech. Rep, 2007:1–45, 2012. 12  
[18] Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang, et al. M<sup>3</sup>vit: Mixture-of-experts vision transformer for efficient multitask learning with model-accelerator co-design. Advances in Neural Information Processing Systems, 35:28441–28457, 2022. 1, 2  
[19] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734, 2017. 12  
[20] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3205–3214, 2019. 12  
[21] Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11543–11552, 2020. 2  
[22] SHI Guangyuan, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. Recon: Reducing conflicting gradients from the root for multi-task learning. In The Eleventh International Conference on Learning Representations, 2022. 2, 3, 6, 7, 8  
[23] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pages 270–287, 2018. 2  
[24] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International conference on machine learning, pages 3854–3863. PMLR, 2020. 2  
[25] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International Conference on Machine Learning, pages 3854–3863. PMLR, 2020. 12  
[26] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 15  
[27] Huimin Huang, Yawen Huang, Lanfen Lin, Ruofeng Tong, Yen-Wei Chen, Hao Zheng, Yuexiang Li, and Yefeng Zheng. Going beyond multi-task dense prediction with synergy embedding models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28181–28190, 2024. 2  
[28] Adrian Javaloy and Isabel Valera. Rotograd: Gradi-´ ent homogenization in multitask learning. arXiv preprint arXiv:2103.02631, 2021. 3  
[29] Wooseong Jeong and Kuk-Jin Yoon. Quantifying task priority for multi-task optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 363–372, 2024. 3  
[30] Wooseong Jeong and Kuk-Jin Yoon. Selective task group updates for multi-task optimization. arXiv preprint arXiv:2502.11986, 2025. 3  
[31] Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, and Mingsheng Long. Forkmerge: Mitigating negative transfer in auxiliary-task learning. Advances in Neural Information Processing Systems, 36, 2024. 2, 3  
[32] Ian T Jollife and Jorge Cadima. Principal component analysis: A review and recent developments. Philos. Trans. R. Soc. A Math. Phys. Eng. Sci, 374(2065):20150202, 2016. 4  
[33] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018. 2, 6, 8  
[34] Jason Liang, Elliot Meyerson, and Risto Miikkulainen. Evolutionary architecture search for deep multitask networks. In Proceedings of the genetic and evolutionary computation conference, pages 466–473, 2018. 2  
[35] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021. 2, 3, 6, 8  
[36] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36, 2024. 2, 6, 8  
[37] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. iclr, 2021. 2, 3, 6, 8  
[38] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019. 2, 6, 8  
[39] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 2  
[40] Yongxi Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, Tara Javidi, and Rogerio Feris. Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5334– 5343, 2017. 12  
[41] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1930–1939, 2018. 1, 12  
[42] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1851–1860, 2019. 6, 12  
[43] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014. 6  
[44] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems, 35:9564– 9576, 2022. 1, 2  
[45] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multitask learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022. 2, 3, 6, 8  
[46] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? Advances in neural information processing systems, 33:512–523, 2020. 5, 15  
[47] Ramakanth Pasunuru and Mohit Bansal. Continual and multi-task architecture search. arXiv preprint arXiv:1906.05226, 2019. 2  
[48] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel´ Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021. 1, 2  
[49] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018. 2, 3, 6, 8  
[50] Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20083–20093, 2023. 2, 3, 6, 8  
[51] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012. 6, 12  
[52] Gianluigi Silvestri. One-shot neural architecture search for deep multi-task learning in computer vision, 2020. 2  
[53] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 1, 12  
[54] Ayan Sinha, Zhao Chen, Vijay Badrinarayanan, and Andrew Rabinovich. Gradient adversarial training of neural networks. 2018. 12  
[55] Guolei Sun, Thomas Probst, Danda Pani Paudel, Nikola Popovic, Menelaos Kanakis, Jagruti Patel, Dengxin Dai, and´ Luc Van Gool. Task switching network for multi-task learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8291–8300, 2021. 12  
[56] Simon Vandenhende, Stamatios Georgoulis, Bert De Brabandere, and Luc Van Gool. Branched multi-task networks: deciding what layers to share. arXiv preprint arXiv:1904.02920, 2019. 12  
[57] Simon Vandenhende, Stamatios Georgoulis, and Luc Van Gool. Mti-net: Multi-scale task interaction networks for multi-task learning. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 527–543. Springer, 2020. 1, 12  
[58] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 568–578, 2021. 2  
[59] Wenxiao Wang, Lu Yao, Long Chen, Binbin Lin, Deng Cai, Xiaofei He, and Wei Liu. Crossformer: A versatile vision transformer hinging on cross-scale attention. arXiv preprint arXiv:2108.00154, 2021.  
[60] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 2  
[61] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 675–684, 2018. 1, 12  
[62] Xiaogang Xu, Hengshuang Zhao, Vibhav Vineet, Ser-Nam Lim, and Antonio Torralba. Mtformer: Multi-task learning via transformer and cross-task reasoning. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII, pages 304–321. Springer, 2022. 2, 8  
[63] Yangyang Xu, Xiangtai Li, Haobo Yuan, Yibo Yang, and Lefei Zhang. Multi-task learning with multi-query transformer for dense prediction. IEEE Transactions on Circuits and Systemsfor Video Technology, 2023. 1, 2  
[64] Yangyang Xu, Yibo Yang, and Lefei Zhang. Demt: Deformable mixer transformer for multi-task learning of dense prediction. In Proceedings of the AAAI conference on artificial intelligence, pages 3072–3080, 2023. 1, 2  
[65] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao. Focal self-attention for local-global interactions in vision transformers. arXiv preprint arXiv:2107.00641, 2021. 2  
[66] Hanrong Ye and Dan Xu. Inverted pyramid multi-task transformer for dense scene understanding. In Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII, pages 514– 530. Springer, 2022. 2, 8, 12  
[67] Hanrong Ye and Dan Xu. Taskprompter: Spatial-channel multi-task prompting for dense scene understanding. In The Eleventh International Conference on Learning Representations, 2022. 1, 2, 8, 12  
[68] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020. 2, 3, 6, 8  
[69] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3712–3722, 2018. 6  
[70] Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. Mixture of attention heads: Selecting attention heads per token. arXiv preprint arXiv:2210.05144, 2022. 1, 2  
[71] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceed ings, Part VI 13, pages 94–108. Springer, 2014. 1, 12  
[72] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4106–4115, 2019. 1, 12

# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning

Supplementary Material

## A. Additional Related Works

Multi-Task Architectures. Various multi-task architectures can be categorized based on how the parameters or features of the sharing network are distributed among tasks. The widely used shared trunk structure comprises a common encoder shared by multiple tasks and a dedicated decoder for each task [11, 41, 53, 71]. A tree-like architecture, with multiple division points for each task group, offers a more generalized structure [4, 25, 40, 56]. The cross-talk architecture employs separate symmetrical networks for each task, utilizing feature exchange between layers at the same depth for information sharing between tasks [20, 61]. The prediction distillation model [16, 57, 61, 72] incorporates cross-task interactions at the end of the shared encoder, while the task switching network [19, 42, 54, 55] changes network parameters depending on the task.

## B. Experimental Settings

## B.1. Datasets

We evaluate our method on three benchmarks: NYUD-v2, PASCAL-Context, and Taskonomy. NYUD-v2 contains 4 vision tasks: Our evaluation is based on depth estimation, semantic segmentation, surface normal prediction, and edge detection. PASCAL-Context contains 5 tasks: We evaluate semantic segmentation, human parts estimation, saliency estimation, surface normal prediction, and edge detection. We used 11 tasks for Taskonomy: We evaluate Depth Euclidean (DE), Depth Zbuffer (DZ), Edge Texture (ET), Keypoints 2D (Key2D), Keypoints 3D (Key3D), Normal (N), Principal Curvature (PC), Reshading (R), Segment Unsup 2d (S2D), and Segment Unsup 2.5D (S25D).

## B.2. Implementation Details

For experiments, we adopt ViT [14] pre-trained on ImageNet-22K [12] as the transformer encoder. The models are trained for 60,000 iterations on both NYUD [51] and PASCAL [17] datasets with batch size 6. We use Adam optimizer with learning rate $2 \times 1 0 ^ { - 5 }$ and $1 \times 1 0 ^ { - 6 }$ of a weight decay with a polynomial learning rate schedule. Following the previous works [66, 67], the cross-entropy loss is used for semantic segmentation, human parts estimation, and saliency, edge detection. Surface normal prediction and depth estimation use L1 loss.

## B.3. Design and Implementation Strategy

To improve efficiency, we perform SVD only once early in training to estimate the feature space for conflict analysis. Gradient conflicts are measured in a pairwise manner across tasks, and the average number of conflicts in each space is used to guide token expansion. Based on this, we statically allocate a small number of task-specific tokens (six in our setup) as learnable parameters, independently applied at each layer. These tokens are fixed during training and do not adapt dynamically. For NYUD-v2 and PASCAL-Context, we use the full training sets to compute gradient statistics, while for Taskonomy, covariance is estimated using 100 randomly sampled mini-batches. The assignment of Token Modulation (TM) and Token Expansion (TE) is determined by a manually chosen activation ratio, which we analyze in Fig. 6. Rather than activating all components uniformly, TM and TE are selectively applied to layers with the highest conflict levels, either individually or jointly, based on their effectiveness in reducing task interference.

## B.4. Evaluation

For semantic segmentation, we utilize mean Intersection over Union (mIoU). Surface normal prediction performance is measured by the mean angular distance between the predicted output and ground truth. Depth estimation is evaluated using Root Mean Squared Error (RMSE). For saliency estimation and human part segmentation, we employ mIoU. Edge detection is assessed using the optimal-dataset-scale F-measure (odsF). For Taskonomy, we adopt RMSE for principal curvature and L1 distance for the remaining tasks.

## C. Additional Experiments

Comparison with Multi-Task Optimization. In Tabs. 8 to 10, we further evaluate the proposed DTME-MTL against previous multi-task optimization approaches using different backbone sizes. Our method demonstrates significant improvements in multi-task performance with minimal increases in parameters. Specifically, DTME-MTL results in a parameter increase of 0.089% for ViT-L, 0.23% for ViT-S, and 0.46% for ViT-T.

Analysis on the Modulator Configuration. In Tab. 11, we show the performance difference based on the configuration of the token modulators. Specifically, we compared the outcomes obtained when employing affine transformation and batch normalization, which could be considered as the most common and straightforward approaches. Through experiments, we find that affine transformations consistently exhibit better performance across all tasks compared to batch normalization layers used as modulators for both datasets.

Table 8. Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-L. Non-converged results are indicated with a dash.

<table><tr><td>Task Metric</td><td>DE L1 Dist. ↓</td><td>DZ L1 Dist. ↓</td><td>EO L1 Dist. ↓</td><td>ET L1 Dist. ↓</td><td>Key2D L1 Dist. ↓</td><td>Key3D L1 Dist. ↓</td><td>N L1 Dist.</td><td>PC RMSE ↓</td><td>R L1 Dist. ↓</td><td>S2D L1 Dist. ↓</td><td>S25D L1 Dist. ↓</td><td> $\triangle_m \uparrow (\%)$ </td></tr><tr><td>ST</td><td>0.0141</td><td>0.0146</td><td>0.0992</td><td>0.1716</td><td>0.1631</td><td>0.0801</td><td>0.2133</td><td>0.7134</td><td>0.1342</td><td>0.1688</td><td>0.1419</td><td>0.00</td></tr><tr><td>GD</td><td>0.0153</td><td>0.0156</td><td>0.1196</td><td>0.1757</td><td>0.1729</td><td>0.0896</td><td>0.2215</td><td>0.7451</td><td>0.1576</td><td>0.1826</td><td>0.1537</td><td>-8.92</td></tr><tr><td>GradDrop</td><td>0.0170</td><td>0.0195</td><td>0.1235</td><td>0.1757</td><td>0.1753</td><td>0.0909</td><td>0.2818</td><td>0.7679</td><td>0.1663</td><td>0.1916</td><td>0.1543</td><td>-17.07</td></tr><tr><td>MGDA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UW</td><td>0.0152</td><td>0.0155</td><td>0.1195</td><td>0.1755</td><td>0.1728</td><td>0.0897</td><td>0.2356</td><td>0.7436</td><td>0.1569</td><td>0.1830</td><td>0.1538</td><td>-9.36</td></tr><tr><td>DWA</td><td>0.0153</td><td>0.0156</td><td>0.1197</td><td>0.1757</td><td>0.1730</td><td>0.0897</td><td>0.2214</td><td>0.7441</td><td>0.1576</td><td>0.1827</td><td>0.1537</td><td>-8.96</td></tr><tr><td>PCGrad</td><td>0.0152</td><td>0.0156</td><td>0.1192</td><td>0.1749</td><td>0.1699</td><td>0.0893</td><td>0.2310</td><td>0.7475</td><td>0.1577</td><td>0.1825</td><td>0.1480</td><td>-8.63</td></tr><tr><td>CAGrad</td><td>0.0155</td><td>0.0156</td><td>0.1175</td><td>0.1756</td><td>0.1649</td><td>0.0860</td><td>0.2421</td><td>0.7544</td><td>0.1591</td><td>0.1854</td><td>0.1554</td><td>-9.32</td></tr><tr><td>IMTL</td><td>0.0151</td><td>0.0156</td><td>0.1194</td><td>0.1755</td><td>0.1726</td><td>0.0895</td><td>0.2199</td><td>0.7432</td><td>0.1569</td><td>0.1824</td><td>0.1533</td><td>-8.57</td></tr><tr><td>Align-MTL</td><td>0.0150</td><td>0.0155</td><td>0.1136</td><td>0.1733</td><td>0.1633</td><td>0.0862</td><td>0.2512</td><td>0.8029</td><td>0.1643</td><td>0.1803</td><td>0.1445</td><td>-8.78</td></tr><tr><td>Nash-MTL</td><td>0.0151</td><td>0.0154</td><td>0.1138</td><td>0.1732</td><td>0.1644</td><td>0.0863</td><td>0.2507</td><td>0.7656</td><td>0.1544</td><td>0.1833</td><td>0.1452</td><td>-7.95</td></tr><tr><td>FAMO</td><td>0.0153</td><td>0.0157</td><td>0.1196</td><td>0.1757</td><td>0.1730</td><td>0.0897</td><td>0.2221</td><td>0.7444</td><td>0.1575</td><td>0.1830</td><td>0.1534</td><td>-8.99</td></tr><tr><td>DTME-MTL</td><td>0.0127</td><td>0.0130</td><td>0.1088</td><td>0.1731</td><td>0.1665</td><td>0.0852</td><td>0.1654</td><td>0.6890</td><td>0.1389</td><td>0.1661</td><td>0.1404</td><td>+2.41</td></tr></table>

Table 9. Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-S. Non-converged results are indicated with a dash.

<table><tr><td>Task Metric</td><td>DE L1 Dist. ↓</td><td>DZ L1 Dist. ↓</td><td>EO L1 Dist. ↓</td><td>ET L1 Dist. ↓</td><td>Key2D L1 Dist. ↓</td><td>Key3D L1 Dist. ↓</td><td>N L1 Dist.</td><td>PC RMSE ↓</td><td>R L1 Dist. ↓</td><td>S2D L1 Dist. ↓</td><td>S25D L1 Dist. ↓</td><td> $\triangle_m \uparrow (\%)$ </td></tr><tr><td>ST 0.0255</td><td>0.0255</td><td>0.1285</td><td>0.1727</td><td>0.1653</td><td>0.0918</td><td>0.3973</td><td>0.8562</td><td>0.1864</td><td>0.1824</td><td>0.1647</td><td>0.00</td><td></td></tr><tr><td>GD</td><td>0.0244</td><td>0.0243</td><td>0.1501</td><td>0.1778</td><td>0.1844</td><td>0.1009</td><td>0.4105</td><td>0.9087</td><td>0.2325</td><td>0.2032</td><td>0.1822</td><td>-8.04</td></tr><tr><td>GradDrop</td><td>0.0253</td><td>0.0253</td><td>0.1533</td><td>0.1785</td><td>0.1865</td><td>0.1021</td><td>0.4399</td><td>0.9246</td><td>0.2408</td><td>0.2063</td><td>0.1791</td><td>-10.42</td></tr><tr><td>MGDA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UW</td><td>0.0242</td><td>0.0242</td><td>0.1498</td><td>0.1778</td><td>0.1847</td><td>0.1007</td><td>0.4064</td><td>0.9079</td><td>0.2312</td><td>0.2033</td><td>0.1822</td><td>-7.74</td></tr><tr><td>DWA</td><td>0.0242</td><td>0.0242</td><td>0.1500</td><td>0.1778</td><td>0.1844</td><td>0.1008</td><td>0.4097</td><td>0.9071</td><td>0.2316</td><td>0.2032</td><td>0.1822</td><td>-7.84</td></tr><tr><td>PCGrad</td><td>0.0248</td><td>0.0248</td><td>0.1501</td><td>0.1755</td><td>0.1761</td><td>0.1001</td><td>0.4306</td><td>0.9181</td><td>0.2371</td><td>0.2023</td><td>0.1772</td><td>-8.12</td></tr><tr><td>CAGrad</td><td>0.0254</td><td>0.0255</td><td>0.1516</td><td>0.1738</td><td>0.1698</td><td>0.0983</td><td>0.4535</td><td>0.9282</td><td>0.2442</td><td>0.2068</td><td>0.1849</td><td>-9.74</td></tr><tr><td>IMTL</td><td>0.0236</td><td>0.0237</td><td>0.1456</td><td>0.1756</td><td>0.1760</td><td>0.0988</td><td>0.4151</td><td>0.9055</td><td>0.2222</td><td>0.2010</td><td>0.1794</td><td>-5.74</td></tr><tr><td>Align-MTL</td><td>0.0266</td><td>0.0264</td><td>0.1499</td><td>0.1736</td><td>0.1700</td><td>0.0986</td><td>0.4659</td><td>0.9868</td><td>0.2604</td><td>0.2030</td><td>0.1780</td><td>-11.51</td></tr><tr><td>Nash-MTL</td><td>0.0235</td><td>0.0235</td><td>0.1432</td><td>0.1745</td><td>0.1718</td><td>0.0975</td><td>0.4230</td><td>0.9225</td><td>0.2268</td><td>0.1985</td><td>0.1775</td><td>-5.41</td></tr><tr><td>FAMO</td><td>0.0243</td><td>0.0243</td><td>0.1499</td><td>0.1778</td><td>0.1846</td><td>0.1008</td><td>0.3841</td><td>0.9080</td><td>0.2321</td><td>0.2027</td><td>0.1816</td><td>-7.31</td></tr><tr><td>DTME-MTL</td><td>0.0196</td><td>0.0200</td><td>0.1372</td><td>0.1754</td><td>0.1712</td><td>0.0958</td><td>0.3129</td><td>0.8333</td><td>0.1955</td><td>0.1907</td><td>0.1698</td><td>+3.62</td></tr></table>

Table 10. Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-T. Non-converged results are indicated with a dash.

<table><tr><td>Task Metric</td><td>DE L1 Dist. ↓</td><td>DZ L1 Dist. ↓</td><td>EO L1 Dist. ↓</td><td>ET L1 Dist. ↓</td><td>Key2D L1 Dist. ↓</td><td>Key3D L1 Dist. ↓</td><td>N L1 Dist.</td><td>PC RMSE ↓</td><td>R L1 Dist. ↓</td><td>S2D L1 Dist. ↓</td><td>S25D L1 Dist. ↓</td><td> $\triangle_m \uparrow (\%)$ </td></tr><tr><td>ST</td><td>0.0250</td><td>0.0256</td><td>0.1388</td><td>0.1755</td><td>0.1670</td><td>0.0958</td><td>0.3856</td><td>0.9066</td><td>0.2132</td><td>0.1878</td><td>0.1722</td><td>0.00</td></tr><tr><td>GD</td><td>0.0266</td><td>0.0278</td><td>0.1593</td><td>0.1794</td><td>0.1865</td><td>0.1047</td><td>0.4752</td><td>0.9467</td><td>0.2568</td><td>0.2081</td><td>0.1897</td><td>-11.10</td></tr><tr><td>GradDrop</td><td>0.0276</td><td>0.0284</td><td>0.1624</td><td>0.1807</td><td>0.1884</td><td>0.1064</td><td>0.4741</td><td>0.9611</td><td>0.2658</td><td>0.2108</td><td>0.1860</td><td>-12.67</td></tr><tr><td>MGDA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UW</td><td>0.0266</td><td>0.0277</td><td>0.1593</td><td>0.1795</td><td>0.1865</td><td>0.1045</td><td>0.4757</td><td>0.9466</td><td>0.2567</td><td>0.2080</td><td>0.1896</td><td>-11.07</td></tr><tr><td>DWA</td><td>0.0266</td><td>0.0274</td><td>0.1593</td><td>0.1794</td><td>0.1866</td><td>0.1045</td><td>0.4743</td><td>0.9465</td><td>0.2567</td><td>0.2080</td><td>0.1897</td><td>-10.95</td></tr><tr><td>PCGrad</td><td>0.0273</td><td>0.0285</td><td>0.1596</td><td>0.1768</td><td>0.1807</td><td>0.1043</td><td>0.4785</td><td>0.9689</td><td>0.2644</td><td>0.2080</td><td>0.1854</td><td>-11.55</td></tr><tr><td>CAGrad</td><td>0.0290</td><td>0.0305</td><td>0.1641</td><td>0.1747</td><td>0.1731</td><td>0.1051</td><td>0.4884</td><td>0.9870</td><td>0.2828</td><td>0.2136</td><td>0.1945</td><td>-14.64</td></tr><tr><td>IMTL</td><td>0.0263</td><td>0.0272</td><td>0.1558</td><td>0.1772</td><td>0.1810</td><td>0.1025</td><td>0.4730</td><td>0.9525</td><td>0.2458</td><td>0.2065</td><td>0.1868</td><td>-9.24</td></tr><tr><td>Align-MTL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Nash-MTL</td><td>0.0261</td><td>0.0270</td><td>0.1536</td><td>0.1762</td><td>0.1766</td><td>0.1017</td><td>0.4590</td><td>0.9649</td><td>0.2496</td><td>0.2039</td><td>0.1846</td><td>-8.28</td></tr><tr><td>FAMO</td><td>0.0266</td><td>0.0275</td><td>0.1592</td><td>0.1795</td><td>0.1865</td><td>0.1047</td><td>0.4746</td><td>0.9466</td><td>0.2566</td><td>0.2080</td><td>0.1898</td><td>-10.97</td></tr><tr><td>DTME-MTL</td><td>0.0236</td><td>0.0241</td><td>0.1494</td><td>0.1765</td><td>0.1790</td><td>0.0998</td><td>0.4138</td><td>0.8921</td><td>0.2290</td><td>0.1959</td><td>0.1824</td><td>-2.88</td></tr></table>

Table 11. We compare task performance based on the configuration of the modulator. Specifically, we compare the performance of tasks using an affine transformation against those using a batch normalization layer as configurations for the modulator.

<table><tr><td rowspan="2">Model</td><td colspan="4">NYUD-v2</td><td colspan="5">PASCAL-Context</td></tr><tr><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td></tr><tr><td>TM+TE (Affine)</td><td>38.27</td><td>0.6370</td><td>21.64</td><td>57.90</td><td>66.18</td><td>56.29</td><td>83.21</td><td>15.26</td><td>47.00</td></tr><tr><td>TM+TE (Batch Norm)</td><td>37.42</td><td>0.6550</td><td>23.16</td><td>56.10</td><td>60.80</td><td>53.29</td><td>82.59</td><td>15.73</td><td>44.90</td></tr></table>

Table 12. We assess task performance by comparing scenarios where we freeze the backbone network after expansion (w/ Freeze) and where we don’t (w/o Freeze).

<table><tr><td rowspan="2">Model</td><td colspan="4">NYUD-v2</td><td colspan="5">PASCAL-Context</td></tr><tr><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Edge odsF ↑</td></tr><tr><td>TM+TE (w/ Freeze)</td><td>34.80</td><td>0.6730</td><td>22.48</td><td>56.00</td><td>58.34</td><td>52.96</td><td>82.86</td><td>15.63</td><td>43.20</td></tr><tr><td>TM+TE (w/o Freeze)</td><td>38.27</td><td>0.6370</td><td>21.64</td><td>57.90</td><td>66.18</td><td>56.29</td><td>83.21</td><td>15.26</td><td>47.00</td></tr></table>

![](images/b6ff4841a2d438cc5b8a7368a557597c220f60dfedc4bb06924155814b70918e.jpg)

(a) Results on NYUD-v2.  
![](images/06b5c9cf33465873024e65c96d556a18631d575ccde4e6bc37e10774803d12df.jpg)  
(b) Results on PASCAL-Context.

Figure 5. We assess the performance of tasks based on the proportion of total variance r. The results are displayed for both (a) NYUD-v2 and (b) PASCAL-Context  
![](images/1a24760a7fa1729d72d286d77a33798c9031833ffc3a42b5c5218f30a71a1b5f.jpg)

(a) Results on NYUD-v2.  
![](images/a726be36e89e334fa1dc33e750edf0f37cf071a21b3719ef172d1d2a837074c4.jpg)  
(b) Results on PASCAL-Context.  
Figure 6. The performance of tasks based on the ratio of the number of expanded layers to the total number of layers. The results are displayed for both (a) NYUD-v2 and (b) PASCAL-Context.

Analyzing Performance Differences with Backbone Network Freezing. In Tab. 12, we examine the performance variation based on whether we freeze the existing backbone network components when training the expanded network after implementing the proposed dynamic token modulation and expansion. The results indicate that training networks without freezing the existing backbone network components leads to significantly better performance compared to training networks with freezing. We guess that allowing modifications to the learned token space after expansion helps the network to dynamically partition the token space for each task.

Influence of r on SVD Approximation. In Fig. 5, we illustrate how the proportion of total variance r impacts the approximation of a token’s range and null space. We assess the performance of tasks across five values of r (1, 10, 100, 500, 1000). Our results suggest that the value of r has minimal impact on task performance, implying that there is less need for extensive tuning of the r parameter to optimize performance. In our other experiments, we chose r as 100 for training.

Impact of the Number of Layers Expanded by DTME-MTL. DTME-MTL expands a subset of transformer layers selected based on the severity of gradient conflicts. In Fig. 6, we analyze how varying the number of expanded layers affects task performance. The x-axis denotes the ratio of expanded layers to the total number of layers. We observe that applying TM+TE to approximately 25%–50% of the layers yields consistent performance gains across tasks while maintaining parameter efficiency. Performance improves as more high-conflict layers are expanded, but begins to degrade when expansion exceeds 50%, especially when low-conflict layers are included. This suggests that overexpansion can be detrimental. Table 2 further confirms that using a moderate expansion ratio (50%) avoids overfitting, whereas Fig. 6 highlights that indiscriminate expansion into less conflicting layers harms performance. These findings underscore the importance of both the extent and location of TM+TE application.

Effect of Swapping Conflict Types. In Tab. 13, we present the results of an experiment on NYUD-v2 where we intentionally swap the conflict types targeted by each method. Specifically, Token Expansion (TE) is applied to layers with severe range space conflict, and Token Modulation (TM) is applied to layers with severe null space conflict—opposite to our standard configuration. This reversal leads to a clear performance drop, confirming that each method is most effective when applied to the type of conflict it is designed to resolve. These results support our design choice of assigning TM to range space conflict and TE to null space conflict.

Table 13. Performance comparison across selection strategies.

<table><tr><td>Method</td><td>Random</td><td>Reverse</td><td>Swap</td><td>Standard</td></tr><tr><td> $\triangle_{m} \uparrow (\%)$ </td><td>-2.966</td><td>-6.167</td><td>-2.608</td><td>+0.044</td></tr></table>

## D. Additional Analysis

## Further Justification for Targeted TM/TE Assignment.

Prior work [46] suggests that fine-tuning from a pretrained model tends to remain in the same loss basin, preserving the structure of the pretrained feature space. Accordingly, we view the token space during fine-tuning as constrained by the span of the pretrained features. If the conflict lies within this span (i.e., the range space), it can be resolved by rotating the token space—achievable via a modulator, since affine transformations include rotation. However, if the conflict resides in the null space, it lies outside the span and cannot be sufficiently addressed by modulation alone. In such cases, expanding the token space with task-specific tokens helps relax this constraint. We theoretically support this view in Propositions 1 and 2 (with proofs in Appendix E), which analyze how each method addresses conflict in its respective subspace. This is further validated empirically: we measure the reduction in gradient conflicts by comparing the start and end of training in each space (Tab. 14). The results show that Token Modulation (TM) is more effective in reducing conflicts in the range space, while Token Expansion (TE) is more effective in the null space. This consistency between theoretical analysis and empirical behavior supports our design choice to selectively apply TM and TE based on the dominant type of conflict in each layer.

Token-Level vs. Parameter-Level Conflict Handling. Parameter-space conflicts reflect an aggregate gradient across all tokens, which makes it difficult to localize or disentangle the source of interference. In contrast, token-level conflicts can be measured for each individual token, allowing more localized and fine-grained analysis. This granularity enables our method to selectively modulate or expand tokens based on where the conflict occurs. Furthermore, by decomposing the token space into range and null components—depending on whether the pretrained model already spans those directions—we adaptively apply Token Modulation (TM) or Token Expansion (TE) to address conflicts. Such space-aware conflict resolution is fundamentally infeasible in parameter space, where task interference is entangled across layers and tokens.

Comparison with LoRA in Multi-Task Inference. As shown in Tab. 1, the baseline (ST) corresponds to full fine-tuning and serves as an upper bound on performance. While LoRA [26] is a parameter-efficient method, assigning a separate LoRA module for each task leads to disjoint sets of task-specific weights. Even when merged into the base model, these configurations require separate forward passes per task, negating the efficiency benefits of multi-task learning (MTL). In contrast, our method maintains shared weights across tasks and allows all outputs to be computed jointly in a single batched tensor operation on GPU. This enables highly parallel inference with only a 13.4% overhead per task, whereas the inference time in LoRA scales linearly with the number of tasks.

Table 14. Reduction in gradient conflict numbers (NYUD-v2).

<table><tr><td>Method</td><td> $Num(g_{\mathcal{R},i} \cdot g_{\mathcal{R},i} \leq 0)$ </td><td> $Num(g_{\mathcal{N},i} \cdot g_{\mathcal{N},i} \leq 0)$ </td></tr><tr><td>TM</td><td>11.60 % ↓</td><td>8.92 % ↓</td></tr><tr><td>TE</td><td>4.64% ↓</td><td>15.44 % ↓</td></tr></table>

## E. Theoretical Analysis

## E.1. Proof of Proposition 1

Proposition 1. When the input token $\mathcal { T } _ { i n }$ for input sample $\mathcal { X } _ { l }$ spans the range space of $\cdot \tilde { \tau } _ { s } ,$ , optimizing the token modulators $\{ \mathcal { M } _ { i } \} _ { i = 1 } ^ { \mathcal { K } }$ reduces gradient conflicts in the row space of $\tilde { \mathcal { T } } _ { s }$ and leads to a reduction in the multi-task loss.

Proof. Let the loss function $\mathcal { L } _ { i }$ be a function of the shared parameters $\Theta _ { s }$ , the token modulator $\mathcal { M } _ { i }$ , and the input data $\mathcal { X } _ { l }$ Since transformers convert input data into tokens, we consider the loss to be a function of one of the input tokens, $\mathcal { T } _ { i n } .$ , rather than $\mathcal { X } ^ { l }$ . To represent the updating step during optimization, we use the superscript t for current variables, such as $\boldsymbol { \Theta } _ { s } ^ { t }$ , and $\mathcal { M } _ { i } ^ { t } .$ , and $t + 1$ for the next-step variables, such as $\Theta _ { s } ^ { t + 1 }$ , and $\mathcal { M } _ { i } ^ { t + 1 }$

In cases where the input token $\mathcal { T } _ { i n }$ spans the row space of $\tilde { \mathcal { T } } _ { s }$ , this can be expressed as follows:

$$
\mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T} \nabla_ {\mathcal {T} _ {i n}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {i n}) \simeq 0 \tag {5}
$$

Since the row space and null space are perpendicular to each other, with their dimensions summing to the entire space, the following holds according to Eq. (5):

$$
\sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\mathcal {T} _ {i n}} \mathcal {L} _ {i} = \sum_ {i = 1} ^ {\mathcal {K}} (\mathcal {U} _ {\mathcal {R}} \mathcal {U} _ {\mathcal {R}} ^ {T} + \mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T}) \nabla_ {\mathcal {T} _ {i n}} \mathcal {L} _ {i} \simeq \sum_ {i = 1} ^ {\mathcal {K}} (\mathcal {U} _ {\mathcal {R}} \mathcal {U} _ {\mathcal {R}} ^ {T}) \nabla_ {\mathcal {T} _ {i n}} \mathcal {L} _ {i} \tag {6}
$$

Let the token modulator $\mathcal { M } _ { i }$ be a $p \times p$ matrix that manipulates the input token $\mathcal { T } _ { i n }$

$$
\sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\mathcal {T} _ {i n}} \mathcal {L} _ {i} = \sum_ {i = 1} ^ {\mathcal {K}} (\mathcal {U} _ {\mathcal {R}} \mathcal {M} _ {i} ^ {t}) (\mathcal {U} _ {\mathcal {R}} \mathcal {M} _ {i} ^ {t}) ^ {T} \cdot \nabla_ {\mathcal {M} _ {i} ^ {t}} \mathcal {L} _ {i} \cdot \nabla_ {\mathcal {T} _ {i n}} \mathcal {M} _ {i} ^ {t} \tag {7}
$$

The total multi-task loss can be represented using a Taylor expansion. Assuming $\eta \ll 1$ , we can ignore the second-order terms of $\eta \colon$

$$
\sum_ {i = 1} ^ {\mathcal {K}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t + 1}, \mathcal {M} _ {i} ^ {t + 1}, \mathcal {T} _ {s}) = \sum_ {i = 1} ^ {\mathcal {K}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) + \sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\Theta_ {s} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) (\Theta_ {s} ^ {t + 1} - \Theta_ {s} ^ {t}) \tag {8}
$$

$$
+ \sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\mathcal {M} _ {i} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) (\mathcal {M} _ {i} ^ {t + 1} - \mathcal {M} _ {i} ^ {t}) \tag {9}
$$

$$
= \sum_ {i = 1} ^ {\mathcal {K}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) - \eta | \sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\Theta_ {s}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) | ^ {2} \tag {10}
$$

$$
- \eta \sum_ {i = 1} ^ {\mathcal {K}} | \nabla_ {\mathcal {M} _ {i} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {M} _ {i} ^ {t}, \mathcal {T} _ {s}) | ^ {2} \tag {11}
$$

By optimizing the modulator $\mathcal { M } _ { i } ^ { t }$ so that $| \nabla _ { \mathcal { M } _ { i } ^ { t } } \mathcal { L } _ { i } ( \Theta _ { s } ^ { t } , \mathcal { M } _ { i } ^ { t } , \mathcal { T } _ { i n } ) |$ approaches zero for each task $i = 1 , 2 , \dots , \kappa .$ , we can alleviate gradient conflicts in the row space of $\tilde { \mathcal { T } } _ { s }$ (as Eq. (7) also approaches zero) and reduce the overall multi-task loss, since Eq. (11) is always greater than or equal to zero. □

## E.2. Proof of Proposition 2

Proposition 2. When the input token $\mathcal { T } _ { i n }$ for input sample $\mathcal { X } _ { l }$ spans the null space of $\tilde { \mathcal { T } } _ { s } ,$ , token expansion using $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { \mathcal { K } }$ alleviates the increase in multi-task loss caused by gradient conflicts in the null space of $\ddot { \tau _ { s } }$

Proof. Let the loss function $\mathcal { L } _ { i }$ be a function of the shared parameters $\Theta _ { s } ^ { t } .$ , the task-specific token $\mathcal { T } _ { i } ^ { t }$ , and the input data $\mathcal { X } ^ { t }$ Similarly, since transformers convert input data into tokens, we consider the loss to be a function of one of the input tokens, $\mathcal { T } _ { i n } ^ { t } .$ , rather than $\mathcal { X } ^ { t }$ . To represent the updating step during optimization, we use the superscript t for current variables, such as $\Theta _ { s } ^ { t } , \mathcal { T } _ { i n } ^ { t }$ and $\mathcal { M } _ { i } ^ { t }$ , and $t + 1$ for the next-step variables, such as $\Theta _ { s } ^ { t + 1 } , T _ { i n } ^ { t + 1 }$ and $\mathcal { M } _ { i } ^ { t + 1 }$

In the case where the input token $\mathcal { T } _ { i n } ^ { t }$ spans the null space of $\tilde { \mathcal { T } } _ { s }$ , this can be expressed as follows:

$$
\sum_ {i = 1} ^ {\mathcal {K}} \mathcal {U} _ {\mathcal {R}} \mathcal {U} _ {\mathcal {R}} ^ {T} \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t}) \simeq 0 \tag {12}
$$

The derivative of the task-specific loss $\mathcal { L } _ { i }$ with respect to the expanded token, including the input token $\mathcal { T } _ { i n } ^ { t }$ and the learnable task-specific tokens $\mathcal { T } _ { i } ^ { t }$ , is given as follows:

$$
\sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {[ \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t} ]} \mathcal {L} _ {i} \tag {13}
$$

$$
= \sum_ {i = 1} ^ {\mathcal {K}} \left(\left[ \begin{array}{l l} \mathcal {U} _ {\mathcal {R}} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & \mathcal {U} _ {\mathcal {R}, i} \end{array} \right] \left[ \begin{array}{l l} \mathcal {U} _ {\mathcal {R}} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & \mathcal {U} _ {\mathcal {R}, i} \end{array} \right] ^ {T} + \left[ \begin{array}{l l} \mathcal {U} _ {\mathcal {N}} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & 0 _ {\mathcal {K} \times \mathcal {K}} \end{array} \right] \left[ \begin{array}{l l} \mathcal {U} _ {\mathcal {N}} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & 0 _ {\mathcal {K} \times \mathcal {K}} \end{array} \right] ^ {T}\right) \left[ \begin{array}{c} \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \\ \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} \end{array} \right] \tag {14}
$$

$$
= \sum_ {i = 1} ^ {\mathcal {K}} \left[ \begin{array}{c c} \mathcal {U} _ {\mathcal {R}} \mathcal {U} _ {\mathcal {R}} ^ {T} + \mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & \mathcal {U} _ {\mathcal {R}, i} \mathcal {U} _ {\mathcal {R}, i} ^ {T} \end{array} \right] \left[ \begin{array}{l} \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \\ \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} \end{array} \right] \tag {15}
$$

$$
\simeq \sum_ {i = 1} ^ {\mathcal {K}} \left[ \begin{array}{c c} \mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T} & 0 _ {d \times \mathcal {K}} \\ 0 _ {\mathcal {K} \times d} & \mathcal {U} _ {\mathcal {R}, i} \mathcal {U} _ {\mathcal {R}, i} ^ {T} \end{array} \right] \left[ \begin{array}{l} \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \\ \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} \end{array} \right] \tag {16}
$$

$$
= \sum_ {i = 1} ^ {\mathcal {K}} \left[ \begin{array}{l} \left(\mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T}\right) \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \\ \left(\mathcal {U} _ {\mathcal {R}, i} \mathcal {U} _ {\mathcal {R}, i} ^ {T}\right) \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} \end{array} \right] \tag {17}
$$

The total multi-task loss can be expressed as follows:

$$
\mathcal {L} _ {i} (\Theta_ {s} ^ {t + 1}, \mathcal {T} _ {i n} ^ {t + 1}, \mathcal {T} _ {i} ^ {t + 1}) = \mathcal {L} _ {i} (\Theta_ {i n} ^ {t}, \mathcal {T} _ {s} ^ {t}, \mathcal {T} _ {i} ^ {t}) + \nabla_ {\Theta_ {s} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {s} ^ {t}, \mathcal {T} _ {i} ^ {t}) (\Theta_ {s} ^ {t + 1} - \Theta_ {s} ^ {t}) \tag {18}
$$

$$
+ \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {s} ^ {t}, \mathcal {T} _ {i} ^ {t}) (\mathcal {T} _ {i n} ^ {t + 1} - \mathcal {T} _ {i n} ^ {t}) \tag {19}
$$

$$
+ \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {s} ^ {t}, \mathcal {T} _ {i} ^ {t}) (\mathcal {T} _ {i} ^ {t + 1} - \mathcal {T} _ {i} ^ {t}) \tag {20}
$$

$$
= \mathcal {L} _ {i} \left(\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t}\right) - \eta \nabla_ {\Theta_ {s} ^ {t}} \mathcal {L} _ {i} \left(\Theta_ {s} ^ {t}, \mathcal {T} _ {s} ^ {t}, \mathcal {T} _ {i} ^ {t}\right) \cdot \sum_ {i = 1} ^ {\mathcal {K}} \nabla_ {\Theta_ {s} ^ {t}} \mathcal {L} _ {i} \left(\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t}\right) \tag {21}
$$

$$
- \eta \left(\mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T}\right) \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \left(\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t}\right) \cdot \sum_ {i = 1} ^ {\mathcal {K}} \left(\mathcal {U} _ {\mathcal {N}} \mathcal {U} _ {\mathcal {N}} ^ {T}\right) \nabla_ {\mathcal {T} _ {i n} ^ {t}} \mathcal {L} _ {i} \left(\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {t} ^ {t}\right) \tag {22}
$$

$$
- \eta (\mathcal {U} _ {\mathcal {R}, i} \mathcal {U} _ {\mathcal {R}, i} ^ {T}) \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^ {t}) \cdot (\mathcal {U} _ {\mathcal {R}, i} \mathcal {U} _ {\mathcal {R}, i} ^ {T}) \nabla_ {\mathcal {T} _ {i} ^ {t}} \mathcal {L} _ {i} (\Theta_ {s} ^ {t}, \mathcal {T} _ {i n} ^ {t}, \mathcal {T} _ {i} ^{t}) \tag {23}
$$

The increase in multi-task loss caused by gradient conflicts in the null space (as described in Eq. (22)) cannot be reduced since the shared token $\mathcal { T } _ { i n } ^ { t }$ is not a learnable parameter. Instead, task-specific tokens $\mathcal { T } _ { i } ^ { t }$ can be added to mitigate the increase in multi-task loss due to null space gradient conflicts by optimizing the learnable parameters $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { \mathcal { K } }$ as described in Eq. (23).

![](images/7bee28fcf45ce8eb717afdd2db352438c7f8ca62338641dcdb773d51601dce92.jpg)