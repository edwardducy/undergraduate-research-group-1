# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning

 Wooseong Jeong Affiliation: KAIST Email: [stk14570@kaist.ac.kr](mailto:)    Kuk-Jin Yoon Affiliation: KAIST Email: [kjyoon@kaist.ac.kr](mailto:) 

###### Abstract

Multi-Task Learning (MTL) enables multiple tasks to be learned within a shared network, but differences in objectives across tasks can cause negative transfer, where the learning of one task degrades another task’s performance. While pre-trained transformers significantly improve MTL performance, their fixed network capacity and rigid structure limit adaptability. Previous dynamic network architectures attempt to address this but are inefficient as they directly convert shared parameters into task-specific ones. We propose Dynamic Token Modulation and Expansion (DTME-MTL), a framework applicable to any transformer-based MTL architecture. DTME-MTL enhances adaptability and reduces overfitting by identifying gradient conflicts in token space and applying adaptive solutions based on conflict type. Unlike prior methods that mitigate negative transfer by duplicating network parameters, DTME-MTL operates entirely in token space, enabling efficient adaptation without excessive parameter growth. Extensive experiments demonstrate that DTME-MTL consistently improves multi-task performance with minimal computational overhead, offering a scalable and effective solution for enhancing transformer-based MTL models.

11footnotetext: Our source code is available at: [https://github.com/wooseong97/DTME-MTL](https://github.com/wooseong97/DTME-MTL "")

## 1 Introduction

Multi-Task Learning (MTL) enables multiple tasks to be learned simultaneously within a shared network, improving generalization, efficiency, and convergence speed compared to training separate models \[[6](#bib.bib6 "")\]. However, conflicting objectives among tasks can lead to negative transfer, where learning one task degrades the performance of another \[[10](#bib.bib10 "")\]. The key challenge lies in designing architectures that effectively balance shared and task-specific representations to mitigate negative transfer.

Transformer-based MTL architectures \[[63](#bib.bib63 ""), [64](#bib.bib64 ""), [67](#bib.bib67 "")\] leverage the strong generalization capabilities of large-scale pre-trained networks such as Vision Transformers (ViTs) \[[14](#bib.bib14 "")\]. By utilizing pre-trained transformers trained on large open-source datasets, these architectures demonstrate improved generalization compared to conventional CNN-based MTL methods \[[16](#bib.bib16 ""), [61](#bib.bib61 ""), [57](#bib.bib57 ""), [72](#bib.bib72 ""), [11](#bib.bib11 ""), [41](#bib.bib41 ""), [53](#bib.bib53 ""), [71](#bib.bib71 "")\]. However, they typically rely on predefined modules, such as Task Prompter mechanisms \[[63](#bib.bib63 ""), [64](#bib.bib64 ""), [67](#bib.bib67 "")\] and Mixture of Experts (MoE) \[[48](#bib.bib48 ""), [70](#bib.bib70 ""), [18](#bib.bib18 ""), [44](#bib.bib44 ""), [9](#bib.bib9 "")\], to separate shared and task-specific components. These rigid structures struggle to adapt to dynamic task relationships, leading to inefficient information sharing and suboptimal performance. The degree of task specialization required varies across different network depths \[[15](#bib.bib15 "")\]: high-level tasks such as semantic segmentation demand greater capacity in deeper layers, while low-level tasks like surface normal estimation rely more on shallower layers. Ideally, MTL architectures should dynamically adjust the allocation of shared and task-specific representations to accommodate these variations. However, existing transformer-based MTL frameworks are inherently constrained by their fixed network capacity, limiting their ability to adapt to evolving task dependencies and effectively mitigate negative transfer.

A straightforward approach to addressing these limitations is to increase the size of the transformer backbone. While this expands network capacity, it has a major drawback: it prevents the use of open-source pre-trained networks, which provide strong initialization and generalization capabilities across multiple tasks. Training a larger network from scratch requires massive computational resources and large-scale datasets, making this approach impractical for many applications. Instead, an effective MTL framework should refine existing architectures to retain the advantages of pre-trained transformers while improving adaptability to task-specific needs.

To achieve this, efficient adaptation methods for pre-trained transformer-based MTL architectures are needed. Unlike approaches that build MTL frameworks from scratch or rely on task optimization within a fixed network capacity, we focus on adapting and enhancing predefined architectures while preserving their core design. This allows existing MTL models to be improved efficiently while dynamically adjusting task-specific representations. Despite its potential, the challenge of how to adaptively expand existing multi-task networks remains an underexplored problem.

One possible approach for adapting models during fine-tuning is the use of multi-task optimization techniques \[[33](#bib.bib33 ""), [38](#bib.bib38 ""), [23](#bib.bib23 ""), [36](#bib.bib36 ""), [13](#bib.bib13 ""), [49](#bib.bib49 ""), [68](#bib.bib68 ""), [35](#bib.bib35 ""), [37](#bib.bib37 ""), [45](#bib.bib45 ""), [50](#bib.bib50 "")\], which mitigate negative transfer by adjusting task loss weights or modifying gradients. While these methods help balance task performance, they remain limited by a fixed network design and cannot expand model capacity.

A more direct approach involves dynamic network architectures, such as Recon \[[22](#bib.bib22 "")\], which directly expand network capacity to mitigate negative transfer. Recon measures conflicting gradients \[[68](#bib.bib68 "")\] in each layer—where gradients from different tasks point in opposing directions—and transforms the most conflicting layers into task-specific ones. While this increases flexibility by expanding the capacity of predefined architectures, directly converting shared parameters into task-specific ones in transformers leads to parameter inefficiency, excessive computational overhead, and a higher risk of overfitting. Consequently, its scalability to large transformer-based architectures is limited.

To address these challenges, we propose Dynamic Token Modulation and Expansion (DTME-MTL), a novel framework designed to improve pre-trained transformer-based MTL architectures. Unlike previous methods that directly manipulate network parameters, our approach mitigates negative transfer by restructuring the token space of multi-task networks. We treat transformer tokens as learnable parameters and analyze their structure using singular value decomposition (SVD) to identify gradient conflicts in token space. These conflicts are categorized into two types: range space conflicts, which are addressed through modulation via affine transformation of existing tokens, and null space conflicts, which are resolved by introducing new task-specific tokens through expansion.

In our experiments, we demonstrate that DTME-MTL effectively enhances multi-task performance with minimal parameter overhead. Additionally, our results highlight that resolving task conflicts in the token space improves adaptability while mitigating overfitting.

Our main contributions are summarized as follows:

*   •

```
We propose DTME-MTL which dynamically modulates and expands token spaces to mitigate negative transfer in transformer-based multi-task architectures.
```
*   •

```
We introduce a structured approach to resolving gradient conflicts in token space by categorizing them into range and null space conflicts, demonstrating how this improves multi-task performance.
```
*   •

```
DTME-MTL is an off-the-shelf solution that seamlessly integrates with existing state-of-the-art transformer-based MTL architectures, enhancing performance with minimal computational overhead.
```
## 2 Related Works

Multi-Task Learning in Vision Transformers. Originally designed for NLP tasks, transformers have outperformed existing CNN models in various computer vision tasks. Attempts have been made to incorporate Vision Transformer \[[14](#bib.bib14 ""), [39](#bib.bib39 ""), [58](#bib.bib58 ""), [65](#bib.bib65 ""), [60](#bib.bib60 ""), [59](#bib.bib59 "")\] in MTL. MTFormer \[[62](#bib.bib62 "")\] employs a shared transformer encoder and decoder with a cross-task attention mechanism. MulT \[[1](#bib.bib1 "")\] utilizes a shared attention mechanism to model task dependencies based on the Swin transformer. InvPT \[[66](#bib.bib66 "")\] focuses on global spatial position and multi-task context for dense prediction tasks through multi-scale feature aggregation. Mixture of Experts (MoE), inspired by the NLP domain, divides the model into predefined expert groups, adaptively shared or devoted to specific tasks during the learning phase \[[48](#bib.bib48 ""), [70](#bib.bib70 ""), [18](#bib.bib18 ""), [44](#bib.bib44 ""), [9](#bib.bib9 ""), [27](#bib.bib27 "")\]. Task prompter \[[63](#bib.bib63 ""), [64](#bib.bib64 ""), [67](#bib.bib67 "")\] uses task-specific tokens to encapsulate task-specific information and employs cross-task interactions to enhance multi-task performance.

Dynamic Network Architectures for MTL. Dynamic networks adapt their structure during training to improve efficiency and task specialization. Several methods have explored dynamic architectures for MTL. Channel-wise dynamic allocation \[[2](#bib.bib2 "")\] assigns different convolutional channels to different tasks, but this method is not directly applicable to transformer-based architectures. Neural Architecture Search (NAS) for MTL \[[34](#bib.bib34 ""), [47](#bib.bib47 ""), [24](#bib.bib24 ""), [21](#bib.bib21 ""), [3](#bib.bib3 ""), [52](#bib.bib52 ""), [5](#bib.bib5 "")\] explores optimal network configurations but is computationally expensive and incompatible with large pre-trained backbone models such as ViTs \[[14](#bib.bib14 "")\]. Recon \[[22](#bib.bib22 "")\] transforms shared parameters directly into task-specific ones to handle conflicting gradients. Unlike most dynamic network architectures, our approach focuses on a dynamic system that can be directly applied to transformer-based multi-task architectures, leveraging pre-trained backbones while maintaining a reasonable computational cost.

![Refer to caption](2507.07485v2/figure/Overview_final.png)

Figure 1: Framework overview of the proposed DTME-MTL. (a) At each network layer, we compute the input token’s range space ℛ⁡(𝒯\~sd)\\mathcal{R}(\\widetilde{\\mathcal{T}}\_{s}^{d}) and their task-specific gradients, determining principal vectors from the uncentered covariance of 𝒯s\\mathcal{T}\_{s}. (b) In cases where task-specific gradients conflict in the range space of 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d} (e.g. gℛ,i⋅gℛ,j≤0g\_{\\mathcal{R},i}\\cdot g\_{\\mathcal{R},j}\\leq 0), modulation is applied to 𝒯s\\mathcal{T}\_{s} by introducing ℳi\\mathcal{M}\_{i} and ℳj\\mathcal{M}\_{j}. (c) When task-specific gradients conflict within the null space of 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d} (e.g. g𝒩,i⋅g𝒩,j≤0g\_{\\mathcal{N},i}\\cdot g\_{\\mathcal{N},j}\\leq 0), task-specific tokens 𝒯i\\mathcal{T}\_{i} and 𝒯j\\mathcal{T}\_{j} are added.

Multi-Task Optimization. Optimizing the MTL aims to address negative transfer by adjusting the relative weighting of task losses or directly manipulating gradients. Task-dependent uncertainty \[[33](#bib.bib33 "")\] is utilized to weigh the loss of multiple tasks. [Liu et al. \[38\]](#bib.bib38 "") considers the rate of loss descent, while \[[23](#bib.bib23 "")\] prioritizes tasks based on difficulty. Recently, [Liu et al. \[36\]](#bib.bib36 "") proposed updating task weights based on the loss history. In contrast, approaches like \[[13](#bib.bib13 ""), [49](#bib.bib49 ""), [68](#bib.bib68 ""), [35](#bib.bib35 ""), [37](#bib.bib37 ""), [45](#bib.bib45 ""), [50](#bib.bib50 "")\] directly modify task gradients to achieve the desired balance. PCGrad \[[68](#bib.bib68 "")\] analyzes negative transfer by identifying conflicting gradients in the shared parameters of the network. [Jiang et al. \[31\]](#bib.bib31 "") suggests a positive link between negative transfer and conflicting gradients in auxiliary task learning. However, the conventional view in MTL considers conflicting gradients a key factor contributing to negative transfer in joint multi-task learning optimization \[[13](#bib.bib13 ""), [49](#bib.bib49 ""), [68](#bib.bib68 ""), [35](#bib.bib35 ""), [37](#bib.bib37 ""), [45](#bib.bib45 ""), [50](#bib.bib50 ""), [29](#bib.bib29 ""), [30](#bib.bib30 "")\], where tasks are learned together rather than serving as auxiliary tasks. Therefore, we adopt a similar perspective. Normalized gradients are employed to prevent spillover between tasks \[[7](#bib.bib7 "")\], whereas [Chen et al. \[8\]](#bib.bib8 "") introduce stochasticity to the network’s parameters based on the consistency in the sign of gradients. RotoGrad \[[28](#bib.bib28 "")\] rotates the feature space of the network to narrow the gap between tasks.

## 3 Preliminaries

In multi-task learning, the network learns a set of tasks {τi}i\=1𝒦\\{\\tau\_{i}\\}\_{i=1}^{\\mathcal{K}} jointly, where 𝒦\\mathcal{K} is the number of tasks. Each task τi\\tau\_{i} has its own loss function ℒi\\mathcal{L}\_{i}. The network parameter Θ\\Theta can be classified into Θ\={Θs,Θ1,…,Θ𝒦}\\Theta=\\{\\Theta\_{s},\\Theta\_{1},...,\\Theta\_{\\mathcal{K}}\\} where Θs\\Theta\_{s} is shared parameter across all tasks and Θi\\Theta\_{i} is task-specific parameters devoted to task τi\\tau\_{i}. Then, the objective function of multi-task learning is to minimize the weighted sum of all tasks’ losses: Θ∗\=arg​minΘ∑i\=1𝒦wiℒi(Θs,Θi)\\Theta^{\*}=\\argmin\_{\\Theta}\\sum^{\\mathcal{K}}\_{i=1}w\_{i}\\mathcal{L}\_{i}(\\Theta\_{s},\\Theta\_{i}) where wiw\_{i} represents the scale of the task-specific loss ℒi\\mathcal{L}\_{i}. A phenomenon called conflicting gradients \[[68](#bib.bib68 "")\], where the gradients of each objective point in different directions, has been identified as a main cause of negative transfer.

###### Definition 1 (Conflicting gradients).

Define gig\_{i} as the gradient of task τi\\tau\_{i} with respect to the shared parameters Θs\\Theta\_{s} as gi\=∇Θsℒi​(Θs,Θi)g\_{i}=\\nabla\_{\\Theta\_{s}}\\mathcal{L}\_{i}(\\Theta\_{s},\\Theta\_{i}). Let gig\_{i} and gjg\_{j} represent the gradients for a pair of tasks τi\\tau\_{i} and τj\\tau\_{j} where i≠ji\\neq j. If gi⋅gj≤0g\_{i}\\cdot g\_{j}\\leq 0, these two gradients are termed conflicting gradients.

However, the role of conflicting gradients remains a topic of debate. While conventional MTL optimization studies \[[13](#bib.bib13 ""), [49](#bib.bib49 ""), [68](#bib.bib68 ""), [35](#bib.bib35 ""), [37](#bib.bib37 ""), [45](#bib.bib45 ""), [50](#bib.bib50 ""), [29](#bib.bib29 "")\] consider conflicting gradients as a main cause of negative transfer, [Jiang et al. \[31\]](#bib.bib31 "") argue that they can serve as a form of regularization that improves generalization when present in network parameters. Our findings align with [Jiang et al. \[31\]](#bib.bib31 "") in that directly resolving conflicting gradients by converting shared parameters into task-specific ones \[[22](#bib.bib22 "")\] leads to overfitting when applied to transformers. In contrast, we propose a token-based network expansion approach that categorizes gradient conflicts within token space and adapts accordingly, mitigating negative transfer while maintaining generalization.

![Refer to caption](2507.07485v2/figure/variance_proportion.png)

Figure 2: The process approximates the range and null spaces of 𝒯\~sd\\tilde{\\mathcal{T}}\_{s}^{d} based on the proportion of total variance, rr. These eigenvalues are arranged in descending order, satisfying λi≥λj\\lambda\_{i}\\geq\\lambda\_{j} if i<ji\<j. If rr is greater than the sum up to λm\\lambda\_{m} and smaller than the sum up to λm+1\\lambda\_{m+1}, then we select the set {λi}i\=1m\\{\\lambda\_{i}\\}\_{i=1}^{m} as Λℛ\\Lambda\_{\\mathcal{R}}, and the remaining set {λi}i\=m+1p\\{\\lambda\_{i}\\}\_{i=m+1}^{p} as Λ𝒩\\Lambda\_{\\mathcal{N}}.

## 4 Method

In order to mitigate negative transfer by ensuring sufficient space for tasks, we adopt token-based network expansion. Initially, we define the token space as the output of each layer in the transformer block through singular value decomposition (SVD). Subsequently, we categorize conflicts in task-specific gradients into two types: conflicts in the range space of tokens and conflicts in the null space of tokens. Finally, based on the type of conflict, we introduce efficient token modulation and expansion techniques for transformer-based multi-task architectures.

### 4.1 Defining Token Space using SVD

In this section, we create a vector space consisting of shared tokens in a transformer, aiming to classify the types of conflicting gradients. More specifically, we approximate the range space and null space of the uncentered covariance of the tokens before applying our methods.

Let’s consider a dataset {𝒳l,𝒴l}l\=1n\\{\\mathcal{X}\_{l},\\mathcal{Y}\_{l}\\}\_{l=1}^{n}, where 𝒳l\\mathcal{X}\_{l} represents the input, 𝒴l\\mathcal{Y}\_{l} denotes the label, and nn is the number of samples. Denote input shared token for a layer dd as 𝒯sl,d\=\[𝒯s,1l,d,𝒯s,2l,d,…,𝒯s,Nl,d\]\\mathcal{T}\_{s}^{l,d}=\[\\mathcal{T}\_{s,1}^{l,d},\\mathcal{T}\_{s,2}^{l,d},...,\\mathcal{T}\_{s,N}^{l,d}\] where NN is the total number of shared tokens in that layer. Every token 𝒯s,kl,d∈ℝp\\mathcal{T}\_{s,k}^{l,d}\\in\\mathbb{R}^{p} represents the output of the transformer layer d−1d-1 for the corresponding input data 𝒳l\\mathcal{X}\_{l}, where pp is the hidden dimension of the token embedding. Let’s consider a total of DD transformer layers. Next, the uncentered covariance of the token in layer dd (where 1≤d≤D1\\leq d\\leq D) is as follows:

𝒯\~sd\=1n​∑l\=1n(𝒯sl,d)​(𝒯sl,d)T\\displaystyle\\widetilde{\\mathcal{T}}\_{s}^{d}=\\frac{1}{n}\\sum\_{l=1}^{n}(\\mathcal{T}\_{s}^{l,d})(\\mathcal{T}\_{s}^{l,d})^{T}

(1)

𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d} is a square matrix of dimensions p×pp\\times p. To define the token space, we apply SVD to 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d}. Following this, we can divide vector space formed by 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d} into its range space ℛ⁡(𝒯\~sd)\\mathcal{R}(\\widetilde{\\mathcal{T}}\_{s}^{d}) and null space 𝒩⁡(𝒯\~sd)\\mathcal{N}(\\widetilde{\\mathcal{T}}\_{s}^{d}) depending on the magnitude of eigenvalue Λ\\Lambda. The process is illustrated below:

𝒯\~sd\=𝒰​Λ​𝒱T,Λ\=\[Λℛ00Λ𝒩\]\\displaystyle\\widetilde{\\mathcal{T}}\_{s}^{d}=\\mathcal{U}\\Lambda\\mathcal{V}^{T},\\hskip 10.0pt\\Lambda=\\begin{bmatrix}\\Lambda\_{\\mathcal{R}}&0\\\\ 0&\\Lambda\_{\\mathcal{N}}\\end{bmatrix}

(2)

where Λ\\Lambda is a diagonal matrix. Each Λℛ\\Lambda\_{\\mathcal{R}} and Λ𝒩\\Lambda\_{\\mathcal{N}} represent submatrices of Λ\\Lambda containing the eigenvalues of the range space and null space, respectively. Both 𝒰\\mathcal{U} and 𝒱\\mathcal{V} are square matrices, each with dimensions p×pp\\times p.

From [Eq. 2](#S4.E2 "In 4.1 Defining Token Space using SVD ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we obtain a mathematical tool to define the range and null space of the covariance of the token, 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d}. To approximate the range space, we choose the eigenvalue Λℛ\\Lambda\_{\\mathcal{R}} along with their corresponding eigenvectors from 𝒰ℛ\\mathcal{U}\_{\\mathcal{R}}. On the other hand, when approximating the null space, we should select the eigenvalues Λ𝒩\\Lambda\_{\\mathcal{N}} and their corresponding eigenvectors from 𝒰𝒩\\mathcal{U}\_{\\mathcal{N}}. Ideally, we should choose eigenvalues that are exactly zero to form the null space. However, in practice, Λ\\Lambda can not be precisely zero. Therefore, it’s essential to establish a criterion for selecting the eigenvalue to distinguish between these two spaces.

Instead of introducing a new manually designed rule for approximating each range and null space of 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d}, we opt to directly employ the evaluation tool for the SVD \[[32](#bib.bib32 "")\] as criteria for determining the range and null space of tokens. In assessing the accuracy of the SVD approximation, the proportion of total variance, denoted as rr, has been employed:

r\=∑λ∈d​i​a​g​(Λ𝒩)λ∑λ∈d​i​a​g​(Λℛ)λ\\displaystyle r=\\frac{\\sum\_{\\lambda\\in diag({\\Lambda\_{\\mathcal{N}}})}\\lambda}{\\sum\_{\\lambda\\in diag({\\Lambda\_{\\mathcal{R}}})}\\lambda}

(3)

The d​i​a​gdiag function serves as an operator, returning a set containing the diagonal entries of the input matrix. In our approach, we employ [Eq. 3](#S4.E3 "In 4.1 Defining Token Space using SVD ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") to directly divide the range and null space of 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d}. As depicted in [Fig. 2](#S3.F2 "In 3 Preliminaries ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), the diagonal elements of the matrix Λ\\Lambda, obtained through the SVD of 𝒯\~sd\\tilde{\\mathcal{T}}\_{s}^{d}, are arranged in descending order based on their magnitudes. We can select the index of the eigenvalue mm such that the sum of eigenvalues up to order mm is smaller than rr, and the sum up to m+1m+1 is larger than rr. This index serves as a boundary to divide the range space and null space of 𝒯\~sd\\tilde{\\mathcal{T}}\_{s}^{d}.

### 4.2 Types of Gradient Conflicts

In Section [4.1](#S4.SS1 "4.1 Defining Token Space using SVD ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we create a pp-dimensional vector space using the uncentered covariance of the shared token 𝒯\~sd\\widetilde{\\mathcal{T}}\_{s}^{d}, linked to the input data set {𝒳}l\=1n\\{\\mathcal{X}\\}\_{l=1}^{n}. This vector space is divided into the range and null space, with each space spanned by eigenvectors corresponding to singular values selected based on a specified ratio rr. In the upcoming sections, we pinpoint the types of gradient conflict within the vector space we’ve constructed. We then address these conflicts adaptively by introducing token modulation and expansion techniques.

Using [Eq. 2](#S4.E2 "In 4.1 Defining Token Space using SVD ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") and [Eq. 3](#S4.E3 "In 4.1 Defining Token Space using SVD ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we can partition the eigenvectors of the pp-dimensional vector space into its range and null space. Now, let’s consider the shared tokens 𝒯sl,d\=\[𝒯s,1l,d,…,𝒯s,Nl,d\]\\mathcal{T}\_{s}^{l,d}=\[\\mathcal{T}\_{s,1}^{l,d},\\dots,\\mathcal{T}\_{s,N}^{l,d}\], omitting the explicit notation of l,dl,d for simplicity. For example, we write 𝒯sl,d→𝒯s\\mathcal{T}\_{s}^{l,d}\\rightarrow\\mathcal{T}\_{s}, 𝒯s,kl,d→𝒯s,k\\mathcal{T}\_{s,k}^{l,d}\\rightarrow\\mathcal{T}\_{s,k}, and 𝒯\~sd→𝒯\~s\\tilde{\\mathcal{T}}\_{s}^{d}\\rightarrow\\tilde{\\mathcal{T}}\_{s}. We treat 𝒯s\\mathcal{T}\_{s} as network parameters, for which gradients can be computed during the backpropagation process. Then, for each loss ℒi\\mathcal{L}\_{i}, the task-specific gradient for 𝒯s,k\\mathcal{T}\_{s,k} is denoted as gi\=∇𝒯s,kℒig\_{i}=\\nabla\_{\\mathcal{T}\_{s,k}}\\mathcal{L}\_{i}. Consequently, we obtain task-specific gradients {gi}i\=1𝒦\\{g\_{i}\\}\_{i=1}^{\\mathcal{K}} corresponding to a set of losses {ℒi}i\=1𝒦\\{\\mathcal{L}\_{i}\\}\_{i=1}^{\\mathcal{K}} for 𝒯s\\mathcal{T}\_{s} as shown in [Fig. 1](#S2.F1 "In 2 Related Works ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")-(a).

Each task-specific gradient gig\_{i} can be decomposed into two components, gℛ,ig\_{\\mathcal{R},i} and g𝒩,ig\_{\\mathcal{N},i}, through projection onto the range and null space of 𝒯\~sd\\tilde{\\mathcal{T}}\_{s}^{d}, respectively. This breakdown is expressed as follows:

gℛ,i\=(𝒰ℛ​𝒰ℛT)​∇𝒯s,kℒi\\displaystyle g\_{\\mathcal{R},i}=(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T})\\nabla\_{\\mathcal{T}\_{s,k}}\\mathcal{L}\_{i}

g𝒩,i\=(𝒰𝒩​𝒰𝒩T)​∇𝒯s,kℒi\\displaystyle g\_{\\mathcal{N},i}=(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{s,k}}\\mathcal{L}\_{i}

(4)

𝒰ℛ\\mathcal{U}\_{\\mathcal{R}} and 𝒰𝒩\\mathcal{U}\_{\\mathcal{N}} are orthogonal matrices that consist of eigenvectors of the range space and null space, respectively, with each column representing one eigenvector. Then, the matrices (𝒰ℛ​𝒰ℛT)(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T}) and (𝒰𝒩​𝒰𝒩T)(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T}) function as projection operators onto the range and null spaces, respectively.

Building upon the concept of conflicting gradients outlined in [Definition 1](#Thmdefinition1 "Definition 1 (Conflicting gradients). ‣ 3 Preliminaries ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we classify conflicts into two types based on the space in which they occur: range space conflicts and null space conflicts. Specifically, conflicts in the range space of tokens occur when gℛ,i⋅gℛ,j≤0g\_{\\mathcal{R},i}\\cdot g\_{\\mathcal{R},j}\\leq 0 for any pair of ii and jj where i≠ji\\neq j. Likewise, conflicts in the null space of tokens emerge when g𝒩,i⋅g𝒩,j≤0g\_{\\mathcal{N},i}\\cdot g\_{\\mathcal{N},j}\\leq 0.

### 4.3 Token Modulation and Expansion

Building on the gradient conflict types defined in [Sec. 4.2](#S4.SS2 "4.2 Types of Gradient Conflicts ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we propose adaptive strategies to mitigate task interference. Specifically, if gradient conflicts occur in the range space, we apply an affine transformation to modulate tokens, while conflicts in the null space are addressed by introducing additional tokens to expand the feature space. This distinction is particularly relevant in the transfer learning setting, where a pretrained transformer backbone is used, and task interference must be handled during fine-tuning. According to \[[46](#bib.bib46 "")\], training from pretrained weights constrains the model within the same basin of the loss landscape, preserving a feature space similar to that of the pretrained network. This insight guides our separation of token space into range and null spaces: conflicts in the row space indicate that the network already has relevant interpretative capabilities and can be resolved through rotation or scaling, whereas conflicts in the null space suggest the need for additional features, requiring token expansion to enhance the model’s capacity.

Token Modulation. In situations where task-specific gradients conflict within the range space of 𝒯\~s\\widetilde{\\mathcal{T}}\_{s}, such as gℛ,i⋅gℛ,j≤0g\_{\\mathcal{R},i}\\cdot g\_{\\mathcal{R},j}\\leq 0, modulators ℳi\\mathcal{M}\_{i} and ℳj\\mathcal{M}\_{j} are added after the shared token 𝒯s\\mathcal{T}\_{s} as shown in [Fig. 1](#S2.F1 "In 2 Related Works ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")-(b). The token modulator ℳ\\mathcal{M} is a straightforward affine transformation that modulates the shared token 𝒯s\\mathcal{T}\_{s} along the channel dimension. To elaborate, considering the embedding dimension of the transformer pp and the number of shared tokens is NN, we can arrange 𝒯s\\mathcal{T}\_{s} in the form \[𝒯s,1,…,𝒯s,N\]\[\\mathcal{T}\_{s,1},\\dots,\\mathcal{T}\_{s,N}\]. This arrangement turns 𝒯s\\mathcal{T}\_{s} into a p×Np\\times N matrix. The modulator ℳ\\mathcal{M}, which incorporates weight and bias W,b∈ℝpW,b\\in\\mathbb{R}^{p}, performs the transformation W⊙𝒯s,i+bW\\odot\\mathcal{T}\_{s,i}+b, where ⊙\\odot denotes elementwise multiplication. When the gradient lies in the row space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s}, [Proposition 1](#Thmproposition1 "Proposition 1. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") demonstrates that applying token modulation can effectively resolve gradient conflicts, lowering the multi-task loss.

###### Proposition 1.

When the input token 𝒯i​n\\mathcal{T}\_{in} for input sample 𝒳l\\mathcal{X}\_{l} spans the range space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s}, optimizing the token modulators {ℳi}i\=1𝒦\\{\\mathcal{M}\_{i}\\}\_{i=1}^{\\mathcal{K}} reduces gradient conflicts in the row space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s} and leads to a reduction in the multi-task loss.

Algorithm 1 DTME-MTL

Data: Task {τi}i\=1𝒦\\{\\tau\_{i}\\}^{\\mathcal{K}}\_{i=1}, Loss function {ℒi}i\=1𝒦\\{\\mathcal{L}\_{i}\\}^{\\mathcal{K}}\_{i=1},  
Dataset {𝒳l,𝒴l}l\=1n\\{\\mathcal{X}\_{l},\\mathcal{Y}\_{l}\\}\_{l=1}^{n}, Shared tokens 𝒯sl,d\={𝒯s,il,d}i\=1N\\mathcal{T}\_{s}^{l,d}=\\{\\mathcal{T}\_{s,i}^{l,d}\\}\_{i=1}^{N}, Depth of the Network DD  
 

1 for *each layer of the network (d←1d\\leftarrow 1 to DD)* do 

    2 Get tokens {𝒯sl,d}l\=1n\\{\\mathcal{T}\_{s}^{l,d}\\}\_{l=1}^{n} for the layer dd corresponding to input {𝒳l}l\=1n\\{\\mathcal{X}\_{l}\\}\_{l=1}^{n} 

    3 Calculate uncentered covariance. 𝒯\~sd\=1n​∑l\=1n(𝒯sl,d)​(𝒯sl,d)T\\widetilde{\\mathcal{T}}\_{s}^{d}=\\frac{1}{n}\\sum\_{l=1}^{n}(\\mathcal{T}\_{s}^{l,d})(\\mathcal{T}\_{s}^{l,d})^{T} 

    4 Singular value decomposition. 𝒰,Λ,𝒱\=S​V​D​(𝒯\~sd)\\mathcal{U},\\Lambda,\\mathcal{V}=SVD(\\widetilde{\\mathcal{T}}\_{s}^{d}) 

    5 Divide range and null space. 𝒰\=\[𝒰ℛ,𝒰𝒩\]\\mathcal{U}=\[\\mathcal{U}\_{\\mathcal{R}},\\mathcal{U}\_{\\mathcal{N}}\] 

    6 Projection to range space. {gℛ,i}i\=1𝒦\={(𝒰ℛ​𝒰ℛT)​∇𝒯s,kl,dℒi}i\=1𝒦\\{g\_{\\mathcal{R},i}\\}\_{i=1}^{\\mathcal{K}}=\\{(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T})\\nabla\_{\\mathcal{T}\_{s,k}^{l,d}}\\mathcal{L}\_{i}\\}\_{i=1}^{\\mathcal{K}} 

    7 Projection to null space. {g𝒩,i}i\=1𝒦\={(𝒰𝒩​𝒰𝒩T)​∇𝒯s,kl,dℒi}i\=1𝒦\\{g\_{\\mathcal{N},i}\\}\_{i=1}^{\\mathcal{K}}=\\{(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{s,k}^{l,d}}\\mathcal{L}\_{i}\\}\_{i=1}^{\\mathcal{K}} 

    8 if *gℛ,i⋅gℛ,j≤0g\_{\\mathcal{R},i}\\cdot g\_{\\mathcal{R},j}\\leq 0* then 

       9 Insert token modulators ℳi\\mathcal{M}\_{i} and ℳj\\mathcal{M}\_{j} 

    10 if *g𝒩,i⋅g𝒩,j≤0g\_{\\mathcal{N},i}\\cdot g\_{\\mathcal{N},j}\\leq 0* then 

       11 Insert task-specific tokens 𝒯i\\mathcal{T}\_{i} and 𝒯j\\mathcal{T}\_{j} 

Table 1: We conduct an ablation study on dynamic token modulation and expansion, evaluating the multi-task performance of our method on NYUD-v2 and PASCAL-Context. The results of TE, TM, and their combination, TE+TM are presented. We employ a shared encoder and multiple decoders, using ViT-T as the backbone network.

NYUD-v2

PASCAL-Context

Semseg

Depth

Normal

Edge

Semseg

Parsing

Saliency

Normal

Edge

Model

mIoU ↑\\uparrow

RMSE ↓\\downarrow

mErr ↓\\downarrow

odsF ↑\\uparrow

mIoU ↑\\uparrow

mIoU ↑\\uparrow

maxF ↑\\uparrow

mErr ↓\\downarrow

odsF ↑\\uparrow

Baseline (ST)

39.35

0.6611

22.14

59.68

67.96

58.90

83.76

15.65

47.70

Baseline (MT)

34.13

0.6732

22.51

55.30

54.47

51.48

82.04

16.22

41.28

TM

37.85

0.6490

21.75

56.92

64.28

55.10

83.02

15.40

45.80

TE

37.25

0.6553

21.87

57.00

60.51

54.00

82.85

15.55

44.98

TM+TE

38.27

0.6370

21.64

57.90

66.18

56.29

83.41

15.26

47.00

Gain (vs. MT)

△\\triangle4.14

△\\triangle0.0362

△\\triangle0.87

△\\triangle2.60

△\\triangle11.71

△\\triangle4.81

△\\triangle1.37

△\\triangle0.96

△\\triangle5.72

△m\\triangle\_{m} ↑\\uparrow

0.044

-1.289

#Param ↑\\uparrow (%)

0.24

0.30

![Refer to caption](2507.07485v2/figure/iter_perform2.png)

Figure 3: Task performance varies based on when we expand the network. To determine the optimal timing, we assess expansions at the beginning of training and at the end of each quarter iteration, monitoring the corresponding changes in performance.

Token Expansion. Similarly, in cases where task-specific gradients conflict within the null space of 𝒯\~s\\widetilde{\\mathcal{T}}\_{s}, such as g𝒩,i⋅g𝒩,j≤0g\_{\\mathcal{N},i}\\cdot g\_{\\mathcal{N},j}\\leq 0, task-specific tokens 𝒯i\\mathcal{T}\_{i} and 𝒯j\\mathcal{T}\_{j} are added alongside shared tokens 𝒯s\\mathcal{T}\_{s} as shown in [Fig. 1](#S2.F1 "In 2 Related Works ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")-(c). The task-specific tokens {𝒯i}i\=1𝒦\\{\\mathcal{T}\_{i}\\}\_{i=1}^{\\mathcal{K}} are concatenated with shared tokens before entering the transformer block. Consequently, each task-specific token acquires task-specific information within that layer. Specifically, in a standard transformer block, self-attention is performed for each pair of tokens in the form of \[𝒯s,1,…,𝒯s,N\]×\[𝒯s,1,…,𝒯s,N\]\[\\mathcal{T}\_{s,1},\\dots,\\mathcal{T}\_{s,N}\]\\times\[\\mathcal{T}\_{s,1},\\dots,\\mathcal{T}\_{s,N}\]. With token expansion, attention is extended to include \[𝒯s,1,…,𝒯s,N\]×\[𝒯1,…,𝒯𝒦\]\[\\mathcal{T}\_{s,1},\\dots,\\mathcal{T}\_{s,N}\]\\times\[\\mathcal{T}\_{1},\\dots,\\mathcal{T}\_{\\mathcal{K}}\] on the output. [Proposition 2](#Thmproposition2 "Proposition 2. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") explains how expanding the token space to address gradient conflicts in the null space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s} leads to a reduction in multi-task loss when the gradient lies in this null space. All proofs are provided in Supple [E](#A5 "Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning").

###### Proposition 2.

When the input token 𝒯i​n\\mathcal{T}\_{in} for input sample 𝒳l\\mathcal{X}\_{l} spans the null space of 𝒯s\~\\tilde{\\mathcal{T}\_{s}}, token expansion using {𝒯i}i\=1𝒦\\{\\mathcal{T}\_{i}\\}\_{i=1}^{\\mathcal{K}} alleviates the increase in multi-task loss caused by gradient conflicts in the null space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s}.

The complete procedure for DTME-MTL is outlined in Alg.[1](#algorithm1 "Algorithm 1 ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"). Handling gradient conflicts in token space improves adaptability and reduces overfitting while being more efficient than addressing conflicts at the parameter level.

Table 2: Performance comparison based on the degree of conflicts in reversed order (Reversed) and randomly selected layers (Random).

|                 |                   |                   |                 |                 |                 |                 |                   |                 |
| --------------- | ----------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | ----------------- | --------------- |
| Semseg          | Depth             | Normal            | Edge            | Semseg          | Parsing         | Saliency        | Normal            | Edge            |
| mIoU ↑\\uparrow | RMSE ↓\\downarrow | mErr ↓\\downarrow | odsF ↑\\uparrow | mIoU ↑\\uparrow | mIoU ↑\\uparrow | maxF ↑\\uparrow | mErr ↓\\downarrow | odsF ↑\\uparrow |
| 38.27           | 0.6370            | 21.64             | 57.90           | 66.18           | 56.29           | 83.21           | 15.26             | 47.00           |
| 36.88           | 0.6567            | 22.27             | 56.30           | 62.12           | 54.43           | 82.95           | 15.55             | 45.80           |
| 34.71           | 0.6898            | 22.59             | 55.80           | 57.84           | 52.82           | 82.75           | 15.74             | 43.20           |

![Refer to caption](2507.07485v2/figure/nyud_conflicts2.png)

(a) NYUD-v2

![Refer to caption](2507.07485v2/figure/pascal_conflicts2.png)

(b) PASCAL-Context

Figure 4: We evaluate the distribution of gradient conflicts by measuring the cosine similarity between task-specific gradients across all shared parameters throughout the optimization process. This is represented as c​o​s​ϕi​jcos\\phi\_{ij} in (a) for NYUD-v2 and in (b) for PASCAL-Context.

## 5 Experiments

### 5.1 Experimental Settings

Datasets and Evaluation. Our method is evaluated on multi-task datasets: NYUD-v2 \[[51](#bib.bib51 "")\], PASCAL-Context \[[43](#bib.bib43 "")\] and Taskonomy \[[69](#bib.bib69 "")\]. Each of them with 4, 5, 11 tasks. To evaluate the performance of tasks, we employed widely used metrics. To evaluate the multi-task performance, we utilize the metric proposed by [Maninis et al. \[42\]](#bib.bib42 ""). It measures the per-task performance Mm,iM\_{m,i} by averaging it with respect to the single-task baseline Mb,iM\_{b,i}, as shown in △m\=(1/𝒦)​∑i\=1𝒦(−1)li​(Mm,i−Mb,i)/Mb,i\\triangle\_{m}=(1/\\mathcal{K})\\sum\_{i=1}^{\\mathcal{K}}(-1)^{l\_{i}}(M\_{m,i}-M\_{b,i})/M\_{b,i}. li\=1l\_{i}=1 if a lower value of the measure MiM\_{i} indicates better performance for task ii, and 0 otherwise.

Baselines and Model Variants. For a comprehensive analysis of the proposed DTME-MTL framework, we adopt a typical experimental setup for MTL in our experiments. In [Tab. 1](#S4.T1 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), ‘Baseline (MT)’ refers to a simple multi-task architecture consisting of a shared transformer backbone and basic task-specific decoders. Each decoder comprises one 3×33\\times 3 Conv-BN-ReLU block. ‘Baseline (ST)’ has the same structure as ‘Baseline (MT)’ but is trained with only a single task. We assess the proposed DTME-MTL framework by expanding the network from ‘Baseline (MT)’ and measure the performance gains achieved by the proposed methods. ‘TM’ (Token Modulation) signifies the addition of the proposed token modulator to ‘baseline (MT)’, while ‘TE’ (Token Expansion) indicates the incorporation of task-specific tokens onto ‘Baseline (MT)’. Finally, ‘TM+TE’ combines both proposed methods. To show how effectively our approach reduces negative transfer, we also compare it with previous multi-task optimization, though our methods can be used alongside them. We include simple gradient descent (GD), GradDrop \[[8](#bib.bib8 "")\], MGDA \[[49](#bib.bib49 "")\], PCGrad \[[68](#bib.bib68 "")\], CAGrad \[[35](#bib.bib35 "")\], IMTL \[[37](#bib.bib37 "")\], Nash-MTL \[[45](#bib.bib45 "")\], and Aligned-MTL \[[50](#bib.bib50 "")\], as well as loss balancing methods such as UW \[[33](#bib.bib33 "")\], DWA \[[38](#bib.bib38 "")\], and FAMO \[[36](#bib.bib36 "")\]. We also compare our results with dynamic network architecture such as Recon \[[22](#bib.bib22 "")\]. Further experimental details are summarized in Supple [B](#A2 "Appendix B Experimental Settings ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning").

Table 3: Comparison of multi-task optimization methods on Taskonomy across 11 tasks. Non-converged results are indicated with a dash.

| Task   | DE     | DZ     | EO     | ET     | Key2D  | Key3D  | N      | PC     | R      | S2D    | S25D     |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | -------- |  |
| 0.0187 | 0.0188 | 0.1301 | 0.1757 | 0.1733 | 0.0942 | 0.3076 | 0.7991 | 0.1826 | 0.1902 | 0.1652 | \- 7.83  |  |
| 0.0315 | 0.0242 | 0.1390 | 0.1776 | 0.1778 | 0.0976 | 0.4564 | 0.8644 | 0.2088 | 0.1995 | 0.1752 | \- 26.11 |  |
| -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -        |  |
| 0.0190 | 0.0190 | 0.1308 | 0.1758 | 0.1734 | 0.0945 | 0.3109 | 0.8009 | 0.1840 | 0.1906 | 0.1657 | \- 8.43  |  |
| 0.0186 | 0.0187 | 0.1294 | 0.1759 | 0.1735 | 0.0938 | 0.2788 | 0.7943 | 0.1805 | 0.1902 | 0.1640 | \- 6.45  |  |
| 0.0217 | 0.0192 | 0.1298 | 0.1775 | 0.1714 | 0.0939 | 0.2856 | 0.7985 | 0.1817 | 0.1927 | 0.1595 | \- 8.29  |  |
| 0.0219 | 0.0203 | 0.1314 | 0.1800 | 0.1665 | 0.0932 | 0.3039 | 0.8121 | 0.1874 | 0.1953 | 0.1673 | \- 10.57 |  |
| 0.0210 | 0.0192 | 0.1282 | 0.1772 | 0.1719 | 0.0936 | 0.2468 | 0.7784 | 0.1734 | 0.1943 | 0.1647 | \- 6.17  |  |
| 0.0189 | 0.0193 | 0.1254 | 0.1728 | 0.1664 | 0.0914 | 0.3524 | 0.8640 | 0.1938 | 0.1889 | 0.1582 | \- 9.41  |  |
| 0.0201 | 0.0184 | 0.1248 | 0.1764 | 0.1701 | 0.0921 | 0.2658 | 0.7793 | 0.1706 | 0.1914 | 0.1624 | \- 5.01  |  |
| 0.0188 | 0.0188 | 0.1300 | 0.1758 | 0.1733 | 0.0942 | 0.3058 | 0.7986 | 0.1826 | 0.1904 | 0.1654 | \- 7.87  |  |
| 0.0150 | 0.0154 | 0.1193 | 0.1733 | 0.1668 | 0.0891 | 0.2038 | 0.7373 | 0.1567 | 0.1773 | 0.1517 | \+ 4.67  |  |

Table 4: Adaptation of DTME-MTL to other state-of-the-art MTL methods on NYUD-v2.

| Task  | Semseg | Depth | Normal | Edge |
| ----- | ------ | ----- | ------ | ---- |
| 50.04 | 0.490  | -     | -      |      |
| 53.56 | 0.5183 | 18.81 | 78.10  |      |
| 54.38 | 0.5020 | 18.51 | 78.20  |      |
| 55.30 | 0.5152 | 18.47 | 78.20  |      |
| 56.36 | 0.5122 | 18.38 | 78.40  |      |

Table 5: Adaptation of DTME-MTL to other state-of-the-art MTL methods on PASCAL-Context.

| Task  | Semseg | Parsing | Saliency | Normal | Edge |
| ----- | ------ | ------- | -------- | ------ | ---- |
| 73.51 | 64.26  | 67.24   | -        | -      |      |
| 79.03 | 67.61  | 84.81   | 14.15    | 73.00  |      |
| 81.91 | 71.13  | 84.96   | 13.73    | 73.80  |      |
| 80.89 | 68.89  | 84.83   | 13.72    | 73.50  |      |
| 81.01 | 69.08  | 84.75   | 13.65    | 73.60  |      |

### 5.2 Experimental Results

Effectiveness of Token Modulation and Expansion. We assess the effectiveness of the proposed methods on the NYUD-v2 and PASCAL-Context datasets, with results detailed in [Tab. 1](#S4.T1 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"). In the last three rows of the table, we depict the performance gains compared to the two baselines and the increased number of parameters in “#​P​a​r​a​m↑\\#Param\\uparrow (%\\%)”. Compared to the Baseline (MT), our methods demonstrate significant performance improvements across all tasks in both datasets. Particularly noteworthy is the substantial increase in multi-task performance achieved with just a 0.2% to 0.3% increase in the total network parameters. Additionally, our approach exhibits comparable performance to Baseline (ST) in a multi-task scenario. This implies that reducing negative transfer is achievable by simply integrating token modulators and task tokens, without complex modules.

Analysis of the Timing of Network Expansion. In [Fig. 3](#S4.F3 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we analyze the performance of each task according to the timing of network expansion using the proposed DTME-MTL. Specifically, the timing for expansion refers to the point at which token modulation and expansion are performed based on calculations of the token space using Singular Value Decomposition and measurement of gradient conflicts. The figure illustrates the performance results when network expansion is conducted at the beginning of training (0t​h0^{th}) and after each quarter of the entire training process (it​hi^{th} 25%25\\% Iter). To ensure fair comparisons, we trained the network using the same number of iterations after the expansion. The results indicate that the optimal timing for expansion may vary across tasks. However, overall, early-stage expansion during network training tends to yield better performance. This aligns with the fact that our approach builds on pre-trained backbone networks.

Analysis of Gradient Conflicts in Parameters. We focus on resolving gradient conflicts in token space. While our primary method operates in token-level representations, we also monitor gradient conflicts in parameter space to better understand training dynamics. In [Fig. 4](#S4.F4 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we visualize the distribution of angles between task-specific gradients of network parameters, categorizing them into different ranges and tracking their frequency over the course of training. When applied to the baseline model, both Token Modulation (TM) and Token Expansion (TE) reduce gradient conflicts in parameter space to some extent. However, as shown in [Tab. 7](#S5.T7 "In 5.2 Experimental Results ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), where our methods are compared with Recon \[[22](#bib.bib22 "")\], we observe important differences. Recon explicitly suppresses conflicts by modifying network parameters whenever the cosine similarity between task gradients becomes negative. Although this reduces gradient conflicts in parameter space, it often results in severe overfitting and degraded multi-task performance. These findings suggest that conflicts in parameter space are not always reliable indicators of negative transfer. Instead, resolving conflicts in token space offers an alternative strategy that avoids overfitting while still mitigating interference. Additional analysis of token-level conflicts is provided in [Appendix D](#A4 "Appendix D Additional Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning").

Computational Cost of DTME-MTL. In [Tab. 6](#S5.T6 "In 5.2 Experimental Results ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we report the time consumption for each process of DTME-MTL on PASCAL-Context using a single NVIDIA RTX A6000. We measure the time required for calculating the token space with SVD and for computing gradient conflicts in the token space of the transformer. The time required for each process increases with the size of the transformer. However, the proposed methods are computationally efficient, requiring approximately 1 hour with ViT-L. Considering that typical multi-task architectures require at least one day of training on a single GPU, the computational cost of DTME-MTL is relatively low. Proposed DTME-MTL increases inference time of each task about 13.4%13.4\\% with ViT-B.

Comparing Performance based on Layer Selection Criteria. In [Tab. 2](#S4.T2 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we applied TM+TE to layers with the highest gradient conflicts between tasks. Results are also shown for randomly chosen layers (Random) or layers with the lowest gradient conflicts (Reverse). The network expansion system, using conflict detection, outperforms random selection across all tasks. Particularly, applying TM+TE to layers with severe conflict levels consistently outperforms its application in layers with lower conflict levels, validating the effectiveness of the strategy.

Comparison with Multi-Task Optimization. In [Tab. 3](#S5.T3 "In 5.1 Experimental Settings ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we compare DTME-MTL with previous multi-task optimization approaches to demonstrate its effectiveness in reducing negative transfer between tasks on the Taskonomy benchmark using ViT-B. DTME-MTL achieves the best multi-task performance, improving each task by an average of 4.67%4.67\\% with only a 0.118%0.118\\% increase in the number of parameters. Although DTME-MTL introduces additional parameters to address negative transfer, making direct comparisons with optimization methods less straightforward, it consistently improves multi-task performance. However, using more task-specific parameters does not always lead to better performance. As as shown in [Tab. 7](#S5.T7 "In 5.2 Experimental Results ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), Recon shows poor results with the vision transformer on NYUD-v2. This comparison highlights that previous multi-task expansion approaches, which naively duplicate network branches, are not only parameter-inefficient but also prone to overfitting due to the increased complexity of transformers.

Table 6: Time consumption of each process in DTME-MTL across different backbone sizes, recorded in minutes.

| Process (min)               | ViT-T | ViT-S | ViT-B | ViT-L |
| --------------------------- | ----- | ----- | ----- | ----- |
| Calculate Token Space (SVD) | 3.61  | 3.74  | 11.54 | 11.96 |
| Calculate Gradient Conflict | 8.25  | 16.43 | 21.94 | 58.66 |

Table 7: Comparison with Recon on NYUD-v2

|       | Semseg | Depth | Normal | Edge  | #Param ↑\\uparrow (%) |
| ----- | ------ | ----- | ------ | ----- | --------------------- |
| 34.13 | 0.673  | 22.51 | 56.38  | 0.0   |                       |
| 31.92 | 0.693  | 23.35 | 52.80  | 23.34 |                       |
| 38.27 | 0.6370 | 21.64 | 57.90  | 0.24  |                       |

Adapting to Multi-Task Architectures. In [Tabs. 4](#S5.T4 "In 5.1 Experimental Settings ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") and [5](#S5.T5 "Table 5 ‣ 5.1 Experimental Settings ‣ 5 Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we compare DTME-MTL with leading multi-task architectures on the NYUD-v2 and PASCAL-Context datasets. We evaluate its multi-task performance against transformer-based approaches. Our method is compatible with any transformer-based multi-task architecture, enabling us to assess its effectiveness by integrating it into two leading models: InvPT and TaskPrompter. DTME-MTL seamlessly enhances these architectures, significantly boosting performance with only a minimal increase in parameters — just 0.048%0.048\\% for InvPT and 0.046%0.046\\% for TaskPrompter.

## 6 Conclusion

We introduced Dynamic Token Modulation and Expansion for Multi-Task Learning, an efficient approach for improving transformer-based MTL architectures. By categorizing gradient conflicts into range space and null space, it adaptively applies token modulation and expansion to mitigate negative transfer and reduce overfitting. DTME-MTL seamlessly integrates with existing transformer-based MTL frameworks, requiring minimal additional parameters. By refining encoded token space, it provides a lightweight and scalable solution for enhancing multi-task performance.

## Acknowledgment

This research was supported by the Challengeable Future Defense Technology Research and Development Program through the Agency For Defense Development(ADD) funded by the Defense Acquisition Program Administration(DAPA) in 2025(No.915102201).

## References

*   \[1\] Deblina Bhattacharjee, Tong Zhang, Sabine Süsstrunk, and Mathieu Salzmann. Mult: an end-to-end multitask learning transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12031–12041, 2022.
*   \[2\] Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 1385–1394, 2019.
*   \[3\] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. *arXiv preprint arXiv:2008.10292*, 2020a.
*   \[4\] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. *arXiv preprint arXiv:2008.10292*, 2020b.
*   \[5\] Ronghong Cai and Jianping Luo. Multi-task learning for multi-objective evolutionary neural architecture search. In *2021 IEEE Congress on Evolutionary Computation (CEC)*, pages 1680–1687. IEEE, 2021.
*   \[6\] Rich Caruana. Multitask learning. *Machine learning*, 28:41–75, 1997.
*   \[7\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International conference on machine learning*, pages 794–803. PMLR, 2018.
*   \[8\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *Advances in Neural Information Processing Systems*, 33:2039–2050, 2020.
*   \[9\] Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik G Learned-Miller, and Chuang Gan. Mod-squad: Designing mixtures of experts as modular multi-task learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11828–11837, 2023.
*   \[10\] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. *arXiv preprint arXiv:2009.09796*, 2020.
*   \[11\] Jifeng Dai, Kaiming He, and Jian Sun. Instance-aware semantic segmentation via multi-task network cascades. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3150–3158, 2016.
*   \[12\] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009.
*   \[13\] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5-6):313–318, 2012.
*   \[14\] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.
*   \[15\] Kshitij Dwivedi and Gemma Roig. Representation similarity analysis for efficient task taxonomy & transfer learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12387–12396, 2019.
*   \[16\] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In *Proceedings of the IEEE international conference on computer vision*, pages 2650–2658, 2015.
*   \[17\] Mark Everingham and John Winn. The pascal visual object classes challenge 2012 (voc2012) development kit. *Pattern Anal. Stat. Model. Comput. Learn., Tech. Rep*, 2007:1–45, 2012.
*   \[18\] Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang, et al. M3vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design. *Advances in Neural Information Processing Systems*, 35:28441–28457, 2022.
*   \[19\] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. *arXiv preprint arXiv:1701.08734*, 2017.
*   \[20\] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 3205–3214, 2019.
*   \[21\] Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pages 11543–11552, 2020.
*   \[22\] SHI Guangyuan, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. Recon: Reducing conflicting gradients from the root for multi-task learning. In *The Eleventh International Conference on Learning Representations*, 2022.
*   \[23\] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In *Proceedings of the European conference on computer vision (ECCV)*, pages 270–287, 2018.
*   \[24\] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International conference on machine learning*, pages 3854–3863. PMLR, 2020a.
*   \[25\] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International Conference on Machine Learning*, pages 3854–3863. PMLR, 2020b.
*   \[26\] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.
*   \[27\] Huimin Huang, Yawen Huang, Lanfen Lin, Ruofeng Tong, Yen-Wei Chen, Hao Zheng, Yuexiang Li, and Yefeng Zheng. Going beyond multi-task dense prediction with synergy embedding models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 28181–28190, 2024.
*   \[28\] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. *arXiv preprint arXiv:2103.02631*, 2021.
*   \[29\] Wooseong Jeong and Kuk-Jin Yoon. Quantifying task priority for multi-task optimization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 363–372, 2024.
*   \[30\] Wooseong Jeong and Kuk-Jin Yoon. Selective task group updates for multi-task optimization. *arXiv preprint arXiv:2502.11986*, 2025.
*   \[31\] Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, and Mingsheng Long. Forkmerge: Mitigating negative transfer in auxiliary-task learning. *Advances in Neural Information Processing Systems*, 36, 2024.
*   \[32\] Ian T Jollife and Jorge Cadima. Principal component analysis: A review and recent developments. *Philos. Trans. R. Soc. A Math. Phys. Eng. Sci*, 374(2065):20150202, 2016.
*   \[33\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 7482–7491, 2018.
*   \[34\] Jason Liang, Elliot Meyerson, and Risto Miikkulainen. Evolutionary architecture search for deep multitask networks. In *Proceedings of the genetic and evolutionary computation conference*, pages 466–473, 2018.
*   \[35\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34:18878–18890, 2021a.
*   \[36\] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. *Advances in Neural Information Processing Systems*, 36, 2024.
*   \[37\] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. iclr, 2021b.
*   \[38\] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 1871–1880, 2019.
*   \[39\] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021c.
*   \[40\] Yongxi Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, Tara Javidi, and Rogerio Feris. Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 5334–5343, 2017.
*   \[41\] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 1930–1939, 2018.
*   \[42\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1851–1860, 2019.
*   \[43\] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 891–898, 2014.
*   \[44\] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. *Advances in Neural Information Processing Systems*, 35:9564–9576, 2022.
*   \[45\] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. *arXiv preprint arXiv:2202.01017*, 2022.
*   \[46\] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? *Advances in neural information processing systems*, 33:512–523, 2020.
*   \[47\] Ramakanth Pasunuru and Mohit Bansal. Continual and multi-task architecture search. *arXiv preprint arXiv:1906.05226*, 2019.
*   \[48\] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34:8583–8595, 2021.
*   \[49\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *Advances in neural information processing systems*, 31, 2018.
*   \[50\] Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20083–20093, 2023.
*   \[51\] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12*, pages 746–760. Springer, 2012.
*   \[52\] Gianluigi Silvestri. One-shot neural architecture search for deep multi-task learning in computer vision, 2020.
*   \[53\] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.
*   \[54\] Ayan Sinha, Zhao Chen, Vijay Badrinarayanan, and Andrew Rabinovich. Gradient adversarial training of neural networks. 2018.
*   \[55\] Guolei Sun, Thomas Probst, Danda Pani Paudel, Nikola Popović, Menelaos Kanakis, Jagruti Patel, Dengxin Dai, and Luc Van Gool. Task switching network for multi-task learning. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 8291–8300, 2021.
*   \[56\] Simon Vandenhende, Stamatios Georgoulis, Bert De Brabandere, and Luc Van Gool. Branched multi-task networks: deciding what layers to share. *arXiv preprint arXiv:1904.02920*, 2019.
*   \[57\] Simon Vandenhende, Stamatios Georgoulis, and Luc Van Gool. Mti-net: Multi-scale task interaction networks for multi-task learning. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16*, pages 527–543. Springer, 2020.
*   \[58\] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 568–578, 2021a.
*   \[59\] Wenxiao Wang, Lu Yao, Long Chen, Binbin Lin, Deng Cai, Xiaofei He, and Wei Liu. Crossformer: A versatile vision transformer hinging on cross-scale attention. *arXiv preprint arXiv:2108.00154*, 2021b.
*   \[60\] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. *Advances in Neural Information Processing Systems*, 34:12077–12090, 2021.
*   \[61\] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 675–684, 2018.
*   \[62\] Xiaogang Xu, Hengshuang Zhao, Vibhav Vineet, Ser-Nam Lim, and Antonio Torralba. Mtformer: Multi-task learning via transformer and cross-task reasoning. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII*, pages 304–321. Springer, 2022.
*   \[63\] Yangyang Xu, Xiangtai Li, Haobo Yuan, Yibo Yang, and Lefei Zhang. Multi-task learning with multi-query transformer for dense prediction. *IEEE Transactions on Circuits and Systems for Video Technology*, 2023a.
*   \[64\] Yangyang Xu, Yibo Yang, and Lefei Zhang. Demt: Deformable mixer transformer for multi-task learning of dense prediction. In *Proceedings of the AAAI conference on artificial intelligence*, pages 3072–3080, 2023b.
*   \[65\] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao. Focal self-attention for local-global interactions in vision transformers. *arXiv preprint arXiv:2107.00641*, 2021.
*   \[66\] Hanrong Ye and Dan Xu. Inverted pyramid multi-task transformer for dense scene understanding. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII*, pages 514–530. Springer, 2022a.
*   \[67\] Hanrong Ye and Dan Xu. Taskprompter: Spatial-channel multi-task prompting for dense scene understanding. In *The Eleventh International Conference on Learning Representations*, 2022b.
*   \[68\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33:5824–5836, 2020.
*   \[69\] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3712–3722, 2018.
*   \[70\] Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. Mixture of attention heads: Selecting attention heads per token. *arXiv preprint arXiv:2210.05144*, 2022.
*   \[71\] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI 13*, pages 94–108. Springer, 2014.
*   \[72\] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe, and Jian Yang. Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4106–4115, 2019.

  
 

Supplementary Material  

## Appendix A Additional Related Works

Multi-Task Architectures. Various multi-task architectures can be categorized based on how the parameters or features of the sharing network are distributed among tasks. The widely used shared trunk structure comprises a common encoder shared by multiple tasks and a dedicated decoder for each task \[[11](#bib.bib11 ""), [41](#bib.bib41 ""), [53](#bib.bib53 ""), [71](#bib.bib71 "")\]. A tree-like architecture, with multiple division points for each task group, offers a more generalized structure \[[40](#bib.bib40 ""), [56](#bib.bib56 ""), [4](#bib.bib4 ""), [25](#bib.bib25 "")\]. The cross-talk architecture employs separate symmetrical networks for each task, utilizing feature exchange between layers at the same depth for information sharing between tasks \[[20](#bib.bib20 ""), [61](#bib.bib61 "")\]. The prediction distillation model \[[16](#bib.bib16 ""), [61](#bib.bib61 ""), [57](#bib.bib57 ""), [72](#bib.bib72 "")\] incorporates cross-task interactions at the end of the shared encoder, while the task switching network \[[55](#bib.bib55 ""), [54](#bib.bib54 ""), [19](#bib.bib19 ""), [42](#bib.bib42 "")\] changes network parameters depending on the task.

## Appendix B Experimental Settings

### B.1 Datasets

We evaluate our method on three benchmarks: NYUD-v2, PASCAL-Context, and Taskonomy. NYUD-v2 contains 4 vision tasks: Our evaluation is based on depth estimation, semantic segmentation, surface normal prediction, and edge detection. PASCAL-Context contains 5 tasks: We evaluate semantic segmentation, human parts estimation, saliency estimation, surface normal prediction, and edge detection. We used 11 tasks for Taskonomy: We evaluate Depth Euclidean (DE), Depth Zbuffer (DZ), Edge Texture (ET), Keypoints 2D (Key2D), Keypoints 3D (Key3D), Normal (N), Principal Curvature (PC), Reshading (R), Segment Unsup 2d (S2D), and Segment Unsup 2.5D (S25D).

### B.2 Implementation Details

For experiments, we adopt ViT \[[14](#bib.bib14 "")\] pre-trained on ImageNet-22K \[[12](#bib.bib12 "")\] as the transformer encoder. The models are trained for 60,000 iterations on both NYUD \[[51](#bib.bib51 "")\] and PASCAL \[[17](#bib.bib17 "")\] datasets with batch size 6. We use Adam optimizer with learning rate 2×2\\times10−510^{-5} and 1×1\\times10−610^{-6} of a weight decay with a polynomial learning rate schedule. Following the previous works \[[66](#bib.bib66 ""), [67](#bib.bib67 "")\], the cross-entropy loss is used for semantic segmentation, human parts estimation, and saliency, edge detection. Surface normal prediction and depth estimation use L1 loss.

### B.3 Design and Implementation Strategy

To improve efficiency, we perform SVD only once early in training to estimate the feature space for conflict analysis. Gradient conflicts are measured in a pairwise manner across tasks, and the average number of conflicts in each space is used to guide token expansion. Based on this, we statically allocate a small number of task-specific tokens (six in our setup) as learnable parameters, independently applied at each layer. These tokens are fixed during training and do not adapt dynamically. For NYUD-v2 and PASCAL-Context, we use the full training sets to compute gradient statistics, while for Taskonomy, covariance is estimated using 100 randomly sampled mini-batches. The assignment of Token Modulation (TM) and Token Expansion (TE) is determined by a manually chosen activation ratio, which we analyze in [Fig. 6](#A3.F6 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"). Rather than activating all components uniformly, TM and TE are selectively applied to layers with the highest conflict levels, either individually or jointly, based on their effectiveness in reducing task interference.

### B.4 Evaluation

For semantic segmentation, we utilize mean Intersection over Union (mIoU). Surface normal prediction performance is measured by the mean angular distance between the predicted output and ground truth. Depth estimation is evaluated using Root Mean Squared Error (RMSE). For saliency estimation and human part segmentation, we employ mIoU. Edge detection is assessed using the optimal-dataset-scale F-measure (odsF). For Taskonomy, we adopt RMSE for principal curvature and L1 distance for the remaining tasks.

Table 8: Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-L. Non-converged results are indicated with a dash.

| Task   | DE     | DZ     | EO     | ET     | Key2D  | Key3D  | N      | PC     | R      | S2D    | S25D   |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  |
| 0.0153 | 0.0156 | 0.1196 | 0.1757 | 0.1729 | 0.0896 | 0.2215 | 0.7451 | 0.1576 | 0.1826 | 0.1537 | -8.92  |  |
| 0.0170 | 0.0195 | 0.1235 | 0.1757 | 0.1753 | 0.0909 | 0.2818 | 0.7679 | 0.1663 | 0.1916 | 0.1543 | -17.07 |  |
| -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      |  |
| 0.0152 | 0.0155 | 0.1195 | 0.1755 | 0.1728 | 0.0897 | 0.2356 | 0.7436 | 0.1569 | 0.1830 | 0.1538 | -9.36  |  |
| 0.0153 | 0.0156 | 0.1197 | 0.1757 | 0.1730 | 0.0897 | 0.2214 | 0.7441 | 0.1576 | 0.1827 | 0.1537 | -8.96  |  |
| 0.0152 | 0.0156 | 0.1192 | 0.1749 | 0.1699 | 0.0893 | 0.2310 | 0.7475 | 0.1577 | 0.1825 | 0.1480 | -8.63  |  |
| 0.0155 | 0.0156 | 0.1175 | 0.1756 | 0.1649 | 0.0860 | 0.2421 | 0.7544 | 0.1591 | 0.1854 | 0.1554 | -9.32  |  |
| 0.0151 | 0.0156 | 0.1194 | 0.1755 | 0.1726 | 0.0895 | 0.2199 | 0.7432 | 0.1569 | 0.1824 | 0.1533 | -8.57  |  |
| 0.0150 | 0.0155 | 0.1136 | 0.1733 | 0.1633 | 0.0862 | 0.2512 | 0.8029 | 0.1643 | 0.1803 | 0.1445 | -8.78  |  |
| 0.0151 | 0.0154 | 0.1138 | 0.1732 | 0.1644 | 0.0863 | 0.2507 | 0.7656 | 0.1544 | 0.1833 | 0.1452 | -7.95  |  |
| 0.0153 | 0.0157 | 0.1196 | 0.1757 | 0.1730 | 0.0897 | 0.2221 | 0.7444 | 0.1575 | 0.1830 | 0.1534 | -8.99  |  |
| 0.0127 | 0.0130 | 0.1088 | 0.1731 | 0.1665 | 0.0852 | 0.1654 | 0.6890 | 0.1389 | 0.1661 | 0.1404 | +2.41  |  |

Table 9: Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-S. Non-converged results are indicated with a dash.

| Task   | DE     | DZ     | EO     | ET     | Key2D  | Key3D  | N      | PC     | R      | S2D    | S25D   |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  |
| 0.0244 | 0.0243 | 0.1501 | 0.1778 | 0.1844 | 0.1009 | 0.4105 | 0.9087 | 0.2325 | 0.2032 | 0.1822 | -8.04  |  |
| 0.0253 | 0.0253 | 0.1533 | 0.1785 | 0.1865 | 0.1021 | 0.4399 | 0.9246 | 0.2408 | 0.2063 | 0.1791 | -10.42 |  |
| -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      |  |
| 0.0242 | 0.0242 | 0.1498 | 0.1778 | 0.1847 | 0.1007 | 0.4064 | 0.9079 | 0.2312 | 0.2033 | 0.1822 | -7.74  |  |
| 0.0242 | 0.0242 | 0.1500 | 0.1778 | 0.1844 | 0.1008 | 0.4097 | 0.9071 | 0.2316 | 0.2032 | 0.1822 | -7.84  |  |
| 0.0248 | 0.0248 | 0.1501 | 0.1755 | 0.1761 | 0.1001 | 0.4306 | 0.9181 | 0.2371 | 0.2023 | 0.1772 | -8.12  |  |
| 0.0254 | 0.0255 | 0.1516 | 0.1738 | 0.1698 | 0.0983 | 0.4535 | 0.9282 | 0.2442 | 0.2068 | 0.1849 | -9.74  |  |
| 0.0236 | 0.0237 | 0.1456 | 0.1756 | 0.1760 | 0.0988 | 0.4151 | 0.9055 | 0.2222 | 0.2010 | 0.1794 | -5.74  |  |
| 0.0266 | 0.0264 | 0.1499 | 0.1736 | 0.1700 | 0.0986 | 0.4659 | 0.9868 | 0.2604 | 0.2030 | 0.1780 | -11.51 |  |
| 0.0235 | 0.0235 | 0.1432 | 0.1745 | 0.1718 | 0.0975 | 0.4230 | 0.9225 | 0.2268 | 0.1985 | 0.1775 | -5.41  |  |
| 0.0243 | 0.0243 | 0.1499 | 0.1778 | 0.1846 | 0.1008 | 0.3841 | 0.9080 | 0.2321 | 0.2027 | 0.1816 | -7.31  |  |
| 0.0196 | 0.0200 | 0.1372 | 0.1754 | 0.1712 | 0.0958 | 0.3129 | 0.8333 | 0.1955 | 0.1907 | 0.1698 | +3.62  |  |

Table 10: Comparison with multi-task optimization approaches on Taskonomy across 11 different tasks with ViT-T. Non-converged results are indicated with a dash.

| Task   | DE     | DZ     | EO     | ET     | Key2D  | Key3D  | N      | PC     | R      | S2D    | S25D   |  |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  |
| 0.0266 | 0.0278 | 0.1593 | 0.1794 | 0.1865 | 0.1047 | 0.4752 | 0.9467 | 0.2568 | 0.2081 | 0.1897 | -11.10 |  |
| 0.0276 | 0.0284 | 0.1624 | 0.1807 | 0.1884 | 0.1064 | 0.4741 | 0.9611 | 0.2658 | 0.2108 | 0.1860 | -12.67 |  |
| -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      |  |
| 0.0266 | 0.0277 | 0.1593 | 0.1795 | 0.1865 | 0.1045 | 0.4757 | 0.9466 | 0.2567 | 0.2080 | 0.1896 | -11.07 |  |
| 0.0266 | 0.0274 | 0.1593 | 0.1794 | 0.1866 | 0.1045 | 0.4743 | 0.9465 | 0.2567 | 0.2080 | 0.1897 | -10.95 |  |
| 0.0273 | 0.0285 | 0.1596 | 0.1768 | 0.1807 | 0.1043 | 0.4785 | 0.9689 | 0.2644 | 0.2080 | 0.1854 | -11.55 |  |
| 0.0290 | 0.0305 | 0.1641 | 0.1747 | 0.1731 | 0.1051 | 0.4884 | 0.9870 | 0.2828 | 0.2136 | 0.1945 | -14.64 |  |
| 0.0263 | 0.0272 | 0.1558 | 0.1772 | 0.1810 | 0.1025 | 0.4730 | 0.9525 | 0.2458 | 0.2065 | 0.1868 | -9.24  |  |
| -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      | -      |  |
| 0.0261 | 0.0270 | 0.1536 | 0.1762 | 0.1766 | 0.1017 | 0.4590 | 0.9649 | 0.2496 | 0.2039 | 0.1846 | -8.28  |  |
| 0.0266 | 0.0275 | 0.1592 | 0.1795 | 0.1865 | 0.1047 | 0.4746 | 0.9466 | 0.2566 | 0.2080 | 0.1898 | -10.97 |  |
| 0.0236 | 0.0241 | 0.1494 | 0.1765 | 0.1790 | 0.0998 | 0.4138 | 0.8921 | 0.2290 | 0.1959 | 0.1824 | -2.88  |  |

Table 11: We compare task performance based on the configuration of the modulator. Specifically, we compare the performance of tasks using an affine transformation against those using a batch normalization layer as configurations for the modulator.

|                 | NYUD-v2           | PASCAL-Context    |                 |                 |                 |                 |                   |                 |
| --------------- | ----------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | ----------------- | --------------- |
| Semseg          | Depth             | Normal            | Edge            | Semseg          | Parsing         | Saliency        | Normal            | Edge            |
| mIoU ↑\\uparrow | RMSE ↓\\downarrow | mErr ↓\\downarrow | odsF ↑\\uparrow | mIoU ↑\\uparrow | mIoU ↑\\uparrow | maxF ↑\\uparrow | mErr ↓\\downarrow | odsF ↑\\uparrow |
| 38.27           | 0.6370            | 21.64             | 57.90           | 66.18           | 56.29           | 83.21           | 15.26             | 47.00           |
| 37.42           | 0.6550            | 23.16             | 56.10           | 60.80           | 53.29           | 82.59           | 15.73             | 44.90           |

## Appendix C Additional Experiments

Comparison with Multi-Task Optimization. In [Tabs. 8](#A2.T8 "In B.4 Evaluation ‣ Appendix B Experimental Settings ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), [9](#A2.T9 "Table 9 ‣ B.4 Evaluation ‣ Appendix B Experimental Settings ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") and [10](#A2.T10 "Table 10 ‣ B.4 Evaluation ‣ Appendix B Experimental Settings ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we further evaluate the proposed DTME-MTL against previous multi-task optimization approaches using different backbone sizes. Our method demonstrates significant improvements in multi-task performance with minimal increases in parameters. Specifically, DTME-MTL results in a parameter increase of 0.089%0.089\\% for ViT-L, 0.23%0.23\\% for ViT-S, and 0.46%0.46\\% for ViT-T.

Table 12: We assess task performance by comparing scenarios where we freeze the backbone network after expansion (w/ Freeze) and where we don’t (w/o Freeze).

|                 | NYUD-v2           | PASCAL-Context    |                 |                 |                 |                 |                   |                 |
| --------------- | ----------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | ----------------- | --------------- |
| Semseg          | Depth             | Normal            | Edge            | Semseg          | Parsing         | Saliency        | Normal            | Edge            |
| mIoU ↑\\uparrow | RMSE ↓\\downarrow | mErr ↓\\downarrow | odsF ↑\\uparrow | mIoU ↑\\uparrow | mIoU ↑\\uparrow | maxF ↑\\uparrow | mErr ↓\\downarrow | odsF ↑\\uparrow |
| 34.80           | 0.6730            | 22.48             | 56.00           | 58.34           | 52.96           | 82.86           | 15.63             | 43.20           |
| 38.27           | 0.6370            | 21.64             | 57.90           | 66.18           | 56.29           | 83.21           | 15.26             | 47.00           |

![Refer to caption](2507.07485v2/figure/svd_r_nyud.png)

(a) Results on NYUD-v2.

![Refer to caption](2507.07485v2/figure/svd_r_pascal.png)

(b) Results on PASCAL-Context.

Figure 5: We assess the performance of tasks based on the proportion of total variance rr. The results are displayed for both (a) NYUD-v2 and (b) PASCAL-Context.

![Refer to caption](2507.07485v2/figure/prop_nyud.png)

(a) Results on NYUD-v2.

![Refer to caption](2507.07485v2/figure/prop_pascal.png)

(b) Results on PASCAL-Context.

Figure 6: The performance of tasks based on the ratio of the number of expanded layers to the total number of layers. The results are displayed for both (a) NYUD-v2 and (b) PASCAL-Context.

Analysis on the Modulator Configuration. In [Tab. 11](#A2.T11 "In B.4 Evaluation ‣ Appendix B Experimental Settings ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we show the performance difference based on the configuration of the token modulators. Specifically, we compared the outcomes obtained when employing affine transformation and batch normalization, which could be considered as the most common and straightforward approaches. Through experiments, we find that affine transformations consistently exhibit better performance across all tasks compared to batch normalization layers used as modulators for both datasets.

Analyzing Performance Differences with Backbone Network Freezing. In [Tab. 12](#A3.T12 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we examine the performance variation based on whether we freeze the existing backbone network components when training the expanded network after implementing the proposed dynamic token modulation and expansion. The results indicate that training networks without freezing the existing backbone network components leads to significantly better performance compared to training networks with freezing. We guess that allowing modifications to the learned token space after expansion helps the network to dynamically partition the token space for each task.

Influence of r on SVD Approximation. In [Fig. 5](#A3.F5 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we illustrate how the proportion of total variance rr impacts the approximation of a token’s range and null space. We assess the performance of tasks across five values of rr (1, 10, 100, 500, 1000). Our results suggest that the value of rr has minimal impact on task performance, implying that there is less need for extensive tuning of the rr parameter to optimize performance. In our other experiments, we chose rr as 100 for training.

Impact of the Number of Layers Expanded by DTME-MTL. DTME-MTL expands a subset of transformer layers selected based on the severity of gradient conflicts. In [Fig. 6](#A3.F6 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we analyze how varying the number of expanded layers affects task performance. The x-axis denotes the ratio of expanded layers to the total number of layers. We observe that applying TM+TE to approximately 25%–50% of the layers yields consistent performance gains across tasks while maintaining parameter efficiency. Performance improves as more high-conflict layers are expanded, but begins to degrade when expansion exceeds 50%, especially when low-conflict layers are included. This suggests that over-expansion can be detrimental. [Table 2](#S4.T2 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") further confirms that using a moderate expansion ratio (50%) avoids overfitting, whereas [Fig. 6](#A3.F6 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") highlights that indiscriminate expansion into less conflicting layers harms performance. These findings underscore the importance of both the extent and location of TM+TE application.

Effect of Swapping Conflict Types. In [Tab. 13](#A3.T13 "In Appendix C Additional Experiments ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), we present the results of an experiment on NYUD-v2 where we intentionally swap the conflict types targeted by each method. Specifically, Token Expansion (TE) is applied to layers with severe range space conflict, and Token Modulation (TM) is applied to layers with severe null space conflict—opposite to our standard configuration. This reversal leads to a clear performance drop, confirming that each method is most effective when applied to the type of conflict it is designed to resolve. These results support our design choice of assigning TM to range space conflict and TE to null space conflict.

Table 13: Performance comparison across selection strategies.

| Method |        |        |        |
| ------ | ------ | ------ | ------ |
| -2.966 | -6.167 | -2.608 | +0.044 |

## Appendix D Additional Analysis

Further Justification for Targeted TM/TE Assignment. Prior work \[[46](#bib.bib46 "")\] suggests that fine-tuning from a pretrained model tends to remain in the same loss basin, preserving the structure of the pretrained feature space. Accordingly, we view the token space during fine-tuning as constrained by the span of the pretrained features. If the conflict lies within this span (i.e., the range space), it can be resolved by rotating the token space—achievable via a modulator, since affine transformations include rotation. However, if the conflict resides in the null space, it lies outside the span and cannot be sufficiently addressed by modulation alone. In such cases, expanding the token space with task-specific tokens helps relax this constraint. We theoretically support this view in Propositions [1](#Thmproposition1 "Proposition 1. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") and [2](#Thmproposition2 "Proposition 2. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") (with proofs in [Appendix E](#A5 "Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")), which analyze how each method addresses conflict in its respective subspace. This is further validated empirically: we measure the reduction in gradient conflicts by comparing the start and end of training in each space ([Tab. 14](#A4.T14 "In Appendix D Additional Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")). The results show that Token Modulation (TM) is more effective in reducing conflicts in the range space, while Token Expansion (TE) is more effective in the null space. This consistency between theoretical analysis and empirical behavior supports our design choice to selectively apply TM and TE based on the dominant type of conflict in each layer.

Token-Level vs. Parameter-Level Conflict Handling. Parameter-space conflicts reflect an aggregate gradient across all tokens, which makes it difficult to localize or disentangle the source of interference. In contrast, token-level conflicts can be measured for each individual token, allowing more localized and fine-grained analysis. This granularity enables our method to selectively modulate or expand tokens based on where the conflict occurs. Furthermore, by decomposing the token space into range and null components—depending on whether the pretrained model already spans those directions—we adaptively apply Token Modulation (TM) or Token Expansion (TE) to address conflicts. Such space-aware conflict resolution is fundamentally infeasible in parameter space, where task interference is entangled across layers and tokens.

Comparison with LoRA in Multi-Task Inference. As shown in [Tab. 1](#S4.T1 "In 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"), the baseline (ST) corresponds to full fine-tuning and serves as an upper bound on performance. While LoRA \[[26](#bib.bib26 "")\] is a parameter-efficient method, assigning a separate LoRA module for each task leads to disjoint sets of task-specific weights. Even when merged into the base model, these configurations require separate forward passes per task, negating the efficiency benefits of multi-task learning (MTL). In contrast, our method maintains shared weights across tasks and allows all outputs to be computed jointly in a single batched tensor operation on GPU. This enables highly parallel inference with only a 13.4% overhead per task, whereas the inference time in LoRA scales linearly with the number of tasks.

Table 14: Reduction in gradient conflict numbers (NYUD-v2).

| Method                 | Num(gℛ,i⋅gℛ,i≤0g\_{\\mathcal{R},i}\\cdot g\_{\\mathcal{R},i}\\leq 0) | Num(g𝒩,i⋅g𝒩,i≤0g\_{\\mathcal{N},i}\\cdot g\_{\\mathcal{N},i}\\leq 0) |
| ---------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 11.60 %↓\\%\\downarrow | 8.92 %↓\\%\\downarrow                                                |                                                                      |
| 4.64%↓\\%\\downarrow   | 15.44 %↓\\%\\downarrow                                               |                                                                      |

## Appendix E Theoretical Analysis

### E.1 Proof of [Proposition 1](#Thmproposition1 "Proposition 1. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")

See [1](#Thmproposition1 "Proposition 1. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")

###### Proof.

Let the loss function ℒi\\mathcal{L}\_{i} be a function of the shared parameters Θs\\Theta\_{s}, the token modulator ℳi\\mathcal{M}\_{i}, and the input data 𝒳l\\mathcal{X}\_{l}. Since transformers convert input data into tokens, we consider the loss to be a function of one of the input tokens, 𝒯i​n\\mathcal{T}\_{in}, rather than 𝒳l\\mathcal{X}^{l}. To represent the updating step during optimization, we use the superscript tt for current variables, such as Θst\\Theta\_{s}^{t}, and ℳit\\mathcal{M}\_{i}^{t}, and t+1t+1 for the next-step variables, such as Θst+1\\Theta\_{s}^{t+1}, and ℳit+1\\mathcal{M}\_{i}^{t+1}.

In cases where the input token 𝒯i​n\\mathcal{T}\_{in} spans the row space of 𝒯s\~\\tilde{\\mathcal{T}\_{s}}, this can be expressed as follows:

𝒰𝒩​𝒰𝒩T​∇𝒯i​nℒi​(Θst,ℳit,𝒯i​n)≃0\\displaystyle\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T}\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{in})\\simeq 0

(5)

Since the row space and null space are perpendicular to each other, with their dimensions summing to the entire space, the following holds according to [Eq. 5](#A5.E5 "In Proof. ‣ E.1 Proof of ‣ Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning"):

∑i\=1𝒦∇𝒯i​nℒi\=∑i\=1𝒦(𝒰ℛ​𝒰ℛT+𝒰𝒩​𝒰𝒩T)​∇𝒯i​nℒi≃∑i\=1𝒦(𝒰ℛ​𝒰ℛT)​∇𝒯i​nℒi\\displaystyle\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{L}\_{i}=\\sum\_{i=1}^{\\mathcal{K}}(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T}+\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{L}\_{i}\\simeq\\sum\_{i=1}^{\\mathcal{K}}(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T})\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{L}\_{i}

(6)

Let the token modulator ℳi\\mathcal{M}\_{i} be a p×pp\\times p matrix that manipulates the input token 𝒯i​n\\mathcal{T}\_{in}.

∑i\=1𝒦∇𝒯i​nℒi\\displaystyle\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{L}\_{i}

\=∑i\=1𝒦(𝒰ℛ​ℳit)​(𝒰ℛ​ℳit)T⋅∇ℳitℒi⋅∇𝒯i​nℳit\\displaystyle=\\sum\_{i=1}^{\\mathcal{K}}(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{M}\_{i}^{t})(\\mathcal{U}\_{\\mathcal{R}}\\mathcal{M}\_{i}^{t})^{T}\\cdot\\nabla\_{\\mathcal{M}\_{i}^{t}}\\mathcal{L}\_{i}\\cdot\\nabla\_{\\mathcal{T}\_{in}}\\mathcal{M}\_{i}^{t}

(7)

The total multi-task loss can be represented using a Taylor expansion. Assuming η≪1\\eta\\ll 1, we can ignore the second-order terms of η\\eta:

∑i\=1𝒦ℒi​(Θst+1,ℳit+1,𝒯s)\=∑i\=1𝒦ℒi​(Θst,ℳit,𝒯s)+∑i\=1𝒦∇Θstℒi​(Θst,ℳit,𝒯s)​(Θst+1−Θst)\\displaystyle\\sum\_{i=1}^{\\mathcal{K}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t+1},\\mathcal{M}\_{i}^{t+1},\\mathcal{T}\_{s})=\\sum\_{i=1}^{\\mathcal{K}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})+\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\Theta\_{s}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})(\\Theta\_{s}^{t+1}-\\Theta\_{s}^{t})

(8)

+∑i\=1𝒦∇ℳitℒi(Θst,ℳit,𝒯s)(ℳit+1−ℳit)\\displaystyle+\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\mathcal{M}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})(\\mathcal{M}\_{i}^{t+1}-\\mathcal{M}\_{i}^{t})

(9)

\=∑i\=1𝒦ℒi​(Θst,ℳit,𝒯s)−η​|∑i\=1𝒦∇Θsℒi​(Θst,ℳit,𝒯s)|2\\displaystyle=\\sum\_{i=1}^{\\mathcal{K}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})-\\eta|\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\Theta\_{s}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})|^{2}

(10)

−η∑i\=1𝒦|∇ℳitℒi(Θst,ℳit,𝒯s)|2\\displaystyle-\\eta\\sum\_{i=1}^{\\mathcal{K}}|\\nabla\_{\\mathcal{M}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{s})|^{2}

(11)

By optimizing the modulator ℳit\\mathcal{M}\_{i}^{t} so that |∇ℳitℒi​(Θst,ℳit,𝒯i​n)||\\nabla\_{\\mathcal{M}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{M}\_{i}^{t},\\mathcal{T}\_{in})| approaches zero for each task i\=1,2,…,𝒦i=1,2,\\dots,\\mathcal{K}, we can alleviate gradient conflicts in the row space of 𝒯\~s\\tilde{\\mathcal{T}}\_{s} (as [Eq. 7](#A5.E7 "In Proof. ‣ E.1 Proof of ‣ Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") also approaches zero) and reduce the overall multi-task loss, since [Eq. 11](#A5.E11 "In Proof. ‣ E.1 Proof of ‣ Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning") is always greater than or equal to zero. ∎

### E.2 Proof of [Proposition 2](#Thmproposition2 "Proposition 2. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")

See [2](#Thmproposition2 "Proposition 2. ‣ 4.3 Token Modulation and Expansion ‣ 4 Method ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")

###### Proof.

Let the loss function ℒi\\mathcal{L}\_{i} be a function of the shared parameters Θst\\Theta\_{s}^{t}, the task-specific token 𝒯it\\mathcal{T}\_{i}^{t}, and the input data 𝒳t\\mathcal{X}^{t}. Similarly, since transformers convert input data into tokens, we consider the loss to be a function of one of the input tokens, 𝒯i​nt\\mathcal{T}\_{in}^{t}, rather than 𝒳t\\mathcal{X}^{t}. To represent the updating step during optimization, we use the superscript tt for current variables, such as Θst\\Theta\_{s}^{t}, 𝒯i​nt\\mathcal{T}\_{in}^{t} and ℳit\\mathcal{M}\_{i}^{t}, and t+1t+1 for the next-step variables, such as Θst+1\\Theta\_{s}^{t+1}, 𝒯i​nt+1\\mathcal{T}\_{in}^{t+1} and ℳit+1\\mathcal{M}\_{i}^{t+1}.

In the case where the input token 𝒯i​nt\\mathcal{T}\_{in}^{t} spans the null space of 𝒯s\~\\tilde{\\mathcal{T}\_{s}}, this can be expressed as follows:

∑i\=1𝒦𝒰ℛ​𝒰ℛT​∇𝒯i​ntℒi​(Θst,𝒯i​nt,𝒯it)≃0\\displaystyle\\sum\_{i=1}^{\\mathcal{K}}\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T}\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})\\simeq 0

(12)

The derivative of the task-specific loss ℒi\\mathcal{L}\_{i} with respect to the expanded token, including the input token 𝒯i​nt\\mathcal{T}\_{in}^{t} and the learnable task-specific tokens 𝒯it\\mathcal{T}\_{i}^{t}, is given as follows:

∑i\=1𝒦\\displaystyle\\sum\_{i=1}^{\\mathcal{K}}

∇\[𝒯i​nt,𝒯it\]ℒi\\displaystyle\\nabla\_{\[\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t}\]}\\mathcal{L}\_{i}

(13)

\=∑i\=1𝒦(\[𝒰ℛ0d×𝒦0𝒦×d𝒰ℛ,i\]​\[𝒰ℛ0d×𝒦0𝒦×d𝒰ℛ,i\]T+\[𝒰𝒩0d×𝒦0𝒦×d0𝒦×𝒦\]​\[𝒰𝒩0d×𝒦0𝒦×d0𝒦×𝒦\]T)​\[∇𝒯i​ntℒi∇𝒯itℒi\]\\displaystyle=\\sum\_{i=1}^{\\mathcal{K}}\\Bigg(\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{R}}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&\\mathcal{U}\_{\\mathcal{R},i}\\\\ \\end{bmatrix}\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{R}}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&\\mathcal{U}\_{\\mathcal{R},i}\\\\ \\end{bmatrix}^{T}+\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{N}}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&0\_{\\mathcal{K}\\times\\mathcal{K}}\\\\ \\end{bmatrix}\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{N}}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&0\_{\\mathcal{K}\\times\\mathcal{K}}\\\\ \\end{bmatrix}^{T}\\Bigg)\\begin{bmatrix}\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}\\\\ \\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}\\\\ \\end{bmatrix}

(14)

\=∑i\=1𝒦\[𝒰ℛ​𝒰ℛT+𝒰𝒩​𝒰𝒩T0d×𝒦0𝒦×d𝒰ℛ,i​𝒰ℛ,iT\]​\[∇𝒯i​ntℒi∇𝒯itℒi\]\\displaystyle=\\sum\_{i=1}^{\\mathcal{K}}\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{R}}\\mathcal{U}\_{\\mathcal{R}}^{T}+\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&\\mathcal{U}\_{\\mathcal{R},i}\\mathcal{U}\_{\\mathcal{R},i}^{T}\\\\ \\end{bmatrix}\\begin{bmatrix}\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}\\\\ \\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}\\\\ \\end{bmatrix}

(15)

≃∑i\=1𝒦\[𝒰𝒩​𝒰𝒩T0d×𝒦0𝒦×d𝒰ℛ,i​𝒰ℛ,iT\]​\[∇𝒯i​ntℒi∇𝒯itℒi\]\\displaystyle\\simeq\\sum\_{i=1}^{\\mathcal{K}}\\begin{bmatrix}\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T}&0\_{d\\times\\mathcal{K}}\\\\ 0\_{\\mathcal{K}\\times d}&\\mathcal{U}\_{\\mathcal{R},i}\\mathcal{U}\_{\\mathcal{R},i}^{T}\\\\ \\end{bmatrix}\\begin{bmatrix}\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}\\\\ \\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}\\\\ \\end{bmatrix}

(16)

\=∑i\=1𝒦\[(𝒰𝒩​𝒰𝒩T)​∇𝒯i​ntℒi(𝒰ℛ,i​𝒰ℛ,iT)​∇𝒯itℒi\]\\displaystyle=\\sum\_{i=1}^{\\mathcal{K}}\\begin{bmatrix}(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}\\\\ (\\mathcal{U}\_{\\mathcal{R},i}\\mathcal{U}\_{\\mathcal{R},i}^{T})\\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}\\\\ \\end{bmatrix}

(17)

The total multi-task loss can be expressed as follows:

ℒi​(Θst+1,𝒯i​nt+1,𝒯it+1)\=ℒi​(Θi​nt,𝒯st,𝒯it)+∇Θstℒi​(Θst,𝒯st,𝒯it)​(Θst+1−Θst)\\displaystyle\\mathcal{L}\_{i}(\\Theta\_{s}^{t+1},\\mathcal{T}\_{in}^{t+1},\\mathcal{T}\_{i}^{t+1})=\\mathcal{L}\_{i}(\\Theta\_{in}^{t},\\mathcal{T}\_{s}^{t},\\mathcal{T}\_{i}^{t})+\\nabla\_{\\Theta\_{s}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{s}^{t},\\mathcal{T}\_{i}^{t})(\\Theta\_{s}^{t+1}-\\Theta\_{s}^{t})

(18)

+∇𝒯i​ntℒi​(Θst,𝒯st,𝒯it)​(𝒯i​nt+1−𝒯i​nt)\\displaystyle+\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{s}^{t},\\mathcal{T}\_{i}^{t})(\\mathcal{T}\_{in}^{t+1}-\\mathcal{T}\_{in}^{t})

(19)

+∇𝒯itℒi​(Θst,𝒯st,𝒯it)​(𝒯it+1−𝒯it)\\displaystyle+\\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{s}^{t},\\mathcal{T}\_{i}^{t})(\\mathcal{T}\_{i}^{t+1}-\\mathcal{T}\_{i}^{t})

(20)

\=ℒi​(Θst,𝒯i​nt,𝒯it)−η​∇Θstℒi​(Θst,𝒯st,𝒯it)⋅∑i\=1𝒦∇Θstℒi​(Θst,𝒯i​nt,𝒯it)\\displaystyle=\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})-\\eta\\nabla\_{\\Theta\_{s}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{s}^{t},\\mathcal{T}\_{i}^{t})\\cdot\\sum\_{i=1}^{\\mathcal{K}}\\nabla\_{\\Theta\_{s}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})

(21)

−η(𝒰𝒩𝒰𝒩T)∇𝒯i​ntℒi(Θst,𝒯i​nt,𝒯it)⋅∑i\=1𝒦(𝒰𝒩𝒰𝒩T)∇𝒯i​ntℒi(Θst,𝒯i​nt,𝒯it)\\displaystyle-\\eta(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})\\cdot\\sum\_{i=1}^{\\mathcal{K}}(\\mathcal{U}\_{\\mathcal{N}}\\mathcal{U}\_{\\mathcal{N}}^{T})\\nabla\_{\\mathcal{T}\_{in}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})

(22)

−η(𝒰ℛ,i𝒰ℛ,iT)∇𝒯itℒi(Θst,𝒯i​nt,𝒯it)⋅(𝒰ℛ,i𝒰ℛ,iT)∇𝒯itℒi(Θst,𝒯i​nt,𝒯it)\\displaystyle-\\eta(\\mathcal{U}\_{\\mathcal{R},i}\\mathcal{U}\_{\\mathcal{R},i}^{T})\\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})\\cdot(\\mathcal{U}\_{\\mathcal{R},i}\\mathcal{U}\_{\\mathcal{R},i}^{T})\\nabla\_{\\mathcal{T}\_{i}^{t}}\\mathcal{L}\_{i}(\\Theta\_{s}^{t},\\mathcal{T}\_{in}^{t},\\mathcal{T}\_{i}^{t})

(23)

The increase in multi-task loss caused by gradient conflicts in the null space (as described in [Eq. 22](#A5.E22 "In Proof. ‣ E.2 Proof of ‣ Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning")) cannot be reduced since the shared token 𝒯i​nt\\mathcal{T}\_{in}^{t} is not a learnable parameter. Instead, task-specific tokens 𝒯it\\mathcal{T}\_{i}^{t} can be added to mitigate the increase in multi-task loss due to null space gradient conflicts by optimizing the learnable parameters {𝒯i}i\=1𝒦\\{\\mathcal{T}\_{i}\\}\_{i=1}^{\\mathcal{K}} as described in [Eq. 23](#A5.E23 "In Proof. ‣ E.2 Proof of ‣ Appendix E Theoretical Analysis ‣ Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning").

∎

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")