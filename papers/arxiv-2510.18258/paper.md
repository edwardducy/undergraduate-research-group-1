# NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective

 Xiaohan Qin    Xiaoxing Wang    Ning Liao    Junchi Yan‡ Affiliation: School of CS & School of AI, Shanghai Jiao Tong University Affiliation: Shanghai Innovation Institute 

###### Abstract

Multi-Task Learning (MTL) enables a single model to learn multiple tasks simultaneously, leveraging knowledge transfer among tasks for enhanced generalization, and has been widely applied across various domains. However, task imbalance remains a major challenge in MTL. Although balancing the convergence speeds of different tasks is an effective approach to address this issue, it is highly challenging to accurately characterize the training dynamics and convergence speeds of multiple tasks within the complex MTL system. To this end, we attempt to analyze the training dynamics in MTL by leveraging Neural Tangent Kernel (NTK) theory and propose a new MTL method, NTKMTL. Specifically, we introduce an extended NTK matrix for MTL and adopt spectral analysis to balance the convergence speeds of multiple tasks, thereby mitigating task imbalance. Based on the approximation via shared representation, we further propose NTKMTL-SR, achieving training efficiency while maintaining competitive performance. Extensive experiments demonstrate that our methods achieve state-of-the-art performance across a wide range of benchmarks, including both multi-task supervised learning and multi-task reinforcement learning. Source code is available at [https://github.com/jianke0604/NTKMTL](https://github.com/jianke0604/NTKMTL "").

## 1 Introduction

Multi-task learning (MTL) \[[9](#bib.bib9 ""), [20](#bib.bib20 ""), [56](#bib.bib56 ""), [45](#bib.bib45 ""), [61](#bib.bib61 "")\] involves training a single model to address multiple tasks concurrently. This approach enables sharing information and representations across tasks, enhancing the model’s generalization capabilities and boosting performance on individual tasks \[[6](#bib.bib6 ""), [50](#bib.bib50 ""), [38](#bib.bib38 "")\]. MTL is particularly advantageous in scenarios with limited computational resources, as it eliminates the need to maintain separate models for each task. Its utility spans a wide range of domains, including computer vision \[[1](#bib.bib1 ""), [63](#bib.bib63 ""), [35](#bib.bib35 "")\], natural language processing \[[10](#bib.bib10 ""), [34](#bib.bib34 ""), [40](#bib.bib40 "")\], and robotics \[[16](#bib.bib16 ""), [55](#bib.bib55 "")\]. Despite these benefits, MTL encounters a significant challenge known as task imbalance, where certain tasks dominate the training process while others suffer from insufficient optimization. Previous studies \[[4](#bib.bib4 ""), [48](#bib.bib48 "")\] have indicated that achieving more balanced optimization across tasks often leads to improved overall performance. Addressing such a task imbalance issue necessitates the development of sophisticated optimization strategies to ensure that all tasks benefit equitably from the shared model parameters.

One widely adopted perspective to address the above issue is to balance the convergence speeds of different tasks \[[32](#bib.bib32 ""), [60](#bib.bib60 ""), [35](#bib.bib35 "")\]. Nevertheless, it is highly challenging to accurately analyze the training dynamics and convergence speeds of multiple tasks within the complex MTL system. Most prior methods \[[35](#bib.bib35 ""), [31](#bib.bib31 "")\] approximate the convergence speeds based on the difference or ratio between consecutive loss values. However, different tasks exhibit vastly different loss scales and heterogeneous ultimate loss minima, thus such a simple approximation fails to accurately capture a task’s convergence capability or potential at a specific training stage. As demonstrated by experimental results on widely used benchmarks such as NYUv2, these methods still exhibit considerable task imbalance. Therefore, there is a pressing need for a tool to characterize MTL training dynamics and task convergence properties with a robust theoretical foundation.

To this end, we attempt to analyze the training dynamics within MTL systems by leveraging Neural Tangent Kernel (NTK) theory\[[17](#bib.bib17 ""), [2](#bib.bib2 ""), [7](#bib.bib7 "")\], which provides insights into the optimization trajectory of deep neural networks and has demonstrated its theoretical efficacy in single-task learning (STL) scenarios. From the perspective of NTK theory, the convergence speed of a neural network can be characterized by the eigenvalues of its corresponding NTK matrix. Specifically, lower-frequency components of the target function typically correspond to larger NTK eigenvalues, which converge faster \[[7](#bib.bib7 ""), [52](#bib.bib52 "")\]. In contrast, higher-frequency components often correspond to smaller NTK eigenvalues, which converge more slowly (or are harder to learn). This phenomenon is known as "spectral bias" in the context of single-task learning \[[22](#bib.bib22 ""), [43](#bib.bib43 ""), [53](#bib.bib53 "")\], which bears a strong resemblance to task imbalance in MTL. As explained by the NTK theory, such a distinction in the training dynamics provides a foundational understanding of why certain aspects of the target function are prioritized over others during the training process. However, despite its potential relevance, the application of NTK theory to the field of MTL has been scarcely explored by prior work.

Building upon the above motivation, this work applies the NTK theory to MTL scenarios and introduces an extended NTK matrix to jointly characterize the training dynamics of multiple tasks. Specifically, the target function in MTL is explicitly decomposed into multiple distinct components, each associated with a different task. Our theoretical analysis shows that, the convergence speed of the overall training error is jointly influenced by the NTK matrices corresponding to each task, and tasks associated with larger NTK eigenvalues can be learned more rapidly by the network, subsequently dominating other tasks and resulting in unsatisfactory performance on the remaining tasks. To address this issue, we propose a new MTL approach, NTKMTL, which assigns appropriate weights during training based on the NTK analysis of each task. This method effectively balances the convergence speeds of different tasks and, consequently, reduces task imbalance. In summary, our contributions can be outlined as follows:

1) We introduce a new perspective on understanding task imbalance in MTL by leveraging NTK theory for analysis. Under this perspective, multiple tasks are viewed as distinct components of the training objective, each characterized by unique NTK spectral properties. These inherent differences in spectral characteristics lead to significant disparities in convergence speeds across tasks, subsequently causing task imbalance.

2) Based on the NTK spectral analysis of different tasks in MTL, we propose a new MTL method, NTKMTL. Both theoretical analysis and experimental results demonstrate that NTKMTL effectively addresses the ill-conditioned distribution of NTK eigenvalues during MTL training, thereby alleviating task imbalance.

3) To enhance the practical applicability and computational efficiency, we further introduce NTKMTL-SR, an efficient approximation that leverages the Shared Representation in MTL. NTKMTL-SR requires only a single gradient backpropagation per iteration for shared parameters, which not only provides computational efficiency but also maintains competitive performance.

4) Extensive experiments validate that our methods achieve state-of-the-art performance across a wide range of benchmarks. The experimental scenarios include both multi-task supervised learning and multi-task reinforcement learning, with task numbers ranging from 2 to 40.

## 2 Related Work

Multi-Task Learning. In multi-task learning (MTL), previous approaches can be broadly classified into two categories: loss-oriented and gradient-oriented methods. Loss-oriented methods primarily aim to address convergence discrepancies arising from differences in loss scales, thereby mitigating task imbalance. These methods often exhibit training efficiency as they only require one backpropagation on the aggregated loss. Approaches include Linear Scalarization (LS), Scale-Invariant (SI), homoscedastic uncertainty weighting \[[24](#bib.bib24 "")\], dynamic weight averaging \[[35](#bib.bib35 "")\], self-paced learning \[[37](#bib.bib37 "")\], geometric loss \[[14](#bib.bib14 "")\], random loss weighting \[[29](#bib.bib29 "")\], impartial loss weighting \[[33](#bib.bib33 "")\], fast adaptive optimization \[[31](#bib.bib31 "")\], and multi-task grouping for alignment \[[48](#bib.bib48 "")\]. On the other hand, gradient-oriented methods \[[11](#bib.bib11 "")\] focus on resolving the gradient conflicts issue among shared parameters, seeking the most favorable updating vector to alleviate task imbalance. These types of methods often achieve better performance due to directly obtaining the gradients for all tasks for optimization. Notable approaches in this category include Multiple Gradient Descent Algorithm \[[46](#bib.bib46 "")\], gradient normalization \[[12](#bib.bib12 "")\], gradient conflicts projection \[[57](#bib.bib57 "")\], gradient sign dropout \[[13](#bib.bib13 "")\], impartial gradient weighting \[[33](#bib.bib33 "")\], conflict-averse gradients \[[32](#bib.bib32 "")\], gradient similarity regularisation \[[51](#bib.bib51 "")\], dual balancing \[[27](#bib.bib27 "")\], stochastic direction-oriented update \[[54](#bib.bib54 "")\], smooth tchebycheff scalarization \[[30](#bib.bib30 "")\], independent gradient alignment \[[47](#bib.bib47 "")\], Nash bargaining solution \[[39](#bib.bib39 "")\], fair resource allocation \[[4](#bib.bib4 "")\], performance informed variance reduction approach \[[41](#bib.bib41 "")\], and consistent multi-task learning with task-specific parameters \[[42](#bib.bib42 "")\].

Neural Tangent Kernel theory. Recent theoretical advancements have modeled neural networks in the limits of infinite width and infinitesimal learning rate as kernel regression using the Neural Tangent Kernel (NTK) \[[17](#bib.bib17 ""), [2](#bib.bib2 ""), [7](#bib.bib7 ""), [22](#bib.bib22 "")\]. Specifically, analyses by \[[26](#bib.bib26 "")\] and \[[2](#bib.bib2 "")\] demonstrate that, during gradient descent, the outputs of a neural network remain close to those of a linear dynamical system, with the convergence speed governed by the eigenvalues of the NTK matrix \[[5](#bib.bib5 ""), [7](#bib.bib7 ""), [52](#bib.bib52 ""), [53](#bib.bib53 "")\]. The NTK’s eigendecomposition reveals that its eigenvalue spectrum decays rapidly as a function of frequency, which explains the well-documented "spectral bias" of deep networks towards learning low-frequency functions \[[22](#bib.bib22 ""), [43](#bib.bib43 ""), [53](#bib.bib53 "")\]. Such spectral bias phenomenon bears a strong resemblance to task imbalance in MTL, which is an effective application of NTK theory in explaining neural network training dynamics in STL scenarios. Building on this observation, this paper extends NTK theory to MTL and proposes a solution to address the task imbalance issue.

## 3 Method

### 3.1 Preliminaries

To establish the foundation for our theoretical investigation, we first review the traditional Neural Tangent Kernel theory that explores the training dynamics of deep neural networks. In the subsequent sections, we extend these theoretical frameworks to the multi-task learning context and develop specific methodologies accordingly.

Neural Tangent Kernel. For a deep neural network ff with parameters θ\\theta and training dataset (𝐱,𝐲)\={(xi,yi)}i\=1n(\\mathbf{x},\\mathbf{y})=\\{(x\_{i},y\_{i})\\}\_{i=1}^{n}, the Neural Tangent Kernel (NTK) 𝒦\\mathcal{K} is defined as

𝒦u​v\=⟨∂f⁡(θ,xu)∂θ,∂f⁡(θ,xv)∂θ⟩.\\mathcal{K}\_{uv}=\\left\\langle\\frac{\\partial f(\\theta,x\_{u})}{\\partial\\theta},\\frac{\\partial f(\\theta,{x}\_{v})}{\\partial\\theta}\\right\\rangle.

(1)

Prior works \[[22](#bib.bib22 ""), [26](#bib.bib26 ""), [2](#bib.bib2 "")\] have demonstrated that under certain conditions (e.g., when the learning rate η\\eta approaches zero), the training dynamics of the neural network can be characterized via gradient flow, which is governed by the following ordinary differential equation (ODE) \[[44](#bib.bib44 "")\]:

d​θ​(t)d​t\=−∇ℒ​(θ).\\frac{d\\theta(t)}{dt}=-\\nabla\\mathcal{L}(\\theta).

(2)

This leads to the following theorem:

###### Theorem 3.1.

Let 𝒪⁡(t)\={f⁡(θ,xi)}i\=1n\\mathcal{O}(t)=\\{f(\\theta,x\_{i})\\}\_{i=1}^{n} be the outputs of the neural network at time tt. 𝐱\={xi}i\=1n\\mathbf{x}=\\{x\_{i}\\}\_{i=1}^{n} is the input data, and 𝐲\={yi}i\=1n\\mathbf{y}=\\{y\_{i}\\}\_{i=1}^{n} is the corresponding label, Then 𝒪⁡(t)\\mathcal{O}(t) follows this evolution:

d​𝒪​(t)d​t\=−𝒦⋅(𝒪(t)−𝐲).\\frac{d\\mathcal{O}(t)}{dt}=-\\mathcal{K}\\cdot(\\mathcal{O}(t)-\\mathbf{y}).

(3)

Detailed analysis can be found in the Appendix. Eq. [3](#S3.E3 "Equation 3 ‣ Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") enables us to utilize the neural tangent kernel to analyze the training dynamics of neural networks. Prior work \[[22](#bib.bib22 "")\] demonstrated that as the network width approaches infinity, the NTK 𝒦\\mathcal{K} remains approximately constant during training. A more widely used result is that during the training of deep neural networks, the kernel function is updated much more slowly than the network’s output \[[52](#bib.bib52 ""), [62](#bib.bib62 "")\]. Therefore, Eq. [3](#S3.E3 "Equation 3 ‣ Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") can also be interpreted as an ODE and provides the following approximation:

𝒪⁡(t)≈(𝐈−e−η​𝒦​t)​𝐲.\\mathcal{O}(t)\\approx(\\mathbf{I}-e^{-\\eta\\mathcal{K}t})\\mathbf{y}.

(4)

Spectral analysis in neural network training. Let us consider the training error 𝒪⁡(t)−𝐲\\mathcal{O}(t)-\\mathbf{y}. Since the NTK matrix must be positive semi-definite, we can take its spectral decomposition 𝒦\=Q​Λ​Q⊤\\mathcal{K}=Q\\Lambda Q^{\\top}, where QQ is an orthogonal matrix and Λ\\Lambda is a diagonal matrix whose entries are the eigenvalues λ\\lambda of 𝒦\\mathcal{K}. Then, since e−η​𝒦​t\=Q​e−η​Λ​t​Q⊤e^{-\\eta\\mathcal{K}t}=Qe^{-\\eta\\Lambda t}Q^{\\top}, we have

Q⊤​(𝒪⁡(t)−y)≈−Q⊤​e−η​𝒦​t​𝐲\=−e−η​Λ​t​Q⊤​𝐲.Q^{\\top}(\\mathcal{O}(t)-\\textbf{y})\\approx-Q^{\\top}e^{-\\eta\\mathcal{K}t}\\mathbf{y}=-e^{-\\eta\\Lambda t}Q^{\\top}\\mathbf{y}.

(5)

This implies that, when considering the convergence of training in the NTK eigenbasis, the ii-th component of the absolute error |Q⊤​(𝒪⁡(t)−y)|i\\left|Q^{\\top}(\\mathcal{O}(t)-y)\\right|\_{i} decays at an approximate exponential rate of η​λi\\eta\\lambda\_{i}. In other words, the components of the target function corresponding to kernel eigenvectors with larger eigenvalues are learned more rapidly, resulting in the high-frequency components of the target function converging exceedingly slowly, to the extent that the neural network is difficult to learn these components.

### 3.2 Extended NTK in Multi-Task Learning

In multi-task learning, a neural network with shared parameters θ\\theta is trained to simultaneously learn kk distinct tasks. In the general case, the overall loss function is defined as

ℒ⁡(θ)\=∑i\=1kℓi​(θ).\\mathcal{L}(\\theta)=\\sum\_{i=1}^{k}\\ell\_{i}(\\theta).

(6)

In this context, by leveraging the gradient flow defined in Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") can be further extended as follows:

###### Theorem 3.2.

Let {𝒪1​(t),𝒪2​(t),…,𝒪k​(t)}\\{\\mathcal{O}\_{1}(t),\\mathcal{O}\_{2}(t),\\ldots,\\mathcal{O}\_{k}(t)\\} denote the outputs of the neural network function {f1,f2,…,fk}\\{f\_{1},f\_{2},\\ldots,f\_{k}\\} for the kk tasks at time tt, and let {𝐲1,𝐲2,…,𝐲k}\\{\\mathbf{y}\_{1},\\mathbf{y}\_{2},\\ldots,\\mathbf{y}\_{k}\\} represent the corresponding labels. Then, the ordinary differential equation in Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") gives the following evolution:

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[𝒦11⋯𝒦1​k⋱𝒦k​1⋯𝒦k​k\]⏟𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\],\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-\\underbrace{\\begin{bmatrix}\\mathcal{K}\_{11}&\\cdots&\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\mathcal{K}\_{k1}&\\cdots&\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\widetilde{\\mathcal{K}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix},

(7)

where 𝒦i​j∈ℝn×n\\mathcal{K}\_{ij}\\in\\mathbb{R}^{n\\times n} and 𝒦i​j\=𝒦j​i⊤\\mathcal{K}\_{ij}=\\mathcal{K}\_{ji}^{\\top} for 1≤i,j≤k1\\leq i,j\\leq k. The (u,v)(u,v)-th entry of 𝒦i​j\\mathcal{K}\_{ij} is defined as

(𝒦i​j)u​v\=⟨∂fi​(θ,xu)∂θ,∂fj​(θ,xv)∂θ⟩.(\\mathcal{K}\_{ij})\_{uv}=\\left\\langle\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta},\\frac{\\partial f\_{j}(\\theta,x\_{v})}{\\partial\\theta}\\right\\rangle.

(8)

Detailed proof can be found in the Appendix. We define the large NTK matrix, formed by the training dynamics of the kk tasks in Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), as the extended NTK matrix in MTL, denoted as 𝒦\~\\widetilde{\\mathcal{K}}. It is straightforward to observe that 𝒦i​i\\mathcal{K}\_{ii} for 1≤i≤k1\\leq i\\leq k, as well as 𝒦\~\\widetilde{\\mathcal{K}} itself, are positive semi-definite matrices. In fact, let JiJ\_{i} denote the Jacobian matrix of fif\_{i} with respect to θ\\theta, then we can observe that:

𝒦i​i\=Ji​Ji⊤,and​𝒦\~\=\[J1Jk\]​\[J1⊤⋯Jk⊤\].\\mathcal{K}\_{ii}=J\_{i}J\_{i}^{\\top},\\ \\text{and}\\ \\widetilde{\\mathcal{K}}=\\begin{bmatrix}J\_{1}\\\\ \\vdots\\\\ J\_{k}\\end{bmatrix}\\begin{bmatrix}J\_{1}^{\\top}&\\cdots&J\_{k}^{\\top}\\end{bmatrix}.

(9)

Analogous to the analysis presented in Sec. [3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), we adopt the approximation derived from the theoretical framework of \[[22](#bib.bib22 "")\]. Consequently, Eq. [4](#S3.E4 "Equation 4 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") is extended to the multi-task learning scenarios as follows:

\[𝒪1​(t)𝒪k​(t)\]≈(𝐈−e−η​𝒦\~​t)​\[𝐲1𝐲k\].\\begin{bmatrix}\\mathcal{O}\_{1}(t)\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)\\end{bmatrix}\\approx(\\mathbf{I}-e^{-\\eta\\widetilde{\\mathcal{K}}t})\\begin{bmatrix}\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathbf{y}\_{k}\\end{bmatrix}.

(10)

As mentioned above, the extended NTK matrix is positive semi-definite. Therefore, similarly to previous steps, we perform its spectral decomposition as 𝒦\~\=Q\~​Λ\~​Q\~⊤\\widetilde{\\mathcal{K}}=\\widetilde{Q}\\widetilde{\\Lambda}\\widetilde{Q}^{\\top}. Consequently, the training error MTL can be approximated by the following evolution:

Q\~⊤​(\[𝒪1​(t)𝒪k​(t)\]−\[𝐲1𝐲k\])≈−e−η​Λ\~​t​Q\~⊤​\[𝐲1𝐲k\].\\widetilde{Q}^{\\top}\\left(\\begin{bmatrix}\\mathcal{O}\_{1}(t)\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)\\end{bmatrix}-\\begin{bmatrix}\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathbf{y}\_{k}\\end{bmatrix}\\right)\\approx-e^{-\\eta\\widetilde{\\Lambda}t}\\widetilde{Q}^{\\top}\\begin{bmatrix}\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathbf{y}\_{k}\\end{bmatrix}.

(11)

Eq. [11](#S3.E11 "Equation 11 ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") illustrates that in the MTL scenario, where the learning objectives are explicitly partitioned into kk components, the training dynamics can still be interpreted through spectral analysis of the extended NTK matrix. The substantial disparities in the distribution of NTK eigenvalues across different tasks give rise to bias in convergence speed, causing the network to be dominated by certain specific tasks and hindering the effective simultaneous learning of all tasks.

### 3.3 The Proposed NTKMTL

Theoretical analysis in Sec. [3.2](#S3.SS2 "3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") and experimental results on extensive benchmarks indicate that the traditional multi-task optimization objective in Eq. [6](#S3.E6 "Equation 6 ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") yields unsatisfactory performance, exhibiting significant task imbalance across various task scenarios. Therefore, existing MTL methods often consider obtaining a weight 𝝎\=(ω1,ω2,…,ωk)⊤\\boldsymbol{\\omega}=(\\omega\_{1},\\omega\_{2},\\ldots,\\omega\_{k})^{\\top} to optimize the weighted objective instead:

ℒ⁡(θ)\=∑i\=1kωi​ℓi​(θ).\\mathcal{L}(\\theta)=\\sum\_{i=1}^{k}\\omega\_{i}\\ell\_{i}(\\theta).

(12)

Specifically, the final gradient is gs\=∑i\=1kωi​gig\_{s}=\\sum\_{i=1}^{k}\\omega\_{i}g\_{i}, where gig\_{i} represents the gradient of the ii-th task’s loss ℓi\\ell\_{i} with respect to the shared parameters, and gsg\_{s} is the final aggregated gradient. Under this formulation, the analysis of the extended NTK matrix for MTL systems presented in Sec. [3.2](#S3.SS2 "3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") is correspondingly modified. Therefore, we further introduce the following proposition:

###### Proposition 3.3.

(Extension of Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")) Let {𝒪i​(t)}i\=1k\\{\\mathcal{O}\_{i}(t)\\}\_{i=1}^{k}, {𝐲i}i\=1k\\{\\mathbf{y}\_{i}\\}\_{i=1}^{k}, and {𝒦i​j}1≤i,j≤k\\{\\mathcal{K}\_{ij}\\}\_{1\\leq i,j\\leq k} be defined as in Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). We now replace the MTL optimization objective with the weighted form as presented in Eq. [12](#S3.E12 "Equation 12 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). Consequently, the ordinary differential equation governing the MTL training dynamics in Eq. [7](#S3.E7 "Equation 7 ‣ Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") becomes:

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[ω12​𝒦11⋯ω1​ωk​𝒦1​k⋱ωk​ω1​𝒦k​1⋯ωk2​𝒦k​k\]⏟𝝎​𝝎⊤⊙𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\].\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-{\\underbrace{\\begin{bmatrix}\\omega\_{1}^{2}\\mathcal{K}\_{11}&\\cdots&\\omega\_{1}\\omega\_{k}\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\omega\_{k}\\omega\_{1}\\mathcal{K}\_{k1}&\\cdots&\\omega\_{k}^{2}\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\boldsymbol{\\omega}\\boldsymbol{\\omega^{\\top}}\\odot\\widetilde{\\mathcal{K}}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix}.

(13)

Detailed proof can be found in the Appendix. The NTK matrix derived from the weighted optimization objective in Eq. [12](#S3.E12 "Equation 12 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") remains positive semi-definite, and thus, the analysis in Sec. [3.2](#S3.SS2 "3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") is still applicable. Proposition [3.3](#S3.Thmtheorem3 "Proposition 3.3. ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") shows that the eigenvalues of the new NTK matrix are strongly correlated with 𝝎\\boldsymbol{\\omega}, allowing 𝝎\\boldsymbol{\\omega} to be used to balance the relative magnitudes of the eigenvalues of the NTK across different tasks, thereby balancing the convergence speeds of different tasks.

Motivated by this, our method is designed based on the following strategy: In each training iteration, we compute the maximum eigenvalue λi\\lambda\_{i} of the NTK matrix 𝒦i​i\\mathcal{K}\_{ii} for task ii, which serves as a good indicator of its current convergence speed. We then derive the task weights {ωi}i\=1k\\{\\omega\_{i}\\}\_{i=1}^{k} by normalizing these eigenvalues {λi}i\=1k\\{\\lambda\_{i}\\}\_{i=1}^{k}. Proposition [3.3](#S3.Thmtheorem3 "Proposition 3.3. ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") provides the most intuitive physical interpretation: With the introduction of ωi\\omega\_{i}, the new NTK matrix for each task becomes ωi2​Ki​i\\omega\_{i}^{2}K\_{ii}, and its maximum eigenvalue is accordingly scaled to ωi2​λi\\omega\_{i}^{2}\\lambda\_{i}. Therefore, to achieve eigenvalue normalization across tasks, we enforce the condition that:

ωi∝1λi,i\=1,…,k.\\omega\_{i}\\propto\\frac{1}{\\sqrt{\\lambda\_{i}}},i=1,\\dots,k.

(14)

Directly employing ωi\=1λi\\omega\_{i}=\\frac{1}{\\sqrt{\\lambda\_{i}}} disregards the original scaling inherent in the task eigenvalues. Consequently, to retain the scale information reflecting the tasks’ current convergence speeds, we utilize the average maximum eigenvalue, λ\~\=1k​∑j\=1kλj\\tilde{\\lambda}=\\frac{1}{k}\\sum\_{j=1}^{k}\\lambda\_{j}, to scale {wi}i\=1k\\{w\_{i}\\}\_{i=1}^{k}. This motivates the following definition for ωi\\omega\_{i}:

ωi\=λ\~λi,i\=1,…,k.\\omega\_{i}=\\sqrt{\\frac{\\tilde{\\lambda}}{\\lambda\_{i}}},\\ i=1,\\dots,k.

(15)

The overall algorithm flow is shown in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). Our design of 𝝎\\boldsymbol{\\omega} enables the effective balancing of convergence speeds across different tasks by balancing the maximum eigenvalues of their NTK matrices, while preserving the scale information of the original eigenvalues. Nevertheless, computing the NTK matrix can be time-consuming since it requires dividing a batch into nn mini-batches and computing their gradients. This limitation has prompted us to find a way to compute the NTK with minimal cost, thereby extending the applicability of our method to more scenarios. The following section provides the solution.

Algorithm 1 NTKMTL

1:  Input: Initial model parameters θ0\\theta\_{0}; Learning rate {ηt}\\{\\eta\_{t}\\}; number nn of mini batches. 

2:  for t\=0t=0 to T−1T-1 do 

3:   Compute \[J1t,⋯,Jkt\]\[J\_{1}^{t},\\cdots,J\_{k}^{t}\] for nn mini-batches, and obtain the gradients \[g1t,⋯,gkt\]\[g\_{1}^{t},\\cdots,g\_{k}^{t}\]. 

4:   Obtain the NTK {𝒦i​it}i\=1k\\{\\mathcal{K}\_{ii}^{t}\\}\_{i=1}^{k} and 𝒦t\~\\widetilde{\\mathcal{K}^{t}} through Eq. [8](#S3.E8 "Equation 8 ‣ Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") and [9](#S3.E9 "Equation 9 ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). 

5:   Compute {ωi}i\=1k\\{\\omega\_{i}\\}\_{i=1}^{k} through Eq. [15](#S3.E15 "Equation 15 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). 

6:   Obtain the update vector gst\=∑i\=1kωi​gitg\_{s}^{t}=\\sum\_{i=1}^{k}\\omega\_{i}g\_{i}^{t}. 

7:   Update θt+1\=θt−ηt​gst\\theta\_{t+1}=\\theta\_{t}-\\eta\_{t}g\_{s}^{t}. 

8:  end for 

Algorithm 2 NTKMTL-SR

1:  Input: Initial model parameters θ0\\theta\_{0}; Learning rate {ηt}\\{\\eta\_{t}\\}; number nn of mini batches. 

2:  for t\=0t=0 to T−1T-1 do 

3:   Compute \[J1t​(z),⋯,Jkt​(z)\]\[J\_{1}^{t}(z),\\cdots,J\_{k}^{t}(z)\] with respect to zz for nn mini batches. \[g1t,⋯,gkt\]\[g\_{1}^{t},\\cdots,g\_{k}^{t}\] 

4:   Obtain {𝒦i​it​(z)}i\=1k\\{\\mathcal{K}\_{ii}^{t}(z)\\}\_{i=1}^{k} and 𝒦t\~​(z)\\widetilde{\\mathcal{K}^{t}}(z) through Eq.[9](#S3.E9 "Equation 9 ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") and [17](#S3.E17 "Equation 17 ‣ 3.4 Approximation via Shared Representation ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). 

5:   Compute {ωi}i\=1k\\{\\omega\_{i}\\}\_{i=1}^{k} through Eq. [18](#S3.E18 "Equation 18 ‣ 3.4 Approximation via Shared Representation ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). 

6:   Obtain aggregated loss ℒ\=∑i\=1kωi​ℓi\\mathcal{L}=\\sum\_{i=1}^{k}\\omega\_{i}\\ell\_{i}. 

7:   Back propagate and update parameters θt+1\\theta\_{t+1}. 

8:  end for 

### 3.4 Approximation via Shared Representation

In MTL, the model typically consists of shared parameters θ\\theta and task-specific parameters for kk tasks, where the number of parameters in the task-specific components (typically 1–2 layers of linear or convolutional layers) is much smaller than the number of shared parameters. We define the shared representation zz as the output of the input x after passing through the shared parameters θ\\theta. Then, by applying the chain rule, we can derive the following:

∂fi​(θ,x)∂θ\=∂fi​(θ,x)∂z⋅∂z∂θ.\\frac{\\partial f\_{i}(\\theta,\\textbf{x})}{\\partial\\theta}=\\frac{\\partial f\_{i}(\\theta,\\textbf{x})}{\\partial z}\\cdot\\frac{\\partial z}{\\partial\\theta}.

(16)

Note that ∂z∂θ\\frac{\\partial z}{\\partial\\theta} is the same for all tasks and acts on all {fi}i\=1k\\{f\_{i}\\}\_{i=1}^{k}. As a result, it further impacts the overall NTK 𝒦\~\\widetilde{\\mathcal{K}}. This implies that 𝒦\~\\widetilde{\\mathcal{K}} can be viewed as a projection of the NTK matrix 𝒦\~​(z)\\widetilde{\\mathcal{K}}(z) of zz onto the feature space through ∂z∂θ\\frac{\\partial z}{\\partial\\theta}. These analyses indicate that, similar to some previous works \[[23](#bib.bib23 ""), [47](#bib.bib47 "")\], NTKMTL has the ability to accelerate computation using shared representation, and we name this approximation algorithm NTKMTL-SR. Specifically, we consider approximating the original method using the NTK analysis of zz, i,e. by replacing Eq. [8](#S3.E8 "Equation 8 ‣ Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") with the following expression:

(𝒦i​j​(z))u​v\=⟨∂fi​(θ,xu)∂z,∂fj​(θ,xv)∂z⟩,(\\mathcal{K}\_{ij}(z))\_{uv}=\\left\\langle\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial z},\\frac{\\partial f\_{j}(\\theta,x\_{v})}{\\partial z}\\right\\rangle,

(17)

which further leads to the subsequent formulation for 𝝎\\boldsymbol{\\omega}:

ωi\=λ\~​(z)λi​(z),i\=1,…,k,\\omega\_{i}=\\sqrt{\\frac{\\tilde{\\lambda}(z)}{\\lambda\_{i}(z)}},\\ i=1,\\dots,k,

(18)

where λi​(z)\\lambda\_{i}(z) represents the maximum eigenvalue of the NTK matrix Ki​i​(z)K\_{ii}(z), and λ\~​(z)\\tilde{\\lambda}(z) is the average of these eigenvalues across all kk tasks, defined as λ\~​(z)\=1k​∑j\=1kλj​(z)\\tilde{\\lambda}(z)=\\frac{1}{k}\\sum\_{j=1}^{k}\\lambda\_{j}(z). Since computing the gradient of fi​(θ,x)f\_{i}(\\theta,\\textbf{x}) with respect to zz only requires backpropagation through the task-specific parameters, it incurs little additional time and memory cost. After obtaining 𝝎\\boldsymbol{\\omega}, instead of computing the gradients separately for each of the kk tasks and then weighting them, we can directly compute the aggregated loss using Eq. [12](#S3.E12 "Equation 12 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") and perform one backpropagation, meaning that we only need to compute the gradient for shared parameters θ\\theta once per iteration.

The overall algorithm is outlined in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). Fig. [1](#S4.F1 "Figure 1 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") illustrates the training speed on the CelebA benchmark with up to 40 tasks, showing that NTKMTL-SR exhibits nearly the same training speed as traditional loss-oriented Linear Scalarization. This further enhances the generalizability of our method under various constraints.

## 4 Experiments

Table 1: Results on NYU-v2 (3-task) dataset. Each experiment is repeated 3 times with different random seeds and the average is reported. The detailed standard error is reported in the Appendix. The best scores are reported in .

 Method Segmentation Depth Surface Normal MR ↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} mIoU ↑\\uparrow Pix Acc ↑\\uparrow Abs Err ↓\\downarrow Rel Err ↓\\downarrow Angle Distance ↓\\downarrow Within t∘t^{\\circ} ↑\\uparrow Mean Median 11.25 22.5 30 STL 38.30 63.76 0.6754 0.2780 25.01 19.21 30.14 57.20 69.15 LS 39.29 65.33 0.5493 0.2263 28.15 23.96 22.09 47.50 61.08 17.44 5.59 SI 38.45 64.27 0.5354 0.2201 27.60 23.37 22.53 48.57 62.32 16.11 4.39 RLW \[[28](#bib.bib28 "")\] 37.17 63.77 0.5759 0.2410 28.27 24.18 22.26 47.05 60.62 20.22 7.78 DWA \[[35](#bib.bib35 "")\] 39.11 65.31 0.5510 0.2285 27.61 23.18 24.17 50.18 62.39 16.33 3.57 UW \[[25](#bib.bib25 "")\] 36.87 63.17 0.5446 0.2260 27.04 22.61 23.54 49.05 63.65 16.00 4.05 MGDA \[[46](#bib.bib46 "")\] 30.47 59.90 0.6070 0.2555 24.88 19.45 29.18 56.88 69.36 11.44 1.38 PCGrad \[[58](#bib.bib58 "")\] 38.06 64.64 0.5550 0.2325 27.41 22.80 23.86 49.83 63.14 16.89 3.97 GradDrop \[[13](#bib.bib13 "")\] 39.39 65.12 0.5455 0.2279 27.48 22.96 23.38 49.44 62.87 15.56 3.58 CAGrad \[[32](#bib.bib32 "")\] 39.79 65.49 0.5486 0.2250 26.31 21.58 25.61 52.36 65.58 11.56 0.20 IMTL-G \[[33](#bib.bib33 "")\] 39.35 65.60 0.5426 0.2256 26.02 21.19 26.20 53.13 66.24 10.89 -0.76 MoCo \[[18](#bib.bib18 "")\] 40.30 66.07 0.5575 0.2135 26.67 21.83 25.61 51.78 64.85 10.89 0.16 Nash-MTL \[[39](#bib.bib39 "")\] 40.13 65.93 0.5261 0.2171 25.26 20.08 28.40 55.47 68.15 8.00 -4.04 Aligned-MTL \[[47](#bib.bib47 "")\] 40.15 66.05 0.5520 0.2291 25.37 19.89 28.30 55.29 67.95 10.44 -3.12 FAMO \[[31](#bib.bib31 "")\] 38.88 64.90 0.5474 0.2194 25.06 19.57 29.21 56.61 68.98 9.00 -4.10 SDMGrad \[[54](#bib.bib54 "")\] 40.47 65.90 0.5225 0.2084 25.07 19.99 28.54 55.74 68.53 6.11 -4.84 DB-MTL \[[27](#bib.bib27 "")\] 41.42 66.45 0.5251 0.2160 25.03 19.50 28.72 56.17 68.73 4.56 -5.36 STCH \[[30](#bib.bib30 "")\] 41.35 66.07 0.4965 0.2010 26.55 21.81 24.84 51.39 64.86 8.22 -1.35 FairGrad \[[4](#bib.bib4 "")\] 39.74 66.01 0.5377 0.2236 24.84 19.60 29.26 56.58 69.16 6.44 -4.66 GO4Align \[[48](#bib.bib48 "")\] 40.42 65.37 0.5492 0.2167 24.76 18.94 30.54 57.87 69.84 5.00 -6.08 NTKMTL 39.68 65.43 0.5296 0.2168 24.24 18.63 30.74 58.72 70.78 4.33 -6.99 NTKMTL-SR 40.23 65.28 0.5261 0.2136 24.88 19.58 29.53 56.67 69.08 5.56 -5.35 

### 4.1 Protocols

We evaluate the performance of our proposed NTKMTL and NTKMTL-SR across a wide range of MTL scenarios, including multi-task supervised learning and multi-task reinforcement learning. For multi-task supervised learning, experiments are conducted on several benchmarks, including dense prediction tasks on the NYUv2 \[[49](#bib.bib49 "")\] and CityScapes \[[15](#bib.bib15 "")\] datasets, regression tasks on the QM9 \[[8](#bib.bib8 "")\] dataset, and image-level classification on the CelebA \[[36](#bib.bib36 "")\] dataset. For multi-task reinforcement learning, experiments are performed in the MT10 environment from the Meta-World benchmark \[[59](#bib.bib59 "")\]. Due to the memory and time cost associated with fully computing the gradients of nn mini-batches for shared parameters to construct the NTK matrix, we set n\=1n=1 for NTKMTL to ensure fairness when comparing with other methods. In this case, it is consistent with other gradient-oriented methods, requiring kk gradient backpropagations per iteration. As for NTKMTL-SR, since its cost of constructing the NTK matrix is minimal, we set n\=4n=4 by default based on experimental validation.

Baselines. We comprehensively compare the proposed NTKMTL and NTKMTL-SR with the following methods: Single-task learning (STL), Linear Scalarization (LS), Scale-Invariant (SI), Dynamic Weight Average (DWA)  \[[35](#bib.bib35 "")\], Uncertainty Weighting (UW)  \[[25](#bib.bib25 "")\], Multi-Gradient Descent Algorithm (MGDA)  \[[46](#bib.bib46 "")\], Random Loss Weighting (RLW)  \[[28](#bib.bib28 "")\], PCGrad  \[[58](#bib.bib58 "")\], GradDrop  \[[13](#bib.bib13 "")\], CAGrad  \[[32](#bib.bib32 "")\], IMTL-G  \[[33](#bib.bib33 "")\], Nash-MTL \[[39](#bib.bib39 "")\], Moco \[[18](#bib.bib18 "")\], Aligned-MTL \[[47](#bib.bib47 "")\], SDMGrad \[[54](#bib.bib54 "")\], DB-MTL \[[27](#bib.bib27 "")\], STCH \[[30](#bib.bib30 "")\], FAMO \[[31](#bib.bib31 "")\], FairGrad \[[4](#bib.bib4 "")\] and GO4Align \[[48](#bib.bib48 "")\].

Metrics. We follow previous works \[[39](#bib.bib39 ""), [4](#bib.bib4 "")\] and use two overall performance metrics for MTL: (1) Δ​m%\\Delta m\\%, the average performance drop relative to the STL baseline: Δ​m%\=1𝒮​∑i\=1𝒮(−1)δi​(Mm,i−Mb,i)Mb,i×100%,\\Delta m\\%=\\frac{1}{\\mathcal{S}}\\sum\_{i=1}^{\\mathcal{S}}(-1)^{\\delta\_{i}}\\frac{(M\_{m,i}-M\_{b,i})}{M\_{b,i}}\\times 100\\%, where 𝒮\\mathcal{S} is the number of metrics. Mb,iM\_{b,i} is the baseline STL metric and Mm,iM\_{m,i} is the metric from the MTL method. δi\=1\\delta\_{i}=1 if a higher value is better for MiM\_{i}, and 0 otherwise. (2) Mean Rank (MR): MR reports the average rank of a method across all tasks, where a lower value indicates better performance. A method with the top rank in all tasks has an MR of 1.

### 4.2 Multi-Task Supervised Learning

Dense Prediction. Widely used benchmarks in this domain include NYUv2 \[[49](#bib.bib49 "")\] and CityScapes \[[15](#bib.bib15 "")\]. The NYUv2 dataset includes three tasks: semantic segmentation, depth estimation, and surface normal prediction, while CityScapes includes semantic segmentation and depth estimation tasks. The experimental results are presented in Table [1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") and [2](#S4.T2 "Table 2 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") in the main text, and Table [7](#A3.T7 "Table 7 ‣ C.2 Detailed Results with Standard Errors ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") in the Appendix.

On the NYUv2 dataset, the difficulty levels of the three tasks show significant variation. Previous methods generally outperform the Single Task Learning (STL) baseline in the tasks of semantic segmentation and depth estimation, but almost all of them consistently underperform STL on the surface normal prediction task, leading to a significant task imbalance in the overall results. In contrast, by leveraging NTK theory to balance convergence speed of each task during training, both NTKMTL and NTKMTL-SR show strong performance in the surface normal prediction task. Notably, among all existing methods, only NTKMTL and GO4Align consistently outperform the STL baseline across all three tasks, achieving a more balanced optimization. Moreover, NTKMTL achieves SOTA on this benchmark with an impressive mean rank of 4.33 and the best performance drop of -6.99%.

Table 2: Results on CityScapes (2-task) and CelebA (40-task). Each experiment is repeated 3 times with different random seeds and the average is reported. Detailed standard error is reported in the Appendix. Best scores are reported in .

 Method CityScapes Celeba MR ↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} MR ↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} LS 8.25 22.60 7.85 4.15 SI 11.50 14.11 9.75 7.20 RLW \[[28](#bib.bib28 "")\] 10.25 24.38 6.90 1.46 DWA \[[35](#bib.bib35 "")\] 7.75 21.45 8.72 3.20 UW \[[25](#bib.bib25 "")\] 7.75 5.89 7.38 3.23 MGDA \[[46](#bib.bib46 "")\] 12.00 44.14 12.97 14.85 PCGrad \[[58](#bib.bib58 "")\] 8.50 18.29 8.53 3.17 CAGrad \[[32](#bib.bib32 "")\] 7.25 11.64 8.10 2.48 IMTL-G \[[33](#bib.bib33 "")\] 5.50 11.10 6.25 0.84 Nash-MTL \[[39](#bib.bib39 "")\] 3.50 6.82 6.50 2.84 FAMO \[[31](#bib.bib31 "")\] 7.75 8.13 6.45 1.21 FairGrad \[[4](#bib.bib4 "")\] 2.25 5.18 6.92 0.37 NTKMTL 7.00 1.92 4.35 -0.77 NTKMTL-SR 6.25 3.84 4.33 0.23 

On the CityScapes dataset, NTKMTL and NTKMTL-SR also perform exceptionally well, as shown in Table [2](#S4.T2 "Table 2 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). More detailed results can be found in Table [7](#A3.T7 "Table 7 ‣ C.2 Detailed Results with Standard Errors ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") in the Appendix. Compared to existing methods that tend to prioritize the optimization of semantic segmentation, NTKMTL and NTKMTL-SR achieve more balanced results across both segmentation and depth estimation tasks, with NTKMTL also obtaining the best performance drop.

Figure 1: Training time per epoch for various methods on CelebA (40-task) dataset.

Image-Level Classification. We also evaluated the performance of our proposed NTKMTL and NTKMTL-SR on the CelebA dataset. CelebA \[[36](#bib.bib36 "")\] is a large-scale facial attribute dataset consisting of over 200K images, each labeled with 40 attributes, such as smiling, wavy hair, and mustache. This task represents a 40-task MTL classification problem, where each task is designed to predict one binary attribute. This benchmark tests both the collaborative optimization capability and efficiency of MTL methods when dealing with a large number of tasks. The results are shown in Table [2](#S4.T2 "Table 2 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). In this challenging setting, NTKMTL achieves better average performance than the STL baseline, as indicated by the negative Δ​m%\\Delta m\\%, which was not achievable by previous methods. NTKMTL also achieved state-of-the-art performance in both MR and Δ​m%\\Delta m\\%.

Fig. [1](#S4.F1 "Figure 1 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") visualizes the training time per epoch for various loss-oriented and gradient-oriented methods. On the CelebA dataset with up to 40 tasks, NTKMTL-SR maintains a speed comparable to loss-oriented methods.

Multi-Task Regression. QM9 \[[8](#bib.bib8 "")\] is a widely used benchmark for multi-task regression, containing over 130K organic molecules represented as graphs. It includes 11 tasks, each requiring the prediction of a molecular property. Due to the large number of tasks, as well as the significant differences in task difficulty and convergence speeds, existing MTL methods exhibit a substantial performance drop compared to the STL baseline.

On this benchmark, previous methods were implemented using a shared MPNN codebase \[[39](#bib.bib39 ""), [31](#bib.bib31 ""), [19](#bib.bib19 "")\]. However, we find that the hyperparameter settings in this codebase are suboptimal (specifically, the improper learning rate scheduler makes the learning rate decay too quickly, resulting in incomplete convergence), potentially leading to unfair comparisons. Therefore, we adjusted specific hyperparameters and reproduced all baselines for a fair comparison. To ensure accurate evaluation, we also reproduced the 11-task STL baselines under the same settings.

Experimental results are presented in Table [3](#S4.T3 "Table 3 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). More detailed explanations and settings are provided in Appendix [C](#A3 "Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). As both STL baselines and MTL methods show significant performance improvements under the new settings, we can observe different behaviors among previous methods. The final Δ​m%\\Delta m\\% of some methods (e.g., LS, RLW, PCGrad, CAGrad) were largely consistent with their originally reported values. For other methods (e.g., SI, UW, FAMO, GO4Align), the Δ​m%\\Delta m\\% significantly improved compared to the reported results. This indicates that the previous hyperparameter settings fail to fully exhibit the capabilities of these methods. Under the new settings that better ensure convergence, their performance shows further improvement. On this benchmark, NTKMTL shows competitive results. NTKMTL-SR surpasses NTKMTL in performance and achieves state-of-the-art results, which we attribute to the natural alignment of the L2 loss used in regression tasks as analyzed in Appendix [B.1](#A2.SS1 "B.1 Proof of Theorem ‣ Appendix B Theoretical Analysis ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). As a result, larger nn produces a more accurate estimate of convergence speed.

Table 3: Results on QM9 (11-task) dataset. All baselines are reproduced under optimized hyperparameter settings. The best scores are reported in . More details are reported in the Appendix.

 Method μ\\mu α\\alpha ϵH​O​M​O\\epsilon\_{HOMO} ϵL​U​M​O\\epsilon\_{LUMO} ⟨R2⟩\\langle R^{2}\\rangle ZPVE U0U\_{0} UU HH GG cvc\_{v} MR↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} MAE ↓\\downarrow STL 0.060 0.156 60.54 51.22 0.419 3.08 39.3 42.9 41.7 43.1 0.061 LS 0.077 0.253 55.95 68.59 4.163 11.08 109.2 109.8 110.1 106.7 0.099 9.82 179.8 SI 0.159 0.242 109.6 96.80 0.732 3.308 34.35 34.37 34.33 35.16 0.081 5.27 39.7 RLW \[[28](#bib.bib28 "")\] 0.090 0.277 62.49 76.39 4.948 12.54 124.8 124.9 125.0 122.1 0.115 12.09 222.6 DWA \[[35](#bib.bib35 "")\] 0.078 0.239 55.17 67.40 3.992 10.92 107.4 108.1 108.3 105.4 0.098 8.73 173.0 UW \[[25](#bib.bib25 "")\] 0.194 0.274 120.6 102.8 0.763 3.698 41.11 41.13 41.16 41.75 0.089 8.27 58.4 MGDA \[[46](#bib.bib46 "")\] 0.154 0.266 95.20 67.51 3.088 4.468 49.38 49.21 49.62 49.69 0.087 8.36 101.4 PCGrad \[[58](#bib.bib58 "")\] 0.078 0.221 59.14 67.82 2.937 6.691 88.24 88.65 88.85 87.36 0.084 7.91 118.6 CAGrad \[[32](#bib.bib32 "")\] 0.083 0.234 57.80 70.98 2.718 5.352 76.47 76.93 77.05 76.32 0.089 8.27 102.4 Nash-MTL \[[39](#bib.bib39 "")\] 0.086 0.218 69.78 66.18 2.153 4.679 59.63 59.94 59.98 59.97 0.082 6.91 72.9 FAMO \[[31](#bib.bib31 "")\] 0.128 0.230 98.09 84.42 0.859 3.541 40.24 40.57 40.62 40.21 0.081 5.91 38.9 FairGrad \[[4](#bib.bib4 "")\] 0.109 0.208 81.74 72.82 1.669 3.418 51.31 51.67 51.72 51.97 0.079 6.64 57.0 GO4Align \[[48](#bib.bib48 "")\] 0.113 0.314 74.46 91.04 0.912 3.632 36.06 36.38 36.41 36.58 0.104 6.64 40.5 NTKMTL 0.091 0.212 70.97 70.81 2.113 3.835 44.18 44.56 44.53 44.38 0.077 5.91 56.7 NTKMTL-SR 0.081 0.207 75.95 69.10 1.176 3.689 40.14 40.46 40.48 40.49 0.074 4.00 30.7 

### 4.3 Multi-Task Reinforcement Learning

We further assess our method using the MT10 benchmark, which consists of 10 robotic manipulation tasks from the MetaWorld environment \[[59](#bib.bib59 "")\]. The goal in this setting is to train a single policy that can generalize across a variety of tasks, including pick-and-place and door-opening. Specifically, we follow the setup outlined in previous works \[[39](#bib.bib39 ""), [31](#bib.bib31 "")\] and use Soft Actor-Critic (SAC) \[[21](#bib.bib21 "")\] as the core algorithm. Our implementation builds upon the MTRL codebase from \[[39](#bib.bib39 ""), [4](#bib.bib4 "")\], training the model for 2 million steps with a batch size of 1280.

Table 4: Results on MT10 benchmark across 10 random seeds.

 Method success rate (mean ± stderr) STL 0.90 ±\\pm 0.03 MTL SAC \[[59](#bib.bib59 "")\] 0.49 ±\\pm 0.07 MTL SAC + TE \[[59](#bib.bib59 "")\] 0.54 ±\\pm 0.05 MH SAC \[[59](#bib.bib59 "")\] 0.61 ±\\pm 0.04 PCGrad \[[58](#bib.bib58 "")\] 0.72 ±\\pm 0.02 CAGrad \[[32](#bib.bib32 "")\] 0.83 ±\\pm 0.05 MoCo \[[18](#bib.bib18 "")\] 0.75 ±\\pm 0.05 Nash-MTL \[[39](#bib.bib39 "")\] 0.91 ±\\pm 0.03 FAMO \[[31](#bib.bib31 "")\] 0.83 ±\\pm 0.05 FairGrad \[[4](#bib.bib4 "")\] 0.84 ±\\pm 0.07 Aligned-MTL \[[47](#bib.bib47 "")\] 0.97 ±\\pm 0.05 NTKMTL 0.96 ±\\pm 0.03 

In contrast to the multi-task supervised learning networks where shared and task-specific parameters can be easily distinguished, the MTRL problems commonly involve learning a single policy. Consequently, it is difficult to partition the parameters into shared and task-specific components in the same manner. Therefore, while applying NTKMTL-SR to this scenario is challenging, we primarily validate the performance of NTKMTL. We compare NTKMTL with several state-of-the-art methods, including Multi-task SAC (MTL SAC) \[[59](#bib.bib59 "")\], Multi-task SAC with Task Encoder (MTL SAC + TE) \[[59](#bib.bib59 "")\], Multi-headed SAC (MH SAC) \[[59](#bib.bib59 "")\], PCGrad \[[58](#bib.bib58 "")\], CAGrad \[[32](#bib.bib32 "")\], MoCo \[[18](#bib.bib18 "")\], Nash-MTL \[[39](#bib.bib39 "")\], FAMO \[[31](#bib.bib31 "")\], FairGrad \[[4](#bib.bib4 "")\] and Aligned-MTL \[[47](#bib.bib47 "")\]. The experimental results are reported in Table [4](#S4.T4 "Table 4 ‣ 4.3 Multi-Task Reinforcement Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). In the MTRL scenario, NTKMTL continues to demonstrate strong performance with a competitive success rate.

## 5 Conclusion, Limitations and Future Work

In this paper, we introduce a new perspective on understanding task imbalance in MTL by leveraging NTK theory for analysis, and propose a new MTL method, NTKMTL. Specifically, we conduct spectral analysis of the NTK matrix during training, adjust the maximum eigenvalues of the task-specific NTK matrices to balance their convergence speeds, thereby mitigating task imbalance. Furthermore, we present NTKMTL-SR, an efficient variant based on approximation via shared representation, which achieves competitive performance with improved training efficiency. Extensive experiments have shown the strong performance of both NTKMTL and NTKMTL-SR, further demonstrating the applicability and generalization of our method across a wide range of scenarios.

Limitations and Future Work. In this work, we proposed the extended NTK 𝒦\~\\widetilde{\\mathcal{K}} for MTL as a tool for more comprehensive analysis of training dynamics across tasks. The weight design of our current method mainly focuses on the analysis of the task-specific NTK matrices 𝒦i​i\\mathcal{K}\_{ii}. The full structure of the extended NTK matrix 𝒦\~\\widetilde{\\mathcal{K}} offers further analytical opportunities. For example, analyzing the off-diagonal 𝒦i​j\\mathcal{K}\_{ij} blocks could reveal novel insights into task interactions or guide the development of task grouping strategies. These potential avenues are left for future investigation.

## References

*   \[1\] Idan Achituve, Haggai Maron, and Gal Chechik. Self-supervised learning for domain adaptation on point clouds. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 123–133, 2021.
*   \[2\] Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning, pages 322–332. PMLR, 2019.
*   \[3\] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.
*   \[4\] Hao Ban and Kaiyi Ji. Fair resource allocation in multi-task learning. arXiv preprint arXiv:2402.15638, 2024.
*   \[5\] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritchman. Frequency bias in neural networks for input of non-uniform density. In International conference on machine learning, pages 685–694. PMLR, 2020.
*   \[6\] Jonathan Baxter. A model of inductive bias learning. Journal of artificial intelligence research, 12:149–198, 2000.
*   \[7\] Alberto Bietti and Julien Mairal. On the inductive bias of neural tangent kernels. Advances in Neural Information Processing Systems, 32, 2019.
*   \[8\] Lorenz C Blum and Jean-Louis Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database gdb-13. Journal of the American Chemical Society, 131(25):8732–8733, 2009.
*   \[9\] Rich Caruana. Multitask learning. Machine learning, 28:41–75, 1997.
*   \[10\] Shijie Chen, Yu Zhang, and Qiang Yang. Multi-task learning in natural language processing: An overview. ACM Computing Surveys, 56(12):1–32, 2024.
*   \[11\] Weiyu Chen, Xiaoyuan Zhang, Baijiong Lin, Xi Lin, Han Zhao, Qingfu Zhang, and James T Kwok. Gradient-based multi-objective deep learning: Algorithms, theories, applications, and beyond. arXiv preprint arXiv:2501.10945, 2025.
*   \[12\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018.
*   \[13\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.
*   \[14\] Sumanth Chennupati, Ganesh Sistu, Senthil Yogamani, and Samir A Rawashdeh. Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019.
*   \[15\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
*   \[16\] Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. In 2017 IEEE international conference on robotics and automation (ICRA), pages 2169–2176. IEEE, 2017.
*   \[17\] Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. arXiv preprint arXiv:1810.02054, 2018.
*   \[18\] Heshan Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen. Mitigating gradient bias in multi-objective learning: A provably convergent approach. International Conference on Learning Representations, 2023.
*   \[19\] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. PMLR, 2017.
*   \[20\] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pages 270–287, 2018.
*   \[21\] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.
*   \[22\] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.
*   \[23\] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In International Conference on Learning Representations.
*   \[24\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
*   \[25\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
*   \[26\] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems, 32, 2019.
*   \[27\] Baijiong Lin, Weisen Jiang, Feiyang Ye, Yu Zhang, Pengguang Chen, Ying-Cong Chen, Shu Liu, and James T Kwok. Dual-balancing for multi-task learning. arXiv preprint arXiv:2308.12029, 2023.
*   \[28\] Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. 2021.
*   \[29\] Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. arXiv preprint arXiv:2111.10603, 2021.
*   \[30\] Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Smooth tchebycheff scalarization for multi-objective optimization. arXiv preprint arXiv:2402.19078, 2024.
*   \[31\] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36, 2024.
*   \[32\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021.
*   \[33\] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. iclr, 2021.
*   \[34\] Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. Adversarial multi-task learning for text classification. arXiv preprint arXiv:1704.05742, 2017.
*   \[35\] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019.
*   \[36\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.
*   \[37\] Keerthiram Murugesan and Jaime Carbonell. Self-paced multitask learning with shared knowledge. arXiv preprint arXiv:1703.00977, 2017.
*   \[38\] Aviv Navon, Idan Achituve, Haggai Maron, Gal Chechik, and Ethan Fetaya. Auxiliary learning by implicit differentiation. arXiv preprint arXiv:2007.02693, 2020.
*   \[39\] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022.
*   \[40\] Jonathan Pilault, Amine Elhattami, and Christopher Pal. Conditionally adaptive multi-task learning: Improving transfer learning in nlp using fewer parameters & less data. arXiv preprint arXiv:2009.09139, 2020.
*   \[41\] Xiaohan Qin, Xiaoxing Wang, and Junchi Yan. Revisiting fairness in multitask learning: A performance-driven approach for variance reduction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20492–20501, 2025.
*   \[42\] Xiaohan Qin, Xiaoxing Wang, and Junchi Yan. Towards consistent multi-task learning: Unlocking the potential of task-specific parameters. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10067–10076, 2025.
*   \[43\] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.
*   \[44\] Daniel A Roberts, Sho Yaida, and Boris Hanin. The principles of deep learning theory, volume 46. Cambridge University Press Cambridge, MA, USA, 2022.
*   \[45\] S Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.
*   \[46\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018.
*   \[47\] Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20083–20093, 2023.
*   \[48\] Jiayi Shen, Cheems Wang, Zehao Xiao, Nanne Van Noord, and Marcel Worring. Go4align: Group optimization for multi-task alignment. arXiv preprint arXiv:2404.06486, 2024.
*   \[49\] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012.
*   \[50\] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International conference on machine learning, pages 9120–9132. PMLR, 2020.
*   \[51\] Vincent Szolnoky, Viktor Andersson, Balázs Kulcsár, and Rebecka Jörnsten. On the interpretability of regularisation for neural networks through model gradient similarity. Advances in Neural Information Processing Systems, 35:16319–16330, 2022.
*   \[52\] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020.
*   \[53\] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.
*   \[54\] Peiyao Xiao, Hao Ban, and Kaiyi Ji. Direction-oriented multi-objective learning: Simple and provable stochastic algorithms. Advances in Neural Information Processing Systems, 36:4509–4533, 2023.
*   \[55\] Caiming Xiong, SHU Tianmin, and Richard Socher. Hierarchical and interpretable skill acquisition in multi-task reinforcement learning, January 24 2023. US Patent 11,562,287.
*   \[56\] Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo. Adatask: A task-aware adaptive learning rate approach to multi-task learning. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 10745–10753, 2023.
*   \[57\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.
*   \[58\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.
*   \[59\] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.
*   \[60\] Hayoung Yun and Hanjoo Cho. Achievement-based training progress balancing for multi-task learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16935–16944, 2023.
*   \[61\] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE transactions on knowledge and data engineering, 34(12):5586–5609, 2021.
*   \[62\] Zelin Zhao, Fenglei Fan, Wenlong Liao, and Junchi Yan. Grounding and enhancing grid-based models for neural fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19425–19435, 2024.
*   \[63\] Ce Zheng, Wenhan Wu, Chen Chen, Taojiannan Yang, Sijie Zhu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A survey. ACM Computing Surveys, 56(1):1–37, 2023.

## Appendix A Definitions of Notations

Due to the numerous concepts and theoretical derivations presented in the paper, we provide detailed definitions of the notations in Table [5](#A1.T5 "Table 5 ‣ Appendix A Definitions of Notations ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") to assist readers in better understanding the content.

Table 5: Definitions of notations in this paper.

| Variable                                                                                                                                                                                                               | Definition |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| the number of tasks                                                                                                                                                                                                    |            |
| the mapping function of a deep neural network                                                                                                                                                                          |            |
| the mapping function of ii-th task in MTL scenario                                                                                                                                                                     |            |
| the number of data points in the dataset, or the number of mini-batches in a batch                                                                                                                                     |            |
| training dataset {(xi,yi)}i\=1n\\{(x\_{i},y\_{i})\\}\_{i=1}^{n}in Single-Task Learning (STL) setting                                                                                                                   |            |
| multiple labels for the kk tasks in MTL setting                                                                                                                                                                        |            |
| the network parameters, specifically referring to the shared parameters in MTL                                                                                                                                         |            |
| the current time step                                                                                                                                                                                                  |            |
| the Neural Tangent Kernel matrix defined in Eq. [1](#S3.E1 "Equation 1 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")              |            |
| the eigenvalues of the NTK                                                                                                                                                                                             |            |
| the overall loss function, with different definitions in different scenarios                                                                                                                                           |            |
| the losses with respect to kk tasks                                                                                                                                                                                    |            |
| the output {f⁡(xi,θ⁡(t))}i\=1n\\{f(x\_{i},\\theta(t))\\}\_{i=1}^{n} of the network in STL setting and time tt                                                                                                          |            |
| the output of the network for kk tasks in MTL setting and time tt                                                                                                                                                      |            |
| the Identity matrix                                                                                                                                                                                                    |            |
| the learning rate at time tt                                                                                                                                                                                           |            |
| the extended NTK matrix defined in Eq. [9](#S3.E9 "Equation 9 ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") |            |
| the Jacobian matrix of fif\_{i} with respect to θ\\theta                                                                                                                                                               |            |
| {ωi}i\=1k\\{\\omega\_{i}\\}\_{i=1}^{k}, representing the weights of the different tasks                                                                                                                                |            |
| the shared representation of the input data in MTL setting                                                                                                                                                             |            |

## Appendix B Theoretical Analysis

### B.1 Proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")

Theorem 3.1. Let 𝒪⁡(t)\={f⁡(θ,xi)}i\=1n\\mathcal{O}(t)=\\{f(\\theta,x\_{i})\\}\_{i=1}^{n} be the outputs of the neural network at time tt. 𝐱\={xi}i\=1n\\mathbf{x}=\\{x\_{i}\\}\_{i=1}^{n} is the input data, and 𝐲\={yi}i\=1n\\mathbf{y}=\\{y\_{i}\\}\_{i=1}^{n} is the corresponding label, Then 𝒪⁡(t)\\mathcal{O}(t) follows this evolution:

d​𝒪​(t)d​t\=−𝒦⋅(𝒪(t)−𝐲).\\frac{d\\mathcal{O}(t)}{dt}=-\\mathcal{K}\\cdot(\\mathcal{O}(t)-\\mathbf{y}).

###### Proof.

Following previous works \[[44](#bib.bib44 ""), [52](#bib.bib52 ""), [62](#bib.bib62 "")\], we build up the analysis framework for NTK in a supervised regression setting. The overall loss ℒ⁡(θ)\\mathcal{L}(\\theta) is defined as:

ℒ⁡(θ)\=∑i\=1n12​(f⁡(θ,xi)−yi)2.\\mathcal{L}(\\theta)=\\sum\_{i=1}^{n}\\frac{1}{2}(f(\\theta,x\_{i})-y\_{i})^{2}.

(19)

Through the gradient flow in Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), we can obtain:

d​θd​t\\displaystyle\\frac{d\\theta}{dt}

\=−∇θℒ​(θ)\\displaystyle=-\\nabla\_{\\theta}\\mathcal{L}(\\theta)

(20)

\=−∑i\=1n∂ℒ∂f⁡(θ,xi)⋅∂f⁡(θ,xi)∂θ\\displaystyle=-\\sum\_{i=1}^{n}\\frac{\\partial\\mathcal{L}}{\\partial f(\\theta,x\_{i})}\\cdot\\frac{\\partial f(\\theta,x\_{i})}{\\partial\\theta}

\=−∑i\=1n∂f⁡(θ,xi)∂θ(f(θ,xi)−yi).\\displaystyle=-\\sum\_{i=1}^{n}\\frac{\\partial f(\\theta,x\_{i})}{\\partial\\theta}(f(\\theta,x\_{i})-y\_{i}).

In fact, for the commonly used cross-entropy loss, the above equation also holds. In this case, considering a data point xjx\_{j}, we have

d​f​(θ,xj)d​t\\displaystyle\\frac{df(\\theta,x\_{j})}{dt}

\=d​f​(θ,xj)d​θ⋅d​θd​t\\displaystyle=\\frac{df(\\theta,x\_{j})}{d\\theta}\\cdot\\frac{d\\theta}{dt}

(21)

\=d​f​(θ,xj)d​θ\[−∑i\=1n∂f⁡(θ,xi)∂θ(f(θ,xi)−yi)\]\\displaystyle=\\frac{df(\\theta,x\_{j})}{d\\theta}\\left\[-\\sum\_{i=1}^{n}\\frac{\\partial f(\\theta,x\_{i})}{\\partial\\theta}(f(\\theta,x\_{i})-y\_{i})\\right\]

\=−∑i\=1n⟨d​f​(θ,xj)d​θ,∂f⁡(θ,xi)∂θ⟩(f(θ,xi)−yi).\\displaystyle=-\\sum\_{i=1}^{n}\\left\\langle\\frac{df(\\theta,x\_{j})}{d\\theta},\\frac{\\partial f(\\theta,x\_{i})}{\\partial\\theta}\\right\\rangle(f(\\theta,x\_{i})-y\_{i}).

Given that 𝒪⁡(t)\={f⁡(xi,θ⁡(t))}i\=1n\\mathcal{O}(t)=\\{f(x\_{i},\\theta(t))\\}\_{i=1}^{n} and 𝐲\={yi}i\=1n\\mathbf{y}=\\{y\_{i}\\}\_{i=1}^{n}, we can express Eq. [21](#A2.E21 "Equation 21 ‣ Proof. ‣ B.1 Proof of Theorem ‣ Appendix B Theoretical Analysis ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") in vector form:

d​𝒪​(t)d​t\=−𝒦⋅(𝒪(t)−𝐲),\\frac{d\\mathcal{O}(t)}{dt}=-\\mathcal{K}\\cdot(\\mathcal{O}(t)-\\mathbf{y}), where 𝒦\\mathcal{K} is defined as

𝒦u​v\=⟨∂f⁡(θ,xu)∂θ,∂f⁡(θ,xv)∂θ⟩.\\mathcal{K}\_{uv}=\\left\\langle\\frac{\\partial f(\\theta,x\_{u})}{\\partial\\theta},\\frac{\\partial f(\\theta,{x}\_{v})}{\\partial\\theta}\\right\\rangle.

Q.E.D.

Discussions. Another equivalent definition of NTK given the Jacobian matrix JJ is:

𝒦\=J​J⊤.\\mathcal{K}=JJ^{\\top}.

(22)

This theorem shows that NTK connects the error term 𝒪⁡(t)−y\\mathcal{O}(t)-\\textbf{y} to the changing rate of the output. Therefore, the theory can be used to analyze the training behaviors of neural networks. ∎

### B.2 Proof of Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")

Theorem 3.2. Let {𝒪1​(t),𝒪2​(t),…,𝒪k​(t)}\\{\\mathcal{O}\_{1}(t),\\mathcal{O}\_{2}(t),\\ldots,\\mathcal{O}\_{k}(t)\\} denote the outputs of the neural network function {f1,f2,…,fk}\\{f\_{1},f\_{2},\\ldots,f\_{k}\\} for the kk task at time tt, and let {𝐲1,𝐲2,…,𝐲k}\\{\\mathbf{y}\_{1},\\mathbf{y}\_{2},\\ldots,\\mathbf{y}\_{k}\\} represent the corresponding labels. Then, the ordinary differential equation in Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") gives the following evolution:

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[𝒦11⋯𝒦1​k⋱𝒦k​1⋯𝒦k​k\]⏟𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\],\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-\\underbrace{\\begin{bmatrix}\\mathcal{K}\_{11}&\\cdots&\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\mathcal{K}\_{k1}&\\cdots&\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\widetilde{\\mathcal{K}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix}, where 𝒦i​j∈ℝn×n\\mathcal{K}\_{ij}\\in\\mathbb{R}^{n\\times n} and 𝒦i​j\=𝒦j​i⊤\\mathcal{K}\_{ij}=\\mathcal{K}\_{ji}^{\\top} for 1≤i,j≤k1\\leq i,j\\leq k. The (u,v)(u,v)-th entry of 𝒦i​j\\mathcal{K}\_{ij} is defined as

(𝒦i​j)u​v\=⟨∂fi​(θ,xu)∂θ,∂fj​(θ,xv)∂θ⟩.(\\mathcal{K}\_{ij})\_{uv}=\\left\\langle\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta},\\frac{\\partial f\_{j}(\\theta,x\_{v})}{\\partial\\theta}\\right\\rangle.

###### Proof.

In multi-task learning, a neural network with shared parameters θ\\theta is trained to simultaneously learn kk distinct tasks. In the general case, the overall loss function is defined as

ℒ⁡(θ)\=∑i\=1kℓi​(θ).\\mathcal{L}(\\theta)=\\sum\_{i=1}^{k}\\ell\_{i}(\\theta).

(23)

Similar to the derivation in Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), by utilizing the gradient flow in Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), we can derive:

d​θd​t\\displaystyle\\frac{d\\theta}{dt}

\=−∇θℒ​(θ)\\displaystyle=-\\nabla\_{\\theta}\\mathcal{L}(\\theta)

(24)

\=−∑i\=1k∑u\=1n∂ℓi∂fi​(θ,xu)⋅∂fi​(θ,xu)∂θ\\displaystyle=-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\frac{\\partial\\ell\_{i}}{\\partial f\_{i}(\\theta,x\_{u})}\\cdot\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}

\=−∑i\=1k∑u\=1n∂fi​(θ,xu)∂θ⋅(fi(θ,xu)−yi,u),\\displaystyle=-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\cdot(f\_{i}(\\theta,x\_{u})-y\_{i,u}), where yi,uy\_{i,u} represents the ground truth label of the uu-th element in the label set 𝐲i\\mathbf{y}\_{i}. For a data point xvx\_{v} and the function fjf\_{j} of task jj, we have:

d​fj​(θ,xv)d​t\\displaystyle\\frac{df\_{j}(\\theta,x\_{v})}{dt}

\=d​fj​(θ,xv)d​θ⋅d​θd​t\\displaystyle=\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta}\\cdot\\frac{d\\theta}{dt}

(25)

\=d​fj​(θ,xv)d​θ\[−∑i\=1k∑u\=1n∂fi​(θ,xu)∂θ⋅(fi(θ,xu)−yi,u)\]\\displaystyle=\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta}\\left\[-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\cdot(f\_{i}(\\theta,x\_{u})-y\_{i,u})\\right\]

\=−∑i\=1k∑u\=1n⟨d​fj​(θ,xv)d​θ,∂fi​(θ,xu)∂θ⟩(fi(θ,xu)−yi,u).\\displaystyle=-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\left\\langle\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta},\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\right\\rangle(f\_{i}(\\theta,x\_{u})-y\_{i,u}).

Rewriting Eq. [25](#A2.E25 "Equation 25 ‣ Proof. ‣ B.2 Proof of Theorem ‣ Appendix B Theoretical Analysis ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") in the form of high-dimensional vectors gives:

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[𝒦11⋯𝒦1​k⋱𝒦k​1⋯𝒦k​k\]⏟𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\],\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-\\underbrace{\\begin{bmatrix}\\mathcal{K}\_{11}&\\cdots&\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\mathcal{K}\_{k1}&\\cdots&\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\widetilde{\\mathcal{K}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix}, where 𝒦i​j∈ℝn×n\\mathcal{K}\_{ij}\\in\\mathbb{R}^{n\\times n} and 𝒦i​j\=𝒦j​i⊤\\mathcal{K}\_{ij}=\\mathcal{K}\_{ji}^{\\top} for 1≤i,j≤k1\\leq i,j\\leq k. The (u,v)(u,v)-th entry of 𝒦i​j\\mathcal{K}\_{ij} is defined as

(𝒦i​j)u​v\=⟨∂fi​(θ,xu)∂θ,∂fj​(θ,xv)∂θ⟩.(\\mathcal{K}\_{ij})\_{uv}=\\left\\langle\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta},\\frac{\\partial f\_{j}(\\theta,x\_{v})}{\\partial\\theta}\\right\\rangle.

Q.E.D.

Discussions. It can be observed that Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") essentially extends the foundational NTK theory from Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1 Preliminaries ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") to the MTL scenario. In this case, the input, output, and NTK matrix each acquire an additional dimension, corresponding to the task-oriented dimension. ∎

### B.3 Proof of Proposition [3.3](#S3.Thmtheorem3 "Proposition 3.3. ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")

Proposition 3.3. (Extension of Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective")) Let {𝒪i​(t)}i\=1k\\{\\mathcal{O}\_{i}(t)\\}\_{i=1}^{k}, {𝐲i}i\=1k\\{\\mathbf{y}\_{i}\\}\_{i=1}^{k}, and {𝒦i​j}1≤i,j≤k\\{\\mathcal{K}\_{ij}\\}\_{1\\leq i,j\\leq k} be defined as in Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). We now replace the MTL optimization objective with the weighted form as presented in Eq. [12](#S3.E12 "Equation 12 ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). Consequently, the ordinary differential equation governing the MTL training dynamics in Eq. [7](#S3.E7 "Equation 7 ‣ Theorem 3.2. ‣ 3.2 Extended NTK in Multi-Task Learning ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") becomes:

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[ω12​𝒦11⋯ω1​ωk​𝒦1​k⋱ωk​ω1​𝒦k​1⋯ωk2​𝒦k​k\]⏟𝝎​𝝎⊤⊙𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\].\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-{\\underbrace{\\begin{bmatrix}\\omega\_{1}^{2}\\mathcal{K}\_{11}&\\cdots&\\omega\_{1}\\omega\_{k}\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\omega\_{k}\\omega\_{1}\\mathcal{K}\_{k1}&\\cdots&\\omega\_{k}^{2}\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\boldsymbol{\\omega}\\boldsymbol{\\omega^{\\top}}\\odot\\widetilde{\\mathcal{K}}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix}.

(26)

###### Proof.

Consider the weighted loss

ℒ⁡(θ)\=∑i\=1kωi​ℓi​(θ),\\mathcal{L}(\\theta)=\\sum\_{i=1}^{k}\\omega\_{i}\\ell\_{i}(\\theta),

(27)

the analysis in Appendix [B.2](#A2.SS2 "B.2 Proof of Theorem ‣ Appendix B Theoretical Analysis ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") becomes

d​θd​t\\displaystyle\\frac{d\\theta}{dt}

\=−∇θℒ​(θ)\\displaystyle=-\\nabla\_{\\theta}\\mathcal{L}(\\theta)

(28)

\=−∑i\=1k∑u\=1n∂ωi​ℓi∂fi​(θ,xu)⋅∂fi​(θ,xu)∂θ\\displaystyle=-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\frac{\\partial\\omega\_{i}\\ell\_{i}}{\\partial f\_{i}(\\theta,x\_{u})}\\cdot\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}

\=−∑i\=1kωi∑u\=1n∂fi​(θ,xu)∂θ⋅(fi(θ,xu)−yi,u).\\displaystyle=-\\sum\_{i=1}^{k}\\omega\_{i}\\sum\_{u=1}^{n}\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\cdot(f\_{i}(\\theta,x\_{u})-y\_{i,u}).

Under the action of 𝝎\\boldsymbol{\\omega}, the network output {𝒪1,…,𝒪k}\\{\\mathcal{O}\_{1},\\ldots,\\mathcal{O}\_{k}\\} is transformed to {ω1​f1,…,ωk​fk}\\{\\omega\_{1}f\_{1},\\ldots,\\omega\_{k}f\_{k}\\}. Consider the output of the network for the jj-th task at a data point xvx\_{v}:

d​ωj​fj​(θ,xv)d​t\\displaystyle\\frac{d\\omega\_{j}f\_{j}(\\theta,x\_{v})}{dt}

\=ωj​d​fj​(θ,xv)d​θ⋅d​θd​t\\displaystyle=\\omega\_{j}\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta}\\cdot\\frac{d\\theta}{dt}

(29)

\=ωjd​fj​(θ,xv)d​θ\[−∑i\=1kωi∑u\=1n∂fi​(θ,xu)∂θ⋅(fi(θ,xu)−yi,u)\]\\displaystyle=\\omega\_{j}\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta}\\left\[-\\sum\_{i=1}^{k}\\omega\_{i}\\sum\_{u=1}^{n}\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\cdot(f\_{i}(\\theta,x\_{u})-y\_{i,u})\\right\]

\=−∑i\=1k∑u\=1nωjωi⟨d​fj​(θ,xv)d​θ,∂fi​(θ,xu)∂θ⟩(fi(θ,xu)−yi,u).\\displaystyle=-\\sum\_{i=1}^{k}\\sum\_{u=1}^{n}\\omega\_{j}\\omega\_{i}\\left\\langle\\frac{df\_{j}(\\theta,x\_{v})}{d\\theta},\\frac{\\partial f\_{i}(\\theta,x\_{u})}{\\partial\\theta}\\right\\rangle(f\_{i}(\\theta,x\_{u})-y\_{i,u}).

Then we have

\[d​𝒪1​(t)d​td​𝒪k​(t)d​t\]\=−\[ω12​𝒦11⋯ω1​ωk​𝒦1​k⋱ωk​ω1​𝒦k​1⋯ωk2​𝒦k​k\]⏟𝝎​𝝎⊤⊙𝒦\~​\[𝒪1​(t)−𝐲1𝒪k​(t)−𝐲k\]\\begin{bmatrix}\\frac{d\\mathcal{O}\_{1}(t)}{dt}\\\\ \\vdots\\\\ \\frac{d\\mathcal{O}\_{k}(t)}{dt}\\end{bmatrix}=-{\\underbrace{\\begin{bmatrix}\\omega\_{1}^{2}\\mathcal{K}\_{11}&\\cdots&\\omega\_{1}\\omega\_{k}\\mathcal{K}\_{1k}\\\\ \\vdots&\\ddots&\\vdots\\\\ \\omega\_{k}\\omega\_{1}\\mathcal{K}\_{k1}&\\cdots&\\omega\_{k}^{2}\\mathcal{K}\_{kk}\\end{bmatrix}}\_{\\boldsymbol{\\omega}\\boldsymbol{\\omega^{\\top}}\\odot\\widetilde{\\mathcal{K}}}}\\begin{bmatrix}\\mathcal{O}\_{1}(t)-\\mathbf{y}\_{1}\\\\ \\vdots\\\\ \\mathcal{O}\_{k}(t)-\\mathbf{y}\_{k}\\end{bmatrix}

Q.E.D.

Discussions. Proposition [3.3](#S3.Thmtheorem3 "Proposition 3.3. ‣ 3.3 The Proposed NTKMTL ‣ 3 Method ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") actually provides an intuition: by adjusting the relative magnitudes of {ωi}i\=1k\\{\\omega\_{i}\\}\_{i=1}^{k}, one can alter the eigenvalue distribution of the NTK matrix, thereby balancing convergence speeds. This serves as the foundation for the design of NTKMTL and NTKMTL-SR. ∎

## Appendix C Detailed Experimental Results

### C.1 Detailed Results on QM9

For QM9 benchmark, previous methods were all implemented based on a shared codebase \[[39](#bib.bib39 ""), [31](#bib.bib31 ""), [4](#bib.bib4 "")\], utilizing the message-passing neural network (MPNN) architecture \[[19](#bib.bib19 "")\]. All the methods were trained for 300 epochs. However, we found that the hyperparameter settings in this codebase were suboptimal (specifically, the improper learning rate scheduler makes the learning rate decay too quickly). Consequently, the reported results of most previous methods had not fully converged, potentially leading to unfair comparisons. Therefore, we readjusted the hyperparameters: we changed the batch size from 120 to 60 and the learning rate scheduler’s patience from 5 to 10. The remaining hyperparameters were kept unchanged, consistent with the original codebase. Subsequently, we reproduced all baselines that had reported results on QM9, and their performance on various tasks significantly improved under these revised settings. To ensure accurate evaluation, we also reproduced the 11-task STL baselines under the same settings.

Experimental results are presented in Table [6](#A3.T6 "Table 6 ‣ C.1 Detailed Results on QM9 ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). For each baseline, the upper row shows results taken from its original paper, while the bottom row (highlighted in ) presents results reproduced by us under new hyperparameter settings. It can be seen that without changing the model architecture, solely by adjusting the learning rate schedule, the performance of all baselines on the 11 tasks has significantly improved, notably extending the Pareto front.

Under the new settings, with both STL baselines and MTL methods showing significant performance improvements, we observed different behaviors among previous methods. The final Δ​m%\\Delta m\\% of some methods (e.g., LS, RLW, PCGrad, CAGrad) were largely consistent with their originally reported values. For other methods (e.g., SI, UW, FAMO, GO4Align), the Δ​m%\\Delta m\\% significantly improved compared to the reported results. This indicates that the previous hyperparameter settings fail to fully exhibit the capabilities of these methods. Under the new settings that better ensure convergence, their performance shows further improvement. Notably, the traditional Scale-Invariant Linear Scalarization (SI) demonstrated extremely superior performance, surpassing the vast majority of recent baselines. This suggests that in cases where differences in loss scales among different tasks are too large, directly eliminating scale differences through logarithmic methods may be an effective solution.

In summary, we identified that the parameter settings in the previous code implementation hindered the smooth convergence of both STL and MTL methods, rendering comparisons made under these conditions unfair. Our experiments verified that this issue can be resolved by simply adjusting the batch size and learning rate scheduler parameters. Under the new settings that better ensure convergence, the performance of both STL methods and numerous MTL methods on the 11 tasks has significantly improved. We believe that fair and transparent reproduction of all previous baseline methods under such parameter settings enables more reasonable comparisons on this benchmark, providing new results that are valuable to the MTL community.

Table 6: Results on QM9 (11-task) dataset. For each baseline, the upper row shows results taken from its original paper, while the bottom row (highlighted in ) presents results reproduced by us under new hyperparameter settings.

 Method μ\\mu α\\alpha ϵH​O​M​O\\epsilon\_{HOMO} ϵL​U​M​O\\epsilon\_{LUMO} ⟨R2⟩\\langle R^{2}\\rangle ZPVE U0U\_{0} UU HH GG cvc\_{v} MR↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} MAE ↓\\downarrow STL 0.067 0.181 60.57 53.91 0.502 4.53 58.8 64.2 63.8 66.2 0.072 STL 0.060 0.156 60.54 51.22 0.419 3.08 39.3 42.9 41.7 43.1 0.061 LS 0.106 0.325 73.57 89.67 5.19 14.06 143.4 144.2 144.6 140.3 0.128 177.6 LS 0.077 0.253 55.95 68.59 4.163 11.08 109.2 109.8 110.1 106.7 0.099 9.82 179.8 SI 0.309 0.345 149.8 135.7 1.00 4.50 55.3 55.75 55.82 55.27 0.112 77.8 SI 0.159 0.242 109.6 96.80 0.732 3.308 34.35 34.37 34.33 35.16 0.081 5.27 39.7 RLW \[[28](#bib.bib28 "")\] 0.113 0.340 76.95 92.76 5.86 15.46 156.3 157.1 157.6 153.0 0.137 203.8 RLW \[[28](#bib.bib28 "")\] 0.090 0.277 62.49 76.39 4.948 12.54 124.8 124.9 125.0 122.1 0.115 12.09 222.6 DWA \[[35](#bib.bib35 "")\] 0.107 0.325 74.06 90.61 5.09 13.99 142.3 143.0 143.4 139.3 0.125 175.3 DWA \[[35](#bib.bib35 "")\] 0.078 0.239 55.17 67.40 3.992 10.92 107.4 108.1 108.3 105.4 0.098 8.73 173.0 UW \[[25](#bib.bib25 "")\] 0.386 0.425 166.2 155.8 1.06 4.99 66.4 66.78 66.80 66.24 0.122 108.0 UW \[[25](#bib.bib25 "")\] 0.194 0.274 120.6 102.8 0.763 3.698 41.11 41.13 41.16 41.75 0.089 8.27 58.4 MGDA \[[46](#bib.bib46 "")\] 0.217 0.368 126.8 104.6 3.22 5.69 88.37 89.4 89.32 88.01 0.120 120.5 MGDA \[[46](#bib.bib46 "")\] 0.154 0.266 95.20 67.51 3.088 4.468 49.38 49.21 49.62 49.69 0.087 8.36 101.4 PCGrad \[[58](#bib.bib58 "")\] 0.106 0.293 75.85 88.33 3.94 9.15 116.36 116.8 117.2 114.5 0.110 125.7 PCGrad \[[58](#bib.bib58 "")\] 0.078 0.221 59.14 67.82 2.937 6.691 88.24 88.65 88.85 87.36 0.084 7.91 118.6 CAGrad \[[32](#bib.bib32 "")\] 0.118 0.321 83.51 94.81 3.21 6.93 113.99 114.3 114.5 112.3 0.116 112.8 CAGrad \[[32](#bib.bib32 "")\] 0.083 0.234 57.80 70.98 2.718 5.352 76.47 76.93 77.05 76.32 0.089 8.27 102.4 Nash-MTL \[[39](#bib.bib39 "")\] 0.102 0.248 82.95 81.89 2.42 5.38 74.5 75.02 75.10 74.16 0.093 62.0 Nash-MTL \[[39](#bib.bib39 "")\] 0.086 0.218 69.78 66.18 2.153 4.679 59.63 59.94 59.98 59.97 0.082 6.91 72.9 FAMO \[[31](#bib.bib31 "")\] 0.15 0.30 94.0 95.2 1.63 4.95 70.82 71.2 71.2 70.3 0.10 58.5 FAMO \[[31](#bib.bib31 "")\] 0.128 0.230 98.09 84.42 0.859 3.541 40.24 40.57 40.62 40.21 0.081 5.91 38.9 FairGrad \[[4](#bib.bib4 "")\] 0.117 0.253 87.57 84.00 2.15 5.07 70.89 71.17 71.21 70.88 0.095 57.9 FairGrad \[[4](#bib.bib4 "")\] 0.109 0.208 81.74 72.82 1.669 3.418 51.31 51.67 51.72 51.97 0.079 6.64 57.0 GO4Align \[[48](#bib.bib48 "")\] 0.17 0.35 102.4 119.0 1.22 4.94 53.9 54.3 54.3 53.9 0.11 52.7 GO4Align \[[48](#bib.bib48 "")\] 0.113 0.314 74.46 91.04 0.912 3.632 36.06 36.38 36.41 36.58 0.104 6.64 40.5 NTKMTL 0.091 0.212 70.97 70.81 2.113 3.835 44.18 44.56 44.53 44.38 0.077 5.91 56.7 NTKMTL-SR 0.081 0.207 75.95 69.10 1.176 3.689 40.14 40.46 40.48 40.49 0.074 4.00 30.7 

### C.2 Detailed Results with Standard Errors

In this section, We provide the detailed experimental results with standard errors for our method, and the results for the baseline methods are taken from their original papers. Since results for CelebA are not reported by some methods \[[47](#bib.bib47 ""), [54](#bib.bib54 ""), [48](#bib.bib48 "")\], these methods are excluded when presenting combined CityScapes and CelebA results in Table [2](#S4.T2 "Table 2 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). Table [7](#A3.T7 "Table 7 ‣ C.2 Detailed Results with Standard Errors ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") provides detailed results solely on the CityScapes dataset, including these methods. Consequently, the mean rank (MR) in Table [7](#A3.T7 "Table 7 ‣ C.2 Detailed Results with Standard Errors ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") slightly differs from that in Table [2](#S4.T2 "Table 2 ‣ 4.2 Multi-Task Supervised Learning ‣ 4 Experiments ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective").

On NYUv2 and CityScapes, we follow the training settings of \[[39](#bib.bib39 ""), [4](#bib.bib4 "")\], including data augmentation for all compared methods. Training runs for 200 epochs, with the learning rate initialized at 10−410^{-4} and reduced to 5×10−55\\times 10^{-5} after 100 epochs. The architecture is the SegNet-based \[[3](#bib.bib3 "")\] Multi-Task Attention Network (MTAN) \[[33](#bib.bib33 "")\]. Batch sizes are 2 (NYUv2) and 8 (CityScapes), and the hyparameter nn for NTKMTL-SR on NYUv2 is set to 2. Our setup for the CelebA benchmark aligns with the configuration detailed in \[[31](#bib.bib31 "")\]. We employ a 99-layer CNN as the network backbone, coupled with separate linear layers for each task. The method is trained for 1515 epochs; optimization is carried out using Adam with a batch size of 256256.

Table 7: Detailed results on CityScapes (2-task) dataset. Each experiment is repeated 3 times with different random seeds and the average is reported. The best scores are reported in .

 Method CityScapes Segmentation Depth MR ↓\\downarrow 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} mIoU ↑\\uparrow Pix Acc ↑\\uparrow Abs Err ↓\\downarrow Rel Err ↓\\downarrow STL 74.01 93.16 0.0125 27.77 LS 75.18 93.49 0.0155 46.77 10.25 22.60 SI 70.95 91.73 0.0161 33.83 14.00 14.11 RLW \[[28](#bib.bib28 "")\] 74.57 93.41 0.0158 47.79 12.25 24.38 DWA \[[35](#bib.bib35 "")\] 75.24 93.52 0.0160 44.37 9.75 21.45 UW \[[25](#bib.bib25 "")\] 72.02 92.85 0.0140 30.13 10.00 5.89 MGDA \[[46](#bib.bib46 "")\] 68.84 91.54 0.0309 33.50 14.75 44.14 PCGrad \[[58](#bib.bib58 "")\] 75.13 93.48 0.0154 42.07 10.50 18.29 CAGrad \[[32](#bib.bib32 "")\] 75.16 93.48 0.0141 37.60 9.25 11.64 IMTL-G \[[33](#bib.bib33 "")\] 75.33 93.49 0.0135 38.41 7.25 11.10 Nash-MTL \[[39](#bib.bib39 "")\] 75.41 93.66 0.0129 35.02 4.75 6.82 FAMO \[[31](#bib.bib31 "")\] 74.54 93.29 0.0145 32.59 9.25 8.13 Aligned-MTL \[[47](#bib.bib47 "")\] 75.77 93.69 0.0133 32.66 3.00 5.27 SDMGrad \[[54](#bib.bib54 "")\] 74.53 93.52 0.0137 34.01 8.25 7.74 GO4Align \[[48](#bib.bib48 "")\] 72.63 93.03 0.0164 27.58 10.75 8.13 FairGrad \[[4](#bib.bib4 "")\] 75.72 93.68 0.0134 32.25 3.25 5.18 NTKMTL 73.71 92.71 0.0136 27.21 8.50 1.92 NTKMTL-SR 72.58 92.93 0.0124 31.65 8.00 3.84 

Table 8: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean and stderr are reported. 

| Method          |                    |                                            |                                            |                |                |
| --------------- | ------------------ | ------------------------------------------ | ------------------------------------------ | -------------- | -------------- |
| Segmentation    | Depth              | 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} | 𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow} |                |                |
| mIoU ↑\\uparrow | Pix Acc ↑\\uparrow | Abs Err ↓\\downarrow                       | Rel Err ↓\\downarrow                       |                |                |
| 73.71           | 92.71              | 0.0136                                     | 27.21                                      | 1.92           | -0.77          |
| ±0.17\\pm 0.17  | ±0.15\\pm 0.15     | ±0.0005\\pm 0.0005                         | ±0.23\\pm 0.23                             | ±0.30\\pm 0.30 | ±0.37\\pm 0.37 |
| 72.58           | 92.93              | 0.0124                                     | 31.65                                      | 3.84           | 0.23           |
| ±0.32\\pm 0.32  | ±0.23\\pm 0.23     | ±0.0004\\pm 0.0004                         | ±0.35\\pm 0.35                             | ±0.37\\pm 0.37 | ±0.46\\pm 0.46 |

Table 9: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean and stderr are reported. 

Segmentation

Depth

Surface Normal

Method

mIoU ↑\\uparrow

Pix Acc ↑\\uparrow

Abs Err ↓\\downarrow

Rel Err ↓\\downarrow

Angle Dist ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

𝚫​𝒎%↓\\boldsymbol{\\Delta m\\%\\downarrow}

Mean

Median

11.25

22.5

30

NTKMTL (mean)

39.68

65.43

0.5296

0.2168

24.24

18.63

30.74

58.72

70.78

-6.99

NTKMTL (stderr)

±0.51\\pm 0.51

±0.24\\pm 0.24

±0.0008\\pm 0.0008

±0.0014\\pm 0.0014

±0.07\\pm 0.07

±0.09\\pm 0.09

±0.14\\pm 0.14

±0.19\\pm 0.19

±0.19\\pm 0.19

±0.38\\pm 0.38

NTKMTL-SR (mean)

40.23

65.28

0.5261

0.2136

24.88

19.58

29.53

56.67

69.08

-5.35

NTKMTL-SR (stderr)

±0.42\\pm 0.42

±0.26\\pm 0.26

±0.0013\\pm 0.0013

±0.0014\\pm 0.0014

±0.11\\pm 0.11

±0.13\\pm 0.13

±0.12\\pm 0.12

±0.15\\pm 0.15

±0.16\\pm 0.16

±0.31\\pm 0.31

### C.3 Visualization experiments for the NTK eigenvalues during training

To support the proposed theory, we conducted experiments on NYUv2, visualizing the change in the maximum eigenvalue of the NTK matrix for the three tasks during training under linear scalarization (i.e., equal weighting). On the NYUv2 dataset, the difficulty levels of the three tasks show significant variation. Previous methods generally outperform the Single Task Learning (STL) baseline in segmentation and depth estimation tasks, but almost all of them consistently underperform STL on the surface normal prediction task, leading to a significant task imbalance in the overall results.

Figure 2: Visualization for the NTK eigenvalues during training.

As shown in Fig. [2](#A3.F2 "Figure 2 ‣ C.3 Visualization experiments for the NTK eigenvalues during training ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"), throughout training, the largest eigenvalue corresponding to surface normal prediction remains substantially smaller than those of segmentation and depth estimation. Given that the maximum NTK eigenvalue reflects a task’s convergence speed, this observation aligns with the empirical finding that many existing MTL algorithms struggle to converge on surface normal prediction. Although some prior methods (e.g., MGDA) attempt to prioritize the most difficult tasks, their performance in surface normal prediction tasks remains unsatisfactory due to the challenge in accurately quantifying the "difficulty" and "convergence speed" of different tasks. In contrast, by leveraging NTK theory to accurately characterize and balance the convergence speed of each task during training, NTKMTL delivers SOTA results on surface normal prediction and is one of only two methods that outperform single-task learning on all three tasks.

### C.4 Ablation Study on the Hyperparameter nn

For NTKMTL-SR, the computational cost of calculating the NTK is minimal, allowing us to further investigate the impact of varying mini-batch sizes nn on the results. Therefore, we set nn to \[1,2,3,4,6\]\[1,2,3,4,6\] and conduct an ablation study on the QM9 (11-task) benchmark. For each value of nn, we conduct 3 repeated experiments with different random seeds and calculate the mean and variance for Δ​m%\\Delta m\\%. The results are shown in Fig. [3](#A3.F3 "Figure 3 ‣ C.4 Ablation Study on the Hyperparameter 𝑛 ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective"). When n\=1n=1, the performance of NTKMTL-SR is comparable to that of NTKMTL. However, when increasing nn from 1 to 2 or more, NTKMTL-SR shows a noticeable improvement in performance, accompanied by a reduction in the variance of performance across repeated experiments. We attribute this to the fact that increasing the number of mini-batches leads to a larger NTK matrix dimension, which in turn reduces stochastic error and allows our method to more accurately characterize the convergence speed of the tasks.

Figure 3: Ablation study on hyperparameter nn on QM9. Each experiment is repeated over 3 random seeds, and the mean and stderr are reported.

However, we also find that the results for n\=4n=4 and n\=6n=6 are almost identical, and the performance differences observed were potentially weaker than the inherent variability stemming from different random seeds. Concurrently, Fig. [4](#A3.F4 "Figure 4 ‣ C.4 Ablation Study on the Hyperparameter 𝑛 ‣ Appendix C Detailed Experimental Results ‣ NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective") visualizes the training time per epoch for various nn values on QM9. Despite only requiring the computation of the maximum eigenvalue of the NTK matrix with respect to zz, training a single epoch when n\=6n=6 already approached 1.71.7 times the duration of the LS method. Overall, we posit that the selection of hyperparameter nn is a trade-off between performance and training speed, and n\=4n=4 generally presents a favorable compromise.

Figure 4: Training time per epoch for LS and NTKMTL-SR (with different hyperparameter nn) on QM9.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")