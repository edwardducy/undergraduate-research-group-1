# NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective

Xiaohan Qin, Xiaoxing Wang, Ning Liao, Junchi Yan<sup>‡</sup>

School of CS & School of AI, Shanghai Jiao Tong University

Shanghai Innovation Institute

## Abstract

Multi-Task Learning (MTL) enables a single model to learn multiple tasks simultaneously, leveraging knowledge transfer among tasks for enhanced generalization, and has been widely applied across various domains. However, task imbalance remains a major challenge in MTL. Although balancing the convergence speeds of different tasks is an effective approach to address this issue, it is highly chal lenging to accurately characterize the training dynamics and convergence speeds of multiple tasks within the complex MTL system. To this end, we attempt to analyze the training dynamics in MTL by leveraging Neural Tangent Kernel (NTK) theory and propose a new MTL method, NTKMTL. Specifically, we introduce an extended NTK matrix for MTL and adopt spectral analysis to balance the convergence speeds of multiple tasks, thereby mitigating task imbalance. Based on the approximation via shared representation, we further propose NTKMTL-SR, achieving training efficiency while maintaining competitive performance. Extensive experiments demonstrate that our methods achieve state-of-the-art performance across a wide range of benchmarks, including both multi-task supervised learning and multi-task reinforcement learning. Source code is available at https://github.com/jianke0604/NTKMTL.

## 1 Introduction

Multi-task learning (MTL) [9, 20, 56, 45, 61] involves training a single model to address multiple tasks concurrently. This approach enables sharing information and representations across tasks, enhancing the model’s generalization capabilities and boosting performance on individual tasks [6, 50, 38]. MTL is particularly advantageous in scenarios with limited computational resources, as it eliminates the need to maintain separate models for each task. Its utility spans a wide range of domains, including computer vision [1, 63, 35], natural language processing [10, 34, 40], and robotics [16, 55]. Despite these benefits, MTL encounters a significant challenge known as task imbalance, where certain tasks dominate the training process while others suffer from insufficient optimization. Previous studies [4, 48] have indicated that achieving more balanced optimization across tasks often leads to improved overall performance. Addressing such a task imbalance issue necessitates the development of sophisticated optimization strategies to ensure that all tasks benefit equitably from the shared model parameters.

One widely adopted perspective to address the above issue is to balance the convergence speeds of different tasks [32, 60, 35]. Nevertheless, it is highly challenging to accurately analyze the training dynamics and convergence speeds of multiple tasks within the complex MTL system. Most prior methods [35, 31] approximate the convergence speeds based on the difference or ratio between consecutive loss values. However, different tasks exhibit vastly different loss scales and heterogeneous ultimate loss minima, thus such a simple approximation fails to accurately capture a task’s convergence capability or potential at a specific training stage. As demonstrated by experimental results on widely used benchmarks such as NYUv2, these methods still exhibit considerable task imbalance. Therefore, there is a pressing need for a tool to characterize MTL training dynamics and task convergence properties with a robust theoretical foundation.

To this end, we attempt to analyze the training dynamics within MTL systems by leveraging Neural Tangent Kernel (NTK) theory[17, 2, 7], which provides insights into the optimization trajectory of deep neural networks and has demonstrated its theoretical efficacy in single-task learning (STL) scenarios. From the perspective of NTK theory, the convergence speed of a neural network can be characterized by the eigenvalues of its corresponding NTK matrix. Specifically, lower-frequency components of the target function typically correspond to larger NTK eigenvalues, which converge faster [7, 52]. In contrast, higher-frequency components often correspond to smaller NTK eigenvalues, which converge more slowly (or are harder to learn). This phenomenon is known as "spectral bias" in the context of single-task learning [22, 43, 53], which bears a strong resemblance to task imbalance in MTL. As explained by the NTK theory, such a distinction in the training dynamics provides a foundational understanding of why certain aspects of the target function are prioritized over others during the training process. However, despite its potential relevance, the application of NTK theory to the field of MTL has been scarcely explored by prior work.

Building upon the above motivation, this work applies the NTK theory to MTL scenarios and introduces an extended NTK matrix to jointly characterize the training dynamics of multiple tasks. Specifically, the target function in MTL is explicitly decomposed into multiple distinct components, each associated with a different task. Our theoretical analysis shows that, the convergence speed of the overall training error is jointly influenced by the NTK matrices corresponding to each task, and tasks associated with larger NTK eigenvalues can be learned more rapidly by the network, subsequently dominating other tasks and resulting in unsatisfactory performance on the remaining tasks. To address this issue, we propose a new MTL approach, NTKMTL, which assigns appropriate weights during training based on the NTK analysis of each task. This method effectively balances the convergence speeds of different tasks and, consequently, reduces task imbalance. In summary, our contributions can be outlined as follows:

1) We introduce a new perspective on understanding task imbalance in MTL by leveraging NTK theory for analysis. Under this perspective, multiple tasks are viewed as distinct components of the training objective, each characterized by unique NTK spectral properties. These inherent differences in spectral characteristics lead to significant disparities in convergence speeds across tasks, subsequently causing task imbalance.

2) Based on the NTK spectral analysis of different tasks in MTL, we propose a new MTL method, NTKMTL. Both theoretical analysis and experimental results demonstrate that NTKMTL effectively addresses the ill-conditioned distribution of NTK eigenvalues during MTL training, thereby alleviating task imbalance.

3) To enhance the practical applicability and computational efficiency, we further introduce NTKMTL-SR, an efficient approximation that leverages the Shared Representation in MTL. NTKMTL-SR requires only a single gradient backpropagation per iteration for shared parameters, which not only provides computational efficiency but also maintains competitive performance.

4) Extensive experiments validate that our methods achieve state-of-the-art performance across a wide range of benchmarks. The experimental scenarios include both multi-task supervised learning and multi-task reinforcement learning, with task numbers ranging from 2 to 40.

## 2 Related Work

Multi-Task Learning. In multi-task learning (MTL), previous approaches can be broadly classified into two categories: loss-oriented and gradient-oriented methods. Loss-oriented methods primarily aim to address convergence discrepancies arising from differences in loss scales, thereby mitigating task imbalance. These methods often exhibit training efficiency as they only require one backpropagation on the aggregated loss. Approaches include Linear Scalarization (LS), Scale-Invariant (SI), homoscedastic uncertainty weighting [24], dynamic weight averaging [35], self-paced learning [37], geometric loss [14], random loss weighting [29], impartial loss weighting [33], fast adaptive optimization [31], and multi-task grouping for alignment [48]. On the other hand, gradient oriented methods [11] focus on resolving the gradient conflicts issue among shared parameters, seeking the most favorable updating vector to alleviate task imbalance. These types of methods often achieve better performance due to directly obtaining the gradients for all tasks for optimization. Notable approaches in this category include Multiple Gradient Descent Algorithm [46], gradient normalization [12], gradient conflicts projection [57], gradient sign dropout [13], impartial gradient weighting [33], conflict-averse gradients [32], gradient similarity regularisation [51], dual balancing [27], stochastic direction-oriented update [54], smooth tchebycheff scalarization [30], independent gradient alignment [47], Nash bargaining solution [39], fair resource allocation [4], performance informed variance reduction approach [41], and consistent multi-task learning with task-specific parameters [42].

Neural Tangent Kernel theory. Recent theoretical advancements have modeled neural networks in the limits of infinite width and infinitesimal learning rate as kernel regression using the Neural Tangent Kernel (NTK) [17, 2, 7, 22]. Specifically, analyses by [26] and [2] demonstrate that, during gradient descent, the outputs of a neural network remain close to those of a linear dynamical system, with the convergence speed governed by the eigenvalues of the NTK matrix [5, 7, 52, 53]. The NTK’s eigendecomposition reveals that its eigenvalue spectrum decays rapidly as a function of frequency, which explains the well-documented "spectral bias" of deep networks towards learning low-frequency functions [22, 43, 53]. Such spectral bias phenomenon bears a strong resemblance to task imbalance in MTL, which is an effective application of NTK theory in explaining neural network training dynamics in STL scenarios. Building on this observation, this paper extends NTK theory to MTL and proposes a solution to address the task imbalance issue.

## 3 Method

## 3.1 Preliminaries

To establish the foundation for our theoretical investigation, we first review the traditional Neural Tangent Kernel theory that explores the training dynamics of deep neural networks. In the subsequent sections, we extend these theoretical frameworks to the multi-task learning context and develop specific methodologies accordingly.

Neural Tangent Kernel. For a deep neural network f with parameters θ and training dataset $( { \bf x } , { \bf y } ) = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , the Neural Tangent Kernel (NTK) K is defined as

$$
\mathcal {K} _ {u v} = \left\langle \frac {\partial f (\theta , x _ {u})}{\partial \theta}, \frac {\partial f (\theta , x _ {v})}{\partial \theta} \right\rangle . \tag {1}
$$

Prior works [22, 26, 2] have demonstrated that under certain conditions (e.g., when the learning rate η approaches zero), the training dynamics of the neural network can be characterized via gradient flow, which is governed by the following ordinary differential equation (ODE) [44]:

$$
\frac {d \theta (t)}{d t} = - \nabla \mathcal {L} (\theta). \tag {2}
$$

This leads to the following theorem:

Theorem 3.1. Let $\mathcal { O } ( t ) = \{ f ( \theta , x _ { i } ) \} _ { i = 1 } ^ { n }$ be the outputs ofthe neural network at time $t . \ \mathbf { x } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ is the input data, and $\mathbf { y } = \{ y _ { i } \} _ { i = 1 } ^ { n }$ is the corresponding label, Then O(t)follows this evolution:

$$
\frac {d \mathcal {O} (t)}{d t} = - \mathcal {K} \cdot (\mathcal {O} (t) - \mathbf {y}). \tag {3}
$$

Detailed analysis can be found in the Appendix. Eq. 3 enables us to utilize the neural tangent kernel to analyze the training dynamics of neural networks. Prior work [22] demonstrated that as the network width approaches infinity, the NTK K remains approximately constant during training. A more widely used result is that during the training of deep neural networks, the kernel function is updated much more slowly than the network’s output [52, 62]. Therefore, Eq. 3 can also be interpreted as an ODE and provides the following approximation:

$$
\mathcal {O} (t) \approx (\mathbf {I} - e ^ {- \eta \mathcal {K} t}) \mathbf {y}. \tag {4}
$$

Spectral analysis in neural network training. Let us consider the training error ${ \mathcal { O } } ( t ) - \mathbf { y }$ . Since the NTK matrix must be positive semi-definite, we can take its spectral decomposition $\dot { \kappa } = Q \Lambda Q ^ { \top }$ where $Q$ is an orthogonal matrix and $\Lambda$ is a diagonal matrix whose entries are the eigenvalues λ of $\kappa$ Then, since $e ^ { - \eta \kappa t } = Q e ^ { - \eta \Lambda t } Q ^ { \top }$ , we have

$$
Q ^ {\top} (\mathcal {O} (t) - \mathbf {y}) \approx - Q ^ {\top} e ^ {- \eta \mathcal {K} t} \mathbf {y} = - e ^ {- \eta \Lambda t} Q ^ {\top} \mathbf {y}. \tag {5}
$$

This implies that, when considering the convergence of training in the NTK eigenbasis, the i-th component of the absolute error $\left| Q ^ { \dagger } ( \mathcal { O } ( t ) - y ) \right| _ { i }$ decays at an approximate exponential rate of $\eta \lambda _ { i }$ In other words, the components of the target function corresponding to kernel eigenvectors with larger eigenvalues are learned more rapidly, resulting in the high-frequency components of the target function converging exceedingly slowly, to the extent that the neural network is difficult to learn these components.

## 3.2 Extended NTK in Multi-Task Learning

In multi-task learning, a neural network with shared parameters $\theta$ is trained to simultaneously learn k distinct tasks. In the general case, the overall loss function is defined as

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {k} \ell_ {i} (\theta). \tag {6}
$$

In this context, by leveraging the gradient flow defined in Eq. 2, Theorem 3.1 can be further extended as follows:

Theorem 3.2. Let $\{ \mathcal { O } _ { 1 } ( t ) , \mathcal { O } _ { 2 } ( t ) , \ldots , \mathcal { O } _ { k } ( t ) \}$ denote the outputs of the neural network function $\{ f _ { 1 } , f _ { 2 } , \ldots , f _ { k } \}$ for the k tasks at time t, and let $\left\{ \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { k } \right\}$ represent the corresponding labels. Then, the ordinary differential equation in Eq. 2 gives the following evolution:

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \mathcal {K} _ {1 1} & \cdots & \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \mathcal {K} _ {k 1} & \cdots & \mathcal {K} _ {k k} \end{array} \right]} _ {\widetilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right], \tag {7}
$$

where $\mathcal { K } _ { i j } \in \mathbb { R } ^ { n \times n }$ and $\begin{array} { r } { \mathcal { K } _ { i j } = \mathcal { K } _ { j i } ^ { \top } f o r \mathrm { 1 } \le i , j \le k . } \end{array}$ . The $( u , v )$ -th entry of $\mathcal { K } _ { i j }$ is defined as

$$
(\mathcal {K} _ {i j}) _ {u v} = \left\langle \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta}, \frac {\partial f _ {j} (\theta , x _ {v})}{\partial \theta} \right\rangle . \tag {8}
$$

Detailed proof can be found in the Appendix. We define the large NTK matrix, formed by the training dynamics of the k tasks in Theorem 3.2, as the extended NTK matrix in MTL, denoted as $\widetilde { \kappa }$ It is straightforward to observe that $\kappa _ { i i }$ for $1 \leq i \leq k ,$ , as well as $\widetilde { \kappa }$ itself, are positive semi-definite matrices. In fact, let $J _ { i }$ denote the Jacobian matrix of $f _ { i }$ with respect to $\theta ,$ , then we can observe that:

$$
\mathcal {K} _ {i i} = J _ {i} J _ {i} ^ {\top}, \text {and} \widetilde {\mathcal {K}} = \left[ \begin{array}{c} J _ {1} \\ \vdots \\ J _ {k} \end{array} \right] \left[ \begin{array}{l l l} J _ {1} ^ {\top} & \dots & J _ {k} ^ {\top} \end{array} \right]. \tag {9}
$$

Analogous to the analysis presented in Sec. 3.1, we adopt the approximation derived from the theoretical framework of [22]. Consequently, Eq. 4 is extended to the multi-task learning scenarios as follows:

$$
\left[ \begin{array}{c} \mathcal {O} _ {1} (t) \\ \vdots \\ \mathcal {O} _ {k} (t) \end{array} \right] \approx (\mathbf {I} - e ^ {- \eta \widetilde {\mathcal {K}} t}) \left[ \begin{array}{c} \mathbf {y} _ {1} \\ \vdots \\ \mathbf {y} _ {k} \end{array} \right]. \tag {10}
$$

As mentioned above, the extended NTK matrix is positive semi-definite. Therefore, similarly to previous steps, we perform its spectral decomposition as $\widetilde { \cal K } = \widetilde { Q } \widetilde { \Lambda } \widetilde { Q } ^ { \top }$ . Consequently, the training error MTL can be approximated by the following evolution:

$$
\widetilde {Q} ^ {\top} \left(\left[ \begin{array}{c} \mathcal {O} _ {1} (t) \\ \vdots \\ \mathcal {O} _ {k} (t) \end{array} \right] - \left[ \begin{array}{c} \mathbf {y} _ {1} \\ \vdots \\ \mathbf {y} _ {k} \end{array} \right]\right) \approx - e ^ {- \eta \widetilde {\Lambda} t} \widetilde {Q} ^ {\top} \left[ \begin{array}{c} \mathbf {y} _ {1} \\ \vdots \\ \mathbf {y} _ {k} \end{array} \right]. \tag {11}
$$

Eq. 11 illustrates that in the MTL scenario, where the learning objectives are explicitly partitioned into k components, the training dynamics can still be interpreted through spectral analysis of the extended NTK matrix. The substantial disparities in the distribution of NTK eigenvalues across different tasks give rise to bias in convergence speed, causing the network to be dominated by certain specific tasks and hindering the effective simultaneous learning of all tasks.

## 3.3 The Proposed NTKMTL

Theoretical analysis in Sec. 3.2 and experimental results on extensive benchmarks indicate that the traditional multi-task optimization objective in Eq. 6 yields unsatisfactory performance, exhibiting significant task imbalance across various task scenarios. Therefore, existing MTL methods often consider obtaining a weigh $\pmb { \omega } = ( \omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { k } ) ^ { \top }$ to optimize the weighted objective instead:

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {k} \omega_ {i} \ell_ {i} (\theta). \tag {12}
$$

Specifically, the final gradient is $\begin{array} { r } { g _ { s } = \sum _ { i = 1 } ^ { k } \omega _ { i } g _ { i } } \end{array}$ , where $g _ { i }$ represents the gradient of the i-th task’s loss $\ell _ { i }$ with respect to the shared parameters, and $g _ { s }$ is the final aggregated gradient. Under this formulation, the analysis of the extended NTK matrix for MTL systems presented in Sec. 3.2 is correspondingly modified. Therefore, we further introduce the following proposition:

Proposition 3.3. (Extension of Theorem 3.2) Let $\{ \mathcal { O } _ { i } ( t ) \} _ { i = 1 } ^ { k } , ~ \{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { k } ,$ and $\{ { K } _ { i j } \} _ { 1 \leq i , j \leq k }$ be defined as in Theorem 3.2. We now replace the MTL optimization objective with the weightedform as presented in Eq. 12. Consequently, the ordinary differential equation governing the MTL training dynamics in Eq. 7 becomes:

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \omega_ {1} ^ {2} \mathcal {K} _ {1 1} & \cdots & \omega_ {1} \omega_ {k} \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \omega_ {k} \omega_ {1} \mathcal {K} _ {k 1} & \cdots & \omega_ {k} ^ {2} \mathcal {K} _ {k k} \end{array} \right]} _ {\boldsymbol {\omega} \boldsymbol {\omega} ^ {\top} \odot \tilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right]. \tag {13}
$$

Detailed proof can be found in the Appendix. The NTK matrix derived from the weighted optimization objective in Eq. 12 remains positive semi-definite, and thus, the analysis in Sec. 3.2 is still applicable. Proposition 3.3 shows that the eigenvalues of the new NTK matrix are strongly correlated with ω, allowing ω to be used to balance the relative magnitudes of the eigenvalues of the NTK across different tasks, thereby balancing the convergence speeds of different tasks.

Motivated by this, our method is designed based on the following strategy: In each training iteration, we compute the maximum eigenvalue $\lambda _ { i }$ of the NTK matrix $\kappa _ { i i }$ for task i, which serves as a good indicator of its current convergence speed. We then derive the task weights $\{ \omega _ { i } \} _ { i = 1 } ^ { k }$ by normalizing these eigenvalues $\{ \lambda _ { i } \} _ { i = 1 } ^ { k }$ . Proposition 3.3 provides the most intuitive physical interpretation: With the introduction of $\omega _ { i }$ , the new NTK matrix for each task becomes $\omega _ { i } ^ { \bar { 2 } } \bar { K _ { i i } }$ , and its maximum eigenvalue is accordingly scaled to $\omega _ { i } ^ { 2 } \lambda _ { i }$ . Therefore, to achieve eigenvalue normalization across tasks, we enforce the condition that:

$$
\omega_ {i} \propto \frac {1}{\sqrt {\lambda_ {i}}}, i = 1, \dots , k. \tag {14}
$$

Directly employing $\begin{array} { r } { \omega _ { i } ~ = ~ { \frac { 1 } { \sqrt { \lambda _ { i } } } } } \end{array}$ disregards the original scaling inherent in the task eigenvalues. Consequently, to retain the scale information reflecting the tasks’ current convergence speeds, we utilize the average maximum eigenvalue, $\begin{array} { r } { \tilde { \lambda } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \lambda _ { j } } \end{array}$ , to scale $\{ w _ { i } \} _ { i = 1 } ^ { k }$ . This motivates the following definition for $\omega _ { i }$ :

$$
\omega_ {i} = \sqrt {\frac {\tilde {\lambda}}{\lambda_ {i}}}, i = 1, \dots , k. \tag {15}
$$

The overall algorithm flow is shown in Algorithm 1. Our design of ω enables the effective balancing of convergence speeds across different tasks by balancing the maximum eigenvalues of their NTK matrices, while preserving the scale information of the original eigenvalues. Nevertheless, computing the NTK matrix can be time-consuming since it requires dividing a batch into n mini-batches and computing their gradients. This limitation has prompted us to find a way to compute the NTK with minimal cost, thereby extending the applicability of our method to more scenarios. The following section provides the solution.

## Algorithm 1 NTKMTL

1: Input: Initial model parameters $\theta _ { 0 } ;$ Learning rate $\{ \eta _ { t } \} ;$ number n of mini batches.  
2: for $t = \dot { 0 }$ to $T - 1$ do  
3: Compute $[ J _ { 1 } ^ { t } , \cdots , J _ { k } ^ { t } ]$ for n mini-batches, and obtain the gradients $[ g _ { 1 } ^ { t } , \cdots , g _ { k } ^ { t } ]$  
4: Obtain the NTK $\{ K _ { i i } ^ { t } \} _ { i = 1 } ^ { k }$ and $\widetilde { \mathcal { K } ^ { t } }$ through Eq. 8 and 9.  
5: Compute $\{ \omega _ { i } \} _ { i = 1 } ^ { k }$ through Eq. 15.  
6: Obtain the update vector $g _ { s } ^ { t }$ = $\sum _ { i = 1 } ^ { k } \omega _ { i } g _ { i } ^ { t } .$  
7: Update $\begin{array} { r } { \dot { \theta _ { t + 1 } } = \theta _ { t } - \eta _ { t } g _ { s } ^ { t } . } \end{array}$  
8: end for

## Algorithm 2 NTKMTL-SR

1: Input: Initial model parameters $\theta _ { 0 } ;$ Learning rate $\{ \eta _ { t } \} ;$ ; number n of mini batches.  
2: for $t = 0$ to $T - 1$ do  
3: Compute $[ J _ { 1 } ^ { t } ( z ) , \cdot \cdot \cdot , J _ { k } ^ { t } ( z ) ]$ with respect to z for n mini batches.  
4: Obtain $\{ \mathcal { K } _ { i i } ^ { t } ( z ) \} _ { i = 1 } ^ { k }$ and $\widetilde { \mathcal { K } } ^ { t } ( z )$ through Eq.9 and 17.  
5: Compute $\{ \omega _ { i } \} _ { i = 1 } ^ { k }$ through Eq. 18.  
6: Obtain aggregated loss $\begin{array} { r } { \mathscr { L } = \sum _ { i = 1 } ^ { k } \omega _ { i } \ell _ { i } . } \end{array}$  
7: Back propagate and update parameters $\theta _ { t + 1 } .$  
8: end for

## 3.4 Approximation via Shared Representation

In MTL, the model typically consists of shared parameters θ and task-specific parameters for k tasks, where the number of parameters in the task-specific components (typically 1–2 layers of linear or convolutional layers) is much smaller than the number of shared parameters. We define the shared representation z as the output of the input x after passing through the shared parameters θ. Then, by applying the chain rule, we can derive the following:

$$
\frac {\partial f _ {i} (\theta , \mathbf {x})}{\partial \theta} = \frac {\partial f _ {i} (\theta , \mathbf {x})}{\partial z} \cdot \frac {\partial z}{\partial \theta}. \tag {16}
$$

Note that $\frac { \partial z } { \partial \theta }$ is the same for all tasks and acts on all $\{ f _ { i } \} _ { i = 1 } ^ { k }$ . As a result, it further impacts the overall NTK ${ \widetilde { \kappa } } .$ . This implies that $\widetilde { \kappa }$ can be viewed as a projection of the NTK matrix $\widetilde { \mathcal { K } } ( z )$ of z onto the feature space through $\textstyle { \frac { \partial z } { \partial \theta } }$ . These analyses indicate that, similar to some previous works [23, 47], NTKMTL has the ability to accelerate computation using shared representation, and we name this approximation algorithm NTKMTL-SR. Specifically, we consider approximating the original method using the NTK analysis of z, i,e. by replacing Eq. 8 with the following expression:

$$
(\mathcal {K} _ {i j} (z)) _ {u v} = \left\langle \frac {\partial f _ {i} (\theta , x _ {u})}{\partial z}, \frac {\partial f _ {j} (\theta , x _ {v})}{\partial z} \right\rangle , \tag {17}
$$

which further leads to the subsequent formulation for ω:

$$
\omega_ {i} = \sqrt {\frac {\tilde {\lambda} (z)}{\lambda_ {i} (z)}}, i = 1, \dots , k, \tag {18}
$$

where $\lambda _ { i } ( z )$ represents the maximum eigenvalue of the NTK matrix $K _ { i i } ( z )$ , and $\tilde { \lambda } ( z )$ is the average of these eigenvalues across all k tasks, defined as $\begin{array} { r } { \tilde { \lambda } ( z ) = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \lambda _ { j } ( z ) } \end{array}$ . Since computing the gradient of $f _ { i } ( \boldsymbol { \theta } , { \bf x } )$ with respect to z only requires backpropagation through the task-specific parameters, it incurs little additional time and memory cost. After obtaining ω, instead of computing the gradients separately for each of the k tasks and then weighting them, we can directly compute the aggregated loss using Eq. 12 and perform one backpropagation, meaning that we only need to compute the gradient for shared parameters θ once per iteration.

The overall algorithm is outlined in Algorithm 2. Fig. 1 illustrates the training speed on the CelebA benchmark with up to 40 tasks, showing that NTKMTL-SR exhibits nearly the same training speed as traditional loss-oriented Linear Scalarization. This further enhances the generalizability of our method under various constraints.

## 4 Experiments

## 4.1 Protocols

We evaluate the performance of our proposed NTKMTL and NTKMTL-SR across a wide range of MTL scenarios, including multi-task supervised learning and multi-task reinforcement learning. For multi-task supervised learning, experiments are conducted on several benchmarks, including dense prediction tasks on the NYUv2 [49] and CityScapes [15] datasets, regression tasks on the QM9 [8] dataset, and image-level classification on the CelebA [36] dataset. For multi-task reinforcement learning, experiments are performed in the MT10 environment from the Meta-World benchmark [59]. Due to the memory and time cost associated with fully computing the gradients of n mini-batches for shared parameters to construct the NTK matrix, we set $n = 1$ for NTKMTL to ensure fairness when comparing with other methods. In this case, it is consistent with other gradient-oriented methods, requiring k gradient backpropagations per iteration. As for NTKMTL-SR, since its cost of constructing the NTK matrix is minimal, we set $n = 4$ by default based on experimental validation.

Table 1: Results on NYU-v2 (3-task) dataset. Each experiment is repeated 3 times with different random seeds and the average is reported. The detailed standard error is reported in the Appendix. The best scores are reported in gray .

<table><tr><td rowspan="3">METHOD</td><td colspan="2">SEGMENTATION</td><td colspan="2">DEPTH</td><td colspan="5">SURFACE NORMAL</td><td rowspan="3">MR↓</td><td rowspan="3">Δm%↓</td></tr><tr><td rowspan="2">MIOU ↑</td><td rowspan="2">PIX ACC ↑</td><td rowspan="2">ABS ERR ↓</td><td rowspan="2">REL ERR ↓</td><td colspan="2">ANGLE DISTANCE ↓</td><td colspan="3">WITHIN  $t^o$ ↑</td></tr><tr><td>MEAN</td><td>MEDIAN</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>STL</td><td>38.30</td><td>63.76</td><td>0.6754</td><td>0.2780</td><td>25.01</td><td>19.21</td><td>30.14</td><td>57.20</td><td>69.15</td><td></td><td></td></tr><tr><td>LS</td><td>39.29</td><td>65.33</td><td>0.5493</td><td>0.2263</td><td>28.15</td><td>23.96</td><td>22.09</td><td>47.50</td><td>61.08</td><td>17.44</td><td>5.59</td></tr><tr><td>SI</td><td>38.45</td><td>64.27</td><td>0.5354</td><td>0.2201</td><td>27.60</td><td>23.37</td><td>22.53</td><td>48.57</td><td>62.32</td><td>16.11</td><td>4.39</td></tr><tr><td>RLW [28]</td><td>37.17</td><td>63.77</td><td>0.5759</td><td>0.2410</td><td>28.27</td><td>24.18</td><td>22.26</td><td>47.05</td><td>60.62</td><td>20.22</td><td>7.78</td></tr><tr><td>DWA [35]</td><td>39.11</td><td>65.31</td><td>0.5510</td><td>0.2285</td><td>27.61</td><td>23.18</td><td>24.17</td><td>50.18</td><td>62.39</td><td>16.33</td><td>3.57</td></tr><tr><td>UW [25]</td><td>36.87</td><td>63.17</td><td>0.5446</td><td>0.2260</td><td>27.04</td><td>22.61</td><td>23.54</td><td>49.05</td><td>63.65</td><td>16.00</td><td>4.05</td></tr><tr><td>MGDA [46]</td><td>30.47</td><td>59.90</td><td>0.6070</td><td>0.2555</td><td>24.88</td><td>19.45</td><td>29.18</td><td>56.88</td><td>69.36</td><td>11.44</td><td>1.38</td></tr><tr><td>PCGRAD [58]</td><td>38.06</td><td>64.64</td><td>0.5550</td><td>0.2325</td><td>27.41</td><td>22.80</td><td>23.86</td><td>49.83</td><td>63.14</td><td>16.89</td><td>3.97</td></tr><tr><td>GRADDROP [13]</td><td>39.39</td><td>65.12</td><td>0.5455</td><td>0.2279</td><td>27.48</td><td>22.96</td><td>23.38</td><td>49.44</td><td>62.87</td><td>15.56</td><td>3.58</td></tr><tr><td>CAGRAD [32]</td><td>39.79</td><td>65.49</td><td>0.5486</td><td>0.2250</td><td>26.31</td><td>21.58</td><td>25.61</td><td>52.36</td><td>65.58</td><td>11.56</td><td>0.20</td></tr><tr><td>IMTL-G [33]</td><td>39.35</td><td>65.60</td><td>0.5426</td><td>0.2256</td><td>26.02</td><td>21.19</td><td>26.20</td><td>53.13</td><td>66.24</td><td>10.89</td><td>-0.76</td></tr><tr><td>MoCO [18]</td><td>40.30</td><td>66.07</td><td>0.5575</td><td>0.2135</td><td>26.67</td><td>21.83</td><td>25.61</td><td>51.78</td><td>64.85</td><td>10.89</td><td>0.16</td></tr><tr><td>NASH-MTL [39]</td><td>40.13</td><td>65.93</td><td>0.5261</td><td>0.2171</td><td>25.26</td><td>20.08</td><td>28.40</td><td>55.47</td><td>68.15</td><td>8.00</td><td>-4.04</td></tr><tr><td>ALIGNED-MTL [47]</td><td>40.15</td><td>66.05</td><td>0.5520</td><td>0.2291</td><td>25.37</td><td>19.89</td><td>28.30</td><td>55.29</td><td>67.95</td><td>10.44</td><td>-3.12</td></tr><tr><td>FAMO [31]</td><td>38.88</td><td>64.90</td><td>0.5474</td><td>0.2194</td><td>25.06</td><td>19.57</td><td>29.21</td><td>56.61</td><td>68.98</td><td>9.00</td><td>-4.10</td></tr><tr><td>SDMGRAD [54]</td><td>40.47</td><td>65.90</td><td>0.5225</td><td>0.2084</td><td>25.07</td><td>19.99</td><td>28.54</td><td>55.74</td><td>68.53</td><td>6.11</td><td>-4.84</td></tr><tr><td>DB-MTL [27]</td><td>41.42</td><td>66.45</td><td>0.5251</td><td>0.2160</td><td>25.03</td><td>19.50</td><td>28.72</td><td>56.17</td><td>68.73</td><td>4.56</td><td>-5.36</td></tr><tr><td>STCH [30]</td><td>41.35</td><td>66.07</td><td>0.4965</td><td>0.2010</td><td>26.55</td><td>21.81</td><td>24.84</td><td>51.39</td><td>64.86</td><td>8.22</td><td>-1.35</td></tr><tr><td>FAIRGRAD [4]</td><td>39.74</td><td>66.01</td><td>0.5377</td><td>0.2236</td><td>24.84</td><td>19.60</td><td>29.26</td><td>56.58</td><td>69.16</td><td>6.44</td><td>-4.66</td></tr><tr><td>GO4ALIGN [48]</td><td>40.42</td><td>65.37</td><td>0.5492</td><td>0.2167</td><td>24.76</td><td>18.94</td><td>30.54</td><td>57.87</td><td>69.84</td><td>5.00</td><td>-6.08</td></tr><tr><td>NTKMTL</td><td>39.68</td><td>65.43</td><td>0.5296</td><td>0.2168</td><td>24.24</td><td>18.63</td><td>30.74</td><td>58.72</td><td>70.78</td><td>4.33</td><td>-6.99</td></tr><tr><td>NTKMTL-SR</td><td>40.23</td><td>65.28</td><td>0.5261</td><td>0.2136</td><td>24.88</td><td>19.58</td><td>29.53</td><td>56.67</td><td>69.08</td><td>5.56</td><td>-5.35</td></tr></table>

Baselines. We comprehensively compare the proposed NTKMTL and NTKMTL-SR with the following methods: Single-task learning (STL), Linear Scalarization (LS), Scale-Invariant (SI), Dynamic Weight Average (DWA) [35], Uncertainty Weighting (UW) [25], Multi-Gradient Descent Algorithm (MGDA) [46], Random Loss Weighting (RLW) [28], PCGrad [58], GradDrop [13], CAGrad [32], IMTL-G [33], Nash-MTL [39], Moco [18], Aligned-MTL [47], SDMGrad [54], DB-MTL [27], STCH [30], FAMO [31], FairGrad [4] and GO4Align [48].

Metrics. We follow previous works [39, 4] and use two overall performance metrics for MTL: (1) ∆m%, the average performance drop relative to the STL baseline: $\Delta m \% =$ $\begin{array} { r } { \frac { 1 } { S } \sum _ { i = 1 } ^ { S } ( - 1 ) ^ { \delta _ { i } } \frac { ( M _ { m , i } - M _ { b , i } ) } { M _ { b , i } } \times \overline { { 1 0 0 \% } } } \end{array}$ , where S is the number of metrics. $M _ { b , i }$ is the baseline STL metric and $M _ { m , i }$ is the metric from the MTL method. $\delta _ { i } = 1$ if a higher value is better for $M _ { i } .$ , and 0 otherwise. (2) Mean Rank (MR): MR reports the average rank of a method across all tasks, where a lower value indicates better performance. A method with the top rank in all tasks has an MR of 1.

## 4.2 Multi-Task Supervised Learning

Dense Prediction. Widely used benchmarks in this domain include NYUv2 [49] and CityScapes [15]. The NYUv2 dataset includes three tasks: semantic segmentation, depth estimation, and surface normal prediction, while CityScapes includes semantic segmentation and depth estimation tasks. The experimental results are presented in Table 1 and 2 in the main text, and Table 7 in the Appendix.

On the NYUv2 dataset, the difficulty levels of the three tasks show significant variation. Previous methods generally outperform the Single Task Learning (STL) baseline in the tasks of semantic segmentation and depth estimation, but almost all of them consistently underperform STL on the surface normal prediction task, leading to a significant task imbalance in the overall results. In contrast, by leveraging NTK theory to balance convergence speed of each task during training, both NTKMTL and NTKMTL-SR show strong performance in the surface normal prediction task. Notably, among all existing methods, only NTKMTL and GO4Align consistently outperform the STL baseline across all three tasks, achieving a more balanced optimization. Moreover, NTKMTL achieves SOTA on this benchmark with an impressive mean rank of 4.33 and the best performance drop of -6.99%.

On the CityScapes dataset, NTKMTL and NTKMTL-SR also perform exceptionally well, as shown in Table 2. More detailed results can be found in Table 7 in the Appendix. Compared to existing methods that tend to prioritize the optimization of semantic segmentation, NTKMTL and NTKMTL-SR achieve more balanced results across both segmentation and depth estimation tasks, with NTKMTL also obtaining the best performance drop.

Image-Level Classification. We also evaluated the performance of our proposed NTKMTL and NTKMTL-SR on the CelebA dataset. CelebA [36] is a large-scale facial attribute dataset consisting of over 200K images, each labeled with 40 attributes, such as smiling, wavy hair, and mustache. This task represents a 40- task MTL classification problem, where each task is designed to predict one binary attribute. This benchmark tests both the collaborative op-

Table 2: Results on CityScapes (2-task) and CelebA (40-task). Each experiment is repeated 3 times with different random seeds and the average is reported. Detailed standard error is reported in the Appendix. Best scores are reported in gray .

<table><tr><td rowspan="2">METHOD</td><td colspan="2">CITYSCAPES</td><td colspan="2">CELEBA</td></tr><tr><td>MR ↓</td><td> $\Delta m\% \downarrow$ </td><td>MR ↓</td><td> $\Delta m\% \downarrow$ </td></tr><tr><td>LS</td><td>8.25</td><td>22.60</td><td>7.85</td><td>4.15</td></tr><tr><td>SI</td><td>11.50</td><td>14.11</td><td>9.75</td><td>7.20</td></tr><tr><td>RLW [28]</td><td>10.25</td><td>24.38</td><td>6.90</td><td>1.46</td></tr><tr><td>DWA [35]</td><td>7.75</td><td>21.45</td><td>8.72</td><td>3.20</td></tr><tr><td>UW [25]</td><td>7.75</td><td>5.89</td><td>7.38</td><td>3.23</td></tr><tr><td>MGDA [46]</td><td>12.00</td><td>44.14</td><td>12.97</td><td>14.85</td></tr><tr><td>PCGRAD [58]</td><td>8.50</td><td>18.29</td><td>8.53</td><td>3.17</td></tr><tr><td>CAGRAD [32]</td><td>7.25</td><td>11.64</td><td>8.10</td><td>2.48</td></tr><tr><td>IMTL-G [33]</td><td>5.50</td><td>11.10</td><td>6.25</td><td>0.84</td></tr><tr><td>NASH-MTL [39]</td><td>3.50</td><td>6.82</td><td>6.50</td><td>2.84</td></tr><tr><td>FAMO [31]</td><td>7.75</td><td>8.13</td><td>6.45</td><td>1.21</td></tr><tr><td>FAIRGRAD [4]</td><td>2.25</td><td>5.18</td><td>6.92</td><td>0.37</td></tr><tr><td>NTKMTL</td><td>7.00</td><td>1.92</td><td>4.35</td><td>-0.77</td></tr><tr><td>NTKMTL-SR</td><td>6.25</td><td>3.84</td><td>4.33</td><td>0.23</td></tr></table>

timization capability and efficiency of MTL methods when dealing with a large number of tasks. The results are shown in Table 2. In this challenging setting, NTKMTL achieves better average performance than the STL baseline, as indicated by the negative ∆m%, which was not achievable by previous methods. NTKMTL also achieved state-of-the-art performance in both MR and ∆m%.

Fig. 1 visualizes the training time per epoch for various loss-oriented and gradient-oriented methods. On the CelebA dataset with up to 40 tasks, NTKMTL-SR maintains a speed comparable to loss-oriented methods.

Multi-Task Regression. QM9 [8] is a widely used benchmark for multi-task regression, containing over 130K organic molecules represented as graphs. It includes 11 tasks, each requiring the prediction of a molecular property. Due to the large number of tasks, as well as the

![](images/096f43df22bee1d2b7ce5d55af1a5a6a79c089593c105d1c538dde229068e4b1.jpg)

<details>
<summary>bar</summary>

| Category | Method | Time (min) | Improvement Factor |
| --- | --- | --- | --- |
| Loss-oriented | LS | ~3.5 | 1.00x |
| Loss-oriented | SI | ~3.5 | 1.00x |
| Loss-oriented | FAMO | ~3.5 | 1.08x |
| Loss-oriented | GO4ALIGN | ~3.5 | 1.03x |
| Gradient-oriented | NTKMTL-SR | ~4.5 | 1.24x |
| Gradient-oriented | MGDA | ~20 | 5.58x |
| Gradient-oriented | PCGrad | ~29 | 8.08x |
| Gradient-oriented | CAGrad | ~16 | 4.47x |
| Gradient-oriented | Nash-MTL | ~22 | 6.11x |
| Gradient-oriented | NTKMTL | ~15 | 4.33x |
</details>

Figure 1: Training time per epoch for various methods on CelebA (40-task) dataset.

significant differences in task difficulty and convergence speeds, existing MTL methods exhibit a substantial performance drop compared to the STL baseline.

On this benchmark, previous methods were implemented using a shared MPNN codebase [39, 31, 19]. However, we find that the hyperparameter settings in this codebase are suboptimal (specifically, the improper learning rate scheduler makes the learning rate decay too quickly, resulting in incomplete convergence), potentially leading to unfair comparisons. Therefore, we adjusted specific hyperparameters and reproduced all baselines for a fair comparison. To ensure accurate evaluation, we also reproduced the 11-task STL baselines under the same settings.

Experimental results are presented in Table 3. More detailed explanations and settings are provided in Appendix C. As both STL baselines and MTL methods show significant performance improvements under the new settings, we can observe different behaviors among previous methods. The final $\Delta m \%$ of some methods (e.g., LS, RLW, PCGrad, CAGrad) were largely consistent with their originally reported values. For other methods (e.g., SI, UW, FAMO, GO4Align), the $\Delta m \%$ significantly improved compared to the reported results. This indicates that the previous hyperparameter settings fail to fully exhibit the capabilities of these methods. Under the new settings that better ensure convergence, their performance shows further improvement. On this benchmark, NTKMTL shows competitive results. NTKMTL-SR surpasses NTKMTL in performance and achieves state-of-the-art results, which we attribute to the natural alignment of the L2 loss used in regression tasks as analyzed in Appendix B.1. As a result, larger n produces a more accurate estimate of convergence speed.

Table 3: Results on QM9 (11-task) dataset. All baselines are reproduced under optimized hyperparameter settings. The best scores are reported in gray . More details are reported in the Appendix.

<table><tr><td rowspan="2">METHOD</td><td> $\mu$ </td><td> $\alpha$ </td><td> $\epsilon_{HOMO}$ </td><td> $\epsilon_{LUMO}$ </td><td> $\langle R^2\rangle$ </td><td>ZPVE</td><td> $U_0$ </td><td>U</td><td>H</td><td>G</td><td> $c_v$ </td><td rowspan="2">MR↓</td><td rowspan="2"> $\Delta m\% \downarrow$ </td></tr><tr><td colspan="11">MAE ↓</td></tr><tr><td>STL</td><td>0.060</td><td>0.156</td><td>60.54</td><td>51.22</td><td>0.419</td><td>3.08</td><td>39.3</td><td>42.9</td><td>41.7</td><td>43.1</td><td>0.061</td><td></td><td></td></tr><tr><td>LS</td><td>0.077</td><td>0.253</td><td>55.95</td><td>68.59</td><td>4.163</td><td>11.08</td><td>109.2</td><td>109.8</td><td>110.1</td><td>106.7</td><td>0.099</td><td>9.82</td><td>179.8</td></tr><tr><td>SI</td><td>0.159</td><td>0.242</td><td>109.6</td><td>96.80</td><td>0.732</td><td>3.308</td><td>34.35</td><td>34.37</td><td>34.33</td><td>35.16</td><td>0.081</td><td>5.27</td><td>39.7</td></tr><tr><td>RLW [28]</td><td>0.090</td><td>0.277</td><td>62.49</td><td>76.39</td><td>4.948</td><td>12.54</td><td>124.8</td><td>124.9</td><td>125.0</td><td>122.1</td><td>0.115</td><td>12.09</td><td>222.6</td></tr><tr><td>DWA [35]</td><td>0.078</td><td>0.239</td><td>55.17</td><td>67.40</td><td>3.992</td><td>10.92</td><td>107.4</td><td>108.1</td><td>108.3</td><td>105.4</td><td>0.098</td><td>8.73</td><td>173.0</td></tr><tr><td>UW [25]</td><td>0.194</td><td>0.274</td><td>120.6</td><td>102.8</td><td>0.763</td><td>3.698</td><td>41.11</td><td>41.13</td><td>41.16</td><td>41.75</td><td>0.089</td><td>8.27</td><td>58.4</td></tr><tr><td>MGDA [46]</td><td>0.154</td><td>0.266</td><td>95.20</td><td>67.51</td><td>3.088</td><td>4.468</td><td>49.38</td><td>49.21</td><td>49.62</td><td>49.69</td><td>0.087</td><td>8.36</td><td>101.4</td></tr><tr><td>PCGRAD [58]</td><td>0.078</td><td>0.221</td><td>59.14</td><td>67.82</td><td>2.937</td><td>6.691</td><td>88.24</td><td>88.65</td><td>88.85</td><td>87.36</td><td>0.084</td><td>7.91</td><td>118.6</td></tr><tr><td>CAGRAD [32]</td><td>0.083</td><td>0.234</td><td>57.80</td><td>70.98</td><td>2.718</td><td>5.352</td><td>76.47</td><td>76.93</td><td>77.05</td><td>76.32</td><td>0.089</td><td>8.27</td><td>102.4</td></tr><tr><td>NASH-MTL [39]</td><td>0.086</td><td>0.218</td><td>69.78</td><td>66.18</td><td>2.153</td><td>4.679</td><td>59.63</td><td>59.94</td><td>59.98</td><td>59.97</td><td>0.082</td><td>6.91</td><td>72.9</td></tr><tr><td>FAMO [31]</td><td>0.128</td><td>0.230</td><td>98.09</td><td>84.42</td><td>0.859</td><td>3.541</td><td>40.24</td><td>40.57</td><td>40.62</td><td>40.21</td><td>0.081</td><td>5.91</td><td>38.9</td></tr><tr><td>FAIRGRAD [4]</td><td>0.109</td><td>0.208</td><td>81.74</td><td>72.82</td><td>1.669</td><td>3.418</td><td>51.31</td><td>51.67</td><td>51.72</td><td>51.97</td><td>0.079</td><td>6.64</td><td>57.0</td></tr><tr><td>GO4ALIGN [48]</td><td>0.113</td><td>0.314</td><td>74.46</td><td>91.04</td><td>0.912</td><td>3.632</td><td>36.06</td><td>36.38</td><td>36.41</td><td>36.58</td><td>0.104</td><td>6.64</td><td>40.5</td></tr><tr><td>NTKMTL</td><td>0.091</td><td>0.212</td><td>70.97</td><td>70.81</td><td>2.113</td><td>3.835</td><td>44.18</td><td>44.56</td><td>44.53</td><td>44.38</td><td>0.077</td><td>5.91</td><td>56.7</td></tr><tr><td>NTKMTL-SR</td><td>0.081</td><td>0.207</td><td>75.95</td><td>69.10</td><td>1.176</td><td>3.689</td><td>40.14</td><td>40.46</td><td>40.48</td><td>40.49</td><td>0.074</td><td>4.00</td><td>30.7</td></tr></table>

## 4.3 Multi-Task Reinforcement Learning

We further assess our method using the MT10 benchmark, which consists of 10 robotic manipulation tasks from the MetaWorld environment [59]. The goal in this setting is to train a single policy that can generalize across a variety of tasks, including pick-and-place and door-opening. Specifically, we follow the setup outlined in previous works [39, 31] and use Soft Actor-Critic (SAC) [21] as the core algorithm. Our implementation builds upon the MTRL codebase from [39, 4], training the model for 2 million steps with a batch size of 1280.

In contrast to the multi-task supervised learning networks where shared and task-specific parameters can be easily distinguished, the MTRL problems commonly involve learning a single policy. Consequently, it is difficult to partition the parameters into shared and task-specific components in the same manner. Therefore, while applying NTKMTL-SR to this scenario is challenging, we primarily validate the performance of NTKMTL. We compare NTKMTL with several state-of-the-art methods, including Multi-task SAC (MTL SAC) [59], Multi-task SAC with Task Encoder (MTL SAC + TE) [59], Multi-headed SAC (MH SAC) [59], PCGrad [58], CAGrad [32], MoCo [18], Nash-MTL [39], FAMO [31], FairGrad [4] and Aligned-MTL [47]. The experimental results are reported in Table 4. In the MTRL scenario, NTKMTL continues to demonstrate strong performance with a competitive success rate.

Table 4: Results on MT10 benchmark across 10 random seeds.

<table><tr><td>METHOD</td><td>SUCCESS RATE(MEAN ± STDERR)</td></tr><tr><td>STL</td><td>0.90 ± 0.03</td></tr><tr><td>MTL SAC [59]</td><td>0.49 ± 0.07</td></tr><tr><td>MTL SAC + TE [59]</td><td>0.54 ± 0.05</td></tr><tr><td>MH SAC [59]</td><td>0.61 ± 0.04</td></tr><tr><td>PCGRAD [58]</td><td>0.72 ± 0.02</td></tr><tr><td>CAGRAD [32]</td><td>0.83 ± 0.05</td></tr><tr><td>MoCo [18]</td><td>0.75 ± 0.05</td></tr><tr><td>NASH-MTL [39]</td><td>0.91 ± 0.03</td></tr><tr><td>FAMO [31]</td><td>0.83 ± 0.05</td></tr><tr><td>FAIRGRAD [4]</td><td>0.84 ± 0.07</td></tr><tr><td>ALIGNED-MTL [47]</td><td>0.97 ± 0.05</td></tr><tr><td>NTKMTL</td><td>0.96 ± 0.03</td></tr></table>

## 5 Conclusion, Limitations and Future Work

In this paper, we introduce a new perspective on understanding task imbalance in MTL by leveraging NTK theory for analysis, and propose a new MTL method, NTKMTL. Specifically, we conduct spectral analysis of the NTK matrix during training, adjust the maximum eigenvalues of the taskspecific NTK matrices to balance their convergence speeds, thereby mitigating task imbalance. Furthermore, we present NTKMTL-SR, an efficient variant based on approximation via shared representation, which achieves competitive performance with improved training efficiency. Extensive experiments have shown the strong performance of both NTKMTL and NTKMTL-SR, further demonstrating the applicability and generalization of our method across a wide range of scenarios.

Limitations and Future Work. In this work, we proposed the extended NTK $\widetilde { \kappa }$ for MTL as a tool for more comprehensive analysis of training dynamics across tasks. The weight design of our current method mainly focuses on the analysis of the task-specific NTK matrices $\kappa _ { i i }$ . The full structure of the extended NTK matrix $\tilde { \kappa }$ offers further analytical opportunities. For example, analyzing the off-diagonal $\kappa _ { i j }$ blocks could reveal novel insights into task interactions or guide the development of task grouping strategies. These potential avenues are left for future investigation.

## References

[1] Idan Achituve, Haggai Maron, and Gal Chechik. Self-supervised learning for domain adaptation on point clouds. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 123–133, 2021.  
[2] Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning, pages 322–332. PMLR, 2019.  
[3] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
[4] Hao Ban and Kaiyi Ji. Fair resource allocation in multi-task learning. arXiv preprint arXiv:2402.15638, 2024.  
[5] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritchman. Frequency bias in neural networks for input of non-uniform density. In International conference on machine learning, pages 685–694. PMLR, 2020.  
[6] Jonathan Baxter. A model of inductive bias learning. Journal ofartificial intelligence research, 12:149–198, 2000.  
[7] Alberto Bietti and Julien Mairal. On the inductive bias of neural tangent kernels. Advances in Neural Information Processing Systems, 32, 2019.  
[8] Lorenz C Blum and Jean-Louis Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database gdb-13. Journal ofthe American Chemical Society, 131(25):8732–8733, 2009.  
[9] Rich Caruana. Multitask learning. Machine learning, 28:41–75, 1997.  
[10] Shijie Chen, Yu Zhang, and Qiang Yang. Multi-task learning in natural language processing: An overview. ACM Computing Surveys, 56(12):1–32, 2024.  
[11] Weiyu Chen, Xiaoyuan Zhang, Baijiong Lin, Xi Lin, Han Zhao, Qingfu Zhang, and James T Kwok. Gradient-based multi-objective deep learning: Algorithms, theories, applications, and beyond. arXiv preprint arXiv:2501.10945, 2025.  
[12] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018.  
[13] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.  
[14] Sumanth Chennupati, Ganesh Sistu, Senthil Yogamani, and Samir A Rawashdeh. Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019.  
[15] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.  
[16] Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. In 2017 IEEE international conference on robotics and automation (ICRA), pages 2169–2176. IEEE, 2017.  
[17] Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. arXiv preprint arXiv:1810.02054, 2018.  
[18] Heshan Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen. Mitigating gradient bias in multi-objective learning: A provably convergent approach. International Conference on Learning Representations, 2023.  
[19] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. PMLR, 2017.  
[20] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pages 270–287, 2018.  
[21] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.  
[22] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.  
[23] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In International Conference on Learning Representations.  
[24] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.  
[25] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.  
[26] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems, 32, 2019.  
[27] Baijiong Lin, Weisen Jiang, Feiyang Ye, Yu Zhang, Pengguang Chen, Ying-Cong Chen, Shu Liu, and James T Kwok. Dual-balancing for multi-task learning. arXiv preprint arXiv:2308.12029, 2023.  
[28] Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. 2021.  
[29] Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. arXiv preprint arXiv:2111.10603, 2021.  
[30] Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Smooth tchebycheff scalarization for multi-objective optimization. arXiv preprint arXiv:2402.19078, 2024.  
[31] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36, 2024.  
[32] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021.  
[33] Liyang Liu, Yi Li, Zhanghui Kuang, J Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. iclr, 2021.  
[34] Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. Adversarial multi-task learning for text classification. arXiv preprint arXiv:1704.05742, 2017.  
[35] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019.  
[36] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.  
[37] Keerthiram Murugesan and Jaime Carbonell. Self-paced multitask learning with shared knowl edge. arXiv preprint arXiv:1703.00977, 2017.  
[38] Aviv Navon, Idan Achituve, Haggai Maron, Gal Chechik, and Ethan Fetaya. Auxiliary learning by implicit differentiation. arXiv preprint arXiv:2007.02693, 2020.  
[39] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022.  
[40] Jonathan Pilault, Amine Elhattami, and Christopher Pal. Conditionally adaptive multi-task learning: Improving transfer learning in nlp using fewer parameters & less data. arXiv preprint arXiv:2009.09139, 2020.  
[41] Xiaohan Qin, Xiaoxing Wang, and Junchi Yan. Revisiting fairness in multitask learning: A performance-driven approach for variance reduction. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 20492–20501, 2025.  
[42] Xiaohan Qin, Xiaoxing Wang, and Junchi Yan. Towards consistent multi-task learning: Unlocking the potential of task-specific parameters. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 10067–10076, 2025.  
[43] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.  
[44] Daniel A Roberts, Sho Yaida, and Boris Hanin. The principles of deep learning theory, volume 46. Cambridge University Press Cambridge, MA, USA, 2022.  
[45] S Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.  
[46] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018.  
[47] Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20083–20093, 2023.  
[48] Jiayi Shen, Cheems Wang, Zehao Xiao, Nanne Van Noord, and Marcel Worring. Go4align: Group optimization for multi-task alignment. arXiv preprint arXiv:2404.06486, 2024.  
[49] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012.  
[50] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International conference on machine learning, pages 9120–9132. PMLR, 2020.  
[51] Vincent Szolnoky, Viktor Andersson, Balázs Kulcsár, and Rebecka Jörnsten. On the interpretability of regularisation for neural networks through model gradient similarity. Advances in Neural Information Processing Systems, 35:16319–16330, 2022.  
[52] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020.  
[53] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.  
[54] Peiyao Xiao, Hao Ban, and Kaiyi Ji. Direction-oriented multi-objective learning: Simple and provable stochastic algorithms. Advances in Neural Information Processing Systems, 36:4509–4533, 2023.  
[55] Caiming Xiong, SHU Tianmin, and Richard Socher. Hierarchical and interpretable skill acquisition in multi-task reinforcement learning, January 24 2023. US Patent 11,562,287.  
[56] Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo. Adatask: A task-aware adaptive learning rate approach to multi-task learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 10745–10753, 2023.  
[57] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.  
[58] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.  
[59] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.  
[60] Hayoung Yun and Hanjoo Cho. Achievement-based training progress balancing for multi-task learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16935–16944, 2023.  
[61] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE transactions on knowledge and data engineering, 34(12):5586–5609, 2021.  
[62] Zelin Zhao, Fenglei Fan, Wenlong Liao, and Junchi Yan. Grounding and enhancing grid-based models for neural fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19425–19435, 2024.  
[63] Ce Zheng, Wenhan Wu, Chen Chen, Taojiannan Yang, Sijie Zhu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A survey. ACM Computing Surveys, 56(1):1–37, 2023.

## A Definitions of Notations

Due to the numerous concepts and theoretical derivations presented in the paper, we provide detailed definitions of the notations in Table 5 to assist readers in better understanding the content.

Table 5: Definitions of notations in this paper.

<table><tr><td>Variable</td><td>Definition</td></tr><tr><td>k</td><td>the number of tasks</td></tr><tr><td>f</td><td>the mapping function of a deep neural network</td></tr><tr><td> $f_i$ </td><td>the mapping function of i-th task in MTL scenario</td></tr><tr><td>n</td><td>the number of data points in the dataset, or the number of mini-batches in a batch</td></tr><tr><td>(x,y)</td><td>training dataset  $\{(x_i,y_i)\}_{i=1}^n$  in Single-Task Learning (STL) setting</td></tr><tr><td> $\{y_i\}_{i=1}^k$ </td><td>multiple labels for the k tasks in MTL setting</td></tr><tr><td>θ</td><td>the network parameters, specifically referring to the shared parameters in MTL</td></tr><tr><td>t</td><td>the current time step</td></tr><tr><td>K</td><td>the Neural Tangent Kernel matrix defined in Eq. 1</td></tr><tr><td>λ</td><td>the eigenvalues of the NTK</td></tr><tr><td>L</td><td>the overall loss function, with different definitions in different scenarios</td></tr><tr><td> $\{\ell_i\}_{i=1}^k$ </td><td>the losses with respect to k tasks</td></tr><tr><td>O(t)</td><td>the output  $\{f(x_i,\theta(t))\}_{i=1}^n$  of the network in STL setting and time t</td></tr><tr><td> $\{\mathcal{O}_i(t)\}_{i=1}^k$ </td><td>the output of the network for k tasks in MTL setting and time t</td></tr><tr><td>I</td><td>the Identity matrix</td></tr><tr><td> $η_t$ </td><td>the learning rate at time t</td></tr><tr><td> $\widetilde{K}$ </td><td>the extended NTK matrix defined in Eq. 9</td></tr><tr><td> $J_i$ </td><td>the Jacobian matrix of  $f_i$  with respect to θ</td></tr><tr><td>ω</td><td> $\{\omega_i\}_{i=1}^k$ , representing the weights of the different tasks</td></tr><tr><td>z</td><td>the shared representation of the input data in MTL setting</td></tr></table>

## B Theoretical Analysis

## B.1 Proof of Theorem 3.1

Theorem 3.1. Let $\mathcal { O } ( t ) = \{ f ( \theta , x _ { i } ) \} _ { i = 1 } ^ { n }$ be the outputs ofthe neural network at time t. $\mathbf { x } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ 1 is the input data, and $\mathbf { y } = \{ y _ { i } \} _ { i = 1 } ^ { n }$ is the corresponding label, Then $\mathcal { O } ( t )$ follows this evolution:

$$
\frac {d \mathcal {O} (t)}{d t} = - \mathcal {K} \cdot (\mathcal {O} (t) - \mathbf {y}).
$$

Proof. Following previous works [44, 52, 62], we build up the analysis framework for NTK in a supervised regression setting. The overall loss $\mathcal { L } ( \boldsymbol { \theta } )$ is defined as:

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {n} \frac {1}{2} (f (\theta , x _ {i}) - y _ {i}) ^ {2}. \tag {19}
$$

Through the gradient flow in Eq. 2, we can obtain:

$$
\begin{array}{l} \frac {d \theta}{d t} = - \nabla_ {\theta} \mathcal {L} (\theta) \\ = - \sum_ {i = 1} ^ {n} \frac {\partial \mathcal {L}}{\partial f (\theta , x _ {i})} \cdot \frac {\partial f (\theta , x _ {i})}{\partial \theta} \tag {20} \\ = - \sum_ {i = 1} ^ {n} \frac {\partial f (\theta , x _ {i})}{\partial \theta} (f (\theta , x _ {i}) - y _ {i}). \\ \end{array}
$$

In fact, for the commonly used cross-entropy loss, the above equation also holds. In this case, considering a data point $x _ { j }$ , we have

$$
\begin{array}{l} \frac {d f (\theta , x _ {j})}{d t} = \frac {d f (\theta , x _ {j})}{d \theta} \cdot \frac {d \theta}{d t} \\ = \frac {d f (\theta , x _ {j})}{d \theta} \left[ - \sum_ {i = 1} ^ {n} \frac {\partial f (\theta , x _ {i})}{\partial \theta} (f (\theta , x _ {i}) - y _ {i}) \right] \tag {21} \\ = - \sum_ {i = 1} ^ {n} \left\langle \frac {d f (\theta , x _ {j})}{d \theta}, \frac {\partial f (\theta , x _ {i})}{\partial \theta} \right\rangle (f (\theta , x _ {i}) - y _ {i}). \\ \end{array}
$$

Given that $\mathcal { O } ( t ) = \{ f ( x _ { i } , \theta ( t ) ) \} _ { i = } ^ { n }$ <sub>1</sub> and $\mathbf { y } = \{ y _ { i } \} _ { i = 1 } ^ { n }$ , we can express Eq. 21 in vector form:

$$
\frac {d \mathcal {O} (t)}{d t} = - \mathcal {K} \cdot (\mathcal {O} (t) - \mathbf {y}),
$$

where $\kappa$ is defined as

$$
\mathcal {K} _ {u v} = \left\langle \frac {\partial f (\theta , x _ {u})}{\partial \theta}, \frac {\partial f (\theta , x _ {v})}{\partial \theta} \right\rangle .
$$

Q.E.D.

Discussions. Another equivalent definition of NTK given the Jacobian matrix J is:

$$
\mathcal {K} = J J ^ {\top}. \tag {22}
$$

This theorem shows that NTK connects the error term ${ \mathcal { O } } ( t ) - \mathbf { y }$ to the changing rate of the output. Therefore, the theory can be used to analyze the training behaviors of neural networks. □

## B.2 Proof of Theorem 3.2

Theorem 3.2. Let $\{ \mathcal { O } _ { 1 } ( t ) , \mathcal { O } _ { 2 } ( t ) , \ldots , \mathcal { O } _ { k } ( t ) \}$ denote the outputs of the neural network function $\{ f _ { 1 } , f _ { 2 } , \ldots , f _ { k } \}$ for the k task at time t, and let $\left\{ \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { k } \right\}$ represent the corresponding labels. Then, the ordinary differential equation in Eq. 2 gives the following evolution:

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \mathcal {K} _ {1 1} & \cdots & \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \mathcal {K} _ {k 1} & \cdots & \mathcal {K} _ {k k} \end{array} \right]} _ {\widetilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right],
$$

where ${ K _ { i j } } \in \mathbb { R } ^ { n \times n }$ and $\begin{array} { r } { \mathcal { K } _ { i j } = \mathcal { K } _ { j i } ^ { \top } f o r \mathrm { 1 } \le i , j \le k } \end{array}$ . The $( u , v )$ -th entry of $\mathcal { K } _ { i j }$ is defined as

$$
(\mathcal {K} _ {i j}) _ {u v} = \left\langle \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta}, \frac {\partial f _ {j} (\theta , x _ {v})}{\partial \theta} \right\rangle .
$$

Proof. In multi-task learning, a neural network with shared parameters θ is trained to simultaneously learn k distinct tasks. In the general case, the overall loss function is defined as

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {k} \ell_ {i} (\theta). \tag {23}
$$

Similar to the derivation in Theorem 3.1, by utilizing the gradient flow in Eq. 2, we can derive:

$$
\begin{array}{l} \frac {d \theta}{d t} = - \nabla_ {\theta} \mathcal {L} (\theta) \\ = - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \frac {\partial \ell_ {i}}{\partial f _ {i} (\theta , x _ {u})} \cdot \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \tag {24} \\ = - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \cdot (f _ {i} (\theta , x _ {u}) - y _ {i, u}), \\ \end{array}
$$

where $y _ { i , u }$ represents the ground truth label of the u-th element in the label set $\mathbf { y } _ { i }$ . For a data point $x _ { v }$ and the function $f _ { j }$ of task $j ,$ we have:

$$
\begin{array}{l} \frac {d f _ {j} (\theta , x _ {v})}{d t} = \frac {d f _ {j} (\theta , x _ {v})}{d \theta} \cdot \frac {d \theta}{d t} \\ = \frac {d f _ {j} (\theta , x _ {v})}{d \theta} \left[ - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \cdot (f _ {i} (\theta , x _ {u}) - y _ {i, u}) \right] \tag {25} \\ = - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \left\langle \frac {d f _ {j} (\theta , x _ {v})}{d \theta}, \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \right\rangle (f _ {i} (\theta , x _ {u}) - y _ {i, u}). \\ \end{array}
$$

Rewriting Eq. 25 in the form of high-dimensional vectors gives:

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \mathcal {K} _ {1 1} & \cdots & \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \mathcal {K} _ {k 1} & \cdots & \mathcal {K} _ {k k} \end{array} \right]} _ {\widetilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right],
$$

where ${ K _ { i j } } \in \mathbb { R } ^ { n \times n }$ and $\mathcal { K } _ { i j } = \mathcal { K } _ { i i } ^ { \top }$ for $1 \leq i , j \leq k$ . The $( u , v )$ -th entry of $\mathcal { K } _ { i j }$ is defined as

$$
(\mathcal {K} _ {i j}) _ {u v} = \left\langle \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta}, \frac {\partial f _ {j} (\theta , x _ {v})}{\partial \theta} \right\rangle .
$$

Q.E.D.

Discussions. It can be observed that Theorem 3.2 essentially extends the foundational NTK theory from Theorem 3.1 to the MTL scenario. In this case, the input, output, and NTK matrix each acquire an additional dimension, corresponding to the task-oriented dimension. □

## B.3 Proof of Proposition 3.3

Proposition 3.3. (Extension of Theorem 3.2) Let $\{ \mathcal { O } _ { i } ( t ) \} _ { i = 1 } ^ { k } , ~ \{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { k } ,$ , and $\{ K _ { i j } \} _ { 1 \leq i , j \leq k }$ be defined as in Theorem 3.2. We now replace the MTL optimization objective with the weightedform as presented in Eq. 12. Consequently, the ordinary differential equation governing the MTL training dynamics in Eq. 7 becomes:

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \omega_ {1} ^ {2} \mathcal {K} _ {1 1} & \cdots & \omega_ {1} \omega_ {k} \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \omega_ {k} \omega_ {1} \mathcal {K} _ {k 1} & \cdots & \omega_ {k} ^ {2} \mathcal {K} _ {k k} \end{array} \right]} _ {\boldsymbol {\omega} \boldsymbol {\omega} ^ {\top} \odot \tilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right]. \tag {26}
$$

Proof. Consider the weighted loss

$$
\mathcal {L} (\theta) = \sum_ {i = 1} ^ {k} \omega_ {i} \ell_ {i} (\theta), \tag {27}
$$

the analysis in Appendix B.2 becomes

$$
\begin{array}{l} \frac {d \theta}{d t} = - \nabla_ {\theta} \mathcal {L} (\theta) \\ = - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \frac {\partial \omega_ {i} \ell_ {i}}{\partial f _ {i} \left(\theta , x _ {u}\right)} \cdot \frac {\partial f _ {i} \left(\theta , x _ {u}\right)}{\partial \theta} \tag {28} \\ = - \sum_ {i = 1} ^ {k} \omega_ {i} \sum_ {u = 1} ^ {n} \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \cdot (f _ {i} (\theta , x _ {u}) - y _ {i, u}). \\ \end{array}
$$

Under the action of $\omega ,$ , the network output $\{ \mathcal { O } _ { 1 } , \ldots , \mathcal { O } _ { k } \}$ is transformed to $\{ \omega _ { 1 } f _ { 1 } , \ldots , \omega _ { k } f _ { k } \}$ . Consider the output of the network for the j-th task at a data point $x _ { v } )$

$$
\begin{array}{l} \frac {d \omega_ {j} f _ {j} (\theta , x _ {v})}{d t} = \omega_ {j} \frac {d f _ {j} (\theta , x _ {v})}{d \theta} \cdot \frac {d \theta}{d t} \\ = \omega_ {j} \frac {d f _ {j} \left(\theta , x _ {v}\right)}{d \theta} \left[ - \sum_ {i = 1} ^ {k} \omega_ {i} \sum_ {u = 1} ^ {n} \frac {\partial f _ {i} \left(\theta , x _ {u}\right)}{\partial \theta} \cdot \left(f _ {i} \left(\theta , x _ {u}\right) - y _ {i, u}\right) \right] \tag {29} \\ = - \sum_ {i = 1} ^ {k} \sum_ {u = 1} ^ {n} \omega_ {j} \omega_ {i} \left\langle \frac {d f _ {j} (\theta , x _ {v})}{d \theta}, \frac {\partial f _ {i} (\theta , x _ {u})}{\partial \theta} \right\rangle (f _ {i} (\theta , x _ {u}) - y _ {i, u}). \\ \end{array}
$$

Then we have

$$
\left[ \begin{array}{c} \frac {d \mathcal {O} _ {1} (t)}{d t} \\ \vdots \\ \frac {d \mathcal {O} _ {k} (t)}{d t} \end{array} \right] = - \underbrace {\left[ \begin{array}{c c c} \omega_ {1} ^ {2} \mathcal {K} _ {1 1} & \cdots & \omega_ {1} \omega_ {k} \mathcal {K} _ {1 k} \\ \vdots & \ddots & \vdots \\ \omega_ {k} \omega_ {1} \mathcal {K} _ {k 1} & \cdots & \omega_ {k} ^ {2} \mathcal {K} _ {k k} \end{array} \right]} _ {\boldsymbol {\omega} \boldsymbol {\omega} ^ {\top} \odot \widetilde {\mathcal {K}}} \left[ \begin{array}{c} \mathcal {O} _ {1} (t) - \mathbf {y} _ {1} \\ \vdots \\ \mathcal {O} _ {k} (t) - \mathbf {y} _ {k} \end{array} \right]
$$

Q.E.D.

Discussions. Proposition 3.3 actually provides an intuition: by adjusting the relative magnitudes of $\{ \omega _ { i } \} _ { i = 1 } ^ { k }$ , one can alter the eigenvalue distribution of the NTK matrix, thereby balancing convergence speeds. This serves as the foundation for the design of NTKMTL and NTKMTL-SR. □

## C Detailed Experimental Results

## C.1 Detailed Results on QM9

For QM9 benchmark, previous methods were all implemented based on a shared codebase [39, 31, 4], utilizing the message-passing neural network (MPNN) architecture [19]. All the methods were trained for 300 epochs. However, we found that the hyperparameter settings in this codebase were suboptimal (specifically, the improper learning rate scheduler makes the learning rate decay too quickly). Consequently, the reported results of most previous methods had not fully converged, potentially leading to unfair comparisons. Therefore, we readjusted the hyperparameters: we changed the batch size from 120 to 60 and the learning rate scheduler’s patience from 5 to 10. The remaining hyperparameters were kept unchanged, consistent with the original codebase. Subsequently, we reproduced all baselines that had reported results on QM9, and their performance on various tasks significantly improved under these revised settings. To ensure accurate evaluation, we also reproduced the 11-task STL baselines under the same settings.

Experimental results are presented in Table 6. For each baseline, the upper row shows results taken from its original paper, while the bottom row (highlighted in green ) presents results reproduced by us under new hyperparameter settings. It can be seen that without changing the model architecture, solely by adjusting the learning rate schedule, the performance of all baselines on the 11 tasks has significantly improved, notably extending the Pareto front.

Under the new settings, with both STL baselines and MTL methods showing significant performance improvements, we observed different behaviors among previous methods. The final $\Delta \bar { m } \%$ of some methods (e.g., LS, RLW, PCGrad, CAGrad) were largely consistent with their originally reported values. For other methods (e.g., SI, UW, FAMO, GO4Align), the $\Delta m \%$ significantly improved compared to the reported results. This indicates that the previous hyperparameter settings fail to fully exhibit the capabilities of these methods. Under the new settings that better ensure convergence, their performance shows further improvement. Notably, the traditional Scale-Invariant Linear Scalarization (SI) demonstrated extremely superior performance, surpassing the vast majority of recent baselines. This suggests that in cases where differences in loss scales among different tasks are too large, directly eliminating scale differences through logarithmic methods may be an effective solution.

Table 6: Results on QM9 (11-task) dataset. For each baseline, the upper row shows results taken from its original paper, while the bottom row (highlighted in green ) presents results reproduced by us under new hyperparameter settings.

<table><tr><td rowspan="2">METHOD</td><td> $\mu$ </td><td> $\alpha$ </td><td> $\epsilon_{HOMO}$ </td><td> $\epsilon_{LUMO}$ </td><td> $\langle R^2\rangle$ </td><td>ZPVE</td><td> $U_0$ </td><td>U</td><td>H</td><td>G</td><td> $c_v$ </td><td rowspan="2">MR↓</td><td rowspan="2"> $\Delta m\% \downarrow$ </td></tr><tr><td colspan="11">MAE ↓</td></tr><tr><td>STL</td><td>0.067</td><td>0.181</td><td>60.57</td><td>53.91</td><td>0.502</td><td>4.53</td><td>58.8</td><td>64.2</td><td>63.8</td><td>66.2</td><td>0.072</td><td></td><td></td></tr><tr><td>STL</td><td>0.060</td><td>0.156</td><td>60.54</td><td>51.22</td><td>0.419</td><td>3.08</td><td>39.3</td><td>42.9</td><td>41.7</td><td>43.1</td><td>0.061</td><td></td><td></td></tr><tr><td>LS</td><td>0.106</td><td>0.325</td><td>73.57</td><td>89.67</td><td>5.19</td><td>14.06</td><td>143.4</td><td>144.2</td><td>144.6</td><td>140.3</td><td>0.128</td><td></td><td>177.6</td></tr><tr><td>LS</td><td>0.077</td><td>0.253</td><td>55.95</td><td>68.59</td><td>4.163</td><td>11.08</td><td>109.2</td><td>109.8</td><td>110.1</td><td>106.7</td><td>0.099</td><td>9.82</td><td>179.8</td></tr><tr><td>SI</td><td>0.309</td><td>0.345</td><td>149.8</td><td>135.7</td><td>1.00</td><td>4.50</td><td>55.3</td><td>55.75</td><td>55.82</td><td>55.27</td><td>0.112</td><td></td><td>77.8</td></tr><tr><td>SI</td><td>0.159</td><td>0.242</td><td>109.6</td><td>96.80</td><td>0.732</td><td>3.308</td><td>34.35</td><td>34.37</td><td>34.33</td><td>35.16</td><td>0.081</td><td>5.27</td><td>39.7</td></tr><tr><td>RLW [28]</td><td>0.113</td><td>0.340</td><td>76.95</td><td>92.76</td><td>5.86</td><td>15.46</td><td>156.3</td><td>157.1</td><td>157.6</td><td>153.0</td><td>0.137</td><td></td><td>203.8</td></tr><tr><td>RLW [28]</td><td>0.090</td><td>0.277</td><td>62.49</td><td>76.39</td><td>4.948</td><td>12.54</td><td>124.8</td><td>124.9</td><td>125.0</td><td>122.1</td><td>0.115</td><td>12.09</td><td>222.6</td></tr><tr><td>DWA [35]</td><td>0.107</td><td>0.325</td><td>74.06</td><td>90.61</td><td>5.09</td><td>13.99</td><td>142.3</td><td>143.0</td><td>143.4</td><td>139.3</td><td>0.125</td><td></td><td>175.3</td></tr><tr><td>DWA [35]</td><td>0.078</td><td>0.239</td><td>55.17</td><td>67.40</td><td>3.992</td><td>10.92</td><td>107.4</td><td>108.1</td><td>108.3</td><td>105.4</td><td>0.098</td><td>8.73</td><td>173.0</td></tr><tr><td>UW [25]</td><td>0.386</td><td>0.425</td><td>166.2</td><td>155.8</td><td>1.06</td><td>4.99</td><td>66.4</td><td>66.78</td><td>66.80</td><td>66.24</td><td>0.122</td><td></td><td>108.0</td></tr><tr><td>UW [25]</td><td>0.194</td><td>0.274</td><td>120.6</td><td>102.8</td><td>0.763</td><td>3.698</td><td>41.11</td><td>41.13</td><td>41.16</td><td>41.75</td><td>0.089</td><td>8.27</td><td>58.4</td></tr><tr><td>MGDA [46]</td><td>0.217</td><td>0.368</td><td>126.8</td><td>104.6</td><td>3.22</td><td>5.69</td><td>88.37</td><td>89.4</td><td>89.32</td><td>88.01</td><td>0.120</td><td></td><td>120.5</td></tr><tr><td>MGDA [46]</td><td>0.154</td><td>0.266</td><td>95.20</td><td>67.51</td><td>3.088</td><td>4.468</td><td>49.38</td><td>49.21</td><td>49.62</td><td>49.69</td><td>0.087</td><td>8.36</td><td>101.4</td></tr><tr><td>PCGRAD [58]</td><td>0.106</td><td>0.293</td><td>75.85</td><td>88.33</td><td>3.94</td><td>9.15</td><td>116.36</td><td>116.8</td><td>117.2</td><td>114.5</td><td>0.110</td><td></td><td>125.7</td></tr><tr><td>PCGRAD [58]</td><td>0.078</td><td>0.221</td><td>59.14</td><td>67.82</td><td>2.937</td><td>6.691</td><td>88.24</td><td>88.65</td><td>88.85</td><td>87.36</td><td>0.084</td><td>7.91</td><td>118.6</td></tr><tr><td>CAGRAD [32]</td><td>0.118</td><td>0.321</td><td>83.51</td><td>94.81</td><td>3.21</td><td>6.93</td><td>113.99</td><td>114.3</td><td>114.5</td><td>112.3</td><td>0.116</td><td></td><td>112.8</td></tr><tr><td>CAGRAD [32]</td><td>0.083</td><td>0.234</td><td>57.80</td><td>70.98</td><td>2.718</td><td>5.352</td><td>76.47</td><td>76.93</td><td>77.05</td><td>76.32</td><td>0.089</td><td>8.27</td><td>102.4</td></tr><tr><td>NASH-MTL [39]</td><td>0.102</td><td>0.248</td><td>82.95</td><td>81.89</td><td>2.42</td><td>5.38</td><td>74.5</td><td>75.02</td><td>75.10</td><td>74.16</td><td>0.093</td><td></td><td>62.0</td></tr><tr><td>NASH-MTL [39]</td><td>0.086</td><td>0.218</td><td>69.78</td><td>66.18</td><td>2.153</td><td>4.679</td><td>59.63</td><td>59.94</td><td>59.98</td><td>59.97</td><td>0.082</td><td>6.91</td><td>72.9</td></tr><tr><td>FAMO [31]</td><td>0.15</td><td>0.30</td><td>94.0</td><td>95.2</td><td>1.63</td><td>4.95</td><td>70.82</td><td>71.2</td><td>71.2</td><td>70.3</td><td>0.10</td><td></td><td>58.5</td></tr><tr><td>FAMO [31]</td><td>0.128</td><td>0.230</td><td>98.09</td><td>84.42</td><td>0.859</td><td>3.541</td><td>40.24</td><td>40.57</td><td>40.62</td><td>40.21</td><td>0.081</td><td>5.91</td><td>38.9</td></tr><tr><td>FAIRGRAD [4]</td><td>0.117</td><td>0.253</td><td>87.57</td><td>84.00</td><td>2.15</td><td>5.07</td><td>70.89</td><td>71.17</td><td>71.21</td><td>70.88</td><td>0.095</td><td></td><td>57.9</td></tr><tr><td>FAIRGRAD [4]</td><td>0.109</td><td>0.208</td><td>81.74</td><td>72.82</td><td>1.669</td><td>3.418</td><td>51.31</td><td>51.67</td><td>51.72</td><td>51.97</td><td>0.079</td><td>6.64</td><td>57.0</td></tr><tr><td>GO4ALIGN [48]</td><td>0.17</td><td>0.35</td><td>102.4</td><td>119.0</td><td>1.22</td><td>4.94</td><td>53.9</td><td>54.3</td><td>54.3</td><td>53.9</td><td>0.11</td><td></td><td>52.7</td></tr><tr><td>GO4ALIGN [48]</td><td>0.113</td><td>0.314</td><td>74.46</td><td>91.04</td><td>0.912</td><td>3.632</td><td>36.06</td><td>36.38</td><td>36.41</td><td>36.58</td><td>0.104</td><td>6.64</td><td>40.5</td></tr><tr><td>NTKMTL</td><td>0.091</td><td>0.212</td><td>70.97</td><td>70.81</td><td>2.113</td><td>3.835</td><td>44.18</td><td>44.56</td><td>44.53</td><td>44.38</td><td>0.077</td><td>5.91</td><td>56.7</td></tr><tr><td>NTKMTL-SR</td><td>0.081</td><td>0.207</td><td>75.95</td><td>69.10</td><td>1.176</td><td>3.689</td><td>40.14</td><td>40.46</td><td>40.48</td><td>40.49</td><td>0.074</td><td>4.00</td><td>30.7</td></tr></table>

In summary, we identified that the parameter settings in the previous code implementation hindered the smooth convergence of both STL and MTL methods, rendering comparisons made under these conditions unfair. Our experiments verified that this issue can be resolved by simply adjusting the batch size and learning rate scheduler parameters. Under the new settings that better ensure convergence, the performance of both STL methods and numerous MTL methods on the 11 tasks has significantly improved. We believe that fair and transparent reproduction of all previous baseline methods under such parameter settings enables more reasonable comparisons on this benchmark, providing new results that are valuable to the MTL community.

## C.2 Detailed Results with Standard Errors

In this section, We provide the detailed experimental results with standard errors for our method, and the results for the baseline methods are taken from their original papers. Since results for CelebA are not reported by some methods [47, 54, 48], these methods are excluded when presenting combined CityScapes and CelebA results in Table 2. Table 7 provides detailed results solely on the CityScapes dataset, including these methods. Consequently, the mean rank (MR) in Table 7 slightly differs from that in Table 2.

On NYUv2 and CityScapes, we follow the training settings of [39, 4], including data augmentation for all compared methods. Training runs for 200 epochs, with the learning rate initialized at $1 0 ^ { - 4 }$ and reduced to $\mathrm { 5 \times 1 0 ^ { - 5 } }$ after 100 epochs. The architecture is the SegNet-based [3] Multi-Task Attention Network (MTAN) [33]. Batch sizes are 2 (NYUv2) and 8 (CityScapes), and the hyparameter n for NTKMTL-SR on NYUv2 is set to 2. Our setup for the CelebA benchmark aligns with the configuration detailed in [31]. We employ a 9-layer CNN as the network backbone, coupled with separate linear layers for each task. The method is trained for 15 epochs; optimization is carried out using Adam with a batch size of 256.

Table 7: Detailed results on CityScapes (2-task) dataset. Each experiment is repeated 3 times with different random seeds and the average is reported. The best scores are reported in gray .

<table><tr><td rowspan="3">METHOD</td><td colspan="6">CITYSCAPES</td></tr><tr><td colspan="2">SEGMENTATION</td><td colspan="2">DEPTH</td><td rowspan="2">MR↓</td><td rowspan="2">Δm% ↓</td></tr><tr><td>MIoU ↑</td><td>PIX ACC ↑</td><td>ABS ERR ↓</td><td>REL ERR ↓</td></tr><tr><td>STL</td><td>74.01</td><td>93.16</td><td>0.0125</td><td>27.77</td><td></td><td></td></tr><tr><td>LS</td><td>75.18</td><td>93.49</td><td>0.0155</td><td>46.77</td><td>10.25</td><td>22.60</td></tr><tr><td>SI</td><td>70.95</td><td>91.73</td><td>0.0161</td><td>33.83</td><td>14.00</td><td>14.11</td></tr><tr><td>RLW [28]</td><td>74.57</td><td>93.41</td><td>0.0158</td><td>47.79</td><td>12.25</td><td>24.38</td></tr><tr><td>DWA [35]</td><td>75.24</td><td>93.52</td><td>0.0160</td><td>44.37</td><td>9.75</td><td>21.45</td></tr><tr><td>UW [25]</td><td>72.02</td><td>92.85</td><td>0.0140</td><td>30.13</td><td>10.00</td><td>5.89</td></tr><tr><td>MGDA [46]</td><td>68.84</td><td>91.54</td><td>0.0309</td><td>33.50</td><td>14.75</td><td>44.14</td></tr><tr><td>PCGRAD [58]</td><td>75.13</td><td>93.48</td><td>0.0154</td><td>42.07</td><td>10.50</td><td>18.29</td></tr><tr><td>CAGRAD [32]</td><td>75.16</td><td>93.48</td><td>0.0141</td><td>37.60</td><td>9.25</td><td>11.64</td></tr><tr><td>IMTL-G [33]</td><td>75.33</td><td>93.49</td><td>0.0135</td><td>38.41</td><td>7.25</td><td>11.10</td></tr><tr><td>NASH-MTL [39]</td><td>75.41</td><td>93.66</td><td>0.0129</td><td>35.02</td><td>4.75</td><td>6.82</td></tr><tr><td>FAMO [31]</td><td>74.54</td><td>93.29</td><td>0.0145</td><td>32.59</td><td>9.25</td><td>8.13</td></tr><tr><td>ALIGNED-MTL [47]</td><td>75.77</td><td>93.69</td><td>0.0133</td><td>32.66</td><td>3.00</td><td>5.27</td></tr><tr><td>SDMGRAD [54]</td><td>74.53</td><td>93.52</td><td>0.0137</td><td>34.01</td><td>8.25</td><td>7.74</td></tr><tr><td>GO4ALIGN [48]</td><td>72.63</td><td>93.03</td><td>0.0164</td><td>27.58</td><td>10.75</td><td>8.13</td></tr><tr><td>FAIRGRAD [4]</td><td>75.72</td><td>93.68</td><td>0.0134</td><td>32.25</td><td>3.25</td><td>5.18</td></tr><tr><td>NTKMTL</td><td>73.71</td><td>92.71</td><td>0.0136</td><td>27.21</td><td>8.50</td><td>1.92</td></tr><tr><td>NTKMTL-SR</td><td>72.58</td><td>92.93</td><td>0.0124</td><td>31.65</td><td>8.00</td><td>3.84</td></tr></table>

Table 8: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean and stderr are reported.

<table><tr><td rowspan="3">Method</td><td colspan="5">CityScapes</td><td>CelebA</td></tr><tr><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="2">Δm% ↓</td><td rowspan="2">Δm% ↓</td></tr><tr><td>mIoU ↑</td><td>Pix Acc ↑</td><td>Abs Err ↓</td><td>Rel Err ↓</td></tr><tr><td>NTKMTL (mean)</td><td>73.71</td><td>92.71</td><td>0.0136</td><td>27.21</td><td>1.92</td><td>-0.77</td></tr><tr><td>NTKMTL (stderr)</td><td>±0.17</td><td>±0.15</td><td>±0.0005</td><td>±0.23</td><td>±0.30</td><td>±0.37</td></tr><tr><td>NTKMTL-SR (mean)</td><td>72.58</td><td>92.93</td><td>0.0124</td><td>31.65</td><td>3.84</td><td>0.23</td></tr><tr><td>NTKMTL-SR (stderr)</td><td>±0.32</td><td>±0.23</td><td>±0.0004</td><td>±0.35</td><td>±0.37</td><td>±0.46</td></tr></table>

Table 9: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean and stderr are reported.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m\% \downarrow$ </td></tr><tr><td rowspan="2">mIoU ↑</td><td rowspan="2">Pix Acc ↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Dist ↓</td><td colspan="3">Within  $t^{\circ} \uparrow$ </td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>NTKMTL (mean)</td><td>39.68</td><td>65.43</td><td>0.5296</td><td>0.2168</td><td>24.24</td><td>18.63</td><td>30.74</td><td>58.72</td><td>70.78</td><td>-6.99</td></tr><tr><td>NTKMTL (stderr)</td><td>±0.51</td><td>±0.24</td><td>±0.0008</td><td>±0.0014</td><td>±0.07</td><td>±0.09</td><td>±0.14</td><td>±0.19</td><td>±0.19</td><td>±0.38</td></tr><tr><td>NTKMTL-SR (mean)</td><td>40.23</td><td>65.28</td><td>0.5261</td><td>0.2136</td><td>24.88</td><td>19.58</td><td>29.53</td><td>56.67</td><td>69.08</td><td>-5.35</td></tr><tr><td>NTKMTL-SR (stderr)</td><td>±0.42</td><td>±0.26</td><td>±0.0013</td><td>±0.0014</td><td>±0.11</td><td>±0.13</td><td>±0.12</td><td>±0.15</td><td>±0.16</td><td>±0.31</td></tr></table>

## C.3 Visualization experiments for the NTK eigenvalues during training

To support the proposed theory, we conducted experiments on NYUv2, visualizing the change in the maximum eigenvalue of the NTK matrix for the three tasks during training under linear scalarization (i.e., equal weighting). On the NYUv2 dataset, the difficulty levels of the three tasks show significant variation. Previous methods generally outperform the Single Task Learning (STL) baseline in segmentation and depth estimation tasks, but almost all of them consistently underperform STL on the surface normal prediction task, leading to a significant task imbalance in the overall results.

![](images/f431891cb43db208088a8573f580c3dd7226bce5adbbf8e4715c5c9d0bc5caf1.jpg)

<details>
<summary>line</summary>

| Iteration | Segmentation | Depth | Normal |
| --- | --- | --- | --- |
| 0 | ~5 | ~10 | ~0.7 |
| 1000 | ~1 | ~1 | ~0.2 |
| 2000 | ~1 | ~1 | ~0.3 |
| 3000 | ~4 | ~2 | ~0.3 |
| 4000 | ~4 | ~2 | ~0.3 |
| 5000 | ~6 | ~1 | ~0.3 |
</details>

Figure 2: Visualization for the NTK eigenvalues during training.

As shown in Fig. 2, throughout training, the largest eigenvalue corresponding to surface normal prediction remains substantially smaller than those of segmentation and depth estimation. Given that the maximum NTK eigenvalue reflects a task’s convergence speed, this observation aligns with the empirical finding that many existing MTL algorithms struggle to converge on surface normal prediction. Although some prior methods (e.g., MGDA) attempt to prioritize the most difficult tasks, their performance in surface normal prediction tasks remains unsatisfactory due to the challenge in accurately quantifying the "difficulty" and "convergence speed" of different tasks. In contrast, by leveraging NTK theory to accurately characterize and balance the convergence speed of each task during training, NTKMTL delivers SOTA results on surface normal prediction and is one of only two methods that outperform single-task learning on all three tasks.

## C.4 Ablation Study on the Hyperparameter n

For NTKMTL-SR, the computational cost of calculating the NTK is minimal, allowing us to further investigate the impact of varying mini-batch sizes n on the results. Therefore, we set n to [1, 2, 3, 4, 6] and conduct an ablation study on the QM9 (11-task) benchmark. For each value of $n ,$ we conduct 3 repeated experiments with different random seeds and calculate the mean and variance for $\Delta m \%$ The results are shown in Fig. 3. When n = 1, the performance of NTKMTL-SR is comparable to that of NTKMTL. However, when increasing n from 1 to 2 or more, NTKMTL-SR shows a noticeable improvement in performance, accompanied by a reduction in the variance of performance across repeated experiments. We attribute this to the fact that increasing the number of mini-batches leads to a larger NTK matrix dimension, which in turn reduces stochastic error and allows our method to more accurately characterize the convergence speed of the tasks.

![](images/731e1cdd6f92818d5aaa372e447ab0d875b577809c07f3eebf9336d377b8f059.jpg)

<details>
<summary>line</summary>

| Hyperparameter n | \(\Delta m% (\)Mean) | \(\Delta m% (\)Lower Bound) | \(\Delta m% (\)Upper Bound) |
| --- | --- | --- | --- |
| 1 | ~48 | ~38 | ~59 |
| 2 | ~34 | ~27 | ~40 |
| 3 | ~36 | ~30 | ~41 |
| 4 | ~31 | ~26 | ~36 |
| 6 | ~30 | ~25 | ~35 |
</details>

Figure 3: Ablation study on hyperparameter n on QM9. Each experiment is repeated over 3 random seeds, and the mean and stderr are reported.

However, we also find that the results for $n = 4$ and $n = 6$ are almost identical, and the performance differences observed were potentially weaker than the inherent variability stemming from different random seeds. Concurrently, Fig. 4 visualizes the training time per epoch for various n values on QM9. Despite only requiring the computation of the maximum eigenvalue of the NTK matrix with respect to z, training a single epoch when $n = 6$ already approached 1.7 times the duration of the LS method. Overall, we posit that the selection of hyperparameter n is a trade-off between performance and training speed, and n = 4 generally presents a favorable compromise.

![](images/96514c058f49f5882b28ddfdbc60e2ca31ca972ebaca30906b4db50711fd4f1c.jpg)

<details>
<summary>bar</summary>

| Category | Time (min) |
| --- | --- |
| LS | 1.50 |
| n=1 | 1.05 |
| n=2 | 1.14 |
| n=3 | 1.27 |
| n=4 | 1.40 |
| n=6 | 1.69 |
</details>

Figure 4: Training time per epoch for LS and NTKMTL-SR (with different hyperparameter n) on QM9.