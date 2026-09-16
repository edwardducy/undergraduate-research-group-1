# TOWARDS IMPARTIAL MULTI-TASK LEARNING

Liyang Liu<sup>1</sup>, Yi Li<sup>2</sup>, Zhanghui Kuang<sup>2</sup>, Jing-Hao Xue<sup>3</sup>, Yimin Chen<sup>2</sup>, Wenming Yang<sup>1∗</sup>, Qingmin Liao<sup>1</sup>, Wayne Zhang<sup>2,4</sup>

<sup>1</sup>Shenzhen International Graduate School/Department of Electronic Engineering, Tsinghua University  
<sup>2</sup>SenseTime Research  
<sup>3</sup>Department of Statistical Science, University College London  
<sup>4</sup>Qing Yuan Research Institute, Shanghai Jiao Tong University {liu-ly14@mails., yang.wenming@sz., liaoqm@}tsinghua.edu.cn {liyi, kuangzhanghui, chenyimin, wayne.zhang}@sensetime.com jinghao.xue@ucl.ac.uk

## ABSTRACT

Multi-task learning (MTL) has been widely used in representation learning. However, na¨ıvely training all tasks simultaneously may lead to the partial training issue, where specific tasks are trained more adequately than others. In this paper, we propose to learn multiple tasks impartially. Specifically, for the task-shared parameters, we optimize the scaling factors via a closed-form solution, such that the aggregated gradient (sum of raw gradients weighted by the scaling factors) has equal projections onto individual tasks. For the task-specific parameters, we dynamically weigh the task losses so that all of them are kept at a comparable scale. Further, we find the above gradient balance and loss balance are complementary and thus propose a hybrid balance method to further improve the performance. Our impartial multi-task learning (IMTL) can be end-to-end trained without any heuristic hyper-parameter tuning, and is general to be applied on all kinds of losses without any distribution assumption. Moreover, our IMTL can converge to similar results even when the task losses are designed to have different scales, and thus it is scale-invariant. We extensively evaluate our IMTL on the standard MTL benchmarks including Cityscapes, NYUv2 and CelebA. It outperforms existing loss weighting methods under the same experimental settings.

## 1 INTRODUCTION

Recent deep networks in computer vision can match or even surpass human beings on some specific tasks separately. However, in reality multiple tasks (e.g., semantic segmentation and depth estimation) must be solved simultaneously. Multi-task learning (MTL) (Caruana, 1997; Evgeniou & Pontil, 2004; Ruder, 2017; Zhang & Yang, 2017) aims at sharing the learned representation among tasks (Zamir et al., 2018) to make them benefit from each other and achieve better results and stronger robustness (Zamir et al., 2020). However, sharing the representation can lead to a partial learning issue: some specific tasks are learned well while others are overlooked, due to the different loss scales or gradient magnitudes of various tasks and the mutual competition among them. Several methods have been proposed to mitigate this issue either via gradient balance such as gradient magnitude normalization (Chen et al., 2018) and Pareto optimality (Sener & Koltun, 2018), or loss balance like homoscedastic uncertainty (Kendall et al., 2018). Gradient balance can evenly learn task-shared parameters while ignoring task-specific ones. Loss balance can prevent MTL from being biased in favor of tasks with large loss scales but cannot ensure the impartial learning of the shared parameters. In this work, we find that gradient balance and loss balance are complementary, and combining the two balances can further improve the results. To this end, we propose impartial MTL (IMTL) via simultaneously balancing gradients and losses across tasks.

For gradient balance, we propose IMTL-G(rad) to learn the scaling factors such that the aggregated gradient of task-shared parameters has equal projections onto the raw gradients of individual tasks (see Fig. 1 (d)). We show that the scaling factor optimization problem is equivalent to finding the angle bisector of gradients from all tasks in geometry, and derive a closed-form solution to it. In contrast with previous gradient balance methods such as GradNorm (Chen et al., 2018), MGDA (Sener & Koltun, 2018) and PCGrad (Yu et al., 2020), which have learning biases in favor of tasks with gradients close to the average gradient direction, those with small gradient magnitudes, and those with large gradient magnitudes, respectively (see Fig. 1 (a), (b) and (c)), in our IMTL-G task-shared parameters can be updated without bias to any task.

![](images/cfb6faaeec1fdc085989b827df75f23a9233b1fa24f80cd15a742a896f157a9e.jpg)  
(a) GradNorm

![](images/9cbb0ed4d9d2668a3f94563c13651b94ece3ca3e1dcc1c762ccddc06f8de9143.jpg)  
(b) MGDA

![](images/3f91d581746d94d0a835203c83a3a55c23c093b710ff09be3fcf00971a4a6154.jpg)  
(c) PCGrad

![](images/f0ad6b89af64872583a2bf74e8382e0619573a05acaf4560e529619de8c5a13c.jpg)  
(d) IMTL-G  
Figure 1: Comparison of gradient balance methods. In (a) to (d), g<sub>1</sub>, g<sub>2</sub> and g<sub>3</sub> represent the gradient computed by the raw loss of each task, respectively. The gray surface represents the plane composed by these gradients. The red arrow denotes the aggregated gradient computed by the weighted sum loss, which is ultimately used to update the model parameters. The blue arrows show the projections of g onto the raw gradients {g<sub>t</sub>}. g has the largest projection on g<sub>2</sub> (nearest to the mean direction), g<sub>3</sub> (smallest magnitude) and g<sub>2</sub> (largest magnitude) for GradNorm, MGDA and PCGrad, respectively, while the projections are equal on {g } in our IMTL-G.

For loss balance, we propose IMTL-L(oss) to automatically learn a loss weighting parameter for each task so that the weighted losses have comparable scales and the effect of different loss scales from various tasks can be canceled-out. Compared with uncertainty weighting (Kendall et al., 2018), which has biases towards regression tasks rather than classification tasks, our IMTL-L treats all tasks equivalently without any bias. Besides, we model the loss balance problem from the optimization perspective without any distribution assumption that is required by (Kendall et al., 2018). Therefore, ours is more general and can be used in any kinds of losses. Moreover, the loss weighting parameters and the network parameters can be jointly learned in an end-to-end fashion in IMTL-L.

Further, we find the above two balances are complementary and can be combined to improve the performance. Specifically, we apply IMTL-G on the task-shared parameters and IMTL-L on the task-specific parameters, leading to the hybrid balance method IMTL. Our IMTL is scale-invariant: the model can converge to similar results even when the same task is designed to have different loss scales, which is common in practice. For example, the scale of the cross-entropy loss in semantic segmentation may have different scales when using “average” or “sum” reduction over locations in the loss computation. We empirically validate that our IMTL is more robust against heavy loss scale changes than its competitors. Meanwhile, our IMTL only adds negligible computational overheads.

We extensively evaluate our proposed IMTL on standard benchmarks: Cityscapes, NYUv2 and CelebA, where the experimental results show that IMTL achieves superior performances under all settings. Besides, considering there lacks a fair and practical benchmark for comparing MTL methods, we unify the experimental settings such as image resolution, data augmentation, network structure, learning rate and optimizer option. We re-implement and compare with the representative MTL methods in a unified framework, which will be publicly available. Our contributions are:

• We propose a novel closed-form gradient balance method, which learns task-shared parameters without any task bias; and we develop a general learnable loss balance method, where no distribution assumption is required and the scale parameters can be jointly trained with the network parameters.  
• We unveil that gradient balance and loss balance are complementary and accordingly propose a hybrid balance method to simultaneously balance gradients and losses.  
• We validate that our proposed IMTL is loss scale-invariant and is more robust against loss scale changes compared with its competitors, and we give in-depth theoretical and experimental analyses on its connections and differences with previous methods.  
• We extensively verify the effectiveness of our IMTL. For fair comparisons, a unified codebase will also be publicly available, where more practical settings are adopted and stronger performances are achieved compared with existing code-bases.

## 2 RELATED WORK

Recent advances in MTL mainly come from two aspects: network structure improvements and loss weighting developments. Network-structure methods based on soft parameter-sharing usually lead to high inference cost (review in Appendix A). Loss weighting methods find loss weights to be multiplied on the raw losses for model optimization. They employ a hard parameter-sharing paradigm (Ruder, 2017), where several light-weight task-specific heads are attached upon the heavy-weight task-agnostic backbone. There are also efforts that learn to group tasks and branch the network in the middle layers (Guo et al., 2020; Standley et al., 2020), which try to achieve better accuracyefficiency trade-off and can be seen as semi-hard parameter-sharing. We believe task grouping and loss weighting are orthogonal and complementary directions to facilitate multi-task learning and can benefit from each other. In this work we focus on loss weighting methods which are the most economic as almost all of the computations are shared across tasks, leading to high inference speed. Task Prioritization (Guo et al., 2018) weights task losses by their difficulties to focus on the harder tasks during training. Uncertainty weighting (Kendall et al., 2018) models the loss weights as dataagnostic task-dependent homoscedastic uncertainty. Then loss weighting is derived from maximum likelihood estimation. GradNorm (Chen et al., 2018) learns the loss weights to enforce the norm of the scaled gradient for each task to be close. MGDA (Sener & Koltun, 2018) casts multi-task learning as multi-object optimization and finds the minimum-norm point in the convex hull composed by the gradients of multiple tasks. Pareto optimality is supposed to be achieved under mild conditions. GLS (Chennupati et al., 2019) instead uses the geometric mean of task-specific losses as the target loss, we will show it actually weights the loss by its reciprocal value. PCGrad (Yu et al., 2020) avoids interferences between tasks by projecting the gradient of one task onto the normal plane of the other. DSG (Lu et al., 2020) dynamically makes a task “stop or $\mathrm { g o } ^ { \mathrm { \prime \prime } }$ by its converging state, where a task is updated only once for a while if it is stopped. Although many loss weighting methods have been proposed, they are seldom open-sourced and rarely compared thoroughly under practical settings where strong performances are achieved, which motivates us to give an in-depth analysis and a fair comparison about them.

## 3 IMPARTIAL MULTI-TASK LEARNING

In MTL, we map a sample x ∈ X to its labels $\{ y _ { t } \in \mathbb { Y } _ { t } \} _ { t \in [ 1 , T ] }$ of all $T$ tasks through multiple taskspecific mappings $\{ f _ { t } : \mathbb { X } \to \mathbb { Y } _ { t } \}$ . In most loss weighting methods, the hard parameter-sharing paradigm is employed, such that $\pmb { f } _ { t }$ is parameterized by heavy-weight task-shared parameters $\pmb \theta$ and light-weight task-specific parameters $\theta _ { t }$ . All tasks take the same shared intermediate feature ${ z } = { f } \left( { { \mathbf { x } } ; { \theta } } \right)$ as input, and the t-th task head outputs the prediction as ${ \pmb f } _ { t } \left( \pmb { x } \right) = { \pmb f } _ { t } \left( \pmb { z } ; { \pmb \theta } _ { t } \right)$ . We aim to find the scaling factors $\left\{ \alpha _ { t } \right\}$ for all $T$ task losses $\{ \hat { L _ { t } } \left( \pmb { f } _ { t } \left( \pmb { x } \right) , \pmb { y } _ { t } \right) \}$ , so that the weighted sum loss $\begin{array} { r } { L = \sum _ { t } \alpha _ { t } L _ { t } } \end{array}$ can be optimized to make all tasks perform well. This poses great challenges because: 1) losses may have distinguished forms such as cross-entropy loss and cosine similarity; 2) the dynamic ranges of losses may differ by orders of magnitude. In this work, we propose a hybrid solution for both the task-shared parameters $\pmb \theta$ and the task-specific parameters $\{ \pmb \theta _ { t } \}$ , as Fig. 2.

## 3.1 GRADIENT BALANCE: IMTL-G

For task-shared parameters $\theta ,$ we can receive $T$ gradients $\{ g _ { t } = \nabla _ { \theta } L _ { t } \}$ via back-propagation from all of the $T$ raw losses $\{ L _ { t } \}$ , and these gradients represent optimal update directions for individual tasks. As the parameters $\pmb { \theta }$ can only be updated with a single gradient, we should compute an aggregated gradient g by the linear combination of $\left\{ \pmb { g } _ { t } \right\}$ . It also implies to find the scaling factors $\left\{ \alpha _ { t } \right\}$ of raw losses $\{ L _ { t } \}$ since $\begin{array} { r } { \pmb { g } = \sum _ { t } ^ { } \alpha _ { t } \pmb { g } _ { t } = \dot { \nabla } \pmb { \theta } \dot { L } = \nabla _ { \pmb { \theta } } \left( \sum _ { t } \dot { \alpha _ { t } } L _ { t } \right) } \end{array}$ Motivated by the principle of balance among tasks, we propose to make the projections of $\textbf {  { g } }$ onto $\left\{ \pmb { g } _ { t } \right\}$ to be equal, as Fig. 1 (d). In this way,

![](images/495fea7e4c44292b0d823e2a559c5dd941c9f10bddbafe16bf17e7c3aa29a072.jpg)

<details>
<summary>text_image</summary>

Impartial
Loss Balance
Gradient Balance
Multi-task Learning
f₁(z; θ₁)
θ₁
f₂(z; θ₂)
θ₂
f₃(z; θ₃)
θ₃
z = f(x; θ)
Heads
Shared Feature
Backbone
x
</details>

Figure 2: Overview of IMTL.

Algorithm 1 Training by Impartial Multi-task Learning

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: input sample $\boldsymbol{x}$, task-specific labels $\{\boldsymbol{y}_t\}$ and learning rate $\eta$
Output: task-shared/-specific parameters $\boldsymbol{\theta}/\{\boldsymbol{\theta}_t\}$, scale parameters $\{s_t\}$
compute task-shared feature $\boldsymbol{z} = \boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\theta})$
for $t = 1$ to $T$ do
    compute task prediction by head network $\boldsymbol{f}_t(\boldsymbol{x}) = \boldsymbol{f}_t^{\text{net}}(\boldsymbol{z}; \boldsymbol{\theta}_t)$
    compute raw loss by loss function $L_t^{\text{raw}} = L_t^{\text{func}}(\boldsymbol{f}_t(\boldsymbol{x}), \boldsymbol{y}_t)$
    compute scaled loss $L_t = ba^{s_t} L_t^{\text{raw}} - s_t$ (default $a = e, b = 1$) $\triangleright$ loss balance
    compute gradient of shared feature $\boldsymbol{z} : \boldsymbol{g}_t = \nabla_{\boldsymbol{z}} L_t$
    compute unit-norm gradient $\boldsymbol{u}_t = \frac{\boldsymbol{g}_t}{\|\boldsymbol{g}_t\|}$
end for
compute gradient differences $\boldsymbol{D}^\top = [\boldsymbol{g}_1^\top - \boldsymbol{g}_2^\top, \cdots, \boldsymbol{g}_1^\top - \boldsymbol{g}_T^\top]$
compute unit-norm gradient differences $\boldsymbol{U}^\top = [\boldsymbol{u}_1^\top - \boldsymbol{u}_2^\top, \cdots, \boldsymbol{u}_1^\top - \boldsymbol{u}_T^\top]$
compute scaling factors for tasks 2 to $T$: $\boldsymbol{\alpha}_{2:T} = \boldsymbol{g}_1 \boldsymbol{U}^\top (\boldsymbol{DU}^\top)^{-1}$ $\triangleright$ gradient balance
compute scaling factors for all tasks: $\boldsymbol{\alpha} = [1 - 1\boldsymbol{\alpha}_{2:T}, \quad \boldsymbol{\alpha}_{2:T}]$
update task-shared parameters $\boldsymbol{\theta} = \boldsymbol{\theta} - \eta\nabla_{\boldsymbol{\theta}} (\sum_t \alpha_t L_t)$
for $t = 1$ to $T$ do
    update task-specific parameters $\boldsymbol{\theta}_t = \boldsymbol{\theta}_t - \eta\nabla_{\boldsymbol{\theta}_t} L_t$
    update loss scale parameter $s_t = s_t - \eta\frac{\partial L_t}{\partial s_t}$
end for
</div>

we treat all tasks equally so that they progress in the same speed and none is left behind. Formally, let $\{ \pmb { u } _ { t } = \pmb { g } _ { t } / \| \pmb { g } _ { t } \| \}$ denote the unit-norm vector of $\left\{ \pmb { g } _ { t } \right\}$ which are row vectors, then we have:

$$
\boldsymbol {g} \boldsymbol {u} _ {1} ^ {\top} = \boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} \Leftrightarrow \boldsymbol {g} \left(\boldsymbol {u} _ {1} - \boldsymbol {u} _ {t}\right) ^ {\top} = 0, \forall 2 \leqslant t \leqslant T. \tag {1}
$$

The above problem is under-determined, but we can obtain the closed-form results of $\{ \alpha _ { t } \}$ by constraining $\textstyle \sum _ { t } \alpha _ { t } \ = \ 1$ . Assume $\begin{array} { l l l l } { { \pmb { \alpha } } } & { { = } } & { { [ \alpha _ { 2 } , \cdots , \alpha _ { T } ] , \pmb { U } ^ { \top } } } & { { = } } & { { \left[ \pmb { u } _ { 1 } ^ { \top } - \pmb { u } _ { 2 } ^ { \top } , \cdots , \pmb { u } _ { 1 } ^ { \top } - \pmb { u } _ { T } ^ { \top } \right] } } \end{array}$ $\pmb { { \cal D } } ^ { \top } = \left[ \pmb { { \pmb { g } } } _ { 1 } ^ { \top } - \pmb { \overline { { { \pmb { g } } } } } _ { 2 } ^ { \top } , \cdots , \pmb { { \pmb { g } } } _ { 1 } ^ { \top } - \pmb { { \pmb { g } } } _ { T } ^ { \top } \right]$ and $\mathbf { 1 } = [ 1 , \cdots , 1 ]$ , from Eq. (1) we can obtain:

$$
\boldsymbol {\alpha} = \boldsymbol {g} _ {1} \boldsymbol {U} ^ {\top} \left(\boldsymbol {D} \boldsymbol {U} ^ {\top}\right) ^ {- 1}. \quad \text {(IMTL - G)} \tag {2}
$$

The detailed derivation is in Appendix B.1. After obtaining α, the scaling factor of the first task can be computed by $\alpha _ { 1 } = 1 - 1 \alpha ^ { \top }$ since $\textstyle \sum _ { t } \alpha _ { t } = 1$ . The optimized $\left\{ \alpha _ { t } \right\}$ are used to compute $L =$ $\textstyle \sum _ { t } \alpha _ { t } { \bar { L _ { t } } } .$ , which is ultimately minimized by SGD to update the model. By now, back-propagation needs to be executed $T$ times to obtain the gradient of each task loss with respect to the heavy-weight task-shared parameters θ, which is time-consuming and non-scalable. We replace the parameterlevel gradients $\{ g _ { t } = \nabla _ { \theta } L _ { t } \}$ with feature-level gradients $\{ \nabla _ { z } L _ { t } \}$ to compute $\left\{ \alpha _ { t } \right\}$ . This implies to achieve gradient balance with respect to the last shared feature z as a surrogate of task-shared parameters θ, since it is possible for the network to back-propagate this balance all the way through the task-shared backbone starting from z. This relaxation allows us to do back propagation through the backbone only once after obtaining $\left\{ \alpha _ { t } \right\}$ , and thus the training time can be dramatically reduced.

## 3.2 LOSS BALANCE: IMTL-L

For the task-specific parameters $\{ \pmb \theta _ { t } \}$ , we cannot employ IMTL-G described above, because $\nabla _ { \pmb { \theta } _ { t } } L _ { \tau } = \mathbf { 0 } , \forall \bar { t } \neq \tau ,$ and thus only the gradient of the corresponding task $\nabla _ { \pmb { \theta } _ { t } } L _ { t }$ can be obtained for each $\theta _ { t }$ . Instead we propose to balance the losses among tasks by forcing the scaled losses $\left\{ \alpha _ { t } L _ { t } \right\}$ to be constant for all tasks, without loss of generality, we take the constant as 1. Then the most direct idea is to compute the scaling factors as $\bar { \{ \alpha _ { t } = 1 / \bar { L } _ { t } \} }$ , but they are sensitive to outlier samples and manifest severe oscillations, so we further propose to learn to scale losses via gradient descent and thus stronger stability can be achieved. Suppose the positive losses $\{ L _ { t } > 0 \}$ are to be balanced, we first introduce a mapping function $h : \mathbb { R } \stackrel { } { \to } \mathbb { R } ^ { + }$ to transform the arbitrarily-ranged learnable scale parameters $\left\{ { { s } _ { t } } \right\}$ to positive scaling factors $\{ h \left( s _ { t } \right) > 0 \}$ , hereafter we abandon the subscript t for brevity. Then we should construct an appropriate scaled loss $g \left( s \right)$ so that both network parameters θ and scale parameter s can be optimized by minimizing $g \left( s \right)$ . On one hand, we balance different tasks by encouraging the scaled losses $h \left( s \right) L \left( \pmb { \theta } \right)$ to be 1 for all tasks, so the optimality $s ^ { \star }$ of s is achieved when $\bar { h ( s ) L } ( \pmb \theta ) = 1$ , or equivalently:

$$
f (s) \equiv h (s) L (\boldsymbol {\theta}) - 1 = 0, \text {if} s = s ^ {\star}. \tag {3}
$$

One may expect to minimize $\left| f \left( s \right) \right| = \left| h \left( s \right) L \left( \pmb { \theta } \right) - 1 \right|$ to find $s ^ { \star }$ , however when $h \left( s \right) L \left( \pmb { \theta } \right) < 1$ the gradient with respect to $\theta , \nabla _ { \theta } \left| f \left( s \right) \right| = - \dot { h } \left( s \right) \nabla _ { \theta } L \left( \theta \right)$ , is in the opposite direction. On the other hand, assume our scaled loss $g \left( s \right)$ is a differentiable convex function with respect to s, then its minimum is achieved if and only if $s = s ^ { \star }$ , where the derivative of $g \left( s \right)$ is zero:

$$
g ^ {\prime} (s) = 0, \text {if} s = s ^ {\star}. \tag {4}
$$

From Eq. (3) and (4) we find that the values of $f \left( s \right)$ and $g ^ { \prime } \left( s \right)$ are both 0 when $s = s ^ { \star }$ , we can then regard $f \left( s \right)$ as the derivative of $g \left( s \right)$ , which is our target scaled loss and used to optimize both the network parameters θ and loss scale parameter $s ,$ then we have:

$$
g ^ {\prime} (s) = f (s) \Leftrightarrow g (s) = \int f (s) \mathrm{d} s = L (\boldsymbol {\theta}) \int h (s) \mathrm{d} s - s. \tag {5}
$$

From Eq. (3) and (5), we notice that both $h \left( s \right)$ and $\textstyle \int h ( s )$ ds denote loss scales, so we have $\begin{array} { r } { \int h \left( s \right) \mathrm { d } s = C h \left( s \right) } \end{array}$ , where $C > 0$ is a constant. According to ordinary differential equation, $\int h \left( s \right)$ ds must be the exponential function: $\textstyle \int h ( s ) \mathrm { d } s = b a ^ { s }$ with $a > 1 , b > 0$ (see Appendix B.2). We then have $g ^ { \prime \prime } \bar { ( s ) } = k a ^ { s } , k > 0 ,$ , which is always positive and verifies our assumption about the convexity of $g \left( s \right)$ . Also note that the gradient of $g \left( s \right)$ with respect to $\theta , \nabla _ { \theta } g \left( s \right) =$ $\begin{array} { r } { \int h \left( s \right) \mathrm { d } s \nabla _ { \pmb { \theta } } L \left( \pmb { \theta } \right) = \dot { b a } ^ { s } \nabla _ { \pmb { \theta } } \dot { L } \left( \pmb { \theta } \right) } \end{array}$ , is in the appropriate direction since $b a ^ { s } > 0$ . As an instantiation, we set $\begin{array} { r } { \int h \left( s \right) \mathrm { d } s = e ^ { s } \left( a = e , b = 1 \right) } \end{array}$ , then

$$
g (s) = e ^ {s} L (\boldsymbol {\theta}) - s, \quad \text {(IMTL - L).} \tag {6}
$$

From $\operatorname { E q }$ . (6) we find that the raw loss is scaled by $e ^ { s }$ , and −s acts as a regularization to avoid the trivial solution $s = - \infty$ while minimizing the scaled loss $g \left( s \right)$ . As for implementation, the task losses $\{ L _ { t } \}$ are scaled by $\{ e ^ { s _ { t } } \}$ , and the scaled losses $\{ e ^ { s _ { t } } L - { s _ { t } } \}$ are used to update both the network parameters $\theta , \{ \theta _ { t } \}$ and the scale parameters $\left\{ { { s } _ { t } } \right\}$

## 3.3 HYBRID BALANCE: IMTL

We have introduced IMTL-G/IMTL-L to achieve gradient/loss balance, and both of them produce scaling factors to be applied on the raw losses. They can be used solely, but we find them complementary and able to be combined to improve the performance. In IMTL-G, even if the raw losses are multiplied by arbitrary (maybe different among tasks) positive factors, the direction of the aggregated gradient g stays unchanged. Because by definition $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { t } } \end{array}$ is the angular bisector of the gradients $\left\{ \pmb { g } _ { t } \right\}$ , and positive scaling will not change the directions of $\left\{ g _ { t } \right\}$ and thus that of g (proof in Theorem 2). So we can also obtain the scale factors $\left\{ \alpha _ { t } \right\}$ in IMTL-G with the losses that have been scaled by $\left\{ { { s } _ { t } } \right\}$ from IMTL-L. IMTL-G and IMTL-L are combined as: 1) the taskspecific parameters $\{ \pmb \theta _ { t } \}$ and scale parameters $\left\{ { { s } _ { t } } \right\}$ are updated by scaled losses $\{ e ^ { s _ { t } } L _ { t } - s _ { t } \} ; 2 )$ the task-shared parameters θ are updated by $\sum _ { t } \dot { \alpha _ { t } } \left( \dot { e } ^ { s _ { t } } L _ { t } \right)$ which is the weighted average of $\left\{ e ^ { s _ { t } } L _ { t } \right\}$ with the weights $\left\{ \alpha _ { t } \right\}$ computed by $\{ \nabla _ { z } ^ { - } ( \overline { { e ^ { s _ { t } } } } L _ { t } ) \}$ using IMTL-G. Note that the regularization terms $\left\{ - s _ { t } \right\}$ in Eq. (6) are constants with respect to θ and z, and thus can be ignored when computing gradients and updating parameters in IMTL-G. In this way, we achieve both gradient balance for task-shared parameters and loss balance for task-specific parameters, leading to our full IMTL as illustrated in Alg. 1.

## 4 DISCUSSION

We draw connections between our method and previous state-of-the-arts <sup>1</sup> in Fig. 3. We will show that previous methods can all be categorized as gradient or loss balance, and thus each of them can be seen as a specification of our method. However, all of them have some intrinsic biases or short-comings leading to inferior performances, which we try to overcome.

![](images/d485c9b7be85093018e945dad2fe5dc2f693cf23bc0901f45e9f536bfb9c6bc4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  subgraph LossBalance["Loss Balance"]
  A["uncertainty\n(α_cls L_cls ≈ 1/2, α_reg L_reg ≈ 1)"] -->|"unified form"| B["IMTL-L\n(α_t L_t ≈ const)"]
  C["GLS\n(α_t L_t = L/T)"] -->|"scale invariant"| B
  end

  B -.->|"complementary"| D["IMTL-G\n(p_t = const)"]

  subgraph GradientBalance["Gradient Balance"]
  D -->|"direction aware"| E["GradNorm\n(p_t ∝ u_t u_s^T)"]
  E -->|"hyper-parameter tuning needed"| F["MGDA\n(p_t ∝ ||g_t||^-1)"]
  E -->|"perpendicular to convex hull"| G["angular bisector of task gradients"]
  G -->|"may degrade to uniform scaling"| H["PCGrad\n(p_t ∝ ||g_t||)"]
  end
```
</details>

Figure 3: Relationship between our IMTL and previous methods. The blue dashed arrow indicates the characteristic of each method. In the loss balance methods, we annotate the scaled loss in the bracket. $L _ { \mathrm { c l s } } , \ L _ { \mathrm { r e g } }$ and $L _ { t }$ are the raw loss of classification, regression and individual task, respectively. $\alpha _ { \mathrm { c l s } } , \alpha _ { \mathrm { r e g } }$ and $\alpha _ { t }$ is the corresponding loss scale. L is the geometric mean loss and $T$ is the task number. In the gradient balance methods, we annotate the projections of the aggregated gradient $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { t } } \end{array}$ onto the raw gradient $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ of the t-th task in the bracket. $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \hat { \pmb { g } _ { t } } \right\|$ is the unit-norm vector, $p _ { t } = g { \mathbf { } } u _ { t } ^ { \top }$ is the projection of $\textbf {  { g } }$ onto $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ and $\begin{array} { r } { \pmb { u } _ { s } = \sum _ { t } \pmb { u } _ { t } } \end{array}$ is the mean direction.

GradNorm (Chen et al., 2018) balances tasks by making the norm of the scaled gradient for each task to be approximately equal. It also introduces the inverse training rate and a hyper-parameter γ to control the strength of approaching the mean gradient norm, such that tasks which learn slower can receive larger gradient magnitudes. However, it does not take into account the relationship of the gradient directions. We show that when the angle between the gradients of each pair of tasks is identical, our IMTL-G leads to the equivalent solution as GradNorm.

Theorem 1. If the angle between any pair of ${ \mathbf { } } { \mathbf { } } { \mathbf { } } u _ { t } , { \mathbf { } } u _ { \tau }$ stays constant: ${ \pmb u } _ { t } { \pmb u } _ { \tau } ^ { \top } = \underline { { C } } _ { 1 }$ , ∀t $\neq \tau$ with $C _ { 1 } < 1$ , then our IMTL-G leads to the same solution as that of GradNorm: $g \mathbf { * } _ { t } ^ { \mathrm { ~ \tiny ~ 1 ~ } } = C _ { 2 } \Leftrightarrow n _ { t } \equiv$ $\lVert \alpha _ { t } \pmb { g } _ { t } \rVert = \alpha _ { t } \lVert \pmb { g } _ { t } \rVert = C _ { 3 }$ . In the above ${ \pmb u } _ { t } = { \pmb g } _ { t } / \| { \pmb g } _ { t } \| , C _ { 1 } , C _ { 2 }$ and $C _ { 3 }$ are constants.

Proof in Appendix C.1. In GradNorm, if without the above constant-angle condition ${ \pmb u } _ { t } { \pmb u } _ { \tau } ^ { \top } = C _ { 1 }$ the projection of the aggregated gradient g onto task-specific gradient, $\begin{array} { r } { \mathbf { \bar { g } } \mathbf { * { \frac { 1 } { t } } } = ( \sum _ { \tau } C _ { 3 } \mathbf { * } _ { \tau } ) \mathbf { * } _ { t } ^ { \top } = } \end{array}$ $\begin{array} { r } { C _ { 3 } \left( \sum _ { \tau } \pmb { u } _ { \tau } \right) \pmb { u } _ { t } ^ { \top } } \end{array}$ , is proportional to $\begin{array} { r } { ( \sum _ { \tau } { \pmb u } _ { \tau } ) { \pmb u } _ { t } ^ { \top } } \end{array}$ . It tends to optimize the “majority tas $z s ^ { , , }$ whose gradient directions are closer to the mean direction $\textstyle \sum _ { t } { \pmb u } _ { t } .$ , resulting in undesired task bias.

MGDA (Sener & Koltun, 2018) finds the weighted average gradient $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { i } } \end{array}$ with minimum norm in the convex hull composed by $\left\{ \pmb { g } _ { t } \right\}$ , so that $\textstyle \sum _ { t } \alpha _ { t } { \bar { = } } { \bar { 1 } }$ and $\alpha _ { t } \geqslant 0 .$ , ∀t. It adopts an iterative method based on Frank-Wolfe algorithm to solve the multi-objective optimization problem. We note the minimum-norm point has a closed-form representation if without the constraints $\{ \alpha _ { t } \geqslant 0 \}$ . In this case, we try to minimize $\begin{array} { r } { \pmb { g } \pmb { g } ^ { \top } = \left( \sum _ { t } \alpha _ { t } \pmb { g } _ { t } \right) \left( \sum _ { \tau } \alpha _ { \tau } \pmb { g } _ { \tau } \right) ^ { \top } } \end{array}$ such that $\textstyle \sum _ { t } \alpha _ { t } = 1$ . It implies $\mathbf { \pmb { g } }$ is perpendicular to the hyper-plane composed by $\left\{ \pmb { g } _ { t } \right\}$ as illustrated in Fig 1 (b), and thus we have:

$$
\boldsymbol {g} \perp (\boldsymbol {g} _ {1} - \boldsymbol {g} _ {t}) \Leftrightarrow \boldsymbol {g} \left(\boldsymbol {g} _ {1} - \boldsymbol {g} _ {t}\right) ^ {\top} = 0, \forall 2 \leqslant t \leqslant T, \tag {7}
$$

and can obtain $\pmb { \alpha } = \pmb { g } _ { 1 } \pmb { D } ^ { \top } \left( \pmb { D } \pmb { D } ^ { \top } \right) ^ { - 1 }$ (see Appendix C.2). From Eq. (7), we note that the aggregated gradient satisfies: $\mathbf { \boldsymbol { g } } \mathbf { \boldsymbol { g } } _ { t } ^ { \intercal } = C$ . Then the projection of $\textbf {  { g } }$ onto $\mathbf { } g _ { t } , g \mathbf { } u _ { t } ^ { \top } = C / \left\| g _ { t } \right\|$ , is inversely proportional to the norm of $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ . So it focuses on tasks with smaller gradient magnitudes, which breaks the task balance. Even with $\{ \alpha _ { t } \geqslant 0 \}$ , the problem still exists (see Appendix C.2) in the original MGDA method. Through experiments, we note that finding the minimum-norm point without the constraints $\{ \alpha _ { t } \geqslant 0 \}$ leads to similar performance as MGDA with the constraints $\{ \alpha _ { t } \geqslant 0 \}$ In our IMTL-G, although we do not constrain $\{ \alpha _ { t } \geqslant 0 \}$ , its loss weighting scales are always positive during the training procedure as shown in Fig. 4.

Uncertainty weighting (Kendall et al., 2018) regards the task uncertainty as loss weight. For regression, it can derive $L _ { 1 }$ loss from Laplace distribution: $- \log p \left( y \mid f \left( x \right) \right) = \left| y - f \left( x \right) \right| / b + \log b ,$ where x is the data sample, $y$ is the ground-truth label, $f$ denotes the prediction model and b is the diversity of Laplace distribution. $L _ { 2 }$ loss can be found in Appendix C.4. For classification, it takes the cross-entropy loss as a scaled categorical distribution and introduces the following approximation:

$$
- \log p (y \mid f (\boldsymbol {x})) = - \log \left[ \operatorname{softmax} _ {y} \left(\frac {f (\boldsymbol {x})}{\sigma^ {2}}\right) \right] \approx - \frac {1}{\sigma^ {2}} \log \left[ \operatorname{softmax} _ {y} (f (\boldsymbol {x})) \right] + \log \sigma , \tag {8}
$$

in which softmax (·) stands for taking the y-th entry after the softmax (·) operator. MTL corresponds to maximizing the joint likelihood of multiple targets, then the derivations yield the scaling factor b/σ for the regression/classification loss. (Kendall et al., 2018) learn b and σ as model parameters which are updated by stochastic gradient descent. However, it is applicable only if we can find appropriate correspondence between the loss and the distribution. It is difficult to be used for losses such as cosine similarity, and it is impossible to traverse all kinds of losses to obtain a unified form for them. Moreover, it sacrifices classification tasks. From Eq. (8) we can find that the scaled cross-entropy loss is approximated as $L = e ^ { 2 s } L _ { \mathrm { c l s } } - s$ if we set $s = - \log \sigma$ . By taking the derivative we have $\partial \bar { L } / \partial s = \bar { 2 e } ^ { \bar { 2 } s } L _ { \mathrm { c l s } } - 1$ . Then s is optimized to make the scaled loss $e ^ { 2 s } L _ { \mathrm { c l s } }$ to be close to $1 / 2$ . However, the scaled $L _ { 1 }$ loss is approximated as $L = e ^ { s } L _ { \mathrm { r e g } } - s$ if we set $s = - \log b ,$ , and taking the derivative we have $\partial L / \partial s = \dot { e } ^ { s } L _ { \mathrm { r e g } } - 1$ . So s is optimized to make the scaled $L _ { 1 }$ loss to achieve 1, which is twice of the classification loss, and thus the classification task is overlooked.

We would like to remark the differences between our IMTL-L and uncertainty weighting (Kendall et al., 2018). Firstly, our derivation is motivated by the fairness among tasks, which intrinsically differs from uncertainty weighting which is based on task uncertainty considering each task independently. Secondly, IMTL-L learns to balance among tasks without any biases, while uncertainty weighting may sacrifice classification tasks to favor regression tasks as derived above. Thirdly, IMTL-L does not depend on any distribution assumptions and thus can be generally applied to various losses including cosine similarity, which uncertainty weighting may have difficulty with. As far as we know, there is no appropriate correspondence between cosine similarity and specific distributions. Lastly, uncertainty weighting needs to deal with different losses case by case, it also introduces approximations in order to derive scaling factors for certain losses (such as cross-entropy loss) which may not be optimal, but our IMTL-L has a unified form for all kinds of losses.

GLS (Chennupati et al., 2019) calculates the target loss as the geometric mean: $L = ( \prod _ { t } L _ { t } ) ^ { \frac { 1 } { T } }$ , then the gradient of $L$ with respect to the model parameters θ can be obtained as Appendix C.5, which can be regarded as to weigh the loss with its reciprocal value. However, as the gradient depends on the value of $L ,$ so it is not scale-invariant to the loss scale changes. Moreover, we find it to be unstable when the number of tasks is large because of the geometric mean computation.

## 5 EXPERIMENTS

In previous methods, various experimental settings have been adopted but there are no extensive comparisons. As one contribution of our work, we re-implement representative methods and present fair comparisons among them under the unified code-base, where more practical settings are adopted and stronger performances are achieved compared with existing code-bases. The implementations exactly follow the original papers and open-sourced code to ensure the correctness. We run experiments on the Cityscapes (Cordts et al., 2016), NYUv2 (Silberman et al., 2012) and CelebA (Liu et al., 2015) dataset to extensively analyze different methods. Details can be found in Appendix D.

Results on Cityscapes. From Tab. 1 we can obtain several informative conclusions. The uniform scaling baseline, which na¨ıvely adds all losses, tends to optimize tasks with larger losses and gradient magnitudes, resulting in severe task bias. Uncertainty weighting (Kendall et al., 2018) sacrifices classification tasks to aid regression ones, leading to significantly worse results on semantic segmentation compared with our IMTL-L. GradNorm (Chen et al., 2018) is very sensitive to the choice of the hyper-parameter $\gamma$ controlling the strength of equal gradient magnitudes, where the default $\gamma = 1 . 5$ works well on NYUv2 but performs badly on Cityscapes. We find its best option is $\gamma = 0$ which makes the scaled gradient norm to be exactly equal. MGDA (Sener & Koltun, 2018) focuses on tasks with smaller gradient magnitudes. So the performance of semantic segmentation is good but the other two tasks have difficulty in converging. In addition, we find our proposed closed-form variant without the hard constraints $\{ \alpha _ { t } \geqslant 0 \}$ achieves similar results as the original iterative method. Through the experiments we notice the closed-form solution almost always yields $\{ \alpha _ { t } \geqslant 0 \}$ . As for PCGrad (Yu et al., 2020), it yields slightly better performance than uniform scaling because its conflict projection will have no effect when the angles between the gradients are equal or less than $\pi / 2$ . In contrast, our IMTL method, in terms of both gradient balance and loss balance, yields competitive performance and achieves the best balance among tasks. Moreover, we verify that the two balances are complementary and can be combined to further improve the performance, with the visualizations in Appendix E. Surprisingly, we find our IMTL can beat the single-task baseline where each task is trained with a separate model. Training multiple tasks simultaneously can learn a better representation from multiple levels of semantics, which can in turn improve individual tasks.

Table 1: Comparison between IMTL and previous methods on Cityscapes, semantic segmentation, instance segmentation and disparity/depth estimation are considered. The first group of columns shows the regular results of different methods. The second group shows the results by manually multiply the semantic segmentation loss with 10 before applying these methods. The subscript numbers show the absolute change after scaling the loss to demonstrate the robustness of various methods. The arrows indicate the values are the higher the better (↑) or the lower the better (↓). The best and runner up results for each task are bold and underlined, respectively.

<table><tr><td>method</td><td>sem. mIoU↑</td><td>ins.  $L_1 \downarrow$ </td><td>disp.  $L_1 \downarrow$ </td><td>sem. mIoU↑ $_{|Δ|↓}$ </td><td>ins.  $L_1 \downarrow_{|\Delta|↓}$ </td><td>disp.  $L_1 \downarrow_{|\Delta|↓}$ </td><td>time s/iter↓</td></tr><tr><td>baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>single-task</td><td>76.67</td><td>21.61</td><td>4.182</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>uniform scaling</td><td>58.99</td><td>18.13</td><td>3.512</td><td>-</td><td>-</td><td>-</td><td>1.201</td></tr><tr><td>loss balance</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>uncertainty (Kendall et al., 2018)</td><td>74.91</td><td>16.43</td><td>2.895</td><td> $74.00_{0.91}$ </td><td> $16.77_{0.34}$ </td><td> $2.930_{0.035}$ </td><td>1.204</td></tr><tr><td>GLS (Chennupati et al., 2019)</td><td>75.65</td><td>17.18</td><td>2.953</td><td> $66.22_{9.43}$ </td><td> $21.09_{3.91}$ </td><td> $3.358_{0.405}$ </td><td>1.202</td></tr><tr><td>IMTL-L</td><td>76.89</td><td>16.69</td><td>2.944</td><td> $75.55_{1.34}$ </td><td> $17.49_{0.80}$ </td><td> $2.972_{0.028}$ </td><td>1.202</td></tr><tr><td>gradient balance</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GradNorm ( $\gamma = 0$ )</td><td>76.27</td><td>17.99</td><td>3.195</td><td> $72.96_{3.31}$ </td><td> $19.36_{1.37}$ </td><td> $3.216_{0.021}$ </td><td>1.741</td></tr><tr><td>GradNorm (Chen et al., 2018)</td><td>52.17</td><td>19.88</td><td>4.098</td><td> $54.23_{2.06}$ </td><td> $20.53_{0.65}$ </td><td> $4.108_{0.010}$ </td><td>1.742</td></tr><tr><td>MGDA (w/o  $\{ \alpha_t \geqslant 0 \}$ )</td><td>76.95</td><td>53.19</td><td>6.296</td><td> $76.36_{0.59}$ </td><td> $29.06_{24.13}$ </td><td> $3.377_{2.919}$ </td><td>1.777</td></tr><tr><td>MGDA (Sener &amp; Koltun, 2018)</td><td>76.56</td><td>53.14</td><td>6.644</td><td> $72.35_{4.21}$ </td><td> $29.38_{23.76}$ </td><td> $3.336_{3.308}$ </td><td>1.732</td></tr><tr><td>PCGrad (Yu et al., 2020)</td><td>60.50</td><td>17.99</td><td>3.450</td><td> $66.33_{5.83}$ </td><td> $17.99_{0.00}$ </td><td> $3.386_{0.064}$ </td><td>2.087</td></tr><tr><td>IMTL-G (exact)</td><td>76.13</td><td>17.46</td><td>2.979</td><td>-</td><td>-</td><td>-</td><td>2.769</td></tr><tr><td>IMTL-G</td><td>76.52</td><td>16.61</td><td>2.997</td><td> $76.06_{0.46}$ </td><td> $17.52_{0.91}$ </td><td> $3.020_{0.023}$ </td><td>1.776</td></tr><tr><td>hybrid balance</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IMTL</td><td>77.00</td><td>15.96</td><td>2.905</td><td> $76.56_{0.44}$ </td><td> $15.85_{0.11}$ </td><td> $2.938_{0.033}$ </td><td>1.795</td></tr></table>

In addition, we present the real-world training time of each iteration for different methods in Tab. 1. As shown, loss balance methods are the most efficient, and our gradient balance method IMTL-G adds acceptable computational overhead, similar to that of GradNorm (Chen et al., 2018) and MGDA (Sener & Koltun, 2018). It benefits from computing gradients with respect to the shared feature maps instead of the shared model parameters (the row of “IMTL-G (exact)”), which brings similar performances but adds significant complexity due to multiple (T) backward passes through the shared parameters. Our IMTL-G only needs to do backward computation on the shared parameters once after obtaining the loss weights via Eq. (2), in which the computation overhead mainly comes from the matrix multiplication rather than the matrix inverse, since the inversed matrix $\bar { D U ^ { \top } } \in \mathbb { R } ^ { ( T - 1 ) \times ( T - 1 ) }$ is small compared with dimension of the shared feature z.

As we outperform MGDA (Sener & Koltun, 2018) and PCGrad (Yu et al., 2020) significantly in terms of the objective metrics shown in Tab. 1, we further compare the qualitative results of our hybrid balance IMTL with the loss balance method uncertainty weighting (Kendall et al., 2018) and the gradient balance method GradNorm (Chen et al., 2018) considering their strong performances (see Fig. 6). For depth estimation we only show predictions at the pixels where ground truth (GT) labels exist to compare with GT, which is different from Fig. 7 where depth predictions are shown for all pixels. Consistent with results in Tab. 1, our IMTL shows visually noticeable improvements especially for the semantic and instance segmentation tasks. It is worth noting that we conduct experiments under strong baselines and practical settings which are seldom explored before, in this case changing the backbone in PSPNet (Zhao et al., 2017) from ResNet-50 to ResNet-101 can only improve mIoU of the semantic segmentation task around 0.5% according to the public code base<sup>2</sup>.

Scale invariance. We are also interested in the scale invariance, which means how the results change with the loss scale. For example, in semantic segmentation, the loss scale is different if we replace the reduction method “mean” (averaged over all locations) with “sum” (summed over all locations) in the cross-entropy loss computation, or the number of the interested classes increases. The scale invariance is beneficial for model robustness. So to simulate this effect, we manually multiply the semantic segmentation loss by 10 and apply the same methods to see how the performances are affected. In the last three columns of Tab. 1 we report the absolute changes resulting from the multiplier. Our IMTL achieves the smallest performance fluctuations and thus the best invariance, while other methods are more or less affected by the loss scale change.

Table 2: Experimental results on the NYUv2 and CelebA datasets, semantic segmentation, surface normal estimation, depth estimation and multi-class classification are considered. Arrows indicate the values are the higher the better (↑) or the lower the better (↓). The best and runner up results in each column are bold and underlined, respectively.

<table><tr><td>method</td><td>sem. mIoU↑</td><td>NYUv2 norm. cos↑</td><td>depth  $L_1 \downarrow$ </td><td>CelebA class.acc. ↑</td></tr><tr><td>baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>single-task</td><td>56.82</td><td>0.8827</td><td>0.5097</td><td>-</td></tr><tr><td>uniform scaling</td><td>57.40</td><td>0.8684</td><td>0.4248</td><td>90.01</td></tr><tr><td>loss balance</td><td></td><td></td><td></td><td></td></tr><tr><td>uncertainty (Kendall et al., 2018)</td><td>57.20</td><td>-</td><td>0.4400</td><td>90.34</td></tr><tr><td>GLS (Chennupati et al., 2019)</td><td>57.84</td><td>0.8762</td><td>0.4243</td><td>-</td></tr><tr><td>IMTL-L</td><td>58.36</td><td>0.8864</td><td>0.4173</td><td>90.54</td></tr><tr><td>gradient balance</td><td></td><td></td><td></td><td></td></tr><tr><td>GradNorm ( $\gamma = 0$ )</td><td>55.96</td><td>0.8818</td><td>0.4317</td><td>90.91</td></tr><tr><td>GradNorm (Chen et al., 2018)</td><td>56.92</td><td>0.8787</td><td>0.4285</td><td>89.92</td></tr><tr><td>MGDA (w/o  $\{ \alpha_t \geqslant 0 \}$ )</td><td>49.43</td><td>0.8877</td><td>0.4839</td><td>89.68</td></tr><tr><td>MGDA (Sener &amp; Koltun, 2018)</td><td>49.44</td><td>0.8875</td><td>0.4759</td><td>90.04</td></tr><tr><td>PCGrad (Yu et al., 2020)</td><td>57.48</td><td>0.8696</td><td>0.4253</td><td>89.99</td></tr><tr><td>IMTL-G</td><td>57.00</td><td>0.8785</td><td>0.4226</td><td>91.03</td></tr><tr><td>hybrid balance</td><td></td><td></td><td></td><td></td></tr><tr><td>IMTL</td><td>58.85</td><td>0.8888</td><td>0.4215</td><td>91.12</td></tr></table>

Results on NYUv2. In Tab. 2 we find similar patterns as on Cityscapes, but NYUv2 is a rather small dataset, so uniform scaling can also obtain reasonable results. Note that uncertainty weighting (Kendall et al., 2018) cannot be directly used to estimate the normal surface when the cosine similarity is used as the loss, since no appropriate distribution can be found to correspond to cosine similarity. In this case, surface normal estimation owns the smallest gradient magnitude, so MGDA (Sener & Koltun, 2018) learns it best but it performs not so well for the rest two tasks. Again, our IMTL performs best taking advantage of the complementary gradient and loss balances.

Results on CelebA. To compare different methods in the many-task setting, in Tab. 2 we also conduct the multi-label classification experiments on the CelebA (Liu et al., 2015) dataset. The mean accuracy of 40 tasks is used as the final metric. Our IMTL outperforms its competitors in the scenario where the task number is large, showing its superiority. Note that in this setting, GLS (Chennupati et al., 2019) has difficulty in converging and no reasonable results can be obtained.

## 6 CONCLUSION

We propose an impartial multi-task learning method integrating gradient balance and loss balance, which are applied on task-shared and task-specific parameters, respectively. Through our in-depth analysis, we have theoretically compared our method with previous state-of-the-arts. We have also showed that those state-of-the-arts can all be categorized as gradient or loss balance, but lead to specific bias among tasks. Through extensive experiments we verify our analysis and demonstrate the effectiveness of our method. Besides, for fair comparisons, we contribute a unified code-base, which adopts more practical settings and delivers stronger performances compared with existing code-bases, and it will be publicly available for future research.

## ACKNOWLEDGEMENTS

This work was supported by the Natural Science Foundation of Guangdong Province (No. 2020A1515010711), the Special Foundation for the Development of Strategic Emerging Industries of Shenzhen (No. JCYJ20200109143010272), and the Innovation and Technology Commission of the Hong Kong Special Administrative Region, China (Enterprise Support Scheme under the Innovation and Technology Fund B/E030/18).

## REFERENCES

Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.  
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4): 834–848, 2017.  
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pp. 794–803, 2018.  
Sumanth Chennupati, Ganesh Sistu, Senthil Yogamani, and Samir A Rawashdeh. Multinet++: Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3213–3223, 2016.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255. IEEE, 2009.  
Theodoros Evgeniou and Massimiliano Pontil. Regularized multi–task learning. In Proceedings of the tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 109–117, 2004.  
Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 3205–3214, 2019.  
Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11543–11552, 2020.  
Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 270–287, 2018.  
Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International Conference on Machine Learning, 2020.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.  
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7132–7141, 2018.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning - Volume 37, pp. 448–456, 2015.  
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7482–7491, 2018.  
Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1871– 1880, 2019.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings ofthe IEEE International Conference on Computer Vision, pp. 3730–3738, 2015.  
Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10437–10446, 2020.  
Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In Proceedings ofthe European Conference on Computer Vision (ECCV), pp. 67–82, 2018.  
Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1851–1860, 2019.  
Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3994–4003, 2016.  
Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun. Megdet: A large mini-batch object detector. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 6181–6189, 2018.  
Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In Advances in Neural Information Processing Systems, pp. 506–516, 2017.  
Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.  
Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 4822–4829, 2019.  
Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, pp. 527–538, 2018.  
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European Conference on Computer Vision, pp. 746–760. Springer, 2012.  
Trevor Standley, Amir R Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International Conference on Machine Learning, 2020.  
Gjorgji Strezoski, Nanne van Noord, and Marcel Worring. Many task learning with task routing. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1375–1384, 2019.  
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. arXiv preprint arXiv:2001.06782, 2020.  
Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3712–3722, 2018.  
Amir R Zamir, Alexander Sax, Nikhil Cheerla, Rohan Suri, Zhangjie Cao, Jitendra Malik, and Leonidas J Guibas. Robust learning through cross-task consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11197–11206, 2020.  
Yu Zhang and Qiang Yang. A survey on multi-task learning. arXiv preprint arXiv:1707.08114, 2017.  
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2881–2890, 2017.

Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In Proceedings ofthe International Conference on Learning Representations, 2017.

## A RELATED WORK OF NETWORK STRUCTURE

Cross-stitch Networks (Misra et al., 2016) learn coefficients to linearly combine activations from multiple tasks to construct better task-specific representations. To break the limitation of channelwise cross-task feature fusion only, NDDR-CNN (Gao et al., 2019) proposes the layer-wise crosschannel feature aggregation as 1 × 1 convolutions on the concatenated feature maps from multiple tasks. More generally, MTL-NAS (Gao et al., 2020) introduces cross-layer connections among tasks to fully exploit the feature sharing from both low and high layers, extending the idea in Sluice Networks (Ruder et al., 2019) by leveraging neural architecture search (Zoph & Le, 2017). The parameters of these methods increase linearly with the number of tasks. To improve the model compactness, Residual Adapters (Rebuffi et al., 2017) introduce a small amount of task-specific parameters for each layer and convolve them with the task-agnostic representations to form the taskrelated ones. MTAN (Liu et al., 2019) generates data-dependent attention tensors by task-specific parameters to attend to the task-shared features. Single-tasking (Maninis et al., 2019) instead applies squeeze-and-excitation (Hu et al., 2018) module to generate attentive vectors for each task. In Task Routing (Strezoski et al., 2019), the attentive vectors are randomly sampled before training and are fixed for each image. Piggyback (Mallya et al., 2018) opts to mask parameter weights in place of activation maps, dealing with task-sharing from another point-of-view. The above methods can share parameters among tasks to a large extent, however, they are not memory-efficient because each task still needs to compute all of its own intermediate feature maps, which also leads to inferior inference speed compared with loss weighting methods.

## B DETAILED DERIVATION

## B.1 GRADIENT BALANCE: IMTL-G

Here we give the detailed derivation of the closed-form solution of our IMTL-G, we also demonstrate the scale-invariance property of our IMTL-G, which is invariant to the scale changes of losses.

Solution. As we want to achieve:

$$
\boldsymbol {g} \boldsymbol {u} _ {1} ^ {\top} = \boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} \Leftrightarrow \boldsymbol {g} \left(\boldsymbol {u} _ {1} - \boldsymbol {u} _ {t}\right) ^ {\top} = 0, \forall 2 \leqslant t \leqslant T, \tag {9}
$$

where $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \pmb { g } _ { t } \right\|$ , recall that we have $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { t } } \end{array}$ and $\textstyle \sum _ { t } \alpha _ { t } = 1$ , if we set ${ \pmb { \alpha } } = [ \alpha _ { 2 } , \cdots , \alpha _ { T } ]$ and $G ^ { \top } = \left[ \pmb { g } _ { 2 } ^ { \top } , \cdots , \pmb { g } _ { T } ^ { \top } \right]$ , then $\alpha _ { 1 } = 1 - 1 \alpha ^ { \top }$ and Eq. (9) can be expanded as:

$$
\left(\sum_ {t} \alpha_ {t} \boldsymbol {g} _ {t}\right) \left[ \boldsymbol {u} _ {1} ^ {\top} - \boldsymbol {u} _ {2} ^ {\top}, \dots , \boldsymbol {u} _ {1} ^ {\top} - \boldsymbol {u} _ {T} ^ {\top} \right] = \mathbf {0} \Leftrightarrow \left[ \begin{array}{l l} 1 - \mathbf {1} \boldsymbol {\alpha} ^ {\top}, & \boldsymbol {\alpha} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {g} _ {1} \\ \boldsymbol {G} \end{array} \right] \boldsymbol {U} ^ {\top} = \mathbf {0}, \tag {10}
$$

where $\pmb { U } ^ { \top } = \left\lceil \pmb { u } _ { 1 } ^ { \top } - \pmb { u } _ { 2 } ^ { \top } , \cdots , \pmb { u } _ { 1 } ^ { \top } - \pmb { u } _ { T } ^ { \top } \right\rceil$ , 1 and 0 indicate the all-one and all-zero row vector, respectively. $\mathrm { E q . } \left( 1 0 \right)$ can be solved by:

$$
\left[ \left(1 - \mathbf {1} \boldsymbol {\alpha} ^ {\top}\right) \boldsymbol {g} _ {1} + \boldsymbol {\alpha} \boldsymbol {G} \right] \boldsymbol {U} ^ {\top} = \boldsymbol {0} \Leftrightarrow \boldsymbol {\alpha} \left(\mathbf {1} ^ {\top} \boldsymbol {g} _ {1} - \boldsymbol {G}\right) \boldsymbol {U} ^ {\top} = \boldsymbol {g} _ {1} \boldsymbol {U} ^ {\top}. \tag {11}
$$

Assume $\pmb { { \cal D } } ^ { \top } = \pmb { \mathfrak { g } } _ { 1 } ^ { \top } \pmb { 1 } - \pmb { { \cal G } } ^ { \top } = \left[ \pmb { \mathfrak { g } } _ { 1 } ^ { \top } - \pmb { \mathfrak { g } } _ { 2 } ^ { \top } , \cdots , \pmb { \mathfrak { g } } _ { 1 } ^ { \top } - \pmb { \mathfrak { g } } _ { T } ^ { \top } \right]$ , then we reach:

$$
\boldsymbol {\alpha} \boldsymbol {D} \boldsymbol {U} ^ {\top} = \boldsymbol {g} _ {1} \boldsymbol {U} ^ {\top} \Leftrightarrow \boldsymbol {\alpha} = \boldsymbol {g} _ {1} \boldsymbol {U} ^ {\top} \left(\boldsymbol {D} \boldsymbol {U} ^ {\top}\right) ^ {- 1}. \tag {12}
$$

Property. We can also prove the aggregated gradient $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { i } } \end{array}$ <sub>t</sub> with $\{ \alpha _ { t } \}$ given in Eq. (12) is invariant to the scale changes of losses $\{ L _ { t } \}$ (or gradients $\{ \overline { { g _ { t } } } = \nabla _ { \pmb { \theta } } L _ { t } \} )$ , as the following theorem.

Theorem 2. Given $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { t } , \sum _ { t } \alpha _ { t } = 1 } \end{array}$ satisfying $\begin{array} { r } { g u _ { t } ^ { \top } = C , } \end{array}$ , when $\{ L _ { t } \}$ are scaled by $\{ k _ { t } > 0 \}$ (equivalently, $\left\{ \pmb { g } _ { t } \right\}$ are scaled by $\begin{array} { r } { \{ k _ { t } \} ) , i f g ^ { \prime } = \sum _ { t } \alpha _ { t } ^ { \prime } ( k _ { t } g _ { t } ) , \sum _ { t } \alpha _ { t } ^ { \prime } = 1 } \end{array}$ satisfies $\pmb { g } ^ { \prime } \pmb { u } _ { t } ^ { \top } =$ C<sup>0</sup>, then $\pmb { g } ^ { \prime } = \lambda \pmb { g }$ . In the above we have $\begin{array} { r } { \pmb { u } _ { t } = \frac { \pmb { g } _ { t } } { \lVert \pmb { g } _ { t } \rVert } = \frac { k _ { t } \pmb { g } _ { t } } { \lVert \pmb { k } _ { t } \pmb { g } _ { t } \rVert } , \lambda , } \end{array}$ C and $C ^ { \prime }$ are constants.

![](images/fd05893f75c71091383f21cb206c3699949b014064c5baaf5b68a9293f6e4819.jpg)  
Figure 4: Loss scales of IMTL-G for different tasks when training on the Cityscapes dataset.

Proof. As we have:

$$
\boldsymbol {g} = \sum_ {t} \alpha_ {t} \boldsymbol {g} _ {t} = \sum_ {t} \frac {\alpha_ {t}}{k _ {t}} k _ {t} \boldsymbol {g} _ {t} \quad \text {and} \quad \boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} = C, \tag {13}
$$

by constructing:

$$
\alpha_ {t} ^ {\prime} = \frac {\alpha_ {t}}{k _ {t}} / \sum_ {\tau} \frac {\alpha_ {\tau}}{k _ {\tau}} \quad \text {and} \quad \boldsymbol {g} ^ {\prime} = \sum_ {t} \alpha_ {t} ^ {\prime} \left(k _ {t} \boldsymbol {g} _ {t}\right) = \boldsymbol {g} / \sum_ {\tau} \frac {\alpha_ {\tau}}{k _ {\tau}} = \lambda \boldsymbol {g}, \tag {14}
$$

we have:

$$
\sum_ {t} \alpha_ {t} ^ {\prime} = 1 \quad \text {and} \quad \boldsymbol {g} ^ {\prime} \boldsymbol {u} _ {t} ^ {\top} = C / \sum_ {\tau} \frac {\alpha_ {\tau}}{k _ {\tau}} = C ^ {\prime}. \tag {15}
$$

From $\operatorname { E q . }$ (12) we know that $\left\{ \alpha _ { t } \right\}$ has a unique solution, and thus $\pmb { g } ^ { \prime }$ satisfying IMTL-G is unique, so it must be the one given by Eq. (14), then we can prove that $g ^ { \prime }$ and g are linearly correlated.

## B.2 LOSS BALANCE: IMTL-L

With the ordinary differential equation, we can derive that the form of the scale function $\textstyle \int h ( s )$ ds in our IMTL-L must be exponential function. As we have:

$$
\int h (s) \mathrm{d} s = C h (s), C > 0. \tag {16}
$$

If we set $y = \int h \left( s \right)$ ds, then:

$$
y = C \frac {\mathrm{d} y}{\mathrm{d} s} \Rightarrow \frac {\mathrm{d} y}{y} = \frac {1}{C} \mathrm{d} s, \tag {17}
$$

By taking the antiderivative:

$$
\int \frac {\mathrm{d} y}{y} = \frac {1}{C} \int \mathrm{d} s \Rightarrow \ln y = \frac {1}{C} s + C ^ {\prime}. \tag {18}
$$

Then we have:

$$
\int h (s) \mathrm{d} s = y = e ^ {C ^ {\prime}} \left(e ^ {\frac {1}{C}}\right) ^ {s} = b a ^ {s}, a > 1, b > 0. \tag {19}
$$

## C DETAILED DISCUSSION

## C.1 CONDITIONAL EQUIVALENCE OF IMTL-G AND GRADNORM

First we introduce the following lemma.

Lemma 3. $I f u _ { t } \pmb { u } _ { \tau } ^ { \top } = C _ { 1 } , \forall t \neq \tau ,$ , then the solution $\left\{ \alpha _ { t } \right\}$ of IMTL-G satisfies $\{ \alpha _ { t } > 0 \}$

Proof. As $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \pmb { g } _ { t } \right\|$ , by constructing $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { i } } \end{array}$ <sub>t</sub> where:

$$
\alpha_ {t} = \| \boldsymbol {g} _ {t} \| ^ {- 1} / \sum_ {\tau} \| \boldsymbol {g} _ {\tau} \| ^ {- 1}, \tag {20}
$$

then we have $\textstyle \sum _ { t } \alpha _ { t } = 1$ and:

$$
\boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} = \left(\sum_ {\tau} \boldsymbol {u} _ {\tau} \boldsymbol {u} _ {t}\right) / \sum_ {\tau} \| \boldsymbol {g} _ {\tau} \| ^ {- 1} = [ (T - 1) C _ {1} + 1 ] / \sum_ {\tau} \| \boldsymbol {g} _ {\tau} \| ^ {- 1} = C _ {2}. \tag {21}
$$

From Eq. (12) we know the solution $\left\{ \alpha _ { t } \right\}$ of IMTL-G is unique, so it must be the one given by Eq. (20) where $\{ \alpha _ { t } > 0 \}$ , so the lemma is proved. □

Then we prove Theorem 1 which states that IMTL-G leads to the same solution as GradNorm when the angle between any pair of gradients $\left\{ \pmb { g } _ { t } \right\}$ is identical: $\mathbf { \boldsymbol { u } } _ { t } \mathbf { \boldsymbol { u } } _ { \tau } ^ { \top } = C _ { 1 } , \ \forall t \neq \tau$

Proof. (⇒ Necessity) Given constant projections in IMTL-G, we have:

$$
\boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} = \left(\sum_ {\tau} \alpha_ {\tau} \boldsymbol {g} _ {\tau}\right) \boldsymbol {u} _ {t} ^ {\top} = C _ {2}. \tag {22}
$$

Recall that $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \pmb { g } _ { t } \right\|$ and $\mathbf { \boldsymbol { \mathsf { u } } } _ { t } \mathbf { \boldsymbol { \mathsf { u } } } _ { \tau } ^ { \top } = C _ { 1 } , \ \forall t \neq \tau .$ . From Lemma 3 we know that $\left\{ \alpha _ { t } \right\}$ given by IMTL-G must satisfy $\{ \alpha _ { t } > 0 \}$ }. If we assume $n _ { t } = \| \alpha _ { t } \pmb { g } _ { t } \|$ , then we know $\alpha _ { t } \pmb { g } _ { t } = n _ { t } \pmb { u } _ { t }$ and:

$$
\sum_ {\tau} n _ {\tau} \boldsymbol {u} _ {\tau} \boldsymbol {u} _ {t} ^ {\top} = \sum_ {\tau \neq t} n _ {\tau} C _ {1} + n _ {t} = C _ {2}. \tag {23}
$$

Now we obtain:

$$
\sum_ {\tau \neq t} n _ {\tau} C _ {1} + n _ {t} = \sum_ {\tau} n _ {\tau} C _ {1} + (1 - C _ {1}) n _ {t} = C _ {2}. \tag {24}
$$

$\mathrm { A s } \ C _ { 1 } < 1$ , we can then prove $n _ { t } = C _ { 3 }$ , ∀t. It implies the norm of the scaled gradient is constant, which is requested by GradNorm (Chen et al., 2018). Moreover, we can obtain the relationship among constants from Eq. (24):

$$
C _ {1} T C _ {3} + (1 - C _ {1}) C _ {3} = C _ {2} \Rightarrow C _ {3} = \frac {C _ {2}}{(T - 1) C _ {1} + 1}. \tag {25}
$$

(⇐ Sufficiency) In GradNorm, $\left\{ \alpha _ { t } \right\}$ are always chosen to satisfy $\{ \alpha _ { t } > 0 \}$ , so if we assume $n _ { t } =$ $\| \alpha _ { t } \pmb { g } _ { t } \|$ , then given the constant norm of the scaled gradient in GradNorm, we have:

$$
\alpha_ {t} \boldsymbol {g} _ {t} = n _ {t} \boldsymbol {u} _ {t} = C _ {3} \boldsymbol {u} _ {t}, \tag {26}
$$

where $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \pmb { g } _ { t } \right\|$ . As we have $\begin{array} { r } { \pmb { g } = \sum _ { t } \alpha _ { t } \pmb { g } _ { t } } \end{array}$ and $\mathbf { \boldsymbol { u } } _ { t } \mathbf { \boldsymbol { u } } _ { \tau } ^ { \top } = C _ { 1 } , \ \forall t \neq \tau$ , then we obtain:

$$
\boldsymbol {g} \boldsymbol {u} _ {t} ^ {\top} = \left(\sum_ {\tau} \alpha_ {\tau} \boldsymbol {g} _ {\tau}\right) \boldsymbol {u} _ {t} ^ {\top} = \left(\sum_ {\tau} C _ {3} \boldsymbol {u} _ {\tau}\right) \boldsymbol {u} _ {t} ^ {\top} = C _ {3} \left[ (T - 1) C _ {1} + 1 \right] = C _ {2}. \tag {27}
$$

It means the projections of g onto $\left\{ \pmb { g } _ { t } \right\}$ are constant, which is requested by our IMTL-G.

Corollary 4. In GradNorm, if the solution $\left\{ \alpha _ { t } \right\}$ satisfies $\textstyle \sum _ { t } \alpha _ { t } = 1$ , then its constants are given by $\begin{array} { r } { C _ { 3 } = 1 / \sum _ { t } \| { \pmb g } _ { t } \| ^ { - 1 } } \end{array}$ and $\begin{array} { r } { C _ { 2 } = \left[ \left( T - 1 \right) C _ { 1 } + 1 \right] / \sum _ { t } { \| { \pmb g } _ { t } \| } ^ { - 1 } } \end{array}$ , and its scaling factors are given $\begin{array} { r } { b y \left\{ \alpha _ { t } = \Vert \pmb { g _ { t } } \Vert ^ { - 1 } / \sum _ { \tau } \Vert \pmb { g _ { \tau } } \Vert ^ { - 1 } \right\} } \end{array}$

Proof. By using $\alpha _ { t } ~ = ~ C _ { 3 } / \left\| g _ { t } \right\|$ from Eq. (26), we have $\begin{array} { r } { \sum _ { t } C _ { 3 } / \left| \left| \pmb { g } _ { t } \right| \right| \ = \ 1 } \end{array}$ , then $C _ { 3 } \ =$ $1 / \textstyle \sum _ { t } \| g _ { t } \| ^ { - 1 }$ , and also we have $\begin{array} { r } { \alpha _ { t } = \left. \pmb { g } _ { t } \right. ^ { - 1 } / \sum _ { \tau } \left. \pmb { g } _ { \tau } \right. ^ { - 1 } } \end{array}$ . As the relationship of $C _ { 2 }$ and $C _ { 3 }$ from Eq. (27) is given by $\begin{array} { r l r } & { } & { C _ { 3 } \left[ \left( T - 1 \right) C _ { 1 } + 1 \right] = C _ { 2 } , \infty C _ { 2 } = \left[ \left( T - 1 \right) C _ { 1 } + 1 \right] / \sum _ { t } \left. \pmb { g } _ { t } \right. ^ { - 1 } } \end{array}$ □

## C.2 CLOSED-FORM SOLUTION OF MGDA

In our relaxed MGDA (Sener & Koltun, 2018) without $\{ \alpha _ { t } \geqslant 0 \}$ , finding $\begin{array} { r } { \mathbf { \phi } \mathbf { g } = \sum _ { t } \alpha _ { t } \mathbf { g } _ { t } } \end{array}$ with $\textstyle \sum _ { t } \alpha _ { t } = 1$ such that g has minimum norm is equivalent to find the normal vector of the hyperplane composed by $\{ g _ { t } \}$ . So we let g to be perpendicular to all of $\{ g _ { 1 } - g _ { t } \}$ on the hyper-plane:

$$
\boldsymbol {g} \perp (\boldsymbol {g} _ {1} - \boldsymbol {g} _ {t}) \Leftrightarrow \boldsymbol {g} \left(\boldsymbol {g} _ {1} - \boldsymbol {g} _ {t}\right) ^ {\top} = 0, \forall 2 \leqslant t \leqslant T. \tag {28}
$$

If we set ${ \pmb { \alpha } } = [ \alpha _ { 2 } , \cdots , \alpha _ { T } ]$ and $G ^ { \top } = \left[ \pmb { g } _ { 2 } ^ { \top } , \cdots , \pmb { g } _ { T } ^ { \top } \right]$ , then we have $\alpha _ { 1 } = 1 - 1 \alpha ^ { \top }$ , and Eq. (28) can be expanded as:

$$
\left(\sum_ {t} \alpha_ {t} \boldsymbol {g} _ {t}\right) \left[ \begin{array}{l l l} \boldsymbol {g} _ {1} ^ {\top} - \boldsymbol {g} _ {2} ^ {\top}, & \dots , \boldsymbol {g} _ {1} ^ {\top} - \boldsymbol {g} _ {T} ^ {\top} \end{array} \right] = \mathbf {0} \Leftrightarrow \left[ \begin{array}{l l} 1 - \mathbf {1} \boldsymbol {\alpha} ^ {\top}, & \boldsymbol {\alpha} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {g} _ {1} \\ \boldsymbol {G} \end{array} \right] \boldsymbol {D} ^ {\top} = \mathbf {0}, \tag {29}
$$

where $\pmb { { \cal D } } ^ { \top } = \left[ \pmb { { \pmb g } } _ { 1 } ^ { \top } - \pmb { { \pmb g } } _ { 2 } ^ { \top } , \cdots , \pmb { { \pmb g } } _ { 1 } ^ { \top } - \pmb { { \pmb g } } _ { T } ^ { \top } \right]$ , 1 and 0 indicates the all-one and all-zero row vector. Eq. (29) can be represented as:

$$
\left[ \left(1 - \mathbf {1} \boldsymbol {\alpha} ^ {\top}\right) \boldsymbol {g} _ {1} + \boldsymbol {\alpha} \boldsymbol {G} \right] \boldsymbol {D} ^ {\top} = \boldsymbol {0} \Leftrightarrow \boldsymbol {\alpha} \left(\mathbf {1} ^ {\top} \boldsymbol {g} _ {1} - \boldsymbol {G}\right) \boldsymbol {D} ^ {\top} = \boldsymbol {g} _ {1} \boldsymbol {D} ^ {\top}.
$$

As we also have ${ \pmb { D } } = { \bf 1 } ^ { \top } { \pmb { g } } _ { 1 } - { \pmb { G } }$ , then the closed-form solution of α is given by:

$$
\alpha \boldsymbol {D} \boldsymbol {D} ^ {\top} = \boldsymbol {g} _ {1} \boldsymbol {D} ^ {\top} \Leftrightarrow \boldsymbol {\alpha} = \boldsymbol {g} _ {1} \boldsymbol {D} ^ {\top} \left(\boldsymbol {D} \boldsymbol {D} ^ {\top}\right) ^ {- 1}. \tag {30}
$$

Bias of MGDA. In the main text we state that MGDA focuses on tasks with small gradient magnitudes, where we relaxed MGDA by not constraining $\{ \alpha _ { t } \geqslant 0 \}$ . However, even with these constraints, the problem still exists. For example in the context of two tasks, assume $\left\| g _ { 1 } \right\| < \left\| g _ { 2 } \right\|$ , if the minimum-norm point of $\textbf {  { g } }$ satisfying ${ \pmb g } = \alpha { \pmb g } _ { 1 } + ( 1 - \alpha ) { \pmb g } _ { 2 }$ is outside the convex hull composed by $\left\{ g _ { 1 } , g _ { 2 } \right\}$ , or equivalently $\alpha > 1$ , MGDA clamps α to $\alpha = 1$ and the optimal $g ^ { \star } = g _ { 1 }$ . Then the projections of $\pmb { g } ^ { \star }$ onto $\pmb { g } _ { 1 }$ and $\mathbf { \delta } _ { \mathbf { { \boldsymbol { g } } } 2 }$ will be $\left\| g _ { 1 } \right\|$ and $\pmb { g } _ { 1 } \pmb { u } _ { 2 } ^ { \top } ~ ( \pmb { u } _ { 2 } = { \pmb { g } } _ { 2 } / \Vert \pmb { g } _ { 2 } \Vert )$ , respectively. As $\lvert | \mathbf { g } _ { 1 } \rvert | > \lvert \mathbf { g } _ { 1 } \mathbf { u } _ { 2 } ^ { \intercal } \rvert$ , so MGDA still focuses on tasks with smaller gradient magnitudes.

## C.3 ANALYSIS OF PCGRAD

PCGrad (Yu et al., 2020) mitigates the gradient conflicts by projecting the gradient of one task to the orthogonal direction of the others, and the aggregated gradient can be written as:

$$
\boldsymbol {g} = \sum_ {t} \left(\boldsymbol {g} _ {t} + \sum_ {\tau} C _ {t \tau} \boldsymbol {u} _ {\tau}\right), \tag {31}
$$

with $\pmb { u } _ { t } = \pmb { g } _ { t } / \left\| \pmb { g } _ { t } \right\|$ and the coefficients:

$$
C _ {t t} = 0, C _ {t \tau} = \left[ - \left(\boldsymbol {g} _ {t} + \sum_ {t ^ {\prime} <   \tau ,} C _ {t t ^ {\prime}} \boldsymbol {u} _ {t ^ {\prime}}\right) \boldsymbol {u} _ {\tau} ^ {\top} \right] _ {+}, \forall t, \tau , \tag {32}
$$

where $[ \cdot ] _ { + }$ means the ReLU operator. Note that the tasks have been shuffled before calculating the aggregated gradient $\mathbf { \pmb { g } }$ to achieve expected symmetry with respect to the task order. Eq. (31) can be represented more compactly in the matrix form:

$$
\boldsymbol {g} = \mathbf {1} \left(\boldsymbol {I} _ {T} + \boldsymbol {C N}\right) \boldsymbol {G} \equiv \alpha \boldsymbol {G}, \tag {33}
$$

where ${ \cal I } _ { T }$ is the identity matrix, $C = \{ C _ { t \tau } \}$ is the coefficient matrix whose entries are given in $\mathtt { E q }$ (32) and ${ \cal N } = \mathrm { d i a g } ( \bar { 1 / } | | g _ { 1 } | | , \cdot \cdot \cdot , 1 / \bar { | | } g _ { T } \bar { | | } )$ is the diagonal normalization matrix. In Eq. (33) we use G and α to denote the raw gradients and scaling factors of all tasks. We find that PCGrad can also be regarded as loss weighting, with the loss weights given by $\pmb { \alpha } = \mathbf { 1 } \left( \pmb { I _ { T } } +\pmb { C N } \right)$ . However, it still may break the balance among tasks. For example with two tasks, assume the angle between the gradients is φ: 1) when $\pi / 2 \leqslant \phi < \pi$ , then $\pmb { C } = \left[ \begin{array} { c c } { 0 } & { - \pmb { g } _ { 1 } \pmb { g } _ { 2 } ^ { \top } / \left. \pmb { g } _ { 2 } \right. } \\ { - \pmb { g } _ { 1 } \pmb { g } _ { 2 } ^ { \top } / \left. \pmb { g } _ { 1 } \right. } & { 0 } \end{array} \right]$ and the projections onto the two raw gradients are $\| \pmb { g } _ { 1 } \| \sin ^ { 2 }$ φ and $\| g _ { 2 } \| \sin ^ { 2 } \phi ; 2 )$ when $0 < \phi < \pi / 2$ then $C = \mathbf { 0 }$ and the projections are $\| g _ { 1 } \| + \| g _ { 2 } \|$ cos φ and $\| g _ { 2 } \| + \| g _ { 1 } \|$ cos φ. In both cases, the projections are equal if and only $\operatorname { i f } \| \pmb { g } _ { 1 } \| = \| \pmb { g } _ { 2 } \|$ . Otherwise, the task with larger gradient magnitude will be trained more sufficiently, which may encounter the same problem as uniform scaling that na¨ıvely adds all the losses despite that the loss scales are highly different.

## C.4 $L _ { 2 }$ LOSS IN UNCERTAINTY WEIGHTING

For regression, uncertainty weighting (Kendall et al., 2018) regards the $L _ { 2 }$ loss as likelihood estimation on the sample target which follows the Gaussian distribution:

$$
- \log p (y \mid f (\boldsymbol {x})) = \frac {1}{2} \left(\frac {1}{\sigma^ {2}} \| y - f (\boldsymbol {x}) \| _ {2} ^ {2} + \log \sigma^ {2}\right), \tag {34}
$$

where $_ { \textbf { \em x } }$ is the data sample, $y$ is the ground-truth label, f denotes the prediction model and $\sigma$ is the standard deviation of Gaussian distribution. By setting $s = - \log \sigma ^ { 2 }$ , the scaled $L _ { 2 }$ loss is $\begin{array} { r } { L = \frac { 1 } { 2 } \left( e ^ { s } L _ { \mathrm { r e g } } - s \right) } \end{array}$ , which has a similar form as the scaled $L _ { 1 }$ loss except the front factor $1 / 2$ . So uncertainty weighting has difficulty in reaching a unified form for all kinds of losses, which is less general than our IMTL-L.

## C.5 GRADIENT OF GEOMETRIC MEAN

GLS (Chennupati et al., 2019) computes the loss as the geometric mean, its gradient with respect to model parameters are:

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} L = \frac {1}{T} \left(\prod_ {t} L _ {t}\right) ^ {\frac {1}{T} - 1} \sum_ {t} \left[ \left(\prod_ {\tau \neq t} L _ {\tau}\right) \nabla_ {\boldsymbol {\theta}} L _ {t} \right] (35) \\ = \frac {1}{T} \left(\prod_ {t} L _ {t}\right) ^ {\frac {1}{T}} \sum_ {t} \frac {\nabla_ {\boldsymbol {\theta}} L _ {t}}{L _ {t}} = \frac {L}{T} \sum_ {t} \frac {1}{L _ {t}} \left(\nabla_ {\boldsymbol {\theta}} L _ {t}\right). (36) \\ \end{array}
$$

where $L$ is the geometric mean loss and T is the task number. It is equivalent to weigh the taskspecific loss with its reciprocal value, except that there exists another term $L / T$ in the front where $L = ( \prod _ { t } L _ { t } ) ^ { \frac { 1 } { T } }$ , so GLS is sensitive to the loss scale changes of $\{ L _ { t } \}$ and not scale-invariant.

## D IMPLEMENTATION DETAILS

To solely compare the loss weighting methods, we fix the network structure and choose ResNet-50 (He et al., 2016) with dilation (Chen et al., 2017) and synchronized (Peng et al., 2018) batch normalization (Ioffe & Szegedy, 2015) as the shared backbone and PSPNet (Zhao et al., 2017) as the task-specific head, and the backbone model weights are pretrained on ImageNet (Deng et al., 2009). Following the common practice of semantic segmentation, in training we adopt augmentations as random resize (between 0.5 to 2), random rotate (between -10 to 10 degrees), Gaussian blur (with a radius of 5) and random horizontal flip. Besides, we apply strided cropping and horizontal flipping as testing augmentations. The predicted results in the overlapped region of different crops are averaged to obtain the aggregated prediction of the whole image. Only pixels with ground truth labels are included in loss and metric computation, while others are ignored. Semantic segmentation, instance segmentation, surface normal estimation and disparity/depth estimation are considered. As for the losses/metrics, semantic segmentation uses cross-entropy/mIoU, surface normal estimation adopts (1 − cos)/cosine similarity and both instance segmentation and disparity/depth estimation use $L _ { 1 }$ loss. We use polynomial learning rate with a power of 0.9, SGD with a momentum of 0.9 and weight decay of $1 0 ^ { - 4 }$ as the optimizer, with the model trained for 200 epochs. After passing through the shared backbone where strided convolutions exist, the feature maps have $1 / 8$ size as that of the input image. Then the results predicted by PSPNet (Zhao et al., 2017) heads are up-sampled to the original image size for loss and metric computation.

![](images/60ab5948fc715e9c133528269c58de29b5dd5ddd135bb7bcdf7edd46b08f64af.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Raw Image"] --> B["Semantic Segmentation"]
  A --> C["Disparity Estimation"]
  B --> D["Foreground Extraction"]
  D --> E["Offset Regression"]
  E --> F["Centroid Computation"]
  F --> G["Pixel Assignment"]
```
</details>

Figure 5: Pipeline used in the Cityscapes visual understanding experiment. The centroids are computed from the offset regression results. Each pixel is assigned to its nearest candidate centroid.

For the Cityscapes dataset, the batch size is 32 $( 2 \times 1 6 ~ \mathrm { G P U s } )$ with the initial learning rate 0.02. We train on the 2975 training images and validate on the 500 validation images $( 1 0 2 4 \times 2 0 4 8$ full resolution) where ground truth labels are provided. Three tasks are considered, namely semantic segmentation, instance segmentation and disparity/depth estimation. Training and testing are done on $7 1 3 \times 7 1 3$ crops. Semantic segmentation is to differentiate among the commonly used 19 classes. Instance segmentation is taken as offset regression, where each pixel $\pmb { p } _ { i } = ( x _ { i } , y _ { i } )$ approximates the relative offset $o _ { i } = ( \mathrm { d } x _ { i } , \mathrm { d } y _ { i } )$ ) with respect to the centroid $\pmb { c } _ { \mathrm { i d } ( \pmb { p } _ { i } ) }$ of its belonging instance id $\left( { { p } _ { i } } \right)$ To conduct inference, we abandon the time-consuming and complicated clustering methods adopted by the previous method (Kendall et al., 2018). Instead, we directly use the offset vectors $\left\{ o _ { i } \right\}$ predicted by the model to find the centroids of instances. By definition, the norm of a centroid’s offset vector should be 0, so we can transform the offset vector norm $\| o _ { i } \|$ to the probability $q _ { i }$ of being a centroid with the exponential function $q _ { i } = e ^ { - \| \mathbf { o } _ { i } \| }$ . Next a $7 \times 7$ edge filter is applied on the centroid probability map to filter out the spurious centroids on object edges resulting from the regression target ambiguity. The locations with centroid probability $q _ { i } < 0 . 1$ are also manually suppressed. Then $7 \times 7$ max-pooling on the filtered probability map is used to produce candidate centroids and filter out duplicate ones. With the predicted centroids $\{ c _ { i } \}$ , we can then assign each pixel $\mathbf { \nabla } p _ { i }$ to its belonging instance id $\left( { { { p } _ { i } } } \right)$ by the distance between its approximated centroids $\pmb { p } _ { i } + \pmb { o } _ { i }$ and the candidate centroids $\left\{ \pmb { c } _ { i } \right\} : \mathrm { i d } \left( \pmb { p } _ { i } \right) = \arg \operatorname* { m i n } _ { j } \left\| \pmb { p } _ { i } + \pmb { o } _ { i } - \pmb { c } _ { j } \right|$ . Depth is measured in pixels by the disparity between the left and right images. Fig. 5 shows the whole process. Note that we need to carefully deal with label transformation during data augmentation. For example, disparity ground truth needs to be up-scaled by s times if the image is up-sampled by s times. Also, the predicted offset vectors of the flipped input should be mirrored to comply with the normal one.

On the NYUv2 dataset, the batch size is 48 $( 6 \times 8 ~ \mathrm { G P U s } )$ with the initial learning rate 0.03. We use the 795 training images for training and the 654 validation images for testing with 480 × 640 full resolution. $4 0 1 \times 4 0 1$ crops are used for training and testing. 13 coarse-grain classes are considered in semantic segmentation. The surface normal is represented by the unit normal vector of the corresponding surface. When doing data augmentation, surface normal ground truth n = $( x , y , z )$ should be processed accordingly. If we resize the image by s times, the z coordinate of the normal vector should be scaled by s and renormalized: $\pmb { n } ^ { \prime } = \left( x , y , s z \right) / \left\| \left( x , y , s z \right) \right\|$ . If the image is rotated by the rotation matrix R, the normal vector should also be in-plane rotated $( x ^ { \prime } , y ^ { \prime } ) \bar { = } \left( x , y \right) R ^ { \top }$ with z unchanged. Moreover, the left-right flip should be applied on the normal vector $\pmb { n } ^ { \prime } = ( - x , y , z )$ when mirroring the image horizontally. During testing, the normal vectors in the overlapped region of crops are averaged and renormalized to produce the aggregated results. Depth is the absolute distance to the camera and measured by meters, which is inverse-proportional to the disparity measurement adopted by Cityscapes. So the depth in meters needs to be scaled by $1 / s$ when the image is scaled by s times, which is the reciprocal of disparity transformation.

CelebA contains 202,599 face images from 10,177 identities, where each image has 40 binary attribute annotations. We train on the 162,770 training images and test on the 19,867 validation images. Most of the implementation details are the same as those on the Cityscapes dataset, except that: 1) we employ the ResNet-18 as the backbone and linear classifiers as the task-specific heads, so totally 40 heads are attached on the backbone ; 2) the binary-cross entropy is used as the classification loss for each attribute; 3) the batch size is 256 (32 × 8 GPUs) and the model is trained from scratch for 100 epochs; 4) the input image has been aligned with the annotated 5 landmarks and cropped to $2 1 8 \times \bar { 1 } 7 8$

## E QUALITATIVE RESULTS

![](images/3bd277a1d853b90b8d12ca7a7d502de29969085905de23dacca75aeda41b972b.jpg)  
Figure 6: Qualitative comparisons between our IMTL and previous methods on Cityscapes.

![](images/64ef7f3133b04d47d14e898b38e1301c1386dfb3ca08e90a3149e9990231cf76.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 street photos showing vehicles, pedestrians, and pedestrian walking in various settings (no visible text or symbols)
</details>

Figure 7: Qualitative results of our IMTL on Cityscapes. Semantic segmentation, instance segmentation and disparity estimation predictions are produced by a single network. The task-shared backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 1024×2048.

![](images/8e331289eb898514862a3d0603d9e63a30cf244b7bee54ea6e2923ebb44893af.jpg)  
Figure 8: Qualitative results of our IMTL on NYUv2. Semantic segmentation, surface normal estimation and depth estimation predictions are produced by a single network. The task-shared backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 480 × 640.