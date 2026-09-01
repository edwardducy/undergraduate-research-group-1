# Towards Impartial Multi-task Learning

<!-- PAGE 1 -->

Under review as a conference paper at ICLR 2021
TOWARDS IMPARTIAL MULTI-TASK LEARNING
Anonymous authors
Paper under double-blind review
ABSTRACT
Multi-task learning (MTL) has been widely used in representation learning. How￾ever, na¨ıvely training all tasks simultaneously may lead to the partial training is￾sue, where specific tasks are trained more adequately than others. In this paper,
we propose to learn multiple tasks impartially. Specifically, for the task-shared
parameters, we optimize the scaling factors via a closed-form solution, such that
the aggregated gradient (sum of raw gradients weighted by the scaling factors) has
equal projections onto individual tasks. For the task-specific parameters, we dy￾namically weigh the task losses so that all of them are kept at a comparable scale.
Further, we find the above gradient balance and loss balance are complementary
and thus propose a hybrid balance method to further improve the performance.
Our impartial multi-task learning (IMTL) can be end-to-end trained without any
heuristic hyper-parameter tuning, and is general to be applied on all kinds of losses
without any distribution assumption. Moreover, our IMTL can converge to sim￾ilar results even when the task losses are designed to have different scales, and
thus it is scale-invariant. We extensively evaluate our IMTL on the standard MTL
benchmarks including Cityscapes, NYUv2 and CelebA. It achieves the new state￾of-the-art among loss weighting methods under the same experimental settings.
1 INTRODUCTION
Recent deep networks in computer vision can match or even surpass human beings on some specific
tasks separately. However, in reality multiple tasks (e.g., semantic segmentation and depth estima￾tion) must be solved simultaneously. Multi-task learning (MTL) (Caruana, 1997; Evgeniou & Pontil,
2004; Ruder, 2017; Zhang & Yang, 2017) aims at sharing the learned representation among tasks
(Zamir et al., 2018) to make them benefit from each other and achieve better results and stronger
robustness (Zamir et al., 2020). However, sharing the representation can lead to a partial learning is￾sue: some specific tasks are learned well while others are overlooked, due to the different loss scales
or gradient magnitudes of various tasks and the mutual competition among them. Several methods
have been proposed to mitigate this issue either via gradient balance such as gradient magnitude
normalization (Chen et al., 2018) and Pareto optimality (Sener & Koltun, 2018), or loss balance
like homoscedastic uncertainty (Kendall et al., 2018). Gradient balance can evenly learn task-shared
parameters while ignoring task-specific ones. Loss balance can prevent MTL from being biased in
favor of tasks with large loss scales but cannot ensure the impartial learning of the shared parame￾ters. In this work, we find that gradient balance and loss balance are complementary, and combining
the two balances can further improve the results. To this end, we propose impartial MTL (IMTL)
via simultaneously balancing gradients and losses across tasks.
For gradient balance, we propose IMTL-G(rad) to learn the scaling factors such that the aggregated
gradient of task-shared parameters has equal projections onto the raw gradients of individual tasks
(see Fig. 1 (d)). We show that the scaling factor optimization problem is equivalent to finding the
angle bisector of gradients from all tasks in geometry, and derive a closed-form solution to it. In
contrast with previous gradient balance methods such as GradNorm (Chen et al., 2018), MGDA
(Sener & Koltun, 2018) and PCGrad (Yu et al., 2020), which have learning biases in favor of tasks
with gradients close to the average gradient direction, those with small gradient magnitudes, and
those with large gradient magnitudes, respectively (see Fig. 1 (a), (b) and (c)), in our IMTL-G
task-shared parameters can be updated without bias to any task.
For loss balance, we propose IMTL-L(oss) to automatically learn a loss weighting parameter for
each task so that the weighted losses have comparable scales and the effect of different loss scales
1

<!-- PAGE 2 -->

Under review as a conference paper at ICLR 2021
g
g1
g2
g3
g
g1
g2
g3
(a) GradNorm
g
g1
g2
g3
g
g1
g2
g3
g
g1
g2
g3
(b) MGDA
g
g1
g2
g3
g
g1
g2
g3
g
g1
g2
g3
g
g1
g2
g3
(c) PCGrad
g
g1
g2
g3
(d) IMTL-G
Figure 1: Comparison of gradient balance methods. In (a) to (d), g1, g2 and g3 represent the
gradient computed by the raw loss of each task, respectively. The gray surface represents the plane
composed by these gradients. The red arrow denotes the aggregated gradient computed by the
weighted sum loss, which is ultimately used to update the model parameters. The blue arrows show
the projections of g onto the raw gradients {gt}. g has the largest projection on g2 (nearest to the
mean direction), g3 (smallest magnitude) and g2 (largest magnitude) for GradNorm, MGDA and
PCGrad, respectively, while the projections are equal on {gt} in our IMTL-G.
from various tasks can be canceled-out. Compared with uncertainty weighting (Kendall et al., 2018),
which has biases towards regression tasks rather than classification tasks, our IMTL-L treats all tasks
equivalently without any bias. Besides, we model the loss balance problem from the optimization
perspective without any distribution assumption that is required by (Kendall et al., 2018). Therefore,
ours is more general and can be used in any kinds of losses. Moreover, the loss weighting parameters
and the network parameters can be jointly learned in an end-to-end fashion in IMTL-L.
Further, we find the above two balances are complementary and can be combined to improve the
performance. Specifically, we apply IMTL-G on the task-shared parameters and IMTL-L on the
task-specific parameters, leading to the hybrid balance method IMTL. Our IMTL is scale-invariant:
the model can converge to similar results even when the same task is designed to have different loss
scales, which is common in practice. For example, the scale of the cross-entropy loss in semantic
segmentation may have different scales when using “average” or “sum” reduction over locations in
the loss computation. We empirically validate that our IMTL is more robust against heavy loss scale
changes than its competitors. Meanwhile, our IMTL only adds negligible computational overheads.
We extensively evaluate our proposed IMTL on standard benchmarks: Cityscapes, NYUv2 and
CelebA, where the experimental results show that IMTL achieves superior performances under all
settings. Besides, considering there lacks a fair and practical benchmark for comparing MTL meth￾ods, we unify the experimental settings such as image resolution, data augmentation, network struc￾ture, learning rate and optimizer option. We re-implement and compare with the representative MTL
methods in a unified framework, which will be publicly available. Our contributions are:
• We propose a novel closed-form gradient balance method, which learns task-shared param￾eters without any task bias; and we develop a general learnable loss balance method, where
no distribution assumption is required and the scale parameters can be jointly trained with
the network parameters.
• We unveil that gradient balance and loss balance are complementary and accordingly pro￾pose a hybrid balance method to simultaneously balance gradients and losses.
• We validate that our proposed IMTL is loss scale-invariant and is more robust against loss
scale changes compared with its competitors, and we give in-depth theoretical and experi￾mental analyses on its connections and differences with previous methods.
• We extensively verify the effectiveness of our IMTL. For fair comparisons, a unified code￾base will also be publicly available, where more practical settings are adopted and stronger
performances are achieved compared with existing code-bases.
2 RELATED WORK
Recent advances in MTL mainly come from two aspects: network structure improvements and loss
weighting developments. Network-structure methods based on soft parameter-sharing usually lead
to high inference cost (review in Appendix A). Loss weighting methods find loss weights to be mul￾tiplied on the raw losses for model optimization. They employ a hard parameter-sharing paradigm
2

<!-- PAGE 3 -->

Under review as a conference paper at ICLR 2021
(Ruder, 2017), where several light-weight task-specific heads are attached upon the heavy-weight
task-agnostic backbone. There are also efforts that learn to group tasks and branch the network in
the middle layers (Guo et al., 2020; Standley et al., 2020), which try to achieve better accuracy￾efficiency trade-off and can be seen as semi-hard parameter-sharing. We believe task grouping and
loss weighting are orthogonal and complementary directions to facilitate multi-task learning and
can benefit from each other. In this work we focus on loss weighting methods which are the most
economic as almost all of the computations are shared across tasks, leading to high inference speed.
Task Prioritization (Guo et al., 2018) weights task losses by their difficulties to focus on the harder
tasks during training. Uncertainty weighting (Kendall et al., 2018) models the loss weights as data￾agnostic task-dependent homoscedastic uncertainty. Then loss weighting is derived from maximum
likelihood estimation. GradNorm (Chen et al., 2018) learns the loss weights to enforce the norm
of the scaled gradient for each task to be close. MGDA (Sener & Koltun, 2018) casts multi-task
learning as multi-object optimization and finds the minimum-norm point in the convex hull com￾posed by the gradients of multiple tasks. Pareto optimality is supposed to be achieved under mild
conditions. GLS (Chennupati et al., 2019) instead uses the geometric mean of task-specific losses as
the target loss, we will show it actually weights the loss by its reciprocal value. PCGrad (Yu et al.,
2020) avoids interferences between tasks by projecting the gradient of one task onto the normal
plane of the other. DSG (Lu et al., 2020) dynamically makes a task “stop or go” by its converging
state, where a task is updated only once for a while if it is stopped. Although many loss weighting
methods have been proposed, they are seldom open-sourced and rarely compared thoroughly under
practical settings where strong performances are achieved, which motivates us to give an in-depth
analysis and a fair comparison about them.
3 IMPARTIAL MULTI-TASK LEARNING
In MTL, we map a sample x ∈ X to its labels {yt ∈ Yt}t∈[1,T]
of all T tasks through multiple task￾specific mappings {ft : X → Yt}. In most loss weighting methods, the hard parameter-sharing
paradigm is employed, such that ft is parameterized by heavy-weight task-shared parameters θ
and light-weight task-specific parameters θt. All tasks take the same shared intermediate feature
z = f (x; θ) as input, and the t-th task head outputs the prediction as ft (x) = ft (z; θt). We aim
to find the scaling factors {αt} for all T task losses {Lt (ft (x), yt)}, so that the weighted sum
loss L =
P
t αtLt can be optimized to make all tasks perform well. This poses great challenges
because: 1) losses may have distinguished forms such as cross-entropy loss and cosine similarity; 2)
the dynamic ranges of losses may differ by orders of magnitude. In this work, we propose a hybrid
solution for both the task-shared parameters θ and the task-specific parameters {θt}, as Fig. 2.
3.1 GRADIENT BALANCE: IMTL-G
Heads
Shared
Feature
Backbone
Loss 
Balance
Gradient
Balance
𝒛 = 𝒇(𝒙; 𝜽)
𝒇𝟏(𝒛;𝜽𝟏) 𝒇𝟐(𝒛;𝜽𝟐) 𝒇𝟑(𝒛;𝜽𝟑)
𝜽
𝜽𝟏 𝜽𝟐 𝜽𝟑
Impartial Multi-task Learning
𝒙
Figure 2: Overview of IMTL.
For task-shared parameters θ, we can receive T
gradients {gt = ∇θLt} via back-propagation
from all of the T raw losses {Lt}, and these
gradients represent optimal update directions
for individual tasks. As the parameters θ can
only be updated with a single gradient, we
should compute an aggregated gradient g by the
linear combination of {gt}. It also implies to
find the scaling factors {αt} of raw losses {Lt},
since g =
P
t αtgt = ∇θL = ∇θ (
P
t αtLt).
Motivated by the principle of balance among
tasks, we propose to make the projections of g
onto {gt} to be equal, as Fig. 1 (d). In this way,
we treat all tasks equally so that they progress in the same speed and none is left behind. Formally,
let {ut = gt/ kgtk} denote the unit-norm vector of {gt} which are row vectors, then we have:
gu>
1 = gu>
t ⇔ g (u1 − ut)
> = 0, ∀ 2 6 t 6 T. (1)
The above problem is under-determined, but we can obtain the closed-form results of {αt} by
constraining P
t αt = 1. Assume α = [α2, · · · , αT ], U > =

u
>
1 − u
>
2
, · · · ,u
>
1 − u
>
T

,
3

<!-- PAGE 4 -->

Under review as a conference paper at ICLR 2021
Algorithm 1 Training by Impartial Multi-task Learning
Input: input sample x, task-specific labels {yt} and learning rate η
Output: task-shared/-specific parameters θ/{θt}, scale parameters {st}
1: compute task-shared feature z = f (x; θ)
2: for t = 1 to T do
3: compute task prediction by head network ft (x) = f
net
t
(z; θt)
4: compute raw loss by loss function L
raw
t = L
func
t
(ft (x), yt)
5: compute scaled loss Lt = bastL
raw
t − st (default a = e, b = 1) . loss balance
6: compute gradient of shared feature z: gt = ∇zLt
7: compute unit-norm gradient ut =
gt
kgtk
8: end for
9: compute gradient differences D> =

g
>
1 − g
>
2
, · · · , g
>
1 − g
>
T

10: compute unit-norm gradient differences U > =

u
>
1 − u
>
2
, · · · ,u
>
1 − u
>
T

11: compute scaling factors for tasks 2 to T: α2:T = g1U >

DU >
−1
. gradient balance
12: compute scaling factors for all tasks: α =

1 − 1α>
2:T
, α2:T

13: update task-shared parameters θ = θ − η∇θ (
P
t αtLt)
14: for t = 1 to T do
15: update task-specific parameters θt = θt − η∇θtLt
16: update loss scale parameter st = st − η
∂Lt
∂st
17: end for
D> =

g
>
1 − g
>
2
, · · · , g
>
1 − g
>
T

and 1 = [1, · · · , 1], from Eq. (1) we can obtain:
α = g1U
>

DU >
−1
. (IMTL-G) (2)
The detailed derivation is in Appendix B.1. After obtaining α, the scaling factor of the first task can
be computed by α1 = 1 − 1α> since P
P t αt = 1. The optimized {αt} are used to compute L =
t αtLt, which is ultimately minimized by SGD to update the model. By now, back-propagation
needs to be executed T times to obtain the gradient of each task loss with respect to the heavy-weight
task-shared parameters θ, which is time-consuming and non-scalable. We replace the parameter￾level gradients {gt = ∇θLt} with feature-level gradients {∇zLt} to compute {αt}. This implies
to achieve gradient balance with respect to the last shared feature z as a surrogate of task-shared
parameters θ, since it is possible for the network to back-propagate this balance all the way through
the task-shared backbone starting from z. This relaxation allows us to do back propagation through
the backbone only once after obtaining {αt}, and thus the training time can be dramatically reduced.
3.2 LOSS BALANCE: IMTL-L
For the task-specific parameters {θt}, we cannot employ IMTL-G described above, because
∇θtLτ = 0, ∀t 6= τ , and thus only the gradient of the corresponding task ∇θtLt can be obtained for
each θt. Instead we propose to balance the losses among tasks by forcing the scaled losses {αtLt}
to be constant for all tasks, without loss of generality, we take the constant as 1. Then the most direct
idea is to compute the scaling factors as {αt = 1/Lt}, but they are sensitive to outlier samples and
manifest severe oscillations, so we further propose to learn to scale losses via gradient descent and
thus stronger stability can be achieved. Suppose the positive losses {Lt > 0} are to be balanced, we
first introduce a mapping function h : R → R
+ to transform the arbitrarily-ranged learnable scale
parameters {st} to positive scaling factors {h (st) > 0}, hereafter we abandon the subscript t for
brevity. Then we should construct an appropriate scaled loss g (s) so that both network parameters
θ and scale parameter s can be optimized by minimizing g (s). On one hand, we balance different
tasks by encouraging the scaled losses h (s)L(θ) to be 1 for all tasks, so the optimality s
? of s is
achieved when h (s)L(θ) = 1, or equivalently:
f (s) ≡ h (s)L(θ) − 1 = 0, if s = s
?
. (3)
One may expect to minimize |f (s)| = |h (s)L(θ) − 1| to find s
?
, however when h (s)L(θ) < 1,
the gradient with respect to θ, ∇θ |f (s)| = −h (s) ∇θL(θ), is in the opposite direction. On the
4

<!-- PAGE 5 -->

Under review as a conference paper at ICLR 2021
other hand, assume our scaled loss g (s) is a differentiable convex function with respect to s, then
its minimum is achieved if and only if s = s
?
, where the derivative of g (s) is zero:
g
0
(s) = 0, if s = s
?
. (4)
From Eq. (3) and (4) we find that the values of f (s) and g
0
(s) are both 0 when s = s
?
, we can then
regard f (s) as the derivative of g (s), which is our target scaled loss and used to optimize both the
network parameters θ and loss scale parameter s, then we have:
g
0
(s) = f (s) ⇔ g (s) = Z
f (s) ds = L(θ)
Z
h (s) ds − s. (5)
From Eq. (3) and (5), we notice that both h (s) and R
h (s) ds denote loss scales, so we have
R
h (s) ds = Ch (s), where C > 0 is a constant. According to ordinary differential equation,
R
h (s) ds must be the exponential function: R
h (s) ds = bas with a > 1, b > 0 (see Appendix
B.2). We then have g
00 (s) = kas
, k > 0, which is always positive and verifies our assump￾tion about the convexity of g (s). Also note that the gradient of g (s) with respect to θ, ∇θg (s)= R
h (s) ds∇θL(θ) = bas∇θL(θ), is in the appropriate direction since bas > 0. As an instantiation,
we set R
h (s) ds = e
s
(a = e, b = 1), then
g (s) = e
sL(θ) − s, (IMTL-L). (6)
From Eq. (6) we find that the raw loss is scaled by e
s
, and −s acts as a regularization to avoid
the trivial solution s = −∞ while minimizing the scaled loss g (s). As for implementation, the
task losses {Lt} are scaled by {e
st }, and the scaled losses {e
stL − st} are used to update both the
network parameters θ, {θt} and the scale parameters {st}.
3.3 HYBRID BALANCE: IMTL
We have introduced IMTL-G/IMTL-L for task-shared/task-specific parameters, and both of them
produce scaling factors to be applied on the raw losses. They can be used solely, but we find them
complementary and able to be combined to improve the performance. In IMTL-G, even if the raw
losses are multiplied by arbitrary (maybe different among tasks) positive factors, the direction of the
aggregated gradient g stays unchanged. Because by definition g =
P
t αtgt is the angular bisector
of the gradients {gt}, and positive scaling will not change the directions of {gt} and thus that of
g (proof in Theorem 2). So we can also obtain the scale factors {αt} in IMTL-G with the losses
that have been scaled by {st} from IMTL-L. IMTL-G and IMTL-L are combined as: 1) the task￾specific parameters {θt} and scale parameters {st} are updated by scaled losses {e
stLt − st}; 2) the
task-shared parameters θ are updated by P
t αt (e
stLt) which is the weighted average of {e
stLt},
with the weights {αt} computed by {∇z (e
stLt)} using IMTL-G. Note that the regularization terms
{−st} in Eq. (6) are constants with respect to θ and z, and thus can be ignored when computing
gradients and updating parameters in IMTL-G. In this way, we achieve both gradient balance for
task-shared parameters and loss balance for task-specific parameters, leading to our full IMTL as
illustrated in Alg. 1.
4 DISCUSSION
We draw connections between our method and previous state-of-the-arts 1
in Fig. 3. We will show
that previous methods can all be categorized as gradient or loss balance, and thus each of them
can be seen as a specification of our method. However, all of them have some intrinsic biases or
short-comings leading to inferior performances, which we try to overcome.
GradNorm (Chen et al., 2018) balances tasks by making the norm of the scaled gradient for each
task to be approximately equal. It also introduces the inverse training rate and a hyper-parameter γ
to control the strength of approaching the mean gradient norm, such that tasks which learn slower
can receive larger gradient magnitudes. However, it does not take into account the relationship of
the gradient directions. We show that when the angle between the gradients of each pair of tasks is
identical, our IMTL-G leads to the equivalent solution as GradNorm.
1Our analysis of PCGrad (Yu et al., 2020) can be found in Appendix C.3.
5

<!-- PAGE 6 -->

Under review as a conference paper at ICLR 2021
loss balance gradient balance
perpendicular
to convex hull 
angular bisector
of task gradients
may degrade to
uniform scaling
hyper-parameter
tuning needed uncertainty
(𝛼cls𝐿cls ≈ 1/2,
𝛼reg𝐿reg ≈ 1)
GLS
(𝛼𝑡𝐿𝑡 = 𝐿/𝑇)
GradNorm
(𝑝𝑡 ∝ 𝒖𝑡𝒖𝑠
⊤)
PCGrad
(𝑝𝑡 ∝ 𝒈𝑡
)
MGDA
(𝑝𝑡 ∝ 𝒈𝑡 −1
)
IMTL-L
(𝛼𝑡𝐿𝑡 ≈ const)
IMTL-G
(𝑝𝑡 = const)
complementary
overlook classification tasks,
distribution assumption
unstable when task
number is large
equal scaled loss
among tasks
Figure 3: Relationship between our IMTL and previous methods. The blue dashed arrow indicates
the characteristic of each method. In the loss balance methods, we annotate the scaled loss in
the bracket. Lcls, Lreg and Lt are the raw loss of classification, regression and individual task,
respectively. αcls, αreg and αt is the corresponding loss scale. L is the geometric mean loss and T
is the task number. In the gradient balance methods, we annotate the projections of the aggregated
gradient g =
P
t αtgt onto the raw gradient gt of the t-th task in the bracket. ut = gt/ kgtk is the
unit-norm vector, pt = gu>
t
is the projection of g onto gt and us =
P
t ut is the mean direction.
Theorem 1. If the angle between any pair of ut,uτ stays constant: utu
>
τ = C1, ∀t 6= τ with
C1 < 1, then our IMTL-G leads to the same solution as that of GradNorm: gu>
t = C2 ⇔ nt ≡
kαtgtk = αt kgtk = C3. In the above ut = gt/ kgtk, C1, C2 and C3 are constants.
Proof in Appendix C.1. In GradNorm, if without the above constant-angle condition utu
>
τ = C1,
the projection of the aggregated gradient g onto task-specific gradient, gu>
t = (P
τ C3uτ )u
>
t =
C3 (
P
τ uτ )u
>
t
, is proportional to (
P
τ uτ )u
>
t
. It tends to optimize the “majority tasks” whose
gradient directions are closer to the mean direction P
t ut, resulting in undesired task bias.
MGDA (Sener & Koltun, 2018) finds the weighted average gradient g =
P
t αtgt with minimum
norm in the convex hull composed by {gt}, so that P
t αt = 1 and αt > 0, ∀t. It adopts an
iterative method based on Frank-Wolfe algorithm, but it is inefficient and unstable. We find it can
be implemented more efficiently if without the constraints {αt > 0}. MGDA tries to minimize
gg> = (P
t αtgt) (P
τ ατgτ )
>
,
P
t αt = 1, it implies that g is perpendicular to the hyper-plane
composed by {gt}, as illustrated in Fig 1 (b). Basically we have:
g ⊥ (g1 − gt) ⇔ g (g1 − gt)
> = 0, ∀ 2 6 t 6 T, (7)
and can obtain α = g1D>

DD>
−1
(see Appendix C.2). From Eq. (7), we note that the relaxed
MGDA satisfies: gg>
t = C. Then the projection of g onto gt, gu>
t = C/ kgtk, is inversely
proportional to the norm of gt. So MGDA focuses on tasks with smaller gradient magnitudes, which
breaks the task balance. Even with constraints {αt > 0}, the problem still exists (see Appendix C.2).
Uncertainty weighting (Kendall et al., 2018) regards the task uncertainty as loss weight. For regres￾sion, it can derive L1 loss from Laplace distribution: − log p (y | f (x)) = |y − f (x)| /b + log b,
where x is the data sample, y is the ground-truth label, f denotes the prediction model and b is the di￾versity of Laplace distribution. L2 loss can be found in Appendix C.4. For classification, it takes the
cross-entropy loss as a scaled categorical distribution and introduces the following approximation:
− log p (y | f (x)) = − log 
softmaxy

f (x)
σ
2
 ≈ −
1
σ
2
log [softmaxy (f (x))] + log σ, (8)
in which softmaxy (·) stands for taking the y-th entry after the softmax (·) operator. MTL corre￾sponds to maximizing the joint likelihood of multiple targets, then the derivations yield the scaling
factor b/σ for the regression/classification loss. (Kendall et al., 2018) learn b and σ as model pa￾rameters which are updated by stochastic gradient descent. However, it is applicable only if we can
find appropriate correspondence between the loss and the distribution. It is difficult to be used for
losses such as cosine similarity, and it is impossible to traverse all kinds of losses to obtain a unified
form for them. Moreover, it sacrifices classification tasks. From Eq. (8) we can find that the scaled
cross-entropy loss is approximated as L = e
2sLcls−s if we set s = − log σ. By taking the derivative
6

<!-- PAGE 7 -->

Under review as a conference paper at ICLR 2021
Table 1: Comparison between IMTL and previous methods on Cityscapes, semantic segmentation,
instance segmentation and disparity/depth estimation are considered. The first group of columns
shows the regular results of different methods. The second group shows the results by manually
multiply the semantic segmentation loss with 10 before applying these methods. The subscript
numbers show the absolute change after scaling the loss to demonstrate the robustness of various
methods. The arrows indicate the values are the higher the better (↑) or the lower the better (↓). The
best and runner up results for each task are bold and underlined, respectively.
method sem. ins. disp. sem. ins. disp. time
mIoU↑ L1 ↓ L1 ↓ mIoU↑|∆|↓ L1 ↓|∆|↓ L1 ↓|∆|↓ s/iter↓
baselines
single-task 76.67 21.61 4.182 - - - -
uniform scaling 58.99 18.13 3.512 - - - 1.201
loss balance
uncertainty (Kendall et al., 2018) 74.91 16.43 2.895 74.000.91 16.770.34 2.9300.035 1.204
GLS (Chennupati et al., 2019) 75.65 17.18 2.953 66.229.43 21.093.91 3.3580.405 1.202
IMTL-L 76.89 16.69 2.944 75.551.34 17.490.80 2.9720.028 1.202
gradient balance
GradNorm (γ = 0) 76.27 17.99 3.195 72.963.31 19.361.37 3.2160.021 1.741
GradNorm (Chen et al., 2018) 52.17 19.88 4.098 54.232.06 20.530.65 4.1080.010 1.742
MGDA (w/o {αt > 0}) 76.95 53.19 6.296 76.360.59 29.0624.13 3.3772.919 1.777
MGDA (Sener & Koltun, 2018) 76.56 53.14 6.644 72.354.21 29.3823.76 3.3363.308 1.732
PCGrad (Yu et al., 2020) 60.50 17.99 3.450 66.335.83 17.990.00 3.3860.064 2.087
IMTL-G (exact) 76.13 17.46 2.979 - - - 2.769
IMTL-G 76.52 16.61 2.997 76.060.46 17.520.91 3.0200.023 1.776
hybrid balance
IMTL 77.00 15.96 2.905 76.560.44 15.850.11 2.9380.033 1.795
we have ∂L/∂s = 2e
2sLcls − 1. Then s is optimized to make the scaled loss e
2sLcls to be close
to 1/2. However, the scaled L1 loss is approximated as L = e
sLreg − s if we set s = − log b, and
taking the derivative we have ∂L/∂s = e
sLreg − 1. So s is optimized to make the scaled L1 loss to
achieve 1, which is twice of the classification loss, and thus the classification task is overlooked.
We would like to remark the differences between our IMTL-L and uncertainty weighting (Kendall
et al., 2018). Firstly, our derivation is motivated by the fairness among tasks, which intrinsically
differs from uncertainty weighting which is based on task uncertainty considering each task inde￾pendently. Secondly, IMTL-L learns to balance among tasks without any biases, while uncertainty
weighting may sacrifice classification tasks to favor regression tasks as derived above. Thirdly,
IMTL-L does not depend on any distribution assumptions and thus can be generally applied to var￾ious losses including cosine similarity, which uncertainty weighting may have difficulty with. As
far as we know, there is no appropriate correspondence between cosine similarity and specific dis￾tributions. Lastly, uncertainty weighting needs to deal with different losses case by case, it also
introduces approximations in order to derive scaling factors for certain losses (such as cross-entropy
loss) which may not be optimal, but our IMTL-L has a unified form for all kinds of losses.
GLS (Chennupati et al., 2019) calculates the target loss as the geometric mean: L = (Q
t Lt)
1
T , then
the gradient of L with respect to the model parameters θ can be obtained as Appendix C.5, which
can be regarded as to weigh the loss with its reciprocal value. However, as the gradient depends
on the value of L, so it is not scale-invariant to the loss scale changes. Moreover, we find it to be
unstable when the number of tasks is large because of the geometric mean computation.
Limitation. Taking fairness among tasks as a proxy is not always beneficial, especially when one of
the tasks is more important than others. In this case we may deliberately down-weight certain tasks
to better aid the most concerned one.
5 EXPERIMENTS
In previous methods, various experimental settings have been adopted but there are no extensive
comparisons. As one contribution of our work, we re-implement representative methods and present
fair comparisons among them under the unified code-base, where more practical settings are adopted
and stronger performances are achieved compared with existing code-bases. The implementations
7

<!-- PAGE 8 -->

Under review as a conference paper at ICLR 2021
exactly follow the original papers and open-sourced code to ensure the correctness. We run exper￾iments on the Cityscapes (Cordts et al., 2016), NYUv2 (Silberman et al., 2012) and CelebA (Liu
et al., 2015) dataset to extensively analyze different methods. Details can be found in Appendix D.
Results on Cityscapes. From Tab. 1 we can obtain several informative conclusions. The uniform
scaling baseline, which na¨ıvely adds all losses, tends to optimize tasks with larger losses and gradi￾ent magnitudes, resulting in severe task bias. Uncertainty weighting (Kendall et al., 2018) sacrifices
classification tasks to aid regression ones, leading to significantly worse results on semantic seg￾mentation compared with our IMTL-L. GradNorm (Chen et al., 2018) is very sensitive to the choice
of the hyper-parameter γ controlling the strength of equal gradient magnitudes, where the default
γ = 1.5 works well on NYUv2 but performs badly on Cityscapes. We find its best option is γ = 0
which makes the scaled gradient norm to be exactly equal. MGDA (Sener & Koltun, 2018) focuses
on tasks with smaller gradient magnitudes. So the performance of semantic segmentation is good but
the other two tasks have difficulty in converging. In addition, we find our proposed closed-form vari￾ant without the hard constraints {αt > 0} achieves similar results as the original iterative method.
Through the experiments we notice the closed-form solution almost always yields {αt > 0}. As
for PCGrad (Yu et al., 2020), it yields slightly better performance than uniform scaling because its
conflict projection will have no effect when the angles between the gradients are equal or less than
π/2. In contrast, our IMTL method, in terms of both gradient balance and loss balance, yields com￾petitive performance and achieves the best balance among tasks. Moreover, we verify that the two
balances are complementary and can be combined to further improve the performance, with the vi￾sualizations in Appendix E. Surprisingly, we find our IMTL can beat the single-task baseline where
each task is trained with a separate model. Training multiple tasks simultaneously can learn a better
representation from multiple levels of semantics, which can in turn improve individual tasks.
In addition, we present the real-world training time of each iteration for different methods in Tab.
1. As shown, loss balance methods are the most efficient, and our gradient balance method IMTL￾G adds acceptable computational overhead, similar to that of GradNorm (Chen et al., 2018) and
MGDA (Sener & Koltun, 2018). It benefits from computing gradients with respect to the shared
feature maps instead of the shared model parameters (the row of “IMTL-G (exact)”), which brings
similar performances but adds significant complexity due to multiple (T) backward passes through
the shared parameters. Our IMTL-G only needs to do backward computation on the shared pa￾rameters once after obtaining the loss weights via Eq. (2), in which the computation overhead
mainly comes from the matrix multiplication rather than the matrix inverse, since the inversed ma￾trix DU > ∈ R
(T −1)×(T −1) is small. We find that the results obtained by approximated gradients
{∇zLt} are even better than those of {∇θLt}, it may result from the use of SGD where we compute
the gradients with a batch of samples instead of the whole dataset. If the computed gradient intro￾duces a random error ∇\θLt = ∇θLt + et, from the stability analysis (Sener & Koltun, 2018) we
know the error in the loss weighting coefficients is bounded kαb − αk2 6 O (maxt ketk2
). We can
expect ketk2
to depend on the dimension of et, and thus we may receive a more accurate solution
via replacing ∇θLt with ∇zLt due to its smaller dimension.
As we outperform MGDA (Sener & Koltun, 2018) and PCGrad (Yu et al., 2020) significantly in
terms of the objective metrics shown in Tab. 1, we further compare the qualitative results of our
hybrid balance IMTL with the loss balance method uncertainty weighting (Kendall et al., 2018) and
the gradient balance method GradNorm (Chen et al., 2018) considering their strong performances
(see Fig. 5). For depth estimation we only show predictions at the pixels where ground truth (GT)
labels exist to compare with GT, which is different from Fig. 6 where depth predictions are shown
for all pixels. Consistent with results in Tab. 1, our IMTL shows visually noticeable improvements
especially for the semantic and instance segmentation tasks. It is worth noting that we conduct
experiments under strong baselines and practical settings which are seldom explored before, in this
case changing the backbone in PSPNet (Zhao et al., 2017) from ResNet-50 to ResNet-101 can only
improve mIoU of the semantic segmentation task around 0.5% according to the public code base2
.
Scale invariance. We are also interested in the scale invariance, which means how the results change
with the loss scale. For example, in semantic segmentation, the loss scale is different if we replace
the reduction method “mean” (averaged over all locations) with “sum” (summed over all locations)
in the cross-entropy loss computation, or the number of the interested classes increases. The scale
invariance is beneficial for model robustness. So to simulate this effect, we manually multiply the
2
https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet
8

<!-- PAGE 9 -->

Under review as a conference paper at ICLR 2021
Table 2: Experimental results on the NYUv2 and CelebA datasets, semantic segmentation, surface
normal estimation, depth estimation and multi-class classification are considered. Arrows indicate
the values are the higher the better (↑) or the lower the better (↓). The best and runner up results in
each column are bold and underlined, respectively.
method
NYUv2 CelebA
sem. norm. depth class.
mIoU↑ cos↑ L1 ↓ acc. ↑
baselines
single-task 56.82 0.8827 0.5097 -
uniform scaling 57.40 0.8684 0.4248 90.01
loss balance
uncertainty (Kendall et al., 2018) 57.20 - 0.4400 90.34
GLS (Chennupati et al., 2019) 57.84 0.8762 0.4243 -
IMTL-L 58.36 0.8864 0.4173 90.54
gradient balance
GradNorm (γ = 0) 55.96 0.8818 0.4317 90.91
GradNorm (Chen et al., 2018) 56.92 0.8787 0.4285 89.92
MGDA (w/o {αt > 0}) 49.43 0.8877 0.4839 89.68
MGDA (Sener & Koltun, 2018) 49.44 0.8875 0.4759 90.04
PCGrad (Yu et al., 2020) 57.48 0.8696 0.4253 89.99
IMTL-G 57.00 0.8785 0.4226 91.03
hybrid balance
IMTL 58.85 0.8888 0.4215 91.12
semantic segmentation loss by 10 and apply the same methods to see how the performances are
affected. In the last three columns of Tab. 1 we report the absolute changes resulting from the
multiplier. Our IMTL achieves the smallest performance fluctuations and thus the best invariance,
while other methods are more or less affected by the loss scale change.
Results on NYUv2. In Tab. 2 we find similar patterns as on Cityscapes, but NYUv2 is a rather
small dataset, so uniform scaling can also obtain reasonable results. Note that uncertainty weight￾ing (Kendall et al., 2018) cannot be directly used to estimate the normal surface when the cosine
similarity is used as the loss, since no appropriate distribution can be found to correspond to cosine
similarity. In this case, surface normal estimation owns the smallest gradient magnitude, so MGDA
(Sener & Koltun, 2018) learns it best but it performs not so well for the rest two tasks. Again, our
IMTL performs best taking advantage of the complementary gradient and loss balances.
Results on CelebA. To compare different methods in the many-task setting, in Tab. 2 we also
conduct the multi-label classification experiments on the CelebA (Liu et al., 2015) dataset. The
mean accuracy of 40 tasks is used as the final metric. Our IMTL outperforms its competitors in
the scenario where the task number is large, showing its superiority. Note that in this setting, GLS
(Chennupati et al., 2019) has difficulty in converging and no reasonable results can be obtained.
6 CONCLUSION AND FUTURE WORK
We propose an impartial multi-task learning method integrating gradient balance and loss balance,
which are applied on task-shared and task-specific parameters, respectively. Through our in-depth
analysis, we have theoretically compared our method with previous state-of-the-arts. We have also
showed that those state-of-the-arts can all be categorized as gradient or loss balance, but lead to
specific bias among tasks. Through extensive experiments we verify our analysis and demonstrate
the effectiveness of our method. Besides, for fair comparisons, we contribute a unified code-base,
which adopts more practical settings and delivers stronger performances compared with existing
code-bases, and it will be publicly available for future research.
In our experiments we find under-fitting is the main reason that prevents the model from performing
well on specific tasks. A more general problem of synchronizing training states among different
tasks where both under-fitting and over-fitting may exist is worth exploring. Besides, the multi￾dataset setting where multiple tasks correspond to multiple datasets that own different characteris￾tics, is practical and as important as the considered multi-label one. We would like to study IMTL
for more scenarios and compare it with other methods in the future.
9

<!-- PAGE 10 -->

Under review as a conference paper at ICLR 2021
REFERENCES
Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille.
Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and
fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):
834–848, 2017.
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient
normalization for adaptive loss balancing in deep multitask networks. In International Conference
on Machine Learning, pp. 794–803, 2018.
Sumanth Chennupati, Ganesh Sistu, Senthil Yogamani, and Samir A Rawashdeh. Multinet++:
Multi-stream feature aggregation and geometric loss strategy for multi-task learning. In Pro￾ceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo
Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic
urban scene understanding. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 3213–3223, 2016.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hier￾archical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition,
pp. 248–255. IEEE, 2009.
Theodoros Evgeniou and Massimiliano Pontil. Regularized multi–task learning. In Proceedings of
the tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
pp. 109–117, 2004.
Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing
in multi-task cnns by neural discriminative dimensionality reduction. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, pp. 3205–3214, 2019.
Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural
architecture search towards general-purpose multi-task learning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 11543–11552, 2020.
Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task priori￾tization for multitask learning. In Proceedings of the European Conference on Computer Vision
(ECCV), pp. 270–287, 2018.
Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In
International Conference on Machine Learning, 2020.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog￾nition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.
770–778, 2016.
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, pp. 7132–7141, 2018.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine
Learning - Volume 37, pp. 448–456, 2015.
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses
for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pp. 7482–7491, 2018.
Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1871–
1880, 2019.
10

<!-- PAGE 11 -->

Under review as a conference paper at ICLR 2021
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild.
In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730–3738, 2015.
Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task
vision and language representation learning. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 10437–10446, 2020.
Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multi￾ple tasks by learning to mask weights. In Proceedings of the European Conference on Computer
Vision (ECCV), pp. 67–82, 2018.
Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multi￾ple tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
pp. 1851–1860, 2019.
Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for
multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pp. 3994–4003, 2016.
Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun.
Megdet: A large mini-batch object detector. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pp. 6181–6189, 2018.
Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with
residual adapters. In Advances in Neural Information Processing Systems, pp. 506–516, 2017.
Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint
arXiv:1706.05098, 2017.
Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task ar￾chitecture learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33,
pp. 4822–4829, 2019.
Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances
in Neural Information Processing Systems, pp. 527–538, 2018.
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and sup￾port inference from rgbd images. In European Conference on Computer Vision, pp. 746–760.
Springer, 2012.
Trevor Standley, Amir R Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese.
Which tasks should be learned together in multi-task learning? In International Conference on
Machine Learning, 2020.
Gjorgji Strezoski, Nanne van Noord, and Marcel Worring. Many task learning with task routing. In
Proceedings of the IEEE International Conference on Computer Vision, pp. 1375–1384, 2019.
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.
Gradient surgery for multi-task learning. arXiv preprint arXiv:2001.06782, 2020.
Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio
Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE Con￾ference on Computer Vision and Pattern Recognition, pp. 3712–3722, 2018.
Amir R Zamir, Alexander Sax, Nikhil Cheerla, Rohan Suri, Zhangjie Cao, Jitendra Malik, and
Leonidas J Guibas. Robust learning through cross-task consistency. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11197–11206, 2020.
Yu Zhang and Qiang Yang. A survey on multi-task learning. arXiv preprint arXiv:1707.08114,
2017.
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing
network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
pp. 2881–2890, 2017.
11

<!-- PAGE 12 -->

Under review as a conference paper at ICLR 2021
Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In Proceedings
of the International Conference on Learning Representations, 2017.
A RELATED WORK OF NETWORK STRUCTURE
Cross-stitch Networks (Misra et al., 2016) learn coefficients to linearly combine activations from
multiple tasks to construct better task-specific representations. To break the limitation of channel￾wise cross-task feature fusion only, NDDR-CNN (Gao et al., 2019) proposes the layer-wise cross￾channel feature aggregation as 1 × 1 convolutions on the concatenated feature maps from multiple
tasks. More generally, MTL-NAS (Gao et al., 2020) introduces cross-layer connections among
tasks to fully exploit the feature sharing from both low and high layers, extending the idea in Sluice
Networks (Ruder et al., 2019) by leveraging neural architecture search (Zoph & Le, 2017). The
parameters of these methods increase linearly with the number of tasks. To improve the model
compactness, Residual Adapters (Rebuffi et al., 2017) introduce a small amount of task-specific
parameters for each layer and convolve them with the task-agnostic representations to form the task￾related ones. MTAN (Liu et al., 2019) generates data-dependent attention tensors by task-specific
parameters to attend to the task-shared features. Single-tasking (Maninis et al., 2019) instead applies
squeeze-and-excitation (Hu et al., 2018) module to generate attentive vectors for each task. In Task
Routing (Strezoski et al., 2019), the attentive vectors are randomly sampled before training and are
fixed for each image. Piggyback (Mallya et al., 2018) opts to mask parameter weights in place of
activation maps, dealing with task-sharing from another point-of-view. The above methods can share
parameters among tasks to a large extent, however, they are not memory-efficient because each task
still needs to compute all of its own intermediate feature maps, which also leads to inferior inference
speed compared with loss weighting methods.
B DETAILED DERIVATION
B.1 GRADIENT BALANCE: IMTL-G
Here we give the detailed derivation of the closed-form solution of our IMTL-G, we also demonstrate
the scale-invariance property of our IMTL-G, which is invariant to the scale changes of losses.
Solution. As we want to achieve:
gu>
1 = gu>
t ⇔ g (u1 − ut)
> = 0, ∀ 2 6 t 6 T, (9)
where ut = gt/ kgtk, recall that we have g =
P
t αtgt and P
t αt = 1, if we set α = [α2, · · · , αT ]
and G> =

g
>
2
, · · · , g
>
T

, then α1 = 1 − 1α> and Eq. (9) can be expanded as:
 X
t
αtgt
!

u
>
1 − u
>
2
, · · · ,u
>
1 − u
>
T

= 0 ⇔

1 − 1α>, α


g1
G

U
> = 0, (10)
where U > =

u
>
1 − u
>
2
, · · · ,u
>
1 − u
>
T

, 1 and 0 indicate the all-one and all-zero row vector,
respectively. Eq. (10) can be solved by:
1 − 1α
>

g1 + αG
U
> = 0 ⇔ α

1
>g1 − G

U
> = g1U
>. (11)
Assume D> = g
>
1 1 − G> =

g
>
1 − g
>
2
, · · · , g
>
1 − g
>
T

, then we reach:
αDU > = g1U
> ⇔ α = g1U
>

DU >
−1
. (12)
Property. We can also prove the aggregated gradient g =
P
t αtgt with {αt} given in Eq. (12) is
invariant to the scale changes of losses {Lt} (or gradients {gt = ∇θLt}), as the following theorem.
Theorem 2. Given g =
P
t αtgt,
P
t αt = 1 satisfying gu>
t = C, when {Lt} are scaled by
{kt > 0} (equivalently, {gt} are scaled by {kt}), if g
0 =
P
t α
0
t
(ktgt),
P
t α
0
t = 1 satisfies g
0u
>
t =
C
0
, then g
0 = λg. In the above we have ut =
gt
kgtk =
ktgt
kktgtk
, λ, C and C
0 are constants.
12

<!-- PAGE 13 -->

Under review as a conference paper at ICLR 2021
Proof. As we have:
g =
X
t
αtgt =
X
t
αt
kt
ktgt and gu>
t = C, (13)
by constructing:
α
0
t =
αt
kt
/
X
τ
ατ
kτ
and g
0 =
X
t
α
0
t
(ktgt) = g/
X
τ
ατ
kτ
= λg, (14)
we have:
X
t
α
0
t = 1 and g
0u
>
t = C/X
τ
ατ
kτ
= C
0
. (15)
From Eq. (12) we know that {αt} has a unique solution, and thus g
0
satisfying IMTL-G is unique,
so it must be the one given by Eq. (14), then we can prove that g
0
and g are linearly correlated.
B.2 LOSS BALANCE: IMTL-L
With the ordinary differential equation, we can derive that the form of the scale function R
h (s) ds
in our IMTL-L must be exponential function. As we have:
Z
h (s) ds = Ch (s), C > 0. (16)
If we set y =
R
h (s) ds, then:
y = C
dy
ds
⇒
dy
y
=
1
C
ds, (17)
By taking the antiderivative:
Z
dy
y
=
1
C
Z
ds ⇒ ln y =
1
C
s + C
0
. (18)
Then we have:
Z
h (s) ds = y = e
C
0

e
1
C
s
= bas
, a > 1, b > 0. (19)
C DETAILED DISCUSSION
C.1 CONDITIONAL EQUIVALENCE OF IMTL-G AND GRADNORM
First we introduce the following lemma.
Lemma 3. If utu
>
τ = C1, ∀t 6= τ , then the solution {αt} of IMTL-G satisfies {αt > 0}.
Proof. As ut = gt/ kgtk, by constructing g =
P
t αtgt where:
αt = kgtk
−1
/
X
τ
kgτ k
−1
, (20)
then we have P
t αt = 1 and:
gu>
t =
 X
τ
uτut
!
/
X
τ
kgτ k
−1 = [(T − 1) C1 + 1] /
X
τ
kgτ k
−1 = C2. (21)
From Eq. (12) we know the solution {αt} of IMTL-G is unique, so it must be the one given by Eq.
(20) where {αt > 0}, so the lemma is proved.
13

<!-- PAGE 14 -->

Under review as a conference paper at ICLR 2021
Then we prove Theorem 1 which states that IMTL-G leads to the same solution as GradNorm when
the angle between any pair of gradients {gt} is identical: utu
>
τ = C1, ∀t 6= τ .
Proof. (⇒ Necessity) Given constant projections in IMTL-G, we have:
gu>
t =
 X
τ
ατgτ
!
u
>
t = C2. (22)
Recall that ut = gt/ kgtk and utu
>
τ = C1, ∀t 6= τ . From Lemma 3 we know that {αt} given by
IMTL-G must satisfy {αt > 0}. If we assume nt = kαtgtk, then we know αtgt = ntut and:
X
τ
nτuτu
>
t =
X
τ6=t
nτC1 + nt = C2. (23)
Now we obtain:
X
τ6=t
nτC1 + nt =
X
τ
nτC1 + (1 − C1) nt = C2. (24)
As C1 < 1, we can then prove nt = C3, ∀t. It implies the norm of the scaled gradient is constant,
which is requested by GradNorm (Chen et al., 2018). Moreover, we can obtain the relationship
among constants from Eq. (24):
C1T C3 + (1 − C1) C3 = C2 ⇒ C3 =
C2
(T − 1) C1 + 1
. (25)
(⇐ Sufficiency) In GradNorm, {αt} are always chosen to satisfy {αt > 0}, so if we assume nt =
kαtgtk, then given the constant norm of the scaled gradient in GradNorm, we have:
αtgt = ntut = C3ut, (26)
where ut = gt/ kgtk. As we have g =
P
t αtgt and utu
>
τ = C1, ∀t 6= τ , then we obtain:
gu>
t =
 X
τ
ατgτ
!
u
>
t =
 X
τ
C3uτ
!
u
>
t = C3 [(T − 1) C1 + 1] = C2. (27)
It means the projections of g onto {gt} are constant, which is requested by our IMTL-G.
Corollary 4. In GradNorm, if the solution {αt} satisfies P
t αt = 1 , then its constants are given
by C3 = 1/
P
t
kgtk
−1
and C2 = [(T − 1) C1 + 1] /
P
t
kgtk
−1
, and its scaling factors are given
by n
αt = kgtk
−1
/
P
τ
kgτ k
−1
o
.
Proof. By using αt = C3/ kgtk from Eq. (26), we have P
t C3/ kgtk = 1, then C3 =
1/
P
t
kgtk
−1
, and also we have αt = kgtk
−1
/
P
τ
kgτ k
−1
. As the relationship of C2 and C3
from Eq. (27) is given by C3 [(T − 1) C1 + 1] = C2, so C2 = [(T − 1) C1 + 1] /
P
t
kgtk
−1
.
C.2 CLOSED-FORM SOLUTION OF MGDA
In our relaxed MGDA (Sener & Koltun, 2018) without {αt > 0}, finding g =
P
P t αtgt with
t αt = 1 such that g has minimum norm is equivalent to find the normal vector of the hyper￾plane composed by {gt}. So we let g to be perpendicular to all of {g1 − gt} on the hyper-plane:
g ⊥ (g1 − gt) ⇔ g (g1 − gt)
> = 0, ∀ 2 6 t 6 T. (28)
If we set α = [α2, · · · , αT ] and G> =

g
>
2
, · · · , g
>
T

, then we have α1 = 1 − 1α>, and Eq. (28)
can be expanded as:
14

<!-- PAGE 15 -->

Under review as a conference paper at ICLR 2021
 X
t
αtgt
!

g
>
1 − g
>
2
, · · · , g
>
1 − g
>
T

= 0 ⇔

1 − 1α>, α


g1
G

D> = 0, (29)
where D> =

g
>
1 − g
>
2
, · · · , g
>
1 − g
>
T

, 1 and 0 indicates the all-one and all-zero row vector. Eq.
(29) can be represented as:
1 − 1α
>

g1 + αG
D> = 0 ⇔ α

1
>g1 − G

D> = g1D>.
As we also have D = 1
>g1 − G, then the closed-form solution of α is given by:
αDD> = g1D> ⇔ α = g1D>

DD>
−1
. (30)
Bias of MGDA. In the main text we state that MGDA focuses on tasks with small gradient mag￾nitudes, where we relaxed MGDA by not constraining {αt > 0}. However, even with these con￾straints, the problem still exists. For example in the context of two tasks, assume kg1k < kg2k, if
the minimum-norm point of g satisfying g = αg1 + (1 − α) g2 is outside the convex hull composed
by {g1, g2}, or equivalently α > 1, MGDA clamps α to α = 1 and the optimal g
? = g1. Then
the projections of g
? onto g1 and g2 will be kg1k and g1u
>
2
(u2 = g2/kg2k), respectively. As
kg1k >

g1u
>
2


, so MGDA still focuses on tasks with smaller gradient magnitudes.
C.3 ANALYSIS OF PCGRAD
PCGrad (Yu et al., 2020) mitigates the gradient conflicts by projecting the gradient of one task to the
orthogonal direction of the others, and the aggregated gradient can be written as:
g =
X
t
 
gt +
X
τ
Ctτuτ
!
, (31)
with ut = gt/ kgtk and the coefficients:
Ctt = 0, Ctτ =

−

gt +
X
t
0<τ,
Ctt0ut
0

 u
>
τ


+
, ∀t, τ, (32)
where [·]+ means the ReLU operator. Note that the tasks have been shuffled before calculating the
aggregated gradient g to achieve expected symmetry with respect to the task order. Eq. (31) can be
represented more compactly in the matrix form:
g = 1 (IT + CN) G ≡ αG, (33)
where IT is the identity matrix, C = {Ctτ } is the coefficient matrix whose entries are given in Eq.
(32) and N = diag (1/ kg1k , · · · , 1/ kgT k) is the diagonal normalization matrix. In Eq. (33) we
use G and α to denote the raw gradients and scaling factors of all tasks. We find that PCGrad can
also be regarded as loss weighting, with the loss weights given by α = 1 (IT + CN). However,
it still may break the balance among tasks. For example with two tasks, assume the angle between
the gradients is φ: 1) when π/2 6 φ < π, then C =

0 −g1g
>
2 / kg2k
−g1g
>
2 / kg1k 0

and the
projections onto the two raw gradients are kg1k sin2 φ and kg2k sin2 φ; 2) when 0 < φ < π/2,
then C = 0 and the projections are kg1k + kg2k cos φ and kg2k + kg1k cos φ. In both cases, the
projections are equal if and only if kg1k = kg2k. Otherwise, the task with larger gradient magnitude
will be trained more sufficiently, which may encounter the same problem as uniform scaling that
na¨ıvely adds all the losses despite that the loss scales are highly different.
15

<!-- PAGE 16 -->

Under review as a conference paper at ICLR 2021
C.4 L2 LOSS IN UNCERTAINTY WEIGHTING
For regression, uncertainty weighting (Kendall et al., 2018) regards the L2 loss as likelihood esti￾mation on the sample target which follows the Gaussian distribution:
− log p (y | f (x)) = 1
2

1
σ
2
ky − f (x)k
2
2 + log σ
2

, (34)
where x is the data sample, y is the ground-truth label, f denotes the prediction model and σ is
the standard deviation of Gaussian distribution. By setting s = − log σ
2
, the scaled L2 loss is
L =
1
2
(e
sLreg − s), which has a similar form as the scaled L1 loss except the front factor 1/2. So
uncertainty weighting has difficulty in reaching a unified form for all kinds of losses, which is less
general than our IMTL-L.
C.5 GRADIENT OF GEOMETRIC MEAN
GLS (Chennupati et al., 2019) computes the loss as the geometric mean, its gradient with respect to
model parameters are:
∇θL =
1
T
 Y
t
Lt
! 1
T −1
X
t




Y
τ6=t
Lτ

 ∇θLt

 (35)
=
1
T
 Y
t
Lt
! 1
T X
t
∇θLt
Lt
=
L
T
X
t
1
Lt
(∇θLt). (36)
where L is the geometric mean loss and T is the task number. It is equivalent to weigh the task￾specific loss with its reciprocal value, except that there exists another term L/T in the front where
L = (Q
t Lt)
1
T , so GLS is sensitive to the loss scale changes of {Lt} and not scale-invariant.
D IMPLEMENTATION DETAILS
To solely compare the loss weighting methods, we fix the network structure and choose ResNet￾50 (He et al., 2016) with dilation (Chen et al., 2017) and synchronized (Peng et al., 2018) batch
normalization (Ioffe & Szegedy, 2015) as the shared backbone and PSPNet (Zhao et al., 2017) as the
task-specific head, and the backbone model weights are pretrained on ImageNet (Deng et al., 2009).
Following the common practice of semantic segmentation, in training we adopt augmentations as
random resize (between 0.5 to 2), random rotate (between -10 to 10 degrees), Gaussian blur (with a
radius of 5) and random horizontal flip. Besides, we apply strided cropping and horizontal flipping as
testing augmentations. The predicted results in the overlapped region of different crops are averaged
to obtain the aggregated prediction of the whole image. Only pixels with ground truth labels are
included in loss and metric computation, while others are ignored. Semantic segmentation, instance
segmentation, surface normal estimation and disparity/depth estimation are considered. As for the
losses/metrics, semantic segmentation uses cross-entropy/mIoU, surface normal estimation adopts
(1 − cos)/cosine similarity and both instance segmentation and disparity/depth estimation use L1
loss. We use polynomial learning rate with a power of 0.9, SGD with a momentum of 0.9 and weight
decay of 10−4
as the optimizer, with the model trained for 200 epochs. After passing through the
shared backbone where strided convolutions exist, the feature maps have 1/8 size as that of the
input image. Then the results predicted by PSPNet (Zhao et al., 2017) heads are up-sampled to the
original image size for loss and metric computation.
For the Cityscapes dataset, the batch size is 32 (2 × 16 GPUs) with the initial learning rate 0.02.
We train on the 2975 training images and validate on the 500 validation images (1024 × 2048 full
resolution) where ground truth labels are provided. Three tasks are considered, namely semantic
segmentation, instance segmentation and disparity/depth estimation. Training and testing are done
on 713×713 crops. Semantic segmentation is to differentiate among the commonly used 19 classes.
Instance segmentation is taken as offset regression, where each pixel pi = (xi
, yi) approximates the
16

<!-- PAGE 17 -->

Under review as a conference paper at ICLR 2021
foreground
masking
semantic segmentation foreground extraction
disparity estimation
offset regression centroid computation pixel assignment
Figure 4: Pipeline used in the Cityscapes visual understanding experiment. The centroids are com￾puted from the offset regression results. Each pixel is assigned to its nearest candidate centroid.
relative offset oi = (dxi
, dyi) with respect to the centroid cid(pi) of its belonging instance id (pi).
To conduct inference, we abandon the time-consuming and complicated clustering methods adopted
by the previous method (Kendall et al., 2018). Instead, we directly use the offset vectors {oi}
predicted by the model to find the centroids of instances. By definition, the norm of a centroid’s
offset vector should be 0, so we can transform the offset vector norm koik to the probability qi of
being a centroid with the exponential function qi = e
−koik
. Next a 7 × 7 edge filter is applied
on the centroid probability map to filter out the spurious centroids on object edges resulting from
the regression target ambiguity. The locations with centroid probability qi < 0.1 are also manually
suppressed. Then 7 × 7 max-pooling on the filtered probability map is used to produce candidate
centroids and filter out duplicate ones. With the predicted centroids {ci}, we can then assign each
pixel pi
to its belonging instance id (pi) by the distance between its approximated centroids pi +oi
and the candidate centroids {ci}: id (pi) = arg minj kpi + oi − cjk. Depth is measured in pixels
by the disparity between the left and right images. Fig. 4 shows the whole process. Note that we
need to carefully deal with label transformation during data augmentation. For example, disparity
ground truth needs to be up-scaled by s times if the image is up-sampled by s times. Also, the
predicted offset vectors of the flipped input should be mirrored to comply with the normal one.
On the NYUv2 dataset, the batch size is 48 (6 × 8 GPUs) with the initial learning rate 0.03. We
use the 795 training images for training and the 654 validation images for testing with 480 × 640
full resolution. 401 × 401 crops are used for training and testing. 13 coarse-grain classes are
considered in semantic segmentation. The surface normal is represented by the unit normal vector
of the corresponding surface. When doing data augmentation, surface normal ground truth n =
(x, y, z) should be processed accordingly. If we resize the image by s times, the z coordinate
of the normal vector should be scaled by s and renormalized: n
0 = (x, y, sz) / k(x, y, sz)k. If
the image is rotated by the rotation matrix R, the normal vector should also be in-plane rotated
(x
0
, y0
) = (x, y) R> with z unchanged. Moreover, the left-right flip should be applied on the normal
vector n
0 = (−x, y, z) when mirroring the image horizontally. During testing, the normal vectors
in the overlapped region of crops are averaged and renormalized to produce the aggregated results.
Depth is the absolute distance to the camera and measured by meters, which is inverse-proportional
to the disparity measurement adopted by Cityscapes. So the depth in meters needs to be scaled by
1/s when the image is scaled by s times, which is the reciprocal of disparity transformation.
CelebA contains 202,599 face images from 10,177 identities, where each image has 40 binary at￾tribute annotations. We train on the 162,770 training images and test on the 19,867 validation
images. Most of the implementation details are the same as those on the Cityscapes dataset, except
that: 1) we employ the ResNet-18 as the backbone and linear classifiers as the task-specific heads,
so totally 40 heads are attached on the backbone ; 2) the binary-cross entropy is used as the clas￾sification loss for each attribute; 3) the batch size is 256 (32 × 8 GPUs) and the model is trained
from scratch for 100 epochs; 3) the input image has been aligned with the annotated 5 landmarks
and cropped to 218 × 178.
E QUALITATIVE RESULTS
17

<!-- PAGE 18 -->

Under review as a conference paper at ICLR 2021
U
n
c
e
r
t
ain
t
y
G
r
a
d
N
o
r
m IM
T
L
(
O
u
r
s
)
G
r
o
u
n
d
Tr
u
t
h
U
n
c
e
r
t
ain
t
y
G
r
a
d
N
o
r
m IM
T
L
(
O
u
r
s
)
G
r
o
u
n
d
Tr
u
t
h
U
n
c
e
r
t
ain
t
y
G
r
a
d
N
o
r
m IM
T
L
(
O
u
r
s
)
G
r
o
u
n
d
Tr
u
t
h
Figure 5: Qualitative comparisons between our IMTL and previous methods on Cityscapes.
18

<!-- PAGE 19 -->

Under review as a conference paper at ICLR 2021
Figure 6: Qualitative results of our IMTL on Cityscapes. Semantic segmentation, instance seg￾mentation and disparity estimation predictions are produced by a single network. The task-shared
backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 1024×2048.
19

<!-- PAGE 20 -->

Under review as a conference paper at ICLR 2021
Figure 7: Qualitative results of our IMTL on NYUv2. Semantic segmentation, surface normal
estimation and depth estimation predictions are produced by a single network. The task-shared
backbone is ResNet-50 and the task-specific heads are PSPNet. The image resolution is 480 × 640.
20