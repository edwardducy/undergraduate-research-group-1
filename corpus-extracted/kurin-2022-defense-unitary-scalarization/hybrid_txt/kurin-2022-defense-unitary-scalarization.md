# In Defense of the Unitary Scalarization for Deep Multi-Task Learning

Vitaly Kurin<sup>∗</sup>

University of Oxford

vitaly.kurin@cs.ox.ac.uk

Alessandro De Palma<sup>∗</sup>

University of Oxford

adepalma@robots.ox.ac.uk

Ilya Kostrikov

University of California, Berkeley

New York University

Shimon Whiteson

University of Oxford

M. Pawan Kumar

University of Oxford

## Abstract

Recent multi-task learning research argues against unitary scalarization, where training simply minimizes the sum of the task losses. Several ad-hoc multi-task op timization algorithms have instead been proposed, inspired by various hypotheses about what makes multi-task settings difficult. The majority of these optimizers require per-task gradients, and introduce significant memory, runtime, and implementation overhead. We show that unitary scalarization, coupled with standard regularization and stabilization techniques from single-task learning, matches or improves upon the performance of complex multi-task optimizers in popular supervised and reinforcement learning settings. We then present an analysis suggesting that many specialized multi-task optimizers can be partly interpreted as forms of regularization, potentially explaining our surprising results. We believe our results call for a critical reevaluation of recent research in the area.

## 1 Introduction

Multi-Task Learning (MTL) [5] exploits similarities between tasks to yield models that are more accurate, generalize better and require less training data. Owing to the success of MTL on traditional machine learning models [3, 16, 22] and of deep single-task learning across a variety of domains, a growing body of research has focused on deep MTL. The most straightforward way to train a neural network for multiple tasks at once is to minimize the sum of per-task losses. Adopting terminology from multi-objective optimization, we call this approach unitary scalarization.

While some work shows that multi-task networks trained via unitary scalarization exhibit superior performance to independent per-task models [29, 35], others suggest the opposite [30, 54, 58]. As a result, many explanations for the difficulty of MTL have been proposed, each motivating a new Specialized Multi-Task Optimizer (SMTO) [11, 42, 54, 62, 66]. These works typically claim that the proposed SMTO outperforms unitary scalarization, in addition to relevant prior work. However, SMTOs usually require access to per-task gradients either with respect to the shared parameters, or to the shared representation. Therefore, their reported performance gain comes at significant computation and memory cost, the overhead scaling linearly with the number of tasks. By contrast, unitary scalarization requires only the average of the gradients across tasks, which can be computed via a single backpropagation.

Existing SMTOs were introduced to solve challenges related to the optimization of the deep MTL problem. We instead postulate that the reported weakness of unitary scalarization is linked to experimental variability or to a lack of regularization, leading to the following contributions:

• A comprehensive experimental evaluation (§4) of recent SMTOs on popular multi-task benchmarks, showing that no SMTO consistently outperforms unitary scalarization in spite of the added complexity and overhead. In particular, either the differences between unitary scalarization and SMTOs are not statistically significant, or they can be bridged by standard regularization and stabilization techniques from the single-task literature. Our reinforcement learning (RL) experiments include optimizers previously applied only to supervised learning.  
• An empirical and technical analysis of the considered SMTOs, suggesting that they reduce overfitting on the multi-task problem and hence act as regularizers (§5). We conduct an ablation study and provide a collection of novel and existing technical results that support this hypothesis.  
• Code to reproduce the experiments, including a unified PyTorch [50] implementation of the considered SMTOs, is available at https://github.com/yobibyte/ unitary-scalarization-dmtl.

We believe that our results suggest that the considered SMTOs can be often replaced by less expensive techniques. We hope that these surprising results stimulate the search for a deeper understanding of MTL.

## 2 Related Work

Before diving into details of specific SMTOs in Section 5, we provide a high-level overview of the deep MTL research. Seminal work in MTL includes hard parameter sharing [6]: sharing neural network parameters between all tasks with, possibly, a separate part of the model for each task. Hard parameter sharing is still the major MTL approach adopted in natural language processing [9, 12], computer vision [46], and speech recognition [53]. In this work, we implicitly assume that each parameter update employs information from all tasks. However, not all works satisfy this assumption, either due to a large number of tasks [4, 36], or simply as an implementation decision [25, 37]. In this setting, MTL resembles other problems dealing with multiple tasks, i.e., continual [32], curriculum [47], and meta-learning [24], which are not the focus of this work.

Many works strive to improve the performance of deep multi-task models. One line of research hypothesizes that conflicting per-task gradient directions lead to suboptimal models, and focuses on explicitly removing such conflicts [11, 28, 41, 42, 62, 66]. Some authors postulate that loss imbalances across tasks hinder learning, proposing loss reweighting methods [10, 30, 40]. Sener and Koltun [54] and Navon et al. [48] propose that tasks compete for model capacity and interpret MTL as multi-objective optimization in order to cope with inter-task competition. Here, we focus on algorithms that explicitly rely on per-task gradients to try to outperform unitary scalarization (§5). Research on multi-task architectures [19, 46] or MTL algorithms exclusively motivated by determin istic loss reweighting [18, 30, 43] are orthogonal to our work. Both topics are investigated by a recent survey on pixel-level multi-task computer vision problems [61], which found that the minimization of tuned weighted sums of losses (scalarizations) is empirically competitive with deterministic loss reweighting and MGDA in the considered settings. These results are extended to popular SMTOs by a critical review from Xin et al. [63], concurrent to our work, which argues that the optimization and generalization performance of SMTOs can be matched by tuning scalarization coefficients. Our work reaches a similar conclusion, demonstrating that unitary scalarization performs on par with SMTOs when coupled with standard and inexpensive regularization or stabilization techniques. In other words, Xin et al. [63] provide complementary support for the link between SMTOs and regularization by showing that tuning scalarization weights positively affects generalization.

In addition to the common supervised settings, we also consider multi-task RL, whose research can be grouped into three categories: the first adds auxiliary tasks providing additional inductive biases to speed up learning [27] on a target task. The second, based on policy distillation, uses per-task teacher models to provide labels for a multi-task model or per-task policies as regularizers [49, 51, 57]. The third directly learns a shared policy [29], possibly via an SMTO [66]. We focus on the third category, whose literature reports varying performance for unitary scalarization (better [29] or worse [66] than per-task models), indicating confounding factors in evaluation pipelines and further motivating our work. PopArt [23, 60] performs scale-invariant value function updates in order to address differences in returns across environments, showing improvements in the multi-task setting while still using unitary scalarization. PopArt does not require per-task gradients but introduces additional hyperparameters. In our work, we address the differences in rewards by normalizing them at the replay buffer level. However, we believe both unitary scalarization and SMTOs might equally benefit from PopArt.

## 3 Multi-Task Learning Optimizers

We will now describe the deep MTL training problem and popular algorithms employed for its solution. Let $( X , Y ) \in \mathbb { R } ^ { d \times n } \times \mathbb { R } ^ { o \times n }$ be the training set, composed of n d-dimensional points and o-dimensional labels. In addition, $\mathcal { L } _ { i } : \mathbb { R } ^ { o \times n } \times \mathbb { R } ^ { o \times n } $ R denotes the loss for the i-th task, $\pmb \theta \in \mathbb { R } ^ { S }$ the parameter space, $\mathcal { T } : = \{ 1 , \dots , m \}$ the set of m tasks. The goal of MTL is to learn a single (generally task-aware) parametrized model $f : \mathbb { R } ^ { S } \times \mathbb { R } ^ { d \times n } \times \mathcal { T }  \mathbb { R } ^ { o \times n }$ that performs well on all tasks $\tau .$ . The parameter space is often split into a set of shared parameters across tasks (generally the majority of the architecture), denoted $\theta _ { \parallel }$ , and (possibly empty) task-specific parameters, denoted $\pmb { \theta } _ { \bot }$ so that $\pmb { \theta } : = [ \pmb { \theta } _ { | | } , \pmb { \theta } _ { \perp } ] ^ { T }$ . In this context, the model f often takes on an encoder-decoder architecture, where the encoder g learns a shared representation across tasks, and the decoders $h _ { i }$ are task-specific predictive heads: $\begin{array} { r } { \dot { f } ( \pmb { \theta } , \boldsymbol { X } , i ) = h _ { i } ( g ( \pmb { \dot { \theta _ { \parallel } } } , \boldsymbol { X } ) , \pmb { \theta } _ { \bot } ) } \end{array}$ . In this case, we denote by $\mathbf { z } = g ( \pmb { \theta } _ { | | } , X ) \in \mathbf { \bar { \mathbb { R } } } ^ { r \times n }$ the r-dimensional shared representation of X.

The training problem for MTL is typically formulated as the sum of the per-task losses [11, 54, 66]:

$$
\min _ {\boldsymbol {\theta}} \left[ \mathcal {L} ^ {\mathrm{MT}} (\boldsymbol {\theta}) := \sum_ {i \in \mathcal {T}} \mathcal {L} _ {i} (f (\boldsymbol {\theta}, X, i), Y) \right]. \tag {1}
$$

Unitary Scalarization The obvious way to minimize the multi-task training objective in equation (1) is to rely on a standard gradient-based algorithm. While, for simplicity, we focus on standard gradient descent rather than mini-batch stochastic gradient descent, the notation can be adapted by replacing the dataset size n by the mini-batch size b. Equation (1) corresponds to a linear scalarization with unitary weights under a multi-objective interpretation of MTL; hence, we call the direct application of gradient descent on equation (1) unitary scalarization. For vanilla gradient descent, this corresponds to taking a step in the opposite direction as the one given by the sum of per-task gradients: $\begin{array} { r } { \nabla _ { \theta } \mathcal { L } ^ { \mathrm { \hat { M } T } } = \sum _ { i \in \mathcal { T } } \breve { \nabla _ { \theta } } \mathcal { L } _ { i } } \end{array}$ . Per-task gradients are not required, as it suffices to directly compute the gradient of the sum $\mathcal { L } ^ { \mathrm { M I } }$ . Hence, when relying on deep learning frameworks based on reverse-mode differentiation, such as PyTorch [50], the backward pass is performed once per iteration (rather than m times). Furthermore, the memory cost is a factor $m$ less than most SMTOs, which require access to each $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \boldsymbol { \imath } }$ . As a consequence, unitary scalarization is simple, fast, and memory efficient. Our experiments demonstrate that, when possibly coupled with single-task regularization such as early stopping, $\ell _ { 2 }$ penalty or dropout layers [56], this simple optimizer is strongly competitive with SMTOs.

MGDA Sener and Koltun [54] point out that equation (1) can be cast as a multi-objective optimization problem with the following objective: $\bar { \mathcal { L } } ^ { \mathrm { M T } } ( \pmb { \theta } ) : = [ \mathcal { L } _ { 1 } ( \pmb { \theta } ) , \dots , \mathcal { L } _ { m } ( \pmb { \theta } ) ] ^ { T }$ . A commonly employed solution concept in multi-objective optimization is Pareto optimality. A point $\pmb { \theta } ^ { * }$ is called Pareto-optimal if, for any another point $\mathbf { \hat { \theta } } ^ { \dagger }$ such that $\exists i \in T : \bar { \mathcal { L } } _ { i } ( \pmb { \theta } ^ { \dagger } ) ^ { \cdot } < \bar { \mathcal { L } } _ { i } ( \pmb { \theta } ^ { * } )$ , then $\exists j \in \mathcal { T } : \mathcal { L } _ { j } ( \pmb { \theta } ^ { \dag } ) > \mathcal { L } _ { j } ( \pmb { \theta } ^ { * } )$ . A necessary condition for Pareto optimality at a point is Pareto stationarity, defined as the lack of a shared descent direction across all losses at that point. Sener and Koltun [54] rely on Multiple-Gradient Descent Algorithm (MGDA) [14] to reach a Pareto-stationary point for shared parameters $\theta _ { \parallel }$ . Intuitively, MGDA proceeds by repeatedly stepping in a shared descent direction [14, 17], which can be found by solving the following optimization problem:

$$
\min _ {\mathbf {g}, \epsilon} \left[ \epsilon + 1 / 2 \| \mathbf {g} \| _ {2} ^ {2} \right] \quad \text {s.t.} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} ^ {T} \mathbf {g} \leq \epsilon \quad \forall i \in \mathcal {T}, \tag {2}
$$

whose dual takes the following form (corresponding to the formulation from Désidéri [14]):

$$
\max _ {\boldsymbol {\alpha} \geq 0} - 1 / 2 \| \mathbf {g} \| _ {2} ^ {2} \quad \text {s.t.} \sum_ {i} \alpha_ {i} \nabla_ {\boldsymbol {\theta} _ {\|}} \mathcal {L} _ {i} = - \mathbf {g}, \quad \sum_ {i \in \mathcal {T}} \alpha_ {i} = 1. \tag {3}
$$

In other words, MGDA takes a step in a direction g given by the negative convex combination of per-task gradients, whose coefficients are given by solving equation (3). In practice, per-task gradients are rescaled before applying MGDA: the original authors’ implementation [54] relies on $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \gets \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \Big / \Big \| \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \Big \| \mathcal { L } _ { i } ( \pmb { \theta } )$ . The convergence of MGDA to a Pareto-stationary point is still guaranteed after normalization [14].

IMTL Impartial Multi-Task Learning (IMTL) [42] is presented as an SMTO that is not biased against any single task. It is composed of two complementary algorithmic blocks: IMTL-L, acting on task losses, and IMTL-G, acting on per-task gradients. IMTL-G follows the intuition that a multi-task optimizer should proceed along a direction $\begin{array} { r } { \mathbf { g } = - \sum _ { i } \alpha _ { i } \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } } \end{array}$ that equally represents per-task gradients. This is formulated analytically by requiring that the cosine similarity between g and each $\bar { \nabla } _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ be the same. To prevent the resulting problem from being underdetermined, Liu et al. [42] add the constraint $\textstyle \sum _ { i \in { \mathcal { T } } } \alpha _ { i } = 1$ , resulting in a problem that admits a closed-form solution for g:

$$
\mathbf {g} ^ {T} \frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1}}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \right\|} = \mathbf {g} ^ {T} \frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|} \forall i \in \mathcal {T} \backslash \{1 \}, \quad \mathbf {g} = - \sum_ {i} \alpha_ {i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}, \quad \sum_ {i \in \mathcal {T}} \alpha_ {i} = 1. \tag {4}
$$

IMTL-L, instead, aims to reweight task losses so that they are all constant over time, and equal to 1. In order to limit oscillations of the scaling factors, the authors propose to learn them jointly with the network by minimizing a common objective via gradient descent. In particular, given $s _ { i } \in \mathbb { R } \forall i \in \mathcal { T }$ , Liu et al. [42] derive the following form for the joint minimization problem: min<sub>s,</sub> $\theta \left[ \sum _ { i } \left( e ^ { s _ { i } } \mathcal { L } _ { i } ( \theta ) - s _ { i } \right) \right]$ . As proved by Liu et al. [42], IMTL-L only has a rescaling effect on the update direction of IMTL-G. Unlike IMTL-G and the other SMTOs presented in this section, IMTL-L rescaling is designed to affect the updates for task-specific parameters $\theta _ { \perp }$ as well.

PCGrad Let us write cos(x, z) for the cosine similarity between vectors x and z. Yu et al. [66] postulate that multi-task convergence is severely slowed down if the following three conditions (named the tragic triad) hold at once: (i) conflicting gradient directions: cos $( \nabla _ { \pmb { \theta } _ { | | } } ^ { - } \mathcal { L } _ { i } , \nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { j } ) < 0$ for some $i , j \in \mathcal { T } ; ( \operatorname { i i } )$ differing gradient magnitudes: $\lVert \nabla _ { { \pmb \theta } _ { \parallel } } \mathcal { L } _ { i } \rVert \gg \lVert \nabla _ { { \pmb \theta } _ { \parallel } } \mathcal { L } _ { j } \rVert$ for some $i , j \in \mathcal { T }$ ; and (iii) the unitary scalarization $\mathcal { L } ^ { \mathrm { M T } }$ has high curvature along $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { \mathrm { M T } }$ . The PCGrad [66] SMTO is presented as a solution to the tragic triad, targeted at the first condition. Consistent with the previous sections, let us denote the update direction by $\mathbf { g } .$ . Furthermore, let $[ \mathbf { x } ] _ { + } : = \mathrm { m a x } ( \mathbf { x } , \mathbf { 0 } )$ . Given per-task gradients $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ , PCGrad iteratively projects each task gradient onto the normal plane of all the gradients with which it conflicts:

$$
\left[ \mathbf {g} _ {i} \leftarrow \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}, \quad \mathbf {g} _ {i} \leftarrow \mathbf {g} _ {i} + \left[ \frac {- \mathbf {g} _ {i} ^ {T} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} (\mathbf {x})}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\| ^ {2}} \right] _ {+} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \forall j \in \mathcal {T} \backslash \{i \} \right] \forall i \in \mathcal {T}, \quad \mathbf {g} = - \sum_ {i \in \mathcal {T}} \mathbf {g} _ {i}, \tag {5}
$$

where the iterative updates of g<sub>i</sub> with respect to $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { j }$ are performed in random order.

GradDrop Chen et al. [11] focus on conflicting signs across task gradient entries, arguing that such conflicts lead to gradient “tug-of-wars". The GradDrop SMTO [11], presented as a solution to this problem, proposes to randomly mask per-task gradients $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ so as to minimize such conflicts. Specifically, GradDrop computes the “positive sign purity" $p _ { j }$ for the task gradient’s j-th entry and then masks the j-th entry of each per-task gradient with probability increasing with $p _ { j }$ , if the entry is negative, or decreasing with $p _ { j }$ , if the entry is positive. Let us write $\mathbf { p } : = [ p _ { 1 } , \hdots , p _ { S } ] ,$ , where S is the dimensionality of the parameter space (see §3),  for the Hadamard product and $\mathbb { 1 } _ { \mathbf { a } }$ for the indicator vector on condition a. Given a vector $\mathbf { u } _ { i }$ , uniformly sampled in [0, 1] at each iteration, GradDrop takes a step in the direction given by:

$$
\mathbf {g} = \sum_ {i \in \mathcal {T}} \left( \begin{array}{l} - \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \odot \mathbb {1} _ {\left(\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} > 0\right)} \odot \mathbb {1} _ {\left(\mathbf {u} _ {i} > \mathbf {p}\right)} \\ - \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \odot \mathbb {1} _ {\left(\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} <   0\right)} \odot \mathbb {1} _ {\left(\mathbf {u} _ {i} <   \mathbf {p}\right)} \end{array} \right), \text {with} \quad \mathbf {p} = \frac {1}{2} \left(1 + \frac {\sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}}{\sum_ {i \in \mathcal {T}} | \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} |}\right). \tag {6}
$$

## 4 Experimental Evaluation

Relying on a unified experimental pipeline, we present an empirical evaluation on common MTL benchmarks of unitary scalarization (§3), of the popular SMTOs presented in §3, and of the recent RLW algorithms [40] due to their similarities with PCGrad and GradDrop (see §5.2). We benchmark against the two RLW instances that showed the best average performance in the original paper: RLW with weights sampled from a Dirichlet distribution (“RLW Diri.”), and RLW with weights sampled from a Normal distribution (“RLW Norm.”). The goal of this section is to assess the efficacy of a popular line of previous work, focusing on a few representative or well-established optimizers. Therefore, we forego comparison with more recent SMTOs [28, 41, 48]. Nevertheless, we point out that these algorithms often lack significant enough improvements over the optimizers we consider, or may have substantial commonalities with them (see §5.2 for Nash-MTL [48], which was published concurrently to the finalization of this work). Whenever appropriate, we employ “Unit. Scal.” as shorthand for unitary scalarization. We first present supervised learning experiments (§4.1), and then evaluate on a popular reinforcement learning benchmark (§4.2).

![](images/8d352fa899bbf50aaa73f65134cbe6146b5daa64f5734d42c819a44ae8b47620.jpg)

<details>
<summary>error_bar</summary>

| Method | Median Accuracy | Min Accuracy | Max Accuracy |
| :--- | :--- | :--- | :--- |
| Unit. Scal. | ~0.948 | ~0.943 | ~0.952 |
| IMTL | ~0.949 | ~0.946 | ~0.951 |
| MGDA | ~0.949 | ~0.946 | ~0.950 |
| GradDrop | ~0.935 | ~0.922 | ~0.948 |
| PCGrad | ~0.948 | ~0.944 | ~0.951 |
| RLW Diri. | ~0.943 | ~0.940 | ~0.946 |
| RLW Norm. | ~0.931 | ~0.931 | ~0.949 |
</details>

(a) Avg. task test accuracy: mean and 95% CI (10 runs).

![](images/ef7ea201015aaabb86448279450c8f619331e6856e770af08f4c9637e308ebe6.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 | Min | Max |
| --- | --- | --- | --- | --- | --- |
| Unit. Scal. | ~3.50 | ~3.58 | ~3.62 | ~3.45 | ~3.70 |
| IMTL | ~3.72 | ~3.85 | ~4.00 | ~3.52 | ~4.15 |
| MGDA | ~3.50 | ~3.58 | ~3.62 | ~3.45 | ~3.75 |
| GradDrop | ~3.50 | ~3.55 | ~3.60 | ~3.48 | ~3.65 |
| PCGrad | ~3.80 | ~3.88 | ~3.92 | ~3.75 | ~4.00 |
| RLW Diri. | ~3.78 | ~3.90 | ~4.00 | ~3.60 | ~4.30 |
| RLW Norm. | ~3.90 | ~4.15 | ~4.22 | ~3.82 | ~4.48 |
</details>

(b) Box plots for the training time of an epoch (10 runs).  
Figure 1: No algorithm outperforms unitary scalarization on the Multi-MNIST dataset.

Our experiments indicate that the performance of unitary scalarization has been consistently underestimated in the literature. By showing the variability between runs and by relying on standard regularization and stabilization techniques from the single-task literature, we demonstrate that no SMTO consistently outperforms unitary scalarization across the considered settings. This result holds in spite of the added complexity and computational overhead associated with most SMTOs. We provide a potential explanation of our results in §5.

## 4.1 Supervised Learning

All the architectures employed in the supervised learning experiments conform to the encoder-decoder structure detailed in §3. Whenever suggested by the original authors for this context, the SMTO implementations rely on per-task gradients with respect to the last shared activation, $\nabla _ { \mathbf { z } } ,$ rather than on the usually more expensive $\nabla _ { \pmb { \theta } } \mathcal { L } _ { i }$ . In particular, this is the case for MGDA, IMTL and GradDrop. See appendix B for details concerning each individual algorithm. Surprisingly, several MTL works [11, 40, 42, 66] report validation results, making it easier to overfit. Instead, following standard machine learning practice, we select a model on the validation set, and later report test metrics for all benchmarks. Validation results are also available in appendix D. Appendix C.1 reports dataset descriptions, the computational setup, hyperparameter and tuning details.

## 4.1.1 Multi-MNIST

We present results on the Multi-MNIST [54] dataset, a simple two-task supervised learning benchmark. We employ a popular architecture from previous work [54, 66] (see appendix C.1), where a single dropout layer [56] (with dropout probability 0.5) is employed in both the encoder and the decoder. $\ell _ { 2 }$ regularization did not improve validation performance and was therefore omitted. Figure 1 reports the average task test accuracy, and the training time per epoch. For each run, the test model was selected as the model with the largest average task validation accuracy across the training epochs. Appendix D presents the results of Figure 1 in tabular form, as well as the average task validation accuracy per epoch. As seen from the overlapping confidence intervals, none of the considered algo rithms clearly outperforms the others. However, GradDrop displays higher experimental variability. Finally, Figure 1(b) shows that unitary scalarization also has among the lowest training times.

## 4.1.2 CelebA

We now show results for the CelebA [44] dataset, a challenging 40-task multi-label classification prob lem. We employ the same architecture as many previous studies [40, 42, 54, 66] (see appendix C.1). We tuned $\ell _ { 2 }$ regularization terms λ for all SMTOs in the following grid: $\lambda \in \{ 0 , 1 0 ^ { - 4 } , \bar { 1 } 0 ^ { - 3 } \}$ . The best validation performance was attained with $\lambda = 1 0 ^ { - 3 }$ for unitary scalarization, IMTL and PCGrad, and with $\lambda = \mathrm { \dot { 1 0 } ^ { - 4 } }$ for MGDA, GradDrop, and RLW. Validation performance was further stabilized by the addition of several dropout layers (see Figure 5), with dropout probabilities from 0.25 to 0.5. We present an ablation study on the effect of regularization on this experiment in §5.1. Figure 10 (appendix D.2) shows that regularization improves the peak average validation performance for all the considered methods. Analogously to our Multi-MNIST results, Figure 2 plots the distribution of the training time per epoch, and the average test task accuracy. As with Multi-MNIST, the test model for each run was the one with maximal average validation task accuracy across epochs. In other words, if the peak is attained before the last epoch, we perform early stopping: as shown in Figure 8(a) in appendix D this is the case for most methods. Due to the large number of tasks, Figure 2(b) shows relatively large runtime differences across methods. PCGrad is the slowest (roughly 35 times slower than unitary scalarization). In fact, amongst the considered algorithms, it is the only one that computes per-task gradients over the parameters $\bar { ( \nabla _ { \pmb { \theta } } \mathcal { L } _ { i } \forall i \in \mathcal { T } ) }$ at each iteration. GradDrop, MGDA and IMTL have overhead factors (compared to unitary scalarization) ranging from roughly 1.05 to 2.4 due to the relatively small size of z for the employed architecture. The overhead of RLW is negligible: roughly 5%. Nevertheless, due to largely overlapping confidence intervals in Figure 2(a), none of the methods consistently outperforms unitary scalarization. In fact, owing to our adoption of explicit regularization techniques (see §5.1) its average performance is superior to that reported in the literature [42, 54].

![](images/8a9cfcde0b12ff3e2d4de0d54905e46e2416dd189874a0bfe027bcc4f7f0e039.jpg)

<details>
<summary>error_bar</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.9082 | ~0.9090 | ~0.9099 |
| IMTL | ~0.9085 | ~0.9091 | ~0.9101 |
| MGDA | ~0.9014 | ~0.9021 | ~0.9033 |
| GradDrop | ~0.9094 | ~0.9098 | ~0.9101 |
| PCGrad | ~0.9081 | ~0.9089 | ~0.9102 |
| RLW Dirl. | ~0.9091 | ~0.9097 | ~0.9104 |
| RLW Norm. | ~0.9084 | ~0.9088 | ~0.9103 |
</details>

(a) Avg. task test accuracy: mean and 95% CI (3 runs).

![](images/d771125ae76b241167d6e6671e578fef1cafebedc058902b628666b126ac9a2d.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 (s) | Q2 (Median) (s) | Q3 (s) |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.5 | ~0.6 | ~0.8 |
| IMTL | ~1.5 | ~2.0 | ~2.5 |
| MGDA | ~600 | ~700 | ~800 |
| GradDrop | ~0.5 | ~0.6 | ~0.8 |
| PCGrad | ~9000 | ~10000 | ~11000 |
| RLW Diri. | ~0.5 | ~0.6 | ~0.8 |
| RLW Norm. | ~0.5 | ~0.6 | ~0.8 |
</details>

(b) Box plots for the training time of an epoch (10 runs).  
Figure 2: While SMTOs display larger runtimes, none of them outperforms the unitary scalarization on the CelebA dataset.

![](images/321ae4f2d8d842476fa340fc0783d539bfca9af4de3c057696f0fab0c969dc32.jpg)

<details>
<summary>error_bar</summary>

| Category | Min | Q1 | Median | Q3 | Max |
| --- | --- | --- | --- | --- | --- |
| Unit_Scal | ~0.0127 | ~0.0127 | ~0.0131 | ~0.0135 | ~0.0135 |
| IMTL | ~0.0125 | ~0.0125 | ~0.0128 | ~0.0131 | ~0.0131 |
| MGDA | ~0.0139 | ~0.0139 | ~0.0142 | ~0.0144 | ~0.0144 |
| GradDrop | ~0.0127 | ~0.0127 | ~0.0129 | ~0.0132 | ~0.0132 |
| PCGrad | ~0.0128 | ~0.0128 | ~0.0132 | ~0.0134 | ~0.0134 |
| RLW_Dirl | ~0.0126 | ~0.0126 | ~0.0130 | ~0.0135 | ~0.0135 |
| RLW_Norm | ~0.0125 | ~0.0125 | ~0.0130 | ~0.0136 | ~0.0136 |
</details>

(a) Absolute depth test error: lower is better.

![](images/261c08f75e8869f64dd612a011c082f06f236410c4115278a027420a05644530.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit_Scal. | ~42 | ~46 | ~50 |
| IMTL | ~38 | ~43 | ~48 |
| MGDA | ~33 | ~48 | ~62 |
| GradDrop | ~39 | ~47 | ~54 |
| PCGrad | ~44 | ~48 | ~52 |
| RLW_Diri. | ~46 | ~49 | ~51 |
| RLW_Norm. | ~44 | ~46 | ~50 |
</details>

(b) Relative depth test error: lower is better.

![](images/7ecd8a824a3fbf5b626c987cec67bcaa91d31d973175858e8505e6f79aecd8b5.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.702 | ~0.703 | ~0.704 |
| IMTL | ~0.686 | ~0.695 | ~0.704 |
| MGDA | ~0.697 | ~0.703 | ~0.703 |
| GradDrop | ~0.699 | ~0.703 | ~0.706 |
| PCGrad | ~0.699 | ~0.705 | ~0.711 |
| RLW Diri. | ~0.702 | ~0.704 | ~0.706 |
| RLW Norm. | ~0.696 | ~0.701 | ~0.705 |
</details>

(c) Test segmentation mIOU: higher is better.

![](images/8b195769779d723734efebe4addf2a6ff1388a5560735fecfca791fe3997170a.jpg)

<details>
<summary>error_bar</summary>

| Method | Min | Median | Max |
| --- | --- | --- | --- |
| Unit_Scal. | ~0.9187 | ~0.9203 | ~0.9235 |
| IMTL | ~0.9154 | ~0.9168 | ~0.9184 |
| MGDA | ~0.9185 | ~0.9189 | ~0.9194 |
| GradDrop | ~0.9180 | ~0.9194 | ~0.9205 |
| PCGrad | ~0.9187 | ~0.9201 | ~0.9207 |
| BLW Diri. | ~0.9185 | ~0.9199 | ~0.9207 |
| BLW Norm. | ~0.9185 | ~0.9191 | ~0.9198 |
</details>

(d) Test segmentation accuracy: higher is better.

![](images/32a069b1a00af25afac91a2457ee9bdef6b328401fd3eb265de9f94ee9c2e03f.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~320 | ~325 | ~330 |
| IMTL | ~740 | ~745 | ~750 |
| MGDA | ~730 | ~735 | ~740 |
| GradDrop | ~520 | ~525 | ~530 |
| PCGrad | ~420 | ~425 | ~430 |
| RLW Diri. | ~310 | ~315 | ~320 |
| RLW Norm. | ~310 | ~315 | ~320 |
</details>

(e) Box plots for the training time of an epoch (10 runs).  
Figure 3: On Cityscapes, none of the SMTOs outperforms unitary scalarization, which proves to be the most cost-effective algorithm. Subfigures (a)-(d) report means for three runs, and their 95% CIs.

## 4.1.3 Cityscapes

In order to complement the multi-task classification experiments for Multi-MNIST and CelebA, we present results for Cityscapes [13], a dataset for semantic understanding of urban street scenes. We rely on a common encoder architecture from the literature [40, 42] (see appendix C.1), with a single dropout layer in the task-specific heads [40]. As for CelebA, unitary scalarization, IMTL, and PCGrad benefit from more regularization than the other optimizers: we employ $\lambda = 1 0 ^ { - 5 }$ for these three algorithms, as it resulted in better validation performance on the majority of metrics, and λ = 0 for the remaining methods. Cityscapes is a heterogeneous MTL problem: it contains tasks of different types whose validation metrics cannot be averaged to perform model selection. Considering the lack of an established procedure in this context, we potentially evaluate a different model for each metric, chosen as the one with the best (maximal or minimal, depending on the metric) validation performance across epochs (we perform per-run early stopping). This procedure maximizes per-task performance, at the cost of increased inference time. If inference time is a priority, an alternative model selection procedure could rely on relative task improvement [28, 41, 48], assuming that per-metric improvements are to be weighted linearly. Nevertheless, any consistently applied model selection scheme serves the main goal of our work: evaluating all SMTOs on a fair ground. Figure 3 shows test results for two metrics per task, and the distribution of the training time per epoch. As with Multi-MNIST and CelebA, no training algorithm clearly outperforms unitary scalarization (significant overlaps across confidence intervals exist), which is again the least expensive method. In contrast with a popular hypothesis [10, 30, 42], this holds in spite of relatively large loss imbalances. In fact, the loss for the depth task is roughly 10 times smaller than that of the segmentation task: see figures $1 7 ( \mathrm { f } ) { - } 1 7 ( \mathrm { g } )$ . Unlike CelebA (see Figure 2(b)), IMTL, MGDA and GradDrop are significantly slower than unitary scalarization (factors from 1.6 to 2.3), due to the relatively (compared to the parameter space) large size of z in the employed architecture. PCGrad, instead, appears to be less expensive (30% more than the baseline), demonstrating the benefits of working on $\nabla _ { \pmb { \theta } } \mathcal { L } _ { i }$ on this model.

![](images/8ad1dc75ba0fdb89c034edd1fdb20b9ecea1a127260742bc06ec502c8e6059a6.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 | IQR |
| --- | --- | --- | --- | --- |
| Unit. Scal. | ~0.825 | ~0.875 | ~0.92 | ~0.095 |
| MGDA | ~0.845 | ~0.90 | ~0.96 | ~0.025 |
| GradDrop | ~0.84 | ~0.89 | ~0.945 | ~0.085 |
| PCGrad | ~0.815 | ~0.855 | ~0.89 | ~0.075 |
| RLW Diri. | ~0.835 | ~0.88 | ~0.925 | ~0.09 |
| RLW Norm. | ~0.86 | ~0.91 | ~0.965 | ~0.095 |
</details>

(a) MT10 (10 repetitions).

![](images/ecc418792b980bb9fa4f9f677dbe794946af227f38e0c749ec2195d141e6d1e3.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.605 | ~0.64 | ~0.67 |
| MGDA | ~0.41 | ~0.47 | ~0.53 |
| GradDrop | ~0.53 | ~0.54 | ~0.55 |
| PCGrad | ~0.64 | ~0.67 | ~0.70 |
| RLW Diri. | ~0.56 | ~0.59 | ~0.62 |
| RLW Norm. | ~0.57 | ~0.59 | ~0.61 |
</details>

(b) MT50 (10 repetitions).

![](images/fbab1db584b6a607b6ddbca42b0cd93100769a9708867ad45a31af542ac179bb.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~300 | ~350 | ~400 |
| IMTL | ~1800 | ~1900 | ~2000 |
| MGDA | ~1900 | ~2000 | ~2100 |
| GradDrop | ~1800 | ~1900 | ~2000 |
| PCGrad | ~1200 | ~1300 | ~1400 |
| RLW Diri. | ~400 | ~400 | ~400 |
| RLW Norm. | ~400 | ~400 | ~400 |
</details>

(c) MT10 (10 repetitions).

![](images/ffba402184a89134e99c6f68f327fcb491f98e3191e6cc605a6fbc3de96142a8.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit.Scal. | ~150 | ~200 | ~250 |
| IMTL | ~2500 | ~2600 | ~2700 |
| MGDA | ~3000 | ~3100 | ~3200 |
| GradDrop | ~3100 | ~3200 | ~3300 |
| PCGrad | ~2400 | ~2500 | ~2600 |
| RLW Diri. | ~250 | ~280 | ~300 |
| RLW Norm. | ~250 | ~260 | ~270 |
</details>

(d) MT50 (10 repetitions).  
Figure 4: On Metaworld, none of the SMTOs significantly outperforms Unit. Scal., which is the least expensive method. Subfigures (a)-(b) report mean and 95% CI for the best (over the updates) average success rate. Subfigures (c)-(d) show box plots for the training time of 10,000 updates.

## 4.2 Reinforcement Learning

For RL experiments, we use Meta-World [65] and the Soft Actor-Critic [20] implementation from [55]. Unlike §4.1, the employed network architecture (see appendix C.1) is fully shared across tasks. Therefore, all SMTO implementations for these experiments rely on per-task gradients with respect to network parameters $\bar { \nabla } _ { \pmb { \theta } } \mathcal { L } _ { i }$ (see §5). Among the SMTOs we consider, PCGrad is the only one developed with the RL setting in mind. For fairness and completeness, we add all the other SMTOs from the supervised learning experiments, and are the first to test these optimizers in the RL setting. To stabilize learning, we increase the replay buffer size, a well known technique in single-task RL, add actor $l _ { 2 }$ regularization, and modify the reward normalization employed by Sodhani et al. [55]. The unitary scalarization performance reported by Yu et al. [66] is considerably lower than that of Sodhani et al. [55], which we believe is due to the lack of reward normalization in the former. Sodhani et al. [55] keep a moving average of rewards in the environment, with a hyperparameter controlling the speed of the moving average. As we show in Figure 16, the learning algorithm is sensitive to that hyperparameter. Moreover, such normalization might make similar transitions have drastically different rewards stored in the replay buffer. To alleviate these issues, we store the raw rewards in the buffer, and normalize only when a mini-batch is sampled.

Figure 4 reports the best average success rate across the updates and the runtime for 10,000 updates. In addition to these summary statistics, reported for consistency with §4.1, the learning curves are shown in appendix E. Our MT10 (10 tasks) results in Figure 4(a) show that by stabilizing the baseline using standard RL techniques, unitary scalarization performs on par with other SMTOs, mirroring our findings in §4.1. This is in contrast with the previous literature, which reported that PCGrad outperforms unitary scalarization [55, 66]. Figure 4(b) presents results on MT50 (50 tasks): similarly to MT10, none of the SMTOs significantly outperforms unitary scalarization, with PCGrad’s average being slightly above unitary scalarization. We speculate that the stochastic loss rescaling performed by PCGrad (see Proposition 3) reduces the differences in task return scales, and expect that methods like PopArt [60] would have a similar effect without requiring access to per-task gradients. While we did not tune hyperparameters for MT50 (we employed those found for MT10), it would be much easier to do that for unitary scalarization due to its lower runtime. In fact, Figure 4(d) shows that a single unitary scalarization run takes roughly 15 hours, whereas PCGrad, MGDA and GradDrop require more than a week. Similarly to MT10, actor regularization pushes the average performance of unitary scalarization higher (see in appendix E.2). Overall, as in the supervised learning setting, unitary scalarization performs comparably to SMTOs despite being simpler and less demanding in both memory and compute. IMTL was unstable on this RL benchmark and all of the runs crashed due to numerical overflow. We hence omit IMTL results from the main body of the paper and show its results in Figure 13 in appendix E, which also describes a possible explanation. We hypothesize that the instability of IMTL is due to lack of bounds on scaling coefficients. See appendix C.2 for hyperparameter settings and ablation studies.

## 5 Regularization in Specialized Multi-Task Optimizers

The empirical results presented in §4 motivate the need to carefully analyze existing SMTOs. We make an initial attempt in this direction by viewing their effects through the lens of regularization.

Let us define a regularizer as a technique to reduce overfitting [15]. We first show that the SMTOs considered in §4 empirically act as regularizers via an ablation study (§5.1). We then take a closer look at their behavior, presenting technical results that support their alternative interpretation as regularizers (§5.2). Finally, $\ S 5 . 3$ provides additional empirical backing for some of the technical results. Unless otherwise stated, we assume that MTL methods apply only to $\pmb { \theta } _ { | | }$ and that standard gradient-based updates are employed for tasks-specific parameters $\dot { \pmb { \theta } } _ { \perp }$ . We furthermore adopt the following shorthands: $\mathcal { L } _ { i } ( \pmb { \theta } )$ for $\dot { \mathcal { L } _ { i } } ( f ( \pmb { \theta } , X , i ) , \dot { Y } )$ ), and $\bar { \nabla } _ { \pmb { \theta } } \mathcal { L } _ { i }$ for $\nabla _ { \pmb { \theta } } \mathcal { L } _ { i } ( f ( \pmb { \theta } , X , i ) , Y )$

## 5.1 Ablation Study

We repeat the experiment from §4.1.2 and remove explicit regularization: no dropout layers are added to the encoder-decoder architecture, and $\lambda = 0$ for all optimizers. In addition, we examine the behavior of two different \` -regularized instances of unitary scalarization: $\lambda = 1 0 ^ { - 4 }$ for “Unit. Scal. $\ell _ { 2 } { } ^ { , 9 } , \lambda = 2 \times 1 0 ^ { - 3 }$ for “Unit. Scal. $\ell _ { 2 } + \mathbf { \vec { \mu } } ^ { \mathbf { > } }$ . Figure 5 shows that SMTOs behave similarly to an $\ell _ { 2 ^ { - } }$ penalized unitary scalarization. Importantly, SMTOs delay overfitting, requiring less early stopping compared to unitary scalarization to obtain comparable performance. In other words, early stopping is sufficient for unitary scalarization to perform on par with SMTOs. Moreover, overfitting is further reduced by “Unit. Scal. $\mathrm { R e g . } ^ { \mathrm { , 5 } } .$ , which plots the regularized unitary scalarization from §4.1.2, with dropout layers and a weight decay of $\lambda = 1 0 ^ { - 3 }$ . Finally, Figure $9 ( \mathrm { a } )$ shows that unregularized unitary scalarization and most SMTOs rapidly drive the training loss of each task towards its global optimum. This suggests that the main difficulty of MTL is not associated with the optimization of its training objective, but rather to incorporating adequate regularization. Additional results are presented in appendix D.2.

## 5.2 Technical Results

All the methods considered in §5.1 regularize more than unitary scalarization. While RLW was shown to reduce overfitting by the original authors [40, theorem 2], we now provide a collection of novel and existing technical results that potentially explain the regularizing behavior of each of the other algorithms, complementing the presentation from $\ S 3$ . In particular, we show that MGDA, IMTL and PCGrad have a larger convergence set than unitary scalarization, reducing the chances to land on sharp local minima [15]. Furthermore, GradDrop and PCGrad introduce significant stochasticity, which is often linked to the same effect [31, 34]. We hope these observations will steer further research.

MGDA Let us denote the convex hull of a set A by $\operatorname { C o n v } ( \mathcal { A } )$ . We now recall a well-known property of MGDA [14] and relate it to the behavior of unitary scalarization.

Proposition 1. The MGDA SMTO [54] converges to a superset of the convergence points of unitary scalarization. More specifically, it converges to any point $\theta _ { \parallel } ^ { * }$ such that: $\mathbf { 0 } \in C o \mathrm { { n } } \nu ( \{ \tilde { \nabla _ { \pmb { \theta } _ { \parallel } ^ { * } } } \mathcal { L } _ { i } | i \in \mathcal { T } \} )$

See appendix B.1 for a simple proof. As a consequence of Proposition 1, MGDA does not necessarily reach a stationary point for $\dot { \mathcal { L } } ^ { \mathrm { M T } }$ (that is, a point for which $\begin{array} { r } { \sum _ { i \in \mathcal { T } } \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } = \mathbf { 0 } ) } \end{array}$ or for any of the losses $\mathcal { L } _ { i } \left( \nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i } = \mathbf { 0 } \right)$ . For example, any point $\theta _ { \parallel }$ for which two per-task gradients point in opposite directions is Pareto stationary. On account of the well-known [15] relationship between underoptimizing (e.g., early stopping [7, 39]) and overfitting, proposition 1 supports the interpretation of MGDA as a regularizer for equation (1). Empirical evidence that MGDA under-optimizes is provided in §5.3, Figure 9(a), and Figure 5, which shows over-regularization. Proposition 1 can be trivially extended to the recent Nash-MTL, which shares the same convergence set [48, Theorem 5.4].

![](images/fa13ce66b4a728ec7d9273a82943e6999254d811eb89b9ee9c9cf39cc903bae6.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. | Unit. Scal. t2 | Unit. Scal. t2 + | Unit. Scal. Reg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 | ~0.893 |
| 5 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 | ~0.912 |
| 10 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 | ~0.900 |
| 15 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 | ~0.899 |
| 20 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 | ~0.898 |
| 25 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 |
| 30 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 |
| 35 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 | ~0.901 |
| 40 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 |
| 45 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 |
| 50 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 | ~0.902 |
</details>

Figure 5: Mean and 95% CI (3 runs) avg. task validation accuracy over epochs on CelebA. SMTOs postpone the onset of overfitting, mirroring the effect of $\ell _ { 2 }$ regularization on unitary scalarization.

![](images/1d19c57d01ad9f9edcf54ccf648ca6198830f95f121eb83055fe7a9e727351c1.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit.Scal. | PCGrad | IMTL | RLW Dirl. | MGDA | GradDrop | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~12 | ~25 | ~12 | ~12 | ~12 | ~12 | ~12 |
| 5 | ~6 | ~13 | ~6 | ~6 | ~6 | ~6 | ~6 |
| 10 | ~8 | ~14 | ~8 | ~7 | ~8 | ~7 | ~7 |
| 15 | ~7 | ~16 | ~9 | ~8 | ~9 | ~8 | ~8 |
| 20 | ~6 | ~13 | ~9 | ~9 | ~9 | ~9 | ~9 |
| 25 | ~5 | ~10 | ~9 | ~9 | ~9 | ~9 | ~9 |
| 30 | ~4 | ~12 | ~9 | ~8 | ~9 | ~9 | ~8 |
| 35 | ~3 | ~7 | ~9 | ~7 | ~9 | ~8 | ~7 |
| 40 | ~2 | ~5 | ~9 | ~6 | ~9 | ~7 | ~6 |
| 45 | ~1.5 | ~3 | ~9 | ~4 | ~9 | ~5 | ~4 |
| 48 | ~0.7 | — | ~9 | — | — | — | — |
</details>

Figure 6: Mean and 95% CI (3 runs) for $\begin{array} { r } { \big \| \breve { \sum } _ { i \in \mathcal { T } } \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \big \| _ { 2 } } \end{array}$ on CelebA. MGDA and IMTL converge away from stationary points of unitary scalarization, indicating under-optimization.

IMTL We now show that aggregating per-task gradients so that their cosine similarity is the same (equation (4)) yields a constrained steepest-descent algorithm (Proposition 2). This view on the update step of IMTL leads to a novel analysis of its convergence points (corollary 1). Proofs can be found in appendix B.2. We will denote by $\operatorname { A f f } ( A )$ the affine hull of a set ${ \mathcal { A } } .$

Proposition 2. IMTL by Liu et al. [42] updates $\theta _ { \parallel }$ by taking a step in the steepest descent direction whose cosine similarity with per-task gradients is the same across tasks.

Corollary 1. IMTL by Liu et al. [42] converges to a superset of the Pareto-stationary points for $\theta _ { \parallel }$ (and hence ofthe convergence points ofthe unitary scalarization). More specifically, it converges to any point $\theta _ { \parallel } ^ { * }$ such that: $\mathbf { 0 } \in \bar { A } f f \left( \left\{ \nabla _ { \pmb { \theta } _ { \parallel } ^ { * } } \mathcal { L } _ { i } / \left\| \nabla _ { \pmb { \theta } _ { \parallel } ^ { * } } \mathcal { L } _ { i } \right\| | i \in T \right\} \right)$

As seen for MGDA, corollary 1 implies that, even if the employed model f has the capacity to reach the minimal loss on $\mathcal { L } ^ { \mathrm { M } \mathbf { \check { I } } }$ , IMTL may stop before reaching a stationary point. Recalling the relationship between under-optimizing and overfitting [15], this supports the interpretation of IMTL as a regularizer for equation (1). This is empirically shown in §5.3, Figures 5, 9(a). In particular, unitary scalarization reaches the same average performance of IMTL but requires earlier stopping.

PCGrad We provide an alternative characterization of the PCGrad update rule, highlighting its stochasticity in the context of its interpretation as loss rescaling [40, 42]. See appendix B.3 for a proof.

Proposition 3. PCGrad is equivalent to a dynamic, and possibly stochastic, loss rescaling for $\theta _ { \parallel }$ . At each iteration, per-task gradients are rescaled asfollows:

$$
\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \leftarrow \left(1 + \sum_ {j \in \mathcal {T} \backslash \{i \}} d _ {j i}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}, d _ {j i} \in \left[ 0, \frac {\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\|}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|} \right].
$$

Furthermore, $i f \left| \mathcal { T } \right| > 2 , d _ { j i }$ is a random variable, and the above range contains its support.

The results from proposition 3 can be easily extended to GradVac [62], which generalizes PCGrad’s projection onto the normal vector to arbitrary target cosine similarities between per-task gradients. When $| \mathcal T | > 2$ , PCGrad corresponds to a stochastic loss re-weighting. As such, PCGrad bears many similarities with Random Loss Weighting (RLW) [40]. RLW proposes to sample scalarization weights from standard probability distributions at each iteration, and proves that this leads the better general ization [40, theorem 2]. Indeed, it is well-known that adding noise to stochastic gradient estimations leads the optimization towards flatter minima, and that such minima may reduce overfitting [31, 34]. In line with the main technical results by Yu et al. [66], we now restrict our focus to two-task problems, which allow for an easy description of PCGrad’s convergence points. The result is largely based on [66, theorem 1]: we relax some of the assumptions and provide a proof in appendix B.3.

Corollary 2. $I f | { \mathcal { T } } | = 2 ,$ , PCGrad will stop at any point where co $\small 3 ( \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } ) = - 1$ . Furthermore, $i f { \mathcal { L } } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are differentiable, and $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { M T }$ is L-Lipschitz with $L > 0$ , PCGrad with step size $\begin{array} { r } { t < \frac { 1 } { L } } \end{array}$ converges to a superset ofthe convergence points ofthe unitary scalarization.

Corollary 2 implies that, when $| \tau | = 2 .$ , PCGrad may under-optimize equation (1) as MGDA and IMTL. In particular, if cos $( \mathrm { \overleftrightarrow { V } } _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } ) = - 1$ , then $\mathbf { 0 } \in \mathrm { C o n v } ( \{  { \nabla } _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } \} )$ (see proposition 1). We believe that PCGrad’s stochasticity and enlarged convergence set potentially explain its regularizing effect.

GradDrop While the motivation behind GradDrop is to avoid entry-wise gradient conflicts across tasks, the main property of the method is to drive the optimization towards “joint minima": points that are stationary for all the individual tasks at once [11, proposition 1]. In other words: $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i } = \mathbf { 0 } \forall i \in$ T. While this property is desirable, we show that it holds beyond GradDrop, and independently of the gradient directions. Under strong assumptions on the model capacity, the above property would trivially hold for unitary scalarization (proposition 5, appendix B.4). Proposition 4 shows that it holds for a simple randomized version of unitary scalarization, which we name Random Grad Drop (RGD).

Proposition 4. Let $\begin{array} { r } { \mathcal { L } ^ { R G D } ( \pmb { \theta } _ { \parallel } ) : = \sum _ { i \in \mathcal { T } } u _ { i } \mathcal { L } _ { i } ( \pmb { \theta } _ { \parallel } ) } \end{array}$ , where $u _ { i } \sim B e r n o u l l i ( p ) \forall i \in \mathcal { T }$ and $p \in ( 0 , 1 ]$ The gradient $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { R G D }$ is always zero if and only $i f \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } = \mathbf { 0 } \forall i \in \mathcal { T }$ . In other words, the result from [11, proposition 1] can be obtained without any information on the sign of per-task gradients.

Proposition 4 (see appendix B.4 for a simple proof) shows that an inexpensive sign-independent stochastic scalarization shares GradDrop’s main reported property. $\mathcal { L } ^ { \mathrm { { R G D } } }$ can be directly cast an instance of RLW, and hence as a regularization method [31, 34]. Furthermore, Figure 12 in appendix D.3 shows that the empirical results of GradDrop on CelebA [44] are closely matched by a sign-agnostic gradient masking, partly undermining the conflicting gradients assumption. We believe that the above results, along with the authors’ original experiments showing that GradDrop delays overfitting on CelebA [11, figure 3], suggest that GradDrop behaves as a regularizer.

## 5.3 Under-Optimization: Empirical Study

As seen in §5.2, MGDA and IMTL might under-optimize equation (1) compared to unitary scalarization due to their larger convergence sets. In order to assess whether this is empirically the case, we estimate $\begin{array} { r } { \big \| \sum _ { i \in \mathcal { T } } \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \big \| _ { 2 } , } \end{array}$ the norm of the unitary scalarization update on shared parameters $\theta _ { \parallel }$ for all optimizers throughout the unregularized CelebA experiment from §5.1. Large magnitudes for $\begin{array} { r } { \big \| \sum _ { i \in \mathcal { T } } \mathbf { \dot { V } } \pmb { \theta } _ { \parallel } \mathbf { \mathcal { L } } _ { i } \big \| _ { 2 } } \end{array}$ towards convergence would indicate that SMTOs steer optimization far from stationary points of unitary scalarization, resulting in under-optimization. We compute the update norm on the mini-batch loss every 100 updates, and report the per-epoch average in Figure 6. Compared with unitary scalarization, most SMTOs have smaller or comparable update magnitude in the first 15 epochs. However, towards convergence, SMTOs display larger $\begin{array} { r } { \big \| \sum _ { i \in \mathcal { T } } \nabla _ { \pmb { \theta } _ { \parallel } } \overline { { \mathcal { L } } } _ { i } \big \| _ { 2 } } \end{array}$ compared to unitary scalarization. In particular, IMTL and MGDA have the largest norm, denoting significant empirical under-optimization. The additional stochasticity of RLW, PCGrad, and GradDrop also appears to lead to larger norm values than unitary scalarization, yet to a lesser degree. Given that MGDA and IMTL incur a larger loss than unitary scalarization in later epochs (see Figure 9(a) in appendix D.2), we can conclude that they guide optimization towards regions of the parameter space that under-optimize equation (1), providing empirical support for our analysis.

## 6 Conclusions

This paper made two main contributions. First, we evaluated popular SMTOs using a single experimental pipeline, including previously unpublished results of MGDA, IMTL, RLW, and GradDrop in the RL setting. Surprisingly, our evaluation showed that none of the SMTOs consistently outperform unitary scalarization, the simplest and least expensive method. Second, in order to explain our surprising results, we postulate that SMTOs act as regularizers and present an analysis that supports our hypothesis. We believe our work calls for further reevaluation of progress in developing principled and efficient MTL algorithms.

We conclude by addressing the limitations of our work. While we covered a wide range of popular benchmarks, we do not exclude the existence of settings where unitary scalarization underperforms: discovering them is an interesting direction for future work. Furthermore, our experimental results were obtained via grid searches under limited compute resources: some of the methods might benefit from further fine-tuning. Nevertheless, we remark that fine-tuning will be easier for unitary scalarization due to its shorter runtimes. Finally, we presented the regularization hypothesis only as a partial explanation of our results: we hope it will steer further analysis and consequently improve the understanding of MTL.

## Acknowledgements

VK was funded by Samsung R&D Institute UK through the EPSRC Centre for Doctoral Training (CDT) in Autonomous Intelligent Machines and Systems (AIMS) at the University of Oxford . ADP was funded by EPSRC for the AIMS CDT, grant EP/L015987/1, and by an IBM PhD fellowship. SW has received funding from the European Research Council under the European Union’s Horizon 2020 research and innovation programme (grant agreement number 637713). The experiments were made possible by a generous equipment grant from NVIDIA. We would like to thank Lin et al. [40], Sodhani et al. [55] and Sener and Koltun [54] for publicly releasing their code. The authors thank Kristian Hartikainen for helpful comments on the RL experiments, and Gabriel Gama for spotting a bug in the logging of training statistics for supervised learning. VK thanks Ryota Tomioka for useful discussions on multitask optimization.

## References

[1] Z. Allen-Zhu, Y. Li, and Z. Song. A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning, 2019.  
[2] V. Badrinarayanan, A. Kendall, and R. Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.  
[3] B. Bakker and T. Heskes. Task clustering and gating for bayesian multitask learning. Journal of Machine Learning Research, 2003.  
[4] Q. Cappart, D. Chételat, E. B. Khalil, A. Lodi, C. Morris, and P. Velickovic. Combinatorial optimization and reasoning with graph neural networks. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, 2021.  
[5] R. Caruana. Multitask learning. Machine Learning, 28(1):41–75, 1997.  
[6] R. Caruana. Multitask learning. PhD thesis, School of Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213, 1997.  
[7] R. Caruana, S. Lawrence, and L. Giles. Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping. In Neural Information Processing Systems, 2000.  
[8] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In European Conference on Computer Vision, 2018.  
[9] S. Chen, Y. Zhang, and Q. Yang. Multi-task learning in natural language processing: An overview. CoRR, 2021.  
[10] Z. Chen, V. Badrinarayanana, C.-Y. Lee, and A. Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, 2018.  
[11] Z. Chen, J. Ngiam, Y. Huang, T. Luong, H. Kretzschmar, Y. Chai, and D. Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Neural Information Processing Systems, 2020.  
[12] R. Collobert and J. Weston. A unified architecture for natural language processing: deep neural networks with multitask learning. In Machine Learning, Proceedings ofthe Twenty-Fifth International Conference (ICML 2008), Helsinki, Finland, June 5-9, 2008, 2008.  
[13] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In Conference on Computer Vision and Pattern Recognition, 2016.  
[14] J. Désidéri. Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. Comptes Rendus Mathematique, 350:313–318, 2012.  
[15] T. Dietterich. Overfitting and undercomputing in machine learning. ACM Computing Surveys, page 326–327, sep 1995.  
[16] T. Evgeniou and M. Pontil. Regularized multi–task learning. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2004.  
[17] J. Fliege and B. F. Svaiter. Steepest descent methods for multicriteria optimization. Mathematical Methods ofOperations Research, 2000.  
[18] M. Guo, A. Haque, D.-A. Huang, S. Yeung, and L. Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.  
[19] P. Guo, C.-Y. Lee, and D. Ulbricht. Learning to branch for multi-task learning. 2020.  
[20] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, 2018.  
[21] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. Conference on Computer Vision and Pattern Recognition, 2016.  
[22] T. Heskes. Empirical bayes for learning to learn. In International Conference on Machine Learning, 2000.  
[23] M. Hessel, H. Soyer, L. Espeholt, W. Czarnecki, S. Schmitt, and H. van Hasselt. Multi-task deep reinforcement learning with popart. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 3796–3803. AAAI Press, 2019.  
[24] T. M. Hospedales, A. Antoniou, P. Micaelli, and A. J. Storkey. Meta-learning in neural networks: A survey. CoRR, 2020.  
[25] W. Huang, I. Mordatch, and D. Pathak. One policy to control them all: Shared modular policies for agent-agnostic control. In Proceedings ofthe 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, 2020.  
[26] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
[27] M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver, and K. Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.  
[28] A. Javaloy and I. Valera. Rotograd: Gradient homogenization in multitask learning. In International Conference on Learning Representations, 2022.  
[29] D. Kalashnikov, J. Varley, Y. Chebotar, B. Swanson, R. Jonschkowski, C. Finn, S. Levine, and K. Hausman. Mt-opt: Continuous multi-task robotic reinforcement learning at scale. CoRR, 2021.  
[30] A. Kendall, Y. Gal, and R. Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.  
[31] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang. On large-batch training for deep learning: Generalization gap and sharp minima. International Conference on Learning Representations, 2017.  
[32] K. Khetarpal, M. Riemer, I. Rish, and D. Precup. Towards continual reinforcement learning: A review and perspectives. arXiv preprint arXiv:2012.13490, 2020.  
[33] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun, editors, International Conference on Learning Representations, 2015.  
[34] B. Kleinberg, Y. Li, and Y. Yuan. An alternative view: When does SGD escape local minima? In International Conference on Machine Learning, 2018.  
[35] I. Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. IEEE Conference on Computer Vision and Pattern Recognition, 2017.  
[36] V. Kurin, S. Godil, S. Whiteson, and B. Catanzaro. Can q-learning with graph networks learn a generalizable branching heuristic for a SAT solver? In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.  
[37] V. Kurin, M. Igl, T. Rocktäschel, W. Boehmer, and S. Whiteson. My body is a cage: the role of morphology in graph-based incompatible control. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021.  
[38] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. IEEE, 1998.  
[39] M. Li, M. Soltanolkotabi, and S. Oymak. Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks. In International Conference on Artificial Intelligence and Statistics, 2020.  
[40] B. Lin, F. Ye, and Y. Zhang. A closer look at loss weighting in multi-task learning. In arXiv preprint arXiv:2111.10603, 2022.  
[41] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 2021.  
[42] L. Liu, Y. Li, Z. Kuang, J.-H. Xue, Y. Chen, W. Yang, Q. Liao, and W. Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2021.  
[43] S. Liu, E. Johns, and A. J. Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1871–1880, 2019.  
[44] Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In Proceedings ofInternational Conference on Computer Vision (ICCV), 2015.  
[45] S. Ma, R. Bassily, and M. Belkin. The power of interpolation: Understanding the effectiveness of sgd in modern over-parametrized learning. In International Conference on Machine Learning, 2018.  
[46] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Cross-stitch networks for multi-task learning. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, 2016.  
[47] S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, and P. Stone. Curriculum learning for reinforcement learning domains: A framework and survey. J. Mach. Learn. Res., 2020.  
[48] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine Learning, 2022.  
[49] E. Parisotto, L. J. Ba, and R. Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.  
[50] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style, high-performance deep learning library. In Neural Information Processing Systems. 2019.  
[51] A. A. Rusu, S. G. Colmenarejo, Ç. Gülçehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell. Policy distillation. In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.  
[52] S. Sabour, N. Frosst, and G. E. Hinton. Dynamic Routing between Capsules. 2017.  
[53] M. L. Seltzer and J. Droppo. Multi-task learning in deep neural networks for improved phoneme recognition. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver, BC, Canada, May 26-31, 2013, 2013.  
[54] O. Sener and V. Koltun. Multi-task learning as multi-objective optimization. In Neural Information Processing Systems, 2018.  
[55] S. Sodhani, A. Zhang, and J. Pineau. Multi-task reinforcement learning with context-based representations. In M. Meila and T. Zhang, editors, International Conference on Machine Learning, 2021.  
[56] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 2014.  
[57] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu. Distral: Robust multitask reinforcement learning. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 4496–4506, 2017.  
[58] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu. Distral: Robust multitask reinforcement learning. In Neural Information Processing Systems, 2017.  
[59] W.-C. Tseng. Weichengtseng/pytorch-pcgrad, 2020. URL https://github.com/ WeiChengTseng/Pytorch-PCGrad.git.  
[60] H. P. van Hasselt, A. Guez, M. Hessel, V. Mnih, and D. Silver. Learning values across many orders of magnitude. Advances in Neural Information Processing Systems, 29:4287–4295, 2016.  
[61] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.  
[62] Z. Wang, Y. Tsvetkov, O. Firat, and Y. Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In International Conference on Learning Representations, 2021.  
[63] D. Xin, B. Ghorbani, A. Garg, O. Firat, and J. Gilmer. Do current multi-task optimization methods in deep learning even help? In Neural Information Processing Systems, 2022.  
[64] F. Yu, V. Koltun, and T. Funkhouser. Dilated residual networks. In Computer Vision and Pattern Recognition, 2017.  
[65] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In L. P. Kaelbling, D. Kragic, and K. Sugiura, editors, 3rd Annual Conference on Robot Learning, 2019.  
[66] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn. Gradient surgery for multi-task learning. In Neural Information Processing Systems, 2020.

## Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes]  
(b) Did you describe the limitations of your work? [Yes] see §6.  
(c) Did you discuss any potential negative societal impacts of your work? [Yes] due to space constraints, we provide a discussion in appendix A.  
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]  
(b) Did you include complete proofs of all theoretical results? [Yes] we provide full proofs in the Appendix, and refer to them in the main body of the paper.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] we provide the code and the instructions in the supplemental material.  
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes]  
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes]  
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] see appendix C.1.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]  
(b) Did you mention the license of the assets? [Yes] appendix C.3 describes licenses of all benchmarks and implementations we used for our work.  
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] we include the code and the instructions on how to replicate the experiments into the supplemental material.  
(d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [N/A]  
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]  
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]  
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

## A Societal Impact

Due to the object of its study, our work does not have a direct societal impact. However, as any machine learning paper, it can potentially negatively effect the society through automation and loss of jobs. While it is hard to anticipate any particular risk, as any technology, if not regulated properly, it might lead to growing social and economic inequality.

On the positive side, our work might have a positive environmental impact since it advocates for simpler and more economical methods which will reduce energy consumption in data centers. Finally, simpler methods are usually easier to understand, which is beneficial in terms of explainability, an important factor for real-life applications.

## B Supplement to the Overview of Multi-Task Optimizers

This section presents the proofs and the technical results omitted from section 5, along with a description of the use of per-task gradients with respect to the last shared activation for encoder-decoder architectures (usually less expensive than per-task gradients with respect to shared parameters).

## B.1 MGDA

Proposition 1. The MGDA SMTO [54] converges to a superset of the convergence points of unitary scalarization. More specifically, it converges to any point $\bar { \theta _ { \parallel } ^ { * } }$ such that: $\mathbf { 0 } \in C o n \nu ( \{ \dot { \nabla } \theta _ { \parallel } ^ { * } \mathcal { L } _ { i } | \dot { \iota } \in \mathcal { T } \} )$

Proof. As shown by Désidéri [14], equation (3) is a simplex-constrained norm-minimization problem. In other words, the argument of the minimum is the projection of 0 onto the feasible set. Therefore:

$$
\mathbf {g} = \mathbf {0} \iff \mathbf {0} \in \mathrm{Conv} (\{\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \mid i \in \mathcal {T} \}).
$$

It then suffices to point out that $\begin{array} { r } { \sum _ { i \in \mathcal { T } } \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } \ = \ \mathbf { 0 } \Longleftrightarrow \ \sum _ { i \in \mathcal { T } } \frac { 1 } { | \mathcal { T } | } \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } \ = \ \mathbf { 0 } \Rightarrow \ \mathbf { 0 } \in \mathcal { } } \end{array}$ $\mathrm { C o n v } ( \{ \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \mid i \in \mathcal { T } \} )$ to conclude the proof. □

Due to the cost of computing per-task gradients, Sener and Koltun [54] propose MGDA-UB, which replaces the gradients wrt the parameters $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ with the gradients wrt the shared activation $\nabla _ { \mathbf { z } } { \mathcal { L } } _ { i }$ in the computation of the coefficients of $\begin{array} { r } { \mathbf { g } \stackrel { \cdot \cdot } { = } - \sum _ { i } \alpha _ { i } \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } } \end{array}$ . This yields an upper bound on the objective of equation (3), thus restricting the set of points the algorithm convergences to. Rather than directly relying on $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i } , \mathbf { g }$ can then be obtained by computing the gradient of $\textstyle \sum _ { i \in T } { \alpha _ { i } } { \mathcal { L } } _ { i }$ via reverse-mode differentiation, hence saving memory and compute.

Corollary 3. The MGDA-UB SMTO by Sener and Koltun [54] converges to any point such that: $\mathbf { 0 } \in C o n \bar { \nu } ( \{ \nabla _ { \mathbf { z } } \mathcal { L } _ { i } \mid i \in \mathcal { T } \} )$ . Furthermore, $i f \frac { \partial \mathbf { z } } { \partial \pmb { \theta } _ { \parallel } }$ is non-singular, it converges to a superset of the convergence points ofthe unitary scalarization.

Proof. The first part of the proof proceeds as the proof of proposition 1, noting that the MGDA-UB update is associated to the following problem:

$$
\begin{array}{l} \max _ {\boldsymbol {\alpha}} \qquad - \frac {1}{2} \left\| \mathbf {g} \right\| _ {2} ^ {2} \\ \text {s.t.} \quad \sum_ {i} \alpha_ {i} \nabla_ {\mathbf {z}} \mathcal {L} _ {i} = - \mathbf {g}, \quad \sum_ {i \in \mathcal {T}} \alpha_ {i} = 1, \\ \alpha_ {i} \geq 0 \quad \forall i \in \mathcal {T}. \\ \end{array}
$$

In order to show that a stationary point of the unitary scalarization satisfies $\mathbf { 0 } \in \mathrm { C o n v } ( \{ \nabla _ { \mathbf { z } ^ { * } } \mathcal { L } _ { i } \mid i \in$ $\tau _  \} )$ , we will assume $\frac { \partial \mathbf { z } } { \partial \pmb { \theta } _ { \parallel } }$ is non-singular, as done by Sener and Koltun [54, theorem 1]. Then, relying

on the chain rule, the result follows from:

$$
\begin{array}{l} \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} = \mathbf {0} \iff \sum_ {i \in \mathcal {T}} \frac {1}{| \mathcal {T} |} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} = \mathbf {0} \\ \Longleftrightarrow \sum_ {i \in \mathcal {T}} \frac {\frac {\partial \mathbf {z}}{\partial \boldsymbol {\theta} _ {\parallel}}}{| \mathcal {T} |} \nabla_ {\mathbf {z}} \mathcal {L} _ {i} = \mathbf {0} \\ \Longleftrightarrow \left(\frac {\partial \mathbf {z}}{\partial \boldsymbol {\theta} _ {\parallel}}\right) ^ {- 1} \frac {\partial \mathbf {z}}{\partial \boldsymbol {\theta} _ {\parallel}} \sum_ {i \in \mathcal {T}} \frac {1}{| \mathcal {T} |} \nabla_ {\mathbf {z}} \mathcal {L} _ {i} = \mathbf {0} \\ \Longleftrightarrow \sum_ {i \in \mathcal {T}} \frac {1}{| \mathcal {T} |} \nabla_ {\mathbf {z}} \mathcal {L} _ {i} = \mathbf {0} \\ \Rightarrow \mathbf {0} \in \operatorname{Conv} (\{\nabla_ {\mathbf {z}} \mathcal {L} _ {i} \mid i \in \mathcal {T} \}) \\ \end{array}
$$

![](images/97f41d56870a8b5f620095541d04ec0604fea285582d4a60d41590246bb715e3.jpg)

## B.2 IMTL

Proposition 2. IMTL by Liu et al. [42] updates $\theta _ { \parallel }$ by taking a step in the steepest descent direction whose cosine similarity with per-task gradients is the same across tasks.

Proof. First, equation (4) solves the linear system in $\pmb { \alpha } : = [ \alpha _ { 1 } , \dots , \alpha _ { m } ]$ given by:

$$
\mathbf {g} ^ {T} \left(\frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1}}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \|} - \frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \|}\right) = \mathbf {0} \quad \forall i \in \mathcal {T} \setminus \{1 \},
$$

$$
\mathbf {g} = - \sum_ {i} \alpha_ {i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}, \quad \sum_ {i \in \mathcal {T}} \alpha_ {i} = 1,
$$

which corresponds to finding a point of $\mathcal { A } ^ { \prime } : = \mathrm { A f f } ( \{ \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } | i \in \mathcal { T } \} )$ ) which is orthogonal to $A : =$ Aff $\left( \left\{ \frac { \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } } { \left\| \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } \right\| } | i \in T \right\} \right)$ . To see this, it suffices to point out that any point orthogonal to A is also orthogonal to the vector subspace spanned by differences of vectors belonging to A. As this subspace has $m - 1$ dimensions, any vector orthogonal to $\left( \frac { \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { 1 } } { \left\| \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { 1 } \right\| } - \frac { \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } } { \left\| \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } \right\| } \right)$ for each $i \in \mathcal { T } \backslash \{ 1 \}$ is orthogonal to the entire subspace.

Second, consider the problem of finding a point in A that is orthogonal to the linear subspace spanned by differences of vectors in A. In other words, we seek the projection of 0 onto A. Recalling the definition of A, we can write:

$$
\begin{array}{l} \max _ {\boldsymbol {\alpha}} \quad - \frac {1}{2} \left\| \mathbf {g} ^ {\prime} \right\| _ {2} ^ {2} \\ \text {s.t.} \quad \sum_ {i} \alpha_ {i} \frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|} = - \mathrm{g} ^ {\prime}, \quad \sum_ {i} \alpha_ {i} = 1. \tag {7} \\ \end{array}
$$

The solution of equation (7) is always collinear to the solution of equation (4). In fact, if a vector $\mathbf { g } \in { \mathcal { A } } ^ { \prime }$ is orthogonal to the affine subspace A (or to the linear subspace spanned by differences of its members), then $\begin{array} { r } { \gamma \mathbf { g } = \left( - \gamma \sum _ { i } \left( \alpha _ { i } \left. \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \right. \right) \frac { \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } } { \left. \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } \right. } \right) } \end{array}$ is orthogonal to A as well, and $\begin{array} { r } { \gamma = \frac { 1 } { \sum _ { i } \left( \alpha _ { i } \left. \nabla _ { \theta _ { \parallel } } \mathcal { L } _ { i } \right. \right) } \implies \gamma \mathbf { g } \in \mathcal { A } . } \end{array}$

Finally, equation (7) differs from equation (3) in two aspects: α is not constrained to be non-negative (hence the convex hull is replaced by the affine hull), and the task vectors are normalized. Therefore, equation (7) is the dual of:

$$
\min _ {\mathbf {g}, \epsilon} \qquad \epsilon + \frac {1}{2} \| \mathbf {g} \| _ {2} ^ {2}
$$

$$
\text {s.t.} \quad \frac {\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} ^ {T}}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|} \mathbf {g} = \epsilon \quad \forall i \in \{1, \dots , m \}. \tag {8}
$$

The proposition then follows by comparing equation (8) with equation (2), and recalling that IMTL-L only adds a scaling factor to the chosen update direction. □

Corollary 1. IMTL by Liu et al. [42] converges to a superset of the Pareto-stationary points for $\theta _ { \parallel }$ (and hence of the convergence points of the unitary scalarization). More specifically, it converges to any point $\theta _ { \parallel } ^ { * }$ such that: $\mathbf { 0 } \in \bar { A } f f \left( \left\{ \nabla _ { \pmb { \theta } _ { \parallel } ^ { * } } \mathcal { L } _ { i } / \left\| \nabla _ { \pmb { \theta } _ { \parallel } ^ { * } } \mathcal { L } _ { i } \right\| | i \in T \right\} \right)$

Proof. Inspecting equation (8), which yields a collinear point to the IMTL update, reveals that IMTL might converge to non Pareto-stationary points: due to the restrictive equality constraints, the minimizer of equation (8) might be 0 even if a descent direction exists. Furthermore, its dual, equation (7), implies that:

$$
\begin{array}{l} \mathbf {g} = \mathbf {0} \iff \mathbf {0} \in \operatorname{Aff} \left(\left\{\frac {\nabla_ {\boldsymbol {\theta} _ {\|}} \mathcal {L} _ {i}}{\| \nabla_ {\boldsymbol {\theta} _ {\|}} \mathcal {L} _ {i} \|} \mid i \in \mathcal {T} \right\}\right) \\ \Longleftrightarrow \mathbf {0} \in \operatorname{Aff} \left(\left\{\nabla_ {\boldsymbol {\theta} _ {\|}} \mathcal {L} _ {i} \mid i \in \mathcal {T} \right\}\right), \\ \end{array}
$$

which, noting that Conv $( { \mathcal { A } } ) \subseteq \operatorname { A f f } ( { \mathcal { A } } )$ for any A, concludes the proof.

![](images/1ba0c291013152d049646388b10d6f62071ff308099c78f742e904ba2fae884e.jpg)

Similarly to MGDA-UB, Liu et al. [42] advocate using $\nabla _ { \mathbf { z } } \mathcal { L } _ { i }$ in place of $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ while solving equation (4), typically reducing the cost of computing the coefficients of $\begin{array} { r } { \mathbf { g } = - \sum _ { i } \alpha _ { i } \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } } \end{array}$

Corollary 4. When employing the approximation of problem (4) that relies on $\nabla _ { \mathbf { z } } { \mathcal { L } } _ { i } ,$ , IMTL by Liu et al. [42] converges to $\begin{array} { r } { \mathbf { 0 } \in A f f \bigg ( \bigg \{ \frac { \nabla _ { \mathbf { z } } \mathcal { L } _ { i } } { \| \nabla _ { \mathbf { z } } \mathcal { L } _ { i } \| } \mid i \in \mathcal { T } \bigg \} \bigg ) . \ I f \frac { \partial \mathbf { z } } { \partial \theta _ { \parallel } } } \end{array}$ is non-singular, this is a superset of of the convergence points ofthe unitary scalarization.

Proof. Following the proof of proposition 2, the following problem yields a collinear point to the $\nabla _ { \mathbf { z } } \mathcal { L } _ { i }$ -approximate IMTL update:

$$
\begin{array}{l} \max _ {\boldsymbol {\alpha}} \qquad - \frac {1}{2} \left\| \mathbf {g} ^ {\prime} \right\| _ {2} ^ {2} \\ \text {s.t.} \quad \sum_ {i} \alpha_ {i} \frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} = - \mathbf {g} ^ {\prime}, \quad \sum_ {i} \alpha_ {i} = 1. \\ \end{array}
$$

Therefore:

$$
\mathbf {g} = \mathbf {0} \iff \mathbf {0} \in \mathrm{Aff} \left(\left\{\frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} \mid i \in \mathcal {T} \right\}\right).
$$

Finally, assuming $\displaystyle \frac { \partial \mathbf { z } } { \partial \pmb { \theta } _ { \parallel } }$ is non-singular, we can replicate the procedure in the proof of corollary 3 to get:

$$
\begin{array}{l} \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} = \mathbf {0} \iff \sum_ {i \in \mathcal {T}} \frac {1}{| \mathcal {T} |} \nabla_ {\mathbf {z}} \mathcal {L} _ {i} = \mathbf {0} \\ \Longleftrightarrow \sum_ {i \in \mathcal {T}} \frac {\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|}{| \mathcal {T} |} \frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} = \mathbf {0} \\ \Longleftrightarrow \left(\frac {| \mathcal {T} |}{\sum_ {i \in \mathcal {T}} \left(\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|\right)}\right) \sum_ {i \in \mathcal {T}} \frac {\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|}{| \mathcal {T} |} \frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} = \mathbf {0} \\ \Rightarrow \mathbf {0} \in \operatorname{Conv} \left(\left\{\frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} \mid i \in \mathcal {T} \right\}\right) \\ \Rightarrow \mathbf {0} \in \mathrm{Aff} \left(\left\{\frac {\nabla_ {\mathbf {z}} \mathcal {L} _ {i}}{\| \nabla_ {\mathbf {z}} \mathcal {L} _ {i} \|} \mid i \in \mathcal {T} \right\}\right), \\ \end{array}
$$

which shows that $\mathrm { A f f } \left( \left\{ \begin{array} { l l } { \nabla _ { \mathbf { z } } \mathcal { L } _ { i } } \end{array} | i \in T \right\} \right)$ contains the convergence points of the unitary scalariza tion. □

## B.3 PCGrad

Proposition 3. PCGrad is equivalent to a dynamic, and possibly stochastic, loss rescaling for $\theta _ { \parallel }$ . At each iteration, per-task gradients are rescaled asfollows:

$$
\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \leftarrow \left(1 + \sum_ {j \in \mathcal {T} \backslash \{i \}} d _ {j i}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}, d _ {j i} \in \left[ 0, \frac {\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\|}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|} \right].
$$

Furthermore, $i f \left| \mathcal { T } \right| > 2 , d _ { j i }$ is a random variable, and the above range contains its support.

Proof. We start by pointing out that:

$$
\begin{array}{l} \left[ \frac {- \mathbf {g} _ {i} ^ {T} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} (\mathbf {x})}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\| ^ {2}} \right] _ {+} = \left[ \frac {- \mathbf {g} _ {i} ^ {T} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} (\mathbf {x})}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\|} \right] _ {+} \frac {1}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\|} \\ = \left[ - \cos (\mathbf {g} _ {i}, \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j}) \| \mathbf {g} _ {i} \| \right] _ {+} \frac {1}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \|} \\ \in \left[ 0, \frac {\| \mathbf {g} _ {i} \|}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \|} \right]. \\ \end{array}
$$

As $\mathbf { g } _ { i }$ is obtained by iterative projections of $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } _ { i }$ onto the normals of $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { j } \forall j \in \mathcal { T } \backslash \{ i \}$ , and the norm of a vector can only decrease or remain unvaried after projections, we can write the coefficient of each g<sub>i</sub> update as:

$$
d _ {i j} := \left[ \frac {- \mathbf {g} _ {i} ^ {T} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} (\mathbf {x})}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\| ^ {2}} \right] _ {+} \in \left[ 0, \frac {\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \right\|}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right\|} \right], \forall i \neq j.
$$

Furthermore, $\operatorname { i f } | \mathcal { T } | > 2$ the contraction factor $\frac { \| \mathbf { g } _ { i } \| } { \left\| \nabla _ { \theta _ { \| } } \mathcal { L } _ { i } \right\| }$ for the norm of $g _ { i }$ depends on the ordering of the projections, which is stochastic by design [66]. Therefore, $d _ { i j }$ a random variable whose support is contained in $\left[ 0 , { \frac { \left\| \nabla _ { { \boldsymbol { \theta } } _ { \parallel } } { \mathcal { L } } _ { i } \right\| } { \left\| \nabla _ { { \boldsymbol { \theta } } _ { \parallel } } { \mathcal { L } } _ { j } \right\| } } \right]$ . Finally, exploiting the definition of $d _ { i j }$ , we can re-write equation (5) as:

$$
\begin{array}{l} - \mathbf {g} = \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} + \sum_ {i \in \mathcal {T}} \sum_ {j \in \mathcal {T} \backslash \{i \}} d _ {i j} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} = \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} + \sum_ {j \in \mathcal {T}} \sum_ {i \in \mathcal {T} \backslash \{j \}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \\ = \sum_ {j \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} + \sum_ {j \in \mathcal {T}} \sum_ {i \in \mathcal {T} \backslash \{j \}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} = \sum_ {j \in \mathcal {T}} \left(\sum_ {i \in \mathcal {T} \backslash \{j \}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} + \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j}\right). \\ \end{array}
$$

Introducing (and then removing, using their definition) dummy variables $d _ { j j } = 1$

$$
\begin{array}{l} - \mathbf {g} = \sum_ {j \in \mathcal {T}} \left(\sum_ {i \in \mathcal {T} \backslash \{j \}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} + d _ {j j} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j}\right) = \sum_ {j \in \mathcal {T}} \left(\sum_ {i \in \mathcal {T}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}\right) = \sum_ {i \in \mathcal {T}} \left(\sum_ {j \in \mathcal {T}} d _ {j i} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i}\right) \\ = \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \left(\sum_ {j \in \mathcal {T}} d _ {j i}\right) = \sum_ {i \in \mathcal {T}} \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {i} \left(1 + \sum_ {j \in \mathcal {T} \backslash \{i \}} d _ {j i}\right), \\ \end{array}
$$

from which the result trivially follows.

![](images/d08a95499f5a3b3e8eecde0ca1c0107581a38ff7548d8b0a1189b140165a88ff.jpg)

Corollary 2. $I f | { \mathcal { T } } | = 2 ,$ , PCGrad will stop at any point where cos $( \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } ) = - 1$ . Furthermore, $i f { \mathcal { L } } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are differentiable, and $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { M T }$ is L-Lipschitz with $L > 0 ,$ , PCGrad with step size $\begin{array} { r } { t < \frac { 1 } { L } } \end{array}$ converges to a superset of the convergence points of the unitary scalarization.

Proof. Let us start from the first statement, which does not require any assumption on the loss landscape. From proposition 3, we get:

$$
\begin{array}{l} - \mathbf {g} = \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \left(1 + d _ {2 1}\right) + \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \left(1 + d _ {1 2}\right) \\ = \left(1 + \left[ \frac {- \cos (\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} , \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2}) \left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \right\|}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \right\|} \right] _ {+}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \\ + \left(1 + \left[ \frac {- \cos (\nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} , \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2}) \left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \right\|}{\left\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \right\|} \right] _ {+}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2}, \\ \end{array}
$$

which shows that, in case of conflicting gradient directions, gradient norms are rebalanced proportionally to the angle between them. For cos $( \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } ) \bar { = } - 1$ , the above evaluates to:

$$
- \mathbf {g} = \left(\frac {\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \| + \| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \|}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \|}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} + \left(\frac {\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {1} \| + \| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \|}{\| \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2} \|}\right) \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {2}.
$$

The first part of the result then follows by pointing out that, if $\cos ( \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } , \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 } ) = - 1$ , then $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 1 } = - \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { 2 }$ , and hence $\mathbf g = \mathbf 0$ . We remark that a similar proof appears in [66, theorem 1 and proposition 1]. However, our derivation relaxes the author’s assumptions on $\mathcal { L } ^ { \mathrm { M T } }$ and is therefore applicable to the training of neural networks.

Finally, given the assumptions on differentiability and smoothness, we need to prove that PCGrad converges to the stationary points of the unitary scalarization: this directly follows from [66, proposition 1]. □

## B.4 GradDrop

Proposition 5. Let us assume, as often demonstrated in the single-task case $I I , 4 5 J ,$ , that the multi-task network has the capacity to interpolate the data on all tasks at once: min $\begin{array} { r } { \mathcal { L } ^ { M T } = \sum _ { i \in \mathcal { T } } } \end{array}$ min<sub>θ</sub> $\mathcal { L } _ { i } ,$ and that its training by gradient descent attains such global minimum. Then, ifinf<sub>θ</sub> $\mathcal { L } _ { i } > - \infty \forall i \in$ T, unitary scalarization converges to a joint minimum.

Proof. It suffices to point out that if $\begin{array} { r } { \mathcal { L } ^ { \mathrm { M T } } ( \pmb { \theta } ^ { * } ) = \sum _ { i \in \mathcal { T } } } \end{array}$ min $\mathcal { L } _ { i } ,$ then the globally optimal loss is attained for all tasks. In other words ${ \mathcal { L } } _ { i } ( \theta ^ { * } ) = .$ min<sub>θ</sub> $\dot { \mathcal { L } } _ { i } \forall i \in \mathcal { T }$ , and hence $\nabla _ { \pmb { \theta } ^ { * } } \mathcal { L } _ { i } = \mathbf { 0 } \forall i \in \mathcal { T }$ (joint minimum). Furthermore, running gradient descent on min $\mathcal { L } ^ { \mathrm { M T } }$ corresponds to the unitary scalarization (§3), which concludes the proof. 口

Proposition 4. Let $\begin{array} { r } { \mathcal { L } ^ { R G D } ( \pmb { \theta } _ { \parallel } ) : = \sum _ { i \in \mathcal { T } } u _ { i } \mathcal { L } _ { i } ( \pmb { \theta } _ { \parallel } ) } \end{array}$ , where $u _ { i } \sim$ Bernou $l l i ( p ) \forall i \in \mathcal { T }$ and $p \in ( 0 , 1 ]$ The gradient $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { R G D }$ is always zero ifand only $i f \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } = \mathbf { 0 } \forall i \in \mathcal { T } .$ . In other words, the result from [11, proposition 1] can be obtained without any information on the sign ofper-task gradients.

Proposition 4 can be proved by adapting the proof from Chen et al. [11, proposition 1]: it suffices to replace $f ( \mathcal P )$ with the Bernoulli parameter $p ,$ which is non-negative by definition. In our opinion, this seriously undermines the conflicting gradient hypothesis that motivated GradDrop. For the reader’s convenience, we now provide a straightforward and self-contained proof.

Proof. Let us start from the statement on $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } ^ { \mathrm { R G D } } . \ \mathrm { I f } \ \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { i } = \mathbf { 0 } \ \forall i \in \mathcal { T }$ , then $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } ^ { \mathrm { R G D } } = \mathbf { 0 }$ with probability one. On the other hand, if $\exists j : \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { j } \neq \mathbf { 0 }$ , then:

$$
\begin{array}{l} \mathbb {P} \left[ \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} ^ {\mathrm{RGD}} \neq \mathbf {0} \right] \geq \mathbb {P} \left[ \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} ^ {\mathrm{RGD}} = \nabla_ {\boldsymbol {\theta} _ {\parallel}} \mathcal {L} _ {j} \right] \\ = p (1 - p) ^ {m - 1} > 0, \\ \end{array}
$$

where the first inequality comes from the fact that $\nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } ^ { \mathrm { R G D } } = \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } _ { j }$ is only one of the many instances of a non-null $\nabla _ { \pmb { \theta } _ { | | } } \mathcal { L } ^ { \mathrm { R G D } }$ □

Let $\mathrm { s i g n } ( \mathbf { x } )$ stand for the element-wise sign operator applied on x. On encoder-decoder architectures, similarly to MGDA and IMTL (see appendices B.1 and B.2), the authors do not apply GradDrop on $\nabla _ { \pmb { \theta } _ { \parallel } } \bar { \mathcal { L } } _ { i } .$ , but rather on a the usually less expensive $\nabla _ { \mathbf { z } } \mathcal { L } _ { i }$ . In more detail, they compute the

GradDrop sign purity scores p from equation (6) on $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \left( \mathrm { s i g n } ( \mathbf { z } ) \odot \nabla _ { \mathbf { z } } \mathcal { L } _ { i } \right) [ i ] \in \mathbb { R } ^ { r } } \end{array}$ , and then apply equation (6) on the $\nabla _ { \mathbf { z } } \mathcal { L } _ { i }$ gradients, yielding a vector $\mathbf { g } _ { z } \in \mathbb { R } ^ { n \times r }$ . Then, relying on reverse-mode differentiation, the update direction in the space of the parameters $\theta _ { \parallel }$ is obtained via a Jacobian-vector product: $\begin{array} { r } { \mathbf { g } = - \left( \frac { \partial \mathbf { z } } { \partial \pmb { \theta } _ { \parallel } } \right) ^ { T } \mathbf { g } _ { z } } \end{array}$ . Such a computation replaces the similar $\begin{array} { r } { \nabla _ { \pmb { \theta } _ { \parallel } } \mathcal { L } ^ { \mathrm { M T } } = \left( \frac { \partial \mathbf { z } } { \partial \pmb { \theta } _ { \parallel } } \right) ^ { T } \nabla _ { \mathbf { z } } \mathcal { L } ^ { \mathrm { M T } } } \end{array}$ from the unitary scalarization.

## C Experimental Setting, Reproducibility

We now present details concerning the experimental settings from §4, including details on the employed open-source software, dataset information, hardware specifications, and hyper-parameters.

## C.1 Supervised Learning

All the experiments were run under Ubuntu 18.04 LTS, on a single GPU per run (using two 8-GPU machines in total). Timing experiments were all run on Nvidia GeForce GTX 1080 Ti GPUs, with an Intel Xeon E5-2650 CPU. The remaining experiments were run on either Nvidia GeForce RTX 2080 Ti GPUs or Nvidia GeForce GTX 1080 Ti GPUs, respectively using an Intel Xeon Gold 6230 CPU or an Intel Xeon E5-2650 CPU.

## C.1.1 MultiMNIST

Multi-MNIST, originally introduced by Sabour et al. [52] and as modified by Sener and Koltun [54], is a simple two-task supervised learning benchmark dataset constructed by uniformly sampling MNIST [38] images, and placing one in the top-left corner, the other in the bottom-right corner. Each of the two overlaid images corresponds to a 10-class classification task. Using the above procedure, we generate the Multi-MNIST training set from the first 50000 MNIST training images, the validation set from the last 10000 training images, and the test set from the original MNIST test set. For consistency with the experimental setup of Sener and Koltun [54], we employ a modified encoderdecoder version of the LeNet architecture [38]. Specifically, the last layer is omitted from the encoder, and two fully-connected layers are employed as task-specific predictive heads. The cross-entropy loss is used for both tasks. All methods are trained for 100 epochs using Adam [33] in the stochastic gradient setting, with an initial learning rate of $\eta = 1 0 ^ { - 2 }$ (tuned in $\breve { \eta } \in \{ 1 0 ^ { - 3 } , \mathsf { \bar { 1 0 } ^ { - 2 } } , 1 0 ^ { - 1 } \}$ and yielding the best validation results for all considered algorithms), exponentially decayed by 0.95 after each epoch, and a mini-batch size of 256.

## C.1.2 CelebA

The CelebA [44] dataset consists of 200, 000 headshots (with standard training, validation and test splits) associated with the presence or absence of 40 attributes. In the MTL literature, is commonly treated as a 40-task classification problem, each task being a binary classification problem for an attribute. As commonly done in previous work [42, 54, 66], we employ an encoder-decoder architecture where the encoder is a ResNet-18 [21] (without the final layer) with batch normalization layers [26], and the per-task decoders are linear classifiers. The cross-entropy loss is used for all tasks. The training is performed from scratch for 50 epochs using Adam, with a mini-batch size of 128 and a per-epoch exponential decay factor of 0.95. As common on this network-dataset combination [11, 40], the initial learning rate is $\eta \ : = \ : 1 0 ^ { - 3 }$ for all methods except for MGDA and IMTL, for which $\eta = 5 \times 1 0 ^ { - 4 }$ yielded a better validation performance. As done by the respective authors, for PCGrad, RLW and GradDrop we use the same learning rate as the unitary scalarization [11, 40, 66].

## C.1.3 Cityscapes

We rely on the version of the dataset pre-processed by Liu et al. [43], which consists of 2, 975 training and 500 test images and presents two tasks: semantic segmentation on 7 classes, and depth estimation. We further split the original training set into a validation set of 595 images, employed to tune hyper-parameters, and a training set of 2380 images. Consistently with recent work [40], we rely on a dilated ResNet-50 architecture pre-trained on ImageNet [64] for the encoder, and on the Atrous Spatial Pyramid Pooling [8], which internally uses batch normalization, as decoders. While more powerful encoders might lead to better performance on Cityscapes, like the SegNet [2] used in [28, 41, 48], we aim to provide a fair comparison of MTL optimizers, rather than maximize overall task performance. Cross-entropy loss is employed on the semantic segmentation task, whereas the $\ell _ { 1 }$ loss is used for the depth estimation. The training is performed by using Adam with a mini-batch size of 32 for 100 epochs, with an initial step size $\eta ^ { \star } = 5 \times 1 0 ^ { - \bar { 4 } }$ resulting in the best validation performance for all algorithms, exponentially decayed by 0.95 at each epoch.

## C.2 Reinforcement Learning

Similarly to the supervised learning experiments, we ran all the experiments under Ubuntu 18.04 LTS using one GPU per run (using six 8-GPU machines in total). Timing experiments were all run using NVIDIA GeForce RTX 2080 Ti GPUs, with an Intel Xeon Gold 6230 CPU. The main bulk of the remaining experiments was run on Nvidia GeForce RTX 2080 Ti GPUs with either Intel Xeon Gold 6230 or Intel Xeon Silver 4216. We utilised NVIDIA GeForce RTX 3080 GPUs with Intel Xeon Gold 6230 CPUs for a small fraction of experiments.

We use Meta-World’s MT10/MT50 for our experiment. The benchmark consists of ten/fifty tasks in which a simulated robot manipulator has to perform various actions, e.g., pressing a button, opening a door, or pushing the block. We use Sodhani et al. [55] for most of the hyperparameters and list them in Table 1. We use bold font where we use a hyperparameter different from Sodhani et al. [55]. Similarly to Sodhani et al. [55], we use the V1 version of Metaworld for our experiments<sup>2</sup>. Sodhani et al. [55] use a shared entropy loss weight α for PCGrad and separate α for unitary scalarization<sup>3</sup>. In our experiments, use shared α for all of the methods for fairness. Since it is a single number (rather than a vector), we used unitary scalarization to update α for all SMTOs apart from PCGrad which was already implemented in [55].

We use the same network architecture as in Sodhani et al. [55], i.e. a three-layered feedforward fully-connected network with 400 hidden units per layer for both, the actor and the critic. The actor is shared across all tasks as well as the critic.

To normalize rewards, we keep track of first and second moments in the buffer and normalise the rewards by their standard deviation: $r _ { i } ^ { \prime } = { { r } _ { i } } / { { { \hat { \sigma } } } _ { i } }$ , where $\hat { \sigma } _ { i }$ is the sample standard deviation of the rewards for environment i.

Sodhani et al. [55] average the gradient for unitary scalarisation and pcgrad, whereas our SMTO implementations sum the gradients, i.e. effectively using larger learning rates (apart from MGDA that assures that all the aggregation weights sum to 1). We tried reducing the learning rate for SMTOs that sum (RLW Norm., RLW Diri., and GradDrop) both for MT10 and MT50, but it worked worse for these methods and we kept the default learning rate for them as well. We had to use a smaller learning rate for IMTL, because with the default one it crashed at the beginning of training due to numerical overflow. Smaller learning rate did not prevent it from crashing, but this happened much later.

We tried $1 0 ^ { 6 } , 2 \times 1 0 ^ { 6 }$ , and $4 \times 1 0 ^ { 6 }$ for the replay buffer size with the last being superior in terms of stability. Additionally, for $l _ { 2 }$ actor regularization, we tried 0.0001 and 0.0003 with the latter being slightly superior for the baseline. We tried the same options for other SMTOs, and picked the best option for each of the method. For MGDA, no regularisation works best, most likely due to a strong regularization effect of the method itself, which is mirrored by our supervised learning results. PCGrad and Graddrop work best with the regularization coefficient of 0.0001. Both RLW variants use the same coefficient as the baseline (0.0003).

For MT50, we took the best MT10 hyperparameters, and we believe one could obtain even better results for unitary scalarisation since it is much faster to tune compared to other SMTOs (e.g. 15 hours for unitary scalarisation vs 9 days for PCGrad).

## C.3 Software Acknowledgments and Licenses

Our codebase is built upon several prior works: [54], [43], [40] and [55]: all of them were released under a MIT license. We also acknowledge Tseng [59], upon which we built some of our code. Multi-MNIST is based on MNIST dataset that is released under Creative Commons Attribution-Share Alike 3.0 license. The code for generating Multi-MNIST dataset was taken from Sener and Koltun [54] released under MIT license. CelebA dataset has a custom license allowing noncommercial research purposes. More details can be found on the project website:http://mmlab. ie.cuhk.edu.hk/projects/CelebA.html. Cityscapes also has a custom license allowing noncommercial research purposes. The full text of the license can be found on the project website:https: //www.cityscapes-dataset.com/license/. Metaworld, used for RL experiments is released under MIT license.

## D Supplementary Supervised Learning Experiments

Table 1: Hyperparameters of the RL experiments. Hyperparameters different from Sodhani et al. [55] are in bold.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">All methods</td></tr><tr><td>- training steps</td><td>2,000,000</td></tr><tr><td>- batch size</td><td>1280</td></tr><tr><td>- Replay buffer size</td><td>4,000,000</td></tr><tr><td>- actor learning rate</td><td>0.0003</td></tr><tr><td>- critic learning rate</td><td>0.0003</td></tr><tr><td>- entropy α learning rate</td><td>0.0003</td></tr><tr><td>- shared entropy α</td><td>True</td></tr><tr><td>- runs</td><td>10</td></tr><tr><td>- discounting γ</td><td>0.99</td></tr><tr><td colspan="2">Unit. Scal.</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0003</td></tr><tr><td colspan="2">PCGrad</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0001</td></tr><tr><td colspan="2">RLW Norm.</td></tr><tr><td>- normal mean</td><td>0</td></tr><tr><td>- normal std</td><td>1</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0003</td></tr><tr><td colspan="2">RLW Diri.</td></tr><tr><td>- α</td><td>1</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0003</td></tr><tr><td colspan="2">GradDrop</td></tr><tr><td>- k</td><td>1</td></tr><tr><td>- p</td><td>0.5</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0001</td></tr><tr><td colspan="2">MGDA</td></tr><tr><td>- gradient normalization</td><td> $L_2$ </td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0</td></tr><tr><td colspan="2">IMTL</td></tr><tr><td>- actor learning rate</td><td>0.00003</td></tr><tr><td>- critic learning rate</td><td>0.00003</td></tr><tr><td>- entropy α learning rate</td><td>0.00003</td></tr><tr><td>- actor  $l_2$  coeff.</td><td>0.0</td></tr></table>

This section presents supervised learning results omitted from §4.1. In particular, we show additional plots for the experiments of §4.1, then present an analysis of the regularising effect of SMTOs in the absence of single-task regularization (§D.2), and conclude with an ablation study on GradDrop’s dependency on the sign of per-task gradients (§D.3).

## D.1 Addendum

This section complements the plots presented in §4.1. In particular, we show the test and runtime results in table form, along with the behavior of the validation metrics and of the training loss over the training epochs. Plots for Multi-MNIST, CelebA, and Cityscapes are reported in Figures 7, 8 and 17, respectively.

The behavior of the CelebA training loss demonstrates heavier regularization (compare with the unregularized plot in Figure 9(a)). Except IMTL and MGDA, for which the tuned values of the weight decay prevent overfitting, the other optimizers display very similar validation and training curves, and start overfitting around epoch 30. Considering that most SMTOs required less regularization (see §4.1.2), the results are consistent with our interpretation of SMTOs as regularizers in §5. The Cityscapes plots display a certain instability across training epochs, as demonstrated by the various peaks and valleys in the metrics. Nevertheless, in spite of a factor 10 difference in scale, both training losses are similarly decreased by most optimizers.

## D.2 Unregularized Experiments

Figures 5, 9(a) and 9(b) respectively report the average task validation accuracy, the multi-task training loss, and the multi-task validation loss at each training epoch. The regularizing effect of SMTOs compared to unitary scalarization is shown by: (i) the delay of the onset of overfitting on the validation data in figure 5, (ii) the reduction of the convergence rate on the training loss in figure 9(a) (compare with figure 8(b)), and (iii) the fact that validation and training losses remain positively correlated for larger numbers of epochs. In fact, the behavior of both the train

ing and validation loss for the SMTOs closely parallels that of $\ell _ { 2 }$ -regularized unitary scalarization, with differing degrees of regularization. We further note that unregularized IMTL displays a certain instability (compare with the regularized version in figure 8(a)).

The addition of dropout layers further reduces overfitting, improves stability (reduced confidence intervals) and pushes the average validation curve upwards, motivating its use on all optimizers for the experiments of §4.1.2. Nevertheless, confidence intervals in Figure 5 still overlap due to the instability of the unregularized unitary scalarization. Figure 11 provides a more detailed comparison over 20 repetitions, confirming that the combined use of dropout layers and $\ell _ { 2 }$ regularization improves average performance and reduces the empirical variance for unitary scalarization. Furthermore, Figure 10 shows that regularization improves the peak average validation performance for all algorithms, demonstrating the need of tuning λ also for SMTOs. We conclude by pointing out that even without regularization, when carefully tuned, the maximal performance over epochs of unitary scalarization is comparable to SMTOs in Figure 5.

![](images/f75c556787df226fb07dcf3fcd2ec98879150f5b705da20b41388f37501ab59a.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Dini. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| 20 | ~0.942 | ~0.943 | ~0.941 | ~0.925 | ~0.941 | ~0.931 | ~0.932 |
| 40 | ~0.948 | ~0.949 | ~0.947 | ~0.933 | ~0.947 | ~0.939 | ~0.940 |
| 60 | ~0.950 | ~0.951 | ~0.950 | ~0.936 | ~0.950 | ~0.942 | ~0.943 |
| 80 | ~0.951 | ~0.952 | ~0.951 | ~0.937 | ~0.951 | ~0.943 | ~0.944 |
| 100 | ~0.951 | ~0.952 | ~0.951 | ~0.937 | ~0.951 | ~0.943 | ~0.944 |
</details>

(a) Mean (and 95% CI) average task validation accuracy per training epoch.

<table><tr><td>MTO</td><td>Average Task Accuracy</td><td>Epoch Runtime [s]</td></tr><tr><td>Unit. Scal.</td><td>9.476e-01 ± 4.368e-03</td><td>[3.510e+00, 3.617e+00]</td></tr><tr><td>IMTL</td><td>9.487e-01 ± 2.533e-03</td><td>[3.695e+00, 3.996e+00]</td></tr><tr><td>MGDA</td><td>9.478e-01 ± 1.977e-03</td><td>[3.491e+00, 3.617e+00]</td></tr><tr><td>GradDrop</td><td>9.347e-01 ± 1.282e-02</td><td>[3.508e+00, 3.589e+00]</td></tr><tr><td>PCGrad</td><td>9.479e-01 ± 3.578e-03</td><td>[3.807e+00, 3.928e+00]</td></tr><tr><td>RLW Diri.</td><td>9.430e-01 ± 2.973e-03</td><td>[3.790e+00, 4.005e+00]</td></tr><tr><td>RLW Norm.</td><td>9.399e-01 ± 8.929e-03</td><td>[3.894e+00, 4.225e+00]</td></tr></table>

(c) Mean and 95% CI of the avg. task test accuracy across runs, and interquartile range for the training time per epoch.

![](images/8a1d006d5789054a7b96da324a1ac085542f4108b6651bdeb95c7ad2139a4a02.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~3.2 | ~3.2 | ~3.2 | ~3.2 | ~3.2 | ~3.2 | ~3.2 |
| 20 | ~0.75 | ~0.65 | ~0.75 | ~1.0 | ~0.85 | ~0.85 | ~1.0 |
| 40 | ~0.65 | ~0.55 | ~0.65 | ~0.9 | ~0.75 | ~0.75 | ~0.9 |
| 60 | ~0.6 | ~0.55 | ~0.6 | ~0.85 | ~0.75 | ~0.75 | ~0.85 |
| 80 | ~0.55 | ~0.55 | ~0.6 | ~0.85 | ~0.75 | ~0.75 | ~0.85 |
| 100 | ~0.55 | ~0.55 | ~0.6 | ~0.85 | ~0.75 | ~0.75 | ~0.85 |
</details>

(b) Mean (and 95% CI) training multi-task loss $\mathcal { L } ^ { \mathrm { M T } }$ per epoch.  
Figure 7: Additional figures for the comparison of various SMTOs with the unitary scalarization on the MultiMNIST dataset [54].

![](images/c03e4c6a359d18a1a6edb31f0c3e3e9416c6cd03285f64e3b9d4dda20e096897.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Dir. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.890 | ~0.890 | ~0.890 | ~0.890 | ~0.890 | ~0.890 | ~0.890 |
| 10 | ~0.911 | ~0.907 | ~0.894 | ~0.912 | ~0.911 | ~0.912 | ~0.912 |
| 20 | ~0.913 | ~0.910 | ~0.902 | ~0.914 | ~0.913 | ~0.914 | ~0.914 |
| 30 | ~0.914 | ~0.913 | ~0.905 | ~0.914 | ~0.914 | ~0.914 | ~0.914 |
| 40 | ~0.913 | ~0.914 | ~0.906 | ~0.913 | ~0.913 | ~0.913 | ~0.913 |
| 50 | ~0.912 | ~0.914 | ~0.906 | ~0.912 | ~0.912 | ~0.912 | ~0.912 |
</details>

(a) Mean (and 95% CI) average task validation accuracy per training epoch.

<table><tr><td>MTO</td><td>Average Task Accuracy</td><td>Epoch Runtime [s]</td></tr><tr><td>Unit. Scal.</td><td>9.090e-01 ± 7.568e-04</td><td>[2.869e+02, 2.878e+02]</td></tr><tr><td>IMTL</td><td>9.093e-01 ± 7.631e-04</td><td>[3.600e+02, 3.621e+02]</td></tr><tr><td>MGDA</td><td>9.022e-01 ± 9.687e-04</td><td>[6.859e+02, 7.194e+02]</td></tr><tr><td>GradDrop</td><td>9.098e-01 ± 3.383e-04</td><td>[3.001e+02, 3.008e+02]</td></tr><tr><td>PCGrad</td><td>9.093e-01 ± 1.108e-03</td><td>[1.015e+04, 1.016e+04]</td></tr><tr><td>RLW Diri.</td><td>9.099e-01 ± 7.845e-04</td><td>[3.040e+02, 3.054e+02]</td></tr><tr><td>RLW Norm.</td><td>9.095e-01 ± 1.012e-03</td><td>[3.028e+02, 3.037e+02]</td></tr></table>

(c) Mean and 95% CI of the avg. task test accuracy across runs, and interquartile range for the training time per epoch.

![](images/8c6c59719a6ce3c06cd283d28d5945b92d9dce08838733d162e210c4307cdf37.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~13.8 | ~13.8 | ~13.8 | ~13.8 | ~13.8 | ~13.8 | ~13.8 |
| 10 | ~8.6 | ~9.2 | ~10.4 | ~8.5 | ~8.5 | ~8.5 | ~8.5 |
| 20 | ~8.0 | ~8.7 | ~9.8 | ~7.9 | ~7.9 | ~7.9 | ~7.9 |
| 30 | ~7.6 | ~8.5 | ~9.4 | ~7.3 | ~7.3 | ~7.3 | ~7.3 |
| 40 | ~7.2 | ~8.2 | ~9.1 | ~6.7 | ~6.7 | ~6.7 | ~6.7 |
| 50 | ~6.8 | ~8.1 | ~8.9 | ~6.1 | ~6.1 | ~6.1 | ~6.1 |
</details>

(b) Mean (and 95% CI) training multi-task loss $\mathcal { L } ^ { \mathrm { M T } }$ per epoch.  
Figure 8: Additional figures for the comparison of various SMTOs with the unitary scalarization on the CelebA [44] dataset.

## D.3 Sign-Agnostic GradDrop

We will now present an ablation study on GradDrop, investigating the effect of the sign of per-task gradients on the SMTO’s performance. Specifically, we compare the performance of GradDrop with a sign-agnostic version of its stochastic gradient masking (which we refer to as “Sign-Agnostic

![](images/3193e82a5aa4e1d14b76e039c62bbba2fa6ceac7235f5cc3a9eb2903dca45c8d.jpg)  
(a) Mean and 95% CI (3 runs) multi-task training loss (b) Mean and 95% CI (3 runs) multi-task validation per epoch. loss per training epoch.

Figure 9: Additional figures for the unregularized comparison of various SMTOs with the unitary scalarization on CelebA. SMTOs provide varying degrees of regularization.  
![](images/c9ab116412c45f3cdb109fc6d7225472c6939b655be01de464ea13d5c7136d57.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Dirl. Reg. |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.892 | ~0.892 | ~0.892 | ~0.892 | ~0.892 | ~0.892 |
| 10 | ~0.911 | ~0.906 | ~0.903 | ~0.911 | ~0.911 | ~0.911 |
| 20 | ~0.913 | ~0.907 | ~0.902 | ~0.914 | ~0.913 | ~0.914 |
| 30 | ~0.914 | ~0.908 | ~0.904 | ~0.914 | ~0.914 | ~0.914 |
| 40 | ~0.913 | ~0.908 | ~0.896 | ~0.913 | ~0.913 | ~0.913 |
| 50 | ~0.912 | ~0.908 | ~0.895 | ~0.912 | ~0.912 | ~0.912 |
</details>

Figure 10: Effect of regularization (dropout layers and weight decay) on the average task validation accuracy for all considered optimizers on the CelebA dataset: regularization improves the average performance of all algorithms.

![](images/1f21f6c36d7068dee6ea48b1d2555783baacb23238fffc34a9dcd1da91161ead.jpg)

<details>
<summary>violin</summary>

| Series | Median | Min | Max | Peak Range |
| --- | --- | --- | --- | --- |
| Unit. Scal. | ~0.913 | ~0.9125 | ~0.9144 | ~0.9128 ~0.9132 |
| Unit. Scal. Reg. | ~0.9146 | ~0.9142 | ~0.9149 | ~0.9145 ~0.9147 |
</details>

Figure 11: Effect of regularization (dropout layers and weight decay) on unitary scalarization on the CelebA dataset: violin plots (20 runs) for the best avg. task validation accuracy over epochs. The width at a given value represents the proportion of runs yielding that result. Regularization improves the average performance while decreasing its variability.

GradDrop"), whose update direction is defined as follows:

$$
\mathbf {g} = - \left(\frac {\partial \mathbf {z}}{\partial \boldsymbol {\theta} _ {\parallel}}\right) ^ {T} \left(\sum_ {i \in \mathcal {T}} \mathbf {u} _ {i} \odot \nabla_ {\mathbf {z}} \mathcal {L} _ {i}\right),
$$

where $\mathbf { u } _ { i } , \nabla _ { \mathbf { z } } \mathcal { L } _ { i } \in \mathbb { R } ^ { n \times r }$ and, for all $i \in \mathcal { T } , \mathbf { u } _ { i }$ is i.i.d. according to ${ \bf u } _ { i } [ j , k ] \sim$ Bernoulli(p) ∀j ∈ $\{ 1 , \dotsc , n \} , k \in \{ 1 , \dotsc , r \}$ . Differently from a similar study carried out by Chen et al. [11], we tuned the hyper-parameter of the sign-agnostic masking in the following range: $p \in$ {0.1, 0.25, 0.5, 0.75, 0.9}.

The experimental setup complies with the one described in appendix C.1. Figure 12, plotting test and validation results for the CelebA dataset [44], shows that the performance of Sign-Agnostic GradDrop closely matches the original algorithm. Therefore, sign conflicts across per-task gradients do not seem to play a significant role in GradDrop’s performance.

## E Supplementary Reinforcement Learning Experiments

## E.1 Addendum

This section presents additional plots for the RL experiments in §4.2. Specifically, Figure 13 re-plots Figure 4(a) and 4(b) with the omitted IMTL results, while Figure 14 shows the learning curves omitted from §4.2. As pointed out in §4.2, none of the IMTL runs successfully terminated due to numerical instability. Indeed, Liu et al. [42] show that, in supervised settings, coefficients do not fluctuate much across epochs [42, Figure 4, appendix B] and never become negative. By contrast, up to 50% of the scaling coefficients α are negative in our experiments, thus reversing subtask gradient directions. MGDA, which constrains the weights, is more stable and is comparable to unitary scalarization. In order to avoid incomplete curves and unfair calculations of the mean, Figure 14 plots the highest value ever achieved by any seed as a dashed horizontal line. The IMTL results in Figure 13, instead, report the best average success rate of each seed until its termination.

![](images/153e2821ed8801b778c562c9da850a0fb074eb8cba35275f73bd621c885ab1a2.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Sign-Agnostic GradDrop | ~0.9089 | ~0.9104 | ~0.9119 |
| GradDrop | ~0.9095 | ~0.9099 | ~0.9101 |
</details>

(a) Mean and 95% CI (3 runs) avg. task test accuracy.  
![](images/5b6455d966534775c1737831930c92420bd753233b30b16be235b0746b9f8dd6.jpg)

<details>
<summary>line</summary>

| Training Epochs | Sign-Agnostic GradDrop (Average Task Accuracy) | GradDrop (Average Task Accuracy) |
| --- | --- | --- |
| 0 | ~0.890 | ~0.890 |
| 10 | ~0.911 | ~0.911 |
| 20 | ~0.914 | ~0.914 |
| 30 | ~0.915 | ~0.914 |
| 40 | ~0.914 | ~0.913 |
| 50 | ~0.913 | ~0.911 |
</details>

(b) Mean and 95% CI (3 runs) avg. task validation accuracy per training epoch.

Figure 12: Comparison of GradDrop [11] with sign-agnostic masking of the shared-representation gradients on the CelebA dataset [44]. No statistically relevant difference between the two methods can be observed for the majority of the epochs.  
![](images/8abf6dd716647dd6ff1559376dac92b758ce2d483b1cd9caec17b14a71d5a2fb.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.85 | ~0.90 | ~0.95 |
| MGDA | ~0.85 | ~0.90 | ~0.95 |
| GradDrop | ~0.85 | ~0.90 | ~0.95 |
| PCGrad | ~0.80 | ~0.85 | ~0.90 |
| RLW Diri. | ~0.85 | ~0.90 | ~0.95 |
| RLW Norm. | ~0.85 | ~0.90 | ~0.95 |
| IMTL | ~0.00 | ~0.05 | ~0.10 |
</details>

(a) MT10 (10 runs per method).

![](images/d4c03b3dcf00078f986f98b3d1740367103b3540333711d476ecb170dd46e845.jpg)

<details>
<summary>boxplot</summary>

| Method | Q1 | Q2 (Median) | Q3 |
| --- | --- | --- | --- |
| Unit. Scal. | ~0.61 | ~0.64 | ~0.67 |
| MGDA | ~0.41 | ~0.48 | ~0.53 |
| GradDrop | ~0.54 | ~0.55 | ~0.56 |
| PCGrad | ~0.64 | ~0.67 | ~0.70 |
| RLW Diri. | ~0.56 | ~0.59 | ~0.62 |
| RLW Norm. | ~0.57 | ~0.59 | ~0.62 |
| IMTL | ~-0.02 | ~0.01 | ~0.05 |
</details>

(b) MT50 (10 runs per method).  
Figure 13: Mean and 95% CI for the best avg. success rate on Metaworld. None of the SMTOs significantly outperforms unitary scalarization.

![](images/a747d327e252b5933235bc64d32a424fff4bdc09213fc4cd1d9027b9b2d67d6a.jpg)

<details>
<summary>line</summary>

| Updates (1e6) | Unit. Scal. | GradDrop | RLW Diri. | IMTL | MGDA | PCGrad | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 |
| 0.25 | ~0.55 | ~0.55 | ~0.55 | 0.2 | ~0.55 | ~0.55 | ~0.55 |
| 0.50 | ~0.70 | ~0.70 | ~0.65 | 0.2 | ~0.70 | ~0.65 | ~0.70 |
| 0.75 | ~0.75 | ~0.75 | ~0.70 | 0.2 | ~0.75 | ~0.70 | ~0.75 |
| 1.00 | ~0.80 | ~0.80 | ~0.75 | 0.2 | ~0.80 | ~0.75 | ~0.80 |
| 1.25 | ~0.80 | ~0.80 | ~0.75 | 0.2 | ~0.80 | ~0.75 | ~0.80 |
| 1.50 | ~0.80 | ~0.80 | ~0.75 | 0.2 | ~0.80 | ~0.75 | ~0.80 |
| 1.75 | ~0.80 | ~0.80 | ~0.75 | 0.2 | ~0.80 | ~0.75 | ~0.80 |
| 2.00 | ~0.80 | ~0.80 | ~0.75 | 0.2 | ~0.80 | ~0.75 | ~0.80 |
</details>

(a) MT10 (10 points per method).

![](images/9590968de0ec798625e6db1484037894fb2470bf086c12b170636214fb15ffeb.jpg)

<details>
<summary>line</summary>

| Updates (1e6) | Unit. Scal. | GradDrop | RLW Diri. | MGDA | PCGrad | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.25 | ~0.32 | ~0.24 | ~0.31 | ~0.28 | ~0.36 | ~0.33 |
| 0.50 | ~0.44 | ~0.32 | ~0.42 | ~0.37 | ~0.47 | ~0.41 |
| 0.75 | ~0.49 | ~0.38 | ~0.46 | ~0.39 | ~0.53 | ~0.46 |
| 1.00 | ~0.52 | ~0.41 | ~0.49 | ~0.41 | ~0.56 | ~0.49 |
| 1.25 | ~0.54 | ~0.43 | ~0.52 | ~0.41 | ~0.58 | ~0.51 |
| 1.50 | ~0.56 | ~0.45 | ~0.54 | ~0.41 | ~0.61 | ~0.53 |
| 1.75 | ~0.58 | ~0.47 | ~0.56 | ~0.41 | ~0.63 | ~0.55 |
| 2.00 | ~0.58 | ~0.46 | ~0.54 | ~0.43 | ~0.64 | ~0.53 |
</details>

(b) MT50 (10 points per method).  
Figure 14: Mean and 95% CI for the avg. success rate on Metaworld. None of the SMTOs significantly outperforms unitary scalarization.

## E.2 Ablation studies

Figure 18 presents our ablations for MT10 experiments. Due to computational constraints, we ran ablations on the unitary scalarization and PCGrad since these are the two methods previously tested in the RL setting.

Figure 15 shows ablation studies on the effect of regularization on MT10 and MT50. In spite of CI overlaps, actor l regularization pushes the average higher on both benchmarks, motivating our use of regularization for the experiments in §4.2. Furthermore, the gap between the averages tends to widen with the number of updates on MT50, suggesting improved stabilization.

## E.3 Sensitivity to Reward Normalization

Figure 16 shows that multitask agent performance is highly sensitive to the reward normalization moving average hyperparameter<sup>4</sup> motivating our buffer normalization in Section 4.2.

![](images/7fbc003ed6fd5eff2c647a273cd0374efb236ebd3f2e9a69f6a626e50b4a5334.jpg)

<details>
<summary>line</summary>

| Updates (1e6) | Unit. Scal. reg (Average success rate) | Unit. Scal. no reg (Average success rate) |
| --- | --- | --- |
| 0.00 | 0.00 | 0.00 |
| 0.25 | ~0.58 | ~0.52 |
| 0.50 | ~0.68 | ~0.62 |
| 0.75 | ~0.76 | ~0.66 |
| 1.00 | ~0.76 | ~0.74 |
| 1.25 | ~0.78 | ~0.74 |
| 1.50 | ~0.81 | ~0.74 |
| 1.75 | ~0.82 | ~0.76 |
| 2.00 | ~0.82 | ~0.72 |
</details>

(a) MT10 average performance (10 runs) and 95% CI.

![](images/6da0160bb1d7c31e761f00a7ed544b41691e3fc8b8c01c2e346ced3810596509.jpg)

<details>
<summary>line</summary>

| Updates (1e6) | Unit. Scal. reg (Average success rate) | Unit. Scal. no reg. (Average success rate) |
| --- | --- | --- |
| 0.00 | 0.00 | 0.00 |
| 0.25 | ~0.30 | ~0.28 |
| 0.50 | ~0.42 | ~0.35 |
| 0.75 | ~0.48 | ~0.42 |
| 1.00 | ~0.51 | ~0.49 |
| 1.25 | ~0.53 | ~0.48 |
| 1.50 | ~0.56 | ~0.51 |
| 1.75 | ~0.58 | ~0.49 |
| 2.00 | ~0.58 | ~0.46 |
</details>

(b) MT50 average performance (5 runs) and 95% CI.  
Figure 15: For both MT10 and MT50, actor $l _ { 2 }$ regularization pushes the average higher for unitary scalarization.

![](images/859ff0d8d427725b454edd43e1c060f0e143fe7b87b0e025e9a2f5de2d31caac.jpg)

<details>
<summary>line</summary>

| Updates (1e6) | \(\alpha = 0.0001\) | \(\alpha = 0.0005\) | \(\alpha = 0.005\) | \(\alpha = 0.01\) | \(\alpha = 0.001 (\)default) | No reward normalisation |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.25 | ~0.45 | ~0.35 | ~0.15 | ~0.10 | ~0.38 | ~0.22 |
| 0.50 | ~0.75 | ~0.75 | ~0.38 | ~0.22 | ~0.55 | ~0.28 |
| 0.75 | ~0.72 | ~0.72 | ~0.45 | ~0.32 | ~0.62 | ~0.32 |
| 1.00 | ~0.72 | ~0.72 | ~0.48 | ~0.35 | ~0.62 | ~0.35 |
| 1.25 | ~0.75 | ~0.65 | ~0.48 | ~0.35 | ~0.62 | ~0.42 |
| 1.50 | ~0.72 | ~0.58 | ~0.42 | ~0.32 | ~0.62 | ~0.42 |
| 1.75 | ~0.72 | ~0.52 | ~0.35 | ~0.32 | ~0.62 | ~0.42 |
| 2.00 | ~0.65 | ~0.45 | ~0.35 | ~0.32 | ~0.58 | ~0.45 |
</details>

Figure 16: The learning outcomes of a Multitask SAC agent vary considerably depending on the reward normalisation hyperparameter. Each of the curves represents and average of 10 runs with shaded 95% confidence interval.

(a) Mean and 95% CI of the test metrics across runs, and interquartile range for the training time per epoch.

<table><tr><td>MTO</td><td>Absolute Depth Error</td><td>Relative Depth Error</td><td>Segmentation Accuracy</td><td>Segmentation mIOU</td><td>Epoch Runtime [s]</td></tr><tr><td>Unit. Scal.</td><td> $1.301e-02 \pm 2.342e-04$ </td><td> $4.761e+01 \pm 5.148e+00$ </td><td> $9.196e-01 \pm 2.913e-04$ </td><td> $7.012e-01 \pm 6.001e-04$ </td><td> $[3.228e+02, 3.241e+02]$ </td></tr><tr><td>IMTL</td><td> $1.281e-02 \pm 7.521e-04$ </td><td> $4.389e+01 \pm 6.984e-01$ </td><td> $9.164e-01 \pm 2.828e-03$ </td><td> $6.967e-01 \pm 4.785e-03$ </td><td> $[7.329e+02, 7.373e+02]$ </td></tr><tr><td>MGDA</td><td> $1.418e-02 \pm 2.331e-04$ </td><td> $4.750e+01 \pm 1.466e+01$ </td><td> $9.189e-01 \pm 2.636e-04$ </td><td> $6.999e-01 \pm 3.124e-03$ </td><td> $[7.251e+02, 7.269e+02]$ </td></tr><tr><td>GradDrop</td><td> $1.293e-02 \pm 2.757e-04$ </td><td> $4.674e+01 \pm 7.709e+00$ </td><td> $9.193e-01 \pm 1.282e-03$ </td><td> $7.024e-01 \pm 3.628e-03$ </td><td> $[5.196e+02, 5.215e+02]$ </td></tr><tr><td>PCGrad</td><td> $1.294e-02 \pm 2.284e-04$ </td><td> $4.380e+01 \pm 5.165e+00$ </td><td> $9.198e-01 \pm 9.119e-04$ </td><td> $7.025e-01 \pm 6.531e-04$ </td><td> $[4.202e+02, 4.212e+02]$ </td></tr><tr><td>RLW Diri.</td><td> $1.305e-02 \pm 4.155e-04$ </td><td> $4.810e+01 \pm 2.259e+00$ </td><td> $9.199e-01 \pm 1.247e-03$ </td><td> $7.037e-01 \pm 1.989e-03$ </td><td> $[3.161e+02, 3.164e+02]$ </td></tr><tr><td>RLW Norm.</td><td> $1.301e-02 \pm 5.528e-04$ </td><td> $4.630e+01 \pm 2.751e+00$ </td><td> $9.192e-01 \pm 4.962e-04$ </td><td> $7.006e-01 \pm 4.580e-03$ </td><td> $[3.194e+02, 3.210e+02]$ </td></tr></table>

![](images/3adc3f5f9c981ac913f7f1189832b20942cff504f988c4e16989edb4b9c2cfc1.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.026 | ~0.026 | ~0.039 | ~0.026 | ~0.026 | ~0.026 | ~0.050 |
| 25 | ~0.014 | ~0.014 | ~0.026 | ~0.014 | ~0.014 | ~0.014 | ~0.014 |
| 55 | ~0.014 | ~0.014 | ~0.014 | ~0.014 | ~0.014 | ~0.014 | ~0.014 |
| 85 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 |
| 100 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 | ~0.013 |
</details>

(b) Mean (and 95% CI) absolute depth validation error per training epoch.

![](images/4322f0a587e5ff27cfd2e01b41280b1e475cdbb1418880dd9741e8164494e620.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~65 | ~75 | ~75 | ~75 | ~75 | ~75 | ~75 |
| 10 | ~38 | ~42 | ~48 | ~35 | ~38 | ~42 | ~42 |
| 20 | ~35 | ~32 | ~38 | ~32 | ~35 | ~38 | ~38 |
| 30 | ~35 | ~32 | ~38 | ~32 | ~35 | ~38 | ~38 |
| 40 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 50 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 60 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 70 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 80 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 90 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
| 100 | ~32 | ~32 | ~32 | ~30 | ~32 | ~32 | ~32 |
</details>

(c) Mean (and 95% CI) relative depth validation error per training epoch.

![](images/cf87bab488b5ca669c45a7d9ddd6ea4d38ab002890b5ac68fc93181b103df914.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.68 | ~0.68 | ~0.68 | ~0.68 | ~0.68 | ~0.68 | ~0.68 |
| 20 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
| 40 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
| 60 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
| 80 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
| 100 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
</details>

(d) Mean (and 95% CI) validation segmentation mIOU per training epoch.

![](images/d66d78fc57dc62538aec7487d1b7b04bed849262002a17df1a9d07ec515499c6.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.895 | ~0.885 | ~0.885 | ~0.885 | ~0.895 | ~0.885 | ~0.885 |
| 20 | ~0.930 | ~0.925 | ~0.930 | ~0.930 | ~0.930 | ~0.930 | ~0.930 |
| 40 | ~0.930 | ~0.925 | ~0.930 | ~0.930 | ~0.930 | ~0.930 | ~0.930 |
| 60 | ~0.930 | ~0.925 | ~0.930 | ~0.930 | ~0.930 | ~0.930 | ~0.930 |
| 80 | ~0.930 | ~0.925 | ~0.930 | ~0.930 | ~0.930 | ~0.930 | ~0.930 |
| 100 | ~0.930 | ~0.925 | ~0.930 | ~0.930 | ~0.930 | ~0.930 | ~0.930 |
</details>

(e) Mean (and 95% CI) validation segmentation accuracy per training epoch.

![](images/4e14d75283ec2874a360df20e9988662b13dba6bc68aa9d5b948d255688412a5.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| 20 | ~0.14 | ~0.13 | ~0.15 | ~0.13 | ~0.14 | ~0.14 | ~0.14 |
| 40 | ~0.12 | ~0.12 | ~0.16 | ~0.12 | ~0.12 | ~0.12 | ~0.12 |
| 60 | ~0.11 | ~0.11 | ~0.12 | ~0.11 | ~0.11 | ~0.11 | ~0.11 |
| 80 | ~0.11 | ~0.11 | ~0.12 | ~0.11 | ~0.11 | ~0.11 | ~0.11 |
| 100 | ~0.10 | ~0.10 | ~0.12 | ~0.10 | ~0.10 | ~0.10 | ~0.10 |
</details>

(f) Mean (and 95% CI) training depth loss per epoch.

![](images/ed350ee83a24e16a4cb31b3c177386ea9e31d410a8cbb17335c34d0c996be590.jpg)

<details>
<summary>line</summary>

| Training Epochs | Unit. Scal. | IMTL | MGDA | GradDrop | PCGrad | RLW Diri. | RLW Norm. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.38 | ~0.38 | ~0.38 | ~0.38 | ~0.38 | ~0.38 | ~0.38 |
| 20 | ~0.12 | ~0.15 | ~0.10 | ~0.12 | ~0.12 | ~0.12 | ~0.12 |
| 40 | ~0.10 | ~0.15 | ~0.10 | ~0.10 | ~0.10 | ~0.10 | ~0.10 |
| 60 | ~0.10 | ~0.13 | ~0.10 | ~0.10 | ~0.10 | ~0.10 | ~0.10 |
| 80 | ~0.10 | ~0.13 | ~0.12 | ~0.10 | ~0.10 | ~0.10 | ~0.10 |
| 100 | ~0.08 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 | ~0.12 |
</details>

(g) Mean (and 95% CI) training segmentation loss per epoch.  
Figure 17: Additional figures for the comparison of SMTOs with the unitary scalarization on the Cityscapes [13] dataset.

![](images/e8414ca97d426de63fbd705242816caec73ea9dd38fd2bc35babb07e117984a5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["&quot;Hyperparameters from [Sodhani et al, 2021"]"] --> B["Use PCGrad parameters for the baseline (shared alpha)"]
  B --> C["Increase replay buffer size"]
  C --> D["Normalise reward at the buffer level"]
  D --> E["Regularize the actor"]
```
</details>

Figure 18: Metaworld’s MT10 ablation experiments.