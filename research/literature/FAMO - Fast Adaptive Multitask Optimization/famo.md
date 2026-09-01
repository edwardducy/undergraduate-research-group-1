# FAMO: Fast Adaptive Multitask Optimization

†Bo Liu, ‡Yihao Feng, †,§Peter Stone, †Qiang Liu

†The University of Texas at Austin, ‡Salesforce AI Research, §Sony AI {bliu, pstone, lqiang}@cs.utexas.edu, yihaof@salesforce.com

# Abstract

One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, in practice, applying gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients (O(k) space and time where k is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (FAMO), a dynamic weighting method that decreases task losses in a balanced way using O(1) space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that FAMO achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency. Code is available at <https://github.com/Cranial-XIX/FAMO>.

# 1 Introduction

Large models trained on diverse data have advanced both computer vision [\[20\]](#page-11-0) and natural language processing [\[4\]](#page-10-0), paving the way for generalist agents capable of multitask learning (MTL) [\[5\]](#page-10-1). Given the substantial size of these models, it is crucial to design MTL methods that are *effective* in terms of task performance and *efficient* in terms of space and time complexities for managing training costs and environmental impacts. This work explores such methods through the lens of optimization.

Perhaps the most intuitive way of solving an MTL problem is to optimize the average loss across all tasks. However, in practice, doing so can lead to models with poor multitask performance: a subset of tasks are *severely under-optimized*. A major reason behind such optimization failure is that a subset of tasks are under-optimized because the average gradient constantly results in small (or even negative) progress on these tasks (see details in Section [2\)](#page-1-0).

To mitigate this problem, gradient manipulation methods [\[43,](#page-12-0) [25,](#page-11-1) [7,](#page-10-2) [24\]](#page-11-2) compute a new update vector in place of the gradient to the average loss, such that all task losses decrease in a more balanced way. The new update vector is often determined by solving an additional optimization problem that involves all task gradients. While these approaches exhibit improved performance, they become computationally expensive when the number of tasks and the model size are large [\[41\]](#page-12-1). This is because they require computing and storing all task gradients at each iteration, thus demanding O(k) space and time complexities, not to mention the overhead introduced by solving the additional optimization problem. In contrast, the average gradient can be efficiently computed in O(1) space and time per iteration because one can first average the task losses and then take the gradient of the average loss.[<sup>1</sup>](#page-0-0) To this end, we ask the following question:

<span id="page-0-0"></span><sup>1</sup>Here, we refer to the situation where a single data x can be used to compute all task losses.

<span id="page-1-3"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Top left: The loss landscape, and individual task losses of a toy 2-task learning problem (★ represents the minimum of task losses). Top right: the runtime of different MTL methods for 50000 steps. Bottom: the loss trajectories of different MTL methods. ADAM fails in 1 out of 5 runs to reach the Pareto front due to CG. FAMO decreases task losses in a balanced way and is the only method matching the O(1) space/time complexity of ADAM. Experimental details and analysis are provided in Section [5.1.](#page-6-0)

### (Q) *Is it possible to design a multi-task learning optimizer that ensures a balanced reduction in losses across all tasks while utilizing* O(1) *space and time per iteration?*

In this work, we present Fast Adaptive Multitask Optimization (FAMO), a simple yet effective adaptive task weighting method to address the above question. On the one hand, FAMO is designed to ensure that all tasks are optimized with approximately similar progress. On the other hand, FAMO leverages the loss history to update the task weighting, hence bypassing the necessity of computing all task gradients. To summarize, our contributions are:

- 1. We introduce FAMO, an MTL optimizer that decreases task losses approximately at *equal rates* while using only O(1) space and time per iteration.
- 2. We demonstrate that FAMO performs comparably to or better than existing gradient manipulation methods on a wide range of standard MTL benchmarks, in terms of standard MTL metrics, while being significantly computationally cheaper.

### 2 Background

In this section, we provide the formal definition of multitask learning, then discuss its optimization challenge, and provide a brief overview of the gradient manipulation methods.

Multitask Learning (MTL) MTL considers optimizing a *single* model with parameter <sup>θ</sup> ∈ <sup>R</sup> <sup>m</sup> that can perform <sup>k</sup> ≥ <sup>2</sup> tasks well, where each task is associated with a loss function <sup>ℓ</sup>i(θ) ∶ <sup>R</sup> <sup>m</sup> → <sup>R</sup>≥<sup>0</sup>. [2](#page-1-1) Then, it is common to optimize the average loss across all tasks:

<span id="page-1-2"></span>min θ∈R<sup>m</sup> {<sup>ℓ</sup>0(θ) ∶= 1 k k ∑ i=1 <sup>ℓ</sup>i(θ)} . (1)

<span id="page-1-0"></span>Optimization Challenge Directly optimizing [\(1\)](#page-1-2) can result in severe under-optimization of a subset of tasks. A major reason behind this optimization challenge is the "generalized" conflicting gradient phenomenon, which we explain in the following. At any time step t, assume one updates the model

<span id="page-1-1"></span><sup>2</sup> In this work, we assume ∀ i, ℓi(θ) <sup>≥</sup> 0, which is true for typical loss functions including mean square and cross-entropy losses. Note that one can always transform ℓ<sup>i</sup> to be non-negative if a loss lower bound is known.

### Algorithm 1 Fast Adaptive Multitask Optimization (FAMO)

1: Input: Initial parameter <sup>θ</sup>0, task losses {<sup>ℓ</sup>i} k i=1 (ensure that <sup>ℓ</sup><sup>i</sup> ≥ <sup>ϵ</sup> > <sup>0</sup>, for instance, by <sup>ℓ</sup><sup>i</sup> ← <sup>ℓ</sup><sup>i</sup> − <sup>ℓ</sup> ∗ <sup>i</sup> + <sup>ϵ</sup>, <sup>ℓ</sup> ∗ <sup>i</sup> = inf<sup>θ</sup> <sup>ℓ</sup>i(θ)), learning rate <sup>α</sup> and <sup>β</sup>, and decay <sup>γ</sup> (= <sup>0</sup>.<sup>001</sup> by default). 2: <sup>ξ</sup><sup>1</sup> ← <sup>0</sup>. // initialize the task logits to all zeros 3: for <sup>t</sup> = <sup>1</sup> ∶ <sup>T</sup> do 4: Compute <sup>z</sup><sup>t</sup> = Softmax(<sup>ξ</sup>t), e.g.,

<sup>z</sup>i,t = exp(<sup>ξ</sup>i,t) ∑ k i=1 exp(<sup>ξ</sup>i,t) .

5: Update the model parameters:

<sup>θ</sup><sup>t</sup>+<sup>1</sup> = <sup>θ</sup><sup>t</sup> − <sup>α</sup> k ∑ i=1 (ct zi,t ℓi,t )∇<sup>ℓ</sup>i,t, where <sup>c</sup><sup>t</sup> = ( k ∑ i=1 zi,t ℓi,t ) <sup>−</sup>1 .

6: Update the logits for task weighting:

<sup>ξ</sup><sup>t</sup>+<sup>1</sup> = <sup>ξ</sup><sup>t</sup> − <sup>β</sup>(<sup>δ</sup><sup>t</sup> + γξt) where <sup>δ</sup><sup>t</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ ∇ ⊺ <sup>z</sup>1,t(<sup>ξ</sup>t) ⋮ ∇ ⊺ <sup>z</sup>k,t(<sup>ξ</sup>t) ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ ⊺ ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ log <sup>ℓ</sup>1,t − log <sup>ℓ</sup>1,t+<sup>1</sup> ⋮ log <sup>ℓ</sup>k,t − log <sup>ℓ</sup>k,t+<sup>1</sup>. ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ .

#### <span id="page-2-1"></span>7: end for

parameter using a gradient descent style iterative update: <sup>θ</sup><sup>t</sup>+<sup>1</sup> = <sup>θ</sup><sup>t</sup> − αd<sup>t</sup> where <sup>α</sup> is the step size and d<sup>t</sup> is the update at time t. Then, we say that conflicting gradients (CG) [\[24,](#page-11-2) [43\]](#page-12-0) happens if

∃i, ℓi(<sup>θ</sup><sup>t</sup>+<sup>1</sup>) − <sup>ℓ</sup>i(<sup>θ</sup>t) ≈ −α∇<sup>ℓ</sup>i(<sup>θ</sup>t) ⊺ <sup>d</sup><sup>t</sup> > <sup>0</sup>.

In other words, certain task's loss is increasing. CG often occurs during optimization and is not inherently detrimental. However, it becomes undesirable when a subset of tasks persistently undergoes under-optimization due to CG. In a more general sense, it is not desirable if a subset of tasks has much slower learning progress compared to the rest of the tasks (even if all task losses are decreasing). This very phenomenon, which we call the "generalized" conflicting gradient, has spurred previous research to mitigate it at each optimization stage [\[43\]](#page-12-0).

Gradient Manipulation Methods Gradient manipulation methods aim to decrease all task losses in a more balanced way by finding a new update d<sup>t</sup> at each step. d<sup>t</sup> is usually a convex combination of task gradients, and therefore the name gradient manipulation (denote ∇<sup>ℓ</sup>i,t = ∇θℓi(<sup>θ</sup>t) for short):

<span id="page-2-0"></span><sup>d</sup><sup>t</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣ ∇ℓ ⊺ 1,t ⋮ ∇ℓ ⊺ k,t ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦ ⊺ <sup>w</sup>t, where <sup>w</sup><sup>t</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ w1,t ⋮ wk,t ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ = <sup>f</sup>(∇<sup>ℓ</sup>1,t, . . . , ∇<sup>ℓ</sup>k,t) ∈ <sup>S</sup>k. (2)

Here, <sup>S</sup><sup>k</sup> = {<sup>w</sup> ∈ <sup>R</sup> k ≥0 ∣ w <sup>⊺</sup><sup>1</sup> = <sup>1</sup>} is the probabilistic simplex, and <sup>w</sup><sup>t</sup> is the task weighting across all tasks. Please refer to Appendix [A](#page-13-0) for details of five state-of-the-art gradient manipulation methods (MGDA, PCGRAD, CAGRAD, IMTL-G, NASHMTL) and their corresponding f. Note that existing gradient manipulation methods require computing and storing k task gradients before applying f to compute dt, which often involves solving an additional optimization problem. As a result, we say these methods require at least O(k) space and time complexity, which makes them slow and memory inefficient when k and model size m are large.

# 3 Fast Adaptive Multitask Optimization (FAMO)

In this section, we introduce FAMO that addresses question Q, which involves two main ideas:

- 1. At each step, decrease all task losses at *an equal rate* as much as possible (Section [3.1\)](#page-3-0).
- 2. Amortize the computation in 1. over time (Section [3.2\)](#page-4-0).

### <span id="page-3-0"></span>3.1 Balanced Rate of Loss Improvement

At time <sup>t</sup>, assume we perform the update <sup>θ</sup><sup>t</sup>+<sup>1</sup> = <sup>θ</sup><sup>t</sup> − αdt, we define the rate of improvement for task i as

<sup>r</sup>i(α, dt) = <sup>ℓ</sup>i,t − <sup>ℓ</sup>i,t+<sup>1</sup> ℓi,t . [3](#page-3-1) (3)

FAMO then seeks an update d<sup>t</sup> that results in the largest *worst-case improvement rate* across all tasks ( 1 2 ∥<sup>d</sup>t∥ is subtracted to prevent an under-specified optimization problem where the objective can be infinitely large):

<span id="page-3-2"></span>max dt∈R<sup>m</sup> min i∈[k] 1 α <sup>r</sup>i(α, dt) − 1 2 ∥<sup>d</sup>t∥ 2 . (4)

When the step size α is small, using Taylor approximation, the problem [\(4\)](#page-3-2) can be approximated by

<span id="page-3-3"></span>max dt∈R<sup>m</sup> min i∈[K] ∇ℓ ⊺ i,td<sup>t</sup> ℓi,t − 1 2 ∥<sup>d</sup>t∥ 2 = (∇log <sup>ℓ</sup>i,t) ⊺ <sup>d</sup><sup>t</sup> − 1 2 ∥<sup>d</sup>t∥ 2 . (5)

Instead of solving the primal problem in [\(5\)](#page-3-3) where <sup>d</sup> ∈ <sup>R</sup> <sup>m</sup> (m can be millions if θ is the parameter of a neural network), we consider its dual problem:

Proposition 3.1. *The dual objective of* [\(5\)](#page-3-3) *is*

<span id="page-3-4"></span>z ∗ t ∈ arg min z∈S<sup>k</sup> 1 2 ∥<sup>J</sup>t<sup>z</sup>∥ , *where* <sup>J</sup><sup>t</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣ ∇log <sup>ℓ</sup> ⊺ 1,t ⋮ ∇log <sup>ℓ</sup> ⊺ k,t ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦ , (6)

*where* z ∗ <sup>t</sup> = [<sup>z</sup> ∗ t,i] *is the optimal combination weights of the gradients, and the optimal update direction is* d ∗ <sup>t</sup> = <sup>J</sup>t<sup>z</sup> ∗ t *.*

*Proof.*

max d∈R<sup>m</sup> min i∈[k] (∇log <sup>ℓ</sup>i,t) ⊺ d − 1 2 ∥d∥ 2 = max d∈R<sup>m</sup> min z∈S<sup>k</sup> ( k ∑ i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t) ⊺ d − 1 2 ∥d∥ 2 = min z∈S<sup>k</sup> max d∈R<sup>m</sup> ( k ∑ i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t) ⊺ d − 1 2 ∥d∥ 2 (strong duality)

Write <sup>g</sup>(d, z) = ( ∑ k i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t) ⊺ d − 2 ∥d∥ 2 , then by setting

∂g ∂d = <sup>0</sup> Ô⇒ <sup>d</sup> ∗ = k ∑ i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t.

Plugging in d ∗ back, we have

max d∈R<sup>m</sup> min i∈[k] (∇log <sup>ℓ</sup>i,t) ⊺ d − 1 2 ∥d∥ 2 = min z∈S<sup>k</sup> 1 2 ∥ k ∑ i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t∥ 2 = min z∈S<sup>k</sup> 1 2 ∥<sup>J</sup>t<sup>z</sup>∥ 2 .

At the optimum, we have d ∗ <sup>t</sup> = <sup>J</sup>t<sup>z</sup> ∗ t .

The dual problem in [\(6\)](#page-3-4) can be viewed as optimizing the log objective of the multiple gradient descent algorithm (MGDA) [\[9,](#page-10-3) [35\]](#page-11-3). Similar to MGDA, [\(6\)](#page-3-4) only involves a decision variable of dimension <sup>k</sup> ≪ <sup>m</sup>. Furthermore, if the optimal combination weights <sup>z</sup> ∗ t is an interior point of Sk, then the improvement rates <sup>r</sup>i(α, d<sup>∗</sup> t ) of the different tasks <sup>i</sup> equal, as we show in the following result.

<span id="page-3-1"></span><sup>3</sup>To avoid division by zero, in practice we add a small constant (e.g., 1e <sup>−</sup> 8) to all losses. For the ease of notation (e.g., ℓi(⋅) <sup>←</sup> ℓi(⋅) <sup>+</sup> 1e <sup>−</sup> 8, we omit it throughout the paper.

Proposition 3.2. *Assume* {<sup>ℓ</sup>i} k i=1 *are smooth and the optimal weights* z ∗ t *in* [\(6\)](#page-3-4) *is an interior point of* Sk*, then*

∀ <sup>i</sup> ≠ <sup>j</sup> ∈ [k], r<sup>∗</sup> i (d ∗ t ) = <sup>r</sup> ∗ j (d ∗ t ),

*where* r ∗ i (d ∗ t ) = lim<sup>α</sup>→<sup>0</sup> α <sup>r</sup>i(α, d<sup>∗</sup> t ).

*Proof.* Consider the Lagrangian form of [\(6\)](#page-3-4)

L(z, λ, µ) = 1 2 k ∑ i=1 <sup>z</sup>i∇log <sup>ℓ</sup>i,t∥ 2 + <sup>λ</sup>( k ∑ i=1 <sup>z</sup><sup>i</sup> − <sup>1</sup>) − k ∑ i=1 µiz<sup>i</sup> , where ∀i, µ<sup>i</sup> ≥ <sup>0</sup>. (7)

When z ∗ reaches the optimum, we have <sup>∂</sup>L(z, λ, µ)/∂z = <sup>0</sup>, recall that <sup>d</sup> ∗ <sup>t</sup> = <sup>J</sup>t<sup>z</sup> ∗ t , then

J ⊺ <sup>t</sup> Jtz ∗ = −<sup>µ</sup> − λ, where <sup>J</sup><sup>t</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣ ∇log <sup>ℓ</sup> ⊺ 1,t ⋮ ∇log <sup>ℓ</sup> ⊺ k,t ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦ Ô⇒ <sup>J</sup> ⊺ <sup>t</sup> d ∗ <sup>t</sup> = −(<sup>µ</sup> + <sup>λ</sup>).

When z ∗ t is an interior point of <sup>S</sup>k, we know that <sup>µ</sup> = <sup>0</sup>. Hence <sup>J</sup> ⊺ t d ∗ <sup>t</sup> = −<sup>λ</sup>. This means,

∀<sup>i</sup> ≠ j, lim<sup>α</sup>→<sup>0</sup> 1 α <sup>r</sup>i(α, d<sup>∗</sup> t ) = ∇log <sup>ℓ</sup> ⊺ i,td ∗ <sup>t</sup> = ∇log <sup>ℓ</sup> ⊺ j,td ∗ <sup>t</sup> = lim<sup>α</sup>→<sup>0</sup> 1 α <sup>r</sup><sup>j</sup> (α, d<sup>∗</sup> t ).

### <span id="page-4-0"></span>3.2 Fast Approximation by Amortizing over Time

Instead of fully solving [\(6\)](#page-3-4) at each optimization step, FAMO performs a single-step gradient descent on z, which amortizes the computation over the optimization trajectory:

<span id="page-4-1"></span><sup>z</sup><sup>t</sup>+<sup>1</sup> = <sup>z</sup><sup>t</sup> − <sup>α</sup><sup>z</sup> ˜δ, where ˜<sup>δ</sup> = ∇<sup>z</sup> 1 2 ∥ k ∑ i=1 <sup>z</sup>i,t∇log <sup>ℓ</sup>i,t∥ 2 = J ⊺ <sup>t</sup> Jtzt. (8)

But then, note that

1 α ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ log <sup>ℓ</sup>1,t − log <sup>ℓ</sup>1,t+<sup>1</sup> ⋮ log <sup>ℓ</sup>k,t − log <sup>ℓ</sup>k,t+<sup>1</sup> ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ ≈ J ⊺ <sup>t</sup> <sup>d</sup><sup>t</sup> = <sup>J</sup> ⊺ <sup>t</sup> Jtzt, (9)

so we can use the change in log losses to approximate the gradient.

In practice, to ensure that <sup>z</sup> always stays in <sup>S</sup>k, we re-parameterize <sup>z</sup> by <sup>ξ</sup> and let <sup>z</sup><sup>t</sup> = Softmax(<sup>ξ</sup>t), where <sup>ξ</sup><sup>t</sup> ∈ <sup>R</sup> <sup>K</sup> are the unconstrained softmax logits. Consequently, we have the following approximate update on ξ from [\(8\)](#page-4-1):

<span id="page-4-2"></span><sup>ξ</sup><sup>t</sup>+<sup>1</sup> = <sup>ξ</sup><sup>t</sup> − βδ, where <sup>δ</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ ∇ ⊺ <sup>z</sup>1,t(ξ) ⋮ ∇ ⊺ <sup>z</sup>k,t(ξ) ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ ⊺ ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ log <sup>ℓ</sup>1,t − log <sup>ℓ</sup>1,t+<sup>1</sup> ⋮ log <sup>ℓ</sup>k,t − log <sup>ℓ</sup>k,t+<sup>1</sup> ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ . (10)

Remark: While it is possible to perform gradient descent on z for other gradient manipulation methods in principle, we will demonstrate in Appendix [B](#page-14-0) that not all such updates can be easily approximated using the change in losses.

### 3.3 Practical Implementation

To facilitate practical implementation, we present two modifications to the update in [\(10\)](#page-4-2).

Re-normalization The suggested update above is a convex combination of the gradients of the log loss, e.g.,

d ∗ = k ∑ i=1 <sup>z</sup>i,t∇log <sup>ℓ</sup>i,t = k ∑ i=1 ( zi,t ℓi,t )∇<sup>ℓ</sup>i,t.

When <sup>ℓ</sup>i,t is small, the multiplicative coefficient <sup>z</sup>i,t ℓi,t can be quite large and result in unstable optimization. Therefore, we propose to multiply d ∗ by a constant ct, such that ctd ∗ can be written as a convex combination of the task gradients just as in other gradient manipulation algorithms (see [\(2\)](#page-2-0) and we provide the corresponding definition of w in the following):

<sup>c</sup><sup>t</sup> = ( k ∑ i=1 zi,t ℓi,t ) <sup>−</sup>1 and <sup>d</sup><sup>t</sup> = <sup>c</sup>t<sup>d</sup> ∗ = k ∑ i=1 <sup>w</sup>i∇<sup>ℓ</sup>i,t, where <sup>w</sup><sup>i</sup> = <sup>c</sup><sup>t</sup> zi,t ℓi,t . (11)

Regularization As we are amortizing the computation over time and the loss objective {<sup>ℓ</sup>i(⋅)}<sup>s</sup> are changing dynamically, it makes sense to focus more on the recent updates of ξ [\[46\]](#page-12-2). To this end, we put a decay term on w such that the resulting ξ<sup>t</sup> is an exponential moving average of its gradient updates:

<sup>ξ</sup><sup>t</sup>+<sup>1</sup> = <sup>ξ</sup><sup>t</sup> − <sup>β</sup>(<sup>δ</sup><sup>t</sup> + γξt) = −β(<sup>δ</sup><sup>t</sup> + (<sup>1</sup> − βγ)<sup>δ</sup><sup>t</sup>−<sup>1</sup> + (<sup>1</sup> − βγ) <sup>δ</sup><sup>t</sup>−<sup>2</sup> + . . . ). (12)

We provide the complete FAMO algorithm in Algorithm [1](#page-2-1) and its pseudocode in Appendix [C.](#page-15-0)

### 3.4 The Continuous Limit of FAMO

One way to characterize FAMO's behavior is to understand the stationary points of the continuoustime limit of FAMO (i.e. when step sizes (α, β) shrink to zero). From Algorithm [1,](#page-2-1) one can derive the following non-autonomous dynamical system (assuming {<sup>ℓ</sup>i} are all smooth):

<span id="page-5-0"></span>[ ˙θ ˙ξ ] <sup>=</sup> <sup>−</sup><sup>c</sup><sup>t</sup> [ Jtz<sup>t</sup> AtJ ⊺ <sup>t</sup> <sup>J</sup>tz<sup>t</sup> + γ ct ξt ] , where <sup>A</sup><sup>t</sup> <sup>=</sup> ⎡ ⎢ ⎢ ⎢ ⎢ ⎣ ∇ ⊺ <sup>z</sup>1,t(<sup>ξ</sup>t) ⋮ ∇ ⊺ <sup>z</sup>k,t(<sup>ξ</sup>t) ⎤ ⎥ ⎥ ⎥ ⎥ ⎦ . (13)

[\(13\)](#page-5-0) reaches its stationary points (or fixed points) when (note that <sup>c</sup><sup>t</sup> > <sup>0</sup>)

[ ˙θ ˙ξ ] <sup>=</sup> <sup>0</sup> Ô⇒ <sup>J</sup>tz<sup>t</sup> <sup>=</sup> <sup>0</sup> and <sup>ξ</sup><sup>t</sup> <sup>=</sup> <sup>0</sup> Ô⇒ k ∑ i=1 ∇log <sup>ℓ</sup>i,t = <sup>0</sup>. (14)

Therefore, the minimum points of ∑ k i=1 log <sup>ℓ</sup>i(θ) are all stationary points of [\(13\)](#page-5-0).

# 4 Related Work

In this section, we summarize existing methods that tackle learning challenges in multitask learning (MTL). The general idea of most existing works is to encourage positive knowledge transfer by sharing parameters while decreasing any potential negative knowledge transfer (a.k.a, interference) during learning. There are three major ways of doing so: task grouping, designing network architectures specifically for MTL, and designing multitask optimization methods.

Task Grouping Task grouping refers to grouping <sup>K</sup> tasks into <sup>N</sup> < <sup>K</sup> clusters and learning <sup>N</sup> models for each cluster. The key is estimating the amount of positive knowledge transfer incurred by grouping certain tasks together and then identifying which tasks should be grouped [\[39,](#page-12-3) [45,](#page-12-4) [38,](#page-12-5) [36,](#page-12-6) [11\]](#page-10-4).

Multitask Architecture Novel neural architectures for MTL include *hard-parameter-sharing* methods, which decompose a neural network into task-specific modules and a shared feature extractor using manually designed heuristics [\[21,](#page-11-4) [29,](#page-11-5) [2\]](#page-10-5), and *soft-parameter-sharing* methods, which learn which parameters to share [\[30,](#page-11-6) [34,](#page-11-7) [12,](#page-10-6) [27\]](#page-11-8). Recent studies extend neural architecture search for MTL by learning where to branch a network to have task-specific modules [\[14,](#page-10-7) [3\]](#page-10-8).

Multitask Optimization The most relevant approach to our method is MTL optimization via task balancing. These methods dynamically re-weight all task losses to mitigate the conflicting gradient issue [\[40,](#page-12-7) [43\]](#page-12-0). The simplest form of gradient manipulation is to re-weight the task losses based on manually designed criteria [\[6,](#page-10-9) [13,](#page-10-10) [18\]](#page-11-9), but these methods are often heuristic and lack theoretical support. Gradient manipulation methods [\[35,](#page-11-3) [43,](#page-12-0) [25,](#page-11-1) [7,](#page-10-2) [16,](#page-10-11) [24,](#page-11-2) [32,](#page-11-10) [26,](#page-11-11) [47\]](#page-12-8) propose to form a new update vector at each optimization by linearly combining task gradients. The local improvements across all tasks using the new update can often be explicitly analyzed, making these methods better understood in terms of convergence. However, it has been observed that gradient manipulation methods are often slow in practice, which may outweigh their performance benefits [\[22\]](#page-11-12). By contrast, FAMO is designed to match the performance of these methods while remaining efficient in terms of memory and computation. Another recent work proposes to sample random task weights at each optimization step for MTL [\[23\]](#page-11-13), which is also computationally efficient. However, we will demonstrate empirically that FAMO performs better than this method.

### 5 Empirical Results

We conduct experiments to answer the following question:

*How does* FAMO *perform in terms of space/time complexities and standard MTL metrics against prior MTL optimizers on standard benchmarks (e.g., supervised and reinforcement MTL problems)?*

In the following, we first use a toy 2-task problem to demonstrate how FAMO mitigates CG while being efficient. Then we show that FAMO performs comparably or even better than state-of-theart gradient manipulation methods on standard multitask supervised and reinforcement learning benchmarks. In addition, FAMO requires significantly lower computation time when K is large compared to other methods. Lastly, we conduct an ablation study on how robust FAMO is to γ. Each subsection first details the experimental setup and then analyzes the results.

#### <span id="page-6-1"></span><span id="page-6-0"></span>5.1 A Toy 2-Task Example

![](_page_6_Figure_5.jpeg)

Figure 2: The average loss L 0 and the two task losses L 1 and L 2 for the toy example.

To better understand the optimization trajectory of FAMO, we adopt the same 2D multitask optimization problem from NASHMTL [\[32\]](#page-11-10) to visualize how FAMO balances different loss objectives. The model parameter <sup>θ</sup> = (<sup>θ</sup>1, θ2) ∈ <sup>R</sup> 2 . The two tasks' objectives and their surface plots are provided in Appendix [D](#page-15-1) and Figure [2.](#page-6-1) We compare FAMO against ADAM [\[19\]](#page-11-14), MGDA [\[35\]](#page-11-3), PCGRAD [\[43\]](#page-12-0), CAGRAD [\[24\]](#page-11-2), and NASHMTL [\[32\]](#page-11-10). We then pick 5 initial points <sup>θ</sup>init ∈ {(−8.5, <sup>7</sup>.5), (−8.5, <sup>5</sup>), (0, <sup>0</sup>), (9, <sup>9</sup>), (10,−8)} and plot the corresponding optimization trajectories with different methods in Figure [1.](#page-1-3) Note that the toy example is constructed such that naively applying ADAM on the average loss can cause the failure of optimization for task 1.

Findings: From Figure [1,](#page-1-3) we observe that FAMO, like all other gradient manipulation methods, mitigates the CG and reaches the Pareto front for all five runs. In the meantime, FAMO performs similarly to NASHMTL and achieves a balanced loss decrease even when the two task losses are improperly scaled. Finally, as shown in the top-right of the plot, FAMO behaves similarly to ADAM in terms of the training time, which is 25× faster than NASHMTL.

#### 5.2 MTL Performance

Multitask Supervised Learning. We consider four supervised benchmarks commonly used in prior MTL research [\[24,](#page-11-2) [27,](#page-11-8) [32,](#page-11-10) [33\]](#page-11-15): NYU-v2 [\[31\]](#page-11-16) (3 tasks), CityScapes [\[8\]](#page-10-12) (2 tasks), QM-9 [\[1\]](#page-10-13) (11 tasks), and CelebA [\[28\]](#page-11-17) (40 tasks). Specifically, NYU-v2 is an indoor scene dataset consisting of 1449 RGBD images and dense per-pixel labeling with 13 classes. The learning objectives include image segmentation, depth prediction, and surface normal prediction based on any scene image. CityScapes dataset is similar to NYU-v2 but contains 5000 street-view RGBD images with per-pixel annotations. QM-9 dataset is a widely used benchmark in graph neural network learning. It consists of >130K molecules represented as graphs annotated with node and edge features. We follow the same experimental setting used in NASHMTL [\[32\]](#page-11-10), where the learning objective is to predict 11 properties of molecules. We use 110K molecules from the QM9 example in PyTorch Geometric [\[10\]](#page-10-14), 10K molecules for validation, and the rest of 10K molecules for testing. The characteristic of this dataset is that the 11 properties are at different scales, posing a challenge for task balancing in MTL. Lastly, CelebA dataset contains 200K face images of 10K different celebrities, and each face image is provided with 40 facial binary attributes. Therefore, CelebA can be viewed as a 40-task MTL problem. Different from NYU-v2, CityScapes, and QM-9, the number of tasks (K) in CelebA is much larger, hence posing a challenge to learning efficiency.

<span id="page-7-0"></span>

| Method      | mIoU ↑ | Segmentation Pix Acc ↑ | Abs Err | Depth ↓ Rel Err | Angle ↓ | Dist ↓ | Surface | Normal Within | t ○ ↑ | MR ↓  | ∆ m % ↓ |
|-------------|--------|------------------------|---------|-----------------|---------|--------|---------|---------------|-------|-------|---------|
|             |        |                        |         |                 | Mean    | Median | 11.25   | 22.5          | 30    |       |         |
| STL         | 38.30  | 63.76                  | 0.6754  | 0.2780          | 25.01   | 19.21  | 30.14   | 57.20         | 69.15 |       |         |
| LS          | 39.29  | 65.33                  | 0.5493  | 0.2263          | 28.15   | 23.96  | 22.09   | 47.50         | 61.08 | 8.89  | 5.59    |
| SI          | 38.45  | 64.27                  | 0.5354  | 0.2201          | 27.60   | 23.37  | 22.53   | 48.57         | 62.32 | 7.89  | 4.39    |
| RLW         | 37.17  | 63.77                  | 0.5759  | 0.2410          | 28.27   | 24.18  | 22.26   | 47.05         | 60.62 | 11.22 | 7.78    |
| DWA         | 39.11  | 65.31                  | 0.5510  | 0.2285          | 27.61   | 23.18  | 24.17   | 50.18         | 62.39 | 7.67  | 3.57    |
| UW          | 36.87  | 63.17                  | 0.5446  | 0.2260          | 27.04   | 22.61  | 23.54   | 49.05         | 63.65 | 7.44  | 4.05    |
| MGDA        | 30.47  | 59.90                  | 0.6070  | 0.2555          | 24.88   | 19.45  | 29.18   | 56.88         | 69.36 | 6.00  | 1.38    |
| PCG RAD     | 38.06  | 64.64                  | 0.5550  | 0.2325          | 27.41   | 22.80  | 23.86   | 49.83         | 63.14 | 8.00  | 3.97    |
| G RAD D ROP | 39.39  | 65.12                  | 0.5455  | 0.2279          | 27.48   | 22.96  | 23.38   | 49.44         | 62.87 | 7.00  | 3.58    |
| CAG RAD     | 39.79  | 65.49                  | 0.5486  | 0.2250          | 26.31   | 21.58  | 25.61   | 52.36         | 65.58 | 4.56  | 0.20    |
| IMTL-G      | 39.35  | 65.60                  | 0.5426  | 0.2256          | 26.02   | 21.19  | 26.20   | 53.13         | 66.24 | 3.78  | -0.76   |
| N ASH MTL   | 40.13  | 65.93                  | 0.5261  | 0.2171          | 25.26   | 20.08  | 28.40   | 55.47         | 68.15 | 2.11  | -4.04   |
| FAMO        | 38.88  | 64.90                  | 0.5474  | 0.2194          | 25.06   | 19.57  | 29.21   | 56.61         | 68.98 | 3.44  | -4.10   |

Table 1: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.

<span id="page-7-1"></span>

| Method    | µ    | α    | ϵ HOMO | ϵ LUMO | ⟨ R ⟩ 2 | ZPVE MAE | U ↓   | 0 U   | H     |       | G c v | MR ↓ | ∆ m % ↓ |
|-----------|------|------|--------|--------|---------|----------|-------|-------|-------|-------|-------|------|---------|
| STL       | 0.07 | 0.18 | 60.6   | 53.9   | 0.50    | 4.53     | 58.8  | 64.2  | 63.8  | 66.2  | 0.07  |      |         |
| LS        | 0.11 | 0.33 | 73.6   | 89.7   | 5.20    | 14.06    | 143.4 | 144.2 | 144.6 | 140.3 | 0.13  | 6.45 | 177.6   |
| SI        | 0.31 | 0.35 | 149.8  | 135.7  | 1.00    | 4.51     | 55.3  | 55.8  | 55.8  | 55.3  | 0.11  | 3.55 | 77.8    |
| RLW       | 0.11 | 0.34 | 76.9   | 92.8   | 5.87    | 15.47    | 156.3 | 157.1 | 157.6 | 153.0 | 0.14  | 8.00 | 203.8   |
| DWA       | 0.11 | 0.33 | 74.1   | 90.6   | 5.09    | 13.99    | 142.3 | 143.0 | 143.4 | 139.3 | 0.13  | 6.27 | 175.3   |
| UW        | 0.39 | 0.43 | 166.2  | 155.8  | 1.07    | 4.99     | 66.4  | 66.8  | 66.8  | 66.2  | 0.12  | 4.91 | 108.0   |
| MGDA      | 0.22 | 0.37 | 126.8  | 104.6  | 3.23    | 5.69     | 88.4  | 89.4  | 89.3  | 88.0  | 0.12  | 5.91 | 120.5   |
| PCG RAD   | 0.11 | 0.29 | 75.9   | 88.3   | 3.94    | 9.15     | 116.4 | 116.8 | 117.2 | 114.5 | 0.11  | 4.73 | 125.7   |
| CAG RAD   | 0.12 | 0.32 | 83.5   | 94.8   | 3.22    | 6.93     | 114.0 | 114.3 | 114.5 | 112.3 | 0.12  | 5.45 | 112.8   |
| IMTL-G    | 0.14 | 0.29 | 98.3   | 93.9   | 1.75    | 5.70     | 101.4 | 102.4 | 102.0 | 100.1 | 0.10  | 4.36 | 77.2    |
| N ASH MTL | 0.10 | 0.25 | 82.9   | 81.9   | 2.43    | 5.38     | 74.5  | 75.0  | 75.1  | 74.2  | 0.09  | 2.09 | 62.0    |
| FAMO      | 0.15 | 0.30 | 94.0   | 95.2   | 1.63    | 4.95     | 70.82 | 71.2  | 71.2  | 70.3  | 0.10  | 3.27 | 58.5    |

Table 2: Results on QM-9 dataset (11 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.

We compare FAMO against 11 MTL optimization methods and a single-task learning baseline: (1) Single task learning (STL), training an independent model (θ for each task; (2) Linear scalarization (LS) baseline that minimizes L 0 ; (3) Scale-invariant (SI) baseline that minimizes ∑<sup>k</sup> logL k (θ), as SI is invariant to any scalar multiplication of task losses; (4) Dynamic Weight Average (DWA) [\[27\]](#page-11-8), a heuristic for adjusting task weights based on rates of loss changes; (5) Uncertainty Weighting (UW) [\[18\]](#page-11-9) uses task uncertainty as a proxy to adjust task weights; (6) Random Loss Weighting (RLW) [\[23\]](#page-11-13) that samples task weighting whose log-probabilities follow the normal distribution; (7) MGDA [\[35\]](#page-11-3) that finds the equal descent direction for each task; (8) PCGRAD [\[43\]](#page-12-0) proposes to project each task gradient to the normal plan of that of other tasks and combining them together in the end; (9) CAGRAD [\[24\]](#page-11-2) optimizes the average loss while explicitly controls the minimum decrease across tasks; (10) IMTL-G [\[25\]](#page-11-1) finds the update direction with equal projections on task gradients; (11) GRADDROP [\[7\]](#page-10-2) that randomly dropout certain dimensions of the task gradients based on how much they conflict; (12) NASHMTL [\[32\]](#page-11-10) formulates MTL as a bargaining game and finds the solution to the game that benefits all tasks. For FAMO, we choose the best hyperparameter <sup>γ</sup> ∈ {0.0001, <sup>0</sup>.001, <sup>0</sup>.01} based on the validation loss. Specifically, we choose <sup>γ</sup> equals <sup>0</sup>.<sup>01</sup> for the CityScapes dataset and 0.001 for the rest of the datasets. See Appendix [E](#page-16-0) for results with error bars.

Evaluations: We consider two metrics [\[32\]](#page-11-10) for MTL: 1) ∆m%, the average per-task performance drop of a method <sup>m</sup> relative to the STL baseline denoted as <sup>b</sup>: <sup>∆</sup>m% = 1 <sup>K</sup> ∑ K k=1 (−1) <sup>δ</sup><sup>k</sup> (<sup>M</sup>m,k − <sup>M</sup>b,k)/<sup>M</sup>b,k × <sup>100</sup>, where <sup>M</sup>b,k and <sup>M</sup>m,k are the STL and <sup>m</sup>'s value for metric <sup>M</sup>k. <sup>δ</sup><sup>k</sup> = <sup>1</sup> (or <sup>0</sup>) if the M<sup>k</sup> is higher (or lower) the better. 2) Mean Rank (MR): the average rank of each method across tasks. For instance, if a method ranks first for every task, MR will be 1.

Findings: Results on the four benchmark datasets are provided in Table [1,](#page-7-0) [2](#page-7-1) and [3.](#page-8-0) We observe that FAMO performs consistently well across different supervised learning MTL benchmarks compared

<span id="page-8-0"></span>

| Method      | mIoU ↑ | Segmentation Pix Acc ↑ | Abs Err ↓ | CityScapes Depth Rel Err ↓ | MR ↓ | ∆ m % ↓ | MR ↓  | CelebA ∆ m % ↓ |
|-------------|--------|------------------------|-----------|----------------------------|------|---------|-------|----------------|
| STL         | 74.01  | 93.16                  | 0.0125    | 27.77                      |      |         |       |                |
| LS          | 70.95  | 91.73                  | 0.0161    | 33.83                      | 6.50 | 14.11   | 4.15  | 6.28           |
| SI          | 70.95  | 91.73                  | 0.0161    | 33.83                      | 9.25 | 14.11   | 7.20  | 7.83           |
| RLW         | 74.57  | 93.41                  | 0.0158    | 47.79                      | 9.25 | 24.38   | 1.46  | 5.22           |
| DWA         | 75.24  | 93.52                  | 0.0160    | 44.37                      | 6.50 | 21.45   | 3.20  | 6.95           |
| UW          | 72.02  | 92.85                  | 0.0140    | 30.13                      | 6.00 | 5.89    | 3.23  | 5.78           |
| MGDA        | 68.84  | 91.54                  | 0.0309    | 33.50                      | 9.75 | 44.14   | 14.85 | 10.93          |
| PCG RAD     | 75.13  | 93.48                  | 0.0154    | 42.07                      | 6.75 | 18.29   | 3.17  | 6.65           |
| G RAD D ROP | 75.27  | 93.53                  | 0.0157    | 47.54                      | 6.00 | 23.73   | 3.29  | 7.80           |
| CAG RAD     | 75.16  | 93.48                  | 0.0141    | 37.60                      | 5.75 | 11.64   | 2.48  | 6.20           |
| IMTL-G      | 75.33  | 93.49                  | 0.0135    | 38.41                      | 4.00 | 11.10   | 0.84  | 4.67           |
| N ASH MTL   | 75.41  | 93.66                  | 0.0129    | 35.02                      | 2.00 | 6.82    | 2.84  | 4.97           |
| FAMO        | 74.54  | 93.29                  | 0.0145    | 32.59                      | 6.25 | 8.13    | 1.21  | 4.72           |

Table 3: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.

to other gradient manipulation methods. In particular, it achieves state-of-the-art results in terms of ∆m% on the NYU-v2 and QM-9 datasets.

Multitask Reinforcement Learning. We further apply FAMO to multitask reinforcement learning (MTRL) problems as MTRL often suffers more from conflicting gradients due to the stochastic nature of reinforcement learning [\[43\]](#page-12-0). Following CAGRAD [\[24\]](#page-11-2), we apply FAMO on the MetaWorld [\[44\]](#page-12-9) MT10 benchmark, which consists of 10 robot manipulation tasks with different reward functions. Following [\[37\]](#page-12-10), we use Soft Actor-Critic (SAC) [\[15\]](#page-10-15) as the underlying RL algorithm, and compare against baseline methods including LS (SAC with a shared model) [\[44\]](#page-12-9), Soft Modularization [\[42\]](#page-12-11) (an MTL network that routes different modules in a shared model to form different policies), PC-GRAD [\[43\]](#page-12-0), CAGRAD and NASHMTL [\[32\]](#page-11-10). The experimental setting and hyperparameters all match exactly with those in CAGRAD. For NASHMTL, we report the results of applying the NASHMTL update once per {1, <sup>50</sup>, <sup>100</sup>} iterations.[<sup>4</sup>](#page-8-1) The results for all methods are provided in Table [5.2.](#page-8-2)

![](_page_8_Figure_5.jpeg)

<span id="page-8-2"></span>Figure 3: Training Success Rate and Time.

| Method |        |                 |        |        |        | Success | ↑         |
|--------|--------|-----------------|--------|--------|--------|---------|-----------|
|        |        |                 |        |        |        | (mean   | ± stderr) |
| LS     | (lower |                 | bound) |        |        | 0.49    | ± 0.07    |
| STL    |        | (proxy          | for    | upper  | bound) | 0.90    | ± 0.03    |
| PCG    |        | RAD             | [43]   |        |        | 0.72    | ± 0.02    |
| S      | OFT    | M ODULARIZATION |        |        | [42]   | 0.73    | ± 0.04    |
| CAG    |        | RAD             |        |        |        | 0.83    | ± 0.05    |
| N      | ASH    | MTL             | [32]   | (every | 1)     | 0.91    | ± 0.03    |
| N      | ASH    | MTL             | [32]   | (every | 50)    | 0.85    | ± 0.02    |
| N      | ASH    | MTL             | [32]   | (every | 100)   | 0.87    | ± 0.03    |
| N      | ASH    | MTL             | (ours) | (every | 1)     | 0.80    | ± 0.13    |
| N      | ASH    | MTL             | (ours) | (every | 50)    | 0.76    | ± 0.10    |
| N      | ASH    | MTL             | (ours) | (every | 100)   | 0.80    | ± 0.12    |
| UW     |        | [18]            |        |        |        | 0.77    | ± 0.05    |
| FAMO   |        | (ours)          |        |        |        | 0.83    | ± 0.05    |

Table 4: MTRL results (averaged over 10 runs) on the Metaworld-10 benchmark.

<span id="page-8-1"></span><sup>4</sup>We could not reproduce the MTRL results of NASHMTL exactly, so we report both the results from the original paper and our reproduced results.

Findings: From Table [5.2,](#page-8-2) we observe that FAMO performs comparably to CAGRAD and outperforms PCGRAD and the average gradient descent baselines by a large margin. FAMO also outperforms NASHMTL based on our implementation. Moreover, FAMO is significantly faster than NASHMTL, even when it is applied once every 100 steps.

#### 5.3 MTL Efficiency (Training Time Comparison)

Figure [4](#page-9-0) provides the FAMO's average training time per epoch against that of the baseline methods.

<span id="page-9-0"></span>![](_page_9_Figure_3.jpeg)

Figure 4: Average training time per epoch for different MTL optimization methods. We report the relative training time of a method to that of the linear scalarization (LS) method (which uses the average gradient).

Findings: From the figure, we observe that FAMO introduces negligible overhead across all benchmark datasets compared to the LS method, which is, in theory, the lower bound for computation time. In contrast, methods like NASHMTL have much longer training time compared to FAMO. More importantly, the computation cost of these methods scales with the number of tasks. In addition, note that these methods also take at least O(K) space to store the task gradients, which is implausible for large models in the many-task setting (i.e., when <sup>m</sup> = ∣θ∣ and <sup>K</sup> are large).

### 5.4 Ablation on γ

In this section, we provide the ablation study on the regularization coefficient γ in Figure [5.](#page-9-1)

<span id="page-9-1"></span>![](_page_9_Figure_8.jpeg)

Figure 5: Ablation over γ: we plot the performance of FAMO (in terms of ∆m% using different values of γ from {0.0001, 0.001, 0.01} on the four supervised MTL benchmarks.

Findings: From Figure [5,](#page-9-1) we can observe that choosing the right regularization coefficient can be crucial. But except for CityScapes, FAMO performs reasonably well using all different γs. The problem with CityScapes is that one of the task losses is close to 0 at the very beginning, hence small changes in task weighting can result in very different loss improvement. Therefore we conjecture that using a larger γ, in this case, can help stabilize MTL.

## 6 Conclusion and Limitations

In this work, we introduce FAMO, a fast optimization method for multitask learning (MTL) that mitigates the conflicting gradients using O(1) space and time. As multitasking large models gain more attention, we believe designing efficient but effective optimizers like FAMO for MTL is crucial. FAMO balances task losses by ensuring each task's loss decreases approximately at an equal rate. Empirically, we observe that FAMO can achieve competitive performance against the state-of-the-art MTL gradient manipulation methods. One limitation of FAMO is its dependency on the regularization parameter γ, which is introduced due to the stochastic update of the task weighting logits w. Future work can investigate a more principled way of determining γ.

# References

<span id="page-10-16"></span><span id="page-10-15"></span><span id="page-10-14"></span><span id="page-10-13"></span><span id="page-10-12"></span><span id="page-10-11"></span><span id="page-10-10"></span><span id="page-10-9"></span><span id="page-10-8"></span><span id="page-10-7"></span><span id="page-10-6"></span><span id="page-10-5"></span><span id="page-10-4"></span><span id="page-10-3"></span><span id="page-10-2"></span><span id="page-10-1"></span><span id="page-10-0"></span>[1] L. C. Blum and J.-L. Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. *J. Am. Chem. Soc.*, 131:8732, 2009. [2] Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1385–1394, 2019. [3] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. *arXiv preprint arXiv:2008.10292*, 2020. [4] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. *arXiv preprint arXiv:2303.12712*, 2023. [5] Rich Caruana. Multitask learning. *Machine learning*, 28(1):41–75, 1997. [6] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, pages 794–803. PMLR, 2018. [7] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *arXiv preprint arXiv:2010.06808*, 2020. [8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3213–3223, 2016. [9] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5-6):313–318, 2012. [10] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. *arXiv preprint arXiv:1903.02428*, 2019. [11] Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. *Advances in Neural Information Processing Systems*, 34:27503–27516, 2021. [12] Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition*, pages 11543–11552, 2020. [13] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In *Proceedings of the European conference on computer vision (ECCV)*, pages 270–287, 2018. [14] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In *International Conference on Machine Learning*, pages 3854–3863. PMLR, 2020. [15] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In *International Conference on Machine Learning*, pages 1861–1870. PMLR, 2018. [16] Adrián Javaloy and Isabel Valera. Rotograd: Dynamic gradient homogenization for multi-task learning. *arXiv preprint arXiv:2103.02631*, 2021. [17] Alexandr Katrutsa, Daniil Merkulov, Nurislam Tursynbek, and Ivan Oseledets. Follow the bisector: a simple method for multi-objective optimization. *arXiv preprint arXiv:2007.06937*, 2020.

<span id="page-11-17"></span><span id="page-11-16"></span><span id="page-11-15"></span><span id="page-11-14"></span><span id="page-11-13"></span><span id="page-11-12"></span><span id="page-11-11"></span><span id="page-11-10"></span><span id="page-11-9"></span><span id="page-11-8"></span><span id="page-11-7"></span><span id="page-11-6"></span><span id="page-11-5"></span><span id="page-11-4"></span><span id="page-11-3"></span><span id="page-11-2"></span><span id="page-11-1"></span><span id="page-11-0"></span>[18] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 7482–7491, 2018. [19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. [20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. *arXiv preprint arXiv:2304.02643*, 2023. [21] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6129–6138, 2017. [22] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and Pawan K Mudigonda. In defense of the unitary scalarization for deep multi-task learning. *Advances in Neural Information Processing Systems*, 35:12169–12183, 2022. [23] Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. *arXiv preprint arXiv:2111.10603*, 2021. [24] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34:18878–18890, 2021. [25] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In *International Conference on Learning Representations*, 2020. [26] Shikun Liu, Stephen James, Andrew J Davison, and Edward Johns. Auto-lambda: Disentangling dynamic task relationships. *arXiv preprint arXiv:2202.03091*, 2022. [27] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1871–1880, 2019. [28] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of International Conference on Computer Vision (ICCV)*, December 2015. [29] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. Learning multiple tasks with multilinear relationship networks. *Advances in neural information processing systems*, 30, 2017. [30] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3994–4003, 2016. [31] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *ECCV*, 2012. [32] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. *arXiv preprint arXiv:2202.01017*, 2022. [33] Lucas Pascal, Pietro Michiardi, Xavier Bost, Benoit Huet, and Maria A Zuluaga. Improved optimization strategies for deep multi-task networks. *arXiv preprint arXiv:2109.11678*, 2021. [34] Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multitask architecture learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pages 4822–4829, 2019. [35] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *arXiv preprint arXiv:1810.04650*, 2018.

<span id="page-12-11"></span><span id="page-12-10"></span><span id="page-12-9"></span><span id="page-12-8"></span><span id="page-12-7"></span><span id="page-12-6"></span><span id="page-12-5"></span><span id="page-12-4"></span><span id="page-12-3"></span><span id="page-12-2"></span><span id="page-12-1"></span><span id="page-12-0"></span>[36] Jiayi Shen, Xiantong Zhen, Marcel Worring, and Ling Shao. Variational multi-task learning with gumbel-softmax priors. *Advances in Neural Information Processing Systems*, 34:21031–21042, 2021. [37] Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-task reinforcement learning with context-based representations. *arXiv preprint arXiv:2102.06177*, 2021. [38] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In *International Conference on Machine Learning*, pages 9120–9132. PMLR, 2020. [39] Sebastian Thrun and Joseph O'Sullivan. Discovering structure in multiple learning tasks: The tc algorithm. In *ICML*, volume 96, pages 489–497, 1996. [40] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021. [41] Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, and Orhan Firat. Do current multi-task optimization methods in deep learning even help? *Advances in Neural Information Processing Systems*, 35:13597–13609, 2022. [42] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. *arXiv preprint arXiv:2003.13661*, 2020. [43] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. *arXiv preprint arXiv:2001.06782*, 2020. [44] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In *Conference on Robot Learning*, pages 1094–1100. PMLR, 2020. [45] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3712–3722, 2018. [46] Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu. On the convergence of stochastic multi-objective gradient manipulation and beyond. *Advances in Neural Information Processing Systems*, 35:38103–38115, 2022. [47] Shijie Zhu, Hui Zhao, Pengjie Wang, Hongbo Deng, Jian Xu, and Bo Zheng. Gradient deconfliction via orthogonal projections onto subspaces for multi-task learning.

### <span id="page-13-0"></span>A Gradient Manipulation Methods

In this section, we provide a brief overview of representative gradient manipulation methods in multitask/multiobjective optimization. Specifically, we will also discuss the connections among these methods.

Multiple Gradient Descent Algorithm (MGDA) [\[9,](#page-10-3) [35\]](#page-11-3) The MGDA algorithm is one of the earliest gradient manipulation methods for multitask learning. In MGDA, the per step update d<sup>t</sup> is found by solving

max d∈R<sup>m</sup> min i∈[k] ∇ℓ ⊺ i,t<sup>d</sup> − 1 2 ∥d∥ .

As a result, the solution d ∗ of MGDA optimizes the "worst improvement" across all tasks or equivalently seeks an *equal* descent across all task losses as much as possible. But in practice, MGDA suffers from slow convergence since the update d ∗ can be very small. For instance, if one task has a very small loss scale, the progress of all other tasks will be bounded by the progress on this task. Note that the original objective in [\(6\)](#page-3-4) is similar to the MGDA objective in the sense that we can view optimizing [\(6\)](#page-3-4) as optimizing the log of the task losses. Hence, when we compare FAMO against MGDA, one can regard FAMO as balancing the *rate* of loss improvement while MGDA balances the absolute improvement across task losses.

Projecting Gradient Descent (PCGRAD) [\[43\]](#page-12-0) PCGRAD initializes v i PC = ∇<sup>ℓ</sup>i,t, then for each task <sup>i</sup>, PCGRAD loops over all task <sup>j</sup> ≠ <sup>i</sup> (in a random order, which is crucial as mentioned in [\[43\]](#page-12-0)) and removes the "conflict"

v i PC ← <sup>v</sup> i PC − v i PC ⊺ ∇<sup>ℓ</sup>j,t ∥<sup>ℓ</sup>j,t∥ <sup>2</sup> ∇<sup>ℓ</sup>j,t if <sup>v</sup> i PC ⊺ ∇<sup>ℓ</sup>j,t < <sup>0</sup>.

In the end, PCGRAD produces <sup>d</sup><sup>t</sup> = <sup>k</sup> ∑ k i=1 v i PC. Due to the construction, PCGRAD will also help improve the "worst improvement" across all tasks since the "conflicts" have been removed. However, due to the stochastic iterative procedural of this algorithm, it is hard to understand PCGRAD from a first principle approach.

Conflict-averse Gradient Descent (CAGRAD) [\[24\]](#page-11-2) d<sup>t</sup> is found by solving

max d∈R<sup>m</sup> min i∈[k] ∇ℓ ⊺ i,t<sup>d</sup> s.t. ∥<sup>d</sup> − ∇<sup>ℓ</sup>0,t∥ ≤ <sup>c</sup>∥∇<sup>ℓ</sup>0,t∥.

Here, <sup>ℓ</sup>0,t = <sup>k</sup> ∑ k i=1 ℓi,t. CAGRAD seeks an update d<sup>t</sup> that optimizes the "worst improvement" as much as possible, conditioned on that the update still decreases the average loss. By controlling the hyperparameter <sup>c</sup>, CAGRAD can recover MGDA (<sup>c</sup> → ∞) and the vanilla averaged gradient descent (<sup>c</sup> → <sup>0</sup>). Due to the extra constraint, CAGRAD provably converges to the stationary points of <sup>ℓ</sup><sup>0</sup> when <sup>0</sup> ≤ <sup>c</sup> < <sup>1</sup>.

Impartial Multi-Task Learning (IMTL-G) [\[25\]](#page-11-1) IMTL-G finds d<sup>t</sup> such that it shares the same cosine similarity with any task gradients:

∀<sup>i</sup> ≠ j, d<sup>⊺</sup> t ∇<sup>ℓ</sup>i,t ∥∇<sup>ℓ</sup>i,t∥ = d ⊺ t ∇<sup>ℓ</sup>j,t ∥∇<sup>ℓ</sup>j,t∥ , and <sup>d</sup><sup>t</sup> = k ∑ i=1 <sup>w</sup>i,t∇<sup>ℓ</sup>i,t, for some <sup>w</sup><sup>t</sup> ∈ <sup>S</sup>k.

The constraint that <sup>d</sup><sup>t</sup> = ∑ k <sup>i</sup>=<sup>1</sup> <sup>w</sup>i,t∇<sup>ℓ</sup>i,t is for preventing the problem from being under-determined. From the above equation, we can see that IMTL-G ignores the "size" of each task gradient and only cares about the "direction". As a result, one can think of IMTL-G as a variant of MGDA that applies to the normalized gradients. By doing so, IMTL-G does not suffer from the straggler effect due to slow objectives. Furthermore, one can view IMTL-G as the equal angle descent, which is also proposed in Katrutsa et al. [\[17\]](#page-10-16), where the objective is to find d such that

∀<sup>i</sup> ≠ j, cos(d,∇<sup>ℓ</sup>i,t) = cos(d,∇<sup>ℓ</sup>j,t).

NASHMTL[\[32\]](#page-11-10) NASHMTL finds d<sup>t</sup> by solving a bargaining game treating the local improvement of each task loss as the utility for each task:

max d∈Rm,∥d∥≤1 k ∑ i=1 log (∇<sup>ℓ</sup> ⊺ i,t<sup>d</sup>).

Note that the objective of <sup>N</sup>ASHMTL implicitly assumes that there exists <sup>d</sup> such that ∀ i, ∇<sup>ℓ</sup> ⊺ i,t<sup>d</sup> > <sup>0</sup> (otherwise we reach the Pareto front). It is easy to see that

max ∥d∥≤1 k ∑ i=1 log (∇<sup>ℓ</sup> ⊺ i,t<sup>d</sup>) = max ∥d∥≤1 k ∑ i=1 log⟨ ∇<sup>ℓ</sup>i,t ∥∇<sup>ℓ</sup>i,t∥ , d⟩ = max ∥d∥≤1 k ∑ i=1 log cos (∇<sup>ℓ</sup>i,t, d).

Therefore, due to the log, NASHMTL also ignores the "size" of task gradients and only cares about their "directions". Moreover, denote <sup>u</sup><sup>i</sup> = ∇ℓi,t ∥∇ℓi,t∥ . Then, according to the KKT condition, we know:

∑ i ui u ⊺ i d − αd = <sup>0</sup>, α ≥ <sup>0</sup> Ô⇒ <sup>d</sup> = 1 α ∑ i 1 u ⊺ i d ui .

Consider when <sup>k</sup> = <sup>2</sup>, if we take the *equal angle descent* direction: <sup>d</sup><sup>∠</sup> = (<sup>u</sup><sup>1</sup> + <sup>u</sup>2)/<sup>2</sup> (note that as <sup>u</sup><sup>1</sup> and u<sup>2</sup> are normalized, their bisector is just their average). Then it is easy to check that

<sup>d</sup><sup>∠</sup> = 1 α ( 2 u ⊺ 1 (<sup>u</sup><sup>1</sup> + <sup>u</sup>2) <sup>u</sup><sup>1</sup> + 2 u ⊺ 2 (<sup>u</sup><sup>1</sup> + <sup>u</sup>2) <sup>u</sup>2), where <sup>α</sup> <sup>=</sup> u ⊺ (<sup>u</sup><sup>1</sup> + <sup>u</sup>2) 4 = u ⊺ 2 (<sup>u</sup><sup>1</sup> + <sup>u</sup>2) 4 .

As a result, we can see that when <sup>k</sup> = <sup>2</sup>, <sup>N</sup>ASHMTL is equivalent to IMTL-G (or the equal angle descent). However, when <sup>k</sup> > <sup>2</sup>, this is not in general true.

Remark Note that all of these gradient manipulation methods require computing and storing K task gradients before applying f to compute dt, which often involves solving an additional optimization problem. Hence, these methods can be slow for large K and large model sizes.

# <span id="page-14-0"></span>B Amortizing other Gradient Manipulation Methods

Although FAMO uses iterative update on w, it is not immediately clear whether we can apply the same amortization easily on other existing gradient manipulation methods. In this section, we discuss such possibilities and point out the challenges.

Amortizing MGDA This is almost the same as in FAMO, except that MGDA acts on the original task losses while FAMO acts on the log of task losses.

Amortizing PCGRAD For PCGRAD, finding the final update vector requires iteratively projecting one task gradient to the other, so there is no straightforward way of bypassing the computation of task gradients.

Amortizing IMTL-G The task weighting in IMTL-G is computed by a series of matrix-matrix and matrix-vector products using task gradients [\[25\]](#page-11-1). Hence, it is also hard to amortize its computation over time.

Therefore, we focus on deriving the amortization for CAGRAD and NASHMTL.

Amortizing CAGRAD For CAGRAD, the dual objective is

<span id="page-14-1"></span>min w∈S<sup>k</sup> <sup>F</sup>(w) = <sup>g</sup> ⊺ <sup>w</sup>g<sup>0</sup> + <sup>c</sup>∥<sup>g</sup>w∥∥<sup>g</sup>0∥, (15)

where <sup>g</sup><sup>0</sup> = ∇<sup>ℓ</sup>0,t and <sup>g</sup><sup>w</sup> = ∑ k <sup>i</sup>=<sup>1</sup> <sup>w</sup>i∇<sup>ℓ</sup><sup>i</sup> . Denote

<sup>G</sup> = ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣ ∇ℓ ⊺ 1,t ⋮ ∇ℓ ⊺ k,t ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦ .

Now, if we take the gradient with respect to w in [\(15\)](#page-14-1), we have:

∂F ∂w = <sup>G</sup> ⊺ <sup>g</sup><sup>0</sup> + <sup>c</sup> ∥<sup>g</sup>0∥ ∥<sup>g</sup>w∥ G ⊺ gw. (16)

As a result, in order to approximate this gradient, one can separately estimate:

G ⊺ <sup>g</sup><sup>0</sup> ≈ <sup>ℓ</sup>(θ) − <sup>ℓ</sup>(<sup>θ</sup> − αg0) α G ⊺ <sup>g</sup><sup>w</sup> ≈ <sup>ℓ</sup>(θ) − <sup>ℓ</sup>(<sup>θ</sup> − αgw) α ∥<sup>g</sup>0∥ ≈ √ 1 <sup>⊺</sup>G⊺g<sup>0</sup> ∥<sup>g</sup>w∥ ≈ √ w⊺G⊺g<sup>w</sup> . (17)

Once all these are estimated, one can combine them together to perform a single update on w. But note that this will require 3 forward and backward passes through the model, making it harder to implement in practice.

Amortizing NASHMTL Per derivation from NASHMTL [\[32\]](#page-11-10), the objective is to solve for w:

G <sup>⊺</sup>Gw = <sup>1</sup> ⊘ w. (18)

One can therefore form an objective:

min w <sup>F</sup>(w) = ∥<sup>G</sup> <sup>⊺</sup>Gw − <sup>1</sup> ⊘ <sup>w</sup>∥ 2 2 . (19)

Taking the derivative of F with respect to w, we have

∂F ∂w = <sup>2</sup><sup>G</sup> <sup>⊺</sup>G(<sup>G</sup> ⊺ <sup>g</sup><sup>w</sup> <sup>−</sup> <sup>1</sup> <sup>⊘</sup> <sup>w</sup>) <sup>+</sup> <sup>2</sup>(<sup>G</sup> ⊺ <sup>g</sup><sup>w</sup> <sup>−</sup> <sup>1</sup> <sup>⊘</sup> <sup>w</sup>) <sup>⊘</sup> (<sup>w</sup> <sup>⊙</sup> <sup>w</sup>). (20)

Therefore, to approximate the gradient of w, one needs to first estimate

G ⊺ <sup>g</sup><sup>w</sup> ≈ <sup>L</sup>(θ) − <sup>L</sup>(<sup>θ</sup> − αgw) α = η. (21)

Then we estimate

G <sup>⊺</sup>G(<sup>η</sup> − <sup>1</sup> ⊘ <sup>w</sup>) ≈ <sup>L</sup>(θ) − <sup>L</sup>(<sup>θ</sup> − αG(<sup>η</sup> − <sup>1</sup> ⊘ <sup>w</sup>)) . (22)

α Again, this results in 3 forward and backward passes through the model, let alone the overhead of resetting the model back to θ (requires a copy of the original weights).

In short, though it is possible to derive fast approximation algorithm to approximate the gradient update on w for some of the existing gradient manipulation methods, it often involves much more complicated computation compared to that of FAMO.

# <span id="page-15-0"></span>C FAMO Pseudocode in PyTorch

We provide the pseudocode for FAMO in Algorithm [2.](#page-16-1) To use FAMO, one just first compute the task losses, call get\_weighted\_loss to get the weighted loss, and do the normal backpropagation through the weighted loss. After that, one call update to update the task weighting.

# <span id="page-15-1"></span>D Toy Example

We provide the task objectives for the toy example in the following. The model parameter <sup>θ</sup> = (<sup>θ</sup>1, θ2) ∈ <sup>R</sup> and the task objectives are L and L :

L 1 (θ) = <sup>0</sup>.<sup>1</sup> ⋅ (<sup>c</sup>1(θ)<sup>f</sup>1(θ) + <sup>c</sup>2(θ)<sup>g</sup>1(θ)) and <sup>L</sup> 2 (θ) = <sup>c</sup>1(θ)<sup>f</sup>2(θ) + <sup>c</sup>2(θ)<sup>g</sup>2(θ), where <sup>f</sup>1(θ) = log ( max(∣0.5(−<sup>θ</sup><sup>1</sup> − <sup>7</sup>) − tanh (−<sup>θ</sup>2)∣, <sup>0</sup>.000005)) + <sup>6</sup>, <sup>f</sup>2(θ) = log ( max(∣0.5(−<sup>θ</sup><sup>1</sup> + <sup>3</sup>) − tanh (−<sup>θ</sup>2) + <sup>2</sup>∣, <sup>0</sup>.000005)) + <sup>6</sup>, <sup>g</sup>1(θ) = ((−<sup>θ</sup><sup>1</sup> + <sup>7</sup>) <sup>2</sup> + <sup>0</sup>.<sup>1</sup> ∗ (−<sup>θ</sup><sup>2</sup> − <sup>8</sup>) 2 )/<sup>10</sup> − <sup>20</sup>, <sup>g</sup>2(θ) = ((−<sup>θ</sup><sup>1</sup> − <sup>7</sup>) <sup>2</sup> + <sup>0</sup>.<sup>1</sup> ∗ (−<sup>θ</sup><sup>2</sup> − <sup>8</sup>) )/<sup>10</sup> − <sup>20</sup>, <sup>c</sup>1(θ) = max(tanh (0.<sup>5</sup> ∗ <sup>θ</sup>2), <sup>0</sup>) and <sup>c</sup>2(θ) = max(tanh (−0.<sup>5</sup> ∗ <sup>θ</sup>2), <sup>0</sup>).

### <span id="page-16-1"></span>Algorithm 2 Implementation of FAMO in PyTorch-like Pseudocode

class FAMO: def \_\_init\_\_(self, num\_tasks, min\_losses, α=0.025, γ=0.001): # min\_losses (num\_tasks,) the loss lower bound for each task. self.min\_losses = min\_losses self.xi = torch.tensor([0.0] \* num\_tasks, requires\_grad=True) self.xi\_opt = torch.optim.Adam([self.xi], lr=α, weight\_decay=γ) def get\_weighted\_loss(self, losses): # losses (num\_tasks,) z = F.softmax(self.xi, -1) D = losses - self.min\_losses + 1e-8 c = 1 / (z / D).sum().detach() loss = (c \* D.log() \* z).sum() return loss def update(self, prev\_losses, curr\_losses): # prev\_losses (num\_tasks,) # curr\_losses (num\_tasks,) delta = (prev\_losses - self.min\_losses + 1e-8).log() - (curr\_losses - self.min\_losses + 1e-8).log() with torch.enable\_grad(): d = torch.autograd.grad(F.softmax(self.xi, -1), self.xi, grad\_outputs=delta.detach())[0] self.xi\_opt.zero\_grad() self.xi.grad = d self.xi\_opt.step

### <span id="page-16-0"></span>E Experimental Results with Error Bars

We followed the exact experimental setup from NASHMTL [\[32\]](#page-11-10). Therefore, the numbers for baseline methods are taken from their original paper. In the following, we provide FAMO's result with error bars.

| Method        | Segmentation mIoU ↑ | Pix Acc ↑ | Depth Abs Err ↓ | Rel Err ↓ | Angle  | Surface Dist ↓ | Normal | Within t ○ | ↑      | ∆ m % ↓ |
|---------------|---------------------|-----------|-----------------|-----------|--------|----------------|--------|------------|--------|---------|
|               |                     |           |                 |           | Mean   | Median         | 11.25  | 22.5       | 30     |         |
| FAMO (mean)   | 38.88               | 64.90     | 0.5474          | 0.2194    | 25.06  | 19.57          | 29.21  | 56.61      | 68.98  | -4.10   |
| FAMO (stderr) | ± 0.54              | ± 0.21    | ± 0.0016        | ± 0.0026  | ± 0.06 | ± 0.09         | ± 0.17 | ± 0.19     | ± 0.14 | ± 0.39  |

Table 5: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.

Table 6: Results on QM-9 dataset (11 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.

Table 7: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and ∆m% are the main metrics for MTL performance.