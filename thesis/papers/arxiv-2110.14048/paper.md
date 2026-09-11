# Conflict-Averse Gradient Descent  
for Multi-task Learning

 †Bo Liu    †Xingchao Liu    ‡Xiaojie Jin    Peter Stone    †Qiang Liu    †The University of Texas at Austin    Sony AI    ‡Bytedance Research{bliu,xcliu,pstone,lqiang}@cs.utexas.edu, xjjin0731@gmail.com 

###### Abstract

The goal of multi-task learning is to enable more efficient learning than single task learning by sharing model structures for a diverse set of tasks. A standard multi-task learning objective is to minimize the average loss across all tasks. While straightforward, using this objective often results in much worse final performance for each task than learning them independently. A major challenge in optimizing a multi-task model is the *conflicting gradients*, where gradients of different task objectives are not well aligned so that following the average gradient direction can be detrimental to specific tasks’ performance. Previous work has proposed several heuristics to manipulate the task gradients for mitigating this problem. But most of them lack convergence guarantee and/or could converge to any Pareto-stationary point. In this paper, we introduce Conflict-Averse Gradient descent (CAGrad) which minimizes the average loss function, while leveraging the worst local improvement of individual tasks to regularize the algorithm trajectory. CAGrad balances the objectives automatically and still provably converges to a minimum over the average loss. It includes the regular gradient descent (GD) and the multiple gradient descent algorithm (MGDA) in the multi-objective optimization (MOO) literature as special cases. On a series of challenging multi-task supervised learning and reinforcement learning tasks, CAGrad achieves improved performance over prior state-of-the-art multi-objective gradient manipulation methods. Code is available at [https://github.com/Cranial-XIX/CAGrad](https://github.com/Cranial-XIX/CAGrad "").

## 1 Introduction

Multi-task learning (MTL) refers to learning a single model that can tackle multiple different tasks \[[11](#bib.bib11 ""), [28](#bib.bib28 ""), [44](#bib.bib44 ""), [38](#bib.bib38 "")\]. By sharing parameters across tasks, MTL methods learn more efficiently with an overall smaller model size compared to learning with separate models \[[38](#bib.bib38 ""), [40](#bib.bib40 ""), [25](#bib.bib25 "")\]. Moreover, it has been shown that MTL could in principle improve the quality of the learned representation and therefore benefit individual tasks \[[35](#bib.bib35 ""), [43](#bib.bib43 ""), [34](#bib.bib34 "")\]. For example, an early MTL result by \[[2](#bib.bib2 "")\] demonstrated that training a neural network to recognize doors could be improved by simultaneously training it to recognize doorknobs.

However, learning multiple tasks simultaneously can be a challenging optimization problem because it involves multiple objectives \[[38](#bib.bib38 "")\]. The most popular MTL objective in practice is the average loss over all tasks. Even when this average loss is exactly the true objective (as opposed to only caring about a single task as in the door/doorknob example), directly optimizing the average loss could lead to undesirable performance, e.g. the optimizer struggles to make progress so the learning performance significantly deteriorates. A known cause of this phenomenon is the *conflicting gradients* \[[41](#bib.bib41 "")\]: gradients from different tasks 1) may have varying scales with the largest gradient dominating the update, and 2) may point in different directions so that directly optimizing the average loss can be quite detrimental to a specific task’s performance.

To address this problem, previous work either adaptively re-weights the objectives of each task based on heuristics \[[3](#bib.bib3 ""), [15](#bib.bib15 "")\] or seeks a better update vector \[[30](#bib.bib30 ""), [41](#bib.bib41 "")\] by manipulating the task gradients. However, existing work often lacks convergence guarantees or only provably converges to any point on the Pareto set of the objectives. This means the final convergence point of these methods may largely depend on the initial model parameters. As a result, it is possible that these methods over-optimize one objective while overlooking the others (See Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conflict-Averse Gradient Descent for Multi-task Learning")).

Motivated by the limitation of current methods, we introduce Conflict-Averse Gradient descent (CAGrad), which reduces the conflict among gradients and still provably converges to a minimum of the average loss. The idea of CAGrad is simple: it looks for an update vector that maximizes the worst local improvement of any objective in a neighborhood of the average gradient. In this way, CAGrad automatically balances different objectives and smoothly converges to an optimal point of the average loss. Specifically, we show that vanilla gradient descent (GD) and the multiple gradient descent algorithm (MGDA) are special cases of CAGrad (See Sec. [3.1](#S3.SS1 "3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). We demonstrate that CAGrad can improve over prior state-of-the-art gradient manipulation methods on a series of challenging multi-task supervised, semi-supervised, and reinforcement learning problems.

![Refer to caption](2110.14048v2/toy.png)

Figure 1: The optimization challenges faced by gradient descent (GD) and existing gradient manipulation methods like the multiple gradient descent algorithm (MGDA) \[[6](#bib.bib6 "")\] and PCGrad \[[41](#bib.bib41 "")\]. MGDA, PCGrad and CAGrad are applied on top of the Adam optimizer \[[16](#bib.bib16 "")\]. For each methods, we repeat 3 runs of optimization from different initial points (labeled with ∙\\bullet). Each optimization trajectory is colored from red to yellow. GD with Adam gets stuck on two of the initial points because the gradient of one task overshadows that of the other task, causing the algorithm to jump back and forth between the walls of a steep valley without making progress along the floor of the valley. MGDA and PCGrad stop optimization as soon as they reach the Pareto set.

## 2 Background

In this section, we first introduce the problem setup of multi-task learning (MTL). Then we analyze the optimization challenge of MTL and discuss the limitation of prior gradient manipulation methods.

### 2.1 Multi-task Learning and its Challenge

In multi-task learning (MTL), we are given K≥2K\\geq 2 different tasks, each of which is associated with a loss function Li​(θ)L\_{i}(\\theta) for a shared set of parameters θ\\theta. The goal is to find an optimal θ∈ℝm\\theta\\in\\mathbb{R}^{m} that achieves low losses across all tasks. In practice, a standard objective for MTL is minimizing the average loss over all tasks:

θ∗\=arg​minθ∈ℝm{L0(θ)≜1K∑i\=1KLi(θ)}.\\theta^{\*}=\\argmin\_{\\theta\\in\\mathbb{R}^{m}}\\left\\{L\_{0}(\\theta)\\triangleq\\frac{1}{K}\\sum\_{i=1}^{K}L\_{i}(\\theta)\\right\\}.

(1)

Unfortunately, directly optimizing ([1](#S2.E1 "In 2.1 Multi-task Learning and its Challenge ‣ 2 Background ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) using gradient descent may significantly compromise the optimization of individual losses in practice. A major source of this phenomenon is known as the conflicting gradients \[[41](#bib.bib41 "")\].

Optimization Challenge: Conflicting Gradients    Denote by gi\=∇Li​(θ)g\_{i}=\\nabla L\_{i}(\\theta) the gradient of task ii, and g0\=∇L0​(θ)\=1K​∑iKgig\_{0}=\\nabla L\_{0}(\\theta)=\\frac{1}{K}\\sum\_{i}^{K}g\_{i} the averaged gradient. With learning rate α∈ℝ+\\alpha\\in\\mathbb{R}^{+}, θ←θ−α​g0\\theta\\leftarrow\\theta-\\alpha g\_{0} is the steepest descent update that appears to be the most natural update to follow when optimizing ([1](#S2.E1 "In 2.1 Multi-task Learning and its Challenge ‣ 2 Background ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). However, g0g\_{0} may conflict with individual gradients, i.e. ∃i,⟨gi,g0⟩<0\\exists~~i,~\\langle g\_{i},g\_{0}\\rangle<0. When this conflict is large, following g0g\_{0} will decrease the performance on task ii. As observed by \[[41](#bib.bib41 "")\] and illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), when θ\\theta is near a steep “valley", where a specific task’s gradient dominates the update, manipulating the direction and magnitude of g0g\_{0} often leads to better optimization.

### 2.2 Prior Attempts and Convergence Issues

Several methods have been proposed to manipulate the task gradients to form a new update vector and have shown improved performance on MTL. Sener et al. apply the multiple-gradient descent algorithm (MGDA) \[[6](#bib.bib6 "")\] for MTL, which directly optimizes towards the Pareto set \[[30](#bib.bib30 "")\]. Chen et al. dynamically re-weight each LiL\_{i} using a pre-defined heuristic \[[3](#bib.bib3 "")\]. More recently, PCGrad identifies conflicting gradients as the motivation behind manipulating the gradients and projects each task gradient to the normal plane of others to reduce the conflict \[[41](#bib.bib41 "")\]. While all these methods have shown success at improving the learning performance of MTL, they manipulate the gradient without respecting the original objective ([1](#S2.E1 "In 2.1 Multi-task Learning and its Challenge ‣ 2 Background ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). Therefore, these methods could in principle converge to any point in the Pareto set (See Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and Sec. [3.2](#S3.SS2 "3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). We provide the detailed algorithms of MGDA and PCGrad in Appendix [A.1](#A1.SS1 "A.1 Multiple Gradient Descent Algorithm (MGDA) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and [A.2](#A1.SS2 "A.2 Projecting Conflicting Gradients (PCGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), and a visualization of the update vector by each method in Fig. [2](#S3.F2 "Figure 2 ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

## 3 Method

We introduce our main algorithm, Conflict-Averse Gradient descent in Sec. [3.1](#S3.SS1 "3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), and then show theoretical analysis in Sec. [3.2](#S3.SS2 "3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

### 3.1 Conflict-Averse Gradient Descent

Assume we update θ\\theta by θ′←θ−α​d\\theta^{\\prime}\\leftarrow\\theta-\\alpha d, where α\\alpha is a step size and dd an update vector. We want to choose dd to decrease not only the average loss L0L\_{0}, but also every individual loss. To do so, we consider the minimum decrease rate across the losses, R⁡(θ,d)\=maxi∈\[K\]⁡{1α​(Li​(θ−α​d)−Li​(θ))}≈−mini∈\[K\]⁡⟨gi,d⟩,R(\\theta,d)=\\max\_{i\\in\[K\]}\\left\\{\\frac{1}{\\alpha}\\left(L\_{i}(\\theta-\\alpha d)-L\_{i}(\\theta)\\right)\\right\\}\\approx-\\min\_{i\\in\[K\]}\\langle g\_{i},d\\rangle,

(2)

where we use the first-order Taylor approximation assuming α\\alpha is small. If R⁡(θ,d)<0R(\\theta,d)<0, it means that all losses are decreased with the update given a sufficiently small α\\alpha. Therefore, R⁡(θ,d)R(\\theta,d) can be regarded as a measurement of conflict among objectives.

With the above measurement, our algorithm finds an update vector that minimizes such conflict to mitigate the optimization challenge while still converging to an optimum of the main objective L0​(θ)L\_{0}(\\theta). To this end, we introduce Conflict-Averse Gradient descent (CAGrad), which on each optimization step determines the update dd by solving the following optimization problem:

maxd∈ℝm⁡mini∈\[K\]​⟨gi,d⟩s.t.‖d−g0‖≤c⁡‖g0‖,\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[K\]}\\langle g\_{i},d\\rangle~~~~\\text{s.t.}~~~~\\left\\lVert d-g\_{0}\\right\\rVert\\leq c\\left\\lVert g\_{0}\\right\\rVert,

(3)

Here, c∈\[0,1)c\\in\[0,1) is a pre-specified hyper-parameter that controls the convergence rate (See Sec. [3.2](#S3.SS2 "3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). The optimization problem  ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) looks for the best update vector within a local ball centered at the averaged gradient g0g\_{0}, which also minimizes the conflict in losses measured by ([2](#S3.E2 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). Since we focus on MTL and choose the average loss as the main objective, g0g\_{0} is the average gradient. However, CAGrad also applies when g0g\_{0} is the gradient of some other user-specified objective. We leave exploring this possibility as a future direction.

#### Dual Objective

The optimization problem ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) involves decision variable dd that has the same dimension as the number of parameters in θ\\theta, which could be millions for a deep neural network. It is not practical to directly solve for dd on every optimization step. However, the dual problem of Eq. ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")), as we will derive in the following, only involves solving for a decision variable w∈ℝKw\\in\\mathbb{R}^{K}, which can be efficiently found using standard optimization libraries \[[7](#bib.bib7 "")\]. Specifically, first note that mini⁡⟨gi,d⟩\=minw∈𝒲⁡⟨∑iwi​gi,d⟩\\min\_{i}\\langle g\_{i},d\\rangle=\\min\_{w\\in\\mathcal{W}}\\langle\\sum\_{i}w\_{i}g\_{i},d\\rangle, where w\=(w1,…,wK)∈ℝKw=(w\_{1},\\dots,w\_{K})\\in\\mathbb{R}^{K} and 𝒲\\mathcal{W} denotes the probability simplex, i.e. 𝒲\={w:∑iwi\=1​and​wi≥0}\\mathcal{W}=\\{w\\colon\\sum\_{i}w\_{i}=1~\\text{and}~w\_{i}\\geq 0\\}. Denote gw\=∑iwi​gig\_{w}=\\sum\_{i}w\_{i}g\_{i} and ϕ\=c2​‖g0‖2\\phi=c^{2}\\left\\lVert g\_{0}\\right\\rVert^{2}. The Lagrangian of the objective in Eq. ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) is

maxd∈ℝm⁡minλ≥0,w∈𝒲​gw⊤​d−λ⁡(‖g0−d‖2−ϕ)/2.\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}g\_{w}^{\\top}d-\\lambda(\\left\\lVert g\_{0}-d\\right\\rVert^{2}-\\phi)/2.

Since the objective for dd is concave with linear constraints, by switching the min\\min and max\\max, we reach the dual form without changing the solution by Slater’s condition:

minλ≥0,w∈𝒲⁡maxd∈ℝm​gw⊤​d−λ​‖g0−d‖2/2+λ​ϕ/2.\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}\\max\_{d\\in\\mathbb{R}^{m}}g\_{w}^{\\top}d-\\lambda\\left\\lVert g\_{0}-d\\right\\rVert^{2}/2+\\lambda\\phi/2.

We end up with the following optimization problem w.r.t. ww after several steps of calculus, w∗\=arg​minw∈𝒲⁡gw⊤​g0+ϕ​‖gw‖,w^{\*}=\\argmin\_{w\\in\\mathcal{W}}g\_{w}^{\\top}g\_{0}+\\sqrt{\\phi}\\left\\lVert g\_{w}\\right\\rVert, where the optimal λ∗\=‖gw∗‖/ϕ1/2\\lambda^{\*}=\\left\\lVert g\_{w^{\*}}\\right\\rVert/\\phi^{1/2} and the optimal update d∗\=g0+gw∗/λ∗d^{\*}=g\_{0}+g\_{w^{\*}}/\\lambda^{\*}. The detailed derivation is provided in Appendix [A.3](#A1.SS3 "A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and the entire CAGrad algorithm is summarized in Alg. [1](#alg1 "Algorithm 1 ‣ Dual Objective ‣ 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). The dimension of ww equals to the number of objectives KK, which usually ranges from 22 to tens and is much smaller than the number of parameters in a neural network. Therefore, in practice, we solve the dual objective to perform the update of CAGrad.

Algorithm 1 Conflict-averse Gradient Descent (CAGrad) for Multi-task Learning

 Input: Initial model parameter vector θ0\\theta\_{0}, differentiable loss functions {Li}i\=1K\\{L\_{i}\\}\_{i=1}^{K}, a constant c∈\[0,1)c\\in\[0,1) and learning rate α∈ℝ+\\alpha\\in\\mathbb{R}^{+}. 

 repeat 

  At the tt-th optimization step, define g0\=1K​∑i\=1K∇Li​(θt−1)g\_{0}=\\frac{1}{K}\\sum\_{i=1}^{K}\\nabla L\_{i}(\\theta\_{t-1}) and ϕ\=c2​‖g0‖2\\phi=c^{2}\\left\\lVert g\_{0}\\right\\rVert^{2}. 

  Solve

minw∈𝒲F(w):=gw⊤g0+ϕ‖gw‖,wheregw\=∑i\=1Kwi∇Li(θt−1).\\min\_{w\\in\\mathcal{W}}F(w):=g\_{w}^{\\top}g\_{0}+\\sqrt{\\phi}\\left\\lVert g\_{w}\\right\\rVert,~\\text{where}~g\_{w}=\\sum\_{i=1}^{K}w\_{i}\\nabla L\_{i}(\\theta\_{t-1}).

  Update θt\=θt−1−α⁡(g0+ϕ1/2‖gw‖​gw).\\theta\_{t}=\\theta\_{t-1}-\\alpha\\left(g\_{0}+\\frac{\\phi^{1/2}}{\\left\\lVert g\_{w}\\right\\rVert}g\_{w}\\right). 

 until convergence 

#### Remark

In Alg. [1](#alg1 "Algorithm 1 ‣ Dual Objective ‣ 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), when c\=0c=0, CAGrad recovers the typical gradient descent with d\=g0d=g\_{0}. On the other hand, when c→∞c\\rightarrow\\infty, then minimizing F⁡(w)F(w) is equivalent to minw⁡‖gw‖\\min\_{w}\\left\\lVert g\_{w}\\right\\rVert. This coincides with the multiple gradient descent algorithm (MGDA) \[[6](#bib.bib6 "")\], which uses the minimum norm vector in the convex hull of the individual gradients as the update direction (see Fig. [2](#S3.F2 "Figure 2 ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"); second column). MGDA is a gradient-based multi-objective optimization designed to converge to an arbitrary point on the Pareto set, that is, it leaves all the points on the Pareto set as fixed points (and hence can not control which specific point it will converge to). It is different from our method which targets to minimize L0L\_{0} while using gradient conflict to regularize the optimization trajectory. As we will analyze in the following section, to guarantee that CAGrad converges to an optimum of L0​(θ)L\_{0}(\\theta), we have to ensure 0≤c<10\\leq c<1.

### 3.2 Convergence Analysis

In this section we first formally introduce the related Pareto concepts and then analyze CAGrad’s convergence property. Particularly, in Alg. [1](#alg1 "Algorithm 1 ‣ Dual Objective ‣ 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), when c<1c<1, CAGrad is guaranteed to converge to a minimum point of the average loss L0L\_{0}.

Pareto Concepts   Unlike single task learning where any two parameter vectors θ1\\theta\_{1} and θ2\\theta\_{2} can be ordered in the sense that either L⁡(θ1)≤L⁡(θ2)L(\\theta\_{1})\\leq L(\\theta\_{2}) or L⁡(θ1)≥L⁡(θ2)L(\\theta\_{1})\\geq L(\\theta\_{2}) holds, MTL could have two parameter vectors where one performs better for task ii and the other performs better for task j≠ij\\neq i. To this end, we need the notion of Pareto-optimality \[[13](#bib.bib13 "")\].

###### Definition 3.1 (Pareto optimal and stationary points).

Let 𝐋⁡(θ)\={Li​(θ):i∈\[K\]}\\boldsymbol{L}(\\theta)=\\{L\_{i}(\\theta)\\colon i\\in\[K\]\\} be a set of differentiable loss functions from ℝm\\mathbb{R}^{m} to ℝ\\mathbb{R}. For two points θ,θ′∈ℝm\\theta,\\theta^{\\prime}\\in\\mathbb{R}^{m}, we say that θ\\theta is Pareto dominated by θ′\\theta^{\\prime}, denoted by 𝐋⁡(θ′)≺𝐋⁡(θ)\\boldsymbol{L}(\\theta^{\\prime})\\prec\\boldsymbol{L}(\\theta), if Li​(θ′)≤Li​(θ)L\_{i}(\\theta^{\\prime})\\leq L\_{i}(\\theta) for all i∈\[K\]i\\in\[K\] and 𝐋⁡(θ′)≠𝐋⁡(θ)\\boldsymbol{L}(\\theta^{\\prime})\\neq\\boldsymbol{L}(\\theta). A point θ∈ℝm\\theta\\in\\mathbb{R}^{m} is said to be Pareto-optimal if there exists no θ′∈ℝm\\theta^{\\prime}\\in\\mathbb{R}^{m} such that 𝐋⁡(θ′)≺𝐋⁡(θ)\\boldsymbol{L}(\\theta^{\\prime})\\prec\\boldsymbol{L}(\\theta). The set of all Pareto-optimal points is called the Pareto set. A point θ\\theta is called Pareto-stationary if we have minw∈𝒲⁡‖gw​(θ)‖\=0\\min\_{w\\in\\mathcal{W}}\\left\\lVert g\_{w}(\\theta)\\right\\rVert=0, where gw(θ)\=∑i\=1Kwi∇Li(θ),g\_{w}(\\theta)=\\sum\_{i=1}^{K}w\_{i}\\nabla L\_{i}(\\theta), and 𝒲\\mathcal{W} is the probability simplex on \[K\].\[K\].

![Refer to caption](2110.14048v2/teasing.png)

Figure 2: The combined update vector dd (in red) of a two-task learning problem with gradient descent (GD), multiple gradient descent algorithm (MGDA), PCGrad and Conflict-Averse Gradient descent (CAGrad). The two task-specific gradients are labeled g1g\_{1} and g2g\_{2}. MGDA’s objective is given in its primal form (See Appendix [A.1](#A1.SS1 "A.1 Multiple Gradient Descent Algorithm (MGDA) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). For PCGrad, each gradient is first projected onto the normal plane of the other (the dashed arrows). Then the final update vector is the average of the two projected gradients. CAGrad finds the best update vector within a ball around the average gradient that maximizes the worse local improvement between task 1 and task 2.

Similar to the case of single-objective differentiable optimization, a local Pareto optimal point θ\\theta must be Pareto stationary (see e.g., \[[6](#bib.bib6 "")\]).

###### Theorem 3.2 (Convergence of CAGrad).

Assume the individual loss functions L0,L1,…,LKL\_{0},L\_{1},\\ldots,L\_{K} are differentiable on ℝm\\mathbb{R}^{m} and their gradients ∇Li​(θ)\\nabla L\_{i}(\\theta) are all HH-Lipschitz, i.e. ‖∇Li​(x)−∇Li​(y)‖≤H⁡‖x−y‖\\left\\lVert\\nabla L\_{i}(x)-\\nabla L\_{i}(y)\\right\\rVert\\leq H\\left\\lVert x-y\\right\\rVert for i\=0,1,…,Ki=0,1,\\ldots,K where 0≤H≤∞0\\leq H\\leq\\infty. Assume L0∗\=infθ∈ℝmL0​(θ)\>−∞L\_{0}^{\*}=\\inf\_{\\theta\\in\\mathbb{R}^{m}}L\_{0}(\\theta)>-\\infty.

With a fixed step size α\\alpha satisfying 0<α≤1/H0<\\alpha\\leq 1/H, we have for the CAGrad in Alg. [1](#alg1 "Algorithm 1 ‣ Dual Objective ‣ 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"):

1) For any c≥1c\\geq 1, all the fixed points of CAGrad are Pareto-stationary points of (L0,L1,…,LK)(L\_{0},L\_{1},\\ldots,L\_{K}).

2) In particular, if we take 0≤c<10\\leq c<1, then CAGrad satisfies

∑t\=0T‖∇L0​(θt)‖2≤2​(L0​(θ0)−L0∗)α⁡(1−c2).\\sum\_{t=0}^{T}\\left\\lVert\\nabla L\_{0}(\\theta\_{t})\\right\\rVert^{2}\\leq\\frac{2(L\_{0}(\\theta\_{0})-L\_{0}^{\*})}{{\\alpha}(1-c^{2})}.

This means that the algorithm converges to a stationary point of ∇L0\\nabla L\_{0} if we take 0≤c<10\\leq c<1. The proof is in Appendix [A.3](#A1.SS3 "A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). As we discuss earlier, unlike our method, MGDA is designed to converge to an arbitrary point on the Pareto set, without explicit control of which point it will converges to. Another algorithm with similar property is PCGrad \[[41](#bib.bib41 "")\], which is a gradient-based algorithm that mitigates the conflicting gradients problem by removing the conflicting components of each gradient with respect to the other gradients before averaging them to form the final update; see Fig. [2](#S3.F2 "Figure 2 ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), third column for the illustration. Similar to MGDA, as shown in \[[41](#bib.bib41 "")\], PCGrad also converges to an arbitrary Pareto point without explicit control of which point it will arrive at.

### 3.3 Practical Speedup

A typical drawback of methods that manipulate gradients is the computation overhead. For computing the optimal update vector, a method usually requires KK back-propagations to find all individual gradients gig\_{i}, in addition to the time required for optimization. This can be prohibitive for the scenario with many tasks. To this end, we propose to only sample a subset of tasks S⊆\[K\]S\\subseteq\[K\], compute their corresponding gradients {gi∣i∈S}\\{g\_{i}\\mid i\\in S\\} and the averaged gradient g0g\_{0}. Then we optimize dd in:

maxd∈ℝm⁡min⁡(⟨K​g0−∑i∈SgiK−|S|,d⟩,mini∈S⁡⟨gi,d⟩)​s.t.​‖d−g0‖≤c⁡‖g0‖\\begin{split}\\max\_{d\\in\\mathbb{R}^{m}}\\min\\bigg(\\langle\\frac{Kg\_{0}-\\sum\_{i\\in S}g\_{i}}{K-|S|},d\\rangle,~~\\min\_{i\\in S}\\langle g\_{i},d\\rangle\\bigg)~~~\\text{s.t.}~~\\left\\lVert d-g\_{0}\\right\\rVert\\leq c\\left\\lVert g\_{0}\\right\\rVert\\end{split}

(4)

Remark Note that the convergence guarantee in Thm. [3.2](#S3.Thmthm2 "Theorem 3.2 (Convergence of CAGrad). ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning") still holds for Eq. [4](#S3.E4 "In 3.3 Practical Speedup ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning") as the constraint does not change (See Appendix [A.3](#A1.SS3 "A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). The time complexity is 𝒪⁡((|S|​N+T)CLOSE\\mathcal{O}((|S|N+T), where NN denotes the time for one pass of back-propagation and TT denotes the optimization time. For few-task learning (K<10K<10), usually T≪NT\\ll N. When S\=\[K\]S=\[K\], we recover the full CAGrad algorithm.

## 4 Related Work

Multi-task Learning   Due to its benefit with regards to data and computational efficiency, multi-task learning (MTL) has broad applications in vision, language, and robotics \[[11](#bib.bib11 ""), [28](#bib.bib28 ""), [22](#bib.bib22 ""), [44](#bib.bib44 ""), [38](#bib.bib38 "")\]. A number of MTL-friendly architectures have been proposed using task-specific modules \[[25](#bib.bib25 ""), [11](#bib.bib11 "")\], attention-based mechanisms \[[21](#bib.bib21 "")\] or activating different paths along the deep networks to tackle MTL \[[27](#bib.bib27 ""), [40](#bib.bib40 "")\]. Apart from designing new architectures, another branch of methods focus on decomposing a large problem into smaller local problems that could be quickly learned by smaller models \[[29](#bib.bib29 ""), [26](#bib.bib26 ""), [37](#bib.bib37 ""), [8](#bib.bib8 "")\]. Then a unified policy is learned from the smaller models using knowledge distillation \[[12](#bib.bib12 "")\].

MTL Optimization   In this work, we focus on the optimization challenge of MTL \[[38](#bib.bib38 "")\]. Gradient manipulation methods are designed specifically to balance the learning of each task. The simplest form of gradient manipulation is to re-weight the task losses based on specific criteria, e.g., uncertainty \[[15](#bib.bib15 "")\], gradient norm \[[3](#bib.bib3 "")\], or difficulty \[[9](#bib.bib9 "")\]. These methods are mostly heuristics and their performance can be unstable. Recently, two methods \[[30](#bib.bib30 ""), [41](#bib.bib41 "")\] that manipulate the gradients to find a better local update vector have become popular. Sener et al \[[30](#bib.bib30 "")\] view MTL as a multi-objective optimization problem, and use multiple gradient descent algorithm for optimization. PCGrad \[[41](#bib.bib41 "")\] identifies a major optimization challenge for MTL, the conflicting gradients, and proposes to project each task gradient to the normal plane of other task gradients before combining them together to form the final update vector. Though yielding good empirical performance, both methods can only guarantee convergence to a Pareto-stationary point, but not knowing where it exactly converges to. More recently, GradDrop \[[4](#bib.bib4 "")\] randomly drops out task gradients based on how much they conflict. IMTL-G \[[20](#bib.bib20 "")\] seeks an update vector that has equal projections on each task gradient. RotoGrad \[[14](#bib.bib14 "")\] separately scales and rotates task gradients to mitigate optimization conflict.

Our method, CAGrad, also manipulates the gradient to find a better optimization trajectory. Like other MTL optimization techniques, CAGrad is model-agnostic. However, unlike prior methods, CAGrad converges to the optimal point in theory and achieves better empirical performance on both toy multi-objective optimization tasks and real-world applications.

## 5 Experiment

We conduct experiments to answer the following questions:

Question (1) Do CAGrad, MGDA and PCGrad behave consistently with their theoretical properties in practice? (yes)

Question (2) Does CAGrad recover GD and MGDA when varying the constant cc? (yes)

Question (3) How does CAGrad perform in both performance and computational efficiency compared to prior state-of-the-art methods, on challenging multi-task learning problems under the supervised, semi-supervised and reinforcement learning settings? (CAGrad improves over prior state-of-the-art methods under all settings)

### 5.1 Convergence and Ablation over c

To answer questions (1) and (2), we create a toy optimization example to evaluate the convergence of CAGrad compared to MGDA and PCGrad. On the same toy example, we ablate over the constant cc and show that CAGrad recovers GD and MGDA with proper cc values. Next, to test CAGrad on more complicated neural models, we perform the same set of experiments on the Multi-Fashion+MNIST benchmark \[[19](#bib.bib19 "")\] with a shrinked LeNet architecture \[[18](#bib.bib18 "")\] (in which each layer has a reduced number of neurons compared to the original LeNet). Please refer to Appendix [B](#A2 "Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") for more details.

For the toy optimization example, we modify the toy example used by Yu et al. \[[41](#bib.bib41 "")\] and consider θ\=(θ1,θ2)∈ℝ2\\theta=(\\theta\_{1},\\theta\_{2})\\in\\mathbb{R}^{2} with the following individual loss functions:

L1​(θ)\\displaystyle L\_{1}(\\theta)

\=c1​(θ)​f1​(θ)+c2​(θ)​g1​(θ)​and​L2​(θ)\=c1​(θ)​f2​(θ)+c2​(θ)​g2​(θ),where\\displaystyle=c\_{1}(\\theta)f\_{1}(\\theta)+c\_{2}(\\theta)g\_{1}(\\theta)~~\\text{and}~~L\_{2}(\\theta)=c\_{1}(\\theta)f\_{2}(\\theta)+c\_{2}(\\theta)g\_{2}(\\theta),~\\text{where}

f1​(θ)\\displaystyle f\_{1}(\\theta)

\=log⁡(max⁡(|0.5​(−θ1−7)−tanh⁡(−θ2)|,0.000005))+6,\\displaystyle=\\log{\\big(\\max(|0.5(-\\theta\_{1}-7)-\\tanh{(-\\theta\_{2})}|,~~0.000005)\\big)}+6, f2​(θ)\\displaystyle f\_{2}(\\theta)

\=log⁡(max⁡(|0.5​(−θ1+3)−tanh⁡(−θ2)+2|,0.000005))+6,\\displaystyle=\\log{\\big(\\max(|0.5(-\\theta\_{1}+3)-\\tanh{(-\\theta\_{2})}+2|,~~0.000005)\\big)}+6, g1​(θ)\\displaystyle g\_{1}(\\theta)

\=((−θ1+7)2+0.1∗(−θ2−8)2)/10−20,\\displaystyle=\\big((-\\theta\_{1}+7)^{2}+0.1\*(-\\theta\_{2}-8)^{2}\\big)/10-20, g2​(θ)\\displaystyle g\_{2}(\\theta)

\=((−θ1−7)2+0.1∗(−θ2−8)2)/10−20,\\displaystyle=\\big((-\\theta\_{1}-7)^{2}+0.1\*(-\\theta\_{2}-8)^{2})\\big/10-20, c1​(θ)\\displaystyle c\_{1}(\\theta)

\=max(tanh(0.5∗θ2),0)andc2(θ)\=max(tanh(−0.5∗θ2),0).\\displaystyle=\\max(\\tanh{(0.5\*\\theta\_{2})},~0)~~\\text{and}~~c\_{2}(\\theta)=\\max(\\tanh{(-0.5\*\\theta\_{2})},~0).

The average loss L0L\_{0} and individual losses L1L\_{1} and L2L\_{2} are shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). We then pick 5 initial parameter vectors θinit∈{(−8.5,7.5),(−8.5,5),(0,0),(9,9),(10,−8)}\\theta\_{\\text{init}}\\in\\{(-8.5,7.5),(-8.5,5),(0,0),(9,9),(10,-8)\\} and plot the corresponding optimization trajectories with different methods in Fig. [3](#S5.F3 "Figure 3 ‣ 5.1 Convergence and Ablation over c ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

![Refer to caption](2110.14048v2/toy_exp.png)

Figure 3: The left four plots are 5 runs of each algorithms from 5 different initial parameter vectors, where trajectories are colored from red to yellow. The right two plots are CAGrad’s results with a varying c∈{0,0.2,0.5,0.8,10}c\\in\\{0,0.2,0.5,0.8,10\\}.

As shown in Fig. [3](#S5.F3 "Figure 3 ‣ 5.1 Convergence and Ablation over c ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), GD gets stuck in 2 out of the 5 runs while other methods all converge to the Pareto set. MGDA and PCGrad converge to different Pareto-stationary points depending on θinit\\theta\_{\\text{init}}. CAGrad with c\=0c=0 recovers GD and CAGrad with c\=10c=10 approximates MGDA well (in theory it requires c→∞c\\rightarrow\\infty to exactly recover MGDA).

Next, we apply the same set of experiments on the multi-task classification benchmark Multi-Fashion+MNIST \[[19](#bib.bib19 "")\]. This benchmark consists of images that are generated by overlaying an image from FashionMNIST dataset \[[39](#bib.bib39 "")\] on top of another image from MNIST dataset \[[5](#bib.bib5 "")\]. The two images are positioned on the top-left and bottom-right separately. We consider a shrinked LeNet as our model, and train it with Adam \[[16](#bib.bib16 "")\] optimizer with a 0.0010.001 learning rate for 50 epochs using a batch size of 256. Due to the highly non-convex nature of the neural network, we are not able to visualize the entire Pareto set. But we provide the final training losses of different methods over three independent runs in Fig. [4](#S5.F4 "Figure 4 ‣ 5.1 Convergence and Ablation over c ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). As shown, CAGrad achieves the lowest average loss with c\=0.2c=0.2. In addition, PCGrad and MGDA focus on optimizing task 1 and task 2 separately. Lastly, CAGrad with c\=0c=0 and c\=10c=10 roughly recovers the final performance of GD and MGDA. By increasing cc, the model performance shifts from more GD-like to more MGDA-like, though due to the non-convex nature of neural networks, CAGrad with 0≤c<10\\leq c<1 does not necessarily converge to the exact same point.

![Refer to caption](2110.14048v2/mnist.png)

Figure 4: The average and individual training losses on the Fashion-and-MNIST benchmark by running GD, MGDA, PCGrad and CAGrad with different cc values. GD gets stuck at the steep valley (the area with a cloud of dots), which other methods can pass. MGDA and PCGrad converge randomly on the Pareto set.

### 5.2 Multi-task Supervised Learning

To answer question (3) in the supervised learning setting, we follow the experiment setup from Yu et al. \[[41](#bib.bib41 "")\] and consider the NYU-v2 and CityScapes vision datasets. NYU-v2 contains 3 tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. CityScapes similarly contains 2 tasks: 7-class semantic segmentation and depth estimation. Here, we follow  \[[41](#bib.bib41 "")\] and combine CAGrad with a state-of-the-art MTL method MTAN \[[21](#bib.bib21 "")\], which applies attention mechanism on top of the SegNet architecture \[[1](#bib.bib1 "")\]. We compare CAGrad with PCGrad, vanilla MTAN and Cross-Stitch \[[25](#bib.bib25 "")\], which is another MTL method that modifies the network architecture. MTAN originally experiments with equal loss weighting and two other dynamic loss weighting heuristics \[[15](#bib.bib15 ""), [3](#bib.bib3 "")\]. For a fair comparison, all methods are applied under the equal weighting scheme and we use the same training setup from \[[3](#bib.bib3 "")\]. We search c∈{0.1,0.2,…​0.9}c\\in\\{0.1,0.2,\\dots 0.9\\} with the best average training loss for CAGrad on both datasets (0.40.4 for NYU-v2 and 0.20.2 for Cityscapes). We perform a two-tailed, Student’s tt-test under *equal sample sizes, unequal variance* setup and mark the results that are significant with an ∗\*. Following Maninis et al.\[[24](#bib.bib24 "")\], we also compute the average per-task performance drop of method mm with respect to the single-tasking baseline bb: Δ​m\=1K​∑i\=1K(−1)li​(Mm,i−Mb,i)/Mb,i\\Delta m=\\frac{1}{K}\\sum\_{i=1}^{K}(-1)^{l\_{i}}(M\_{m,i}-M\_{b,i})/M\_{b,i} where li\=1l\_{i}=1 if a higher value is better for a criterion MiM\_{i} on task ii and 00 otherwise. The single-tasking baseline (independent) refers to training individual tasks with a vanilla SegNet. Results are shown in Tab. [1](#S5.T1 "Table 1 ‣ 5.2 Multi-task Supervised Learning ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and Tab. [2](#S5.T2 "Table 2 ‣ 5.2 Multi-task Supervised Learning ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

Segmentation

Depth

Surface Normal

#P.

Method

(Higher Better)

(Lower Better)

 Angle Distance (Lower Better) 

 Within t∘t^{\\circ} (Higher Better) 

Δ​m%↓\\Delta m\\%\\downarrow

mIoU

Pix Acc

Abs Err

Rel Err

Mean

Median

11.25

22.5

30

3

Independent

38.30

63.76

0.6754

0.2780

25.01

19.21

30.14

57.20

69.15

≈\\approx3

Cross-Stitch \[[25](#bib.bib25 "")\]

37.42

63.51

0.5487

0.2188

∗28.85

∗24.52

∗22.75

∗46.58

∗59.56

6.96

1.771.77

MTAN \[[21](#bib.bib21 "")\]

39.29

65.33

0.5493

0.2263

∗28.15

∗23.96

∗22.09

∗47.50

∗61.08

5.59

1.771.77

MGDA \[[30](#bib.bib30 "")\]

∗30.47

∗59.90

∗0.6070

∗0.2555

24.88

19.45

29.18

56.88

69.36

1.38

1.771.77

PCGrad \[[41](#bib.bib41 "")\]

38.06

64.64

0.5550

0.2325

∗27.41

∗22.80

∗23.86

∗49.83

∗63.14

3.97

1.771.77

GradDrop \[[4](#bib.bib4 "")\]

39.39

65.12

0.5455

0.2279

∗27.48

∗22.96

∗23.38

∗49.44

∗62.87

3.58

1.771.77

CAGrad (ours)

39.79

65.49

0.5486

0.2250

26.31

21.58

25.61

52.36

65.58

0.20

Table 1: Multi-task learning results on NYU-v2 dataset. #P denotes the relative model size compared to the vanilla SegNet. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result among all multi-task methods is marked in bold. MGDA, PCGrad, GradDrop and CAGrad are applied on the MTAN backbone. CAGrad has statistically significant improvement over baselines methods with an ∗\*, tested with a pp-value of 0.1.

Segmentation

Depth

#P.

Method

(Higher Better)

(Lower Better)

Δ​m%↓\\Delta m\\%\\downarrow

mIoU

Pix Acc

Abs Err

Rel Err

2

Independent

74.01

93.16

0.0125

27.77

≈\\approx3

Cross-Stitch \[[25](#bib.bib25 "")\]

∗73.08

∗92.79

∗0.0165

∗118.5

90.02

1.771.77

MTAN \[[21](#bib.bib21 "")\]

75.18

93.49

∗0.0155

∗46.77

22.60

1.771.77

MGDA \[[30](#bib.bib30 "")\]

∗68.84

∗91.54

0.0309

33.50

44.14

1.771.77

PCGrad \[[41](#bib.bib41 "")\]

75.13

93.48

0.0154

42.07

18.29

1.771.77

GradDrop \[[4](#bib.bib4 "")\]

75.27

93.53

∗0.0157

∗47.54

23.73

1.771.77

CAGrad (ours)

75.16

93.48

0.0141

37.60

11.64

Table 2: Multi-task learning results on CityScapes Challenge. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result among all multi-task methods is marked in bold. PCGrad and CAGrad are applied on the MTAN backbone. CAGrad has statistically significant improvement over baselines methods with an ∗\*, tested with a pp-value of 0.1.

Given the single task performance, CAGrad performs better on the task that is overlooked by other methods (Surface Normal in NYU-v2 and Depth in CityScapes) and matches other methods’ performance on the rest of the tasks. We also provide the final test losses and the per-epoch training time of each method in Fig. [5](#A2.F5 "Figure 5 ‣ Experiment Details ‣ B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") in Appendix [B.2](#A2.SS2 "B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

### 5.3 Multi-task Reinforcement Learning

To answer question (3) in the reinforcement learning (RL) setting, we apply CAGrad on the MT10 and MT50 benchmarks from the Meta-World environment \[[42](#bib.bib42 "")\]. In particular, MT10 and MT50 contains 10 and 50 robot manipulation tasks. Following \[[33](#bib.bib33 "")\], we use Soft Actor-Critic (SAC) \[[10](#bib.bib10 "")\] as the underlying RL training algorithm. We compare against Multi-task SAC (SAC with a shared model), Multi-headed SAC (SAC with a shared backbone and task-specific head), Multi-task SAC + Task Encoder (SAC with a shared model and the input includes a task embedding) \[[42](#bib.bib42 "")\] and PCGrad \[[41](#bib.bib41 "")\]. We also compare with Soft Modularization \[[40](#bib.bib40 "")\] that routes different modules in a shared model to form different policies. Lastly, we also include a recent method (CARE) that considers language metadata and uses a mixture of expert encoder for MTL. We follow the same experiment setup from \[[33](#bib.bib33 "")\]. The results are shown in Tab. [3](#S5.T3 "Table 3 ‣ 5.3 Multi-task Reinforcement Learning ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). CAGrad outperforms all baselines except for CARE which benefits from extra information from the metadata. We also apply the practical speedup in Sec. [3.3](#S3.SS3 "3.3 Practical Speedup ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and sub-sample 4 and 8 tasks for MT10 and MT50 (CAGrad-Fast). CAGrad-fast achieves comparable performance against the state-of-the-art method while achieving a 22x (MT10) and 55x (MT50) speedup over PCGrad. We provide a visualization of tasks from MT10 and MT50, and the comparison of computational efficiency in Appendix [B.3](#A2.SS3 "B.3 Multi-task Reinforcement Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

Metaworld MT10

Metaworld MT50

Method

success

success

(mean ±\\pm stderr)

(mean ±\\pm stderr)

Multi-task SAC \[[42](#bib.bib42 "")\]

0.49 ±\\pm0.073

0.36 ±\\pm0.013

Multi-task SAC + Task Encoder \[[42](#bib.bib42 "")\]

0.54 ±\\pm0.047

0.40 ±\\pm0.024

Multi-headed SAC \[[42](#bib.bib42 "")\]

0.61 ±\\pm0.036

0.45 ±\\pm0.064

PCGrad \[[41](#bib.bib41 "")\]

0.72 ±\\pm0.022

0.50 ±\\pm0.017

Soft Modularization \[[40](#bib.bib40 "")\]

0.73 ±\\pm0.043

0.50 ±\\pm0.035

CAGrad (ours)

0.83 ±\\pm0.045

0.52 ±\\pm0.023

CAGrad-Fast (ours)

0.82 ±\\pm0.039

0.50 ±\\pm0.016

CARE \[[33](#bib.bib33 "")\]

0.84 ±\\pm0.051

0.54 ±\\pm0.031

One SAC agent per task (upper bound)

0.90 ±\\pm0.032

0.74 ±\\pm0.041

Table 3: Multi-task reinforcement learning results on the Metaworld benchmarks. Results are averaged over 10 independent runs and the best result is marked in bold.

### 5.4 Semi-supervised Learning with Auxiliary Tasks

Training with auxiliary tasks to improve the performance of a main task is another popular application of MTL. Here, we take semi-supervised learning as an instance. We combine different optimization algorithms with Auxiliary Task Reweighting for Minimum-data Learning (ARML) \[[31](#bib.bib31 "")\], a state-of-the-art semi-supervised learning algorithm. The loss function is composed of the main task and two auxiliary tasks:

L0\=LC​E​(θ,Dl)+w1​La​u​x1​(θ,Du)+w2​La​u​x2​(θ,Du),L\_{0}=L\_{CE}(\\theta;D\_{l})+w\_{1}L\_{aux}^{1}(\\theta;D\_{u})+w\_{2}L\_{aux}^{2}(\\theta;D\_{u}),

(5)

where LC​EL\_{CE} is the main cross-entropy classification loss on the labeled dataset DlD\_{l}, and La​u​x1,La​u​x2L\_{aux}^{1},L\_{aux}^{2} are auxiliary unsupervised learning losses on the unlabeled dataset DuD\_{u}. We use the same w1w\_{1} and w2w\_{2} from ARML, and use the CIFAR10 dataset \[[17](#bib.bib17 "")\], which contains 50,000 training images and 10,000 test images. 10% of the training images is held out as the validation set. We test PCGrad, MGDA and CAGrad with 500, 1000 and 2000 labeled images. The rest of the training set is used for auxiliary tasks. For all the methods, we use the same labeled dataset, the same learning rate and train them for 200 epochs with the Adam \[[16](#bib.bib16 "")\] optimizer. Please refer to Appendix [B.4](#A2.SS4 "B.4 Semi-Supervised Learning with Auxiliary Tasks ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") for more experimental details. Results are shown in Tab. [4](#S5.T4 "Table 4 ‣ 5.4 Semi-supervised Learning with Auxiliary Tasks ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). With all the different number of labels, CAGrad yields the best averaged test accuracy. We observed that MGDA performs much worse than the ARML baseline, because it significantly overlooks the main classification task. We also compare different gradient manipulation methods on the same task with GradNorm \[[3](#bib.bib3 "")\], which dynamically adjusts w1w\_{1} and w2w\_{2} during training. The results and conclusions are similar to those for ARML.

Method

500 labels

1000 labels

2000 labels

ARML \[[31](#bib.bib31 "")\]

67.05 ±\\pm0.16

73.22 ±\\pm0.26

81.35 ±\\pm0.36

ARML + PCGrad \[[41](#bib.bib41 "")\]

67.49 ±\\pm0.64

73.23 ±\\pm0.62

81.91 ±\\pm0.19

ARML + MGDA \[[30](#bib.bib30 "")\]

49.27 ±\\pm0.68

60.11 ±\\pm2.35

60.78 ±\\pm0.17

ARML + CAGrad (Ours)

68.25 ±\\pm0.37

74.37 ±\\pm0.42

82.81 ±\\pm0.48

GradNorm  \[[3](#bib.bib3 "")\]

67.35 ±\\pm0.15

73.53 ±\\pm0.23

81.03 ±\\pm0.71

GradNorm + PCGrad \[[41](#bib.bib41 "")\]

67.83 ±\\pm0.19

73.91 ±\\pm0.09

82.72 ±\\pm0.19

GradNorm + MGDA \[[30](#bib.bib30 "")\]

36.99 ±\\pm2.11

57.94 ±\\pm0.92

59.12 ±\\pm0.63

GradNorm + CAGrad (Ours)

67.53 ±\\pm0.26

74.72 ±\\pm0.19

83.15 ±\\pm0.56

Table 4: Semi-supervised Learning with auxiliary tasks on CIFAR10. We report the average test accuracy over 3 independent runs for each method and mark the best result in bold.

## 6 Conclusion

In this work, we introduce the Conflict-Averse Gradient descent (CAGrad) algorithm that explicitly optimizes the minimum decrease rate of any specific task’s loss while still provably converging to the optimum of the average loss. CAGrad generalizes the gradient descent and multiple gradient descent algorithm, and demonstrates improved performance across several challenging multi-task learning problems compared to the state-of-the-art methods. While we focus mainly on optimizing the average loss, an interesting future direction is to look at main objectives other than the average loss under the multi-task setting.

## Acknowledgements

The research was conducted in the statistical learning and AI group (SLAI) and the Learning Agents Research Group (LARG) in computer science at UT Austin. SLAI research is supported in part by CAREER-1846421, SenSE-2037267, EAGER-2041327, and Office of Navy Research, and NSF AI Institute for Foundations of Machine Learning (IFML). LARG research is supported in part by NSF (CPS-1739964, IIS-1724157, FAIN-2019844), ONR (N00014-18-2243), ARO (W911NF-19-2-0333), DARPA, Lockheed Martin, GM, Bosch, and UT Austin’s Good Systems grand challenge. Peter Stone serves as the Executive Director of Sony AI America and receives financial compensation for this work. The terms of this arrangement have been reviewed and approved by the University of Texas at Austin in accordance with its policy on objectivity in research. Xingchao Liu is supported in part by a funding from BP.

## References

*   \[1\] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.
*   \[2\] Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.
*   \[3\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pages 794–803. PMLR, 2018.
*   \[4\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. arXiv preprint arXiv:2010.06808, 2020.
*   \[5\] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.
*   \[6\] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012.
*   \[7\] Steven Diamond and Stephen Boyd. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83):1–5, 2016.
*   \[8\] Dibya Ghosh, Avi Singh, Aravind Rajeswaran, Vikash Kumar, and Sergey Levine. Divide-and-conquer reinforcement learning. arXiv preprint arXiv:1711.09874, 2017.
*   \[9\] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European Conference on Computer Vision (ECCV), pages 270–287, 2018.
*   \[10\] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, pages 1861–1870. PMLR, 2018.
*   \[11\] Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. arXiv preprint arXiv:1611.01587, 2016.
*   \[12\] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
*   \[13\] Harold M Hochman and James D Rodgers. Pareto optimal redistribution. The American economic review, 59(4):542–557, 1969.
*   \[14\] Adrián Javaloy and Isabel Valera. Rotograd: Dynamic gradient homogenization for multi-task learning. arXiv preprint arXiv:2103.02631, 2021.
*   \[15\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
*   \[16\] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
*   \[17\] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
*   \[18\] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
*   \[19\] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, and Sam Kwong. Pareto multi-task learning. arXiv preprint arXiv:1912.12854, 2019.
*   \[20\] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2020.
*   \[21\] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1871–1880, 2019.
*   \[22\] Xingchao Liu, Xing Han, Na Zhang, and Qiang Liu. Certified monotonic neural networks. arXiv preprint arXiv:2011.10219, 2020.
*   \[23\] Debabrata Mahapatra and Vaibhav Rajan. Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization. In International Conference on Machine Learning, pages 6597–6607. PMLR, 2020.
*   \[24\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1851–1860, 2019.
*   \[25\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3994–4003, 2016.
*   \[26\] Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. arXiv preprint arXiv:1511.06342, 2015.
*   \[27\] Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv preprint arXiv:1711.01239, 2017.
*   \[28\] Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.
*   \[29\] Andrei A Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. arXiv preprint arXiv:1511.06295, 2015.
*   \[30\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. arXiv preprint arXiv:1810.04650, 2018.
*   \[31\] Baifeng Shi, Judy Hoffman, Kate Saenko, Trevor Darrell, and Huijuan Xu. Auxiliary task reweighting for minimum-data learning. Advances in Neural Information Processing Systems, 33, 2020.
*   \[32\] Shagun Sodhani and Amy Zhang. Mtrl - multi task rl algorithms. Github, 2021.
*   \[33\] Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-task reinforcement learning with context-based representations. arXiv preprint arXiv:2102.06177, 2021.
*   \[34\] Charles Stein. Inadmissibility of the usual estimator for the mean of a multivariate normal distribution. In Contribution to the Theory of Statistics, pages 197–206. University of California Press, 2020.
*   \[35\] Kevin Swersky, Jasper Snoek, and Ryan Prescott Adams. Multi-task bayesian optimization. 2013.
*   \[36\] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 1195–1204, 2017.
*   \[37\] Yee Whye Teh, Victor Bapst, Wojciech Marian Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. arXiv preprint arXiv:1707.04175, 2017.
*   \[38\] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
*   \[39\] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.
*   \[40\] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. arXiv preprint arXiv:2003.13661, 2020.
*   \[41\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. arXiv preprint arXiv:2001.06782, 2020.
*   \[42\] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pages 1094–1100. PMLR, 2020.
*   \[43\] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3712–3722, 2018.
*   \[44\] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 2021.

## Checklist

1.  1.

    For all authors…

    1.  (a)

        Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? \[Yes\] See Sec. [3.2](#S3.SS2 "3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning") for the convergence analysis, Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conflict-Averse Gradient Descent for Multi-task Learning") for the challenges faced by previous methods, and Sec. [5](#S5 "5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning") for empirical evaluation of these challenges and the advantage of CAGrad.

    2.  (b)

        Did you describe the limitations of your work? \[Yes\] See Sec. [6](#S6 "6 Conclusion ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). Currently we mainly focus on optimizing the average loss, which could be replaced by other main objectives.

    3.  (c)

        Did you discuss any potential negative societal impacts of your work? \[N/A\] Our method does not have potential negative societal impacts.

    4.  (d)

        Have you read the ethics review guidelines and ensured that your paper conforms to them? \[Yes\]
2.  2.

    If you are including theoretical results…

    1.  (a)

        Did you state the full set of assumptions of all theoretical results? \[Yes\] The assumptions are stated in Thm. [3.2](#S3.Thmthm2 "Theorem 3.2 (Convergence of CAGrad). ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

    2.  (b)

        Did you include complete proofs of all theoretical results? \[Yes\] The complete proof is included in Appendix A.3.
3.  3.

    If you ran experiments…

    1.  (a)

        Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? \[Yes\] We mention most of the details to reproduce the result in Sec. [5](#S5 "5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and provide the rest of details of each experiment in Appendix.B. The code comes with the supplementary material.

    2.  (b)

        Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? \[Yes\] See Appendix.B and Sec. [5](#S5 "5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

    3.  (c)

        Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? \[Yes\] For each experiment except for the toy (since there is no stochasticity), we run over multiple (≥3\\geq 3) seeds.

    4.  (d)

        Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? \[Yes\] We explicitly compare the computational efficiency in Fig. [5](#A2.F5 "Figure 5 ‣ Experiment Details ‣ B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). More details on the resources are provided in the corresponding sections in Appendix.B.
4.  4.

    If you are using existing assets (e.g., code, data, models) or curating/releasing new assets…

    1.  (a)

        If your work uses existing assets, did you cite the creators? \[Yes\] For most of the experiment, we follow the exact experiment setup and use the corresponding open-source code from previous works and have cited and compared against them.

    2.  (b)

        Did you mention the license of the assets? \[Yes\] All code and data are publicly available under MIT license

    3.  (c)

        Did you include any new assets either in the supplemental material or as a URL? \[No\] No new assets are introduced for our experiment. The only thing we modified is a shrinked LeNet, where the details are provided in Appendix.B.

    4.  (d)

        Did you discuss whether and how consent was obtained from people whose data you’re using/curating? \[N/A\]

    5.  (e)

        Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? \[N/A\] The data we use are publicly available data that has been used by a lot of prior research. There should be no personally identifiable information or offensive content.
5.  5.

    If you used crowdsourcing or conducted research with human subjects…

    1.  (a)

        Did you include the full text of instructions given to participants and screenshots, if applicable? \[N/A\] No human subjects involved.

    2.  (b)

        Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? \[N/A\]

    3.  (c)

        Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? \[N/A\]
## Appendix A Algorithm Details

In this section, we first formally introduce the Multiple Gradient Descent Algorithm and the Projecting Conflicting Gradients method. Then we provide the full proof of Thm. [3.2](#S3.Thmthm2 "Theorem 3.2 (Convergence of CAGrad). ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

### A.1 Multiple Gradient Descent Algorithm (MGDA)

The Multiple Gradient Descent Algorithm (MGDA) explicitly optimizes towards a Pareto-optimal point for multiple objectives (See the definition [3.1](#S3.Thmthm1 "Definition 3.1 (Pareto optimal and stationary points). ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). It is known that a necessary condition for θ\\theta to be a Pareto-optimal point is that we could find a convex combination of the task gradients at θ\\theta that results in the 00 vector. Therefore, MGDA proposes to minimize the minimum possible convex combination of task gradients:

min⁡12​‖∑i\=1Kwi​gi‖2,s.t.​∑i\=1Kwi\=1,and​∀i,wi≥0.\\min\\frac{1}{2}\\left\\lVert\\sum\_{i=1}^{K}w\_{i}g\_{i}\\right\\rVert^{2},~~\\text{s.t.}~~\\sum\_{i=1}^{K}w\_{i}=1,~\\text{and}~~\\forall i,w\_{i}\\geq 0.

(6)

We call this the *dual* objective for MGDA, as the primal objective of MGDA has a close connection to CAGrad’s primal objective in Eq. ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")). Specifically, the *primal* objective of MGDA is

max‖d‖≤1⁡mini​⟨d,gi⟩.\\max\_{\\left\\lVert d\\right\\rVert\\leq 1}\\min\_{i}\\langle d,g\_{i}\\rangle.

(7)

To see the primal-dual relationship, denote gw\=∑iwi​gig\_{w}=\\sum\_{i}w\_{i}g\_{i}, where w∈𝒲≜{w∈ℝK:∑iwi\=1,wi≥0,∀i∈\[K\]}.w\\in\\mathcal{W}\\triangleq\\{w\\in\\mathbb{R}^{K}\\colon~~\\sum\_{i}w\_{i}=1,~~w\_{i}\\geq 0,\\forall i\\in\[K\]\\}. Note that mini⁡⟨gi,d⟩\=minw∈𝒲⁡⟨∑iwi​gi,d⟩\\min\_{i}\\langle g\_{i},d\\rangle=\\min\_{w\\in\\mathcal{W}}\\langle\\sum\_{i}w\_{i}g\_{i},d\\rangle. The Lagrangian of Eq. ([7](#A1.E7 "In A.1 Multiple Gradient Descent Algorithm (MGDA) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) is

maxd⁡minλ≥0,w∈𝒲​⟨d,gw⟩−λ2​(‖d‖2−1).\\max\_{d}\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}\\langle d,g\_{w}\\rangle-\\frac{\\lambda}{2}(\\left\\lVert d\\right\\rVert^{2}-1).

(8)

Since the problem is a convex programming and the Slater’s condition holds when c\>0c>0 (On the other hand, if c\=0c=0, then it is easy to check that all the results hold trivially), the strong duality holds and we can exchange the min and max:

minλ≥0,w∈𝒲⁡maxd​⟨d,gw⟩−λ2​(‖d‖2−1).\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}\\max\_{d}\\langle d,g\_{w}\\rangle-\\frac{\\lambda}{2}(\\left\\lVert d\\right\\rVert^{2}-1).

(9)

The optimal d∗\=gw/λd^{\*}=g\_{w}/\\lambda and the resulting primal objective is therefore

minλ≥0,w∈𝒲⁡λ⁡(12​‖gw‖2+1).\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}\\lambda(\\frac{1}{2}\\left\\lVert g\_{w}\\right\\rVert^{2}+1).

(10)

Here, λ\\lambda corresponds to the constraint ‖d‖≤1\\left\\lVert d\\right\\rVert\\leq 1. If we fix λ\\lambda to be any constant, then we recover the dual objective in Eq. ([6](#A1.E6 "In A.1 Multiple Gradient Descent Algorithm (MGDA) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")).

#### Remark

Looking at the primal form of MGDA in Eq. ([7](#A1.E7 "In A.1 Multiple Gradient Descent Algorithm (MGDA) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")), the major difference between MGDA and CAGrad is that the new update vector dd is searched around the 00 vector for MGDA and g0g\_{0} for CAGrad. Therefore, theoretically both MGDA and CAGrad optimizes the worst local update, but MGDA is more conservative and can converge to any point on the Pareto set without explicit control (See Thm. 2 from  \[[6](#bib.bib6 "")\]). This also explains MGDA’s behavior in practice that it often learns much slower than other methods.

### A.2 Projecting Conflicting Gradients (PCGrad)

Identifying that a major challenge for multi-task optimization is the conflicting gradient, Yu et al. \[[41](#bib.bib41 "")\] propose to project each task gradient to the normal plane of others before combining them together to form the final update vector. In the following, we provide the full algorithm of the Projecting Conflicting Gradients (PCGrad):

Algorithm 2 Projecting Conflicting Gradient Update Rule

 Input: model parameter vector θ\\theta and differentiable loss functions {Li}i\=1K\\{L\_{i}\\}\_{i=1}^{K}. 

 gi←∇θLi​(θ)g\_{i}\\leftarrow\\nabla\_{\\theta}L\_{i}(\\theta). 

 giPC\=gi,∀ig^{\\text{PC}}\_{i}=g\_{i},~\\forall i. 

 for task i∈\[K\]i\\in\[K\] do 

  for j≠i∈\[K\]j\\neq i\\in\[K\] in random order do 

   if giPC⋅gj<0g\_{i}^{\\text{PC}}\\cdot g\_{j}<0 then 

    giPC\=giPC−giPC⋅gj‖gj‖2​gjg\_{i}^{\\text{PC}}=g\_{i}^{\\text{PC}}-\\frac{g\_{i}^{\\text{PC}}\\cdot g\_{j}}{\\left\\lVert g\_{j}\\right\\rVert^{2}}g\_{j}. 

   end if 

  end for 

 end for 

 Return the new update vector d\=gPC\=1K​∑igiPCd=g^{\\text{PC}}=\\frac{1}{K}\\sum\_{i}g\_{i}^{\\text{PC}}. 

Fig. [2](#S3.F2 "Figure 2 ‣ 3.2 Convergence Analysis ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning") provides a visualization of PCGrad’s update rule for two-task learning (the 3rd column). Different from MGDA and CAGrad, PCGrad does not have a clear optimization objective at each step, which makes it hard to analyze PCGrad’s convergence guarantee in general. In practice, the random ordering to do the projection is particularly important for PCGrad to work well \[[41](#bib.bib41 "")\], which suggests that the intuition of removing the “conflicting" part of each gradient might not be always correct. For the convergence analysis, Yu et al. establishes the convergence guarantee for PCGrad only under the two-task learning setting. Moreover, PCGrad is only guaranteed to converge to the Pareto set without explict control over which point it will arrive at (See Thm. [A.1](#A1.Thmthm1 "Theorem A.1 (Convergence of PCGrad []). ‣ A.2 Projecting Conflicting Gradients (PCGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") in the following).

###### Theorem A.1 (Convergence of PCGrad \[[41](#bib.bib41 "")\]).

Consider two-task learning, assume the loss functions L1L\_{1} and L2L\_{2} are convex and differentiable. Suppose the gradient of L0\=(L1+L2)/2L\_{0}=(L\_{1}+L\_{2})/2 is HH-Lipschitz with H\>0H>0. Then, the PCGrad update rule with step size t≤1/Ht\\leq 1/H will converge to a Pareto-stationary point.

### A.3 Conflit-Averse Gradient descent (CAGrad)

We provide the full derivation of CAGrad and the proof for its convergence in this section. Our proof assumes L0L\_{0} is a general function with gradient g0\=∇L0g\_{0}=\\nabla L\_{0}, that is, it does not have to be the average of LiL\_{i} as the case we focus on in the main paper.

###### Lemma A.2.

Let d∗d^{\*} be the solution of

maxd∈ℝm⁡mini∈\[K\]​gi⊤​d​s.t.‖g0−d‖≤c⁡‖g0‖,\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[K\]}g\_{i}^{\\top}d~~~s.t.~~~\\left\\lVert g\_{0}-d\\right\\rVert\\leq c\\left\\lVert g\_{0}\\right\\rVert, where c≥0c\\geq 0, and g0,g1,…,gK∈ℝmg\_{0},g\_{1},\\ldots,g\_{K}\\in\\mathbb{R}^{m}. Then we have

d∗\=g0+c​‖g0‖‖gw∗‖​gw∗,d^{\*}=g\_{0}+\\frac{c\\left\\lVert g\_{0}\\right\\rVert}{\\left\\lVert g\_{w^{\*}}\\right\\rVert}g\_{w^{\*}}, where gw∗\=∑iwi∗​gig\_{w^{\*}}=\\sum\_{i}w^{\*}\_{i}g\_{i} and w∗w^{\*} is the solution of

minw≥𝒲⁡gw⊤​g0+c⁡‖g0‖​‖gw‖,\\displaystyle\\min\_{w\\geq\\mathcal{W}}g\_{w}^{\\top}g\_{0}+c\\left\\lVert g\_{0}\\right\\rVert\\left\\lVert g\_{w}\\right\\rVert,

(11)

where 𝒲\={w∈ℝK:∑iwi\=1,wi≥0,∀i∈\[K\]}.\\mathcal{W}=\\{w\\in\\mathbb{R}^{K}\\colon~~\\sum\_{i}w\_{i}=1,~~w\_{i}\\geq 0,\\forall i\\in\[K\]\\}. In addition, mini⁡gi⊤​d∗\=gw∗⊤​g0+c⁡‖g0‖​‖gw∗‖.\\displaystyle\\min\_{i}g\_{i}^{\\top}d^{\*}=g\_{w^{\*}}^{\\top}g\_{0}+c\\left\\lVert g\_{0}\\right\\rVert\\left\\lVert g\_{w^{\*}}\\right\\rVert.

(12)

###### Proof.

Denote ϕ\=c2​‖g0‖2\\phi=c^{2}\\left\\lVert g\_{0}\\right\\rVert^{2}. Note that mini⁡⟨gi,d⟩\=minw∈𝒲⁡⟨∑iwi​gi,d⟩\\min\_{i}\\langle g\_{i},d\\rangle=\\min\_{w\\in\\mathcal{W}}\\langle\\sum\_{i}w\_{i}g\_{i},d\\rangle. The Lagrangian of the objective in Eq. ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) is

maxd∈ℝm⁡minλ≥0,w∈𝒲​gw⊤​d−λ2​(‖g0−d‖2−ϕ).\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}g\_{w}^{\\top}d-\\frac{\\lambda}{2}(\\left\\lVert g\_{0}-d\\right\\rVert^{2}-\\phi).

Since the problem is a convex programming and the Slater’s condition holds when c\>0c>0 (On the other hand, if c\=0c=0, then it is easy to check that all the results hold trivially), the strong duality holds and we can exchange the min and max:

minλ≥0,w∈𝒲⁡maxd∈ℝm​gw⊤​d−λ2​‖g0−d‖2+λ​ϕ2.\\min\_{\\lambda\\geq 0,w\\in\\mathcal{W}}\\max\_{d\\in\\mathbb{R}^{m}}g\_{w}^{\\top}d-\\frac{\\lambda}{2}\\left\\lVert g\_{0}-d\\right\\rVert^{2}+\\frac{\\lambda\\phi}{2}.

With λ,w\\lambda,w fixing, the optimal dd is achieved when d\=g0+gw/λd=g\_{0}+g\_{w}/\\lambda, yielding the following dual problem

minw,λ≥0⁡gw⊤​(g0+gw/λ)−λ2​‖gw/λ‖2+λ2​ϕ.\\min\_{w,\\lambda\\geq 0}g\_{w}^{\\top}(g\_{0}+g\_{w}/\\lambda)-\\frac{\\lambda}{2}\\left\\lVert g\_{w}/\\lambda\\right\\rVert^{2}+\\frac{\\lambda}{2}\\phi.

This is equivalent to

minw,λ≥0⁡gw⊤​g0+12​λ​‖gw‖2+λ​ϕ2.\\min\_{w,\\lambda\\geq 0}g\_{w}^{\\top}g\_{0}+\\frac{1}{2\\lambda}\\left\\lVert g\_{w}\\right\\rVert^{2}+\\frac{\\lambda\\phi}{2}.

Optimizing out the λ\\lambda we have

minw∈𝒲⁡gw⊤​g0+ϕ​‖gw‖,\\min\_{w\\in\\mathcal{W}}g\_{w}^{\\top}g\_{0}+\\sqrt{\\phi}\\left\\lVert g\_{w}\\right\\rVert, where the optimal λ\=‖gw‖/ϕ1/2\\lambda=\\left\\lVert g\_{w}\\right\\rVert/\\phi^{1/2}. This solves the problem. ([12](#A1.E12 "In Lemma A.2. ‣ A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) is the consequence of the strong duality. ∎

#### Convergence Analysis

###### Assumption A.3.

Assume individual loss functions L0,L1,…,LKL\_{0},L\_{1},\\ldots,L\_{K} are differentiable on ℝm\\mathbb{R}^{m} and their gradients ∇Li​(θ)\\nabla L\_{i}(\\theta) are all HH-Lipschitz, i.e. ‖∇Li​(x)−∇Li​(y)‖≤H⁡‖x−y‖\\left\\lVert\\nabla L\_{i}(x)-\\nabla L\_{i}(y)\\right\\rVert\\leq H\\left\\lVert x-y\\right\\rVert for i\=0,1,…,Ki=0,1,\\ldots,K, where H∈(0,∞)H\\in(0,\\infty). Assume L0∗\=infθ∈ℝmL0​(θ)\>−∞L\_{0}^{\*}=\\inf\_{\\theta\\in\\mathbb{R}^{m}}L\_{0}(\\theta)>-\\infty.

###### Theorem A.4 (Convergence of CAGrad).

Assume Assumption [A.3](#A1.Thmthm3 "Assumption A.3. ‣ Convergence Analysis ‣ A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") holds. With a fixed step size α\\alpha satisfying 0<α≤1/H0<\\alpha\\leq 1/H, we have for the CAGrad in Alg. [1](#alg1 "Algorithm 1 ‣ Dual Objective ‣ 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"):

1) If 0≤c<10\\leq c<1, then CAGrad converges to stationary points of L0L\_{0} convergence rate in that

∑t\=0T‖g0​(θt)‖2≤2​(L0​(θ0)−L0∗)α⁡(1−c2).\\sum\_{t=0}^{T}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}\\leq\\frac{2(L\_{0}(\\theta\_{0})-L\_{0}^{\*})}{{\\alpha}(1-c^{2})}.

2) For any c≥0c\\geq 0, all the fixed point of CAGrad are Pareto-stationary points of (L0,L1,…,LK)(L\_{0},L\_{1},\\ldots,L\_{K}).

###### Proof.

We will first prove 1). Consider the tt-th optimization step and denote d∗​(θt)d^{\*}(\\theta\_{t}) the update direction obtained by solving ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) at the tt-th iteration. Then we have

L0​(θt+1)−L0​(θt)\\displaystyle L\_{0}(\\theta\_{t+1})-L\_{0}(\\theta\_{t})

\=L0​(θt−α​d∗​(θt))−L0​(θt)\\displaystyle=L\_{0}(\\theta\_{t}-\\alpha d^{\*}(\\theta\_{t}))-L\_{0}(\\theta\_{t})

≤−α​g0​(θt)⊤​d∗​(θt)+H​α22​‖d∗​(θt)‖2\\displaystyle\\leq-\\alpha g\_{0}(\\theta\_{t})^{\\top}d^{\*}(\\theta\_{t})+\\frac{H\\alpha^{2}}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}

≤−αg0(θt)⊤d∗(θt)+α2‖d∗(θt)‖2//α≤1/H\\displaystyle\\leq-\\alpha g\_{0}(\\theta\_{t})^{\\top}d^{\*}(\\theta\_{t})+\\frac{\\alpha}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}~~~~~~~\\text{{\\color\[rgb\]{1,0,1}//$\\alpha\\leq 1/H$}}

≤−α2(‖g0(θt‖2+‖d∗(θt)‖2−‖g0(θt)−d∗(θt)‖2)+α2‖d∗(θt)‖2\\displaystyle\\leq-\\frac{\\alpha}{2}\\left(\\left\\lVert g\_{0}(\\theta\_{t}\\right\\rVert^{2}+\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}-\\left\\lVert g\_{0}(\\theta\_{t})-d^{\*}(\\theta\_{t})\\right\\rVert^{2}\\right)+\\frac{\\alpha}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}

\=−α2​(‖g0​(θt)‖2−‖d∗​(θt)−g0​(θt)‖2)\\displaystyle=-\\frac{\\alpha}{2}\\left(\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}-\\left\\lVert d^{\*}(\\theta\_{t})-g\_{0}(\\theta\_{t})\\right\\rVert^{2}\\right)

≤−α2(1−c2)‖g0(θt)‖2//by the constraint in ([3](#S3.E3 "In 3.1 Conflict-Averse Gradient Descent ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"))\\displaystyle\\leq-\\frac{\\alpha}{2}(1-c^{2})\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}~~~~~~~~~~\\text{{\\color\[rgb\]{1,0,1}//by the constraint in \\eqref{eq:CAGrad}}}

Using telescoping sums, we have L0(θT+1)−L0(0)\=−(α/2)(1−c2)∑t\=0T‖g0(θt)‖2L\_{0}(\\theta\_{T+1})-L\_{0}(0)=-(\\alpha/2)(1-c^{2})\\sum\_{t=0}^{T}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}. Therefore

mint≤T⁡‖g0​(θt)‖2≤1T+1​∑t\=0T‖g0​(θt)‖2≤2​(L0​(0)−L0​(θT+1))α⁡(1−c2)​(T+1).\\min\_{t\\leq T}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}\\leq\\frac{1}{T+1}\\sum\_{t=0}^{T}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}\\leq\\frac{2(L\_{0}(0)-L\_{0}(\\theta\_{T+1}))}{{\\alpha}(1-c^{2})(T+1)}.

Therefore, if L0L\_{0} is lower bounded, that is, L0∗:=infθ∈ℝmL0​(θ)\>−∞L\_{0}^{\*}:=\\inf\_{\\theta\\in\\mathbb{R}^{m}}L\_{0}(\\theta)>-\\infty, then mint≤T⁡‖g0​(θt)‖2\=O⁡(1/T)\\min\_{t\\leq T}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}=O(1/T).

For general c≥0c\\geq 0, in the fixed point, we have d∗​(θ)\=g0​(θ)+λ​gw∗​(θ)\=0d^{\*}(\\theta)=g\_{0}(\\theta)+\\lambda g\_{w^{\*}}(\\theta)=0, which readily match the definition of Pareto Stationarity. ∎

In the following, we show an additional result that when c≥1c\\geq 1, and we use a properly decaying step size, the limit points of CAGrad are either stationary points of L0L\_{0}, or Pareto-stationary points of (L1,…,LK)(L\_{1},\\ldots,L\_{K}).

###### Theorem A.5.

Under Assumption [A.3](#A1.Thmthm3 "Assumption A.3. ‣ Convergence Analysis ‣ A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), assume c≥1c\\geq 1 and we a time varying step size satisfying

αt≤‖gwt∗​(θt)‖H⁡(c−1)​‖g0​(θt)‖,\\alpha\_{t}\\leq\\frac{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}{H(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}, where wt∗w^{\*}\_{t} is the solution of ([11](#A1.E11 "In Lemma A.2. ‣ A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")) at the tt-th iteration, then we have

∑t\=0Tαt​‖g0​(θt)‖​‖gwt∗​(θt)‖≤2​mini⁡(Li​(θ0)−Li​(θT+1))(c−1).\\displaystyle\\sum\_{t=0}^{T}\\alpha\_{t}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert\\leq 2\\frac{\\min\_{i}(L\_{i}(\\theta\_{0})-L\_{i}(\\theta\_{T+1}))}{(c-1)}.

Therefore, if we hae Li∗\=infθ∈ℝmL⁡(θ)\>−∞L\_{i}^{\*}=\\inf\_{\\theta\\in\\mathbb{R}^{m}}L(\\theta)>-\\infty and c\>1c>1, then we have αt​‖g0​(θt)‖​‖gwt∗​(θt)‖→0\\alpha\_{t}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert\\to 0 as t→∞t\\to\\infty, meaning that we have either αt→0\\alpha\_{t}\\to 0, or ‖g0​(θt)‖→0\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\to 0 or ‖gwt∗​(θt)‖→0\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert\\to 0.

In this case, the actual behavior of the algorithm depends on the specific choice of the step size. For example, if we take αt\=‖gwt∗​(θt)‖H⁡(c−1)​‖g0​(θt)‖\\alpha\_{t}=\\frac{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}{H(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}, then the result becomes

∑t\=0T‖gwt∗​(θt)‖2≤2​H​mini⁡(Li​(θ0)−Li​(θT+1)).\\sum\_{t=0}^{T}\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert^{2}\\leq 2H{\\min\_{i}(L\_{i}(\\theta\_{0})-L\_{i}(\\theta\_{T+1}))}.

which ensures ‖gwt∗​(θt)‖2→0\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert^{2}\\to 0.

###### Proof.

For any task i∈\[K\]i\\in\[K\], Li​(θt+1)−Li​(θ)\\displaystyle L\_{i}(\\theta\_{t+1})-L\_{i}(\\theta)

≤−αt​gi​(θt)⊤​d∗​(θt)+H​αt22​‖d∗​(θt)‖2\\displaystyle\\leq-\\alpha\_{t}g\_{i}(\\theta\_{t})^{\\top}d^{\*}(\\theta\_{t})+\\frac{H\\alpha\_{t}^{2}}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}

≤−αt​mini​gi​(θt)⊤​d∗​(θt)+H​αt22​‖d∗​(θt)‖2\\displaystyle\\leq-\\alpha\_{t}\\min\_{i}g\_{i}(\\theta\_{t})^{\\top}d^{\*}(\\theta\_{t})+\\frac{H\\alpha\_{t}^{2}}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}

≤−αt(gwt∗(θt)⊤g0(θt)+c‖g0(θt)‖‖gwt∗(θt)‖)+H​αt22‖d∗(θt)‖2//by ([12](#A1.E12 "In Lemma A.2. ‣ A.3 Conflit-Averse Gradient descent (CAGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"))\\displaystyle\\leq-\\alpha\_{t}\\left(g\_{w^{\*}\_{t}}(\\theta\_{t})^{\\top}g\_{0}(\\theta\_{t})+c\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert\\right)+\\frac{H\\alpha\_{t}^{2}}{2}\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}~~~~~~~\\text{{\\color\[rgb\]{1,0,1}//by \\eqref{equ:strongduality}}}

Meanwhile, note that

‖d∗​(θt)‖2\\displaystyle\\left\\lVert d^{\*}(\\theta\_{t})\\right\\rVert^{2}

\=‖g0​(θt)+c⁡‖g0​(θt)‖‖gwt∗​(θt)‖​gwt∗​(θt)‖2\\displaystyle=\\left\\lVert g\_{0}(\\theta\_{t})+\\frac{c\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}{g\_{w^{\*}\_{t}}(\\theta\_{t})}\\right\\rVert^{2}

\=(c2+1)​‖g0​(θt)‖2+2​c⁡‖g0​(θt)‖‖gwt∗​(θt)‖​g0​(θt)⊤​gwt∗​(θt)\\displaystyle=(c^{2}+1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}+2\\frac{c\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}g\_{0}(\\theta\_{t})^{\\top}{g\_{w^{\*}\_{t}}(\\theta\_{t})}

\=2​c​‖g0​(θt)‖‖gwt∗​(θt)‖​(gwt∗​(θt)⊤​g0​(θt)+c⁡‖g0​(θt)‖​‖gwt∗​(θt)‖)+(1−c2)​‖g0​(θt)‖2.\\displaystyle=2c\\frac{\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}\\left(g\_{w^{\*}\_{t}}(\\theta\_{t})^{\\top}g\_{0}(\\theta\_{t})+c\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert\\right)+(1-c^{2})\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}.

Therefore, Li​(θt+1)−Li​(θ)\\displaystyle L\_{i}(\\theta\_{t+1})-L\_{i}(\\theta)

≤−αt​(1−H​αt​c​‖g0​(θt)‖‖gwt∗​(θt)‖)​(gwt∗​(θt)⊤​g0​(θt)+c⁡‖g0​(θt)‖​‖gwt∗​(θt)‖)+H​αt22​(c2−1)​‖g0​(θt)‖2\\displaystyle\\leq-\\alpha\_{t}\\left(1-{H\\alpha\_{t}}c\\frac{\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}\\right)\\left(g\_{w^{\*}\_{t}}(\\theta\_{t})^{\\top}g\_{0}(\\theta\_{t})+c\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert\\right)+\\frac{H\\alpha\_{t}^{2}}{2}(c^{2}-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}

≤(∗)−αt​(1−H​αt​c​‖g0​(θt)‖‖gwt∗​(θt)‖)​(c−1)​‖g0​(θt)‖​‖gwt∗​(θt)‖−H​αt22​(c2−1)​‖g0​(θt)‖2\\displaystyle\\overset{(\*)}{\\leq}-\\alpha\_{t}\\left(1-{H\\alpha\_{t}}c\\frac{\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert}{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}\\right)(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert-\\frac{H\\alpha\_{t}^{2}}{2}(c^{2}-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}

\=−αt​(c−1)​‖g0​(θt)‖​‖gwt∗​(θt)‖+H​αt22​(c−1)2​‖g0​(θt)‖2\\displaystyle=-\\alpha\_{t}(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert+\\frac{H\\alpha\_{t}^{2}}{2}(c-1)^{2}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert^{2}

≤−12αt(c−1)‖g0(θt)‖‖gwt∗(θt)‖//assume αt≤‖gwt∗​(θt)‖H⁡(c−1)​‖g0​(θt)‖, c≥1\\displaystyle\\leq-\\frac{1}{2}\\alpha\_{t}(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert~~~~~~~\\text{{\\color\[rgb\]{1,0,1}//assume $\\alpha\_{{\\color\[rgb\]{1,0,0}t}}\\leq\\frac{\\left\\lVert g\_{w^{\*}\_{t}}(\\theta\_{t})\\right\\rVert}{H(c-1)\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert},$ $c\\geq 1$}}

where inequality (\*) uses Cauchy-Schwarz inequality. Therefore, a telescoping sum gives

∑t\=0Tαt​‖g0​(θt)‖​‖gwt∗​(θt)‖≤2​mini⁡(Li​(θ0)−Li​(θT+1))(c−1),\\sum\_{t=0}^{T}\\alpha\_{t}\\left\\lVert g\_{0}(\\theta\_{t})\\right\\rVert\\left\\lVert g\_{w\_{t}^{\*}}(\\theta\_{t})\\right\\rVert\\leq 2\\frac{\\min\_{i}(L\_{i}(\\theta\_{0})-L\_{i}(\\theta\_{T+1}))}{(c-1)}, when c≥1c\\geq 1.

∎

## Appendix B Experiment Details

### B.1 Multi-Fashion+MNIST

#### Experiment Details   

We follow the experiment setup from \[[23](#bib.bib23 "")\] and use the same shrinked LeNet that consists of the following layers as the shared base network: Conv(1,5,9,1), MaxPool2d(2), ReLU, BatchNorm2d(5), Conv2d(5,10,5,1), MaxPool2d(2), ReLU, BatchNorm1d(250), Linear(250, 50). Then a task-specific linear head Linear(50, 10) is attached to the shared base for the MNIST and FashionMNIST prediction. We use Adam \[[16](#bib.bib16 "")\] optimizer with a 0.001 learning rate and 0.01 weight decay, and then train for 50 epochs with a batch size of 256. The training set consists of 120000 images of size 36x36 and the test set consists of 20000 images of the same size.

### B.2 Multi-task Supervised Learning

#### Experiment Details   

For the multi-task supervised learning experiments on the NYU-v2 and CityScapes datasets, we follow exactly the same setup from MTAN \[[21](#bib.bib21 "")\]. We describe the details in the following. We adopt the SegNet \[[1](#bib.bib1 "")\] architecture as the backbone network and apply the attention mechanism from MTAN \[[21](#bib.bib21 "")\] on top of it. For the CityScapes dataset, we use the 7-class semantics labels. We train MTAN, Cross-Stitch, PCGrad and CAGrad with 200 epochs with a batch size of 2 for NYU-v2 and a batch size of 8 for CityScapes, using the Adam \[[16](#bib.bib16 "")\] optimizer with a learning rate of 0.0001. We further decay the learning rate to 0.00005 at the 100th epoch. As Liu et al. do not separately create a validation set, they average the test performance of each method in the last 10 epochs. We follow this and also average the test performance over the last 10 epochs, but additionally run over 3 seeds and calculate the mean and the standard error. We train CAGrad with c∈{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}c\\in\\{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9\\} and pick the best cc using their corresponding averaged training performance (c\=0.4c=0.4 for NYU-v2 and c\=0.4c=0.4 for CityScapes).

We also provide the final test losses and the per-epoch training times of each method in Fig. [5](#A2.F5 "Figure 5 ‣ Experiment Details ‣ B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

![Refer to caption](2110.14048v2/nyu.png)

Figure 5: Test loss and training time comparison on NYU-v2 and Cityscapes.

#### More Ablation Studies on NYU-v2 and CityScapes Datasets

We conduct the following additional studies on NYU-v2 and CityScapes datasets: 1) How do different methods perform when we additional apply the uncertain weight method \[[15](#bib.bib15 "")\]? 2) How do CAGrad perform with different values of cc? 3) How does PCGrad perform when we enlarge the learning rate? Specifically we double the learning rate to 2e-4. Results are provided in Tab. [5](#A2.T5 "Table 5 ‣ More Ablation Studies on NYU-v2 and CityScapes Datasets ‣ B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning") and Tab. [6](#A2.T6 "Table 6 ‣ More Ablation Studies on NYU-v2 and CityScapes Datasets ‣ B.2 Multi-task Supervised Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). We can see that CAGrad perform consistently with different values of 0<c<10\<c<1. PCGrad with larger learning rate will not perform better. Under the uncertain weights, MTAN and PCGrad indeed perform better but CAGrad is still comparable or better than them.

Segmentation

Depth

Surface Normal

#P.

Method

(Higher Better)

(Lower Better)

 Angle Distance (Lower Better) 

 Within t∘t^{\\circ} (Higher Better) 

Δ​m%↓\\Delta m\\%\\downarrow

mIoU

Pix Acc

Abs Err

Rel Err

Mean

Median

11.25

22.5

30

3

Independent

38.30

63.76

0.6754

0.2780

25.01

19.21

30.14

57.20

69.15

≈\\approx3

Cross-Stitch \[[25](#bib.bib25 "")\]

37.42

63.51

0.5487

0.2188

28.85

24.52

22.75

46.58

59.56

6.96

1.771.77

MTAN \[[21](#bib.bib21 "")\]

39.29

65.33

0.5493

0.2263

28.15

23.96

22.09

47.50

61.08

5.59

1.771.77

MGDA \[[30](#bib.bib30 "")\]

30.47

59.90

0.6070

0.2555

24.88

19.45

29.18

56.88

69.36

1.38

1.771.77

PCGrad \[[41](#bib.bib41 "")\] (lr=1e-4)

38.06

64.64

0.5550

0.2325

27.41

22.80

23.86

49.83

63.14

3.97

1.771.77

PCGrad \[[41](#bib.bib41 "")\] (lr=2e-4)

37.70

63.40

0.5871

0.2482

28.18

24.09

21.94

47.20

60.87

8.12

1.771.77

GradDrop \[[4](#bib.bib4 "")\]

39.39

65.12

0.5455

0.2279

27.48

22.96

23.38

49.44

62.87

3.58

1.771.77

CAGrad (cc\=0.2)

39.15

65.45

0.5563

0.2295

26.74

21.93

25.17

51.55

64.70

1.55

1.771.77

CAGrad (cc\=0.4)

39.79

65.49

0.5486

0.2250

26.31

21.58

25.61

52.36

65.58

0.20

1.771.77

CAGrad (cc\=0.6)

39.54

65.60

0.5340

0.2199

25.87

20.94

25.88

53.78

67.00

-1.36

1.771.77

CAGrad (cc\=0.8)

39.18

64.97

0.5379

0.2229

25.42

20.47

27.37

54.73

67.73

-2.29

1.771.77

MTAN \[[21](#bib.bib21 "")\] (Uncert. Weights)

38.74

64.70

0.5360

0.2243

26.52

21.71

25.50

52.02

65.14

0.75

1.771.77

PCGrad \[[41](#bib.bib41 "")\] (Uncert. Weights)

37.81

64.35

0.5318

0.2242

26.53

21.73

25.45

51.98

65.16

1.04

1.771.77

CAGrad (cc\=0.2) (Uncert. Weights)

38.87

65.19

0.5357

0.2227

26.38

21.64

25.66

52.21

65.39

0.319

1.771.77

CAGrad (cc\=0.4) (Uncert. Weights)

38.89

64.98

0.5313

0.2242

25.71

20.72

26.89

54.14

67.13

-1.59

1.771.77

CAGrad (cc\=0.6) (Uncert. Weights)

39.80

65.32

0.5334

0.2242

25.69

20.91

26.89

54.14

67.13

-1.59

1.771.77

CAGrad (cc\=0.8) (Uncert. Weights)

39.20

65.15

0.5322

0.2202

25.28

20.17

27.83

55.41

68.25

-3.14

Table 5: Multi-task learning results on NYU-v2 dataset. #P denotes the relative model size compared to the vanilla SegNet. Each experiment is repeated over 3 random seeds and the mean is reported.

Segmentation

Depth

#P.

Method

(Higher Better)

(Lower Better)

Δ​m%↓\\Delta m\\%\\downarrow

mIoU

Pix Acc

Abs Err

Rel Err

2

Independent

74.01

93.16

0.0125

27.77

≈\\approx3

Cross-Stitch \[[25](#bib.bib25 "")\]

73.08

92.79

0.0165

118.5

90.02

1.771.77

MTAN \[[21](#bib.bib21 "")\]

75.18

93.49

0.0155

46.77

22.60

1.771.77

MGDA \[[30](#bib.bib30 "")\]

68.84

91.54

0.0309

33.50

44.14

1.771.77

PCGrad \[[41](#bib.bib41 "")\]

75.13

93.48

0.0154

42.07

18.29

1.771.77

GradDrop \[[4](#bib.bib4 "")\]

75.27

93.53

0.0157

47.54

23.73

1.771.77

CAGrad (cc\=0.2)

75.18

93.49

0.0140

40.12

13.69

1.771.77

CAGrad (cc\=0.4)

75.16

93.48

0.0141

37.60

11.64

1.771.77

CAGrad (cc\=0.6)

74.31

93.39

0.0151

34.84

11.46

1.771.77

CAGrad (cc\=0.8)

74.95

93.50

0.0143

36.05

10.74

1.771.77

MTAN \[[21](#bib.bib21 "")\] (Uncert. Weights)

75.02

93.36

0.0139

35.56

9.48

1.771.77

PCGrad \[[41](#bib.bib41 "")\] (Uncert. Weights)

74.68

93.36

0.0135

34.00

7.26

1.771.77

CAGrad (cc\=0.2) (Uncert. Weights)

75.05

93.45

0.0140

34.33

8.40

1.771.77

CAGrad (cc\=0.4) (Uncert. Weights)

74.90

93.46

0.0141

34.84

9.13

1.771.77

CAGrad (cc\=0.6) (Uncert. Weights)

74.89

93.45

0.0136

35.17

8.48

1.771.77

CAGrad (cc\=0.8) (Uncert. Weights)

75.38

93.48

0.0141

35.54

9.63

Table 6: Multi-task learning results on CityScapes Challenge. Each experiment is repeated over 3 random seeds and the mean is reported.

### B.3 Multi-task Reinforcement Learning

#### Experiment Details   

The multi-task reinforcement learning experiments follow the exact setup from CARE \[[33](#bib.bib33 "")\]. Specifically, it is built on top of the MTRL codebase \[[32](#bib.bib32 "")\]. We consider the MT10 and MT50 benchmarks from the MetaWorld environment \[[42](#bib.bib42 "")\]. A visualization of the 50 tasks from MT50 is provided in Fig. [6](#A2.F6 "Figure 6 ‣ Experiment Details ‣ B.3 Multi-task Reinforcement Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning").

![Refer to caption](2110.14048v2/mt50.png)

Figure 6: The 50 tasks in MT50 benchmark \[[42](#bib.bib42 "")\].

The MT10 benchmark consists of a subset of 10 tasks from the MT50 task pool. For all methods, we use Soft Actor Critic (SAC) \[[10](#bib.bib10 "")\] as the underlying reinforcement learning algorithm. All methods are trained over 2 million steps with a batch size of 1280. Following CARE \[[32](#bib.bib32 "")\], we evaluate each method once every 10000 steps, and report the highest average test performance of a method over 10 random seeds over the entire training stage. For CAGrad-Fast, we sub-sample 4 and 8 tasks randomly at each optimization step as the SS (See Eq. ([4](#S3.E4 "In 3.3 Practical Speedup ‣ 3 Method ‣ Conflict-Averse Gradient Descent for Multi-task Learning"))) for the MT10 and MT50 experiments. For CAGrad, since MT10 and MT50 have 10 and 50 tasks, much more than the number of tasks in supervised MTL, so instead of using standard optimization library to solve the CAGrad objective, we apply 20 gradient descent steps to approximately solve the objective. The gradient descent is performed with a learning rate of 25 for MT10 and 50 for MT50, with a momentum of 0.5. We search the best cc from {0.1,0.5,0.9}\\{0.1,0.5,0.9\\} for MT10 and MT50 (c\=0.9c=0.9 for MT10 and c\=0.5c=0.5 for MT50). The computation efficiency is compared in Tab. [7](#A2.T7 "Table 7 ‣ Experiment Details ‣ B.3 Multi-task Reinforcement Learning ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning").  

Method

MT10 Time (sec)

MT50 Time (sec)

PCGrad

9.7

59.8

CAGrad

10.3

27.8

CAGrad-Fast

4.8

11.4

Table 7: The training time per update step for PCGrad, CAGrad and CAGrad-Fast on MT10/50.

In principle, PCGrad should have the same time complexity as CAGrad. However, in practice, PCGrad projects the gradients following a random ordering of the tasks in a sequential fashion (See Alg. [2](#alg2 "Algorithm 2 ‣ A.2 Projecting Conflicting Gradients (PCGrad) ‣ Appendix A Algorithm Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning")), so it requires a for loop over that task ordering, which makes it slow for a large number of tasks. Combined with the results from Tab. [3](#S5.T3 "Table 3 ‣ 5.3 Multi-task Reinforcement Learning ‣ 5 Experiment ‣ Conflict-Averse Gradient Descent for Multi-task Learning"), we see that CAGrad-Fast achieves comparable or better results than PCGrad with a roughly 2x and 5x speedup on MT10 and MT50.

### B.4 Semi-Supervised Learning with Auxiliary Tasks

#### Experiment Details   

We provide the hyperparameters for reproducing the experiments in our main text. All the methods are applied upon the original ARML baseline, with the same configuration in \[[31](#bib.bib31 "")\]. Specifically, the batch size is 256256 and the optimizer is Adam. The learning rate is initialized to 0.0050.005 in the first 160,000160,000 iterations and decay to 0.0010.001 in the rest iterations. The backbone networks is a WRN-28-2 model. To stablize the training process, the features are extracted by a moving-averaged model like in  \[[36](#bib.bib36 "")\] with a moving-average factor of 0.950.95. For PCGrad and MGDA, we use their official implementation without any change. For CAGrad (our method), we fix c\=0.1c=0.1 in all the experiments. The labeled images are randomly selected from the whole training set, and we repeat the experiments for 3 times on the same set of labeled images. We report the test accuracy of the model with the highest validation accuracy.

#### Training Losses   

We analyze the training losses of different methods to demonstrate the difference between these optimization methods. We report the losses, LC​EL\_{CE}, La​u​x1L\_{aux}^{1} and La​u​x2L\_{aux}^{2}, of the last epoch, when the number of labeled images is 2,0002,000. The losses are listed in Tab. [8](#A2.T8 "Table 8 ‣ Training Losses ‣ B.4 Semi-Supervised Learning with Auxiliary Tasks ‣ Appendix B Experiment Details ‣ Conflict-Averse Gradient Descent for Multi-task Learning"). We have two key observations: (1) MGDA totally ignores the main task LC​EL\_{CE}, yet it has the smallest loss on the second auxiliary task La​u​x2L\_{aux}^{2}. This implies MGDA finds a sub-optimal solution on the Pareto front. (2) PCGrad and CAGrad can both decrease the averaged loss L0L\_{0} compared with the baseline ARML, however, CAGrad yields a smaller L0L\_{0} than PCGrad.

Method

LC​EL\_{CE}

La​u​x1L\_{aux}^{1}

La​u​x2L\_{aux}^{2}

L0L\_{0}

ARML \[[31](#bib.bib31 "")\]

0.0 ±\\pm0.0

0.0574 ±\\pm0.0036

-0.4946 ±\\pm0.0010

-0.4372 ±\\pm0.0046

ARML + PCGrad \[[41](#bib.bib41 "")\]

0.0 ±\\pm0.0

0.0494 ±\\pm0.0088

-0.4943 ±\\pm0.0007

-0.4449 ±\\pm0.0095

ARML + MGDA \[[30](#bib.bib30 "")\]

0.407 ±\\pm0.018

0.0453 ±\\pm0.0049

-0.4980 ±\\pm0.0007

-0.0463 ±\\pm0.0233

ARML + CAGrad (Ours)

0.0 ±\\pm0.0

0.0419 ±\\pm0.0034

-0.4926 ±\\pm0.0023

-0.4507 ±\\pm0.0058

Table 8: The Training Losses in the Last Epoch when the number of the labeled images is 2,0002,000. Values that are smaller than 10−610^{-6} are replaced by 00. We report the averaged losses over 3 independent runs for each method, and mark the smallest losses in bold.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")