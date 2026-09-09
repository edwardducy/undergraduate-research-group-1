# FAMO: Fast Adaptive Multitask Optimization

 †Bo Liu    ‡Yihao Feng    Peter Stone    †Qiang Liu    †The University of Texas at Austin    ‡Salesforce AI Research    Sony AI{bliu, pstone, lqiang}@cs.utexas.edu,  yihaof@salesforce.com 

###### Abstract

One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, in practice, applying gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients (𝒪⁡(k)\\mathcal{O}(k) space and time where kk is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (FAMO), a dynamic weighting method that decreases task losses in a balanced way using 𝒪⁡(1)\\mathcal{O}(1) space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that FAMO achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency. Code is available at [https://github.com/Cranial-XIX/FAMO](https://github.com/Cranial-XIX/FAMO "").

## 1 Introduction

Large models trained on diverse data have advanced both computer vision \[[20](#bib.bib20 "")\] and natural language processing \[[4](#bib.bib4 "")\], paving the way for generalist agents capable of multitask learning (MTL) \[[5](#bib.bib5 "")\]. Given the substantial size of these models, it is crucial to design MTL methods that are *effective* in terms of task performance and *efficient* in terms of space and time complexities for managing training costs and environmental impacts. This work explores such methods through the lens of optimization.

Perhaps the most intuitive way of solving an MTL problem is to optimize the average loss across all tasks. However, in practice, doing so can lead to models with poor multitask performance: a subset of tasks are *severely under-optimized*. A major reason behind such optimization failure is that a subset of tasks are under-optimized because the average gradient constantly results in small (or even negative) progress on these tasks (see details in Section [2](#S2.SS0.SSS0.Px2 "Optimization Challenge ‣ 2 Background ‣ FAMO: Fast Adaptive Multitask Optimization")).

To mitigate this problem, gradient manipulation methods \[[43](#bib.bib43 ""), [25](#bib.bib25 ""), [7](#bib.bib7 ""), [24](#bib.bib24 "")\] compute a new update vector in place of the gradient to the average loss, such that all task losses decrease in a more balanced way. The new update vector is often determined by solving an additional optimization problem that involves all task gradients. While these approaches exhibit improved performance, they become computationally expensive when the number of tasks and the model size are large \[[41](#bib.bib41 "")\]. This is because they require computing and storing all task gradients at each iteration, thus demanding 𝒪⁡(k)\\mathcal{O}(k) space and time complexities, not to mention the overhead introduced by solving the additional optimization problem. In contrast, the average gradient can be efficiently computed in 𝒪⁡(1)\\mathcal{O}(1) space and time per iteration because one can first average the task losses and then take the gradient of the average loss.11 1 Here, we refer to the situation where a single data xx can be used to compute all task losses. To this end, we ask the following question:

(Q)(Q)  *Is it possible to design a multi-task learning optimizer that ensures a balanced reduction in losses across all tasks while utilizing 𝒪⁡(1)\\mathcal{O}(1) space and time per iteration?*

![Refer to caption](2306.03792v3/figures/famo.png)

Figure 1: Top left: The loss landscape, and individual task losses of a toy 2-task learning problem (\\filledstar\\filledstar represents the minimum of task losses). Top right: the runtime of different MTL methods for 50000 steps. Bottom: the loss trajectories of different MTL methods. Adam fails in 1 out of 5 runs to reach the Pareto front due to CG. FAMO decreases task losses in a balanced way and is the only method matching the 𝒪⁡(1)\\mathcal{O}(1) space/time complexity of Adam. Experimental details and analysis are provided in Section [5.1](#S5.SS1 "5.1 A Toy 2-Task Example ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"). 

In this work, we present Fast Adaptive Multitask Optimization (FAMO), a simple yet effective adaptive task weighting method to address the above question. On the one hand, FAMO is designed to ensure that all tasks are optimized with approximately similar progress. On the other hand, FAMO leverages the loss history to update the task weighting, hence bypassing the necessity of computing all task gradients. To summarize, our contributions are:

1.  1.

```
We introduce FAMO, an MTL optimizer that decreases task losses approximately at *equal rates* while using only 𝒪⁡(1)\\mathcal{O}(1) space and time per iteration.
```
2.  2.

```
We demonstrate that FAMO performs comparably to or better than existing gradient manipulation methods on a wide range of standard MTL benchmarks, in terms of standard MTL metrics, while being significantly computationally cheaper.
```
## 2 Background

In this section, we provide the formal definition of multitask learning, then discuss its optimization challenge, and provide a brief overview of the gradient manipulation methods.

#### Multitask Learning (MTL)

MTL considers optimizing a *single* model with parameter θ∈ℝm\\theta\\in\\mathbb{R}^{m} that can perform k≥2k\\geq 2 tasks well, where each task is associated with a loss function ℓi​(θ):ℝm→ℝ≥0\\ell\_{i}(\\theta):\\mathbb{R}^{m}\\rightarrow\\mathbb{R}\_{\\geq 0}.22 2 In this work, we assume ∀i,ℓi​(θ)≥0\\forall~i,~\\ell\_{i}(\\theta)\\geq 0, which is true for typical loss functions including mean square and cross-entropy losses. Note that one can always transform ℓi\\ell\_{i} to be non-negative if a loss lower bound is known. Then, it is common to optimize the average loss across all tasks:

minθ∈ℝm{ℓ0(θ):=1k∑i\=1kℓi(θ)}.\\min\_{\\theta\\in\\mathbb{R}^{m}}\\left\\{\\ell\_{0}(\\theta)\\vcentcolon=\\frac{1}{k}\\sum\_{i=1}^{k}\\ell\_{i}(\\theta)\\right\\}.

(1)

#### Optimization Challenge

Directly optimizing ([1](#S2.E1 "In Multitask Learning (MTL) ‣ 2 Background ‣ FAMO: Fast Adaptive Multitask Optimization")) can result in severe under-optimization of a subset of tasks. A major reason behind this optimization challenge is the “generalized" conflicting gradient phenomenon, which we explain in the following. At any time step tt, assume one updates the model parameter using a gradient descent style iterative update: θt+1\=θt−α​dt\\theta\_{t+1}=\\theta\_{t}-\\alpha d\_{t} where α\\alpha is the step size and dtd\_{t} is the update at time tt. Then, we say that conflicting gradients (CG) \[[24](#bib.bib24 ""), [43](#bib.bib43 "")\] happens if

∃i,ℓi(θt+1)−ℓi(θt)≈−α∇ℓi(θt)⊤dt\>0.\\exists i,~~\\ell\_{i}(\\theta\_{t+1})-\\ell\_{i}(\\theta\_{t})\\approx-\\alpha\\nabla\\ell\_{i}(\\theta\_{t})^{\\top}d\_{t}>0.

In other words, certain task’s loss is increasing. CG often occurs during optimization and is not inherently detrimental. However, it becomes undesirable when a subset of tasks persistently undergoes under-optimization due to CG. In a more general sense, it is not desirable if a subset of tasks has much slower learning progress compared to the rest of the tasks (even if all task losses are decreasing). This very phenomenon, which we call the “generalized" conflicting gradient, has spurred previous research to mitigate it at each optimization stage \[[43](#bib.bib43 "")\].

#### Gradient Manipulation Methods

Gradient manipulation methods aim to decrease all task losses in a more balanced way by finding a new update dtd\_{t} at each step. dtd\_{t} is usually a convex combination of task gradients, and therefore the name gradient manipulation (denote ∇ℓi,t\=∇θℓi​(θt)\\nabla\\ell\_{i,t}=\\nabla\_{\\theta}\\ell\_{i}(\\theta\_{t}) for short):

dt\=\[∇ℓ1,t⊤∇ℓk,t⊤\]⊤​wt,where​wt\=\[w1,twk,t\]\=f⁡(∇ℓ1,t,…,∇ℓk,t)∈𝕊k.d\_{t}=\\begin{bmatrix}\\nabla\\ell\_{1,t}^{\\top}\\\\ \\vdots\\\\ \\nabla\\ell\_{k,t}^{\\top}\\end{bmatrix}^{\\top}w\_{t},~~~\\text{where}~~w\_{t}=\\begin{bmatrix}w\_{1,t}\\\\ \\vdots\\\\ w\_{k,t}\\end{bmatrix}=f\\big(\\nabla\\ell\_{1,t},\\dots,\\nabla\\ell\_{k,t}\\big)\\in\\mathbb{S}\_{k}.

(2)

Here, 𝕊k\={w∈ℝ≥0k∣w⊤​𝟏\=1}\\mathbb{S}\_{k}=\\{w\\in\\mathbb{R}^{k}\_{\\geq 0}\\mid w^{\\top}\\bm{1}=1\\} is the probabilistic simplex, and 𝒘t\\bm{w}\_{t} is the task weighting across all tasks. Please refer to Appendix [A](#A1 "Appendix A Gradient Manipulation Methods ‣ FAMO: Fast Adaptive Multitask Optimization") for details of five state-of-the-art gradient manipulation methods (MGDA, PCGrad, CAGrad, IMTL-G, NashMTL) and their corresponding ff. Note that existing gradient manipulation methods require computing and storing kk task gradients before applying ff to compute dtd\_{t}, which often involves solving an additional optimization problem. As a result, we say these methods require at least 𝒪⁡(k)\\mathcal{O}(k) space and time complexity, which makes them slow and memory inefficient when kk and model size mm are large.

## 3 Fast Adaptive Multitask Optimization (FAMO)

In this section, we introduce FAMO that addresses question QQ, which involves two main ideas:

1.  1.

```
At each step, decrease all task losses at *an equal rate* as much as possible (Section [3.1](#S3.SS1 "3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")).
```
2.  2.

```
Amortize the computation in 1. over time (Section [3.2](#S3.SS2 "3.2 Fast Approximation by Amortizing over Time ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")).
```
Algorithm 1 Fast Adaptive Multitask Optimization (FAMO)

1:  Input: Initial parameter θ0\\theta\_{0}, task losses {ℓi}i\=1k\\{\\ell\_{i}\\}\_{i=1}^{k} (ensure that ℓi≥ϵ\>0\\ell\_{i}\\geq\\epsilon>0, for instance, by ℓi←ℓi−ℓi∗+ϵ\\ell\_{i}\\leftarrow\\ell\_{i}-\\ell\_{i}^{\*}+\\epsilon, ℓi∗\=infθℓi​(θ)\\ell\_{i}^{\*}=\\inf\_{\\theta}\\ell\_{i}(\\theta)), learning rate α\\alpha and β\\beta, and decay γ\\gamma (\=0.001\=0.001 by default). 

2:  ξ1←0{\\xi}\_{1}\\leftarrow{0}.         // initialize the task logits to all zeros 

3:  for t\=1:Tt=1:T do 

4:   Compute zt\=Softmax​(ξt){z}\_{t}=\\textbf{Softmax}({\\xi}\_{t}), e.g., zi,t\=exp⁡(ξi,t)∑i\=1kexp⁡(ξi,t).z\_{i,t}=\\frac{\\exp(\\xi\_{i,t})}{\\sum\_{i=1}^{k}\\exp(\\xi\_{i,t})}.

5:   Update the model parameters:

θt+1\=θt−α∑i\=1k(ctzi,tℓi,t)∇ℓi,t,wherect\=(∑i\=1kzi,tℓi,t)−1.\\theta\_{t+1}=\\theta\_{t}-\\alpha\\sum\_{i=1}^{k}\\big(c\_{t}\\frac{z\_{i,t}}{\\ell\_{i,t}}\\big)\\nabla\\ell\_{i,t},~~\\text{where}~~c\_{t}=\\big(\\sum\_{i=1}^{k}\\frac{z\_{i,t}}{\\ell\_{i,t}}\\big)^{-1}.

6:   Update the logits for task weighting:

ξt+1\=ξt−β⁡(δt+γ​ξt)​where​δt\=\[∇⊤z1,t​(ξt)∇⊤zk,t​(ξt)\]⊤​\[log⁡ℓ1,t−log⁡ℓ1,t+1log⁡ℓk,t−log⁡ℓk,t+1.\].{\\xi}\_{t+1}={\\xi}\_{t}-\\beta\\big({\\delta}\_{t}+\\gamma{\\xi}\_{t}\\big)~~\\text{where}~~{\\delta}\_{t}=\\begin{bmatrix}\\nabla^{\\top}z\_{1,t}({\\xi}\_{t})\\\\ \\vdots\\\\ \\nabla^{\\top}z\_{k,t}({\\xi}\_{t})\\\\ \\end{bmatrix}^{\\top}\\begin{bmatrix}\\log\\ell\_{1,t}-\\log\\ell\_{1,t+1}\\\\ \\vdots\\\\ \\log\\ell\_{k,t}-\\log\\ell\_{k,t+1}.\\end{bmatrix}.

7:  end for 

### 3.1 Balanced Rate of Loss Improvement

At time tt, assume we perform the update θt+1\=θt−α​dt\\theta\_{t+1}=\\theta\_{t}-\\alpha d\_{t}, we define the rate of improvement for task ii as

ri​(α,dt)\=ℓi,t−ℓi,t+1ℓi,t.r\_{i}(\\alpha,d\_{t})=\\frac{\\ell\_{i,t}-\\ell\_{i,t+1}}{\\ell\_{i,t}}.

(3)

FAMO then seeks an update dtd\_{t} that results in the largest *worst-case improvement rate* across all tasks (12​‖dt‖\\frac{1}{2}\\norm{d\_t} is subtracted to prevent an under-specified optimization problem where the objective can be infinitely large):

maxdt∈ℝm⁡mini∈\[k\]​1α​ri​(α,dt)−12​‖dt‖2.\\max\_{d\_{t}\\in\\mathbb{R}^{m}}\\min\_{i\\in\[k\]}\\frac{1}{\\alpha}r\_{i}(\\alpha,d\_{t})-\\frac{1}{2}\\norm{d\_t}^{2}.

(4)

When the step size α\\alpha is small, using Taylor approximation, the problem ([4](#S3.E4 "In 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) can be approximated by

maxdt∈ℝm⁡mini∈\[K\]​∇ℓi,t⊤dtℓi,t−12​‖dt‖2\=(∇log⁡ℓi,t)⊤​dt−12​‖dt‖2.\\max\_{d\_{t}\\in\\mathbb{R}^{m}}\\min\_{i\\in\[K\]}\\frac{\\nabla\\ell\_{i,t}^{\\top}d\_{t}}{\\ell\_{i,t}}-\\frac{1}{2}\\norm{d\_t}^{2}=\\big(\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d\_{t}-\\frac{1}{2}\\norm{d\_t}^{2}.

(5)

Instead of solving the primal problem in ([5](#S3.E5 "In 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) where d∈ℝmd\\in\\mathbb{R}^{m} (mm can be millions if θ\\theta is the parameter of a neural network), we consider its dual problem:

###### Proposition 3.1.

The dual objective of ([5](#S3.E5 "In 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) is

zt∗∈arg​minz∈𝕊k⁡12​‖Jt​z‖2,where​Jt\=\[∇log⁡ℓ1,t⊤∇log⁡ℓk,t⊤\],z\_{t}^{\*}\\in\\argmin\_{z\\in\\mathbb{S}\_{k}}\\frac{1}{2}\\norm{ J\_t z}^{2},~~~\\text{where}~~~J\_{t}=\\begin{bmatrix}\\nabla\\log\\ell\_{1,t}^{\\top}\\\\ \\vdots\\\\ \\nabla\\log\\ell\_{k,t}^{\\top}\\end{bmatrix},

(6)

where zt∗\=\[zt,i∗\]z\_{t}^{\*}=\[z\_{t,i}^{\*}\] is the optimal combination weights of the gradients, and the optimal update direction is dt∗\=Jt​zt∗d\_{t}^{\*}=J\_{t}z\_{t}^{\*}.

###### Proof.

maxd∈ℝm⁡mini∈\[k\]​(∇log⁡ℓi,t)⊤​d−12​‖d‖2\\displaystyle\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[k\]}\\big(\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d-\\frac{1}{2}\\norm{d}^{2}

\=\\displaystyle=

maxd∈ℝmminz∈𝕊k(∑i\=1kzi∇logℓi,t)⊤d−12‖d‖2\\displaystyle\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{{z}\\in\\mathbb{S}\_{k}}\\big(\\sum\_{i=1}^{k}z\_{i}\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d-\\frac{1}{2}\\norm{d}^{2}~~~~~~~

\=\\displaystyle=

minz∈𝕊kmaxd∈ℝm(∑i\=1kzi∇logℓi,t)⊤d−12‖d‖2(strong duality)\\displaystyle\\min\_{{z}\\in\\mathbb{S}\_{k}}\\max\_{d\\in\\mathbb{R}^{m}}\\big(\\sum\_{i=1}^{k}z\_{i}\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d-\\frac{1}{2}\\norm{d}^{2}\\qquad{\\color\[rgb\]{1,0,1}\\text{(strong duality)}}

Write g(d,z)\=(∑i\=1kzi∇logℓi,t)⊤d−12‖d‖2g(d,{z})=\\big(\\sum\_{i=1}^{k}z\_{i}\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d-\\frac{1}{2}\\norm{d}^{2}, then by setting

∂g∂d\=0⟹d∗\=∑i\=1kzi∇logℓi,t.\\partialderivative{g}{d}={0}\\quad\\Longrightarrow\\quad d^{\*}=\\sum\_{i=1}^{k}z\_{i}\\nabla\\log\\ell\_{i,t}.

Plugging in d∗d^{\*} back, we have

maxd∈ℝm⁡mini∈\[k\]​(∇log⁡ℓi,t)⊤​d−12​‖d‖2\=minz∈𝕊k⁡12​‖∑i\=1kzi∇logℓi,t‖2\=minz∈𝕊k⁡12​‖Jt​z‖2.\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[k\]}\\big(\\nabla\\log\\ell\_{i,t}\\big)^{\\top}d-\\frac{1}{2}\\norm{d}^{2}=\\min\_{z\\in\\mathbb{S}\_{k}}\\frac{1}{2}\\norm{\\sum\_{i=1}^k z\_i \\nabla\\log\\ell\_{i,t}}^{2}=\\min\_{z\\in\\mathbb{S}\_{k}}\\frac{1}{2}\\norm{J\_t z}^{2}.

At the optimum, we have dt∗\=Jt​zt∗d\_{t}^{\*}=J\_{t}z\_{t}^{\*}. ∎

The dual problem in ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) can be viewed as optimizing the log objective of the multiple gradient descent algorithm (MGDA) \[[9](#bib.bib9 ""), [35](#bib.bib35 "")\]. Similar to MGDA, ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) only involves a decision variable of dimension k≪mk\\ll m. Furthermore, if the optimal combination weights zt∗z^{\*}\_{t} is an interior point of 𝕊k\\mathbb{S}\_{k}, then the improvement rates ri​(α,dt∗)r\_{i}(\\alpha,d\_{t}^{\*}) of the different tasks ii equal, as we show in the following result.

###### Proposition 3.2.

Assume {ℓi}i\=1k\\{\\ell\_{i}\\}\_{i=1}^{k} are smooth and the optimal weights zt∗z^{\*}\_{t} in ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) is an interior point of 𝕊k\\mathbb{S}\_{k}, then

∀i≠j∈\[k\],ri∗​(dt∗)\=rj∗​(dt∗),\\forall~i\\neq j\\in\[k\],\\qquad r\_{i}^{\*}(d\_{t}^{\*})=r\_{j}^{\*}(d\_{t}^{\*}), where ri∗​(dt∗)\=limα→01α​ri​(α,dt∗).r\_{i}^{\*}(d\_{t}^{\*})=\\lim\_{\\alpha\\rightarrow 0}\\frac{1}{\\alpha}r\_{i}(\\alpha,d\_{t}^{\*}).

###### Proof.

Consider the Lagrangian form of ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization"))

ℒ⁡(z,λ,μ)\=12​‖∑i\=1kzi∇logℓi,t‖2+λ⁡(∑i\=1kzi−1)−∑i\=1kμi​zi,where​∀i,μi≥0.\\mathcal{L}({z},\\lambda,{\\mu})=\\frac{1}{2}\\norm{\\sum\_{i=1}^k z\_i \\nabla\\log\\ell\_{i,t}}^{2}+\\lambda\\big(\\sum\_{i=1}^{k}z\_{i}-1\\big)-\\sum\_{i=1}^{k}\\mu\_{i}z\_{i},~~\\text{where}~\\forall i,\\mu\_{i}\\geq 0.

(7)

When z∗{z}^{\*} reaches the optimum, we have ∂ℒ⁡(z,λ,μ)/∂z\=0\\partial\\mathcal{L}({z},\\lambda,{\\mu})/\\partial{z}={0}, recall that dt∗\=Jt​zt∗d\_{t}^{\*}=J\_{t}z\_{t}^{\*}, then

Jt⊤Jtz∗\=−μ−λ,whereJt\=\[∇log⁡ℓ1,t⊤∇log⁡ℓk,t⊤\]⟹Jt⊤dt∗\=−(μ+λ).\\displaystyle{J}\_{t}^{\\top}{J}\_{t}{z}^{\*}=-{\\mu}-\\lambda,~~~\\text{where}~~~{J}\_{t}=\\begin{bmatrix}\\nabla\\log\\ell\_{1,t}^{\\top}\\\\ \\vdots\\\\ \\nabla\\log\\ell\_{k,t}^{\\top}\\end{bmatrix}\\quad\\Longrightarrow\\quad{J}\_{t}^{\\top}d\_{t}^{\*}=-({\\mu}+\\lambda).

When zt∗{z}\_{t}^{\*} is an interior point of 𝕊k\\mathbb{S}\_{k}, we know that μ\=0{\\mu}={0}. Hence Jt⊤​dt∗\=−λ{J}\_{t}^{\\top}d\_{t}^{\*}=-\\lambda. This means, ∀i≠j,limα→01α​ri​(α,dt∗)\=∇log⁡ℓi,t⊤​dt∗\=∇log⁡ℓj,t⊤​dt∗\=limα→01α​rj​(α,dt∗).\\forall i\\neq j,\\qquad\\lim\_{\\alpha\\rightarrow 0}\\frac{1}{\\alpha}r\_{i}(\\alpha,d\_{t}^{\*})=\\nabla\\log\\ell\_{i,t}^{\\top}d\_{t}^{\*}=\\nabla\\log\\ell\_{j,t}^{\\top}d\_{t}^{\*}=\\lim\_{\\alpha\\rightarrow 0}\\frac{1}{\\alpha}r\_{j}(\\alpha,d\_{t}^{\*}).

∎

### 3.2 Fast Approximation by Amortizing over Time

Instead of fully solving ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) at each optimization step, FAMO performs a single-step gradient descent on z{z}, which amortizes the computation over the optimization trajectory:

zt+1\=zt−αz​δ\~,where​δ\~\=∇z12​‖∑i\=1kzi,t∇logℓi,t‖2\=Jt⊤​Jt​zt.{z}\_{t+1}={z}\_{t}-\\alpha\_{z}\\tilde{\\delta},~~~~\\text{where}~~\\tilde{\\delta}=\\nabla\_{{z}}\\frac{1}{2}\\norm{\\sum\_{i=1}^k z\_{i,t} \\nabla\\log\\ell\_{i,t}}^{2}={J}\_{t}^{\\top}{J}\_{t}{z}\_{t}.

(8)

But then, note that

1α​\[log⁡ℓ1,t−log⁡ℓ1,t+1log⁡ℓk,t−log⁡ℓk,t+1\]≈Jt⊤​dt\=Jt⊤​Jt​zt,\\frac{1}{\\alpha}\\begin{bmatrix}\\log\\ell\_{1,t}-\\log\\ell\_{1,t+1}\\\\ \\vdots\\\\ \\log\\ell\_{k,t}-\\log\\ell\_{k,t+1}\\end{bmatrix}\\approx{J}\_{t}^{\\top}d\_{t}={J}\_{t}^{\\top}{J}\_{t}{z}\_{t},

(9)

so we can use the change in log losses to approximate the gradient.

In practice, to ensure that z{z} always stays in 𝕊k\\mathbb{S}\_{k}, we re-parameterize z{z} by ξ{\\xi} and let zt\=Softmax​(ξt){z}\_{t}=\\textbf{Softmax}({\\xi}\_{t}), where ξt∈ℝK{\\xi}\_{t}\\in\\mathbb{R}^{K} are the unconstrained softmax logits. Consequently, we have the following approximate update on ξ{\\xi} from ([8](#S3.E8 "In 3.2 Fast Approximation by Amortizing over Time ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")):

ξt+1\=ξt−β​δ,where​δ\=\[∇⊤z1,t​(ξ)∇⊤zk,t​(ξ)\]⊤​\[log⁡ℓ1,t−log⁡ℓ1,t+1log⁡ℓk,t−log⁡ℓk,t+1\].{\\xi}\_{t+1}={\\xi}\_{t}-\\beta\\delta,~~~~\\text{where}~~\\delta=\\begin{bmatrix}\\nabla^{\\top}z\_{1,t}({\\xi})\\\\ \\vdots\\\\ \\nabla^{\\top}z\_{k,t}({\\xi})\\\\ \\end{bmatrix}^{\\top}\\begin{bmatrix}\\log\\ell\_{1,t}-\\log\\ell\_{1,t+1}\\\\ \\vdots\\\\ \\log\\ell\_{k,t}-\\log\\ell\_{k,t+1}\\end{bmatrix}.

(10)

Remark: While it is possible to perform gradient descent on z{z} for other gradient manipulation methods in principle, we will demonstrate in Appendix [B](#A2 "Appendix B Amortizing other Gradient Manipulation Methods ‣ FAMO: Fast Adaptive Multitask Optimization") that not all such updates can be easily approximated using the change in losses.

### 3.3 Practical Implementation

To facilitate practical implementation, we present two modifications to the update in ([10](#S3.E10 "In 3.2 Fast Approximation by Amortizing over Time ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")).

#### Re-normalization

The suggested update above is a convex combination of the gradients of the log loss, e.g., d∗\=∑i\=1kzi,t∇logℓi,t\=∑i\=1k(zi,tℓi,t)∇ℓi,t.d^{\*}=\\sum\_{i=1}^{k}z\_{i,t}\\nabla\\log\\ell\_{i,t}=\\sum\_{i=1}^{k}\\big(\\frac{z\_{i,t}}{\\ell\_{i,t}}\\big)\\nabla\\ell\_{i,t}.

When ℓi,t\\ell\_{i,t} is small, the multiplicative coefficient zi,tℓi,t\\frac{z\_{i,t}}{\\ell\_{i,t}} can be quite large and result in unstable optimization. Therefore, we propose to multiply d∗d^{\*} by a constant ctc\_{t}, such that ct​d∗c\_{t}d^{\*} can be written as a convex combination of the task gradients just as in other gradient manipulation algorithms (see ([2](#S2.E2 "In Gradient Manipulation Methods ‣ 2 Background ‣ FAMO: Fast Adaptive Multitask Optimization")) and we provide the corresponding definition of w{w} in the following):

ct\=(∑i\=1kzi,tℓi,t)−1anddt\=ctd∗\=∑i\=1kwi∇ℓi,t,wherewi\=ctzi,tℓi,t.c\_{t}=\\big(\\sum\_{i=1}^{k}\\frac{z\_{i,t}}{\\ell\_{i,t}}\\big)^{-1}~~~~\\text{and}~~~~d\_{t}=c\_{t}d^{\*}=\\sum\_{i=1}^{k}w\_{i}\\nabla\\ell\_{i,t},~~~\\text{where}~~w\_{i}=c\_{t}\\frac{z\_{i,t}}{\\ell\_{i,t}}.

(11)

#### Regularization

As we are amortizing the computation over time and the loss objective {ℓi​(⋅)}\\{\\ell\_{i}(\\cdot)\\}s are changing dynamically, it makes sense to focus more on the recent updates of ξ{\\xi} \[[46](#bib.bib46 "")\]. To this end, we put a decay term on w{w} such that the resulting ξt{\\xi}\_{t} is an exponential moving average of its gradient updates:

ξt+1\=ξt−β⁡(δt+γ​ξt)\=−β⁡(δt+(1−β​γ)​δt−1+(1−β​γ)2​δt−2+…).{\\xi}\_{t+1}={\\xi}\_{t}-\\beta(\\delta\_{t}+\\gamma{\\xi}\_{t})=-\\beta\\big(\\delta\_{t}+(1-\\beta\\gamma)\\delta\_{t-1}+(1-\\beta\\gamma)^{2}\\delta\_{t-2}+\\dots\\big).

(12)

We provide the complete FAMO algorithm in Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization") and its pseudocode in Appendix [C](#A3 "Appendix C FAMO Pseudocode in PyTorch ‣ FAMO: Fast Adaptive Multitask Optimization").

### 3.4 The Continuous Limit of FAMO

One way to characterize FAMO’s behavior is to understand the stationary points of the continuous-time limit of FAMO (i.e. when step sizes (α,β)(\\alpha,\\beta) shrink to zero). From Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization"), one can derive the following non-autonomous dynamical system (assuming {ℓi}\\{\\ell\_{i}\\} are all smooth):

\[θ˙ξ˙\]\=−ct​\[Jt​ztAt​Jt⊤​Jt​zt+γct​ξt\],where​At\=\[∇⊤z1,t​(ξt)∇⊤zk,t​(ξt)\].\\begin{bmatrix}~\\dot{\\theta}~\\\\ ~\\dot{{\\xi}}~\\end{bmatrix}=-c\_{t}\\begin{bmatrix}{J}\_{t}{z}\_{t}\\\\ {A}\_{t}{J}\_{t}^{\\top}{J}\_{t}{z}\_{t}+\\frac{\\gamma}{c\_{t}}{\\xi}\_{t}\\end{bmatrix},~~\\text{where}~{A}\_{t}=\\begin{bmatrix}\\nabla^{\\top}z\_{1,t}({\\xi}\_{t})\\\\ \\vdots\\\\ \\nabla^{\\top}z\_{k,t}({\\xi}\_{t})\\\\ \\end{bmatrix}.

(13)

([13](#S3.E13 "In 3.4 The Continuous Limit of FAMO ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) reaches its stationary points (or fixed points) when (note that ct\>0c\_{t}>0)

\[θ˙ξ˙\]\=0⟹Jt​zt\=0​and​ξt\=0⟹∑i\=1k∇log⁡ℓi,t\=0.\\begin{bmatrix}~\\dot{\\theta}~\\\\ ~\\dot{{\\xi}}~\\end{bmatrix}=0~~\\Longrightarrow~~{J}\_{t}{z}\_{t}=0~~\\text{and}~~{\\xi}\_{t}=0~~\\Longrightarrow~~\\sum\_{i=1}^{k}\\nabla\\log\\ell\_{i,t}=0.

(14)

Therefore, the minimum points of ∑i\=1klog⁡ℓi​(θ)\\sum\_{i=1}^{k}\\log\\ell\_{i}(\\theta) are all stationary points of ([13](#S3.E13 "In 3.4 The Continuous Limit of FAMO ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")).

## 4 Related Work

In this section, we summarize existing methods that tackle learning challenges in multitask learning (MTL). The general idea of most existing works is to encourage positive knowledge transfer by sharing parameters while decreasing any potential negative knowledge transfer (a.k.a, interference) during learning. There are three major ways of doing so: task grouping, designing network architectures specifically for MTL, and designing multitask optimization methods.

#### Task Grouping

Task grouping refers to grouping KK tasks into N<KN\<K clusters and learning NN models for each cluster. The key is estimating the amount of positive knowledge transfer incurred by grouping certain tasks together and then identifying which tasks should be grouped \[[39](#bib.bib39 ""), [45](#bib.bib45 ""), [38](#bib.bib38 ""), [36](#bib.bib36 ""), [11](#bib.bib11 "")\].

#### Multitask Architecture

Novel neural architectures for MTL include *hard-parameter-sharing* methods, which decompose a neural network into task-specific modules and a shared feature extractor using manually designed heuristics \[[21](#bib.bib21 ""), [29](#bib.bib29 ""), [2](#bib.bib2 "")\], and *soft-parameter-sharing* methods, which learn which parameters to share \[[30](#bib.bib30 ""), [34](#bib.bib34 ""), [12](#bib.bib12 ""), [27](#bib.bib27 "")\]. Recent studies extend neural architecture search for MTL by learning where to branch a network to have task-specific modules \[[14](#bib.bib14 ""), [3](#bib.bib3 "")\].

#### Multitask Optimization

The most relevant approach to our method is MTL optimization via task balancing. These methods dynamically re-weight all task losses to mitigate the conflicting gradient issue \[[40](#bib.bib40 ""), [43](#bib.bib43 "")\]. The simplest form of gradient manipulation is to re-weight the task losses based on manually designed criteria \[[6](#bib.bib6 ""), [13](#bib.bib13 ""), [18](#bib.bib18 "")\], but these methods are often heuristic and lack theoretical support. Gradient manipulation methods \[[35](#bib.bib35 ""), [43](#bib.bib43 ""), [25](#bib.bib25 ""), [7](#bib.bib7 ""), [16](#bib.bib16 ""), [24](#bib.bib24 ""), [32](#bib.bib32 ""), [26](#bib.bib26 ""), [47](#bib.bib47 "")\] propose to form a new update vector at each optimization by linearly combining task gradients. The local improvements across all tasks using the new update can often be explicitly analyzed, making these methods better understood in terms of convergence. However, it has been observed that gradient manipulation methods are often slow in practice, which may outweigh their performance benefits \[[22](#bib.bib22 "")\]. By contrast, FAMO is designed to match the performance of these methods while remaining efficient in terms of memory and computation. Another recent work proposes to sample random task weights at each optimization step for MTL \[[23](#bib.bib23 "")\], which is also computationally efficient. However, we will demonstrate empirically that FAMO performs better than this method.

## 5 Empirical Results

We conduct experiments to answer the following question:

*How does FAMO perform in terms of space/time complexities and standard MTL metrics against prior MTL optimizers on standard benchmarks (e.g., supervised and reinforcement MTL problems)?*

In the following, we first use a toy 2-task problem to demonstrate how FAMO mitigates CG while being efficient. Then we show that FAMO performs comparably or even better than state-of-the-art gradient manipulation methods on standard multitask supervised and reinforcement learning benchmarks. In addition, FAMO requires significantly lower computation time when KK is large compared to other methods. Lastly, we conduct an ablation study on how robust FAMO is to γ\\gamma. Each subsection first details the experimental setup and then analyzes the results.

### 5.1 A Toy 2-Task Example

![Refer to caption](2306.03792v3/figures/toy_plot.png)

Figure 2: The average loss L0L^{0} and the two task losses L1L^{1} and L2L^{2} for the toy example.

To better understand the optimization trajectory of FAMO, we adopt the same 2D multitask optimization problem from NashMTL \[[32](#bib.bib32 "")\] to visualize how FAMO balances different loss objectives. The model parameter θ\=(θ1,θ2)∈ℝ2\\theta=(\\theta\_{1},\\theta\_{2})\\in\\mathbb{R}^{2}. The two tasks’ objectives and their surface plots are provided in Appendix [D](#A4 "Appendix D Toy Example ‣ FAMO: Fast Adaptive Multitask Optimization") and Figure [2](#S5.F2 "Figure 2 ‣ 5.1 A Toy 2-Task Example ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"). We compare FAMO against Adam \[[19](#bib.bib19 "")\], MGDA \[[35](#bib.bib35 "")\], PCGrad \[[43](#bib.bib43 "")\], CAGrad \[[24](#bib.bib24 "")\], and NashMTL \[[32](#bib.bib32 "")\]. We then pick 5 initial points θinit∈{(−8.5,7.5),(−8.5,5),(0,0),(9,9),(10,−8)}\\theta\_{\\text{init}}\\in\\{(-8.5,7.5),(-8.5,5),(0,0),(9,9),(10,-8)\\} and plot the corresponding optimization trajectories with different methods in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FAMO: Fast Adaptive Multitask Optimization"). Note that the toy example is constructed such that naively applying Adam on the average loss can cause the failure of optimization for task 1.

Findings: From Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FAMO: Fast Adaptive Multitask Optimization"), we observe that FAMO, like all other gradient manipulation methods, mitigates the CG and reaches the Pareto front for all five runs. In the meantime, FAMO performs similarly to NashMTL and achieves a balanced loss decrease even when the two task losses are improperly scaled. Finally, as shown in the top-right of the plot, FAMO behaves similarly to Adam in terms of the training time, which is 25×\\times faster than NashMTL.

### 5.2 MTL Performance

#### Multitask Supervised Learning.

We consider four supervised benchmarks commonly used in prior MTL research \[[24](#bib.bib24 ""), [27](#bib.bib27 ""), [32](#bib.bib32 ""), [33](#bib.bib33 "")\]: NYU-v2 \[[31](#bib.bib31 "")\] (3 tasks), CityScapes \[[8](#bib.bib8 "")\] (2 tasks), QM-9 \[[1](#bib.bib1 "")\] (11 tasks), and CelebA \[[28](#bib.bib28 "")\] (40 tasks). Specifically, NYU-v2 is an indoor scene dataset consisting of 1449 RGBD images and dense per-pixel labeling with 13 classes. The learning objectives include image segmentation, depth prediction, and surface normal prediction based on any scene image. CityScapes dataset is similar to NYU-v2 but contains 5000 street-view RGBD images with per-pixel annotations. QM-9 dataset is a widely used benchmark in graph neural network learning. It consists of \>\>130K molecules represented as graphs annotated with node and edge features. We follow the same experimental setting used in NashMTL \[[32](#bib.bib32 "")\], where the learning objective is to predict 11 properties of molecules. We use 110K molecules from the QM9 example in PyTorch Geometric \[[10](#bib.bib10 "")\], 10K molecules for validation, and the rest of 10K molecules for testing. The characteristic of this dataset is that the 11 properties are at different scales, posing a challenge for task balancing in MTL. Lastly, CelebA dataset contains 200K face images of 10K different celebrities, and each face image is provided with 40 facial binary attributes. Therefore, CelebA can be viewed as a 40-task MTL problem. Different from NYU-v2, CityScapes, and QM-9, the number of tasks (KK) in CelebA is much larger, hence posing a challenge to learning efficiency.

We compare FAMO against 11 MTL optimization methods and a single-task learning baseline: (1) Single task learning (STL), training an independent model (θ\\theta for each task; (2) Linear scalarization (LS) baseline that minimizes L0L^{0}; (3) Scale-invariant (SI) baseline that minimizes ∑klog⁡Lk​(θ)\\sum\_{k}\\log L^{k}(\\theta), as SI is invariant to any scalar multiplication of task losses; (4) Dynamic Weight Average (DWA) \[[27](#bib.bib27 "")\], a heuristic for adjusting task weights based on rates of loss changes; (5) Uncertainty Weighting (UW) \[[18](#bib.bib18 "")\] uses task uncertainty as a proxy to adjust task weights; (6) Random Loss Weighting (RLW) \[[23](#bib.bib23 "")\] that samples task weighting whose log-probabilities follow the normal distribution; (7) MGDA \[[35](#bib.bib35 "")\] that finds the equal descent direction for each task; (8) PCGrad \[[43](#bib.bib43 "")\] proposes to project each task gradient to the normal plan of that of other tasks and combining them together in the end; (9) CAGrad \[[24](#bib.bib24 "")\] optimizes the average loss while explicitly controls the minimum decrease across tasks; (10) IMTL-G \[[25](#bib.bib25 "")\] finds the update direction with equal projections on task gradients; (11) GradDrop \[[7](#bib.bib7 "")\] that randomly dropout certain dimensions of the task gradients based on how much they conflict; (12) NashMTL \[[32](#bib.bib32 "")\] formulates MTL as a bargaining game and finds the solution to the game that benefits all tasks. For FAMO, we choose the best hyperparameter γ∈{0.0001,0.001,0.01}\\gamma\\in\\{0.0001,0.001,0.01\\} based on the validation loss. Specifically, we choose γ\\gamma equals 0.010.01 for the CityScapes dataset and 0.0010.001 for the rest of the datasets. See Appendix [E](#A5 "Appendix E Experimental Results with Error Bars ‣ FAMO: Fast Adaptive Multitask Optimization") for results with error bars.

|                 |                    |                      |                      |                         |                                |                 |                               |       |       |       |
| --------------- | ------------------ | -------------------- | -------------------- | ----------------------- | ------------------------------ | --------------- | ----------------------------- | ----- | ----- | ----- |
| mIoU ↑\\uparrow | Pix Acc ↑\\uparrow | Abs Err ↓\\downarrow | Rel Err ↓\\downarrow | Angle Dist ↓\\downarrow | Within t∘t^{\\circ} ↑\\uparrow | MR ↓\\downarrow | Δ​m%\\Delta m\\% ↓\\downarrow |       |       |       |
| Mean            | Median             | 11.25                | 22.5                 | 30                      |                                |                 |                               |       |       |       |
| 38.30           | 63.76              | 0.6754               | 0.2780               | 25.01                   | 19.21                          | 30.14           | 57.20                         | 69.15 |       |       |
| 39.29           | 65.33              | 0.5493               | 0.2263               | 28.15                   | 23.96                          | 22.09           | 47.50                         | 61.08 | 8.89  | 5.59  |
| 38.45           | 64.27              | 0.5354               | 0.2201               | 27.60                   | 23.37                          | 22.53           | 48.57                         | 62.32 | 7.89  | 4.39  |
| 37.17           | 63.77              | 0.5759               | 0.2410               | 28.27                   | 24.18                          | 22.26           | 47.05                         | 60.62 | 11.22 | 7.78  |
| 39.11           | 65.31              | 0.5510               | 0.2285               | 27.61                   | 23.18                          | 24.17           | 50.18                         | 62.39 | 7.67  | 3.57  |
| 36.87           | 63.17              | 0.5446               | 0.2260               | 27.04                   | 22.61                          | 23.54           | 49.05                         | 63.65 | 7.44  | 4.05  |
| 30.47           | 59.90              | 0.6070               | 0.2555               | 24.88                   | 19.45                          | 29.18           | 56.88                         | 69.36 | 6.00  | 1.38  |
| 38.06           | 64.64              | 0.5550               | 0.2325               | 27.41                   | 22.80                          | 23.86           | 49.83                         | 63.14 | 8.00  | 3.97  |
| 39.39           | 65.12              | 0.5455               | 0.2279               | 27.48                   | 22.96                          | 23.38           | 49.44                         | 62.87 | 7.00  | 3.58  |
| 39.79           | 65.49              | 0.5486               | 0.2250               | 26.31                   | 21.58                          | 25.61           | 52.36                         | 65.58 | 4.56  | 0.20  |
| 39.35           | 65.60              | 0.5426               | 0.2256               | 26.02                   | 21.19                          | 26.20           | 53.13                         | 66.24 | 3.78  | -0.76 |
| 40.13           | 65.93              | 0.5261               | 0.2171               | 25.26                   | 20.08                          | 28.40           | 55.47                         | 68.15 | 2.11  | -4.04 |
| 38.88           | 64.90              | 0.5474               | 0.2194               | 25.06                   | 19.57                          | 29.21           | 56.61                         | 68.98 | 3.44  | -4.10 |

Table 1: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

| Method | μ\\mu | α\\alpha | ϵHOMO\\epsilon\_{\\text{HOMO}} | ϵLUMO\\epsilon\_{\\text{LUMO}} | ⟨R2⟩\\langle R^{2}\\rangle | ZPVE  | U0U\_{0} | UU    | HH    | GG   | cvc\_{v} | MR ↓\\downarrow | Δ​m%\\Delta m\\% ↓\\downarrow |
| ------ | ----- | -------- | ------------------------------ | ------------------------------ | -------------------------- | ----- | -------- | ----- | ----- | ---- | -------- | --------------- | ----------------------------- |
|        |       |          |                                |                                |                            |       |          |       |       |      |          |                 |                               |
| 0.11   | 0.33  | 73.6     | 89.7                           | 5.20                           | 14.06                      | 143.4 | 144.2    | 144.6 | 140.3 | 0.13 | 6.45     | 177.6           |                               |
| 0.31   | 0.35  | 149.8    | 135.7                          | 1.00                           | 4.51                       | 55.3  | 55.8     | 55.8  | 55.3  | 0.11 | 3.55     | 77.8            |                               |
| 0.11   | 0.34  | 76.9     | 92.8                           | 5.87                           | 15.47                      | 156.3 | 157.1    | 157.6 | 153.0 | 0.14 | 8.00     | 203.8           |                               |
| 0.11   | 0.33  | 74.1     | 90.6                           | 5.09                           | 13.99                      | 142.3 | 143.0    | 143.4 | 139.3 | 0.13 | 6.27     | 175.3           |                               |
| 0.39   | 0.43  | 166.2    | 155.8                          | 1.07                           | 4.99                       | 66.4  | 66.8     | 66.8  | 66.2  | 0.12 | 4.91     | 108.0           |                               |
| 0.22   | 0.37  | 126.8    | 104.6                          | 3.23                           | 5.69                       | 88.4  | 89.4     | 89.3  | 88.0  | 0.12 | 5.91     | 120.5           |                               |
| 0.11   | 0.29  | 75.9     | 88.3                           | 3.94                           | 9.15                       | 116.4 | 116.8    | 117.2 | 114.5 | 0.11 | 4.73     | 125.7           |                               |
| 0.12   | 0.32  | 83.5     | 94.8                           | 3.22                           | 6.93                       | 114.0 | 114.3    | 114.5 | 112.3 | 0.12 | 5.45     | 112.8           |                               |
| 0.14   | 0.29  | 98.3     | 93.9                           | 1.75                           | 5.70                       | 101.4 | 102.4    | 102.0 | 100.1 | 0.10 | 4.36     | 77.2            |                               |
| 0.10   | 0.25  | 82.9     | 81.9                           | 2.43                           | 5.38                       | 74.5  | 75.0     | 75.1  | 74.2  | 0.09 | 2.09     | 62.0            |                               |
| 0.15   | 0.30  | 94.0     | 95.2                           | 1.63                           | 4.95                       | 70.82 | 71.2     | 71.2  | 70.3  | 0.10 | 3.27     | 58.5            |                               |

Table 2: Results on QM-9 dataset (11 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

| Method          |                    |                      |                               |                 |                               |       |       |
| --------------- | ------------------ | -------------------- | ----------------------------- | --------------- | ----------------------------- | ----- | ----- |
| Segmentation    | Depth              | MR ↓\\downarrow      | Δ​m%\\Delta m\\% ↓\\downarrow | MR ↓\\downarrow | Δ​m%\\Delta m\\% ↓\\downarrow |       |       |
| mIoU ↑\\uparrow | Pix Acc ↑\\uparrow | Abs Err ↓\\downarrow | Rel Err ↓\\downarrow          |                 |                               |       |       |
| 74.01           | 93.16              | 0.0125               | 27.77                         |                 |                               |       |       |
| 70.95           | 91.73              | 0.0161               | 33.83                         | 6.50            | 14.11                         | 4.15  | 6.28  |
| 70.95           | 91.73              | 0.0161               | 33.83                         | 9.25            | 14.11                         | 7.20  | 7.83  |
| 74.57           | 93.41              | 0.0158               | 47.79                         | 9.25            | 24.38                         | 1.46  | 5.22  |
| 75.24           | 93.52              | 0.0160               | 44.37                         | 6.50            | 21.45                         | 3.20  | 6.95  |
| 72.02           | 92.85              | 0.0140               | 30.13                         | 6.00            | 5.89                          | 3.23  | 5.78  |
| 68.84           | 91.54              | 0.0309               | 33.50                         | 9.75            | 44.14                         | 14.85 | 10.93 |
| 75.13           | 93.48              | 0.0154               | 42.07                         | 6.75            | 18.29                         | 3.17  | 6.65  |
| 75.27           | 93.53              | 0.0157               | 47.54                         | 6.00            | 23.73                         | 3.29  | 7.80  |
| 75.16           | 93.48              | 0.0141               | 37.60                         | 5.75            | 11.64                         | 2.48  | 6.20  |
| 75.33           | 93.49              | 0.0135               | 38.41                         | 4.00            | 11.10                         | 0.84  | 4.67  |
| 75.41           | 93.66              | 0.0129               | 35.02                         | 2.00            | 6.82                          | 2.84  | 4.97  |
| 74.54           | 93.29              | 0.0145               | 32.59                         | 6.25            | 8.13                          | 1.21  | 4.72  |

Table 3: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

Evaluations: We consider two metrics \[[32](#bib.bib32 "")\] for MTL: 1) Δ​m%\\Delta m\\%, the average per-task performance drop of a method mm relative to the STL baseline denoted as bb: 𝚫​𝒎%\=1K​∑k\=1K(−1)δk​(Mm,k−Mb,k)/Mb,k×100,\\bm{\\Delta}\\bm{m}\\%=\\frac{1}{K}\\sum\_{k=1}^{K}(-1)^{\\delta\_{k}}(M\_{m,k}-M\_{b,k})/M\_{b,k}\\times 100, where Mb,kM\_{b,k} and Mm,kM\_{m,k} are the STL and mm’s value for metric MkM\_{k}. δk\=1\\delta\_{k}=1 (or 00) if the MkM\_{k} is higher (or lower) the better. 2) Mean Rank (MR): the average rank of each method across tasks. For instance, if a method ranks first for every task, MR will be 1.

Findings: Results on the four benchmark datasets are provided in Table [1](#S5.T1 "Table 1 ‣ Multitask Supervised Learning. ‣ 5.2 MTL Performance ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"), [2](#S5.T2 "Table 2 ‣ Multitask Supervised Learning. ‣ 5.2 MTL Performance ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization") and [3](#S5.T3 "Table 3 ‣ Multitask Supervised Learning. ‣ 5.2 MTL Performance ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"). We observe that FAMO performs consistently well across different supervised learning MTL benchmarks compared to other gradient manipulation methods. In particular, it achieves state-of-the-art results in terms of Δ​m%\\Delta m\\% on the NYU-v2 and QM-9 datasets.

#### Multitask Reinforcement Learning.

We further apply FAMO to multitask reinforcement learning (MTRL) problems as MTRL often suffers more from conflicting gradients due to the stochastic nature of reinforcement learning \[[43](#bib.bib43 "")\]. Following CAGrad \[[24](#bib.bib24 "")\], we apply FAMO on the MetaWorld \[[44](#bib.bib44 "")\] MT10 benchmark, which consists of 10 robot manipulation tasks with different reward functions. Following \[[37](#bib.bib37 "")\], we use Soft Actor-Critic (SAC) \[[15](#bib.bib15 "")\] as the underlying RL algorithm, and compare against baseline methods including LS (SAC with a shared model) \[[44](#bib.bib44 "")\], Soft Modularization \[[42](#bib.bib42 "")\] (an MTL network that routes different modules in a shared model to form different policies), PCGrad \[[43](#bib.bib43 "")\], CAGrad and NashMTL \[[32](#bib.bib32 "")\]. The experimental setting and hyperparameters all match exactly with those in CAGrad. For NashMTL, we report the results of applying the NashMTL update once per {1,50,100}\\{1,50,100\\} iterations.44 4 We could not reproduce the MTRL results of NashMTL exactly, so we report both the results from the original paper and our reproduced results. The results for all methods are provided in Table [5.2](#S5.SS2.SSS0.Px2 "Multitask Reinforcement Learning. ‣ 5.2 MTL Performance ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization").

![[Uncaptioned image]](2306.03792v3/figures/mt10.png) 

Figure 3: Training Success Rate and Time.

| Method              |
| ------------------- |
| (mean ±\\pm stderr) |
| 0.49 ±\\pm0.07      |
| 0.90 ±\\pm0.03      |
| 0.72 ±\\pm0.02      |
| 0.73 ±\\pm0.04      |
| 0.83 ±\\pm0.05      |
| 0.91 ±\\pm0.03      |
| 0.85 ±\\pm0.02      |
| 0.87 ±\\pm0.03      |
| 0.80 ±\\pm0.13      |
| 0.76 ±\\pm0.10      |
| 0.80 ±\\pm0.12      |
| 0.77 ±\\pm0.05      |
| 0.83 ±\\pm0.05      |

Table 4: MTRL results (averaged over 10 runs) on the Metaworld-10 benchmark.

Findings: From Table [5.2](#S5.SS2.SSS0.Px2 "Multitask Reinforcement Learning. ‣ 5.2 MTL Performance ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"), we observe that FAMO performs comparably to CAGrad and outperforms PCGrad and the average gradient descent baselines by a large margin. FAMO also outperforms NashMTL based on our implementation. Moreover, FAMO is significantly faster than NashMTL, even when it is applied once every 100 steps.

### 5.3 MTL Efficiency (Training Time Comparison)

Figure [4](#S5.F4 "Figure 4 ‣ 5.3 MTL Efficiency (Training Time Comparison) ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization") provides the FAMO’s average training time per epoch against that of the baseline methods.

![Refer to caption](2306.03792v3/figures/time.png)

Figure 4: Average training time per epoch for different MTL optimization methods. We report the relative training time of a method to that of the linear scalarization (LS) method (which uses the average gradient).

Findings: From the figure, we observe that FAMO introduces negligible overhead across all benchmark datasets compared to the LS method, which is, in theory, the lower bound for computation time. In contrast, methods like NashMTL have much longer training time compared to FAMO. More importantly, the computation cost of these methods scales with the number of tasks. In addition, note that these methods also take at least 𝒪⁡(K)\\mathcal{O}(K) space to store the task gradients, which is implausible for large models in the many-task setting (i.e., when m\=|θ|m=|\\theta| and KK are large).

### 5.4 Ablation on γ\\gamma

In this section, we provide the ablation study on the regularization coefficient γ\\gamma in Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Ablation on 𝛾 ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization").

![Refer to caption](2306.03792v3/figures/gamma.png)

Figure 5: Ablation over γ\\gamma: we plot the performance of FAMO (in terms of Δ​m%\\Delta m\\% using different values of γ\\gamma from {0.0001,0.001,0.01}\\{0.0001,0.001,0.01\\} on the four supervised MTL benchmarks.

Findings: From Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Ablation on 𝛾 ‣ 5 Empirical Results ‣ FAMO: Fast Adaptive Multitask Optimization"), we can observe that choosing the right regularization coefficient can be crucial. But except for CityScapes, FAMO performs reasonably well using all different γ\\gammas. The problem with CityScapes is that one of the task losses is close to 0 at the very beginning, hence small changes in task weighting can result in very different loss improvement. Therefore we conjecture that using a larger γ\\gamma, in this case, can help stabilize MTL.

## 6 Conclusion and Limitations

In this work, we introduce FAMO, a fast optimization method for multitask learning (MTL) that mitigates the conflicting gradients using 𝒪⁡(1)\\mathcal{O}(1) space and time. As multitasking large models gain more attention, we believe designing efficient but effective optimizers like FAMO for MTL is crucial. FAMO balances task losses by ensuring each task’s loss decreases approximately at an equal rate. Empirically, we observe that FAMO can achieve competitive performance against the state-of-the-art MTL gradient manipulation methods. One limitation of FAMO is its dependency on the regularization parameter γ\\gamma, which is introduced due to the stochastic update of the task weighting logits 𝒘\\bm{w}. Future work can investigate a more principled way of determining γ\\gamma.

## References

*   \[1\] L. C. Blum and J.-L. Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. J. Am. Chem. Soc., 131:8732, 2009.
*   \[2\] Felix JS Bragman, Ryutaro Tanno, Sebastien Ourselin, Daniel C Alexander, and Jorge Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1385–1394, 2019.
*   \[3\] David Bruggemann, Menelaos Kanakis, Stamatios Georgoulis, and Luc Van Gool. Automated search for resource-efficient branched multi-task networks. arXiv preprint arXiv:2008.10292, 2020.
*   \[4\] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.
*   \[5\] Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.
*   \[6\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pages 794–803. PMLR, 2018.
*   \[7\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. arXiv preprint arXiv:2010.06808, 2020.
*   \[8\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
*   \[9\] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012.
*   \[10\] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.
*   \[11\] Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34:27503–27516, 2021.
*   \[12\] Yuan Gao, Haoping Bai, Zequn Jie, Jiayi Ma, Kui Jia, and Wei Liu. Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11543–11552, 2020.
*   \[13\] Michelle Guo, Albert Haque, De-An Huang, Serena Yeung, and Li Fei-Fei. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pages 270–287, 2018.
*   \[14\] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. Learning to branch for multi-task learning. In International Conference on Machine Learning, pages 3854–3863. PMLR, 2020.
*   \[15\] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, pages 1861–1870. PMLR, 2018.
*   \[16\] Adrián Javaloy and Isabel Valera. Rotograd: Dynamic gradient homogenization for multi-task learning. arXiv preprint arXiv:2103.02631, 2021.
*   \[17\] Alexandr Katrutsa, Daniil Merkulov, Nurislam Tursynbek, and Ivan Oseledets. Follow the bisector: a simple method for multi-objective optimization. arXiv preprint arXiv:2007.06937, 2020.
*   \[18\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
*   \[19\] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
*   \[20\] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.
*   \[21\] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6129–6138, 2017.
*   \[22\] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and Pawan K Mudigonda. In defense of the unitary scalarization for deep multi-task learning. Advances in Neural Information Processing Systems, 35:12169–12183, 2022.
*   \[23\] Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. arXiv preprint arXiv:2111.10603, 2021.
*   \[24\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021.
*   \[25\] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2020.
*   \[26\] Shikun Liu, Stephen James, Andrew J Davison, and Edward Johns. Auto-lambda: Disentangling dynamic task relationships. arXiv preprint arXiv:2202.03091, 2022.
*   \[27\] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1871–1880, 2019.
*   \[28\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
*   \[29\] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. Learning multiple tasks with multilinear relationship networks. Advances in neural information processing systems, 30, 2017.
*   \[30\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3994–4003, 2016.
*   \[31\] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.
*   \[32\] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022.
*   \[33\] Lucas Pascal, Pietro Michiardi, Xavier Bost, Benoit Huet, and Maria A Zuluaga. Improved optimization strategies for deep multi-task networks. arXiv preprint arXiv:2109.11678, 2021.
*   \[34\] Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 4822–4829, 2019.
*   \[35\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. arXiv preprint arXiv:1810.04650, 2018.
*   \[36\] Jiayi Shen, Xiantong Zhen, Marcel Worring, and Ling Shao. Variational multi-task learning with gumbel-softmax priors. Advances in Neural Information Processing Systems, 34:21031–21042, 2021.
*   \[37\] Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-task reinforcement learning with context-based representations. arXiv preprint arXiv:2102.06177, 2021.
*   \[38\] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International Conference on Machine Learning, pages 9120–9132. PMLR, 2020.
*   \[39\] Sebastian Thrun and Joseph O’Sullivan. Discovering structure in multiple learning tasks: The tc algorithm. In ICML, volume 96, pages 489–497, 1996.
*   \[40\] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
*   \[41\] Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, and Orhan Firat. Do current multi-task optimization methods in deep learning even help? Advances in Neural Information Processing Systems, 35:13597–13609, 2022.
*   \[42\] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. arXiv preprint arXiv:2003.13661, 2020.
*   \[43\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. arXiv preprint arXiv:2001.06782, 2020.
*   \[44\] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pages 1094–1100. PMLR, 2020.
*   \[45\] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3712–3722, 2018.
*   \[46\] Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu. On the convergence of stochastic multi-objective gradient manipulation and beyond. Advances in Neural Information Processing Systems, 35:38103–38115, 2022.
*   \[47\] Shijie Zhu, Hui Zhao, Pengjie Wang, Hongbo Deng, Jian Xu, and Bo Zheng. Gradient deconfliction via orthogonal projections onto subspaces for multi-task learning.

## Appendix A Gradient Manipulation Methods

In this section, we provide a brief overview of representative gradient manipulation methods in multitask/multiobjective optimization. Specifically, we will also discuss the connections among these methods.

#### Multiple Gradient Descent Algorithm (MGDA) \[[9](#bib.bib9 ""), [35](#bib.bib35 "")\]

The MGDA algorithm is one of the earliest gradient manipulation methods for multitask learning. In MGDA, the per step update dtd\_{t} is found by solving

maxd∈ℝm⁡mini∈\[k\]⁡∇ℓi,t⊤​d−12​‖d‖2.\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[k\]}\\nabla\\ell\_{i,t}^{\\top}d-\\frac{1}{2}\\norm{d}^{2}.

As a result, the solution d∗d^{\*} of MGDA optimizes the “worst improvement" across all tasks or equivalently seeks an *equal* descent across all task losses as much as possible. But in practice, MGDA suffers from slow convergence since the update d∗d^{\*} can be very small. For instance, if one task has a very small loss scale, the progress of all other tasks will be bounded by the progress on this task. Note that the original objective in ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) is similar to the MGDA objective in the sense that we can view optimizing ([6](#S3.E6 "In Proposition 3.1. ‣ 3.1 Balanced Rate of Loss Improvement ‣ 3 Fast Adaptive Multitask Optimization (FAMO) ‣ FAMO: Fast Adaptive Multitask Optimization")) as optimizing the log of the task losses. Hence, when we compare FAMO against MGDA, one can regard FAMO as balancing the *rate* of loss improvement while MGDA balances the absolute improvement across task losses.

#### Projecting Gradient Descent (PCGrad) \[[43](#bib.bib43 "")\]

PCGrad initializes vPCi\=∇ℓi,tv^{i}\_{\\text{PC}}=\\nabla\\ell\_{i,t}, then for each task ii, PCGrad loops over all task j≠ij\\neq i (in a random order, which is crucial as mentioned in \[[43](#bib.bib43 "")\]) and removes the “conflict"

viPC←viPC−viPC⊤∇ℓj,t‖ℓj,t‖2∇ℓj,tifviPC⊤∇ℓj,t<0.v^{i}\_{\\text{PC}}\\leftarrow v^{i}\_{\\text{PC}}-\\frac{{v^{i}\_{\\text{PC}}}^{\\top}\\nabla\\ell\_{j,t}}{\\norm{\\ell\_{j,t}}^{2}}\\nabla\\ell\_{j,t}~~~~\\text{if}~~~~{v^{i}\_{\\text{PC}}}^{\\top}\\nabla\\ell\_{j,t}<0.

In the end, PCGrad produces dt\=1k​∑i\=1kvPCid\_{t}=\\frac{1}{k}\\sum\_{i=1}^{k}v^{i}\_{\\text{PC}}. Due to the construction, PCGrad will also help improve the “worst improvement" across all tasks since the “conflicts" have been removed. However, due to the stochastic iterative procedural of this algorithm, it is hard to understand PCGrad from a first principle approach.

#### Conflict-averse Gradient Descent (CAGrad) \[[24](#bib.bib24 "")\]

dtd\_{t} is found by solving

maxd∈ℝm⁡mini∈\[k\]⁡∇ℓi,t⊤​ds.t.‖d−∇ℓ0,t‖≤c​‖∇ℓ0,t‖.\\max\_{d\\in\\mathbb{R}^{m}}\\min\_{i\\in\[k\]}\\nabla\\ell\_{i,t}^{\\top}d~~~~\\text{s.t.}~~~~\\norm{d - \\nabla\\ell\_{0,t}}\\leq c\\norm{\\nabla\\ell\_{0,t}}.

Here, ℓ0,t\=1k​∑i\=1kℓi,t\\ell\_{0,t}=\\frac{1}{k}\\sum\_{i=1}^{k}\\ell\_{i,t}. CAGrad seeks an update dtd\_{t} that optimizes the “worst improvement" as much as possible, conditioned on that the update still decreases the average loss. By controlling the hyperparameter cc, CAGrad can recover MGDA (c→∞c\\rightarrow\\infty) and the vanilla averaged gradient descent (c→0c\\rightarrow 0). Due to the extra constraint, CAGrad provably converges to the stationary points of ℓ0\\ell\_{0} when 0≤c<10\\leq c<1.

#### Impartial Multi-Task Learning (IMTL-G) \[[25](#bib.bib25 "")\]

IMTL-G finds dtd\_{t} such that it shares the same cosine similarity with any task gradients:

∀i≠j,dt⊤∇ℓi,t‖∇ℓi,t‖\=dt⊤∇ℓj,t‖∇ℓj,t‖,anddt\=∑i\=1kwi,t∇ℓi,t,for somewt∈𝕊k.\\forall i\\neq j,~~~d\_{t}^{\\top}\\frac{\\nabla\\ell\_{i,t}}{\\norm{\\nabla\\ell\_{i,t}}}=d\_{t}^{\\top}\\frac{\\nabla\\ell\_{j,t}}{\\norm{\\nabla\\ell\_{j,t}}},~~~\\text{and}~~~d\_{t}=\\sum\_{i=1}^{k}w\_{i,t}\\nabla\\ell\_{i,t},~~\\text{for some}~~{w}\_{t}\\in\\mathbb{S}\_{k}.

The constraint that dt\=∑i\=1kwi,t∇ℓi,td\_{t}=\\sum\_{i=1}^{k}w\_{i,t}\\nabla\\ell\_{i,t} is for preventing the problem from being under-determined. From the above equation, we can see that IMTL-G ignores the “size" of each task gradient and only cares about the “direction". As a result, one can think of IMTL-G as a variant of MGDA that applies to the normalized gradients. By doing so, IMTL-G does not suffer from the straggler effect due to slow objectives. Furthermore, one can view IMTL-G as the equal angle descent, which is also proposed in Katrutsa et al. \[[17](#bib.bib17 "")\], where the objective is to find dd such that

∀i≠j,cos⁡(d,∇ℓi,t)\=cos⁡(d,∇ℓj,t).\\forall i\\neq j,\\qquad\\cos(d, \\nabla\\ell\_{i,t})=\\cos(d, \\nabla\\ell\_{j,t}).

#### NashMTL\[[32](#bib.bib32 "")\]

NashMTL finds dtd\_{t} by solving a bargaining game treating the local improvement of each task loss as the utility for each task:

max⁡∑i\=1kd∈ℝm,‖d‖≤1⁡log(∇ℓi,t⊤d).\\max\_{d\\in\\mathbb{R}^{m},\\norm{d}\\leq 1}\\sum\_{i=1}^{k}\\log\\big(\\nabla\\ell\_{i,t}^\\top d\\big).

Note that the objective of NashMTL implicitly assumes that there exists dd such that ∀i,∇ℓi,t⊤d\>0\\forall~i,~~\\nabla\\ell\_{i,t}^{\\top}d>0 (otherwise we reach the Pareto front). It is easy to see that

max⁡∑i\=1k‖d‖≤1⁡log(∇ℓi,t⊤d)\=max⁡∑i\=1k‖d‖≤1⁡log⁡⟨∇ℓi,t‖∇ℓi,t‖,d⟩\=max⁡∑i\=1k‖d‖≤1⁡log⁡cos⁡(∇ℓi,t,d).\\max\_{\\norm{d}\\leq 1}\\sum\_{i=1}^{k}\\log\\big(\\nabla\\ell\_{i,t}^\\top d\\big)=\\max\_{\\norm{d}\\leq 1}\\sum\_{i=1}^{k}\\log\\langle\\frac{\\nabla\\ell\_{i,t}}{\\norm{\\nabla\\ell\_{i,t}}},d\\rangle=\\max\_{\\norm{d}\\leq 1}\\sum\_{i=1}^{k}\\log\\cos\\big(\\nabla\\ell\_{i,t}, d\\big).

Therefore, due to the log\\log, NashMTL also ignores the “size" of task gradients and only cares about their “directions". Moreover, denote ui\=∇ℓi,t‖∇ℓi,t‖u\_{i}=\\frac{\\nabla\\ell\_{i,t}}{\\norm{\\nabla\\ell\_{i,t}}}. Then, according to the KKT condition, we know:

∑iuiui⊤​d−αd\=0,α≥0⟹d\=1α∑i1ui⊤​dui.\\sum\_{i}\\frac{u\_{i}}{u\_{i}^{\\top}d}-\\alpha d=0,\\quad\\alpha\\geq 0\\qquad\\Longrightarrow\\qquad d=\\frac{1}{\\alpha}\\sum\_{i}\\frac{1}{u\_{i}^{\\top}d}u\_{i}.

Consider when k\=2k=2, if we take the *equal angle descent* direction: d∠\=(u1+u2)/2d\_{\\angle}=(u\_{1}+u\_{2})/2 (note that as u1u\_{1} and u2u\_{2} are normalized, their bisector is just their average). Then it is easy to check that

d∠\=1α​(2u1⊤​(u1+u2)​u1+2u2⊤​(u1+u2)​u2),where​α\=u1⊤​(u1+u2)4\=u2⊤​(u1+u2)4.d\_{\\angle}=\\frac{1}{\\alpha}\\bigg(\\frac{2}{u\_{1}^{\\top}(u\_{1}+u\_{2})}u\_{1}+\\frac{2}{u\_{2}^{\\top}(u\_{1}+u\_{2})}u\_{2}\\bigg),~~\\text{where}~~\\alpha=\\frac{u\_{1}^{\\top}(u\_{1}+u\_{2})}{4}=\\frac{u\_{2}^{\\top}(u\_{1}+u\_{2})}{4}.

As a result, we can see that when k\=2k=2, NashMTL is equivalent to IMTL-G (or the equal angle descent). However, when k\>2k>2, this is not in general true.

Remark Note that all of these gradient manipulation methods require computing and storing KK task gradients before applying ff to compute dtd\_{t}, which often involves solving an additional optimization problem. Hence, these methods can be slow for large KK and large model sizes.

## Appendix B Amortizing other Gradient Manipulation Methods

Although FAMO uses iterative update on w{w}, it is not immediately clear whether we can apply the same amortization easily on other existing gradient manipulation methods. In this section, we discuss such possibilities and point out the challenges.

#### Amortizing MGDA

This is almost the same as in FAMO, except that MGDA acts on the original task losses while FAMO acts on the log of task losses.

#### Amortizing PCGrad

For PCGrad, finding the final update vector requires iteratively projecting one task gradient to the other, so there is no straightforward way of bypassing the computation of task gradients.

#### Amortizing IMTL-G

The task weighting in IMTL-G is computed by a series of matrix-matrix and matrix-vector products using task gradients \[[25](#bib.bib25 "")\]. Hence, it is also hard to amortize its computation over time.

Therefore, we focus on deriving the amortization for CAGrad and NashMTL.

#### Amortizing CAGrad

For CAGrad, the dual objective is

minw∈𝕊k⁡F⁡(w)\=gw⊤​g0+c​‖gw‖​‖g0‖,\\min\_{{w}\\in\\mathbb{S}\_{k}}F({w})=g\_{{w}}^{\\top}g\_{0}+c\\norm{g\_{{w}}}\\norm{g\_0},

(15)

where g0\=∇ℓ0,tg\_{0}=\\nabla\\ell\_{0,t} and gw\=∑i\=1kwi∇ℓig\_{{w}}=\\sum\_{i=1}^{k}w\_{i}\\nabla\\ell\_{i}. Denote

G\=\[∇ℓ1,t⊤∇ℓk,t⊤\].{G}=\\begin{bmatrix}\\nabla\\ell\_{1,t}^{\\top}\\\\ \\vdots\\\\ \\nabla\\ell\_{k,t}^{\\top}\\end{bmatrix}.

Now, if we take the gradient with respect to w{w} in ([15](#A2.E15 "In Amortizing CAGrad ‣ Appendix B Amortizing other Gradient Manipulation Methods ‣ FAMO: Fast Adaptive Multitask Optimization")), we have:

∂F∂w\=G⊤​g0+c​‖g0‖‖gw‖​G⊤​gw.\\partialderivative{F}{{w}}={G}^{\\top}g\_{0}+c\\frac{\\norm{g\_0}}{\\norm{g\_{{w}}}}{G}^{\\top}g\_{{w}}.

(16)

As a result, in order to approximate this gradient, one can separately estimate:

G⊤​g0≈ℓ⁡(θ)−ℓ⁡(θ−α​g0)αG⊤​gw≈ℓ⁡(θ)−ℓ⁡(θ−α​gw)α‖g0‖≈1⊤​G⊤​g0‖gw‖≈w⊤​G⊤​gw.\\begin{split}{G}^{\\top}g\_{0}&\\approx\\frac{{\\ell}(\\theta)-{\\ell}(\\theta-\\alpha g\_{0})}{\\alpha}\\\\ {G}^{\\top}g\_{{w}}&\\approx\\frac{{\\ell}(\\theta)-{\\ell}(\\theta-\\alpha g\_{{w}})}{\\alpha}\\\\ \\norm{g\_0}&\\approx\\sqrt{{1}^{\\top}{G}^{\\top}g\_{0}}\\\\ \\norm{g\_{{w}}}&\\approx\\sqrt{{w}^{\\top}{G}^{\\top}g\_{{w}}}\\\\ \\end{split}.

(17)

Once all these are estimated, one can combine them together to perform a single update on w{w}. But note that this will require 3 forward and backward passes through the model, making it harder to implement in practice.

#### Amortizing NashMTL

Per derivation from NashMTL \[[32](#bib.bib32 "")\], the objective is to solve for w{w}:

G⊤​G​w\=1⊘w.{G}^{\\top}{G}{w}={1}\\oslash{w}.

(18)

One can therefore form an objective:

minw⁡F⁡(w)\=‖G⊤​G​w−1⊘w‖22.\\min\_{{w}}F({w})=\\norm{{G}^\\top{G} {w} - {1} \\oslash{w} }^{2}\_{2}.

(19)

Taking the derivative of FF with respect to w{w}, we have

∂F∂w\=2​G⊤​G​(G⊤​gw−1⊘w)+2​(G⊤​gw−1⊘w)⊘(w⊙w).\\partialderivative{F}{{w}}=2{G}^{\\top}{G}\\bigg({G}^{\\top}g\_{{w}}-{1}\\oslash{w}\\bigg)+2\\bigg({G}^{\\top}g\_{{w}}-{1}\\oslash{w}\\bigg)\\oslash({w}\\odot{w}).

(20)

Therefore, to approximate the gradient of w{w}, one needs to first estimate

G⊤​gw≈L⁡(θ)−L⁡(θ−α​gw)α\=η.{G}^{\\top}g\_{{w}}\\approx\\frac{{L}(\\theta)-{L}(\\theta-\\alpha g\_{{w}})}{\\alpha}={\\eta}.

(21)

Then we estimate

G⊤​G​(η−1⊘w)≈L⁡(θ)−L⁡(θ−α​G​(η−1⊘w))α.{G}^{\\top}{G}({\\eta}-{1}\\oslash{w})\\approx\\frac{{L}(\\theta)-{L}(\\theta-\\alpha{G}({\\eta}-{1}\\oslash{w}))}{\\alpha}.

(22)

Again, this results in 3 forward and backward passes through the model, let alone the overhead of resetting the model back to θ\\theta (requires a copy of the original weights).

In short, though it is possible to derive fast approximation algorithm to approximate the gradient update on w{w} for some of the existing gradient manipulation methods, it often involves much more complicated computation compared to that of FAMO.

## Appendix C FAMO Pseudocode in PyTorch

We provide the pseudocode for FAMO in Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix C FAMO Pseudocode in PyTorch ‣ FAMO: Fast Adaptive Multitask Optimization"). To use FAMO, one just first compute the task losses, call get\_weighted\_loss to get the weighted loss, and do the normal backpropagation through the weighted loss. After that, one call update to update the task weighting.

class FAMO:  
def \_\_init\_\_(self, num\_tasks, min\_losses, α\\alpha\=0.025, γ\\gamma\=0.001):  
   \# min\_losses  (num\_tasks,) the loss lower bound for each task.      self.min\_losses = min\_losses  
    self.xi = torch.tensor(\[0.0\] \* num\_tasks, requires\_grad=True)  
    self.xi\_opt = torch.optim.Adam(\[self.xi\], lr=α\\alpha, weight\_decay=γ\\gamma)  
  
def get\_weighted\_loss(self, losses):  
   \# losses  (num\_tasks,)      z = F.softmax(self.xi, -1)  
    D = losses - self.min\_losses + 1e-8  
    c = 1 / (z / D).sum().detach()  
    loss = (c \* D.log() \* z).sum()  
    return loss  
  
def update(self, prev\_losses, curr\_losses):  
   \# prev\_losses  (num\_tasks,)  
   \# curr\_losses  (num\_tasks,)      delta = (prev\_losses - self.min\_losses + 1e-8).log() -  
            (curr\_losses - self.min\_losses + 1e-8).log()  
    with torch.enable\_grad():  
        d = torch.autograd.grad(F.softmax(self.xi, -1),  
                                self.xi,  
                                grad\_outputs=delta.detach())\[0\]  
    self.xi\_opt.zero\_grad()  
    self.xi.grad = d  
    self.xi\_opt.step  

Algorithm 2 Implementation of FAMO in PyTorch-like Pseudocode

## Appendix D Toy Example

We provide the task objectives for the toy example in the following. The model parameter θ\=(θ1,θ2)∈ℝ2\\theta=(\\theta\_{1},\\theta\_{2})\\in\\mathbb{R}^{2} and the task objectives are L1L^{1} and L2L^{2}:

L1​(θ)\\displaystyle L^{1}(\\theta)

\=0.1⋅(c1​(θ)​f1​(θ)+c2​(θ)​g1​(θ))​and​L2​(θ)\=c1​(θ)​f2​(θ)+c2​(θ)​g2​(θ),where\\displaystyle=0.1\\cdot(c\_{1}(\\theta)f\_{1}(\\theta)+c\_{2}(\\theta)g\_{1}(\\theta))~~\\text{and}~~L^{2}(\\theta)=c\_{1}(\\theta)f\_{2}(\\theta)+c\_{2}(\\theta)g\_{2}(\\theta),~\\text{where}

f1​(θ)\\displaystyle f\_{1}(\\theta)

\=log⁡((max⁡(|0.5​(−θ1−7)−tanh⁡((−θ2))|,0.000005)))+6,\\displaystyle=\\log{\\big(\\max(|0.5(-\\theta\_1-7)-\\tanh{(-\\theta\_2)}|,~~0.000005)\\big)}+6, f2​(θ)\\displaystyle f\_{2}(\\theta)

\=log⁡((max⁡(|0.5​(−θ1+3)−tanh⁡((−θ2))+2|,0.000005)))+6,\\displaystyle=\\log{\\big(\\max(|0.5(-\\theta\_1+3)-\\tanh{(-\\theta\_2)}+2|,~~0.000005)\\big)}+6, g1​(θ)\\displaystyle g\_{1}(\\theta)

\=((−θ1+7)2+0.1∗(−θ2−8)2)/10−20,\\displaystyle=\\big((-\\theta\_{1}+7)^{2}+0.1\*(-\\theta\_{2}-8)^{2}\\big)/10-20, g2​(θ)\\displaystyle g\_{2}(\\theta)

\=((−θ1−7)2+0.1∗(−θ2−8)2)/10−20,\\displaystyle=\\big((-\\theta\_{1}-7)^{2}+0.1\*(-\\theta\_{2}-8)^{2})\\big/10-20, c1​(θ)\\displaystyle c\_{1}(\\theta)

\=max⁡(tanh⁡((0.5∗θ2)),0)​and​c2​(θ)\=max⁡(tanh((−0.5∗θ2)),0).\\displaystyle=\\max(\\tanh{(0.5\*\\theta\_2)},~0)~~\\text{and}~~c\_{2}(\\theta)=\\max(\\tanh{(-0.5\*\\theta\_2)},~0).

## Appendix E Experimental Results with Error Bars

We followed the exact experimental setup from NashMTL \[[32](#bib.bib32 "")\]. Therefore, the numbers for baseline methods are taken from their original paper. In the following, we provide FAMO’s result with error bars.

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

Δ​m%\\Delta m\\% ↓\\downarrow

Mean

Median

11.25

22.5

30

FAMO (mean)

38.88

64.90

0.5474

0.2194

25.06

19.57

29.21

56.61

68.98

-4.10

FAMO (stderr)

±\\pm0.54

±\\pm0.21

±\\pm0.0016

±\\pm0.0026

±\\pm0.06

±\\pm0.09

±\\pm0.17

±\\pm0.19

±\\pm0.14

±\\pm0.39

Table 5: Results on NYU-v2 dataset (3 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

| Method        | μ\\mu       | α\\alpha    | ϵHOMO\\epsilon\_{\\text{HOMO}} | ϵLUMO\\epsilon\_{\\text{LUMO}} | ⟨R2⟩\\langle R^{2}\\rangle | ZPVE        | U0U\_{0}  | UU        | HH        | GG        | cvc\_{v}    | Δ​m%\\Delta m\\% ↓\\downarrow |
| ------------- | ----------- | ----------- | ------------------------------ | ------------------------------ | -------------------------- | ----------- | --------- | --------- | --------- | --------- | ----------- | ----------------------------- |
| FAMO (mean)   | 0.15        | 0.30        | 94.0                           | 95.2                           | 1.63                       | 4.95        | 70.82     | 71.2      | 71.2      | 70.3      | 0.10        | 58.5                          |
| FAMO (stderr) | ±\\pm0.0046 | ±\\pm0.0070 | ±\\pm3.074                     | ±\\pm2.413                     | ±\\pm0.0211                | ±\\pm0.0871 | ±\\pm2.17 | ±\\pm2.19 | ±\\pm2.19 | ±\\pm2.21 | ±\\pm0.0026 | ±\\pm3.26                     |

Table 6: Results on QM-9 dataset (11 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

| Method          |                    |                               |                               |           |           |
| --------------- | ------------------ | ----------------------------- | ----------------------------- | --------- | --------- |
| Segmentation    | Depth              | Δ​m%\\Delta m\\% ↓\\downarrow | Δ​m%\\Delta m\\% ↓\\downarrow |           |           |
| mIoU ↑\\uparrow | Pix Acc ↑\\uparrow | Abs Err ↓\\downarrow          | Rel Err ↓\\downarrow          |           |           |
| 74.54           | 93.29              | 0.0145                        | 32.59                         | 8.13      | 1.21      |
| ±\\pm0.11       | ±\\pm0.04          | ±\\pm0.0009                   | ±\\pm1.06                     | ±\\pm1.98 | ±\\pm0.24 |

Table 7: Results on CityScapes (2 tasks) and CelebA (40 tasks) datasets. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. MR and Δ​m%\\Delta m\\% are the main metrics for MTL performance.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")