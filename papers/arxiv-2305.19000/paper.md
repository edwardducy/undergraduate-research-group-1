# Independent Component Alignment for Multi-Task Learning

 Dmitry Senushkin  Nikolay Patakin  Arseny Kuznetsov   Anton Konushin  
Samsung Research  
{d.senushkin, n.patakin, a.konushin}@samsung.com 

###### Abstract

In a multi-task learning (MTL) setting, a single model is trained to tackle a diverse set of tasks jointly. Despite rapid progress in the field, MTL remains challenging due to optimization issues such as conflicting and dominating gradients. In this work, we propose using a condition number of a linear system of gradients as a stability criterion of an MTL optimization. We theoretically demonstrate that a condition number reflects the aforementioned optimization issues. Accordingly, we present Aligned-MTL, a novel MTL optimization approach based on the proposed criterion, that eliminates instability in the training process by aligning the orthogonal components of the linear system of gradients. While many recent MTL approaches guarantee convergence to a minimum, task trade-offs cannot be specified in advance. In contrast, Aligned-MTL provably converges to an optimal point with pre-defined task-specific weights, which provides more control over the optimization result. Through experiments, we show that the proposed approach consistently improves performance on a diverse set of MTL benchmarks, including semantic and instance segmentation, depth estimation, surface normal estimation, and reinforcement learning. The source code is publicly available at [https://github.com/SamsungLabs/MTL](https://github.com/SamsungLabs/MTL "").

## 1 Introduction

In a multi-task learning (MTL), several tasks are solved jointly by a single model \[[2](#bib.bib2 ""), [10](#bib.bib10 "")\]. In such a scenario, information can be shared across tasks, which may improve the generalization and boost the performance for all objectives. Moreover, MTL can be extremely useful when computational resources are constrained, so it is crucial to have a single model capable of solving various tasks \[[30](#bib.bib30 ""), [17](#bib.bib17 ""), [19](#bib.bib19 "")\]. In reinforcement learning \[[39](#bib.bib39 ""), [50](#bib.bib50 "")\], MTL setting arises naturally, when a single agent is trained to perform multiple tasks.

Several MTL approaches \[[28](#bib.bib28 ""), [35](#bib.bib35 ""), [29](#bib.bib29 ""), [24](#bib.bib24 ""), [15](#bib.bib15 ""), [42](#bib.bib42 ""), [31](#bib.bib31 "")\] focus on designing specific network architectures and elaborate strategies of sharing parameters and representations across tasks for a given set of tasks. Yet, such complicated and powerful models are extremely challenging to train.

Direct optimization of an objective averaged across tasks might experience issues \[[54](#bib.bib54 "")\] related to conflicting and dominating gradients. Such gradients destabilize the training process and degrade the overall performance. Accordingly, some other MTL approaches address these issues with multi-task gradient descent: either using gradient altering \[[54](#bib.bib54 ""), [48](#bib.bib48 ""), [9](#bib.bib9 ""), [27](#bib.bib27 "")\] or task balancing \[[28](#bib.bib28 ""), [16](#bib.bib16 ""), [25](#bib.bib25 "")\]. Many recent MTL methods \[[48](#bib.bib48 ""), [27](#bib.bib27 ""), [37](#bib.bib37 "")\] guarantee convergence to a minimum, yet task trade-offs cannot be specified in advance. Unfortunately, the lack of control over relative task importance may cause some tasks to be compromised in favor of others \[[37](#bib.bib37 "")\].

In this work, we analyze the multi-task optimization challenges from the perspective of stability of a linear system of gradients. Specifically, we propose using a condition number of a linear system of gradients as a stability criterion of an MTL optimization. According to our thorough theoretical analysis, there is a strong relation between the condition number and conflicting and dominating gradients issues. We exploit this feature to create Aligned-MTL, a novel gradient manipulation approach, which is the major contribution of this work. Our approach resolves gradient conflicts and eliminates dominating gradients by aligning principal components of a gradient matrix, which makes the training process more stable. In contrast to other existing methods (*e.g*. \[[37](#bib.bib37 ""), [48](#bib.bib48 ""), [54](#bib.bib54 ""), [27](#bib.bib27 "")\]), Aligned-MTL has a provable guarantee of convergence to an optimum with pre-defined task weights.

We provide an in-depth theoretical analysis of the proposed method and extensively verify its effectiveness. Aligned-MTL consistently outperforms previous methods on various benchmarks. First, we evaluate the proposed approach on the problem of scene understanding; specifically, we perform joint instance segmentation, semantic segmentation, depth and surface normal estimation on two challenging datasets – Cityscapes \[[6](#bib.bib6 "")\] and NYUv2 \[[36](#bib.bib36 "")\]. Second, we apply our method to multi-task reinforcement learning and conduct experiments with the MT10 dataset \[[55](#bib.bib55 "")\]. Lastly, in order to analyze generalization performance, Aligned-MTL has been applied to two different network architectures, namely PSPNet \[[48](#bib.bib48 "")\] and MTAN \[[28](#bib.bib28 "")\], in the scene understanding experiments.

## 2 Related Work

(a) Uniform

![Refer to caption](2305.19000v1/2_cagrad_landscape.png)

(b) CAGrad (c\=0.4c=0.4) \[[26](#bib.bib26 "")\]

![Refer to caption](2305.19000v1/2_imtl_landscape.png)

(c) IMTL \[[27](#bib.bib27 "")\]

![Refer to caption](2305.19000v1/2_nashmtl_landscape.png)

(d) Nash-MTL \[[37](#bib.bib37 "")\]

![Refer to caption](2305.19000v1/2_amgda_landscape.png)

(e) Aligned-MTL (ours)

Figure 1: Comparison of MTL approaches on a challenging synthetic two-task benchmark \[[26](#bib.bib26 ""), [37](#bib.bib37 "")\]. We visualize optimization trajectories w.r.t. objectives value (ℒ1\\mathcal{L}\_{1} and ℒ2\\mathcal{L}\_{2}, top row), and cumulative objective w.r.t. parameters (θ1\\theta\_{1} and θ2\\theta\_{2}, bottom row). Initialization points are marked with ∙\\bullet, the Pareto front ([Def. 1](#Thmdefinition1 "Definition 1. ‣ 5.3 Convergence Analysis ‣ 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning")) is denoted as . Other MTL approaches produce noisy optimization trajectories ([Figs. 1a](#S2.F1.sf1 "In Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning"), [1b](#S2.F1.sf2 "Figure 1b ‣ Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning"), [1c](#S2.F1.sf3 "Figure 1c ‣ Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning") and [1d](#S2.F1.sf4 "Figure 1d ‣ Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning")) inside areas with conflicting and dominating gradients ([Fig. 2](#S3.F2 "In 3.1 Notation ‣ 3 Multi-Task Learning ‣ Independent Component Alignment for Multi-Task Learning")). In contrast, our approach converges to the global optimum (★\\bigstar) robustly. Approaches aiming to find a Pareto-stationary solution (such as [Fig. 1c](#S2.F1.sf3 "In Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning") and [Fig. 1d](#S2.F1.sf4 "In Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning")) terminate once the Pareto front is first reached, as a result, they might provide a suboptimal solution. Differently, Aligned-MTL drifts along the Pareto front and provably converges to the optimum w.r.t. pre-defined tasks weights.

A multi-task setting \[[2](#bib.bib2 ""), [44](#bib.bib44 ""), [8](#bib.bib8 "")\] is leveraged in computer vision \[[1](#bib.bib1 ""), [19](#bib.bib19 ""), [56](#bib.bib56 ""), [38](#bib.bib38 ""), [16](#bib.bib16 "")\], natural language processing \[[5](#bib.bib5 ""), [32](#bib.bib32 ""), [11](#bib.bib11 "")\], speech processing \[[47](#bib.bib47 "")\], and robotics \[[52](#bib.bib52 ""), [23](#bib.bib23 "")\] applications. Prior MTL approaches formulate the total objective as a weighted sum of task-specific objectives, with weights being manually tuned \[[17](#bib.bib17 ""), [21](#bib.bib21 ""), [34](#bib.bib34 "")\]. However, finding optimal weights via grid search is computationally inefficient. Kendall et al. \[[16](#bib.bib16 "")\] overcame this limitation, assigning task weights according to the homoscedastic uncertainty of each task. Other recent methods, such as GradNorm \[[3](#bib.bib3 "")\] and DWA \[[28](#bib.bib28 "")\], optimize weights based on task-specific learning rates or by random weighting \[[25](#bib.bib25 "")\].

The most similar to Aligned-MTL approaches (*e.g*. \[[54](#bib.bib54 ""), [9](#bib.bib9 ""), [27](#bib.bib27 ""), [26](#bib.bib26 ""), [37](#bib.bib37 "")\]) aim to mitigate effects of conflicting or dominating gradients. Conflicting gradients having opposing directions often induce a negative transfer (*e.g*. \[[22](#bib.bib22 "")\]). Among all approaches tackling this problem, the best results are obtained by those based on an explicit gradient modulation \[[26](#bib.bib26 ""), [54](#bib.bib54 ""), [27](#bib.bib27 "")\] where a gradient of a task which conflicts with a gradient of some other task is replaced with a modified, non-conflicting, gradient. Specifically, PCGrad \[[54](#bib.bib54 "")\] proposes a ”gradient surgery” which decorrelates a system of vectors, while CAGrad \[[26](#bib.bib26 "")\] aims at finding a conflict-averse direction to minimize overall conflicts. GradDrop \[[4](#bib.bib4 "")\] forces task gradients sign consistency. Other methods also address an issue of dominating gradients. Nash-MTL \[[37](#bib.bib37 "")\] leverages advances of game theory \[[37](#bib.bib37 "")\], while IMTL \[[27](#bib.bib27 "")\] searches for a gradient direction where all the cosine similarities are equal.

Several recent works \[[41](#bib.bib41 ""), [43](#bib.bib43 "")\] investigate a multiple-gradient descent algorithm (MGDA \[[9](#bib.bib9 ""), [12](#bib.bib12 ""), [45](#bib.bib45 "")\]) for MTL: these methods search for a direction that decreases all objectives according to multi-objective Karush–Kuhn–Tucker (KKT) conditions \[[20](#bib.bib20 "")\]. Sener and Koltun \[[48](#bib.bib48 "")\] propose extending the classical MGDA \[[9](#bib.bib9 "")\] so it scales well to high-dimensional problems for a specific use case. However, all the described approaches converge to an arbitrary Pareto-stationary solution, leading to a risk of imbalanced task performance.

## 3 Multi-Task Learning

Multi-task learning implies optimizing a single model with respect to multiple objectives. The recent works \[[54](#bib.bib54 ""), [9](#bib.bib9 ""), [39](#bib.bib39 ""), [16](#bib.bib16 "")\] have found that this learning problem is difficult to solve by reducing it to a standard single-task approach. In this section, we introduce a general notation and describe frequent challenges arising in gradient optimization in MTL.

### 3.1 Notation

In MTL, there are T\>1T>1 tasks. Each task is associated with an objective ℒi​(𝜽){\\mathcal{L}}\_{i}({\\bm{\\theta}}) depending on a set of model parameters 𝜽{\\bm{\\theta}} shared between all tasks. The goal of MTL training is to find a parameter 𝜽{\\bm{\\theta}} that minimizes an average loss:

𝜽∗\=argminθ∈ℝm{ℒ0(𝜽)\=def∑i\=1T1Tℒi(𝜽).}\\displaystyle{\\bm{\\theta}}^{\*}=\\arg\\min\_{\\theta\\in\\mathbb{R}^{m}}\\bigg\\{{\\mathcal{L}}\_{0}({\\bm{\\theta}})\\stackrel{{\\scriptstyle\\text{def}}}{{=}}\\sum\_{i=1}^{T}\\frac{1}{T}{\\mathcal{L}}\_{i}({\\bm{\\theta}}).\\bigg\\}

(1)

We introduce the following notation: 𝒈i\=∇ℒi​(𝜽){\\bm{g}}\_{i}=\\nabla{\\mathcal{L}}\_{i}({\\bm{\\theta}}) – individual task gradients; ℒ0​(𝜽){\\mathcal{L}}\_{0}({\\bm{\\theta}}) – a cumulative objective; 𝑮\={𝒈1,⋯,𝒈T}{\\bm{G}}=\\{{\\bm{g}}\_{1},\\cdots,{\\bm{g}}\_{T}\\} – a gradient matrix; wi\=def1Tw\_{i}\\stackrel{{\\scriptstyle\\text{def}}}{{=}}\\frac{1}{T} – pre-defined task weights. The task weights are supposed to be fixed. We omit task-specific parameters in our notation, since they are independent and not supposed to be balanced.

![Refer to caption](2305.19000v1/zhest.png)

Figure 2: Synthetic two-task MTL benchmark \[[26](#bib.bib26 ""), [37](#bib.bib37 "")\]. Loss landscapes w.r.t. individual objectives are depicted on the right side. The cumulative loss landscape (on the left side) contains areas with conflicting and dominating gradeints. 

### 3.2 Challenges

In practice, directly solving a multi-objective optimization problem via gradient descent may significantly compromise the optimization of individual objectives \[[54](#bib.bib54 "")\]. Simple averaging of gradients across tasks makes a cumulative gradient biased towards the gradient with the largest magnitude, which might cause overfitting for a subset of tasks. Conflicting gradients with negative cosine distance complicate the training process as well; along with dominating gradients, they increase inter-step direction volatility that decreases overall performance ([Fig. 1](#S2.F1 "In 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning")). To mitigate the undesired effects of conflicting and dominating gradients in MTL, we propose a criterion that is strongly correlated with the presence of such optimization issues. This measure is a condition number of a linear system of gradients.

## 4 Stability

The prevailing challenges in MTL are arguably task dominance and conflicting gradients, accordingly, various criteria for indicating and measuring these issues have been formulated. For instance, a gradient dominance can be measured with a gradient magnitude similarity (\[[54](#bib.bib54 "")\] Def. 2). Similarly, gradient conflicts can be estimated as a cosine distance between vectors (\[[54](#bib.bib54 "")\] Def. 1, \[[26](#bib.bib26 "")\]). However, each of these metrics describes a specific characteristic of a linear system of gradients, and cannot provide a comprehensive assessment if taken separately. We show that our stability criterion indicates the presence of both MTL challenges ([Fig. 4](#S6.F4 "In 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning")); importantly, it describes a whole linear system and can be trivially measured on any set of gradients. Together with magnitude similarity and cosine distance, this criterion accurately describes the training process.

### 4.1 Condition Number

Generally, the stability of an algorithm is its sensitivity to an input perturbation, or, in other words, how much the output changes if an input gets perturbed. In numerical analysis, the stability of a linear system is measured by a condition number of its matrix. In a multi-task optimization, a cumulative gradient is a linear combination of task gradients: 𝒈\=𝑮​𝒘{\\bm{g}}={\\bm{G}}{\\bm{w}}. Thus, the stability of a linear system of gradients can be measured as the condition number of a gradient matrix 𝑮{\\bm{G}}. The value of this stability criterion is equal to the ratio of the maximum and minimum singular values of the corresponding matrix:

κ⁡(𝑮)\=σm​a​xσm​i​n.\\kappa({\\bm{G}})=\\frac{\\sigma\_{max}}{\\sigma\_{min}}.

(2)

#### Remark.

A linear system is well-defined if its condition number is equal to one, and ill-posed if it is non-finite. A standard assumption for multi-task optimization is that a gradient system is not ill-posed, *i.e*. task gradients are linearly independent. In this work, we suppose that the linear independence assumption holds unless otherwise stated.

### 4.2 Condition Number and MTL Challenges

The dependence between the stability criterion and MTL challenges is two-fold. Let us consider a gradient system having a minimal condition number. According to the singular value decomposition theorem, its gradient matrix 𝑮^\\hat{{\\bm{G}}} with κ⁡(𝑮^)\=1\\kappa(\\hat{{\\bm{G}}})=1 must be orthogonal with equal singular values:

𝑮^\=𝑼​𝚺​𝑽⊤,where𝚺\=σ​𝑰\\hat{{\\bm{G}}}={\\bm{U}}{\\bm{\\Sigma}}{\\bm{V}}^{\\top},\\quad\\text{where}\\quad{\\bm{\\Sigma}}=\\sigma{\\bm{I}}

(3)

Moreover, since 𝑼,𝑽{\\bm{U}},{\\bm{V}} matrices are orthonormal, individual task gradients norms are equal to σ\\sigma. Thus, minimizing the condition number of the linear system of gradients leads to mitigating dominance and conflicts within this system.

On the other hand, if an initial linear system of gradients is not well-defined, reducing neither gradient conflict nor dominance only does not guarantee minimizing a condition number. The stability criterion reaches its minimum iff both issues are solved jointly and gradients are orthogonal. This restriction eliminates positive task gradients interference (co-directed gradients may produce κ\>1\\kappa>1), but it can guarantee the absence of negative interaction, which is essential for a stable training. Noisy convergence trajectories w.r.t. objectives values ([Fig. 1](#S2.F1 "In 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning"), top row) indicate instability of the training process.

To demonstrate the relation between MTL challenges and our stability criterion, we conduct a synthetic experiment as proposed in \[[26](#bib.bib26 ""), [37](#bib.bib37 "")\]. There are two objectives to be optimized, and the optimization landscape contains areas with conflicting and dominating gradients. We compare our approach against recent approaches that do not handle stability issues, yielding noisy trajectories in problematic areas. By enforcing stability, our method performs well on the synthetic benchmark.

(a) Initial gradients

(b) Gradients aligned via Aligned-MTL

(c) Accumulated MTL gradient

Figure 3: Geometric interpretation of our approach on a two-task MTL. Here, individual task gradients g1g\_{1} and g2g\_{2} are directed oppositely (conflict) and have different magnitude (dominance) ([Fig. 3a](#S4.F3.sf1 "In Figure 3 ‣ 4.2 Condition Number and MTL Challenges ‣ 4 Stability ‣ Independent Component Alignment for Multi-Task Learning")). Aligned-MTL enforces stability via aligning principal components u1u\_{1}, u2u\_{2} of an initial linear system of gradients. This can be interpreted as re-scaling axes of a coordinate system set by principal components, so that singular values of gradient matrix σ1\\sigma\_{1} and σ2\\sigma\_{2} are rescaled to be equal to the minimal singular value (σ2\\sigma\_{2}, in this case). The aligned gradients g1^\\hat{g\_{1}}, g2^\\hat{g\_{2}} are orthogonal (non-conflicting) and of equal magnitude (non-dominant) ([Fig. 3b](#S4.F3.sf2 "In Figure 3 ‣ 4.2 Condition Number and MTL Challenges ‣ 4 Stability ‣ Independent Component Alignment for Multi-Task Learning")). Finally, the aligned gradients are summed up with pre-defined tasks weights ww and 1−w1-w, resulting in a cumulative gradient g0^\\hat{g\_{0}} ([Fig. 3c](#S4.F3.sf3 "In Figure 3 ‣ 4.2 Condition Number and MTL Challenges ‣ 4 Stability ‣ Independent Component Alignment for Multi-Task Learning")). 

## 5 Aligned-MTL

We suppose that multi-task gradient optimization should successfully resolve the main MTL challenges: conflicts and dominance in gradient system. Unlike existing approaches \[[54](#bib.bib54 ""), [51](#bib.bib51 "")\] that focus on directly resolving the optimization problems, we develop an algorithm that handles issues related to the stability of a linear system of gradients and accordingly addresses both gradient conflicts and dominance.

Specifically, we aim to find a cumulative gradient g0^\\hat{g\_{0}}, so that ‖𝒈0−𝒈^0‖22\\|{\\bm{g}}\_{0}-\\hat{{\\bm{g}}}\_{0}\\|^{2}\_{2} is minimal, while a linear system of gradients is stable (κ⁡(𝑮^)\=1)\\big(\\kappa(\\hat{{\\bm{G}}})=1\\big). This constraint is defined up to an arbitrary positive scaling coefficient. Here, we assume σ\=1\\sigma=1 for simplicity. By applying a triangle inequality to the initial problem, we derive ‖𝒈0−𝒈^0‖22≤‖𝑮−𝑮^‖F2​‖𝒘‖22\\|{\\bm{g}}\_{0}-\\hat{{\\bm{g}}}\_{0}\\|^{2}\_{2}\\leq\\|{\\bm{G}}-\\hat{{\\bm{G}}}\\|^{2}\_{F}\\|{\\bm{w}}\\|^{2}\_{2}. Thereby, we consider the following optimization task:

min𝑮^⁡‖𝑮−𝑮^‖F2s.t.𝑮^⊤​𝑮^\=𝑰\\displaystyle\\min\_{\\hat{{\\bm{G}}}}\\|{\\bm{G}}-\\hat{{\\bm{G}}}\\|^{2}\_{F}\\quad\\text{s.t.}\\quad\\hat{{\\bm{G}}}^{\\top}\\hat{{\\bm{G}}}={\\bm{I}}

(4)

The stability criterion, a condition number, defines a linear system up to an arbitrary positive scale. To alleviate this ambiguity, we choose the largest scale that guarantees convergence to the optimum of an original problem ([Eq. 1](#S3.E1 "In 3.1 Notation ‣ 3 Multi-Task Learning ‣ Independent Component Alignment for Multi-Task Learning")): this is a minimal singular value of an initial gradient matrix σ\=σm​i​n​(𝑮)\>0\\sigma=\\sigma\_{min}({\\bm{G}})>0. The final linear system of gradients defined by 𝑮^\\hat{{\\bm{G}}} satisfies the optimality condition in terms of a condition number.

### 5.1 Gradient Matrix Alignment

The problem [Eq. 4](#S5.E4 "In 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning") can be treated as a special kind of Procrustes problem \[[46](#bib.bib46 "")\]. Fortunately, there exists a closed-form solution of this task. To obtain such a solution, we perform a singular value decomposition (SVD) and rescale singular values corresponding to principal components so that they are equal to the smallest singular value.

Technically, the matrix alignment can be performed in the parameter space or in the task space; being equivalent, these options have different computational costs. This duality is caused by SVD providing two different eigen decompositions of Gram matrices 𝑮⊤​𝑮{\\bm{G}}^{\\top}{\\bm{G}} and 𝑮​𝑮⊤{\\bm{G}}{\\bm{G}}^{\\top}:

𝑮^\=σ​𝑼​𝑽⊤\=σ​𝑼​𝚺−1​𝑼⊤⏟Parameter space​𝑮\=σ​𝑮​𝑽​𝚺−1​𝑽⊤⏟Task space\\hat{{\\bm{G}}}=\\sigma{\\bm{U}}{\\bm{V}}^{\\top}=\\sigma\\underbrace{{\\bm{U}}{\\bm{\\Sigma}}^{-1}{\\bm{U}}^{\\top}}\_{\\text{Parameter space}}{\\bm{G}}=\\sigma{\\bm{G}}\\underbrace{{\\bm{V}}{\\bm{\\Sigma}}^{-1}{\\bm{V}}^{\\top}}\_{\\text{Task space}}

(5)

We perform the gradient matrix alignment at each optimization step. Since the number of tasks TT is relatively small compared to the number of parameters, we operate in a task space. This makes a gradient matrix alignment more computationally efficient as we need to perform an eigen decomposition of a small T×TT\\times T matrix. [Fig. 3](#S4.F3 "In 4.2 Condition Number and MTL Challenges ‣ 4 Stability ‣ Independent Component Alignment for Multi-Task Learning") provides a geometric interpretation of our approach, while pseudo-code is given in [Alg. 1](#alg1 "In 5.2 Aligned-MTL: Upper Bound Approximation ‣ 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning").

#### Remark.

If an initial matrix 𝑮{\\bm{G}} is singular (gradients are linear dependent), then the smallest singular value is zero. Fortunately, the singular value decomposition provides a unique solution even in this case; yet, we need to choose the smallest singular value greater than zero as a global scale.

### 5.2 Aligned-MTL: Upper Bound Approximation

The major limitation of our approach is the need to run multiple backward passes through the shared part of the model to calculate the gradient matrix. The backward passes are computationally demanding, and the training time depends linearly on the number of tasks: if it is large, our approach may be non-applicable in practice.

This limitation can be mitigated for encoder-decoder networks, where each task prediction is computed using the same shared representation. We can employ the chain rule trick \[[48](#bib.bib48 "")\] to upper-bound an original objective ([Eq. 4](#S5.E4 "In 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning")):

‖𝑮−𝑮^‖F2≤‖∂𝑯∂θ‖F2​‖𝒁−𝒁^‖F2\\|{\\bm{G}}-\\hat{{\\bm{G}}}\\|^{2}\_{F}\\leq\\bigg\\|\\frac{\\partial{\\bm{H}}}{\\partial\\theta}\\bigg\\|^{2}\_{F}\\|{\\bm{Z}}-\\hat{{\\bm{Z}}}\\|^{2}\_{F}

(6)

Here, 𝑯{\\bm{H}} stands for a hidden shared representation, and 𝒁{\\bm{Z}} and 𝒁^\\hat{{\\bm{Z}}} are gradients of objective w.r.t. a shared representation of the initial and aligned linear system of gradients, respectively. Thus, the gradient alignment can be performed for a shared representation:

min𝒁^⁡‖𝒁−𝒁^‖F2s.t.𝒁^⊤​𝒁^\=𝑰\\displaystyle\\min\_{\\hat{{\\bm{Z}}}}\\|{\\bm{Z}}-\\hat{{\\bm{Z}}}\\|^{2}\_{F}\\quad\\text{s.t.}\\quad\\hat{{\\bm{Z}}}^{\\top}\\hat{{\\bm{Z}}}={\\bm{I}}

(7)

Aligning gradients of shared representation does not require additional backward passes, since matrix 𝒁{\\bm{Z}} is computed during a conventional backward pass. We refer to such an approximation of Aligned-MTL as to Aligned-MTL-UB. With O⁡(1)O(1) time complexity w.r.t. the number of tasks, this approximation tends to be significantly more efficient than the original Aligned-MTL having O⁡(T)O(T) time complexity.

Algorithm 1 Gradient matrix alignment

G∈ℝ|θ|×TG\\in\\mathbb{R}^{|\\theta|\\times T} – gradient matrix,     w∈ℝTw\\in\\mathbb{R}^{T} – task importance 

/\* Compute task space Gram matrix \*/ 

𝑴←𝑮⊤​𝑮{\\bm{M}}\\leftarrow{\\bm{G}}^{\\top}{\\bm{G}} 

 /\* Compute eigenvalues and eigenvectors of MM \*/ 

(λ,𝑽)←e​i​g​h​(𝑴)(\\lambda,{\\bm{V}})\\leftarrow eigh({\\bm{M}}) 

𝚺−1←diag(1λ1,⋯1λR){\\bm{\\Sigma}}^{-1}\\leftarrow diag\\left(\\sqrt{\\frac{1}{\\lambda\_{1}}},\\cdots\\sqrt{\\frac{1}{\\lambda\_{R}}}\\right) 

/\* Compute balance transformation \*/ 

𝑩←λR​𝑽​𝚺−1​𝑽⊤{\\bm{B}}\\leftarrow\\sqrt{\\lambda\_{R}}{\\bm{V}}{\\bm{\\Sigma}}^{-1}{\\bm{V}}^{\\top} 

𝜶←𝑩​𝒘{\\bm{\\alpha}}\\leftarrow{\\bm{B}}{\\bm{w}} 

return​𝑮​𝜶\\textbf{return}\\ {\\bm{G}}{\\bm{\\alpha}} 

### 5.3 Convergence Analysis

In this section, we formulate a theorem regarding the convergence of our approach. Similar to single-task optimization converging to a stationary point, our MTL approach converges to a Pareto-stationary solution.

###### Definition 1.

A solution 𝛉∗∈𝚯{\\bm{\\theta}}^{\*}\\in\\bm{\\Theta} is called Pareto-stationary iff there exists a convex combination of the gradient-vectors that is equal to zero. All possible Pareto-stationary solutions form a Pareto set (or Pareto front). 

The overall model performance may vary significantly within points of the Pareto front. Recent MTL approaches \[[37](#bib.bib37 ""), [27](#bib.bib27 "")\] that provably converge to an arbitrary Pareto-stationary solution, tend to overfit to a subset of tasks. In contrast, our approach converges to a Pareto-stationary point with pre-defined tasks weights, thus providing more control over an optimization result [Eq. 1](#S3.E1 "In 3.1 Notation ‣ 3 Multi-Task Learning ‣ Independent Component Alignment for Multi-Task Learning").

###### Theorem 1.

Assume ℒ0​(𝛉),…,ℒT​(𝛉){\\mathcal{L}}\_{0}({\\bm{\\theta}}),\\dots,{\\mathcal{L}}\_{T}({\\bm{\\theta}}) are lower-bounded continuously differentiable functions with Lipschitz continuous gradients with Λ\>0\\Lambda>0. A gradient descent with aligned gradient and step size α≤1Λ\\alpha\\leq\\frac{1}{\\Lambda} converges linearly to a Pareto stationary point where ∇ℒ0​(𝛉)\=0\\nabla{\\mathcal{L}}\_{0}({\\bm{\\theta}})=0. 

A similar theorem is valid for aligning gradients in the shared representation space (Aligned-MTL upper-bound approximation is described in [Sec. 5.2](#S5.SS2 "5.2 Aligned-MTL: Upper Bound Approximation ‣ 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning")). Mathematical proofs of both versions of this theorem versions are provided in supplementary materials.

## 6 Experiments

(a) Condition Number

(b) Gradient Magnitude Similarity \[[54](#bib.bib54 "")\]

(c) Gradient Conflicts

Figure 4: Empirical evaluation of a stability criterion. We plot a condition number ([Eq. 2](#S4.E2 "In 4.1 Condition Number ‣ 4 Stability ‣ Independent Component Alignment for Multi-Task Learning")), gradient magnitude similarity \[[54](#bib.bib54 "")\], and minimal cosine distance during training on the CityScapes three-task benchmark. This benchmark suffers from high dominance since instance segmentation loss is of much larger scale than the others. The most intuitive way to define the dominance is the maximum ratio of task gradients magnitudes. The condition number coincides with this definition in a nearly orthogonal case, as in this benchmark [Fig. 4c](#S6.F4.sf3 "In Figure 4 ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning"). However, gradient magnitude similarity measure [Fig. 4b](#S6.F4.sf2 "In Figure 4 ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning") proposed in \[[54](#bib.bib54 "")\] does not reveal much correlation with a condition number(and, accordingly, with a maximal gradients magnitude ratio) [Fig. 4a](#S6.F4.sf1 "In Figure 4 ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning"), so we assume it does not represent dominance issues comprehensively. From the empirical point of view [Table 1](#S6.T1 "In 6.1 Synthetic Example ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning"), the value of a target metric is more correlated with the condition number, than with the gradient magnitude similarity. 

We empirically demonstrate the effectiveness of the proposed approach on various multi-task learning benchmarks, including scene understanding, multi-target regression, and reinforcement learning.

Competitors. We consider the following MTL approaches: (1) Linear Scalarization (LS, Uniform baseline): optimizing a uniformly weighted sum of individual task objectives, *i.e*. 1T​∑tℒt\\frac{1}{T}\\sum\_{t}{{\\mathcal{L}}\_{t}}; (2) Dymanic Weight Average (DWA) \[[28](#bib.bib28 "")\]: adjusting task weights based on the rates of loss changes over time; (3) Uncertainty \[[16](#bib.bib16 "")\] weighting; (4) MGDA \[[9](#bib.bib9 "")\]: a multi-objective optimization with KKT \[[20](#bib.bib20 "")\] conditions; (5) MGDA-UB \[[48](#bib.bib48 "")\]: optimizing an upper bound for the MGDA optimization objective; (6) GradNorm \[[3](#bib.bib3 "")\]: normalizing the gradients to balance the learning of multiple tasks; (7) GradDrop \[[4](#bib.bib4 "")\]: forcing the sign consistency between task gradients; (8) PCGrad \[[54](#bib.bib54 "")\]: performing gradient projection to avoid the negative interactions between tasks gradients; (9) GradVac \[[51](#bib.bib51 "")\]: leveraging task relatedness to set gradient similarity objectives and adaptively align task gradients, (10) CAGrad \[[26](#bib.bib26 "")\]: finding a conflict-averse gradients; (11) IMTL \[[27](#bib.bib27 "")\]: aligning projections to task gradients; (12) Nash-MTL \[[37](#bib.bib37 "")\]: utilizing a bargaining games for gradient computation, and (13) Random loss weighting (RLW) \[[25](#bib.bib25 "")\] with normal distribution. The proposed approach and the baseline methods are implemented using the PyTorch framework \[[40](#bib.bib40 "")\]. The technical details on the training schedules and a complete listing of hyperparameters are provided in supplementary materials.

Evaluation metrics. Besides task specific metrics we follow Maninis *et al*. \[[33](#bib.bib33 "")\] and report a model performance drop relative to a single task baseline averaged over tasks: Δ​mt​a​s​k\=1T​∑t\=1T∑k\=1nt(−1)σt​k​(Mm,t​k−Mb,t​k)/Mb,t​k\\Delta m\_{task}=\\frac{1}{T}\\sum\_{t=1}^{T}\\sum\_{k=1}^{n\_{t}}(-1)^{\\sigma\_{tk}}(M\_{m,tk}-M\_{b,tk})/M\_{b,tk} – or over metrics: Δ​mm​e​t​r​i​c\=1T​∑t\=1T(−1)σt​(Mm,t−Mb,t)/Mb,t\\Delta m\_{metric}=\\frac{1}{T}\\sum\_{t=1}^{T}(-1)^{\\sigma\_{t}}(M\_{m,t}-M\_{b,t})/M\_{b,t}. Here, Mm,t​kM\_{m,tk} denotes the performance of a model mm on a task tt, measured with a metric kk. Similarly, Mb,t​kM\_{b,tk} is a performance of a single-task tt baseline; ntn\_{t} denotes number of metrics per task tt. σt​k\=1\\sigma\_{tk}=1 if higher values of metric is better, and σt​k\=0\\sigma\_{tk}=0 otherwise. We mostly rely on the task-weighted measure since the metric-weighted criterion tends to be biased to a task with high number of metrics.

### 6.1 Synthetic Example

To illustrate the proposed approach, we consider a synthetic MTL task ([Fig. 2](#S3.F2 "In 3.1 Notation ‣ 3 Multi-Task Learning ‣ Independent Component Alignment for Multi-Task Learning")) introduced in \[[26](#bib.bib26 "")\] (a formal definition is provided in the supplementary material). As shown in [Fig. 1](#S2.F1 "In 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning"), we perform optimization from five initial points tagged with ∙\\bullet. IMTL \[[27](#bib.bib27 "")\], and Nash-MTL \[[37](#bib.bib37 "")\] aims at finding Pareto-stationary solution ([Def. 1](#Thmdefinition1 "Definition 1. ‣ 5.3 Convergence Analysis ‣ 5 Aligned-MTL ‣ Independent Component Alignment for Multi-Task Learning")). As a result, they terminate optimization once they reach a solution in the Pareto front. Accordingly, the final result strongly depends on an initialization point, and the optimization may not converge to the global optimum ★\\bigstar in some cases ([Fig. 1c](#S2.F1.sf3 "In Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning") and [Fig. 1d](#S2.F1.sf4 "In Figure 1 ‣ 2 Related Work ‣ Independent Component Alignment for Multi-Task Learning")). Meanwhile, Aligned-MTL provides a less noisy and more stable trajectory, and provably converges to an optimum.

Table 1: Scene understanding (CityScapes: three tasks). We report PSPNet \[[57](#bib.bib57 ""), [48](#bib.bib48 "")\] model performance averaged over 3 random seeds. The best scores are provided in .

|            |            |          |            |
| ---------- | ---------- | -------- | ---------- |
| mIoU \[%\] | L1 \[px\]  | MSE      |            |
| 66.7366.73 | 10.5510.55 | 0.330.33 | –          |
| 52.9852.98 | 10.8910.89 | 0.390.39 | 14.3014.30 |
| 51.2651.26 | 10.2510.25 | 0.410.41 | 15.5815.58 |
| 53.1553.15 | 10.2210.22 | 0.400.40 | 13.2013.20 |
| 60.1260.12 | 9.879.87   | 0.330.33 | 1.531.53   |
| 66.7266.72 | 17.0217.02 | 0.330.33 | 20.6220.62 |
| 66.3766.37 | 18.6318.63 | 0.320.32 | 25.0525.05 |
| 57.2457.24 | 10.2910.29 | 0.350.35 | 6.556.55   |
| 52.9852.98 | 10.0910.09 | 0.400.40 | 12.5012.50 |
| 54.0654.06 | 9.919.91   | 0.380.38 | 10.0010.00 |
| 54.0754.07 | 10.3910.39 | 0.400.40 | 12.9912.99 |
| 64.3364.33 | 10.1510.15 | 0.340.34 | 1.461.46   |
| 65.1365.13 | 11.5811.58 | 0.320.32 | 3.103.10   |
| 64.8464.84 | 11.9011.90 | 0.370.37 | 9.389.38   |
| 67.0667.06 | 10.6310.63 | 0.330.33 | −0.02-0.02 |
| 66.0766.07 | 10.5410.54 | 0.320.32 | −0.35-0.35 |

Table 2: Scene understanding (NYUv2, three tasks). We report MTAN \[[28](#bib.bib28 "")\] model performance averaged over 3 random seeds. The best scores are provided in . 

Segmentation ↑\\uparrow

Depth ↓\\downarrow

Surface normals ↓\\downarrow

Angle Dist. ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

𝚫​𝐦%\\mathbf{\\Delta m\\%} ↓\\downarrow

𝚫​𝐦%\\mathbf{\\Delta m\\%} ↓\\downarrow

Method

mIoU

Pix Acc

Abs.

Rel.

Mean

Median

11.25

22.5

30

Metric-weighted

Task-weighted

Single task baselines

38.3038.30

63.7663.76

0.680.68

0.280.28

25.0125.01

19.2119.21

30.1430.14

57.2057.20

69.1569.15

–

–

Baseline: Uniform

39.2939.29

65.3365.33

0.54930.5493

0.22630.2263

28.1528.15

23.9623.96

22.0922.09

47.5047.50

61.0861.08

5.46

-1.07

RLW \[[25](#bib.bib25 "")\]

37.1737.17

63.7763.77

0.580.58

0.240.24

28.2728.27

24.1824.18

22.2622.26

47.0547.05

60.6260.62

7.67

2.00

DWA \[[28](#bib.bib28 "")\]

39.1139.11

65.3165.31

0.550.55

0.230.23

27.6127.61

23.1823.18

24.1724.17

50.1850.18

62.3962.39

3.49

-2.06

Uncertainty \[[16](#bib.bib16 "")\]

36.8736.87

63.1763.17

0.540.54

0.230.23

27.0427.04

22.6122.61

23.5423.54

49.0549.05

63.6563.65

4.01

-0.97

MGDA \[[48](#bib.bib48 "")\]

30.4730.47

59.9059.90

0.610.61

0.260.26

24.8824.88

19.4519.45

29.1829.18

56.8856.88

69.3669.36

1.47

1.79

GradNorm \[[3](#bib.bib3 "")\]

20.0920.09

52.0652.06

0.720.72

0.280.28

24.8324.83

18.8618.86

30.8130.81

57.9457.94

69.7369.73

7.22

11.51

GradDrop \[[4](#bib.bib4 "")\]

39.3939.39

65.1265.12

0.550.55

0.230.23

27.4827.48

22.9622.96

23.3823.38

49.4449.44

62.8762.87

3.61

-2.03

PCGrad \[[54](#bib.bib54 "")\]

38.0638.06

64.6464.64

0.560.56

0.230.23

27.4127.41

22.8022.80

23.8623.86

49.8349.83

63.1463.14

3.83

-1.33

GradVac \[[51](#bib.bib51 "")\]

37.5337.53

64.3564.35

0.560.56

0.240.24

27.6627.66

23.3823.38

22.8322.83

48.6648.66

62.2162.21

5.44

0.01

CAGrad \[[26](#bib.bib26 "")\]

39.7939.79

65.4965.49

0.550.55

0.230.23

26.3126.31

21.5821.58

25.6125.61

52.3652.36

65.5865.58

0.29

-4.18

IMTL \[[27](#bib.bib27 "")\]

39.3539.35

65.6065.60

0.540.54

0.230.23

26.0226.02

21.1921.19

26.2026.20

53.1353.13

66.2466.24

-0.59

-4.76

Nash-MTL \[[37](#bib.bib37 "")\]

40.1340.13

65.9365.93

0.530.53

0.220.22

25.2625.26

20.0820.08

28.4028.40

55.4755.47

68.1568.15

-4.04

-7.56

Aligned-MTL (ours)

40.8240.82

66.3366.33

0.530.53

0.220.22

25.1925.19

19.7119.71

28.8828.88

56.2356.23

68.5468.54

-4.93

-8.40

Aligned-MTL-UB (ours)

43.1143.11

67.2267.22

0.550.55

0.220.22

25.6725.67

20.5720.57

27.5827.58

54.3754.37

67.1267.12

-3.48

-7.83

Table 3: Scene understanding (NYUv2, three tasks). We report PSPNet \[[57](#bib.bib57 ""), [48](#bib.bib48 "")\] model performance averaged over 3 random seeds. The best scores are provided in . 

Segmentation ↑\\uparrow

Depth ↓\\downarrow

Surface normals ↓\\downarrow

Angle Dist. ↓\\downarrow

Within t∘t^{\\circ} ↑\\uparrow

𝚫​𝐦%\\mathbf{\\Delta m\\%} ↓\\downarrow

𝚫​𝐦%\\mathbf{\\Delta m\\%} ↓\\downarrow

Method

mIoU

Pix Acc

Abs.

Rel.

Mean

Median

11.25

22.5

30

Metric-weighted

Task-weighted

Single task baselines

49.3749.37

72.0372.03

0.520.52

0.240.24

22.9722.97

16.9416.94

0.340.34

0.620.62

0.730.73

–

–

Baseline:Uniform

45.2145.21

69.7069.70

0.490.49

0.210.21

26.1026.10

21.0821.08

0.260.26

0.520.52

0.660.66

8.97

4.72

RLW \[[25](#bib.bib25 "")\]

46.1946.19

69.7169.71

0.460.46

0.190.19

26.0926.09

21.0921.09

0.270.27

0.530.53

0.660.66

6.67

1.73

DWA \[[28](#bib.bib28 "")\]

45.8345.83

69.6569.65

0.500.50

0.220.22

26.1026.10

21.2721.27

0.260.26

0.520.52

0.660.66

9.64

5.61

MGDA \[[48](#bib.bib48 "")\]

40.9640.96

65.8065.80

0.540.54

0.220.22

23.3623.36

17.4517.45

0.330.33

0.610.61

0.720.72

3.54

4.24

MGDA-UB \[[48](#bib.bib48 "")\]

41.1541.15

65.1065.10

0.530.53

0.220.22

23.4223.42

17.6017.60

0.320.32

0.600.60

0.720.72

4.02

4.40

GradNorm \[[3](#bib.bib3 "")\]

45.6345.63

69.6469.64

0.480.48

0.200.20

25.4625.46

20.0620.06

0.280.28

0.550.55

0.670.67

5.88

2.18

GradDrop \[[4](#bib.bib4 "")\]

45.6945.69

70.1370.13

0.490.49

0.200.20

26.1626.16

21.2121.21

0.260.26

0.520.52

0.650.65

8.60

3.92

PCGrad \[[54](#bib.bib54 "")\]

46.3746.37

69.6969.69

0.480.48

0.200.20

26.0026.00

21.0521.05

0.260.26

0.530.53

0.660.66

7.78

3.17

GradVac \[[51](#bib.bib51 "")\]

46.6546.65

69.9769.97

0.490.49

0.210.21

25.9525.95

20.8820.88

0.270.27

0.530.53

0.660.66

7.89

3.75

CAGrad \[[26](#bib.bib26 "")\]

45.4645.46

69.3569.35

0.470.47

0.200.20

24.2824.28

18.7318.73

0.300.30

0.580.58

0.700.70

2.66

0.13

IMTL \[[27](#bib.bib27 "")\]

44.0244.02

68.5668.56

0.470.47

0.190.19

23.6923.69

18.0318.03

0.320.32

0.590.59

0.720.72

0.76

-1.02

Nash-MTL \[[37](#bib.bib37 "")\]

47.2547.25

70.3870.38

0.460.46

0.200.20

23.9523.95

18.8318.83

0.310.31

0.590.59

0.710.71

1.13

-1.48

Aligned-MTL (ours)

46.7046.70

69.9769.97

0.460.46

0.190.19

24.1924.19

18.7718.77

0.300.30

0.580.58

0.710.71

1.44

-1.55

Aligned-MTL-UB (ours)

46.4746.47

69.9269.92

0.480.48

0.200.20

24.3724.37

18.8818.88

0.300.30

0.580.58

0.700.70

2.70

0.07

### 6.2 Scene Understanding

The evaluation is performed on NYUv2 \[[36](#bib.bib36 "")\] and CityScapes \[[6](#bib.bib6 ""), [7](#bib.bib7 "")\] datasets. We leverage two network architectures: Multi-Task Attention Network (MTAN) \[[28](#bib.bib28 "")\] and Pyramid Scene Parsing Network (PSPNet) \[[57](#bib.bib57 ""), [48](#bib.bib48 "")\] on scene understanding benchmarks. MTAN applies a multi-task specific attention mechanism built upon MTL SegNet \[[16](#bib.bib16 "")\]. PSPNet features a dilated ResNet \[[14](#bib.bib14 "")\] backbone and multiple decoders with pyramid parsing modules \[[57](#bib.bib57 "")\]. Both networks were previously used in MTL benchmarks \[[48](#bib.bib48 "")\].

NYUv2. Following Liu *et al*. \[[28](#bib.bib28 ""), [26](#bib.bib26 ""), [37](#bib.bib37 "")\], we evaluate the performance of our approach on the NYUv2 \[[36](#bib.bib36 "")\] dataset by jointly solving semantic segmentation, depth estimation, and surface normal estimation tasks. We use both MTAN \[[28](#bib.bib28 "")\] and PSPNet \[[48](#bib.bib48 "")\] model architectures.

For MTAN, we strictly follow the training procedure described in \[[37](#bib.bib37 ""), [26](#bib.bib26 "")\]: training at 384×\\times288 resolution for 200 epochs with Adam \[[18](#bib.bib18 "")\] optimizer and 10−410^{-4} initial learning rate, halved after 100 epochs. The evaluation results are presented in [Table 2](#S6.T2 "In 6.1 Synthetic Example ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning"). We report metric values averaged across three random initializations as in previous works. We calculate both metric-weighted measure to compare with previous works alongside a task-weighted Δ​m\\Delta m modification. We claim the latter measure to be more important, as it is not biased towards surface normal estimation, thereby assessing overall performance more fairly. Accordingly, it exposes inconsistent task performance of GradNorm \[[3](#bib.bib3 "")\] and MGDA \[[48](#bib.bib48 "")\], which are biased towards surface normal estimation task and perform poorly on semantic segmentation. Although the MTAN model is not encoder-decoder architecture, our Aligned-MTL-UB approach outperforms all previous MTL optimization methods according to task-weighted Δ​m\\Delta m. Our original Aligned-MTL approach improves model performance even further in terms of both metrics.

We report results of PSPNet ([Table 3](#S6.T3 "In 6.1 Synthetic Example ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning")), trained on NYUv2 \[[36](#bib.bib36 "")\] following the same experimental setup. PSPNet architecture establishes much stronger baselines for all three tasks than vanilla SegNet. As a result, most of MTL approaches fail to outperform single-task models. According to the task-weighted metric, only two previous approaches provide solutions better than single-task baselines, while our Aligned-MTL approach demonstrates the best results.

CityScapes: two-task. We follow Liu *et al*. \[[26](#bib.bib26 "")\] experimental setup for Cityscapes \[[6](#bib.bib6 "")\], which implies jointly addressing semantic segmentation and depth estimation with a single MTAN \[[28](#bib.bib28 "")\] model. According to it, the original 19 semantic segmentation categories are classified into 7 categories. Our Aligned-MTL approach demonstrates the best results according to semantic segmentation and overall Δ​m\\Delta m metric. Our upper bound approximation of our Aligned-MTL again achieves a competitive performance, although MTAN does not satisfy architectural requirements.

CityScapes: three-task. We adopt a more challenging experimental setup \[[16](#bib.bib16 ""), [48](#bib.bib48 "")\], and address MTL with disparity estimation and instance and semantic segmentation tasks. The instance segmentation is reformulated as a centroid regression \[[16](#bib.bib16 "")\], so that the instance objective has a much larger scale than others. In this benchmark, we utilize the training setup proposed by Sener and Koltun \[[48](#bib.bib48 "")\]: 100 epochs, Adam optimizer with learning rate 10−410^{-4}. Input images are rescaled to 256×512256\\times 512, and a full set of labels is used for semantic segmentation. While many recent approaches experience a considerable performance drop ([Table 1](#S6.T1 "In 6.1 Synthetic Example ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning")), our method performs robustly even in this challenging scenario.

### 6.3 Multi-task Reinforcement Learning

Following \[[26](#bib.bib26 ""), [37](#bib.bib37 ""), [54](#bib.bib54 "")\], we consider an MTL reinforcement learning benchmark MT10 in a MetaWorld \[[55](#bib.bib55 "")\] environment. In this benchmark, a robot is being trained to perform actions, *e.g*. pressing a button and opening a window. Each action is treated as a task, and the primary goal is to successfully perform a total of 10 diverse manipulation tasks. In this experiment, we compare against the optimization-based baseline Soft Actor-Critic (SAC) \[[13](#bib.bib13 "")\] trained with various gradient altering methods \[[26](#bib.bib26 ""), [37](#bib.bib37 ""), [54](#bib.bib54 "")\]. We also consider MTL-RL \[[49](#bib.bib49 "")\]-based approaches: specifically, MTL SAC with a shared model, Multi-task SAC with task encoder (MTL SAC + TE) \[[55](#bib.bib55 "")\], Multi-headed SAC (MH SAC) with task-specific heads \[[55](#bib.bib55 "")\], Soft Modularization (SM) \[[53](#bib.bib53 "")\] and CARE \[[49](#bib.bib49 "")\]. The Aligned-MTL method has higher success rates, superseding competitors by a notable margin.

### 6.4 Empirical Analysis of Stability Criterion

In this section, we analyze gradient magnitude similarity, cosine distance, and condition number empirically. We use CityScapes three-task benchmark for this purpose, which suffers from the dominating gradients. According to the gradient magnitude similarity measure, PCGrad \[[54](#bib.bib54 "")\], Uniform, and CAGrad \[[26](#bib.bib26 "")\] tend to suffer from gradient dominance. For PCGrad and Uniform baseline, imbalanced convergence rates for different tasks result in a suboptimal solution ([Table 1](#S6.T1 "In 6.1 Synthetic Example ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning")). Differently, a well-performing CAGrad is misleadingly marked as problematic by gradient magnitude similarity. In contrast, the stability criterion – condition number – reveals domination issues for PCGrad and Uniform baselines and indicates a sufficient balance of different tasks for CAGrad (κ≈5\\kappa\\approx 5). Thus, the condition number exposes the training issues more evidently ([Fig. 4a](#S6.F4.sf1 "In Figure 4 ‣ 6 Experiments ‣ Independent Component Alignment for Multi-Task Learning")). The experimental evaluation shows that with κ≤10\\kappa\\leq 10, model tends to converge to an optimum with better overall performance.

Table 4: Scene understanding (CityScapes: two tasks). MTAN \[[28](#bib.bib28 "")\] model performance is reported as average over 3 random seeds. The best scores are provided in .

|                       |                     |                      |                      |            |
| --------------------- | ------------------- | -------------------- | -------------------- | ---------- |
| mIoU \[%\] ↑\\uparrow | Pix. Acc ↑\\uparrow | Abs Err ↓\\downarrow | Rel Err ↓\\downarrow |            |
| 74.0174.01            | 93.1693.16          | 0.01250.0125         | 27.7727.77           | –          |
| 75.1875.18            | 93.4993.49          | 0.01550.0155         | 46.7746.77           | 22.6022.60 |
| 74.5774.57            | 93.4193.41          | 0.01580.0158         | 47.7947.79           | 24.3724.37 |
| 75.2475.24            | 93.5293.52          | 0.01600.0160         | 44.3744.37           | 21.4321.43 |
| 72.0272.02            | 92.8592.85          | 0.01400.0140         | 30.1330.13           | 5.885.88   |
| 68.8468.84            | 91.5491.54          | 0.03090.0309         | 33.5033.50           | 44.1444.14 |
| 73.7273.72            | 93.0493.04          | 0.01240.0124         | 34.1134.11           | 5.635.63   |
| 75.2775.27            | 93.5393.53          | 0.01570.0157         | 47.5447.54           | 23.6723.67 |
| 75.1375.13            | 93.4893.48          | 0.01540.0154         | 42.0742.07           | 18.2118.21 |
| 75.1675.16            | 93.4893.48          | 0.01410.0141         | 37.6037.60           | 11.5811.58 |
| 75.3375.33            | 93.4993.49          | 0.01350.0135         | 38.4138.41           | 11.0411.04 |
| 75.4175.41            | 93.6693.66          | 0.01290.0129         | 35.0235.02           | 6.726.72   |
| 75.7775.77            | 93.6993.69          | 0.01330.0133         | 32.6632.66           | 5.275.27   |
| 74.8974.89            | 93.4693.46          | 0.01310.0131         | 33.9233.92           | 6.376.37   |

Table 5: Reinforcement learning (MT10). Average success rate on validation over 10 seeds. 

|              | Success ± SEM |
| ------------ | ------------- |
| 0.90 ± 0.032 |               |
| 0.49 ± 0.073 |               |
| 0.54 ± 0.047 |               |
| 0.61 ± 0.036 |               |
| 0.73 ± 0.043 |               |
| 0.84 ± 0.051 |               |
| 0.72 ± 0.022 |               |
| 0.83 ± 0.045 |               |
| 0.91 ± 0.031 |               |
| 0.97 ± 0.045 |               |

## 7 Discussion

The main limitation of Aligned-MTL is its computational optimization cost which scales linearly with the number of tasks. The upper-bound approximation of the Aligned-MTL method can be efficiently applied for encoder-decoder architectures using the same Jacobian over the shared representation. This approximation reduces instability, yet, it does not eliminate it since the Jacobian cannot be aligned. For non-encoder-decoder networks, upper-bound approximation has no theoretical guarantees but still can be leveraged as a heuristic and even provide a decent performance.

## 8 Conclusion

In this work, we introduced a stability criterion for multi-task learning, and proposed a novel gradient manipulation approach that optimizes this criterion. Our Aligned-MTL approach stabilize the training procedure by aligning the principal components of the gradient matrix. In contrast to many previous methods, this approach guarantees convergence to the local optimum with pre-defined task weights, providing a better control over the optimization results. Additionally, we presented a computationally efficient approximation of Aligned-MTL. Through extensive evaluation, we proved our approach consistently outperforms previous MTL optimization methods on various benchmarks including scene understanding and multi-task reinforcement learning.

Acknowledgements. We sincerely thank Anna Vorontsova, Iaroslav Melekhov, Mikhail Romanov, Juho Kannala and Arno Solin for their helpful comments, disscussions and proposed improvements regarding this paper. 

## References

*   \[1\] Hakan Bilen and Andrea Vedaldi. Integrated perception with recurrent multi-task neural networks. In Advances in Neural Information Processing Systems (NIPS), volume 29, pages 235–243. Curran Associates, Inc., 2016.
*   \[2\] Richard Caruana. Multitask learning: A knowledge-based source of inductive bias. In Proceedings of the Tenth International Conference on Machine Learning (ICML), pages 41–48. Morgan Kaufmann, 1993.
*   \[3\] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 794–803. PMLR, 2018.
*   \[4\] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 2039–2050. Curran Associates, Inc., 2020.
*   \[5\] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning (ICML), pages 160–167. ACM, 2008.
*   \[6\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
*   \[7\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Scharwächter, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset. In CVPR Workshop on The Future of Datasets in Vision, 2015.
*   \[8\] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796, 2020.
*   \[9\] Jean-Antoine Désidéri. Multiple-gradient descent algorithm for multiobjective optimization. In European Congress on Computational Methods in Applied Sciences and Engineering (ECCOMAS), 2012.
*   \[10\] Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In ICCV, pages 2051–2060, 2017.
*   \[11\] Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. Multi-task learning for multiple language translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing, pages 1723–1732. Association for Computational Linguistics, 2015.
*   \[12\] Jörg Fliege and Benar Fux Svaiter. Steepest descent methods for multicriteria optimization. Mathematical Methods of Operations Research, 51:479–494, 2000.
*   \[13\] Tuomas Haarnoja, Aurick Zhou, P. Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In ICML, 2018.
*   \[14\] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.
*   \[15\] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzȩbski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR, 2019.
*   \[16\] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, pages 7482–7491, 2018.
*   \[17\] Alex Kendall, Matthew Grimes, and Roberto Cipolla. Posenet: A convolutional network for real-time 6-dof camera relocalization. In ICCV, pages 2938–2946, 2015.
*   \[18\] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, ICLR, 2015.
*   \[19\] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In CVPR, pages 6129–6138, 2017.
*   \[20\] Harold W. Kuhn and Albert W. Tucker. Nonlinear programming. In Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability. University of California Press, 1951.
*   \[21\] Zakaria Laskar, Iaroslav Melekhov, Surya Kalia, and Juho Kannala. Camera relocalization by computing pairwise relative poses using convolutional neural network. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops, pages 920–929, 2017.
*   \[22\] Hae Beom Lee, Eunho Yang, and Sung Ju Hwang. Deep asymmetric multi-task feature learning. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 2956–2964. PMLR, 2018.
*   \[23\] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 1:1334–1373, 2016.
*   \[24\] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Universal representations: A unified look at multiple task and domain learning. arXiv preprint arXiv:2204.02744, 2022.
*   \[25\] Baijiong Lin, Feiyand Ye, Yu Zhang, and Ivor W. Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. arXiv preprint arXiv:2111.10603, 2022.
*   \[26\] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 18878–18890. Curran Associates, Inc., 2021.
*   \[27\] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In ICLR, 2021.
*   \[28\] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. In CVPR, pages 1871–1880, 2019.
*   \[29\] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. In CVPR, pages 1871–1880, 2019.
*   \[30\] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, volume 57. Association for Computational Linguistics, 2019.
*   \[31\] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In CVPR, pages 10434–10443, 2020.
*   \[32\] Minh-Thang Luong, Quoc Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. ICLR, 2015.
*   \[33\] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
*   \[34\] Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, and Esa Rahtu. Image-based localization using hourglass networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops, pages 870–877, 2017.
*   \[35\] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In CVPR, pages 3994–4003, 2016.
*   \[36\] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.
*   \[37\] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, pages 16428–16446. PMLR, 2022.
*   \[38\] Vladimir Nekrasov, Thanuja Dharmasiri, Andrew Spek, Tom Drummond, Chunhua Shen, and Ian Reid. Real-time joint semantic segmentation and depth estimation using asymmetric annotations. In International Conference on Robotics and Automation (ICRA), pages 7101–7107. IEEE, 2019.
*   \[39\] Emilio Parisotto, Lei Jimmy Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In ICLR, 2016.
*   \[40\] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, pages 8026–8037. Curran Associates, Inc., 2019.
*   \[41\] Sebastian Peitz and Michael Dellnitz. Gradient-Based Multiobjective Optimization with Uncertainties, pages 159–182. Springer International Publishing, 2018.
*   \[42\] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pages 487–503. Association for Computational Linguistics, 2021.
*   \[43\] Fabrice Poirion, Quentin Mercier, and Jean-Antoine Désidéri. Descent algorithm for nonsmooth stochastic multiobjective optimization. Computational Optimization and Applications, (2):317–331, 2017.
*   \[44\] Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.
*   \[45\] Stefan Schäffler, Richard R. Schultz, and Konstanze Weinzierl. Stochastic Method for the Solution of Unconstrained Vector Optimization Problems. Journal of Optimization Theory and Applications, 114:209–222, 2002.
*   \[46\] Peter Schönemann. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1–10, 1966.
*   \[47\] Michael L. Seltzer and Jasha Droppo. Multi-task learning in deep neural networks for improved phoneme recognition. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6965–6969. IEEE, 2013.
*   \[48\] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, pages 527–538. Curran Associates, Inc., 2018.
*   \[49\] Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-task reinforcement learning with context-based representations. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9767–9779. PMLR, 18–24 Jul 2021.
*   \[50\] Yee Teh, Victor Bapst, Wojciech M. Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In Advances in Neural Information Processing Systems (NIPS), volume 30, page 4499–4509. Curran Associates, Inc., 2017.
*   \[51\] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In ICLR, 2021.
*   \[52\] Markus Wulfmeier, Abbas Abdolmaleki, Roland Hafner, Jost Tobias Springenberg, Michael Neunert, Noah Siegel, Tim Hertweck, Thomas Lampe, Nicolas Heess, and Martin Riedmiller. Compositional transfer in hierarchical reinforcement learning. In Proceedings of Robotics: Science and Systems, 2020.
*   \[53\] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA, 2020. Curran Associates Inc.
*   \[54\] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 5824–5836. Curran Associates, Inc., 2020.
*   \[55\] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning (CoRL), 2019.
*   \[56\] Amir R. Zamir, Alexander Sax, William Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling Task Transfer Learning. In CVPR, pages 3712–3722, 2018.
*   \[57\] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In CVPR, pages 6230–6239, 2017.

## Appendix A Convergence Analysis

#### Synopsis.

In these theorems, we prove that the worst case performance of Aligned-MTL and Aligned-MTL-UB approaches is no worse than of standard gradient descent. The constraints mentioned in convergence theorems below are mild enough to be satisfied in practice. Our approach converges to a Pareto-stationary point with pre-defined tasks weights, thus providing more control over an optimization result.

###### Lemma 1.

Assume ℒ⁡(𝛉){\\mathcal{L}}({\\bm{\\theta}}) to be continuously differentiable and ∇ℒ​(𝛉)\\nabla{\\mathcal{L}}({\\bm{\\theta}}) to be Lipschitz continuous with Λ\>0\\Lambda>0. Then, the following restriction holds for a gradient descent with a step size α\\alpha and an update rule 𝐫{\\bm{r}}:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α⁡⟨∇ℒ​(𝜽t),𝒓⟩−α2​Λ2​‖𝒓‖2.{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\alpha\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{r}}\\hskip 1.42262pt\\rangle-\\frac{\\alpha^{2}\\Lambda}{2}\\|{\\bm{r}}\\|^{2}.

(8)

###### Proof.

Let us consider a gradient descent 𝛉t+1\=𝛉t+𝛅{\\bm{\\theta}}\_{t+1}={\\bm{\\theta}}\_{t}+{\\bm{\\delta}}, where 𝛅\=−α​𝐫{\\bm{\\delta}}=-\\alpha{\\bm{r}}. From the fundamental theorem of calculus, we derive:

ℒ⁡(𝜽t+𝜹)−ℒ⁡(𝜽t)\=∫01⟨∇ℒ​(𝜽t+s​𝜹),𝜹⟩​𝑑s.{\\mathcal{L}}({\\bm{\\theta}}\_{t}+{\\bm{\\delta}})-{\\mathcal{L}}({\\bm{\\theta}}\_{t})=\\int\_{0}^{1}\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}+s{\\bm{\\delta}}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\,\\mathrm{d}s.

(9)

By adding and subtracting the value ⟨∇ℒ​(𝛉t),δ⟩\=∫01⟨∇ℒ​(𝛉t),δ⟩​𝑑s\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),\\delta\\hskip 1.42262pt\\rangle=\\int\_{0}^{1}\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),\\delta\\hskip 1.42262pt\\rangle\\,\\mathrm{d}s, we obtain:

ℒ⁡(𝜽t+1)−ℒ⁡(𝜽t)\=⟨∇ℒ​(𝜽t),𝜹⟩+\\displaystyle{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})-{\\mathcal{L}}({\\bm{\\theta}}\_{t})=\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\ +

(10)

+∫01⟨∇ℒ(𝜽t+s𝜹)−∇ℒ(𝜽t),𝜹⟩ds.\\displaystyle+\\int\_{0}^{1}\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}+s{\\bm{\\delta}})-\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\,\\mathrm{d}s.

(11)

Since the gradient satisfies the Lipschitz condition ‖∇ℒ​(𝛉t+s​𝛅)−∇ℒ​(𝛉t)‖≤Λ​‖𝛉t+s​𝛅−𝛉t‖\\|\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}+s{\\bm{\\delta}})-\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t})\\|\\leq\\Lambda\\|{\\bm{\\theta}}\_{t}+s{\\bm{\\delta}}-{\\bm{\\theta}}\_{t}\\| and due to inequality ⟨x,y⟩≤‖x‖​‖y‖\\langle\\hskip 1.42262ptx,y\\hskip 1.42262pt\\rangle\\leq\\|x\\|\\|y\\|, we can transform the integral as following:

ℒ⁡(𝜽t+1)−ℒ⁡(𝜽t)\=⟨∇ℒ​(𝜽t),𝜹⟩+\\displaystyle{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})-{\\mathcal{L}}({\\bm{\\theta}}\_{t})=\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\ +

+∫01⟨∇ℒ(𝜽t+s𝜹)−∇ℒ(𝜽t),𝜹⟩ds≤\\displaystyle+\\int\_{0}^{1}\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}+s{\\bm{\\delta}})-\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\,\\mathrm{d}s\\leq

⟨∇ℒ​(𝜽t),𝜹⟩+∫01Λ​‖𝜽t+s​𝜹−𝜽t‖​‖𝜹‖​𝑑s≤\\displaystyle\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{\\delta}}\\hskip 1.42262pt\\rangle\\ +\\ \\int\_{0}^{1}\\Lambda\\|{\\bm{\\theta}}\_{t}+s{\\bm{\\delta}}-{\\bm{\\theta}}\_{t}\\|\\|{\\bm{\\delta}}\\|\\mathrm{d}s\\leq

−α⁡⟨∇ℒ​(𝜽t),𝒓⟩+Λ​∫01‖−s​α​𝒓‖2​‖−α​𝒓‖2​𝑑s≤\\displaystyle-\\alpha\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{r}}\\hskip 1.42262pt\\rangle\\ +\\ \\Lambda\\int\_{0}^{1}\\|-s\\alpha{\\bm{r}}\\|\_{2}\\|-\\alpha{\\bm{r}}\\|\_{2}\\mathrm{d}s\\leq

−α⁡⟨∇ℒ​(𝜽t),𝒓⟩+α2​Λ|𝒓|∫012⁡s​𝑑s≤\\displaystyle-\\alpha\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{r}}\\hskip 1.42262pt\\rangle\\ +\\ \\alpha^{2}\\Lambda\\|{\\bm{r}}\\|^{2}\\int\_{0}^{1}s\\mathrm{d}s\\leq

−α⁡⟨∇ℒ​(𝜽t),𝒓⟩+α2​Λ​‖𝒓‖2\\displaystyle-\\alpha\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{r}}\\hskip 1.42262pt\\rangle+\\alpha^{2}\\Lambda\\|{\\bm{r}}\\|^{2}

Therefore, we obtain the final constraint:

ℒ⁡(𝜽t+1)−ℒ⁡(𝜽t)≤−α⁡⟨∇ℒ​(𝜽t),𝒓⟩+α2​Λ​‖𝒓‖2.{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})-{\\mathcal{L}}({\\bm{\\theta}}\_{t})\\leq-\\alpha\\langle\\hskip 1.42262pt\\nabla{\\mathcal{L}}({\\bm{\\theta}}\_{t}),{\\bm{r}}\\hskip 1.42262pt\\rangle+\\alpha^{2}\\Lambda\\|{\\bm{r}}\\|^{2}.

(12)

###### Theorem 2 (Aligned-MTL).

Assume ℒ0​(𝛉),…,ℒT​(𝛉){\\mathcal{L}}\_{0}({\\bm{\\theta}}),\\dots,{\\mathcal{L}}\_{T}({\\bm{\\theta}}) are lower-bounded continuously differentiable functions with Lipschitz continuous gradients with Λ\>0\\Lambda>0. A gradient descent with an aligned gradient and a step size α≤1Λ\\alpha\\leq\\frac{1}{\\Lambda} converges linearly to a Pareto-stationary point where ∇ℒ0​(𝛉)\=0\\nabla{\\mathcal{L}}\_{0}({\\bm{\\theta}})=0. 

###### Proof (Aligned-MTL).

Given the aforementioned assumptions, the cumulative objective satisfies [Lemma 1](#Thmlemma1 "Lemma 1. ‣ Synopsis. ‣ Appendix A Convergence Analysis ‣ Independent Component Alignment for Multi-Task Learning") with 𝐫\=𝐆^​𝐰\=g^0{\\bm{r}}=\\hat{{\\bm{G}}}{\\bm{w}}=\\hat{g}\_{0} and ∇ℒ0​(𝛉)\=𝐆​𝐰\=𝐠0\\nabla{\\mathcal{L}}\_{0}({\\bm{\\theta}})={\\bm{G}}{\\bm{w}}={\\bm{g}}\_{0}:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α​𝒈0⊤​𝒈^0−α2​Λ2​‖𝒈0^‖2.{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\alpha{\\bm{g}}\_{0}^{\\top}\\hat{{\\bm{g}}}\_{0}-\\frac{\\alpha^{2}\\Lambda}{2}\\|\\hat{{\\bm{g}}\_{0}}\\|^{2}.

(13)

According to SVD, 𝐆\=𝐔​𝚺​𝐕⊤{\\bm{G}}={\\bm{U}}{\\bm{\\Sigma}}{\\bm{V}}^{\\top}, 𝚺\=diag⁡{σ1,…,σR}{\\bm{\\Sigma}}=\\diag\\{\\sigma\_{1},\\ldots,\\sigma\_{R}\\} where R\=rank⁡𝐆R=\\rank{\\bm{G}}, and 𝐔⊤​𝐔\=𝐈{\\bm{U}}^{\\top}{\\bm{U}}={\\bm{I}}. By definition of the Aligned-MTL, we get:

𝒈0⊤​𝒈^0\=σR​𝒘⊤​𝑽​𝚺​𝑼⊤​𝑼​𝑽⊤​𝒘\=\\displaystyle{\\bm{g}}\_{0}^{\\top}\\hat{{\\bm{g}}}\_{0}=\\sigma\_{R}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{\\Sigma}}{\\bm{U}}^{\\top}{\\bm{U}}{\\bm{V}}^{\\top}{\\bm{w}}=

\=σR​𝒘⊤​𝑽​𝚺​𝑽⊤​𝒘\=∑r\=1RσR​σr​(𝒘⊤​𝒗r)2\\displaystyle=\\sigma\_{R}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{\\Sigma}}{\\bm{V}}^{\\top}{\\bm{w}}=\\sum\_{r=1}^{R}\\sigma\_{R}\\sigma\_{r}({\\bm{w}}^{\\top}{\\bm{v}}\_{r})^{2}

Similarly, ‖𝐠^0‖2\=∑r\=1RσR2​(𝐰⊤​𝐯r)2\\|\\hat{{\\bm{g}}}\_{0}\\|^{2}=\\sum\_{r=1}^{R}\\sigma\_{R}^{2}({\\bm{w}}^{\\top}{\\bm{v}}\_{r})^{2}. Since α≤1Λ\\alpha\\leq\\frac{1}{\\Lambda} and 𝐰⊤​𝐯r\>ε{\\bm{w}}^{\\top}{\\bm{v}}\_{r}>\\varepsilon, [Eq. 13](#A1.E13 "In Proof (Aligned-MTL). ‣ Synopsis. ‣ Appendix A Convergence Analysis ‣ Independent Component Alignment for Multi-Task Learning") can be further bounded:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥σR2​α2​∑r\=1R(2​σrσR−1)⏟\>1​(𝒘⊤​𝒗r)2⏟\>‖𝑽​𝒘‖2\>ε2\>\\displaystyle{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\sigma\_{R}^{2}\\frac{\\alpha}{2}\\underbrace{\\sum\_{r=1}^{R}\\underbrace{\\bigg(2\\frac{\\sigma\_{r}}{\\sigma\_{R}}-1\\bigg)}\_{>1}\\bigg({\\bm{w}}^{\\top}{\\bm{v}}\_{r}\\bigg)^{2}}\_{>\\|{\\bm{V}}{\\bm{w}}\\|^{2}>\\varepsilon^{2}}>

\>α​σR22​ε2σ12​σ12.\\displaystyle>\\frac{\\alpha\\sigma\_{R}^{2}}{2}\\frac{\\varepsilon^{2}}{\\sigma\_{1}^{2}}\\sigma\_{1}^{2}.

The dominance is always finite: σRσ1\>C\\frac{\\sigma\_{R}}{\\sigma\_{1}}>C. Moreover, σ1\=max𝐱≠0⁡‖𝐆​𝐱‖‖𝐱‖\\sigma\_{1}=\\max\_{{\\bm{x}}\\neq 0}\\frac{\\|{\\bm{G}}{\\bm{x}}\\|}{\\|{\\bm{x}}\\|}, therefore σ1≥‖𝐠0‖‖𝐰‖\\sigma\_{1}\\geq\\frac{\\|{\\bm{g}}\_{0}\\|}{\\|{\\bm{w}}\\|}. Respectively:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)\>α​ε2​C22​‖𝒘‖2​‖𝒈0‖2.{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})>\\frac{\\alpha\\varepsilon^{2}C^{2}}{2\\|{\\bm{w}}\\|^{2}}\\|{\\bm{g}}\_{0}\\|^{2}.

(14)

The sequence of ℒ⁡(𝛉t){\\mathcal{L}}({\\bm{\\theta}}\_{t}) is monotonically decreasing and bounded (under assumption), and hence converging. Then ℒ⁡(𝛉t)−ℒ⁡(𝛉t+1)→0{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\rightarrow 0 if t→∞t\\rightarrow\\infty. Thereby, we have a local convergence of the gradient descent:

‖𝒈0‖2<2​‖𝒘‖2α​C2​ϵ2​(ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1))→0ast→∞.\\|{\\bm{g}}\_{0}\\|^{2}<\\frac{2\\|{\\bm{w}}\\|^{2}}{\\alpha C^{2}\\epsilon^{2}}\\bigg({\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\bigg)\\rightarrow 0\\quad\\text{as}\\quad t\\rightarrow\\infty.

(15)

The same estimate appears in case of the gradient descent. Accordingly, the convergence of Aligned-MTL is similar to that of the gradient descent, *i.e*. linear – 𝒪⁡(1T)\\mathcal{O}(\\frac{1}{T}). 

###### Theorem 3 (A-MTL-UB).

Assume ℒ0​(𝛉),…,ℒT​(𝛉){\\mathcal{L}}\_{0}({\\bm{\\theta}}),\\dots,{\\mathcal{L}}\_{T}({\\bm{\\theta}}) are lower-bounded continuously differentiable functions with Lipschitz continuous gradients with Λ\>0\\Lambda>0. Suppose 𝐉\=∂𝐇∂𝛉{\\bm{J}}=\\frac{\\partial{\\bm{H}}}{\\partial{\\bm{\\theta}}} to be a full rank, *i.e*. rank⁡𝐉\=min⁡{|𝛉|,|𝐇|}\\rank{\\bm{J}}=\\min\\{|{\\bm{\\theta}}|,|{\\bm{H}}|\\}. A gradient descent with an aligned gradient and a step size α≤1Λ\\alpha\\leq\\frac{1}{\\Lambda} converges linearly to a Pareto-stationary point where ∇ℒ0​(𝛉)\=0\\nabla{\\mathcal{L}}\_{0}({\\bm{\\theta}})=0. 

###### Proof (Aligned-MTL-UB).

Similarly to the [Theorem 2](#Thmtheorem2 "Theorem 2 (Aligned-MTL). ‣ Synopsis. ‣ Appendix A Convergence Analysis ‣ Independent Component Alignment for Multi-Task Learning"), under the aforementioned assumptions, the cumulative objective satisfies [Lemma 1](#Thmlemma1 "Lemma 1. ‣ Synopsis. ‣ Appendix A Convergence Analysis ‣ Independent Component Alignment for Multi-Task Learning") with 𝐫\=σR​𝐉​𝐙^​𝐰\=𝐠^0{\\bm{r}}=\\sigma\_{R}{\\bm{J}}\\hat{{\\bm{Z}}}{\\bm{w}}=\\hat{{\\bm{g}}}\_{0} and ∇ℒ0​(𝛉)\=𝐉​𝐙​𝐰\=𝐠0\\nabla{\\mathcal{L}}\_{0}({\\bm{\\theta}})={\\bm{J}}{\\bm{Z}}{\\bm{w}}={\\bm{g}}\_{0}:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α​𝒈0⊤​𝒈^0−α2​Λ2​‖𝒈0^‖2.{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\alpha{\\bm{g}}\_{0}^{\\top}\\hat{{\\bm{g}}}\_{0}-\\frac{\\alpha^{2}\\Lambda}{2}\\|\\hat{{\\bm{g}}\_{0}}\\|^{2}.

(16)

According to SVD, 𝐙\=𝐔​𝚺​𝐕⊤{\\bm{Z}}={\\bm{U}}{\\bm{\\Sigma}}{\\bm{V}}^{\\top}, 𝚺\=diag⁡{σ1,…,σR}{\\bm{\\Sigma}}=\\diag\\{\\sigma\_{1},\\ldots,\\sigma\_{R}\\} where R\=rank⁡𝐙R=\\rank{\\bm{Z}}, and 𝐔⊤​𝐔\=𝐈{\\bm{U}}^{\\top}{\\bm{U}}={\\bm{I}}. By definition of the Aligned-MTL-UB, we get:

𝒈0⊤​𝒈^0\=σR​𝒘⊤​𝑽​𝚺​𝑼⊤​𝑱⊤​𝑱​𝑼​𝑽⊤​𝒘\\displaystyle{\\bm{g}}\_{0}^{\\top}\\hat{{\\bm{g}}}\_{0}=\\sigma\_{R}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{\\Sigma}}{\\bm{U}}^{\\top}{\\bm{J}}^{\\top}{\\bm{J}}{\\bm{U}}{\\bm{V}}^{\\top}{\\bm{w}}

𝒈^0⊤​𝒈^0\=σR2​𝒘⊤​𝑽​𝑼⊤​𝑱⊤​𝑱​𝑼​𝑽⊤​𝒘\\displaystyle\\hat{{\\bm{g}}}\_{0}^{\\top}\\hat{{\\bm{g}}}\_{0}=\\sigma\_{R}^{2}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{U}}^{\\top}{\\bm{J}}^{\\top}{\\bm{J}}{\\bm{U}}{\\bm{V}}^{\\top}{\\bm{w}}

Since 𝐉{\\bm{J}} is full rank, 𝐉⊤​𝐉{\\bm{J}}^{\\top}{\\bm{J}} is positive definite. Any positive definite matrix is congruent to a diagonal (𝐃{\\bm{D}}) with positive and ordered eigenvalues on the main diagonal. Thus, replacing all eigenvalues λi2\\lambda\_{i}^{2} with the smallest one λK2\\lambda\_{K}^{2} does not increase the inner product produced by this matrix: 𝐱​𝐃​𝐱≥λK​𝐱⊤​𝐱{\\bm{x}}{\\bm{D}}{\\bm{x}}\\geq\\lambda\_{K}{\\bm{x}}^{\\top}{\\bm{x}}. By taking this into consideration, we can bound the right side of [Eq. 16](#A1.E16 "In Proof (Aligned-MTL-UB). ‣ Synopsis. ‣ Appendix A Convergence Analysis ‣ Independent Component Alignment for Multi-Task Learning"):

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α2​(2​𝒈0−𝒈^0)⊤​𝒈^0≥\\displaystyle{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\frac{\\alpha}{2}(2{\\bm{g}}\_{0}-\\hat{{\\bm{g}}}\_{0})^{\\top}\\hat{{\\bm{g}}}\_{0}\\geq

α2​(2​σR​𝒘⊤​𝑽​𝚺​𝑼⊤−σR2​𝒘⊤​𝑽​𝑼⊤)​𝑱⊤​𝑱​𝑼​𝑽⊤​𝒘≥\\displaystyle\\frac{\\alpha}{2}(2\\sigma\_{R}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{\\Sigma}}{\\bm{U}}^{\\top}-\\sigma\_{R}^{2}{\\bm{w}}^{\\top}{\\bm{V}}{\\bm{U}}^{\\top}){\\bm{J}}^{\\top}{\\bm{J}}{\\bm{U}}{\\bm{V}}^{\\top}{\\bm{w}}\\geq

σR2​λK2​∑r\=1R(2​σrσR−1)⏟\>1​(𝒘⊤​𝒗r)2⏟\>‖𝑽​𝒘‖2\>ε2\\displaystyle\\sigma\_{R}^{2}\\lambda\_{K}^{2}\\underbrace{\\sum\_{r=1}^{R}\\underbrace{\\bigg(2\\frac{\\sigma\_{r}}{\\sigma\_{R}}-1\\bigg)}\_{>1}\\bigg({\\bm{w}}^{\\top}{\\bm{v}}\_{r}\\bigg)^{2}}\_{>\\|{\\bm{V}}{\\bm{w}}\\|^{2}>\\varepsilon^{2}}

Thus:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α​ε2​σR2​λK22​σ12​λ12​σ12​λ12.{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\frac{\\alpha\\varepsilon^{2}\\sigma\_{R}^{2}\\lambda\_{K}^{2}}{2\\sigma\_{1}^{2}\\lambda\_{1}^{2}}\\sigma\_{1}^{2}\\lambda\_{1}^{2}.

(17)

Following the assumption, σRσ1\>Cσ\\frac{\\sigma\_{R}}{\\sigma\_{1}}>C\_{\\sigma} and λKλ1\>Cλ\\frac{\\lambda\_{K}}{\\lambda\_{1}}>C\_{\\lambda}. Moreover, σ1\=max𝐱≠0⁡‖𝐙​𝐱‖‖𝐱‖≥‖𝐙​𝐰‖‖𝐰‖\\sigma\_{1}=\\max\_{{\\bm{x}}\\neq 0}\\frac{\\|{\\bm{Z}}{\\bm{x}}\\|}{\\|{\\bm{x}}\\|}\\geq\\frac{\\|{\\bm{Z}}{\\bm{w}}\\|}{\\|{\\bm{w}}\\|} and λ1\=‖𝐉‖\\lambda\_{1}=\\|{\\bm{J}}\\|. Therefore, we obtain the final bound:

ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1)≥α​ε2​Cσ2​Cλ22​‖𝒘‖2​‖𝑮𝒁​𝒘‖2​‖𝑱‖2≥\\displaystyle{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\geq\\frac{\\alpha\\varepsilon^{2}C\_{\\sigma}^{2}C\_{\\lambda}^{2}}{2\\|{\\bm{w}}\\|^{2}}\\|{\\bm{G}}\_{{\\bm{Z}}}{\\bm{w}}\\|^{2}\\|{\\bm{J}}\\|^{2}\\geq

α​ε2​Cσ2​Cλ22​‖𝒘‖2​‖𝒈0‖2.\\displaystyle\\frac{\\alpha\\varepsilon^{2}C\_{\\sigma}^{2}C\_{\\lambda}^{2}}{2\\|{\\bm{w}}\\|^{2}}\\|{\\bm{g}}\_{0}\\|^{2}.

The sequence of ℒ⁡(𝛉t){\\mathcal{L}}({\\bm{\\theta}}\_{t}) is monotonically decreasing and bounded (under assumption), and hence converging. Then ℒ⁡(𝛉t)−ℒ⁡(𝛉t+1)→0{\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\rightarrow 0 if t→∞t\\rightarrow\\infty. Thereby, we have a local convergence of the gradient descent:

‖𝒈0‖2<2​‖𝒘‖2α​Cσ2​Cλ2​ε2​(ℒ⁡(𝜽t)−ℒ⁡(𝜽t+1))→0ast→∞.\\|{\\bm{g}}\_{0}\\|^{2}<\\frac{2\\|{\\bm{w}}\\|^{2}}{\\alpha C\_{\\sigma}^{2}C\_{\\lambda}^{2}\\varepsilon^{2}}\\bigg({\\mathcal{L}}({\\bm{\\theta}}\_{t})-{\\mathcal{L}}({\\bm{\\theta}}\_{t+1})\\bigg)\\to 0\\quad\\text{as}\\quad t\\to\\infty.

(18)

## Appendix B Condition Number

The stability criterion is closely related to the dominance and conflicts. We can find a functional dependence between them for some special cases: a) gradients 𝒈1{\\bm{g}}\_{1} and 𝒈2{\\bm{g}}\_{2} have equal magnitude but not orthogonal, b) they are othogonal but have different norms. To this end, we formulate the following colloraries.

###### Collorary 1.

Given 𝐠1⟂𝐠2{\\bm{g}}\_{1}\\perp{\\bm{g}}\_{2} condition number κ\\kappa is

κ\=max⁡{‖𝒈1‖‖𝒈2‖,‖𝒈2‖‖𝒈1‖}\\kappa=\\max\\bigg\\{\\frac{\\|{\\bm{g}}\_{1}\\|}{\\|{\\bm{g}}\_{2}\\|},\\frac{\\|{\\bm{g}}\_{2}\\|}{\\|{\\bm{g}}\_{1}\\|}\\bigg\\}

###### Proof.

By initial assumtions the Gram matrix 𝐆⊤​𝐆{\\bm{G}}^{\\top}{\\bm{G}} is diagonal:

𝑮⊤​𝑮\=diag⁡{‖𝒈1‖2,‖𝒈2‖2}{\\bm{G}}^{\\top}{\\bm{G}}=\\diag\\{\\|{\\bm{g}}\_{1}\\|^{2},\\|{\\bm{g}}\_{2}\\|^{2}\\}

At the same time, this matrix can be factorized using eigen decomposition:

𝑮⊤​𝑮\=𝑽​𝚺2​𝑽⊤,𝑽​𝑽⊤\=𝑰,𝚺\=diag⁡{σ1,σ2}{\\bm{G}}^{\\top}{\\bm{G}}={\\bm{V}}{\\bm{\\Sigma}}^{2}{\\bm{V}}^{\\top},\\quad{\\bm{V}}{\\bm{V}}^{\\top}={\\bm{I}},\\quad{\\bm{\\Sigma}}=\\diag\\{\\sigma\_{1},\\sigma\_{2}\\}

Thus, the singular values are proportional to the gradient magnitudes up to a symmetric swap to keep ordering of singular values. The coefficient of proportionality is not valuable, since the condition number is invariant to the global scale. Therefore, we derive:

κ\=max⁡{‖𝒈1‖‖𝒈2‖,‖𝒈2‖‖𝒈1‖}\\kappa=\\max\\bigg\\{\\frac{\\|{\\bm{g}}\_{1}\\|}{\\|{\\bm{g}}\_{2}\\|},\\frac{\\|{\\bm{g}}\_{2}\\|}{\\|{\\bm{g}}\_{1}\\|}\\bigg\\}

Figure 5: The condition number depends on the angle between gradient vectors. Due to the symmetry one of the principal components is a bisectrix of this angle.

###### Collorary 2.

Given 𝐠1{\\bm{g}}\_{1} and 𝐠2{\\bm{g}}\_{2} with equal magnitudes, *i.e*. ‖𝐠1‖\=‖𝐠2‖\\|{\\bm{g}}\_{1}\\|=\\|{\\bm{g}}\_{2}\\|, and with α\\alpha angle in between the condition number κ\\kappa is

κ\={tan⁡(α/2)π4<α/2≤π2c​t​a​n⁡(α/2)0<α/2<π4\\kappa=\\begin{cases}\\tan(\\alpha/2)&\\frac{\\pi}{4}<\\alpha/2\\leq\\frac{\\pi}{2}\\\\ \\mathop{ctan}\\nolimits(\\alpha/2)&0<\\alpha/2<\\frac{\\pi}{4}\\\\ \\end{cases}

(19)

###### Proof.

The direct collorary of SVD states, that the principal components 𝐮i{\\bm{u}}\_{i} are direction with maximum norm of projections over all gradients. Formally:

σ1\=max‖𝒙‖\=1⁡‖𝑮⊤​𝒙‖\=‖𝑮⊤​𝒖1‖\\displaystyle\\sigma\_{1}=\\max\_{\\|{\\bm{x}}\\|=1}\\|{\\bm{G}}^{\\top}{\\bm{x}}\\|=\\|{\\bm{G}}^{\\top}{\\bm{u}}\_{1}\\|

σ2\=max‖𝒙‖\=1,𝒙⟂𝒖1⁡‖𝑮⊤​𝒙‖\=‖𝑮⊤​𝒖2‖\\displaystyle\\sigma\_{2}=\\max\_{\\|{\\bm{x}}\\|=1,{\\bm{x}}\\perp{\\bm{u}}\_{1}}\\|{\\bm{G}}^{\\top}{\\bm{x}}\\|=\\|{\\bm{G}}^{\\top}{\\bm{u}}\_{2}\\|

Since the gradients have the same length, one of the principal components is the bisectrix of angle between them. For clarity, we suppose, that the bisectrix is the second component. Then, the singular values can be computed trivially ([Fig. 5](#A2.F5 "In Appendix B Condition Number ‣ Independent Component Alignment for Multi-Task Learning")):

σ1\=2​sin⁡(α/2)​‖𝒈1‖\\displaystyle\\sigma\_{1}=\\sqrt{2}\\sin(\\alpha/2)\\|{\\bm{g}}\_{1}\\|

σ2\=2​cos⁡(α/2)​‖𝒈1‖\\displaystyle\\sigma\_{2}=\\sqrt{2}\\cos(\\alpha/2)\\|{\\bm{g}}\_{1}\\|

Accroding to these expressions the condition number is tangent or cotangent up to a symmetric swap to keep ordering of singular values. In orthoginal case, the condition number is unit. 

## Appendix C Synthetic Example

The synthetic example is a two-task objective containing areas with the presence of conflicting and dominating gradients between loss components. Formally, we use the same objective as in previous works\[[26](#bib.bib26 ""), [37](#bib.bib37 "")\]:

ℒ1\=c1​(𝜽)​f1​(𝜽)+c2​(𝜽)​g1​(𝜽)\\displaystyle{\\mathcal{L}}\_{1}=c\_{1}({\\bm{\\theta}})f\_{1}({\\bm{\\theta}})+c\_{2}({\\bm{\\theta}})g\_{1}({\\bm{\\theta}})

ℒ2\=c1​(𝜽)​f2​(𝜽)+c2​(𝜽)​g2​(𝜽)\\displaystyle{\\mathcal{L}}\_{2}=c\_{1}({\\bm{\\theta}})f\_{2}({\\bm{\\theta}})+c\_{2}({\\bm{\\theta}})g\_{2}({\\bm{\\theta}})

𝜽∈ℝ2\\displaystyle{\\bm{\\theta}}\\in\\mathbb{R}^{2}

where

h1​(𝜽)\=|(−θ1−7)2−tanh⁡(−θ2)|\\displaystyle h\_{1}({\\bm{\\theta}})=\\left|\\frac{(-\\theta\_{1}-7)}{2}-\\tanh{(-\\theta\_{2})}\\right|

h2​(𝜽)\=|(−θ1+3)2−tanh⁡(−θ2)+2|\\displaystyle h\_{2}({\\bm{\\theta}})=\\bigg|\\frac{(-\\theta\_{1}+3)}{2}-\\tanh{(-\\theta\_{2})+2}\\bigg|

c1​(θ)\=max⁡(tanh⁡(θ22),0)\\displaystyle c\_{1}(\\theta)=\\max(\\tanh\\bigg(\\frac{\\theta\_{2}}{2}\\bigg),0)

c2​(θ)\=max⁡(tanh⁡(−θ22),0)\\displaystyle c\_{2}(\\theta)=\\max(\\tanh\\bigg(\\frac{-\\theta\_{2}}{2}\\bigg),0)

f1​(𝜽)\=log⁡max⁡(h1​(𝜽),5⋅10−6)+6\\displaystyle f\_{1}({\\bm{\\theta}})=\\log\\max\\left(h\_{1}({\\bm{\\theta}}),5\\cdot 10^{-6}\\right)+6

f2​(𝜽)\=log⁡max⁡(h2​(𝜽),5⋅10−6)+6\\displaystyle f\_{2}({\\bm{\\theta}})=\\log\\max\\left(h\_{2}({\\bm{\\theta}}),5\\cdot 10^{-6}\\right)+6

g1​(𝜽)\=(−θ−7)2+0.1​(−θ2−8)210−20\\displaystyle g\_{1}({\\bm{\\theta}})=\\frac{(-\\theta-7)^{2}+0.1(-\\theta\_{2}-8)^{2}}{10}-20

g2​(𝜽)\=(−θ+7)2+0.1​(−θ2−8)210−20\\displaystyle g\_{2}({\\bm{\\theta}})=\\frac{(-\\theta+7)^{2}+0.1(-\\theta\_{2}-8)^{2}}{10}-20

We perform minimization starting from five initial points: \[−8.5,7.5\],\[0.0,0.0\],\[9.0,9.0\],\[−7.5,−0.5\],\[9,−1.0\]\[-8.5,7.5\],\[0.0,0.0\],\[9.0,9.0\],\[-7.5,-0.5\],\[9,-1.0\]. We use Adam \[[18](#bib.bib18 "")\] optimizer with learning rate 10−310^{-3} and optimize for 35k iterations. We demonstrate that our method is able to converge to the optimums with varying pre-defined task weights in [Fig. 6](#A3.F6 "In Appendix C Synthetic Example ‣ Independent Component Alignment for Multi-Task Learning"). For this purpose we explore a number of task convex combinations, such that ℒ0\=α​ℒ1+(1−α)​ℒ2\\mathcal{L}\_{0}=\\alpha\\mathcal{L}\_{1}+(1-\\alpha)\\mathcal{L}\_{2}

Figure 6: Comparison of MTL optimization methods on synthetic two-task benchmark \[[26](#bib.bib26 ""), [37](#bib.bib37 "")\]. We explore convergence of various methods with varying pre-defined task weights. Methods that guarantee only Pareto-front convergence (such as IMTL\[[27](#bib.bib27 "")\] and NashMTL \[[37](#bib.bib37 "")\]) fail to achieve global optimum (defined by ★\\bigstar) and converge to an arbitrary Pareto-front solution with unknown task balance. Unlike previous methods, our Aligned-MTL approach respects pre-defined task weights and converges to the global optimum for all task weights combinations and initialization points (∙\\bullet), except one extreme case. Moreover, our method provides stable and less noisy trajectories than other methods.

|                                | 0.1​ℒ1+0.9​ℒ20.1\\mathcal{L}\_{1}+0.9\\mathcal{L}\_{2} | 0.3​ℒ1+0.7​ℒ20.3\\mathcal{L}\_{1}+0.7\\mathcal{L}\_{2} | 0.5​ℒ1+0.5​ℒ20.5\\mathcal{L}\_{1}+0.5\\mathcal{L}\_{2} | 0.7​ℒ1+0.3​ℒ20.7\\mathcal{L}\_{1}+0.3\\mathcal{L}\_{2} | 0.9​ℒ1+0.1​ℒ20.9\\mathcal{L}\_{1}+0.1\\mathcal{L}\_{2} |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ |
| Uniform                        | ![Refer to caption](2305.19000v1/traj_ls_0.1.png)      | ![Refer to caption](2305.19000v1/traj_ls_0.3.png)      | ![Refer to caption](2305.19000v1/traj_ls_0.5.png)      | ![Refer to caption](2305.19000v1/traj_ls_0.7.png)      | ![Refer to caption](2305.19000v1/traj_ls_0.9.png)      |
| PCGrad\[[54](#bib.bib54 "")\]  | ![Refer to caption](2305.19000v1/traj_pcgrad_0.1.png)  | ![Refer to caption](2305.19000v1/traj_pcgrad_0.3.png)  | ![Refer to caption](2305.19000v1/traj_pcgrad_0.5.png)  | ![Refer to caption](2305.19000v1/traj_pcgrad_0.7.png)  | ![Refer to caption](2305.19000v1/traj_pcgrad_0.9.png)  |
| CAGrad\[[26](#bib.bib26 "")\]  | ![Refer to caption](2305.19000v1/traj_cagrad_0.1.png)  | ![Refer to caption](2305.19000v1/traj_cagrad_0.3.png)  | ![Refer to caption](2305.19000v1/traj_cagrad_0.5.png)  | ![Refer to caption](2305.19000v1/traj_cagrad_0.7.png)  | ![Refer to caption](2305.19000v1/traj_cagrad_0.9.png)  |
| IMTL\[[27](#bib.bib27 "")\]    | ![Refer to caption](2305.19000v1/traj_imtl_0.1.png)    | ![Refer to caption](2305.19000v1/traj_imtl_0.3.png)    | ![Refer to caption](2305.19000v1/traj_imtl_0.5.png)    | ![Refer to caption](2305.19000v1/traj_imtl_0.7.png)    | ![Refer to caption](2305.19000v1/traj_imtl_0.9.png)    |
| NashMTL\[[37](#bib.bib37 "")\] | ![Refer to caption](2305.19000v1/traj_nashmtl_0.1.png) | ![Refer to caption](2305.19000v1/traj_nashmtl_0.3.png) | ![Refer to caption](2305.19000v1/traj_nashmtl_0.5.png) | ![Refer to caption](2305.19000v1/traj_nashmtl_0.7.png) | ![Refer to caption](2305.19000v1/traj_nashmtl_0.9.png) |
| Ours, Align-MTL                | ![Refer to caption](2305.19000v1/traj_amgdaw_0.1.png)  | ![Refer to caption](2305.19000v1/traj_amgdaw_0.3.png)  | ![Refer to caption](2305.19000v1/traj_amgdaw_0.5.png)  | ![Refer to caption](2305.19000v1/traj_amgdaw_0.7.png)  | ![Refer to caption](2305.19000v1/traj_amgdaw_0.9.png)  |
|                                |                                                        |                                                        |                                                        |                                                        |                                                        |
|                                |                                                        |                                                        |                                                        |                                                        |                                                        |
|                                |                                                        |                                                        |                                                        |                                                        |                                                        |
|                                |                                                        |                                                        |                                                        |                                                        |                                                        |

## Appendix D Implementation details

CityScapes three-task. Following MGDA-UB training setup \[[48](#bib.bib48 "")\], we train PSPNet\[[57](#bib.bib57 "")\] model for 100100 epochs using Adam optimizer with learning rate 10−410^{-4}. Train batch size is set to 88. Images from training set are resized into 512×256512\\times 256 resolution. We augment training set using random rotation and horizontal flips. The performance is averaged across 33 random initializations.

CityScapes two-task. We follow CAGrad \[[26](#bib.bib26 "")\] training setup and train MTAN \[[28](#bib.bib28 "")\] model. Semantic labels are groupped into 77 classes. Batch size is set to 88, learning rate of Adam optimizer is set to 10−410^{-4}. Models are trained for 200200 epochs and learning rate is halved after 100100 epochs. The performance is averaged over last 1010 epochs and 33 random seeds.

Figure 7: Empirical evaluation of a stability criterion. We plot a condition number, gradient magnitude similarity \[[54](#bib.bib54 "")\], minimal cosine between gradient pairs (conflicts) and maximum gradient norm ratio, *i.e*. maxi≠j⁡{‖𝒈i‖/‖𝒈j‖}\\max\_{i\\neq j}\\{\\|{\\bm{g}}\_{i}\\|/\\|{\\bm{g}}\_{j}\\|\\}, during training of PSPNet \[[57](#bib.bib57 ""), [48](#bib.bib48 "")\] and MTAN \[[28](#bib.bib28 "")\] on the NYUv2 benchmark. Unlike Cityscapes with three tasks (figure from the main paper), on NYUv2 gradients do not differ drastically in magnitudes but tend to have more conflicts (the cosine between gradients are negative, except for PCGrad). These figures indicate a high correlation between condition number, gradient norm ratios and gradient magnitude similarity. Our Aligned-MTL approach eliminates dominance (κ\=1\\kappa=1, r\=1r=1, G​M​S\=1GMS=1) and conflicts (mini≠j⁡c​o​s​(gi,gj)\=0\\min\_{i\\neq j}cos(g\_{i},g\_{j})=0) by design.

![Refer to caption](2305.19000v1/sup_nyu_plot.png) 

NYUv2 three-task. \[[28](#bib.bib28 ""), [26](#bib.bib26 ""), [37](#bib.bib37 "")\] We train both PSPNet models \[[57](#bib.bib57 ""), [48](#bib.bib48 "")\] and MTAN \[[28](#bib.bib28 "")\] models in our training setup with the same hyperparameters set. We use Adam \[[18](#bib.bib18 "")\] optimizer with learning rate 10−410^{-4}. Models are trained for 200200 epochs and batch size 22. Images from training set are randomly scaled and cropped into 384×288384\\times 288 resolution. The performance is averaged across 33 random seeds.

Reinforcement learning. We follow CAGrad \[[26](#bib.bib26 "")\] and use the implementation originally proposed and developed by \[[49](#bib.bib49 "")\]. The execution config was adapded from CAGrad \[[26](#bib.bib26 "")\]. The global evaluation pipeline is similar to previous works \[[37](#bib.bib37 ""), [26](#bib.bib26 "")\]. The performance is averaged over 10 random seeds.

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")