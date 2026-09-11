# Multi-Task Learning as a Bargaining Game

 Aviv Navon Affiliation: Bar-Ilan University, Ramat Gan, Israel Correspondence to: [aviv.navon@biu.ac.il](mailto:aviv.navon@biu.ac.il)    Aviv Shamsian Affiliation: Bar-Ilan University, Ramat Gan, Israel Correspondence to: [aviv.shamsian@live.biu.ac.il](mailto:aviv.shamsian@live.biu.ac.il)    Idan Achituve Affiliation: Bar-Ilan University, Ramat Gan, Israel    Haggai Maron Affiliation: Nvidia, Tel-Aviv, Israel    Kenji Kawaguchi Affiliation: National University of Singapore    Gal Chechik Affiliation: Bar-Ilan University, Ramat Gan, Israel Affiliation: Nvidia, Tel-Aviv, Israel    Ethan Fetaya Affiliation: Bar-Ilan University, Ramat Gan, Israel 

###### Abstract

In Multi-task learning (MTL), a joint model is trained to simultaneously make predictions for several tasks. Joint training reduces computation costs and improves data efficiency; however, since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance than its corresponding single-task counterparts. A common method for alleviating this issue is to combine per-task gradients into a joint update direction using a particular heuristic. In this paper, we propose viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction of parameter update. Under certain assumptions, the bargaining problem has a unique solution, known as the *Nash Bargaining Solution*, which we propose to use as a principled approach to multi-task learning. We describe a new MTL optimization procedure, Nash-MTL, and derive theoretical guarantees for its convergence. Empirically, we show that Nash-MTL achieves state-of-the-art results on multiple MTL benchmarks in various domains.

###### Keywords: 

Machine Learning, ICML 

††affiliationnotice: Equal contribution

## 1 Introduction

![Refer to caption](2202.01017v2/Figures/toy/toy_pareto_2d.png)

Figure 1: Illustrative example: Optimization trajectories in loss space. Shown are 5 different initializations (black dots ∙\\mathchoice{\\mathbin{\\vbox{\\hbox{\\scalebox{1.}{$\\displaystyle\\bullet$}}}}}{\\mathbin{\\vbox{\\hbox{\\scalebox{1.}{$\\textstyle\\bullet$}}}}}{\\mathbin{\\vbox{\\hbox{\\scalebox{1.}{$\\scriptstyle\\bullet$}}}}}{\\mathbin{\\vbox{\\hbox{\\scalebox{1.}{$\\scriptscriptstyle\\bullet$}}}}}), and their trajectories are colored from orange to purple. Losses have a large difference in scale. See Appendix [B](#A2 "Appendix B Experimental Details ‣ Multi-Task Learning as a Bargaining Game") for details. For linear scalarization (LS), PCGrad, and CAGrad, the optimization process is controlled by the gradient of ℓ2\\ell\_{2}, since it has a larger magnitude, resulting in imbalanced solutions between tasks (mostly ending at the bottom right). These three methods also fail to converge to an optimal solution for the rightmost initialization points. In contrast, MGDA is inclined towards the task with the smallest gradient magnitude (ℓ1\\ell\_{1}). Our method, Nash-MTL, is invariant to changes in loss scale and produces solutions that are well balanced across the Pareto front.

In many real-world applications, one needs to solve several tasks simultaneously using limited computational or data resources. For example, perception for autonomous vehicles requires lane detection, object detection, and free-space estimation, which must all run in parallel and in real-time. This is normally solved via multi-task learning (MTL), where one model is jointly trained on several learning tasks ([Caruana, 1997](#bib.bib6 ""); [Ruder, 2017](#bib.bib44 ""); [Crawshaw, 2020](#bib.bib10 "")). Multi-task learning was also shown to improve generalization in theory ([Baxter, 2000](#bib.bib4 "")) and in practice ([Liu et al., 2019a](#bib.bib31 ""); [Achituve et al., 2021](#bib.bib1 ""); [Navon et al., 2021a](#bib.bib37 ""), e.g., auxiliary learning,).

Unfortunately, MTL often causes performance degradation compared to single-task models ([Standley et al., 2020](#bib.bib51 "")). A main reason for such degradation is gradients conflict  ([Yu et al., 2020a](#bib.bib58 ""); [Wang et al., 2020](#bib.bib56 ""); [Liu et al., 2021a](#bib.bib29 "")). These per-task gradients may have conflicting directions or a large difference in magnitudes, with the largest gradient dominating the update direction. The degraded performance of MTL due to poor training, compared with its potential to improve performance due to better generalization, has a major impact on many real-world systems. Improving MTL optimization algorithms is therefore an important task with significant implications to many systems.

Currently, most MTL optimization algorithms  ([Sener & Koltun, 2018](#bib.bib46 ""); [Yu et al., 2020a](#bib.bib58 ""); [Liu et al., 2021a](#bib.bib29 "")) follow a general scheme. First, compute the gradients for all tasks g1,…,gKg\_{1},...,g\_{K}. Next, combine those gradients into a joint direction, Δ\=𝒜⁡(g1,…,gK)\\Delta=\\mathcal{A}(g\_{1},...,g\_{K}) using an aggregation algorithm 𝒜\\mathcal{A}. Finally, update model parameters using a single-task optimization algorithm, replacing the gradients with Δ\\Delta. Multiple heuristics were proposed for the aggregation algorithm 𝒜\\mathcal{A}. However, to the best of our knowledge, a principled, axiomatic, approach to gradient aggregation is still missing.

Here we address the gradient combination step by viewing it as a cooperative bargaining game ([Thomson, 1994](#bib.bib54 "")). Each task represents a player, whose utility is derived from its gradient, and players negotiate to reach an agreed direction. This formulation allows us to use results from game theory literature that analyze this problem from an axiomatic perspective. In his seminal paper, [Nash (1953)](#bib.bib36 "") presented an axiomatic approach to the bargaining problem and showed that under certain axioms, the bargaining problem has a unique solution known as the *Nash Bargaining Solution*. This solution is known to be proportionally fair, where any alternative will have a negative average relative change. This proportionally fair update allows us to find a solution that works for all tasks without being dominated by a single large gradient.

Building on Nash’s results, we propose a novel MTL optimization algorithm, named Nash-MTL, where the gradients are combined at each step using the Nash bargaining solution. We first characterize the Nash bargaining solution for MTL and derive an efficient algorithm to approximate its value. Then, we analyze our approach theoretically and establish convergence guarantees in the convex and nonconvex cases. Finally, we show empirically that our Nash-MTL approach achieves state-of-the-art results on four MTL benchmarks on a variety of challenges ranging from computer vision and quantum chemistry to reinforcement learning. To support future research and the reproducibility of the results, we make our source code publicly available at: [https://github.com/AvivNavon/nash-mtl](https://github.com/AvivNavon/nash-mtl "").

## 2 Background

### 2.1 Pareto Optimality

Optimization for MTL is a specific case of multiple-objective optimization (MOO). Given objective functions ℓ1,…,ℓK\\ell\_{1},...,\\ell\_{K}, the performance of solution a xx is measured by the vector of objective values (ℓ1​(x),…,ℓK​(x))(\\ell\_{1}(x),...,\\ell\_{K}(x)). One main property of MOO is that since there is no natural linear ordering on vectors it is not always possible to compare solutions so there is no clear optimal value.

We say that a solution xx dominates x′x^{\\prime} if it is better on one or more objectives and not worse on any other objectives. A solution that is not dominated by any other is called *Pareto optimal*, and the set of all such solutions is called the *Pareto front*. It is important to note that there is no clear way to select between different Pareto optimal solutions without additional assumptions or prior about the user preferences ([Navon et al., 2021b](#bib.bib38 "")). For non-convex problems, a point is defined as local Pareto optimal if it is Pareto optimal in some open set containing it. Further, a point is called *Pareto stationary* if there exists a convex combination of the gradients at this point that equals zero. Pareto stationarity is a necessary condition for Pareto optimality.

### 2.2 Nash Bargaining Solution

We provide a brief background on cooperative bargaining games and the Nash bargaining solution, see [Thomson (1994)](#bib.bib54 "") for more details. In a bargaining problem, we have KK players, each with their own utility function ui:A∪{D}→ℝu\_{i}:A\\cup\\{D\\}\\rightarrow\\mathbb{R}, which they wish to maximize. AA is the set of possible agreements and DD is the disagreement point which the players default to if they fail to reach an agreement. We define the set of possible payoffs as U\={(u1​(x),…,uK​(x)):x∈A}⊂ℝKU=\\{(u\_{1}(x),...,u\_{K}(x)):\\,x\\in A\\}\\subset\\mathbb{R}^{K} and d\=(u1​(D),…,uK​(D))d=(u\_{1}(D),...,u\_{K}(D)). We assume UU is convex, compact and that there exists a point in UU that strictly dominates dd, namely there exists a u∈Uu\\in U such that ∀i:ui\>di\\forall i:u\_{i}>d\_{i}.

[Nash (1953)](#bib.bib36 "") showed that for such payoff set UU, the two-player bargaining problem has a unique solutionthat satisfies the following properties or axioms: Pareto optimality, symmetry, independence of irrelevant alternatives, and invariant to affine transformations. This was later extended to multiple players ([Szép & Forgó, 1985](#bib.bib53 "")).

###### Axiom 2.1.

Pareto optimality: The agreed solution must not be dominated by another option, i.e. there cannot be any other agreement that is better for at least one player and not worse for any of the players.

As it is a cooperative game, it makes little sense that the players will curtail another player without any personal gains, so it is natural to assume the agreed solution will not be dominated by another.

###### Axiom 2.2.

Symmetry: The solution should be invariant to permuting the order of the players.

###### Axiom 2.3.

Independence of irrelevant alternatives (IIA): If we enlarge the of possible payoffs to U\~⊋U\\tilde{U}\\supsetneq U, and the solution is in the original set UU, u∗∈Uu^{\*}\\in U, then the agreed point when the set of possible payoffs is UU will stay u∗u^{\*}.

###### Axiom 2.4.

Invariance to affine transformation: If we transform each utility function ui​(x)u\_{i}(x) to u\~i​(x)\=ci⋅ui​(x)+bi\\tilde{u}\_{i}(x)=c\_{i}\\cdot u\_{i}(x)+b\_{i} with ci\>0c\_{i}>0 then if the original agreement had utilities (y1,…,yk)(y\_{1},...,y\_{k}) the agreement after the transformation has utilities (c1​y1+b1,…,ck​yk+bk)(c\_{1}y\_{1}+b\_{1},...,c\_{k}y\_{k}+b\_{k})

We argue that in the MTL setting, it is natural to require axioms 2.1-2.3. Axiom 2.4, in our mind, is the only non-natural assumption used by the Nash bargaining solution in the context of MTL. We argue that indeed it is a desired property that is helpful for MTL. Axiom 2.4 means that the solution does not take into account the gradients’ norms but rather treats all of them the same, as if they were normalized. Without enforcing this assumption, the solution can easily be dominated by a single direction (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Multi-Task Learning as a Bargaining Game")). We further validate the importance of this assumption by investigating a scale-invariant baseline in Section [6](#S6 "6 Experiments ‣ Multi-Task Learning as a Bargaining Game").

The unique point satisfying all these axioms is called the Nash bargaining solution and is given as

u∗\=\\displaystyle u^{\*}=

arg⁡max⁡∑iu∈U⁡log⁡(ui−di)\\displaystyle\\arg\\max\_{u\\in U}\\sum\_{i}\\log(u\_{i}-d\_{i})

(1)

s.t.∀i:ui\>di\\displaystyle s.t.\\,\\,\\forall i:\\,u\_{i}>d\_{i}

## 3 Method

We now describe our Nash-MTL method in detail. We first formalize the gradient combination step as a bargaining game and analyze the Nash bargaining solution for this game. We then describe our algorithm to approximate the solution efficiently. We note that the computational cost of that approximation is critical because this approximation is executed for each gradient update. To simplify the notation, we do not distinguish between shared and task-specific parameters. We note, however, that task-specific parameters have no contribution to the Nash bargaining solution calculation.

### 3.1 Nash Bargaining Multi-Task Learning

Given an MTL optimization problem and parameters θ\\theta, we search for an update vector Δ​θ\\Delta\\theta in the ball of radius ϵ\\epsilon centered around zero, BϵB\_{\\epsilon}. We frame this as a bargaining problem with the agreement set BϵB\_{\\epsilon} and the disagreement point at 00, i.e., staying at the current parameters θ\\theta. We define the utility function for each player as ui​(Δ​θ)\=gi⊤​Δ​θu\_{i}(\\Delta\\theta)=g\_{i}^{\\top}\\Delta\\theta where gig\_{i} is the gradient of the loss of task ii at θ\\theta. We note that since the agreement set is compact and convex and the utilities are linear then the set of possible payoffs is also compact and convex.

Our main assumption, besides the ones used by Nash, is that if θ\\theta is not Pareto stationary then the gradients are linearly independent (see further discussion on this assumption in Section [5](#S5 "5 Analysis ‣ Multi-Task Learning as a Bargaining Game")). Under this assumption, we also have that the disagreement point, Δ​θ\=0\\Delta\\theta=0 is dominated by another in BϵB\_{\\epsilon}. We now show that if θ\\theta is not on the Pareto front, the unique Nash bargaining solution has the following form:

###### Claim 3.1.

Let GG be the d×Kd\\times K matrix whose columns are the gradients gig\_{i}. The solution to arg⁡max⁡∑iΔ​θ∈Bϵ⁡log⁡(Δ​θ⊤​gi)\\arg\\max\_{\\Delta\\theta\\in B\_{\\epsilon}}\\sum\_{i}\\log(\\Delta\\theta^{\\top}g\_{i}) is (up to scaling) ∑iαi​gi\\sum\_{i}\\alpha\_{i}g\_{i} where α∈ℝ+K\\alpha\\in\\mathbb{R}^{K}\_{+} is the solution to G⊤​G​α\=1/αG^{\\top}G\\alpha=1/\\alpha where 1/α1/\\alpha is the element-wise reciprocal.

###### Proof.

The derivative of this objective is ∑i\=1K1Δ​θ⊤​gi​gi\\sum\_{i=1}^{K}\\frac{1}{\\Delta\\theta^{\\top}g\_{i}}g\_{i}. For all vectors Δ​θ\\Delta\\theta such that ∀i:Δ​θT​gi\>0\\forall i:\\Delta\\theta^{T}g\_{i}>0 the utilities are monotonically increasing with the norm of Δ​θ\\Delta\\theta. Thus, from the Pareto optimality assumption by Nash, the optimal solution has to be on the boundary of BϵB\_{\\epsilon}. From this we see that the gradient at the optimal point ∑i\=1K1Δ​θ⊤​gi​gi\\sum\_{i=1}^{K}\\frac{1}{\\Delta\\theta^{\\top}g\_{i}}g\_{i} must be in the radial direction, i.e., ∑i\=1K1Δ​θ⊤​gi​gi|Δ​θ\\sum\_{i=1}^{K}\\frac{1}{\\Delta\\theta^{\\top}g\_{i}}g\_{i}\\parallel\\Delta\\theta or ∑i\=1K1Δ​θ⊤​gi​gi\=λ​Δ​θ\\sum\_{i=1}^{K}\\frac{1}{\\Delta\\theta^{\\top}g\_{i}}g\_{i}=\\lambda\\Delta\\theta. Since the gradients are independent we must have Δ​θ\=∑iαi​gi\\Delta\\theta=\\sum\_{i}\\alpha\_{i}g\_{i} and ∀i:1Δ​θ⊤​gi\=λ​αi\\forall i:\\frac{1}{\\Delta\\theta^{\\top}g\_{i}}=\\lambda\\alpha\_{i} or ∀i:Δ​θ⊤​gi\=1λ​αi\\forall i:{\\Delta\\theta^{\\top}g\_{i}}=\\frac{1}{\\lambda\\alpha\_{i}}. As the inner product must be positive for a descent direction we can conclude λ\>0\\lambda>0; we set λ\=1\\lambda=1 to ascertain the direction of Δ​θ\\Delta\\theta (the norm might be larger then ϵ\\epsilon). Now finding the bargaining solution is reduced to finding α∈ℝK\\alpha\\in\\mathbb{R}^{K} with αi\>0\\alpha\_{i}>0 such that ∀i:Δ​θ⊤​gi\=∑jαj​gj⊤​gi\=1αi\\forall i:{\\Delta\\theta^{\\top}g\_{i}}=\\sum\_{j}\\alpha\_{j}g\_{j}^{\\top}g\_{i}=\\frac{1}{\\alpha\_{i}}. This is equivalent to requiring that G⊤​G​α\=1/αG^{\\top}G\\alpha=1/\\alpha where 1/α1/\\alpha is the element-wise reciprocal. ∎

We now provide some intuition for this solution. First, if all gig\_{i} are orthogonal we get αi\=1/‖gi‖\\alpha\_{i}=1/||g\_{i}|| and Δ​θ\=∑gi‖gi‖\\Delta\\theta=\\sum\\frac{g\_{i}}{||g\_{i}||} which is the obvious scale invariant solution. When they are not orthogonal, we get

αi​‖gi‖2+∑j≠iαj​gj⊤​gi\=1/αi\\alpha\_{i}||g\_{i}||^{2}+\\sum\_{j\\neq i}\\alpha\_{j}g\_{j}^{\\top}g\_{i}=1/\\alpha\_{i}

(2)

We can consider ∑j≠iαj​gj⊤​gi\=(∑j≠iαj​gj)⊤​gi\\sum\_{j\\neq i}\\alpha\_{j}g\_{j}^{\\top}g\_{i}=\\left(\\sum\_{j\\neq i}\\alpha\_{j}g\_{j}\\right)^{\\top}g\_{i} as the interaction between task ii and the other tasks; If it is positive there is a positive interaction and the other gradients aid the ii’th task, and if it is negative they hamper it. When there is a negative interaction, the LHS of Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Nash Bargaining Multi-Task Learning ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") decreases and as a result, αi\\alpha\_{i} increases to compensate for it. Conversely, where there is a positive interaction αi\\alpha\_{i} will decrease.

Algorithm 1 Nash-MTL

 Input: θ(0)\\theta^{(0)} – initial parameter vector, {ℓi}i\=1K\\{\\ell\_{i}\\}\_{i=1}^{K} – differentiable loss functions, η\\eta – learning rate 

 for t\=1,…,Tt=1,...,T do 

  Compute task gradients gi(t)\=∇θ(t−1)ℓig^{(t)}\_{i}=\\nabla\_{\\theta^{(t-1)}}\\ell\_{i} 

  Set G(t)G^{(t)} the matrix with columns gi(t)g^{(t)}\_{i} 

  Solve for α\\alpha: (G(t))⊤​G(t)​α\=1/α(G^{(t)})^{\\top}G^{(t)}\\alpha=1/\\alpha to obtain α(t)\\alpha^{(t)} 

  Update the parameters θ(t)\=θ(t)−η​G(t)​α(t)\\theta^{(t)}=\\theta^{(t)}-\\eta G^{(t)}\\alpha^{(t)} 

 end for 

 Return: θ(T)\\theta^{(T)} 

![Refer to caption](2202.01017v2/Figures/update_illustration/update_direction_draw.png)

Figure 2: Visualization of the update direction: We show the update direction (blue) obtained by various methods on three gradients in ℝ3\\mathbb{R}^{3}. We rescaled the returned vectors for better visibility, showing only the direction. We further show the size of the projection (red) of the update to each gradient direction (black). Nash-MTL produce an update direction with the most balanced projections.

### 3.2 Solving 𝐆⊤​𝐆​α\=1/α\\mathbf{G^{\\top}G\\alpha}=1/\\alpha

Here we describe how to efficiently approximate the optimal solution for G⊤​G​α\=1/αG^{\\top}G\\alpha=1/\\alpha through a sequence of convex optimization problems. We define a βi​(α)\=gi⊤​G​α\\beta\_{i}(\\alpha)=g\_{i}^{\\top}G\\alpha, and wish to find α\\alpha such that αi\=1/βi\\alpha\_{i}=1/\\beta\_{i} for all ii, or equivalently log⁡(αi)+log⁡(βi​(α))\=0\\log(\\alpha\_{i})+\\log(\\beta\_{i}(\\alpha))=0. Denote φi​(α)\=log⁡(αi)+log⁡(βi)\\varphi\_{i}(\\alpha)=\\log(\\alpha\_{i})+\\log(\\beta\_{i}) and φ⁡(α)\=∑iφi​(α)\\varphi(\\alpha)=\\sum\_{i}\\varphi\_{i}(\\alpha). With that, our goal is to find a non-negative α\\alpha such that φi​(α)\=0\\varphi\_{i}(\\alpha)=0 for all ii. We can write this as the following optimization problem

minα\\displaystyle\\min\_{\\alpha}

∑iφi​(α)\\displaystyle\\sum\_{i}\\varphi\_{i}(\\alpha)

(3)

s.t.​∀i,\\displaystyle\\text{s.t.}\\forall i,\\quad

−φi​(α)≤0\\displaystyle-\\varphi\_{i}(\\alpha)\\leq 0

αi\>0.\\displaystyle\\alpha\_{i}>0\\quad.

The constraints in this problem are convex and linear and the objective is concave. We first try to solve the following convex surrogate objective

minα\\displaystyle\\min\_{\\alpha}

∑iβi​(α)\\displaystyle\\sum\_{i}\\beta\_{i}(\\alpha)

(4)

s.t.​∀i,\\displaystyle\\text{s.t.}\\forall i,\\quad

−φi​(α)≤0\\displaystyle-\\varphi\_{i}(\\alpha)\\leq 0

αi\>0.\\displaystyle\\alpha\_{i}>0\\quad.

Here, we minimize ∑iβi\\sum\_{i}\\beta\_{i} under the constraint βi\=gi⊤​G​α≥1/αi\\beta\_{i}=g\_{i}^{\\top}G\\alpha\\geq 1/\\alpha\_{i}. While this objective is not equivalent to the original problem, we found it very useful. In many cases, it produces exact solutions with φ⁡(α)\=0\\varphi(\\alpha)=0 as required.

To further improve our approximation, we considered the following problem, minα\\displaystyle\\min\_{\\alpha}

∑iβi​(α)+φ⁡(α)\\displaystyle\\sum\_{i}\\beta\_{i}(\\alpha)+\\varphi(\\alpha)

(5)

s.t.​∀i,\\displaystyle\\text{s.t.}\\forall i,\\quad

−φi​(α)≤0\\displaystyle-\\varphi\_{i}(\\alpha)\\leq 0

αi\>0.\\displaystyle\\alpha\_{i}>0\\quad.

Adding φ⁡(α)\\varphi(\\alpha) to the objective may further reduce it, moving it closer to zero; however, it renders the problem to be non-convex. Despite that, our solution can be improved iteratively by replacing the concave term φ⁡(α)\\varphi(\\alpha) with its first-order approximation φ\~τ(α)\=φ(α(τ))+∇φ(α(τ))⊤(α−α(τ))\\tilde{\\varphi}\_{\\tau}(\\alpha)=\\varphi(\\alpha^{(\\tau)})+\\nabla\\varphi(\\alpha^{(\\tau)})^{\\top}(\\alpha-\\alpha^{(\\tau)}). Where, α(τ)\\alpha^{(\\tau)} is the solution at iteration τ\\tau. Note that we replace φ\\varphi with φ\~\\tilde{\\varphi} only in the objective and keep φ⁡(α)\\varphi(\\alpha) as is in the constraint: i.e., min⁡∑iα⁡βi​(α)+φ\~τ​(α)​ s.t. −φi​(α)≤0​ and ​αi\>0\\min\_{\\alpha}\\sum\_{i}\\beta\_{i}(\\alpha)+\\tilde{\\varphi}\_{\\tau}(\\alpha)\\text{ s.t. }-\\varphi\_{i}(\\alpha)\\leq 0\\text{ and }\\alpha\_{i}>0 for all ii. This sequential optimization approach is a variation of the concave-convex procedure (CCP) ([Yuille & Rangarajan, 2003](#bib.bib60 ""); [Lipp & Boyd, 2016](#bib.bib28 "")). Therefore the sequence {α(τ)}τ\\{\\alpha^{(\\tau)}\\}\_{\\tau} converges to a critical point of the original non-convex problem in Eq. [5](#S3.E5 "Equation 5 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") based on previous theory of CCP by [Sriperumbudur & Lanckriet (2009)](#bib.bib50 ""). Moreover, since we do not modify the constraint, α(τ)\\alpha^{(\\tau)} always satisfies the constraint of the original problem for any τ\\tau. Finally, the following proposition shows that original objective monotonically decreases with τ\\tau:

###### Proposition 3.2.

Denote the objective for the optimization problem in Eq. [5](#S3.E5 "Equation 5 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") by ϕ⁡(α)\=∑iβi​(α)+φ⁡(α)\\phi(\\alpha)=\\sum\_{i}\\beta\_{i}(\\alpha)+\\varphi(\\alpha). Then, ϕ⁡(α(τ+1))≤ϕ⁡(α(τ))\\phi\\left(\\alpha^{(\\tau+1)}\\right)\\leq\\phi\\left(\\alpha^{(\\tau)}\\right) for all τ≥1\\tau\\geq 1.

We provide proof and further discussion in Appendix [A](#A1 "Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game"). In practice, we limit the sequence of CCP to 20 in all experiments, with the exception of Section [6.3](#S6.SS3 "6.3 Multi-Task Reinforcement Learning ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game") for which we use a single step. We found the improved solution to have a limited effect on the MTL performance (see Appendix [D.2](#A4.SS2 "D.2 Effect of the Number of CCP steps ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game")).

### 3.3 Practical Speedup

One shortcoming of many leading MTL methods is that all task gradients are required for obtaining the joint update direction. When the number of tasks KK becomes large, this may be too computationally expensive as it requires one to perform KK backward passes through the shared backbone to compute the KK gradients. Prior work suggested using a subset of tasks ([Liu et al., 2021a](#bib.bib29 "")) or replacing the task gradients with the feature level gradient ([Sener & Koltun, 2018](#bib.bib46 ""); [Liu et al., 2021b](#bib.bib30 ""); [Javaloy & Valera, 2021](#bib.bib21 "")) as potential practical speedups. We emphasize that this issue is not unique to our method, but rather is shared to all methods that compute all gradients for all tasks.

In practice, we found that using feature-level gradients as a surrogate to the gradient of the shared parameters dramatically degrades the performance of our method. See Appendix [C](#A3 "Appendix C Computing Task Gradient at the Features-Level ‣ Multi-Task Learning as a Bargaining Game") for empirical results and further discussion. As an alternative, we suggest updating the gradient weights α(t)\\alpha^{(t)} once every few iterations instead of every iteration. This simple yet effective solution greatly reduces the runtime (up to ∼×10\\sim\\times 10 for QM9 and ∼×5\\sim\\times 5 for MT10) while maintaining high performance. In Section [6.4](#S6.SS4 "6.4 Scaling-up Nash-MTL ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game") we provide experimental results while varying the frequency of task weights update on the QM9 dataset and the MT10 benchmark. Our results show that Nash-MTL runtime can be reduced to about the same as linear scalarization (or STL) while maintaining competitive results compared to other baselines; However, in some cases, we do see a noticeable drop in performance compared with our standard approach.

## 4 Related Work

In multitask learning (MTL), one simultaneously solves several learning problems while sharing information among tasks ([Caruana, 1997](#bib.bib6 ""); [Ruder, 2017](#bib.bib44 "")), commonly through a joint hidden representation ([Zhang et al., 2014](#bib.bib62 ""); [Dai et al., 2016](#bib.bib12 ""); [Pinto & Gupta, 2017](#bib.bib40 ""); [Zhao et al., 2018](#bib.bib63 ""); [Liu et al., 2019b](#bib.bib32 "")).

![Refer to caption](2202.01017v2/Figures/qm9/qm_9_delta_m_sem.png)

Figure 3: QM9. Test Δm\\Delta\_{m} throughout the training process averaged over 3 random seeds.

Studies in the literature proposed several explanations for the difficulty in the optimization process of MTL, such as conflicting gradients ([Wang et al., 2020](#bib.bib56 ""); [Yu et al., 2020a](#bib.bib58 "")), or plateaus in the loss landscape ([Schaul et al., 2019](#bib.bib45 "")). Other studies aimed at improving multitask learning by proposing novel architectures ([Misra et al., 2016](#bib.bib35 ""); [Hashimoto et al., 2017](#bib.bib19 ""); [Liu et al., 2019b](#bib.bib32 ""); [Chen et al., 2020](#bib.bib8 "")). We focus on weighting the gradients of the tasks via an axiomatic approach that is agnostic to the architecture used. Studies in a similar vein proposed to weigh the task losses with various approaches, such as the uncertainty of the tasks ([Kendall et al., 2018](#bib.bib22 "")), the norm of the gradients ([Chen et al., 2018](#bib.bib7 "")), random weights ([Lin et al., 2021](#bib.bib27 "")), and similarity of the gradients ([Du et al., 2018](#bib.bib14 ""); [Suteu & Guo, 2019](#bib.bib52 "")). These methods are mostly heuristic and can have unstable performance ([Liu et al., 2021a](#bib.bib29 "")). Recently, several studies proposed MTL approaches based on the multiple-gradient descent algorithm (MGDA) for multi-objective optimization ([Désidéri, 2012](#bib.bib13 "")). This is an appealing approach since, under mild conditions, convergence to a Pareto stationary point is guaranteed. [Sener & Koltun (2018)](#bib.bib46 "") cast the multi-objective problem to multi-task problem and suggest task weighting based on the Frank-Wolfe algorithm ([Jaggi, 2013](#bib.bib20 "")). [Liu et al. (2021a)](#bib.bib29 "") searches for an update direction in a neighborhood of the average gradient that maximizes the worst improvement of any task. Unlike these studies, we propose an MTL approach based on a Bargaining game that can find solutions that are Pareto optimal and proportionally fair.

The closest work to our approach, to the best of our knowledge, is [Liu et al. (2021b)](#bib.bib30 ""). There, the authors propose to look for a fair gradient direction where all the cosine similarities are equal. We note that this update direction satisfies all of the Nash axioms except for Pareto optimally. Thus, unlike our proportionally fair approach, it can settle for a sub-optimal solution for the sake of fairness.

Finally, we note that the Nash bargaining solution was effectively applied to problems in various fields such as communication ([Zhang et al., 2008](#bib.bib61 ""); [Leshem & Zehavi, 2011](#bib.bib26 ""); [Shi et al., 2018](#bib.bib47 "")), economics ([Dagan & Volij, 1993](#bib.bib11 "")), and computing ([Grosu et al., 2002](#bib.bib17 "")), and to several learning setups, such as reinforcement learning ([Qiao et al., 2006](#bib.bib41 "")), Bayesian optimization ([Binois et al., 2020](#bib.bib5 "")), clustering ([Rezaee et al., 2021](#bib.bib43 "")), federated learning ([Kim, 2021](#bib.bib23 "")), and multi-armed bandits ([Baek & Farias, 2021](#bib.bib3 "")).

Table 1: QM9. Test performance averaged over 3 random seeds.

|          | MR ↓\\downarrow  | 𝚫𝐦%\\mathbf{\\Delta\_{m}\\%} ↓\\downarrow |
| -------- | ---------------- | ----------------------------------------- |
| LS       | 6.8              | 177.6±3.4177.6\\pm~~3.4                   |
| SI       | 4.0              | 77.8±9.2\~~77.8\\pm~~9.2                  |
| RLW      | 8.2              | 203.8±3.4203.8\\pm~~3.4                   |
| DWA      | 6.4              | 175.3±6.3175.3\\pm~~6.3                   |
| UW       | 5.3              | 108.0±22.5108.0\\pm 22.5                  |
| MGDA     | 5.9              | 120.5±2.0120.5\\pm~~2.0                   |
| PCGrad   | 5.0              | 125.7±10.3125.7\\pm 10.3                  |
| CAGrad   | 5.7              | 112.8±4.0112.8\\pm~~4.0                   |
| IMTL-G   | 4.7              | 77.2±9.3\~~77.2\\pm~~9.3                  |
| Nash-MTL | 2.5\\mathbf{2.5} | 62.0±1.4\\mathbf{~62.0\\pm~1.4}           |

Table 2: NYUv2. Test performance for three tasks: semantic segmentation, depth estimation, and surface normal. Values are averages over 3 random seeds.

|                      |                      |      |                        |                        |       |                             |                      |                                |                      |                 |                                        |  |                      |  |                    |                      |  |
| -------------------- | -------------------- | ---- | ---------------------- | ---------------------- | ----- | --------------------------- | -------------------- | ------------------------------ | -------------------- | --------------- | -------------------------------------- |  | -------------------- |  | ------------------ | -------------------- |  |
| mIoU ↑\\uparrow      | Pix Acc ↑\\uparrow   |      | Abs Err ↓\\downarrow   | Rel Err ↓\\downarrow   |       | Angle Distance ↓\\downarrow |                      | Within t∘t^{\\circ} ↑\\uparrow |                      | MR ↓\\downarrow | 𝚫​𝐦%↓\\mathbf{\\Delta m\\%}\\downarrow |  |                      |  |                    |                      |  |
|                      |                      | Mean | Median                 |                        | 11.25 |                             | 22.5                 |                                | 30                   |                 |                                        |  |                      |  |                    |                      |  |
| 38.3038.30           | 63.7663.76           |      | 0.67540.6754           | 0.27800.2780           |       | 25.0125.01                  | 19.2119.21           |                                | 30.1430.14           |                 | 57.2057.20                             |  | 69.1569.15           |  |                    |                      |  |
| 39.2939.29           | 65.3365.33           |      | 0.54930.5493           | 0.22630.2263           |       | 28.1528.15                  | 23.9623.96           |                                | 22.0922.09           |                 | 47.5047.50                             |  | 61.0861.08           |  |   8.118.11         | 5.595.59             |  |
| 38.4538.45           | 64.2764.27           |      | 0.53540.5354           | 0.22010.2201           |       | 27.6027.60                  | 23.3723.37           |                                | 22.5322.53           |                 | 48.5748.57                             |  | 62.3262.32           |  |   7.117.11         | 4.394.39             |  |
| 37.1737.17           | 63.7763.77           |      | 0.57590.5759           | 0.24100.2410           |       | 28.2728.27                  | 24.1824.18           |                                | 22.2622.26           |                 | 47.0547.05                             |  | 60.6260.62           |  | 10.1110.11         | 7.787.78             |  |
| 39.1139.11           | 65.3165.31           |      | 0.55100.5510           | 0.22850.2285           |       | 27.6127.61                  | 23.1823.18           |                                | 24.1724.17           |                 | 50.1850.18                             |  | 62.3962.39           |  | 6.886.88           | 3.573.57             |  |
| 36.8736.87           | 63.1763.17           |      | 0.54460.5446           | 0.22600.2260           |       | 27.0427.04                  | 22.6122.61           |                                | 23.5423.54           |                 | 49.0549.05                             |  | 63.6563.65           |  | 6.446.44           | 4.054.05             |  |
| 30.4730.47           | 59.9059.90           |      | 0.60700.6070           | 0.25550.2555           |       | 24.88\\mathbf{24.88}        | 19.45\\mathbf{19.45} |                                | 29.18\\mathbf{29.18} |                 | 56.88\\mathbf{56.88}                   |  | 69.36\\mathbf{69.36} |  | 5.445.44           | 1.381.38             |  |
| 38.0638.06           | 64.6464.64           |      | 0.55500.5550           | 0.23250.2325           |       | 27.4127.41                  | 22.8022.80           |                                | 23.8623.86           |                 | 49.8349.83                             |  | 63.1463.14           |  | 6.886.88           | 3.973.97             |  |
| 39.3939.39           | 65.1265.12           |      | 0.54550.5455           | 0.22790.2279           |       | 27.4827.48                  | 22.9622.96           |                                | 23.3823.38           |                 | 49.4449.44                             |  | 62.8762.87           |  | 6.446.44           | 3.583.58             |  |
| 39.7939.79           | 65.4965.49           |      | 0.54860.5486           | 0.22500.2250           |       | 26.3126.31                  | 21.5821.58           |                                | 25.6125.61           |                 | 52.3652.36                             |  | 65.5865.58           |  | 3.773.77           | 0.200.20             |  |
| 39.3539.35           | 65.6065.60           |      | 0.54260.5426           | 0.22560.2256           |       | 26.0226.02                  | 21.1921.19           |                                |   26.226.2           |                 | 53.1353.13                             |  | 66.2466.24           |  | 3.113.11           | −0.76-0.76           |  |
| 40.13\\mathbf{40.13} | 65.93\\mathbf{65.93} |      | 0.5261\\mathbf{0.5261} | 0.2171\\mathbf{0.2171} |       | 25.2625.26                  | 20.0820.08           |                                | 28.428.4             |                 | 55.4755.47                             |  | 68.1568.15           |  | 1.55\\mathbf{1.55} | −4.04\\mathbf{-4.04} |  |

## 5 Analysis

We now analyze the convergence of our method in the convex and non-convex cases. As even single-task non-convex optimization might only converge to a stationary point, we will prove convergence to a Pareto stationary point, i.e., a point where some convex combination of the gradients is zero. As stated, we also assume that the gradients are independent while not at a Pareto stationary point. Independence of the gradients is a slightly stronger assumption than Pareto stationarity but is needed to exclude degenerate edge cases such as two identical tasks.

We note that by substituting local Pareto optimality for Pareto stationarity in Assumption [5.1](#S5.Thmtheorem1 "Assumption 5.1. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game") we can show convergence to a local Pareto optimal point. However, this assumption has strong implications, as it implies we avoid local maxima and saddle points of any specific task. Since our update rule is a descent direction for all tasks, we can reasonably assume that our algorithm avoids local maxima points. Furthermore, it was shown that first-order methods avoid saddle points ([Panageas et al., 2019](#bib.bib39 "")), giving credence to this stronger assumption. Nevertheless, we take a conservative approach and state our results with the weaker assumption.

We formally make the following assumptions:

###### Assumption 5.1.

We assume that for a sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} generated by our algorithm, the set of the gradient vectors g1(t),…,gK(t)g\_{1}^{(t)},...,g\_{K}^{(t)} at any point on the sequence and at any partial limit are linearly independent unless that point is a Pareto stationary point.

Table 3: CityScapes. Test performance for two tasks: semantic segmentation and depth estimation. Value are averages over 3 random seeds.

| Segmentation |       | Depth |  |        |       |  |                    |        |
| ------------ | ----- | ----- |  | ------ | ----- |  | ------------------ | ------ |
|              |       |       |  |        |       |  |                    |        |
|              |       |       |  |        |       |  |                    |        |
| LS           | 75.18 | 93.49 |  | 0.0155 | 46.77 |  | 6.126.12           | 22.60  |
| SI           | 70.95 | 91.73 |  | 0.0161 | 33.83 |  | 8.008.00           | 14.11  |
| RLW          | 74.57 | 93.41 |  | 0.0158 | 47.79 |  | 9.259.25           | 24.38  |
| DWA          | 75.24 | 93.52 |  | 0.0160 | 44.37 |  | 6.006.00           | 21.45  |
| UW           | 72.02 | 92.85 |  | 0.0140 | 30.13 |  | 5.255.25           |   5.89 |
| MGDA         | 68.84 | 91.54 |  | 0.0309 | 33.50 |  | 8.758.75           | 44.14  |
| PCGrad       | 75.13 | 93.48 |  | 0.0154 | 42.07 |  | 6.376.37           | 18.29  |
| GradDrop     | 75.27 | 93.53 |  | 0.0157 | 47.54 |  | 5.505.50           | 23.73  |
| CAGrad       | 75.16 | 93.48 |  | 0.0141 | 37.60 |  | 5.375.37           | 11.64  |
| IMTL-G       | 75.33 | 93.49 |  | 0.0135 | 38.41 |  | 3.623.62           | 11.10  |
| Nash-MTL     | 75.41 | 93.66 |  | 0.0129 | 35.02 |  | 1.75\\mathbf{1.75} | 6.82   |

###### Assumption 5.2.

We assume that all loss functions are differentiable, bounded below and that all sub-level sets are bounded. The input domain is open and convex.

###### Assumption 5.3.

We assume that all the loss functions are L-smooth, ‖∇ℓi​(x)−∇ℓi​(y)‖≤L​‖x−y‖.||\\nabla\\ell\_{i}(x)-\\nabla\\ell\_{i}(y)||\\leq L||x-y||\\quad.

(6)

###### Theorem 5.4.

Let {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} be the sequence generated by the update rule θ(t+1)\=θ(t)−μ(t)​Δ​θ(t)\\theta^{(t+1)}=\\theta^{(t)}-\\mu^{(t)}\\Delta\\theta^{(t)} where Δ​θ(t)\=∑i\=1Kαi(t)​gi(t)\\Delta\\theta^{(t)}=\\sum\_{i=1}^{K}\\alpha^{(t)}\_{i}g\_{i}^{(t)} is the Nash bargaining solution (G(t))⊤​G(t)​α(t)\=1/α(t)(G^{(t)})^{\\top}G^{(t)}\\alpha^{(t)}=1/\\alpha^{(t)}. Set μ(t)\=mini∈\[K\]⁡1L​K​αi(t)\\mu^{(t)}=\\min\\limits\_{i\\in\[K\]}\\frac{1}{LK\\alpha^{(t)}\_{i}}. Then, the sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} has a subsequence that converges to a Pareto stationary point θ∗\\theta^{\*}. Moreover all the loss functions (ℓ1​(θ(t)),…,ℓK​(θ(t)))(\\ell\_{1}(\\theta^{(t)}),...,\\ell\_{K}(\\theta^{(t)})) converge to (ℓ1​(θ∗),…,ℓK​(θ∗))(\\ell\_{1}(\\theta^{\*}),...,\\ell\_{K}(\\theta^{\*})).

###### Proof sketch.

We can show that μ(t)\=mini⁡1αi(t)→0\\mu^{(t)}=\\min\_{i}\\frac{1}{\\alpha^{(t)}\_{i}}\\rightarrow 0 so ‖α(t)‖→∞||\\alpha^{(t)}||\\rightarrow\\infty. We also show that ‖1/α(t)‖||1/\\alpha^{(t)}|| is bounded. As (G(t))⊤​G(t)​α(t)\=1/α(t)(G^{(t)})^{\\top}G^{(t)}\\alpha^{(t)}=1/\\alpha^{(t)} this means that the smallest singluar value of (G(t))⊤​G(t)(G^{(t)})^{\\top}G^{(t)} must converge to zero. From compactness {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} has a converging subsequence whose limit we denote as θ∗\\theta^{\*}. From continuity we get that the gradients Gram matrix G⊤​GG^{\\top}G computed at θ∗\\theta^{\*} must have a zero singular value and therefore the gradients are linearly dependent. From our assumption this means that θ∗\\theta^{\*} is Pareto stationary. As the losses are monotonically decearsing and bounded below they must converge and to the subsequence limit of (ℓ1​(θ∗),…,ℓK​(θ∗))(\\ell\_{1}(\\theta^{\*}),...,\\ell\_{K}(\\theta^{\*})). ∎

If we also assume convexity, we can strengthen our claim

###### Theorem 5.5.

Let {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} be the sequence generated by the update rule θ(t+1)\=θ(t)−μ(t)​Δ​θ(t)\\theta^{(t+1)}=\\theta^{(t)}-\\mu^{(t)}\\Delta\\theta^{(t)} where Δ​θ(t)\=∑i\=1Kαi(t)​gi(t)\\Delta\\theta^{(t)}=\\sum\_{i=1}^{K}\\alpha^{(t)}\_{i}g\_{i}^{(t)} is the Nash bargaining solution (G(t))⊤​G(t)​α(t)\=1/α(t)(G^{(t)})^{\\top}G^{(t)}\\alpha^{(t)}=1/\\alpha^{(t)}. Set μ(t)\=mini∈\[K\]⁡1L​K​αi(t)\\mu^{(t)}=\\min\\limits\_{i\\in\[K\]}\\frac{1}{LK\\alpha^{(t)}\_{i}}. If we assume that all the loss functions are convex, then the sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} converges to a Pareto optimal point θ∗\\theta^{\*}.

See the full proofs in the appendix Sec. [A](#A1 "Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game").

## 6 Experiments

We evaluate Nash-MTL on diverse multi-task learning problems. The experiments show the superiority of Nash-MTL over previous MTL methods. To support future research and the reproducibility of the results, we will make our source code publicly available. Additional experimental results and details are provided in Appendix [B](#A2 "Appendix B Experimental Details ‣ Multi-Task Learning as a Bargaining Game").

Compared methods: We compare the following approaches: (1) Our proposed Nash-MTL algorithm described in Section [3](#S3 "3 Method ‣ Multi-Task Learning as a Bargaining Game"); (2) Single task learning (STL), training an independent model for each task; (3) Linear scalarization (LS) baseline which minimizes ∑kℓk\\sum\_{k}\\ell\_{k}; (4) Scale-invariant (SI) baseline which minimizes ∑klog⁡ℓk\\sum\_{k}\\log\\ell\_{k}. This baseline is invariant to rescaling each loss with a positive number; (5) Dynamic Weight Average (DWA) ([Liu et al., 2019b](#bib.bib32 "")) adjusts task weights based on the rates of loss changes over time; (6) Uncertainty weighting (UW) ([Kendall et al., 2018](#bib.bib22 "")) uses task uncertainty quantification to adjust task weights; (7) MGDA ([Sener & Koltun, 2018](#bib.bib46 "")) finds a convex combination of gradients with a minimal norm; (8) Random loss weighting (RLW) with normal distribution, scales the losses according to randomly sampled task weights ([Lin et al., 2021](#bib.bib27 "")); (9) PCGrad ([Yu et al., 2020a](#bib.bib58 "")) removes conflicting components of each gradient w.r.t the other gradients; (10) GradDrop ([Chen et al., 2020](#bib.bib8 "")) randomly drops components of the task gradients based on how much they conflict; (11) CAGrad ([Liu et al., 2021a](#bib.bib29 "")) optimizes for the average loss while explicitly controlling the minimum decrease rate across tasks; (12) IMTL-G ([Liu et al., 2021b](#bib.bib30 "")) uses an update direction with equal projections on task gradients. IMTL-G is applied to the feature-level gradients, as was suggested by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but its performance was even worse.

Evaluation. For each experiment, we report the common evaluation metrics for each task. Since naturally MTL does not carry a single objective and since the scale of per-task metrics often varies significantly, we report two metrics that capture the overall performance: (1) 𝚫𝐦%\\mathbf{\\Delta\_{m}\\%}, the average per-task performance drop of method mm relative to the STL baseline denoted bb. Formally, Δm%\=1K​∑k\=1K(−1)δk​(Mm,k−Mb,k)/Mb,k\\Delta\_{m}\\%=\\frac{1}{K}\\sum\_{k=1}^{K}(-1)^{\\delta\_{k}}(M\_{m,k}-M\_{b,k})/M\_{b,k}, where Mb,kM\_{b,k} is the value of metric MkM\_{k} obtained by the baseline and Mm,kM\_{m,k} by the compared method. δk\=1\\delta\_{k}=1 if a higher value is better for a metric MkM\_{k} and 0 otherwise ([Maninis et al., 2019](#bib.bib33 ""); [Liu et al., 2021a](#bib.bib29 "")). (2) Mean Rank (MR): The average rank of each method across the different tasks (lower is better). A method receives the best value, MR\=1\\text{MR}=1, if it ranks first in all tasks.

### 6.1 Multi-Task Regression for QM9

We evaluate Nash-MTL on predicting 11 properties of molecules from the QM9 dataset ([Ramakrishnan et al., 2014](#bib.bib42 "")), a widely used benchmark for graph neural networks. QM9 consists of ∼130​K\\sim 130K molecules represented as graphs annotated with both node and edge features. We used the QM9 example in PyTorch Geometric ([Fey & Lenssen, 2019](#bib.bib15 "")), and use 110K molecules for training, 10K for validation, and 10K as a test set. As each task target range is at a different scale, this could be an issue for other methods that are not scale-invariant like ours. For fairness, we normalized each task target to have zero mean and unit standard deviation. We use the popular GNN model from [Gilmer et al. (2017)](#bib.bib16 ""), a network comprised of several concatenated message passing layers, which update the node features based on both node and edge features, followed by the pooling operator from [Vinyals et al. (2015)](#bib.bib55 ""). Specifically, we used the implementation from [Fey & Lenssen (2019)](#bib.bib15 ""). We train each method for 300300 epochs and search for the best learning-rate (lr) given by the Δm\\Delta\_{m} performance on the validation set. We use a learning-rate scheduler to reduce the lr once the validation Δm\\Delta\_{m} metric has stopped improving. The validation set is also used for early stopping.

Predicting molecular properties in QM9 poses a significant challenge for MTL methods because the number of tasks is large and because the loss scales vary significantly. The scale issue is only partially resolved by normalization because some tasks are easier to learn than others. Prior work found that single-task learning significantly improves performance on all targets compared to MTL methods ([Maron et al., 2019](#bib.bib34 ""); [Klicpera et al., 2020](#bib.bib25 "")).

Results are shown in Figure [3](#S4.F3 "Figure 3 ‣ 4 Related Work ‣ Multi-Task Learning as a Bargaining Game") and Table [1](#S4.T1 "Table 1 ‣ 4 Related Work ‣ Multi-Task Learning as a Bargaining Game"). Nash-MTL achieves the best performance in terms of both MR and Δm\\Delta\_{m}. Interestingly, most MTL methods fall short compared to the simple scale-invariant baseline, which ignores gradient interaction, except for IMTL-G whose performance is on par with this baseline. This result shows that the scale-invariant property of our approach can be beneficial. See Appendix [D.1](#A4.SS1 "D.1 Full Results for Multi-task Regression ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game") for the per-task evaluation results.

### 6.2 Scene Understanding

We follow the protocol of ([Liu et al., 2019b](#bib.bib32 "")) and evaluate Nash-MTL on the NYUv2 and Cityscapes datasets ([Silberman et al., 2012](#bib.bib48 ""); [Cordts et al., 2016](#bib.bib9 "")). NYUv2 is an indoor scene dataset that consists of 1449 RGBD images and dense per-pixel labeling with 13 classes. We use this dataset as a multitask learning benchmark for semantic segmentation, depth estimation, and surface normal prediction.

The CityScapes dataset ([Cordts et al., 2016](#bib.bib9 "")) contains 5000 high-resolution street-view images with dense per-pixel annotations. We use this dataset as a multitask learning benchmark for semantic segmentation and depth estimation. To speed up the training phase, all images were resized to 128×256128\\times 256. The original dataset contains 19 categories for pixel-wise semantic segmentation, together with ground-truth depth maps. For segmentation, we used a coarser version of the labels with 7 classes.

For all MTL methods, we train a Multi-Task Attention Network (MTAN) ([Liu et al., 2019b](#bib.bib32 "")) which adds an attention mechanism on top of the SegNet architecture ([Badrinarayanan et al., 2017](#bib.bib2 "")). We follow the training procedure from [Liu et al. (2019b)](#bib.bib32 ""); [Yu et al. (2020a)](#bib.bib58 ""); [Liu et al. (2021a)](#bib.bib29 ""). Each method is trained for 200200 epochs with the Adam optimizer ([Kingma & Ba, 2015](#bib.bib24 "")) and an initial learning-rate of 1​e−41e-4. The learning-rate is halved to 5​e−55e-5 after 100100 epochs. As in ([Liu et al., 2021a](#bib.bib29 "")) The STL baseline refers to training task-specific SegNet models.

The results are presented in Table [2](#S4.T2 "Table 2 ‣ 4 Related Work ‣ Multi-Task Learning as a Bargaining Game") and Table [3](#S5.T3 "Table 3 ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game"). Our method, Nash-MTL, achieves the best MR in both datasets, the best Δm\\Delta\_{m} in NYUv2 and the seconds to best Δm\\Delta\_{m} in the CityScapes experiment. Nash-MTL performance is well balanced across tasks. MGDA is primarily focused on the task of predicting surface normals and achieves poor performance on the other two tasks. The inherent biasedness of MGDA towards the task with the smallest gradient magnitude was previously discussed in [Liu et al. (2021b)](#bib.bib30 ""). We note that the optimal solution under Nash-MTL for the two tasks case is equivalent to independently normalizing each gradient and summing with equal weights. While this is a fairly simple approach for MTL, we show that it outperforms almost all the compared MTL methods on the two-tasks CityScapes benchmark.

### 6.3 Multi-Task Reinforcement Learning

We consider a multi-task RL problem and evaluate Nash-MTL on the MT10 environment from the Meta-World benchmark ([Yu et al., 2020b](#bib.bib59 "")). This benchmark involves a simulated robot trained to perform actions like pressing a button and opening a window, each action treated as a task, for a total of 10 tasks. The goal is to learn a policy that can succeed across all the diverse sets of manipulation tasks.

Table 4: MT10. Average success over 10 random seeds.

|              | Success ±\\pm SEM                  |
| ------------ | ---------------------------------- |
| MTL SAC      | 0.49±0.0730.49\\pm 0.073           |
| MTL SAC + TE | 0.54±0.0470.54\\pm 0.047           |
| MH SAC       | 0.61±0.0360.61\\pm 0.036           |
| SM           | 0.73±0.0430.73\\pm 0.043           |
| CARE         | 0.84±0.0510.84\\pm 0.051           |
| PCGrad       | 0.72±0.0220.72\\pm 0.022           |
| CAGrad       | 0.83±0.0450.83\\pm 0.045           |
| Nash-MTL     | 0.91±0.031\\mathbf{0.91\\pm 0.031} |

Following previous works on MTL-RL ([Yu et al., 2020a](#bib.bib58 ""); [Liu et al., 2021a](#bib.bib29 ""); [Sodhani et al., 2021](#bib.bib49 "")), we use Soft Actor-Critic (SAC) ([Haarnoja et al., 2018](#bib.bib18 "")) as the base RL algorithm. Along with the MTL methods (1) CAGrad ([Liu et al., 2021a](#bib.bib29 "")) and (2) PCGrad ([Yu et al., 2020a](#bib.bib58 "")) applied to a shared model SAC, we evaluate the following methods: (3) STL, one SAC model per task; (4) MTL SAC with a shared model; (5) Multi-task SAC with task encoder (MTL SAC + TE,  [Yu et al. (2020b)](#bib.bib59 "")); (6) Multi-headed SAC (MH SAC) with task-specific heads ([Yu et al., 2020b](#bib.bib59 "")); (7) Soft Modularization (SM, [Yang et al. (2020)](#bib.bib57 "")) which estimates per-task routes for different tasks in a shared model, and; (8) CARE ([Sodhani et al., 2021](#bib.bib49 "")) which utilizes language metadata and employs a mixture of encoders. We follow the same experiment setup from [Sodhani et al. (2021)](#bib.bib49 ""); [Liu et al. (2021a)](#bib.bib29 "") to train all methods over 2 million steps and report the mean success over 1010 random seeds with fixed evaluation frequency. The results are presented in Table [4](#S6.T4 "Table 4 ‣ 6.3 Multi-Task Reinforcement Learning ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game").

Nash-MTL achieves the best performance by a large margin. In addition, Nash-MTL is the only MTL method to reach the same performance as the per-task SAC STL baseline.

### 6.4 Scaling-up Nash-MTL

One of the major drawbacks of the SOTA MTL methods is that they require access to all task gradients to compute the optimal update direction ([Sener & Koltun, 2018](#bib.bib46 ""); [Yu et al., 2020a](#bib.bib58 ""); [Liu et al., 2021b](#bib.bib30 ""); [Liu et al., 2021a](#bib.bib29 "")). This requires one to perform KK backward passes at each optimization step, thus scales poorly with the number of tasks. Previous works suggested using a subset of tasks ([Liu et al., 2021a](#bib.bib29 "")) or replacing the task gradients with the feature-level gradient ([Sener & Koltun, 2018](#bib.bib46 ""); [Liu et al., 2021b](#bib.bib30 ""); [Javaloy & Valera, 2021](#bib.bib21 "")) as potential speedups. In our experiments, we found that using the feature-level gradients can greatly reduce Nash-MTL performance (Appendix [C](#A3 "Appendix C Computing Task Gradient at the Features-Level ‣ Multi-Task Learning as a Bargaining Game")). However, here we show that the simple solution of updating task weights less frequently maintains good performance while dramatically reducing the training time.

Table 5: Training runtime per episode and average success for the MT10 benchmark, computed over 1010 random seeds while varying the frequency of task weights updates in Nash-MTL.

|              | Success ±\\pm SEM        | Runtime\[Sec.\] |
| ------------ | ------------------------ | --------------- |
| MTL-SAC      | 0.49±0.0730.49\\pm 0.073 | 7.3             |
| PCGrad       | 0.72±0.0220.72\\pm 0.022 | 9.7             |
| CAGrad       | 0.83±0.0450.83\\pm 0.045 | 20.9            |
| Nash-MTL     | 0.91±0.0310.91\\pm 0.031 | 40.7            |
| Nash-MTL-50  | 0.85±0.0220.85\\pm 0.022 | 8.6             |
| Nash-MTL-100 | 0.87±0.0330.87\\pm 0.033 | 7.9             |

One approach to alleviate this issue is to update the task weights less frequently, and use these weights in subsequent steps. We evaluate this approach using the QM9 dataset and the MT10 benchmark and present the result in Figure [4](#S6.F4 "Figure 4 ‣ 6.4 Scaling-up Nash-MTL ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game") and Table [5](#S6.T5 "Table 5 ‣ 6.4 Scaling-up Nash-MTL ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game"). We denote Nash-MTL with task weight update every TT optimization steps with Nash-MTL-TT.

The results show that Nash-MTL is fairly robust to varying intervals between weights updates. While this simple approach results in a small degradation in performance, it can dramatically decrease the training time of our method. For example, on the QM9, updating the weights every 55/5050 steps results in a ×3.7/9.8\\times 3.7/9.8 speedup w.r.t updating the weights at each step. On the MT10 environment, updating the weights every 100100 steps result in ∼×10\\sim\\times 10 speedup (only ∼×1.1\\sim\\times 1.1 slower than the fastest baseline) while outperforming all other MTL baseline method (Table [5](#S6.T5 "Table 5 ‣ 6.4 Scaling-up Nash-MTL ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game")).

![Refer to caption](2202.01017v2/Figures/ablation/qm9_ablation_update_every.png)  

Figure 4: Test Δm\\Delta\_{m} for the QM9 dataset, averaged over 3 random seeds, for different intervals of task weights update.

## 7 Conclusion

In this work, we present Nash-MTL, a novel and principled approach for multitask learning. We frame the gradient combination step in MTL as a bargaining game and use the Nash bargaining solution to find the optimal update direction. We highlight the importance of the scale invariance approach for multitask learning, specifically for setups with varying loss scales and gradient magnitudes. We provide a theoretical convergence analysis for Nash-MTL, showing that it converges to a Pareto optimal and Pareto stationary points in the convex and non-convex settings, respectively. Finally, our experiments show that Nash-MTL achieves state-of-the-art results on various benchmarks across multiple domains.

## 8 Acknowledgements

This work was funded by the Israeli innovation authority through the AVATAR consortium; by the Israel Science Foundation (ISF grant 737/2018); and by an equipment grant to GC and Bar Ilan University (ISF grant 2332/18).

## References

*   Achituve et al. (2021) Achituve, I., Maron, H., and Chechik, G. Self-supervised learning for domain adaptation on point clouds. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 123–133, 2021.
*   Badrinarayanan et al. (2017) Badrinarayanan, V., Kendall, A., and Cipolla, R. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2481–2495, 2017.
*   Baek & Farias (2021) Baek, J. and Farias, V. F. Fair exploration via axiomatic bargaining. *arXiv preprint arXiv:2106.02553*, 2021.
*   Baxter (2000) Baxter, J. A model of inductive bias learning. *J. Artif. Intell. Res.*, 2000.
*   Binois et al. (2020) Binois, M., Picheny, V., Taillandier, P., and Habbal, A. The Kalai-Smorodinsky solution for many-objective Bayesian optimization. *J. Mach. Learn. Res.*, 21(150):1–42, 2020.
*   Caruana (1997) Caruana, R. Multitask learning. *Machine learning*, 28(1):41–75, 1997.
*   Chen et al. (2018) Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, pp. 794–803. PMLR, 2018.
*   Chen et al. (2020) Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., and Anguelov, D. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *ArXiv*, abs/2010.06808, 2020.
*   Cordts et al. (2016) Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., and Schiele, B. The cityscapes dataset for semantic urban scene understanding. In *Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
*   Crawshaw (2020) Crawshaw, M. Multi-task learning with deep neural networks: A survey. *arXiv preprint arXiv:2009.09796*, 2020.
*   Dagan & Volij (1993) Dagan, N. and Volij, O. The bankruptcy problem: a cooperative bargaining approach. *Mathematical Social Sciences*, 26(3):287–297, 1993.
*   Dai et al. (2016) Dai, J., He, K., and Sun, J. Instance-aware semantic segmentation via multi-task network cascades. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 3150–3158, 2016.
*   Désidéri (2012) Désidéri, J.-A. Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5-6):313–318, 2012.
*   Du et al. (2018) Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., and Lakshminarayanan, B. Adapting auxiliary losses using gradient similarity. *arXiv preprint arXiv:1812.02224*, 2018.
*   Fey & Lenssen (2019) Fey, M. and Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. In *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.
*   Gilmer et al. (2017) Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In *International conference on machine learning*, pp. 1263–1272. PMLR, 2017.
*   Grosu et al. (2002) Grosu, D., Chronopoulos, A. T., and Leung, M.-Y. Load balancing in distributed systems: An approach using cooperative games. In *Proceedings 16th International Parallel and Distributed Processing Symposium*, pp. 10–pp. IEEE, 2002.
*   Haarnoja et al. (2018) Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement, 2018.
*   Hashimoto et al. (2017) Hashimoto, K., Xiong, C., Tsuruoka, Y., and Socher, R. A joint many-task model: Growing a neural network for multiple nlp tasks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 1923–1933, 2017.
*   Jaggi (2013) Jaggi, M. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In *International Conference on Machine Learning*, pp. 427–435. PMLR, 2013.
*   Javaloy & Valera (2021) Javaloy, A. and Valera, I. Rotograd: Dynamic gradient homogenization for multi-task learning. *arXiv preprint arXiv:2103.02631*, 2021.
*   Kendall et al. (2018) Kendall, A., Gal, Y., and Cipolla, R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 7482–7491, 2018.
*   Kim (2021) Kim, S. Cooperative federated learning-based task offloading scheme for tactical edge networks. *IEEE Access*, 9:145739–145747, 2021.
*   Kingma & Ba (2015) Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *CoRR*, abs/1412.6980, 2015.
*   Klicpera et al. (2020) Klicpera, J., Groß, J., and Günnemann, S. Directional message passing for molecular graphs. *ArXiv*, abs/2003.03123, 2020.
*   Leshem & Zehavi (2011) Leshem, A. and Zehavi, E. Smart carrier sensing for distributed computation of the generalized nash bargaining solution. In *2011 17th International Conference on Digital Signal Processing (DSP)*, pp. 1–5. IEEE, 2011.
*   Lin et al. (2021) Lin, B., Ye, F., and Zhang, Y. A closer look at loss weighting in multi-task learning. *arXiv preprint arXiv:2111.10603*, 2021.
*   Lipp & Boyd (2016) Lipp, T. and Boyd, S. Variations and extension of the convex–concave procedure. *Optimization and Engineering*, 17(2):263–287, 2016.
*   Liu et al. (2021a) Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. Conflict-averse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34, 2021a.
*   Liu et al. (2021b) Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., and Zhang, W. Towards impartial multi-task learning. In *International Conference on Learning Representations*, 2021b.
*   Liu et al. (2019a) Liu, S., Davison, A., and Johns, E. Self-supervised generalisation with meta auxiliary learning. *Advances in Neural Information Processing Systems*, 32, 2019a.
*   Liu et al. (2019b) Liu, S., Johns, E., and Davison, A. J. End-to-end multi-task learning with attention. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1871–1880, 2019b.
*   Maninis et al. (2019) Maninis, K.-K., Radosavovic, I., and Kokkinos, I. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1851–1860, 2019.
*   Maron et al. (2019) Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y. Provably powerful graph networks. *arXiv preprint arXiv:1905.11136*, 2019.
*   Misra et al. (2016) Misra, I., Shrivastava, A., Gupta, A., and Hebert, M. Cross-stitch networks for multi-task learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 3994–4003, 2016.
*   Nash (1953) Nash, J. Two-person cooperative games. *Econometrica*, 21(1):128–140, 1953. ISSN 00129682, 14680262. URL [http://www.jstor.org/stable/1906951](http://www.jstor.org/stable/1906951 "").
*   Navon et al. (2021a) Navon, A., Achituve, I., Maron, H., Chechik, G., and Fetaya, E. Auxiliary learning by implicit differentiation. In *International Conference on Learning Representations (ICLR)*, 2021a.
*   Navon et al. (2021b) Navon, A., Shamsian, A., Chechik, G., and Fetaya, E. Learning the pareto front with hypernetworks. In *International Conference on Learning Representations*, 2021b. URL [https://openreview.net/forum?id=NjF772F4ZZR](https://openreview.net/forum?id=NjF772F4ZZR "").
*   Panageas et al. (2019) Panageas, I., Piliouras, G., and Wang, X. First-order methods almost always avoid saddle points: The case of vanishing step-sizes. In *Neural Information Processing Systems (NeurIPS)*, 2019.
*   Pinto & Gupta (2017) Pinto, L. and Gupta, A. Learning to push by grasping: Using multiple tasks for effective learning. In *2017 IEEE international conference on robotics and automation (ICRA)*, pp. 2161–2168. IEEE, 2017.
*   Qiao et al. (2006) Qiao, H., Rozenblit, J., Szidarovszky, F., and Yang, L. Multi-agent learning model with bargaining. In *Proceedings of the 2006 winter simulation conference*, pp. 934–940. IEEE, 2006.
*   Ramakrishnan et al. (2014) Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014.
*   Rezaee et al. (2021) Rezaee, M. J., Eshkevari, M., Saberi, M., and Hussain, O. GBK-means clustering algorithm: An improvement to the K-means algorithm based on the bargaining game. *Knowledge-Based Systems*, 213:106672, 2021.
*   Ruder (2017) Ruder, S. An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*, 2017.
*   Schaul et al. (2019) Schaul, T., Borsa, D., Modayil, J., and Pascanu, R. Ray interference: a source of plateaus in deep reinforcement learning. *arXiv preprint arXiv:1904.11455*, 2019.
*   Sener & Koltun (2018) Sener, O. and Koltun, V. Multi-task learning as multi-objective optimization. In *Advances in Neural Information Processing Systems*, pp. 527–538, 2018.
*   Shi et al. (2018) Shi, C., Wang, F., Salous, S., Zhou, J., and Hu, Z. Nash bargaining game-theoretic framework for power control in distributed multiple-radar architecture underlying wireless communication system. *Entropy*, 20(4):267, 2018.
*   Silberman et al. (2012) Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In *European conference on computer vision*, pp. 746–760. Springer, 2012.
*   Sodhani et al. (2021) Sodhani, S., Zhang, A., and Pineau, J. Multi-task reinforcement learning with context-based representations. *arXiv preprint arXiv:2102.06177*, 2021.
*   Sriperumbudur & Lanckriet (2009) Sriperumbudur, B. K. and Lanckriet, G. R. On the convergence of the concave-convex procedure. In *Nips*, volume 9, pp. 1759–1767. Citeseer, 2009.
*   Standley et al. (2020) Standley, T., Zamir, A. R., Chen, D., Guibas, L. J., Malik, J., and Savarese, S. Which tasks should be learned together in multi-task learning? In *International Conference on Machine Learning ICML*, 2020.
*   Suteu & Guo (2019) Suteu, M. and Guo, Y. Regularizing deep multi-task networks using orthogonal gradients. *arXiv preprint arXiv:1912.06844*, 2019.
*   Szép & Forgó (1985) Szép, J. and Forgó, F. *Introduction to the Theory of Games*. Springer, 1985.
*   Thomson (1994) Thomson, W. Chapter 35 cooperative models of bargaining. volume 2 of *Handbook of Game Theory with Economic Applications*, pp. 1237–1284. Elsevier, 1994.
*   Vinyals et al. (2015) Vinyals, O., Bengio, S., and Kudlur, M. Order matters: Sequence to sequence for sets. *arXiv preprint arXiv:1511.06391*, 2015.
*   Wang et al. (2020) Wang, Z., Tsvetkov, Y., Firat, O., and Cao, Y. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In *International Conference on Learning Representations*, 2020.
*   Yang et al. (2020) Yang, R., Xu, H., Wu, Y., and Wang, X. Multi-task reinforcement learning with soft modularization. *ArXiv*, abs/2003.13661, 2020.
*   Yu et al. (2020a) Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. In *Advances in Neural Information Processing Systems*, 2020a.
*   Yu et al. (2020b) Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., and Levine, S. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In *Conference on Robot Learning*, pp. 1094–1100. PMLR, 2020b.
*   Yuille & Rangarajan (2003) Yuille, A. L. and Rangarajan, A. The concave-convex procedure. *Neural computation*, 15(4):915–936, 2003.
*   Zhang et al. (2008) Zhang, Z., Shi, J., Chen, H.-H., Guizani, M., and Qiu, P. A cooperation strategy based on nash bargaining solution in cooperative relay networks. *IEEE Transactions on Vehicular Technology*, 57(4):2570–2577, 2008.
*   Zhang et al. (2014) Zhang, Z., Luo, P., Loy, C. C., and Tang, X. Facial landmark detection by deep multi-task learning. In *European conference on computer vision*, pp. 94–108. Springer, 2014.
*   Zhao et al. (2018) Zhao, X., Li, H., Shen, X., Liang, X., and Wu, Y. A modulation module for multi-task learning with applications in image retrieval. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 401–416, 2018.

## Appendix A Proofs

###### Lemma A.1.

If ℒ\\mathcal{L} is differential and L-smooth (assumption [5.3](#S5.Thmtheorem3 "Assumption 5.3. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game")) then ℒ(θ′)≤ℒ(θ)+∇ℒ(θ)⊤(θ′−θ)+L2∥θ′−θ∥2\\mathcal{L}(\\theta^{\\prime})\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)+\\frac{L}{2}\\|\\theta^{\\prime}-\\theta\\|^{2}.

###### Proof.

Fix θ,θ′∈dom⁡(ℒ)⊆ℝd\\theta,\\theta^{\\prime}\\in\\dom(\\mathcal{L})\\subseteq\\mathbb{R}^{d}. Since dom⁡(ℒ)\\dom(\\mathcal{L}) is a convex and open set, there exists ϵ\>0\\epsilon>0 such that θ+t⁡(θ′−θ)∈dom⁡(ℒ)\\theta+t(\\theta^{\\prime}-\\theta)\\in\\dom(\\mathcal{L}) for all t∈\[−ϵ,1+ϵ\]t\\in\[-\\epsilon,1+\\epsilon\]. Set ϵ\>0\\epsilon>0 to be such a number. Thus, we can define a function ℒ¯:\[−ϵ,1+ϵ\]→ℝ\\bar{\\mathcal{L}}:\[-\\epsilon,1+\\epsilon\]\\rightarrow\\mathbb{R} by ℒ¯​(t)\=ℒ⁡(θ+t⁡(θ′−θ))\\bar{\\mathcal{L}}(t)=\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta)). With this, ℒ¯​(1)\=ℒ​(θ′)\\bar{\\mathcal{L}}(1)=\\mathcal{L}(\\theta^{\\prime}), ℒ¯​(0)\=ℒ​(θ)\\bar{\\mathcal{L}}(0)=\\mathcal{L}(\\theta), and ∇ℒ¯(t)\=∇ℒ(θ+t(θ′−θ))⊤(θ′−θ)\\nabla\\bar{\\mathcal{L}}(t)=\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))^{\\top}(\\theta^{\\prime}-\\theta) for t∈\[0,1\]⊂(−ϵ,1+ϵ)t\\in\[0,1\]\\subset(-\\epsilon,1+\\epsilon). From Assumption [5.3](#S5.Thmtheorem3 "Assumption 5.3. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game"), ‖∇ℒ​(θ′)−∇ℒ​(θ)‖≤L​‖θ′−θ‖\\|\\nabla\\mathcal{L}(\\theta^{\\prime})-\\nabla\\mathcal{L}(\\theta)\\|\\leq L\\|\\theta^{\\prime}-\\theta\\|, therefore

‖∇ℒ¯​(t′)−∇ℒ¯​(t)‖\\displaystyle\\|\\nabla\\bar{\\mathcal{L}}(t^{\\prime})-\\nabla\\bar{\\mathcal{L}}(t)\\|

\=∥\[∇ℒ(θ+t′(θ′−θ))−∇ℒ(θ+t(θ′−θ))⊤(θ′−θ)∥\\displaystyle=\\|\[\\nabla\\mathcal{L}(\\theta+t^{\\prime}(\\theta^{\\prime}-\\theta))-\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))^{\\top}(\\theta^{\\prime}-\\theta)\\|

≤‖θ′−θ‖​‖∇ℒ​(θ+t′​(θ′−θ))−∇ℒ​(θ+t⁡(θ′−θ))‖\\displaystyle\\leq\\|\\theta^{\\prime}-\\theta\\|\\|\\nabla\\mathcal{L}(\\theta+t^{\\prime}(\\theta^{\\prime}-\\theta))-\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))\\|

≤L​‖θ′−θ‖​‖(t′−t)​(θ′−θ)‖\\displaystyle\\leq L\\|\\theta^{\\prime}-\\theta\\|\\|(t^{\\prime}-t)(\\theta^{\\prime}-\\theta)\\|

≤L​‖θ′−θ‖2​‖t′−t‖.\\displaystyle\\leq L\\|\\theta^{\\prime}-\\theta\\|^{2}\\|t^{\\prime}-t\\|.

Hence, ∇ℒ¯:\[0,1\]→ℝ\\nabla\\bar{\\mathcal{L}}:\[0,1\]\\rightarrow\\mathbb{R} is Lipschitz continuous, and therefore continuous. By using the fundamental theorem of calculus with the continuous function ∇ℒ¯:\[0,1\]→ℝ\\nabla\\bar{\\mathcal{L}}:\[0,1\]\\rightarrow\\mathbb{R}, ℒ⁡(θ′)\\displaystyle\\mathcal{L}(\\theta^{\\prime})

\=ℒ(θ)+∫01∇ℒ(θ+t(θ′−θ))⊤(θ′−θ)dt\\displaystyle=\\mathcal{L}(\\theta)+\\int\_{0}^{1}\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))^{\\top}(\\theta^{\\prime}-\\theta)dt

\=ℒ(θ)+∇ℒ(θ)⊤(θ′−θ)+∫01(∇ℒ(θ+t(θ′−θ))−∇ℒ(θ))⊤(θ′−θ)dt\\displaystyle=\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)+\\int\_{0}^{1}\\left(\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))-\\nabla\\mathcal{L}(\\theta)\\right)^{\\top}(\\theta^{\\prime}-\\theta)dt

≤ℒ(θ)+∇ℒ(θ)⊤(θ′−θ)+∫01∥∇ℒ(θ+t(θ′−θ))−∇ℒ(θ)∥∥θ′−θ∥dt\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)+\\int\_{0}^{1}\\|\\nabla\\mathcal{L}(\\theta+t(\\theta^{\\prime}-\\theta))-\\nabla\\mathcal{L}(\\theta)\\|\\|\\theta^{\\prime}-\\theta\\|dt

≤ℒ(θ)+∇ℒ(θ)⊤(θ′−θ)+∫01tL∥θ′−θ∥2dt\\displaystyle\\leq\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)+\\int\_{0}^{1}tL\_{\\mathcal{}}\\|\\theta^{\\prime}-\\theta\\|^{2}dt

\=ℒ(θ)+∇ℒ(θ)⊤(θ′−θ)+L2∥θ′−θ∥2.\\displaystyle=\\mathcal{L}(\\theta)+\\nabla\\mathcal{L}(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)+\\frac{L}{2}\\|\\theta^{\\prime}-\\theta\\|^{2}.

(7)

∎

Theorem (5.4). Let {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} be the sequence generated by the update rule θ(t+1)\=θ(t)−μ(t)​Δ​θ(t)\\theta^{(t+1)}=\\theta^{(t)}-\\mu^{(t)}\\Delta\\theta^{(t)} where Δ​θ(t)\=∑i\=1Kαi(t)​gi(t)\\Delta\\theta^{(t)}=\\sum\_{i=1}^{K}\\alpha^{(t)}\_{i}g\_{i}^{(t)} is the Nash bargaining solution (G(t))⊤​G(t)​α(t)\=1/α(t)(G^{(t)})^{\\top}G^{(t)}\\alpha^{(t)}=1/\\alpha^{(t)}. Set μ(t)\=mini∈\[K\]⁡1L​K​αi(t)\\mu^{(t)}=\\min\\limits\_{i\\in\[K\]}\\frac{1}{LK\\alpha^{(t)}\_{i}}. The sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} has a subsequence that converges to a Pareto stationary point θ∗\\theta^{\*}. Moreover all the loss functions (ℓ1​(θ(t)),…,ℓK​(θ(t)))(\\ell\_{1}(\\theta^{(t)}),...,\\ell\_{K}(\\theta^{(t)})) converge to (ℓ1​(θ∗),…,ℓK​(θ∗))(\\ell\_{1}(\\theta^{\*}),...,\\ell\_{K}(\\theta^{\*})). 

###### Proof.

We first note that if for some step we reach a Pareto stationary solution the algorithm halts and sequence stays fixed at that point and therefore converges; Next, we assume that we never get to an exact Pareto stationary solution at any finite step.

We note that the norm of Δ​θ(t)\\Delta\\theta^{(t)} is K\\sqrt{K} as ‖Δ​θ(t)‖2\=∑i\=1Kαi​gi⊤​Δ​θ(t)\=∑i\=1Kαi⋅1/αi\=K||\\Delta\\theta^{(t)}||^{2}=\\sum\_{i=1}^{K}\\alpha\_{i}g\_{i}^{\\top}\\Delta\\theta^{(t)}=\\sum\_{i=1}^{K}\\alpha\_{i}\\cdot 1/\\alpha\_{i}=K. For each loss ℓi\\ell\_{i} we have using Lemma [A.1](#A1.Thmtheorem1 "Lemma A.1. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game")

ℓi(θ(t+1))≤ℓi(θ(t))−μ(t)∇ℓi(θ(t))⊤Δθ(t)+L2||μ(t)Δθ(t)||2\=\\displaystyle\\ell\_{i}(\\theta^{(t+1)})\\leq\\ell\_{i}(\\theta^{(t)})-\\mu^{(t)}\\nabla\\ell\_{i}(\\theta^{(t)})^{\\top}\\Delta\\theta^{(t)}+\\frac{L}{2}||\\mu^{(t)}\\Delta\\theta^{(t)}||^{2}=

(8)

ℓi​(θ(t))−μ(t)​1αi(t)+(μ(t))2​L​K2\\displaystyle\\ell\_{i}(\\theta^{(t)})-\\mu^{(t)}\\frac{1}{\\alpha^{(t)}\_{i}}+\\frac{(\\mu^{(t)})^{2}LK}{2}

(9)

\=ℓi​(θ(t))−μ(t)αi(t)+μ(t)2​minj​1αj(t)≤ℓi​(θ(t))−μ(t)2​αi(t)<ℓi​(θ(t))\\displaystyle=\\ell\_{i}(\\theta^{(t)})-\\frac{\\mu^{(t)}}{\\alpha^{(t)}\_{i}}+\\frac{\\mu^{(t)}}{2}\\min\_{j}\\frac{1}{\\alpha\_{j}^{(t)}}\\leq\\ell\_{i}(\\theta^{(t)})-\\frac{\\mu^{(t)}}{2\\alpha^{(t)}\_{i}}<\\ell\_{i}(\\theta^{(t)})

(10)

  

This shows that our update decreases all the loss functions. We can average over inequality [9](#A1.E9 "Equation 9 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game") over all losses and get for ℒ⁡(θ)\=1K​∑i\=1Kℓi​(θ)\\mathcal{L}(\\theta)=\\frac{1}{K}\\sum\_{i=1}^{K}\\ell\_{i}(\\theta):

ℒ⁡(θ(t+1))≤ℒ⁡(θ(t))−μ(t)​1K​∑i\=1K1αi(t)+(μ(t))2​L​K2≤ℒ⁡(θ(t))−L​K​(μ(t))2+(μ(t))2​L​K2\=ℒ⁡(θ(t))−L​K​(μ(t))22.\\displaystyle\\mathcal{L}(\\theta^{(t+1)})\\leq\\mathcal{L}(\\theta^{(t)})-\\mu^{(t)}\\frac{1}{K}\\sum\_{i=1}^{K}\\frac{1}{\\alpha^{(t)}\_{i}}+\\frac{(\\mu^{(t)})^{2}LK}{2}\\leq\\mathcal{L}(\\theta^{(t)})-LK(\\mu^{(t)})^{2}+\\frac{(\\mu^{(t)})^{2}LK}{2}=\\mathcal{L}(\\theta^{(t)})-\\frac{LK(\\mu^{(t)})^{2}}{2}.

(11)

From this we can conclude that ∑τ\=1tL​K​(μ(τ))22≤ℒ⁡(θ1)−ℒ⁡(θ(t+1))\\sum\_{\\tau=1}^{t}\\frac{LK(\\mu^{(\\tau)})^{2}}{2}\\leq\\mathcal{L}(\\theta\_{1})-\\mathcal{L}(\\theta^{(t+1)}). As ℒ⁡(θ(t))\\mathcal{L}(\\theta^{(t)}) is bounded below we must have that the infinite series ∑t\=1∞L​K​(μ(t))22<∞\\sum\_{t=1}^{\\infty}\\frac{LK(\\mu^{(t)})^{2}}{2}<\\infty, and also μ(t)→0\\mu^{(t)}\\to 0. It follows that mini∈\[K\]⁡1/αi(t)→0\\min\_{i\\in\[K\]}1/\\alpha^{(t)}\_{i}\\to 0 and therefore ‖α(t)‖→∞||\\alpha^{(t)}||\\to\\infty.  

We will now show that ‖1/α(t)‖||1/\\alpha^{(t)}|| is bounded for t→∞t\\to\\infty. As the sequence ℒ⁡(θ(t))\\mathcal{L}(\\theta^{(t)}) is decreasing we have that the sequence θ(t)\\theta^{(t)} is in the sublevel set {θ:ℒ⁡(θ)≤ℒ⁡(θ0)}\\{\\theta:\\mathcal{L}(\\theta)\\leq\\mathcal{L}(\\theta\_{0})\\} which is closed and bounded and therefore compact. If follows that there exists M<∞M<\\infty such that ‖gi(t)‖≤M||g^{(t)}\_{i}||\\leq M for all tt and i∈\[K\]i\\in\[K\]. We have for all ii and tt, |1/αi(t)|\=|(gi(t))T​θ(t)|≤K​‖gi(t)‖≤K​M<∞|1/\\alpha^{(t)}\_{i}|=|(g^{(t)}\_{i})^{T}\\theta^{(t)}|\\leq\\sqrt{K}||g^{(t)}\_{i}||\\leq\\sqrt{K}M<\\infty, and so ‖1/α(t)‖||1/\\alpha^{(t)}|| is bounded. Combining these two results we have ‖1/α(t)‖≥σK​((G(t))⊤​G(t))​‖α(t)‖||1/\\alpha^{(t)}||\\geq\\sigma\_{K}((G^{(t)})^{\\top}G^{(t)})||\\alpha^{(t)}|| where σK​((G(t))⊤​G(t))\\sigma\_{K}((G^{(t)})^{\\top}G^{(t)}) is the smallest singular value of (G(t))⊤​G(t)(G^{(t)})^{\\top}G^{(t)}. Since the norm of α(t)\\alpha^{(t)} goes to infinity and the norm 1/α(t)1/\\alpha^{(t)} is bounded, it follows that σK​((G(t))⊤​G(t))→0\\sigma\_{K}((G^{(t)})^{\\top}G^{(t)})\\to 0.

Now, since {θ:ℒ⁡(θ)≤ℒ​θ0}\\{\\theta:\\mathcal{L}(\\theta)\\leq\\mathcal{L}{\\theta\_{0}}\\} is compact there exists a subsequence θ(tj)\\theta^{(t\_{j})} that converges to some point θ∗\\theta^{\*}. As σK​((G(t))T​G(t))→0\\sigma\_{K}((G^{(t)})^{T}G^{(t)})\\rightarrow 0 we have from continuity that σK​(G∗⊤​G∗)\=0\\sigma\_{K}(G\_{\*}^{\\top}G\_{\*})=0 where G∗G\_{\*} is the matrix of gradients at θ∗\\theta^{\*}. This means that the gradients at θ\\theta are linearly dependent and therefore θ∗\\theta^{\*} is Pareto stationary by assumption [5.1](#S5.Thmtheorem1 "Assumption 5.1. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game"). As for all ii the sequence {ℓi​(θ(t))}t\=1∞\\{\\ell\_{i}(\\theta^{(t)})\\}\_{t=1}^{\\infty} is monotonically decreasing and bounded below they all converges. Since ℓi​(θ∗)\\ell\_{i}(\\theta^{\*}) is the limit of a subsequence we get that ℓi​(θ(t))→t→∞ℓi​(θ∗)\\ell\_{i}(\\theta^{(t)})\\xrightarrow{t\\rightarrow\\infty}\\ell\_{i}(\\theta^{\*}).

∎

We now show that if we add a convexity assumption then we can prove convergence to the Pareto front.

Theorem (5.5). Let {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} be the sequence generated by the update rule θ(t+1)\=θ(t)−μ(t)​Δ​θ(t)\\theta^{(t+1)}=\\theta^{(t)}-\\mu^{(t)}\\Delta\\theta^{(t)} where Δ​θ(t)\=∑i\=1Kαi(t)​gi(t)\\Delta\\theta^{(t)}=\\sum\_{i=1}^{K}\\alpha^{(t)}\_{i}g\_{i}^{(t)} is the Nash bargaining solution (G(t))⊤​G(t)​α(t)\=1/α(t)(G^{(t)})^{\\top}G^{(t)}\\alpha^{(t)}=1/\\alpha^{(t)}. Set μ(t)\=mini∈\[K\]⁡1L​K​αi(t)\\mu^{(t)}=\\min\\limits\_{i\\in\[K\]}\\frac{1}{LK\\alpha^{(t)}\_{i}}. If we also assume that all the loss functions are convex then the sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} converges to a Pareto optimal point θ∗\\theta^{\*}. 

###### Proof.

We note that this proof uses intermediate results from the proof of theorem [5.4](#S5.Thmtheorem4 "Theorem 5.4. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game"). Given theorem [5.4](#S5.Thmtheorem4 "Theorem 5.4. ‣ 5 Analysis ‣ Multi-Task Learning as a Bargaining Game") it suffices to prove that the sequence {θ(t)}t\=1∞\\{\\theta^{(t)}\\}\_{t=1}^{\\infty} converges, that would mean it converges to the partial limit θ∗\\theta^{\*} that is Pareto stationary, and from convexity it would be Pareto optimal (as the optimizer of the convex combination of losses). For a convex and differential loss function, we have

ℓ(θ′)≥ℓ(θ)+∇ℓ(θ)⊤(θ′−θ)\\ell(\\theta^{\\prime})\\geq\\ell(\\theta)+\\nabla\\ell(\\theta)^{\\top}(\\theta^{\\prime}-\\theta)

(12)

We can bound

‖θ(t+1)−θ∗‖2\\displaystyle||\\theta^{(t+1)}-\\theta^{\*}||^{2}

\=‖θ(t)−μ(t)​Δ​θ(t)−θ∗‖2\\displaystyle=||\\theta^{(t)}-\\mu^{(t)}\\Delta\\theta^{(t)}-\\theta^{\*}||^{2}

(13)

\=‖θ(t)−θ∗‖2+(μ(t))2​‖Δ​θ(t)‖2−2​μ(t)​(Δ​θ(t))⊤​(θ(t)−θ∗)\\displaystyle=||\\theta^{(t)}-\\theta^{\*}||^{2}+(\\mu^{(t)})^{2}||\\Delta\\theta^{(t)}||^{2}-2\\mu^{(t)}(\\Delta\\theta^{(t)})^{\\top}(\\theta^{(t)}-\\theta^{\*})

(14)

\=‖θ(t)−θ∗‖2+(μ(t))2​K−2​μ(t)​∑iαi(t)​(gi(t))⊤​(θ(t)−θ∗)\\displaystyle=||\\theta^{(t)}-\\theta^{\*}||^{2}+(\\mu^{(t)})^{2}K-2\\mu^{(t)}\\sum\_{i}\\alpha^{(t)}\_{i}(g^{(t)}\_{i})^{\\top}(\\theta^{(t)}-\\theta^{\*})

(15)

≤‖θ(t)−θ∗‖2+(μ(t))2​K+2​μ(t)​∑iαi(t)​(ℓi​(θ∗)−ℓi​(θ(t)))\\displaystyle\\leq||\\theta^{(t)}-\\theta^{\*}||^{2}+(\\mu^{(t)})^{2}K+2\\mu^{(t)}\\sum\_{i}\\alpha^{(t)}\_{i}(\\ell\_{i}(\\theta^{\*})-\\ell\_{i}(\\theta^{(t)}))

(16)

≤‖θ(t)−θ∗‖2+(μ(t))2​K+2​μ(t)​∑iαi(t)​(ℓi​(θ(t+1))−ℓi​(θ(t)))\\displaystyle\\leq||\\theta^{(t)}-\\theta^{\*}||^{2}+(\\mu^{(t)})^{2}K+2\\mu^{(t)}\\sum\_{i}\\alpha^{(t)}\_{i}(\\ell\_{i}(\\theta^{(t+1)})-\\ell\_{i}(\\theta^{(t)}))

(17)

≤‖θ(t)−θ∗‖2+(μ(t))2​K−2​μ(t)​∑iαi(t)​μ(t)2​αi(t)\\displaystyle\\leq||\\theta^{(t)}-\\theta^{\*}||^{2}+(\\mu^{(t)})^{2}K-2\\mu^{(t)}\\sum\_{i}\\alpha^{(t)}\_{i}\\frac{\\mu^{(t)}}{2\\alpha^{(t)}\_{i}}

(18)

\=‖θ(t)−θ∗‖2\\displaystyle=||\\theta^{(t)}-\\theta^{\*}||^{2}

(19)

In Eq. [15](#A1.E15 "Equation 15 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game") we use the definition of Δ​θ(t)\\Delta\\theta^{(t)} and the fact that its norm equals K\\sqrt{K}. In Eq. [16](#A1.E16 "Equation 16 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game") we use convexity and Eq. [12](#A1.E12 "Equation 12 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game"). Eq. [17](#A1.E17 "Equation 17 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game") uses the fact that we show the losses are monotonically decreasing and converging to ℓi​(θ∗)\\ell\_{i}(\\theta^{\*}). In Eq. [18](#A1.E18 "Equation 18 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game") we use Eq. [10](#A1.E10 "Equation 10 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game").

We have that the sequence ‖θ(t)−θ∗‖||\\theta^{(t)}-\\theta^{\*}|| is monotonically decreasing and bounded below by zero. Also, it has a subsequence that converges to zero, and so it must hold that the sequence ‖θ(t)−θ∗‖||\\theta^{(t)}-\\theta^{\*}|| also converge to zero, or equivalently θ(t)→θ∗\\theta^{(t)}\\to\\theta^{\*}.

∎

Proposition (3.1). Denote the objective for the optimization problem in Eq. [5](#S3.E5 "Equation 5 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") by ϕ⁡(α)\=∑iβi​(α)+φ⁡(α)\\phi(\\alpha)=\\sum\_{i}\\beta\_{i}(\\alpha)+\\varphi(\\alpha). Then, ϕ⁡(α(τ+1))≤ϕ⁡(α(τ))\\phi\\left(\\alpha^{(\\tau+1)}\\right)\\leq\\phi\\left(\\alpha^{(\\tau)}\\right) for all τ≥1\\tau\\geq 1. 

###### Proof.

In our concave-convex procedure, we use the following linearization at the τ\\tau-th iteration:

φ\~τ(α)\=φ(α(τ))+∇φ(α(τ))⊤(α−α(τ)).\\tilde{\\varphi}\_{\\tau}(\\alpha)=\\varphi(\\alpha^{(\\tau)})+\\nabla\\varphi(\\alpha^{(\\tau)})^{\\top}(\\alpha-\\alpha^{(\\tau)}).

Then, φ\~τ​(α(τ))\=φ⁡(α(τ)).\\displaystyle\\tilde{\\varphi}\_{\\tau}(\\alpha^{(\\tau)})=\\varphi(\\alpha^{(\\tau)}).

(20)

Moreover, since φ\\varphi is concave and differentiable, we have that

φ(α(τ+1))≤φ(α(τ))+∇φ(α(τ))⊤(α(τ+1)−α(τ))\=φ\~τ(α(τ+1)).\\displaystyle\\varphi(\\alpha^{(\\tau+1)})\\leq\\varphi(\\alpha^{(\\tau)})+\\nabla\\varphi(\\alpha^{(\\tau)})^{\\top}(\\alpha^{(\\tau+1)}-\\alpha^{(\\tau)})=\\tilde{\\varphi}\_{\\tau}(\\alpha^{(\\tau+1)}).

(21)

Furthermore, since we minimize the convex objective ∑iβi​(α)+φ\~​(α)\\sum\_{i}\\beta\_{i}(\\alpha)+\\tilde{\\varphi}(\\alpha) at each iteration of our concave-convex procedure (in the convex feasible set), ∑iβi​(α(τ))+φ\~τ​(α(τ))≥∑iβi​(α(τ+1))+φ\~τ​(α(τ+1)).\\displaystyle\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau)})+\\tilde{\\varphi}\_{\\tau}(\\alpha^{(\\tau)})\\geq\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau+1)})+\\tilde{\\varphi}\_{\\tau}(\\alpha^{(\\tau+1)}).

(22)

Using Eq. [20](#A1.E20 "Equation 20 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game")–Eq. [22](#A1.E22 "Equation 22 ‣ Proof. ‣ Appendix A Proofs ‣ Multi-Task Learning as a Bargaining Game"), we have that

ϕ⁡(α(τ))\=∑iβi​(α(τ))+φ⁡(α(τCLOSE)\=∑iβi​(α(τ))+φ\~τ​(α(τ))\\displaystyle\\phi(\\alpha^{(\\tau)})=\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau)})+\\varphi(\\alpha^{(\\tau})=\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau)})+\\tilde{\\varphi}\_{\\tau}(\\alpha^{(\\tau)})

≥∑iβi​(α(τ+1))+φ\~t​(α(τ+1))\\displaystyle\\geq\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau+1)})+\\tilde{\\varphi}\_{t}(\\alpha^{(\\tau+1)})

≥∑iβi​(α(τ+1))+φ⁡(α(τ+1))\=ϕ⁡(α(τ+1)).\\displaystyle\\geq\\sum\_{i}\\beta\_{i}(\\alpha^{(\\tau+1)})+\\varphi(\\alpha^{(\\tau+1)})=\\phi(\\alpha^{(\\tau+1)}).

This proves the statement.

∎

## Appendix B Experimental Details

We provide here full experimental details for all experiments described in the main text.

Implementation Details. We apply all gradient manipulation methods to the gradients of the shared weights, with the exception of IMTL-G, which was applied to the feature-level gradients, as was originally proposed by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but it did not perform as well. We set the CAGrad’s cc hyperparameter to 0.4, which was reported to yield the best performance for NYUv2 and Cityscapes ([Liu et al., 2021a](#bib.bib29 "")). For DWA ([Liu et al., 2019b](#bib.bib32 "")) we set the temperature hyperparameter to 22 which was found empirically to be optimum across all architectures. For RLW ([Lin et al., 2021](#bib.bib27 "")) we sample the weights from a normal distribution.

QM9. We adapt the QM9 example in PyTorch Geometric ([Fey & Lenssen, 2019](#bib.bib15 "")), and train the popular GNN model from [Gilmer et al. (2017)](#bib.bib16 ""). We use the publicly available11 1 [https://github.com/pyg-team/pytorch\_geometric/blob/master/examples/qm9\_nn\_conv.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py "") implementation, the implementation is provided by [Fey & Lenssen (2019)](#bib.bib15 ""). We use 110K molecules for training, 10K for validation, and 10K as a test set. Each task’s targets are normalized to have zero mean and unit standard deviation. We train each method for 300300 epochs with batch-size of 120120 and search for learning-rate (lr) in {1​e−3,5​e−4,1​e−4}\\{1e-3,5e-4,1e-4\\}. We use a ReduceOnPlateau scheduler to decrease the lr when the validation Δm\\Delta\_{m} metric stops improving. Additionally, we use the validation Δm\\Delta\_{m} for early stopping.

Scene Understanding. We follow the training and evaluation procedure used in previous work on MTL ([Liu et al., 2019b](#bib.bib32 ""); [Yu et al., 2020a](#bib.bib58 ""); [Liu et al., 2021a](#bib.bib29 "")). However, unlike ([Liu et al., 2019b](#bib.bib32 "")), we add data augmentations (DA) during training for all the compared methods, similar to ([Liu et al., 2021a](#bib.bib29 ""); [Liu et al., 2021b](#bib.bib30 "")). We train each method for 200200 epochs with an initial learning-rate of 1​e−41e-4. The learning-rate is reduced to 5​e−55e-5 after 100100 epochs. For MTL methods, we train a Multi-Task Attention Network (MTAN) ([Liu et al., 2019b](#bib.bib32 "")) built upon SegNet ([Badrinarayanan et al., 2017](#bib.bib2 "")). Similar to previous works ([Liu et al., 2021a](#bib.bib29 "")), the STL baseline refers to training task-specific SegNet models. We use a batch size of 2 and 8 for NYUv2 and CityScapes respectively. To align with previous work on MTL [Liu et al. (2019b)](#bib.bib32 ""); [Yu et al. (2020a)](#bib.bib58 ""); [Liu et al. (2021a)](#bib.bib29 "") we report the test performance averaged over the last 1010 epochs.

MT10. Following previous works ([Yu et al., 2020a](#bib.bib58 ""); [Liu et al., 2021a](#bib.bib29 ""); [Sodhani et al., 2021](#bib.bib49 "")), we use multitask Soft Actor-Critic (SAC) ([Haarnoja et al., 2018](#bib.bib18 "")) as the base RL algorithm for PCGrad, CAGrad, and Nash-MTL. We follow the same experiment setup from and evaluation protocol as in [Sodhani et al. (2021)](#bib.bib49 ""); [Liu et al. (2021a)](#bib.bib29 ""). Each method is trained over 2 million steps with a batch size of 1280. The agent is evaluated once every 1010K environment steps to obtain the average success over tasks. The reported success rate for the agent is the best average performance over all evaluation steps. We repeat this procedure over 1010 random seeds, and the performance of each method is obtained by averaging the mean success over all random seeds. For all Nash-MTL experiments, we use a single CCP step in order to speed up computation.

![Refer to caption](2202.01017v2/Figures/toy/average_loss_3d.png)

(a) Average loss

![Refer to caption](2202.01017v2/Figures/toy/loss_1_3d.png)

(b) ℓ1\\ell\_{1}

![Refer to caption](2202.01017v2/Figures/toy/loss_2_3d.png)

(c) ℓ2\\ell\_{2}

Figure 5: Illustrative example. Visualization of the loss surfaces in our illustrative example of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Multi-Task Learning as a Bargaining Game")

Illustrative Example. We provide here the details for the illustrative example of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Multi-Task Learning as a Bargaining Game"). We use a slightly modified version of the illustrative example in ([Liu et al., 2021a](#bib.bib29 "")). We first present the learning problem from  ([Liu et al., 2021a](#bib.bib29 "")): Let θ\=(θ1,θ2)∈ℝ2\\theta=(\\theta\_{1},\\theta\_{2})\\in\\mathbb{R}^{2}, and consider the following objectives:

ℓ\~1​(θ)\=c1​(θ)​f1​(θ)+c2​(θ)​g1​(θ)andℓ\~2​(θ)\=c1​(θ)​f2​(θ)+c2​(θ)​g2​(θ),where\\displaystyle\\tilde{\\ell}\_{1}(\\theta)=c\_{1}(\\theta)f\_{1}(\\theta)+c\_{2}(\\theta)g\_{1}(\\theta)\\quad\\text{and}\\quad\\tilde{\\ell}\_{2}(\\theta)=c\_{1}(\\theta)f\_{2}(\\theta)+c\_{2}(\\theta)g\_{2}(\\theta),\\text{where}

f1​(θ)\=log⁡(max⁡(|0.5​(−θ1−7)−tanh⁡(−θ2)|,5​e−6))+6,\\displaystyle f\_{1}(\\theta)=\\log(\\max(|0.5(-\\theta\_{1}-7)-\\tanh(-\\theta\_{2})|,5e-6))+6, f2​(θ)\=log⁡(max⁡(|0.5​(−θ1+3)−tanh⁡(−θ2)+2|,5​e−6))+6,\\displaystyle f\_{2}(\\theta)=\\log(\\max(|0.5(-\\theta\_{1}+3)-\\tanh(-\\theta\_{2})+2|,5e-6))+6, g1​(θ)\=((−θ1+7)2+0.1⋅(−θ2−8)2)/10−20,\\displaystyle g\_{1}(\\theta)=((-\\theta\_{1}+7)^{2}+0.1\\cdot(-\\theta\_{2}-8)^{2})/10-20, g2​(θ)\=((−θ1−7)2+0.1⋅(−θ2−8)2)/10−20,\\displaystyle g\_{2}(\\theta)=((-\\theta\_{1}-7)^{2}+0.1\\cdot(-\\theta\_{2}-8)^{2})/10-20, c1​(θ)\=max⁡(tanh⁡(0.5​θ2),0)andc2​(θ)\=max⁡(tanh⁡(−0.5​θ2),0)\\displaystyle c\_{1}(\\theta)=\\max(\\tanh(0.5\\theta\_{2}),0)\\quad\\text{and}\\quad c\_{2}(\\theta)=\\max(\\tanh(-0.5\\theta\_{2}),0)

We now set ℓ1\=0.1⋅ℓ\~1\\ell\_{1}=0.1\\cdot\\tilde{\\ell}\_{1} and ℓ2\=ℓ\~2\\ell\_{2}=\\tilde{\\ell}\_{2} as our objectives, see Figure [5](#A2.F5 "Figure 5 ‣ Appendix B Experimental Details ‣ Multi-Task Learning as a Bargaining Game"). We use five different initialization points {(−8.5,7.5),(0.0,0.0),(9.0,9.0),(−7.5,−0.5),(9,−1.0)}\\{(-8.5,7.5),(0.0,0.0),(9.0,9.0),(-7.5,-0.5),(9,-1.0)\\}. We use the Adam optimizer and train each method for 35K iteration with learning rate of 1​e−31e-3.

## Appendix C Computing Task Gradient at the Features-Level

One common approach for speeding and scaling up MTL methods is using feature-level gradients (from the representation layer) as a surrogate for the task-level gradients computed over the entire shared backbone ([Sener & Koltun, 2018](#bib.bib46 ""); [Liu et al., 2021b](#bib.bib30 ""); [Javaloy & Valera, 2021](#bib.bib21 "")). In this section we evaluate Nash-MTL while using the feature-level gradients for computing the Nash bargaining solution. On the QM9 dataset, we found this approach to accelerate training by ∼×6\\sim\\times 6. However, this acceleration method greatly hurts the performance of Nash-MTL, yielding a test Δm\\Delta\_{m} of 179.2179.2 (compared to 62.062.0 when using full gradients). This result is not surprising, since we are mainly interested in the inner products of gradients. Consider gi⊤​gj\=(∇θz​∇zℓi)⊤​∇θz​∇zℓjg\_{i}^{\\top}g\_{j}=(\\nabla\_{\\theta}z\\nabla\_{z}\\ell\_{i})^{\\top}\\nabla\_{\\theta}z\\nabla\_{z}\\ell\_{j}, where zz is the feature representation and θ\\theta the shared parameters vector. We see that for ∇zℓi⊤​∇zℓj\\nabla\_{z}\\ell\_{i}^{\\top}\\nabla\_{z}\\ell\_{j} to accurately approximate gi⊤​gjg\_{i}^{\\top}g\_{j} we need ∇θz⊤​∇θz≈I\\nabla\_{\\theta}z^{\\top}\\nabla\_{\\theta}z\\approx I which is a strong and restricting requirement.

![Refer to caption](2202.01017v2/Figures/ablation/nyu_delta_ccp_steps.png)

Figure 6: NYUv2. The mean and standard divination of test Δm\\Delta\_{m} throughout the training process, for Nash-MTL with 1, 20, and 40 CCP steps.

## Appendix D Additional Experiments

### D.1 Full Results for Multi-task Regression

We provide here the full results for the QM9 experiment of Section [6.1](#S6.SS1 "6.1 Multi-Task Regression for QM9 ‣ 6 Experiments ‣ Multi-Task Learning as a Bargaining Game"). The results for all methods over all 1111 tasks are presented in Table [6](#A4.T6 "Table 6 ‣ D.1 Full Results for Multi-task Regression ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game"). Nash-MTL achieves the best Δm\\Delta\_{m} and MR performance. Despite being a simple approach, SI performs well compared to more sophisticated baselines. It achieves the third/second best Δm\\Delta\_{m} and MR respectively. The other scale-invariant method, IMTL-G, also performs well in this learning setup.

Table 6: QM9. Test performance averaged over 3 random seeds.

|          | μ\\mu                | α\\alpha             | ϵHOMO\\epsilon\_{\\text{HOMO}} | ϵLUMO\\epsilon\_{\\text{LUMO}} | ⟨R2⟩\\langle R^{2}\\rangle | ZPVE               | U0U\_{0}           | UU                   | HH                   | GG                   | cvc\_{v}             |                  |                    |
| -------- | -------------------- | -------------------- | ------------------------------ | ------------------------------ | -------------------------- | ------------------ | ------------------ | -------------------- | -------------------- | -------------------- | -------------------- | ---------------- | ------------------ |
|          |                      |                      |                                |                                |                            |                    |                    |                      |                      |                      |                      |                  |                    |
|          |                      |                      |                                |                                |                            |                    |                    |                      |                      |                      |                      |                  |                    |
| SI       | 0.309                | 0.345                | 149.8                          | 135.7                          | 1.00\\mathbf{1.00}         | 4.50\\mathbf{4.50} | 55.3\\mathbf{55.3} | 55.75\\mathbf{55.75} | 55.82\\mathbf{55.82} | 55.27\\mathbf{55.27} | 0.112                | 4.0              | 77.8               |
| RLW      | 0.113                | 0.340                | 76.95                          | 92.76                          | 5.86                       | 15.46              | 156.3              | 157.1                | 157.6                | 153.0                | 0.137                | 8.2              | 203.8              |
| DWA      | 0.107                | 0.325                | 74.06                          | 90.61                          | 5.09                       | 13.99              | 142.3              | 143.0                | 143.4                | 139.3                | 0.125                | 6.4              | 175.3              |
| UW       | 0.386                | 0.425                | 166.2                          | 155.8                          | 1.06                       | 4.99               | 66.4               | 66.78                | 66.80                | 66.24                | 0.122                | 5.3              | 108.0              |
| MGDA     | 0.217                | 0.368                | 126.8                          | 104.6                          | 3.22                       | 5.69               | 88.37              | 89.4                 | 89.32                | 88.01                | 0.120                | 5.9              | 120.5              |
| PCGrad   | 0.106                | 0.293                | 75.85                          | 88.33                          | 3.94                       | 9.15               | 116.36             | 116.8                | 117.2                | 114.5                | 0.110                | 5.0              | 125.7              |
| CAGrad   | 0.118                | 0.321                | 83.51                          | 94.81                          | 3.21                       | 6.93               | 113.99             | 114.3                | 114.5                | 112.3                | 0.116                | 5.7              | 112.8              |
| IMTL-G   | 0.136                | 0.287                | 98.31                          | 93.96                          | 1.75                       | 5.69               | 101.4              | 102.4                | 102.0                | 100.1                | 0.096                | 4.7              | 77.2               |
| Nash-MTL | 0.102\\mathbf{0.102} | 0.248\\mathbf{0.248} | 82.95                          | 81.89\\mathbf{81.89}           | 2.42                       | 5.38               | 74.5               | 75.02                | 75.10                | 74.16                | 0.093\\mathbf{0.093} | 2.5\\mathbf{2.5} | 62.0\\mathbf{62.0} |

### D.2 Effect of the Number of CCP steps

In this section, we investigate the effect of varying the number of CCP steps in our efficient approximation to G⊤​G​α\=1/αG^{\\top}G\\alpha=1/\\alpha (presented in Section [3.2](#S3.SS2 "3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game")). We use the NYUv2 dataset and train Nash-MTL with CCP sequences of 1, 20, and 40 steps at each (parameters) optimization step.

We found that increasing the CCP sequence improves the approximation to the optimal α\\alpha. Using a single CCP iteration results with G⊤​G​α≈1/αG^{\\top}G\\alpha\\approx 1/\\alpha in 91.5%91.5\\% of the optimization steps, whereas increasing the number of iterations to 20 increases the proportion of optimal solutions to 93.5%93.5\\%. However, we found the improved solution to have no significant improvement in MTL performance. Figure [6](#A3.F6 "Figure 6 ‣ Appendix C Computing Task Gradient at the Features-Level ‣ Multi-Task Learning as a Bargaining Game") presents the test Δm\\Delta\_{m} throughout the training process.

### D.3 Modifying the CCP Objective

In this section we examine the effect of changing the objective of the CCP procedure described in [3.2](#S3.SS2 "3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") (Eq. [5](#S3.E5 "Equation 5 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game")). Here we first solve the convex optimization problem of Eq. [4](#S3.E4 "Equation 4 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game") to obtain α0\\alpha\_{0}. If G⊤​G​α0≈1/α0G^{\\top}G\\alpha\_{0}\\approx 1/\\alpha\_{0} we stop. Else we use the CCP procedure with objective φ⁡(α)\\varphi(\\alpha), starting at α0\\alpha\_{0} (dropping the addition ∑iβi\\sum\_{i}\\beta\_{i} term from Eq. [5](#S3.E5 "Equation 5 ‣ 3.2 Solving 𝐆^⊤⁢𝐆⁢𝛼=1/𝛼 ‣ 3 Method ‣ Multi-Task Learning as a Bargaining Game")). While this objective is more natural, in practice we observe a performance degradation in terms of MTL performance. We obtain Δm\=64.4\\Delta\_{m}=64.4 for the QM9 dataset (vs. 6262 reported in the paper), Δm\=−3.5\\Delta\_{m}=-3.5 (vs. −4-4) for NYUv2 and Δm\=8.8\\Delta\_{m}=8.8 (vs. 6.86.8) for Cityscapes.

### D.4 Visualizing Task Weights

Our method, Nash-MTL, can essentially be viewed as a principled approach for producing dynamic task weights. Here we visualize these task weights throughout the training process using the NYUv2 dataset (Figure [7](#A4.F7 "Figure 7 ‣ D.4 Visualizing Task Weights ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game")) and the MT10 dataset (Figure [7(b)](#A4.F7.sf2 "Figure 7(b) ‣ Figure 7 ‣ D.4 Visualizing Task Weights ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game")).

![Refer to caption](2202.01017v2/Figures/nyu/nyu_weights.png)

(a) NYUv2

![Refer to caption](2202.01017v2/Figures/mt10/mt10_we_100_weights.png)

(b) MT10

Figure 7: Task Weights. Task weights obtained from Nash-MTL throughout the optimization process, for (a) NYUv2, and; (b) MT10 with weight update frequency of 100. For better visualization, each point corresponds to a moving average with window size 200.

### D.5 Verifying the Task Independence Assumption

Here we provide an empirical justification for our assumption in Section [3](#S3 "3 Method ‣ Multi-Task Learning as a Bargaining Game") which we state here once again: we assume that the task gradients are linearly independent for each point θ\\theta that is not Pareto stationary. To investigate whether this assumption holds in our experiments, we observe the smallest singular value of gradients Gram matrix σK​(G⊤​G)\\sigma\_{K}(G^{\\top}G). The results are presented in Figure [8](#A4.F8 "Figure 8 ‣ D.5 Verifying the Task Independence Assumption ‣ Appendix D Additional Experiments ‣ Multi-Task Learning as a Bargaining Game"). We see that for both datasets the σk\\sigma\_{k} decreases as the learning progresses. For the NYUv2 experiment, the smallest singular value remains fairly large throughout the entire training process. On the QM9 dataset, σK\\sigma\_{K} decreases more significantly, to around ∼1​e−8\\sim 1e-8.

![Refer to caption](2202.01017v2/Figures/nyu/nyu_sigma_min.png)

(a) NYUv2

![Refer to caption](2202.01017v2/Figures/qm9/qm9_sigma_min.png)

(b) QM9

Figure 8: Smallest singular value of G⊤​GG^{\\top}G throughout the training process.

Table 7: QM9. Runtime per epoch in minutes.

|             | Runtime \[Min.\] |
| ----------- | ---------------- |
| LS          | 0.540.54         |
| MGDA        | 7.257.25         |
| PCGrad      | 7.477.47         |
| CAGrad      | 6.856.85         |
| Nash-MTL    | 6.766.76         |
| Nash-MTL-5  | 1.811.81         |
| Nash-MTL-50 | 0.690.69         |

[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")