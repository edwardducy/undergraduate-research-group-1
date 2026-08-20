# Multi-Task Learning as a Bargaining Game

Aviv Navon \* 1 Aviv Shamsian \* 1 Idan Achituve <sup>1</sup> Haggai Maron <sup>2</sup> Kenji Kawaguchi <sup>3</sup> Gal Chechik 1 2 Ethan Fetaya <sup>1</sup>

## Abstract

In Multi-task learning (MTL), a joint model is trained to simultaneously make predictions for several tasks. Joint training reduces computation costs and improves data efficiency; however, since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance than its corresponding single-task counterparts. A common method for alleviating this issue is to combine per-task gradients into a joint update direction using a particular heuristic. In this paper, we propose viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction of parameter update. Under certain assumptions, the bargaining problem has a unique solution, known as the *Nash Bargaining Solution*, which we propose to use as a principled approach to multi-task learning. We describe a new MTL optimization procedure, Nash-MTL, and derive theoretical guarantees for its convergence. Empirically, we show that Nash-MTL achieves state-ofthe-art results on multiple MTL benchmarks in various domains.

## 1. Introduction

In many real-world applications, one needs to solve several tasks simultaneously using limited computational or data resources. For example, perception for autonomous vehicles requires lane detection, object detection, and free-space estimation, which must all run in parallel and in real-time. This is normally solved via multi-task learning (MTL), where one model is jointly trained on several learning tasks [\(Caruana,](#page-9-0) [1997;](#page-9-0) [Ruder,](#page-10-0) [2017;](#page-10-0) [Crawshaw,](#page-9-1) [2020\)](#page-9-1). Multi-task learning was also shown to improve generalization in theory [\(Baxter,](#page-9-2)

[2000\)](#page-9-2) and in practice (e.g., auxiliary learning, [Liu et al.,](#page-10-1) [2019a;](#page-10-1) [Achituve et al.,](#page-8-0) [2021;](#page-8-0) [Navon et al.,](#page-10-2) [2021a\)](#page-10-2).

Unfortunately, MTL often causes performance degradation compared to single-task models [\(Standley et al.,](#page-10-3) [2020\)](#page-10-3). A main reason for such degradation is gradients conflict [\(Yu](#page-11-0) [et al.,](#page-11-0) [2020a;](#page-11-0) [Wang et al.,](#page-11-1) [2020;](#page-11-1) [Liu et al.,](#page-9-3) [2021a\)](#page-9-3). These per-task gradients may have conflicting directions or a large difference in magnitudes, with the largest gradient dominating the update direction. The degraded performance of MTL due to poor training, compared with its potential to improve performance due to better generalization, has a major impact on many real-world systems. Improving MTL optimization algorithms is therefore an important task with significant implications to many systems.

Currently, most MTL optimization algorithms [\(Sener &](#page-10-4) [Koltun,](#page-10-4) [2018;](#page-10-4) [Yu et al.,](#page-11-0) [2020a;](#page-11-0) [Liu et al.,](#page-9-3) [2021a\)](#page-9-3) follow a general scheme. First, compute the gradients for all tasks g1, ..., gK. Next, combine those gradients into a joint direction, ∆ = A(g1, ..., gK) using an aggregation algorithm A. Finally, update model parameters using a single-task optimization algorithm, replacing the gradients with ∆. Multiple heuristics were proposed for the aggregation algorithm A. However, to the best of our knowledge, a principled, axiomatic, approach to gradient aggregation is still missing.

Here we address the gradient combination step by viewing it as a cooperative bargaining game [\(Thomson,](#page-10-5) [1994\)](#page-10-5). Each task represents a player, whose utility is derived from its gradient, and players negotiate to reach an agreed direction. This formulation allows us to use results from game theory literature that analyze this problem from an axiomatic perspective. In his seminal paper, [Nash](#page-10-6) [\(1953\)](#page-10-6) presented an axiomatic approach to the bargaining problem and showed that under certain axioms, the bargaining problem has a unique solution known as the *Nash Bargaining Solution*. This solution is known to be proportionally fair, where any alternative will have a negative average relative change. This proportionally fair update allows us to find a solution that works for all tasks without being dominated by a single large gradient.

Building on Nash's results, we propose a novel MTL optimization algorithm, named *Nash-MTL*, where the gradients are combined at each step using the Nash bargaining so-

<sup>\*</sup>Equal contribution <sup>1</sup>Bar-Ilan University, Ramat Gan, Israel <sup>2</sup>Nvidia, Tel-Aviv, Israel <sup>3</sup>National University of Singapore. Correspondence to: Aviv Navon <aviv.navon@biu.ac.il>, Aviv Shamsian <aviv.shamsian@live.biu.ac.il>.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1. *Illustrative example*: Optimization trajectories in loss space. Shown are 5 different initializations (black dots •), and their trajectories are colored from orange to purple. Losses have a large difference in scale. See Appendix [B](#page-15-0) for details. For linear scalarization (LS), PCGrad, and CAGrad, the optimization process is controlled by the gradient of `2, since it has a larger magnitude, resulting in imbalanced solutions between tasks (mostly ending at the bottom right). These three methods also fail to converge to an optimal solution for the rightmost initialization points. In contrast, MGDA is inclined towards the task with the smallest gradient magnitude (`1). Our method, *Nash-MTL*, is invariant to changes in loss scale and produces solutions that are well balanced across the Pareto front.

lution. We first characterize the Nash bargaining solution for MTL and derive an efficient algorithm to approximate its value. Then, we analyze our approach theoretically and establish convergence guarantees in the convex and nonconvex cases. Finally, we show empirically that our Nash-MTL approach achieves state-of-the-art results on four MTL benchmarks on a variety of challenges ranging from computer vision and quantum chemistry to reinforcement learning. To support future research and the reproducibility of the results, we make our source code publicly available at: <https://github.com/AvivNavon/nash-mtl>.

## 2. Background

### 2.1. Pareto Optimality

Optimization for MTL is a specific case of multipleobjective optimization (MOO). Given objective functions `1, ..., `K, the performance of solution a x is measured by the vector of objective values (`1(x), ..., `K(x)). One main property of MOO is that since there is no natural linear ordering on vectors it is not always possible to compare solutions so there is no clear optimal value.

We say that a solution x dominates x 0 if it is better on one or more objectives and not worse on any other objectives. A solution that is not dominated by any other is called *Pareto optimal*, and the set of all such solutions is called the *Pareto front*. It is important to note that there is no clear way to select between different Pareto optimal solutions without additional assumptions or prior about the user preferences [\(Navon et al.,](#page-10-7) [2021b\)](#page-10-7). For non-convex problems, a point is defined as local Pareto optimal if it is Pareto optimal in some open set containing it. Further, a point is called *Pareto stationary* if there exists a convex combination of the gradients at this point that equals zero. Pareto stationarity is a necessary condition for Pareto optimality.

#### 2.2. Nash Bargaining Solution

We provide a brief background on cooperative bargaining games and the Nash bargaining solution, see [Thom](#page-10-5)[son](#page-10-5) [\(1994\)](#page-10-5) for more details. In a bargaining problem, we have K players, each with their own utility function ui : A ∪ {D} → <sup>R</sup>, which they wish to maximize. A is the set of possible agreements and D is the disagreement point which the players default to if they fail to reach an agreement. We define the set of possible payoffs as U = {(u1(x), ..., uK(x)) : x ∈ A} ⊂ <sup>R</sup> <sup>K</sup> and d = (u1(D), ..., uK(D)). We assume U is convex, compact and that there exists a point in U that strictly dominates d, namely there exists a u ∈ U such that ∀i : u<sup>i</sup> > d<sup>i</sup> .

[Nash](#page-10-6) [\(1953\)](#page-10-6) showed that for such payoff set U, the twoplayer bargaining problem has a unique solutionthat satisfies the following properties or axioms: Pareto optimality, symmetry, independence of irrelevant alternatives, and invariant to affine transformations. This was later extended to multiple players [\(Szep & Forg](#page-10-8) ´ o´, [1985\)](#page-10-8).

Axiom 2.1. Pareto optimality: The agreed solution must not be dominated by another option, i.e. there cannot be any other agreement that is better for at least one player and not worse for any of the players.

As it is a cooperative game, it makes little sense that the players will curtail another player without any personal gains, so it is natural to assume the agreed solution will not be dominated by another.

Axiom 2.2. Symmetry: The solution should be invariant to permuting the order of the players.

Axiom 2.3. Independence of irrelevant alternatives (IIA): If we enlarge the of possible payoffs to U˜ ) U, and the solution is in the original set U, u <sup>∗</sup> ∈ U, then the agreed point when the set of possible payoffs is U will stay u ∗ .

Axiom 2.4. Invariance to affine transformation: If we transform each utility function ui(x) to u˜i(x) = c<sup>i</sup> ·ui(x) + b<sup>i</sup> with c<sup>i</sup> > 0 then if the original agreement had utilities (y1, ..., yk) the agreement after the transformation has utilities (c1y<sup>1</sup> + b1, ..., cky<sup>k</sup> + bk)

We argue that in the MTL setting, it is natural to require axioms 2.1-2.3. Axiom 2.4, in our mind, is the only nonnatural assumption used by the Nash bargaining solution in the context of MTL. We argue that indeed it is a desired property that is helpful for MTL. Axiom 2.4 means that the solution does not take into account the gradients' norms but rather treats all of them the same, as if they were normalized. Without enforcing this assumption, the solution can easily be dominated by a single direction (see Figure [1\)](#page-1-0). We further validate the importance of this assumption by investigating a scale-invariant baseline in Section [6.](#page-6-0)

The unique point satisfying all these axioms is called the Nash bargaining solution and is given as

$$u^* = \arg\max_{u \in U} \sum_i \log(u_i - d_i) \quad (1)$$

$$s.t. \forall i: u_i > d_i$$

## <span id="page-2-1"></span>3. Method

We now describe our Nash-MTL method in detail. We first formalize the gradient combination step as a bargaining game and analyze the Nash bargaining solution for this game. We then describe our algorithm to approximate the solution efficiently. We note that the computational cost of that approximation is critical because this approximation is executed for each gradient update. To simplify the notation, we do not distinguish between shared and task-specific parameters. We note, however, that task-specific parameters have no contribution to the Nash bargaining solution calculation.

#### 3.1. Nash Bargaining Multi-Task Learning

Given an MTL optimization problem and parameters θ, we search for an update vector ∆θ in the ball of radius centered around zero, B. We frame this as a bargaining problem with the agreement set B and the disagreement point at 0, i.e., staying at the current parameters θ. We define the utility function for each player as ui(∆θ) = g > <sup>i</sup> ∆θ where g<sup>i</sup> is the gradient of the loss of task i at θ. We note that since the agreement set is compact and convex and the utilities are linear then the set of possible payoffs is also compact and convex.

Our main assumption, besides the ones used by Nash, is that if θ is not Pareto stationary then the gradients are linearly independent (see further discussion on this assumption in Section [5\)](#page-5-0). Under this assumption, we also have that the

disagreement point, ∆θ = 0 is dominated by another in B. We now show that if θ is not on the Pareto front, the unique Nash bargaining solution has the following form:

Claim 3.1. *Let* G *be the* d × K *matrix whose columns are the gradients* g<sup>i</sup> *. The solution to* arg max∆θ∈B P i log(∆θ <sup>&</sup>gt;gi) *is (up to scaling)* P <sup>i</sup> αig<sup>i</sup> *where* α ∈ R K <sup>+</sup> *is the solution to* G>Gα = 1/α *where* <sup>1</sup>/α *is the element-wise reciprocal.*

*Proof.* The derivative of this objective is P<sup>K</sup> i=1 ∆θ>g<sup>i</sup> gi . For all vectors ∆θ such that ∀i : ∆θ T g<sup>i</sup> > 0 the utilities are monotonically increasing with the norm of ∆θ. Thus, from the Pareto optimality assumption by Nash, the optimal solution has to be on the boundary of B. From this we see that the gradient at the optimal point P<sup>K</sup> i=1 1 ∆θ>g<sup>i</sup> gi must be in the radial direction, i.e., P<sup>K</sup> i=1 1 ∆θ>g<sup>i</sup> g<sup>i</sup> k ∆θ or P<sup>K</sup> i=1 ∆θ>g<sup>i</sup> g<sup>i</sup> = λ∆θ. Since the gradients are independent we must have ∆θ = P <sup>i</sup> αig<sup>i</sup> and ∀i : 1 ∆θ>g<sup>i</sup> = λα<sup>i</sup> or ∀i : ∆θ <sup>&</sup>gt;g<sup>i</sup> = 1 λα<sup>i</sup> . As the inner product must be positive for a descent direction we can conclude λ > 0; we set λ = 1 to ascertain the direction of ∆θ (the norm might be larger then ). Now finding the bargaining solution is reduced to finding α ∈ R <sup>K</sup> with α<sup>i</sup> > 0 such that ∀i : ∆θ <sup>&</sup>gt;g<sup>i</sup> = P <sup>j</sup> αjg > j g<sup>i</sup> = 1 α<sup>i</sup> . This is equivalent to requiring that G>Gα = 1/α where 1/α is the element-wise reciprocal.

We now provide some intuition for this solution. First, if all g<sup>i</sup> are orthogonal we get α<sup>i</sup> = 1/||g<sup>i</sup> || and ∆θ = P <sup>g</sup><sup>i</sup> ||gi|| which is the obvious scale invariant solution. When they are not orthogonal, we get

<span id="page-2-0"></span>
$$\alpha_i \|g_i\|^2 + \sum_{j \neq i} \alpha_j g_j^\top g_i = 1/\alpha_i \quad (2)$$

We can consider P <sup>j</sup>6=<sup>i</sup> αjg > j g<sup>i</sup> = P <sup>j</sup>6=<sup>i</sup> αjg<sup>j</sup> > g<sup>i</sup> as the interaction between task i and the other tasks; If it is positive there is a positive interaction and the other gradients aid the i'th task, and if it is negative they hamper it. When there is a negative interaction, the LHS of Eq. [2](#page-2-0) decreases and as a result, α<sup>i</sup> increases to compensate for it. Conversely, where there is a positive interaction α<sup>i</sup> will decrease.

#### <span id="page-2-2"></span>3.2. Solving G>Gα = 1/α

Here we describe how to efficiently approximate the optimal solution for G>Gα = 1/α through a sequence of convex optimization problems. We define a βi(α) = g > <sup>i</sup> Gα, and wish to find α such that α<sup>i</sup> = 1/β<sup>i</sup> for all i, or equivalently log(αi) + log(βi(α)) = 0. Denote ϕi(α) = log(αi) + log(βi) and ϕ(α) = P <sup>i</sup> ϕi(α). With that, our goal is to find a non-negative α such that ϕi(α) = 0 for all i. We can

![](_page_3_Diagram_2.jpeg)

Figure 2. *Visualization of the update direction*: We show the update direction (blue) obtained by various methods on three gradients in R . We rescaled the returned vectors for better visibility, showing only the direction. We further show the size of the projection (red) of the update to each gradient direction (black). Nash-MTL produce an update direction with the most balanced projections.

#### Algorithm 1 Nash-MTL

Input: θ (0) – initial parameter vector, {`i} K <sup>i</sup>=1 – differentiable loss functions, η – learning rate for t = 1, ..., T do Compute task gradients g (t) <sup>i</sup> = ∇<sup>θ</sup> (t−1) `<sup>i</sup> Set G (t) the matrix with columns g (t) i Solve for α: (G (t) ) <sup>&</sup>gt;G (t)α = 1/α to obtain α Update the parameters θ (t) = θ (t) − ηG(t)α end for Return: θ (T )

write this as the following optimization problem

$$\min_{\alpha} \sum_i \varphi_i(\alpha) \quad (3)$$

$$\text{s.t. } \forall i, \quad -\varphi_i(\alpha) \leq 0$$

$$\alpha_i > 0 \quad .$$

The constraints in this problem are convex and linear and the objective is concave. We first try to solve the following convex surrogate objective

$$\min_{\alpha} \sum_i \beta_i(\alpha) \quad (4)$$

$$\text{s.t.} \forall i, \quad -\varphi_i(\alpha) \leq 0$$

$$\alpha_i > 0 \quad .$$

Here, we minimize P i β<sup>i</sup> under the constraint β<sup>i</sup> = g > <sup>i</sup> Gα ≥ 1/α<sup>i</sup> . While this objective is not equivalent to the original problem, we found it very useful. In many cases, it produces exact solutions with ϕ(α) = 0 as required.

To further improve our approximation, we considered the following problem,

$$\min_{\alpha} \sum_i \beta_i(\alpha) + \varphi(\alpha) \quad (5)$$

$$\text{s.t.} \forall i, \quad -\varphi_i(\alpha) \leq 0$$

$$\alpha_i > 0 \quad .$$

Adding ϕ(α) to the objective may further reduce it, moving it closer to zero; however, it renders the problem to be nonconvex. Despite that, our solution can be improved iteratively by replacing the concave term ϕ(α) with its first-order approximation ϕ˜<sup>τ</sup> (α) = ϕ(α (τ) ) + ∇ϕ(α (τ) ) <sup>&</sup>gt;(α − α (τ) ). Where, α (τ) is the solution at iteration τ . Note that we replace ϕ with ϕ˜ only in the objective and keep ϕ(α) as is in the constraint: i.e., min<sup>α</sup> P i βi(α)+ ˜ϕ<sup>τ</sup> (α) s.t. −ϕi(α) ≤ 0 and α<sup>i</sup> > 0 for all i. This sequential optimization approach is a variation of the concave-convex procedure (CCP) [\(Yuille & Rangarajan,](#page-11-2) [2003;](#page-11-2) [Lipp & Boyd,](#page-9-4) [2016\)](#page-9-4). Therefore the sequence {α (τ)}<sup>τ</sup> converges to a critical point of the original non-convex problem in Eq. [5](#page-3-0) based on previous theory of CCP by [Sriperumbudur & Lanckriet](#page-10-9) [\(2009\)](#page-10-9). Moreover, since we do not modify the constraint, α (τ) always satisfies the constraint of the original problem for any τ . Finally, the following proposition shows that original objective monotonically decreases with τ :

Proposition 3.2. *Denote the objective for the optimization problem in Eq. [5](#page-3-0) by* φ(α) = P i βi(α) + ϕ(α)*. Then,* φ α (τ+1) ≤ φ α (τ) *for all* τ ≥ 1*.*

<span id="page-3-1"></span>We provide proof and further discussion in Appendix [A.](#page-12-0) In practice, we limit the sequence of CCP to 20 in all experiments, with the exception of Section [6.3](#page-7-0) for which we use a single step. We found the improved solution to have a limited effect on the MTL performance (see Appendix [D.2\)](#page-16-0).

#### 3.3. Practical Speedup

<span id="page-3-0"></span>One shortcoming of many leading MTL methods is that all task gradients are required for obtaining the joint update direction. When the number of tasks K becomes large, this may be too computationally expensive as it requires one to perform K backward passes through the shared backbone to compute the K gradients. Prior work suggested using a subset of tasks [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3) or replacing the task gradients with the feature level gradient [\(Sener & Koltun,](#page-10-4) [2018;](#page-10-4) [Liu et al.,](#page-10-10) [2021b;](#page-10-10) [Javaloy & Valera,](#page-9-5) [2021\)](#page-9-5) as potential practical speedups. We emphasize that this issue is not

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 3. *QM9*. Test ∆<sup>m</sup> throughout the training process averaged over 3 random seeds.

unique to our method, but rather is shared to all methods that compute all gradients for all tasks.

In practice, we found that using feature-level gradients as a surrogate to the gradient of the shared parameters dramatically degrades the performance of our method. See Appendix [C](#page-15-1) for empirical results and further discussion. As an alternative, we suggest updating the gradient weights α (t) once every few iterations instead of every iteration. This simple yet effective solution greatly reduces the runtime (up to ∼ ×10 for QM9 and ∼ ×5 for MT10) while maintaining high performance. In Section [6.4](#page-8-1) we provide experimental results while varying the frequency of task weights update on the QM9 dataset and the MT10 benchmark. Our results show that Nash-MTL runtime can be reduced to about the same as linear scalarization (or STL) while maintaining competitive results compared to other baselines; However, in some cases, we do see a noticeable drop in performance compared with our standard approach.

## 4. Related Work

In multitask learning (MTL), one simultaneously solves several learning problems while sharing information among tasks [\(Caruana,](#page-9-0) [1997;](#page-9-0) [Ruder,](#page-10-0) [2017\)](#page-10-0), commonly through a joint hidden representation [\(Zhang et al.,](#page-11-3) [2014;](#page-11-3) [Dai et al.,](#page-9-6) [2016;](#page-9-6) [Pinto & Gupta,](#page-10-11) [2017;](#page-10-11) [Zhao et al.,](#page-11-4) [2018;](#page-11-4) [Liu et al.,](#page-10-12) [2019b\)](#page-10-12). Studies in the literature proposed several explanations for the difficulty in the optimization process of MTL, such as conflicting gradients [\(Wang et al.,](#page-11-1) [2020;](#page-11-1) [Yu et al.,](#page-11-0) [2020a\)](#page-11-0), or plateaus in the loss landscape [\(Schaul et al.,](#page-10-13) [2019\)](#page-10-13). Other studies aimed at improving multitask learning by proposing novel architectures [\(Misra et al.,](#page-10-14) [2016;](#page-10-14) [Hashimoto et al.,](#page-9-7) [2017;](#page-9-7) [Liu et al.,](#page-10-12) [2019b;](#page-10-12) [Chen et al.,](#page-9-8) [2020\)](#page-9-8). We focus on weighting the gradients of the tasks via an axiomatic approach that is agnostic to the architecture used. Studies in a similar vein proposed to weigh the task losses

<span id="page-4-1"></span>Table 1. *QM9*. Test performance averaged over 3 random seeds.

|          | MR ↓ |     | ∆ | m % ↓ |   |
|----------|------|-----|---|-------|---|
| LS       | 6.8  | 177 | 6 | ± 3   | 4 |
| SI       | 4.0  | 77  | 8 | ± 9   | 2 |
| RLW      | 8.2  | 203 | 8 | ± 3   | 4 |
| DWA      | 6.4  | 175 | 3 | ± 6   | 3 |
| UW       | 5.3  | 108 | 0 | ± 22  | 5 |
| MGDA     | 5.9  | 120 | 5 | ± 2   | 0 |
| PCGrad   | 5.0  | 125 | 7 | ± 10  | 3 |
| CAGrad   | 5.7  | 112 | 8 | ± 4   | 0 |
| IMTL-G   | 4.7  | 77  | 2 | ± 9   | 3 |
| Nash-MTL | 2 5  | 62  | 0 | ± 1   | 4 |

with various approaches, such as the uncertainty of the tasks [\(Kendall et al.,](#page-9-9) [2018\)](#page-9-9), the norm of the gradients [\(Chen](#page-9-10) [et al.,](#page-9-10) [2018\)](#page-9-10), random weights [\(Lin et al.,](#page-9-11) [2021\)](#page-9-11), and similarity of the gradients [\(Du et al.,](#page-9-12) [2018;](#page-9-12) [Suteu & Guo,](#page-10-15) [2019\)](#page-10-15). These methods are mostly heuristic and can have unstable performance [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3). Recently, several studies proposed MTL approaches based on the multiple-gradient descent algorithm (MGDA) for multi-objective optimization [\(Desid](#page-9-13) ´ eri ´ , [2012\)](#page-9-13). This is an appealing approach since, under mild conditions, convergence to a Pareto stationary point is guaranteed. [Sener & Koltun](#page-10-4) [\(2018\)](#page-10-4) cast the multi-objective problem to multi-task problem and suggest task weighting based on the Frank-Wolfe algorithm [\(Jaggi,](#page-9-14) [2013\)](#page-9-14). [Liu et al.](#page-9-3) [\(2021a\)](#page-9-3) searches for an update direction in a neighborhood of the average gradient that maximizes the worst improvement of any task. Unlike these studies, we propose an MTL approach based on a Bargaining game that can find solutions that are Pareto optimal and proportionally fair.

The closest work to our approach, to the best of our knowledge, is [Liu et al.](#page-10-10) [\(2021b\)](#page-10-10). There, the authors propose to look for a fair gradient direction where all the cosine similarities are equal. We note that this update direction satisfies all of the Nash axioms except for Pareto optimally. Thus, unlike our proportionally fair approach, it can settle for a sub-optimal solution for the sake of fairness.

Finally, we note that the Nash bargaining solution was effectively applied to problems in various fields such as communication [\(Zhang et al.,](#page-11-5) [2008;](#page-11-5) [Leshem & Zehavi,](#page-9-15) [2011;](#page-9-15) [Shi](#page-10-16) [et al.,](#page-10-16) [2018\)](#page-10-16), economics [\(Dagan & Volij,](#page-9-16) [1993\)](#page-9-16), and computing [\(Grosu et al.,](#page-9-17) [2002\)](#page-9-17), and to several learning setups, such as reinforcement learning [\(Qiao et al.,](#page-10-17) [2006\)](#page-10-17), Bayesian optimization [\(Binois et al.,](#page-9-18) [2020\)](#page-9-18), clustering [\(Rezaee et al.,](#page-10-18) [2021\)](#page-10-18), federated learning [\(Kim,](#page-9-19) [2021\)](#page-9-19), and multi-armed bandits [\(Baek & Farias,](#page-9-20) [2021\)](#page-9-20).

<span id="page-5-2"></span>Table 2. *NYUv2*. Test performance for three tasks: semantic segmentation, depth estimation, and surface normal. Values are averages over 3 random seeds.

|          |    | mIoU ↑ | Pix | Acc ↑ | Abs | Err  | Depth ↓ Rel | Err  | ↓    | Angle | Distance | ↓  | Surface |    | Normal Within | ◦ t | ↑  |    |    | MR ↓ | ∆m  | % ↓  |
|----------|----|--------|-----|-------|-----|------|-------------|------|------|-------|----------|----|---------|----|---------------|-----|----|----|----|------|-----|------|
|          |    |        |     |       |     |      |             |      | Mean |       | Median   |    | 11.25   |    | 22.5          |     |    | 30 |    |      |     |      |
| STL      | 38 | 30     | 63  | 76    | 0   | 6754 | 0           | 2780 | 25   | 01    | 19       | 21 | 30      | 14 | 57            | 20  | 69 | 15 |    |      |     |      |
| LS       | 39 | 29     | 65  | 33    | 0   | 5493 | 0           | 2263 | 28   | 15    | 23       | 96 | 22      | 09 | 47            | 50  | 61 | 08 | 8  | 11   | 5   | 59   |
| SI       | 38 | 45     | 64  | 27    | 0   | 5354 | 0           | 2201 | 27   | 60    | 23       | 37 | 22      | 53 | 48            | 57  | 62 | 32 | 7  | 11   | 4   | 39   |
| RLW      | 37 | 17     | 63  | 77    | 0   | 5759 | 0           | 2410 | 28   | 27    | 24       | 18 | 22      | 26 | 47            | 05  | 60 | 62 | 10 | 11   | 7   | 78   |
| DWA      | 39 | 11     | 65  | 31    | 0   | 5510 | 0           | 2285 | 27   | 61    | 23       | 18 | 24      | 17 | 50            | 18  | 62 | 39 | 6  | 88   | 3   | 57   |
| UW       | 36 | 87     | 63  | 17    | 0   | 5446 | 0           | 2260 | 27   | 04    | 22       | 61 | 23      | 54 | 49            | 05  | 63 | 65 | 6  | 44   | 4   | 05   |
| MGDA     | 30 | 47     | 59  | 90    | 0   | 6070 | 0           | 2555 | 24   | 88    | 19       | 45 | 29      | 18 | 56            | 88  | 69 | 36 | 5  | 44   | 1   | 38   |
| PCGrad   | 38 | 06     | 64  | 64    | 0   | 5550 | 0           | 2325 | 27   | 41    | 22       | 80 | 23      | 86 | 49            | 83  | 63 | 14 | 6  | 88   | 3   | 97   |
| GradDrop | 39 | 39     | 65  | 12    | 0   | 5455 | 0           | 2279 | 27   | 48    | 22       | 96 | 23      | 38 | 49            | 44  | 62 | 87 | 6  | 44   | 3   | 58   |
| CAGrad   | 39 | 79     | 65  | 49    | 0   | 5486 | 0           | 2250 | 26   | 31    | 21       | 58 | 25      | 61 | 52            | 36  | 65 | 58 | 3  | 77   | 0   | 20   |
| IMTL-G   | 39 | 35     | 65  | 60    | 0   | 5426 | 0           | 2256 | 26   | 02    | 21       | 19 | 26      | 2  | 53            | 13  | 66 | 24 | 3  | 11   | −   | 0 76 |
| Nash-MTL | 40 | 13     | 65  | 93    | 0   | 5261 | 0           | 2171 | 25   | 26    | 20       | 08 | 28      | 4  | 55            | 47  | 68 | 15 | 1  | 55   | − 4 | 04   |

### <span id="page-5-0"></span>5. Analysis

We now analyze the convergence of our method in the convex and non-convex cases. As even single-task non-convex optimization might only converge to a stationary point, we will prove convergence to a Pareto stationary point, i.e., a point where some convex combination of the gradients is zero. As stated, we also assume that the gradients are independent while not at a Pareto stationary point. Independence of the gradients is a slightly stronger assumption than Pareto stationarity but is needed to exclude degenerate edge cases such as two identical tasks.

We note that by substituting local Pareto optimality for Pareto stationarity in Assumption [5.1](#page-5-1) we can show convergence to a local Pareto optimal point. However, this assumption has strong implications, as it implies we avoid local maxima and saddle points of any specific task. Since our update rule is a descent direction for all tasks, we can reasonably assume that our algorithm avoids local maxima points. Furthermore, it was shown that first-order methods avoid saddle points [\(Panageas et al.,](#page-10-19) [2019\)](#page-10-19), giving credence to this stronger assumption. Nevertheless, we take a conservative approach and state our results with the weaker assumption.

We formally make the following assumptions:

<span id="page-5-1"></span>Assumption 5.1. We assume that for a sequence {θ (t)}<sup>∞</sup> t=1 generated by our algorithm, the set of the gradient vectors g (t) 1 , ..., g (t) <sup>K</sup> at any point on the sequence and at any partial limit are linearly independent unless that point is a Pareto stationary point.

Assumption 5.2. We assume that all loss functions are differentiable, bounded below and that all sub-level sets are bounded. The input domain is open and convex.

<span id="page-5-3"></span>Assumption 5.3. We assume that all the loss functions are L-smooth,

$$\|\nabla \ell_i(x) - \nabla \ell_i(y)\| \leq L\|x - y\| \quad . \quad (6)$$

<span id="page-5-4"></span>Theorem 5.4. *Let* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *be the sequence generated by the update rule* θ (t+1) = θ (t) − µ (t)∆θ (t) *where* ∆θ (t) = P<sup>K</sup> <sup>i</sup>=1 α (t) i g (t) i *is the Nash bargaining solution* (G(t) ) <sup>&</sup>gt;G(t)α (t) = 1/α(t) *. Set* µ (t) = min i∈[K] 1 LKα(t) *. Then, the sequence* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *has a subsequence that converges to a Pareto stationary point* θ ∗ *. Moreover all the loss functions* (`1(θ (t) ), ..., `K(θ (t) )) *converge to* (`1(θ ∗ ), ..., `K(θ ∗ ))*.*

*Proof sketch.* We can show that µ (t) = min<sup>i</sup> 1 α → 0 so ||α (t) || → ∞. We also show that ||1/α(t) || is bounded. As (G(t) ) <sup>&</sup>gt;G(t)α (t) = 1/α(t) this means that the smallest singluar value of (G(t) ) <sup>&</sup>gt;G(t) must converge to zero. From compactness {θ (t)}<sup>∞</sup> <sup>t</sup>=1 has a converging subsequence whose limit we denote as θ ∗ . From continuity we get that the gradients Gram matrix G>G computed at θ <sup>∗</sup> must have a zero singular value and therefore the gradients are linearly dependent. From our assumption this means that θ ∗ is Pareto stationary. As the losses are monotonically decearsing and bounded below they must converge and to the subsequence limit of (`1(θ ∗ ), ..., `K(θ ∗ )).

If we also assume convexity, we can strengthen our claim

Theorem 5.5. *Let* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *be the sequence generated by the update rule* θ (t+1) = θ (t) − µ (t)∆θ (t) *where* ∆θ (t) = P<sup>K</sup> <sup>i</sup>=1 α (t) i g (t) i *is the Nash bargaining solution* (G(t) ) <sup>&</sup>gt;G(t)α (t) = 1/α(t) *. Set* µ (t) = min i∈[K] LKα(t) *. If*

<span id="page-6-1"></span>Table 3. *CityScapes*. Test performance for two tasks: semantic segmentation and depth estimation. Value are averages over 3 random seeds.

|          | mIoU  | Segmentation ↑ Pix Acc ↑ | Abs Err | Depth ↓ Rel Err ↓ |   | MR ↓ | ∆ m % ↓ |
|----------|-------|--------------------------|---------|-------------------|---|------|---------|
| STL      | 74.01 | 93.16                    | 0.0125  | 27.77             |   |      |         |
| LS       | 75.18 | 93.49                    | 0.0155  | 46.77             | 6 | 12   | 22.60   |
| SI       | 70.95 | 91.73                    | 0.0161  | 33.83             | 8 | 00   | 14.11   |
| RLW      | 74.57 | 93.41                    | 0.0158  | 47.79             | 9 | 25   | 24.38   |
| DWA      | 75.24 | 93.52                    | 0.0160  | 44.37             | 6 | 00   | 21.45   |
| UW       | 72.02 | 92.85                    | 0.0140  | 30.13             | 5 | 25   | 5.89    |
| MGDA     | 68.84 | 91.54                    | 0.0309  | 33.50             | 8 | 75   | 44.14   |
| PCGrad   | 75.13 | 93.48                    | 0.0154  | 42.07             | 6 | 37   | 18.29   |
| GradDrop | 75.27 | 93.53                    | 0.0157  | 47.54             | 5 | 50   | 23.73   |
| CAGrad   | 75.16 | 93.48                    | 0.0141  | 37.60             | 5 | 37   | 11.64   |
| IMTL-G   | 75.33 | 93.49                    | 0.0135  | 38.41             | 3 | 62   | 11.10   |
| Nash-MTL | 75.41 | 93.66                    | 0.0129  | 35.02             | 1 | 75   | 6.82    |

*we assume that all the loss functions are convex, then the sequence* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *converges to a Pareto optimal point* θ ∗ *.*

See the full proofs in the appendix Sec. [A.](#page-12-0)

## <span id="page-6-0"></span>6. Experiments

We evaluate Nash-MTL on diverse multi-task learning problems. The experiments show the superiority of Nash-MTL over previous MTL methods. To support future research and the reproducibility of the results, we will make our source code publicly available. Additional experimental results and details are provided in Appendix [B.](#page-14-0)

Compared methods: We compare the following approaches: (1) Our proposed Nash-MTL algorithm described in Section [3;](#page-2-1) (2) Single task learning (STL), training an independent model for each task; (3) Linear scalarization (LS) baseline which minimizes P k `k; (4) Scale-invariant (SI) baseline which minimizes P k log `k. This baseline is invariant to rescaling each loss with a positive number; (5) Dynamic Weight Average (DWA) [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12) adjusts task weights based on the rates of loss changes over time; (6) Uncertainty weighting (UW) [\(Kendall et al.,](#page-9-9) [2018\)](#page-9-9) uses task uncertainty quantification to adjust task weights; (7) MGDA [\(Sener & Koltun,](#page-10-4) [2018\)](#page-10-4) finds a convex combination of gradients with a minimal norm; (8) Random loss weighting (RLW) with normal distribution, scales the losses according to randomly sampled task weights [\(Lin](#page-9-11) [et al.,](#page-9-11) [2021\)](#page-9-11); (9) PCGrad [\(Yu et al.,](#page-11-0) [2020a\)](#page-11-0) removes conflicting components of each gradient w.r.t the other gradients; (10) GradDrop [\(Chen et al.,](#page-9-8) [2020\)](#page-9-8) randomly drops components of the task gradients based on how much they conflict; (11) CAGrad [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3) optimizes for the average loss while explicitly controlling the minimum decrease rate across tasks; (12) IMTL-G [\(Liu et al.,](#page-10-10) [2021b\)](#page-10-10) uses an update

direction with equal projections on task gradients. IMTL-G is applied to the feature-level gradients, as was suggested by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but its performance was even worse.

Evaluation. For each experiment, we report the common evaluation metrics for each task. Since naturally MTL does not carry a single objective and since the scale of per-task metrics often varies significantly, we report two metrics that capture the overall performance: (1) ∆m%, the average per-task performance drop of method m relative to the STL baseline denoted b. Formally, ∆m% = K P<sup>K</sup> <sup>k</sup>=1(−1)<sup>δ</sup><sup>k</sup> (Mm,k − <sup>M</sup>b,k)/Mb,k, where <sup>M</sup>b,k is the value of metric M<sup>k</sup> obtained by the baseline and Mm,k by the compared method. δ<sup>k</sup> = 1 if a higher value is better for a metric M<sup>k</sup> and 0 otherwise [\(Maninis et al.,](#page-10-20) [2019;](#page-10-20) [Liu](#page-9-3) [et al.,](#page-9-3) [2021a\)](#page-9-3). (2) Mean Rank (MR): The average rank of each method across the different tasks (lower is better). A method receives the best value, MR = 1, if it ranks first in all tasks.

#### <span id="page-6-2"></span>6.1. Multi-Task Regression for QM9

We evaluate Nash-MTL on predicting 11 properties of molecules from the QM9 dataset [\(Ramakrishnan et al.,](#page-10-21) [2014\)](#page-10-21), a widely used benchmark for graph neural networks. QM9 consists of ∼ 130K molecules represented as graphs annotated with both node and edge features. We used the QM9 example in PyTorch Geometric [\(Fey & Lenssen,](#page-9-21) [2019\)](#page-9-21), and use 110K molecules for training, 10K for validation, and 10K as a test set. As each task target range is at a different scale, this could be an issue for other methods that are not scale-invariant like ours. For fairness, we normalized each task target to have zero mean and unit standard deviation. We use the popular GNN model from

[Gilmer et al.](#page-9-22) [\(2017\)](#page-9-22), a network comprised of several concatenated message passing layers, which update the node features based on both node and edge features, followed by the pooling operator from [Vinyals et al.](#page-10-22) [\(2015\)](#page-10-22). Specifically, we used the implementation from [Fey & Lenssen](#page-9-21) [\(2019\)](#page-9-21). We train each method for 300 epochs and search for the best learning-rate (lr) given by the ∆<sup>m</sup> performance on the validation set. We use a learning-rate scheduler to reduce the lr once the validation ∆<sup>m</sup> metric has stopped improving. The validation set is also used for early stopping.

Predicting molecular properties in QM9 poses a significant challenge for MTL methods because the number of tasks is large and because the loss scales vary significantly. The scale issue is only partially resolved by normalization because some tasks are easier to learn than others. Prior work found that single-task learning significantly improves performance on all targets compared to MTL methods [\(Maron](#page-10-23) [et al.,](#page-10-23) [2019;](#page-10-23) [Klicpera et al.,](#page-9-23) [2020\)](#page-9-23).

Results are shown in Figure [3](#page-4-0) and Table [1.](#page-4-1) Nash-MTL achieves the best performance in terms of both MR and ∆m. Interestingly, most MTL methods fall short compared to the simple scale-invariant baseline, which ignores gradient interaction, except for IMTL-G whose performance is on par with this baseline. This result shows that the scaleinvariant property of our approach can be beneficial. See Appendix [D.1](#page-16-1) for the per-task evaluation results.

#### 6.2. Scene Understanding

We follow the protocol of [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12) and evaluate Nash-MTL on the NYUv2 and Cityscapes datasets [\(Silber](#page-10-24)[man et al.,](#page-10-24) [2012;](#page-10-24) [Cordts et al.,](#page-9-24) [2016\)](#page-9-24). NYUv2 is an indoor scene dataset that consists of 1449 RGBD images and dense per-pixel labeling with 13 classes. We use this dataset as a multitask learning benchmark for semantic segmentation, depth estimation, and surface normal prediction.

The CityScapes dataset [\(Cordts et al.,](#page-9-24) [2016\)](#page-9-24) contains 5000 high-resolution street-view images with dense per-pixel annotations. We use this dataset as a multitask learning benchmark for semantic segmentation and depth estimation. To speed up the training phase, all images were resized to 128 × 256. The original dataset contains 19 categories for pixel-wise semantic segmentation, together with groundtruth depth maps. For segmentation, we used a coarser version of the labels with 7 classes.

For all MTL methods, we train a Multi-Task Attention Network (MTAN) [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12) which adds an attention mechanism on top of the SegNet architecture [\(Badri](#page-8-2)[narayanan et al.,](#page-8-2) [2017\)](#page-8-2). We follow the training procedure from [Liu et al.](#page-10-12) [\(2019b\)](#page-10-12); [Yu et al.](#page-11-0) [\(2020a\)](#page-11-0); [Liu et al.](#page-9-3) [\(2021a\)](#page-9-3). Each method is trained for 200 epochs with the Adam optimizer [\(Kingma & Ba,](#page-9-25) [2015\)](#page-9-25) and an initial learning-rate

<span id="page-7-1"></span>Table 4. *MT10*. Average success over 10 random seeds.

|           |      | Success | ±   | SEM |
|-----------|------|---------|-----|-----|
| STL SAC   | 0    | 90      | ± 0 | 032 |
| MTL SAC   | 0    | 49      | ± 0 | 073 |
| MTL SAC + | TE 0 | 54      | ± 0 | 047 |
| MH SAC    | 0    | 61      | ± 0 | 036 |
| SM        | 0    | 73      | ± 0 | 043 |
| CARE      | 0    | 84      | ± 0 | 051 |
| PCGrad    | 0    | 72      | ± 0 | 022 |
| CAGrad    | 0    | 83      | ± 0 | 045 |
| Nash-MTL  | 0    | 91      | ± 0 | 031 |

of 1e − 4. The learning-rate is halved to 5e − 5 after 100 epochs. As in [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3) The STL baseline refers to training task-specific SegNet models.

The results are presented in Table [2](#page-5-2) and Table [3.](#page-6-1) Our method, Nash-MTL, achieves the best MR in both datasets, the best ∆<sup>m</sup> in NYUv2 and the seconds to best ∆<sup>m</sup> in the CityScapes experiment. Nash-MTL performance is well balanced across tasks. MGDA is primarily focused on the task of predicting surface normals and achieves poor performance on the other two tasks. The inherent biasedness of MGDA towards the task with the smallest gradient magnitude was previously discussed in [Liu et al.](#page-10-10) [\(2021b\)](#page-10-10). We note that the optimal solution under Nash-MTL for the two tasks case is equivalent to independently normalizing each gradient and summing with equal weights. While this is a fairly simple approach for MTL, we show that it outperforms almost all the compared MTL methods on the two-tasks CityScapes benchmark.

#### <span id="page-7-0"></span>6.3. Multi-Task Reinforcement Learning

We consider a multi-task RL problem and evaluate Nash-MTL on the MT10 environment from the Meta-World benchmark [\(Yu et al.,](#page-11-6) [2020b\)](#page-11-6). This benchmark involves a simulated robot trained to perform actions like pressing a button and opening a window, each action treated as a task, for a total of 10 tasks. The goal is to learn a policy that can succeed across all the diverse sets of manipulation tasks. Following previous works on MTL-RL [\(Yu et al.,](#page-11-0) [2020a;](#page-11-0) [Liu et al.,](#page-9-3) [2021a;](#page-9-3) [Sodhani et al.,](#page-10-25) [2021\)](#page-10-25), we use Soft Actor-Critic (SAC) [\(Haarnoja et al.,](#page-9-26) [2018\)](#page-9-26) as the base RL algorithm. Along with the MTL methods (1) CAGrad [\(Liu](#page-9-3) [et al.,](#page-9-3) [2021a\)](#page-9-3) and (2) PCGrad [\(Yu et al.,](#page-11-0) [2020a\)](#page-11-0) applied to a shared model SAC, we evaluate the following methods: (3) STL, one SAC model per task; (4) MTL SAC with a shared model; (5) Multi-task SAC with task encoder (MTL SAC + TE, [Yu et al.](#page-11-6) [\(2020b\)](#page-11-6)); (6) Multi-headed SAC (MH SAC) with task-specific heads [\(Yu et al.,](#page-11-6) [2020b\)](#page-11-6); (7) Soft Modularization (SM, [Yang et al.](#page-11-7) [\(2020\)](#page-11-7)) which estimates per-task routes for different tasks in a shared model, and;

<span id="page-8-3"></span>![](_page_8_Figure_1.jpeg)

<span id="page-8-4"></span>Table 5. Training runtime per episode and average success for the MT10 benchmark, computed over 10 random seeds while varying the frequency of task weights updates in Nash-MTL.

|              |   | Success | ±   | SEM |      |
|--------------|---|---------|-----|-----|------|
| MTL-SAC      | 0 | 49      | ± 0 | 073 | 7.3  |
| PCGrad       | 0 | 72      | ± 0 | 022 | 9.7  |
| CAGrad       | 0 | 83      | ± 0 | 045 | 20.9 |
| Nash-MTL     | 0 | 91      | ± 0 | 031 | 40.7 |
| Nash-MTL-50  | 0 | 85      | ± 0 | 022 | 8.6  |
| Nash-MTL-100 | 0 | 87      | ± 0 | 033 | 7.9  |

(8) CARE [\(Sodhani et al.,](#page-10-25) [2021\)](#page-10-25) which utilizes language metadata and employs a mixture of encoders. We follow the same experiment setup from [Sodhani et al.](#page-10-25) [\(2021\)](#page-10-25); [Liu](#page-9-3) [et al.](#page-9-3) [\(2021a\)](#page-9-3) to train all methods over 2 million steps and report the mean success over 10 random seeds with fixed evaluation frequency. The results are presented in Table [4.](#page-7-1)

Nash-MTL achieves the best performance by a large margin. In addition, Nash-MTL is the only MTL method to reach the same performance as the per-task SAC STL baseline.

#### <span id="page-8-1"></span>6.4. Scaling-up Nash-MTL

One of the major drawbacks of the SOTA MTL methods is that they require access to all task gradients to compute the optimal update direction [\(Sener & Koltun,](#page-10-4) [2018;](#page-10-4) [Yu](#page-11-0) [et al.,](#page-11-0) [2020a;](#page-11-0) [Liu et al.,](#page-10-10) [2021b](#page-10-10)[;a\)](#page-9-3). This requires one to perform K backward passes at each optimization step, thus scales poorly with the number of tasks. Previous works suggested using a subset of tasks [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3) or replacing the task gradients with the feature-level gradient [\(Sener &](#page-10-4) [Koltun,](#page-10-4) [2018;](#page-10-4) [Liu et al.,](#page-10-10) [2021b;](#page-10-10) [Javaloy & Valera,](#page-9-5) [2021\)](#page-9-5) as potential speedups. In our experiments, we found that using the feature-level gradients can greatly reduce Nash-MTL performance (Appendix [C\)](#page-15-1). However, here we show that the simple solution of updating task weights less frequently maintains good performance while dramatically reducing the training time.

One approach to alleviate this issue is to update the task weights less frequently, and use these weights in subsequent steps. We evaluate this approach using the QM9 dataset and the MT10 benchmark and present the result in Figure [4](#page-8-3) and Table [5.](#page-8-4) We denote Nash-MTL with task weight update every T optimization steps with Nash-MTL-T.

The results show that Nash-MTL is fairly robust to varying intervals between weights updates. While this simple approach results in a small degradation in performance, it can dramatically decrease the training time of our method. For example, on the QM9, updating the weights every 5/50 steps results in a ×3.7/9.8 speedup w.r.t updating the weights at

Figure 4. Test ∆<sup>m</sup> for the QM9 dataset, averaged over 3 random seeds, for different intervals of task weights update.

each step. On the MT10 environment, updating the weights every 100 steps result in ∼ ×10 speedup (only ∼ ×1.1 slower than the fastest baseline) while outperforming all other MTL baseline method (Table [5\)](#page-8-4).

## 7. Conclusion

In this work, we present Nash-MTL, a novel and principled approach for multitask learning. We frame the gradient combination step in MTL as a bargaining game and use the Nash bargaining solution to find the optimal update direction. We highlight the importance of the scale invariance approach for multitask learning, specifically for setups with varying loss scales and gradient magnitudes. We provide a theoretical convergence analysis for Nash-MTL, showing that it converges to a Pareto optimal and Pareto stationary points in the convex and non-convex settings, respectively. Finally, our experiments show that Nash-MTL achieves state-of-the-art results on various benchmarks across multiple domains.

### 8. Acknowledgements

This work was funded by the Israeli innovation authority through the AVATAR consortium; by the Israel Science Foundation (ISF grant 737/2018); and by an equipment grant to GC and Bar Ilan University (ISF grant 2332/18).

## References

<span id="page-8-2"></span><span id="page-8-0"></span>Achituve, I., Maron, H., and Chechik, G. Self-supervised learning for domain adaptation on point clouds. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 123–133, 2021. Badrinarayanan, V., Kendall, A., and Cipolla, R. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2481–2495, 2017.

- <span id="page-9-26"></span><span id="page-9-25"></span><span id="page-9-23"></span><span id="page-9-19"></span><span id="page-9-17"></span><span id="page-9-15"></span><span id="page-9-14"></span><span id="page-9-11"></span><span id="page-9-9"></span><span id="page-9-7"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span>Baek, J. and Farias, V. F. Fair exploration via axiomatic bargaining. *arXiv preprint arXiv:2106.02553*, 2021. Baxter, J. A model of inductive bias learning. *J. Artif. Intell. Res.*, 2000. Binois, M., Picheny, V., Taillandier, P., and Habbal, A. The Kalai-Smorodinsky solution for many-objective Bayesian optimization. *J. Mach. Learn. Res.*, 21(150):1–42, 2020. Caruana, R. Multitask learning. *Machine learning*, 28(1): 41–75, 1997. Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich,
- <span id="page-9-24"></span><span id="page-9-22"></span><span id="page-9-21"></span><span id="page-9-20"></span><span id="page-9-18"></span><span id="page-9-16"></span><span id="page-9-13"></span><span id="page-9-12"></span><span id="page-9-10"></span><span id="page-9-8"></span><span id="page-9-6"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International Conference on Machine Learning*, pp. 794–803. PMLR, 2018. Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., and Anguelov, D. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *ArXiv*, abs/2010.06808, 2020. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., and Schiele, B. The cityscapes dataset for semantic urban scene understanding. In *Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. Crawshaw, M. Multi-task learning with deep neural networks: A survey. *arXiv preprint arXiv:2009.09796*, 2020. Dagan, N. and Volij, O. The bankruptcy problem: a cooperative bargaining approach. *Mathematical Social Sciences*, 26(3):287–297, 1993. Dai, J., He, K., and Sun, J. Instance-aware semantic segmentation via multi-task network cascades. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 3150–3158, 2016. Desid ´ eri, J.-A. Multiple-gradient descent algorithm ´ (MGDA) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5-6):313–318, 2012. Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., and Lakshminarayanan, B. Adapting auxiliary losses using gradient similarity. *arXiv preprint arXiv:1812.02224*, 2018. Fey, M. and Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. In *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In *International conference on machine learning*, pp. 1263–1272. PMLR, 2017. Grosu, D., Chronopoulos, A. T., and Leung, M.-Y. Load balancing in distributed systems: An approach using cooperative games. In *Proceedings 16th International Parallel and Distributed Processing Symposium*, pp. 10–pp. IEEE, 2002. Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actorcritic: Off-policy maximum entropy deep reinforcement, 2018. Hashimoto, K., Xiong, C., Tsuruoka, Y., and Socher, R. A joint many-task model: Growing a neural network for multiple nlp tasks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 1923–1933, 2017. Jaggi, M. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In *International Conference on Machine Learning*, pp. 427–435. PMLR, 2013. Javaloy, A. and Valera, I. Rotograd: Dynamic gradient homogenization for multi-task learning. *arXiv preprint arXiv:2103.02631*, 2021. Kendall, A., Gal, Y., and Cipolla, R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 7482–7491, 2018. Kim, S. Cooperative federated learning-based task offloading scheme for tactical edge networks. *IEEE Access*, 9: 145739–145747, 2021. Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *CoRR*, abs/1412.6980, 2015. Klicpera, J., Groß, J., and Gunnemann, S. Direc- ¨ tional message passing for molecular graphs. *ArXiv*, abs/2003.03123, 2020. Leshem, A. and Zehavi, E. Smart carrier sensing for distributed computation of the generalized nash bargaining solution. In *2011 17th International Conference on Digital Signal Processing (DSP)*, pp. 1–5. IEEE, 2011. Lin, B., Ye, F., and Zhang, Y. A closer look at loss weighting in multi-task learning. *arXiv preprint arXiv:2111.10603*, 2021. Lipp, T. and Boyd, S. Variations and extension of the convex–concave procedure. *Optimization and Engineering*, 17(2):263–287, 2016. Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. Conflictaverse gradient descent for multi-task learning. *Advances in Neural Information Processing Systems*, 34, 2021a.

- <span id="page-10-21"></span><span id="page-10-18"></span><span id="page-10-12"></span><span id="page-10-10"></span><span id="page-10-1"></span><span id="page-10-0"></span>Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., and Zhang, W. Towards impartial multitask learning. In *International Conference on Learning Representations*, 2021b. Liu, S., Davison, A., and Johns, E. Self-supervised generalisation with meta auxiliary learning. *Advances in Neural Information Processing Systems*, 32, 2019a. Liu, S., Johns, E., and Davison, A. J. End-to-end multi-task learning with attention. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1871–1880, 2019b. Maninis, K.-K., Radosavovic, I., and Kokkinos, I. Attentive single-tasking of multiple tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1851–1860, 2019. Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman,
- <span id="page-10-25"></span><span id="page-10-24"></span><span id="page-10-23"></span><span id="page-10-22"></span><span id="page-10-20"></span><span id="page-10-19"></span><span id="page-10-17"></span><span id="page-10-16"></span><span id="page-10-15"></span><span id="page-10-14"></span><span id="page-10-13"></span><span id="page-10-11"></span><span id="page-10-9"></span><span id="page-10-8"></span><span id="page-10-7"></span><span id="page-10-6"></span><span id="page-10-5"></span><span id="page-10-4"></span><span id="page-10-3"></span><span id="page-10-2"></span>Y. Provably powerful graph networks. *arXiv preprint arXiv:1905.11136*, 2019. Misra, I., Shrivastava, A., Gupta, A., and Hebert, M. Crossstitch networks for multi-task learning. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 3994–4003, 2016. Nash, J. Two-person cooperative games. *Econometrica*, 21 (1):128–140, 1953. ISSN 00129682, 14680262. URL <http://www.jstor.org/stable/1906951>. Navon, A., Achituve, I., Maron, H., Chechik, G., and Fetaya,
- E. Auxiliary learning by implicit differentiation. In *International Conference on Learning Representations (ICLR)*, 2021a. Navon, A., Shamsian, A., Chechik, G., and Fetaya, E. Learning the pareto front with hypernetworks. In *International Conference on Learning Representations*, 2021b. URL [https://openreview.net/forum?](https://openreview.net/forum?id=NjF772F4ZZR) [id=NjF772F4ZZR](https://openreview.net/forum?id=NjF772F4ZZR). Panageas, I., Piliouras, G., and Wang, X. First-order methods almost always avoid saddle points: The case of vanishing step-sizes. In *Neural Information Processing Systems (NeurIPS)*, 2019. Pinto, L. and Gupta, A. Learning to push by grasping: Using multiple tasks for effective learning. In *2017 IEEE international conference on robotics and automation (ICRA)*, pp. 2161–2168. IEEE, 2017. Qiao, H., Rozenblit, J., Szidarovszky, F., and Yang, L. Multiagent learning model with bargaining. In *Proceedings of the 2006 winter simulation conference*, pp. 934–940. IEEE, 2006. Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld,
  - O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014. Rezaee, M. J., Eshkevari, M., Saberi, M., and Hussain, O. GBK-means clustering algorithm: An improvement to the K-means algorithm based on the bargaining game. *Knowledge-Based Systems*, 213:106672, 2021. Ruder, S. An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*, 2017. Schaul, T., Borsa, D., Modayil, J., and Pascanu, R. Ray interference: a source of plateaus in deep reinforcement learning. *arXiv preprint arXiv:1904.11455*, 2019. Sener, O. and Koltun, V. Multi-task learning as multiobjective optimization. In *Advances in Neural Information Processing Systems*, pp. 527–538, 2018. Shi, C., Wang, F., Salous, S., Zhou, J., and Hu, Z. Nash bargaining game-theoretic framework for power control in distributed multiple-radar architecture underlying wireless communication system. *Entropy*, 20(4):267, 2018. Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In *European conference on computer vision*, pp. 746–760. Springer, 2012. Sodhani, S., Zhang, A., and Pineau, J. Multi-task reinforcement learning with context-based representations. *arXiv preprint arXiv:2102.06177*, 2021. Sriperumbudur, B. K. and Lanckriet, G. R. On the convergence of the concave-convex procedure. In *Nips*, volume 9, pp. 1759–1767. Citeseer, 2009. Standley, T., Zamir, A. R., Chen, D., Guibas, L. J., Malik, J., and Savarese, S. Which tasks should be learned together in multi-task learning? In *International Conference on Machine Learning ICML*, 2020. Suteu, M. and Guo, Y. Regularizing deep multi-task networks using orthogonal gradients. *arXiv preprint arXiv:1912.06844*, 2019. Szep, J. and Forg ´ o, F. ´ *Introduction to the Theory of Games*. Springer, 1985. Thomson, W. Chapter 35 cooperative models of bargaining. volume 2 of *Handbook of Game Theory with Economic Applications*, pp. 1237–1284. Elsevier, 1994. Vinyals, O., Bengio, S., and Kudlur, M. Order matters: Sequence to sequence for sets. *arXiv preprint arXiv:1511.06391*, 2015.

<span id="page-11-7"></span><span id="page-11-6"></span><span id="page-11-5"></span><span id="page-11-4"></span><span id="page-11-3"></span><span id="page-11-2"></span><span id="page-11-1"></span><span id="page-11-0"></span>Wang, Z., Tsvetkov, Y., Firat, O., and Cao, Y. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In *International Conference on Learning Representations*, 2020. Yang, R., Xu, H., Wu, Y., and Wang, X. Multi-task reinforcement learning with soft modularization. *ArXiv*, abs/2003.13661, 2020. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. In *Advances in Neural Information Processing Systems*, 2020a. Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., and Levine, S. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In *Conference on Robot Learning*, pp. 1094–1100. PMLR, 2020b. Yuille, A. L. and Rangarajan, A. The concave-convex procedure. *Neural computation*, 15(4):915–936, 2003. Zhang, Z., Shi, J., Chen, H.-H., Guizani, M., and Qiu, P. A cooperation strategy based on nash bargaining solution in cooperative relay networks. *IEEE Transactions on Vehicular Technology*, 57(4):2570–2577, 2008. Zhang, Z., Luo, P., Loy, C. C., and Tang, X. Facial landmark detection by deep multi-task learning. In *European conference on computer vision*, pp. 94–108. Springer, 2014. Zhao, X., Li, H., Shen, X., Liang, X., and Wu, Y. A modulation module for multi-task learning with applications in image retrieval. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 401–416, 2018.

#### <span id="page-12-0"></span>A. Proofs

<span id="page-12-1"></span>Lemma A.1. *If* L *is differential and L-smooth (assumption [5.3\)](#page-5-3) then* L(θ 0 ) ≤ L(θ) + ∇L(θ) <sup>&</sup>gt;(θ <sup>0</sup> − θ) + <sup>L</sup> 2 kθ <sup>0</sup> − θk 2 *.*

*Proof.* Fix θ, θ<sup>0</sup> ∈ dom(L) ⊆ <sup>R</sup> d . Since dom(L) is a convex and open set, there exists > 0 such that θ + t(θ <sup>0</sup> − θ) ∈ dom(L) for all t ∈ [−, 1 + ]. Set > 0 to be such a number. Thus, we can define a function L¯ : [−, 1 + ] → <sup>R</sup> by L¯(t) = L(θ + t(θ <sup>0</sup> − θ)). With this, L¯(1) = L(θ 0 ), L¯(0) = L(θ), and ∇L¯(t) = ∇L(θ + t(θ <sup>0</sup> − θ))<sup>&</sup>gt;(θ <sup>0</sup> − θ) for t ∈ [0, 1] ⊂ (−, 1 + ). From Assumption [5.3,](#page-5-3) k∇L(θ 0 ) − ∇L(θ)k ≤ Lkθ <sup>0</sup> − θk, therefore

$$\begin{aligned} \|\nabla \bar{\mathcal{L}}(t') - \nabla \bar{\mathcal{L}}(t)\| &= \|[\nabla \mathcal{L}(\theta + t'(\theta' - \theta)) - \nabla \mathcal{L}(\theta + t(\theta' - \theta))]^\top (\theta' - \theta)\| \\ &\leq \|\theta' - \theta\| \|\nabla \mathcal{L}(\theta + t'(\theta' - \theta)) - \nabla \mathcal{L}(\theta + t(\theta' - \theta))\| \\ &\leq L \|\theta' - \theta\| \|(t' - t)(\theta' - \theta)\| \\ &\leq L \|\theta' - \theta\|^2 \|t' - t\|. \end{aligned}$$

Hence, ∇L¯ : [0, 1] → <sup>R</sup> is Lipschitz continuous, and therefore continuous. By using the fundamental theorem of calculus with the continuous function ∇L¯ : [0, 1] → <sup>R</sup>,

$$\begin{aligned}\mathcal{L}(\theta') &= \mathcal{L}(\theta) + \int_0^1 \nabla \mathcal{L}(\theta + t(\theta' - \theta))^\top (\theta' - \theta) dt \\ &= \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top (\theta' - \theta) + \int_0^1 (\nabla \mathcal{L}(\theta + t(\theta' - \theta)) - \nabla \mathcal{L}(\theta))^\top (\theta' - \theta) dt \\ &\leq \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top (\theta' - \theta) + \int_0^1 \|\nabla \mathcal{L}(\theta + t(\theta' - \theta)) - \nabla \mathcal{L}(\theta)\| \|\theta' - \theta\| dt \\ &\leq \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top (\theta' - \theta) + \int_0^1 t L \|\theta' - \theta\|^2 dt \\ &= \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top (\theta' - \theta) + \frac{L}{2} \|\theta' - \theta\|^2.\end{aligned}\tag{7}$$

<span id="page-12-3"></span><span id="page-12-2"></span>

Theorem (5.4). *Let* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *be the sequence generated by the update rule* θ (t+1) = θ (t) − µ (t)∆θ (t) *where* ∆θ (t) = P<sup>K</sup> <sup>i</sup>=1 α (t) i g (t) i *is the Nash bargaining solution* (G(t) ) <sup>&</sup>gt;G(t)α (t) = 1/α(t) *. Set* µ (t) = min i∈[K] 1 LKα(t) *. The sequence* {θ (t)}<sup>∞</sup> t=1 *has a subsequence that converges to a Pareto stationary point* θ ∗ *. Moreover all the loss functions* (`1(θ (t) ), ..., `K(θ (t) )) *converge to* (`1(θ ∗ ), ..., `K(θ ∗ ))*.*

*Proof.* We first note that if for some step we reach a Pareto stationary solution the algorithm halts and sequence stays fixed at that point and therefore converges; Next, we assume that we never get to an exact Pareto stationary solution at any finite step.

We note that the norm of ∆θ (t) is √ K as ||∆θ (t) ||<sup>2</sup> = P<sup>K</sup> <sup>i</sup>=1 αig > <sup>i</sup> ∆θ (t) = P<sup>K</sup> <sup>i</sup>=1 α<sup>i</sup> · 1/α<sup>i</sup> = K. For each loss `<sup>i</sup> we have using Lemma [A.1](#page-12-1)

$$\ell_i(\theta^{(t+1)}) \leq \ell_i(\theta^{(t)}) - \mu^{(t)} \nabla \ell_i(\theta^{(t)})^\top \Delta \theta^{(t)} + \frac{L}{2} \|\mu^{(t)} \Delta \theta^{(t)}\|^2 = \quad (8)$$

$$\ell_i(\theta^{(t)}) - \mu^{(t)} \frac{1}{\alpha_i^{(t)}} + \frac{(\mu^{(t)})^2 LK}{2} \quad (9)$$

$$= \ell_i(\theta^{(t)}) - \frac{\mu^{(t)}}{\alpha_i^{(t)}} + \frac{\mu^{(t)}}{2} \min_j \frac{1}{\alpha_j^{(t)}} \leq \ell_i(\theta^{(t)}) - \frac{\mu^{(t)}}{2\alpha_i^{(t)}} < \ell_i(\theta^{(t)}) \quad (10)$$

This shows that our update decreases all the loss functions. We can average over inequality [9](#page-12-2) over all losses and get for L(θ) = <sup>1</sup> K P<sup>K</sup> <sup>i</sup>=1 `i(θ):

$$\mathcal{L}(\theta^{(t+1)}) \leq \mathcal{L}(\theta^{(t)}) - \mu^{(t)} \frac{1}{K} \sum_{i=1}^K \frac{1}{\alpha_i^{(t)}} + \frac{(\mu^{(t)})^2 LK}{2} \leq \mathcal{L}(\theta^{(t)}) - LK(\mu^{(t)})^2 + \frac{(\mu^{(t)})^2 LK}{2} = \mathcal{L}(\theta^{(t)}) - \frac{LK(\mu^{(t)})^2}{2}. \quad (11)$$

From this we can conclude that P<sup>t</sup> τ=1 LK(µ ) <sup>2</sup> ≤ L(θ1) − L(θ (t+1)). As L(θ (t) ) is bounded below we must have that the infinite series P<sup>∞</sup> t=1 LK(µ ) <sup>2</sup> < ∞, and also µ (t) <sup>→</sup> <sup>0</sup>. It follows that mini∈[K] <sup>1</sup>/α(t) <sup>i</sup> → 0 and therefore ||α (t) || → ∞.

We will now show that ||1/α(t) || is bounded for t → ∞. As the sequence L(θ (t) ) is decreasing we have that the sequence θ (t) is in the sublevel set {θ : L(θ) ≤ L(θ0)} which is closed and bounded and therefore compact. If follows that there exists M < ∞ such that ||g (t) i || ≤ <sup>M</sup> for all <sup>t</sup> and <sup>i</sup> ∈ [K]. We have for all <sup>i</sup> and <sup>t</sup>, |1/α(t) i | = |(g (t) i ) T θ (t) | ≤ √ K||g (t) i || ≤ √ KM < ∞, and so ||1/α(t) || is bounded. Combining these two results we have ||1/α(t) || ≥ σK((G(t) ) <sup>&</sup>gt;G(t) )||α (t) where σK((G(t) ) <sup>&</sup>gt;G(t) ) is the smallest singular value of (G(t) ) <sup>&</sup>gt;G(t) . Since the norm of α (t) goes to infinity and the norm 1/α(t) is bounded, it follows that σK((G(t) ) <sup>&</sup>gt;G(t) ) → 0.

Now, since {θ : L(θ) ≤ Lθ0} is compact there exists a subsequence θ (t<sup>j</sup> ) that converges to some point θ ∗ . As σK((G(t) ) <sup>T</sup> G(t) ) → 0 we have from continuity that σK(G<sup>&</sup>gt; <sup>∗</sup> G∗) = 0 where G<sup>∗</sup> is the matrix of gradients at θ ∗ . This means that the gradients at θ are linearly dependent and therefore θ ∗ is Pareto stationary by assumption [5.1.](#page-5-1) As for all i the sequence {`i(θ (t) )}<sup>∞</sup> <sup>t</sup>=1 is monotonically decreasing and bounded below they all converges. Since `i(θ ∗ ) is the limit of a subsequence we get that `i(θ (t) ) <sup>t</sup>→∞ −−−→ `i(<sup>θ</sup> ∗ ).

<span id="page-13-4"></span><span id="page-13-3"></span><span id="page-13-1"></span><span id="page-13-0"></span>

We now show that if we add a convexity assumption then we can prove convergence to the Pareto front.

Theorem (5.5). *Let* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *be the sequence generated by the update rule* θ (t+1) = θ (t) − µ (t)∆θ (t) *where* ∆θ (t) = P<sup>K</sup> <sup>i</sup>=1 α (t) i g (t) i *is the Nash bargaining solution* (G(t) ) <sup>&</sup>gt;G(t)α (t) = 1/α(t) *. Set* µ (t) = min i∈[K] 1 LKα(t) i *. If we also assume that all the loss functions are convex then the sequence* {θ (t)}<sup>∞</sup> <sup>t</sup>=1 *converges to a Pareto optimal point* θ ∗ *.*

*Proof.* We note that this proof uses intermediate results from the proof of theorem [5.4.](#page-5-4) Given theorem [5.4](#page-5-4) it suffices to prove that the sequence {θ (t)}<sup>∞</sup> <sup>t</sup>=1 converges, that would mean it converges to the partial limit θ ∗ that is Pareto stationary, and from convexity it would be Pareto optimal (as the optimizer of the convex combination of losses). For a convex and differential loss function, we have

<span id="page-13-2"></span>
$$\ell(\theta') \geq \ell(\theta) + \nabla \ell(\theta)^\top (\theta' - \theta) \quad (12)$$

We can bound

$$\|\theta^{(t+1)} - \theta^*\|^2 = \|\theta^{(t)} - \mu^{(t)} \Delta\theta^{(t)} - \theta^*\|^2 \quad (13)$$

$$= \|\theta^{(t)} - \theta^*\|^2 + (\mu^{(t)})^2 \|\Delta\theta^{(t)}\|^2 - 2\mu^{(t)}(\Delta\theta^{(t)})^\top(\theta^{(t)} - \theta^*) \quad (14)$$

$$= \|\theta^{(t)} - \theta^*\|^2 + (\mu^{(t)})^2 K - 2\mu^{(t)} \sum \alpha_i^{(t)} (g_i^{(t)})^\top (\theta^{(t)} - \theta^*) \quad (15)$$

$$\leq \|\theta^{(t)} - \theta^*\|^2 + (\mu^{(t)})^2 K + 2\mu^{(t)} \sum_i \alpha_i^{(t)} (\ell_i(\theta^*) - \ell_i(\theta^{(t)})) \quad (16)$$

$$\leq ||\theta^{(t)} - \theta^*||^2 + (\mu^{(t)})^2 K + 2\mu^{(t)} \sum_i \alpha_i^{(t)} (\ell_i(\theta^{(t+1)}) - \ell_i(\theta^{(t)})) \quad (17)$$

$$\leq \|\theta^{(t)} - \theta^*\|^2 + (\mu^{(t)})^2 K - 2\mu^{(t)} \sum_i \alpha_i^{(t)} \frac{\mu^{(t)}}{2\alpha_i^{(t)}} \quad (18)$$

$$= \|\theta^{(t)} - \theta^*\|^2 \quad (19)$$

In Eq. [15](#page-13-0) we use the definition of ∆θ (t) and the fact that its norm equals √ K. In Eq. [16](#page-13-1) we use convexity and Eq. [12.](#page-13-2) Eq. [17](#page-13-3) uses the fact that we show the losses are monotonically decreasing and converging to `i(θ ∗ ). In Eq. [18](#page-13-4) we use Eq. [10.](#page-12-3)

We have that the sequence ||θ (t) − θ ∗ || is monotonically decreasing and bounded below by zero. Also, it has a subsequence that converges to zero, and so it must hold that the sequence ||θ (t) − θ ∗ || also converge to zero, or equivalently θ (t) → θ ∗ .

Proposition (3.1). *Denote the objective for the optimization problem in Eq. [5](#page-3-0) by* φ(α) = P i βi(α) + ϕ(α)*. Then,* φ α (τ+1) ≤ φ α (τ) *for all* τ ≥ 1*.*

*Proof.* In our concave-convex procedure, we use the following linearization at the τ -th iteration:

$$\tilde{\varphi}_\tau(\alpha) = \varphi(\alpha^{(\tau)}) + \nabla \varphi(\alpha^{(\tau)})^\top (\alpha - \alpha^{(\tau)}).$$

Then,

<span id="page-14-2"></span><span id="page-14-1"></span>
$$\tilde{\varphi}_\tau(\alpha^{(\tau)}) = \varphi(\alpha^{(\tau)}). \quad (20)$$

Moreover, since ϕ is concave and differentiable, we have that

$$\varphi(\alpha^{(\tau+1)}) \leq \varphi(\alpha^{(\tau)}) + \nabla \varphi(\alpha^{(\tau)})^\top (\alpha^{(\tau+1)} - \alpha^{(\tau)}) = \tilde{\varphi}_\tau(\alpha^{(\tau+1)}). \quad (21)$$

Furthermore, since we minimize the convex objective P i βi(α) + ˜ϕ(α) at each iteration of our concave-convex procedure (in the convex feasible set),

$$\sum_i \beta_i(\alpha^{(\tau)}) + \tilde{\varphi}_\tau(\alpha^{(\tau)}) \geq \sum_i \beta_i(\alpha^{(\tau+1)}) + \tilde{\varphi}_\tau(\alpha^{(\tau+1)}). \quad (22)$$

Using Eq. [20–](#page-14-1)Eq. [22,](#page-14-2) we have that

$$\begin{aligned}\phi(\alpha^{(\tau)}) &= \sum_i \beta_i(\alpha^{(\tau)}) + \varphi(\alpha^{(\tau)}) = \sum_i \beta_i(\alpha^{(\tau)}) + \tilde{\varphi}_\tau(\alpha^{(\tau)}) \geq \sum_i \beta_i(\alpha^{(\tau+1)}) + \tilde{\varphi}_t(\alpha^{(\tau+1)}) \\ &\geq \sum_i \beta_i(\alpha^{(\tau+1)}) + \varphi(\alpha^{(\tau+1)}) = \phi(\alpha^{(\tau+1)}).\end{aligned}$$

This proves the statement.

## <span id="page-14-0"></span>B. Experimental Details

We provide here full experimental details for all experiments described in the main text.

Implementation Details. We apply all gradient manipulation methods to the gradients of the shared weights, with the exception of IMTL-G, which was applied to the feature-level gradients, as was originally proposed by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but it did not perform as well. We set the CAGrad's c hyperparameter to 0.4, which was reported to yield the best performance for NYUv2 and Cityscapes [\(Liu](#page-9-3) [et al.,](#page-9-3) [2021a\)](#page-9-3). For DWA [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12) we set the temperature hyperparameter to 2 which was found empirically to be optimum across all architectures. For RLW [\(Lin et al.,](#page-9-11) [2021\)](#page-9-11) we sample the weights from a normal distribution.

QM9. We adapt the QM9 example in PyTorch Geometric [\(Fey & Lenssen,](#page-9-21) [2019\)](#page-9-21), and train the popular GNN model from [Gilmer et al.](#page-9-22) [\(2017\)](#page-9-22). We use the publicly available[<sup>1</sup>](#page-14-3) implementation, the implementation is provided by [Fey & Lenssen](#page-9-21) [\(2019\)](#page-9-21). We use 110K molecules for training, 10K for validation, and 10K as a test set. Each task's targets are normalized to have zero mean and unit standard deviation. We train each method for 300 epochs with batch-size of 120 and search for learning-rate (lr) in {1e − 3, 5e − 4, 1e − 4}. We use a ReduceOnPlateau scheduler to decrease the lr when the validation ∆<sup>m</sup> metric stops improving. Additionally, we use the validation ∆<sup>m</sup> for early stopping.

Scene Understanding. We follow the training and evaluation procedure used in previous work on MTL [\(Liu et al.,](#page-10-12) [2019b;](#page-10-12) [Yu et al.,](#page-11-0) [2020a;](#page-11-0) [Liu et al.,](#page-9-3) [2021a\)](#page-9-3). However, unlike [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12), we add data augmentations (DA) during training for

<span id="page-14-3"></span><sup>1</sup>[https://github.com/pyg-team/pytorch\\_geometric/blob/master/examples/qm9\\_nn\\_conv.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py)

<span id="page-15-0"></span>![](_page_15_Figure_1.jpeg)

Figure 5. *Illustrative example*. Visualization of the loss surfaces in our illustrative example of Figure [1](#page-1-0)

all the compared methods, similar to [\(Liu et al.,](#page-9-3) [2021a](#page-9-3)[;b\)](#page-10-10). We train each method for 200 epochs with an initial learning-rate of 1e − 4. The learning-rate is reduced to 5e − 5 after 100 epochs. For MTL methods, we train a Multi-Task Attention Network (MTAN) [\(Liu et al.,](#page-10-12) [2019b\)](#page-10-12) built upon SegNet [\(Badrinarayanan et al.,](#page-8-2) [2017\)](#page-8-2). Similar to previous works [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3), the STL baseline refers to training task-specific SegNet models. We use a batch size of 2 and 8 for NYUv2 and CityScapes respectively. To align with previous work on MTL [Liu et al.](#page-10-12) [\(2019b\)](#page-10-12); [Yu et al.](#page-11-0) [\(2020a\)](#page-11-0); [Liu et al.](#page-9-3) [\(2021a\)](#page-9-3) we report the test performance averaged over the last 10 epochs.

MT10. Following previous works [\(Yu et al.,](#page-11-0) [2020a;](#page-11-0) [Liu et al.,](#page-9-3) [2021a;](#page-9-3) [Sodhani et al.,](#page-10-25) [2021\)](#page-10-25), we use multitask Soft Actor-Critic (SAC) [\(Haarnoja et al.,](#page-9-26) [2018\)](#page-9-26) as the base RL algorithm for PCGrad, CAGrad, and Nash-MTL. We follow the same experiment setup from and evaluation protocol as in [Sodhani et al.](#page-10-25) [\(2021\)](#page-10-25); [Liu et al.](#page-9-3) [\(2021a\)](#page-9-3). Each method is trained over 2 million steps with a batch size of 1280. The agent is evaluated once every 10K environment steps to obtain the average success over tasks. The reported success rate for the agent is the best average performance over all evaluation steps. We repeat this procedure over 10 random seeds, and the performance of each method is obtained by averaging the mean success over all random seeds. For all Nash-MTL experiments, we use a single CCP step in order to speed up computation.

Illustrative Example. We provide here the details for the illustrative example of Figure [1.](#page-1-0) We use a slightly modified version of the illustrative example in [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3). We first present the learning problem from [\(Liu et al.,](#page-9-3) [2021a\)](#page-9-3): Let θ = (θ1, θ2) ∈ <sup>R</sup> 2 , and consider the following objectives:

$$\begin{aligned} \tilde{\ell}_1(\theta) &= c_1(\theta)f_1(\theta) + c_2(\theta)g_1(\theta) \quad \text{and} \quad \tilde{\ell}_2(\theta) = c_1(\theta)f_2(\theta) + c_2(\theta)g_2(\theta), \text{ where} \\ f_1(\theta) &= \log(\max(|0.5(-\theta_1 - 7) - \tanh(-\theta_2)|, 5e - 6)) + 6, \\ f_2(\theta) &= \log(\max(|0.5(-\theta_1 + 3) - \tanh(-\theta_2) + 2|, 5e - 6)) + 6, \\ g_1(\theta) &= ((-\theta_1 + 7)^2 + 0.1 \cdot (-\theta_2 - 8)^2)/10 - 20, \\ g_2(\theta) &= ((-\theta_1 - 7)^2 + 0.1 \cdot (-\theta_2 - 8)^2)/10 - 20, \\ c_1(\theta) &= \max(\tanh(0.5\theta_2), 0) \quad \text{and} \quad c_2(\theta) = \max(\tanh(-0.5\theta_2), 0) \end{aligned}$$

We now set `<sup>1</sup> = 0.1 · ˜`<sup>1</sup> and `<sup>2</sup> = ˜`<sup>2</sup> as our objectives, see Figure [5.](#page-15-0) We use five different initialization points {(−8.5, 7.5),(0.0, 0.0),(9.0, 9.0),(−7.5, −0.5),(9, −1.0)}. We use the Adam optimizer and train each method for 35K iteration with learning rate of 1e − 3.

## <span id="page-15-1"></span>C. Computing Task Gradient at the Features-Level

One common approach for speeding and scaling up MTL methods is using feature-level gradients (from the representation layer) as a surrogate for the task-level gradients computed over the entire shared backbone [\(Sener & Koltun,](#page-10-4) [2018;](#page-10-4) [Liu et al.,](#page-10-10) [2021b;](#page-10-10) [Javaloy & Valera,](#page-9-5) [2021\)](#page-9-5). In this section we evaluate Nash-MTL while using the feature-level gradients for computing the Nash bargaining solution. On the QM9 dataset, we found this approach to accelerate training by ∼ ×6. However, this acceleration method greatly hurts the performance of Nash-MTL, yielding a test ∆<sup>m</sup> of 179.2 (compared to 62.0 when using full gradients). This result is not surprising, since we are mainly interested in the inner products of gradients. Consider

<span id="page-16-3"></span>![](_page_16_Figure_1.jpeg)

Figure 6. *NYUv2.* The mean and standard divination of test ∆<sup>m</sup> throughout the training process, for Nash-MTL with 1, 20, and 40 CCP steps.

g > i g<sup>j</sup> = (∇θz∇z`i) <sup>&</sup>gt;∇θz∇z`<sup>j</sup> , where z is the feature representation and θ the shared parameters vector. We see that for ∇z` > <sup>i</sup> ∇z`<sup>j</sup> to accurately approximate g > i g<sup>j</sup> we need ∇θz <sup>&</sup>gt;∇θz ≈ I which is a strong and restricting requirement.

## D. Additional Experiments

### <span id="page-16-1"></span>D.1. Full Results for Multi-task Regression

We provide here the full results for the QM9 experiment of Section [6.1.](#page-6-2) The results for all methods over all 11 tasks are presented in Table [6.](#page-16-2) Nash-MTL achieves the best ∆<sup>m</sup> and MR performance. Despite being a simple approach, *SI* performs well compared to more sophisticated baselines. It achieves the third/second best ∆<sup>m</sup> and MR respectively. The other scale-invariant method, *IMTL-G*, also performs well in this learning setup.

Table 6. *QM9*. Test performance averaged over 3 random seeds.

<span id="page-16-2"></span>

|          | µ     | α     |  HOMO |  LUMO | 2 h R i | ZPVE MAE | U 0 ↓  | U     | H     | G     | c v   | MR ↓ | ∆ m % ↓ |
|----------|-------|-------|--------|--------|---------|----------|--------|-------|-------|-------|-------|------|---------|
| STL      | 0 067 | 0 181 | 60 57  | 53 91  | 0 502   | 4 53     | 58 8   | 64 2  | 63 8  | 66 2  | 0 072 |      |         |
| LS       | 0 106 | 0 325 | 73 57  | 89 67  | 5 19    | 14 06    | 143 4  | 144.2 | 144.6 | 140.3 | 0.128 | 6.8  | 177.6   |
| SI       | 0.309 | 0.345 | 149.8  | 135.7  | 1 00    | 4 50     | 55 3   | 55 75 | 55 82 | 55 27 | 0.112 | 4.0  | 77.8    |
| RLW      | 0.113 | 0.340 | 76.95  | 92.76  | 5.86    | 15.46    | 156.3  | 157.1 | 157.6 | 153.0 | 0.137 | 8.2  | 203.8   |
| DWA      | 0.107 | 0.325 | 74.06  | 90.61  | 5.09    | 13.99    | 142.3  | 143.0 | 143.4 | 139.3 | 0.125 | 6.4  | 175.3   |
| UW       | 0.386 | 0.425 | 166.2  | 155.8  | 1.06    | 4.99     | 66.4   | 66.78 | 66.80 | 66.24 | 0.122 | 5.3  | 108.0   |
| MGDA     | 0.217 | 0.368 | 126.8  | 104.6  | 3.22    | 5.69     | 88.37  | 89.4  | 89.32 | 88.01 | 0.120 | 5.9  | 120.5   |
| PCGrad   | 0.106 | 0.293 | 75.85  | 88.33  | 3.94    | 9.15     | 116.36 | 116.8 | 117.2 | 114.5 | 0.110 | 5.0  | 125.7   |
| CAGrad   | 0.118 | 0.321 | 83.51  | 94.81  | 3.21    | 6.93     | 113.99 | 114.3 | 114.5 | 112.3 | 0.116 | 5.7  | 112.8   |
| IMTL-G   | 0.136 | 0.287 | 98.31  | 93.96  | 1.75    | 5.69     | 101.4  | 102.4 | 102.0 | 100.1 | 0.096 | 4.7  | 77.2    |
| Nash-MTL | 0 102 | 0 248 | 82.95  | 81 89  | 2.42    | 5.38     | 74.5   | 75.02 | 75.10 | 74.16 | 0 093 | 2 5  | 62 0    |

#### <span id="page-16-0"></span>D.2. Effect of the Number of CCP steps

In this section, we investigate the effect of varying the number of CCP steps in our efficient approximation to G>Gα = 1/α (presented in Section [3.2\)](#page-2-2). We use the NYUv2 dataset and train Nash-MTL with CCP sequences of 1, 20, and 40 steps at each (parameters) optimization step.

We found that increasing the CCP sequence improves the approximation to the optimal α. Using a single CCP iteration results with G>Gα ≈ 1/α in 91.5% of the optimization steps, whereas increasing the number of iterations to 20 increases the proportion of optimal solutions to 93.5%. However, we found the improved solution to have no significant improvement

<span id="page-17-0"></span>![](_page_17_Figure_1.jpeg)

Figure 7. *Task Weights*. Task weights obtained from Nash-MTL throughout the optimization process, for (a) NYUv2, and; (b) MT10 with weight update frequency of 100. For better visualization, each point corresponds to a moving average with window size 200.

<span id="page-17-1"></span>Table 7. *QM9*. Runtime per epoch in minutes.

|             | Runtime | [Min.] |
|-------------|---------|--------|
| LS          | 0       | 54     |
| MGDA        | 7       | 25     |
| PCGrad      | 7       | 47     |
| CAGrad      | 6       | 85     |
| Nash-MTL    | 6       | 76     |
| Nash-MTL-5  | 1       | 81     |
| Nash-MTL-50 | 0       | 69     |

in MTL performance. Figure [6](#page-16-3) presents the test ∆<sup>m</sup> throughout the training process.

#### D.3. Modifying the CCP Objective

In this section we examine the effect of changing the objective of the CCP procedure described in [3.2](#page-2-2) (Eq. [5\)](#page-3-0). Here we first solve the convex optimization problem of Eq. [4](#page-3-1) to obtain α0. If G>Gα<sup>0</sup> ≈ 1/α<sup>0</sup> we stop. Else we use the CCP procedure with objective ϕ(α), starting at α<sup>0</sup> (dropping the addition P i βi term from Eq. [5\)](#page-3-0). While this objective is more natural, in practice we observe a performance degradation in terms of MTL performance. We obtain ∆<sup>m</sup> = 64.4 for the QM9 dataset (vs. 62 reported in the paper), ∆<sup>m</sup> = −3.5 (vs. −4) for NYUv2 and ∆<sup>m</sup> = 8.8 (vs. 6.8) for Cityscapes.

#### D.4. Visualizing Task Weights

Our method, Nash-MTL, can essentially be viewed as a principled approach for producing dynamic task weights. Here we visualize these task weights throughout the training process using the NYUv2 dataset (Figure [7\)](#page-17-0) and the MT10 dataset (Figure [7\(b\)\)](#page-17-1).

#### D.5. Verifying the Task Independence Assumption

Here we provide an empirical justification for our assumption in Section [3](#page-2-1) which we state here once again: we assume that the task gradients are linearly independent for each point θ that is not Pareto stationary. To investigate whether this assumption holds in our experiments, we observe the smallest singular value of gradients Gram matrix σK(G>G). The results are presented in Figure [8.](#page-18-0) We see that for both datasets the σ<sup>k</sup> decreases as the learning progresses. For the NYUv2 experiment, the smallest singular value remains fairly large throughout the entire training process. On the QM9 dataset, σ<sup>K</sup> decreases more significantly, to around ∼ 1e − 8.

<span id="page-18-0"></span>![](_page_18_Figure_1.jpeg)

Figure 8. *Smallest singular value of* G <sup>&</sup>gt;G *throughout the training process.*