## Aviv Navon <sup>\*</sup> <sup>1</sup> Aviv Shamsian <sup>\*</sup> <sup>1</sup> Idan Achituve <sup>1</sup> Haggai Maron <sup>2</sup> Kenji Kawaguchi <sup>3</sup> Gal Chechik <sup>1</sup> <sup>2</sup> Ethan Fetaya <sup>1</sup>

## Abstract

In Multi-task learning (MTL), a joint model is trained to simultaneously make predictions for several tasks. Joint training reduces computation costs and improves data efficiency; however, since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance than its corresponding single-task counterparts. A common method for alleviating this issue is to combine per-task gradients into a joint update direction using a particular heuristic. In this paper, we propose viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction of parameter update. Under certain assumptions, the bargaining problem has a unique solution, known as the Nash Bargaining Solution, which we propose to use as a principled approach to multi-task learning. We describe a new MTL optimization procedure, Nash-MTL, and derive theoretical guarantees for its convergence. Empirically, we show that Nash-MTL achieves state-ofthe-art results on multiple MTL benchmarks in various domains.

## 1. Introduction

In many real-world applications, one needs to solve several tasks simultaneously using limited computational or data resources. For example, perception for autonomous vehicles requires lane detection, object detection, and free-space estimation, which must all run in parallel and in real-time. This is normally solved via multi-task learning (MTL), where one model is jointly trained on several learning tasks (Caruana, 1997; Ruder, 2017; Crawshaw, 2020). Multi-task learning was also shown to improve generalization in theory (Baxter,

<sup>\*</sup>Equal contribution <sup>1</sup>Bar-Ilan University, Ramat Gan, Israel <sup>2</sup>Nvidia, Tel-Aviv, Israel <sup>3</sup>National University of Singapore. Correspondence to: Aviv Navon <aviv.navon@biu.ac.il>, Aviv Shamsian <aviv.shamsian@live.biu.ac.il>.

Proceedings of the 39<sup>th</sup> International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

2000) and in practice (e.g., auxiliary learning, Liu et al., 2019a; Achituve et al., 2021; Navon et al., 2021a).

Unfortunately, MTL often causes performance degradation compared to single-task models (Standley et al., 2020). A main reason for such degradation is gradients conflict (Yu et al., 2020a; Wang et al., 2020; Liu et al., 2021a). These per-task gradients may have conflicting directions or a large difference in magnitudes, with the largest gradient dominating the update direction. The degraded performance of MTL due to poor training, compared with its potential to improve performance due to better generalization, has a major impact on many real-world systems. Improving MTL optimization algorithms is therefore an important task with significant implications to many systems.

Currently, most MTL optimization algorithms (Sener & Koltun, 2018; Yu et al., 2020a; Liu et al., 2021a) follow a general scheme. First, compute the gradients for all tasks g<sub>1</sub>, ..., g<sub>K</sub>. Next, combine those gradients into a joint direction, $\Delta = \mathcal { A } ( g _ { 1 } , . . . , g _ { K } )$ using an aggregation algorithm A. Finally, update model parameters using a single-task optimization algorithm, replacing the gradients with ∆. Multiple heuristics were proposed for the aggregation algorithm A. However, to the best of our knowledge, a principled, axiomatic, approach to gradient aggregation is still missing.

Here we address the gradient combination step by viewing it as a cooperative bargaining game (Thomson, 1994). Each task represents a player, whose utility is derived from its gradient, and players negotiate to reach an agreed direction. This formulation allows us to use results from game theory literature that analyze this problem from an axiomatic perspective. In his seminal paper, Nash (1953) presented an axiomatic approach to the bargaining problem and showed that under certain axioms, the bargaining problem has a unique solution known as the Nash Bargaining Solution. This solution is known to be proportionally fair, where any alternative will have a negative average relative change. This proportionally fair update allows us to find a solution that works for all tasks without being dominated by a single large gradient.

Building on Nash’s results, we propose a novel MTL optimization algorithm, named Nash-MTL, where the gradients are combined at each step using the Nash bargaining solution. We first characterize the Nash bargaining solution for MTL and derive an efficient algorithm to approximate its value. Then, we analyze our approach theoretically and establish convergence guarantees in the convex and nonconvex cases. Finally, we show empirically that our Nash-MTL approach achieves state-of-the-art results on four MTL benchmarks on a variety of challenges ranging from computer vision and quantum chemistry to reinforcement learning. To support future research and the reproducibility of the results, we make our source code publicly available at: https://github.com/AvivNavon/nash-mtl.

![](images/186bc7472e5d373312a81fec4b481cbd264fdcdec25dc44255644ec4efba6afc.jpg)

<details>
<summary>scatter</summary>

| Algorithm | Initial Point (X, Y) |
| --- | --- |
| LS | ~0.1, ~-4.5, ~0, ~3, ~7, ~8 |
| PCGrad | ~0.1, ~-4.5, ~0, ~3, ~7, ~8 |
| CAGrad | ~0.1, ~-4.5, ~0, ~3, ~7, ~8 |
| MGDA | ~0.1, ~-4.5, ~0, ~3, ~7, ~8 |
| Nash-MTL (Ours) | ~0.1, ~-4.5, ~0, ~3, ~7, ~8 |
</details>

Figure 1. Illustrative example: Optimization trajectories in loss space. Shown are 5 different initializations (black dots •), and their trajectories are colored from orange to purple. Losses have a large difference in scale. See Appendix B for details. For linear scalarization (LS), PCGrad, and CAGrad, the optimization process is controlled by the gradient of $\ell _ { 2 } ,$ since it has a larger magnitude, resulting in imbalanced solutions between tasks (mostly ending at the bottom right). These three methods also fail to converge to an optimal solution for the rightmost initialization points. In contrast, MGDA is inclined towards the task with the smallest gradient magnitude (\`<sub>1</sub>). Our method, Nash-MTL, is invariant to changes in loss scale and produces solutions that are well balanced across the Pareto front.

## 2. Background

## 2.1. Pareto Optimality

Optimization for MTL is a specific case of multipleobjective optimization (MOO). Given objective functions $\ell _ { 1 } , . . . , \ell _ { K }$ , the performance of solution a x is measured by the vector of objective values $( \ell _ { 1 } ( x ) , . . . , \ell _ { K } ( x ) )$ ). One main property of MOO is that since there is no natural linear ordering on vectors it is not always possible to compare solutions so there is no clear optimal value.

We say that a solution x dominates $x ^ { \prime }$ if it is better on one or more objectives and not worse on any other objectives. A solution that is not dominated by any other is called Pareto optimal, and the set of all such solutions is called the Paretofront. It is important to note that there is no clear way to select between different Pareto optimal solutions without additional assumptions or prior about the user preferences (Navon et al., 2021b). For non-convex problems, a point is defined as local Pareto optimal if it is Pareto optimal in some open set containing it. Further, a point is called Pareto stationary if there exists a convex combination of the gradients at this point that equals zero. Pareto stationarity is a necessary condition for Pareto optimality.

## 2.2. Nash Bargaining Solution

We provide a brief background on cooperative bargaining games and the Nash bargaining solution, see Thomson (1994) for more details. In a bargaining problem, we have K players, each with their own utility function $u _ { i } : A \cup \{ D \}  \mathbb { R }$ , which they wish to maximize. A is the set of possible agreements and D is the disagreement point which the players default to if they fail to reach an agreement. We define the set of possible payoffs as $U \ = \ \{ ( u _ { 1 } ( x ) , . . . , u _ { K } ( x ) ) : x \in A \bar  \} \ \subset \ \mathbb { R } ^ { K }$ and $d = ( u _ { 1 } ( D ) , . . . , u _ { K } ( D ) )$ . We assume U is convex, compact and that there exists a point in U that strictly dominates d, namely there exists a $u \in U$ such that ∀i : $; u _ { i } > d _ { i }$

Nash (1953) showed that for such payoff set U, the twoplayer bargaining problem has a unique solutionthat satisfies the following properties or axioms: Pareto optimality, symmetry, independence of irrelevant alternatives, and invariant to affine transformations. This was later extended to multiple players (Szep & Forg´ o´, 1985).

Axiom 2.1. Pareto optimality: The agreed solution must not be dominated by another option, i.e. there cannot be any other agreement that is better for at least one player and not worse for any of the players.

As it is a cooperative game, it makes little sense that the players will curtail another player without any personal gains, so it is natural to assume the agreed solution will not be dominated by another.

Axiom 2.2. Symmetry: The solution should be invariant to permuting the order of the players.

Axiom 2.3. Independence of irrelevant alternatives (IIA): If we enlarge the of possible payoffs to ${ \tilde { U } } \ { \underset { \neq } { \supset } } \ U ,$ and the solution is in the original set $U , u ^ { * } \in U ,$ , then the agreed point when the set of possible payoffs is U will stay $u ^ { * }$

Axiom 2.4. Invariance to affine transformation: If we transform each utility function $u _ { i } ( x )$ to $\tilde { u } _ { i } ( x ) = c _ { i } \cdot u _ { i } ( x ) +$ $b _ { i }$ with $c _ { i } > 0$ then if the original agreement had utilities $( y _ { 1 } , . . . , y _ { k } )$ the agreement after the transformation has utilities $( c _ { 1 } y _ { 1 } + b _ { 1 } , . . . , c _ { k } y _ { k } + b _ { k } )$

We argue that in the MTL setting, it is natural to require axioms 2.1-2.3. Axiom 2.4, in our mind, is the only nonnatural assumption used by the Nash bargaining solution in the context of MTL. We argue that indeed it is a desired property that is helpful for MTL. Axiom 2.4 means that the solution does not take into account the gradients’ norms but rather treats all of them the same, as if they were normalized. Without enforcing this assumption, the solution can easily be dominated by a single direction (see Figure 1). We further validate the importance of this assumption by investigating a scale-invariant baseline in Section 6.

The unique point satisfying all these axioms is called the Nash bargaining solution and is given as

$$
u ^ {*} = \arg \max _ {u \in U} \sum_ {i} \log (u _ {i} - d _ {i}) \tag {1}
$$

$$
s. t. \forall i: u _ {i} > d _ {i}
$$

## 3. Method

We now describe our Nash-MTL method in detail. We first formalize the gradient combination step as a bargaining game and analyze the Nash bargaining solution for this game. We then describe our algorithm to approximate the solution efficiently. We note that the computational cost of that approximation is critical because this approximation is executed for each gradient update. To simplify the notation, we do not distinguish between shared and task-specific parameters. We note, however, that task-specific parameters have no contribution to the Nash bargaining solution calculation.

## 3.1. Nash Bargaining Multi-Task Learning

Given an MTL optimization problem and parameters $\theta ,$ we search for an update vector ∆θ in the ball of radius  centered around zero, $B _ { \epsilon }$ . We frame this as a bargaining problem with the agreement set $B _ { \epsilon }$ and the disagreement point at 0, i.e., staying at the current parameters θ. We define the utility function for each player as $u _ { i } ( \Delta \theta ) \ : = \ : g _ { i } ^ { \top } \Delta \theta$ where $g _ { i }$ is the gradient of the loss of task i at θ. We note that since the agreement set is compact and convex and the utilities are linear then the set of possible payoffs is also compact and convex.

Our main assumption, besides the ones used by Nash, is that if θ is not Pareto stationary then the gradients are linearly independent (see further discussion on this assumption in Section 5). Under this assumption, we also have that the disagreement point, $\Delta \theta = 0$ is dominated by another in $B _ { \epsilon }$ We now show that if θ is not on the Pareto front, the unique Nash bargaining solution has the following form:

Claim 3.1. Let G be the $d \ \times \ K$ matrix whose columns are the gradients $g _ { i } .$ The solution to arg max $\begin{array} { r } { \Delta \theta \in B _ { \epsilon } \sum _ { i } \log ( \Delta \theta ^ { \top } g _ { i } ) } \end{array}$ is (up to scaling) $\sum _ { i } \alpha _ { i } g _ { i }$ where $\alpha \in \mathbb { R } _ { + } ^ { K }$ is the solution to $G ^ { \top } G \alpha = 1 / \alpha$ where $1 / \alpha$ is the element-wise reciprocal.

Proof. The derivative of this objective is $\scriptstyle \sum _ { i = 1 } ^ { K } { \frac { 1 } { \Delta \theta ^ { \top } g _ { i } } } g _ { i }$ For all vectors $\Delta \theta$ such that $\forall i : \Delta \theta ^ { T } g _ { i } > 0$ the utilities are monotonically increasing with the norm of $\Delta \theta .$ . Thus, from the Pareto optimality assumption by Nash, the optimal solution has to be on the boundary of $B _ { \epsilon } .$ From this we see that the gradient at the optimal point $\scriptstyle \sum _ { i = 1 } ^ { K } { \frac { 1 } { \Delta \theta ^ { \top } g _ { i } } } g _ { i }$ must be in the radial direction, i.e., $\begin{array} { r } { \sum _ { i = 1 } ^ { K } \frac { 1 } { \Delta \theta ^ { \top } g _ { i } } g _ { i } \parallel \bar { \Delta } \theta } \end{array}$ or $\begin{array} { r } { \sum _ { i = 1 } ^ { K } \frac { 1 } { \Delta \theta ^ { \top } g _ { i } } g _ { i } = \lambda \Delta \theta } \end{array}$ . Since the gradients are independent we must have $\begin{array} { r } { \Delta \theta = \sum _ { i } \alpha _ { i } g _ { i } } \end{array}$ and $\begin{array} { r } { \forall i : \frac { 1 } { { \Delta \theta } ^ { \top } g _ { i } } = \lambda \alpha _ { i } } \end{array}$ or $\begin{array} { r } { \forall i \ : \ \Delta \theta ^ { \top } g _ { i } \ = \ \frac { 1 } { \lambda \alpha _ { i } } } \end{array}$ . As the inner product must be positive for a descent direction we can conclude $\lambda > 0 ;$ we set $\lambda = 1$ to ascertain the direction of $\Delta \theta$ (the norm might be larger then ). Now finding the bargaining solution is reduced to finding $\alpha \in \mathbb { R } ^ { K }$ with $\alpha _ { i } > 0$ such that ∀i : $\begin{array} { r } { \Delta \theta ^ { \top } g _ { i } = \sum _ { j } \alpha _ { j } g _ { j } ^ { \top } \bar { g } _ { i } = \frac { 1 } { \alpha _ { i } } } \end{array}$ . This is equivalent to requiring that $G ^ { \top } G \alpha = 1 / \alpha$ where $1 / \alpha$ is the element-wise reciprocal. □

We now provide some intuition for this solution. First, if all $g _ { i }$ are orthogonal we get $\alpha _ { i } = 1 / | | g _ { i } | |$ and $\begin{array} { r } { \Delta \theta = \sum \frac { g _ { i } } { | | g _ { i } | | } } \end{array}$ which is the obvious scale invariant solution. When they are not orthogonal, we get

$$
\alpha_ {i} | | g _ {i} | | ^ {2} + \sum_ {j \neq i} \alpha_ {j} g _ {j} ^ {\top} g _ {i} = 1 / \alpha_ {i} \tag {2}
$$

We can consider $\begin{array} { r } { \sum _ { j \neq i } { \alpha _ { j } g _ { j } ^ { \top } } g _ { i } = \left( \sum _ { j \neq i } { \alpha _ { j } g _ { j } } \right) ^ { \top } g _ { i } } \end{array}$ as the interaction between task i and the other tasks; If it is positive there is a positive interaction and the other gradients aid the i’th task, and if it is negative they hamper it. When there is a negative interaction, the LHS of Eq. 2 decreases and as a result, $\alpha _ { i }$ increases to compensate for it. Conversely, where there is a positive interaction $\alpha _ { i }$ will decrease.

## 3.2. Solving $\mathbf { G } ^ { \top } \mathbf { G } \alpha = 1 / \alpha$

Here we describe how to efficiently approximate the optimal solution for $G ^ { \top } G \alpha = 1 / \alpha$ through a sequence of convex optimization problems. We define a $\beta _ { i } ( \alpha ) = g _ { i } ^ { \top } G \alpha$ , and wish to find α such that $\alpha _ { i } = 1 / \beta _ { i }$ for all i, or equivalently $\log ( \alpha _ { i } ) + \log ( \beta _ { i } ( \alpha ) ) = 0$ . Denote $\varphi _ { i } ( \alpha ) = \log ( \alpha _ { i } ) +$ log(β<sub>i</sub>) and $\begin{array} { r } { \varphi ( \alpha ) = \sum _ { i } \varphi _ { i } ( \alpha ) } \end{array}$ . With that, our goal is to find a non-negative α such that $\varphi _ { i } ( \alpha ) = 0$ for all i. We can write this as the following optimization problem

![](images/d8a6d70fb59ac03d5e240e448e9e38f10728469d1ba2808ccf9aa531707c0a45.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  subgraph MGDA
  A["Task gradient"] --> B["Δθ"]
  B --> C["Projection"]
  end
  subgraph PCGrad
  D["Task gradient"] --> E["Δθ"]
  E --> F["Projection"]
  end
  subgraph CAGrad
  G["Task gradient"] --> H["Δθ"]
  H --> I["Projection"]
  end
  subgraph Nash_MTL["Nash-MTL"]
  J["Task gradient"] --> K["Δθ"]
  K --> L["Projection"]
  end
```
</details>

Figure 2. Visualization ofthe update direction: We show the update direction (blue) obtained by various methods on three gradients in $\mathbb { R } ^ { 3 }$ We rescaled the returned vectors for better visibility, showing only the direction. We further show the size of the projection (red) of the update to each gradient direction (black). Nash-MTL produce an update direction with the most balanced projections.

Algorithm 1 Nash-MTL

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: $\theta^{(0)}$ – initial parameter vector, $\{\ell_i\}_{i=1}^K$ – differentiable loss functions, $\eta$ – learning rate
for $t = 1, ..., T$ do
    Compute task gradients $g_i^{(t)} = \nabla_{\theta^{(t-1)}} \ell_i$
    Set $G^{(t)}$ the matrix with columns $g_i^{(t)}$
    Solve for $\alpha$: $(G^{(t)})^\top G^{(t)} \alpha = 1/\alpha$ to obtain $\alpha^{(t)}$
    Update the parameters $\theta^{(t)} = \theta^{(t)} - \eta G^{(t)} \alpha^{(t)}$
end for
Return: $\theta^{(T)}$
</div>

$$
\min _ {\alpha} \sum_ {i} \varphi_ {i} (\alpha) \tag {3}
$$

$$
\mathrm{s.t.} \forall i, \quad - \varphi_ {i} (\alpha) \leq 0
$$

$$
\alpha_ {i} > 0 \quad .
$$

The constraints in this problem are convex and linear and the objective is concave. We first try to solve the following convex surrogate objective

$$
\min _ {\alpha} \sum_ {i} \beta_ {i} (\alpha) \tag {4}
$$

$$
\mathrm{s.t.} \forall i, \quad - \varphi_ {i} (\alpha) \leq 0
$$

$$
\alpha_ {i} > 0 \quad .
$$

Here, we minimize $\sum _ { i } \beta _ { i }$ under the constraint $\beta _ { i } \mathrm { ~  ~ { ~ = ~ } ~ }$ $g _ { i } ^ { \top } G \alpha \geq 1 / \alpha _ { i }$ . While this objective is not equivalent to the original problem, we found it very useful. In many cases, it produces exact solutions with $\varphi ( \alpha ) = 0$ as required.

To further improve our approximation, we considered the following problem,

$$
\min _ {\alpha} \sum_ {i} \beta_ {i} (\alpha) + \varphi (\alpha) \tag {5}
$$

$$
\mathrm{s.t.} \forall i, \quad - \varphi_ {i} (\alpha) \leq 0
$$

$$
\alpha_ {i} > 0 \quad .
$$

Adding $\varphi ( \alpha )$ to the objective may further reduce it, moving it closer to zero; however, it renders the problem to be nonconvex. Despite that, our solution can be improved iteratively by replacing the concave term ϕ(α) with its first-order approximation $\tilde { \varphi _ { \tau } } ( \alpha ) = \varphi ( \alpha ^ { ( \tau ) } ) + \overleftarrow { \nabla } \varphi ( \alpha ^ { ( \tau ) } ) ^ { \top } ( \alpha - \alpha ^ { ( \tau ) } )$ Where, $\alpha ^ { ( \tau ) }$ is the solution at iteration $\tau .$ . Note that we replace $\varphi$ with $\tilde { \varphi }$ only in the objective and keep $\varphi ( \alpha )$ as is in the constraint: i.e., mi $\begin{array} { r } { \mathsf { \Lambda } _ { 1 \alpha } \sum _ { i } \beta _ { i } ( \alpha ) + \tilde { \varphi } _ { \tau } ( \alpha ) \mathrm { s . t . } - \varphi _ { i } ( \alpha ) \leq } \end{array}$ 0 and $\alpha _ { i } \ > \ 0$ for all i. This sequential optimization approach is a variation of the concave-convex procedure (CCP) (Yuille & Rangarajan, 2003; Lipp & Boyd, 2016). Therefore the sequence $\{ \alpha ^ { ( \tau ) } \}$ <sub>τ</sub> converges to a critical point of the original non-convex problem in Eq. 5 based on previous theory of CCP by Sriperumbudur & Lanckriet (2009). Moreover, since we do not modify the constraint, $\alpha ^ { ( \tau ) }$ always satisfies the constraint of the original problem for any $\tau .$ Finally, the following proposition shows that original objective monotonically decreases with τ:

Proposition 3.2. Denote the objective for the optimization problem in Eq. 5 by $\begin{array} { r } { \phi ( \alpha ) ~ = ~ \sum _ { i } \beta _ { i } ( \alpha ) + \varphi ( \alpha ) } \end{array}$ . Then, $\phi \left( \alpha ^ { ( \tau + 1 ) } \right) \leq \phi \left( \alpha ^ { ( \tau ) } \right) f o r a l l \tau \geq 1$

We provide proof and further discussion in Appendix A. In practice, we limit the sequence of CCP to 20 in all experiments, with the exception of Section 6.3 for which we use a single step. We found the improved solution to have a limited effect on the MTL performance (see Appendix D.2).

## 3.3. Practical Speedup

One shortcoming of many leading MTL methods is that all task gradients are required for obtaining the joint update direction. When the number of tasks K becomes large, this may be too computationally expensive as it requires one to perform K backward passes through the shared backbone to compute the K gradients. Prior work suggested using a subset of tasks (Liu et al., 2021a) or replacing the task gradients with the feature level gradient (Sener & Koltun, 2018; Liu et al., 2021b; Javaloy & Valera, 2021) as potential practical speedups. We emphasize that this issue is not unique to our method, but rather is shared to all methods that compute all gradients for all tasks.

![](images/58d8b72d7474c5b73bacf232f3cca60d629b7293a8c16733e390d7eaf8e5c509.jpg)

<details>
<summary>line</summary>

| Step | Nash-MTL (Ours) | IMTL-G | SI | CAGrad | PCGrad | RLW | UW | DWA | LS | MGDA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 | ~6000 |
| 50000 | ~250 | ~250 | ~250 | ~300 | ~400 | ~450 | ~350 | ~350 | ~350 | ~450 |
| 100000 | ~150 | ~150 | ~150 | ~200 | ~250 | ~300 | ~250 | ~250 | ~250 | ~300 |
| 150000 | ~120 | ~120 | ~120 | ~180 | ~220 | ~250 | ~220 | ~220 | ~220 | ~250 |
| 200000 | ~90 | ~110 | ~110 | ~170 | ~210 | ~240 | ~210 | ~210 | ~210 | ~240 |
| 250000 | ~85 | ~110 | ~110 | ~170 | ~210 | ~240 | ~210 | ~210 | ~210 | ~240 |
</details>

Figure 3. QM9. Test $\Delta _ { m }$ throughout the training process averaged over 3 random seeds.

In practice, we found that using feature-level gradients as a surrogate to the gradient of the shared parameters dramatically degrades the performance of our method. See Appendix C for empirical results and further discussion. As an alternative, we suggest updating the gradient weights $\alpha ^ { ( t ) }$ once every few iterations instead of every iteration. This simple yet effective solution greatly reduces the runtime (up to ∼ ×10 for QM9 and $\sim \times 5$ for MT10) while maintaining high performance. In Section 6.4 we provide experimental results while varying the frequency of task weights update on the QM9 dataset and the MT10 benchmark. Our results show that Nash-MTL runtime can be reduced to about the same as linear scalarization (or STL) while maintaining competitive results compared to other baselines; However, in some cases, we do see a noticeable drop in performance compared with our standard approach.

## 4. Related Work

In multitask learning (MTL), one simultaneously solves several learning problems while sharing information among tasks (Caruana, 1997; Ruder, 2017), commonly through a joint hidden representation (Zhang et al., 2014; Dai et al., 2016; Pinto & Gupta, 2017; Zhao et al., 2018; Liu et al., 2019b). Studies in the literature proposed several explanations for the difficulty in the optimization process of MTL, such as conflicting gradients (Wang et al., 2020; Yu et al., 2020a), or plateaus in the loss landscape (Schaul et al., 2019). Other studies aimed at improving multitask learning by proposing novel architectures (Misra et al., 2016; Hashimoto et al., 2017; Liu et al., 2019b; Chen et al., 2020). We focus on weighting the gradients of the tasks via an axiomatic approach that is agnostic to the architecture used. Studies in a similar vein proposed to weigh the task losses with various approaches, such as the uncertainty of the tasks (Kendall et al., 2018), the norm of the gradients (Chen et al., 2018), random weights (Lin et al., 2021), and similar ity of the gradients (Du et al., 2018; Suteu & Guo, 2019). These methods are mostly heuristic and can have unstable performance (Liu et al., 2021a). Recently, several studies proposed MTL approaches based on the multiple-gradient descent algorithm (MGDA) for multi-objective optimization (Desid ´ eri ´ , 2012). This is an appealing approach since, under mild conditions, convergence to a Pareto stationary point is guaranteed. Sener & Koltun (2018) cast the multi-objective problem to multi-task problem and suggest task weighting based on the Frank-Wolfe algorithm (Jaggi, 2013). Liu et al. (2021a) searches for an update direction in a neighborhood of the average gradient that maximizes the worst improvement of any task. Unlike these studies, we propose an MTL approach based on a Bargaining game that can find solutions that are Pareto optimal and proportionally fair.

Table 1. QM9. Test performance averaged over 3 random seeds.

<table><tr><td></td><td>MR ↓</td><td> $\Delta_{\text{m}} \%$  ↓</td></tr><tr><td>LS</td><td>6.8</td><td>177.6 ± 3.4</td></tr><tr><td>SI</td><td>4.0</td><td>77.8 ± 9.2</td></tr><tr><td>RLW</td><td>8.2</td><td>203.8 ± 3.4</td></tr><tr><td>DWA</td><td>6.4</td><td>175.3 ± 6.3</td></tr><tr><td>UW</td><td>5.3</td><td>108.0 ± 22.5</td></tr><tr><td>MGDA</td><td>5.9</td><td>120.5 ± 2.0</td></tr><tr><td>PCGrad</td><td>5.0</td><td>125.7 ± 10.3</td></tr><tr><td>CAGrad</td><td>5.7</td><td>112.8 ± 4.0</td></tr><tr><td>IMTL-G</td><td>4.7</td><td>77.2 ± 9.3</td></tr><tr><td>Nash-MTL</td><td>2.5</td><td>62.0 ± 1.4</td></tr></table>

The closest work to our approach, to the best of our knowledge, is Liu et al. (2021b). There, the authors propose to look for a fair gradient direction where all the cosine similarities are equal. We note that this update direction satisfies all of the Nash axioms except for Pareto optimally. Thus, unlike our proportionally fair approach, it can settle for a sub-optimal solution for the sake of fairness.

Finally, we note that the Nash bargaining solution was effectively applied to problems in various fields such as communication (Zhang et al., 2008; Leshem & Zehavi, 2011; Shi et al., 2018), economics (Dagan & Volij, 1993), and computing (Grosu et al., 2002), and to several learning setups, such as reinforcement learning (Qiao et al., 2006), Bayesian optimization (Binois et al., 2020), clustering (Rezaee et al., 2021), federated learning (Kim, 2021), and multi-armed bandits (Baek & Farias, 2021).

Table 2. NYUv2. Test performance for three tasks: semantic segmentation, depth estimation, and surface normal. Values are averages over 3 random seeds.

<table><tr><td rowspan="3"></td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3">MR↓</td><td rowspan="3">Δm%↓</td></tr><tr><td rowspan="2">mIoU↑</td><td rowspan="2">Pix Acc↑</td><td rowspan="2">Abs Err ↓</td><td rowspan="2">Rel Err ↓</td><td colspan="2">Angle Distance ↓</td><td colspan="3">Within  $t^{\circ}$  ↑</td></tr><tr><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>STL</td><td>38.30</td><td>63.76</td><td>0.6754</td><td>0.2780</td><td>25.01</td><td>19.21</td><td>30.14</td><td>57.20</td><td>69.15</td><td></td><td></td></tr><tr><td>LS</td><td>39.29</td><td>65.33</td><td>0.5493</td><td>0.2263</td><td>28.15</td><td>23.96</td><td>22.09</td><td>47.50</td><td>61.08</td><td>8.11</td><td>5.59</td></tr><tr><td>SI</td><td>38.45</td><td>64.27</td><td>0.5354</td><td>0.2201</td><td>27.60</td><td>23.37</td><td>22.53</td><td>48.57</td><td>62.32</td><td>7.11</td><td>4.39</td></tr><tr><td>RLW</td><td>37.17</td><td>63.77</td><td>0.5759</td><td>0.2410</td><td>28.27</td><td>24.18</td><td>22.26</td><td>47.05</td><td>60.62</td><td>10.11</td><td>7.78</td></tr><tr><td>DWA</td><td>39.11</td><td>65.31</td><td>0.5510</td><td>0.2285</td><td>27.61</td><td>23.18</td><td>24.17</td><td>50.18</td><td>62.39</td><td>6.88</td><td>3.57</td></tr><tr><td>UW</td><td>36.87</td><td>63.17</td><td>0.5446</td><td>0.2260</td><td>27.04</td><td>22.61</td><td>23.54</td><td>49.05</td><td>63.65</td><td>6.44</td><td>4.05</td></tr><tr><td>MGDA</td><td>30.47</td><td>59.90</td><td>0.6070</td><td>0.2555</td><td>24.88</td><td>19.45</td><td>29.18</td><td>56.88</td><td>69.36</td><td>5.44</td><td>1.38</td></tr><tr><td>PCGrad</td><td>38.06</td><td>64.64</td><td>0.5550</td><td>0.2325</td><td>27.41</td><td>22.80</td><td>23.86</td><td>49.83</td><td>63.14</td><td>6.88</td><td>3.97</td></tr><tr><td>GradDrop</td><td>39.39</td><td>65.12</td><td>0.5455</td><td>0.2279</td><td>27.48</td><td>22.96</td><td>23.38</td><td>49.44</td><td>62.87</td><td>6.44</td><td>3.58</td></tr><tr><td>CAGrad</td><td>39.79</td><td>65.49</td><td>0.5486</td><td>0.2250</td><td>26.31</td><td>21.58</td><td>25.61</td><td>52.36</td><td>65.58</td><td>3.77</td><td>0.20</td></tr><tr><td>IMTL-G</td><td>39.35</td><td>65.60</td><td>0.5426</td><td>0.2256</td><td>26.02</td><td>21.19</td><td>26.2</td><td>53.13</td><td>66.24</td><td>3.11</td><td>-0.76</td></tr><tr><td>Nash-MTL</td><td>40.13</td><td>65.93</td><td>0.5261</td><td>0.2171</td><td>25.26</td><td>20.08</td><td>28.4</td><td>55.47</td><td>68.15</td><td>1.55</td><td>-4.04</td></tr></table>

## 5. Analysis

We now analyze the convergence of our method in the convex and non-convex cases. As even single-task non-convex optimization might only converge to a stationary point, we will prove convergence to a Pareto stationary point, i.e., a point where some convex combination of the gradients is zero. As stated, we also assume that the gradients are independent while not at a Pareto stationary point. Independence of the gradients is a slightly stronger assumption than Pareto stationarity but is needed to exclude degenerate edge cases such as two identical tasks.

We note that by substituting local Pareto optimality for Pareto stationarity in Assumption 5.1 we can show convergence to a local Pareto optimal point. However, this assumption has strong implications, as it implies we avoid local maxima and saddle points of any specific task. Since our update rule is a descent direction for all tasks, we can reasonably assume that our algorithm avoids local maxima points. Furthermore, it was shown that first-order methods avoid saddle points (Panageas et al., 2019), giving credence to this stronger assumption. Nevertheless, we take a conservative approach and state our results with the weaker assumption.

We formally make the following assumptions:

Assumption 5.1. We assume that for a sequence $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ generated by our algorithm, the set of the gradient vectors $g _ { 1 } ^ { ( t ) } , . . . , g _ { K } ^ { ( t ) }$ at any point on the sequence and at any partial limit are linearly independent unless that point is a Pareto stationary point.

Assumption 5.2. We assume that all loss functions are differentiable, bounded below and that all sub-level sets are bounded. The input domain is open and convex.

Assumption 5.3. We assume that all the loss functions are L-smooth,

$$
\left| \left| \nabla \ell_ {i} (x) - \nabla \ell_ {i} (y) \right| \right| \leq L \| x - y \| \quad . \tag {6}
$$

Theorem 5.4. Let $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ be the sequence generated by the update rule $\hat { \theta } ^ { ( t + \bar { 1 } ) } \stackrel { - } { = } \theta ^ { ( t ) } - \bar { \mu } ^ { ( t ) } \Delta \bar { \theta ^ { ( t ) } }$ where $\begin{array} { r } { \dot { \Delta { \theta } } ^ { ( t ) } = \dot { \sum } _ { i = 1 } ^ { K } \alpha _ { i } ^ { ( t ) } g _ { i } ^ { ( t ) } } \end{array}$ is the Nash bargaining solution $( G ^ { ( t ) } ) ^ { \top } G ^ { ( \overline { { { t ) } } } } \alpha ^ { ( \overline { { { t } } } ) } = 1 / \overline { { { \alpha } } } ^ { ( t ) }$ . Set $\mu ^ { ( t ) } = \operatorname* { m i n } _ { i \in [ K ] } \frac { \bar { 1 } } { L K \alpha _ { i } ^ { ( t ) } }$ . Then, the sequence $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ has a subsequence that converges to a Pareto stationary point $\theta ^ { * }$ . Moreover all the loss functions $( \ell _ { 1 } ( \theta ^ { ( t ) } ) , . . . , \ell _ { K } ( \dot { \theta } ^ { ( t ) } ) )$ converge to $( \ell _ { 1 } ( \theta ^ { * } ) , . . . , \ell _ { K } ( \theta ^ { * } ) )$ .

Proofsketch. We can show that $\begin{array} { r } { \mu ^ { ( t ) } = \operatorname* { m i n } _ { i } { \frac { 1 } { \alpha _ { i } ^ { ( t ) } } } \to 0 } \end{array}$ so $\vert \vert \alpha ^ { ( t ) } \vert \vert  \infty$ . We also show that $| | 1 / \alpha ^ { ( t ) } | |$ is bounded. $\begin{array} { r } { \ddot { \mathrm { A s } } \ ( \ddot { G } ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } \alpha ^ { ( t ) } = 1 / \alpha ^ { ( t ) } } \end{array}$ this means that the smallest singluar value of $( G ^ { ( \dot { t } ) } ) ^ { \top } G ^ { ( t ) }$ must converge to zero. From compactness $\{ \theta ^ { ( { \dot { t } } ) } \} _ { t = 1 } ^ { \infty }$ has a converging subsequence whose limit we denote as $\theta ^ { * }$ . From continuity we get that the gradients Gram matrix $G ^ { \top } G$ computed at $\theta ^ { * }$ must have a zero singular value and therefore the gradients are linearly dependent. From our assumption this means that $\theta ^ { * }$ is Pareto stationary. As the losses are monotonically decearsing and bounded below they must converge and to the subsequence limit of $( \ell _ { 1 } ( \theta ^ { * } ) , . . . , \ell _ { K } ( \theta ^ { * } ) )$ ). □

If we also assume convexity, we can strengthen our claim

Theorem 5.5. Let $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty } .$ be the sequence generated by the update rule $\hat { \theta ^ { ( t + 1 ) } } \ \bar { \ } = \ \theta ^ { ( t ) } \ - \ \bar { \mu } ^ { ( t ) } \Delta \bar { \theta ^ { ( t ) } }$ where $\begin{array} { r } { \dot { \Delta { \theta } } ^ { ( t ) } = \dot { \sum } _ { i = 1 } ^ { K } \alpha _ { i } ^ { ( t ) } g _ { i } ^ { ( t ) } } \end{array}$ is the Nash bargaining solution $( G ^ { ( t ) } ) ^ { \top } G ^ { ( \overline { { { t } } } ) } \alpha ^ { ( \overline { { { t } } } ) } = 1 / \alpha ^ { ( t ) }$ . Set $\mu ^ { ( t ) } = \operatorname* { m i n } _ { i \in [ K ] } \frac { 1 } { L K \alpha _ { i } ^ { ( t ) } }$ . If we assume that all the loss functions are convex, then the sequence $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ converges to a Pareto optimal point $\theta ^ { * }$ .

Table 3. CityScapes. Test performance for two tasks: semantic segmentation and depth estimation. Value are averages over 3 random seeds.

<table><tr><td rowspan="2"></td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="2">MR↓</td><td rowspan="2"> $\Delta_{m}\% \downarrow$ </td></tr><tr><td>mIoU ↑</td><td>Pix Acc ↑</td><td>Abs Err ↓</td><td>Rel Err↓</td></tr><tr><td>STL</td><td>74.01</td><td>93.16</td><td>0.0125</td><td>27.77</td><td></td><td></td></tr><tr><td>LS</td><td>75.18</td><td>93.49</td><td>0.0155</td><td>46.77</td><td>6.12</td><td>22.60</td></tr><tr><td>SI</td><td>70.95</td><td>91.73</td><td>0.0161</td><td>33.83</td><td>8.00</td><td>14.11</td></tr><tr><td>RLW</td><td>74.57</td><td>93.41</td><td>0.0158</td><td>47.79</td><td>9.25</td><td>24.38</td></tr><tr><td>DWA</td><td>75.24</td><td>93.52</td><td>0.0160</td><td>44.37</td><td>6.00</td><td>21.45</td></tr><tr><td>UW</td><td>72.02</td><td>92.85</td><td>0.0140</td><td>30.13</td><td>5.25</td><td>5.89</td></tr><tr><td>MGDA</td><td>68.84</td><td>91.54</td><td>0.0309</td><td>33.50</td><td>8.75</td><td>44.14</td></tr><tr><td>PCGrad</td><td>75.13</td><td>93.48</td><td>0.0154</td><td>42.07</td><td>6.37</td><td>18.29</td></tr><tr><td>GradDrop</td><td>75.27</td><td>93.53</td><td>0.0157</td><td>47.54</td><td>5.50</td><td>23.73</td></tr><tr><td>CAGrad</td><td>75.16</td><td>93.48</td><td>0.0141</td><td>37.60</td><td>5.37</td><td>11.64</td></tr><tr><td>IMTL-G</td><td>75.33</td><td>93.49</td><td>0.0135</td><td>38.41</td><td>3.62</td><td>11.10</td></tr><tr><td>Nash-MTL</td><td>75.41</td><td>93.66</td><td>0.0129</td><td>35.02</td><td>1.75</td><td>6.82</td></tr></table>

See the full proofs in the appendix Sec. A.

## 6. Experiments

We evaluate Nash-MTL on diverse multi-task learning problems. The experiments show the superiority of Nash-MTL over previous MTL methods. To support future research and the reproducibility of the results, we will make our source code publicly available. Additional experimental results and details are provided in Appendix B.

Compared methods: We compare the following approaches: (1) Our proposed Nash-MTL algorithm described in Section 3; (2) Single task learning (STL), training an independent model for each task; (3) Linear scalarization (LS) baseline which minimizes $\textstyle \sum _ { k } \ell _ { k } ; ( 4 )$ Scale-invariant (SI) baseline which minimizes $\textstyle \sum _ { k } \log \ell _ { k } .$ This baseline is invariant to rescaling each loss with a positive number; (5) Dynamic Weight Average (DWA) (Liu et al., 2019b) adjusts task weights based on the rates of loss changes over time; (6) Uncertainty weighting (UW) (Kendall et al., 2018) uses task uncertainty quantification to adjust task weights; (7) MGDA (Sener & Koltun, 2018) finds a convex combination of gradients with a minimal norm; (8) Random loss weighting (RLW) with normal distribution, scales the losses according to randomly sampled task weights (Lin et al., 2021); (9) PCGrad (Yu et al., 2020a) removes conflicting components of each gradient w.r.t the other gradients; (10) GradDrop (Chen et al., 2020) randomly drops components of the task gradients based on how much they conflict; (11) CAGrad (Liu et al., 2021a) optimizes for the average loss while explicitly controlling the minimum decrease rate across tasks; (12) IMTL-G (Liu et al., 2021b) uses an update direction with equal projections on task gradients. IMTL-G is applied to the feature-level gradients, as was suggested by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but its performance was even worse.

Evaluation. For each experiment, we report the common evaluation metrics for each task. Since naturally MTL does not carry a single objective and since the scale of per-task metrics often varies significantly, we report two metrics that capture the overall performance: (1) $\Delta _ { \mathbf { m } } \% .$ the average per-task performance drop of method m relative to the STL baseline denoted b. Formally, $\Delta _ { m } \% =$ $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( - 1 ) ^ { \delta _ { k } } ( M _ { m , k } - M _ { b , k } ) / M _ { b , k } } \end{array}$ , where $M _ { b , k }$ is the value of metric $M _ { k }$ obtained by the baseline and $M _ { m , k }$ by the compared method. $\delta _ { k } = 1$ if a higher value is better for a metric $M _ { k }$ and 0 otherwise (Maninis et al., 2019; Liu et al., 2021a). (2) Mean Rank (MR): The average rank of each method across the different tasks (lower is better). A method receives the best value, MR = 1, if it ranks first in all tasks.

## 6.1. Multi-Task Regression for QM9

We evaluate Nash-MTL on predicting 11 properties of molecules from the QM9 dataset (Ramakrishnan et al., 2014), a widely used benchmark for graph neural networks. QM9 consists of ∼ 130K molecules represented as graphs annotated with both node and edge features. We used the QM9 example in PyTorch Geometric (Fey & Lenssen, 2019), and use 110K molecules for training, 10K for validation, and 10K as a test set. As each task target range is at a different scale, this could be an issue for other methods that are not scale-invariant like ours. For fairness, we normalized each task target to have zero mean and unit standard deviation. We use the popular GNN model from

Gilmer et al. (2017), a network comprised of several concatenated message passing layers, which update the node features based on both node and edge features, followed by the pooling operator from Vinyals et al. (2015). Specifically, we used the implementation from Fey & Lenssen (2019). We train each method for 300 epochs and search for the best learning-rate (lr) given by the $\Delta _ { m }$ performance on the validation set. We use a learning-rate scheduler to reduce the lr once the validation $\Delta _ { m }$ metric has stopped improving. The validation set is also used for early stopping.

Predicting molecular properties in QM9 poses a significant challenge for MTL methods because the number of tasks is large and because the loss scales vary significantly. The scale issue is only partially resolved by normalization because some tasks are easier to learn than others. Prior work found that single-task learning significantly improves performance on all targets compared to MTL methods (Maron et al., 2019; Klicpera et al., 2020).

Results are shown in Figure 3 and Table 1. Nash-MTL achieves the best performance in terms of both MR and $\Delta _ { m }$ Interestingly, most MTL methods fall short compared to the simple scale-invariant baseline, which ignores gradient interaction, except for IMTL-G whose performance is on par with this baseline. This result shows that the scaleinvariant property of our approach can be beneficial. See Appendix D.1 for the per-task evaluation results.

## 6.2. Scene Understanding

We follow the protocol of (Liu et al., 2019b) and evaluate Nash-MTL on the NYUv2 and Cityscapes datasets (Silberman et al., 2012; Cordts et al., 2016). NYUv2 is an indoor scene dataset that consists of 1449 RGBD images and dense per-pixel labeling with 13 classes. We use this dataset as a multitask learning benchmark for semantic segmentation, depth estimation, and surface normal prediction.

The CityScapes dataset (Cordts et al., 2016) contains 5000 high-resolution street-view images with dense per-pixel annotations. We use this dataset as a multitask learning benchmark for semantic segmentation and depth estimation. To speed up the training phase, all images were resized to $1 2 8 \times 2 5 6$ . The original dataset contains 19 categories for pixel-wise semantic segmentation, together with groundtruth depth maps. For segmentation, we used a coarser version of the labels with 7 classes.

For all MTL methods, we train a Multi-Task Attention Network (MTAN) (Liu et al., 2019b) which adds an attention mechanism on top of the SegNet architecture (Badrinarayanan et al., 2017). We follow the training procedure from Liu et al. (2019b); Yu et al. (2020a); Liu et al. (2021a). Each method is trained for 200 epochs with the Adam optimizer (Kingma & Ba, 2015) and an initial learning-rate of 1e − 4. The learning-rate is halved to 5e − 5 after 100 epochs. As in (Liu et al., 2021a) The STL baseline refers to training task-specific SegNet models.

Table 4. MT10. Average success over 10 random seeds.

<table><tr><td></td><td>Success ± SEM</td></tr><tr><td>STL SAC</td><td>0.90 ± 0.032</td></tr><tr><td>MTL SAC</td><td>0.49 ± 0.073</td></tr><tr><td>MTL SAC + TE</td><td>0.54 ± 0.047</td></tr><tr><td>MH SAC</td><td>0.61 ± 0.036</td></tr><tr><td>SM</td><td>0.73 ± 0.043</td></tr><tr><td>CARE</td><td>0.84 ± 0.051</td></tr><tr><td>PCGrad</td><td>0.72 ± 0.022</td></tr><tr><td>CAGrad</td><td>0.83 ± 0.045</td></tr><tr><td>Nash-MTL</td><td>0.91 ± 0.031</td></tr></table>

The results are presented in Table 2 and Table 3. Our method, Nash-MTL, achieves the best MR in both datasets, the best $\Delta _ { m }$ in NYUv2 and the seconds to best $\Delta _ { m }$ in the CityScapes experiment. Nash-MTL performance is well balanced across tasks. MGDA is primarily focused on the task of predicting surface normals and achieves poor performance on the other two tasks. The inherent biasedness of MGDA towards the task with the smallest gradient magnitude was previously discussed in Liu et al. (2021b). We note that the optimal solution under Nash-MTL for the two tasks case is equivalent to independently normalizing each gradient and summing with equal weights. While this is a fairly simple approach for MTL, we show that it outperforms almost all the compared MTL methods on the two-tasks CityScapes benchmark.

## 6.3. Multi-Task Reinforcement Learning

We consider a multi-task RL problem and evaluate Nash-MTL on the MT10 environment from the Meta-World benchmark (Yu et al., 2020b). This benchmark involves a simulated robot trained to perform actions like pressing a button and opening a window, each action treated as a task, for a total of 10 tasks. The goal is to learn a policy that can succeed across all the diverse sets of manipulation tasks. Following previous works on MTL-RL (Yu et al., 2020a; Liu et al., 2021a; Sodhani et al., 2021), we use Soft Actor-Critic (SAC) (Haarnoja et al., 2018) as the base RL algorithm. Along with the MTL methods (1) CAGrad (Liu et al., 2021a) and (2) PCGrad (Yu et al., 2020a) applied to a shared model SAC, we evaluate the following methods: (3) STL, one SAC model per task; (4) MTL SAC with a shared model; (5) Multi-task SAC with task encoder (MTL SAC + TE, Yu et al. (2020b)); (6) Multi-headed SAC (MH SAC) with task-specific heads (Yu et al., 2020b); (7) Soft Modularization (SM, Yang et al. (2020)) which estimates per-task routes for different tasks in a shared model, and;

Table 5. Training runtime per episode and average success for the MT10 benchmark, computed over 10 random seeds while varying the frequency of task weights updates in Nash-MTL.

<table><tr><td></td><td>Success ± SEM</td><td>Runtime[Sec.]</td></tr><tr><td>MTL-SAC</td><td>0.49 ± 0.073</td><td>7.3</td></tr><tr><td>PCGrad</td><td>0.72 ± 0.022</td><td>9.7</td></tr><tr><td>CAGrad</td><td>0.83 ± 0.045</td><td>20.9</td></tr><tr><td>Nash-MTL</td><td>0.91 ± 0.031</td><td>40.7</td></tr><tr><td>Nash-MTL-50</td><td>0.85 ± 0.022</td><td>8.6</td></tr><tr><td>Nash-MTL-100</td><td>0.87 ± 0.033</td><td>7.9</td></tr></table>

(8) CARE (Sodhani et al., 2021) which utilizes language metadata and employs a mixture of encoders. We follow the same experiment setup from Sodhani et al. (2021); Liu et al. (2021a) to train all methods over 2 million steps and report the mean success over 10 random seeds with fixed evaluation frequency. The results are presented in Table 4.

Nash-MTL achieves the best performance by a large margin. In addition, Nash-MTL is the only MTL method to reach the same performance as the per-task SAC STL baseline.

## 6.4. Scaling-up Nash-MTL

One of the major drawbacks of the SOTA MTL methods is that they require access to all task gradients to compute the optimal update direction (Sener & Koltun, 2018; Yu et al., 2020a; Liu et al., 2021b;a). This requires one to perform K backward passes at each optimization step, thus scales poorly with the number of tasks. Previous works suggested using a subset of tasks (Liu et al., 2021a) or replacing the task gradients with the feature-level gradient (Sener & Koltun, 2018; Liu et al., 2021b; Javaloy & Valera, 2021) as potential speedups. In our experiments, we found that using the feature-level gradients can greatly reduce Nash-MTL performance (Appendix C). However, here we show that the simple solution of updating task weights less frequently maintains good performance while dramatically reducing the training time.

One approach to alleviate this issue is to update the task weights less frequently, and use these weights in subsequent steps. We evaluate this approach using the QM9 dataset and the MT10 benchmark and present the result in Figure 4 and Table 5. We denote Nash-MTL with task weight update every T optimization steps with Nash-MTL-T.

The results show that Nash-MTL is fairly robust to varying intervals between weights updates. While this simple approach results in a small degradation in performance, it can dramatically decrease the training time of our method. For example, on the QM9, updating the weights every 5/50 steps results in a ×3.7/9.8 speedup w.r.t updating the weights at each step. On the MT10 environment, updating the weights every 100 steps result in $\sim \times 1 0$ speedup (only ∼ ×1.1 slower than the fastest baseline) while outperforming all other MTL baseline method (Table 5).

![](images/6c6956701a13d7c440f9165a443b285e7bc77883a8c725f183abde063bc9f11c.jpg)

<details>
<summary>line</summary>

| Weight update freq. | \(\Delta m\) |
| --- | --- |
| 12 | ~62 |
| 12 | ~74 |
| 5 | ~77.5 |
| 10 | ~79 |
| 25 | ~82 |
| 50 | ~85 |
| 75 | ~89 |
</details>

Figure 4. Test $\Delta _ { m }$ for the QM9 dataset, averaged over 3 random seeds, for different intervals of task weights update.

## 7. Conclusion

In this work, we present Nash-MTL, a novel and principled approach for multitask learning. We frame the gradient combination step in MTL as a bargaining game and use the Nash bargaining solution to find the optimal update direction. We highlight the importance of the scale invariance approach for multitask learning, specifically for setups with varying loss scales and gradient magnitudes. We provide a theoretical convergence analysis for Nash-MTL, showing that it converges to a Pareto optimal and Pareto stationary points in the convex and non-convex settings, respectively. Finally, our experiments show that Nash-MTL achieves state-of-the-art results on various benchmarks across multiple domains.

## 8. Acknowledgements

This work was funded by the Israeli innovation authority through the AVATAR consortium; by the Israel Science Foundation (ISF grant 737/2018); and by an equipment grant to GC and Bar Ilan University (ISF grant 2332/18).

## References

Achituve, I., Maron, H., and Chechik, G. Self-supervised learning for domain adaptation on point clouds. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 123–133, 2021.  
Badrinarayanan, V., Kendall, A., and Cipolla, R. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
Baek, J. and Farias, V. F. Fair exploration via axiomatic bargaining. arXiv preprint arXiv:2106.02553, 2021.  
Baxter, J. A model of inductive bias learning. J. Artif. Intell. Res., 2000.  
Binois, M., Picheny, V., Taillandier, P., and Habbal, A. The Kalai-Smorodinsky solution for many-objective Bayesian optimization. J. Mach. Learn. Res., 21(150):1–42, 2020.  
Caruana, R. Multitask learning. Machine learning, 28(1): 41–75, 1997.  
Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pp. 794–803. PMLR, 2018.  
Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., and Anguelov, D. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. ArXiv, abs/2010.06808, 2020.  
Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., and Schiele, B. The cityscapes dataset for semantic urban scene understanding. In Proc. ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
Crawshaw, M. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796, 2020.  
Dagan, N. and Volij, O. The bankruptcy problem: a cooperative bargaining approach. Mathematical Social Sciences, 26(3):287–297, 1993.  
Dai, J., He, K., and Sun, J. Instance-aware semantic segmentation via multi-task network cascades. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 3150–3158, 2016.  
Desid´ eri, J.-A. Multiple-gradient descent algorithm´ (MGDA) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012.  
Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., and Lakshminarayanan, B. Adapting auxiliary losses using gradient similarity. arXiv preprint arXiv:1812.02224, 2018.  
Fey, M. and Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In International conference on machine learning, pp. 1263–1272. PMLR, 2017.  
Grosu, D., Chronopoulos, A. T., and Leung, M.-Y. Load balancing in distributed systems: An approach using cooperative games. In Proceedings 16th International Parallel and Distributed Processing Symposium, pp. 10–pp. IEEE, 2002.  
Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actorcritic: Off-policy maximum entropy deep reinforcement, 2018.  
Hashimoto, K., Xiong, C., Tsuruoka, Y., and Socher, R. A joint many-task model: Growing a neural network for multiple nlp tasks. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 1923–1933, 2017.  
Jaggi, M. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In International Conference on Machine Learning, pp. 427–435. PMLR, 2013.  
Javaloy, A. and Valera, I. Rotograd: Dynamic gradient homogenization for multi-task learning. arXiv preprint arXiv:2103.02631, 2021.  
Kendall, A., Gal, Y., and Cipolla, R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7482–7491, 2018.  
Kim, S. Cooperative federated learning-based task offloading scheme for tactical edge networks. IEEE Access, 9: 145739–145747, 2021.  
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2015.  
Klicpera, J., Groß, J., and Gunnemann, S. Direc-¨ tional message passing for molecular graphs. ArXiv, abs/2003.03123, 2020.  
Leshem, A. and Zehavi, E. Smart carrier sensing for distributed computation of the generalized nash bargaining solution. In 2011 17th International Conference on Digital Signal Processing (DSP), pp. 1–5. IEEE, 2011.  
Lin, B., Ye, F., and Zhang, Y. A closer look at loss weighting in multi-task learning. arXiv preprint arXiv:2111.10603, 2021.  
Lipp, T. and Boyd, S. Variations and extension of the convex–concave procedure. Optimization and Engineering, 17(2):263–287, 2016.  
Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. Conflictaverse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34, 2021a.  
Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., and Zhang, W. Towards impartial multitask learning. In International Conference on Learning Representations, 2021b.  
Liu, S., Davison, A., and Johns, E. Self-supervised generalisation with meta auxiliary learning. Advances in Neural Information Processing Systems, 32, 2019a.  
Liu, S., Johns, E., and Davison, A. J. End-to-end multi-task learning with attention. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1871–1880, 2019b.  
Maninis, K.-K., Radosavovic, I., and Kokkinos, I. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1851–1860, 2019.  
Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y. Provably powerful graph networks. arXiv preprint arXiv:1905.11136, 2019.  
Misra, I., Shrivastava, A., Gupta, A., and Hebert, M. Crossstitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3994–4003, 2016.  
Nash, J. Two-person cooperative games. Econometrica, 21 (1):128–140, 1953. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/1906951.  
Navon, A., Achituve, I., Maron, H., Chechik, G., and Fetaya, E. Auxiliary learning by implicit differentiation. In International Conference on Learning Representations (ICLR), 2021a.  
Navon, A., Shamsian, A., Chechik, G., and Fetaya, E. Learning the pareto front with hypernetworks. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum? id=NjF772F4ZZR.  
Panageas, I., Piliouras, G., and Wang, X. First-order methods almost always avoid saddle points: The case of vanishing step-sizes. In Neural Information Processing Systems (NeurIPS), 2019.  
Pinto, L. and Gupta, A. Learning to push by grasping: Using multiple tasks for effective learning. In 2017 IEEE international conference on robotics and automation (ICRA), pp. 2161–2168. IEEE, 2017.  
Qiao, H., Rozenblit, J., Szidarovszky, F., and Yang, L. Multiagent learning model with bargaining. In Proceedings ofthe 2006 winter simulation conference, pp. 934–940. IEEE, 2006.  
Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014.  
Rezaee, M. J., Eshkevari, M., Saberi, M., and Hussain, O. GBK-means clustering algorithm: An improvement to the K-means algorithm based on the bargaining game. Knowledge-Based Systems, 213:106672, 2021.  
Ruder, S. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.  
Schaul, T., Borsa, D., Modayil, J., and Pascanu, R. Ray interference: a source of plateaus in deep reinforcement learning. arXiv preprint arXiv:1904.11455, 2019.  
Sener, O. and Koltun, V. Multi-task learning as multiobjective optimization. In Advances in Neural Information Processing Systems, pp. 527–538, 2018.  
Shi, C., Wang, F., Salous, S., Zhou, J., and Hu, Z. Nash bargaining game-theoretic framework for power control in distributed multiple-radar architecture underlying wireless communication system. Entropy, 20(4):267, 2018.  
Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pp. 746–760. Springer, 2012.  
Sodhani, S., Zhang, A., and Pineau, J. Multi-task reinforcement learning with context-based representations. arXiv preprint arXiv:2102.06177, 2021.  
Sriperumbudur, B. K. and Lanckriet, G. R. On the convergence of the concave-convex procedure. In Nips, volume 9, pp. 1759–1767. Citeseer, 2009.  
Standley, T., Zamir, A. R., Chen, D., Guibas, L. J., Malik, J., and Savarese, S. Which tasks should be learned together in multi-task learning? In International Conference on Machine Learning ICML, 2020.  
Suteu, M. and Guo, Y. Regularizing deep multi-task networks using orthogonal gradients. arXiv preprint arXiv:1912.06844, 2019.  
Szep, J. and Forg´ o, F.´ Introduction to the Theory ofGames. Springer, 1985.  
Thomson, W. Chapter 35 cooperative models of bargaining. volume 2 of Handbook ofGame Theory with Economic Applications, pp. 1237–1284. Elsevier, 1994.  
Vinyals, O., Bengio, S., and Kudlur, M. Order matters: Sequence to sequence for sets. arXiv preprint arXiv:1511.06391, 2015.  
Wang, Z., Tsvetkov, Y., Firat, O., and Cao, Y. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In International Conference on Learning Representations, 2020.  
Yang, R., Xu, H., Wu, Y., and Wang, X. Multi-task reinforcement learning with soft modularization. ArXiv, abs/2003.13661, 2020.  
Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, 2020a.  
Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., and Levine, S. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pp. 1094–1100. PMLR, 2020b.  
Yuille, A. L. and Rangarajan, A. The concave-convex procedure. Neural computation, 15(4):915–936, 2003.  
Zhang, Z., Shi, J., Chen, H.-H., Guizani, M., and Qiu, P. A cooperation strategy based on nash bargaining solution in cooperative relay networks. IEEE Transactions on Vehicular Technology, 57(4):2570–2577, 2008.  
Zhang, Z., Luo, P., Loy, C. C., and Tang, X. Facial landmark detection by deep multi-task learning. In European conference on computer vision, pp. 94–108. Springer, 2014.  
Zhao, X., Li, H., Shen, X., Liang, X., and Wu, Y. A modulation module for multi-task learning with applications in image retrieval. In Proceedings ofthe European Conference on Computer Vision (ECCV), pp. 401–416, 2018.

## A. Proofs

Lemma A.1. If L is differential and L-smooth (assumption 5.3) then $\begin{array} { r } { \mathcal L ( \theta ^ { \prime } ) \le \mathcal L ( \theta ) + \nabla \mathcal L ( \theta ) ^ { \top } ( \theta ^ { \prime } - \theta ) + \frac { L } { 2 } \| \theta ^ { \prime } - \theta \| ^ { 2 } . } \end{array}$

Proof. Fix $\theta , \theta ^ { \prime } \in \mathrm { d o m } ( \mathcal { L } ) \subseteq \mathbb { R } ^ { d }$ . Since $\mathrm { d o m } ( \mathcal { L } )$ is a convex and open set, there exists $\epsilon > 0$ such that $\theta + t ( \theta ^ { \prime } - \theta ) \in$ dom(L) for all $t \in [ - \epsilon , 1 + \epsilon ]$ . Set $\epsilon > 0$ to be such a number. Thus, we can define a function $\bar { \mathcal { L } } : [ - \epsilon , 1 + \epsilon ] $ R by $\bar { \mathcal { L } } ( t ) = \mathcal { L } ( \theta + t ( \theta ^ { \prime } - \theta ) )$ ). With this, $\bar { \mathcal { L } } ( 1 ) = \mathcal { L } ( \theta ^ { \prime } ) , \bar { \mathcal { L } } ( 0 ) = \mathcal { L } ( \theta )$ , and $\nabla \bar { \mathcal { L } } ( t ) = \nabla \mathcal { L } ( \theta + t ( \theta ^ { \prime } - \bar { \theta } ) ) ^ { \top } ( \theta ^ { \prime } - \bar { \theta } )$ for $t \in [ 0 , 1 ] \subset ( - \epsilon , 1 + \epsilon )$ . From Assumption 5.3, $\lvert \nabla \mathcal { L } ( \theta ^ { \prime } ) - \nabla \mathcal { L } ( \theta ) \rvert \rvert \leq L \lvert \lvert \theta ^ { \prime } - \theta \rvert \rvert$ , therefore

$$
\begin{array}{l} \| \nabla \bar {\mathcal {L}} (t ^ {\prime}) - \nabla \bar {\mathcal {L}} (t) \| = \| [ \nabla \mathcal {L} (\theta + t ^ {\prime} (\theta^ {\prime} - \theta)) - \nabla \mathcal {L} (\theta + t (\theta^ {\prime} - \theta)) ^ {\top} (\theta^ {\prime} - \theta) \| \\ \leq \| \theta^ {\prime} - \theta \| \| \nabla \mathcal {L} (\theta + t ^ {\prime} (\theta^ {\prime} - \theta)) - \nabla \mathcal {L} (\theta + t (\theta^ {\prime} - \theta)) \| \\ \leq L \| \theta^ {\prime} - \theta \| \| (t ^ {\prime} - t) (\theta^ {\prime} - \theta) \| \\ \leq L \| \theta^ {\prime} - \theta \| ^ {2} \| t ^ {\prime} - t \|. \\ \end{array}
$$

Hence, $\nabla \bar { \mathcal { L } } : [ 0 , 1 ] \to \mathbb { R }$ is Lipschitz continuous, and therefore continuous. By using the fundamental theorem of calculus with the continuous function $\nabla \bar { \mathcal { L } } : [ 0 , 1 ] \to \mathbb { R }$

$$
\begin{array}{l} \mathcal {L} (\theta^ {\prime}) = \mathcal {L} (\theta) + \int_ {0} ^ {1} \nabla \mathcal {L} (\theta + t (\theta^ {\prime} - \theta)) ^ {\top} (\theta^ {\prime} - \theta) d t \\ = \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {\top} (\theta^ {\prime} - \theta) + \int_ {0} ^ {1} \left(\nabla \mathcal {L} (\theta + t (\theta^ {\prime} - \theta)) - \nabla \mathcal {L} (\theta)\right) ^ {\top} (\theta^ {\prime} - \theta) d t \\ \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {\top} (\theta^ {\prime} - \theta) + \int_ {0} ^ {1} \| \nabla \mathcal {L} (\theta + t (\theta^ {\prime} - \theta)) - \nabla \mathcal {L} (\theta) \| \| \theta^ {\prime} - \theta \| d t \\ \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {\top} (\theta^ {\prime} - \theta) + \int_ {0} ^ {1} t L \| \theta^ {\prime} - \theta \| ^ {2} d t \\ = \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {\top} (\theta^ {\prime} - \theta) + \frac {L}{2} \| \theta^ {\prime} - \theta \| ^ {2}. \tag {7} \\ \end{array}
$$

![](images/a609e8d86877d3baae125f81793e628846fff94255b9adb653f5179e35c9be9b.jpg)

Theorem (5.4). Let $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ be the sequence generated by the update rule $\theta ^ { ( t + 1 ) } = \theta ^ { ( t ) } - \mu ^ { ( t ) } \Delta \theta ^ { ( t ) }$ where $\Delta \theta ^ { ( t ) } =$ $\textstyle \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { ( t ) } g _ { i } ^ { ( t ) }$ is the Nash bargaining solution $( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } \alpha ^ { ( t ) } = 1 / \alpha ^ { ( t ) }$ . Set $\begin{array} { r } { \mu ^ { ( t ) } = \underset { i \in [ K ] } { \operatorname* { m i n } } \frac { 1 } { L K \alpha _ { i } ^ { ( t ) } } . } \end{array}$ . The sequence $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ has a subsequence that converges to a Pareto stationary point $\theta ^ { * }$ . Moreover all the loss functions $( \ell _ { 1 } ( \theta ^ { ( t ) } ) , . . . , \ell _ { K } ( \theta ^ { ( t ) } ) )$ converge to $( \ell _ { 1 } ( \theta ^ { * } ) , . . . , \ell _ { K } ( \theta ^ { * } ) )$ .

Proof. We first note that if for some step we reach a Pareto stationary solution the algorithm halts and sequence stays fixed at that point and therefore converges; Next, we assume that we never get to an exact Pareto stationary solution at any finite step.

We note that the norm of $\Delta \theta ^ { ( t ) }$ is $\begin{array} { r } { \sqrt { K } \mathrm { ~ a s ~ } | | \Delta \theta ^ { ( t ) } | | ^ { 2 } = \sum _ { i = 1 } ^ { K } \alpha _ { i } g _ { i } ^ { \top } \Delta \theta ^ { ( t ) } = \sum _ { i = 1 } ^ { K } \alpha _ { i } \cdot 1 / \alpha _ { i } = K } \end{array}$ . For each loss $\ell _ { i }$ we have using Lemma A.1

$$
\ell_ {i} (\theta^ {(t + 1)}) \leq \ell_ {i} (\theta^ {(t)}) - \mu^ {(t)} \nabla \ell_ {i} (\theta^ {(t)}) ^ {\top} \Delta \theta^ {(t)} + \frac {L}{2} | | \mu^ {(t)} \Delta \theta^ {(t)} | | ^ {2} = \tag {8}
$$

$$
\ell_ {i} (\theta^ {(t)}) - \mu^ {(t)} \frac {1}{\alpha_ {i} ^ {(t)}} + \frac {(\mu^ {(t)}) ^ {2} L K}{2} \tag {9}
$$

$$
= \ell_ {i} \left(\theta^ {(t)}\right) - \frac {\mu^ {(t)}}{\alpha_ {i} ^ {(t)}} + \frac {\mu^ {(t)}}{2} \min _ {j} \frac {1}{\alpha_ {j} ^ {(t)}} \leq \ell_ {i} \left(\theta^ {(t)}\right) - \frac {\mu^ {(t)}}{2 \alpha_ {i} ^ {(t)}} <   \ell_ {i} \left(\theta^ {(t)}\right) \tag {10}
$$

This shows that our update decreases all the loss functions. We can average over inequality 9 over all losses and get for $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \ell _ { i } ( \boldsymbol { \theta } ) } \end{array}$ :

$$
\mathcal {L} \left(\theta^ {(t + 1)}\right) \leq \mathcal {L} \left(\theta^ {(t)}\right) - \mu^ {(t)} \frac {1}{K} \sum_ {i = 1} ^ {K} \frac {1}{\alpha_ {i} ^ {(t)}} + \frac {\left(\mu^ {(t)}\right) ^ {2} L K}{2} \leq \mathcal {L} \left(\theta^ {(t)}\right) - L K \left(\mu^ {(t)}\right) ^ {2} + \frac {\left(\mu^ {(t)}\right) ^ {2} L K}{2} = \mathcal {L} \left(\theta^ {(t)}\right) - \frac {L K \left(\mu^ {(t)}\right) ^ {2}}{2}. \tag {11}
$$

From this we can conclude that $\begin{array} { r } { \sum _ { \tau = 1 } ^ { t } \frac { L K ( \mu ^ { ( \tau ) } ) ^ { 2 } } { 2 } \leq \mathcal { L } ( \theta _ { 1 } ) - \mathcal { L } ( \theta ^ { ( t + 1 ) } ) . \mathrm { ~ A s ~ } \mathcal { L } ( \theta ^ { ( t ) } ) } \end{array}$ is bounded below we must have that the infinite serie $\begin{array} { r } { \sum _ { t = 1 } ^ { \infty } \frac { L K ( \mu ^ { ( t ) } ) ^ { 2 } } { 2 } < \infty , } \end{array}$ , and also $\mu ^ { ( t ) } \to 0$ . It follows that mi $1 _ { i \in [ K ] } 1 / \alpha _ { i } ^ { ( t ) } \to 0$ and therefore $\vert \vert \alpha ^ { ( t ) } \vert \vert  \infty$

We will now show that $| | 1 / \alpha ^ { ( t ) } | |$ is bounded for $t \to \infty$ . As the sequence ${ \mathcal { L } } ( \theta ^ { ( t ) } )$ is decreasing we have that the sequence $\boldsymbol { \theta } ^ { ( t ) }$ is in the sublevel set $\{ \theta : { \mathcal { L } } ( \theta ) \leq { \mathcal { L } } ( \theta _ { 0 } ) \}$ which is closed and bounded and therefore compact. If follows that there exists $M < \infty$ such that $| | g _ { i } ^ { ( t ) } | | \leq M$ for all t and $i \in [ K ]$ . We have for all i and $t , | 1 / \alpha _ { i } ^ { ( t ) } | = | ( g _ { i } ^ { \overline { { ( t ) } } } ) ^ { T } \theta ^ { ( t ) } | \leq \sqrt { K } | | g _ { i } ^ { ( t ) } | | \leq$ $\sqrt { K } M < \infty$ , and so $| | 1 / \alpha ^ { ( t ) } | |$ is bounded. Combining these two results we have $| | 1 / \alpha ^ { ( t ) } | | \geq \sigma _ { K } ( ( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } ) | | \alpha ^ { ( t ) } | |$ where $\sigma _ { K } ( ( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } )$ is the smallest singular value of $( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) }$ . Since the norm of $\alpha ^ { ( t ) }$ goes to infinity and the norm $1 / \alpha ^ { ( t ) }$ is bounded, it follows that $\sigma _ { K } ( ( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } )  0$

Now, since $\{ \theta : { \mathcal { L } } ( \theta ) \leq { \mathcal { L } } \theta _ { 0 } \}$ is compact there exists a subsequence $\theta ^ { ( t _ { j } ) }$ that converges to some point $\theta ^ { * }$ . As $\sigma _ { K } ( ( G ^ { ( t ) } ) ^ { T } \dot { G } ^ { ( t ) } ) \to \dot { 0 }$ we have from continuity that $\sigma _ { K } ( G _ { * } ^ { \top } G _ { * } ) = 0$ where $G _ { * }$ is the matrix of gradients at $\theta ^ { * }$ . This means that the gradients at θ are linearly dependent and therefore $\theta ^ { * }$ is Pareto stationary by assumption 5.1. As for all i the sequence $\{ \ell _ { i } ( \theta ^ { ( t ) } ) \} _ { t = } ^ { \infty }$ is monotonically decreasing and bounded below they all converges. Since $\ell _ { i } ( \theta ^ { * } )$ is the limit of a subsequence we get that $\ell _ { i } ( \theta ^ { ( t ) } ) \xrightarrow { t  \infty } \ell _ { i } ( \theta ^ { * } )$ .

![](images/c565126e732e9230b07a3d749f3854751cdef3b7bb69faf526d02961b56fe441.jpg)

We now show that if we add a convexity assumption then we can prove convergence to the Pareto front.

Theorem (5.5). Let $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ be the sequence generated by the update rule $\theta ^ { ( t + 1 ) } = \theta ^ { ( t ) } - \mu ^ { ( t ) } \Delta \theta ^ { ( t ) }$ where $\Delta \theta ^ { ( t ) } =$ $\textstyle \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { ( t ) } g _ { i } ^ { ( t ) }$ is the Nash bargaining solution $( G ^ { ( t ) } ) ^ { \top } G ^ { ( t ) } \alpha ^ { ( t ) } = 1 / \alpha ^ { ( t ) }$ . Set $\begin{array} { r } { \mu ^ { ( t ) } = \underset { i \in [ K ] } { \operatorname* { m i n } } \frac { 1 } { L K \alpha _ { i } ^ { ( t ) } } } \end{array}$ . Ifwe also assume that all the lossfunctions are convex then the sequence $\{ \theta ^ { ( t ) } \} _ { t = 1 } ^ { \infty }$ converges to a Pareto optimal point $\theta ^ { * }$

Proof. We note that this proof uses intermediate results from the proof of theorem 5.4. Given theorem 5.4 it suffices to prove that the sequence $\bar { \{ \theta ^ { ( t ) } \} } _ { t = 1 } ^ { \infty }$ converges, that would mean it converges to the partial limit θ<sup>∗</sup> that is Pareto stationary, and from convexity it would be Pareto optimal (as the optimizer of the convex combination of losses). For a convex and differential loss function, we have

$$
\ell (\theta^ {\prime}) \geq \ell (\theta) + \nabla \ell (\theta) ^ {\top} (\theta^ {\prime} - \theta) \tag {12}
$$

We can bound

$$
\left| \left| \theta^ {(t + 1)} - \theta^ {*} \right| \right| ^ {2} = \left| \left| \theta^ {(t)} - \mu^ {(t)} \Delta \theta^ {(t)} - \theta^ {*} \right| \right| ^ {2} \tag {13}
$$

$$
= | | \theta^ {(t)} - \theta^ {*} | | ^ {2} + (\mu^ {(t)}) ^ {2} | | \Delta \theta^ {(t)} | | ^ {2} - 2 \mu^ {(t)} (\Delta \theta^ {(t)}) ^ {\top} (\theta^ {(t)} - \theta^ {*}) \tag {14}
$$

$$
= | | \theta^ {(t)} - \theta^ {*} | | ^ {2} + (\mu^ {(t)}) ^ {2} K - 2 \mu^ {(t)} \sum_ {i} \alpha_ {i} ^ {(t)} (g _ {i} ^ {(t)}) ^ {\top} (\theta^ {(t)} - \theta^ {*}) \tag {15}
$$

$$
\leq | | \theta^ {(t)} - \theta^ {*} | | ^ {2} + (\mu^ {(t)}) ^ {2} K + 2 \mu^ {(t)} \sum_ {i} \alpha_ {i} ^ {(t)} (\ell_ {i} (\theta^ {*}) - \ell_ {i} (\theta^ {(t)})) \tag {16}
$$

$$
\leq | | \theta^ {(t)} - \theta^ {*} | | ^ {2} + (\mu^ {(t)}) ^ {2} K + 2 \mu^ {(t)} \sum_ {i} \alpha_ {i} ^ {(t)} (\ell_ {i} (\theta^ {(t + 1)}) - \ell_ {i} (\theta^ {(t)})) \tag {17}
$$

$$
\leq | | \theta^ {(t)} - \theta^ {*} | | ^ {2} + (\mu^ {(t)}) ^ {2} K - 2 \mu^ {(t)} \sum_ {i} \alpha_ {i} ^ {(t)} \frac {\mu^ {(t)}}{2 \alpha_ {i} ^ {(t)}} \tag {18}
$$

$$
= | | \theta^ {(t)} - \theta^ {*} | | ^ {2} \tag {19}
$$

In Eq. 15 we use the definition of $\Delta \theta ^ { ( t ) }$ and the fact that its norm equals $\sqrt { K }$ . In $\operatorname { E q }$ . 16 we use convexity and Eq. 12. Eq. 17 uses the fact that we show the losses are monotonically decreasing and converging to $\ell _ { i } ( \theta ^ { * } )$ . In Eq. 18 we use Eq. 10.

We have that the sequence $| | \theta ^ { ( t ) } - \theta ^ { * } | |$ is monotonically decreasing and bounded below by zero. Also, it has a subsequence that converges to zero, and so it must hold that the sequence $\left| \left| \theta ^ { ( t ) } - \theta ^ { * } \right| \right|$ also converge to zero, or equivalently $\theta ^ { ( t ) } \to \theta ^ { * }$

Proposition (3.1). Denote the objective for the optimization problem in $E q .$ . 5 by $\begin{array} { r } { \phi ( \alpha ) = \sum _ { i } \beta _ { i } ( \alpha ) + \varphi ( \alpha ) } \end{array}$ . Then, $\phi \left( \bar { \alpha ^ { ( \tau + 1 ) } } \right) \leq \phi \left( \alpha ^ { ( \tau ) } \right)$ for all $\tau \geq 1$

Proof. In our concave-convex procedure, we use the following linearization at the τ -th iteration:

$$
\tilde {\varphi} _ {\tau} (\alpha) = \varphi (\alpha^ {(\tau)}) + \nabla \varphi (\alpha^ {(\tau)}) ^ {\top} (\alpha - \alpha^ {(\tau)}).
$$

Then,

$$
\tilde {\varphi} _ {\tau} (\alpha^ {(\tau)}) = \varphi (\alpha^ {(\tau)}). \tag {20}
$$

Moreover, since $\varphi$ is concave and differentiable, we have that

$$
\varphi (\alpha^ {(\tau + 1)}) \leq \varphi (\alpha^ {(\tau)}) + \nabla \varphi (\alpha^ {(\tau)}) ^ {\top} (\alpha^ {(\tau + 1)} - \alpha^ {(\tau)}) = \tilde {\varphi} _ {\tau} (\alpha^ {(\tau + 1)}). \tag {21}
$$

Furthermore, since we minimize the convex objective $\begin{array} { r } { \sum _ { i } \beta _ { i } ( \alpha ) + \tilde { \varphi } ( \alpha ) } \end{array}$ at each iteration of our concave-convex procedure (in the convex feasible set),

$$
\sum_ {i} \beta_ {i} (\alpha^ {(\tau)}) + \tilde {\varphi} _ {\tau} (\alpha^ {(\tau)}) \geq \sum_ {i} \beta_ {i} (\alpha^ {(\tau + 1)}) + \tilde {\varphi} _ {\tau} (\alpha^ {(\tau + 1)}). \tag {22}
$$

Using Eq. 20–Eq. 22, we have that

$$
\begin{array}{l} \phi (\alpha^ {(\tau)}) = \sum_ {i} \beta_ {i} (\alpha^ {(\tau)}) + \varphi (\alpha^ {(\tau)}) = \sum_ {i} \beta_ {i} (\alpha^ {(\tau)}) + \tilde {\varphi} _ {\tau} (\alpha^ {(\tau)}) \geq \sum_ {i} \beta_ {i} (\alpha^ {(\tau + 1)}) + \tilde {\varphi} _ {t} (\alpha^ {(\tau + 1)}) \\ \geq \sum_ {i} \beta_ {i} (\alpha^ {(\tau + 1)}) + \varphi (\alpha^ {(\tau + 1)}) = \phi (\alpha^ {(\tau + 1)}). \\ \end{array}
$$

This proves the statement.

## B. Experimental Details

We provide here full experimental details for all experiments described in the main text.

Implementation Details. We apply all gradient manipulation methods to the gradients of the shared weights, with the exception of IMTL-G, which was applied to the feature-level gradients, as was originally proposed by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but it did not perform as well. We set the CAGrad’s c hyperparameter to 0.4, which was reported to yield the best performance for NYUv2 and Cityscapes (Liu et al., 2021a). For DWA (Liu et al., 2019b) we set the temperature hyperparameter to 2 which was found empirically to be optimum across all architectures. For RLW (Lin et al., 2021) we sample the weights from a normal distribution.

QM9. We adapt the QM9 example in PyTorch Geometric (Fey & Lenssen, 2019), and train the popular GNN model from Gilmer et al. (2017). We use the publicly available<sup>1</sup> implementation, the implementation is provided by Fey & Lenssen (2019). We use 110K molecules for training, 10K for validation, and 10K as a test set. Each task’s targets are normalized to have zero mean and unit standard deviation. We train each method for 300 epochs with batch-size of 120 and search for learning-rate (lr) in $\{ 1 e - 3 , 5 e - 4 , 1 e - 4 \}$ . We use a ReduceOnPlateau scheduler to decrease the lr when the validation $\Delta _ { m }$ metric stops improving. Additionally, we use the validation $\Delta _ { m }$ for early stopping.

Scene Understanding. We follow the training and evaluation procedure used in previous work on MTL (Liu et al., 2019b; Yu et al., 2020a; Liu et al., 2021a). However, unlike (Liu et al., 2019b), we add data augmentations (DA) during training for all the compared methods, similar to (Liu et al., 2021a;b). We train each method for 200 epochs with an initial learning-rate of 1e − 4. The learning-rate is reduced to 5e − 5 after 100 epochs. For MTL methods, we train a Multi-Task Attention Network (MTAN) (Liu et al., 2019b) built upon SegNet (Badrinarayanan et al., 2017). Similar to previous works (Liu et al., 2021a), the STL baseline refers to training task-specific SegNet models. We use a batch size of 2 and 8 for NYUv2 and CityScapes respectively. To align with previous work on MTL Liu et al. (2019b); Yu et al. (2020a); Liu et al. (2021a) we report the test performance averaged over the last 10 epochs

![](images/b1ebfe3ce49db8fd4d0e6b8dcb1a25b2a861e5d04ca3aeeeac8e9a6dfafe4238.jpg)

<details>
<summary>surface_3d</summary>

| \(\theta1\) | \(\theta2\) | Value |
| --- | --- | --- |
| -10~10 | -10~10 | ~-7.5 ~2.5 |
</details>

(a) Average loss

![](images/8c5648e4fbfdf758ff9ddeeeeeea497f52c169dfd7b25a946e03ab1fcc17dc9b.jpg)

<details>
<summary>surface_3d</summary>

| \(\theta1\) | \(\theta2\) | Value |
| --- | --- | --- |
| -10~10 | -10~10 | ~-1.5 ~0.5 |
</details>

(b) $\ell _ { 1 }$

![](images/8f6660e49bdc7955c3e333fbf5414d08b19bdf1013e3a762d8766835a5aebfc2.jpg)

<details>
<summary>surface_3d</summary>

| \(\theta1\) | \(\theta2\) | Value |
| --- | --- | --- |
| -10~10 | -10~10 | ~-15 ~5 |
</details>

(c) $\ell _ { 2 }$  
Figure 5. Illustrative example. Visualization of the loss surfaces in our illustrative example of Figure 1

MT10. Following previous works (Yu et al., 2020a; Liu et al., 2021a; Sodhani et al., 2021), we use multitask Soft Actor-Critic (SAC) (Haarnoja et al., 2018) as the base RL algorithm for PCGrad, CAGrad, and Nash-MTL. We follow the same experiment setup from and evaluation protocol as in Sodhani et al. (2021); Liu et al. (2021a). Each method is trained over 2 million steps with a batch size of 1280. The agent is evaluated once every 10K environment steps to obtain the average success over tasks. The reported success rate for the agent is the best average performance over all evaluation steps. We repeat this procedure over 10 random seeds, and the performance of each method is obtained by averaging the mean success over all random seeds. For all Nash-MTL experiments, we use a single CCP step in order to speed up computation.

Illustrative Example. We provide here the details for the illustrative example of Figure 1. We use a slightly modified version of the illustrative example in (Liu et al., 2021a). We first present the learning problem from (Liu et al., 2021a): Let $\theta = ( \theta _ { 1 } , \theta _ { 2 } ) \in \mathbb { R } ^ { 2 }$ , and consider the following objectives:

$$
\tilde {\ell} _ {1} (\theta) = c _ {1} (\theta) f _ {1} (\theta) + c _ {2} (\theta) g _ {1} (\theta) \quad \text {and} \quad \tilde {\ell} _ {2} (\theta) = c _ {1} (\theta) f _ {2} (\theta) + c _ {2} (\theta) g _ {2} (\theta), \text {where}
$$

$$
f _ {1} (\theta) = \log (\max (| 0. 5 (- \theta_ {1} - 7) - \tanh (- \theta_ {2}) |, 5 e - 6)) + 6,
$$

$$
f _ {2} (\theta) = \log (\max (| 0. 5 (- \theta_ {1} + 3) - \tanh (- \theta_ {2}) + 2 |, 5 e - 6)) + 6,
$$

$$
g _ {1} (\theta) = ((- \theta_ {1} + 7) ^ {2} + 0. 1 \cdot (- \theta_ {2} - 8) ^ {2}) / 1 0 - 2 0,
$$

$$
g _ {2} (\theta) = ((- \theta_ {1} - 7) ^ {2} + 0. 1 \cdot (- \theta_ {2} - 8) ^ {2}) / 1 0 - 2 0,
$$

$$
c _ {1} (\theta) = \max (\tanh (0. 5 \theta_ {2}), 0) \quad \text { and } \quad c _ {2} (\theta) = \max (\tanh (- 0. 5 \theta_ {2}), 0)
$$

We now set $\ell _ { 1 } = 0 . 1 \cdot \tilde { \ell } _ { 1 }$ and $\ell _ { 2 } ~ = ~ \tilde { \ell } _ { 2 }$ as our objectives, see Figure 5. We use five different initialization points $\{ ( - 8 . 5 , 7 . 5 ) , ( 0 . 0 , 0 . 0 ) , ( 9 . 0 , 9 . 0 ) , ( - 7 . 5 , - 0 . 5 ) , ( 9 , - 1 . 0 ) \}$ }. We use the Adam optimizer and train each method for 35K iteration with learning rate of $1 e - 3$

## C. Computing Task Gradient at the Features-Level

One common approach for speeding and scaling up MTL methods is using feature-level gradients (from the representation layer) as a surrogate for the task-level gradients computed over the entire shared backbone (Sener & Koltun, 2018; Liu et al., 2021b; Javaloy & Valera, 2021). In this section we evaluate Nash-MTL while using the feature-level gradients for computing the Nash bargaining solution. On the QM9 dataset, we found this approach to accelerate training by $\sim \times 6$ . However, this acceleration method greatly hurts the performance of Nash-MTL, yielding a test $\Delta _ { m }$ of 179.2 (compared to 62.0 when using full gradients). This result is not surprising, since we are mainly interested in the inner products of gradients. Consider $g _ { i } ^ { \top } g _ { j } = ( \nabla _ { \theta } z \nabla _ { z } \ell _ { i } ) ^ { \top } \nabla _ { \theta } z \nabla _ { z } \ell _ { j }$ , where z is the feature representation and θ the shared parameters vector. We see that for $\nabla _ { z } \ell _ { i } ^ { \top } \nabla _ { z } \ell _ { j }$ to accurately approximate $g _ { i } ^ { \top } g _ { j }$ we need $\nabla _ { \theta } z ^ { \top } \nabla _ { \theta } z \approx I$ which is a strong and restricting requirement.

![](images/011b3706342711563a1759fb76daf2fc465d93ac1acb06beba56abae33a91cb6.jpg)

<details>
<summary>line</summary>

| Epoch | CCP Steps::1 | CCP Steps::20 | CCP Steps::40 |
| --- | --- | --- | --- |
| 0 | ~61 | ~61 | ~61 |
| 25 | ~20 | ~19 | ~18 |
| 50 | ~10 | ~9 | ~8 |
| 75 | ~5 | ~4 | ~3 |
| 100 | ~-1 | ~-2 | ~-1 |
| 125 | ~-2 | ~-3 | ~-2 |
| 150 | ~-3 | ~-4 | ~-3 |
| 175 | ~-3 | ~-4 | ~-3 |
| 200 | ~-3 | ~-4 | ~-3 |
</details>

Figure 6. NYUv2. The mean and standard divination of test $\Delta _ { m }$ throughout the training process, for Nash-MTL with 1, 20, and 40 CCP steps.

## D. Additional Experiments

## D.1. Full Results for Multi-task Regression

We provide here the full results for the QM9 experiment of Section 6.1. The results for all methods over all 11 tasks are presented in Table 6. Nash-MTL achieves the best $\Delta _ { m }$ and MR performance. Despite being a simple approach, SI performs well compared to more sophisticated baselines. It achieves the third/second best $\Delta _ { m }$ and MR respectively. The other scale-invariant method, IMTL-G, also performs well in this learning setup.

Table 6. QM9. Test performance averaged over 3 random seeds.

<table><tr><td rowspan="2"></td><td> $\mu$ </td><td> $\alpha$ </td><td> $\epsilon_{\text{HOMO}}$ </td><td> $\epsilon_{\text{LUMO}}$ </td><td> $\langle R^2 \rangle$ </td><td>ZPVE</td><td> $U_0$ </td><td>U</td><td>H</td><td>G</td><td> $c_v$ </td><td rowspan="2">MR ↓</td><td rowspan="2"> $\Delta_m \% \downarrow$ </td></tr><tr><td colspan="5"></td><td colspan="6">MAE ↓</td></tr><tr><td>STL</td><td>0.067</td><td>0.181</td><td>60.57</td><td>53.91</td><td>0.502</td><td>4.53</td><td>58.8</td><td>64.2</td><td>63.8</td><td>66.2</td><td>0.072</td><td></td><td></td></tr><tr><td>LS</td><td>0.106</td><td>0.325</td><td>73.57</td><td>89.67</td><td>5.19</td><td>14.06</td><td>143.4</td><td>144.2</td><td>144.6</td><td>140.3</td><td>0.128</td><td>6.8</td><td>177.6</td></tr><tr><td>SI</td><td>0.309</td><td>0.345</td><td>149.8</td><td>135.7</td><td>1.00</td><td>4.50</td><td>55.3</td><td>55.75</td><td>55.82</td><td>55.27</td><td>0.112</td><td>4.0</td><td>77.8</td></tr><tr><td>RLW</td><td>0.113</td><td>0.340</td><td>76.95</td><td>92.76</td><td>5.86</td><td>15.46</td><td>156.3</td><td>157.1</td><td>157.6</td><td>153.0</td><td>0.137</td><td>8.2</td><td>203.8</td></tr><tr><td>DWA</td><td>0.107</td><td>0.325</td><td>74.06</td><td>90.61</td><td>5.09</td><td>13.99</td><td>142.3</td><td>143.0</td><td>143.4</td><td>139.3</td><td>0.125</td><td>6.4</td><td>175.3</td></tr><tr><td>UW</td><td>0.386</td><td>0.425</td><td>166.2</td><td>155.8</td><td>1.06</td><td>4.99</td><td>66.4</td><td>66.78</td><td>66.80</td><td>66.24</td><td>0.122</td><td>5.3</td><td>108.0</td></tr><tr><td>MGDA</td><td>0.217</td><td>0.368</td><td>126.8</td><td>104.6</td><td>3.22</td><td>5.69</td><td>88.37</td><td>89.4</td><td>89.32</td><td>88.01</td><td>0.120</td><td>5.9</td><td>120.5</td></tr><tr><td>PCGrad</td><td>0.106</td><td>0.293</td><td>75.85</td><td>88.33</td><td>3.94</td><td>9.15</td><td>116.36</td><td>116.8</td><td>117.2</td><td>114.5</td><td>0.110</td><td>5.0</td><td>125.7</td></tr><tr><td>CAGrad</td><td>0.118</td><td>0.321</td><td>83.51</td><td>94.81</td><td>3.21</td><td>6.93</td><td>113.99</td><td>114.3</td><td>114.5</td><td>112.3</td><td>0.116</td><td>5.7</td><td>112.8</td></tr><tr><td>IMTL-G</td><td>0.136</td><td>0.287</td><td>98.31</td><td>93.96</td><td>1.75</td><td>5.69</td><td>101.4</td><td>102.4</td><td>102.0</td><td>100.1</td><td>0.096</td><td>4.7</td><td>77.2</td></tr><tr><td>Nash-MTL</td><td>0.102</td><td>0.248</td><td>82.95</td><td>81.89</td><td>2.42</td><td>5.38</td><td>74.5</td><td>75.02</td><td>75.10</td><td>74.16</td><td>0.093</td><td>2.5</td><td>62.0</td></tr></table>

## D.2. Effect of the Number of CCP steps

In this section, we investigate the effect of varying the number of CCP steps in our efficient approximation to $G ^ { \top } G \alpha = 1 / \alpha$ (presented in Section 3.2). We use the NYUv2 dataset and train Nash-MTL with CCP sequences of 1, 20, and 40 steps at each (parameters) optimization step.

We found that increasing the CCP sequence improves the approximation to the optimal α. Using a single CCP iteration results with $G ^ { \top } G \alpha \approx 1 \bar { / } \alpha$ in 91.5% of the optimization steps, whereas increasing the number of iterations to 20 increases the proportion of optimal solutions to 93.5%. However, we found the improved solution to have no significant improvement in MTL performance. Figure 6 presents the test $\Delta _ { m }$ throughout the training process.

![](images/189b7bcb94100f35b125daac65113c2adf318301f0a336e97498d8fe5eb58a0b.jpg)

<details>
<summary>line</summary>

| Step | Segmentation (Weight) | Depth (Weight) | Normal (Weight) |
| --- | --- | --- | --- |
| 0 | ~0.5 | ~1.0 | ~2.5 |
| 10000 | ~0.3 | ~0.4 | ~2.5 |
| 20000 | ~0.2 | ~0.3 | ~2.5 |
| 30000 | ~0.2 | ~0.3 | ~2.5 |
| 40000 | ~0.2 | ~0.3 | ~2.5 |
| 50000 | ~0.2 | ~0.3 | ~2.5 |
| 60000 | ~0.2 | ~0.3 | ~2.5 |
| 70000 | ~0.2 | ~0.3 | ~2.5 |
| 80000 | ~0.2 | ~0.3 | ~2.5 |
</details>

(a) NYUv2

![](images/440d5d0bc614df1ba95513e7c8b1901d16c4a925314f85c61e89c3b2d7bcbfa9.jpg)

<details>
<summary>line</summary>

| Step | button-press-topdown | drawer-close | peg-insert-side | push | window-close | door-open | drawer-open | pick-place | reach | window-open |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 | ~0.3 |
| 10000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 20000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 30000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 40000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 50000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 60000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
| 70000 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 | ~0.15 |
</details>

(b) MT10  
Figure 7. Task Weights. Task weights obtained from Nash-MTL throughout the optimization process, for (a) NYUv2, and; (b) MT10 with weight update frequency of 100. For better visualization, each point corresponds to a moving average with window size 200.

Table 7. QM9. Runtime per epoch in minutes.

<table><tr><td></td><td>Runtime [Min.]</td></tr><tr><td>LS</td><td>0.54</td></tr><tr><td>MGDA</td><td>7.25</td></tr><tr><td>PCGrad</td><td>7.47</td></tr><tr><td>CAGrad</td><td>6.85</td></tr><tr><td>Nash-MTL</td><td>6.76</td></tr><tr><td>Nash-MTL-5</td><td>1.81</td></tr><tr><td>Nash-MTL-50</td><td>0.69</td></tr></table>

## D.3. Modifying the CCP Objective

In this section we examine the effect of changing the objective of the CCP procedure described in 3.2 (Eq. 5). Here we first solve the convex optimization problem of Eq. 4 to obtain $\alpha _ { 0 } .$ . If $G ^ { \top } G \alpha _ { 0 } \approx 1 / \alpha _ { 0 }$ we stop. Else we use the CCP procedure with objective $\varphi ( \alpha )$ , starting at $\alpha _ { 0 }$ (dropping the addition $\textstyle \sum _ { i } \beta _ { i }$ term from Eq. 5). While this objective is more natural, in practice we observe a performance degradation in terms of MTL performance. We obtain $\Delta _ { m } = 6 4 . 4$ for the QM9 dataset (vs. 62 reported in the paper), $\Delta _ { m } = - 3 . 5 ( \mathrm { v s . } - 4 )$ for NYUv2 and $\Delta _ { m } = 8 . 8$ (vs. 6.8) for Cityscapes.

## D.4. Visualizing Task Weights

Our method, Nash-MTL, can essentially be viewed as a principled approach for producing dynamic task weights. Here we visualize these task weights throughout the training process using the NYUv2 dataset (Figure 7) and the MT10 dataset (Figure 7(b)).

## D.5. Verifying the Task Independence Assumption

Here we provide an empirical justification for our assumption in Section 3 which we state here once again: we assume that the task gradients are linearly independent for each point θ that is not Pareto stationary. To investigate whether this assumption holds in our experiments, we observe the smallest singular value of gradients Gram matrix $\sigma _ { K } ( G ^ { \top } G )$ . The results are presented in Figure 8. We see that for both datasets the $\sigma _ { k }$ decreases as the learning progresses. For the NYUv2 experiment, the smallest singular value remains fairly large throughout the entire training process. On the QM9 dataset, $\sigma _ { K }$ decreases more significantly, to around ∼ 1e − 8.

![](images/68167bfa8b5424e0118b0fec7bb6c4c62f6c526dd42b4b8c776d452737958b33.jpg)

<details>
<summary>line</summary>

| Step | \(\sigma K(G^T G)\) |
| --- | --- |
| 0 | ~0.25 |
| 10000 | ~0.03 |
| 20000 | ~0.02 |
| 30000 | ~0.02 |
| 40000 | ~0.02 |
| 50000 | ~0.02 |
| 60000 | ~0.01 |
| 70000 | ~0.01 |
| 80000 | ~0.01 |
</details>

(a) NYUv2

![](images/4aae683cdec520d04db44fe12641d4789d5913601aef326e64665ad63c653cdd.jpg)

<details>
<summary>line</summary>

| Step | \(\sigma K(G^T G)\) |
| --- | --- |
| 0 | ~10^-3 |
| 50000 | ~10^-6 |
| 100000 | ~10^-7 |
| 150000 | ~10^-8 |
| 200000 | ~10^-9 |
| 250000 | ~10^-9 |
</details>

(b) QM9  
Figure 8. Smallest singular value of $\mathbf { \Sigma } _ { G } ^ { \top } G$ throughout the training process.