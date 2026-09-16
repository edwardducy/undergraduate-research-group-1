# Gradient Surgery for Multi-Task Learning

Tianhe Yu<sup>1</sup>, Saurabh Kumar<sup>1</sup>, Abhishek Gupta<sup>2</sup>, Sergey Levine<sup>2</sup>, Karol Hausman<sup>3</sup>, Chelsea Finn<sup>1</sup>

Stanford University<sup>1</sup>, UC Berkeley<sup>2</sup>, Robotics at Google<sup>3</sup>

tianheyu@cs.stanford.edu

## Abstract

While deep learning and deep reinforcement learning (RL) systems have demonstrated impressive results in domains such as image classification, game playing, and robotic control, data efficiency remains a major challenge. Multi-task learning has emerged as a promising approach for sharing structure across multiple tasks to enable more efficient learning. However, the multi-task setting presents a number of optimization challenges, making it difficult to realize large efficiency gains compared to learning tasks independently. The reasons why multi-task learning is so challenging compared to single-task learning are not fully understood. In this work, we identify a set of three conditions of the multi-task optimization landscape that cause detrimental gradient interference, and develop a simple yet general approach for avoiding such interference between task gradients. We propose a form of gradient surgery that projects a task’s gradient onto the normal plane of the gradient of any other task that has a conflicting gradient. On a series of challenging multi-task supervised and multi-task RL problems, this approach leads to substantial gains in efficiency and performance. Further, it is model-agnostic and can be combined with previously-proposed multi-task architectures for enhanced performance.

## 1 Introduction

While deep learning and deep reinforcement learning (RL) have shown considerable promise in enabling systems to learn complex tasks, the data requirements of current methods make it difficult to learn a breadth of capabilities, particularly when all tasks are learned individually from scratch. A natural approach to such multi-task learning problems is to train a network on all tasks jointly, with the aim of discovering shared structure across the tasks in a way that achieves greater efficiency and performance than solving tasks individually. However, learning multiple tasks all at once results is a difficult optimization problem, sometimes leading to worse overall performance and data efficiency compared to learning tasks individually [42, 50]. These optimization challenges are so prevalent that multiple multi-task RL algorithms have considered using independent training as a subroutine of the algorithm before distilling the independent models into a multi-tasking model [32, 42, 50, 21, 56], producing a multi-task model but losing out on the efficiency gains over independent training. If we could tackle the optimization challenges of multi-task learning effectively, we may be able to actually realize the hypothesized benefits of multi-task learning without the cost in final performance.

While there has been a significant amount of research in multi-task learning [6, 49], the optimization challenges are not well understood. Prior work has described varying learning speeds of different tasks [8, 26] and plateaus in the optimization landscape [52] as potential causes, whereas a range of other works have focused on the model architecture [40, 33]. In this work, we instead hypothesize that one of the main optimization issues in multi-task learning arises from gradients from different tasks conflicting with one another in a way that is detrimental to making progress. We define two gradients to be conflicting if they point away from one another, i.e., have a negative cosine similarity. We hypothesize that such conflict is detrimental when a) conflicting gradients coincide with b) high positive curvature and c) a large difference in gradient magnitudes.

34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.

As an illustrative example, consider the 2D optimization landscapes of two task objectives in Figure 1a-c. The optimization landscape of each task consists of a deep valley, a property that has been observed in neural network optimization landscapes [22], and the bottom of each valley is characterized by high positive curvature and large differences in the task gradient magnitudes. Under such circumstances, the multi-task gradient is dominated by one task gradient, which comes at the cost of degrading the performance of the other task. Further, due to high

![](images/284e88009b4d81e0d816f88e05a7b369f63fb43f1d10443ee01b91cb4a4badc5.jpg)

<details>
<summary>heatmap</summary>

| Objective | Optimization Method |
| --- | --- |
| Task 1 Objective | Adam + PCGrad (Black line with arrow) |
| Task 2 Objective | Adam + PCGrad (Blue arrow) |
| Adam | Adam + PCGrad (Black line with arrow) |
| Adam + PCGrad | Adam + PCGrad (Black line with arrow) |
</details>

Figure 1: Visualization of PCGrad on a 2D multi-task optimization problem. (a) A multi-task objective landscape. (b) & (c) Contour plots of the individual task objectives that comprise (a). (d) Trajectory of gradient updates on the multi-task objective using the Adam optimizer. The gradient vectors of the two tasks at the end of the trajectory are indicated by blue and red arrows, where the relative lengths are on a log scale.(e) Trajectory of gradient updates on the multi-task objective using Adam with PCGrad. For (d) and (e), the optimization trajectory goes from black to yellow.

curvature, the improvement in the dominating task may be overestimated, while the degradation in performance of the non-dominating task may be underestimated. As a result, the optimizer struggles to make progress on the optimization objective. In Figure 1d), the optimizer reaches the deep valley of task 1, but is unable to traverse the valley in a parameter setting where there are conflicting gradients, high curvature, and a large difference in gradient magnitudes (see gradients plotted in Fig. 1d). In Section 5.3, we find experimentally that this tragic triad also occurs in a higher-dimensional neural network multi-task learning problem.

The core contribution of this work is a method for mitigating gradient interference by altering the gradients directly, i.e. by performing “gradient surgery.” If two gradients are conflicting, we alter the gradients by projecting each onto the normal plane of the other, preventing the interfering components of the gradient from being applied to the network. We refer to this particular form of gradient surgery as projecting conflicting gradients (PCGrad). PCGrad is model-agnostic, requiring only a single modification to the application of gradients. Hence, it is easy to apply to a range of problem settings, including multi-task supervised learning and multi-task reinforcement learning, and can also be readily combined with other multi-task learning approaches, such as those that modify the architecture. We theoretically prove the local conditions under which PCGrad improves upon standard multi-task gradient descent, and we empirically evaluate PCGrad on a variety of challenging problems, including multi-task CIFAR classification, multi-objective scene understanding, a challenging multi task RL domain, and goal-conditioned RL. Across the board, we find PCGrad leads to substantial improvements in terms of data efficiency, optimization speed, and final performance compared to prior approaches, including a more than 30% absolute improvement in multi-task reinforcement learning problems. Further, on multi-task supervised learning tasks, PCGrad can be successfully combined with prior state-of-the-art methods for multi-task learning for even greater performance.

## 2 Multi-Task Learning with PCGrad

While the multi-task problem can in principle be solved by simply applying a standard single-task algorithm with a suitable task identifier provided to the model, or a simple multi-head or multi-output model, a number of prior works [42, 50, 53] have found this learning problem to be difficult. In this section, we introduce notation, identify possible causes for the difficulty of multi-task optimization, propose a simple and general approach to mitigate it, and theoretically analyze the proposed approach.

## 2.1 Preliminaries: Problem and Notation

The goal of multi-task learning is to find parameters $\theta$ of a model $f _ { \theta }$ that achieve high average performance across all the training tasks drawn from a distribution of tasks $p ( \tau )$ . More formally, we aim to solve the problem: $\operatorname* { n i n } _ { \theta } \mathbb { E } _ { \mathcal { T } _ { i } \sim p ( \mathcal { T } ) } \left[ \mathcal { L } _ { i } ( \theta ) \right]$ ], where $\mathcal { L } _ { i }$ is a loss function for the i-th task $\mathcal { T } _ { i }$ that we want to minimize. For a set of tasks, $\{ \mathcal { T } _ { i } \}$ , we denote the multi-task loss as $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \sum _ { i } \mathcal { L } _ { i } ( \boldsymbol { \theta } ) } \end{array}$ and the gradients of each task as $\mathbf { g } _ { i } = \nabla \dot { \mathcal { L } } _ { i } ( \dot { \theta } )$ for a particular θ. (We drop the reliance on θ in the notation for brevity.) To obtain a model that solves a specific task from the task distribution $p ( \mathcal { T } )$ we define a task-conditioned model $f _ { \boldsymbol { \theta } } ( \boldsymbol { y } | \boldsymbol { x } , z _ { i } )$ , with input $x ,$ output y, and encoding $z _ { i }$ for task $\tau _ { i } .$ which could be provided as a one-hot vector or in any other form.

## 2.2 The Tragic Triad: Conflicting Gradients, Dominating Gradients, High Curvature

We hypothesize that a key optimization issue in multi-task learning arises from conflicting gradients, where gradients for different tasks point away from one another as measured by a negative inner product. However, conflicting gradients are not detrimental on their own. Indeed, simply averaging task gradients should provide the correct solution to descend the multi-task objective. However, there are conditions under which such conflicting gradients lead to significantly degraded performance. Consider a two-task optimization problem. If the gradient of one task is much larger in magnitude than the other, it will dominate the average gradient. If there is also high positive curvature along the directions of the task gradients, then the improvement in performance from the dominating task may be significantly overestimated, while the degradation in performance from the dominated task may be significantly underestimated. Hence, we can characterize the co-occurrence of three conditions as follows: (a) when gradients from multiple tasks are in conflict with one another (b) when the difference in gradient magnitudes is large, leading to some task gradients dominating others, and (c) when there is high curvature in the multi-task optimization landscape. We formally define the three conditions below.

Definition 1. We define $\phi _ { i j }$ as the angle between two task gradients $\mathbf { g } _ { i }$ and $\mathbf { g } _ { j }$ . We define the gradients as conflicting when cos $\phi _ { i j } < 0$

Definition 2. We define the gradient magnitude similarity between two gradients $\mathbf { g } _ { i }$ and $\mathbf { g } _ { j }$ as $\begin{array} { r } { \Phi ( \mathbf { g } _ { i } , \mathbf { g } _ { j } ) = \frac { 2 \| \mathbf { g } _ { i } \| _ { 2 } \| \mathbf { g } _ { j } \| _ { 2 } } { \| \mathbf { g } _ { i } \| _ { 2 } ^ { 2 } + \| \mathbf { g } _ { j } \| _ { 2 } ^ { 2 } } } \end{array}$

When the magnitude of two gradients is the same, this value is equal to 1. As the gradient magnitudes become increasingly different, this value goes to zero.

Definition 3. We define multi-task curvature as ${ \bf H } ( { \mathcal L } ; \theta , \theta ^ { \prime } ) ~ = ~ \int _ { 0 } ^ { 1 } \nabla { \mathcal L } ( \theta ) ^ { T } \nabla ^ { 2 } { \mathcal L } ( \theta + a ( \theta ^ { \prime } -$ $\theta ) ) { \nabla } \mathcal { L } ( \theta ) d a ,$ which is the averaged curvature ofL between θ and θ<sup>0</sup> in the direction ofthe multi-task gradient $\nabla { \mathcal { L } } ( \theta )$

When $\mathbf { H } ( \mathcal { L } ; \theta , \theta ^ { \prime } ) > C$ for some large positive constant $C ,$ for model parameters θ and $\theta ^ { \prime }$ at the current and next iteration, we characterize the optimization landscape as having high curvature.

We aim to study the tragic triad and observe the presence of the three conditions through two examples. First, consider the two-dimensional optimization landscape illustrated in Fig. 1a, where the landscape for each task objective corresponds to a deep and curved valley with large curvatures (Fig. 1b and 1c). The optima of this multi-task objective correspond to where the two valleys meet. More details on the optimization landscape are in Appendix D. Particular points of this optimization landscape exhibit the three described conditions, and we observe that, the Adam [30] optimizer stalls precisely at one of these points (see Fig. 1d), preventing it from reaching an optimum. This provides some empirical evidence for our hypothesis. Our experiments in Section 5.3 further suggest that this phenomenon occurs in multi-task learning with deep networks. Motivated by these observations, we develop an algorithm that aims to alleviate the optimization challenges caused by conflicting gradients, dominating gradients, and high curvature, which we describe next.

## 2.3 PCGrad: Project Conflicting Gradients

Our goal is to break one condition of the tragic triad by directly altering the gradients themselves to prevent conflict. In this section, we outline our approach for altering the gradients. In the next section, we will theoretically show that de-conflicting gradients can benefit multi-task learning when dominating gradients and high curvatures are present.

To be maximally effective and widely applicable, we aim to alter the gradients in a way that allows for positive interactions between the task gradients and does not introduce assumptions on the form of the model. Hence, when gradients do not conflict, we do not change the gradients. When gradients do conflict, the goal of PCGrad is to modify the gradients for each task so as to minimize negative conflict with other task gradients, which will in turn mitigate under- and over-estimation problems arising from high curvature.

To deconflict gradients during optimization, PCGrad adopts a simple procedure: if the gradients between two tasks are in conflict, i.e. their cosine similarity is negative, we project the gradient of each task onto the normal plane of the gradient of the other task. This amounts to removing the conflicting component of the gradient for the task, thereby reducing the amount of destructive gradient interference between tasks. A pictorial description of this idea is shown in Fig. 2.

Algorithm 1 PCGrad Update Rule

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Require: Model parameters $\theta$, task minibatch $\mathcal{B} = \{\mathcal{T}_k\}$ $\mathbf{g}_k \leftarrow \nabla_\theta \mathcal{L}_k(\theta) \ \forall k$ $\mathbf{g}_k^{\text{PC}} \leftarrow \mathbf{g}_k \ \forall k$
for $\mathcal{T}_i \in \mathcal{B}$ do
    for $\mathcal{T}_j \stackrel{\text{uniformly}}{\sim} \mathcal{B} \setminus \mathcal{T}_i$ in random order do
        if $\mathbf{g}_i^{\text{PC}} \cdot \mathbf{g}_j &lt; 0$ then
            // Subtract the projection of $\mathbf{g}_i^{\text{PC}}$ onto $\mathbf{g}_j$
            Set $\mathbf{g}_i^{\text{PC}} = \mathbf{g}_i^{\text{PC}} - \frac{\mathbf{g}_i^{\text{PC}} \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$
return update $\Delta\theta = \mathbf{g}^{\text{PC}} = \sum_i \mathbf{g}_i^{\text{PC}}$
</div>

![](images/b968c68df9fa112bae5c47760fb6d776f129fde49adbaf69f8ccea2403bd90d7.jpg)  
(a)

![](images/7b5e12997e8f55113123b166c26e0d72ce4ff0d63dd61bd416ab8c31bc1d6d85.jpg)  
(b)

![](images/e70396ba3373f21d136996ef4eedae2cf50b9d3dbd3ba7d78f32b2f560ea8d74.jpg)  
(c)

![](images/b3acf980d374fadb6edc63f0acd34289f255639a6d7c84fe1dc5263282b0355f.jpg)  
(d)  
Figure 2: Conflicting gradients and PCGrad. In (a), tasks i and j have conflicting gradient directions, which can lead to destructive interference. In (b) and (c), we illustrate the PCGrad algorithm in the case where gradients are conflicting. PCGrad projects task i’s gradient onto the normal vector of task j’s gradient, and vice versa. Non-conflicting task gradients (d) are not altered under PCGrad, allowing for constructive interaction.

Suppose the gradient for task $\mathcal { T } _ { i }$ is $\mathbf { g } _ { i }$ , and the gradient for task $\tau _ { j }$ is ${ \bf { g } } _ { j }$ . PCGrad proceeds as follows: (1) First, it determines whether $\mathbf { g } _ { i }$ conflicts with $\mathbf { g } _ { j }$ by computing the cosine similarity between vectors $\mathbf { g } _ { i }$ and ${ \bf g } _ { j }$ , where negative values indicate conflicting gradients. (2) If the cosine similarity is negative, we replace $\mathbf { g } _ { i }$ by its projection onto the normal plane of $\begin{array} { r } { \mathbf { g } _ { j } \colon \mathbf { g } _ { i } = \mathbf { g } _ { i } - \frac { \mathbf { g } _ { i } \cdot \mathbf { g } _ { j } } { \| \mathbf { g } _ { i } \| ^ { 2 } } \mathbf { g } _ { j } } \end{array}$ . If the gradients are not in conflict, i.e. cosine similarity is non-negative, the original gradient $\mathbf { g } _ { i }$ remains unaltered. (3) PCGrad repeats this process across all of the other tasks sampled in random order from the current batch $\bar { T _ { j } } \forall j \dag \ d j$ , resulting in the gradient $\mathbf { g } _ { i } ^ { \mathrm { P C } }$ that is applied for task $\mathcal { T } _ { i }$ . We perform the same procedure for all tasks in the batch to obtain their respective gradients. The full update procedure is described in Algorithm 1 and a discussion on using a random task order is included in Appendix H.

This procedure, while simple to implement, ensures that the gradients that we apply for each task per batch interfere minimally with the other tasks in the batch, mitigating the conflicting gradient problem, producing a variant on standard first-order gradient descent in the multi-objective setting. In practice, PCGrad can be combined with any gradient-based optimizer, including commonly used methods such as SGD with momentum and Adam [30], by simply passing the computed update to the respective optimizer instead of the original gradient. Our experimental results verify the hypothesis that this procedure reduces the problem of conflicting gradients, and find that, as a result, learning progress is substantially improved.

## 2.4 Theoretical Analysis of PCGrad

In this section, we theoretically analyze the performance of PCGrad with two tasks:

Definition 4. Consider two task loss functions $\mathcal { L } _ { 1 } : \mathbb { R } ^ { n } $ R and $\mathcal { L } _ { 2 } : \mathbb { R } ^ { n }  \mathbb { R }$ . We define the two-task learning objective as $\mathcal { L } ( \dot { \theta } ) = \mathcal { L } _ { 1 } ( \theta ) + \mathcal { L } _ { 2 } ( \theta )$ for all $\theta \in \mathbb { R } ^ { n }$ , where $\mathbf { g } _ { 1 } = \nabla \mathcal { L } _ { 1 } ( \theta )$ $\mathbf { g } _ { 2 } = \nabla \mathcal { L } _ { 2 } ( \theta )$ , and $\mathbf { g } = \mathbf { g } _ { 1 } + \mathbf { g } _ { 2 }$

We first aim to verify that the PCGrad update corresponds to a sensible optimization procedure under simplifying assumptions. We analyze convergence of PCGrad in the convex setting, under standard assumptions in Theorem 1. For additional analysis on convergence, including the non-convex setting, with more than two tasks, and with momentum-based optimizers, see Appendices A.1 and A.4

Theorem 1. Assume $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are convex and differentiable. Suppose the gradient of L is $L _ { - }$ Lipschitz with $L > 0 .$ . Then, the PCGrad update rule with step size $t \overset { \_ } { \leq } \frac { 1 } { L }$ will converge to either (1) a location in the optimization landscape where cos $\left( \phi _ { 1 2 } \right) = - 1 o r \left( 2 \right)$ the optimal value $\mathcal { L } ( \theta ^ { \ast } )$ .

Proof. See Appendix A.1.

![](images/165f4b4eec915a01a6d030aa1acebbb79a8961dee90d166bece8b7d4a79e257f.jpg)

Theorem 1 states that application of the PCGrad update in the two-task setting with a convex and Lipschitz multi-task loss function L leads to convergence to either the minimizer of $\mathcal { L }$ or a potentially sub-optimal objective value. A sub-optimal solution occurs when the cosine similarity between the gradients of the two tasks is exactly −1, i.e. the gradients directly conflict, leading to zero gradient after applying PCGrad. However, in practice, since we are using SGD, which is a noisy estimate of the true batch gradients, the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be $- 1$ , thus avoiding this scenario. Note that, in theory, convergence may be slow if cos $\left( \phi _ { 1 2 } \right)$ hovers near −1. However, we don’t observe this in practice, as seen in the objective-wise learning curves in Appendix B.

Now that we have checked the sensibility of PCGrad, we aim to understand how PCGrad relates to the three conditions in the tragic triad. In particular, we derive sufficient conditions under which PCGrad achieves lower loss after one update. Here, we still analyze the two task setting, but no longer assume convexity of the loss functions.

Definition 5. We define the multi-task curvature bounding measure $\xi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) = 0 $ $\begin{array} { r } { \cos ^ { 2 } \phi _ { 1 2 } ) \frac { \| \mathbf { g } _ { 1 } - \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } { \| \mathbf { g } _ { 1 } + \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } } \end{array}$

With the above definition, we present our next theorem:

Theorem 2. Suppose L is differentiable and the gradient ofL is Lipschitz continuous with constant $L > 0 .$ . Let $\theta ^ { M T }$ and θ<sup>PCGrad</sup> be the parameters after applying one update to θ with g and PCGradmodified gradient $\mathbf { g } ^ { P C }$ respectively, with step size $t > 0 .$ . Moreover, assume $\mathbf { H } ( \mathcal { L } ; \bar { \theta , } \bar { \theta ^ { M T } } ) \geq \ell \| \mathbf { g } \| _ { 2 } ^ { 2 }$ for some constant $\ell \leq L ,$ , i.e. the multi-task curvature is lower-bounded. Then $\dot { \mathcal { L } } ( \theta ^ { P C G r a d } ) \overset { ^ { \prime } } { \leq } \mathcal { L } ( \theta ^ { M T } )$ $\begin{array} { r } { i f ( \pmb { a } ) \cos \phi _ { 1 2 } \leq - \Phi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) , ( b ) \ell \geq \xi \big ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } \big ) L , a n d ( \pmb { c } ) t \geq \frac { 2 } { \ell - \xi \left( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } \right) L } . } \end{array}$

Proof. See Appendix A.2.

![](images/6e0c2a5917271aee7b0045d7812ad9e1be51070392d089120c98a964d02bd414.jpg)

Intuitively, Theorem 2 implies that PCGrad achieves lower loss value after a single gradient update compared to standard gradient descent in multi-task learning when (i) the angle between task gradients is not too small, i.e. the two tasks need to conflict sufficiently (condition (a)), (ii) the difference in magnitude needs to be sufficiently large (condition (a)), (iii) the curvature of the multi-task gradient should be large (condition (b)), (iv) and the learning rate should be big enough so that large curvature would lead to overestimation of performance improvement on the dominating task and underestimation of performance degradation on the dominated task (condition (c)). These first three points (i-iii) correspond to exactly the triad of conditions outlined in Section 2.2, while the latter condition (iv) is desirable as we hope to learn quickly. We empirically validate that the first three points, (i-iii), are frequently met in a neural network multi-task learning problem in Figure 4 in Section 5.3. For additional analysis, including complete sufficient and necessary conditions for the PCGrad update to outperform the vanilla multi-task gradient, see Appendix ${ \mathrm { A } } . 3 .$

## 3 PCGrad in Practice

We use PCGrad in supervised learning and reinforcement learning problems with multiple tasks or goals. Here, we discuss the practical application of PCGrad to those settings.

In multi-task supervised learning, each task $\mathcal { T } _ { i } \sim p ( \mathcal { T } )$ has a corresponding training dataset $\mathcal { D } _ { i }$ consisting of labeled training examples, i.e. $\mathcal { D } _ { i } = \{ ( x , y ) _ { n } \}$ . The objective for each task in this supervised setting is then defined as $\mathcal { L } _ { i } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { i } } \left[ - \log f _ { \theta } ( y | x , z _ { i } ) \right]$ ], where $z _ { i }$ is a one-hot encoding of task $\mathcal { T } _ { i }$ . At each training step, we randomly sample a batch of data points $\boldsymbol { B }$ from the whole dataset $\bar { \bigcup _ { i } } \mathcal { D } _ { i }$ and then group the sampled data with the same task encoding into small batches denoted as $B _ { i }$ for each $\mathcal { T } _ { i }$ represented in B. We denote the set of tasks appearing in B as $B _ { T }$ . After sampling, we precompute the gradient of each task in $B _ { T }$ as $\nabla _ { \theta } \mathcal { L } _ { i } ( \theta ) \stackrel { - } { = } \mathbb { E } _ { ( x , y ) \sim \mathcal { B } _ { i } } \left[ - \nabla _ { \theta } \log f _ { \theta } ( y | \bar { x , } z _ { i } ) \right]$ Given the set of precomputed gradients $\nabla _ { \theta } \mathcal { L } _ { i } ( \theta )$ , we also precompute the cosine similarity between all pairs of the gradients in the set. Using the pre-computed gradients and their similarities, we can obtain the PCGrad update by following Algorithm 1, without re-computing task gradients nor backpropagating into the network. Since the PCGrad procedure is only modifying the gradients of shared parameters in the optimization step, it is model-agnostic and can be applied to any architecture with shared parameters. We empirically validate PCGrad with multiple architectures in Section 5.

For multi-task RL and goal-conditioned RL, PCGrad can be readily applied to policy gradient methods by directly updating the computed policy gradient of each task, following Algorithm 1, analogous to the supervised learning setting. For actor-critic algorithms, it is also straightforward to apply PCGrad: we simply replace task gradients for both the actor and the critic by their gradients computed via PCGrad. For more details on the practical implementation for RL, see Appendix C.

## 4 Related Work

Algorithms for multi-task learning typically consider how to train a single model that can solve a variety of different tasks $[ 6 , 2 , 4 9 ]$ . The multi-task formulation has been applied to many different settings, including supervised learning [63, 35, 60, 53, 62] and reinforcement-learning [17, 58], as well as many different domains, such as vision [3, 39, 31, 33, 62], language [11, 15, 38, 44] and robotics [45, 59, 25]. While multi-task learning has the promise of accelerating acquisition of large task repertoires, in practice it presents a challenging optimization problem, which has been tackled in several ways in prior work.

A number of architectural solutions have been proposed to the multi-task learning problem based on multiple modules or paths [19, 14, 40, 51, 46, 57, 46], or using attention-based architectures [33, 37]. Our work is agnostic to the model architecture and can be combined with prior architectural approaches in a complementary fashion. A different set of multi-task learning approaches aim to decompose the problem into multiple local problems, often corresponding to each task, that are significantly easier to learn, akin to divide and conquer algorithms [32, 50, 42, 56, 21, 13]. Eventually, the local models are combined into a single, multi-task policy using different distillation techniques (outlined in [27, 13]). In contrast to these methods, we propose a simple and cogent scheme for multi-task learning that allows us to learn the tasks simultaneously using a single, shared model without the need for network distillation.

Similarly to our work, a number of prior approaches have observed the difficulty of optimization in multi-task learning [26, 29, 52, 55]. Our work suggests that the challenge in multi-task learning may be attributed to what we describe as the tragic triad of multi-task learning (i.e., conflicting gradients, high curvature, and large gradient differences), which we address directly by introducing a simple and practical algorithm that deconflicts gradients from different tasks. Prior works combat optimization challenges by rescaling task gradients [53, 9]. We alter both the magnitude and direction of the gradient, which we find to be critical for good performance (see Fig. 3). Prior work has also used the cosine similarity between gradients to define when an auxiliary task might be useful [16] or when two tasks are related [55]. We similarly use cosine similarity between gradients to determine if the gradients between a pair of tasks are in conflict. Unlike Du et al. [16], we use this measure for effective multi-task learning, instead of ignoring auxiliary objectives. Overall, we empirically compare our approach to a number of these prior approaches [53, 9, 55], and observe superior performance with PCGrad.

Multiple approaches to continual learning have studied how to prevent gradient updates from adversely affecting previously-learned tasks through various forms of gradient projection [36, 7, 18, 23]. These methods focus on sequential learning settings, and solve for the gradient projections using quadratic programming [36], only project onto the normal plane of the average gradient of past tasks [7], or project the current task gradients onto the orthonormal set of previous task gradients [18]. In contrast, our work focuses on positive transfer when simultaneously learning multiple tasks, does not require solving a QP, and iteratively projects the gradients of each task instead of averaging or only projecting the current task gradient. Finally, our method is distinct from and solves a different problem than the projected gradient method [5], which is an approach for constrained optimization that projects gradients onto the constraint manifold.

## 5 Experiments

The goal of our experiments is to study the following questions: (1) Does PCGrad make the optimiza tion problems easier for various multi-task learning problems including supervised, reinforcement, and goal-conditioned reinforcement learning settings across different task families? (2) Can PCGrad be combined with other multi-task learning approaches to further improve performance? (3) Is the tragic triad of multi-task learning a major factor in making optimization for multi-task learning challenging? To broadly evaluate PCGrad, we consider multi-task supervised learning, multi-task RL, and goal-conditioned RL problems. We include the results on goal-conditioned RL in Appendix F.

During our evaluation, we tune the parameters of the baselines independently, ensuring that all methods were fairly provided with equal model and training capacity. PCGrad inherits the hyperparameters of the respective baseline method in all experiments, and has no additional hyperparameters. For more details on the experimental set-up and model architectures, see Appendix J. The code is available online<sup>1</sup>.

## 5.1 Multi-Task Supervised Learning

To answer question (1) in the supervised learning setting and question (2), we perform experiments on five standard multi-task supervised learning datasets: MultiMNIST, CityScapes, CelebA, multi-task CIFAR-100 and NYUv2. We include the results on MultiMNIST and CityScapes in Appendix E.

<table><tr><td rowspan="3">#P.</td><td rowspan="3">Architecture</td><td rowspan="3">Weighting</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="3">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td rowspan="3"> $\approx 3$ </td><td rowspan="3">Cross-Stitch $^{\ddagger}$ </td><td>Equal Weights</td><td>14.71</td><td>50.23</td><td>0.6481</td><td>0.2871</td><td>33.56</td><td>28.58</td><td>20.08</td><td>40.54</td><td>51.97</td></tr><tr><td>Uncert. Weights*</td><td>15.69</td><td>52.60</td><td>0.6277</td><td>0.2702</td><td>32.69</td><td>27.26</td><td>21.63</td><td>42.84</td><td>54.45</td></tr><tr><td>DWA $^{\dagger}, T = 2$ </td><td>16.11</td><td>53.19</td><td>0.5922</td><td>0.2611</td><td>32.34</td><td>26.91</td><td>21.81</td><td>43.14</td><td>54.92</td></tr><tr><td rowspan="3">1.77</td><td rowspan="3">MTAN $^{\dagger}$ </td><td>Equal Weights</td><td>17.72</td><td>55.32</td><td>0.5906</td><td>0.2577</td><td>31.44</td><td>25.37</td><td>23.17</td><td>45.65</td><td>57.48</td></tr><tr><td>Uncert. Weights*</td><td>17.67</td><td>55.61</td><td>0.5927</td><td>0.2592</td><td>31.25</td><td>25.57</td><td>22.99</td><td>45.83</td><td>57.67</td></tr><tr><td>DWA $^{\dagger}, T = 2$ </td><td>17.15</td><td>54.97</td><td>0.5956</td><td>0.2569</td><td>31.60</td><td>25.46</td><td>22.48</td><td>44.86</td><td>57.24</td></tr><tr><td>1.77</td><td>MTAN $^{\dagger}$ + PCGrad (ours)</td><td>Uncert. Weights*</td><td>20.17</td><td>56.65</td><td>0.5904</td><td>0.2467</td><td>30.01</td><td>24.83</td><td>22.28</td><td>46.12</td><td>58.77</td></tr></table>

Table 1: Three-task learning on the NYUv2 dataset: 13-class semantic segmentation, depth estimation, and surface normal prediction results. #P shows the total number of network parameters. We highlight the best performing combination of multi-task architecture and weighting in bold. The top validation scores for each task are annotated with boxes. The symbols indicate prior methods: <sup>∗</sup>: [28], <sup>†</sup>: [33], <sup>‡</sup>: [40]. Performance of other methods as reported in Liu et al. [33].

For CIFAR-100, we follow Rosenbaum et al. [46] to treat 20 coarse labels in the dataset as distinct tasks, creating a dataset with 20 tasks, with 2500 training instances and 500 test instances per task. We combine PCGrad with a powerful multi-task learning architecture, routing networks [46, 47], by applying PCGrad only to the shared parameters. For the details of this comparison, see Appendix J.1. As shown in Table 2, applying PCGrad to a single network achieves 71% classification accuracy, which outperforms most of the prior methods such as cross-stitch [40] and independent training, sug-

<table><tr><td></td><td>% accuracy</td></tr><tr><td>task specific, 1-fc [46]</td><td>42</td></tr><tr><td>task specific, all-fc [46]</td><td>49</td></tr><tr><td>cross stitch, all-fc [40]</td><td>53</td></tr><tr><td>routing, all-fc + WPL [47]</td><td>74.7</td></tr><tr><td>independent</td><td>67.7</td></tr><tr><td>PCGrad (ours)</td><td>71</td></tr><tr><td>routing-all-fc + WPL + PCGrad (ours)</td><td>77.5</td></tr></table>

Table 2: CIFAR-100 multi-task results. When combined with routing networks, PCGrad leads to a large improvement.

gesting that sharing representations across tasks is conducive for good performance. While routing networks achieve better performance than PCGrad on its own, they are complementary: combining PCGrad with routing networks leads to a 2.8% absolute improvement in test accuracy.

We also aim to use PCGrad to tackle a multi-label classfication problem, which is a commonly used benchmark for multi-task learning. In multi-label classification, given a set of attributes, the model needs to decide whether each attribute describes the input. Hence, it is essentially a binary classification problem for each attribute. We choose the CelebA dataset [34], which consists of 200K face images with 40 attributes. Since for each attribution, it is a binary classfication problem and thus we convert it to a 40-way multi-task learning problem following [53]. We use the same architecture as in [53].

We use the binary classification error averaged across all 40 tasks to evaluate the performance as in [53]. Similar to the MultiMNIST results, we compare PCGrad to Sener and Koltun [53] by rerunning the open-sourced code provided in [53]. As shown in Table 3, PCGrad outperforms Sener and Koltun [53], suggesting that PCGrad is effective in multi-label classification and can also improve multi-task supervised learning performance when the number of tasks is high.

<table><tr><td></td><td>average classification error</td></tr><tr><td>Sener and Koltun [53]</td><td>8.95</td></tr><tr><td>PCGrad (ours)</td><td>8.69</td></tr></table>

Table 3: CelebA results. We show the average classification error across all 40 tasks in CelebA. PCGrad outperforms the prior method Sener and Koltun [53] in this dataset.

Finally, we combine PCGrad with another state-of-art multi-task learning algorithm, MTAN [33], and evaluate the performance on a more challenging indoor scene dataset, NYUv2, which contains 3 tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. We compare MTAN with PCGrad to a list of methods mentioned in Appendix J.1, where each method is trained with three different weighting schemes as in [33], equal weighting, weight uncertainty [28], and DWA [33]. We only run MTAN with PCGrad with weight uncertainty as we find weight uncertainty as the most effective scheme for training MTAN. The results comparing Cross-Stitch, MTAN and MTAN + PCGrad are presented in Table 1 while the full comparison can be found in Table 8 in the Appendix J.4. MTAN with PCGrad is able to achieve the best scores in 8 out of the 9 categories where there are 3 categories per task.

![](images/49e97f7f333fc730726d57812936f1a3473a48d0f486eff8e715b8f4f1a7d077.jpg)

<details>
<summary>line</summary>

| Number of thousand env steps | MT10::SAC | MT10::Multi-head SAC | MT10::Independent | MT10::SAC+PCGrad (ours) | MT10::SAC+PCGrad dir | MT10::SAC+PCGrad mag | MT10::SAC+GradNorm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | ~0.05 | ~0.08 | ~0.02 | ~0.05 | ~0.05 | ~0.02 | ~0.02 |
| 1000 | ~0.12 | ~0.32 | ~0.25 | ~0.65 | ~0.35 | ~0.15 | ~0.12 |
| 2000 | ~0.15 | ~0.32 | ~0.45 | ~0.78 | ~0.42 | ~0.18 | ~0.15 |
| 3000 | ~0.18 | ~0.35 | ~0.65 | ~0.82 | ~0.48 | ~0.22 | ~0.18 |
| 4000 | ~0.20 | ~0.38 | ~0.75 | ~0.85 | ~0.52 | ~0.25 | ~0.22 |
| 5000 | ~0.22 | ~0.42 | ~0.85 | ~0.92 | ~0.58 | ~0.32 | ~0.28 |
</details>

Figure 3: For the two plots on the left, we show learning curves on MT10 and MT50 respectively. PCGrad significantly outperforms the other methods in terms of both success rates and data efficiency. In the rightmost plot, we present the ablation study on only using the magnitude and the direction of gradients modified by PCGrad and a comparison to GradNorm [8]. PCGrad outperforms both ablations and GradNorm, indicating the importance of modifying both the gradient directions and magnitudes in multi-task learning.

Our multi-task supervised learning results indicate that PCGrad can be seamlessly combined with state-of-art multi-task learning architectures and further improve their results on established supervised multi-task learning benchmarks. We include more results of PCGrad combined with more multi-task learning architectures in Appendix I.

## 5.2 Multi-Task Reinforcement Learning

To answer question (2) in the RL setting, we first consider the multi-task RL problem and evaluate our algorithm on the recently proposed Meta-World benchmark [61]. In particular, we test all methods on the MT10 and MT50 benchmarks in Meta-World, which contain 10 and 50 manipulation tasks respectively shown in Figure 10. in Appendix J.2.

The results are shown in left two plots in Figure 3. PCGrad combined with SAC learns all tasks with the best data efficiency and successfully solves all of the 10 tasks in MT10 and about 70% of the 50 tasks in MT50. Training a single SAC policy and a multi-head policy is unable to acquire half of the skills in both MT10 and MT50, suggesting that eliminating gradient interference across tasks can significantly boost performance of multi-task RL. Training independent SAC agents is able to eventually solve all tasks in MT10 and 70% of the tasks in MT50, but requires about 2 millions and 15 millions more samples than PCGrad with SAC in MT10 and MT50 respectively, implying that applying PCGrad can result in leveraging shared structure among tasks that expedites multi-task learning. As noted by Yu et al. [61], these tasks involve distinct behavior motions, which makes learning all tasks with a single policy challenging as demonstrated by poor baseline performance. The ability to learn these tasks together opens the door for a number of interesting extensions to meta-learning and generalization to novel task families.

Since the PCGrad update affects both the gradient direction and the gradient magnitude, we perform an ablation study that tests two variants of PCGrad: (1) only applying the gradient direction corrected with PCGrad while keeping the gradient magnitude unchanged and (2) only applying the gradient magnitude computed by PCGrad while keeping the gradient direction unchanged. We further run a direction comparison to GradNorm [8], which also scales only the magnitudes of the task gradients. As shown in the rightmost plot in Figure 3, both variants and GradNorm perform worse than PCGrad and the variant where we only vary the gradient magnitude is much worse than PCGrad. This emphasizes the importance of the orientation change, which is particularly notable as multiple prior works only alter gradient magnitudes [8, 53]. We also notice that the variant of PCGrad where only the gradient magnitudes change achieves comparable performance to GradNorm, which suggests that it is important to modify both the gradient directions and magnitudes to eliminate interference and achieve good multi-task learning results. Finally, to test the importance of keeping positive cosine similarities between tasks for positive transfer, we compare PCGrad to a recently proposed method in [55] that regularizes cosine similarities of different task gradients towards 0. PCGrad outperforms Suteu and Guo [55] by a large margin. We leave details of the comparison to Appendix G.

![](images/2a211bf97b7ba8a1c5cb247d28bfec7da27210bfbd1fdca42cabf190baef938a.jpg)  
Figure 4: An empirical analysis of the theoretical conditions discussed in Theorem $^ { 2 , }$ showing the first 100 iterations of training on two RL tasks, reach and press button top. Left: The estimated value of the multi-task curvature. We observe high multi-task curvatures exist throughout training, providing evidence for condition (b) in Theorem 2. Middle: The solid lines show the percentage gradients with positive cosine similarity between two task gradients, while the dotted lines and dashed lines show the percentage of iterations in which condition (a) and the implication of condition (b) $( \xi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) \leq 1 )$ in Theorem 2 are held respectively, among iterations when the cosine similarity is negative. Right: The average return of each task achieved by SAC and SAC combined with PCGrad. From the plots in the Middle and on the Right, we can tell that condition (a) holds most of the time for both Adam and Adam combined with PCGrad when they haven’t solved Task 2 and as soon as Adam combined PCGrad starts to learn Task 2, the percentage of condition (a) held starts to decline. This observation suggests that condition (a) is a key factor for PCGrad excelling in multi-task learning.

## 5.3 Empirical Analysis of the Tragic Triad

Finally, to answer question (1), we compare the performance of standard multi-task SAC and the multi-task SAC with PCGrad. We evaluate each method on two tasks, reach and press button top, in the Meta-World [61] benchmark. As shown in the leftmost plot in Figure 4, we plot the multitask curvature, which is computed as $\begin{array} { r } { \mathbf { H } ( \mathcal { L } ; \theta ^ { t } , \theta ^ { t + 1 } ) = 2 \cdot \left[ \mathcal { L } ( \theta ^ { t + 1 } ) - \mathcal { L } ( \theta ^ { t } ) - \smile \lor _ { \theta ^ { t } } \mathcal { L } ( \theta ^ { t } ) ^ { T } ( \theta ^ { t + 1 } - \theta ^ { t } ) \right] } \end{array}$ by Taylor’s Theorem where $\mathcal { L }$ is the multi-task loss, and $\bar { \theta } ^ { t }$ and $\theta ^ { t + 1 }$ are the parameters at iteration t and t + 1. During the training process, the multi-task curvature stays positive and is increasing for both Adam and Adam combined PCGrad, suggesting that condition (b) in Theorem 2 that the multi-task curvature is lower bounded by some positive value is widely held empirically. To further analyze conditions in Theorem 2 empirically, we plot the percentage of condition (a) (i.e. conflicting gradients) and the implication of condition (b) $( \xi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) \leq 1 )$ in Theorem 2 being held among the total number of iterations where the cosine similarity is negative in the plot in the middle of Figure 4. Along with the plot on the right in Figure 4, which presents the average return of the two tasks during training, we can see that while Adam and Adam with PCGrad haven’t received reward signal from Task 2, condition (a) and the implication of condition (b) stay held and as soon as Adam with PCGrad begins to solve Task 2, the percentage of condition (a) and the implication of condition (b) being held start to decrease. Such a pattern suggests that conflicting gradients, high curvatures and dominating gradients indeed produce considerable challenges in optimization before multi-task learner gains any useful learning signal, which also implies that the tragic triad may indeed be the determining factor where PCGrad can lead to better performance gain over standard multi-task learning in practice.

## 6 Conclusion

In this work, we identified a set of conditions that underlies major challenges in multi-task optimiza tion: conflicting gradients, high positive curvature, and large gradient differences. We proposed a simple algorithm (PCGrad) to mitigate these challenges via “gradient surgery.” PCGrad provides a simple solution to mitigating gradient interference, which substantially improves optimization performance. We provide simple didactic examples and subsequently show significant improvement in optimization for a variety of multi-task supervised learning and reinforcement learning problems. We show that, when some optimization challenges of multi-task learning are alleviated by PCGrad, we can obtain hypothesized benefits in efficiency and asymptotic performance of multi-task settings.

While we studied multi-task supervised learning and multi-task reinforcement learning in this work, we suspect the problem of conflicting gradients to be prevalent in a range of other settings and applications, such as meta-learning, continual learning, multi-goal imitation learning [10], and multi-task problems in natural language processing applications [38]. Due to its simplicity and model-agnostic nature, we expect that applying PCGrad in these domains to be a promising avenue for future investigation. Further, the general idea of gradient surgery may be an important ingredient for alleviating a broader class of optimization challenges in deep learning, such as the challenges in the stability challenges in two-player games [48] and multi-agent optimizations [41]. We believe this work to be a step towards simple yet general techniques for addressing some of these challenges.

## Broader Impact

Applications and Benefits. Despite recent success, current deep learning and deep RL methods mostly focus on tackling a single specific task from scratch. Prior methods have proposed methods that can perform multiple tasks, but they often yield comparable or even higher data complexity compared to learning each task individually. Our method enables deep learning systems that mitigate inferences between differing tasks and thus achieves data-efficient multi-task learning. Since our method is general and simple to apply to various problems, there are many possible real-world applications, including but not limited to computer vision systems, autonomous driving, and robotics. For computer vision systems, our method can be used to develop algorithms that enable efficient classification, instance and semantics segmentation and object detection at the same time, which could improve performances of computer vision systems by reusing features obtained from each task and lead to a leap in real-world domains such as autonomous driving. For robotics, there are many situations where multi-task learning is needed. For example, surgical robots are required to perform a wide range of tasks such as stitching and removing tumour from the patient’s body. Kitchen robots should be able to complete multiple chores such as cooking and washing dishes at the same time. Hence, our work represents a step towards making multi-task reinforcement learning more applicable to those settings.

Risks. However, there are potential risks that apply to all machine learning and reinforcement learning systems including ours, including but not limited to safety, reward specification in RL which is often difficult to acquire in the real world, bias in supervised learning systems due to the composition of training data, and compute/data-intensive training procedures. For example, safety issues arise when autonomous driving cars fail to generalize to out-of-distribution data, which leads to crashing or even hurting people. Moreover, reward specification in RL is generally inaccessible in the real world, making RL unable to scale to real robots. In supervised learning domains, learned models could inherit the bias that exists in the training dataset. Furthermore, training procedures of ML models are generally compute/data-intensive, which cause inequitable access to these models. Our method is not immune to these risks. Hence, we encourage future research to design more robust and safe multi-task RL algorithms that can prevent unsafe behaviors. It is also important to push research in self-supervised and unsupervised multi-task RL in order to resolve the issue of reward specification. For supervised learning, we recommend researchers to publish their trained multi-task learning models to make access to those models equitable to everyone in field and develop new datasets that can mitigate biases and also be readily used in multi-task learning.

## Acknowledgments and Disclosure of Funding

The authors would like to thank Annie Xie for reviewing an earlier draft of the paper, Eric Mitchell for technical guidance, and Aravind Rajeswaran and Deirdre Quillen for helpful discussions. Tianhe Yu is partially supported by Intel Corporation. Saurabh Kumar is supported by an NSF Graduate Research Fellowship and the Stanford Knight Hennessy Fellowship. Abhishek Gupta is supported by an NSF Graduate Research Fellowship. Chelsea Finn is a CIFAR Fellow in the Learning in Machines and Brains program.

## References

[1] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12), 2017.  
[2] Bart Bakker and Tom Heskes. Task clustering and gating for bayesian multitask learning. J. Mach. Learn. Res., 2003.  
[3] Hakan Bilen and Andrea Vedaldi. Integrated perception with recurrent multi-task neural networks. In Advances in Neural Information Processing Systems, 2016.  
[4] Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.  
[5] Paul H Calamai and Jorge J More. Projected gradient methods for linearly constrained problems.´ Mathematical programming, 39(1), 1987.  
[6] Rich Caruana. Multitask learning. Machine Learning, 1997.  
[7] Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv:1812.00420, 2018.  
[8] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. arXiv:1711.02257, 2017.  
[9] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, 2018.  
[10] Felipe Codevilla, Matthias Miiller, Antonio Lopez, Vladlen Koltun, and Alexey Dosovitskiy.´ End-to-end driving via conditional imitation learning. In International Conference on Robotics and Automation (ICRA), 2018.  
[11] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In International Conference on Machine Learning, 2008.  
[12] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.  
[13] Wojciech M. Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant M. Jayakumar, Grzegorz Swirszcz, and Max Jaderberg. Distilling policy distillation. In International Conference on Artificial Intelligence and Statistics, AISTATS, 2019.  
[14] Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. CoRR, abs/1609.07088, 2016.  
[15] Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. Multi-task learning for multi ple language translation. In Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2015.  
[16] Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting auxiliary losses using gradient similarity. CoRR, abs/1812.02224, 2018.  
[17] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward,´ Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: scalable distributed deep-rl with importance weighted actor-learner architectures. In International Conference on Machine Learning, 2018.  
[18] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. arXiv preprint arXiv:1910.07104, 2019.  
[19] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A. Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. CoRR, abs/1701.08734, 2017. URL http://arxiv.org/abs/1701. 08734.  
[20] William Fulton. Eigenvalues, invariant factors, highest weights, and schubert calculus. Bulletin ofthe American Mathematical Society, 37(3):209–249, 2000.  
[21] Dibya Ghosh, Avi Singh, Aravind Rajeswaran, Vikash Kumar, and Sergey Levine. Divide-and conquer reinforcement learning. CoRR, abs/1711.09874, 2017.  
[22] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. arXiv:1412.6544, 2014.  
[23] Yunhui Guo, Mingrui Liu, Tianbao Yang, and Tajana Rosing. Improved schemes for episodic memory-based lifelong learning, 2020.  
[24] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. International Conference on Machine Learning, 2018.  
[25] Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller. Learning an embedding space for transferable robot skills. 2018.  
[26] Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado van Hasselt. Multi-task deep reinforcement learning with popart. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 2019.  
[27] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.  
[28] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Computer Vision and Pattern Recognition, 2018.  
[29] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, 2018.  
[30] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv:1412.6980, 2014.  
[31] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In Computer Vision and Pattern Recognition, 2017.  
[32] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal ofMachine Learning Research, 17(1), 2016.  
[33] Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. CoRR, abs/1803.10704, 2018.  
[34] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.  
[35] Mingsheng Long and Jianmin Wang. Learning multiple tasks with deep relationship networks. arXiv:1506.02117, 2, 2015.  
[36] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems, 2017.  
[37] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. CoRR, abs/1904.08918, 2019.  
[38] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. arXiv:1806.08730, 2018.  
[39] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Computer Vision and Pattern Recognition, 2016.  
[40] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Conference on Computer Vision and Pattern Recognition, CVPR, 2016.  
[41] Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control, 54(1), 2009.  
[42] Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. arXiv:1511.06342, 2015.  
[43] Boris T Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, 1964.  
[44] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 2019.  
[45] Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave, Tom Van de Wiele, Volodymyr Mnih, Nicolas Heess, and Jost Tobias Springenberg. Learning by playing-solving sparse reward tasks from scratch. arXiv:1802.10567, 2018.  
[46] Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. International Conference on Learning Representations (ICLR), 2018.  
[47] Clemens Rosenbaum, Ignacio Cases, Matthew Riemer, and Tim Klinger. Routing networks and the challenges of modular and compositional computation. arXiv:1904.12774, 2019.  
[48] Kevin Roth, Aurelien Lucchi, Sebastian Nowozin, and Thomas Hofmann. Stabilizing training of generative adversarial networks through regularization. In Advances in Neural Information Processing Systems, 2017.  
[49] Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv:1706.05098, 2017.  
[50] Andrei A. Rusu, Sergio Gomez Colmenarejo, C¸ aglar Gul¨ c¸ehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. In International Conference on Learning Representations, ICLR, 2016.  
[51] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv:1606.04671, 2016.  
[52] Tom Schaul, Diana Borsa, Joseph Modayil, and Razvan Pascanu. Ray interference: a source of plateaus in deep reinforcement learning. arXiv:1904.11455, 2019.  
[53] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, 2018.  
[54] Yuekai Sun. Notes on first-order methods for minimizing smooth functions, 2015. https://web.stanford.edu/class/msande318/notes/ notes-first-order-smooth.pdf.  
[55] Mihai Suteu and Yike Guo. Regularizing deep multi-task networks using orthogonal gradients. arXiv preprint arXiv:1912.06844, 2019.  
[56] Yee Teh, Victor Bapst, Wojciech M Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In Advances in Neural Information Processing Systems, 2017.  
[57] Simon Vandenhende, Bert De Brabandere, and Luc Van Gool. Branched multi-task networks: Deciding what layers to share. CoRR, abs/1904.02920, 2019.  
[58] Aaron Wilson, Alan Fern, Soumya Ray, and Prasad Tadepalli. Multi-task reinforcement learning: a hierarchical bayesian approach. In International Conference on Machine Learning, 2007.  
[59] Markus Wulfmeier, Abbas Abdolmaleki, Roland Hafner, Jost Tobias Springenberg, Michael Neunert, Tim Hertweck, Thomas Lampe, Noah Siegel, Nicolas Heess, and Martin Riedmiller. Regularized hierarchical policies for compositional transfer in robotics. arXiv:1906.11228, 2019.  
[60] Yongxin Yang and Timothy M Hospedales. Trace norm regularised deep multi-task learning. arXiv:1606.04038, 2016.  
[61] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta-reinforcement learning. arXiv:1910.10897, 2019.  
[62] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Computer Vision and Pattern Recognition, 2018.  
[63] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Facial landmark detection by deep multi-task learning. In European conference on computer vision. Springer, 2014.

## Appendix

## A Proofs

## A.1 Proof of Theorem 1

Theorem 1. Assume $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are convex and differentiable. Suppose the gradient of L is $L \mathrm { - }$ Lipschitz with $L > 0 .$ . Then, the PCGrad update rule with step size $\begin{array} { r } { t \le \frac { 1 } { L } } \end{array}$ will converge to either (1) a location in the optimization landscape where cos $\left( \phi _ { 1 2 } \right) = - 1 o r \left( 2 \right)$ the optimal value $\mathcal { L } ( \theta ^ { \ast } )$ .

Proof. We will use the shorthand $| | \cdot | |$ to denote the $L _ { \mathrm { { 2 } } } \mathrm { { - n o r m } }$ and $\nabla { \mathcal { L } } = \nabla _ { \theta } { \mathcal { L } } ,$ , where θ is the parameter vector. Following Definition 1 and 4, let $\mathbf { g _ { 1 } } = \nabla \mathcal { L } _ { 1 } , \mathbf { g _ { 2 } } = \nabla \mathcal { L } _ { 2 } , \mathbf { g } = \nabla \mathcal { L } = \mathbf { g _ { 1 } } + \mathbf { g _ { 2 } }$ and $\phi _ { 1 2 }$ be the angle between g<sub>1</sub> and $\mathbf { g _ { 2 } }$

At each PCGrad update, we have two cases: $c o s ( \phi _ { 1 2 } ) \ge 0 \mathrm { o r } \mathrm { c o s } ( \phi _ { 1 2 } ) < 0 $

If $\cos ( \phi _ { 1 2 } ) \geq 0$ , then we apply the standard gradient descent update using $\begin{array} { r } { t \le \frac { 1 } { L } } \end{array}$ , which leads to a strict decrease in the objective function value $\bar { \mathcal { L } } ( \boldsymbol { \theta } )$ (since it is also convex) unless $\begin{array} { r } { \nabla \mathcal { L } ( \theta ) = 0 } \end{array}$ , which occurs only when $\theta = \theta ^ { * } \left[ 4 \right]$

In the case that $\cos ( \phi _ { 1 2 } ) < 0$ , we proceed as follows:

Our assumption that $\nabla \mathcal { L }$ is Lipschitz continuous with constant L implies that $\nabla ^ { 2 } \mathcal { L } ( \theta ) - L I$ is a negative semi-definite matrix. Using this fact, we can perform a quadratic expansion of L around $\mathcal { L } ( \bar { \boldsymbol { \theta } } )$ and obtain the following inequality:

$$
\begin{array}{l} \mathcal {L} (\theta^ {+}) \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {T} (\theta^ {+} - \theta) + \frac {1}{2} \nabla^ {2} \mathcal {L} (\theta) | | \theta^ {+} - \theta | | ^ {2} \\ \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {T} (\theta^ {+} - \theta) + \frac {1}{2} L | | \theta^ {+} - \theta | | ^ {2} \\ \end{array}
$$

Now, we can plug in the PCGrad update by letting $\begin{array} { r } { \theta ^ { + } = \theta - t \cdot ( { \bf g } - \frac { { \bf g } _ { 1 } \cdot { \bf g } _ { 2 } } { | | { \bf g } _ { 1 } | | ^ { 2 } } { \bf g } _ { 1 } - \frac { { \bf g } _ { 1 } \cdot { \bf g } _ { 2 } } { | | { \bf g } _ { 2 } | | ^ { 2 } } { \bf g } _ { 2 } ) } \end{array}$ . We then get:

$$
\mathcal {L} (\theta^ {+}) \leq \mathcal {L} (\theta) + t \cdot \mathbf {g} ^ {T} (- \mathbf {g} + \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{| | \mathbf {g _ {1}} | | ^ {2}} \mathbf {g _ {1}} + \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{| | \mathbf {g _ {2}} | | ^ {2}} \mathbf {g _ {2}}) + \frac {1}{2} L t ^ {2} \| \mathbf {g} - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{| | \mathbf {g _ {1}} | | ^ {2}} \mathbf {g _ {1}} - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{| | \mathbf {g _ {2}} | | ^ {2}} \mathbf {g _ {2}} \| ^ {2}
$$

(Expanding, using the identity $\bf { g } = \bf { g } _ { 1 } + \bf { g } _ { 2 } )$

$$
\begin{array}{l} = \mathcal {L} (\theta) + t \left(- | | \mathbf {g _ {1}} | | ^ {2} - | | \mathbf {g _ {2}} | | ^ {2} + \frac {(\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) ^ {2}}{| | \mathbf {g _ {1}} | | ^ {2}} + \frac {(\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) ^ {2}}{| | \mathbf {g _ {2}} | | ^ {2}}\right) + \frac {1}{2} L t ^ {2} | | \mathbf {g _ {1}} + \mathbf {g _ {2}} \\ - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\left| \left| \mathbf {g _ {1}} \right| \right| ^ {2}} \mathbf {g _ {1}} - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\left| \left| \mathbf {g _ {2}} \right| \right| ^ {2}} \mathbf {g _ {2}} \| ^ {2} \\ \end{array}
$$

(Expanding further and re-arranging terms)

$$
\begin{array}{l} = \mathcal {L} (\theta) - (t - \frac {1}{2} L t ^ {2}) (| | \mathbf {g _ {1}} | | ^ {2} + | | \mathbf {g _ {2}} | | ^ {2} - \frac {(\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) ^ {2}}{| | \mathbf {g _ {1}} | | ^ {2}} - \frac {(\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) ^ {2}}{| | \mathbf {g _ {2}} | | ^ {2}}) \\ - L t ^ {2} (\mathbf {g _ {1}} \cdot \mathbf {g _ {2}} - \frac {(\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) ^ {2}}{| | \mathbf {g _ {1}} | | ^ {2} | | \mathbf {g _ {2}} | | ^ {2}} \mathbf {g _ {1}} \cdot \mathbf {g _ {2}}) \\ \end{array}
$$

(Using the identity $\cos ( \phi _ { 1 2 } ) = \frac { \bf { g } _ { 1 } \cdot \bf { g } _ { 2 } } { | | \bf { g } _ { 1 } | | | \bf { g } _ { 2 } | | } )$

$$
\begin{array}{l} = \mathcal {L} (\theta) - (t - \frac {1}{2} L t ^ {2}) [ (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {1}} | | ^ {2} + (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {2}} | | ^ {2} ] \\ - L t ^ {2} (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {1}} | | | | \mathbf {g _ {2}} | | \cos (\phi_ {1 2}) \tag {1} \\ \end{array}
$$

(Note that $\cos ( \phi _ { 1 2 } ) < 0$ so the final term is non-negative)

Using $\begin{array} { r } { t \le \frac { 1 } { L } } \end{array}$ , we know that $\begin{array} { r } { - ( 1 - \frac 1 2 L t ) = \frac 1 2 L t - 1 \le \frac 1 2 L ( 1 / L ) - 1 = \frac { - 1 } { 2 } } \end{array}$ and $L t ^ { 2 } \leq t .$

Plugging this into the last expression above, we can conclude the following:

$$
\begin{array}{l} \mathcal {L} (\theta^ {+}) \leq \mathcal {L} (\theta) - \frac {1}{2} t [ (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {1}} | | ^ {2} + (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {2}} | | ^ {2} ] \\ - t (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {1}} | | | | \mathbf {g _ {2}} | | \cos (\phi_ {1 2}) \\ = \mathcal {L} (\theta) - \frac {1}{2} t (1 - \cos^ {2} (\phi_ {1 2})) [ | | \mathbf {g _ {1}} | | ^ {2} + 2 | | \mathbf {g _ {1}} | | | | \mathbf {g _ {2}} | | \cos (\phi_ {1 2}) + | | \mathbf {g _ {2}} | | ^ {2} ] \\ = \mathcal {L} (\theta) - \frac {1}{2} t (1 - \cos^ {2} (\phi_ {1 2})) [ | | \mathbf {g _ {1}} | | ^ {2} + 2 \mathbf {g _ {1}} \cdot \mathbf {g _ {2}} + | | \mathbf {g _ {2}} | | ^ {2} ] \\ = \mathcal {L} (\theta) - \frac {1}{2} t (1 - \cos^ {2} (\phi_ {1 2})) | | \mathbf {g _ {1}} + \mathbf {g _ {2}} | | ^ {2} \\ = \mathcal {L} (\theta) - \frac {1}{2} t (1 - \cos^ {2} (\phi_ {1 2})) \| \mathbf {g} \| ^ {2} \\ \end{array}
$$

I $\dot { \cdot } \cos ( \phi _ { 1 2 } ) > - 1$ , then $\begin{array} { r } { \frac { 1 } { \mathcal { 1 } } t ( 1 - \cos ^ { 2 } ( \phi _ { 1 2 } ) ) \lVert \mathbf { g } \rVert ^ { 2 } } \end{array}$ will always be positive unless $\mathbf g = 0 $ . This inequality implies that the objective function value strictly decreases with each iteration where cos $\left( \phi _ { 1 2 } \right) > - 1$

Hence repeatedly applying PCGrad process can either reach the optimal value $\mathcal { L } ( \theta ) = \mathcal { L } ( \theta ^ { \ast } )$ or cos $\left( \phi _ { 1 2 } \right) = - 1$ , in which case $\begin{array} { r } { \frac { 1 } { 2 } t ( 1 - \cos ^ { 2 } ( \phi _ { 1 2 } ) ) \lVert \mathbf { g } \rVert ^ { 2 } = 0 } \end{array}$ . Note that this result only holds when we choose t to be small enough, i.e. $\begin{array} { r } { t \le \frac { 1 } { L } } \end{array}$

![](images/b905b1fa6592dc0883990b7eb68f7013395fafcbd9cd0d35e4b8a4317ba2b30a.jpg)

Corollary 1. Assume the n objectives $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 } , . . . , \mathcal { L } _ { n }$ are convex and differentiable. Suppose the gradient of L is Lipschitz continuous with constant $L > 0 .$ . Assume that cos $\begin{array} { r } { ( \mathbf { g } , \mathbf { g } ^ { P C } ) \geq \frac { 1 } { 2 } } \end{array}$ . Then, the PCGrad update rule with step size $\begin{array} { r } { t \leq \frac { 1 } { I , } } \end{array}$ will converge to either (1) a location in the optimization landscape where cos $\left( \mathbf { g } _ { i } , \mathbf { g } _ { j } \right) = - 1 \forall i , j \stackrel { \sim } { o r } \left( 2 \right)$ the optimal value $\mathcal { L } ( \theta ^ { \ast } )$

Proof. Our assumption that $\nabla \mathcal { L }$ is Lipschitz continuous with constant L implies that $\nabla ^ { 2 } \mathcal { L } ( \theta ) - L I$ is a negative semi-definite matrix. Using this fact, we can perform a quadratic expansion of L around $\mathcal { L } ( \boldsymbol { \theta } )$ and obtain the following inequality:

$$
\begin{array}{l} \mathcal {L} (\theta^ {+}) \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {T} (\theta^ {+} - \theta) + \frac {1}{2} \nabla^ {2} \mathcal {L} (\theta) | | \theta^ {+} - \theta | | ^ {2} \\ \leq \mathcal {L} (\theta) + \nabla \mathcal {L} (\theta) ^ {T} (\theta^ {+} - \theta) + \frac {1}{2} L | | \theta^ {+} - \theta | | ^ {2} \\ \end{array}
$$

Now, we can plug in the PCGrad update by letting $\mathbf { \nabla } \theta ^ { + } = \theta - t \cdot \mathbf { g } ^ { \mathrm { P C } }$ . We then get:

$$
\mathcal {L} (\boldsymbol {\theta} ^ {+}) \leq \mathcal {L} (\boldsymbol {\theta}) - t \cdot \mathbf {g} ^ {T} \mathbf {g} ^ {\mathrm{PC}} + \frac {1}{2} L t ^ {2} | | \mathbf {g} ^ {\mathrm{PC}} | | ^ {2}
$$

(Using the assumption that $\cos ( \mathbf { g } , \mathbf { g } ^ { \mathrm { P C } } ) \geq \frac { 1 } { 2 } . )$

$$
\begin{array}{l} \leq \mathcal {L} (\theta) - \frac {1}{2} t | | \mathbf {g} | | \cdot | | \mathbf {g} ^ {\mathrm{PC}} | | + \frac {1}{2} L t ^ {2} | | \mathbf {g} ^ {\mathrm{PC}} | | ^ {2} \\ \leq \mathcal {L} (\theta) - \frac {1}{2} t | | \mathbf {g} | | \cdot | | \mathbf {g} ^ {\mathrm{PC}} | | + \frac {1}{2} L t ^ {2} | | \mathbf {g} ^ {\mathrm{PC}} | | \cdot | | \mathbf {g} |\| \\ \end{array}
$$

Note that $- \textstyle { \frac { 1 } { 2 } } t | | \mathbf { g } | | \cdot | | \mathbf { g } ^ { \mathrm { P C } } | | + \frac { 1 } { 2 } L t ^ { 2 } | | \mathbf { g } ^ { \mathrm { P C } } | | \cdot | | \mathbf { g } | | \leq 0$ when $\begin{array} { r } { t \leq \frac { 1 } { I } } \end{array}$ . Further, when $\begin{array} { r } { t < \frac { 1 } { L } , - \frac { 1 } { 2 } t | | \mathbf { g } | | } \end{array}$ $| | \mathbf { g } ^ { \mathrm { P C } } | | + \frac { 1 } { 2 } \bar { L t } ^ { 2 } | | \mathbf { g } ^ { \mathrm { P C } } | | \cdot | | \mathbf { g } | | = \bar { 0 }$ if and only $\mathrm { i f } | | \mathbf { g } | | = 0 \mathrm { o r } | | \mathbf { g } ^ { \mathrm { P C } } | | = 0$

Hence repeatedly applying PCGrad process can either reach the optimal value $\mathcal { L } ( \theta ) = \mathcal { L } ( \theta ^ { \ast } )$ or a location in the optimization landscape where $\cos ( \mathbf { g } _ { i } , \mathbf { g } _ { j } ) = - 1$ for all pairs of tasks $i , j$ . Note that this result only holds when we choose t to be small enough, i.e. $\begin{array} { r } { t \le \frac { 1 } { L } } \end{array}$ □

Proposition 1. Assume $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are differentiable but possibly non-convex. Suppose the gradient ofL is Lipschitz continuous with constant $\dot { L } > 0 .$ . Then, the PCGrad update rule with step size $\textstyle t \leq { \frac { 1 } { I } }$ will converge to either (1) a location in the optimization landscape where co $; ( \phi _ { 1 2 } ) = - 1 o r ( 2 ) f i n \ ' d$ a $\theta _ { k }$ that is almost a stationary point.

Proof. Following Definition 1 and 4, let $\mathbf { g _ { 1 } } _ { k } = \nabla \mathcal { L } _ { 1 }$ at iteration k, $\mathbf { g _ { 2 } } _ { k } = \nabla \mathcal { L } _ { 2 }$ at iteration k, and ${ \bf g } _ { k } = \nabla \mathcal { L } = { \bf g _ { 1 } } _ { k } + { \bf g _ { 2 } } _ { k }$ at iteration $k ,$ and $\phi _ { 1 2 , k }$ be the angle between $\mathbf { g _ { 1 } } _ { k }$ and $\mathbf { g _ { 2 \it k } }$

From the proof of Theorem 1, when $\cos ( \phi _ { 1 2 } , k ) < 0$ we have:

$$
| | \mathbf {g} _ {k} | | ^ {2} \leq \frac {2}{t} \frac {\mathcal {L} (\theta_ {k - 1}) - \mathcal {L} (\theta_ {k})}{(1 - \cos^ {2} (\phi_ {1 2 , k}))}.
$$

Thus, we have:

$$
\begin{array}{l} \min _ {0 \leq k \leq K} | | \mathbf {g} _ {k} | | ^ {2} \leq \frac {1}{K} \sum_ {i = 0} ^ {K - 1} | | \mathbf {g} _ {i} | | ^ {2} \\ \leq \frac {2}{K t} \sum_ {i = 0} ^ {K - 1} \frac {\mathcal {L} (\theta_ {i - 1}) - \mathcal {L} (\theta_ {i})}{(1 - \cos^ {2} (\phi_ {1 2 , i}))} \\ \end{array}
$$

If at any iteration, cos $\left( \phi _ { 1 2 , k } \right) = - 1$ , then the optimization will stop at that point. $\mathrm { I f } \forall k \in [ 0 , K ] .$ cos $\left( \phi _ { 1 2 , k } \right) \ge \alpha > - 1$ , then, we have:

$$
\begin{array}{l} \min _ {0 \leq k \leq K} | | \mathbf {g} _ {k} | | ^ {2} \leq \frac {2}{K (1 - \alpha^ {2}) t} \sum_ {i = 0} ^ {K - 1} (\mathcal {L} (\theta_ {i - 1}) - \mathcal {L} (\theta_ {i})) \\ = \frac {2}{K (1 - \alpha^ {2}) t} (\mathcal {L} (\theta_ {0}) - \mathcal {L} (\theta_ {K})) \\ \leq \frac {2}{K (1 - \alpha^ {2}) t} (\mathcal {L} (\theta_ {0}) - \mathcal {L} ^ {*}). \\ \end{array}
$$

where $\mathcal { L } ^ { \ast }$ is the minimal function value.

![](images/d7899fe006048a87ee6e5c2657a05af45512b10a97e01581636d2a3a0c4e7fac.jpg)

Note that the convergence rate of PCGrad in the non-convex setting largely depends on the value of α and generally how small $\cos ( \phi _ { 1 2 , k } )$ is on average.

## A.2 Proof of Theorem 2

Theorem 2. Suppose L is differentiable and the gradient ofL is Lipschitz continuous with constant $L > 0 . \ L e t \theta ^ { M T }$ and θ<sup>PCGrad</sup> be the parameters after applying one update to θ with g and $P C G r a d \cdot$ modified gradient $\mathbf { g } ^ { P C }$ respectively, with step size $t > 0 .$ . Moreover, assume $\mathbf { H } ( \mathcal { L } ; \boldsymbol { \theta } , \boldsymbol { \theta } ^ { M T } ) \geq \ell \| \mathbf { g } \| _ { 2 } ^ { 2 }$ for some constant $\ell \leq L ,$ , i.e. the multi-task curvature is lower-bounded. Then $\dot { \mathcal { L } } ( \theta ^ { P C G r a d } ) \overset { \prime } { \leq } \mathcal { L } ( \tilde { \theta } ^ { M T } )$ if(a) cos $\begin{array} { r } { \phi _ { 1 2 } \le - \Phi ( \mathbf { g _ { 1 } } , \mathbf { g _ { 2 } } ) , ( \pmb { b } ) \ell \ge \xi ( \mathbf { g _ { 1 } } , \mathbf { g _ { 2 } } ) L , } \end{array}$ and $\begin{array} { r } { \mathbf { \tilde { \rho } } ( \mathbf { \tilde { \rho } } ) \ t \geq \frac { 2 } { \ell - \xi ( \mathbf { g _ { 1 } } , \mathbf { g _ { 2 } } ) L } . } \end{array}$

Proof. Note that $\theta ^ { \mathrm { M T } } = \theta - t \cdot \mathbf { g }$ and $\begin{array} { r } { \theta ^ { \mathrm { P C G r a d } } = \theta - t ( \mathbf { g } - \frac { \mathbf { g } _ { 1 } \cdot \mathbf { g } _ { 2 } } { | | \mathbf { g } _ { 1 } | | ^ { 2 } } \mathbf { g _ { 1 } } - \frac { \mathbf { g } _ { 1 } \cdot \mathbf { g } _ { 2 } } { | | \mathbf { g } _ { 2 } | | ^ { 2 } } \mathbf { g _ { 2 } } ) } \end{array}$ . Based on the condition that $\mathbf { H } ( { \mathcal { L } } ; \theta , \theta ^ { \mathrm { M T } } ) \geq \ell \| \mathbf { g } \| _ { 2 } ^ { 2 }$ , we first apply Taylor’s Theorem to ${ \mathcal { L } } ( \theta ^ { \mathrm { M T } } )$ and obtain the following result:

$$
\begin{array}{l} \mathcal {L} (\theta^ {\mathrm{MT}}) = \mathcal {L} (\theta) + \mathbf {g} ^ {T} (- t \mathbf {g}) + \int_ {0} ^ {1} (- t \mathbf {g}) ^ {T} \frac {\nabla^ {2} \mathcal {L} (\theta + a \cdot (- t \mathbf {g}))}{2} (- t \mathbf {g}) d a \\ \geq \mathcal {L} (\theta) + \mathbf {g} ^ {T} (- t \mathbf {g}) + t ^ {2} \cdot \frac {1}{2} \ell \cdot \| \mathbf {g} \| _ {2} ^ {2} \\ = \mathcal {L} (\theta) - t \| \mathbf {g} \| _ {2} ^ {2} + \frac {1}{2} \ell t ^ {2} \| \mathbf {g} \| _ {2} ^ {2} \\ = \mathcal {L} (\theta) + (\frac {1}{2} \ell t ^ {2} - t) \| \mathbf {g} \| _ {2} ^ {2} \tag {2} \\ \end{array}
$$

where the first inequality follows from Definition 3 and the assumption $\mathbf { H } ( \mathcal { L } ; \boldsymbol { \theta } , \boldsymbol { \theta } ^ { \mathrm { M T } } ) \geq \ell \| \mathbf { g } \| _ { 2 } ^ { 2 }$ . From equation 1, we have the simplified upper bound for ${ \mathcal { L } } ( \theta ^ { \mathrm { P C G r a d } } )$ :

$$
\mathcal {L} \left(\theta^ {\mathrm{PCGrad}}\right) \leq \mathcal {L} (\theta) - \left(1 - \cos^ {2} \phi_ {1 2}\right) \left[ \left(t - \frac {1}{2} L t ^ {2}\right) \cdot \left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {1}} \| _ {2} ^ {2}\right) + L t ^ {2} \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} \right] \tag {3}
$$

Apply Equation 2 and Equation 3 and we have the following inequality:

$$
\begin{array}{l} \mathcal {L} (\theta^ {\mathrm{MT}}) - \mathcal {L} (\theta^ {\mathrm{PCGrad}}) \geq \mathcal {L} (\theta) + (\frac {1}{2} \ell t ^ {2} - t) \| \mathbf {g} \| _ {2} ^ {2} - \mathcal {L} (\theta) \\ + (1 - \cos^ {2} \phi_ {1 2}) [ (t - \frac {1}{2} L t ^ {2}) (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) + L t ^ {2} \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} ] \\ = (\frac {1}{2} \ell t ^ {2} - t) \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} + (1 - \cos^ {2} \phi_ {1 2}) \biggl [ (t - \frac {1}{2} L t ^ {2}) \cdot (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) + L t ^ {2} \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} \biggr ] \\ = \left(\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2} - 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}) L\right) t ^ {2} \\ - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) t \\ = \left(\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2} L\right) t ^ {2} \\ \left. - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) t \right. \\ = t \cdot \left[ \left(\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L\right) t \right. \\ \left. - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \tag {4} \\ \end{array}
$$

Since cos $\begin{array} { r } { \phi _ { 1 2 } \le - \Phi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) = - \frac { 2 \| \mathbf { g } _ { 1 } \| _ { 2 } \| \mathbf { g } _ { 2 } \| _ { 2 } } { \| \mathbf { g } _ { 1 } \| _ { 2 } ^ { 2 } + \| \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } \mathrm { ~ a n d ~ } \ell \ge \xi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) = \frac { ( 1 - \cos ^ { 2 } \phi _ { 1 2 } ) ( \| \mathbf { g } _ { 1 } - \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } ) } { \| \mathbf { g } _ { 1 } + \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } L , } \end{array}$ we have

$$
\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2} L \geq 0
$$

and

$$
(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} \geq 0.
$$

By the condition that $\begin{array} { r } { t \geq \frac { 2 } { \ell - \xi ( \mathbf { g } _ { 1 } , \mathbf { g } _ { 2 } ) L } = \frac { 2 } { \ell - \frac { ( 1 - \cos ^ { 2 } \phi _ { 1 2 } ) \| \mathbf { g } _ { 1 } - \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } { \| \mathbf { g } _ { 1 } + \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } } L } } \end{array}$ and monotonicity of linear functions, we have the following:

$$
\begin{array}{l} \mathcal {L} (\theta^ {\mathrm{MT}}) - \mathcal {L} (\theta^ {\mathrm{PCGrad}}) \geq [ \left(\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} \cdot \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2} L\right) \cdot \frac {2}{\ell - \frac {(1 - \cos^ {2} \phi_ {1 2}) \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}}{\| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2}} L} \\ \left. - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \cdot t \\ = [ \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \cdot \left(\ell - \frac {(1 - \cos^ {2} \phi_ {1 2}) \cdot \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}}{\| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2}}) L\right) \cdot \frac {1}{\ell - \frac {(1 - \cos^ {2} \phi_ {1 2}) \| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}}{\| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2}} L} \\ \left. - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \cdot t \\ = \left[ \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \cdot t \\ = \left[ \| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \cdot t \\ = (1 - \cos^ {2} \phi_ {1 2}) (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cdot t \\ \geq 0 \\ \end{array}
$$

![](images/2aa7a0cc4b3f11c375c5878485e61df82884457fd0d1a19a726c4dd715b4c023.jpg)

## A.3 PCGrad: Sufficient and Necessary Conditions for Loss Improvement

Beyond the sufficient conditions shown in Theorem 2, we also present the sufficient and necessary conditions under which PCGrad achieves lower loss after one gradient update in Theorem 3 in the two-task setting.

Theorem 3. Suppose L is differentiable and the gradient ofL is Lipschitz continuous with constant $L > 0 . \ L e t \theta ^ { M T }$ and θ<sup>PCGrad</sup> be the parameters after applying one update to θ with g and PCGradmodified gradient $\mathbf { g } ^ { P C }$ respectively, with step size $t > 0 .$ Moreover, assume $\mathbf { H } ( \mathcal { L } ; \boldsymbol { \theta } , \boldsymbol { \theta } ^ { M T } ) \ge \ell \| \mathbf { g } \| _ { 2 } ^ { 2 }$ for some constant $\ell \leq L ,$ , i.e. the multi-task curvature is lower-bounded. Then $\dot { \mathcal { L } } ( \theta ^ { P C G r a d } ) \overset { \prime } { \leq } \mathcal { L } ( \tilde { \theta ^ { M T } } )$ if and only if

• −Φ(g<sub>1</sub>, g<sub>2</sub>) ≤ cos φ<sub>12</sub> < 0  
• \` ≤ ξ(g<sub>1</sub>, g<sub>2</sub>)L  
• 0 < t ≤ (kg<sub>1</sub>k<sup>2</sup><sub>2</sub>+kg<sub>2</sub>k<sup>2</sup><sub>2</sub>) cos<sup>2</sup> φ<sub>12</sub>+2kg<sub>1</sub>k<sub>2</sub>kg<sub>2</sub>k<sub>2</sub> cos φ<sub>12</sub> <sup>1</sup> kg<sub>1</sub>+g<sub>2</sub>k<sup>2</sup>\`− 1−cos<sup>2</sup> 2 <sup>φ12</sup> (kg<sub>1</sub>−g<sub>2</sub>k<sup>2</sup>)L

or

• cos φ<sub>12</sub> ≤ −Φ(g<sub>1</sub>, g<sub>2</sub>)  
• \` ≥ ξ(g<sub>1</sub>, g<sub>2</sub>)L  
• t ≥ (kg<sub>1</sub>k<sup>2</sup>+kg<sub>2</sub>k<sup>2</sup>) cos<sup>2</sup> φ<sub>12</sub>+2kg<sub>1</sub>k<sub>2</sub>kg<sub>2</sub>k<sub>2</sub> cos φ<sub>12</sub> <sup>1</sup> kg<sub>1</sub>+g<sub>2</sub>k<sup>2</sup>\`− cos<sup>2</sup> 2 <sup>φ12</sup> (kg<sub>1</sub>−g<sub>2</sub>k<sup>2</sup><sub>2</sub>)L

Proof. To show the necessary conditions, from Equation $^ { 4 , }$ all we need is

$$
\begin{array}{l} t \cdot [ (\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L) t \\ \left. - \left(\left(\| \mathbf {g} _ {\mathbf {1}} \| _ {2} ^ {2} + \| \mathbf {g} _ {\mathbf {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g} _ {\mathbf {1}} \| _ {2} \| \mathbf {g} _ {\mathbf {2}} \| _ {2} \cos \phi_ {1 2}\right) \right] \geq 0 \tag {5} \\ \end{array}
$$

Since $t \geq 0$ , it reduces to show

$$
\begin{array}{l} (\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L) t \\ - \left(\left(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}\right) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}\right) \geq 0 \tag {6} \\ \end{array}
$$

For Equation 6 to hold while ensuring that $t \geq 0$ , there are two cases:

• <sup>1</sup> kg<sub>1</sub> + g<sub>2</sub>k<sup>2</sup><sub>2</sub>\` − (1 − cos<sup>2</sup> φ<sub>12</sub>)(kg<sub>1</sub>k<sup>2</sup><sub>2</sub> + kg<sub>2</sub>k<sup>2</sup><sub>2</sub>)L ≥ 0,  
• <sup>1</sup> kg<sub>1</sub> + g<sub>2</sub>k<sup>2</sup><sub>2</sub>\` − (1 − cos<sup>2</sup> φ<sub>12</sub>)(kg<sub>1</sub>k<sup>2</sup><sub>2</sub> + kg<sub>2</sub>k<sup>2</sup><sub>2</sub>)L ≤ 0,

$$
(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} \geq 0,
$$

$$
t \geq \frac {(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}}{\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L}
$$

$$
(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2} \leq 0,
$$

$$
t \geq \frac {(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}}{\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L}
$$

, which can be simplified to

$$
\begin{array}{l} \ell \geq \frac {(1 - \cos^ {2} \phi_ {1 2}) (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2})}{\| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2}} L = \xi (\mathbf {g _ {1}}, \mathbf {g _ {2}}), \\ t \geq \frac {(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}}{\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L} \\ \ell \leq \frac {(1 - \cos^ {2} \phi_ {1 2}) (\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2})}{\| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2}} L = \xi (\mathbf {g _ {1}}, \mathbf {g _ {2}}), \\ 0 <   t \leq \frac {(\| \mathbf {g _ {1}} \| _ {2} ^ {2} + \| \mathbf {g _ {2}} \| _ {2} ^ {2}) \cos^ {2} \phi_ {1 2} + 2 \| \mathbf {g _ {1}} \| _ {2} \| \mathbf {g _ {2}} \| _ {2} \cos \phi_ {1 2}}{\frac {1}{2} \| \mathbf {g _ {1}} + \mathbf {g _ {2}} \| _ {2} ^ {2} \ell - \frac {1 - \cos^ {2} \phi_ {1 2}}{2} (\| \mathbf {g _ {1}} - \mathbf {g _ {2}} \| _ {2} ^ {2}) L}. \\ \end{array}
$$

• cos φ<sub>12</sub> ≤ <sup>2kg1k2kg2k2</sup><sub>kg k2+kg k2</sub> = −Φ(g<sub>1</sub>, g<sub>2</sub>),

<sup>2kg1k2kg2k2</sup><sub>kg k2+kg k2</sub> = −Φ(g<sub>1</sub>, g<sub>2</sub>) ≤ cos φ<sub>12</sub> < 0,

The sufficient conditions hold as we can plug the conditions to RHS of Equation 6 and achieve non-negative result. □

## A.4 Convergence of PCGrad with Momentum-Based Gradient Descent

In this subsection, we show convergence of PCGrad with momentum-based methods, which is more aligned with our practical implementation. In our analysis, we consider the heavy ball method [43] as follows:

$$
\theta_ {k + 1} \leftarrow \theta_ {k} - \alpha_ {k} \nabla \mathcal {L} (\theta_ {k}) + \beta_ {k} (\theta_ {k} - \theta_ {k - 1})
$$

where k denotes the k-th step and $\alpha _ { k }$ and $\beta _ { k }$ are step sizes for the gradient and momentum at step k respectively. We now present our theorem.

Theorem 4. Assume $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ are µ - and µ -strongly convex and also $\boldsymbol { L } _ { 1 } \boldsymbol { \cdot }$ - and $L _ { 2 } .$ -smooth respectively where $\mu _ { 1 } , \mu _ { 2 } , L _ { 1 } , L _ { 2 } > 0 \nonumber$ . Define $\phi _ { 1 2 } ^ { k }$ as the angle between two task gradients g<sub>1</sub> $\left( \theta _ { k } \right)$ and $\mathbf { g _ { 2 } } ( \theta _ { k } )$ and define $\begin{array} { r } { R _ { k } = \frac { \| \mathbf { g } _ { 1 } ( \theta _ { k } ) \| } { \| \mathbf { g } _ { 2 } ( \theta _ { k } ) \| } } \end{array}$ . Denote $\mu _ { k } = ( 1 - \cos \phi _ { 1 2 } ^ { k } / R _ { k } ) \mu _ { 1 } + ( 1 - \cos \phi _ { 1 2 } ^ { k } \cdot R _ { k } ) \mu _ { 2 }$ and ${ \cal L } _ { k } = ( 1 - \cos \phi _ { 1 2 } ^ { k } / R _ { k } ) L _ { 1 } + ( 1 - \cos \phi _ { 1 2 } ^ { k } \cdot R _ { k } ) L _ { 2 }$ Then, the PCGrad update rule of the heavy ball method with step sizes $\begin{array} { r } { \alpha _ { k } = \frac { 4 } { \sqrt { L _ { k } } + \sqrt { \mu _ { k } } } } \end{array}$ and $\beta _ { k } = \operatorname* { m a x } \{ | 1 - \sqrt { \alpha _ { k } \mu _ { k } } | , | 1 - \sqrt { \alpha _ { k } L _ { k } } | \} ^ { 2 }$ will converge linearly to either (1) a location in the optimization landscape where $\cos ( \phi _ { 1 2 } ^ { k } ) = - 1 o r ( 2 )$ the optimal value $\mathcal { L } ( \theta ^ { \ast } )$

Proof. We first observe that the PCGrad-modified gradient $\mathbf { g } ^ { \mathrm { P C } }$ has the following identity:

$$
\begin{array}{l} \mathbf {g} ^ {\mathrm{PC}} = \mathbf {g} - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\left\| \mathbf {g _ {1}} \right\| ^ {2}} \mathbf {g _ {1}} - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\left\| \mathbf {g _ {2}} \right\| ^ {2}} \mathbf {g _ {2}} \\ = (1 - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\| \mathbf {g _ {1}} \| ^ {2}}) \mathbf {g _ {1}} + (1 - \frac {\mathbf {g _ {1}} \cdot \mathbf {g _ {2}}}{\| \mathbf {g _ {2}} \| ^ {2}}) \mathbf {g _ {2}} \\ = (1 - \frac {\mathbf {g} _ {1} \cdot \mathbf {g} _ {2}}{\| \mathbf {g} _ {1} \| \| \mathbf {g} _ {2} \|} \frac {\| \mathbf {g} _ {2} \|}{\| \mathbf {g} _ {1} \|}) \mathbf {g} _ {1} + (1 - \frac {\mathbf {g} _ {1} \cdot \mathbf {g} _ {2}}{\| \mathbf {g} _ {1} \| \| \mathbf {g} _ {2} \|} \frac {\| \mathbf {g} _ {1} \|}{\| \mathbf {g} _ {2} \|}) \mathbf {g} _ {2} \\ = (1 - \cos \phi_ {1 2} / R) \mathbf {g _ {1}} + (1 - \cos \phi_ {1 2} \cdot R) \mathbf {g _ {2}}. \tag {7} \\ \end{array}
$$

Applying Equation 7, we can write the PCGrad update rule of the heavy ball method in matrix form as follows:

$$
\begin{array}{l} \left\| \left[ \begin{array}{c} \theta_ {k + 1} - \theta^ {*} \\ \theta_ {k} - \theta^ {*} \end{array} \right] \right\| _ {2} = \left\| \left[ \begin{array}{c} \theta_ {k} + \beta_ {k} (\theta_ {k} - \theta_ {k - 1}) - \theta^ {*} \\ \theta_ {k} - \theta^ {*} \end{array} \right] - \alpha_ {k} \left[ \begin{array}{c} \mathbf {g} ^ {\mathrm{PC}} (\theta_ {k}) \\ 0 \end{array} \right] \right\| _ {2} \\ = \left\| \left[ \begin{array}{c} \theta_ {k} + \beta_ {k} (\theta_ {k} - \theta_ {k - 1}) - \theta^ {*} \\ \theta_ {k} - \theta^ {*} \end{array} \right] \right. \\ \left. - \alpha_ {k} \left[ \begin{array}{c} (1 - \cos \phi_ {1 2} ^ {k} / R _ {k}) \mathbf {g _ {1}} (\theta_ {k}) + (1 - \cos \phi_ {1 2} ^ {k} \cdot R _ {k}) \mathbf {g _ {2}} (\theta_ {k}) \\ 0 \end{array} \right] \right\| _ {2} \\ = \left\| \left[ \begin{array}{c} \theta_ {k} + \beta_ {k} (\theta_ {k} - \theta_ {k - 1}) - \theta^ {*} \\ \theta_ {k} - \theta^ {*} \end{array} \right] \right. \\ \left. - \alpha_ {k} \left[ \begin{array}{c} \left[ (1 - \cos \phi_ {1 2} ^ {k} / R _ {k}) \nabla^ {2} \mathcal {L} _ {1} (z _ {k}) + (1 - \cos \phi_ {1 2} ^ {k} \cdot R _ {k}) \nabla^ {2} \mathcal {L} _ {2} (z _ {k} ^ {\prime}) \right] (\theta_ {k} - \theta^ {*}) \\ 0 \end{array} \right] \right\| _ {2} \\ \end{array}
$$

for some z<sub>k</sub>, z<sup>0</sup> on the line segment between θ<sub>k</sub> and θ<sup>∗</sup>

$$
\begin{array}{l} = \left\| \left[ \begin{array}{c c} (1 + \beta_ {k}) I - \alpha_ {k} H _ {k} & - \beta_ {k} I \\ I & 0 \end{array} \right] \left[ \begin{array}{c} \theta_ {k} - \theta^ {*} \\ \theta_ {k - 1} - \theta^ {*} \end{array} \right] \right\| _ {2} \\ \leq \left\| \left[ \begin{array}{c c} (1 + \beta_ {k}) I - \alpha_ {k} H _ {k} & - \beta_ {k} I \\ I & 0 \end{array} \right] \right\| _ {2} \left\| \left[ \begin{array}{c} \theta_ {k} - \theta^ {*} \\ \theta_ {k - 1} - \theta^ {*} \end{array} \right] \right\| _ {2} \\ \end{array}
$$

where $H _ { k } = ( 1 - \cos \phi _ { 1 2 } ^ { k } / R _ { k } ) \nabla ^ { 2 } \mathcal { L } _ { 1 } ( z _ { k } ) + ( 1 - \cos \phi _ { 1 2 } ^ { k } \cdot R _ { k } ) \nabla ^ { 2 } \mathcal { L } _ { 2 } ( z _ { k } ^ { \prime } ) .$

By strong convexity and smoothness of $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ , we have the eigenvalues of $\nabla ^ { 2 } \mathcal { L } _ { 1 } ( z _ { k } )$ are between $\mu _ { 1 }$ and $L _ { 1 }$ . Similarly, the eigenvalues of $\nabla ^ { 2 } \mathcal { L } _ { 2 } ( z _ { k } ^ { \prime } )$ are between $\mu _ { 2 }$ and $L _ { 2 }$ . Thus the eigenvalues of $H _ { k }$ are between $\mu _ { k } = ( 1 - \cos \phi _ { 1 2 } ^ { k } / R _ { k } ) \mu _ { 1 } + ( 1 - \cos \phi _ { 1 2 } ^ { k } \cdot R _ { k } ) \mu _ { 2 }$ and $L _ { k } \ =$ $( 1 - \cos \phi _ { 1 2 } ^ { k } / R _ { k } ) L _ { 1 } + ( 1 - \cos \phi _ { 1 2 } ^ { k } \cdot R _ { k } ) L _ { 2 } \ [ 2 0 ]$ . Hence following Lemma 3.1 in [54], we have

$$
\left\| \left[ \begin{array}{c c} (1 + \beta_ {k}) I - \alpha_ {k} H _ {k} & - \beta_ {k} I \\ I & 0 \end{array} \right] \right\| _ {2} \leq \max \{| 1 - \sqrt {\alpha_ {k} \mu_ {k}} |, | 1 - \sqrt {\alpha_ {k} L _ {k}} | \}.
$$

Thus we have

$$
\begin{array}{l} \left\| \left[ \begin{array}{c} \theta_ {k + 1} - \theta^ {*} \\ \theta_ {k} - \theta^ {*} \end{array} \right] \right\| _ {2} \leq \max \{| 1 - \sqrt {\alpha_ {k} \mu_ {k}} |, | 1 - \sqrt {\alpha_ {k} L _ {k}} | \} \left\| \left[ \begin{array}{c} \theta_ {k} - \theta^ {*} \\ \theta_ {k - 1} - \theta^ {*} \end{array} \right] \right\| _ {2} \\ = \frac {\sqrt {\kappa_ {k}} - 1}{\sqrt {\kappa_ {k}} + 1} \left\| \left[ \begin{array}{l} \theta_ {k} - \theta^ {*} \\ \theta_ {k - 1} - \theta^ {*} \end{array} \right] \right\| _ {2} \tag {8} \\ \leq \left\| \left[ \begin{array}{c} \theta_ {k} - \theta^ {*} \\ \theta_ {k - 1} - \theta^ {*} \end{array} \right] \right\| _ {2} \\ \end{array}
$$

where $\begin{array} { r } { \kappa _ { k } = \frac { L _ { k } } { \mu _ { k } } } \end{array}$ and Equation 8 follows from substitution $\begin{array} { r } { \alpha _ { k } = \frac { 4 } { \sqrt { L _ { k } } + \sqrt { \mu _ { k } } } } \end{array}$ . Hence PCGrad with heavy ball method converges linearly if cos $\phi _ { 1 2 } ^ { k } \neq - 1$ □

## B Empirical Objective-Wise Evaluations of PCGrad

In this section, we visualize the per-task training loss and validation loss curves respectively on NYUv2. The goal of measuring objective-wise performance is to study the convergence of PCGrad in practice, particularly amidst the possibility of slow convergence due to cosine similarities near -1, as discussed in Section 2.4.

We show the objective-wise evaluation results on NYUv2 in Figure 5. For evaluations on NYUv2, PCGrad + MTAN attains similar training convergence rate compared to MTAN in three tasks in NYUv2 while converging faster and achieving lower validation loss in 2 out of 3 tasks. Note that in task 0 of the NYUv2 dataset, both methods seem to overfit, suggesting a better regularization scheme for this domain.

In general, these results suggest that PCGrad has a regularization effect on supervised multi-task learning, rather than an improvement on optimization speed or convergence. We hypothesize that this regularization is caused by PCGrad leading to greater sharing of representations across tasks, such that the supervision for one task better regularizes the training of another. This regularization effect seems notably different from the effect of PCGrad on reinforcement learning problems, where PCGrad dramatically improves training performance. This suggests that multi-task supervised learning and multi-task reinforcement learning problems may have distinct challenges.

![](images/065fd3f78a530318ae690ec4315ab6d7f973aa17413e4d8d0dd0350e023cbcf4.jpg)  
Figure 5: Empirical objective-wise evaluations on NYUv2. On the top row, we show the objective-wise training learning curves and on the bottom row, we show the objective-wise validation learning curves. PCGrad+MTAN converges with a similar rate compared to MTAN in training and for validation losses, PCGrad+MTAN converges faster and obtains a lower final validation loss in two out of three tasks. This result corroborate that in practice, PCGrad does not exhibit the potential slow convergence problem shown in Theorem 1.

## C Practical Details of PCGrad on Multi-Task and Goal-Conditioned Reinforcement Learning

In our experiments, we apply PCGrad to the soft actor-critic (SAC) algorithm [24], an off-policy RL method. In SAC, we employ a Q-learning style gradient to compute the gradient of the Q-function network, $Q _ { \phi } ( s , a , z _ { i } )$ , often known as the critic, and a reparameterization-style gradient to compute the gradient of the policy network $\pi _ { \theta } ( a | s , z _ { i } )$ , often known as the actor. For sampling, we instantiate a set of replay buffers $\{ \dot { \mathcal { D } } _ { i } \} _ { \mathcal { T } _ { i } \sim p ( \mathcal { T } ) }$ . Training and data collection are alternated throughout training. During a data collection step, we run the policy $\pi _ { \theta }$ on all the tasks $\mathcal { T } _ { i } \sim p ( \mathcal { T } )$ to collect an equal number of paths for each task and store the paths of each task $\mathcal { T } _ { i }$ into the corresponding replay buffer $\mathcal { D } _ { i }$ . At each training step, we sample an equal amount of data from each replay buffer $\mathcal { D } _ { i }$ to form a stratified batch. For each task $\mathcal { T } _ { i } \sim p ( \mathcal { T } )$ ), the parameters of the critic θ are optimized to minimize the soft Bellman residual:

$$
J _ {Q} ^ {(i)} (\phi) = \mathbb {E} _ {(s _ {t}, a _ {t}, z _ {i}) \sim \mathcal {D} _ {i}} \left[ Q _ {\phi} (s _ {t}, a _ {t}, z _ {i}) - (r (s _ {t}, a _ {t}, z _ {i}) + \gamma V _ {\bar {\phi}} (s _ {t + 1}, z _ {i})) \right], \tag {9}
$$

$$
V _ {\bar {\phi}} (s _ {t + 1}, z _ {i}) = \mathbb {E} _ {a _ {t + 1} \sim \pi_ {\theta}} \left[ Q _ {\bar {\phi}} (s _ {t + 1}, a _ {t + 1}, z _ {i}) - \alpha \log \pi_ {\theta} (a _ {t + 1} | s _ {t + 1}, z _ {i}) \right], \tag {10}
$$

where $\gamma$ is the discount factor, $\bar { \phi }$ are the delayed parameters, and α is a learnable temperature that automatically adjusts the weight of the entropy term. For each task $\mathcal { T } _ { i } \sim p ( \mathcal { T } )$ , the parameters of the policy $\pi _ { \theta }$ are trained to minimize the following objective

$$
J _ {\pi} ^ {(i)} (\theta) = \mathbb {E} _ {s _ {t} \sim \mathcal {D} _ {i}} \left[ \mathbb {E} _ {a _ {t} \sim \pi_ {\theta} (a _ {t} | s _ {t}, z _ {i}))} \left[ \alpha \log \pi_ {\theta} (a _ {t} | s _ {t}, z _ {i}) - Q _ {\phi} (s _ {t}, a _ {t}, z _ {i}) \right] \right]. \tag {11}
$$

We compute $\nabla _ { \phi } J _ { Q } ^ { ( i ) } ( \phi )$ and $\nabla _ { \theta } J _ { \pi } ^ { ( i ) } ( \theta )$ for all $\begin{array} { r } { T _ { i } \sim p ( \mathcal T ) } \end{array}$ and apply PCGrad to both following Algorithm 1.

In the context of $\mathbf { S A C }$ specifically, we also propose to learn the temperature α for adjusting entropy of the policy on a per-task basis. This allows the method to control the entropy of the multi-task policy per-task. The motivation is that if we use a single learnable temperature for adjusting entropy of the multi-task policy $\pi _ { \theta } ( a | s , z _ { i } )$ , SAC may stop exploring once all easier tasks are solved, leading to poor performance on tasks that are harder or require more exploration. To address this issue, we propose to learn the temperature on a per-task basis as mentioned in Section $^ { 3 , }$ i.e. using a parametrized model to represent $\alpha _ { \psi } ( z _ { i } )$ . This allows the method to control the entropy of $\pi _ { \theta } ( a | s , z _ { i } )$ per-task. We optimize the parameters of $\alpha _ { \psi } ( z _ { i } )$ using the same constrained optimization framework as in [24].

When applying PCGrad to goal-conditioned RL, we represent $p ( \mathcal { T } )$ as a distribution of goals and let z be the encoding of a goal. Similar to the multi-task supervised learning setting discussed in Section $3 ,$ PCGrad may be combined with various architectures designed for multi-task and goal-conditioned RL [19, 14], where PCGrad operates on the gradients of shared parameters, leaving task-specific parameters untouched.

## D 2D Optimization Landscape Details

To produce the 2D optimization visualizations in Figure 1, we used a parameter vector $\theta = \left[ \theta _ { 1 } , \theta _ { 2 } \right] \in$ $\mathbb { R } ^ { 2 }$ and the following task loss functions:

$$
\mathcal {L} _ {1} (\theta) = 2 0 \log (\max (|. 5 \theta_ {1} + \tanh (\theta_ {2}) |, 0. 0 0 0 0 0 5))
$$

$$
\mathcal {L} _ {2} (\theta) = 2 5 \log (\max (|. 5 \theta_ {1} - \tanh (\theta_ {2}) + 2 |, 0. 0 0 0 0 0 5))
$$

The multi-task objective is $\mathcal { L } ( \theta ) = \mathcal { L } _ { 1 } ( \theta ) + \mathcal { L } _ { 2 } ( \theta )$ . We initialized $\theta = [ 0 . 5 , - 3 ]$ and performed 500,000 gradient updates to minimize $\mathcal { L }$ using the Adam optimizer with learning rate 0.001. We compared using Adam for each update to using Adam in conjunction with the PCGrad method presented in Section 2.3.

## E Additional Multi-Task Supervised Learning Results

We present our multi-task supervised learning results on MultiMNIST and CityScapes here.

MultiMNIST. Following the same set-up of Sener and Koltun [53], for each image, we sample a different one uniformly at random. Then we put one of the image on the top left and the other one on the bottom right. The two tasks in the multi-task learning problem are to classify the digits on the top left (task-L) and bottom right (task-R) respectively. We construct such 60K examples. We combine PCGrad with the same backbone architecture used in [53] and compare its performance to Sener and Koltun [53] by running the open-sourced code provided in [53]. As shown in Table 4, PCGrad results 0.13% and 0.55% improvement over [53] in left and right digit accuracy respectively.

<table><tr><td></td><td>left digit</td><td>right digit</td></tr><tr><td>Sener and Koltun [53]</td><td>96.45</td><td>94.95</td></tr><tr><td>PCGrad (ours)</td><td>96.58</td><td>95.50</td></tr></table>

Table 4: MultiMNIST results. PCGrad achieves improvements over the approach by Sener and Koltun [53] in both left and right digit classfication accuracy.

CityScapes. The CityScapes dataset [12] contains 19 classes of street-view images resized to 128×256. There are two tasks in this dataset: semantic segmentation and depth estimation. Following the setup in Liu et al. [33], we pair the depth estimation task with semantic segmentation using the coarser 7 categories instead of the finer 19 classes in the original CityScapes dataset. Similar to NYUv2 evaluations described in Section 5, we also combine PCGrad with MTAN [33] and compare it to a range of methods discussed in Appendix J.1. For the combination of PCGrad and MTAN, we only use equal weighting as discussed in [33] as we find it working well in practice. We present the results in Table 5. As shown in Table 5, PCGrad + MTAN outperforms MTAN in three out of four scores while obtaining the top scores in both mIoU abd pixel accuracy for the semantic segmentation task, suggesting the effectiveness of PCGrad on realistic image datasets. We also provide the full results including three different weighting schemes in Table 7 in Appendix J.4.

<table><tr><td rowspan="2">#P.</td><td rowspan="2">Architecture</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td></tr><tr><td>(Higher mIoU)</td><td>Better) Pix Acc</td><td>(Lower Abs Err)</td><td>Better) Rel Err</td></tr><tr><td>2</td><td>One Task</td><td>51.09</td><td>90.69</td><td>0.0158</td><td>34.17</td></tr><tr><td>3.04</td><td>STAN</td><td>51.90</td><td>90.87</td><td>0.0145</td><td>27.46</td></tr><tr><td>1.75</td><td>Split, Wide</td><td>50.17</td><td>90.63</td><td>0.0167</td><td>44.73</td></tr><tr><td>2</td><td>Split, Deep</td><td>49.85</td><td>88.69</td><td>0.0180</td><td>43.86</td></tr><tr><td>3.63</td><td>Dense</td><td>51.91</td><td>90.89</td><td>0.0138</td><td>27.21</td></tr><tr><td>≈2</td><td>Cross-Stitch [39]</td><td>50.08</td><td>90.33</td><td>0.0154</td><td>34.49</td></tr><tr><td>1.65</td><td>MTAN</td><td>53.04</td><td>91.11</td><td>0.0144</td><td>33.63</td></tr><tr><td>1.65</td><td>PCGrad+MTAN (Ours)</td><td>53.59</td><td>91.45</td><td>0.0171</td><td>31.34</td></tr></table>

Table 5: We present the 7-class semantic segmentation and depth estimation results on CityScapes dataset. We use #P to denote the number of parameters of the network. We use box and bold text to highlight the method that achieves the best validation score for each task. As seen in the results, PCGrad+MTAN with equal weights outperforms MTAN with equal weights in three out of four scores while achieving the top score both scores in the segmentation task.

## F Goal-Conditioned Reinforcement Learning Results

For our goal-conditioned RL evaluation, we adopt the goal-conditioned robotic pushing task with a Sawyer robot where the goals are represented as the concatenations of the initial positions of the puck to be pushed and the its goal location, both of which are uniformly sampled (details in Appendix J.3). We also apply the temperature adjustment strategy as discussed in Section 3 to predict the temperature for entropy term given the goal. We summarize the results in Figure 6. PCGrad with SAC achieves better performance in terms of average distance to the goal position, while the vanilla SAC agent is struggling to successfully accomplish the task. This suggests that PCGrad is able to ease the RL optimization problem also when the task distribution is continuous.

## G Comparison to CosReg

We compare PCGrad to a prior method CosReg [55], which adds a regularization term to force the cosine similarity between gradients of two different tasks to stay 0. PCGrad achieves much better average success rate in MT10 benchmark as shown in Figure 7. Hence, while it’s important to reduce interference between tasks, it’s also crucial to keep the task gradients that enjoy positive cosine similarities in order to ensure sharing across tasks.

![](images/de01944b34cacad0533d2f3f058a2fd2d6a638c4969255ff4c4b22fb68cd3a36.jpg)

<details>
<summary>line</summary>

| Number of thousand env steps | SAC (Distance to the goal) | SAC+PCGrad (ours) (Distance to the goal) |
| --- | --- | --- |
| 0 | ~0.315 | ~0.35 |
| 100 | ~0.312 | ~0.342 |
| 200 | ~0.318 | ~0.355 |
| 300 | ~0.312 | ~0.38 |
| 400 | ~0.322 | ~0.365 |
| 500 | ~0.328 | ~0.422 |
| 600 | ~0.34 | ~0.415 |
| 700 | ~0.315 | ~0.43 |
| 800 | ~0.312 | ~0.365 |
| 900 | ~0.31 | ~0.335 |
| 1000 | ~0.315 | ~0.328 |
| 1100 | ~0.312 | ~0.315 |
| 1200 | ~0.31 | ~0.305 |
| 1300 | ~0.312 | ~0.285 |
| 1400 | ~0.315 | ~0.275 |
| 1500 | ~0.312 | ~0.228 |
| 1600 | ~0.315 | ~0.242 |
| 1700 | ~0.312 | ~0.242 |
| 1800 | ~0.31 | ~0.205 |
| 1900 | ~0.312 | ~0.212 |
| 2000 | ~0.31 | ~0.222 |
| 2100 | ~0.312 | ~0.20 |
| 2200 | ~0.31 | ~0.198 |
| 2300 | ~0.312 | ~0.198 |
| 2400 | ~0.31 | ~0.185 |
| 2500 | ~0.312 | ~0.195 |
| 2600 | ~0.315 | ~0.192 |
| 2700 | ~0.318 | ~0.18 |
| 2800 | ~0.315 | ~0.168 |
| 2900 | ~0.318 | ~0.168 |
| 3000 | ~0.315 | ~0.178 |
| 3100 | ~0.318 | ~0.175 |
| 3200 | ~0.315 | ~0.155 |
| 3300 | ~0.318 | ~0.165 |
| 3400 | ~0.315 | ~0.172 |
| 3500 | ~0.318 | ~0.155 |
| 3600 | ~0.315 | ~0.165 |
| 3700 | ~0.318 | ~0.175 |
| 3800 | ~0.315 | ~0.16 |
| 3900 | ~0.318 | ~0.16 |
| 4000 | ~0.315 | ~0.158 |
| 4100 | ~0.318 | ~0.155 |
| 4200 | ~0.315 | ~0.152 |
| 4300 | ~0.318 | ~0.155 |
| 4400 | ~0.315 | ~0.158 |
| 4500 | ~0.318 | ~0.155 |
| 4600 | ~0.315 | ~0.14 |
| 4700 | ~0.318 | ~0.15 |
| 4800 | ~0.315 | ~0.152 |
| 4900 | ~0.318 | ~0.155 |
| 5000 | ~0.315 | ~0.16 |
| 5100 | ~0.318 | ~0.148 |
| 5200 | ~0.315 | ~0.168 |
| 5300 | ~0.318 | ~0.148 |
| 5400 | ~0.315 | ~0.148 |
| 5500 | ~0.318 | ~0.148 |
| 5600 | ~0.315 | ~0.148 |
| 5700 | ~0.318 | ~0.148 |
| 5800 | ~0.315 | ~0.148 |
| 59<fcel>~<fcel>~<nl>

</details>

Figure 6: We present the goal-conditioned RL results. PCGrad outperforms vanilla SAC in terms of both average distance the goal and data efficiency.

![](images/e88517572e4226332d3b8c0f8a5a4cead9585c8e271e3b68288e231824e357cf.jpg)

<details>
<summary>line</summary>

| Number of thousand env steps | SAC+Cosine Reg (Success Rate) | SAC+PCGrad (ours) (Success Rate) |
| --- | --- | --- |
| 0 | ~0.05 | ~0.05 |
| 1000 | ~0.20 | ~0.60 |
| 2000 | ~0.20 | ~0.75 |
| 3000 | ~0.25 | ~0.80 |
| 4000 | ~0.25 | ~0.85 |
| 5000 | ~0.30 | ~0.90 |
</details>

Figure 7: Comparison between PCGrad and CosReg [55]. PCGrad outperforms CosReg, suggesting that we should both reduce the interference and keep shared structure across tasks.

## H Ablation study on the task order

As stated on line 4 in Algorithm 1, we sample the tasks from the batch and randomly shuffle the order of the tasks before performing the update steps in PCGrad. With random shuffling, we make PCGrad symmetric w.r.t. the task order in expectation. In Figure 8, we observe that PCGrad with a random task order achieves better performance between PCGrad with a fixed task order in the setting of MT50 where the number of tasks is large and the conflicting gradient phenomenon is much more likely to happen.

## I Combining PCGrad with other architectures

In this subsection, we test whether PCGrad can improve performances when combined with more methods. In Table 6, we find that PCGrad does improve the performance in all four metrics of the three tasks on the NYUv2 dataset when combined with Cross-Stitch [39] and Dense. In Figure 9, we also show that PCGrad + Multi-head SAC outperforms Multi-head SAC on its own. These results suggest that PCGrad can be flexibly combined with any multi-task learning architectures to further improve performance.

![](images/2a938b6769c194b7203a553d6cf761cdb61d2d2aaaf2c82552ea3f8ec370250b.jpg)

<details>
<summary>line</summary>

| Number of thousand env steps | SAC+PCGrad (ours) | SAC+PCGrad fixed (ours) |
| --- | --- | --- |
| 0 | ~0.03 | ~0.03 |
| 500 | ~0.11 | ~0.16 |
| 1000 | ~0.18 | ~0.24 |
| 1500 | ~0.24 | ~0.30 |
| 2000 | ~0.29 | ~0.33 |
| 2500 | ~0.32 | ~0.35 |
| 3000 | ~0.39 | ~0.37 |
| 3500 | ~0.44 | ~0.37 |
| 4000 | ~0.47 | ~0.37 |
| 4500 | ~0.51 | ~0.39 |
| 5000 | ~0.52 | ~0.41 |
| 5500 | ~0.49 | ~0.38 |
| 6000 | ~0.52 | ~0.41 |
| 6500 | ~0.56 | ~0.42 |
| 7000 | ~0.59 | ~0.44 |
| 7500 | ~0.59 | ~0.43 |
| 8000 | ~0.58 | ~0.45 |
| 8500 | ~0.61 | ~0.41 |
| 9000 | ~0.64 | ~0.43 |
| 9500 | ~0.62 | ~0.45 |
| 10000 | ~0.64 | ~0.42 |
| 10500 | ~0.66 | ~0.41 |
| 11000 | ~0.68 | ~0.42 |
</details>

Figure 8: Ablation study on using a fixed task order during PCGrad. PCGrad with a random task order does significantly better PCGrad with a fixed task order in MT50 benchmark.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Segmentation mIoU</td><td rowspan="2">Depth Abs Err</td><td colspan="2">Surface Normal</td></tr><tr><td>Angle Distance</td><td>Within 11.25°</td></tr><tr><td>Cross-Stitch</td><td>15.69</td><td>0.6277</td><td>32.69</td><td>21.63</td></tr><tr><td>Cross-Stitch + PCGrad</td><td>18.14</td><td>0.5805</td><td>31.38</td><td>21.75</td></tr><tr><td>Dense</td><td>16.48</td><td>0.6282</td><td>31.68</td><td>21.73</td></tr><tr><td>Dense + PCGrad</td><td>18.08</td><td>0.5850</td><td>30.17</td><td>23.29</td></tr></table>

Table 6: We show the performances of PCGrad combined with other methods on three-task learning on the NYUv2 dataset, where PCGrad further improves the results of prior multi-task learning architectures.

## J Experiment Details

## J.1 Multi-Task Supervised Learning Experiment Details

For all the multi-task supervised learning experiments, PCGrad converges within 12 hours on a NVIDIA TITAN RTX GPU while the vanilla models without PCGrad converge within 8 hours. PCGrad consumes at most 10 GB memory on GPU while the vanilla method consumes 6GB on GPU among all experiments.

For our CIFAR-100 multi-task experiment, we adopt the architecture used in [47], which is a convolutional neural network that consists of 3 convolutional layers with 160 3 × 3 filters each layer and 2 fully connected layers with 320 hidden units. As for experiments on the NYUv2 dataset, we follow [33] to use SegNet [1] as the backbone architecture.

We use five algorithms as baselines in the CIFAR-100 multi-task experiment: task specific-1-fc [46]: a convolutional neural network shared across tasks except that each task has a separate last fully-connected layer, task specific-1-fc [46] : all the convolutional layers shared across tasks with separate fully-connected layers for each task, cross stitch-all-fc [40]: one convolutional neural network per task along with cross-stitch units to share features across tasks, routing-all-fc + WPL [47]: a network that employs a trainable router trained with multi-agent RL algorithm (WPL) to select trainable functions for each task, independent: training separate neural networks for each task.

For comparisons on the NYUv2 dataset, we consider 5 baselines: Single Task, One Task: the vanilla SegNet used for single-task training, Single Task, STAN [33]: the single-task version of MTAN as mentioned below, Multi-Task, Split, Wide / Deep [33]: the standard SegNet shared for all three tasks except that each task has a separate last layer for final task-specific prediction with two variants Wide and Deep specified in [33], Multi-Task Dense: a shared network followed by separate task-specific networks, Multi-Task Cross-Stitch [40]: similar to the baseline used in CIFAR-100 experiment but with SegNet as the backbone, MTAN [33]: a shared network with a soft-attention module for each task.

## J.2 Multi-Task Reinforcement Learning Experiment Details

Our reinforcement learning experiments all use the SAC [24] algorithm as the base algorithm, where the actor and the critic are represented as 6-layer fully-connected feedforward neural networks for all methods. The numbers of hidden units of each layer of the neural networks are 160, 300 and 200 for MT10, MT50 and goal-conditioned RL respectively. For the multi-task RL experiments, PCGrad + SAC converges in 1 day (5M simulation steps) and 5 days (20M simulation steps) on the MT10 and MT50 benchmarks respectively on a NVIDIA TITAN RTX GPU while vanilla SAC converges in 12 hours and 3 days on the two benchmarks respectively. PCGrad + SAC consumes 1 GB and 6 GB memory on GPU on the MT10 and MT50 benchmarks respectively while the vanilla SAC consumes 0.5 GB and 3 GB respectively.

![](images/69d3a8842c76fc8d93a85461cfbdc047885e682b236902e36ae6391522c068be.jpg)

<details>
<summary>line</summary>

| Number of thousand env steps | Multi-head SAC (Success Rate) | Multi-head SAC+PCGrad (Success Rate) |
| --- | --- | --- |
| 0 | ~0.12 | ~0.02 |
| 500 | ~0.22 | ~0.23 |
| 1000 | ~0.30 | ~0.38 |
| 1500 | ~0.32 | ~0.52 |
| 2000 | ~0.31 | ~0.65 |
| 2500 | ~0.32 | ~0.70 |
| 3000 | ~0.34 | ~0.75 |
| 3500 | ~0.36 | ~0.80 |
| 4000 | ~0.38 | ~0.82 |
| 4500 | ~0.42 | ~0.84 |
| 5000 | ~0.40 | ~0.84 |
</details>

Figure 9: We show the comparison between Multi-head SAC and Multi-head SAC + PCGrad on MT10. Multi-head SAC + PCGrad outperforms Multi-head SAC, suggesting that PCGrad can improves the performance of multi-headed architectures in the multi-task RL settings.

![](images/42190f082f90f4e59a90be4e8db2d76d757a718236d4458452614bc892184ec9.jpg)  
Figure 10: The 50 tasks of MT50 from Meta-World [61]. MT10 is a subset of these tasks, which includes reach, push, pick & place, open drawer, close drawer, open door, press button top, open window, close window, and insert peg inside.

In the case of multi-task reinforcement learning, we evaluate our algorithm on the recently proposed Meta-World benchmark [61]. This benchmark includes a variety of simulated robotic manipulation tasks contained in a shared, table-top environment with a simulated Sawyer arm (visualized in Fig. 10). In particular, we use the multi-task benchmarks MT10 and MT50, which consists of the 10 tasks and 50 tasks respectively depicted in Fig. 10 that require diverse strategies to solve them, which makes them difficult to optimize jointly with a single policy. Note that MT10 is a subset of MT50. At each data collection step, we collect 600 samples for each task, and at each training step, we sample 128 datapoints per task from corresponding replay buffers. We measure success according to the metrics used in the Meta-World benchmark where the reported the success rates are averaged across tasks. For all methods, we apply the temperature adjustment strategy as discussed in Section 3 to learn a separate alpha term per task as the task encoding in MT10 and MT50 is just a one-hot encoding.

On the multi-task and goal-conditioned RL domain, we apply PCGrad to the vanilla SAC algorithm with task encoding as part of the input to the actor and the critic as described in Section 3 and compare PCGrad to the vanilla SAC without PCGrad and training actors and critics for each task individually (Independent).

## J.3 Goal-conditioned Experiment Details

We use the pushing environment from the Meta-World benchmark [61] as shown in Figure 10. In this environment, the table spans from [−0.4, 0.2] to [0.4, 1.0] in the 2D space. To construct the goals, we sample the intial positions of the puck from the range [−0.2, 0.6] to [0.2, 0.7] on the table and the goal positions from the range [−0.2, 0.85] to [0.2, 0.95] on the table. The goal is represented as a concatenation of the initial puck position and the goal position. Since in the goal-conditioned setting, the task distribution is continuous, we sample a minibatch of 9 goals and 128 samples per goal at each training iteration and also sample 600 samples per goal in the minibatch at each data collection step.

## J.4 Full CityScapes and NYUv2 Results

We provide the full comparison on the CityScapes and NYUv2 datasets in Table 7 and Table 8 respectively.

<table><tr><td rowspan="2">#P.</td><td rowspan="2">Architecture</td><td rowspan="2">Weighting</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td></tr><tr><td>(Higher mIoU)</td><td>Better) Pix Acc</td><td>(Lower Abs Err)</td><td>Better) Rel Err</td></tr><tr><td>2</td><td>One Task</td><td>n.a.</td><td>51.09</td><td>90.69</td><td>0.0158</td><td>34.17</td></tr><tr><td>3.04</td><td>STAN</td><td>n.a.</td><td>51.90</td><td>90.87</td><td>0.0145</td><td>27.46</td></tr><tr><td rowspan="3">1.75</td><td rowspan="3">Split, Wide</td><td>Equal Weights</td><td>50.17</td><td>90.63</td><td>0.0167</td><td>44.73</td></tr><tr><td>Uncert. Weights [28]</td><td>51.21</td><td>90.72</td><td>0.0158</td><td>44.01</td></tr><tr><td>DWA,  $T = 2$ </td><td>50.39</td><td>90.45</td><td>0.0164</td><td>43.93</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Split, Deep</td><td>Equal Weights</td><td>49.85</td><td>88.69</td><td>0.0180</td><td>43.86</td></tr><tr><td>Uncert. Weights [28]</td><td>48.12</td><td>88.68</td><td>0.0169</td><td>39.73</td></tr><tr><td>DWA,  $T = 2$ </td><td>49.67</td><td>88.81</td><td>0.0182</td><td>46.63</td></tr><tr><td rowspan="3">3.63</td><td rowspan="3">Dense</td><td>Equal Weights</td><td>51.91</td><td>90.89</td><td>0.0138</td><td>27.21</td></tr><tr><td>Uncert. Weights [28]</td><td>51.89</td><td>91.22</td><td>0.0134</td><td>25.36</td></tr><tr><td>DWA,  $T = 2$ </td><td>51.78</td><td>90.88</td><td>0.0137</td><td>26.67</td></tr><tr><td rowspan="3"> $\approx 2$ </td><td rowspan="3">Cross-Stitch [39]</td><td>Equal Weights</td><td>50.08</td><td>90.33</td><td>0.0154</td><td>34.49</td></tr><tr><td>Uncert. Weights [28]</td><td>50.31</td><td>90.43</td><td>0.0152</td><td>31.36</td></tr><tr><td>DWA,  $T = 2$ </td><td>50.33</td><td>90.55</td><td>0.0153</td><td>33.37</td></tr><tr><td rowspan="3">1.65</td><td rowspan="3">MTAN</td><td>Equal Weights</td><td>53.04</td><td>91.11</td><td>0.0144</td><td>33.63</td></tr><tr><td>Uncert. Weights [28]</td><td>53.86</td><td>91.10</td><td>0.0144</td><td>35.72</td></tr><tr><td>DWA,  $T = 2$ </td><td>53.29</td><td>91.09</td><td>0.0144</td><td>34.14</td></tr><tr><td>1.65</td><td>PCGrad+MTAN (Ours)</td><td>Equal Weights</td><td>53.59</td><td>91.45</td><td>0.0171</td><td>31.34</td></tr></table>

Table 7: We present the 7-class semantic segmentation and depth estimation results on CityScapes dataset. We use #P to denote the number of parameters of the network, and the best performing variant of each architecture is highlighted in bold. We use box to highlight the method that achieves the best validation score for each task. As seen in the results, PCGrad+MTAN with equal weights outperforms MTAN with equal weights in three out of four scores while achieving the top score in pixel accuracy across all methods.

<table><tr><td rowspan="3">Type</td><td rowspan="3">#P.</td><td rowspan="3">Architecture</td><td rowspan="3">Weighting</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td></tr><tr><td colspan="2">(Higher Better)</td><td colspan="2">(Lower Better)</td><td colspan="2">Angle Distance (Lower Better)</td><td colspan="3">Within  $t^{\circ}$ (Higher Better)</td></tr><tr><td>mIoU</td><td>Pix Acc</td><td>Abs Err</td><td>Rel Err</td><td>Mean</td><td>Median</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td rowspan="2">Single Task</td><td>3</td><td>One Task</td><td>n.a.</td><td>15.10</td><td>51.54</td><td>0.7508</td><td>0.3266</td><td>31.76</td><td>25.51</td><td>22.12</td><td>45.33</td><td>57.13</td></tr><tr><td>4.56</td><td> $STAN^†$ </td><td>n.a.</td><td>15.73</td><td>52.89</td><td>0.6935</td><td>0.2891</td><td>32.09</td><td>26.32</td><td>21.49</td><td>44.38</td><td>56.51</td></tr><tr><td rowspan="16">Multi Task</td><td rowspan="3">1.75</td><td rowspan="3">Split, Wide</td><td>Equal Weights</td><td>15.89</td><td>51.19</td><td>0.6494</td><td>0.2804</td><td>33.69</td><td>28.91</td><td>18.54</td><td>39.91</td><td>52.02</td></tr><tr><td>Uncert. Weights*</td><td>15.86</td><td>51.12</td><td>0.6040</td><td>0.2570</td><td>32.33</td><td>26.62</td><td>21.68</td><td>43.59</td><td>55.36</td></tr><tr><td> $DWA^†, T = 2$ </td><td>16.92</td><td>53.72</td><td>0.6125</td><td>0.2546</td><td>32.34</td><td>27.10</td><td>20.69</td><td>42.73</td><td>54.74</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Split, Deep</td><td>Equal Weights</td><td>13.03</td><td>41.47</td><td>0.7836</td><td>0.3326</td><td>38.28</td><td>36.55</td><td>9.50</td><td>27.11</td><td>39.63</td></tr><tr><td>Uncert. Weights*</td><td>14.53</td><td>43.69</td><td>0.7705</td><td>0.3340</td><td>35.14</td><td>32.13</td><td>14.69</td><td>34.52</td><td>46.94</td></tr><tr><td> $DWA^†, T = 2$ </td><td>13.63</td><td>44.41</td><td>0.7581</td><td>0.3227</td><td>36.41</td><td>34.12</td><td>12.82</td><td>31.12</td><td>43.48</td></tr><tr><td rowspan="3">4.95</td><td rowspan="3">Dense</td><td>Equal Weights</td><td>16.06</td><td>52.73</td><td>0.6488</td><td>0.2871</td><td>33.58</td><td>28.01</td><td>20.07</td><td>41.50</td><td>53.35</td></tr><tr><td>Uncert. Weights*</td><td>16.48</td><td>54.40</td><td>0.6282</td><td>0.2761</td><td>31.68</td><td>25.68</td><td>21.73</td><td>44.58</td><td>56.65</td></tr><tr><td> $DWA^†, T = 2$ </td><td>16.15</td><td>54.35</td><td>0.6059</td><td>0.2593</td><td>32.44</td><td>27.40</td><td>20.53</td><td>42.76</td><td>54.27</td></tr><tr><td rowspan="3">≈3</td><td rowspan="3">Cross- $Stitch^‡$ </td><td>Equal Weights</td><td>14.71</td><td>50.23</td><td>0.6481</td><td>0.2871</td><td>33.56</td><td>28.58</td><td>20.08</td><td>40.54</td><td>51.97</td></tr><tr><td>Uncert. Weights*</td><td>15.69</td><td>52.60</td><td>0.6277</td><td>0.2702</td><td>32.69</td><td>27.26</td><td>21.63</td><td>42.84</td><td>54.45</td></tr><tr><td> $DWA^†, T = 2$ </td><td>16.11</td><td>53.19</td><td>0.5922</td><td>0.2611</td><td>32.34</td><td>26.91</td><td>21.81</td><td>43.14</td><td>54.92</td></tr><tr><td rowspan="3">1.77</td><td rowspan="3"> $MTAN^†$ </td><td>Equal Weights</td><td>17.72</td><td>55.32</td><td>0.5906</td><td>0.2577</td><td>31.44</td><td>25.37</td><td>23.17</td><td>45.65</td><td>57.48</td></tr><tr><td>Uncert. Weights*</td><td>17.67</td><td>55.61</td><td>0.5927</td><td>0.2592</td><td>31.25</td><td>25.57</td><td>22.99</td><td>45.83</td><td>57.67</td></tr><tr><td> $DWA^†, T = 2$ </td><td>17.15</td><td>54.97</td><td>0.5956</td><td>0.2569</td><td>31.60</td><td>25.46</td><td>22.48</td><td>44.86</td><td>57.24</td></tr><tr><td>1.77</td><td> $MTAN^† + PCGrad (ours)$ </td><td>Uncert. Weights*</td><td>20.17</td><td>56.65</td><td>0.5904</td><td>0.2467</td><td>30.01</td><td>24.83</td><td>22.28</td><td>46.12</td><td>58.77</td></tr></table>

Table 8: We present the full results on three tasks on the NYUv2 dataset: 13-class semantic segmentation, depth estimation, and surface normal prediction results. #P shows the total number of network parameters. We highlight the best performing combination of multi-task architecture and weighting in bold. The top validation scores for each task are annotated with boxes. The symbols indicate prior methods: <sup>∗</sup>: [28], <sup>†</sup>: [33], <sup>‡</sup>: [40]. Performance of other methods taken from [33].